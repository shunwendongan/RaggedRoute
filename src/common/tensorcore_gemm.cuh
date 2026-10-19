#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorcore_gemm.h"

namespace raggedroute::ops::tensorcore_detail {

constexpr int kThreads = 128;

inline bool supported_type(ScalarType type) {
  return type == ScalarType::kFp16 || type == ScalarType::kBf16;
}

inline bool aligned(const void* pointer, std::uintptr_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

inline unsigned int ceil_div(int value, int divisor) {
  return static_cast<unsigned int>((static_cast<std::uint64_t>(value) + divisor - 1) /
                                   divisor);
}

template <bool BFloat16>
__device__ __forceinline__ float to_float(std::uint16_t bits) {
  if constexpr (BFloat16) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
  } else {
    return __half2float(__ushort_as_half(bits));
  }
}

__device__ __forceinline__ void copy_async_16(std::uint16_t* destination,
                                              const std::uint16_t* source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto address = static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(address), "l"(source) : "memory");
#else
  *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
#endif
}

__device__ __forceinline__ void commit_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

template <int Pending>
__device__ __forceinline__ void wait_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(Pending) : "memory");
#endif
}

// PTX ISA mma.m16n8k16 floating-point fragment mapping:
// A x4 matrices are TL, BL, TR, BR; B x2.trans are the two K halves.
// Every lane participates and all 8-element row addresses are 16-byte aligned.
__device__ __forceinline__ void load_a(std::uint32_t (&fragment)[4],
                                        const std::uint16_t* pointer) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto address = static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(fragment[0]), "=r"(fragment[1]),
                 "=r"(fragment[2]), "=r"(fragment[3])
               : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void load_b(std::uint32_t (&fragment)[2],
                                        const std::uint16_t* pointer) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto address = static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(fragment[0]), "=r"(fragment[1]) : "r"(address) : "memory");
#endif
}

template <bool BFloat16>
__device__ __forceinline__ void mma(float (&accumulator)[4],
                                     const std::uint32_t (&a)[4],
                                     const std::uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (BFloat16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(accumulator[0]), "+f"(accumulator[1]),
                   "+f"(accumulator[2]), "+f"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(accumulator[0]), "+f"(accumulator[1]),
                   "+f"(accumulator[2]), "+f"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
  }
#endif
}

// Both staging modes have the same vectorization and masked scalar tail. The
// asynchronous variant changes the copy mechanism and uses two owned buffers.
template <int BM, int BN, int BK, bool Asynchronous>
__device__ __forceinline__ void stage(const std::uint16_t* a,
                                       const std::uint16_t* b,
                                       int rows, int n, int k,
                                       int row_base, int column_base, int k_base,
                                       std::uint16_t* a_shared,
                                       std::uint16_t* b_shared) {
  constexpr int AS = BK + 8;
  constexpr int BS = BN + 8;
  const int thread = static_cast<int>(threadIdx.x);
  for (int vector = thread; vector < BM * BK / 8; vector += kThreads) {
    const int row = vector / (BK / 8);
    const int column = (vector % (BK / 8)) * 8;
    auto* destination = a_shared + row * AS + column;
    const bool valid_row = row_base + row < rows;
    const bool full = valid_row && k_base + column + 7 < k;
    if (full) {
      const auto* source = a + static_cast<std::size_t>(row_base + row) * k + k_base + column;
      if ((reinterpret_cast<std::uintptr_t>(source) & 15U) == 0) {
        if constexpr (Asynchronous) copy_async_16(destination, source);
        else *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
        continue;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      destination[j] = valid_row && k_base + column + j < k
          ? a[static_cast<std::size_t>(row_base + row) * k + k_base + column + j]
          : 0;
    }
  }
  for (int vector = thread; vector < BK * BN / 8; vector += kThreads) {
    const int row = vector / (BN / 8);
    const int column = (vector % (BN / 8)) * 8;
    auto* destination = b_shared + row * BS + column;
    const bool valid_row = k_base + row < k;
    const bool full = valid_row && column_base + column + 7 < n;
    if (full) {
      const auto* source = b + static_cast<std::size_t>(k_base + row) * n + column_base + column;
      if ((reinterpret_cast<std::uintptr_t>(source) & 15U) == 0) {
        if constexpr (Asynchronous) copy_async_16(destination, source);
        else *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
        continue;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      destination[j] = valid_row && column_base + column + j < n
          ? b[static_cast<std::size_t>(k_base + row) * n + column_base + column + j]
          : 0;
    }
  }
  if constexpr (Asynchronous) commit_async();
}

template <int BM, int BN, int BK, bool BFloat16>
__device__ __forceinline__ void simt_tile(const std::uint16_t* a,
                                           const std::uint16_t* b, float* c,
                                           int rows, int n, int k,
                                           int row_base, int column_base) {
  for (int index = static_cast<int>(threadIdx.x); index < BM * BN; index += kThreads) {
    const int row = row_base + index / BN;
    const int column = column_base + index % BN;
    if (row < rows && column < n) {
      float value = 0.0F;
      for (int inner = 0; inner < k; ++inner) {
        value = __fmaf_rn(to_float<BFloat16>(a[static_cast<std::size_t>(row) * k + inner]),
                          to_float<BFloat16>(b[static_cast<std::size_t>(inner) * n + column]),
                          value);
      }
      c[static_cast<std::size_t>(row) * n + column] = value;
    }
  }
}

template <int BM, int BN, int BK, bool Asynchronous, bool BFloat16>
__device__ __forceinline__ void compute(const std::uint16_t* a,
                                         const std::uint16_t* b, float* c,
                                         int rows, int n, int k,
                                         int row_base, int column_base,
                                         std::uint16_t* a_stages,
                                         std::uint16_t* b_stages) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int WarpRows = BM == 16 ? 1 : 2;
  constexpr int WarpColumns = 4 / WarpRows;
  constexpr int WarpM = BM / WarpRows;
  constexpr int WarpN = BN / WarpColumns;
  constexpr int AM = WarpM / 16;
  constexpr int AN = WarpN / 8;
  constexpr int AS = BK + 8;
  constexpr int BS = BN + 8;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int warp_m = (warp / WarpColumns) * WarpM;
  const int warp_n = (warp % WarpColumns) * WarpN;
  float accumulator[AM][AN][4] = {};
  const int tiles = static_cast<int>((static_cast<std::int64_t>(k) + BK - 1) / BK);

  if constexpr (Asynchronous) {
    if (tiles > 0) stage<BM, BN, BK, true>(a, b, rows, n, k, row_base, column_base,
                                          0, a_stages, b_stages);
  }
  for (int tile = 0; tile < tiles; ++tile) {
    const int current = Asynchronous ? (tile & 1) : 0;
    if constexpr (Asynchronous) {
      if (tile + 1 < tiles) {
        const int next = current ^ 1;
        stage<BM, BN, BK, true>(a, b, rows, n, k, row_base, column_base,
                                (tile + 1) * BK, a_stages + next * BM * AS,
                                b_stages + next * BK * BS);
        wait_async<1>();
      } else {
        wait_async<0>();
      }
    } else {
      stage<BM, BN, BK, false>(a, b, rows, n, k, row_base, column_base,
                               tile * BK, a_stages, b_stages);
    }
    __syncthreads();
    const auto* a_tile = a_stages + current * BM * AS;
    const auto* b_tile = b_stages + current * BK * BS;
#pragma unroll
    for (int inner = 0; inner < BK; inner += 16) {
      std::uint32_t a_fragment[AM][4];
      std::uint32_t b_fragment[AN][2];
#pragma unroll
      for (int mi = 0; mi < AM; ++mi) {
        load_a(a_fragment[mi], a_tile + (warp_m + mi * 16 + lane % 16) * AS +
                                 inner + (lane / 16) * 8);
      }
#pragma unroll
      for (int ni = 0; ni < AN; ++ni) {
        load_b(b_fragment[ni], b_tile + (inner + lane % 16) * BS + warp_n + ni * 8);
      }
#pragma unroll
      for (int mi = 0; mi < AM; ++mi) {
#pragma unroll
        for (int ni = 0; ni < AN; ++ni) {
          mma<BFloat16>(accumulator[mi][ni], a_fragment[mi], b_fragment[ni]);
        }
      }
    }
    // No producer may overwrite a shared buffer until all its consumers finish.
    if (tile + 1 < tiles) __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < AM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < AN; ++ni) {
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        const int row = row_base + warp_m + mi * 16 + lane / 4 + (element / 2) * 8;
        const int column = column_base + warp_n + ni * 8 + (lane % 4) * 2 + element % 2;
        if (row < rows && column < n) c[static_cast<std::size_t>(row) * n + column] =
                                        accumulator[mi][ni][element];
      }
    }
  }
#else
  simt_tile<BM, BN, BK, BFloat16>(a, b, c, rows, n, k, row_base, column_base);
#endif
}

inline cudaError_t validate_dense(const void* a, const void* b, const float* c,
                                  int m, int n, int k, ScalarType type) {
  if (!supported_type(type) || m < 0 || n < 0 || k < 0) return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || !aligned(c, alignof(float)) ||
      (k > 0 && (a == nullptr || b == nullptr || !aligned(a, 2) || !aligned(b, 2))))
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

inline cudaError_t validate_grouped(const void* a, const void* b,
                                    const std::int32_t* offsets, const float* c,
                                    int experts, int k, int n, int max_rows,
                                    ScalarType type) {
  if (!supported_type(type) || experts < 0 || k < 0 || n < 0 || max_rows < 0)
    return cudaErrorInvalidValue;
  if (experts == 0 || n == 0 || max_rows == 0) return cudaSuccess;
  if (experts > 65535 || offsets == nullptr || !aligned(offsets, alignof(std::int32_t)))
    return cudaErrorInvalidValue;
  return validate_dense(a, b, c, max_rows, n, k, type);
}

}  // namespace raggedroute::ops::tensorcore_detail
