// This target intentionally has no runtime test. It only proves that the CUDA
// toolkit can compile the declared low-precision storage types for the target
// architecture selected by CMake.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void raggedroute_dtype_compile_probe(unsigned int* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = static_cast<unsigned int>(
        sizeof(__half) + sizeof(__nv_bfloat16) + sizeof(__nv_fp8_e4m3) + sizeof(__nv_fp8_e5m2) +
        sizeof(__nv_fp6_e2m3) + sizeof(__nv_fp6_e3m2) + sizeof(__nv_fp4_e2m1));
  }
}
