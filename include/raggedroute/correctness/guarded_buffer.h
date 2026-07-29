#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {

inline void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

inline std::uint64_t fnv1a_bytes(const void* data, std::size_t bytes) {
  const auto* input = static_cast<const unsigned char*>(data);
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t index = 0; index < bytes; ++index) {
    hash ^= input[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <typename T>
std::uint64_t hash_vector(const std::vector<T>& values) {
  return fnv1a_bytes(values.data(), values.size() * sizeof(T));
}

template <typename T>
class GuardedDeviceBuffer {
 public:
  static constexpr std::size_t kRedzoneBytes = 256;
  static constexpr unsigned char kCanary = 0xa5;
  static constexpr unsigned char kPoison = 0xcd;

  GuardedDeviceBuffer() = default;
  GuardedDeviceBuffer(std::size_t count, cudaStream_t stream) { resize(count, stream); }
  GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
  GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
  ~GuardedDeviceBuffer() { reset(); }

  void resize(std::size_t count, cudaStream_t stream) {
    reset();
    count_ = count;
    payload_bytes_ = checked_mul(count, sizeof(T), "guarded payload");
    if (payload_bytes_ == 0) return;
    allocation_bytes_ = checked_add(checked_add(kRedzoneBytes, payload_bytes_, "front redzone"),
                                    kRedzoneBytes, "back redzone");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&allocation_), allocation_bytes_),
               "cudaMalloc guarded buffer");
    payload_ = reinterpret_cast<T*>(allocation_ + kRedzoneBytes);
    cuda_check(cudaMemsetAsync(allocation_, kCanary, allocation_bytes_, stream),
               "initialize canaries");
    cuda_check(cudaMemsetAsync(payload_, kPoison, payload_bytes_, stream), "initialize poison");
  }

  void reset() noexcept {
    if (allocation_ != nullptr) cudaFree(allocation_);
    allocation_ = nullptr;
    payload_ = nullptr;
    count_ = payload_bytes_ = allocation_bytes_ = 0;
  }

  void copy_from_host(const std::vector<T>& host, cudaStream_t stream) {
    if (host.size() != count_) throw std::invalid_argument("guarded buffer size mismatch");
    if (payload_bytes_ != 0) {
      cuda_check(
          cudaMemcpyAsync(payload_, host.data(), payload_bytes_, cudaMemcpyHostToDevice, stream),
          "guarded H2D");
    }
  }

  std::vector<T> copy_to_host(cudaStream_t stream) const {
    std::vector<T> host(count_);
    if (payload_bytes_ != 0) {
      cuda_check(
          cudaMemcpyAsync(host.data(), payload_, payload_bytes_, cudaMemcpyDeviceToHost, stream),
          "guarded D2H");
      cuda_check(cudaStreamSynchronize(stream), "guarded D2H sync");
    }
    return host;
  }

  std::uint64_t payload_hash(cudaStream_t stream) const {
    return hash_vector(copy_to_host(stream));
  }

  bool canaries_intact(cudaStream_t stream) const {
    if (allocation_ == nullptr) return true;
    std::vector<unsigned char> front(kRedzoneBytes), back(kRedzoneBytes);
    cuda_check(
        cudaMemcpyAsync(front.data(), allocation_, kRedzoneBytes, cudaMemcpyDeviceToHost, stream),
        "front canary D2H");
    cuda_check(cudaMemcpyAsync(back.data(), allocation_ + kRedzoneBytes + payload_bytes_,
                               kRedzoneBytes, cudaMemcpyDeviceToHost, stream),
               "back canary D2H");
    cuda_check(cudaStreamSynchronize(stream), "canary sync");
    const auto valid = [](const std::vector<unsigned char>& bytes) {
      return std::all_of(bytes.begin(), bytes.end(),
                         [](unsigned char value) { return value == kCanary; });
    };
    return valid(front) && valid(back);
  }

  T* data() { return payload_; }
  const T* data() const { return payload_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return payload_bytes_; }

 private:
  unsigned char* allocation_ = nullptr;
  T* payload_ = nullptr;
  std::size_t count_ = 0;
  std::size_t payload_bytes_ = 0;
  std::size_t allocation_bytes_ = 0;
};

struct LaunchObservation {
  cudaError_t launch_error = cudaSuccess;
  cudaError_t execution_error = cudaSuccess;
};

template <typename Launch>
LaunchObservation observe_launch(Launch&& launch, cudaStream_t stream) {
  cudaGetLastError();
  LaunchObservation observation;
  observation.launch_error = std::forward<Launch>(launch)();
  if (observation.launch_error == cudaSuccess) observation.launch_error = cudaPeekAtLastError();
  if (observation.launch_error == cudaSuccess) {
    observation.execution_error = cudaStreamSynchronize(stream);
  }
  return observation;
}

}  // namespace raggedroute::correctness
