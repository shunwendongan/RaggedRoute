#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "raggedroute/benchmark/types.h"

namespace raggedroute::benchmark {

// The runner owns timing and reporting. Each adapter owns operator semantics,
// buffers, reference data, reset rules, and operator-specific metadata.
class BenchmarkAdapter {
 public:
  virtual ~BenchmarkAdapter() = default;

  virtual std::string operator_name() const = 0;
  virtual std::string variant_name() const = 0;
  virtual std::string description() const = 0;
  virtual bool supports(MeasurementLevel level) const = 0;
  virtual RepeatPolicy repeat_policy(MeasurementLevel level) const = 0;

  virtual void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) = 0;

  // Runs outside the timed interval. At L1 this is where an adapter restores
  // explicit preconditions such as counts_zeroed/cursor_zeroed.
  virtual void prepare_sample(MeasurementLevel level, cudaStream_t stream) = 0;

  // Enqueues exactly one measured invocation. L2 must include every reset or
  // dynamic device-side metadata operation required by the public operator.
  virtual void enqueue(MeasurementLevel level, cudaStream_t stream) = 0;

  // Validates the output of one clean invocation. It must never execute inside
  // the timed interval.
  virtual ValidationResult validate(cudaStream_t stream) = 0;

  virtual FieldMap case_config() const = 0;
  virtual FieldMap variant_config() const = 0;
  virtual WorkEstimate work_estimate(MeasurementLevel level) const = 0;
  virtual std::size_t workspace_bytes() const { return 0; }
  virtual std::vector<std::string> excluded_steps(MeasurementLevel level) const = 0;
};

using AdapterPtr = std::unique_ptr<BenchmarkAdapter>;

}  // namespace raggedroute::benchmark
