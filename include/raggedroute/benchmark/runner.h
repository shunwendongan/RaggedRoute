#pragma once

#include <cuda_runtime_api.h>

#include <string>

#include "raggedroute/benchmark/adapter.h"
#include "raggedroute/benchmark/types.h"

namespace raggedroute::benchmark {

EnvironmentInfo collect_environment();
BenchmarkRecord run_benchmark(BenchmarkAdapter& adapter, const RunOptions& options,
                              cudaStream_t stream);
MeasurementSummary summarize_samples(const std::vector<double>& samples_us);
std::string record_to_json(const BenchmarkRecord& record);
void append_jsonl(const std::string& path, const BenchmarkRecord& record);

}  // namespace raggedroute::benchmark
