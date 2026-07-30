#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

class DenseGemmAdapter final : public BenchmarkAdapter {
 public:
  std::string operator_name() const override { return "dense_gemm"; }
  std::string variant_name() const override { return "cuda_naive"; }
  std::string description() const override { return "One CUDA thread per FP32 output element"; }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    m_ = get_int_option(options, "M", 128);
    n_ = get_int_option(options, "N", 128);
    k_ = get_int_option(options, "K", 128);
    a_host_ = make_random_floats(static_cast<std::size_t>(m_) * k_, seed);
    b_host_ = make_random_floats(static_cast<std::size_t>(k_) * n_, seed + 1);
    expected_.assign(static_cast<std::size_t>(m_) * n_, 0.0F);
    for (int row = 0; row < m_; ++row) {
      for (int column = 0; column < n_; ++column) {
        float accumulator = 0.0F;
        for (int inner = 0; inner < k_; ++inner) {
          accumulator += a_host_[static_cast<std::size_t>(row) * k_ + inner] *
                         b_host_[static_cast<std::size_t>(inner) * n_ + column];
        }
        expected_[static_cast<std::size_t>(row) * n_ + column] = accumulator;
      }
    }
    a_.resize(a_host_.size());
    b_.resize(b_host_.size());
    c_.resize(expected_.size());
    a_.copy_from_host(a_host_, stream);
    b_.copy_from_host(b_host_, stream);
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(ops::launch_dense_gemm_naive(a_.data(), b_.data(), c_.data(), m_, n_, k_, stream),
                 "launch_dense_gemm_naive");
      return;
    }
    DenseGemmArgs args;
    args.a = a_.data();
    args.b = b_.data();
    args.c = c_.data();
    args.m = m_;
    args.n = n_;
    args.k = k_;
    operator_check(dense_gemm(args, make_runtime_context(stream, architecture_)),
                   "dense_gemm operator");
  }
  ValidationResult validate(cudaStream_t stream) override {
    return compare_floats(c_.copy_to_host(stream), expected_, 1.0e-5, 2.0e-5 * k_);
  }
  FieldMap case_config() const override {
    return {{"M", static_cast<std::int64_t>(m_)},
            {"N", static_cast<std::int64_t>(n_)},
            {"K", static_cast<std::int64_t>(k_)},
            {"dtype", std::string("fp32")},
            {"accumulator_dtype", std::string("fp32")},
            {"layout_a", std::string("row_major")},
            {"layout_b", std::string("row_major")},
            {"layout_c", std::string("row_major")},
            {"alpha", 1.0},
            {"beta", 0.0}};
  }
  FieldMap variant_config() const override {
    return {{"threads_per_block", static_cast<std::int64_t>(256)},
            {"outputs_per_thread", static_cast<std::int64_t>(1)},
            {"math_path", std::string("cuda_core_scalar")}};
  }
  WorkEstimate work_estimate() const override {
    WorkEstimate work;
    work.flops = 2.0 * m_ * n_ * k_;
    work.logical_bytes =
        sizeof(float) * (static_cast<double>(m_) * k_ + static_cast<double>(k_) * n_ +
                         static_cast<double>(m_) * n_);
    work.operator_metrics["output_elements"] = static_cast<std::int64_t>(m_) * n_;
    return work;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"input_generation", "cpu_reference", "h2d_copy", "workspace_allocation"};
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"M", "N", "K"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown dense_gemm param: " + name);
    }
  }
  int m_ = 0, n_ = 0, k_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  std::vector<float> a_host_, b_host_, expected_;
  DeviceBuffer<float> a_, b_, c_;
};

}  // namespace

AdapterPtr make_dense_gemm_adapter() { return std::make_unique<DenseGemmAdapter>(); }

}  // namespace raggedroute::benchmark
