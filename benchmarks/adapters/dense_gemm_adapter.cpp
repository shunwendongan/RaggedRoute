#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

class DenseGemmAdapter final : public BenchmarkAdapter {
 public:
  explicit DenseGemmAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  ~DenseGemmAdapter() override {
#if RAGGEDROUTE_HAS_CUBLAS
    library_baseline::destroy_dense_cublaslt_plan(cublaslt_plan_);
    library_baseline::destroy_dense_cublas_plan(cublas_plan_);
#endif
  }
  std::string operator_name() const override { return "dense_gemm"; }
  std::string variant_name() const override { return variant_name_; }
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
#if RAGGEDROUTE_HAS_CUBLAS
    if (variant_name_ == "cublaslt") {
      cublaslt_plan_ = library_baseline::create_dense_cublaslt_plan(
          m_, n_, k_, 64ULL * 1024ULL * 1024ULL, &library_workspace_bytes_);
      library_workspace_.resize(library_workspace_bytes_);
    } else if (variant_name_ == "cublas") {
      cublas_plan_ = library_baseline::create_dense_cublas_plan();
    }
#endif
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
#if RAGGEDROUTE_HAS_CUBLAS
    if (variant_name_ == "cublaslt") {
      library_baseline::launch_dense_cublaslt(cublaslt_plan_, a_.data(), b_.data(), c_.data(),
                                              library_workspace_.data(), library_workspace_bytes_,
                                              stream);
      return;
    }
    if (variant_name_ == "cublas") {
      library_baseline::launch_dense_cublas(cublas_plan_, a_.data(), b_.data(), c_.data(), m_, n_,
                                            k_, stream);
      return;
    }
#endif
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(ops::launch_dense_gemm_naive(a_.data(), b_.data(), c_.data(), m_, n_, k_, stream),
                 "launch_dense_gemm_naive");
      return;
    }
    DenseGemmArgs args;
    args.a.data = a_.data();
    args.b.data = b_.data();
    args.c.data = c_.data();
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
    if (variant_name_ == "cublaslt") {
      return {{"api", std::string("cublasLtMatmul")},
              {"heuristic_rank", static_cast<std::int64_t>(0)},
              {"workspace_limit_bytes", static_cast<std::int64_t>(64ULL * 1024ULL * 1024ULL)},
              {"compute_type", std::string("CUBLAS_COMPUTE_32F_PEDANTIC")}};
    }
    if (variant_name_ == "cublas") {
      return {{"api", std::string("cublasSgemm")},
              {"math_mode", std::string("strict_fp32")},
              {"layout_adapter", std::string("row_major_as_transposed_column_major")}};
    }
    return {{"threads_per_block", static_cast<std::int64_t>(256)},
            {"outputs_per_thread", static_cast<std::int64_t>(1)},
            {"math_path", std::string("cuda_core_scalar")}};
  }
  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    work.flops = 2.0 * m_ * n_ * k_;
    work.logical_bytes =
        sizeof(float) * (static_cast<double>(m_) * k_ + static_cast<double>(k_) * n_ +
                         static_cast<double>(m_) * n_);
    work.operator_metrics["output_elements"] = static_cast<std::int64_t>(m_) * n_;
    return work;
  }
  std::size_t workspace_bytes() const override { return library_workspace_bytes_; }
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
  std::string variant_name_;
#if RAGGEDROUTE_HAS_CUBLAS
  library_baseline::DenseCublasLtPlan* cublaslt_plan_ = nullptr;
  library_baseline::DenseCublasPlan* cublas_plan_ = nullptr;
#endif
  std::size_t library_workspace_bytes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  std::vector<float> a_host_, b_host_, expected_;
  DeviceBuffer<float> a_, b_, c_;
  DeviceBuffer<std::uint8_t> library_workspace_;
};

}  // namespace

AdapterPtr make_dense_gemm_adapter(const std::string& variant_name) {
  return std::make_unique<DenseGemmAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
