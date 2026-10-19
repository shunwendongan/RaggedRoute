#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/tensorcore_gemm.h"
#include "dense_gemm/cuda_candidate/optimized_internal.h"
#include "grouped_gemm/cuda_candidate/grouped_optimized_internal.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/tensorcore_adapter.h"
#include "raggedroute/benchmark/tensorcore_baselines.h"

#if RAGGEDROUTE_HAS_CUTLASS
#include <cutlass/version.h>
#endif

namespace raggedroute::benchmark {
namespace {
namespace rc = correctness;
namespace lib = tensorcore_baseline;

bool candidate_variant(const std::string& variant) {
  return variant == "cuda_mma_sync" || variant == "cuda_mma_async" ||
         variant == "cuda_mma_sync_convert" || variant == "cuda_mma_async_convert";
}

std::string dependency_revision(const std::string& variant) {
  if (variant == "cublas" || variant == "cublaslt") {
    return std::string("CUDA Toolkit ") + RAGGEDROUTE_CUDATOOLKIT_VERSION;
  }
  if (variant == "cutlass_grouped") {
#if RAGGEDROUTE_HAS_CUTLASS
    return "CUTLASS_VERSION=" + std::to_string(CUTLASS_VERSION) +
        "; configured=" + RAGGEDROUTE_CUTLASS_CONFIG_REVISION +
        "; CUDA Toolkit " + RAGGEDROUTE_CUDATOOLKIT_VERSION;
#else
    return "unavailable";
#endif
  }
  return "not_applicable";
}

rc::NumericSummary summarize_error(const std::vector<double>& actual,
                                 const std::vector<double>& expected) {
  rc::NumericSummary result;
  long double squared_error = 0.0L, squared_reference = 0.0L;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double absolute = std::abs(actual[i] - expected[i]);
    result.max_abs_error = std::max(result.max_abs_error, absolute);
    result.max_rel_error = std::max(result.max_rel_error,
        absolute / std::max(std::abs(expected[i]), 1.0e-30));
    squared_error += static_cast<long double>(absolute) * absolute;
    squared_reference += static_cast<long double>(expected[i]) * expected[i];
  }
  result.normalized_l2_error = static_cast<double>(squared_reference == 0.0L
      ? std::sqrt(squared_error) : std::sqrt(squared_error / squared_reference));
  return result;
}

class TensorCoreGemmAdapter final : public BenchmarkAdapter {
 public:
  TensorCoreGemmAdapter(std::string name, std::string variant)
      : name_(std::move(name)), variant_(std::move(variant)),
        grouped_(name_ == "grouped_gemm_tc") {}
  ~TensorCoreGemmAdapter() override {
    if (pinned_offsets_ != nullptr) cudaFreeHost(pinned_offsets_);
#if RAGGEDROUTE_HAS_CUBLAS
    lib::destroy_gemm_plan(gemm_plan_);
#endif
#if RAGGEDROUTE_HAS_CUTLASS
    lib::destroy_grouped_cutlass_plan(cutlass_plan_);
#endif
  }
  std::string operator_name() const override { return name_; }
  std::string variant_name() const override { return variant_; }
  std::string description() const override {
    return "Explicit FP16/BF16 inputs, FP32 accumulation/output GEMM research benchmark";
  }
  bool supports(MeasurementLevel level) const override {
    // This includes conversions, library host launches and device metadata when required.
    return level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    const std::string dtype = get_option(options, "dtype", "fp16");
    if (dtype != "fp16" && dtype != "bf16") {
      throw std::invalid_argument("Tensor Core benchmark dtype must be fp16 or bf16");
    }
    input_type_ = dtype == "fp16" ? ScalarType::kFp16 : ScalarType::kBf16;
    input_dtype_ = dtype;
    fp32_parent_ = variant_ == "fp32_parent";
    conversion_ = variant_.find("_convert") != std::string::npos;
    config_.block_m = get_int_option(options, "tile_m", grouped_ ? 32 : 64);
    config_.block_n = get_int_option(options, "tile_n", grouped_ ? 64 : 32);
    config_.block_k = get_int_option(options, "tile_k", 32);
    config_.asynchronous = variant_.find("async") != std::string::npos;
    if ((config_.block_k != 32 && config_.block_k != 64) ||
        (grouped_ ? ((config_.block_m != 16 && config_.block_m != 32) || config_.block_n != 64)
                  : (config_.block_m != 64 || (config_.block_n != 32 && config_.block_n != 64)))) {
      throw std::invalid_argument("unsupported bounded Tensor Core tile configuration");
    }
    k_ = get_int_option(options, "K", 128, 0);
    n_ = get_int_option(options, "N", 128, 0);
    int device = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "get Tensor Core benchmark device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "get Tensor Core capability");
    sm_ = properties.major * 10 + properties.minor;
    if (grouped_) {
      tokens_ = get_int_option(options, "T", 128, 0);
      experts_ = get_int_option(options, "E", 16);
      top_k_ = get_int_option(options, "top_k", std::min(2, experts_));
      if (experts_ > 64 || top_k_ > experts_) {
        throw std::invalid_argument("grouped Tensor Core benchmark requires top_k<=E<=64");
      }
      m_ = checked_int_product(tokens_, top_k_, "R=T*top_k");
      distribution_ = get_option(options, "distribution", "uniform");
      zipf_s_ = get_double_option(options, "zipf_s", 1.0);
      const auto ids = tokens_ == 0 ? std::vector<std::int32_t>{}
          : make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed + 1);
      counts_ = counts_from_ids(ids, experts_);
      offsets_host_ = offsets_from_counts(counts_);
      actual_max_ = *std::max_element(counts_.begin(), counts_.end());
      max_expert_tokens_ = get_int_option(options, "max_expert_tokens", actual_max_, 0);
      if (max_expert_tokens_ < actual_max_) {
        throw std::invalid_argument("max_expert_tokens must bound all device offset row counts");
      }
      active_experts_ = static_cast<int>(std::count_if(counts_.begin(), counts_.end(),
                                                     [](int value) { return value > 0; }));
      offsets_.resize(offsets_host_.size());
      offsets_.copy_from_host(offsets_host_, stream);
    } else {
      m_ = get_int_option(options, "M", 128, 0);
    }
    checked_int_product(m_, k_, "A elements");
    checked_int_product(checked_int_product(grouped_ ? experts_ : 1, k_, "expert K"),
                        n_, "B elements");
    checked_int_product(m_, n_, "C elements");
    a_master_host_ = make_random_floats(static_cast<std::size_t>(m_) * k_, seed);
    b_master_host_ = make_random_floats(static_cast<std::size_t>(grouped_ ? experts_ : 1) * k_ * n_,
                                       seed + (grouped_ ? 2 : 1));
    a_master_.assign(a_master_host_.begin(), a_master_host_.end());
    b_master_.assign(b_master_host_.begin(), b_master_host_.end());
    std::vector<std::uint16_t> a_storage(a_master_.size()), b_storage(b_master_.size());
    quantize_host(a_master_, a_storage, a_quantized_);
    quantize_host(b_master_, b_storage, b_quantized_);
    // The oracle always accumulates in FP64 over the exact values stored on the device.
    quantized_expected_ = reference(a_quantized_, b_quantized_);
    master_expected_ = reference(a_master_, b_master_);
    a_.resize(a_storage.size());
    b_.resize(b_storage.size());
    c_.resize(static_cast<std::size_t>(m_) * n_);
    a_.copy_from_host(a_storage, stream);
    b_.copy_from_host(b_storage, stream);
    if (conversion_ || fp32_parent_) {
      a_fp32_.resize(a_master_host_.size());
      b_fp32_.resize(b_master_host_.size());
      a_fp32_.copy_from_host(a_master_host_, stream);
      b_fp32_.copy_from_host(b_master_host_, stream);
    }
    if (variant_ == "cublas" || variant_ == "cublaslt") {
#if RAGGEDROUTE_HAS_CUBLAS
      try {
        gemm_plan_ = lib::create_gemm_plan(input_type_, variant_ == "cublaslt", m_, n_, k_);
      } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.rfind("not_supported:", 0) == 0) {
          throw std::runtime_error("unsupported_baseline: " + message);
        }
        throw;
      }
      workspace_bytes_ = lib::gemm_workspace_bytes(gemm_plan_);
      if (grouped_) {
        pinned_offset_bytes_ = (static_cast<std::size_t>(experts_) + 1) * sizeof(std::int32_t);
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pinned_offsets_), pinned_offset_bytes_,
                                 cudaHostAllocDefault), "allocate grouped cuBLAS pinned offsets");
        workspace_bytes_ += pinned_offset_bytes_;
      }
#endif
    } else if (variant_ == "cutlass_grouped") {
#if RAGGEDROUTE_HAS_CUTLASS
      cutlass_plan_ = lib::create_grouped_cutlass_plan(input_type_, a_.data(), b_.data(), c_.data(),
          offsets_.data(), offsets_host_.data(), experts_, k_, n_);
      if (cutlass_plan_ == nullptr) {
        throw std::runtime_error("unsupported_baseline: CUTLASS TensorOp grouped alignment/shape");
      }
      workspace_bytes_ = lib::grouped_cutlass_workspace_bytes(cutlass_plan_);
#endif
    }
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel, cudaStream_t stream) override {
    if (m_ == 0 || n_ == 0) return;
    if (conversion_) {
      cuda_check(ops::launch_gemm_quantize(a_fp32_.data(), a_.data(), a_.size(), input_type_, stream),
                 "quantize GEMM A inside timed boundary");
      cuda_check(ops::launch_gemm_quantize(b_fp32_.data(), b_.data(), b_.size(), input_type_, stream),
                 "quantize GEMM B inside timed boundary");
    }
    if (variant_ == "cublas" || variant_ == "cublaslt") {
#if RAGGEDROUTE_HAS_CUBLAS
      if (grouped_) {
        // cuBLAS GEMM shapes and pointers are host arguments. Consume the current
        // device offsets on every invocation and include D2H + synchronization.
        cuda_check(cudaMemcpyAsync(pinned_offsets_, offsets_.data(), pinned_offset_bytes_,
                                    cudaMemcpyDeviceToHost, stream), "grouped cuBLAS offsets D2H");
        cuda_check(cudaStreamSynchronize(stream), "grouped cuBLAS offsets D2H sync inside L2");
        for (int e = 0; e < experts_; ++e) {
          const int begin = pinned_offsets_[e], end = pinned_offsets_[e + 1];
          if (begin < 0 || end < begin || end > m_) {
            throw std::runtime_error("grouped cuBLAS received invalid device offsets");
          }
          const int rows = end - begin;
          if (rows == 0) continue;
          lib::launch_gemm(gemm_plan_, k_ == 0 ? a_.data() : a_.data() + static_cast<std::size_t>(begin) * k_,
              k_ == 0 ? b_.data() : b_.data() + static_cast<std::size_t>(e) * k_ * n_,
              c_.data() + static_cast<std::size_t>(begin) * n_, rows, n_, k_, stream);
        }
      } else {
        lib::launch_gemm(gemm_plan_, a_.data(), b_.data(), c_.data(), m_, n_, k_, stream);
      }
#endif
      return;
    }
    if (variant_ == "cutlass_grouped") {
#if RAGGEDROUTE_HAS_CUTLASS
      lib::launch_grouped_cutlass(cutlass_plan_, stream);
#endif
      return;
    }
    if (fp32_parent_) {
      if (grouped_) {
        cuda_check(ops::launch_grouped_gemm_sm86_fp32_v9_balanced_32x64(a_fp32_.data(),
            b_fp32_.data(), offsets_.data(), c_.data(), experts_, k_, n_, max_expert_tokens_, m_,
            stream), "launch original FP32 grouped V9");
      } else {
        cuda_check(ops::launch_dense_gemm_optimized(a_fp32_.data(), b_fp32_.data(), c_.data(),
            m_, n_, k_, ops::kDenseGemmRegisterTiledV3_64x32AsyncImplementation, stream),
            "launch original FP32 dense V3 with existing shape fallback");
      }
      return;
    }
    if (variant_ == "cuda_lowp_simt") {
      if (grouped_) {
        cuda_check(ops::launch_grouped_gemm_lowp_simt(a_.data(), b_.data(), offsets_.data(),
            c_.data(), experts_, k_, n_, max_expert_tokens_, input_type_, stream), "grouped lowp SIMT");
      } else {
        cuda_check(ops::launch_dense_gemm_lowp_simt(a_.data(), b_.data(), c_.data(), m_, n_, k_,
            input_type_, stream), "dense lowp SIMT");
      }
      return;
    }
    if (grouped_) {
      cuda_check(ops::launch_grouped_gemm_tensorcore(a_.data(), b_.data(), offsets_.data(),
          c_.data(), experts_, k_, n_, max_expert_tokens_, input_type_, config_, stream),
          "grouped Tensor Core candidate");
    } else {
      cuda_check(ops::launch_dense_gemm_tensorcore(a_.data(), b_.data(), c_.data(), m_, n_, k_,
          input_type_, config_, stream), "dense Tensor Core candidate");
    }
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto host = c_.copy_to_host(stream);
    const std::vector<double> actual(host.begin(), host.end());
    const auto& a = fp32_parent_ ? a_master_ : a_quantized_;
    const auto& b = fp32_parent_ ? b_master_ : b_quantized_;
    const auto& expected = fp32_parent_ ? master_expected_ : quantized_expected_;
    rc::CaseDescriptor descriptor;
    descriptor.case_id = "tensorcore_benchmark_validation";
    descriptor.operator_name = name_;
    descriptor.variant_name = variant_;
    descriptor.tensor_types.input = fp32_parent_ ? ScalarType::kFp32 : input_type_;
    descriptor.tensor_types.accumulator = descriptor.tensor_types.output = ScalarType::kFp32;
    descriptor.tensor_types.math_mode = fp32_parent_ ? rc::MathMode::kStrictFp32
        : input_type_ == ScalarType::kFp16 ? rc::MathMode::kFp16AccFp32 : rc::MathMode::kBf16AccFp32;
    rc::CheckReport report;
    if (!grouped_) {
      report = rc::compare_gemm(descriptor, actual, expected, a, b, m_, n_, k_, ScalarType::kFp32);
    } else {
      for (int e = 0; e < experts_; ++e) {
        const std::size_t begin = static_cast<std::size_t>(offsets_host_[e]);
        const std::size_t end = static_cast<std::size_t>(offsets_host_[e + 1]);
        if (begin == end) continue;
        const auto expert_report = rc::compare_gemm(descriptor,
            std::vector<double>(actual.begin() + begin * n_, actual.begin() + end * n_),
            std::vector<double>(expected.begin() + begin * n_, expected.begin() + end * n_),
            std::vector<double>(a.begin() + begin * k_, a.begin() + end * k_),
            std::vector<double>(b.begin() + static_cast<std::size_t>(e) * k_ * n_,
                                b.begin() + static_cast<std::size_t>(e + 1) * k_ * n_),
            static_cast<int>(end - begin), n_, k_, ScalarType::kFp32);
        if (!expert_report.ok()) { report = expert_report; break; }
      }
    }
    implementation_error_ = summarize_error(actual, expected);
    quantized_input_error_ = summarize_error(actual, quantized_expected_);
    original_input_error_ = summarize_error(actual, master_expected_);
    return {report.ok(), report.ok() ? "matched FP64 oracle within FP32 accumulation bound"
        : report.failures.front().message, implementation_error_.max_abs_error,
          implementation_error_.max_rel_error};
  }

  FieldMap case_config() const override {
    FieldMap result = {{"K", static_cast<std::int64_t>(k_)}, {"N", static_cast<std::int64_t>(n_)},
        {"dtype", fp32_parent_ ? std::string("fp32") : input_dtype_},
        {"output_dtype", std::string("fp32")}, {"accumulator_dtype", std::string("fp32")},
        {"oracle_accumulator_dtype", std::string("fp64")}, {"layout_a", std::string("row_major")},
        {"layout_b", std::string("row_major")}, {"layout_c", std::string("row_major")},
        {"alpha", 1.0}, {"beta", 0.0},
        {"input_boundary", std::string(conversion_ || fp32_parent_ ? "fp32_master" : "native_low_precision")}};
    if (grouped_) {
      result["T"] = static_cast<std::int64_t>(tokens_);
      result["E"] = static_cast<std::int64_t>(experts_);
      result["top_k"] = static_cast<std::int64_t>(top_k_);
      result["R"] = static_cast<std::int64_t>(m_);
      result["distribution"] = distribution_;
      result["zipf_s"] = zipf_s_;
      result["active_experts"] = static_cast<std::int64_t>(active_experts_);
      result["max_expert_tokens"] = static_cast<std::int64_t>(max_expert_tokens_);
    } else {
      result["M"] = static_cast<std::int64_t>(m_);
    }
    return result;
  }

  FieldMap variant_config() const override {
    const bool candidate = candidate_variant(variant_);
    const bool library = variant_ == "cublas" || variant_ == "cublaslt" || variant_ == "cutlass_grouped";
    FieldMap result = {{"implementation_category", std::string(library ? "library_baseline" : "in_tree_cuda")},
        {"implementation_version", std::string("raggedroute.tensorcore.v1")},
        {"implementation_revision", std::string(RAGGEDROUTE_GIT_SHA)},
        {"dependency_revision", dependency_revision(variant_)},
        {"algorithm_id", variant_}, {"math_mode", fp32_parent_ ? std::string("strict_fp32") : input_dtype_ + "_acc_fp32"},
        {"actual_path", actual_path()}, {"device_sm", static_cast<std::int64_t>(sm_)},
        {"output_dtype", std::string("fp32")}, {"accumulator_dtype", std::string("fp32")},
        {"conversion_in_timed_boundary", conversion_},
        {"runtime_status", std::string("benchmark_only_not_promoted")},
        {"implementation_reference_max_abs_error", implementation_error_.max_abs_error},
        {"implementation_reference_max_rel_error", implementation_error_.max_rel_error},
        {"implementation_reference_normalized_l2_error", implementation_error_.normalized_l2_error},
        {"quantized_reference_max_abs_error", quantized_input_error_.max_abs_error},
        {"quantized_reference_max_rel_error", quantized_input_error_.max_rel_error},
        {"quantized_reference_normalized_l2_error", quantized_input_error_.normalized_l2_error},
        {"original_fp32_reference_max_abs_error", original_input_error_.max_abs_error},
        {"original_fp32_reference_max_rel_error", original_input_error_.max_rel_error},
        {"original_fp32_reference_normalized_l2_error", original_input_error_.normalized_l2_error}};
    if (candidate) {
      result["tile_m"] = static_cast<std::int64_t>(config_.block_m);
      result["tile_n"] = static_cast<std::int64_t>(config_.block_n);
      result["tile_k"] = static_cast<std::int64_t>(config_.block_k);
      result["threads_per_block"] = static_cast<std::int64_t>(128);
      result["pipeline_stages"] = static_cast<std::int64_t>(config_.asynchronous ? 2 : 1);
      result["staging"] = std::string(config_.asynchronous ? "cp_async_16B_double_buffer" : "synchronous_shared");
      result["mma_instruction"] = std::string("mma.sync.aligned.m16n8k16.row.col.f32.") +
          (input_type_ == ScalarType::kFp16 ? "f16.f16.f32" : "bf16.bf16.f32");
      result["scheduler"] = std::string(grouped_ ? "device_offsets_direct_expert_row_column_grid" : "direct_row_column_grid");
    }
    if (variant_ == "cublas" && grouped_) {
      result["launches"] = static_cast<std::int64_t>(active_experts_);
      result["scheduler"] = std::string("host_loop_over_current_device_offsets_active_experts");
      result["offsets_d2h_and_stream_sync_in_timed_boundary"] = true;
      result["pinned_host_offset_bytes"] = static_cast<std::int64_t>(pinned_offset_bytes_);
    }
    if (variant_ == "cutlass_grouped") {
      result["operator_class"] = std::string("TensorOp");
      result["device_metadata_preparation_in_timed_boundary"] = true;
    }
    return result;
  }
  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    work.flops = 2.0 * m_ * n_ * k_;
    const double input_bytes = fp32_parent_ ? 4.0 : 2.0;
    work.logical_bytes = input_bytes * (static_cast<double>(m_) * k_ +
        static_cast<double>(grouped_ ? active_experts_ : 1) * k_ * n_) + 4.0 * m_ * n_;
    if (conversion_) work.logical_bytes += 6.0 * (a_.size() + b_.size());
    work.operator_metrics["output_elements"] = static_cast<std::int64_t>(m_) * n_;
    return work;
  }
  std::size_t workspace_bytes() const override {
    return workspace_bytes_ + (conversion_ ? a_.bytes() + b_.bytes() : 0);
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    std::vector<std::string> steps = {"input_generation", "cpu_fp64_reference", "h2d_copy", "workspace_allocation"};
    if (grouped_) steps.push_back("fixture_route_generation");
    if (!conversion_) steps.push_back("input_quantization");
    return steps;
  }

 private:
  void reject_unknown(const OptionMap& options) const {
    const std::set<std::string> allowed = {"M", "N", "K", "dtype", "tile_m", "tile_n", "tile_k",
        "T", "E", "top_k", "distribution", "zipf_s", "max_expert_tokens"};
    for (const auto& entry : options) {
      if (!allowed.count(entry.first) || (!grouped_ && (entry.first == "T" || entry.first == "E" ||
          entry.first == "top_k" || entry.first == "distribution" || entry.first == "zipf_s" ||
          entry.first == "max_expert_tokens")) || (grouped_ && entry.first == "M")) {
        throw std::invalid_argument("unknown " + name_ + " param: " + entry.first);
      }
    }
  }
  void quantize_host(const std::vector<double>& master, std::vector<std::uint16_t>& storage,
                     std::vector<double>& values) const {
    values.resize(master.size());
    for (std::size_t i = 0; i < master.size(); ++i) {
      storage[i] = static_cast<std::uint16_t>(rc::encode_scalar(input_type_, master[i]));
      values[i] = rc::decode_scalar(input_type_, storage[i]);
    }
  }
  std::vector<double> reference(const std::vector<double>& a, const std::vector<double>& b) const {
    if (grouped_) return rc::grouped_gemm_reference(a, b, offsets_host_, experts_, k_, n_);
    return rc::dense_gemm_reference(a, b, {}, m_, n_, k_, 1.0, 0.0);
  }
  std::string actual_path() const {
    if (m_ == 0 || n_ == 0) return "empty_noop";
    if (k_ == 0) return "zero_fill_no_mma";
    if (fp32_parent_) {
      if (grouped_) return "strict_fp32_v9_with_existing_shape_fallback";
      if (m_ % 64 == 0 && n_ % 32 == 0 && k_ % 16 == 0) return "strict_fp32_v3_64x32_async";
      return m_ % 16 == 0 && n_ % 16 == 0 && k_ % 16 == 0
          ? "strict_fp32_tiled_vector_existing_v3_fallback" : "strict_fp32_tiled_scalar_existing_v3_fallback";
    }
    if (candidate_variant(variant_)) return sm_ >= 80 ? "mma_sync_m16n8k16" : "simt_architecture_fallback";
    if (variant_ == "cuda_lowp_simt") return "low_precision_input_fp32_fma_simt";
    if (variant_ == "cutlass_grouped") return "cutlass_tensorop_grouped";
    return variant_ + "_library_selected_algorithm";
  }

  std::string name_, variant_, input_dtype_;
  bool grouped_ = false, fp32_parent_ = false, conversion_ = false;
  ScalarType input_type_ = ScalarType::kFp16;
  ops::TensorCoreConfig config_;
  int m_ = 0, n_ = 0, k_ = 0, tokens_ = 0, experts_ = 1, top_k_ = 1;
  int sm_ = 0, actual_max_ = 0, max_expert_tokens_ = 0, active_experts_ = 1;
  std::string distribution_;
  double zipf_s_ = 1.0;
  std::size_t workspace_bytes_ = 0;
  std::size_t pinned_offset_bytes_ = 0;
  std::int32_t* pinned_offsets_ = nullptr;
  lib::GemmPlan* gemm_plan_ = nullptr;
  lib::GroupedCutlassPlan* cutlass_plan_ = nullptr;
  std::vector<std::int32_t> counts_, offsets_host_;
  std::vector<float> a_master_host_, b_master_host_;
  std::vector<double> a_master_, b_master_, a_quantized_, b_quantized_;
  std::vector<double> quantized_expected_, master_expected_;
  DeviceBuffer<std::uint16_t> a_, b_;
  DeviceBuffer<float> c_, a_fp32_, b_fp32_;
  DeviceBuffer<std::int32_t> offsets_;
  rc::NumericSummary implementation_error_, quantized_input_error_, original_input_error_;
};
}  // namespace

std::vector<VariantDescriptor> tensorcore_gemm_variant_descriptors(const std::string& name) {
  if (name != "dense_gemm_tc" && name != "grouped_gemm_tc") return {};
  std::vector<VariantDescriptor> descriptors;
  for (const std::string variant : {"cuda_mma_sync", "cuda_mma_async", "cuda_lowp_simt",
      "cuda_mma_sync_convert", "cuda_mma_async_convert", "fp32_parent"}) {
    descriptors.push_back({variant, "in_tree_cuda", "raggedroute.tensorcore.v1", "not_applicable",
        variant, variant == "fp32_parent" ? "strict_fp32" : "input_dtype_acc_fp32"});
  }
#if RAGGEDROUTE_HAS_CUBLAS
  descriptors.push_back({"cublas", "library_baseline", "cuda.cublas", dependency_revision("cublas"), "GemmEx", "input_dtype_acc_fp32"});
  if (name == "dense_gemm_tc") descriptors.push_back({"cublaslt", "library_baseline", "cuda.cublasLt", dependency_revision("cublaslt"), "LtMatmul", "input_dtype_acc_fp32"});
#endif
#if RAGGEDROUTE_HAS_CUTLASS
  if (name == "grouped_gemm_tc") descriptors.push_back({"cutlass_grouped", "library_baseline", "cutlass.TensorOp.Grouped", dependency_revision("cutlass_grouped"), "TensorOpGrouped", "input_dtype_acc_fp32"});
#endif
  return descriptors;
}

AdapterPtr make_tensorcore_gemm_adapter(const std::string& name, const std::string& variant) {
  const auto descriptors = tensorcore_gemm_variant_descriptors(name);
  if (std::none_of(descriptors.begin(), descriptors.end(), [&variant](const VariantDescriptor& item) {
        return item.name == variant;
      })) throw std::invalid_argument("unknown Tensor Core GEMM operator/variant: " + name + "/" + variant);
  return std::make_unique<TensorCoreGemmAdapter>(name, variant);
}
}  // namespace raggedroute::benchmark
