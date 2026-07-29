#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {
namespace {

std::string escape_json(const std::string& input) {
  std::ostringstream output;
  for (unsigned char value : input) {
    switch (value) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (value < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(value) << std::dec;
        } else {
          output << static_cast<char>(value);
        }
    }
  }
  return output.str();
}

void quoted(std::ostringstream& output, const std::string& value) {
  output << '"' << escape_json(value) << '"';
}

void number_or_null(std::ostringstream& output, double value) {
  if (std::isfinite(value)) {
    output << std::setprecision(17) << value;
  } else {
    output << "null";
  }
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open correctness JSON: " + path);
  std::ostringstream content;
  content << input.rdbuf();
  return content.str();
}

void write_file(const std::string& path, const std::string& content) {
  const std::filesystem::path output(path);
  if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create correctness JSON: " + path);
  stream << content << '\n';
  if (!stream) throw std::runtime_error("failed to write correctness JSON: " + path);
}

std::string json_string(const std::string& json, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::invalid_argument("missing JSON string field: " + key);
  }
  return match[1].str();
}

std::int64_t json_integer(const std::string& json, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::invalid_argument("missing JSON integer field: " + key);
  }
  return std::stoll(match[1].str());
}

std::map<std::string, std::int64_t> json_shape(const std::string& json) {
  const std::regex object_pattern("\\\"shape\\\"\\s*:\\s*\\{([^}]*)\\}");
  std::smatch object;
  if (!std::regex_search(json, object, object_pattern)) {
    throw std::invalid_argument("missing JSON shape object");
  }
  const std::string body = object[1].str();
  const std::regex field_pattern("\\\"([^\\\"]+)\\\"\\s*:\\s*(-?[0-9]+)");
  std::map<std::string, std::int64_t> shape;
  for (std::sregex_iterator iterator(body.begin(), body.end(), field_pattern), end; iterator != end;
       ++iterator) {
    shape[(*iterator)[1].str()] = std::stoll((*iterator)[2].str());
  }
  return shape;
}

}  // namespace

std::string case_to_json(const CaseDescriptor& descriptor) {
  std::ostringstream output;
  output << '{';
  quoted(output, "schema_version");
  output << ':';
  quoted(output, "raggedroute.correctness.case.v1");
  output << ',';
  quoted(output, "case_id");
  output << ':';
  quoted(output, descriptor.case_id);
  output << ',';
  quoted(output, "operator");
  output << ':';
  quoted(output, descriptor.operator_name);
  output << ',';
  quoted(output, "variant");
  output << ':';
  quoted(output, descriptor.variant_name);
  output << ',';
  quoted(output, "seed");
  output << ':' << descriptor.seed << ',';
  quoted(output, "input_dtype");
  output << ':';
  quoted(output, to_string(descriptor.tensor_types.input));
  output << ',';
  quoted(output, "accumulator_dtype");
  output << ':';
  quoted(output, to_string(descriptor.tensor_types.accumulator));
  output << ',';
  quoted(output, "output_dtype");
  output << ':';
  quoted(output, to_string(descriptor.tensor_types.output));
  output << ',';
  quoted(output, "math_mode");
  output << ':';
  quoted(output, to_string(descriptor.tensor_types.math_mode));
  output << ',';
  quoted(output, "data_pattern");
  output << ':';
  quoted(output, descriptor.data_pattern);
  output << ',';
  quoted(output, "route_distribution");
  output << ':';
  quoted(output, descriptor.route_distribution);
  output << ',';
  quoted(output, "shape");
  output << ":{";
  bool first = true;
  for (const auto& [name, value] : descriptor.shape) {
    if (!first) output << ',';
    first = false;
    quoted(output, name);
    output << ':' << value;
  }
  output << "}}";
  return output.str();
}

CaseDescriptor case_from_json(const std::string& json) {
  CaseDescriptor descriptor;
  descriptor.case_id = json_string(json, "case_id");
  descriptor.operator_name = json_string(json, "operator");
  descriptor.variant_name = json_string(json, "variant");
  descriptor.seed = static_cast<std::uint64_t>(json_integer(json, "seed"));
  descriptor.tensor_types.input = parse_scalar_type(json_string(json, "input_dtype"));
  descriptor.tensor_types.accumulator = parse_scalar_type(json_string(json, "accumulator_dtype"));
  descriptor.tensor_types.output = parse_scalar_type(json_string(json, "output_dtype"));
  descriptor.tensor_types.math_mode = parse_math_mode(json_string(json, "math_mode"));
  descriptor.data_pattern = json_string(json, "data_pattern");
  descriptor.route_distribution = json_string(json, "route_distribution");
  descriptor.shape = json_shape(json);
  return descriptor;
}

void save_case_json(const std::string& path, const CaseDescriptor& descriptor) {
  write_file(path, case_to_json(descriptor));
}

CaseDescriptor load_case_json(const std::string& path) { return case_from_json(read_file(path)); }

std::string report_to_json(const CheckReport& report, const CaseDescriptor& descriptor) {
  std::ostringstream output;
  output << '{';
  quoted(output, "schema_version");
  output << ':';
  quoted(output, "raggedroute.correctness.failure.v1");
  output << ',';
  quoted(output, "status");
  output << ':';
  quoted(output, to_string(report.status));
  output << ',';
  quoted(output, "case");
  output << ':' << case_to_json(descriptor) << ',';
  quoted(output, "launch_error");
  output << ':' << static_cast<int>(report.launch_error) << ',';
  quoted(output, "execution_error");
  output << ':' << static_cast<int>(report.execution_error) << ',';
  quoted(output, "canaries_ok");
  output << ':' << (report.canaries_ok ? "true" : "false") << ',';
  quoted(output, "inputs_unchanged");
  output << ':' << (report.inputs_unchanged ? "true" : "false") << ',';
  quoted(output, "skip_reason");
  output << ':';
  quoted(output, report.skip_reason);
  output << ',';
  quoted(output, "numeric");
  output << ":{";
  quoted(output, "max_abs_error");
  output << ':';
  number_or_null(output, report.numeric.max_abs_error);
  output << ',';
  quoted(output, "max_rel_error");
  output << ':';
  number_or_null(output, report.numeric.max_rel_error);
  output << ',';
  quoted(output, "normalized_l2_error");
  output << ':';
  number_or_null(output, report.numeric.normalized_l2_error);
  output << "},";
  quoted(output, "failures");
  output << ":[";
  for (std::size_t index = 0; index < report.failures.size(); ++index) {
    if (index != 0) output << ',';
    const CheckFailure& failure = report.failures[index];
    output << '{';
    quoted(output, "check");
    output << ':';
    quoted(output, failure.check);
    output << ',';
    quoted(output, "message");
    output << ':';
    quoted(output, failure.message);
    if (failure.linear_index) {
      output << ',';
      quoted(output, "linear_index");
      output << ':' << *failure.linear_index;
    }
    if (failure.actual) {
      output << ',';
      quoted(output, "actual");
      output << ':';
      number_or_null(output, *failure.actual);
    }
    if (failure.expected) {
      output << ',';
      quoted(output, "expected");
      output << ':';
      number_or_null(output, *failure.expected);
    }
    if (failure.allowed_error) {
      output << ',';
      quoted(output, "allowed_error");
      output << ':';
      number_or_null(output, *failure.allowed_error);
    }
    output << '}';
  }
  output << "]}";
  return output.str();
}

void save_failure_artifact(const std::string& path, const CheckReport& report,
                           const CaseDescriptor& descriptor) {
  write_file(path, report_to_json(report, descriptor));
}

}  // namespace raggedroute::correctness
