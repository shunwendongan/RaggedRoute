#include "raggedroute/benchmark/types.h"

#include <stdexcept>

namespace raggedroute::benchmark {

std::string to_string(MeasurementLevel value) {
  switch (value) {
    case MeasurementLevel::kKernelBody:
      return "L1_kernel_body";
    case MeasurementLevel::kOperatorSteady:
      return "L2_operator_steady";
    case MeasurementLevel::kChainSteady:
      return "L3_chain_steady";
    case MeasurementLevel::kHostCall:
      return "L4_host_call";
  }
  throw std::invalid_argument("unknown measurement level");
}

std::string to_string(Protocol value) {
  switch (value) {
    case Protocol::kSmoke:
      return "smoke";
    case Protocol::kRelease:
      return "release";
  }
  throw std::invalid_argument("unknown benchmark protocol");
}

std::string to_string(CacheMode value) {
  switch (value) {
    case CacheMode::kWarm:
      return "warm";
    case CacheMode::kColdScrub:
      return "cold_scrub";
  }
  throw std::invalid_argument("unknown cache mode");
}

MeasurementLevel parse_measurement_level(const std::string& value) {
  if (value == "L1_kernel_body" || value == "l1") {
    return MeasurementLevel::kKernelBody;
  }
  if (value == "L2_operator_steady" || value == "l2") {
    return MeasurementLevel::kOperatorSteady;
  }
  if (value == "L3_chain_steady" || value == "l3") {
    return MeasurementLevel::kChainSteady;
  }
  if (value == "L4_host_call" || value == "l4") {
    return MeasurementLevel::kHostCall;
  }
  throw std::invalid_argument("invalid measurement level: " + value);
}

Protocol parse_protocol(const std::string& value) {
  if (value == "smoke") return Protocol::kSmoke;
  if (value == "release") return Protocol::kRelease;
  throw std::invalid_argument("invalid benchmark protocol: " + value);
}

CacheMode parse_cache_mode(const std::string& value) {
  if (value == "warm") return CacheMode::kWarm;
  if (value == "cold_scrub") return CacheMode::kColdScrub;
  throw std::invalid_argument("invalid cache mode: " + value);
}

}  // namespace raggedroute::benchmark
