#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aevum_cuda_register_tune {

inline bool enabled(const char* value) {
  if (!value || std::string(value) == "0") return false;
  if (std::string(value) == "1") return true;
  throw std::runtime_error("AEVUM_CUDA_REGISTER_TUNE must be 0 or 1");
}

// PRPLL f222e885: search kernel launch bounds / register caps separately for
// each FFT. This bounded neighbourhood keeps the current production profile
// as its centre. It does not reset NOREG or unrelated validated settings.
inline std::vector<std::string> candidates(
    unsigned fft_type,
    const std::vector<std::pair<std::string, std::string>>& baseline) {
  if (fft_type != 1 && fft_type != 2)
    throw std::runtime_error("CUDA register tuning requires Type1 or Type2");
  const std::string small = fft_type == 1 ? "31" : "32";
  const std::vector<std::string> keys = {
      "REGCF" + std::string(fft_type == 1 ? "3161" : "3261"),
      "REGMI61", "REGMO61", "REGTS61",
      "REGMI" + small, "REGMO" + small, "REGTS" + small};
  const std::map<std::string, std::string> current(baseline.begin(), baseline.end());
  std::vector<std::string> result;
  // -1: compiler default; 1..4: minimum blocks/SM; >=32: register caps.
  // Value-major ordering visits every active kernel before trying the next
  // limit, so a short resumed invocation cannot starve the last field.
  for (int limit : {-1, 1, 2, 3, 4, 64, 80, 96, 128}) {
    for (const auto& key : keys) {
      const std::string value = std::to_string(limit);
      const auto found = current.find(key);
      if (found == current.end() || found->second != value)
        result.push_back(key + '=' + value);
    }
  }
  return result;
}

} // namespace aevum_cuda_register_tune
