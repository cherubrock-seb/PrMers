#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace aevum_autotune {

constexpr unsigned kSchema = 1;

enum class Mode { Off, Auto, Retune };
enum class Workload : uint32_t {
  Generic = 0,
  Prp = 1,
  Ll = 2,
  Pm1 = 3,
  Pm1Lowmem = 4,
  Pm1Ultralowmem = 5,
  Ecm = 6,
};

struct Record {
  std::string key;
  std::string plan;
  bool prepared_mul_lead = false;
  double plan_speedup = 1.0;
  double implementation_speedup = 1.0;
  uint64_t tune_ms = 0;
  uint64_t created_unix = 0;
};

Mode modeFromEnvironment();
std::string modeName(Mode mode);
std::string workloadClass(Workload workload, std::size_t register_count);
uint32_t exponentBandStart(uint32_t exponent);
std::string makeKey(const std::string& engine_version,
                    const std::string& vendor,
                    const std::string& device,
                    const std::string& driver,
                    const std::string& runtime,
                    Workload workload,
                    std::size_t register_count,
                    uint32_t exponent,
                    const std::string& relevant_flags);
std::filesystem::path cachePath();
std::optional<Record> load(const std::filesystem::path& path, const std::string& key);
void storeAtomic(const std::filesystem::path& path, const Record& record);

// Environment variables which directly choose an FFT/tuning implementation.
// AUTOTUNE controls/cache paths are intentionally not blockers.
bool hasManualPlanOverrideEnvironment();

} // namespace aevum_autotune
