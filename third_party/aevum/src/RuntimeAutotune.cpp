#include "RuntimeAutotune.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace aevum_autotune {
namespace {
std::string envString(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "";
}

std::string clean(std::string text) {
  std::replace(text.begin(), text.end(), '\t', ' ');
  std::replace(text.begin(), text.end(), '\n', ' ');
  std::replace(text.begin(), text.end(), '\r', ' ');
  return text;
}

bool parseBool(const std::string& s, bool& value) {
  if (s == "0") { value = false; return true; }
  if (s == "1") { value = true; return true; }
  return false;
}

std::vector<std::string> splitTabs(const std::string& line) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t pos = line.find('\t', start);
    out.push_back(line.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
    if (pos == std::string::npos) break;
    start = pos + 1;
  }
  return out;
}
} // namespace

Mode modeFromEnvironment() {
  std::string value = envString("AEVUM_AUTOTUNE");
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return char(std::tolower(c)); });
  if (value.empty() || value == "auto" || value == "on" || value == "1") return Mode::Auto;
  if (value == "off" || value == "0") return Mode::Off;
  if (value == "retune" || value == "force-retune" || value == "force") return Mode::Retune;
  throw std::runtime_error("AEVUM_AUTOTUNE must be off, auto, or retune");
}

std::string modeName(Mode mode) {
  switch (mode) {
    case Mode::Off: return "off";
    case Mode::Retune: return "retune";
    default: return "auto";
  }
}

std::string workloadClass(Workload workload, std::size_t register_count) {
  switch (workload) {
    case Workload::Prp: return "square-prp";
    case Workload::Ll: return "square-ll";
    case Workload::Pm1:
    case Workload::Pm1Lowmem: return register_count > 16 ? "pm1-mixed" : "pm1-exp";
    case Workload::Pm1Ultralowmem: return "pm1-ultralow";
    case Workload::Ecm: return "ecm-mixed";
    default: return "generic";
  }
}

uint32_t exponentBandStart(uint32_t exponent) {
  constexpr uint32_t band = 10000000u;
  return (exponent / band) * band;
}

std::string makeKey(const std::string& engine_version,
                    const std::string& vendor,
                    const std::string& device,
                    const std::string& driver,
                    const std::string& runtime,
                    Workload workload,
                    std::size_t register_count,
                    uint32_t exponent,
                    const std::string& relevant_flags) {
  const uint32_t band = exponentBandStart(exponent);
  std::ostringstream out;
  out << "schema=" << kSchema
      << "|engine=" << clean(engine_version)
      << "|vendor=" << clean(vendor)
      << "|device=" << clean(device)
      << "|driver=" << clean(driver)
      << "|runtime=" << clean(runtime)
      << "|workload=" << workloadClass(workload, register_count)
      << "|band=" << band << '-' << (uint64_t(band) + 9999999u)
      << "|regs=" << (register_count <= 4 ? "1-4" : register_count <= 16 ? "5-16" : "17+")
      << "|flags=" << clean(relevant_flags);
  return out.str();
}

std::filesystem::path cachePath() {
  if (const char* explicit_path = std::getenv("AEVUM_AUTOTUNE_CACHE")) {
    if (*explicit_path) return std::filesystem::absolute(explicit_path);
  }
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    if (*xdg) return std::filesystem::path(xdg) / "prmers" / "aevum-autotune-v1.tsv";
  }
  if (const char* home = std::getenv("HOME")) {
    if (*home) return std::filesystem::path(home) / ".cache" / "prmers" / "aevum" / "autotune-v1.tsv";
  }
  return std::filesystem::absolute(".aevum-autotune-v1.tsv");
}

std::optional<Record> load(const std::filesystem::path& path, const std::string& key) {
  std::ifstream in(path);
  if (!in) return std::nullopt;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto f = splitTabs(line);
    if (f.size() != 8 || f[0] != "1") continue;
    if (f[1] != key) continue;
    try {
      Record r;
      r.key = f[1];
      r.plan = f[2];
      if (r.plan.empty() || !parseBool(f[3], r.prepared_mul_lead)) continue;
      r.plan_speedup = std::stod(f[4]);
      r.implementation_speedup = std::stod(f[5]);
      r.tune_ms = std::stoull(f[6]);
      r.created_unix = std::stoull(f[7]);
      if (!(r.plan_speedup > 0.0) || !(r.implementation_speedup > 0.0)) continue;
      return r;
    } catch (...) {
      // Corrupt records are input data. Ignore them and trigger a safe retune.
    }
  }
  return std::nullopt;
}

void storeAtomic(const std::filesystem::path& path, const Record& record) {
  if (record.key.empty() || record.plan.empty()) throw std::runtime_error("invalid Aevum autotune cache record");
  std::error_code ec;
  const auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent, ec);
  if (ec) throw std::runtime_error("cannot create Aevum autotune cache directory: " + ec.message());

  std::vector<std::string> keep;
  {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') { keep.push_back(line); continue; }
      const auto f = splitTabs(line);
      if (f.size() >= 2 && f[1] == record.key) continue;
      keep.push_back(line);
    }
  }

  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path temp = path;
  temp += ".tmp." + std::to_string(nonce);
  {
    std::ofstream out(temp, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write Aevum autotune cache");
    out << "# AEVUM runtime autotune cache v1\n";
    for (const auto& line : keep) {
      if (line.rfind("# AEVUM runtime autotune cache", 0) == 0) continue;
      if (!line.empty()) out << line << '\n';
    }
    out << "1\t" << clean(record.key)
        << '\t' << clean(record.plan)
        << '\t' << (record.prepared_mul_lead ? 1 : 0)
        << '\t' << std::setprecision(8) << record.plan_speedup
        << '\t' << std::setprecision(8) << record.implementation_speedup
        << '\t' << record.tune_ms
        << '\t' << record.created_unix << '\n';
    out.flush();
    if (!out) throw std::runtime_error("cannot flush Aevum autotune cache");
  }
#if defined(_WIN32)
  // MoveFileEx with REPLACE_EXISTING keeps the cache replacement atomic on the
  // supported local Windows filesystems; WRITE_THROUGH avoids publishing an
  // entry before the replacement has reached stable storage.
  if (!MoveFileExW(temp.wstring().c_str(), path.wstring().c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    const auto code = GetLastError();
    std::filesystem::remove(temp);
    throw std::runtime_error("cannot atomically replace Aevum autotune cache (Win32 error " +
                             std::to_string(static_cast<unsigned long>(code)) + ")");
  }
#else
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    std::filesystem::remove(temp);
    throw std::runtime_error("cannot atomically replace Aevum autotune cache: " + ec.message());
  }
#endif
}

bool hasManualPlanOverrideEnvironment() {
  static const char* names[] = {
    "AEVUM_TUNE_DIR", "AEVUM_GB202_TUNE", "AEVUM_RADIX1K",
    "AEVUM_TYPE4_MULTI_Q", "AEVUM_PFA_USE", "AEVUM_GF61_LIMB32",
    "AEVUM_CARRY_WMUL", "AEVUM_PRP_MIDDLE1", "AEVUM_REG_LEAD_CACHE",
    "PRMERS_AEVUM_PRP_FFT", "PRMERS_AEVUM_LL_FFT", "PRMERS_AEVUM_PM1_FFT",
    "PRMERS_AEVUM_ECM_FFT"
  };
  for (const char* name : names) {
    const char* value = std::getenv(name);
    if (value && *value) return true;
  }
  return false;
}

} // namespace aevum_autotune
