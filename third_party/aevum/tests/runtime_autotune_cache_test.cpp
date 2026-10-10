#include "RuntimeAutotune.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#if defined(_WIN32)
static void set_env(const char* k, const char* v) { _putenv_s(k, v ? v : ""); }
#else
#include <cstdlib>
static void set_env(const char* k, const char* v) { if (v) setenv(k, v, 1); else unsetenv(k); }
#endif

int main() {
  using namespace aevum_autotune;
  const auto tmp = std::filesystem::temp_directory_path() / "aevum-autotune-cache-test.tsv";
  std::error_code ec;
  std::filesystem::remove(tmp, ec);

  const std::string key = makeKey("v100.11", "4318", "RTX 3080", "driver-a", "OpenCL 3.0",
                                  Workload::Prp, 8, 180000007u, "radix1k=4;multiq=1");
  assert(key.find("workload=square-prp") != std::string::npos);
  assert(key.find("band=180000000-189999999") != std::string::npos);
  assert(key != makeKey("v100.11", "4318", "RTX 3080", "driver-b", "OpenCL 3.0",
                        Workload::Prp, 8, 180000007u, "radix1k=4;multiq=1"));
  assert(key != makeKey("v100.11", "4318", "RTX 3080", "driver-a", "OpenCL 3.0",
                        Workload::Ecm, 51, 180000007u, "radix1k=4;multiq=1"));
  assert(key != makeKey("v100.11", "4318", "RTX 3080", "driver-a", "OpenCL 3.0",
                        Workload::Prp, 8, 200000007u, "radix1k=4;multiq=1"));

  Record a{key, "4:512:8:512:202", true, 1.72, 1.08, 4321, 123, true, {}};
  storeAtomic(tmp, a);
  auto loaded = load(tmp, key);
  assert(loaded && loaded->plan == a.plan && loaded->prepared_mul_lead);
  assert(loaded->tune_ms == 4321);

  Record replacement{key, "1:512:16:512:202", false, 1.10, 1.0, 2222, 456, true, {}};
  storeAtomic(tmp, replacement);
  loaded = load(tmp, key);
  assert(loaded && loaded->plan == replacement.plan && !loaded->prepared_mul_lead);
  assert(!load(tmp, key + "-stale"));
  assert(loaded->complete && loaded->tried.empty());

  // A search cut short is stored as deferred with the plans already tried,
  // and a complete record keeps the original eight fields.
  Record deferred{key, "1:512:16:512:202", false, 1.10, 1.0, 7100, 789, false, {"1:1K:8:512:202", "1:512:16:512:202"}};
  storeAtomic(tmp, deferred);
  loaded = load(tmp, key);
  assert(loaded && !loaded->complete && loaded->tried == deferred.tried);
  assert(loaded->plan == deferred.plan && loaded->tune_ms == 7100);
  deferred.tried.clear();
  storeAtomic(tmp, deferred);
  loaded = load(tmp, key);
  assert(loaded && !loaded->complete && loaded->tried.empty());
  storeAtomic(tmp, replacement);
  {
    std::ifstream in(tmp);
    std::string line, last;
    while (std::getline(in, line)) if (line.rfind("1\t", 0) == 0) last = line;
    size_t tabs = 0;
    for (char c : last) tabs += c == '\t';
    assert(tabs == 7);
  }
  loaded = load(tmp, key);
  assert(loaded && loaded->complete && loaded->tried.empty());
  {
    std::ofstream out(tmp, std::ios::app);
    out << "1\tbad-state\t1:512:16:512:202\t0\t1.1\t1\t1\t1\tpartial\n";
    out << "1\tempty-plan\t1:512:16:512:202\t0\t1.1\t1\t1\t1\tdeferred:a,,b\n";
  }
  assert(!load(tmp, "bad-state"));
  assert(!load(tmp, "empty-plan"));

  // Corruption is ignored rather than trusted.
  {
    std::ofstream out(tmp, std::ios::app);
    out << "1\tbroken-key\t4:bad\tnot-bool\tnan\t0\tbad\tbad\n";
  }
  assert(!load(tmp, "broken-key"));

  set_env("AEVUM_AUTOTUNE", "off"); assert(modeFromEnvironment() == Mode::Off);
  set_env("AEVUM_AUTOTUNE", "auto"); assert(modeFromEnvironment() == Mode::Auto);
  set_env("AEVUM_AUTOTUNE", "retune"); assert(modeFromEnvironment() == Mode::Retune);
  set_env("AEVUM_AUTOTUNE", "force-retune"); assert(modeFromEnvironment() == Mode::Retune);
  set_env("AEVUM_AUTOTUNE", "FORCE-RETUNE"); assert(modeFromEnvironment() == Mode::Retune);
  set_env("AEVUM_AUTOTUNE", nullptr);

  set_env("AEVUM_TUNE_DIR", nullptr); assert(!hasManualPlanOverrideEnvironment());
  set_env("AEVUM_TUNE_DIR", "/tmp/pinned-tune-root");
  assert(!hasManualPlanOverrideEnvironment());
  set_env("AEVUM_TUNE_DIR", nullptr);

  set_env("AEVUM_RADIX1K", "8");
  assert(hasManualPlanOverrideEnvironment());
  set_env("AEVUM_RADIX1K", nullptr);
  assert(!hasManualPlanOverrideEnvironment());

  std::filesystem::remove(tmp, ec);
  std::cout << "runtime_autotune_cache_test: OK\n";
}
