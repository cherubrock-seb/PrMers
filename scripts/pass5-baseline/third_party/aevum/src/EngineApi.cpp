/*
Aevum register API adaptation, Copyright 2026 cherubrock-seb.
Derived from GPUOwl/PRPLL by Mihai Preda and George Woltman.
Licensed under GNU GPL version 3. See LICENSE and UPSTREAM.md.
*/

#include "EngineApi.h"

#include "Args.h"
#include "Background.h"
#include "Context.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "TrigBufCache.h"
#include "TuneEntry.h"
#include "RuntimeAutotune.h"
#include "common.h"
#include "gpuid.h"
#include "version.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_set>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_error(const char* text) {
  g_last_error = text ? text : "unknown Aevum engine error";
}

void set_error(const std::exception& e) {
  g_last_error = e.what();
}

std::string openclDeviceInfoString(cl_device_id device, cl_device_info param) {
  size_t bytes = 0;
  if (clGetDeviceInfo(device, param, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0)
    return {};
  std::string value(bytes, '\0');
  if (clGetDeviceInfo(device, param, bytes, value.data(), nullptr) != CL_SUCCESS)
    return {};
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::string openclDeviceName(cl_device_id device) {
  return openclDeviceInfoString(device, CL_DEVICE_NAME);
}

std::string openclVendorId(cl_device_id device) {
  uint32_t value = 0;
  if (clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(value), &value, nullptr) != CL_SUCCESS) return {};
  return std::to_string(static_cast<unsigned>(value));
}

bool gb202DeviceName(const std::string& name) {
  return name.find("RTX 5090") != std::string::npos ||
         name.find("GB202") != std::string::npos;
}

bool envExactly(const char* name, const char* expected) {
  const char* value = std::getenv(name);
  return value && std::strcmp(value, expected) == 0;
}

double median3(std::array<double, 3> values) {
  std::sort(values.begin(), values.end());
  return values[1];
}

unsigned boundedEnvUnsigned(const char* name, unsigned fallback, unsigned lo, unsigned hi) {
  const char* text = std::getenv(name);
  if (!text || !*text) return fallback;
  char* end = nullptr;
  const unsigned long v = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') return fallback;
  return static_cast<unsigned>(std::max<unsigned long>(lo, std::min<unsigned long>(hi, v)));
}

double positiveEnvDouble(const char* name, double fallback, double lo, double hi) {
  const char* text = std::getenv(name);
  if (!text || !*text) return fallback;
  char* end = nullptr;
  const double v = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(v)) return fallback;
  return std::max(lo, std::min(hi, v));
}

bool usableTuneEntry(const Args& args, uint32_t exponent) {
  for (const TuneEntry& tuned : TuneEntry::readTuneFile(args)) {
    if (tuned.fft.shape.fft_type != FFT3161 || tuned.fft.isPfa()) continue;
    const double bpw = exponent / double(tuned.fft.size());
    if (bpw < tuned.fft.minBpw()) continue;
    if (tuned.fft.maxExp() * args.fftOverdrive >= exponent) return true;
  }
  return false;
}

std::optional<FFTConfig> admissiblePlan(const Args& args, uint32_t exponent, const std::string& spec) {
  try {
    FFTConfig fft(spec);
    if (fft.isPfa()) return std::nullopt;  // Pass-3 PFA9 candidates produced WORD MISMATCH.
    if (fft.shape.fft_type != FFT3161 && fft.shape.fft_type != FFT323161) return std::nullopt;
    const double bpw = exponent / double(fft.size());
    if (bpw < fft.minBpw()) return std::nullopt;
    if (fft.maxExp() * args.fftOverdrive < exponent) return std::nullopt;
    return fft;
  } catch (...) {
    return std::nullopt;
  }
}

std::vector<std::string> autotuneCandidates(const Args& args,
                                            uint32_t exponent,
                                            aevum_autotune::Workload workload,
                                            const FFTConfig& native,
                                            unsigned cap) {
  std::vector<std::string> candidates;
  std::unordered_set<std::string> seen{native.spec()};
  auto add = [&](const std::string& spec) {
    if (candidates.size() >= cap || seen.count(spec)) return;
    const auto fft = admissiblePlan(args, exponent, spec);
    if (!fft) return;
    const std::string normalized = fft->spec();
    if (seen.insert(normalized).second) candidates.push_back(normalized);
  };

  const bool prp_like = workload == aevum_autotune::Workload::Prp;
  // Measured Pass-3/Range-Sweep seeds. They are seeds, never unconditional
  // choices: every runtime candidate must pass exact differential + timing.
  if (prp_like && exponent >= 80000000u && exponent < 170000000u) {
    add("1:512:8:512:101");
    add("1:512:8:512:202");
  }
  if (exponent >= 170000000u && exponent <= 197999999u) {
    // Explicitly probe the transform-size boundary.  This is the large measured
    // opportunity on both target GPU families, but other workloads still time it
    // rather than inheriting the PRP decision.
    add("4:512:8:512:202");
    if (prp_like) add("4:1K:8:256:101");
  }
  if (prp_like && exponent >= 198000000u && exponent <= 230000000u) {
    add("1:512:16:512:202");
    add("1:1K:8:512:202");
  }

  // Same shape, opposite proven stock variant is the cheapest serious neighbor.
  if (native.variant == 101 || native.variant == 202) {
    FFTConfig alternate(native.shape, native.variant == 101 ? 202 : 101, CARRY_AUTO);
    add(alternate.spec());
  }

  // Add only a tiny nearest-neighbor set around native AUTO.  Enumeration is
  // CPU-only and used to rank candidates; actual GPU benchmarks remain capped.
  struct Ranked { double score; std::string spec; };
  std::vector<Ranked> ranked;
  for (const FFTShape& shape : FFTShape::allShapes()) {
    for (u32 variant : {101u, 202u}) {
      FFTConfig fft(shape, variant, CARRY_AUTO);
      if (!admissiblePlan(args, exponent, fft.spec())) continue;
      if (seen.count(fft.spec())) continue;
      const double ratio = double(fft.size()) / double(native.size());
      if (ratio < 0.55 || ratio > 1.35) continue;
      const double geometry = (shape.width == native.shape.width ? 0.0 : 0.08) +
                              (shape.height == native.shape.height ? 0.0 : 0.05) +
                              (shape.middle == native.shape.middle ? 0.0 : 0.03);
      ranked.push_back({std::abs(std::log(ratio)) + geometry, fft.spec()});
    }
  }
  std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
    if (a.score != b.score) return a.score < b.score;
    return a.spec < b.spec;
  });
  for (const auto& r : ranked) add(r.spec);
  return candidates;
}

Words deterministicResidue(uint32_t exponent, uint32_t seed) {
  Words words(nWords(exponent));
  uint32_t x = seed;
  for (auto& w : words) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    w = x;
  }
  if (exponent % 32) words.back() &= (uint32_t(1) << (exponent % 32)) - 1;
  return words;
}

void runPlanSequence(Gpu& gpu,
                     aevum_autotune::Workload workload,
                     Buffer<Word>& value,
                     Buffer<double>* prepared,
                     unsigned units) {
  const bool lead = gpu.regSupportsLeadCache();
  if (workload == aevum_autotune::Workload::Ll) {
    for (unsigned i = 0; i < units; ++i) {
      if (gpu.regSupportsFusedLL()) gpu.regSquareStep(value, i != 0, i + 1 != units, true);
      else { gpu.regSquare(value); gpu.regSubU32(value, 2); }
    }
    return;
  }
  if (workload == aevum_autotune::Workload::Pm1 ||
      workload == aevum_autotune::Workload::Pm1Lowmem ||
      workload == aevum_autotune::Workload::Pm1Ultralowmem) {
    if (!prepared) throw std::runtime_error("autotune PM1 prepared buffer missing");
    for (unsigned i = 0; i < units; ++i) {
      if (lead) {
        gpu.regSquareStep(value, false, true);
        gpu.regSquareStep(value, true, false);
      } else {
        gpu.regSquare(value); gpu.regSquare(value);
      }
      gpu.regMulPrepared(value, *prepared, 1);
    }
    return;
  }
  if (workload == aevum_autotune::Workload::Ecm) {
    if (!prepared) throw std::runtime_error("autotune ECM prepared buffer missing");
    for (unsigned i = 0; i < units; ++i) {
      gpu.regSquare(value);
      gpu.regMulPrepared(value, *prepared, 1);
    }
    return;
  }
  // PRP/generic: long square-heavy retained-width chain when supported.
  for (unsigned i = 0; i < units; ++i) {
    if (lead) gpu.regSquareStep(value, i != 0, i + 1 != units, false);
    else gpu.regSquare(value);
  }
}

struct PlanComparison {
  bool exact = false;
  double native_seconds = 0.0;
  double candidate_seconds = 0.0;
  double speedup = 0.0;
};

PlanComparison comparePlans(uint32_t exponent,
                            aevum_autotune::Workload workload,
                            GpuCommon shared,
                            const FFTConfig& native_fft,
                            const FFTConfig& candidate_fft) {
  auto native_gpu = Gpu::make(exponent, shared, native_fft, {}, false);
  auto candidate_gpu = Gpu::make(exponent, shared, candidate_fft, {}, false);
  auto native_regs = native_gpu->makeBufVector(2);
  auto candidate_regs = candidate_gpu->makeBufVector(2);
  auto native_prepared = native_gpu->makeTransformBufVector(1);
  auto candidate_prepared = candidate_gpu->makeTransformBufVector(1);
  const Words value_seed = deterministicResidue(exponent, 0x12345678u);
  const Words mul_seed = deterministicResidue(exponent, 0x9e3779b9u);

  auto reset = [&](Gpu& gpu, std::vector<Buffer<Word>>& regs, std::vector<Buffer<double>>& prepared) {
    gpu.regWrite(regs[0], value_seed);
    gpu.regWrite(regs[1], mul_seed);
    gpu.regPrepare(prepared[0], regs[1]);
    gpu.regSync();
  };
  const bool needs_prepared = workload == aevum_autotune::Workload::Pm1 ||
      workload == aevum_autotune::Workload::Pm1Lowmem ||
      workload == aevum_autotune::Workload::Pm1Ultralowmem ||
      workload == aevum_autotune::Workload::Ecm;

  reset(*native_gpu, native_regs, native_prepared);
  reset(*candidate_gpu, candidate_regs, candidate_prepared);
  runPlanSequence(*native_gpu, workload, native_regs[0], needs_prepared ? &native_prepared[0] : nullptr, 5);
  runPlanSequence(*candidate_gpu, workload, candidate_regs[0], needs_prepared ? &candidate_prepared[0] : nullptr, 5);
  native_gpu->regSync(); candidate_gpu->regSync();
  const Words reference = native_gpu->regRead(native_regs[0]);
  const Words candidate = candidate_gpu->regRead(candidate_regs[0]);
  if (reference != candidate) return {};

  const unsigned units = exponent <= 30000000u ? 64u : exponent <= 120000000u ? 32u :
                         exponent <= 180000000u ? 20u : 12u;
  auto timed = [&](Gpu& gpu, std::vector<Buffer<Word>>& regs, std::vector<Buffer<double>>& prepared) {
    reset(gpu, regs, prepared);
    const auto start = std::chrono::steady_clock::now();
    runPlanSequence(gpu, workload, regs[0], needs_prepared ? &prepared[0] : nullptr, units);
    gpu.regSync();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  // Warmup is deliberately outside timing.
  reset(*native_gpu, native_regs, native_prepared);
  runPlanSequence(*native_gpu, workload, native_regs[0], needs_prepared ? &native_prepared[0] : nullptr, 3);
  native_gpu->regSync();
  reset(*candidate_gpu, candidate_regs, candidate_prepared);
  runPlanSequence(*candidate_gpu, workload, candidate_regs[0], needs_prepared ? &candidate_prepared[0] : nullptr, 3);
  candidate_gpu->regSync();

  std::array<double,3> nt{}, ct{};
  for (unsigned rep = 0; rep < 3; ++rep) {
    if ((rep & 1u) == 0) {
      nt[rep] = timed(*native_gpu, native_regs, native_prepared);
      ct[rep] = timed(*candidate_gpu, candidate_regs, candidate_prepared);
    } else {
      ct[rep] = timed(*candidate_gpu, candidate_regs, candidate_prepared);
      nt[rep] = timed(*native_gpu, native_regs, native_prepared);
    }
  }
  PlanComparison out;
  out.exact = true;
  out.native_seconds = median3(nt);
  out.candidate_seconds = median3(ct);
  if (out.candidate_seconds > 0.0) out.speedup = out.native_seconds / out.candidate_seconds;
  return out;
}

struct BridgeComparison {
  bool exact = false;
  double canonical_seconds = 0.0;
  double bridge_seconds = 0.0;
  double speedup = 0.0;
};

BridgeComparison comparePreparedMulLead(Gpu& gpu, uint32_t exponent) {
  if (!gpu.regSupportsPreparedMulLead()) return {};
  auto regs = gpu.makeBufVector(2);
  auto prepared = gpu.makeTransformBufVector(1);
  const Words value_seed = deterministicResidue(exponent, 0x243f6a88u);
  const Words mul_seed = deterministicResidue(exponent, 0xb7e15162u);

  auto reset = [&] {
    gpu.regWrite(regs[0], value_seed);
    gpu.regWrite(regs[1], mul_seed);
    gpu.regPrepare(prepared[0], regs[1]);
    gpu.regSync();
  };
  auto canonical = [&](unsigned pairs) {
    for (unsigned i = 0; i < pairs; ++i) {
      gpu.regSquare(regs[0]);
      gpu.regMulPrepared(regs[0], prepared[0], 1);
    }
  };
  auto bridged = [&](unsigned pairs) {
    const unsigned operations = pairs * 2;
    for (unsigned op = 0; op < operations; ++op) {
      const bool lead_in = op != 0;
      const bool lead_out = op + 1 != operations;
      if ((op & 1u) == 0) gpu.regSquareStep(regs[0], lead_in, lead_out, false);
      else gpu.regMulPreparedStep(regs[0], prepared[0], lead_in, lead_out);
    }
  };

  reset(); canonical(4); gpu.regSync();
  const Words reference = gpu.regRead(regs[0]);
  reset(); bridged(4); gpu.regSync();
  const Words candidate = gpu.regRead(regs[0]);
  if (reference != candidate) return {};

  const unsigned pairs = exponent <= 30000000u ? 32u : exponent <= 120000000u ? 16u : 8u;
  auto timed = [&](bool use_bridge) {
    reset();
    const auto start = std::chrono::steady_clock::now();
    if (use_bridge) bridged(pairs); else canonical(pairs);
    gpu.regSync();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  // Compile/warm all kernels before measurement.
  reset(); canonical(2); gpu.regSync();
  reset(); bridged(2); gpu.regSync();

  std::array<double,3> base{}, opt{};
  for (unsigned rep = 0; rep < 3; ++rep) {
    if ((rep & 1u) == 0) {
      base[rep] = timed(false); opt[rep] = timed(true);
    } else {
      opt[rep] = timed(true); base[rep] = timed(false);
    }
  }
  BridgeComparison out;
  out.exact = true;
  out.canonical_seconds = median3(base);
  out.bridge_seconds = median3(opt);
  if (out.bridge_seconds > 0.0) out.speedup = out.canonical_seconds / out.bridge_seconds;
  return out;
}

class Runtime {
public:
  Runtime(uint32_t exponent, size_t register_count, uint32_t device, bool verbose, const char* fft_spec, const char* tune_dir, uint32_t workload)
      : exponent_(exponent),
        register_count_(register_count),
        workload_(workload <= static_cast<uint32_t>(aevum_autotune::Workload::Ecm)
                    ? static_cast<aevum_autotune::Workload>(workload)
                    : aevum_autotune::Workload::Generic),
        word_count_(nWords(exponent)),
        args_(true),
        context_(nullptr),
        cache_(nullptr) {
    if (exponent_ < 3) throw std::runtime_error("Aevum exponent must be at least 3");
    if (register_count_ == 0) throw std::runtime_error("Aevum register count must be positive");

    args_.device = static_cast<int>(device);
    args_.verbose = verbose;
    args_.safeMath = true;
    // Rejected on both target GPUs at p=21000029. Never route to carryM.
    args_.clean = true;
    args_.useCache = true;
    args_.cacheDir = std::filesystem::absolute(".aevum-kernel-cache");
    args_.proofResultDir = std::filesystem::absolute(".aevum-proof");
    args_.proofToVerifyDir = std::filesystem::absolute(".aevum-proof-tmp");
    if (tune_dir && *tune_dir) args_.masterDir = std::filesystem::absolute(tune_dir);
    args_.setDefaults();
    args_.profile = std::getenv("AEVUM_PROFILE_KERNELS") && std::strcmp(std::getenv("AEVUM_PROFILE_KERNELS"), "1") == 0;

    // Device-scoped tuning must use the actual OpenCL device selected by -d.
    cl_device_id selected_device = getDevice(device);
    const std::string device_name = openclDeviceName(selected_device);
    const std::string device_vendor = openclVendorId(selected_device);
    const std::string device_driver = openclDeviceInfoString(selected_device, CL_DRIVER_VERSION);
    const std::string device_runtime = openclDeviceInfoString(selected_device, CL_DEVICE_VERSION);
    if (verbose) {
      log("AEVUM_DEVICE vendor_id='%s' name='%s' driver='%s' runtime='%s' device=%u\n",
          device_vendor.c_str(), device_name.c_str(), device_driver.c_str(),
          device_runtime.c_str(), static_cast<unsigned>(device));
    }

    const char* radix1k_env = std::getenv("AEVUM_RADIX1K");
    if (radix1k_env && *radix1k_env &&
        std::strcmp(radix1k_env, "4") != 0 &&
        std::strcmp(radix1k_env, "8") != 0) {
      throw std::runtime_error("AEVUM_RADIX1K must be exactly 4 or 8");
    }

    context_ = std::make_unique<Context>(selected_device);
    cache_ = std::make_unique<TrigBufCache>(context_.get());

    shared_.context = context_.get();
    shared_.args = &args_;
    shared_.bufCache = cache_.get();
    shared_.background = &background_;

    std::string spec = fft_spec ? fft_spec : "";
    const bool explicit_fft_spec = !spec.empty();
    const auto autotune_mode = aevum_autotune::modeFromEnvironment();
    const auto autotune_cache_path = aevum_autotune::cachePath();
    bool autotune_cache_miss = false;
    bool autotune_cache_hit = false;
    bool autotune_runtime_ran = false;
    double autotune_plan_speedup = 1.0;
    double autotune_impl_speedup = 1.0;
    uint64_t autotune_elapsed_ms = 0;
    std::string autotune_key;
    std::optional<aevum_autotune::Record> cached_record;

    // Issue #36 / GB202 measured FFT-shape profile.
    // Preserve the validated built-in profile ahead of the generic runtime tuner.
    // Historical issue #36 notes listed LOADS/STORES/TABMUL_CHAIN32/MODM31/ZEROHACK_W;
    // those knobs are not present here and are deliberately not synthesized.
    const bool gb202_forced = envExactly("AEVUM_GB202_TUNE", "force");
    const bool gb202_disabled = envExactly("AEVUM_GB202_TUNE", "0");
    const char* tune_env = std::getenv("AEVUM_TUNE_DIR");
    const bool user_tune_dir = tune_env && *tune_env;
    const bool gb202_range = exponent_ >= 146000000u && exponent_ <= 150000000u;
    const bool gb202_profile =
        spec.empty() && !gb202_disabled && !user_tune_dir && gb202_range &&
        (gb202_forced || gb202DeviceName(device_name));

    if (gb202_profile) {
      spec = "1:512:8:512:202";
      if (verbose) {
        log("Aevum GB202 native tune: device='%s', exponent=%u, "
            "FFT=1:512:8:512:202%s.\n",
            device_name.c_str(), exponent_,
            gb202_forced ? " (forced validation)" : "");
      }
    }

    // Type4 plan candidates need the same production queue policy as the final
    // engine.  Manual AEVUM_TYPE4_MULTI_Q is a higher-precedence override and
    // causes the runtime tuner to be bypassed below.
    int multi_q_default = 1;
    if (const char* value = std::getenv("AEVUM_TYPE4_MULTI_Q")) multi_q_default = std::atoi(value) != 0;
    if (multi_q_default) args_.flags["MULTI_Q"] = "1";

    const bool manual_plan_env = aevum_autotune::hasManualPlanOverrideEnvironment();
    const bool compatible_tune_entry = spec.empty() && usableTuneEntry(args_, exponent_);
    const bool workload_tunable = workload_ != aevum_autotune::Workload::Pm1Ultralowmem;
    const bool autotune_eligible = autotune_mode != aevum_autotune::Mode::Off &&
        !explicit_fft_spec && !gb202_profile && !manual_plan_env &&
        !compatible_tune_entry && workload_tunable;

    FFTConfig native_fft = FFTConfig::bestFit(args_, exponent_, spec);
    std::string selected_spec = native_fft.spec();

    if (autotune_eligible) {
      const std::string& vendor = device_vendor;
      const std::string& driver = device_driver;
      const std::string& runtime = device_runtime;
      std::ostringstream flags;
      flags << "radix1k=" << (aevumRadix8For1K() ? 8 : 4)
            << ";multiq=" << multi_q_default
            << ";safe=1;clean=1";
      if (const char* ll = std::getenv("AEVUM_FUSED_LL")) flags << ";fusedll=" << ll;
      if (const char* bridge = std::getenv("AEVUM_PREPARED_MUL_LEAD")) flags << ";prep-lead=" << bridge;
      autotune_key = aevum_autotune::makeKey(VERSION, vendor, device_name, driver, runtime,
                                              workload_, register_count_, exponent_, flags.str());

      if (autotune_mode == aevum_autotune::Mode::Auto) {
        cached_record = aevum_autotune::load(autotune_cache_path, autotune_key);
        if (cached_record) {
          const auto cached_fft = admissiblePlan(args_, exponent_, cached_record->plan);
          if (cached_fft) {
            selected_spec = cached_fft->spec();
            autotune_plan_speedup = cached_record->plan_speedup;
            autotune_impl_speedup = cached_record->implementation_speedup;
            autotune_elapsed_ms = cached_record->tune_ms;
            autotune_cache_hit = true;
            if (verbose) {
              log("Aevum autotune: cache hit workload=%s plan=%s prep-lead=%d cached-plan-advantage=%.4fx cached-impl-advantage=%.4fx prior-tune=%" PRIu64 "ms.\n",
                  aevum_autotune::workloadClass(workload_, register_count_).c_str(),
                  selected_spec.c_str(), cached_record->prepared_mul_lead ? 1 : 0,
                  cached_record->plan_speedup, cached_record->implementation_speedup,
                  cached_record->tune_ms);
            }
          } else if (verbose) {
            log("Aevum autotune: stale/invalid cache plan ignored; safe retune.\n");
          }
        }
      }

      if (!autotune_cache_hit) {
        autotune_cache_miss = true;
        autotune_runtime_ran = true;
        const auto tune_start = std::chrono::steady_clock::now();
        const unsigned candidate_cap = boundedEnvUnsigned("AEVUM_AUTOTUNE_MAX_CANDIDATES", 5, 2, 10);
        const unsigned budget_ms = boundedEnvUnsigned("AEVUM_AUTOTUNE_BUDGET_MS", 7000, 1000, 30000);
        const double threshold = positiveEnvDouble("AEVUM_AUTOTUNE_MIN_GAIN", 1.04, 1.01, 1.20);
        const auto candidates = autotuneCandidates(args_, exponent_, workload_, native_fft, candidate_cap);
        double best_speedup = 1.0;
        std::string best_spec = native_fft.spec();
        unsigned tested = 0;

        if (verbose) {
          const uint32_t band = aevum_autotune::exponentBandStart(exponent_);
          log("Aevum autotune: cache %s workload=%s band=%u-%u native=%s candidates<=%u budget=%ums.\n",
              autotune_mode == aevum_autotune::Mode::Retune ? "bypass(retune)" : "miss",
              aevum_autotune::workloadClass(workload_, register_count_).c_str(),
              band, band + 9999999u, native_fft.spec().c_str(), candidate_cap, budget_ms);
        }
        for (const std::string& candidate_spec : candidates) {
          const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - tune_start).count();
          if (elapsed >= budget_ms || tested >= candidate_cap) break;
          const auto candidate_fft = admissiblePlan(args_, exponent_, candidate_spec);
          if (!candidate_fft) continue;
          ++tested;
          try {
            const PlanComparison cmp = comparePlans(exponent_, workload_, shared_, native_fft, *candidate_fft);
            if (!cmp.exact) {
              if (verbose) log("Aevum autotune: reject %s (WORD MISMATCH).\n", candidate_spec.c_str());
              continue;
            }
            if (verbose) {
              log("Aevum autotune: candidate=%s median=%.6fs native=%.6fs speedup=%.4fx.\n",
                  candidate_spec.c_str(), cmp.candidate_seconds, cmp.native_seconds, cmp.speedup);
            }
            if (cmp.speedup > best_speedup) {
              best_speedup = cmp.speedup;
              best_spec = candidate_fft->spec();
            }
          } catch (const std::exception& e) {
            if (verbose) log("Aevum autotune: candidate=%s rejected (%s).\n", candidate_spec.c_str(), e.what());
          } catch (...) {
            if (verbose) log("Aevum autotune: candidate=%s rejected (engine exception).\n", candidate_spec.c_str());
          }
        }
        if (best_speedup >= threshold) {
          selected_spec = best_spec;
          autotune_plan_speedup = best_speedup;
        } else {
          selected_spec = native_fft.spec();
          autotune_plan_speedup = 1.0;
        }
        autotune_elapsed_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tune_start).count());
        if (verbose) {
          log("Aevum autotune: selected=%s advantage=%.4fx tested=%u tune=%" PRIu64 "ms.\n",
              selected_spec.c_str(), autotune_plan_speedup, tested, autotune_elapsed_ms);
        }
      }
    } else if (verbose && autotune_mode != aevum_autotune::Mode::Off) {
      const char* reason = explicit_fft_spec ? "explicit FFT plan" :
          gb202_profile ? "validated GB202 profile" :
          manual_plan_env ? "manual tuning/environment override" :
          compatible_tune_entry ? "compatible tune.txt entry" :
          !workload_tunable ? "ultra-low-memory workload" : "policy";
      log("Aevum autotune: bypassed (%s); manual/source precedence preserved.\n", reason);
    } else if (verbose && autotune_mode == aevum_autotune::Mode::Off) {
      log("Aevum autotune: OFF; native/manual selection unchanged.\n");
    }

    FFTConfig fft = admissiblePlan(args_, exponent_, selected_spec).value_or(native_fft);

    if (verbose) {
      const bool uses_1k = fft.shape.width == 1024 || fft.shape.height == 1024;
      if (uses_1k) {
        log("Aevum 1K radix policy: radix-%u (%s).\n",
            aevumRadix8For1K() ? 8u : 4u,
            aevumRadix8For1K()
              ? "explicit AEVUM_RADIX1K=8 override"
              : "safe default; set AEVUM_RADIX1K=8 only for a measured tune");
      } else {
        log("Aevum 1K radix policy: not used by selected shape; default remains radix-4.\n");
      }
    }

    // Full FFT323161 has three independent residue planes.  Overlap the
    // GF61 queue with the FP32+GF31 queue by default; the planes touch
    // disjoint transform ranges and are synchronized before carry.  This is
    // useful for both the stock power-of-two plan and the PFA9 diagnostic.
    if (fft.shape.fft_type == FFT323161) {
      int multi_q = 1;
      if (const char* value = std::getenv("AEVUM_TYPE4_MULTI_Q"))
        multi_q = std::atoi(value) != 0;
      if (multi_q) {
        args_.flags["MULTI_Q"] = "1";
        if (verbose) log("Aevum optimized full type-4 PFA9: concurrent GF61 and FP32+GF31 queues enabled.\n");
      } else if (verbose) {
        log("Aevum full type-4: multi-queue overlap disabled by AEVUM_TYPE4_MULTI_Q=0.\n");
      }
    }

    bool apple_diagnostic_plane = false;
#if defined(__APPLE__)
    if (const char* diagnostic = std::getenv("AEVUM_APPLE_DIAGNOSTIC_PLANES")) {
      apple_diagnostic_plane = diagnostic[0] == '1' && diagnostic[1] == '\0' &&
                               (fft.shape.fft_type == FFT31 || fft.shape.fft_type == FFT61);
    }
#endif
    const bool supported_aevum_type = fft.shape.fft_type == FFT3161 ||
                                      fft.shape.fft_type == FFT323161;
    if (!supported_aevum_type && !apple_diagnostic_plane)
      throw std::runtime_error("Aevum engine requires FFT3161 or FFT323161");
#if defined(__APPLE__)
    if (verbose && apple_diagnostic_plane) {
      log("Apple Aevum plane-isolation diagnostic: using FFT type %d (%s).\n",
          static_cast<int>(fft.shape.fft_type), fft.shape.fft_type == FFT31 ? "GF31 only" : "GF61 only");
    }
#endif

    std::vector<KeyVal> pfa_use;
    if (fft.isPfa()) {
      if (const char* text = std::getenv("AEVUM_PFA_USE")) {
        std::string uses(text);
        size_t pos = 0;
        while (pos < uses.size()) {
          const size_t comma = uses.find(',', pos);
          const std::string item = uses.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
          const size_t equal = item.find('=');
          if (equal == std::string::npos || equal == 0 || equal + 1 == item.size())
            throw std::runtime_error("AEVUM_PFA_USE expects KEY=VALUE comma-separated entries");
          pfa_use.emplace_back(item.substr(0, equal), item.substr(equal + 1));
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
        if (verbose && !pfa_use.empty()) log("Aevum PFA validated tune override: %s\n", text);
      }
    }
#if !defined(__APPLE__)
    if (const char* value = std::getenv("AEVUM_GF61_LIMB32")) {
      if (std::strcmp(value,"0") && std::strcmp(value,"1"))
        throw std::runtime_error("AEVUM_GF61_LIMB32 must be 0 or 1");
      pfa_use.emplace_back("AEVUM_GF61_LIMB32",value);
    }
    if (const char* value = std::getenv("AEVUM_CARRY_WMUL")) {
      if (std::strcmp(value,"1") && std::strcmp(value,"2") && std::strcmp(value,"4"))
        throw std::runtime_error("AEVUM_CARRY_WMUL must be 1, 2 or 4");
      pfa_use.emplace_back("WMUL",value);
    }
#endif
    // PRP-only runtime dispatch; not a global compiler option and never used by LL.
    if (const char* value = std::getenv("AEVUM_PRP_MIDDLE1")) {
      if (std::strcmp(value,"0") && std::strcmp(value,"1"))
        throw std::runtime_error("AEVUM_PRP_MIDDLE1 must be 0 or 1");
      args_.flags["PRP_MIDDLE1"] = value;
    }
    gpu_ = Gpu::make(exponent_, shared_, fft, pfa_use, verbose);
    transform_size_ = gpu_->getFFTSize();

    const char* ll_env = std::getenv("AEVUM_FUSED_LL");
    const bool validated_ll_device = device_name.find("RTX 3080") != std::string::npos ||
        device_name.find("gfx906") != std::string::npos || device_name.find("Radeon VII") != std::string::npos;
    fused_ll_enabled_ = gpu_->regSupportsFusedLL() &&
        (ll_env ? std::strcmp(ll_env, "1") == 0 : validated_ll_device);
    lead_cache_enabled_ = gpu_->regSupportsLeadCache();
    // The power-of-two lead cache is validated and enabled by default.  The
    // PFA9 bridge changes the carry/pack boundary and therefore remains an
    // explicit experiment until the word-exact GPU differential and an A/B
    // throughput run have passed on the target device.
    if (fft.isPfa()) {
      const char* value = std::getenv("AEVUM_PFA_LEAD_BRIDGE");
      lead_cache_enabled_ = lead_cache_enabled_ && value && std::atoi(value) != 0;
    }
    if (const char* value = std::getenv("AEVUM_REG_LEAD_CACHE"))
      lead_cache_enabled_ = lead_cache_enabled_ && std::atoi(value) != 0;
    if (verbose) {
      if (lead_cache_enabled_) {
        if (fft.isPfa())
          log("Aevum PFA9 lead bridge enabled: consecutive square_mul(reg,1) calls fuse carryB with the next fftP gather.\n");
        else
          log("Aevum register lead cache enabled: consecutive square_mul(reg,1) calls retain the width transform and use carryFused.\n");
      } else if (!fft.isPfa()) {
        log("Aevum register lead cache disabled; set AEVUM_REG_LEAD_CACHE=1 only on a supported non-Apple, short-carry plan.\n");
      }
    }

    // Pass-4 shared engine implementation gate.  Explicit 0/1 always wins.
    // Otherwise a cache hit is benchmark-free; a cache miss may run one exact
    // differential + tiny A/B only for workloads that actually use prepared
    // multiplication (P-1/ECM, including Gaussian variants through the same API).
    const char* bridge_env = std::getenv("AEVUM_PREPARED_MUL_LEAD");
    bool bridge_manual = false;
    bool bridge_manual_value = false;
    if (bridge_env && *bridge_env && std::strcmp(bridge_env, "auto") != 0) {
      if (std::strcmp(bridge_env, "0") != 0 && std::strcmp(bridge_env, "1") != 0)
        throw std::runtime_error("AEVUM_PREPARED_MUL_LEAD must be 0, 1, or auto");
      bridge_manual = true;
      bridge_manual_value = std::strcmp(bridge_env, "1") == 0;
    }
    const bool bridge_supported = lead_cache_enabled_ && gpu_->regSupportsPreparedMulLead();
    if (bridge_manual) {
      if (bridge_manual_value && !bridge_supported)
        throw std::runtime_error("AEVUM_PREPARED_MUL_LEAD=1 requires a non-PFA short-carry plan with lead cache enabled");
      prepared_mul_lead_enabled_ = bridge_manual_value && bridge_supported;
      if (verbose) log("Aevum prepared-multiply lead bridge: manual override=%d.\n", prepared_mul_lead_enabled_ ? 1 : 0);
    } else if (autotune_cache_hit && cached_record) {
      prepared_mul_lead_enabled_ = cached_record->prepared_mul_lead && bridge_supported;
    } else if (autotune_runtime_ran && bridge_supported &&
               (workload_ == aevum_autotune::Workload::Pm1 ||
                workload_ == aevum_autotune::Workload::Pm1Lowmem ||
                workload_ == aevum_autotune::Workload::Ecm)) {
      const auto impl_start = std::chrono::steady_clock::now();
      try {
        const BridgeComparison cmp = comparePreparedMulLead(*gpu_, exponent_);
        if (!cmp.exact) {
          prepared_mul_lead_enabled_ = false;
          if (verbose) log("Aevum prepared-multiply lead bridge: WORD MISMATCH, auto-reverted.\n");
        } else {
          const double threshold = positiveEnvDouble("AEVUM_AUTOTUNE_IMPL_MIN_GAIN", 1.03, 1.01, 1.20);
          autotune_impl_speedup = cmp.speedup;
          prepared_mul_lead_enabled_ = cmp.speedup >= threshold;
          if (verbose) {
            log("Aevum prepared-multiply lead bridge: exact=yes canonical=%.6fs bridge=%.6fs speedup=%.4fx selected=%d.\n",
                cmp.canonical_seconds, cmp.bridge_seconds, cmp.speedup,
                prepared_mul_lead_enabled_ ? 1 : 0);
          }
        }
      } catch (const std::exception& e) {
        prepared_mul_lead_enabled_ = false;
        if (verbose) log("Aevum prepared-multiply lead bridge: auto-reverted (%s).\n", e.what());
      } catch (...) {
        prepared_mul_lead_enabled_ = false;
        if (verbose) log("Aevum prepared-multiply lead bridge: auto-reverted (engine exception).\n");
      }
      autotune_elapsed_ms += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - impl_start).count());
    }

    if (autotune_cache_miss && !autotune_key.empty()) {
      aevum_autotune::Record record;
      record.key = autotune_key;
      record.plan = fft.spec();
      record.prepared_mul_lead = prepared_mul_lead_enabled_;
      record.plan_speedup = autotune_plan_speedup;
      record.implementation_speedup = autotune_impl_speedup;
      record.tune_ms = autotune_elapsed_ms;
      record.created_unix = static_cast<uint64_t>(std::time(nullptr));
      try {
        aevum_autotune::storeAtomic(autotune_cache_path, record);
        if (verbose) log("Aevum autotune: persistent cache updated at %s.\n", autotune_cache_path.string().c_str());
      } catch (const std::exception& e) {
        if (verbose) log("Aevum autotune: cache write skipped (%s); engine continues safely.\n", e.what());
      }
    }
    if (verbose && bridge_supported && !prepared_mul_lead_enabled_) {
      log("Aevum prepared-multiply lead bridge: OFF (canonical multiply boundary retained).\n");
    } else if (verbose && prepared_mul_lead_enabled_) {
      log("Aevum prepared-multiply lead bridge: ON (carryFused retained-width shared arithmetic path).\n");
    }

    buffers_ = gpu_->makeBufVector(static_cast<u32>(register_count_));
    size_t prepared_count = std::min<size_t>(register_count_, 2);
    if (const char* value = std::getenv("AEVUM_PREPARED_CACHE")) {
      char* end = nullptr;
      const unsigned long requested = std::strtoul(value, &end, 10);
      if (end != value && *end == '\0') prepared_count = std::min<size_t>(register_count_, std::min<unsigned long>(requested, 32));
    }
    prepared_buffers_ = gpu_->makeTransformBufVector(static_cast<u32>(prepared_count));
    prepared_slots_.resize(prepared_count);
#if defined(__APPLE__)
    // Apple cl2Metal may honor the restrict contract of regAdd and miscompile
    // an in-place dst==src doubling.  A second tiny register buffer lets the
    // exact same add-and-double algorithm ping-pong without aliasing.
    small_factor_scratch_ = gpu_->makeBufVector(2);
#else
    small_factor_scratch_ = gpu_->makeBufVector(1);
#endif
#if defined(__APPLE__)
    if (verbose && fft.shape.fft_type == FFT3161) {
      log("Apple Aevum compatibility: register upload/readback uses direct global transpose kernels without LDS.\n");
      log("Apple Aevum compatibility: set_u32 uses canonical compact-word upload; raw register fill is disabled.\n");
      log("Apple Aevum compatibility: FFT3161 remains GF31+GF61; fftP uses fully global scalar/radix/twiddle staging for both CRT planes.\n");
      log("Apple Aevum compatibility: GF61 middle-in in-place stages use single-pointer no-alias dispatch.\n");
      log("Apple Aevum compatibility: fftHinGF61 uses exact global staging; non-Apple keeps the upstream monolithic kernel.\n");
      log("Apple Aevum compatibility: prepared tailMulLowGF61 uses exact staged height FFT/pairing; prepared operand remains read-only.\n");
      log("Apple Aevum performance: queue pacing uses nonblocking clFlush; marker polling is disabled by default.\n");
      log("Apple Aevum diagnostic: set AEVUM_APPLE_QUEUE_MARKER_WAIT=1 to restore legacy marker polling.\n");
      log("Apple Aevum diagnostic: set AEVUM_APPLE_STAGE_FINISH=1 to serialize every staged GF61 kernel.\n");
    }
#endif
    gpu_->regSync();
  }

  ~Runtime() noexcept {
    try {
      flush_pending_square();
      if (gpu_) gpu_->regSync();
    } catch (...) {
      // Destruction is best-effort; all explicit API calls report failures.
    }
  }

  size_t transform_size() const { return transform_size_; }
  size_t word_count() const { return word_count_; }

  void sync() {
    flush_pending_square();
    gpu_->regSync();
  }

  void profile_report(bool emit) {
    sync();
    gpu_->regProfileReport(emit);
  }

  void set_u32(size_t dst, uint32_t value) {
    check_reg(dst);
    flush_pending_square();
    invalidate_if(dst);
    gpu_->regSetU32(reg(dst), value);
  }

  void set_words(size_t dst, const uint32_t* words, size_t count) {
    check_reg(dst);
    flush_pending_square();
    invalidate_if(dst);
    if (!words || count != word_count_) throw std::runtime_error("invalid Aevum word buffer");
    Words v(words, words + count);
    if (exponent_ % 32) v.back() &= (uint32_t(1) << (exponent_ % 32)) - 1;
    gpu_->regWrite(reg(dst), v);
  }

  void get_words(size_t src, uint32_t* words, size_t count) {
    check_reg(src);
    flush_pending_square();
    if (!words || count != word_count_) throw std::runtime_error("invalid Aevum output word buffer");
    Words v = gpu_->regRead(reg(src));
    if (v.empty()) v.assign(word_count_, 0);
    if (v.size() != word_count_) throw std::runtime_error("unexpected Aevum residue size");
    std::copy(v.begin(), v.end(), words);
  }

  void copy(size_t dst, size_t src) {
    check_reg(dst);
    check_reg(src);
    flush_pending_square();
    if (dst != src) {
      invalidate_if(dst);
      gpu_->regCopy(reg(dst), reg(src));
    }
  }

  void prepare(size_t dst, size_t src) {
    check_reg(dst);
    check_reg(src);
    flush_pending_square();
    if (dst != src) {
      invalidate_if(dst);
      gpu_->regCopy(reg(dst), reg(src));
    }
    if (prepared_buffers_.empty()) return;
    const size_t slot = acquire_prepared(dst);
    gpu_->regPrepare(prepared_buffers_[slot], reg(dst));
  }

  void square_mul(size_t index, uint32_t factor) {
    check_reg(index);
    if (factor == 0) throw std::runtime_error("Aevum square factor must be positive");

    // Keep one logical arithmetic operation pending.  The next compatible
    // operation executes the previous one with leadOut=WIDTH, so the hot
    // square/prepared-multiply chain stays transformed.  Every observable
    // API boundary flushes the final pending operation to canonical words.
    if (lead_cache_enabled_ && factor == 1) {
      if (pending_kind_ != PendingKind::None && pending_reg_ != index) flush_pending_square();
      invalidate_if(index);
      if (pending_kind_ != PendingKind::None) {
        execute_pending(true);
        pending_lead_width_ = true;
      } else {
        pending_lead_width_ = false;
      }
      pending_kind_ = PendingKind::Square;
      pending_reg_ = index;
      return;
    }

    flush_pending_square();
    invalidate_if(index);
    gpu_->regSquare(reg(index), 1);
    multiply_small(index, factor);
  }

  void mul(size_t dst, size_t src, uint32_t factor) {
    check_reg(dst);
    check_reg(src);
    if (factor == 0) throw std::runtime_error("Aevum multiplication factor must be positive");

#if !defined(__APPLE__)
    // Shared Pass-4 engine optimization.  A prepared multiply can consume
    // and/or produce the same retained-width state used by the square chain.
    // This is gated independently and never applies to LL's fused square-2.
    if (prepared_mul_lead_enabled_ && factor == 1 && dst != src) {
      size_t slot = find_prepared(src, true);
      if (slot != no_prepared) {
        if (pending_kind_ != PendingKind::None && pending_reg_ != dst) flush_pending_square();
        bool retained = false;
        if (pending_kind_ != PendingKind::None) {
          execute_pending(true);
          retained = true;
        }
        invalidate_if(dst);
        pending_kind_ = PendingKind::PreparedMul;
        pending_reg_ = dst;
        pending_prepared_slot_ = slot;
        pending_lead_width_ = retained;
        return;
      }
    }
#endif

    flush_pending_square();
    size_t slot = find_prepared(src, true);
#if defined(__APPLE__)
    // ECM keeps more prepared constants than the deliberately small Apple LRU.
    // Rebuild an evicted operand from its canonical register instead of falling
    // through to the unsupported generic Apple tailMul path.
    if (slot == no_prepared) {
      if (prepared_buffers_.empty())
        throw std::runtime_error("Apple Aevum prepared multiplication cache is unavailable");
      slot = acquire_prepared(src);
      gpu_->regPrepare(prepared_buffers_[slot], reg(src));
    }
    gpu_->regMulPrepared(reg(dst), prepared_buffers_[slot], 1);
#else
    if (slot != no_prepared) gpu_->regMulPrepared(reg(dst), prepared_buffers_[slot], 1);
    else gpu_->regMul(reg(dst), reg(src), 1);
#endif
    invalidate_if(dst);
    multiply_small(dst, factor);
  }

  void add(size_t dst, size_t src) {
    check_reg(dst);
    check_reg(src);
    flush_pending_square();
    invalidate_if(dst);
    gpu_->regAddWords(reg(dst), reg(src));
  }

  void sub_reg(size_t dst, size_t src) {
    check_reg(dst);
    check_reg(src);
    flush_pending_square();
    invalidate_if(dst);
    gpu_->regSubWords(reg(dst), reg(src));
  }

  void sub_u32(size_t dst, uint32_t value) {
    check_reg(dst);
    // Fold precisely one adjacent square-minus-two into the existing LL carry.
    // A second subtraction or any observable boundary must materialize first.
    if (fused_ll_enabled_ && pending_kind_ == PendingKind::Square && pending_reg_ == dst && !pending_ll_ && value == 2) {
      pending_ll_ = true;
      return;
    }
    flush_pending_square();
    invalidate_if(dst);
    gpu_->regSubU32(reg(dst), value);
  }

  bool equal(size_t lhs, size_t rhs) {
    check_reg(lhs);
    check_reg(rhs);
    flush_pending_square();
    return gpu_->regEqual(reg(lhs), reg(rhs));
  }

  void debug_square_trace(size_t src, uint64_t* trace, size_t trace_count) {
    check_reg(src);
    flush_pending_square();
    if (!trace || trace_count < 12) throw std::runtime_error("Aevum square trace buffer must contain at least 12 uint64 values");
    if (small_factor_scratch_.empty()) throw std::runtime_error("Aevum square trace scratch is unavailable");
    gpu_->regCopy(small_factor_scratch_[0], reg(src));
    gpu_->regDebugSquareTrace(small_factor_scratch_[0], trace, trace_count);
  }

private:
  static constexpr size_t no_prepared = std::numeric_limits<size_t>::max();
  enum class PendingKind { None, Square, PreparedMul };

  struct PreparedSlot {
    size_t reg = no_prepared;
    uint64_t stamp = 0;
  };

  void execute_pending(bool lead_out) {
    if (pending_kind_ == PendingKind::None) return;
    const PendingKind kind = pending_kind_;
    const size_t index = pending_reg_;
    const size_t prepared_slot = pending_prepared_slot_;
    const bool lead_in = pending_lead_width_;
    const bool ll = pending_ll_;

    // Clear first so an exception cannot leave a stale transformed-state tag.
    pending_kind_ = PendingKind::None;
    pending_reg_ = no_prepared;
    pending_prepared_slot_ = no_prepared;
    pending_lead_width_ = false;
    pending_ll_ = false;

    if (kind == PendingKind::Square) {
      gpu_->regSquareStep(reg(index), lead_in, lead_out, ll);
      return;
    }
    if (prepared_slot >= prepared_buffers_.size())
      throw std::runtime_error("stale Aevum prepared-multiply lead slot");
    gpu_->regMulPreparedStep(reg(index), prepared_buffers_[prepared_slot], lead_in, lead_out);
  }

  void flush_pending_square() { execute_pending(false); }

  size_t find_prepared(size_t index, bool touch) {
    for (size_t i = 0; i < prepared_slots_.size(); ++i) {
      if (prepared_slots_[i].reg == index) {
        if (touch) prepared_slots_[i].stamp = ++prepared_clock_;
        return i;
      }
    }
    return no_prepared;
  }

  size_t acquire_prepared(size_t index) {
    const size_t existing = find_prepared(index, true);
    if (existing != no_prepared) return existing;
    size_t slot = no_prepared;
    for (size_t i = 0; i < prepared_slots_.size(); ++i) {
      if (prepared_slots_[i].reg == no_prepared) { slot = i; break; }
      if (slot == no_prepared || prepared_slots_[i].stamp < prepared_slots_[slot].stamp) slot = i;
    }
    prepared_slots_[slot].reg = index;
    prepared_slots_[slot].stamp = ++prepared_clock_;
    return slot;
  }

  void invalidate_if(size_t index) {
    for (auto& slot : prepared_slots_) {
      if (slot.reg == index) slot.reg = no_prepared;
    }
  }

  void multiply_small(size_t index, uint32_t factor) {
    if (factor == 1) return;
    if (small_factor_scratch_.empty()) throw std::runtime_error("Aevum small-factor scratch is unavailable");

    gpu_->regCopy(small_factor_scratch_[0], reg(index));
    gpu_->regSetU32(reg(index), 0);

    uint32_t bits = factor;
#if defined(__APPLE__)
    size_t current = 0;
    size_t next = 1;
    while (bits != 0) {
      if (bits & 1u) gpu_->regAddWords(reg(index), small_factor_scratch_[current]);
      bits >>= 1;
      if (bits != 0) {
        gpu_->regCopy(small_factor_scratch_[next], small_factor_scratch_[current]);
        gpu_->regAddWords(small_factor_scratch_[next], small_factor_scratch_[current]);
        std::swap(current, next);
      }
    }
#else
    while (bits != 0) {
      if (bits & 1u) gpu_->regAddWords(reg(index), small_factor_scratch_[0]);
      bits >>= 1;
      if (bits != 0) gpu_->regAddWords(small_factor_scratch_[0], small_factor_scratch_[0]);
    }
#endif
  }

  void check_reg(size_t index) const {
    if (index >= register_count_) throw std::runtime_error("Aevum register index out of range");
  }

  Buffer<Word>& reg(size_t index) { return buffers_.at(index); }

  uint32_t exponent_;
  size_t register_count_;
  aevum_autotune::Workload workload_;
  size_t word_count_;
  size_t transform_size_{};
  Args args_;
  std::unique_ptr<Context> context_;
  Background background_;
  std::unique_ptr<TrigBufCache> cache_;
  GpuCommon shared_{};
  std::unique_ptr<Gpu> gpu_;
  std::vector<Buffer<Word>> buffers_;
  std::vector<Buffer<double>> prepared_buffers_;
  std::vector<PreparedSlot> prepared_slots_;
  std::vector<Buffer<Word>> small_factor_scratch_;
  uint64_t prepared_clock_ = 0;
  bool fused_ll_enabled_ = false;
  bool pending_ll_ = false;
  bool lead_cache_enabled_ = false;
  bool prepared_mul_lead_enabled_ = false;
  PendingKind pending_kind_ = PendingKind::None;
  size_t pending_reg_ = no_prepared;
  size_t pending_prepared_slot_ = no_prepared;
  bool pending_lead_width_ = false;
};

template <class F>
int invoke(F&& f) {
  try {
    g_last_error.clear();
    f();
    return 1;
  } catch (const std::exception& e) {
    set_error(e);
  } catch (const std::string& e) {
    set_error(e.c_str());
  } catch (const char* e) {
    set_error(e);
  } catch (...) {
    set_error("unknown Aevum engine exception");
  }
  return 0;
}

Runtime& runtime(aevum_engine_handle handle) {
  if (!handle) throw std::runtime_error("null Aevum engine handle");
  return *static_cast<Runtime*>(handle);
}

} // namespace

extern "C" {

const char* aevum_engine_version(void) { return VERSION; }
const char* aevum_engine_last_error(void) { return g_last_error.c_str(); }

int aevum_engine_resolve_fft(uint32_t exponent, const char* fft_spec, char* output, size_t output_size) {
  try {
    g_last_error.clear();
    if (!output || output_size == 0) throw std::runtime_error("invalid Aevum FFT output buffer");
    Args args(true);
    if (const char* tune = std::getenv("AEVUM_TUNE_DIR")) {
      if (*tune) args.masterDir = std::filesystem::absolute(tune);
    }
    // Keep this resolver strictly device-neutral: TuneEntry only needs
    // masterDir/fftOverdrive here. Args::setDefaults() queries OpenCL device
    // metadata and would make host-only policy tests require an installed ICD.
    FFTConfig fft = FFTConfig::bestFit(args, exponent, fft_spec ? fft_spec : "");
    const std::string resolved = fft.spec();
    if (resolved.size() + 1 > output_size) throw std::runtime_error("Aevum FFT output buffer is too small");
    std::memcpy(output, resolved.c_str(), resolved.size() + 1);
    return 1;
  } catch (const std::exception& e) {
    set_error(e);
  } catch (const std::string& e) {
    set_error(e.c_str());
  } catch (const char* e) {
    set_error(e);
  } catch (...) {
    set_error("unknown Aevum FFT selection exception");
  }
  return 0;
}

aevum_engine_handle aevum_engine_create_ex(uint32_t exponent, size_t register_count, uint32_t device, int verbose, const char* fft_spec, const char* tune_dir, uint32_t workload) {
  try {
    g_last_error.clear();
    return new Runtime(exponent, register_count, device, verbose != 0, fft_spec, tune_dir, workload);
  } catch (const std::exception& e) {
    set_error(e);
  } catch (const std::string& e) {
    set_error(e.c_str());
  } catch (const char* e) {
    set_error(e);
  } catch (...) {
    set_error("unknown Aevum engine exception");
  }
  return nullptr;
}

aevum_engine_handle aevum_engine_create(uint32_t exponent, size_t register_count, uint32_t device, int verbose, const char* fft_spec, const char* tune_dir) {
  return aevum_engine_create_ex(exponent, register_count, device, verbose, fft_spec, tune_dir, AEVUM_WORKLOAD_GENERIC);
}

void aevum_engine_destroy(aevum_engine_handle handle) {
  delete static_cast<Runtime*>(handle);
}

size_t aevum_engine_transform_size(aevum_engine_handle handle) {
  try { return runtime(handle).transform_size(); } catch (...) { return 0; }
}

size_t aevum_engine_word_count(aevum_engine_handle handle) {
  try { return runtime(handle).word_count(); } catch (...) { return 0; }
}

int aevum_engine_sync(aevum_engine_handle handle) { return invoke([&] { runtime(handle).sync(); }); }
int aevum_engine_set_u32(aevum_engine_handle handle, size_t dst, uint32_t value) { return invoke([&] { runtime(handle).set_u32(dst, value); }); }
int aevum_engine_set_words(aevum_engine_handle handle, size_t dst, const uint32_t* words, size_t count) { return invoke([&] { runtime(handle).set_words(dst, words, count); }); }
int aevum_engine_get_words(aevum_engine_handle handle, size_t src, uint32_t* words, size_t count) { return invoke([&] { runtime(handle).get_words(src, words, count); }); }
int aevum_engine_copy(aevum_engine_handle handle, size_t dst, size_t src) { return invoke([&] { runtime(handle).copy(dst, src); }); }
int aevum_engine_prepare(aevum_engine_handle handle, size_t dst, size_t src) { return invoke([&] { runtime(handle).prepare(dst, src); }); }
int aevum_engine_square_mul(aevum_engine_handle handle, size_t reg, uint32_t factor) { return invoke([&] { runtime(handle).square_mul(reg, factor); }); }
int aevum_engine_mul(aevum_engine_handle handle, size_t dst, size_t src, uint32_t factor) { return invoke([&] { runtime(handle).mul(dst, src, factor); }); }
int aevum_engine_add(aevum_engine_handle handle, size_t dst, size_t src) { return invoke([&] { runtime(handle).add(dst, src); }); }
int aevum_engine_sub_reg(aevum_engine_handle handle, size_t dst, size_t src) { return invoke([&] { runtime(handle).sub_reg(dst, src); }); }
int aevum_engine_sub_u32(aevum_engine_handle handle, size_t dst, uint32_t value) { return invoke([&] { runtime(handle).sub_u32(dst, value); }); }
int aevum_engine_equal(aevum_engine_handle handle, size_t lhs, size_t rhs, int* equal_out) {
  return invoke([&] {
    if (!equal_out) throw std::runtime_error("null Aevum equality output");
    *equal_out = runtime(handle).equal(lhs, rhs) ? 1 : 0;
  });
}

int aevum_engine_debug_square_trace(aevum_engine_handle handle, size_t src, uint64_t* trace, size_t trace_count) {
  return invoke([&] { runtime(handle).debug_square_trace(src, trace, trace_count); });
}

} // extern "C"

extern "C" int aevum_engine_profile_report(aevum_engine_handle handle, int emit) {
  return invoke([&] { runtime(handle).profile_report(emit != 0); });
}
