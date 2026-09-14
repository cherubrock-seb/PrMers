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
#include "PrpUseTune.h"
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

bool pm1Factor3Workload(aevum_autotune::Workload workload) {
  return workload == aevum_autotune::Workload::Pm1 ||
         workload == aevum_autotune::Workload::Pm1Lowmem ||
         workload == aevum_autotune::Workload::Pm1Ultralowmem;
}

// square_mul(x,3) needs more dynamic range than a pure square.
// A plan admissible for exponent E is not necessarily admissible for
// x <- 3*x^2.  Account explicitly for log2(3) extra bits/word.
bool smallFactorCapacitySafe(const Args& args,
                             uint32_t exponent,
                             const FFTConfig& fft,
                             uint32_t factor) {
  if (factor <= 1u) return true;
  const double bpw = double(exponent) / double(fft.size());
  const double required_bpw = bpw + std::log2(double(factor));
  const double allowed_bpw = double(fft.maxBpw()) * args.fftOverdrive;
  return required_bpw <= allowed_bpw;
}

bool factor3CapacitySafe(const Args& args,
                         uint32_t exponent,
                         const FFTConfig& fft) {
  return smallFactorCapacitySafe(args, exponent, fft, 3u);
}

FFTConfig promoteToFactor3SafePlan(const Args& args,
                                   uint32_t exponent,
                                   FFTConfig fft) {
  for (unsigned attempt = 0; attempt < 8; ++attempt) {
    if (factor3CapacitySafe(args, exponent, fft)) return fft;

    // Convert the factor-3 headroom into an equivalent exponent requirement.
    const uint64_t effective_exponent =
        static_cast<uint64_t>(std::ceil(
            double(exponent) +
            std::log2(3.0) * double(fft.size())));

    FFTConfig next = FFTConfig::bestFit(args, effective_exponent, "");

    if (next.size() <= fft.size()) {
      // Defensive escape from a selector boundary that does not advance.
      next = FFTConfig::bestFit(
          args,
          effective_exponent + static_cast<uint64_t>(fft.size()),
          "");
    }

    if (next.size() <= fft.size())
      throw std::runtime_error(
          "Aevum cannot find a factor-3-safe P-1 FFT plan");

    fft = next;
  }

  throw std::runtime_error(
      "Aevum factor-3 P-1 FFT promotion did not converge");
}

bool usableTuneEntry(const Args& args,
                     uint32_t exponent,
                     aevum_autotune::Workload workload) {
  for (const TuneEntry& tuned : TuneEntry::readTuneFile(args)) {
    if (tuned.fft.shape.fft_type != FFT3161 || tuned.fft.isPfa()) continue;
    const double bpw = exponent / double(tuned.fft.size());
    if (bpw < tuned.fft.minBpw()) continue;
    if (tuned.fft.maxExp() * args.fftOverdrive < exponent) continue;
    if (pm1Factor3Workload(workload) &&
        !factor3CapacitySafe(args, exponent, tuned.fft)) continue;
    return true;
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
    if (pm1Factor3Workload(workload) &&
        !factor3CapacitySafe(args, exponent, *fft)) return;
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

// Production-faithful PRP probe used by the implementation (-use) tuner.
// The public engine does not time a naked Gpu::regSquareStep chain: it owns the
// configured register set, prepared/scratch buffers, and keeps one square
// pending so consecutive square_mul(reg,1) calls retain the width transform.
// Keeping the tuner on the same path avoids validating a micro-workload that
// can disagree with the real PRP engine.
class PrpUseProbe {
public:
  PrpUseProbe(uint32_t exponent, size_t register_count, GpuCommon shared,
              const FFTConfig& fft, const std::vector<KeyVal>& use)
      : gpu_(Gpu::make(exponent, shared, fft, use, false)) {
    const auto count = static_cast<u32>(std::max<size_t>(register_count, 1));
    regs_ = gpu_->makeBufVector(count);
    prepared_ = gpu_->makeTransformBufVector(static_cast<u32>(std::min<size_t>(register_count, 2)));
#if defined(__APPLE__)
    scratch_ = gpu_->makeBufVector(2);
#else
    scratch_ = gpu_->makeBufVector(1);
#endif
    lead_ = gpu_->regSupportsLeadCache();
    // Mirror the production Runtime allocation/initialization footprint used
    // by the benchmark. The prepared operand is not part of PRP timing, but it
    // is materialized before timing in the real engine.
    if (regs_.size() > 1) {
      gpu_->regSetU32(regs_[1], 7);
      if (!prepared_.empty()) gpu_->regPrepare(prepared_[0], regs_[1]);
      gpu_->regSync();
    }
  }

  Gpu& gpu() { return *gpu_; }
  Buffer<Word>& value() { return regs_[0]; }

  void reset(const Words& seed) {
    sync();
    gpu_->regWrite(regs_[0], seed);
    gpu_->regSync();
  }

  void square() {
    if (!lead_) {
      gpu_->regSquare(regs_[0], 1);
      return;
    }
    // Exact analogue of Runtime::square_mul(reg,1): execute the previous
    // pending square with leadOut=WIDTH, then retain the current square.
    if (pending_) {
      gpu_->regSquareStep(regs_[0], pending_lead_, true, false);
      pending_lead_ = true;
    } else {
      pending_lead_ = false;
    }
    pending_ = true;
  }

  void sequence(unsigned units) {
    for (unsigned i = 0; i < units; ++i) square();
    sync();
  }

  void sync() {
    if (pending_) {
      gpu_->regSquareStep(regs_[0], pending_lead_, false, false);
      pending_ = false;
      pending_lead_ = false;
    }
    gpu_->regSync();
  }

  Words read() {
    sync();
    return gpu_->regRead(regs_[0]);
  }

private:
  std::unique_ptr<Gpu> gpu_;
  std::vector<Buffer<Word>> regs_;
  std::vector<Buffer<double>> prepared_;
  std::vector<Buffer<Word>> scratch_;
  bool lead_ = false;
  bool pending_ = false;
  bool pending_lead_ = false;
};

// PRP use selection is deliberately downstream of all shape selection paths.
std::vector<KeyVal> selectPrpUse(uint32_t exponent, size_t regs, GpuCommon shared,
    const FFTConfig& fft, const std::string& identity, uint64_t shape_ms, bool shape_ran,
    const std::vector<KeyVal>& existing, bool verbose) {
  using namespace aevum_prp_use;
  auto report=[&](const char* source,const std::string& profile,double gain=1) {
    if(verbose)log("AEVUM_PRP_USE source=%s shape=%s profile=%s gain=%.5f\n",source,fft.spec().c_str(),profile.empty()?"defaults":profile.c_str(),gain);
  };
  if(const char* manual=std::getenv("AEVUM_PRP_USE")) {
    auto use=parse(manual); report("manual",normalize(use)); return use;
  }
  // Issue #36: native tune entries may carry both a validated FFT shape and
  // validated -use policy. Preserve those flags exactly as a higher-priority
  // source; automatic tuning never overwrites a matching tune entry.
  for(const auto& entry:TuneEntry::readTuneFile(*shared.args)) {
    if(entry.fft.spec()==fft.spec() && !entry.use.empty() && entry.fft.maxExp()>=exponent) {
      report("tune-entry",normalize(entry.use));return entry.use;
    }
  }
  const auto control=mode();
  bool manual_kernel=false;
  for(const char* name:{"AEVUM_PFA_USE","AEVUM_RADIX1K","AEVUM_TYPE4_MULTI_Q","AEVUM_CARRY_WMUL",
      "AEVUM_GF61_LIMB32","AEVUM_PRP_MIDDLE1","AEVUM_REG_LEAD_CACHE"})
    if(const char* value=std::getenv(name))if(*value)manual_kernel=true;
  if(control==aevum_autotune::Mode::Off || manual_kernel || fft.isPfa()) {report("defaults/bypass","");return {};}

  std::string flags=normalize(existing);
  for(const auto& [k,v]:shared.args->flags)flags+=';'+k+'='+v;
  const auto cache_key=key(identity,fft.spec(),exponent,flags);
  if(control==aevum_autotune::Mode::Auto)if(auto r=load(cache_key)) {
    auto use=r->plan=="defaults"?std::vector<KeyVal>{}:parse(r->plan);
    report("cache-hit",normalize(use),r->implementation_speedup);return use;
  }

  const bool nvidia_asm=isNvidiaGpu(shared.context->deviceId()) && !shared.args->uses("NO_ASM");
  const auto profiles=candidates(nvidia_asm);
  // A cold FFT-shape tune gets a smaller implementation slice. Once the shape
  // is cached the bounded neighbourhood expands automatically. Progress stores
  // the prior plan size so a later expansion restarts finalist confirmation.
  const unsigned cap=shape_ran && shape_ms>=3000 ? 6u : 12u;
  const unsigned requested=boundedEnvUnsigned("AEVUM_PRP_USE_BUDGET_MS",cap<=6?6000:10000,100,12000);
  const uint64_t budget=std::min<uint64_t>(requested,shape_ran?(shape_ms<16000?16000-shape_ms:0):requested);
  const unsigned full_planned=static_cast<unsigned>(profiles.size());
  const unsigned planned=std::min<unsigned>(cap,full_planned);

  if(control==aevum_autotune::Mode::Retune) clearProgress(cache_key);
  ResumeState state;
  if(control==aevum_autotune::Mode::Auto)if(auto saved=loadProgress(cache_key))state=*saved;
  auto valid_profile=[&](const std::string& profile) {
    return std::find(profiles.begin(),profiles.begin()+planned,profile)!=profiles.begin()+planned;
  };
  bool resume_valid=state.next_screen<=planned && state.next_finalist<=state.top.size() && state.top.size()<=4;
  for(const auto& [profile,gain]:state.top)resume_valid=resume_valid && valid_profile(profile) && std::isfinite(gain) && gain>0;
  if(!state.winner.empty())resume_valid=resume_valid && valid_profile(state.winner) && std::isfinite(state.winning_gain) && state.winning_gain>=1.03;
  if(!resume_valid){state={};clearProgress(cache_key);}
  if(state.planned && state.planned<planned && state.next_finalist) {
    // New candidates became eligible after the expensive shape tune completed.
    // Their screening may alter the top-four shortlist, so previous finalist
    // progress cannot be considered complete.
    state.next_finalist=0;state.winner.clear();state.winning_gain=1.0;
  }
  state.planned=planned;

  const auto start=std::chrono::steady_clock::now();
  auto elapsed=[&] {return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();};
  SearchProgress progress{.planned=planned,.screened=state.next_screen};
  std::string winner=state.winner;double winning_gain=state.winning_gain;
  struct Screen {std::string profile;double gain;};
  std::vector<Screen> top;for(const auto& [profile,gain]:state.top)top.push_back({profile,gain});
  auto save_state=[&] {
    state.planned=planned;state.top.clear();for(const auto& x:top)state.top.push_back({x.profile,x.gain});
    state.winner=winner;state.winning_gain=winning_gain;storeProgress(cache_key,state);
  };
  auto rank=[&](const std::string& profile,double gain) {
    // Screening is intentionally permissive. The final real-PRP gate remains
    // strict (>=3%, >=2/3 wins, worst >=0.985).
    if(gain<1.005)return;
    top.push_back({profile,gain});
    std::sort(top.begin(),top.end(),[](const auto& a,const auto& b){return a.gain>b.gain;});
    top.erase(std::unique(top.begin(),top.end(),[](const auto& a,const auto& b){return a.profile==b.profile;}),top.end());
    if(top.size()>4)top.resize(4);
  };
  auto merged=[&](const std::string& profile) {
    auto use=existing;
    if(!profile.empty())for(const auto& kv:parse(profile)) {
      use.erase(std::remove_if(use.begin(),use.end(),[&](const KeyVal& x){return x.first==kv.first;}),use.end());
      use.push_back(kv);
    }
    return use;
  };
  const Words seed=deterministicResidue(exponent,0x12345678u);
  auto exact_profile=[&](const std::string& profile) {
    Words reference;
    {
      PrpUseProbe base(exponent,regs,shared,fft,existing);
      base.reset(seed);base.sequence(16);reference=base.read();
    }
    {
      PrpUseProbe candidate(exponent,regs,shared,fft,merged(profile));
      candidate.gpu().regPrpRoe(true);candidate.reset(seed);candidate.sequence(16);
      const Words value=candidate.read();
      if(reference!=value)throw std::runtime_error("WORD MISMATCH");
      if(candidate.gpu().regPrpRoe(false)>=0.35)throw std::runtime_error("ROE gate");
    }
  };
  auto prp_cost=[&](const std::string& profile,int quick) {
    // Upstream AEVUM uses Gpu::timePRP itself for tune.cpp. It exercises the
    // true PRP state/check/square path, includes its own warm-up and correctness
    // check, and avoids the synthetic square-chain confirmation that hid the
    // RTX3080 ~7% implementation gain in R4.
    auto gpu=Gpu::make(exponent,shared,fft,merged(profile),false);
    const double us=gpu->timePRP(quick);
    if(!std::isfinite(us) || us<=0.0 || us>=100000.0)throw std::runtime_error("invalid PRP timing/check");
    return us;
  };

  if(verbose)log("AEVUM_PRP_USE search-v6 shape=%s candidates<=%u budget=%" PRIu64 "ms resume_screen=%u resume_finalist=%u metric=timePRP\n",
      fft.spec().c_str(),cap,budget,state.next_screen,state.next_finalist);
  try {
    if(budget>=100 && state.next_screen<planned) {
      // One discarded real-PRP pass stabilizes clocks/compiled default kernels
      // before the first ranking sample of this invocation.
      (void)prp_cost("",10);
      for(unsigned index=state.next_screen;index<planned;++index) {
        if(elapsed()>=static_cast<int64_t>(budget)){progress.interrupted=true;break;}
        const auto& profile=profiles[index];
        try {
          exact_profile(profile);
          double a,b;
          // Candidate construction/compilation is outside timePRP's timer.
          // Alternate order to reduce DVFS drift while keeping objects serial,
          // so baseline and candidate never contend for device memory/queues.
          if((index&1u)==0){a=prp_cost("",10);b=prp_cost(profile,10);}
          else{b=prp_cost(profile,10);a=prp_cost("",10);}
          const double gain=a/b;
          if(verbose)log("AEVUM_USE_SCREEN profile=%s exact=1 metric=timePRP gain=%.5f baseline_us=%.4f candidate_us=%.4f\n",
              profile.c_str(),gain,a,b);
          rank(profile,gain);
        }catch(const std::exception& e){if(verbose)log("AEVUM_USE_REJECT profile=%s reason=%s\n",profile.c_str(),e.what());}
        catch(...){if(verbose)log("AEVUM_USE_REJECT profile=%s reason=engine-exception\n",profile.c_str());}
        state.next_screen=index+1;progress.screened=state.next_screen;save_state();
        if(elapsed()>=static_cast<int64_t>(budget)){progress.interrupted=state.next_screen<planned;break;}
        if(!top.empty() && elapsed()>static_cast<int64_t>(budget)/2 && state.next_screen<planned){progress.interrupted=true;break;}
      }
    }

    if(state.next_screen==planned) {
      progress.screened=planned;progress.finalists=top.size();
      if(state.next_finalist>top.size())state.next_finalist=0;
      if(budget>=100 && state.next_finalist<top.size()) {
        const int confirm_quick=exponent<=30000000u?8:exponent<=120000000u?9:10;
        for(unsigned fi=state.next_finalist;fi<top.size();++fi) {
          if(elapsed()>=static_cast<int64_t>(budget)){progress.interrupted=true;break;}
          const auto screened=top[fi];
          try {
            exact_profile(screened.profile);
            // Discard one sample of each side. The measured samples below use
            // separate Gpu objects and upstream timePRP, closely matching the
            // production tuning target while excluding compilation/JIT time.
            (void)prp_cost("",10);(void)prp_cost(screened.profile,10);
            std::array<double,3>a{},b{};unsigned wins=0;double worst=100;
            for(unsigned repeat=0;repeat<3;++repeat) {
              if(repeat&1u){b[repeat]=prp_cost(screened.profile,confirm_quick);a[repeat]=prp_cost("",confirm_quick);}
              else{a[repeat]=prp_cost("",confirm_quick);b[repeat]=prp_cost(screened.profile,confirm_quick);}
              const double gain=a[repeat]/b[repeat];wins+=gain>=1.02;worst=std::min(worst,gain);
            }
            const double gain=median3(a)/median3(b);
            const bool keep=gain>=1.03 && wins>=2 && worst>=0.985;
            if(verbose)log("AEVUM_USE_CONFIRM profile=%s exact=1 metric=timePRP median_gain=%.5f worst=%.5f keep=%d baseline_us=%.4f,%.4f,%.4f candidate_us=%.4f,%.4f,%.4f\n",
                screened.profile.c_str(),gain,worst,keep,a[0],a[1],a[2],b[0],b[1],b[2]);

            // R6 final authority: timePRP is an excellent cheap ranker, but it
            // times AEVUM's native PRP/check loop rather than the public engine
            // square_mul hot path used by PrMers.  Real Radeon hardware showed
            // a false +18% timePRP winner at 150M which was only +1% through
            // the engine.  Before any positive cache entry is committed, replay
            // the same 64-warmup + 256-square sequence as aevum_engine_bench,
            // alternating baseline/candidate x3.  This keeps the fast ranking
            // stage while making production-engine throughput the final truth.
            bool engine_keep=false; double engine_gain=1.0;
            if(keep) {
              auto engine_cost=[&](const std::string& profile) {
                PrpUseProbe probe(exponent,regs,shared,fft,merged(profile));
                probe.reset(seed);probe.sequence(64); // production benchmark warmup
                probe.reset(seed);
                const auto begin=std::chrono::steady_clock::now();
                probe.sequence(256);
                const double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                if(!std::isfinite(secs) || secs<=0.0)throw std::runtime_error("invalid engine-style PRP timing");
                return secs;
              };
              std::array<double,3> ea{},eb{};unsigned ewins=0;double eworst=100;
              for(unsigned repeat=0;repeat<3;++repeat) {
                if(repeat&1u){eb[repeat]=engine_cost(screened.profile);ea[repeat]=engine_cost("");}
                else{ea[repeat]=engine_cost("");eb[repeat]=engine_cost(screened.profile);}
                const double eg=ea[repeat]/eb[repeat];ewins+=eg>=1.02;eworst=std::min(eworst,eg);
              }
              engine_gain=median3(ea)/median3(eb);
              engine_keep=engine_gain>=1.03 && ewins>=2 && eworst>=0.985;
              if(verbose)log("AEVUM_USE_ENGINE_GATE profile=%s exact=1 metric=square_mul256 median_gain=%.5f worst=%.5f keep=%d baseline_s=%.6f,%.6f,%.6f candidate_s=%.6f,%.6f,%.6f\n",
                  screened.profile.c_str(),engine_gain,eworst,engine_keep,ea[0],ea[1],ea[2],eb[0],eb[1],eb[2]);
            }
            if(keep && engine_keep && engine_gain>winning_gain){winner=screened.profile;winning_gain=engine_gain;}
          }catch(const std::exception& e){if(verbose)log("AEVUM_USE_REJECT profile=%s reason=%s\n",screened.profile.c_str(),e.what());}
          catch(...){if(verbose)log("AEVUM_USE_REJECT profile=%s reason=engine-exception\n",screened.profile.c_str());}
          state.next_finalist=fi+1;progress.finalized=state.next_finalist;save_state();
          if(elapsed()>=static_cast<int64_t>(budget) && state.next_finalist<top.size()){progress.interrupted=true;break;}
        }
      } else progress.finalized=state.next_finalist;
    }
  }catch(const std::exception& e){progress.interrupted=true;if(verbose)log("AEVUM_USE_REJECT reason=%s\n",e.what());}
  catch(...){progress.interrupted=true;if(verbose)log("AEVUM_USE_REJECT reason=engine-exception\n");}

  progress.screened=state.next_screen;
  if(state.next_screen==planned){progress.finalists=top.size();progress.finalized=state.next_finalist;}
  if(state.next_screen<planned || (state.next_screen==planned && state.next_finalist<top.size()))progress.interrupted=true;
  else progress.interrupted=false;
  // A cold shape-tune invocation may intentionally expose only the first half
  // of the bounded -use neighbourhood.  Never turn that reduced scope into a
  // permanent positive/negative cache decision: the next AUTO invocation has
  // a cached shape and must expand to the complete candidate set first.
  if(planned<full_planned)progress.interrupted=true;
  const auto decision=progress.decision(!winner.empty());
  const char* state_name=decision==Decision::Positive?"completed-positive":decision==Decision::Negative?"completed-negative":"deferred";
  if(verbose)log("AEVUM_USE_DECISION state=%s screened=%u/%u full_candidates=%u finalized=%u/%u elapsed=%" PRIu64 "ms next_screen=%u next_finalist=%u best_gain=%.5f\n",
      state_name,progress.screened,progress.planned,full_planned,progress.finalized,progress.finalists,uint64_t(elapsed()),state.next_screen,state.next_finalist,winning_gain);
  try{persistDecision(cache_key,decision,winner,winning_gain,elapsed());}
  catch(...){if(verbose)log("AEVUM_PRP_USE cache-write-failed; using completed decision\n");}
  report(decision==Decision::Deferred?"deferred":decision==Decision::Positive?"measured":"measured-defaults",winner,winning_gain);
  // A deferred winner is not yet final: safe defaults remain active until all
  // shortlisted finalists have been compared and a completed-positive record
  // is atomically committed.
  return decision==Decision::Positive?parse(winner):std::vector<KeyVal>{};
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
    // these native knobs are exposed independently through the Pass-5 PRP use selector.
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
    const bool compatible_tune_entry =
        spec.empty() && usableTuneEntry(args_, exponent_, workload_);
    const bool workload_tunable = workload_ != aevum_autotune::Workload::Pm1Ultralowmem;
    const bool autotune_eligible = autotune_mode != aevum_autotune::Mode::Off &&
        !explicit_fft_spec && !gb202_profile && !manual_plan_env &&
        !compatible_tune_entry && workload_tunable;

    FFTConfig native_fft = FFTConfig::bestFit(args_, exponent_, spec);

    if (!explicit_fft_spec &&
        !manual_plan_env &&
        !gb202_profile &&
        pm1Factor3Workload(workload_) &&
        !factor3CapacitySafe(args_, exponent_, native_fft)) {
      const std::string unsafe_spec = native_fft.spec();
      const double unsafe_bpw =
          double(exponent_) / double(native_fft.size());

      native_fft =
          promoteToFactor3SafePlan(args_, exponent_, native_fft);

      if (verbose) {
        log("Aevum P-1 factor-3 capacity guard: "
            "%s rejected at %.2f bpw; promoted to %s "
            "(+log2(3) arithmetic headroom).\n",
            unsafe_spec.c_str(),
            unsafe_bpw,
            native_fft.spec().c_str());
      }
    }

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
          if (cached_fft &&
              (!pm1Factor3Workload(workload_) ||
               factor3CapacitySafe(args_, exponent_, *cached_fft))) {
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

    // Runtime backstop for every square/multiply followed by a small integer
    // factor. Automatic P-1 and Gaussian modes select a safe plan ahead of
    // time; explicit/manual unsafe plans fail loudly instead of returning
    // a silently corrupted residue.
    small_factor_headroom_bits_ =
        double(fft.maxBpw()) * args_.fftOverdrive -
        double(exponent_) / double(fft.size());

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
#if !defined(__APPLE__) && !defined(CUDA_BACKEND)
    if (workload_ == aevum_autotune::Workload::Prp && !fft.isPfa()) {
      const auto identity = aevum_autotune::makeKey(VERSION, device_vendor, device_name, device_driver, device_runtime,
          workload_, register_count_, exponent_, "safe=1;clean=1;ordinal="+std::to_string(device)+
              ";opencl-c="+openclDeviceInfoString(selected_device,CL_DEVICE_OPENCL_C_VERSION));
      auto profile = selectPrpUse(exponent_, register_count_, shared_, fft, identity,
          autotune_cache_hit ? 0 : autotune_elapsed_ms, autotune_runtime_ran, pfa_use, verbose);
      for (const auto& kv : profile) {
        pfa_use.erase(std::remove_if(pfa_use.begin(),pfa_use.end(),[&](const KeyVal& x){return x.first==kv.first;}),pfa_use.end());
        pfa_use.push_back(kv);
      }
    }
#endif
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

    const double required_bits = std::log2(double(factor));
    if (required_bits > small_factor_headroom_bits_ + 1.0e-9) {
      throw std::runtime_error(
          "Aevum small-factor capacity exceeded: factor=" +
          std::to_string(factor) +
          " requires +" + std::to_string(required_bits) +
          " bits of FFT headroom, available=" +
          std::to_string(small_factor_headroom_bits_));
    }

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
  double small_factor_headroom_bits_ = 0.0;
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
