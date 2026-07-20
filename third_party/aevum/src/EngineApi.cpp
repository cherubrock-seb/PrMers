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
#include "common.h"
#include "gpuid.h"
#include "version.h"

#include <algorithm>
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

class Runtime {
public:
  Runtime(uint32_t exponent, size_t register_count, uint32_t device, bool verbose, const char* fft_spec, const char* tune_dir)
      : exponent_(exponent),
        register_count_(register_count),
        word_count_(nWords(exponent)),
        args_(true),
        context_(nullptr),
        cache_(nullptr) {
    if (exponent_ < 3) throw std::runtime_error("Aevum exponent must be at least 3");
    if (register_count_ == 0) throw std::runtime_error("Aevum register count must be positive");

    args_.device = static_cast<int>(device);
    args_.verbose = verbose;
    args_.safeMath = true;
    args_.clean = true;
    args_.useCache = true;
    args_.cacheDir = std::filesystem::absolute(".aevum-kernel-cache");
    args_.proofResultDir = std::filesystem::absolute(".aevum-proof");
    args_.proofToVerifyDir = std::filesystem::absolute(".aevum-proof-tmp");
    if (tune_dir && *tune_dir) args_.masterDir = std::filesystem::absolute(tune_dir);
    args_.setDefaults();

    context_ = std::make_unique<Context>(getDevice(device));
    cache_ = std::make_unique<TrigBufCache>(context_.get());

    shared_.context = context_.get();
    shared_.args = &args_;
    shared_.bufCache = cache_.get();
    shared_.background = &background_;

    const std::string spec = fft_spec ? fft_spec : "";
    FFTConfig fft = FFTConfig::bestFit(args_, exponent_, spec);

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
    gpu_ = Gpu::make(exponent_, shared_, fft, pfa_use, verbose);
    transform_size_ = gpu_->getFFTSize();

    lead_cache_enabled_ = gpu_->regSupportsLeadCache();
    if (const char* value = std::getenv("AEVUM_REG_LEAD_CACHE"))
      lead_cache_enabled_ = lead_cache_enabled_ && std::atoi(value) != 0;
    if (verbose) {
      if (lead_cache_enabled_) {
        log("Aevum register lead cache enabled: consecutive square_mul(reg,1) calls retain the width transform and use carryFused.\n");
      } else if (!fft.isPfa()) {
        log("Aevum register lead cache disabled; set AEVUM_REG_LEAD_CACHE=1 only on a supported non-Apple, short-carry plan.\n");
      }
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
    invalidate_if(index);

    // The register API exposes one square at a time, while upstream PRPLL
    // keeps a LEAD_WIDTH transform between consecutive squarings.  Keep one
    // logical square pending: the next call executes the previous square with
    // leadOut=WIDTH, and a read/copy/checkpoint executes the final square with
    // leadOut=NONE.  Thus every externally visible value is canonical while
    // the hot PRP loop uses the same carryFused chain as PRPLL.
    if (lead_cache_enabled_ && factor == 1) {
      if (pending_reg_ != no_prepared && pending_reg_ != index) flush_pending_square();
      if (pending_reg_ == no_prepared) {
        pending_reg_ = index;
        pending_lead_width_ = false;
        return;
      }
      gpu_->regSquareStep(reg(index), pending_lead_width_, true);
      pending_lead_width_ = true;
      return;
    }

    flush_pending_square();
    gpu_->regSquare(reg(index), 1);
    multiply_small(index, factor);
  }

  void mul(size_t dst, size_t src, uint32_t factor) {
    check_reg(dst);
    check_reg(src);
    flush_pending_square();
    if (factor == 0) throw std::runtime_error("Aevum multiplication factor must be positive");
    const size_t slot = find_prepared(src, true);
#if defined(__APPLE__)
    if (slot == no_prepared) {
      throw std::runtime_error("Apple Aevum multiplication requires set_multiplicand/prepare; unprepared generic tailMul is not used");
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

  struct PreparedSlot {
    size_t reg = no_prepared;
    uint64_t stamp = 0;
  };

  void flush_pending_square() {
    if (pending_reg_ == no_prepared) return;
    const size_t index = pending_reg_;
    const bool lead_in = pending_lead_width_;
    // Clear first so an exception cannot leave a stale transformed-state tag.
    pending_reg_ = no_prepared;
    pending_lead_width_ = false;
    gpu_->regSquareStep(reg(index), lead_in, false);
  }

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
  bool lead_cache_enabled_ = false;
  size_t pending_reg_ = no_prepared;
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

aevum_engine_handle aevum_engine_create(uint32_t exponent, size_t register_count, uint32_t device, int verbose, const char* fft_spec, const char* tune_dir) {
  try {
    g_last_error.clear();
    return new Runtime(exponent, register_count, device, verbose != 0, fft_spec, tune_dir);
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
