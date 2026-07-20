// Copyright (C) Mihai Preda.

#include "FFTConfig.h"
#include "Args.h"
#include "common.h"
#include "log.h"
#include "TuneEntry.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <array>
#include <map>
#include <cinttypes>
#include <cstdlib>
#include <stdexcept>

using namespace std;

struct FftBpw {
  string fft;
  array<float, NUM_BPW_ENTRIES> bpw;
};

map<string, array<float, NUM_BPW_ENTRIES>> BPW {
#include "fftbpw.h"
};

namespace {

bool startsWith(const string& s, const string& prefix) { return s.rfind(prefix, 0) == 0; }

u64 mersenne2ReferencePowerOfTwo(u64 exponent) {
  for (u64 n = 256; n <= (u64(1) << 34); n <<= 1) {
    const double safe = std::log2(double(n)) + 2.0 * (double(exponent) / double(n) + 1.0);
    if (safe < 92.0) return n;
  }
  return 0;
}

u32 parseInt(const string& s) {
  // if (s.empty()) { return 1; }
  assert(!s.empty());
  char c = s.back();
  u32 multiple = c == 'k' || c == 'K' ? 1024 : c == 'm' || c == 'M' ? 1024 * 1024 : 1;
  return u32(strtod(s.c_str(), nullptr) * multiple);
}

} // namespace

// Accepts:
// - a prefix indicating FFT_type (if not specified, default is FP64)
// - a single config: 1K:13:256
// - a size: 6.5M
// - a range of sizes: 6.5M-7M
// - a list: 6M-7M,1K:13:256
vector<FFTShape> FFTShape::multiSpec(const string& iniSpec) {
  if (iniSpec.empty()) { return allShapes(); }

  vector<FFTShape> ret;

  for (const string &spec : split(iniSpec, ',')) {
    enum FFT_TYPES fft_type = FFT3161;
    auto parts = split(spec, ':');
    if (parseInt(parts[0]) < 60) {      // Look for a prefix specifying the FFT type
      fft_type = (enum FFT_TYPES) parseInt(parts[0]);
      parts = vector(next(parts.begin()), parts.end());
    }
    assert(parts.size() <= 3);
    if (parts.size() == 3) {
      u32 width = parseInt(parts[0]);
      u32 middle = parseInt(parts[1]);
      u32 height = parseInt(parts[2]);
      ret.push_back({fft_type, width, middle, height});
      continue;
    }
    assert(parts.size() == 1);

    parts = split(spec, '-');
    assert(parts.size() >= 1 && parts.size() <= 2);
    u32 sizeFrom = parseInt(parts[0]);
    u32 sizeTo = parts.size() == 2 ? parseInt(parts[1]) : sizeFrom;
    auto shapes = allShapes(sizeFrom, sizeTo);
    if (shapes.empty()) {
      log("Could not find a FFT config for '%s'\n", spec.c_str());
      throw "Invalid FFT spec";
    }
    ret.insert(ret.end(), shapes.begin(), shapes.end());
  }
  return ret;
}

vector<FFTShape> FFTShape::allShapes(u32 sizeFrom, u32 sizeTo) {
  vector<FFTShape> configs;
  for (enum FFT_TYPES type : {FFT3161}) {
    for (u32 width : {256, 512, 1024, 4096}) {
      for (u32 height : {256, 512, 1024}) {
        if (width == 256 && height == 1024) { continue; } // Skip because we prefer width >= height
        for (u32 middle : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
          if (type != FFT64 && (middle & (middle - 1))) continue;   // Reject non-power-of-two NTTs
          u32 sz = width * height * middle * 2;
          if (sizeFrom <= sz && sz <= sizeTo) {
            configs.push_back({type, width, middle, height});
          }
        }
      }
    }
  }
  std::sort(configs.begin(), configs.end(),
            [](const FFTShape &a, const FFTShape &b) {
              if (a.size() != b.size()) { return (a.size() < b.size()); }
              if (a.width != b.width) {
                if (a.width == 1024 || b.width == 1024) { return a.width == 1024; }
                return a.width < b.width;
              }
              return a.height < b.height;
            });
  return configs;
}

FFTShape::FFTShape(const string& spec) {
  assert(!spec.empty());
  enum FFT_TYPES fft_type = FFT3161;
  vector<string> v = split(spec, ':');
  if (parseInt(v[0]) < 60) {      // Look for a prefix specifying the FFT type
    fft_type = (enum FFT_TYPES) parseInt(v[0]);
    for (u32 i = 1; i < v.size(); ++i) v[i-1] = v[i];
    v.resize(v.size() - 1);
  }
  assert(v.size() == 3);
  *this = FFTShape{fft_type, v.at(0), v.at(1), v.at(2)};
}

FFTShape::FFTShape(enum FFT_TYPES t, const string& w, const string& m, const string& h) :
  FFTShape{t, parseInt(w), parseInt(m), parseInt(h)}
{}

FFTShape::FFTShape(enum FFT_TYPES t, u32 w, u32 m, u32 h) :
  fft_type{t}, width{w}, middle{m}, height{h} {
  assert(w && m && h);

  // Un-initialized shape, don't set BPW
  if (w == 1 && m == 1 && h == 1) { return; }

  // PFA replaces the power-of-two middle axis by a 3/9 Good-Thomas axis.
  // Type 1 keeps the paired GF31/GF61 arithmetic.  Experimental type 4 adds
  // the upstream FP32 residue plane and currently supports radix 9 only.
  // Until dedicated tune data is collected, inherit the next stock
  // power-of-two plan and keep a conservative capacity margin.
  if ((t == FFT3161 && (m == 3 || m == 9)) || (t == FFT323161 && m == 9)) {
    const u32 reference_middle = m == 3 ? 4 : 16;
    bpw = FFTShape{t, w, reference_middle, h}.bpw;
    for (float& value : bpw) value -= 0.20f;
    return;
  }

  string s = spec();
  if (auto it = BPW.find(s); it != BPW.end()) {
    bpw = it->second;
  } else {
    if (height > width) {
      bpw = FFTShape{t, h, m, w}.bpw;
    } else {
      // Manipulate the shape into something that was likely pre-computed
      u32 orig_w = w;
      u32 orig_m = m;
      u32 orig_h = h;
      while (m < 9) { m *= 2; w /= 2; }
      while (w >= 4*h) { w /= 2; h *= 2; }
      while (w < h || w < 256 || w == 2048) { w *= 2; h /= 2; }
      while (h < 256) { h *= 2; m /= 2; }
      if (m == 1) m = 2;

      // Make up some defaults (should only happen for experimental FFT types (t >= 52)
      if (w == orig_w && m == orig_m && h == orig_h) {
        bpw = {18.1f, 18.1f, 18.1f, 18.1f, 18.1f, 18.1f};
        log("ERROR: BPW info for %s not found, using default of 18.1.\n", s.c_str());
      }

      // Try the modified shape
      else {
        bpw = FFTShape{t, w, m, h}.bpw;
        for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) bpw[j] -= 0.05f;   // Assume this fft spec is worse than measured fft specs
        if (this->isFavoredShape()) {  // Don't output this warning message for non-favored shapes (we expect the BPW info to be missing)
          printf("BPW info for %s not found, defaults={", s.c_str());
          for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) printf("%s%.2f", j ? ", " : "", (double) bpw[j]);
          printf("}\n");
        }
      }
    }
  }
}

float FFTShape::carry32BPW() const {
  // The formula below was validated empirically with -carryTune

  // We observe that FFT 6.5M (1024:13:256) has safe carry32 up to 18.35 BPW
  // while the 0.5*log2() models the impact of FFT size changes.
  // We model carry with a Gumbel distrib similar to the one used for ROE, and measure carry with
  // -use STATS=1. See -carryTune

//GW:  I have no idea why this is needed.  Without it, -tune fails on FFT sizes from 256K to 1M
// Perhaps it has something to do with RNDVALdoubleToLong in carryutil
if (18.35 + 0.5 * (log2(13 * 1024 * 512) - log2(size())) > 19.0) return 19.0f;

  return float(18.35 + 0.5 * (log2(13 * 1024 * 512) - log2(size())));
}

bool FFTShape::needsLargeCarry(u64 E) const {
  return E / double(size()) > carry32BPW();
}

// Return TRUE for "favored" shapes.  That is, those that are most likely to be useful.  To save time in generating bpw data, only these favored
// shapes have their bpw data pre-computed.  Bpw for non-favored shapes is guessed from the bpw data we do have.  Also. -tune will normally only
// time favored shapes.  These are the rules for deciding favored shapes:
//      WIDTH >= HEIGHT
//      WIDTH=4K:  HEIGHT>=512, MIDDLE>=9       (2*8 combos)
//      WIDTH=1K:  MIDDLE>=5                    (3*12 combos)
//      WIDTH=512: MIDDLE>=4                    (2*13 combos)
//      WIDTH=256: MIDDLE>=1                    (16 combos)
bool FFTShape::isFavoredShape() const {
  return width >= height &&
        ((width == 4096 && height >= 512 && middle >= 9) ||
         (width == 1024 && middle >= 5) ||
         (width == 512 && middle >= 4) ||
         (width == 256 && middle >= 1));
}

FFTConfig::FFTConfig(const string& input_spec) {
  string spec = input_spec;
  u32 requested_pfa = 0;
  bool adaptive_type4 = false;
  bool force_full_type4 = false;
  if (startsWith(spec, "pfa3:")) { requested_pfa = 3; spec = spec.substr(5); }
  else if (startsWith(spec, "pfa9full:")) { requested_pfa = 9; force_full_type4 = true; spec = spec.substr(9); }
  else if (startsWith(spec, "pfa9fast:")) { requested_pfa = 9; adaptive_type4 = true; spec = spec.substr(9); }
  else if (startsWith(spec, "pfa9:")) { requested_pfa = 9; spec = spec.substr(5); }
  auto v = split(spec, ':');

  enum FFT_TYPES fft_type = FFT3161;
  if (parseInt(v[0]) < 60) {      // Look for a prefix specifying the FFT type
    fft_type = (enum FFT_TYPES) parseInt(v[0]);
    for (u32 i = 1; i < v.size(); ++i) v[i-1] = v[i];
    v.resize(v.size() - 1);
  }

  // Sanity check the spec
  if (v.size() >= 3) {
    u32 w = parseInt(v[0]);
    u32 m = parseInt(v[1]);
    u32 h = parseInt(v[2]);
    if (w != 256 && w != 512 && w != 1024 && w != 4096) {
      log("Width must be 256, 512, 1024, or 4096.\n");
      throw "Invalid FFT spec";
    }
    if (m < 2 || m > 16) {
      log("Middle must be between 1 and 16.\n");
      throw "Invalid FFT spec";
    }
    if (h != 256 && h != 512 && h != 1024) {
      log("Height must be 256, 512, 1024.\n");
      throw "Invalid FFT spec";
    }
    if (fft_type != FFT64 && fft_type != FFT32 && (m & (m - 1)) &&
        !(requested_pfa && m == requested_pfa)) {
      log("NTT middle must be a power of two unless an explicit pfa3:/pfa9: plan is used.\n");
      throw "Invalid FFT spec";
    }
  }
  
  if (v.size() == 1) {
    *this = {FFTShape::multiSpec(spec).front(), LAST_VARIANT, CARRY_AUTO};
  } if (v.size() == 3) {
    *this = {FFTShape{fft_type, v[0], v[1], v[2]}, LAST_VARIANT, CARRY_AUTO};
  } else if (v.size() == 4) {
    *this = {FFTShape{fft_type, v[0], v[1], v[2]}, parseInt(v[3]), CARRY_AUTO};
  } else if (v.size() == 5) {
    int c = parseInt(v[4]);
    assert(c == 0 || c == 1);
    *this = {FFTShape{fft_type, v[0], v[1], v[2]}, parseInt(v[3]), c == 0 ? CARRY_32 : CARRY_64};
  } else {
    throw "FFT spec";
  }
  if (requested_pfa) {
    const bool paired_ntt = shape.fft_type == FFT3161 &&
                            (requested_pfa == 3 || requested_pfa == 9);
    const bool hybrid_fp32_crt = shape.fft_type == FFT323161 && requested_pfa == 9;
    if ((!paired_ntt && !hybrid_fp32_crt) || shape.middle != requested_pfa)
      throw std::runtime_error("PFA plans require FFT3161 middle 3/9 or FFT323161 middle 9");
    pfa_radix = requested_pfa;
    // A type-4 PFA9 request is capacity-adaptive by default.  The FP32 plane
    // is mathematically redundant whenever the exact GF31*GF61 CRT limit is
    // sufficient, so do not pay for a third transform unless pfa9full: is
    // explicitly requested.  pfa9fast: remains a compatibility alias.
    adaptive_type4_request = shape.fft_type == FFT323161 && !force_full_type4;
    if (adaptive_type4 && shape.fft_type != FFT323161)
      throw std::runtime_error("pfa9fast requires an FFT323161 type-4 request");
    if (force_full_type4 && shape.fft_type != FFT323161)
      throw std::runtime_error("pfa9full requires an FFT323161 type-4 request");
  }
}

FFTConfig::FFTConfig(FFTShape shape, u32 variant, enum CARRY_KIND carry) :
  shape{shape},
  variant{variant},
  carry{carry}
{
  assert(variant_W(variant) < N_VARIANT_W);
  assert(variant_M(variant) < N_VARIANT_M);
  assert(variant_H(variant) < N_VARIANT_H);

  if      (shape.fft_type == FFT64)     FFT_FP64 = 1, FFT_FP32 = 0, NTT_GF31 = 0, NTT_GF61 = 0, WordSize = 4;
  else if (shape.fft_type == FFT3161)   FFT_FP64 = 0, FFT_FP32 = 0, NTT_GF31 = 1, NTT_GF61 = 1, WordSize = 8;
  else if (shape.fft_type == FFT3261)   FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 0, NTT_GF61 = 1, WordSize = 8;
  else if (shape.fft_type == FFT61)     FFT_FP64 = 0, FFT_FP32 = 0, NTT_GF31 = 0, NTT_GF61 = 1, WordSize = 4;
  else if (shape.fft_type == FFT323161) FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 1, NTT_GF61 = 1, WordSize = 8;
  else if (shape.fft_type == FFT3231)   FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 1, NTT_GF61 = 0, WordSize = 4;
  else if (shape.fft_type == FFT6431)   FFT_FP64 = 1, FFT_FP32 = 0, NTT_GF31 = 1, NTT_GF61 = 0, WordSize = 8;
  else if (shape.fft_type == FFT31)     FFT_FP64 = 0, FFT_FP32 = 0, NTT_GF31 = 1, NTT_GF61 = 0, WordSize = 4;
  else if (shape.fft_type == FFT32)     FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 0, NTT_GF61 = 0, WordSize = 4;
  else throw "FFT type";
}

string FFTConfig::spec() const {
  string s = shape.spec() + ":" + to_string(variant_W(variant)) + to_string(variant_M(variant)) + to_string(variant_H(variant));
  if (isPfa()) s = string("pfa") + to_string(pfa_radix) + ":" + s;
  return carry == CARRY_AUTO ? s : (s + (carry == CARRY_32 ? ":0" : ":1"));
}

float FFTConfig::maxBpw() const {
  float b;
  // Look up the pre-computed maximum bpw.  The lookup table contains data for variants 000, 101, 202, 010, 111, 212.
  // For 4K width, the lookup table contains data for variants 100, 101, 202, 110, 111, 212 since BCAST only works for width <= 1024.
  if (variant_W(variant) == variant_H(variant) ||
      (shape.width > 1024 && variant_W(variant) == 1 && variant_H(variant) == 0)) {
    b = shape.bpw[variant_M(variant) * 3 + variant_H(variant)];
  }
  // Interpolate for the maximum bpw.  This might could be improved upon.  However, I doubt people will use these variants often.
  else {
    float b1 = shape.bpw[variant_M(variant) * 3 + variant_W(variant)];
    float b2 = shape.bpw[variant_M(variant) * 3 + variant_H(variant)];
    b = (b1 + b2) / 2.0f;
  }
  // Only some FFTs support both 32 and 64 bit carries.
  return (carry == CARRY_32 && (shape.fft_type == FFT64 || shape.fft_type == FFT3231)) ? std::min(shape.carry32BPW(), b) : b;
}

FFTConfig FFTConfig::bestFit(const Args& args, u64 E, const string& spec) {
  // Native Good-Thomas PFA selector. pfa:auto chooses an odd-radix plan only
  // when the actual Aevum transform-size reduction is large enough to cover
  // the odd-axis work. Forced pfa:3 and pfa:9 remain available for diagnostics.
  if (spec == "pfa:auto" || spec == "pfa:3" || spec == "pfa:9") {
    const u32 forced = spec == "pfa:3" ? 3u : spec == "pfa:9" ? 9u : 0u;
    FFTConfig stock = bestFit(args, E, "");
    // The optimized radix-9 path is selected directly from the real Aevum
    // stock/PFA size ratio. Device tuning can still override tail variants,
    // but is no longer required to enable pfa:auto.
    vector<FFTConfig> candidates;
    for (u32 radix : {9u, 3u}) {
      if (forced && radix != forced) continue;
      for (u32 width : {256u, 512u, 1024u, 4096u}) {
        for (u32 height : {256u, 512u, 1024u}) {
          if (width == 256 && height == 1024) continue;
          // RTX 3080 measurements favor variant 202 for radix-9.  Keep
          // radix-3 on the validated 101 pipeline.
          const u32 variant = radix == 9 ? 202u : 101u;
          FFTConfig c{FFTShape{FFT3161, width, radix, height}, variant, CARRY_AUTO};
          c.pfa_radix = radix;
          const double bpw = E / double(c.size());
          if (bpw < c.minBpw()) continue;
          if (c.maxExp() * args.fftOverdrive >= E) candidates.push_back(c);
        }
      }
    }
    if (candidates.empty()) {
      if (forced) throw std::runtime_error("No admissible native PFA radix-3/9 plan");
      log("Aevum PFA auto: no admissible odd-radix plan; stock %s retained.\n",
          stock.spec().c_str());
      return stock;
    }
    sort(candidates.begin(), candidates.end(), [](const FFTConfig& a, const FFTConfig& b) {
      if (a.size() != b.size()) return a.size() < b.size();
      return a.pfa_radix > b.pfa_radix;
    });
    FFTConfig selected = candidates.front();
    const double ratio = double(stock.size()) / double(selected.size());

    // PFA-9 is enabled automatically for the 1.778x reduction windows that
    // have enough headroom to absorb the odd-axis work. PFA-3 is enabled for the 1.333x reduction windows after
    // the RTX 3080 measurements confirmed a speed win.
    const double minimum = selected.pfa_radix == 9 ? 1.60 : 1.30;
    if (!forced && ratio < minimum) {
      const u64 reference = mersenne2ReferencePowerOfTwo(E);
      if (ratio < 1.0) {
        log("Aevum PFA auto: stock %s (%" PRIu64 " words) retained; candidate %s (%" PRIu64
            " words) is %.3fx larger than the actual Aevum stock plan.\n",
            stock.spec().c_str(), stock.size(), selected.spec().c_str(), selected.size(), 1.0 / ratio);
      } else {
        log("Aevum PFA auto: stock %s (%" PRIu64 " words) retained; candidate %s (%" PRIu64
            " words) reduces size by only %.3fx, below the current %.2fx overhead gate.\n",
            stock.spec().c_str(), stock.size(), selected.spec().c_str(), selected.size(), ratio, minimum);
      }
      if (reference && reference != stock.size()) {
        log("Aevum PFA note: the mersenne2 reference table compares against conservative power-of-two n=%" PRIu64
            "; Aevum's tuned stock selector uses n=%" PRIu64 ".\n", reference, stock.size());
      }
      return stock;
    }
    log("Aevum native PFA: %s selected for exponent %" PRIu64
        " (stock %s, actual Aevum transform reduction %.3fx; auto gate %.2fx).\n",
        selected.spec().c_str(), E, stock.spec().c_str(), ratio, minimum);
    return selected;
  }

  // A FFT-spec was given, simply take the first FFT from the spec that can handle E
  if (!spec.empty()) {
    FFTConfig fft{spec};

    // FFT323161 only pays for itself when its FP32 plane is required to lift
    // the coefficient range beyond the 92-bit GF31*GF61 CRT.  Every ordinary
    // pfa9:4 request is therefore capacity-adaptive: if the exact paired NTT
    // fits, execute the same requested shape/variant without the redundant
    // FP32 plane.  Use pfa9full:4 only for the true three-plane diagnostic path.
    if (fft.adaptive_type4_request) {
      FFTConfig paired{FFTShape{FFT3161, fft.shape.width, fft.shape.middle, fft.shape.height},
                       fft.variant, fft.carry};
      paired.pfa_radix = fft.pfa_radix;
      paired.adaptive_type4_request = true;
      const double requested_bpw = E / double(fft.size());
      const double paired_limit = paired.maxBpw() * args.fftOverdrive;
      if (paired.maxExp() * args.fftOverdrive >= E) {
        paired.adaptive_type4_elided = true;
        log("Aevum optimized type-4 PFA9: FP32 plane elided at %.2f bpw; "
            "exact FFT3161 limit %.2f bpw, executing %s.\n",
            requested_bpw, paired_limit, paired.spec().c_str());
        return paired;
      }
      log("Aevum optimized type-4 PFA9: %.2f bpw exceeds exact FFT3161 limit "
          "%.2f bpw; retaining full FP32+GF31+GF61 plan %s.\n",
          requested_bpw, paired_limit, fft.spec().c_str());
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
    if (!supported_aevum_type && !apple_diagnostic_plane) {
      log("Aevum accepts FFT type 1 or FFT type 4 (power-of-two or explicit PFA9).\n");
      throw "Aevum FFT type";
    }
#if defined(__APPLE__)
    if (apple_diagnostic_plane) {
      log("Apple Aevum diagnostic: explicit single-plane FFT type %d enabled.\n",
          static_cast<int>(fft.shape.fft_type));
    }
#endif
    if (fft.maxExp() * args.fftOverdrive < E) {
      log("Warning: %s (max %" PRIu64 ") may be too small for %" PRIu64 "\n", fft.spec().c_str(), fft.maxExp(), E);
    }
    return fft;
  }

  // The standard GPUOwl tune file contains FP64 shapes, including non-power-of-two
  // middle dimensions.  Aevum is NTT-only, so select directly from FFT3161 shapes.
  for (const FFTShape& shape : FFTShape::allShapes()) {
    if (shape.fft_type != FFT3161) continue;
    for (u32 v : {101, 202}) {
      FFTConfig fft{shape, v, CARRY_AUTO};
      const double bits_per_word = E / double(shape.size());
      if (bits_per_word < fft.minBpw()) continue;
      if (fft.maxExp() * args.fftOverdrive >= E) {
        log("Aevum auto FFT: %s for exponent %" PRIu64 "\n", fft.spec().c_str(), E);
        return fft;
      }
    }
  }

  log("No admissible FFT3161 plan for exponent %" PRIu64 "\n", E);
  throw "No admissible FFT3161 plan";
}


string numberK(u64 n) {
  u32 K = 1024;
  u32 M = K * K;

  if (n % M == 0) { return to_string(n / M) + 'M'; }

  char buf[64];
  if (n >= M && (n * u64(100)) % M == 0) {
    snprintf(buf, sizeof(buf), "%.2f", float(n) / M);
    return string(buf) + 'M';
  } else if (n >= K) {
    snprintf(buf, sizeof(buf), "%g", float(n) / K);
    return string(buf) + 'K';
  } else {
    return to_string(n);
  }
}
