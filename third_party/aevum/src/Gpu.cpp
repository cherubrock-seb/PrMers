// Copyright (C) Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
#include "TimeInfo.h"
#include "Trig.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "Queue.h"
#include "Task.h"
#include "KernelCompiler.h"
#include "Saver.h"
#include "timeutil.h"
#include "TrigBufCache.h"
#include "fs.h"
#include "Sha3Hash.h"

#include <algorithm>
#include <bitset>
#include <limits>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <array>
#include <cinttypes>
#include <cstring>
#include <cstdlib>
#include <cinttypes>

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#ifndef M_LN2l
#define M_LN2l 0.69314718055994530941723212145818L
#endif

#ifndef M_LN2
#define M_LN2 0.69314718055994530941723212145818
#endif

#define CARRY_LEN 8

namespace {

u32 kAt(u32 H, u32 line, u32 col) { return (line + col * H) * 2; }

double weight(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l((long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double invWeight(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l(-(long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

// MSVC does not truly support long double.  Use expm1 rather than exp2 and subtracting one.
double weightM1(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1l(M_LN2l * (long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double invWeightM1(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1l(M_LN2l * - (long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double boundUnderOne(double x) { return std::min(x, nexttoward(1, 0)); }

float weight32(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2((double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

float invWeight32(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2(-(double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

float weightM132(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1(M_LN2 * (double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

float invWeightM132(u32 N, u64 E, u32 H, u32 line, u32 col, u32 rep) {
  return expm1(M_LN2 * - (double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

Weights genWeights(FFTConfig fft, u64 E, u32 W, u32 H, u32 nW, bool nvidiaGpu) {
  u32 N = 2u * W * H;
  u32 groupWidth = W / nW;

  vector<double> weightsConstIF;
  vector<double> weightsIF;

  if (fft.FFT_FP64) {
    // Inverse + Forward
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      auto iw = invWeight(N, E, H, 0, thread, 0);
      auto w = weight(N, E, H, 0, thread, 0);
      weightsIF.push_back(2 * boundUnderOne(iw));
      weightsIF.push_back(2 * w);
    }

    // the group order matches CarryA/M (not fftP/CarryFused).
    for (u32 gy = 0; gy < H; ++gy) {
      weightsIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
      weightsIF.push_back(weightM1(N, E, H, gy, 0, 0));
    }

    // nVidia GPUs have a fast constant cache that only works on buffer sizes less than 64KB.  Create two smaller buffers
    // that can be used to create the large group order buffer with a single multiply.
    if (nvidiaGpu) {
      for (u32 gy = 0; gy < 64; ++gy) {
        weightsConstIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
        weightsConstIF.push_back(weightM1(N, E, H, gy, 0, 0));
      }
      for (u32 gy = 0; gy < H; gy += 64) {
        weightsConstIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
        weightsConstIF.push_back(weightM1(N, E, H, gy, 0, 0));
      }
    }
  }

  else if (fft.FFT_FP32) {
    vector<float> weightsConstIF32;
    vector<float> weightsIF32;
    // Inverse + Forward
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      auto iw = invWeight32(N, E, H, 0, thread, 0) ;
      auto w = weight32(N, E, H, 0, thread, 0) ;
      // Weights are scaled by 2^-24 and 2^48 so that multiplicaton by 1/epsilon does not generate infinty results (width and height variant 2).
      iw = iw * 281474976710656.0f;
      w = w * 0.000000059604644775390625f;
      weightsIF32.push_back(iw);
      weightsIF32.push_back(w);
    }

    // the group order matches CarryA/M (not fftP/CarryFused).
    for (u32 gy = 0; gy < H; ++gy) {
      weightsIF32.push_back(invWeightM132(N, E, H, gy, 0, 0));
      weightsIF32.push_back(weightM132(N, E, H, gy, 0, 0));
    }

    // nVidia GPUs have a fast constant cache that only works on buffer sizes less than 64KB.  Create two smaller buffers
    // that can be used to create the large group order buffer with a single multiply.
    if (nvidiaGpu) {
      for (u32 gy = 0; gy < 64; ++gy) {
        weightsConstIF32.push_back(invWeightM132(N, E, H, gy, 0, 0));
        weightsConstIF32.push_back(weightM132(N, E, H, gy, 0, 0));
      }
      for (u32 gy = 0; gy < H; gy += 64) {
        weightsConstIF32.push_back(invWeightM132(N, E, H, gy, 0, 0));
        weightsConstIF32.push_back(weightM132(N, E, H, gy, 0, 0));
      }
    }

    // Copy the float vectors to the double vectors
    weightsIF.resize(weightsIF32.size() / 2);
    memcpy((double *) weightsIF.data(), weightsIF32.data(), weightsIF32.size() * sizeof(float));
    weightsConstIF.resize(weightsConstIF32.size() / 2);
    memcpy((double *) weightsConstIF.data(), weightsConstIF32.data(), weightsConstIF32.size() * sizeof(float));
  }

  return Weights{weightsConstIF, weightsIF};
}

string toLiteral(i32 value) { return to_string(value); }
string toLiteral(u32 value) { return to_string(value) + 'u'; }
[[maybe_unused]] string toLiteral(long value) { return to_string(value) + "ll"; }
[[maybe_unused]] string toLiteral(unsigned long value) { return to_string(value) + "ull"; }
[[maybe_unused]] string toLiteral(long long value) { return to_string(value) + "ll"; }
[[maybe_unused]] string toLiteral(unsigned long long value) { return to_string(value) + "ull"; }

template<typename F>
string floatingLiteral(F value) {
  static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
  std::ostringstream ss;
  ss << std::setprecision(numeric_limits<F>::max_digits10) << value;
  if constexpr (std::is_same_v<F, float>) ss << "f";
  string s = std::move(ss).str();

  // Verify an exact round-trip without relying on overload resolution between
  // integer and floating-point toLiteral() overloads.
  [[maybe_unused]] F back = 0;
  if constexpr (std::is_same_v<F, float>) {
    sscanf(s.c_str(), "%f", &back);
  } else {
    sscanf(s.c_str(), "%lf", &back);
  }
  assert(back == value);

  return s;
}

string toLiteral(float value) { return floatingLiteral(value); }
string toLiteral(double value) { return floatingLiteral(value); }

template<typename T>
string toLiteral(const vector<T>& v) {
  assert(!v.empty());
  string s = "{";
  for (auto x : v) {
    s += toLiteral(x) + ",";
  }
  s += "}";
  return s;
}

template<typename T, size_t N>
string toLiteral(const std::array<T, N>& v) {
  string s = "{";
  for (T x : v) {
    s += toLiteral(x) + ",";
  }
  s += "}";
  return s;
}

string toLiteral(const string& s) { return s; }

[[maybe_unused]] string toLiteral(float2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(double2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(int2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(uint2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }
[[maybe_unused]] string toLiteral(ulong2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }

template<typename T>
string toDefine(const string& k, T v) { return " -D"s + k + '=' + toLiteral(v); }

template<typename T>
string toDefine(const T& vect) {
  string s;
  for (const auto& [k, v] : vect) { s += toDefine(k, v); }
  return s;
}

constexpr bool isInList(const string& s, initializer_list<string> list) {
  for (const string& e : list) { if (e == s) { return true; }}
  return false;
}

string clDefines(const Args& args, cl_device_id id, FFTConfig fft, const vector<KeyVal>& extraConf, u64 E, bool doLog,
                 bool &tail_single_wide, bool &tail_single_kernel, u32 &in_place, u32 &pad_size, u32 &wmul) {
  map<string, string> config;

  // Highest priority is the requested "extra" conf
  config.insert(extraConf.begin(), extraConf.end());

  // Next, args config
  config.insert(args.flags.begin(), args.flags.end());

  // Lowest priority: the per-FFT config if any
  if (auto it = args.perFftConfig.find(fft.shape.spec()); it != args.perFftConfig.end()) {
    // log("Found %s\n", fft.shape.spec().c_str());
    config.insert(it->second.begin(), it->second.end());
  }

#if defined(__APPLE__)
  // Keep the stock LDS transpose algorithms on Apple, but lower the middle
  // work-group pressure for the legacy OpenCL-to-Metal compiler. Explicit
  // command-line/per-FFT values above always win over these defaults.
  config.try_emplace("IN_WG", "64");
  config.try_emplace("OUT_WG", "64");
  config.try_emplace("IN_SIZEX", "16");
  config.try_emplace("OUT_SIZEX", "16");
  config.try_emplace("MIDDLE_IN_LDS_TRANSPOSE", "1");
  config.try_emplace("MIDDLE_OUT_LDS_TRANSPOSE", "1");
  config.try_emplace("NO_ASM", "1");
  config.try_emplace("PAD", "0");
#endif

  // Default value for -use options that must also be parsed in C++ code
  tail_single_wide = 0, tail_single_kernel = 1;         // Default tailSquare is double-wide in one kernel
  in_place = 0;                                         // Default is not in-place
  wmul = 2;                                             // Default is carryFused processes two lines at a time
  pad_size = isAmdGpu(id) ? 256 : 0;                    // Default is 256 bytes for AMD, 0 for others

  // Validate -use options
  for (const auto& [k, v] : config) {
    bool isValid = isInList(k, {
                              "FAST_BARRIER",
                              "STATS",
                              "IN_SIZEX",
                              "IN_WG",
                              "OUT_SIZEX",
                              "OUT_WG",
                              "UNROLL_H",
                              "UNROLL_W",
                              "ZEROHACK_H",
                              "ZEROHACK_W",
                              "NO_ASM",
                              "DEBUG",
                              "CARRY64",
                              "BIGLIT",                 // Deprecated
                              "NONTEMPORAL",            // Deprecated
                              "INPLACE",
                              "PAD",
                              "MIDDLE_IN_LDS_TRANSPOSE",
                              "MIDDLE_OUT_LDS_TRANSPOSE",
                              "TAIL_KERNELS",
                              "TAIL_TRIGS",
                              "TAIL_TRIGS31",
                              "TAIL_TRIGS32",
                              "TAIL_TRIGS61",
                              "TABMUL_CHAIN",
                              "TABMUL_CHAIN31",
                              "TABMUL_CHAIN32",
                              "TABMUL_CHAIN61",
                              "MODM31",
                              "LOADS","STORES",
                              "NOREG",                  // CUDA - experimental
                              "WMUL"
                            });
    if (!isValid) {
      log("Warning: unrecognized -use key '%s'\n", k.c_str());
    }

    // Some -use options are needed in both OpenCL code and C++ initialization code
    if (k == "TAIL_KERNELS") {
      if (atoi(v.c_str()) == 0) tail_single_wide = 1, tail_single_kernel = 1;
      if (atoi(v.c_str()) == 1) tail_single_wide = 1, tail_single_kernel = 0;
      if (atoi(v.c_str()) == 2) tail_single_wide = 0, tail_single_kernel = 1;
      if (atoi(v.c_str()) == 3) tail_single_wide = 0, tail_single_kernel = 0;
    }
    if (k == "INPLACE") in_place = atoi(v.c_str());
    if (k == "WMUL") wmul = atoi(v.c_str());
    if (k == "PAD") pad_size = atoi(v.c_str());
  }

#if defined(__APPLE__)
  // Apple's OpenCL-to-Metal compiler rejects the combined double-wide
  // tailSquareGF61 pipeline (TAIL_KERNELS=2) after successfully creating the
  // staged GF61 middle-in chain.  Keep the same double-wide LDS algorithm,
  // but split the two exceptional lines (0 and H/2) into tailSquareZeroGF61.
  // TAIL_KERNELS=3 is an existing upstream execution mode; only Apple FFT3161
  // is forced to it.  Linux, Windows and CUDA retain the requested/default mode.
  if (fft.shape.fft_type == FFT3161 && (tail_single_wide || tail_single_kernel)) {
    log("Apple OpenCL 1.2 compatibility: forcing TAIL_KERNELS=3 (double-wide, two kernels) for GF61 tailSquare.\n");
    tail_single_wide = false;
    tail_single_kernel = false;
    config["TAIL_KERNELS"] = "3";
  }
#endif

#if defined(__APPLE__)
  // The staged GF61 middle-in path uses the existing non-in-place scratch/output
  // ping-pong. Keep Apple FFT3161 on that validated layout even if a tune file
  // requests INPLACE=1. Other platforms and FFT types retain the request.
  if (fft.shape.fft_type == FFT3161 && in_place != 0) {
    log("Apple OpenCL 1.2 compatibility: forcing INPLACE=0 for staged GF61 middle-in.\n");
    in_place = 0;
    config["INPLACE"] = "0";
  }
#endif

  // Maximum WMUL is 32KB / (WIDTH * SHUFL_BYTES_W).  If using the 32KB maximum, LDS padding must be disabled.
  // Furthermore, I've seen the CUDA compiler refuse to create a kernel with 1024 threads.  Thus, we limit WMUL to 2 for a 1K width and to 1 for a 4K width.
  {
    u32 shufl_bytes_w = args.value("SHUFL_BYTES_W", 8);
    u32 max_wmul = 32768 / (fft.shape.width * shufl_bytes_w);
    if (max_wmul > 2 && fft.shape.width >= 1024) max_wmul = 2;
    if (max_wmul > 1 && fft.shape.width >= 4096) max_wmul = 1;
    if (wmul > max_wmul) {
      wmul = max_wmul;
      config["WMUL"] = to_string(wmul);
      log("WMUL setting too large for this FFT width.  Changing to WMUL=%d\n", wmul);
    }
    if (fft.shape.width * shufl_bytes_w * wmul >= 32768) {
      log("Local shared memory limit of 32KB exceeded.  Changing to LDSPAD_W=0\n");
      config["LDSPAD_W"] = to_string(0);
    }
  }

  string defines = toDefine(config);
  if (doLog) { log("config: %s\n", defines.c_str()); }

  defines += toDefine(initializer_list<pair<string, u32>>{
                    {"EXP", E},
                    {"WIDTH", fft.shape.width},
                    {"SMALL_HEIGHT", fft.shape.height},
                    {"MIDDLE", fft.shape.middle},
                    {"CARRY_LEN", CARRY_LEN},
                    {"NW", fft.shape.nW()},
                    {"NH", fft.shape.nH()}
                  });

  if (isAmdGpu(id)) { defines += toDefine("AMDGPU", 1); }
  if (isNvidiaGpu(id)) { defines += toDefine("NVIDIAGPU", 1); }
  if (isNvidiaGpu(id)) { defines += toDefine("CC", getNvidiaComputeCapability(id)); }

  if ((fft.carry == CARRY_AUTO && fft.shape.needsLargeCarry(E)) || (fft.carry == CARRY_64)) {
    if (doLog) { log("Using CARRY64\n"); }
    defines += toDefine("CARRY64", 1);
  }

  u32 N = fft.shape.size();
  defines += toDefine("FFT_VARIANT", fft.variant);
  defines += toDefine("MAXBPW", (u32)(fft.maxBpw() * 100.0f));

  if (fft.FFT_FP64 || fft.FFT_FP32) {
    defines += toDefine("WEIGHT_STEP", weightM1(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1));
    defines += toDefine("IWEIGHT_STEP", invWeightM1(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1));
    if (fft.FFT_FP64) defines += toDefine("TAILT", root1Fancy(fft.shape.height * 2, 1));
    else defines += toDefine("TAILT", root1FancyFP32(fft.shape.height * 2, 1));

    TrigCoefs coefs = trigCoefs(fft.shape.size() / 4);
    defines += toDefine("TRIG_SCALE", int(coefs.scale));
    defines += toDefine("TRIG_SIN",  coefs.sinCoefs);
    defines += toDefine("TRIG_COS",  coefs.cosCoefs);
  }
  if (fft.NTT_GF31) {
    defines += toDefine("TAILTGF31", root1GF31(fft.shape.height * 2, 1));
  }
  if (fft.NTT_GF61) {
    defines += toDefine("TAILTGF61", root1GF61(fft.shape.height * 2, 1));
  }

  // Send the FFT/NTT type and booleans that enable/disable code for each possible FP and NTT
  defines += toDefine("FFT_TYPE", (int) fft.shape.fft_type);
  defines += toDefine("WordSize", fft.WordSize);

  // When using multiple NTT primes or hybrid FFT/NTT, each FFT/NTT prime's data buffer and trig values are combined into one buffer.
  // The openCL code needs to know the offset to the data and trig values.  Distances are in "number of double2 values".
  if (fft.FFT_FP64 && fft.NTT_GF31) {
    // GF31 data is located after the FP64 data.  Compute size of the FP64 data and trigs.
    defines += toDefine("DISTGF31",      FP64_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF31", SMALLTRIG_FP64_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF31", MIDDLETRIG_FP64_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF31", SMALLTRIGCOMBO_FP64_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.FFT_FP32 && fft.NTT_GF31 && fft.NTT_GF61) {
    // GF31 and GF61 data is located after the FP32 data.  Compute size of the FP32 data and trigs.
    u32 sz1, sz2, sz3, sz4;
    defines += toDefine("DISTGF31",      sz1 = FP32_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF31", sz2 = SMALLTRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF31", sz3 = MIDDLETRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF31", sz4 = SMALLTRIGCOMBO_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTGF61",      sz1 + GF31_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF61", sz2 + SMALLTRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF61", sz3 + MIDDLETRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF61", sz4 + SMALLTRIGCOMBO_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.FFT_FP32 && fft.NTT_GF31) {
    // GF31 data is located after the FP32 data.  Compute size of the FP32 data and trigs.
    defines += toDefine("DISTGF31",      FP32_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF31", SMALLTRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF31", MIDDLETRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF31", SMALLTRIGCOMBO_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.FFT_FP32 && fft.NTT_GF61) {
    // GF61 data is located after the FP32 data.  Compute size of the FP32 data and trigs.
    defines += toDefine("DISTGF61",      FP32_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF61", SMALLTRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF61", MIDDLETRIG_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF61", SMALLTRIGCOMBO_FP32_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.NTT_GF31 && fft.NTT_GF61) {
    defines += toDefine("DISTGF31",      0);
    defines += toDefine("DISTWTRIGGF31", 0);
    defines += toDefine("DISTMTRIGGF31", 0);
    defines += toDefine("DISTHTRIGGF31", 0);
    // GF61 data is located after the GF31 data.  Compute size of the GF31 data and trigs.
    defines += toDefine("DISTGF61",      GF31_DATA_SIZE(fft.shape.width, fft.shape.middle, fft.shape.height, in_place, pad_size) / 2);
    defines += toDefine("DISTWTRIGGF61", SMALLTRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
    defines += toDefine("DISTMTRIGGF61", MIDDLETRIG_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height));
    defines += toDefine("DISTHTRIGGF61", SMALLTRIGCOMBO_GF31_DIST(fft.shape.width, fft.shape.middle, fft.shape.height, fft.shape.nH()));
  }
  else if (fft.NTT_GF31) {
    defines += toDefine("DISTGF31",      0);
    defines += toDefine("DISTWTRIGGF31", 0);
    defines += toDefine("DISTMTRIGGF31", 0);
    defines += toDefine("DISTHTRIGGF31", 0);
  }
  else if (fft.NTT_GF61) {
    defines += toDefine("DISTGF61",      0);
    defines += toDefine("DISTWTRIGGF61", 0);
    defines += toDefine("DISTMTRIGGF61", 0);
    defines += toDefine("DISTHTRIGGF61", 0);
  }

  // Calculate fractional bits-per-word = (E % N) / N * 2^64
  u32 bpw_hi = (u64(E % N) << 32) / N;
  u32 bpw_lo = (((u64(E % N) << 32) % N) << 32) / N;
  u64 bpw = (u64(bpw_hi) << 32) + bpw_lo;
  bpw--; // bpw must not be an exact value -- it must be less than exact value to get last biglit value right
  defines += toDefine("FRAC_BPW_HI", (u32) (bpw >> 32));
  defines += toDefine("FRAC_BPW_LO", (u32) bpw);

  return defines;
}

template<typename T>
pair<vector<T>, vector<T>> split(const vector<T>& v, const vector<u32>& select) {
  vector<T> a;
  vector<T> b;
  auto selIt = select.begin();
  u32 selNext = selIt == select.end() ? u32(-1) : *selIt;
  for (u32 i = 0; i < v.size(); ++i) {
    if (i == selNext) {
      b.push_back(v[i]);
      ++selIt;
      selNext = selIt == select.end() ? u32(-1) : *selIt;
    } else {
      a.push_back(v[i]);
    }
  }
  return {a, b};
}

RoeInfo roeStat(const vector<float>& roe) {
  double sumRoe = 0;
  double sum2Roe = 0;
  double maxRoe = 0;

  for (auto xf : roe) {
    double x = xf;
    assert(x >= 0);
    maxRoe = max(x, maxRoe);
    sumRoe  += x;
    sum2Roe += x * x;
  }
  u32 n = roe.size();

  double sdRoe = sqrt(n * sum2Roe - sumRoe * sumRoe) / n;
  double meanRoe = sumRoe / n;

  return {n, maxRoe, meanRoe, sdRoe};
}

class IterationTimer {
  Timer timer;
  u64 kStart;

public:
  explicit IterationTimer(u64 kStart) : kStart(kStart) { }

  float reset(u64 k) {
    float secs = timer.reset();

    u64 its = max(u64(1), k - kStart);
    kStart = k;
    return secs / its;
  }
};

u32 baseCheckStep(u32 blockSize) {
  switch (blockSize) {
    case 200:  return 40'000;
    case 400:  return 160'000;
    case 500:  return 200'000;
    case 1000: return 1'000'000;
    default:
      assert(false);
      return 0;
  }
}

u32 checkStepForErrors(u32 blockSize, u32 nErrors) {
  u32 step = baseCheckStep(blockSize);
  return nErrors ? step / 2 : step;
}

string toHex(u32 x) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%08x", x);
  return buf;
}

string toHex(const vector<u32>& v) {
  string s;
  for (auto it = v.rbegin(), end = v.rend(); it != end; ++it) {
    s += toHex(*it);
  }
  return s;
}

string formatSecsPerIter(float secsPerIter) {
  char buf[64];
  float usecsPerIter = secsPerIter * 1.0e6f;    // Convert to micro-seconds
  if (usecsPerIter > 1000.0f) {
    snprintf(buf, sizeof(buf), "%4.0f", usecsPerIter);
  } else {
    snprintf(buf, sizeof(buf), "%6.1f", usecsPerIter);
  }
  return string(buf);  
}

} // namespace

// --------

unique_ptr<Gpu> Gpu::make(u64 E, GpuCommon shared, FFTConfig fftConfig, const vector<KeyVal>& extraConf, bool logFftSize) {
  return make_unique<Gpu>(shared, fftConfig, E, extraConf, logFftSize);
}

Gpu::~Gpu() {
  // Background tasks may have captured *this*, so wait until those are complete before destruction
  background->waitEmpty();
}

// Part of GPU initialization is to compute the default number of registers each kernel should target during compilation.
// Kernel register usage is critical for maximizing GPU occupancy.  The default values can be overrriden with command line arguments.
// This feature currently only works for the CUDA compiler.
string Gpu::numCudaRegisters(enum WHICH_KERNEL which_kernel) {
#if CUDA_BACKEND
  int regs = 0;
  const char *use_override = "";
  // Allow command line to prefer the CUDA compiler's default number of registers
  if (args.value("NOREG", 0)) return string("");
  // Determine a kernel specific default maximum number of GPU registers (values set to -1 have not been tuned for best default value)
  switch (which_kernel) {
  case CARRYFUSED:         // Register usage depends on NW, the FFT/NTT type, and perhaps the long carry setting
    switch (fft.shape.fft_type) {
    case FFT64:
      regs = nW == 8 ? 80 : 64;
      use_override = "REGCF64";
      break;
    case FFT3161:
      regs = nW == 8 ? 96 : 64;
      use_override = "REGCF3161";
      break;
    case FFT3261:
      regs = nW == 8 ? 96 : 64;
      use_override = "REGCF3261";
      break;
    case FFT61:
      regs = nW == 8 ? 80 : 64;
      use_override = "REGCF61";
      break;
    case FFT323161:
      regs = nW == 8 ? 128 : 80;
      use_override = "REGCF323161";
      break;
    case FFT3231:
      regs = -1;
      use_override = "REGCF3231";
      break;
    case FFT6431:
      regs = -1;
      use_override = "REGCF6431";
      break;
    case FFT31:
      regs = -1;
      use_override = "REGCF31";
      break;
    case FFT32:
      regs = -1;
      use_override = "REGCF32";
      break;
    }
    break;
  case MIDIN:              // Register usage depends on MIDDLE and the FP32/FP64
    if (fft.FFT_FP64) {
      if (fft.shape.middle >= 16) regs = 96;
      else if (fft.shape.middle >= 14) regs = 88;
      else if (fft.shape.middle >= 13) regs = 80;
      else if (fft.shape.middle >= 10) regs = 72;
      else if (fft.shape.middle >= 7) regs = 64;
      else if (fft.shape.middle >= 5) regs = 56;
      else if (fft.shape.middle >= 4) regs = 48;
      else regs = -1;
      use_override = "REGMI64";
    } else {
      if (fft.shape.middle == 16) regs = 56;
      else if (fft.shape.middle == 8) regs = 40;
      else if (fft.shape.middle == 4) regs = 32;
      else regs = -1;
      use_override = "REGMI32";
    }
    break;
  case MIDIN31:            // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 56;
    else if (fft.shape.middle == 8) regs = 44;
    else if (fft.shape.middle == 4) regs = 32;
    else regs = -1;  
    use_override = "REGMI31";
    break;
  case MIDIN61:            // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 96;
    else if (fft.shape.middle == 8) regs = 72;
    else if (fft.shape.middle == 4) regs = 64;
    else regs = -1;  
    use_override = "REGMI61";
    break;
  case TAIL:               // Register usage depends on NH and the FP32/FP64 (assumes double-wide kernel)
    if (fft.FFT_FP64) {
      regs = nH == 8 ? 88 : 64;
      use_override = "REGTS64";
    } else {
      regs = nH == 8 ? 64 : 48;
      use_override = "REGTS32";
    }
    break;
  case TAIL31:             // Register usage depends on NH (assumes double-wide kernel)
    regs = nH == 8 ? 64 : 48;
    use_override = "REGTS31";
    break;
  case TAIL61:             // Register usage depends on NH (assumes double-wide kernel)
    regs = nH == 8 ? 96 : 64;
    use_override = "REGTS61";
    break;
  case MIDOUT:             // Register usage depends on MIDDLE and the FFT/NTT type
    if (fft.FFT_FP64) {
      if (fft.shape.middle >= 15) regs = 96;
      else if (fft.shape.middle >= 14) regs = 88;
      else if (fft.shape.middle >= 11) regs = 80;
      else if (fft.shape.middle >= 10) regs = 72;
      else if (fft.shape.middle >= 7) regs = 64;
      else if (fft.shape.middle >= 5) regs = 56;
      else if (fft.shape.middle >= 4) regs = 48;
      else regs = -1;
      use_override = "REGMO64";
    } else {
      if (fft.shape.middle == 16) regs = 56;
      else if (fft.shape.middle == 8) regs = 40;
      else if (fft.shape.middle == 4) regs = 32;
      else regs = -1;
      use_override = "REGMO32";
    }
    break;
  case MIDOUT31:           // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 48;
    else if (fft.shape.middle == 8) regs = 40;
    else if (fft.shape.middle == 4) regs = 32;
    else regs = -1;
    use_override = "REGMO31";
    break;
  case MIDOUT61:           // Register usage depends on MIDDLE
    if (fft.shape.middle == 16) regs = 96;
    else if (fft.shape.middle == 8) regs = 64;
    else if (fft.shape.middle == 4) regs = 64;
    else regs = -1;
    use_override = "REGMO61";
    break;
  }
  // Get the optional override register count
  int override_regs = args.value(use_override, 0);
  // If a specified override is small, use the count as a CUDA launch_bounds rather than a maximum register count
  if (override_regs && (override_regs > 0 && override_regs <= 16)) return string("-DCUDA_MIN_BLOCKS=") + to_string(override_regs) + " ";
  // If specified, override the default maximum register count
  if (override_regs) regs = override_regs;
  // Sometimes the results using CUDA compiler's default launch_bounds without setting an explicit launch bounds or maxrrregcount can't be beat
  if (regs == -1) return string("");
  // Format an explicit register count setting
  return string("--maxrregcount=") + to_string(regs) + " ";
#else
  return string("");
#endif
}

// Kernels are compiled one at a time, but OpenCL source files contain multiple kernels.  This routine set the #defines necessary so that only one kernel is compiled.
// While not strictly necessary, startup speed will be a bit faster if we do less compilations.
string Gpu::kernelDefines(enum WHICH_KERNEL_TYPE which_kernel) {
  string defines;
  // Determine the kernel specific #defines
  switch (which_kernel) {
  case KFP:         // FP64 or FP32 kernel
    defines += toDefine("FFT_FP64", (int) fft.FFT_FP64);
    defines += toDefine("FFT_FP32", (int) fft.FFT_FP32);
    defines += toDefine("NTT_GF31", 0);
    defines += toDefine("NTT_GF61", 0);
    break;
  case K31:         // GF1 kernel
    defines += toDefine("FFT_FP64", 0);
    defines += toDefine("FFT_FP32", 0);
    defines += toDefine("NTT_GF31", (int) fft.NTT_GF31);
    defines += toDefine("NTT_GF61", 0);
    break;
  case K61:         // GF61 kernel
    defines += toDefine("FFT_FP64", 0);
    defines += toDefine("FFT_FP32", 0);
    defines += toDefine("NTT_GF31", 0);
    defines += toDefine("NTT_GF61", (int) fft.NTT_GF61);
    break;
  case KALL:        // Kernels, like carryFused, that need all defines set properly
    defines += toDefine("FFT_FP64", (int) fft.FFT_FP64);
    defines += toDefine("FFT_FP32", (int) fft.FFT_FP32);
    defines += toDefine("NTT_GF31", (int) fft.NTT_GF31);
    defines += toDefine("NTT_GF61", (int) fft.NTT_GF61);
    break;
  }
  return defines + " ";
}

#define ROE_SIZE 100000
#define CARRY_SIZE 100000

Gpu::Gpu(GpuCommon s, FFTConfig fft, u64 E, const vector<KeyVal>& extraConf, bool logFftSize) :
  shared(s),
  background{shared.background},
  args{*shared.args},
  E(E),
  N(fft.shape.size()),
  fft(fft),
  WIDTH(fft.shape.width),
  SMALL_H(fft.shape.height),
  BIG_H(SMALL_H * fft.shape.middle),
  hN(N / 2),
  nW(fft.shape.nW()),
  nH(fft.shape.nH()),
  useLongCarry{args.carry == CARRY_64},
  queue{*shared.context, args.profile},
  auxQueues{},    
  compiler{args, shared.context, clDefines(args, shared.context->deviceId(), fft, extraConf, E, logFftSize, tail_single_wide, tail_single_kernel, in_place, pad_size, wmul)},

#define K(name, ...) name(#name, &compiler, profile.make(#name), &queue, __VA_ARGS__)

  K(kfftMidIn,             "fftmiddlein.cl",  "fftMiddleIn",  hN / (BIG_H / SMALL_H), (kernelDefines(KFP) + numCudaRegisters(MIDIN)).c_str()),
  K(kfftHin,               "ffthin.cl",  "fftHin",  hN / nH, kernelDefines(KFP).c_str()),
  K(ktailSquareZero,       "tailsquare.cl", "tailSquareZero", SMALL_H / nH * 2, kernelDefines(KFP).c_str()),
  K(ktailSquare,           "tailsquare.cl", "tailSquare",
                                               !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2, (kernelDefines(KFP) + numCudaRegisters(TAIL)).c_str()),    // Single-wide tailSquare with one kernel
  K(ktailMul,              "tailmul.cl", "tailMul", hN / nH / 2, kernelDefines(KFP).c_str()),
  K(ktailMulLow,           "tailmul.cl", "tailMul", hN / nH / 2, (kernelDefines(KFP) + "-DMUL_LOW=1").c_str()),
  K(kfftMidOut,            "fftmiddleout.cl", "fftMiddleOut", hN / (BIG_H / SMALL_H), (kernelDefines(KFP) + numCudaRegisters(MIDOUT)).c_str()),
  K(kfftW,                 "fftw.cl", "fftW", hN / nW, kernelDefines(KFP).c_str()),

  K(kfftMidInGF31,         "fftmiddlein.cl",  "fftMiddleInGF31",  hN / (BIG_H / SMALL_H), (kernelDefines(K31) + numCudaRegisters(MIDIN31)).c_str()),
  K(kfftHinGF31,           "ffthin.cl",  "fftHinGF31",  hN / nH, kernelDefines(K31).c_str()),
  K(ktailSquareZeroGF31,   "tailsquare.cl", "tailSquareZeroGF31", SMALL_H / nH * 2, kernelDefines(K31).c_str()),
  K(ktailSquareGF31,       "tailsquare.cl", "tailSquareGF31",
                                               !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2, (kernelDefines(K31) + numCudaRegisters(TAIL31)).c_str()),  // Single-wide tailSquare with one kernel
  K(ktailMulGF31,          "tailmul.cl", "tailMulGF31", hN / nH / 2, kernelDefines(K31).c_str()),
  K(ktailMulLowGF31,       "tailmul.cl", "tailMulGF31", hN / nH / 2, (kernelDefines(K31) + "-DMUL_LOW=1").c_str()),
  K(kfftMidOutGF31,        "fftmiddleout.cl", "fftMiddleOutGF31", hN / (BIG_H / SMALL_H), (kernelDefines(K31) + numCudaRegisters(MIDOUT31)).c_str()),
  K(kfftWGF31,             "fftw.cl", "fftWGF31", hN / nW, kernelDefines(K31).c_str()),

  K(kfftMidInGF61,         "fftmiddlein.cl",  "fftMiddleInGF61",  hN / (BIG_H / SMALL_H), (kernelDefines(K61) + numCudaRegisters(MIDIN61)).c_str()),
#if defined(__APPLE__)
  K(kfftHinGF61,           "ffthin.cl",  "fftHinGF61ApplePlaceholder", hN / nH, kernelDefines(K61).c_str()),
  K(kfftHinGF61LoadScalarApple, "ffthin.cl", "fftHinGF61LoadScalarApple", hN, kernelDefines(K61).c_str()),
  K(kfftHinGF61FftRadixApple,   "ffthin.cl", "fftHinGF61FftRadixApple", hN / nH, kernelDefines(K61).c_str()),
  K(kfftHinGF61FftTwiddleApple, "ffthin.cl", "fftHinGF61FftTwiddleApple", hN, kernelDefines(K61).c_str()),
  K(kfftHinGF61FftShuffleApple, "ffthin.cl", "fftHinGF61FftShuffleApple", hN, kernelDefines(K61).c_str()),
  K(kfftHinGF61FftFinalApple,   "ffthin.cl", "fftHinGF61FftFinalApple", hN / nH, kernelDefines(K61).c_str()),
#else
  K(kfftHinGF61,           "ffthin.cl",  "fftHinGF61",  hN / nH, kernelDefines(K61).c_str()),
#endif
  K(ktailSquareZeroGF61,   "tailsquare.cl", "tailSquareZeroGF61", SMALL_H / nH * 2, kernelDefines(K61).c_str()),
#if defined(__APPLE__)
  K(ktailSquareGF61,       "tailsquare.cl", "tailSquareGF61ApplePlaceholder", SMALL_H / nH, kernelDefines(K61).c_str()),
#else
  K(ktailSquareGF61,       "tailsquare.cl", "tailSquareGF61",
                                               !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2, (kernelDefines(K61) + numCudaRegisters(TAIL61)).c_str()),  // Single-wide tailSquare with one kernel
#endif
#if defined(__APPLE__)
  K(ktailMulGF61,          "tailmul.cl", "tailMulGF61ApplePlaceholder", hN / nH / 2, kernelDefines(K61).c_str()),
  K(ktailMulGF61LoadScalarApple, "tailmul.cl", "tailMulGF61LoadScalarApple", hN, kernelDefines(K61).c_str()),
  K(ktailMulGF61FftRadixApple, "tailmul.cl", "tailMulGF61FftRadixApple", hN / nH, kernelDefines(K61).c_str()),
  K(ktailMulGF61FftTwiddleApple, "tailmul.cl", "tailMulGF61FftTwiddleApple", hN, kernelDefines(K61).c_str()),
  K(ktailMulGF61FftShuffleApple, "tailmul.cl", "tailMulGF61FftShuffleApple", hN, kernelDefines(K61).c_str()),
  K(ktailMulGF61FftFinalApple, "tailmul.cl", "tailMulGF61FftFinalApple", hN / nH, kernelDefines(K61).c_str()),
  K(ktailMulGF61PairSpecialScalarApple, "tailmul.cl", "tailMulGF61PairSpecialScalarApple",
      SMALL_H, kernelDefines(K61).c_str()),
  K(ktailMulGF61PairNormalScalarApple, "tailmul.cl", "tailMulGF61PairNormalScalarApple",
      (hN / nH - SMALL_H / nH * 2) * nH / 2, kernelDefines(K61).c_str()),
#else
  K(ktailMulGF61,          "tailmul.cl", "tailMulGF61", hN / nH / 2, kernelDefines(K61).c_str()),
#endif
  K(ktailMulLowGF61,       "tailmul.cl", "tailMulGF61", hN / nH / 2, (kernelDefines(K61) + "-DMUL_LOW=1").c_str()),
#if defined(__APPLE__)
  K(kfftMidOutGF61,        "fftmiddleout.cl", fft.shape.middle >= 8 ? "fftMiddleOutGF61ApplePlaceholder" : "fftMiddleOutGF61", hN / (BIG_H / SMALL_H), (kernelDefines(K61) + numCudaRegisters(MIDOUT61)).c_str()),
  K(kfftMidOutGF61LoadScalarApple, "fftmiddleout.cl", "fftMiddleOutGF61LoadScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
  K(kfftMidOutGF61MulScalarApple, "fftmiddleout.cl", "fftMiddleOutGF61MulScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
  K(kfftMidOutGF61FftApple, "fftmiddleout.cl", "fftMiddleOutGF61FftApple",
      hN / (BIG_H / SMALL_H), kernelDefines(K61).c_str()),
  K(kfftMidOutGF61Mul2ScalarApple, "fftmiddleout.cl", "fftMiddleOutGF61Mul2ScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
  K(kfftMidOutGF61WriteScalarApple, "fftmiddleout.cl", "fftMiddleOutGF61WriteScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
#else
  K(kfftMidOutGF61,        "fftmiddleout.cl", "fftMiddleOutGF61", hN / (BIG_H / SMALL_H), (kernelDefines(K61) + numCudaRegisters(MIDOUT61)).c_str()),
#endif
#if defined(__APPLE__)
  K(kfftWGF61,             "fftw.cl", "fftWGF61ApplePlaceholder", hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61LoadScalarApple, "fftw.cl", "fftWGF61LoadScalarApple", hN, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthRadixApple, "fftw.cl", "fftWGF61WidthRadixApple", hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle1Apple,   "fftw.cl", "fftWGF61TwiddleShuffle1Apple",   hN, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle4Apple,   "fftw.cl", "fftWGF61TwiddleShuffle4Apple",   hN, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle8Apple,   "fftw.cl", "fftWGF61TwiddleShuffle8Apple",   hN, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle16Apple,  "fftw.cl", "fftWGF61TwiddleShuffle16Apple",  hN, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle64Apple,  "fftw.cl", "fftWGF61TwiddleShuffle64Apple",  hN, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle256Apple, "fftw.cl", "fftWGF61TwiddleShuffle256Apple", hN, kernelDefines(K61).c_str()),
  K(kfftWGF61TwiddleShuffle512Apple, "fftw.cl", "fftWGF61TwiddleShuffle512Apple", hN, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthFinalApple, "fftw.cl", "fftWGF61WidthFinalApple", hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61LoadStage1FusedApple, "fftw.cl", "fftWGF61LoadStage1FusedApple", hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused1Apple,   "fftw.cl", "fftWGF61WidthStageFused1Apple",   hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused4Apple,   "fftw.cl", "fftWGF61WidthStageFused4Apple",   hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused8Apple,   "fftw.cl", "fftWGF61WidthStageFused8Apple",   hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused16Apple,  "fftw.cl", "fftWGF61WidthStageFused16Apple",  hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused64Apple,  "fftw.cl", "fftWGF61WidthStageFused64Apple",  hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused256Apple, "fftw.cl", "fftWGF61WidthStageFused256Apple", hN / nW, kernelDefines(K61).c_str()),
  K(kfftWGF61WidthStageFused512Apple, "fftw.cl", "fftWGF61WidthStageFused512Apple", hN / nW, kernelDefines(K61).c_str()),
#else
  K(kfftWGF61,             "fftw.cl", "fftWGF61", hN / nW, kernelDefines(K61).c_str()),
#endif

  K(kfftP,                 "fftp.cl", "fftP", hN / nW, kernelDefines(KALL).c_str()),
#if defined(__APPLE__)
  K(kfftMidInGF61LoadScalarApple, "fftmiddlein.cl", "fftMiddleInGF61LoadScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
  K(kfftMidInGF61Mul2FactorScalarApple, "fftmiddlein.cl", "fftMiddleInGF61Mul2FactorScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
  K(kfftMidInGF61ApplyScalarApple, "fftmiddlein.cl", "fftMiddleInGF61ApplyScalarApple",
      hN / (BIG_H / SMALL_H) * fft.shape.middle, kernelDefines(K61).c_str()),
  K(kfftMidInGF61FftApple,      "fftmiddlein.cl", "fftMiddleInGF61FftApple",      hN / (BIG_H / SMALL_H), kernelDefines(K61).c_str()),
  K(kfftMidInGF61MulApple,      "fftmiddlein.cl", "fftMiddleInGF61MulApple",      hN / (BIG_H / SMALL_H), kernelDefines(K61).c_str()),
  K(kfftMidInGF61TransposeApple,"fftmiddlein.cl", "fftMiddleInGF61TransposeApple",hN / (BIG_H / SMALL_H), kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61LoadApple,       "tailsquare.cl", "tailSquareZeroGF61LoadApple",       SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61FftRadixApple,   "tailsquare.cl", "tailSquareZeroGF61FftRadixApple",   SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61FftTwiddleApple, "tailsquare.cl", "tailSquareZeroGF61FftTwiddleApple", SMALL_H * 2,      kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61FftShuffleApple, "tailsquare.cl", "tailSquareZeroGF61FftShuffleApple", SMALL_H * 2,      kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61FftFinalApple,   "tailsquare.cl", "tailSquareZeroGF61FftFinalApple",   SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61ReverseGlobalApple, "tailsquare.cl", "tailSquareZeroGF61ReverseGlobalApple", SMALL_H * 2, kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61PairApple,    "tailsquare.cl", "tailSquareZeroGF61PairApple",    SMALL_H,          kernelDefines(K61).c_str()),
  K(ktailSquareZeroGF61WriteDirectApple,   "tailsquare.cl", "tailSquareZeroGF61WriteDirectApple",   SMALL_H, kernelDefines(K61).c_str()),
  K(ktailSquareGF61LoadScalarApple, "tailsquare.cl", "tailSquareGF61LoadScalarApple",
      (hN / nH - SMALL_H / nH * 2) * nH, kernelDefines(K61).c_str()),
  K(ktailSquareGF61FftRadixApple, "tailsquare.cl", "tailSquareGF61FftRadixApple",
      hN / nH - SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareGF61FftTwiddleApple, "tailsquare.cl", "tailSquareGF61FftTwiddleApple",
      (hN / nH - SMALL_H / nH * 2) * nH, kernelDefines(K61).c_str()),
  K(ktailSquareGF61FftShuffleApple, "tailsquare.cl", "tailSquareGF61FftShuffleApple",
      (hN / nH - SMALL_H / nH * 2) * nH, kernelDefines(K61).c_str()),
  K(ktailSquareGF61FftFinalApple, "tailsquare.cl", "tailSquareGF61FftFinalApple",
      hN / nH - SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareGF61ReverseCrossApple, "tailsquare.cl", "tailSquareGF61ReverseCrossApple",
      (hN / nH - SMALL_H / nH * 2) * nH, kernelDefines(K61).c_str()),
  K(ktailSquareGF61PairApple, "tailsquare.cl", "tailSquareGF61PairApple",
      (hN / nH - SMALL_H / nH * 2) * nH / 2, kernelDefines(K61).c_str()),
  K(ktailSquareGF61LoadStageFusedApple, "tailsquare.cl", "tailSquareGF61LoadStageFusedApple",
      hN / nH - SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareGF61StageFusedApple, "tailsquare.cl", "tailSquareGF61StageFusedApple",
      hN / nH - SMALL_H / nH * 2, kernelDefines(K61).c_str()),
  K(ktailSquareGF61PairCrossFusedApple, "tailsquare.cl", "tailSquareGF61PairCrossFusedApple",
      (hN / nH - SMALL_H / nH * 2) * nH / 2, kernelDefines(K61).c_str()),
  K(ktailSquareGF61FinalPairFirstFusedApple, "tailsquare.cl", "tailSquareGF61FinalPairFirstFusedApple",
      (hN / nH - SMALL_H / nH * 2) / 2, kernelDefines(K61).c_str()),
  K(kfftP31Apple,          "fftp.cl", "fftP31Apple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WeightScalarApple, "fftp.cl", "fftP31WeightScalarApple", hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthRadixApple,   "fftp.cl", "fftP31WidthRadixApple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle1Apple,   "fftp.cl", "fftP31TwiddleShuffle1Apple",   hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle4Apple,   "fftp.cl", "fftP31TwiddleShuffle4Apple",   hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle8Apple,   "fftp.cl", "fftP31TwiddleShuffle8Apple",   hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle16Apple,  "fftp.cl", "fftP31TwiddleShuffle16Apple",  hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle64Apple,  "fftp.cl", "fftP31TwiddleShuffle64Apple",  hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle256Apple, "fftp.cl", "fftP31TwiddleShuffle256Apple", hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31TwiddleShuffle512Apple, "fftp.cl", "fftP31TwiddleShuffle512Apple", hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthFinalApple,        "fftp.cl", "fftP31WidthFinalApple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WeightStage1FusedApple, "fftp.cl", "fftP31WeightStage1FusedApple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused1Apple,   "fftp.cl", "fftP31WidthStageFused1Apple",   hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused4Apple,   "fftp.cl", "fftP31WidthStageFused4Apple",   hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused8Apple,   "fftp.cl", "fftP31WidthStageFused8Apple",   hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused16Apple,  "fftp.cl", "fftP31WidthStageFused16Apple",  hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused64Apple,  "fftp.cl", "fftP31WidthStageFused64Apple",  hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused256Apple, "fftp.cl", "fftP31WidthStageFused256Apple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP31WidthStageFused512Apple, "fftp.cl", "fftP31WidthStageFused512Apple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WeightScalarApple, "fftp.cl", "fftP61WeightScalarApple", hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthRadixApple,   "fftp.cl", "fftP61WidthRadixApple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle1Apple,   "fftp.cl", "fftP61TwiddleShuffle1Apple",   hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle4Apple,   "fftp.cl", "fftP61TwiddleShuffle4Apple",   hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle8Apple,   "fftp.cl", "fftP61TwiddleShuffle8Apple",   hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle16Apple,  "fftp.cl", "fftP61TwiddleShuffle16Apple",  hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle64Apple,  "fftp.cl", "fftP61TwiddleShuffle64Apple",  hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle256Apple, "fftp.cl", "fftP61TwiddleShuffle256Apple", hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61TwiddleShuffle512Apple, "fftp.cl", "fftP61TwiddleShuffle512Apple", hN, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthFinalApple,    "fftp.cl", "fftP61WidthFinalApple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WeightStage1FusedApple, "fftp.cl", "fftP61WeightStage1FusedApple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused1Apple,   "fftp.cl", "fftP61WidthStageFused1Apple",   hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused4Apple,   "fftp.cl", "fftP61WidthStageFused4Apple",   hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused8Apple,   "fftp.cl", "fftP61WidthStageFused8Apple",   hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused16Apple,  "fftp.cl", "fftP61WidthStageFused16Apple",  hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused64Apple,  "fftp.cl", "fftP61WidthStageFused64Apple",  hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused256Apple, "fftp.cl", "fftP61WidthStageFused256Apple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
  K(kfftP61WidthStageFused512Apple, "fftp.cl", "fftP61WidthStageFused512Apple", hN / nW, (kernelDefines(KALL) + "-DAEVUM_APPLE_SPLIT_FFTP=1").c_str()),
#endif
  K(kCarryA,               "carry.cl", "carry", hN / CARRY_LEN, kernelDefines(KALL).c_str()),
  K(kCarryAROE,            "carry.cl", "carry", hN / CARRY_LEN, (kernelDefines(KALL) + "-DROE=1").c_str()),
  K(kCarryM,               "carry.cl", "carry", hN / CARRY_LEN, (kernelDefines(KALL) + "-DMUL3=1").c_str()),
  K(kCarryMROE,            "carry.cl", "carry", hN / CARRY_LEN, (kernelDefines(KALL) + "-DMUL3=1 -DROE=1").c_str()),
  K(kCarryLL,              "carry.cl", "carry", hN / CARRY_LEN, (kernelDefines(KALL) + "-DLL=1").c_str()),
  K(kCarryFused,           "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, (kernelDefines(KALL) + numCudaRegisters(CARRYFUSED)).c_str()),
  K(kCarryFusedROE,        "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, (kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DROE=1").c_str()),
  K(kCarryFusedMul,        "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, (kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DMUL3=1").c_str()),
  K(kCarryFusedMulROE,     "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, (kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DMUL3=1 -DROE=1").c_str()),
  K(kCarryFusedLL,         "carryfused.cl", "carryFused", WIDTH * (BIG_H + wmul) / nW, (kernelDefines(KALL) + numCudaRegisters(CARRYFUSED) + "-DLL=1").c_str()),

  K(carryB,                "carryb.cl", "carryB",   hN / CARRY_LEN, kernelDefines(KALL).c_str()),

  // 64
#if defined(__APPLE__)
  // Apple OpenCL-to-Metal corrupts the stock 64x64 LDS transpose even on an
  // all-zero register.  The direct global kernels launch one work-item per
  // Word2 and preserve the exact sequential<->transposed index mapping.
  K(transpIn,  "transpose.cl", "transposeInAppleGlobal",  hN),
  K(transpOut, "transpose.cl", "transposeOutAppleGlobal", hN),
#else
  K(transpIn,  "transpose.cl", "transposeIn",  hN / 64),
  K(transpOut, "transpose.cl", "transposeOut", hN / 64),
#endif

  K(readResidue, "etc.cl", "readResidue", 32, "-DREADRESIDUE=1"),

  // 256
  K(kernIsEqual, "etc.cl", "isEqual", 256 * 256, "-DISEQUAL=1"),
  K(sum64,       "etc.cl", "sum64",   256 * 256, "-DSUM64=1"),
  K(regAdd,      "etc.cl", "regAdd",      256 * 256, "-DREGADD=1"),
  K(regSub,      "etc.cl", "regSub",      256 * 256, "-DREGSUB=1"),
  K(regSubValue, "etc.cl", "regSubValue", 1,         "-DREGSUBVALUE=1"),

  K(testTrig, "selftest.cl", "testTrig", 256 * 256),
  K(testFFT4, "selftest.cl", "testFFT4", 256),
  K(testFFT14, "selftest.cl", "testFFT14", 256),
  K(testFFT15, "selftest.cl", "testFFT15", 256),
  K(testFFT, "selftest.cl", "testFFT", 256),
  K(testTime, "selftest.cl", "testTime", 4096 * 64),

#undef K

  bufTrigH{shared.bufCache->smallTrigCombo(shared.args, fft, WIDTH, fft.shape.middle, SMALL_H, nH, tail_single_wide)},
  bufTrigM{shared.bufCache->middleTrig(shared.args, fft, SMALL_H, BIG_H / SMALL_H, WIDTH)},
  bufTrigW{shared.bufCache->smallTrig(shared.args, fft, WIDTH, nW, fft.shape.middle, SMALL_H, nH, tail_single_wide)},

  weights{genWeights(fft, E, WIDTH, BIG_H, nW, isNvidiaGpu(shared.context->deviceId()))},
  bufConstWeights{shared.context, std::move(weights.weightsConstIF)},
  bufWeights{shared.context,      std::move(weights.weightsIF)},

#define BUF(name, ...) name{profile.make(#name), &queue, __VA_ARGS__}

  // GPU Buffers containing integer data.  Since this buffer is type i64, if fft.WordSize < 8 then we need less memory allocated.
  BUF(bufData, N * fft.WordSize / sizeof(Word)),
  BUF(bufAux, N * fft.WordSize / sizeof(Word)),
  BUF(bufCheck, N * fft.WordSize / sizeof(Word)),
  // Every double-word (i.e. N/2) produces one carry. In addition we may have one extra group thus WIDTH more carries.
  BUF(bufCarry, N / 2 + WIDTH),
  BUF(bufReady, (N / 2 + WIDTH) / 32), // Every wavefront (32 or 64 lanes) needs to signal "carry is ready"

  BUF(bufSmallOut, 256),
  BUF(bufSumOut,     1),
  BUF(bufTrue,       1),
  BUF(bufROE, ROE_SIZE),
  BUF(bufStatsCarry, CARRY_SIZE),

  BUF(buf1, TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)),
  BUF(buf2, TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)),
  BUF(buf3, TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size)),
#if defined(__APPLE__)
  BUF(bufAppleTailZeroGF61, fft.NTT_GF61 ? 8 * SMALL_H : 0),
  // One GF61-only plane for Apple tailMul ping-pong.  This is smaller than a
  // full combined GF31/GF61 transform buffer and is unused on other platforms.
  BUF(bufAppleTailMulGF61, fft.NTT_GF61 ? GF61_DATA_SIZE(WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size) : 0),
#endif
#undef BUF

  statsBits{u32(args.value("STATS", 0))},
  timeBufVect{profile.make("proofBufVect")},

  recorded_kernels{},
  recorded_kernel_args{}
{    

#if defined(__APPLE__)
  apple_fused_tailsquare_gf61 = fft.shape.fft_type == FFT3161 && fft.NTT_GF61 && nH > 1;
  // v0.3.56 does not ship the matching OpenCL bridge kernel.
  // Keep the validated v0.3.54 tailSquare path; width fusions remain enabled.
  apple_bridge_fused_tailsquare_gf61 = false;
  auto envEnabled = [](const char* name) {
    if (const char* value = std::getenv(name)) return *value != '\0' && std::strcmp(value, "0") != 0;
    return false;
  };
  if (envEnabled("AEVUM_APPLE_STAGE_FINISH")) {
    apple_stage_finish = true;
    log("Apple Aevum diagnostic: strict staged clFinish serialization enabled.\n");
  }
  apple_fused_fftp_width = fft.shape.fft_type == FFT3161 && fft.NTT_GF31 && fft.NTT_GF61 && nW > 1;
  apple_fused_fftp_weight_first = apple_fused_fftp_width;
  if (envEnabled("AEVUM_APPLE_FFTP_V55")) {
    apple_fused_fftp_width = false;
    apple_fused_fftp_weight_first = false;
  }
  if (envEnabled("AEVUM_APPLE_FFTP_STAGE_ONLY")) {
    apple_fused_fftp_weight_first = false;
  }
  auto preloadFftpStage = [&](u32 stage, bool gf61) {
    Kernel* k = nullptr;
    if (!gf61) {
      switch (stage) {
        case 1: k=&kfftP31WidthStageFused1Apple; break;
        case 4: k=&kfftP31WidthStageFused4Apple; break;
        case 8: k=&kfftP31WidthStageFused8Apple; break;
        case 16: k=&kfftP31WidthStageFused16Apple; break;
        case 64: k=&kfftP31WidthStageFused64Apple; break;
        case 256: k=&kfftP31WidthStageFused256Apple; break;
        case 512: k=&kfftP31WidthStageFused512Apple; break;
      }
    } else {
      switch (stage) {
        case 1: k=&kfftP61WidthStageFused1Apple; break;
        case 4: k=&kfftP61WidthStageFused4Apple; break;
        case 8: k=&kfftP61WidthStageFused8Apple; break;
        case 16: k=&kfftP61WidthStageFused16Apple; break;
        case 64: k=&kfftP61WidthStageFused64Apple; break;
        case 256: k=&kfftP61WidthStageFused256Apple; break;
        case 512: k=&kfftP61WidthStageFused512Apple; break;
      }
    }
    if (!k) throw std::runtime_error("Unsupported Apple fused fftP stage " + std::to_string(stage));
    k->startLoad(&compiler);
    k->finishLoad();
  };
  const u32 apple_width_workgroup = WIDTH / nW;
  if (apple_fused_fftp_width) {
    try {
      for (u32 stage=1; stage<apple_width_workgroup; stage*=nW) {
        preloadFftpStage(stage, false);
        preloadFftpStage(stage, true);
      }
    } catch (const std::exception& e) {
      apple_fused_fftp_width = false;
      apple_fused_fftp_weight_first = false;
      log("Apple Aevum performance: fused fftP width stages unavailable (%s); using v0.3.55 split fftP.\n", e.what());
    } catch (...) {
      apple_fused_fftp_width = false;
      apple_fused_fftp_weight_first = false;
      log("Apple Aevum performance: fused fftP width stages unavailable; using v0.3.55 split fftP.\n");
    }
  }
  if (apple_fused_fftp_weight_first) {
    try {
      kfftP31WeightStage1FusedApple.startLoad(&compiler);
      kfftP31WeightStage1FusedApple.finishLoad();
      kfftP61WeightStage1FusedApple.startLoad(&compiler);
      kfftP61WeightStage1FusedApple.finishLoad();
    } catch (const std::exception& e) {
      apple_fused_fftp_weight_first = false;
      log("Apple Aevum performance: fused fftP weight+stage1 unavailable (%s); using stage-only fusion.\n", e.what());
    } catch (...) {
      apple_fused_fftp_weight_first = false;
      log("Apple Aevum performance: fused fftP weight+stage1 unavailable; using stage-only fusion.\n");
    }
  }
  u32 apple_fftp_stages = 0;
  for (u32 stage=1; stage<apple_width_workgroup; stage*=nW) ++apple_fftp_stages;
  const u32 apple_fftp_v55_dispatches = 2 * (2 * apple_fftp_stages + 2);
  const u32 apple_fftp_stage_dispatches = 2 * (apple_fftp_stages + 2);
  const u32 apple_fftp_turbo_dispatches = 2 * (apple_fftp_stages + 1);
  if (apple_fused_fftp_weight_first) {
    log("Apple Aevum performance: fused CRT fftP enabled (weight+stage1 and radix+twiddle+shuffle); dispatches %u -> %u -> %u.\n",
        apple_fftp_v55_dispatches, apple_fftp_stage_dispatches, apple_fftp_turbo_dispatches);
  } else if (apple_fused_fftp_width) {
    log("Apple Aevum performance: fused CRT fftP width stages enabled; dispatches %u -> %u.\n",
        apple_fftp_v55_dispatches, apple_fftp_stage_dispatches);
  } else if (envEnabled("AEVUM_APPLE_FFTP_V55")) {
    log("Apple Aevum diagnostic: v0.3.55 split fftP selected by AEVUM_APPLE_FFTP_V55.\n");
  }
  apple_fused_fftw_width = fft.shape.fft_type == FFT3161 && fft.NTT_GF61 && nW > 1;
  apple_fused_fftw_load_first = apple_fused_fftw_width;
  if (envEnabled("AEVUM_APPLE_FFTW_V55")) {
    apple_fused_fftw_width = false;
    apple_fused_fftw_load_first = false;
  }
  if (envEnabled("AEVUM_APPLE_FFTW_STAGE_ONLY")) {
    apple_fused_fftw_load_first = false;
  }
  auto preloadFftwStage = [&](u32 stage) {
    Kernel* k = nullptr;
    switch (stage) {
      case 1: k=&kfftWGF61WidthStageFused1Apple; break;
      case 4: k=&kfftWGF61WidthStageFused4Apple; break;
      case 8: k=&kfftWGF61WidthStageFused8Apple; break;
      case 16: k=&kfftWGF61WidthStageFused16Apple; break;
      case 64: k=&kfftWGF61WidthStageFused64Apple; break;
      case 256: k=&kfftWGF61WidthStageFused256Apple; break;
      case 512: k=&kfftWGF61WidthStageFused512Apple; break;
    }
    if (!k) throw std::runtime_error("Unsupported Apple fused fftW stage " + std::to_string(stage));
    k->startLoad(&compiler);
    k->finishLoad();
  };
  if (apple_fused_fftw_width) {
    try {
      for (u32 stage=1; stage<apple_width_workgroup; stage*=nW) preloadFftwStage(stage);
    } catch (const std::exception& e) {
      apple_fused_fftw_width = false;
      apple_fused_fftw_load_first = false;
      log("Apple Aevum performance: fused GF61 fftW width stages unavailable (%s); using v0.3.55 split fftW.\n", e.what());
    } catch (...) {
      apple_fused_fftw_width = false;
      apple_fused_fftw_load_first = false;
      log("Apple Aevum performance: fused GF61 fftW width stages unavailable; using v0.3.55 split fftW.\n");
    }
  }
  if (apple_fused_fftw_load_first) {
    try {
      kfftWGF61LoadStage1FusedApple.startLoad(&compiler);
      kfftWGF61LoadStage1FusedApple.finishLoad();
    } catch (const std::exception& e) {
      apple_fused_fftw_load_first = false;
      log("Apple Aevum performance: fused GF61 fftW load+stage1 unavailable (%s); using stage-only fusion.\n", e.what());
    } catch (...) {
      apple_fused_fftw_load_first = false;
      log("Apple Aevum performance: fused GF61 fftW load+stage1 unavailable; using stage-only fusion.\n");
    }
  }
  const u32 apple_fftw_v55_dispatches = 2 * apple_fftp_stages + 2;
  const u32 apple_fftw_stage_dispatches = apple_fftp_stages + 2;
  const u32 apple_fftw_turbo_dispatches = apple_fftp_stages + 1;
  if (apple_fused_fftw_load_first) {
    log("Apple Aevum performance: fused GF61 fftW enabled (load+stage1 and radix+twiddle+shuffle); dispatches %u -> %u -> %u.\n",
        apple_fftw_v55_dispatches, apple_fftw_stage_dispatches, apple_fftw_turbo_dispatches);
  } else if (apple_fused_fftw_width) {
    log("Apple Aevum performance: fused GF61 fftW width stages enabled; dispatches %u -> %u.\n",
        apple_fftw_v55_dispatches, apple_fftw_stage_dispatches);
  } else if (envEnabled("AEVUM_APPLE_FFTW_V55")) {
    log("Apple Aevum diagnostic: v0.3.55 split GF61 fftW selected by AEVUM_APPLE_FFTW_V55.\n");
  }
  if (envEnabled("AEVUM_APPLE_TAILSQUARE_LEGACY")) {
    apple_fused_tailsquare_gf61 = false;
    apple_bridge_fused_tailsquare_gf61 = false;
  }
  if (apple_fused_tailsquare_gf61) {
    // Apple compiles OpenCL kernels to Metal when clCreateKernel is reached.
    // Preload the v0.3.54 kernels before any arithmetic is modified; if the
    // driver rejects one of them, retain the validated v0.3.53 staged path.
    try {
      ktailSquareGF61LoadStageFusedApple.startLoad(&compiler);
      ktailSquareGF61LoadStageFusedApple.finishLoad();
      ktailSquareGF61StageFusedApple.startLoad(&compiler);
      ktailSquareGF61StageFusedApple.finishLoad();
      ktailSquareGF61PairCrossFusedApple.startLoad(&compiler);
      ktailSquareGF61PairCrossFusedApple.finishLoad();
    } catch (const std::exception& e) {
      apple_fused_tailsquare_gf61 = false;
      apple_bridge_fused_tailsquare_gf61 = false;
      log("Apple Aevum performance: fused GF61 tailSquare unavailable (%s); using validated legacy staging.\n", e.what());
    } catch (...) {
      apple_fused_tailsquare_gf61 = false;
      apple_bridge_fused_tailsquare_gf61 = false;
      log("Apple Aevum performance: fused GF61 tailSquare unavailable; using validated legacy staging.\n");
    }
  }
  if (envEnabled("AEVUM_APPLE_TAILSQUARE_V54")) {
    apple_bridge_fused_tailsquare_gf61 = false;
  }
  if (apple_bridge_fused_tailsquare_gf61) {
    // v0.3.55 additionally composes final-forward radix, both directional
    // pairings and the first inverse stage.  Preload it separately so a Metal
    // compiler rejection falls back to the already validated v0.3.54 path.
    try {
      ktailSquareGF61FinalPairFirstFusedApple.startLoad(&compiler);
      ktailSquareGF61FinalPairFirstFusedApple.finishLoad();
    } catch (const std::exception& e) {
      apple_bridge_fused_tailsquare_gf61 = false;
      log("Apple Aevum performance: bridge-fused GF61 tailSquare unavailable (%s); using v0.3.54 fused staging.\n", e.what());
    } catch (...) {
      apple_bridge_fused_tailsquare_gf61 = false;
      log("Apple Aevum performance: bridge-fused GF61 tailSquare unavailable; using v0.3.54 fused staging.\n");
    }
  }
  u32 stage_count = 0;
  for (u32 stage = 1; stage < SMALL_H / nH; stage *= nH) ++stage_count;
  const u32 legacy_dispatches = 6 * stage_count + 6;
  const u32 fused_dispatches = 2 * stage_count + 3;
  const u32 bridge_dispatches = 2 * stage_count + 1;
  if (apple_bridge_fused_tailsquare_gf61) {
    log("Apple Aevum performance: bridge-fused GF61 tailSquare enabled (final-forward radix + R-pair-R + first inverse stage); normal-tail dispatches %u -> %u -> %u.\n",
        legacy_dispatches, fused_dispatches, bridge_dispatches);
  } else if (apple_fused_tailsquare_gf61) {
    log("Apple Aevum performance: v0.3.54 fused GF61 tailSquare stages enabled; normal-tail dispatches %u -> %u.\n",
        legacy_dispatches, fused_dispatches);
    if (envEnabled("AEVUM_APPLE_TAILSQUARE_V54")) {
      log("Apple Aevum diagnostic: v0.3.54 fused tailSquare selected by AEVUM_APPLE_TAILSQUARE_V54.\n");
    }
  } else if (envEnabled("AEVUM_APPLE_TAILSQUARE_LEGACY")) {
    log("Apple Aevum diagnostic: legacy staged GF61 tailSquare selected by AEVUM_APPLE_TAILSQUARE_LEGACY.\n");
  }
#endif

  float bitsPerWord = E / float(N);
  if (logFftSize) {
    log("FFT: %s %s (%.2f bpw)\n", numberK(N).c_str(), fft.spec().c_str(), bitsPerWord);

    // Sometimes we do want to run a FFT beyond a reasonable BPW (e.g. during -ztune), and these situations
    // coincide with logFftSize == false
    if (fft.maxExp() < E) {
      log("Warning: %s (max %" PRIu64 ") may be too small for %" PRIu64 "\n", fft.spec().c_str(), fft.maxExp(), E);
    }
  }

  if (bitsPerWord < fft.minBpw()) {
    log("FFT size too large for exponent (%.2f bits/word < %.2f bits/word).\n", bitsPerWord, fft.minBpw());
    throw "FFT size too large";
  }

  useLongCarry = useLongCarry || (bitsPerWord < 10.0);

  if (useLongCarry) { log("Using long carry!\n"); }
#if defined(__APPLE__)
  if (fft.shape.fft_type == FFT3161) {
    log("Apple OpenCL 1.2 compatibility: targeted fftP, GF61 middle-in, tail staging and fftW global ping-pong staging; middle LDS transposes retained (IN_WG=64, OUT_WG=64).\n");
  }
#endif

  if (fft.FFT_FP64 || fft.FFT_FP32) {
    kfftMidIn.setFixedArgs(2, bufTrigM);
    kfftHin.setFixedArgs(2, bufTrigH);
    ktailSquareZero.setFixedArgs(2, bufTrigH);
    ktailSquare.setFixedArgs(2, bufTrigH);
    ktailMulLow.setFixedArgs(3, bufTrigH);
    ktailMul.setFixedArgs(3, bufTrigH);
    kfftMidOut.setFixedArgs(2, bufTrigM);
    kfftW.setFixedArgs(2, bufTrigW);
  }

  if (fft.NTT_GF31) {
    kfftMidInGF31.setFixedArgs(2, bufTrigM);
    kfftHinGF31.setFixedArgs(2, bufTrigH);
    ktailSquareZeroGF31.setFixedArgs(2, bufTrigH);
    ktailSquareGF31.setFixedArgs(2, bufTrigH);
    ktailMulLowGF31.setFixedArgs(3, bufTrigH);
    ktailMulGF31.setFixedArgs(3, bufTrigH);
    kfftMidOutGF31.setFixedArgs(2, bufTrigM);
    kfftWGF31.setFixedArgs(2, bufTrigW);
  }

  if (fft.NTT_GF61) {
    kfftMidInGF61.setFixedArgs(2, bufTrigM);
#if defined(__APPLE__)
    kfftMidInGF61Mul2FactorScalarApple.setFixedArgs(1, bufTrigM);
    kfftMidInGF61MulApple.setFixedArgs(1, bufTrigM);
#endif
    kfftHinGF61.setFixedArgs(2, bufTrigH);
#if defined(__APPLE__)
    kfftHinGF61FftTwiddleApple.setFixedArgs(2, bufTrigH);
#endif
    ktailSquareZeroGF61.setFixedArgs(2, bufTrigH);
    ktailSquareGF61.setFixedArgs(2, bufTrigH);
    ktailMulLowGF61.setFixedArgs(3, bufTrigH);
    ktailMulGF61.setFixedArgs(3, bufTrigH);
    kfftMidOutGF61.setFixedArgs(2, bufTrigM);
    kfftWGF61.setFixedArgs(2, bufTrigW);
#if defined(__APPLE__)
    for (Kernel* k : {&kfftWGF61TwiddleShuffle1Apple, &kfftWGF61TwiddleShuffle4Apple,
                      &kfftWGF61TwiddleShuffle8Apple, &kfftWGF61TwiddleShuffle16Apple,
                      &kfftWGF61TwiddleShuffle64Apple, &kfftWGF61TwiddleShuffle256Apple,
                      &kfftWGF61TwiddleShuffle512Apple,
                      &kfftWGF61LoadStage1FusedApple,
                      &kfftWGF61WidthStageFused1Apple, &kfftWGF61WidthStageFused4Apple,
                      &kfftWGF61WidthStageFused8Apple, &kfftWGF61WidthStageFused16Apple,
                      &kfftWGF61WidthStageFused64Apple, &kfftWGF61WidthStageFused256Apple,
                      &kfftWGF61WidthStageFused512Apple}) {
      k->setFixedArgs(2, bufTrigW);
    }
#endif
  }

  if (fft.FFT_FP64 || fft.FFT_FP32) {                         // The FP versions take bufWeight arguments
    kfftP.setFixedArgs(2, bufTrigW, bufWeights);
    for (Kernel* k : {&kCarryA, &kCarryAROE, &kCarryM, &kCarryMROE, &kCarryLL}) { k->setFixedArgs(3, bufCarry, bufWeights); }
    for (Kernel* k : {&kCarryA, &kCarryM, &kCarryLL}) { k->setFixedArgs(5, bufStatsCarry); }
    for (Kernel* k : {&kCarryAROE, &kCarryMROE})      { k->setFixedArgs(5, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedROE, &kCarryFusedMul, &kCarryFusedMulROE, &kCarryFusedLL}) {
      k->setFixedArgs(3, bufCarry, bufReady, bufTrigW, bufConstWeights, bufWeights);
    }
    for (Kernel* k : {&kCarryFusedROE, &kCarryFusedMulROE})           { k->setFixedArgs(8, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedMul, &kCarryFusedLL}) { k->setFixedArgs(8, bufStatsCarry); }
  } else {
    kfftP.setFixedArgs(2, bufTrigW);
#if defined(__APPLE__)
    kfftP31Apple.setFixedArgs(2, bufTrigW);
    for (Kernel* k : {&kfftP31TwiddleShuffle1Apple, &kfftP31TwiddleShuffle4Apple,
                      &kfftP31TwiddleShuffle8Apple, &kfftP31TwiddleShuffle16Apple,
                      &kfftP31TwiddleShuffle64Apple, &kfftP31TwiddleShuffle256Apple,
                      &kfftP31TwiddleShuffle512Apple,
                      &kfftP61TwiddleShuffle1Apple, &kfftP61TwiddleShuffle4Apple,
                      &kfftP61TwiddleShuffle8Apple, &kfftP61TwiddleShuffle16Apple,
                      &kfftP61TwiddleShuffle64Apple, &kfftP61TwiddleShuffle256Apple,
                      &kfftP61TwiddleShuffle512Apple,
                      &kfftP31WeightStage1FusedApple,
                      &kfftP31WidthStageFused1Apple, &kfftP31WidthStageFused4Apple,
                      &kfftP31WidthStageFused8Apple, &kfftP31WidthStageFused16Apple,
                      &kfftP31WidthStageFused64Apple, &kfftP31WidthStageFused256Apple,
                      &kfftP31WidthStageFused512Apple,
                      &kfftP61WeightStage1FusedApple,
                      &kfftP61WidthStageFused1Apple, &kfftP61WidthStageFused4Apple,
                      &kfftP61WidthStageFused8Apple, &kfftP61WidthStageFused16Apple,
                      &kfftP61WidthStageFused64Apple, &kfftP61WidthStageFused256Apple,
                      &kfftP61WidthStageFused512Apple}) {
      k->setFixedArgs(2, bufTrigW);
    }
#endif
    for (Kernel* k : {&kCarryA, &kCarryAROE, &kCarryM, &kCarryMROE, &kCarryLL}) { k->setFixedArgs(3, bufCarry); }
    for (Kernel* k : {&kCarryA, &kCarryM, &kCarryLL}) { k->setFixedArgs(4, bufStatsCarry); }
    for (Kernel* k : {&kCarryAROE, &kCarryMROE})      { k->setFixedArgs(4, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedROE, &kCarryFusedMul, &kCarryFusedMulROE, &kCarryFusedLL}) {
      k->setFixedArgs(3, bufCarry, bufReady, bufTrigW);
    }
    for (Kernel* k : {&kCarryFusedROE, &kCarryFusedMulROE}) { k->setFixedArgs(6, bufROE); }
    for (Kernel* k : {&kCarryFused, &kCarryFusedMul, &kCarryFusedLL}) { k->setFixedArgs(6, bufStatsCarry); }
  }

  carryB.setFixedArgs(1, bufCarry);

  kernIsEqual.setFixedArgs(2, bufTrue);

  bufReady.zero();
  bufROE.zero();
  bufStatsCarry.zero();
  bufTrue.write({1});

  if (args.verbose) {
    selftestTrig();
  }

  // If MULTI_Q option is set there are fewer kernels executed in the main queue, but there are some additional syncEvents and waits.
  // That's a total of 4 kernels (carryFused, MidIn/Out, TailSquare) plus 1 syncEvent plus 1 or more syncWaits.
  // If MULTI_Q option is not set there is carryFused + MidIn/Out and TailSquare for each NTT modulus.
  // NOTE: We dont take into account the optional tailSquareZero kernel.  We theoretically should.
  if (args.value("MULTI_Q", 0))
    queue.setSquareKernels(5 + ((fft.FFT_FP64 + fft.FFT_FP32 + fft.NTT_GF31 + fft.NTT_GF61) - 1));
  else
    queue.setSquareKernels(1 + 3 * (fft.FFT_FP64 + fft.FFT_FP32 + fft.NTT_GF31 + fft.NTT_GF61));
  queue.finish();
}

// Optionallly split some of the MiddleIn/Tail/MiddleOut kernels off od executing on the main queue to run on an auxiliary queue.
// This will increase GPU occupancy but will negatively impact L2 cache coherency.
// If the L2 cache is large enough so that all FFT data fits in the cache, this ought to be a win.
// If the L2 cache is small enough such that L2 cache hits are very low anyway, this might be a win.

void Gpu::splitQueue(void) {

  // If MULTI_Q -use not set, return
  if (!args.value("MULTI_Q", 0)) return;

  // Create aux queues.  For now, we only have one auxiliary queue.  We could do more.
  if (auxQueues.size() == 0) {
    auxQueues.push_back(Queue{*shared.context, args.profile, true});
  }

  // Queue a sync event in the main queue.  Have all auxiliary queues wait on the event.
  EventHolder event = queue.createSyncEvent();
  for (size_t i = 0; i < auxQueues.size(); ++i) {
    auxQueues[i].waitForSyncEvent(&event);
  }

  // Assign kernels to running on the main queue or an auxiliary queue

  int which_queue = -1;

  // For no particularly good reason, put a kernel that operates on 64-bit vaules in the main queue.
  if (fft.NTT_GF61) {
    if (which_queue != -1) {
      kfftMidInGF61.setQueue(&auxQueues[which_queue]);
#if defined(__APPLE__)
      kfftMidInGF61LoadScalarApple.setQueue(&auxQueues[which_queue]);
      kfftMidInGF61Mul2FactorScalarApple.setQueue(&auxQueues[which_queue]);
      kfftMidInGF61ApplyScalarApple.setQueue(&auxQueues[which_queue]);
      kfftMidInGF61FftApple.setQueue(&auxQueues[which_queue]);
      kfftMidInGF61MulApple.setQueue(&auxQueues[which_queue]);
      kfftMidInGF61TransposeApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61LoadApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61FftRadixApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61FftTwiddleApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61FftShuffleApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61FftFinalApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61ReverseGlobalApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61PairApple.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF61WriteDirectApple.setQueue(&auxQueues[which_queue]);
      ktailSquareGF61LoadStageFusedApple.setQueue(&auxQueues[which_queue]);
      ktailSquareGF61StageFusedApple.setQueue(&auxQueues[which_queue]);
      ktailSquareGF61PairCrossFusedApple.setQueue(&auxQueues[which_queue]);
      ktailSquareGF61FinalPairFirstFusedApple.setQueue(&auxQueues[which_queue]);
#endif
      kfftHinGF61.setQueue(&auxQueues[which_queue]);
#if defined(__APPLE__)
      kfftHinGF61LoadScalarApple.setQueue(&auxQueues[which_queue]);
      kfftHinGF61FftRadixApple.setQueue(&auxQueues[which_queue]);
      kfftHinGF61FftTwiddleApple.setQueue(&auxQueues[which_queue]);
      kfftHinGF61FftShuffleApple.setQueue(&auxQueues[which_queue]);
      kfftHinGF61FftFinalApple.setQueue(&auxQueues[which_queue]);
#endif
      ktailSquareZeroGF61.setQueue(&auxQueues[which_queue]);
      ktailSquareGF61.setQueue(&auxQueues[which_queue]);
      ktailMulGF61.setQueue(&auxQueues[which_queue]);
#if defined(__APPLE__)
      ktailMulGF61LoadScalarApple.setQueue(&auxQueues[which_queue]);
      ktailMulGF61FftRadixApple.setQueue(&auxQueues[which_queue]);
      ktailMulGF61FftTwiddleApple.setQueue(&auxQueues[which_queue]);
      ktailMulGF61FftShuffleApple.setQueue(&auxQueues[which_queue]);
      ktailMulGF61FftFinalApple.setQueue(&auxQueues[which_queue]);
      ktailMulGF61PairSpecialScalarApple.setQueue(&auxQueues[which_queue]);
      ktailMulGF61PairNormalScalarApple.setQueue(&auxQueues[which_queue]);
#endif
      ktailMulLowGF61.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF61.setQueue(&auxQueues[which_queue]);
#if defined(__APPLE__)
      kfftMidOutGF61LoadScalarApple.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF61MulScalarApple.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF61FftApple.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF61Mul2ScalarApple.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF61WriteScalarApple.setQueue(&auxQueues[which_queue]);
#endif
      kfftWGF61.setQueue(&auxQueues[which_queue]);
#if defined(__APPLE__)
      kfftWGF61LoadScalarApple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthRadixApple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle1Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle4Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle8Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle16Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle64Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle256Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61TwiddleShuffle512Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthFinalApple.setQueue(&auxQueues[which_queue]);
      kfftWGF61LoadStage1FusedApple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused1Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused4Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused8Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused16Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused64Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused256Apple.setQueue(&auxQueues[which_queue]);
      kfftWGF61WidthStageFused512Apple.setQueue(&auxQueues[which_queue]);
#endif
    }
    which_queue++;
  }

  if (fft.FFT_FP64 || fft.FFT_FP32) {
    if (which_queue != -1) {
      kfftMidIn.setQueue(&auxQueues[which_queue]);
      kfftHin.setQueue(&auxQueues[which_queue]);
      ktailSquareZero.setQueue(&auxQueues[which_queue]);
      ktailSquare.setQueue(&auxQueues[which_queue]);
      ktailMul.setQueue(&auxQueues[which_queue]);
      ktailMulLow.setQueue(&auxQueues[which_queue]);
      kfftMidOut.setQueue(&auxQueues[which_queue]);
      kfftW.setQueue(&auxQueues[which_queue]);
    }
    // For no particularly good reason, put kernels that operate on 32-bit value in the same queue unless there are no kernels operating on 64-bit values
    if (fft.FFT_FP64 || (fft.FFT_FP32 && which_queue == -1)) {
      which_queue++;
    }
  }

  if (fft.NTT_GF31) {
    if (which_queue != -1) {
      kfftMidInGF31.setQueue(&auxQueues[which_queue]);
      kfftHinGF31.setQueue(&auxQueues[which_queue]);
      ktailSquareZeroGF31.setQueue(&auxQueues[which_queue]);
      ktailSquareGF31.setQueue(&auxQueues[which_queue]);
      ktailMulGF31.setQueue(&auxQueues[which_queue]);
      ktailMulLowGF31.setQueue(&auxQueues[which_queue]);
      kfftMidOutGF31.setQueue(&auxQueues[which_queue]);
      kfftWGF31.setQueue(&auxQueues[which_queue]);
    }
    //which_queue++;
  }
}

void Gpu::mergeQueue(void) {

  // If MULTI_Q -use not set, return
  if (!args.value("MULTI_Q", 0)) return;

  // Queue a sync event in each auxiliary queue(s).  Wait on the event(s) in the main queue.
  for (size_t i = 0; i < auxQueues.size(); ++i) {
    EventHolder event = auxQueues[i].createSyncEvent();
    queue.waitForSyncEvent(&event);
  }

  // Return kernels to running on the main queue
  // NOTE: I believe there is no need to switch queues back and forth between the main and auxiliary queues.  No one currently uses the cache_group == 0 option.
  if (fft.NTT_GF61) {
    kfftMidInGF61.setQueue(&queue);
#if defined(__APPLE__)
    kfftMidInGF61LoadScalarApple.setQueue(&queue);
    kfftMidInGF61Mul2FactorScalarApple.setQueue(&queue);
    kfftMidInGF61ApplyScalarApple.setQueue(&queue);
    kfftMidInGF61FftApple.setQueue(&queue);
    kfftMidInGF61MulApple.setQueue(&queue);
    kfftMidInGF61TransposeApple.setQueue(&queue);
    ktailSquareZeroGF61LoadApple.setQueue(&queue);
    ktailSquareZeroGF61FftRadixApple.setQueue(&queue);
    ktailSquareZeroGF61FftTwiddleApple.setQueue(&queue);
    ktailSquareZeroGF61FftShuffleApple.setQueue(&queue);
    ktailSquareZeroGF61FftFinalApple.setQueue(&queue);
    ktailSquareZeroGF61ReverseGlobalApple.setQueue(&queue);
    ktailSquareZeroGF61PairApple.setQueue(&queue);
    ktailSquareZeroGF61WriteDirectApple.setQueue(&queue);
    ktailSquareGF61LoadStageFusedApple.setQueue(&queue);
    ktailSquareGF61StageFusedApple.setQueue(&queue);
    ktailSquareGF61PairCrossFusedApple.setQueue(&queue);
    ktailSquareGF61FinalPairFirstFusedApple.setQueue(&queue);
#endif
    kfftHinGF61.setQueue(&queue);
#if defined(__APPLE__)
    kfftHinGF61LoadScalarApple.setQueue(&queue);
    kfftHinGF61FftRadixApple.setQueue(&queue);
    kfftHinGF61FftTwiddleApple.setQueue(&queue);
    kfftHinGF61FftShuffleApple.setQueue(&queue);
    kfftHinGF61FftFinalApple.setQueue(&queue);
#endif
    ktailSquareZeroGF61.setQueue(&queue);
    ktailSquareGF61.setQueue(&queue);
    ktailMulGF61.setQueue(&queue);
#if defined(__APPLE__)
    ktailMulGF61LoadScalarApple.setQueue(&queue);
    ktailMulGF61FftRadixApple.setQueue(&queue);
    ktailMulGF61FftTwiddleApple.setQueue(&queue);
    ktailMulGF61FftShuffleApple.setQueue(&queue);
    ktailMulGF61FftFinalApple.setQueue(&queue);
    ktailMulGF61PairSpecialScalarApple.setQueue(&queue);
    ktailMulGF61PairNormalScalarApple.setQueue(&queue);
#endif
    ktailMulLowGF61.setQueue(&queue);
    kfftMidOutGF61.setQueue(&queue);
#if defined(__APPLE__)
    kfftMidOutGF61LoadScalarApple.setQueue(&queue);
    kfftMidOutGF61MulScalarApple.setQueue(&queue);
    kfftMidOutGF61FftApple.setQueue(&queue);
    kfftMidOutGF61Mul2ScalarApple.setQueue(&queue);
    kfftMidOutGF61WriteScalarApple.setQueue(&queue);
#endif
    kfftWGF61.setQueue(&queue);
#if defined(__APPLE__)
    kfftWGF61LoadScalarApple.setQueue(&queue);
    kfftWGF61WidthRadixApple.setQueue(&queue);
    kfftWGF61TwiddleShuffle1Apple.setQueue(&queue);
    kfftWGF61TwiddleShuffle4Apple.setQueue(&queue);
    kfftWGF61TwiddleShuffle8Apple.setQueue(&queue);
    kfftWGF61TwiddleShuffle16Apple.setQueue(&queue);
    kfftWGF61TwiddleShuffle64Apple.setQueue(&queue);
    kfftWGF61TwiddleShuffle256Apple.setQueue(&queue);
    kfftWGF61TwiddleShuffle512Apple.setQueue(&queue);
    kfftWGF61WidthFinalApple.setQueue(&queue);
    kfftWGF61LoadStage1FusedApple.setQueue(&queue);
    kfftWGF61WidthStageFused1Apple.setQueue(&queue);
    kfftWGF61WidthStageFused4Apple.setQueue(&queue);
    kfftWGF61WidthStageFused8Apple.setQueue(&queue);
    kfftWGF61WidthStageFused16Apple.setQueue(&queue);
    kfftWGF61WidthStageFused64Apple.setQueue(&queue);
    kfftWGF61WidthStageFused256Apple.setQueue(&queue);
    kfftWGF61WidthStageFused512Apple.setQueue(&queue);
#endif
  }
  if (fft.FFT_FP64 || fft.FFT_FP32) {
    kfftMidIn.setQueue(&queue);
    kfftHin.setQueue(&queue);
    ktailSquareZero.setQueue(&queue);
    ktailSquare.setQueue(&queue);
    ktailMul.setQueue(&queue);
    ktailMulLow.setQueue(&queue);
    kfftMidOut.setQueue(&queue);
    kfftW.setQueue(&queue);
  }
  if (fft.NTT_GF31) {
    kfftMidInGF31.setQueue(&queue);
    kfftHinGF31.setQueue(&queue);
    ktailSquareZeroGF31.setQueue(&queue);
    ktailSquareGF31.setQueue(&queue);
    ktailMulGF31.setQueue(&queue);
    ktailMulLowGF31.setQueue(&queue);
    kfftMidOutGF31.setQueue(&queue);
    kfftWGF31.setQueue(&queue);
  }
}

// Replay the recorded bottom half kernels in a cache friendly order.  We support several
// options here using multiple openCl command queues.
void Gpu::replay(void) {

  // If using multiple command queues, handle that now.
  splitQueue();

  // For better L2 cache locality, operate on all the FP data, then operate on all the GF31 data, then GF61.
  for (int cache_group = 1; cache_group <= NUM_CACHE_GROUPS; ++cache_group) {

    // Check for irrelevant cache gouup
    if (cache_group == 1 && !(fft.FFT_FP64 || fft.FFT_FP32)) continue;
    if (cache_group == 2 && !fft.NTT_GF31) continue;
    if (cache_group == 3 && !fft.NTT_GF61) continue;

    // Iterate over the recorded kernels
    int arg = 0;
    for (auto kern : recorded_kernels) {

      // Call the appropriate kernel
      if (kern == KMIDIN) {
        Buffer<double> *buf = recorded_kernel_args[arg++];
        // If not in place, the input is from the scratch buffer
        Buffer<double> *in = in_place ? buf : &buf3;
        Buffer<double> *out = buf;
        if (cache_group == 1) kfftMidIn(*out, *in);
        if (cache_group == 2) kfftMidInGF31(*out, *in);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161) {
            // Apple scalarizes the first middle-in phase: source load, trig
            // factor generation, and element-wise application are separate
            // pipelines with one GF61 value per work-item. The original input
            // scratch becomes the factor buffer after the load. FFT and final
            // LDS transpose preserve the stock algorithm and data layout.
            kfftMidInGF61LoadScalarApple(*out, *in); appleStageFinish();
            kfftMidInGF61Mul2FactorScalarApple(*in); appleStageFinish();
            kfftMidInGF61ApplyScalarApple(*out, *in); appleStageFinish();
            kfftMidInGF61FftApple(*in, *out); appleStageFinish();
            kfftMidInGF61MulApple(*in); appleStageFinish();
            kfftMidInGF61TransposeApple(*out, *in); appleStageFinish();
          } else {
            kfftMidInGF61(*out, *in);
          }
#else
          kfftMidInGF61(*out, *in);
#endif
        }
      }

      if (kern == KFFTHIN) {
        Buffer<double> *out = recorded_kernel_args[arg++];
        Buffer<double> *in = recorded_kernel_args[arg++];
        if (cache_group == 1) kfftHin(*out, *in);
        if (cache_group == 2) kfftHinGF31(*out, *in);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161) {
            // Exact global-memory decomposition of the upstream fftHinGF61.
            // The source GF61 plane is free after the scalar load and serves
            // as the alternate ping-pong bank; no extra transform allocation.
            kfftHinGF61LoadScalarApple(*out, *in);
            Buffer<double>* current = out;
            const u32 groupSize = SMALL_H / nH;
            for (u32 stage = 1; stage < groupSize; stage *= nH) {
              kfftHinGF61FftRadixApple(*current);
              kfftHinGF61FftTwiddleApple(*current, stage);
              Buffer<double>* next = current == out ? in : out;
              kfftHinGF61FftShuffleApple(*current, *next, stage);
              current = next;
            }
            if (current == out) kfftHinGF61FftRadixApple(*out);
            else kfftHinGF61FftFinalApple(*current, *out);
          } else {
            kfftHinGF61(*out, *in);
          }
#else
          kfftHinGF61(*out, *in);
#endif
        }
      }

      if (kern == KTAILSQUARE) {
        Buffer<double> *buf = recorded_kernel_args[arg++];
        // If not in place, the output is to the scratch buffer
        Buffer<double> *in = buf;
        Buffer<double> *out = in_place ? buf : &buf3;
        if (!tail_single_kernel) {
          if (cache_group == 1) ktailSquareZero(*out, *in);
          if (cache_group == 2) ktailSquareZeroGF31(*out, *in);
          if (cache_group == 3) {
#if defined(__APPLE__)
            if (fft.shape.fft_type == FFT3161) {
              ktailSquareZeroGF61LoadApple(bufAppleTailZeroGF61, *in);
              appleStageFinish();
              auto runAppleTailZeroGF61Fft = [&]() {
                const u32 groupSize = SMALL_H / nH;
                u32 bank = 0;
                for (u32 stage = 1; stage < groupSize; stage *= nH) {
                  ktailSquareZeroGF61FftRadixApple(bufAppleTailZeroGF61, bank);
                  appleStageFinish();
                  ktailSquareZeroGF61FftTwiddleApple(bufAppleTailZeroGF61, bufTrigH, bank, stage);
                  appleStageFinish();
                  const u32 nextBank = bank ^ 1u;
                  ktailSquareZeroGF61FftShuffleApple(bufAppleTailZeroGF61, bank, nextBank, stage);
                  appleStageFinish();
                  bank = nextBank;
                }
                ktailSquareZeroGF61FftFinalApple(bufAppleTailZeroGF61, bank);
                appleStageFinish();
              };
              runAppleTailZeroGF61Fft();
              ktailSquareZeroGF61ReverseGlobalApple(bufAppleTailZeroGF61, 0u, 1u);
              appleStageFinish();
              ktailSquareZeroGF61PairApple(bufAppleTailZeroGF61, bufTrigH, 1u);
              appleStageFinish();
              ktailSquareZeroGF61ReverseGlobalApple(bufAppleTailZeroGF61, 1u, 0u);
              appleStageFinish();
              runAppleTailZeroGF61Fft();
              ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, 0u, 0u);
              appleStageFinish();
              ktailSquareZeroGF61WriteDirectApple(*out, bufAppleTailZeroGF61, SMALL_H, (fft.shape.middle / 2u) * SMALL_H);
              appleStageFinish();
            } else {
              ktailSquareZeroGF61(*out, *in);
            }
#else
            ktailSquareZeroGF61(*out, *in);
#endif
          }
        }
        if (cache_group == 1) ktailSquare(*out, *in);
        if (cache_group == 2) ktailSquareGF31(*out, *in);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161) {
            const u32 groupSize = SMALL_H / nH;
            if (apple_fused_tailsquare_gf61 && groupSize > 1) {
              // First height FFT: fuse transpose load with the first
              // radix/twiddle/shuffle stage, then use one fused global
              // scatter kernel per remaining stage.
              ktailSquareGF61LoadStageFusedApple(*out, *in, bufTrigH, 1u);
              appleStageFinish();
              Buffer<double> *current = out;
              for (u32 stage = nH; stage < groupSize; stage *= nH) {
                Buffer<double> *next = current == out ? in : out;
                ktailSquareGF61StageFusedApple(*current, *next, bufTrigH, stage);
                appleStageFinish();
                current = next;
              }
              if (apple_bridge_fused_tailsquare_gf61) {
                // Compose the forward final radix, exact R-pair-R operation
                // and inverse f=1 stage into one private-register bridge.
                Buffer<double> *next = current == out ? in : out;
                ktailSquareGF61FinalPairFirstFusedApple(*current, *next, bufTrigH);
                appleStageFinish();
                current = next;
                for (u32 stage = nH; stage < groupSize; stage *= nH) {
                  next = current == out ? in : out;
                  ktailSquareGF61StageFusedApple(*current, *next, bufTrigH, stage);
                  appleStageFinish();
                  current = next;
                }
              } else {
                // Validated v0.3.54 path retained for differential testing and
                // automatic fallback when the bridge kernel is unavailable.
                ktailSquareGF61FftFinalApple(*current, *out);
                appleStageFinish();
                ktailSquareGF61PairCrossFusedApple(*out, *in, bufTrigH);
                appleStageFinish();
                current = in;
                for (u32 stage = 1; stage < groupSize; stage *= nH) {
                  Buffer<double> *next = current == out ? in : out;
                  ktailSquareGF61StageFusedApple(*current, *next, bufTrigH, stage);
                  appleStageFinish();
                  current = next;
                }
              }
              ktailSquareGF61FftFinalApple(*current, *out);
              appleStageFinish();
            } else {
              // Conservative v0.3.53 path.  It remains available for
              // differential validation with AEVUM_APPLE_TAILSQUARE_LEGACY=1.
              ktailSquareGF61LoadScalarApple(*out, *in);
              appleStageFinish();
              auto runAppleTailGF61FftLegacy = [&]() {
                Buffer<double> *current = out;
                for (u32 stage = 1; stage < groupSize; stage *= nH) {
                  ktailSquareGF61FftRadixApple(*current);
                  appleStageFinish();
                  ktailSquareGF61FftTwiddleApple(*current, bufTrigH, stage);
                  appleStageFinish();
                  Buffer<double> *next = current == out ? in : out;
                  ktailSquareGF61FftShuffleApple(*current, *next, stage);
                  appleStageFinish();
                  current = next;
                }
                ktailSquareGF61FftFinalApple(*current, *out);
                appleStageFinish();
              };
              runAppleTailGF61FftLegacy();
              ktailSquareGF61ReverseCrossApple(*out, *in);
              appleStageFinish();
              ktailSquareGF61PairApple(*in, bufTrigH);
              appleStageFinish();
              ktailSquareGF61ReverseCrossApple(*in, *out);
              appleStageFinish();
              runAppleTailGF61FftLegacy();
            }
          } else {
            ktailSquareGF61(*out, *in);
          }
#else
          ktailSquareGF61(*out, *in);
#endif
        }
      }

      if (kern == KTAILMUL) {
        Buffer<double> *buf = recorded_kernel_args[arg++];
        Buffer<double> *in2 = recorded_kernel_args[arg++];
        // If not in place, the output is to the scratch buffer
        Buffer<double> *in1 = buf;
        Buffer<double> *out = in_place ? buf : &buf3;
        if (cache_group == 1) ktailMul(*out, *in1, *in2);
        if (cache_group == 2) ktailMulGF31(*out, *in1, *in2);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161) {
            if (in_place || out == in1 || out == in2 || in1 == in2)
              throw std::runtime_error("Apple staged GF61 tailMul requires three distinct transform buffers");
            if (!(fft.NTT_GF31 && fft.NTT_GF61) || fft.FFT_FP32 || fft.FFT_FP64)
              throw std::runtime_error("Apple staged GF61 tailMul expects the GF31/GF61 Aevum layout");
            const u32 fullBase = GF31_DATA_SIZE(WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size) / 2;
            const u32 rawBase = 0;
            Buffer<double>* raw = &bufAppleTailMulGF61;

            auto runAppleTailMulGF61Fft = [&](Buffer<double>* start, u32 startBase,
                                               Buffer<double>* alternate, u32 alternateBase,
                                               Buffer<double>* final, u32 finalBase) {
              Buffer<double>* current = start;
              u32 currentBase = startBase;
              Buffer<double>* next = alternate;
              u32 nextBase = alternateBase;
              const u32 groupSize = SMALL_H / nH;
              for (u32 stage = 1; stage < groupSize; stage *= nH) {
                ktailMulGF61FftRadixApple(*current, currentBase);
                ktailMulGF61FftTwiddleApple(*current, currentBase, bufTrigH, stage);
                ktailMulGF61FftShuffleApple(*current, currentBase, *next, nextBase, stage);
                std::swap(current, next);
                std::swap(currentBase, nextBase);
              }
              ktailMulGF61FftFinalApple(*current, currentBase, *final, finalBase);
            };

            // First operand: consumed temporary in1 -> final transformed data in out.
            ktailMulGF61LoadScalarApple(*out, fullBase, *in1, fullBase);
            runAppleTailMulGF61Fft(out, fullBase, in1, fullBase, out, fullBase);

            // Preserved multiplicand: copy/transform through in1 and the GF61-only scratch.
            ktailMulGF61LoadScalarApple(*in1, fullBase, *in2, fullBase);
            runAppleTailMulGF61Fft(in1, fullBase, raw, rawBase, in1, fullBase);

            // Scalar direct composition of the stock reverse/pairMul/reverse
            // sequence.  Special self-paired lines and normal partner lines
            // are separate pipelines, each keeping only four GF61 values.
            ktailMulGF61PairSpecialScalarApple(*raw, rawBase, *out, fullBase, *in1, fullBase, bufTrigH);
            ktailMulGF61PairNormalScalarApple(*raw, rawBase, *out, fullBase, *in1, fullBase, bufTrigH);
            runAppleTailMulGF61Fft(raw, rawBase, in1, fullBase, out, fullBase);
          } else {
            ktailMulGF61(*out, *in1, *in2);
          }
#else
          ktailMulGF61(*out, *in1, *in2);
#endif
        }
      }

      if (kern == KTAILMULLOW) {
        Buffer<double> *buf = recorded_kernel_args[arg++];
        Buffer<double> *in2 = recorded_kernel_args[arg++];
        // If not in place, the output is to the scratch buffer
        Buffer<double> *in1 = buf;
        Buffer<double> *out = in_place ? buf : &buf3;
        if (cache_group == 1) ktailMulLow(*out, *in1, *in2);
        if (cache_group == 2) ktailMulLowGF31(*out, *in1, *in2);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161) {
            // `in2` is already in the exact fully height-transformed layout
            // produced by fftHinGF61 during regPrepare.  Apple rejects only
            // the stock monolithic MUL_LOW tail kernel, so transform the live
            // operand with the same staged fft_HEIGHT used by generic
            // tailMulGF61, pair it directly with the preserved prepared
            // operand, then apply the unchanged staged output transform.
            if (in_place || out == in1 || out == in2 || in1 == in2)
              throw std::runtime_error("Apple staged GF61 tailMulLow requires three distinct transform buffers");
            if (!(fft.NTT_GF31 && fft.NTT_GF61) || fft.FFT_FP32 || fft.FFT_FP64)
              throw std::runtime_error("Apple staged GF61 tailMulLow expects the GF31/GF61 Aevum layout");
            const u32 fullBase = GF31_DATA_SIZE(WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size) / 2;
            const u32 rawBase = 0;
            Buffer<double>* raw = &bufAppleTailMulGF61;

            auto runAppleTailMulLowGF61Fft = [&](Buffer<double>* start, u32 startBase,
                                                  Buffer<double>* alternate, u32 alternateBase,
                                                  Buffer<double>* final, u32 finalBase) {
              Buffer<double>* current = start;
              u32 currentBase = startBase;
              Buffer<double>* next = alternate;
              u32 nextBase = alternateBase;
              const u32 groupSize = SMALL_H / nH;
              for (u32 stage = 1; stage < groupSize; stage *= nH) {
                ktailMulGF61FftRadixApple(*current, currentBase);
                ktailMulGF61FftTwiddleApple(*current, currentBase, bufTrigH, stage);
                ktailMulGF61FftShuffleApple(*current, currentBase, *next, nextBase, stage);
                std::swap(current, next);
                std::swap(currentBase, nextBase);
              }
              ktailMulGF61FftFinalApple(*current, currentBase, *final, finalBase);
            };

            // Live operand: fftMiddleIn layout -> fully height transformed.
            ktailMulGF61LoadScalarApple(*out, fullBase, *in1, fullBase);
            runAppleTailMulLowGF61Fft(out, fullBase, in1, fullBase, out, fullBase);

            // Prepared operand `in2` is read-only and already in the stock
            // MUL_LOW p/q layout.  Compose the exact reverse/pairMul/reverse
            // coordinates directly into the GF61 scratch plane.
            ktailMulGF61PairSpecialScalarApple(*raw, rawBase, *out, fullBase, *in2, fullBase, bufTrigH);
            ktailMulGF61PairNormalScalarApple(*raw, rawBase, *out, fullBase, *in2, fullBase, bufTrigH);

            // Exact stock fft_HEIGHT2 result back in the fused output layout.
            runAppleTailMulLowGF61Fft(raw, rawBase, in1, fullBase, out, fullBase);
          } else {
            ktailMulLowGF61(*out, *in1, *in2);
          }
#else
          ktailMulLowGF61(*out, *in1, *in2);
#endif
        }
      }

      if (kern == KMIDOUT) {
        Buffer<double> *buf = recorded_kernel_args[arg++];
        // If not in place, the input is from the scratch buffer
        Buffer<double> *in = in_place ? buf : &buf3;
        Buffer<double> *out = buf;
        if (cache_group == 1) kfftMidOut(*out, *in);
        if (cache_group == 2) kfftMidOutGF31(*out, *in);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161 && fft.shape.middle >= 8) {
            kfftMidOutGF61LoadScalarApple(bufAppleTailMulGF61, *in);
            kfftMidOutGF61MulScalarApple(bufAppleTailMulGF61, bufTrigM);
            kfftMidOutGF61FftApple(bufAppleTailMulGF61);
            kfftMidOutGF61Mul2ScalarApple(bufAppleTailMulGF61, bufTrigM);
            kfftMidOutGF61WriteScalarApple(*out, bufAppleTailMulGF61);
          } else {
            kfftMidOutGF61(*out, *in);
          }
#else
          kfftMidOutGF61(*out, *in);
#endif
        }
      }

      if (kern == KFFTW) {
        Buffer<double> *out = recorded_kernel_args[arg++];
        Buffer<double> *in = recorded_kernel_args[arg++];
        if (cache_group == 1) kfftW(*out, *in);
        if (cache_group == 2) kfftWGF31(*out, *in);
        if (cache_group == 3) {
#if defined(__APPLE__)
          if (fft.shape.fft_type == FFT3161) {
            if (out == in) throw std::runtime_error("Apple staged GF61 fftW requires distinct input/output buffers");
            Buffer<double>* current = out;
            Buffer<double>* alternate = in;
            const u32 width_workgroup = WIDTH / nW;
            u32 firstStage = 1;
            if (apple_fused_fftw_load_first) {
              kfftWGF61LoadStage1FusedApple(*out, *in);
              appleStageFinish();
              current = out;
              alternate = in;
              firstStage = nW;
            } else {
              kfftWGF61LoadScalarApple(*out, *in);
              appleStageFinish();
            }
            for (u32 stage = firstStage; stage < width_workgroup; stage *= nW) {
              if (apple_fused_fftw_width) {
                switch (stage) {
                  case 1:   kfftWGF61WidthStageFused1Apple(*alternate, *current); break;
                  case 4:   kfftWGF61WidthStageFused4Apple(*alternate, *current); break;
                  case 8:   kfftWGF61WidthStageFused8Apple(*alternate, *current); break;
                  case 16:  kfftWGF61WidthStageFused16Apple(*alternate, *current); break;
                  case 64:  kfftWGF61WidthStageFused64Apple(*alternate, *current); break;
                  case 256: kfftWGF61WidthStageFused256Apple(*alternate, *current); break;
                  case 512: kfftWGF61WidthStageFused512Apple(*alternate, *current); break;
                  default: throw std::runtime_error("Unsupported Apple fused GF61 fftW width stage " + std::to_string(stage));
                }
                appleStageFinish();
              } else {
                kfftWGF61WidthRadixApple(*current);
                appleStageFinish();
                switch (stage) {
                  case 1:   kfftWGF61TwiddleShuffle1Apple(*alternate, *current); break;
                  case 4:   kfftWGF61TwiddleShuffle4Apple(*alternate, *current); break;
                  case 8:   kfftWGF61TwiddleShuffle8Apple(*alternate, *current); break;
                  case 16:  kfftWGF61TwiddleShuffle16Apple(*alternate, *current); break;
                  case 64:  kfftWGF61TwiddleShuffle64Apple(*alternate, *current); break;
                  case 256: kfftWGF61TwiddleShuffle256Apple(*alternate, *current); break;
                  case 512: kfftWGF61TwiddleShuffle512Apple(*alternate, *current); break;
                  default: throw std::runtime_error("Unsupported Apple GF61 fftW width stage " + std::to_string(stage));
                }
                appleStageFinish();
              }
              std::swap(current, alternate);
            }
            if (current == out) {
              kfftWGF61WidthRadixApple(*out);
              appleStageFinish();
            } else {
              kfftWGF61WidthFinalApple(*out, *current);
              appleStageFinish();
            }
          } else {
            kfftWGF61(*out, *in);
          }
#else
          kfftWGF61(*out, *in);
#endif
        }
      }
    }
  }

  // Empty the recorded kernels queue
  recorded_kernels.clear();
  recorded_kernel_args.clear();

  // If using multiple command queues, go back to a single command queue
  mergeQueue();
}

// Call the appropriate kernels to support hybrid FFTs and NTTs

void Gpu::fftP(Buffer<double>& buf, Buffer<Word>& in) {
  // If not in place, instead write the output to the scratch buffer
  Buffer<double> *out = in_place ? &buf : &buf3;
#if defined(__APPLE__)
  if (fft.shape.fft_type == FFT3161) {
    // Apple-only fully global fftP. The v0.3.48 trace showed that normal and
    // serialized runs already diverged at fftP. GF31 was the last fftP plane
    // still using LDS. Decompose both CRT planes into scalar weighting plus
    // global radix/twiddle/permutation stages. Non-Apple dispatch remains the
    // original monolithic kfftP call below.
    Buffer<double>* alternate = (out == &buf3) ? &buf : &buf3;
    const u32 width_workgroup = WIDTH / nW;

    auto runGF31 = [&]() {
      Buffer<double>* current = out;
      Buffer<double>* next = alternate;
      u32 firstStage = 1;
      if (apple_fused_fftp_weight_first) {
        kfftP31WeightStage1FusedApple(*alternate, in);
        appleStageFinish();
        current = alternate;
        next = out;
        firstStage = nW;
      } else {
        kfftP31WeightScalarApple(*out, in);
        appleStageFinish();
      }
      for (u32 stage = firstStage; stage < width_workgroup; stage *= nW) {
        if (apple_fused_fftp_width) {
          switch (stage) {
            case 1:   kfftP31WidthStageFused1Apple(*next, *current); break;
            case 4:   kfftP31WidthStageFused4Apple(*next, *current); break;
            case 8:   kfftP31WidthStageFused8Apple(*next, *current); break;
            case 16:  kfftP31WidthStageFused16Apple(*next, *current); break;
            case 64:  kfftP31WidthStageFused64Apple(*next, *current); break;
            case 256: kfftP31WidthStageFused256Apple(*next, *current); break;
            case 512: kfftP31WidthStageFused512Apple(*next, *current); break;
            default: throw std::runtime_error("Unsupported Apple fused GF31 width stage " + std::to_string(stage));
          }
          appleStageFinish();
        } else {
          kfftP31WidthRadixApple(*current);
          appleStageFinish();
          switch (stage) {
            case 1:   kfftP31TwiddleShuffle1Apple(*next, *current); break;
            case 4:   kfftP31TwiddleShuffle4Apple(*next, *current); break;
            case 8:   kfftP31TwiddleShuffle8Apple(*next, *current); break;
            case 16:  kfftP31TwiddleShuffle16Apple(*next, *current); break;
            case 64:  kfftP31TwiddleShuffle64Apple(*next, *current); break;
            case 256: kfftP31TwiddleShuffle256Apple(*next, *current); break;
            case 512: kfftP31TwiddleShuffle512Apple(*next, *current); break;
            default: throw std::runtime_error("Unsupported Apple GF31 width stage " + std::to_string(stage));
          }
          appleStageFinish();
        }
        std::swap(current, next);
      }
      if (current == out) kfftP31WidthRadixApple(*out);
      else kfftP31WidthFinalApple(*out, *current);
      appleStageFinish();
    };

    auto runGF61 = [&]() {
      Buffer<double>* current = out;
      Buffer<double>* next = alternate;
      u32 firstStage = 1;
      if (apple_fused_fftp_weight_first) {
        kfftP61WeightStage1FusedApple(*alternate, in);
        appleStageFinish();
        current = alternate;
        next = out;
        firstStage = nW;
      } else {
        kfftP61WeightScalarApple(*out, in);
        appleStageFinish();
      }
      for (u32 stage = firstStage; stage < width_workgroup; stage *= nW) {
        if (apple_fused_fftp_width) {
          switch (stage) {
            case 1:   kfftP61WidthStageFused1Apple(*next, *current); break;
            case 4:   kfftP61WidthStageFused4Apple(*next, *current); break;
            case 8:   kfftP61WidthStageFused8Apple(*next, *current); break;
            case 16:  kfftP61WidthStageFused16Apple(*next, *current); break;
            case 64:  kfftP61WidthStageFused64Apple(*next, *current); break;
            case 256: kfftP61WidthStageFused256Apple(*next, *current); break;
            case 512: kfftP61WidthStageFused512Apple(*next, *current); break;
            default: throw std::runtime_error("Unsupported Apple fused GF61 width stage " + std::to_string(stage));
          }
          appleStageFinish();
        } else {
          kfftP61WidthRadixApple(*current);
          appleStageFinish();
          switch (stage) {
            case 1:   kfftP61TwiddleShuffle1Apple(*next, *current); break;
            case 4:   kfftP61TwiddleShuffle4Apple(*next, *current); break;
            case 8:   kfftP61TwiddleShuffle8Apple(*next, *current); break;
            case 16:  kfftP61TwiddleShuffle16Apple(*next, *current); break;
            case 64:  kfftP61TwiddleShuffle64Apple(*next, *current); break;
            case 256: kfftP61TwiddleShuffle256Apple(*next, *current); break;
            case 512: kfftP61TwiddleShuffle512Apple(*next, *current); break;
            default: throw std::runtime_error("Unsupported Apple GF61 width stage " + std::to_string(stage));
          }
          appleStageFinish();
        }
        std::swap(current, next);
      }
      if (current == out) kfftP61WidthRadixApple(*out);
      else kfftP61WidthFinalApple(*out, *current);
      appleStageFinish();
    };

    runGF31();
    runGF61();
    return;
  }
#endif
  kfftP(*out, in);
}

void Gpu::fftMidIn(Buffer<double>& buf) {
  // Record this call for later playback
  recorded_kernels.push_back(KMIDIN);
  recorded_kernel_args.push_back(&buf);
}

void Gpu::fftHin(Buffer<double>& out, Buffer<double>& in) {
  // Record this call for later playback
  recorded_kernels.push_back(KFFTHIN);
  recorded_kernel_args.push_back(&out);
  recorded_kernel_args.push_back(&in);
}

void Gpu::tailSquare(Buffer<double>& buf) {
  // Record this call for later playback
  recorded_kernels.push_back(KTAILSQUARE);
  recorded_kernel_args.push_back(&buf);
}

void Gpu::tailMul(Buffer<double>& buf, Buffer<double>& in2) {
  // Record this call for later playback
  recorded_kernels.push_back(KTAILMUL);
  recorded_kernel_args.push_back(&buf);
  recorded_kernel_args.push_back(&in2);
}

void Gpu::tailMulLow(Buffer<double>& buf, Buffer<double>& in2) {
  // Record this call for later playback
  recorded_kernels.push_back(KTAILMULLOW);
  recorded_kernel_args.push_back(&buf);
  recorded_kernel_args.push_back(&in2);
}

void Gpu::fftMidOut(Buffer<double>& buf) {
  // Record this call for later playback
  recorded_kernels.push_back(KMIDOUT);
  recorded_kernel_args.push_back(&buf);
}

void Gpu::fftW(Buffer<double>& out, Buffer<double>& in) {
  // Record this call for later playback
  recorded_kernels.push_back(KFFTW);
  recorded_kernel_args.push_back(&out);
  recorded_kernel_args.push_back(&in);
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
}

void Gpu::carryA(Buffer<Word>& out, Buffer<double>& in) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryAROE(out, in, roePos++)
                   : kCarryA(out, in, updateCarryPos(1 << 2));
}

void Gpu::carryM(Buffer<Word>& out, Buffer<double>& in) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryMROE(out, in, roePos++)
                   : kCarryM(out, in, updateCarryPos(1 << 3));
}

void Gpu::carryLL(Buffer<Word>& out, Buffer<double>& in) {
  kCarryLL(out, in, updateCarryPos(1 << 2));
}

void Gpu::carryFused(Buffer<double>& buf) {
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
  assert(roePos <= ROE_SIZE);
  // Like fftP, if not in place write the output to the scratch buffer
  Buffer<double> *in = &buf;
  Buffer<double> *out = in_place ? &buf : &buf3;
  roePos < wantROE ? kCarryFusedROE(*out, *in, roePos++)
                   : kCarryFused(*out, *in, updateCarryPos(1 << 0));
}

void Gpu::carryFusedMul(Buffer<double>& buf) {
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
  assert(roePos <= ROE_SIZE);
  // Like fftP, if not in place write the output to the scratch buffer
  Buffer<double> *in = &buf;
  Buffer<double> *out = in_place ? &buf : &buf3;
  roePos < wantROE ? kCarryFusedMulROE(*out, *in, roePos++)
                   : kCarryFusedMul(*out, *in, updateCarryPos(1 << 1));
}

void Gpu::carryFusedLL(Buffer<double>& buf) {
  // This kernel always ends the "bottom half".  Replay the recorded kernel calls.
  replay();
  // Like fftP, if not in place write the output to the scratch buffer
  Buffer<double> *in = &buf;
  Buffer<double> *out = in_place ? &buf : &buf3;
  kCarryFusedLL(*out, *in, updateCarryPos(1 << 0));
}


#if 0
void Gpu::measureTransferSpeed() {
  u32 SIZE_MB = 16;
  vector<double> data(SIZE_MB * 1024 * 1024, 1);
  Buffer<double> buf{profile.make("DMA"), &queue, SIZE};

  Timer t;
  for (int i = 0; i < 4; ++i) {
    buf.write(data);
    log("buffer Write : %f GB/s\n", double(SIZE / 1024 / 1024) * sizeof(double) / (1024 * t.reset()));
  }

  for (int i = 0; i < 4; ++i) {
    buf.read(data);
    // queue.finish();
    log("buffer READ : %f GB/s\n", double(SIZE / 1024 / 1024) * sizeof(double) / (1024 * t.reset()));
  }

  queue.finish();
}
#endif

u32 Gpu::updateCarryPos(u32 bit) {
  return (statsBits & bit) && (carryPos < CARRY_SIZE) ? carryPos++ : carryPos;
}

vector<Buffer<Word>> Gpu::makeBufVector(u32 size) {
  vector<Buffer<Word>> r;
  const size_t words = N * fft.WordSize / sizeof(Word);
  for (u32 i = 0; i < size; ++i) { r.emplace_back(timeBufVect, &queue, words); }
  return r;
}

vector<Buffer<double>> Gpu::makeTransformBufVector(u32 size) {
  vector<Buffer<double>> r;
  const size_t transform_words = TOTAL_DATA_SIZE(fft, WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size);
  for (u32 i = 0; i < size; ++i) { r.emplace_back(timeBufVect, &queue, transform_words); }
  return r;
}


#if defined(__APPLE__)
void Gpu::appleStageFinish() {
  if (apple_stage_finish) queue.finish();
}
#endif

namespace {
uint64_t aevumTraceHashBytes(const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}
}

void Gpu::regDebugSquareTrace(Buffer<Word>& io, u64* trace, size_t trace_count) {
  if (!trace || trace_count < 12) throw std::runtime_error("square trace requires 12 uint64 values");
  std::fill(trace, trace + trace_count, 0);

  const size_t gf31_doubles = fft.NTT_GF31 ? GF31_DATA_SIZE(WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size) : 0;
  const size_t gf61_doubles = fft.NTT_GF61 ? GF61_DATA_SIZE(WIDTH, fft.shape.middle, SMALL_H, in_place, pad_size) : 0;

  auto hashTransform = [&](Buffer<double>& b, size_t slot) {
    queue.finish();
    auto values = b.read();
    if (gf31_doubles) trace[slot] = aevumTraceHashBytes(values.data(), gf31_doubles * sizeof(double));
    if (gf61_doubles) trace[slot + 1] = aevumTraceHashBytes(values.data() + gf31_doubles, gf61_doubles * sizeof(double));
  };

  fftP(buf1, io);
  hashTransform(in_place ? buf1 : buf3, 0);

  fftMidIn(buf1);
  replay();
  hashTransform(buf1, 2);

  tailSquare(buf1);
  replay();
  hashTransform(in_place ? buf1 : buf3, 4);

  fftMidOut(buf1);
  replay();
  hashTransform(buf1, 6);

  fftW(buf3, buf1);
  hashTransform(buf3, 8);

  carryA(io, buf3);
  carryB(io);
  queue.finish();
  auto words = io.read();
  trace[10] = aevumTraceHashBytes(words.data(), words.size() * sizeof(Word));
  trace[11] = words.empty() ? 0 : static_cast<uint64_t>(words[0]);
}

void Gpu::regSync() { queue.finish(); }

void Gpu::regCopy(Buffer<Word>& dst, const Buffer<Word>& src) { dst << src; }

void Gpu::regWrite(Buffer<Word>& dst, const Words& words) { writeIn(dst, words); }

Words Gpu::regRead(Buffer<Word>& src) { return readAndCompress(src); }

void Gpu::regAddWords(Buffer<Word>& dst, const Buffer<Word>& src) { regAdd(dst, src); }

void Gpu::regSubWords(Buffer<Word>& dst, const Buffer<Word>& src) { regSub(dst, src); }

void Gpu::regSetU32(Buffer<Word>& dst, u32 value) {
#if defined(__APPLE__)
  // The register representation is balanced-digit and transposed.  Writing a
  // raw first device word is not equivalent to importing an integer (notably
  // once the low digit overflows its signed range), and Apple OpenCL may also
  // return stale data after clEnqueueFillBuffer on these large register
  // buffers.  Reuse the canonical compact-word upload path instead.
  writeIn(dst, makeWords(E, value));
#else
  // Preserve the upstream fast constant initialization on non-Apple builds.
  dst.set(static_cast<Word>(value));
#endif
}

void Gpu::regSubU32(Buffer<Word>& dst, u32 value) { regSubValue(dst, value); }

bool Gpu::regEqual(Buffer<Word>& lhs, Buffer<Word>& rhs) { return isEqual(lhs, rhs); }

void Gpu::regSquare(Buffer<Word>& io, u32 factor) {
  if (factor != 1) throw std::runtime_error("regSquare factor must be handled by EngineApi");
  square(io, io, LEAD_NONE, LEAD_NONE, false, false);
}

void Gpu::regPrepare(Buffer<Word>& src) {
#if defined(__APPLE__)
  // Preserve the exact FFT3161 arithmetic while avoiding the Apple-only
  // staged generic tailMulGF61 path.  The prepared multiplicand is advanced
  // one unchanged upstream step further (fftHin) and is then consumed by the
  // existing MUL_LOW tail kernel.
  fftP(buf1, src);
  fftMidIn(buf1);
  fftHin(buf2, buf1);
#else
  fftP(buf1, src);
  fftMidIn(buf1);
#endif
  replay();
}

void Gpu::regPrepare(Buffer<double>& prepared, Buffer<Word>& src) {
#if defined(__APPLE__)
  fftP(buf1, src);
  fftMidIn(buf1);
  fftHin(prepared, buf1);
#else
  fftP(prepared, src);
  fftMidIn(prepared);
#endif
  replay();
}

void Gpu::regMulPrepared(Buffer<Word>& dst, u32 factor) {
  if (factor != 1) throw std::runtime_error("regMulPrepared factor must be handled by EngineApi");
#if defined(__APPLE__)
  fftP(buf1, dst);
  fftMidIn(buf1);
  tailMulLow(buf1, buf2);
  fftMidOut(buf1);
  fftW(buf3, buf1);
  if (mulRoePos.empty() || mulRoePos.back() < roePos) mulRoePos.push_back(roePos);
  carryA(dst, buf3);
  carryB(dst);
#else
  mul(dst, buf1, buf2, false);
#endif
}

void Gpu::regMulPrepared(Buffer<Word>& dst, Buffer<double>& prepared, u32 factor) {
  if (factor != 1) throw std::runtime_error("regMulPrepared factor must be handled by EngineApi");
#if defined(__APPLE__)
  fftP(buf1, dst);
  fftMidIn(buf1);
  tailMulLow(buf1, prepared);
  fftMidOut(buf1);
  fftW(buf3, buf1);
  if (mulRoePos.empty() || mulRoePos.back() < roePos) mulRoePos.push_back(roePos);
  carryA(dst, buf3);
  carryB(dst);
#else
  mul(dst, prepared, buf1, false);
#endif
}

void Gpu::regMul(Buffer<Word>& dst, Buffer<Word>& src, u32 factor) {
  if (factor != 1) throw std::runtime_error("regMul factor must be handled by EngineApi");
  modMul(dst, src, false);
}

pair<RoeInfo, RoeInfo> Gpu::readROE() {
  assert(roePos <= ROE_SIZE);
  if (roePos) {
    vector<float> roe = bufROE.read(roePos);
    assert(roe.size() == roePos);
    bufROE.zero(roePos);
    roePos = 0;
    auto [squareRoe, mulRoe] = split(roe, mulRoePos);
    mulRoePos.clear();
    return {roeStat(squareRoe), roeStat(mulRoe)};
  } else {
    return {};
  }
}

RoeInfo Gpu::readCarryStats() {
  assert(carryPos <= CARRY_SIZE);
  if (carryPos == 0) { return {}; }
  vector<float> carry = bufStatsCarry.read(carryPos);
  assert(carry.size() == carryPos);
  bufStatsCarry.zero(carryPos);
  carryPos = 0;

  RoeInfo ret = roeStat(carry);

#if 0
  log("%s\n", ret.toString().c_str());

  std::sort(carry.begin(), carry.end());
  File fo = File::openAppend("carry.txt");
  auto it = carry.begin();
  u32 n = carry.size();
  u32 c = 0;
  for (int i=0; i < 500; ++i) {
    double y = 0.23 + (0.48 - 0.23) / 500 * i;
    while (it < carry.end() && *it < y) {
      ++c;
      ++it;
    }
    fo.printf("%f %f\n", y, c / double(n));
  }

  // for (auto x : carry) { fo.printf("%f\n", x); }
  fo.printf("\n\n");
#endif

  return ret;
}

template<typename T>
static bool isAllZero(vector<T> v) { return std::all_of(v.begin(), v.end(), [](T x) { return x == 0;}); }

// Read from GPU, verifying the transfer, and retry on failure.
vector<Word> Gpu::readChecked(Buffer<Word>& buf) {
#if defined(__APPLE__)
  // Apple's legacy OpenCL implementation gives nondeterministic results for
  // the highly contended 64-bit checksum assembled from global 32-bit atomics.
  // Verify the transfer instead with two independently enqueued synchronous
  // transpose/read operations and compare every word.  This is only used on
  // infrequent host reads (save/check/residue), not in the arithmetic hot path.
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    queue.finish();
    vector<Word> first = readOut(buf);
    queue.finish();
    vector<Word> second = readOut(buf);
    if (first == second) {
      if (isAllZero(first)) {
        log("Read ZERO\n");
        return {};
      }
      return first;
    }

    size_t mismatch = 0;
    while (mismatch < first.size() && first[mismatch] == second[mismatch]) ++mismatch;
    log("GPU double-read mismatch at word %zu: %016" PRIx64 " != %016" PRIx64 "\n",
        mismatch, mismatch < first.size() ? u64(first[mismatch]) : 0,
        mismatch < second.size() ? u64(second[mismatch]) : 0);
  }
  throw "GPU persistent double-read errors";
#else
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    bufSumOut.zero();
    sum64(bufSumOut, N, buf);
    vector<u64> expectedVect(1);
    bufSumOut.readAsync(expectedVect);

    vector<Word> data = readOut(buf);
    u64 hostSum = 0;
    for (auto it = data.begin(), end = data.end(); it < end; ++it) hostSum += u64(*it);

    u64 gpuSum = expectedVect[0];
    if (hostSum == gpuSum) {
      // A buffer containing all-zero is exceptional, so mark that through the empty vector.
      if (gpuSum == 0 && isAllZero(data)) {
        log("Read ZERO\n");
        return {};
      }
      return data;
    }

    log("GPU read failed: %016" PRIx64 " (gpu) != %016" PRIx64 " (host)\n", gpuSum, hostSum);
  }
  throw "GPU persistent read errors";
#endif
}

Words Gpu::readAndCompress(Buffer<Word>& buf)  { return compactBits(readChecked(buf), E); }
vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
vector<u32> Gpu::readData() { return readAndCompress(bufData); }

// ioA := ioA * inB; inB must be the output of fftMidIn; inB is preserved
void Gpu::mul(Buffer<Word>& ioA, Buffer<double>& inB, Buffer<double>& tmp1, bool mul3) {
  fftP(tmp1, ioA);
  fftMidIn(tmp1);
  tailMul(tmp1, inB);
  fftMidOut(tmp1);
  fftW(buf3, tmp1);

  // Register the current ROE pos as multiplication (vs. a squaring)
  if (mulRoePos.empty() || mulRoePos.back() < roePos) { mulRoePos.push_back(roePos); }

  if (mul3) { carryM(ioA, buf3); } else { carryA(ioA, buf3); }
  carryB(ioA);
}

// ioA := ioA * inB; inB will end up in buf1 in the LEAD_MIDDLE state
void Gpu::modMul(Buffer<Word>& ioA, Buffer<Word>& inB, bool mul3) {
  modMul(ioA, inB, LEAD_NONE, mul3);
};

// ioA := ioA * inB; inB will end up in buf1 in the LEAD_MIDDLE state
void Gpu::modMul(Buffer<Word>& ioA, Buffer<Word>& inB, enum LEAD_TYPE leadInB, bool mul3) {
  if (leadInB == LEAD_NONE) fftP(buf1, inB);
  if (leadInB != LEAD_MIDDLE) fftMidIn(buf1);
  replay();  // Work around an odd bug.  The above executed fftP writing to buf3 if !in_place and queued fftMidIn.  If we don't replay now, mul will call fftP again overwriting buf3.
  mul(ioA, buf1, buf2, mul3);
};

void Gpu::writeState(u64 k, const vector<u32>& check, u32 blockSize) {
  assert(blockSize > 0);
  writeIn(bufCheck, check);

  bufData << bufCheck;
  bufAux  << bufCheck;

  if (k) {  // Only verify bufData that was read in from a save file
    u32 n;
    for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
      squareLoop(bufData, 0, n);
      modMul(bufData, bufAux);
      bufAux << bufData;
    }

    assert((n & (n - 1)) == 0);
    assert(blockSize % n == 0);

    blockSize /= n;
    assert(blockSize >= 2);

    for (u32 i = 0; i < blockSize - 2; ++i) {
      squareLoop(bufData, 0, n);
      modMul(bufData, bufAux);
    }

    squareLoop(bufData, 0, n);
  }
  modMul(bufData, bufAux, true);
}

bool Gpu::doCheck(u32 blockSize) {
  squareLoop(bufAux, bufCheck, 0, blockSize, true);
  modMul(bufCheck, bufData);
  return isEqual(bufCheck, bufAux);
}

void Gpu::logTimeKernels() {
  auto prof = profile.get();
  u64 total = 0;
  for (const TimeInfo* p : prof) { total += p->times[2]; }
  if (!total) { return; } // no profile
  
  char buf[256];
  // snprintf(buf, sizeof(buf), "Profile:\n ");

  string s = "Profile:\n";
  for (const TimeInfo* p : prof) {
    u32 n = p->n;
    assert(n);
    double f = 1e-3 / n;
    double percent = 100.0 / total * p->times[2];
    if (!args.verbose && percent < 0.2) { break; }
    snprintf(buf, sizeof(buf),
             args.verbose ? "%s %5.2f%% %-11s : %6.0f us/call x %5d calls  (%.3f %.0f)\n"
                          : "%s %5.2f%% %-11s %4.0f x%6d  %.3f %.0f\n",
             logContext().c_str(),
             percent, p->name.c_str(), p->times[2] * f, n, p->times[0] * (f * 1e-3), p->times[1] * (f * 1e-3));
    s += buf;
  }
  log("%s", s.c_str());
  // log("Total time %.3fs\n", total * 1e-9);
  profile.reset();
}

vector<Word> Gpu::readWords(Buffer<Word> &buf) {
  // GPU is returning either 4-byte or 8-byte integers.  C++ code is expecting 8-byte integers.  Handle the "no conversion" case.
  if (fft.WordSize == 8) return buf.read();
  // Convert 32-bit GPU Words into 64-bit C++ Words
  vector<Word> GPUdata = buf.read();
  vector<Word> CPUdata;
  CPUdata.resize(GPUdata.size() * 2);
  for (u32 i = 0; i < GPUdata.size(); ++i) {
    CPUdata[2*i] = (i32) GPUdata[i];
    CPUdata[2*i+1] = (GPUdata[i] >> 32);
  }
  return CPUdata;
}

void Gpu::writeWords(Buffer<Word>& buf, vector<Word> &words) {
  // GPU is expecting either 4-byte or 8-byte integers.  C++ code is using 8-byte integers.  Handle the "no conversion" case.
  if (fft.WordSize == 8) buf.write(std::move(words));
  // Convert 64-bit C++ Words into 32-bit GPU Words
  else {
    vector<Word> GPUdata;
    GPUdata.resize(words.size() / 2);
    assert((words.size() & 1) == 0);
    for (u32 i = 0; i < words.size(); i += 2) {
      GPUdata[i/2] = ((i64) words[i+1] << 32) | (u32) words[i];
    }
    buf.write(std::move(GPUdata));
  }
}

vector<Word> Gpu::readOut(Buffer<Word> &buf) {
  transpOut(bufAux, buf);
  return readWords(bufAux);
}

void Gpu::writeIn(Buffer<Word>& buf, const vector<u32>& words) { writeIn(buf, expandBits(words, N, E)); }

void Gpu::writeIn(Buffer<Word>& buf, vector<Word>&& words) {
  writeWords(bufAux, words);
  transpIn(buf, bufAux);
}

Words Gpu::expExp2(const Words& A, u32 n) {
  u32 logStep   = 10000;
  u32 blockSize = 100;
  
  writeIn(bufData, std::move(A));
  IterationTimer timer{0};
  u32 k = 0;
  while (k < n) {
    u32 its = std::min(blockSize, n - k);
    squareLoop(bufData, 0, its);
    k += its;
    queue.finish();
    if (k % logStep == 0) {
      float secsPerIt = timer.reset(k);
      log("%u / %u, %s us/it\n", k, n, formatSecsPerIter(secsPerIt).c_str());
    }
  }
  return readData();
}

// A:= A^h * B
void Gpu::expMul(Buffer<Word>& A, u64 h, Buffer<Word>& B) {
  exponentiate(A, h);
  modMul(A, B);
}

// return A^x * B
Words Gpu::expMul(const Words& A, u64 h, const Words& B, bool doSquareB) {
  writeIn(bufCheck, B);
  if (doSquareB) { square(bufCheck); }

  writeIn(bufData, A);
  expMul(bufData, h, bufCheck);
  return readData();
}

static bool testBit(u64 x, int bit) { return x & (u64(1) << bit); }

// See "left-to-right binary exponentiation" on wikipedia
void Gpu::exponentiate(Buffer<Word>& bufInOut, u64 exp) {
  if (exp == 0) {
    bufInOut.set(1);
  } else if (exp > 1) {
    fftP(buf1, bufInOut);
    fftMidIn(buf1);
    fftHin(buf2, buf1); // save fully FFTed "base" to buf2
    bool midInAlreadyDone = 1;

    int p = 63;
    while (!testBit(exp, p)) { --p; }

    for (--p; ; --p) {
      if (!midInAlreadyDone) fftMidIn(buf1);
      tailSquare(buf1);
      fftMidOut(buf1);
      midInAlreadyDone = 0;

      if (testBit(exp, p)) {
        doCarry(buf1, bufInOut);
        fftMidIn(buf1);
        tailMulLow(buf1, buf2);
        fftMidOut(buf1);
      }

      if (!p) { break; }

      doCarry(buf1, bufInOut);
    }

    fftW(buf3, buf1);
    carryA(bufInOut, buf3);
    carryB(bufInOut);
  }
}

// does either carryFused() or the expanded version depending on useLongCarry
void Gpu::doCarry(Buffer<double>& in, Buffer<Word>& wordBuf) {
  if (useLongCarry) {
    fftW(buf3, in);
    carryA(wordBuf, buf3);
    carryB(wordBuf);
    fftP(in, wordBuf);
  } else {
    carryFused(in);
  }
}

// Use buf1 (and buf23 if not in place) to do a single squaring.
void Gpu::square(Buffer<Word>& out, Buffer<Word>& in, enum LEAD_TYPE leadIn, enum LEAD_TYPE leadOut, bool doMul3, bool doLL) {
  // leadOut = LEAD_MIDDLE is not supported (slower than LEAD_WIDTH)
  assert(leadOut != LEAD_MIDDLE);
  // LL does not do Mul3
  assert(!(doMul3 && doLL));

  // In place FFTs use buf1.  Not in place FFTs also use buf3.
  // If leadIn is LEAD_NONE, in contains the input data, squaring starts at fftP
  // If leadIn is LEAD_WIDTH, buf1 (or buf3 if not in place) contains the input data, squaring starts at fftMidIn
  // If leadIn is LEAD_MIDDLE, buf1 contains the input data, squaring starts at tailSquare
  // If leadOut is LEAD_WIDTH, then buf1 (or buf3 if not in place) will contain the output of carryFused -- to be used as input to the next squaring.
  if (leadIn == LEAD_NONE) fftP(buf1, in);
  if (leadIn != LEAD_MIDDLE) fftMidIn(buf1);
  tailSquare(buf1);
  fftMidOut(buf1);

  // If leadOut is not allowed then we cannot use the faster carryFused kernel
  if (leadOut == LEAD_NONE) {
    fftW(buf3, buf1);
    if (!doLL && !doMul3) {
      carryA(out, buf3);
    } else if (doLL) {
      carryLL(out, buf3);
    } else {
      carryM(out, buf3);
    }
    carryB(out);
  }

  // Use CarryFused
  else {
    assert(!useLongCarry);
    assert(!doMul3);
    if (doLL) {
      carryFusedLL(buf1);
    } else {
      carryFused(buf1);
    }
  }
}

u32 Gpu::squareLoop(Buffer<Word>& out, Buffer<Word>& in, u64 from, u64 to, bool doTailMul3) {
  assert(from < to);
  enum LEAD_TYPE leadIn = LEAD_NONE;
  for (u64 k = from; k < to; ++k) {
    enum LEAD_TYPE leadOut = useLongCarry || (k == to - 1) ? LEAD_NONE : LEAD_WIDTH;
    square(out, (k==from) ? in : out, leadIn, leadOut, doTailMul3 && (k == to - 1));
    leadIn = leadOut;
  }
  return to;
}

bool Gpu::isEqual(Buffer<Word>& in1, Buffer<Word>& in2) {
  kernIsEqual(in1, in2);
  int isEq = 0;
  bufTrue.read(&isEq, 1);
  if (!isEq) { bufTrue.write({1}); }
  return isEq;
}

u64 Gpu::bufResidue(Buffer<Word> &buf) {
  readResidue(bufSmallOut, buf);
  vector<Word> words = readWords(bufSmallOut);

  int carry = 0;
  for (int i = 0; i < 32; ++i) {
    u32 len = bitlen(N, E, N - 32 + i);
    i64 w = (i64) words[i] + carry;
    carry = (int) (w >> len);
  }

  u64 res = 0;
  int hasBits = 0;
  for (int k = 0; k < 32 && hasBits < 64; ++k) {
    u32 len = bitlen(N, E, k);
    i64 tmp = (i64) words[32 + k] + carry;
    carry = (int) (tmp >> len);
    u64 w = tmp - ((i64) carry << len);
    assert(w < (1ULL << len));
    res += w << hasBits;
    hasBits += len;
  }
  return res;
}

static string formatETA(u32 secs) {
  u32 etaMins = (secs + 30) / 60;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  char buf[64];
  if (days) {
    snprintf(buf, sizeof(buf), "%dd %02d:%02d", days, hours, mins);
  } else {
    snprintf(buf, sizeof(buf), "%02d:%02d", hours, mins);
  }
  return string(buf);  
}

static string getETA(u32 step, u32 total, float secsPerStep) {
  u32 etaSecs = max(0u, u32((total - step) * secsPerStep));
  return formatETA(etaSecs);
}

string RoeInfo::toString() const {
  if (!N) { return {}; }

  char buf[256];
  snprintf(buf, sizeof(buf), "Z(%u)=%.1f Max %f mean %f sd %f (%f, %f)",
           N, z(.5f), max, mean, sd, gumbelMiu, gumbelBeta);
  return buf;
}

static string makeLogStr(const string& status, u64 k, u64 res, float secsPerIt, u64 nIters) {
  char buf[256];
  
  snprintf(buf, sizeof(buf), "%2s %9" PRIu64 " %016" PRIx64 " %s ETA %s; ",
           status.c_str(), k, res, /* k / float(nIters) * 100, */
           formatSecsPerIter(secsPerIt).c_str(), getETA(k, nIters, secsPerIt).c_str());
  return buf;
}

void Gpu::doBigLog(u64 k, u64 res, bool checkOK, float secsPerIt, u64 nIters, u32 nErrors) {
  auto [roeSq, roeMul] = readROE();
  double z = roeSq.z();
  zAvg.update(z, roeSq.N);
  if (roeSq.max > 0.005)
    log("%sZ=%.0f (avg %.1f), ROEmax=%.3f, ROEavg=%.3f. %s\n", makeLogStr(checkOK ? "OK" : "EE", k, res, secsPerIt, nIters).c_str(),
        z, zAvg.avg(), roeSq.max, roeSq.mean, (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());
  else
    log("%sZ=%.0f (avg %.1f) %s\n", makeLogStr(checkOK ? "OK" : "EE", k, res, secsPerIt, nIters).c_str(),
        z, zAvg.avg(), (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());

  if (roeSq.N > 2 && (z < 6 || (fft.shape.fft_type == FFT64 && z < 20))) {
    log("Danger ROE! Z=%.1f is too small, increase precision or FFT size!\n", z);
  }

  // Unless ROE log is not explicitly requested, measure only a few iterations to minimize overhead
  wantROE = args.logROE ? ROE_SIZE : 400;

  RoeInfo carryStats = readCarryStats();
  if (carryStats.N > 2) {
    u32 m = ldexp(carryStats.max, 32);
    double z = carryStats.z();
    log("Carry: %x Z(%u)=%.1f\n", m, carryStats.N, z);
  }
}

bool Gpu::equals9(const Words& a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

int ulps(double a, double b) {
  if (a == 0 && b == 0) { return 0; }

  u64 aa = as<u64>(a);
  u64 bb = as<u64>(b);
  bool sameSign = (aa >> 63) == (bb >> 63);
  int delta = sameSign ? bb - aa : bb + aa;
  return delta;
}

[[maybe_unused]] static double trigNorm(double c, double s) {
  double c2 = c * c;
  double err = fma(c, c, -c2);
  double norm = c2 + fma(s, s, err);
  return norm;
}

void Gpu::selftestTrig() {

#if FFT_FP64
  const u32 n = hN / 8;
  testTrig(buf1);
  vector<double> trig = buf1.read(n * 2);
  int sup = 0, sdown = 0;
  int cup = 0, cdown = 0;
  int oneUp = 0, oneDown = 0;
  for (u32 k = 0; k < n; ++k) {
    double c = trig[2*k];
    double s = trig[2*k + 1];

#if 0
    auto [refCos, refSin] = root1(hN, k);
#else
    long double angle = M_PIl * k / (hN/2);
    double refSin = sinl(angle);
    double refCos = cosl(angle);
#endif

    if (s > refSin) { ++sup; }
    if (s < refSin) { ++sdown; }
    if (c > refCos) { ++cup; }
    if (c < refCos) { ++cdown; }
    
    double norm = trigNorm(c, s);

    if (norm < 1.0) { ++oneDown; }
    if (norm > 1.0) { ++oneUp; }
  }

  log("TRIG sin(): imperfect %d / %d (%.2f%%), balance %d\n",
      sup + sdown, n, (sup + sdown) * 100.0 / n, sup - sdown);
  log("TRIG cos(): imperfect %d / %d (%.2f%%), balance %d\n",
      cup + cdown, n, (cup + cdown) * 100.0 / n, cup - cdown);
  log("TRIG norm: up %d, down %d\n", oneUp, oneDown);
#endif

  if (isAmdGpu(shared.context->deviceId())) {
    vector<string> WHATS {"V_NOP", "V_ADD_I32", "V_FMA_F32", "V_ADD_F64", "V_FMA_F64", "V_MUL_F64", "V_MAD_U64_U32"};
    for (int w = 0; w < int(WHATS.size()); ++w) {
      const int what = w;
      testTime(what, bufCarry);
      vector<i64> times = bufCarry.read(4096 * 2);
      [[maybe_unused]] i64 prev = 0;
      u64 min = -1;
      u64 sum = 0;
      for (int i = 0; i < int(times.size()); ++i) {
        i64 x = times[i];
#if 0
        if (x != prev) {
          log("%4d : %ld\n", i, x);
          prev = x;
        }
#endif
        if (x > 0 && u64(x) < min) { min = x; }
        if (x > 0) { sum += x; }
      }
      log("%-15s : %.2f cycles latency; time min: %d; avg %.0f\n",
          WHATS[w].c_str(), double(min - 40) / 48, int(min), double(sum) / times.size());
    }
  }
}

static u32 mod3(const std::vector<u32> &words) {
  u32 r = 0;
  // uses the fact that 2**32 % 3 == 1.
  for (u32 w : words) { r += w % 3; }
  return r % 3;
}

static void doDiv3(u64 E, Words& words) {
  u32 r = (3 - mod3(words)) % 3;
  assert(r < 3);
  int topBits = E % 32;
  assert(topBits > 0 && topBits < 32);
  {
    u64 w = (u64(r) << topBits) + words.back();
    words.back() = w / 3;
    r = w % 3;
  }
  for (auto it = words.rbegin() + 1, end = words.rend(); it != end; ++it) {
    u64 w = (u64(r) << 32) + *it;
    *it = w / 3;
    r = w % 3;
  }
}

void Gpu::doDiv9(u64 E, Words& words) {
  doDiv3(E, words);
  doDiv3(E, words);
}

fs::path Gpu::saveProof(const Args& args, ProofSet& proofSet) {
  bool problem_proof = false;
  for ( ; ; ) {
    for (int retry = 0; retry == 0 || (retry == 1 && !problem_proof); ++retry) {
      auto [proof, hashes] = proofSet.computeProof(this);
      fs::path tmpFile = proof.file(args.proofToVerifyDir);
      proof.save(tmpFile);
            
      fs::path proofFile = proof.file(args.proofResultDir);

      bool ok = Proof::load(tmpFile).verify(this, hashes);
      log("Proof '%s' verification %s\n", tmpFile.string().c_str(), ok ? "OK" : "FAILED");
      if (ok) {
        fancyRename(tmpFile, proofFile);
        log("Proof '%s' generated\n", proofFile.string().c_str());
        return proofFile;
      }
    }
    problem_proof = true;
    proofSet.reducePower();
    if (proofSet.power < 4) break;
  }
  throw "bad proof generation";
}

PRPState Gpu::loadPRP(Saver<PRPState>& saver) {
  for (int nTries = 0; nTries < 2; ++nTries) {
    if (nTries) {
      saver.dropMostRecent();    // Try an earlier savefile
    }

    PRPState state = saver.load();
    writeState(state.k, state.check, state.blockSize);
    u64 res = dataResidue();

    if (res == state.res64) {
      log("OK %9" PRIu64 " on-load: blockSize %d, %016" PRIx64 "\n", state.k, state.blockSize, res);
      return state;
      // return {loaded.k, loaded.blockSize, loaded.nErrors};
    }

    log("EE %9" PRIu64 " on-load: %016" PRIx64 " vs. %016" PRIx64 "\n", state.k, res, state.res64);

    if (!state.k) { break; }  // We failed on PRP start
  }

  throw "Error on load";
}

u32 Gpu::getProofPower(u64 k) {
  u32 power = ProofSet::effectivePower(E, args.getProofPow(E), k);

  if (power != args.getProofPow(E)) {
    log("Proof using power %u (vs %u)\n", power, args.getProofPow(E));
  }

  if (!power) {
    log("Proof generation disabled!\n");
  } else {
    log("Proof of power %u requires about %.1fGB of disk space\n", power, ProofSet::diskUsageGB(E, power));
  }
  return power;
}

tuple<bool, RoeInfo> Gpu::measureCarry() {
  u32 blockSize{}, iters{}, warmup{};

  blockSize = 200;
  iters = 2000;
  warmup = 50;

  assert(iters % blockSize == 0);

  u32 k = 0;
  PRPState state{E, 0, blockSize, 3, makeWords(E, 1), 0};
  writeState(state.k, state.check, state.blockSize);
  {
    u64 res = dataResidue();
    if (res != state.res64) {
      log("residue expected %016" PRIx64 " found %016" PRIx64 "\n", state.res64, res);
    }
    assert(res == state.res64);
  }

  enum LEAD_TYPE leadIn = LEAD_NONE;
  modMul(bufCheck, bufData, leadIn);
  leadIn = LEAD_MIDDLE;

  enum LEAD_TYPE leadOut = useLongCarry ? LEAD_NONE : LEAD_WIDTH;
  square(bufData, bufData, leadIn, leadOut);
  leadIn = leadOut;
  ++k;

  while (k < warmup) {
    square(bufData, bufData, leadIn, leadOut);
    leadIn = leadOut;
    ++k;
  }

  readCarryStats(); // ignore the warm-up iterations

  if (Signal::stopRequested()) { throw "stop requested"; }

  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, leadOut);
      leadIn = leadOut;
      ++k;
    }
    square(bufData, bufData, leadIn, LEAD_NONE);
    leadIn = LEAD_NONE;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData, leadIn);
    leadIn = LEAD_MIDDLE;
    if (Signal::stopRequested()) { throw "stop requested"; }
  }

  [[maybe_unused]] u64 res = dataResidue();
  if (Signal::stopRequested()) { throw "stop requested"; }

  bool ok = doCheck(blockSize);
  auto stats = readCarryStats();

  // log("%s %016" PRIx64 " %s\n", ok ? "OK" : "EE", res, roe.toString(statsBits).c_str());
  return {ok, stats};
}

tuple<bool, u64, RoeInfo, RoeInfo> Gpu::measureROE(bool quick) {
  u32 blockSize{}, iters{}, warmup{};

  if (true) {
    blockSize = 200;
    iters = 2000;
    warmup = 50;
  } else {
    blockSize = 500;
    iters = 10'000;
    warmup = 100;
  }

  assert(iters % blockSize == 0);

  wantROE = ROE_SIZE; // should be large enough to capture fully this measureROE()

  u32 k = 0;
  PRPState state{E, 0, blockSize, 3, makeWords(E, 1), 0};
  writeState(state.k, state.check, state.blockSize);
  {
    u64 res = dataResidue();
    if (res != state.res64) {
      log("residue expected %016" PRIx64 " found %016" PRIx64 "\n", state.res64, res);
    }
    assert(res == state.res64);
  }

  enum LEAD_TYPE leadIn = LEAD_NONE;
  modMul(bufCheck, bufData, leadIn);
  leadIn = LEAD_MIDDLE;

  enum LEAD_TYPE leadOut = useLongCarry ? LEAD_NONE : LEAD_WIDTH;
  square(bufData, bufData, leadIn, leadOut);
  leadIn = leadOut;
  ++k;

  while (k < warmup) {
    square(bufData, bufData, leadIn, leadOut);
    leadIn = leadOut;
    ++k;
  }

  readROE(); // ignore the warm-up iterations

  if (Signal::stopRequested()) { throw "stop requested"; }

  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, leadOut);
      leadIn = leadOut;
      ++k;
    }
    square(bufData, bufData, leadIn, LEAD_NONE);
    leadIn = LEAD_NONE;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData, leadIn);
    leadIn = LEAD_MIDDLE;
    if (Signal::stopRequested()) { throw "stop requested"; }
  }

  [[maybe_unused]] u64 res = dataResidue();
  if (Signal::stopRequested()) { throw "stop requested"; }

  bool ok = doCheck(blockSize);
  auto roes = readROE();

  wantROE = 0;
  // log("%s %016" PRIx64 " %s\n", ok ? "OK" : "EE", res, roe.toString(statsBits).c_str());
  return {ok, res, roes.first, roes.second};
}

double Gpu::timePRP(int quick) {        // Quick varies from 1 (slowest, longest) to 10 (quickest, shortest)
  u32 blockSize{}, iters{}, warmup{};

  if (quick == 10)     iters =   400, blockSize = 200;
  else if (quick == 9) iters =   600, blockSize = 300;
  else if (quick == 8) iters =   900, blockSize = 300;
  else if (quick == 7) iters =  1200, blockSize = 400;
  else if (quick == 6) iters =  1800, blockSize = 600;
  else if (quick == 5) iters =  3000, blockSize = 1000;
  else if (quick == 4) iters =  5000, blockSize = 1000;
  else if (quick == 3) iters =  8000, blockSize = 1000;
  else if (quick == 2) iters = 12000, blockSize = 1000;
  else if (quick == 1) iters = 20000, blockSize = 1000;
  warmup = 20;

  assert(iters % blockSize == 0);

  u32 k = 0;
  PRPState state{E, 0, blockSize, 3, makeWords(E, 1), 0};
  writeState(state.k, state.check, state.blockSize);
  assert(dataResidue() == state.res64);

  enum LEAD_TYPE leadIn = LEAD_NONE;
  modMul(bufCheck, bufData, leadIn);
  leadIn = LEAD_MIDDLE;

  enum LEAD_TYPE leadOut = useLongCarry ? LEAD_NONE : LEAD_WIDTH;
  square(bufData, bufData, leadIn, leadOut);
  leadIn = leadOut;
  ++k;

  while (k < warmup) {
    square(bufData, bufData, leadIn, leadOut);
    leadIn = leadOut;
    ++k;
  }
  queue.finish();
  if (Signal::stopRequested()) { throw "stop requested"; }

  Timer t;
  queue.setSquareTime(0);     // Busy wait on nVidia to get the most accurate timings while tuning
  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, leadOut);
      leadIn = leadOut;
      ++k;
    }
    square(bufData, bufData, leadIn, LEAD_NONE);
    leadIn = LEAD_NONE;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData, leadIn);
    leadIn = LEAD_MIDDLE;
    if (Signal::stopRequested()) { throw "stop requested"; }
  }
  queue.finish();
  double secsPerIt = t.reset() / (iters - warmup);

  if (Signal::stopRequested()) { throw "stop requested"; }

  u64 res = dataResidue();
  bool ok = doCheck(blockSize);
  if (!ok) {
    log("Error %016" PRIx64 "\n", res);
    secsPerIt = 0.1; // a large value to mark the error
  }
  return secsPerIt * 1e6;
}

PRPResult Gpu::isPrimePRP(const Task& task) {
  assert(E == task.exponent);

  // This timer is used to measure total elapsed time to be written to the savefile.
  Timer elapsedTimer;

  u32 nErrors = 0;
  int nSeqErrors = 0;
  u64 lastFailedRes64 = 0;
  u32 logStep = args.logStep;

 reload:
  elapsedTimer.reset();
  u32 blockSize{};
  u64 k{};
  double elapsedBefore = 0;

  {
    PRPState state = loadPRP(*getSaver());
    nErrors = std::max(nErrors, state.nErrors);
    blockSize = state.blockSize;
    k = state.k;
    elapsedBefore = state.elapsed;
  }

  assert(blockSize > 0 && logStep % blockSize == 0);

  u32 checkStep = checkStepForErrors(blockSize, nErrors);
  assert(checkStep % logStep == 0);

  u32 power = getProofPower(k);
  
  ProofSet proofSet{E, power};

  bool isPrime = false;

  u64 finalRes64 = 0;
  vector<u32> res2048;

  // We extract the res64 at kEnd.
  // For M=2^E-1, residue "type-3" == 3^(M+1), and residue "type-1" == type-3 / 9,
  // See http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  // For both type-1 and type-3 we need to do E squarings (as M+1==2^E).
  const u64 kEnd = E;
  assert(k < kEnd);

  // We continue beyound kEnd: to the next multiple of blockSize, to do a check there
  u64 kEndEnd = roundUp(kEnd, blockSize);

  bool skipNextCheckUpdate = false;

  u64 persistK = proofSet.next(k);
  enum LEAD_TYPE leadIn = LEAD_NONE;

  assert(k % blockSize == 0);
  assert(checkStep % blockSize == 0);

  const u64 startK = k;
  IterationTimer iterationTimer{k};

  wantROE = 0; // skip the initial iterations

  while (true) {
    assert(k < kEndEnd);

    if (!wantROE && k - startK > 30) { wantROE = args.logROE ? ROE_SIZE : 2'000; }

    if (skipNextCheckUpdate) {
      skipNextCheckUpdate = false;
    } else if (k % blockSize == 0) {
      modMul(bufCheck, bufData, leadIn);
      leadIn = LEAD_MIDDLE;
    }

    ++k; // !! early inc

    bool doStop = (k % blockSize == 0) && (Signal::stopRequested() || (args.iters && k - startK >= args.iters));
    bool doCheck = doStop || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);
    bool doLog = k % logStep == 0;
    enum LEAD_TYPE leadOut = doCheck || doLog || k == persistK || k == kEnd || useLongCarry ? LEAD_NONE : LEAD_WIDTH;

    if (doStop) { log("Stopping, please wait..\n"); }

    square(bufData, bufData, leadIn, leadOut, false);
    leadIn = leadOut;

    if (k == persistK) {
      vector<Word> rawData = readChecked(bufData);
      if (rawData.empty()) {
        log("Data error ZERO\n");
        ++nErrors;
        goto reload;
      }
      (*background)([=, E=this->E] { ProofSet::save(E, power, k, compactBits(rawData, E)); });
      persistK = proofSet.next(k);
    }

    if (k == kEnd) {
      Words words = readData();
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);
      res2048.clear();
      assert(words.size() >= 64);
      res2048.insert(res2048.end(), words.begin(), std::next(words.begin(), 64));
      log("%s %8" PRIu64 " / %" PRIu64 ", %s\n", isPrime ? "PP" : "CC", kEnd, E, hex(finalRes64).c_str());
    }

    if (!doCheck && !doLog) continue;

    u64 res = dataResidue();
    float secsPerIt = iterationTimer.reset(k);
    queue.setSquareTime((int) (secsPerIt * 1'000'000));

    vector<Word> rawCheck = readChecked(bufCheck);
    if (rawCheck.empty()) {
      ++nErrors;
      log("%9" PRIu64 " %016" PRIx64 " read NULL check\n", k, res);
      if (++nSeqErrors > 2) { throw "sequential errors"; }
      goto reload;
    }

    if (!doCheck) {
      (*background)([=, this] {
        getSaver()->saveUnverified({E, k, blockSize, res, compactBits(rawCheck, E), nErrors,
                                    elapsedBefore + elapsedTimer.at()});
      });

      log("   %9" PRIu64 " %016" PRIx64 " %s\n", k, res, formatSecsPerIter(secsPerIt).c_str());
      RoeInfo carryStats = readCarryStats();
      if (carryStats.N) {
        u32 m = ldexp(carryStats.max, 32);
        double z = carryStats.z();
        log("Carry: %x Z(%u)=%.1f\n", m, carryStats.N, z);
      }
    } else {
      bool ok = this->doCheck(blockSize);
      [[maybe_unused]] float secsCheck = iterationTimer.reset(k);

      if (ok) {
        nSeqErrors = 0;
        // lastFailedRes64 = 0;
        skipNextCheckUpdate = true;

        if (k < kEnd) {
          (*background)([=, this, rawCheck = std::move(rawCheck)] {
            getSaver()->save({E, k, blockSize, res, compactBits(rawCheck, E), nErrors, elapsedBefore + elapsedTimer.at()});
          });
        }

        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);

        if (k >= kEndEnd) {
          fs::path proofFile = saveProof(args, proofSet);
          return {isPrime, finalRes64, nErrors, proofFile.string(), toHex(res2048)};
        }        
      } else {
        ++nErrors;
        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);
        if (++nSeqErrors > 2) {
          log("%d sequential errors, will stop.\n", nSeqErrors);
          throw "too many errors";
        }
        if (res == lastFailedRes64) {
          log("Consistent error %016" PRIx64 ", will stop.\n", res);
          throw "consistent error";
        }
        lastFailedRes64 = res;
        if (!doStop) { goto reload; }
      }

      logTimeKernels();

      if (doStop) {
        queue.finish();
        throw "stop requested";
      }

      iterationTimer.reset(k);
    }
  }
}

LLResult Gpu::isPrimeLL(const Task& task) {
  assert(E == task.exponent);
  wantROE = 0;

  Timer elapsedTimer;

  Saver<LLState> saver{E, 1000, args.nSavefiles};

  reload:
  elapsedTimer.reset();

  u64 startK = 0;
  double elapsedBefore = 0;
  {
    LLState state = saver.load();

    elapsedBefore = state.elapsed;
    startK = state.k;
    u64 expectedRes = (u64(state.data[1]) << 32) | state.data[0];
    writeIn(bufData, std::move(state.data));
    u64 res = dataResidue();
    if (res != expectedRes) { throw "Invalid savefile (res64)"; }
    assert(res == expectedRes);
    log("LL loaded @ %" PRIu64 " : %016" PRIx64 "\n", startK, res);
  }

  IterationTimer iterationTimer{startK};

  u64 k = startK;
  u64 kEnd = E - 2;
  enum LEAD_TYPE leadIn = LEAD_NONE;

  while (true) {
    ++k;
    bool doStop = (k >= kEnd) || (args.iters && k - startK >= args.iters);

    if (Signal::stopRequested()) {
      doStop = true;
      log("Stopping, please wait..\n");
    }

    bool doLog = (k % args.logStep == 0) || doStop;
    enum LEAD_TYPE leadOut = doLog || useLongCarry ? LEAD_NONE : LEAD_WIDTH;

    squareLL(bufData, leadIn, leadOut);
    leadIn = leadOut;

    if (!doLog) continue;

    u64 res64 = 0;
    auto data = readData();
    bool isAllZero = data.empty();

    if (isAllZero) {
      if (k < kEnd) {
        log("Error: early ZERO @ %" PRIu64 "\n", k);
        if (doStop) {
          throw "stop requested";
        } else {
          goto reload;
        }
      }
      res64 = 0;
    } else {
      assert(data.size() >= 2);
      res64 = (u64(data[1]) << 32) | data[0];
      saver.save({E, k, std::move(data), elapsedBefore + elapsedTimer.at()});
    }

    float secsPerIt = iterationTimer.reset(k);
    queue.setSquareTime((int) (secsPerIt * 1'000'000));
    log("%9" PRIu64 " %016" PRIx64 " %s ETA %s\n", k, res64, formatSecsPerIter(secsPerIt).c_str(), getETA(k, kEnd, secsPerIt).c_str());

    if (k >= kEnd) { return {isAllZero, res64}; }

    if (doStop) { throw "stop requested"; }
  }
}

array<u64, 4> Gpu::isCERT(const Task& task) {
  assert(E == task.exponent);
  wantROE = 0;

  // Get CERT start value
  char fname[32];
  sprintf(fname, "M%" PRIu64 ".cert", E);

// AutoPrimenet.py does not add the cert entry to worktodo.txt until it has successfully downloaded the .cert file.

  { // Enclosing this code in braces ensures the file will be closed by the File destructor.  The later file deletion requires the file be closed in Windows.
    File fi = File::openReadThrow(fname);
    u32 nBytes = u32((E - 1) / 8 + 1);
    Words B = fi.readBytesLE(nBytes);
    writeIn(bufData, std::move(B));
  }

  Timer elapsedTimer;

  elapsedTimer.reset();

  u32 startK = 0;

  IterationTimer iterationTimer{startK};

  u32 k = 0;
  u32 kEnd = task.squarings;
  enum LEAD_TYPE leadIn = LEAD_NONE;

  while (true) {
    ++k;
    bool doStop = (k >= kEnd);

    if (Signal::stopRequested()) {
      doStop = true;
      log("Stopping, please wait..\n");
    }

    bool doLog = (k % 100'000 == 0) || doStop;
    enum LEAD_TYPE leadOut = doLog || useLongCarry ? LEAD_NONE : LEAD_WIDTH;

    squareCERT(bufData, leadIn, leadOut);
    leadIn = leadOut;

    if (!doLog) continue;

    Words data = readData();
    assert(data.size() >= 2);
    u64 res64 = (u64(data[1]) << 32) | data[0];

    float secsPerIt = iterationTimer.reset(k);
    queue.setSquareTime((int) (secsPerIt * 1'000'000));
    log("%7u / %7u %016" PRIx64 " %s ETA %s\n", k, kEnd, res64, formatSecsPerIter(secsPerIt).c_str(), getETA(k, kEnd, secsPerIt).c_str());

    if (k >= kEnd) {
      fs::remove (fname);
      return std::move(SHA3{}.update(data.data(), (E-1)/8+1)).finish();
    }

    if (doStop) { throw "stop requested"; }
  }
}


void Gpu::clear(bool isPRP) {
  if (isPRP) {
    Saver<PRPState>::clear(E);
  } else {
    Saver<LLState>::clear(E);
  }
}

Saver<PRPState> *Gpu::getSaver() {
  if (!saver) { saver = make_unique<Saver<PRPState>>(E, args.blockSize, args.nSavefiles); }
  return saver.get();
}
