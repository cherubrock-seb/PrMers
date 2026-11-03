#!/usr/bin/env bash
# Host-only regression: compile and execute the production GF61 tail policy
# helper with small device/configuration stubs. No OpenCL device is needed.
set -euo pipefail

SOURCE_ROOT="$(cd "$1" && pwd)"
TEMP_ROOT="$(mktemp -d "/tmp/aevum-gf61-policy.XXXXXXXX")"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

GPU="$SOURCE_ROOT/src/Gpu.cpp"
TAIL="$SOURCE_ROOT/src/cl/tailutil.cl"
SQUARE="$SOURCE_ROOT/src/cl/tailsquare.cl"

test -f "$GPU"
test -f "$TAIL"
test -f "$SQUARE"

grep -Fq 'tailGF61KernelDefines(args, shared.context->deviceId(), fft,' "$GPU"
grep -Fq '#if TAIL_GF61_REVERSE16' "$SQUARE"
grep -Fq 'name == "gfx906" || name.rfind("gfx906:", 0) == 0' "$GPU"
grep -Fq 'local GF61 lds[SMALL_HEIGHT];' "$SQUARE"
grep -Fq 'local GF61 lds[LDS_BYTES / sizeof(GF61)];' "$SQUARE"
grep -Fq '#define TAIL_GF61_REVERSE16 0' "$TAIL"
grep -Fq 'if (SHUFL_BYTES_H == 16 || (NTT_GF61 && TAIL_GF61_REVERSE16))' "$TAIL"

cat > "$TEMP_ROOT/policy.cpp" <<'CPP'
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using KeyVal = pair<string, string>;
using cl_device_id = unsigned int;

struct FakeShape {
  int height;
  string spec() const { return "pfa7:1:512:7:512:202"; }
};

struct FFTConfig {
  bool pfa;
  bool NTT_GF61;
  FakeShape shape;
  bool isPfa() const { return pfa; }
};

struct Args {
  map<string, string> flags;
  map<string, vector<KeyVal>> perFftConfig;
};

bool isAmdGpu(cl_device_id id) {
  return id != 0x10deu;
}

string getDeviceName(cl_device_id id) {
  switch (id) {
    case 0x1002u: return "gfx906:sramecc+:xnack-";
    case 0x1003u: return "gfx906";
    case 0x1004u: return "gfx1103";
    case 0x1005u: return "gfx9060";
    case 0x1006u: return "Radeon VII";
    case 0x10deu: return "NVIDIA GeForce RTX 3080";
    default: throw runtime_error("unexpected fake device");
  }
}
CPP

python3 -c 'from pathlib import Path; import sys; s=Path(sys.argv[1]).read_text(); needle="static string tailGF61KernelDefines("; assert s.count(needle)==1,"HELPER_NOT_UNIQUE"; a=s.index(needle); b=s.index("\n}\n\nGpu::Gpu(",a)+2; print(s[a:b])' \
  "$GPU" >> "$TEMP_ROOT/policy.cpp"

cat >> "$TEMP_ROOT/policy.cpp" <<'CPP'

static int checked = 0;

void verify(string label,
            bool should_enable,
            cl_device_id device,
            bool pfa,
            bool gf61,
            int height,
            vector<KeyVal> extra = {},
            map<string, string> flags = {},
            map<string, vector<KeyVal>> per = {}) {
  Args args{std::move(flags), std::move(per)};
  FFTConfig fft{pfa, gf61, FakeShape{height}};

  const string input = " -DUNRELATED_SENTINEL=1";
  const string expected =
      input + (should_enable ? " -DTAIL_GF61_REVERSE16=1" : "");

  const string actual =
      tailGF61KernelDefines(args, device, fft, extra, input);

  if (actual != expected) {
    throw runtime_error(
        "PR260 POLICY FAILURE: " + label +
        ", actual='" + actual +
        "', expected='" + expected + "'");
  }
  ++checked;
}

int main() {
  constexpr cl_device_id AMD = 0x1002u;
  constexpr cl_device_id NVIDIA = 0x10deu;
  constexpr cl_device_id AMD_GFX906_BARE = 0x1003u;
  constexpr cl_device_id AMD_GFX1103 = 0x1004u;
  constexpr cl_device_id AMD_GFX9060 = 0x1005u;
  constexpr cl_device_id AMD_RADEON_ALIAS = 0x1006u;

  verify("AMD PFA7 GF61 H512 default", true, AMD, true, true, 512);
  verify("AMD PFA9 GF61 H512 default", true, AMD, true, true, 512);
  verify("AMD gfx906 bare name default", true,
         AMD_GFX906_BARE, true, true, 512);
  verify("AMD gfx1103 canonical path", false,
         AMD_GFX1103, true, true, 512);
  verify("AMD gfx9060 prefix collision rejected", false,
         AMD_GFX9060, true, true, 512);
  verify("AMD Radeon alias unverified canonical path", false,
         AMD_RADEON_ALIAS, true, true, 512);

  verify("extraConf explicit zero", false, AMD, true, true, 512,
         {{"SHUFL_BYTES_H", "0"}});
  verify("extraConf explicit four", false, AMD, true, true, 512,
         {{"SHUFL_BYTES_H", "4"}});
  verify("extraConf explicit eight", false, AMD, true, true, 512,
         {{"SHUFL_BYTES_H", "8"}});
  verify("extraConf explicit sixteen", false, AMD, true, true, 512,
         {{"SHUFL_BYTES_H", "16"}});
  verify("global explicit zero", false, AMD, true, true, 512,
         {}, {{"SHUFL_BYTES_H", "0"}});
  verify("global explicit eight", false, AMD, true, true, 512,
         {}, {{"SHUFL_BYTES_H", "8"}});
  verify("per-FFT explicit zero", false, AMD, true, true, 512,
         {}, {}, {{"pfa7:1:512:7:512:202", {{"SHUFL_BYTES_H", "0"}}}});
  verify("per-FFT explicit eight", false, AMD, true, true, 512,
         {}, {}, {{"pfa7:1:512:7:512:202", {{"SHUFL_BYTES_H", "8"}}}});
  verify("combined explicit overrides", false, AMD, true, true, 512,
         {{"SHUFL_BYTES_H", "16"}}, {{"SHUFL_BYTES_H", "8"}},
         {{"pfa7:1:512:7:512:202", {{"SHUFL_BYTES_H", "0"}}}});

  verify("unrelated global option", true, AMD, true, true, 512,
         {}, {{"OTHER", "1"}});
  verify("unrelated per-FFT shape", true, AMD, true, true, 512,
         {}, {}, {{"pfa9:1:512:9:1K:202", {{"SHUFL_BYTES_H", "8"}}}});

  verify("NVIDIA non-target", false, NVIDIA, true, true, 512);
  verify("non-PFA non-target", false, AMD, false, true, 512);
  verify("non-GF61 non-target", false, AMD, true, false, 512);
  verify("H256 non-target", false, AMD, true, true, 256);
  verify("H1024 non-target", false, AMD, true, true, 1024);

  if (checked != 22) {
    throw runtime_error("Unexpected policy case count");
  }

  cout << "PR260_GF61_REVERSE16_POLICY_TEST=PASS" << endl;
  cout << "EXECUTED_CASES=" << checked << endl;
  cout << "TESTED_IMPLEMENTATION=EXTRACTED_PRODUCTION_CXX_HELPER" << endl;
  cout << "EXPLICIT_SHUFL_BYTES_H_PRESERVED=PASS" << endl;
  cout << "NVIDIA_NONPFA_NONGF61_NONH512=PASS" << endl;
  cout << "KERNEL_REVERSE16_SOURCE_CONTRACT=PASS" << endl;
}
CPP

"$CXX" \
  -std=c++20 -O2 -Wall -Wextra -Werror \
  "$TEMP_ROOT/policy.cpp" -o "$TEMP_ROOT/policy-test"

"$TEMP_ROOT/policy-test"
