// The CUDA backend's choice of NVRTC --gpu-architecture (plain ints, no CUDA needed).
#include "cuda/NvrtcArch.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static void expect(int cc, const std::vector<int>& archs, const std::string& want, const char* what) {
  const std::string got = nvrtc_arch::choose(cc, archs);
  if (got != want) {
    std::printf("FAIL: %s: cc %d -> '%s', expected '%s'\n", what, cc, got.c_str(), want.c_str());
    ++failures;
  }
}

int main() {
  // nvrtcGetSupportedArchs() of NVRTC 12.9, and the same window cut the way the neighbouring toolkits cut it: without
  // Maxwell, Pascal and Volta (CUDA 13 dropped sm_50..sm_72), and without anything after Hopper (CUDA 11.8 stops at sm_90).
  const std::vector<int> cuda129 = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 120, 121};
  std::vector<int> cuda13, cuda118;
  for (int a : cuda129) {
    if (a >= 75) cuda13.push_back(a);
    if (a <= 90) cuda118.push_back(a);
  }

  expect(61, cuda129, "sm_61", "Pascal on CUDA 12.9");
  expect(120, cuda129, "sm_120", "Blackwell on CUDA 12.9");

  expect(86, cuda13, "sm_86", "a GPU NVRTC knows keeps its own sm_XY");
  expect(120, cuda13, "sm_120", "Blackwell on CUDA 13");
  expect(61, cuda118, "sm_61", "Pascal on CUDA 11.8");
  expect(61, cuda13, "", "Pascal on CUDA 13: nothing at or below sm_61");
  expect(70, cuda13, "", "Volta on CUDA 13");
  expect(120, cuda118, "compute_90", "Blackwell on CUDA 11.8: newest PTX below the GPU");
  expect(100, cuda118, "compute_90", "sm_100 on CUDA 11.8");
  expect(76, cuda13, "compute_75", "an unlisted capability between two listed ones");
  expect(75, cuda13, "sm_75", "the oldest arch NVRTC supports");
  expect(61, {}, "sm_61", "unknown list (NVRTC < 11.2) keeps the old behaviour");
  expect(120, {}, "sm_120", "unknown list, new GPU");

  if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
  std::printf("cuda nvrtc arch test ok\n");
  return 0;
}
