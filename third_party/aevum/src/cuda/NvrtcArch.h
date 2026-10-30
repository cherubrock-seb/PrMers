// Chooses NVRTC's --gpu-architecture for a GPU.
//
// Each NVRTC accepts only a window of architectures: CUDA 13 dropped sm_50..sm_72 (Maxwell, Pascal, Volta), and an
// older toolkit does not know the newest GPUs (CUDA 11.8 stops at sm_90, so no sm_100/sm_120).  Asking for an
// architecture outside the window fails every kernel with "invalid value for --gpu-architecture".
//
// The GPU's own sm_XY is used whenever NVRTC has it (nothing changes there).  Otherwise compile for the newest
// compute_XY not above the GPU: PTX for a virtual architecture, which the driver JIT-compiles for any GPU of at least
// that capability (the shim loads PTX with cuModuleLoadData either way).  If NVRTC supports nothing that old, there is
// no usable architecture and the caller must explain.
//
// Plain integers and no CUDA includes, so the choice is tested on a host without CUDA (tests/cuda_nvrtc_arch_test.cpp).
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace nvrtc_arch {

// cc: the GPU's compute capability as major * 10 + minor (61 for sm_61).  archs: what nvrtcGetSupportedArchs()
// returned, in the same encoding; empty when unknown (NVRTC older than 11.2), which keeps the GPU's own sm_XY.
// Returns "sm_XY", "compute_XY", or "" when NVRTC supports nothing at or below the GPU.
inline std::string choose(int cc, const std::vector<int>& archs) {
  if (archs.empty() || std::find(archs.begin(), archs.end(), cc) != archs.end()) { return "sm_" + std::to_string(cc); }
  int best = 0;
  for (int a : archs) { if (a <= cc) { best = std::max(best, a); } }
  return best ? "compute_" + std::to_string(best) : "";
}

}  // namespace nvrtc_arch
