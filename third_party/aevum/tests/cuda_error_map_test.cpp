// Host test, no CUDA and no device: the CUDA backend's mapping of a CUDA driver result (a plain int here) to the
// status Aevum reports.  Only a result that means a lost, reset or faulted device or context may read as "the GPU
// was reset or lost"; the backend used to turn every failure into CL_OUT_OF_RESOURCES, which does.
#include "cuda/CudaErrorMap.h"
#include "GpuLost.h"

#include <cstdio>

static int failures = 0;

#define EXPECT(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } else { std::printf("ok:   %s\n", msg); } } while (0)

using gpulost::Kind;
using gpulost::Phase;

static Kind kindOf(int cu, Phase phase) { return gpulost::classify(cuda_error_map::toClStatus(cu), phase); }

int main() {
  EXPECT(cuda_error_map::toClStatus(0) == 0, "CUDA_SUCCESS is CL_SUCCESS");

  // The results that mean the device or context is gone or faulted: a lost GPU at run time and at setup.
  const int lost[] = {3, 4, 46, 201, 214, 700, 702, 709, 714, 715, 716, 717, 718, 719};
  for (int cu : lost) {
    char what[96];
    std::snprintf(what, sizeof(what), "CUDA result %d reads as a reset or lost GPU at run time", cu);
    EXPECT(kindOf(cu, Phase::Run) == Kind::DeviceLost, what);
    EXPECT(kindOf(cu, Phase::Create) == Kind::DeviceLost, "... and from a creation call");
  }

  // Out of memory is out of memory in either phase.
  EXPECT(cuda_error_map::toClStatus(2) == -4, "CUDA_ERROR_OUT_OF_MEMORY maps to the allocation status");
  EXPECT(kindOf(2, Phase::Create) == Kind::OutOfMemory && kindOf(2, Phase::Run) == Kind::OutOfMemory,
         "CUDA_ERROR_OUT_OF_MEMORY reads as out of memory");

  // Everything else keeps the generic message, and in particular is never -5.
  const int other[] = {1 /*invalid value*/, 100 /*no device*/, 101, 200, 400, 500, 600, 701 /*launch out of resources*/, 710, 800, 999};
  for (int cu : other) {
    char what[96];
    std::snprintf(what, sizeof(what), "CUDA result %d keeps the generic message", cu);
    EXPECT(kindOf(cu, Phase::Run) == Kind::Other && kindOf(cu, Phase::Create) == Kind::Other, what);
    EXPECT(cuda_error_map::toClStatus(cu) != gpulost::kOutOfResources, "... and is not mapped to CL_OUT_OF_RESOURCES");
  }

  return failures ? 1 : 0;
}
