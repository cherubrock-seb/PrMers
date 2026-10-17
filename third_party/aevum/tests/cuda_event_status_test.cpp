// Host test, no CUDA and no device: what the CUDA backend reports for a cuEventQuery / cuStreamSynchronize result
// (a plain int here).  A sticky CUDA error (a faulted kernel) used to read as CL_COMPLETE from clGetEventInfo and as
// CL_SUCCESS from clFinish, so the fault surfaced only on the next copy, as CL_OUT_OF_RESOURCES.
#include "cuda/CudaEventStatus.h"

#include <cstdio>

using namespace cuda_error_map;

static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } } while (0)

int main() {
  EXPECT(eventStatusFromCu(0) == kClComplete && kClComplete == 0, "CUDA_SUCCESS is CL_COMPLETE");
  EXPECT(eventStatusFromCu(600) == kClRunning && kClRunning == 1, "CUDA_ERROR_NOT_READY is CL_RUNNING");

  // The sticky errors of a faulted kernel are a negative status: Queue::waitForMarkerEvent throws on it.
  const int sticky[] = {700 /*illegal address*/, 719 /*launch failed*/, 715, 716, 717, 718, 714, 702 /*timeout*/, 214 /*ECC*/};
  for (int cu : sticky) {
    char what[96];
    std::snprintf(what, sizeof what, "CUDA result %d is a negative event status, the device-lost one", cu);
    EXPECT(eventStatusFromCu(cu) < 0 && eventStatusFromCu(cu) == kClDeviceNotAvailable, what);
    EXPECT(eventStatusFromCu(cu) == toClStatus(cu), "... the same status the enqueue calls and clFinish report");
  }

  // Any other failure of the query is an error status too, never complete or running.
  EXPECT(eventStatusFromCu(2) == kClMemObjectAllocationFailure, "an out-of-memory result keeps its allocation status");
  for (int cu = 1; cu <= 1000; ++cu) {
    if (cu == 600) continue;
    int st = eventStatusFromCu(cu);
    if (st >= 0) { std::printf("FAIL: CUDA result %d gives non-error event status %d\n", cu, st); ++failures; }
  }
  EXPECT(eventStatusFromCu(-1) < 0, "a negative (invalid) result is an error status");

  // clFinish: the status is the mapped error, success only for CUDA_SUCCESS.
  EXPECT(toClStatus(0) == kClSuccess, "clFinish after a clean synchronize is CL_SUCCESS");
  EXPECT(toClStatus(700) < 0 && toClStatus(719) < 0, "clFinish after a faulted kernel is an error");

  std::printf("%s\n", failures ? "FAIL" : "ok");
  return failures ? 1 : 0;
}
