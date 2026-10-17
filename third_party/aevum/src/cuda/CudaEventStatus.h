// What the CUDA backend's clGetEventInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) reports for the result of cuEventQuery.
//
// OpenCL's execution status is CL_COMPLETE (0), CL_RUNNING (1), ..., or a negative error code for a command that
// terminated abnormally.  cuEventQuery returns CUDA_SUCCESS for a finished event and CUDA_ERROR_NOT_READY (600) for
// one still pending; anything else is the sticky error of a faulted context (CUDA_ERROR_ILLEGAL_ADDRESS,
// CUDA_ERROR_LAUNCH_FAILED, ...) and must not read as "complete", or Queue::waitForMarkerEvent never learns that
// the kernels before the marker died.  Same table as the enqueue calls use (CudaErrorMap.h).
//
// Plain ints and no CUDA includes, so it is tested on a host without CUDA (tests/cuda_event_status_test.cpp).
#pragma once

#include "CudaErrorMap.h"

namespace cuda_error_map {

constexpr int kClComplete = 0;
constexpr int kClRunning = 1;
constexpr int kCuErrorNotReady = 600;  // CUDA_ERROR_NOT_READY

// The execution status for a cuEventQuery result: CL_COMPLETE, CL_RUNNING, or the negative status of the error.
constexpr int eventStatusFromCu(int cu) {
  if (cu == kSuccess) return kClComplete;
  if (cu == kCuErrorNotReady) return kClRunning;
  return toClStatus(cu);
}

}  // namespace cuda_error_map
