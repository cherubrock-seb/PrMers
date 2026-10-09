// Maps a CUDA driver API result (CUresult, as a plain int) to the OpenCL-style status the rest of Aevum works with.
//
// The CUDA backend used to turn every failure into CL_OUT_OF_RESOURCES (-5), which GpuLost.h reads as "the GPU was
// reset or lost" when it comes from a run-time call.  Only the results that really mean a lost, reset or faulted
// device or context may map to the lost-GPU status; everything else keeps a status that gets the generic message.
//
// Plain ints and no CUDA includes, so the table is tested on a host without CUDA (tests/cuda_error_map_test.cpp).
// The numbers are those of cuda.h's CUresult.
#pragma once

namespace cuda_error_map {

constexpr int kClSuccess = 0;
constexpr int kClDeviceNotAvailable = -2;           // reads as a lost GPU in GpuLost.h
constexpr int kClMemObjectAllocationFailure = -4;   // reads as out of memory
constexpr int kClInvalidOperation = -59;            // any other failure: the generic message

enum CuResult : int {
  kSuccess = 0,
  kOutOfMemory = 2,                // CUDA_ERROR_OUT_OF_MEMORY
  kNotInitialized = 3,             // CUDA_ERROR_NOT_INITIALIZED
  kDeinitialized = 4,              // CUDA_ERROR_DEINITIALIZED
  kDeviceUnavailable = 46,         // CUDA_ERROR_DEVICE_UNAVAILABLE (cudaErrorDevicesUnavailable)
  kInvalidContext = 201,           // CUDA_ERROR_INVALID_CONTEXT (cudaErrorDeviceUninitialized)
  kEccUncorrectable = 214,         // CUDA_ERROR_ECC_UNCORRECTABLE
  kIllegalAddress = 700,           // CUDA_ERROR_ILLEGAL_ADDRESS (sticky)
  kLaunchTimeout = 702,            // CUDA_ERROR_LAUNCH_TIMEOUT (watchdog)
  kContextIsDestroyed = 709,       // CUDA_ERROR_CONTEXT_IS_DESTROYED
  kHardwareStackError = 714,       // CUDA_ERROR_HARDWARE_STACK_ERROR (sticky)
  kIllegalInstruction = 715,       // CUDA_ERROR_ILLEGAL_INSTRUCTION (sticky)
  kMisalignedAddress = 716,        // CUDA_ERROR_MISALIGNED_ADDRESS (sticky)
  kInvalidAddressSpace = 717,      // CUDA_ERROR_INVALID_ADDRESS_SPACE (sticky)
  kInvalidPc = 718,                // CUDA_ERROR_INVALID_PC (sticky)
  kLaunchFailed = 719,             // CUDA_ERROR_LAUNCH_FAILED (sticky)
};

// A result that means the device or its context is gone or faulted for good, so that every later call fails.
// Deliberately not included: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (701, a launch configuration that does not fit),
// CUDA_ERROR_INVALID_VALUE and the like (a bad argument), CUDA_ERROR_UNKNOWN (999, could be anything).
constexpr bool isLostOrFaulted(int cu) {
  switch (cu) {
    case kNotInitialized: case kDeinitialized: case kDeviceUnavailable: case kInvalidContext:
    case kEccUncorrectable: case kIllegalAddress: case kLaunchTimeout: case kContextIsDestroyed:
    case kHardwareStackError: case kIllegalInstruction: case kMisalignedAddress: case kInvalidAddressSpace:
    case kInvalidPc: case kLaunchFailed:
      return true;
    default:
      return false;
  }
}

constexpr int toClStatus(int cu) {
  if (cu == kSuccess) return kClSuccess;
  if (cu == kOutOfMemory) return kClMemObjectAllocationFailure;
  if (isLostOrFaulted(cu)) return kClDeviceNotAvailable;
  return kClInvalidOperation;
}

}  // namespace cuda_error_map
