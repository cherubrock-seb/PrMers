// A stand-in for the NVRTC header, for host tests that compile src/cuda/clwrap_cuda.cpp without CUDA installed (see
// tests/cuda_stub/cuda.h).
#pragma once

typedef enum { NVRTC_SUCCESS = 0 } nvrtcResult;

nvrtcResult nvrtcVersion(int* major, int* minor);
nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs);
nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs);
