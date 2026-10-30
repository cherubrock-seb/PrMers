#pragma once

// The kernel "binary" of the CUDA shim: what the kernel cache stores (CL_PROGRAM_BINARIES) and hands back
// (clCreateProgramWithBinary).  It is the PTX text, then, when NVRTC produced a CUBIN for this device, a marker line
// and the CUBIN.  The PTX stays, and stays first, because clCreateKernel reads each kernel's declared work-group size
// (.maxntid) from the PTX text and the PTX register-limit path re-JITs it; a CUBIN carries neither as text.  A blob
// without the marker is a plain PTX, the format of every cache file written before the CUBIN was kept, and loads as
// before.  Ported from gwoltman/gpuowl 57947a13 and c151cf27.

#include <cstring>
#include <string>

namespace cuda_blob {

inline constexpr char kCubinMarker[] = "\n// PRPLL-CUBIN\n";
inline constexpr size_t kCubinMarkerLen = sizeof(kCubinMarker) - 1;

inline std::string join(const std::string& ptx, const std::string& cubin) {
  if (cubin.empty()) return ptx;
  std::string blob;
  blob.reserve(ptx.size() + kCubinMarkerLen + cubin.size());
  blob += ptx;
  blob += kCubinMarker;
  blob += cubin;
  return blob;
}

enum class Split { Ok, BareElf };

// Splits a cached blob into its PTX and CUBIN.  A bare ELF is a CUBIN without its PTX, which leaves nothing to read
// the kernels' work-group sizes from: it is refused (BareElf), so the caller recompiles and overwrites it.
inline Split split(const std::string& blob, std::string& ptx, std::string& cubin) {
  ptx.clear();
  cubin.clear();
  if (blob.compare(0, 4, "\177ELF", 4) == 0) return Split::BareElf;
  const size_t mark = blob.find(kCubinMarker);
  if (mark == std::string::npos) {
    ptx = blob;
  } else {
    ptx = blob.substr(0, mark);
    cubin = blob.substr(mark + kCubinMarkerLen);
  }
  return Split::Ok;
}

// -use PTX_VERSION=x.y (undocumented, use at your own risk): the PTX ISA version to relabel the PTX as, so that a
// driver older than this NVRTC will JIT-compile it (e.g. a CUDA 13 NVRTC on a CUDA 12.8 driver, PTX ISA 8.7).  Only
// safe if the PTX uses nothing newer than x.y; the driver's JIT rejects code that does.  "" when not requested.
inline std::string ptxVersionOverride(const std::string& buildOptions) {
  static constexpr char key[] = "-DPTX_VERSION=";
  const size_t pos = buildOptions.find(key);
  if (pos == std::string::npos) return {};
  const size_t start = pos + std::strlen(key);
  return buildOptions.substr(start, buildOptions.find_first_of(" \t", start) - start);
}

// Rewrites the PTX's ".version" directive to `version`.  Returns the version it replaced, or "" if the PTX has none.
inline std::string setPtxVersion(std::string& ptx, const std::string& version) {
  size_t pos = ptx.compare(0, 9, ".version ") == 0 ? 0 : ptx.find("\n.version ");
  if (pos == std::string::npos) return {};
  if (pos) ++pos;
  const size_t start = pos + std::strlen(".version ");
  size_t end = ptx.find_first_of(" \t\r\n", start);
  if (end == std::string::npos) end = ptx.size();
  std::string old = ptx.substr(start, end - start);
  ptx.replace(start, end - start, version);
  return old;
}

}  // namespace cuda_blob
