// The CUDA shim's kernel-cache blob (PTX, then the CUBIN behind a marker) and the PTX_VERSION relabeling.
#include "cuda/CudaBinaryBlob.h"

#include <cstdio>
#include <string>

static int failures = 0;
static void require(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

int main() {
  using namespace cuda_blob;
  const std::string ptx = "//\n.version 9.1\n.target sm_89\n.entry k(\n.maxntid 256, 1, 1\n{\n}\n";
  std::string cubin("\177ELF\2\1\1", 7);
  cubin += std::string("\0\0\0binary\0data", 14);   // embedded NULs must survive

  // Without a CUBIN the blob is the plain PTX: what older cache files hold, and what a PTX-only build writes.
  require(join(ptx, "") == ptx, "a blob without a CUBIN is the plain PTX");
  std::string p, c;
  require(split(ptx, p, c) == Split::Ok && p == ptx && c.empty(), "a plain-PTX cache file loads as PTX only");

  // With a CUBIN: PTX first (clCreateKernel reads .maxntid from it), then the marker, then the CUBIN, byte-exact.
  const std::string blob = join(ptx, cubin);
  require(blob.compare(0, ptx.size(), ptx) == 0, "the PTX comes first");
  require(blob.size() == ptx.size() + kCubinMarkerLen + cubin.size(), "the blob is PTX + marker + CUBIN");
  require(split(blob, p, c) == Split::Ok && p == ptx && c == cubin, "the blob splits back into the PTX and the CUBIN");

  // A bare CUBIN has no PTX to read the work-group sizes from: refused, so the kernel is recompiled.
  p = "stale"; c = "stale";
  require(split(cubin, p, c) == Split::BareElf && p.empty() && c.empty(), "a bare CUBIN is refused");

  // PTX_VERSION is read from the build options; absent, nothing is relabeled.
  require(ptxVersionOverride("-DEXP=1u -DPTX_VERSION=8.7 -DWIDTH=256u") == "8.7", "PTX_VERSION in the middle");
  require(ptxVersionOverride("-DEXP=1u -DPTX_VERSION=8.5") == "8.5", "PTX_VERSION at the end");
  require(ptxVersionOverride("-DEXP=1u -DWIDTH=256u").empty(), "no PTX_VERSION");
  require(ptxVersionOverride("").empty(), "empty options");

  // The relabeling rewrites only the .version directive.
  std::string relabeled = ptx;
  require(setPtxVersion(relabeled, "8.7") == "9.1", "the old version is returned");
  require(relabeled == "//\n.version 8.7\n.target sm_89\n.entry k(\n.maxntid 256, 1, 1\n{\n}\n",
          "only the version number changes");
  std::string first = ".version 9.0\n.target sm_80\n";
  require(setPtxVersion(first, "8.0") == "9.0" && first == ".version 8.0\n.target sm_80\n",
          ".version on the first line");
  std::string last = "//\n.version 9.0";
  require(setPtxVersion(last, "8.7") == "9.0" && last == "//\n.version 8.7", ".version at the very end");
  std::string none = "// no version directive\n.target sm_80\n";
  require(setPtxVersion(none, "8.7").empty() && none == "// no version directive\n.target sm_80\n",
          "a PTX without .version is left alone");
  std::string comment = "// .version 1.0 in a comment\n.version 9.1\n";
  require(setPtxVersion(comment, "8.7") == "9.1" && comment == "// .version 1.0 in a comment\n.version 8.7\n",
          "a .version inside a comment is not the directive");

  if (failures) { std::printf("cuda binary blob test: %d FAILED\n", failures); return 1; }
  std::printf("cuda binary blob test PASS\n");
  return 0;
}
