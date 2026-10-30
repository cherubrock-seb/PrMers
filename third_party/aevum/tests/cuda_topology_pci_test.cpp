// The CUDA shim answers CL_DEVICE_TOPOLOGY_AMD from the device's PCI bus id, so `-pci <bus:device.function>` can select a CUDA
// device the way it selects an AMD one (getBdfFromDevice reads the bus, device and function from this struct).
//
// Builds the real src/cuda/clwrap_cuda.cpp against the stand-in CUDA driver in tests/cuda_stub, whose bus id this test sets:
// no GPU and no CUDA install needed.
#include "cuda/tinycuda.h"
#include "stub_driver.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
using cuda_stub::pciBusId;
using cuda_stub::pciBusIdResult;
int failures = 0;

void require(bool ok, const std::string& what) {
  if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}
}  // namespace

static int topology(cl_device_id dev, cl_device_topology_amd* top) {
  std::memset(top, 0xEE, sizeof(*top));
  return clGetDeviceInfo(dev, CL_DEVICE_TOPOLOGY_AMD, sizeof(*top), top, nullptr);
}

// The "bus:device.function" text getBdfFromDevice (src/clwrap.cpp) makes from the struct.
static std::string bdfOf(const cl_device_topology_amd& top) {
  char s[32];
  std::snprintf(s, sizeof s, "%02x:%02x.%x", (unsigned) (unsigned char) top.pcie.bus, (unsigned) top.pcie.device,
                (unsigned) top.pcie.function);
  return s;
}

static void expectBdf(cl_device_id dev, const char* driverSays, const char* want) {
  pciBusId = driverSays;
  pciBusIdResult = CUDA_SUCCESS;
  cl_device_topology_amd top;
  int err = topology(dev, &top);
  require(err == CL_SUCCESS, std::string("topology of \"") + driverSays + "\" succeeds");
  if (err != CL_SUCCESS) { return; }
  require(top.pcie.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD, std::string("PCIe type tag for \"") + driverSays + "\"");
  require(bdfOf(top) == want, std::string("\"") + driverSays + "\" gives " + want + ", got " + bdfOf(top));
}

static void expectInvalid(cl_device_id dev, const char* driverSays, CUresult driverResult, const char* what) {
  pciBusId = driverSays;
  pciBusIdResult = driverResult;
  cl_device_topology_amd top;
  require(topology(dev, &top) == CL_INVALID_VALUE, what);
}

int main() {
  cl_device_id dev{};
  unsigned n = 0;
  require(clGetDeviceIDs(nullptr, 0, 1, &dev, &n) == CL_SUCCESS && n == 1, "clGetDeviceIDs");

  expectBdf(dev, "0000:6a:00.0", "6a:00.0");
  expectBdf(dev, "0000:01:00.0", "01:00.0");
  expectBdf(dev, "0000:c1:1f.7", "c1:1f.7");        // bus above 0x7f must not turn negative
  expectBdf(dev, "0000:C1:00.1", "c1:00.1");        // upper-case hex
  expectBdf(dev, "00000000:ff:00.0", "ff:00.0");    // a longer domain
  expectBdf(dev, "6a:00.0", "6a:00.0");             // a missing domain is tolerated
  expectBdf(dev, "0001:3b:00.0", "3b:00.0");        // a non-zero domain: the bus is what -pci names

  expectInvalid(dev, "0000:6a:00.0", CUDA_ERROR_INVALID_VALUE, "a driver that cannot say gives CL_INVALID_VALUE");
  expectInvalid(dev, "0000:6a:00.0", CUDA_ERROR_NOT_INITIALIZED, "an uninitialised driver gives CL_INVALID_VALUE");
  expectInvalid(dev, "", CUDA_SUCCESS, "an empty bus id gives CL_INVALID_VALUE");
  expectInvalid(dev, "garbage", CUDA_SUCCESS, "an unparsable bus id gives CL_INVALID_VALUE");
  expectInvalid(dev, "0000:6a", CUDA_SUCCESS, "a truncated bus id gives CL_INVALID_VALUE");

  // A size query reports the struct size and writes nothing; a too-small buffer is left alone.
  pciBusId = "0000:6a:00.0";
  pciBusIdResult = CUDA_SUCCESS;
  size_t len = 0;
  require(clGetDeviceInfo(dev, CL_DEVICE_TOPOLOGY_AMD, 0, nullptr, &len) == CL_SUCCESS && len == sizeof(cl_device_topology_amd),
          "size query reports sizeof(cl_device_topology_amd)");
  char small[sizeof(cl_device_topology_amd)];
  std::memset(small, 0x5A, sizeof small);
  clGetDeviceInfo(dev, CL_DEVICE_TOPOLOGY_AMD, sizeof(cl_device_topology_amd) - 1, small, nullptr);
  bool untouched = true;
  for (char c : small) { untouched = untouched && c == 0x5A; }
  require(untouched, "a too-small buffer is not written");

  // The other AMD-only queries have no CUDA counterpart and still say so.
  u64 scratch[8];
  require(clGetDeviceInfo(dev, CL_DEVICE_BOARD_NAME_AMD, sizeof scratch, scratch, nullptr) == CL_INVALID_VALUE,
          "CL_DEVICE_BOARD_NAME_AMD is still unsupported");
  require(clGetDeviceInfo(dev, CL_DEVICE_PCIE_ID_AMD, sizeof scratch, scratch, nullptr) == CL_INVALID_VALUE,
          "CL_DEVICE_PCIE_ID_AMD is still unsupported");

  if (failures) { std::printf("cuda topology pci test: %d FAILED\n", failures); return 1; }
  std::printf("cuda topology pci test PASS\n");
  return 0;
}
