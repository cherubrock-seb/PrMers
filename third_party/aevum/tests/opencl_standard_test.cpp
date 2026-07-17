#include "OpenCLStandard.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void expect(const std::string& got, const std::string& expected, const char* name) {
  if (got != expected) {
    std::cerr << name << ": expected '" << expected << "', got '" << got << "'\n";
    std::exit(1);
  }
}

} // namespace

int main() {
  expect(aevum_opencl_standard_arg("OpenCL C 1.2 ", "OpenCL 1.2 ", false),
         "-cl-std=CL1.2 ", "OpenCL C 1.2");
  expect(aevum_opencl_standard_arg("", "OpenCL 1.2 ", false),
         "-cl-std=CL1.2 ", "device-version fallback");
  expect(aevum_opencl_standard_arg("OpenCL C 3.0 ", "OpenCL 3.0 ", false),
         "-cl-std=CL2.0 ", "OpenCL C 3.0");
  expect(aevum_opencl_standard_arg("OpenCL C 3.0 ", "OpenCL 3.0 ", true),
         "-cl-std=CL1.2 ", "Apple forced CL1.2");
  expect(aevum_opencl_standard_arg("", "", false),
         "-cl-std=CL1.2 ", "unknown implementation fallback");
  std::cout << "OpenCL standard selection tests passed\n";
  return 0;
}
