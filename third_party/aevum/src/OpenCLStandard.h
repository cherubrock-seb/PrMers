#pragma once

#include <string>

inline bool aevum_reports_opencl_1_x(const std::string& version) {
  return version.find("OpenCL C 1.") != std::string::npos ||
         version.find("OpenCL 1.") != std::string::npos;
}

inline std::string aevum_opencl_standard_arg(const std::string& opencl_c_version,
                                             const std::string& device_version,
                                             const bool force_cl12) {
  if (force_cl12 ||
      aevum_reports_opencl_1_x(opencl_c_version) ||
      (opencl_c_version.empty() && aevum_reports_opencl_1_x(device_version)) ||
      (opencl_c_version.empty() && device_version.empty())) {
    return "-cl-std=CL1.2 ";
  }
  return "-cl-std=CL2.0 ";
}
