// An out-of-range OpenCL device index must raise a clear error instead of
// reading past the end of the device list (which crashed inside the driver).
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "marin/ocl.h"

int main()
{
    const ocl::platform platform;
    const std::size_t count = platform.get_device_count();
    const std::size_t bad = count + 6;  // e.g. -d 7 with a single device

    int failures = 0;
    auto expect_error = [&](const char* what, auto&& fn) {
        try {
            fn();
            std::cerr << "FAIL: " << what << " accepted device index " << bad << "\n";
            ++failures;
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            if (msg.find("Invalid OpenCL device index " + std::to_string(bad)) == std::string::npos) {
                std::cerr << "FAIL: " << what << " unclear message: " << msg << "\n";
                ++failures;
            }
        }
    };

    expect_error("get_device", [&] { (void)platform.get_device(bad); });
    expect_error("get_platform", [&] { (void)platform.get_platform(bad); });
    expect_error("device ctor", [&] { ocl::device d(platform, bad, false); });

    if (count > 0) {
        // A valid index keeps working.
        (void)platform.get_device(0);
    }

    if (failures) return EXIT_FAILURE;
    std::cout << "Marin invalid device index regression: PASS\n";
    return EXIT_SUCCESS;
}
