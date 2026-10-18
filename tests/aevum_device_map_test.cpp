// PrMers numbers OpenCL GPUs only; the Aevum plugin numbers every device of every platform.  The PrMers -d index
// must be translated to the plugin's numbering, or a CPU OpenCL runtime makes Aevum open a different device.
#include "aevum/DeviceMap.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using aevum_device_map::map_index;
using aevum_device_map::Status;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

static void expect(const std::vector<int>& prmers, const std::vector<int>& aevum, std::size_t d,
                   Status status, std::size_t index, const std::string& what) {
    const auto r = map_index(prmers, aevum, d);
    check(r.status == status && r.index == index,
          what + ": got status " + std::to_string(static_cast<int>(r.status)) + " index " + std::to_string(r.index));
}

int main() {
    // GPU-only system: the numberings agree.
    expect({10, 11}, {10, 11}, 1, Status::mapped, 1, "GPUs only");

    // Intel platform lists its CPU runtime before the iGPU, NVIDIA platform after it.
    // PrMers: 0 = Intel iGPU, 1 = NVIDIA.  Aevum: 0 = Intel CPU, 1 = Intel iGPU, 2 = NVIDIA.
    expect({20, 30}, {21, 20, 30}, 0, Status::mapped, 1, "Intel CPU runtime shifts the iGPU");
    expect({20, 30}, {21, 20, 30}, 1, Status::mapped, 2, "Intel CPU runtime shifts the NVIDIA GPU");

    // PoCL platform enumerated first, two AMD GPUs after it.
    expect({40, 41}, {99, 40, 41}, 1, Status::mapped, 2, "PoCL first, second GPU");

    // CPU runtime enumerated last: no shift.
    expect({40, 41}, {40, 41, 99}, 1, Status::mapped, 1, "CPU platform last");

    // No GPU anywhere: PrMers falls back to all devices, same order as Aevum.
    expect({99}, {99}, 0, Status::mapped, 0, "CPU-only fallback");

    // Nothing to translate when PrMers sees no device.
    expect({}, {}, 3, Status::no_devices, 3, "no devices");

    expect({40, 41}, {99, 40, 41}, 2, Status::out_of_range, 2, "index past the end");
    expect({40, 41}, {99, 40}, 1, Status::not_found, 1, "device missing from the plugin list");

    if (failures) return EXIT_FAILURE;
    std::puts("aevum device map: PASS");
    return EXIT_SUCCESS;
}
