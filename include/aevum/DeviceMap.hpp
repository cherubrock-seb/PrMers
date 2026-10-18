#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

// PrMers (-d, Marin, the GUI) numbers OpenCL GPU devices only, across platforms, and falls back to all devices
// when there is no GPU.  The Aevum plugin numbers every OpenCL device of every platform, CPUs included.  The two
// numberings agree only when no platform exposes a CPU device, so a PrMers index must be translated before it is
// handed to the plugin.
namespace aevum_device_map {

enum class Status {
    mapped,        // index holds the Aevum index of the PrMers device
    no_devices,    // PrMers sees no OpenCL device: nothing to translate, pass the index through
    out_of_range,  // the PrMers index does not name a device
    not_found      // the PrMers device is missing from Aevum's list
};

struct Result {
    Status status;
    std::size_t index;
};

template <class Id>
Result map_index(const std::vector<Id>& prmers_devices, const std::vector<Id>& aevum_devices, std::size_t prmers_index) {
    if (prmers_devices.empty()) return {Status::no_devices, prmers_index};
    if (prmers_index >= prmers_devices.size()) return {Status::out_of_range, prmers_index};
    const auto it = std::find(aevum_devices.begin(), aevum_devices.end(), prmers_devices[prmers_index]);
    if (it == aevum_devices.end()) return {Status::not_found, prmers_index};
    return {Status::mapped, static_cast<std::size_t>(it - aevum_devices.begin())};
}

}  // namespace aevum_device_map
