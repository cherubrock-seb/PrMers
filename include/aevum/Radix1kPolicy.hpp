#pragma once

#include <string>

// Which radix the Aevum plugin uses for 1K FFT dimensions, so that the backend
// log reports what the engine will actually run.
//
// The engine API does not report this, so the plugin's own rule is mirrored here. The rule lives in
// third_party/aevum/src/FFTConfig.h (aevumRadix8For1K) and
// third_party/aevum/src/EngineApi.cpp (aevumSetDefaultRadix8For1K(isNvidiaGpu(device)) at engine creation):
//   - Apple builds always use radix-4: the staged pipelines require it, even with AEVUM_RADIX1K=8;
//   - otherwise AEVUM_RADIX1K, when set, decides (8 only for exactly "8");
//   - otherwise radix-8 on an NVIDIA device (PCI vendor 0x10DE, also the CUDA plugin) and radix-4 elsewhere.
// The plugin's AEVUM_FORCE_RADIX4_1K build option for host tests is not visible from here.
namespace aevum_policy {

struct Radix1k {
    unsigned radix;
    const char* source;
};

inline Radix1k radix1k_policy(const char* env_value, bool apple_build, bool nvidia_device) {
    if (apple_build) return {4u, "apple-forced"};
    if (env_value && *env_value)
        return {(env_value[0] == '8' && env_value[1] == '\0') ? 8u : 4u, "explicit-override"};
    return nvidia_device ? Radix1k{8u, "nvidia-default"} : Radix1k{4u, "safe-default"};
}

inline std::string radix1k_detail(const Radix1k& policy) {
    return "radix1k=" + std::to_string(policy.radix) + " " + policy.source;
}

} // namespace aevum_policy
