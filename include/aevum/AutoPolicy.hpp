#pragma once

#include "marin/engine.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct AevumAutoDecision {
    bool use_aevum = false;
    // True only when PrMers auto policy intentionally promotes a resolved
    // explicit plan over plugin-native auto. Ordinary native-auto decisions
    // keep this false so issue #36 still reaches Aevum as plugin-auto.
    bool force_fft_spec = false;
    // True only for the low-range ordinary PRP family where the native <=512K
    // Aevum plan is quarantined but a larger plan may still be both correct and
    // faster than Marin. create_gpu() performs the actual differential/timing gate.
    bool runtime_compare = false;
    std::size_t aevum_transform = 0;
    std::size_t marin_transform = 0;
    std::string fft_spec;
    std::string detail;
};

AevumAutoDecision aevum_auto_decide(std::uint32_t exponent,
                                    std::size_t register_count,
                                    engine::gpu_workload workload,
                                    const std::string& fft_spec = "");

const char* aevum_workload_name(engine::gpu_workload workload);
