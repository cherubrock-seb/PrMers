#pragma once

#include "marin/engine.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct AevumAutoDecision {
    bool use_aevum = false;
    std::size_t aevum_transform = 0;
    std::size_t marin_transform = 0;
    std::string fft_spec;
    std::string detail;
};

AevumAutoDecision aevum_auto_decide(std::uint32_t exponent,
                                    std::size_t register_count,
                                    engine::gpu_workload workload);

const char* aevum_workload_name(engine::gpu_workload workload);
