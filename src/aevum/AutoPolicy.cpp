#include "aevum/AutoPolicy.hpp"
#include "aevum/EngineAevum.hpp"
#include "marin/ibdwt.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

double parse_env_ratio(const char* name, const double fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || parsed <= 0.0) return fallback;
    return parsed;
}

struct PolicyProfile {
    double limit = 1.0;
    const char* name = "generic";
    const char* env_name = nullptr;
};

PolicyProfile profile_for(const engine::gpu_workload workload, const std::size_t register_count) {
    switch (workload) {
        case engine::gpu_workload::prp:
            return {1.00, "PRP/LL throughput", "AEVUM_AUTO_PRP_MAX_RATIO"};
        case engine::gpu_workload::ll:
            return {1.00, "PRP/LL throughput", "AEVUM_AUTO_LL_MAX_RATIO"};
        case engine::gpu_workload::pm1:
            if (register_count <= 16) {
                // Stage 1 uses few registers. Aevum is worthwhile only with a clear
                // transform-size advantage. On the measured Radeon VII case,
                // M136279841 is 4M words in Aevum versus 8M in Marin and Aevum wins.
                return {0.75, "P-1 Stage 1", "AEVUM_AUTO_PM1_STAGE1_MAX_RATIO"};
            }
            // Stage 2 uses many resident registers, so equal transform sizes are
            // acceptable and the smaller Aevum representation can reduce memory.
            return {1.00, "P-1 multi-register/Stage 2", "AEVUM_AUTO_PM1_STAGE2_MAX_RATIO"};
        case engine::gpu_workload::ecm:
            // ECM has 51 long-lived registers and many mixed operations. Stay
            // conservative unless Aevum saves at least 25% of transform length.
            return {0.75, "ECM conservative", "AEVUM_AUTO_ECM_MAX_RATIO"};
        default:
            return {0.75, "generic conservative", "AEVUM_AUTO_GENERIC_MAX_RATIO"};
    }
}

} // namespace

const char* aevum_workload_name(const engine::gpu_workload value) {
    switch (value) {
        case engine::gpu_workload::prp: return "PRP";
        case engine::gpu_workload::ll: return "LL";
        case engine::gpu_workload::pm1: return "P-1";
        case engine::gpu_workload::ecm: return "ECM";
        default: return "generic";
    }
}

AevumAutoDecision aevum_auto_decide(const std::uint32_t exponent,
                                    const std::size_t register_count,
                                    const engine::gpu_workload workload) {
    AevumAutoDecision result;
    std::string reason;
    if (!aevum_engine_resolve_auto_fft(exponent, &result.aevum_transform, &result.fft_spec, &reason)) {
        result.detail = reason;
        return result;
    }

    result.marin_transform = ibdwt::transform_size(exponent);
    const double ratio = result.marin_transform == 0
        ? 1000.0
        : static_cast<double>(result.aevum_transform) / static_cast<double>(result.marin_transform);

    const PolicyProfile profile = profile_for(workload, register_count);
    double limit = profile.limit;
    // Global override first, then the workload-specific override.
    limit = parse_env_ratio("AEVUM_AUTO_MAX_RATIO", limit);
    if (profile.env_name) limit = parse_env_ratio(profile.env_name, limit);

    std::ostringstream out;
    out << "profile=" << profile.name
        << ", regs=" << register_count
        << ", Aevum=" << result.aevum_transform
        << ", Marin=" << result.marin_transform
        << ", ratio=" << std::fixed << std::setprecision(2) << ratio
        << ", limit=" << std::fixed << std::setprecision(2) << limit
        << ", FFT=" << result.fft_spec;
    result.detail = out.str();
    result.use_aevum = ratio <= limit;
    return result;
}
