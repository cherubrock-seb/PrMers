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
    bool aevum_compatible = true;
    const char* compatibility_reason = nullptr;
};

PolicyProfile profile_for(const engine::gpu_workload workload, const std::size_t register_count) {
    switch (workload) {
        case engine::gpu_workload::prp:
            return {1.00, "PRP/LL throughput", "AEVUM_AUTO_PRP_MAX_RATIO"};
        case engine::gpu_workload::ll:
            return {1.00, "PRP/LL throughput", "AEVUM_AUTO_LL_MAX_RATIO"};
        case engine::gpu_workload::pm1:
            if (register_count <= 16) {
                // Normal Stage 1 uses the generic engine API. Aevum is worthwhile
                // only with a clear transform-size advantage.
                return {0.75, "P-1 Stage 1", "AEVUM_AUTO_PM1_STAGE1_MAX_RATIO"};
            }
            return {1.00, "P-1 multi-register/Stage 2", "AEVUM_AUTO_PM1_STAGE2_MAX_RATIO"};
        case engine::gpu_workload::pm1_lowmem:
            // The 3-register low-memory implementation uses generic set/pow/mul
            // operations and is valid on both engines. Keep the same conservative
            // transform advantage as normal Stage 1.
            return {0.75, "P-1 low-memory (3-register)", "AEVUM_AUTO_PM1_LOWMEM_MAX_RATIO"};
        case engine::gpu_workload::pm1_ultralowmem:
            // The one-register implementation encodes multiply-by-3 in Marin's
            // fast3 square operation. It is an algorithmic incompatibility, not a
            // performance preference.
            return {0.0, "P-1 ultra-low-memory (1-register)", nullptr, false,
                    "Marin fast3-only one-register algorithm"};
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
        case engine::gpu_workload::pm1_lowmem: return "P-1 low-memory";
        case engine::gpu_workload::pm1_ultralowmem: return "P-1 ultra-low-memory";
        case engine::gpu_workload::ecm: return "ECM";
        default: return "generic";
    }
}

AevumAutoDecision aevum_auto_decide(const std::uint32_t exponent,
                                    const std::size_t register_count,
                                    const engine::gpu_workload workload,
                                    const std::string& fft_spec) {
    AevumAutoDecision result;
    result.marin_transform = ibdwt::transform_size(exponent);

    const PolicyProfile profile = profile_for(workload, register_count);
    if (!profile.aevum_compatible) {
        std::ostringstream out;
        out << "profile=" << profile.name
            << ", regs=" << register_count
            << ", Marin=" << result.marin_transform
            << ", compatibility=Marin-only";
        if (profile.compatibility_reason) out << " (" << profile.compatibility_reason << ")";
        result.detail = out.str();
        result.use_aevum = false;
        return result;
    }

    std::string reason;
    const bool resolved = fft_spec.empty()
        ? aevum_engine_resolve_auto_fft(exponent, &result.aevum_transform, &result.fft_spec, &reason)
        : aevum_engine_resolve_fft(exponent, fft_spec, &result.aevum_transform, &result.fft_spec, &reason);
    if (!resolved) {
        result.detail = reason;
        return result;
    }

    const double ratio = result.marin_transform == 0
        ? 1000.0
        : static_cast<double>(result.aevum_transform) / static_cast<double>(result.marin_transform);

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
