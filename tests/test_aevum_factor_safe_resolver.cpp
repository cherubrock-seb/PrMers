// Host-only regression for the factor-safe Aevum resolver used by LL-SAFE.
#include "aevum/EngineAevum.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

static void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

static int run() {
    constexpr std::uint32_t exponent = 21000013u;
    std::size_t transform = 0;
    std::string plan;
    std::string reason;

    require(aevum_engine_resolve_factor_safe_fft(
                exponent, 1u, &transform, &plan, &reason),
            "factor-1 control must resolve");
    require(transform == 524288u, "factor-1 control must retain the 512K plan");
    require(plan == "1:256:4:256:101", "unexpected factor-1 plan: " + plan);

    transform = 0;
    plan.clear();
    reason.clear();
    require(aevum_engine_resolve_factor_safe_fft(
                exponent, 3u, &transform, &plan, &reason),
            "factor-3 request must find a safe fallback");
    require(transform == 1048576u, "factor-3 request must promote to the 1M plan");
    require(plan == "1:512:4:256:101", "unexpected factor-3 plan: " + plan);

    setenv("FAKE_AEVUM_FACTOR_SAFE_NONE", "1", 1);
    transform = 0;
    plan.clear();
    reason.clear();
    require(!aevum_engine_resolve_factor_safe_fft(
                exponent, 3u, &transform, &plan, &reason),
            "factor-3 request must fail closed when no safe plan exists");
    require(reason.find("No admissible factor-safe plan") != std::string::npos,
            "fail-closed reason must be actionable: " + reason);
    unsetenv("FAKE_AEVUM_FACTOR_SAFE_NONE");

    reason.clear();
    require(!aevum_engine_resolve_factor_safe_fft(
                exponent, 0u, &transform, &plan, &reason),
            "factor zero must be rejected");
    require(reason.find("positive") != std::string::npos,
            "factor-zero rejection must explain the invariant");

    std::cout << "Aevum factor-safe resolver test passed" << std::endl;
    return 0;
}

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
