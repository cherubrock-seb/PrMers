#include "aevum/AutoPolicy.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void expect(bool actual, bool expected, const char* label) {
    if (actual != expected) throw std::runtime_error(std::string(label) + " decision mismatch");
}

int main() {
    auto small = aevum_auto_decide(1362763u, 8, engine::gpu_workload::prp);
    expect(small.use_aevum, false, "small PRP");
    if (small.aevum_transform != 262144 || small.marin_transform != 65536) return 2;

    auto large = aevum_auto_decide(136279841u, 8, engine::gpu_workload::prp);
    expect(large.use_aevum, true, "large PRP");
    if (large.aevum_transform != 4194304 || large.marin_transform != 8388608) return 3;

    // Issue #36 regression: the device-neutral default must use Aevum's native
    // automatic selector rather than PrMers' optional throughput:prp policy.
    // At the reporter's 147.8M exponent, native Aevum selects Type1 FFT3161.
    auto issue36_default = aevum_auto_decide(147800003u, 8, engine::gpu_workload::prp);
    expect(issue36_default.use_aevum, true, "issue36 native-auto PRP");
    if (issue36_default.fft_spec.rfind("1:", 0) != 0) {
        std::cerr << "issue36 default selected " << issue36_default.fft_spec << std::endl;
        return 36;
    }
    if (issue36_default.force_fft_spec) {
        std::cerr << "issue36 default unexpectedly forced " << issue36_default.fft_spec << std::endl;
        return 37;
    }

#if !defined(__APPLE__)
    // Pass 4: default runtime autotune owns this boundary, so host policy must
    // not force the older v100.09 seed before the device-specific measurement.
    auto boundary197 = aevum_auto_decide(196999969u, 8, engine::gpu_workload::prp);
    expect(boundary197.use_aevum, true, "197M runtime-autotune admission");
    if (boundary197.force_fft_spec || boundary197.fft_spec.rfind("1:", 0) != 0) {
        std::cerr << "197M runtime auto unexpectedly forced " << boundary197.fft_spec << std::endl;
        return 38;
    }
#if !defined(_WIN32)
    // OFF restores the exact v100.09 host-side bridge for reproducibility and
    // strict manual-control precedence.
    setenv("AEVUM_AUTOTUNE", "off", 1);
    auto legacy_boundary197 = aevum_auto_decide(196999969u, 8, engine::gpu_workload::prp);
    unsetenv("AEVUM_AUTOTUNE");
    if (!legacy_boundary197.force_fft_spec ||
        legacy_boundary197.aevum_transform != 4194304u ||
        legacy_boundary197.fft_spec != "4:1K:8:256:101" ||
        legacy_boundary197.detail.find("boundary-bridge=1") == std::string::npos) return 138;
#endif

    // At 220M the same 4M plan is beyond the measured 46.97 bpw gate.
    auto beyond_bridge = aevum_auto_decide(220000001u, 8, engine::gpu_workload::prp);
    expect(beyond_bridge.use_aevum, true, "220M native auto after bridge");
    if (beyond_bridge.force_fft_spec || beyond_bridge.fft_spec.rfind("1:", 0) != 0) {
        std::cerr << "220M unexpectedly bridged to " << beyond_bridge.fft_spec << std::endl;
        return 39;
    }
#endif

#if defined(__APPLE__)
    // Apple OpenCL 1.2 deliberately disables native PFA and Type4.
    // Verify platform rejection/fallback rather than Linux/Windows plans.
    auto pfa3 = aevum_auto_decide(100000019u, 8, engine::gpu_workload::prp, "pfa:auto");
    expect(pfa3.use_aevum, false, "Apple PFA-3 disabled");
    if (!pfa3.fft_spec.empty()) return 40;

    auto pfa9 = aevum_auto_decide(175000001u, 8, engine::gpu_workload::prp, "pfa:auto");
    expect(pfa9.use_aevum, false, "Apple PFA-9 disabled");
    if (!pfa9.fft_spec.empty()) return 50;

    auto require_type1 = [](const AevumAutoDecision& d, int code) {
        if (d.fft_spec.rfind("1:", 0) != 0) {
            std::cerr << "Apple selector did not resolve Type1: " << d.fft_spec << std::endl;
            std::exit(code);
        }
    };

    require_type1(aevum_auto_decide(175000039u, 8, engine::gpu_workload::prp, "throughput:auto"), 60);
    require_type1(aevum_auto_decide(175000039u, 8, engine::gpu_workload::prp, "throughput:prp"), 61);
    require_type1(aevum_auto_decide(175000039u, 18, engine::gpu_workload::ll, "throughput:ll"), 62);
    require_type1(aevum_auto_decide(55050557u, 11, engine::gpu_workload::pm1, "throughput:pm1"), 63);
    require_type1(aevum_auto_decide(55050557u, 51, engine::gpu_workload::ecm, "throughput:ecm"), 64);
#else
    auto pfa3 = aevum_auto_decide(100000019u, 8, engine::gpu_workload::prp, "pfa:auto");
    expect(pfa3.use_aevum, true, "PFA-3 PRP");
    if (pfa3.aevum_transform != 3145728 || pfa3.fft_spec.rfind("pfa3:", 0) != 0) return 4;

    auto pfa9 = aevum_auto_decide(175000001u, 8, engine::gpu_workload::prp, "pfa:auto");
    expect(pfa9.use_aevum, true, "PFA-9 PRP");
    if (pfa9.aevum_transform != 4718592 || pfa9.marin_transform != 10485760 ||
        pfa9.fft_spec.rfind("pfa9:", 0) != 0) return 5;

    auto throughput175 = aevum_auto_decide(175000039u, 8, engine::gpu_workload::prp, "throughput:auto");
    expect(throughput175.use_aevum, true, "M175 throughput auto");
    if (throughput175.aevum_transform != 4194304 ||
        throughput175.fft_spec != "4:512:8:512:202") return 6;

    auto prp_workload = aevum_auto_decide(175000039u, 8, engine::gpu_workload::prp, "throughput:prp");
    if (!prp_workload.use_aevum || prp_workload.fft_spec != "4:512:8:512:202") return 61;

    auto ll_workload = aevum_auto_decide(175000039u, 18, engine::gpu_workload::ll, "throughput:ll");
    if (!ll_workload.use_aevum || ll_workload.fft_spec != "4:1K:2:1K:202") return 62;

    auto pm1_workload = aevum_auto_decide(55050557u, 11, engine::gpu_workload::pm1, "throughput:pm1");
    if (!pm1_workload.use_aevum || pm1_workload.fft_spec != "4:256:16:256:202") return 63;

    auto ecm_workload = aevum_auto_decide(55050557u, 51, engine::gpu_workload::ecm, "throughput:ecm");
    if (!ecm_workload.use_aevum || ecm_workload.fft_spec != "1:1K:4:256:101") return 64;
#endif

    auto small_stage1 = aevum_auto_decide(1362763u, 11, engine::gpu_workload::pm1);
    expect(small_stage1.use_aevum, false, "small P-1 Stage 1");

    auto medium_stage1 = aevum_auto_decide(16279841u, 11, engine::gpu_workload::pm1);
    expect(medium_stage1.use_aevum, true, "medium P-1 Stage 1");

    auto large_stage1 = aevum_auto_decide(136279841u, 11, engine::gpu_workload::pm1);
    expect(large_stage1.use_aevum, true, "large P-1 Stage 1");

    auto stage2 = aevum_auto_decide(136279841u, 158, engine::gpu_workload::pm1);
    expect(stage2.use_aevum, true, "P-1 Stage 2");


    auto small_ll = aevum_auto_decide(1362763u, 18, engine::gpu_workload::ll);
    expect(small_ll.use_aevum, false, "small LL");

    auto large_ll = aevum_auto_decide(136279841u, 18, engine::gpu_workload::ll);
    expect(large_ll.use_aevum, true, "large LL");

    auto lowmem_small = aevum_auto_decide(1362763u, 3, engine::gpu_workload::pm1_lowmem);
    expect(lowmem_small.use_aevum, false, "small P-1 low-memory");

    auto lowmem_large = aevum_auto_decide(136279841u, 3, engine::gpu_workload::pm1_lowmem);
    expect(lowmem_large.use_aevum, true, "large P-1 low-memory");

    auto ultralow = aevum_auto_decide(2147483647u, 1, engine::gpu_workload::pm1_ultralowmem);
    expect(ultralow.use_aevum, false, "P-1 ultra-low-memory compatibility");
    if (ultralow.detail.find("Marin-only") == std::string::npos) return 8;

    auto small_ecm = aevum_auto_decide(1362763u, 51, engine::gpu_workload::ecm);
    expect(small_ecm.use_aevum, false, "small ECM");

    auto large_ecm = aevum_auto_decide(136279841u, 51, engine::gpu_workload::ecm);
    expect(large_ecm.use_aevum, true, "large ECM");

    auto too_small = aevum_auto_decide(216091u, 8, engine::gpu_workload::prp);
    expect(too_small.use_aevum, false, "unsupported PRP");

#if !defined(__APPLE__)
    // GMNet/PrimePages known Gaussian-Mersenne prime exponent 3,704,053.
    // Gaussian arithmetic is lifted to 4p = 14,816,212, which is large
    // enough to exercise the real Aevum policy while remaining a practical
    // installation smoke-test size.
    constexpr std::uint32_t gaussian_smoke_lift = 4u * 3704053u;

    auto gaussian_prp = aevum_auto_decide(
        gaussian_smoke_lift, 1, engine::gpu_workload::prp, "throughput:prp");
    expect(gaussian_prp.use_aevum, true, "Gaussian lifted PRP");
    if (gaussian_prp.aevum_transform != 524288 ||
        gaussian_prp.fft_spec != "4:256:4:256:202") return 9;

    auto gaussian_pm1 = aevum_auto_decide(
        gaussian_smoke_lift, 15, engine::gpu_workload::pm1, "throughput:pm1");
    expect(gaussian_pm1.use_aevum, true, "Gaussian lifted P-1");
    if (gaussian_pm1.aevum_transform != 524288 ||
        gaussian_pm1.fft_spec != "4:256:4:256:202") return 10;

    auto gaussian_ecm = aevum_auto_decide(
        gaussian_smoke_lift, 24, engine::gpu_workload::ecm, "throughput:ecm");
    expect(gaussian_ecm.use_aevum, true, "Gaussian lifted ECM");
    if (gaussian_ecm.aevum_transform != 524288 ||
        gaussian_ecm.fft_spec != "1:256:4:256:101") return 11;
#endif

#if !defined(_WIN32)
    setenv("AEVUM_AUTO_PM1_STAGE1_MAX_RATIO", "0.40", 1);
    auto forced_conservative = aevum_auto_decide(136279841u, 11, engine::gpu_workload::pm1);
    expect(forced_conservative.use_aevum, false, "P-1 Stage 1 env override");
    unsetenv("AEVUM_AUTO_PM1_STAGE1_MAX_RATIO");
#endif

    std::cout << "Aevum auto policy tests passed" << std::endl;
    return 0;
}
