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

    auto pfa3 = aevum_auto_decide(100000019u, 8, engine::gpu_workload::prp, "pfa:auto");
    expect(pfa3.use_aevum, true, "PFA-3 PRP");
    if (pfa3.aevum_transform != 3145728 || pfa3.fft_spec.rfind("pfa3:", 0) != 0) return 4;

    auto pfa9 = aevum_auto_decide(175000001u, 8, engine::gpu_workload::prp, "pfa:auto");
    expect(pfa9.use_aevum, true, "PFA-9 PRP");
    if (pfa9.aevum_transform != 4718592 || pfa9.marin_transform != 10485760 ||
        pfa9.fft_spec.rfind("pfa9:", 0) != 0) return 5;

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

#if !defined(_WIN32)
    setenv("AEVUM_AUTO_PM1_STAGE1_MAX_RATIO", "0.40", 1);
    auto forced_conservative = aevum_auto_decide(136279841u, 11, engine::gpu_workload::pm1);
    expect(forced_conservative.use_aevum, false, "P-1 Stage 1 env override");
    unsetenv("AEVUM_AUTO_PM1_STAGE1_MAX_RATIO");
#endif

    std::cout << "Aevum auto policy tests passed" << std::endl;
    return 0;
}
