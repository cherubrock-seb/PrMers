// Device correctness/performance regression gate for production proof forward reuse.
#define CL_TARGET_OPENCL_VERSION 120
#include "math/Precompute.hpp"
#include "math/Carry.hpp"
#include "opencl/Context.hpp"
#include "opencl/Program.hpp"
#include "opencl/Buffers.hpp"
#include "opencl/Kernels.hpp"
#include "opencl/NttEngine.hpp"

#include <gmp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#ifndef PRMERS_KERNEL_DIR
#define PRMERS_KERNEL_DIR "kernels"
#endif

static void create_kernels(opencl::Kernels& kernels) {
    const char* names[] = {
        "kernel_sub2","kernel_sub1","kernel_carry","kernel_carry_2",
        "kernel_inverse_ntt_radix4_mm","kernel_ntt_radix4_last_m1_n4",
        "kernel_ntt_radix4_last_m1_n4_nosquare","kernel_inverse_ntt_radix4_mm_last",
        "kernel_ntt_radix4_last_m1","kernel_ntt_radix4_last_m1_nosquare",
        "kernel_ntt_radix4_mm_first","kernel_ntt_radix4_mm_m8",
        "kernel_ntt_radix4_mm_m4","kernel_ntt_radix4_mm_m2",
        "kernel_ntt_radix4_mm_m16","kernel_ntt_radix4_mm_m32",
        "kernel_inverse_ntt_radix4_m1","kernel_inverse_ntt_radix4_m1_n4",
        "kernel_ntt_radix4_inverse_mm_2steps","kernel_inverse_ntt_radix4_mm",
        "kernel_inverse_ntt_radix4_mm_last","kernel_ntt_radix4_inverse_mm_2steps_last",
        "kernel_ntt_radix4_mm_2steps","kernel_ntt_radix4_mm_2steps_first",
        "kernel_ntt_radix2_square_radix2","kernel_ntt_radix4_radix2_square_radix2_radix4",
        "kernel_ntt_radix4_square_radix4","kernel_pointwise_mul","kernel_ntt_radix2",
        "kernel_res64_display","kernel_ntt_radix5_mm_first",
        "kernel_ntt_inverse_radix5_mm_last","check_equal"
    };
    for (const char* n : names) {
        try { kernels.createKernel(n); }
        catch (...) {
            // Some duplicated/optional names above are harmless only if already created.
            try { (void)kernels.getKernel(n); }
            catch (...) { throw; }
        }
    }
}

static void words_to_mpz(mpz_t out, const std::vector<uint64_t>& words,
                         const std::vector<int>& widths) {
    mpz_set_ui(out, 0);
    mpz_t t;
    mpz_init(t);
    size_t bit = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        mpz_set_ui(t, 0);
        mpz_import(t, 1, -1, sizeof(uint64_t), 0, 0, &words[i]);
        mpz_mul_2exp(t, t, bit);
        mpz_add(out, out, t);
        bit += static_cast<size_t>(widths[i]);
    }
    mpz_clear(t);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const uint64_t p = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 100003ULL;
    const int batches = argc > 3 ? std::atoi(argv[3]) : 5;
    const uint64_t denseExp = 0xffffffffffffffffULL;

    math::Precompute pre(p);
    const size_t n = pre.getN();
    const size_t bytes = n * sizeof(uint64_t);

    prmers::ocl::Context ctx(device, 0, false, false);
    ctx.computeOptimalSizes(n, pre.getDigitWidth(), p, false, 0, 0);
    prmers::ocl::Program program(ctx, ctx.getDevice(), std::string(PRMERS_KERNEL_DIR) + "/prmers.cl", pre, "", false);
    opencl::Buffers buffers(ctx, pre);
    opencl::Kernels kernels(program.getProgram(), ctx.getQueue());
    create_kernels(kernels);
    opencl::NttEngine ntt(ctx, kernels, buffers, pre, false);
    math::Carry carry(ctx, ctx.getQueue(), program.getProgram(), n, pre.getDigitWidth(), buffers.digitWidthMaskBuf);

    cl_int err = CL_SUCCESS;
    cl_mem base = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) return 2;
    cl_mem out = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) return 2;
    cl_mem alias = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) return 2;

    std::vector<uint64_t> input(n, 0), actual(n), aliasOut(n), baseAfter(n);
    input[0] = 3;

    auto write = [&](cl_mem b) {
        if (clEnqueueWriteBuffer(ctx.getQueue(), b, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr) != CL_SUCCESS) std::abort();
    };
    auto read = [&](cl_mem b, std::vector<uint64_t>& v) {
        if (clEnqueueReadBuffer(ctx.getQueue(), b, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr) != CL_SUCCESS) std::abort();
    };
    auto run = [&](uint64_t e, bool aliased, std::vector<uint64_t>& dst) {
        if (aliased) {
            write(alias);
            ntt.powInPlace(alias, alias, e, carry, bytes);
            clFinish(ctx.getQueue());
            read(alias, dst);
        } else {
            write(base);
            ntt.powInPlace(out, base, e, carry, bytes);
            clFinish(ctx.getQueue());
            read(out, dst);
            read(base, baseAfter);
            if (baseAfter != input) {
                std::printf("BASE_PRESERVATION=FAIL exp=%llu\n", (unsigned long long)e);
                std::exit(3);
            }
        }
    };

    mpz_t M, expected, got;
    mpz_inits(M, expected, got, nullptr);
    mpz_ui_pow_ui(M, 2, static_cast<unsigned long>(p));
    mpz_sub_ui(M, M, 1);

    const uint64_t exps[] = {
        0ULL,1ULL,2ULL,3ULL,5ULL,6ULL,7ULL,
        0xaaaaaaaaaaaaaaaaULL,0xf0f0f0f0f0f0f0f0ULL,0xffffffffffffffffULL
    };

    int checked = 0;
    for (uint64_t e : exps) {
        run(e, false, actual);
        words_to_mpz(got, actual, pre.getDigitWidth());
        mpz_set_ui(expected, 3);
        mpz_powm_ui(expected, expected, static_cast<unsigned long>(e), M);
        mpz_mod(got, got, M);
        if (mpz_cmp(got, expected) != 0) {
            std::printf("ORACLE_MISMATCH exp=%llu\n",(unsigned long long)e);
            return 5;
        }
        if (e == 3 || e == 7 || e == denseExp) {
            run(e, true, aliasOut);
            if (aliasOut != actual) {
                std::printf("ALIAS_MISMATCH exp=%llu\n",(unsigned long long)e);
                return 6;
            }
        }
        ++checked;
    }

    auto timeOne = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        run(denseExp, false, actual);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double,std::micro>(t1-t0).count();
    };

    run(denseExp, false, actual);

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(batches));
    for (int b = 0; b < batches; ++b) samples.push_back(timeOne());
    const double med = median(samples);

    std::printf("P=%llu N=%zu CHECKED_EXPONENTS=%d\n",(unsigned long long)p,n,checked);
    std::printf("K1_PRODUCTION_EXACTNESS=PASS\n");
    std::printf("PROD_SAMPLES_US=");
    for (size_t i=0;i<samples.size();++i) std::printf("%s%.3f",i?",":"",samples[i]);
    std::printf("\nPROD_MEDIAN_US=%.3f\n",med);
    std::printf("K1_PRODUCTION_DEVICE_GATE=PASS\n");

    mpz_clears(M, expected, got, nullptr);
    clReleaseMemObject(base);
    clReleaseMemObject(out);
    clReleaseMemObject(alias);
    return 0;
}
