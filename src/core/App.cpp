// src/core/App.cpp
/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
//#define CL_TARGET_OPENCL_VERSION 200
#define NOMINMAX
#include "core/App.hpp"
#include "core/InheritedSignals.hpp"
#include "core/LegacyLlGuard.hpp"
#include "core/AlgoUtils.hpp"
#include "core/GmChainProgress.hpp"
#include "core/QuickChecker.hpp"
#include "core/Printer.hpp"
#include "core/ProofSet.hpp"
#include "core/ProofSetMarin.hpp"
#include "math/Carry.hpp"
#include "util/GmpUtils.hpp"
#include "io/WorktodoParser.hpp"
#include "io/MersFileName.hpp"
#include "io/ExponentInput.hpp"
#include "io/WorktodoManager.hpp"
#include "marin/engine.h"
#include "aevum/AutoPolicy.hpp"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <map>
#include <future>
#include <algorithm>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#ifdef _WIN32
# include <windows.h>
#endif
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <tuple>
#include <atomic>
#include <mutex>
#include "core/StopRestartGate.hpp"
#include <fstream>
#include <memory>
#include <optional>
#include <cmath>
#include <thread>
#include <gmp.h>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <set>

//namespace {
//std::atomic<bool> g_stop{false};
//void g_onSig(int){ g_stop = true; }
//}

namespace fs = std::filesystem;

using namespace std::chrono;
using core::algo::format_res64_hex;
using core::algo::format_res2048_hex;
using core::algo::helperu;
using core::algo::mod3_words;
using core::algo::delete_checkpoints;
using core::algo::to_uppercase;
using core::algo::div3_words;
using core::algo::pack_words_from_eng_digits;
using core::algo::prp3_div9;
using core::algo::parseConfigFile;
using core::algo::interrupted;
using core::algo::handle_sigint;
using core::algo::writeStageResult;
using core::algo::restart_self;
using core::algo::buildE;
using core::algo::evenGapBound;
using core::algo::primeCountApprox;
using core::algo::read_mers_file;
using core::algo::writeEcmResumeLine;
using core::algo::mpz_to_lower_hex;
using core::algo::ecm_checksum_pminus1;
using core::algo::CHKSUMMOD;
using core::algo::write_prime95_s1_from_bytes;
using core::algo::checksum_prime95_s1;
using core::algo::hex_to_bytes_reversed_pad8;
using core::algo::parse_ecm_resume_line;
using core::algo::read_text_file;
using core::algo::write_u32;
using core::algo::write_i32;
using core::algo::write_u64;
using core::algo::write_u16;
using core::algo::write_f64;
using core::algo::write_u8;
using core::algo::hex_to_le_bytes_pad4;
using core::algo::buildE2;
using core::algo::product_prefix_fit_u64;
using core::algo::product_tree_range_u64;
using core::algo::compute_X_with_dots;
using core::algo::gcd_with_dots;


namespace core {



static uint64_t askExponentInteractively() {
  uint64_t exponent = 0;
  #ifdef _WIN32
    char buffer[32];
    MessageBoxA(
      nullptr,
      "PrMers: GPU accelerated Mersenne primality tester\n\n"
      "You'll now be asked which exponent you'd like to test.",
      "PrMers - Select Exponent",
      MB_OK | MB_ICONINFORMATION
    );
    std::cout << "Enter the exponent to test (e.g. 21701): ";
    std::cin.getline(buffer, sizeof(buffer));
    if (!io::parseExponentAnswer(buffer, exponent)) {
      std::cerr << "Invalid input. Aborting.\n";
      std::exit(1);
    }
    return exponent;
  #else
    std::cout << "============================================\n"
              << " PrMers: GPU-accelerated Mersenne primality test\n"
              << " Powered by OpenCL | NTT | LL | PRP | IBDWT\n"
              << "============================================\n\n";
    std::string input;
    std::cout << "Enter the exponent to test (e.g. 21701): ";
    std::getline(std::cin, input);
    if (!io::parseExponentAnswer(input, exponent)) {
      std::cerr << "Invalid input. Aborting.\n";
      std::exit(1);
    }
    return exponent;
  #endif
}
static engine::gpu_backend gaussian_selected_backend(const io::CliOptions& options) {
#if defined(__APPLE__)
    return (!options.marin || options.force_engine_marin)
        ? engine::gpu_backend::marin
        : (options.aevum ? engine::gpu_backend::aevum : engine::gpu_backend::marin);
#else
    return (!options.marin || options.force_engine_marin)
        ? engine::gpu_backend::marin
        : (options.aevum ? engine::gpu_backend::aevum : engine::gpu_backend::auto_select);
#endif
}

static std::string gaussian_workload_fft_spec(const io::CliOptions& options,
                                               const engine::gpu_workload workload) {
    if (options.aevum_fft_spec_explicit) return options.aevum_fft_spec;
#if defined(__APPLE__)
    if (workload == engine::gpu_workload::prp || workload == engine::gpu_workload::ll)
        return "pow2:auto";
    return {};
#else
    const char* override_value = nullptr;
    const char* fallback = nullptr;
    switch (workload) {
        case engine::gpu_workload::prp:
            override_value = std::getenv("PRMERS_AEVUM_PRP_FFT");
            // Device-neutral default: delegate to the native Aevum auto selector.
            // The old PRP workload fallback was calibrated on RTX 3080 and can
            // choose a much slower Type4 plan on other GPUs (issue #36 / RTX 5090).
            fallback = "";
            break;
        case engine::gpu_workload::pm1:
        case engine::gpu_workload::pm1_lowmem:
            override_value = std::getenv("PRMERS_AEVUM_PM1_FFT");
            // Empty delegates to the Pass-4 workload-aware runtime tuner.
            fallback = "";
            break;
        case engine::gpu_workload::ecm:
            override_value = std::getenv("PRMERS_AEVUM_ECM_FFT");
            fallback = "";
            break;
        default:
            return options.aevum_fft_spec;
    }
    return override_value && *override_value ? override_value : fallback;
#endif
}

static void configure_gaussian_phase_backend(io::CliOptions& options,
                                              const engine::gpu_workload workload,
                                              const char* phase) {
    const std::string fft_spec = gaussian_workload_fft_spec(options, workload);
    engine::configure_gpu_backend(gaussian_selected_backend(options), fft_spec, workload);
    options.aevum_fft_spec = fft_spec;
    std::cout << "[Gaussian backend policy] " << phase
              << " workload=" << aevum_workload_name(workload)
              << " backend=" << engine::configured_gpu_backend_name();
    if (!fft_spec.empty()) std::cout << " fft=" << fft_spec;
    std::cout << "\n";
}

void App::tuneIterforce() {
    double maxTestSeconds = 60.0;
    uint64_t defaultTestIters = 1024;
    uint64_t sampleIters = std::min<uint64_t>(64, defaultTestIters);
    std::cout << "Sampling " << sampleIters << " iterations at iterforce=10\n";
    double sampleIps = measureIps(10, sampleIters);
    std::cout << "Estimated IPS: " << sampleIps << "\n";

    uint64_t testIters = static_cast<uint64_t>(sampleIps * maxTestSeconds);
    if (testIters == 0) testIters = defaultTestIters;
    std::cout << "Test iterations: " << testIters << " (~" << maxTestSeconds << "s)\n";

    uint64_t boundHigh = static_cast<uint64_t>(sampleIps * maxTestSeconds);
    if (boundHigh < 1) boundHigh = 1;
    std::cout << "Range: [1, " << boundHigh << "]\n";

    uint64_t low = 1, high = boundHigh;
    uint64_t best = low;
    double bestIps = 0.0;

    while (low < high) {
        uint64_t mid = low + (high - low) / 2;
        double ipsMid = measureIps(mid, testIters);
        double ipsNext = measureIps(mid + 50, testIters);
        std::cout << "iterforce=" << mid << " IPS=" << ipsMid
                  << " | " << (mid+1) << " IPS=" << ipsNext << "\n";
        if (ipsMid < ipsNext) {
            low = mid + 50;
            if (ipsNext > bestIps) { bestIps = ipsNext; best = mid + 50; }
        } else {
            high = mid;
            if (ipsMid > bestIps) { bestIps = ipsMid; best = mid; }
        }
    }
    std::cout << "Optimal iterforce=" << best << " IPS=" << bestIps << "\n";
    
}



double App::measureIps(uint64_t testIterforce, uint64_t testIters) {
    uint64_t oldIterforce = options.iterforce;
    options.iterforce = testIterforce;

    if (!buffers->input) {
        std::vector<uint64_t> x(precompute.getN(), 0ULL);
        x[0] = (options.mode == "prp") ? 3ULL : 4ULL;
        cl_int err = CL_SUCCESS;
        buffers->input = clCreateBuffer(
            context.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            x.size() * sizeof(uint64_t),
            x.data(), &err
        );
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer input failed");
    }

    math::Carry carry(
        context,
        context.getQueue(),
        program->getProgram(),
        precompute.getN(),
        precompute.getDigitWidth(),
        buffers->digitWidthMaskBuf
    );

    const int barWidth = 40;
    uint64_t markInterval = std::max<uint64_t>(1, testIters / barWidth);
    std::cout << "    Running iterforce=" << testIterforce << ": [";

    auto start = high_resolution_clock::now();
    for (uint64_t iter = 1; iter <= testIters; ++iter) {
        nttEngine->forward(buffers->input, iter - 1);
        nttEngine->inverse(buffers->input, iter - 1);
        carry.carryGPU(
            buffers->input,
            buffers->blockCarryBuf,
            precompute.getN() * sizeof(uint64_t)
        );
        if (iter % options.iterforce == 0) clFinish(context.getQueue());
        if (iter % markInterval == 0) std::cout << "." << std::flush;
    }
    clFinish(context.getQueue());
    auto end = high_resolution_clock::now();
    std::cout << "]\n";

    options.iterforce = oldIterforce;
    double seconds = duration_cast<duration<double>>(end - start).count();
    return testIters / seconds;
}




void App::adoptLegacyProofResidues(uint64_t resumeIter)
{
    if (options.mode != "prp" || !options.proof || options.wagstaff || resumeIter == 0)
        return;
    // The GPU proof reads the residues through proofManager and the CPU
    // fallback through proofManagerMarin, so both must agree on where they are.
    std::string note;
    std::string noteMarin;
    const bool adopted = proofManager.adoptLegacyResidues(
        static_cast<uint32_t>(resumeIter), note);
    const bool adoptedMarin = proofManagerMarin.adoptLegacyResidues(
        static_cast<uint32_t>(resumeIter), noteMarin);
    if (adopted != adoptedMarin)
        std::cerr << "Warning: the proof residue locations of the two proof backends differ" << std::endl;
    if (note.empty()) note = noteMarin;
    if (note.empty()) return;
    std::cout << note << std::endl;
    if (guiServer_) guiServer_->appendLog(note);
}

void App::ensureProofGpuBackend()
{
    if (buffers && program && kernels && nttEngine) return;

    if (buffers || program || kernels || nttEngine) {
        throw std::runtime_error(
            "Partial PrMers GPU proof backend initialization state");
    }

    try {
        buffers.emplace(context, precompute);

        program.emplace(
            context,
            context.getDevice(),
            options.kernel_path,
            precompute,
            options.build_options,
            options.debug);

        kernels.emplace(program->getProgram(), context.getQueue());

        const std::vector<std::string> kernelNames = {
            "kernel_sub2",
            "kernel_sub1",
            "kernel_carry",
            "kernel_carry_2",
            "kernel_inverse_ntt_radix4_mm",
            "kernel_ntt_radix4_last_m1_n4",
            "kernel_ntt_radix4_last_m1_n4_nosquare",
            "kernel_inverse_ntt_radix4_mm_last",
            "kernel_ntt_radix4_last_m1",
            "kernel_ntt_radix4_last_m1_nosquare",
            "kernel_ntt_radix4_mm_first",
            "kernel_ntt_radix4_mm_m8",
            "kernel_ntt_radix4_mm_m4",
            "kernel_ntt_radix4_mm_m2",
            "kernel_ntt_radix4_mm_m16",
            "kernel_ntt_radix4_mm_m32",
            "kernel_inverse_ntt_radix4_m1",
            "kernel_inverse_ntt_radix4_m1_n4",
            "kernel_ntt_radix4_inverse_mm_2steps",
            "kernel_ntt_radix4_inverse_mm_2steps_last",
            "kernel_ntt_radix4_mm_2steps",
            "kernel_ntt_radix4_mm_2steps_first",
            "kernel_ntt_radix2_square_radix2",
            "kernel_ntt_radix4_radix2_square_radix2_radix4",
            "kernel_ntt_radix4_square_radix4",
            "kernel_pointwise_mul",
            "kernel_ntt_radix2",
            "kernel_res64_display",
            "kernel_ntt_radix5_mm_first",
            "kernel_ntt_inverse_radix5_mm_last",
            "check_equal"
        };

        for (const auto& name : kernelNames)
            kernels->createKernel(name);

        nttEngine.emplace(
            context,
            *kernels,
            *buffers,
            precompute,
            options.debug);
    }
    catch (...) {
        nttEngine.reset();
        kernels.reset();
        program.reset();
        buffers.reset();
        throw;
    }
}


App::App(int argc, char** argv)
  : argc_(argc)
  , argv_(argv)
  ,options([&]{
    std::vector<std::string> merged;
    merged.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-config") == 0 && i+1 < argc) {
            auto cfg_args = parseConfigFile(argv[i+1]);
            merged.insert(merged.end(), cfg_args.begin(), cfg_args.end());
            i++;
        } else {
            merged.emplace_back(argv[i]);
        }
    }

    const bool cliForceLlUnsafe = std::find(merged.begin() + 1, merged.end(), "-llunsafe") != merged.end();

    std::vector<char*> c_argv;
    for (auto& s : merged)
        c_argv.push_back(const_cast<char*>(s.c_str()));
    c_argv.push_back(nullptr);
    auto o = io::CliParser::parse(static_cast<int>(merged.size()), c_argv.data());
    const std::string cliMode = o.mode;
    if (o.submit || !o.password.empty()) {
        std::cerr << "Warning: PrimeNet submission is not available in this build; "
                     "-submit and -password are ignored. Submit results manually." << std::endl;
    }

    io::WorktodoParser wp{o.worktodo_path};
    // An explicit Gaussian-Mersenne command is self-contained and must not be
    // replaced by an unrelated default worktodo entry.
    std::optional<io::WorktodoEntry> worktodo_entry;
    if (!o.gaussian_mersenne) worktodo_entry = wp.parse();
    if (auto e = worktodo_entry) {
        o.exponent     = e->exponent;
        if (o.wagstaff) {
            // The command line doubled its own exponent when it parsed
            // -wagstaff; the worktodo exponent replaces it, so double that one
            // too instead of testing (2^(p/2)+1)/3.
            if (const uint64_t wagstaffExponent = io::wagstaffExponentForEntry(*e)) {
                // The doubled exponent is subject to the same limit the command line applies to
                // -wagstaff after doubling.
                if (const std::string limitError = io::exponentLimitError(wagstaffExponent, true);
                    !limitError.empty()) {
                    std::cerr << limitError << " (" << e->rawLine << ")" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                o.exponent = wagstaffExponent;
            } else {
                std::cerr << "Warning: -wagstaff only applies to PRP worktodo entries without "
                             "known factors; ignoring it for: " << e->rawLine << std::endl;
                io::dropWagstaff(o);
            }
        }
        if (e->gaussianMersenne) {
            o.gaussian_mersenne = true;
            o.gm_prp_only = e->gmPrpOnly;
            o.gm_family = e->gmFamily;
            o.gm_pipeline = e->gmPipeline;
            o.gm_pipeline_proth = e->gmPipelineProth;
            o.mode = e->gmPipeline ? "gm-chain"
                   : (e->ecmTest ? "gm-ecm"
                   : (e->pm1Test ? "gm-pm1"
                   : (e->gmPrpOnly ? "gm-prp" : "gm-proth")));
            o.proof = false;
            o.B1 = e->B1;
            o.B2 = e->B2;
            o.K = e->curves;
            o.nmax = e->curves;
            o.gm_base = e->gmBase;
            o.gm_sieve_limit = e->gmSieveLimit;
            o.gm_factor_chunk_bits = e->gmFactorChunkBits;
            o.gm_pipeline_ecm_B1 = e->gmEcmB1;
            o.gm_pipeline_ecm_B2 = e->gmEcmB2;
            o.gm_pipeline_ecm_curves = e->gmEcmCurves;
            o.sigma = e->sigma;
            if (e->ecmTest) o.compute_edwards = false;
        } else {
            o.mode = e->ecmTest ? "ecm"
                   : (e->prpTest ? "prp"
                   : (e->llTest ? ((e->doubleCheck && !cliForceLlUnsafe) ? "llsafe" : "ll")
                   : (e->pm1Test ? "pm1" : "")));
        }
        o.aid          = e->aid;
        o.doubleCheck  = e->doubleCheck;

        // Prime95/PrimeNet DoubleCheck= entries are Lucas-Lehmer
        // double-check work. By default, route them to the checked
        // LL-SAFE GPU path (same as CLI -ll). If the user explicitly
        // adds -llunsafe, honor that CLI override and run the faster
        // unsafe LL path. Never attempt PRP proof generation for LL work.
        if (e->llTest) {
            o.proof = false;
            if (e->doubleCheck) {
                if (cliForceLlUnsafe) {
                    std::cout << "DoubleCheck worktodo + -llunsafe -> LL-UNSAFE GPU path, no PRP proof generation.\n";
                } else {
                    std::cout << "DoubleCheck worktodo -> LL-SAFE GPU path (Gerbicz-Li), no PRP proof generation.\n";
                }
            }
        }
        o.sieveDepth   = e->sieveDepth;
        o.knownFactors = e->knownFactors;
        o.knownFactors_start.assign(e->knownFactors.begin(), e->knownFactors.end());


        if (e->pm1Test) {
            o.B1 = e->B1;
            o.B2 = e->B2;
            o.sieveDepth = e->sieveDepth;
            o.B2Start = e->B2Start;
        }

        if (e->ecmTest) {
            o.B1  = e->B1;
            o.B2  = e->B2;
            o.nmax = e->curves;
            o.K    = e->curves;
        }

        activeWorktodoRawLine_ = e->rawLine;
        hasWorktodoEntry_ = true;
    }


    if (!hasWorktodoEntry_ && o.exponent == 0) {
    std::cerr << "Error: no valid entry in " 
                << o.worktodo_path
                << " and no exponent provided on the command line\n";


    if (!o.gui) {
        // The prompt answer replaces the (absent) command-line exponent after CliParser has
        // doubled it for -wagstaff and checked the limits, so do both here as well.
        const uint64_t answer = askExponentInteractively();
        o.exponent = o.wagstaff ? 2 * answer : answer;
        if (const std::string limitError = io::exponentLimitError(o.exponent, o.wagstaff);
            !limitError.empty()) {
            std::cerr << limitError << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    o.mode = "prp";
    //std::exit(-1);
    }
    // The mode may have come from a worktodo LL entry (Test=), which the command-line check in main.cpp
    // cannot see: apply the same rule here.
    if (const std::string why = core::legacyLlRejection(o.mode, o.marin, o.allow_unvalidated_legacy_ll); !why.empty()) {
        throw std::runtime_error(why + " (worktodo LL entry)");
    }
    // A command-line LL run was already warned about in main.cpp; only a mode taken from worktodo is new.
    if (cliMode != "ll") {
        if (const std::string warn = core::legacyLlWarning(o.mode, o.marin, o.allow_unvalidated_legacy_ll); !warn.empty()) {
            std::cerr << warn << std::endl;
        }
    }
    engine::gpu_workload workload = engine::gpu_workload::generic;
    if (o.mode == "prp" || o.mode == "gm-proth" || o.mode == "gm-prp") workload = engine::gpu_workload::prp;
    else if (o.mode == "gm-pm1" || o.mode == "gm-chain") workload = engine::gpu_workload::pm1;
    else if (o.mode == "gm-ecm" || o.mode == "gm-ecm-special32") workload = engine::gpu_workload::ecm;
    else if (o.mode == "ll" || o.mode == "llsafe" || o.mode == "llsafe2") workload = engine::gpu_workload::ll;
    else if (o.mode == "pm1") {
        if (o.pm1_ultralowmem) workload = engine::gpu_workload::pm1_ultralowmem;
        else if (o.pm1_lowmem) workload = engine::gpu_workload::pm1_lowmem;
        else workload = engine::gpu_workload::pm1;
    }
    else if (o.mode == "ecm") workload = engine::gpu_workload::ecm;

    // Select Aevum plans by workload, not by transform length alone.  PRP,
    // LL, P-1 Stage 1 and ECM use different operation mixes and therefore may
    // prefer different FFT323161 geometries.  Explicit -aevum-fft requests are
    // never rewritten.
    if (o.aevum_fft_spec.empty()) {
#if defined(__APPLE__)
        if (workload == engine::gpu_workload::prp ||
            workload == engine::gpu_workload::ll) {
            o.aevum_fft_spec = "pow2:auto";
            if (o.aevum) {
                std::cout << "[Backend Apple Aevum] PRP/LL uses staged stock Type1 FFT3161; Type4/PFA remains disabled.\n";
            }
        }
#else
        const char* plan_override = nullptr;
        switch (workload) {
            case engine::gpu_workload::prp:
                plan_override = std::getenv("PRMERS_AEVUM_PRP_FFT");
                // Empty spec means plugin-native auto selection, identical to
                // standalone Aevum unless the user supplies an explicit override.
                o.aevum_fft_spec = plan_override && *plan_override
                    ? plan_override : "";
                break;
            case engine::gpu_workload::ll:
                plan_override = std::getenv("PRMERS_AEVUM_LL_FFT");
                o.aevum_fft_spec = plan_override && *plan_override
                    ? plan_override : "";
                break;
            case engine::gpu_workload::pm1:
            case engine::gpu_workload::pm1_lowmem:
                plan_override = std::getenv("PRMERS_AEVUM_PM1_FFT");
                o.aevum_fft_spec = plan_override && *plan_override
                    ? plan_override : "";
                break;
            case engine::gpu_workload::ecm:
                plan_override = std::getenv("PRMERS_AEVUM_ECM_FFT");
                o.aevum_fft_spec = plan_override && *plan_override
                    ? plan_override : "";
                break;
            default:
                break;
        }
        if (plan_override && *plan_override) {
            std::cout << "[Aevum workload plan override] "
                      << aevum_workload_name(workload) << "="
                      << o.aevum_fft_spec << "\n";
        }
#endif
    }
#if defined(__APPLE__)
    if (o.aevum_fft_spec.rfind("throughput:", 0) == 0) {
        std::cout << "[Backend Apple Aevum] workload throughput selector normalized to pow2:auto (stock Type1 FFT3161).\n";
        o.aevum_fft_spec = "pow2:auto";
    }
    // Apple ships the legacy OpenCL 1.2 stack. Keep Marin as the safe default
    // on macOS and make Aevum an explicit opt-in only.
    const engine::gpu_backend selected_backend = (!o.marin || o.force_engine_marin)
        ? engine::gpu_backend::marin
        : (o.aevum ? engine::gpu_backend::aevum : engine::gpu_backend::marin);
    if (o.marin && !o.aevum && !o.force_engine_marin) {
        std::cout << "[Backend macOS] Marin selected by platform default; use -aevum to opt in.\n";
    }
#else
    const engine::gpu_backend selected_backend = (!o.marin || o.force_engine_marin)
        ? engine::gpu_backend::marin
        : (o.aevum ? engine::gpu_backend::aevum : engine::gpu_backend::auto_select);
#endif
    engine::configure_gpu_backend(selected_backend, o.aevum_fft_spec, workload);
    return o;
  }())
  , context(options.device_id,
        static_cast<std::size_t>(options.enqueue_max),
        options.cl_queue_throttle_active,
        options.debug /*, options.marin */)
  , precompute(options.exponent)
  , backupManager(
        context.getQueue(),
        static_cast<unsigned>(options.backup_interval),
        precompute.getN(),
        options.save_path,
        options.exponent,
        options.mode,
        options.B1,
        options.B2,
        options.wagstaff,
        options.marin
    )
  , proofManager(
        options.exponent,
        static_cast<int>([this]() -> uint32_t {
            if (!this->options.proof) return 0u;
            if (this->options.manual_proofPower) return this->options.proofPower;
            return ProofSet::bestPower(this->options.exponent);
        }()),
        context.getQueue(),
        precompute.getN(),
        precompute.getDigitWidth(),
        options.knownFactors,
        options.save_path
    ),
    proofManagerMarin(
        options.exponent,
        static_cast<int>([this]() -> uint32_t {
            if (!this->options.proof) return 0u;
            if (this->options.manual_proofPower) return this->options.proofPower;
            return ProofSet::bestPower(this->options.exponent);
        }()),
        context.getQueue(),
        precompute.getN(),
        precompute.getDigitWidth(),
        options.knownFactors,
        options.save_path
    )
  , spinner()
  , logger(options.output_path)
  , timer()
  , timer2()
{
    worktodoParser_ = std::make_unique<io::WorktodoParser>(options.worktodo_path);
    // The worktodo entry was parsed once in the options initializer.  Do not
    // parse it again here, otherwise startup diagnostics are printed twice.
    
    context.computeOptimalSizes(
        precompute.getN(),
        precompute.getDigitWidth(),
        options.exponent,
        options.debug,
        options.max_local_size1,
        options.max_local_size5
    );
    if ((!options.aevum && !options.aevum_auto) || options.force_engine_marin) {
        ensureProofGpuBackend();
    }

    // SIGINT, SIGTERM and SIGHUP (and the Windows console events) all stop the run the same way.
    core::algo::install_stop_handlers();
}



int App::exportResumeFromMersFile(const std::string& mersPath,
                                  const std::string& savePath)
{
    std::vector<uint64_t> v(precompute.getN(), 0ULL);
    if (!read_mers_file(mersPath, v)) return -1;

    std::string fname = std::filesystem::path(mersPath).filename().string();
    uint32_t p = 0;
    uint64_t B1 = 0;
    if (!io::parseMersFileName(fname, p, B1))
        return std::cerr << "Invalid filename format, expected <p>pm<B1>.mers\n", -1;
    const size_t pos_dot = fname.rfind('.');

    mpz_class Mp = (mpz_class(1) << p) - 1;
    mpz_class X  = util::vectToMpz(v, precompute.getDigitWidth(), Mp);

    std::string out = savePath.empty()
        ? (fname.substr(0, pos_dot) + ".save")
        : savePath;

    writeEcmResumeLine(out, B1, p, X);
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "GMP ECM file written to: " << out << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    return 0;
}



int App::convertEcmResumeToPrime95(const std::string& ecmPath, const std::string& outPath, const std::string& date_start, const std::string& date_end){
    std::string txt;
    if(!read_text_file(ecmPath, txt)) return -1;
    uint64_t B1=0; uint32_t p=0; std::string hexX;
    if(!parse_ecm_resume_line(txt, B1, p, hexX)) return -2;
    std::vector<uint8_t> data = hex_to_bytes_reversed_pad8(hexX);
    std::string out = outPath;
    if(out.empty()){
        std::ostringstream name; name << 'm' << std::setw(7) << std::setfill('0') << p;
        std::filesystem::path outp = std::filesystem::path(ecmPath).parent_path() / name.str();
        out = outp.string();
    }
    if(!write_prime95_s1_from_bytes(out, p, B1, data, date_start, date_end)) return -3;
    std::cout << "Prime95 S1 file written to: " << out << std::endl;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "Prime95 S1 file written to: " << out << std::endl;
        guiServer_->appendLog(oss.str());
    }
    return 0;
}

static volatile sig_atomic_t prmers_bench_stop = 0;
// Also a Stop for the GUI idle loop that follows, and for a restart already committed.
static void prmers_bench_sigint(int) { prmers_bench_stop = 1; core::stop_or_exit(); }

static std::string fmt_dhms(double s) {
    if (s < 0) s = 0;
    uint64_t t = (uint64_t)(s + 0.5);
    uint64_t d = t / 86400;
    uint64_t h = (t % 86400) / 3600;
    uint64_t m = (t % 3600) / 60;
    uint64_t sec = t % 60;
    std::ostringstream o;
    if (d) o << d << "d " << h << "h " << std::setw(2) << std::setfill('0') << m << "m " << std::setw(2) << std::setfill('0') << sec << "s";
    else o << h << "h " << std::setw(2) << std::setfill('0') << m << "m " << std::setw(2) << std::setfill('0') << sec << "s";
    return o.str();
}

uint32_t transformsize_custom(uint64_t exponent) {
    uint64_t log_n = 0;
    uint64_t w = 0;
    do {
        ++log_n;
        w = exponent >> log_n;
    } while ((w + 1) * 2 + log_n >= 63);
    uint64_t n2 = uint64_t(1) << log_n;
    if (n2 >= 128) {
        uint64_t n5 = (n2 >> 3) * 5u;
        if (n5 >= 80) {
            uint64_t w5 = exponent / n5;
            long double cost5 = std::log2((long double)n5) + 2.0L * (w5 + 1);
            if (cost5 < 64.0L) return (uint32_t)n5;
        }
    }
    if (exponent > 1207959503) n2 = (n2 / 4) * 5;
    return (uint32_t)n2;
}



int App::runGpuBenchmarkMarin() {
    if (guiServer_) {
        std::ostringstream oss;
        oss << "BENCH";
        guiServer_->setStatus(oss.str());
    }
    auto fmt_pct = [&](double x){ std::ostringstream o; o<<std::fixed<<std::setprecision(1)<<x*100.0<<"%"; return o.str(); };

    std::string gpu_name = "Unknown";
    std::string gpu_vendor = "Unknown";
    std::string driver_ver = "Unknown";
    uint32_t cu = 0;
    uint64_t vram = 0;
    uint64_t lmem = 0;
    std::string fp64 = "Unknown";
    #ifdef CL_VERSION_1_0
    {
        // Enumerate only GPU devices
        cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
        std::vector<cl_platform_id> plats(np); if (np) clGetPlatformIDs(np, plats.data(), nullptr);
        std::vector<cl_device_id> devs;
        for (auto pid : plats) {
            cl_uint nd = 0; clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
            if (!nd) continue;
            size_t old = devs.size(); devs.resize(old + nd);
            clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, nd, devs.data() + old, nullptr);
        }
        size_t idx = (size_t)options.device_id;
        if (!devs.empty() && idx < devs.size()) {
            cl_device_id d = devs[idx];
            auto get_str = [&](cl_device_info p) {
                size_t sz = 0; clGetDeviceInfo(d, p, 0, nullptr, &sz);
                std::string s(sz, '\0'); if (sz) clGetDeviceInfo(d, p, sz, s.data(), nullptr);
                if (!s.empty() && s.back() == '\0') s.pop_back();
                return s;
            };
            gpu_name = get_str(CL_DEVICE_NAME);
            gpu_vendor = get_str(CL_DEVICE_VENDOR);
            driver_ver = get_str(CL_DRIVER_VERSION);
            cl_uint u = 0; clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(u), &u, nullptr); cu = (uint32_t)u;
            cl_ulong gm = 0; clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gm), &gm, nullptr); vram = (uint64_t)gm;
            cl_ulong lm = 0; clGetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lm), &lm, nullptr); lmem = (uint64_t)lm;
            cl_device_fp_config df = 0; clGetDeviceInfo(d, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(df), &df, nullptr); fp64 = (df ? "Yes" : "No");
        }
    }
    #endif
    std::cout << "GPU: " << gpu_vendor << " " << gpu_name << " | Driver: " << driver_ver << " | CUs: " << cu << " | VRAM: " << (vram / (1024*1024)) << " MB | LocalMem: " << (lmem / 1024) << " KB | FP64: " << fp64 << "\n";
    if (guiServer_) {
                                        std::ostringstream oss;
                                        oss  << "GPU: " << gpu_vendor << " " << gpu_name << " | Driver: " << driver_ver << " | CUs: " << cu << " | VRAM: " << (vram / (1024*1024)) << " MB | LocalMem: " << (lmem / 1024) << " KB | FP64: " << fp64 << "\n";
                            guiServer_->appendLog(oss.str());
    }
    std::vector<uint32_t> exps = {
        127u, 1279u, 2203u, 9941u, 44497u, 756839u, 3021377u, 37156667u, 57885161u, 77232917u, 82589933u, 136279841u,
        146410013u, 161051017u, 177156127u, 180000017u, 200000033u, 220000013u, 250000013u, 280000027u, 300000007u,
        320000077u, 340000019u, 360000019u, 400000009u, 500000003u, 600000001u
    };

    struct Task { uint32_t ts; uint32_t p; };
    std::vector<Task> tasks;
    tasks.reserve(exps.size());
    for (auto p : exps) tasks.push_back({transformsize_custom(p), p});

    struct Row { uint32_t ts; uint32_t p; double ips; double eta_prp; };
    std::vector<Row> rows;

    auto print_live = [&](size_t i, size_t n, uint32_t ts, uint32_t p, double frac, double ips_live, double eta_all){
        std::ostringstream o;
        o << "\r[" << (i+1) << "/" << n << "] TS=" << std::setw(9) << ts << " p=" << std::setw(10) << p
          << " " << std::setw(6) << fmt_pct((i + frac) / n)
          << " ips=" << std::fixed << std::setprecision(2) << std::setw(10) << ips_live
          << " ETA=" << fmt_dhms(eta_all) << "    ";
        std::cout << o.str() << std::flush;
        if (guiServer_) {
                                        std::ostringstream oss;
                                        oss  << "\r[" << (i+1) << "/" << n << "] TS=" << std::setw(9) << ts << " p=" << std::setw(10) << p
          << " " << std::setw(6) << fmt_pct((i + frac) / n)
          << " ips=" << std::fixed << std::setprecision(2) << std::setw(10) << ips_live
          << " ETA=" << fmt_dhms(eta_all) << "    ";

                            guiServer_->appendLog(oss.str());
        }
    };

    auto old_handler = std::signal(SIGINT, prmers_bench_sigint);
#ifdef SIGTERM
    auto old_term_handler = std::signal(SIGTERM, prmers_bench_sigint);
#endif
#ifdef SIGHUP
    auto old_hup_handler = std::signal(SIGHUP, prmers_bench_sigint);
    if (old_hup_handler == SIG_IGN || core::algo::sighup_inherited_ignored()) std::signal(SIGHUP, SIG_IGN);   // nohup
#endif
    const size_t R0 = 0, R1 = 1;
    double sum_time = 0.0;

    for (size_t ti = 0; ti < tasks.size(); ++ti) {
        if (prmers_bench_stop || core::algo::stop_requested_any()) break;
        uint32_t p = tasks[ti].p;
        engine* eng = nullptr;
        try { eng = engine::create_gpu(p, static_cast<size_t>(6), static_cast<size_t>(options.device_id), false/*, options.chunk256*/); } catch (...) { eng = nullptr; }
        if (!eng) continue;
        eng->set(R1, 1);
        eng->set(R0, 3);

        uint32_t warm = 96;
        for (uint32_t i = 0; i < warm && !prmers_bench_stop && !core::algo::stop_requested_any(); ++i) eng->square_mul(R0);

        uint32_t ts =  eng->get_size();
        double target = (ts >= 33554432u) ? 10.0 : (ts >= 8388608u ? 8.0 : (ts >= 2621440u ? 6.0 : 5.0));

        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t cnt = 0;
        double last_update = 0.0;

        for (;;) {
            if (prmers_bench_stop || core::algo::stop_requested_any()) break;
            eng->square_mul(R0);
            ++cnt;
            auto t1 = std::chrono::high_resolution_clock::now();
            double e = std::chrono::duration<double>(t1 - t0).count();
            double frac = std::min(1.0, e / target);
            double per_task_est = (ti==0 && frac>0.0) ? (e/frac) : ((sum_time + e) / (ti + frac));
            double eta_all = per_task_est * ((tasks.size() - (ti + 1)) + (1.0 - frac));
            double ips_live = cnt / std::max(1e-9, e);
            if (e - last_update >= 0.2 || frac >= 1.0) { print_live(ti, tasks.size(), ts, p, frac, ips_live, eta_all); last_update = e; }
            if (e >= target) break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        sum_time += elapsed;

        if (!prmers_bench_stop && !core::algo::stop_requested_any()) {
            double ips = cnt / std::max(1e-9, elapsed);
            double eta_prp = (double)p / std::max(1e-9, ips);
            rows.push_back({ts, p, ips, eta_prp});
        }

        delete eng;
        if (prmers_bench_stop || core::algo::stop_requested_any()) break;
    }

    std::signal(SIGINT, old_handler);
#ifdef SIGTERM
    std::signal(SIGTERM, old_term_handler);
#endif
#ifdef SIGHUP
    std::signal(SIGHUP, old_hup_handler);
#endif
    std::cout << "\n";
    std::cout << "Transform  Exponent      Iter/s       PRP_ETA\n";
    if (guiServer_) {
                                        std::ostringstream oss;
                                        oss  << "Transform  Exponent      Iter/s       PRP_ETA";

                            guiServer_->appendLog(oss.str());
        }
    for (auto &r : rows) {
        std::cout << std::setw(9) << r.ts << "  " << std::setw(10) << r.p
                  << "  " << std::fixed << std::setprecision(2) << std::setw(10) << r.ips
                  << "  " << std::setw(14) << fmt_dhms(r.eta_prp) << "\n";
        if (guiServer_) {
                                        std::ostringstream oss;
                                        oss  << std::setw(9) << r.ts << "  " << std::setw(10) << r.p
                  << "  " << std::fixed << std::setprecision(2) << std::setw(10) << r.ips
                  << "  " << std::setw(14) << fmt_dhms(r.eta_prp);

                            guiServer_->appendLog(oss.str());
        }
    }
    double prmers_score_val = 0.0;
    if (!rows.empty()) {
        const double base_ts = 8388608.0;
        double wsum = 0.0, logsum = 0.0;
        for (const auto& r : rows) {
            double ips_base = (r.ips * (double)r.ts) / base_ts;
            double w = std::log2((double)std::max<uint32_t>(r.ts, 2));
            if (r.p >= 1000000u) w *= 1.25;
            logsum += std::log(std::max(ips_base, 1e-9)) * w;
            wsum += w;
        }
        double gmean = std::exp(logsum / std::max(1e-9, wsum));
        prmers_score_val = 100.0 * (gmean / 400.0);
        if (prmers_score_val < 0.0) prmers_score_val = 0.0;
        if (prmers_score_val > 100.0) prmers_score_val = 100.0;
    }
    std::cout << "\nPRMERS_SCORE|" << std::fixed << std::setprecision(2) << prmers_score_val << "/100\n";
    std::cout << gpu_vendor << " " << gpu_name << " has a PRMERS SCORE OF " << std::fixed << std::setprecision(2) << prmers_score_val << "/100\n";
    if (guiServer_) {
        std::ostringstream oss;
        oss  << "\nPRMERS_SCORE|" << std::fixed << std::setprecision(2) << prmers_score_val << "/100\n";

        guiServer_->appendLog(oss.str());
    }
    if (guiServer_) {
        std::ostringstream oss;
        oss << gpu_vendor << " " << gpu_name << " has a PRMERS SCORE OF " << std::fixed << std::setprecision(2) << prmers_score_val << "/100\n";

        guiServer_->appendLog(oss.str());
    }

    return 0;
}

namespace {
  // The stop flag and the GUI "Append & Run" pending flag live in core::g_stop_restart_gate, which
  // orders a Stop against a restart (see core/StopRestartGate.hpp). The append handler (HTTP worker
  // thread) only marks the entry pending; it never interrupts a running test: the running mode picks
  // the line up when its entry finishes, and the idle loops in App::run() (main thread) start it when
  // nothing is running.
  bool stop_requested() noexcept { return core::g_stop_restart_gate.stopRequested(); }
  // Signal handler and the GUI's Stop. Async-signal-safe.
  void handle_signal(int) noexcept {
    core::algo::interrupted.store(true, std::memory_order_relaxed);
    core::stop_or_exit();
  }
}


#if defined(_WIN32)
#include <windows.h>
#include <csignal>
#include <cstdlib>
static void install_signal_handlers() { core::algo::install_stop_handlers(); }
#else
#include <signal.h>
static void install_signal_handlers() {
    // The same SIGINT / SIGTERM / SIGHUP handling as the CLI modes (core::algo::install_stop_handlers).
    core::algo::install_stop_handlers();
    struct sigaction ign; sigemptyset(&ign.sa_mask); ign.sa_flags = 0; ign.sa_handler = SIG_IGN;
#ifdef SIGPIPE
    sigaction(SIGPIPE, &ign, nullptr);
#endif
}
#endif



// Called from the main thread while the GUI is idle: if "Append & Run" added a line since the last
// check, restart so the new entry starts. Nothing is running at this point, so no work is lost.
// A Stop wins over the restart at every step: before the claim it makes the claim fail; after it,
// restart_self's commit fails (and restart_self returns); once committed, the Stop ends the process
// before the relaunch (core::stop_or_exit). The appended line stays in worktodo in every case.
static void gui_restart_for_appended_entry(const std::shared_ptr<ui::WebGuiServer>& gui, int argc, char** argv) {
    if (core::g_stop_restart_gate.claimAppendedEntry() != core::StopRestartGate::Claim::Restart) return;
    std::cout << "Starting appended worktodo entry" << std::endl;
    if (gui) {
        gui->appendLog("Starting appended worktodo entry");
        gui->stop();
    }
    restart_self(argc, argv);
    // Only reached when a Stop came first; the idle loop then ends.
    std::cout << "The appended entry stays in worktodo for the next start." << std::endl;
}

static bool file_non_empty(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return false;
    f.peek();
    return !f.eof();
}


int App::run() {

    //std::cout << "host : " << options.http_host << "\n";
    
    if (options.gui) {
        install_signal_handlers();
        if (options.http_port == 0) options.http_port = 3131;
        ui::WebGuiConfig cfg;
        cfg.port = options.http_port;
        cfg.bind_host = options.http_host;
        cfg.advertise_host = options.http_host;
       // std::cout << "host : " << cfg.bind_host << "\n";
        cfg.lanipv4 = options.ipv4;
        cfg.worktodo_path = options.worktodo_path;
        cfg.save_path = options.save_path.empty() ? "." : options.save_path;
        cfg.kernel_path = options.kernel_path;
        cfg.worktodo_line_ok = [](const std::string& line) { return io::WorktodoParser::isValidEntryLine(line); };

        cfg.config_path = options.config_path.empty() ? "./settings.cfg" : options.config_path;
        cfg.results_path = (std::filesystem::path(options.save_path.empty() ? "." : options.save_path) / "results.txt").string();
        static std::atomic<bool> gui_alive{true};
        auto submitFn = [this](const std::string& line){
            // Runs on the HTTP worker thread: only append. Restarting from here would kill the
            // running test without a checkpoint (and repeated requests would keep restarting it).
            if (!io::WorktodoParser::appendLine(this->options.worktodo_path, line)) {
                std::cerr << "Failed to append to " << this->options.worktodo_path << "\n";
                if (guiServer_) guiServer_->appendLog("Failed to append to " + this->options.worktodo_path);
                return;
            }
            std::cout << "worktodo appended; it starts after the current entry finishes (immediately if idle)\n";
            if (guiServer_) guiServer_->appendLog("worktodo appended; it starts after the current entry finishes (immediately if idle)");
            core::g_stop_restart_gate.markAppendPending();
        };
        auto stopFn = [&](){
            handle_signal(SIGINT);
            gui_alive = false;
        };
        guiServer_ = std::make_shared<ui::WebGuiServer>(cfg, submitFn, stopFn);
        ui::WebGuiServer::setInstance(guiServer_);
        const std::string gui_workload = options.mode == "pm1" ? "P-1"
                                       : options.mode == "ecm" ? "ECM"
                                       : (options.mode == "ll" || options.mode == "llsafe" || options.mode == "llsafe2") ? "LL"
                                       : (options.mode == "gm-proth" || options.mode == "gm-prp" || options.mode == "gm-pm1" || options.mode == "gm-ecm" || options.mode == "gm-ecm-special32") ? "Gaussian-Mersenne"
                                       : "PRP";
        if (!options.marin) {
            guiServer_->setBackendInfo("Internal PrMers NTT", "Internal NTT", gui_workload,
                                       "selected by the legacy -marin option");
        } else if (options.force_engine_marin) {
            guiServer_->setBackendInfo("Forced Marin", "Pending", gui_workload,
                                       "Marin engine will be created on first arithmetic operation");
        } else if (options.aevum) {
            guiServer_->setBackendInfo("Forced Aevum", "Pending", gui_workload,
                                       "Aevum engine will be created on first arithmetic operation; native PFA is automatic for PRP/LL", 0, 0,
                                       options.aevum_fft_spec);
        } else {
#if defined(__APPLE__)
            guiServer_->setBackendInfo("macOS default", "Marin", gui_workload,
                                       "Marin is the macOS default; use -aevum to opt in explicitly");
#else
            guiServer_->setBackendInfo("Auto", "Pending", gui_workload,
                                       "backend will be selected from workload and transform sizes");
#endif
        }
        if (!guiServer_->start()) {
            // No GUI: carry on headless if there is work, otherwise there is nothing left to wait for
            // (the idle loop below can only be ended through the GUI or a signal).
            std::cerr << "The GUI could not be started." << std::endl;
 if (!hasWorktodoEntry_ && options.exponent == 0 && !options.bench && options.filemers.empty()) {
                std::cerr << "Nothing to run and no GUI; exiting." << std::endl;
                return 1;
            }
            std::cerr << "Continuing without the GUI." << std::endl;
            guiServer_.reset();
            ui::WebGuiServer::setInstance(nullptr);
            options.gui = false;
        } else {
            std::cout << "GUI " << guiServer_->url() << std::endl;
        }
        const bool gui_loopback = !options.ipv4 &&
            (options.http_host.empty() || options.http_host == "localhost" ||
             options.http_host.rfind("127.", 0) == 0);
        if (guiServer_ && !gui_loopback) {
            std::cout << "Warning: the GUI is reachable from other machines on your network. "
                         "Anyone who can connect to it can control this PrMers instance." << std::endl;
        }
        // Nothing runnable: an empty worktodo, or one whose lines parse() skips (comments, unsupported or
        // malformed entries) with no exponent given. Running with exponent 0 would only fail; wait for
        // "Append & Run" instead. Only wait when the GUI actually started.
        if (guiServer_ && (!file_non_empty(cfg.worktodo_path) || (!hasWorktodoEntry_ && options.exponent == 0 &&
                                                       !options.bench && options.filemers.empty()))) {
            if (file_non_empty(cfg.worktodo_path)) {
                std::cout << "No runnable entry in " << cfg.worktodo_path << "; waiting for a new entry.\n";
                guiServer_->appendLog("No runnable entry in " + cfg.worktodo_path + "; waiting for a new entry.");
            }
            guiServer_->setStatus("Idle");
            while (!stop_requested() && gui_alive) {
                gui_restart_for_appended_entry(guiServer_, argc_, argv_);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            if (guiServer_) guiServer_->stop();
            return 0;
        }
    }

    // The historical one-register ultra-low-memory P-1 path is intentionally
    // Marin-only: it relies on Marin's fast3 operation and cannot be represented
    // by the generic engine::Reg API without adding a second register. Reject an
    // explicit Aevum request cleanly before allocating the Aevum plugin. Auto
    // mode is handled by the compatibility-aware policy and selects Marin.
    if (options.mode == "pm1" && options.pm1_ultralowmem && options.aevum) {
        std::cerr << "[Backend Compatibility] -pm1-ultralowmem is a Marin fast3-only "
                     "algorithm and cannot be forced to Aevum. Use automatic mode or "
                     "-engine-marin.\n";
        if (guiServer_) {
            guiServer_->setBackendInfo("Forced Aevum rejected", "Marin required", "P-1 ultra-low-memory",
                                       "one-register fast3-only algorithm");
            guiServer_->appendLog("[Backend Compatibility] Forced Aevum rejected for P-1 ultra-low-memory.");
        }
        return 2;
    }

    int rc = 1;
    bool ran = false;
    if (options.mode == "gm-chain") {
        // Run a complete conditional pipeline independently for each requested
        // Gaussian family.  This avoids repeating work on a family already
        // eliminated by an earlier phase and lets every phase use its native
        // Aevum/Marin workload policy.
        const std::string pipeline_mode = options.mode;
        const std::string requested_family = options.gm_family;
        const std::uint64_t pm1_B1 = options.B1;
        const std::uint64_t pm1_B2 = options.B2;
        const std::uint64_t ecm_B1 = options.gm_pipeline_ecm_B1;
        const std::uint64_t ecm_B2 = options.gm_pipeline_ecm_B2;
        const std::uint64_t ecm_curves = options.gm_pipeline_ecm_curves;
        const std::uint64_t sieve_limit = options.gm_sieve_limit;
        const bool saved_prp_only = options.gm_prp_only;

        std::cout << "[GMCHAIN] family=" << requested_family
                  << " P-1 B1=" << pm1_B1 << " B2=" << pm1_B2;
        if (ecm_curves != 0)
            std::cout << ", then ECM B1=" << ecm_B1 << " B2=" << ecm_B2
                      << " curves=" << ecm_curves;
        if (options.gm_pipeline_proth)
            std::cout << ", then GM Proth / GQ Fermat PRP if no factor is found.\n";
        else
            std::cout << ", then stop after factoring.\n";

        // Completed phases/families are recorded so a restarted chain resumes
        // after them instead of repeating the whole pipeline. The job key is the
        // raw worktodo line (or the chain parameters for a command-line chain).
        const std::filesystem::path chain_dir =
            options.save_path.empty() ? std::filesystem::path(".")
                                      : std::filesystem::path(options.save_path);
        std::string chain_key = activeWorktodoRawLine_;
        if (chain_key.empty()) {
            chain_key = "gmchain|p=" + std::to_string(options.exponent) +
                        "|family=" + requested_family +
                        "|pm1=" + std::to_string(pm1_B1) + "," + std::to_string(pm1_B2) +
                        "|ecm=" + std::to_string(ecm_B1) + "," + std::to_string(ecm_B2) +
                        "," + std::to_string(ecm_curves) +
                        "|proth=" + (options.gm_pipeline_proth ? "1" : "0") +
                        "|sieve=" + std::to_string(sieve_limit);
        }
        core::gm_chain_progress::Progress chain_progress(
            chain_dir / ("gm_chain_p" + std::to_string(options.exponent) + "_phases.done"),
            chain_key);
        namespace gcp = core::gm_chain_progress;
        // A record that exists but cannot be used is reported once, here, rather
        // than silently redoing finished work.
        if (const std::string chain_notice = chain_progress.notice(); !chain_notice.empty()) {
            std::cerr << chain_notice << std::endl;
            if (guiServer_) guiServer_->appendLog(chain_notice);
        }

        auto run_family_pipeline = [&](const std::string& family) -> int {
            if (const auto finished = chain_progress.family_rc(family)) {
                std::cout << "[GMCHAIN] " << family << " already completed (rc=" << *finished
                          << "); skipping. Delete " << chain_progress.file().string()
                          << " to start the chain over.\n";
                return *finished;
            }
            options.gm_family = family;
            options.mode = "gm-pm1";
            options.B1 = pm1_B1;
            options.B2 = pm1_B2;
            options.K = 0;
            options.nmax = 0;
            options.gm_sieve_limit = sieve_limit;
            options.gm_prp_only = false;
            int family_rc = 1;
            if (chain_progress.has(gcp::phase_token(family, "pm1"))) {
                std::cout << "[GMCHAIN] " << family << " P-1 already completed; skipping.\n";
            } else {
                configure_gaussian_phase_backend(options, engine::gpu_workload::pm1, "P-1");
                family_rc = runGaussianMersennePM1();
                if (!interrupted && family_rc == 1) chain_progress.mark(gcp::phase_token(family, "pm1"));
            }

            if (!interrupted && family_rc == 1 && ecm_curves != 0) {
                if (chain_progress.has(gcp::phase_token(family, "ecm"))) {
                    std::cout << "[GMCHAIN] " << family << " ECM already completed; skipping.\n";
                } else {
                    options.mode = "gm-ecm";
                    options.B1 = ecm_B1;
                    options.B2 = ecm_B2;
                    options.K = ecm_curves;
                    options.nmax = ecm_curves;
                    options.gm_sieve_limit = 0; // already performed by P-1
                    configure_gaussian_phase_backend(options, engine::gpu_workload::ecm, "ECM");
                    family_rc = runGaussianMersenneECM();
                    if (!interrupted && family_rc == 1) chain_progress.mark(gcp::phase_token(family, "ecm"));
                }
            }

            if (!interrupted && family_rc == 1 && options.gm_pipeline_proth) {
                options.mode = "gm-proth";
                options.gm_prp_only = false;
                options.B1 = 0;
                options.B2 = 0;
                options.K = 0;
                options.nmax = 0;
                options.gm_sieve_limit = 0; // already performed by P-1
                configure_gaussian_phase_backend(options, engine::gpu_workload::prp,
                                                  family == "GQ" ? "GQ Fermat PRP" : "GM Proth");
                family_rc = runGaussianMersenne();
            }
            if (!interrupted && family_rc != 2) chain_progress.mark(gcp::done_token(family, family_rc));
            return family_rc;
        };

        if (requested_family == "BOTH") {
            const int gm_rc = run_family_pipeline("GM");
            if (interrupted) {
                rc = gm_rc;
            } else {
                const int gq_rc = run_family_pipeline("GQ");
                rc = (gm_rc == 2 || gq_rc == 2) ? 2
                   : (gm_rc == 0 && gq_rc == 0) ? 0 : 1;
            }
        } else {
            rc = run_family_pipeline(requested_family);
        }
        ran = true;
        if (!interrupted && rc != 2) chain_progress.clear();

        options.mode = pipeline_mode;
        options.gm_family = requested_family;
        options.gm_prp_only = saved_prp_only;
        options.B1 = pm1_B1;
        options.B2 = pm1_B2;
        options.K = ecm_curves;
        options.nmax = ecm_curves;
        options.gm_sieve_limit = sieve_limit;
    }
    if (options.mode == "gm-proth" || options.mode == "gm-prp") {
        rc = runGaussianMersenne();
        ran = true;
    }
    if (options.mode == "gm-pm1") {
        rc = runGaussianMersennePM1();
        ran = true;
    }
    if (options.mode == "gm-ecm" || options.mode == "gm-ecm-special32" || options.mode == "gm-ecm-special4096") {
        rc = runGaussianMersenneECM();
        ran = true;
    }
    if(options.mode == "ecm"){
        if(options.compute_edwards){
            rc = runECMMarinTwistedEdwards();
        }
        else{
            rc = runECMMarin();
        }
        ran = true;
    }
    if(options.mode == "memtest"){
        rc = runMemtestOpenCL();
        ran = true;
    }

    if (options.mode == "llsafe") {
        rc = runLlSafeMarin();
        ran = true;
    } 
    if (options.mode == "llsafe2") {
        rc = runLlSafeMarinDoubling();
        ran = true;
    }
    else if (options.mode == "pm1" && options.marin /*&& options.B2 <= 0*/) {
        if (options.exponent > 89) {
            int rc_local = 0;
            bool ran_local = false;

            std::ostringstream s1; s1 << "pm1_m_" << options.exponent << ".ckpt";
            std::ostringstream s2; s2 << "pm1_s2_m_" << options.exponent << ".ckpt";
            const std::string s1f = s1.str();
            const std::string s2f = s2.str();

            bool haveS1 = File(s1f).exists() || File(s1f + ".old").exists();
            bool haveS2 = File(s2f).exists() || File(s2f + ".old").exists() || File(s2f + ".new").exists();
            if(options.s3only){
                std::ostringstream msg;
                msg << "S3 only requested " 
                    << "-> jumping to runPM1Stage3Marin()";
                std::cout << msg.str() << std::endl;
                if (guiServer_) { guiServer_->appendLog(msg.str()); guiServer_->setStatus("Stage 3 only"); }

                rc_local = runPM1Stage3Marin();
                ran_local = true;
            }
            if(options.s4only){
                std::ostringstream msg;
                msg << "S4 only requested " 
                    << "-> jumping to runPM1Stage4Marin()";
                std::cout << msg.str() << std::endl;
                if (guiServer_) { guiServer_->appendLog(msg.str()); guiServer_->setStatus("Stage 4 only"); }

                rc_local = runPM1Stage4Marin();
                ran_local = true;
            }
            else if(options.torus){
                //options.torus = runPM1Stage1SLnTorusMarin();
                std::ostringstream msg;
                msg << "TORUS only requested " 
                    << "-> jumping to runPM1Stage1SLnTorusMarin()";
                std::cout << msg.str() << std::endl;
                if (guiServer_) { guiServer_->appendLog(msg.str()); guiServer_->setStatus("Stage TORUS only"); }

                rc_local = runPM1Stage1SLnTorusMarin();
                ran_local = true;
            }
            else if (options.pm1_s2_resume2reg && options.B2 > options.B1 && options.nmax == 0 && options.K == 0) {
                std::ostringstream msg;
                msg << "P-1 Stage 2 resume2reg requested "
                    << "-> jumping directly to runPM1Stage2Marin()";
                std::cout << msg.str() << std::endl;
                if (guiServer_) { guiServer_->appendLog(msg.str()); guiServer_->setStatus("P-1 Stage 2 resume2reg"); }

                rc_local = pm1Stage2ExitCode(runPM1Stage2Marin());
                ran_local = true;
            }
            else {
                // A stage-2 checkpoint is resumed by runPM1Marin: it reads the
                // finished stage-1 checkpoint, runs stage 2 from where it
                // stopped and then does the cleanup (stage-1 checkpoint, worktodo
                // line, next entry).  Calling stage 2 directly skipped all that.
                std::string msg;
                if (haveS1 || haveS2) {
                    std::ostringstream oss;
                    oss << "Detected P-1 checkpoint(s): "
                        << (haveS2 ? "[Stage 2] " : "")
                        << (haveS1 ? "[Stage 1] " : "")
                        << "-> running Stage 1 with " << engine::configured_gpu_backend_name()
                        << (haveS2 ? " (it resumes Stage 2 from its checkpoint)" : "");
                    msg = oss.str();
                } else {
                    msg = std::string("No P-1 checkpoints found -> running Stage 1 with ") + engine::configured_gpu_backend_name();
                }
                std::cout << msg << std::endl;
                if (guiServer_) { guiServer_->appendLog(msg); guiServer_->setStatus("Running P-1 Stage 1"); }

                rc_local = runPM1Marin();
                ran_local = true;
            }

            rc = rc_local;
            ran = ran_local;
        }
        else {
            std::cout << "P-1 factoring (stage 1) need exponent > 89" << std::endl;
        }
    } else if (options.exportmers) {
        rc = exportResumeFromMersFile(options.filemers, "");
        ran = true;
    } else if (options.bench) {
        rc = runGpuBenchmarkMarin();
        ran = true;
    } else if ((options.mode == "prp" || options.mode == "ll") && options.marin) {
        rc = runPrpOrLlMarin();
        ran = true;
    } else if (options.mode == "prp" || options.mode == "ll") {
        rc = runPrpOrLl();
        ran = true;
    } else if (options.mode == "pm1") {
        if (options.exponent >= 241) {
            rc = runPM1();
            ran = true;
        } else {
            std::cout << "P-1 factoring (stage 1) need exponent >= 241" << std::endl;
        }
    }
    // Native Gaussian-Mersenne worktodo entries are handled here because the
    // dedicated modes intentionally remain isolated from the historical
    // Prime95-compatible mode implementations. A completed task (factor,
    // no-factor, prime or composite) is archived, then PrMers restarts on the
    // next non-comment line. Interrupted/error runs keep the current line.
    if (hasWorktodoEntry_ && options.gaussian_mersenne && !interrupted &&
        (rc == 0 || rc == 1)) {
        if (worktodoParser_ && worktodoParser_->removeProcessedLine(activeWorktodoRawLine_)) {
            std::cout << "Gaussian-Mersenne entry removed from "
                      << options.worktodo_path << " and saved to worktodo_save.txt\n";
            const bool pending = io::WorktodoParser::hasPendingEntry(options.worktodo_path);
            if (pending) {
                std::cout << "Restarting for next Gaussian-Mersenne worktodo entry.\n";
                restart_self(argc_, argv_);
            } else {
                std::cout << "No more Gaussian-Mersenne worktodo entries.\n";
            }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n";
            return 2;
        }
    }

    if (options.gui) {
        if (guiServer_) {
            std::cout << "GUI " << guiServer_->url() << std::endl;
            guiServer_->setStatus(ran ? "Completed" : "Idle");
        }
        install_signal_handlers();
        // Idle until Ctrl-C / the GUI's Stop. A stop that already ended the run must not be cleared:
        // resetting the stop flag here made the process keep idling ("Completed") until a second signal, and
        // let a pending "Append & Run" restart the queue the user had just stopped.
        while (!stop_requested()) {
            gui_restart_for_appended_entry(guiServer_, argc_, argv_);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (guiServer_) guiServer_->stop();
    }

    return rc;
}





} // namespace core
