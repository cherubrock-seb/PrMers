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
#include "core/AlgoUtils.hpp"
#include "core/QuickChecker.hpp"
#include "core/Printer.hpp"
#include "core/ProofSet.hpp"
#include "core/ProofSetMarin.hpp"
#include "math/Carry.hpp"
#include "util/GmpUtils.hpp"
#include "io/WorktodoParser.hpp"
#include "io/WorktodoManager.hpp"
#include "io/CurlClient.hpp"
#include "marin/engine.h"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <map>
#include <future>
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
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <tuple>
#include <atomic>
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



static int askExponentInteractively() {
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
    return std::atoi(buffer);
  #else
    std::cout << "============================================\n"
              << " PrMers: GPU-accelerated Mersenne primality test\n"
              << " Powered by OpenCL | NTT | LL | PRP | IBDWT\n"
              << "============================================\n\n";
    std::string input;
    std::cout << "Enter the exponent to test (e.g. 21701): ";
    std::getline(std::cin, input);
    try {
      return std::stoi(input);
    } catch (...) {
      std::cerr << "Invalid input. Aborting.\n";
      std::exit(1);
    }
  #endif
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

    std::vector<char*> c_argv;
    for (auto& s : merged)
        c_argv.push_back(const_cast<char*>(s.c_str()));
    c_argv.push_back(nullptr);
    auto o = io::CliParser::parse(static_cast<int>(merged.size()), c_argv.data());

    io::WorktodoParser wp{o.worktodo_path};
    if (auto e = wp.parse()) {
        o.exponent     = e->exponent;
        o.mode         = e->ecmTest ? "ecm"
                        : (e->prpTest ? "prp"
                        : (e->llTest ? "ll"
                        : (e->pm1Test ? "pm1" : "")));
        o.aid          = e->aid;
        o.knownFactors = e->knownFactors;

        if (e->pm1Test) {
            o.B1 = e->B1;
            o.B2 = e->B2;
        }

        if (e->ecmTest) {
            o.B1  = e->B1;
            o.B2  = e->B2;
            o.nmax = e->curves;
            o.K    = e->curves;
        }

        hasWorktodoEntry_ = true;
    }


    if (!hasWorktodoEntry_ && o.exponent == 0) {
    std::cerr << "Error: no valid entry in " 
                << options.worktodo_path 
                << " and no exponent provided on the command line\n";
                
    
    if (!options.gui) {
        o.exponent = static_cast<uint64_t>(askExponentInteractively());
    }
    o.mode = "prp";
    //std::exit(-1);
    }
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
        options.knownFactors
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
        options.knownFactors
    )
  , spinner()
  , logger(options.output_path)
  , timer()
  , timer2()
{
    worktodoParser_ = std::make_unique<io::WorktodoParser>(options.worktodo_path);
    if (auto e = worktodoParser_->parse()) {
        hasWorktodoEntry_ = true;
    }
    
    context.computeOptimalSizes(
        precompute.getN(),
        precompute.getDigitWidth(),
        options.exponent,
        options.debug,
        options.max_local_size1,
        options.max_local_size5
    );
    //if(!options.marin){
        buffers.emplace(context, precompute);
        program.emplace(context, context.getDevice(), options.kernel_path, precompute,options.build_options, options.debug);
        kernels.emplace(program->getProgram(), context.getQueue());
        


        std::vector<std::string> kernelNames = {
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
        for (auto& name : kernelNames) {
            kernels->createKernel(name);
        }
        nttEngine.emplace(context, *kernels, *buffers, precompute, /*options.mode == "pm1",*/ options.debug);
    //}

    std::signal(SIGINT, handle_sigint);
}



int App::exportResumeFromMersFile(const std::string& mersPath,
                                  const std::string& savePath)
{
    std::vector<uint64_t> v(precompute.getN(), 0ULL);
    if (!read_mers_file(mersPath, v)) return -1;

    std::string fname = std::filesystem::path(mersPath).filename().string();
    size_t pos_pm = fname.find("pm");
    size_t pos_dot = fname.rfind('.');
    if (pos_pm == std::string::npos || pos_dot == std::string::npos || pos_pm >= pos_dot)
        return std::cerr << "Invalid filename format, expected <p>pm<B1>.mers\n", -1;

    std::string p_str  = fname.substr(0, pos_pm);
    std::string b1_str = fname.substr(pos_pm + 2, pos_dot - (pos_pm + 2));
    uint32_t p  = std::stoul(p_str);
    uint64_t B1 = std::stoull(b1_str);

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
static void prmers_bench_sigint(int) { prmers_bench_stop = 1; }

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
    const size_t R0 = 0, R1 = 1;
    double sum_time = 0.0;

    for (size_t ti = 0; ti < tasks.size(); ++ti) {
        if (prmers_bench_stop) break;
        uint32_t p = tasks[ti].p;
        engine* eng = nullptr;
        try { eng = engine::create_gpu(p, static_cast<size_t>(6), static_cast<size_t>(options.device_id), false/*, options.chunk256*/); } catch (...) { eng = nullptr; }
        if (!eng) continue;
        eng->set(R1, 1);
        eng->set(R0, 3);

        uint32_t warm = 96;
        for (uint32_t i = 0; i < warm && !prmers_bench_stop; ++i) eng->square_mul(R0);

        uint32_t ts =  eng->get_size();
        double target = (ts >= 33554432u) ? 10.0 : (ts >= 8388608u ? 8.0 : (ts >= 2621440u ? 6.0 : 5.0));

        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t cnt = 0;
        double last_update = 0.0;

        for (;;) {
            if (prmers_bench_stop) break;
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

        if (!prmers_bench_stop) {
            double ips = cnt / std::max(1e-9, elapsed);
            double eta_prp = (double)p / std::max(1e-9, ips);
            rows.push_back({ts, p, ips, eta_prp});
        }

        delete eng;
        if (prmers_bench_stop) break;
    }

    std::signal(SIGINT, old_handler);
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
  std::atomic<bool> g_stop{false};
  void handle_signal(int) noexcept {
    g_stop.store(true, std::memory_order_relaxed);
    core::algo::interrupted.store(true, std::memory_order_relaxed);
  }
}


#if defined(_WIN32)
#include <windows.h>
#include <csignal>
static BOOL WINAPI prmers_ctrl_handler(DWORD type){
    switch(type){
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT: handle_signal(SIGINT); return TRUE;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT: handle_signal(SIGTERM); return TRUE;
        default: return FALSE;
    }
}
static void install_signal_handlers() { SetConsoleCtrlHandler(prmers_ctrl_handler, TRUE); }
#else
#include <signal.h>
static void install_signal_handlers() {
    struct sigaction sa; sigemptyset(&sa.sa_mask); sa.sa_flags = 0; sa.sa_handler = handle_signal;
#ifdef SIGINT
    sigaction(SIGINT, &sa, nullptr);
#endif
#ifdef SIGTERM
    sigaction(SIGTERM, &sa, nullptr);
#endif
#ifdef SIGHUP
    sigaction(SIGHUP, &sa, nullptr);
#endif
    struct sigaction ign; sigemptyset(&ign.sa_mask); ign.sa_flags = 0; ign.sa_handler = SIG_IGN;
#ifdef SIGPIPE
    sigaction(SIGPIPE, &ign, nullptr);
#endif
}
#endif



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
        cfg.config_path = "./settings.cfg";
        cfg.results_path = "./results.txt";
        static std::atomic<bool> gui_alive{true};
        auto submitFn = [this](const std::string& line){
            std::ofstream out(this->options.worktodo_path, std::ios::app);
            out << line << "\n";
            out.close();
            std::cout << "worktodo appended\n";
            if (guiServer_) guiServer_->stop();
            restart_self(argc_, argv_);
        };
        auto stopFn = [&](){
            handle_signal(SIGINT);
            gui_alive = false;
        };
        guiServer_ = std::make_shared<ui::WebGuiServer>(cfg, submitFn, stopFn);
        ui::WebGuiServer::setInstance(guiServer_);
        guiServer_->start();
        std::cout << "GUI " << guiServer_->url() << std::endl;
        if (!file_non_empty(cfg.worktodo_path)) {
            guiServer_->setStatus("Idle");
            while (!g_stop && gui_alive) std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (guiServer_) guiServer_->stop();
            return 0;
        }
    }

    int rc = 1;
    bool ran = false;
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
    if (options.mode == "checksum") {
        std::cout << "Checksum mode\n";
        std::cout << "Paste the JSON to fix, then press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows):\n";
        std::ostringstream inbuf;
        inbuf << std::cin.rdbuf();
        std::string json = inbuf.str();
        if (json.find('{') == std::string::npos) {
            std::cerr << "No JSON received.\n";
        } else {
            std::string sum   = io::recomputeChecksumFromSubmittedJson(json);
            std::string fixed = io::rewriteChecksumInSubmittedJson(json);
            std::cout << "\nComputed checksum: " << sum << "\n";
            std::cout << "\nUpdated JSON:\n" << fixed << "\n";
        }
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

            if ((haveS2) && options.nmax == 0  && options.K == 0) {
                std::ostringstream msg;
                msg << "Detected P-1 checkpoint(s): "
                    << (haveS2 ? "[Stage 2] " : "")
                    << (haveS1 ? "[Stage 1] " : "")
                    << "→ jumping to runPM1Stage2Marin()";
                std::cout << msg.str() << std::endl;
                if (guiServer_) { guiServer_->appendLog(msg.str()); guiServer_->setStatus("Resuming P-1 Stage 2"); }

                rc_local = runPM1Stage2Marin();
                ran_local = true;
            } else {
                if(haveS1){
                    std::ostringstream msg;
                    msg << "Detected P-1 checkpoint(s): "
                    << (haveS2 ? "[Stage 2] " : "")
                    << (haveS1 ? "[Stage 1] " : "")
                    << "→ jumping to Stage1()";
                    std::cout << msg.str() << std::endl;
                }
                std::string msg = "No P-1 checkpoints found → running Stage 1 (runPM1Marin)";
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
        if (options.exponent > 89) {
            rc = runPM1();
            ran = true;
        } else {
            std::cout << "P-1 factoring (stage 1) need exponent > 89" << std::endl;
        }
    }

    if (options.gui) {
        if (guiServer_) {
            std::cout << "GUI " << guiServer_->url() << std::endl;
            guiServer_->setStatus(ran ? "Completed" : "Idle");
        }
        install_signal_handlers();
        g_stop = 0;
        while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (guiServer_) guiServer_->stop();
    }

    return rc;
}





} // namespace core
