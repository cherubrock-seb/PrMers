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

namespace {
std::atomic<bool> g_stop{false};
void g_onSig(int){ g_stop = true; }
}

namespace fs = std::filesystem;

using namespace std::chrono;

namespace core {

static std::atomic<bool> interrupted{false};
static void handle_sigint(int) { interrupted = true; }

static std::vector<std::string> parseConfigFile(const std::string& config_path) {
    std::ifstream config(config_path);
    std::vector<std::string> args;
    std::string line;

    if (!config.is_open()) {
        std::cerr << "Warning: no config file: " << config_path << std::endl;
        if (auto g = ui::WebGuiServer::instance()) {
            std::ostringstream oss;
            oss << "Warning: no config file: " << config_path;
            g->appendLog(oss.str());
        }
        return args;
    }

    std::cout << "Loading options from config file: " << config_path << std::endl;
    if (auto g = ui::WebGuiServer::instance()) {
        std::ostringstream oss;
        oss << "Loading options from config file: " << config_path;
        g->appendLog(oss.str());
    }

    while (std::getline(config, line)) {
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) args.push_back(token);
    }

    if (!args.empty()) {
        std::cout << "Options from config file:" << std::endl;
        if (auto g = ui::WebGuiServer::instance()) g->appendLog("Options from config file:");
        for (const auto& arg : args) std::cout << "  " << arg << std::endl;
        if (auto g = ui::WebGuiServer::instance()) {
            std::ostringstream oss;
            for (const auto& arg : args) oss << "  " << arg << std::endl;
            g->appendLog(oss.str());
        }
    } else {
        std::cout << "No options found in config file." << std::endl;
        if (auto g = ui::WebGuiServer::instance()) g->appendLog("No options found in config file.");
    }

    return args;
}

inline void writeStageResult(const std::string& file, const std::string& message) {
    std::ofstream out(file, std::ios::app);
    if (!out) {
        std::cerr << "Cannot open " << file << " for writing\n";
        if (auto g = ui::WebGuiServer::instance()) {
            std::ostringstream oss;
            oss << "Cannot open " << file << " for writing";
            g->appendLog(oss.str());
        }
        return;
    }
    out << message << '\n';
}


void restart_self(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() > 1 && args[1].find_first_not_of("0123456789") == std::string::npos) {
        args.erase(args.begin() + 1);
    }

#ifdef _WIN32
    std::string command = "\"" + args[0] + "\"";
    for (size_t i = 1; i < args.size(); ++i) command += " \"" + args[i] + "\"";
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, const_cast<char*>(command.c_str()), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        exit(0);
    } else {
        std::cerr << "Failed to restart program (CreateProcess failed)" << std::endl;
        if (auto g = ui::WebGuiServer::instance()) g->appendLog("Failed to restart program (CreateProcess failed)");
    }
#else
    std::cout << "\nRestarting program without exponent:\n";
    if (auto g = ui::WebGuiServer::instance()) g->appendLog("\nRestarting program without exponent:\n");
    for (const auto& arg : args) std::cout << "   " << arg << std::endl;
    if (auto g = ui::WebGuiServer::instance()) {
        std::ostringstream oss;
        for (const auto& arg : args) oss << "  " << arg << std::endl;
        g->appendLog(oss.str());
    }
    std::vector<char*> exec_args;
    for (auto& s : args) exec_args.push_back(const_cast<char*>(s.c_str()));
    exec_args.push_back(nullptr);
    execv(exec_args[0], exec_args.data());
    std::cerr << "Failed to restart program (execv failed)" << std::endl;
    if (auto g = ui::WebGuiServer::instance()) g->appendLog("Failed to restart program (execv failed)");
#endif
}


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
            o.mode         = e->prpTest ? "prp" : (e->llTest ? "ll" : (e->pm1Test ? "pm1" : ""));
            o.aid          = e->aid;
            o.knownFactors = e->knownFactors;
            if (e->pm1Test) {
                o.B1 = e->B1;
                o.B2 = e->B2;
            }
            hasWorktodoEntry_ = true;
      }

      if (!hasWorktodoEntry_ && o.exponent == 0) {
        std::cerr << "Error: no valid entry in " 
                  << options.worktodo_path 
                  << " and no exponent provided on the command line\n";
                  
        
        if (!options.gui) {
            o.exponent = askExponentInteractively();
        }
        o.mode = "prp";
        //std::exit(-1);
      }
      return o;
  }())
  , context(options.device_id,options.enqueue_max,options.cl_queue_throttle_active, options.debug,options.marin)
  , precompute(options.exponent)
  , backupManager(
        context.getQueue(),
        options.backup_interval,
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
        [this]() -> uint32_t {
            if (!this->options.proof) return 0;
            if (this->options.manual_proofPower) return this->options.proofPower;
            return ProofSet::bestPower(this->options.exponent);
        }(),
        context.getQueue(),
        precompute.getN(),
        precompute.getDigitWidth(),
        options.knownFactors
    ),
    proofManagerMarin(
        options.exponent,
        [this]() -> uint32_t {
            if (!this->options.proof) return 0;
            if (this->options.manual_proofPower) return this->options.proofPower;
            return ProofSet::bestPower(this->options.exponent);
        }(),
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
        nttEngine.emplace(context, *kernels, *buffers, precompute, options.mode == "pm1", options.debug);
    //}

    std::signal(SIGINT, handle_sigint);
}

static inline std::vector<uint32_t> pack_words_from_eng_digits(const engine::digit& d, uint32_t E) {
    const size_t totalWords = (E + 31) / 32;
    std::vector<uint32_t> out(totalWords, 0u);

    uint64_t acc = 0;
    int acc_bits = 0;
    size_t o = 0;

    const size_t n = d.get_size();
    for (size_t i = 0; i < n; ++i) {
        uint32_t w = uint32_t(d.width(i));
        uint32_t v = d.val(i);
        if (w < 32) v &= uint32_t((uint64_t(1) << w) - 1);
        acc |= (uint64_t)v << acc_bits;
        acc_bits += int(w);
        while (acc_bits >= 32 && o < totalWords) {
            out[o++] = uint32_t(acc & 0xFFFFFFFFu);
            acc >>= 32;
            acc_bits -= 32;
        }
    }
    if (o < totalWords) out[o++] = uint32_t(acc & 0xFFFFFFFFu);
    return out;
}

static inline std::vector<uint64_t> helperu(const engine::digit& d) {
    std::vector<uint64_t> out;
    out.reserve(d.get_size());
    for (size_t i = 0; i < d.get_size(); ++i) {
        uint64_t x = (uint64_t(d.width(i)) << 32) | uint64_t(d.val(i));
        out.push_back(x);
    }
    return out;
}


static inline uint32_t mod3_words(const std::vector<uint32_t>& W) {
    uint32_t r = 0; for (uint32_t w : W) r = (r + (w % 3)) % 3; return r;
}
static inline void div3_words(uint32_t E, std::vector<uint32_t>& W) {
    uint32_t r = (3 - mod3_words(W)) % 3;
    int topBits = int(E % 32);
    { uint64_t t = (uint64_t(r) << topBits) + W.back(); W.back() = uint32_t(t / 3); r = uint32_t(t % 3); }
    for (auto it = W.rbegin() + 1; it != W.rend(); ++it) { uint64_t t = (uint64_t(r) << 32) + *it; *it = uint32_t(t / 3); r = uint32_t(t % 3); }
}
static inline void prp3_div9(uint32_t E, std::vector<uint32_t>& W) { div3_words(E, W); div3_words(E, W); }

static inline std::string format_res64_hex(const std::vector<uint32_t>& W) {
    uint64_t r64 = (uint64_t(W.size() > 1 ? W[1] : 0) << 32) | (W.empty() ? 0u : W[0]);
    std::ostringstream oss; oss << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << r64; return oss.str();
}
static inline std::string format_res2048_hex(const std::vector<uint32_t>& W) {
    std::ostringstream oss; oss << std::hex << std::nouppercase << std::setfill('0');
    for (int i = 63; i >= 0; --i) {
        uint32_t w = (i < int(W.size())) ? W[i] : 0u;
        oss << std::setw(8) << w;
    }
    return oss.str();
}

static inline void delete_checkpoints(uint32_t p, bool wagstaff,bool pm1, bool llsafe, const std::string& dir = ".")
{
    std::string prefix;

    if (wagstaff) prefix += "wagstaff_";
    if (pm1)      prefix += "pm1_";
    if (llsafe)   prefix += "llsafe_";
    
    fs::path base = fs::path(dir) / (prefix + "m_" + std::to_string(p) + ".ckpt");
    std::error_code ec;
    fs::remove(base, ec);
    fs::remove(base.string() + ".old", ec);
    fs::remove(base.string() + ".new", ec);
}



inline void to_uppercase(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::toupper(c); });
}

int App::runPrpOrLlMarin()
{
    if (guiServer_) {
        guiServer_->setProgress(0, 100, "Started");
        guiServer_->setStatus("PrMers");
    }
    //if (guiServer_) guiServer_->appendLog("Hello world");

    Printer::banner(options);
    if (auto code = QuickChecker::run(options.exponent)) return *code;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = options.debug;

    engine* eng = engine::create_gpu(p, static_cast<size_t>(8), static_cast<size_t>(options.device_id), verbose  /*,options.chunk256*/);

    auto to_hex16 = [](uint64_t u){ std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u; return ss.str(); };

    if (verbose) std::cout << "Testing 2^" << p << " - 1, " << eng->get_size() << " 64-bit words..." << std::endl;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "Testing 2^" << p << " - 1";
        guiServer_->setStatus(oss.str());
    }
    if (guiServer_) {
            std::ostringstream oss;
            oss << "Testing 2^" << p << " - 1, " << eng->get_size() << " 64-bit words...";
            guiServer_->appendLog(oss.str());
    }
    if (options.proof) {
        uint32_t proofPower = options.manual_proofPower ? options.proofPower : ProofSetMarin::bestPower(options.exponent);
        options.proofPower = proofPower;
        double diskUsageGB = ProofSetMarin::diskUsageGB(options.exponent, proofPower);
        std::cout << "Proof of power " << proofPower << " requires about "
                  << std::fixed << std::setprecision(2) << diskUsageGB << "GB of disk space" << std::endl;
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Proof of power " << proofPower << " requires about "
                  << std::fixed << std::setprecision(2) << diskUsageGB << "GB of disk space";
            guiServer_->appendLog(oss.str());
        }
    }
    std::ostringstream ck;
    if (options.wagstaff) ck << "wagstaff_";
    ck << "m_" << p << ".ckpt";
    const std::string ckpt_file = ck.str();

    auto read_ckpt = [&](const std::string& file, uint32_t& ri, double& et)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 1) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != p) return -2;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = eng->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!eng->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt = [&](uint32_t i, double et){
        const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
        {
            File f(newf, "wb");
            int version = 1;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return;
            if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!eng->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file.c_str(), &s) == 0) && (std::rename(ckpt_file.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file.c_str());
    };

    const size_t R0 = 0, R1 = 1, R2 = 2, R3 = 3, R4 = 4, R5 = 5, RBASE = 6, RTMP=7;
    uint32_t ri = 0; double restored_time = 0;
    int r = read_ckpt(ckpt_file, ri, restored_time);
    if (r < 0) r = read_ckpt(ckpt_file + ".old", ri, restored_time);
    if (r == 0) {
        std::cout << "Resuming from a checkpoint." << std::endl;
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Resuming from a checkpoint.";
            guiServer_->appendLog(oss.str());
        }
        
    } else {
        ri = 0;
        restored_time = 0;
        eng->set(R1, 1);
        eng->set(R0, (options.mode == "prp") ? 3 : 4);
    }

    eng->copy(R4, R0);//Last correct state
    eng->copy(R5, R1);//Last correct bufd
    eng->set(RBASE, 3);
    eng->set_multiplicand(RTMP, RBASE);
    logger.logStart(options);
    timer.start();
    timer2.start();

    const uint32_t B_GL = std::max<uint32_t>(uint32_t(std::sqrt(p)), 2u);
    const auto start_clock = std::chrono::high_resolution_clock::now();
    auto lastBackup = start_clock;
    auto lastDisplay = start_clock;

    uint64_t totalIters = options.mode == "prp" ? p : p - 2;
    
    if(options.wagstaff){
        totalIters /= 2;
    }

    uint64_t itersave =  backupManager.loadGerbiczIterSave();
    uint64_t jsave = backupManager.loadGerbiczJSave();
    if(jsave==0){
        jsave = totalIters - 1;
    }

    uint64_t L = options.exponent;
    uint64_t B = (uint64_t)(std::sqrt((double)L));
    double desiredIntervalSeconds = 600.0;
    uint64_t checkpass = 0;

    uint64_t checkpasslevel_auto = (uint64_t)((1000 * desiredIntervalSeconds) / (double)B);
    if (checkpasslevel_auto == 0) checkpasslevel_auto = (totalIters/B)/((uint64_t)(std::sqrt((double)B)));
    uint64_t checkpasslevel = (options.checklevel > 0)
        ? options.checklevel
        : checkpasslevel_auto;
    if(checkpasslevel==0)
        checkpasslevel=1;
    uint64_t resumeIter = ri;
    uint64_t startIter  = ri;
    uint64_t lastIter   = ri ? ri - 1 : 0;
    uint64_t lastJ      = p - 1 - ri;
    std::string res64_x;
    
    spinner.displayProgress(resumeIter, totalIters, 0.0, 0.0, options.wagstaff ? p / 2 : p, resumeIter, startIter, res64_x, guiServer_ ? guiServer_.get() : nullptr);
    bool errordone = false;
    if(options.wagstaff){
        std::cout << "[WAGSTAFF MODE] This test will check if (2^" << options.exponent/2 << " + 1)/3 is PRP prime" << std::endl;
        if (guiServer_) {
            std::ostringstream oss;
            oss << "[WAGSTAFF MODE] This test will check if (2^" << options.exponent/2 << " + 1)/3 is PRP prime";
            guiServer_->appendLog(oss.str());
        }
    }
    std::vector<uint32_t> words;

    bool is_divisible_by_9 = (((mpz_class(1) << options.exponent) - 1)%9==0);
    if(is_divisible_by_9){
        std::cout << "M_"<< options.exponent <<" IS DIVISIBLE BY 9" << std::endl;
    }

    for (uint64_t iter = resumeIter, j= totalIters-resumeIter-1; iter < totalIters; ++iter, --j) {
        lastJ = j;
        lastIter = iter;
        if (interrupted)
        {
            const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
            save_ckpt(iter, elapsed_time);
            delete eng;
            std::cout << "\nInterrupted by user, state saved at iteration " << iter << " j=" << j << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\nInterrupted by user, state saved at iteration " << iter << " j=" << j << std::endl;
                guiServer_->appendLog(oss.str());
            }
            logger.logEnd(elapsed_time);
            return 0;
        }
        auto now0 = std::chrono::high_resolution_clock::now();

        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval))
        {
            const double elapsed_time = std::chrono::duration<double>(now0 - start_clock).count() + restored_time;
            std::cout << "\nBackup point done at iter + 1=" << iter + 1 << " start...." << std::endl;
            save_ckpt(iter, elapsed_time);
            lastBackup = now0;
            std::cout << "\nBackup point done at iter + 1=" << iter + 1 << " done...." << std::endl;
            spinner.displayBackupInfo(iter + 1, totalIters, timer.elapsed(), res64_x);
        }
        eng->square_mul(R0);
        if (options.mode == "ll") {
            eng->sub(R0, 2);
        }

        if (options.erroriter > 0 && (iter + 1) == options.erroriter && !errordone) {
            errordone = true;
            //eng->error();
            eng->sub(R0, 2);
            std::cout << "Injected error at iteration " << (iter + 1) << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Injected error at iteration " << (iter + 1);
                guiServer_->appendLog(oss.str());
            }
        }

        if (options.mode == "prp" && options.gerbiczli && ((j != 0 && (j % B == 0)) || iter == totalIters - 1)) {
            checkpass += 1;
            eng->copy(R3, R1);
            eng->set_multiplicand(R2, R0);
            eng->mul(R1, R2);
            bool condcheck = !(checkpass != checkpasslevel && (iter != totalIters - 1));
            
            if (condcheck) {
                    checkpass = 0;
                    uint64_t modB = (options.exponent % B == 0 ? B : options.exponent % B);
                    uint64_t loop_count = (B > modB ? B - modB - 1 : 0);

                    for (uint64_t z = 0; z < loop_count; ++z) {
                        eng->square_mul(R3);
                    }
                    if(options.exponent % B == 0 ){
                        eng->mul(R3, RTMP);
                    }
                    else{
                        eng->square_mul(R3, 3);
                    }
                    for (uint64_t z = 0; z < ((options.exponent % B == 0 ? B : options.exponent % B)); ++z) {
                        eng->square_mul(R3);
                    }
                    mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
                    eng->get_mpz(z0, R3); eng->get_mpz(z1, R1);
                    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;

                    bool is_eq = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);

                     mpz_clears(z0, z1, nullptr);
                    if (!is_eq) 
                    { 
                        //delete eng; 
                        //throw std::runtime_error("Gerbicz-Li error checking failed!"); 
                        std::cout << "[Gerbicz Li] Mismatch \n"
                            << "[Gerbicz Li] Check FAILED! iter=" << (iter + 1) << "\n"
                            << "[Gerbicz Li] Restore iter=" << itersave << " (j=" << jsave << ")\n";
                        if (guiServer_) {
                            std::ostringstream oss;
                            oss << "[Gerbicz Li] Mismatch \n"
                            << "[Gerbicz Li] Check FAILED! iter=" << (iter + 1) << "\n"
                            << "[Gerbicz Li] Restore iter=" << itersave << " (j=" << jsave << ")\n";
                            guiServer_->appendLog(oss.str());
                        }
                        j = jsave;
                        iter = itersave;
                        lastIter = itersave;
                        lastIter = iter;
                        if (iter == 0) {
                            iter = iter - 1;
                            j = j + 1;
                        }
                        checkpass = 0;
                        options.gerbicz_error_count += 1;
                        eng->copy(R0, R4);
                        eng->copy(R1, R5);
                    }
                    else{
                        std::cout << "[Gerbicz Li] Check passed! iter=" << (iter + 1) << "\n";
                        if (guiServer_) {
                            std::ostringstream oss;
                            oss << "[Gerbicz Li] Check passed! iter=" << (iter + 1) << "\n";
                            guiServer_->appendLog(oss.str());
                        }
                        eng->copy(R4, R0);//Last correct state
                        eng->copy(R5, R1);//Last correct bufd
                        itersave = iter;
                        jsave = j;
                        cl_event postEvt;
                    }
            }
            

        } 

        auto now = std::chrono::high_resolution_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDisplay).count() >= 10)
        {
            spinner.displayProgress(
                iter + 1,
                totalIters,
                timer.elapsed(),
                timer2.elapsed(),
                options.wagstaff ? p / 2 : p,
                resumeIter,
                startIter,
                res64_x, 
                guiServer_ ? guiServer_.get() : nullptr
            );
            timer2.start();
            lastDisplay = now;
            resumeIter = iter + 1;
        }

        if (options.mode == "prp"  && options.proof && (iter + 1) < totalIters && proofManagerMarin.shouldCheckpoint(iter+1)) {
            engine::digit d(eng, R0);
            proofManagerMarin.checkpointMarin(d, iter + 1);
        }

    }

    if (options.proof) {
        engine::digit d(eng, R0);
        proofManagerMarin.checkpointMarin(d, totalIters);
    }

    bool is_prp_prime = false;
    engine::digit digit(eng, R0);
    std::vector<uint64_t> d = helperu(digit);
    if (options.mode == "ll") {
        is_prp_prime = (digit.equal_to(0) || digit.equal_to_Mp());
    }
    else{
        is_prp_prime = digit.equal_to(9);
    }
    words = pack_words_from_eng_digits(digit, p);

    std::string res64_hex;
    std::string res2048_hex;
    if (!is_divisible_by_9) {
        if (options.mode == "prp") prp3_div9(p, words);
        res64_hex    = format_res64_hex(words);
        res2048_hex  = format_res2048_hex(words);

    } else {
        mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
        mpz_t z0; mpz_inits(z0, nullptr);
        eng->get_mpz(z0, R0);
       // std::cout << "RESULTAT=" << z0 << "\n";
        unsigned t=0;
        mpz_class tmp = Mp;
        while (mpz_divisible_ui_p(tmp.get_mpz_t(), 3)) {
            mpz_divexact_ui(tmp.get_mpz_t(), tmp.get_mpz_t(), 3);
            ++t;
        }
        //std::cout << "t=" << t << "\n";
        mpz_class m3;
        mpz_ui_pow_ui(m3.get_mpz_t(), 3, t); 
        mpz_class u = Mp / m3;

        mpz_class a_u;
        mpz_mod(a_u.get_mpz_t(), z0, u.get_mpz_t());  // a_u = z0 mod u

        // res_u = a_u * 9^{-1} mod u
        mpz_class inv9, res_u;
        if (mpz_invert(inv9.get_mpz_t(), mpz_class(9).get_mpz_t(), u.get_mpz_t()) == 0)
            throw std::runtime_error("inv(9, u) failed");
        res_u = (a_u * inv9) % u;

        // calcul de k = (-res_u * u^{-1}) mod m3
        mpz_class inv_u_mod_m3;
        if (mpz_invert(inv_u_mod_m3.get_mpz_t(), u.get_mpz_t(), m3.get_mpz_t()) == 0)
            throw std::runtime_error("inv(u, m3) failed");
        mpz_class k = (-res_u * inv_u_mod_m3) % m3;
        if (k < 0) k += m3; // assurer k ∈ [0, m3)

        // x = res_u + k * u
        mpz_class x = res_u + k * u;
        x %= Mp;
        mpz_class res64, res2048;
        mpz_mod_2exp(res64.get_mpz_t(), x.get_mpz_t(), 64);       // res64 = x mod 2^64
        mpz_mod_2exp(res2048.get_mpz_t(), x.get_mpz_t(), 2048);   // res2048 = x mod 2^2048

        //std::cout << "[CRT] res64    = 0x" << res64.get_str(16) << "\n";
        //std::cout << "[CRT] res2048  = 0x" << res2048.get_str(16) << "\n";
        std::ostringstream oss64;
        oss64 << std::hex << std::setfill('0') << std::setw(16) << res64.get_ui();
        res64_hex = oss64.str();
        res2048_hex = res2048.get_str(16);

        if (res2048_hex.length() < 512) {
            res2048_hex = std::string(512 - res2048_hex.length(), '0') + res2048_hex;
        }


    }
   // to_uppercase(res64_hex);
   // to_uppercase(res2048_hex);
    spinner.displayProgress(
        totalIters,
        totalIters,
        timer.elapsed(),
        timer2.elapsed(),
        options.wagstaff ? p / 2 : p,
        resumeIter,
        startIter,
        res64_hex, 
        guiServer_ ? guiServer_.get() : nullptr
    );

    if (options.wagstaff) {
            mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
            mpz_class Fp = (mpz_class(1) << options.exponent/2) + 1;
            mpz_class rM = util::vectToMpz(d,
                                        precompute.getDigitWidth(),
                                        Mp);
            mpz_class rF = rM % Fp;
            bool isWagstaffPRP = (rF == 9);
            const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
            if (isWagstaffPRP) {
                std::cout << "Wagstaff PRP confirmed: (2^"<< options.exponent/2 <<"+1)/3 is a probable prime.\n";
                if (guiServer_) {
                            std::ostringstream oss;
                            oss << "Wagstaff PRP confirmed: (2^"<< options.exponent/2 <<"+1)/3 is a probable prime.\n";
                            guiServer_->appendLog(oss.str());
                }
            } else {
                std::cout << "Not a Wagstaff PRP.\n";
                if (guiServer_) {
                            std::ostringstream oss;
                            oss << "Not a Wagstaff PRP.\n";
                            guiServer_->appendLog(oss.str());
                }
            }

            logger.logEnd(elapsed_time);
            return isWagstaffPRP ? 0 : 1;
    }
    
    const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
    if (options.mode == "prp") {
        std::cout << "2^" << p << " - 1 is " << (is_prp_prime ? "a probable prime" : ("composite, res64 = " + (res64_hex))) << ", time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl;
        if (guiServer_) {
                std::ostringstream oss;
                oss << "2^" << p << " - 1 is " << (is_prp_prime ? "a probable prime" : ("composite, res64 = " + (res64_hex))) << ", time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl;
                guiServer_->appendLog(oss.str());
        }
    }
    else{
        std::cout << "2^" << p << " - 1 is " << (is_prp_prime ? "prime" : "composite");
        if (guiServer_) {
                std::ostringstream oss;
                oss  << "2^" << p << " - 1 is " << (is_prp_prime ? "prime" : "composite");
                guiServer_->appendLog(oss.str());
        }
    }		
    logger.logEnd(elapsed_time);

    if (options.proof) {
        cl_command_queue queue = context.getQueue();
        math::Carry carry(
            context,
            queue,
            program->getProgram(),
            precompute.getN(),
            precompute.getDigitWidth(),
            buffers->digitWidthMaskBuf
        );
        uint32_t proofPower = (options.proofPower);
        while (proofPower >= 0) {
            try {
                std::cout << "\nGenerating PRP proof file..." << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "\nGenerating PRP proof file...";
                    guiServer_->appendLog(oss.str());
                }
                options.proofPower = proofPower;
                auto proofFilePath = proofManager.proof(context, *nttEngine, carry, proofPower, options.verify);
                options.proofFile = proofFilePath.string();
                std::cout << "Proof file saved: " << proofFilePath << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Proof file saved: " << proofFilePath;
                    guiServer_->appendLog(oss.str());
                }
                break;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Proof generation failed: " << e.what() << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Warning: Proof generation failed: " << e.what();
                    guiServer_->appendLog(oss.str());
                }
                if (proofPower == 0) break;
                --proofPower;
                std::cout << "Retrying proof generation with reduced power: " << proofPower << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Retrying proof generation with reduced power: " << proofPower;
                    guiServer_->appendLog(oss.str());
                }
            }
        }
    }

    std::string json;
    if (!options.knownFactors.empty()) {
        auto [isPrime, res64, res2048] = io::JsonBuilder::computeResultMarin(d, options);
    
        is_prp_prime = isPrime;
        json = io::JsonBuilder::generate(
            options,
            static_cast<int>(context.getTransformSize()),
            isPrime,
            res64,
            res2048
        );

        Printer::finalReport(
            options,
            elapsed_time,
            json,
            isPrime
        );
    }
    else{
        json = io::JsonBuilder::generate(
            options,
            static_cast<int>(context.getTransformSize()),
            is_prp_prime,
            res64_hex,
            res2048_hex
        );

        Printer::finalReport(
            options,
            elapsed_time,
            json,
            is_prp_prime
        );
    }
    bool skippedSubmission = false;
    if (options.submit) {
        bool noAsk = options.noAsk || hasWorktodoEntry_;
        if (noAsk && options.password.empty()) {
            std::cerr << "No password provided with --noask; skipping submission.\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss  <<  "No password provided with --noask; skipping submission.\n";
                guiServer_->appendLog(oss.str());
            }
        } else {
            if(!options.gui){
                std::string response;
                std::cout << "Do you want to send the result to PrimeNet (https://www.mersenne.org) ? (y/n): ";
                std::getline(std::cin, response);
                if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
                    std::cout << "Result not sent." << std::endl;
                    skippedSubmission = true;
                }
                if (!skippedSubmission && options.user.empty()) {
                    std::cout << "\nEnter your PrimeNet username (Don't have an account? Create one at https://www.mersenne.org): ";
                    std::getline(std::cin, options.user);
                }
                if (!skippedSubmission && options.user.empty()) {
                    std::cerr << "No username entered; skipping submission.\n";
                    skippedSubmission = true;
                }
                if (!skippedSubmission) {
                    if (!noAsk && options.password.empty()) {
                        options.password = io::CurlClient::promptHiddenPassword();
                    }
                    bool success = io::CurlClient::sendManualResultWithLogin(
                        json,
                        options.user,
                        options.password
                    );
                    if (!success) {
                        std::cerr << "Submission to PrimeNet failed\n";
                    }
                }
            }
        }
    }

    backupManager.clearState();
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, options.mode, json);
    wm.appendToResultsTxt(json);
    delete_checkpoints(p, options.wagstaff, false, false); 
    backupManager.clearState();
    if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss  <<   "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
                guiServer_->appendLog(oss.str());
            }
            std::ifstream f(options.worktodo_path);
            std::string l;
            bool more = false;
            while (std::getline(f, l)) {
                if (!l.empty() && l[0] != '#') { more = true; break; }
            }
            f.close();
            if (more) {
                std::cout << "Restarting for next entry in worktodo.txt\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss  <<   "Restarting for next entry in worktodo.txt\n";
                    guiServer_->appendLog(oss.str());
                }
                restart_self(argc_, argv_);
            } else {
                std::cout << "No more entries in worktodo.txt, exiting.\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss  <<    "No more entries in worktodo.txt, exiting.\n";
                    guiServer_->appendLog(oss.str());
                }
                if (!options.gui) {
                    std::exit(0);
                }
            }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n";
            if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "Failed to update " << options.worktodo_path << "\n";
                    guiServer_->appendLog(oss.str());
            }
            if (!options.gui) {
                std::exit(-1);
            }
        }
    }

    delete eng;
    return is_prp_prime ? 0 : 1;
}

int App::runLlSafeMarin()
{
    Printer::banner(options);
    std::cout << "[Lucas Lehmer GPU SAFE Mode with error detection]\n";
    if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "[Lucas Lehmer GPU SAFE Mode with error detection]\n";
                    guiServer_->appendLog(oss.str());
    }
    
    if (auto code = QuickChecker::run(options.exponent)) return *code;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = options.debug;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "Testing 2^" << p << " - 1";
        guiServer_->setStatus(oss.str());
    }
    engine* eng = engine::create_gpu(p, static_cast<size_t>(8), static_cast<size_t>(options.device_id), verbose);
    if (verbose) std::cout << "Testing 2^" << p << " - 1 (LL-safe, GPU), " << eng->get_size() << " 64-bit words..." << std::endl;
    if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "Testing 2^" << p << " - 1 (LL-safe, GPU), " << eng->get_size() << " 64-bit words..." << std::endl;
                    guiServer_->appendLog(oss.str());
    }
    std::ostringstream ck; ck << "llsafe_m_" << p << ".ckpt";
    const std::string ckpt_file = ck.str();

    auto read_ckpt = [&](const std::string& file, uint32_t& ri, double& et)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 1) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != p) return -2;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = eng->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!eng->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt = [&](uint32_t i, double et){
        const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
        {
            File f(newf, "wb");
            int version = 1;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return;
            if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!eng->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file.c_str(), &s) == 0) && (std::rename(ckpt_file.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file.c_str());
    };

    const size_t RV = 0, RU = 1, RVC = 2, RUC = 3, RTMP = 4, RVCHK = 5, RUCHK = 6, RSCR = 7;

    uint32_t ri = 0; double restored_time = 0.0;
    int r = read_ckpt(ckpt_file, ri, restored_time);
    if (r < 0) r = read_ckpt(ckpt_file + ".old", ri, restored_time);
    if (r == 0) {
        std::cout << "Resuming from a checkpoint." << std::endl;
        if (guiServer_) {
                    std::ostringstream oss;
                    oss  <<  "Resuming from a checkpoint." << std::endl;
                    guiServer_->appendLog(oss.str());
        }
    } else {
        ri = 0;
        restored_time = 0.0;
        eng->set(RV, 4);
        eng->set(RU, 2);
    }

    eng->copy(RVC, RV);
    eng->copy(RUC, RU);
    eng->copy(RVCHK, RVC);
    eng->copy(RUCHK, RUC);

    logger.logStart(options);
    timer.start();
    timer2.start();

    const auto start_clock = std::chrono::high_resolution_clock::now();
    auto lastBackup = start_clock;
    auto lastDisplay = start_clock;

    uint64_t totalIters = (p >= 2) ? (uint64_t)(p - 2) : 0ULL;

    uint64_t L = options.exponent;
    uint64_t B = (options.llsafe_block > 0) ? (uint64_t)options.llsafe_block
                                            : (uint64_t)(options.exponent/std::sqrt((double)L));;
    
    if (B < 1) B = 1;
    if (B > totalIters) B = totalIters;

    uint64_t resumeIter = ri;
    uint64_t startIter  = ri;

    spinner.displayProgress(resumeIter, totalIters, 0.0, 0.0, p, resumeIter, startIter, "", guiServer_ ? guiServer_.get() : nullptr);

    bool errordone = false;
    uint64_t itersave = (ri / B) * B;

    for (uint64_t iter = resumeIter; iter < totalIters; ++iter) {
        if (interrupted) {
            const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
            save_ckpt((uint32_t)iter, elapsed_time);
            delete eng;
            std::cout << "\nInterrupted by user, state saved at iteration " << iter << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "\nInterrupted by user, state saved at iteration " << iter << std::endl;
                    guiServer_->appendLog(oss.str());
            }
            logger.logEnd(elapsed_time);
            return 0;
        }
        auto now0 = std::chrono::high_resolution_clock::now();
        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            const double elapsed_time = std::chrono::duration<double>(now0 - start_clock).count() + restored_time;
            std::cout << "\nBackup point done at iter=" << iter << " start...." << std::endl;
            save_ckpt((uint32_t)iter, elapsed_time);
            lastBackup = now0;
            std::cout << "\nBackup point done at iter=" << iter << " done...." << std::endl;   
            spinner.displayBackupInfo(iter + 1, totalIters, timer.elapsed(), "");
        }
        if (options.erroriter > 0 && (iter + 1) == (uint64_t)options.erroriter && !errordone) {
            errordone = true;
            eng->sub(RV, 2);
            std::cout << "Injected error at iteration " << (iter + 1) << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "Injected error at iteration " << (iter + 1) << std::endl;
                    guiServer_->appendLog(oss.str());
            }
        }

        eng->set_multiplicand(RTMP, RV);
        eng->mul(RU, RTMP);
        eng->square_mul(RV);
        eng->sub(RV, 2);



        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDisplay).count() >= 10) {
            spinner.displayProgress(iter + 1, totalIters, timer.elapsed(), timer2.elapsed(), p, resumeIter, startIter, "", guiServer_ ? guiServer_.get() : nullptr);
            timer2.start();
            lastDisplay = now;
            resumeIter = iter + 1;
        }

        bool boundary = (((iter + 1) % B) == 0) || (iter + 1 == totalIters);
        if (boundary) {
            uint64_t blk = ((iter + 1) % B == 0) ? B : ((iter + 1) - itersave);
            eng->copy(RVCHK, RVC);
            eng->copy(RUCHK, RUC);
            for (uint64_t z = 0; z < blk; ++z) {
                eng->set_multiplicand(RTMP, RVCHK);
                eng->mul(RUCHK, RTMP);
                eng->square_mul(RVCHK);
                eng->sub(RVCHK, 2);
            }
            //bool okV = eng->is_equal(RVCHK, RV);
            mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
            eng->get_mpz(z0, RVCHK); eng->get_mpz(z1, RV);
            bool okV = (mpz_cmp(z0, z1)==0);
            //mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
            //bool okV = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);

            //bool okU = eng->is_equal(RUCHK, RU);
            mpz_inits(z0, z1, nullptr);
            eng->get_mpz(z0, RUCHK); eng->get_mpz(z1, RU);
            bool okU = (mpz_cmp(z0, z1)==0);
            //mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
            //bool okU = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);

            mpz_clears(z0, z1, nullptr);
            if (!(okV && okU)) {
                std::cout << "[Error check] Mismatch \n"
                          << "[Error check] Check FAILED! iter=" << iter << "\n"
                          << "[Error check] Restore iter=" << itersave << "\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss  << "[Error check] Mismatch \n"
                          << "[Error check] Check FAILED! iter=" << iter << "\n"
                          << "[Error check] Restore iter=" << itersave << "\n";
                    guiServer_->appendLog(oss.str());
                }
                eng->copy(RV, RVC);
                eng->copy(RU, RUC);
                if (itersave == 0) {
                    iter = (uint64_t)-1;
                    resumeIter = 0;
                } else {
                    iter = itersave - 1;
                    resumeIter = itersave;
                }
            } else {
                std::cout << "[Error check] Check passed! iter=" << iter << "\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[Error check] Check passed! iter=" << iter << "\n";
                    guiServer_->appendLog(oss.str());
                }
                eng->copy(RVC, RV);
                eng->copy(RUC, RU);
                itersave = iter + 1;
            }
        }
    }

    engine::digit dV(eng, RV);
    bool is_prime = (dV.equal_to(0) || dV.equal_to_Mp());

    std::vector<uint32_t> words = pack_words_from_eng_digits(dV, p);
    if (is_prime && dV.equal_to_Mp()) {
        std::fill(words.begin(), words.end(), 0u);
    }
    std::string res64_hex   = format_res64_hex(words);
    std::string res2048_hex = format_res2048_hex(words);

    spinner.displayProgress(totalIters, totalIters, timer.elapsed(), timer2.elapsed(), p, resumeIter, startIter, res64_hex, guiServer_ ? guiServer_.get() : nullptr);

    const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
    std::cout << "2^" << p << " - 1 is " << (is_prime ? "prime" : ("composite, res64 = " + res64_hex)) << std::endl;
    logger.logEnd(elapsed_time);
    if (guiServer_) {
                    std::ostringstream oss;
                    oss << "2^" << p << " - 1 is " << (is_prime ? "prime" : ("composite, res64 = " + res64_hex)) << std::endl;
    
                    guiServer_->appendLog(oss.str());
    }
    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), is_prime, res64_hex, res2048_hex);
    Printer::finalReport(options, elapsed_time, json, is_prime);

    if (options.submit && !options.gui) {
        bool noAsk = options.noAsk || hasWorktodoEntry_;
        if (noAsk && options.password.empty()) {
            std::cerr << "No password provided with --noask; skipping submission.\n";
        } else {
            std::string response;
            std::cout << "Do you want to send the result to PrimeNet (https://www.mersenne.org) ? (y/n): ";
            std::getline(std::cin, response);
            bool skippedSubmission = false;
            if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
                std::cout << "Result not sent." << std::endl;
                skippedSubmission = true;
            }
            if (!skippedSubmission && options.user.empty()) {
                std::cout << "\nEnter your PrimeNet username (Don't have an account? Create one at https://www.mersenne.org): ";
                std::getline(std::cin, options.user);
            }
            if (!skippedSubmission && options.user.empty()) {
                std::cerr << "No username entered; skipping submission.\n";
                skippedSubmission = true;
            }
            if (!skippedSubmission) {
                if (!noAsk && options.password.empty()) {
                    options.password = io::CurlClient::promptHiddenPassword();
                }
                bool success = io::CurlClient::sendManualResultWithLogin(json, options.user, options.password);
                if (!success) {
                    std::cerr << "Submission to PrimeNet failed\n";
                }
            }
        }
    }

    delete_checkpoints(options.exponent, options.wagstaff, true, true);
    delete eng;
    return is_prime ? 0 : 1;
}



int App::runPrpOrLl() {
    
    Printer::banner(options);
    if (auto code = QuickChecker::run(options.exponent))
        return *code;

    cl_command_queue queue   = context.getQueue();
    size_t          queued   = 0;
    
    uint64_t p = options.exponent;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "Testing 2^" << p << " - 1";
        guiServer_->setStatus(oss.str());
    }
    uint64_t totalIters = options.mode == "prp" ? p : p - 2;
    if(options.wagstaff){
        totalIters /= 2;
    }


    std::vector<uint64_t> x(precompute.getN(), 0ULL);
    uint64_t resumeIter = backupManager.loadState(x);
    if (resumeIter == 0) {
        x[0] = (options.mode == "prp") ? 3ULL : 4ULL;
        if(options.debug){
            std::cout << "Initial x[0] set to " << x[0]
                    << " (" << (options.mode == "prp" ? "PRP" : "LL")
                    << " mode)" << std::endl;
        }
    }
    buffers->input = clCreateBuffer(
        context.getContext(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        x.size() * sizeof(uint64_t),
        x.data(), nullptr
    );
    std::cout << "Sampling 100 iterations for IPS estimation...\n";
    if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Sampling 100 iterations for IPS estimation...\n";
                    guiServer_->appendLog(oss.str());
    }
    double sampleIps = measureIps(options.iterforce, 100);
    std::cout << "Estimated IPS: " << sampleIps << "\n";
    if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Estimated IPS: " << sampleIps << "\n";
                    guiServer_->appendLog(oss.str());
    }
    if (options.tune) {
        tuneIterforce();
        return 0;
    }
    clEnqueueWriteBuffer(
            context.getQueue(),
            buffers->input,
            CL_TRUE, 0,
            x.size() * sizeof(uint64_t),
            x.data(),
            0, nullptr, nullptr
        );
    

    math::Carry carry(
        context,
        queue,
        program->getProgram(),
        precompute.getN(),
        precompute.getDigitWidth(),
        buffers->digitWidthMaskBuf
    );

    // Display proof disk usage estimate at start of computation
    if (options.proof) {
        uint32_t proofPower = options.manual_proofPower ? options.proofPower : ProofSet::bestPower(options.exponent);
        options.proofPower = proofPower;
        double diskUsageGB = ProofSet::diskUsageGB(options.exponent, proofPower);
        std::cout << "Proof of power " << proofPower << " requires about "
                << std::fixed << std::setprecision(2) << diskUsageGB
                << "GB of disk space" << std::endl;
        if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Proof of power " << proofPower << " requires about "
                << std::fixed << std::setprecision(2) << diskUsageGB
                << "GB of disk space" << std::endl;
                    guiServer_->appendLog(oss.str());
        }
    }

    logger.logStart(options);
    timer.start();
    timer2.start();
    auto startTime  = high_resolution_clock::now();
    auto lastBackup = startTime;
    auto lastDisplay = startTime;
    uint64_t lastIter = resumeIter;
    uint64_t startIter = resumeIter;
    
    uint64_t L = options.exponent;
    uint64_t B = (uint64_t)(std::sqrt((double)L));

    size_t limbs = precompute.getN();
    size_t limbBytes = limbs * sizeof(uint64_t);
    cl_int err;
    //cl_mem r2,save,bufd,buf3;

    if(options.mode=="prp" && options.gerbiczli){
        // See: An Efficient Modular Exponentiation Proof Scheme, 
        //§2, Darren Li, Yves Gallot, https://arxiv.org/abs/2209.15623
        std::vector<uint64_t> hotsd(precompute.getN(), 0ULL);
        hotsd[0] = 1ULL;
        backupManager.loadGerbiczLiBufDState(hotsd);
    
        buffers->bufd = clCreateBuffer(
            context.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            hotsd.size() * sizeof(uint64_t),
            hotsd.data(),  &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate bufd: " << err << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate bufd: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
            }
            exit(1);
        }
        
        backupManager.loadGerbiczLiCorrectBufDState(hotsd);
        
        buffers->last_correct_bufd = clCreateBuffer(
            context.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            hotsd.size() * sizeof(uint64_t),
            hotsd.data(),  &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate last_correct_bufd: " << err << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate last_correct_bufd: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
            }
            exit(1);
        }
        std::vector<uint64_t> hots3(precompute.getN(), 0ULL);
        hots3[0] = 3ULL;
        
        buffers->r2 = clCreateBuffer(
            context.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            hots3.size() * sizeof(uint64_t),
            hots3.data(),  &err
            );
            if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate r2: " << err << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate r2: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
            }
            exit(1);
        }


        buffers->save = clCreateBuffer(
            context.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            hots3.size() * sizeof(uint64_t),
            hots3.data(),  &err
            );
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate save: " << err << std::endl; 
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate save: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
            }
            exit(1);
        }
        
        backupManager.loadGerbiczLiCorrectState(hots3);
            
        buffers->last_correct_state = clCreateBuffer(
            context.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            hots3.size() * sizeof(uint64_t),
            hots3.data(),  &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate last_correct_state: " << err << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate last_correct_state: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
            }
            exit(1);
        }

        
    }
   
    std::vector<uint64_t> hostR2(precompute.getN());
   
    //gchk.init(buffers->input, resumeIter);
    bool errordone = false;
    //uint64_t checkpasslevel = (totalIters/B)/((uint64_t)(std::sqrt((double)B)));

    double desiredIntervalSeconds = 600.0;
    uint64_t checkpasslevel_auto = (uint64_t)((sampleIps * desiredIntervalSeconds) / (double)B);

    if (checkpasslevel_auto == 0) checkpasslevel_auto = (totalIters/B)/((uint64_t)(std::sqrt((double)B)));

    uint64_t checkpasslevel = (options.checklevel > 0)
        ? options.checklevel
        : checkpasslevel_auto;



    uint64_t itersave =  backupManager.loadGerbiczIterSave();
        
    uint64_t checkpass = 0;
    uint64_t jsave = backupManager.loadGerbiczJSave();
    if(jsave==0){
        jsave = totalIters - 1;
    }
    if(checkpasslevel==0)
        checkpasslevel=1;
    if(options.wagstaff){
        std::cout << "[WAGSTAFF MODE] This test will check if (2^" << options.exponent/2 << " + 1)/3 is PRP prime" << std::endl;
        if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[WAGSTAFF MODE] This test will check if (2^" << options.exponent/2 << " + 1)/3 is PRP prime" << std::endl;
                    guiServer_->appendLog(oss.str());
        }
    }
    if(options.gerbiczli && options.debug){
        std::cout << "[Gerbicz Li] B=" << B << std::endl;
        std::cout << "[Gerbicz Li] Checkpasslevel=" << checkpasslevel << std::endl;
        std::cout << "[Gerbicz Li] j=" << totalIters-resumeIter-1 << std::endl;
        std::cout << "[Gerbicz Li] iter=" << resumeIter << std::endl;
        std::cout << "[Gerbicz Li] jsave=" << jsave << std::endl;
        std::cout << "[Gerbicz Li] itersave=" << itersave << std::endl;
        if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[Gerbicz Li] B=" << B << std::endl << "[Gerbicz Li] Checkpasslevel=" << checkpasslevel << std::endl << "[Gerbicz Li] j=" << totalIters-resumeIter-1 << std::endl << "[Gerbicz Li] iter=" << resumeIter << std::endl << "[Gerbicz Li] jsave=" << jsave << std::endl << "[Gerbicz Li] itersave=" << itersave << std::endl;
                    guiServer_->appendLog(oss.str());
        }
    }
    uint64_t lastJ = totalIters-resumeIter-1;
    spinner.displayProgress(resumeIter, totalIters, 0.0, 0.0, options.wagstaff ? p / 2 : p,resumeIter, resumeIter,"", guiServer_ ? guiServer_.get() : nullptr);
   
    cl_mem outOkBuf  = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate outOkBuf: " << err << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate outOkBuf: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
                    return 0;
            }
            exit(1);
    }
    cl_mem outIdxBuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS) {
            std::cerr << "Failed to allocate outIdxBuf: " << err << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Failed to allocate outIdxBuf: " << err << std::endl;
                    guiServer_->appendLog(oss.str());
                    return 0;
            }
            exit(1);
    }
    
    
    for (uint64_t iter = resumeIter, j= totalIters-resumeIter-1; iter < totalIters && !interrupted; ++iter, --j) {
        lastJ = j;
        lastIter = iter;
        if (options.erroriter > 0 && iter + 1 == options.erroriter && !errordone) {
            errordone = true;
            uint64_t limb0;
            clEnqueueReadBuffer(
                context.getQueue(),
                buffers->input,
                CL_TRUE, 0,
                sizeof(uint64_t),
                &limb0,
                0, nullptr, nullptr
            );
            limb0 += 1;
            clEnqueueWriteBuffer(
                context.getQueue(),
                buffers->input,
                CL_TRUE, 0,
                sizeof(uint64_t),
                &limb0,
                0, nullptr, nullptr
            );
            std::cout << "Injected error at iteration " << (iter + 1) << std::endl;
            if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Injected error at iteration " << (iter + 1) << std::endl;
                    guiServer_->appendLog(oss.str());
            }
        }
        
        queued += nttEngine->forward(buffers->input, iter);
        queued += nttEngine->inverse(buffers->input, iter);
        carry.carryGPU(
            buffers->input,
            buffers->blockCarryBuf,
            precompute.getN() * sizeof(uint64_t)
        );
        queued += 2;
        
        if ((options.res64_display_interval != 0)&& ( ((iter+1) % options.res64_display_interval) == 0 )) {

            std::vector<uint64_t> hostData(precompute.getN());
            std::string res64_x;  

            {
                clEnqueueReadBuffer(
                    context.getQueue(),
                    buffers->input,
                    CL_TRUE, 0,
                    hostData.size() * sizeof(uint64_t),
                    hostData.data(),
                    0, nullptr, nullptr
                );

                
                bool debug = false;
                if(debug){
                    for (size_t i = 0; i < hostData.size(); ++i)
                    {
                        std::cout << hostData[i] << " ";
                    }
                    std::cout << std::endl;
                }
                {
                    auto dataCopy       = hostData;
                    auto elapsed        = timer.elapsed();
                    auto transformSize  = static_cast<int>(context.getTransformSize());
                    auto optsCopy       = options;
                    auto digitWidth     = precompute.getDigitWidth();
                    auto iterCopy       = iter;

                    std::thread([dataCopy, optsCopy, digitWidth, elapsed, transformSize, iterCopy]() {
                        auto localRes64 = io::JsonBuilder::computeRes64Iter(
                            dataCopy,
                            optsCopy,
                            digitWidth,
                            elapsed,
                            transformSize
                        );
                        std::cout << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                        /*if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                                guiServer_->appendLog(oss.str());
                        }*/
                    }).detach();
                }


            }    
        }
        auto now = high_resolution_clock::now();
          
        if ((options.iterforce > 0 && (iter+1)%options.iterforce == 0 && iter>0) || (((iter+1)%options.iterforce == 0))) { 
            
            if((iter+1)%1000000000 != 0){
                char dummy;
                clEnqueueReadBuffer(
                        context.getQueue(),
                        buffers->input,
                        CL_TRUE, 0,
                        sizeof(dummy),
                        &dummy,
                        0, nullptr, nullptr
                    );
            }
            else{
                std::vector<uint64_t> hostData(precompute.getN());
                std::string res64_x;  
                clEnqueueReadBuffer(
                    context.getQueue(),
                    buffers->input,
                    CL_TRUE, 0,
                    hostData.size() * sizeof(uint64_t),
                    hostData.data(),
                    0, nullptr, nullptr
                );

                
                bool debug = false;
                if(debug){
                    for (size_t i = 0; i < hostData.size(); ++i)
                    {
                        std::cout << hostData[i] << " ";
                    }
                    std::cout << std::endl;
                }
                {
                    auto dataCopy       = hostData;
                    auto elapsed        = timer.elapsed();
                    auto transformSize  = static_cast<int>(context.getTransformSize());
                    auto optsCopy       = options;
                    auto digitWidth     = precompute.getDigitWidth();
                    auto iterCopy       = iter;

                    std::thread([dataCopy, optsCopy, digitWidth, elapsed, transformSize, iterCopy]() {
                        auto localRes64 = io::JsonBuilder::computeRes64Iter(
                            dataCopy,
                            optsCopy,
                            digitWidth,
                            elapsed,
                            transformSize
                        );
                        std::cout << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                        /*if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                                guiServer_->appendLog(oss.str());
                        }*/
                    }).detach();
                }


            } 
            
            
            if ((((now - lastDisplay >= seconds(10)))) ) {
                std::string res64_x;

                spinner.displayProgress(
                    iter+1,
                    totalIters,
                    timer.elapsed(),
                    timer2.elapsed(),
                    options.wagstaff ? p / 2 : p,
                    resumeIter,
                    startIter,
                    res64_x, 
                    guiServer_ ? guiServer_.get() : nullptr
                );
                timer2.start();
                lastDisplay = now;
                resumeIter = iter+1;
            }
            queued = 0;
                
        }
        
        
        if (options.mode == "ll") {
            kernels->runSub2(buffers->input);
        }


        if (options.proof && iter + 1 < totalIters) {
            proofManager.checkpoint(buffers->input, iter + 1);
        }

        if (options.mode == "prp" && options.gerbiczli && ((j != 0 && (j % B == 0)) || iter == totalIters - 1)) {
            // See: An Efficient Modular Exponentiation Proof Scheme, 
            //§2, Darren Li, Yves Gallot, https://arxiv.org/abs/2209.15623
            auto printLine = [&](cl_mem& bufz, const std::string& name) {
                std::vector<uint64_t> buf(precompute.getN());
                clEnqueueReadBuffer(context.getQueue(), bufz, CL_TRUE, 0, limbBytes, buf.data(), 0, nullptr, nullptr);
                std::ostringstream oss;
                oss << name << ": ";
                for (size_t idx = 0; idx < buf.size(); ++idx) {
                    oss << buf[idx];
                    if (idx + 1 < buf.size()) oss << ", ";
                }
                std::cout << oss.str() << std::endl;
                if (guiServer_) {
                                //std::ostringstream oss;
                                //oss << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                                guiServer_->appendLog(oss.str());
                }
            };

            checkpass += 1;
            bool condcheck = !(checkpass != checkpasslevel && (iter != totalIters - 1));
            nttEngine->copy(buffers->input, buffers->save, limbBytes);
            if (condcheck) nttEngine->copy(buffers->bufd, buffers->r2, limbBytes);
            nttEngine->forward_simple(buffers->bufd, 0);
            nttEngine->forward_simple(buffers->save, 0);
            nttEngine->pointwiseMul(buffers->bufd, buffers->save);
            nttEngine->inverse_simple(buffers->bufd, 0);
            carry.carryGPU(buffers->bufd, buffers->blockCarryBuf, limbBytes);

            if (condcheck) {
                nttEngine->copy(buffers->input, buffers->save, limbBytes);

                checkpass = 0;
                nttEngine->copy(buffers->r2, buffers->input, limbBytes);
                for (uint64_t z = 0; z < B - ((options.exponent % B == 0 ? B : options.exponent % B)); ++z) {
                    nttEngine->forward(buffers->input, iter);
                    nttEngine->inverse(buffers->input, iter);
                    carry.carryGPU(
                        buffers->input,
                        buffers->blockCarryBuf,
                        precompute.getN() * sizeof(uint64_t)
                    );
                }
                carry.carryGPU3(
                    buffers->input,
                    buffers->blockCarryBuf,
                    precompute.getN() * sizeof(uint64_t)
                );
                for (uint64_t z = 0; z < ((options.exponent % B == 0 ? B : options.exponent % B)); ++z) {
                    nttEngine->forward(buffers->input, iter);
                    nttEngine->inverse(buffers->input, iter);
                    carry.carryGPU(
                        buffers->input,
                        buffers->blockCarryBuf,
                        precompute.getN() * sizeof(uint64_t)
                    );
                }

                cl_uint ok = 1u;
                clEnqueueWriteBuffer(context.getQueue(), outOkBuf,  CL_FALSE, 0, sizeof(ok),  &ok,  0, nullptr, nullptr);
                
                kernels->runCheckEqual(
                    buffers->bufd,
                    buffers->input,
                    outOkBuf,
                    static_cast<cl_uint>(precompute.getN())
                );

                cl_event okEvt;
                clEnqueueReadBuffer(context.getQueue(), outOkBuf, CL_FALSE, 0, sizeof(ok), &ok, 0, nullptr, &okEvt);
                clWaitForEvents(1, &okEvt);

                if (ok == 1u) {
                    std::cout << "[Gerbicz Li] Check passed! iter=" << iter << "\n";
                    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "[Gerbicz Li] Check passed! iter=" << iter << "\n";
                                guiServer_->appendLog(oss.str());
                    }
                    nttEngine->copy(buffers->save, buffers->input, limbBytes);
                    nttEngine->copy(buffers->save, buffers->last_correct_state, limbBytes);
                    nttEngine->copy(buffers->bufd, buffers->last_correct_bufd, limbBytes);
                    itersave = iter;
                    jsave = j;
                    cl_event postEvt;
                    clEnqueueMarkerWithWaitList(context.getQueue(), 0, nullptr, &postEvt);
                    clWaitForEvents(1, &postEvt);
                } else {
                    std::cout << "[Gerbicz Li] Mismatch \n"
                            << "[Gerbicz Li] Check FAILED! iter=" << iter << "\n"
                            << "[Gerbicz Li] Restore iter=" << itersave << " (j=" << jsave << ")\n";
                    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "[Gerbicz Li] Mismatch \n"
                            << "[Gerbicz Li] Check FAILED! iter=" << iter << "\n"
                            << "[Gerbicz Li] Restore iter=" << itersave << " (j=" << jsave << ")\n";
                                guiServer_->appendLog(oss.str());
                    }
                    j = jsave;
                    iter = itersave;
                    lastIter = itersave;
                    lastIter = iter;
                    if (iter == 0) {
                        iter = iter - 1;
                        j = j + 1;
                    }
                    checkpass = 0;
                    options.gerbicz_error_count += 1;
                    nttEngine->copy(buffers->last_correct_state, buffers->input, limbBytes);
                    nttEngine->copy(buffers->last_correct_bufd, buffers->bufd, limbBytes);
                    cl_event postEvt;
                    clEnqueueMarkerWithWaitList(context.getQueue(), 0, nullptr, &postEvt);
                    clWaitForEvents(1, &postEvt);
                }
            }
        }




        if ((now - lastBackup >= seconds(options.backup_interval))) {
                std::string res64_x;
                backupManager.saveState(buffers->input, iter);
                backupManager.saveGerbiczLiState(buffers->last_correct_state ,buffers->bufd,buffers->last_correct_bufd , itersave, jsave);
                lastBackup = now;
                double backupElapsed = timer.elapsed();
                std::vector<uint64_t> hostData(precompute.getN());
                clEnqueueReadBuffer(
                    context.getQueue(),
                    buffers->input,
                    CL_TRUE, 0,
                    hostData.size() * sizeof(uint64_t),
                    hostData.data(),
                    0, nullptr, nullptr
                );
                {
                    auto dataCopy       = hostData;
                    auto elapsed        = timer.elapsed();
                    auto transformSize  = static_cast<int>(context.getTransformSize());
                    auto optsCopy       = options;
                    auto digitWidth     = precompute.getDigitWidth();
                    auto iterCopy       = iter;

                    std::thread([dataCopy, optsCopy, digitWidth, elapsed, transformSize, iterCopy]() {
                        auto localRes64 = io::JsonBuilder::computeRes64Iter(
                            dataCopy,
                            optsCopy,
                            digitWidth,
                            elapsed,
                            transformSize
                        );
                        std::cout << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                        /*if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Iter: " << iterCopy + 1 << "| Res64: " << localRes64 << std::endl;
                                guiServer_->appendLog(oss.str());
                        }*/
                    }).detach();
                }
                spinner.displayBackupInfo(
                    iter+1,
                    totalIters,
                    backupElapsed,
                    res64_x
                );
            }

        

    }
    if (outOkBuf != nullptr)  clReleaseMemObject(outOkBuf);
    if (outIdxBuf != nullptr) clReleaseMemObject(outIdxBuf);

    if (interrupted) {
        std::cout << "\nInterrupted signal received\n " << std::endl;
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nInterrupted signal received\n " << std::endl;
                                guiServer_->appendLog(oss.str());
                        }
        clFinish(queue);
        queued = 0;
        backupManager.saveState(buffers->input, lastIter);
        backupManager.saveGerbiczLiState(buffers->last_correct_state ,buffers->bufd,buffers->last_correct_bufd , itersave, jsave);
        
        std::cout << "\nInterrupted by user, state saved at iteration "
                  << lastIter << " last j = " << lastJ << std::endl;
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nInterrupted by user, state saved at iteration "
                  << lastIter << " last j = " << lastJ << std::endl;
                                guiServer_->appendLog(oss.str());
                        }
        return 0;
    }
    if (queued > 0) {
        clFinish(queue);
        queued = 0;
    }
    std::vector<uint64_t> hostData(precompute.getN());
    std::string res64_x;  
    


    {
        clEnqueueReadBuffer(
            context.getQueue(),
            buffers->input,
            CL_TRUE, 0,
            hostData.size() * sizeof(uint64_t),
            hostData.data(),
            0, nullptr, nullptr
        );

         
        carry.handleFinalCarry(hostData,
                               precompute.getDigitWidth());
        if (options.wagstaff) {
            spinner.displayProgress(
                lastIter+1,
                totalIters,
                timer.elapsed(),
                timer2.elapsed(),
                options.wagstaff ? p / 2 : p,
                resumeIter,
                startIter,
                res64_x, 
                guiServer_ ? guiServer_.get() : nullptr
            );
            mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
            mpz_class Fp = (mpz_class(1) << options.exponent/2) + 1;

            mpz_class rM = util::vectToMpz(hostData,
                                        precompute.getDigitWidth(),
                                        Mp);

            //std::cout << "\nResidue modulo (2^p-1)*(2^p+1) : " << rM << '\n';

            mpz_class rF = rM % Fp;

            //std::cout << "Residue modulo 2^p+1          : " << rF << '\n';

            bool isWagstaffPRP = (rF == 9);

            double gpuElapsed = timer.elapsed();
            std::cout << "Total GPU time: " << gpuElapsed << " seconds." << std::endl;
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Total GPU time: " << gpuElapsed << " seconds." << std::endl;
                                guiServer_->appendLog(oss.str());
            }
            if (isWagstaffPRP) {
                std::cout << "Wagstaff PRP confirmed: (2^"<< options.exponent/2 <<"+1)/3 is a probable prime.\n";
                if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Wagstaff PRP confirmed: (2^"<< options.exponent/2 <<"+1)/3 is a probable prime.\n";
                                guiServer_->appendLog(oss.str());
                }
            } else {
                std::cout << "Not a Wagstaff PRP.\n";
                if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Not a Wagstaff PRP.\n";
                                guiServer_->appendLog(oss.str());
                }
            }

            return isWagstaffPRP ? 0 : 1;
        }

        clEnqueueWriteBuffer(
            context.getQueue(),
            buffers->input,
            CL_TRUE, 0,
            hostData.size() * sizeof(uint64_t),
            hostData.data(),
            0, nullptr, nullptr
        );

        // Checkpoint the final result at iteration totalIters (after all p iterations are complete)
        if (options.proof) {
            proofManager.checkpoint(buffers->input, totalIters);
        }

        bool debug = false;
        if(debug){
            for (size_t i = 0; i < hostData.size(); ++i)
            {
                std::cout << hostData[i] << " ";
            }
            std::cout << std::endl;
        }
        res64_x = io::JsonBuilder::computeRes64(
            hostData,
            options,
            precompute.getDigitWidth(),
            timer.elapsed(),
            static_cast<int>(context.getTransformSize())
        );
        
        spinner.displayProgress(
                        lastIter+1,
                        totalIters,
                        timer.elapsed(),
                        timer2.elapsed(),
                        options.wagstaff ? p / 2 : p,
                        resumeIter,
                        startIter,
                        res64_x, 
                        guiServer_ ? guiServer_.get() : nullptr
                    );
        backupManager.saveState(buffers->input, lastIter);
        backupManager.saveGerbiczLiState(buffers->last_correct_state ,buffers->bufd,buffers->last_correct_bufd , itersave, jsave);
        
                
    }
    
    double gpuElapsed = timer.elapsed();
    std::cout << "Total GPU time: " << gpuElapsed << " seconds." << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Total GPU time: " << gpuElapsed << " seconds." << std::endl;
                                guiServer_->appendLog(oss.str());
    }
    // Generate proof file after successful completion
    if (options.proof) {
        int proofPower = static_cast<int>(options.proofPower);
        while (proofPower >= 0) {
            try {
                std::cout << "\nGenerating PRP proof file..." << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "\nGenerating PRP proof file..." << std::endl;
                    guiServer_->appendLog(oss.str());
                }
                options.proofPower = static_cast<decltype(options.proof)>(proofPower);
                auto proofFilePath = proofManager.proof(context, *nttEngine, carry, proofPower, options.verify);
                options.proofFile = proofFilePath.string();
                std::cout << "Proof file saved: " << proofFilePath << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Proof file saved: " << proofFilePath << std::endl;
                    guiServer_->appendLog(oss.str());
                }
                break;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Proof generation failed: " << e.what() << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Warning: Proof generation failed: " << e.what() << std::endl;
                    guiServer_->appendLog(oss.str());
                }
                if (proofPower == 0) break;
                --proofPower;
                std::cout << "Retrying proof generation with reduced power: " << proofPower << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Retrying proof generation with reduced power: " << proofPower << std::endl;
                    guiServer_->appendLog(oss.str());
                }
            }
        }
    }

    

    double finalElapsed = timer.elapsed();
    logger.logEnd(finalElapsed);

    std::vector<uint64_t> hostResult(precompute.getN());
    clEnqueueReadBuffer(
        context.getQueue(),
        buffers->input,
        CL_TRUE, 0,
        hostResult.size() * sizeof(uint64_t),
        hostResult.data(),
        0, nullptr, nullptr
    );


    auto [isPrime, res64, res2048] = io::JsonBuilder::computeResult(hostResult, options, precompute.getDigitWidth());

    std::string json = io::JsonBuilder::generate(
        options,
        static_cast<int>(context.getTransformSize()),
        isPrime,
        res64,
        res2048
    );


    Printer::finalReport(
        options,
        finalElapsed,
        json,
        isPrime
    );
    bool skippedSubmission = false;

    if (options.submit && !options.gui) {
        bool noAsk = options.noAsk || hasWorktodoEntry_;

        if (noAsk && options.password.empty()) {
            std::cerr << "No password provided with --noask; skipping submission.\n";
        }
        else {
            std::string response;
            std::cout << "Do you want to send the result to PrimeNet (https://www.mersenne.org) ? (y/n): ";
            std::getline(std::cin, response);
            if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
                std::cout << "Result not sent." << std::endl;
                skippedSubmission = true;
            }

            if (!skippedSubmission && options.user.empty()) {
                std::cout << "\nEnter your PrimeNet username (Don't have an account? Create one at https://www.mersenne.org): ";
                std::getline(std::cin, options.user);
            }
            if (!skippedSubmission && options.user.empty()) {
                std::cerr << "No username entered; skipping submission.\n";
                skippedSubmission = true;
            }

            if (!skippedSubmission) {
                if (!noAsk && options.password.empty()) {
                    options.password = io::CurlClient::promptHiddenPassword();
                }

                bool success = io::CurlClient::sendManualResultWithLogin(
                    json,
                    options.user,
                    options.password
                );
                if (!success) {
                    std::cerr << "Submission to PrimeNet failed\n";
                }
            }
        }
    }


    backupManager.clearState();
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, options.mode, json);
    wm.appendToResultsTxt(json);

    if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
                      guiServer_->appendLog(oss.str());
            }
            std::ifstream f(options.worktodo_path);
            std::string    l;
            bool           more = false;
            while (std::getline(f, l)) {
                if (!l.empty() && l[0] != '#') {
                    more = true;
                    break;
                }
            }
            f.close();

            if (more) {
                std::cout << "Restarting for next entry in worktodo.txt\n";
                if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
                      guiServer_->appendLog(oss.str());
                }
                restart_self(argc_, argv_);
            } else {
                std::cout << "No more entries in worktodo.txt, exiting.\n";
                if (!options.gui) {
                    std::exit(0);
                }
            }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n";
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Failed to update " << options.worktodo_path << "\n";
                      guiServer_->appendLog(oss.str());
            }
            if (!options.gui) {
                std::exit(-1);
            }
        }
    }


    return isPrime ? 0 : 1;
}





mpz_class buildE(uint64_t B1) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now(), last = t0;

    std::vector<uint8_t> sieve((B1 >> 1) + 1, 1);
    std::vector<uint64_t> primes;
    primes.reserve(B1 ? B1 / std::log(double(B1)) : 1);
    for (uint64_t i = 3; i <= B1; i += 2) {
        if (sieve[i >> 1]) {
            primes.push_back(i);
            if (uint64_t ii = uint64_t(i) * i; ii <= B1)
                for (uint64_t j = ii; j <= B1; j += (i << 1))
                    sieve[j >> 1] = 0;
        }
    }

    size_t total = primes.size();
    std::atomic<size_t> next{0}, done{0};
    unsigned th = std::thread::hardware_concurrency();
    if (!th) th = 4;
    std::vector<mpz_class> part(th, 1);
    std::vector<std::thread> workers;

    for (unsigned t = 0; t < th; ++t)
        workers.emplace_back([&, t] {
            while (true) {
                size_t idx = next.fetch_add(1);
                if (idx >= total) break;
                uint64_t p = primes[idx];
                mpz_class pw;
                mpz_set_ui(pw.get_mpz_t(), static_cast<unsigned long>(p));
                unsigned long lim1 = static_cast<unsigned long>(B1 / p);
                mpz_class limit(lim1);
                //while (pw <= limit) pw *= mpz_class(p);
                while (pw <= limit) pw *= mpz_class(static_cast<unsigned long>(p));
                part[t] *= pw;
                done.fetch_add(1, std::memory_order_relaxed);
                if (interrupted) return;
            }
        });

    mpz_class E = 1;
    mpz_class pw2 = 2;
    unsigned long lim2 = static_cast<unsigned long>(B1 / 2);
    mpz_class limit2; mpz_set_ui(limit2.get_mpz_t(), lim2);

    while (pw2 <= limit2) pw2 *= 2;
    E *= pw2;

    std::cout << "Building E:   0%  ETA  --:--:--" << std::flush;
    while (done.load() < total && !interrupted) {
        auto now = clock::now();
        if (now - last >= std::chrono::milliseconds(500)) {
            double prog = double(done.load()) / total;
            double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
            long sec = long(eta + 0.5);
            int h = int(sec / 3600), m = int((sec % 3600) / 60), s = int(sec % 60);
            std::cout << "\rBuilding E: " << std::setw(3) << int(prog * 100)
                      << "%  ETA "
                      << std::setw(2) << std::setfill('0') << h << ':'
                      << std::setw(2) << m << ':'
                      << std::setw(2) << s << std::setfill(' ')
                      << std::flush;
            last = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (auto &w : workers) w.join();
    for (auto &p : part) E *= p;
    if (interrupted) {
        std::cout << "\n\nInterrupted signal received — using partial E computed so far.\n\n";
        for (auto &w : workers)
            if (w.joinable()) w.join();

        for (auto &p : part) E *= p;
        mp_bitcnt_t bits = mpz_sizeinbase(E.get_mpz_t(), 2);
        std::cout << "\nlog2(E) ≈ " << bits << " bits" << std::endl;
        interrupted = false; 
        return E;
    }

    std::cout << "\rBuilding E: 100%  ETA  00:00:00\n";
    return E;
}




static unsigned long evenGapBound(const mpz_class& B2) {
    double ln = std::log(mpz_get_d(B2.get_mpz_t()));
    double bound = std::ceil((ln * ln) / 2.0);
    return bound < 2 ? 1 : static_cast<unsigned long>(bound);
}

static size_t primeCountApprox(const mpz_class& low, const mpz_class& high) {
    auto li = [](double x) {
        double l = std::log(x);
        return x / l + x / (l * l);
    };
    double a = mpz_get_d(low.get_mpz_t());
    double b = mpz_get_d(high.get_mpz_t());
    double diff = li(b) - li(a);
    return diff > 0.0 ? static_cast<size_t>(diff) : 0;
}



int App::runPM1Stage2() {
    using namespace std::chrono;
    bool debug = false;
    mpz_class B1(static_cast<unsigned long>(options.B1));
    mpz_class B2(static_cast<unsigned long>(options.B2));
    if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 factoring stage 2";
        guiServer_->setStatus(oss.str());
    }
    if (B2 <= B1) { std::cerr << "Stage 2 error B2 < B1.\n"; 
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Stage 2 error B2 < B1.\n";
                      guiServer_->appendLog(oss.str());
    }
    return -1; }
    
    if (debug) std::cout << "[DEBUG] Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    std::cout << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    unsigned long nbEven = evenGapBound(B2);
    size_t limbs = precompute.getN();
    size_t limbBytes = limbs * sizeof(uint64_t);

    cl_int err;
    buffers->Hbuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    clEnqueueCopyBuffer(context.getQueue(), buffers->input, buffers->Hbuf, 0, 0, limbBytes, 0, nullptr, nullptr);

    math::Carry carry(context, context.getQueue(), program->getProgram(),
                      precompute.getN(), precompute.getDigitWidth(), buffers->digitWidthMaskBuf);
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;

    buffers->evenPow.resize(nbEven, nullptr);
    std::cout << "Stage 2: Will precompute " << nbEven << " powers of H^2, H^.." << "." << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Stage 2: Will precompute " << nbEven << " powers of H^2, H^.." << "." << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    nttEngine->forward(buffers->input, 0);
    nttEngine->inverse(buffers->input, 0);
    carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);

    buffers->evenPow[0] = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    nttEngine->copy(buffers->input, buffers->evenPow[0], limbBytes);
    auto ensureEvenPow = [&](unsigned long needIdx) {
        while (buffers->evenPow.size() <= needIdx) {
            size_t kPrev = buffers->evenPow.size() - 1;
            cl_mem buf = clCreateBuffer(context.getContext(),
                                        CL_MEM_READ_WRITE,
                                        limbBytes, nullptr, &err);

            nttEngine->copy(buffers->evenPow[kPrev], buf, limbBytes);
            nttEngine->mulInPlace(buf, buffers->evenPow[0], carry, limbBytes);
            buffers->evenPow.push_back(buf);
        }
    };
    int pct = -1;
    std::cout << "Precomputing H powers: 0%" << std::flush;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Precomputing H powers..." << std::endl;
                      guiServer_->appendLog(oss.str());
    }
    for (unsigned long k = 1; k < nbEven; ++k) {
        buffers->evenPow[k] = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
        nttEngine->copy(buffers->evenPow[k - 1], buffers->evenPow[k], limbBytes);
        nttEngine->mulInPlace(buffers->evenPow[k], buffers->evenPow[0], carry, limbBytes);
        
        int newPct = int((k + 1) * 100 / nbEven);
        if (newPct > pct) { pct = newPct; std::cout << "\rPrecomputing H powers: " << pct << "%" << std::flush; }
    }
    for (unsigned long k = 0; k < nbEven; ++k) {
        
        nttEngine->copy(buffers->evenPow[k], buffers->input, limbBytes);
        nttEngine->forward_simple(buffers->input,0);
        nttEngine->copy(buffers->input, buffers->evenPow[k], limbBytes);
        
    }

    std::cout << "\rPrecomputing H powers: 100%" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\rPrecomputing H powers: 100%" << std::endl;
                      guiServer_->appendLog(oss.str());
    }
   

    buffers->Hq = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    nttEngine->copy(buffers->Hbuf, buffers->input, limbBytes);

    mpz_class p_prev;
    mpz_class p;
    mpz_nextprime(p_prev.get_mpz_t(), B1.get_mpz_t());
    p = p_prev;

    size_t bitlen = mpz_sizeinbase(p.get_mpz_t(), 2);
    for (int64_t i = static_cast<int64_t>(bitlen) - 2; i >= 0; --i) {
        nttEngine->forward(buffers->input, 0);
        nttEngine->inverse(buffers->input, 0);
        carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
        if (mpz_tstbit(p.get_mpz_t(), i)) {
            nttEngine->mulInPlace(buffers->input, buffers->Hbuf, carry, limbBytes);
        }
    }
    nttEngine->copy(buffers->input, buffers->Hq, limbBytes);

    buffers->Qbuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    std::vector<uint64_t> one(limbs, 0ULL); one[0] = 1ULL;
    clEnqueueWriteBuffer(context.getQueue(), buffers->Qbuf, CL_TRUE, 0, limbBytes, one.data(), 0, nullptr, nullptr);

    buffers->tmp = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);

    size_t     idx     = 0;
    //uint64_t resumeIdx = 0;
    uint64_t resumeIdx = backupManager.loadStatePM1S2(buffers->Hq, buffers->Qbuf, limbBytes);

    mpz_nextprime(p_prev.get_mpz_t(), B1.get_mpz_t());
    p = p_prev;
    idx = 0;
    for (; idx < resumeIdx; ++idx) {
        p_prev = p;
        mpz_nextprime(p.get_mpz_t(), p.get_mpz_t());
    }


    std::cout << "\r[DEBUG] p_prev=" << p_prev << std::endl;
    std::cout << "\r[DEBUG] p=" << p << std::endl;
    std::cout << "\r[DEBUG] idx=" << idx << std::endl;
    
    
    size_t totalPrimes = primeCountApprox(B1, B2);

    timer.start(); timer2.start();
    auto start = high_resolution_clock::now();
    auto lastDisplay = start;
    auto lastBackup  = start;

    for (; p <= B2; ++idx) {
        if (idx) {
            mpz_class d = p - p_prev;
            unsigned long idxGap = mpz_get_ui(d.get_mpz_t()) / 2 - 1;
            ensureEvenPow(idxGap);
            
            nttEngine->forward_simple(buffers->Hq, 0);
            nttEngine->pointwiseMul(buffers->Hq, buffers->evenPow[idxGap]);
            nttEngine->inverse_simple(buffers->Hq, 0);
            carry.carryGPU(buffers->Hq, buffers->blockCarryBuf, limbBytes);
            
        }

        nttEngine->copy(buffers->Hq, buffers->tmp, limbBytes);
        nttEngine->subOne(buffers->tmp);
        nttEngine->forward_simple(buffers->tmp, 0);
        nttEngine->forward_simple(buffers->Qbuf, 0);
        nttEngine->pointwiseMul(buffers->Qbuf, buffers->tmp);
        nttEngine->inverse_simple(buffers->Qbuf, 0);
        carry.carryGPU(buffers->Qbuf, buffers->blockCarryBuf, limbBytes);
        
        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            double done = static_cast<double>(idx + 1);
            double doneSinceResume = static_cast<double>(idx + 1 - resumeIdx);
            double percent = totalPrimes ? done / static_cast<double>(totalPrimes) * 100.0 : 0.0;
            double elapsedSec = duration<double>(now - start).count();
            double ips = doneSinceResume > 0 ? doneSinceResume / elapsedSec : 0.0;

            double remaining = totalPrimes > done ? static_cast<double>(totalPrimes) - done : 0.0;
            double etaSec = ips > 0.0 ? remaining / ips : 0.0;
            int days = static_cast<int>(etaSec) / 86400;
            int hours = (static_cast<int>(etaSec) % 86400) / 3600;
            int minutes = (static_cast<int>(etaSec) % 3600) / 60;
            int seconds = static_cast<int>(etaSec) % 60;
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << percent << "% | "
                      << "prime: " << p.get_ui() << " | "
                      << "Iter: " << (idx + 1) << " | "
                      << "Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s | "
                      << "IPS: " << std::fixed << std::setprecision(2) << ips << " | "
                      << "ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::endl;
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Progress: " << std::fixed << std::setprecision(2) << percent << "% | "
                      << "prime: " << p.get_ui() << " | "
                      << "Iter: " << (idx + 1) << " | "
                      << "Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s | "
                      << "IPS: " << std::fixed << std::setprecision(2) << ips << " | "
                      << "ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                      << std::endl;
                      guiServer_->appendLog(oss.str());
            }
            lastDisplay = now;
        }
        /*if (duration_cast<seconds>(now - lastBackup).count() >= 180) {
            backupManager.saveStatePM1S2(Hq, Qbuf, idx, limbBytes);
            lastBackup = now;
        }*/
        if (options.iterforce2 > 0 && (idx + 1) % options.iterforce2 == 0) {
            char dummy;
            clEnqueueReadBuffer(context.getQueue(), buffers->input, CL_TRUE, 0, sizeof(dummy), &dummy, 0, nullptr, nullptr);
        }


        p_prev = p;
        mpz_nextprime(p.get_mpz_t(), p.get_mpz_t());
        /*if (interrupted) {
            clFinish(context.getQueue());
            backupManager.saveStatePM1S2(Hq, Qbuf, idx, limbBytes);
            std::cout << "\r[DEBUG] p_prev=" << p_prev << std::endl;
            std::cout << "\r[DEBUG] p=" << p << std::endl;
             std::cout << "\r[DEBUG] idx=" << idx << std::endl;
            return 0;
        }*/
        if (interrupted) {
            clFinish(context.getQueue());
            backupManager.saveStatePM1S2(buffers->Hq, buffers->Qbuf, idx, limbBytes);
            return 0;
        }

    }

    std::vector<uint64_t> hostQ(limbs);
    clEnqueueReadBuffer(context.getQueue(), buffers->Qbuf, CL_TRUE, 0, limbBytes, hostQ.data(), 0, nullptr, nullptr);
    carry.handleFinalCarry(hostQ, precompute.getDigitWidth());
    mpz_class Q = util::vectToMpz(hostQ, precompute.getDigitWidth(), Mp);
    mpz_class g; mpz_gcd(g.get_mpz_t(), Q.get_mpz_t(), Mp.get_mpz_t());
    bool found = g != 1 && g != Mp;
    std::string filename = "stage2_result_B2_" + B2.get_str() +
                        "_p_" + std::to_string(options.exponent) + ".txt";

    if (found) {
        char* s = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B2=" + B2.get_str() + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
                      guiServer_->appendLog(oss.str());
            }
        options.knownFactors.push_back(std::string(s));
        std::free(s);
    } else {
        writeStageResult(filename, "No factor P-1 up to B2=" + B2.get_str());
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
                      guiServer_->appendLog(oss.str());
        }
    }

    /*
    if (found) {
        char* s = mpz_get_str(nullptr, 10, g);
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
        std::free(s);
    } else {
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
    }*/


    //backupManager.clearState();
    return found ? 0 : 1;
}

static constexpr uint64_t CHKSUMMOD = 4294967291ULL;

static uint32_t ecm_checksum_pminus1(uint64_t B1, uint32_t p, const mpz_class& X_raw) {
    mpz_class N = (mpz_class(1) << p) - 1;

    uint64_t n = mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD);
    uint64_t x = mpz_fdiv_ui(X_raw.get_mpz_t(), CHKSUMMOD);
    uint64_t b = B1 % CHKSUMMOD;

    uint64_t acc = (b * n) % CHKSUMMOD;
    acc = (acc * x) % CHKSUMMOD;
    return static_cast<uint32_t>(acc);
}

static std::string mpz_to_lower_hex(const mpz_class& z){
    char* s = mpz_get_str(nullptr, 16, z.get_mpz_t());
    std::string hex = s ? s : "";
    std::free(s);
    size_t i = 0; while (i + 1 < hex.size() && hex[i] == '0') ++i;
    return hex.substr(i);
}

void writeEcmResumeLine(const std::string& path,
                        uint64_t B1, uint32_t p,
                        const mpz_class& X)
{
    const uint32_t chk = ecm_checksum_pminus1(B1, p, X);

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: could not write GMP-ECM resume file to " << path << std::endl;
        return;
    }
    out << "METHOD=P-1; "
        << "B1=" << B1 << "; "
        << "N=2^" << p << "-1; "
        << "X=0x" << mpz_to_lower_hex(X) << "; "
        << "CHECKSUM=" << chk << "; "
        << "PROGRAM=PrMers; X0=0x3; Y=0x0; Y0=0x0; WHO=; TIME=;"
        << '\n';

    std::cout << "GMP-ECM resume file written to: " << path << std::endl;
}


static bool read_mers_file(const std::string& path, std::vector<uint64_t>& v) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot open " << path << " for reading\n";
        return false;
    }

    //in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(v.data()),
        static_cast<std::streamsize>(v.size() * sizeof(uint64_t)));

    std::streamsize readBytes = in.gcount();
    if (readBytes < static_cast<std::streamsize>(sizeof(uint64_t))) {
        std::cerr << "Error: file too small: " << path << std::endl;
        return false;
    }

    if (readBytes != static_cast<std::streamsize>(v.size() * sizeof(uint64_t))) {
        std::cerr << "Warning: partial read from " << path
                  << " (" << readBytes << " / " << v.size() * sizeof(uint64_t) << " bytes)\n";
    }

    return true;
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

static std::vector<uint8_t> hex_to_le_bytes_pad4(const mpz_class& X) {
    char* s = mpz_get_str(nullptr, 16, X.get_mpz_t());
    std::string hex = s ? s : "";
    std::free(s);
    if (hex.empty()) hex = "0";
    if (hex.size() & 1) hex.insert(hex.begin(), '0');
    size_t pad = (8 - (hex.size() & 7)) & 7;
    if (pad) hex.insert(0, pad, '0');
    std::vector<uint8_t> be; be.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        uint8_t b = (uint8_t)std::stoi(hex.substr(i, 2), nullptr, 16);
        be.push_back(b);
    }
    std::vector<uint8_t> le(be.rbegin(), be.rend());
    return le;
}

static void write_u32(std::ofstream& o, uint32_t v){ o.write(reinterpret_cast<const char*>(&v),4); }
static void write_i32(std::ofstream& o, int32_t v){ o.write(reinterpret_cast<const char*>(&v),4); }
static void write_u64(std::ofstream& o, uint64_t v){ o.write(reinterpret_cast<const char*>(&v),8); }
static void write_u16(std::ofstream& o, uint16_t v){ o.write(reinterpret_cast<const char*>(&v),2); }
static void write_f64(std::ofstream& o, double v){ o.write(reinterpret_cast<const char*>(&v),8); }
static void write_u8 (std::ofstream& o, uint8_t  v){ o.write(reinterpret_cast<const char*>(&v),1); }

static bool read_text_file(const std::string& path, std::string& out){
    std::ifstream f(path);
    if(!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

static bool parse_ecm_resume_line(const std::string& t, uint64_t& B1, uint32_t& p, std::string& hexX){
    size_t iB1 = t.find("B1=");
    if(iB1==std::string::npos) return false;
    size_t eB1 = t.find(';', iB1);
    if(eB1==std::string::npos) return false;
    B1 = std::stoull(t.substr(iB1+3, eB1-(iB1+3)));

    size_t iN = t.find("N=2^");
    if(iN==std::string::npos) return false;
    size_t eN = t.find('-', iN);
    if(eN==std::string::npos) return false;
    p = (uint32_t)std::stoul(t.substr(iN+4, eN-(iN+4)));

    size_t iX = t.find("X=0x");
    if(iX==std::string::npos) return false;
    size_t eX = t.find(';', iX);
    if(eX==std::string::npos) return false;
    hexX = t.substr(iX+4, eX-(iX+4));
    return true;
}

static std::vector<uint8_t> hex_to_bytes_reversed_pad8(const std::string& hex){
    std::string h = hex;
    if(h.empty()) h = "0";
    size_t pad = (8 - (h.size() & 7)) & 7;
    if(pad) h.insert(0, pad, '0');
    std::vector<uint8_t> data; data.reserve(h.size()/2);
    size_t bytes = h.size()/2;
    for(size_t i=0;i<bytes;++i){
        size_t pos = h.size() - (i+1)*2;
        uint8_t b = (uint8_t)std::stoul(h.substr(pos,2), nullptr, 16);
        data.push_back(b);
    }
    return data;
}

static uint32_t checksum_prime95_s1(uint64_t B1, const std::vector<uint8_t>& data){
    uint64_t sum32 = 0;
    for(size_t i=0;i+3<data.size();i+=4){
        uint32_t w = (uint32_t)data[i] | ((uint32_t)data[i+1]<<8) | ((uint32_t)data[i+2]<<16) | ((uint32_t)data[i+3]<<24);
        sum32 += w;
    }
    uint64_t chk64 = ((B1<<1) + 6u + (data.size()>>1) + sum32) & 0xFFFFFFFFULL;
    return (uint32_t)chk64;
}

static bool write_prime95_s1_from_bytes(const std::string& outPath, uint32_t p, uint64_t B1, const std::vector<uint8_t>& data, const std::string& date_start, const std::string& date_end){
    std::ofstream out(outPath, std::ios::binary);
    if(!out) return false;
    uint32_t chk = checksum_prime95_s1(B1, data);
    write_u32(out, 830093643u);
    write_u32(out, 8u);
    write_f64(out, 1.0);
    write_i32(out, 2);
    write_u32(out, p);
    write_i32(out, -1);
    write_u8 (out, (uint8_t)'S');
    write_u8 (out, (uint8_t)'1');
    write_u16(out, 0);
    write_u64(out, 0);
    write_f64(out, 1.0);
    write_u32(out, chk);
    write_i32(out, 5);
    write_u64(out, B1);
    write_u64(out, B1);
    write_i32(out, 1);
    write_i32(out, (int32_t)(data.size()>>2));
    out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tmv{};
    #if defined(_WIN32)
    gmtime_s(&tmv, &tt);
    #else
    std::tm* tmp = std::gmtime(&tt);
    if (tmp) tmv = *tmp;
    #endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
    std::ostringstream ts;
    ts << buf << '.' << std::setw(3) << std::setfill('0') << (int)ms.count();
    std::string ds = date_start.empty() ? ts.str() : date_start;
    std::string de = date_end.empty() ? ts.str() : date_end;

    std::string os_str;
    #if defined(_WIN32)
    os_str = "Windows";
    #elif defined(__APPLE__)
    os_str = "macOS";
    #elif defined(__linux__)
    os_str = "Linux";
    #else
    os_str = "Unknown";
    #endif

    std::string arch_str;
    #if defined(__x86_64__) || defined(_M_X64)
    arch_str = "x86_64";
    #elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    arch_str = "x86_32";
    #elif defined(__aarch64__) || defined(_M_ARM64)
    arch_str = "ARM64";
    #elif defined(__arm__) || defined(_M_ARM)
    arch_str = "ARM";
    #else
    arch_str = "unknown";
    #endif

    std::string json = std::string("{\"programs\":[{\"work\":{\"type\":\"PM1\",\"stage\":\"1\"},\"program\":{\"name\":\"prmers\",\"version\":\"" + core::PRMERS_VERSION + "\"},\"os\":{\"os\":\"") + os_str + "\",\"architecture\":\"" + arch_str + "\"},\"date_start\":\"" + ds + "\",\"date_end\":\"" + de + "\"}]}";

    auto crc32 = [](const uint8_t* d, size_t n){
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < n; ++i) {
            crc ^= d[i];
            for (int k = 0; k < 8; ++k) {
                uint32_t mask = -(crc & 1u);
                crc = (crc >> 1) ^ (0xEDB88320u & mask);
            }
        }
        return ~crc;
    };

    const char magic[16] = {'M','O','R','E','I','N','F','O','J','S','O','N','D','A','T','A'};
    uint32_t version = 1u;
    std::vector<uint8_t> json_bytes(json.begin(), json.end());
    uint32_t crc = crc32(json_bytes.data(), json_bytes.size());
    uint32_t chunk_size = 8u + (uint32_t)json_bytes.size();

    out.write(magic, 16);
    write_u32(out, chunk_size);
    write_u32(out, version);
    write_u32(out, crc);
    if (!json_bytes.empty())
        out.write(reinterpret_cast<const char*>(json_bytes.data()), (std::streamsize)json_bytes.size());

    return (bool)out;
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



int App::runPM1() {

    uint64_t B1 = options.B1;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 factoring stage 1";
        guiServer_->setStatus(oss.str());
    }
    std::cout << "Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
                      guiServer_->appendLog(oss.str());
            }
    mpz_class E = backupManager.loadExponent();
    if(E==0){    
        if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Building E....";
                      guiServer_->appendLog(oss.str());
        }
        E = buildE(B1);
        E *= mpz_class(2) * mpz_class(static_cast<unsigned long>(options.exponent));


    }
    
    //std::cout << "[DEBUG] E=" << E << std::endl;
    mp_bitcnt_t bits = mpz_sizeinbase(E.get_mpz_t(), 2);
    std::vector<uint64_t> x(precompute.getN(), 0ULL);
    uint64_t resumeIter = backupManager.loadState(x);
    if(resumeIter==0){
        x[0] = 1ULL;
        resumeIter = bits;
    }
    buffers->input = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x.size() * sizeof(uint64_t), x.data(), nullptr);

    math::Carry carry(context, context.getQueue(), program->getProgram(), precompute.getN(), precompute.getDigitWidth(), buffers->digitWidthMaskBuf);


    timer.start();
    timer2.start();
    auto startTime  = high_resolution_clock::now();
    auto lastDisplay = startTime;
    interrupted.store(false, std::memory_order_relaxed);

    uint64_t startIter = resumeIter;
    uint64_t lastIter = resumeIter;
    backupManager.saveState(buffers->input, resumeIter,&E);
    spinner.displayProgress(
                    bits-resumeIter,
                    bits,
                    timer.elapsed(),
                    timer2.elapsed(),
                    options.exponent,
                    resumeIter,
                    resumeIter,
                    "", 
                    guiServer_ ? guiServer_.get() : nullptr
                );
    auto start_sys = std::chrono::system_clock::now();
    for (mp_bitcnt_t i = resumeIter; i > 0; --i) {
        lastIter = i;
        if (interrupted) {
            std::cout << "\nInterrupted signal received\n " << std::endl;
            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nInterrupted signal received\n " << std::endl;
                      guiServer_->appendLog(oss.str());
            }
            clFinish(context.getQueue());
            backupManager.saveState(buffers->input, lastIter-1, &E);
            //backupManager.saveState(buffers->input, lastIter);
            //std::cout << "\nInterrupted by user, state saved at iteration "
            //        << lastIter << std::endl;
            return 0;
        }
        nttEngine->forward(buffers->input, 0);
        nttEngine->inverse(buffers->input, 0);
        carry.carryGPU(buffers->input, buffers->blockCarryBuf, precompute.getN() * sizeof(uint64_t));



        if (mpz_tstbit(E.get_mpz_t(), i - 1)) {
            carry.carryGPU3(buffers->input, buffers->blockCarryBuf, precompute.getN() * sizeof(uint64_t));
            
        }
        if ((options.iterforce > 0 && (i+1)%options.iterforce == 0 && i>0) || (((i+1)%options.iterforce == 0))) { 
            
            //if((i+1)%100000 != 0){

                char dummy;
                clEnqueueReadBuffer(
                        context.getQueue(),
                        buffers->input,
                        CL_TRUE, 0,
                        sizeof(dummy),
                        &dummy,
                        0, nullptr, nullptr
                    );
                
            
            //}

            
        }
        
        auto now = high_resolution_clock::now();
        if ((((now - lastDisplay >= seconds(180)))) ) {
                    backupManager.saveState(buffers->input, lastIter-1);
        }
        if ((((now - lastDisplay >= seconds(10)))) ) {
                std::string res64_x;
                

                spinner.displayProgress(
                    bits-i-1,
                    bits,
                    timer.elapsed(),
                    timer2.elapsed(),
                    options.exponent,
                    resumeIter,
                    startIter,
                    res64_x, 
                    guiServer_ ? guiServer_.get() : nullptr
                );
                timer2.start();
                lastDisplay = now;
                resumeIter = bits-i-1;
            }
        //clFinish(context.getQueue());

    }
    //return 0;
    std::string res64_x;
    spinner.displayProgress(
        bits,
        bits,
        timer.elapsed(),
        timer2.elapsed(),
        options.exponent,
        resumeIter,
        startIter,
        res64_x, 
        guiServer_ ? guiServer_.get() : nullptr
    );
    backupManager.saveState(buffers->input, lastIter, &E);


    std::cout << "\nStart get result from GPU" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "\nStart get result from GPU" << std::endl;
                      guiServer_->appendLog(oss.str());
            }
    std::vector<uint64_t> hostData(precompute.getN());

    clEnqueueReadBuffer(context.getQueue(), buffers->input, CL_TRUE, 0,
                        hostData.size() * sizeof(uint64_t), hostData.data(),
                        0, nullptr, nullptr);
    std::cout << "Handle final carry start" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "Handle final carry start" << std::endl;
                      guiServer_->appendLog(oss.str());
            }

    carry.handleFinalCarry(hostData, precompute.getDigitWidth());
    std::cout << "vectToResidue start" << std::endl;
    if (guiServer_) {
                                std::ostringstream oss;
                                oss << "vectToResidue start" << std::endl;
                      guiServer_->appendLog(oss.str());
            }
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X = util::vectToMpz(hostData, precompute.getDigitWidth(), Mp);
    auto fmt = [](const std::chrono::system_clock::time_point& tp){
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
        std::time_t tt = system_clock::to_time_t(tp);
        std::tm tmv{};
        #if defined(_WIN32)
        gmtime_s(&tmv, &tt);
        #else
        std::tm* tmp = std::gmtime(&tt);
        if (tmp) tmv = *tmp;
        #endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        std::ostringstream s;
        s << buf << '.' << std::setw(3) << std::setfill('0') << (int)(ms.count());
        return s.str();
    };
    auto end_sys = std::chrono::system_clock::now();
    
    std::string ds = fmt(start_sys);
    std::string de = fmt(end_sys);
    if(options.resume){
        
        writeEcmResumeLine("resume_p" + std::to_string(options.exponent) + "_B1_" +
                    std::to_string(options.B1) + ".save",
                    options.B1, options.exponent, X);
        convertEcmResumeToPrime95("resume_p" + std::to_string(options.exponent) + "_B1_" +
                    std::to_string(options.B1) + ".save","resume_p" + std::to_string(options.exponent) + "_B1_" +
                    std::to_string(options.B1) + ".p95", ds, de);
                    
    }
    
    //std::cout << "digitWidths = ";
    //for (int w : precompute.getDigitWidth()) std::cout << w << " ";
    //std::cout << "\n";
    //gmp_printf("X final  = %Zd\n", X);

    X -= 1;

    mpz_class g;
    mpz_gcd(g.get_mpz_t(), X.get_mpz_t(), Mp.get_mpz_t());


/*
    mpz_t Mp;  mpz_init(Mp);
    mpz_ui_pow_ui(Mp, 2, options.exponent);
    mpz_sub_ui(Mp, Mp, 1);

    mpz_t r;   mpz_init(r);
    vectToResidue(r, hostData, precompute.getDigitWidth(), Mp);
    std::cout << "\n GCD start\n" << std::endl;

    mpz_sub_ui(r, r, 1);
    mpz_t g;   mpz_init(g);
    mpz_gcd(g, r, Mp);*/

    //gmp_printf("GCD(x - 1, 2^%u - 1) = %Zd\n", options.exponent, g);

    bool factorFound = g != 1 && g != Mp;

    std::string filename = "stage1_result_B1_" + std::to_string(B1) +
                        "_p_" + std::to_string(options.exponent) + ".txt";

    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B1=" + std::to_string(B1) + "  factor=" + std::string(fstr));
        std::free(fstr);
    } else {
        writeStageResult(filename, "No factor up to B1=" + std::to_string(B1));
    }


    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
        std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
        if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "\nP-1 factor stage 1 found: " << fstr << std::endl;
                      guiServer_->appendLog(oss.str());
            }
        options.knownFactors.push_back(std::string(fstr));
        std::free(fstr);
        std::cout << "\n";
        if(options.B2>0){
            runPM1Stage2();
        }
//        else{
            //backupManager.clearState();
//        }
        //return 0;
    }
    else{
        std::cout << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
        if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
                      guiServer_->appendLog(oss.str());
            }
        if(options.B2>0){
            runPM1Stage2();
        }
    }
/*    else{
            backupManager.clearState();
    }
    backupManager.clearState();*/
    std::string json = io::JsonBuilder::generate(
        options,
        static_cast<int>(context.getTransformSize()),
        false,
        "",
        ""
    );
    std::cout << "Manual submission JSON:\n" << json << "\n";
   /*if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "Manual submission JSON:\n" << json << "\n";
                      guiServer_->appendLog(oss.str());
            }*/
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, options.mode, json);
    wm.appendToResultsTxt(json);
    
     if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
            if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
                      guiServer_->appendLog(oss.str());
            }
            std::ifstream f(options.worktodo_path);
            std::string    l;
            bool           more = false;
            while (std::getline(f, l)) {
                if (!l.empty() && l[0] != '#') {
                    more = true;
                    break;
                }
            }
            f.close();

            if (more) {
                std::cout << "Restarting for next entry in worktodo.txt\n";
                if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
                      guiServer_->appendLog(oss.str());
                }
                restart_self(argc_, argv_);
            } else {
                std::cout << "No more entries in worktodo.txt, exiting.\n";
                if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "No more entries in worktodo.txt, exiting.\n";
                      guiServer_->appendLog(oss.str());
                }
                if (!options.gui) {
                    std::exit(0);
                }
            }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n";
            if (guiServer_) {
                                std::ostringstream oss;
                                oss  << "Failed to update " << options.worktodo_path << "\n";
                      guiServer_->appendLog(oss.str());
                }
            if (!options.gui) {
                std::exit(-1);
            }
        }
    }
    return 1;
}

mpz_class gcd_with_dots(const mpz_class& A, const mpz_class& B) {
    std::atomic<bool> done{false};
    std::thread ticker([&]{
        using namespace std::chrono;
        const char* msg = "Computing GCD (this may take a while) ";
        std::cout << msg << std::flush;
        size_t dots = 0, wrap = 60;
        while (!done.load(std::memory_order_relaxed)) {
            std::cout << '.' << std::flush;
            if (++dots % wrap == 0) std::cout << '\n' << msg << std::flush;
            std::this_thread::sleep_for(milliseconds(300));
        }
        std::cout << " done.\n";
    });
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), A.get_mpz_t(), B.get_mpz_t());
    done.store(true, std::memory_order_relaxed);
    ticker.join();
    return g;
}

#include <atomic>
#include <thread>
#include <chrono>

static mpz_class compute_X_with_dots(engine* eng, engine::Reg reg, const mpz_class& Mp) {
    using namespace std::chrono;

    std::atomic<bool> done{false};
    std::thread ticker([&]{
        const char* msg = "Constructing and reducing large integer ";
        std::cout << msg << std::flush;
        size_t dots = 0, wrap = 60;
        while (!done.load(std::memory_order_relaxed)) {
            std::cout << '.' << std::flush;
            if (++dots % wrap == 0) std::cout << '\n' << msg << std::flush;
            std::this_thread::sleep_for(milliseconds(300));
        }
        std::cout << " done.\n";
    });

    mpz_class X(0);
    mpz_t tmp; mpz_init(tmp);
    eng->get_mpz(tmp, reg);
    mpz_set(X.get_mpz_t(), tmp);
    mpz_clear(tmp);

    X %= Mp;

    done.store(true, std::memory_order_relaxed);
    ticker.join();
    return X;
}



static mpz_class product_tree_range_u64(const std::vector<uint64_t>& v, size_t lo, size_t hi, size_t leaf, int par) {
    size_t n = hi - lo;
    if (n == 0) return mpz_class(1);
    if (n <= leaf) {
        mpz_class r = 1;
        for (size_t i = lo; i < hi; ++i) {
            mpz_class t; mpz_set_ui(t.get_mpz_t(), (unsigned long)v[i]);
            r *= t;
        }
        return r;
    }
    size_t mid = lo + (n >> 1);
    if (par > 1) {
        auto fut = std::async(std::launch::async, [&]{ return product_tree_range_u64(v, lo, mid, leaf, par >> 1); });
        mpz_class right = product_tree_range_u64(v, mid, hi, leaf, par - (par >> 1));
        mpz_class left = fut.get();
        return left * right;
    } else {
        mpz_class left  = product_tree_range_u64(v, lo, mid, leaf, 1);
        mpz_class right = product_tree_range_u64(v, mid, hi, leaf, 1);
        return left * right;
    }
}

static size_t product_prefix_fit_u64(const std::vector<uint64_t>& v, size_t lo, size_t hi, const mpz_class& Ecur, uint64_t maxBits, mpz_class& outProd, size_t leaf, int par) {
    mpz_class P = product_tree_range_u64(v, lo, hi, leaf, par);
    mpz_class Etmp = Ecur * P;
    if (mpz_sizeinbase(Etmp.get_mpz_t(), 2) <= maxBits) { outProd = P; return hi; }
    if (hi - lo == 1) { outProd = 1; return lo; }
    size_t mid = lo + ((hi - lo) >> 1);
    mpz_class Pleft;
    size_t k = product_prefix_fit_u64(v, lo, mid, Ecur, maxBits, Pleft, leaf, par > 1 ? (par >> 1) : 1);
    if (k == mid) {
        mpz_class E2 = Ecur * Pleft;
        mpz_class Pright;
        size_t k2 = product_prefix_fit_u64(v, mid, hi, E2, maxBits, Pright, leaf, par > 1 ? (par - (par >> 1)) : 1);
        outProd = Pleft * Pright;
        return k2;
    } else {
        outProd = Pleft;
        return k;
    }
}

mpz_class buildE2(uint64_t B1, uint64_t startPrime, uint64_t maxBits, uint64_t& nextStart, bool includeTwo) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now(), last = t0;
    nextStart = 0;
    if (B1 < 3) return includeTwo ? mpz_class(2) : mpz_class(1);

    uint64_t s = startPrime < 3 ? 3 : (startPrime | 1ULL);
    uint64_t R = (uint64_t)std::sqrt((long double)B1);
    if ((R & 1ULL) == 0) R -= 1;

    std::vector<uint8_t> base((R >> 1) + 1, 1);
    for (uint64_t i = 3; i * i <= R; i += 2)
        if (base[i >> 1])
            for (uint64_t j = i * i; j <= R; j += (i << 1))
                base[j >> 1] = 0;

    std::vector<uint64_t> small_primes;
    for (uint64_t i = 3; i <= R; i += 2)
        if (base[i >> 1]) small_primes.push_back(i);

    mpz_class E = 1;
    std::vector<uint64_t> batch;
    std::vector<uint64_t> batch_primes;
    batch.reserve(1u << 16);
    batch_primes.reserve(1u << 16);

    if (includeTwo) {
        uint64_t pw2 = 2;
        while (pw2 <= B1 / 2) pw2 <<= 1;
        batch.push_back(pw2);
        batch_primes.push_back(2);
    }

    const uint64_t span = 1ULL << 24;
    if ((s & 1ULL) == 0) s += 1;
    uint64_t totalOdd = (B1 >= s) ? (((B1 - s) >> 1) + 1) : 0;
    std::cout << "Building E-chunk:   0%  ETA  --:--:--" << std::flush;

    auto flush_batch = [&](bool final_segment)->bool{
        if (batch.empty()) return true;
        size_t leaf = 16;
        int par = (int)std::thread::hardware_concurrency(); if (par <= 0) par = 2;
        mpz_class Pfit;
        size_t used = product_prefix_fit_u64(batch, 0, batch.size(), E, maxBits, Pfit, leaf, par);
        E *= Pfit;
        if (used < batch.size() && mpz_cmp_ui(E.get_mpz_t(), 1) != 0) { nextStart = batch_primes[used]; return false; }
        batch.clear();
        batch_primes.clear();
        if (final_segment && nextStart == 0) std::cout << "\rBuilding E-chunk: 100%  ETA  00:00:00\n";
        return true;
    };

    uint64_t low = s;
    while (low <= B1 && !interrupted) {
        uint64_t high = low + span - 1;
        if (high > B1) high = B1;
        if ((high & 1ULL) == 0) high -= 1;
        if (high < low) break;

        size_t len = size_t(((high - low) >> 1) + 1);
        std::vector<uint8_t> seg(len, 1);

        for (uint64_t q : small_primes) {
            uint64_t q2 = q * q;
            uint64_t start = (q2 > low) ? q2 : ((low + q - 1) / q) * q;
            if ((start & 1ULL) == 0) start += q;
            if (start < low) start += q;
            for (uint64_t j = start; j <= high; j += (q << 1)) {
                size_t idx = size_t((j - low) >> 1);
                seg[idx] = 0;
            }
        }

        for (uint64_t n = low; n <= high; n += 2) {
            if (!seg[size_t((n - low) >> 1)]) {
                auto now = clock::now();
                if (now - last >= std::chrono::milliseconds(500)) {
                    uint64_t progressedOdd = ((n >= s) ? (((n - s) >> 1) + 1) : 0);
                    double prog = totalOdd ? (double)progressedOdd / (double)totalOdd : 1.0;
                    double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
                    long sec = long(eta + 0.5);
                    int h = int(sec / 3600), m = int((sec % 3600) / 60), ss = int(sec % 60);
                    std::cout << "\rBuilding E-chunk: " << std::setw(3) << int(prog * 100)
                              << "%  ETA "
                              << std::setw(2) << std::setfill('0') << h << ':'
                              << std::setw(2) << m << ':'
                              << std::setw(2) << ss << std::setfill(' ')
                              << std::flush;
                    last = now;
                }
                continue;
            }
            uint64_t p = n;
            uint64_t pw = p;
            while (pw <= B1 / p) pw *= p;
            batch.push_back(pw);
            batch_primes.push_back(p);

            if (batch.size() >= (1u << 16)) {
                if (!flush_batch(false)) goto done;
            }

            auto now = clock::now();
            if (now - last >= std::chrono::milliseconds(500)) {
                uint64_t progressedOdd = ((n >= s) ? (((n - s) >> 1) + 1) : 0);
                double prog = totalOdd ? (double)progressedOdd / (double)totalOdd : 1.0;
                double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
                long sec = long(eta + 0.5);
                int h = int(sec / 3600), m = int((sec % 3600) / 60), ss = int(sec % 60);
                std::cout << "\rBuilding E-chunk: " << std::setw(3) << int(prog * 100)
                          << "%  ETA "
                          << std::setw(2) << std::setfill('0') << h << ':'
                          << std::setw(2) << m << ':'
                          << std::setw(2) << ss << std::setfill(' ')
                          << std::flush;
                last = now;
            }
            if (interrupted) break;
        }

        if (!flush_batch(false)) goto done;

        low = high + 2;
        auto now = clock::now();
        uint64_t progressedOdd = ((low > s) ? (((std::min(low - 2, B1) - s) >> 1) + 1) : 0);
        double prog = totalOdd ? (double)progressedOdd / (double)totalOdd : 1.0;
        double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
        long sec = long(eta + 0.5);
        int h = int(sec / 3600), m = int((sec % 3600) / 60), ss = int(sec % 60);
        std::cout << "\rBuilding E-chunk: " << std::setw(3) << int(prog * 100)
                  << "%  ETA "
                  << std::setw(2) << std::setfill('0') << h << ':'
                  << std::setw(2) << m << ':'
                  << std::setw(2) << ss << std::setfill(' ')
                  << std::flush;
    }

done:
    if (!batch.empty() && nextStart == 0) flush_batch(true);
    if (interrupted) {
        std::cout << "\n\nInterrupted signal received — using partial E computed so far.\n\n";
        mp_bitcnt_t bits = mpz_sizeinbase(E.get_mpz_t(), 2);
        std::cout << "\nlog2(E) ≈ " << bits << " bits" << std::endl;
        interrupted = false;
        return E;
    }
    if (nextStart == 0) std::cout << "\rBuilding E-chunk: 100%  ETA  00:00:00\n";
    return E;
}

int App::runPM1Stage2Marin() {
    using namespace std::chrono;
    if (guiServer_) { std::ostringstream oss; oss << "P-1 factoring stage 2"; guiServer_->setStatus(oss.str()); }
    bool debug = options.debug;
    uint64_t B1u = options.B1, B2u = options.B2;
    mpz_class B1(static_cast<unsigned long>(B1u)), B2(static_cast<unsigned long>(B2u));
    if (B2 <= B1) {
        std::cerr << "Stage 2 error B2 < B1.\n";
        if (guiServer_) { std::ostringstream oss; oss << "Stage 2 error B2 < B1.\n"; guiServer_->appendLog(oss.str()); }
        return -1;
    }
    std::cout << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl; guiServer_->appendLog(oss.str()); }
    uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = options.debug;
    const size_t baseRegs = 11;
    const size_t RSTATE=0, RACC_L=1, RACC_R=2, RCHK=3, RPOW=4, RTMP=5;
    std::ostringstream ck2; ck2 << "pm1_s2_m_" << pexp << ".ckpt";
    const std::string ckpt_file_s2 = ck2.str();
    auto read_ckpt_s2 = [&](engine* e, const std::string& file, uint64_t& saved_p, uint64_t& saved_idx, double& et, uint64_t& sB1, uint64_t& sB2)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 1) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB1), sizeof(sB1))) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB2), sizeof(sB2))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_p), sizeof(saved_p))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_idx), sizeof(saved_idx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    auto save_ckpt_s2 = [&](engine* e, uint64_t cur_p, uint64_t cur_idx, double et){
        const std::string oldf = ckpt_file_s2 + ".old", newf = ckpt_file_s2 + ".new";
        {
            File f(newf, "wb");
            int version = 1;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&pexp), sizeof(pexp))) return;
            if (!f.write(reinterpret_cast<const char*>(&B1u), sizeof(B1u))) return;
            if (!f.write(reinterpret_cast<const char*>(&B2u), sizeof(B2u))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_p), sizeof(cur_p))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_idx), sizeof(cur_idx))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!e->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file_s2.c_str(), &s) == 0) && (std::rename(ckpt_file_s2.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file_s2.c_str());
    };
    unsigned long nbEven = evenGapBound(B2);
    if (nbEven == 0) nbEven = 1;
    size_t regCount = baseRegs + nbEven + 2;
    engine* eng = engine::create_gpu(pexp, regCount, static_cast<size_t>(options.device_id), verbose);
    const size_t REVEN = baseRegs;
    const size_t RSAVE_Q = baseRegs + nbEven;
    const size_t RSAVE_HQ = baseRegs + nbEven + 1;
    uint64_t resume_idx = 0;
    uint64_t resume_p_u64 = 0;
    double restored_time = 0.0;
    uint64_t s2B1=0, s2B2=0;
    int rs2 = read_ckpt_s2(eng, ckpt_file_s2, resume_p_u64, resume_idx, restored_time, s2B1, s2B2);
    bool resumed_s2 = (rs2 == 0) && (s2B1 == B1u) && (s2B2 == B2u);
    if (!resumed_s2) {
        engine* eng_load = engine::create_gpu(pexp, baseRegs, static_cast<size_t>(options.device_id), verbose);
        std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
        const std::string ckpt_file = ck.str();
        auto read_ckpt = [&](engine* e, const std::string& file)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 3) return -2;
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != pexp) return -2;
            uint32_t ri = 0; double et = 0.0;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!e->set_checkpoint(data)) return -2;
            uint64_t tmp64;
            if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
            if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
            if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
            if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
            uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
            uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
            if (eacc_len) { std::string skip; skip.resize(eacc_len); if (!f.read(skip.data(), eacc_len)) return -2; }
            uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
            if (wbits_len) { std::string skip; skip.resize(wbits_len); if (!f.read(skip.data(), wbits_len)) return -2; }
            uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
            if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
            if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
            if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
            if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
            if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
            if (!f.check_crc32()) return -2;
            return 0;
        };
        int rr = read_ckpt(eng_load, ckpt_file);
        if (rr < 0) rr = read_ckpt(eng_load, ckpt_file + ".old");
        if (rr != 0) { delete eng_load; delete eng; std::cerr << "Stage 2: cannot load pm1 stage1 checkpoint.\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 2: cannot load pm1 stage1 checkpoint.\n"; guiServer_->appendLog(oss.str()); } return -2; }
        mpz_t H; mpz_init(H); eng_load->get_mpz(H, static_cast<engine::Reg>(RSTATE)); delete eng_load;
        eng->set_mpz(static_cast<engine::Reg>(RSTATE), H);
        mpz_clear(H);
        std::cout << "Precomputing H even powers..." << std::endl;
        if (guiServer_) { std::ostringstream oss; oss << "Precomputing H even powers..."; guiServer_->appendLog(oss.str()); }
        eng->copy(static_cast<engine::Reg>(REVEN + 0), static_cast<engine::Reg>(RSTATE));
        eng->square_mul(static_cast<engine::Reg>(REVEN + 0));
        int pct = -1;
        for (unsigned long k = 1; k < nbEven; ++k) {
            eng->copy(static_cast<engine::Reg>(REVEN + k), static_cast<engine::Reg>(REVEN + k - 1));
            eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(REVEN + 0));
            eng->mul(static_cast<engine::Reg>(REVEN + k), static_cast<engine::Reg>(RTMP));
            int newPct = int((k + 1) * 100 / nbEven);
            if (newPct > pct) {
                pct = newPct;
                std::cout << "\rPrecomputing H powers: " << pct << "%" << std::flush;
                if (guiServer_) { std::ostringstream oss; oss << "Precomputing H powers: " << pct << "%"; guiServer_->appendLog(oss.str()); }
            }
        }
        for (unsigned long k = 0; k < nbEven; ++k) {
            eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(REVEN + k));
            eng->copy(static_cast<engine::Reg>(REVEN + k), static_cast<engine::Reg>(RTMP));
        }
        std::cout << "\n";
        eng->set(static_cast<engine::Reg>(RACC_L), 1);
    }
    auto t0 = high_resolution_clock::now();
    auto start_clock = high_resolution_clock::now();
    auto lastBackup = start_clock;
    auto lastDisplay = start_clock;
    mpz_class p_prev; mpz_nextprime(p_prev.get_mpz_t(), B1.get_mpz_t());
    mpz_class p = p_prev;
    uint64_t idx = 0;
    if (resumed_s2) {
        mpz_set_ui(p.get_mpz_t(), static_cast<unsigned long>(resume_p_u64));
        idx = resume_idx;
        std::cout << "Resuming Stage 2 from checkpoint at prime " << p.get_ui() << " (idx=" << idx << ")\n";
        if (guiServer_) { std::ostringstream oss; oss << "Resuming Stage 2 from checkpoint at prime " << p.get_ui() << " (idx=" << idx << ")"; guiServer_->appendLog(oss.str()); }
        t0 = high_resolution_clock::now() - duration_cast<high_resolution_clock::duration>(duration<double>(restored_time));
        start_clock = high_resolution_clock::now();
        lastBackup = start_clock;
        lastDisplay = start_clock;
    } else {
        mpz_class p0 = p_prev;
        eng->pow(static_cast<engine::Reg>(RACC_R), static_cast<engine::Reg>(RSTATE), mpz_get_ui(p0.get_mpz_t()));
        if (debug) {
            mpz_t zh, zhq, zq; mpz_inits(zh, zhq, zq, nullptr);
            eng->get_mpz(zh, static_cast<engine::Reg>(RSTATE));
            eng->get_mpz(zhq, static_cast<engine::Reg>(RACC_R));
            eng->get_mpz(zq, static_cast<engine::Reg>(RACC_L));
            std::cout << "[DEBUG S2] H=" << mpz_class(zh) << std::endl;
            std::cout << "[DEBUG S2] p0=" << p0.get_ui() << std::endl;
            std::cout << "[DEBUG S2] H^p0=" << mpz_class(zhq) << std::endl;
            std::cout << "[DEBUG S2] Q0=" << mpz_class(zq) << std::endl;
            mpz_clears(zh, zhq, zq, nullptr);
        }
    }
    size_t totalPrimes = primeCountApprox(B1, B2);
    auto start = high_resolution_clock::now();
    uint64_t primes_since_check = 0;
    uint64_t checkpasslevel = options.checklevel > 0 ? options.checklevel : std::max<uint64_t>(1, (uint64_t)std::sqrt((double)std::max<size_t>(1, totalPrimes)));
    eng->copy(static_cast<engine::Reg>(RSAVE_Q), static_cast<engine::Reg>(RACC_L));
    eng->copy(static_cast<engine::Reg>(RSAVE_HQ), static_cast<engine::Reg>(RACC_R));
    mpz_class blockStartP = p;
    auto start_sys = std::chrono::system_clock::now();

    for (;; ++idx) {
        if (p > B2) break;
        eng->copy(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RACC_R));
        eng->sub(static_cast<engine::Reg>(RTMP), 1);
        eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RTMP));
        eng->mul(static_cast<engine::Reg>(RACC_L), static_cast<engine::Reg>(RPOW));
        mpz_class nextp = p; mpz_nextprime(nextp.get_mpz_t(), nextp.get_mpz_t());
        if (nextp > B2) { ++idx; break; }
        mpz_class dgap = nextp - p;
        uint64_t gap = mpz_get_ui(dgap.get_mpz_t());
        uint64_t idxGap = (gap >> 1) - 1;
        //eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(REVEN + idxGap));
        eng->mul(static_cast<engine::Reg>(RACC_R), static_cast<engine::Reg>(REVEN + idxGap));
        p = nextp;
        ++primes_since_check;
        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            double done = static_cast<double>(idx + 1);
            double percent = totalPrimes ? done / static_cast<double>(totalPrimes) * 100.0 : 0.0;
            double elapsedSec = duration<double>(now - start).count();
            double ips = done > 0 ? done / elapsedSec : 0.0;
            double remaining = totalPrimes > done ? static_cast<double>(totalPrimes) - done : 0.0;
            double etaSec = ips > 0.0 ? remaining / ips : 0.0;
            int days = static_cast<int>(etaSec) / 86400;
            int hours = (static_cast<int>(etaSec) % 86400) / 3600;
            int minutes = (static_cast<int>(etaSec) % 3600) / 60;
            int seconds = static_cast<int>(etaSec) % 60;
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << percent << "% | prime: " << p.get_ui() << " | Iter: " << (idx + 1) << " | Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s | IPS: " << std::fixed << std::setprecision(2) << ips << " | ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r" << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Progress: " << std::fixed << std::setprecision(2) << percent << "% | prime: " << p.get_ui() << " | Iter: " << (idx + 1) << " | Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s | IPS: " << std::fixed << std::setprecision(2) << ips << " | ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"; guiServer_->appendLog(oss.str()); 
                              guiServer_->setProgress(done, totalPrimes, "");
            }
            lastDisplay = now;
        }
        auto now0 = high_resolution_clock::now();
        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now0 - t0).count();
            std::cout << "\nBackup Stage 2 at prime " << p.get_ui() << " idx=" << idx << " start...\n";
            save_ckpt_s2(eng, static_cast<uint64_t>(p.get_ui()), idx, et);
            lastBackup = now0;
            std::cout << "Backup Stage 2 done.\n";
            if (guiServer_) { std::ostringstream oss; oss << "Backup Stage 2 at prime " << p.get_ui() << " idx=" << idx; guiServer_->appendLog(oss.str()); }
        }
        if (options.iterforce2 > 0 && (idx + 1) % options.iterforce2 == 0) { eng->copy(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RSTATE)); }
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count();
            save_ckpt_s2(eng, static_cast<uint64_t>(p.get_ui()), idx, et);
            delete eng;
            std::cout << "\nInterrupted by user, Stage 2 state saved at prime " << p.get_ui() << " idx=" << idx << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "\nInterrupted by user, Stage 2 state saved at prime " << p.get_ui() << " idx=" << idx; guiServer_->appendLog(oss.str()); }
            return 0;
        }
    }
    auto end_sys = std::chrono::system_clock::now();
    auto fmt = [](const std::chrono::system_clock::time_point& tp){
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
        std::time_t tt = system_clock::to_time_t(tp);
        std::tm tmv{};
        #if defined(_WIN32)
        gmtime_s(&tmv, &tt);
        #else
        std::tm* tmp = std::gmtime(&tt);
        if (tmp) tmv = *tmp;
        #endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        std::ostringstream s;
        s << buf << '.' << std::setw(3) << std::setfill('0') << (int)(ms.count());
        return s.str();
    };
    
    std::string ds = fmt(start_sys);
    std::string de = fmt(end_sys);
    auto t1 = high_resolution_clock::now();
    double elapsed = duration<double>(t1 - t0).count();
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RACC_L), Mp);
    if (options.resume) { writeEcmResumeLine("resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) +  "_B2_" + std::to_string(options.B2) + ".save", options.B1, options.exponent, X); convertEcmResumeToPrime95("resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) +  "_B2_" + std::to_string(options.B2) + ".save", "resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) +  "_B2_" + std::to_string(options.B2) + ".p95", ds, de); }
    mpz_class g = gcd_with_dots(X, Mp);
    bool found = g != 1 && g != Mp;
    std::cout << "\nElapsed time (stage 2) = " << std::fixed << std::setprecision(2) << elapsed << " s." << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "Elapsed time (stage 2) = " << std::fixed << std::setprecision(2) << elapsed << " s."; guiServer_->appendLog(oss.str()); }
    std::string filename = "stage2_result_B2_" + B2.get_str() + "_p_" + std::to_string(options.exponent) + ".txt";
    if (found) {
        char* s = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B2=" + B2.get_str() + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n'; guiServer_->appendLog(oss.str()); }
        options.knownFactors.push_back(std::string(s));
        std::free(s);
    } else {
        writeStageResult(filename, "No factor P-1 up to B2=" + B2.get_str());
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n'; guiServer_->appendLog(oss.str()); }
    }
    
    std::remove(ckpt_file_s2.c_str());
    std::remove((ckpt_file_s2 + ".old").c_str());
    std::remove((ckpt_file_s2 + ".new").c_str());
    
    
    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
   // wm.saveIndividualJson(options.exponent, options.mode, json);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage2", json);
    wm.appendToResultsTxt(json);
   /* if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
            if (guiServer_) { std::ostringstream oss; oss << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n"; guiServer_->appendLog(oss.str()); }
            std::ifstream f(options.worktodo_path);
            std::string l; bool more = false; while (std::getline(f, l)) { if (!l.empty() && l[0] != '#') { more = true; break; } }
            f.close();
            if (more) { std::cout << "Restarting for next entry in worktodo.txt\n"; if (guiServer_) { std::ostringstream oss; oss << "Restarting for next entry in worktodo.txt\n"; guiServer_->appendLog(oss.str()); } restart_self(argc_, argv_); }
            else { std::cout << "No more entries in worktodo.txt, exiting.\n"; if (guiServer_) { std::ostringstream oss; oss << "No more entries in worktodo.txt, exiting.\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(0);} }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n"; if (guiServer_) { std::ostringstream oss; oss << "Failed to update " << options.worktodo_path << "\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(-1);}
        }
    }*/
    
    delete eng;
    return found ? 0 : 1;
}

/* ===== n^K Stage-2 (Topics in advanced scientific computation. by: Crandall, Richard E) ===== */
/* Fast b^{n^K} (Stirling init + z-chain); product of differences; GCD. */
int App::runPM1Stage2MarinNKVersion() {
    using namespace std::chrono;
    const uint32_t pexp  = (uint32_t)options.exponent;
    const uint32_t K     = (uint32_t)options.K;
    const uint64_t nmax  = (uint64_t)options.nmax;
    if (K == 0 || nmax == 0) { std::cout << "Nothing to do (K=0 or nmax=0)\n"; return 0; }
    if (guiServer_) { std::ostringstream oss; oss << "P-1 Stage 2 (n^K) — K=" << K << ", nmax=" << nmax; guiServer_->setStatus(oss.str()); }

    const size_t RSTATE=0, RACC=1, RTMP=2, RPOW=3, RDIFF=4, RONE=5;
    size_t regCount = 6 + (size_t)K + 1 + (size_t)nmax;

    engine* eng_s1 = engine::create_gpu(pexp, 11, (size_t)options.device_id, options.debug);
    std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
    auto read_ckpt_s1 = [&](engine* e, const std::string& file)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 3) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        uint32_t ri = 0; double et = 0.0;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        uint64_t tmp64;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        if (!f.read(reinterpret_cast<char*>(&tmp64), sizeof(tmp64))) return -2;
        uint8_t inlot=0; if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
        uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
        if (eacc_len) { std::string skip; skip.resize(eacc_len); if (!f.read(skip.data(), eacc_len)) return -2; }
        uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
        if (wbits_len) { std::string skip; skip.resize(wbits_len); if (!f.read(skip.data(), wbits_len)) return -2; }
        uint64_t chunkIdx=0, startP=0; uint8_t first=0; uint64_t processedBits=0, bitsInChunk=0;
        if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
        if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
        if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
        if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    int rr = read_ckpt_s1(eng_s1, ck.str());
    if (rr < 0) rr = read_ckpt_s1(eng_s1, ck.str() + ".old");
    if (rr != 0) { delete eng_s1; std::cout << "Stage 2 (n^K): cannot load stage-1 checkpoint\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 2 (n^K): cannot load stage-1 checkpoint"; guiServer_->appendLog(oss.str()); } return -2; }
    mpz_t H; mpz_init(H); eng_s1->get_mpz(H, (engine::Reg)0); delete eng_s1;

    engine* eng = engine::create_gpu(pexp, regCount, (size_t)options.device_id, options.debug);
    eng->set_mpz((engine::Reg)RSTATE, H);
    mpz_clear(H);

    auto pow_big_mpz = [&](size_t dst, size_t baseReg, const mpz_class& e)->bool{
        eng->set((engine::Reg)dst, 1);
        if (mpz_sgn(e.get_mpz_t()) == 0) return true;
        eng->set_multiplicand((engine::Reg)RTMP, (engine::Reg)baseReg);
        size_t nb = mpz_sizeinbase(e.get_mpz_t(), 2);
        const size_t chunk = 4096;
        for (size_t k = 0; k < nb; ++k) {
            eng->square_mul((engine::Reg)dst);
            size_t bit = nb - 1 - k;
            if (mpz_tstbit(e.get_mpz_t(), bit)) eng->mul((engine::Reg)dst, (engine::Reg)RTMP);
            if ((k % chunk) == 0 && interrupted) return false;
        }
        return true;
    };

    std::vector<std::vector<mpz_class>> S(K+1, std::vector<mpz_class>(K+1));
    S[0][0] = 1;
    for (uint32_t n = 1; n <= K; ++n) { S[n][0] = mpz_class(0); for (uint32_t j = 1; j <= n; ++j) S[n][j] = mpz_class(j) * S[n-1][j] + S[n-1][j-1]; }
    std::vector<mpz_class> fact(K+1); fact[0] = 1; for (uint32_t j = 1; j <= K; ++j) fact[j] = fact[j-1] * j;

    size_t Z0 = 6, VAL0 = Z0 + (size_t)K + 1;
    eng->set((engine::Reg)(Z0 + 0), 1);
    for (uint32_t j = 1; j <= K; ++j) { mpz_class e = fact[j] * S[K][j]; if (!pow_big_mpz(Z0 + j, RSTATE, e)) { delete eng; std::cout << "Interrupted during initialization\n"; return 0; } }

    eng->set((engine::Reg)RACC, 1);
    eng->set((engine::Reg)RONE, 1);

    auto t0 = high_resolution_clock::now();
    auto last = high_resolution_clock::now();

    for (uint64_t m = 1; m <= nmax; ++m) {
        for (uint32_t q = 0; q < K; ++q) { 
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)(Z0 + q + 1)); 
            eng->mul((engine::Reg)(Z0 + q), (engine::Reg)RPOW); 
            }
        eng->copy((engine::Reg)(VAL0 + (m-1)), (engine::Reg)(Z0 + 0));
        auto now = high_resolution_clock::now();
        if (duration_cast<milliseconds>(now - last).count() >= 300) {
            double done = double(m), total = double(nmax);
            double elapsed = duration<double>(now - t0).count();
            double ips = done / std::max(1e-9, elapsed);
            double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
            std::cout << "build " << m << "/" << nmax << " | " << std::fixed << std::setprecision(2) << (done*100.0/total) << "% | ETA " << (int(eta)/3600) << "h " << (int(eta)%3600)/60 << "m\r" << std::flush;
            last = now;
        }
        if (interrupted) { delete eng; std::cout << "\nInterrupted by user\n"; return 0; }
    }
    std::cout << "\nAccumulating pairwise differences on GPU\n";

    uint64_t totalPairs = (nmax > 1) ? (nmax * (nmax - 1)) / 2 : 0, pairsDone = 0;
    for (uint64_t i = 0; i < nmax; ++i) {
        for (uint64_t j = i + 1; j < nmax; ++j) {
            eng->copy((engine::Reg)RDIFF, (engine::Reg)(VAL0 + j));
            eng->sub_reg((engine::Reg)RDIFF, (engine::Reg)(VAL0 + i));
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)RDIFF);
            eng->mul((engine::Reg)RACC, (engine::Reg)RPOW);
            ++pairsDone;
            /* Verification correct pour sub reg
            mpz_class Mpx = (mpz_class(1) << pexp) - 1; 
            mpz_t za, zb, zd_gpu, zd_exp;
            mpz_inits(za, zb, zd_gpu, zd_exp, nullptr);
            eng->get_mpz(za, (engine::Reg)(VAL0 + j));
            eng->get_mpz(zb, (engine::Reg)(VAL0 + i));
            eng->get_mpz(zd_gpu, (engine::Reg)RDIFF);
            mpz_sub(zd_exp, za, zb);
            mpz_mod(zd_exp, zd_exp, Mpx.get_mpz_t());
            if (mpz_cmp(zd_gpu, zd_exp) != 0) {
                std::cerr << "sub_reg mismatch at i=" << i << " j=" << j << std::endl;
                std::cerr << "gpu=" << mpz_class(zd_gpu).get_str() << std::endl;
                std::cerr << "exp=" << mpz_class(zd_exp).get_str() << std::endl;
                std::abort();
            }
            
            mpz_clears(za, zb, zd_gpu, zd_exp, nullptr);*/
            auto now = high_resolution_clock::now();
            if (duration_cast<milliseconds>(now - last).count() >= 400) {
                double done = double(pairsDone), total = double(totalPairs);
                double elapsed = duration<double>(now - t0).count();
                double ips = done / std::max(1e-9, elapsed);
                double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                std::cout << "pairs " << pairsDone << "/" << totalPairs << " | " << std::fixed << std::setprecision(2) << (total ? (done*100.0/total) : 100.0) << "% | ETA " << (int(eta)/3600) << "h " << (int(eta)%3600)/60 << "m\r" << std::flush;
                last = now;
            }
            if (interrupted) { delete eng; std::cout << "\nInterrupted by user\n"; return 0; }
        }
    }
    std::cout << "\nComputing GCD...\n";

    mpz_t Xz; mpz_init(Xz); eng->get_mpz(Xz, (engine::Reg)RACC); mpz_class Mp = (mpz_class(1) << pexp) - 1; mpz_class X; mpz_set(X.get_mpz_t(), Xz); mpz_clear(Xz);
    mpz_class g; mpz_gcd(g.get_mpz_t(), X.get_mpz_t(), Mp.get_mpz_t());

    bool found = (g > 1 && g < Mp);
    if (found) { std::cout << "Stage 2 n^K Factor found : " << g.get_str() << std::endl; if (guiServer_) { std::ostringstream oss; oss << "Stage 2 n^K Factor found : " << g.get_str(); guiServer_->appendLog(oss.str()); } }
    else { std::cout << "No factor" << std::endl; if (guiServer_) { std::ostringstream oss; oss << "No factor"; guiServer_->appendLog(oss.str()); } }

    double elapsed = duration<double>(high_resolution_clock::now() - t0).count();
    std::cout << "Elapsed (n^K) = " << std::fixed << std::setprecision(2) << elapsed << " s\n";
    delete eng;
    return found ? 0 : 1;
}

static inline unsigned u64_bits(uint64_t x){
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return x ? 64u - static_cast<unsigned>(std::countl_zero(x)) : 0u;
#elif defined(_MSC_VER)
    if (!x) return 0u;
    unsigned long idx;
#if defined(_M_X64) || defined(_M_ARM64)
    _BitScanReverse64(&idx, x);
    return idx + 1u;
#else
    if (x >> 32) { _BitScanReverse(&idx, static_cast<unsigned long>(x >> 32)); return idx + 33u; }
    _BitScanReverse(&idx, static_cast<unsigned long>(x));
    return idx + 1u;
#endif
#else
    return x ? 64u - static_cast<unsigned>(__builtin_clzll(x)) : 0u;
#endif
}

int App::runECMMarin()
{
    using namespace std;
    using namespace std::chrono;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const uint64_t B1 = options.B1 ? options.B1 : 1000000ULL;
    const uint64_t B2 = options.B2 ? options.B2 : 0ULL;
    //const bool verbose = options.debug;
    const bool verbose = options.debug;
    
    uint64_t curves = options.nmax ? options.nmax : (options.K ? options.K : 250);

    auto splitmix64 = [](uint64_t& x)->uint64_t{ x += 0x9E3779B97f4A7C15ULL; uint64_t z=x; z^=z>>30; z*=0xBF58476D1CE4E5B9ULL; z^=z>>27; z*=0x94D049BB133111EBULL; z^=z>>31; return z; };
    auto rnd_mpz = [&](const mpz_class& N, uint64_t& seed, uint64_t bits)->mpz_class{
        mpz_class z=0; for (size_t i=0;i<bits;i+=64){ z <<= 64; z += (unsigned long)splitmix64(seed); } z %= N; if (z<=2) z+=3; return z;
    };
    auto hex64 = [&](const mpz_class& z)->string{
        std::ostringstream ss; ss<<std::uppercase<<std::hex<<std::setw(16)<<std::setfill('0')<<static_cast<uint64_t>(mpz_get_ui(z.get_mpz_t())); return ss.str();
    };
    auto fmt_hms = [&](double s)->string{
        uint64_t u = (uint64_t)(s + 0.5); uint64_t h=u/3600, m=(u%3600)/60, se=u%60; std::ostringstream ss; ss<<h<<"h "<<m<<"m "<<se<<"s"; return ss.str();
    };

    mpz_class N = (mpz_class(1) << p) - 1;

    mpz_class K = 1;
    uint32_t primesB1 = 0;
    {
        uint64_t b = B1;
        vector<uint8_t> sieve(b/2+1, 1);
        for (uint64_t i=3;i*i<=b;i+=2) if (sieve[i>>1]) for (uint64_t j=i*i;j<=b;j+=i<<1) sieve[j>>1]=0;
        auto apply_prime = [&](uint64_t q){ uint64_t pw=q; while (pw <= b / q) pw *= q; mpz_class t; mpz_set_ui(t.get_mpz_t(), (unsigned long)pw); K *= t; ++primesB1; };
        apply_prime(2);
        for (uint64_t q=3;q<=b;q+=2) if (sieve[q>>1]) apply_prime(q);
    }
    size_t nb = mpz_sizeinbase(K.get_mpz_t(), 2);

    std::vector<uint32_t> primesS2;
    if (B2 > B1) {
        uint64_t b = B2;
        vector<uint8_t> sieve(b/2+1, 1);
        for (uint64_t i=3;i*i<=b;i+=2) if (sieve[i>>1]) for (uint64_t j=i*i;j<=b;j+=i<<1) sieve[j>>1]=0;
        if (B1 < 2 && 2 <= B2) primesS2.push_back(2);
        for (uint64_t q=3;q<=B2;q+=2) if (sieve[q>>1] && q > B1) primesS2.push_back((uint32_t)q);
    }

    {
        std::ostringstream hdr;
        hdr<<"[ECM] N=M_"<<p<<"  B1="<<B1<<"  B2="<<B2<<"  curves="<<curves<<"\n";
        hdr<<"[ECM] Stage1: product of prime powers up to B1, primes used="<<primesB1<<", K_bits="<<nb<<"\n";
        if (!primesS2.empty()) {
            hdr<<"[ECM] Stage2: primes in ("<<B1<<","<<B2<<"], count="<<primesS2.size();
            hdr<<", first="<<(primesS2.size()?primesS2.front():0)<<", last="<<(primesS2.size()?primesS2.back():0)<<"\n";
        } else {
            hdr<<"[ECM] Stage2: disabled\n";
        }
        std::cout<<hdr.str();
        if (guiServer_) guiServer_->appendLog(hdr.str());
    }

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    uint64_t seed0 = now_ns ^ ((uint64_t)p<<32) ^ B1;

    size_t transform_size_once = 0;

    for (uint64_t c = 0; c < curves; ++c)
    {
        engine* eng = engine::create_gpu(p, static_cast<size_t>(18), static_cast<size_t>(options.device_id), verbose);
        if (transform_size_once == 0) {
            transform_size_once = eng->get_size();
            std::ostringstream os; os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id;
            std::cout<<os.str()<<std::endl; if (guiServer_) guiServer_->appendLog(os.str());
        }

        uint64_t seed = seed0 + c*0xD1342543DE82EF95ULL;
        mpz_class sigma = rnd_mpz(N, seed, 128);
        mpz_class u = (sigma*sigma - 5) % N; if (u < 0) u += N;
        mpz_class v = (4*sigma) % N;

        mpz_class g; mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t());
        if (g > 1 && g < N) {
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl;
            if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found at curve "<<(c+1)<<": "<<g.get_str(); guiServer_->appendLog(oss.str()); }
            delete eng; return 0;
        }
        mpz_class invv; if (!mpz_invert(invv.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t())) {
            mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t());
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl;
            if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found (invert v failed): "<<g.get_str(); guiServer_->appendLog(oss.str()); }
            delete eng; return 0;
        }
        mpz_class t = (u * invv) % N; mpz_class x0 = (t*t - 2) % N; if (x0 < 0) x0 += N;

        mpz_class t1 = (v - u) % N; if (t1 < 0) t1 += N; mpz_class t1_3 = (t1*t1) % N; t1_3 = (t1_3 * t1) % N;
        mpz_class t2 = (3*u + v) % N; if (t2 < 0) t2 += N; mpz_class num = (t1_3 * t2) % N;
        mpz_class den = 0; { mpz_class u3 = (u*u) % N; u3 = (u3*u) % N; den = (4 * u3) % N; den = (den * v) % N; }
        mpz_gcd(g.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()); if (g > 1 && g < N) {
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl;
            if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found in parameterization: "<<g.get_str(); guiServer_->appendLog(oss.str()); }
            delete eng; return 0;
        }
        mpz_class invden; if (!mpz_invert(invden.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t())) {
            mpz_gcd(g.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t());
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl;
            if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found (invert den failed): "<<g.get_str(); guiServer_->appendLog(oss.str()); }
            delete eng; return 0;
        }
        mpz_class A = (num * invden) % N; A = (A - 2) % N; if (A < 0) A += N;
        mpz_class inv4; mpz_invert(inv4.get_mpz_t(), mpz_class(4).get_mpz_t(), N.get_mpz_t());
        mpz_class A24 = ((A + 2) % N) * inv4 % N;

        const size_t RX0=0, RZ0=1, RX1=2, RZ1=3, RXD=4, RZD=5, RA24=6, RT0=7, RT1=8, RT2=9, RT3=10, RU=11, RV=12, RM=13;

        eng->set((engine::Reg)RZ0, 0u);
        eng->set((engine::Reg)RX0, 1u);
        eng->set((engine::Reg)RZ1, 1u);
        { mpz_t z; mpz_init(z); mpz_set(z, x0.get_mpz_t()); eng->set_mpz((engine::Reg)RX1, z); mpz_set(z, x0.get_mpz_t()); eng->set_mpz((engine::Reg)RXD, z); mpz_set_ui(z, 1); eng->set_mpz((engine::Reg)RZD, z); mpz_set(z, A24.get_mpz_t()); eng->set_mpz((engine::Reg)RA24, z); mpz_clear(z); }

        uint64_t cnt_xdbl=0, cnt_xadd=0, cnt_mul=0, cnt_sqr=0;

        auto mul_inplace = [&](size_t dst, size_t src){ eng->set_multiplicand((engine::Reg)RM, (engine::Reg)src); eng->mul((engine::Reg)dst, (engine::Reg)RM); ++cnt_mul; };

        auto xDBL = [&](size_t X1,size_t Z1,size_t X2,size_t Z2){
            eng->copy((engine::Reg)RT0, (engine::Reg)X1);
            eng->add((engine::Reg)RT0, (engine::Reg)Z1);
            eng->copy((engine::Reg)RT1, (engine::Reg)RT0);
            eng->square_mul((engine::Reg)RT1); ++cnt_sqr;
            eng->copy((engine::Reg)RT2, (engine::Reg)X1);
            eng->sub_reg((engine::Reg)RT2, (engine::Reg)Z1);
            eng->copy((engine::Reg)RT3, (engine::Reg)RT2);
            eng->square_mul((engine::Reg)RT3); ++cnt_sqr;
            eng->copy((engine::Reg)X2, (engine::Reg)RT1);
            mul_inplace(X2, RT3);
            eng->copy((engine::Reg)RT0, (engine::Reg)RT1);
            eng->sub_reg((engine::Reg)RT0, (engine::Reg)RT3);
            eng->copy((engine::Reg)RT2, (engine::Reg)RT0);
            mul_inplace(RT2, RA24);
            eng->add((engine::Reg)RT2, (engine::Reg)RT3);
            eng->copy((engine::Reg)Z2, (engine::Reg)RT2);
            mul_inplace(Z2, RT0);
            ++cnt_xdbl;
        };

        auto xADD = [&](size_t X1,size_t Z1,size_t X2,size_t Z2,size_t XD,size_t ZD,size_t X3,size_t Z3){
            eng->copy((engine::Reg)RT0, (engine::Reg)X1);
            eng->add((engine::Reg)RT0, (engine::Reg)Z1);
            eng->copy((engine::Reg)RT1, (engine::Reg)X1);
            eng->sub_reg((engine::Reg)RT1, (engine::Reg)Z1);
            eng->copy((engine::Reg)RT2, (engine::Reg)X2);
            eng->add((engine::Reg)RT2, (engine::Reg)Z2);
            eng->copy((engine::Reg)RT3, (engine::Reg)X2);
            eng->sub_reg((engine::Reg)RT3, (engine::Reg)Z2);
            eng->copy((engine::Reg)RU, (engine::Reg)RT3);
            mul_inplace(RU, RT0);
            eng->copy((engine::Reg)RV, (engine::Reg)RT2);
            mul_inplace(RV, RT1);
            eng->copy((engine::Reg)X3, (engine::Reg)RU);
            eng->add((engine::Reg)X3, (engine::Reg)RV);
            eng->square_mul((engine::Reg)X3); ++cnt_sqr;
            mul_inplace(X3, ZD);
            eng->copy((engine::Reg)Z3, (engine::Reg)RU);
            eng->sub_reg((engine::Reg)Z3, (engine::Reg)RV);
            eng->square_mul((engine::Reg)Z3); ++cnt_sqr;
            mul_inplace(Z3, XD);
            ++cnt_xadd;
        };

        std::ostringstream head;
        head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | sigma64=0x"<<hex64(sigma)<<" x0_64=0x"<<hex64(x0)<<" A_64=0x"<<hex64(A)<<" A24_64=0x"<<hex64(A24)<<" | K_bits="<<nb;
        std::cout<<head.str()<<std::endl;
        if (guiServer_) guiServer_->appendLog(head.str());

        std::ostringstream ck; ck << "ecm_m_" << p << "_c" << c << ".ckpt";
        const std::string ckpt_file = ck.str();

        auto save_ckpt = [&](uint32_t i, double et){
            const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
            { File f(newf, "wb"); int version = 1; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; uint32_t nbb = (uint32_t)nb; if (!f.write(reinterpret_cast<const char*>(&nbb), sizeof(nbb))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
            std::error_code ec; fs::remove(ckpt_file + ".old", ec); fs::rename(ckpt_file, ckpt_file + ".old", ec); fs::rename(ckpt_file + ".new", ckpt_file, ec); fs::remove(ckpt_file + ".old", ec);
        };
        auto read_ckpt = [&](const std::string& file, uint32_t& ri, uint32_t& rnb, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 1) return -2;
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
            if (!f.read(reinterpret_cast<char*>(&rnb), sizeof(rnb))) return -2;
            uint64_t rB1 = 0; if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            if (rnb != nb || rB1 != B1) return -2;
            return 0;
        };

        uint32_t start_i = 0, nb_ck = 0; double saved_et = 0.0;
        (void)read_ckpt(ckpt_file, start_i, nb_ck, saved_et);

        auto t0 = high_resolution_clock::now();
        auto last_save = t0;
        size_t step_bits = std::max<size_t>(nb/20, 1);
        size_t next_mark = start_i + step_bits;

        for (size_t i = start_i; i < nb; ++i){
            size_t bit = nb - 1 - i;
            mp_bitcnt_t mb = static_cast<mp_bitcnt_t>(bit);
            int b = mpz_tstbit(K.get_mpz_t(), mb) ? 1 : 0;
            if (b==0) { xADD(RX1,RZ1,RX0,RZ0,RXD,RZD,RT0,RT1); eng->copy((engine::Reg)RX1,(engine::Reg)RT0); eng->copy((engine::Reg)RZ1,(engine::Reg)RT1); xDBL(RX0,RZ0,RT0,RT1); eng->copy((engine::Reg)RX0,(engine::Reg)RT0); eng->copy((engine::Reg)RZ0,(engine::Reg)RT1); }
            else      { xADD(RX0,RZ0,RX1,RZ1,RXD,RZD,RT0,RT1); eng->copy((engine::Reg)RX0,(engine::Reg)RT0); eng->copy((engine::Reg)RZ0,(engine::Reg)RT1); xDBL(RX1,RZ1,RT0,RT1); eng->copy((engine::Reg)RX1,(engine::Reg)RT0); eng->copy((engine::Reg)RZ1,(engine::Reg)RT1); }

            auto now = high_resolution_clock::now();
            if (i + 1 >= next_mark || i + 1 == nb) {
                double done = double(i + 1), total = double(nb);
                double elapsed = duration<double>(now - t0).count() + saved_et;
                double ips = done / std::max(1e-9, elapsed);
                double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                std::ostringstream line;
                line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 "<<(i+1)<<"/"<<nb<<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%)"
                    <<" | ETA "<<fmt_hms(eta)
                    <<" | xDBL="<<cnt_xdbl<<" xADD="<<cnt_xadd<<" sqr="<<cnt_sqr<<" mul="<<cnt_mul;
                std::cout<<line.str()<<std::flush;
                next_mark += step_bits;
            }
            if (duration_cast<seconds>(now - last_save).count() >= 10) {
                double elapsed = duration<double>(now - t0).count() + saved_et;
                save_ckpt((uint32_t)(i + 1), elapsed);
                last_save = now;
            }
            if (interrupted) {
                double elapsed = duration<double>(now - t0).count() + saved_et;
                save_ckpt((uint32_t)(i + 1), elapsed);
                std::cout<<"\n[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<nb<<"\n";
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<nb; guiServer_->appendLog(oss.str()); }
                delete eng;
                return 0;
            }
        }
        std::cout<<std::endl;

        std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | GCD after Stage1..."<<std::endl;
        mpz_t Zk; mpz_init(Zk); eng->get_mpz(Zk, (engine::Reg)RZ0);
        mpz_class Zfin; mpz_set(Zfin.get_mpz_t(), Zk); mpz_clear(Zk);
        Zfin %= N; if (Zfin < 0) Zfin += N;
        mpz_class gg; mpz_gcd(gg.get_mpz_t(), Zfin.get_mpz_t(), N.get_mpz_t());
        bool found = (gg > 1 && gg < N);

        double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count();
        {
            std::ostringstream s1;
            s1<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 elapsed="<<fixed<<setprecision(2)<<elapsed_stage1<<" s"
              <<" | xDBL="<<cnt_xdbl<<" xADD="<<cnt_xadd<<" sqr="<<cnt_sqr<<" mul="<<cnt_mul;
            std::cout<<s1.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(s1.str());
        }

        if (found) {
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<gg.get_str()<<std::endl;
            if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found at curve "<<(c+1)<<": "<<gg.get_str(); guiServer_->appendLog(oss.str()); }
            std::error_code ec0; fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0);
            delete eng;
            return 0;
        }

        if (B2 > B1) {
            std::ostringstream s2h; s2h<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 start, primes="<<primesS2.size();
            std::cout<<s2h.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2h.str());

            std::ostringstream ck2; ck2 << "ecm2_m_" << p << "_c" << c << ".ckpt";
            const std::string ckpt2 = ck2.str();

            auto save_ckpt2 = [&](uint32_t idx, double et){
                const std::string oldf = ckpt2 + ".old", newf = ckpt2 + ".new";
                { File f(newf, "wb"); int version = 2; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&idx), sizeof(idx))) return; uint32_t cnt = (uint32_t)primesS2.size(); if (!f.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&B2), sizeof(B2))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
                std::error_code ec; fs::remove(ckpt2 + ".old", ec); fs::rename(ckpt2, ckpt2 + ".old", ec); fs::rename(ckpt2 + ".new", ckpt2, ec); fs::remove(ckpt2 + ".old", ec);
            };
            auto read_ckpt2 = [&](const std::string& file, uint32_t& idx, uint32_t& cnt, double& et)->int{
                File f(file);
                if (!f.exists()) return -1;
                int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
                if (version != 2) return -2;
                uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
                if (rp != p) return -2;
                if (!f.read(reinterpret_cast<char*>(&idx), sizeof(idx))) return -2;
                if (!f.read(reinterpret_cast<char*>(&cnt), sizeof(cnt))) return -2;
                uint64_t b1s=0,b2s=0; if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return -2; if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!f.read(data.data(), cksz)) return -2;
                if (!eng->set_checkpoint(data)) return -2;
                if (!f.check_crc32()) return -2;
                if (cnt != primesS2.size() || b1s != B1 || b2s != B2) return -2;
                return 0;
            };

            uint32_t s2_idx = 0, s2_cnt = 0; double s2_et = 0.0;
            int rr2 = read_ckpt2(ckpt2, s2_idx, s2_cnt, s2_et);
            if (rr2 < 0) rr2 = read_ckpt2(ckpt2 + ".old", s2_idx, s2_cnt, s2_et);

            auto ladder_mul_small = [&](size_t Xin,size_t Zin, uint64_t m, size_t Xout,size_t Zout){
                eng->set((engine::Reg)RX0, 1u);
                eng->set((engine::Reg)RZ0, 0u);
                eng->copy((engine::Reg)RX1, (engine::Reg)Xin);
                eng->copy((engine::Reg)RZ1, (engine::Reg)Zin);
                eng->copy((engine::Reg)RXD, (engine::Reg)Xin);
                eng->copy((engine::Reg)RZD, (engine::Reg)Zin);
                size_t nbq = u64_bits(m);
                for (size_t bi=0; bi<nbq; ++bi){
                    size_t bit = nbq - 1 - bi;
                    int b = ((m >> bit) & 1ULL) ? 1 : 0;
                    if (b==0) { xADD(RX1,RZ1,RX0,RZ0,RXD,RZD,RT0,RT1); eng->copy((engine::Reg)RX1,(engine::Reg)RT0); eng->copy((engine::Reg)RZ1,(engine::Reg)RT1); xDBL(RX0,RZ0,RT0,RT1); eng->copy((engine::Reg)RX0,(engine::Reg)RT0); eng->copy((engine::Reg)RZ0,(engine::Reg)RT1); }
                    else      { xADD(RX0,RZ0,RX1,RZ1,RXD,RZD,RT0,RT1); eng->copy((engine::Reg)RX0,(engine::Reg)RT0); eng->copy((engine::Reg)RZ0,(engine::Reg)RT1); xDBL(RX1,RZ1,RT0,RT1); eng->copy((engine::Reg)RX1,(engine::Reg)RT0); eng->copy((engine::Reg)RZ1,(engine::Reg)RT1); }
                }
                eng->copy((engine::Reg)Xout, (engine::Reg)RX0);
                eng->copy((engine::Reg)Zout, (engine::Reg)RZ0);
            };

            auto t2_0 = high_resolution_clock::now();
            auto last2_save = t2_0;
            size_t step_p = std::max<size_t>(primesS2.size()/20, 1);
            size_t next_p = s2_idx + step_p;

            size_t Xcur = RX0, Zcur = RZ0;

            for (uint32_t i = s2_idx; i < (uint32_t)primesS2.size(); ++i) {
                uint64_t q = primesS2[i];
                ladder_mul_small(Xcur, Zcur, q, RT0, RT1);
                eng->copy((engine::Reg)Xcur, (engine::Reg)RT0);
                eng->copy((engine::Reg)Zcur, (engine::Reg)RT1);

                if (i + 1 >= next_p || i + 1 == primesS2.size()) {
                    double done = double(i + 1), total = double(primesS2.size());
                    double elapsed = duration<double>(high_resolution_clock::now() - t2_0).count() + s2_et;
                    double ips = done / std::max(1e-9, elapsed);
                    double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                    std::ostringstream line;
                    line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 "<<(i+1)<<"/"<<primesS2.size()<<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%)"
                        <<" | ETA "<<fmt_hms(eta);
                    std::cout<<line.str()<<std::flush;
                    next_p += step_p;
                }

                auto now2 = high_resolution_clock::now();
                if (duration_cast<seconds>(now2 - last2_save).count() >= 10) {
                    double elapsed = duration<double>(now2 - t2_0).count() + s2_et;
                    save_ckpt2((uint32_t)(i + 1), elapsed);
                    last2_save = now2;
                }
                if (interrupted) {
                    double elapsed = duration<double>(high_resolution_clock::now() - t2_0).count() + s2_et;
                    save_ckpt2((uint32_t)(i + 1), elapsed);
                    std::cout<<"\n[ECM] Interrupted at Stage2 curve "<<(c+1)<<" index "<<(i+1)<<"/"<<primesS2.size()<<"\n";
                    if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at Stage2 curve "<<(c+1)<<" index "<<(i+1)<<"/"<<primesS2.size(); guiServer_->appendLog(oss.str()); }
                    delete eng;
                    return 0;
                }
            }
            std::cout<<std::endl;

            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | GCD after Stage2..."<<std::endl;
            mpz_t Zk2; mpz_init(Zk2); eng->get_mpz(Zk2, (engine::Reg)Xcur==RX0?(engine::Reg)RZ0:(engine::Reg)Zcur);
            mpz_class Zfin2; mpz_set(Zfin2.get_mpz_t(), Zk2); mpz_clear(Zk2);
            Zfin2 %= N; if (Zfin2 < 0) Zfin2 += N;
            mpz_class gg2; mpz_gcd(gg2.get_mpz_t(), Zfin2.get_mpz_t(), N.get_mpz_t());
            bool found2 = (gg2 > 1 && gg2 < N);

            std::error_code ec2; fs::remove(ckpt2, ec2); fs::remove(ckpt2 + ".old", ec2); fs::remove(ckpt2 + ".new", ec2);

            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + s2_et;
            {
                std::ostringstream s2s; s2s<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 elapsed="<<fixed<<setprecision(2)<<elapsed2<<" s";
                std::cout<<s2s.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2s.str());
            }

            if (found2) {
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<gg2.get_str()<<std::endl;
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found at Stage2 curve "<<(c+1)<<": "<<gg2.get_str(); guiServer_->appendLog(oss.str()); }
                std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);
                delete eng;
                return 0;
            }
        }

        std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);
        double elapsed = duration<double>(high_resolution_clock::now() - t0).count();
        {
            std::ostringstream fin; fin<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" done in "<<fixed<<setprecision(2)<<elapsed<<" s";
            std::cout<<fin.str()<<std::endl; if (guiServer_) guiServer_->appendLog(fin.str());
        }

        delete eng;
    }

    std::cout<<"[ECM] No factor found"<<std::endl;
    return 1;
}





int App::runPM1Marin() {
    if (guiServer_) { std::ostringstream oss; oss << "P-1 factoring stage 1"; guiServer_->setStatus(oss.str()); }
    bool debug = false;
    uint64_t B1 = options.B1;
    std::cout << "[Backend Marin] Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "[Backend Marin] Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl; guiServer_->appendLog(oss.str()); }
    const double L_est_bits = 1.4426950408889634 * static_cast<double>(B1);
    const uint64_t MAX_E_BITS = options.max_e_bits;
    std::cout << "MAX_E_BITS = " << MAX_E_BITS << " bits (≈ " << (MAX_E_BITS >> 23) << " MiB)" << std::endl;
    uint64_t estChunks = std::max<uint64_t>(1, (uint64_t)std::ceil(L_est_bits / (double)MAX_E_BITS));
    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = options.debug;
    engine* eng = engine::create_gpu(p, static_cast<size_t>(11), static_cast<size_t>(options.device_id), verbose);
    const size_t RSTATE=0, RACC_L=1, RACC_R=2, RCHK=3, RPOW=4, RTMP=5, RSTART=6, RSAVE_S=7, RSAVE_L=8, RSAVE_R=9, RBASE=10;
    std::ostringstream ck; ck << "pm1_m_" << p << ".ckpt";
    const std::string ckpt_file = ck.str();
    auto save_ckpt = [&](uint32_t i, double et, uint64_t chk, uint64_t blks, uint64_t bib, uint64_t cbl, uint8_t inlot, const mpz_class& ceacc, const mpz_class& cwbits, uint64_t chunkIdx, uint64_t startP, uint8_t first, uint64_t processedBits, uint64_t bitsInChunk){
        const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
        { File f(newf, "wb"); int version = 3; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; if (!f.write(reinterpret_cast<const char*>(&chk), sizeof(chk))) return; if (!f.write(reinterpret_cast<const char*>(&blks), sizeof(blks))) return; if (!f.write(reinterpret_cast<const char*>(&bib), sizeof(bib))) return; if (!f.write(reinterpret_cast<const char*>(&cbl), sizeof(cbl))) return; if (!f.write(reinterpret_cast<const char*>(&inlot), sizeof(inlot))) return; char* eacc_hex_c = mpz_get_str(nullptr, 16, ceacc.get_mpz_t()); uint32_t eacc_len = eacc_hex_c ? (uint32_t)std::strlen(eacc_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&eacc_len), sizeof(eacc_len))) { if (eacc_hex_c) std::free(eacc_hex_c); return; } if (eacc_len && !f.write(eacc_hex_c, eacc_len)) { std::free(eacc_hex_c); return; } if (eacc_hex_c) std::free(eacc_hex_c); char* wbits_hex_c = mpz_get_str(nullptr, 16, cwbits.get_mpz_t()); uint32_t wbits_len = wbits_hex_c ? (uint32_t)std::strlen(wbits_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&wbits_len), sizeof(wbits_len))) { if (wbits_hex_c) std::free(wbits_hex_c); return; } if (wbits_len && !f.write(wbits_hex_c, wbits_len)) { std::free(wbits_hex_c); return; } if (wbits_hex_c) std::free(wbits_hex_c); if (!f.write(reinterpret_cast<const char*>(&chunkIdx), sizeof(chunkIdx))) return; if (!f.write(reinterpret_cast<const char*>(&startP), sizeof(startP))) return; if (!f.write(reinterpret_cast<const char*>(&first), sizeof(first))) return; if (!f.write(reinterpret_cast<const char*>(&processedBits), sizeof(processedBits))) return; if (!f.write(reinterpret_cast<const char*>(&bitsInChunk), sizeof(bitsInChunk))) return; f.write_crc32(); }
        std::error_code ec; fs::remove(oldf, ec); fs::rename(ckpt_file, oldf, ec); fs::rename(ckpt_file + ".new", ckpt_file, ec); fs::remove(oldf, ec);
    };
    auto read_ckpt = [&](const std::string& file, uint32_t& ri, double& et, uint64_t& chk, uint64_t& blks, uint64_t& bib, uint64_t& cbl, uint8_t& inlot, mpz_class& ceacc, mpz_class& cwbits, uint64_t& chunkIdx, uint64_t& startP, uint8_t& first, uint64_t& processedBits, uint64_t& bitsInChunk)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 3) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != p) return -2;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = eng->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!eng->set_checkpoint(data)) return -2;
        if (!f.read(reinterpret_cast<char*>(&chk), sizeof(chk))) return -2;
        if (!f.read(reinterpret_cast<char*>(&blks), sizeof(blks))) return -2;
        if (!f.read(reinterpret_cast<char*>(&bib), sizeof(bib))) return -2;
        if (!f.read(reinterpret_cast<char*>(&cbl), sizeof(cbl))) return -2;
        if (!f.read(reinterpret_cast<char*>(&inlot), sizeof(inlot))) return -2;
        uint32_t eacc_len = 0; if (!f.read(reinterpret_cast<char*>(&eacc_len), sizeof(eacc_len))) return -2;
        std::string eacc_hex; eacc_hex.resize(eacc_len);
        if (eacc_len && !f.read(eacc_hex.data(), eacc_len)) return -2;
        if (eacc_len) mpz_set_str(ceacc.get_mpz_t(), eacc_hex.c_str(), 16); else ceacc = 0;
        uint32_t wbits_len = 0; if (!f.read(reinterpret_cast<char*>(&wbits_len), sizeof(wbits_len))) return -2;
        std::string wbits_hex; wbits_hex.resize(wbits_len);
        if (wbits_len && !f.read(wbits_hex.data(), wbits_len)) return -2;
        if (wbits_len) mpz_set_str(cwbits.get_mpz_t(), wbits_hex.c_str(), 16); else cwbits = 0;
        if (!f.read(reinterpret_cast<char*>(&chunkIdx), sizeof(chunkIdx))) return -2;
        if (!f.read(reinterpret_cast<char*>(&startP), sizeof(startP))) return -2;
        if (!f.read(reinterpret_cast<char*>(&first), sizeof(first))) return -2;
        if (!f.read(reinterpret_cast<char*>(&processedBits), sizeof(processedBits))) return -2;
        if (!f.read(reinterpret_cast<char*>(&bitsInChunk), sizeof(bitsInChunk))) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    timer.start();
    timer2.start();
    auto start_clock = std::chrono::high_resolution_clock::now();
    auto lastDisplay = start_clock;
    auto lastBackup = start_clock;
    interrupted.store(false, std::memory_order_relaxed);
    eng->set(RSTATE, 1);
    eng->set(RACC_L, 1);
    eng->set(RACC_R, 1);
    eng->copy(RSTART, RSTATE);
    eng->copy(RSAVE_S, RSTATE);
    eng->copy(RSAVE_L, RACC_L);
    eng->copy(RSAVE_R, RACC_R);
    uint64_t chunkIndex = 0;
    uint64_t startPrime = 3;
    bool firstChunk = true;
    uint64_t processed_total_bits = 0;
    uint32_t resumeI_ck = 0;
    double restored_time = 0.0;
    uint64_t gl_checkpass_ck = 0, gl_blocks_since_check_ck = 0, gl_bits_in_block_ck = 0, gl_current_block_len_ck = 0, bits_in_chunk_ck = 0;
    uint8_t in_lot_ck = 0, firstChunk_ck = 1;
    mpz_class eacc_ck = 0, wbits_ck = 0;
    bool restored = false;
    int rr = read_ckpt(ckpt_file, resumeI_ck, restored_time, gl_checkpass_ck, gl_blocks_since_check_ck, gl_bits_in_block_ck, gl_current_block_len_ck, in_lot_ck, eacc_ck, wbits_ck, chunkIndex, startPrime, firstChunk_ck, processed_total_bits, bits_in_chunk_ck);
    if (rr < 0) rr = read_ckpt(ckpt_file + ".old", resumeI_ck, restored_time, gl_checkpass_ck, gl_blocks_since_check_ck, gl_bits_in_block_ck, gl_current_block_len_ck, in_lot_ck, eacc_ck, wbits_ck, chunkIndex, startPrime, firstChunk_ck, processed_total_bits, bits_in_chunk_ck);
    if (rr == 0) { restored = true; firstChunk = (firstChunk_ck != 0); }
    auto start_sys = std::chrono::system_clock::now();
    while (true) {
        bool errordone = false;
        bool useFast3Candidate = firstChunk;
        uint64_t nextStart = 0;
        mpz_class Echunk;
        if (useFast3Candidate) {
            uint64_t twoe = 2ULL * (uint64_t)options.exponent;
            uint64_t extra = 0; { uint64_t t = twoe; while (t) { extra++; t >>= 1; } if (extra == 0) extra = 1; }
            uint64_t estBits = (uint64_t)std::ceil(L_est_bits) + extra + 8;
            if (estBits <= MAX_E_BITS) { Echunk = buildE(B1); nextStart = 0; }
            else { Echunk = buildE2(B1, startPrime, MAX_E_BITS, nextStart, firstChunk); }
        } else {
            Echunk = buildE2(B1, startPrime, MAX_E_BITS, nextStart, firstChunk);
        }
        if (firstChunk) Echunk *= mpz_class(2) * mpz_class(static_cast<unsigned long>(options.exponent));
        bool useFast3 = useFast3Candidate && (nextStart == 0);
        mp_bitcnt_t bits = mpz_sizeinbase(Echunk.get_mpz_t(), 2);
        if (bits == 0) break;
        if (restored && bits_in_chunk_ck) bits = (mp_bitcnt_t)bits_in_chunk_ck;
        chunkIndex = std::max<uint64_t>(chunkIndex, 1);
        std::cout << "\nChunk " << chunkIndex << "/" << estChunks << "  bits=" << bits << (useFast3 ? " [fast3]" : "") << std::endl;
        if (guiServer_) { std::ostringstream oss; oss << "Chunk " << chunkIndex << "/" << estChunks << "  bits=" << bits << (useFast3 ? " [fast3]" : ""); guiServer_->appendLog(oss.str()); }
        uint64_t B = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)bits));
        double desiredIntervalSeconds = 600.0;
        uint64_t checkpass = 0;
        uint64_t checkpasslevel_auto = (uint64_t)((1000 * desiredIntervalSeconds) / (double)B);
        if (checkpasslevel_auto == 0) checkpasslevel_auto = ((uint64_t)bits/B)/((uint64_t)(std::sqrt((double)B)));
        uint64_t checkpasslevel = (options.checklevel > 0)
            ? options.checklevel
            : checkpasslevel_auto;
        if(checkpasslevel==0)
            checkpasslevel=1;
        //uint64_t checkpass = (options.checklevel > 0) ? options.checklevel : 1;
        auto chunkStart = std::chrono::high_resolution_clock::now();
        bool tunedCheckpass = false;
        uint64_t resumeI = restored ? (uint64_t)resumeI_ck : (uint64_t)bits;
        uint64_t lastIter = resumeI;
        uint64_t blocks_since_check = restored ? gl_blocks_since_check_ck : 0;
        uint64_t bits_in_block = restored ? gl_bits_in_block_ck : 0;
        uint64_t current_block_len = restored && gl_current_block_len_ck ? gl_current_block_len_ck : (((uint64_t)((resumeI - 1) % B)) + 1);
        mpz_class eacc = restored ? eacc_ck : 0;
        mpz_class wbits = restored ? wbits_ck : 0;
        uint64_t gl_checkpass = restored ? gl_checkpass_ck : 0;
        bool in_lot = restored ? (in_lot_ck != 0) : false;
        spinner.displayProgress2(
            processed_total_bits + (restored ? (bits - resumeI) : 0),
            processed_total_bits + bits,
            timer.elapsed() + restored_time,
            timer2.elapsed(),
            options.exponent,
            processed_total_bits + (restored ? (bits - resumeI) : 0),
            processed_total_bits,
            "",
            guiServer_ ? guiServer_.get() : nullptr,
            chunkIndex,
            estChunks,
            (restored ? (bits - resumeI) : 0),
            bits,
            true
        );
        if (!restored) {
            if (firstChunk) { eng->set(RBASE, 3); eng->set(RSTATE, 1); }
            else { eng->copy(RBASE, RSTATE); eng->set(RSTATE, 1); }
        }
        for (mp_bitcnt_t i = (mp_bitcnt_t)resumeI; i > 0; --i) {
            lastIter = i;
            if (interrupted) {
                std::cout << "\nInterrupted by user, state saved at iteration " << i << std::endl;
                if (guiServer_) { std::ostringstream oss; oss << "\nInterrupted signal received\n "; guiServer_->appendLog(oss.str()); }
                save_ckpt((uint32_t)lastIter, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time, gl_checkpass, blocks_since_check, bits_in_block, current_block_len, in_lot ? 1 : 0, eacc, wbits, chunkIndex, startPrime, firstChunk ? 1 : 0, processed_total_bits + (bits - i), (uint64_t)bits);
                delete eng;
                return 0;
            }
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastBackup).count() >= options.backup_interval) {
                std::cout << "\nBackup point done at i=" << i << " start...." << std::endl;
                save_ckpt((uint32_t)lastIter, std::chrono::duration<double>(now - start_clock).count() + restored_time, gl_checkpass, blocks_since_check, bits_in_block, current_block_len, in_lot ? 1 : 0, eacc, wbits, chunkIndex, startPrime, firstChunk ? 1 : 0, processed_total_bits + (bits - i), (uint64_t)bits);
                std::cout << "\nBackup point done at i=" << i << " done...." << std::endl;
                lastBackup = now;
            }
            if (bits_in_block == 0) {
                current_block_len = ((uint64_t)((i - 1) % B)) + 1;
                if (current_block_len == B) {
                    if (gl_checkpass == 0 && blocks_since_check == 0 && wbits == 0 && eacc == 0) {
                        eng->set(RACC_L, 1);
                        eng->set(RACC_R, 1);
                        eng->copy(RSAVE_S, RSTATE);
                        eng->set(RSAVE_L, 1);
                        eng->set(RSAVE_R, 1);
                        eacc = 0;
                        blocks_since_check = 0;
                        wbits = 0;
                        in_lot = true;
                    }
                } else {
                    in_lot = false;
                    gl_checkpass = 0;
                    eacc = 0;
                    blocks_since_check = 0;
                    wbits = 0;
                }
                eng->copy(RSTART, RSTATE);
            }
            int b = mpz_tstbit(Echunk.get_mpz_t(), i - 1) ? 1 : 0;
            if (useFast3) { if (b) eng->square_mul(RSTATE, 3); else eng->square_mul(RSTATE); }
            else { eng->square_mul(RSTATE); if (b) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RSTATE, RTMP); } }
            wbits <<= 1; if (b) wbits += 1;
            bits_in_block += 1;
            if (options.erroriter > 0 && (resumeI - i + 1) == options.erroriter && !errordone) { errordone = true; eng->sub(RSTATE, 2); std::cout << "Injected error at iteration " << (resumeI - i + 1) << std::endl; if (guiServer_) { std::ostringstream oss; oss << "Injected error at iteration " << (resumeI - i + 1); guiServer_->appendLog(oss.str()); } }
            bool end_block = (bits_in_block == current_block_len);
            if (end_block) {
                if (current_block_len == B) {
                    eng->set_multiplicand(RTMP, RSTART);
                    eng->mul(RACC_L, RTMP);
                    eng->set_multiplicand(RTMP, RSTATE);
                    eng->mul(RACC_R, RTMP);
                    eacc += wbits;
                    blocks_since_check += 1;
                    gl_checkpass += 1;
                    /*if (!tunedCheckpass && options.checklevel == 0) {
                        uint64_t processedChunk = bits - i + 1;
                        double elapsedChunk = std::chrono::duration<double>(now - chunkStart).count();
                        if (elapsedChunk > 0.0 && processedChunk >= B) {
                            double sampleIps = (double)processedChunk / elapsedChunk;
                            uint64_t checkpasslevel_auto = (uint64_t)((sampleIps * desiredIntervalSeconds) / (double)B);
                            if (checkpasslevel_auto == 0) checkpasslevel_auto = std::max<uint64_t>(1, (bits / B) / (uint64_t)std::sqrt((double)B));
                            checkpass = checkpasslevel_auto;
                            tunedCheckpass = true;
                        }
                    }*/
                    bool doCheck = options.gerbiczli && in_lot && (gl_checkpass == checkpass || i == 1);
                    if (doCheck) {
                        std::cout << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                        eng->copy(RCHK, RACC_L);
                        for (uint64_t k = 0; k < B; ++k) eng->square_mul(RCHK);
                        eng->set(RPOW, 1);
                        size_t eb = mpz_sizeinbase(eacc.get_mpz_t(), 2);
                        for (size_t k = eb; k-- > 0;) {
                            if (useFast3) { if (mpz_tstbit(eacc.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); }
                            else { eng->square_mul(RPOW); if (mpz_tstbit(eacc.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } }
                        }
                        eng->set_multiplicand(RTMP, RPOW);
                        eng->mul(RCHK, RTMP);
                        mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
                        eng->get_mpz(z0, RCHK); eng->get_mpz(z1, RACC_R);
                        //if (mpz_cmp(z0, z1) != 0) throw std::runtime_error("Gerbicz-Li error checking failed!");
                        bool ok = (mpz_cmp(z0, z1) == 0);
                        //mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
                        //bool ok = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);

                        mpz_clears(z0, z1, nullptr);
                        //bool ok = eng->is_equal(RCHK, RACC_R);
                        if (!ok) { std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; guiServer_->appendLog(oss.str()); } options.gerbicz_error_count += 1; eng->copy(RSTATE, RSAVE_S); eng->set(RACC_L, 1); eng->set(RACC_R, 1); eng->copy(RSTART, RSTATE); i = (mp_bitcnt_t)(i + blocks_since_check * B); eacc = 0; blocks_since_check = 0; wbits = 0; gl_checkpass = 0; bits_in_block = 0; continue; }
                        else { std::cout << "[Gerbicz Li] Check passed\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Check passed\n"; guiServer_->appendLog(oss.str()); } eng->copy(RSAVE_S, RSTATE); eng->set(RACC_L, 1); eng->set(RACC_R, 1); eacc = 0; blocks_since_check = 0; gl_checkpass = 0; }
                    }
                } else {
                    if (options.gerbiczli) {
                        eng->copy(RCHK, RSTART);
                        for (uint64_t k = 0; k < current_block_len; ++k) eng->square_mul(RCHK);
                        eng->set(RPOW, 1);
                        size_t wb = mpz_sizeinbase(wbits.get_mpz_t(), 2);
                        for (size_t k = wb; k-- > 0;) {
                            if (useFast3) { if (mpz_tstbit(wbits.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); }
                            else { eng->square_mul(RPOW); if (mpz_tstbit(wbits.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } }
                        }
                        eng->set_multiplicand(RTMP, RPOW);
                        eng->mul(RCHK, RTMP);
                        //bool ok0 = eng->is_equal(RCHK, RSTATE);
                        mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
                        eng->get_mpz(z0, RCHK); eng->get_mpz(z1, RSTATE);
                        //if (mpz_cmp(z0, z1) != 0) throw std::runtime_error("Gerbicz-Li error checking failed!");
                        bool ok0 = (mpz_cmp(z0, z1) == 0);
                        //mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
                        //bool ok0 = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);
                        mpz_clears(z0, z1, nullptr);
                        if (!ok0) { std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n"; guiServer_->appendLog(oss.str()); } options.gerbicz_error_count += 1; eng->copy(RSTATE, RSTART); i = (mp_bitcnt_t)(i + current_block_len); wbits = 0; bits_in_block = 0; continue; }
                        else { std::cout << "[Gerbicz Li] Check passed\n"; if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz Li] Check passed\n"; guiServer_->appendLog(oss.str()); } eng->copy(RSAVE_S, RSTATE); eng->set(RACC_L, 1); eng->set(RACC_R, 1); eacc = 0; blocks_since_check = 0; gl_checkpass = 0; }
                    }
                }
                bits_in_block = 0;
                wbits = 0;
            }
            auto now2 = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now2 - lastDisplay).count() >= 10) {
                std::string res64_x;
                spinner.displayProgress2(
                    processed_total_bits + (bits - i + 1),
                    processed_total_bits + bits,
                    timer.elapsed() + restored_time,
                    timer2.elapsed(),
                    options.exponent,
                    processed_total_bits + (bits - i + 1),
                    processed_total_bits,
                    res64_x,
                    guiServer_ ? guiServer_.get() : nullptr,
                    chunkIndex,
                    estChunks,
                    (bits - i + 1),
                    bits,
                    false
                );
                timer2.start();
                lastDisplay = now2;
            }
        }
        if (bits_in_block != 0 && options.gerbiczli) {
            mpz_class wtail = wbits;
            uint64_t bt = bits_in_block;
            eng->copy(RCHK, RSTART);
            for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK);
            eng->set(RPOW, 1);
            size_t wbl = mpz_sizeinbase(wtail.get_mpz_t(), 2);
            for (size_t k = wbl; k-- > 0;) {
                if (useFast3) { if (mpz_tstbit(wtail.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); }
                else { eng->square_mul(RPOW); if (mpz_tstbit(wtail.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } }
            }
            eng->set_multiplicand(RTMP, RPOW);
            eng->mul(RCHK, RTMP);
            //bool ok_tail = eng->is_equal(RCHK, RSTATE);
            mpz_t z0, z1; mpz_inits(z0, z1, nullptr);
            eng->get_mpz(z0, RCHK); eng->get_mpz(z1, RSTATE);
            bool ok_tail = (mpz_cmp(z0, z1) == 0);
            //mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
            //bool ok_tail = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);
            mpz_clears(z0, z1, nullptr);
            if (!ok_tail) { eng->copy(RSTATE, RSTART); eng->copy(RCHK, RSTART); for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK); eng->set(RPOW, 1); size_t wbl2 = mpz_sizeinbase(wtail.get_mpz_t(), 2); for (size_t k = wbl2; k-- > 0;) { if (useFast3) { if (mpz_tstbit(wtail.get_mpz_t(), k)) eng->square_mul(RPOW, 3); else eng->square_mul(RPOW); } else { eng->square_mul(RPOW); if (mpz_tstbit(wtail.get_mpz_t(), k)) { eng->set_multiplicand(RTMP, RBASE); eng->mul(RPOW, RTMP); } } } eng->set_multiplicand(RTMP, RPOW); eng->mul(RSTATE, RTMP); }
            bits_in_block = 0;
            wbits = 0;
        }
        processed_total_bits += bits;
        restored = false;
        firstChunk = false;
        if (nextStart == 0) break;
        startPrime = nextStart | 1ULL;
        chunkIndex += 1;
    }
    const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
    std::string res64_done;
    spinner.displayProgress2(
        processed_total_bits,
        processed_total_bits,
        timer.elapsed() + restored_time,
        timer2.elapsed(),
        options.exponent,
        processed_total_bits,
        processed_total_bits,
        res64_done,
        guiServer_ ? guiServer_.get() : nullptr,
        std::max<uint64_t>(chunkIndex, estChunks),
        estChunks,
        1,
        1,
        true
    );
    std::cout << "Elapsed time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "Elapsed time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl; guiServer_->appendLog(oss.str()); }
    //engine::digit d(eng, RSTATE);
    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RSTATE), Mp);
    auto end_sys = std::chrono::system_clock::now();
    auto fmt = [](const std::chrono::system_clock::time_point& tp){
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
        std::time_t tt = system_clock::to_time_t(tp);
        std::tm tmv{};
        #if defined(_WIN32)
        gmtime_s(&tmv, &tt);
        #else
        std::tm* tmp = std::gmtime(&tt);
        if (tmp) tmv = *tmp;
        #endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        std::ostringstream s;
        s << buf << '.' << std::setw(3) << std::setfill('0') << (int)(ms.count());
        return s.str();
    };
    std::string ds = fmt(start_sys);
    std::string de = fmt(end_sys);
    if (options.resume) { writeEcmResumeLine("resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) + ".save", options.B1, options.exponent, X); convertEcmResumeToPrime95("resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) + ".save", "resume_p" + std::to_string(options.exponent) + "_B1_" + std::to_string(options.B1) + ".p95", ds, de); }
    X -= 1;
    mpz_class g = gcd_with_dots(X, Mp);
    bool factorFound = (g != 1) && (g != Mp);
    std::string filename = "stage1_result_B1_" + std::to_string(B1) + "_p_" + std::to_string(options.exponent) + ".txt";
    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B1=" + std::to_string(B1) + "  factor=" + std::string(fstr));
        std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
        if (guiServer_) { std::ostringstream oss; oss << "\nP-1 factor stage 1 found: " << fstr << std::endl; guiServer_->appendLog(oss.str()); }
        options.knownFactors.push_back(std::string(fstr));
        std::free(fstr);
        std::cout << "\n";
    } else {
        writeStageResult(filename, "No factor up to B1=" + std::to_string(B1));
        std::cout << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
        if (guiServer_) { std::ostringstream oss; oss << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl; guiServer_->appendLog(oss.str()); }
    }
    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage1", json);
    wm.appendToResultsTxt(json);

    if(options.B2 > 0){
        {
            const double elapsed_time_ck =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count()
                + restored_time;

            save_ckpt(
                0,                 // i
                elapsed_time_ck,   // et
                0,                 // chk
                0,                 // blks
                0,                 // bib
                0,                 // cbl
                0,                 // inlot
                mpz_class(0),      // ceacc
                mpz_class(0),      // cwbits
                chunkIndex,        // chunkIdx
                startPrime,        // startP
                firstChunk ? 1 : 0,// first
                processed_total_bits, // processedBits
                0                  // bitsInChunk
            );
        }
        //options.B2 = 214439;
        factorFound = runPM1Stage2Marin() || factorFound;
    }
   if(options.nmax > 0 && options.K > 0){
        {
            std::cout << "P-1 STAGE 2 IN **** n^K variant  n=" << options.nmax << " K=" << options.K << "******\n";
    
            const double elapsed_time_ck =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count()
                + restored_time;

            save_ckpt(
                0,                 // i
                elapsed_time_ck,   // et
                0,                 // chk
                0,                 // blks
                0,                 // bib
                0,                 // cbl
                0,                 // inlot
                mpz_class(0),      // ceacc
                mpz_class(0),      // cwbits
                chunkIndex,        // chunkIdx
                startPrime,        // startP
                firstChunk ? 1 : 0,// first
                processed_total_bits, // processedBits
                0                  // bitsInChunk
            );
        }
        //options.B2 = 214439;
        factorFound = runPM1Stage2MarinNKVersion() || factorFound;
    }
    //else{
    delete_checkpoints(options.exponent, options.wagstaff, true, false);
    delete eng;
    if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n";
            if (guiServer_) { std::ostringstream oss; oss << "Entry removed from " << options.worktodo_path << " and saved to worktodo_save.txt\n"; guiServer_->appendLog(oss.str()); }
            std::ifstream f(options.worktodo_path);
            std::string l; bool more = false; while (std::getline(f, l)) { if (!l.empty() && l[0] != '#') { more = true; break; } }
            f.close();
            if (more) { std::cout << "Restarting for next entry in worktodo.txt\n"; if (guiServer_) { std::ostringstream oss; oss << "Restarting for next entry in worktodo.txt\n"; guiServer_->appendLog(oss.str()); } restart_self(argc_, argv_); }
            else { std::cout << "No more entries in worktodo.txt, exiting.\n"; if (guiServer_) { std::ostringstream oss; oss << "No more entries in worktodo.txt, exiting.\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(0);} }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n"; if (guiServer_) { std::ostringstream oss; oss << "Failed to update " << options.worktodo_path << "\n"; guiServer_->appendLog(oss.str()); } if (!options.gui) {std::exit(-1);}
        }
    }
    //}
    //delete eng;
    return factorFound ? 0 : 1;
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

static std::atomic<bool> g_stop{false};
static void handle_signal(int) noexcept { g_stop = true; interrupted = true; }

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
        rc = runECMMarin();
        ran = true;
    }
    if (options.mode == "llsafe") {
        rc = runLlSafeMarin();
        ran = true;
    } else if (options.mode == "pm1" && options.marin /*&& options.B2 <= 0*/) {
        if (options.exponent > 89) {
            int rc_local = 0;
            bool ran_local = false;

            std::ostringstream s1; s1 << "pm1_m_" << options.exponent << ".ckpt";
            std::ostringstream s2; s2 << "pm1_s2_m_" << options.exponent << ".ckpt";
            const std::string s1f = s1.str();
            const std::string s2f = s2.str();

            bool haveS1 = File(s1f).exists() || File(s1f + ".old").exists();
            bool haveS2 = File(s2f).exists() || File(s2f + ".old").exists() || File(s2f + ".new").exists();

            if ((haveS2 || haveS1) && options.nmax == 0  && options.K == 0) {
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
