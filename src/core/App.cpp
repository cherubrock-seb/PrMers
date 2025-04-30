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
#define CL_TARGET_OPENCL_VERSION 200
#include "core/App.hpp"
#include "core/QuickChecker.hpp"
#include "core/Printer.hpp"
#include "math/Carry.hpp"
#include "io/WorktodoParser.hpp"
#include "io/WorktodoManager.hpp"
#include "io/CurlClient.hpp"
#include "math/GerbiczLiChecker.hpp"
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
#include <atomic>
#include <fstream>
#include <memory>
#include <optional>
#include <cmath>



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
        return args;
    }

    std::cout << "Loading options from config file: " << config_path << std::endl;

    while (std::getline(config, line)) {
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            args.push_back(token);
        }
    }

    if (!args.empty()) {
        std::cout << "Options from config file:" << std::endl;
        for (const auto& arg : args) {
            std::cout << "  " << arg << std::endl;
        }
    } else {
        std::cout << "No options found in config file." << std::endl;
    }

    return args;
}



void restart_self(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() > 1 && args[1].find_first_not_of("0123456789") == std::string::npos) {
        args.erase(args.begin() + 1); 
    }

#ifdef _WIN32
    std::string command = "\"" + args[0] + "\"";
    for (size_t i = 1; i < args.size(); ++i) {
        command += " \"" + args[i] + "\"";
    }
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (CreateProcessA(
            NULL,
            const_cast<char*>(command.c_str()),
            NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        exit(0);
    } else {
        std::cerr << "Failed to restart program (CreateProcess failed)" << std::endl;
    }

#else
    std::cout << "\nRestarting program without exponent:\n";
    for (const auto& arg : args) {
        std::cout << "   " << arg << std::endl;
    }

    std::vector<char*> exec_args;
    for (auto& s : args) exec_args.push_back(const_cast<char*>(s.c_str()));
    exec_args.push_back(nullptr);

    execv(exec_args[0], exec_args.data());

    std::cerr << "Failed to restart program (execv failed)" << std::endl;
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
          o.exponent = e->exponent;
          o.mode     = e->prpTest ? "prp" : "ll";
          o.aid      = e->aid;
          hasWorktodoEntry_ = true;
      }
      if (!hasWorktodoEntry_ && o.exponent == 0) {
        std::cerr << "Error: no valid entry in " 
                  << options.worktodo_path 
                  << " and no exponent provided on the command line\n";
        o.exponent = askExponentInteractively();
        o.mode = "prp";
        //std::exit(-1);
      }
      return o;
  }())
  , context(options.device_id,options.enqueue_max)
  , precompute(options.exponent)
  , backupManager(
        context.getQueue(),
        options.backup_interval,
        precompute.getN(),
        options.save_path,
        options.exponent,
        options.mode
    )
  , proofManager(
        options.exponent,
        options.proof,
        context.getQueue(),
        precompute.getN()
    )
  , spinner()
  , logger(options.output_path)
  , timer()
{
    worktodoParser_ = std::make_unique<io::WorktodoParser>(options.worktodo_path);
    if (auto e = worktodoParser_->parse()) {
        hasWorktodoEntry_ = true;
    }
    context.computeOptimalSizes(
        precompute.getN(),
        precompute.getDigitWidth(),
        options.exponent,
        options.debug
    );

    buffers.emplace(context, precompute);
    program.emplace(context, context.getDevice(), options.kernel_path);
    kernels.emplace(program->getProgram(), context.getQueue());
    nttEngine.emplace(context, *kernels, *buffers, precompute);
    {
        uint32_t tmp = options.exponent;
        int L = 0;
        while (tmp) { tmp >>= 1; ++L; }
        size_t B_GL = static_cast<size_t>(std::sqrt(L));
        checker = std::make_unique<math::GerbiczLiChecker>(3ULL, 1ULL, B_GL);

    }

    std::vector<std::string> kernelNames = {
        "kernel_sub2",
        "kernel_carry",
        "kernel_carry_2",
        "kernel_inverse_ntt_radix4_mm",
        "kernel_ntt_radix4_last_m1_n4",
        "kernel_inverse_ntt_radix4_mm_last",
        "kernel_ntt_radix4_last_m1",
        "kernel_ntt_radix4_mm_first",
        "kernel_ntt_radix4_mm",
        "kernel_inverse_ntt_radix4_m1",
        "kernel_inverse_ntt_radix4_m1_n4",
        "kernel_ntt_radix4_inverse_mm_2steps",
        "kernel_ntt_radix4_mm_2steps",
        "kernel_ntt_radix4_mm_3steps",
        "kernel_ntt_radix2_square_radix2",
        "kernel_ntt_radix4_radix2_square_radix2_radix4",
        "kernel_pointwise_mul",
        "kernel_ntt_radix2",
        "kernel_carry_mul_base"
    };
    for (auto& name : kernelNames) {
        kernels->createKernel(name);
    }

    std::signal(SIGINT, handle_sigint);
}

int App::run() {
    Printer::banner(options);
    if (auto code = QuickChecker::run(options.exponent))
        return *code;

    cl_command_queue queue   = context.getQueue();
    size_t          queueCap = context.getQueueSize();
    size_t          queued   = 0;

    uint32_t p = options.exponent;
    uint32_t totalIters = options.mode == "prp" ? p : p - 2;

    std::vector<uint64_t> x(precompute.getN(), 0ULL);
    uint32_t resumeIter = backupManager.loadState(x);
    if (resumeIter == 0) {
        x[0] = (options.mode == "prp") ? 3ULL : 4ULL;
        std::cout << "Initial x[0] set to " << x[0]
                  << " (" << (options.mode == "prp" ? "PRP" : "LL")
                  << " mode)" << std::endl;
    }
    buffers->input = clCreateBuffer(
        context.getContext(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        x.size() * sizeof(uint64_t),
        x.data(), nullptr
    );

    cl_mem dBuf = clCreateBuffer(
        context.getContext(),
        CL_MEM_READ_WRITE,
        precompute.getN() * sizeof(uint64_t), nullptr, nullptr
    );
    {
        std::vector<uint64_t> initD(precompute.getN(), 0ULL);
        initD[0] = 1;
        clEnqueueWriteBuffer(queue, dBuf, CL_TRUE, 0,
                             initD.size() * sizeof(uint64_t),
                             initD.data(), 0, nullptr, nullptr);
    }

    std::vector<uint64_t> xPrev(precompute.getN()), dPrev(precompute.getN());
    clEnqueueReadBuffer(queue, buffers->input, CL_TRUE, 0,
                        xPrev.size()*sizeof(uint64_t), xPrev.data(),
                        0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, dBuf, CL_TRUE, 0,
                        dPrev.size()*sizeof(uint64_t), dPrev.data(),
                        0, nullptr, nullptr);
    const size_t B = checker->getBlockSize();

    math::Carry carry(
        context,
        queue,
        program->getProgram(),
        precompute.getN(),
        precompute.getDigitWidth()
    );

    auto mulNTT = [&](cl_mem A, cl_mem B_mem) {
        queued += nttEngine->forward_simple(A, 0);
        queued += nttEngine->forward_simple(B_mem, 0);
        queued += nttEngine->pointwiseMul(A, B_mem);
        queued += nttEngine->inverse_simple(A, 0);
        carry.carryGPU(A, buffers->blockCarryBuf,
                       precompute.getN() * sizeof(uint64_t));
        queued += 2;
    };
    size_t N = precompute.getN();
    auto expModNTT = [&](cl_mem baseBuf, uint64_t exp, cl_mem resultBuf){
        std::vector<uint64_t> one(N,0);
        one[0] = 1;
        clEnqueueWriteBuffer(queue, resultBuf, CL_TRUE, 0,
                            N * sizeof(uint64_t), one.data(),
                            0, nullptr, nullptr);
        cl_mem tmpBuf = clCreateBuffer(context.getContext(),
                                    CL_MEM_READ_WRITE,
                                    N * sizeof(uint64_t),
                                    nullptr, nullptr);
        clEnqueueCopyBuffer(queue, baseBuf, tmpBuf, 0, 0,
                            N * sizeof(uint64_t),
                            0, nullptr, nullptr);
        while (exp) {
            if (exp & 1) {
                mulNTT(resultBuf, tmpBuf);
            }
            mulNTT(tmpBuf, tmpBuf);
            exp >>= 1;
        }
        clReleaseMemObject(tmpBuf);
    };

    logger.logStart(options);
    timer.start();

    auto startTime  = high_resolution_clock::now();
    auto lastBackup = startTime;
    auto lastDisplay = startTime;
    spinner.displayProgress(resumeIter, totalIters, 0.0, p,resumeIter,"");
    uint32_t lastIter = resumeIter;

    for (uint32_t iter = resumeIter; iter < totalIters && !interrupted; ++iter) {
        lastIter = iter;

        queued += nttEngine->forward(buffers->input, iter);
        /*if (queueCap > 0 && ++queued >= queueCap) { clFlush(queue); queued = 0; spinner.displayProgress(iter - resumeIter,
                                   totalIters,
                                   timer.elapsed(),
                                   p);}*/

        queued += nttEngine->inverse(buffers->input, iter);
        
       if(options.gerbiczli){
            carry.carryGPU_mul_base(
                buffers->input,
                buffers->blockCarryBuf,
                precompute.getN() * sizeof(uint64_t)
            );
        }
        else{
            carry.carryGPU(
                buffers->input,
                buffers->blockCarryBuf,
                precompute.getN() * sizeof(uint64_t)
            );
        }
        queued += 2;
        if (queueCap > 0 && queued >= queueCap) { 
            //std::cout << "Flush\n";
            clFinish(queue);
        }
        if (queueCap==0 || (queueCap > 0 && queued >= queueCap)) { 
            queued = 0;
            auto now = high_resolution_clock::now();
            
            
            if (now - lastDisplay >= seconds(2)) {
                std::vector<uint64_t>  hostData(precompute.getN());
                clEnqueueReadBuffer(
                    context.getQueue(),
                    buffers->input,
                    CL_TRUE, 0,
                    hostData.size() * sizeof(unsigned long),
                    hostData.data(),
                    0, nullptr, nullptr
                );
                std::string res64;
                if(queueCap > 0){
                    res64 = io::JsonBuilder::computeRes64(
                    hostData,
                    options,
                    precompute.getDigitWidth(),
                    timer.elapsed(),
                    static_cast<int>(context.getTransformSize())
                    );
                }

                spinner.displayProgress(
                    iter,
                    totalIters,
                    timer.elapsed(),
                    p,
                    resumeIter,
                    res64
                );
                lastDisplay = now;
            }
            if (now - lastBackup >= seconds(options.backup_interval)) {
                backupManager.saveState(buffers->input, iter);
                lastBackup = now;
                double backupElapsed = timer.elapsed();
                std::vector<uint64_t> hostData(precompute.getN());
                clEnqueueReadBuffer(
                    context.getQueue(),
                    buffers->input,
                    CL_TRUE, 0,
                    hostData.size() * sizeof(unsigned long),
                    hostData.data(),
                    0, nullptr, nullptr
                );
                std::string res64 = io::JsonBuilder::computeRes64(
                    hostData,
                    options,
                    precompute.getDigitWidth(),
                    backupElapsed,
                    static_cast<int>(context.getTransformSize())
                );
                spinner.displayBackupInfo(
                    iter,
                    totalIters,
                    backupElapsed,
                    res64
                );
            }    
        }
        
        if (options.mode == "ll") {
            kernels->runSub2(buffers->input);
            if (queueCap > 0 && ++queued >= queueCap) { clFlush(queue); queued = 0; }
        }


        if (options.gerbiczli && ((iter+1) % B) == 0) {
            // partial checkpoint d ← d * x
            mulNTT(dBuf, buffers->input);

            // r1 = d * x
            cl_mem r1Buf = clCreateBuffer(context.getContext(),
                                        CL_MEM_READ_WRITE,
                                        N * sizeof(uint64_t),
                                        nullptr, nullptr);
            clEnqueueCopyBuffer(queue, dBuf, r1Buf, 0, 0,
                                N * sizeof(uint64_t), 0, nullptr, nullptr);
            mulNTT(r1Buf, buffers->input);

            // aPow = a^B
            uint64_t aVal = (options.mode == "prp" ? 3ULL : 4ULL);
            cl_mem aBaseBuf = clCreateBuffer(context.getContext(),
                                            CL_MEM_READ_WRITE,
                                            N * sizeof(uint64_t),
                                            nullptr, nullptr);
            {
                std::vector<uint64_t> tmp(N,0);
                tmp[0] = aVal;
                clEnqueueWriteBuffer(queue, aBaseBuf, CL_TRUE, 0,
                                    N * sizeof(uint64_t), tmp.data(),
                                    0, nullptr, nullptr);
            }
            cl_mem aPowBuf = clCreateBuffer(context.getContext(),
                                            CL_MEM_READ_WRITE,
                                            N * sizeof(uint64_t),
                                            nullptr, nullptr);
            expModNTT(aBaseBuf, B, aPowBuf);

            // dPrevPow = dPrev^(2^B)
            cl_mem dPrevBuf = clCreateBuffer(context.getContext(),
                                            CL_MEM_READ_WRITE,
                                            N * sizeof(uint64_t),
                                            nullptr, nullptr);
            clEnqueueWriteBuffer(queue, dPrevBuf, CL_TRUE, 0,
                                N * sizeof(uint64_t), dPrev.data(),
                                0, nullptr, nullptr);
            cl_mem dPrevPowBuf = clCreateBuffer(context.getContext(),
                                                CL_MEM_READ_WRITE,
                                                N * sizeof(uint64_t),
                                                nullptr, nullptr);
            expModNTT(dPrevBuf, uint64_t(1) << B, dPrevPowBuf);

            // r2 = aPow * dPrevPow
            mulNTT(aPowBuf, dPrevPowBuf);
            cl_mem r2Buf = aPowBuf;

            clFinish(queue);

            std::vector<uint64_t> hostR1(N), hostR2(N);
            clEnqueueReadBuffer(queue, r1Buf, CL_TRUE, 0,
                                N * sizeof(uint64_t), hostR1.data(),
                                0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, r2Buf, CL_TRUE, 0,
                                N * sizeof(uint64_t), hostR2.data(),
                                0, nullptr, nullptr);

            if (hostR1 != hostR2) {
                clEnqueueWriteBuffer(queue, buffers->input, CL_TRUE, 0,
                                    N * sizeof(uint64_t), xPrev.data(),
                                    0, nullptr, nullptr);
                clEnqueueWriteBuffer(queue, dBuf, CL_TRUE, 0,
                                    N * sizeof(uint64_t), dPrev.data(),
                                    0, nullptr, nullptr);
                clFinish(queue);
                std::cerr << "Gerbicz–Li failed at iter=" << (iter+1) << "\n";
                clReleaseMemObject(r1Buf);
                clReleaseMemObject(r2Buf);
                clReleaseMemObject(dPrevBuf);
                clReleaseMemObject(dPrevPowBuf);
                break;
            }

            // update snapshots
            std::vector<uint64_t> hostX(N), hostD(N);
            clEnqueueReadBuffer(queue, buffers->input, CL_TRUE, 0,
                                N * sizeof(uint64_t), hostX.data(),
                                0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, dBuf, CL_TRUE, 0,
                                N * sizeof(uint64_t), hostD.data(),
                                0, nullptr, nullptr);
            xPrev = hostX;
            dPrev = hostD;

            clReleaseMemObject(r1Buf);
            clReleaseMemObject(r2Buf);
            clReleaseMemObject(dPrevBuf);
            clReleaseMemObject(dPrevPowBuf);
        }


        //proofManager.checkpoint(buffers->input, iter);

    }

    if (interrupted) {
        std::cout << "\nInterrupted signal received\n " << std::endl;
        clFinish(queue);
        backupManager.saveState(buffers->input, lastIter);
        std::cout << "\nInterrupted by user, state saved at iteration "
                  << lastIter << std::endl;
        return 0;
    }
    if (queued > 0) {
        clFinish(queue);
    }
    std::vector<uint64_t> hostData(precompute.getN());
    std::string res64_x;  

    {
        clEnqueueReadBuffer(
            context.getQueue(),
            buffers->input,
            CL_TRUE, 0,
            hostData.size() * sizeof(unsigned long),
            hostData.data(),
            0, nullptr, nullptr
        );

         
        carry.handleFinalCarry(hostData,
                               precompute.getDigitWidth());

        clEnqueueWriteBuffer(
            context.getQueue(),
            buffers->input,
            CL_TRUE, 0,
            hostData.size() * sizeof(unsigned long),
            hostData.data(),
            0, nullptr, nullptr
        );
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
                        p,
                        resumeIter,
                        res64_x
                    );
    }

    
    double finalElapsed = timer.elapsed();

    std::string json = io::JsonBuilder::generate(
        hostData,
        options,
        precompute.getDigitWidth(),
        finalElapsed,
        static_cast<int>(context.getTransformSize())
    );


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

    auto now2 = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now2);
    char timestampBuf[100];
    std::strftime(timestampBuf,
                  sizeof(timestampBuf),
                  "%Y-%m-%dT%H:%M:%SZ",
                  std::gmtime(&t));

    Printer::finalReport(
        options,
        hostResult,
        res64_x,
        precompute.getN(),
        timestampBuf,
        finalElapsed,
        json
    );
    bool skippedSubmission = false;

    if (options.submit) {
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


    bool isPrime = (options.mode == "prp")
                   ? (hostResult[0] == 9
                      && std::all_of(hostResult.begin()+1,
                                     hostResult.end(),
                                     [](uint64_t v){ return v == 0; }))
                   : std::all_of(hostResult.begin(),
                                 hostResult.end(),
                                 [](uint64_t v){ return v == 0; });
    backupManager.clearState();
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, options.mode, json);
    wm.appendToResultsTxt(json);

    if (hasWorktodoEntry_) {
        if (worktodoParser_->removeFirstProcessed()) {
            std::cout << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";

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
                restart_self(argc_, argv_);
            } else {
                std::cout << "No more entries in worktodo.txt, exiting.\n";
                std::exit(0);
            }
        } else {
            std::cerr << "Failed to update " << options.worktodo_path << "\n";
            std::exit(-1);
        }
    }


    return isPrime ? 0 : 1;
}

} // namespace core
