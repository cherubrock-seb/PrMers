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
        options.debug
    );

    buffers.emplace(context, precompute);
    program.emplace(context, context.getDevice(), options.kernel_path, precompute);
    kernels.emplace(program->getProgram(), context.getQueue());
    nttEngine.emplace(context, *kernels, *buffers, precompute);
    

    std::vector<std::string> kernelNames = {
        "kernel_sub2",
        "kernel_carry",
        "kernel_carry_2",
        "kernel_inverse_ntt_radix4_mm",
        "kernel_ntt_radix4_last_m1_n4",
        "kernel_inverse_ntt_radix4_mm_last",
        "kernel_ntt_radix4_last_m1",
        "kernel_ntt_radix4_mm_first",
        "kernel_ntt_radix4_mm_m8",
        "kernel_ntt_radix4_mm_m4",
        "kernel_inverse_ntt_radix4_m1",
        "kernel_inverse_ntt_radix4_m1_n4",
        "kernel_ntt_radix4_inverse_mm_2steps",
        "kernel_ntt_radix4_mm_2steps",
        "kernel_ntt_radix4_mm_3steps",
        "kernel_ntt_radix2_square_radix2",
        "kernel_ntt_radix4_radix2_square_radix2_radix4",
        "kernel_pointwise_mul",
        "kernel_ntt_radix2",
        "kernel_res64_display"
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
    cl_mem outBuf;
    if (options.res64_display_interval != 0){
        cl_uint totalWords = (options.exponent - 1u) / 32u + 1u;
        size_t  outBytes   = size_t(totalWords) * sizeof(cl_uint);
        cl_int err;
        outBuf = clCreateBuffer(
            context.getContext(),
            CL_MEM_WRITE_ONLY,
            outBytes,
            nullptr,
            &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Erreur clCreateBuffer(outBuf): " << err << std::endl;
            std::exit(1);
        }
    }
    math::Carry carry(
        context,
        queue,
        program->getProgram(),
        precompute.getN(),
        precompute.getDigitWidth()
    );


    logger.logStart(options);
    timer.start();
    timer2.start();
    auto startTime  = high_resolution_clock::now();
    auto lastBackup = startTime;
    auto lastDisplay = startTime;
    spinner.displayProgress(resumeIter, totalIters, 0.0, 0.0, p,resumeIter,"");
    uint32_t lastIter = resumeIter;

    for (uint32_t iter = resumeIter; iter < totalIters && !interrupted; ++iter) {
        lastIter = iter;

        queued += nttEngine->forward(buffers->input, iter);
        /*if (queueCap > 0 && ++queued >= queueCap) { clFlush(queue); queued = 0; spinner.displayProgress(iter - resumeIter,
                                   totalIters,
                                   timer.elapsed(),
                                   p);}*/

        queued += nttEngine->inverse(buffers->input, iter);
        

        carry.carryGPU(
            buffers->input,
            buffers->blockCarryBuf,
            precompute.getN() * sizeof(uint64_t)
        );
        queued += 2;

        if ((options.res64_display_interval != 0)&& ( ((iter+1) % options.res64_display_interval) == 0 )) {
            cl_kernel dispK = kernels->getKernel("kernel_res64_display");
            cl_uint modeInt = (options.mode=="prp") ? 1u : 0u;
            cl_uint nWords    = static_cast<cl_uint>(precompute.getN());
            cl_uint iterdisp    = static_cast<cl_uint>(iter+1);
            clSetKernelArg(dispK, 0, sizeof(cl_mem), &buffers->input);
            clSetKernelArg(dispK, 1, sizeof(cl_uint), &options.exponent);
            clSetKernelArg(dispK, 2, sizeof(cl_uint), &nWords);
            clSetKernelArg(dispK, 3, sizeof(cl_uint), &modeInt);
            clSetKernelArg(dispK, 4, sizeof(cl_uint), &iterdisp);
            clSetKernelArg(dispK, 5, sizeof(cl_mem), &outBuf);

            size_t global = 1, local = 1;
            clEnqueueNDRangeKernel(
                queue,
                dispK,
                1,
                nullptr,
                &global,
                &local,
                0, nullptr, nullptr
            );
            queued += 1;
        }

        if ((options.iterforce > 0 && (iter+1)%options.iterforce == 0 && iter>0) || (options.iterforce==0 && (queueCap > 0 && queued >= queueCap))) { 
            //std::cout << "Flush\n";
            clFinish(queue);
        }
        if ((options.iterforce > 0 && (iter+1)%options.iterforce == 0 && iter>0) || queueCap==0 || (options.iterforce==0 &&(queueCap > 0 && queued >= queueCap))) { 
            
            queued = 0;
            
            auto now = high_resolution_clock::now();
            
            if ((options.iterforce > 0 && (iter+1)%options.iterforce == 0 && iter>0) || (options.iterforce==0 &&(((now - lastDisplay >= seconds(10)))) )) {
                std::string res64;


                spinner.displayProgress(
                    iter+1,
                    totalIters,
                    timer.elapsed(),
                    timer2.elapsed(),
                    p,
                    resumeIter,
                    res64
                );
                timer2.start();
                lastDisplay = now;
                resumeIter = iter+1;
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
                    hostData.size() * sizeof(uint64_t),
                    hostData.data(),
                    0, nullptr, nullptr
                );
                std::string res64 = io::JsonBuilder::computeRes64Iter(
                    hostData,
                    options,
                    precompute.getDigitWidth(),
                    backupElapsed,
                    static_cast<int>(context.getTransformSize())
                );
                spinner.displayBackupInfo(
                    iter+1,
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
            hostData.size() * sizeof(uint64_t),
            hostData.data(),
            0, nullptr, nullptr
        );

         
        carry.handleFinalCarry(hostData,
                               precompute.getDigitWidth());

        clEnqueueWriteBuffer(
            context.getQueue(),
            buffers->input,
            CL_TRUE, 0,
            hostData.size() * sizeof(uint64_t),
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
                        timer2.elapsed(),
                        p,
                        resumeIter,
                        res64_x
                    );
        backupManager.saveState(buffers->input, lastIter);
         if (options.res64_display_interval != 0){
            clReleaseMemObject(outBuf);
         }
        
                
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
