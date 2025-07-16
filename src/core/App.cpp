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
#include "math/Carry.hpp"
#include "io/WorktodoParser.hpp"
#include "io/WorktodoManager.hpp"
#include "io/CurlClient.hpp"
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
#include <atomic>
#include <fstream>
#include <memory>
#include <optional>
#include <cmath>
#include <thread>
#include <gmp.h>
#include <cstddef>
#include <deque>

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

inline void writeStageResult(const std::string& file,
                             const std::string& message)
{
    std::ofstream out(file, std::ios::app);
    if (!out) {
        std::cerr << "Cannot open " << file << " for writing\n";
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

    std::vector<uint64_t> x(precompute.getN(), 0ULL);
    x[0] = (options.mode == "prp") ? 3ULL : 4ULL;
    cl_mem inputBuf = clCreateBuffer(
        context.getContext(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        x.size() * sizeof(uint64_t),
        x.data(), nullptr
    );

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
        nttEngine->forward(inputBuf, iter - 1);
        nttEngine->inverse(inputBuf, iter - 1);
        carry.carryGPU(
            inputBuf,
            buffers->blockCarryBuf,
            precompute.getN() * sizeof(uint64_t)
        );

        if (iter % options.iterforce == 0) {
            
            clFinish(context.getQueue());
            //usleep(10000);
            //clFlush(context.getQueue());

            std::cout << "";
        }
        if (iter % markInterval == 0) {
            std::cout << "." << std::flush;
        }
    }

    clFinish(context.getQueue());
    //usleep(10000);
    auto end = high_resolution_clock::now();

    std::cout << "]\n";

    clReleaseMemObject(inputBuf);
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
  , context(options.device_id,options.enqueue_max,options.cl_queue_throttle_active)
  , precompute(options.exponent)
  , backupManager(
        context.getQueue(),
        options.backup_interval,
        precompute.getN(),
        options.save_path,
        options.exponent,
        options.mode,
        options.B1,
        options.B2
    )
  , proofManager(
        options.exponent,
        options.proof ? ProofSet::bestPower(options.exponent) : 0,
        context.getQueue(),
        precompute.getN(),
        precompute.getDigitWidth()
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
        options.max_local_size1
    );
    buffers.emplace(context, precompute);
    program.emplace(context, context.getDevice(), options.kernel_path, precompute,options.build_options);
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
        "kernel_ntt_inverse_radix5_mm_last"
    };
    for (auto& name : kernelNames) {
        kernels->createKernel(name);
    }
    nttEngine.emplace(context, *kernels, *buffers, precompute, options.mode == "pm1");


    std::signal(SIGINT, handle_sigint);
}



int App::runPrpOrLl() {
    if (options.tune ) {
        tuneIterforce();
        return 0;
    }
    Printer::banner(options);
    if (auto code = QuickChecker::run(options.exponent))
        return *code;

    cl_command_queue queue   = context.getQueue();
    size_t          queued   = 0;
    
    uint64_t p = options.exponent;
    uint64_t totalIters = options.mode == "prp" ? p : p - 2;

    std::vector<uint64_t> x(precompute.getN(), 0ULL);
    uint64_t resumeIter = backupManager.loadState(x);
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
   /* if (options.res64_display_interval != 0){
        cl_mem outBuf;
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
    }*/
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
        int proofPower = ProofSet::bestPower(options.exponent);
        options.proofPower = proofPower;
        double diskUsageGB = ProofSet::diskUsageGB(options.exponent, proofPower);
        std::cout << "Proof of power " << proofPower << " requires about "
                << std::fixed << std::setprecision(2) << diskUsageGB
                << "GB of disk space" << std::endl;
    }

    logger.logStart(options);
    timer.start();
    timer2.start();
    auto startTime  = high_resolution_clock::now();
    auto lastBackup = startTime;
    auto lastDisplay = startTime;
    spinner.displayProgress(resumeIter, totalIters, 0.0, 0.0, p,resumeIter, resumeIter,"");
    uint64_t lastIter = resumeIter;
    uint64_t startIter = resumeIter;
    
    
    for (uint64_t iter = resumeIter; iter < totalIters && !interrupted; ++iter) {
        lastIter = iter;

        
        queued += nttEngine->forward(buffers->input, iter);
        /*if (queueCap > 0 && ++queued >= queueCap) { clFlush(queue); queued = 0; spinner.displayProgress(iter - resumeIter,
                                   totalIters,
                                   timer.elapsed(),
                                   p);}*/
        //usleep(1500);
        queued += nttEngine->inverse(buffers->input, iter);
        //usleep(1500);
       
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
                    }).detach();
                }


            }      
            /*    
            GPU display // not used
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
            queued += 1;*/
        }
        auto now = high_resolution_clock::now();
          
        if ((options.iterforce > 0 && (iter+1)%options.iterforce == 0 && iter>0) || (((iter+1)%options.iterforce == 0))) { 
            
            if((iter+1)%100000 != 0){
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
                    }).detach();
                }


            } 
            
            //clFinish(context.getQueue());
            //std::cout << "800 usleep(1500) are done"<< std::endl;

            //if (totalUs_.count() == 0) {
            /*    auto t3 = std::chrono::high_resolution_clock::now();
                clFinish(context.getQueue());
                auto t4 = std::chrono::high_resolution_clock::now();
                
                auto t1 = std::chrono::high_resolution_clock::now();
                totalUs_ = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
                std::chrono::microseconds totalClfinish_ = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
                
                unitWait = totalUs_.count()/(4*(options.iterforce+1));
                std::cout << "totalUs_.count() us " << totalUs_.count() << std::endl;
                std::cout << "totalClfinish_.count() us " << totalClfinish_.count() << std::endl;
                
                t0 = std::chrono::high_resolution_clock::now();*/
                //std::cout << "WAIT IS " << unitWait << std::endl;
                //clReleaseEvent(markerEvent);
            //}
            /*else{
                usleep(1123765);
            }*/
            
            if ((((now - lastDisplay >= seconds(10)))) ) {
                std::string res64_x;
                /*std::vector<uint64_t> hostData(precompute.getN());
                clEnqueueReadBuffer(
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
                        );*/

                spinner.displayProgress(
                    iter+1,
                    totalIters,
                    timer.elapsed(),
                    timer2.elapsed(),
                    p,
                    resumeIter,
                    startIter,
                    res64_x
                );
                timer2.start();
                lastDisplay = now;
                resumeIter = iter+1;
            }
            queued = 0;
            if ((now - lastBackup >= seconds(options.backup_interval))) {
                std::string res64_x;
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
        
        
        if (options.mode == "ll") {
            kernels->runSub2(buffers->input);
        }


        if (options.proof && iter + 1 < totalIters) {
            proofManager.checkpoint(buffers->input, iter + 1);
        }
    }
    
    if (interrupted) {
        std::cout << "\nInterrupted signal received\n " << std::endl;
        clFinish(queue);
        queued = 0;
        backupManager.saveState(buffers->input, lastIter);
        std::cout << "\nInterrupted by user, state saved at iteration "
                  << lastIter << std::endl;
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
                        p,
                        resumeIter,
                        startIter,
                        res64_x
                    );
        backupManager.saveState(buffers->input, lastIter);
         /*if (options.res64_display_interval != 0){
            clReleaseMemObject(outBuf);
         }*/
        
                
    }
    
    double gpuElapsed = timer.elapsed();
    std::cout << "Total GPU time: " << gpuElapsed << " seconds." << std::endl;

    // Generate proof file after successful completion
    if (options.proof) {
        try {
            std::cout << "\nGenerating PRP proof file..." << std::endl;
            auto proofFilePath = proofManager.proof();
            options.proofFile = proofFilePath.string();  // Set proof file path
            std::cout << "Proof file saved: " << proofFilePath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Proof generation failed: " << e.what() << std::endl;
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

    //std::time_t t = std::chrono::system_clock::to_time_t(now2);
    char timestampBuf[32];
    std::time_t t = std::time(nullptr);
    std::tm timeinfo;

    #ifdef _WIN32
        gmtime_s(&timeinfo, &t);
    #else
        std::tm* tmp = std::gmtime(&t);
        if (tmp != nullptr)
            timeinfo = *tmp;
    #endif

    std::strftime(timestampBuf, sizeof(timestampBuf), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);

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
    mpz_class limit2(lim2);
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
struct MpzWrapper {
    mpz_t val;
    MpzWrapper() { mpz_init(val); }
    ~MpzWrapper() { mpz_clear(val); }

    mpz_t& get() { return val; }
    const mpz_t& get() const { return val; }
};

void vectToMpz2(mpz_t out,
                const std::vector<uint64_t>& v,
                const std::vector<int>& widths,
                const mpz_t Mp)
{
    const size_t n = v.size();
    const unsigned T = std::thread::hardware_concurrency();
    std::vector<MpzWrapper> partial(T);

    std::vector<unsigned> total_width(T, 0);
    std::atomic<size_t> global_count{0};

    for (unsigned t = 0; t < T; ++t)
        mpz_init(partial[t].get());

    std::vector<std::thread> threads(T);
    size_t chunk = (n + T - 1) / T;

    for (unsigned t = 0; t < T; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, n);
        threads[t] = std::thread([&, start, end, t]() {
            mpz_t acc;
            mpz_init_set_ui(acc, 0);
            for (ptrdiff_t i = ptrdiff_t(end) - 1; i >= ptrdiff_t(start); --i) {
                mpz_mul_2exp(acc, acc, widths[i]);
                mpz_add_ui(acc, acc, v[i]);
                if (mpz_cmp(acc, Mp) >= 0)
                    mpz_sub(acc, acc, Mp);
                total_width[t] += widths[i];

                size_t count = ++global_count;
                if (count % 10000 == 0 || count == n) {
                    double progress = 100.0 * count / n;
                    printf("\rProgress: %.2f%%", progress);
                    fflush(stdout);
                }
            }
            mpz_set(partial[t].get(), acc);
            mpz_clear(acc);
        });
    }

    for (auto& th : threads) th.join();
    printf("\n");

    mpz_set_ui(out, 0);
    for (int t = T - 1; t >= 0; --t) {
        mpz_mul_2exp(out, out, total_width[t]);
        mpz_add(out, out, partial[t].get());
        if (mpz_cmp(out, Mp) >= 0)
            mpz_sub(out, out, Mp);
        //mpz_clear(partial[t].get());
    }
}

void App::gpuCopy(cl_command_queue q, cl_mem src, cl_mem dst, size_t bytes)
{
    clEnqueueCopyBuffer(context.getQueue(), src, dst, 0, 0, bytes, 0, nullptr, nullptr);
}

void App::gpuSquareInPlace(
                                    cl_mem A,
                                    math::Carry& carry,
                                    size_t limbBytes,
                                    cl_mem blockCarryBuf)
{
    buffers->input = A;
    nttEngine->forward(buffers->input, 0);   
    //ntt.pointwiseMul(A, A);
    nttEngine->inverse(buffers->input, 0); 
    //carry.carryGPU(A, blockCarryBuf, limbBytes);
}

void App::gpuMulInPlace(
                                 cl_mem A, cl_mem B,
                                 math::Carry& carry,
                                 size_t limbBytes,
                                 cl_mem blockCarryBuf)
{
    
    gpuCopy(context.getQueue(), A, buffers->input, limbBytes);
    cl_int err;
    nttEngine->forward_simple(buffers->input, 0);
    cl_mem temp = clCreateBuffer(
        context.getContext(),
        CL_MEM_READ_WRITE,
        limbBytes,
        nullptr,
        &err
    );
    gpuCopy(context.getQueue(), buffers->input, temp, limbBytes);
   
    gpuCopy(context.getQueue(), B, buffers->input, limbBytes);
    nttEngine->forward_simple(buffers->input, 0);
    nttEngine->pointwiseMul(buffers->input, temp);
    nttEngine->inverse_simple(buffers->input, 0);
    clReleaseMemObject(temp);
    
}

void App::subOneGPU(cl_mem buf)
{
    kernels->runSub1(buf);
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
    if (B2 <= B1) { std::cerr << "Stage 2 error B2 < B1.\n"; return -1; }

    if (debug) std::cout << "[DEBUG] Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;
    std::cout << "\nStart a P-1 factoring : Stage 2 Bounds: B1 = " << B1 << ", B2 = " << B2 << std::endl;

    unsigned long nbEven = evenGapBound(B2);
    size_t limbs = precompute.getN();
    size_t limbBytes = limbs * sizeof(uint64_t);

    cl_int err;
    cl_mem Hbuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    clEnqueueCopyBuffer(context.getQueue(), buffers->input, Hbuf, 0, 0, limbBytes, 0, nullptr, nullptr);

    math::Carry carry(context, context.getQueue(), program->getProgram(),
                      precompute.getN(), precompute.getDigitWidth(), buffers->digitWidthMaskBuf);
    mpz_t Mp; mpz_init(Mp);
    mpz_ui_pow_ui(Mp, 2, options.exponent);
    mpz_sub_ui(Mp, Mp, 1);

    std::vector<cl_mem> evenPow(nbEven);
    std::cout << "Stage 2: Will precompute " << nbEven << " powers of H^2, H^.." << "." << std::endl;

    nttEngine->forward(buffers->input, 0);
    nttEngine->inverse(buffers->input, 0);
    carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);

    evenPow[0] = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    gpuCopy(context.getQueue(), buffers->input, evenPow[0], limbBytes);
    auto ensureEvenPow = [&](unsigned long needIdx) {
        while (evenPow.size() <= needIdx) {
            size_t kPrev = evenPow.size() - 1;
            cl_mem buf = clCreateBuffer(context.getContext(),
                                        CL_MEM_READ_WRITE,
                                        limbBytes, nullptr, &err);

            gpuCopy(context.getQueue(), evenPow[kPrev], buf, limbBytes);
            gpuMulInPlace(buf, evenPow[0], carry, limbBytes, buffers->blockCarryBuf);
            carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
            gpuCopy(context.getQueue(), buffers->input, buf, limbBytes);

            evenPow.push_back(buf);
        }
    };
    int pct = -1;
    std::cout << "Precomputing H powers: 0%" << std::flush;
    for (unsigned long k = 1; k < nbEven; ++k) {
        evenPow[k] = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
        gpuCopy(context.getQueue(), evenPow[k - 1], evenPow[k], limbBytes);
        gpuMulInPlace(evenPow[k], evenPow[0], carry, limbBytes, buffers->blockCarryBuf);
        carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
        gpuCopy(context.getQueue(), buffers->input, evenPow[k], limbBytes);
        int newPct = int((k + 1) * 100 / nbEven);
        if (newPct > pct) { pct = newPct; std::cout << "\rPrecomputing H powers: " << pct << "%" << std::flush; }
    }
    std::cout << "\rPrecomputing H powers: 100%" << std::endl;

    cl_mem Hq = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    gpuCopy(context.getQueue(), Hbuf, buffers->input, limbBytes);

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
            gpuMulInPlace(buffers->input, Hbuf, carry, limbBytes, buffers->blockCarryBuf);
            carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
        }
    }
    gpuCopy(context.getQueue(), buffers->input, Hq, limbBytes);

    cl_mem Qbuf = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);
    std::vector<uint64_t> one(limbs, 0ULL); one[0] = 1ULL;
    clEnqueueWriteBuffer(context.getQueue(), Qbuf, CL_TRUE, 0, limbBytes, one.data(), 0, nullptr, nullptr);

    cl_mem tmp = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, limbBytes, nullptr, &err);




    size_t     idx     = 0;
    uint64_t resumeIdx = 0;
    //uint64_t resumeIdx = backupManager.loadStatePM1S2(Hq, Qbuf, limbBytes);

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
            gpuCopy(context.getQueue(), Hq, buffers->input, limbBytes);
            mpz_class d = p - p_prev;
            unsigned long idxGap = mpz_get_ui(d.get_mpz_t()) / 2 - 1;
            ensureEvenPow(idxGap);
            cl_mem Hd = evenPow[idxGap];

            gpuMulInPlace(buffers->input, Hd, carry, limbBytes, buffers->blockCarryBuf);
            carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
            gpuCopy(context.getQueue(), buffers->input, Hq, limbBytes);
        }

        gpuCopy(context.getQueue(), buffers->input, tmp, limbBytes);
        subOneGPU(tmp);
        gpuCopy(context.getQueue(), Qbuf, buffers->input, limbBytes);
        gpuMulInPlace(buffers->input, tmp, carry, limbBytes, buffers->blockCarryBuf);
        carry.carryGPU(buffers->input, buffers->blockCarryBuf, limbBytes);
        gpuCopy(context.getQueue(), buffers->input, Qbuf, limbBytes);

        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - lastDisplay).count() >= 3) {
            double done = static_cast<double>(idx + 1);
            double percent = totalPrimes ? done / static_cast<double>(totalPrimes) * 100.0 : 0.0;
            double elapsedSec = duration<double>(now - start).count();
            double ips = done / elapsedSec;
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
            lastDisplay = now;
        }
        /*if (duration_cast<seconds>(now - lastBackup).count() >= 180) {
            backupManager.saveStatePM1S2(Hq, Qbuf, idx, limbBytes);
            lastBackup = now;
        }*/
        if (options.iterforce > 0 && (idx + 1) % options.iterforce == 0) {
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
            //backupManager.saveStatePM1S2(Hq, Qbuf, idx, limbBytes);
            return 0;
        }

    }

    std::vector<uint64_t> hostQ(limbs);
    clEnqueueReadBuffer(context.getQueue(), Qbuf, CL_TRUE, 0, limbBytes, hostQ.data(), 0, nullptr, nullptr);
    carry.handleFinalCarry(hostQ, precompute.getDigitWidth());
    mpz_t Q; mpz_init(Q); vectToMpz2(Q, hostQ, precompute.getDigitWidth(), Mp);
    mpz_t g; mpz_init(g); mpz_gcd(g, Q, Mp);
    bool found = mpz_cmp_ui(g, 1) && mpz_cmp(g, Mp);
    std::string filename = "stage2_result_B2_" + B2.get_str() +
                        "_p_" + std::to_string(options.exponent) + ".txt";

    if (found) {
        char* s = mpz_get_str(nullptr, 10, g);
        writeStageResult(filename, "B2=" + B2.get_str() + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 2) found : " << s << '\n';
        std::free(s);
    } else {
        writeStageResult(filename, "No factor P-1 up to B2=" + B2.get_str());
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
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



int App::runPM1() {

    uint64_t B1 = options.B1;

    std::cout << "Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
    mpz_class E = backupManager.loadExponent();
    if(E==0){    
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
                    ""
                );
    for (mp_bitcnt_t i = resumeIter; i > 0; --i) {
        lastIter = i;
        if (interrupted) {
            std::cout << "\nInterrupted signal received\n " << std::endl;
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
                    res64_x
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
        res64_x
    );
    backupManager.saveState(buffers->input, lastIter, &E);
    
    std::cout << "\nStart get result from GPU" << std::endl;
    std::vector<uint64_t> hostData(precompute.getN());

    clEnqueueReadBuffer(context.getQueue(), buffers->input, CL_TRUE, 0,
                        hostData.size() * sizeof(uint64_t), hostData.data(),
                        0, nullptr, nullptr);
    std::cout << "Handle final carry start" << std::endl;

    carry.handleFinalCarry(hostData, precompute.getDigitWidth());
    std::cout << "vectToResidue start" << std::endl;

    mpz_t X; mpz_init(X);
    mpz_t Mp;  mpz_init(Mp);
    mpz_ui_pow_ui(Mp, 2, options.exponent);
    mpz_sub_ui(Mp, Mp, 1);
    vectToMpz2(X, hostData, precompute.getDigitWidth(), Mp);
    //std::cout << "digitWidths = ";
    //for (int w : precompute.getDigitWidth()) std::cout << w << " ";
    //std::cout << "\n";
    //gmp_printf("X final  = %Zd\n", X);

    mpz_sub_ui(X, X, 1);


    
    mpz_t g; mpz_init(g);
    mpz_gcd(g, X, Mp);


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

    bool factorFound = mpz_cmp_ui(g, 1) && mpz_cmp(g, Mp);

    std::string filename = "stage1_result_B1_" + std::to_string(B1) +
                        "_p_" + std::to_string(options.exponent) + ".txt";

    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g);
        writeStageResult(filename, "B1=" + std::to_string(B1) + "  factor=" + std::string(fstr));
        std::free(fstr);
    } else {
        writeStageResult(filename, "No factor up to B1=" + std::to_string(B1));
    }


    if (factorFound) {
        char* fstr = mpz_get_str(nullptr, 10, g);
        std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
        std::free(fstr);
        std::cout << "\n";
        if(options.B2>0){
            runPM1Stage2();
        }
//        else{
            //backupManager.clearState();
//        }
        return 0;
    }
    std::cout << "\nNo P-1 (stage 1) factor up to B1=" << B1 << "\n" << std::endl;
    if(options.B2>0){
        runPM1Stage2();
    }
/*    else{
            backupManager.clearState();
    }
    backupManager.clearState();*/
    return 1;
}




int App::run() {
    if(options.mode == "prp" || options.mode == "ll"){
        return runPrpOrLl();
    }
    else if(options.mode == "pm1"){
        if(options.exponent > 89){
            return runPM1();
        }
        else{
            std::cout << "P-1 factoring (stage 1) need exponent > 89" << std::endl;
        }
    }

    return 1;
}


} // namespace core
