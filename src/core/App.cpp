// src/core/App.cpp
#define CL_TARGET_OPENCL_VERSION 200
#include "core/App.hpp"
#include "core/QuickChecker.hpp"
#include "core/Printer.hpp"
#include "math/Carry.hpp"

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif

#include <csignal>
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <atomic>

using namespace std::chrono;

namespace core {

static std::atomic<bool> interrupted{false};
static void handle_sigint(int) { interrupted = true; }

App::App(int argc, char** argv)
  : options(io::CliParser::parse(argc, argv))
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
        "kernel_ntt_radix4_radix2_square_radix2_radix4"
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

    math::Carry carry(
        context,
        queue,
        program->getProgram(),
        precompute.getN(),
        precompute.getDigitWidth()
    );

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
        

        carry.carryGPU(
            buffers->input,
            buffers->blockCarryBuf,
            precompute.getN() * sizeof(uint64_t)
        );
        queued += 2;
        if (queueCap > 0 && queued >= queueCap) { 
            //std::cout << "Flush\n";
            clFinish(queue); queued = 0;
            auto now = high_resolution_clock::now();
            
            
            if (now - lastDisplay >= seconds(2)) {
                std::vector<unsigned long> hostData(precompute.getN());
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
                timer.elapsed(),
                static_cast<int>(context.getTransformSize())
                );
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
                std::vector<unsigned long> hostData(precompute.getN());
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

        proofManager.checkpoint(buffers->input, iter);

        
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
    std::vector<unsigned long> hostData(precompute.getN());
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
    }

    
    double finalElapsed = timer.elapsed();

    std::string json = io::JsonBuilder::generate(
        hostData,
        options,
        precompute.getDigitWidth(),
        finalElapsed,
        static_cast<int>(context.getTransformSize())
    );
    if (options.submit)
        io::CurlClient::submit(json,
                              options.user,
                              options.password);

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

    bool isPrime = (options.mode == "prp")
                   ? (hostResult[0] == 9
                      && std::all_of(hostResult.begin()+1,
                                     hostResult.end(),
                                     [](uint64_t v){ return v == 0; }))
                   : std::all_of(hostResult.begin(),
                                 hostResult.end(),
                                 [](uint64_t v){ return v == 0; });
    backupManager.clearState();
    return isPrime ? 0 : 1;
}

} // namespace core
