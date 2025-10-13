#include "core/App.hpp"
#include "core/AlgoUtils.hpp"
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

using namespace core;
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
        
        queued += static_cast<std::size_t>(
              nttEngine->forward(buffers->input, static_cast<std::size_t>(iter)));
        queued += static_cast<std::size_t>(
                    nttEngine->inverse(buffers->input, static_cast<std::size_t>(iter)));

        carry.carryGPU(
            buffers->input,
            buffers->blockCarryBuf,
            precompute.getN() * sizeof(uint64_t)
        );
        queued += 2;
        
        const uint64_t interval = static_cast<uint64_t>(options.res64_display_interval);
        if (interval != 0 && (((static_cast<uint64_t>(iter) + 1ULL) % interval) == 0ULL)) {

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
            /*auto printLine = [&](cl_mem& bufz, const std::string& name) {
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
            };*/

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
                auto proofFilePath = proofManager.proof(context, *nttEngine, carry,
                                        static_cast<uint32_t>(proofPower),
                                        options.verify);
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
