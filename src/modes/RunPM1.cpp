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
#include "marin/engine.h"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <map>
#include <future>
#include <cinttypes>
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
#include <unordered_set>
#include <limits>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

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
using core::algo::evenGapBound2;
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
        if (mpz_tstbit(p.get_mpz_t(), static_cast<mp_bitcnt_t>(i))) {
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
    //auto lastBackup  = start;

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

int App::runPM1Stage2Marin() {
    using namespace std::chrono;
    if (guiServer_) { std::ostringstream oss; oss << "P-1 factoring stage 2"; guiServer_->setStatus(oss.str()); }
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
    const bool verbose = true;
    static constexpr size_t baseRegsStage2 = 13;
    static constexpr size_t baseRegsStage1 = 11;
    const size_t RSTATE=0, RACC_L=1, RACC_R=2, RPOW=4, RTMP=5, RREF=6;
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
    unsigned long nbEven = evenGapBound2(
        B2,
        (int)options.device_id,
        (size_t)context.getTransformSize(),
        (size_t)(baseRegsStage2 + 2),
        static_cast<double>(options.memlim) / 100.0
    );
    if (nbEven == 0) nbEven = evenGapBound(B2);
    std::cout << "\nnbEven = " << nbEven << std::endl;
    if (nbEven == 0) nbEven = 1;
    mpz_class p0; mpz_nextprime(p0.get_mpz_t(), B1.get_mpz_t());
    if (p0 > B2) {
        std::cout << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\nNo factor P-1 (stage 2) until B2 = " << B2 << '\n'; guiServer_->appendLog(oss.str()); }
        return 1;
    }
    std::vector<uint32_t> idxGap;
    {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now(), last = t0;
        if (B2u <= 3 || B2u <= B1u) { idxGap.clear(); }
        std::vector<uint8_t> sieve((B2u >> 1) + 1, 1);
        uint64_t r = (uint64_t)std::sqrt((long double)B2u);
        uint64_t steps = (r >= 3) ? ((r - 3) / 2 + 1) : 0, done = 0;
        std::cout << "Sieve primes:   0%  ETA  --:--:--" << std::flush;
        for (uint64_t i = 3; i <= r; i += 2) {
            if (sieve[i >> 1]) {
                uint64_t ii = i * i;
                for (uint64_t j = ii; j <= B2u; j += (i << 1)) sieve[j >> 1] = 0;
            }
            ++done;
            auto now = clock::now();
            if (now - last >= std::chrono::milliseconds(400)) {
                double prog = steps ? double(done) / steps : 1.0;
                double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
                long sec = long(eta + 0.5);
                int h = int(sec / 3600), m = int((sec % 3600) / 60), s = int(sec % 60);
                std::cout << "\rSieve primes: " << std::setw(3) << int(prog * 100)
                        << "%  ETA "
                        << std::setw(2) << std::setfill('0') << h << ':'
                        << std::setw(2) << m << ':'
                        << std::setw(2) << s << std::setfill(' ')
                        << std::flush;
                last = now;
            }
            if (interrupted) break;
        }
        std::cout << "\rSieve primes: 100%  ETA  00:00:00\n";
        uint64_t start = (B1u >= 2) ? (B1u + 1) : 3;
        if ((start & 1) == 0) ++start;
        while (start <= B2u && !sieve[start >> 1]) start += 2;
        if (start > B2u) { idxGap.clear(); }
        uint64_t total_pr = 0;
        for (uint64_t q = start; q <= B2u; q += 2) if (sieve[q >> 1]) ++total_pr;
        if (total_pr <= 1) { idxGap.clear(); }
        idxGap.reserve(total_pr ? (size_t)(total_pr - 1) : 0);
        uint64_t processed = 0;
        t0 = clock::now(); last = t0;
        std::cout << "Build gaps:   0%  ETA  --:--:--" << std::flush;
        uint64_t p = start;
        for (uint64_t q = p + 2; q <= B2u; q += 2) {
            if (!sieve[q >> 1]) continue;
            uint64_t gap = q - p;
            uint64_t ig = (gap >> 1) - 1;
            idxGap.push_back((uint32_t)ig);
            p = q;
            ++processed;
            auto now = clock::now();
            if (now - last >= std::chrono::milliseconds(400)) {
                double prog = total_pr ? double(processed) / double(total_pr) : 1.0;
                double eta = prog ? std::chrono::duration<double>(now - t0).count() * (1.0 - prog) / prog : 0.0;
                long sec = long(eta + 0.5);
                int h = int(sec / 3600), m = int((sec % 3600) / 60), s = int(sec % 60);
                std::cout << "\rBuild gaps: " << std::setw(3) << int(prog * 100)
                        << "%  ETA "
                        << std::setw(2) << std::setfill('0') << h << ':'
                        << std::setw(2) << m << ':'
                        << std::setw(2) << s << std::setfill(' ')
                        << std::flush;
                last = now;
            }
            if (interrupted) break;
        }
        std::cout << "\rBuild gaps: 100%  ETA  00:00:00\n";
    }
    std::unordered_map<uint64_t,int> k2slot;
    k2slot.reserve(nbEven*2);
    std::vector<uint64_t> ks_order;
    ks_order.reserve(nbEven);
    {
        uint64_t cum_m = 0;
        for (size_t i = 0; i < idxGap.size() && k2slot.size() < nbEven; ++i) {
            cum_m += (uint64_t)idxGap[i] + 1;
            uint64_t k = cum_m - 1;
            if (!k2slot.count(k)) {
                k2slot.emplace(k, (int)k2slot.size());
                ks_order.push_back(k);
            }
        }
        if (k2slot.empty()) {
            k2slot.emplace(0,0);
            ks_order.push_back(0);
        }
    }
    std::vector<uint64_t> ks_sorted = ks_order;
    std::sort(ks_sorted.begin(), ks_sorted.end());
    for (size_t i = 0; i < ks_sorted.size(); ++i) {
        k2slot[ks_sorted[i]] = (int)i;
    }
    size_t usedSlots = ks_sorted.size();
    size_t regCount = baseRegsStage2 + nbEven + 2;
    engine* eng = engine::create_gpu(pexp, regCount, (size_t)options.device_id, verbose);
    const size_t REVEN = baseRegsStage2;
    const size_t RSAVE_Q = baseRegsStage2 + nbEven;
    const size_t RSAVE_HQ = baseRegsStage2 + nbEven + 1;
    uint64_t resume_idx = 0;
    uint64_t resume_p_u64 = 0;
    double restored_time = 0.0;
    uint64_t s2B1=0, s2B2=0;
    int rs2 = read_ckpt_s2(eng, ckpt_file_s2, resume_p_u64, resume_idx, restored_time, s2B1, s2B2);
    bool resumed_s2 = (rs2 == 0) && (s2B1 == B1u) && (s2B2 == B2u);
    if (!resumed_s2) {
        engine* eng_load = engine::create_gpu(pexp, baseRegsStage1, static_cast<size_t>(options.device_id), verbose);
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
        eng->copy(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RSTATE));
        eng->square_mul(static_cast<engine::Reg>(RPOW));
        eng->copy(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RPOW));
        uint64_t cur_k = 0;
        int pct = -1;
        size_t produced = 0;
        for (size_t t = 0; t < ks_sorted.size(); ++t) {
            uint64_t target_k = ks_sorted[t];
            while (cur_k < target_k) {
                eng->set_multiplicand(static_cast<engine::Reg>(RREF), static_cast<engine::Reg>(RPOW));
                eng->mul(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RREF));
                ++cur_k;
            }
            size_t slot = REVEN + static_cast<size_t>(k2slot[target_k]);
            eng->copy(static_cast<engine::Reg>(slot), static_cast<engine::Reg>(RTMP));
            ++produced;
            int newPct = int((produced * 100ull) / std::max<size_t>(1, usedSlots));
            if (newPct > pct) {
                pct = newPct;
                std::cout << "\rPrecomputing H powers: " << pct << "%" << std::flush;
                if (guiServer_) { std::ostringstream oss; oss << "Precomputing H powers: " << pct << "%"; guiServer_->appendLog(oss.str()); }
            }
            if (interrupted) {
                std::cout << "\nPrecomputation cancelled by user (Ctrl-C). You can reduce GPU memory usage by passing `-memlim <percent>` (e.g., -memlim 30) to precompute fewer even powers.\n";
                if (guiServer_) { std::ostringstream oss; oss << "Precomputation cancelled. Tip: use `-memlim <percent>` (e.g., -memlim 30) to precompute fewer values."; guiServer_->appendLog(oss.str()); }
                interrupted = false;
                delete eng;
                return 0;
            }
        }
        std::cout << "\n";
        for (size_t j = 0; j < usedSlots; ++j) {
            const size_t slot = REVEN + j;
            eng->set_multiplicand(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(slot));
            eng->copy(static_cast<engine::Reg>(slot), static_cast<engine::Reg>(RTMP));
        }
        std::cout << "\n";
        eng->set(static_cast<engine::Reg>(RACC_L), 1);
    }
    size_t totalPrimes = idxGap.size() + 1;
    auto t0 = high_resolution_clock::now();
    auto lastBackup = t0;
    auto lastDisplay = t0;
    mpz_class p = p0;
    uint64_t idx = 0;
    if (resumed_s2) {
        mpz_class rp; mpz_set_ui(rp.get_mpz_t(), static_cast<unsigned long>(resume_p_u64));
        size_t iStart = 0;
        mpz_class cur = p0;
        while (iStart < idxGap.size() && cur < rp) {
            unsigned long add = (unsigned long)(2ull * (static_cast<uint64_t>(idxGap[iStart]) + 1ull));
            mpz_add_ui(cur.get_mpz_t(), cur.get_mpz_t(), add);
            ++iStart;
        }
        if (cur != rp) { delete eng; std::cerr << "Stage 2: resume prime not found in precomputed sequence.\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 2: resume prime not found in precomputed sequence.\n"; guiServer_->appendLog(oss.str()); } return -3; }
        p = cur;
        idx = resume_idx;
        std::cout << "Resuming Stage 2 from checkpoint at prime " << p.get_ui() << " (idx=" << idx << ")\n";
        if (guiServer_) { std::ostringstream oss; oss << "Resuming Stage 2 from checkpoint at prime " << p.get_ui() << " (idx=" << idx << ")"; guiServer_->appendLog(oss.str()); }
        t0 = high_resolution_clock::now() - duration_cast<high_resolution_clock::duration>(duration<double>(restored_time));
        lastBackup = high_resolution_clock::now();
        lastDisplay = lastBackup;
    } else {
        eng->pow(static_cast<engine::Reg>(RACC_R), static_cast<engine::Reg>(RSTATE), mpz_get_ui(p0.get_mpz_t()));
    }
    uint64_t primes_since_check = 0;
    eng->copy(static_cast<engine::Reg>(RSAVE_Q), static_cast<engine::Reg>(RACC_L));
    eng->copy(static_cast<engine::Reg>(RSAVE_HQ), static_cast<engine::Reg>(RACC_R));
    size_t iStartRun = 0;
    uint64_t p_ui = mpz_get_ui(p0.get_mpz_t());
    size_t base_done = 0;
    (void)base_done;
    auto start_sys = std::chrono::system_clock::now();
    iStartRun = resume_idx;
    auto start = t0;
    eng->set_multiplicand2(static_cast<engine::Reg>(RREF), static_cast<engine::Reg>(RACC_R));
    uint64_t cum_m = 0;
    std::unordered_set<uint64_t> k_present;
    k_present.reserve(ks_sorted.size()*2);
    for (auto v: ks_sorted) k_present.insert(v);
    size_t usedSlotsD = ks_sorted.size();
    size_t storedEven = usedSlotsD;
    std::cout << "Stored even powers: " << storedEven << "/" << nbEven << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "Stored even powers: " << storedEven << "/" << nbEven; guiServer_->appendLog(oss.str()); }

    for (size_t i = iStartRun; ; ++i, ++idx) {
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count();
            save_ckpt_s2(eng, static_cast<uint64_t>(p.get_ui()), idx, et);
            delete eng;
            std::cout << "\nInterrupted by user, Stage 2 state saved at prime " << p.get_ui() << " idx=" << idx << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "\nInterrupted by user, Stage 2 state saved at prime " << p.get_ui() << " idx=" << idx; guiServer_->appendLog(oss.str()); }
            return 0;
        }
        eng->sub(static_cast<engine::Reg>(RACC_R), 1);
        eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RACC_R));
        eng->mul(static_cast<engine::Reg>(RACC_L), static_cast<engine::Reg>(RPOW));
        if (i >= idxGap.size()) {break; }
        const uint64_t m_i = (uint64_t)idxGap[i] + 1;
        cum_m += m_i;
        const uint64_t k = cum_m - 1;
        eng->copy(static_cast<engine::Reg>(RACC_R), static_cast<engine::Reg>(RREF));
        auto it = k2slot.find(k);
        if (it != k2slot.end()) {
            eng->mul_new(static_cast<engine::Reg>(RACC_R), static_cast<engine::Reg>(REVEN + (size_t)it->second));
        } else {
            eng->set_multiplicand2(static_cast<engine::Reg>(RREF), static_cast<engine::Reg>(RACC_R));
            cum_m = 0;
        }
        p_ui += 2ull * m_i;
        if ((i + 1) < idxGap.size()) {
            const uint64_t m_next = (uint64_t)idxGap[i + 1] + 1;
            const uint64_t k_next = cum_m + m_next - 1;
            if (!k_present.count(k_next)) {
                eng->set_multiplicand2(static_cast<engine::Reg>(RREF), static_cast<engine::Reg>(RACC_R));
                cum_m = 0;
            }
        }
        ++primes_since_check;
        auto now = high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDisplay).count() >= 3) {
            const size_t done_abs  = idx + 1;
            const double percent   = totalPrimes ? 100.0 * double(done_abs) / double(totalPrimes) : 100.0;
            const double elapsedSec= std::chrono::duration<double>(now - start).count();
            const double ips       = (elapsedSec > 0.0) ? double(done_abs) / elapsedSec : 0.0;
            const double remaining = (totalPrimes > done_abs) ? double(totalPrimes - done_abs) : 0.0;
            const double etaSec    = (ips > 0.0) ? remaining / ips : 0.0;
            int days = int(etaSec) / 86400;
            int hours = (int(etaSec) % 86400) / 3600;
            int minutes = (int(etaSec) % 3600) / 60;
            int seconds = int(etaSec) % 60;
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << percent
                    << "% | Iter: " << done_abs
                    << " | prime: " << p_ui
                    << " | Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s"
                    << " | IPS: " << std::fixed << std::setprecision(2) << ips
                    << " | ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r"
                    << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Progress: " << std::fixed << std::setprecision(2) << percent
                    << "% | Iter: " << done_abs
                    << " | prime: " << p_ui
                    << " | Elapsed: " << std::fixed << std::setprecision(2) << elapsedSec << "s"
                    << " | IPS: " << std::fixed << std::setprecision(2) << ips
                    << " | ETA: " << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r";
                guiServer_->appendLog(oss.str());
                guiServer_->setProgress(double(done_abs), double(totalPrimes), "");
            }
            lastDisplay = now;
        }
        auto now0 = high_resolution_clock::now();
        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now0 - t0).count();
            std::cout << "\nBackup Stage 2 at prime " << p_ui << " idx=" << idx << " start...\n";
            save_ckpt_s2(eng, p_ui, idx, et);
            lastBackup = now0;
            std::cout << "Backup Stage 2 done.\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Backup Stage 2 at prime " << p_ui << " idx=" << idx;
                guiServer_->appendLog(oss.str());
            }
        }
        if (options.iterforce2 > 0 && (idx + 1) % options.iterforce2 == 0) { eng->copy(static_cast<engine::Reg>(RTMP), static_cast<engine::Reg>(RSTATE)); }
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
        if (guiServer_) { std::ostringstream oss; oss << "\n>>>  Factor P-1 (stage 2) found : " << s << "\n"; guiServer_->appendLog(oss.str()); }
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
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage2", json);
    wm.appendToResultsTxt(json);
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

static bool load_pm1_s1_from_save(const std::string& path,
                                  uint64_t& B1_out,
                                  uint32_t& p_out,
                                  mpz_class& X_out)
{
    std::string txt;
    if (!read_text_file(path, txt)) return false;

    std::string hexX;
    if (!parse_ecm_resume_line(txt, B1_out, p_out, hexX)) return false;

    if (hexX.empty()) return false;

    mpz_set_str(X_out.get_mpz_t(), hexX.c_str(), 16);
    return true;
}

static bool load_pm1_s1_from_p95(const std::string& path,
                                 uint64_t& B1_out,
                                 uint32_t& p_out,
                                 mpz_class& X_out)
{
    std::vector<uint8_t> data;

    if (!core::algo::read_prime95_s1_to_bytes(path, p_out, B1_out, data)) {
        return false;
    }
    if (data.empty()) return false;

    mpz_import(X_out.get_mpz_t(),
               data.size(),   
               -1,            
               1,             
               0,             
               0,             
               data.data());

    return true;
}



static mpz_class buildE_full(uint64_t B1)
{
    if (B1 < 2)
        return mpz_class(1);

    mpz_class E(1);

    if (B1 >= 2) {
        uint64_t pw2 = 2;
        while (pw2 <= B1 / 2)
            pw2 <<= 1;
        mpz_mul_ui(E.get_mpz_t(), E.get_mpz_t(), pw2);
    }

    if (B1 < 3)
        return E;
    uint64_t R = (uint64_t)std::sqrt((long double)B1);
    if (R < 3) R = 3;

    std::vector<uint8_t> base((R >> 1) + 1, 1);
    for (uint64_t i = 3; i * i <= R; i += 2) {
        if (base[i >> 1]) {
            for (uint64_t j = i * i; j <= R; j += (i << 1))
                base[j >> 1] = 0;
        }
    }

    std::vector<uint64_t> small_primes;
    for (uint64_t i = 3; i <= R; i += 2)
        if (base[i >> 1])
            small_primes.push_back(i);

    const uint64_t span = 1ULL << 24;
    uint64_t low = 3;
    if ((low & 1ULL) == 0) low += 1;

    while (low <= B1) {
        uint64_t high;
        if (B1 - low + 1 <= span)
            high = B1;
        else
            high = low + span - 1;

        if ((high & 1ULL) == 0) high -= 1;
        if (high < low) break;

        size_t len = (size_t)(((high - low) >> 1) + 1);
        std::vector<uint8_t> seg(len, 1);

        for (uint64_t q : small_primes) {
            uint64_t q2 = q * q;
            if (q2 > high) break;

            uint64_t start = (q2 > low)
                           ? q2
                           : ((low + q - 1) / q) * q;
            if ((start & 1ULL) == 0) start += q;
            if (start < low) start += q;

            for (uint64_t j = start; j <= high; j += (q << 1)) {
                size_t idx = (size_t)((j - low) >> 1);
                seg[idx] = 0;
            }
        }

        for (uint64_t n = low; n <= high; n += 2) {
            size_t idx = (size_t)((n - low) >> 1);
            if (!seg[idx]) continue;

            uint64_t p  = n;
            uint64_t pw = p;
            while (pw <= B1 / p)
                pw *= p;

            mpz_mul_ui(E.get_mpz_t(), E.get_mpz_t(), pw);
        }

        if (high >= B1)
            break;
        low = high + 2;
    }

    return E;
}


static mpz_class buildE_incremental(uint64_t B1_old, uint64_t B1_new)
{
    if (B1_new <= B1_old)
        return mpz_class(1);

    mpz_class E_old = buildE_full(B1_old);
    mpz_class E_new = buildE_full(B1_new);

    mpz_class E_diff;
    mpz_divexact(E_diff.get_mpz_t(), E_new.get_mpz_t(), E_old.get_mpz_t());
    return E_diff;
}


static inline void mpz_mul_u64(mpz_class& a, uint64_t x) {
    if (x <= (uint64_t)std::numeric_limits<unsigned long>::max()) {
        mpz_mul_ui(a.get_mpz_t(), a.get_mpz_t(), (unsigned long)x);
    } else {
        mpz_class t;
        t = x;
        mpz_mul(a.get_mpz_t(), a.get_mpz_t(), t.get_mpz_t());
    }
}

static std::vector<uint32_t> primes_up_to(uint32_t n) {
    std::vector<uint8_t> is(n + 1, 1);
    if (n >= 0) is[0] = 0;
    if (n >= 1) is[1] = 0;
    for (uint32_t i = 2; (uint64_t)i * i <= n; ++i) {
        if (!is[i]) continue;
        for (uint64_t j = (uint64_t)i * i; j <= n; j += i) is[(size_t)j] = 0;
    }
    std::vector<uint32_t> pr;
    pr.reserve(n / 10);
    for (uint32_t i = 2; i <= n; ++i) if (is[i]) pr.push_back(i);
    return pr;
}


static inline double now_seconds_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now() - t0).count();
}

static constexpr uint64_t SIEVE_BLOCK = 1ULL << 20;

static mpz_class buildE_incremental_fast(uint64_t B1_old, uint64_t B1_new)
{
    if (B1_new <= B1_old) return mpz_class(1);
    if (B1_old < 2) B1_old = 1;

    mpz_class E_diff(1);

    const uint64_t high = B1_new;
    const uint32_t root = (uint32_t)std::sqrt((long double)high);
    auto primes = primes_up_to(root);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto t_last = t0;

    auto should_print = [&]() {
        auto t = clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(t - t_last).count() >= 400) {
            t_last = t;
            return true;
        }
        return false;
    };

    auto print_line = [&](const char* phase, double pct, uint64_t cur, uint64_t total) {
        double elapsed = now_seconds_since(t0);
        double eta = (pct > 0.0001) ? elapsed * (1.0 / (pct / 100.0) - 1.0) : 0.0;
        std::fprintf(stderr,
            "\r[%s] %6.2f%%  (%" PRIu64 "/%" PRIu64 ")  elapsed=%.1fs  eta=%.1fs      ",
            phase, pct, cur, total, elapsed, eta);
        std::fflush(stderr);
    };

    const uint64_t total_primes = (uint64_t)primes.size();
    for (uint64_t idx = 0; idx < total_primes; ++idx) {
        uint32_t p = primes[(size_t)idx];
        uint64_t pp = (uint64_t)p;

        uint64_t pw = pp;
        while (pw <= B1_old / pp) pw *= pp;

        while (pw <= B1_new / pp) {
            pw *= pp;
            mpz_mul_u64(E_diff, pp);
        }

        if (should_print()) {
            double pct = total_primes ? (100.0 * (double)(idx + 1) / (double)total_primes) : 100.0;
            print_line("prime-powers", pct, idx + 1, total_primes);
        }
    }
    const uint64_t total_range = (B1_new - B1_old);
    uint64_t done_range = 0;

    uint64_t start = B1_old + 1;
    while (start <= B1_new) {
        uint64_t end = std::min(B1_new, start + SIEVE_BLOCK - 1);
        size_t len = (size_t)(end - start + 1);

        std::vector<uint8_t> seg(len, 1);

        for (uint32_t q : primes) {
            uint64_t qq = (uint64_t)q * (uint64_t)q;
            if (qq > end) break;

            uint64_t first = (start + q - 1) / q;
            uint64_t j = first * (uint64_t)q;
            if (j < qq) j = qq;

            for (; j <= end; j += q) {
                seg[(size_t)(j - start)] = 0;
            }
        }

        for (size_t i = 0; i < len; ++i) {
            uint64_t n = start + (uint64_t)i;
            if (n < 2) continue;
            if (!seg[i]) continue;
            mpz_mul_u64(E_diff, n);
        }

        done_range += (end - start + 1);

        if (should_print()) {
            double pct = total_range ? (100.0 * (double)done_range / (double)total_range) : 100.0;
            print_line("sieve-range", pct, done_range, total_range);
        }

        start = end + 1;
    }

    std::fprintf(stderr, "\r[done]                                   elapsed=%.1fs                    \n",
                 now_seconds_since(t0));

    return E_diff;
}



int App::runPM1Marin() {
   if (guiServer_) {
        std::ostringstream oss;
        oss << "P-1 factoring stage 1";
        guiServer_->setStatus(oss.str());
    }

    uint64_t B1_new = options.B1;
    uint64_t B1_old = options.B1old;
    bool doExtend = (B1_old > 0 && B1_new > B1_old);

    std::cout << "[Backend Marin] Start a P-1 factoring stage 1 up to B1="
              << B1_new << (doExtend ? " (EXTEND mode)" : "") << std::endl;

    if (guiServer_) {
        std::ostringstream oss;
        oss << "[Backend Marin] Start a P-1 factoring stage 1 up to B1="
            << B1_new << (doExtend ? " (EXTEND mode)" : "");
        guiServer_->appendLog(oss.str());
    }

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    mpz_class X_old;
    if (doExtend) {
        std::string basePath = options.pm1_extend_save_path;
        if (basePath.empty()) {
            basePath = "resume_p" + std::to_string(options.exponent) +
                    "_B1_" + std::to_string(B1_old);
        }

        std::string resumeSave = basePath;
        std::string resumeP95  = basePath;

        if (resumeSave.size() >= 5 &&
            resumeSave.substr(resumeSave.size() - 5) == ".save")
        {
            resumeP95 = resumeSave.substr(0, resumeSave.size() - 5) + ".p95";
        }
        else if (resumeSave.size() >= 4 &&
                resumeSave.substr(resumeSave.size() - 4) == ".p95")
        {
            resumeP95  = resumeSave;
            resumeSave = resumeSave.substr(0, resumeSave.size() - 4) + ".save";
        }
        else {
            resumeSave += ".save";
            resumeP95  += ".p95";
        }

        uint64_t B1_file = 0;
        uint32_t p_file  = 0;
        std::string usedPath;

        if (load_pm1_s1_from_save(resumeSave, B1_file, p_file, X_old)) {
            usedPath = resumeSave;
        }
        else if (load_pm1_s1_from_p95(resumeP95, B1_file, p_file, X_old)) {
            usedPath = resumeP95;
        }
        else {
            std::cerr << "Cannot load PM1 S1 state from \"" << resumeSave
                    << "\" nor from \"" << resumeP95 << "\"\n";
            return -1;
        }

        if (B1_file != B1_old || p_file != p) {
            std::cerr << "Mismatch between resume file (B1=" << B1_file
                    << ", p=" << p_file << ") and options (B1old=" << B1_old
                    << ", p=" << p << ")\n";
            return -1;
        }

        std::cout << "Extending PM1 from B1=" << B1_old << " to B1="
                << B1_new << " using state from " << usedPath << "\n";
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Extending PM1 from B1=" << B1_old << " to B1="
                << B1_new << " using state from " << usedPath << "\n";
            guiServer_->appendLog(oss.str());
        }
    }
    //bool debug = false;
    uint64_t B1 = options.B1;
    std::cout << "[Backend Marin] Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "[Backend Marin] Start a P-1 factoring stage 1 up to B1=" << B1 << std::endl; guiServer_->appendLog(oss.str()); }
    const double L_est_bits = 1.4426950408889634 * static_cast<double>(B1);
    const uint64_t MAX_E_BITS = options.max_e_bits;
    std::cout << "MAX_E_BITS = " << MAX_E_BITS << " bits (≈ " << (MAX_E_BITS >> 23) << " MiB)" << std::endl;
    uint64_t estChunks = std::max<uint64_t>(1, (uint64_t)std::ceil(L_est_bits / (double)MAX_E_BITS));
    //const uint32_t p = static_cast<uint32_t>(options.exponent);
    //const bool verbose = true;//options.debug;
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

    if (doExtend) {    
        mpz_t Xtmp;
        mpz_init(Xtmp);
        mpz_set(Xtmp, X_old.get_mpz_t()); 

        eng->set_mpz(static_cast<engine::Reg>(RBASE), Xtmp);

        mpz_clear(Xtmp);

        eng->set(static_cast<engine::Reg>(RSTATE), 1);
        eng->set(static_cast<engine::Reg>(RACC_L), 1);
        eng->set(static_cast<engine::Reg>(RACC_R), 1);
        eng->set(static_cast<engine::Reg>(RSTART), 1);
        eng->copy(static_cast<engine::Reg>(RSAVE_S), static_cast<engine::Reg>(RSTATE));
        eng->copy(static_cast<engine::Reg>(RSAVE_L), static_cast<engine::Reg>(RACC_L));
        eng->copy(static_cast<engine::Reg>(RSAVE_R), static_cast<engine::Reg>(RACC_R));
    }
    else{
        eng->set(RSTATE, 1);
        eng->set(RACC_L, 1);
        eng->set(RACC_R, 1);
        eng->copy(RSTART, RSTATE);
        eng->copy(RSAVE_S, RSTATE);
        eng->copy(RSAVE_L, RACC_L);
        eng->copy(RSAVE_R, RACC_R);
    }
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
    if (doExtend) {
        std::cout << "Building E_diff for (B1old=" << B1_old << ", B1new=" << B1_new << ")...\n" << std::flush;
        mpz_class E_diff = buildE_incremental_fast(B1_old, B1_new);
        std::cout << "E_diff built (" << mpz_sizeinbase(E_diff.get_mpz_t(), 2) << " bits)\n" << std::flush;

        mp_bitcnt_t bits = mpz_sizeinbase(E_diff.get_mpz_t(), 2);

        std::cout << "Extending PM1 exponent: E_diff has " << bits << " bits\n";
        if (guiServer_) {
            std::ostringstream oss;
            oss << "Extending PM1 exponent: E_diff has " << bits << " bits\n";
            guiServer_->appendLog(oss.str());
        }

        if (bits == 0) {
            std::cout << "Nothing to extend (E_diff = 1)\n";
        } else {
            // Réinitialise les timers locaux pour l'extension
            start_clock = std::chrono::high_resolution_clock::now();
            lastDisplay  = start_clock;
            lastBackup   = start_clock;

            // Checkpoint spécifique à la phase d'extension
            std::ostringstream ckext;
            ckext << "pm1_m_" << p << "_ext.ckpt";
            const std::string ckpt_file_ext = ckext.str();

            auto save_ckpt_ext = [&](uint32_t i, double et, uint64_t chk, uint64_t blks, uint64_t bib, uint64_t cbl, uint8_t inlot, const mpz_class& ceacc, const mpz_class& cwbits, uint64_t chunkIdx, uint64_t startP, uint8_t first, uint64_t processedBits, uint64_t bitsInChunk){
                const std::string oldf = ckpt_file_ext + ".old", newf = ckpt_file_ext + ".new";
                { File f(newf, "wb"); int version = 3; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; if (!f.write(reinterpret_cast<const char*>(&chk), sizeof(chk))) return; if (!f.write(reinterpret_cast<const char*>(&blks), sizeof(blks))) return; if (!f.write(reinterpret_cast<const char*>(&bib), sizeof(bib))) return; if (!f.write(reinterpret_cast<const char*>(&cbl), sizeof(cbl))) return; if (!f.write(reinterpret_cast<const char*>(&inlot), sizeof(inlot))) return; char* eacc_hex_c = mpz_get_str(nullptr, 16, ceacc.get_mpz_t()); uint32_t eacc_len = eacc_hex_c ? (uint32_t)std::strlen(eacc_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&eacc_len), sizeof(eacc_len))) { if (eacc_hex_c) std::free(eacc_hex_c); return; } if (eacc_len && !f.write(eacc_hex_c, eacc_len)) { std::free(eacc_hex_c); return; } if (eacc_hex_c) std::free(eacc_hex_c); char* wbits_hex_c = mpz_get_str(nullptr, 16, cwbits.get_mpz_t()); uint32_t wbits_len = wbits_hex_c ? (uint32_t)std::strlen(wbits_hex_c) : 0; if (!f.write(reinterpret_cast<const char*>(&wbits_len), sizeof(wbits_len))) { if (wbits_hex_c) std::free(wbits_hex_c); return; } if (wbits_len && !f.write(wbits_hex_c, wbits_len)) { std::free(wbits_hex_c); return; } if (wbits_hex_c) std::free(wbits_hex_c); if (!f.write(reinterpret_cast<const char*>(&chunkIdx), sizeof(chunkIdx))) return; if (!f.write(reinterpret_cast<const char*>(&startP), sizeof(startP))) return; if (!f.write(reinterpret_cast<const char*>(&first), sizeof(first))) return; if (!f.write(reinterpret_cast<const char*>(&processedBits), sizeof(processedBits))) return; if (!f.write(reinterpret_cast<const char*>(&bitsInChunk), sizeof(bitsInChunk))) return; f.write_crc32(); }
                std::error_code ec; fs::remove(oldf, ec); fs::rename(ckpt_file_ext, oldf, ec); fs::rename(ckpt_file_ext + ".new", ckpt_file_ext, ec); fs::remove(oldf, ec);
            };

            auto read_ckpt_ext = [&](const std::string& file, uint32_t& ri, double& et, uint64_t& chk, uint64_t& blks, uint64_t& bib, uint64_t& cbl, uint8_t& inlot, mpz_class& ceacc, mpz_class& cwbits, uint64_t& chunkIdx, uint64_t& startP, uint8_t& first, uint64_t& processedBits, uint64_t& bitsInChunk)->int{
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

            uint32_t resumeI_ext_ck = 0;
            double restored_time_ext = 0.0;
            uint64_t gl_checkpass_ext_ck = 0, gl_blocks_since_check_ext_ck = 0, gl_bits_in_block_ext_ck = 0, gl_current_block_len_ext_ck = 0, bits_in_chunk_ext_ck = 0;
            uint8_t in_lot_ext_ck = 0, firstChunk_ext_ck = 0;
            mpz_class eacc_ext_ck = 0, wbits_ext_ck = 0;
            uint64_t chunkIdx_ext = 0;
            uint64_t startP_ext = 0;
            uint64_t processedBits_ext_ck = 0;
            bool restored_ext = false;

            int rr_ext = read_ckpt_ext(ckpt_file_ext, resumeI_ext_ck, restored_time_ext, gl_checkpass_ext_ck, gl_blocks_since_check_ext_ck, gl_bits_in_block_ext_ck, gl_current_block_len_ext_ck, in_lot_ext_ck, eacc_ext_ck, wbits_ext_ck, chunkIdx_ext, startP_ext, firstChunk_ext_ck, processedBits_ext_ck, bits_in_chunk_ext_ck);
            if (rr_ext < 0) rr_ext = read_ckpt_ext(ckpt_file_ext + ".old", resumeI_ext_ck, restored_time_ext, gl_checkpass_ext_ck, gl_blocks_since_check_ext_ck, gl_bits_in_block_ext_ck, gl_current_block_len_ext_ck, in_lot_ext_ck, eacc_ext_ck, wbits_ext_ck, chunkIdx_ext, startP_ext, firstChunk_ext_ck, processedBits_ext_ck, bits_in_chunk_ext_ck);
            if (rr_ext == 0 && bits_in_chunk_ext_ck == (uint64_t)bits) {
                restored_ext = true;
            }
            restored_time = restored_ext ? restored_time_ext : 0.0;

            uint64_t B = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)bits));
            double desiredIntervalSeconds = 600.0;
            uint64_t checkpasslevel_auto =
                (uint64_t)((1000.0 * desiredIntervalSeconds) / (double)B);
            if (checkpasslevel_auto == 0) {
                uint64_t tmpB = (uint64_t)std::sqrt((double)B);
                if (tmpB == 0) tmpB = 1;
                checkpasslevel_auto = ((uint64_t)bits / B) / tmpB;
            }
            uint64_t checkpasslevel = (options.checklevel > 0)
                ? options.checklevel
                : checkpasslevel_auto;
            if (checkpasslevel == 0) checkpasslevel = 1;

            uint64_t blocks_since_check = restored_ext ? gl_blocks_since_check_ext_ck : 0;
            uint64_t bits_in_block      = restored_ext ? gl_bits_in_block_ext_ck     : 0;
            uint64_t current_block_len  = restored_ext && gl_current_block_len_ext_ck
                                          ? gl_current_block_len_ext_ck
                                          : (((uint64_t)(( (restored_ext ? resumeI_ext_ck : (uint32_t)bits ) - 1) % B)) + 1);
            mpz_class eacc              = restored_ext ? eacc_ext_ck : 0;
            mpz_class wbits             = restored_ext ? wbits_ext_ck : 0;
            uint64_t gl_checkpass       = restored_ext ? gl_checkpass_ext_ck : 0;
            bool in_lot                 = restored_ext ? (in_lot_ext_ck != 0) : false;
            bool errordone              = false;

            mp_bitcnt_t resumeI = restored_ext ? (mp_bitcnt_t)resumeI_ext_ck : bits;
            uint64_t processed_bits_ext_base = restored_ext ? processedBits_ext_ck : 0;

            // Affichage de départ (même style que la branche normale)
            {
                std::string res64_x_ext;
                uint64_t perChunkDone0 = restored_ext ? (bits - resumeI) : 0;
                uint64_t globalDone0   = processed_bits_ext_base + perChunkDone0;
                spinner.displayProgress2(
                    globalDone0,
                    bits,
                    timer.elapsed() + restored_time,
                    timer2.elapsed(),
                    options.exponent,
                    globalDone0,
                    processed_bits_ext_base,
                    res64_x_ext,
                    guiServer_ ? guiServer_.get() : nullptr,
                    1,
                    1,
                    perChunkDone0,
                    bits,
                    true
                );
                timer2.start();
            }

            uint64_t lastIter_ext = (uint64_t)resumeI;

            eng->copy(RTMP, RBASE);
            eng->set_multiplicand(RBASE, RTMP);
            for (mp_bitcnt_t i = resumeI; i > 0; --i) {
                lastIter_ext = (uint64_t)i;

                if (interrupted) {
                    std::cout << "\nInterrupted by user, state saved at iteration " << i << std::endl;
                    if (guiServer_) { std::ostringstream oss; oss << "\nInterrupted signal received\n "; guiServer_->appendLog(oss.str()); }

                    auto now_int = std::chrono::high_resolution_clock::now();
                    double elapsed_ext = std::chrono::duration<double>(now_int - start_clock).count() + restored_time;
                    save_ckpt_ext(
                        (uint32_t)lastIter_ext,
                        elapsed_ext,
                        gl_checkpass,
                        blocks_since_check,
                        bits_in_block,
                        current_block_len,
                        in_lot ? 1 : 0,
                        eacc,
                        wbits,
                        1,          // chunkIdx
                        0,          // startP
                        1,          // first
                        processed_bits_ext_base + (bits - i),
                        (uint64_t)bits
                    );
                    delete eng;
                    return 0;
                }

                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastBackup).count() >= options.backup_interval) {
                    std::cout << "\nBackup point done at i=" << i << " start...." << std::endl;
                    double elapsed_ext = std::chrono::duration<double>(now - start_clock).count() + restored_time;
                    save_ckpt_ext(
                        (uint32_t)lastIter_ext,
                        elapsed_ext,
                        gl_checkpass,
                        blocks_since_check,
                        bits_in_block,
                        current_block_len,
                        in_lot ? 1 : 0,
                        eacc,
                        wbits,
                        1,          // chunkIdx
                        0,          // startP
                        1,          // first
                        processed_bits_ext_base + (bits - i),
                        (uint64_t)bits
                    );
                    std::cout << "\nBackup point done at i=" << i << " done...." << std::endl;
                    lastBackup = now;
                }

                if (bits_in_block == 0) {
                    current_block_len = ((uint64_t)((i - 1) % B)) + 1;
                    if (current_block_len == B) {
                        if (gl_checkpass == 0 &&
                            blocks_since_check == 0 &&
                            wbits == 0 &&
                            eacc == 0)
                        {
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

                eng->square_mul(RSTATE);
                int b = mpz_tstbit(E_diff.get_mpz_t(), i - 1) ? 1 : 0;
                if (b) {
                    //eng->set_multiplicand(RTMP, RBASE);
                    eng->mul(RSTATE, RBASE);
                }

                // Injection d'erreur comme dans la branche normale
                if (options.erroriter > 0 &&
                    (bits - i + 1) == options.erroriter &&
                    !errordone)
                {
                    errordone = true;
                    eng->sub(RSTATE, 33);
                    std::cout << "Injected error at iteration " << (bits - i + 1) << std::endl;
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "Injected error at iteration " << (bits - i + 1);
                        guiServer_->appendLog(oss.str());
                    }
                }

                wbits <<= 1;
                if (b) wbits += 1;
                bits_in_block += 1;
                
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

                        bool doCheck = options.gerbiczli && in_lot &&
                                       (gl_checkpass == checkpasslevel || i == 1);
                        if (doCheck) {
                            std::cout << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                                guiServer_->appendLog(oss.str());
                            }

                            eng->copy(RCHK, RACC_L);
                            for (uint64_t k = 0; k < B; ++k)
                                eng->square_mul(RCHK);

                            eng->set(RPOW, 1);
                            size_t eb = mpz_sizeinbase(eacc.get_mpz_t(), 2);
                            for (size_t k = eb; k-- > 0;) {
                                eng->square_mul(RPOW);
                                if (mpz_tstbit(eacc.get_mpz_t(), k)) {
                                    //eng->set_multiplicand(RTMP, RBASE);
                                    eng->mul(RPOW, RBASE);
                                }
                            }
                            eng->set_multiplicand(RTMP, RPOW);
                            eng->mul(RCHK, RTMP);

                            mpz_t z0, z1;
                            mpz_inits(z0, z1, nullptr);
                            eng->get_mpz(z0, RCHK);
                            eng->get_mpz(z1, RACC_R);
                            bool ok = (mpz_cmp(z0, z1) == 0);
                            mpz_clears(z0, z1, nullptr);

                            if (!ok) {
                                std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                options.gerbicz_error_count += 1;
                                eng->copy(RSTATE, RSAVE_S);
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eng->copy(RSTART, RSTATE);
                                i = (mp_bitcnt_t)(i + blocks_since_check * B);
                                eacc = 0;
                                blocks_since_check = 0;
                                wbits = 0;
                                gl_checkpass = 0;
                                bits_in_block = 0;
                                continue;
                            } else {
                                std::cout << "[Gerbicz Li] Check passed\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Check passed\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                eng->copy(RSAVE_S, RSTATE);
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eacc = 0;
                                blocks_since_check = 0;
                                gl_checkpass = 0;
                            }
                        }
                    } else {
                        if (options.gerbiczli) {
                            std::cout << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                            if (guiServer_) {
                                std::ostringstream oss;
                                oss << "[Gerbicz Li] Start a Gerbicz Li check....\n";
                                guiServer_->appendLog(oss.str());
                            }
                            eng->copy(RCHK, RSTART);
                            for (uint64_t k = 0; k < current_block_len; ++k)
                                eng->square_mul(RCHK);

                            eng->set(RPOW, 1);
                            size_t wb = mpz_sizeinbase(wbits.get_mpz_t(), 2);
                            for (size_t k = wb; k-- > 0;) {
                                eng->square_mul(RPOW);
                                if (mpz_tstbit(wbits.get_mpz_t(), k)) {
                                    //eng->set_multiplicand(RTMP, RBASE);
                                    eng->mul(RPOW, RBASE);
                                }
                            }
                            eng->set_multiplicand(RTMP, RPOW);
                            eng->mul(RCHK, RTMP);

                            mpz_t z0, z1;
                            mpz_inits(z0, z1, nullptr);
                            eng->get_mpz(z0, RCHK);
                            eng->get_mpz(z1, RSTATE);
                            bool ok0 = (mpz_cmp(z0, z1) == 0);
                            mpz_clears(z0, z1, nullptr);

                            if (!ok0) {
                                std::cout << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Mismatch : Last correct state will be restored\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                options.gerbicz_error_count += 1;
                                eng->copy(RSTATE, RSTART);
                                i = (mp_bitcnt_t)(i + current_block_len);
                                wbits = 0;
                                bits_in_block = 0;
                                continue;
                            } else {
                                std::cout << "[Gerbicz Li] Check passed\n";
                                if (guiServer_) {
                                    std::ostringstream oss;
                                    oss << "[Gerbicz Li] Check passed\n";
                                    guiServer_->appendLog(oss.str());
                                }
                                eng->copy(RSAVE_S, RSTATE);
                                eng->set(RACC_L, 1);
                                eng->set(RACC_R, 1);
                                eacc = 0;
                                blocks_since_check = 0;
                                gl_checkpass = 0;
                            }
                        }
                    }
                    bits_in_block = 0;
                    wbits = 0;
                }

                auto now2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now2 - lastDisplay).count() >= 10) {
                    std::string res64_x_ext;
                    uint64_t bitsDoneNow = processed_bits_ext_base + (bits - i + 1);
                    spinner.displayProgress2(
                        bitsDoneNow,
                        bits,
                        timer.elapsed() + restored_time,
                        timer2.elapsed(),
                        options.exponent,
                        bitsDoneNow,
                        processed_bits_ext_base,
                        res64_x_ext,
                        guiServer_ ? guiServer_.get() : nullptr,
                        1,
                        1,
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
                    eng->square_mul(RPOW);
                    if (mpz_tstbit(wtail.get_mpz_t(), k)) {
                        //eng->set_multiplicand(RTMP, RBASE);
                        eng->mul(RPOW, RBASE);
                    }
                }
                eng->set_multiplicand(RTMP, RPOW);
                eng->mul(RCHK, RTMP);

                mpz_t z0, z1;
                mpz_inits(z0, z1, nullptr);
                eng->get_mpz(z0, RCHK);
                eng->get_mpz(z1, RSTATE);
                bool ok_tail = (mpz_cmp(z0, z1) == 0);
                mpz_clears(z0, z1, nullptr);

                if (!ok_tail) {
                    eng->copy(RSTATE, RSTART);
                    eng->copy(RCHK, RSTART);
                    for (uint64_t k = 0; k < bt; ++k) eng->square_mul(RCHK);
                    eng->set(RPOW, 1);
                    size_t wbl2 = mpz_sizeinbase(wtail.get_mpz_t(), 2);
                    for (size_t k = wbl2; k-- > 0;) {
                        eng->square_mul(RPOW);
                        if (mpz_tstbit(wtail.get_mpz_t(), k)) {
                            eng->set_multiplicand(RTMP, RBASE);
                            eng->mul(RPOW, RTMP);
                        }
                    }
                    eng->set_multiplicand(RTMP, RPOW);
                    eng->mul(RSTATE, RTMP);
                }
            }

            // Progress final pour l'extension
            {
                std::string res64_done_ext;
                uint64_t bitsDoneAll = processed_bits_ext_base + (uint64_t)bits;
                spinner.displayProgress2(
                    bitsDoneAll,
                    bits,
                    timer.elapsed() + restored_time,
                    timer2.elapsed(),
                    options.exponent,
                    bitsDoneAll,
                    processed_bits_ext_base,
                    res64_done_ext,
                    guiServer_ ? guiServer_.get() : nullptr,
                    1,
                    1,
                    1,
                    1,
                    true
                );
            }

            std::cout << "\nExtension exponentiation done.\n";
        }

        // Now RSTATE = X_old ^ E_diff = 3^{E(B1_new)}.

        mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
        mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RSTATE), Mp);

        // Write new resume & p95 if you like
        if (options.resume) {
            auto now_sys = std::chrono::system_clock::now();
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
            std::string ds = fmt(now_sys);
            std::string de = ds;

            writeEcmResumeLine("resume_p" + std::to_string(options.exponent) +
                               "_B1_" + std::to_string(B1_new) + ".save",
                               B1_new, options.exponent, X);

            convertEcmResumeToPrime95("resume_p" + std::to_string(options.exponent) +
                                      "_B1_" + std::to_string(B1_new) + ".save",
                                      "resume_p" + std::to_string(options.exponent) +
                                      "_B1_" + std::to_string(B1_new) + ".p95",
                                      ds, de);
        }

        // Do gcd and results file exactly as in the normal Stage1 end:
        X -= 1;
        mpz_class g = gcd_with_dots(X, Mp);
        bool factorFound = (g != 1) && (g != Mp);

        std::string filename = "stage1_result_B1_" + std::to_string(B1_new) +
                               "_p_" + std::to_string(options.exponent) + ".txt";

        if (factorFound) {
            char* fstr = mpz_get_str(nullptr, 10, g.get_mpz_t());
            writeStageResult(filename, "B1=" + std::to_string(B1_new) +
                                        "  factor=" + std::string(fstr));
            std::cout << "\nP-1 factor stage 1 found: " << fstr << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\nP-1 factor stage 1 found: " << fstr << std::endl;
                guiServer_->appendLog(oss.str());
            }
            options.knownFactors.push_back(std::string(fstr));
            std::free(fstr);
            std::cout << "\n";
        } else {
            writeStageResult(filename, "No factor up to B1=" + std::to_string(B1_new));
            std::cout << "\nNo P-1 (stage 1) factor up to B1=" + std::to_string(B1_new) << "\n" << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "\nNo P-1 (stage 1) factor up to B1=" + std::to_string(B1_new) << "\n";
                guiServer_->appendLog(oss.str());
            }
        }

        // JSON + worktodo removal same as usual ...
        std::string json = io::JsonBuilder::generate(
            options,
            static_cast<int>(context.getTransformSize()),
            false,
            "",
            ""
        );
        std::cout << "Manual submission JSON:\n" << json << "\n";
        io::WorktodoManager wm(options);
        wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage1_ext", json);
        wm.appendToResultsTxt(json);
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
        return factorFound ? 0 : 1;
    }
    else{
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
            //auto chunkStart = std::chrono::high_resolution_clock::now();
            //bool tunedCheckpass = false;
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
                            //mpz_class Mp = (mpz_class)(((mpz_class)1 << options.exponent) - 1);
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
                            //mpz_class Mp = (mpz_class) (((mpz_class)1 << options.exponent) - 1);
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
                //mpz_class Mp = (mpz_class)(((mpz_class)1 << options.exponent) - 1);
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
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
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

/*
2025 - Cherubrock
P-1 Stage 3 — Paired Multiplicative Offsets (k = 0, ±1, ±2, …)
*/

int App::runPM1Stage3Marin() {
    using namespace std::chrono;

    if (guiServer_) { std::ostringstream oss; oss << "P-1 factoring stage 3"; guiServer_->setStatus(oss.str()); }
    const uint64_t B3u = options.B3;
    if (B3u == 0) { std::cout << "Stage 3 skipped (B3=0)\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 3 skipped (B3=0)"; guiServer_->appendLog(oss.str()); } return 1; }

    const uint32_t pexp = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;

    mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
    bool foundIntermediate = false;
    mpz_class midFactor;

    static constexpr size_t baseRegsStage1 = 11;
    // +2 registres pour H^k et H^{-k}
    const size_t RSTATE=0, RACC=1, RSQ=2, RMINUS=3, RPLUS=4, RPOW=5, RHINV=6, RHK=7, RHIK=8, RHINV_2=9, RHINV_3=10, RHK_2=11, RHK_3=12, RSTART_SQ=13, RCHK_SQ=14;
    static constexpr size_t regsS3 = 15;

    std::ostringstream ck3; ck3 << "pm1_s3_m_" << pexp << ".ckpt";
    const std::string ckpt_file_s3 = ck3.str();

    auto read_ckpt_s3 = [&](engine* e, const std::string& file, uint64_t& saved_b, double& et, uint64_t& sB3)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        if (version != 2) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != pexp) return -2;
        if (!f.read(reinterpret_cast<char*>(&sB3), sizeof(sB3))) return -2;
        if (!f.read(reinterpret_cast<char*>(&saved_b), sizeof(saved_b))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        const size_t cksz = e->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!e->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };
    auto save_ckpt_s3 = [&](engine* e, uint64_t cur_b, double et){
        const std::string oldf = ckpt_file_s3 + ".old", newf = ckpt_file_s3 + ".new";
        {
            File f(newf, "wb");
            int version = 2;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&pexp), sizeof(pexp))) return;
            if (!f.write(reinterpret_cast<const char*>(&B3u), sizeof(B3u))) return;
            if (!f.write(reinterpret_cast<const char*>(&cur_b), sizeof(cur_b))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            const size_t cksz = e->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!e->get_checkpoint(data)) return;
            if (!f.write(data.data(), cksz)) return;
            f.write_crc32();
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file_s3.c_str(), &s) == 0) && (std::rename(ckpt_file_s3.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file_s3.c_str());
    };

    engine* eng = engine::create_gpu(pexp, regsS3, static_cast<size_t>(options.device_id), verbose);

    // Charger H (RSTATE) et H^{-1} (RHINV) depuis checkpoint S1
    {
        mpz_t Hs1; mpz_init(Hs1);

        if (options.s3only) {
            uint64_t B1_file = 0;
            uint32_t p_file  = 0;
            mpz_class X_s1;

            std::string basePath = options.pm1_extend_save_path;
            if (basePath.empty()) {
                basePath = "resume_p" + std::to_string(options.exponent) +
                           "_B1_" + std::to_string(options.B1);
            }

            std::string resumeSave = basePath;
            std::string resumeP95  = basePath;

            if (resumeSave.size() >= 5 &&
                resumeSave.substr(resumeSave.size() - 5) == ".save")
            {
                resumeP95 = resumeSave.substr(resumeSave.size() - 5) + ".p95";
            }
            else if (resumeSave.size() >= 4 &&
                     resumeSave.substr(resumeSave.size() - 4) == ".p95")
            {
                resumeP95  = resumeSave;
                resumeSave = resumeSave.substr(resumeSave.size() - 4) + ".save";
            }
            else {
                resumeSave += ".save";
                resumeP95  += ".p95";
            }

            std::string usedPath;

            if (load_pm1_s1_from_save(resumeSave, B1_file, p_file, X_s1)) {
                usedPath = resumeSave;
            }
            else if (load_pm1_s1_from_p95(resumeP95, B1_file, p_file, X_s1)) {
                usedPath = resumeP95;
            }
            else {
                std::cerr << "Cannot load PM1 S1 state from \"" << resumeSave
                          << "\" nor from \"" << resumeP95 << "\"\n";
                mpz_clear(Hs1);
                delete eng;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Stage 3: cannot load PM1 S1 state from " << resumeSave
                        << " nor " << resumeP95;
                    guiServer_->appendLog(oss.str());
                }
                return -2;
            }

            if (p_file != pexp) {
                std::cerr << "Mismatch between S1 resume file (p=" << p_file
                          << ") and options (p=" << pexp << ")\n";
                mpz_clear(Hs1);
                delete eng;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Stage 3: mismatch between S1 resume file (p=" << p_file
                        << ") and options (p=" << pexp << ")";
                    guiServer_->appendLog(oss.str());
                }
                return -2;
            }

            std::cout << "Stage 3: using PM1 S1 state from " << usedPath << "\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Stage 3: using PM1 S1 state from " << usedPath;
                guiServer_->appendLog(oss.str());
            }

            mpz_set(Hs1, X_s1.get_mpz_t());
        } else {
            engine* eng_load = engine::create_gpu(pexp, baseRegsStage1, static_cast<size_t>(options.device_id), verbose);
            std::ostringstream ck; ck << "pm1_m_" << pexp << ".ckpt";
            const std::string ckpt_file = ck.str();

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

            int rr = read_ckpt_s1(eng_load, ckpt_file);
            if (rr < 0) rr = read_ckpt_s1(eng_load, ckpt_file + ".old");
            if (rr != 0) { delete eng_load; delete eng; std::cerr << "Stage 3: cannot load pm1 stage1 checkpoint.\n"; if (guiServer_) { std::ostringstream oss; oss << "Stage 3: cannot load pm1 stage1 checkpoint.\n"; guiServer_->appendLog(oss.str()); } mpz_clear(Hs1); return -2; }

            eng_load->get_mpz(Hs1, static_cast<engine::Reg>(RSTATE));
            delete eng_load;
        }

        mpz_t Mp_local, Hinv;
        mpz_init(Mp_local); mpz_init(Hinv);
        mpz_ui_pow_ui(Mp_local, 2, pexp); mpz_sub_ui(Mp_local, Mp_local, 1);
        if (mpz_invert(Hinv, Hs1, Mp_local) == 0) {
            mpz_clear(Mp_local); mpz_clear(Hinv); mpz_clear(Hs1);
            delete eng;
            std::cerr << "Stage 3: H not invertible mod Mp.\n";
            if (guiServer_) { std::ostringstream oss; oss << "Stage 3: H not invertible mod Mp."; guiServer_->appendLog(oss.str()); }
            return -3;
        }
        eng->set_mpz(static_cast<engine::Reg>(RSTATE), Hs1);     // H
        eng->set_mpz(static_cast<engine::Reg>(RHINV),  Hinv);    // H^{-1}
        mpz_clear(Mp_local); mpz_clear(Hinv); mpz_clear(Hs1);
    }

    eng->set(static_cast<engine::Reg>(RACC), 1);
    eng->copy(static_cast<engine::Reg>(RSQ),  static_cast<engine::Reg>(RSTATE));

    bool resumed_s3 = false;
    uint64_t resume_b = 0, s3B3 = 0;
    double restored_time = 0.0;
    {
        int rs3 = read_ckpt_s3(eng, ckpt_file_s3, resume_b, restored_time, s3B3);
        if (rs3 == 0 && s3B3 == B3u) {
            resumed_s3 = true;
            if (guiServer_) { std::ostringstream oss; oss << "Resuming Stage 3 at b=" << resume_b << "/" << B3u; guiServer_->appendLog(oss.str()); }
            std::cout << "Resuming Stage 3 at b=" << resume_b << "/" << B3u << "\n";
        }
    }

    auto t0 = high_resolution_clock::now();
    auto lastBackup  = t0;
    auto lastDisplay = t0;

    const uint64_t Kmax = std::min<uint64_t>(3, std::max<uint64_t>(1, options.B4));
    uint64_t b = resumed_s3 ? resume_b : 0;
    if (!resumed_s3) {
        std::cout << "Start P-1 Stage 3 up to B3=" << B3u << " [±k, k≤" << Kmax << "]\n";
        if (guiServer_) { std::ostringstream oss; oss << "Start P-1 Stage 3 up to B3=" << B3u << " [±k, k≤" << Kmax << "]"; guiServer_->appendLog(oss.str()); }
    }

    // --- Inverses : RHINV = H^{-1}, RHINV_2 = H^{-2}, RHINV_3 = H^{-3}
    eng->copy((engine::Reg)RHIK, (engine::Reg)RHINV);              // RHIK = H^{-1}
    eng->set_multiplicand((engine::Reg)RHINV, (engine::Reg)RHIK);  // mul(..., RHINV) = *H^{-1}
    eng->square_mul((engine::Reg)RHIK);                            // RHIK = (H^{-1})^2 = H^{-2}
    eng->set_multiplicand((engine::Reg)RHINV_2, (engine::Reg)RHIK);// RHINV_2 = H^{-2}
    eng->mul((engine::Reg)RHIK, (engine::Reg)RHINV);               // RHIK *= H^{-1} → H^{-3}
    eng->set_multiplicand((engine::Reg)RHINV_3, (engine::Reg)RHIK);// RHINV_3 = H^{-3}

    // --- Puissances positives : RHK = H, RHK_2 = H^2, RHK_3 = H^3
    eng->copy((engine::Reg)RPOW, (engine::Reg)RSTATE);             // RPOW = H
    eng->set_multiplicand((engine::Reg)RHK, (engine::Reg)RPOW);    // RHK = H

    eng->square_mul((engine::Reg)RPOW);                            // RPOW = H^2
    eng->set_multiplicand((engine::Reg)RHK_2, (engine::Reg)RPOW);  // RHK_2 = H^2

    eng->mul((engine::Reg)RPOW, (engine::Reg)RHK);                 // RPOW *= H → H^3
    eng->set_multiplicand((engine::Reg)RHK_3, (engine::Reg)RPOW);  // RHK_3 = H^3

    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));
    eng->square_mul(static_cast<engine::Reg>(RSQ));

    uint64_t total_iters = B3u;
    uint64_t B = std::max<uint64_t>(1, (uint64_t)std::sqrt((double)std::max<uint64_t>(1, total_iters)));
    double desiredIntervalSeconds = 600.0;
    uint64_t checkpasslevel_auto = (uint64_t)((1000.0 * desiredIntervalSeconds) / (double)B);
    if (checkpasslevel_auto == 0) {
        uint64_t tmpB = (uint64_t)std::sqrt((double)B);
        if (tmpB == 0) tmpB = 1;
        checkpasslevel_auto = (total_iters / B) / tmpB;
    }
    uint64_t checkpasslevel = (options.checklevel > 0)
        ? options.checklevel
        : checkpasslevel_auto;
    if (checkpasslevel == 0) checkpasslevel = 1;

    uint64_t steps_in_block = 0;
    uint64_t blocks_since_last_check = 0;
    bool errordone = false;
    uint64_t originalB3 = options.B3;

    for (; b < B3u; ++b) {
        if (interrupted) {
            double et = duration<double>(high_resolution_clock::now() - t0).count() + restored_time;
            save_ckpt_s3(eng, b, et);
            delete eng;
            std::cout << "\nInterrupted by user, Stage 3 state saved at b=" << b << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Interrupted, Stage 3 saved at b=" << b; guiServer_->appendLog(oss.str()); }
            return 0;
        }

        if (steps_in_block == 0) {
            eng->copy((engine::Reg)RSTART_SQ, (engine::Reg)RSQ);
        }

        // RSQ := RSQ^2  (RSQ = H^{2^b} → H^{2^{b+1}})
        eng->square_mul(static_cast<engine::Reg>(RSQ));

        if (options.erroriter > 0 &&
            (b + 1) == options.erroriter &&
            !errordone)
        {
            errordone = true;
            eng->sub(static_cast<engine::Reg>(RSQ), 33);
            std::cout << "Injected error at Stage 3 iteration " << (b + 1) << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss << "Injected error at Stage 3 iteration " << (b + 1);
                guiServer_->appendLog(oss.str());
            }
        }

        // Init H^k et H^{-k}
        //eng->copy(static_cast<engine::Reg>(RHK),  static_cast<engine::Reg>(RSTATE)); // H^1
        //eng->copy(static_cast<engine::Reg>(RHIK), static_cast<engine::Reg>(RHINV));  // H^{-1}
        
        
        //for (uint64_t k = 1; k <= Kmax; ++k) {
            // (RSQ * H^{-k} - 1)

        for (uint64_t k = 1; k <= Kmax; ++k) {
            engine::Reg regHplus, regHminus;
            if (k == 1) {
                regHplus  = (engine::Reg)RHK;
                regHminus = (engine::Reg)RHINV;
            } else if (k == 2) {
                regHplus  = (engine::Reg)RHK_2;
                regHminus = (engine::Reg)RHINV_2;
            } else { // k == 3
                regHplus  = (engine::Reg)RHK_3;
                regHminus = (engine::Reg)RHINV_3;
            }

            // (RSQ * H^{-k} - 1)
            eng->copy((engine::Reg)RMINUS, (engine::Reg)RSQ);
            eng->mul ((engine::Reg)RMINUS, regHminus);
            eng->sub ((engine::Reg)RMINUS, 1);
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)RMINUS);
            eng->mul ((engine::Reg)RACC, (engine::Reg)RPOW);

            // (RSQ * H^{+k} - 1)
            eng->copy((engine::Reg)RPLUS, (engine::Reg)RSQ);
            eng->mul ((engine::Reg)RPLUS, regHplus);
            eng->sub ((engine::Reg)RPLUS, 1);
            eng->set_multiplicand((engine::Reg)RPOW, (engine::Reg)RPLUS);
            eng->mul ((engine::Reg)RACC, (engine::Reg)RPOW);
        }

        

/*
            // (RSQ * H^{+k} - 1)
            eng->copy(static_cast<engine::Reg>(RPLUS), static_cast<engine::Reg>(RSQ));
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RHK));
            eng->mul (static_cast<engine::Reg>(RPLUS), static_cast<engine::Reg>(RPOW));
            eng->sub (static_cast<engine::Reg>(RPLUS), 1);
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RPLUS));
            eng->mul (static_cast<engine::Reg>(RACC),  static_cast<engine::Reg>(RPOW));

            // k <- k+1 : H^k *= H ; H^{-k} *= H^{-1}
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RSTATE));
            eng->mul (static_cast<engine::Reg>(RHK),  static_cast<engine::Reg>(RPOW));
            eng->set_multiplicand(static_cast<engine::Reg>(RPOW), static_cast<engine::Reg>(RHINV));
            eng->mul (static_cast<engine::Reg>(RHIK), static_cast<engine::Reg>(RPOW));*/
        //}

        steps_in_block++;

        bool end_block = (steps_in_block == B) || (b + 1 == B3u);
        if (end_block) {
            blocks_since_last_check++;

            bool doCheck = options.gerbiczli &&
                           (blocks_since_last_check >= checkpasslevel || (b + 1 == B3u));
            if (doCheck) {
                std::cout << "[Gerbicz Li] Stage 3 check start\n";
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "[Gerbicz Li] Stage 3 check start";
                    guiServer_->appendLog(oss.str());
                }

                eng->copy((engine::Reg)RCHK_SQ, (engine::Reg)RSTART_SQ);
                for (uint64_t k = 0; k < steps_in_block; ++k) {
                    eng->square_mul((engine::Reg)RCHK_SQ);
                }

                mpz_t z0, z1;
                mpz_inits(z0, z1, nullptr);
                eng->get_mpz(z0, (engine::Reg)RCHK_SQ);
                eng->get_mpz(z1, (engine::Reg)RSQ);
                bool ok = (mpz_cmp(z0, z1) == 0);
                mpz_clears(z0, z1, nullptr);

                if (!ok) {
                    std::cout << "[Gerbicz Li] Stage 3 mismatch : last correct RSQ will be restored\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "[Gerbicz Li] Stage 3 mismatch : last correct RSQ will be restored";
                        guiServer_->appendLog(oss.str());
                    }
                    options.gerbicz_error_count += 1;
                    eng->copy((engine::Reg)RSQ, (engine::Reg)RSTART_SQ);
                    eng->set(static_cast<engine::Reg>(RACC), 1);
                } else {
                    std::cout << "[Gerbicz Li] Stage 3 check passed\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "[Gerbicz Li] Stage 3 check passed";
                        guiServer_->appendLog(oss.str());
                    }

                    mpz_class X_acc = compute_X_with_dots(eng, static_cast<engine::Reg>(RACC), Mp);
                    mpz_class g_mid = gcd_with_dots(X_acc, Mp);
                    bool found_mid = g_mid != 1 && g_mid != Mp;
                    if (found_mid) {
                        char* s = mpz_get_str(nullptr, 10, g_mid.get_mpz_t());
                        std::string fname_mid = "stage3_result_B3_" + std::to_string(b + 1) + "_p_" + std::to_string(options.exponent) + ".txt";
                        writeStageResult(fname_mid, "B3=" + std::to_string(b + 1) + "  factor=" + std::string(s));
                        std::cout << "\n>>>  Factor P-1 (stage 3, intermediate) found : " << s << '\n';
                        if (guiServer_) {
                            std::ostringstream oss;
                            oss << "\n>>>  Factor P-1 (stage 3, intermediate) found : " << s << "\n";
                            guiServer_->appendLog(oss.str());
                        }
                        options.knownFactors.push_back(std::string(s));
                        midFactor = g_mid;
                        foundIntermediate = true;
                        uint64_t tmpB3 = options.B3;
                        options.B3 = b + 1;
                        std::string json_mid = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
                        std::cout << "Manual submission JSON (stage 3 intermediate):\n" << json_mid << "\n";
                        io::WorktodoManager wm_mid(options);
                        wm_mid.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage3_intermediate", json_mid);
                        wm_mid.appendToResultsTxt(json_mid);
                        options.B3 = tmpB3;
                        std::free(s);
                    }

                    eng->set(static_cast<engine::Reg>(RACC), 1);
                    blocks_since_last_check = 0;
                }
            }

            steps_in_block = 0;
        }

        auto now = high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDisplay).count() >= 3) {
            double percent = B3u ? (100.0 * double(b + 1) / double(B3u)) : 100.0;
            double elapsedSec= std::chrono::duration<double>(now - t0).count() + restored_time;
            double etaSec = (b + 1) ? elapsedSec * (double(B3u - (b + 1)) / double(b + 1)) : 0.0;
            int days = int(etaSec) / 86400;
            int hours = (int(etaSec) % 86400) / 3600;
            int minutes = (int(etaSec) % 3600) / 60;
            int seconds = int(etaSec) % 60;
            std::cout << "Stage3: " << std::fixed << std::setprecision(2) << percent
                      << "% | b=" << (b + 1) << "/" << B3u
                      << " | K=" << Kmax
                      << " | Elapsed " << elapsedSec << "s | ETA "
                      << days << "d " << hours << "h " << minutes << "m " << seconds << "s\r" << std::endl;
            if (guiServer_) {
                std::ostringstream oss; oss << "Stage3: " << std::fixed << std::setprecision(2) << percent
                                            << "% | b=" << (b + 1) << "/" << B3u
                                            << " | K=" << Kmax;
                guiServer_->appendLog(oss.str());
                guiServer_->setProgress(double(b + 1), double(B3u), "Stage 3");
            }
            lastDisplay = now;
        }
        if (now - lastBackup >= std::chrono::seconds(options.backup_interval)) {
            double et = duration<double>(now - t0).count() + restored_time;
            std::cout << "\nBackup Stage 3 at b=" << (b + 1) << " start...\n";
            save_ckpt_s3(eng, b + 1, et);
            lastBackup = now;
            std::cout << "Backup Stage 3 done.\n";
            if (guiServer_) { std::ostringstream oss; oss << "Backup Stage 3 at b=" << (b + 1); guiServer_->appendLog(oss.str()); }
        }
    }

    auto t1 = high_resolution_clock::now();
    double elapsed = duration<double>(t1 - t0).count() + restored_time;

    mpz_class X  = compute_X_with_dots(eng, static_cast<engine::Reg>(RACC), Mp);
    mpz_class g  = gcd_with_dots(X, Mp);
    bool found_final = g != 1 && g != Mp;
    bool found = found_final || foundIntermediate;

    std::cout << "\nElapsed time (stage 3) = " << std::fixed << std::setprecision(2) << elapsed << " s." << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "Elapsed time (stage 3) = " << std::fixed << std::setprecision(2) << elapsed << " s."; guiServer_->appendLog(oss.str()); }

    std::string filename = "stage3_result_B3_" + std::to_string(B3u) + "_p_" + std::to_string(options.exponent) + ".txt";
    if (found_final) {
        char* s = mpz_get_str(nullptr, 10, g.get_mpz_t());
        writeStageResult(filename, "B3=" + std::to_string(B3u) + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 3) found : " << s << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\n>>>  Factor P-1 (stage 3) found : " << s << "\n"; guiServer_->appendLog(oss.str()); }
        options.knownFactors.push_back(std::string(s));
        std::free(s);
    } else if (!foundIntermediate) {
        writeStageResult(filename, "No factor P-1 up to B3=" + std::to_string(B3u));
        std::cout << "\nNo factor P-1 (stage 3) until B3 = " << B3u << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\nNo factor P-1 (stage 3) until B3 = " << B3u << '\n'; guiServer_->appendLog(oss.str()); }
    } else {
        char* s = mpz_get_str(nullptr, 10, midFactor.get_mpz_t());
        writeStageResult(filename, "B3=" + std::to_string(B3u) + "  factor=" + std::string(s));
        std::cout << "\n>>>  Factor P-1 (stage 3) found (from intermediate) : " << s << '\n';
        if (guiServer_) { std::ostringstream oss; oss << "\n>>>  Factor P-1 (stage 3) found (from intermediate) : " << s << "\n"; guiServer_->appendLog(oss.str()); }
        std::free(s);
    }

    std::remove(ckpt_file_s3.c_str());
    std::remove((ckpt_file_s3 + ".old").c_str());
    std::remove((ckpt_file_s3 + ".new").c_str());

    std::string json = io::JsonBuilder::generate(options, static_cast<int>(context.getTransformSize()), false, "", "");
    std::cout << "Manual submission JSON:\n" << json << "\n";
    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, std::string(options.mode) + "_stage3", json);
    wm.appendToResultsTxt(json);

    delete eng;
    return found ? 0 : 1;
}

