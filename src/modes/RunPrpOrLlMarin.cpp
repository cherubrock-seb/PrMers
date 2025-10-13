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
    const bool verbose = true;//options.debug;

    engine* eng = engine::create_gpu(p, static_cast<size_t>(8), static_cast<size_t>(options.device_id), verbose  /*,options.chunk256*/);

    //auto to_hex16 = [](uint64_t u){ std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u; return ss.str(); };

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

    //const uint32_t B_GL = std::max<uint32_t>(uint32_t(std::sqrt(p)), 2u);
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
    (void) lastIter;
    (void) lastJ;
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
                        //cl_event postEvt;
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
        for (uint32_t k = proofPower; /*no-cond*/; ) {
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
            if (k == 0) break;
            --k;
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