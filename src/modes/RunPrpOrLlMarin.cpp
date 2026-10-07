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
    if (auto code = QuickChecker::run(options, hasWorktodoEntry_)) return *code;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;//options.debug;

    engine* eng = engine::create_gpu(p, static_cast<size_t>(8), static_cast<size_t>(options.device_id), verbose  /*,options.chunk256*/);
    const bool active_backend_aevum = eng->is_aevum_backend();
    const int active_transform_words = static_cast<int>(eng->get_size());
    const char* active_backend_name = active_backend_aevum ? "Aevum" : "Marin";

    //auto to_hex16 = [](uint64_t u){ std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u; return ss.str(); };

    if (verbose) {
        if (options.mode == "ll") {
            std::cout << "LL-UNSAFE on 2^" << p << " - 1 using " << active_backend_name
                      << " engine with " << eng->get_size() << " words" << std::endl;
        } else {
            std::cout << "PRP on 2^" << p << " - 1 using " << active_backend_name
                      << " engine with " << eng->get_size() << " words" << std::endl;
        }
    }
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
    if (options.mode == "prp" && options.proof) {
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
    if (options.mode == "ll") ck << "llunsafe_";
    ck << "m_" << p << ".ckpt";
    const std::string ckpt_file = ck.str();
    const uint32_t checkpoint_mode = options.mode == "prp" ? 1u : 2u;
    const uint32_t checkpoint_backend = eng->is_aevum_backend() ? 2u : 1u;

    // Version 3 adds the Gerbicz-Li block size after the elapsed time; it is written only when
    // the block size differs from sqrt(p), the size of earlier versions, which version 2 implies.
    // Version 4 (written whenever Gerbicz-Li checking is on) always carries the block size (0 means
    // sqrt(p)) followed by goodIter, the iteration of the verified state held in R4/R5.  R0/R1 are
    // the state at the saved iteration, which may be past the last full check and so unverified.
    // Older files do not record goodIter, so their R4/R5 cannot be placed in the iteration sequence.
    // While the restore point is still the never-verified state of such a file, saves keep the old
    // format (version 3, or 2 for sqrt(p)): writing version 4 would record that state as verified, and
    // a restart would then skip the early check and the hint about the old file.
    const bool gl_active = options.mode == "prp" && options.gerbiczli;
    uint32_t ckpt_block = 0;
    uint64_t goodIter = 0;
    // True while R4/R5 hold the state of an older checkpoint that has not passed a full check yet.
    bool restore_point_unverified = false;
    auto read_ckpt = [&](const std::string& file, uint32_t& ri, double& et, uint32_t& block,
                         bool& has_good, uint32_t& good)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != p) return -2;
        if (version > 4) return -2;
        if (version >= 2) {
            uint32_t saved_mode = 0, saved_backend = 0;
            if (!f.read(reinterpret_cast<char*>(&saved_mode), sizeof(saved_mode))) return -2;
            if (!f.read(reinterpret_cast<char*>(&saved_backend), sizeof(saved_backend))) return -2;
            if (saved_mode != checkpoint_mode || saved_backend != checkpoint_backend) {
                std::cout << "[Checkpoint] Ignoring incompatible " << file
                          << " (mode/backend mismatch)." << std::endl;
                return -3;
            }
        } else if (version == 1) {
            // Legacy checkpoints have no backend tag.  They are safe to keep only
            // for the original Marin PRP path.  LL-unsafe now has its own filename.
            if (checkpoint_mode != 1u || checkpoint_backend != 1u) {
                std::cout << "[Checkpoint] Ignoring legacy untagged " << file
                          << " for this mode/backend." << std::endl;
                return -3;
            }
        } else return -2;
        if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        block = 0;
        has_good = false;
        good = 0;
        if (version >= 3) {
            if (!f.read(reinterpret_cast<char*>(&block), sizeof(block))) return -2;
            if (block < 2 && !(version >= 4 && block == 0)) return -2;
        }
        if (version >= 4) {
            if (!f.read(reinterpret_cast<char*>(&good), sizeof(good))) return -2;
            if (good > ri) return -2;
            has_good = true;
        }
        const size_t cksz = eng->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!eng->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt = [&](uint32_t i, double et){
        const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
        auto write_new = [&]() -> bool {
            File f(newf, "wb");
 if (!f.exists()) return false;
 const bool record_good = gl_active && !restore_point_unverified;
 int version = record_good ? 4 : (ckpt_block ? 3 : 2);
 if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return false;
 if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return false;
 if (!f.write(reinterpret_cast<const char*>(&checkpoint_mode), sizeof(checkpoint_mode))) return false;
 if (!f.write(reinterpret_cast<const char*>(&checkpoint_backend), sizeof(checkpoint_backend))) return false;
 if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return false;
 if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return false;
 if ((ckpt_block || version >= 4) && !f.write(reinterpret_cast<const char*>(&ckpt_block), sizeof(ckpt_block))) return false;
 if (version >= 4) {
 const uint32_t good = static_cast<uint32_t>(goodIter);
 if (!f.write(reinterpret_cast<const char*>(&good), sizeof(good))) return false;
 }
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!eng->get_checkpoint(data)) return false;
            if (!f.write(data.data(), cksz)) return false;
            return f.write_crc32() && f.close();
        };
        if (!write_new()) {
            std::cout << "[Checkpoint] Warning: could not write checkpoint " << newf
                      << "; continuing without saving it (the previous checkpoint, if any, is kept)." << std::endl;
            return;
        }
        std::remove(oldf.c_str());
        struct stat s;
        if ((stat(ckpt_file.c_str(), &s) == 0) && (std::rename(ckpt_file.c_str(), oldf.c_str()) != 0)) return;
        std::rename(newf.c_str(), ckpt_file.c_str());
    };

    const size_t R0 = 0, R1 = 1, R2 = 2, R3 = 3, R4 = 4, R5 = 5, RBASE = 6, RTMP=7;
    uint32_t ri = 0; double restored_time = 0; uint32_t saved_block = 0;
    bool ckpt_has_good = false; uint32_t ckpt_good = 0;
    int r = read_ckpt(ckpt_file, ri, restored_time, saved_block, ckpt_has_good, ckpt_good);
    if (r < 0) r = read_ckpt(ckpt_file + ".old", ri, restored_time, saved_block, ckpt_has_good, ckpt_good);
    if (r != 0) saved_block = 0;
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

    // A test resumed at iteration ri needs every proof residue of the points
    // before ri. If some are missing (or damaged), lower the proof power to
    // the highest one the residues on disk still allow, or drop the proof,
    // now and not after the whole test has run.
    if (r == 0 && ri > 0 && options.mode == "prp" && options.proof) {
        const uint32_t wanted = proofManagerMarin.power();
        const uint32_t usable = ProofSetMarin::effectivePower(options.exponent, wanted, static_cast<uint32_t>(ri));
        if (usable != wanted) {
            std::ostringstream oss;
            if (usable == 0) {
                oss << "Proof residues before iteration " << ri << " are missing: proof generation disabled for this test.";
                options.proof = false;
                options.proofFile.clear();
            } else {
                oss << "Proof residues before iteration " << ri << " are missing: proof of power " << usable << " (instead of " << wanted << ").";
            }
            std::cout << oss.str() << std::endl;
            if (guiServer_) guiServer_->appendLog(oss.str());
            proofManagerMarin.setPower(usable);
            options.proofPower = usable;
        }
    }

    // R4/R5 hold the last verified state.  A fresh start, or a file that does not record
    // goodIter, has no better candidate than the state just loaded.  A version 4 file keeps
    // its own R4/R5 and goodIter: R0/R1 may have been saved after the last full check, so the
    // next check must be able to roll back past them instead of blessing them.
    const bool resumed_verified_state = (r == 0 && ckpt_has_good && gl_active);
    const bool resumed_unverified_legacy = (r == 0 && !ckpt_has_good && gl_active);
    if (resumed_verified_state) {
        goodIter = ckpt_good;
        if (goodIter < ri) {
            std::cout << "[Gerbicz Li] Checkpoint at iteration " << ri << " is past the last verified iteration "
                      << goodIter << "; it will be checked at the next full check." << std::endl;
        }
    } else {
        eng->copy(R4, R0);//Last correct state
        eng->copy(R5, R1);//Last correct bufd
        goodIter = ri;
    }
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

    // Random corruption should disappear after restoring R4/R5.
    // Repeating the same failure three times indicates a deterministic
    // bad transform/plan, so do not loop forever.
    uint32_t gl_failure_streak = 0;
    constexpr uint32_t gl_failure_limit = 3;
    restore_point_unverified = resumed_unverified_legacy;

    uint64_t L = options.exponent;
    // Gerbicz-Li block size B. A full check costs B squarings and each block one multiplication,
    // so with a check about every 600000 iterations B = 1000 costs ~0.3%, where sqrt(p), the size
    // of earlier versions, costs ~2% at the wavefront. A resumed test keeps its block size.
    const uint64_t legacy_B = (uint64_t)(std::sqrt((double)L));
    uint64_t B = legacy_B;
    if (saved_block != 0) {
        B = saved_block;
    } else if (r != 0 && options.mode == "prp" && options.gerbiczli && !options.wagstaff) {
        B = std::min<uint64_t>(options.gl_block >= 2 ? options.gl_block : 1000, legacy_B);
    }
    ckpt_block = (B == legacy_B) ? 0 : static_cast<uint32_t>(B);
    double desiredIntervalSeconds = 600.0;
    uint64_t checkpass = 0;

    uint64_t checkpasslevel_auto = (uint64_t)((1000 * desiredIntervalSeconds) / (double)B);
    if (checkpasslevel_auto == 0) checkpasslevel_auto = (totalIters/B)/((uint64_t)(std::sqrt((double)B)));
    uint64_t checkpasslevel = (options.checklevel > 0)
        ? options.checklevel
        : checkpasslevel_auto;
    if(checkpasslevel==0)
        checkpasslevel=1;
    if (resumed_verified_state && ri > goodIter && ri < totalIters && totalIters > 0) {
        // Count the block boundaries since the last full check so that a run which is
        // interrupted often still reaches its next check on schedule.
        const uint64_t hi = (totalIters - 1 - goodIter) / B;
        const uint64_t lo = (totalIters - ri - 1) / B;
        checkpass = std::min<uint64_t>(hi - lo, checkpasslevel - 1);
    } else if (resumed_unverified_legacy) {
        // R0/R1 of an older checkpoint were never verified: check at the first block boundary.
        checkpass = checkpasslevel - 1;
    }
    if (options.mode == "prp" && options.gerbiczli) {
        std::ostringstream oss;
        oss << "[Gerbicz Li] block size " << B << ", full check every " << B * checkpasslevel << " iterations";
        std::cout << oss.str() << std::endl;
        if (guiServer_) guiServer_->appendLog(oss.str());
    }
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

    // A proof residue that cannot be written or read back makes the proof
    // impossible: say so now and carry on with the PRP instead of failing at
    // the end, or losing the test.
    auto saveProofResidue = [&](engine::digit& d, uint64_t iteration) {
        try {
            proofManagerMarin.checkpointMarin(d, static_cast<uint32_t>(iteration));
        } catch (const core::ProofCheckpointError& e) {
            const std::string msg = std::string("Error: ") + e.what() +
                ". No proof will be produced for this test; "
                "the PRP continues.";
            std::cerr << msg << std::endl;
            if (guiServer_)
                guiServer_->appendLog(msg);
            options.proof = false;
            options.proofFile.clear();
        }
    };

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
                            << "[Gerbicz Li] Restore iter=" << goodIter
                            << " (j=" << (totalIters > goodIter ? totalIters - goodIter - 1 : 0) << ")\n";
                        if (guiServer_) {
                            std::ostringstream oss;
                            oss << "[Gerbicz Li] Mismatch \n"
                            << "[Gerbicz Li] Check FAILED! iter=" << (iter + 1) << "\n"
                            << "[Gerbicz Li] Restore iter=" << goodIter
                            << " (j=" << (totalIters > goodIter ? totalIters - goodIter - 1 : 0) << ")\n";
                            guiServer_->appendLog(oss.str());
                        }
                        checkpass = 0;
                        options.gerbicz_error_count += 1;
                        ++gl_failure_streak;

                        if (gl_failure_streak >= gl_failure_limit) {
                            const std::string reason =
                                "Gerbicz-Li check failed " +
                                std::to_string(gl_failure_streak) +
                                " times in a row from verified iteration " +
                                std::to_string(goodIter) +
                                "; retrying the same state will not help. "
                                "Try -engine-marin or another -aevum-fft plan." +
                                (restore_point_unverified
                                     ? std::string(" The checkpoint resumed from was written by an older version "
                                                   "and was never verified; if it is corrupt, delete ") + ckpt_file +
                                       " to restart from iteration 0."
                                     : std::string());

                            std::cout << "[Gerbicz Li] " << reason << "\n";
                            if (guiServer_) guiServer_->appendLog(
                                std::string("[Gerbicz Li] ") + reason
                            );

                            delete eng;
                            throw std::runtime_error(reason);
                        }

                        eng->copy(R0, R4);
                        eng->copy(R1, R5);

                        // Original loop executes ++iter and --j after this body.
                        // Position it one step before goodIter so the next
                        // arithmetic iteration is exactly the verified state.
                        iter = (goodIter == 0)
                            ? ~uint64_t{0}
                            : goodIter - 1;
                        j = totalIters - goodIter;
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
                        goodIter = iter + 1;
                        gl_failure_streak = 0;
                        restore_point_unverified = false;
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
            saveProofResidue(d, iter + 1);
        }

    }

    // Save the finished (and, with Gerbicz-Li, verified) state.  If the result
    // cannot be saved below, the checkpoint is kept, and this one lets the rerun
    // resume at the end instead of redoing everything since the last periodic
    // backup.  It is removed together with the other checkpoints once the result
    // is saved.  The engine is released before the proof is made, so this has to
    // happen now.
    if (options.mode == "prp" && !(r == 0 && ri == totalIters)) {
        const double final_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
        save_ckpt(static_cast<uint32_t>(totalIters), final_elapsed);
    }

    if (options.mode == "prp" && options.proof) {
        engine::digit d(eng, R0);
        saveProofResidue(d, totalIters);
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
    if (options.mode == "ll" && is_prp_prime && digit.equal_to_Mp()) {
        // The verdict accepts both representations of zero; report the canonical one.
        std::fill(words.begin(), words.end(), 0u);
    }

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

    std::string wagstaff_json;
    if (options.wagstaff) {
            mpz_class Mp = (mpz_class(1) << options.exponent) - 1;
            mpz_class Fp = (mpz_class(1) << options.exponent/2) + 1;
            // Read the exact residue from the engine: the digit widths of the
            // active backend (Marin or Aevum) can differ from Precompute's.
            mpz_class rM;
            {
                mpz_t z0;
                mpz_init(z0);
                eng->get_mpz(z0, R0);
                mpz_set(rM.get_mpz_t(), z0);
                mpz_clear(z0);
            }
            rM %= Mp;
            mpz_class rF = rM % Fp;
            bool isWagstaffPRP = (rF == 9);
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


            // Fall through to the common ending so the verdict is recorded, the checkpoint is
            // removed and a worktodo entry is retired. The Wagstaff number is not 2^p - 1, so it
            // gets its own record: the PrimeNet record for the exponent p would claim 2^p - 1.
            is_prp_prime = isWagstaffPRP;
            wagstaff_json = core::algo::wagstaff_result_json(options.exponent, isWagstaffPRP);
    }
    
    const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
    if (options.wagstaff) {
        // The verdict on the Wagstaff number was reported above.
    }
    else if (options.mode == "prp") {
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

    // Persist a completed PRP/cofactor result before optional proof
    // generation. Proof metadata is intentionally disabled here.
    // The normal final path overwrites this JSON after proof success.
    if (options.mode == "prp" && !options.wagstaff) {
        auto provisionalOptions = options;
        provisionalOptions.proof = false;
        provisionalOptions.proofFile.clear();

        std::string provisionalJson;

        if (!provisionalOptions.knownFactors.empty()) {
            auto [provisionalPrime,
                  provisionalRes64,
                  provisionalRes2048] =
                io::JsonBuilder::computeResultMarin(
                    d, provisionalOptions);

            provisionalJson = io::JsonBuilder::generate(
                provisionalOptions,
                active_transform_words,
                provisionalPrime,
                provisionalRes64,
                provisionalRes2048);
        } else {
            provisionalJson = io::JsonBuilder::generate(
                provisionalOptions,
                active_transform_words,
                is_prp_prime,
                res64_hex,
                res2048_hex);
        }

        io::WorktodoManager provisionalWm(provisionalOptions);
        provisionalWm.saveIndividualJson(
            provisionalOptions.exponent,
            provisionalOptions.mode,
            provisionalJson);
    }

    // The residues are only deleted once a requested proof has been made (and
    // verified, unless -noverify); see ProofSetMarin::residueAction.
    const bool proofRequested = options.mode == "prp" && options.proof;
    bool proofCompleted = false;
    if (options.mode == "prp" && options.proof) {
        try {
            std::cout << "\nGenerating PRP proof file..." << std::endl;
            if (guiServer_)
                guiServer_->appendLog("\nGenerating PRP proof file...");

            // PRP arithmetic is complete and all canonical proof checkpoints
            // are already on disk. Release the arithmetic backend before
            // initializing the proof GPU stack to avoid overlapping VRAM use.
            if (eng) {
                delete eng;
                eng = nullptr;
            }

            std::filesystem::path proofFilePath;
            bool gpuProofReady = false;
            bool gpuProofSucceeded = false;
            std::string gpuProofError;

            try {
                ensureProofGpuBackend();

                if (!buffers || !program || !kernels || !nttEngine)
                    throw std::runtime_error(
                        "PrMers GPU proof backend initialization incomplete");

                gpuProofReady = true;
            }
            catch (const std::exception& e) {
                gpuProofError = e.what();
            }

            if (gpuProofReady) {
                std::cout << "[Proof backend] GPU NTT"
                          << " (PRP backend was "
                          << active_backend_name << ")"
                          << std::endl;

                math::Carry carry(
                    context,
                    context.getQueue(),
                    program->getProgram(),
                    precompute.getN(),
                    precompute.getDigitWidth(),
                    buffers->digitWidthMaskBuf
                );

                uint32_t proofPower = options.proofPower;

                for (;;) {
                    try {
                        options.proofPower = proofPower;

                        proofFilePath = proofManager.proof(
                            context,
                            *nttEngine,
                            carry,
                            proofPower,
                            options.verify
                        );
                        gpuProofSucceeded = true;
                        break;
                    }
                    catch (const core::ProofVerificationError&) {
                        // Not retried and not replaced by the unverified CPU
                        // proof: handled by the outer catch.
                        throw;
                    }
                    catch (const std::exception& e) {
                        gpuProofError = e.what();

                        std::cerr
                            << "Warning: GPU proof generation failed: "
                            << e.what() << std::endl;

                        if (proofPower == 0)
                            break;

                        --proofPower;

                        std::cout
                            << "Retrying proof generation with reduced power: "
                            << proofPower << std::endl;
                    }
                }
            }
            if (!gpuProofReady || !gpuProofSucceeded) {
                std::cerr
                    << "[Proof backend] GPU NTT unavailable, "
                    << "falling back to CPU GMP: "
                    << gpuProofError << std::endl;

                proofFilePath = proofManagerMarin.proof();
                // The GPU retries above lowered options.proofPower; the CPU
                // proof always uses the full power the checkpoints were saved
                // for, and that is what the result JSON must report.
                options.proofPower = proofManagerMarin.power();
            }

            options.proofFile = proofFilePath.string();
            proofCompleted = true;

            std::cout
                << "Proof file saved: "
                << proofFilePath << std::endl;

            if (guiServer_)
                guiServer_->appendLog(
                    "Proof file saved: " +
                    proofFilePath.string());
        }
        catch (const core::ProofVerificationError& e) {
            const std::string msg = std::string("Error: ") + e.what() +
                ". No proof will be reported for this test; "
                "the PRP result is still valid.";
            std::cerr << msg << std::endl;
            if (guiServer_)
                guiServer_->appendLog(msg);
            options.proof = false;
            options.proofFile.clear();
        }
        catch (const std::exception& e) {
            std::cerr
                << "Warning: Proof generation failed: "
                << e.what() << std::endl;

            if (guiServer_)
                guiServer_->appendLog(
                    std::string("Warning: Proof generation failed: ") +
                    e.what());

            // No proof file exists: do not report proof metadata.
            options.proof = false;
            options.proofFile.clear();
        }
    }

    std::string json;
    if (options.wagstaff) {
        json = wagstaff_json;
    }
    else if (!options.knownFactors.empty()) {
        auto [isPrime, res64, res2048] = io::JsonBuilder::computeResultMarin(d, options);
    
        is_prp_prime = isPrime;
        json = io::JsonBuilder::generate(
            options,
            active_transform_words,
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
            active_transform_words,
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
    /*bool skippedSubmission = false;
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
    }*/

    // End of job, in this order: 1. save the result, 2. retire the worktodo entry,
    // 3. only then delete this run's checkpoint and, when applicable, proof residues.
    // If saving or retirement fails, resumable state is kept.
    io::WorktodoManager wm(options);
    bool resultSaved = wm.saveIndividualJson(options.wagstaff ? options.exponent / 2 : options.exponent,
                                             options.wagstaff ? "wagstaff" : options.mode, json);
    resultSaved = wm.appendToResultsTxt(json) && resultSaved;
    const bool retired = resultSaved && hasWorktodoEntry_ &&
                         worktodoParser_->removeProcessedLine(activeWorktodoRawLine_);
    const bool entryRetired = !hasWorktodoEntry_ || retired;
    if (resultSaved && entryRetired) {
        // Remove this run's own checkpoint (llunsafe_ for LL, wagstaff_ for Wagstaff,
        // plain for PRP). Using the exact ckpt_file avoids deleting another mode's state.
        std::error_code ec;
        std::filesystem::remove(ckpt_file, ec);
        std::filesystem::remove(ckpt_file + ".old", ec);
        std::filesystem::remove(ckpt_file + ".new", ec);
        backupManager.clearState();
    }
    const auto residueAction = ProofSetMarin::residueAction(
        options.mode == "prp", options.wagstaff, proofRequested,
        proofCompleted, resultSaved, entryRetired);
    if (residueAction == ProofSetMarin::ResidueAction::Clear) {
        // The proof (if any) is written and the plain PRP test is over: the
        // residues are of no further use and can consume many gigabytes.
        ProofSetMarin::clearResidues(options.exponent);
    } else if (residueAction != ProofSetMarin::ResidueAction::NotApplicable) {
        const std::string msg =
            ProofSetMarin::residuesKeptMessage(options.exponent, residueAction);
        std::cerr << msg << std::endl;
        if (guiServer_)
            guiServer_->appendLog(msg);
    }
    if (!resultSaved || !entryRetired) {
        const std::string msg =
            (resultSaved ? "Failed to update " + options.worktodo_path : std::string("Result could not be saved"))
            + "; keeping the checkpoint"
            + (hasWorktodoEntry_ ? " and the entry in " + options.worktodo_path : std::string());
        std::cerr << msg << "\n";
        if (guiServer_)
            guiServer_->appendLog(msg + "\n");
    }
    if (hasWorktodoEntry_ && resultSaved) {
        if (retired) {
            std::cout << "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
            if (guiServer_) {
                std::ostringstream oss;
                oss  <<   "Entry removed from " << options.worktodo_path
                      << " and saved to worktodo_save.txt\n";
                guiServer_->appendLog(oss.str());
            }
            bool more = io::WorktodoParser::hasPendingEntry(options.worktodo_path);
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
            if (!options.gui) {
                std::exit(-1);
            }
        }
    }

    if (eng) delete eng;
    return is_prp_prime ? 0 : 1;
}
