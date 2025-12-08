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

int App::runLlSafeMarinDoubling()
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
    const bool verbose = true;//options.debug;
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

    const size_t RV = 0, RU = 1, RVC = 2, RUC = 3, RTMP = 4, RVCHK = 5, RUCHK = 6, RSCR [[maybe_unused]] = 7;

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

    /*if (options.submit && !options.gui) {
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
    }*/

    delete_checkpoints(options.exponent, options.wagstaff, true, true);
    delete eng;
    return is_prime ? 0 : 1;
}

int App::runLlSafeMarin()
{
    if (guiServer_) { guiServer_->setProgress(0, 100, "Started"); guiServer_->setStatus("LL-SAFE"); }
    Printer::banner(options);
    if (auto code = QuickChecker::run(options.exponent)) return *code;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const bool verbose = true;
    engine* eng = engine::create_gpu(p, static_cast<size_t>(18), static_cast<size_t>(options.device_id), verbose);
    if (verbose) std::cout << "LL-SAFE on 2^" << p << " - 1 using Marin engine with " << eng->get_size() << " words" << std::endl;
    if (guiServer_) { std::ostringstream oss; oss << "LL-SAFE on 2^" << p << " - 1"; guiServer_->setStatus(oss.str()); }

    std::ostringstream ck; ck << "llsafe_m_" << p << ".ckpt";
    const std::string ckpt_file = ck.str();

    auto read_ckpt = [&](const std::string& file, uint64_t& iter_done, uint64_t& itersave, uint64_t& jsave, double& et)->int{
        File f(file);
        if (!f.exists()) return -1;
        int version = 1; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
        uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
        if (rp != p) return -2;
        if (!f.read(reinterpret_cast<char*>(&iter_done), sizeof(iter_done))) return -2;
        if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
        if (version >= 2) {
            if (!f.read(reinterpret_cast<char*>(&itersave), sizeof(itersave))) return -2;
            if (!f.read(reinterpret_cast<char*>(&jsave), sizeof(jsave))) return -2;
        } else {
            const uint64_t T = (p > 1) ? (uint64_t)(p - 1) : 0ull;
            itersave = iter_done ? iter_done - 1 : 0;
            jsave = (T > iter_done) ? (T - iter_done - 1) : 0;
        }
        const size_t cksz = eng->get_checkpoint_size();
        std::vector<char> data(cksz);
        if (!f.read(data.data(), cksz)) return -2;
        if (!eng->set_checkpoint(data)) return -2;
        if (!f.check_crc32()) return -2;
        return 0;
    };

    auto save_ckpt = [&](uint64_t iter_done, uint64_t itersave, uint64_t jsave, double et){
        const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
        {
            File f(newf, "wb");
            int version = 2;
            if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
            if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return;
            if (!f.write(reinterpret_cast<const char*>(&iter_done), sizeof(iter_done))) return;
            if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return;
            if (!f.write(reinterpret_cast<const char*>(&itersave), sizeof(itersave))) return;
            if (!f.write(reinterpret_cast<const char*>(&jsave), sizeof(jsave))) return;
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

    const size_t RRES_A=0, RRES_B=1, RACC_A=2, RACC_B=3, RCHK_A=4, RCHK_B=5, RSAVE_R_A=6, RSAVE_R_B=7, RSAVE_F_A=8, RSAVE_F_B=9, RBASE_A=10, RBASE_B=11, RTa=12, RTb=13, RM0=14, RM1=15, RPREV_A=16, RPREV_B=17;

    auto pair_square = [&](size_t A, size_t B){
        eng->copy(static_cast<engine::Reg>(RTa), static_cast<engine::Reg>(A));
        eng->copy(static_cast<engine::Reg>(RTb), static_cast<engine::Reg>(B));
        eng->square_mul(static_cast<engine::Reg>(A));
        eng->square_mul(static_cast<engine::Reg>(B), static_cast<uint32_t>(3));
        eng->add(static_cast<engine::Reg>(A), static_cast<engine::Reg>(B));
        eng->copy(static_cast<engine::Reg>(B), static_cast<engine::Reg>(RTa));
        eng->set_multiplicand(static_cast<engine::Reg>(RM0), static_cast<engine::Reg>(RTb));
        eng->mul(static_cast<engine::Reg>(B), static_cast<engine::Reg>(RM0), static_cast<uint32_t>(2));
    };

    auto pair_mul_by = [&](size_t A, size_t B, size_t C, size_t D){
        eng->set_multiplicand(static_cast<engine::Reg>(RM0), static_cast<engine::Reg>(C));
        eng->set_multiplicand(static_cast<engine::Reg>(RM1), static_cast<engine::Reg>(D));
        eng->copy(static_cast<engine::Reg>(RTa), static_cast<engine::Reg>(A));
        eng->copy(static_cast<engine::Reg>(RTb), static_cast<engine::Reg>(B));
        eng->copy(static_cast<engine::Reg>(A), static_cast<engine::Reg>(RTa));
        eng->mul(static_cast<engine::Reg>(A), static_cast<engine::Reg>(RM0));
        eng->copy(static_cast<engine::Reg>(B), static_cast<engine::Reg>(RTa));
        eng->mul(static_cast<engine::Reg>(B), static_cast<engine::Reg>(RM1));
        eng->copy(static_cast<engine::Reg>(RTa), static_cast<engine::Reg>(RTb));
        eng->mul(static_cast<engine::Reg>(RTa), static_cast<engine::Reg>(RM1), static_cast<uint32_t>(3));
        eng->add(static_cast<engine::Reg>(A), static_cast<engine::Reg>(RTa));
        eng->copy(static_cast<engine::Reg>(RTa), static_cast<engine::Reg>(RTb));
        eng->mul(static_cast<engine::Reg>(RTa), static_cast<engine::Reg>(RM0));
        eng->add(static_cast<engine::Reg>(B), static_cast<engine::Reg>(RTa));
    };

    auto pair_equal = [&](size_t A1, size_t B1, size_t A2, size_t B2)->bool{
        mpz_t z1a,z1b,z2a,z2b; mpz_inits(z1a,z1b,z2a,z2b,nullptr);
        eng->get_mpz(z1a, static_cast<engine::Reg>(A1));
        eng->get_mpz(z1b, static_cast<engine::Reg>(B1));
        eng->get_mpz(z2a, static_cast<engine::Reg>(A2));
        eng->get_mpz(z2b, static_cast<engine::Reg>(B2));
        bool ok = (mpz_cmp(z1a,z2a) == 0) && (mpz_cmp(z1b,z2b) == 0);
        mpz_clears(z1a,z1b,z2a,z2b,nullptr);
        return ok;
    };

    auto res64_pair = [&](std::string& out){
        engine::digit dA(eng, static_cast<engine::Reg>(RRES_A));
        engine::digit dB(eng, static_cast<engine::Reg>(RRES_B));
        std::vector<uint32_t> WA = pack_words_from_eng_digits(dA, p);
        std::vector<uint32_t> WB = pack_words_from_eng_digits(dB, p);
        std::string ha = format_res64_hex(WA), hb = format_res64_hex(WB);
        std::ostringstream os; os << "A:" << ha << " B:" << hb; out = os.str();
    };

    uint64_t iter_done = 0; double restored_time = 0.0;
    uint64_t itersave_meta = 0, jsave_meta = 0;
    int r = read_ckpt(ckpt_file, iter_done, itersave_meta, jsave_meta, restored_time);
    if (r < 0) r = read_ckpt(ckpt_file + ".old", iter_done, itersave_meta, jsave_meta, restored_time);
    if (r == 0) {
        std::cout << "Resuming from a checkpoint." << std::endl;
        if (guiServer_) { std::ostringstream oss; oss << "Resuming from a checkpoint."; guiServer_->appendLog(oss.str()); }
    } else {
        iter_done = 0;
        restored_time = 0.0;
        eng->set(static_cast<engine::Reg>(RRES_A), static_cast<uint32_t>(2));
        eng->set(static_cast<engine::Reg>(RRES_B), static_cast<uint32_t>(1));
        eng->set(static_cast<engine::Reg>(RACC_A), static_cast<uint32_t>(1));
        eng->set(static_cast<engine::Reg>(RACC_B), static_cast<uint32_t>(0));
        eng->copy(static_cast<engine::Reg>(RSAVE_R_A), static_cast<engine::Reg>(RRES_A));
        eng->copy(static_cast<engine::Reg>(RSAVE_R_B), static_cast<engine::Reg>(RRES_B));
        eng->copy(static_cast<engine::Reg>(RSAVE_F_A), static_cast<engine::Reg>(RACC_A));
        eng->copy(static_cast<engine::Reg>(RSAVE_F_B), static_cast<engine::Reg>(RACC_B));
    }

    eng->set(static_cast<engine::Reg>(RBASE_A), static_cast<uint32_t>(2));
    eng->set(static_cast<engine::Reg>(RBASE_B), static_cast<uint32_t>(1));

    logger.logStart(options);
    timer.start();
    timer2.start();

    const uint64_t totalIters = (p > 1) ? (uint64_t)(p - 1) : 0ull;
    uint64_t L = options.exponent;
    uint64_t B = std::sqrt((double)L);
    if (B < 1) B = 1;
    if (B > totalIters && totalIters > 0) B = totalIters;

    double desiredIntervalSeconds = 600.0;
    uint64_t checkpass = 0;
    uint64_t checkpasslevel_auto = (uint64_t)((1000.0 * desiredIntervalSeconds) / (double)B);
    if (checkpasslevel_auto == 0 && B > 0) checkpasslevel_auto = (totalIters / std::max<uint64_t>(B,1)) / std::max<uint64_t>(1,(uint64_t)std::sqrt((double)B));
    uint64_t checkpasslevel = options.checklevel > 0 ? (uint64_t)options.checklevel : checkpasslevel_auto;
    if (checkpasslevel == 0) checkpasslevel = 1;

    auto start_clock = std::chrono::high_resolution_clock::now();
    auto lastBackup = start_clock;
    auto lastDisplay = start_clock;

    uint64_t resumeIter = iter_done;
    uint64_t startIter  = iter_done;
    uint64_t itersave = (r==0) ? itersave_meta : (resumeIter ? resumeIter - 1 : 0);
    uint64_t jsave = (r==0) ? jsave_meta : (totalIters - resumeIter - 1);
    std::string res64_x; res64_pair(res64_x);
    spinner.displayProgress((uint32_t)resumeIter, (uint32_t)totalIters, 0.0, 0.0, p, (uint32_t)resumeIter, (uint32_t)startIter, res64_x, guiServer_ ? guiServer_.get() : nullptr);

    bool errordone = false;
    for (uint64_t iter = resumeIter, j = totalIters - resumeIter - 1; iter < totalIters; ++iter, --j)
    {
        if (interrupted) {
            const double elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_clock).count() + restored_time;
            save_ckpt(iter, itersave, jsave, elapsed_time);
            std::cout << "\nInterrupted by user, state saved at iteration " << iter << " j=" << j << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Interrupted by user, state saved at iteration " << iter << " j=" << j; guiServer_->appendLog(oss.str()); }
            spinner.displayBackupInfo((uint32_t)iter, (uint32_t)totalIters, timer.elapsed(), res64_x, guiServer_ ? guiServer_.get() : nullptr);
            logger.logEnd(elapsed_time);
            delete eng;
            return 0;
        }

        auto now0 = std::chrono::high_resolution_clock::now();
        if (now0 - lastBackup >= std::chrono::seconds(options.backup_interval))
        {
            const double elapsed_time = std::chrono::duration<double>(now0 - start_clock).count() + restored_time;
            std::cout << "\nBackup point done at iter + 1=" << iter + 1 << " start...." << std::endl;
            save_ckpt(iter, itersave, jsave, elapsed_time);
            lastBackup = now0;
            std::cout << "Backup point done at iter + 1=" << iter + 1 << " done." << std::endl;
            spinner.displayBackupInfo((uint32_t)(iter + 1), (uint32_t)totalIters, timer.elapsed(), res64_x, guiServer_ ? guiServer_.get() : nullptr);
        }

        if (options.erroriter > 0 && (iter + 1) == (uint64_t)options.erroriter && !errordone) {
            errordone = true;
            eng->sub(static_cast<engine::Reg>(RRES_A), static_cast<uint32_t>(2));
            std::cout << "Injected error at iteration " << (iter + 1) << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "Injected error at iteration " << (iter + 1); guiServer_->appendLog(oss.str()); }
        }

        if (iter + 1 == totalIters) {
            eng->copy(static_cast<engine::Reg>(RPREV_A), static_cast<engine::Reg>(RRES_A));
            eng->copy(static_cast<engine::Reg>(RPREV_B), static_cast<engine::Reg>(RRES_B));
        }
        pair_square(RRES_A, RRES_B);

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDisplay).count() >= 10) {
            spinner.displayProgress((uint32_t)(iter + 1), (uint32_t)totalIters, timer.elapsed(), timer2.elapsed(), p, (uint32_t)resumeIter, (uint32_t)startIter, "", guiServer_ ? guiServer_.get() : nullptr);
            timer2.start();
            lastDisplay = now;
            resumeIter = iter + 1;
        }

        bool boundary = ((j != 0 && (j % B == 0)) || (iter == totalIters - 1));
        if (!boundary) continue;

        checkpass += 1;
        eng->copy(static_cast<engine::Reg>(RCHK_A), static_cast<engine::Reg>(RACC_A));
        eng->copy(static_cast<engine::Reg>(RCHK_B), static_cast<engine::Reg>(RACC_B));
        pair_mul_by(RACC_A, RACC_B, RRES_A, RRES_B);

        bool do_check = (checkpass >= checkpasslevel) || (iter == totalIters - 1);
        if (!do_check) continue;

        const uint64_t T = totalIters;
        const uint64_t modB = (T % B == 0 ? B : T % B);
        uint64_t loop_count = (B > modB ? B - modB - 1 : 0);
        for (uint64_t z = 0; z < loop_count; ++z) pair_square(RCHK_A, RCHK_B);
        if (modB == B) { pair_mul_by(RCHK_A, RCHK_B, RBASE_A, RBASE_B); }
        else { pair_square(RCHK_A, RCHK_B); pair_mul_by(RCHK_A, RCHK_B, RBASE_A, RBASE_B); }
        for (uint64_t z = 0; z < modB; ++z) pair_square(RCHK_A, RCHK_B);

        if (!pair_equal(RCHK_A, RCHK_B, RACC_A, RACC_B)) {
            uint64_t blk_start = (itersave + 1);
            uint64_t blk_end = (iter + 1);
            std::cout << "[Gerbicz-Li] Check FAILED at iter=" << (iter + 1) << " block=[" << blk_start << ".." << blk_end << "]" << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz-Li] Check FAILED at iter=" << (iter + 1) << " block=[" << blk_start << ".." << blk_end << "]"; guiServer_->appendLog(oss.str()); }
            options.gerbicz_error_count += 1;

            eng->copy(static_cast<engine::Reg>(RRES_A), static_cast<engine::Reg>(RSAVE_R_A));
            eng->copy(static_cast<engine::Reg>(RRES_B), static_cast<engine::Reg>(RSAVE_R_B));
            eng->copy(static_cast<engine::Reg>(RACC_A), static_cast<engine::Reg>(RSAVE_F_A));
            eng->copy(static_cast<engine::Reg>(RACC_B), static_cast<engine::Reg>(RSAVE_F_B));

            checkpass = 0;
            resumeIter = (itersave == 0) ? 0 : (itersave + 1);
            if (itersave == 0) { iter = (uint64_t)-1; j = jsave+1; } else { iter = itersave; j = jsave; }
            std::cout << "[Gerbicz-Li] Restore iter=" << iter + 1 << std::endl;
            std::cout << "[Gerbicz-Li] Restore j=" << j - 1 << std::endl;
            continue;
        } else {
            uint64_t blk_start = (itersave + 1);
            uint64_t blk_end = (iter + 1);
            std::cout << "[Gerbicz-Li] Check OK at iter=" << (iter + 1) << " block=[" << blk_start << ".." << blk_end << "]" << std::endl;
            if (guiServer_) { std::ostringstream oss; oss << "[Gerbicz-Li] Check OK at iter=" << (iter + 1) << " block=[" << blk_start << ".." << blk_end << "]"; guiServer_->appendLog(oss.str()); }
            eng->copy(static_cast<engine::Reg>(RSAVE_R_A), static_cast<engine::Reg>(RRES_A));
            eng->copy(static_cast<engine::Reg>(RSAVE_R_B), static_cast<engine::Reg>(RRES_B));
            eng->copy(static_cast<engine::Reg>(RSAVE_F_A), static_cast<engine::Reg>(RACC_A));
            eng->copy(static_cast<engine::Reg>(RSAVE_F_B), static_cast<engine::Reg>(RACC_B));
            itersave = iter;
            jsave = j;
            checkpass = 0;
        }
    }

    mpz_class Mp = (mpz_class(1) << p) - 1;
    mpz_class Mp_minus_1 = Mp - 1;

    mpz_t za, zb; mpz_inits(za, zb, nullptr);
    eng->get_mpz(za, static_cast<engine::Reg>(RRES_A));
    eng->get_mpz(zb, static_cast<engine::Reg>(RRES_B));
    bool okA = (mpz_cmp(za, Mp_minus_1.get_mpz_t()) == 0);
    bool okB = (mpz_cmp_ui(zb, 0) == 0);
    bool is_prime = okA && okB;
    mpz_clears(za, zb, nullptr);

    eng->copy(static_cast<engine::Reg>(RPREV_A), static_cast<engine::Reg>(RPREV_A));
    eng->add(static_cast<engine::Reg>(RPREV_A), static_cast<engine::Reg>(RPREV_A));
    engine::digit dLL(eng, static_cast<engine::Reg>(RPREV_A));
    std::vector<uint32_t> WLL = pack_words_from_eng_digits(dLL, p);
    std::string ll_res64   = format_res64_hex(WLL);
    std::string ll_res2048 = format_res2048_hex(WLL);

    auto end_clock = std::chrono::high_resolution_clock::now();
    const double elapsed_time = std::chrono::duration<double>(end_clock - start_clock).count() + restored_time;

    spinner.displayProgress((uint64_t)totalIters, (uint64_t)totalIters, timer.elapsed(), timer2.elapsed(), p, (uint64_t)totalIters, (uint64_t)iter_done, ll_res64, guiServer_ ? guiServer_.get() : nullptr);

    std::cout << "LL residue s_{p-2} res64=0x" << ll_res64 << std::endl;
    std::cout << "LL residue s_{p-2} res2048=0x" << ll_res2048 << std::endl;
    std::cout << "2^" << p << " - 1 is " << (is_prime ? "prime" : "composite") << ", time = " << std::fixed << std::setprecision(2) << elapsed_time << " s." << std::endl;
    if (guiServer_) {
        std::ostringstream oss;
        oss << "LL residue s_{p-2} res64=0x" << ll_res64 << "\n";
        oss << "LL residue s_{p-2} res2048=0x" << ll_res2048 << "\n";
        oss << "2^" << p << " - 1 is " << (is_prime ? "prime" : "composite") << ", time = " << std::fixed << std::setprecision(2) << elapsed_time << " s.";
        guiServer_->appendLog(oss.str());
    }

    std::string json = io::JsonBuilder::generate(
        options,
        static_cast<int>(context.getTransformSize()),
        is_prime,
        ll_res64,
        ll_res2048
    );
    Printer::finalReport(options, elapsed_time, json, is_prime);

    io::WorktodoManager wm(options);
    wm.saveIndividualJson(options.exponent, "llsafe", json);
    wm.appendToResultsTxt(json);

    delete_checkpoints(p, options.wagstaff, false, true);
    logger.logEnd(elapsed_time);
    delete eng;
    return is_prime ? 0 : 1;
}