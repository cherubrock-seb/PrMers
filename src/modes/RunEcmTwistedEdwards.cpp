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
# include <io.h>
#else
# include <unistd.h>
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
#include <ctime>

using namespace core;
using namespace std::chrono;
using std::string;
using std::vector;
using std::ostringstream;
using std::ofstream;
using std::cout;
using std::endl;

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
namespace fs = std::filesystem;

static bool ecm_progress_use_single_line() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return ::isatty(fileno(stdout)) != 0;
#endif
}

static void ecm_print_progress_line(const std::string& s) {
    static const bool single_line = ecm_progress_use_single_line();
    if (single_line) {
        std::cout << '\r' << s << std::flush;
    } else {
        std::cout << s << std::endl;
    }
}


static uint64_t ecm_isqrt_u64(uint64_t x) {
    long double r = std::sqrt((long double)x);
    uint64_t y = (uint64_t)r;
    while ((y + 1) > 0 && (y + 1) <= x / (y + 1)) ++y;
    while (y > 0 && y > x / y) --y;
    return y;
}

static std::vector<uint32_t> ecm_sieve_base_primes(uint32_t limit) {
    std::vector<uint8_t> isPrime(limit + 1, 1);
    if (limit == 0) {
        isPrime[0] = 0;
    } else {
        isPrime[0] = 0;
        isPrime[1] = 0;
    }
    for (uint32_t p = 2; (uint64_t)p * p <= limit; ++p) {
        if (!isPrime[p]) continue;
        for (uint64_t m = (uint64_t)p * p; m <= limit; m += p) isPrime[(size_t)m] = 0;
    }
    std::vector<uint32_t> primes;
    primes.reserve(limit / 10 + 16);
    for (uint32_t i = 2; i <= limit; ++i) if (isPrime[i]) primes.push_back(i);
    return primes;
}

static void ecm_segmented_primes_odd(uint64_t low, uint64_t high,
                                     const std::vector<uint32_t>& basePrimes,
                                     std::vector<uint64_t>& out) {
    out.clear();
    if (high < 2 || low > high) return;
    if (low <= 2 && 2 <= high) out.push_back(2);

    if (low < 3) low = 3;
    if ((low & 1ULL) == 0) ++low;
    if ((high & 1ULL) == 0) --high;
    if (low > high) return;

    const uint64_t nOdds = ((high - low) >> 1) + 1;
    std::vector<uint8_t> isPrime((size_t)nOdds, 1);

    for (uint32_t p32 : basePrimes) {
        if (p32 == 2) continue;
        const uint64_t p = (uint64_t)p32;
        const uint64_t p2 = p * p;
        if (p2 > high) break;
        uint64_t start = ((low + p - 1) / p) * p;
        if (start < p2) start = p2;
        if ((start & 1ULL) == 0) start += p;
        for (uint64_t x = start; x <= high; x += (p << 1)) {
            isPrime[(size_t)((x - low) >> 1)] = 0;
        }
    }

    out.reserve((size_t)(nOdds / 10 + 16));
    for (uint64_t i = 0; i < nOdds; ++i) {
        if (isPrime[(size_t)i]) out.push_back(low + (i << 1));
    }
}


namespace ecm_local {

struct EC_mod4 {
    struct Pt { mpz_class x, y; bool inf=false; };

    static inline void norm(mpz_class& z, const mpz_class& N) {
        mpz_mod(z.get_mpz_t(), z.get_mpz_t(), N.get_mpz_t());
        if (z < 0) z += N;
    }

    static Pt dbl(const Pt& P, const mpz_class& N) {
        if (P.inf) return P;
        mpz_class num = 3 * P.x * P.x + 4;
        mpz_class den = 2 * P.y, inv;
        norm(num, N); norm(den, N);
        if (mpz_sgn(den.get_mpz_t()) == 0) return Pt{{}, {}, true};
        if (!mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()))
            return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        mpz_class x3 = (lambda * lambda - 2 * P.x) % N; if (x3 < 0) x3 += N;
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }

    static Pt add(const Pt& P, const Pt& Q, const mpz_class& N) {
        if (P.inf) {return Q;}
        if (Q.inf) {return P;}
        if (P.x == Q.x) {
            mpz_class ysum = (P.y + Q.y) % N; if (ysum < 0) ysum += N;
            if (ysum == 0) return Pt{{}, {}, true};
            return dbl(P, N);
        }
        mpz_class num = Q.y - P.y; norm(num, N);
        mpz_class den = Q.x - P.x; norm(den, N);
        mpz_class inv;
        if (!mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()))
            return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        mpz_class x3 = (lambda * lambda - P.x - Q.x) % N; if (x3 < 0) x3 += N;
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }
    #ifdef _MSC_VER
    #  include <intrin.h>
    #endif

    static inline int msb_index_u64(uint64_t n) {
        if (!n) return -1;
    #if defined(_MSC_VER) && !defined(__clang__)
        unsigned long idx;
    #if defined(_M_X64) || defined(_M_ARM64)
        _BitScanReverse64(&idx, n);
        return (int)idx;
    #else
        // 32-bit MSVC fallback
        unsigned long hi = (unsigned long)(n >> 32);
        if (hi) { _BitScanReverse(&idx, hi); return (int)idx + 32; }
        _BitScanReverse(&idx, (unsigned long)(n & 0xFFFFFFFFu));
        return (int)idx;
    #endif
    #else
        // GCC/Clang
        return 63 - __builtin_clzll(n);
    #endif
    }

    static void get(uint64_t n, int s1, int t1, const mpz_class& N, mpz_class& s, mpz_class& t) {
        Pt P0, P;
        P0.x = s1; if (s1 < 0) P0.x += N; P0.x %= N;
        P0.y = t1; if (t1 < 0) P0.y += N; P0.y %= N;
        P    = P0;

        int msb = msb_index_u64(n);
        for (int b = msb - 1; b >= 0; --b) {
            P = dbl(P, N);
            if (((n >> b) & 1ULL) != 0) P = add(P, P0, N);
            if (P.inf) { s = 0; t = 0; return; }
        }
        s = P.x; t = P.y;
    }
};

} // namespace ecm_local

namespace core {

int App::runECMMarinTwistedEdwards()
{
    using namespace std;
    using namespace std::chrono;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const uint64_t B1 = options.B1 ? options.B1 : 1000000ULL;
    const uint64_t B2 = options.B2 ? options.B2 : 0ULL;
    uint64_t curves = options.nmax ? options.nmax : (options.K ? options.K : 250);
    const bool verbose = true;
    const bool forceCurve = (options.curve_seed != 0ULL);
    const bool forceSigma = !options.sigma.empty();
    if (forceCurve) curves =1ULL;
    const uint32_t progress_interval_ms = (options.ecm_progress_interval_ms > 0) ? options.ecm_progress_interval_ms : 2000;

    auto u64_bits = [](uint64_t x)->size_t{ if(!x) return 1; size_t n=0; while(x){ ++n; x>>=1; } return n; };

    auto splitmix64_step = [](uint64_t& x)->uint64_t{
        x += 0x9E3779B97f4A7C15ULL;
        uint64_t z=x;
        z^=z>>30; z*=0xBF58476D1CE4E5B9ULL;
        z^=z>>27; z*=0x94D049BB133111EBULL;
        z^=z>>31;
        return z;
    };
    auto splitmix64_u64 = [&](uint64_t seed0)->uint64_t{
        uint64_t s=seed0;
        return splitmix64_step(s);
    };
    auto mix64 = [&](uint64_t seed, uint64_t idx)->uint64_t{
        uint64_t x = seed ^ 0x9E3779B97f4A7C15ULL;
        x ^= (idx+1) * 0xBF58476D1CE4E5B9ULL;
        x ^= (idx+0x100) * 0x94D049BB133111EBULL;
        return splitmix64_u64(x);
    };
    auto rnd_mpz_bits = [&](const mpz_class& N, uint64_t seed0, unsigned bits)->mpz_class{
        mpz_class z = 0;
        uint64_t s = seed0;
        for (unsigned i=0;i<bits;i+=64){
            z <<= 64;
            z += (unsigned long)splitmix64_step(s);
        }
        z %= N;
        if (z <= 2) z += 3;
        return z;
    };
    auto fmt_hms = [&](double s)->string{
        uint64_t u=(uint64_t)(s+0.5);
        uint64_t h=u/3600,m=(u%3600)/60,se=u%60;
        ostringstream ss; ss<<h<<"h "<<m<<"m "<<se<<"s";
        return ss.str();
    };

    mpz_class N = (mpz_class(1) << p) - 1;
    const mpz_class N_full = N;

    auto run_start = high_resolution_clock::now();
    uint32_t bits_B1 = u64_bits(B1);
    uint32_t bits_B2 = u64_bits(B2 ? B2 : B1);
    uint64_t mersenne_digits = (uint64_t)mpz_sizeinbase(N.get_mpz_t(),10);

    bool wrote_result = false;
    string mode_name = "twisted_edwards";
    const bool te_use_torsion16 = (!options.notorsion && options.torsion16) && !forceSigma;
    string torsion_name = te_use_torsion16 ? string("16") : string("none");
    string result_status = "not_found";
    mpz_class result_factor = 0;
    uint64_t curves_tested_for_found = 0;
    size_t transform_size_once = 0;

    auto write_result = [&](){
        if (wrote_result) return;
        double tot = duration<double>(high_resolution_clock::now() - run_start).count();
        uint64_t tested = curves_tested_for_found ?
                          curves_tested_for_found :
                          ((result_status=="found") ? curves_tested_for_found : curves);
        ofstream jf("ecm_result.json", ios::app);
        ostringstream js;
        js<<"{";
        js<<"\"exponent\":"<<p<<",";
        js<<"\"mode\":\""<<mode_name<<"\",";
        js<<"\"torsion\":\""<<torsion_name<<"\",";
        js<<"\"B1\":"<<B1<<",";
        js<<"\"B2\":"<<B2<<",";
        js<<"\"bits_B1\":"<<bits_B1<<",";
        js<<"\"bits_B2\":"<<bits_B2<<",";
        js<<"\"factor\":\""<<(result_factor>0? result_factor.get_str() : string("0"))<<"\",";
        js<<"\"factor_digits\":"<<(result_factor>0? (int)mpz_sizeinbase(result_factor.get_mpz_t(),10) : 0)<<",";
        js<<"\"mersenne_digits\":"<<mersenne_digits<<",";
        js<<fixed<<setprecision(6);
        js<<"\"elapsed_total_s\":"<<tot<<",";
        js<<"\"elapsed_avg_per_curve_s\":"<<(tested? (tot/double(tested)) : 0.0)<<",";
        js<<"\"curves_tested\":"<<tested<<",";
        js<<"\"status\":\""<<result_status<<"\"";
        js<<"}";
        jf<<js.str()<<"\n";
        wrote_result = true;
    };

    auto is_known = [&](const mpz_class& g)->bool{
        for (auto &s: options.knownFactors){
            if (s.empty()) continue;
            mpz_class f;
            if (mpz_set_str(f.get_mpz_t(), s.c_str(), 0) != 0) continue;
            if (f < 0) f = -f;
            if (f > 1 && g == f) return true;
        }
        return false;
    };

    auto publish_json = [&](){
        vector<string> saved = options.knownFactors;
        if (result_factor > 0) {
            options.knownFactors.clear();
            options.knownFactors.push_back(result_factor.get_str());
        }
        string json_out = io::JsonBuilder::generate(options, static_cast<int>(transform_size_once), false, string(), string());
        ofstream jf2("ecm_result.json", ios::app);
        jf2 << json_out << "\n";
        cout << "[ECM] json for manual submit to primenet:\n" << json_out << endl;
        options.knownFactors = saved;
        io::WorktodoManager wm(options);
        wm.appendToResultsTxt(json_out);

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
    };


    {
        vector<string> kf = options.knownFactors;
        vector<pair<mpz_class,unsigned>> acc;
        mpz_class C = N;
        for (auto& ss : kf) {
            if (ss.empty()) continue;
            mpz_class f;
            if (mpz_set_str(f.get_mpz_t(), ss.c_str(), 0) != 0) continue;
            if (f < 0) f = -f;
            if (f <= 1) continue;
            mpz_class gk;
            mpz_gcd(gk.get_mpz_t(), f.get_mpz_t(), N.get_mpz_t());
            if (gk > 1) {
                unsigned m = 0;
                while (mpz_divisible_p(C.get_mpz_t(), gk.get_mpz_t())) { C /= gk; ++m; }
                if (m) acc.push_back({gk, m});
            }
        }
        if (!acc.empty()) {
            for (auto& kv : acc) while (mpz_divisible_p(N.get_mpz_t(), kv.first.get_mpz_t())) N /= kv.first;
            if (N == 1) {
                std::cout << "[ECM] Trivial after removing known factors." << std::endl;
                write_result();
                publish_json();
                return 0;
            }
        }
    }

    const std::string ecm_stage1_resume_save_file =
        "resume_p" + std::to_string(p) + "_ECM_TE_B1_" + std::to_string(B1) + ".save";
    const std::string ecm_stage1_resume_p95_file =
        "resume_p" + std::to_string(p) + "_ECM_TE_B1_" + std::to_string(B1) + ".p95";

    auto ecm_now_string = [&]() -> std::string {
        std::time_t t = std::time(nullptr);
        std::tm tmv{};
    #if defined(_WIN32)
        localtime_s(&tmv, &t);
    #else
        localtime_r(&t, &tmv);
    #endif
        std::ostringstream oss;
        oss << std::put_time(&tmv, "%a %b %d %H:%M:%S %Y");
        return oss.str();
    };

    auto append_ecm_stage1_resume_line = [&](uint64_t curve_idx, const mpz_class& Aresume, const mpz_class& xAff, const mpz_class* sigmaForP95)->void {
        mpz_class Ared = Aresume;
        mpz_mod(Ared.get_mpz_t(), Ared.get_mpz_t(), N.get_mpz_t());
        if (Ared < 0) Ared += N;

        mpz_class xred = xAff;
        mpz_mod(xred.get_mpz_t(), xred.get_mpz_t(), N.get_mpz_t());
        if (xred < 0) xred += N;

        const std::string ecmResumeProgram = std::string("PrMers ") + core::PRMERS_VERSION;
        const std::string ecmResumeTime = ecm_now_string();
        const std::string ecmResumeWho = options.user;

        mpz_class chk;
        mpz_set_ui(chk.get_mpz_t(), (unsigned long)B1);
        chk *= mpz_class(mpz_fdiv_ui(Ared.get_mpz_t(), CHKSUMMOD));
        chk *= mpz_class(mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD));
        chk *= mpz_class(mpz_fdiv_ui(xred.get_mpz_t(), CHKSUMMOD));
        const uint32_t chk_u = (uint32_t)mpz_fdiv_ui(chk.get_mpz_t(), CHKSUMMOD);

        {
            std::ofstream out(ecm_stage1_resume_save_file, std::ios::out | std::ios::app);
            if (!out) {
                std::ostringstream oss;
                oss << "[ECM] Warning: cannot append Stage1 resume to '" << ecm_stage1_resume_save_file << "'";
                std::cerr << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
            } else {
                out << "METHOD=ECM; B1=" << B1
                    << "; N=" << N.get_str()
                    << "; X=0x" << xred.get_str(16)
                    << "; A=" << Ared.get_str()
                    << "; CHECKSUM=" << chk_u
                    << "; PROGRAM=" << ecmResumeProgram
                    << "; X0=0x0; Y0=0x0;";
                if (!ecmResumeWho.empty()) out << " WHO=" << ecmResumeWho << ";";
                out << " TIME=" << ecmResumeTime << ";\n";
                out.flush();
            }
        }

        if (sigmaForP95 != nullptr) {
            const mpz_class& sigma = *sigmaForP95;

            mpz_class chk2;
            mpz_set_ui(chk2.get_mpz_t(), (unsigned long)B1);
            chk2 *= mpz_class(mpz_fdiv_ui(sigma.get_mpz_t(), CHKSUMMOD));
            chk2 *= mpz_class(mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD));
            chk2 *= mpz_class(mpz_fdiv_ui(xred.get_mpz_t(), CHKSUMMOD));
            const uint32_t chk_u2 = (uint32_t)mpz_fdiv_ui(chk2.get_mpz_t(), CHKSUMMOD);

            const std::string nField = (N == N_full) ? ("2^" + std::to_string(p) + "-1") : N.get_str();

            std::ofstream outp(ecm_stage1_resume_p95_file, std::ios::out | std::ios::app);
            if (!outp) {
                std::ostringstream oss;
                oss << "[ECM] Warning: cannot append Prime95 Stage2 resume to '" << ecm_stage1_resume_p95_file << "'";
                std::cerr << oss.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(oss.str());
            } else {
                outp << "METHOD=ECM; SIGMA=" << sigma.get_str()
                     << "; B1=" << B1
                     << "; N=" << nField
                     << "; X=0x" << xred.get_str(16)
                     << "; CHECKSUM=" << chk_u2
                     << "; PROGRAM=" << ecmResumeProgram
                     << "; X0=0x0; Y0=0x0;";
                if (!ecmResumeWho.empty()) outp << " WHO=" << ecmResumeWho << ";";
                outp << " TIME=" << ecmResumeTime << ";\n";
                outp.flush();
            }
        }

        std::ostringstream oss;
        oss << "[ECM] Curve " << (curve_idx + 1) << "/" << curves
            << " | Stage1 resume appended: " << ecm_stage1_resume_save_file;
        if (sigmaForP95 != nullptr) oss << " + " << ecm_stage1_resume_p95_file;
        else oss << " (Prime95 export skipped: TE curve is not a GMP-ECM SIGMA family)";
        std::cout << oss.str() << std::endl;
        if (guiServer_) guiServer_->appendLog(oss.str());
    };

    vector<uint64_t> primesB1_v, primesS2_v;
    {
        const uint64_t Pmax = (B2 > B1) ? B2 : B1;
        const uint64_t root = ecm_isqrt_u64(Pmax);
        const std::vector<uint32_t> basePrimes = ecm_sieve_base_primes((uint32_t)root);

        const uint64_t SEG = 1000000ULL;
        std::vector<uint64_t> segPrimes;

        for (uint64_t lo = 2; lo <= B1; ) {
            uint64_t hi = lo + SEG - 1;
            if (hi < lo || hi > B1) hi = B1;
            ecm_segmented_primes_odd(lo, hi, basePrimes, segPrimes);
            primesB1_v.insert(primesB1_v.end(), segPrimes.begin(), segPrimes.end());
            if (hi == B1) break;
            lo = hi + 1;
        }

        if (B2 > B1) {
            uint64_t lo = B1 + 1;
            while (lo <= B2) {
                uint64_t hi = lo + SEG - 1;
                if (hi < lo || hi > B2) hi = B2;
                ecm_segmented_primes_odd(lo, hi, basePrimes, segPrimes);
                primesS2_v.insert(primesS2_v.end(), segPrimes.begin(), segPrimes.end());
                if (hi == B2) break;
                lo = hi + 1;
            }
        }

        std::cout << "[ECM] Prime counts: B1=" << primesB1_v.size()
                  << ", S2=" << primesS2_v.size() << std::endl;
    }

    mpz_class K(1);
    for (uint32_t q : primesB1_v) {
        uint64_t m = q;
        while (m <= B1 / q) m *= q;
        mpz_mul_ui(K.get_mpz_t(), K.get_mpz_t(), m);
    }
    size_t Kbits = mpz_sizeinbase(K.get_mpz_t(),2);



    const int backup_period = options.backup_interval > 0 ? options.backup_interval : 10;

    string torsion_last = torsion_name;

    uint64_t resume_curve_idx  = 0;
    uint64_t resume_curve_seed = 0;
    bool     have_resume_seed  = false;

    auto try_probe_te_ckpt = [&](const std::string& file, uint64_t& out_seed)->bool {
        File f(file);
        if (!f.exists()) return false;

        int version = 0;
        if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return false;
        if (version != 1) return false; // checkpoints S1 Twisted-Edwards

        uint32_t rp = 0;
        if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return false;
        if (rp != p) return false;

        uint32_t i   = 0;
        uint32_t nbb = 0;
        uint64_t rB1 = 0;
        double   et  = 0.0;
        if (!f.read(reinterpret_cast<char*>(&i),   sizeof(i)))   return false;
        if (!f.read(reinterpret_cast<char*>(&nbb), sizeof(nbb))) return false;
        if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return false;
        if (!f.read(reinterpret_cast<char*>(&et),  sizeof(et)))  return false;
        if (rB1 != B1) return false;

        uint64_t saved_curve_seed = 0;
        uint8_t  saved_torsion16  = 0;
        if (!f.read(reinterpret_cast<char*>(&saved_curve_seed), sizeof(saved_curve_seed))) return false;
        if (!f.read(reinterpret_cast<char*>(&saved_torsion16),  sizeof(saved_torsion16)))  return false;
        uint8_t current_torsion16 = te_use_torsion16 ? 1 : 0;
        if (saved_torsion16 != current_torsion16) return false;

        out_seed = saved_curve_seed;
        return true;
    };

    if (!options.seed) {
        for (uint64_t c = 0; c < curves; ++c) {
            const std::string ckpt_file   = "ecm_te_m_"  + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
            const std::string ckpt_file_o = ckpt_file + ".old";

            uint64_t s = 0;
            if (try_probe_te_ckpt(ckpt_file, s) || try_probe_te_ckpt(ckpt_file_o, s)) {
                resume_curve_idx  = c;
                resume_curve_seed = s;
                have_resume_seed  = true;
                break;
            }
        }
    }

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

    uint64_t base_seed;
    if (options.seed) {
        base_seed = options.seed;
    } else if (have_resume_seed) {
        base_seed = resume_curve_seed;
    } else {
        base_seed = (now_ns ^ ((uint64_t)p<<32) ^ B1);
    }

    std::cout << "[ECM] seed=" << base_seed;
    if (!options.seed && have_resume_seed) {
        std::cout << " (resumed from checkpoint, curve index " << (resume_curve_idx + 1) << ")";
    }
    std::cout << std::endl;

    uint64_t start_curve = 0;
    if (!options.seed && have_resume_seed) {
        start_curve = resume_curve_idx;
    }

    for (uint64_t c = start_curve; c < curves; ++c)

    {
        if (core::algo::interrupted) {
            result_status = "interrupted";
            curves_tested_for_found = c;
            options.curves_tested_for_found = (uint32_t)c;
            write_result();
            return 2;
        }

        engine* eng = engine::create_gpu(p, static_cast<size_t>(51), static_cast<size_t>(options.device_id), verbose);
        if (!eng) {
            std::cout<<"[ECM] GPU engine unavailable\n";
            result_status = "error";
            write_result();
            return 1;
        }
        if (transform_size_once == 0) {
            transform_size_once = eng->get_size();
            std::ostringstream os;
            os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id;
            std::cout<<os.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(os.str());
        }

        uint32_t s2_idx = 0, s2_cnt = 0; 
        double   s2_et  = 0.0;
        bool     resume_stage2 = false; 
        (void) resume_stage2;

        uint64_t curve_seed;
        if (!options.seed && have_resume_seed && c == resume_curve_idx) {
            curve_seed = resume_curve_seed;
            base_seed  = curve_seed;
        } else {
            curve_seed = mix64(base_seed, c);
            if (forceCurve){
                curve_seed = options.curve_seed;
                base_seed  = curve_seed;
            }
        }

        std::cout << "[ECM] curve_seed=" << curve_seed << std::endl;
        options.curve_seed = curve_seed;
        options.base_seed  = base_seed;
        


        const std::string ckpt_file      = "ecm_te_m_"  + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt2_file     = "ecm2_te_m_" + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt_legacy    = "ecm_m_"     + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt2_legacy   = "ecm2_m_"    + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";

        
        auto save_ckpt = [&](uint32_t i, double et){
            const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
            {
                File f(newf, "wb");
                int version = 1;
                if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return;
                if (!f.write(reinterpret_cast<const char*>(&p),       sizeof(p)))       return;
                if (!f.write(reinterpret_cast<const char*>(&i),       sizeof(i)))       return;
                uint32_t nbb = (uint32_t)mpz_sizeinbase(K.get_mpz_t(),2);
                if (!f.write(reinterpret_cast<const char*>(&nbb),     sizeof(nbb)))     return;
                if (!f.write(reinterpret_cast<const char*>(&B1),      sizeof(B1)))      return;
                if (!f.write(reinterpret_cast<const char*>(&et),      sizeof(et)))      return;

                if (!f.write(reinterpret_cast<const char*>(&curve_seed), sizeof(curve_seed))) return;
                uint8_t torsion16_flag = te_use_torsion16 ? 1 : 0;
                if (!f.write(reinterpret_cast<const char*>(&torsion16_flag), sizeof(torsion16_flag))) return;

                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!eng->get_checkpoint(data)) return;
                if (!f.write(data.data(), cksz)) return;

                f.write_crc32();
            }
            std::error_code ec;
            fs::remove(ckpt_file + ".old", ec);
            fs::rename(ckpt_file,         ckpt_file + ".old", ec);
            fs::rename(ckpt_file + ".new",ckpt_file,          ec);
            fs::remove(ckpt_file + ".old", ec);
        };

        auto read_ckpt_one = [&](const std::string& file, uint32_t& ri, uint32_t& rnb, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;

            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 1) return -2;

            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp),  sizeof(rp)))  return -2;
            if (rp != p) return -2;

            if (!f.read(reinterpret_cast<char*>(&ri),  sizeof(ri)))  return -2;
            if (!f.read(reinterpret_cast<char*>(&rnb), sizeof(rnb))) return -2;

            uint64_t rB1 = 0;
            if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et),  sizeof(et)))  return -2;

            uint64_t saved_curve_seed = 0;
            uint8_t  saved_torsion16  = 0;
            if (!f.read(reinterpret_cast<char*>(&saved_curve_seed), sizeof(saved_curve_seed))) return -2;
            if (!f.read(reinterpret_cast<char*>(&saved_torsion16),  sizeof(saved_torsion16)))  return -2;
            uint8_t current_torsion16 = te_use_torsion16 ? 1 : 0;
            if (saved_curve_seed != curve_seed || saved_torsion16 != current_torsion16) return -2;

            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data))  return -2;
            if (!f.check_crc32())            return -2;

            if (rnb != mpz_sizeinbase(K.get_mpz_t(),2) || rB1 != B1) return -2;
            return 0;
        };
        auto read_ckpt = [&](uint32_t& ri, uint32_t& rnb, double& et)->int{
            int rr = read_ckpt_one(ckpt_file, ri, rnb, et);
            if (rr < 0) rr = read_ckpt_one(ckpt_file + ".old", ri, rnb, et);
            if (rr < 0) rr = read_ckpt_one(ckpt_legacy, ri, rnb, et);         
            if (rr < 0) rr = read_ckpt_one(ckpt_legacy + ".old", ri, rnb, et);
            return rr;
        };

        auto save_ckpt2 = [&](uint32_t idx, double et, uint32_t cnt_bits){
            const std::string oldf = ckpt2_file + ".old", newf = ckpt2_file + ".new";
            {
                File f(newf, "wb");
                int version = 4;
                if (!f.write(reinterpret_cast<const char*>(&version),  sizeof(version)))  return;
                if (!f.write(reinterpret_cast<const char*>(&p),        sizeof(p)))        return;
                if (!f.write(reinterpret_cast<const char*>(&idx),      sizeof(idx)))      return;
                if (!f.write(reinterpret_cast<const char*>(&cnt_bits), sizeof(cnt_bits))) return;
                if (!f.write(reinterpret_cast<const char*>(&B1),       sizeof(B1)))       return;
                if (!f.write(reinterpret_cast<const char*>(&B2),       sizeof(B2)))       return;
                uint64_t seed64 = (uint64_t)curve_seed;
                if (!f.write(reinterpret_cast<const char*>(&seed64),   sizeof(seed64)))   return;
                uint8_t torsion16_flag = te_use_torsion16 ? 1 : 0;
                if (!f.write(reinterpret_cast<const char*>(&torsion16_flag), sizeof(torsion16_flag))) return;
                if (!f.write(reinterpret_cast<const char*>(&et),       sizeof(et)))       return;

                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!eng->get_checkpoint(data)) return;
                if (!f.write(data.data(), cksz)) return;

                f.write_crc32();
            }
            std::error_code ec;
            fs::remove(ckpt2_file + ".old", ec);
            if (fs::exists(ckpt2_file)) fs::rename(ckpt2_file, ckpt2_file + ".old", ec);
            fs::rename(ckpt2_file + ".new", ckpt2_file, ec);
            fs::remove(ckpt2_file + ".new", ec);
        };

        auto read_ckpt2_one = [&](const std::string& file, uint32_t& idx, uint32_t& cnt_bits, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;

            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 2 && version != 3 && version != 4) return -2;

            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;

            if (!f.read(reinterpret_cast<char*>(&idx),      sizeof(idx)))      return -2;
            if (!f.read(reinterpret_cast<char*>(&cnt_bits), sizeof(cnt_bits))) return -2;

            uint64_t b1s = 0, b2s = 0;
            if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return -2;

            uint64_t saved_seed = 0;
            uint8_t  saved_tor  = 0;
            if (version == 4) {
                if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_tor),  sizeof(saved_tor)))  return -2;
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            } else if (version == 3) {
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_seed), sizeof(saved_seed))) return -2;
                if (!f.read(reinterpret_cast<char*>(&saved_tor),  sizeof(saved_tor)))  return -2;
            } else {
                if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            }

            if (version >= 3) {
                uint8_t current_tor = te_use_torsion16 ? 1 : 0;
                if (saved_seed != (uint64_t)curve_seed || saved_tor != current_tor) return -2;
            }

            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32())           return -2;

            if (b1s != B1 || b2s != B2) return -2;
            return 0;
        };
        auto read_ckpt2 = [&](uint32_t& idx, uint32_t& cnt_bits, double& et)->int{
            int rr = read_ckpt2_one(ckpt2_file, idx, cnt_bits, et);
            if (rr < 0) rr = read_ckpt2_one(ckpt2_file + ".old", idx, cnt_bits, et);
            return rr;
        };

        auto addm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r=a+b;
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto subm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r=a-b;
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto mulm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{
            mpz_class r;
            mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto sqrm = [&](const mpz_class& a)->mpz_class{
            mpz_class r;
            mpz_mul(r.get_mpz_t(), a.get_mpz_t(), a.get_mpz_t());
            mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
            if (r<0) r+=N;
            return r;
        };
        auto invm = [&](const mpz_class& a, mpz_class& inv)->int{
            if (mpz_sgn(a.get_mpz_t())==0) return -1;
            if (mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t())) return 0;
            mpz_class g;
            mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) {
                result_factor = g;
                result_status = "found";
                return 1;
            }
            return -1;
        };
        
       // auto sqrQ = [](const mpq_class& z)->mpq_class { return z*z; };
        /*auto pow4Q = [&](const mpq_class& z)->mpq_class { mpq_class t=sqrQ(z); return t*t; };

        auto mpq_to_mod = [&](const mpq_class& q, mpz_class& out)->int{
            mpz_class num(q.get_num()), den(q.get_den());
            mpz_class inv;
            int r = invm(den, inv);
            if (r==1) return 1;    
            if (r<0)  return -1;   
            out = mulm(num, inv);
            return 0;
        };*/

        struct QPt { mpq_class x, y; bool inf=false; };
        /*auto q_add = [&](const QPt& P, const QPt& Q)->QPt{
            if (P.inf) {return Q;} if (Q.inf) {return P;}
            if (P.x==Q.x && P.y==-Q.y) return QPt{{}, {}, true};
            mpq_class lambda;
            if (P.x==Q.x && P.y==Q.y) {
                lambda = (3*sqrQ(P.x) + mpq_class(4)) / (mpq_class(2)*P.y);
            } else {
                lambda = (Q.y - P.y) / (Q.x - P.x);
            }
            mpq_class x3 = sqrQ(lambda) - P.x - Q.x;
            mpq_class y3 = lambda*(P.x - x3) - P.y;
            return QPt{x3,y3,false};
        };*/
        /*auto q_mul = [&](QPt P, uint32_t k)->QPt{
            QPt R{{}, {}, true};
            while (k){
                if (k&1u) R = q_add(R,P);
                P = q_add(P,P);
                k >>= 1u;
            }
            return R;
        };*/

        mpz_class aE, dE, X0, Y0;
        mpz_class sigma_resume;
        bool have_sigma_resume = false;
        
        string torsion_used = te_use_torsion16 ? string("16") : string("none");
                             // (options.torsion16 ? string("16") : string("8"));
        bool built = false;

        // In-memory checkpoint of the last known good state for fast recovery on errors
        std::vector<char> last_good_state;
        bool have_last_good_state = false;
        uint32_t last_good_iter = 0;
        uint32_t current_iter_for_invariant = 0;
        const size_t ctx_ckpt_size = eng->get_checkpoint_size();

        auto force_on_curve = [&](mpz_class& aE_ref,
                                  mpz_class& dE_ref,
                                  const mpz_class& X,
                                  const mpz_class& Y)->bool
        {
            mpz_class X2 = sqrm(X);
            mpz_class Y2 = sqrm(Y);
            mpz_class XY2 = mulm(X2, Y2);
            mpz_class invXY2;
            int r = invm(XY2, invXY2);
            if (r==1) return false;
            if (r<0) return false;
            dE_ref = mulm(subm(addm(mulm(aE_ref, X2), Y2), mpz_class(1)), invXY2);
            return true;
        };
        // ---- torsion 16 (Gallot / Theorem 2.5) : a = 1 ----
        if (te_use_torsion16) {
            bool ok = false;
            for (uint32_t tries = 0; tries < 128 && !ok; ++tries) {
                //uint64_t m = (mix64(base_seed, c ^ (0x9E37u + tries)) | 1ULL) % 1000000ULL;
                uint64_t m = (mix64(curve_seed, tries) | 1ULL) % 1000000ULL;
                if (m < 2) m = 2;

                mpz_class s, t;
                ecm_local::EC_mod4::get(m, 4, 8, N, s, t);

                aE = mpz_class(1);

                auto inv_or_gcd = [&](const mpz_class& den, mpz_class& inv)->bool {
                    int r = invm(den, inv);
                    if (r == 1) {
                        curves_tested_for_found = c+1; options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result(); publish_json(); delete eng; return false;
                    }
                    return (r == 0);
                };

                mpz_class inv, alpha, alpha2, r, x, y;
                mpz_class den = subm(s, mpz_class(4));
                if (!inv_or_gcd(den, inv)) return 0;
                alpha  = mulm(addm(t, mpz_class(8)), inv);
                alpha2 = sqrm(alpha);

                den = subm(mpz_class(8), alpha2);
                if (!inv_or_gcd(den, inv)) return 0;
                r = mulm(addm(mpz_class(8), mulm(mpz_class(2), alpha)), inv);

                mpz_class two_r_minus1 = subm(mulm(mpz_class(2), r), mpz_class(1));
                mpz_class t1 = sqrm(two_r_minus1);

                mpz_class numD = addm(subm(mulm(mpz_class(8), sqrm(r)), mulm(mpz_class(8), r)), mpz_class(1));
                mpz_class t1sq = sqrm(t1);
                if (!inv_or_gcd(t1sq, inv)) return 0;
                dE = mulm(numD, inv);

                mpz_class numX = mulm(subm(mpz_class(8), alpha2), subm(mulm(mpz_class(2), sqrm(r)), mpz_class(1)));
                mpz_class denX = addm(subm(mulm(mpz_class(2), s), alpha2), mpz_class(4));
                if (!inv_or_gcd(denX, inv)) return 0;
                X0 = mulm(numX, inv);

                mpz_class denY = subm(mulm(mpz_class(4), r), mpz_class(3));
                if (!inv_or_gcd(denY, inv)) return 0;
                Y0 = mulm(t1, inv);

                if (X0 == 0 || Y0 == 0) continue;

                auto X2 = sqrm(X0), Y2 = sqrm(Y0);
                auto L  = addm(X2, Y2);
                auto R  = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
                if (subm(L, R) != 0) continue;

                built = ok = true;
                torsion_used = "16";
            }

        }

        if (!built && !te_use_torsion16) {
            bool sigma_built = false;
            for (uint32_t tries = 0; tries < 256 && !sigma_built; ++tries) {
                mpz_class sigma_mpz;
                if (forceSigma) {
                    if (tries > 0) break;
                    if (mpz_set_str(sigma_mpz.get_mpz_t(), options.sigma.c_str(), 0) != 0) {
                        std::cerr << "[ECM] Invalid -sigma value: " << options.sigma << std::endl;
                        delete eng;
                        return 0;
                    }
                } else {
                    const uint64_t sigma_seed = (tries == 0) ? curve_seed : mix64(curve_seed, 0x5EED5EEDULL + uint64_t(tries));
                    sigma_mpz = rnd_mpz_bits(N, sigma_seed, 64);
                }

                sigma_mpz %= N;
                if (sigma_mpz < 0) sigma_mpz += N;
                if (sigma_mpz == 0) sigma_mpz = 1;

                auto inv_or_finish = [&](const mpz_class& den, mpz_class& inv)->int {
                    int r = invm(den, inv);
                    if (r == 1) {
                        curves_tested_for_found = c+1;
                        options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result();
                        publish_json();
                        delete eng;
                        return 1;
                    }
                    return r;
                };

                mpz_class sigma2 = sqrm(sigma_mpz);
                mpz_class su = subm(sigma2, mpz_class(5));
                mpz_class sv = mulm(mpz_class(4), sigma_mpz);

                mpz_class inv_sv;
                {
                    int r = inv_or_finish(sv, inv_sv);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }

                mpz_class denA = mulm(mpz_class(4), mulm(mulm(sqrm(su), su), sv));
                mpz_class inv_denA;
                {
                    int r = inv_or_finish(denA, inv_denA);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }

                mpz_class tnum = mulm(sqrm(subm(sv, su)), subm(sv, su));
                mpz_class Aplus2 = mulm(mulm(tnum, addm(mulm(mpz_class(3), su), sv)), inv_denA);
                mpz_class Bmont = mulm(su, inv_sv);
                mpz_class invB;
                {
                    int r = inv_or_finish(Bmont, invB);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }

                aE = mulm(Aplus2, invB);
                dE = mulm(subm(Aplus2, mpz_class(4)), invB);

                mpz_class sp = addm(sigma2, mpz_class(5));
                mpz_class denX = mulm(mulm(subm(su, sv), addm(su, sv)), sp);
                mpz_class inv_denX;
                {
                    int r = inv_or_finish(denX, inv_denX);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }
                X0 = mulm(mulm(sqrm(su), sv), inv_denX);

                mpz_class su3 = mulm(sqrm(su), su);
                mpz_class sv3 = mulm(sqrm(sv), sv);
                mpz_class denY = addm(su3, sv3);
                mpz_class inv_denY;
                {
                    int r = inv_or_finish(denY, inv_denY);
                    if (r == 1) return 0;
                    if (r < 0) {
                        if (forceSigma) {
                            std::cerr << "[ECM] Invalid/singular -sigma on this modulus" << std::endl;
                            delete eng;
                            return 0;
                        }
                        continue;
                    }
                }
                Y0 = mulm(subm(su3, sv3), inv_denY);

                if (X0 == 0 || Y0 == 0) {
                    if (forceSigma) {
                        std::cerr << "[ECM] Invalid/singular -sigma generated a degenerate point" << std::endl;
                        delete eng;
                        return 0;
                    }
                    continue;
                }

                auto X2 = sqrm(X0), Y2 = sqrm(Y0);
                auto L  = addm(mulm(aE, X2), Y2);
                auto R  = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
                if (subm(L, R) != 0) {
                    if (forceSigma) {
                        std::cerr << "[ECM] Internal curve conversion failure for -sigma" << std::endl;
                        delete eng;
                        return 0;
                    }
                    continue;
                }

                options.sigma_hex = sigma_mpz.get_str(16);
                sigma_resume = sigma_mpz;
                have_sigma_resume = true;
                built = true;
                sigma_built = true;
                torsion_used = "none";
            }
        }

        if (!built && !te_use_torsion16) {
            std::cout << "[ECM] Could not build a SIGMA-based Twisted Edwards curve for this seed, retrying curve\n";
            delete eng;
            continue;
        }

        if(!built){
            aE = subm(N, mpz_class(2));
            for (int tries=0; tries<256 && !built; ++tries){
                X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xC0FFEEull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 256);
                Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xFACEFEEDull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 256);
                if (X0==0 || Y0==0) continue;
                if (!force_on_curve(aE, dE, X0, Y0)) {
                    if (result_factor>0){
                        curves_tested_for_found = c+1;
                        options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result();
                        publish_json();
                        delete eng;
                        return 0;
                    }
                    continue;
                }
                built = true;
            }
        }
        {
            std::ostringstream head;
            head << "[ECM] Curve " << (c+1) << "/" << curves
                 << " | twisted_edwards"
                 << " | torsion=" << torsion_used
                 << " | K_bits=" << mpz_sizeinbase(K.get_mpz_t(),2)
                 << " | seed=" << curve_seed;
            if (have_sigma_resume) head << " | sigma";
            std::cout << head.str() << std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());
        }
        
        auto check_invariant = [&](){
            auto Xv = compute_X_with_dots(eng,(engine::Reg)3,N);
            auto Yv = compute_X_with_dots(eng,(engine::Reg)4,N);
            auto Zv = compute_X_with_dots(eng,(engine::Reg)1,N);
            auto Tv = compute_X_with_dots(eng,(engine::Reg)5,N);
            auto lhs = addm(mulm(aE, sqrm(Xv)), sqrm(Yv));
            auto rhs = addm(sqrm(Zv), mulm(dE, sqrm(Tv)));
            auto rel = subm(lhs, rhs);
            if (rel != 0){
                std::cout << "[ECM] invariant FAIL " //(a="
                          /*<< aE*/ << ")\n";
                return false;
            }
            else{
                std::cout << "[ECM] check invariant OK " // (a="
                          /*<< aE*/ << ")\n";
                // Save an in-memory checkpoint of this good state
                if (ctx_ckpt_size > 0) {
                    last_good_state.resize(ctx_ckpt_size);
                    if (eng->get_checkpoint(last_good_state)) {
                        have_last_good_state = true;
                        last_good_iter = current_iter_for_invariant;
                    }
                }
                return true;
            }
        };

        torsion_last = torsion_used;
        torsion_name = torsion_used;

        std::cout<<"[ECM] torsion="<<torsion_used<<std::endl;

        if (!built){
            delete eng;
            continue;
        }

        {
            mpz_class X2 = sqrm(X0);
            mpz_class Y2 = sqrm(Y0);
            mpz_class L = addm(mulm(aE, X2), Y2);
            mpz_class R = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
            std::cout<<"[ECM] check_on_curve="<<(subm(L,R)==0? "OK":"FAIL")<<std::endl;
            std::cout<<"[ECM] Compute in Twisted Edwards Mode"<<std::endl;
            if (subm(L,R)!=0){
                delete eng;
                continue;
            }
        }

        {
            std::ostringstream head;
            head<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                <<" | twisted_edwards | torsion="<<torsion_used
                <<" | K_bits="<<Kbits<<
                "| curve="<<curve_seed;
            std::cout<<head.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());
        }

        {
            mpz_class dummyA24(0);
            mpz_class x0_ref = X0;
        }


        auto hadamard = [&](size_t a, size_t b, size_t s, size_t d){
            eng->addsub((engine::Reg)s, (engine::Reg)d, (engine::Reg)a, (engine::Reg)b); // s=a+b, d=a-b
        };
        /*auto hadamard_copy = [&](size_t a, size_t b, size_t s, size_t d, size_t s_copy, size_t d_copy){
            eng->addsub_copy((engine::Reg)s,(engine::Reg)d,(engine::Reg)s_copy,(engine::Reg)d_copy,
                            (engine::Reg)a,(engine::Reg)b);
        };*/


        // Inputs/outputs mapping :
        // X1=R3, Y1=R4, Z1=R1, T1=R5
        // X2=R6, Y2=R7, Z2=1 (affine), T2=R9
        // constants: a=R16, d=R29

        // eADD_RP: Twisted Edwards extended coordinates (X,Y,Z,T)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2
        // C = d*T1*T2
        // D = Z1*Z2 (here Z2 = 1 so D = Z1)
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1, 35=D1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)7,(engine::Reg)6); // 36=S2, 37=D2

            // 30 = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)6);   // multiplicand <- X2
            eng->mul_copy((engine::Reg)30,(engine::Reg)11, (engine::Reg)39);               // 30 = X1*X2

            // 31 = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);   // multiplicand <- Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

            // 32 = C = d*T1*T2
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)9);   // multiplicand <- T2
            eng->mul((engine::Reg)32,(engine::Reg)46);               // 32 = T1*T2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)29);  // multiplicand <- d
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            hadamard((engine::Reg)1,(engine::Reg)32, (engine::Reg)42,(engine::Reg)41);

            // 38 = E = S1*S2 - X1X2 - Y1Y2
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)30);           // 38 -= X1*X2
            eng->sub_reg((engine::Reg)38,(engine::Reg)31);           // 38 -= Y1*Y2  => E

            // 39 = a*X1*X2 ; 40 = H = Y1*Y2 - a*X1*X2
            //eng->copy((engine::Reg)39,(engine::Reg)30);              // 39 <- X1*X2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);  // multiplicand <- a   
            eng->mul((engine::Reg)39,(engine::Reg)43);               // 39 = a*X1*X2
            eng->copy((engine::Reg)40,(engine::Reg)31);              // 40 <- Y1*Y2
            eng->sub_reg((engine::Reg)40,(engine::Reg)39);           // 40 = H

            // X3 = E*(D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)

            // Z3 = (D - C)*(D + C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)

            // Y3 = H*(D + C)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H

            // T3 = E*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H
        };


        
//Swap 6,7,9 by 47,48,49

        // Inputs/outputs mapping :
        // X1=R3, Y1=R4, Z1=R1, T1=R5
        // X2=R6, Y2=R7, Z2=1 (affine), T2=R9
        // constants: a=R16, d=R29

        // eADD_RP: Twisted Edwards extended coordinates (X,Y,Z,T)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2
        // C = d*T1*T2
        // D = Z1*Z2 (here Z2 = 1 so D = Z1)
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP_2 = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1, 35=D1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)48,(engine::Reg)47); // 36=S2, 37=D2

            // 30 = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)47);   // multiplicand <- X2
            eng->mul_copy((engine::Reg)30,(engine::Reg)11, (engine::Reg)39);               // 30 = X1*X2

            // 31 = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)48);   // multiplicand <- Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

            // 32 = C = d*T1*T2
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)49);  // multiplicand <- T2
            eng->mul((engine::Reg)32,(engine::Reg)50);               // 32 = T1*T2
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            hadamard((engine::Reg)1,(engine::Reg)32, (engine::Reg)42,(engine::Reg)41);

            // 38 = E = S1*S2 - X1X2 - Y1Y2
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)30);           // 38 -= X1*X2
            eng->sub_reg((engine::Reg)38,(engine::Reg)31);           // 38 -= Y1*Y2  => E

            // 39 = a*X1*X2 ; 40 = H = Y1*Y2 - a*X1*X2
            //eng->copy((engine::Reg)39,(engine::Reg)30);              // 39 <- X1*X2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);  // multiplicand <- a   
            eng->mul((engine::Reg)39,(engine::Reg)43);               // 39 = a*X1*X2
            eng->copy((engine::Reg)40,(engine::Reg)31);              // 40 <- Y1*Y2
            eng->sub_reg((engine::Reg)40,(engine::Reg)39);           // 40 = H

            // X3 = E*(D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)

            // Z3 = (D - C)*(D + C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)

            // Y3 = H*(D + C)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H

            // T3 = E*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H
        };



        // eDBL_XYTZ: Twisted Edwards doubling (X,Y,Z,T) with general a (a in R43)
        // Formulas:
        // A = X^2
        // B = Y^2
        // C = 2*Z^2
        // D = a*A
        // E = 2*T*Z
        // G = D + B
        // H = D - B
        // F = G - C
        // X3 = E*F
        // Y3 = G*H
        // T3 = E*H
        // Z3 = F*G
        auto eDBL_XYTZ = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            // E = 2*T*Z  (compute first while Z is intact)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)RZ); // 11 = Z
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // RT = T*Z
            eng->add((engine::Reg)RT,(engine::Reg)RT);              // RT = 2*T*Z = E

            // C = 2*Z^2 (no const-mul: square then add)
            eng->square_mul((engine::Reg)RZ);                       // RZ = Z^2
            eng->add((engine::Reg)RZ,(engine::Reg)RZ);              // RZ = 2*Z^2 = C

            // A = X^2 ; B = Y^2  (in-place)
            eng->square_mul((engine::Reg)RX);                       // RX = A
            eng->square_mul((engine::Reg)RY);                       // RY = B

            // D = a*A
            eng->mul((engine::Reg)RX,(engine::Reg)43);              // RX = D = a*A

            // G = D + B ; H = D - B  (Hadamard on RX=D and RY=B)
            hadamard((engine::Reg)RX,(engine::Reg)RY,
                    (engine::Reg)23,(engine::Reg)25);              // 23=G, 25=H

            // F = G - C
            eng->copy((engine::Reg)24,(engine::Reg)23);             // 24 = G
            eng->sub_reg((engine::Reg)24,(engine::Reg)RZ);          // 24 = F = G - C

            // X3 = E*F ; Z3 = F*G   (first and only set_multiplicand for F)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)24); // 11 = F
            eng->copy((engine::Reg)RX,(engine::Reg)RT);             // RX <- E
            eng->mul((engine::Reg)RX,(engine::Reg)11);              // X3 = E*F
            eng->copy((engine::Reg)RZ,(engine::Reg)23);             // RZ <- G
            eng->mul((engine::Reg)RZ,(engine::Reg)11);              // Z3 = G*F

            // Y3 = G*H ; T3 = E*H   (second and last set_multiplicand for H)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25); // 11 = H
            eng->copy((engine::Reg)RY,(engine::Reg)23);             // RY <- G
            eng->mul((engine::Reg)RY,(engine::Reg)11);              // Y3 = G*H
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // T3 = E*H
        };


        // eDBL_XYTZ_notwist (a = 1)
        // A = X^2, B = Y^2, C = 2*Z^2, E = 2*T*Z
        // G = A + B, H = A - B, F = G - C
        // X3 = E*F, Z3 = F*G, Y3 = G*H, T3 = E*H
        auto eDBL_XYTZ_notwist = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            // --- E = 2*T*Z  (avant d’écraser Z)
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)RZ); // 11 = Z
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // RT = T*Z
            eng->add((engine::Reg)RT,(engine::Reg)RT);              // RT = 2*T*Z = E

            // --- C = 2*Z^2
            eng->square_mul((engine::Reg)RZ);                       // RZ = Z^2
            eng->add((engine::Reg)RZ,(engine::Reg)RZ);              // RZ = 2*Z^2 = C

            // --- A = X^2, B = Y^2  (in-place)
            eng->square_mul((engine::Reg)RX);                       // RX = A
            eng->square_mul((engine::Reg)RY);                       // RY = B

            // --- G = A + B ; H = A - B  + copie de G en 24 (économise un copy)
            // 23=G, 25=H, 24=G (copie), 22=d_copy (scratch, ignoré)
            eng->addsub_copy((engine::Reg)23,(engine::Reg)25,(engine::Reg)24,(engine::Reg)22,
                            (engine::Reg)RX,(engine::Reg)RY);

            // --- F = G - C  (F en 24, G reste en 23)
            eng->sub_reg((engine::Reg)24,(engine::Reg)RZ);          // 24 = F = G - C

            // --- X3 = E*F ; Z3 = F*G
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)24); // 11 = F
            eng->copy((engine::Reg)RX,(engine::Reg)RT);             // RX <- E
            eng->mul((engine::Reg)RX,(engine::Reg)11);              // X3 = E*F
            eng->copy((engine::Reg)RZ,(engine::Reg)23);             // RZ <- G
            eng->mul((engine::Reg)RZ,(engine::Reg)11);              // Z3 = G*F

            // --- Y3 = G*H ; T3 = E*H
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25); // 11 = H
            eng->copy((engine::Reg)RY,(engine::Reg)23);             // RY <- G
            eng->mul((engine::Reg)RY,(engine::Reg)11);              // Y3 = G*H
            eng->mul((engine::Reg)RT,(engine::Reg)11);              // T3 = E*H
        };

        // eADD_RP_notwist : Twisted Edwards addition (a = 1, Z2 = 1)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2 (ici a=1)
        // C = d*T1*T2
        // D = Z1
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP_notwist = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1=Y1+X1, 35=D1=Y1-X1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)7,(engine::Reg)6); // 36=S2=Y2+X2, 37=D2=Y2-X2

            // 30 = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)6);   // multiplicand <- X2
            eng->mul((engine::Reg)30,(engine::Reg)11);               // 30 = X1*X2

            // 31 = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);   // multiplicand <- Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

            // sum & H d’un coup
            eng->addsub((engine::Reg)20,(engine::Reg)40, (engine::Reg)31,(engine::Reg)30); // 20 = Y1Y2+X1X2, 40 = H

            // 32 = C = d*T1*T2   (T2 via 46, d via 45)
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            eng->mul((engine::Reg)32,(engine::Reg)46);               // 32 = T1*T2
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            eng->addsub((engine::Reg)42,(engine::Reg)41, (engine::Reg)1,(engine::Reg)32);

            // 38 = E = S1*S2 - sum
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)20);           // 38 -= sum  => E

            // --- profiter de 11 = H : calculer Y3 et T3 avant de changer 11 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H

            // --- bascule unique vers (D - C) pour X3 et Z3 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)
        };
        // eADD_RP_notwist : Twisted Edwards addition (a = 1, Z2 = 1)
        // Formulas:
        // E = X1*Y2 + Y1*X2
        // H = Y1*Y2 - a*X1*X2 (ici a=1)
        // C = d*T1*T2
        // D = Z1
        // X3 = E*(D - C)
        // Y3 = H*(D + C)
        // T3 = E*H
        // Z3 = (D - C)*(D + C)
        auto eADD_RP_notwist_2 = [&](){
            // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
            eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3);          // 34=S1=Y1+X1, 35=D1=Y1-X1
            eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)48,(engine::Reg)47);        // 36=S2=Y2+X2, 37=D2=Y2-X2

            // 30 = X1*X2  (X2-)
            eng->copy((engine::Reg)30,(engine::Reg)3);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)47);   // multiplicand <- X2-
            eng->mul((engine::Reg)30,(engine::Reg)11);                // 30 = X1*X2

            // 31 = Y1*Y2  (Y2-)
            eng->copy((engine::Reg)31,(engine::Reg)4);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)48);   // multiplicand <- Y2-
            eng->mul((engine::Reg)31,(engine::Reg)11);                // 31 = Y1*Y2

            // sum & H d’un coup
            eng->addsub((engine::Reg)20,(engine::Reg)40, (engine::Reg)31,(engine::Reg)30); // 20 = Y1Y2+X1X2, 40 = H

            // 32 = C = d*T1*T2  (T2- via 49, puis d)
            eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)49);  // multiplicand <- T2-
            eng->mul((engine::Reg)32,(engine::Reg)50);               // 32 = T1*T2
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)29);  // multiplicand <- d
            eng->mul((engine::Reg)32,(engine::Reg)45);               // 32 = d*T1*T2 = C

            // 42 = D + C, 41 = D - C  (D = Z1)
            eng->addsub((engine::Reg)42,(engine::Reg)41, (engine::Reg)1,(engine::Reg)32);

            // 38 = E = S1*S2 - sum
            eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
            eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)20);           // 38 -= sum => E

            // --- profiter de 11 = H : calculer Y3 et T3 avant de changer 11 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);  // multiplicand <- H
            eng->copy((engine::Reg)4,(engine::Reg)42);               // Y3 <- (D + C)
            eng->mul((engine::Reg)4,(engine::Reg)11);                // Y3 = (D + C)*H
            eng->copy((engine::Reg)5,(engine::Reg)38);               // T3 <- E
            eng->mul((engine::Reg)5,(engine::Reg)11);                // T3 = E*H

            // --- bascule unique vers (D - C) pour X3 et Z3 ---
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);  // multiplicand <- (D - C)
            eng->copy((engine::Reg)3,(engine::Reg)38);               // X3 <- E
            eng->mul((engine::Reg)3,(engine::Reg)11);                // X3 = E*(D - C)
            eng->copy((engine::Reg)1,(engine::Reg)42);               // Z3 <- (D + C)
            eng->mul((engine::Reg)1,(engine::Reg)11);                // Z3 = (D + C)*(D - C)
        };


        mpz_t za; mpz_init(za); mpz_set(za, aE.get_mpz_t()); eng->set_mpz((engine::Reg)16, za); mpz_clear(za);
        static const mpz_class TWO = 2;
        mpz_class two_dE = mulm(dE, TWO);
        mpz_t tmp; mpz_init_set(tmp, two_dE.get_mpz_t());
        eng->set_mpz((engine::Reg)17, tmp);
        mpz_clear(tmp);
        mpz_t zd; mpz_init(zd); mpz_set(zd, dE.get_mpz_t());
        eng->set_mpz((engine::Reg)29, zd);
        mpz_clear(zd);
        eng->set((engine::Reg)0, 0u);
        eng->set((engine::Reg)1, 1u);
        mpz_t zx; mpz_init(zx); mpz_set(zx, X0.get_mpz_t()); eng->set_mpz((engine::Reg)6, zx); mpz_clear(zx);
        mpz_t zy; mpz_init(zy); mpz_set(zy, Y0.get_mpz_t()); eng->set_mpz((engine::Reg)7, zy); mpz_clear(zy);
        eng->set((engine::Reg)8, 2u);
        eng->copy((engine::Reg)9,(engine::Reg)6);
        eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
        eng->mul((engine::Reg)9,(engine::Reg)11);
        eng->set((engine::Reg)1, 1u);
        eng->copy((engine::Reg)3,(engine::Reg)6);
        eng->copy((engine::Reg)4,(engine::Reg)7);
        eng->copy((engine::Reg)5,(engine::Reg)9);
        eng->set_multiplicand((engine::Reg)43,(engine::Reg)16); // a
        eng->set_multiplicand((engine::Reg)44,(engine::Reg)8);  // 2
        eng->set_multiplicand((engine::Reg)45,(engine::Reg)29); // d
        eng->set_multiplicand((engine::Reg)46,(engine::Reg)9);  // T2 (will be T_pos)
        uint32_t start_i = 0, nb_ck = 0;
        double   saved_et = 0.0;
        int rr = read_ckpt(start_i, nb_ck, saved_et);
        
        int rr2 = read_ckpt2(s2_idx, s2_cnt, s2_et); 
        resume_stage2 = (rr2 == 0);
        
        

        bool resumed = (rr == 0 && start_i > 0);
        if (!resumed) {
            saved_et = 0.0;
            nb_ck = 0;
        } else {
            // Treat the resumed state as a valid last-good checkpoint
            if (ctx_ckpt_size > 0) {
                last_good_state.resize(ctx_ckpt_size);
                if (eng->get_checkpoint(last_good_state)) {
                    have_last_good_state = true;
                    last_good_iter = start_i;
                    current_iter_for_invariant = start_i;
                }
            }
        }

        auto t0 = high_resolution_clock::now();
        auto last_save = t0;
        auto last_ui   = t0;
        auto last_check = t0;
        size_t last_ui_done = start_i;
        double ema_ips_stage1 = 0.0;

        std::cout<<"[ECM] stage1_begin Kbits="<<Kbits<<std::endl;
        std::vector<short> naf_vec; naf_vec.reserve((size_t)Kbits + 2);
        {
            mpz_class ec = K;
            mpz_ptr e = ec.get_mpz_t();
            for (; mpz_size(e) != 0; )
            {
                short di = 0;
                if (mpz_odd_p(e))
                {
                    unsigned long limb0 = (mpz_size(e) > 0) ? mpz_getlimbn(e, 0) : 0ul;
                    short mod4 = short(limb0 & 3u);
                    di = (mod4 == 1) ? 1 : -1;
                    if (di > 0) mpz_sub_ui(e, e, 1u); else mpz_add_ui(e, e, 1u);
                }
                naf_vec.push_back(di);
                mpz_fdiv_q_2exp(e, e, 1);
            }
            while (!naf_vec.empty() && naf_vec.back()==0) naf_vec.pop_back();
        }
        size_t naf_len = naf_vec.size();
        if (naf_len == 0) { std::cout<<std::endl; }
        if (naf_len == 0) { }
        mpz_class T0_mpz = mulm(X0, Y0);
        mpz_class X0_neg = subm(N, X0);
        mpz_class T0_neg = subm(N, T0_mpz);
        mpz_t zXpos; mpz_init_set(zXpos, X0.get_mpz_t());
        mpz_t zYpos; mpz_init_set(zYpos, Y0.get_mpz_t());
        mpz_t zTpos; mpz_init_set(zTpos, T0_mpz.get_mpz_t());
        mpz_t zXneg; mpz_init_set(zXneg, X0_neg.get_mpz_t());
        mpz_t zYneg; mpz_init_set(zYneg, Y0.get_mpz_t());
        mpz_t zTneg; mpz_init_set(zTneg, T0_neg.get_mpz_t());

        size_t total_steps = (naf_len>=1 ? naf_len-1 : 0);

        eng->set_mpz((engine::Reg)6,  zXpos);   // X2 =  +P.x
        eng->set_mpz((engine::Reg)7,  zYpos);   // Y2 =  +P.y
        eng->set_mpz((engine::Reg)9,  zTpos);   // T2 =  +P.t
        eng->set_mpz((engine::Reg)47, zXneg);   // cache −P.x
        eng->set_mpz((engine::Reg)48, zYneg);   // cache −P.y
        eng->set_mpz((engine::Reg)49, zTneg);   // cache −P.t
        eng->set_multiplicand((engine::Reg)43,(engine::Reg)16); // a
        eng->set_multiplicand((engine::Reg)44,(engine::Reg)8);  // 2
        eng->set_multiplicand((engine::Reg)45,(engine::Reg)29); // d
        eng->set_multiplicand((engine::Reg)46,(engine::Reg)9);  // T2 = +P.t
        eng->set_multiplicand((engine::Reg)50,(engine::Reg)49); // T2neg = −P.t

        if (!resumed && naf_len) {
            short top = naf_vec[naf_len - 1];
            if (top < 0) {
                eng->set_mpz((engine::Reg)3, zXneg);
                eng->set_mpz((engine::Reg)4, zYneg);
                eng->set_mpz((engine::Reg)5, zTneg);
            } else {
                eng->set_mpz((engine::Reg)3, zXpos);
                eng->set_mpz((engine::Reg)4, zYpos);
                eng->set_mpz((engine::Reg)5, zTpos);
            }
        }
        bool errordone = false;
        bool fatal_error = false;
        if(resume_stage2){
            start_i = total_steps;
        }
        for (uint32_t i = start_i; i < total_steps; ++i) {
            if (core::algo::interrupted) {
                double elapsed = duration<double>(high_resolution_clock::now() - t0).count() + saved_et;
                save_ckpt((uint32_t)i, elapsed);
                result_status = "interrupted";
                curves_tested_for_found = c+1;
                options.curves_tested_for_found = (uint32_t)(c+1);
                write_result();
                delete eng;
                return 2;
            }

            
            if(te_use_torsion16){
                eDBL_XYTZ_notwist(3,4,1,5);
            }
            else{
                eDBL_XYTZ(3,4,1,5);
            }
            short di = naf_vec[naf_len - 2 - i];
            if (di != 0){
                if (di > 0){
                    /*eng->set_mpz((engine::Reg)6, zXpos);
                    eng->set_mpz((engine::Reg)7, zYpos);
                    eng->set_mpz((engine::Reg)9, zTpos);*/
                    if(te_use_torsion16) eADD_RP_notwist(); else eADD_RP();
                } else {
                    //eng->set_mpz((engine::Reg)6, zXneg);
                    //eng->set_mpz((engine::Reg)7, zYneg);
                    //eng->set_mpz((engine::Reg)9, zTneg);
                    if(te_use_torsion16) eADD_RP_notwist_2(); else eADD_RP_2();
                }
            }
            if (options.erroriter > 0 && (i + 1) == options.erroriter && !errordone) {
                errordone = true;
                //eng->error();
                eng->sub(1, 2);
                std::cout << "Injected error at iteration " << (i + 1) << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss << "Injected error at iteration " << (i + 1);
                    guiServer_->appendLog(oss.str());
                }
            }
            auto now = high_resolution_clock::now();
            if (duration_cast<seconds>(now - last_check).count() >= options.ecm_check_interval || i+1 == total_steps) {

                std::cout<<"\n[ECM] Error check ...."<<std::endl;
                // remember which iteration this invariant corresponds to
                current_iter_for_invariant = i + 1;
                if(check_invariant()){
                    std::cout<<"[ECM] Error check Done ! ...."<<std::endl;
                }
                else{
                    std::cout<<"[ECM] Error detected!!!!!!!! ...."<<std::endl;
                    if (have_last_good_state) {
                        options.invarianterror += 1;
                        std::cout << "[ECM] Restoring last known good state at iteration "
                                  << last_good_iter << " and retrying from there." << std::endl;
                        if (eng->set_checkpoint(last_good_state)) {
                            // rewind loop to last_good_iter so it will be processed again
                            i = (last_good_iter > 0) ? (last_good_iter - 1) : 0;
                            last_check = now;
                            last_save  = now;
                            last_ui = now;
                            last_ui_done = (last_good_iter > 0) ? size_t(last_good_iter) : size_t(0);
                            ema_ips_stage1 = 0.0;
                            continue;
                        } else {
                            std::cout << "[ECM] Failed to restore last good state, aborting curve." << std::endl;
                        }
                    } else {
                        std::cout << "[ECM] No saved good state available, aborting curve." << std::endl;
                    }
                    fatal_error = true;
                    break;
                }
                    last_check = now;
            }
            if (duration_cast<milliseconds>(now - last_ui).count() >= progress_interval_ms || i+1 == total_steps) {
                const size_t done_u = i + 1;
                const double done = double(done_u), total = double(total_steps ? total_steps : 1);
                const double elapsed = duration<double>(now - t0).count() + saved_et;
                const double avg_ips = done / std::max(1e-9, elapsed);
                const double dt_ui = duration<double>(now - last_ui).count();
                const double dd_ui = double(done_u - last_ui_done);
                const double inst_ips = (dt_ui > 1e-9 && dd_ui >= 0.0) ? (dd_ui / dt_ui) : avg_ips;
                if (ema_ips_stage1 <= 0.0) ema_ips_stage1 = inst_ips;
                else ema_ips_stage1 = 0.75 * ema_ips_stage1 + 0.25 * inst_ips;
                const double eta_ips = (ema_ips_stage1 > 0.0) ? ema_ips_stage1 : avg_ips;
                const double eta = (total > done && eta_ips > 0.0) ? (total - done) / eta_ips : 0.0;
                std::ostringstream line;
                line << "[ECM] Curve " << (c+1) << "/" << curves << " | Stage1 " << (i+1) << "/" << total_steps
                     << " (" << std::fixed << std::setprecision(2) << (done * 100.0 / total) << "%)"
                     << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                     << " | ETA " << fmt_hms(eta);
                ecm_print_progress_line(line.str());
                last_ui_done = done_u;
                last_ui = now;
            }
            if (duration_cast<seconds>(now - last_save).count() >= backup_period) {
                double elapsed = duration<double>(now - t0).count() + saved_et;
                save_ckpt((uint32_t)(i + 1), elapsed);
                last_save = now;
            }
        }
        std::cout << std::endl;
        mpz_clear(zXpos); mpz_clear(zYpos); mpz_clear(zTpos);
        mpz_clear(zXneg); mpz_clear(zYneg); mpz_clear(zTneg);

        if (fatal_error) {
            std::cout << "[ECM] Curve " << (c+1) << " aborted after unrecoverable error, skipping\n";
            delete eng;
            continue;
        }

        if (resume_stage2 && (s2_cnt != (uint32_t)primesS2_v.size() || s2_idx > (uint32_t)primesS2_v.size())) {
            resume_stage2 = false;
            s2_idx = 0;
            s2_cnt = 0;
            s2_et = 0.0;
        }

        if (!resume_stage2) {
            mpz_class Zacc = compute_X_with_dots(eng, (engine::Reg)5, N);
            mpz_class g = gcd_with_dots(Zacc, N);
            if (g == N) {
                std::cout<<"[ECM] Curve "<<(c+1)<<": singular or failure, retrying\n";
                delete eng;
                continue;
            }

            bool found = (g > 1 && g < N);

            double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count() + saved_et;
            {
                std::ostringstream s1;
                s1<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                   <<" | Stage1 elapsed="<<fixed<<setprecision(2)<<elapsed_stage1<<" s";
                std::cout<<s1.str()<<std::endl;
                if (guiServer_) guiServer_->appendLog(s1.str());
            }
            std::cout<<"\n[ECM] Error check ...."<<std::endl;

            if (!found && g == 1) {
                mpz_class Zv = compute_X_with_dots(eng, (engine::Reg)1, N);
                mpz_class Yv = compute_X_with_dots(eng, (engine::Reg)4, N);

                mpz_class den_u = subm(Zv, Yv), inv_den_u;
                int r_u = invm(den_u, inv_den_u);
                if (r_u == 1) found = true;
                if (!found && r_u == 0) {
                    mpz_class uAff = mulm(addm(Zv, Yv), inv_den_u);

                    mpz_class numA = mulm(mpz_class(2), addm(aE, dE));
                    mpz_class denA = subm(aE, dE), invDenA;
                    int r_a = invm(denA, invDenA);
                    if (r_a == 1) found = true;
                    if (!found && r_a == 0) {
                        mpz_class Aresume = mulm(numA, invDenA);
                        append_ecm_stage1_resume_line(c, Aresume, uAff, have_sigma_resume ? &sigma_resume : nullptr);
                    }
                }
            }

            if (found) {
                bool known = is_known(result_factor > 1 ? result_factor : g);
                const mpz_class& gf = (result_factor > 1 ? result_factor : g);
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                         <<(known?" | known factor=":" | factor=")<<gf.get_str()<<std::endl;
                std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
                if (guiServer_) {
                    std::ostringstream oss;
                    oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gf.get_str();
                    guiServer_->appendLog(oss.str());
                }
                if (!known) {
                    std::error_code ec0; fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0);
                    if (!(result_factor > 1)) result_factor = gf;
                    result_status = "found";
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
            }
        }

        if (B2 > B1 && !primesS2_v.empty()) {
            mpz_class A24;
            if (!resume_stage2) {
                mpz_class Zv = compute_X_with_dots(eng, (engine::Reg)1, N);
                mpz_class Yv = compute_X_with_dots(eng, (engine::Reg)4, N);

                mpz_class den = subm(Zv, Yv), invden;
                { int r = invm(den, invden); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class u = mulm(addm(Zv, Yv), invden);

                mpz_class numA = mulm(mpz_class(2), addm(aE, dE));
                mpz_class denA = subm(aE, dE), invDenA;
                { int r = invm(denA, invDenA); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class A = mulm(numA, invDenA);
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);

                mpz_t zA24; mpz_init_set(zA24, A24.get_mpz_t()); eng->set_mpz((engine::Reg)6, zA24); mpz_clear(zA24);
                eng->set_multiplicand((engine::Reg)12, (engine::Reg)6);

                mpz_t zu; mpz_init_set(zu, u.get_mpz_t()); eng->set_mpz((engine::Reg)4, zu); mpz_clear(zu);
                eng->set((engine::Reg)5, 1u);
                eng->set((engine::Reg)0, 1u);
                eng->set((engine::Reg)1, 0u);
                eng->copy((engine::Reg)2, (engine::Reg)4);
                eng->copy((engine::Reg)3, (engine::Reg)5);
            } else {
                mpz_class numA = mulm(mpz_class(2), addm(aE, dE));
                mpz_class denA = subm(aE, dE), invDenA;
                { int r = invm(denA, invDenA); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class A = mulm(numA, invDenA);
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);

                mpz_t zA24; mpz_init_set(zA24, A24.get_mpz_t()); eng->set_mpz((engine::Reg)6, zA24); mpz_clear(zA24);
                eng->set_multiplicand((engine::Reg)12, (engine::Reg)6);

                std::ostringstream s2r; s2r << "[ECM] Curve " << (c+1) << "/" << curves
                                           << " | Resuming Stage2 at prime-index " << s2_idx << "/" << primesS2_v.size();
                std::cout << s2r.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(s2r.str());
            }

            eng->set_multiplicand((engine::Reg)13, (engine::Reg)4);
            eng->set_multiplicand((engine::Reg)14, (engine::Reg)5);

            const uint32_t MAX_S2_CHUNK_BITS = 262144;
            auto u64_bitlen = [&](uint64_t v)->uint32_t { uint32_t n = 0; do { ++n; v >>= 1; } while (v); return n; };

            uint64_t approx_total_bits = 0;
            for (uint64_t q : primesS2_v) approx_total_bits += (uint64_t)u64_bitlen(q);

            auto t2_0 = high_resolution_clock::now();
            auto last2_save = t2_0, last2_ui = t2_0;
            double saved_et2 = resume_stage2 ? s2_et : 0.0;

            uint64_t done_bits_est = 0;
            for (uint32_t ii = 0; ii < std::min<uint32_t>(s2_idx, (uint32_t)primesS2_v.size()); ++ii) {
                uint64_t v = primesS2_v[ii];
                do { ++done_bits_est; v >>= 1; } while (v);
            }
            uint64_t last2_done_bits = done_bits_est;
            double ema_ips_stage2 = 0.0;

            auto normalize_for_next_chunk = [&](mpz_class& xOut)->int {
                mpz_t tx, tz;
                mpz_init(tx); mpz_init(tz);
                eng->get_mpz(tx, (engine::Reg)0);
                eng->get_mpz(tz, (engine::Reg)1);
                mpz_class Xs(tx), Zs(tz);
                mpz_clear(tx); mpz_clear(tz);

                mpz_class gz; mpz_gcd(gz.get_mpz_t(), Zs.get_mpz_t(), N.get_mpz_t());
                if (gz > 1 && gz < N) {
                    std::cout << "[ECM] Curve " << (c+1) << "/" << curves << " | factor=" << gz.get_str() << std::endl;
                    result_factor = gz; result_status = "found";
                    return 1;
                }

                mpz_class invz;
                if (invm(Zs, invz) != 0) return -1;
                xOut = mulm(Xs, invz);
                return 0;
            };

            auto hadamard = [&](size_t a, size_t b, size_t sx, size_t dx){ eng->addsub((engine::Reg)sx, (engine::Reg)dx, (engine::Reg)a, (engine::Reg)b); };
            auto hadamard_copy = [&](size_t a, size_t b, size_t sx, size_t dx, size_t sc, size_t dc){ eng->addsub_copy((engine::Reg)sx,(engine::Reg)dx,(engine::Reg)sc,(engine::Reg)dc,(engine::Reg)a,(engine::Reg)b); };
            auto xDBLADD_strict = [&](size_t X1,size_t Z1, size_t X2,size_t Z2){
                hadamard_copy(X1, Z1, 25, 24, 10, 9);
                hadamard(X2, Z2, 8, 7);
                eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);  eng->mul((engine::Reg)9,(engine::Reg)11);
                eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);  eng->mul((engine::Reg)10,(engine::Reg)11);
                hadamard(9, 10, X2, Z2);
                eng->square_mul((engine::Reg)X2);
                eng->square_mul((engine::Reg)Z2); eng->mul((engine::Reg)Z2,(engine::Reg)13);
                eng->square_mul_copy((engine::Reg)25,(engine::Reg)X1);
                eng->square_mul((engine::Reg)24);
                eng->sub_reg((engine::Reg)25,(engine::Reg)24);
                eng->set_multiplicand((engine::Reg)15,(engine::Reg)24);
                eng->mul((engine::Reg)X1,(engine::Reg)15);
                eng->set_multiplicand((engine::Reg)15,(engine::Reg)25);
                eng->mul((engine::Reg)25,(engine::Reg)12);
                eng->add((engine::Reg)25,(engine::Reg)24);
                eng->mul_copy((engine::Reg)25,(engine::Reg)15,(engine::Reg)Z1);
            };

            bool stop_after_chunk = false;
            while (s2_idx < (uint32_t)primesS2_v.size()) {
                mpz_class Echunk(1);
                uint32_t chunk_start = s2_idx;
                uint32_t chunk_end = s2_idx;
                while (chunk_end < (uint32_t)primesS2_v.size()) {
                    mpz_mul_ui(Echunk.get_mpz_t(), Echunk.get_mpz_t(), primesS2_v[chunk_end]);
                    ++chunk_end;
                    if (mpz_sizeinbase(Echunk.get_mpz_t(), 2) >= MAX_S2_CHUNK_BITS && chunk_end > chunk_start) break;
                }

                const uint32_t chunk_bits = (uint32_t)mpz_sizeinbase(Echunk.get_mpz_t(), 2);
                for (uint32_t i = 0; i < chunk_bits; ++i) {
                    const uint32_t bit = chunk_bits - 1 - i;
                    const int b = mpz_tstbit(Echunk.get_mpz_t(), bit) ? 1 : 0;
                    if (b == 0) xDBLADD_strict(0,1, 2,3);
                    else        xDBLADD_strict(2,3, 0,1);

                    auto now2 = high_resolution_clock::now();
                    if (duration_cast<milliseconds>(now2 - last2_ui).count() >= progress_interval_ms || i + 1 == chunk_bits) {
                        const uint64_t done_u = done_bits_est + i + 1;
                        const double done = double(done_u);
                        const double total = double(std::max<uint64_t>(1ULL, approx_total_bits));
                        const double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                        const double avg_ips = done / std::max(1e-9, elapsed);
                        const double dt_ui = duration<double>(now2 - last2_ui).count();
                        const double dd_ui = double(done_u - last2_done_bits);
                        const double inst_ips = (dt_ui > 1e-9 && dd_ui >= 0.0) ? (dd_ui / dt_ui) : avg_ips;
                        if (ema_ips_stage2 <= 0.0) ema_ips_stage2 = inst_ips;
                        else ema_ips_stage2 = 0.75 * ema_ips_stage2 + 0.25 * inst_ips;
                        const double eta_ips = (ema_ips_stage2 > 0.0) ? ema_ips_stage2 : avg_ips;
                        const double eta = (total > done && eta_ips > 0.0) ? (total - done) / eta_ips : 0.0;
                        std::ostringstream line;
                        line << "[ECM] Curve " << (c+1) << "/" << curves
                             << " | Stage2 " << std::fixed << std::setprecision(1) << (100.0 * done / total) << "%"
                             << " | primes " << (chunk_start + 1) << "-" << chunk_end << "/" << primesS2_v.size()
                             << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                             << " | ETA " << fmt_hms(eta);
                        ecm_print_progress_line(line.str());
                        last2_done_bits = done_u;
                        last2_ui = now2;
                    }
                    if (interrupted) stop_after_chunk = true;
                }

                done_bits_est += chunk_bits;
                s2_idx = chunk_end;

                if (s2_idx < (uint32_t)primesS2_v.size()) {
                    mpz_class xNext;
                    int nr = normalize_for_next_chunk(xNext);
                    if (nr == 1) {
                        std::cout << std::endl;
                        curves_tested_for_found = c+1; options.curves_tested_for_found = c+1;
                        write_result(); publish_json(); delete eng; return 0;
                    }
                    if (nr < 0) {
                        std::cout << "\n[ECM] Curve " << (c+1) << ": Stage2 normalization failed, retrying\n";
                        delete eng; continue;
                    }
                    mpz_t xNextTmp; mpz_init_set(xNextTmp, xNext.get_mpz_t());
                    eng->set_mpz((engine::Reg)4, xNextTmp);
                    mpz_clear(xNextTmp);
                    eng->set((engine::Reg)5, 1u);
                    eng->set((engine::Reg)0, 1u);
                    eng->set((engine::Reg)1, 0u);
                    eng->copy((engine::Reg)2, (engine::Reg)4);
                    eng->copy((engine::Reg)3, (engine::Reg)5);
                    eng->set_multiplicand((engine::Reg)13, (engine::Reg)4);
                    eng->set_multiplicand((engine::Reg)14, (engine::Reg)5);
                }

                auto now2 = high_resolution_clock::now();
                if (duration_cast<seconds>(now2 - last2_save).count() >= backup_period || stop_after_chunk || s2_idx == (uint32_t)primesS2_v.size()) {
                    const double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                    save_ckpt2(s2_idx, elapsed, (uint32_t)primesS2_v.size());
                    last2_save = now2;
                }

                if (stop_after_chunk) {
                    std::cout << "\n[ECM] Interrupted at Stage2 curve " << (c+1)
                              << " prime-index " << s2_idx << "/" << primesS2_v.size() << "\n";
                    if (guiServer_) {
                        std::ostringstream oss;
                        oss << "[ECM] Interrupted at Stage2 curve " << (c+1) << " prime-index " << s2_idx << "/" << primesS2_v.size();
                        guiServer_->appendLog(oss.str());
                    }
                    curves_tested_for_found = c+1; options.curves_tested_for_found=(uint32_t)(c+1);
                    write_result(); delete eng; return 0;
                }
            }
            std::cout << std::endl;

            eng->copy((engine::Reg)7,(engine::Reg)1);
            mpz_class Zres = compute_X_with_dots(eng, (engine::Reg)7, N);
            mpz_class gg2 = gcd_with_dots(Zres, N);
            if (gg2 == N) { std::cout<<"[ECM] Curve "<<(c+1)<<": Stage2 gcd=N, retrying\n"; delete eng; continue; }

            std::error_code ec2;
            fs::remove(ckpt2_file, ec2);
            fs::remove(ckpt2_file + ".old", ec2);
            fs::remove(ckpt2_file + ".new", ec2);

            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
            { std::ostringstream s2s; s2s<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 elapsed="<<std::fixed<<std::setprecision(2)<<elapsed2<<" s"; std::cout<<s2s.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2s.str()); }
            bool found2 = (gg2 > 1 && gg2 < N);
            if (found2) {
                bool known = is_known(gg2);
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<(known?" | known factor=":" | factor=")<<gg2.get_str()<<std::endl;
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gg2.get_str(); guiServer_->appendLog(oss.str()); }
                if (!known) { std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec); result_factor=gg2; result_status="found"; curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; }
            }
        }





        std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);
        { std::ostringstream fin; fin<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" done"; std::cout<<fin.str()<<std::endl; if (guiServer_) guiServer_->appendLog(fin.str()); }
        delete eng;
    }

    std::cout<<"[ECM] No factor found"<<std::endl;
    result_status = "not_found";
    curves_tested_for_found = 0;
    options.curves_tested_for_found = 0;
    write_result();
    return 1;
}

} // namespace core
