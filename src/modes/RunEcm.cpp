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
#include <unordered_set>
#include <ctime>

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

int App::runECMMarin()
{
    using namespace std;
    using namespace std::chrono;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const uint64_t B1 = options.B1 ? options.B1 : 1000000ULL;
    const uint64_t B2 = options.B2 ? options.B2 : 0ULL;
    uint64_t curves = options.nmax ? options.nmax : (options.K ? options.K : 250);
    const bool verbose = true;
    const bool forceCurveSeed = (options.curve_seed != 0ULL);   // deterministic RNG seed
    const bool forceSigma = !options.sigma.empty();      // force Suyama sigma (curve)
    const uint32_t ui_interval_ms = options.ecm_progress_interval_ms ? options.ecm_progress_interval_ms : 2000;

   // const uint32_t ui_interval_ms = options.ecm_progress_interval_ms ? options.ecm_progress_interval_ms : 2000;
    if (forceCurveSeed || forceSigma) curves = 1ULL;

    auto splitmix64_step = [](uint64_t& x)->uint64_t{ x += 0x9E3779B97f4A7C15ULL; uint64_t z=x; z^=z>>30; z*=0xBF58476D1CE4E5B9ULL; z^=z>>27; z*=0x94D049BB133111EBULL; z^=z>>31; return z; };
    auto splitmix64_u64 = [&](uint64_t seed0)->uint64_t{ uint64_t s=seed0; return splitmix64_step(s); };
    auto mix64 = [&](uint64_t seed, uint64_t idx)->uint64_t{
        uint64_t x = seed ^ 0x9E3779B97f4A7C15ULL;
        x ^= (idx+1) * 0xBF58476D1CE4E5B9ULL;
        x ^= (idx+0x100) * 0x94D049BB133111EBULL;
        return splitmix64_u64(x);
    };
    auto rnd_mpz_bits = [&](const mpz_class& N, uint64_t seed0, unsigned bits)->mpz_class{
        mpz_class z = 0;
        uint64_t s = seed0;
        for (unsigned i=0;i<bits;i+=64){ z <<= 64; z += (unsigned long)splitmix64_step(s); }
        z %= N; if (z <= 2) z += 3; return z;
    };
    auto fmt_hms = [&](double s)->string{ uint64_t u=(uint64_t)(s+0.5); uint64_t h=u/3600,m=(u%3600)/60,se=u%60; ostringstream ss; ss<<h<<"h "<<m<<"m "<<se<<"s"; return ss.str(); };
    //auto mpz_from_u64 = [](uint64_t v)->mpz_class{ mpz_class z; mpz_import(z.get_mpz_t(), 1, 1, sizeof(v), 0, 0, &v); return z; };
    auto u64_bits = [](uint64_t x)->size_t{ if(!x) return 1; size_t n=0; while(x){ ++n; x>>=1; } return n; };

    auto run_start = high_resolution_clock::now();
    mpz_class N_full = (mpz_class(1) << p) - 1;
    mpz_class N = N_full;
    uint32_t mersenne_digits = (uint32_t)mpz_sizeinbase(N_full.get_mpz_t(), 10);
    uint64_t bits_B1 = (uint64_t)u64_bits(B1);
    uint64_t bits_B2 = B2 ? (uint64_t)u64_bits(B2) : 0;

    string mode_name = "";
    string torsion_name = "";
    int pm_effective = -1;
    if (forceSigma || options.notorsion) pm_effective = options.edwards ? 3 : 0;
    else if (options.torsion16) pm_effective = options.edwards ? 4 : 1;
    else pm_effective = options.edwards ? 5 : 2;
    mode_name = ((pm_effective==0||pm_effective==1||pm_effective==2) ? "montgomery" : "edwards--conv-->montgomery");
    if (pm_effective==0 || pm_effective==3) torsion_name = "none";
    else if (pm_effective==1 || pm_effective==4) torsion_name = "16";
    else torsion_name = "8";

    bool wrote_result = false;
    string result_status = "not_found";
    mpz_class result_factor = 0;
    uint64_t curves_tested_for_found = 0;

    auto write_result = [&](){
        if (wrote_result) return;
        double tot = duration<double>(high_resolution_clock::now() - run_start).count();
        uint64_t tested = curves_tested_for_found ? curves_tested_for_found : ((result_status=="found") ? curves_tested_for_found : curves);
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
            mpz_class f; if (mpz_set_str(f.get_mpz_t(), s.c_str(), 0) != 0) continue;
            if (f < 0) f = -f;
            if (f > 1 && g == f) return true;
        }
        return false;
    };

    size_t transform_size_once = 0;
    auto hash64_str = [&](const std::string& s)->uint64_t{
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char ch : s) { h ^= (uint64_t)ch; h *= 1099511628211ULL; }
        return h;
    };
    auto publish_json = [&](){
        vector<string> saved = options.knownFactors;
        if (result_factor > 0) { options.knownFactors.clear(); options.knownFactors.push_back(result_factor.get_str()); }
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
        for (auto& s : kf) {
            if (s.empty()) continue;
            mpz_class f; if (mpz_set_str(f.get_mpz_t(), s.c_str(), 0) != 0) continue;
            if (f < 0) f = -f;
            if (f <= 1) continue;
            mpz_class g; mpz_gcd(g.get_mpz_t(), f.get_mpz_t(), N.get_mpz_t());
            if (g > 1) { unsigned m=0; while (mpz_divisible_p(C.get_mpz_t(), g.get_mpz_t())) { C /= g; ++m; } if (m) acc.push_back({g,m}); }
        }
        if (!acc.empty()) {
            for (auto& kv: acc) while (mpz_divisible_p(N.get_mpz_t(), kv.first.get_mpz_t())) N /= kv.first;
            if (N == 1) { std::cout << "[ECM] Trivial after removing known factors." << std::endl; write_result(); publish_json(); return 0; }
        }
    }

    vector<uint64_t> primesB1_v, primesS2_v;
    {
        const uint64_t Pmax = (B2 > B1) ? B2 : B1;
        const uint64_t root = ecm_isqrt_u64(Pmax);
        const std::vector<uint32_t> basePrimes = ecm_sieve_base_primes((uint32_t)root);

        const uint64_t SEG = 1000000ULL;
        std::vector<uint64_t> segPrimes;
        for (uint64_t lo = 2; lo <= Pmax; ) {
            uint64_t hi = lo + SEG - 1;
            if (hi < lo || hi > Pmax) hi = Pmax;
            ecm_segmented_primes_odd(lo, hi, basePrimes, segPrimes);
            for (uint64_t q : segPrimes) {
                if (q <= B1) primesB1_v.push_back(q);
                else if (q <= B2) primesS2_v.push_back(q);
            }
            if (hi == Pmax) break;
            lo = hi + 1;
        }

        std::cout << "[ECM] Prime counts: B1=" << primesB1_v.size()
                  << ", S2=" << primesS2_v.size() << std::endl;
    }



    std::vector<uint32_t> s2_chunk_ends;
    std::vector<uint64_t> s2_chunk_prefix_iters;
    uint64_t s2_total_iters_precomputed = 0;
    if (B2 > B1 /*&& !primesS2_v.empty()*/) {
        const uint32_t MAX_S2_CHUNK_BITS_PRE = 262144;
        mpz_class EchunkPre(1);
        uint32_t idxPre = 0;
        while (idxPre < (uint32_t)primesS2_v.size()) {
            EchunkPre = 1;
            const uint32_t chunkStartPre = idxPre;
            while (idxPre < (uint32_t)primesS2_v.size()) {
                mpz_mul_ui(EchunkPre.get_mpz_t(), EchunkPre.get_mpz_t(), primesS2_v[idxPre]);
                ++idxPre;
                if (mpz_sizeinbase(EchunkPre.get_mpz_t(), 2) >= MAX_S2_CHUNK_BITS_PRE && idxPre > chunkStartPre) break;
            }
            const uint32_t cb = (uint32_t)mpz_sizeinbase(EchunkPre.get_mpz_t(), 2);
            s2_total_iters_precomputed += cb;
            s2_chunk_ends.push_back(idxPre);
            s2_chunk_prefix_iters.push_back(s2_total_iters_precomputed);
        }
    }

    vector<uint64_t> s1_factors;
    mpz_class K(1);
    for (uint32_t q : primesB1_v) { uint64_t m = q; while (m <= B1 / q) m *= q; s1_factors.push_back(m); mpz_mul_ui(K.get_mpz_t(), K.get_mpz_t(), m); }

    {
        std::ostringstream hdr;
        hdr<<"[ECM] N=M_"<<p<<"  B1="<<B1<<"  B2="<<B2<<"  curves="<<curves<<"\n";
        hdr<<"[ECM] Stage1: prime powers up to B1="<<primesB1_v.size()<<", K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<"\n";
        if (!primesS2_v.empty()) hdr<<"[ECM] Stage2 primes ("<<B1<<","<<B2<<"] count="<<primesS2_v.size()<<"\n"; else hdr<<"[ECM] Stage2: disabled\n";
        std::cout<<hdr.str(); if (guiServer_) guiServer_->appendLog(hdr.str());
    }

    //auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
   // uint64_t base_seed = options.seed ? options.seed : (now_ns ^ ((uint64_t)p<<32) ^ B1);
   // std::cout << "[ECM] seed=" << base_seed << std::endl;

    const int backup_period = options.backup_interval > 0 ? options.backup_interval : 10;
    //const std::string ckpt2 = "ecm2_m_p" + std::to_string(p) + "_curve" + std::to_string(curve_seed) + ".ckpt2";
    for (uint64_t c = 0; c < curves; ++c)
    {
        result_factor = 0;
        result_status = "NF";
        curves_tested_for_found = (uint32_t)(c + 1);
        options.curves_tested_for_found = (uint32_t)(c + 1);

        auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

        uint64_t base_seed = 0;
        if (forceCurveSeed) {
            base_seed = options.curve_seed;
        } else if (options.seed) {
            base_seed = options.seed;
        } else if (forceSigma) {
            base_seed = mix64(hash64_str(options.sigma) ^ (((uint64_t)p) << 32) ^ B1, 0xA5C31E29ULL);
        } else {
            base_seed = (now_ns ^ ((uint64_t)p<<32) ^ B1);
        }

        uint64_t curve_seed = mix64(base_seed, c);
        if (forceCurveSeed) { curve_seed = options.curve_seed; base_seed = curve_seed; }

        const std::string ckpt_file = "ecm_m_"  + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";
        const std::string ckpt2     = "ecm2_m_" + std::to_string(p) + "_c" + std::to_string(c) + ".ckpt";

        engine* eng = engine::create_gpu(p, static_cast<size_t>(32), static_cast<size_t>(options.device_id), verbose);
        if (!eng) { std::cout<<"[ECM] GPU engine unavailable\n"; write_result(); publish_json(); return 1; }
        if (transform_size_once == 0) { transform_size_once = eng->get_size(); std::ostringstream os; os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id; std::cout<<os.str()<<std::endl; if (guiServer_) guiServer_->appendLog(os.str()); }

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
                uint8_t torsion16_flag = (!options.notorsion && options.torsion16) ? 1 : 0;
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

        auto read_ckpt = [&](const std::string& file, uint32_t& ri, uint32_t& rnb, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 1) return -2;
            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
            if (!f.read(reinterpret_cast<char*>(&rnb), sizeof(rnb))) return -2;
            uint64_t rB1 = 0;
            if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et),  sizeof(et)))  return -2;
            uint64_t saved_curve_seed = 0;
            uint8_t  saved_torsion16  = 0;
            if (!f.read(reinterpret_cast<char*>(&saved_curve_seed), sizeof(saved_curve_seed))) return -2;
            if (!f.read(reinterpret_cast<char*>(&saved_torsion16),  sizeof(saved_torsion16)))  return -2;
            uint8_t current_torsion16 = (!options.notorsion && options.torsion16) ? 1 : 0;
            if (saved_curve_seed != curve_seed || saved_torsion16 != current_torsion16) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            if (rnb != mpz_sizeinbase(K.get_mpz_t(),2) || rB1 != B1) return -2;
            return 0;
        };

        auto save_ckpt2 = [&](uint32_t idx, double et, uint32_t cnt_bits){
            const std::string oldf = ckpt2 + ".old", newf = ckpt2 + ".new";
            {
                File f(newf, "wb");
                int version = 4;
                if (!f.write(reinterpret_cast<const char*>(&version),  sizeof(version)))  return -1;
                if (!f.write(reinterpret_cast<const char*>(&p),        sizeof(p)))        return -1;
                if (!f.write(reinterpret_cast<const char*>(&idx),      sizeof(idx)))      return -1;
                if (!f.write(reinterpret_cast<const char*>(&cnt_bits), sizeof(cnt_bits))) return -1;
                if (!f.write(reinterpret_cast<const char*>(&B1),       sizeof(B1)))       return -1;
                if (!f.write(reinterpret_cast<const char*>(&B2),       sizeof(B2)))       return -1;
                uint64_t seed64 = (uint64_t)curve_seed;
                if (!f.write(reinterpret_cast<const char*>(&seed64),   sizeof(seed64)))   return -1;
                uint8_t torsion16_flag = (!options.notorsion && options.torsion16) ? 1 : 0;
                if (!f.write(reinterpret_cast<const char*>(&torsion16_flag), sizeof(torsion16_flag))) return -1;
                if (!f.write(reinterpret_cast<const char*>(&et),       sizeof(et)))       return -1;
                const size_t cksz = eng->get_checkpoint_size();
                std::vector<char> data(cksz);
                if (!eng->get_checkpoint(data)) return -1;
                if (!f.write(data.data(), cksz)) return -1;
                f.write_crc32();
            }
            std::error_code ec;
            fs::remove(ckpt2 + ".old", ec);
            if (fs::exists(ckpt2)) fs::rename(ckpt2, ckpt2 + ".old", ec);
            fs::rename(ckpt2 + ".new", ckpt2, ec);
            fs::remove(ckpt2 + ".new", ec);
            return 0;
        };

        auto read_ckpt2 = [&](const std::string& file, uint32_t& idx, uint32_t& cnt_bits, double& et)->int{
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
                uint8_t current_tor = (!options.notorsion && options.torsion16) ? 1 : 0;
                if (saved_seed != (uint64_t)curve_seed || saved_tor != current_tor) return -2;
            }

            if (b1s != B1 || b2s != B2) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            return 0;
        };

        auto write_gp = [&](const std::string& mode, const std::string& tors, const mpz_class& Nref, uint32_t pe, uint64_t b1e, uint64_t b2e, uint64_t seed_base, uint64_t seed_curve, const mpz_class* sigma_opt, const mpz_class* r_opt, const mpz_class* v_opt, const mpz_class* aE_opt, const mpz_class* dE_opt, const mpz_class& A24_ref, const mpz_class& x0_ref)->void{
            (void) A24_ref;
            (void) Nref;
            std::ofstream gp("lastcurve.gp");
            gp<<"p="<<pe<<"; N=2^p-1;\n";
            gp<<"B1="<<b1e<<"; B2="<<b2e<<";\n";
            gp<<"seed_base="<<seed_base<<"; seed_curve="<<seed_curve<<";\n";
            gp<<"default(parisize, 64*10^6);\n";
            gp<<"modN(x)=Mod(x,N);\n";
            if (mode=="montgomery" && tors=="none") {
                gp<<"sigma="<<(sigma_opt? sigma_opt->get_str() : "0")<<";\n";
                gp<<"u=modN(sigma^2-5); v=modN(4*sigma);\n";
                gp<<"u3=modN(u^3); v3=modN(v^3); t0=modN(4*u3*v);\n";
                gp<<"A=lift(modN(((v-u)^2*(v-u)*(3*u+v))/t0 - 2));\n";
                gp<<"A24=lift(modN((A+2)/4));\n";
                gp<<"x0=lift(modN(u3/v3));\n";
            } else if (mode=="montgomery" && tors=="16") {
                gp<<"r="<<(r_opt? r_opt->get_str() : "0")<<";\n";
                gp<<"A=lift(modN((8*r^4-16*r^3+16*r^2-8*r+1)/(4*r^2)));\n";
                gp<<"A24=lift(modN((A+2)/4));\n";
                gp<<"x0=lift(modN(modN(1)/2 - r^2));\n";
            } else if (mode=="montgomery" && tors=="8") {
                gp<<"v="<<(v_opt? v_opt->get_str() : "0")<<";\n";
                gp<<"A=lift(modN(-((4*v+1)^2+16*v)));\n";
                gp<<"A24=lift(modN((A+2)/4));\n";
                gp<<"x0=lift(modN(4*v+1));\n";
            } else if (mode=="edwards--conv-->montgomery") {
                gp<<"aE="<<(aE_opt? aE_opt->get_str() : "0")<<"; dE="<<(dE_opt? dE_opt->get_str() : "0")<<";\n";
                gp<<"A=lift(modN(2*(aE+dE)/(aE-dE)));\n";
                gp<<"A24=lift(modN((A+2)/4));\n";
                gp<<"x0="<<x0_ref.get_str()<<";\n";
            }
            gp<<"\\print(\"A24=\",A24);\n";
            gp<<"\\print(\"x0=\",x0);\n";
            gp.close();
        };

        const std::string ecm_stage1_resume_save_file =
            "resume_p" + std::to_string(p) + "_ECM_B1_" + std::to_string(B1) + ".save";
        const std::string ecm_stage1_resume_p95_file =
            "resume_p" + std::to_string(p) + "_ECM_B1_" + std::to_string(B1) + ".p95";

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

        auto append_ecm_stage1_resume_line = [&](uint64_t curve_idx,
                                                const mpz_class& A24v,
                                                const mpz_class& xAffRaw,
                                                const mpz_class* sigmaForP95)->void {
            mpz_class xAff = xAffRaw;
            mpz_mod(xAff.get_mpz_t(), xAff.get_mpz_t(), N.get_mpz_t());
            if (xAff < 0) xAff += N;

            mpz_class Aresume;
            mpz_mul_ui(Aresume.get_mpz_t(), A24v.get_mpz_t(), 4UL);
            Aresume -= 2;
            mpz_mod(Aresume.get_mpz_t(), Aresume.get_mpz_t(), N.get_mpz_t());
            if (Aresume < 0) Aresume += N;

            const std::string ecmResumeProgram = std::string("PrMers ") + core::PRMERS_VERSION;
            const std::string ecmResumeTime = ecm_now_string();
            const std::string ecmResumeWho = options.user;

            {
                mpz_class chk;
                mpz_set_ui(chk.get_mpz_t(), (unsigned long)B1);
                chk *= mpz_class(mpz_fdiv_ui(Aresume.get_mpz_t(), CHKSUMMOD));
                chk *= mpz_class(mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD));
                chk *= mpz_class(mpz_fdiv_ui(xAff.get_mpz_t(), CHKSUMMOD));
                const uint32_t chk_u = (uint32_t)mpz_fdiv_ui(chk.get_mpz_t(), CHKSUMMOD);

                std::ofstream out(ecm_stage1_resume_save_file, std::ios::out | std::ios::app);
                if (!out) {
                    std::ostringstream oss;
                    oss << "[ECM] Warning: cannot append Stage1 resume to '" << ecm_stage1_resume_save_file << "'";
                    std::cerr << oss.str() << std::endl;
                    if (guiServer_) guiServer_->appendLog(oss.str());
                } else {
                    out << "METHOD=ECM; B1=" << B1
                        << "; N=" << N.get_str()
                        << "; X=0x" << xAff.get_str(16)
                        << "; A=" << Aresume.get_str()
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

                mpz_class chk;
                mpz_set_ui(chk.get_mpz_t(), (unsigned long)B1);
                chk *= mpz_class(mpz_fdiv_ui(sigma.get_mpz_t(), CHKSUMMOD));
                chk *= mpz_class(mpz_fdiv_ui(N.get_mpz_t(), CHKSUMMOD));
                chk *= mpz_class(mpz_fdiv_ui(xAff.get_mpz_t(), CHKSUMMOD));
                const uint32_t chk_u = (uint32_t)mpz_fdiv_ui(chk.get_mpz_t(), CHKSUMMOD);

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
                        << "; X=0x" << xAff.get_str(16)
                        << "; CHECKSUM=" << chk_u
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
            else oss << " (Prime95 export skipped: curve family is A-based/custom, no GMP-ECM SIGMA line)";
            std::cout << oss.str() << std::endl;
            if (guiServer_) guiServer_->appendLog(oss.str());
        };

        uint32_t s2_idx = 0, s2_cnt = 0; double s2_et = 0.0;
        bool resume_stage2 = false; { int rr2 = read_ckpt2(ckpt2, s2_idx, s2_cnt, s2_et); if (rr2 < 0) rr2 = read_ckpt2(ckpt2 + ".old", s2_idx, s2_cnt, s2_et); resume_stage2 = (rr2 == 0); }

        std::cout << "[ECM] curve_seed=" << curve_seed << std::endl;
        options.curve_seed = curve_seed;
        options.base_seed = base_seed;
        auto addm = [&](mpz_class a, mpz_class b)->mpz_class{ mpz_class r=a+b; r%=N; if (r<0) r+=N; return r; };
        auto subm = [&](mpz_class a, mpz_class b)->mpz_class{ mpz_class r=a-b; r%=N; if (r<0) r+=N; return r; };
        auto mulm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto sqrm = [&](const mpz_class& a)->mpz_class{ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), a.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };

        auto invm = [&](const mpz_class& a, mpz_class& inv)->int{
            if (mpz_sgn(a.get_mpz_t())==0) return -1;
            if (mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t())) return 0;
            mpz_class g; mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) {
                bool known = is_known(g);
                std::cout<<"[ECM] "<<(known?"known factor=":"factor=")<<g.get_str()<<std::endl;
                if (!known) {
                    options.knownFactors.push_back(g.get_str());
                    result_factor = g; result_status = "found";
                    curves_tested_for_found = (uint32_t)(c+1); options.curves_tested_for_found = (uint32_t)(c+1);
                } else {
                    result_factor = 0; result_status = "NF";
                    curves_tested_for_found = (uint32_t)(c+1); options.curves_tested_for_found = (uint32_t)(c+1);
                }
                return 1;
            }
            return -1;
        };

        mpz_class A24, x0;
        mpz_class sigma_resume; bool have_sigma_resume = false;

        if (resume_stage2) {
            if (pm_effective==0 || pm_effective==1 || pm_effective==2) mode_name="montgomery"; else mode_name="edwards--conv-->montgomery";
            if (pm_effective==0 || pm_effective==3) torsion_name="none"; else if (pm_effective==1 || pm_effective==4) torsion_name="16"; else torsion_name="8";
        }

        auto hadamard = [&](size_t a, size_t b, size_t s, size_t d){
            eng->addsub((engine::Reg)s, (engine::Reg)d, (engine::Reg)a, (engine::Reg)b);
        };
        auto hadamard_copy = [&](size_t a, size_t b, size_t s, size_t d, size_t s_copy, size_t d_copy){
            eng->addsub_copy((engine::Reg)s,(engine::Reg)d,(engine::Reg)s_copy,(engine::Reg)d_copy,
                            (engine::Reg)a,(engine::Reg)b);
        };

        // xDBLADD_strict: xADD for (X2,Z2), then xDBL for (X1,Z1), using hadamard_copy to avoid copies
        auto xDBLADD_strict = [&](size_t X1,size_t Z1, size_t X2,size_t Z2){
            // xADD part
            hadamard_copy(X1, Z1, 25, 24, 10, 9);                    // 25=S1, 24=D1, 10=S1_copy, 9=D1_copy
            hadamard(X2, Z2, 8, 7);                                  // 8=S2, 7=D2
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);   // *S2
            eng->mul((engine::Reg)9,(engine::Reg)11);                // 9 = t1 = D1_copy * S2
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);   // *D2
            eng->mul((engine::Reg)10,(engine::Reg)11);               // 10 = t2 = S1_copy * D2
            hadamard(9, 10, X2, Z2);                                 // X2=t1+t2, Z2=t1−t2
            eng->square_mul((engine::Reg)X2);                        // X2=(t1+t2)^2
            //eng->mul((engine::Reg)X2,(engine::Reg)14);               // X2=k14*...
            eng->square_mul((engine::Reg)Z2);                        // Z2=(t1−t2)^2
            eng->mul((engine::Reg)Z2,(engine::Reg)13);               // Z2=k13*...

            // xDBL part (U,V,E with fused square+copy; no extra copy for Z1)
            eng->square_mul_copy((engine::Reg)25,(engine::Reg)X1);   // 25=U=(X1+Z1)^2, and X1=U
            eng->square_mul((engine::Reg)24);                        // 24=V=(X1−Z1)^2
            eng->sub_reg((engine::Reg)25,(engine::Reg)24);           // 25=E=U−V
            eng->set_multiplicand((engine::Reg)15,(engine::Reg)24);  // *V
            eng->mul((engine::Reg)X1,(engine::Reg)15);               // X1=U*V
            eng->set_multiplicand((engine::Reg)15,(engine::Reg)25);  // multiplicand=E
            //eng->mul((engine::Reg)25,(engine::Reg)12);               // 25=A24*E
            //eng->add((engine::Reg)25,(engine::Reg)24);               // 25=A24*E + V
            eng->mul_add((engine::Reg)25,(engine::Reg)12,(engine::Reg)24);
            eng->mul_copy((engine::Reg)25,(engine::Reg)15,(engine::Reg)Z1); // Z1=(A24*E+V)*E
        };

        // xDBLADD_strict: xADD for (X2,Z2), then xDBL for (X1,Z1), using hadamard_copy to avoid copies
        auto xDBLADD_strict_s2 = [&](size_t X1,size_t Z1, size_t X2,size_t Z2){
            hadamard_copy(X1, Z1, 25, 24, 10, 9);
            hadamard(X2, Z2, 8, 7);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);
            eng->mul((engine::Reg)9,(engine::Reg)11);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
            eng->mul((engine::Reg)10,(engine::Reg)11);
            hadamard(9, 10, X2, Z2);
            eng->square_mul((engine::Reg)X2);
            eng->mul((engine::Reg)X2,(engine::Reg)14);
            eng->square_mul((engine::Reg)Z2);
            eng->mul((engine::Reg)Z2,(engine::Reg)13);

            eng->square_mul_copy((engine::Reg)25,(engine::Reg)X1);
            eng->square_mul((engine::Reg)24);
            eng->sub_reg((engine::Reg)25,(engine::Reg)24);
            eng->set_multiplicand((engine::Reg)15,(engine::Reg)24);
            eng->mul((engine::Reg)X1,(engine::Reg)15);
            eng->set_multiplicand((engine::Reg)15,(engine::Reg)25);
            eng->mul_add((engine::Reg)25,(engine::Reg)12,(engine::Reg)24);
            eng->mul_copy((engine::Reg)25,(engine::Reg)15,(engine::Reg)Z1);
        };
        
        auto xDBLADD_strict2 = [&](size_t X1,size_t Z1, size_t X2,size_t Z2){
            hadamard_copy(X1, Z1, 25, 24, 10, 9);
            hadamard(X2, Z2, 8, 7);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);
            eng->mul((engine::Reg)9,(engine::Reg)11);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
            eng->mul((engine::Reg)10,(engine::Reg)11);
            hadamard(9, 10, X2, Z2);
            eng->square_mul((engine::Reg)X2);
            eng->mul((engine::Reg)X2,(engine::Reg)14);
            eng->square_mul((engine::Reg)Z2);
            eng->mul((engine::Reg)Z2,(engine::Reg)13);
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

        if (!resume_stage2)
        {
            int picked_mode = -1;
            if (forceSigma || options.notorsion) picked_mode = options.edwards ? 3 : 0;
            else if (options.torsion16) picked_mode = options.edwards ? 4 : 1;
            else picked_mode = options.edwards ? 5 : 2;

            if (picked_mode == 0)
            {
                mode_name="montgomery"; torsion_name="none";

                mpz_class sigma_mpz;
                if (forceCurveSeed){
                    curve_seed = options.curve_seed;
                }

                if (forceSigma) {
                    if (mpz_set_str(sigma_mpz.get_mpz_t(), options.sigma.c_str(), 0) != 0) {
                        std::cerr << "[ECM] Invalid -sigma value: " << options.sigma << std::endl;
                        delete eng;
                        return 0;
                    }
                } else {
                    sigma_mpz = rnd_mpz_bits(N, curve_seed, 64);
                }

                options.sigma_hex = sigma_mpz.get_str(16);
                sigma_resume = sigma_mpz; have_sigma_resume = true;
                mpz_class u = subm(sqrm(sigma_mpz), mpz_class(5));
                mpz_class v = (mpz_class(4) * sigma_mpz) % N;
                mpz_class g; mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t());
                if (g > 1 && g < N) {
                    bool known = is_known(g);
                    std::cout<<"[ECM] "<<(known?"known factor=":"factor=")<<g.get_str()<<std::endl;
                    if (!known) { options.knownFactors.push_back(g.get_str()); result_factor=g; result_status="found"; }
                    else { result_factor=0; result_status="NF"; }
                    curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);
                    write_result(); publish_json(); delete eng; continue;
                }

                mpz_class t0 = mulm(mpz_class(4), mulm(mulm(sqrm(u), u), v));
                mpz_class invt; { int r = invm(t0, invt);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }

                mpz_class tnum = mulm(sqrm(subm(v,u)), subm(v,u));
                mpz_class Araw = mulm(tnum, addm(mulm(mpz_class(3),u), v));
                Araw = mulm(Araw, invt);
                mpz_class A = subm(Araw, mpz_class(2));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                mpz_class u3 = mulm(sqrm(u), u);
                mpz_class v3 = mulm(sqrm(v), v);
                mpz_class invv3;
                {
                    int r = invm(v3, invv3);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                x0 = mulm(u3, invv3);
                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | montgomery | torsion=none | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<curve_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
            }
            else if (picked_mode == 1)
            {
                mode_name="montgomery"; torsion_name="16";
                auto ec_add = [&](const mpz_class& x1, const mpz_class& y1, const mpz_class& x2, const mpz_class& y2, mpz_class& xr, mpz_class& yr)->int{
                    if (x1==x2 && (y1+ y2)%N==0) return -1;
                    mpz_class num = subm(y2, y1);
                    mpz_class den = subm(x2, x1);
                    mpz_class inv; int r = invm(den, inv);
                    if (r==1) return 1;
                    if (r<0) return -1;
                    mpz_class lam = mulm(num, inv);
                    xr = subm(subm(sqrm(lam), x1), x2);
                    yr = subm(mulm(lam, subm(x1, xr)), y1);
                    return 0;
                };
                auto ec_dbl = [&](const mpz_class& x1, const mpz_class& y1, mpz_class& xr, mpz_class& yr)->int{
                    mpz_class num = addm(mulm(mpz_class(3), sqrm(x1)), mpz_class(4));
                    mpz_class den = mulm(mpz_class(2), y1);
                    mpz_class inv; int r = invm(den, inv);
                    if (r==1) return 1;
                    if (r<0) return -1;
                    mpz_class lam = mulm(num, inv);
                    xr = subm(sqrm(lam), mulm(mpz_class(2), x1));
                    yr = subm(mulm(lam, subm(x1, xr)), y1);
                    return 0;
                };
                auto ec_mul = [&](uint64_t k, mpz_class x, mpz_class y, mpz_class& xr, mpz_class& yr)->int{
                    bool init=false; mpz_class X=x, Y=y, RX=0, RY=0;
                    for (int i=(int)u64_bits(k)-1;i>=0;--i){
                        if (init){ int r = ec_dbl(RX, RY, RX, RY); if (r) return r; }
                        if ((k>>i)&1ULL){
                            if (!init){ RX=X; RY=Y; init=true; }
                            else { int r = ec_add(RX, RY, X, Y, RX, RY); if (r) return r; }
                        }
                    }
                    xr = RX; yr=RY; return 0;
                };

                if (forceCurveSeed){
                    base_seed = options.curve_seed;
                }
                curve_seed = base_seed;
                uint64_t k = 2 + (mix64(base_seed, c ^ 0xA5A5A5A5ULL) % 64ULL);
                std::stringstream ss;
                ss << std::hex << std::setw(16) << std::setfill('0') << k;
                options.sigma_hex = ss.str();
                mpz_class s = mpz_class(4), t = mpz_class(8);
                int rmul = ec_mul(k, s, t, s, t);
                if (rmul==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                if (rmul<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }

                mpz_class den = subm(s, mpz_class(4));
                mpz_class inv; { int r = invm(den, inv);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                mpz_class alpha = mulm(addm(t, mpz_class(8)), inv);

                mpz_class numr = addm(mpz_class(8), mulm(mpz_class(2), alpha));
                mpz_class denr = subm(mpz_class(8), sqrm(alpha));
                mpz_class invdenr; { int r = invm(denr, invdenr);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                mpz_class rpar = mulm(numr, invdenr);

                mpz_class r2 = sqrm(rpar);
                mpz_class r3 = mulm(r2, rpar);
                mpz_class r4 = sqrm(r2);
                mpz_class A_num = addm(subm(addm(subm(mulm(mpz_class(8), r4), mulm(mpz_class(16), r3)), mulm(mpz_class(16), r2)), mulm(mpz_class(8), rpar)), mpz_class(1));
                mpz_class A_den = mulm(mpz_class(4), r2);
                mpz_class invAden; { int r = invm(A_den, invAden);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                mpz_class A = mulm(A_num, invAden);
                mpz_class inv4; { int r = invm(mpz_class(4), inv4);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                A24 = mulm(addm(A, mpz_class(2)), inv4);

                mpz_class inv2; { int r = invm(mpz_class(2), inv2);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                x0 = subm(inv2, r2);

                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | montgomery | torsion=16 | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
            }
            else if (picked_mode == 2)
            {
                if (forceCurveSeed){
                    curve_seed = options.curve_seed;
                }
                mode_name="montgomery"; torsion_name="8";
                mpz_class a = rnd_mpz_bits(N, curve_seed ^ 0xD1E2C3B4A5968775ULL, 128);
                options.sigma_hex = a.get_str(16);
                mpz_class a2 = sqrm(a);
                mpz_class denv = subm(mulm(mpz_class(48), a2), mpz_class(1));
                mpz_class invdenv; { int r = invm(denv, invdenv);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                mpz_class v = mulm(mulm(mpz_class(4), a2), invdenv);
                mpz_class fourv = mulm(mpz_class(4), v);
                mpz_class one = mpz_class(1);
                mpz_class A = subm(mpz_class(0), addm(sqrm(addm(fourv, one)), mulm(mpz_class(16), v)));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                x0 = addm(mulm(mpz_class(4), v), mpz_class(1));

                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | montgomery | torsion=8 | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
            }
            else if (picked_mode == 3)
            {
                mode_name="edwards--conv-->montgomery"; torsion_name="none";
                mpz_class sigma_mpz;
                if (forceCurveSeed){
                    curve_seed = options.curve_seed;
                }

                if (forceSigma) {
                    if (mpz_set_str(sigma_mpz.get_mpz_t(), options.sigma.c_str(), 0) != 0) {
                        std::cerr << "[ECM] Invalid -sigma value: " << options.sigma << std::endl;
                        delete eng;
                        return 0;
                    }
                } else {
                    sigma_mpz = rnd_mpz_bits(N, curve_seed, 192);
                }

                options.sigma_hex = sigma_mpz.get_str(16);
                sigma_resume = sigma_mpz; have_sigma_resume = true;
                mpz_class u = subm(sqrm(sigma_mpz), mpz_class(5));
                mpz_class v = (mpz_class(4) * sigma_mpz) % N;
                mpz_class g; mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t());
                if (g > 1 && g < N) {
                    bool known = is_known(g);
                    std::cout<<"[ECM] "<<(known?"known factor=":"factor=")<<g.get_str()<<std::endl;
                    if (!known) { options.knownFactors.push_back(g.get_str()); result_factor=g; result_status="found"; }
                    else { result_factor=0; result_status="NF"; }
                    curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);
                    write_result(); publish_json(); delete eng; continue;
                }

                mpz_class t0 = mulm(mpz_class(4), mulm(mulm(sqrm(u), u), v));
                mpz_class invt; { int r = invm(t0, invt);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                mpz_class tnum = mulm(sqrm(subm(v,u)), subm(v,u));
                mpz_class Araw = mulm(tnum, addm(mulm(mpz_class(3),u), v));
                Araw = mulm(Araw, invt);
                mpz_class A = subm(Araw, mpz_class(2));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                mpz_class u3 = mulm(sqrm(u), u);
                mpz_class v3 = mulm(sqrm(v), v);
                mpz_class invv3;
                {
                    int r = invm(v3, invv3);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                x0 = mulm(u3, invv3);
                mpz_class aE = addm(A, mpz_class(2));
                mpz_class dE = subm(A, mpz_class(2));
                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | edwards --conv-->montgomery  | torsion=none | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
            }
            else
            {
                mode_name="edwards--conv-->montgomery"; torsion_name="8";
                if (forceCurveSeed){
                    curve_seed = options.curve_seed;
                }
                mpz_class a = rnd_mpz_bits(N, curve_seed ^ 0xD1E2C3B4A5968775ULL, 128);
                options.sigma_hex = a.get_str(16);
                mpz_class a2 = sqrm(a);
                mpz_class denv = subm(mulm(mpz_class(48), a2), mpz_class(1));
                mpz_class invdenv; { int r = invm(denv, invdenv);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                mpz_class v = mulm(mulm(mpz_class(4), a2), invdenv);
                mpz_class fourv = mulm(mpz_class(4), v);
                mpz_class one = mpz_class(1);
                mpz_class A = subm(mpz_class(0), addm(sqrm(addm(fourv, one)), mulm(mpz_class(16), v)));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4);
                    if (r==1){ curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                    if (r<0){ result_factor=0; result_status="NF"; curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); publish_json(); delete eng; continue; }
                }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                x0 = addm(mulm(mpz_class(4), v), mpz_class(1));
                mpz_class aE = addm(A, mpz_class(2));
                mpz_class dE = subm(A, mpz_class(2));
                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | edwards --conv-->montgomery  | torsion=8 | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
            }

            mpz_t zA24; mpz_init(zA24); mpz_set(zA24, A24.get_mpz_t()); eng->set_mpz((engine::Reg)6, zA24); mpz_clear(zA24);
            eng->set_multiplicand((engine::Reg)12, (engine::Reg)6);

            eng->set((engine::Reg)0, 1u);
            eng->set((engine::Reg)1, 0u);
            eng->set((engine::Reg)3, 1u);
            mpz_t zx0; mpz_init(zx0); mpz_set(zx0, x0.get_mpz_t()); eng->set_mpz((engine::Reg)2, zx0); mpz_clear(zx0);
            mpz_t zxd; mpz_init(zxd); mpz_set(zxd, x0.get_mpz_t()); eng->set_mpz((engine::Reg)4, zxd); mpz_clear(zxd);
            eng->set((engine::Reg)5, 1u);
            eng->set_multiplicand((engine::Reg)13, (engine::Reg)4);
            eng->set_multiplicand((engine::Reg)14, (engine::Reg)5);

            std::ostringstream head2;
            head2<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 start";
            std::cout<<head2.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head2.str());

            uint32_t start_i = 0, nb_ck = 0; double saved_et = 0.0;
            int rr_ck = read_ckpt(ckpt_file, start_i, nb_ck, saved_et);
            if (rr_ck < 0) rr_ck = read_ckpt(ckpt_file + ".old", start_i, nb_ck, saved_et);
            if (rr_ck != 0) { start_i = 0; nb_ck = 0; saved_et = 0.0; }
            auto t0 = high_resolution_clock::now(); auto last_save = t0; auto last_ui = t0;
            size_t total_bits = mpz_sizeinbase(K.get_mpz_t(),2);
            size_t last_ui_done = start_i;
            double ema_ips_stage1 = 0.0;

            for (size_t i = start_i; i < total_bits; ++i){
                size_t bit = total_bits - 1 - i; int b = mpz_tstbit(K.get_mpz_t(), static_cast<mp_bitcnt_t>(bit)) ? 1 : 0;
                if (b==0) { xDBLADD_strict(0,1, 2,3); }
                else      { xDBLADD_strict(2,3, 0,1); }

                auto now = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now - last_ui).count() >= ui_interval_ms || i+1 == total_bits) {
                    const size_t done_u = i + 1;
                    const double done = double(done_u), total = double(total_bits);
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
                    line << "\r[ECM] Curve " << (c+1) << "/" << curves
                        << " | Stage1 " << (i+1) << "/" << total_bits
                        << " (" << std::fixed << std::setprecision(1)
                        << (done * 100.0 / total) << "%)"
                        << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                        << " | ETA " << fmt_hms(eta);
                    std::cout << line.str() << std::flush;
                    last_ui_done = done_u;
                    last_ui = now;
                }

                if (duration_cast<seconds>(now - last_save).count() >= backup_period) { double elapsed = duration<double>(now - t0).count() + saved_et; save_ckpt((uint32_t)(i + 1), elapsed); last_save = now; }
                if (interrupted) { double elapsed = duration<double>(now - t0).count() + saved_et; save_ckpt((uint32_t)(i + 1), elapsed); std::cout<<"\n[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<total_bits<<"\n"; if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<total_bits; guiServer_->appendLog(oss.str()); } curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1); write_result(); delete eng; return 0; }
            }
            std::cout<<std::endl;

            mpz_class Zfin = compute_X_with_dots(eng, (engine::Reg)1, N);
            mpz_class gg = gcd_with_dots(Zfin, N);
            if (gg == N) { std::cout<<"[ECM] Curve "<<(c+1)<<": singular or failure, retrying\n"; delete eng; continue; }
            bool found = (gg > 1 && gg < N);

            double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count();
            { std::ostringstream s1; s1<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 elapsed="<<fixed<<setprecision(2)<<elapsed_stage1<<" s"; std::cout<<s1.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s1.str()); }

            mpz_class xAff = x0;
            if (gg == 1) {
                mpz_class Xv = compute_X_with_dots(eng, (engine::Reg)0, N);
                mpz_class invZ;
                if (invm(Zfin, invZ) == 0) xAff = mulm(Xv, invZ);
            }
            append_ecm_stage1_resume_line(c, A24, xAff, have_sigma_resume ? &sigma_resume : nullptr);

            if (found) {
                bool known = is_known(gg);
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<(known?" | known factor=":" | factor=")<<gg.get_str()<<std::endl;
                std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gg.get_str(); guiServer_->appendLog(oss.str()); }

                std::error_code ec0;
                fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0);
                fs::remove(ckpt2, ec0); fs::remove(ckpt2 + ".old", ec0); fs::remove(ckpt2 + ".new", ec0);

                curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);

                if (!known) {
                    options.knownFactors.push_back(gg.get_str());
                    result_factor=gg; result_status="found";
                    options.B2 = 0;
                    write_result(); publish_json();
                    options.B2 = B2;
                } else {
                    result_factor=0; result_status="NF";
                    options.B2 = 0;
                    write_result(); publish_json();
                    options.B2 = B2;
                }

                if((B2 <= B1)){
                    delete eng;
                    continue;
                }
            }
            else{
                result_factor=0; result_status="NF";
                curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);
                write_result(); publish_json();
            }
        }
        else
        {
            std::ostringstream s2r; s2r<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Resuming Stage2 at index "<<s2_idx;
            std::cout<<s2r.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2r.str());
        }

        std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
        if (B2 > B1) {
            const uint32_t baseX = 4, baseZ = 5;
            const uint32_t totalS2Primes = (uint32_t)primesS2_v.size();
            const uint32_t MAX_S2_CHUNK_BITS = 262144;
            const uint64_t total_s2_iters = std::max<uint64_t>(1ULL, s2_total_iters_precomputed);

            if (!resume_stage2) {
                mpz_t tx, tz;
                mpz_init(tx); mpz_init(tz);
                eng->get_mpz(tx, (engine::Reg)0);
                eng->get_mpz(tz, (engine::Reg)1);
                mpz_class Xs(tx), Zs(tz);
                mpz_clear(tx); mpz_clear(tz);

                mpz_class gz; mpz_gcd(gz.get_mpz_t(), Zs.get_mpz_t(), N.get_mpz_t());
                if (gz > 1 && gz < N) {
                    bool known = is_known(gz);
                    std::cout << "[ECM] Curve " << (c+1) << "/" << curves << (known ? " | known factor=" : " | factor=") << gz.get_str() << std::endl;
                    if (!known) { options.knownFactors.push_back(gz.get_str()); result_factor = gz; result_status = "found"; }
                    else { result_factor = 0; result_status = "NF"; }
                    curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);
                }

                mpz_class Qx = Xs, Qz = Zs;
                mpz_class invz;
                if (invm(Zs, invz) == 0) { Qx = mulm(Xs, invz); Qz = 1; }

                mpz_t tQx; mpz_init_set(tQx, Qx.get_mpz_t());
                mpz_t tQz; mpz_init_set(tQz, Qz.get_mpz_t());
                eng->set_mpz((engine::Reg)4, tQx);
                eng->set_mpz((engine::Reg)5, tQz);
                mpz_clear(tQx); mpz_clear(tQz);

                eng->set((engine::Reg)0, 1u);
                eng->set((engine::Reg)1, 0u);
                eng->copy((engine::Reg)2, (engine::Reg)4);
                eng->copy((engine::Reg)3, (engine::Reg)5);
                eng->set_multiplicand((engine::Reg)13, (engine::Reg)4);
                eng->set_multiplicand((engine::Reg)14, (engine::Reg)5);
            } else {
                std::ostringstream s2r; s2r << "[ECM] Curve " << (c+1) << "/" << curves
                                        << " | Resuming Stage2 at prime-index " << s2_idx << "/" << primesS2_v.size();
                std::cout << s2r.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(s2r.str());
                eng->set_multiplicand((engine::Reg)13, (engine::Reg)4);
                eng->set_multiplicand((engine::Reg)14, (engine::Reg)5);
            }

            auto t2_0 = high_resolution_clock::now();
            auto last2_save = t2_0, last2_ui = t2_0;
            double saved_et2 = resume_stage2 ? s2_et : 0.0;

            uint64_t done_bits_est = 0;
            for (size_t ci = 0; ci < s2_chunk_ends.size(); ++ci) {
                if (s2_chunk_ends[ci] <= s2_idx) done_bits_est = s2_chunk_prefix_iters[ci];
                else break;
            }
            uint64_t last2_ui_done = done_bits_est;
            double ema_ips_stage2 = 0.0;

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
                    if (b == 0) xDBLADD_strict_s2(0,1, 2,3);
                    else        xDBLADD_strict_s2(2,3, 0,1);

                    auto now2 = high_resolution_clock::now();
                    if (duration_cast<milliseconds>(now2 - last2_ui).count() >= ui_interval_ms || i + 1 == chunk_bits) {
                        const uint64_t done_u = done_bits_est + i + 1;
                        const double done = double(done_u);
                        const double total = double(total_s2_iters);
                        const double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                        const double avg_ips = done / std::max(1e-9, elapsed);
                        const double dt_ui = duration<double>(now2 - last2_ui).count();
                        const double dd_ui = double(done_u - last2_ui_done);
                        const double inst_ips = (dt_ui > 1e-9 && dd_ui >= 0.0) ? (dd_ui / dt_ui) : avg_ips;
                        if (ema_ips_stage2 <= 0.0) ema_ips_stage2 = inst_ips;
                        else ema_ips_stage2 = 0.75 * ema_ips_stage2 + 0.25 * inst_ips;
                        const double eta_ips = (ema_ips_stage2 > 0.0) ? ema_ips_stage2 : avg_ips;
                        const double eta = (total > done && eta_ips > 0.0) ? (total - done) / eta_ips : 0.0;
                        std::ostringstream line;
                        line << "\r[ECM] Curve " << (c+1) << "/" << curves
                            << " | Stage2 " << std::fixed << std::setprecision(1) << (100.0 * done / total) << "%"
                            << " | primes " << (chunk_start + 1) << "-" << chunk_end << "/" << primesS2_v.size()
                            << " | it/s " << std::fixed << std::setprecision(1) << eta_ips
                            << " | ETA " << fmt_hms(eta);
                        std::cout << line.str() << std::flush;
                        last2_ui_done = done_u;
                        last2_ui = now2;
                    }
                    if (interrupted) stop_after_chunk = true;
                }

                done_bits_est += chunk_bits;
                s2_idx = chunk_end;

                if (s2_idx < (uint32_t)primesS2_v.size()) {
                    mpz_t tx, tz;
                    mpz_init(tx); mpz_init(tz);
                    eng->get_mpz(tx, (engine::Reg)0);
                    eng->get_mpz(tz, (engine::Reg)1);
                    mpz_class Xs(tx), Zs(tz);
                    mpz_clear(tx); mpz_clear(tz);

                    mpz_class gz; mpz_gcd(gz.get_mpz_t(), Zs.get_mpz_t(), N.get_mpz_t());
                    if (gz > 1 && gz < N) {
                        bool known = is_known(gz);
                        std::cout << "[ECM] Curve " << (c+1) << "/" << curves << (known ? " | known factor=" : " | factor=") << gz.get_str() << std::endl;
                        if (!known) { options.knownFactors.push_back(gz.get_str()); result_factor = gz; result_status = "found"; }
                        else { result_factor = 0; result_status = "NF"; }
                        curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);
                    }

                    mpz_t tQx; mpz_init_set(tQx, Xs.get_mpz_t());
                    mpz_t tQz; mpz_init_set(tQz, Zs.get_mpz_t());
                    eng->set_mpz((engine::Reg)4, tQx);
                    eng->set_mpz((engine::Reg)5, tQz);
                    mpz_clear(tQx); mpz_clear(tQz);

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
                    curves_tested_for_found = (uint32_t)(c+1); options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result(); delete eng; return 0;
                }
            }
            std::cout << std::endl;

            eng->copy((engine::Reg)7, (engine::Reg)1);
            mpz_class Zres = compute_X_with_dots(eng, (engine::Reg)7, N);
            mpz_class gg2  = gcd_with_dots(Zres, N);
            if (gg2 == N) { std::cout << "[ECM] Curve " << (c+1) << ": Stage2 gcd=N, retrying\n"; delete eng; continue; }

            std::error_code ec2;
            fs::remove(ckpt2, ec2); fs::remove(ckpt2 + ".old", ec2); fs::remove(ckpt2 + ".new", ec2);

            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
            { std::ostringstream s2s; s2s << "[ECM] Curve " << (c+1) << "/" << curves << " | Stage2 elapsed=" << std::fixed << std::setprecision(2) << elapsed2 << " s"; std::cout << s2s.str() << std::endl; if (guiServer_) guiServer_->appendLog(s2s.str()); }

            bool found2 = (gg2 > 1 && gg2 < N);
            if (found2) {
                bool known = is_known(gg2);
                std::cout << "[ECM] Curve " << (c+1) << "/" << curves << (known ? " | known factor=" : " | factor=") << gg2.get_str() << std::endl;
                if (guiServer_) { std::ostringstream oss; oss << "[ECM] " << (known ? "Known " : "") << "factor: " << gg2.get_str(); guiServer_->appendLog(oss.str()); }

                std::error_code ec;
                fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);

                curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);

                if (!known) {
                    options.knownFactors.push_back(gg2.get_str());
                    result_factor = gg2; result_status = "found";
                    write_result(); publish_json();
                } else {
                    result_factor = 0; result_status = "NF";
                    write_result(); publish_json();
                }
            }
            else{
                result_factor=0; result_status="NF";
                curves_tested_for_found=(uint32_t)(c+1); options.curves_tested_for_found=(uint32_t)(c+1);
                write_result(); publish_json();
            }
        }

        std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);
        { std::ostringstream fin; fin<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" done"; std::cout<<fin.str()<<std::endl; if (guiServer_) guiServer_->appendLog(fin.str()); }
        delete eng;
    }

    if (result_status != "found") {
        std::cout<<"[ECM] No factor found"<<std::endl;
        write_result();
        publish_json();
        return 1;
    }

    return 0;
}