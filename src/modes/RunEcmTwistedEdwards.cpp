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
#include <unordered_set>

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
    const bool forceSeed = (options.curve_seed != 0ULL);
    if (forceSeed) curves = 1ULL;
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

    auto run_start = high_resolution_clock::now();
    uint32_t bits_B1 = u64_bits(B1);
    uint32_t bits_B2 = u64_bits(B2 ? B2 : B1);
    uint64_t mersenne_digits = (uint64_t)mpz_sizeinbase(N.get_mpz_t(),10);

    bool wrote_result = false;
    string mode_name = "twisted_edwards";
    string torsion_name = options.notorsion ? string("none") :
                          (options.torsion16 ? string("16") : string("8"));
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

    vector<uint64_t> primesB1_v, primesS2_v;
    {
        uint64_t Pmax = B2 ? B2 : B1;
        vector<char> sieve(Pmax + 1, 1);
        sieve[0]=0;
        if (Pmax >= 1) sieve[1]=0;
        for (uint64_t q=2;q*q<=Pmax;++q)
            if (sieve[q])
                for (uint64_t k=q*q;k<=Pmax;k+=q) sieve[k]=0;
        for (uint64_t q=2;q<=B1;++q) if (sieve[q]) primesB1_v.push_back((uint32_t)q);
        if (B2 > B1)
            for (uint64_t q=B1+1;q<=B2;++q) if (sieve[q]) primesS2_v.push_back((uint64_t)q);
        std::cout<<"[ECM] Prime counts: B1="<<primesB1_v.size()<<", S2="<<primesS2_v.size()<<std::endl;
    }

    mpz_class K(1);
    for (uint32_t q : primesB1_v) {
        uint64_t m = q;
        while (m <= B1 / q) m *= q;
        mpz_mul_ui(K.get_mpz_t(), K.get_mpz_t(), m);
    }
    size_t Kbits = mpz_sizeinbase(K.get_mpz_t(),2);

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    uint64_t base_seed = options.seed ? options.seed : (now_ns ^ ((uint64_t)p<<32) ^ B1);
    std::cout << "[ECM] seed=" << base_seed << std::endl;

    const int backup_period = options.backup_interval > 0 ? options.backup_interval : 10;

    string torsion_last = torsion_name;

    for (uint64_t c = 0; c < curves; ++c)
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

        std::ostringstream ck;  ck << "ecm_m_"  << p << "_c" << c << ".ckpt";
        std::ostringstream ck2; ck2<< "ecm2_m_" << p << "_c" << c << ".ckpt";
        const std::string ckpt_file = ck.str(), ckpt2 = ck2.str();

        auto save_ckpt = [&](uint32_t i, double et){
            const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
            { File f(newf, "wb"); int version = 1; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; uint32_t nbb = (uint32_t)mpz_sizeinbase(K.get_mpz_t(),2); if (!f.write(reinterpret_cast<const char*>(&nbb), sizeof(nbb))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
            std::error_code ec; fs::remove(ckpt_file + ".old", ec); fs::rename(ckpt_file, ckpt_file + ".old", ec); fs::rename(ckpt_file + ".new", ckpt_file, ec); fs::remove(ckpt_file + ".old", ec);
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
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
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
            { File f(newf, "wb"); int version = 2; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&idx), sizeof(idx))) return; if (!f.write(reinterpret_cast<const char*>(&cnt_bits), sizeof(cnt_bits))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&B2), sizeof(B2))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
            std::error_code ec; fs::remove(ckpt2 + ".old", ec); fs::rename(ckpt2, ckpt2 + ".old", ec); fs::rename(ckpt2 + ".new", ckpt2, ec); fs::remove(ckpt2 + ".old", ec);
        };
        auto read_ckpt2 = [&](const std::string& file, uint32_t& idx, uint32_t& cnt_bits, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 2) return -2;
            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;
            if (!f.read(reinterpret_cast<char*>(&idx), sizeof(idx))) return -2;
            if (!f.read(reinterpret_cast<char*>(&cnt_bits), sizeof(cnt_bits))) return -2;
            uint64_t b1s = 0, b2s = 0;
            if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            if (b1s != B1 || b2s != B2) return -2;
            return 0;
        };

        uint32_t s2_idx = 0, s2_cnt = 0; double s2_et = 0.0;
        bool resume_stage2 = false; { int rr2 = read_ckpt2(ckpt2, s2_idx, s2_cnt, s2_et); if (rr2 < 0) rr2 = read_ckpt2(ckpt2 + ".old", s2_idx, s2_cnt, s2_et); resume_stage2 = (rr2 == 0); }

        uint64_t curve_seed = mix64(base_seed, c);
        if (forceSeed) {
            curve_seed = options.curve_seed;
            base_seed = curve_seed;
        }
        std::cout << "[ECM] curve_seed=" << curve_seed << std::endl;
        options.curve_seed = curve_seed;
        options.base_seed = base_seed;

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
        
        string torsion_used = options.notorsion ? string("none") : string("16");
                             // (options.torsion16 ? string("16") : string("8"));
        bool built = false;

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
        if (!options.notorsion && options.torsion16) {
            bool ok = false;
            for (uint32_t tries = 0; tries < 128 && !ok; ++tries) {
                uint64_t m = (mix64(base_seed, c ^ (0x9E37u + tries)) | 1ULL) % 1000000ULL;
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
        /*
        auto check_invariant = [&](){
            auto Xv = compute_X_with_dots(eng,(engine::Reg)3,N);
            auto Yv = compute_X_with_dots(eng,(engine::Reg)4,N);
            auto Zv = compute_X_with_dots(eng,(engine::Reg)1,N);
            auto Tv = compute_X_with_dots(eng,(engine::Reg)5,N);
            auto lhs = addm(mulm(aE, sqrm(Xv)), sqrm(Yv));
            auto rhs = addm(sqrm(Zv), mulm(dE, sqrm(Tv)));
            auto rel = subm(lhs, rhs);
            if (rel != 0){std::cout << "[ECM] invariant FAIL (a="
                                    << aE << ")\n";}
                                    else{
                        std::cout << "[ECM] check invariant OK (a="
                                    << aE << ")\n";
                                    }
        };*/
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
                <<" | K_bits="<<Kbits<<" | seed="<<curve_seed;
            std::cout<<head.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());
        }

        {
            mpz_class dummyA24(0);
            mpz_class x0_ref = X0;
        }

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
        eng->set_multiplicand((engine::Reg)43,(engine::Reg)16);
        eng->set_multiplicand((engine::Reg)44,(engine::Reg)8);
        eng->set_multiplicand((engine::Reg)45,(engine::Reg)29);
        eng->set_multiplicand((engine::Reg)46,(engine::Reg)9);
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
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)49);  // multiplicand <- T2
            eng->mul((engine::Reg)32,(engine::Reg)11);               // 32 = T1*T2
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

            // --- G = A + B ; H = A - B  (Hadamard)
            hadamard((engine::Reg)RX,(engine::Reg)RY,               // RX,RY non conservés
                    (engine::Reg)23,(engine::Reg)25);              // 23=G, 25=H

            // --- F = G - C
            eng->copy((engine::Reg)24,(engine::Reg)23);             // 24 = G
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
// H = Y1*Y2 - a*X1*X2 (here a=1)
// C = d*T1*T2
// D = Z1
// X3 = E*(D - C)
// Y3 = H*(D + C)
// T3 = E*H
// Z3 = (D - C)*(D + C)
auto eADD_RP_notwist = [&](){
    // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
    eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3); // 34=S1, 35=D1
    eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)7,(engine::Reg)6); // 36=S2, 37=D2

    // 30 = X1*X2
    eng->copy((engine::Reg)30,(engine::Reg)3);
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)6);   // multiplicand <- X2
    eng->mul((engine::Reg)30,(engine::Reg)11);               // 30 = X1*X2

    // 31 = Y1*Y2
    eng->copy((engine::Reg)31,(engine::Reg)4);
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);   // multiplicand <- Y2
    eng->mul((engine::Reg)31,(engine::Reg)11);               // 31 = Y1*Y2

    // sum and H in one shot
    eng->addsub((engine::Reg)20,(engine::Reg)40, (engine::Reg)31,(engine::Reg)30); // 20 = Y1Y2+X1X2, 40 = H

    // 32 = C = d*T1*T2  (use pre-bound 46 = T2+, 45 = d)
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
// eADD_RP_notwist : Twisted Edwards addition (a = 1, Z2 = 1)
// Formulas:
// E = X1*Y2 + Y1*X2
// H = Y1*Y2 - a*X1*X2 (here a=1)
// C = d*T1*T2
// D = Z1
// X3 = E*(D - C)
// Y3 = H*(D + C)
// T3 = E*H
// Z3 = (D - C)*(D + C)
auto eADD_RP_notwist_2 = [&](){
    // Hadamards: S1,D1 = Y1±X1 ; S2,D2 = Y2±X2
    eng->addsub((engine::Reg)34,(engine::Reg)35,  (engine::Reg)4,(engine::Reg)3);          // 34=S1, 35=D1
    eng->addsub((engine::Reg)36,(engine::Reg)37,  (engine::Reg)48,(engine::Reg)47);        // 36=S2, 37=D2

    // 30 = X1*X2  (X2-)
    eng->copy((engine::Reg)30,(engine::Reg)3);
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)47);   // multiplicand <- X2-
    eng->mul((engine::Reg)30,(engine::Reg)11);                // 30 = X1*X2

    // 31 = Y1*Y2  (Y2-)
    eng->copy((engine::Reg)31,(engine::Reg)4);
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)48);   // multiplicand <- Y2-
    eng->mul((engine::Reg)31,(engine::Reg)11);                // 31 = Y1*Y2

    // sum and H in one shot
    eng->addsub((engine::Reg)20,(engine::Reg)40, (engine::Reg)31,(engine::Reg)30); // 20=sum, 40=H

    // 32 = C = d*T1*T2  (T2- then d)
    eng->copy((engine::Reg)32,(engine::Reg)5);               // 32 <- T1
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)49);  // multiplicand <- T2-
    eng->mul((engine::Reg)32,(engine::Reg)11);               // 32 = T1*T2
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)29);  // multiplicand <- d
    eng->mul((engine::Reg)32,(engine::Reg)11);               // 32 = d*T1*T2 = C

    // 42 = D + C, 41 = D - C  (D = Z1)
    eng->addsub((engine::Reg)42,(engine::Reg)41, (engine::Reg)1,(engine::Reg)32);

    // 38 = E = S1*S2 - sum
    eng->copy((engine::Reg)38,(engine::Reg)34);              // 38 <- S1
    eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);  // multiplicand <- S2
    eng->mul((engine::Reg)38,(engine::Reg)11);               // 38 = S1*S2
    eng->sub_reg((engine::Reg)38,(engine::Reg)20);           // 38 -= sum => E

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


        uint32_t start_i = 0, nb_ck = 0; double saved_et = 0.0; (void)read_ckpt(ckpt_file, start_i, nb_ck, saved_et);
        auto t0 = high_resolution_clock::now(); auto last_save = t0; auto last_ui = t0;
        //size_t total_steps = (Kbits>=1? Kbits-1 : 0);

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
        if (naf_len)
        {
            short top = naf_vec[naf_len - 1];
            if (top < 0)
            {
                eng->set_mpz((engine::Reg)3, zXneg);
                eng->set_mpz((engine::Reg)4, zYneg);
                eng->set_mpz((engine::Reg)5, zTneg);
            }
        }
        size_t total_steps = (naf_len>=1? naf_len-1 : 0);
        uint32_t i = 0;
        eng->set_mpz((engine::Reg)6, zXpos);
        eng->set_mpz((engine::Reg)7, zYpos);
        eng->set_mpz((engine::Reg)9, zTpos);
        eng->set_mpz((engine::Reg)47, zXneg);
        eng->set_mpz((engine::Reg)48, zYneg);
        eng->set_mpz((engine::Reg)49, zTneg);
        for (i = 0; i < total_steps; ++i){
            if (core::algo::interrupted) {
                double elapsed = duration<double>(high_resolution_clock::now() - t0).count() + saved_et; save_ckpt((uint32_t)(i + 1), elapsed);
                result_status = "interrupted";
                curves_tested_for_found = c+1;
                options.curves_tested_for_found = (uint32_t)(c+1);
                write_result();
                delete eng;
                return 2;
            }
            if((!options.notorsion && options.torsion16)){
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
                    if((!options.notorsion && options.torsion16)) eADD_RP_notwist(); else eADD_RP();
                } else {
                    //eng->set_mpz((engine::Reg)6, zXneg);
                    //eng->set_mpz((engine::Reg)7, zYneg);
                    //eng->set_mpz((engine::Reg)9, zTneg);
                    if((!options.notorsion && options.torsion16)) eADD_RP_notwist_2(); else eADD_RP_2();
                }
            }

            auto now = high_resolution_clock::now();
            if (duration_cast<milliseconds>(now - last_ui).count() >= 400 || i+1 == total_steps) {
                double done = double(i + 1), total = double(total_steps? total_steps:1);
                double elapsed = duration<double>(now - t0).count() + saved_et;
                double ips = done / std::max(1e-9, elapsed);
                double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                std::ostringstream line;
                line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 "<<(i+1)<<"/"<<total_steps
                    <<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%) | ETA "<<fmt_hms(eta);
                std::cout<<line.str()<<std::flush;
                last_ui = now;
            }
            if (duration_cast<seconds>(now - last_save).count() >= backup_period) {
                double elapsed = duration<double>(now - t0).count() + saved_et;
                save_ckpt((uint32_t)(i + 1), elapsed);
                last_save = now;
            }
        }
        std::cout<<std::endl;
        mpz_clear(zXpos); mpz_clear(zYpos); mpz_clear(zTpos);
        mpz_clear(zXneg); mpz_clear(zYneg); mpz_clear(zTneg);

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

        if (found) {
            bool known = is_known(g);
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves
                     <<(known?" | known factor=":" | factor=")<<g.get_str()<<std::endl;
            std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<g.get_str();
                guiServer_->appendLog(oss.str());
            }
            if (!known) {
                std::error_code ec0; fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0);
                result_factor = g;
                result_status = "found";
                curves_tested_for_found = c+1;
                options.curves_tested_for_found = (uint32_t)(c+1);
                write_result();
                publish_json();
                delete eng;
                return 0;
            }
        }

        if (B2 > B1) {
            mpz_class M(1);
            for (uint64_t q : primesS2_v) mpz_mul_ui(M.get_mpz_t(), M.get_mpz_t(), q);
            uint32_t stage2_bits = (uint32_t)mpz_sizeinbase(M.get_mpz_t(), 2);
            if (resume_stage2 && s2_cnt != stage2_bits) { resume_stage2 = false; s2_idx = 0; s2_et = 0.0; }
            uint32_t start_bit = resume_stage2 ? s2_idx : 0;
            auto t2_0 = high_resolution_clock::now(); auto last2_save = t2_0; auto last2_ui = t2_0; double saved_et2 = s2_et;

            mpz_class Zv = compute_X_with_dots(eng, (engine::Reg)1, N);
            mpz_class Yv = compute_X_with_dots(eng, (engine::Reg)4, N);

            auto addm = [&](mpz_class a, mpz_class b){ mpz_class r=a+b; r%=N; if (r<0) r+=N; return r; };
            auto subm = [&](mpz_class a, mpz_class b){ mpz_class r=a-b; r%=N; if (r<0) r+=N; return r; };
            auto mulm = [&](const mpz_class& a, const mpz_class& b){ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
            auto invm = [&](const mpz_class& a, mpz_class& inv)->int{
                if (mpz_sgn(a.get_mpz_t())==0) return -1;
                if (mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t())) return 0;
                mpz_class g; mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
                if (g > 1 && g < N) { std::cout<<"[ECM] factor="<<g.get_str()<<std::endl; result_factor=g; result_status="found"; return 1; }
                return -1;
            };

            mpz_class den = subm(Zv, Yv), invden;
            { int r = invm(den, invden); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
            mpz_class u = mulm(addm(Zv, Yv), invden);

            // A = 2*(aE+dE)/(aE-dE), A24 = (A+2)/4
            mpz_class numA = mulm(mpz_class(2), addm(aE, dE));
            mpz_class denA = subm(aE, dE), invDenA;
            { int r = invm(denA, invDenA); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
            mpz_class A = mulm(numA, invDenA);
            mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r) { curves_tested_for_found=c+1; options.curves_tested_for_found=c+1; write_result(); publish_json(); delete eng; return 0; } }
            mpz_class A24 = mulm(addm(A, mpz_class(2)), inv4);

            mpz_t zA24; mpz_init_set(zA24, A24.get_mpz_t()); eng->set_mpz((engine::Reg)6, zA24); mpz_clear(zA24);
            eng->set_multiplicand((engine::Reg)12, (engine::Reg)6);

            const uint32_t baseX = 4, baseZ = 5;
            mpz_t zu; mpz_init_set(zu, u.get_mpz_t()); eng->set_mpz((engine::Reg)baseX, zu); mpz_clear(zu);
            eng->set((engine::Reg)baseZ, 1u);

            if (!resume_stage2) {
                eng->set((engine::Reg)0, 1u);
                eng->set((engine::Reg)1, 0u);
                eng->copy((engine::Reg)2, (engine::Reg)baseX);
                eng->copy((engine::Reg)3, (engine::Reg)baseZ);
                eng->set_multiplicand((engine::Reg)13, (engine::Reg)2);
                eng->set_multiplicand((engine::Reg)14, (engine::Reg)3);
            }

            auto hadamard = [&](size_t a, size_t b, size_t s, size_t d){
                eng->addsub((engine::Reg)s, (engine::Reg)d, (engine::Reg)a, (engine::Reg)b);
            };
            auto hadamard_copy = [&](size_t a, size_t b, size_t s, size_t d, size_t sc, size_t dc){
                eng->addsub_copy((engine::Reg)s,(engine::Reg)d,(engine::Reg)sc,(engine::Reg)dc,(engine::Reg)a,(engine::Reg)b);
            };
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

            for (uint32_t i = start_bit; i < stage2_bits; ++i){
                uint32_t bit = stage2_bits - 1 - i;
                int b = mpz_tstbit(M.get_mpz_t(), bit) ? 1 : 0;
                if (b==0) xDBLADD_strict(0,1, 2,3);
                else      xDBLADD_strict(2,3, 0,1);

                auto now2 = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now2 - last2_ui).count() >= 400 || i+1==stage2_bits){
                    double done = double(i+1), total = double(stage2_bits);
                    double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                    double ips = done / std::max(1e-9, elapsed);
                    double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                    std::ostringstream line;
                    line << "\r[ECM] Curve " << (c+1) << "/" << curves
                        << " | Stage2 " << (i+1) << "/" << stage2_bits
                        << " (" << std::fixed << std::setprecision(2) << (total? (done*100.0/total):100.0)
                        << "%) | ETA " << fmt_hms(eta);
                    std::cout << line.str() << std::flush; last2_ui = now2;
                }
                if (duration_cast<seconds>(now2 - last2_save).count() >= backup_period){
                    double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                    save_ckpt2((uint32_t)(i + 1), elapsed, stage2_bits); last2_save = now2;
                }
                if (interrupted){
                    double elapsed = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
                    save_ckpt2((uint32_t)(i+1), elapsed, stage2_bits);
                    std::cout << "\n[ECM] Interrupted at Stage2 curve " << (c+1)
                            << " bit " << (i+1) << "/" << stage2_bits << "\n";
                    if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at Stage2 curve "<<(c+1)<<" bit "<<(i+1)<<"/"<<stage2_bits; guiServer_->appendLog(oss.str()); }
                    curves_tested_for_found=c+1; options.curves_tested_for_found=(uint32_t)(c+1); write_result(); delete eng; return 0;
                }
            }
            std::cout << std::endl;

            eng->copy((engine::Reg)7,(engine::Reg)1);
            mpz_class Zres = compute_X_with_dots(eng, (engine::Reg)7, N);
            mpz_class gg2 = gcd_with_dots(Zres, N);
            if (gg2 == N) { std::cout<<"[ECM] Curve "<<(c+1)<<": Stage2 gcd=N, retrying\n"; delete eng; continue; }

            std::error_code ec2; fs::remove(ckpt2, ec2); fs::remove(ckpt2 + ".old", ec2); fs::remove(ckpt2 + ".new", ec2);
            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
            { std::ostringstream s2s; s2s<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 elapsed="<<std::fixed<<std::setprecision(2)<<elapsed2<<" s"; std::cout<<s2s.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2s.str()); }
            bool found2 = (gg2 > 1 && gg2 < N);
            if (found2) 
            {
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
