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
namespace ecm_local {

struct EC_mod4 {
    struct Pt { mpz_class x, y; bool inf=false; };

    static inline void norm(mpz_class& z, const mpz_class& N) {
        mpz_mod(z.get_mpz_t(), z.get_mpz_t(), N.get_mpz_t());
        if (z < 0) z += N;
    }

    // y^2 = x^3 + 4x - 16  (a=4, b=-16)
    static Pt dbl(const Pt& P, const mpz_class& N) {
        if (P.inf) return P;
        // lambda = (3*x^2 + 4) / (2*y)
        mpz_class num = 3 * P.x * P.x + 4;
        mpz_class den = 2 * P.y, inv;
        norm(num, N); norm(den, N);
        if (mpz_sgn(den.get_mpz_t()) == 0) return Pt{{}, {}, true};        // point d'ordre 2
        if (!mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()))   // non-inversible -> "inf"
            return Pt{{}, {}, true};
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        // x3 = lambda^2 - 2*x1
        mpz_class x3 = (lambda * lambda - 2 * P.x) % N; if (x3 < 0) x3 += N;
        // y3 = lambda*(x1 - x3) - y1
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }

    static Pt add(const Pt& P, const Pt& Q, const mpz_class& N) {
        if (P.inf) return Q;
        if (Q.inf) return P;
        if (P.x == Q.x) {
            // P == ±Q
            mpz_class ysum = (P.y + Q.y) % N; if (ysum < 0) ysum += N;
            if (ysum == 0) return Pt{{}, {}, true};        // P + (-P) = O
            return dbl(P, N);                              // P == Q
        }
        // lambda = (y2 - y1) / (x2 - x1)
        mpz_class num = Q.y - P.y; norm(num, N);
        mpz_class den = Q.x - P.x; norm(den, N);
        mpz_class inv;
        if (!mpz_invert(inv.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()))
            return Pt{{}, {}, true};                       // non-inversible -> "inf"
        mpz_class lambda = (num * inv) % N; if (lambda < 0) lambda += N;

        // x3 = lambda^2 - x1 - x2
        mpz_class x3 = (lambda * lambda - P.x - Q.x) % N; if (x3 < 0) x3 += N;
        // y3 = lambda*(x1 - x3) - y1
        mpz_class y3 = (lambda * (P.x - x3) - P.y) % N; if (y3 < 0) y3 += N;
        return Pt{x3, y3, false};
    }

    // (s,t) = n * (s1,t1) mod N on t^2 = s^3 + 4s - 16
    static void get(uint64_t n, int s1, int t1, const mpz_class& N, mpz_class& s, mpz_class& t) {
        Pt P0, P;
        P0.x = s1; if (s1 < 0) P0.x += N; P0.x %= N;
        P0.y = t1; if (t1 < 0) P0.y += N; P0.y %= N;
        P    = P0;

        // Exclut le bit de poids fort (schéma "R = 2R; if(bit) R += P0")
        int msb = 63 - __builtin_clzll(n);
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

    auto write_gp = [&](const std::string& mode,
                        const std::string& tors,
                        const mpz_class& Nref,
                        uint32_t pe,
                        uint64_t b1e,
                        uint64_t b2e,
                        uint64_t seed_base,
                        uint64_t seed_curve,
                        const mpz_class* sigma_opt,
                        const mpz_class* r_opt,
                        const mpz_class* v_opt,
                        const mpz_class* aE_opt,
                        const mpz_class* dE_opt,
                        const mpz_class& A24_ref,
                        const mpz_class& x0_ref)->void
    {
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
            gp<<"x=u^3; t0=modN(4*x*v);\n";
            gp<<"A=lift(modN(((v-u)^2*(v-u)*(3*u+v))/t0 - 2));\n";
            gp<<"A24=lift(modN((A+2)/4));\n";
            gp<<"x0=lift(modN(((v-u)^2)/(4*u*v)));\n";
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
        } else if (mode=="edwards") {
            gp<<"aE="<<(aE_opt? aE_opt->get_str() : "0")<<"; dE="<<(dE_opt? dE_opt->get_str() : "0")<<";\n";
            gp<<"A=lift(modN(2*(aE+dE)/(aE-dE)));\n";
            gp<<"A24=lift(modN((A+2)/4));\n";
            gp<<"x0="<<x0_ref.get_str()<<";\n";
        }
        gp<<"\\print(\"A24=\",A24);\n";
        gp<<"\\print(\"x0=\",x0);\n";
        gp.close();
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

        engine* eng = engine::create_gpu(p, static_cast<size_t>(34), static_cast<size_t>(options.device_id), verbose);
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
        
        auto sqrQ = [](const mpq_class& z)->mpq_class { return z*z; };
        auto pow4Q = [&](const mpq_class& z)->mpq_class { mpq_class t=sqrQ(z); return t*t; };

        auto mpq_to_mod = [&](const mpq_class& q, mpz_class& out)->int{
            mpz_class num(q.get_num()), den(q.get_den());
            mpz_class inv;
            int r = invm(den, inv);
            if (r==1) return 1;    
            if (r<0)  return -1;   
            out = mulm(num, inv);
            return 0;
        };

        // Rational EC over Q: y^2 = x^3 + 4x - 16
        struct QPt { mpq_class x, y; bool inf=false; };
        auto q_add = [&](const QPt& P, const QPt& Q)->QPt{
            if (P.inf) return Q; if (Q.inf) return P;
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
        };
        auto q_mul = [&](QPt P, uint32_t k)->QPt{
            QPt R{{}, {}, true};
            while (k){
                if (k&1u) R = q_add(R,P);
                P = q_add(P,P);
                k >>= 1u;
            }
            return R;
        };

        mpz_class aE, dE, X0, Y0;
        
        string torsion_used = options.notorsion ? string("none") :
                              (options.torsion16 ? string("16") : string("8"));
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
                // m pseudo-aléatoire et non nul
                uint64_t m = (mix64(base_seed, c ^ (0x9E37u + tries)) | 1ULL) % 1000000ULL;
                if (m < 2) m = 2;

                // (s,t) = m*(4,8) sur t^2 = s^3 + 4 s - 16  (mod N)
                mpz_class s, t;
                // EC_modular<4>::get(m, 4, 8, N, s, t);
                ecm_local::EC_mod4::get(m, 4, 8, N, s, t);


                // a = 1
                aE = mpz_class(1);

                // Helpers locaux mod N
                auto inv_or_gcd = [&](const mpz_class& den, mpz_class& inv)->bool {
                    int r = invm(den, inv);                   // ton invm() retourne 1 si facteur trouvé
                    if (r == 1) {
                        curves_tested_for_found = c+1; options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result(); publish_json(); delete eng; return false; // exit courant
                    }
                    return (r == 0);
                };

                // alpha = (t+8)/(s-4)
                mpz_class inv, alpha, alpha2, r, x, y;
                mpz_class den = subm(s, mpz_class(4));
                if (!inv_or_gcd(den, inv)) return 0;
                alpha  = mulm(addm(t, mpz_class(8)), inv);
                alpha2 = sqrm(alpha);

                // r = (8 + 2 alpha) / (8 - alpha^2)
                den = subm(mpz_class(8), alpha2);
                if (!inv_or_gcd(den, inv)) return 0;
                r = mulm(addm(mpz_class(8), mulm(mpz_class(2), alpha)), inv);

                // t1 = (2r - 1)^2
                mpz_class two_r_minus1 = subm(mulm(mpz_class(2), r), mpz_class(1));
                mpz_class t1 = sqrm(two_r_minus1);

                // d = (8 r^2 - 8 r + 1) / ( (2r - 1)^4 )
                mpz_class numD = addm(subm(mulm(mpz_class(8), sqrm(r)), mulm(mpz_class(8), r)), mpz_class(1));
                mpz_class t1sq = sqrm(t1);
                if (!inv_or_gcd(t1sq, inv)) return 0;
                dE = mulm(numD, inv);

                // x = ((8 - alpha^2) * (2 r^2 - 1)) / (2 s - alpha^2 + 4)
                mpz_class numX = mulm(subm(mpz_class(8), alpha2), subm(mulm(mpz_class(2), sqrm(r)), mpz_class(1)));
                mpz_class denX = addm(subm(mulm(mpz_class(2), s), alpha2), mpz_class(4));
                if (!inv_or_gcd(denX, inv)) return 0;
                X0 = mulm(numX, inv);

                // y = ( (2r - 1)^2 ) / (4 r - 3)
                mpz_class denY = subm(mulm(mpz_class(4), r), mpz_class(3));
                if (!inv_or_gcd(denY, inv)) return 0;
                Y0 = mulm(t1, inv);

                if (X0 == 0 || Y0 == 0) continue;

                // Vérif point sur a=1 : X^2 + Y^2 = 1 + d X^2 Y^2
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
        auto check_invariant = [&](){
            auto Xv = compute_X_with_dots(eng,(engine::Reg)3,N);
            auto Yv = compute_X_with_dots(eng,(engine::Reg)4,N);
            auto Zv = compute_X_with_dots(eng,(engine::Reg)1,N);
            auto Tv = compute_X_with_dots(eng,(engine::Reg)5,N);
            auto lhs = addm(mulm(aE, sqrm(Xv)), sqrm(Yv));     // a*X^2 + Y^2
            auto rhs = addm(sqrm(Zv), mulm(dE, sqrm(Tv)));     // Z^2 + d*T^2
            auto rel = subm(lhs, rhs);
            if (rel != 0){std::cout << "[ECM] invariant FAIL (a="
                                    << aE << ")\n";}
                                    else{
                        std::cout << "[ECM] check invariant OK (a="
                                    << aE << ")\n";
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
                <<" | K_bits="<<Kbits<<" | seed="<<base_seed;
            std::cout<<head.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());
        }

        {
            mpz_class dummyA24(0);
            mpz_class x0_ref = X0;
            write_gp("edwards", torsion_used, N, p, B1, B2,
                     base_seed, curve_seed,
                     nullptr, nullptr, nullptr,
                     &aE, &dE, dummyA24, x0_ref);
        }

        mpz_t za; mpz_init(za); mpz_set(za, aE.get_mpz_t()); eng->set_mpz((engine::Reg)16, za); mpz_clear(za);
        static const mpz_class TWO = 2;
        mpz_class two_dE = mulm(dE, TWO);
        mpz_t tmp; mpz_init_set(tmp, two_dE.get_mpz_t());
        eng->set_mpz((engine::Reg)17, tmp);
        mpz_clear(tmp);
        // ➕ add this: keep plain d in reg 29 for the generic addition
        mpz_t zd; mpz_init(zd); mpz_set(zd, dE.get_mpz_t());
        eng->set_mpz((engine::Reg)29, zd);   // 29 = d
        mpz_clear(zd);
        eng->set((engine::Reg)0, 0u);
        eng->set((engine::Reg)1, 1u);
        mpz_t zx; mpz_init(zx); mpz_set(zx, X0.get_mpz_t()); eng->set_mpz((engine::Reg)6, zx); mpz_clear(zx);
        mpz_t zy; mpz_init(zy); mpz_set(zy, Y0.get_mpz_t()); eng->set_mpz((engine::Reg)7, zy); mpz_clear(zy);
        eng->set((engine::Reg)8, 2u);              // constante 2 en registre 8
        eng->copy((engine::Reg)9,(engine::Reg)6);  // 9 = X0
        eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
        eng->mul((engine::Reg)9,(engine::Reg)11);  // 9 = T2 = X0*Y0
        eng->set((engine::Reg)1, 1u);              // Z1 = 1
        eng->copy((engine::Reg)3,(engine::Reg)6);  // X1 = X0
        eng->copy((engine::Reg)4,(engine::Reg)7);  // Y1 = Y0
        eng->copy((engine::Reg)5,(engine::Reg)9);  // T1 = T0



        // Generic extended twisted-Edwards addition (works for any a):
        // A = X1*X2
        // B = Y1*Y2
        // C = d*T1*T2
        // D = Z1*Z2  (Z2=1 here)
        // E = (X1+Y1)*(X2+Y2) - A - B
        // F = D - C
        // G = D + C
        // H = B - a*A
        // X3 = E*F
        // Y3 = G*H
        // T3 = E*H
        // Z3 = F*G
        auto eADD_RP = [&](){
            // A = X1*X2
            eng->copy((engine::Reg)30,(engine::Reg)3);                 // 30 = X1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)6);     // *X2
            eng->mul((engine::Reg)30,(engine::Reg)11);                 // 30 = A

            // B = Y1*Y2
            eng->copy((engine::Reg)31,(engine::Reg)4);                 // 31 = Y1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);     // *Y2
            eng->mul((engine::Reg)31,(engine::Reg)11);                 // 31 = B

            // C = d*T1*T2
            eng->copy((engine::Reg)32,(engine::Reg)5);                 // 32 = T1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)9);     // *T2
            eng->mul((engine::Reg)32,(engine::Reg)11);                 // 32 = T1*T2
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)29);    // *d
            eng->mul((engine::Reg)32,(engine::Reg)11);                 // 32 = C

            // D = Z1*Z2, but Z2=1 => D = Z1
            eng->copy((engine::Reg)33,(engine::Reg)1);                 // 33 = D

            // S1 = X1+Y1, S2 = X2+Y2
            eng->addsub((engine::Reg)34,(engine::Reg)35,               // 34=S1, 35=Y1-X1 (unused)
                        (engine::Reg)4,(engine::Reg)3);                // Y1, X1
            eng->addsub((engine::Reg)36,(engine::Reg)37,               // 36=S2, 37=Y2-X2 (unused)
                        (engine::Reg)7,(engine::Reg)6);                // Y2, X2

            // E = (X1+Y1)*(X2+Y2) - A - B
            eng->copy((engine::Reg)38,(engine::Reg)34);                // 38 = S1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)36);    // *S2
            eng->mul((engine::Reg)38,(engine::Reg)11);                 // 38 = S1*S2
            eng->sub_reg((engine::Reg)38,(engine::Reg)30);             // -A
            eng->sub_reg((engine::Reg)38,(engine::Reg)31);             // -B  => 38 = E

            // F = D - C ; G = D + C
            eng->copy((engine::Reg)41,(engine::Reg)33);                // 41 = D
            eng->sub_reg((engine::Reg)41,(engine::Reg)32);             // 41 = F
            eng->copy((engine::Reg)42,(engine::Reg)33);                // 42 = D
            eng->add    ((engine::Reg)42,(engine::Reg)32);             // 42 = G

            // H = B - a*A
            eng->copy((engine::Reg)39,(engine::Reg)30);                // 39 = A
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);    // *a
            eng->mul((engine::Reg)39,(engine::Reg)11);                 // 39 = a*A
            eng->copy((engine::Reg)40,(engine::Reg)31);                // 40 = B
            eng->sub_reg((engine::Reg)40,(engine::Reg)39);             // 40 = H

            // X3 = E*F
            eng->copy((engine::Reg)3,(engine::Reg)38);                 // X3 = E
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)41);    // *F
            eng->mul((engine::Reg)3,(engine::Reg)11);

            // Y3 = G*H
            eng->copy((engine::Reg)4,(engine::Reg)42);                 // Y3 = G
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);    // *H
            eng->mul((engine::Reg)4,(engine::Reg)11);

            // T3 = E*H
            eng->copy((engine::Reg)5,(engine::Reg)38);                 // T3 = E
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)40);    // *H
            eng->mul((engine::Reg)5,(engine::Reg)11);

            // Z3 = F*G
            eng->copy((engine::Reg)1,(engine::Reg)41);                 // Z3 = F
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)42);    // *G
            eng->mul((engine::Reg)1,(engine::Reg)11);
        };





        // A=X1^2, B=Y1^2, C=2*Z1^2, D=a*A,
        // E=2*T1*Z1      (== 2*X1*Y1)
        // G=D+B, F=G-C, H=D-B,
        // X3=E*F, Y3=G*H, T3=E*H, Z3=F*G
        auto eDBL_XYTZ = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            // A = X1^2
            eng->copy((engine::Reg)18,(engine::Reg)RX);
            eng->square_mul((engine::Reg)18);                       // 18=A

            // B = Y1^2
            eng->copy((engine::Reg)19,(engine::Reg)RY);
            eng->square_mul((engine::Reg)19);                       // 19=B

            // C = 2*Z1^2
            eng->copy((engine::Reg)20,(engine::Reg)RZ);
            eng->square_mul((engine::Reg)20);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);  // *2
            eng->mul((engine::Reg)20,(engine::Reg)11);              // 20=C

            // D = a*A
            eng->copy((engine::Reg)21,(engine::Reg)18);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)16); // *a
            eng->mul((engine::Reg)21,(engine::Reg)11);              // 21=D

            // E = 2*T1*Z1  (== 2*X1*Y1)
            eng->copy((engine::Reg)22,(engine::Reg)RT);             // 22 = T1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)RZ); // *Z1
            eng->mul((engine::Reg)22,(engine::Reg)11);              // 22 = T1*Z1
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);  // *2
            eng->mul((engine::Reg)22,(engine::Reg)11);              // 22 = E

            // G = D + B ; H = D - B ; F = G - C
            eng->copy((engine::Reg)23,(engine::Reg)21); eng->add    ((engine::Reg)23,(engine::Reg)19); // 23=G
            eng->copy((engine::Reg)25,(engine::Reg)21); eng->sub_reg((engine::Reg)25,(engine::Reg)19); // 25=H
            eng->copy((engine::Reg)24,(engine::Reg)23); eng->sub_reg((engine::Reg)24,(engine::Reg)20); // 24=F

            // X3 = E * F
            eng->copy((engine::Reg)RX,(engine::Reg)22);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)24);
            eng->mul((engine::Reg)RX,(engine::Reg)11);

            // Y3 = G * H
            eng->copy((engine::Reg)RY,(engine::Reg)23);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25);
            eng->mul((engine::Reg)RY,(engine::Reg)11);

            // T3 = E * H
            eng->copy((engine::Reg)RT,(engine::Reg)22);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25);
            eng->mul((engine::Reg)RT,(engine::Reg)11);

            // Z3 = F * G
            eng->copy((engine::Reg)RZ,(engine::Reg)24);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)23);
            eng->mul((engine::Reg)RZ,(engine::Reg)11);
        };


        auto t0 = high_resolution_clock::now();
        auto last_ui = t0;
        size_t total_steps = (Kbits>=1? Kbits-1 : 0);

        std::cout<<"[ECM] stage1_begin Kbits="<<Kbits<<std::endl;
        //check_invariant();
        for (size_t i = 0; i < total_steps; ++i){
            if (core::algo::interrupted) {
                result_status = "interrupted";
                curves_tested_for_found = c+1;
                options.curves_tested_for_found = (uint32_t)(c+1);
                write_result();
                delete eng;
                return 2;
            }
            size_t bit = Kbits - 2 - i;
            int b = mpz_tstbit(K.get_mpz_t(), static_cast<mp_bitcnt_t>(bit)) ? 1 : 0;
            eDBL_XYTZ(3,4,1,5);
            if (b) eADD_RP();
            //if (i == 0) { check_invariant(); }


            auto now = high_resolution_clock::now();
            if (duration_cast<milliseconds>(now - last_ui).count() >= 400 || i+1 == total_steps) {
                double done = double(i + 1), total = double(total_steps? total_steps:1);
                double elapsed = duration<double>(now - t0).count();
                double ips = done / std::max(1e-9, elapsed);
                double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                std::ostringstream line;
                line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 "<<(i+1)<<"/"<<total_steps
                    <<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%) | ETA "<<fmt_hms(eta);
                std::cout<<line.str()<<std::flush;
                last_ui = now;
            }
        }
        //check_invariant();
        std::cout<<std::endl;

        mpz_class Zacc = compute_X_with_dots(eng, (engine::Reg)5, N);
        mpz_class g = gcd_with_dots(Zacc, N);
        if (g == N) {
            std::cout<<"[ECM] Curve "<<(c+1)<<": singular or failure, retrying\n";
            delete eng;
            continue;
        }

        bool found = (g > 1 && g < N);

        double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count();
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
            std::cout << "[ECM] Last curve written to 'lastcurve.gp' (PARI/GP script)." << std::endl;
            std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
            if (guiServer_) {
                std::ostringstream oss;
                oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<g.get_str();
                guiServer_->appendLog(oss.str());
            }
            if (!known) {
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
