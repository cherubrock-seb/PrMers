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
//using core::algo::u64_bits;

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

        // aE = -1 pour toutes les variantes, dE choisi pour que le point soit sur la courbe
        if (!options.notorsion) {
            bool ok = false;
            for (int tries=0; tries<64 && !ok; ++tries){
                mpz_class w = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xA5A5u) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 128);
                mpz_class u1 = subm(mulm(mpz_class(2), w), mpz_class(1));
                mpz_class u2 = addm(mulm(mpz_class(2), w), mpz_class(1));
                mpz_class inv_u1, inv_u2;
                int r1 = invm(u1, inv_u1);
                int r2 = invm(u2, inv_u2);
                if (r1==1 || r2==1) {
                    curves_tested_for_found = c+1;
                    options.curves_tested_for_found = (uint32_t)(c+1);
                    write_result();
                    publish_json();
                    delete eng;
                    return 0;
                }
                if (r1<0 || r2<0) continue;
                mpz_class inv_u1_4 = sqrm(sqrm(inv_u1));
                mpz_class u2_4 = sqrm(sqrm(u2));
                aE = subm(N, mpz_class(1));
                dE = mulm(inv_u1_4, u2_4);
                X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xB4B4u) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 192);
                Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xC3C3u) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x8BADF00Dull)), 192);
                if (X0==0 || Y0==0) continue;
                if (!force_on_curve(aE, dE, X0, Y0)) {
                    if (result_factor>0) {
                        curves_tested_for_found = c+1;
                        options.curves_tested_for_found = (uint32_t)(c+1);
                        write_result();
                        publish_json();
                        delete eng;
                        return 0;
                    }
                    continue;
                }
                ok = true;
            }
            if (!ok) {
                aE = subm(N, mpz_class(1));
                for (int tries=0; tries<256 && !built; ++tries){
                    X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xDEADu) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 256);
                    Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xBEEFull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 256);
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
                    torsion_used += "-fallback";
                    built = true;
                }
            } else built = true;
        } else {
            aE = subm(N, mpz_class(1));
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

        eng->set((engine::Reg)0, 0u);
        eng->set((engine::Reg)1, 1u);
        mpz_t zx; mpz_init(zx); mpz_set(zx, X0.get_mpz_t()); eng->set_mpz((engine::Reg)6, zx); mpz_clear(zx);
        mpz_t zy; mpz_init(zy); mpz_set(zy, Y0.get_mpz_t()); eng->set_mpz((engine::Reg)7, zy); mpz_clear(zy);
        eng->set((engine::Reg)8, 2u);
        eng->set_multiplicand((engine::Reg)15,(engine::Reg)8);
        eng->copy((engine::Reg)8,(engine::Reg)15);

        eng->copy((engine::Reg)9,(engine::Reg)6);
        eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
        eng->mul((engine::Reg)9,(engine::Reg)11);
        eng->set((engine::Reg)1, 1u);

        eng->copy((engine::Reg)3,(engine::Reg)6);
        eng->copy((engine::Reg)4,(engine::Reg)7);
        eng->copy((engine::Reg)5,(engine::Reg)9);
        eng->set_multiplicand((engine::Reg)14,(engine::Reg)17);
        eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);
        eng->copy((engine::Reg)17,(engine::Reg)14);
        eng->copy((engine::Reg)16,(engine::Reg)11);

        auto eADD_RP = [&](){
            eng->addsub(
                (engine::Reg)20, (engine::Reg)18,
                (engine::Reg)4,  (engine::Reg)3
            ); // S1, D1
            eng->addsub(
                (engine::Reg)21, (engine::Reg)19,
                (engine::Reg)7,  (engine::Reg)6
            ); // S2, D2

            eng->set_multiplicand((engine::Reg)11, (engine::Reg)19);
            eng->mul((engine::Reg)18, (engine::Reg)11);   // A

            eng->set_multiplicand((engine::Reg)12, (engine::Reg)21);
            eng->mul((engine::Reg)20, (engine::Reg)12);   // B

            eng->copy((engine::Reg)22, (engine::Reg)5);   // T1
            eng->set_multiplicand((engine::Reg)13, (engine::Reg)9);
            eng->mul((engine::Reg)22, (engine::Reg)13);   // T1*T2
            eng->mul((engine::Reg)22, (engine::Reg)17);   // D = 2*d*T1*T2

            eng->copy((engine::Reg)23, (engine::Reg)1);
            eng->mul((engine::Reg)23, (engine::Reg)8);    // C = 2*Z1*Z2

            eng->copy((engine::Reg)24, (engine::Reg)20);
            eng->sub_reg((engine::Reg)24, (engine::Reg)18); // E = B-A

            eng->copy((engine::Reg)25, (engine::Reg)23);
            eng->sub_reg((engine::Reg)25, (engine::Reg)22); // F = C-D

            eng->copy((engine::Reg)26, (engine::Reg)23);
            eng->add((engine::Reg)26, (engine::Reg)22);     // G = C+D

            eng->copy((engine::Reg)27, (engine::Reg)20);
            eng->add((engine::Reg)27, (engine::Reg)18);     // H = B+A

            eng->copy((engine::Reg)3, (engine::Reg)24);
            eng->set_multiplicand((engine::Reg)11, (engine::Reg)25);
            eng->mul((engine::Reg)3, (engine::Reg)11);      // X3 = E*F

            eng->copy((engine::Reg)4, (engine::Reg)26);
            eng->set_multiplicand((engine::Reg)12, (engine::Reg)27);
            eng->mul((engine::Reg)4, (engine::Reg)12);      // Y3 = G*H

            eng->copy((engine::Reg)5, (engine::Reg)24);
            eng->mul((engine::Reg)5, (engine::Reg)12);      // T3 = E*H

            eng->copy((engine::Reg)1, (engine::Reg)25);
            eng->set_multiplicand((engine::Reg)14, (engine::Reg)26);
            eng->mul((engine::Reg)1, (engine::Reg)14);      // Z3 = F*G
        };

        auto eDBL_XYTZ = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            eng->copy((engine::Reg)18, (engine::Reg)RX);
            eng->square_mul((engine::Reg)18);              // A

            eng->copy((engine::Reg)19, (engine::Reg)RY);
            eng->square_mul((engine::Reg)19);              // B

            eng->copy((engine::Reg)20, (engine::Reg)RZ);
            eng->square_mul((engine::Reg)20, 2u);          // C = 2*Z1^2

            eng->copy((engine::Reg)21, (engine::Reg)0);
            eng->sub_reg((engine::Reg)21, (engine::Reg)18); // D = -A

            eng->copy((engine::Reg)22, (engine::Reg)RX);
            eng->add((engine::Reg)22, (engine::Reg)RY);
            eng->square_mul((engine::Reg)22);
            eng->sub_reg((engine::Reg)22, (engine::Reg)18);
            eng->sub_reg((engine::Reg)22, (engine::Reg)19); // E

            eng->addsub(
                (engine::Reg)23, (engine::Reg)25,
                (engine::Reg)21, (engine::Reg)19
            ); // D+B, D-B

            eng->copy((engine::Reg)24, (engine::Reg)23);
            eng->sub_reg((engine::Reg)24, (engine::Reg)20); // F

            eng->copy((engine::Reg)RX, (engine::Reg)22);
            eng->set_multiplicand((engine::Reg)12, (engine::Reg)24);
            eng->mul((engine::Reg)RX, (engine::Reg)12);     // X3 = E*F

            eng->copy((engine::Reg)RY, (engine::Reg)23);
            eng->set_multiplicand((engine::Reg)13, (engine::Reg)25);
            eng->mul((engine::Reg)RY, (engine::Reg)13);     // Y3 = (D+B)*(D-B)

            eng->copy((engine::Reg)RT, (engine::Reg)22);
            eng->mul((engine::Reg)RT, (engine::Reg)13);     // T3 = E*(D-B)

            eng->copy((engine::Reg)RZ, (engine::Reg)24);
            eng->set_multiplicand((engine::Reg)15, (engine::Reg)23);
            eng->mul((engine::Reg)RZ, (engine::Reg)15);     // Z3 = F*(D+B)
        };

        auto t0 = high_resolution_clock::now();
        auto last_ui = t0;
        size_t total_steps = (Kbits>=1? Kbits-1 : 0);

        std::cout<<"[ECM] stage1_begin Kbits="<<Kbits<<std::endl;

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
