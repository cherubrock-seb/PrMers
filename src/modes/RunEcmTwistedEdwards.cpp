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

    auto splitmix64_step = [](uint64_t& x)->uint64_t{ x += 0x9E3779B97f4A7C15ULL; uint64_t z=x; z^=z>>30; z*=0xBF58476D1CE4E5B9ULL; z^=z>>27; z*=0x94D049BB133111EBULL; z^=z>>31; return z; };
    auto splitmix64_u64 = [&](uint64_t seed0)->uint64_t{ uint64_t s=seed0; return splitmix64_step(s); };
    auto mix64 = [&](uint64_t seed, uint64_t idx)->uint64_t{ uint64_t x = seed ^ 0x9E3779B97f4A7C15ULL; x ^= (idx+1) * 0xBF58476D1CE4E5B9ULL; x ^= (idx+0x100) * 0x94D049BB133111EBULL; return splitmix64_u64(x); };
    auto rnd_mpz_bits = [&](const mpz_class& N, uint64_t seed0, unsigned bits)->mpz_class{ mpz_class z = 0; uint64_t s = seed0; for (unsigned i=0;i<bits;i+=64){ z <<= 64; z += (unsigned long)splitmix64_step(s); } z %= N; if (z <= 2) z += 3; return z; };
    auto fmt_hms = [&](double s)->string{ uint64_t u=(uint64_t)(s+0.5); uint64_t h=u/3600,m=(u%3600)/60,se=u%60; ostringstream ss; ss<<h<<"h "<<m<<"m "<<se<<"s"; return ss.str(); };

    mpz_class N = (mpz_class(1) << p) - 1;

    bool wrote = false;
    string status = "not_found";
    mpz_class factor = 0;
    uint64_t curves_done = 0;
    string torsion_last = options.notorsion ? string("none") : (options.torsion16 ? string("2x16") : string("2x8"));

    auto write_result = [&](){
        if (wrote) return;
        ofstream jf("ecm_result.json", ios::app);
        ostringstream js;
        js<<"{";
        js<<"\"program\":\"PrMers\",";
        js<<"\"version\":\""<<core::PRMERS_VERSION <<"\",";
        js<<"\"worktype\":\"ECM\",";
        js<<"\"exponent\":"<<p<<",";
        js<<"\"mode\":\"twisted_edwards\",";
        js<<"\"torsion\":\""<<torsion_last<<"\",";
        js<<"\"B1\":"<<B1<<",";
        js<<"\"B2\":"<<B2<<",";
        js<<"\"curves_requested\":"<<curves<<",";
        js<<"\"curves_tested\":"<<(curves_done?curves_done:curves)<<",";
        js<<"\"status\":\""<<status<<"\",";
        js<<"\"factor\":\""<<(factor>0? factor.get_str() : string("0"))<<"\"";
        js<<"}";
        jf<<js.str()<<"\n";
        std::cout<<js.str()<<std::endl;
        wrote = true;
    };

    vector<uint64_t> primesB1_v, primesS2_v;
    {
        uint64_t Pmax = B2 ? B2 : B1;
        vector<char> sieve(Pmax + 1, 1);
        sieve[0]=0;
        if (Pmax >= 1) sieve[1]=0;
        for (uint64_t q=2;q*q<=Pmax;++q) if (sieve[q]) for (uint64_t k=q*q;k<=Pmax;k+=q) sieve[k]=0;
        for (uint64_t q=2;q<=B1;++q) if (sieve[q]) primesB1_v.push_back((uint32_t)q);
        if (B2 > B1) for (uint64_t q=B1+1;q<=B2;++q) if (sieve[q]) primesS2_v.push_back((uint64_t)q);
        std::cout<<"[ECM] Prime counts: B1="<<primesB1_v.size()<<", S2="<<primesS2_v.size()<<std::endl;
    }

    mpz_class K(1);
    for (uint32_t q : primesB1_v) { uint64_t m = q; while (m <= B1 / q) m *= q; mpz_mul_ui(K.get_mpz_t(), K.get_mpz_t(), m); }
    size_t Kbits = mpz_sizeinbase(K.get_mpz_t(),2);

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    uint64_t base_seed = options.seed ? options.seed : (now_ns ^ ((uint64_t)p<<32) ^ B1);
    std::cout << "[ECM] seed=" << base_seed << std::endl;

    static size_t transform_size_once = 0;

    for (uint64_t c = 0; c < curves; ++c)
    {
        if (core::algo::interrupted) { status = "interrupted"; write_result(); return 2; }

        engine* eng = engine::create_gpu(p, static_cast<size_t>(34), static_cast<size_t>(options.device_id), verbose);
        if (!eng) { std::cout<<"[ECM] GPU engine unavailable\n"; write_result(); return 1; }
        if (transform_size_once == 0) { transform_size_once = eng->get_size(); std::ostringstream os; os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id; std::cout<<os.str()<<std::endl; if (guiServer_) guiServer_->appendLog(os.str()); }

        uint64_t curve_seed = forceSeed ? options.curve_seed : mix64(base_seed, c);
        if (forceSeed){ base_seed = curve_seed; }
        options.curve_seed = curve_seed;
        options.base_seed = base_seed;
        std::cout << "[ECM] curve_seed=" << curve_seed << std::endl;

        auto addm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{ mpz_class r=a+b; mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto subm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{ mpz_class r=a-b; mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto mulm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto sqrm = [&](const mpz_class& a)->mpz_class{ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), a.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto invm = [&](const mpz_class& a, mpz_class& inv)->int{
            if (mpz_sgn(a.get_mpz_t())==0) return -1;
            if (mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t())) return 0;
            mpz_class g; mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) { factor = g; status = "found"; curves_done=c+1; return 1; }
            return -1;
        };

        mpz_class aE, dE, X0, Y0;
        string torsion_used = options.notorsion ? string("none") : (options.torsion16 ? string("2x16") : string("2x8"));
        bool built = false;

        auto force_on_curve = [&](mpz_class& aE, mpz_class& dE, const mpz_class& X0, const mpz_class& Y0)->bool{
            mpz_class X2 = sqrm(X0), Y2 = sqrm(Y0), XY2 = mulm(X2, Y2), invXY2;
            int r = invm(XY2, invXY2);
            if (r==1) return false;
            if (r<0) return false;
            dE = mulm(subm(addm(mulm(aE, X2), Y2), mpz_class(1)), invXY2);
            return true;
        };

        if (!options.notorsion)
        {
            bool ok = false;
            for (int tries=0; tries<64 && !ok; ++tries){
                // torsion seed w
                mpz_class w = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xA5A5u) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 128);
                // u1 = 2w - 1
                mpz_class u1 = subm(mulm(mpz_class(2), w), mpz_class(1));
                // u2 = 2w + 1
                mpz_class u2 = addm(mulm(mpz_class(2), w), mpz_class(1));
                mpz_class inv_u1, inv_u2;
                int r1 = invm(u1, inv_u1);
                int r2 = invm(u2, inv_u2);
                if (r1==1 || r2==1) { write_result(); delete eng; return 0; }
                if (r1<0 || r2<0) continue;
                // a = -1
                mpz_class inv_u1_4 = sqrm(sqrm(inv_u1));
                mpz_class u2_4 = sqrm(sqrm(u2));
                aE = subm(N, mpz_class(1));
                // d = (u2/u1)^4
                dE = mulm(inv_u1_4, u2_4);
                X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xB4B4u) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 192);
                Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xC3C3u) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x8BADF00Dull)), 192);
                if (X0==0 || Y0==0) continue;
                if (!force_on_curve(aE, dE, X0, Y0)) { if (factor>0){ write_result(); delete eng; return 0; } continue; }
                ok = true;
            }
            if (!ok){
                // fallback a = -1
                aE = subm(N, mpz_class(1));
                for (int tries=0; tries<256 && !built; ++tries){
                    X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xDEADu) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 256);
                    Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xBEEFull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 256);
                    if (X0==0 || Y0==0) continue;
                    if (!force_on_curve(aE, dE, X0, Y0)) { if (factor>0){ write_result(); delete eng; return 0; } continue; }
                    torsion_used += "-fallback";
                    built = true;
                }
            } else built = true;
        }
        else
        {
            // notorsion a = -1
            aE = subm(N, mpz_class(1));
            for (int tries=0; tries<256 && !built; ++tries){
                X0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xC0FFEEull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x9E37u)), 256);
                Y0 = rnd_mpz_bits(N, curve_seed ^ (static_cast<uint64_t>(0xFACEFEEDull) + static_cast<uint64_t>(tries)*static_cast<uint64_t>(0x94D0u)), 256);
                if (X0==0 || Y0==0) continue;
                if (!force_on_curve(aE, dE, X0, Y0)) { if (factor>0){ write_result(); delete eng; return 0; } continue; }
                built = true;
            }
        }

        torsion_last = torsion_used;
        std::cout<<"[ECM] torsion="<<torsion_used<<std::endl;

        if (!built){ delete eng; continue; }

        {
            mpz_class X2 = sqrm(X0), Y2 = sqrm(Y0);
            mpz_class L = addm(mulm(aE, X2), Y2);
            mpz_class R = addm(mpz_class(1), mulm(dE, mulm(X2, Y2)));
            std::cout<<"[ECM] check_on_curve="<<(subm(L,R)==0? "OK":"FAIL")<<std::endl;
            std::cout<<"[ECM] Compute in Twisted Edwards Mode" <<std::endl;
            if (subm(L,R)!=0){ delete eng; continue; }
        }

        mpz_t za; mpz_init(za); mpz_set(za, aE.get_mpz_t()); eng->set_mpz((engine::Reg)16, za); mpz_clear(za);
        mpz_t zd; mpz_init(zd); mpz_set(zd, dE.get_mpz_t()); eng->set_mpz((engine::Reg)17, zd); mpz_clear(zd);

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
        eng->set((engine::Reg)30, 1u);

        eng->copy((engine::Reg)3,(engine::Reg)6);
        eng->copy((engine::Reg)4,(engine::Reg)7);
        eng->copy((engine::Reg)1,(engine::Reg)30);
        eng->copy((engine::Reg)5,(engine::Reg)9);
        eng->set_multiplicand((engine::Reg)14,(engine::Reg)17);
        eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);
        eng->copy((engine::Reg)17,(engine::Reg)14);
        eng->copy((engine::Reg)16,(engine::Reg)11);
        // Twisted Edwards addition: A=(Y1-X1)(Y2-X2), B=(Y1+X1)(Y2+X2), C=2Z1Z2, D=2dT1T2, X3=(B-A)(C-D), Y3=(B+A)(C+D), T3=(B-A)(B+A), Z3=(C-D)(C+D)
        auto eADD_RP = [&](){
            eng->copy((engine::Reg)18,(engine::Reg)4);            // Y1
            eng->sub_reg((engine::Reg)18,(engine::Reg)3);         // A part: Y1-X1
            eng->copy((engine::Reg)19,(engine::Reg)7);            // Y2
            eng->sub_reg((engine::Reg)19,(engine::Reg)6);         // Y2-X2
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)19);
            eng->mul((engine::Reg)18,(engine::Reg)11);            // A
            eng->copy((engine::Reg)20,(engine::Reg)4);            // Y1
            eng->add((engine::Reg)20,(engine::Reg)3);             // Y1+X1
            eng->copy((engine::Reg)21,(engine::Reg)7);            // Y2
            eng->add((engine::Reg)21,(engine::Reg)6);             // Y2+X2
            eng->set_multiplicand((engine::Reg)12,(engine::Reg)21);
            eng->mul((engine::Reg)20,(engine::Reg)12);            // B
            eng->copy((engine::Reg)22,(engine::Reg)5);            // T1
            eng->set_multiplicand((engine::Reg)13,(engine::Reg)9);//
            eng->mul((engine::Reg)22,(engine::Reg)13);            // T1*T2
            //eng->set_multiplicand((engine::Reg)14,(engine::Reg)17);
            eng->mul((engine::Reg)22,(engine::Reg)17,2u);         // D = 2*d*T1*T2
            eng->copy((engine::Reg)23,(engine::Reg)1);            // Z1
            //eng->set_multiplicand((engine::Reg)15,(engine::Reg)8);
            eng->mul((engine::Reg)23,(engine::Reg)8);         // C = 2*Z1*Z2
            eng->copy((engine::Reg)24,(engine::Reg)20);
            eng->sub_reg((engine::Reg)24,(engine::Reg)18);        // E = B-A
            eng->copy((engine::Reg)25,(engine::Reg)23);
            eng->sub_reg((engine::Reg)25,(engine::Reg)22);        // F = C-D
            eng->copy((engine::Reg)26,(engine::Reg)23);
            eng->add((engine::Reg)26,(engine::Reg)22);            // G = C+D
            eng->copy((engine::Reg)27,(engine::Reg)20);
            eng->add((engine::Reg)27,(engine::Reg)18);            // H = B+A
            eng->copy((engine::Reg)3,(engine::Reg)24);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)25);
            eng->mul((engine::Reg)3,(engine::Reg)11);             // X3 = E*F
            eng->copy((engine::Reg)4,(engine::Reg)26);
            eng->set_multiplicand((engine::Reg)12,(engine::Reg)27);
            eng->mul((engine::Reg)4,(engine::Reg)12);             // Y3 = G*H
            eng->copy((engine::Reg)5,(engine::Reg)24);
            eng->set_multiplicand((engine::Reg)13,(engine::Reg)27);
            eng->mul((engine::Reg)5,(engine::Reg)13);             // T3 = E*H
            eng->copy((engine::Reg)1,(engine::Reg)25);
            eng->set_multiplicand((engine::Reg)14,(engine::Reg)26);
            eng->mul((engine::Reg)1,(engine::Reg)14);             // Z3 = F*G
        };

        // Twisted Edwards doubling: A=X1^2, B=Y1^2, C=2Z1^2, D=a*A, E=(X1+Y1)^2-A-B, F=D+C, G=D-B, H=F, X3=E*(F-C), Y3=H*G, T3=E*G, Z3=(F-C)*H
        auto eDBL_XYTZ = [&](size_t RX,size_t RY,size_t RZ,size_t RT){
            eng->copy((engine::Reg)18,(engine::Reg)RX);
            eng->square_mul((engine::Reg)18);                     // A = X1^2
            eng->copy((engine::Reg)19,(engine::Reg)RY);
            eng->square_mul((engine::Reg)19);                     // B = Y1^2
            eng->copy((engine::Reg)20,(engine::Reg)RZ);
            eng->square_mul((engine::Reg)20);
            eng->add((engine::Reg)20,(engine::Reg)20);            // C = 2*Z1^2
            eng->copy((engine::Reg)21,(engine::Reg)18);
            //eng->set_multiplicand((engine::Reg)11,(engine::Reg)16);
            eng->mul((engine::Reg)21,(engine::Reg)16);            // D = a*A
            eng->copy((engine::Reg)22,(engine::Reg)RX);
            eng->add((engine::Reg)22,(engine::Reg)RY);
            eng->square_mul((engine::Reg)22);
            eng->sub_reg((engine::Reg)22,(engine::Reg)18);
            eng->sub_reg((engine::Reg)22,(engine::Reg)19);        // E = (X1+Y1)^2 - A - B
            eng->copy((engine::Reg)23,(engine::Reg)21);
            eng->add((engine::Reg)23,(engine::Reg)19);            // Gtmp = D + B
            eng->copy((engine::Reg)24,(engine::Reg)23);
            eng->sub_reg((engine::Reg)24,(engine::Reg)20);        // F = Gtmp - C
            eng->copy((engine::Reg)25,(engine::Reg)21);
            eng->sub_reg((engine::Reg)25,(engine::Reg)19);        // H = D - B
            eng->copy((engine::Reg)RX,(engine::Reg)22);
            eng->set_multiplicand((engine::Reg)12,(engine::Reg)24);
            eng->mul((engine::Reg)RX,(engine::Reg)12);            // X3 = E*F
            eng->copy((engine::Reg)RY,(engine::Reg)23);
            eng->set_multiplicand((engine::Reg)13,(engine::Reg)25);
            eng->mul((engine::Reg)RY,(engine::Reg)13);            // Y3 = (D+B)*(D-B)
            eng->copy((engine::Reg)RT,(engine::Reg)22);
            eng->set_multiplicand((engine::Reg)14,(engine::Reg)25);
            eng->mul((engine::Reg)RT,(engine::Reg)14);            // T3 = E*H
            eng->copy((engine::Reg)RZ,(engine::Reg)24);
            eng->set_multiplicand((engine::Reg)15,(engine::Reg)23);
            eng->mul((engine::Reg)RZ,(engine::Reg)15);            // Z3 = F*(D+B)
        };

        auto t0 = high_resolution_clock::now();
        auto last_ui = t0;
        size_t total_steps = (Kbits>=1? Kbits-1 : 0);

        std::cout<<"[ECM] stage1_begin Kbits="<<Kbits<<std::endl;

        for (size_t i = 0; i < total_steps; ++i){
            if (core::algo::interrupted) { status = "interrupted"; curves_done = c+1; write_result(); delete eng; return 2; }
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
                line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 "<<(i+1)<<"/"<<total_steps<<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%) | ETA "<<fmt_hms(eta);
                std::cout<<line.str()<<std::flush; last_ui = now;
            }
        }
        std::cout<<std::endl;

        mpz_class Zacc = compute_X_with_dots(eng, (engine::Reg)5, N);
        mpz_class g = gcd_with_dots(Zacc, N);
        std::cout<<"[ECM] gcd(acc,N)="<<g.get_str()<<std::endl;

        if (g != N && g > 1) {
            bool known = false;
            for (auto &s: options.knownFactors){
                if (s.empty()) continue;
                mpz_class f; if (mpz_set_str(f.get_mpz_t(), s.c_str(), 0) != 0) continue;
                if (f < 0) f = -f;
                if (f > 1 && g == f) { known = true; break; }
            }
            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<(known?" | known factor=":" | factor=")<<g.get_str()<<std::endl;
            if (!known) { factor=g; status="found"; curves_done=c+1; write_result(); delete eng; return 0; }
        }

        curves_done = c+1;
        delete eng;
    }

    std::cout<<"[ECM] No factor found"<<std::endl;
    write_result();
    return 1;
}
}
