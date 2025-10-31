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

int App::runECMMarin()
{
    using namespace std;
    using namespace std::chrono;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const uint64_t B1 = options.B1 ? options.B1 : 1000000ULL;
    const uint64_t B2 = options.B2 ? options.B2 : 0ULL;
    uint64_t curves = options.nmax ? options.nmax : (options.K ? options.K : 250);
    const bool verbose = true;
    const bool forceSigma = (options.sigma != 0ULL);
    if (forceSigma) curves = 1ULL;

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
    auto mpz_from_u64 = [](uint64_t v)->mpz_class{ mpz_class z; mpz_import(z.get_mpz_t(), 1, 1, sizeof(v), 0, 0, &v); return z; };
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
    mode_name = ((pm_effective==0||pm_effective==1||pm_effective==2) ? "montgomery" : "edwards");
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

    vector<uint32_t> primesB1_v, primesS2_v;
    {
        uint64_t Pmax = B2 ? B2 : B1;
        vector<char> sieve(Pmax + 1, 1);
        sieve[0]=0; if (Pmax >= 1) sieve[1]=0;
        for (uint64_t q=2;q*q<=Pmax;++q) if (sieve[q]) for (uint64_t k=q*q;k<=Pmax;k+=q) sieve[k]=0;
        for (uint64_t q=2;q<=B1;++q) if (sieve[q]) primesB1_v.push_back((uint32_t)q);
        if (B2 > B1) for (uint64_t q=B1+1;q<=B2;++q) if (sieve[q]) primesS2_v.push_back((uint32_t)q);
        std::cout<<"[ECM] Prime counts: B1="<<primesB1_v.size()<<", S2="<<primesS2_v.size()<<std::endl;
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

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    uint64_t base_seed = options.seed ? options.seed : (now_ns ^ ((uint64_t)p<<32) ^ B1);
    std::cout << "[ECM] seed=" << base_seed << std::endl;

    const int backup_period = options.backup_interval > 0 ? options.backup_interval : 10;

    for (uint64_t c = 0; c < curves; ++c)
    {
        engine* eng = engine::create_gpu(p, static_cast<size_t>(18), static_cast<size_t>(options.device_id), verbose);
        if (!eng) { std::cout<<"[ECM] GPU engine unavailable\n"; write_result(); publish_json(); return 1; }
        if (transform_size_once == 0) { transform_size_once = eng->get_size(); std::ostringstream os; os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id; std::cout<<os.str()<<std::endl; if (guiServer_) guiServer_->appendLog(os.str()); }

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

        auto save_ckpt2 = [&](uint32_t idx, double et){
            const std::string oldf = ckpt2 + ".old", newf = ckpt2 + ".new";
            { File f(newf, "wb"); int version = 2; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&idx), sizeof(idx))) return; uint32_t cnt = (uint32_t)primesS2_v.size(); if (!f.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&B2), sizeof(B2))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
            std::error_code ec; fs::remove(ckpt2 + ".old", ec); fs::rename(ckpt2, ckpt2 + ".old", ec); fs::rename(ckpt2 + ".new", ckpt2, ec); fs::remove(ckpt2 + ".old", ec);
        };
        auto read_ckpt2 = [&](const std::string& file, uint32_t& idx, uint32_t& cnt, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0;
            if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 2) return -2;
            uint32_t rp = 0;
            if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;
            if (!f.read(reinterpret_cast<char*>(&idx), sizeof(idx))) return -2;
            if (!f.read(reinterpret_cast<char*>(&cnt), sizeof(cnt))) return -2;
            uint64_t b1s = 0, b2s = 0;
            if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            if (cnt != primesS2_v.size() || b1s != B1 || b2s != B2) return -2;
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

        uint32_t s2_idx = 0, s2_cnt = 0; double s2_et = 0.0;
        bool resume_stage2 = false; { int rr2 = read_ckpt2(ckpt2, s2_idx, s2_cnt, s2_et); if (rr2 < 0) rr2 = read_ckpt2(ckpt2 + ".old", s2_idx, s2_cnt, s2_et); resume_stage2 = (rr2 == 0); }

        uint64_t curve_seed = mix64(base_seed, c);
        std::cout << "[ECM] curve_seed=" << curve_seed << std::endl;

        auto addm = [&](mpz_class a, mpz_class b)->mpz_class{ mpz_class r=a+b; r%=N; if (r<0) r+=N; return r; };
        auto subm = [&](mpz_class a, mpz_class b)->mpz_class{ mpz_class r=a-b; r%=N; if (r<0) r+=N; return r; };
        auto mulm = [&](const mpz_class& a, const mpz_class& b)->mpz_class{ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto sqrm = [&](const mpz_class& a)->mpz_class{ mpz_class r; mpz_mul(r.get_mpz_t(), a.get_mpz_t(), a.get_mpz_t()); mpz_mod(r.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t()); if (r<0) r+=N; return r; };
        auto invm = [&](const mpz_class& a, mpz_class& inv)->int{
            if (mpz_sgn(a.get_mpz_t())==0) return -1;
            if (mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t())) return 0;
            mpz_class g; mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) { std::cout<<"[ECM] factor="<<g.get_str()<<std::endl; result_factor = g; result_status = "found"; return 1; }
            return -1;
        };

        mpz_class A24, x0;

        if (resume_stage2) {
            if (pm_effective==0 || pm_effective==1 || pm_effective==2) mode_name="montgomery"; else mode_name="edwards";
            if (pm_effective==0 || pm_effective==3) torsion_name="none"; else if (pm_effective==1 || pm_effective==4) torsion_name="16"; else torsion_name="8";
        }

        auto xDBL = [&](size_t X1,size_t Z1,size_t X2,size_t Z2){
            eng->copy((engine::Reg)7,(engine::Reg)X1);
            eng->add((engine::Reg)7,(engine::Reg)Z1);
            eng->copy((engine::Reg)8,(engine::Reg)7);
            eng->square_mul((engine::Reg)8);
            eng->copy((engine::Reg)9,(engine::Reg)X1);
            eng->sub_reg((engine::Reg)9,(engine::Reg)Z1);
            eng->copy((engine::Reg)10,(engine::Reg)9);
            eng->square_mul((engine::Reg)10);
            eng->copy((engine::Reg)X2,(engine::Reg)8);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)10);
            eng->mul((engine::Reg)X2,(engine::Reg)11);
            eng->copy((engine::Reg)9,(engine::Reg)8);
            eng->sub_reg((engine::Reg)9,(engine::Reg)10);
            eng->copy((engine::Reg)Z2,(engine::Reg)9);
            eng->mul((engine::Reg)Z2,(engine::Reg)12);
            eng->add((engine::Reg)Z2,(engine::Reg)10);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)9);
            eng->mul((engine::Reg)Z2,(engine::Reg)11);
        };
        auto xADD = [&](size_t X1,size_t Z1,size_t X2,size_t Z2,size_t X3,size_t Z3){
            eng->copy((engine::Reg)7,(engine::Reg)X1);
            eng->sub_reg((engine::Reg)7,(engine::Reg)Z1);
            eng->copy((engine::Reg)8,(engine::Reg)X2);
            eng->add((engine::Reg)8,(engine::Reg)Z2);
            eng->copy((engine::Reg)9,(engine::Reg)7);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)8);
            eng->mul((engine::Reg)9,(engine::Reg)11);
            eng->copy((engine::Reg)8,(engine::Reg)X1);
            eng->add((engine::Reg)8,(engine::Reg)Z1);
            eng->copy((engine::Reg)7,(engine::Reg)X2);
            eng->sub_reg((engine::Reg)7,(engine::Reg)Z2);
            eng->copy((engine::Reg)10,(engine::Reg)8);
            eng->set_multiplicand((engine::Reg)11,(engine::Reg)7);
            eng->mul((engine::Reg)10,(engine::Reg)11);
            eng->copy((engine::Reg)X3,(engine::Reg)9);
            eng->add((engine::Reg)X3,(engine::Reg)10);
            eng->square_mul((engine::Reg)X3);
            eng->mul((engine::Reg)X3,(engine::Reg)14);
            eng->copy((engine::Reg)Z3,(engine::Reg)9);
            eng->sub_reg((engine::Reg)Z3,(engine::Reg)10);
            eng->square_mul((engine::Reg)Z3);
            eng->mul((engine::Reg)Z3,(engine::Reg)13);
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
                if (forceSigma) { sigma_mpz = mpz_from_u64(options.sigma); sigma_mpz %= N; if (sigma_mpz<=2) sigma_mpz+=3; }
                else            { sigma_mpz = rnd_mpz_bits(N, curve_seed, 192); }
                mpz_class u = subm(sqrm(sigma_mpz), mpz_class(5));
                mpz_class v = (mpz_class(4) * sigma_mpz) % N;
                mpz_class g; mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t()); if (g > 1 && g < N) { std::cout<<"[ECM] factor="<<g.get_str()<<std::endl; result_factor=g; result_status="found"; curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; }
                mpz_class t0 = mulm(mpz_class(4), mulm(mulm(sqrm(u), u), v));
                mpz_class invt; { int r = invm(t0, invt); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class tnum = mulm(sqrm(subm(v,u)), subm(v,u));
                mpz_class Araw = mulm(tnum, addm(mulm(mpz_class(3),u), v));
                Araw = mulm(Araw, invt);
                mpz_class A = subm(Araw, mpz_class(2));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                mpz_class den = mulm(mpz_class(4), mulm(u, v)); { int r = invm(den, den); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class num = sqrm(subm(v,u));
                x0 = mulm(num, den);
                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | montgomery | torsion=none | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
                write_gp("montgomery","none", N, p, B1, B2, base_seed, curve_seed, &sigma_mpz, nullptr, nullptr, nullptr, nullptr, A24, x0);
            }
            else if (picked_mode == 1)
            {
                mode_name="montgomery"; torsion_name="16";
                auto ec_add = [&](const mpz_class& x1, const mpz_class& y1, const mpz_class& x2, const mpz_class& y2, mpz_class& xr, mpz_class& yr)->int{
                    if (x1==x2 && (y1+ y2)%N==0) return -1;
                    mpz_class num = subm(y2, y1);
                    mpz_class den = subm(x2, x1);
                    mpz_class inv; int r = invm(den, inv); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); return 1; } if (r<0) { curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); return -1; }
                    mpz_class lam = mulm(num, inv);
                    xr = subm(subm(sqrm(lam), x1), x2);
                    yr = subm(mulm(lam, subm(x1, xr)), y1);
                    return 0;
                };
                auto ec_dbl = [&](const mpz_class& x1, const mpz_class& y1, mpz_class& xr, mpz_class& yr)->int{
                    mpz_class num = addm(mulm(mpz_class(3), sqrm(x1)), mpz_class(4));
                    mpz_class den = mulm(mpz_class(2), y1);
                    mpz_class inv; int r = invm(den, inv); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); return 1; } if (r<0) { curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); return -1; }
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

                uint64_t k = 2 + (mix64(base_seed, c ^ 0xA5A5A5A5ULL) % 64ULL);
                mpz_class s = mpz_class(4), t = mpz_class(8);
                int rmul = ec_mul(k, s, t, s, t); if (rmul==1){ delete eng; publish_json(); return 0; } if (rmul<0){ delete eng; publish_json(); return 0; }

                mpz_class den = subm(s, mpz_class(4));
                mpz_class inv; { int r = invm(den, inv); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class alpha = mulm(addm(t, mpz_class(8)), inv);

                mpz_class numr = addm(mpz_class(8), mulm(mpz_class(2), alpha));
                mpz_class denr = subm(mpz_class(8), sqrm(alpha));
                mpz_class invdenr; { int r = invm(denr, invdenr); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class rpar = mulm(numr, invdenr);

                mpz_class r2 = sqrm(rpar);
                mpz_class r3 = mulm(r2, rpar);
                mpz_class r4 = sqrm(r2);
                mpz_class A_num = addm(subm(addm(subm(mulm(mpz_class(8), r4), mulm(mpz_class(16), r3)), mulm(mpz_class(16), r2)), mulm(mpz_class(8), rpar)), mpz_class(1));
                mpz_class A_den = mulm(mpz_class(4), r2);
                mpz_class invAden; { int r = invm(A_den, invAden); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class A = mulm(A_num, invAden);
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);

                mpz_class inv2; { int r = invm(mpz_class(2), inv2); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                x0 = subm(inv2, r2);

                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | montgomery | torsion=16 | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
                write_gp("montgomery","16", N, p, B1, B2, base_seed, curve_seed, nullptr, &rpar, nullptr, nullptr, nullptr, A24, x0);
            }
            else if (picked_mode == 2)
            {
                mode_name="montgomery"; torsion_name="8";
                mpz_class a = rnd_mpz_bits(N, curve_seed ^ 0xD1E2C3B4A5968775ULL, 128);
                mpz_class a2 = sqrm(a);
                mpz_class denv = subm(mulm(mpz_class(48), a2), mpz_class(1));
                mpz_class invdenv; { int r = invm(denv, invdenv); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class v = mulm(mulm(mpz_class(4), a2), invdenv);
                mpz_class fourv = mulm(mpz_class(4), v);
                mpz_class one = mpz_class(1);
                mpz_class A = subm(mpz_class(0), addm(sqrm(addm(fourv, one)), mulm(mpz_class(16), v)));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                x0 = addm(mulm(mpz_class(4), v), mpz_class(1));

                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | montgomery | torsion=8 | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
                write_gp("montgomery","8", N, p, B1, B2, base_seed, curve_seed, nullptr, nullptr, &v, nullptr, nullptr, A24, x0);
            }
            else if (picked_mode == 3)
            {
                mode_name="edwards"; torsion_name="none";
                mpz_class sigma_mpz;
                if (forceSigma) { sigma_mpz = mpz_from_u64(options.sigma); sigma_mpz %= N; if (sigma_mpz<=2) sigma_mpz+=3; }
                else            { sigma_mpz = rnd_mpz_bits(N, curve_seed, 192); }
                mpz_class u = subm(sqrm(sigma_mpz), mpz_class(5));
                mpz_class v = (mpz_class(4) * sigma_mpz) % N;
                mpz_class g; mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t()); if (g > 1 && g < N) { std::cout<<"[ECM] factor="<<g.get_str()<<std::endl; result_factor=g; result_status="found"; curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; }
                mpz_class t0 = mulm(mpz_class(4), mulm(mulm(sqrm(u), u), v));
                mpz_class invt; { int r = invm(t0, invt); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class tnum = mulm(sqrm(subm(v,u)), subm(v,u));
                mpz_class Araw = mulm(tnum, addm(mulm(mpz_class(3),u), v));
                Araw = mulm(Araw, invt);
                mpz_class A = subm(Araw, mpz_class(2));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                mpz_class den = mulm(mpz_class(4), mulm(u, v)); { int r = invm(den, den); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class num = sqrm(subm(v,u));
                x0 = mulm(num, den);
                mpz_class aE = addm(A, mpz_class(2));
                mpz_class dE = subm(A, mpz_class(2));
                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | edwards | torsion=none | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
                write_gp("edwards","none", N, p, B1, B2, base_seed, curve_seed, nullptr, nullptr, nullptr, &aE, &dE, A24, x0);
            }
            else
            {
                mode_name="edwards"; torsion_name="8";
                mpz_class a = rnd_mpz_bits(N, curve_seed ^ 0xD1E2C3B4A5968775ULL, 128);
                mpz_class a2 = sqrm(a);
                mpz_class denv = subm(mulm(mpz_class(48), a2), mpz_class(1));
                mpz_class invdenv; { int r = invm(denv, invdenv); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                mpz_class v = mulm(mulm(mpz_class(4), a2), invdenv);
                mpz_class fourv = mulm(mpz_class(4), v);
                mpz_class one = mpz_class(1);
                mpz_class A = subm(mpz_class(0), addm(sqrm(addm(fourv, one)), mulm(mpz_class(16), v)));
                mpz_class inv4; { int r = invm(mpz_class(4), inv4); if (r==1){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } if (r<0){ curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; } }
                A24 = mulm(addm(A, mpz_class(2)), inv4);
                x0 = addm(mulm(mpz_class(4), v), mpz_class(1));
                mpz_class aE = addm(A, mpz_class(2));
                mpz_class dE = subm(A, mpz_class(2));
                std::ostringstream head;
                head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | edwards | torsion=8 | K_bits="<<mpz_sizeinbase(K.get_mpz_t(),2)<<" | seed="<<base_seed;
                std::cout<<head.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head.str());
                write_gp("edwards","8", N, p, B1, B2, base_seed, curve_seed, nullptr, nullptr, &v, &aE, &dE, A24, x0);
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

            uint64_t cnt_xdbl=0,cnt_xadd=0,cnt_sqr=0,cnt_mul=0;

            std::ostringstream head2;
            head2<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 start";
            std::cout<<head2.str()<<std::endl; if (guiServer_) guiServer_->appendLog(head2.str());

            uint32_t start_i = 0, nb_ck = 0; double saved_et = 0.0; (void)read_ckpt(ckpt_file, start_i, nb_ck, saved_et);
            auto t0 = high_resolution_clock::now(); auto last_save = t0; auto last_ui = t0;
            size_t total_bits = mpz_sizeinbase(K.get_mpz_t(),2);
            for (size_t i = start_i; i < total_bits; ++i){
                size_t bit = total_bits - 1 - i; int b = mpz_tstbit(K.get_mpz_t(), static_cast<mp_bitcnt_t>(bit)) ? 1 : 0;
                if (b==0) { xADD(2,3,0,1,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); xDBL(0,1,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); }
                else      { xADD(0,1,2,3,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); xDBL(2,3,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); }
                auto now = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now - last_ui).count() >= 400 || i+1 == total_bits) {
                    double done = double(i + 1), total = double(total_bits);
                    double elapsed = duration<double>(now - t0).count() + saved_et;
                    double ips = done / std::max(1e-9, elapsed);
                    double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                    std::ostringstream line;
                    line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 "<<(i+1)<<"/"<<total_bits<<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%) | ETA "<<fmt_hms(eta)<<" | xDBL="<<cnt_xdbl<<" xADD="<<cnt_xadd<<" sqr="<<cnt_sqr<<" mul="<<cnt_mul;
                    std::cout<<line.str()<<std::flush; last_ui = now;
                }
                if (duration_cast<seconds>(now - last_save).count() >= backup_period) { double elapsed = duration<double>(now - t0).count() + saved_et; save_ckpt((uint32_t)(i + 1), elapsed); last_save = now; }
                if (interrupted) { double elapsed = duration<double>(now - t0).count() + saved_et; save_ckpt((uint32_t)(i + 1), elapsed); std::cout<<"\n[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<total_bits<<"\n"; if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<total_bits; guiServer_->appendLog(oss.str()); } curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; }
            }
            std::cout<<std::endl;

            mpz_class Zfin = compute_X_with_dots(eng, (engine::Reg)1, N);
            mpz_class gg = gcd_with_dots(Zfin, N);
            if (gg == N) { std::cout<<"[ECM] Curve "<<(c+1)<<": singular or failure, retrying\n"; delete eng; continue; }
            bool found = (gg > 1 && gg < N);

            double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count();
            { std::ostringstream s1; s1<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 elapsed="<<fixed<<setprecision(2)<<elapsed_stage1<<" s"; std::cout<<s1.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s1.str()); }

            if (found) {
                bool known = is_known(gg);
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<(known?" | known factor=":" | factor=")<<gg.get_str()<<std::endl;
                std::cout << "[ECM] Last curve written to 'lastcurve.gp' (PARI/GP script)." << std::endl;
                std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gg.get_str(); guiServer_->appendLog(oss.str()); }
                if (!known) { std::error_code ec0; fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0); result_factor=gg; result_status="found"; curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0; }
            }
        }
        else
        {
            std::ostringstream s2r; s2r<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Resuming Stage2 at index "<<s2_idx<<"/"<<primesS2_v.size()<<" ("<<fixed<<setprecision(2)<<(primesS2_v.empty()?0.0: (100.0*double(s2_idx)/double(primesS2_v.size())))<<"%)";
            std::cout<<s2r.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2r.str());
        }
        std::cout << "[ECM] Last curve written to 'lastcurve.gp' (PARI/GP script)." << std::endl;
        std::cout << "[ECM] This result has been added to ecm_result.json" << std::endl;

        if (B2 > B1) {
            auto ladder_mul_small = [&](size_t Xin,size_t Zin, uint64_t m, size_t Xout,size_t Zout){
                eng->set((engine::Reg)0, 1u);
                eng->set((engine::Reg)1, 0u);
                eng->copy((engine::Reg)2, (engine::Reg)Xin);
                eng->copy((engine::Reg)3, (engine::Reg)Zin);
                eng->set_multiplicand((engine::Reg)13, (engine::Reg)2);
                eng->set_multiplicand((engine::Reg)14, (engine::Reg)3);
                size_t nbq = u64_bits(m);
                for (size_t i = 0; i < nbq; ++i){
                    size_t bit = nbq - 1 - i;
                    int b = int((m >> bit) & 1ULL);
                    if (b==0){ xADD(2,3,0,1,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); xDBL(0,1,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); }
                    else     { xADD(0,1,2,3,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); xDBL(2,3,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); }
                }
                eng->copy((engine::Reg)Xout, (engine::Reg)0);
                eng->copy((engine::Reg)Zout, (engine::Reg)1);
            };

            uint32_t start_i = 0; double saved = 0.0; if (resume_stage2) { start_i = s2_idx; saved = s2_et; }
            auto t2_0 = high_resolution_clock::now(); auto last2_save = t2_0; auto last2_ui = t2_0; double saved_et2 = saved;

            if (!resume_stage2) eng->set((engine::Reg)15, 1u);

            const uint32_t baseX = 4, baseZ = 5;
            if (!resume_stage2) { eng->copy((engine::Reg)baseX, (engine::Reg)0); eng->copy((engine::Reg)baseZ, (engine::Reg)1); }

            for (size_t i = start_i; i < primesS2_v.size(); ++i){
                uint32_t q = primesS2_v[i];
                ladder_mul_small(baseX, baseZ, q, 7, 8);
                eng->set_multiplicand((engine::Reg)11, (engine::Reg)8);
                eng->mul((engine::Reg)15, (engine::Reg)11);

                auto now2 = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now2 - last2_ui).count() >= 400 || i+1 == primesS2_v.size()) {
                    double done = double(i + 1), total = double(primesS2_v.size());
                    double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                    double ips = done / std::max(1e-9, elapsed);
                    double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                    std::ostringstream line;
                    line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 "<<(i+1)<<"/"<<primesS2_v.size()<<" ("<<fixed<<setprecision(2)<<((primesS2_v.size()? (done*100.0/total):100.0))<<"%) | ETA "<<fmt_hms(eta);
                    std::cout<<line.str()<<std::flush; last2_ui = now2;
                }
                if (duration_cast<seconds>(now2 - last2_save).count() >= backup_period) {
                    double elapsed = duration<double>(now2 - t2_0).count() + saved_et2; save_ckpt2((uint32_t)(i + 1), elapsed); last2_save = now2;
                }
                if (interrupted) {
                    double elapsed = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2; save_ckpt2((uint32_t)(i + 1), elapsed);
                    std::cout<<"\n[ECM] Interrupted at Stage2 curve "<<(c+1)<<" index "<<(i+1)<<"/"<<primesS2_v.size()<<"\n";
                    if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at Stage2 curve "<<(c+1)<<" index "<<(i+1)<<"/"<<primesS2_v.size(); guiServer_->appendLog(oss.str()); }
                    curves_tested_for_found=c+1; options.curves_tested_for_found = c+1 ; write_result(); publish_json(); delete eng; return 0;
                }
            }
            std::cout<<std::endl;

            mpz_class Acc = compute_X_with_dots(eng, (engine::Reg)15, N);
            mpz_class gg2 = gcd_with_dots(Acc, N);
            if (gg2 == N) { std::cout<<"[ECM] Curve "<<(c+1)<<": Stage2 gcd=N, retrying\n"; delete eng; continue; }
            bool found2 = (gg2 > 1 && gg2 < N);

            std::error_code ec2; fs::remove(ckpt2, ec2); fs::remove(ckpt2 + ".old", ec2); fs::remove(ckpt2 + ".new", ec2);

            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
            { std::ostringstream s2s; s2s<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 elapsed="<<fixed<<setprecision(2)<<elapsed2<<" s"; std::cout<<s2s.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2s.str()); }

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
    write_result();
    publish_json();
    return 1;
}
