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

int App::runECMMarin()
{
    using namespace std;
    using namespace std::chrono;

    const uint32_t p = static_cast<uint32_t>(options.exponent);
    const uint64_t B1 = options.B1 ? options.B1 : 1000000ULL;
    const uint64_t B2 = options.B2 ? options.B2 : 0ULL;
    const bool verbose = true;//options.debug;
    uint64_t curves = options.nmax ? options.nmax : (options.K ? options.K : 250);

    auto splitmix64 = [](uint64_t& x)->uint64_t{ x += 0x9E3779B97f4A7C15ULL; uint64_t z=x; z^=z>>30; z*=0xBF58476D1CE4E5B9ULL; z^=z>>27; z*=0x94D049BB133111EBULL; z^=z>>31; return z; };
    auto rnd_mpz = [&](const mpz_class& N, uint64_t& seed, uint64_t bits)->mpz_class{
        mpz_class z=0; for (size_t i=0;i<bits;i+=64){ z <<= 64; z += (unsigned long)splitmix64(seed); } z %= N; if (z<=2) z+=3; return z;
    };
    auto hex64 = [&](const mpz_class& z)->string{
        std::ostringstream ss; ss<<std::uppercase<<std::hex<<std::setw(16)<<std::setfill('0')<<static_cast<uint64_t>(mpz_get_ui(z.get_mpz_t())); return ss.str();
    };
    auto fmt_hms = [&](double s)->string{
        uint64_t u = (uint64_t)(s + 0.5); uint64_t h=u/3600, m=(u%3600)/60, se=u%60; std::ostringstream ss; ss<<h<<"h "<<m<<"m "<<se<<"s"; return ss.str();
    };
    auto u64_bits = [](uint64_t x)->size_t{ if(!x) return 1; size_t n=0; while(x){ ++n; x>>=1; } return n; };
    auto is_known = [&](const mpz_class& f)->bool{
        for (const auto& s : options.knownFactors) {
            if (s.empty()) continue;
            mpz_class z;
            if (mpz_set_str(z.get_mpz_t(), s.c_str(), 0) != 0) continue;
            if (z < 0) z = -z;
            if (f == z) return true;
        }
        return false;
    };

    mpz_class N = (mpz_class(1) << p) - 1;
    {
        std::vector<std::string> kf = options.knownFactors;
        std::vector<std::pair<mpz_class,unsigned>> accepted;
        mpz_class C = N;
        for (const auto& s : kf) {
            if (s.empty()) continue;
            mpz_class f;
            if (mpz_set_str(f.get_mpz_t(), s.c_str(), 0) != 0) {
                std::ostringstream os; os << "[ECM] Ignoring invalid known factor string: " << s;
                std::cout << os.str() << std::endl;
                if (guiServer_) guiServer_->appendLog(os.str());
                continue;
            }
            if (f < 0) f = -f;
            if (f <= 1) continue;
            mpz_class g; mpz_gcd(g.get_mpz_t(), f.get_mpz_t(), N.get_mpz_t());
            if (g > 1) {
                unsigned m = 0;
                while (mpz_divisible_p(C.get_mpz_t(), g.get_mpz_t())) { C /= g; ++m; }
                if (m) accepted.push_back({g, m});
            }
        }
        if (!accepted.empty()) {
            std::ostringstream oss;
            oss << "[ECM] Known factors accepted: ";
            for (size_t i=0;i<accepted.size();++i) {
                if (i) oss << " * ";
                oss << accepted[i].first.get_str() << "^" << accepted[i].second;
            }
            oss << "\n[ECM] Remaining cofactor: " << C.get_str();
            std::cout << oss.str() << std::endl;
            if (guiServer_) guiServer_->appendLog(oss.str());
            int pr = mpz_probab_prime_p(C.get_mpz_t(), 25);
            if (C == 1) { std::cout << "[ECM] Fully factored from known factors\n"; return 0; }
            if (pr) {
                std::ostringstream os2; os2 << "[ECM] Cofactor appears prime: " << C.get_str();
                std::cout << os2.str()<<std::endl;
                if (guiServer_) guiServer_->appendLog(os2.str());
            }
        } else {
            std::cout << "[ECM] No usable known factors provided\n";
        }
    }

    mpz_class K = 1;
    uint32_t primesB1 = 0;
    {
        std::atomic<bool> done{false};
        std::thread ticker([&]{
            const char* msg = "[ECM] Stage1 prep: building K up to B1 ";
            std::cout<<msg<<std::flush;
            size_t dots=0, wrap=60;
            while(!done.load(std::memory_order_relaxed)){
                std::cout<<'.'<<std::flush;
                if(++dots % wrap == 0) std::cout<<'\n'<<msg<<std::flush;
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            std::cout<<" done\n";
        });
        uint64_t b = B1;
        vector<uint8_t> sieve(b/2+1, 1);
        for (uint64_t i=3;i*i<=b;i+=2) if (sieve[i>>1]) for (uint64_t j=i*i;j<=b;j+=i<<1) sieve[j>>1]=0;
        auto apply_prime = [&](uint64_t q){ uint64_t pw=q; while (pw <= b / q) pw *= q; mpz_class t; mpz_set_ui(t.get_mpz_t(), (unsigned long)pw); K *= t; ++primesB1; };
        apply_prime(2);
        for (uint64_t q=3;q<=b;q+=2) if (sieve[q>>1]) apply_prime(q);
        done.store(true, std::memory_order_relaxed);
        ticker.join();
    }
    size_t nb = mpz_sizeinbase(K.get_mpz_t(), 2);

    std::vector<uint32_t> primesS2;
    if (B2 > B1) {
        std::atomic<bool> done{false};
        std::thread ticker([&]{
            const char* msg = "[ECM] Stage2 prep: sieving primes in (B1,B2] ";
            std::cout<<msg<<std::flush;
            size_t dots=0, wrap=60;
            while(!done.load(std::memory_order_relaxed)){
                std::cout<<'.'<<std::flush;
                if(++dots % wrap == 0) std::cout<<'\n'<<msg<<std::flush;
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            std::cout<<" done\n";
        });
        uint64_t b = B2;
        vector<uint8_t> sieve(b/2+1, 1);
        for (uint64_t i=3;i*i<=b;i+=2) if (sieve[i>>1]) for (uint64_t j=i*i;j<=b;j+=i<<1) sieve[j>>1]=0;
        if (B1 < 2 && 2 <= B2) primesS2.push_back(2);
        for (uint64_t q=3;q<=B2;q+=2) if (sieve[q>>1] && q > B1) primesS2.push_back((uint32_t)q);
        done.store(true, std::memory_order_relaxed);
        ticker.join();
    }

    {
        std::ostringstream hdr;
        hdr<<"[ECM] N=M_"<<p<<"  B1="<<B1<<"  B2="<<B2<<"  curves="<<curves<<"\n";
        hdr<<"[ECM] Stage1: product of prime powers up to B1, primes used="<<primesB1<<", K_bits="<<nb<<"\n";
        if (!primesS2.empty()) {
            hdr<<"[ECM] Stage2: primes in ("<<B1<<","<<B2<<"], count="<<primesS2.size();
            hdr<<", first="<<(primesS2.size()?primesS2.front():0)<<", last="<<(primesS2.size()?primesS2.back():0)<<"\n";
        } else {
            hdr<<"[ECM] Stage2: disabled\n";
        }
        std::cout<<hdr.str();
        if (guiServer_) guiServer_->appendLog(hdr.str());
    }

    auto now_ns = (uint64_t)duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    uint64_t seed0 = now_ns ^ ((uint64_t)p<<32) ^ B1;

    size_t transform_size_once = 0;
    const int backup_period = options.backup_interval > 0 ? options.backup_interval : 10;

    for (uint64_t c = 0; c < curves; ++c)
    {
        engine* eng = engine::create_gpu(p, static_cast<size_t>(18), static_cast<size_t>(options.device_id), verbose);
        if (transform_size_once == 0) {
            transform_size_once = eng->get_size();
            std::ostringstream os; os<<"[ECM] Transform size="<<transform_size_once<<" words, device_id="<<options.device_id;
            std::cout<<os.str()<<std::endl; if (guiServer_) guiServer_->appendLog(os.str());
        }

        std::ostringstream ck;  ck << "ecm_m_"  << p << "_c" << c << ".ckpt";
        std::ostringstream ck2; ck2<< "ecm2_m_" << p << "_c" << c << ".ckpt";
        const std::string ckpt_file = ck.str();
        const std::string ckpt2     = ck2.str();

        auto save_ckpt = [&](uint32_t i, double et){
            const std::string oldf = ckpt_file + ".old", newf = ckpt_file + ".new";
            { File f(newf, "wb"); int version = 1; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&i), sizeof(i))) return; uint32_t nbb = (uint32_t)nb; if (!f.write(reinterpret_cast<const char*>(&nbb), sizeof(nbb))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
            std::error_code ec; fs::remove(ckpt_file + ".old", ec); fs::rename(ckpt_file, ckpt_file + ".old", ec); fs::rename(ckpt_file + ".new", ckpt_file, ec); fs::remove(ckpt_file + ".old", ec);
        };
        auto read_ckpt = [&](const std::string& file, uint32_t& ri, uint32_t& rnb, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 1) return -2;
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;
            if (!f.read(reinterpret_cast<char*>(&ri), sizeof(ri))) return -2;
            if (!f.read(reinterpret_cast<char*>(&rnb), sizeof(rnb))) return -2;
            uint64_t rB1 = 0; if (!f.read(reinterpret_cast<char*>(&rB1), sizeof(rB1))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            if (rnb != nb || rB1 != B1) return -2;
            return 0;
        };

        auto save_ckpt2 = [&](uint32_t idx, double et){
            const std::string oldf = ckpt2 + ".old", newf = ckpt2 + ".new";
            { File f(newf, "wb"); int version = 2; if (!f.write(reinterpret_cast<const char*>(&version), sizeof(version))) return; if (!f.write(reinterpret_cast<const char*>(&p), sizeof(p))) return; if (!f.write(reinterpret_cast<const char*>(&idx), sizeof(idx))) return; uint32_t cnt = (uint32_t)primesS2.size(); if (!f.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt))) return; if (!f.write(reinterpret_cast<const char*>(&B1), sizeof(B1))) return; if (!f.write(reinterpret_cast<const char*>(&B2), sizeof(B2))) return; if (!f.write(reinterpret_cast<const char*>(&et), sizeof(et))) return; const size_t cksz = eng->get_checkpoint_size(); std::vector<char> data(cksz); if (!eng->get_checkpoint(data)) return; if (!f.write(data.data(), cksz)) return; f.write_crc32(); }
            std::error_code ec; fs::remove(ckpt2 + ".old", ec); fs::rename(ckpt2, ckpt2 + ".old", ec); fs::rename(ckpt2 + ".new", ckpt2, ec); fs::remove(ckpt2 + ".old", ec);
        };
        auto read_ckpt2 = [&](const std::string& file, uint32_t& idx, uint32_t& cnt, double& et)->int{
            File f(file);
            if (!f.exists()) return -1;
            int version = 0; if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return -2;
            if (version != 2) return -2;
            uint32_t rp = 0; if (!f.read(reinterpret_cast<char*>(&rp), sizeof(rp))) return -2;
            if (rp != p) return -2;
            if (!f.read(reinterpret_cast<char*>(&idx), sizeof(idx))) return -2;
            if (!f.read(reinterpret_cast<char*>(&cnt), sizeof(cnt))) return -2;
            uint64_t b1s=0,b2s=0; if (!f.read(reinterpret_cast<char*>(&b1s), sizeof(b1s))) return -2; if (!f.read(reinterpret_cast<char*>(&b2s), sizeof(b2s))) return -2;
            if (!f.read(reinterpret_cast<char*>(&et), sizeof(et))) return -2;
            const size_t cksz = eng->get_checkpoint_size();
            std::vector<char> data(cksz);
            if (!f.read(data.data(), cksz)) return -2;
            if (!eng->set_checkpoint(data)) return -2;
            if (!f.check_crc32()) return -2;
            if (cnt != primesS2.size() || b1s != B1 || b2s != B2) return -2;
            return 0;
        };

        uint32_t s2_idx = 0, s2_cnt = 0; double s2_et = 0.0;
        bool resume_stage2 = false;
        {
            int rr2 = read_ckpt2(ckpt2, s2_idx, s2_cnt, s2_et);
            if (rr2 < 0) rr2 = read_ckpt2(ckpt2 + ".old", s2_idx, s2_cnt, s2_et);
            resume_stage2 = (rr2 == 0);
        }

        uint64_t seed = seed0 + c*0xD1342543DE82EF95ULL;
        mpz_class sigma, u, v, A, A24, x0;

        if (!resume_stage2)
        {
            sigma = rnd_mpz(N, seed, 128);
            u = (sigma*sigma - 5) % N; if (u < 0) u += N;
            v = (4*sigma) % N;

            mpz_class g; mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t());
            if (g > 1 && g < N) { std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl; if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found at curve "<<(c+1)<<": "<<g.get_str(); guiServer_->appendLog(oss.str()); } delete eng; return 0; }
            mpz_class invv; if (!mpz_invert(invv.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t())) { mpz_gcd(g.get_mpz_t(), v.get_mpz_t(), N.get_mpz_t()); std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl; if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found (invert v failed): "<<g.get_str(); guiServer_->appendLog(oss.str()); } delete eng; return 0; }
            mpz_class t = (u * invv) % N; x0 = (t*t - 2) % N; if (x0 < 0) x0 += N;

            mpz_class t1 = (v - u) % N; if (t1 < 0) t1 += N; mpz_class t1_3 = (t1*t1) % N; t1_3 = (t1_3 * t1) % N;
            mpz_class t2 = (3*u + v) % N; if (t2 < 0) t2 += N; mpz_class num = (t1_3 * t2) % N;
            mpz_class den = 0; { mpz_class u3 = (u*u) % N; u3 = (u3*u) % N; den = (4 * u3) % N; den = (den * v) % N; }
            mpz_gcd(g.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()); if (g > 1 && g < N) { std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl; if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found in parameterization: "<<g.get_str(); guiServer_->appendLog(oss.str()); } delete eng; return 0; }
            mpz_class invden; if (!mpz_invert(invden.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t())) { mpz_gcd(g.get_mpz_t(), den.get_mpz_t(), N.get_mpz_t()); std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" factor="<<g.get_str()<<std::endl; if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Factor found (invert den failed): "<<g.get_str(); guiServer_->appendLog(oss.str()); } delete eng; return 0; }
            A = (num * invden) % N; A = (A - 2) % N; if (A < 0) A += N;
            mpz_class inv4; mpz_invert(inv4.get_mpz_t(), mpz_class(4).get_mpz_t(), N.get_mpz_t());
            A24 = ((A + 2) % N) * inv4 % N;

            const size_t RX0=0, RZ0=1, RX1=2, RZ1=3, RXD=4, RZD=5, RA24=6, RT0=7, RT1=8, RT2=9, RT3=10, RU=11, RV=12, RM=13;

            eng->set((engine::Reg)RZ0, 0u);
            eng->set((engine::Reg)RX0, 1u);
            eng->set((engine::Reg)RZ1, 1u);
            { mpz_t z; mpz_init(z); mpz_set(z, x0.get_mpz_t()); eng->set_mpz((engine::Reg)RX1, z); mpz_set(z, x0.get_mpz_t()); eng->set_mpz((engine::Reg)RXD, z); mpz_set_ui(z, 1); eng->set_mpz((engine::Reg)RZD, z); mpz_set(z, A24.get_mpz_t()); eng->set_mpz((engine::Reg)RA24, z); mpz_clear(z); }

            uint64_t cnt_xdbl=0, cnt_xadd=0, cnt_mul=0, cnt_sqr=0;

            auto mul_inplace = [&](size_t dst, size_t src){ eng->set_multiplicand((engine::Reg)RM, (engine::Reg)src); eng->mul((engine::Reg)dst, (engine::Reg)RM); ++cnt_mul; };
            auto xDBL = [&](size_t X1,size_t Z1,size_t X2,size_t Z2){
                eng->copy((engine::Reg)RT0, (engine::Reg)X1);
                eng->add((engine::Reg)RT0, (engine::Reg)Z1);
                eng->copy((engine::Reg)RT1, (engine::Reg)RT0);
                eng->square_mul((engine::Reg)RT1); ++cnt_sqr;
                eng->copy((engine::Reg)RT2, (engine::Reg)X1);
                eng->sub_reg((engine::Reg)RT2, (engine::Reg)Z1);
                eng->copy((engine::Reg)RT3, (engine::Reg)RT2);
                eng->square_mul((engine::Reg)RT3); ++cnt_sqr;
                eng->copy((engine::Reg)X2, (engine::Reg)RT1);
                mul_inplace(X2, RT3);
                eng->copy((engine::Reg)RT0, (engine::Reg)RT1);
                eng->sub_reg((engine::Reg)RT0, (engine::Reg)RT3);
                eng->copy((engine::Reg)RT2, (engine::Reg)RT0);
                mul_inplace(RT2, RA24);
                eng->add((engine::Reg)RT2, (engine::Reg)RT3);
                eng->copy((engine::Reg)Z2, (engine::Reg)RT2);
                mul_inplace(Z2, RT0);
                ++cnt_xdbl;
            };
            auto xADD = [&](size_t X1,size_t Z1,size_t X2,size_t Z2,size_t XD,size_t ZD,size_t X3,size_t Z3){
                eng->copy((engine::Reg)RT0, (engine::Reg)X1);
                eng->add((engine::Reg)RT0, (engine::Reg)Z1);
                eng->copy((engine::Reg)RT1, (engine::Reg)X1);
                eng->sub_reg((engine::Reg)RT1, (engine::Reg)Z1);
                eng->copy((engine::Reg)RT2, (engine::Reg)X2);
                eng->add((engine::Reg)RT2, (engine::Reg)Z2);
                eng->copy((engine::Reg)RT3, (engine::Reg)X2);
                eng->sub_reg((engine::Reg)RT3, (engine::Reg)Z2);
                eng->copy((engine::Reg)RU, (engine::Reg)RT3);
                mul_inplace(RU, RT0);
                eng->copy((engine::Reg)RV, (engine::Reg)RT2);
                mul_inplace(RV, RT1);
                eng->copy((engine::Reg)X3, (engine::Reg)RU);
                eng->add((engine::Reg)X3, (engine::Reg)RV);
                eng->square_mul((engine::Reg)X3); ++cnt_sqr;
                mul_inplace(X3, ZD);
                eng->copy((engine::Reg)Z3, (engine::Reg)RU);
                eng->sub_reg((engine::Reg)Z3, (engine::Reg)RV);
                eng->square_mul((engine::Reg)Z3); ++cnt_sqr;
                mul_inplace(Z3, XD);
                ++cnt_xadd;
            };

            std::ostringstream head;
            head<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | sigma64=0x"<<hex64(sigma)<<" x0_64=0x"<<hex64(x0)<<" A_64=0x"<<hex64(A)<<" A24_64=0x"<<hex64(A24)<<" | K_bits="<<nb;
            std::cout<<head.str()<<std::endl;
            if (guiServer_) guiServer_->appendLog(head.str());

            uint32_t start_i = 0, nb_ck = 0; double saved_et = 0.0;
            (void)read_ckpt(ckpt_file, start_i, nb_ck, saved_et);

            auto t0 = high_resolution_clock::now();
            auto last_save = t0;
            auto last_ui = t0;

            for (size_t i = start_i; i < nb; ++i){
                size_t bit = nb - 1 - i;
                mp_bitcnt_t mb = static_cast<mp_bitcnt_t>(bit);
                int b = mpz_tstbit(K.get_mpz_t(), mb) ? 1 : 0;
                if (b==0) { xADD(2,3,0,1,4,5,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); xDBL(0,1,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); }
                else      { xADD(0,1,2,3,4,5,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); xDBL(2,3,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); }

                auto now = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now - last_ui).count() >= 400 || i+1 == nb) {
                    double done = double(i + 1), total = double(nb);
                    double elapsed = duration<double>(now - t0).count() + saved_et;
                    double ips = done / std::max(1e-9, elapsed);
                    double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                    std::ostringstream line;
                    line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 "<<(i+1)<<"/"<<nb<<" ("<<fixed<<setprecision(2)<<(done*100.0/total)<<"%)"
                        <<" | ETA "<<fmt_hms(eta)
                        <<" | xDBL="<<cnt_xdbl<<" xADD="<<cnt_xadd<<" sqr="<<cnt_sqr<<" mul="<<cnt_mul;
                    std::cout<<line.str()<<std::flush;
                    last_ui = now;
                }
                if (duration_cast<seconds>(now - last_save).count() >= backup_period) {
                    double elapsed = duration<double>(now - t0).count() + saved_et;
                    save_ckpt((uint32_t)(i + 1), elapsed);
                    last_save = now;
                }
                if (interrupted) {
                    double elapsed = duration<double>(now - t0).count() + saved_et;
                    save_ckpt((uint32_t)(i + 1), elapsed);
                    std::cout<<"\n[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<nb<<"\n";
                    if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at curve "<<(c+1)<<", bit "<<(i+1)<<"/"<<nb; guiServer_->appendLog(oss.str()); }
                    delete eng;
                    return 0;
                }
            }
            std::cout<<std::endl;

            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | GCD after Stage1..."<<std::endl;
            mpz_class Zfin = compute_X_with_dots(eng, (engine::Reg)1, N);
            mpz_class gg = gcd_with_dots(Zfin, N);
            bool found = (gg > 1 && gg < N);

            double elapsed_stage1 = duration<double>(high_resolution_clock::now() - t0).count();
            {
                std::ostringstream s1;
                s1<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage1 elapsed="<<fixed<<setprecision(2)<<elapsed_stage1<<" s";
                std::cout<<s1.str()<<std::endl;
                if (guiServer_) guiServer_->appendLog(s1.str());
            }

            if (found) {
                bool known = is_known(gg);
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<(known?" | known factor=":" | factor=")<<gg.get_str()<<std::endl;
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gg.get_str(); guiServer_->appendLog(oss.str()); }
                if (!known) { std::error_code ec0; fs::remove(ckpt_file, ec0); fs::remove(ckpt_file + ".old", ec0); fs::remove(ckpt_file + ".new", ec0); delete eng; return 0; }
            }
        }
        else
        {
            std::ostringstream s2r; s2r<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Resuming Stage2 at index "<<s2_idx<<"/"<<primesS2.size()<<" ("<<fixed<<setprecision(2)<<(primesS2.empty()?0.0: (100.0*double(s2_idx)/double(primesS2.size())))<<"%)";
            std::cout<<s2r.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2r.str());
        }

        if (B2 > B1) {
            bool use_bsgs = options.bsgs ? true : false;
            uint32_t brent_deg = 1; if (options.brent > 1) brent_deg = (uint32_t)options.brent; else if (options.brent) brent_deg = 2;
            if (!resume_stage2) { std::ostringstream s2h; s2h<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 start, primes="<<primesS2.size(); if (use_bsgs) s2h<<" | bsgs"; if (brent_deg>1) s2h<<" | brent_deg="<<brent_deg; std::cout<<s2h.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2h.str()); }

            auto ladder_mul_small = [&](size_t Xin,size_t Zin, uint64_t m, size_t Xout,size_t Zout){
                eng->set((engine::Reg)0, 1u);
                eng->set((engine::Reg)1, 0u);
                eng->copy((engine::Reg)2, (engine::Reg)Xin);
                eng->copy((engine::Reg)3, (engine::Reg)Zin);
                eng->copy((engine::Reg)4, (engine::Reg)Xin);
                eng->copy((engine::Reg)5, (engine::Reg)Zin);
                size_t nbq = u64_bits(m);
                auto mul_inplace = [&](size_t dst, size_t src){ eng->set_multiplicand((engine::Reg)13, (engine::Reg)src); eng->mul((engine::Reg)dst, (engine::Reg)13); };
                auto xDBL = [&](size_t X1,size_t Z1,size_t X2,size_t Z2){
                    eng->copy((engine::Reg)7, (engine::Reg)X1);
                    eng->add((engine::Reg)7, (engine::Reg)Z1);
                    eng->copy((engine::Reg)8, (engine::Reg)7);
                    eng->square_mul((engine::Reg)8);
                    eng->copy((engine::Reg)9, (engine::Reg)X1);
                    eng->sub_reg((engine::Reg)9, (engine::Reg)Z1);
                    eng->copy((engine::Reg)10, (engine::Reg)9);
                    eng->square_mul((engine::Reg)10);
                    eng->copy((engine::Reg)X2, (engine::Reg)8);
                    mul_inplace(X2, 10);
                    eng->copy((engine::Reg)7, (engine::Reg)8);
                    eng->sub_reg((engine::Reg)7, (engine::Reg)10);
                    eng->copy((engine::Reg)9, (engine::Reg)7);
                    mul_inplace(9, 6);
                    eng->add((engine::Reg)9, (engine::Reg)10);
                    eng->copy((engine::Reg)Z2, (engine::Reg)9);
                    mul_inplace(Z2, 7);
                };
                auto xADD = [&](size_t X1,size_t Z1,size_t X2,size_t Z2,size_t XD,size_t ZD,size_t X3,size_t Z3){
                    eng->copy((engine::Reg)7, (engine::Reg)X1);
                    eng->add((engine::Reg)7, (engine::Reg)Z1);
                    eng->copy((engine::Reg)8, (engine::Reg)X1);
                    eng->sub_reg((engine::Reg)8, (engine::Reg)Z1);
                    eng->copy((engine::Reg)9, (engine::Reg)X2);
                    eng->add((engine::Reg)9, (engine::Reg)Z2);
                    eng->copy((engine::Reg)10, (engine::Reg)X2);
                    eng->sub_reg((engine::Reg)10, (engine::Reg)Z2);
                    eng->copy((engine::Reg)11, (engine::Reg)10);
                    mul_inplace(11, 7);
                    eng->copy((engine::Reg)12, (engine::Reg)9);
                    mul_inplace(12, 8);
                    eng->copy((engine::Reg)X3, (engine::Reg)11);
                    eng->add((engine::Reg)X3, (engine::Reg)12);
                    eng->square_mul((engine::Reg)X3);
                    mul_inplace(X3, ZD);
                    eng->copy((engine::Reg)Z3, (engine::Reg)11);
                    eng->sub_reg((engine::Reg)Z3, (engine::Reg)12);
                    eng->square_mul((engine::Reg)Z3);
                    mul_inplace(Z3, XD);
                };
                for (size_t bi=0; bi<nbq; ++bi){
                    size_t bit = nbq - 1 - bi;
                    int b = ((m >> bit) & 1ULL) ? 1 : 0;
                    if (b==0) { xADD(2,3,0,1,4,5,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); xDBL(0,1,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); }
                    else      { xADD(0,1,2,3,4,5,7,8); eng->copy((engine::Reg)0,(engine::Reg)7); eng->copy((engine::Reg)1,(engine::Reg)8); xDBL(2,3,7,8); eng->copy((engine::Reg)2,(engine::Reg)7); eng->copy((engine::Reg)3,(engine::Reg)8); }
                }
                eng->copy((engine::Reg)Xout, (engine::Reg)0);
                eng->copy((engine::Reg)Zout, (engine::Reg)1);
            };

            auto t2_0 = high_resolution_clock::now();
            auto last2_save = t2_0;
            auto last2_ui = t2_0;

            size_t Xcur = 0, Zcur = 1;

            uint32_t start_idx = resume_stage2 ? s2_idx : 0;
            double saved_et2 = resume_stage2 ? s2_et : 0.0;

            auto mul_ov = [](uint64_t a, uint64_t b, uint64_t& out)->bool{
                if (a==0 || b==0){ out = 0; return false; }
                const uint64_t U = ~uint64_t(0);
                if (a > U / b) return true;
                out = a * b;
                return false;
            };
            auto pow_ov = [&](uint64_t base, uint32_t exp, uint64_t& out)->bool{
                out = 1;
                for (uint32_t k=0;k<exp;k++){
                    uint64_t t=0;
                    if (mul_ov(out, base, t)) return true;
                    out = t;
                }
                return false;
            };

            const uint32_t block_cap = options.bsgs ? 8u : 1u;
            uint64_t Macc = 1;
            uint32_t in_block = 0;

            for (uint32_t i = start_idx; i < (uint32_t)primesS2.size(); ++i) {
                uint64_t q = primesS2[i];
                uint64_t mexp = q;
                bool big = false;
                if (brent_deg > 1) {
                    if (pow_ov(q, brent_deg, mexp)) big = true;
                }

                if (!big && options.bsgs && in_block < block_cap) {
                    uint64_t tmp=0;
                    if (!mul_ov(Macc, mexp, tmp)) {
                        Macc = tmp;
                        ++in_block;
                    } else {
                        if (Macc > 1) {
                            ladder_mul_small(Xcur, Zcur, Macc, 7, 8);
                            eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                            eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                            Macc = 1;
                            in_block = 0;
                        }
                        ladder_mul_small(Xcur, Zcur, mexp, 7, 8);
                        eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                        eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                    }
                } else {
                    if (Macc > 1) {
                        ladder_mul_small(Xcur, Zcur, Macc, 7, 8);
                        eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                        eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                        Macc = 1;
                        in_block = 0;
                    }
                    if (big) {
                        for (uint32_t k=0;k<brent_deg;k++){
                            ladder_mul_small(Xcur, Zcur, q, 7, 8);
                            eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                            eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                        }
                    } else {
                        ladder_mul_small(Xcur, Zcur, mexp, 7, 8);
                        eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                        eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                    }
                }

                auto now2 = high_resolution_clock::now();
                if (duration_cast<milliseconds>(now2 - last2_ui).count() >= 400 || i+1 == primesS2.size()) {
                    double done = double(i + 1), total = double(primesS2.size());
                    double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                    double ips = done / std::max(1e-9, elapsed);
                    double eta = (total > done && ips > 0.0) ? (total - done) / ips : 0.0;
                    std::ostringstream line;
                    line<<"\r[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 "<<(i+1)<<"/"<<primesS2.size()<<" ("<<fixed<<setprecision(2)<<(total? (done*100.0/total):100.0)<<"%)"
                        <<" | ETA "<<fmt_hms(eta);
                    std::cout<<line.str()<<std::flush;
                    last2_ui = now2;
                }

                if (duration_cast<seconds>(now2 - last2_save).count() >= backup_period) {
                    if (Macc > 1) {
                        ladder_mul_small(Xcur, Zcur, Macc, 7, 8);
                        eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                        eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                        Macc = 1;
                        in_block = 0;
                    }
                    double elapsed = duration<double>(now2 - t2_0).count() + saved_et2;
                    save_ckpt2((uint32_t)(i + 1), elapsed);
                    last2_save = now2;
                }
                if (interrupted) {
                    if (Macc > 1) {
                        ladder_mul_small(Xcur, Zcur, Macc, 7, 8);
                        eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                        eng->copy((engine::Reg)Zcur, (engine::Reg)8);
                        Macc = 1;
                        in_block = 0;
                    }
                    double elapsed = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
                    save_ckpt2((uint32_t)(i + 1), elapsed);
                    std::cout<<"\n[ECM] Interrupted at Stage2 curve "<<(c+1)<<" index "<<(i+1)<<"/"<<primesS2.size()<<"\n";
                    if (guiServer_) { std::ostringstream oss; oss<<"[ECM] Interrupted at Stage2 curve "<<(c+1)<<" index "<<(i+1)<<"/"<<primesS2.size(); guiServer_->appendLog(oss.str()); }
                    delete eng;
                    return 0;
                }
            }
            if (Macc > 1) {
                ladder_mul_small(Xcur, Zcur, Macc, 7, 8);
                eng->copy((engine::Reg)Xcur, (engine::Reg)7);
                eng->copy((engine::Reg)Zcur, (engine::Reg)8);
            }
            std::cout<<std::endl;

            std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | GCD after Stage2..."<<std::endl;
            mpz_class Zfin2 = compute_X_with_dots(eng, (engine::Reg)Zcur, N);
            mpz_class gg2 = gcd_with_dots(Zfin2, N);
            bool found2 = (gg2 > 1 && gg2 < N);

            std::error_code ec2; fs::remove(ckpt2, ec2); fs::remove(ckpt2 + ".old", ec2); fs::remove(ckpt2 + ".new", ec2);

            double elapsed2 = duration<double>(high_resolution_clock::now() - t2_0).count() + saved_et2;
            { std::ostringstream s2s; s2s<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" | Stage2 elapsed="<<fixed<<setprecision(2)<<elapsed2<<" s"; std::cout<<s2s.str()<<std::endl; if (guiServer_) guiServer_->appendLog(s2s.str()); }

            if (found2) {
                bool known = is_known(gg2);
                std::cout<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<(known?" | known factor=":" | factor=")<<gg2.get_str()<<std::endl;
                if (guiServer_) { std::ostringstream oss; oss<<"[ECM] "<<(known?"Known ":"")<<"factor: "<<gg2.get_str(); guiServer_->appendLog(oss.str()); }
                if (!known) { std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec); delete eng; return 0; }
            }
        }

        std::error_code ec; fs::remove(ckpt_file, ec); fs::remove(ckpt_file + ".old", ec); fs::remove(ckpt_file + ".new", ec);
        static thread_local std::chrono::high_resolution_clock::time_point curve_t0 = high_resolution_clock::now();
        //double elapsed = duration<double>(high_resolution_clock::now() - curve_t0).count();
        { std::ostringstream fin; fin<<"[ECM] Curve "<<(c+1)<<"/"<<curves<<" done"; std::cout<<fin.str()<<std::endl; if (guiServer_) guiServer_->appendLog(fin.str()); }
        curve_t0 = high_resolution_clock::now();

        delete eng;
    }

    std::cout<<"[ECM] No factor found"<<std::endl;
    return 1;
}
