#include "core/App.hpp"
#include "core/AlgoUtils.hpp"
#include "core/Version.hpp"
#include "marin/engine.h"
#include "marin/file.h"

#include <gmpxx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using core::algo::buildE;
using core::algo::interrupted;

constexpr const char* GM_ECM_OPT_RELEASE = "v99.98";
constexpr std::array<char, 8> GM_OPT_MAGIC{{'P','R','G','M','O','P','T','1'}};
constexpr std::uint32_t GM_OPT_CHECKPOINT_VERSION = 1;

struct OptTarget {
    std::string family = "GM";
    std::uint32_t p = 0;
    std::uint32_t lift = 0;
    std::uint64_t middle = 0;
    int chi = 0;
    mpz_class n;
};

std::uint64_t add_mod_u64_opt(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
    return a >= mod - b ? a - (mod - b) : a + b;
}

std::uint64_t mul_mod_u64_opt(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
    a %= mod;
    b %= mod;
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned __int64 high = 0;
    const unsigned __int64 low = _umul128(a, b, &high);
    unsigned __int64 remainder = 0;
    (void)_udiv128(high, low, mod, &remainder);
    return static_cast<std::uint64_t>(remainder);
#elif defined(__SIZEOF_INT128__)
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % mod);
#else
    std::uint64_t r = 0;
    while (b != 0) {
        if ((b & 1U) != 0) r = add_mod_u64_opt(r, a, mod);
        b >>= 1U;
        if (b != 0) a = add_mod_u64_opt(a, a, mod);
    }
    return r;
#endif
}

std::uint64_t pow_mod_u64_opt(std::uint64_t a, std::uint64_t e, std::uint64_t mod) {
    std::uint64_t r = 1 % mod;
    while (e != 0) {
        if ((e & 1U) != 0) r = mul_mod_u64_opt(r, a, mod);
        e >>= 1U;
        if (e != 0) a = mul_mod_u64_opt(a, a, mod);
    }
    return r;
}

bool is_prime_u64_opt(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint32_t p : {2U,3U,5U,7U,11U,13U,17U,19U,23U,29U,31U,37U}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    std::uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1U) == 0) { d >>= 1U; ++s; }
    constexpr std::array<std::uint64_t, 7> bases{{
        2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL
    }};
    for (std::uint64_t a : bases) {
        if (a % n == 0) continue;
        std::uint64_t x = pow_mod_u64_opt(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_u64_opt(x, x, n);
            if (x == n - 1) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}

OptTarget make_opt_target(std::uint64_t p64, std::string family) {
    std::transform(family.begin(), family.end(), family.begin(),
                   [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
    if (p64 < 3 || (p64 & 1ULL) == 0 || !is_prime_u64_opt(p64))
        throw std::runtime_error("Gaussian ECM requires an odd prime exponent p >= 3");
    if (p64 > std::numeric_limits<std::uint32_t>::max() / 4ULL)
        throw std::runtime_error("Gaussian ECM requires 4p <= 2^32-1");
    if (family != "GM" && family != "GQ")
        throw std::runtime_error("Gaussian ECM target family must be GM or GQ");

    OptTarget t;
    t.family = family;
    t.p = static_cast<std::uint32_t>(p64);
    t.lift = static_cast<std::uint32_t>(4ULL * p64);
    t.middle = (p64 + 1) / 2;
    const std::uint64_t r = p64 & 7ULL;
    t.chi = (r == 1 || r == 7) ? 1 : -1;
    const bool gq = family == "GQ";

    t.n = mpz_class(1) << p64;
    const mpz_class mid = mpz_class(1) << t.middle;
    if ((!gq && t.chi > 0) || (gq && t.chi < 0)) t.n -= mid;
    else t.n += mid;
    t.n += 1;

    if (gq) {
        if (!mpz_divisible_ui_p(t.n.get_mpz_t(), 5))
            throw std::runtime_error("Gaussian GQ numerator is not divisible by 5");
        mpz_divexact_ui(t.n.get_mpz_t(), t.n.get_mpz_t(), 5);
    }
    return t;
}

mpz_class mod_pos_opt(mpz_class x, const mpz_class& n) {
    x %= n;
    if (x < 0) x += n;
    return x;
}

mpz_class gcd_opt(const mpz_class& a, const mpz_class& n) {
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), n.get_mpz_t());
    return g;
}

bool proper_factor_opt(const mpz_class& g, const mpz_class& n) {
    return g > 1 && g < n;
}

mpz_class project_reg_opt(engine* eng, engine::Reg reg, const mpz_class& n) {
    mpz_t z;
    mpz_init(z);
    eng->get_mpz(z, reg);
    mpz_class out(z);
    mpz_clear(z);
    return mod_pos_opt(out, n);
}

void set_reg_opt(engine* eng, engine::Reg reg, const mpz_class& v, const mpz_class& n) {
    mpz_class x = mod_pos_opt(v, n);
    mpz_t z;
    mpz_init_set(z, x.get_mpz_t());
    eng->set_mpz(reg, z);
    mpz_clear(z);
}

std::uint64_t splitmix64_opt(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

std::string low_hex_opt(const mpz_class& x, unsigned bits = 64) {
    mpz_class low;
    mpz_fdiv_r_2exp(low.get_mpz_t(), x.get_mpz_t(), bits);
    std::string s = low.get_str(16);
    const std::size_t width = bits / 4;
    if (s.size() < width) s.insert(0, width - s.size(), '0');
    return s;
}

std::string json_escape_opt(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        if (c == '\\') out << "\\\\";
        else if (c == '"') out << "\\\"";
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else out << c;
    }
    return out.str();
}

void write_opt_result(const std::filesystem::path& dir,
                      const OptTarget& t,
                      std::uint64_t B1,
                      std::uint64_t B2,
                      std::uint64_t curves,
                      const std::optional<std::uint64_t>& curve,
                      const std::optional<std::uint64_t>& sigma,
                      int stage,
                      const std::optional<mpz_class>& factor,
                      const std::string& backend,
                      int device,
                      double elapsed) {
    std::filesystem::create_directories(dir);
    const std::string prefix = t.family == "GQ" ? "gq" : "gm";
    const std::filesystem::path result =
        dir / (prefix + "_ecm_p" + std::to_string(t.p) + "_result.json");

    std::ostringstream j;
    j << std::fixed << std::setprecision(3);
    j << "{\n"
      << "  \"schema_version\": 2,\n"
      << "  \"program\": \"PrMers\",\n"
      << "  \"program_version\": \"" << GM_ECM_OPT_RELEASE << "\",\n"
      << "  \"program_build\": \"" << json_escape_opt(core::PRMERS_VERSION) << "\",\n"
      << "  \"family\": \"gaussian-pair\",\n"
      << "  \"target_family\": \"" << t.family << "\",\n"
      << "  \"mode\": \"gm-ecm\",\n"
      << "  \"engine\": \"montgomery-fused-bsgs\",\n"
      << "  \"outcome\": \"" << (factor ? "factor" : "no-factor") << "\",\n"
      << "  \"stage\": " << stage << ",\n"
      << "  \"exponent\": " << t.p << ",\n"
      << "  \"B1\": \"" << B1 << "\",\n"
      << "  \"B2\": " << (B2 > B1 ? ("\"" + std::to_string(B2) + "\"") : "null") << ",\n"
      << "  \"curves\": " << curves << ",\n"
      << "  \"curve\": " << (curve ? std::to_string(*curve) : "null") << ",\n"
      << "  \"sigma\": " << (sigma ? ("\"" + std::to_string(*sigma) + "\"") : "null") << ",\n"
      << "  \"factor\": " << (factor ? ("\"" + factor->get_str() + "\"") : "null") << ",\n"
      << "  \"backend\": \"" << json_escape_opt(backend) << "\",\n"
      << "  \"device\": \"device " << device << "\",\n"
      << "  \"elapsed_seconds\": " << elapsed << "\n"
      << "}";

    {
        std::ofstream out(result);
        out << j.str() << '\n';
    }
    {
        std::ofstream out(dir / "results.txt", std::ios::app);
        out << j.str() << '\n';
    }
    std::cout << "Result file: \"" << result.string() << "\"\n";
}

struct SuyamaSetupOpt {
    mpz_class x;
    mpz_class a24;
    mpz_class factor;
    bool ok = false;
};

SuyamaSetupOpt make_suyama_opt(const mpz_class& n, std::uint64_t sigma64) {
    SuyamaSetupOpt out;
    mpz_class sigma;
    mpz_import(sigma.get_mpz_t(), 1, 1, sizeof(sigma64), 0, 0, &sigma64);
    sigma %= n;
    if (sigma < 6) sigma += 6;

    const mpz_class u = mod_pos_opt(sigma * sigma - 5, n);
    const mpz_class v = mod_pos_opt(4 * sigma, n);
    const mpz_class xproj = mod_pos_opt(u * u * u, n);
    const mpz_class zproj = mod_pos_opt(v * v * v, n);

    mpz_class g = gcd_opt(zproj, n);
    if (proper_factor_opt(g, n)) { out.factor = g; return out; }

    mpz_class invz;
    if (mpz_invert(invz.get_mpz_t(), zproj.get_mpz_t(), n.get_mpz_t()) == 0) {
        out.factor = gcd_opt(zproj, n);
        return out;
    }
    out.x = mod_pos_opt(xproj * invz, n);

    const mpz_class vu = mod_pos_opt(v - u, n);
    const mpz_class numerator = mod_pos_opt(vu * vu * vu * (3 * u + v), n);
    const mpz_class denominator = mod_pos_opt(16 * xproj * v, n);
    g = gcd_opt(denominator, n);
    if (proper_factor_opt(g, n)) { out.factor = g; return out; }

    mpz_class invden;
    if (mpz_invert(invden.get_mpz_t(), denominator.get_mpz_t(), n.get_mpz_t()) == 0) {
        out.factor = gcd_opt(denominator, n);
        return out;
    }
    out.a24 = mod_pos_opt(numerator * invden, n);
    out.ok = true;
    return out;
}

struct MRegsOpt {
    engine::Reg xa = 0, za = 1, xb = 2, zb = 3;
    engine::Reg a24 = 4, xdiff = 5, mxdiff = 6, ma24 = 7;
    engine::Reg A = 8, B = 9, C = 10, D = 11;
    engine::Reg mA = 12, mB = 13;
    engine::Reg DA = 14, CB = 15, sum = 16, diff = 17;
    engine::Reg AA = 18, BB = 19, mBB = 20;
    engine::Reg E = 21, tmp = 22, mTmp = 23;
    static constexpr std::size_t core_count = 24;
};

void set_mont_constants_opt(engine* eng,
                            const MRegsOpt& r,
                            const mpz_class& xdiff,
                            const mpz_class& a24,
                            const mpz_class& n) {
    set_reg_opt(eng, r.xdiff, xdiff, n);
    set_reg_opt(eng, r.a24, a24, n);
    eng->set_multiplicand(r.mxdiff, r.xdiff);
    eng->set_multiplicand(r.ma24, r.a24);
}

/*
 * Fused xDBL.
 *
 * The legacy GM-ECM code computes the same formulas with separate copy/mul
 * launches.  v99.98 uses the engine operations that Aevum/Marin already expose:
 *   addsub_copy()      keeps (X+Z),(X-Z) and their square inputs in one step;
 *   xdbl_tail_uv()     fuses the two independent tail multiplications.
 * No curve mathematics or sigma sequence changes.
 */
void mont_double_fused_opt(engine* eng,
                           const MRegsOpt& r,
                           engine::Reg x,
                           engine::Reg z) {
    eng->addsub_copy(r.A, r.B, r.AA, r.BB, x, z);
    eng->square_mul(r.AA);
    eng->square_mul(r.BB);
    eng->copy(x, r.AA);
    eng->xdbl_tail_uv(x, z, r.AA, r.BB, r.ma24, r.mTmp, r.mBB);
}

/*
 * Fused Montgomery xDBLADD, algebraically identical to the legacy ladder.
 * Two independent products DA/CB are issued through mul_pair_prepared(), and
 * the double tail uses xdbl_tail_uv().
 */
void mont_dbladd_fused_opt(engine* eng,
                           const MRegsOpt& r,
                           engine::Reg x2, engine::Reg z2,
                           engine::Reg x3, engine::Reg z3) {
    eng->addsub_copy(r.A, r.B, r.AA, r.BB, x2, z2);
    eng->addsub(r.C, r.D, x3, z3);

    eng->set_multiplicand(r.mA, r.A);
    eng->set_multiplicand(r.mB, r.B);
    eng->copy(r.DA, r.D);
    eng->copy(r.CB, r.C);
    eng->mul_pair_prepared(r.DA, r.mA, r.CB, r.mB);

    eng->addsub(r.sum, r.diff, r.DA, r.CB);
    eng->copy(x3, r.sum);
    eng->copy(z3, r.diff);
    eng->square_mul(x3);
    eng->square_mul(z3);
    eng->mul(z3, r.mxdiff);

    eng->square_mul(r.AA);
    eng->square_mul(r.BB);
    eng->copy(x2, r.AA);
    eng->xdbl_tail_uv(x2, z2, r.AA, r.BB, r.ma24, r.mTmp, r.mBB);
}

void mont_init_opt(engine* eng,
                   const MRegsOpt& r,
                   const mpz_class& x,
                   const mpz_class& a24,
                   const mpz_class& n) {
    set_mont_constants_opt(eng, r, x, a24, n);
    set_reg_opt(eng, r.xa, x, n);
    eng->set(r.za, 1u);
    eng->copy(r.xb, r.xa);
    eng->copy(r.zb, r.za);
    mont_double_fused_opt(eng, r, r.xb, r.zb);
}

bool mont_ladder_fused_opt(engine* eng,
                           const MRegsOpt& r,
                           const mpz_class& scalar,
                           std::uint64_t& remaining,
                           const std::function<void(std::uint64_t)>& checkpoint,
                           const std::function<double()>& elapsed,
                           const std::string& label,
                           bool progress = true) {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(mpz_sizeinbase(scalar.get_mpz_t(), 2));
    if (bits <= 1) { remaining = 0; return true; }
    const std::uint64_t work = bits - 1;
    if (remaining == 0 || remaining > work) remaining = work;

    auto last_display = Clock::now();
    auto last_backup = Clock::now();

    while (remaining > 0) {
        if (interrupted) {
            checkpoint(remaining);
            std::cout << "Interrupted; optimized GM ECM checkpoint saved with "
                      << remaining << " fused-ladder bits remaining.\n";
            return false;
        }

        const std::uint64_t bit_index = remaining - 1;
        if (mpz_tstbit(scalar.get_mpz_t(), bit_index)) {
            mont_dbladd_fused_opt(eng, r, r.xb, r.zb, r.xa, r.za);
        } else {
            mont_dbladd_fused_opt(eng, r, r.xa, r.za, r.xb, r.zb);
        }
        --remaining;

        const auto now = Clock::now();
        if (progress &&
            (now - last_display >= std::chrono::seconds(10) || remaining == 0)) {
            eng->sync();
            const double pct =
                100.0 * static_cast<double>(work - remaining) / static_cast<double>(work);
            std::cout << std::fixed << std::setprecision(2)
                      << label << ": " << pct << "% | "
                      << (work - remaining) << "/" << work
                      << " fused ladder bits | elapsed " << elapsed() << " s\n";
            last_display = now;
        }
        if (progress && now - last_backup >= std::chrono::seconds(60)) {
            checkpoint(remaining);
            last_backup = now;
        }
    }
    return true;
}

struct PointOpt {
    mpz_class x;
    mpz_class z;
    mpz_class factor;
    bool normalized = false;
};

PointOpt project_point_opt(engine* eng,
                           const MRegsOpt& r,
                           const mpz_class& n) {
    PointOpt out;
    out.x = project_reg_opt(eng, r.xa, n);
    out.z = project_reg_opt(eng, r.za, n);

    mpz_class g = gcd_opt(out.z, n);
    if (proper_factor_opt(g, n)) { out.factor = g; return out; }

    mpz_class invz;
    if (mpz_invert(invz.get_mpz_t(), out.z.get_mpz_t(), n.get_mpz_t()) == 0) {
        out.factor = gcd_opt(out.z, n);
        return out;
    }
    out.x = mod_pos_opt(out.x * invz, n);
    out.z = 1;
    out.normalized = true;
    return out;
}

bool scalar_point_opt(engine* eng,
                      const MRegsOpt& r,
                      const mpz_class& base_x,
                      const mpz_class& a24,
                      const mpz_class& n,
                      std::uint64_t scalar) {
    if (scalar == 0) {
        eng->set(r.xa, 1u);
        eng->set(r.za, 0u);
        return true;
    }
    mont_init_opt(eng, r, base_x, a24, n);
    mpz_class k;
    mpz_import(k.get_mpz_t(), 1, 1, sizeof(scalar), 0, 0, &scalar);
    std::uint64_t rem =
        static_cast<std::uint64_t>(mpz_sizeinbase(k.get_mpz_t(), 2)) - 1;
    auto no_ckpt = [](std::uint64_t) {};
    auto elapsed = []() { return 0.0; };
    return mont_ladder_fused_opt(eng, r, k, rem, no_ckpt, elapsed, "", false);
}

/*
 * General differential xADD:
 *   out = P + Q, with D = P - Q supplied in projective x/z coordinates.
 * This is the primitive needed for the Stage-2 giant recurrence.
 */
void mont_xadd_general_opt(engine* eng,
                           const MRegsOpt& r,
                           engine::Reg x1, engine::Reg z1,
                           engine::Reg x2, engine::Reg z2,
                           engine::Reg xd, engine::Reg zd,
                           engine::Reg xo, engine::Reg zo) {
    eng->addsub(r.A, r.B, x1, z1);
    eng->addsub(r.C, r.D, x2, z2);

    eng->set_multiplicand(r.mA, r.A);
    eng->set_multiplicand(r.mB, r.B);
    eng->copy(r.DA, r.D);
    eng->copy(r.CB, r.C);
    eng->mul_pair_prepared(r.DA, r.mA, r.CB, r.mB);

    eng->addsub(r.sum, r.diff, r.DA, r.CB);
    eng->copy(xo, r.sum);
    eng->copy(zo, r.diff);
    eng->square_mul(xo);
    eng->square_mul(zo);

    eng->set_multiplicand(r.mBB, zd);
    eng->set_multiplicand(r.mTmp, xd);
    eng->mul_pair_prepared(xo, r.mBB, zo, r.mTmp);
}

std::vector<std::uint32_t> simple_primes_opt(std::uint64_t limit) {
    if (limit < 2) return {};
    if (limit > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("GM ECM optimized Stage 2 requires B2 <= 2^32-1");

    const std::uint32_t n = static_cast<std::uint32_t>(limit);
    std::vector<bool> prime(static_cast<std::size_t>(n) + 1, true);
    prime[0] = false;
    if (n >= 1) prime[1] = false;
    for (std::uint32_t q = 2;
         static_cast<std::uint64_t>(q) * q <= n; ++q) {
        if (!prime[q]) continue;
        for (std::uint64_t m = static_cast<std::uint64_t>(q) * q;
             m <= n; m += q) {
            prime[static_cast<std::size_t>(m)] = false;
        }
    }
    std::vector<std::uint32_t> out;
    for (std::uint32_t q = 2; q <= n; ++q)
        if (prime[q]) out.push_back(q);
    return out;
}

std::vector<std::uint32_t> primes_range_opt(std::uint64_t low,
                                             std::uint64_t high) {
    if (high <= low || high < 2) return {};
    if (high > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("GM ECM optimized Stage 2 requires B2 <= 2^32-1");

    const std::uint64_t root =
        static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(high))) + 1;
    const auto base = simple_primes_opt(root);
    constexpr std::uint64_t span = 4'000'000ULL;
    std::vector<std::uint32_t> out;

    std::uint64_t begin = std::max<std::uint64_t>(2, low + 1);
    for (std::uint64_t seg_low = begin; seg_low <= high; ) {
        const std::uint64_t seg_high = std::min(high, seg_low + span - 1);
        std::vector<bool> prime(static_cast<std::size_t>(seg_high - seg_low + 1), true);
        for (std::uint32_t q32 : base) {
            const std::uint64_t q = q32;
            if (q * q > seg_high) break;
            std::uint64_t first = ((seg_low + q - 1) / q) * q;
            if (first < q * q) first = q * q;
            for (std::uint64_t m = first; m <= seg_high; m += q)
                prime[static_cast<std::size_t>(m - seg_low)] = false;
        }
        for (std::uint64_t n = seg_low; n <= seg_high; ++n)
            if (prime[static_cast<std::size_t>(n - seg_low)])
                out.push_back(static_cast<std::uint32_t>(n));
        if (seg_high == high) break;
        seg_low = seg_high + 1;
    }
    return out;
}

std::uint64_t env_u64_opt(const char* name, std::uint64_t def) {
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') return def;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || v == 0) return def;
    return static_cast<std::uint64_t>(v);
}

struct Stage2Entry {
    std::uint32_t prime = 0;
    std::uint64_t k = 0;
    std::uint32_t d = 0;
};

struct Stage2Plan {
    std::uint64_t D = 0;
    std::vector<Stage2Entry> entries;
    std::vector<std::uint32_t> baby_d;
    std::uint64_t k_min = 0;
    std::uint64_t k_max = 0;
};

Stage2Plan make_stage2_plan(std::uint64_t B1, std::uint64_t B2) {
    Stage2Plan p;
    if (B2 <= B1) return p;

    std::uint64_t D = env_u64_opt("PRMERS_GM_ECM_BSGS_D",
                                  (B1 < 100 || B2 <= 10'000 ? 30ULL : 210ULL));
    if (D < 6 || (D & 1ULL) != 0 || D > 10'000)
        throw std::runtime_error("PRMERS_GM_ECM_BSGS_D must be even and in [6,10000]");

    const auto primes = primes_range_opt(B1, B2);
    p.D = D;
    std::set<std::uint32_t> babies;

    for (std::uint32_t q : primes) {
        const std::uint64_t k =
            (static_cast<std::uint64_t>(q) + D / 2) / D;
        if (k == 0) continue;
        const std::uint64_t kd = k * D;
        const std::uint64_t delta =
            kd >= q ? kd - q : static_cast<std::uint64_t>(q) - kd;
        if (delta == 0 || delta > D / 2) continue;
        if (std::gcd(delta, D) != 1) continue;
        if (delta > std::numeric_limits<std::uint32_t>::max()) continue;

        p.entries.push_back(Stage2Entry{
            q, k, static_cast<std::uint32_t>(delta)
        });
        babies.insert(static_cast<std::uint32_t>(delta));
    }

    p.baby_d.assign(babies.begin(), babies.end());
    if (!p.entries.empty()) {
        p.k_min = p.entries.front().k;
        p.k_max = p.entries.back().k;
    }

    // Keep the first production implementation safely within a modest register
    // count. D=210 needs 24 baby x-coordinates (48 registers).
    if (p.baby_d.size() > 64 && std::getenv("PRMERS_GM_ECM_BSGS_D") == nullptr) {
        // Automatic large-D selection should never reach this today; if future
        // policies change, fall back to the proven D=210 footprint.
        D = 210;
        p = Stage2Plan{};
        p.D = D;
        const auto primes2 = primes_range_opt(B1, B2);
        std::set<std::uint32_t> babies2;
        for (std::uint32_t q : primes2) {
            const std::uint64_t k =
                (static_cast<std::uint64_t>(q) + D / 2) / D;
            if (k == 0) continue;
            const std::uint64_t kd = k * D;
            const std::uint64_t delta =
                kd >= q ? kd - q : static_cast<std::uint64_t>(q) - kd;
            if (delta == 0 || delta > D / 2 || std::gcd(delta, D) != 1) continue;
            p.entries.push_back(Stage2Entry{
                q, k, static_cast<std::uint32_t>(delta)
            });
            babies2.insert(static_cast<std::uint32_t>(delta));
        }
        p.baby_d.assign(babies2.begin(), babies2.end());
        if (!p.entries.empty()) {
            p.k_min = p.entries.front().k;
            p.k_max = p.entries.back().k;
        }
    }
    return p;
}

struct OptLayout {
    std::size_t baby_base = MRegsOpt::core_count;
    std::size_t baby_count = 0;
    engine::Reg base_x = 0;
    engine::Reg base_z = 0;
    engine::Reg next_x = 0;
    engine::Reg next_z = 0;
    std::size_t count = MRegsOpt::core_count;

    explicit OptLayout(std::size_t n) : baby_count(n) {
        if (n == 0) {
            count = MRegsOpt::core_count;
            return;
        }
        const std::size_t after_babies = baby_base + 2 * n;
        base_x = static_cast<engine::Reg>(after_babies);
        base_z = static_cast<engine::Reg>(after_babies + 1);
        next_x = static_cast<engine::Reg>(after_babies + 2);
        next_z = static_cast<engine::Reg>(after_babies + 3);
        count = after_babies + 4;
    }

    engine::Reg baby_x(std::size_t i) const {
        return static_cast<engine::Reg>(baby_base + 2 * i);
    }
    engine::Reg baby_z(std::size_t i) const {
        return static_cast<engine::Reg>(baby_base + 2 * i + 1);
    }
};

struct OptCheckpointHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t phase;
    std::uint32_t p;
    std::uint32_t lift;
    std::uint32_t curve;
    std::uint32_t baby_count;
    std::uint64_t B1;
    std::uint64_t B2;
    std::uint64_t sigma;
    std::uint64_t D;
    std::uint64_t token;
    std::uint64_t aux;
    double elapsed;
    std::uint64_t checkpoint_bytes;
};

bool load_opt_checkpoint(const std::filesystem::path& path,
                         engine* eng,
                         const OptTarget& t,
                         std::uint32_t phase,
                         std::uint32_t curve,
                         std::uint64_t B1,
                         std::uint64_t B2,
                         std::uint64_t sigma,
                         std::uint64_t D,
                         std::size_t baby_count,
                         std::uint64_t& token,
                         std::uint64_t& aux,
                         double& restored) {
    File f(path.string());
    if (!f.exists()) return false;
    OptCheckpointHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (!std::equal(GM_OPT_MAGIC.begin(), GM_OPT_MAGIC.end(), h.magic) ||
        h.version != GM_OPT_CHECKPOINT_VERSION ||
        h.phase != phase || h.p != t.p || h.lift != t.lift ||
        h.curve != curve || h.B1 != B1 || h.B2 != B2 ||
        h.sigma != sigma || h.D != D ||
        h.baby_count != baby_count ||
        h.checkpoint_bytes != eng->get_checkpoint_size()) return false;

    std::vector<char> data(eng->get_checkpoint_size());
    if (!f.read(data.data(), data.size()) ||
        !f.check_crc32() ||
        !eng->set_checkpoint(data)) return false;

    token = h.token;
    aux = h.aux;
    restored = h.elapsed;
    return true;
}

void save_opt_checkpoint(const std::filesystem::path& path,
                         engine* eng,
                         const OptTarget& t,
                         std::uint32_t phase,
                         std::uint32_t curve,
                         std::uint64_t B1,
                         std::uint64_t B2,
                         std::uint64_t sigma,
                         std::uint64_t D,
                         std::size_t baby_count,
                         std::uint64_t token,
                         std::uint64_t aux,
                         double elapsed) {
    eng->sync();

    OptCheckpointHeader h{};
    std::copy(GM_OPT_MAGIC.begin(), GM_OPT_MAGIC.end(), h.magic);
    h.version = GM_OPT_CHECKPOINT_VERSION;
    h.phase = phase;
    h.p = t.p;
    h.lift = t.lift;
    h.curve = curve;
    h.baby_count = static_cast<std::uint32_t>(baby_count);
    h.B1 = B1;
    h.B2 = B2;
    h.sigma = sigma;
    h.D = D;
    h.token = token;
    h.aux = aux;
    h.elapsed = elapsed;
    h.checkpoint_bytes = eng->get_checkpoint_size();

    std::vector<char> data(eng->get_checkpoint_size());
    if (!eng->get_checkpoint(data))
        throw std::runtime_error("cannot capture optimized GM ECM checkpoint");

    const std::filesystem::path np = path.string() + ".new";
    const std::filesystem::path op = path.string() + ".old";
    {
        File f(np.string(), "wb");
        if (!f.write(reinterpret_cast<const char*>(&h), sizeof(h)) ||
            !f.write(data.data(), data.size()))
            throw std::runtime_error("cannot write optimized GM ECM checkpoint");
        f.write_crc32();
    }

    std::error_code ec;
    std::filesystem::remove(op, ec);
    ec.clear();
    if (std::filesystem::exists(path)) std::filesystem::rename(path, op, ec);
    ec.clear();
    std::filesystem::rename(np, path, ec);
    if (ec)
        throw std::runtime_error("cannot install optimized GM ECM checkpoint: " +
                                 ec.message());
}

void clear_opt_checkpoint(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".old", ec);
    std::filesystem::remove(path.string() + ".new", ec);
}

std::size_t baby_index_opt(const Stage2Plan& plan, std::uint32_t d) {
    const auto it = std::lower_bound(plan.baby_d.begin(), plan.baby_d.end(), d);
    if (it == plan.baby_d.end() || *it != d)
        throw std::runtime_error("internal GM ECM BSGS baby lookup failure");
    return static_cast<std::size_t>(it - plan.baby_d.begin());
}

} // namespace

namespace core {

int App::runGaussianMersenneECMOptimized() {
    // Keep every unusual/safety mode on the already-proven implementation.
    // The optimized path is deliberately narrow until it has production hours.
    if (options.gm_safe_replay) {
        std::cout << "[GM ECM v99.98] -gm-safe requested; using legacy replay path.\n";
        return runGaussianMersenneECMLegacy();
    }
    if (!options.notorsion || options.torsion16) {
        std::cout << "[GM ECM v99.98] explicit torsion family requested; "
                     "using legacy path.\n";
        return runGaussianMersenneECMLegacy();
    }
    if (options.gm_sieve_limit != 0) {
        std::cout << "[GM ECM v99.98] non-zero -gm-sieve keeps the legacy "
                     "sieve+ECM path. Use -gm-sieve 0 for fused/BSGS mode.\n";
        return runGaussianMersenneECMLegacy();
    }

    std::string family = options.gm_family;
    std::transform(family.begin(), family.end(), family.begin(),
                   [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
    if (family == "BOTH") {
        const std::string saved = options.gm_family;
        options.gm_family = "GM";
        const int gm_rc = runGaussianMersenneECMOptimized();
        if (interrupted) { options.gm_family = saved; return gm_rc; }
        options.gm_family = "GQ";
        const int gq_rc = runGaussianMersenneECMOptimized();
        options.gm_family = saved;
        if (gm_rc == 2 || gq_rc == 2) return 2;
        return (gm_rc == 0 && gq_rc == 0) ? 0 : 1;
    }

    OptTarget t;
    try { t = make_opt_target(options.exponent, family); }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 2;
    }

    const std::uint64_t B1 = options.B1 != 0 ? options.B1 : 50000ULL;
    const std::uint64_t B2 = options.B2;
    const std::uint64_t curves =
        options.K != 0 ? options.K : (options.nmax != 0 ? options.nmax : 20ULL);
    const std::filesystem::path save_dir =
        options.save_path.empty() ? "." : options.save_path;
    std::filesystem::create_directories(save_dir);

    Stage2Plan s2plan;
    try { s2plan = make_stage2_plan(B1, B2); }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 2;
    }

    const OptLayout layout(s2plan.baby_d.size());
    const mpz_class K = buildE(B1);
    const std::uint64_t kbits =
        static_cast<std::uint64_t>(mpz_sizeinbase(K.get_mpz_t(), 2));

    const auto job_start = Clock::now();
    auto job_elapsed = [&]() {
        return std::chrono::duration<double>(Clock::now() - job_start).count();
    };

    std::cout << "Gaussian pair ECM factoring\n"
              << "  p              : " << t.p << "\n"
              << "  target family  : " << t.family << "\n"
              << "  lift exponent  : " << t.lift << "\n"
              << "  B1 / B2        : " << B1 << " / " << B2 << "\n"
              << "  curves         : " << curves << "\n"
              << "  Stage1 engine  : Montgomery fused ladder (v99.98)\n"
              << "  Stage1 bits    : " << kbits << "\n"
              << "  curve stream   : exact legacy Suyama sigma sequence\n";

    if (B2 > B1) {
        std::cout << "  Stage2 engine  : Montgomery baby-step/giant-step\n"
                  << "  Stage2 D       : " << s2plan.D << "\n"
                  << "  Stage2 primes  : " << s2plan.entries.size() << "\n"
                  << "  baby residues  : " << s2plan.baby_d.size() << "\n";
    } else {
        std::cout << "  Stage2 engine  : disabled (B2 <= B1)\n";
    }
    std::cout << "  registers      : " << layout.count << "\n";

    std::unique_ptr<engine> eng(
        engine::create_gpu(t.lift, layout.count,
                           static_cast<std::size_t>(options.device_id), true));
    if (!eng) {
        std::cerr << "[GM ECM v99.98] GPU engine unavailable.\n";
        return 2;
    }
    const std::string backend = eng->is_aevum_backend() ? "Aevum" : "Marin";
    std::cout << "  backend        : " << backend << "\n"
              << "  transform      : " << eng->get_size() << " words\n";

    // Checkpoints include all registers. Avoid uninitialized/zero scratch
    // diagnostics before the first save.
    for (std::size_t i = 0; i < layout.count; ++i)
        eng->set(static_cast<engine::Reg>(i), 1u);

    const MRegsOpt r;
    const std::uint64_t base_seed =
        options.curve_seed != 0 ? options.curve_seed : 0x474d45434d763938ULL;

    for (std::uint64_t curve = 0; curve < curves; ++curve) {
        std::uint64_t sigma = 0;
        if (!options.sigma.empty() && curve == 0) {
            try { sigma = std::stoull(options.sigma); }
            catch (...) {
                std::cerr << "Invalid -sigma for Gaussian ECM; expected unsigned 64-bit integer.\n";
                return 2;
            }
        } else {
            sigma = 6 + (splitmix64_opt(base_seed + curve) %
                         0x7ffffffffffffff0ULL);
        }

        std::cout << "\n[GM ECM] curve " << (curve + 1) << "/" << curves
                  << " sigma=" << sigma << "\n";

        SuyamaSetupOpt setup = make_suyama_opt(t.n, sigma);
        if (proper_factor_opt(setup.factor, t.n)) {
            std::cout << ">>> Gaussian pair ECM setup factor: "
                      << setup.factor << "\n";
            write_opt_result(save_dir, t, B1, B2, curves, curve + 1, sigma,
                             0, setup.factor, backend, options.device_id,
                             job_elapsed());
            return 0;
        }
        if (!setup.ok) {
            std::cout << "[GM ECM v99.98] singular setup; next curve.\n";
            continue;
        }

        const std::filesystem::path s1ck =
            save_dir / ((t.family == "GQ" ? "gq" : "gm") +
                        std::string("_ecm_p") + std::to_string(t.p) +
                        "_c" + std::to_string(curve) + "_stage1_fused.ckpt");

        std::uint64_t remaining = kbits > 0 ? kbits - 1 : 0;
        std::uint64_t unused_aux = 0;
        double restored = 0.0;
        bool resumed_s1 = false;

        if (options.resume) {
            resumed_s1 = load_opt_checkpoint(
                s1ck, eng.get(), t, 1, static_cast<std::uint32_t>(curve),
                B1, B2, sigma, s2plan.D, s2plan.baby_d.size(),
                remaining, unused_aux, restored);
            if (!resumed_s1) {
                resumed_s1 = load_opt_checkpoint(
                    s1ck.string() + ".old", eng.get(), t, 1,
                    static_cast<std::uint32_t>(curve), B1, B2, sigma,
                    s2plan.D, s2plan.baby_d.size(),
                    remaining, unused_aux, restored);
            }
        }

        if (!resumed_s1) {
            mont_init_opt(eng.get(), r, setup.x, setup.a24, t.n);
            remaining = kbits > 0 ? kbits - 1 : 0;
        } else {
            std::cout << "[GM ECM v99.98] resuming fused Stage1 with "
                      << remaining << " bits remaining.\n";
        }

        const auto curve_start = Clock::now();
        auto elapsed = [&]() {
            return restored +
                std::chrono::duration<double>(Clock::now() - curve_start).count();
        };
        auto save_s1 = [&](std::uint64_t rem) {
            save_opt_checkpoint(
                s1ck, eng.get(), t, 1, static_cast<std::uint32_t>(curve),
                B1, B2, sigma, s2plan.D, s2plan.baby_d.size(),
                rem, 0, elapsed());
        };

        if (!mont_ladder_fused_opt(
                eng.get(), r, K, remaining, save_s1, elapsed,
                "GM ECM Stage 1 fused curve " + std::to_string(curve + 1))) {
            return 0;
        }

        clear_opt_checkpoint(s1ck);
        eng->sync();

        PointOpt Q = project_point_opt(eng.get(), r, t.n);
        if (proper_factor_opt(Q.factor, t.n)) {
            std::cout << ">>> Gaussian pair ECM Stage 1 factor: "
                      << Q.factor << "\n";
            write_opt_result(save_dir, t, B1, B2, curves, curve + 1, sigma,
                             1, Q.factor, backend, options.device_id,
                             job_elapsed());
            return 0;
        }
        if (!Q.normalized) {
            std::cout << "[GM ECM v99.98] Stage1 singular/trivial point; next curve.\n";
            continue;
        }

        std::cout << "[GM ECM] Stage 1 no factor | x low64=0x"
                  << low_hex_opt(Q.x)
                  << " | elapsed=" << std::fixed << std::setprecision(2)
                  << elapsed() << " s\n";

        if (B2 <= B1 || s2plan.entries.empty()) {
            if (B2 > B1)
                std::cout << "[GM ECM Stage 2 BSGS] no usable Stage2 primes.\n";
            continue;
        }

        const std::filesystem::path s2ck =
            save_dir / ((t.family == "GQ" ? "gq" : "gm") +
                        std::string("_ecm_p") + std::to_string(t.p) +
                        "_c" + std::to_string(curve) + "_stage2_bsgs.ckpt");

        std::uint64_t current_k = s2plan.k_min;
        std::uint64_t terms_since_gcd = 0;
        double s2_restored = 0.0;
        bool resumed_s2 = false;

        if (options.resume) {
            resumed_s2 = load_opt_checkpoint(
                s2ck, eng.get(), t, 2, static_cast<std::uint32_t>(curve),
                B1, B2, sigma, s2plan.D, s2plan.baby_d.size(),
                current_k, terms_since_gcd, s2_restored);
            if (!resumed_s2) {
                resumed_s2 = load_opt_checkpoint(
                    s2ck.string() + ".old", eng.get(), t, 2,
                    static_cast<std::uint32_t>(curve), B1, B2, sigma,
                    s2plan.D, s2plan.baby_d.size(),
                    current_k, terms_since_gcd, s2_restored);
            }
        }

        // r.E is the Stage2 product accumulator. r.AA/r.BB are cross-product
        // scratch and r.tmp becomes the prepared cross term.
        const engine::Reg ACC = r.E;
        const engine::Reg CROSS1 = r.AA;
        const engine::Reg CROSS2 = r.BB;
        const engine::Reg MCROSS = r.tmp;

        if (!resumed_s2) {
            std::cout << "[GM ECM Stage 2 BSGS] precomputing "
                      << s2plan.baby_d.size() << " baby points...\n";

            // Baby coordinates are stored directly in multiplicand form.
            for (std::size_t i = 0; i < s2plan.baby_d.size(); ++i) {
                const std::uint64_t d = s2plan.baby_d[i];
                if (!scalar_point_opt(eng.get(), r, Q.x, setup.a24, t.n, d))
                    return 0;
                eng->copy(layout.baby_x(i), r.xa);
                eng->copy(layout.baby_z(i), r.za);
                eng->set_multiplicand(layout.baby_x(i), layout.baby_x(i));
                eng->set_multiplicand(layout.baby_z(i), layout.baby_z(i));
            }

            // R = [D]Q.
            if (!scalar_point_opt(eng.get(), r, Q.x, setup.a24, t.n, s2plan.D))
                return 0;
            eng->copy(layout.base_x, r.xa);
            eng->copy(layout.base_z, r.za);

            // Temporarily keep [k_min D]Q in next_x/next_z.
            if (s2plan.k_min >
                std::numeric_limits<std::uint64_t>::max() / s2plan.D)
                throw std::runtime_error("GM ECM BSGS giant scalar overflow");
            const std::uint64_t s0 = s2plan.k_min * s2plan.D;
            if (!scalar_point_opt(eng.get(), r, Q.x, setup.a24, t.n, s0))
                return 0;
            eng->copy(layout.next_x, r.xa);
            eng->copy(layout.next_z, r.za);

            // [((k_min+1)D)]Q becomes current+1 in xb/zb.
            if (s2plan.k_min + 1 >
                std::numeric_limits<std::uint64_t>::max() / s2plan.D)
                throw std::runtime_error("GM ECM BSGS giant scalar overflow");
            const std::uint64_t s1 = (s2plan.k_min + 1) * s2plan.D;
            if (!scalar_point_opt(eng.get(), r, Q.x, setup.a24, t.n, s1))
                return 0;
            eng->copy(r.xb, r.xa);
            eng->copy(r.zb, r.za);

            // Restore current giant [k_min D]Q to xa/za.
            eng->copy(r.xa, layout.next_x);
            eng->copy(r.za, layout.next_z);

            eng->set(ACC, 1u);
            current_k = s2plan.k_min;
            terms_since_gcd = 0;
        } else {
            std::cout << "[GM ECM Stage 2 BSGS] resuming at giant k="
                      << current_k << ", pending product terms="
                      << terms_since_gcd << ".\n";
        }

        const std::uint64_t gcd_batch =
            std::max<std::uint64_t>(16,
                env_u64_opt("PRMERS_GM_ECM_BSGS_GCD_BATCH", 256));
        const auto s2_start = Clock::now();
        auto s2_elapsed = [&]() {
            return s2_restored +
                std::chrono::duration<double>(Clock::now() - s2_start).count();
        };
        auto last_report = Clock::now();
        auto last_save = Clock::now();

        std::size_t entry_index = static_cast<std::size_t>(
            std::lower_bound(
                s2plan.entries.begin(), s2plan.entries.end(), current_k,
                [](const Stage2Entry& e, std::uint64_t k){ return e.k < k; })
            - s2plan.entries.begin());

        auto flush_gcd = [&]() -> int {
            if (terms_since_gcd == 0) return 0;
            eng->sync();
            const mpz_class acc = project_reg_opt(eng.get(), ACC, t.n);
            const mpz_class g = gcd_opt(acc, t.n);
            if (proper_factor_opt(g, t.n)) {
                std::cout << "\n>>> Gaussian pair ECM Stage 2 factor: "
                          << g << "\n";
                write_opt_result(save_dir, t, B1, B2, curves,
                                 curve + 1, sigma, 2, g, backend,
                                 options.device_id, job_elapsed());
                return 1;
            }
            if (g == t.n) {
                std::cout << "\n[GM ECM Stage 2 BSGS] gcd=target inside a "
                             "product batch; falling back to the legacy "
                             "Stage2 for deterministic isolation.\n";
                return 2;
            }
            eng->set(ACC, 1u);
            terms_since_gcd = 0;
            return 0;
        };

        while (current_k <= s2plan.k_max) {
            if (interrupted) {
                save_opt_checkpoint(
                    s2ck, eng.get(), t, 2, static_cast<std::uint32_t>(curve),
                    B1, B2, sigma, s2plan.D, s2plan.baby_d.size(),
                    current_k, terms_since_gcd, s2_elapsed());
                std::cout << "\nInterrupted; BSGS checkpoint saved at k="
                          << current_k << ".\n";
                return 0;
            }

            // All stage2 primes represented by q = kD +/- d use the same
            // current giant [kD]Q.
            while (entry_index < s2plan.entries.size() &&
                   s2plan.entries[entry_index].k == current_k) {
                const Stage2Entry& e = s2plan.entries[entry_index];
                const std::size_t bi = baby_index_opt(s2plan, e.d);

                // CROSS1 = Xg*Zb - Zg*Xb.
                eng->copy(CROSS1, r.xa);
                eng->copy(CROSS2, r.za);
                eng->mul_pair_prepared(
                    CROSS1, layout.baby_z(bi),
                    CROSS2, layout.baby_x(bi));
                eng->sub_reg(CROSS1, CROSS2);

                eng->set_multiplicand(MCROSS, CROSS1);
                eng->mul(ACC, MCROSS);
                ++terms_since_gcd;
                ++entry_index;

                if (terms_since_gcd >= gcd_batch) {
                    const int fr = flush_gcd();
                    if (fr == 1) {
                        clear_opt_checkpoint(s2ck);
                        return 0;
                    }
                    if (fr == 2) {
                        clear_opt_checkpoint(s2ck);
                        return runGaussianMersenneECMLegacy();
                    }
                }
            }

            // Advance giant pair:
            // xa/za=[kD]Q, xb/zb=[(k+1)D]Q, base=[D]Q.
            // next = curr + base, with difference prev.
            if (current_k < s2plan.k_max) {
                mont_xadd_general_opt(
                    eng.get(), r,
                    r.xb, r.zb,
                    layout.base_x, layout.base_z,
                    r.xa, r.za,
                    layout.next_x, layout.next_z);
                eng->copy(r.xa, r.xb);
                eng->copy(r.za, r.zb);
                eng->copy(r.xb, layout.next_x);
                eng->copy(r.zb, layout.next_z);
            }
            ++current_k;

            const auto now = Clock::now();
            if (now - last_report >= std::chrono::seconds(10) ||
                current_k > s2plan.k_max) {
                eng->sync();
                const double frac = s2plan.entries.empty() ? 1.0 :
                    static_cast<double>(entry_index) /
                    static_cast<double>(s2plan.entries.size());
                const double eta = frac > 0.0 ?
                    s2_elapsed() * (1.0 - frac) / frac : 0.0;
                std::cout << "\rGM ECM Stage 2 BSGS curve " << (curve + 1)
                          << ": " << std::fixed << std::setprecision(2)
                          << (100.0 * frac) << "% | primes "
                          << entry_index << "/" << s2plan.entries.size()
                          << " | giant k " << current_k << "/"
                          << (s2plan.k_max + 1)
                          << " | elapsed " << s2_elapsed() << " s"
                          << " | ETA " << eta << " s" << std::flush;
                last_report = now;
            }

            if (now - last_save >= std::chrono::seconds(
                    options.backup_interval > 0 ? options.backup_interval : 120)) {
                save_opt_checkpoint(
                    s2ck, eng.get(), t, 2, static_cast<std::uint32_t>(curve),
                    B1, B2, sigma, s2plan.D, s2plan.baby_d.size(),
                    current_k, terms_since_gcd, s2_elapsed());
                last_save = now;
            }
        }

        const int fr = flush_gcd();
        std::cout << "\n";
        if (fr == 1) {
            clear_opt_checkpoint(s2ck);
            return 0;
        }
        if (fr == 2) {
            clear_opt_checkpoint(s2ck);
            return runGaussianMersenneECMLegacy();
        }

        clear_opt_checkpoint(s2ck);
        std::cout << "[GM ECM Stage 2 BSGS] curve " << (curve + 1)
                  << " no factor through B2=" << B2
                  << " | elapsed=" << s2_elapsed() << " s\n";
    }

    std::cout << "No Gaussian pair ECM factor found in "
              << curves << " curve(s).\n";
    write_opt_result(save_dir, t, B1, B2, curves,
                     std::nullopt, std::nullopt,
                     B2 > B1 ? 2 : 1,
                     std::nullopt, backend, options.device_id,
                     job_elapsed());
    return 1;
}

} // namespace core
