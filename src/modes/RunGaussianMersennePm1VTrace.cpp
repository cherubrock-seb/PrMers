// PrMers v100.13 RC1 - Gaussian-Mersenne P-1 V-trace lift
//
// Default fresh GM/GQ P-1 jobs with B2>B1 use a denominator-free V-trace
// Stage 2 in Z/(2^(4p)-1), with all factor decisions projected to the exact
// selected Gaussian norm.  The v100.12 product-exponent implementation remains
// available byte-for-byte through runGaussianMersennePM1Legacy() and is used
// automatically for resume/safe-replay/classic requests.
//
// Mathematical contract:
//   N (GM or GQ target) divides L = 2^(4p)-1.
//   H is the Stage-1 residue in Z/LZ.
//   h = H mod N.  If gcd(h,N)=1, compute hinv = h^-1 mod N on CPU and inject
//   that representative into Z/LZ.  It need not invert H modulo L.
//   V_1 = H + hinv, V_{n+1}=V_1 V_n - V_{n-1}.
//   Projection modulo N gives V_n = h^n + h^-n.
//   For q=kD +/- j, V_{kD}-V_j contains the P-1 condition h^q-1.
// Thus one trace term can cover the symmetric prime pair kD-j, kD+j.
//
// Copyright 2026 cherubrock-seb
// SPDX-License-Identifier: MIT

#include "core/App.hpp"
#include "core/AlgoUtils.hpp"
#include "core/Version.hpp"
#include "marin/engine.h"

#include <gmpxx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
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
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using core::algo::buildE;
using core::algo::interrupted;

struct GmVTarget {
    std::string family;
    std::uint32_t p = 0;
    std::uint32_t lift = 0;
    std::uint64_t middle = 0;
    int chi = 0;
    mpz_class n;
};

static std::string upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

static std::uint64_t add_mod_u64(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
    return a >= mod - b ? a - (mod - b) : a + b;
}

static std::uint64_t mul_mod_u64(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
#if defined(__SIZEOF_INT128__)
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(a % mod) * static_cast<unsigned __int128>(b % mod)) % mod);
#else
    a %= mod;
    b %= mod;
    std::uint64_t out = 0;
    while (b) {
        if (b & 1U) out = add_mod_u64(out, a, mod);
        b >>= 1U;
        if (b) a = add_mod_u64(a, a, mod);
    }
    return out;
#endif
}

static std::uint64_t pow_mod_u64(std::uint64_t a, std::uint64_t e, std::uint64_t mod) {
    std::uint64_t out = 1 % mod;
    while (e) {
        if (e & 1U) out = mul_mod_u64(out, a, mod);
        e >>= 1U;
        if (e) a = mul_mod_u64(a, a, mod);
    }
    return out;
}

static bool is_prime_u64(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint32_t p : {2U, 3U, 5U, 7U, 11U, 13U, 17U, 19U, 23U, 29U, 31U, 37U}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    std::uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1U) == 0) {
        d >>= 1U;
        ++s;
    }
    constexpr std::array<std::uint64_t, 7> bases{
        2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL};
    for (std::uint64_t a : bases) {
        if (a % n == 0) continue;
        std::uint64_t x = pow_mod_u64(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witnessed_composite = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n - 1) {
                witnessed_composite = false;
                break;
            }
        }
        if (witnessed_composite) return false;
    }
    return true;
}

static GmVTarget make_target(std::uint64_t p64, std::string family) {
    family = upper_copy(family);
    if (family != "GM" && family != "GQ")
        throw std::runtime_error("GM V-trace requires -gm-family GM or GQ");
    if (p64 < 3 || (p64 & 1U) == 0 || !is_prime_u64(p64))
        throw std::runtime_error("Gaussian pair factoring requires an odd prime exponent p >= 3");
    if (p64 > std::numeric_limits<std::uint32_t>::max() / 4ULL)
        throw std::runtime_error("Gaussian lift requires 4p <= 2^32-1");

    GmVTarget t;
    t.family = family;
    t.p = static_cast<std::uint32_t>(p64);
    t.lift = static_cast<std::uint32_t>(4ULL * p64);
    t.middle = (p64 + 1ULL) / 2ULL;
    const std::uint64_t r = p64 & 7ULL;
    t.chi = (r == 1 || r == 7) ? 1 : -1;

    t.n = mpz_class(1) << p64;
    const mpz_class mid = mpz_class(1) << t.middle;
    const bool gq = family == "GQ";
    if ((!gq && t.chi > 0) || (gq && t.chi < 0)) t.n -= mid;
    else t.n += mid;
    t.n += 1;
    if (gq) {
        if (!mpz_divisible_ui_p(t.n.get_mpz_t(), 5))
            throw std::runtime_error("GQ numerator is not divisible by 5");
        mpz_divexact_ui(t.n.get_mpz_t(), t.n.get_mpz_t(), 5);
    }
    return t;
}

static mpz_class mod_positive(mpz_class x, const mpz_class& n) {
    x %= n;
    if (x < 0) x += n;
    return x;
}

static void set_reg_mpz(engine* eng, engine::Reg reg, const mpz_class& value) {
    mpz_t z;
    mpz_init_set(z, value.get_mpz_t());
    eng->set_mpz(reg, z);
    mpz_clear(z);
}

static mpz_class get_reg_mpz(engine* eng, engine::Reg reg) {
    mpz_t z;
    mpz_init(z);
    eng->get_mpz(z, reg);
    mpz_class out(z);
    mpz_clear(z);
    return out;
}

static mpz_class project_reg(engine* eng, engine::Reg reg, const mpz_class& n) {
    return mod_positive(get_reg_mpz(eng, reg), n);
}

static mpz_class proper_gcd(const mpz_class& a, const mpz_class& n) {
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), n.get_mpz_t());
    return g;
}

static bool proper_factor(const mpz_class& g, const mpz_class& n) {
    return g > 1 && g < n;
}

static std::vector<std::uint32_t> simple_primes(std::uint64_t limit) {
    if (limit < 2) return {};
    if (limit > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("GM V-trace currently requires B2 <= 2^32-1");
    const std::uint32_t n = static_cast<std::uint32_t>(limit);
    std::vector<bool> prime(static_cast<std::size_t>(n) + 1, true);
    prime[0] = false;
    if (n >= 1) prime[1] = false;
    for (std::uint32_t q = 2; static_cast<std::uint64_t>(q) * q <= n; ++q) {
        if (!prime[q]) continue;
        for (std::uint64_t m = static_cast<std::uint64_t>(q) * q; m <= n; m += q)
            prime[static_cast<std::size_t>(m)] = false;
    }
    std::vector<std::uint32_t> out;
    for (std::uint32_t q = 2; q <= n; ++q)
        if (prime[q]) out.push_back(q);
    return out;
}

static std::vector<std::uint32_t> primes_in_range(std::uint64_t low, std::uint64_t high) {
    if (high <= low || high < 2) return {};
    if (high > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("GM V-trace currently requires B2 <= 2^32-1");

    const std::uint64_t root =
        static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(high))) + 1;
    const auto base = simple_primes(root);
    constexpr std::uint64_t span = 4'000'000ULL;
    std::vector<std::uint32_t> out;

    std::uint64_t begin = std::max<std::uint64_t>(2, low + 1);
    for (std::uint64_t seg_low = begin; seg_low <= high;) {
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

static std::uint64_t find_admissible_small_factor(const GmVTarget& t, std::uint64_t limit) {
    if (limit < 5) return 0;
    if (mpz_divisible_ui_p(t.n.get_mpz_t(), 5) && mpz_cmp_ui(t.n.get_mpz_t(), 5) != 0)
        return 5;
    const std::uint64_t step = 4ULL * static_cast<std::uint64_t>(t.p);
    if (step > limit) return 0;
    const std::uint64_t max_k = (limit - 1) / step;
    if (max_k > 100'000'000ULL)
        throw std::runtime_error("-gm-sieve would inspect more than 100000000 candidates");
    for (std::uint64_t k = 1; k <= max_k; ++k) {
        const std::uint64_t q = step * k + 1;
        if (q > limit) break;
        if (!is_prime_u64(q)) continue;
#if ULONG_MAX >= UINT64_MAX
        if (mpz_divisible_ui_p(t.n.get_mpz_t(), static_cast<unsigned long>(q)) &&
            mpz_cmp_ui(t.n.get_mpz_t(), static_cast<unsigned long>(q)) != 0)
            return q;
#else
        mpz_class qz;
        mpz_import(qz.get_mpz_t(), 1, -1, sizeof(q), 0, 0, &q);
        if (mpz_divisible_p(t.n.get_mpz_t(), qz.get_mpz_t())) return q;
#endif
    }
    return 0;
}

static std::string low_hex(const mpz_class& x, unsigned bits = 64) {
    mpz_class low;
    mpz_fdiv_r_2exp(low.get_mpz_t(), x.get_mpz_t(), bits);
    std::string s = low.get_str(16);
    const std::size_t w = bits / 4;
    if (s.size() < w) s.insert(0, w - s.size(), '0');
    return s;
}

static std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0' || parsed == 0) return fallback;
    return static_cast<std::uint64_t>(parsed);
}

static bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    const std::string s(v);
    return s != "0" && s != "off" && s != "OFF" && s != "false" && s != "FALSE";
}

static std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"': o << "\\\""; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
                } else {
                    o << static_cast<char>(c);
                }
        }
    }
    return o.str();
}

static std::string iso8601_utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

static void write_result(const std::filesystem::path& dir,
                         const GmVTarget& t,
                         const char* outcome,
                         int stage,
                         std::uint64_t B1,
                         std::uint64_t B2,
                         const std::string& backend,
                         std::uint64_t D,
                         double elapsed,
                         int device_id,
                         const std::optional<std::string>& factor) {
    std::filesystem::create_directories(dir);
    const std::string prefix = t.family == "GQ" ? "gq" : "gm";
    const std::filesystem::path file =
        dir / (prefix + "_pm1_p" + std::to_string(t.p) + "_result.json");

    std::ostringstream j;
    j << std::fixed << std::setprecision(3);
    j << "{\n"
      << "  \"schema_version\": 2,\n"
      << "  \"program\": \"PrMers\",\n"
      << "  \"program_version\": \"v100.13-rc1\",\n"
      << "  \"program_build\": \"" << json_escape(core::PRMERS_VERSION) << "\",\n"
      << "  \"family\": \"gaussian-pair\",\n"
      << "  \"target_family\": \"" << t.family << "\",\n"
      << "  \"mode\": \"gm-pm1\",\n"
      << "  \"outcome\": \"" << json_escape(outcome) << "\",\n"
      << "  \"stage\": " << stage << ",\n"
      << "  \"exponent\": " << t.p << ",\n"
      << "  \"B1\": \"" << B1 << "\",\n"
      << "  \"B2\": \"" << B2 << "\",\n"
      << "  \"curves\": null,\n"
      << "  \"curve\": null,\n"
      << "  \"sigma\": null,\n"
      << "  \"factor\": ";
    if (factor) j << "\"" << json_escape(*factor) << "\"";
    else j << "null";
    j << ",\n"
      << "  \"factor_source\": \"vtrace-lift D=" << D << "\",\n"
      << "  \"backend\": \"" << json_escape(backend) << "\",\n"
      << "  \"device\": \"device " << device_id << "\",\n"
      << "  \"elapsed_seconds\": " << std::max(0.0, elapsed) << ",\n"
      << "  \"timestamp\": \"" << iso8601_utc_now() << "\"\n"
      << "}";

    {
        std::ofstream out(file);
        out << j.str() << '\n';
    }
    {
        std::ofstream out(dir / "results.txt", std::ios::app);
        out << j.str() << '\n';
    }
    std::cout << "Result file: " << file << "\n";
}

struct TraceTask {
    std::uint64_t k = 0;
    std::uint64_t j = 0;
    bool operator<(const TraceTask& other) const {
        return k < other.k || (k == other.k && j < other.j);
    }
};

struct VRegs {
    static constexpr engine::Reg ACC = 0;
    static constexpr engine::Reg V1 = 1;
    static constexpr engine::Reg VD = 2;
    static constexpr engine::Reg GPREV = 3;
    static constexpr engine::Reg GCUR = 4;
    static constexpr engine::Reg GNEXT = 5;
    static constexpr engine::Reg BPREV = 6;
    static constexpr engine::Reg BCUR = 7;
    static constexpr engine::Reg BNEXT = 8;
    static constexpr engine::Reg TMP = 9;
    static constexpr engine::Reg MUL_V1 = 10;
    static constexpr engine::Reg MUL_VD = 11;
    static constexpr engine::Reg MUL_TMP = 12;
    static constexpr std::size_t BABY_BASE = 13;
};

static std::uint64_t choose_default_D(std::uint64_t B1, std::uint64_t B2) {
    const std::uint64_t forced_env = env_u64("PRMERS_GM_VTRACE_D", 0);
    if (forced_env) return forced_env;
    // D=630 gives 72 coprime half-residues: comfortably below 1 GiB at the
    // 21M-25M GMNet lift sizes while reducing giant steps to ~B2/630.
    for (std::uint64_t d : {630ULL, 420ULL, 210ULL, 90ULL, 30ULL}) {
        if (d <= B1 && d <= B2) return d;
    }
    return 0;
}

static double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

namespace core {

int App::runGaussianMersennePM1() {
    const std::string family = upper_copy(options.gm_family);

    if (family == "BOTH") {
        const std::string saved = options.gm_family;
        options.gm_family = "GM";
        const int gm_rc = runGaussianMersennePM1();
        if (interrupted) {
            options.gm_family = saved;
            return gm_rc;
        }
        options.gm_family = "GQ";
        const int gq_rc = runGaussianMersennePM1();
        options.gm_family = saved;
        if (gm_rc == 2 || gq_rc == 2) return 2;
        return (gm_rc == 0 && gq_rc == 0) ? 0 : 1;
    }

    const std::uint64_t B1 = options.B1 != 0 ? options.B1 : 100000ULL;
    const std::uint64_t B2 = options.B2;

    // Preserve explicit safe/legacy paths. CliOptions::resume is true by
    // default in PrMers, so it must NOT be used as a fresh-vs-resume signal.
    if (B2 <= B1 ||
        options.gm_safe_replay ||
        options.pm1_vtrace_off ||
        env_truthy("PRMERS_GM_PM1_PRODUCT_STAGE2")) {
        if (B2 > B1) {
            std::cout << "[GM-PM1] v100.13 fast V-trace bypassed: "
                      << (options.gm_safe_replay ? "-gm-safe requested"
                          : options.pm1_vtrace_off ? "-pm1-vtrace-off requested"
                          : "PRMERS_GM_PM1_PRODUCT_STAGE2 requested")
                      << ". Using v100.12 product-exponent implementation.\n";
        }
        return runGaussianMersennePM1Legacy();
    }

    GmVTarget t;
    try {
        t = make_target(options.exponent, family);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 2;
    }

    const std::uint32_t base = options.gm_base != 0 ? options.gm_base : 3U;
    const std::filesystem::path save_dir =
        options.save_path.empty() ? "." : options.save_path;

    // Preserve legacy checkpoint/resume semantics only when a real legacy
    // checkpoint exists. `options.resume` itself is true by default.
    const std::string ckpt_prefix =
        (t.family == "GQ" ? "gq" : "gm") +
        std::string("_pm1_p") + std::to_string(t.p);
    const std::filesystem::path legacy_s1 =
        save_dir / (ckpt_prefix + "_stage1.ckpt");
    const std::filesystem::path legacy_s2 =
        save_dir / (ckpt_prefix + "_stage2.ckpt");
    const bool legacy_checkpoint_present =
        std::filesystem::exists(legacy_s1) ||
        std::filesystem::exists(legacy_s1.string() + ".old") ||
        std::filesystem::exists(legacy_s2) ||
        std::filesystem::exists(legacy_s2.string() + ".old");
    if (legacy_checkpoint_present) {
        std::cout << "[GM-PM1] legacy checkpoint detected; "
                     "using resumable v100.12 product-exponent path.\n";
        return runGaussianMersennePM1Legacy();
    }

    const auto job_start = Clock::now();

    std::cout << "Gaussian pair P-1 factoring v100.13 V-trace\n"
              << "  p              : " << t.p << "\n"
              << "  target family  : " << t.family << "\n"
              << "  lift exponent  : " << t.lift << " (target | 2^(4p)-1)\n"
              << "  B1 / B2        : " << B1 << " / " << B2 << "\n"
              << "  base           : " << base << "\n";

    if (options.gm_sieve_limit != 0) {
        try {
            const std::uint64_t sf = find_admissible_small_factor(t, options.gm_sieve_limit);
            if (sf != 0) {
                std::cout << ">>> Gaussian pair admissible-sieve factor: " << sf << "\n";
                write_result(save_dir, t, "factor", 0, B1, B2, "CPU sieve", 0,
                             seconds_since(job_start), options.device_id, std::to_string(sf));
                return 0;
            }
        } catch (const std::exception& ex) {
            std::cerr << "[GM-PM1] admissible sieve error: " << ex.what() << "\n";
            return 2;
        }
    }

    mpz_class base_gcd;
    mpz_gcd_ui(base_gcd.get_mpz_t(), t.n.get_mpz_t(), base);
    if (proper_factor(base_gcd, t.n)) {
        std::cout << ">>> Gaussian pair P-1 base-gcd factor: " << base_gcd << "\n";
        write_result(save_dir, t, "factor", 0, B1, B2, "CPU gcd", 0,
                     seconds_since(job_start), options.device_id, base_gcd.get_str());
        return 0;
    }

    // ----- Stage 1: same exact exponent as the legacy implementation.
    mpz_class smooth;
    try {
        smooth = buildE(B1);
    } catch (const std::exception& ex) {
        std::cerr << "[GM-PM1] buildE failed: " << ex.what() << "\n";
        return 2;
    }
    const mpz_class structural = mpz_class(4) * t.p;
    mpz_class exponent;
    mpz_lcm(exponent.get_mpz_t(), smooth.get_mpz_t(), structural.get_mpz_t());
    const std::uint64_t e_bits =
        static_cast<std::uint64_t>(mpz_sizeinbase(exponent.get_mpz_t(), 2));
    std::cout << "  Stage 1 bits   : " << e_bits << "\n";

    std::unique_ptr<engine> s1;
    try {
        s1.reset(engine::create_gpu(t.lift, 3, static_cast<std::size_t>(options.device_id), true));
    } catch (const std::exception& ex) {
        std::cerr << "[GM-PM1] Stage 1 engine allocation failed: " << ex.what() << "\n";
        return 2;
    }
    if (!s1) return 2;
    for (std::size_t r = 0; r < 3; ++r) s1->set(r, 1);
    const std::string s1_backend = s1->is_aevum_backend() ? "Aevum" : "Marin";
    std::cout << "  Stage 1 backend: " << s1_backend << "\n";

    constexpr engine::Reg RH = 0;
    s1->set(RH, 1);
    const auto s1_start = Clock::now();
    auto last_report = s1_start;
    for (std::uint64_t i = e_bits; i > 0; --i) {
        const bool bit = mpz_tstbit(exponent.get_mpz_t(), i - 1) != 0;
        s1->square_mul(RH, bit ? base : 1U);
        if ((i & 4095ULL) == 0 || i == 1) {
            s1->sync();
            if (interrupted) {
                std::cout << "[GM-PM1] Interrupted during fast Stage 1. "
                             "No compact V-trace checkpoint was written; rerun with "
                             "PRMERS_GM_PM1_PRODUCT_STAGE2=1 or -gm-safe to use the legacy path.\n";
                return 0;
            }
            const auto now = Clock::now();
            if (now - last_report >= std::chrono::seconds(5) || i == 1) {
                const std::uint64_t done = e_bits - i + 1;
                const double sec = seconds_since(s1_start);
                std::cout << std::fixed << std::setprecision(2)
                          << "GM P-1 Stage 1: "
                          << (100.0 * static_cast<double>(done) / static_cast<double>(e_bits))
                          << "% | " << done << "/" << e_bits
                          << " bits | bit-IPS " << (sec > 0 ? done / sec : 0.0)
                          << " | elapsed " << sec << " s\n";
                last_report = now;
            }
        }
    }
    s1->sync();

    const mpz_class H_lift = get_reg_mpz(s1.get(), RH);
    mpz_class h = mod_positive(H_lift, t.n);
    mpz_class g = proper_gcd(mod_positive(h - 1, t.n), t.n);
    std::cout << "Stage 1 residue low64: 0x" << low_hex(h) << "\n";
    if (proper_factor(g, t.n)) {
        std::cout << ">>> Gaussian pair P-1 Stage 1 factor: " << g << "\n";
        write_result(save_dir, t, "factor", 1, B1, B2, s1_backend, 0,
                     seconds_since(job_start), options.device_id, g.get_str());
        if (!options.pm1_continue_stage2_after_factor) return 0;
        std::cout << "[GM-PM1] Continuing Stage 2 by explicit "
                     "-pm1-continue-stage2-after-factor.\n";
    } else if (g == t.n) {
        std::cout << "[GM-PM1] Stage 1 gcd=target. Falling back to the legacy "
                     "implementation for its isolation/retry semantics.\n";
        return runGaussianMersennePM1Legacy();
    } else {
        std::cout << "No Gaussian pair P-1 Stage 1 factor.\n";
    }

    // An inverse modulo the exact target is all V-trace needs.  We intentionally
    // do not require an inverse in the lifted ring.
    mpz_class hg = proper_gcd(h, t.n);
    if (proper_factor(hg, t.n)) {
        std::cout << ">>> Gaussian pair P-1 Stage 2 setup factor: " << hg << "\n";
        write_result(save_dir, t, "factor", 2, B1, B2, s1_backend, 0,
                     seconds_since(job_start), options.device_id, hg.get_str());
        return 0;
    }
    mpz_class hinv;
    if (mpz_invert(hinv.get_mpz_t(), h.get_mpz_t(), t.n.get_mpz_t()) == 0) {
        std::cout << "[GM-PM1] h is not invertible modulo the exact target; "
                     "falling back to legacy product-exponent Stage 2.\n";
        return runGaussianMersennePM1Legacy();
    }

    std::uint64_t D = options.pm1_vtrace_D;
    if (D == 0) D = choose_default_D(B1, B2);
    if (D < 4 || (D & 1ULL) != 0 || D > B1) {
        std::cout << "[GM-PM1] V-trace D=" << D
                  << " is not valid for this range (need even 4<=D<=B1); "
                     "falling back to legacy product-exponent Stage 2.\n";
        return runGaussianMersennePM1Legacy();
    }

    std::vector<std::uint32_t> primes;
    try {
        primes = primes_in_range(B1, B2);
    } catch (const std::exception& ex) {
        std::cerr << "[GM-PM1] Stage 2 sieve failed: " << ex.what() << "\n";
        return 2;
    }
    primes.erase(std::remove(primes.begin(), primes.end(), t.p), primes.end());
    if (primes.empty()) {
        write_result(save_dir, t, "no-factor", 2, B1, B2, s1_backend, D,
                     seconds_since(job_start), options.device_id, std::nullopt);
        return 1;
    }

    std::set<TraceTask> unique_tasks;
    std::set<std::uint64_t> unique_babies;
    for (const std::uint32_t q32 : primes) {
        const std::uint64_t q = q32;
        std::uint64_t k = q / D;
        const std::uint64_t rem = q - k * D;
        std::uint64_t j = rem;
        if (rem > D / 2) {
            ++k;
            j = D - rem;
        }
        if (j == 0 || k == 0) {
            std::cout << "[GM-PM1] Prime " << q
                      << " is not representable by the selected D under the "
                         "production V-trace assumptions; falling back to legacy Stage 2.\n";
            return runGaussianMersennePM1Legacy();
        }
        unique_tasks.insert(TraceTask{k, j});
        unique_babies.insert(j);
    }

    std::vector<TraceTask> tasks(unique_tasks.begin(), unique_tasks.end());
    std::vector<std::uint64_t> babies(unique_babies.begin(), unique_babies.end());
    std::map<std::uint64_t, std::size_t> baby_index;
    for (std::size_t i = 0; i < babies.size(); ++i) baby_index[babies[i]] = i;

    const std::size_t reg_count = VRegs::BABY_BASE + babies.size();
    const std::uint64_t paired = primes.size() >= tasks.size() ? primes.size() - tasks.size() : 0;
    std::cout << "[GM-PM1-VTRACE] D=" << D
              << " | primes=" << primes.size()
              << " | unique trace terms=" << tasks.size()
              << " | paired-away=" << paired
              << " | babies=" << babies.size()
              << " | registers=" << reg_count << "\n";

    std::unique_ptr<engine> eng;
    try {
        eng.reset(engine::create_gpu(t.lift, reg_count,
                                     static_cast<std::size_t>(options.device_id), true));
    } catch (const std::exception& ex) {
        std::cerr << "[GM-PM1-VTRACE] engine allocation failed: " << ex.what()
                  << "\n[GM-PM1-VTRACE] Falling back to legacy Stage 2.\n";
        return runGaussianMersennePM1Legacy();
    }
    if (!eng) return 2;
    const bool aevum = eng->is_aevum_backend();
    const std::string backend = aevum ? "Aevum" : "Marin";
    std::cout << "[GM-PM1-VTRACE] backend=" << backend
              << " | lift words=" << eng->get_size() << "\n";
    for (std::size_t r = 0; r < reg_count; ++r) eng->set(r, 1);

    // V1 = h + h^-1 after projection modulo N.
    set_reg_mpz(eng.get(), VRegs::V1, h);
    set_reg_mpz(eng.get(), VRegs::TMP, hinv);
    eng->add(VRegs::V1, VRegs::TMP);
    eng->set_multiplicand(VRegs::MUL_V1, VRegs::V1);

    // Build only the baby traces actually referenced by (B1,B2].
    eng->set(VRegs::BPREV, 2);        // V0
    eng->copy(VRegs::BCUR, VRegs::V1); // V1
    std::size_t next_baby = 0;
    for (std::uint64_t n = 1; n <= D; ++n) {
        while (next_baby < babies.size() && babies[next_baby] == n) {
            eng->copy(VRegs::BABY_BASE + next_baby, VRegs::BCUR);
            ++next_baby;
        }
        if (n == D) {
            eng->copy(VRegs::VD, VRegs::BCUR);
            break;
        }
        // V_{n+1} = V1*V_n - V_{n-1}
        eng->copy(VRegs::BNEXT, VRegs::BCUR);
        eng->mul(VRegs::BNEXT, VRegs::MUL_V1);
        eng->sub_reg(VRegs::BNEXT, VRegs::BPREV);
        eng->copy(VRegs::BPREV, VRegs::BCUR);
        eng->copy(VRegs::BCUR, VRegs::BNEXT);
        if (interrupted) return 0;
    }

    // Giant recurrence starts at V0,V_D.  Aevum's prepared-multiplicand cache is
    // intentionally refreshed before each giant multiply because preparing a
    // trace term may evict VD.
    eng->set(VRegs::GPREV, 2);
    eng->copy(VRegs::GCUR, VRegs::VD);
    eng->set(VRegs::ACC, 1);
    std::uint64_t current_k = 1;

    const std::uint64_t gcd_terms =
        std::max<std::uint64_t>(64, env_u64("PRMERS_GM_VTRACE_GCD_TERMS", 2048));
    std::uint64_t batch_terms = 0;
    std::uint64_t completed = 0;
    const auto s2_start = Clock::now();
    last_report = s2_start;

    auto check_accumulator = [&](bool final) -> int {
        if (batch_terms == 0 && !final) return -1;
        eng->sync();
        const mpz_class acc = project_reg(eng.get(), VRegs::ACC, t.n);
        const mpz_class gg = proper_gcd(acc, t.n);
        if (proper_factor(gg, t.n)) {
            std::cout << ">>> Gaussian pair P-1 Stage 2 V-trace factor: " << gg << "\n";
            write_result(save_dir, t, "factor", 2, B1, B2, backend, D,
                         seconds_since(job_start), options.device_id, gg.get_str());
            return 0;
        }
        if (gg == t.n) {
            std::cerr << "[GM-PM1-VTRACE] accumulator gcd=target. "
                         "Lower PRMERS_GM_VTRACE_GCD_TERMS (current "
                      << gcd_terms << ") to isolate the factor; refusing to claim no-factor.\n";
            return 2;
        }
        eng->set(VRegs::ACC, 1);
        batch_terms = 0;
        return -1;
    };

    for (const TraceTask& task : tasks) {
        while (current_k < task.k) {
            eng->copy(VRegs::GNEXT, VRegs::GCUR);
            eng->set_multiplicand(VRegs::MUL_VD, VRegs::VD);
            eng->mul(VRegs::GNEXT, VRegs::MUL_VD);
            eng->sub_reg(VRegs::GNEXT, VRegs::GPREV);
            eng->copy(VRegs::GPREV, VRegs::GCUR);
            eng->copy(VRegs::GCUR, VRegs::GNEXT);
            ++current_k;
        }

        const auto bi = baby_index.find(task.j);
        if (bi == baby_index.end()) {
            std::cerr << "[GM-PM1-VTRACE] internal baby lookup failure\n";
            return 2;
        }

        eng->copy(VRegs::TMP, VRegs::GCUR);
        eng->sub_reg(VRegs::TMP, VRegs::BABY_BASE + bi->second);
        eng->set_multiplicand(VRegs::MUL_TMP, VRegs::TMP);
        eng->mul(VRegs::ACC, VRegs::MUL_TMP);
        ++batch_terms;
        ++completed;

        if (batch_terms >= gcd_terms) {
            const int rc = check_accumulator(false);
            if (rc >= 0) return rc;
        }

        const auto now = Clock::now();
        if (now - last_report >= std::chrono::seconds(5) || completed == tasks.size()) {
            eng->sync();
            const double sec = seconds_since(s2_start);
            const double frac = tasks.empty() ? 1.0
                : static_cast<double>(completed) / static_cast<double>(tasks.size());
            const double eta = frac > 0.0 ? sec * (1.0 - frac) / frac : 0.0;
            std::cout << std::fixed << std::setprecision(2)
                      << "GM P-1 V-trace Stage 2: " << 100.0 * frac << "%"
                      << " | terms " << completed << "/" << tasks.size()
                      << " | k=" << current_k
                      << " | term-IPS " << (sec > 0 ? completed / sec : 0.0)
                      << " | elapsed " << sec << " s"
                      << " | ETA " << eta << " s\n";
            last_report = now;
        }

        if (interrupted) {
            std::cout << "[GM-PM1-VTRACE] interrupted at a clean term boundary. "
                         "Compact V-trace resume is intentionally not enabled in RC1; "
                         "rerun with PRMERS_GM_PM1_PRODUCT_STAGE2=1 to use the legacy resumable path.\n";
            return 0;
        }
    }

    if (batch_terms != 0) {
        const int rc = check_accumulator(true);
        if (rc >= 0) return rc;
    }

    const double s2_seconds = seconds_since(s2_start);
    std::cout << std::fixed << std::setprecision(2)
              << "GM P-1 V-trace Stage 2 complete: " << s2_seconds << " s"
              << " | " << tasks.size() << " unique terms from " << primes.size()
              << " primes | backend=" << backend << "\n";
    std::cout << "No Gaussian pair P-1 factor through B2=" << B2 << ".\n";
    write_result(save_dir, t, "no-factor", 2, B1, B2, backend, D,
                 seconds_since(job_start), options.device_id, std::nullopt);
    return 1;
}

}  // namespace core
