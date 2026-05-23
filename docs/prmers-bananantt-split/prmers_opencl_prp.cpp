/*
PrMers BananaNTT Split

Copyright 2026, Sébastien "Cherubrock"
GPU mixed-radix CRT/PFA half-real NTT for Mersenne testing.

This code is part of the PrMers experimental branch.

Project:
https://github.com/cherubrock-seb/PrMers/tree/main/docs/prmers-bananantt-split

CPU prototype:
https://github.com/cherubrock-seb/PrMers/tree/main/docs/mersenne2_mixed_crt_2d_half_fast

Original reference code by Yves Gallot:
https://github.com/galloty/mersenne2

This version keeps the power-of-two axis as a half-real GF(p^2) transform,
similar in spirit to mersenne2, and separates the odd axis with CRT/PFA
indexing.

The goal is to test transform sizes of the form odd * 2^m, for example
9 * 2^19, while still storing two real coefficients per complex value.

This is experimental research code. It is distributed in the hope that it
will be useful. Please give feedback or improvements if you test it.
*/
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <iostream>
#include <limits>
#include <sstream>
#include <ctime>
#include <stdexcept>
#include <system_error>
#include <string>
#include <vector>
#include <utility>
#include <atomic>
#include <thread>
#include <csignal>
#include <gmp.h>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

namespace {

static constexpr const char* BANANANTT_PROGRAM_NAME = "banana";
static constexpr const char* BANANANTT_PROGRAM_VERSION = "0.74.00-alpha";
static constexpr unsigned BANANANTT_PROGRAM_PORT = 8;
static constexpr double BANANANTT_DEFAULT_GERBICZ_TARGET_SECONDS = 600.0;
static constexpr double BANANANTT_DEFAULT_GERBICZ_MIN_SECONDS = 120.0;
static constexpr double BANANANTT_DEFAULT_QUEUE_GUARD_SECONDS = 2.0;
static constexpr double BANANANTT_DEFAULT_GERBICZ_BOUNDARY_SECONDS = 2.0;

std::atomic<bool> g_stop_requested{false};

void handle_sigint(int) { g_stop_requested.store(true, std::memory_order_relaxed); }

struct InterruptedRun : public std::exception {
    const char* what() const noexcept override { return "run interrupted"; }
};
struct RuntimeHooks {
    std::uint32_t exponent = 0;
    std::string mode_label = "BananaNTT mixed CRT";
    std::uint32_t res64_every = 0;
    bool json_enabled = true;
    std::string json_path;
    std::string output_dir;
    std::string results_path = "./results.txt";
    bool append_results = true;
    bool proof_checkpoints = false;
    std::uint32_t proof_power = 0;
    std::string proof_dir;
    std::string last_proof_file;
    std::uint32_t gerbicz_interval = 0;
    bool gerbicz_enabled = true;
    bool gerbicz_gpu_verify = true;
    bool gerbicz_user_checklevel = false;
    std::uint32_t gerbicz_checklevel = 0;
    std::uint32_t gerbicz_block = 0;
    double gerbicz_target_seconds = BANANANTT_DEFAULT_GERBICZ_TARGET_SECONDS;
    double gerbicz_estimated_ips = 0.0;
    bool gerbicz_user_seconds = false;
    double gerbicz_boundary_seconds = BANANANTT_DEFAULT_GERBICZ_BOUNDARY_SECONDS;
    bool gerbicz_verbose = false;
    bool gerbicz_progress = false;
    std::uint64_t gerbicz_errors = 0;
    std::uint32_t error_iter = 0;
    std::uint32_t error_limb = 0;
    std::uint64_t error_delta = 1;
    bool error_injected = false;
    std::uint32_t last_iter = 0;
    std::uint64_t last_res64 = 0;
    bool backup_enabled = true;
    bool resume_enabled = true;
    bool save_on_interrupt = true;
    std::string backup_path;
    std::string resume_path;
    std::string backup_dir = "save";
    std::uint32_t backup_every_iters = 0;
    double backup_every_seconds = 300.0;
    std::uint32_t queue_guard_depth = 0;
    bool queue_guard_auto = true;
    double queue_guard_seconds = BANANANTT_DEFAULT_QUEUE_GUARD_SECONDS;
};

RuntimeHooks g_runtime;

static inline std::uint32_t bananantt_clamp_u32(std::uint32_t v, std::uint32_t lo, std::uint32_t hi) {
    return std::min<std::uint32_t>(hi, std::max<std::uint32_t>(lo, v));
}

static std::uint32_t bananantt_auto_queue_guard(double ips, double seconds) {
    if (seconds <= 0.0) return 0u;
    ips = std::max(1.0, ips);
    const double raw = std::ceil(ips * seconds);
    if (raw <= 1.0) return 1u;
    if (raw >= 65536.0) return 65536u;
    return bananantt_clamp_u32(static_cast<std::uint32_t>(raw), 256u, 65536u);
}

static inline bool parse_bool_env(const char* name, bool defv) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defv;
    if (std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0 ||
        std::strcmp(s, "FALSE") == 0 || std::strcmp(s, "no") == 0 ||
        std::strcmp(s, "NO") == 0 || std::strcmp(s, "off") == 0 ||
        std::strcmp(s, "OFF") == 0) return false;
    return true;
}

static inline bool env_has_value(const char* name) {
    const char* s = std::getenv(name);
    return s && *s;
}

static inline bool parse_bool_env_alias(const char* name, const char* legacy_name, bool defv) {
    if (env_has_value(name)) return parse_bool_env(name, defv);
    if (legacy_name && env_has_value(legacy_name)) return parse_bool_env(legacy_name, defv);
    return defv;
}

static inline bool mixed_center_f48_delayed_scale_61() {
    const bool all = parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE",
                                          "PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE", false);
    return parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE_61",
                                "PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE_61", all);
}

static inline bool mixed_center_f48_delayed_scale_31() {
    /* GF31 is cheap in shifts, but the delayed-scale F48 form removes two
       early pair shifts in the hot center path. Keep it controllable. */
    bool def31 = true;
    if (env_has_value("PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE") ||
        env_has_value("PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE")) {
        def31 = parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE",
                                     "PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE", false);
    }
    return parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_DELAYED_SCALE_31",
                                "PRMERS_CRT_MIXED_CENTER_F48_LATE_SCALE_31", def31);
}

static inline bool mixed_center_f48_twin_symmetry_61() {
    const bool all = parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_TWIN_SYMMETRY",
                                          "PRMERS_CRT_MIXED_F48_TWIN_SYMMETRY", true);
    return parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_TWIN_SYMMETRY_61",
                                "PRMERS_CRT_MIXED_F48_TWIN_SYMMETRY_61", all);
}

static inline bool mixed_center_f48_twin_symmetry_31() {
    const bool all = parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_TWIN_SYMMETRY",
                                          "PRMERS_CRT_MIXED_F48_TWIN_SYMMETRY", true);
    return parse_bool_env_alias("PRMERS_CRT_MIXED_CENTER_F48_TWIN_SYMMETRY_31",
                                "PRMERS_CRT_MIXED_F48_TWIN_SYMMETRY_31", all);
}
}

namespace crt_tune {

static inline cl_uint env_u32(const char* name, cl_uint defv, cl_uint minv, cl_uint maxv) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defv;
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || v < minv || v > maxv) return defv;
    return static_cast<cl_uint>(v);
}

static inline cl_uint round_pow2_clamped(cl_uint v, cl_uint minv, cl_uint maxv) {
    if (v < minv) v = minv;
    if (v > maxv) v = maxv;
    cl_uint r = 1u;
    while ((r << 1u) <= v && (r << 1u) <= maxv) r <<= 1u;
    return (r < minv) ? minv : r;
}

static inline cl_uint garner_items(cl_uint digit_n, cl_uint min_digit_width, cl_uint configured_items) {
    cl_uint defv = std::max<cl_uint>(8u, configured_items);
    if (parse_bool_env("PRMERS_CRT_GARNER_X2", true) &&
        min_digit_width == 32u && digit_n >= 65536u && configured_items >= 32u) {
        defv = 64u;
    }
    return env_u32("PRMERS_CRT_GARNER_ITEMS", defv, 4u, 64u);
}

static inline cl_uint garner_local(cl_uint segments) {
    
    
    cl_uint defv = 16u;
    return round_pow2_clamped(env_u32("PRMERS_CRT_GARNER_LOCAL", defv, 16u, 256u), 16u, 256u);
}

}

namespace gf61 {


static std::uint64_t P = (std::uint64_t(1) << 61) - 1;
static std::uint64_t ROOT_H0 = 264036120304204ULL;
static std::uint64_t ROOT_H1 = 4677669021635377ULL;
static unsigned ROOT_ORDER_LOG2 = 62;
static unsigned FIELD_BITS = 61;
static const char* FIELD_NAME = "GF(M61^2)";

inline void configure_field(unsigned bits) {
    if (bits == 61) {
        P = (std::uint64_t(1) << 61) - 1;
        ROOT_H0 = 264036120304204ULL;
        ROOT_H1 = 4677669021635377ULL;
        ROOT_ORDER_LOG2 = 62;
        FIELD_BITS = 61;
        FIELD_NAME = "GF(M61^2)";
    } else if (bits == 31) {
        P = (std::uint64_t(1) << 31) - 1;
        
        
        ROOT_H0 = 2105104135ULL;
        ROOT_H1 = 2126293891ULL;
        ROOT_ORDER_LOG2 = 32;
        FIELD_BITS = 31;
        FIELD_NAME = "GF(M31^2)";
    } else {
        throw std::runtime_error("unsupported field bits; expected 61 or 31");
    }
}

struct Elem {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    Elem() = default;
    Elem(std::uint64_t aa, std::uint64_t bb) : a(aa), b(bb) {}
};

inline std::uint64_t add_mod(std::uint64_t x, std::uint64_t y) {
    std::uint64_t s = x + y;
    s = (s & P) + (s >> FIELD_BITS);
    if (s >= P) s -= P;
    return s;
}

inline std::uint64_t sub_mod(std::uint64_t x, std::uint64_t y) {
    return (x >= y) ? (x - y) : (P - (y - x));
}

inline std::uint64_t reduce_u128(__uint128_t x) {
    std::uint64_t lo = static_cast<std::uint64_t>(x) & P;
    std::uint64_t hi = static_cast<std::uint64_t>(x >> FIELD_BITS);
    std::uint64_t r = lo + hi;
    r = (r & P) + (r >> FIELD_BITS);
    if (r >= P) r -= P;
    return r;
}

inline std::uint64_t mul_mod(std::uint64_t x, std::uint64_t y) {
    return reduce_u128(static_cast<__uint128_t>(x) * static_cast<__uint128_t>(y));
}

std::uint64_t pow_mod(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    while (e) {
        if (e & 1ULL) r = mul_mod(r, a);
        a = mul_mod(a, a);
        e >>= 1U;
    }
    return r;
}

inline Elem add(const Elem& x, const Elem& y) {
    return Elem(add_mod(x.a, y.a), add_mod(x.b, y.b));
}

inline Elem sub(const Elem& x, const Elem& y) {
    return Elem(sub_mod(x.a, y.a), sub_mod(x.b, y.b));
}

inline Elem mul(const Elem& x, const Elem& y) {
    const std::uint64_t ac = mul_mod(x.a, y.a);
    const std::uint64_t bd = mul_mod(x.b, y.b);
    const std::uint64_t ad = mul_mod(x.a, y.b);
    const std::uint64_t bc = mul_mod(x.b, y.a);
    return Elem(sub_mod(ac, bd), add_mod(ad, bc));
}

inline Elem sqr(const Elem& x) {
    const std::uint64_t aa = mul_mod(x.a, x.a);
    const std::uint64_t bb = mul_mod(x.b, x.b);
    const std::uint64_t ab = mul_mod(x.a, x.b);
    return Elem(sub_mod(aa, bb), add_mod(ab, ab));
}

inline Elem conj(const Elem& x) {
    return Elem(x.a, x.b == 0 ? 0 : (P - x.b));
}

Elem inv(const Elem& x) {
    const std::uint64_t denom = add_mod(mul_mod(x.a, x.a), mul_mod(x.b, x.b));
    const std::uint64_t inv_denom = pow_mod(denom, P - 2);
    const Elem c = conj(x);
    return Elem(mul_mod(c.a, inv_denom), mul_mod(c.b, inv_denom));
}

Elem primitive_root_pow2(std::size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) throw std::runtime_error("NTT size must be a power of two");
    unsigned k = 0;
    while ((std::size_t(1) << k) < n) ++k;
    if (k > ROOT_ORDER_LOG2) throw std::runtime_error(std::string("NTT size exceeds 2-adic order in ") + FIELD_NAME);
    Elem r(ROOT_H0, ROOT_H1);
    for (unsigned i = 0; i < ROOT_ORDER_LOG2 - k; ++i) r = sqr(r);
    return r;
}

std::uint64_t primitive_root_odd_scalar(std::size_t n) {
    if (n <= 1) return 1;
    if (((P - 1) % n) != 0) throw std::runtime_error("unsupported odd radix for base field");
    for (std::uint64_t g = 2; g < 1000000; ++g) {
        const std::uint64_t z = pow_mod(g, (P - 1) / n);
        if (pow_mod(z, n) != 1) continue;
        bool primitive = true;
        std::size_t t = n;
        for (std::size_t f = 2; f <= t / f; ++f) {
            if ((t % f) == 0) {
                if (pow_mod(z, n / f) == 1) primitive = false;
                while ((t % f) == 0) t /= f;
            }
        }
        if (t > 1 && pow_mod(z, n / t) == 1) primitive = false;
        if (primitive) return z;
    }
    throw std::runtime_error("cannot find odd radix root in base field");
}

}

namespace ibdwt {

static unsigned g_capacity_bits = 61;
static unsigned g_shift_mod_bits = 61;


static unsigned g_recovery_headroom_bits = 2;

inline void configure_capacity(unsigned capacity_bits, unsigned shift_mod_bits, unsigned recovery_headroom_bits = 2) {
    g_capacity_bits = capacity_bits;
    g_shift_mod_bits = shift_mod_bits;
    g_recovery_headroom_bits = recovery_headroom_bits;
}

struct Layout {
    std::uint32_t p = 0;
    unsigned ln = 0;
    std::size_t n = 0;
    std::size_t pow2_n = 0;
    std::uint32_t odd = 1;
    std::vector<std::uint8_t> digit_width;
};

inline unsigned max_digit_width_for_log2(std::uint32_t p, unsigned ln) {
    const std::uint64_t n = std::uint64_t(1) << ln;
    return static_cast<unsigned>((std::uint64_t(p) + n - 1u) >> ln);
}

inline unsigned required_recovery_bits(std::uint32_t p, unsigned ln) {
    const unsigned max_w = max_digit_width_for_log2(p, ln);
    return ln + 2u * max_w + g_recovery_headroom_bits;
}

inline unsigned transform_size_log2(std::uint32_t p) {
    unsigned ln = 2;
    while (required_recovery_bits(p, ln) >= g_capacity_bits) {
        ++ln;
        if (ln >= 30) {
            throw std::runtime_error("unable to find safe transform size for selected modulus");
        }
    }
    return ln;
}

inline std::uint64_t inv_mod_u64(std::uint64_t a, std::uint64_t m) {
    a %= m;
    std::int64_t t = 0, nt = 1;
    std::int64_t r = static_cast<std::int64_t>(m), nr = static_cast<std::int64_t>(a);
    while (nr != 0) {
        const std::int64_t q = r / nr;
        const std::int64_t tt = t - q * nt; t = nt; nt = tt;
        const std::int64_t rr = r - q * nr; r = nr; nr = rr;
    }
    if (r != 1) throw std::runtime_error("transform length is not invertible modulo field shift period");
    if (t < 0) t += static_cast<std::int64_t>(m);
    return static_cast<std::uint64_t>(t);
}

inline std::uint8_t log2_root_two(std::size_t n) {
    
    
    return static_cast<std::uint8_t>(inv_mod_u64(n % g_shift_mod_bits, g_shift_mod_bits));
}

inline std::uint8_t shift_from_r_host(std::uint64_t r, std::uint32_t lr2, std::uint32_t field_bits) {
    if (r == 0u) return 0u;
    const std::uint32_t x = static_cast<std::uint32_t>((r * std::uint64_t(lr2)) % field_bits);
    return static_cast<std::uint8_t>(((field_bits + 1u) - x) % field_bits);
}

inline std::vector<std::uint8_t> make_unweight_shift_table(const Layout& layout, std::uint32_t field_bits, std::uint32_t lr2) {
    std::vector<std::uint8_t> out(layout.n);
    const std::uint64_t n64 = static_cast<std::uint64_t>(layout.n);
    const std::uint64_t p64 = static_cast<std::uint64_t>(layout.p);
    for (std::size_t j = 0; j < layout.n; ++j) {
        const std::uint64_t r = (static_cast<std::uint64_t>(j) * p64) % n64;
        out[j] = shift_from_r_host(r, lr2, field_bits);
    }
    return out;
}

static std::uint32_t bit_reverse(std::uint32_t x, unsigned bits) {
    std::uint32_t r = 0;
    for (unsigned i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

inline void fill_digit_widths(Layout& out) {
    out.digit_width.assign(out.n, 0);
    std::uint32_t prev_ceil = 0;
    for (std::size_t j = 0; j <= out.n; ++j) {
        const std::uint64_t qj = std::uint64_t(out.p) * std::uint64_t(j);
        const std::uint32_t ceil_qj_n = (j == 0) ? 0u : static_cast<std::uint32_t>((qj + out.n - 1u) / out.n);
        if (j > 0) out.digit_width[j - 1] = static_cast<std::uint8_t>(ceil_qj_n - prev_ceil);
        prev_ceil = ceil_qj_n;
    }
}

inline Layout make_layout(std::uint32_t p) {
    Layout out;
    out.p = p;
    out.ln = transform_size_log2(p);
    out.pow2_n = std::size_t(1) << out.ln;
    out.odd = 1;
    out.n = out.pow2_n;
    fill_digit_widths(out);
    return out;
}

inline unsigned required_recovery_bits_mixed(std::uint32_t p, std::size_t n) {
    const unsigned max_w = static_cast<unsigned>((std::uint64_t(p) + n - 1u) / n);
    unsigned log2n_ceil = 0;
    std::size_t t = 1;
    while (t < n) { t <<= 1; ++log2n_ceil; }
    return log2n_ceil + 2u * max_w + g_recovery_headroom_bits;
}

inline Layout make_layout_mixed(std::uint32_t p, std::uint32_t forced_odd) {
    if (forced_odd <= 1u) return make_layout(p);
    if (!(forced_odd == 3u || forced_odd == 9u)) {
        throw std::runtime_error("GPU mixed CRT/PFA path currently supports --crt-odd-radix 3 or 9");
    }
    Layout best;
    best.p = p;
    best.n = std::numeric_limits<std::size_t>::max();
    for (unsigned ln = 2; ln <= 30; ++ln) {
        const std::size_t pow2_n = std::size_t(1) << ln;
        const std::size_t n = pow2_n * std::size_t(forced_odd);
        if (n == 0 || n > p) continue;
        if (required_recovery_bits_mixed(p, n) < g_capacity_bits) {
            if (n < best.n) {
                best.p = p;
                best.ln = ln;
                best.pow2_n = pow2_n;
                best.odd = forced_odd;
                best.n = n;
            }
        }
    }
    if (best.n == std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("unable to find safe mixed odd-radix CRT/PFA transform size");
    }
    fill_digit_widths(best);
    return best;
}

static std::vector<std::uint64_t> from_small(std::uint64_t value, const Layout& layout) {
    std::vector<std::uint64_t> digits(layout.n, 0);
    for (std::size_t i = 0; i < layout.n; ++i) {
        const std::uint8_t w = layout.digit_width[i];
        const std::uint64_t mask = (w == 64) ? ~0ULL : ((std::uint64_t(1) << w) - 1ULL);
        digits[i] = value & mask;
        value >>= w;
    }
    if (value != 0) throw std::runtime_error("from_small overflow for layout");
    return digits;
}

static bool all_max_digits(const std::vector<std::uint64_t>& digits, const Layout& layout) {
    for (std::size_t i = 0; i < layout.n; ++i) {
        const std::uint8_t w = layout.digit_width[i];
        const std::uint64_t mask = (std::uint64_t(1) << w) - 1ULL;
        if (digits[i] != mask) return false;
    }
    return true;
}

static void canonicalize_zero(std::vector<std::uint64_t>& digits, const Layout& layout) {
    if (all_max_digits(digits, layout)) std::fill(digits.begin(), digits.end(), 0ULL);
}

static bool equals_small(const std::vector<std::uint64_t>& digits, const Layout& layout, std::uint64_t value) {
    return digits == from_small(value, layout);
}

static Layout make_layout_for_n(std::uint32_t p, std::size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) throw std::runtime_error("layout n must be a power of two");
    Layout out;
    out.p = p;
    out.n = n;
    out.ln = 0;
    while ((std::size_t(1) << out.ln) < n) ++out.ln;
    out.digit_width.assign(out.n, 0);

    std::uint32_t prev_ceil = 0;
    for (std::size_t j = 0; j <= out.n; ++j) {
        const std::uint64_t qj = std::uint64_t(p) * std::uint64_t(j);
        const std::uint32_t ceil_qj_n = (j == 0) ? 0u : static_cast<std::uint32_t>(((qj - 1u) >> out.ln) + 1u);
        if (j > 0) out.digit_width[j - 1] = static_cast<std::uint8_t>(ceil_qj_n - prev_ceil);
        prev_ceil = ceil_qj_n;
    }
    return out;
}

static std::vector<std::uint64_t> square_mod_mersenne_exact_digits(
    const std::vector<std::uint64_t>& digits,
    const Layout& layout)
{
    if (digits.size() != layout.n) throw std::runtime_error("exact square: digit/layout size mismatch");

    mpz_t x, y, mod, low, high, tmp;
    mpz_inits(x, y, mod, low, high, tmp, nullptr);

    std::vector<std::uint64_t> out(layout.n, 0);
    try {
        
        
        std::uint64_t off = 0;
        for (std::size_t i = 0; i < layout.n; ++i) {
            const unsigned w = layout.digit_width[i];
            if (digits[i]) {
                mpz_set_ui(tmp, static_cast<unsigned long>(digits[i]));
                mpz_mul_2exp(tmp, tmp, static_cast<mp_bitcnt_t>(off));
                mpz_add(x, x, tmp);
            }
            off += w;
        }

        
        mpz_set_ui(mod, 1ul);
        mpz_mul_2exp(mod, mod, static_cast<mp_bitcnt_t>(layout.p));
        mpz_sub_ui(mod, mod, 1ul);

        mpz_mul(y, x, x);

        
        mpz_tdiv_r_2exp(low, y, static_cast<mp_bitcnt_t>(layout.p));
        mpz_tdiv_q_2exp(high, y, static_cast<mp_bitcnt_t>(layout.p));
        mpz_add(y, low, high);
        while (mpz_cmp(y, mod) > 0) {
            mpz_tdiv_r_2exp(low, y, static_cast<mp_bitcnt_t>(layout.p));
            mpz_tdiv_q_2exp(high, y, static_cast<mp_bitcnt_t>(layout.p));
            mpz_add(y, low, high);
        }
        if (mpz_cmp(y, mod) == 0) mpz_set_ui(y, 0ul);

        off = 0;
        for (std::size_t i = 0; i < layout.n; ++i) {
            const unsigned w = layout.digit_width[i];
            mpz_tdiv_q_2exp(tmp, y, static_cast<mp_bitcnt_t>(off));
            mpz_tdiv_r_2exp(tmp, tmp, static_cast<mp_bitcnt_t>(w));
            out[i] = static_cast<std::uint64_t>(mpz_get_ui(tmp));
            off += w;
        }
    } catch (...) {
        mpz_clears(x, y, mod, low, high, tmp, nullptr);
        throw;
    }

    mpz_clears(x, y, mod, low, high, tmp, nullptr);
    return out;
}

static void digits_to_mpz(mpz_t out, const std::vector<std::uint64_t>& digits, const Layout& layout) {
    if (digits.size() != layout.n) throw std::runtime_error("digits_to_mpz: digit/layout size mismatch");
    mpz_set_ui(out, 0ul);
    mpz_t tmp;
    mpz_init(tmp);
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < layout.n; ++i) {
        const unsigned w = layout.digit_width[i];
        if (digits[i]) {
            mpz_set_ui(tmp, static_cast<unsigned long>(digits[i]));
            mpz_mul_2exp(tmp, tmp, static_cast<mp_bitcnt_t>(off));
            mpz_add(out, out, tmp);
        }
        off += w;
    }
    mpz_clear(tmp);
}

static std::vector<std::uint64_t> mpz_to_digits(const mpz_t value, const Layout& layout) {
    std::vector<std::uint64_t> out(layout.n, 0);
    mpz_t tmp;
    mpz_init(tmp);
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < layout.n; ++i) {
        const unsigned w = layout.digit_width[i];
        mpz_tdiv_q_2exp(tmp, value, static_cast<mp_bitcnt_t>(off));
        mpz_tdiv_r_2exp(tmp, tmp, static_cast<mp_bitcnt_t>(w));
        out[i] = static_cast<std::uint64_t>(mpz_get_ui(tmp));
        off += w;
    }
    mpz_clear(tmp);
    return out;
}

static void fold_mersenne_mod(mpz_t y, const Layout& layout) {
    mpz_t mod, low, high;
    mpz_inits(mod, low, high, nullptr);
    mpz_set_ui(mod, 1ul);
    mpz_mul_2exp(mod, mod, static_cast<mp_bitcnt_t>(layout.p));
    mpz_sub_ui(mod, mod, 1ul);
    while (mpz_cmp(y, mod) > 0) {
        mpz_tdiv_r_2exp(low, y, static_cast<mp_bitcnt_t>(layout.p));
        mpz_tdiv_q_2exp(high, y, static_cast<mp_bitcnt_t>(layout.p));
        mpz_add(y, low, high);
    }
    if (mpz_cmp(y, mod) == 0) mpz_set_ui(y, 0ul);
    mpz_clears(mod, low, high, nullptr);
}

static std::vector<std::uint64_t> mul_mod_mersenne_exact_digits(
    const std::vector<std::uint64_t>& a,
    const std::vector<std::uint64_t>& b,
    const Layout& layout)
{
    if (a.size() != layout.n || b.size() != layout.n) {
        throw std::runtime_error("exact multiply: digit/layout size mismatch");
    }
    mpz_t x, y, z;
    mpz_inits(x, y, z, nullptr);
    try {
        digits_to_mpz(x, a, layout);
        digits_to_mpz(y, b, layout);
        mpz_mul(z, x, y);
        fold_mersenne_mod(z, layout);
        auto out = mpz_to_digits(z, layout);
        mpz_clears(x, y, z, nullptr);
        return out;
    } catch (...) {
        mpz_clears(x, y, z, nullptr);
        throw;
    }
}

static std::vector<std::uint64_t> mul_small_mod_mersenne_exact_digits(
    const std::vector<std::uint64_t>& a,
    std::uint64_t k,
    const Layout& layout)
{
    return mul_mod_mersenne_exact_digits(a, from_small(k, layout), layout);
}

}

namespace clwrap {

inline void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        throw std::runtime_error(oss.str());
    }
}

static bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

static std::string dirname_of(const std::string& path) {
    const std::size_t p = path.find_last_of("/\\");
    if (p == std::string::npos) return ".";
    if (p == 0) return "/";
    return path.substr(0, p);
}

static std::string basename_of(const std::string& path) {
    const std::size_t p = path.find_last_of("/\\");
    return (p == std::string::npos) ? path : path.substr(p + 1);
}

static std::string resolve_kernel_path(const std::string& requested, const char* argv0) {
    std::vector<std::string> candidates;
    candidates.push_back(requested);
    if (argv0 && *argv0) {
        const std::string exe_dir = dirname_of(argv0);
        candidates.push_back(exe_dir + "/" + basename_of(requested));
    }
    candidates.push_back("./" + basename_of(requested));

    for (const std::string& c : candidates) {
        if (file_exists(c)) return c;
    }
    throw std::runtime_error("unable to open kernel file: " + requested +
                             " (also tried executable directory and current directory)");
}

static std::string load_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("unable to open kernel file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    const std::string head = s.substr(0, std::min<std::size_t>(s.size(), 4096));
    if (head.find("#include <CL/cl.h>") != std::string::npos ||
        head.find("#include <OpenCL/opencl.h>") != std::string::npos ||
        head.find("int main(") != std::string::npos) {
        throw std::runtime_error("kernel path points to the C++ host file, not to the .cl OpenCL kernel: " + path);
    }
    return s;
}

static std::uint64_t fnv1a64_bytes(const void* data, std::size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

static std::uint64_t fnv1a64_string(const std::string& s) {
    return fnv1a64_bytes(s.data(), s.size());
}

static std::string hex64(std::uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

static std::string sanitize_token(std::string s) {
    for (char& c : s) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!std::isalnum(u) && c != '_' && c != '-' && c != '.') c = '_';
    }
    while (!s.empty() && s.back() == '_') s.pop_back();
    if (s.empty()) s = "unknown";
    if (s.size() > 80) s.resize(80);
    return s;
}

static std::string cl_device_string(cl_device_id dev, cl_device_info what) {
    size_t sz = 0;
    if (clGetDeviceInfo(dev, what, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) return "unknown";
    std::string s(sz, '\0');
    if (clGetDeviceInfo(dev, what, sz, s.data(), nullptr) != CL_SUCCESS) return "unknown";
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s.empty() ? "unknown" : s;
}

static std::vector<unsigned char> load_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    if (n <= 0) return {};
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> out(static_cast<std::size_t>(n));
    in.read(reinterpret_cast<char*>(out.data()), n);
    if (!in) return {};
    return out;
}

static bool write_binary_file_atomic(const std::filesystem::path& path, const unsigned char* data, std::size_t n) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
        if (!out) return false;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    return !ec;
}

static std::filesystem::path opencl_cache_dir() {
    if (const char* e = std::getenv("PRMERS_OCL_CACHE_DIR")) {
        if (*e) return std::filesystem::path(e);
    }
    return std::filesystem::path(".ocl_cache");
}

static std::filesystem::path opencl_cache_path(cl_device_id dev, const std::string& kernel_path, const std::string& source, const std::string& build_opts, int field_bits) {
    const std::string name = cl_device_string(dev, CL_DEVICE_NAME);
    const std::string driver = cl_device_string(dev, CL_DRIVER_VERSION);
    std::string key;
    key.reserve(source.size() + build_opts.size() + kernel_path.size() + 256);
    key += BANANANTT_PROGRAM_VERSION;
    key.push_back('\n');
    key += kernel_path;
    key.push_back('\n');
    key += source;
    key.push_back('\n');
    key += build_opts;
    key.push_back('\n');
    key += name;
    key.push_back('\n');
    key += driver;
    key.push_back('\n');
    key += std::to_string(field_bits);
    const std::string file = std::string("field") + std::to_string(field_bits) + "_" + sanitize_token(name) + "_" + hex64(fnv1a64_string(key)) + ".bin";
    return opencl_cache_dir() / file;
}

static bool save_opencl_binary(cl_program program, const std::filesystem::path& path) {
    size_t bin_size = 0;
    if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, nullptr) != CL_SUCCESS) return false;
    if (bin_size == 0) return false;
    std::vector<unsigned char> bin(bin_size);
    unsigned char* ptr = bin.data();
    if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &ptr, nullptr) != CL_SUCCESS) return false;
    return write_binary_file_atomic(path, bin.data(), bin.size());
}

struct DeviceInfo {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string name;
    std::size_t max_work_group_size = 0;
    cl_ulong local_mem_size = 0;
};

std::vector<DeviceInfo> list_devices() {
    cl_uint nplat = 0;
    check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> plats(nplat);
    check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

    std::vector<DeviceInfo> out;
    for (cl_platform_id plat : plats) {
        cl_uint ndev = 0;
        cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
        if (err == CL_DEVICE_NOT_FOUND) continue;
        check(err, "clGetDeviceIDs(count)");
        std::vector<cl_device_id> devs(ndev);
        check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr), "clGetDeviceIDs(list)");
        for (cl_device_id dev : devs) {
            size_t sz = 0;
            check(clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &sz), "clGetDeviceInfo(name size)");
            std::string name(sz, '\0');
            check(clGetDeviceInfo(dev, CL_DEVICE_NAME, sz, name.data(), nullptr), "clGetDeviceInfo(name)");
            while (!name.empty() && (name.back() == '\0' || name.back() == '\n' || name.back() == '\r')) name.pop_back();
            size_t max_wg = 0;
            cl_ulong local_mem = 0;
            check(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr), "clGetDeviceInfo(max wg)");
            check(clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, nullptr), "clGetDeviceInfo(local mem)");
            out.push_back(DeviceInfo{plat, dev, name, max_wg, local_mem});
        }
    }
    if (out.empty()) throw std::runtime_error("no OpenCL devices found");
    return out;
}

struct StageInfo {
    std::uint32_t len = 0;
    std::uint32_t half_len = 0;
    std::uint32_t offset = 0;
};

struct ProfileEntry {
    double ms = 0.0;
    std::uint64_t launches = 0;
};

struct GpuPrp {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel k_weight_first_stage_dif = nullptr;
    cl_kernel k_weight_first_stage_dif_wg16 = nullptr;
    cl_kernel k_weight_first_stage_dif_wg64 = nullptr;
    cl_kernel k_weight_first_stage_dif_radix4_wg64 = nullptr;
    cl_kernel k_weight_first_stage_dif_radix4_wg128 = nullptr;
    cl_kernel k_ntt_stage_dif = nullptr;
    cl_kernel k_ntt_stage_dif_len2048 = nullptr;
    cl_kernel k_ntt_stage_dit = nullptr;
    cl_kernel k_ntt_stage_dif_radix4 = nullptr;
    cl_kernel k_ntt_stage_dif_radix4_wg128 = nullptr;
    cl_kernel k_ntt_stage_dif_radix4x2 = nullptr;
    cl_kernel k_ntt_stage_dif_radix4x2_wg128 = nullptr;
    cl_kernel k_ntt_stage_dit_radix4 = nullptr;
    cl_kernel k_ntt_stage_dit_radix4_wg128 = nullptr;
    cl_kernel k_ntt_stage_dit_radix4x2 = nullptr;
    cl_kernel k_ntt_stage_dit_radix4x2_wg128 = nullptr;
    cl_kernel k_last_stage_dit_unweight = nullptr;
    cl_kernel k_last_stage_dit_unweight_wg16 = nullptr;
    cl_kernel k_last_stage_dit_unweight_wg64 = nullptr;
    cl_kernel k_last_stage_dit_radix4_unweight_wg64 = nullptr;
    cl_kernel k_last_stage_dit_radix4_unweight_wg128 = nullptr;
    cl_kernel k_pointwise_sqr = nullptr;
    cl_kernel k_center_fused_8 = nullptr;
    cl_kernel k_center_fused_16 = nullptr;
    cl_kernel k_center_fused_32 = nullptr;
    cl_kernel k_center_fused_64 = nullptr;
    cl_kernel k_center_fused_128 = nullptr;
    cl_kernel k_center_fused_256 = nullptr;
    cl_kernel k_center_fused_256_explicit = nullptr;
    cl_kernel k_center_fused_512 = nullptr;
    cl_kernel k_center_fused_1024 = nullptr;
    cl_kernel k_center_fused_2048 = nullptr;
    cl_kernel k_center_fused_4096 = nullptr;
    cl_kernel k_forward_bridge_64_to_16 = nullptr;
    cl_kernel k_inverse_bridge_16_to_64 = nullptr;
    cl_kernel k_forward_bridge_256_to_64 = nullptr;
    cl_kernel k_inverse_bridge_64_to_256 = nullptr;
    cl_kernel k_forward_bridge_512_to_256 = nullptr;
    cl_kernel k_inverse_bridge_256_to_512 = nullptr;
    cl_kernel k_forward_bridge_1024_to_512 = nullptr;
    cl_kernel k_inverse_bridge_512_to_1024 = nullptr;
    cl_kernel k_forward_bridge_1024_to_256 = nullptr;
    cl_kernel k_inverse_bridge_256_to_1024 = nullptr;
    cl_kernel k_forward_bridge_2048_to_256 = nullptr;
    cl_kernel k_inverse_bridge_256_to_2048 = nullptr;
    cl_kernel k_forward_ext_1024_to_256_explicit = nullptr;
    cl_kernel k_forward_ext_1024_to_256_explicit_wg128 = nullptr;
    cl_kernel k_inverse_ext_256_to_1024_explicit = nullptr;
    cl_kernel k_inverse_ext_256_to_1024_explicit_wg128 = nullptr;
    cl_kernel k_forward_ext_2048_to_256_explicit = nullptr;
    cl_kernel k_inverse_ext_256_to_2048_explicit = nullptr;
    cl_kernel k_mul_small = nullptr;
    cl_kernel k_carry_block_local = nullptr;
    cl_kernel k_carry_block_prefix = nullptr;
    cl_kernel k_carry_block_prefix_chunked64 = nullptr;
    cl_kernel k_carry_block_apply_incoming = nullptr;
    cl_kernel k_carry_block_apply_incoming_serial = nullptr;
    cl_kernel k_carry_final_wrap = nullptr;
    cl_kernel k_carry_clear_pending = nullptr;
    cl_kernel k_crt_garner_segment_first = nullptr;
    cl_kernel k_crt_carry_segment_pass = nullptr;
    cl_kernel k_crt_carry_cleanup_serial = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout_mask32 = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout_mask32_base32_fast = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout_mask32_base32_x2 = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout_mask32_base32_u32lean = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout_coeffhi_mask32_base32 = nullptr;
    cl_kernel k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2 = nullptr;
    cl_kernel k_crt_last_unweight_garner_segment_first_oneout = nullptr;
    cl_kernel k_crt_last_unweight_garner_segment_first_oneout_mask32 = nullptr;
    cl_kernel k_crt_carry_segment_pass_oneout = nullptr;
    cl_kernel k_crt_carry_cleanup_serial_oneout = nullptr;
    cl_kernel k_crt_carry_cleanup_parallel_oneout = nullptr;
    cl_kernel k_crt_mixed_carry_pack_next_lds_61x31 = nullptr;

    cl_mem bufDigits = nullptr;
    cl_mem bufDigits32 = nullptr;
    
    
    cl_mem crtInputDigits = nullptr;
    bool crtLastUnweightPending = false;
    bool crtCoeffPending = false;
    bool crtFirstCarryReady = false;
    cl_mem bufField = nullptr;
    cl_mem bufWidth = nullptr;
    cl_mem bufWidthMask32 = nullptr;
    cl_mem bufUnweightShift = nullptr;
    cl_mem bufTwFwd = nullptr;
    cl_mem bufTwInv = nullptr;
    cl_mem bufOddFwd = nullptr;
    cl_mem bufOddInv = nullptr;
    cl_mem bufOddScratch = nullptr;
    cl_mem bufBlockCarry = nullptr;
    cl_mem bufBlockValueLo = nullptr;
    cl_mem bufBlockBits = nullptr;
    cl_mem bufBlockThreshold = nullptr;
    cl_mem bufBlockMode = nullptr;
    cl_mem bufBlockIncoming = nullptr;
    cl_mem bufFinalCarry = nullptr;
    cl_mem bufSegValueLo = nullptr;
    cl_mem bufSegBits = nullptr;
    cl_mem bufSegThreshold = nullptr;
    cl_mem bufSegMode = nullptr;
    cl_mem bufCarryPending = nullptr;
    cl_mem bufCarryStats = nullptr;
    cl_mem bufCrtCarryLo1 = nullptr;
    cl_mem bufCrtCarryHi1 = nullptr;
    cl_mem bufCrtCarryLo2 = nullptr;
    cl_mem bufCrtCarryHi2 = nullptr;
    cl_uint carry_buffer_blocks = 0;
    cl_uint carry_buffer_segments = 0;
    cl_uint carry_buffer_digits = 0;
    cl_uint carry_passes = 6;
    cl_uint min_digit_width = 32;
    cl_uint carry_stats_last_runs = 0;
    cl_uint carry_stats_last_rounds = 0;

    std::size_t n = 0;
    cl_uint log_n = 0;
    cl_uint exponent_p = 0;
    cl_uint lr2 = 0;
    std::size_t max_work_group_size = 0;
    cl_ulong local_mem_size = 0;
    bool profile_kernels = false;
    bool prefer_radix4x2 = true;
    unsigned field_bits = 61;
    std::size_t field_elem_size = sizeof(gf61::Elem);
    
    cl_event pending_wait_event = nullptr;
    std::vector<StageInfo> stages;
    std::vector<std::pair<std::string, cl_event>> pending_profile_events;
    std::map<std::string, ProfileEntry> profile_totals;
    std::vector<std::string> profile_order;

    ~GpuPrp() {
        if (bufCrtCarryHi2) clReleaseMemObject(bufCrtCarryHi2);
        if (bufCrtCarryLo2) clReleaseMemObject(bufCrtCarryLo2);
        if (bufCrtCarryHi1) clReleaseMemObject(bufCrtCarryHi1);
        if (bufCrtCarryLo1) clReleaseMemObject(bufCrtCarryLo1);
        if (bufCarryStats) clReleaseMemObject(bufCarryStats);
        if (bufCarryPending) clReleaseMemObject(bufCarryPending);
        if (bufSegMode) clReleaseMemObject(bufSegMode);
        if (bufSegThreshold) clReleaseMemObject(bufSegThreshold);
        if (bufSegBits) clReleaseMemObject(bufSegBits);
        if (bufSegValueLo) clReleaseMemObject(bufSegValueLo);
        if (bufFinalCarry) clReleaseMemObject(bufFinalCarry);
        if (bufBlockIncoming) clReleaseMemObject(bufBlockIncoming);
        if (bufBlockMode) clReleaseMemObject(bufBlockMode);
        if (bufBlockThreshold) clReleaseMemObject(bufBlockThreshold);
        if (bufBlockBits) clReleaseMemObject(bufBlockBits);
        if (bufBlockValueLo) clReleaseMemObject(bufBlockValueLo);
        if (bufBlockCarry) clReleaseMemObject(bufBlockCarry);
        if (bufOddScratch) clReleaseMemObject(bufOddScratch);
        if (bufOddInv) clReleaseMemObject(bufOddInv);
        if (bufOddFwd) clReleaseMemObject(bufOddFwd);
        if (bufTwInv) clReleaseMemObject(bufTwInv);
        if (bufTwFwd) clReleaseMemObject(bufTwFwd);
        if (bufWidth) clReleaseMemObject(bufWidth);
        if (bufWidthMask32) clReleaseMemObject(bufWidthMask32);
        if (bufUnweightShift) clReleaseMemObject(bufUnweightShift);
        if (bufField) clReleaseMemObject(bufField);
        if (pending_wait_event) clReleaseEvent(pending_wait_event);
        if (bufDigits) clReleaseMemObject(bufDigits);
        if (bufDigits32) clReleaseMemObject(bufDigits32);
        if (k_carry_clear_pending) clReleaseKernel(k_carry_clear_pending);
        if (k_crt_mixed_carry_pack_next_lds_61x31) clReleaseKernel(k_crt_mixed_carry_pack_next_lds_61x31);
        if (k_crt_carry_cleanup_parallel_oneout) clReleaseKernel(k_crt_carry_cleanup_parallel_oneout);
        if (k_crt_carry_cleanup_serial_oneout) clReleaseKernel(k_crt_carry_cleanup_serial_oneout);
        if (k_crt_carry_segment_pass_oneout) clReleaseKernel(k_crt_carry_segment_pass_oneout);
        if (k_crt_last_unweight_garner_segment_first_oneout) clReleaseKernel(k_crt_last_unweight_garner_segment_first_oneout);
        if (k_crt_last_unweight_garner_segment_first_oneout_mask32) clReleaseKernel(k_crt_last_unweight_garner_segment_first_oneout_mask32);
        if (k_crt_garner_segment_first_oneout_coeffhi_mask32_base32) clReleaseKernel(k_crt_garner_segment_first_oneout_coeffhi_mask32_base32);
        if (k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2) clReleaseKernel(k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2);
        if (k_crt_garner_segment_first_oneout) clReleaseKernel(k_crt_garner_segment_first_oneout);
        if (k_crt_garner_segment_first_oneout_mask32) clReleaseKernel(k_crt_garner_segment_first_oneout_mask32);
        if (k_crt_garner_segment_first_oneout_mask32_base32_fast) clReleaseKernel(k_crt_garner_segment_first_oneout_mask32_base32_fast);
        if (k_crt_garner_segment_first_oneout_mask32_base32_x2) clReleaseKernel(k_crt_garner_segment_first_oneout_mask32_base32_x2);
        if (k_crt_garner_segment_first_oneout_mask32_base32_u32lean) clReleaseKernel(k_crt_garner_segment_first_oneout_mask32_base32_u32lean);
        if (k_crt_carry_cleanup_serial) clReleaseKernel(k_crt_carry_cleanup_serial);
        if (k_crt_carry_segment_pass) clReleaseKernel(k_crt_carry_segment_pass);
        if (k_crt_garner_segment_first) clReleaseKernel(k_crt_garner_segment_first);
        if (k_carry_final_wrap) clReleaseKernel(k_carry_final_wrap);
        if (k_carry_block_apply_incoming_serial) clReleaseKernel(k_carry_block_apply_incoming_serial);
        if (k_carry_block_apply_incoming) clReleaseKernel(k_carry_block_apply_incoming);
        if (k_carry_block_prefix_chunked64) clReleaseKernel(k_carry_block_prefix_chunked64);
        if (k_carry_block_prefix) clReleaseKernel(k_carry_block_prefix);
        if (k_carry_block_local) clReleaseKernel(k_carry_block_local);
        if (k_mul_small) clReleaseKernel(k_mul_small);
        if (k_inverse_ext_256_to_2048_explicit) clReleaseKernel(k_inverse_ext_256_to_2048_explicit);
        if (k_forward_ext_2048_to_256_explicit) clReleaseKernel(k_forward_ext_2048_to_256_explicit);
        if (k_inverse_ext_256_to_1024_explicit_wg128) clReleaseKernel(k_inverse_ext_256_to_1024_explicit_wg128);
        if (k_inverse_ext_256_to_1024_explicit) clReleaseKernel(k_inverse_ext_256_to_1024_explicit);
        if (k_forward_ext_1024_to_256_explicit_wg128) clReleaseKernel(k_forward_ext_1024_to_256_explicit_wg128);
        if (k_forward_ext_1024_to_256_explicit) clReleaseKernel(k_forward_ext_1024_to_256_explicit);
        if (k_inverse_bridge_256_to_2048) clReleaseKernel(k_inverse_bridge_256_to_2048);
        if (k_forward_bridge_2048_to_256) clReleaseKernel(k_forward_bridge_2048_to_256);
        if (k_inverse_bridge_256_to_1024) clReleaseKernel(k_inverse_bridge_256_to_1024);
        if (k_forward_bridge_1024_to_256) clReleaseKernel(k_forward_bridge_1024_to_256);
        if (k_inverse_bridge_512_to_1024) clReleaseKernel(k_inverse_bridge_512_to_1024);
        if (k_forward_bridge_1024_to_512) clReleaseKernel(k_forward_bridge_1024_to_512);
        if (k_inverse_bridge_256_to_512) clReleaseKernel(k_inverse_bridge_256_to_512);
        if (k_forward_bridge_512_to_256) clReleaseKernel(k_forward_bridge_512_to_256);
        if (k_inverse_bridge_64_to_256) clReleaseKernel(k_inverse_bridge_64_to_256);
        if (k_forward_bridge_256_to_64) clReleaseKernel(k_forward_bridge_256_to_64);
        if (k_inverse_bridge_16_to_64) clReleaseKernel(k_inverse_bridge_16_to_64);
        if (k_forward_bridge_64_to_16) clReleaseKernel(k_forward_bridge_64_to_16);
        if (k_center_fused_4096) clReleaseKernel(k_center_fused_4096);
        if (k_center_fused_2048) clReleaseKernel(k_center_fused_2048);
        if (k_center_fused_1024) clReleaseKernel(k_center_fused_1024);
        if (k_center_fused_512) clReleaseKernel(k_center_fused_512);
        if (k_center_fused_256) clReleaseKernel(k_center_fused_256);
        if (k_center_fused_256_explicit) clReleaseKernel(k_center_fused_256_explicit);
        if (k_center_fused_128) clReleaseKernel(k_center_fused_128);
        if (k_center_fused_64) clReleaseKernel(k_center_fused_64);
        if (k_center_fused_32) clReleaseKernel(k_center_fused_32);
        if (k_center_fused_16) clReleaseKernel(k_center_fused_16);
        if (k_center_fused_8) clReleaseKernel(k_center_fused_8);
        if (k_pointwise_sqr) clReleaseKernel(k_pointwise_sqr);
        if (k_last_stage_dit_radix4_unweight_wg128) clReleaseKernel(k_last_stage_dit_radix4_unweight_wg128);
        if (k_last_stage_dit_radix4_unweight_wg64) clReleaseKernel(k_last_stage_dit_radix4_unweight_wg64);
        if (k_last_stage_dit_unweight_wg64) clReleaseKernel(k_last_stage_dit_unweight_wg64);
        if (k_last_stage_dit_unweight_wg16) clReleaseKernel(k_last_stage_dit_unweight_wg16);
        if (k_last_stage_dit_unweight) clReleaseKernel(k_last_stage_dit_unweight);
        if (k_ntt_stage_dit_radix4x2_wg128) clReleaseKernel(k_ntt_stage_dit_radix4x2_wg128);
        if (k_ntt_stage_dit_radix4x2) clReleaseKernel(k_ntt_stage_dit_radix4x2);
        if (k_ntt_stage_dit_radix4_wg128) clReleaseKernel(k_ntt_stage_dit_radix4_wg128);
        if (k_ntt_stage_dit_radix4) clReleaseKernel(k_ntt_stage_dit_radix4);
        if (k_ntt_stage_dif_radix4x2_wg128) clReleaseKernel(k_ntt_stage_dif_radix4x2_wg128);
        if (k_ntt_stage_dif_radix4x2) clReleaseKernel(k_ntt_stage_dif_radix4x2);
        if (k_ntt_stage_dif_radix4_wg128) clReleaseKernel(k_ntt_stage_dif_radix4_wg128);
        if (k_ntt_stage_dif_radix4) clReleaseKernel(k_ntt_stage_dif_radix4);
        if (k_ntt_stage_dit) clReleaseKernel(k_ntt_stage_dit);
        if (k_ntt_stage_dif_len2048) clReleaseKernel(k_ntt_stage_dif_len2048);
        if (k_ntt_stage_dif) clReleaseKernel(k_ntt_stage_dif);
        if (k_weight_first_stage_dif_radix4_wg128) clReleaseKernel(k_weight_first_stage_dif_radix4_wg128);
        if (k_weight_first_stage_dif_radix4_wg64) clReleaseKernel(k_weight_first_stage_dif_radix4_wg64);
        if (k_weight_first_stage_dif_wg64) clReleaseKernel(k_weight_first_stage_dif_wg64);
        if (k_weight_first_stage_dif_wg16) clReleaseKernel(k_weight_first_stage_dif_wg16);
        if (k_weight_first_stage_dif) clReleaseKernel(k_weight_first_stage_dif);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

static std::vector<gf61::Elem> build_stage_twiddles(std::size_t n, bool inverse, std::vector<StageInfo>& stages_out) {
    gf61::Elem root = gf61::primitive_root_pow2(n);
    if (inverse) root = gf61::inv(root);
    std::vector<gf61::Elem> all;
    stages_out.clear();
    std::uint32_t offset = 0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        gf61::Elem wlen = root;
        for (std::size_t step = len; step < n; step <<= 1) wlen = gf61::sqr(wlen);
        const std::size_t half_len = len >> 1;
        stages_out.push_back(StageInfo{static_cast<std::uint32_t>(len), static_cast<std::uint32_t>(half_len), offset});
        gf61::Elem w(1, 0);
        for (std::size_t j = 0; j < half_len; ++j) {
            all.push_back(w);
            w = gf61::mul(w, wlen);
        }
        offset += static_cast<std::uint32_t>(half_len);
    }

    return all;
}

struct PackedElem32 {
    std::uint32_t a;
    std::uint32_t b;
};

static std::size_t current_field_elem_size() {
    return (gf61::FIELD_BITS == 31) ? sizeof(PackedElem32) : sizeof(gf61::Elem);
}

static std::vector<std::uint8_t> pack_field_elems(const std::vector<gf61::Elem>& in) {
    if (gf61::FIELD_BITS != 31) {
        std::vector<std::uint8_t> out(in.size() * sizeof(gf61::Elem));
        if (!in.empty()) std::memcpy(out.data(), in.data(), out.size());
        return out;
    }
    std::vector<std::uint8_t> out(in.size() * sizeof(PackedElem32));
    auto* p32 = reinterpret_cast<PackedElem32*>(out.data());
    constexpr std::uint64_t mask = (1ull << 31) - 1ull;
    for (std::size_t i = 0; i < in.size(); ++i) {
        p32[i].a = static_cast<std::uint32_t>(in[i].a & mask);
        p32[i].b = static_cast<std::uint32_t>(in[i].b & mask);
    }
    return out;
}

GpuPrp make_gpu(const DeviceInfo& info,
                const std::string& kernel_path,
                const ibdwt::Layout& layout,
                bool profile_kernels = false,
                bool prefer_radix4x2 = true,
                cl_context shared_context = nullptr,
                cl_command_queue shared_queue = nullptr) {
    GpuPrp gpu;
    gpu.profile_kernels = profile_kernels;
    gpu.prefer_radix4x2 = prefer_radix4x2;
    gpu.field_bits = gf61::FIELD_BITS;
    gpu.field_elem_size = current_field_elem_size();
    cl_int err = CL_SUCCESS;
    if (shared_context) {
        gpu.context = shared_context;
        check(clRetainContext(gpu.context), "clRetainContext");
    } else {
        gpu.context = clCreateContext(nullptr, 1, &info.device, nullptr, nullptr, &err);
        check(err, "clCreateContext");
    }
    if (shared_queue) {
        gpu.queue = shared_queue;
        check(clRetainCommandQueue(gpu.queue), "clRetainCommandQueue");
    } else {
#if defined(CL_VERSION_2_0)
    if (profile_kernels) {
        const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        gpu.queue = clCreateCommandQueueWithProperties(gpu.context, info.device, props, &err);
        if (err != CL_SUCCESS || !gpu.queue) {
            err = CL_SUCCESS;
            gpu.queue = clCreateCommandQueue(gpu.context, info.device, CL_QUEUE_PROFILING_ENABLE, &err);
        }
    } else {
        gpu.queue = clCreateCommandQueue(gpu.context, info.device, 0, &err);
    }
#else
    gpu.queue = clCreateCommandQueue(gpu.context, info.device, profile_kernels ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
#endif
    check(err, profile_kernels ? "clCreateCommandQueue(profile)" : "clCreateCommandQueue");
    }

    const std::string source = load_text_file(kernel_path);
    std::string build_opts = std::string("-DFIELD_BITS=") + std::to_string(gf61::FIELD_BITS);
    if (std::getenv("PRMERS_GF31_4MUL")) build_opts += " -DCRT_GF31_KARATSUBA=0";
    build_opts += std::string(" -DCRT_MIXED_F48_DELAYED_SCALE_61=") +
                  (mixed_center_f48_delayed_scale_61() ? "1" : "0");
    build_opts += std::string(" -DCRT_MIXED_F48_DELAYED_SCALE_31=") +
                  (mixed_center_f48_delayed_scale_31() ? "1" : "0");
    build_opts += std::string(" -DCRT_MIXED_F48_TWIN_SYMMETRY_61=") +
                  (mixed_center_f48_twin_symmetry_61() ? "1" : "0");
    build_opts += std::string(" -DCRT_MIXED_F48_TWIN_SYMMETRY_31=") +
                  (mixed_center_f48_twin_symmetry_31() ? "1" : "0");
    if (parse_bool_env("PRMERS_OCL_FAST_BUILD_OPTS", true)) {
        build_opts += " -cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math";
    }
    if (const char* extra = std::getenv("PRMERS_OCL_FLAGS")) {
        build_opts += " ";
        build_opts += extra;
    }
    const bool show_build = parse_bool_env("PRMERS_SHOW_OCL_BUILD", true);
    const bool show_spinner = parse_bool_env("PRMERS_OCL_BUILD_SPINNER", true);
    const bool binary_cache = parse_bool_env("PRMERS_OCL_BINARY_CACHE", true);
    std::filesystem::path cache_path;
    bool cache_hit = false;
    if (binary_cache) {
        cache_path = opencl_cache_path(info.device, kernel_path, source, build_opts, gf61::FIELD_BITS);
        const std::vector<unsigned char> bin = load_binary_file(cache_path);
        if (!bin.empty()) {
            const unsigned char* bptr = bin.data();
            size_t bsize = bin.size();
            cl_int bin_status = CL_SUCCESS;
            gpu.program = clCreateProgramWithBinary(gpu.context, 1, &info.device, &bsize, &bptr, &bin_status, &err);
            if (err == CL_SUCCESS && bin_status == CL_SUCCESS && gpu.program) {
                err = clBuildProgram(gpu.program, 1, &info.device, build_opts.c_str(), nullptr, nullptr);
                if (err == CL_SUCCESS) {
                    cache_hit = true;
                    if (show_build) {
                        std::cerr << "OpenCL build: loaded binary cache for GF(M" << gf61::FIELD_BITS
                                  << "^2): " << cache_path.string() << std::endl;
                    }
                } else {
                    clReleaseProgram(gpu.program);
                    gpu.program = nullptr;
                    err = CL_SUCCESS;
                    if (show_build) std::cerr << "OpenCL build: binary cache rejected, compiling source" << std::endl;
                }
            } else {
                if (gpu.program) clReleaseProgram(gpu.program);
                gpu.program = nullptr;
                err = CL_SUCCESS;
                if (show_build) std::cerr << "OpenCL build: binary cache invalid, compiling source" << std::endl;
            }
        }
    }
    if (!cache_hit) {
        const char* src_ptr = source.c_str();
        const size_t src_len = source.size();
        gpu.program = clCreateProgramWithSource(gpu.context, 1, &src_ptr, &src_len, &err);
        check(err, "clCreateProgramWithSource");
        std::atomic<bool> build_done{false};
        std::thread build_watchdog;
        if (show_build) {
            std::cerr << "OpenCL build: compiling " << kernel_path
                      << " for GF(M" << gf61::FIELD_BITS << "^2)"
                      << " with options: " << build_opts << "\n" << std::flush;
            if (show_spinner) {
                std::cerr << "OpenCL build: waiting for GF(M" << gf61::FIELD_BITS
                          << "^2) [not stuck] ..." << std::flush;
            }
            build_watchdog = std::thread([&build_done, show_spinner]() {
                using clock = std::chrono::steady_clock;
                const auto start = clock::now();
                const char spin[4] = {'|', '/', '-', '\\'};
                unsigned tick = 0;
                while (!build_done.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    if (build_done.load(std::memory_order_relaxed)) break;
                    const double sec = std::chrono::duration<double>(clock::now() - start).count();
                    if (show_spinner) {
                        std::cerr << "\rOpenCL build: waiting " << spin[tick++ & 3]
                                  << " " << std::fixed << std::setprecision(1) << sec
                                  << " s [not stuck]" << std::flush;
                    } else if (static_cast<int>(sec) > 0 && (static_cast<int>(sec) % 5) == 0) {
                        std::cerr << "OpenCL build: still compiling after "
                                  << std::fixed << std::setprecision(0) << sec
                                  << " s..." << std::endl;
                    }
                }
            });
        }
        err = clBuildProgram(gpu.program, 1, &info.device, build_opts.c_str(), nullptr, nullptr);
        build_done.store(true, std::memory_order_relaxed);
        if (build_watchdog.joinable()) build_watchdog.join();
        if (show_build) {
            if (show_spinner) {
                std::cerr << "\rOpenCL build: done for GF(M" << gf61::FIELD_BITS
                          << "^2)                                    " << std::endl;
            } else {
                std::cerr << "OpenCL build: done for GF(M" << gf61::FIELD_BITS << "^2)" << std::endl;
            }
        }
        if (err == CL_SUCCESS && binary_cache) {
            if (save_opencl_binary(gpu.program, cache_path)) {
                if (show_build) std::cerr << "OpenCL build: saved binary cache: " << cache_path.string() << std::endl;
            } else if (show_build) {
                std::cerr << "OpenCL build: could not save binary cache" << std::endl;
            }
        }
    }
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(gpu.program, info.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size) clGetProgramBuildInfo(gpu.program, info.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        throw std::runtime_error("Build failed:\n" + log);
    }

    auto make_kernel = [&](const char* name, cl_kernel* out) {
        *out = clCreateKernel(gpu.program, name, &err);
        check(err, name);
    };
    auto make_optional_kernel = [&](const char* name, cl_kernel* out) {
        *out = clCreateKernel(gpu.program, name, &err);
        if (err != CL_SUCCESS) {
            *out = nullptr;
            err = CL_SUCCESS;
        }
    };
    make_kernel("gf61_weight_first_stage_dif", &gpu.k_weight_first_stage_dif);
    make_kernel("gf61_weight_first_stage_dif_wg16", &gpu.k_weight_first_stage_dif_wg16);
    make_kernel("gf61_weight_first_stage_dif_wg64", &gpu.k_weight_first_stage_dif_wg64);
    make_kernel("gf61_weight_first_stage_dif_radix4_wg64", &gpu.k_weight_first_stage_dif_radix4_wg64);
    make_kernel("gf61_weight_first_stage_dif_radix4_wg128", &gpu.k_weight_first_stage_dif_radix4_wg128);
    make_kernel("gf61_ntt_stage_dif", &gpu.k_ntt_stage_dif);
    make_kernel("gf61_ntt_stage_dif_len2048", &gpu.k_ntt_stage_dif_len2048);
    make_kernel("gf61_ntt_stage_dit", &gpu.k_ntt_stage_dit);
    make_kernel("gf61_ntt_stage_dif_radix4", &gpu.k_ntt_stage_dif_radix4);
    make_kernel("gf61_ntt_stage_dif_radix4_wg128", &gpu.k_ntt_stage_dif_radix4_wg128);
    make_kernel("gf61_ntt_stage_dif_radix4x2", &gpu.k_ntt_stage_dif_radix4x2);
    make_kernel("gf61_ntt_stage_dif_radix4x2_wg128", &gpu.k_ntt_stage_dif_radix4x2_wg128);
    make_kernel("gf61_ntt_stage_dit_radix4", &gpu.k_ntt_stage_dit_radix4);
    make_kernel("gf61_ntt_stage_dit_radix4_wg128", &gpu.k_ntt_stage_dit_radix4_wg128);
    make_kernel("gf61_ntt_stage_dit_radix4x2", &gpu.k_ntt_stage_dit_radix4x2);
    make_kernel("gf61_ntt_stage_dit_radix4x2_wg128", &gpu.k_ntt_stage_dit_radix4x2_wg128);
    make_kernel("gf61_last_stage_dit_unweight", &gpu.k_last_stage_dit_unweight);
    make_kernel("gf61_last_stage_dit_unweight_wg16", &gpu.k_last_stage_dit_unweight_wg16);
    make_kernel("gf61_last_stage_dit_unweight_wg64", &gpu.k_last_stage_dit_unweight_wg64);
    make_kernel("gf61_last_stage_dit_radix4_unweight_wg64", &gpu.k_last_stage_dit_radix4_unweight_wg64);
    make_kernel("gf61_last_stage_dit_radix4_unweight_wg128", &gpu.k_last_stage_dit_radix4_unweight_wg128);
    make_kernel("gf61_pointwise_sqr", &gpu.k_pointwise_sqr);
    make_kernel("gf61_center_fused_8", &gpu.k_center_fused_8);
    make_kernel("gf61_center_fused_16", &gpu.k_center_fused_16);
    make_kernel("gf61_center_fused_32", &gpu.k_center_fused_32);
    make_kernel("gf61_center_fused_64", &gpu.k_center_fused_64);
    make_kernel("gf61_center_fused_128", &gpu.k_center_fused_128);
    make_kernel("gf61_center_fused_256", &gpu.k_center_fused_256);
    make_kernel("gf61_center_fused_256_explicit", &gpu.k_center_fused_256_explicit);
    make_kernel("gf61_center_fused_512", &gpu.k_center_fused_512);
    make_kernel("gf61_center_fused_1024", &gpu.k_center_fused_1024);
    make_kernel("gf61_center_fused_2048", &gpu.k_center_fused_2048);
    make_optional_kernel("gf61_center_fused_4096", &gpu.k_center_fused_4096);
    make_kernel("gf61_forward_bridge_64_to_16", &gpu.k_forward_bridge_64_to_16);
    make_kernel("gf61_inverse_bridge_16_to_64", &gpu.k_inverse_bridge_16_to_64);
    make_kernel("gf61_forward_bridge_256_to_64", &gpu.k_forward_bridge_256_to_64);
    make_kernel("gf61_inverse_bridge_64_to_256", &gpu.k_inverse_bridge_64_to_256);
    make_kernel("gf61_forward_bridge_512_to_256", &gpu.k_forward_bridge_512_to_256);
    make_kernel("gf61_inverse_bridge_256_to_512", &gpu.k_inverse_bridge_256_to_512);
    make_kernel("gf61_forward_bridge_1024_to_512", &gpu.k_forward_bridge_1024_to_512);
    make_kernel("gf61_inverse_bridge_512_to_1024", &gpu.k_inverse_bridge_512_to_1024);
    make_kernel("gf61_forward_bridge_1024_to_256", &gpu.k_forward_bridge_1024_to_256);
    make_kernel("gf61_inverse_bridge_256_to_1024", &gpu.k_inverse_bridge_256_to_1024);
    make_kernel("gf61_forward_bridge_2048_to_256", &gpu.k_forward_bridge_2048_to_256);
    make_kernel("gf61_inverse_bridge_256_to_2048", &gpu.k_inverse_bridge_256_to_2048);
    make_kernel("gf61_forward_ext_1024_to_256_explicit", &gpu.k_forward_ext_1024_to_256_explicit);
    make_kernel("gf61_forward_ext_1024_to_256_explicit_wg128", &gpu.k_forward_ext_1024_to_256_explicit_wg128);
    make_kernel("gf61_inverse_ext_256_to_1024_explicit", &gpu.k_inverse_ext_256_to_1024_explicit);
    make_kernel("gf61_inverse_ext_256_to_1024_explicit_wg128", &gpu.k_inverse_ext_256_to_1024_explicit_wg128);
    make_kernel("gf61_forward_ext_2048_to_256_explicit", &gpu.k_forward_ext_2048_to_256_explicit);
    make_kernel("gf61_inverse_ext_256_to_2048_explicit", &gpu.k_inverse_ext_256_to_2048_explicit);
    make_kernel("gf61_mul_small_digits", &gpu.k_mul_small);
    make_kernel("gf61_carry_segment_first", &gpu.k_carry_block_local);
    make_kernel("gf61_carry_segment_pass", &gpu.k_carry_block_prefix);
    make_kernel("gf61_carry_cleanup_serial_segments", &gpu.k_carry_final_wrap);
    make_kernel("gf61_carry_clear_pending", &gpu.k_carry_clear_pending);
    make_kernel("gf61_crt_garner_segment_first", &gpu.k_crt_garner_segment_first);
    make_kernel("gf61_crt_carry_segment_pass", &gpu.k_crt_carry_segment_pass);
    make_kernel("gf61_crt_carry_cleanup_serial", &gpu.k_crt_carry_cleanup_serial);
    make_kernel("gf61_crt_garner_segment_first_oneout", &gpu.k_crt_garner_segment_first_oneout);
    make_kernel("gf61_crt_garner_segment_first_oneout_mask32", &gpu.k_crt_garner_segment_first_oneout_mask32);
    make_kernel("gf61_crt_garner_segment_first_oneout_mask32_base32_fast", &gpu.k_crt_garner_segment_first_oneout_mask32_base32_fast);
    make_kernel("gf61_crt_garner_segment_first_oneout_mask32_base32_x2", &gpu.k_crt_garner_segment_first_oneout_mask32_base32_x2);
    make_kernel("gf61_crt_garner_segment_first_oneout_mask32_base32_u32lean", &gpu.k_crt_garner_segment_first_oneout_mask32_base32_u32lean);
    make_kernel("gf61_crt_garner_segment_first_oneout_coeffhi_mask32_base32", &gpu.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32);
    make_kernel("gf61_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2", &gpu.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2);
    make_kernel("gf61_crt_last_unweight_garner_segment_first_oneout", &gpu.k_crt_last_unweight_garner_segment_first_oneout);
    make_kernel("gf61_crt_last_unweight_garner_segment_first_oneout_mask32", &gpu.k_crt_last_unweight_garner_segment_first_oneout_mask32);
    make_kernel("gf61_crt_carry_segment_pass_oneout", &gpu.k_crt_carry_segment_pass_oneout);
    make_kernel("gf61_crt_carry_cleanup_serial_oneout", &gpu.k_crt_carry_cleanup_serial_oneout);
    make_kernel("gf61_crt_carry_cleanup_parallel_oneout", &gpu.k_crt_carry_cleanup_parallel_oneout);
    make_kernel("gf61_crt_mixed_carry_pack_next_lds_61x31", &gpu.k_crt_mixed_carry_pack_next_lds_61x31);

    gpu.n = layout.n;
    gpu.exponent_p = static_cast<cl_uint>(layout.p);
    gpu.lr2 = static_cast<cl_uint>(ibdwt::log2_root_two(layout.n));
    {
        cl_uint min_width = 64u;
        for (std::uint8_t w : layout.digit_width) {
            if (w != 0u && static_cast<cl_uint>(w) < min_width) min_width = static_cast<cl_uint>(w);
        }
        if (min_width == 0u || min_width > 64u) min_width = 8u;
        gpu.min_digit_width = min_width;
        gpu.carry_passes = std::max<cl_uint>(4u, std::min<cl_uint>(12u, ((64u + min_width - 1u) / min_width) + 2u)) - 1;
    }
    gpu.max_work_group_size = info.max_work_group_size;
    gpu.local_mem_size = info.local_mem_size;
    const std::size_t n = layout.n;
    gpu.bufDigits = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, n * sizeof(std::uint64_t), nullptr, &err);
    check(err, "clCreateBuffer(bufDigits)");
    if (gpu.field_bits == 31) {
        
        
        gpu.bufDigits32 = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, n * sizeof(std::uint32_t), nullptr, &err);
        check(err, "clCreateBuffer(bufDigits32)");
    }
    gpu.bufField = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, n * gpu.field_elem_size, nullptr, &err);
    check(err, "clCreateBuffer(bufField)");
    gpu.bufWidth = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(std::uint8_t), const_cast<std::uint8_t*>(layout.digit_width.data()), &err);
    check(err, "clCreateBuffer(bufWidth)");

    const auto shift_table = ibdwt::make_unweight_shift_table(layout, static_cast<std::uint32_t>(gpu.field_bits), gpu.lr2);
    gpu.bufUnweightShift = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          n * sizeof(std::uint8_t), const_cast<std::uint8_t*>(shift_table.data()), &err);
    check(err, "clCreateBuffer(bufUnweightShift)");

    
    std::vector<std::uint32_t> width_mask32((n + 31u) >> 5, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        if (layout.digit_width[i] != gpu.min_digit_width) width_mask32[i >> 5] |= (std::uint32_t(1) << (i & 31u));
    }
    gpu.bufWidthMask32 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        width_mask32.size() * sizeof(std::uint32_t),
                                        width_mask32.data(), &err);
    check(err, "clCreateBuffer(bufWidthMask32)");

    const std::size_t twiddle_n = (layout.odd > 1u) ? layout.pow2_n : n;
    std::vector<StageInfo> stages_fwd;
    auto tw_fwd = build_stage_twiddles(twiddle_n, false, stages_fwd);
    std::vector<StageInfo> stages_inv;
    auto tw_inv = build_stage_twiddles(twiddle_n, true, stages_inv);
    if (stages_fwd.size() != stages_inv.size()) throw std::runtime_error("stage count mismatch");
    gpu.stages = stages_fwd;
    gpu.log_n = static_cast<cl_uint>(stages_fwd.size());

    auto tw_fwd_packed = pack_field_elems(tw_fwd);
    auto tw_inv_packed = pack_field_elems(tw_inv);
    gpu.bufTwFwd = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tw_fwd_packed.size(), tw_fwd_packed.data(), &err);
    check(err, "clCreateBuffer(bufTwFwd)");
    gpu.bufTwInv = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tw_inv_packed.size(), tw_inv_packed.data(), &err);
    check(err, "clCreateBuffer(bufTwInv)");

    if (layout.odd > 1u) {
        const std::size_t odd = layout.odd;
        const std::uint64_t root = gf61::primitive_root_odd_scalar(odd);
        const std::uint64_t root_inv = gf61::pow_mod(root, odd - 1u);
        const std::uint64_t inv_odd = gf61::pow_mod(static_cast<std::uint64_t>(odd), gf61::P - 2u);
        std::vector<gf61::Elem> odd_fwd(odd * odd), odd_inv(odd * odd);
        for (std::size_t k = 0; k < odd; ++k) {
            for (std::size_t j = 0; j < odd; ++j) {
                odd_fwd[k * odd + j] = gf61::Elem(gf61::pow_mod(root, static_cast<std::uint64_t>(j * k)), 0);
                const std::uint64_t w = gf61::pow_mod(root_inv, static_cast<std::uint64_t>(j * k));
                odd_inv[k * odd + j] = gf61::Elem(gf61::mul_mod(w, inv_odd), 0);
            }
        }
        auto odd_fwd_packed = pack_field_elems(odd_fwd);
        auto odd_inv_packed = pack_field_elems(odd_inv);
        gpu.bufOddFwd = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, odd_fwd_packed.size(), odd_fwd_packed.data(), &err);
        check(err, "clCreateBuffer(bufOddFwd)");
        gpu.bufOddInv = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, odd_inv_packed.size(), odd_inv_packed.data(), &err);
        check(err, "clCreateBuffer(bufOddInv)");
        gpu.bufOddScratch = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, n * gpu.field_elem_size, nullptr, &err);
        check(err, "clCreateBuffer(bufOddScratch)");
    }

    return gpu;
}

static bool g_planner_debug = false;
static std::string g_single_center_mode = "normal";
static bool g_local_block_lds_disabled = false;
static cl_uint g_local_block_lds_override = 0;
static cl_uint g_last_local_block_lds = 0;
static bool g_force_strict_reference = false;
static bool g_disable_crt_fused_pipeline = false;
static bool g_crt_radix8_global = true;
static cl_uint g_crt_center_chunk = 512u;
static cl_uint g_crt_lds_stage = 0u;
static cl_uint g_crt_lds_tile = 2u;
static cl_uint g_crt_head_radix8 = 0u;


static int g_crt_defused_schedule = 0;
static int g_crt_fwd8_61_wg = 64;

static int g_crt_defused_edge_fuse = 0;
static uint32_t g_crt_defused_edge_radix = 4u;
static uint32_t g_crt_odd_radix = 9u;
static std::string g_crt_mixed_row_core = "auto";
static std::string g_crt_mixed_row_fuse_both = "off";
static std::string g_crt_center_mode = "halfreal";
static bool g_crt_halfreal_validate = false;
static uint32_t g_crt_halfreal_validate_iters = 1;
static bool g_crt_halfreal_validate_random = false;
// For large mixed odd-radix tests the exact CPU reference can be far too slow.
// This switch validates the selected LDS/fused mixed path against the generic
// mixed GPU path, keeping the same PFA digit order and avoiding the CPU square.
static bool g_crt_mixed_gpu_reference = false;
// v57 experimental: carry boundary can pre-pack the next iteration.
// The next square then starts directly from a61/a31 and skips the head pack.
static bool g_crt_mixed_skip_pack_this_square = false;
// v58 experimental: fuse the post-Garner carry pass with the next mixed pack.
// This is only armed by the main mixed loop for iter+1 and only when the tail
// already produced the first carry vectors.
static bool g_crt_mixed_carry_pack_next_request = false;
static bool g_crt_mixed_carry_pack_next_done = false;
static std::size_t g_crt_halfreal_dump_count = 32;
static std::string g_crt_halfreal_dump_prefix = "halfreal_debug";


static int g_crt_halfreal_flags61 = 48;
static int g_crt_halfreal_flags31 = 48;
static bool g_crt_halfreal_autoprobe = false;
static bool g_crt_halfreal_probe_exhaustive = true;


static int g_crt_defused_edge_mode = 0;

static inline cl_uint crt_halfreal_effective_flags61() {
    
    
    return static_cast<cl_uint>((g_crt_halfreal_flags61 < 0) ? 16 : (g_crt_halfreal_flags61 & 63));
}

static inline cl_uint crt_halfreal_effective_flags31() {
    return static_cast<cl_uint>((g_crt_halfreal_flags31 < 0) ? crt_halfreal_effective_flags61() : (g_crt_halfreal_flags31 & 63));
}

static std::string crt_halfreal_flag_desc(unsigned f) {
    std::ostringstream oss;
    if (f & 16u) oss << "bitrev";
    else if (f & 1u) oss << "radix8-digitrev";
    else oss << "linear";
    oss << ",W=" << (f & 2u ? "twi" : "twf")
        << ",O=" << (f & 4u ? "+i" : "-i")
        << ",pack=" << (f & 8u ? "E-iO" : "E+iO")
        << ((f & 32u) ? ",pair-fast" : "");
    return oss.str();
}

static std::string crt_halfreal_flags_desc() {
    std::ostringstream oss;
    const unsigned f61 = crt_halfreal_effective_flags61();
    const unsigned f31 = crt_halfreal_effective_flags31();
    oss << "M61=" << f61 << " [" << crt_halfreal_flag_desc(f61) << "]"
        << ", M31=" << f31 << " [" << crt_halfreal_flag_desc(f31) << "]";
    return oss.str();
}


struct CarryConfig {
    cl_uint block_size = 256;
    cl_uint items_per_worker = 4;
    cl_uint local_size = 64;
    cl_uint num_blocks = 0;
};

static CarryConfig choose_carry_config(const DeviceInfo& dev, std::size_t n, cl_uint block_override, cl_uint items_override) {
    auto fits = [&](cl_uint block_size, cl_uint items_per_worker) -> bool {
        if (block_size == 0 || items_per_worker == 0 || (block_size % items_per_worker) != 0) return false;
        const cl_uint local_size = block_size / items_per_worker;
        const std::size_t local_bytes = std::size_t(block_size) * (sizeof(std::uint64_t) + sizeof(std::uint8_t));
        return local_size <= dev.max_work_group_size && local_bytes <= static_cast<std::size_t>(dev.local_mem_size);
    };

    CarryConfig cfg;
    if (block_override != 0) cfg.block_size = block_override;
    if (items_override != 0) cfg.items_per_worker = items_override;

    const bool user_overrode = (block_override != 0 || items_override != 0);
    if (!user_overrode) {
        const bool gfx906_like = (dev.max_work_group_size <= 256 && dev.local_mem_size <= 65536);
        if (gfx906_like) {
            if (n <= 1024u && fits(64u, 4u)) {
                cfg.block_size = 64u;
                cfg.items_per_worker = 4u;
            } else if (n >= (1u << 20) && fits(1024u, 64u)) {
                cfg.block_size = 1024u;
                cfg.items_per_worker = 64u;
            } else if (n >= (1u << 16) && fits(512u, 32u)) {
                cfg.block_size = 512u;
                cfg.items_per_worker = 32u;
            }
        }
    }

    if (cfg.block_size == 0 || cfg.items_per_worker == 0 || (cfg.block_size % cfg.items_per_worker) != 0) {
        throw std::runtime_error("carry config requires block_size % items_per_worker == 0 and both non-zero");
    }
    cfg.local_size = cfg.block_size / cfg.items_per_worker;

    if (!fits(cfg.block_size, cfg.items_per_worker)) {
        const std::pair<cl_uint, cl_uint> fallbacks[] = {
            {1024u,64u}, {512u,32u}, {256u,4u}, {128u,4u}, {64u,4u}, {64u,2u}
        };
        bool found = false;
        for (auto [b, v] : fallbacks) {
            if (fits(b, v)) {
                cfg.block_size = b;
                cfg.items_per_worker = v;
                cfg.local_size = b / v;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("no valid carry block configuration for this device");
    }

    cfg.num_blocks = static_cast<cl_uint>((n + cfg.block_size - 1) / cfg.block_size);
    return cfg;
}


static CarryConfig choose_crt_carry_config(const DeviceInfo& dev, std::size_t n, cl_uint block_override, cl_uint items_override) {
    auto fits = [&](cl_uint block_size, cl_uint items_per_worker) -> bool {
        if (block_size == 0 || items_per_worker == 0 || (block_size % items_per_worker) != 0) return false;
        const cl_uint local_size = block_size / items_per_worker;
        return local_size <= dev.max_work_group_size;
    };

    CarryConfig cfg;
    if (block_override != 0) cfg.block_size = block_override;
    if (items_override != 0) cfg.items_per_worker = items_override;

    const bool user_overrode = (block_override != 0 || items_override != 0);
    if (!user_overrode) {
        const std::pair<cl_uint, cl_uint> preferred[] = {
            
            
            {512u,32u}, {256u,16u}, {256u,8u}, {1024u,64u}, {256u,4u}, {128u,4u}, {64u,2u}
        };
        bool found = false;
        for (auto [b, v] : preferred) {
            if (fits(b, v)) {
                cfg.block_size = b;
                cfg.items_per_worker = v;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("no valid CRT carry block configuration for this device");
    }

    if (cfg.block_size == 0 || cfg.items_per_worker == 0 || (cfg.block_size % cfg.items_per_worker) != 0) {
        throw std::runtime_error("CRT carry config requires block_size % items_per_worker == 0 and both non-zero");
    }
    cfg.local_size = cfg.block_size / cfg.items_per_worker;

    if (!fits(cfg.block_size, cfg.items_per_worker)) {
        const std::pair<cl_uint, cl_uint> fallbacks[] = {
            {512u,32u}, {256u,16u}, {256u,8u}, {1024u,64u}, {256u,4u}, {128u,4u}, {64u,2u}
        };
        bool found = false;
        for (auto [b, v] : fallbacks) {
            if (fits(b, v)) {
                cfg.block_size = b;
                cfg.items_per_worker = v;
                cfg.local_size = b / v;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("no valid CRT carry block configuration for this device");
    }

    cfg.num_blocks = static_cast<cl_uint>((n + cfg.block_size - 1) / cfg.block_size);
    return cfg;
}


struct CenterKernelConfig {
    cl_kernel kernel = nullptr;
    cl_uint chunk = 0;
    cl_uint local_size = 0;
    bool enabled = false;
};

struct BridgeKernelConfig {
    cl_kernel forward_kernel = nullptr;
    cl_kernel inverse_kernel = nullptr;
    cl_uint outer_chunk = 0;
    cl_uint inner_chunk = 0;
    cl_uint local_size = 0;
    bool enabled = false;
};

static bool is_gfx906_like(const GpuPrp& gpu) {
    return gpu.max_work_group_size <= 256 && gpu.local_mem_size <= 65536;
}

static void profile_record_completed_event(GpuPrp& gpu, const std::string& label, cl_event ev) {
    if (!gpu.profile_kernels || !ev) return;
    cl_ulong t0 = 0, t1 = 0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr) == CL_SUCCESS &&
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr) == CL_SUCCESS && t1 >= t0) {
        auto it = gpu.profile_totals.find(label);
        if (it == gpu.profile_totals.end()) gpu.profile_order.push_back(label);
        auto& slot = gpu.profile_totals[label];
        slot.ms += double(t1 - t0) * 1e-6;
        slot.launches += 1;
    }
}

static void profile_flush_pending(GpuPrp& gpu) {
    if (!gpu.profile_kernels) return;
    for (auto& item : gpu.pending_profile_events) {
        if (item.second) {
            clWaitForEvents(1, &item.second);
            profile_record_completed_event(gpu, item.first, item.second);
            clReleaseEvent(item.second);
        }
    }
    gpu.pending_profile_events.clear();
}


static void profile_clear(GpuPrp& gpu) {
    for (auto& item : gpu.pending_profile_events) {
        if (item.second) clReleaseEvent(item.second);
    }
    gpu.pending_profile_events.clear();
    gpu.profile_order.clear();
    gpu.profile_totals.clear();
}

static bool profile_has_data(const GpuPrp& gpu) {
    return gpu.profile_kernels && !gpu.profile_order.empty();
}

static void profile_print_summary(const GpuPrp& gpu, const std::string& title = "Kernel profile summary") {
    if (!gpu.profile_kernels) return;
    if (gpu.profile_order.empty()) {
        std::cout << title << ": no event timing data collected.\n";
        return;
    }
    double total_ms = 0.0;
    for (const auto& name : gpu.profile_order) total_ms += gpu.profile_totals.at(name).ms;
    if (total_ms <= 0.0) {
        std::cout << title << ": profiling events were recorded but timings are unavailable on this queue/device.\n";
        return;
    }
    std::cout << title << ":\n";
    std::vector<std::pair<std::string, ProfileEntry>> rows;
    rows.reserve(gpu.profile_order.size());
    for (const auto& name : gpu.profile_order) rows.push_back({name, gpu.profile_totals.at(name)});
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    for (const auto& row : rows) {
        const double pct = 100.0 * row.second.ms / total_ms;
        std::cout << "  " << row.first << ": " << std::fixed << std::setprecision(3) << row.second.ms
                  << " ms total (" << std::setprecision(1) << pct << "%, launches=" << row.second.launches << ")\n";
    }
}

static inline size_t round_up_size(size_t value, size_t align) {
    if (align == 0) return value;
    return ((value + align - 1) / align) * align;
}

static size_t clamp_local_to_global(size_t preferred, size_t global) {
    if (global == 0) return preferred;
    size_t local = preferred;
    while (local > 1 && (local > global || (global % local) != 0)) local >>= 1;
    return local ? local : 1;
}

static void set_pending_wait_event(GpuPrp& gpu, cl_event ev) {
    if (gpu.pending_wait_event) clReleaseEvent(gpu.pending_wait_event);
    gpu.pending_wait_event = ev;
}

static cl_event take_pending_wait_event(GpuPrp& gpu) {
    cl_event ev = gpu.pending_wait_event;
    gpu.pending_wait_event = nullptr;
    return ev;
}

static void release_pending_wait_event(GpuPrp& gpu) {
    if (gpu.pending_wait_event) {
        clReleaseEvent(gpu.pending_wait_event);
        gpu.pending_wait_event = nullptr;
    }
}

static cl_event enqueue_queue_marker(GpuPrp& gpu, const char* what) {
    cl_event ev = nullptr;
    cl_event wait_ev = take_pending_wait_event(gpu);
    const cl_uint wait_count = wait_ev ? 1u : 0u;
    const cl_event* wait_list = wait_ev ? &wait_ev : nullptr;
#if defined(CL_VERSION_1_2)
    check(clEnqueueMarkerWithWaitList(gpu.queue, wait_count, wait_list, &ev), what);
#else
    if (wait_ev) check(clEnqueueWaitForEvents(gpu.queue, wait_count, wait_list), what);
    check(clEnqueueMarker(gpu.queue, &ev), what);
#endif
    if (wait_ev) clReleaseEvent(wait_ev);
    return ev;
}

static void enqueue_kernel(GpuPrp& gpu, cl_kernel kernel, size_t global, const size_t* local, const char* what, const char* profile_label) {
    cl_event ev = nullptr;
    cl_event wait_ev = take_pending_wait_event(gpu);
    const cl_uint wait_count = wait_ev ? 1u : 0u;
    const cl_event* wait_list = wait_ev ? &wait_ev : nullptr;
    check(clEnqueueNDRangeKernel(gpu.queue, kernel, 1, nullptr, &global, local,
                                 wait_count, wait_list, gpu.profile_kernels ? &ev : nullptr), what);
    if (wait_ev) clReleaseEvent(wait_ev);
    if (gpu.profile_kernels && ev) gpu.pending_profile_events.push_back({profile_label, ev});
}

static cl_uint choose_default_center_cap(const GpuPrp& gpu) {
    if (is_gfx906_like(gpu)) {
        if (gpu.log_n >= 20u) return 256u;
        if (gpu.log_n >= 16u) return 256u;
        return 512u;
    }
    return 2048u;
}

static CenterKernelConfig choose_center_kernel(GpuPrp& gpu, cl_uint center_max = 0) {
    const cl_uint cap = center_max ? center_max : choose_default_center_cap(gpu);
    struct Candidate { cl_uint chunk; cl_uint wg; cl_kernel GpuPrp::*kernel_member; };
    const bool prefer_explicit_256 = is_gfx906_like(gpu) && gpu.log_n >= 20u;
    const Candidate candidates[] = {
        {4096u, 64u, &GpuPrp::k_center_fused_4096},
        {2048u, 64u, &GpuPrp::k_center_fused_2048},
        {1024u, 64u, &GpuPrp::k_center_fused_1024},
        {512u, 64u, &GpuPrp::k_center_fused_512},
        {256u, 64u, prefer_explicit_256 ? &GpuPrp::k_center_fused_256_explicit : &GpuPrp::k_center_fused_256},
        {128u, 32u, &GpuPrp::k_center_fused_128},
        {64u, 16u, &GpuPrp::k_center_fused_64},
        {32u, 16u, &GpuPrp::k_center_fused_32},
        {16u, 8u, &GpuPrp::k_center_fused_16},
        {8u, 8u, &GpuPrp::k_center_fused_8},
    };
    for (const auto& cand : candidates) {
        if (cand.chunk > cap) continue;
        const std::size_t local_bytes = std::size_t(cand.chunk) * gpu.field_elem_size;
        if (gpu.n < std::size_t(cand.chunk) * 2u) continue;
        if ((gpu.n % cand.chunk) != 0u) continue;
        if (cand.wg > gpu.max_work_group_size) continue;
        if (local_bytes > static_cast<std::size_t>(gpu.local_mem_size)) continue;
        cl_kernel k = gpu.*(cand.kernel_member);
        if (!k) continue;
        return CenterKernelConfig{k, cand.chunk, cand.wg, true};
    }
    return {};
}

static BridgeKernelConfig choose_bridge_kernel(GpuPrp& gpu, const CenterKernelConfig& center) {
    if (is_gfx906_like(gpu) && gpu.log_n >= 20u) return {};
    struct Candidate {
        cl_uint outer_chunk;
        cl_uint inner_chunk;
        cl_uint wg;
        cl_kernel GpuPrp::*forward_member;
        cl_kernel GpuPrp::*inverse_member;
    };
    const Candidate candidates[] = {
        {2048u, 256u, 64u, &GpuPrp::k_forward_bridge_2048_to_256, &GpuPrp::k_inverse_bridge_256_to_2048},
        {1024u, 512u, 64u, &GpuPrp::k_forward_bridge_1024_to_512, &GpuPrp::k_inverse_bridge_512_to_1024},
        {1024u, 256u, 64u, &GpuPrp::k_forward_bridge_1024_to_256, &GpuPrp::k_inverse_bridge_256_to_1024},
        {512u, 256u, 64u, &GpuPrp::k_forward_bridge_512_to_256, &GpuPrp::k_inverse_bridge_256_to_512},
        {256u, 64u, 64u, &GpuPrp::k_forward_bridge_256_to_64, &GpuPrp::k_inverse_bridge_64_to_256},
        {64u, 16u, 16u, &GpuPrp::k_forward_bridge_64_to_16, &GpuPrp::k_inverse_bridge_16_to_64},
    };
    for (const auto& cand : candidates) {
        if (center.chunk != cand.inner_chunk) continue;
        if (gpu.n < std::size_t(cand.outer_chunk) * 2u) continue;
        if ((gpu.n % cand.outer_chunk) != 0u) continue;
        const std::size_t local_bytes = std::size_t(cand.outer_chunk) * gpu.field_elem_size;
        if (cand.wg > gpu.max_work_group_size) continue;
        if (local_bytes > static_cast<std::size_t>(gpu.local_mem_size)) continue;
        cl_kernel fk = gpu.*(cand.forward_member);
        cl_kernel ik = gpu.*(cand.inverse_member);
        if (!fk || !ik) continue;
        return BridgeKernelConfig{fk, ik, cand.outer_chunk, cand.inner_chunk, cand.wg, true};
    }
    return {};
}

static BridgeKernelConfig choose_local_block_lds_kernel(GpuPrp& gpu) {
    if (g_force_strict_reference || g_local_block_lds_disabled) return {};
    struct Candidate {
        cl_uint outer_chunk;
        cl_uint inner_chunk;
        cl_uint wg;
        cl_kernel GpuPrp::*forward_member;
        cl_kernel GpuPrp::*inverse_member;
    };
    const Candidate candidates[] = {
        {2048u, 256u, 64u, &GpuPrp::k_forward_bridge_2048_to_256, &GpuPrp::k_inverse_bridge_256_to_2048},
        {1024u, 256u, 64u, &GpuPrp::k_forward_bridge_1024_to_256, &GpuPrp::k_inverse_bridge_256_to_1024},
        {512u, 256u, 64u, &GpuPrp::k_forward_bridge_512_to_256, &GpuPrp::k_inverse_bridge_256_to_512},
    };
    for (const auto& cand : candidates) {
        if (g_local_block_lds_override && cand.outer_chunk != g_local_block_lds_override) continue;
        
        
        if (gpu.n < std::size_t(cand.outer_chunk) * 2u) continue;
        if ((gpu.n % cand.outer_chunk) != 0u) continue;
        const std::size_t local_bytes = std::size_t(cand.outer_chunk) * gpu.field_elem_size;
        if (cand.wg > gpu.max_work_group_size) continue;
        if (local_bytes > static_cast<std::size_t>(gpu.local_mem_size)) continue;
        cl_kernel fk = gpu.*(cand.forward_member);
        cl_kernel ik = gpu.*(cand.inverse_member);
        if (!fk || !ik) continue;
        return BridgeKernelConfig{fk, ik, cand.outer_chunk, cand.inner_chunk, cand.wg, true};
    }
    return {};
}


static std::string describe_local_block_lds_choice(GpuPrp& gpu) {
    if (g_local_block_lds_disabled) return "off";
    const BridgeKernelConfig lds = g_force_strict_reference ? BridgeKernelConfig{} : choose_local_block_lds_kernel(gpu);
    if (!lds.enabled) {
        if (g_local_block_lds_override) return std::to_string(g_local_block_lds_override) + "(unavailable)";
        return "auto->off";
    }
    if (g_local_block_lds_override) return std::to_string(lds.outer_chunk);
    return std::string("auto->") + std::to_string(lds.outer_chunk);
}

static std::pair<cl_kernel, size_t> choose_weight_first_kernel(GpuPrp& gpu) {
    const size_t global = gpu.n / 2;
    if (global >= 64u && gpu.k_weight_first_stage_dif_wg64 && gpu.max_work_group_size >= 64u) return {gpu.k_weight_first_stage_dif_wg64, 64u};
    if (global >= 16u && gpu.k_weight_first_stage_dif_wg16 && gpu.max_work_group_size >= 16u) return {gpu.k_weight_first_stage_dif_wg16, 16u};
    return {gpu.k_weight_first_stage_dif, 0u};
}

static std::pair<cl_kernel, size_t> choose_last_stage_kernel(GpuPrp& gpu) {
    const size_t global = gpu.n / 2;
    if (global >= 64u && gpu.k_last_stage_dit_unweight_wg64 && gpu.max_work_group_size >= 64u) return {gpu.k_last_stage_dit_unweight_wg64, 64u};
    if (global >= 16u && gpu.k_last_stage_dit_unweight_wg16 && gpu.max_work_group_size >= 16u) return {gpu.k_last_stage_dit_unweight_wg16, 16u};
    return {gpu.k_last_stage_dit_unweight, 0u};
}

static std::pair<cl_kernel, size_t> choose_weight_first_radix4_kernel(GpuPrp& gpu) {
    const size_t global = gpu.n / 8;
    if (global >= 128u && gpu.k_weight_first_stage_dif_radix4_wg128 && gpu.max_work_group_size >= 128u) return {gpu.k_weight_first_stage_dif_radix4_wg128, 128u};
    if (global >= 64u && gpu.k_weight_first_stage_dif_radix4_wg64 && gpu.max_work_group_size >= 64u) return {gpu.k_weight_first_stage_dif_radix4_wg64, 64u};
    return {nullptr, 0u};
}

static std::pair<cl_kernel, size_t> choose_last_stage_radix4_unweight_kernel(GpuPrp& gpu) {
    const size_t global = gpu.n / 8;
    if (global >= 128u && gpu.k_last_stage_dit_radix4_unweight_wg128 && gpu.max_work_group_size >= 128u) return {gpu.k_last_stage_dit_radix4_unweight_wg128, 128u};
    if (global >= 64u && gpu.k_last_stage_dit_radix4_unweight_wg64 && gpu.max_work_group_size >= 64u) return {gpu.k_last_stage_dit_radix4_unweight_wg64, 64u};
    return {nullptr, 0u};
}


static bool can_use_true_ext1024_path(GpuPrp& gpu, cl_uint center_max) {
    (void)center_max;
    if (!gpu.k_forward_ext_1024_to_256_explicit && !gpu.k_forward_ext_1024_to_256_explicit_wg128) return false;
    if (!gpu.k_inverse_ext_256_to_1024_explicit && !gpu.k_inverse_ext_256_to_1024_explicit_wg128) return false;
    if (!gpu.k_center_fused_256_explicit) return false;
    if (gpu.n < 1024u) return false;
    if ((gpu.n % 1024u) != 0u) return false;
    return true;
}

static void enqueue_true_ext1024_forward(GpuPrp& gpu) {
    cl_kernel k = gpu.k_forward_ext_1024_to_256_explicit;
    size_t local = 64u;
    if (gpu.k_forward_ext_1024_to_256_explicit_wg128 && gpu.max_work_group_size >= 128u) {
        k = gpu.k_forward_ext_1024_to_256_explicit_wg128;
        local = 128u;
    }
    check(clSetKernelArg(k, 0, sizeof(cl_mem), &gpu.bufField), "set ext1024_fwd a");
    check(clSetKernelArg(k, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ext1024_fwd tw");
    const size_t global = (gpu.n / 1024u) * local;
    enqueue_kernel(gpu, k, global, &local, "enqueue true_ext1024_forward", "ntt_external_forward");
}

static void enqueue_true_ext1024_inverse(GpuPrp& gpu) {
    cl_kernel k = gpu.k_inverse_ext_256_to_1024_explicit;
    size_t local = 64u;
    if (gpu.k_inverse_ext_256_to_1024_explicit_wg128 && gpu.max_work_group_size >= 128u) {
        k = gpu.k_inverse_ext_256_to_1024_explicit_wg128;
        local = 128u;
    }
    check(clSetKernelArg(k, 0, sizeof(cl_mem), &gpu.bufField), "set ext1024_inv a");
    check(clSetKernelArg(k, 1, sizeof(cl_mem), &gpu.bufTwInv), "set ext1024_inv tw");
    const size_t global = (gpu.n / 1024u) * local;
    enqueue_kernel(gpu, k, global, &local, "enqueue true_ext1024_inverse", "ntt_external_inverse");
}

static bool can_use_true_ext2048_path(GpuPrp& gpu, cl_uint center_max) {
    (void)center_max;
    if (!gpu.k_forward_ext_2048_to_256_explicit) return false;
    if (!gpu.k_inverse_ext_256_to_2048_explicit) return false;
    if (!gpu.k_center_fused_256_explicit) return false;
    if (gpu.n < 2048u) return false;
    if ((gpu.n % 2048u) != 0u) return false;
    return true;
}

static void enqueue_true_ext2048_forward(GpuPrp& gpu) {
    check(clSetKernelArg(gpu.k_forward_ext_2048_to_256_explicit, 0, sizeof(cl_mem), &gpu.bufField), "set ext2048_fwd a");
    check(clSetKernelArg(gpu.k_forward_ext_2048_to_256_explicit, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ext2048_fwd tw");
    const size_t global = (gpu.n / 2048u) * 64u;
    const size_t local = 64u;
    enqueue_kernel(gpu, gpu.k_forward_ext_2048_to_256_explicit, global, &local, "enqueue true_ext2048_forward", "ntt_external_forward");
}

static void enqueue_true_ext2048_inverse(GpuPrp& gpu) {
    check(clSetKernelArg(gpu.k_inverse_ext_256_to_2048_explicit, 0, sizeof(cl_mem), &gpu.bufField), "set ext2048_inv a");
    check(clSetKernelArg(gpu.k_inverse_ext_256_to_2048_explicit, 1, sizeof(cl_mem), &gpu.bufTwInv), "set ext2048_inv tw");
    const size_t global = (gpu.n / 2048u) * 64u;
    const size_t local = 64u;
    enqueue_kernel(gpu, gpu.k_inverse_ext_256_to_2048_explicit, global, &local, "enqueue true_ext2048_inverse", "ntt_external_inverse");
}

static void enqueue_bridge_forward(GpuPrp& gpu, const BridgeKernelConfig& cfg) {
    if (!cfg.enabled) return;
    check(clSetKernelArg(cfg.forward_kernel, 0, sizeof(cl_mem), &gpu.bufField), "set bridge_fwd a");
    check(clSetKernelArg(cfg.forward_kernel, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set bridge_fwd tw");
    const size_t global = (gpu.n / cfg.outer_chunk) * cfg.local_size;
    const size_t local = cfg.local_size;
    check(clEnqueueNDRangeKernel(gpu.queue, cfg.forward_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "enqueue bridge_forward");
}

static void enqueue_bridge_inverse(GpuPrp& gpu, const BridgeKernelConfig& cfg) {
    if (!cfg.enabled) return;
    check(clSetKernelArg(cfg.inverse_kernel, 0, sizeof(cl_mem), &gpu.bufField), "set bridge_inv a");
    check(clSetKernelArg(cfg.inverse_kernel, 1, sizeof(cl_mem), &gpu.bufTwInv), "set bridge_inv tw");
    const size_t global = (gpu.n / cfg.outer_chunk) * cfg.local_size;
    const size_t local = cfg.local_size;
    check(clEnqueueNDRangeKernel(gpu.queue, cfg.inverse_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "enqueue bridge_inverse");
}

static void enqueue_center_fused(GpuPrp& gpu, const CenterKernelConfig& cfg, const char* profile_label = "center_fused") {
    if (!cfg.enabled) return;
    check(clSetKernelArg(cfg.kernel, 0, sizeof(cl_mem), &gpu.bufField), "set center_fused a");
    check(clSetKernelArg(cfg.kernel, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set center_fused tw_fwd");
    check(clSetKernelArg(cfg.kernel, 2, sizeof(cl_mem), &gpu.bufTwInv), "set center_fused tw_inv");
    const size_t global = (gpu.n / cfg.chunk) * cfg.local_size;
    const size_t local = cfg.local_size;
    enqueue_kernel(gpu, cfg.kernel, global, &local, "enqueue center_fused", profile_label);
}

static void enqueue_forward_pipeline_partial(GpuPrp& gpu, cl_uint stop_chunk);

static void enqueue_forward_pipeline(GpuPrp& gpu) {
    enqueue_forward_pipeline_partial(gpu, 0u);
}

static void enqueue_forward_pipeline_partial(GpuPrp& gpu, cl_uint stop_chunk) {
    const StageInfo& first = gpu.stages.back();

    std::vector<StageInfo> todo;
    for (auto it = gpu.stages.rbegin() + 1; it != gpu.stages.rend(); ++it) {
        const StageInfo& st = *it;
        if (stop_chunk != 0u && st.len <= stop_chunk) break;
        todo.push_back(st);
    }

    std::size_t idx = 0;
    const bool can_fuse_weight_radix4 =
        todo.size() >= 2 &&
        todo[0].len * 2u == first.len &&
        todo[1].len * 2u == todo[0].len &&
        gpu.n >= 512u;

    if (!g_force_strict_reference && can_fuse_weight_radix4) {
        const auto fused = choose_weight_first_radix4_kernel(gpu);
        if (fused.first) {
            cl_mem first_digits = gpu.crtInputDigits ? gpu.crtInputDigits : gpu.bufDigits;
            check(clSetKernelArg(fused.first, 0, sizeof(cl_mem), &first_digits), "set weight_radix4 digits");
            check(clSetKernelArg(fused.first, 1, sizeof(cl_uint), &gpu.exponent_p), "set weight_radix4 p");
            check(clSetKernelArg(fused.first, 2, sizeof(cl_uint), &gpu.lr2), "set weight_radix4 lr2");
            check(clSetKernelArg(fused.first, 3, sizeof(cl_mem), &gpu.bufField), "set weight_radix4 field");
            check(clSetKernelArg(fused.first, 4, sizeof(cl_mem), &gpu.bufTwFwd), "set weight_radix4 tw0");
            check(clSetKernelArg(fused.first, 5, sizeof(cl_mem), &gpu.bufTwFwd), "set weight_radix4 tw1");
            check(clSetKernelArg(fused.first, 6, sizeof(cl_mem), &gpu.bufTwFwd), "set weight_radix4 tw2");
            check(clSetKernelArg(fused.first, 7, sizeof(cl_uint), &first.offset), "set weight_radix4 off0");
            check(clSetKernelArg(fused.first, 8, sizeof(cl_uint), &todo[0].offset), "set weight_radix4 off1");
            check(clSetKernelArg(fused.first, 9, sizeof(cl_uint), &todo[1].offset), "set weight_radix4 off2");
            check(clSetKernelArg(fused.first, 10, sizeof(cl_uint), &first.len), "set weight_radix4 len");
            const size_t global = gpu.n / 8;
            const size_t local = fused.second;
            enqueue_kernel(gpu, fused.first, global, &local, "enqueue weight_first_stage_dif_radix4", "weight_first_radix4");
            idx = 2;
        }
    }

    if (idx == 0) {
        const auto weight_kernel = choose_weight_first_kernel(gpu);
        check(clSetKernelArg(weight_kernel.first, 0, sizeof(cl_mem), &gpu.bufDigits), "set weight_first digits");
        check(clSetKernelArg(weight_kernel.first, 1, sizeof(cl_uint), &gpu.exponent_p), "set weight_first p");
        check(clSetKernelArg(weight_kernel.first, 2, sizeof(cl_uint), &gpu.lr2), "set weight_first lr2");
        check(clSetKernelArg(weight_kernel.first, 3, sizeof(cl_mem), &gpu.bufField), "set weight_first field");
        check(clSetKernelArg(weight_kernel.first, 4, sizeof(cl_mem), &gpu.bufTwFwd), "set weight_first tw");
        check(clSetKernelArg(weight_kernel.first, 5, sizeof(cl_uint), &first.offset), "set weight_first off");
        check(clSetKernelArg(weight_kernel.first, 6, sizeof(cl_uint), &first.len), "set weight_first len");
        check(clSetKernelArg(weight_kernel.first, 7, sizeof(cl_uint), &first.half_len), "set weight_first half");
        const size_t global = gpu.n / 2;
        const size_t* local_ptr = weight_kernel.second ? &weight_kernel.second : nullptr;
        enqueue_kernel(gpu, weight_kernel.first, global, local_ptr, "enqueue weight_first_stage_dif", "weight_first_stage_dif");
    }

    for (; idx < todo.size();) {
        const StageInfo& st = todo[idx];
        const bool can_radix4x2 =
            !g_force_strict_reference &&
            gpu.prefer_radix4x2 &&
            (idx + 3 < todo.size()) &&
            (todo[idx + 1].len * 2u == st.len) &&
            (todo[idx + 2].len * 4u == st.len) &&
            (todo[idx + 3].len * 8u == st.len) &&
            st.len >= 16u &&
            (gpu.k_ntt_stage_dif_radix4x2 || gpu.k_ntt_stage_dif_radix4x2_wg128) &&
            gpu.n >= 1024u;
        if (can_radix4x2) {
            cl_kernel k = gpu.k_ntt_stage_dif_radix4x2;
            size_t local = 64u;
            if (gpu.k_ntt_stage_dif_radix4x2_wg128 && gpu.max_work_group_size >= 128u) {
                k = gpu.k_ntt_stage_dif_radix4x2_wg128;
                local = 128u;
            }
            const StageInfo& st2 = todo[idx + 1];
            const StageInfo& st3 = todo[idx + 2];
            const StageInfo& st4 = todo[idx + 3];
            check(clSetKernelArg(k, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif_radix4x2 a");
            check(clSetKernelArg(k, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4x2 tw1");
            check(clSetKernelArg(k, 2, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4x2 tw2");
            check(clSetKernelArg(k, 3, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4x2 tw3");
            check(clSetKernelArg(k, 4, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4x2 tw4");
            check(clSetKernelArg(k, 5, sizeof(cl_uint), &st.offset), "set ntt_stage_dif_radix4x2 off1");
            check(clSetKernelArg(k, 6, sizeof(cl_uint), &st2.offset), "set ntt_stage_dif_radix4x2 off2");
            check(clSetKernelArg(k, 7, sizeof(cl_uint), &st3.offset), "set ntt_stage_dif_radix4x2 off3");
            check(clSetKernelArg(k, 8, sizeof(cl_uint), &st4.offset), "set ntt_stage_dif_radix4x2 off4");
            check(clSetKernelArg(k, 9, sizeof(cl_uint), &st.len), "set ntt_stage_dif_radix4x2 len");
            const size_t global = gpu.n / 16;
            local = clamp_local_to_global(local, global);
            enqueue_kernel(gpu, k, global, &local, "enqueue ntt_stage_dif_radix4x2 fwd", "ntt_radix4x2_forward");
            idx += 4;
            continue;
        }
        const bool can_pair = !g_force_strict_reference && (idx + 1 < todo.size()) && (todo[idx + 1].len * 2u == st.len) && st.len >= 4u && (gpu.k_ntt_stage_dif_radix4 || gpu.k_ntt_stage_dif_radix4_wg128) && gpu.n >= 256u;
        if (can_pair) {
            cl_kernel k = gpu.k_ntt_stage_dif_radix4;
            size_t local = 64u;
            if (gpu.k_ntt_stage_dif_radix4_wg128 && gpu.max_work_group_size >= 128u) {
                k = gpu.k_ntt_stage_dif_radix4_wg128;
                local = 128u;
            }
            const StageInfo& st2 = todo[idx + 1];
            check(clSetKernelArg(k, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif_radix4 a");
            check(clSetKernelArg(k, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4 tw1");
            check(clSetKernelArg(k, 2, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_radix4 tw2");
            check(clSetKernelArg(k, 3, sizeof(cl_uint), &st.offset), "set ntt_stage_dif_radix4 off1");
            check(clSetKernelArg(k, 4, sizeof(cl_uint), &st2.offset), "set ntt_stage_dif_radix4 off2");
            check(clSetKernelArg(k, 5, sizeof(cl_uint), &st.len), "set ntt_stage_dif_radix4 len");
            const size_t global = gpu.n / 4;
            local = clamp_local_to_global(local, global);
            enqueue_kernel(gpu, k, global, &local, "enqueue ntt_stage_dif_radix4 fwd", "ntt_radix4_forward");
            idx += 2;
            continue;
        }
        const bool use_dedicated_2048 = !g_force_strict_reference && (st.len == 2048u) && gpu.k_ntt_stage_dif_len2048 && gpu.max_work_group_size >= 256u && gpu.n >= 2048u && ((gpu.n % 2048u) == 0u);
        if (use_dedicated_2048) {
            check(clSetKernelArg(gpu.k_ntt_stage_dif_len2048, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif_len2048 a");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_len2048, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif_len2048 tw");
            check(clSetKernelArg(gpu.k_ntt_stage_dif_len2048, 2, sizeof(cl_uint), &st.offset), "set ntt_stage_dif_len2048 off");
            const size_t global = (gpu.n / 2048u) * 256u;
            const size_t local = 256u;
            enqueue_kernel(gpu, gpu.k_ntt_stage_dif_len2048, global, &local, "enqueue ntt_stage_dif_len2048 fwd", "ntt_stage_dif_2048");
            idx += 1;
            continue;
        }
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 0, sizeof(cl_mem), &gpu.bufField), "set ntt_stage_dif a");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 1, sizeof(cl_mem), &gpu.bufTwFwd), "set ntt_stage_dif tw");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 2, sizeof(cl_uint), &st.offset), "set ntt_stage_dif off");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 3, sizeof(cl_uint), &st.len), "set ntt_stage_dif len");
        check(clSetKernelArg(gpu.k_ntt_stage_dif, 4, sizeof(cl_uint), &st.half_len), "set ntt_stage_dif half");
        const size_t global = gpu.n / 2;
        enqueue_kernel(gpu, gpu.k_ntt_stage_dif, global, nullptr, "enqueue ntt_stage_dif fwd", "ntt_stage_dif");
        idx += 1;
    }
}

static void enqueue_inverse_pipeline_partial(GpuPrp& gpu, cl_uint skip_chunk);

static void enqueue_inverse_pipeline(GpuPrp& gpu) {
    enqueue_inverse_pipeline_partial(gpu, 0u);
}

static void enqueue_inverse_pipeline_partial(GpuPrp& gpu, cl_uint skip_chunk) {
    const StageInfo& last = gpu.stages.back();
    std::vector<StageInfo> todo;
    for (std::size_t idx = 0; idx + 1 < gpu.stages.size(); ++idx) {
        const StageInfo& st = gpu.stages[idx];
        if (skip_chunk != 0u && st.len <= skip_chunk) continue;
        todo.push_back(st);
    }

    const bool can_fuse_last_radix4 =
        todo.size() >= 2 &&
        todo[todo.size() - 2].len * 2u == todo.back().len &&
        todo.back().len == last.half_len &&
        gpu.n >= 512u;

    std::size_t todo_limit = todo.size();
    std::pair<cl_kernel, size_t> fused_last{nullptr, 0u};
    if (!g_force_strict_reference && can_fuse_last_radix4) {
        fused_last = choose_last_stage_radix4_unweight_kernel(gpu);
        if (fused_last.first) todo_limit = todo.size() - 2;
    }

    for (std::size_t idx = 0; idx < todo_limit;) {
        const StageInfo& st = todo[idx];
        const bool can_radix4x2 =
            !g_force_strict_reference &&
            gpu.prefer_radix4x2 &&
            (idx + 3 < todo_limit) &&
            (todo[idx + 1].len == st.len * 2u) &&
            (todo[idx + 2].len == st.len * 4u) &&
            (todo[idx + 3].len == st.len * 8u) &&
            st.len >= 2u &&
            (gpu.k_ntt_stage_dit_radix4x2 || gpu.k_ntt_stage_dit_radix4x2_wg128) &&
            gpu.n >= 1024u;
        if (can_radix4x2) {
            cl_kernel k = gpu.k_ntt_stage_dit_radix4x2;
            size_t local = 64u;
            if (gpu.k_ntt_stage_dit_radix4x2_wg128 && gpu.max_work_group_size >= 128u) {
                k = gpu.k_ntt_stage_dit_radix4x2_wg128;
                local = 128u;
            }
            const StageInfo& st2 = todo[idx + 1];
            const StageInfo& st3 = todo[idx + 2];
            const StageInfo& st4 = todo[idx + 3];
            check(clSetKernelArg(k, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit_radix4x2 a");
            check(clSetKernelArg(k, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4x2 tw1");
            check(clSetKernelArg(k, 2, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4x2 tw2");
            check(clSetKernelArg(k, 3, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4x2 tw3");
            check(clSetKernelArg(k, 4, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4x2 tw4");
            check(clSetKernelArg(k, 5, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit_radix4x2 off1");
            check(clSetKernelArg(k, 6, sizeof(cl_uint), &st2.offset), "set inv_ntt_stage_dit_radix4x2 off2");
            check(clSetKernelArg(k, 7, sizeof(cl_uint), &st3.offset), "set inv_ntt_stage_dit_radix4x2 off3");
            check(clSetKernelArg(k, 8, sizeof(cl_uint), &st4.offset), "set inv_ntt_stage_dit_radix4x2 off4");
            check(clSetKernelArg(k, 9, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit_radix4x2 len");
            const size_t global = gpu.n / 16;
            local = clamp_local_to_global(local, global);
            enqueue_kernel(gpu, k, global, &local, "enqueue ntt_stage_dit_radix4x2 inv", "ntt_radix4x2_inverse");
            idx += 4;
            continue;
        }
        const bool can_pair = !g_force_strict_reference && (idx + 1 < todo_limit) && (todo[idx + 1].len == st.len * 2u) && st.len >= 2u && (gpu.k_ntt_stage_dit_radix4 || gpu.k_ntt_stage_dit_radix4_wg128) && gpu.n >= 256u;
        if (can_pair) {
            cl_kernel k = gpu.k_ntt_stage_dit_radix4;
            size_t local = 64u;
            if (gpu.k_ntt_stage_dit_radix4_wg128 && gpu.max_work_group_size >= 128u) {
                k = gpu.k_ntt_stage_dit_radix4_wg128;
                local = 128u;
            }
            const StageInfo& st2 = todo[idx + 1];
            check(clSetKernelArg(k, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit_radix4 a");
            check(clSetKernelArg(k, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4 tw1");
            check(clSetKernelArg(k, 2, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit_radix4 tw2");
            check(clSetKernelArg(k, 3, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit_radix4 off1");
            check(clSetKernelArg(k, 4, sizeof(cl_uint), &st2.offset), "set inv_ntt_stage_dit_radix4 off2");
            check(clSetKernelArg(k, 5, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit_radix4 len");
            const size_t global = gpu.n / 4;
            local = clamp_local_to_global(local, global);
            enqueue_kernel(gpu, k, global, &local, "enqueue ntt_stage_dit_radix4 inv", "ntt_radix4_inverse");
            idx += 2;
            continue;
        }
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 0, sizeof(cl_mem), &gpu.bufField), "set inv_ntt_stage_dit a");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 1, sizeof(cl_mem), &gpu.bufTwInv), "set inv_ntt_stage_dit tw");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 2, sizeof(cl_uint), &st.offset), "set inv_ntt_stage_dit off");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 3, sizeof(cl_uint), &st.len), "set inv_ntt_stage_dit len");
        check(clSetKernelArg(gpu.k_ntt_stage_dit, 4, sizeof(cl_uint), &st.half_len), "set inv_ntt_stage_dit half");
        const size_t global = gpu.n / 2;
        enqueue_kernel(gpu, gpu.k_ntt_stage_dit, global, nullptr, "enqueue ntt_stage_dit inv", "ntt_stage_dit");
        idx += 1;
    }

    const cl_uint log_n = static_cast<cl_uint>(gpu.stages.size());
    if (fused_last.first) {
        const StageInfo& st1 = todo[todo.size() - 2];
        const StageInfo& st2 = todo.back();
        check(clSetKernelArg(fused_last.first, 0, sizeof(cl_mem), &gpu.bufField), "set last_radix4_unweight a");
        check(clSetKernelArg(fused_last.first, 1, sizeof(cl_mem), &gpu.bufTwInv), "set last_radix4_unweight tw1");
        check(clSetKernelArg(fused_last.first, 2, sizeof(cl_mem), &gpu.bufTwInv), "set last_radix4_unweight tw2");
        check(clSetKernelArg(fused_last.first, 3, sizeof(cl_mem), &gpu.bufTwInv), "set last_radix4_unweight tw3");
        check(clSetKernelArg(fused_last.first, 4, sizeof(cl_uint), &gpu.exponent_p), "set last_radix4_unweight p");
        check(clSetKernelArg(fused_last.first, 5, sizeof(cl_uint), &gpu.lr2), "set last_radix4_unweight lr2");
        check(clSetKernelArg(fused_last.first, 6, sizeof(cl_mem), &gpu.bufDigits), "set last_radix4_unweight digits");
        check(clSetKernelArg(fused_last.first, 7, sizeof(cl_uint), &log_n), "set last_radix4_unweight logn");
        check(clSetKernelArg(fused_last.first, 8, sizeof(cl_uint), &st1.offset), "set last_radix4_unweight off1");
        check(clSetKernelArg(fused_last.first, 9, sizeof(cl_uint), &st2.offset), "set last_radix4_unweight off2");
        check(clSetKernelArg(fused_last.first, 10, sizeof(cl_uint), &last.offset), "set last_radix4_unweight off3");
        check(clSetKernelArg(fused_last.first, 11, sizeof(cl_uint), &st1.len), "set last_radix4_unweight len");
        const size_t global = gpu.n / 8;
        const size_t local = fused_last.second;
        enqueue_kernel(gpu, fused_last.first, global, &local, "enqueue last_stage_dit_radix4_unweight", "last_stage_radix4_unweight");
        return;
    }

    const auto last_kernel = choose_last_stage_kernel(gpu);
    check(clSetKernelArg(last_kernel.first, 0, sizeof(cl_mem), &gpu.bufField), "set last_stage a");
    check(clSetKernelArg(last_kernel.first, 1, sizeof(cl_mem), &gpu.bufTwInv), "set last_stage tw");
    check(clSetKernelArg(last_kernel.first, 2, sizeof(cl_uint), &gpu.exponent_p), "set last_stage p");
    check(clSetKernelArg(last_kernel.first, 3, sizeof(cl_uint), &gpu.lr2), "set last_stage lr2");
    check(clSetKernelArg(last_kernel.first, 4, sizeof(cl_mem), &gpu.bufDigits), "set last_stage digits");
    check(clSetKernelArg(last_kernel.first, 5, sizeof(cl_uint), &log_n), "set last_stage logn");
    check(clSetKernelArg(last_kernel.first, 6, sizeof(cl_uint), &last.offset), "set last_stage off");
    check(clSetKernelArg(last_kernel.first, 7, sizeof(cl_uint), &last.len), "set last_stage len");
    check(clSetKernelArg(last_kernel.first, 8, sizeof(cl_uint), &last.half_len), "set last_stage half");
    const size_t global = gpu.n / 2;
    const size_t* local_ptr = last_kernel.second ? &last_kernel.second : nullptr;
    enqueue_kernel(gpu, last_kernel.first, global, local_ptr, "enqueue last_stage_dit_unweight", "last_stage_dit_unweight");
}

static bool enqueue_square_mod_single_halfreal(GpuPrp& gpu);

static void enqueue_square_mod(GpuPrp& gpu, cl_uint center_max = 0) {
    const cl_uint plan_log_n = gpu.log_n ? gpu.log_n : static_cast<cl_uint>(gpu.stages.size());
    const bool cond_log_n_1024 = plan_log_n >= 20u;
    const bool cond_n_1024 = gpu.n >= 1024u;
    const bool cond_mod_1024 = ((gpu.n % 1024u) == 0u);
    const bool cond_k_fwd_1024 = (gpu.k_forward_ext_1024_to_256_explicit != nullptr);
    const bool cond_k_inv_1024 = (gpu.k_inverse_ext_256_to_1024_explicit != nullptr);
    const bool cond_k_center_256 = (gpu.k_center_fused_256_explicit != nullptr);
    const bool cond_true_1024 = (!g_force_strict_reference && !g_local_block_lds_disabled && g_local_block_lds_override == 0u) && cond_log_n_1024 && cond_n_1024 && cond_mod_1024 && cond_k_fwd_1024 && cond_k_inv_1024 && cond_k_center_256;
    const bool cond_true_2048 = (!g_force_strict_reference && !g_local_block_lds_disabled && g_local_block_lds_override == 0u) && (can_use_true_ext2048_path(gpu, center_max) && plan_log_n >= 20u);
    if (g_planner_debug) {
        std::cerr
            << "[planner-debug] log_n=" << plan_log_n
            << " n=" << gpu.n
            << " center_max=" << center_max
            << " cond_log_n_1024=" << (cond_log_n_1024 ? 1 : 0)
            << " cond_n_1024=" << (cond_n_1024 ? 1 : 0)
            << " cond_mod_1024=" << (cond_mod_1024 ? 1 : 0)
            << " cond_k_fwd_1024=" << (cond_k_fwd_1024 ? 1 : 0)
            << " cond_k_inv_1024=" << (cond_k_inv_1024 ? 1 : 0)
            << " cond_k_center_256=" << (cond_k_center_256 ? 1 : 0)
            << " cond_true_1024=" << (cond_true_1024 ? 1 : 0)
            << " cond_true_2048=" << (cond_true_2048 ? 1 : 0)
            << " local_block_override=" << g_local_block_lds_override
            << " local_block_disabled=" << (g_local_block_lds_disabled ? 1 : 0)
            << "\n";
    }

    if (g_single_center_mode == "halfreal") {
        enqueue_square_mod_single_halfreal(gpu);
        g_last_local_block_lds = 0;
        return;
    }

    g_last_local_block_lds = 0;
    {
        const BridgeKernelConfig lds = g_force_strict_reference ? BridgeKernelConfig{} : choose_local_block_lds_kernel(gpu);
        cl_kernel center256 = gpu.k_center_fused_256;
        if (gpu.k_center_fused_256_explicit) center256 = gpu.k_center_fused_256_explicit;
        const bool can_center256 = center256 && gpu.max_work_group_size >= 64u
            && gpu.local_mem_size >= static_cast<cl_ulong>(256u * gpu.field_elem_size)
            && ((gpu.n % 256u) == 0u);
        if (lds.enabled && can_center256) {
            if (g_planner_debug) {
                std::cerr << "[planner] LOCAL_BLOCK_LDS path outer=" << lds.outer_chunk
                          << " inner=" << lds.inner_chunk << " wg=" << lds.local_size << "\n";
            }
            const CenterKernelConfig center{center256, 256u, 64u, true};
            enqueue_forward_pipeline_partial(gpu, lds.outer_chunk);
            enqueue_bridge_forward(gpu, lds);
            enqueue_center_fused(gpu, center);
            enqueue_bridge_inverse(gpu, lds);
            enqueue_inverse_pipeline_partial(gpu, lds.outer_chunk);
            g_last_local_block_lds = lds.outer_chunk;
            return;
        }
    }

    if (cond_true_1024) {
        if (g_planner_debug) std::cerr << "[planner] TRUE_EXT1024 path\n";
        const CenterKernelConfig center{gpu.k_center_fused_256_explicit, 256u, 64u, true};
        enqueue_forward_pipeline_partial(gpu, 1024u);
        enqueue_true_ext1024_forward(gpu);
        enqueue_center_fused(gpu, center);
        enqueue_true_ext1024_inverse(gpu);
        enqueue_inverse_pipeline_partial(gpu, 1024u);
        return;
    }
    if (cond_true_2048) {
        if (g_planner_debug) std::cerr << "[planner] TRUE_EXT2048 path\n";
        const CenterKernelConfig center{gpu.k_center_fused_256_explicit, 256u, 64u, true};
        enqueue_forward_pipeline_partial(gpu, 2048u);
        enqueue_true_ext2048_forward(gpu);
        enqueue_center_fused(gpu, center);
        enqueue_true_ext2048_inverse(gpu);
        enqueue_inverse_pipeline_partial(gpu, 2048u);
        return;
    }
    if (g_planner_debug) std::cerr << "[planner] fallback path\n";
    const CenterKernelConfig center = choose_center_kernel(gpu, center_max);
    if (center.enabled) {
        const BridgeKernelConfig bridge = choose_bridge_kernel(gpu, center);
        if (g_planner_debug) {
            std::cerr
                << "[planner-debug-fallback] center_enabled=" << (center.enabled ? 1 : 0)
                << " center_chunk=" << center.chunk
                << " center_local=" << center.local_size
                << " bridge_enabled=" << (bridge.enabled ? 1 : 0)
                << " bridge_outer_chunk=" << bridge.outer_chunk
                << " bridge_inner_chunk=" << bridge.inner_chunk
                << " bridge_local=" << bridge.local_size
                << "\n";
        }
        const cl_uint outer_stop = bridge.enabled ? bridge.outer_chunk : center.chunk;
        enqueue_forward_pipeline_partial(gpu, outer_stop);
        if (bridge.enabled) enqueue_bridge_forward(gpu, bridge);
        enqueue_center_fused(gpu, center);
        if (bridge.enabled) enqueue_bridge_inverse(gpu, bridge);
        enqueue_inverse_pipeline_partial(gpu, outer_stop);
        return;
    }
    enqueue_forward_pipeline(gpu);
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_pointwise_sqr, 0, sizeof(cl_mem), &gpu.bufField), "set pointwise_sqr a");
    check(clSetKernelArg(gpu.k_pointwise_sqr, 1, sizeof(cl_uint), &n_u32), "set pointwise_sqr n");
    {
        const size_t global = gpu.n;
        enqueue_kernel(gpu, gpu.k_pointwise_sqr, global, nullptr, "enqueue pointwise_sqr", "pointwise_sqr");
    }
    enqueue_inverse_pipeline(gpu);
}


static void ensure_carry_buffers(GpuPrp& gpu) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    if (gpu.carry_buffer_digits >= n_u32 && gpu.bufBlockCarry && gpu.bufBlockIncoming && gpu.bufCrtCarryHi1 && gpu.bufCrtCarryHi2 && gpu.bufCarryPending && gpu.bufCarryStats) return;

    if (gpu.bufCrtCarryHi2) { clReleaseMemObject(gpu.bufCrtCarryHi2); gpu.bufCrtCarryHi2 = nullptr; }
    if (gpu.bufCrtCarryHi1) { clReleaseMemObject(gpu.bufCrtCarryHi1); gpu.bufCrtCarryHi1 = nullptr; }
    if (gpu.bufCarryStats) { clReleaseMemObject(gpu.bufCarryStats); gpu.bufCarryStats = nullptr; }
    if (gpu.bufCarryPending) { clReleaseMemObject(gpu.bufCarryPending); gpu.bufCarryPending = nullptr; }
    if (gpu.bufSegMode) { clReleaseMemObject(gpu.bufSegMode); gpu.bufSegMode = nullptr; }
    if (gpu.bufSegThreshold) { clReleaseMemObject(gpu.bufSegThreshold); gpu.bufSegThreshold = nullptr; }
    if (gpu.bufSegBits) { clReleaseMemObject(gpu.bufSegBits); gpu.bufSegBits = nullptr; }
    if (gpu.bufSegValueLo) { clReleaseMemObject(gpu.bufSegValueLo); gpu.bufSegValueLo = nullptr; }
    if (gpu.bufFinalCarry) { clReleaseMemObject(gpu.bufFinalCarry); gpu.bufFinalCarry = nullptr; }
    if (gpu.bufBlockIncoming) { clReleaseMemObject(gpu.bufBlockIncoming); gpu.bufBlockIncoming = nullptr; }
    if (gpu.bufBlockMode) { clReleaseMemObject(gpu.bufBlockMode); gpu.bufBlockMode = nullptr; }
    if (gpu.bufBlockThreshold) { clReleaseMemObject(gpu.bufBlockThreshold); gpu.bufBlockThreshold = nullptr; }
    if (gpu.bufBlockBits) { clReleaseMemObject(gpu.bufBlockBits); gpu.bufBlockBits = nullptr; }
    if (gpu.bufBlockValueLo) { clReleaseMemObject(gpu.bufBlockValueLo); gpu.bufBlockValueLo = nullptr; }
    if (gpu.bufBlockCarry) { clReleaseMemObject(gpu.bufBlockCarry); gpu.bufBlockCarry = nullptr; }

    cl_int err = CL_SUCCESS;
    const std::size_t elems = std::size_t(std::max<cl_uint>(n_u32, 1u));
    gpu.bufBlockCarry = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, elems * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(carryPing)");
    gpu.bufBlockIncoming = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, elems * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(carryPong)");
    gpu.bufCrtCarryHi1 = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, elems * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(crtCarryHiPing)");
    gpu.bufCrtCarryHi2 = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, elems * sizeof(cl_ulong), nullptr, &err);
    check(err, "clCreateBuffer(crtCarryHiPong)");
    gpu.bufCarryPending = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    check(err, "clCreateBuffer(carryPending)");
    gpu.bufCarryStats = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sizeof(cl_uint) * 8u, nullptr, &err);
    check(err, "clCreateBuffer(carryStats)");
    {
        const cl_uint zero_stats[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
        check(clEnqueueWriteBuffer(gpu.queue, gpu.bufCarryStats, CL_TRUE, 0, sizeof(zero_stats), zero_stats, 0, nullptr, nullptr),
              "clEnqueueWriteBuffer(carryStats zero)");
        gpu.carry_stats_last_runs = 0u;
        gpu.carry_stats_last_rounds = 0u;
    }

    gpu.carry_buffer_digits = n_u32;
    gpu.carry_buffer_blocks = 0;
    gpu.carry_buffer_segments = 0;
}

static void enqueue_carry(GpuPrp& gpu, const CarryConfig& cfg) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    ensure_carry_buffers(gpu);

    const cl_uint items = std::max<cl_uint>(1u, static_cast<cl_uint>(cfg.items_per_worker));
    const cl_uint segments = static_cast<cl_uint>((gpu.n + std::size_t(items) - 1u) / std::size_t(items));
    const size_t local = std::max<std::size_t>(1u, static_cast<std::size_t>(cfg.local_size));
    const size_t global = round_up_size(std::max<std::size_t>(1u, segments), local);

    cl_mem carry_in = gpu.bufBlockCarry;
    cl_mem carry_out = gpu.bufBlockIncoming;

    check(clSetKernelArg(gpu.k_carry_block_local, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_segment_first digits");
    check(clSetKernelArg(gpu.k_carry_block_local, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_segment_first widths");
    check(clSetKernelArg(gpu.k_carry_block_local, 2, sizeof(cl_mem), &carry_in), "set carry_segment_first out");
    check(clSetKernelArg(gpu.k_carry_block_local, 3, sizeof(cl_mem), &gpu.bufCarryPending), "set carry_segment_first pending");
    check(clSetKernelArg(gpu.k_carry_block_local, 4, sizeof(cl_uint), &n_u32), "set carry_segment_first n");
    check(clSetKernelArg(gpu.k_carry_block_local, 5, sizeof(cl_uint), &segments), "set carry_segment_first segments");
    check(clSetKernelArg(gpu.k_carry_block_local, 6, sizeof(cl_uint), &items), "set carry_segment_first items");
    enqueue_kernel(gpu, gpu.k_carry_block_local, global, &local, "enqueue carry segment first", "carry_first");

    for (cl_uint pass = 1; pass < gpu.carry_passes; ++pass) {
        const cl_uint set_pending = (pass + 1u == gpu.carry_passes) ? 1u : 0u;
        check(clSetKernelArg(gpu.k_carry_block_prefix, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_segment_pass digits");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_segment_pass widths");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 2, sizeof(cl_mem), &carry_in), "set carry_segment_pass in");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 3, sizeof(cl_mem), &carry_out), "set carry_segment_pass out");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 4, sizeof(cl_mem), &gpu.bufCarryPending), "set carry_segment_pass pending");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 5, sizeof(cl_uint), &n_u32), "set carry_segment_pass n");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 6, sizeof(cl_uint), &segments), "set carry_segment_pass segments");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 7, sizeof(cl_uint), &items), "set carry_segment_pass items");
        check(clSetKernelArg(gpu.k_carry_block_prefix, 8, sizeof(cl_uint), &set_pending), "set carry_segment_pass set_pending");
        enqueue_kernel(gpu, gpu.k_carry_block_prefix, global, &local, "enqueue carry segment pass", "carry_pass");
        std::swap(carry_in, carry_out);
    }

    check(clSetKernelArg(gpu.k_carry_final_wrap, 0, sizeof(cl_mem), &gpu.bufDigits), "set carry_cleanup digits");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 1, sizeof(cl_mem), &gpu.bufWidth), "set carry_cleanup widths");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 2, sizeof(cl_mem), &carry_in), "set carry_cleanup carry");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 3, sizeof(cl_mem), &gpu.bufCarryPending), "set carry_cleanup pending");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 4, sizeof(cl_mem), &gpu.bufCarryStats), "set carry_cleanup stats");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 5, sizeof(cl_uint), &n_u32), "set carry_cleanup n");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 6, sizeof(cl_uint), &segments), "set carry_cleanup segments");
    check(clSetKernelArg(gpu.k_carry_final_wrap, 7, sizeof(cl_uint), &items), "set carry_cleanup items");
    {
        const size_t one = 1;
        enqueue_kernel(gpu, gpu.k_carry_final_wrap, one, &one, "enqueue carry cleanup", "carry_cleanup");
    }
}

static void print_carry_stats_if_changed(GpuPrp& gpu, std::uint32_t iter_plus_one) {
    if (!gpu.bufCarryStats) return;

    cl_uint stats[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    check(clEnqueueReadBuffer(gpu.queue, gpu.bufCarryStats, CL_TRUE, 0, sizeof(stats), stats, 0, nullptr, nullptr),
          "clEnqueueReadBuffer(carryStats)");

    const cl_uint runs = stats[0];
    const cl_uint rounds_total = stats[1];
    const cl_uint max_rounds = stats[2];
    const cl_uint last_rounds = stats[3];

    const cl_uint delta_runs = runs - gpu.carry_stats_last_runs;
    const cl_uint delta_rounds = rounds_total - gpu.carry_stats_last_rounds;

    const double avg_all = runs ? (double(rounds_total) / double(runs)) : 0.0;
    const double avg_recent = delta_runs ? (double(delta_rounds) / double(delta_runs)) : 0.0;


    gpu.carry_stats_last_runs = runs;
    gpu.carry_stats_last_rounds = rounds_total;
}

static void enqueue_mul_small(GpuPrp& gpu, cl_uint k) {
    const cl_uint n_u32 = static_cast<cl_uint>(gpu.n);
    check(clSetKernelArg(gpu.k_mul_small, 0, sizeof(cl_mem), &gpu.bufDigits), "set mul_small digits");
    check(clSetKernelArg(gpu.k_mul_small, 1, sizeof(cl_uint), &k), "set mul_small k");
    check(clSetKernelArg(gpu.k_mul_small, 2, sizeof(cl_uint), &n_u32), "set mul_small n");
    const size_t global = gpu.n;
    enqueue_kernel(gpu, gpu.k_mul_small, global, nullptr, "enqueue mul_small", "mul_small");
}

static void upload_digits(GpuPrp& gpu, const std::vector<std::uint64_t>& digits) {
    check(clEnqueueWriteBuffer(gpu.queue, gpu.bufDigits, CL_FALSE, 0, digits.size() * sizeof(std::uint64_t), digits.data(), 0, nullptr, nullptr),
          "write digits");
}

static std::vector<std::uint64_t> read_digits(GpuPrp& gpu) {
    std::vector<std::uint64_t> digits(gpu.n);
    check(clEnqueueReadBuffer(gpu.queue, gpu.bufDigits, CL_TRUE, 0, digits.size() * sizeof(std::uint64_t), digits.data(), 0, nullptr, nullptr),
          "read digits");
    return digits;
}


static void enqueue_crt_garner_carry_gpu(GpuPrp& gpu61, GpuPrp& gpu31, const CarryConfig& cfg, bool oneout_digits = false) {
    if (gpu61.context != gpu31.context) {
        throw std::runtime_error("CRT GPU Garner requires shared OpenCL context");
    }
    ensure_carry_buffers(gpu61);
    const cl_uint digit_n = static_cast<cl_uint>(gpu61.n);
    
    
    const cl_uint minw = std::max<cl_uint>(1u, gpu61.min_digit_width);
    cl_uint items = crt_tune::garner_items(digit_n, minw, static_cast<cl_uint>(cfg.items_per_worker));
    const bool precomputed_first_carry = gpu61.crtFirstCarryReady;
    const bool prefer_coeffhi_x2 = gpu61.crtCoeffPending &&
                                   parse_bool_env("PRMERS_CRT_GARNER_X2", true) &&
                                   gpu61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2;
    if (precomputed_first_carry) {
        items = 64u;
    } else if (gpu61.crtLastUnweightPending && items != 32u) {
        items = 32u;
    } else if (gpu61.crtCoeffPending) {
        if (prefer_coeffhi_x2) items = 64u;
        else if (items != 32u) items = 32u;
    }
    const cl_uint segments = static_cast<cl_uint>((gpu61.n + std::size_t(items) - 1u) / std::size_t(items));
    const std::size_t local = static_cast<std::size_t>(crt_tune::garner_local(segments));
    const std::size_t global = round_up_size(std::max<std::size_t>(1u, segments), local);

    const cl_uint bits_per_segment = std::max<cl_uint>(1u, items * minw);
    
    
    const cl_uint crt_passes = std::max<cl_uint>(1u, std::min<cl_uint>(8u, (128u + bits_per_segment - 1u) / bits_per_segment));

    cl_uint zero = 0;
    if (!precomputed_first_carry) {
        check(clEnqueueWriteBuffer(gpu61.queue, gpu61.bufCarryPending, CL_FALSE, 0, sizeof(zero), &zero, 0, nullptr, nullptr), "crt clear pending");
    }

    const cl_mem gpu31_digits = gpu31.bufDigits32 ? gpu31.bufDigits32 : gpu31.bufDigits;

    const bool widthmask_any_ok = parse_bool_env("PRMERS_CRT_WIDTHMASK32", true) && oneout_digits && gpu61.bufWidthMask32;
    const bool widthmask32_ok = widthmask_any_ok && (items == 32u);
    const bool widthmask64_ok = widthmask_any_ok && (items == 64u);
    const bool use_coeffhi_base32_x2 = gpu61.crtCoeffPending && widthmask64_ok && oneout_digits &&
                                       parse_bool_env("PRMERS_CRT_GARNER_X2", true) &&
                                       gpu61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2;
    const bool use_coeffhi_base32 = !use_coeffhi_base32_x2 && gpu61.crtCoeffPending && widthmask32_ok && oneout_digits &&
                                    gpu61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32;
    if (gpu61.crtCoeffPending && !use_coeffhi_base32_x2 && !use_coeffhi_base32) {
        throw std::runtime_error("CRT precombined tail requested, but coeffhi mask32/base32 Garner kernel is unavailable");
    }
    const bool use_fused_last_garner_mask32 = !use_coeffhi_base32_x2 && !use_coeffhi_base32 && gpu61.crtLastUnweightPending && widthmask32_ok &&
                                              gpu61.k_crt_last_unweight_garner_segment_first_oneout_mask32;
    const bool use_fused_last_garner = !use_coeffhi_base32_x2 && !use_coeffhi_base32 && gpu61.crtLastUnweightPending && oneout_digits && !use_fused_last_garner_mask32 &&
                                       gpu61.k_crt_last_unweight_garner_segment_first_oneout;
    const bool use_widthmask32_garner = widthmask32_ok && !gpu61.crtLastUnweightPending && !use_coeffhi_base32_x2 && !use_coeffhi_base32 &&
                                        gpu61.k_crt_garner_segment_first_oneout_mask32;
    const bool use_widthmask64_base32_x2 = widthmask64_ok && !gpu61.crtLastUnweightPending && !use_coeffhi_base32_x2 && !use_coeffhi_base32 &&
                                           parse_bool_env("PRMERS_CRT_GARNER_X2", true) &&
                                           gpu61.min_digit_width == 32u &&
                                           gpu61.k_crt_garner_segment_first_oneout_mask32_base32_x2;
    const bool use_widthmask32_base32_u32lean = use_widthmask32_garner &&
                                                parse_bool_env("PRMERS_CRT_GARNER_U32LEAN", false) &&
                                                gpu61.min_digit_width == 32u &&
                                                gpu61.k_crt_garner_segment_first_oneout_mask32_base32_u32lean;
    const bool use_widthmask32_base32_fast = !use_widthmask32_base32_u32lean && use_widthmask32_garner &&
                                            parse_bool_env("PRMERS_CRT_GARNER_BASE32_FAST", true) &&
                                            gpu61.min_digit_width == 32u &&
                                            gpu61.k_crt_garner_segment_first_oneout_mask32_base32_fast;
    cl_kernel k_first = use_coeffhi_base32_x2 ? gpu61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32_x2
                      : use_coeffhi_base32 ? gpu61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32
                      : (use_fused_last_garner_mask32 ? gpu61.k_crt_last_unweight_garner_segment_first_oneout_mask32
                         : (use_fused_last_garner ? gpu61.k_crt_last_unweight_garner_segment_first_oneout
                            : (use_widthmask64_base32_x2 ? gpu61.k_crt_garner_segment_first_oneout_mask32_base32_x2
                               : (use_widthmask32_base32_u32lean ? gpu61.k_crt_garner_segment_first_oneout_mask32_base32_u32lean
                                  : (use_widthmask32_base32_fast ? gpu61.k_crt_garner_segment_first_oneout_mask32_base32_fast
                                     : (use_widthmask32_garner ? gpu61.k_crt_garner_segment_first_oneout_mask32
                                        : (oneout_digits ? gpu61.k_crt_garner_segment_first_oneout : gpu61.k_crt_garner_segment_first)))))));
    cl_kernel k_pass  = oneout_digits ? gpu61.k_crt_carry_segment_pass_oneout  : gpu61.k_crt_carry_segment_pass;
    cl_kernel k_clean = oneout_digits ? gpu61.k_crt_carry_cleanup_serial_oneout : gpu61.k_crt_carry_cleanup_serial;

    const bool can_carry_pack_next_lds = g_crt_mixed_carry_pack_next_request && oneout_digits && precomputed_first_carry &&
        parse_bool_env("PRMERS_CRT_MIXED_CARRY_PACK_NEXT_LDS", false) &&
        parse_bool_env("PRMERS_CRT_MIXED_CARRY_PACK_ASSUME_NO_WRAP", true) &&
        gpu61.k_crt_mixed_carry_pack_next_lds_61x31 &&
        gpu61.n == gpu31.n && g_crt_odd_radix == 9u && (items == 32u || items == 64u) && crt_passes == 1u &&
        gpu61.bufField && gpu31.bufField && gpu61.bufOddFwd && gpu31.bufOddFwd &&
        gpu61.bufUnweightShift && gpu31.bufUnweightShift && gpu61.bufWidth;
    g_crt_mixed_carry_pack_next_done = false;

    auto enqueue_profiled_kernel = [&](cl_kernel kernel,
                                       const std::size_t* global_ptr,
                                       const std::size_t* local_ptr,
                                       cl_uint wait_count,
                                       const cl_event* wait_list,
                                       const char* what,
                                       const char* label,
                                       cl_event* out_event = nullptr) {
        cl_event ev = nullptr;
        cl_event* ev_ptr = (gpu61.profile_kernels || out_event) ? &ev : nullptr;
        check(clEnqueueNDRangeKernel(gpu61.queue, kernel, 1, nullptr, global_ptr, local_ptr,
                                     wait_count, wait_list, ev_ptr), what);
        if (gpu61.profile_kernels && ev) {
            if (out_event) clRetainEvent(ev);
            gpu61.pending_profile_events.push_back({label, ev});
        }
        if (out_event) *out_event = ev;
    };

    int arg = 0;
    if (!precomputed_first_carry) {
    if (use_fused_last_garner || use_fused_last_garner_mask32) {
        const cl_uint ntt_n = static_cast<cl_uint>(gpu61.n);
        const cl_uint log_n = static_cast<cl_uint>(gpu61.log_n);
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufField), "crt_last_garner arg0 a61");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu31.bufField), "crt_last_garner arg1 a31");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufTwInv), "crt_last_garner arg2 tw61");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu31.bufTwInv), "crt_last_garner arg3 tw31");
        cl_mem width_arg = use_fused_last_garner_mask32 ? gpu61.bufWidthMask32 : gpu61.bufWidth;
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &width_arg), "crt_last_garner arg4 width/mask32");
        if (use_fused_last_garner_mask32) {
            cl_uint width_base = static_cast<cl_uint>(gpu61.min_digit_width);
            check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &width_base), "crt_last_garner arg4b width_base");
        }
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufDigits), "crt_last_garner arg5 digits61");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufBlockCarry), "crt_last_garner arg6 lo");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufCrtCarryHi1), "crt_last_garner arg7 hi");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "crt_last_garner arg8 pending");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &ntt_n), "crt_last_garner arg9 n");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &gpu61.exponent_p), "crt_last_garner arg10 p");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &gpu61.lr2), "crt_last_garner arg11 lr2_61");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &gpu31.lr2), "crt_last_garner arg12 lr2_31");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &log_n), "crt_last_garner arg13 log_n");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &digit_n), "crt_last_garner arg14 digit_n");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &segments), "crt_last_garner arg15 seg");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &items), "crt_last_garner arg16 items");
    } else if (use_widthmask32_garner || use_widthmask64_base32_x2 || use_coeffhi_base32_x2 || use_coeffhi_base32) {
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufDigits), "crt_first_mask32 arg0");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu31_digits), "crt_first_mask32 arg1");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufWidthMask32), "crt_first_mask32 arg2");
        cl_uint width_base = static_cast<cl_uint>(gpu61.min_digit_width);
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &width_base), "crt_first_mask32 arg2b width_base");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufBlockCarry), "crt_first_mask32 arg3 lo");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufCrtCarryHi1), "crt_first_mask32 arg4 hi");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "crt_first_mask32 arg5 pending");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &digit_n), "crt_first_mask32 arg6 n");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &segments), "crt_first_mask32 arg7 seg");
    } else {
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufDigits), "crt_first arg0");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu31_digits), "crt_first arg1");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufWidth), "crt_first arg2");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufBlockCarry), "crt_first arg3 lo");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufCrtCarryHi1), "crt_first arg4 hi");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "crt_first arg5 pending");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &digit_n), "crt_first arg6 n");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &segments), "crt_first arg7 seg");
        check(clSetKernelArg(k_first, arg++, sizeof(cl_uint), &items), "crt_first arg8 items");
    }
    }
    if (!precomputed_first_carry) {
        cl_event crt_wait_ev = take_pending_wait_event(gpu61);
        const char* first_label = use_coeffhi_base32_x2
            ? (gpu61.min_digit_width == 32u ? "crt_garner_first_coeffhi_mask32_base32_x2" : "crt_garner_first_coeffhi_mask32_anybase_x2")
            : use_coeffhi_base32
            ? (gpu61.min_digit_width == 32u ? "crt_garner_first_coeffhi_mask32_base32" : "crt_garner_first_coeffhi_mask32_anybase")
            : use_fused_last_garner_mask32 ? "crt_last_unweight_garner_first_mask32"
            : use_fused_last_garner ? "crt_last_unweight_garner_first"
            : use_widthmask64_base32_x2 ? "crt_garner_first_mask32_base32_x2"
            : use_widthmask32_base32_u32lean ? "crt_garner_first_mask32_base32_u32lean"
            : use_widthmask32_base32_fast ? "crt_garner_first_mask32_base32_fast"
            : use_widthmask32_garner ? "crt_garner_first_mask32"
            : oneout_digits ? "crt_garner_first_oneout"
            : "crt_garner_first";
        enqueue_profiled_kernel(k_first, &global, &local,
                                crt_wait_ev ? 1u : 0u, crt_wait_ev ? &crt_wait_ev : nullptr,
                                "enqueue crt first", first_label);
        if (crt_wait_ev) clReleaseEvent(crt_wait_ev);
    }
    gpu61.crtLastUnweightPending = false;
    gpu61.crtCoeffPending = false;
    gpu61.crtFirstCarryReady = false;

    cl_mem lo_in = gpu61.bufBlockCarry;
    cl_mem hi_in = gpu61.bufCrtCarryHi1;
    cl_mem lo_out = gpu61.bufBlockIncoming;
    cl_mem hi_out = gpu61.bufCrtCarryHi2;
    for (cl_uint pass = 0; pass < crt_passes; ++pass) {
        
        
        check(clEnqueueWriteBuffer(gpu61.queue, gpu61.bufCarryPending, CL_FALSE, 0, sizeof(zero), &zero, 0, nullptr, nullptr), "crt reset pass pending");
        if (pass == 0u && can_carry_pack_next_lds) {
            const cl_uint odd = 9u;
            const cl_uint pow2_n = digit_n / odd;
            const cl_uint bseg_count = pow2_n / items;
            const std::size_t carry_pack_local = 256u;
            const std::size_t carry_pack_global = round_up_size(static_cast<std::size_t>(bseg_count) * carry_pack_local, carry_pack_local);
            cl_kernel k_cp = gpu61.k_crt_mixed_carry_pack_next_lds_61x31;
            arg = 0;
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu61.bufDigits), "carry_pack arg0 digits");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu61.bufWidth), "carry_pack arg1 width");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &lo_in), "carry_pack arg2 lo_in");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &hi_in), "carry_pack arg3 hi_in");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &lo_out), "carry_pack arg4 lo_out");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &hi_out), "carry_pack arg5 hi_out");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "carry_pack arg6 pending");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu61.bufField), "carry_pack arg7 a61");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu31.bufField), "carry_pack arg8 a31");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu61.bufOddFwd), "carry_pack arg9 mat61");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu31.bufOddFwd), "carry_pack arg10 mat31");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu61.bufUnweightShift), "carry_pack arg11 shift61");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_mem), &gpu31.bufUnweightShift), "carry_pack arg12 shift31");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_uint), &digit_n), "carry_pack arg13 n");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_uint), &segments), "carry_pack arg14 segments");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_uint), &items), "carry_pack arg15 items");
            check(clSetKernelArg(k_cp, arg++, sizeof(cl_uint), &pow2_n), "carry_pack arg16 pow2_n");
            enqueue_profiled_kernel(k_cp, &carry_pack_global, &carry_pack_local, 0, nullptr,
                                    "enqueue crt carry-pack-next lds", "crt_mixed_carry_pack_next_lds_61x31");
            g_crt_mixed_carry_pack_next_done = true;
        } else {
            arg = 0;
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &gpu61.bufDigits), "crt_pass arg0");
            if (!oneout_digits) check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &gpu31_digits), "crt_pass arg1");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &gpu61.bufWidth), "crt_pass arg2");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &lo_in), "crt_pass arg3 lo_in");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &hi_in), "crt_pass arg4 hi_in");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &lo_out), "crt_pass arg5 lo_out");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &hi_out), "crt_pass arg6 hi_out");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "crt_pass arg7 pending");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_uint), &digit_n), "crt_pass arg8 n");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_uint), &segments), "crt_pass arg9 seg");
            check(clSetKernelArg(k_pass, arg++, sizeof(cl_uint), &items), "crt_pass arg10 items");
            enqueue_profiled_kernel(k_pass, &global, &local, 0, nullptr,
                                    "enqueue crt pass", oneout_digits ? "crt_carry_pass_oneout" : "crt_carry_pass");
        }
        std::swap(lo_in, lo_out);
        std::swap(hi_in, hi_out);
    }

    const std::size_t one = 1;
    cl_event crt_done_ev = nullptr;

    
    const bool force_serial_cleanup = (std::getenv("PRMERS_CRT_SERIAL_CLEANUP") != nullptr);
    const bool force_parallel_cleanup = (std::getenv("PRMERS_CRT_PARALLEL_CLEANUP") != nullptr);
    
    
    const bool enable_parallel_cleanup = oneout_digits && force_parallel_cleanup && !force_serial_cleanup &&
                                         gpu61.k_crt_carry_cleanup_parallel_oneout;
    if (enable_parallel_cleanup) {
        check(clEnqueueWriteBuffer(gpu61.queue, gpu61.bufCarryPending, CL_FALSE, 0, sizeof(zero), &zero, 0, nullptr, nullptr),
              "crt reset parallel cleanup pending");
        
        
        arg = 0;
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &gpu61.bufDigits), "crt_parallel_cleanup arg0 digits");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &gpu61.bufWidth), "crt_parallel_cleanup arg1 widths");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &lo_in), "crt_parallel_cleanup arg2 lo_in");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &hi_in), "crt_parallel_cleanup arg3 hi_in");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &lo_out), "crt_parallel_cleanup arg4 lo_out");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &hi_out), "crt_parallel_cleanup arg5 hi_out");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "crt_parallel_cleanup arg6 pending");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_uint), &digit_n), "crt_parallel_cleanup arg7 n");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_uint), &segments), "crt_parallel_cleanup arg8 seg");
        check(clSetKernelArg(gpu61.k_crt_carry_cleanup_parallel_oneout, arg++, sizeof(cl_uint), &items), "crt_parallel_cleanup arg9 items");
        enqueue_profiled_kernel(gpu61.k_crt_carry_cleanup_parallel_oneout, &global, &local, 0, nullptr,
                                "enqueue crt parallel cleanup", "crt_parallel_cleanup_oneout");

        lo_in = lo_out;
        hi_in = hi_out;
    }

    arg = 0;
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_mem), &gpu61.bufDigits), "crt_cleanup arg0");
    if (!oneout_digits) check(clSetKernelArg(k_clean, arg++, sizeof(cl_mem), &gpu31_digits), "crt_cleanup arg1");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_mem), &gpu61.bufWidth), "crt_cleanup arg2");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_mem), &lo_in), "crt_cleanup arg3 lo");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_mem), &hi_in), "crt_cleanup arg4 hi");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_mem), &gpu61.bufCarryPending), "crt_cleanup arg5 pending");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_uint), &digit_n), "crt_cleanup arg6 n");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_uint), &segments), "crt_cleanup arg7 seg");
    check(clSetKernelArg(k_clean, arg++, sizeof(cl_uint), &items), "crt_cleanup arg8 items");
    enqueue_profiled_kernel(k_clean, &one, nullptr, 0, nullptr,
                            "enqueue crt cleanup", oneout_digits ? "crt_cleanup_serial_oneout" : "crt_cleanup_serial",
                            &crt_done_ev);
    if (crt_done_ev) set_pending_wait_event(gpu31, crt_done_ev);

}

static cl_kernel load_kernel_optional(cl_program program, const char* name) {
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(program, name, &err);
    return (err == CL_SUCCESS) ? k : nullptr;
}


struct CrtFusedKernels {
    cl_kernel weight_first = nullptr;
    cl_kernel fwd_r2 = nullptr;
    cl_kernel fwd_r4 = nullptr;
    cl_kernel fwd_r8 = nullptr;
    cl_kernel fwd_bridge512 = nullptr;
    cl_kernel fwd_bridge1024 = nullptr;
    cl_kernel fwd_lds512 = nullptr;
    cl_kernel inv_lds512 = nullptr;
    cl_kernel fwd_lds_stage = nullptr;
    cl_kernel inv_lds_stage = nullptr;
    cl_kernel fwd_lds_stage_tile2 = nullptr;
    cl_kernel inv_lds_stage_tile2 = nullptr;
    cl_kernel center256 = nullptr;
    cl_kernel center8_dualwave = nullptr;
    cl_kernel center16_dualwave = nullptr;
    cl_kernel center32_dualwave = nullptr;
    cl_kernel center64_dualwave = nullptr;
    cl_kernel center128_dualwave = nullptr;
    cl_kernel center256_dualwave = nullptr;
    cl_kernel center512_dualwave = nullptr;
    cl_kernel center512_reglds = nullptr;
    cl_kernel center1024_dualwave = nullptr;
    cl_kernel inv_bridge512 = nullptr;
    cl_kernel inv_bridge1024 = nullptr;
    cl_kernel inv_r2 = nullptr;
    cl_kernel inv_r4 = nullptr;
    cl_kernel inv_r8 = nullptr;
    cl_kernel last_unweight = nullptr;
    cl_kernel last_unweight16 = nullptr;

    
    cl_kernel weight_first61 = nullptr;
    cl_kernel weight_first31 = nullptr;
    cl_kernel weight_edge61 = nullptr;
    cl_kernel weight_edge31 = nullptr;
    cl_kernel fwd_r2_61 = nullptr;
    cl_kernel fwd_r4_61 = nullptr;
    cl_kernel fwd_r8_61 = nullptr;
    cl_kernel fwd_r8_61_wg128 = nullptr;
    cl_kernel fwd_lds_stage_61 = nullptr;
    cl_kernel fwd_lds_stage_61_512opt = nullptr;
    cl_kernel fwd_lds_stage_61_tile4 = nullptr;
    std::array<cl_kernel, 8> fwd_lds_stage_61_1lds{};
    std::array<cl_kernel, 8> inv_lds_stage_61_1lds{};
    cl_kernel inv_r2_61 = nullptr;
    cl_kernel inv_r4_61 = nullptr;
    cl_kernel inv_r8_61 = nullptr;
    cl_kernel inv_lds_stage_61 = nullptr;
    cl_kernel inv_lds_stage_61_512opt = nullptr;
    cl_kernel inv_lds_stage_61_tile4 = nullptr;
    cl_kernel center512_61 = nullptr;
    cl_kernel center512_61_opt = nullptr;
    cl_kernel last_unweight61 = nullptr;
    cl_kernel last_unweight_edge61 = nullptr;
    cl_kernel fwd_r2_31 = nullptr;
    cl_kernel fwd_r4_31 = nullptr;
    cl_kernel fwd_r8_31 = nullptr;
    cl_kernel fwd_lds_stage_31 = nullptr;
    cl_kernel fwd_lds_stage_31_512opt = nullptr;
    cl_kernel fwd_lds_stage_31_tile4 = nullptr;
    std::array<cl_kernel, 8> fwd_lds_stage_31_1lds{};
    std::array<cl_kernel, 8> inv_lds_stage_31_1lds{};
    cl_kernel inv_r2_31 = nullptr;
    cl_kernel inv_r4_31 = nullptr;
    cl_kernel inv_r8_31 = nullptr;
    cl_kernel inv_lds_stage_31 = nullptr;
    cl_kernel inv_lds_stage_31_512opt = nullptr;
    cl_kernel inv_lds_stage_31_tile4 = nullptr;
    cl_kernel center512_31 = nullptr;
    cl_kernel center512_31_opt = nullptr;
    cl_kernel last_unweight31 = nullptr;
    cl_kernel last_unweight_edge31 = nullptr;
    cl_kernel weight_fwd8_61 = nullptr;
    cl_kernel weight_fwd8_31 = nullptr;
    cl_kernel inv4_last_unweight61 = nullptr;
    cl_kernel inv4_last_unweight31 = nullptr;
    cl_kernel inv4_last_unweight16_61 = nullptr;
    cl_kernel inv4_last_unweight16_31 = nullptr;
    cl_kernel inv4_last_unweight16_crtcoeff = nullptr;

    cl_kernel halfreal_pack61 = nullptr;
    cl_kernel halfreal_pack31 = nullptr;
    cl_kernel halfreal_center61 = nullptr;
    cl_kernel halfreal_center31 = nullptr;
    cl_kernel halfreal_bitrev_swap61 = nullptr;
    cl_kernel halfreal_bitrev_swap31 = nullptr;
    cl_kernel halfreal_center512_pair61 = nullptr;
    cl_kernel halfreal_center512_pair31 = nullptr;
    cl_kernel halfreal_lds512_pair61 = nullptr;
    cl_kernel halfreal_lds512_pair31 = nullptr;
    cl_kernel halfreal_head_lds512_61 = nullptr;
    cl_kernel halfreal_head_lds512_31 = nullptr;
    cl_kernel halfreal_head_lds512_precrt = nullptr;
    cl_kernel halfreal_tail_lds512_unpack61 = nullptr;
    cl_kernel halfreal_tail_lds512_unpack31 = nullptr;
    cl_kernel halfreal_tail_lds512_precrt = nullptr;
    cl_kernel halfreal_head_ldspow2_61 = nullptr;
    cl_kernel halfreal_head_ldspow2_31 = nullptr;
    cl_kernel halfreal_tail_ldspow2_unpack61 = nullptr;
    cl_kernel halfreal_tail_ldspow2_unpack31 = nullptr;
    cl_kernel halfreal_unpack61 = nullptr;
    cl_kernel halfreal_unpack31 = nullptr;
    cl_kernel mixed_pack61 = nullptr;
    cl_kernel mixed_pack31 = nullptr;
    cl_kernel mixed_odd_dft61 = nullptr;
    cl_kernel mixed_odd_dft31 = nullptr;
    cl_kernel mixed_fwd_r2_61 = nullptr;
    cl_kernel mixed_inv_r2_61 = nullptr;
    cl_kernel mixed_fwd_r2_31 = nullptr;
    cl_kernel mixed_inv_r2_31 = nullptr;
    cl_kernel mixed_center61 = nullptr;
    cl_kernel mixed_center31 = nullptr;
    cl_kernel mixed_lds512_pair61 = nullptr;
    cl_kernel mixed_lds512_pair31 = nullptr;
    cl_kernel mixed_lds512_pair_1lds61 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_rega_f48_61 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_rega_twinline_f48_61 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_rega_twinline_f48_31 = nullptr;
    cl_kernel mixed_lds512_pair_1lds31 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_f48_self61 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_f48_nonself61 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_f48_self31 = nullptr;
    cl_kernel mixed_lds512_pair_1lds_f48_nonself31 = nullptr;
    cl_kernel mixed_lds1024_pair61 = nullptr;
    cl_kernel mixed_lds1024_pair31 = nullptr;
    cl_kernel mixed_lds_any_pair61 = nullptr;
    cl_kernel mixed_lds_any_pair31 = nullptr;
    cl_kernel mixed_lds_any_pair_1lds61 = nullptr;
    cl_kernel mixed_lds_any_pair_1lds31 = nullptr;
    cl_kernel mixed_lds_any_pair_1lds_f48_self61 = nullptr;
    cl_kernel mixed_lds_any_pair_1lds_f48_nonself61 = nullptr;
    cl_kernel mixed_lds_any_pair_1lds_f48_self31 = nullptr;
    cl_kernel mixed_lds_any_pair_1lds_f48_nonself31 = nullptr;
    cl_kernel mixed_lds_any_pair_both = nullptr;
    cl_kernel mixed_lds512_pair_both = nullptr;
    cl_kernel mixed_fwd_lds512_both = nullptr;
    cl_kernel mixed_inv_lds512_both = nullptr;
    cl_kernel mixed_fwd_lds_any_both = nullptr;
    cl_kernel mixed_inv_lds_any_both = nullptr;
    cl_kernel mixed_unpack61 = nullptr;
    cl_kernel mixed_unpack31 = nullptr;
    cl_kernel mixed_pack_odd_fwd61 = nullptr;
    cl_kernel mixed_pack_odd_fwd31 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile7_61 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile7_31 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_61 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_31 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_shift_61 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_shift_31 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_shift_lmat_61 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_shift_lmat_31 = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile14_shift_lmat_both = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile28_shift_lmat_both = nullptr;
    cl_kernel mixed_pack_odd_fwd_tile7_both = nullptr;
    cl_kernel mixed_pack_odd_fwd_shift61 = nullptr;
    cl_kernel mixed_pack_odd_fwd_shift31 = nullptr;
    cl_kernel mixed_pack_odd_fwd_both_shift = nullptr;
    cl_kernel mixed_odd_inv_unpack61 = nullptr;
    cl_kernel mixed_odd_inv_unpack31 = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi_tile7 = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi_tile14 = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi_tile14_shift = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat = nullptr;
    cl_kernel mixed_odd9_inv_precrt_garner64_lmat = nullptr;
    cl_kernel mixed_odd9_inv_precrt_garner9seg_lmat = nullptr;
    cl_kernel mixed_odd9_inv_precrt_garner9seg30_pair_lmat = nullptr;
    cl_kernel mixed_odd9_inv_precrt_garner9seg30_pair_smat = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi_outpar = nullptr;
    cl_kernel mixed_odd_inv_precrt_coeffhi_shift = nullptr;
    cl_kernel mixed_residues_to_coeffhi = nullptr;

    explicit CrtFusedKernels(cl_program program) {
        weight_first = load_kernel_optional(program, "gf61_crt_weight_first_stage_dif_radix4_wg64");
        fwd_r2 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix2");
        fwd_r4 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix4");
        fwd_r8 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix8");
        fwd_bridge512 = load_kernel_optional(program, "gf61_crt_forward_bridge_512_to_256");
        fwd_bridge1024 = load_kernel_optional(program, "gf61_crt_forward_bridge_1024_to_256");
        fwd_lds512 = load_kernel_optional(program, "gf61_crt_forward_lds512_to_center");
        inv_lds512 = load_kernel_optional(program, "gf61_crt_inverse_lds512_from_center");
        fwd_lds_stage = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2");
        inv_lds_stage = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2");
        fwd_lds_stage_tile2 = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_tile2");
        inv_lds_stage_tile2 = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_tile2");
        center256 = load_kernel_optional(program, "gf61_crt_center_fused_256");
        center8_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_8_dualwave");
        center16_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_16_dualwave");
        center32_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_32_dualwave");
        center64_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_64_dualwave");
        center128_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_128_dualwave");
        center256_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_256_dualwave");
        center512_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_512_dualwave");
        center512_reglds = load_kernel_optional(program, "gf61_crt_center_fused_512_reglds");
        center1024_dualwave = load_kernel_optional(program, "gf61_crt_center_fused_1024_dualwave");
        inv_bridge512 = load_kernel_optional(program, "gf61_crt_inverse_bridge_256_to_512");
        inv_bridge1024 = load_kernel_optional(program, "gf61_crt_inverse_bridge_256_to_1024");
        inv_r2 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix2");
        inv_r4 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix4");
        inv_r8 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix8");
        last_unweight = load_kernel_optional(program, "gf61_crt_last_stage_dit_radix4_unweight_wg64");
        last_unweight16 = load_kernel_optional(program, "gf61_crt_last_stage_dit_radix16_unweight");

        weight_first61 = load_kernel_optional(program, "gf61_crt_weight_first_stage_dif_radix4_wg64_61");
        weight_first31 = load_kernel_optional(program, "gf61_crt_weight_first_stage_dif_radix4_wg64_31");
        weight_edge61 = load_kernel_optional(program, "gf61_crt_weight_first_stage_dif_edge_61");
        weight_edge31 = load_kernel_optional(program, "gf61_crt_weight_first_stage_dif_edge_31");
        fwd_r2_61 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix2_61");
        fwd_r4_61 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix4_61");
        fwd_r8_61 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix8_61");
        fwd_r8_61_wg128 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix8_61_wg128");
        fwd_lds_stage_61 = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_61");
        fwd_lds_stage_61_512opt = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_61_512opt");
        fwd_lds_stage_61_tile4 = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_61_tile4");
        inv_r2_61 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix2_61");
        inv_r4_61 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix4_61");
        inv_r8_61 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix8_61");
        inv_lds_stage_61 = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_61");
        inv_lds_stage_61_512opt = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_61_512opt");
        inv_lds_stage_61_tile4 = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_61_tile4");
        center512_61 = load_kernel_optional(program, "gf61_crt_center_fused_512_61");
        center512_61_opt = load_kernel_optional(program, "gf61_crt_center_fused_512_61_opt");
        last_unweight61 = load_kernel_optional(program, "gf61_crt_last_stage_dit_radix4_unweight_wg64_61");
        last_unweight_edge61 = load_kernel_optional(program, "gf61_crt_last_stage_dit_unweight_edge_61");
        fwd_r2_31 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix2_31");
        fwd_r4_31 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix4_31");
        fwd_r8_31 = load_kernel_optional(program, "gf61_crt_ntt_stage_dif_radix8_31");
        fwd_lds_stage_31 = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_31");
        fwd_lds_stage_31_512opt = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_31_512opt");
        fwd_lds_stage_31_tile4 = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_31_tile4");
        inv_r2_31 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix2_31");
        inv_r4_31 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix4_31");
        inv_r8_31 = load_kernel_optional(program, "gf61_crt_ntt_stage_dit_radix8_31");
        inv_lds_stage_31 = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_31");
        inv_lds_stage_31_512opt = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_31_512opt");
        inv_lds_stage_31_tile4 = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_31_tile4");

        const char* mixed_stage_sizes[8] = {"8", "16", "32", "64", "128", "256", "512", "1024"};
        char kname[128];
        for (int i = 0; i < 8; ++i) {
            std::snprintf(kname, sizeof(kname), "gf61_crt_lds_stage_dif_pow2_61_1lds_%s", mixed_stage_sizes[i]);
            fwd_lds_stage_61_1lds[static_cast<size_t>(i)] = load_kernel_optional(program, kname);
            std::snprintf(kname, sizeof(kname), "gf61_crt_lds_stage_dit_pow2_61_1lds_%s", mixed_stage_sizes[i]);
            inv_lds_stage_61_1lds[static_cast<size_t>(i)] = load_kernel_optional(program, kname);
            std::snprintf(kname, sizeof(kname), "gf61_crt_lds_stage_dif_pow2_31_1lds_%s", mixed_stage_sizes[i]);
            fwd_lds_stage_31_1lds[static_cast<size_t>(i)] = load_kernel_optional(program, kname);
            std::snprintf(kname, sizeof(kname), "gf61_crt_lds_stage_dit_pow2_31_1lds_%s", mixed_stage_sizes[i]);
            inv_lds_stage_31_1lds[static_cast<size_t>(i)] = load_kernel_optional(program, kname);
        }
        center512_31 = load_kernel_optional(program, "gf61_crt_center_fused_512_31");
        center512_31_opt = load_kernel_optional(program, "gf61_crt_center_fused_512_31_opt");
        last_unweight31 = load_kernel_optional(program, "gf61_crt_last_stage_dit_radix4_unweight_wg64_31");
        last_unweight_edge31 = load_kernel_optional(program, "gf61_crt_last_stage_dit_unweight_edge_31");
        weight_fwd8_61 = load_kernel_optional(program, "gf61_crt_weight_radix4_fwd_radix8_61");
        weight_fwd8_31 = load_kernel_optional(program, "gf61_crt_weight_radix4_fwd_radix8_31");
        inv4_last_unweight61 = load_kernel_optional(program, "gf61_crt_inv_radix4_last_unweight_61");
        inv4_last_unweight31 = load_kernel_optional(program, "gf61_crt_inv_radix4_last_unweight_31");
        inv4_last_unweight16_61 = load_kernel_optional(program, "gf61_crt_inv_radix4_last_unweight_block16_61");
        inv4_last_unweight16_31 = load_kernel_optional(program, "gf61_crt_inv_radix4_last_unweight_block16_31");
        inv4_last_unweight16_crtcoeff = load_kernel_optional(program, "gf61_crt_inv_radix4_last_unweight_block16_crtcoeff");
        halfreal_pack61 = load_kernel_optional(program, "gf61_crt_halfreal_pack_weight_61");
        halfreal_pack31 = load_kernel_optional(program, "gf61_crt_halfreal_pack_weight_31");
        halfreal_center61 = load_kernel_optional(program, "gf61_crt_halfreal_center_61");
        halfreal_center31 = load_kernel_optional(program, "gf61_crt_halfreal_center_31");
        halfreal_bitrev_swap61 = load_kernel_optional(program, "gf61_crt_halfreal_bitrev_swap_61");
        halfreal_bitrev_swap31 = load_kernel_optional(program, "gf61_crt_halfreal_bitrev_swap_31");
        halfreal_center512_pair61 = load_kernel_optional(program, "gf61_crt_halfreal_center512_pair_61");
        halfreal_center512_pair31 = load_kernel_optional(program, "gf61_crt_halfreal_center512_pair_31");
        halfreal_lds512_pair61 = load_kernel_optional(program, "gf61_crt_halfreal_lds512_pair_61");
        halfreal_lds512_pair31 = load_kernel_optional(program, "gf61_crt_halfreal_lds512_pair_31");
        halfreal_head_lds512_61 = load_kernel_optional(program, "gf61_crt_halfreal_pack_lds512_dif_61");
        halfreal_head_lds512_31 = load_kernel_optional(program, "gf61_crt_halfreal_pack_lds512_dif_31");
        halfreal_head_lds512_precrt = load_kernel_optional(program, "gf61_crt_halfreal_pack_lds512_dif_precrt");
        halfreal_tail_lds512_unpack61 = load_kernel_optional(program, "gf61_crt_halfreal_dit512_unpack_61");
        halfreal_tail_lds512_unpack31 = load_kernel_optional(program, "gf61_crt_halfreal_dit512_unpack_31");
        halfreal_tail_lds512_precrt = load_kernel_optional(program, "gf61_crt_halfreal_dit512_unpack_precrt");
        halfreal_head_ldspow2_61 = load_kernel_optional(program, "gf61_crt_halfreal_pack_ldspow2_dif_61");
        halfreal_head_ldspow2_31 = load_kernel_optional(program, "gf61_crt_halfreal_pack_ldspow2_dif_31");
        halfreal_tail_ldspow2_unpack61 = load_kernel_optional(program, "gf61_crt_halfreal_ditpow2_unpack_61");
        halfreal_tail_ldspow2_unpack31 = load_kernel_optional(program, "gf61_crt_halfreal_ditpow2_unpack_31");
        halfreal_unpack61 = load_kernel_optional(program, "gf61_crt_halfreal_unpack_unweight_61");
        halfreal_unpack31 = load_kernel_optional(program, "gf61_crt_halfreal_unpack_unweight_31");
        mixed_pack61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_61");
        mixed_pack31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_31");
        mixed_odd_dft61 = load_kernel_optional(program, "gf61_crt_mixed_odd_dft_61");
        mixed_odd_dft31 = load_kernel_optional(program, "gf61_crt_mixed_odd_dft_31");
        mixed_fwd_r2_61 = load_kernel_optional(program, "gf61_crt_mixed_ntt_stage_dif_radix2_61");
        mixed_inv_r2_61 = load_kernel_optional(program, "gf61_crt_mixed_ntt_stage_dit_radix2_61");
        mixed_fwd_r2_31 = load_kernel_optional(program, "gf61_crt_mixed_ntt_stage_dif_radix2_31");
        mixed_inv_r2_31 = load_kernel_optional(program, "gf61_crt_mixed_ntt_stage_dit_radix2_31");
        mixed_center61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_center_61");
        mixed_center31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_center_31");
        mixed_lds512_pair61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_61");
        mixed_lds512_pair31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_31");
        mixed_lds512_pair_1lds61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_61");
        mixed_lds512_pair_1lds_rega_f48_61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_rega_f48_61");
        mixed_lds512_pair_1lds_rega_twinline_f48_61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_rega_twinline_f48_61");
        mixed_lds512_pair_1lds_rega_twinline_f48_31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_rega_twinline_f48_31");
        mixed_lds512_pair_1lds31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_31");
        mixed_lds512_pair_1lds_f48_self61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_self_61");
        mixed_lds512_pair_1lds_f48_nonself61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_nonself_61");
        mixed_lds512_pair_1lds_f48_self31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_self_31");
        mixed_lds512_pair_1lds_f48_nonself31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_1lds_f48_nonself_31");
        mixed_lds1024_pair61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds1024_pair_61");
        mixed_lds1024_pair31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds1024_pair_31");
        mixed_lds_any_pair61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_61");
        mixed_lds_any_pair31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_31");
        mixed_lds_any_pair_1lds61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_1lds_61");
        mixed_lds_any_pair_1lds31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_1lds_31");
        mixed_lds_any_pair_1lds_f48_self61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_self_61");
        mixed_lds_any_pair_1lds_f48_nonself61 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_nonself_61");
        mixed_lds_any_pair_1lds_f48_self31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_self_31");
        mixed_lds_any_pair_1lds_f48_nonself31 = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_1lds_f48_nonself_31");
        mixed_lds_any_pair_both = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds_pair_any_61x31");
        mixed_lds512_pair_both = load_kernel_optional(program, "gf61_crt_mixed_halfreal_lds512_pair_61x31");
        mixed_fwd_lds512_both = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_61x31_512opt");
        mixed_inv_lds512_both = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_61x31_512opt");
        mixed_fwd_lds_any_both = load_kernel_optional(program, "gf61_crt_lds_stage_dif_pow2_61x31_any");
        mixed_inv_lds_any_both = load_kernel_optional(program, "gf61_crt_lds_stage_dit_pow2_61x31_any");
        mixed_unpack61 = load_kernel_optional(program, "gf61_crt_mixed_unpack_unweight_61");
        mixed_unpack31 = load_kernel_optional(program, "gf61_crt_mixed_unpack_unweight_31");
        mixed_pack_odd_fwd61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_61");
        mixed_pack_odd_fwd31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_31");
        mixed_pack_odd_fwd_tile7_61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile7_61");
        mixed_pack_odd_fwd_tile7_31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile7_31");
        mixed_pack_odd_fwd_tile14_61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_61");
        mixed_pack_odd_fwd_tile14_31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_31");
        mixed_pack_odd_fwd_tile14_shift_61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_61");
        mixed_pack_odd_fwd_tile14_shift_31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_31");
        mixed_pack_odd_fwd_tile14_shift_lmat_61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61");
        mixed_pack_odd_fwd_tile14_shift_lmat_31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_31");
        mixed_pack_odd_fwd_tile14_shift_lmat_both = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61x31");
        mixed_pack_odd_fwd_tile28_shift_lmat_both = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile28_shift_lmat_61x31");
        mixed_pack_odd_fwd_tile7_both = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_tile7_61x31");
        mixed_pack_odd_fwd_shift61 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_shift_61");
        mixed_pack_odd_fwd_shift31 = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_shift_31");
        mixed_pack_odd_fwd_both_shift = load_kernel_optional(program, "gf61_crt_mixed_pack_weight_odd_fwd_both_shift");
        mixed_odd_inv_unpack61 = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_unpack_unweight_61");
        mixed_odd_inv_unpack31 = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_unpack_unweight_31");
        mixed_odd_inv_precrt_coeffhi = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi");
        mixed_odd_inv_precrt_coeffhi_tile7 = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi_tile7");
        mixed_odd_inv_precrt_coeffhi_tile14 = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi_tile14");
        mixed_odd_inv_precrt_coeffhi_tile14_shift = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi_tile14_shift");
        mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat");
        mixed_odd9_inv_precrt_garner64_lmat = load_kernel_optional(program, "gf61_crt_mixed_odd9_inv_precrt_garner64_lmat");
        mixed_odd9_inv_precrt_garner9seg_lmat = load_kernel_optional(program, "gf61_crt_mixed_odd9_inv_precrt_garner9seg_lmat");
        mixed_odd9_inv_precrt_garner9seg30_pair_lmat = load_kernel_optional(program, "gf61_crt_mixed_odd9_inv_precrt_garner9seg30_pair_lmat");
        mixed_odd9_inv_precrt_garner9seg30_pair_smat = load_kernel_optional(program, "gf61_crt_mixed_odd9_inv_precrt_garner9seg30_pair_smat");
        mixed_odd_inv_precrt_coeffhi_outpar = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi_outpar");
        mixed_odd_inv_precrt_coeffhi_shift = load_kernel_optional(program, "gf61_crt_mixed_odd_inv_precrt_coeffhi_shift");
        mixed_residues_to_coeffhi = load_kernel_optional(program, "gf61_crt_mixed_residues_to_coeffhi");
    }
    ~CrtFusedKernels() {
        if (weight_first) clReleaseKernel(weight_first);
        if (fwd_r2) clReleaseKernel(fwd_r2);
        if (fwd_r4) clReleaseKernel(fwd_r4);
        if (fwd_r8) clReleaseKernel(fwd_r8);
        if (fwd_bridge512) clReleaseKernel(fwd_bridge512);
        if (fwd_bridge1024) clReleaseKernel(fwd_bridge1024);
        if (fwd_lds512) clReleaseKernel(fwd_lds512);
        if (inv_lds512) clReleaseKernel(inv_lds512);
        if (fwd_lds_stage) clReleaseKernel(fwd_lds_stage);
        if (inv_lds_stage) clReleaseKernel(inv_lds_stage);
        if (fwd_lds_stage_tile2) clReleaseKernel(fwd_lds_stage_tile2);
        if (inv_lds_stage_tile2) clReleaseKernel(inv_lds_stage_tile2);
        if (center256) clReleaseKernel(center256);
        if (center8_dualwave) clReleaseKernel(center8_dualwave);
        if (center16_dualwave) clReleaseKernel(center16_dualwave);
        if (center32_dualwave) clReleaseKernel(center32_dualwave);
        if (center64_dualwave) clReleaseKernel(center64_dualwave);
        if (center128_dualwave) clReleaseKernel(center128_dualwave);
        if (center256_dualwave) clReleaseKernel(center256_dualwave);
        if (center512_dualwave) clReleaseKernel(center512_dualwave);
        if (center512_reglds) clReleaseKernel(center512_reglds);
        if (center1024_dualwave) clReleaseKernel(center1024_dualwave);
        if (inv_bridge512) clReleaseKernel(inv_bridge512);
        if (inv_bridge1024) clReleaseKernel(inv_bridge1024);
        if (inv_r2) clReleaseKernel(inv_r2);
        if (inv_r4) clReleaseKernel(inv_r4);
        if (inv_r8) clReleaseKernel(inv_r8);
        if (last_unweight) clReleaseKernel(last_unweight);
        if (last_unweight16) clReleaseKernel(last_unweight16);
        if (weight_first61) clReleaseKernel(weight_first61);
        if (weight_first31) clReleaseKernel(weight_first31);
        if (weight_edge61) clReleaseKernel(weight_edge61);
        if (weight_edge31) clReleaseKernel(weight_edge31);
        if (fwd_r2_61) clReleaseKernel(fwd_r2_61);
        if (fwd_r4_61) clReleaseKernel(fwd_r4_61);
        if (fwd_r8_61) clReleaseKernel(fwd_r8_61);
        if (fwd_r8_61_wg128) clReleaseKernel(fwd_r8_61_wg128);
        if (fwd_lds_stage_61) clReleaseKernel(fwd_lds_stage_61);
        if (fwd_lds_stage_61_512opt) clReleaseKernel(fwd_lds_stage_61_512opt);
        if (fwd_lds_stage_61_tile4) clReleaseKernel(fwd_lds_stage_61_tile4);
        for (cl_kernel k : fwd_lds_stage_61_1lds) if (k) clReleaseKernel(k);
        for (cl_kernel k : inv_lds_stage_61_1lds) if (k) clReleaseKernel(k);
        if (inv_r2_61) clReleaseKernel(inv_r2_61);
        if (inv_r4_61) clReleaseKernel(inv_r4_61);
        if (inv_r8_61) clReleaseKernel(inv_r8_61);
        if (inv_lds_stage_61) clReleaseKernel(inv_lds_stage_61);
        if (inv_lds_stage_61_512opt) clReleaseKernel(inv_lds_stage_61_512opt);
        if (inv_lds_stage_61_tile4) clReleaseKernel(inv_lds_stage_61_tile4);
        if (center512_61) clReleaseKernel(center512_61);
        if (center512_61_opt) clReleaseKernel(center512_61_opt);
        if (last_unweight61) clReleaseKernel(last_unweight61);
        if (last_unweight_edge61) clReleaseKernel(last_unweight_edge61);
        if (fwd_r2_31) clReleaseKernel(fwd_r2_31);
        if (fwd_r4_31) clReleaseKernel(fwd_r4_31);
        if (fwd_r8_31) clReleaseKernel(fwd_r8_31);
        if (fwd_lds_stage_31) clReleaseKernel(fwd_lds_stage_31);
        if (fwd_lds_stage_31_512opt) clReleaseKernel(fwd_lds_stage_31_512opt);
        if (fwd_lds_stage_31_tile4) clReleaseKernel(fwd_lds_stage_31_tile4);
        for (cl_kernel k : fwd_lds_stage_31_1lds) if (k) clReleaseKernel(k);
        for (cl_kernel k : inv_lds_stage_31_1lds) if (k) clReleaseKernel(k);
        if (inv_r2_31) clReleaseKernel(inv_r2_31);
        if (inv_r4_31) clReleaseKernel(inv_r4_31);
        if (inv_r8_31) clReleaseKernel(inv_r8_31);
        if (inv_lds_stage_31) clReleaseKernel(inv_lds_stage_31);
        if (inv_lds_stage_31_512opt) clReleaseKernel(inv_lds_stage_31_512opt);
        if (inv_lds_stage_31_tile4) clReleaseKernel(inv_lds_stage_31_tile4);
        if (center512_31) clReleaseKernel(center512_31);
        if (center512_31_opt) clReleaseKernel(center512_31_opt);
        if (last_unweight31) clReleaseKernel(last_unweight31);
        if (last_unweight_edge31) clReleaseKernel(last_unweight_edge31);
        if (weight_fwd8_61) clReleaseKernel(weight_fwd8_61);
        if (weight_fwd8_31) clReleaseKernel(weight_fwd8_31);
        if (inv4_last_unweight61) clReleaseKernel(inv4_last_unweight61);
        if (inv4_last_unweight31) clReleaseKernel(inv4_last_unweight31);
        if (inv4_last_unweight16_61) clReleaseKernel(inv4_last_unweight16_61);
        if (inv4_last_unweight16_31) clReleaseKernel(inv4_last_unweight16_31);
        if (inv4_last_unweight16_crtcoeff) clReleaseKernel(inv4_last_unweight16_crtcoeff);
        if (halfreal_pack61) clReleaseKernel(halfreal_pack61);
        if (halfreal_pack31) clReleaseKernel(halfreal_pack31);
        if (halfreal_center61) clReleaseKernel(halfreal_center61);
        if (halfreal_center31) clReleaseKernel(halfreal_center31);
        if (halfreal_bitrev_swap61) clReleaseKernel(halfreal_bitrev_swap61);
        if (halfreal_bitrev_swap31) clReleaseKernel(halfreal_bitrev_swap31);
        if (halfreal_center512_pair61) clReleaseKernel(halfreal_center512_pair61);
        if (halfreal_center512_pair31) clReleaseKernel(halfreal_center512_pair31);
        if (halfreal_lds512_pair61) clReleaseKernel(halfreal_lds512_pair61);
        if (halfreal_lds512_pair31) clReleaseKernel(halfreal_lds512_pair31);
        if (halfreal_head_lds512_61) clReleaseKernel(halfreal_head_lds512_61);
        if (halfreal_head_lds512_31) clReleaseKernel(halfreal_head_lds512_31);
        if (halfreal_head_lds512_precrt) clReleaseKernel(halfreal_head_lds512_precrt);
        if (halfreal_tail_lds512_unpack61) clReleaseKernel(halfreal_tail_lds512_unpack61);
        if (halfreal_tail_lds512_unpack31) clReleaseKernel(halfreal_tail_lds512_unpack31);
        if (halfreal_tail_lds512_precrt) clReleaseKernel(halfreal_tail_lds512_precrt);
        if (halfreal_head_ldspow2_61) clReleaseKernel(halfreal_head_ldspow2_61);
        if (halfreal_head_ldspow2_31) clReleaseKernel(halfreal_head_ldspow2_31);
        if (halfreal_tail_ldspow2_unpack61) clReleaseKernel(halfreal_tail_ldspow2_unpack61);
        if (halfreal_tail_ldspow2_unpack31) clReleaseKernel(halfreal_tail_ldspow2_unpack31);
        if (halfreal_unpack61) clReleaseKernel(halfreal_unpack61);
        if (halfreal_unpack31) clReleaseKernel(halfreal_unpack31);
        if (mixed_pack61) clReleaseKernel(mixed_pack61);
        if (mixed_pack31) clReleaseKernel(mixed_pack31);
        if (mixed_odd_dft61) clReleaseKernel(mixed_odd_dft61);
        if (mixed_odd_dft31) clReleaseKernel(mixed_odd_dft31);
        if (mixed_fwd_r2_61) clReleaseKernel(mixed_fwd_r2_61);
        if (mixed_inv_r2_61) clReleaseKernel(mixed_inv_r2_61);
        if (mixed_fwd_r2_31) clReleaseKernel(mixed_fwd_r2_31);
        if (mixed_inv_r2_31) clReleaseKernel(mixed_inv_r2_31);
        if (mixed_center61) clReleaseKernel(mixed_center61);
        if (mixed_center31) clReleaseKernel(mixed_center31);
        if (mixed_lds512_pair61) clReleaseKernel(mixed_lds512_pair61);
        if (mixed_lds512_pair31) clReleaseKernel(mixed_lds512_pair31);
        if (mixed_lds512_pair_1lds61) clReleaseKernel(mixed_lds512_pair_1lds61);
        if (mixed_lds512_pair_1lds_rega_f48_61) clReleaseKernel(mixed_lds512_pair_1lds_rega_f48_61);
        if (mixed_lds512_pair_1lds_rega_twinline_f48_61) clReleaseKernel(mixed_lds512_pair_1lds_rega_twinline_f48_61);
        if (mixed_lds512_pair_1lds_rega_twinline_f48_31) clReleaseKernel(mixed_lds512_pair_1lds_rega_twinline_f48_31);
        if (mixed_lds512_pair_1lds31) clReleaseKernel(mixed_lds512_pair_1lds31);
        if (mixed_lds512_pair_1lds_f48_self61) clReleaseKernel(mixed_lds512_pair_1lds_f48_self61);
        if (mixed_lds512_pair_1lds_f48_nonself61) clReleaseKernel(mixed_lds512_pair_1lds_f48_nonself61);
        if (mixed_lds512_pair_1lds_f48_self31) clReleaseKernel(mixed_lds512_pair_1lds_f48_self31);
        if (mixed_lds512_pair_1lds_f48_nonself31) clReleaseKernel(mixed_lds512_pair_1lds_f48_nonself31);
        if (mixed_lds1024_pair61) clReleaseKernel(mixed_lds1024_pair61);
        if (mixed_lds1024_pair31) clReleaseKernel(mixed_lds1024_pair31);
        if (mixed_lds_any_pair61) clReleaseKernel(mixed_lds_any_pair61);
        if (mixed_lds_any_pair31) clReleaseKernel(mixed_lds_any_pair31);
        if (mixed_lds_any_pair_1lds61) clReleaseKernel(mixed_lds_any_pair_1lds61);
        if (mixed_lds_any_pair_1lds31) clReleaseKernel(mixed_lds_any_pair_1lds31);
        if (mixed_lds_any_pair_1lds_f48_self61) clReleaseKernel(mixed_lds_any_pair_1lds_f48_self61);
        if (mixed_lds_any_pair_1lds_f48_nonself61) clReleaseKernel(mixed_lds_any_pair_1lds_f48_nonself61);
        if (mixed_lds_any_pair_1lds_f48_self31) clReleaseKernel(mixed_lds_any_pair_1lds_f48_self31);
        if (mixed_lds_any_pair_1lds_f48_nonself31) clReleaseKernel(mixed_lds_any_pair_1lds_f48_nonself31);
        if (mixed_lds_any_pair_both) clReleaseKernel(mixed_lds_any_pair_both);
        if (mixed_lds512_pair_both) clReleaseKernel(mixed_lds512_pair_both);
        if (mixed_fwd_lds512_both) clReleaseKernel(mixed_fwd_lds512_both);
        if (mixed_inv_lds512_both) clReleaseKernel(mixed_inv_lds512_both);
        if (mixed_fwd_lds_any_both) clReleaseKernel(mixed_fwd_lds_any_both);
        if (mixed_inv_lds_any_both) clReleaseKernel(mixed_inv_lds_any_both);
        if (mixed_unpack61) clReleaseKernel(mixed_unpack61);
        if (mixed_unpack31) clReleaseKernel(mixed_unpack31);
        if (mixed_pack_odd_fwd61) clReleaseKernel(mixed_pack_odd_fwd61);
        if (mixed_pack_odd_fwd31) clReleaseKernel(mixed_pack_odd_fwd31);
        if (mixed_pack_odd_fwd_tile7_61) clReleaseKernel(mixed_pack_odd_fwd_tile7_61);
        if (mixed_pack_odd_fwd_tile7_31) clReleaseKernel(mixed_pack_odd_fwd_tile7_31);
        if (mixed_pack_odd_fwd_tile14_61) clReleaseKernel(mixed_pack_odd_fwd_tile14_61);
        if (mixed_pack_odd_fwd_tile14_31) clReleaseKernel(mixed_pack_odd_fwd_tile14_31);
        if (mixed_pack_odd_fwd_tile14_shift_61) clReleaseKernel(mixed_pack_odd_fwd_tile14_shift_61);
        if (mixed_pack_odd_fwd_tile14_shift_31) clReleaseKernel(mixed_pack_odd_fwd_tile14_shift_31);
        if (mixed_pack_odd_fwd_tile14_shift_lmat_61) clReleaseKernel(mixed_pack_odd_fwd_tile14_shift_lmat_61);
        if (mixed_pack_odd_fwd_tile14_shift_lmat_31) clReleaseKernel(mixed_pack_odd_fwd_tile14_shift_lmat_31);
        if (mixed_pack_odd_fwd_tile14_shift_lmat_both) clReleaseKernel(mixed_pack_odd_fwd_tile14_shift_lmat_both);
        if (mixed_pack_odd_fwd_tile28_shift_lmat_both) clReleaseKernel(mixed_pack_odd_fwd_tile28_shift_lmat_both);
        if (mixed_pack_odd_fwd_tile7_both) clReleaseKernel(mixed_pack_odd_fwd_tile7_both);
        if (mixed_pack_odd_fwd_shift61) clReleaseKernel(mixed_pack_odd_fwd_shift61);
        if (mixed_pack_odd_fwd_shift31) clReleaseKernel(mixed_pack_odd_fwd_shift31);
        if (mixed_pack_odd_fwd_both_shift) clReleaseKernel(mixed_pack_odd_fwd_both_shift);
        if (mixed_odd_inv_unpack61) clReleaseKernel(mixed_odd_inv_unpack61);
        if (mixed_odd_inv_unpack31) clReleaseKernel(mixed_odd_inv_unpack31);
        if (mixed_odd_inv_precrt_coeffhi) clReleaseKernel(mixed_odd_inv_precrt_coeffhi);
        if (mixed_odd_inv_precrt_coeffhi_tile7) clReleaseKernel(mixed_odd_inv_precrt_coeffhi_tile7);
        if (mixed_odd_inv_precrt_coeffhi_tile14) clReleaseKernel(mixed_odd_inv_precrt_coeffhi_tile14);
        if (mixed_odd_inv_precrt_coeffhi_tile14_shift) clReleaseKernel(mixed_odd_inv_precrt_coeffhi_tile14_shift);
        if (mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat) clReleaseKernel(mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat);
        if (mixed_odd9_inv_precrt_garner64_lmat) clReleaseKernel(mixed_odd9_inv_precrt_garner64_lmat);
        if (mixed_odd9_inv_precrt_garner9seg_lmat) clReleaseKernel(mixed_odd9_inv_precrt_garner9seg_lmat);
        if (mixed_odd9_inv_precrt_garner9seg30_pair_lmat) clReleaseKernel(mixed_odd9_inv_precrt_garner9seg30_pair_lmat);
        if (mixed_odd9_inv_precrt_garner9seg30_pair_smat) clReleaseKernel(mixed_odd9_inv_precrt_garner9seg30_pair_smat);
        if (mixed_odd_inv_precrt_coeffhi_outpar) clReleaseKernel(mixed_odd_inv_precrt_coeffhi_outpar);
        if (mixed_odd_inv_precrt_coeffhi_shift) clReleaseKernel(mixed_odd_inv_precrt_coeffhi_shift);
        if (mixed_residues_to_coeffhi) clReleaseKernel(mixed_residues_to_coeffhi);
    }
    bool mixed_odd_available() const {
        return mixed_pack61 && mixed_pack31 && mixed_odd_dft61 && mixed_odd_dft31 &&
               mixed_fwd_r2_61 && mixed_inv_r2_61 && mixed_fwd_r2_31 && mixed_inv_r2_31 &&
               mixed_center61 && mixed_center31 && mixed_unpack61 && mixed_unpack31;
    }

    bool mixed_odd_lds512_available() const {
        return mixed_lds512_pair61 && mixed_lds512_pair31 &&
               fwd_lds_stage_61 && fwd_lds_stage_31 &&
               inv_lds_stage_61 && inv_lds_stage_31;
    }
    bool mixed_odd_lds1024_available() const {
        return mixed_lds1024_pair61 && mixed_lds1024_pair31 &&
               fwd_lds_stage_61 && fwd_lds_stage_31 &&
               inv_lds_stage_61 && inv_lds_stage_31;
    }
    bool mixed_odd_lds_any_available() const {
        return mixed_lds_any_pair61 && mixed_lds_any_pair31 &&
               fwd_lds_stage_61 && fwd_lds_stage_31 &&
               inv_lds_stage_61 && inv_lds_stage_31;
    }
    bool mixed_odd_edge_fused_available() const {
        return mixed_pack_odd_fwd61 && mixed_pack_odd_fwd31 &&
               mixed_odd_inv_unpack61 && mixed_odd_inv_unpack31;
    }
    bool mixed_odd_shift_lut_available() const {
        return mixed_pack_odd_fwd_shift61 && mixed_pack_odd_fwd_shift31 && mixed_odd_inv_precrt_coeffhi_shift;
    }
    bool mixed_odd_pack_both_available() const { return mixed_pack_odd_fwd_both_shift != nullptr || mixed_pack_odd_fwd_tile7_both != nullptr || mixed_pack_odd_fwd_tile14_shift_lmat_both != nullptr || mixed_pack_odd_fwd_tile28_shift_lmat_both != nullptr; }
    bool mixed_odd_pack_tile7_available() const { return mixed_pack_odd_fwd_tile7_61 && mixed_pack_odd_fwd_tile7_31; }
    bool mixed_odd_pack_tile14_available() const { return mixed_pack_odd_fwd_tile14_61 && mixed_pack_odd_fwd_tile14_31; }
    bool mixed_odd_pack_tile14_shift_available() const { return mixed_pack_odd_fwd_tile14_shift_61 && mixed_pack_odd_fwd_tile14_shift_31; }
    bool mixed_odd_pack_tile14_shift_lmat_available() const { return mixed_pack_odd_fwd_tile14_shift_lmat_61 && mixed_pack_odd_fwd_tile14_shift_lmat_31; }
    bool mixed_odd_pack_tile14_shift_lmat_both_available() const { return mixed_pack_odd_fwd_tile14_shift_lmat_both != nullptr || mixed_pack_odd_fwd_tile28_shift_lmat_both != nullptr; }
    bool mixed_odd_pack_tile7_both_available() const { return mixed_pack_odd_fwd_tile7_both != nullptr; }
    bool mixed_odd_center_both_available() const { return mixed_lds_any_pair_both != nullptr || mixed_lds512_pair_both != nullptr; }
    bool mixed_odd_center_any_both_available() const { return mixed_lds_any_pair_both != nullptr; }
    bool mixed_odd_lds512_both_available() const { return mixed_fwd_lds512_both != nullptr && mixed_inv_lds512_both != nullptr; }
    bool mixed_odd_lds_any_both_available() const { return mixed_fwd_lds_any_both != nullptr && mixed_inv_lds_any_both != nullptr; }
    bool mixed_odd9_precrt_garner64_available() const { return mixed_odd9_inv_precrt_garner64_lmat != nullptr; }
    bool mixed_odd9_precrt_garner9seg_available() const { return mixed_odd9_inv_precrt_garner9seg_lmat != nullptr; }
    bool mixed_odd9_precrt_garner9seg30_pair_available() const { return mixed_odd9_inv_precrt_garner9seg30_pair_lmat != nullptr || mixed_odd9_inv_precrt_garner9seg30_pair_smat != nullptr; }

    bool mixed_odd_precrt_coeffhi_available() const {
        return mixed_odd_inv_precrt_coeffhi != nullptr || mixed_odd_inv_precrt_coeffhi_tile7 != nullptr ||
               mixed_odd_inv_precrt_coeffhi_tile14 != nullptr ||
               mixed_odd_inv_precrt_coeffhi_tile14_shift != nullptr ||
               mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat != nullptr ||
               mixed_odd_inv_precrt_coeffhi_outpar != nullptr || mixed_odd_inv_precrt_coeffhi_shift != nullptr;
    }
    bool available() const {
        return weight_first && fwd_r4 && fwd_bridge512 && center256 && inv_bridge512 && inv_r4 && last_unweight;
    }
    bool defused_fast_available() const {
        return weight_first61 && weight_first31 && weight_edge61 && weight_edge31 &&
               fwd_r2_61 && fwd_r4_61 && fwd_r8_61 && inv_r2_61 && inv_r4_61 && inv_r8_61 && center512_61 && last_unweight61 && last_unweight_edge61 &&
               fwd_r2_31 && fwd_r4_31 && fwd_r8_31 && inv_r2_31 && inv_r4_31 && inv_r8_31 && center512_31 && last_unweight31 && last_unweight_edge31;
    }
    bool defused_forward_edge_available() const {
        return weight_fwd8_61 && weight_fwd8_31;
    }
    bool defused_tail_edge_available() const {
        return inv4_last_unweight61 && inv4_last_unweight31;
    }
    bool boundary1024_available() const {
        return fwd_bridge1024 && inv_bridge1024;
    }
    bool last16_available() const {
        return last_unweight16 != nullptr;
    }
};

template <typename T>
static inline void set_karg(cl_kernel k, cl_uint& arg, const T& value, const char* what) {
    check(clSetKernelArg(k, arg++, sizeof(T), &value), what);
}
static inline void set_karg_mem(cl_kernel k, cl_uint& arg, const cl_mem& value, const char* what) {
    check(clSetKernelArg(k, arg++, sizeof(cl_mem), &value), what);
}


static inline cl_uint floor_pow2_leq(cl_uint x) {
    if (x == 0u) return 0u;
    cl_uint p = 1u;
    while ((p << 1u) != 0u && (p << 1u) <= x) p <<= 1u;
    return p;
}

static inline const char* crt_lds_stage_fwd_label(cl_uint radix) {
    switch (radix) {
        case 512u: return "crt_lds_stage512_forward";
        case 256u: return "crt_lds_stage256_forward";
        case 128u: return "crt_lds_stage128_forward";
        case 64u:  return "crt_lds_stage64_forward";
        case 32u:  return "crt_lds_stage32_forward";
        case 16u:  return "crt_lds_stage16_forward";
        case 8u:   return "crt_lds_stage8_forward";
        case 4u:   return "crt_lds_stage4_forward";
        default:   return "crt_lds_stage2_forward";
    }
}

static inline const char* crt_lds_stage_inv_label(cl_uint radix) {
    switch (radix) {
        case 512u: return "crt_lds_stage512_inverse";
        case 256u: return "crt_lds_stage256_inverse";
        case 128u: return "crt_lds_stage128_inverse";
        case 64u:  return "crt_lds_stage64_inverse";
        case 32u:  return "crt_lds_stage32_inverse";
        case 16u:  return "crt_lds_stage16_inverse";
        case 8u:   return "crt_lds_stage8_inverse";
        case 4u:   return "crt_lds_stage4_inverse";
        default:   return "crt_lds_stage2_inverse";
    }
}

static inline const char* crt_lds_stage_fwd_tile2_label(cl_uint radix) {
    switch (radix) {
        case 512u: return "crt_lds_stage512_forward_tile2";
        case 256u: return "crt_lds_stage256_forward_tile2";
        case 128u: return "crt_lds_stage128_forward_tile2";
        case 64u:  return "crt_lds_stage64_forward_tile2";
        case 32u:  return "crt_lds_stage32_forward_tile2";
        case 16u:  return "crt_lds_stage16_forward_tile2";
        default:   return "crt_lds_stage_forward_tile2";
    }
}

static inline const char* crt_lds_stage_inv_tile2_label(cl_uint radix) {
    switch (radix) {
        case 512u: return "crt_lds_stage512_inverse_tile2";
        case 256u: return "crt_lds_stage256_inverse_tile2";
        case 128u: return "crt_lds_stage128_inverse_tile2";
        case 64u:  return "crt_lds_stage64_inverse_tile2";
        case 32u:  return "crt_lds_stage32_inverse_tile2";
        case 16u:  return "crt_lds_stage16_inverse_tile2";
        default:   return "crt_lds_stage_inverse_tile2";
    }
}

struct CrtLdsPlanStep {
    cl_uint len_before;
    cl_uint radix;
    bool global_stage;
};


static CrtFusedKernels& single_halfreal_kernels(cl_program program) {
    static std::map<cl_program, CrtFusedKernels*> cache;
    auto& ptr = cache[program];
    if (!ptr) ptr = new CrtFusedKernels(program);
    return *ptr;
}

static bool enqueue_square_mod_single_halfreal(GpuPrp& gpu) {
    if (gpu.n < 2u || (gpu.n & 1u)) throw std::runtime_error("single halfreal mode requires an even transform length");
    CrtFusedKernels& fk = single_halfreal_kernels(gpu.program);
    if (!fk.halfreal_pack61 || !fk.halfreal_center61 || !fk.halfreal_unpack61 ||
        !fk.fwd_r2_61 || !fk.fwd_r4_61 || !fk.fwd_r8_61 || !fk.inv_r2_61 || !fk.inv_r4_61 || !fk.inv_r8_61) {
        throw std::runtime_error("single halfreal kernels are missing from the OpenCL program");
    }

    const cl_uint n = static_cast<cl_uint>(gpu.n);
    const cl_uint m = n >> 1;
    const cl_uint p = gpu.exponent_p;
    const cl_uint logm = gpu.log_n - 1u;
    const size_t local64 = 64u;
    const size_t g_m64 = round_up_size((size_t)m, local64);
    const size_t g_center64 = round_up_size((size_t)(m / 2u + 1u), local64);
    const cl_uint flags = crt_halfreal_effective_flags61();
    const bool flags_are_linear = ((flags & (1u | 16u)) == 0u);
    const bool linear_center_default = flags_are_linear;
    const bool linear_center = parse_bool_env("PRMERS_SINGLE_HALFREAL_LINEAR_CENTER", linear_center_default);
    if (linear_center && !flags_are_linear) throw std::runtime_error("PRMERS_SINGLE_HALFREAL_LINEAR_CENTER requires flags without digitrev/bitrev bits");
    if (linear_center && !fk.halfreal_bitrev_swap61) throw std::runtime_error("single halfreal linear-center requested but bitrev-swap kernel is missing");

    const bool fast512_req = parse_bool_env("PRMERS_SINGLE_HALFREAL_FAST512", true);
    const bool fast512_pair_req = parse_bool_env("PRMERS_SINGLE_HALFREAL_LDS_PAIR", true);
    const bool fast512 = fast512_req && fast512_pair_req && !linear_center &&
                         m >= 512u * 512u && ((m & 511u) == 0u) &&
                         (flags & 16u) && (flags & 32u) &&
                         fk.halfreal_head_lds512_61 && fk.halfreal_tail_lds512_unpack61 && fk.halfreal_lds512_pair61;
    const cl_uint fast512_head_base = fast512 ? (m >> 9) : m;

    auto pack = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_pack61, arg, gpu.bufDigits, "set single half pack digits");
        set_karg_mem(fk.halfreal_pack61, arg, gpu.bufField, "set single half pack a");
        set_karg(fk.halfreal_pack61, arg, n, "set single half pack n");
        set_karg(fk.halfreal_pack61, arg, p, "set single half pack p");
        set_karg(fk.halfreal_pack61, arg, gpu.lr2, "set single half pack lr2");
        enqueue_kernel(gpu, fk.halfreal_pack61, g_m64, &local64, "enqueue single half pack", "single_halfreal_pack_weight");
    };
    auto head_lds512_pack = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_head_lds512_61, arg, gpu.bufDigits, "set single half head LDS512 digits");
        set_karg_mem(fk.halfreal_head_lds512_61, arg, gpu.bufField, "set single half head LDS512 a");
        set_karg_mem(fk.halfreal_head_lds512_61, arg, gpu.bufTwFwd, "set single half head LDS512 tw");
        set_karg(fk.halfreal_head_lds512_61, arg, n, "set single half head LDS512 n");
        set_karg(fk.halfreal_head_lds512_61, arg, p, "set single half head LDS512 p");
        set_karg(fk.halfreal_head_lds512_61, arg, gpu.lr2, "set single half head LDS512 lr2");
        const size_t global = (size_t)(m >> 9) * local64;
        enqueue_kernel(gpu, fk.halfreal_head_lds512_61, global, &local64, "enqueue single half head LDS512", "single_halfreal_head_lds512");
    };
    auto center = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_center61, arg, gpu.bufField, "set single half center a");
        set_karg_mem(fk.halfreal_center61, arg, gpu.bufTwFwd, "set single half center twf");
        set_karg_mem(fk.halfreal_center61, arg, gpu.bufTwInv, "set single half center twi");
        set_karg(fk.halfreal_center61, arg, n, "set single half center n");
        set_karg(fk.halfreal_center61, arg, flags, "set single half center flags");
        enqueue_kernel(gpu, fk.halfreal_center61, g_center64, &local64, "enqueue single half center", "single_halfreal_center");
    };
    auto center_lds512_pair = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_lds512_pair61, arg, gpu.bufField, "set single half LDS512 pair a");
        set_karg_mem(fk.halfreal_lds512_pair61, arg, gpu.bufTwFwd, "set single half LDS512 pair twf");
        set_karg_mem(fk.halfreal_lds512_pair61, arg, gpu.bufTwInv, "set single half LDS512 pair twi");
        set_karg(fk.halfreal_lds512_pair61, arg, n, "set single half LDS512 pair n");
        set_karg(fk.halfreal_lds512_pair61, arg, flags, "set single half LDS512 pair flags");
        const cl_uint pair_blocks = (m >> 10) + 1u;
        const size_t global = (size_t)pair_blocks * local64;
        enqueue_kernel(gpu, fk.halfreal_lds512_pair61, global, &local64, "enqueue single half LDS512 pair", "single_halfreal_lds512_pair");
    };
    auto bitrev_swap = [&](const char* profile_name) {
        if (!linear_center) return;
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_bitrev_swap61, arg, gpu.bufField, "set single half bitrev swap a");
        set_karg(fk.halfreal_bitrev_swap61, arg, m, "set single half bitrev swap m");
        enqueue_kernel(gpu, fk.halfreal_bitrev_swap61, g_m64, &local64, "enqueue single half bitrev swap", profile_name);
    };
    auto unpack = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_unpack61, arg, gpu.bufField, "set single half unpack a");
        set_karg_mem(fk.halfreal_unpack61, arg, gpu.bufDigits, "set single half unpack digits");
        set_karg(fk.halfreal_unpack61, arg, n, "set single half unpack n");
        set_karg(fk.halfreal_unpack61, arg, p, "set single half unpack p");
        set_karg(fk.halfreal_unpack61, arg, gpu.lr2, "set single half unpack lr2");
        set_karg(fk.halfreal_unpack61, arg, logm, "set single half unpack logm");
        enqueue_kernel(gpu, fk.halfreal_unpack61, g_m64, &local64, "enqueue single half unpack", "single_halfreal_unpack_unweight");
    };
    auto tail_lds512_unpack = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.halfreal_tail_lds512_unpack61, arg, gpu.bufField, "set single half tail LDS512 a");
        set_karg_mem(fk.halfreal_tail_lds512_unpack61, arg, gpu.bufTwInv, "set single half tail LDS512 tw");
        set_karg_mem(fk.halfreal_tail_lds512_unpack61, arg, gpu.bufDigits, "set single half tail LDS512 digits");
        set_karg(fk.halfreal_tail_lds512_unpack61, arg, n, "set single half tail LDS512 n");
        set_karg(fk.halfreal_tail_lds512_unpack61, arg, p, "set single half tail LDS512 p");
        set_karg(fk.halfreal_tail_lds512_unpack61, arg, gpu.lr2, "set single half tail LDS512 lr2");
        set_karg(fk.halfreal_tail_lds512_unpack61, arg, logm, "set single half tail LDS512 logm");
        const size_t global = (size_t)(m >> 9) * local64;
        enqueue_kernel(gpu, fk.halfreal_tail_lds512_unpack61, global, &local64, "enqueue single half tail LDS512", "single_halfreal_tail_lds512");
    };
    auto fwd = [&](const StageInfo& st, uint32_t radix) {
        cl_kernel k = (radix == 8u) ? fk.fwd_r8_61 : (radix == 4u ? fk.fwd_r4_61 : fk.fwd_r2_61);
        cl_uint arg = 0;
        set_karg_mem(k, arg, gpu.bufField, "set single half fwd a");
        set_karg_mem(k, arg, gpu.bufTwFwd, "set single half fwd tw");
        set_karg(k, arg, m, "set single half fwd n");
        set_karg(k, arg, st.len, "set single half fwd len");
        enqueue_kernel(gpu, k, round_up_size(std::max<size_t>(1, m / radix), local64), &local64,
                       "enqueue single half fwd", radix == 8u ? "single_halfreal_fwd_radix8" : (radix == 4u ? "single_halfreal_fwd_radix4" : "single_halfreal_fwd_radix2"));
    };
    auto inv = [&](const StageInfo& st, uint32_t radix) {
        cl_kernel k = (radix == 8u) ? fk.inv_r8_61 : (radix == 4u ? fk.inv_r4_61 : fk.inv_r2_61);
        cl_uint arg = 0;
        set_karg_mem(k, arg, gpu.bufField, "set single half inv a");
        set_karg_mem(k, arg, gpu.bufTwInv, "set single half inv tw");
        set_karg(k, arg, m, "set single half inv n");
        set_karg(k, arg, st.len, "set single half inv len");
        enqueue_kernel(gpu, k, round_up_size(std::max<size_t>(1, m / radix), local64), &local64,
                       "enqueue single half inv", radix == 8u ? "single_halfreal_inv_radix8" : (radix == 4u ? "single_halfreal_inv_radix4" : "single_halfreal_inv_radix2"));
    };

    auto run_forward = [&]() {
        int i = (int)gpu.stages.size() - 1;
        while (i >= 0 && gpu.stages[(size_t)i].len > fast512_head_base) --i;
        while (i >= 0) {
            const uint32_t len = gpu.stages[(size_t)i].len;
            if (fast512 && len <= 512u) break;
            if (len >= 8u && i >= 2) { fwd(gpu.stages[(size_t)i], 8u); i -= 3; }
            else if (len >= 4u && i >= 1) { fwd(gpu.stages[(size_t)i], 4u); i -= 2; }
            else { fwd(gpu.stages[(size_t)i], 2u); --i; }
        }
    };
    auto run_inverse = [&]() {
        size_t i = 0;
        while (i < gpu.stages.size() && gpu.stages[i].len <= m) {
            const uint32_t len = gpu.stages[i].len;
            if (fast512 && len <= 512u) { ++i; continue; }
            if (len > fast512_head_base) break;
            if (len * 4u <= fast512_head_base && i + 2 < gpu.stages.size() && gpu.stages[i + 2].len <= fast512_head_base) { inv(gpu.stages[i], 8u); i += 3; }
            else if (len * 2u <= fast512_head_base && i + 1 < gpu.stages.size() && gpu.stages[i + 1].len <= fast512_head_base) { inv(gpu.stages[i], 4u); i += 2; }
            else { inv(gpu.stages[i], 2u); ++i; }
        }
    };

    if (fast512) head_lds512_pack();
    else pack();
    run_forward();
    if (fast512) center_lds512_pair();
    else {
        bitrev_swap("single_halfreal_bitrev_to_linear");
        center();
        bitrev_swap("single_halfreal_bitrev_from_linear");
    }
    run_inverse();
    if (fast512) tail_lds512_unpack();
    else unpack();
    return true;
}

static bool enqueue_square_mod_crt_mixed_odd(GpuPrp& g61, GpuPrp& g31, CrtFusedKernels& fk) {
    if (!fk.mixed_odd_available()) return false;
    if (g61.n != g31.n) return false;
    const cl_uint odd = g_crt_odd_radix;
    if (!(odd == 3u || odd == 9u)) return false;
    const cl_uint n = g61.n;
    if ((n % odd) != 0u) return false;
    const cl_uint pow2_n = n / odd;
    if (pow2_n < 4u || (pow2_n & (pow2_n - 1u)) != 0u) return false;
    const cl_uint row_m = pow2_n >> 1;
    const cl_uint storage = odd * row_m;
    const cl_uint p = g61.exponent_p;
    const cl_uint log_m = ([](cl_uint v){ cl_uint r=0; while (v > 1u) { v >>= 1; ++r; } return r; }(row_m));
    const size_t local64 = 64u;
    const size_t local128 = 128u;
    const size_t local256 = 256u;
    const size_t g_storage = round_up_size(static_cast<size_t>(storage), local64);
    const size_t g_row_m = round_up_size(static_cast<size_t>(row_m), local64);
    const size_t g_center_scalar = round_up_size(static_cast<size_t>(odd) * static_cast<size_t>(row_m), local64);
    const cl_uint flags61 = crt_halfreal_effective_flags61();
    const cl_uint flags31 = crt_halfreal_effective_flags31();

    
    const bool mixed_lds_disabled = parse_bool_env("PRMERS_CRT_MIXED_LDS_DISABLE", false) ||
                                    parse_bool_env("PRMERS_CRT_MIXED_LDS512_DISABLE", false);
    const bool mixed_lds512_opt = parse_bool_env("PRMERS_CRT_MIXED_LDS512_OPT", true);
    const char* env_center_global = std::getenv("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS");
    const char* env_center_legacy = std::getenv("PRMERS_CRT_MIXED_CENTER512_SINGLE_LDS");
    const bool center_global_set = (env_center_global && *env_center_global) || (env_center_legacy && *env_center_legacy);
    const bool center_global_val = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS",
                                                  parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SINGLE_LDS", true));
    const bool mixed_center_single_lds_61 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61",
                                                           center_global_set ? center_global_val : true);
    const bool mixed_center_single_lds_31 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31", false);
    const bool mixed_center_single_lds = mixed_center_single_lds_61 || mixed_center_single_lds_31;
    const bool mixed_center512_single_lds_61 = mixed_center_single_lds_61;
    const bool mixed_center512_single_lds_31 = mixed_center_single_lds_31;
    const char* env_center_split_f48_61 = std::getenv("PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61");
    const char* env_center_split_f48_31 = std::getenv("PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31");
    const bool mixed_center_split_f48_61 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61",
        env_center_split_f48_61 ? true : parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SPLIT_F48_61", false));
    const bool mixed_center_split_f48_31 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31",
        env_center_split_f48_31 ? true : parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SPLIT_F48_31", false));
    const bool mixed_center512_split_f48_61 = mixed_center_split_f48_61;
    const bool mixed_center512_split_f48_31 = mixed_center_split_f48_31;

    const char* env_stage_global = std::getenv("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS");
    const bool stage_global_set = env_stage_global && *env_stage_global;
    const bool stage_global_val = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS", false);
    const bool mixed_stage_single_lds_61 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61",
                                                          stage_global_set ? stage_global_val : true);
    const bool mixed_stage_single_lds_31 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31",
                                                          stage_global_set ? stage_global_val : false);
    const bool mixed_stage_single_lds = mixed_stage_single_lds_61 || mixed_stage_single_lds_31;
    const bool mixed_row_force_generic = (g_crt_mixed_row_core == "generic");
    const bool mixed_row_force_lds = (g_crt_mixed_row_core == "lds");
    const bool mixed_row_force_lds512 = (g_crt_mixed_row_core == "lds512");
    const bool mixed_row_force_lds1024 = (g_crt_mixed_row_core == "lds1024");
    const bool mixed_row_center_512 = (g_crt_center_chunk == 512u);
    const bool mixed_row_center_1024 = (g_crt_center_chunk == 1024u);
    const cl_uint mixed_row_center = mixed_row_force_lds512 ? 512u :
                                     (mixed_row_force_lds1024 ? 1024u : g_crt_center_chunk);
    const auto mixed_valid_pow2_lds = [](cl_uint v) {
        return v == 8u || v == 16u || v == 32u || v == 64u ||
               v == 128u || v == 256u || v == 512u || v == 1024u;
    };
    const bool mixed_row_center_valid = mixed_valid_pow2_lds(mixed_row_center);
    const bool mixed_row_force_any_lds = mixed_row_force_lds || mixed_row_force_lds512 || mixed_row_force_lds1024;
    const bool mixed_row_auto_lds = (g_crt_mixed_row_core == "auto") &&
                                    (mixed_row_center_512 || mixed_row_center_1024) &&
                                    g_crt_lds_stage >= 8u;
    if (mixed_row_force_lds512 && !mixed_row_center_512) {
        throw std::runtime_error("--crt-mixed-row-core lds512 requires --crt-local-square/--crt-mixed-row-center 512");
    }
    if (mixed_row_force_lds1024 && !mixed_row_center_1024) {
        throw std::runtime_error("--crt-mixed-row-core lds1024 requires --crt-local-square/--crt-mixed-row-center 1024");
    }
    if (mixed_row_force_any_lds && (!mixed_row_center_valid || row_m < mixed_row_center)) {
        throw std::runtime_error("forced mixed odd LDS row core requires --crt-mixed-row-center in 8..1024 and row_m >= center");
    }
    const bool use_mixed_lds_any = !mixed_lds_disabled && !mixed_row_force_generic &&
                                   mixed_row_center_valid && row_m >= mixed_row_center &&
                                   (mixed_row_force_any_lds || mixed_row_auto_lds) &&
                                   (flags61 & 16u) && (flags31 & 16u) &&
                                   fk.mixed_odd_lds_any_available();
    const bool use_mixed_lds1024 = use_mixed_lds_any && mixed_row_center == 1024u && fk.mixed_odd_lds1024_available();
    const bool use_mixed_lds512 = use_mixed_lds_any && !use_mixed_lds1024 && mixed_row_center == 512u && fk.mixed_odd_lds512_available();
    if (mixed_row_force_any_lds && !use_mixed_lds_any) {
        throw std::runtime_error("forced mixed odd LDS row core requested but mixed LDS any-center kernels are not available for this build/flags");
    }
    const bool use_mixed_fused_edges = parse_bool_env("PRMERS_CRT_MIXED_FUSE_ODD_EDGES", true) &&
                                       fk.mixed_odd_edge_fused_available();
    const bool use_mixed_precrt_coeffhi = parse_bool_env("PRMERS_CRT_MIXED_PRECRT_COEFFHI", true) &&
                                          use_mixed_fused_edges &&
                                          g31.bufDigits32 &&
                                          g61.bufWidthMask32 &&
                                          fk.mixed_odd_precrt_coeffhi_available();
    
    
    const bool use_mixed_shift_lut = parse_bool_env("PRMERS_CRT_MIXED_SHIFT_LUT", odd == 9u) &&
                                     fk.mixed_odd_shift_lut_available() &&
                                     g61.bufUnweightShift && g31.bufUnweightShift;
    const bool use_mixed_tile14_shift = use_mixed_shift_lut &&
                                        parse_bool_env("PRMERS_CRT_MIXED_TILE14_SHIFT", true) &&
                                        odd == 9u &&
                                        fk.mixed_odd_pack_tile14_shift_available() &&
                                        fk.mixed_odd_inv_precrt_coeffhi_tile14_shift != nullptr;
    const bool use_mixed_tile14_shift_lmat = use_mixed_tile14_shift &&
                                             parse_bool_env("PRMERS_CRT_MIXED_TILE14_LMAT", true) &&
                                             fk.mixed_odd_pack_tile14_shift_lmat_available() &&
                                             fk.mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat != nullptr;
    const bool use_mixed_precrt_garner64 = use_mixed_precrt_coeffhi && use_mixed_tile14_shift_lmat &&
                                           parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER", true) &&
                                           g61.bufWidthMask32 && g61.bufUnweightShift && g31.bufUnweightShift &&
                                           (fk.mixed_odd9_precrt_garner9seg30_pair_available() || fk.mixed_odd9_precrt_garner9seg_available() || fk.mixed_odd9_precrt_garner64_available());
    const bool allow_mixed_host_flush = parse_bool_env("PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH",
                                                       parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false));
    auto maybe_flush_mixed = [&](cl_command_queue q) {
        if (allow_mixed_host_flush) clFlush(q);
    };
    const bool use_mixed_tile14 = parse_bool_env("PRMERS_CRT_MIXED_TILE14", true) &&
                                  !use_mixed_shift_lut &&
                                  odd == 9u;
    const bool use_mixed_precrt_tile14_shift = use_mixed_tile14_shift &&
                                               use_mixed_precrt_coeffhi;
    const bool use_mixed_precrt_tile14 = !use_mixed_precrt_tile14_shift &&
                                         use_mixed_tile14 &&
                                         use_mixed_precrt_coeffhi &&
                                         fk.mixed_odd_inv_precrt_coeffhi_tile14 != nullptr;
    const bool use_mixed_precrt_tile7 = !use_mixed_precrt_tile14_shift &&
                                        !use_mixed_precrt_tile14 &&
                                        parse_bool_env("PRMERS_CRT_MIXED_PRECRT_TILE7", true) &&
                                        use_mixed_precrt_coeffhi &&
                                        !use_mixed_shift_lut &&
                                        odd == 9u &&
                                        fk.mixed_odd_inv_precrt_coeffhi_tile7 != nullptr;
    
    
    const bool use_mixed_precrt_outpar = parse_bool_env("PRMERS_CRT_MIXED_PRECRT_OUTPAR", false) &&
                                         use_mixed_precrt_coeffhi &&
                                         !use_mixed_precrt_tile14_shift &&
                                         !use_mixed_precrt_tile14 &&
                                         !use_mixed_precrt_tile7 &&
                                         !use_mixed_shift_lut &&
                                         fk.mixed_odd_inv_precrt_coeffhi_outpar != nullptr;
    
    
    const bool use_mixed_precrt_split = parse_bool_env("PRMERS_CRT_MIXED_PRECRT_SPLIT", false) &&
                                        use_mixed_fused_edges &&
                                        g31.bufDigits32 &&
                                        g61.bufWidthMask32 &&
                                        fk.mixed_residues_to_coeffhi != nullptr;
    const bool use_mixed_pack_tile14_shift = use_mixed_tile14_shift &&
                                             use_mixed_fused_edges;
    const bool use_mixed_pack_tile14 = !use_mixed_pack_tile14_shift &&
                                       use_mixed_tile14 &&
                                       use_mixed_fused_edges && !use_mixed_shift_lut &&
                                       fk.mixed_odd_pack_tile14_available();
    const bool use_mixed_pack_tile7 = !use_mixed_pack_tile14 &&
                                      parse_bool_env("PRMERS_CRT_MIXED_PACK_TILE7", true) &&
                                      use_mixed_fused_edges && !use_mixed_shift_lut && odd == 9u &&
                                      fk.mixed_odd_pack_tile7_available();
    const bool use_mixed_pack_tile14_shift_lmat_both = use_mixed_tile14_shift_lmat &&
                                                       use_mixed_fused_edges &&
                                                       parse_bool_env("PRMERS_CRT_MIXED_FUSE_PACK_BOTH", true) &&
                                                       fk.mixed_odd_pack_tile14_shift_lmat_both_available();
    const bool use_mixed_pack_both = parse_bool_env("PRMERS_CRT_MIXED_FUSE_PACK_BOTH", true) &&
                                     use_mixed_fused_edges &&
                                     (use_mixed_pack_tile14_shift_lmat_both ||
                                      (!use_mixed_pack_tile14_shift &&
                                       ((use_mixed_shift_lut && fk.mixed_pack_odd_fwd_both_shift != nullptr) ||
                                        (use_mixed_pack_tile7 && fk.mixed_odd_pack_tile7_both_available()))));
    const std::string fuse_both_mode = g_crt_mixed_row_fuse_both;
    const bool fuse_both_force = (fuse_both_mode == "force");
    const bool fuse_center_env = parse_bool_env("PRMERS_CRT_MIXED_FUSE_CENTER_BOTH", false);
    const bool fuse_stage_env = parse_bool_env("PRMERS_CRT_MIXED_FUSE_LDS_BOTH", false);
    const bool explicit_center_both = fuse_center_env || fuse_both_force ||
                                      fuse_both_mode == "center" || fuse_both_mode == "all";
    const bool explicit_stage_both = fuse_stage_env || fuse_both_force ||
                                     fuse_both_mode == "stage" || fuse_both_mode == "all";
    const bool center_fuse_overrides_single = parse_bool_env("PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS", false) ||
                                              parse_bool_env("PRMERS_CRT_MIXED_FUSE_OVERRIDES_SINGLE_LDS", false);
    const bool stage_fuse_overrides_single = parse_bool_env("PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS", false) ||
                                             parse_bool_env("PRMERS_CRT_MIXED_FUSE_OVERRIDES_SINGLE_LDS", false);
    const bool auto_center_both = (fuse_both_mode == "auto" && use_mixed_lds512 && !mixed_center_single_lds);
    const bool auto_stage_both = (fuse_both_mode == "auto" && use_mixed_lds512 && !mixed_stage_single_lds);
    const bool want_mixed_center_both_raw = explicit_center_both || auto_center_both;
    const bool want_mixed_stage_both_raw = explicit_stage_both || auto_stage_both;
    const bool want_mixed_center_both = use_mixed_lds_any && want_mixed_center_both_raw &&
                                        (!mixed_center_single_lds || center_fuse_overrides_single);
    const bool want_mixed_stage_both = use_mixed_lds_any && want_mixed_stage_both_raw &&
                                       (!mixed_stage_single_lds || stage_fuse_overrides_single);

    if (fuse_both_force && mixed_center_single_lds && !center_fuse_overrides_single) {
        throw std::runtime_error("--crt-mixed-row-fuse-both force conflicts with PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61/31; set PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS=1 or clear the center single-LDS flags");
    }
    if (fuse_both_force && mixed_stage_single_lds && !stage_fuse_overrides_single) {
        throw std::runtime_error("--crt-mixed-row-fuse-both force conflicts with PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61/31; set PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS=1 or clear the stage single-LDS flags");
    }

    // A fused 61x31 pair-center needs two pair buffers for each field.
    // The generic 1024 version would exceed the 48 KiB LDS limit on NVIDIA
    // (ptxas reports about 0xc010 bytes), so fused center is intentionally
    // capped at 512 and center=1024 falls back to separate GF61/GF31 kernels.
    const bool mixed_center_both_len_ok = (mixed_row_center <= 512u);
    const bool use_mixed_center_both = want_mixed_center_both && mixed_center_both_len_ok &&
        ((mixed_row_center == 512u && fk.mixed_lds512_pair_both) || fk.mixed_odd_center_any_both_available());
    const bool use_mixed_stage_both = want_mixed_stage_both &&
        (fk.mixed_odd_lds_any_both_available() || (use_mixed_lds512 && mixed_lds512_opt && fk.mixed_odd_lds512_both_available()));
    if (fuse_both_force && use_mixed_lds_any && (!use_mixed_center_both || !use_mixed_stage_both)) {
        if (want_mixed_center_both && !mixed_center_both_len_ok) {
            throw std::runtime_error("--crt-mixed-row-fuse-both force requested center=1024, but fused 61x31 pair-center is capped at 512 to stay under the NVIDIA 48 KiB LDS limit");
        }
        throw std::runtime_error("--crt-mixed-row-fuse-both force requested but 61x31 center/stage kernels are not available");
    }

    auto pack61 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_pack61, arg, g61.bufDigits, "set mixed pack61 digits");
        set_karg_mem(fk.mixed_pack61, arg, g61.bufField, "set mixed pack61 a");
        set_karg(fk.mixed_pack61, arg, n, "set mixed pack61 n");
        set_karg(fk.mixed_pack61, arg, p, "set mixed pack61 p");
        set_karg(fk.mixed_pack61, arg, g61.lr2, "set mixed pack61 lr2");
        set_karg(fk.mixed_pack61, arg, odd, "set mixed pack61 odd");
        set_karg(fk.mixed_pack61, arg, pow2_n, "set mixed pack61 pow2_n");
        enqueue_kernel(g61, fk.mixed_pack61, g_storage, &local64, "enqueue mixed pack61", "crt_mixed_pack_weight_61");
    };
    auto pack31 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_pack31, arg, g61.bufDigits, "set mixed pack31 digits");
        set_karg_mem(fk.mixed_pack31, arg, g31.bufField, "set mixed pack31 a");
        set_karg(fk.mixed_pack31, arg, n, "set mixed pack31 n");
        set_karg(fk.mixed_pack31, arg, p, "set mixed pack31 p");
        set_karg(fk.mixed_pack31, arg, g31.lr2, "set mixed pack31 lr2");
        set_karg(fk.mixed_pack31, arg, odd, "set mixed pack31 odd");
        set_karg(fk.mixed_pack31, arg, pow2_n, "set mixed pack31 pow2_n");
        enqueue_kernel(g31, fk.mixed_pack31, g_storage, &local64, "enqueue mixed pack31", "crt_mixed_pack_weight_31");
    };
    auto pack_odd_fwd_both = [&]() {
        if (use_mixed_pack_tile14_shift_lmat_both) {
            const bool use_pack_tile28 = (odd == 9u) && parse_bool_env("PRMERS_CRT_MIXED_PACK_TILE28_61X31", true) &&
                                         fk.mixed_pack_odd_fwd_tile28_shift_lmat_both;
            cl_kernel kpack = use_pack_tile28 ? fk.mixed_pack_odd_fwd_tile28_shift_lmat_both : fk.mixed_pack_odd_fwd_tile14_shift_lmat_both;
            const size_t* local_pack = use_pack_tile28 ? &local256 : &local128;
            const size_t tile = use_pack_tile28 ? 28u : 14u;
            const size_t wg = use_pack_tile28 ? 256u : 128u;
            const char* label = use_pack_tile28 ?
                "crt_mixed_pack_weight_odd_fwd_tile28_shift_lmat_61x31" :
                "crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61x31";
            cl_uint arg = 0;
            set_karg_mem(kpack, arg, g61.bufDigits, "set mixed pack+odd lmat both digits");
            set_karg_mem(kpack, arg, g61.bufField, "set mixed pack+odd lmat both a61");
            set_karg_mem(kpack, arg, g31.bufField, "set mixed pack+odd lmat both a31");
            set_karg_mem(kpack, arg, g61.bufOddFwd, "set mixed pack+odd lmat both mat61");
            set_karg_mem(kpack, arg, g31.bufOddFwd, "set mixed pack+odd lmat both mat31");
            set_karg_mem(kpack, arg, g61.bufUnweightShift, "set mixed pack+odd lmat both shift61");
            set_karg_mem(kpack, arg, g31.bufUnweightShift, "set mixed pack+odd lmat both shift31");
            set_karg(kpack, arg, odd, "set mixed pack+odd lmat both odd");
            set_karg(kpack, arg, pow2_n, "set mixed pack+odd lmat both pow2_n");
            const size_t head_global = round_up_size(((static_cast<size_t>(row_m) + tile - 1u) / tile) * wg, *local_pack);
            enqueue_kernel(g61, kpack, head_global, local_pack,
                           "enqueue mixed fused pack+odd lmat both", label);
        } else if (use_mixed_pack_tile7 && fk.mixed_pack_odd_fwd_tile7_both) {
            cl_uint arg = 0;
            set_karg_mem(fk.mixed_pack_odd_fwd_tile7_both, arg, g61.bufDigits, "set mixed pack+odd tile7 both digits");
            set_karg_mem(fk.mixed_pack_odd_fwd_tile7_both, arg, g61.bufField, "set mixed pack+odd tile7 both a61");
            set_karg_mem(fk.mixed_pack_odd_fwd_tile7_both, arg, g31.bufField, "set mixed pack+odd tile7 both a31");
            set_karg_mem(fk.mixed_pack_odd_fwd_tile7_both, arg, g61.bufOddFwd, "set mixed pack+odd tile7 both mat61");
            set_karg_mem(fk.mixed_pack_odd_fwd_tile7_both, arg, g31.bufOddFwd, "set mixed pack+odd tile7 both mat31");
            set_karg(fk.mixed_pack_odd_fwd_tile7_both, arg, n, "set mixed pack+odd tile7 both n");
            set_karg(fk.mixed_pack_odd_fwd_tile7_both, arg, p, "set mixed pack+odd tile7 both p");
            set_karg(fk.mixed_pack_odd_fwd_tile7_both, arg, g61.lr2, "set mixed pack+odd tile7 both lr2_61");
            set_karg(fk.mixed_pack_odd_fwd_tile7_both, arg, g31.lr2, "set mixed pack+odd tile7 both lr2_31");
            set_karg(fk.mixed_pack_odd_fwd_tile7_both, arg, odd, "set mixed pack+odd tile7 both odd");
            set_karg(fk.mixed_pack_odd_fwd_tile7_both, arg, pow2_n, "set mixed pack+odd tile7 both pow2_n");
            const size_t head_global = round_up_size(((static_cast<size_t>(row_m) + 6u) / 7u) * 64u, local64);
            enqueue_kernel(g61, fk.mixed_pack_odd_fwd_tile7_both, head_global, &local64,
                           "enqueue mixed fused pack+odd tile7 both", "crt_mixed_pack_weight_odd_fwd_tile7_61x31");
        } else {
            cl_uint arg = 0;
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g61.bufDigits, "set mixed pack+odd both digits");
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g61.bufField, "set mixed pack+odd both a61");
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g31.bufField, "set mixed pack+odd both a31");
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g61.bufOddFwd, "set mixed pack+odd both mat61");
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g31.bufOddFwd, "set mixed pack+odd both mat31");
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g61.bufUnweightShift, "set mixed pack+odd both shift61");
            set_karg_mem(fk.mixed_pack_odd_fwd_both_shift, arg, g31.bufUnweightShift, "set mixed pack+odd both shift31");
            set_karg(fk.mixed_pack_odd_fwd_both_shift, arg, odd, "set mixed pack+odd both odd");
            set_karg(fk.mixed_pack_odd_fwd_both_shift, arg, pow2_n, "set mixed pack+odd both pow2_n");
            enqueue_kernel(g61, fk.mixed_pack_odd_fwd_both_shift, g_row_m, &local64,
                           "enqueue mixed fused pack+odd both", "crt_mixed_pack_weight_odd_fwd_both_shift");
        }
        cl_event both_ready = enqueue_queue_marker(g61, "mixed odd pack both ready");
        set_pending_wait_event(g31, both_ready);
        if (g61.queue != g31.queue) maybe_flush_mixed(g61.queue);
    };

    auto pack_odd_fwd61 = [&]() {
        cl_kernel kpack = use_mixed_tile14_shift_lmat ? fk.mixed_pack_odd_fwd_tile14_shift_lmat_61 :
                          (use_mixed_pack_tile14_shift ? fk.mixed_pack_odd_fwd_tile14_shift_61 :
                          (use_mixed_pack_tile14 ? fk.mixed_pack_odd_fwd_tile14_61 :
                          (use_mixed_pack_tile7 ? fk.mixed_pack_odd_fwd_tile7_61 :
                          (use_mixed_shift_lut ? fk.mixed_pack_odd_fwd_shift61 : fk.mixed_pack_odd_fwd61))));
        cl_uint arg = 0;
        set_karg_mem(kpack, arg, g61.bufDigits, "set mixed fused pack+odd61 digits");
        set_karg_mem(kpack, arg, g61.bufField, "set mixed fused pack+odd61 a");
        set_karg_mem(kpack, arg, g61.bufOddFwd, "set mixed fused pack+odd61 mat");
        if (use_mixed_shift_lut) {
            set_karg_mem(kpack, arg, g61.bufUnweightShift, "set mixed fused pack+odd61 shift");
            set_karg(kpack, arg, odd, "set mixed fused pack+odd61 odd");
            set_karg(kpack, arg, pow2_n, "set mixed fused pack+odd61 pow2_n");
        } else {
            set_karg(kpack, arg, n, "set mixed fused pack+odd61 n");
            set_karg(kpack, arg, p, "set mixed fused pack+odd61 p");
            set_karg(kpack, arg, g61.lr2, "set mixed fused pack+odd61 lr2");
            set_karg(kpack, arg, odd, "set mixed fused pack+odd61 odd");
            set_karg(kpack, arg, pow2_n, "set mixed fused pack+odd61 pow2_n");
        }
        const size_t head_global = (use_mixed_pack_tile14 || use_mixed_pack_tile14_shift) ?
            round_up_size(((static_cast<size_t>(row_m) + 13u) / 14u) * 128u, local128) :
            (use_mixed_pack_tile7 ?
            round_up_size(((static_cast<size_t>(row_m) + 6u) / 7u) * 64u, local64) : g_row_m);
        const char* head_name = use_mixed_tile14_shift_lmat ? "crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_61" :
                                (use_mixed_pack_tile14_shift ? "crt_mixed_pack_weight_odd_fwd_tile14_shift_61" :
                                (use_mixed_pack_tile14 ? "crt_mixed_pack_weight_odd_fwd_tile14_61" :
                                (use_mixed_pack_tile7 ? "crt_mixed_pack_weight_odd_fwd_tile7_61" :
                                (use_mixed_shift_lut ? "crt_mixed_pack_weight_odd_fwd_shift_61" : "crt_mixed_pack_weight_odd_fwd_61"))));
        enqueue_kernel(g61, kpack, head_global, (use_mixed_pack_tile14 || use_mixed_pack_tile14_shift) ? &local128 : &local64, "enqueue mixed fused pack+odd61", head_name);
    };
    auto pack_odd_fwd31 = [&]() {
        cl_kernel kpack = use_mixed_tile14_shift_lmat ? fk.mixed_pack_odd_fwd_tile14_shift_lmat_31 :
                          (use_mixed_pack_tile14_shift ? fk.mixed_pack_odd_fwd_tile14_shift_31 :
                          (use_mixed_pack_tile14 ? fk.mixed_pack_odd_fwd_tile14_31 :
                          (use_mixed_pack_tile7 ? fk.mixed_pack_odd_fwd_tile7_31 :
                          (use_mixed_shift_lut ? fk.mixed_pack_odd_fwd_shift31 : fk.mixed_pack_odd_fwd31))));
        cl_uint arg = 0;
        set_karg_mem(kpack, arg, g31.crtInputDigits ? g31.crtInputDigits : g61.bufDigits, "set mixed fused pack+odd31 digits");
        set_karg_mem(kpack, arg, g31.bufField, "set mixed fused pack+odd31 a");
        set_karg_mem(kpack, arg, g31.bufOddFwd, "set mixed fused pack+odd31 mat");
        if (use_mixed_shift_lut) {
            set_karg_mem(kpack, arg, g31.bufUnweightShift, "set mixed fused pack+odd31 shift");
            set_karg(kpack, arg, odd, "set mixed fused pack+odd31 odd");
            set_karg(kpack, arg, pow2_n, "set mixed fused pack+odd31 pow2_n");
        } else {
            set_karg(kpack, arg, n, "set mixed fused pack+odd31 n");
            set_karg(kpack, arg, p, "set mixed fused pack+odd31 p");
            set_karg(kpack, arg, g31.lr2, "set mixed fused pack+odd31 lr2");
            set_karg(kpack, arg, odd, "set mixed fused pack+odd31 odd");
            set_karg(kpack, arg, pow2_n, "set mixed fused pack+odd31 pow2_n");
        }
        const size_t head_global = (use_mixed_pack_tile14 || use_mixed_pack_tile14_shift) ?
            round_up_size(((static_cast<size_t>(row_m) + 13u) / 14u) * 128u, local128) :
            (use_mixed_pack_tile7 ?
            round_up_size(((static_cast<size_t>(row_m) + 6u) / 7u) * 64u, local64) : g_row_m);
        const char* head_name = use_mixed_tile14_shift_lmat ? "crt_mixed_pack_weight_odd_fwd_tile14_shift_lmat_31" :
                                (use_mixed_pack_tile14_shift ? "crt_mixed_pack_weight_odd_fwd_tile14_shift_31" :
                                (use_mixed_pack_tile14 ? "crt_mixed_pack_weight_odd_fwd_tile14_31" :
                                (use_mixed_pack_tile7 ? "crt_mixed_pack_weight_odd_fwd_tile7_31" :
                                (use_mixed_shift_lut ? "crt_mixed_pack_weight_odd_fwd_shift_31" : "crt_mixed_pack_weight_odd_fwd_31"))));
        enqueue_kernel(g31, kpack, head_global, (use_mixed_pack_tile14 || use_mixed_pack_tile14_shift) ? &local128 : &local64, "enqueue mixed fused pack+odd31", head_name);
    };

    auto odd_dft61 = [&](cl_mem mat, const char* label) {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_odd_dft61, arg, g61.bufField, "set mixed odd61 a");
        set_karg_mem(fk.mixed_odd_dft61, arg, g61.bufOddScratch, "set mixed odd61 scratch");
        set_karg_mem(fk.mixed_odd_dft61, arg, mat, "set mixed odd61 matrix");
        set_karg(fk.mixed_odd_dft61, arg, odd, "set mixed odd61 odd");
        set_karg(fk.mixed_odd_dft61, arg, row_m, "set mixed odd61 row_m");
        enqueue_kernel(g61, fk.mixed_odd_dft61, g_row_m, &local64, "enqueue mixed odd61", label);
    };
    auto odd_dft31 = [&](cl_mem mat, const char* label) {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_odd_dft31, arg, g31.bufField, "set mixed odd31 a");
        set_karg_mem(fk.mixed_odd_dft31, arg, g31.bufOddScratch, "set mixed odd31 scratch");
        set_karg_mem(fk.mixed_odd_dft31, arg, mat, "set mixed odd31 matrix");
        set_karg(fk.mixed_odd_dft31, arg, odd, "set mixed odd31 odd");
        set_karg(fk.mixed_odd_dft31, arg, row_m, "set mixed odd31 row_m");
        enqueue_kernel(g31, fk.mixed_odd_dft31, g_row_m, &local64, "enqueue mixed odd31", label);
    };

    
    auto fwd61 = [&](const StageInfo& st) {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_fwd_r2_61, arg, g61.bufField, "set mixed fwd61 a");
        set_karg_mem(fk.mixed_fwd_r2_61, arg, g61.bufTwFwd, "set mixed fwd61 tw");
        set_karg(fk.mixed_fwd_r2_61, arg, row_m, "set mixed fwd61 row_m");
        set_karg(fk.mixed_fwd_r2_61, arg, odd, "set mixed fwd61 odd");
        set_karg(fk.mixed_fwd_r2_61, arg, st.offset, "set mixed fwd61 tw_offset");
        set_karg(fk.mixed_fwd_r2_61, arg, st.len, "set mixed fwd61 len");
        set_karg(fk.mixed_fwd_r2_61, arg, st.half_len, "set mixed fwd61 half_len");
        enqueue_kernel(g61, fk.mixed_fwd_r2_61, round_up_size(static_cast<size_t>(odd) * static_cast<size_t>(row_m >> 1), local64), &local64, "enqueue mixed fwd61", "crt_mixed_fwd_radix2_61");
    };
    auto fwd31 = [&](const StageInfo& st) {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_fwd_r2_31, arg, g31.bufField, "set mixed fwd31 a");
        set_karg_mem(fk.mixed_fwd_r2_31, arg, g31.bufTwFwd, "set mixed fwd31 tw");
        set_karg(fk.mixed_fwd_r2_31, arg, row_m, "set mixed fwd31 row_m");
        set_karg(fk.mixed_fwd_r2_31, arg, odd, "set mixed fwd31 odd");
        set_karg(fk.mixed_fwd_r2_31, arg, st.offset, "set mixed fwd31 tw_offset");
        set_karg(fk.mixed_fwd_r2_31, arg, st.len, "set mixed fwd31 len");
        set_karg(fk.mixed_fwd_r2_31, arg, st.half_len, "set mixed fwd31 half_len");
        enqueue_kernel(g31, fk.mixed_fwd_r2_31, round_up_size(static_cast<size_t>(odd) * static_cast<size_t>(row_m >> 1), local64), &local64, "enqueue mixed fwd31", "crt_mixed_fwd_radix2_31");
    };
    auto inv61 = [&](const StageInfo& st) {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_inv_r2_61, arg, g61.bufField, "set mixed inv61 a");
        set_karg_mem(fk.mixed_inv_r2_61, arg, g61.bufTwInv, "set mixed inv61 tw");
        set_karg(fk.mixed_inv_r2_61, arg, row_m, "set mixed inv61 row_m");
        set_karg(fk.mixed_inv_r2_61, arg, odd, "set mixed inv61 odd");
        set_karg(fk.mixed_inv_r2_61, arg, st.offset, "set mixed inv61 tw_offset");
        set_karg(fk.mixed_inv_r2_61, arg, st.len, "set mixed inv61 len");
        set_karg(fk.mixed_inv_r2_61, arg, st.half_len, "set mixed inv61 half_len");
        enqueue_kernel(g61, fk.mixed_inv_r2_61, round_up_size(static_cast<size_t>(odd) * static_cast<size_t>(row_m >> 1), local64), &local64, "enqueue mixed inv61", "crt_mixed_inv_radix2_61");
    };
    auto inv31 = [&](const StageInfo& st) {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_inv_r2_31, arg, g31.bufField, "set mixed inv31 a");
        set_karg_mem(fk.mixed_inv_r2_31, arg, g31.bufTwInv, "set mixed inv31 tw");
        set_karg(fk.mixed_inv_r2_31, arg, row_m, "set mixed inv31 row_m");
        set_karg(fk.mixed_inv_r2_31, arg, odd, "set mixed inv31 odd");
        set_karg(fk.mixed_inv_r2_31, arg, st.offset, "set mixed inv31 tw_offset");
        set_karg(fk.mixed_inv_r2_31, arg, st.len, "set mixed inv31 len");
        set_karg(fk.mixed_inv_r2_31, arg, st.half_len, "set mixed inv31 half_len");
        enqueue_kernel(g31, fk.mixed_inv_r2_31, round_up_size(static_cast<size_t>(odd) * static_cast<size_t>(row_m >> 1), local64), &local64, "enqueue mixed inv31", "crt_mixed_inv_radix2_31");
    };

    auto mixed_lds_label = [](const char* dir, cl_uint radix, const char* mod, bool opt512, bool one_lds) -> std::string {
        std::string s = "crt_mixed_lds" + std::to_string(radix) + "_" + dir;
        if (opt512) s += "_opt";
        else if (one_lds) s += "_1lds";
        s += "_";
        s += mod;
        return s;
    };
    auto lds_radix_slot = [](cl_uint radix) -> int {
        switch (radix) {
            case 8u: return 0; case 16u: return 1; case 32u: return 2; case 64u: return 3;
            case 128u: return 4; case 256u: return 5; case 512u: return 6; case 1024u: return 7;
            default: return -1;
        }
    };
    auto pick_stage_1lds = [&](const std::array<cl_kernel, 8>& ks, cl_uint radix) -> cl_kernel {
        const int idx = lds_radix_slot(radix);
        return (idx >= 0) ? ks[static_cast<size_t>(idx)] : nullptr;
    };

    auto lds_fwd61 = [&](cl_uint len, cl_uint radix) {
        const bool opt512 = (radix == 512u && mixed_lds512_opt && len == radix && fk.fwd_lds_stage_61_512opt);
        const cl_kernel k1 = mixed_stage_single_lds_61 ? pick_stage_1lds(fk.fwd_lds_stage_61_1lds, radix) : nullptr;
        const bool one_lds = (!opt512 && k1 != nullptr);
        cl_kernel k = opt512 ? fk.fwd_lds_stage_61_512opt : (one_lds ? k1 : fk.fwd_lds_stage_61);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set mixed lds fwd61 a");
        set_karg_mem(k, arg, g61.bufTwFwd, "set mixed lds fwd61 tw");
        set_karg(k, arg, storage, "set mixed lds fwd61 storage");
        set_karg(k, arg, len, "set mixed lds fwd61 len");
        set_karg(k, arg, radix, "set mixed lds fwd61 radix");
        const size_t groups = (static_cast<size_t>(storage) / static_cast<size_t>(len)) * static_cast<size_t>(len / radix);
        const std::string label = mixed_lds_label("forward", radix, "61", opt512, one_lds);
        enqueue_kernel(g61, k, groups * local64, &local64, "enqueue mixed LDS fwd61", label.c_str());
    };
    auto lds_fwd31 = [&](cl_uint len, cl_uint radix) {
        const bool opt512 = (radix == 512u && mixed_lds512_opt && len == radix && fk.fwd_lds_stage_31_512opt);
        const cl_kernel k1 = mixed_stage_single_lds_31 ? pick_stage_1lds(fk.fwd_lds_stage_31_1lds, radix) : nullptr;
        const bool one_lds = (!opt512 && k1 != nullptr);
        cl_kernel k = opt512 ? fk.fwd_lds_stage_31_512opt : (one_lds ? k1 : fk.fwd_lds_stage_31);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g31.bufField, "set mixed lds fwd31 a");
        set_karg_mem(k, arg, g31.bufTwFwd, "set mixed lds fwd31 tw");
        set_karg(k, arg, storage, "set mixed lds fwd31 storage");
        set_karg(k, arg, len, "set mixed lds fwd31 len");
        set_karg(k, arg, radix, "set mixed lds fwd31 radix");
        const size_t groups = (static_cast<size_t>(storage) / static_cast<size_t>(len)) * static_cast<size_t>(len / radix);
        const std::string label = mixed_lds_label("forward", radix, "31", opt512, one_lds);
        enqueue_kernel(g31, k, groups * local64, &local64, "enqueue mixed LDS fwd31", label.c_str());
    };
    auto lds_inv61 = [&](cl_uint base_len, cl_uint radix) {
        const bool opt512 = (radix == 512u && mixed_lds512_opt && base_len == 1u && fk.inv_lds_stage_61_512opt);
        const cl_kernel k1 = mixed_stage_single_lds_61 ? pick_stage_1lds(fk.inv_lds_stage_61_1lds, radix) : nullptr;
        const bool one_lds = (!opt512 && k1 != nullptr);
        cl_kernel k = opt512 ? fk.inv_lds_stage_61_512opt : (one_lds ? k1 : fk.inv_lds_stage_61);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set mixed lds inv61 a");
        set_karg_mem(k, arg, g61.bufTwInv, "set mixed lds inv61 tw");
        set_karg(k, arg, storage, "set mixed lds inv61 storage");
        set_karg(k, arg, base_len, "set mixed lds inv61 base_len");
        set_karg(k, arg, radix, "set mixed lds inv61 radix");
        const size_t groups = (static_cast<size_t>(storage) / static_cast<size_t>(base_len * radix)) * static_cast<size_t>(base_len);
        const std::string label = mixed_lds_label("inverse", radix, "61", opt512, one_lds);
        enqueue_kernel(g61, k, groups * local64, &local64, "enqueue mixed LDS inv61", label.c_str());
    };
    auto lds_inv31 = [&](cl_uint base_len, cl_uint radix) {
        const bool opt512 = (radix == 512u && mixed_lds512_opt && base_len == 1u && fk.inv_lds_stage_31_512opt);
        const cl_kernel k1 = mixed_stage_single_lds_31 ? pick_stage_1lds(fk.inv_lds_stage_31_1lds, radix) : nullptr;
        const bool one_lds = (!opt512 && k1 != nullptr);
        cl_kernel k = opt512 ? fk.inv_lds_stage_31_512opt : (one_lds ? k1 : fk.inv_lds_stage_31);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g31.bufField, "set mixed lds inv31 a");
        set_karg_mem(k, arg, g31.bufTwInv, "set mixed lds inv31 tw");
        set_karg(k, arg, storage, "set mixed lds inv31 storage");
        set_karg(k, arg, base_len, "set mixed lds inv31 base_len");
        set_karg(k, arg, radix, "set mixed lds inv31 radix");
        const size_t groups = (static_cast<size_t>(storage) / static_cast<size_t>(base_len * radix)) * static_cast<size_t>(base_len);
        const std::string label = mixed_lds_label("inverse", radix, "31", opt512, one_lds);
        enqueue_kernel(g31, k, groups * local64, &local64, "enqueue mixed LDS inv31", label.c_str());
    };
    auto sync_g31_to_g61 = [&](const char* label) {
        cl_event ev = enqueue_queue_marker(g31, label);
        set_pending_wait_event(g61, ev);
        if (g31.queue != g61.queue) maybe_flush_mixed(g31.queue);
    };
    auto sync_g61_to_g31 = [&](const char* label) {
        cl_event ev = enqueue_queue_marker(g61, label);
        set_pending_wait_event(g31, ev);
        if (g61.queue != g31.queue) maybe_flush_mixed(g61.queue);
    };
    auto lds_fwd_both = [&](cl_uint len, cl_uint radix) {
        sync_g31_to_g61("mixed odd gf31 ready for fused lds fwd");
        const bool use512 = (radix == 512u && mixed_lds512_opt && len == radix && fk.mixed_fwd_lds512_both);
        cl_kernel k = use512 ? fk.mixed_fwd_lds512_both : fk.mixed_fwd_lds_any_both;
        if (!k) throw std::runtime_error("fused 61x31 LDS forward kernel requested but not available");
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set mixed lds fwd both a61");
        set_karg_mem(k, arg, g31.bufField, "set mixed lds fwd both a31");
        set_karg_mem(k, arg, g61.bufTwFwd, "set mixed lds fwd both tw61");
        set_karg_mem(k, arg, g31.bufTwFwd, "set mixed lds fwd both tw31");
        set_karg(k, arg, storage, "set mixed lds fwd both storage");
        set_karg(k, arg, len, "set mixed lds fwd both len");
        set_karg(k, arg, radix, "set mixed lds fwd both radix");
        const size_t groups = (static_cast<size_t>(storage) / static_cast<size_t>(len)) * static_cast<size_t>(len / radix);
        const std::string label = mixed_lds_label("forward", radix, "61x31", use512, false);
        enqueue_kernel(g61, k, groups * local64, &local64,
                       "enqueue mixed LDS fwd both", label.c_str());
        sync_g61_to_g31("mixed odd fused lds fwd ready");
    };
    auto lds_inv_both = [&](cl_uint base_len, cl_uint radix) {
        sync_g31_to_g61("mixed odd gf31 ready for fused lds inv");
        const bool use512 = (radix == 512u && mixed_lds512_opt && base_len == 1u && fk.mixed_inv_lds512_both);
        cl_kernel k = use512 ? fk.mixed_inv_lds512_both : fk.mixed_inv_lds_any_both;
        if (!k) throw std::runtime_error("fused 61x31 LDS inverse kernel requested but not available");
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set mixed lds inv both a61");
        set_karg_mem(k, arg, g31.bufField, "set mixed lds inv both a31");
        set_karg_mem(k, arg, g61.bufTwInv, "set mixed lds inv both tw61");
        set_karg_mem(k, arg, g31.bufTwInv, "set mixed lds inv both tw31");
        set_karg(k, arg, storage, "set mixed lds inv both storage");
        set_karg(k, arg, base_len, "set mixed lds inv both base_len");
        set_karg(k, arg, radix, "set mixed lds inv both radix");
        const size_t groups = (static_cast<size_t>(storage) / static_cast<size_t>(base_len * radix)) * static_cast<size_t>(base_len);
        const std::string label = mixed_lds_label("inverse", radix, "61x31", use512, false);
        enqueue_kernel(g61, k, groups * local64, &local64,
                       "enqueue mixed LDS inv both", label.c_str());
        sync_g61_to_g31("mixed odd fused lds inv ready");
    };

    auto center61 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_center61, arg, g61.bufField, "set mixed center61 a");
        set_karg_mem(fk.mixed_center61, arg, g61.bufTwFwd, "set mixed center61 twf");
        set_karg_mem(fk.mixed_center61, arg, g61.bufTwInv, "set mixed center61 twi");
        set_karg(fk.mixed_center61, arg, pow2_n, "set mixed center61 pow2_n");
        set_karg(fk.mixed_center61, arg, odd, "set mixed center61 odd");
        set_karg(fk.mixed_center61, arg, flags61, "set mixed center61 flags");
        enqueue_kernel(g61, fk.mixed_center61, g_center_scalar, &local64, "enqueue mixed center61", "crt_mixed_halfreal_center_61");
    };
    auto center31 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_center31, arg, g31.bufField, "set mixed center31 a");
        set_karg_mem(fk.mixed_center31, arg, g31.bufTwFwd, "set mixed center31 twf");
        set_karg_mem(fk.mixed_center31, arg, g31.bufTwInv, "set mixed center31 twi");
        set_karg(fk.mixed_center31, arg, pow2_n, "set mixed center31 pow2_n");
        set_karg(fk.mixed_center31, arg, odd, "set mixed center31 odd");
        set_karg(fk.mixed_center31, arg, flags31, "set mixed center31 flags");
        enqueue_kernel(g31, fk.mixed_center31, g_center_scalar, &local64, "enqueue mixed center31", "crt_mixed_halfreal_center_31");
    };
    auto center_lds512_61 = [&]() {
        const size_t blocks = static_cast<size_t>(row_m >> 9);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        const bool split_f48 = mixed_center512_single_lds_61 && mixed_center512_split_f48_61 && flags61 == 48u &&
                               fk.mixed_lds512_pair_1lds_f48_self61 && fk.mixed_lds512_pair_1lds_f48_nonself61;
        auto set_center_args61 = [&](cl_kernel k) {
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set mixed lds512 center61 a");
            set_karg_mem(k, arg, g61.bufTwFwd, "set mixed lds512 center61 twf");
            set_karg_mem(k, arg, g61.bufTwInv, "set mixed lds512 center61 twi");
            set_karg(k, arg, pow2_n, "set mixed lds512 center61 pow2_n");
            set_karg(k, arg, odd, "set mixed lds512 center61 odd");
            set_karg(k, arg, flags61, "set mixed lds512 center61 flags");
        };
        const bool rega_requested = mixed_center512_single_lds_61 && flags61 == 48u &&
                              (parse_bool_env("PRMERS_CRT_MIXED_CENTER_REGA_61", true) ||
                               parse_bool_env("PRMERS_CRT_MIXED_CENTER_PRIVATE_A_61", false));
        const bool rega_twinline = rega_requested && parse_bool_env("PRMERS_CRT_MIXED_CENTER_TWINLINE_61", true) &&
                                   fk.mixed_lds512_pair_1lds_rega_twinline_f48_61;
        if (rega_twinline) {
            set_center_args61(fk.mixed_lds512_pair_1lds_rega_twinline_f48_61);
            enqueue_kernel(g61, fk.mixed_lds512_pair_1lds_rega_twinline_f48_61,
                           static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                           "enqueue mixed lds512 center61 regA twinline", "crt_mixed_lds512_center_1lds_rega_twinline_f48_61");
            return;
        }
        const bool rega_f48 = rega_requested && fk.mixed_lds512_pair_1lds_rega_f48_61;
        if (rega_f48) {
            set_center_args61(fk.mixed_lds512_pair_1lds_rega_f48_61);
            enqueue_kernel(g61, fk.mixed_lds512_pair_1lds_rega_f48_61,
                           static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                           "enqueue mixed lds512 center61 regA", "crt_mixed_lds512_center_1lds_rega_f48_61");
            return;
        }
        if (split_f48) {
            if (pair_blocks > 2u) {
                set_center_args61(fk.mixed_lds512_pair_1lds_f48_nonself61);
                enqueue_kernel(g61, fk.mixed_lds512_pair_1lds_f48_nonself61,
                               static_cast<size_t>(odd) * (pair_blocks - 2u) * local64, &local64,
                               "enqueue mixed lds512 center61 nonself", "crt_mixed_lds512_center_1lds_f48_nonself_61");
            }
            const size_t self_pairs = (blocks > 1u) ? 2u : 1u;
            set_center_args61(fk.mixed_lds512_pair_1lds_f48_self61);
            enqueue_kernel(g61, fk.mixed_lds512_pair_1lds_f48_self61,
                           static_cast<size_t>(odd) * self_pairs * local64, &local64,
                           "enqueue mixed lds512 center61 self", "crt_mixed_lds512_center_1lds_f48_self_61");
            return;
        }
        if (mixed_center512_single_lds_61 && !fk.mixed_lds512_pair_1lds61)
            throw std::runtime_error("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=1 requested but center61 one-LDS kernel is not available");
        cl_kernel k = mixed_center512_single_lds_61 ? fk.mixed_lds512_pair_1lds61 : fk.mixed_lds512_pair61;
        const char* label = (k == fk.mixed_lds512_pair_1lds61) ?
            "crt_mixed_lds512_center_1lds_61" : "crt_mixed_lds512_center_61";
        if (!k) throw std::runtime_error("mixed LDS512 center61 kernel requested but not available");
        set_center_args61(k);
        enqueue_kernel(g61, k, static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                       "enqueue mixed lds512 center61", label);
    };
    auto center_lds512_31 = [&]() {
        const size_t blocks = static_cast<size_t>(row_m >> 9);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        const bool split_f48 = mixed_center512_single_lds_31 && mixed_center512_split_f48_31 && flags31 == 48u &&
                               fk.mixed_lds512_pair_1lds_f48_self31 && fk.mixed_lds512_pair_1lds_f48_nonself31;
        auto set_center_args31 = [&](cl_kernel k) {
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set mixed lds512 center31 a");
            set_karg_mem(k, arg, g31.bufTwFwd, "set mixed lds512 center31 twf");
            set_karg_mem(k, arg, g31.bufTwInv, "set mixed lds512 center31 twi");
            set_karg(k, arg, pow2_n, "set mixed lds512 center31 pow2_n");
            set_karg(k, arg, odd, "set mixed lds512 center31 odd");
            set_karg(k, arg, flags31, "set mixed lds512 center31 flags");
        };
        const bool rega31 = mixed_center512_single_lds_31 && flags31 == 48u &&
                           parse_bool_env("PRMERS_CRT_MIXED_CENTER_REGA_31", true) &&
                           parse_bool_env("PRMERS_CRT_MIXED_CENTER_TWINLINE_31", true) &&
                           fk.mixed_lds512_pair_1lds_rega_twinline_f48_31;
        if (rega31) {
            set_center_args31(fk.mixed_lds512_pair_1lds_rega_twinline_f48_31);
            enqueue_kernel(g31, fk.mixed_lds512_pair_1lds_rega_twinline_f48_31,
                           static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                           "enqueue mixed lds512 center31 regA twinline", "crt_mixed_lds512_center_1lds_rega_twinline_f48_31");
            return;
        }
        if (split_f48) {
            if (pair_blocks > 2u) {
                set_center_args31(fk.mixed_lds512_pair_1lds_f48_nonself31);
                enqueue_kernel(g31, fk.mixed_lds512_pair_1lds_f48_nonself31,
                               static_cast<size_t>(odd) * (pair_blocks - 2u) * local64, &local64,
                               "enqueue mixed lds512 center31 nonself", "crt_mixed_lds512_center_1lds_f48_nonself_31");
            }
            const size_t self_pairs = (blocks > 1u) ? 2u : 1u;
            set_center_args31(fk.mixed_lds512_pair_1lds_f48_self31);
            enqueue_kernel(g31, fk.mixed_lds512_pair_1lds_f48_self31,
                           static_cast<size_t>(odd) * self_pairs * local64, &local64,
                           "enqueue mixed lds512 center31 self", "crt_mixed_lds512_center_1lds_f48_self_31");
            return;
        }
        if (mixed_center512_single_lds_31 && !fk.mixed_lds512_pair_1lds31)
            throw std::runtime_error("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=1 requested but center31 one-LDS kernel is not available");
        cl_kernel k = mixed_center512_single_lds_31 ? fk.mixed_lds512_pair_1lds31 : fk.mixed_lds512_pair31;
        const char* label = (k == fk.mixed_lds512_pair_1lds31) ?
            "crt_mixed_lds512_center_1lds_31" : "crt_mixed_lds512_center_31";
        if (!k) throw std::runtime_error("mixed LDS512 center31 kernel requested but not available");
        set_center_args31(k);
        enqueue_kernel(g31, k, static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                       "enqueue mixed lds512 center31", label);
    };
    auto center_lds1024_61 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_lds1024_pair61, arg, g61.bufField, "set mixed lds1024 center61 a");
        set_karg_mem(fk.mixed_lds1024_pair61, arg, g61.bufTwFwd, "set mixed lds1024 center61 twf");
        set_karg_mem(fk.mixed_lds1024_pair61, arg, g61.bufTwInv, "set mixed lds1024 center61 twi");
        set_karg(fk.mixed_lds1024_pair61, arg, pow2_n, "set mixed lds1024 center61 pow2_n");
        set_karg(fk.mixed_lds1024_pair61, arg, odd, "set mixed lds1024 center61 odd");
        set_karg(fk.mixed_lds1024_pair61, arg, flags61, "set mixed lds1024 center61 flags");
        const size_t blocks = static_cast<size_t>(row_m >> 10);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        enqueue_kernel(g61, fk.mixed_lds1024_pair61, static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                       "enqueue mixed lds1024 center61", "crt_mixed_lds1024_center_61");
    };
    auto center_lds1024_31 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_lds1024_pair31, arg, g31.bufField, "set mixed lds1024 center31 a");
        set_karg_mem(fk.mixed_lds1024_pair31, arg, g31.bufTwFwd, "set mixed lds1024 center31 twf");
        set_karg_mem(fk.mixed_lds1024_pair31, arg, g31.bufTwInv, "set mixed lds1024 center31 twi");
        set_karg(fk.mixed_lds1024_pair31, arg, pow2_n, "set mixed lds1024 center31 pow2_n");
        set_karg(fk.mixed_lds1024_pair31, arg, odd, "set mixed lds1024 center31 odd");
        set_karg(fk.mixed_lds1024_pair31, arg, flags31, "set mixed lds1024 center31 flags");
        const size_t blocks = static_cast<size_t>(row_m >> 10);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        enqueue_kernel(g31, fk.mixed_lds1024_pair31, static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                       "enqueue mixed lds1024 center31", "crt_mixed_lds1024_center_31");
    };

    auto center_lds_any_61 = [&](cl_uint center_len) {
        const bool use1 = mixed_center_single_lds_61 && fk.mixed_lds_any_pair_1lds61;
        const cl_uint log_c = ([](cl_uint v){ cl_uint r=0; while (v > 1u) { v >>= 1; ++r; } return r; }(center_len));
        const size_t blocks = static_cast<size_t>(row_m >> log_c);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        auto set_center_args61 = [&](cl_kernel k) {
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set mixed lds any center61 a");
            set_karg_mem(k, arg, g61.bufTwFwd, "set mixed lds any center61 twf");
            set_karg_mem(k, arg, g61.bufTwInv, "set mixed lds any center61 twi");
            set_karg(k, arg, pow2_n, "set mixed lds any center61 pow2_n");
            set_karg(k, arg, odd, "set mixed lds any center61 odd");
            set_karg(k, arg, flags61, "set mixed lds any center61 flags");
            set_karg(k, arg, center_len, "set mixed lds any center61 center_len");
        };
        const bool split_f48 = use1 && mixed_center_split_f48_61 && flags61 == 48u &&
            fk.mixed_lds_any_pair_1lds_f48_self61 && fk.mixed_lds_any_pair_1lds_f48_nonself61;
        if (split_f48) {
            if (pair_blocks > 2u) {
                set_center_args61(fk.mixed_lds_any_pair_1lds_f48_nonself61);
                const std::string label = std::string("crt_mixed_lds") + std::to_string(center_len) + "_center_1lds_f48_nonself_61";
                enqueue_kernel(g61, fk.mixed_lds_any_pair_1lds_f48_nonself61,
                               static_cast<size_t>(odd) * (pair_blocks - 2u) * local64, &local64,
                               "enqueue mixed lds any center61 nonself", label.c_str());
            }
            const size_t self_pairs = (blocks > 1u) ? 2u : 1u;
            set_center_args61(fk.mixed_lds_any_pair_1lds_f48_self61);
            const std::string label = std::string("crt_mixed_lds") + std::to_string(center_len) + "_center_1lds_f48_self_61";
            enqueue_kernel(g61, fk.mixed_lds_any_pair_1lds_f48_self61,
                           static_cast<size_t>(odd) * self_pairs * local64, &local64,
                           "enqueue mixed lds any center61 self", label.c_str());
            return;
        }
        cl_kernel k = use1 ? fk.mixed_lds_any_pair_1lds61 : fk.mixed_lds_any_pair61;
        if (!k) throw std::runtime_error("mixed LDS any center61 kernel requested but not available");
        set_center_args61(k);
        const std::string label = std::string("crt_mixed_lds") + std::to_string(center_len) +
                                  (use1 ? "_center_1lds_61" : "_center_61");
        enqueue_kernel(g61, k, static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                       "enqueue mixed lds any center61", label.c_str());
    };
    auto center_lds_any_31 = [&](cl_uint center_len) {
        const bool use1 = mixed_center_single_lds_31 && fk.mixed_lds_any_pair_1lds31;
        const cl_uint log_c = ([](cl_uint v){ cl_uint r=0; while (v > 1u) { v >>= 1; ++r; } return r; }(center_len));
        const size_t blocks = static_cast<size_t>(row_m >> log_c);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        auto set_center_args31 = [&](cl_kernel k) {
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set mixed lds any center31 a");
            set_karg_mem(k, arg, g31.bufTwFwd, "set mixed lds any center31 twf");
            set_karg_mem(k, arg, g31.bufTwInv, "set mixed lds any center31 twi");
            set_karg(k, arg, pow2_n, "set mixed lds any center31 pow2_n");
            set_karg(k, arg, odd, "set mixed lds any center31 odd");
            set_karg(k, arg, flags31, "set mixed lds any center31 flags");
            set_karg(k, arg, center_len, "set mixed lds any center31 center_len");
        };
        const bool split_f48 = use1 && mixed_center_split_f48_31 && flags31 == 48u &&
            fk.mixed_lds_any_pair_1lds_f48_self31 && fk.mixed_lds_any_pair_1lds_f48_nonself31;
        if (split_f48) {
            if (pair_blocks > 2u) {
                set_center_args31(fk.mixed_lds_any_pair_1lds_f48_nonself31);
                const std::string label = std::string("crt_mixed_lds") + std::to_string(center_len) + "_center_1lds_f48_nonself_31";
                enqueue_kernel(g31, fk.mixed_lds_any_pair_1lds_f48_nonself31,
                               static_cast<size_t>(odd) * (pair_blocks - 2u) * local64, &local64,
                               "enqueue mixed lds any center31 nonself", label.c_str());
            }
            const size_t self_pairs = (blocks > 1u) ? 2u : 1u;
            set_center_args31(fk.mixed_lds_any_pair_1lds_f48_self31);
            const std::string label = std::string("crt_mixed_lds") + std::to_string(center_len) + "_center_1lds_f48_self_31";
            enqueue_kernel(g31, fk.mixed_lds_any_pair_1lds_f48_self31,
                           static_cast<size_t>(odd) * self_pairs * local64, &local64,
                           "enqueue mixed lds any center31 self", label.c_str());
            return;
        }
        cl_kernel k = use1 ? fk.mixed_lds_any_pair_1lds31 : fk.mixed_lds_any_pair31;
        if (!k) throw std::runtime_error("mixed LDS any center31 kernel requested but not available");
        set_center_args31(k);
        const std::string label = std::string("crt_mixed_lds") + std::to_string(center_len) +
                                  (use1 ? "_center_1lds_31" : "_center_31");
        enqueue_kernel(g31, k, static_cast<size_t>(odd) * pair_blocks * local64, &local64,
                       "enqueue mixed lds any center31", label.c_str());
    };

    auto center_lds_both = [&](cl_uint center_len) {
        if (center_len > 512u) {
            throw std::runtime_error("fused 61x31 LDS center supports center_len <= 512 only; use off/auto/all fallback for center=1024");
        }
        cl_event gf31_fwd_ready = enqueue_queue_marker(g31, "mixed odd gf31 ready for fused center");
        set_pending_wait_event(g61, gf31_fwd_ready);
        if (g31.queue != g61.queue) maybe_flush_mixed(g31.queue);

        const bool use512 = (center_len == 512u && fk.mixed_lds512_pair_both);
        cl_kernel k = use512 ? fk.mixed_lds512_pair_both : fk.mixed_lds_any_pair_both;
        if (!k) throw std::runtime_error("fused 61x31 LDS center kernel requested but not available");
        const cl_uint log_c = ([](cl_uint v){ cl_uint r=0; while (v > 1u) { v >>= 1; ++r; } return r; }(center_len));
        const size_t blocks = static_cast<size_t>(row_m >> log_c);
        const size_t pair_blocks = (blocks >> 1u) + 1u;
        const size_t groups = static_cast<size_t>(odd) * pair_blocks;
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set mixed lds both a61");
        set_karg_mem(k, arg, g31.bufField, "set mixed lds both a31");
        set_karg_mem(k, arg, g61.bufTwFwd, "set mixed lds both twf61");
        set_karg_mem(k, arg, g61.bufTwInv, "set mixed lds both twi61");
        set_karg_mem(k, arg, g31.bufTwFwd, "set mixed lds both twf31");
        set_karg_mem(k, arg, g31.bufTwInv, "set mixed lds both twi31");
        set_karg(k, arg, pow2_n, "set mixed lds both pow2_n");
        set_karg(k, arg, odd, "set mixed lds both odd");
        set_karg(k, arg, flags61, "set mixed lds both flags61");
        set_karg(k, arg, flags31, "set mixed lds both flags31");
        if (!use512) set_karg(k, arg, center_len, "set mixed lds both center_len");
        const std::string label = "crt_mixed_lds" + std::to_string(center_len) + "_center_61x31" + (use512 ? std::string("_opt") : std::string());
        enqueue_kernel(g61, k, groups * local64, &local64,
                       "enqueue mixed LDS center both", label.c_str());

        cl_event center_ready = enqueue_queue_marker(g61, "mixed odd fused center ready");
        set_pending_wait_event(g31, center_ready);
        if (g61.queue != g31.queue) maybe_flush_mixed(g61.queue);
    };

    auto unpack61 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_unpack61, arg, g61.bufField, "set mixed unpack61 a");
        set_karg_mem(fk.mixed_unpack61, arg, g61.bufDigits, "set mixed unpack61 digits");
        set_karg(fk.mixed_unpack61, arg, n, "set mixed unpack61 n");
        set_karg(fk.mixed_unpack61, arg, p, "set mixed unpack61 p");
        set_karg(fk.mixed_unpack61, arg, g61.lr2, "set mixed unpack61 lr2");
        set_karg(fk.mixed_unpack61, arg, odd, "set mixed unpack61 odd");
        set_karg(fk.mixed_unpack61, arg, pow2_n, "set mixed unpack61 pow2_n");
        set_karg(fk.mixed_unpack61, arg, log_m, "set mixed unpack61 log_m");
        enqueue_kernel(g61, fk.mixed_unpack61, g_storage, &local64, "enqueue mixed unpack61", "crt_mixed_unpack_unweight_61");
    };
    auto unpack31 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_unpack31, arg, g31.bufField, "set mixed unpack31 a");
        set_karg_mem(fk.mixed_unpack31, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set mixed unpack31 digits");
        set_karg(fk.mixed_unpack31, arg, n, "set mixed unpack31 n");
        set_karg(fk.mixed_unpack31, arg, p, "set mixed unpack31 p");
        set_karg(fk.mixed_unpack31, arg, g31.lr2, "set mixed unpack31 lr2");
        set_karg(fk.mixed_unpack31, arg, odd, "set mixed unpack31 odd");
        set_karg(fk.mixed_unpack31, arg, pow2_n, "set mixed unpack31 pow2_n");
        set_karg(fk.mixed_unpack31, arg, log_m, "set mixed unpack31 log_m");
        enqueue_kernel(g31, fk.mixed_unpack31, g_storage, &local64, "enqueue mixed unpack31", "crt_mixed_unpack_unweight_31");
    };


    auto odd_inv_unpack61 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_odd_inv_unpack61, arg, g61.bufField, "set mixed fused inv+unpack61 a");
        set_karg_mem(fk.mixed_odd_inv_unpack61, arg, g61.bufDigits, "set mixed fused inv+unpack61 digits");
        set_karg_mem(fk.mixed_odd_inv_unpack61, arg, g61.bufOddInv, "set mixed fused inv+unpack61 mat");
        set_karg(fk.mixed_odd_inv_unpack61, arg, n, "set mixed fused inv+unpack61 n");
        set_karg(fk.mixed_odd_inv_unpack61, arg, p, "set mixed fused inv+unpack61 p");
        set_karg(fk.mixed_odd_inv_unpack61, arg, g61.lr2, "set mixed fused inv+unpack61 lr2");
        set_karg(fk.mixed_odd_inv_unpack61, arg, odd, "set mixed fused inv+unpack61 odd");
        set_karg(fk.mixed_odd_inv_unpack61, arg, pow2_n, "set mixed fused inv+unpack61 pow2_n");
        set_karg(fk.mixed_odd_inv_unpack61, arg, log_m, "set mixed fused inv+unpack61 log_m");
        enqueue_kernel(g61, fk.mixed_odd_inv_unpack61, g_row_m, &local64, "enqueue mixed fused inv+unpack61", "crt_mixed_odd_inv_unpack_unweight_61");
    };
    auto odd_inv_unpack31 = [&]() {
        cl_uint arg = 0;
        set_karg_mem(fk.mixed_odd_inv_unpack31, arg, g31.bufField, "set mixed fused inv+unpack31 a");
        set_karg_mem(fk.mixed_odd_inv_unpack31, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set mixed fused inv+unpack31 digits");
        set_karg_mem(fk.mixed_odd_inv_unpack31, arg, g31.bufOddInv, "set mixed fused inv+unpack31 mat");
        set_karg(fk.mixed_odd_inv_unpack31, arg, n, "set mixed fused inv+unpack31 n");
        set_karg(fk.mixed_odd_inv_unpack31, arg, p, "set mixed fused inv+unpack31 p");
        set_karg(fk.mixed_odd_inv_unpack31, arg, g31.lr2, "set mixed fused inv+unpack31 lr2");
        set_karg(fk.mixed_odd_inv_unpack31, arg, odd, "set mixed fused inv+unpack31 odd");
        set_karg(fk.mixed_odd_inv_unpack31, arg, pow2_n, "set mixed fused inv+unpack31 pow2_n");
        set_karg(fk.mixed_odd_inv_unpack31, arg, log_m, "set mixed fused inv+unpack31 log_m");
        enqueue_kernel(g31, fk.mixed_odd_inv_unpack31, g_row_m, &local64, "enqueue mixed fused inv+unpack31", "crt_mixed_odd_inv_unpack_unweight_31");
    };


    auto odd_inv_precrt_coeffhi = [&]() {
        
        
        cl_event gf31_ready = enqueue_queue_marker(g31, "mixed odd gf31 ready for preCRT coeffhi");
        set_pending_wait_event(g61, gf31_ready);
        if (g31.queue != g61.queue) maybe_flush_mixed(g31.queue);

        cl_kernel ktail = use_mixed_tile14_shift_lmat ? fk.mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat :
                          (use_mixed_precrt_tile14_shift ? fk.mixed_odd_inv_precrt_coeffhi_tile14_shift :
                          (use_mixed_shift_lut ? fk.mixed_odd_inv_precrt_coeffhi_shift :
                          (use_mixed_precrt_tile14 ? fk.mixed_odd_inv_precrt_coeffhi_tile14 :
                          (use_mixed_precrt_tile7 ? fk.mixed_odd_inv_precrt_coeffhi_tile7 :
                          (use_mixed_precrt_outpar ? fk.mixed_odd_inv_precrt_coeffhi_outpar : fk.mixed_odd_inv_precrt_coeffhi)))));
        cl_uint arg = 0;
        set_karg_mem(ktail, arg, g61.bufField, "set mixed preCRT a61");
        set_karg_mem(ktail, arg, g31.bufField, "set mixed preCRT a31");
        set_karg_mem(ktail, arg, g61.bufDigits, "set mixed preCRT coeff lo");
        set_karg_mem(ktail, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set mixed preCRT coeff hi");
        set_karg_mem(ktail, arg, g61.bufOddInv, "set mixed preCRT mat61");
        set_karg_mem(ktail, arg, g31.bufOddInv, "set mixed preCRT mat31");
        if (use_mixed_shift_lut) {
            set_karg_mem(ktail, arg, g61.bufUnweightShift, "set mixed preCRT shift61");
            set_karg_mem(ktail, arg, g31.bufUnweightShift, "set mixed preCRT shift31");
            set_karg(ktail, arg, odd, "set mixed preCRT odd");
            set_karg(ktail, arg, pow2_n, "set mixed preCRT pow2_n");
            set_karg(ktail, arg, log_m, "set mixed preCRT log_m");
        } else {
            set_karg(ktail, arg, n, "set mixed preCRT n");
            set_karg(ktail, arg, p, "set mixed preCRT p");
            set_karg(ktail, arg, g61.lr2, "set mixed preCRT lr2_61");
            set_karg(ktail, arg, g31.lr2, "set mixed preCRT lr2_31");
            set_karg(ktail, arg, odd, "set mixed preCRT odd");
            set_karg(ktail, arg, pow2_n, "set mixed preCRT pow2_n");
            set_karg(ktail, arg, log_m, "set mixed preCRT log_m");
        }
        const size_t tail_global = (use_mixed_precrt_tile14 || use_mixed_precrt_tile14_shift) ?
            round_up_size(((static_cast<size_t>(row_m) + 13u) / 14u) * 128u, local128) :
            (use_mixed_precrt_tile7 ?
            round_up_size(((static_cast<size_t>(row_m) + 6u) / 7u) * 64u, local64) :
            (use_mixed_precrt_outpar ? g_storage : g_row_m));
        const char* tail_name = use_mixed_tile14_shift_lmat ? "crt_mixed_odd_inv_precrt_coeffhi_tile14_shift_lmat" :
                                (use_mixed_precrt_tile14_shift ? "crt_mixed_odd_inv_precrt_coeffhi_tile14_shift" :
                                (use_mixed_shift_lut ? "crt_mixed_odd_inv_precrt_coeffhi_shift" :
                                (use_mixed_precrt_tile14 ? "crt_mixed_odd_inv_precrt_coeffhi_tile14" :
                                (use_mixed_precrt_tile7 ? "crt_mixed_odd_inv_precrt_coeffhi_tile7" :
                                (use_mixed_precrt_outpar ? "crt_mixed_odd_inv_precrt_coeffhi_outpar" : "crt_mixed_odd_inv_precrt_coeffhi")))));
        enqueue_kernel(g61, ktail, tail_global, (use_mixed_precrt_tile14 || use_mixed_precrt_tile14_shift) ? &local128 : &local64,
                       "enqueue mixed odd preCRT coeffhi", tail_name);
        g61.crtCoeffPending = true;
        g61.crtLastUnweightPending = false;
    };

    auto odd_inv_precrt_garner64 = [&]() {
        cl_event gf31_ready = enqueue_queue_marker(g31, "mixed odd gf31 ready for fused preCRT+Garner");
        set_pending_wait_event(g61, gf31_ready);
        if (g31.queue != g61.queue) maybe_flush_mixed(g31.queue);

        ensure_carry_buffers(g61);
        cl_uint zero = 0;
        check(clEnqueueWriteBuffer(g61.queue, g61.bufCarryPending, CL_FALSE, 0, sizeof(zero), &zero, 0, nullptr, nullptr), "mixed fused precrt-garner clear pending");

        const cl_uint digit_n = static_cast<cl_uint>(g61.n);
        const cl_uint items = 64u;
        const cl_uint segments = static_cast<cl_uint>((static_cast<std::size_t>(digit_n) + items - 1u) / items);
        const cl_uint width_base = static_cast<cl_uint>(g61.min_digit_width);
        const bool base_9seg_ok = !parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_NATURAL64", false) &&
                                  odd == 9u && (pow2_n & 63u) == 0u && digit_n == odd * pow2_n;
        const bool want_9seg30_pair = base_9seg_ok && width_base == 30u &&
                                     parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30", true);
        const bool use_9seg30_smat = want_9seg30_pair &&
                                     parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30_SMAT", true) &&
                                     fk.mixed_odd9_inv_precrt_garner9seg30_pair_smat;
        const bool use_9seg30_pair = want_9seg30_pair && !use_9seg30_smat &&
                                     fk.mixed_odd9_inv_precrt_garner9seg30_pair_lmat;
        const bool use_9seg = !use_9seg30_smat && !use_9seg30_pair && fk.mixed_odd9_inv_precrt_garner9seg_lmat && base_9seg_ok;
        const size_t local = (use_9seg30_smat || use_9seg30_pair) ? 128u : (use_9seg ? 256u : 64u);
        const size_t groups = (use_9seg30_smat || use_9seg30_pair || use_9seg) ? static_cast<size_t>(pow2_n >> 6)
                                                                               : static_cast<size_t>(segments);
        const size_t global = round_up_size(groups * local, local);
        cl_kernel k = use_9seg30_smat ? fk.mixed_odd9_inv_precrt_garner9seg30_pair_smat
                                      : (use_9seg30_pair ? fk.mixed_odd9_inv_precrt_garner9seg30_pair_lmat
                                                         : (use_9seg ? fk.mixed_odd9_inv_precrt_garner9seg_lmat
                                                                     : fk.mixed_odd9_inv_precrt_garner64_lmat));
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set mixed fused precrt-garner a61");
        set_karg_mem(k, arg, g31.bufField, "set mixed fused precrt-garner a31");
        set_karg_mem(k, arg, g61.bufOddInv, "set mixed fused precrt-garner mat61");
        set_karg_mem(k, arg, g31.bufOddInv, "set mixed fused precrt-garner mat31");
        set_karg_mem(k, arg, g61.bufUnweightShift, "set mixed fused precrt-garner shift61");
        set_karg_mem(k, arg, g31.bufUnweightShift, "set mixed fused precrt-garner shift31");
        set_karg_mem(k, arg, g61.bufWidthMask32, "set mixed fused precrt-garner width mask");
        set_karg(k, arg, width_base, "set mixed fused precrt-garner width base");
        set_karg_mem(k, arg, g61.bufDigits, "set mixed fused precrt-garner digits");
        set_karg_mem(k, arg, g61.bufBlockCarry, "set mixed fused precrt-garner carry lo");
        set_karg_mem(k, arg, g61.bufCrtCarryHi1, "set mixed fused precrt-garner carry hi");
        set_karg_mem(k, arg, g61.bufCarryPending, "set mixed fused precrt-garner pending");
        set_karg(k, arg, pow2_n, "set mixed fused precrt-garner pow2_n");
        set_karg(k, arg, log_m, "set mixed fused precrt-garner log_m");
        set_karg(k, arg, digit_n, "set mixed fused precrt-garner digit_n");
        set_karg(k, arg, segments, "set mixed fused precrt-garner segments");
        enqueue_kernel(g61, k, global, &local, "enqueue mixed fused preCRT+Garner",
                       use_9seg30_smat ? "crt_mixed_odd9_precrt_garner9seg30_pair_smat" :
                       (use_9seg30_pair ? "crt_mixed_odd9_precrt_garner9seg30_pair_lmat" :
                       (use_9seg ? "crt_mixed_odd9_precrt_garner9seg_lmat" : "crt_mixed_odd9_precrt_garner64_lmat")));
        g61.crtFirstCarryReady = true;
        g61.crtCoeffPending = false;
        g61.crtLastUnweightPending = false;
    };

    auto residues_to_coeffhi = [&]() {
        
        
        cl_event gf31_ready = enqueue_queue_marker(g31, "mixed odd gf31 residues ready for coeffhi");
        set_pending_wait_event(g61, gf31_ready);
        if (g31.queue != g61.queue) maybe_flush_mixed(g31.queue);

        cl_uint arg = 0;
        set_karg_mem(fk.mixed_residues_to_coeffhi, arg, g61.bufDigits, "set mixed residues coeff lo/res61");
        set_karg_mem(fk.mixed_residues_to_coeffhi, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set mixed residues coeff hi/res31");
        set_karg(fk.mixed_residues_to_coeffhi, arg, n, "set mixed residues n");
        enqueue_kernel(g61, fk.mixed_residues_to_coeffhi,
                       round_up_size(static_cast<size_t>(n), local64), &local64,
                       "enqueue mixed residues to coeffhi", "crt_mixed_residues_to_coeffhi");
        g61.crtCoeffPending = true;
        g61.crtLastUnweightPending = false;
    };

    const bool use_prepacked_head = g_crt_mixed_skip_pack_this_square && use_mixed_pack_both;
    if (use_prepacked_head) {
        // a61/a31 already contain pack+weight+oddDFT from the previous carry boundary.
        // Keep this limited to the fused 61x31 pack path so both queues see the same marker logic.
    } else if (use_mixed_pack_both) {
        pack_odd_fwd_both();
    } else if (use_mixed_fused_edges) {
        pack_odd_fwd61(); pack_odd_fwd31();
    } else {
        pack61(); pack31();
        odd_dft61(g61.bufOddFwd, "crt_mixed_odd_fwd_61");
        odd_dft31(g31.bufOddFwd, "crt_mixed_odd_fwd_31");
    }

    if (use_mixed_lds_any) {
        const cl_uint center_len = mixed_row_center;
        const cl_uint requested_stage_cap = (g_crt_lds_stage >= 8u) ? g_crt_lds_stage : center_len;
        const cl_uint stage_cap = std::max<cl_uint>(2u, std::min<cl_uint>(requested_stage_cap, row_m));
        cl_uint cur = row_m;
        while (cur > center_len) {
            cl_uint radix = floor_pow2_leq(std::min<cl_uint>(stage_cap, cur / center_len));
            if (radix < 2u) radix = 2u;
            if (use_mixed_stage_both &&
                (fk.mixed_fwd_lds_any_both || (use_mixed_lds512 && mixed_lds512_opt && radix == 512u && (cur / radix) == 1u && fk.mixed_fwd_lds512_both))) {
                lds_fwd_both(cur, radix);
            } else {
                lds_fwd61(cur, radix);
                lds_fwd31(cur, radix);
            }
            cur /= radix;
        }
        if (use_mixed_center_both) {
            center_lds_both(center_len);
        } else if (use_mixed_lds512) {
            center_lds512_61();
            center_lds512_31();
        } else if ((mixed_center_single_lds_61 || mixed_center_single_lds_31) &&
                   ((mixed_center_single_lds_61 ? fk.mixed_lds_any_pair_1lds61 : fk.mixed_lds_any_pair61) || fk.mixed_lds_any_pair61) &&
                   ((mixed_center_single_lds_31 ? fk.mixed_lds_any_pair_1lds31 : fk.mixed_lds_any_pair31) || fk.mixed_lds_any_pair31)) {
            center_lds_any_61(center_len);
            center_lds_any_31(center_len);
        } else if (use_mixed_lds1024) {
            center_lds1024_61();
            center_lds1024_31();
        } else {
            center_lds_any_61(center_len);
            center_lds_any_31(center_len);
        }
        cur = center_len;
        while (cur < row_m) {
            cl_uint radix = floor_pow2_leq(std::min<cl_uint>(stage_cap, row_m / cur));
            if (radix < 2u) radix = 2u;
            if (use_mixed_stage_both &&
                (fk.mixed_inv_lds_any_both || (use_mixed_lds512 && mixed_lds512_opt && radix == 512u && cur == 1u && fk.mixed_inv_lds512_both))) {
                lds_inv_both(cur, radix);
            } else {
                lds_inv61(cur, radix);
                lds_inv31(cur, radix);
            }
            cur *= radix;
        }
    } else {
        for (int i = static_cast<int>(g61.stages.size()) - 1; i >= 0; --i) {
            const StageInfo& st = g61.stages[static_cast<size_t>(i)];
            if (st.len <= row_m) { fwd61(st); fwd31(st); }
        }
        center61(); center31();
        for (const StageInfo& st : g61.stages) {
            if (st.len <= row_m) { inv61(st); inv31(st); }
        }
    }

    if (use_mixed_precrt_split) {
        odd_inv_unpack61();
        odd_inv_unpack31();
        residues_to_coeffhi();
    } else if (use_mixed_precrt_garner64) {
        odd_inv_precrt_garner64();
    } else if (use_mixed_precrt_coeffhi) {
        odd_inv_precrt_coeffhi();
    } else if (use_mixed_fused_edges) {
        odd_inv_unpack61(); odd_inv_unpack31();
    } else {
        odd_dft61(g61.bufOddInv, "crt_mixed_odd_inv_norm_61");
        odd_dft31(g31.bufOddInv, "crt_mixed_odd_inv_norm_31");
        unpack61(); unpack31();
    }

    // Do not force a host/GPU sync at the end of the mixed odd square by default.
    // The following Garner/carry kernels are enqueued on the GF61 queue, and the
    // mixed odd tail already inserts the GF31->GF61 marker when it reads a31.
    // enqueue_crt_garner_carry_gpu() then publishes the cleanup event back to GF31
    // for the next iteration.  Keep the old behaviour available for debugging.
    if (parse_bool_env("PRMERS_CRT_MIXED_FINISH_AFTER_SQUARE", false)) {
        check(clFinish(g61.queue), "clFinish mixed odd gf61");
        check(clFinish(g31.queue), "clFinish mixed odd gf31");
    }
    return true;
}


// v57 safe pre-pack test: after carry has stabilized digits, build the
// pack+weight+oddDFT input for the next iteration.  This deliberately reuses
// the proven v53 tile28 61x31 pack kernel; the next square skips only the head
// pack enqueue.  It does not fuse carry arithmetic yet, so it is a correctness
// and scheduling experiment rather than the final memory-pass-saving kernel.
static bool enqueue_crt_mixed_pack_next_after_carry(GpuPrp& g61, GpuPrp& g31, CrtFusedKernels& fk) {
    const cl_uint odd = g_crt_odd_radix;
    if (odd != 9u) return false;
    if (g61.n != g31.n) return false;
    const cl_uint n = g61.n;
    if ((n % odd) != 0u) return false;
    const cl_uint pow2_n = n / odd;
    if (pow2_n < 4u || (pow2_n & (pow2_n - 1u)) != 0u) return false;
    const cl_uint row_m = pow2_n >> 1;
    if (!fk.mixed_pack_odd_fwd_tile28_shift_lmat_both) return false;
    if (!g61.bufUnweightShift || !g31.bufUnweightShift || !g61.bufOddFwd || !g31.bufOddFwd) return false;

    const size_t local256 = 256u;
    const size_t tile = 28u;
    const size_t wg = 256u;
    const size_t head_global = round_up_size(((static_cast<size_t>(row_m) + tile - 1u) / tile) * wg, local256);
    cl_kernel kpack = fk.mixed_pack_odd_fwd_tile28_shift_lmat_both;

    cl_uint arg = 0;
    set_karg_mem(kpack, arg, g61.bufDigits, "set mixed pack-next digits");
    set_karg_mem(kpack, arg, g61.bufField, "set mixed pack-next a61");
    set_karg_mem(kpack, arg, g31.bufField, "set mixed pack-next a31");
    set_karg_mem(kpack, arg, g61.bufOddFwd, "set mixed pack-next mat61");
    set_karg_mem(kpack, arg, g31.bufOddFwd, "set mixed pack-next mat31");
    set_karg_mem(kpack, arg, g61.bufUnweightShift, "set mixed pack-next shift61");
    set_karg_mem(kpack, arg, g31.bufUnweightShift, "set mixed pack-next shift31");
    set_karg(kpack, arg, odd, "set mixed pack-next odd");
    set_karg(kpack, arg, pow2_n, "set mixed pack-next pow2_n");

    enqueue_kernel(g61, kpack, head_global, &local256,
                   "enqueue mixed pack-next tile28 61x31",
                   "crt_mixed_pack_next_tile28_shift_lmat_61x31");
    cl_event both_ready = enqueue_queue_marker(g61, "mixed pack-next both ready");
    set_pending_wait_event(g31, both_ready);
    if (parse_bool_env("PRMERS_CRT_MIXED_ALLOW_HOST_FLUSH",
                       parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false)) && g61.queue != g31.queue) {
        clFlush(g61.queue);
    }
    return true;
}

static bool enqueue_square_mod_crt_defused_fast(GpuPrp& g61, GpuPrp& g31, CrtFusedKernels& fk) {
    if (g_crt_odd_radix > 1u) return enqueue_square_mod_crt_mixed_odd(g61, g31, fk);
    if (!fk.defused_fast_available()) return false;
    if (g61.n != g31.n) return false;
    if (g61.n < 512u || (g61.n & 511u) != 0u) return false;
    if (g_force_strict_reference) return false;

    const cl_uint n = g61.n;
    const cl_uint p = g61.exponent_p;
    const cl_uint log_n = ([](cl_uint v){ cl_uint r=0; while (v > 1u) { v >>= 1; ++r; } return r; }(n));
    const size_t local64 = 64u;
    const cl_uint center_chunk = 512u;
    const auto& stages = g61.stages;

    
    const cl_uint requested_lds = floor_pow2_leq(std::min<cl_uint>(g_crt_lds_stage, 512u));
    const cl_uint max_ratio = (n / 4u) / center_chunk;
    cl_uint defused_lds_radix = 0u;
    if (requested_lds >= 16u && max_ratio >= 16u &&
        fk.fwd_lds_stage_61 && fk.inv_lds_stage_61 && fk.fwd_lds_stage_31 && fk.inv_lds_stage_31) {
        defused_lds_radix = floor_pow2_leq(std::min<cl_uint>(requested_lds, std::min<cl_uint>(512u, max_ratio)));
    }
    
    
    const bool allow_defused_lds_lt512 = parse_bool_env("PRMERS_CRT_LDS_STAGE_LT512", false);
    const bool use_defused_lds_stage = (defused_lds_radix == 512u) ||
        (allow_defused_lds_lt512 && defused_lds_radix >= 16u);
    const cl_uint global_stop_chunk = use_defused_lds_stage ? center_chunk * defused_lds_radix : center_chunk;
    const uint32_t edge_radix = (g_crt_defused_edge_radix == 2u || g_crt_defused_edge_radix == 4u ||
                                   g_crt_defused_edge_radix == 8u || g_crt_defused_edge_radix == 16u)
                                      ? g_crt_defused_edge_radix : 4u;
    const cl_uint edge_log = (edge_radix == 2u) ? 1u : (edge_radix == 4u) ? 2u : (edge_radix == 8u) ? 3u : 4u;

    
    const bool force_generic_edge = (g_crt_defused_edge_mode == 2);
    const bool legacy_radix4_edge = (edge_radix == 4u) && !force_generic_edge;

    
    const bool fuse_forward_edge = ((g_crt_defused_edge_fuse & 1) != 0) &&
                                   legacy_radix4_edge &&
                                   fk.defused_forward_edge_available() &&
                                   ((n >> 2) > 4u * global_stop_chunk);
    const bool fuse_tail_edge = ((g_crt_defused_edge_fuse & 2) != 0) &&
                                legacy_radix4_edge &&
                                fk.defused_tail_edge_available() &&
                                (n >= 4096u);
    
    
    const bool request_precrt_block16 = parse_bool_env("PRMERS_CRT_TAIL_PRECRT_BLOCK16", false);
    const bool use_precrt_block16 = fuse_tail_edge && request_precrt_block16 &&
                                    (g_crt_defused_schedule == 0) &&
                                    fk.inv4_last_unweight16_crtcoeff &&
                                    g61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32 &&
                                    g61.bufWidthMask32 && g61.min_digit_width == 32u;
    const bool use_block16_tail = !use_precrt_block16 && fuse_tail_edge &&
                                  (std::getenv("PRMERS_CRT_TAIL_BLOCK16") != nullptr) &&
                                  fk.inv4_last_unweight16_61 && fk.inv4_last_unweight16_31;

    const bool lds_hotopt = parse_bool_env("PRMERS_CRT_LDS_HOTOPT", true);
    
    
    const bool use_lds_tile4 = use_defused_lds_stage && defused_lds_radix == 512u &&
                               parse_bool_env("PRMERS_CRT_LDS_TILE4", false) &&
                               fk.fwd_lds_stage_61_tile4 && fk.inv_lds_stage_61_tile4 &&
                               fk.fwd_lds_stage_31_tile4 && fk.inv_lds_stage_31_tile4;
    const bool use_lds_stage512_opt = !use_lds_tile4 && use_defused_lds_stage && defused_lds_radix == 512u &&
                                      (lds_hotopt || parse_bool_env("PRMERS_CRT_LDS_STAGE512_OPT", false)) &&
                                      fk.fwd_lds_stage_61_512opt && fk.inv_lds_stage_61_512opt &&
                                      fk.fwd_lds_stage_31_512opt && fk.inv_lds_stage_31_512opt;
    const bool use_center512_opt = (lds_hotopt || parse_bool_env("PRMERS_CRT_CENTER512_OPT", false)) &&
                                   fk.center512_61_opt && fk.center512_31_opt;

    const uint32_t forward_edge_stop = static_cast<uint32_t>(n / edge_radix);
    const uint32_t inverse_edge_stop = static_cast<uint32_t>((2ull * n) / edge_radix);
    const cl_uint inverse_stop = inverse_edge_stop;
    const bool use_generic_edge = force_generic_edge || (edge_radix != 4u);

    const bool allow_crt_host_flush = parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false);
    auto flush_async_queues = [&]() {
        // Host-side flush is disabled by default.  Event dependencies are enough
        // for correctness; explicit clFlush/clFinish in the hot loop can distort
        // timing and reduce throughput.  Re-enable only for driver debugging with
        // PRMERS_CRT_ALLOW_HOST_FLUSH=1.
        if (allow_crt_host_flush && g31.queue != g61.queue) {
            clFlush(g61.queue);
            clFlush(g31.queue);
        }
    };

    auto enqueue_weight61 = [&]() {
        const StageInfo& first = stages.back();
        cl_uint arg = 0;
        if (fuse_forward_edge) {
            set_karg_mem(fk.weight_fwd8_61, arg, g61.bufDigits, "set defused edge fwd61 digits");
            set_karg_mem(fk.weight_fwd8_61, arg, g61.bufField, "set defused edge fwd61 field");
            set_karg_mem(fk.weight_fwd8_61, arg, g61.bufTwFwd, "set defused edge fwd61 tw");
            set_karg(fk.weight_fwd8_61, arg, p, "set defused edge fwd61 p");
            set_karg(fk.weight_fwd8_61, arg, g61.lr2, "set defused edge fwd61 lr2");
            set_karg(fk.weight_fwd8_61, arg, n, "set defused edge fwd61 n");
            size_t global = n / 4u;
            enqueue_kernel(g61, fk.weight_fwd8_61, global, &local64, "enqueue defused edge fwd61", "crt_defused_weight_r4_fwd_r8_61");
            return;
        }
        if (use_generic_edge) {
            set_karg_mem(fk.weight_edge61, arg, g61.bufDigits, "set defused weight-edge61 digits");
            set_karg_mem(fk.weight_edge61, arg, g61.bufField, "set defused weight-edge61 field");
            set_karg_mem(fk.weight_edge61, arg, g61.bufTwFwd, "set defused weight-edge61 tw");
            set_karg(fk.weight_edge61, arg, p, "set defused weight-edge61 p");
            set_karg(fk.weight_edge61, arg, g61.lr2, "set defused weight-edge61 lr2");
            set_karg(fk.weight_edge61, arg, n, "set defused weight-edge61 n");
            set_karg(fk.weight_edge61, arg, edge_log, "set defused weight-edge61 log_radix");
            size_t global = round_up_size(static_cast<size_t>(n / edge_radix), local64);
            std::string lbl = "crt_defused_weight_first_r" + std::to_string(edge_radix) + "_61";
            enqueue_kernel(g61, fk.weight_edge61, global, &local64, "enqueue defused weight-edge61", lbl.c_str());
            return;
        }
        set_karg_mem(fk.weight_first61, arg, g61.bufDigits, "set defused weight61 digits");
        set_karg_mem(fk.weight_first61, arg, g61.bufField, "set defused weight61 field");
        set_karg_mem(fk.weight_first61, arg, g61.bufTwFwd, "set defused weight61 tw");
        set_karg(fk.weight_first61, arg, p, "set defused weight61 p");
        set_karg(fk.weight_first61, arg, g61.lr2, "set defused weight61 lr2");
        set_karg(fk.weight_first61, arg, n, "set defused weight61 n");
        set_karg(fk.weight_first61, arg, first.len, "set defused weight61 len");
        size_t global = n / 4u;
        enqueue_kernel(g61, fk.weight_first61, global, &local64, "enqueue defused weight61", "crt_defused_weight_first61");
    };

    auto enqueue_weight31 = [&]() {
        const StageInfo& first = stages.back();
        cl_uint arg = 0;
        if (fuse_forward_edge) {
            set_karg_mem(fk.weight_fwd8_31, arg, g61.bufDigits, "set defused edge fwd31 digits");
            set_karg_mem(fk.weight_fwd8_31, arg, g31.bufField, "set defused edge fwd31 field");
            set_karg_mem(fk.weight_fwd8_31, arg, g31.bufTwFwd, "set defused edge fwd31 tw");
            set_karg(fk.weight_fwd8_31, arg, p, "set defused edge fwd31 p");
            set_karg(fk.weight_fwd8_31, arg, g31.lr2, "set defused edge fwd31 lr2");
            set_karg(fk.weight_fwd8_31, arg, n, "set defused edge fwd31 n");
            size_t global = n / 4u;
            enqueue_kernel(g31, fk.weight_fwd8_31, global, &local64, "enqueue defused edge fwd31", "crt_defused_weight_r4_fwd_r8_31");
            return;
        }
        if (use_generic_edge) {
            set_karg_mem(fk.weight_edge31, arg, g61.bufDigits, "set defused weight-edge31 digits");
            set_karg_mem(fk.weight_edge31, arg, g31.bufField, "set defused weight-edge31 field");
            set_karg_mem(fk.weight_edge31, arg, g31.bufTwFwd, "set defused weight-edge31 tw");
            set_karg(fk.weight_edge31, arg, p, "set defused weight-edge31 p");
            set_karg(fk.weight_edge31, arg, g31.lr2, "set defused weight-edge31 lr2");
            set_karg(fk.weight_edge31, arg, n, "set defused weight-edge31 n");
            set_karg(fk.weight_edge31, arg, edge_log, "set defused weight-edge31 log_radix");
            size_t global = round_up_size(static_cast<size_t>(n / edge_radix), local64);
            std::string lbl = "crt_defused_weight_first_r" + std::to_string(edge_radix) + "_31";
            enqueue_kernel(g31, fk.weight_edge31, global, &local64, "enqueue defused weight-edge31", lbl.c_str());
            return;
        }
        set_karg_mem(fk.weight_first31, arg, g61.bufDigits, "set defused weight31 digits");
        set_karg_mem(fk.weight_first31, arg, g31.bufField, "set defused weight31 field");
        set_karg_mem(fk.weight_first31, arg, g31.bufTwFwd, "set defused weight31 tw");
        set_karg(fk.weight_first31, arg, p, "set defused weight31 p");
        set_karg(fk.weight_first31, arg, g31.lr2, "set defused weight31 lr2");
        set_karg(fk.weight_first31, arg, n, "set defused weight31 n");
        set_karg(fk.weight_first31, arg, first.len, "set defused weight31 len");
        size_t global = n / 4u;
        enqueue_kernel(g31, fk.weight_first31, global, &local64, "enqueue defused weight31", "crt_defused_weight_first31");
    };

    auto enqueue_forward_stage61 = [&](const StageInfo& st, cl_uint radix) {
        cl_uint arg = 0;
        if (radix == 8u) {
            const bool use_wg128 = (g_crt_fwd8_61_wg == 128) && fk.fwd_r8_61_wg128;
            cl_kernel k = use_wg128 ? fk.fwd_r8_61_wg128 : fk.fwd_r8_61;
            const size_t local_fwd8 = use_wg128 ? 128u : local64;
            set_karg_mem(k, arg, g61.bufField, "set defused fwd8 61 a");
            set_karg_mem(k, arg, g61.bufTwFwd, "set defused fwd8 61 tw");
            set_karg(k, arg, n, "set defused fwd8 61 n");
            set_karg(k, arg, st.len, "set defused fwd8 61 len");
            size_t global = n / 8u;
            std::string label;
            if (std::getenv("PRMERS_PROFILE_FWD8_LEN")) {
                label = std::string(use_wg128 ? "crt_defused_fwd_radix8_61_wg128_len" : "crt_defused_fwd_radix8_61_len") + std::to_string(st.len);
            } else {
                label = use_wg128 ? "crt_defused_fwd_radix8_61_wg128" : "crt_defused_fwd_radix8_61";
            }
            enqueue_kernel(g61, k, global, &local_fwd8, "enqueue defused fwd8 61", label.c_str());
        } else if (radix == 4u) {
            set_karg_mem(fk.fwd_r4_61, arg, g61.bufField, "set defused fwd4 61 a");
            set_karg_mem(fk.fwd_r4_61, arg, g61.bufTwFwd, "set defused fwd4 61 tw");
            set_karg(fk.fwd_r4_61, arg, n, "set defused fwd4 61 n");
            set_karg(fk.fwd_r4_61, arg, st.len, "set defused fwd4 61 len");
            size_t global = n / 4u;
            enqueue_kernel(g61, fk.fwd_r4_61, global, &local64, "enqueue defused fwd4 61", "crt_defused_fwd_radix4_61");
        } else {
            set_karg_mem(fk.fwd_r2_61, arg, g61.bufField, "set defused fwd2 61 a");
            set_karg_mem(fk.fwd_r2_61, arg, g61.bufTwFwd, "set defused fwd2 61 tw");
            set_karg(fk.fwd_r2_61, arg, n, "set defused fwd2 61 n");
            set_karg(fk.fwd_r2_61, arg, st.len, "set defused fwd2 61 len");
            size_t global = n / 2u;
            enqueue_kernel(g61, fk.fwd_r2_61, global, &local64, "enqueue defused fwd2 61", "crt_defused_fwd_radix2_61");
        }
    };

    auto enqueue_forward_stage31 = [&](const StageInfo& st, cl_uint radix) {
        cl_uint arg = 0;
        if (radix == 8u) {
            set_karg_mem(fk.fwd_r8_31, arg, g31.bufField, "set defused fwd8 31 a");
            set_karg_mem(fk.fwd_r8_31, arg, g31.bufTwFwd, "set defused fwd8 31 tw");
            set_karg(fk.fwd_r8_31, arg, n, "set defused fwd8 31 n");
            set_karg(fk.fwd_r8_31, arg, st.len, "set defused fwd8 31 len");
            size_t global = n / 8u;
            enqueue_kernel(g31, fk.fwd_r8_31, global, &local64, "enqueue defused fwd8 31", "crt_defused_fwd_radix8_31");
        } else if (radix == 4u) {
            set_karg_mem(fk.fwd_r4_31, arg, g31.bufField, "set defused fwd4 31 a");
            set_karg_mem(fk.fwd_r4_31, arg, g31.bufTwFwd, "set defused fwd4 31 tw");
            set_karg(fk.fwd_r4_31, arg, n, "set defused fwd4 31 n");
            set_karg(fk.fwd_r4_31, arg, st.len, "set defused fwd4 31 len");
            size_t global = n / 4u;
            enqueue_kernel(g31, fk.fwd_r4_31, global, &local64, "enqueue defused fwd4 31", "crt_defused_fwd_radix4_31");
        } else {
            set_karg_mem(fk.fwd_r2_31, arg, g31.bufField, "set defused fwd2 31 a");
            set_karg_mem(fk.fwd_r2_31, arg, g31.bufTwFwd, "set defused fwd2 31 tw");
            set_karg(fk.fwd_r2_31, arg, n, "set defused fwd2 31 n");
            set_karg(fk.fwd_r2_31, arg, st.len, "set defused fwd2 31 len");
            size_t global = n / 2u;
            enqueue_kernel(g31, fk.fwd_r2_31, global, &local64, "enqueue defused fwd2 31", "crt_defused_fwd_radix2_31");
        }
    };

    auto enqueue_forward_interleaved = [&]() {
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len > forward_edge_stop) continue;
            if (st.len <= global_stop_chunk) break;
            
            
            if (fuse_forward_edge && (st.len == (n >> 2) || st.len == (n >> 3) || st.len == (n >> 4))) continue;
            cl_uint radix = 2u;
            if ((st.len > 4u * global_stop_chunk) && idx >= 2) radix = 8u;
            else if ((st.len > 2u * global_stop_chunk) && idx >= 1) radix = 4u;
            enqueue_forward_stage61(st, radix);
            enqueue_forward_stage31(st, radix);
            if (radix == 8u) idx -= 2u;
            else if (radix == 4u) --idx;
        }
    };

    auto enqueue_forward_all61 = [&]() {
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len > forward_edge_stop) continue;
            if (st.len <= global_stop_chunk) break;
            
            
            if (fuse_forward_edge && (st.len == (n >> 2) || st.len == (n >> 3) || st.len == (n >> 4))) continue;
            cl_uint radix = 2u;
            if ((st.len > 4u * global_stop_chunk) && idx >= 2) radix = 8u;
            else if ((st.len > 2u * global_stop_chunk) && idx >= 1) radix = 4u;
            enqueue_forward_stage61(st, radix);
            if (radix == 8u) idx -= 2u;
            else if (radix == 4u) --idx;
        }
    };

    auto enqueue_forward_all31 = [&]() {
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len > forward_edge_stop) continue;
            if (st.len <= global_stop_chunk) break;
            
            
            if (fuse_forward_edge && (st.len == (n >> 2) || st.len == (n >> 3) || st.len == (n >> 4))) continue;
            cl_uint radix = 2u;
            if ((st.len > 4u * global_stop_chunk) && idx >= 2) radix = 8u;
            else if ((st.len > 2u * global_stop_chunk) && idx >= 1) radix = 4u;
            enqueue_forward_stage31(st, radix);
            if (radix == 8u) idx -= 2u;
            else if (radix == 4u) --idx;
        }
    };


    auto enqueue_forward_top_radix8_61 = [&]() -> bool {
        
        
        if (fuse_forward_edge) return false;
        const cl_uint top_len = n >> 2;
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len > forward_edge_stop) continue;
            if (st.len <= global_stop_chunk) break;
            if (st.len == top_len && (st.len > 4u * global_stop_chunk) && idx >= 2) {
                enqueue_forward_stage61(st, 8u);
                return true;
            }
        }
        return false;
    };

    auto enqueue_forward_after_top61 = [&]() {
        
        
        const cl_uint top0 = n >> 2;
        const cl_uint top1 = n >> 3;
        const cl_uint top2 = n >> 4;
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len > forward_edge_stop) continue;
            if (st.len <= global_stop_chunk) break;
            if (st.len == top0 || st.len == top1 || st.len == top2) continue;
            if (fuse_forward_edge && (st.len == top0 || st.len == top1 || st.len == top2)) continue;
            cl_uint radix = 2u;
            if ((st.len > 4u * global_stop_chunk) && idx >= 2) radix = 8u;
            else if ((st.len > 2u * global_stop_chunk) && idx >= 1) radix = 4u;
            enqueue_forward_stage61(st, radix);
            if (radix == 8u) idx -= 2u;
            else if (radix == 4u) --idx;
        }
    };

    auto enqueue_defused_lds_forward61 = [&]() {
        cl_kernel k = use_lds_tile4 ? fk.fwd_lds_stage_61_tile4 : (use_lds_stage512_opt ? fk.fwd_lds_stage_61_512opt : fk.fwd_lds_stage_61);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set defused lds fwd61 a");
        set_karg_mem(k, arg, g61.bufTwFwd, "set defused lds fwd61 tw");
        set_karg(k, arg, n, "set defused lds fwd61 n");
        set_karg(k, arg, global_stop_chunk, "set defused lds fwd61 len");
        set_karg(k, arg, defused_lds_radix, "set defused lds fwd61 radix");
        const size_t cols = global_stop_chunk / defused_lds_radix;
        const size_t groups_per_block = use_lds_tile4 ? ((cols + 3u) >> 2u) : cols;
        size_t global = (n / global_stop_chunk) * groups_per_block * local64;
        enqueue_kernel(g61, k, global, &local64, "enqueue defused LDS fwd 61",
                       use_lds_tile4 ? "crt_lds_stage512_forward_tile4_61" : (use_lds_stage512_opt ? "crt_lds_stage512_forward_opt_61" : crt_lds_stage_fwd_label(defused_lds_radix)));
    };
    auto enqueue_defused_lds_forward31 = [&]() {
        cl_kernel k = use_lds_tile4 ? fk.fwd_lds_stage_31_tile4 : (use_lds_stage512_opt ? fk.fwd_lds_stage_31_512opt : fk.fwd_lds_stage_31);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g31.bufField, "set defused lds fwd31 a");
        set_karg_mem(k, arg, g31.bufTwFwd, "set defused lds fwd31 tw");
        set_karg(k, arg, n, "set defused lds fwd31 n");
        set_karg(k, arg, global_stop_chunk, "set defused lds fwd31 len");
        set_karg(k, arg, defused_lds_radix, "set defused lds fwd31 radix");
        const size_t cols = global_stop_chunk / defused_lds_radix;
        const size_t groups_per_block = use_lds_tile4 ? ((cols + 3u) >> 2u) : cols;
        size_t global = (n / global_stop_chunk) * groups_per_block * local64;
        enqueue_kernel(g31, k, global, &local64, "enqueue defused LDS fwd 31",
                       use_lds_tile4 ? "crt_lds_stage512_forward_tile4_31" : (use_lds_stage512_opt ? "crt_lds_stage512_forward_opt_31" : crt_lds_stage_fwd_label(defused_lds_radix)));
    };

    auto enqueue_center61 = [&]() {
        cl_kernel k = use_center512_opt ? fk.center512_61_opt : fk.center512_61;
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set defused center61 a");
        set_karg_mem(k, arg, g61.bufTwFwd, "set defused center61 twf");
        set_karg_mem(k, arg, g61.bufTwInv, "set defused center61 twi");
        set_karg(k, arg, n, "set defused center61 n");
        size_t global = (n / 512u) * local64;
        enqueue_kernel(g61, k, global, &local64, "enqueue defused center61",
                       use_center512_opt ? "crt_defused_lds_square512_opt_61" : "crt_defused_lds_square512_61");
    };
    auto enqueue_center31 = [&]() {
        cl_kernel k = use_center512_opt ? fk.center512_31_opt : fk.center512_31;
        cl_uint arg = 0;
        set_karg_mem(k, arg, g31.bufField, "set defused center31 a");
        set_karg_mem(k, arg, g31.bufTwFwd, "set defused center31 twf");
        set_karg_mem(k, arg, g31.bufTwInv, "set defused center31 twi");
        set_karg(k, arg, n, "set defused center31 n");
        size_t global = (n / 512u) * local64;
        enqueue_kernel(g31, k, global, &local64, "enqueue defused center31",
                       use_center512_opt ? "crt_defused_lds_square512_opt_31" : "crt_defused_lds_square512_31");
    };

    auto enqueue_defused_lds_inverse61 = [&]() {
        cl_kernel k = use_lds_tile4 ? fk.inv_lds_stage_61_tile4 : (use_lds_stage512_opt ? fk.inv_lds_stage_61_512opt : fk.inv_lds_stage_61);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g61.bufField, "set defused lds inv61 a");
        set_karg_mem(k, arg, g61.bufTwInv, "set defused lds inv61 tw");
        set_karg(k, arg, n, "set defused lds inv61 n");
        set_karg(k, arg, center_chunk, "set defused lds inv61 base_len");
        set_karg(k, arg, defused_lds_radix, "set defused lds inv61 radix");
        const size_t cols = center_chunk;
        const size_t groups_per_block = use_lds_tile4 ? ((cols + 3u) >> 2u) : cols;
        size_t global = (n / global_stop_chunk) * groups_per_block * local64;
        enqueue_kernel(g61, k, global, &local64, "enqueue defused LDS inv 61",
                       use_lds_tile4 ? "crt_lds_stage512_inverse_tile4_61" : (use_lds_stage512_opt ? "crt_lds_stage512_inverse_opt_61" : crt_lds_stage_inv_label(defused_lds_radix)));
    };
    auto enqueue_defused_lds_inverse31 = [&]() {
        cl_kernel k = use_lds_tile4 ? fk.inv_lds_stage_31_tile4 : (use_lds_stage512_opt ? fk.inv_lds_stage_31_512opt : fk.inv_lds_stage_31);
        cl_uint arg = 0;
        set_karg_mem(k, arg, g31.bufField, "set defused lds inv31 a");
        set_karg_mem(k, arg, g31.bufTwInv, "set defused lds inv31 tw");
        set_karg(k, arg, n, "set defused lds inv31 n");
        set_karg(k, arg, center_chunk, "set defused lds inv31 base_len");
        set_karg(k, arg, defused_lds_radix, "set defused lds inv31 radix");
        const size_t cols = center_chunk;
        const size_t groups_per_block = use_lds_tile4 ? ((cols + 3u) >> 2u) : cols;
        size_t global = (n / global_stop_chunk) * groups_per_block * local64;
        enqueue_kernel(g31, k, global, &local64, "enqueue defused LDS inv 31",
                       use_lds_tile4 ? "crt_lds_stage512_inverse_tile4_31" : (use_lds_stage512_opt ? "crt_lds_stage512_inverse_opt_31" : crt_lds_stage_inv_label(defused_lds_radix)));
    };

    auto enqueue_inverse_stage61 = [&](const StageInfo& st, cl_uint radix) {
        cl_uint arg = 0;
        if (radix == 8u) {
            set_karg_mem(fk.inv_r8_61, arg, g61.bufField, "set defused inv8 61 a");
            set_karg_mem(fk.inv_r8_61, arg, g61.bufTwInv, "set defused inv8 61 tw");
            set_karg(fk.inv_r8_61, arg, n, "set defused inv8 61 n");
            set_karg(fk.inv_r8_61, arg, st.len, "set defused inv8 61 len");
            size_t global = n / 8u;
            enqueue_kernel(g61, fk.inv_r8_61, global, &local64, "enqueue defused inv8 61", "crt_defused_inv_radix8_61");
        } else if (radix == 4u) {
            set_karg_mem(fk.inv_r4_61, arg, g61.bufField, "set defused inv4 61 a");
            set_karg_mem(fk.inv_r4_61, arg, g61.bufTwInv, "set defused inv4 61 tw");
            set_karg(fk.inv_r4_61, arg, n, "set defused inv4 61 n");
            set_karg(fk.inv_r4_61, arg, st.len, "set defused inv4 61 len");
            size_t global = n / 4u;
            enqueue_kernel(g61, fk.inv_r4_61, global, &local64, "enqueue defused inv4 61", "crt_defused_inv_radix4_61");
        } else {
            set_karg_mem(fk.inv_r2_61, arg, g61.bufField, "set defused inv2 61 a");
            set_karg_mem(fk.inv_r2_61, arg, g61.bufTwInv, "set defused inv2 61 tw");
            set_karg(fk.inv_r2_61, arg, n, "set defused inv2 61 n");
            set_karg(fk.inv_r2_61, arg, st.len, "set defused inv2 61 len");
            size_t global = n / 2u;
            enqueue_kernel(g61, fk.inv_r2_61, global, &local64, "enqueue defused inv2 61", "crt_defused_inv_radix2_61");
        }
    };

    auto enqueue_inverse_stage31 = [&](const StageInfo& st, cl_uint radix) {
        cl_uint arg = 0;
        if (radix == 8u) {
            set_karg_mem(fk.inv_r8_31, arg, g31.bufField, "set defused inv8 31 a");
            set_karg_mem(fk.inv_r8_31, arg, g31.bufTwInv, "set defused inv8 31 tw");
            set_karg(fk.inv_r8_31, arg, n, "set defused inv8 31 n");
            set_karg(fk.inv_r8_31, arg, st.len, "set defused inv8 31 len");
            size_t global = n / 8u;
            enqueue_kernel(g31, fk.inv_r8_31, global, &local64, "enqueue defused inv8 31", "crt_defused_inv_radix8_31");
        } else if (radix == 4u) {
            set_karg_mem(fk.inv_r4_31, arg, g31.bufField, "set defused inv4 31 a");
            set_karg_mem(fk.inv_r4_31, arg, g31.bufTwInv, "set defused inv4 31 tw");
            set_karg(fk.inv_r4_31, arg, n, "set defused inv4 31 n");
            set_karg(fk.inv_r4_31, arg, st.len, "set defused inv4 31 len");
            size_t global = n / 4u;
            enqueue_kernel(g31, fk.inv_r4_31, global, &local64, "enqueue defused inv4 31", "crt_defused_inv_radix4_31");
        } else {
            set_karg_mem(fk.inv_r2_31, arg, g31.bufField, "set defused inv2 31 a");
            set_karg_mem(fk.inv_r2_31, arg, g31.bufTwInv, "set defused inv2 31 tw");
            set_karg(fk.inv_r2_31, arg, n, "set defused inv2 31 n");
            set_karg(fk.inv_r2_31, arg, st.len, "set defused inv2 31 len");
            size_t global = n / 2u;
            enqueue_kernel(g31, fk.inv_r2_31, global, &local64, "enqueue defused inv2 31", "crt_defused_inv_radix2_31");
        }
    };

    auto enqueue_inverse_interleaved = [&]() {
        for (std::size_t si = 0; si < stages.size(); ++si) {
            const StageInfo& st = stages[si];
            if (st.len <= global_stop_chunk) continue;
            if (st.len >= inverse_stop) break;
            
            
            if (fuse_tail_edge && (st.len == (n >> 3) || st.len == (n >> 2))) continue;
            cl_uint radix = 2u;
            if ((st.len * 4u < inverse_stop) && (st.len >= global_stop_chunk)) radix = 8u;
            else if (st.len * 2u < inverse_stop) radix = 4u;
            enqueue_inverse_stage61(st, radix);
            enqueue_inverse_stage31(st, radix);
            if (radix == 8u) si += 2u;
            else if (radix == 4u) ++si;
        }
    };

    auto enqueue_inverse_all61 = [&]() {
        for (std::size_t si = 0; si < stages.size(); ++si) {
            const StageInfo& st = stages[si];
            if (st.len <= global_stop_chunk) continue;
            if (st.len >= inverse_stop) break;
            
            
            if (fuse_tail_edge && (st.len == (n >> 3) || st.len == (n >> 2))) continue;
            cl_uint radix = 2u;
            if ((st.len * 4u < inverse_stop) && (st.len >= global_stop_chunk)) radix = 8u;
            else if (st.len * 2u < inverse_stop) radix = 4u;
            enqueue_inverse_stage61(st, radix);
            if (radix == 8u) si += 2u;
            else if (radix == 4u) ++si;
        }
    };

    auto enqueue_inverse_all31 = [&]() {
        for (std::size_t si = 0; si < stages.size(); ++si) {
            const StageInfo& st = stages[si];
            if (st.len <= global_stop_chunk) continue;
            if (st.len >= inverse_stop) break;
            
            
            if (fuse_tail_edge && (st.len == (n >> 3) || st.len == (n >> 2))) continue;
            cl_uint radix = 2u;
            if ((st.len * 4u < inverse_stop) && (st.len >= global_stop_chunk)) radix = 8u;
            else if (st.len * 2u < inverse_stop) radix = 4u;
            enqueue_inverse_stage31(st, radix);
            if (radix == 8u) si += 2u;
            else if (radix == 4u) ++si;
        }
    };

    auto enqueue_last61 = [&]() {
        const StageInfo& st = stages[stages.size() - 2];
        cl_uint arg = 0;
        if (fuse_tail_edge) {
            cl_kernel k_tail61 = use_block16_tail ? fk.inv4_last_unweight16_61 : fk.inv4_last_unweight61;
            set_karg_mem(k_tail61, arg, g61.bufDigits, "set defused edge tail61 digits");
            set_karg_mem(k_tail61, arg, g61.bufField, "set defused edge tail61 field");
            set_karg_mem(k_tail61, arg, g61.bufTwInv, "set defused edge tail61 tw");
            set_karg(k_tail61, arg, p, "set defused edge tail61 p");
            set_karg(k_tail61, arg, g61.lr2, "set defused edge tail61 lr2");
            set_karg(k_tail61, arg, n, "set defused edge tail61 n");
            set_karg(k_tail61, arg, log_n, "set defused edge tail61 logn");
            size_t global = use_block16_tail ? (n / 16u) : (n / 4u);
            enqueue_kernel(g61, k_tail61, global, &local64, "enqueue defused edge tail61",
                           use_block16_tail ? "crt_defused_inv_r4_last_unweight16_61" : "crt_defused_inv_r4_last_unweight61");
            return;
        }
        if (use_generic_edge) {
            set_karg_mem(fk.last_unweight_edge61, arg, g61.bufDigits, "set defused last-edge61 digits");
            set_karg_mem(fk.last_unweight_edge61, arg, g61.bufField, "set defused last-edge61 field");
            set_karg_mem(fk.last_unweight_edge61, arg, g61.bufTwInv, "set defused last-edge61 tw");
            set_karg(fk.last_unweight_edge61, arg, p, "set defused last-edge61 p");
            set_karg(fk.last_unweight_edge61, arg, g61.lr2, "set defused last-edge61 lr2");
            set_karg(fk.last_unweight_edge61, arg, n, "set defused last-edge61 n");
            set_karg(fk.last_unweight_edge61, arg, log_n, "set defused last-edge61 logn");
            set_karg(fk.last_unweight_edge61, arg, edge_log, "set defused last-edge61 log_radix");
            size_t global = round_up_size(static_cast<size_t>(n / edge_radix), local64);
            std::string lbl = "crt_defused_last_unweight_r" + std::to_string(edge_radix) + "_61";
            enqueue_kernel(g61, fk.last_unweight_edge61, global, &local64, "enqueue defused last-edge61", lbl.c_str());
            return;
        }
        set_karg_mem(fk.last_unweight61, arg, g61.bufDigits, "set defused last61 digits");
        set_karg_mem(fk.last_unweight61, arg, g61.bufField, "set defused last61 a");
        set_karg_mem(fk.last_unweight61, arg, g61.bufTwInv, "set defused last61 tw");
        set_karg(fk.last_unweight61, arg, p, "set defused last61 p");
        set_karg(fk.last_unweight61, arg, g61.lr2, "set defused last61 lr2");
        set_karg(fk.last_unweight61, arg, n, "set defused last61 n");
        set_karg(fk.last_unweight61, arg, st.len, "set defused last61 len");
        set_karg(fk.last_unweight61, arg, log_n, "set defused last61 logn");
        size_t global = n / 4u;
        enqueue_kernel(g61, fk.last_unweight61, global, &local64, "enqueue defused last61", "crt_defused_last_unweight61");
    };
    auto enqueue_last31 = [&]() {
        const StageInfo& st = stages[stages.size() - 2];
        cl_mem digits31 = g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits;
        cl_uint arg = 0;
        if (fuse_tail_edge) {
            cl_kernel k_tail31 = use_block16_tail ? fk.inv4_last_unweight16_31 : fk.inv4_last_unweight31;
            set_karg_mem(k_tail31, arg, digits31, "set defused edge tail31 digits");
            set_karg_mem(k_tail31, arg, g31.bufField, "set defused edge tail31 field");
            set_karg_mem(k_tail31, arg, g31.bufTwInv, "set defused edge tail31 tw");
            set_karg(k_tail31, arg, p, "set defused edge tail31 p");
            set_karg(k_tail31, arg, g31.lr2, "set defused edge tail31 lr2");
            set_karg(k_tail31, arg, n, "set defused edge tail31 n");
            set_karg(k_tail31, arg, log_n, "set defused edge tail31 logn");
            size_t global = use_block16_tail ? (n / 16u) : (n / 4u);
            enqueue_kernel(g31, k_tail31, global, &local64, "enqueue defused edge tail31",
                           use_block16_tail ? "crt_defused_inv_r4_last_unweight16_31" : "crt_defused_inv_r4_last_unweight31");
            return;
        }
        if (use_generic_edge) {
            set_karg_mem(fk.last_unweight_edge31, arg, digits31, "set defused last-edge31 digits");
            set_karg_mem(fk.last_unweight_edge31, arg, g31.bufField, "set defused last-edge31 field");
            set_karg_mem(fk.last_unweight_edge31, arg, g31.bufTwInv, "set defused last-edge31 tw");
            set_karg(fk.last_unweight_edge31, arg, p, "set defused last-edge31 p");
            set_karg(fk.last_unweight_edge31, arg, g31.lr2, "set defused last-edge31 lr2");
            set_karg(fk.last_unweight_edge31, arg, n, "set defused last-edge31 n");
            set_karg(fk.last_unweight_edge31, arg, log_n, "set defused last-edge31 logn");
            set_karg(fk.last_unweight_edge31, arg, edge_log, "set defused last-edge31 log_radix");
            size_t global = round_up_size(static_cast<size_t>(n / edge_radix), local64);
            std::string lbl = "crt_defused_last_unweight_r" + std::to_string(edge_radix) + "_31";
            enqueue_kernel(g31, fk.last_unweight_edge31, global, &local64, "enqueue defused last-edge31", lbl.c_str());
            return;
        }
        set_karg_mem(fk.last_unweight31, arg, digits31, "set defused last31 digits");
        set_karg_mem(fk.last_unweight31, arg, g31.bufField, "set defused last31 a");
        set_karg_mem(fk.last_unweight31, arg, g31.bufTwInv, "set defused last31 tw");
        set_karg(fk.last_unweight31, arg, p, "set defused last31 p");
        set_karg(fk.last_unweight31, arg, g31.lr2, "set defused last31 lr2");
        set_karg(fk.last_unweight31, arg, n, "set defused last31 n");
        set_karg(fk.last_unweight31, arg, st.len, "set defused last31 len");
        set_karg(fk.last_unweight31, arg, log_n, "set defused last31 logn");
        size_t global = n / 4u;
        enqueue_kernel(g31, fk.last_unweight31, global, &local64, "enqueue defused last31", "crt_defused_last_unweight31");
    };

    auto enqueue_last_crtcoeff_block16 = [&]() {
        cl_mem coeff_hi = g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits;
        cl_uint arg = 0;
        set_karg_mem(fk.inv4_last_unweight16_crtcoeff, arg, g61.bufDigits, "set precrt tail coeff_lo");
        set_karg_mem(fk.inv4_last_unweight16_crtcoeff, arg, coeff_hi, "set precrt tail coeff_hi");
        set_karg_mem(fk.inv4_last_unweight16_crtcoeff, arg, g61.bufField, "set precrt tail a61");
        set_karg_mem(fk.inv4_last_unweight16_crtcoeff, arg, g61.bufTwInv, "set precrt tail tw61");
        set_karg_mem(fk.inv4_last_unweight16_crtcoeff, arg, g31.bufField, "set precrt tail a31");
        set_karg_mem(fk.inv4_last_unweight16_crtcoeff, arg, g31.bufTwInv, "set precrt tail tw31");
        set_karg(fk.inv4_last_unweight16_crtcoeff, arg, p, "set precrt tail p");
        set_karg(fk.inv4_last_unweight16_crtcoeff, arg, g61.lr2, "set precrt tail lr2_61");
        set_karg(fk.inv4_last_unweight16_crtcoeff, arg, g31.lr2, "set precrt tail lr2_31");
        set_karg(fk.inv4_last_unweight16_crtcoeff, arg, n, "set precrt tail n");
        set_karg(fk.inv4_last_unweight16_crtcoeff, arg, log_n, "set precrt tail logn");
        size_t global = n / 16u;
        enqueue_kernel(g61, fk.inv4_last_unweight16_crtcoeff, global, &local64,
                       "enqueue defused precrt block16 tail",
                       "crt_defused_inv_r4_last_unweight16_crtcoeff");
        g61.crtCoeffPending = true;
    };


    if (g_crt_center_mode == "halfreal") {
        if (!fk.halfreal_pack61 || !fk.halfreal_pack31 || !fk.halfreal_center61 || !fk.halfreal_center31 ||
            !fk.halfreal_unpack61 || !fk.halfreal_unpack31) {
            throw std::runtime_error("halfreal CRT kernels are missing from the OpenCL program");
        }
        if (n < 2u || (n & 1u)) throw std::runtime_error("halfreal CRT mode requires an even transform length");
        const uint32_t m = n >> 1;
        const uint32_t logm = log_n - 1u;
        const size_t g_m64 = round_up_size((size_t)m, local64);
        
        
        const size_t g_center64 = round_up_size((size_t)(m / 2u + 1u), local64);
        const cl_uint half_flags61 = crt_halfreal_effective_flags61();
        const cl_uint half_flags31 = crt_halfreal_effective_flags31();
        const bool half_flags_are_linear = (((half_flags61 | half_flags31) & (1u | 16u)) == 0u);
        const bool half_linear_center_default = half_flags_are_linear;
        const bool half_linear_center = parse_bool_env("PRMERS_CRT_HALFREAL_LINEAR_CENTER", half_linear_center_default);
        if (half_linear_center && !half_flags_are_linear) {
            throw std::runtime_error("PRMERS_CRT_HALFREAL_LINEAR_CENTER requires flags without digitrev/bitrev bits (use e.g. --crt-halfreal-flags 32, not 48)");
        }
        if (half_linear_center && (!fk.halfreal_bitrev_swap61 || !fk.halfreal_bitrev_swap31)) {
            throw std::runtime_error("halfreal linear-center requested but bitrev-swap kernels are missing");
        }

        
        const bool half_lds_hotopt = parse_bool_env("PRMERS_CRT_HALFREAL_LDS_HOTOPT", false);
        const bool half_use_lds512 = (m >= 512u) && (g_crt_lds_stage >= 512u) &&
            fk.fwd_lds_stage_61 && fk.inv_lds_stage_61 &&
            fk.fwd_lds_stage_31 && fk.inv_lds_stage_31;
        const bool half_use_lds512_opt = half_use_lds512 && half_lds_hotopt &&
            fk.fwd_lds_stage_61_512opt && fk.inv_lds_stage_61_512opt &&
            fk.fwd_lds_stage_31_512opt && fk.inv_lds_stage_31_512opt;
        const bool half_head_lds_req = parse_bool_env("PRMERS_CRT_HALFREAL_HEAD_LDS512", true);
        const char* half_head_env_s = std::getenv("PRMERS_CRT_HALFREAL_HEAD_LDS");
        const bool half_head_exact = (half_head_env_s && *half_head_env_s);
        const uint32_t half_head_max_raw = crt_tune::env_u32("PRMERS_CRT_HALFREAL_HEAD_LDS_MAX", 1024u, 8u, 1024u);
        const uint32_t half_head_max = crt_tune::round_pow2_clamped(half_head_max_raw, 8u, 1024u);
        const uint32_t half_head_default_raw = (g_crt_lds_stage >= 512u) ? 1024u : ((g_crt_lds_stage >= 8u) ? g_crt_lds_stage : 512u);
        const uint32_t half_head_default = half_head_exact ? half_head_default_raw : std::min<uint32_t>(half_head_default_raw, half_head_max);
        uint32_t half_head_env = crt_tune::env_u32("PRMERS_CRT_HALFREAL_HEAD_LDS", half_head_default, 0u, 1024u);
        if (!half_head_exact && half_head_env > half_head_max) half_head_env = half_head_max;
        auto half_valid_head_radix = [&](uint32_t r) -> bool {
            return r == 1024u || r == 512u || r == 256u || r == 128u || r == 64u || r == 32u || r == 16u || r == 8u;
        };
        const bool half_force_generic_head = parse_bool_env("PRMERS_CRT_HALFREAL_HEAD_LDS_GENERIC", false);

        
        const bool half_strict_residual = parse_bool_env("PRMERS_CRT_HALFREAL_STRICT_RESIDUAL", false);
        const bool half_allow_mixed_residual = parse_bool_env("PRMERS_CRT_HALFREAL_ALLOW_MIXED_RESIDUAL", !half_strict_residual);
        auto half_residual_is_radix8_clean = [&](uint32_t r) -> bool {
            if (!half_use_lds512) return true;
            if (r == 0u || (m % r) != 0u) return false;
            const uint32_t head_base = m / r;
            if (head_base < 512u || (head_base % 512u) != 0u) return false;
            uint32_t residual = head_base / 512u;
            while (residual > 1u && (residual % 8u) == 0u) residual /= 8u;
            return residual == 1u;
        };
        auto half_head_can_use = [&](uint32_t r) -> bool {
            if (!half_valid_head_radix(r)) return false;
            const bool m_ok = (m >= r * 512u) && ((m % r) == 0u);
            const bool residual_ok = half_allow_mixed_residual || half_residual_is_radix8_clean(r);
            const bool have_special_512 = (r == 512u) && !half_force_generic_head &&
                fk.halfreal_head_lds512_61 && fk.halfreal_head_lds512_31 &&
                fk.halfreal_tail_lds512_unpack61 && fk.halfreal_tail_lds512_unpack31;
            const bool have_generic = fk.halfreal_head_ldspow2_61 && fk.halfreal_head_ldspow2_31 &&
                fk.halfreal_tail_ldspow2_unpack61 && fk.halfreal_tail_ldspow2_unpack31;
            return m_ok && residual_ok && (have_special_512 || have_generic);
        };

        uint32_t half_head_tail_radix = 0u;
        if (half_head_lds_req && g_crt_lds_stage >= 8u && half_head_env != 0u) {
            uint32_t r = half_valid_head_radix(half_head_env) ? half_head_env : half_head_default;
            if (half_head_exact) {
                half_head_tail_radix = half_head_can_use(r) ? r : 0u;
            } else {
                if (!half_valid_head_radix(r)) r = 512u;
                while (r >= 8u) {
                    if (half_head_can_use(r)) { half_head_tail_radix = r; break; }
                    r >>= 1u;
                }
            }
        }
        const bool half_disable_pow2_headtail = parse_bool_env("PRMERS_CRT_HALFREAL_DISABLE_POW2_HEADTAIL", false);
        const bool half_force_generic_headtail = parse_bool_env("PRMERS_CRT_HALFREAL_FORCE_GENERIC_HEADTAIL", false);
        
        
        if (half_head_tail_radix != 0u && half_disable_pow2_headtail) {
            half_head_tail_radix = 0u;
        }
        const bool half_use_head_lds = (half_head_tail_radix != 0u);
        const bool half_head_tail_use_special512 = half_use_head_lds && half_head_tail_radix == 512u &&
            !half_force_generic_head && fk.halfreal_head_lds512_61 && fk.halfreal_head_lds512_31 &&
            fk.halfreal_tail_lds512_unpack61 && fk.halfreal_tail_lds512_unpack31;
        const bool half_tail_precrt = half_head_tail_use_special512 &&
            parse_bool_env("PRMERS_CRT_HALFREAL_TAIL_PRECRT", true) &&
            fk.halfreal_tail_lds512_precrt && g61.min_digit_width == 32u && g61.bufWidthMask32 &&
            g61.k_crt_garner_segment_first_oneout_coeffhi_mask32_base32;
        const bool half_head_precrt = half_head_tail_use_special512 &&
            parse_bool_env("PRMERS_CRT_HALFREAL_HEAD_PRECRT", false) &&
            fk.halfreal_head_lds512_precrt;
        const uint32_t half_head_base_len = half_use_head_lds ? (m / half_head_tail_radix) : m;
        (void)half_force_generic_headtail;

        
        const bool half_lds_pair_req = parse_bool_env(
            "PRMERS_CRT_HALFREAL_LDS_PAIR",
            half_head_tail_use_special512);

        const bool half_lds_split_req = half_lds_pair_req
            ? false
            : parse_bool_env("PRMERS_CRT_HALFREAL_LDS_SPLIT", half_use_head_lds);

        
        const bool half_allow_lds_without_head = parse_bool_env("PRMERS_CRT_HALFREAL_LDS_NOHEAD", false) ||
            parse_bool_env("PRMERS_CRT_HALFREAL_LDS_SPLIT_NOHEAD", false);
        const bool half_local_layout_safe = half_use_head_lds || half_allow_lds_without_head;
        const bool half_use_mid_lds512 = half_use_lds512 && half_local_layout_safe;
        const bool half_use_mid_lds512_opt = half_use_mid_lds512 && half_use_lds512_opt;

        const bool half_use_lds512_pair = half_use_mid_lds512 && half_lds_pair_req && half_local_layout_safe && !half_linear_center &&
            fk.halfreal_lds512_pair61 && fk.halfreal_lds512_pair31;

        const bool half_use_lds512_split = half_use_mid_lds512 && half_lds_split_req && !half_use_lds512_pair && half_local_layout_safe && !half_linear_center &&
            fk.halfreal_center512_pair61 && fk.halfreal_center512_pair31;
        const uint32_t half_lds_radix = half_use_mid_lds512 ? 512u : 0u;
        const uint32_t half_center_chunk = half_use_mid_lds512 ? 1u : 0u;
        const uint32_t half_global_stop_chunk = half_use_mid_lds512 ? 512u : 0u;

        auto pack61 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_pack61, arg, g61.bufDigits, "set half pack61 digits");
            set_karg_mem(fk.halfreal_pack61, arg, g61.bufField, "set half pack61 a");
            set_karg(fk.halfreal_pack61, arg, n, "set half pack61 n");
            set_karg(fk.halfreal_pack61, arg, p, "set half pack61 p");
            set_karg(fk.halfreal_pack61, arg, g61.lr2, "set half pack61 lr2");
            enqueue_kernel(g61, fk.halfreal_pack61, g_m64, &local64, "enqueue half pack61", "crt_halfreal_pack_weight_61");
        };
        auto pack31 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_pack31, arg, g61.bufDigits, "set half pack31 digits");
            set_karg_mem(fk.halfreal_pack31, arg, g31.bufField, "set half pack31 a");
            set_karg(fk.halfreal_pack31, arg, n, "set half pack31 n");
            set_karg(fk.halfreal_pack31, arg, p, "set half pack31 p");
            set_karg(fk.halfreal_pack31, arg, g31.lr2, "set half pack31 lr2");
            enqueue_kernel(g31, fk.halfreal_pack31, g_m64, &local64, "enqueue half pack31", "crt_halfreal_pack_weight_31");
        };
        auto center61 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_center61, arg, g61.bufField, "set half center61 a");
            set_karg_mem(fk.halfreal_center61, arg, g61.bufTwFwd, "set half center61 twf");
            set_karg_mem(fk.halfreal_center61, arg, g61.bufTwInv, "set half center61 twi");
            set_karg(fk.halfreal_center61, arg, n, "set half center61 n");
            set_karg(fk.halfreal_center61, arg, half_flags61, "set half center61 flags");
            enqueue_kernel(g61, fk.halfreal_center61, g_center64, &local64, "enqueue half center61", "crt_halfreal_center_61");
        };
        auto center31 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_center31, arg, g31.bufField, "set half center31 a");
            set_karg_mem(fk.halfreal_center31, arg, g31.bufTwFwd, "set half center31 twf");
            set_karg_mem(fk.halfreal_center31, arg, g31.bufTwInv, "set half center31 twi");
            set_karg(fk.halfreal_center31, arg, n, "set half center31 n");
            set_karg(fk.halfreal_center31, arg, half_flags31, "set half center31 flags");
            enqueue_kernel(g31, fk.halfreal_center31, g_center64, &local64, "enqueue half center31", "crt_halfreal_center_31");
        };
        auto bitrev_swap61 = [&](const char* profile_name) {
            if (!half_linear_center) return;
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_bitrev_swap61, arg, g61.bufField, "set half bitrev swap61 a");
            set_karg(fk.halfreal_bitrev_swap61, arg, m, "set half bitrev swap61 m");
            enqueue_kernel(g61, fk.halfreal_bitrev_swap61, g_m64, &local64, "enqueue half bitrev swap61", profile_name);
        };
        auto bitrev_swap31 = [&](const char* profile_name) {
            if (!half_linear_center) return;
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_bitrev_swap31, arg, g31.bufField, "set half bitrev swap31 a");
            set_karg(fk.halfreal_bitrev_swap31, arg, m, "set half bitrev swap31 m");
            enqueue_kernel(g31, fk.halfreal_bitrev_swap31, g_m64, &local64, "enqueue half bitrev swap31", profile_name);
        };

        auto unpack61 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_unpack61, arg, g61.bufField, "set half unpack61 a");
            set_karg_mem(fk.halfreal_unpack61, arg, g61.bufDigits, "set half unpack61 digits");
            set_karg(fk.halfreal_unpack61, arg, n, "set half unpack61 n");
            set_karg(fk.halfreal_unpack61, arg, p, "set half unpack61 p");
            set_karg(fk.halfreal_unpack61, arg, g61.lr2, "set half unpack61 lr2");
            set_karg(fk.halfreal_unpack61, arg, logm, "set half unpack61 logm");
            enqueue_kernel(g61, fk.halfreal_unpack61, g_m64, &local64, "enqueue half unpack61", "crt_halfreal_unpack_unweight_61");
        };
        auto unpack31 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_unpack31, arg, g31.bufField, "set half unpack31 a");
            set_karg_mem(fk.halfreal_unpack31, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set half unpack31 digits31");
            set_karg(fk.halfreal_unpack31, arg, n, "set half unpack31 n");
            set_karg(fk.halfreal_unpack31, arg, p, "set half unpack31 p");
            set_karg(fk.halfreal_unpack31, arg, g31.lr2, "set half unpack31 lr2");
            set_karg(fk.halfreal_unpack31, arg, logm, "set half unpack31 logm");
            enqueue_kernel(g31, fk.halfreal_unpack31, g_m64, &local64, "enqueue half unpack31", "crt_halfreal_unpack_unweight_31");
        };

        auto head_lds512_pack_fwd61 = [&]() {
            cl_kernel k = half_head_tail_use_special512 ? fk.halfreal_head_lds512_61 : fk.halfreal_head_ldspow2_61;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufDigits, "set half head LDS pack61 digits");
            set_karg_mem(k, arg, g61.bufField, "set half head LDS pack61 a");
            set_karg_mem(k, arg, g61.bufTwFwd, "set half head LDS pack61 tw");
            set_karg(k, arg, n, "set half head LDS pack61 n");
            set_karg(k, arg, p, "set half head LDS pack61 p");
            set_karg(k, arg, g61.lr2, "set half head LDS pack61 lr2");
            if (!half_head_tail_use_special512) set_karg(k, arg, half_head_tail_radix, "set half head LDS pack61 radix");
            const size_t global = (size_t)(m / half_head_tail_radix) * local64;
            const std::string prof = std::string("crt_halfreal_head_pack_lds") + std::to_string(half_head_tail_radix) + "_61";
            enqueue_kernel(g61, k, global, &local64, "enqueue half head LDS pack61", prof.c_str());
        };
        auto head_lds512_pack_fwd31 = [&]() {
            cl_kernel k = half_head_tail_use_special512 ? fk.halfreal_head_lds512_31 : fk.halfreal_head_ldspow2_31;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufDigits, "set half head LDS pack31 digits");
            set_karg_mem(k, arg, g31.bufField, "set half head LDS pack31 a");
            set_karg_mem(k, arg, g31.bufTwFwd, "set half head LDS pack31 tw");
            set_karg(k, arg, n, "set half head LDS pack31 n");
            set_karg(k, arg, p, "set half head LDS pack31 p");
            set_karg(k, arg, g31.lr2, "set half head LDS pack31 lr2");
            if (!half_head_tail_use_special512) set_karg(k, arg, half_head_tail_radix, "set half head LDS pack31 radix");
            const size_t global = (size_t)(m / half_head_tail_radix) * local64;
            const std::string prof = std::string("crt_halfreal_head_pack_lds") + std::to_string(half_head_tail_radix) + "_31";
            enqueue_kernel(g31, k, global, &local64, "enqueue half head LDS pack31", prof.c_str());
        };
        auto head_lds512_pack_fwd_precrt = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_head_lds512_precrt, arg, g61.bufDigits, "set half head precrt digits");
            set_karg_mem(fk.halfreal_head_lds512_precrt, arg, g61.bufField, "set half head precrt a61");
            set_karg_mem(fk.halfreal_head_lds512_precrt, arg, g61.bufTwFwd, "set half head precrt tw61");
            set_karg_mem(fk.halfreal_head_lds512_precrt, arg, g31.bufField, "set half head precrt a31");
            set_karg_mem(fk.halfreal_head_lds512_precrt, arg, g31.bufTwFwd, "set half head precrt tw31");
            set_karg(fk.halfreal_head_lds512_precrt, arg, n, "set half head precrt n");
            set_karg(fk.halfreal_head_lds512_precrt, arg, p, "set half head precrt p");
            set_karg(fk.halfreal_head_lds512_precrt, arg, g61.lr2, "set half head precrt lr2_61");
            set_karg(fk.halfreal_head_lds512_precrt, arg, g31.lr2, "set half head precrt lr2_31");
            const size_t global = (size_t)(m / 512u) * local64;
            enqueue_kernel(g61, fk.halfreal_head_lds512_precrt, global, &local64,
                           "enqueue half head LDS precrt", "crt_halfreal_head_pack_lds512_precrt");

            
            cl_event head_done = enqueue_queue_marker(g61, "half head precrt ready for gf31");
            set_pending_wait_event(g31, head_done);
        };
        auto tail_lds512_inv_unpack61 = [&]() {
            cl_kernel k = half_head_tail_use_special512 ? fk.halfreal_tail_lds512_unpack61 : fk.halfreal_tail_ldspow2_unpack61;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set half tail LDS unpack61 a");
            set_karg_mem(k, arg, g61.bufTwInv, "set half tail LDS unpack61 tw");
            set_karg_mem(k, arg, g61.bufDigits, "set half tail LDS unpack61 digits");
            set_karg(k, arg, n, "set half tail LDS unpack61 n");
            set_karg(k, arg, p, "set half tail LDS unpack61 p");
            set_karg(k, arg, g61.lr2, "set half tail LDS unpack61 lr2");
            set_karg(k, arg, logm, "set half tail LDS unpack61 logm");
            if (!half_head_tail_use_special512) set_karg(k, arg, half_head_tail_radix, "set half tail LDS unpack61 radix");
            const size_t global = (size_t)(m / half_head_tail_radix) * local64;
            const std::string prof = std::string("crt_halfreal_tail_lds") + std::to_string(half_head_tail_radix) + "_unpack_61";
            enqueue_kernel(g61, k, global, &local64, "enqueue half tail LDS unpack61", prof.c_str());
        };
        auto tail_lds512_inv_unpack31 = [&]() {
            cl_kernel k = half_head_tail_use_special512 ? fk.halfreal_tail_lds512_unpack31 : fk.halfreal_tail_ldspow2_unpack31;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set half tail LDS unpack31 a");
            set_karg_mem(k, arg, g31.bufTwInv, "set half tail LDS unpack31 tw");
            set_karg_mem(k, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set half tail LDS unpack31 digits");
            set_karg(k, arg, n, "set half tail LDS unpack31 n");
            set_karg(k, arg, p, "set half tail LDS unpack31 p");
            set_karg(k, arg, g31.lr2, "set half tail LDS unpack31 lr2");
            set_karg(k, arg, logm, "set half tail LDS unpack31 logm");
            if (!half_head_tail_use_special512) set_karg(k, arg, half_head_tail_radix, "set half tail LDS unpack31 radix");
            const size_t global = (size_t)(m / half_head_tail_radix) * local64;
            const std::string prof = std::string("crt_halfreal_tail_lds") + std::to_string(half_head_tail_radix) + "_unpack_31";
            enqueue_kernel(g31, k, global, &local64, "enqueue half tail LDS unpack31", prof.c_str());
        };
        auto tail_lds512_inv_unpack_precrt = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_tail_lds512_precrt, arg, g61.bufField, "set half tail precrt a61");
            set_karg_mem(fk.halfreal_tail_lds512_precrt, arg, g61.bufTwInv, "set half tail precrt tw61");
            set_karg_mem(fk.halfreal_tail_lds512_precrt, arg, g31.bufField, "set half tail precrt a31");
            set_karg_mem(fk.halfreal_tail_lds512_precrt, arg, g31.bufTwInv, "set half tail precrt tw31");
            set_karg_mem(fk.halfreal_tail_lds512_precrt, arg, g61.bufDigits, "set half tail precrt coeff_lo");
            set_karg_mem(fk.halfreal_tail_lds512_precrt, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set half tail precrt coeff_hi");
            set_karg(fk.halfreal_tail_lds512_precrt, arg, n, "set half tail precrt n");
            set_karg(fk.halfreal_tail_lds512_precrt, arg, p, "set half tail precrt p");
            set_karg(fk.halfreal_tail_lds512_precrt, arg, g61.lr2, "set half tail precrt lr2_61");
            set_karg(fk.halfreal_tail_lds512_precrt, arg, g31.lr2, "set half tail precrt lr2_31");
            set_karg(fk.halfreal_tail_lds512_precrt, arg, logm, "set half tail precrt logm");
            const size_t global = (size_t)(m / 512u) * local64;
            enqueue_kernel(g61, fk.halfreal_tail_lds512_precrt, global, &local64,
                           "enqueue half tail LDS precrt", "crt_halfreal_tail_lds512_precrt");
            g61.crtCoeffPending = true;
        };
        auto fwd61_m = [&](const StageInfo& st, uint32_t radix) {
            cl_kernel k = (radix == 8u) ? fk.fwd_r8_61 : (radix == 4u ? fk.fwd_r4_61 : fk.fwd_r2_61);
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set half fwd61 a");
            set_karg_mem(k, arg, g61.bufTwFwd, "set half fwd61 tw");
            set_karg(k, arg, m, "set half fwd61 n");
            set_karg(k, arg, st.len, "set half fwd61 len");
            enqueue_kernel(g61, k, round_up_size(std::max<size_t>(1, m / radix), local64), &local64, "enqueue half fwd61", radix == 8u ? "crt_halfreal_fwd_radix8_61" : (radix == 4u ? "crt_halfreal_fwd_radix4_61" : "crt_halfreal_fwd_radix2_61"));
        };
        auto fwd31_m = [&](const StageInfo& st, uint32_t radix) {
            cl_kernel k = (radix == 8u) ? fk.fwd_r8_31 : (radix == 4u ? fk.fwd_r4_31 : fk.fwd_r2_31);
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set half fwd31 a");
            set_karg_mem(k, arg, g31.bufTwFwd, "set half fwd31 tw");
            set_karg(k, arg, m, "set half fwd31 n");
            set_karg(k, arg, st.len, "set half fwd31 len");
            enqueue_kernel(g31, k, round_up_size(std::max<size_t>(1, m / radix), local64), &local64, "enqueue half fwd31", radix == 8u ? "crt_halfreal_fwd_radix8_31" : (radix == 4u ? "crt_halfreal_fwd_radix4_31" : "crt_halfreal_fwd_radix2_31"));
        };
        auto inv61_m = [&](const StageInfo& st, uint32_t radix) {
            cl_kernel k = (radix == 8u) ? fk.inv_r8_61 : (radix == 4u ? fk.inv_r4_61 : fk.inv_r2_61);
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set half inv61 a");
            set_karg_mem(k, arg, g61.bufTwInv, "set half inv61 tw");
            set_karg(k, arg, m, "set half inv61 n");
            set_karg(k, arg, st.len, "set half inv61 len");
            enqueue_kernel(g61, k, round_up_size(std::max<size_t>(1, m / radix), local64), &local64, "enqueue half inv61", radix == 8u ? "crt_halfreal_inv_radix8_61" : (radix == 4u ? "crt_halfreal_inv_radix4_61" : "crt_halfreal_inv_radix2_61"));
        };
        auto inv31_m = [&](const StageInfo& st, uint32_t radix) {
            cl_kernel k = (radix == 8u) ? fk.inv_r8_31 : (radix == 4u ? fk.inv_r4_31 : fk.inv_r2_31);
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set half inv31 a");
            set_karg_mem(k, arg, g31.bufTwInv, "set half inv31 tw");
            set_karg(k, arg, m, "set half inv31 n");
            set_karg(k, arg, st.len, "set half inv31 len");
            enqueue_kernel(g31, k, round_up_size(std::max<size_t>(1, m / radix), local64), &local64, "enqueue half inv31", radix == 8u ? "crt_halfreal_inv_radix8_31" : (radix == 4u ? "crt_halfreal_inv_radix4_31" : "crt_halfreal_inv_radix2_31"));
        };
        
        
        auto enqueue_half_lds_forward61 = [&]() {
            cl_kernel k = half_use_mid_lds512_opt ? fk.fwd_lds_stage_61_512opt : fk.fwd_lds_stage_61;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set half LDS fwd61 a");
            set_karg_mem(k, arg, g61.bufTwFwd, "set half LDS fwd61 tw");
            set_karg(k, arg, m, "set half LDS fwd61 n");
            set_karg(k, arg, half_global_stop_chunk, "set half LDS fwd61 len");
            set_karg(k, arg, half_lds_radix, "set half LDS fwd61 radix");
            const size_t global = (size_t)(m / half_global_stop_chunk) * (size_t)half_center_chunk * local64;
            enqueue_kernel(g61, k, global, &local64, "enqueue half LDS fwd61",
                           half_use_mid_lds512_opt ? "crt_half_lds_stage512_forward_opt_61" : "crt_half_lds_stage512_forward_generic_61");
        };
        auto enqueue_half_lds_forward31 = [&]() {
            cl_kernel k = half_use_mid_lds512_opt ? fk.fwd_lds_stage_31_512opt : fk.fwd_lds_stage_31;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set half LDS fwd31 a");
            set_karg_mem(k, arg, g31.bufTwFwd, "set half LDS fwd31 tw");
            set_karg(k, arg, m, "set half LDS fwd31 n");
            set_karg(k, arg, half_global_stop_chunk, "set half LDS fwd31 len");
            set_karg(k, arg, half_lds_radix, "set half LDS fwd31 radix");
            const size_t global = (size_t)(m / half_global_stop_chunk) * (size_t)half_center_chunk * local64;
            enqueue_kernel(g31, k, global, &local64, "enqueue half LDS fwd31",
                           half_use_mid_lds512_opt ? "crt_half_lds_stage512_forward_opt_31" : "crt_half_lds_stage512_forward_generic_31");
        };
        auto enqueue_half_lds_inverse61 = [&]() {
            cl_kernel k = half_use_mid_lds512_opt ? fk.inv_lds_stage_61_512opt : fk.inv_lds_stage_61;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g61.bufField, "set half LDS inv61 a");
            set_karg_mem(k, arg, g61.bufTwInv, "set half LDS inv61 tw");
            set_karg(k, arg, m, "set half LDS inv61 n");
            set_karg(k, arg, half_center_chunk, "set half LDS inv61 base_len");
            set_karg(k, arg, half_lds_radix, "set half LDS inv61 radix");
            const size_t global = (size_t)(m / half_global_stop_chunk) * (size_t)half_center_chunk * local64;
            enqueue_kernel(g61, k, global, &local64, "enqueue half LDS inv61",
                           half_use_mid_lds512_opt ? "crt_half_lds_stage512_inverse_opt_61" : "crt_half_lds_stage512_inverse_generic_61");
        };
        auto enqueue_half_lds_inverse31 = [&]() {
            cl_kernel k = half_use_mid_lds512_opt ? fk.inv_lds_stage_31_512opt : fk.inv_lds_stage_31;
            cl_uint arg = 0;
            set_karg_mem(k, arg, g31.bufField, "set half LDS inv31 a");
            set_karg_mem(k, arg, g31.bufTwInv, "set half LDS inv31 tw");
            set_karg(k, arg, m, "set half LDS inv31 n");
            set_karg(k, arg, half_center_chunk, "set half LDS inv31 base_len");
            set_karg(k, arg, half_lds_radix, "set half LDS inv31 radix");
            const size_t global = (size_t)(m / half_global_stop_chunk) * (size_t)half_center_chunk * local64;
            enqueue_kernel(g31, k, global, &local64, "enqueue half LDS inv31",
                           half_use_mid_lds512_opt ? "crt_half_lds_stage512_inverse_opt_31" : "crt_half_lds_stage512_inverse_generic_31");
        };

        auto enqueue_half_lds_pair61 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_lds512_pair61, arg, g61.bufField, "set half LDS pair61 a");
            set_karg_mem(fk.halfreal_lds512_pair61, arg, g61.bufTwFwd, "set half LDS pair61 twf");
            set_karg_mem(fk.halfreal_lds512_pair61, arg, g61.bufTwInv, "set half LDS pair61 twi");
            set_karg(fk.halfreal_lds512_pair61, arg, n, "set half LDS pair61 n");
            set_karg(fk.halfreal_lds512_pair61, arg, half_flags61, "set half LDS pair61 flags");
            const uint32_t pair_blocks = (m >> 10) + 1u;
            const size_t global = (size_t)pair_blocks * local64;
            enqueue_kernel(g61, fk.halfreal_lds512_pair61, global, &local64, "enqueue half LDS pair61", "crt_halfreal_lds512_pair_61");
        };
        auto enqueue_half_lds_pair31 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_lds512_pair31, arg, g31.bufField, "set half LDS pair31 a");
            set_karg_mem(fk.halfreal_lds512_pair31, arg, g31.bufTwFwd, "set half LDS pair31 twf");
            set_karg_mem(fk.halfreal_lds512_pair31, arg, g31.bufTwInv, "set half LDS pair31 twi");
            set_karg(fk.halfreal_lds512_pair31, arg, n, "set half LDS pair31 n");
            set_karg(fk.halfreal_lds512_pair31, arg, half_flags31, "set half LDS pair31 flags");
            const uint32_t pair_blocks = (m >> 10) + 1u;
            const size_t global = (size_t)pair_blocks * local64;
            enqueue_kernel(g31, fk.halfreal_lds512_pair31, global, &local64, "enqueue half LDS pair31", "crt_halfreal_lds512_pair_31");
        };

        auto enqueue_half_center512_pair61 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_center512_pair61, arg, g61.bufField, "set half center512 pair61 a");
            set_karg_mem(fk.halfreal_center512_pair61, arg, g61.bufTwFwd, "set half center512 pair61 twf");
            set_karg_mem(fk.halfreal_center512_pair61, arg, g61.bufTwInv, "set half center512 pair61 twi");
            set_karg(fk.halfreal_center512_pair61, arg, n, "set half center512 pair61 n");
            set_karg(fk.halfreal_center512_pair61, arg, half_flags61, "set half center512 pair61 flags");
            const size_t global = (size_t)(m / 512u) * local64;
            enqueue_kernel(g61, fk.halfreal_center512_pair61, global, &local64, "enqueue half center512 pair61", "crt_halfreal_center512_pair_61");
        };
        auto enqueue_half_center512_pair31 = [&]() {
            cl_uint arg = 0;
            set_karg_mem(fk.halfreal_center512_pair31, arg, g31.bufField, "set half center512 pair31 a");
            set_karg_mem(fk.halfreal_center512_pair31, arg, g31.bufTwFwd, "set half center512 pair31 twf");
            set_karg_mem(fk.halfreal_center512_pair31, arg, g31.bufTwInv, "set half center512 pair31 twi");
            set_karg(fk.halfreal_center512_pair31, arg, n, "set half center512 pair31 n");
            set_karg(fk.halfreal_center512_pair31, arg, half_flags31, "set half center512 pair31 flags");
            const size_t global = (size_t)(m / 512u) * local64;
            enqueue_kernel(g31, fk.halfreal_center512_pair31, global, &local64, "enqueue half center512 pair31", "crt_halfreal_center512_pair_31");
        };

        auto run_forward_m = [&](auto&& f61, auto&& f31) {
            int i = (int)g61.stages.size() - 1;
            const uint32_t fwd_start_len = half_use_head_lds ? half_head_base_len : m;
            while (i >= 0 && g61.stages[(size_t)i].len > fwd_start_len) --i;
            while (i >= 0) {
                const uint32_t len = g61.stages[(size_t)i].len;
                if (half_use_mid_lds512 && len <= half_global_stop_chunk) break;
                if (len >= 8u && i >= 2) { f61(g61.stages[(size_t)i], 8u); f31(g61.stages[(size_t)i], 8u); i -= 3; }
                else if (len >= 4u && i >= 1) { f61(g61.stages[(size_t)i], 4u); f31(g61.stages[(size_t)i], 4u); i -= 2; }
                else { f61(g61.stages[(size_t)i], 2u); f31(g61.stages[(size_t)i], 2u); --i; }
            }
            if (half_use_mid_lds512 && !half_use_lds512_pair) { enqueue_half_lds_forward61(); enqueue_half_lds_forward31(); }
        };
        auto run_inverse_m = [&](auto&& f61, auto&& f31) {
            if (half_use_mid_lds512 && !half_use_lds512_pair) { enqueue_half_lds_inverse61(); enqueue_half_lds_inverse31(); }
            const uint32_t inv_limit_len = half_use_head_lds ? half_head_base_len : m;
            size_t i = 0;
            while (i < g61.stages.size() && g61.stages[i].len <= m) {
                const uint32_t len = g61.stages[i].len;
                if (half_use_mid_lds512 && len <= half_global_stop_chunk) { ++i; continue; }
                if (len > inv_limit_len) break;
                if (len * 4u <= inv_limit_len && i + 2 < g61.stages.size() && g61.stages[i + 2].len <= inv_limit_len) { f61(g61.stages[i], 8u); f31(g61.stages[i], 8u); i += 3; }
                else if (len * 2u <= inv_limit_len && i + 1 < g61.stages.size() && g61.stages[i + 1].len <= inv_limit_len) { f61(g61.stages[i], 4u); f31(g61.stages[i], 4u); i += 2; }
                else { f61(g61.stages[i], 2u); f31(g61.stages[i], 2u); ++i; }
            }
        };

        if (half_use_head_lds) {
            if (half_head_precrt) {
                head_lds512_pack_fwd_precrt();
            } else {
                head_lds512_pack_fwd61();
                head_lds512_pack_fwd31();
            }
        } else {
            pack61(); pack31();
        }
        run_forward_m(fwd61_m, fwd31_m);
        if (half_use_lds512_pair) {
            
            
            enqueue_half_lds_pair61();
            enqueue_half_lds_pair31();
        } else if (half_use_lds512_split) {
            
            
            enqueue_half_center512_pair61();
            enqueue_half_center512_pair31();
        } else {
            
            
            bitrev_swap61("crt_halfreal_bitrev_to_linear_61");
            bitrev_swap31("crt_halfreal_bitrev_to_linear_31");
            center61(); center31();
            bitrev_swap61("crt_halfreal_bitrev_from_linear_61");
            bitrev_swap31("crt_halfreal_bitrev_from_linear_31");
        }
        run_inverse_m(inv61_m, inv31_m);
        if (half_use_head_lds) {
            if (half_tail_precrt) {
                cl_event gf31_tail_ready = enqueue_queue_marker(g31, "halfreal gf31 ready for tail precrt");
                set_pending_wait_event(g61, gf31_tail_ready);
                tail_lds512_inv_unpack_precrt();
            } else {
                tail_lds512_inv_unpack61();
                tail_lds512_inv_unpack31();
            }
        } else {
            unpack61(); unpack31();
        }
        
        
        if (parse_bool_env("PRMERS_CRT_HALFREAL_FINISH", false)) {
            cl_int ferr = clFlush(g61.queue);
            if (ferr != CL_SUCCESS) {
                std::cerr << "Error: halfreal clFlush GF61 failed: " << ferr << "\n";
                return false;
            }
            ferr = clFlush(g31.queue);
            if (ferr != CL_SUCCESS) {
                std::cerr << "Error: halfreal clFlush GF31 failed: " << ferr << "\n";
                return false;
            }
            ferr = clFinish(g61.queue);
            if (ferr != CL_SUCCESS) {
                std::cerr << "Error: halfreal clFinish GF61 failed: " << ferr << "\n";
                return false;
            }
            ferr = clFinish(g31.queue);
            if (ferr != CL_SUCCESS) {
                std::cerr << "Error: halfreal clFinish GF31 failed: " << ferr << "\n";
                return false;
            }
        } else {
            if (!half_tail_precrt) {
                cl_event gf31_half_done = enqueue_queue_marker(g31, "halfreal gf31 done before Garner");
                set_pending_wait_event(g61, gf31_half_done);
            }
            flush_async_queues();
        }
        return true;
    }


    const int schedule = g_crt_defused_schedule;

    
    if (schedule == 6) {
        
        
        enqueue_weight61();
        enqueue_weight31();
        cl_event gf31_weight_done = enqueue_queue_marker(g31, "defused gf31 weight done");
        set_pending_wait_event(g61, gf31_weight_done);
        flush_async_queues();

        const bool did_top61 = enqueue_forward_top_radix8_61();
        cl_event gf61_top_fwd_done = enqueue_queue_marker(g61, "defused gf61 top fwd8 done");

        set_pending_wait_event(g31, gf61_top_fwd_done);
        enqueue_forward_all31();
        if (use_defused_lds_stage) enqueue_defused_lds_forward31();
        enqueue_center31();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse31();
        enqueue_inverse_all31();
        enqueue_last31();

        if (did_top61) enqueue_forward_after_top61();
        else enqueue_forward_all61();
        if (use_defused_lds_stage) enqueue_defused_lds_forward61();
        enqueue_center61();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse61();
        enqueue_inverse_all61();
        enqueue_last61();

        cl_event gf31_done = enqueue_queue_marker(g31, "defused gf31 done");
        set_pending_wait_event(g61, gf31_done);
        flush_async_queues();
        return true;
    }

    if (schedule == 3) {
        
        
        enqueue_weight61();
        enqueue_weight31();
        cl_event gf31_weight_done = enqueue_queue_marker(g31, "defused gf31 weight done");
        flush_async_queues();

        enqueue_forward_all61();
        if (use_defused_lds_stage) enqueue_defused_lds_forward61();
        enqueue_center61();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse61();
        enqueue_inverse_all61();
        set_pending_wait_event(g61, gf31_weight_done);
        enqueue_last61();
        cl_event gf61_done = enqueue_queue_marker(g61, "defused gf61 done before gf31 body");
        set_pending_wait_event(g31, gf61_done);

        enqueue_forward_all31();
        if (use_defused_lds_stage) enqueue_defused_lds_forward31();
        enqueue_center31();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse31();
        enqueue_inverse_all31();
        enqueue_last31();

        cl_event gf31_done = enqueue_queue_marker(g31, "defused gf31 done");
        set_pending_wait_event(g61, gf31_done);
        flush_async_queues();
        return true;
    }

    if (schedule == 5) {
        
        
        enqueue_weight61();
        enqueue_weight31();
        cl_event gf31_weight_done = enqueue_queue_marker(g31, "defused gf31 weight done");
        flush_async_queues();

        enqueue_forward_all61();
        cl_event gf61_forward_done = enqueue_queue_marker(g61, "defused gf61 forward done");
        set_pending_wait_event(g31, gf61_forward_done);

        if (use_defused_lds_stage) enqueue_defused_lds_forward61();
        enqueue_center61();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse61();
        enqueue_inverse_all61();

        enqueue_forward_all31();
        if (use_defused_lds_stage) enqueue_defused_lds_forward31();
        enqueue_center31();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse31();
        enqueue_inverse_all31();
        enqueue_last31();

        set_pending_wait_event(g61, gf31_weight_done);
        enqueue_last61();

        cl_event gf31_done = enqueue_queue_marker(g31, "defused gf31 done");
        set_pending_wait_event(g61, gf31_done);
        flush_async_queues();
        return true;
    }

    if (schedule == 4) {
        
        
        enqueue_weight31();
        enqueue_forward_all31();
        if (use_defused_lds_stage) enqueue_defused_lds_forward31();
        enqueue_center31();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse31();
        enqueue_inverse_all31();
        enqueue_last31();
        cl_event gf31_body_done = enqueue_queue_marker(g31, "defused gf31 serial body done");
        set_pending_wait_event(g61, gf31_body_done);

        enqueue_weight61();
        enqueue_forward_all61();
        if (use_defused_lds_stage) enqueue_defused_lds_forward61();
        enqueue_center61();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse61();
        enqueue_inverse_all61();
        enqueue_last61();

        
        cl_event gf31_done = enqueue_queue_marker(g31, "defused gf31 done");
        set_pending_wait_event(g61, gf31_done);
        flush_async_queues();
        return true;
    }

    enqueue_weight61();
    enqueue_weight31();
    
    cl_event gf31_weight_done = enqueue_queue_marker(g31, "defused gf31 weight done");
    flush_async_queues();

    if (schedule == 1) {
        
        
        enqueue_forward_interleaved();
        if (use_defused_lds_stage) {
            enqueue_defused_lds_forward61();
            enqueue_defused_lds_forward31();
        }
        enqueue_center61();
        enqueue_center31();
        if (use_defused_lds_stage) {
            enqueue_defused_lds_inverse61();
            enqueue_defused_lds_inverse31();
        }
        enqueue_inverse_interleaved();
        enqueue_last31();
        set_pending_wait_event(g61, gf31_weight_done);
        enqueue_last61();
    } else if (schedule == 2) {
        
        
        enqueue_forward_all31();
        if (use_defused_lds_stage) enqueue_defused_lds_forward31();
        enqueue_center31();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse31();
        enqueue_inverse_all31();
        enqueue_last31();

        enqueue_forward_all61();
        if (use_defused_lds_stage) enqueue_defused_lds_forward61();
        enqueue_center61();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse61();
        enqueue_inverse_all61();
        set_pending_wait_event(g61, gf31_weight_done);
        enqueue_last61();
    } else {
        
        
        enqueue_forward_all61();
        if (use_defused_lds_stage) enqueue_defused_lds_forward61();
        enqueue_center61();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse61();
        enqueue_inverse_all61();

        enqueue_forward_all31();
        if (use_defused_lds_stage) enqueue_defused_lds_forward31();
        enqueue_center31();
        if (use_defused_lds_stage) enqueue_defused_lds_inverse31();
        enqueue_inverse_all31();
        if (use_precrt_block16) {
            cl_event gf31_body_done = enqueue_queue_marker(g31, "defused gf31 done before precrt block16 tail");
            if (gf31_weight_done) clReleaseEvent(gf31_weight_done);
            set_pending_wait_event(g61, gf31_body_done);
            enqueue_last_crtcoeff_block16();
            flush_async_queues();
            return true;
        }
        enqueue_last31();

        set_pending_wait_event(g61, gf31_weight_done);
        enqueue_last61();
    }

    cl_event gf31_done = enqueue_queue_marker(g31, "defused gf31 done");
    set_pending_wait_event(g61, gf31_done);
    flush_async_queues();
    return true;
}

static bool enqueue_square_mod_crt_fused_gpuowl_like(GpuPrp& g61, GpuPrp& g31, CrtFusedKernels& fk, bool split_center, bool fused_center_lockstep) {
    if (!fk.available()) return false;
    if (g61.n != g31.n) return false;
    if (g61.n < 512u || (g61.n & 511u) != 0u) return false;
    if (g_force_strict_reference) return false;

    const cl_uint n = g61.n;
    const cl_uint p = g61.exponent_p;
    const cl_uint log_n = ([](cl_uint v){ cl_uint r=0; while (v > 1u) { v >>= 1; ++r; } return r; }(n));
    const size_t local64 = 64u;
    const size_t local128 = 128u;
    
    
    const size_t local_bridge512 = 128u;
    const size_t local_bridge1024 = 256u;
    const bool use_boundary1024 = fk.boundary1024_available() && (std::getenv("PRMERS_CRT_BOUNDARY512") == nullptr) && ((log_n & 1u) == 0u) && n >= 16384u;
    const bool use_last16_unweight = use_boundary1024 && fk.last16_available() && (std::getenv("PRMERS_CRT_LAST16_UNWEIGHT") != nullptr);

    
    {
        const StageInfo& first = g61.stages.back();
        cl_uint arg = 0;
        set_karg_mem(fk.weight_first, arg, g61.bufDigits, "set crt weight digits");
        set_karg_mem(fk.weight_first, arg, g61.bufField, "set crt weight field61");
        set_karg_mem(fk.weight_first, arg, g61.bufTwFwd, "set crt weight tw61");
        set_karg_mem(fk.weight_first, arg, g31.bufField, "set crt weight field31");
        set_karg_mem(fk.weight_first, arg, g31.bufTwFwd, "set crt weight tw31");
        set_karg(fk.weight_first, arg, p, "set crt weight p");
        set_karg(fk.weight_first, arg, g61.lr2, "set crt weight lr2 61");
        set_karg(fk.weight_first, arg, g31.lr2, "set crt weight lr2 31");
        set_karg(fk.weight_first, arg, n, "set crt weight n");
        set_karg(fk.weight_first, arg, first.len, "set crt weight len");
        size_t global = n / 4u;
        enqueue_kernel(g61, fk.weight_first, global, &local64, "enqueue crt fused weight_first", "crt_fused_weight_first");
    }

    const auto& stages = g61.stages;
    const bool use_global_radix8 = g_crt_radix8_global && fk.fwd_r8 && fk.inv_r8 && fk.fwd_r4 && fk.inv_r4 && fk.fwd_r2 && fk.inv_r2 && n >= 2048u;
    const cl_uint center_chunk = use_global_radix8 ? std::min<cl_uint>(std::max<cl_uint>(g_crt_center_chunk, 8u), 1024u) : 256u;
    const cl_uint local_stage_max = std::min<cl_uint>(std::max<cl_uint>(g_crt_lds_stage, 0u), 512u);
    
    
    const bool allow_strided_lds_stage = parse_bool_env("PRMERS_CRT_ALLOW_STRIDED_LDS_STAGE", false);
    const bool use_multi_lds_stage = allow_strided_lds_stage && use_global_radix8 && fk.fwd_lds_stage && fk.inv_lds_stage && local_stage_max >= 16u && n >= 512u;
    const bool use_lds512_stage = !use_multi_lds_stage && use_global_radix8 && fk.fwd_lds512 && fk.inv_lds512 && g_crt_lds_stage >= 512u && center_chunk < 512u && n >= 512u;
    const cl_uint global_stop_chunk = use_multi_lds_stage ? (n / 4u) : (use_lds512_stage ? 512u : center_chunk);
    bool need_forward_bridge = !use_global_radix8;
    std::vector<CrtLdsPlanStep> crt_lds_plan;
    bool crt_head_r8_used = false;

    if (use_global_radix8) {
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len == n || st.len == n / 2u) continue;
            if (st.len <= global_stop_chunk) break;

            const bool can_r8 = (st.len > 4u * global_stop_chunk) && idx >= 2;
            if (can_r8) {
                cl_uint arg = 0;
                set_karg_mem(fk.fwd_r8, arg, g61.bufField, "set crt fwd8 a61");
                set_karg_mem(fk.fwd_r8, arg, g61.bufTwFwd, "set crt fwd8 tw61");
                set_karg_mem(fk.fwd_r8, arg, g31.bufField, "set crt fwd8 a31");
                set_karg_mem(fk.fwd_r8, arg, g31.bufTwFwd, "set crt fwd8 tw31");
                set_karg(fk.fwd_r8, arg, n, "set crt fwd8 n");
                set_karg(fk.fwd_r8, arg, st.len, "set crt fwd8 len");
                size_t global = n / 8u;
                enqueue_kernel(g61, fk.fwd_r8, global, &local64, "enqueue crt fused fwd radix8", "crt_fused_fwd_radix8");
                idx -= 2u;
                continue;
            }

            const bool can_r4 = (st.len > 2u * global_stop_chunk) && idx >= 1;
            if (can_r4) {
                cl_uint arg = 0;
                set_karg_mem(fk.fwd_r4, arg, g61.bufField, "set crt fwd4 a61");
                set_karg_mem(fk.fwd_r4, arg, g61.bufTwFwd, "set crt fwd4 tw61");
                set_karg_mem(fk.fwd_r4, arg, g31.bufField, "set crt fwd4 a31");
                set_karg_mem(fk.fwd_r4, arg, g31.bufTwFwd, "set crt fwd4 tw31");
                set_karg(fk.fwd_r4, arg, n, "set crt fwd4 n");
                set_karg(fk.fwd_r4, arg, st.len, "set crt fwd4 len");
                size_t global = n / 4u;
                enqueue_kernel(g61, fk.fwd_r4, global, &local64, "enqueue crt fused fwd radix4", "crt_fused_fwd_radix4");
                --idx;
                continue;
            }

            cl_uint arg = 0;
            set_karg_mem(fk.fwd_r2, arg, g61.bufField, "set crt fwd2 a61");
            set_karg_mem(fk.fwd_r2, arg, g61.bufTwFwd, "set crt fwd2 tw61");
            set_karg_mem(fk.fwd_r2, arg, g31.bufField, "set crt fwd2 a31");
            set_karg_mem(fk.fwd_r2, arg, g31.bufTwFwd, "set crt fwd2 tw31");
            set_karg(fk.fwd_r2, arg, n, "set crt fwd2 n");
            set_karg(fk.fwd_r2, arg, st.len, "set crt fwd2 len");
            size_t global = n / 2u;
            enqueue_kernel(g61, fk.fwd_r2, global, &local64, "enqueue crt fused fwd radix2", "crt_fused_fwd_radix2");
        }
    } else {
        
        
        for (std::size_t idx = stages.size(); idx-- > 0;) {
            const StageInfo& st = stages[idx];
            if (st.len == n || st.len == n / 2u) continue;
            if (st.len <= (use_boundary1024 ? 1024u : 512u)) break;
            if (st.len < 4u) continue;
            cl_uint arg = 0;
            set_karg_mem(fk.fwd_r4, arg, g61.bufField, "set crt fwd a61");
            set_karg_mem(fk.fwd_r4, arg, g61.bufTwFwd, "set crt fwd tw61");
            set_karg_mem(fk.fwd_r4, arg, g31.bufField, "set crt fwd a31");
            set_karg_mem(fk.fwd_r4, arg, g31.bufTwFwd, "set crt fwd tw31");
            set_karg(fk.fwd_r4, arg, n, "set crt fwd n");
            set_karg(fk.fwd_r4, arg, st.len, "set crt fwd len");
            size_t global = n / 4u;
            enqueue_kernel(g61, fk.fwd_r4, global, &local64, "enqueue crt fused fwd radix4", "crt_fused_fwd_radix4");
            if (idx > 0) --idx;
        }
    }

    if (use_multi_lds_stage) {
        cl_uint cur_len = n / 4u;

        if (g_crt_head_radix8 != 0u && cur_len >= center_chunk * 8u) {
            cl_uint arg = 0;
            set_karg_mem(fk.fwd_r8, arg, g61.bufField, "set crt head fwd8 a61");
            set_karg_mem(fk.fwd_r8, arg, g61.bufTwFwd, "set crt head fwd8 tw61");
            set_karg_mem(fk.fwd_r8, arg, g31.bufField, "set crt head fwd8 a31");
            set_karg_mem(fk.fwd_r8, arg, g31.bufTwFwd, "set crt head fwd8 tw31");
            set_karg(fk.fwd_r8, arg, n, "set crt head fwd8 n");
            set_karg(fk.fwd_r8, arg, cur_len, "set crt head fwd8 len");
            size_t global = n / 8u;
            enqueue_kernel(g61, fk.fwd_r8, global, &local64, "enqueue crt head radix8", "crt_head_fwd_radix8");
            cur_len >>= 3;
            crt_head_r8_used = true;
        }

        while (cur_len > center_chunk) {
            cl_uint max_radix = cur_len / center_chunk;
            if (max_radix > local_stage_max) max_radix = local_stage_max;
            cl_uint radix = floor_pow2_leq(max_radix);
            if (radix > 512u) radix = 512u;
            if (radix < 2u) return false;

            if (radix < 16u) {
                if (radix >= 8u) {
                    cl_uint arg = 0;
                    set_karg_mem(fk.fwd_r8, arg, g61.bufField, "set crt tail fwd8 a61");
                    set_karg_mem(fk.fwd_r8, arg, g61.bufTwFwd, "set crt tail fwd8 tw61");
                    set_karg_mem(fk.fwd_r8, arg, g31.bufField, "set crt tail fwd8 a31");
                    set_karg_mem(fk.fwd_r8, arg, g31.bufTwFwd, "set crt tail fwd8 tw31");
                    set_karg(fk.fwd_r8, arg, n, "set crt tail fwd8 n");
                    set_karg(fk.fwd_r8, arg, cur_len, "set crt tail fwd8 len");
                    size_t global = n / 8u;
                    enqueue_kernel(g61, fk.fwd_r8, global, &local64, "enqueue crt tail radix8", "crt_tail_fwd_radix8");
                } else if (radix >= 4u) {
                    cl_uint arg = 0;
                    set_karg_mem(fk.fwd_r4, arg, g61.bufField, "set crt tail fwd4 a61");
                    set_karg_mem(fk.fwd_r4, arg, g61.bufTwFwd, "set crt tail fwd4 tw61");
                    set_karg_mem(fk.fwd_r4, arg, g31.bufField, "set crt tail fwd4 a31");
                    set_karg_mem(fk.fwd_r4, arg, g31.bufTwFwd, "set crt tail fwd4 tw31");
                    set_karg(fk.fwd_r4, arg, n, "set crt tail fwd4 n");
                    set_karg(fk.fwd_r4, arg, cur_len, "set crt tail fwd4 len");
                    size_t global = n / 4u;
                    enqueue_kernel(g61, fk.fwd_r4, global, &local64, "enqueue crt tail radix4", "crt_tail_fwd_radix4");
                } else {
                    cl_uint arg = 0;
                    set_karg_mem(fk.fwd_r2, arg, g61.bufField, "set crt tail fwd2 a61");
                    set_karg_mem(fk.fwd_r2, arg, g61.bufTwFwd, "set crt tail fwd2 tw61");
                    set_karg_mem(fk.fwd_r2, arg, g31.bufField, "set crt tail fwd2 a31");
                    set_karg_mem(fk.fwd_r2, arg, g31.bufTwFwd, "set crt tail fwd2 tw31");
                    set_karg(fk.fwd_r2, arg, n, "set crt tail fwd2 n");
                    set_karg(fk.fwd_r2, arg, cur_len, "set crt tail fwd2 len");
                    size_t global = n / 2u;
                    enqueue_kernel(g61, fk.fwd_r2, global, &local64, "enqueue crt tail radix2", "crt_tail_fwd_radix2");
                }
                crt_lds_plan.push_back({cur_len, radix, true});
                cur_len /= radix;
                continue;
            }

            const bool use_tile2 = (g_crt_lds_tile >= 2u) && fk.fwd_lds_stage_tile2 && radix >= 16u && (cur_len / radix) >= 2u;
            cl_kernel k_lds = use_tile2 ? fk.fwd_lds_stage_tile2 : fk.fwd_lds_stage;
            cl_uint arg = 0;
            set_karg_mem(k_lds, arg, g61.bufField, "set crt lds stage fwd a61");
            set_karg_mem(k_lds, arg, g61.bufTwFwd, "set crt lds stage fwd tw61");
            set_karg_mem(k_lds, arg, g31.bufField, "set crt lds stage fwd a31");
            set_karg_mem(k_lds, arg, g31.bufTwFwd, "set crt lds stage fwd tw31");
            set_karg(k_lds, arg, n, "set crt lds stage fwd n");
            set_karg(k_lds, arg, cur_len, "set crt lds stage fwd len");
            set_karg(k_lds, arg, radix, "set crt lds stage fwd radix");
            const size_t local_lds = 128u;
            const size_t groups = use_tile2 ? ((static_cast<size_t>(n) / cur_len) * (((cur_len / radix) + 1u) >> 1u)) : (static_cast<size_t>(n) / radix);
            const size_t global = groups * local_lds;
            enqueue_kernel(g61, k_lds, global, &local_lds, "enqueue crt lds stage forward", use_tile2 ? crt_lds_stage_fwd_tile2_label(radix) : crt_lds_stage_fwd_label(radix));
            crt_lds_plan.push_back({cur_len, radix, false});
            cur_len /= radix;
        }
        if (cur_len != center_chunk) return false;
    }

    if (need_forward_bridge) {
        cl_uint arg = 0;
        if (use_boundary1024) {
            set_karg_mem(fk.fwd_bridge1024, arg, g61.bufField, "set crt fwd bridge1024 a61");
            set_karg_mem(fk.fwd_bridge1024, arg, g31.bufField, "set crt fwd bridge1024 a31");
            set_karg_mem(fk.fwd_bridge1024, arg, g61.bufTwFwd, "set crt fwd bridge1024 tw61");
            set_karg_mem(fk.fwd_bridge1024, arg, g31.bufTwFwd, "set crt fwd bridge1024 tw31");
            set_karg(fk.fwd_bridge1024, arg, n, "set crt fwd bridge1024 n");
            const size_t global = (n / 1024u) * local_bridge1024;
            enqueue_kernel(g61, fk.fwd_bridge1024, global, &local_bridge1024, "enqueue crt fused fwd bridge1024", "crt_fused_fwd_bridge1024");
        } else {
            set_karg_mem(fk.fwd_bridge512, arg, g61.bufField, "set crt fwd bridge a61");
            set_karg_mem(fk.fwd_bridge512, arg, g61.bufTwFwd, "set crt fwd bridge tw61");
            set_karg_mem(fk.fwd_bridge512, arg, g31.bufField, "set crt fwd bridge a31");
            set_karg_mem(fk.fwd_bridge512, arg, g31.bufTwFwd, "set crt fwd bridge tw31");
            set_karg(fk.fwd_bridge512, arg, n, "set crt fwd bridge n");
            const size_t global = (n / 512u) * local_bridge512;
            enqueue_kernel(g61, fk.fwd_bridge512, global, &local_bridge512, "enqueue crt fused fwd bridge512", "crt_fused_fwd_bridge512");
        }
    }

    if (use_lds512_stage) {
        cl_uint arg = 0;
        set_karg_mem(fk.fwd_lds512, arg, g61.bufField, "set crt lds512 fwd a61");
        set_karg_mem(fk.fwd_lds512, arg, g61.bufTwFwd, "set crt lds512 fwd tw61");
        set_karg_mem(fk.fwd_lds512, arg, g31.bufField, "set crt lds512 fwd a31");
        set_karg_mem(fk.fwd_lds512, arg, g31.bufTwFwd, "set crt lds512 fwd tw31");
        set_karg(fk.fwd_lds512, arg, n, "set crt lds512 fwd n");
        set_karg(fk.fwd_lds512, arg, center_chunk, "set crt lds512 fwd target");
        const size_t local_lds = 128u;
        const size_t global = (n / 512u) * local_lds;
        enqueue_kernel(g61, fk.fwd_lds512, global, &local_lds, "enqueue crt lds512 forward", "crt_lds512_forward");
    }

    {
        if (!split_center && (fk.center256 || fk.center256_dualwave)) {
            cl_kernel k_center = nullptr;
            const char* label = nullptr;
            size_t local_center = 128u;
            if (!fused_center_lockstep) {
                if (center_chunk == 1024u) {
                    k_center = fk.center1024_dualwave;
                    label = "crt_lds_square1024";
                } else if (center_chunk == 512u) {
                    
                    
                    const bool use_reglds_center512 = parse_bool_env("PRMERS_CRT_USE_REGLDS_CENTER512", false);
                    if (use_reglds_center512 && fk.center512_reglds) {
                        k_center = fk.center512_reglds;
                        label = "crt_reglds_square512";
                    } else {
                        k_center = fk.center512_dualwave;
                        label = "crt_lds_square512";
                    }
                } else if (center_chunk == 256u) {
                    k_center = fk.center256_dualwave;
                    label = "crt_lds_square256";
                } else if (center_chunk == 128u) {
                    k_center = fk.center128_dualwave;
                    label = "crt_lds_square128";
                } else if (center_chunk == 64u) {
                    k_center = fk.center64_dualwave;
                    label = "crt_lds_square64";
                } else if (center_chunk == 32u) {
                    k_center = fk.center32_dualwave;
                    label = "crt_lds_square32";
                } else if (center_chunk == 16u) {
                    k_center = fk.center16_dualwave;
                    label = "crt_lds_square16";
                } else if (center_chunk == 8u) {
                    k_center = fk.center8_dualwave;
                    label = "crt_lds_square8";
                }
            } else if (center_chunk == 256u) {
                k_center = fk.center256;
                label = "crt_lds_square256_lockstep";
                local_center = 64u;
            }
            if (!k_center) return false;
            cl_uint arg = 0;
            set_karg_mem(k_center, arg, g61.bufField, "set crt center a61");
            set_karg_mem(k_center, arg, g61.bufTwFwd, "set crt center twf61");
            set_karg_mem(k_center, arg, g61.bufTwInv, "set crt center twi61");
            set_karg_mem(k_center, arg, g31.bufField, "set crt center a31");
            set_karg_mem(k_center, arg, g31.bufTwFwd, "set crt center twf31");
            set_karg_mem(k_center, arg, g31.bufTwInv, "set crt center twi31");
            set_karg(k_center, arg, n, "set crt center n");
            const size_t global = (n / center_chunk) * local_center;
            enqueue_kernel(g61, k_center, global, &local_center, "enqueue crt fused center", label);
        } else {
            CenterKernelConfig c61 = choose_center_kernel(g61, center_chunk);
            CenterKernelConfig c31 = choose_center_kernel(g31, center_chunk);
            if (!c61.enabled || !c31.enabled || c61.chunk != center_chunk || c31.chunk != center_chunk) return false;
            if (g31.queue != g61.queue) {
                cl_event bridge_done = enqueue_queue_marker(g61, "crt fused forward bridge marker");
                set_pending_wait_event(g31, bridge_done);
                if (parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false)) clFlush(g61.queue);
            }
            enqueue_center_fused(g61, c61, "crt_split_center61");
            enqueue_center_fused(g31, c31, "crt_split_center31");
            if (g31.queue != g61.queue) {
                cl_event center31_done = enqueue_queue_marker(g31, "crt fused gf31 center marker");
                set_pending_wait_event(g61, center31_done);
                if (parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false)) clFlush(g31.queue);
            }
        }
    }

    if (use_lds512_stage) {
        cl_uint arg = 0;
        set_karg_mem(fk.inv_lds512, arg, g61.bufField, "set crt lds512 inv a61");
        set_karg_mem(fk.inv_lds512, arg, g61.bufTwInv, "set crt lds512 inv tw61");
        set_karg_mem(fk.inv_lds512, arg, g31.bufField, "set crt lds512 inv a31");
        set_karg_mem(fk.inv_lds512, arg, g31.bufTwInv, "set crt lds512 inv tw31");
        set_karg(fk.inv_lds512, arg, n, "set crt lds512 inv n");
        set_karg(fk.inv_lds512, arg, center_chunk, "set crt lds512 inv target");
        const size_t local_lds = 128u;
        const size_t global = (n / 512u) * local_lds;
        enqueue_kernel(g61, fk.inv_lds512, global, &local_lds, "enqueue crt lds512 inverse", "crt_lds512_inverse");
    }

    if (!use_global_radix8) {
        cl_uint arg = 0;
        if (use_boundary1024) {
            set_karg_mem(fk.inv_bridge1024, arg, g61.bufField, "set crt inv bridge1024 a61");
            set_karg_mem(fk.inv_bridge1024, arg, g31.bufField, "set crt inv bridge1024 a31");
            set_karg_mem(fk.inv_bridge1024, arg, g61.bufTwInv, "set crt inv bridge1024 tw61");
            set_karg_mem(fk.inv_bridge1024, arg, g31.bufTwInv, "set crt inv bridge1024 tw31");
            set_karg(fk.inv_bridge1024, arg, n, "set crt inv bridge1024 n");
            const size_t global = (n / 1024u) * local_bridge1024;
            enqueue_kernel(g61, fk.inv_bridge1024, global, &local_bridge1024, "enqueue crt fused inv bridge1024", "crt_fused_inv_bridge1024");
        } else {
            set_karg_mem(fk.inv_bridge512, arg, g61.bufField, "set crt inv bridge a61");
            set_karg_mem(fk.inv_bridge512, arg, g61.bufTwInv, "set crt inv bridge tw61");
            set_karg_mem(fk.inv_bridge512, arg, g31.bufField, "set crt inv bridge a31");
            set_karg_mem(fk.inv_bridge512, arg, g31.bufTwInv, "set crt inv bridge tw31");
            set_karg(fk.inv_bridge512, arg, n, "set crt inv bridge n");
            const size_t global = (n / 512u) * local_bridge512;
            enqueue_kernel(g61, fk.inv_bridge512, global, &local_bridge512, "enqueue crt fused inv bridge512", "crt_fused_inv_bridge512");
        }
    }

    if (use_multi_lds_stage) {
        cl_uint cur_len = center_chunk;
        for (auto it = crt_lds_plan.rbegin(); it != crt_lds_plan.rend(); ++it) {
            const cl_uint radix = it->radix;
            if (it->global_stage) {
                if (radix >= 8u) {
                    cl_uint arg = 0;
                    set_karg_mem(fk.inv_r8, arg, g61.bufField, "set crt tail inv8 a61");
                    set_karg_mem(fk.inv_r8, arg, g61.bufTwInv, "set crt tail inv8 tw61");
                    set_karg_mem(fk.inv_r8, arg, g31.bufField, "set crt tail inv8 a31");
                    set_karg_mem(fk.inv_r8, arg, g31.bufTwInv, "set crt tail inv8 tw31");
                    set_karg(fk.inv_r8, arg, n, "set crt tail inv8 n");
                    set_karg(fk.inv_r8, arg, cur_len, "set crt tail inv8 len");
                    size_t global = n / 8u;
                    enqueue_kernel(g61, fk.inv_r8, global, &local64, "enqueue crt tail inverse radix8", "crt_tail_inv_radix8");
                } else if (radix >= 4u) {
                    cl_uint arg = 0;
                    set_karg_mem(fk.inv_r4, arg, g61.bufField, "set crt tail inv4 a61");
                    set_karg_mem(fk.inv_r4, arg, g61.bufTwInv, "set crt tail inv4 tw61");
                    set_karg_mem(fk.inv_r4, arg, g31.bufField, "set crt tail inv4 a31");
                    set_karg_mem(fk.inv_r4, arg, g31.bufTwInv, "set crt tail inv4 tw31");
                    set_karg(fk.inv_r4, arg, n, "set crt tail inv4 n");
                    set_karg(fk.inv_r4, arg, cur_len, "set crt tail inv4 len");
                    size_t global = n / 4u;
                    enqueue_kernel(g61, fk.inv_r4, global, &local64, "enqueue crt tail inverse radix4", "crt_tail_inv_radix4");
                } else {
                    cl_uint arg = 0;
                    set_karg_mem(fk.inv_r2, arg, g61.bufField, "set crt tail inv2 a61");
                    set_karg_mem(fk.inv_r2, arg, g61.bufTwInv, "set crt tail inv2 tw61");
                    set_karg_mem(fk.inv_r2, arg, g31.bufField, "set crt tail inv2 a31");
                    set_karg_mem(fk.inv_r2, arg, g31.bufTwInv, "set crt tail inv2 tw31");
                    set_karg(fk.inv_r2, arg, n, "set crt tail inv2 n");
                    set_karg(fk.inv_r2, arg, cur_len, "set crt tail inv2 len");
                    size_t global = n / 2u;
                    enqueue_kernel(g61, fk.inv_r2, global, &local64, "enqueue crt tail inverse radix2", "crt_tail_inv_radix2");
                }
                cur_len *= radix;
                continue;
            }

            const bool use_tile2 = (g_crt_lds_tile >= 2u) && fk.inv_lds_stage_tile2 && radix >= 16u && cur_len >= 2u;
            cl_kernel k_lds = use_tile2 ? fk.inv_lds_stage_tile2 : fk.inv_lds_stage;
            cl_uint arg = 0;
            set_karg_mem(k_lds, arg, g61.bufField, "set crt lds stage inv a61");
            set_karg_mem(k_lds, arg, g61.bufTwInv, "set crt lds stage inv tw61");
            set_karg_mem(k_lds, arg, g31.bufField, "set crt lds stage inv a31");
            set_karg_mem(k_lds, arg, g31.bufTwInv, "set crt lds stage inv tw31");
            set_karg(k_lds, arg, n, "set crt lds stage inv n");
            set_karg(k_lds, arg, cur_len, "set crt lds stage inv base len");
            set_karg(k_lds, arg, radix, "set crt lds stage inv radix");
            const size_t local_lds = 128u;
            const size_t groups = use_tile2 ? ((static_cast<size_t>(n) / (static_cast<size_t>(cur_len) * radix)) * ((cur_len + 1u) >> 1u)) : (static_cast<size_t>(n) / radix);
            const size_t global = groups * local_lds;
            enqueue_kernel(g61, k_lds, global, &local_lds, "enqueue crt lds stage inverse", use_tile2 ? crt_lds_stage_inv_tile2_label(radix) : crt_lds_stage_inv_label(radix));
            cur_len *= radix;
        }

        if (crt_head_r8_used) {
            cl_uint arg = 0;
            set_karg_mem(fk.inv_r8, arg, g61.bufField, "set crt head inv8 a61");
            set_karg_mem(fk.inv_r8, arg, g61.bufTwInv, "set crt head inv8 tw61");
            set_karg_mem(fk.inv_r8, arg, g31.bufField, "set crt head inv8 a31");
            set_karg_mem(fk.inv_r8, arg, g31.bufTwInv, "set crt head inv8 tw31");
            set_karg(fk.inv_r8, arg, n, "set crt head inv8 n");
            set_karg(fk.inv_r8, arg, cur_len, "set crt head inv8 len");
            size_t global = n / 8u;
            enqueue_kernel(g61, fk.inv_r8, global, &local64, "enqueue crt head inverse radix8", "crt_head_inv_radix8");
            cur_len <<= 3;
        }
    }

    const cl_uint inverse_stop = use_last16_unweight ? (n / 8u) : (n / 2u);
    if (use_global_radix8) {
        for (std::size_t si = 0; si < stages.size(); ++si) {
            const StageInfo& st = stages[si];
            if (st.len <= global_stop_chunk) continue;
            if (st.len >= inverse_stop) break;

            const bool can_r8 = (st.len * 4u < inverse_stop) && (st.len >= global_stop_chunk);
            if (can_r8) {
                cl_uint arg = 0;
                set_karg_mem(fk.inv_r8, arg, g61.bufField, "set crt inv8 a61");
                set_karg_mem(fk.inv_r8, arg, g61.bufTwInv, "set crt inv8 tw61");
                set_karg_mem(fk.inv_r8, arg, g31.bufField, "set crt inv8 a31");
                set_karg_mem(fk.inv_r8, arg, g31.bufTwInv, "set crt inv8 tw31");
                set_karg(fk.inv_r8, arg, n, "set crt inv8 n");
                set_karg(fk.inv_r8, arg, st.len, "set crt inv8 len");
                size_t global = n / 8u;
                enqueue_kernel(g61, fk.inv_r8, global, &local64, "enqueue crt fused inv radix8", "crt_fused_inv_radix8");
                si += 2u;
                continue;
            }

            const bool can_r4 = (st.len * 2u < inverse_stop);
            if (can_r4) {
                cl_uint arg = 0;
                set_karg_mem(fk.inv_r4, arg, g61.bufField, "set crt inv4 a61");
                set_karg_mem(fk.inv_r4, arg, g61.bufTwInv, "set crt inv4 tw61");
                set_karg_mem(fk.inv_r4, arg, g31.bufField, "set crt inv4 a31");
                set_karg_mem(fk.inv_r4, arg, g31.bufTwInv, "set crt inv4 tw31");
                set_karg(fk.inv_r4, arg, n, "set crt inv4 n");
                set_karg(fk.inv_r4, arg, st.len, "set crt inv4 len");
                size_t global = n / 4u;
                enqueue_kernel(g61, fk.inv_r4, global, &local64, "enqueue crt fused inv radix4", "crt_fused_inv_radix4");
                ++si;
                continue;
            }

            cl_uint arg = 0;
            set_karg_mem(fk.inv_r2, arg, g61.bufField, "set crt inv2 a61");
            set_karg_mem(fk.inv_r2, arg, g61.bufTwInv, "set crt inv2 tw61");
            set_karg_mem(fk.inv_r2, arg, g31.bufField, "set crt inv2 a31");
            set_karg_mem(fk.inv_r2, arg, g31.bufTwInv, "set crt inv2 tw31");
            set_karg(fk.inv_r2, arg, n, "set crt inv2 n");
            set_karg(fk.inv_r2, arg, st.len, "set crt inv2 len");
            size_t global = n / 2u;
            enqueue_kernel(g61, fk.inv_r2, global, &local64, "enqueue crt fused inv radix2", "crt_fused_inv_radix2");
        }
    } else {
        
        for (std::size_t si = 0; si < stages.size(); ++si) {
            const StageInfo& st = stages[si];
            if (st.len <= (use_boundary1024 ? 1024u : 512u)) continue;
            if (st.len >= inverse_stop) break;
            cl_uint arg = 0;
            set_karg_mem(fk.inv_r4, arg, g61.bufField, "set crt inv a61");
            set_karg_mem(fk.inv_r4, arg, g61.bufTwInv, "set crt inv tw61");
            set_karg_mem(fk.inv_r4, arg, g31.bufField, "set crt inv a31");
            set_karg_mem(fk.inv_r4, arg, g31.bufTwInv, "set crt inv tw31");
            set_karg(fk.inv_r4, arg, n, "set crt inv n");
            set_karg(fk.inv_r4, arg, st.len, "set crt inv len");
            size_t global = n / 4u;
            enqueue_kernel(g61, fk.inv_r4, global, &local64, "enqueue crt fused inv radix4", "crt_fused_inv_radix4");
            ++si;
        }
    }

    {
        cl_uint arg = 0;
        if (use_last16_unweight) {
            set_karg_mem(fk.last_unweight16, arg, g61.bufField, "set crt last16 a61");
            set_karg_mem(fk.last_unweight16, arg, g31.bufField, "set crt last16 a31");
            set_karg_mem(fk.last_unweight16, arg, g61.bufDigits, "set crt last16 digits61");
            set_karg_mem(fk.last_unweight16, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set crt last16 digits31");
            set_karg_mem(fk.last_unweight16, arg, g61.bufTwInv, "set crt last16 tw61");
            set_karg_mem(fk.last_unweight16, arg, g31.bufTwInv, "set crt last16 tw31");
            set_karg(fk.last_unweight16, arg, n, "set crt last16 n");
            set_karg(fk.last_unweight16, arg, p, "set crt last16 p");
            set_karg(fk.last_unweight16, arg, log_n, "set crt last16 log_n");
            set_karg(fk.last_unweight16, arg, g61.lr2, "set crt last16 lr2 61");
            set_karg(fk.last_unweight16, arg, g31.lr2, "set crt last16 lr2 31");
            size_t global = n / 16u;
            enqueue_kernel(g61, fk.last_unweight16, global, &local64, "enqueue crt fused last16/unweight", "crt_fused_last16_unweight");
        } else {
            const bool fuse_last_garner = parse_bool_env("PRMERS_CRT_FUSE_LAST_GARNER", false) && g61.k_crt_last_unweight_garner_segment_first_oneout;
            if (fuse_last_garner) {
                g61.crtLastUnweightPending = true;
            } else {
                const StageInfo& st = stages[stages.size() - 2];
                set_karg_mem(fk.last_unweight, arg, g61.bufDigits, "set crt last digits61");
                set_karg_mem(fk.last_unweight, arg, g31.bufDigits32 ? g31.bufDigits32 : g31.bufDigits, "set crt last digits31");
                set_karg_mem(fk.last_unweight, arg, g61.bufField, "set crt last a61");
                set_karg_mem(fk.last_unweight, arg, g61.bufTwInv, "set crt last tw61");
                set_karg_mem(fk.last_unweight, arg, g31.bufField, "set crt last a31");
                set_karg_mem(fk.last_unweight, arg, g31.bufTwInv, "set crt last tw31");
                set_karg(fk.last_unweight, arg, p, "set crt last p");
                set_karg(fk.last_unweight, arg, g61.lr2, "set crt last lr2 61");
                set_karg(fk.last_unweight, arg, g31.lr2, "set crt last lr2 31");
                set_karg(fk.last_unweight, arg, n, "set crt last n");
                set_karg(fk.last_unweight, arg, st.len, "set crt last len");
                set_karg(fk.last_unweight, arg, log_n, "set crt last log_n");
                size_t global = n / 4u;
                enqueue_kernel(g61, fk.last_unweight, global, &local64, "enqueue crt fused last/unweight", "crt_fused_last_unweight");
            }
        }
    }
    return true;
}


static void dump_u64_vector(const std::string& path, const std::vector<std::uint64_t>& v, std::size_t count) {
    std::ofstream out(path);
    if (!out) return;
    const std::size_t n = std::min(count, v.size());
    for (std::size_t i = 0; i < n; ++i) out << i << " " << v[i] << "\n";
}

static void dump_u64_diff(const std::string& path,
                          const std::vector<std::uint64_t>& ref,
                          const std::vector<std::uint64_t>& got,
                          std::size_t count) {
    std::ofstream out(path);
    if (!out) return;
    const std::size_t n = std::min({count, ref.size(), got.size()});
    for (std::size_t i = 0; i < n; ++i) {
        if (ref[i] != got[i]) {
            out << i << " ref=" << ref[i] << " got=" << got[i]
                << " xor=" << (ref[i] ^ got[i]) << "\n";
        }
    }
}

static void finish_crt_queues(GpuPrp& g61, GpuPrp& g31) {
    if (g61.queue) clFinish(g61.queue);
    if (g31.queue && g31.queue != g61.queue) clFinish(g31.queue);
}

static bool validate_crt_halfreal_one_square(GpuPrp& g61,
                                             GpuPrp& g31,
                                             const CarryConfig& carry_cfg,
                                             CrtFusedKernels& fk) {
    if (g_crt_center_mode != "halfreal") return true;

    const bool need_probe = g_crt_halfreal_autoprobe && (g_crt_halfreal_flags61 < 0 || g_crt_halfreal_flags31 < 0);
    const bool need_validate = g_crt_halfreal_validate;
    if (!need_probe && !need_validate) return true;

    struct ValidatorProfileGuard {
        GpuPrp& g61;
        GpuPrp& g31;
        bool saved61;
        bool saved31;
        bool muted;
        ValidatorProfileGuard(GpuPrp& a, GpuPrp& b, bool mute)
            : g61(a), g31(b), saved61(a.profile_kernels), saved31(b.profile_kernels), muted(mute) {
            if (muted) {
                profile_clear(g61);
                profile_clear(g31);
                g61.profile_kernels = false;
                g31.profile_kernels = false;
            }
        }
        ~ValidatorProfileGuard() {
            if (muted) {
                profile_clear(g61);
                profile_clear(g31);
                g61.profile_kernels = saved61;
                g31.profile_kernels = saved31;
                profile_clear(g61);
                profile_clear(g31);
            }
        }
    };

    const bool profile_validator = parse_bool_env("PRMERS_CRT_PROFILE_VALIDATOR", false);
    ValidatorProfileGuard validator_profile_guard(g61, g31, !profile_validator);

    
    if (g_crt_odd_radix > 1u && need_probe) {
        std::cout << "mixed CRT/PFA odd-radix path: classic halfreal autoprobe skipped "
                  << "(N is odd*2^m and digit order is CRT/PFA).\n";
    }

    const uint32_t steps = std::max<uint32_t>(1u, g_crt_halfreal_validate_iters);
    const bool halfreal_dump_always = parse_bool_env("PRMERS_CRT_HALFREAL_DUMP_ALWAYS", false);
    const std::string saved_mode = g_crt_center_mode;
    const int saved_flags61 = g_crt_halfreal_flags61;
    const int saved_flags31 = g_crt_halfreal_flags31;
    const std::vector<std::uint64_t> original = read_digits(g61);

    const bool mixed_odd_validation = (g_crt_odd_radix > 1u);
    
    
    const ibdwt::Layout exact_layout = mixed_odd_validation
        ? ibdwt::make_layout_mixed(g61.exponent_p, g_crt_odd_radix)
        : ibdwt::make_layout_for_n(g61.exponent_p, g61.n);
    if (exact_layout.n != g61.n) {
        std::ostringstream oss;
        oss << "halfreal validation layout mismatch: exact_layout.n=" << exact_layout.n
            << " gpu.n=" << g61.n << " odd=" << g_crt_odd_radix;
        throw std::runtime_error(oss.str());
    }
    const bool mixed_gpu_ref_enabled = mixed_odd_validation &&
        (g_crt_mixed_gpu_reference || parse_bool_env("PRMERS_CRT_MIXED_GPU_REFERENCE", false));
    const bool exact_cpu_ref_enabled = parse_bool_env("PRMERS_CRT_HALFREAL_CPU_REFERENCE", true);
    const cl_uint exact_cpu_ref_max_p = crt_tune::env_u32("PRMERS_CRT_HALFREAL_CPU_REF_MAX_P", 5000000u, 0u, 200000000u);
    const bool exact_cpu_ref_available = exact_cpu_ref_enabled &&
        !(mixed_odd_validation && mixed_gpu_ref_enabled) &&
        g61.exponent_p <= exact_cpu_ref_max_p;
    
    
    const cl_uint exact_cpu_ref_iters = exact_cpu_ref_available
        ? crt_tune::env_u32("PRMERS_CRT_HALFREAL_CPU_REF_ITERS", 1u, 1u, std::max<cl_uint>(1u, steps))
        : 0u;
    if (mixed_odd_validation && need_validate && !exact_cpu_ref_available && !mixed_gpu_ref_enabled) {
        std::cerr << "mixed CRT/PFA validator requires a reference. For large p use "
                  << "--crt-mixed-gpu-reference, or set PRMERS_CRT_MIXED_GPU_REFERENCE=1. "
                  << "For exact CPU reference, increase PRMERS_CRT_HALFREAL_CPU_REF_MAX_P.\n";
        return false;
    }

    auto upload_state = [&](const std::vector<std::uint64_t>& v) {
        upload_digits(g61, v);
        g31.crtInputDigits = g61.bufDigits;
        g61.crtLastUnweightPending = false;
        g61.crtCoeffPending = false;
        finish_crt_queues(g61, g31);
    };

    auto make_random_state = [&]() {
        std::vector<std::uint64_t> v(g61.n);
        std::uint64_t x = 0x9e3779b97f4a7c15ull ^ ((std::uint64_t)g61.exponent_p << 17) ^ (std::uint64_t)g61.n;
        const std::uint32_t w = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(62u, g61.min_digit_width));
        const std::uint64_t mask = (w >= 64u) ? ~0ull : ((1ull << w) - 1ull);
        for (std::size_t i = 0; i < v.size(); ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            v[i] = x & mask;
        }
        return v;
    };

    auto run_one_square = [&](const std::vector<std::uint64_t>& input,
                              const std::string& mode) -> std::vector<std::uint64_t> {
        upload_state(input);

        
        if (mode == "normal" || mode == "reference" || mode == "strict") {
            if (exact_cpu_ref_available) {
                return ibdwt::square_mod_mersenne_exact_digits(input, exact_layout);
            }

            if (mixed_odd_validation && mixed_gpu_ref_enabled) {
                const std::string saved_center = g_crt_center_mode;
                const std::string saved_row_core = g_crt_mixed_row_core;
                const std::string saved_fuse_both = g_crt_mixed_row_fuse_both;
                const cl_uint saved_center_chunk = g_crt_center_chunk;
                const cl_uint saved_lds_stage = g_crt_lds_stage;

                // Same mixed CRT/PFA digit order, but the row transform uses the
                // slow/simple generic radix-2 path. This gives a practical GPU
                // reference for huge p where exact CPU squaring is not usable.
                g_crt_center_mode = "halfreal";
                g_crt_mixed_row_core = "generic";
                g_crt_mixed_row_fuse_both = "off";
                g_crt_center_chunk = 512u;
                g_crt_lds_stage = 0u;

                auto restore_mixed_ref_options = [&]() {
                    g_crt_center_mode = saved_center;
                    g_crt_mixed_row_core = saved_row_core;
                    g_crt_mixed_row_fuse_both = saved_fuse_both;
                    g_crt_center_chunk = saved_center_chunk;
                    g_crt_lds_stage = saved_lds_stage;
                };

                try {
                    if (!enqueue_square_mod_crt_defused_fast(g61, g31, fk)) {
                        throw std::runtime_error("mixed GPU validator: generic mixed reference enqueue failed");
                    }
                    enqueue_crt_garner_carry_gpu(g61, g31, carry_cfg, true);
                    finish_crt_queues(g61, g31);
                    std::vector<std::uint64_t> out = read_digits(g61);
                    restore_mixed_ref_options();
                    return out;
                } catch (...) {
                    restore_mixed_ref_options();
                    throw;
                }
            }

            const std::string saved_center = g_crt_center_mode;
            const bool saved_strict = g_force_strict_reference;
            const bool saved_local_disabled = g_local_block_lds_disabled;
            const cl_uint saved_local_override = g_local_block_lds_override;

            g_crt_center_mode = "normal";
            g_force_strict_reference = true;
            g_local_block_lds_disabled = true;
            g_local_block_lds_override = 0u;

            
            enqueue_square_mod(g31, 0u);
            cl_event gf31_square_done = enqueue_queue_marker(g31, "halfreal validator strict gf31 square done");
            set_pending_wait_event(g61, gf31_square_done);
            if (parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false) && g31.queue != g61.queue) clFlush(g31.queue);
            enqueue_square_mod(g61, 0u);

            enqueue_crt_garner_carry_gpu(g61, g31, carry_cfg, true);
            finish_crt_queues(g61, g31);

            g_force_strict_reference = saved_strict;
            g_local_block_lds_disabled = saved_local_disabled;
            g_local_block_lds_override = saved_local_override;
            g_crt_center_mode = saved_center;
            return read_digits(g61);
        }

        g_crt_center_mode = mode;
        if (!enqueue_square_mod_crt_defused_fast(g61, g31, fk)) {
            throw std::runtime_error("halfreal validation: square enqueue failed in mode " + mode);
        }
        
        
        enqueue_crt_garner_carry_gpu(g61, g31, carry_cfg, true);
        finish_crt_queues(g61, g31);
        return read_digits(g61);
    };

    auto first_mismatch = [](const std::vector<std::uint64_t>& a,
                             const std::vector<std::uint64_t>& b) -> std::size_t {
        const std::size_t n = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < n; ++i) if (a[i] != b[i]) return i;
        return std::numeric_limits<std::size_t>::max();
    };

    auto dump_step = [&](const std::string& tag,
                         const std::vector<std::uint64_t>& before,
                         const std::vector<std::uint64_t>& normal,
                         const std::vector<std::uint64_t>& half) {
        if (!g_crt_halfreal_dump_count) return;
        dump_u64_vector(g_crt_halfreal_dump_prefix + "_input" + tag + ".txt", before, g_crt_halfreal_dump_count);
        dump_u64_vector(g_crt_halfreal_dump_prefix + "_normal" + tag + ".txt", normal, g_crt_halfreal_dump_count);
        dump_u64_vector(g_crt_halfreal_dump_prefix + "_halfreal" + tag + ".txt", half, g_crt_halfreal_dump_count);
        dump_u64_diff(g_crt_halfreal_dump_prefix + "_diff" + tag + ".txt", normal, half, std::max<std::size_t>(g_crt_halfreal_dump_count, 1024));
        std::ofstream meta(g_crt_halfreal_dump_prefix + "_meta" + tag + ".txt");
        if (meta) {
            meta << "p " << g61.exponent_p << "\n";
            meta << "n " << g61.n << "\n";
            meta << "count " << std::min(g_crt_halfreal_dump_count, before.size()) << "\n";
            meta << "flags61 " << g_crt_halfreal_flags61 << "\n";
            meta << "flags31 " << g_crt_halfreal_flags31 << "\n";
        }
    };

    try {
        if (need_probe && !mixed_odd_validation) {
            const std::vector<std::uint64_t> probe_state = make_random_state();
            const std::vector<std::uint64_t> normal = run_one_square(probe_state, "normal");

            struct Candidate { int f61; int f31; };
            std::vector<Candidate> candidates;
            candidates.reserve(g_crt_halfreal_probe_exhaustive ? 1056 : 32);
            
            
            candidates.push_back({48, 48});
            candidates.push_back({16, 16});
            for (int f = 0; f < 32; ++f) candidates.push_back({f, f});
            if (g_crt_halfreal_probe_exhaustive) {
                for (int f61 = 0; f61 < 32; ++f61) {
                    for (int f31 = 0; f31 < 32; ++f31) {
                        if (f61 == f31) continue;
                        candidates.push_back({f61, f31});
                    }
                }
            }

            bool found = false;
            std::size_t best_bad = std::numeric_limits<std::size_t>::max();
            Candidate best{saved_flags61 < 0 ? 16 : (saved_flags61 & 63), saved_flags31 < 0 ? (saved_flags61 < 0 ? 16 : (saved_flags61 & 63)) : (saved_flags31 & 63)};
            std::uint64_t best_ref = 0, best_got = 0;

            std::cout << "CRT halfreal probe: trying " << candidates.size()
                      << " convention pair(s) on one random square\n";
            for (const Candidate& c : candidates) {
                g_crt_halfreal_flags61 = c.f61;
                g_crt_halfreal_flags31 = c.f31;
                const std::vector<std::uint64_t> half = run_one_square(probe_state, "halfreal");
                const std::size_t bad = first_mismatch(normal, half);
                if (bad == std::numeric_limits<std::size_t>::max()) {
                    found = true;
                    std::cout << "CRT halfreal probe: selected " << crt_halfreal_flags_desc() << "\n";
                    break;
                }
                if (best_bad == std::numeric_limits<std::size_t>::max() || bad < best_bad) {
                    best_bad = bad;
                    best = c;
                    if (bad < normal.size()) { best_ref = normal[bad]; best_got = half[bad]; }
                }
            }

            if (!found) {
                g_crt_halfreal_flags61 = best.f61;
                g_crt_halfreal_flags31 = best.f31;
                const std::vector<std::uint64_t> half = run_one_square(probe_state, "halfreal");
                dump_step("_probe_fail", probe_state, normal, half);
                std::cerr << "CRT halfreal probe: FAIL, no convention pair matched validator reference\n"
                          << "  best tested " << crt_halfreal_flags_desc() << "\n"
                          << "  first mismatch digit=" << best_bad << "\n"
                          << "  normal=" << best_ref << "\n"
                          << "  halfreal=" << best_got << "\n"
                          << "  dump prefix: " << g_crt_halfreal_dump_prefix << "_*.txt\n";
                g_crt_halfreal_flags61 = saved_flags61;
                g_crt_halfreal_flags31 = saved_flags31;
                upload_state(original);
                g_crt_center_mode = saved_mode;
                return false;
            }
        } else {
            
            g_crt_halfreal_flags61 = (saved_flags61 < 0) ? 16 : (saved_flags61 & 63);
            g_crt_halfreal_flags31 = (saved_flags31 < 0) ? g_crt_halfreal_flags61 : (saved_flags31 & 63);
        }

        if (need_validate) {
            std::vector<std::uint64_t> normal_state = g_crt_halfreal_validate_random ? make_random_state() : original;
            std::vector<std::uint64_t> half_state = normal_state;
            const uint32_t reference_steps = exact_cpu_ref_available
                ? std::min<uint32_t>(steps, std::max<cl_uint>(1u, exact_cpu_ref_iters))
                : steps;

            for (uint32_t iter = 1; iter <= reference_steps; ++iter) {
                const std::vector<std::uint64_t> before = normal_state;
                normal_state = run_one_square(before, "normal");
                half_state   = run_one_square(half_state, "halfreal");
                if (halfreal_dump_always && iter == 1u) {
                    dump_step("_iter1", before, normal_state, half_state);
                }

                if (normal_state != half_state) {
                    const std::size_t bad = first_mismatch(normal_state, half_state);
                    dump_step("_iter" + std::to_string(iter), before, normal_state, half_state);
                    std::cerr << "CRT halfreal validator: FAIL at square " << iter << "\n"
                              << "  convention: " << crt_halfreal_flags_desc() << "\n"
                              << "  first mismatch digit=" << bad << "\n";
                    if (bad < before.size() && bad < normal_state.size() && bad < half_state.size()) {
                        std::cerr << "  input=" << before[bad] << "\n"
                                  << "  reference=" << normal_state[bad] << "\n"
                                  << "  halfreal=" << half_state[bad] << "\n";
                    }
                    std::cerr << "  dump prefix: " << g_crt_halfreal_dump_prefix << "_*.txt\n";
                    upload_state(original);
                    g_crt_center_mode = saved_mode;
                    return false;
                }
            }

            std::cout << "CRT halfreal validator: OK, " << reference_steps;
            if (exact_cpu_ref_available) {
                std::cout << " exact CPU reference square(s) match halfreal";
                if (reference_steps < steps) {
                    std::cout << " (requested " << steps
                              << "; exact CPU reference capped by PRMERS_CRT_HALFREAL_CPU_REF_ITERS)";
                }
            } else if (mixed_odd_validation && mixed_gpu_ref_enabled) {
                std::cout << " mixed GPU generic-reference square(s) match selected halfreal path";
            } else {
                std::cout << " square(s) match fallback GPU strict reference";
            }
            std::cout << (g_crt_halfreal_validate_random ? " from random state" : " from PRP seed")
                      << ", " << crt_halfreal_flags_desc() << std::endl;
        }
    } catch (...) {
        g_crt_center_mode = saved_mode;
        upload_state(original);
        throw;
    }

    g_crt_center_mode = saved_mode;
    upload_state(original);
    return true;
}


}

namespace mersenne_prp {

// BananaNTT output helpers are defined later in this namespace.
// Keep explicit prototypes here because the CRT run loop can optionally
// print res64, write checkpoints, and emit JSON before the helper bodies.
static std::string hex64(std::uint64_t x);
static std::uint64_t residue64_from_digits(const std::vector<std::uint64_t>& digits, const ibdwt::Layout& layout);
static void write_bananantt_checkpoint(std::uint32_t p, std::uint32_t iter,
                                       const std::vector<std::uint64_t>& digits,
                                       const ibdwt::Layout& layout);
static void write_bananantt_json(std::uint32_t p, const std::string& mode, const std::string& status,
                                 std::uint32_t iter, std::uint64_t res64, double seconds,
                                 const ibdwt::Layout& layout);
static void write_bananantt_json_from_digits(std::uint32_t p, const std::string& status,
                                             const std::vector<std::uint64_t>& digits,
                                             const ibdwt::Layout& layout);
static std::uint32_t best_proof_power_banana(std::uint32_t E);
static bool proof_is_in_points_banana(std::uint32_t E, std::uint32_t npower, std::uint32_t k);

struct GerbiczLiHostChecker {
    bool enabled = false;
    std::uint32_t B = 0;
    std::uint32_t r = 0;
    std::uint32_t checklevel = 1;
    std::uint32_t checkpass = 0;
    std::uint32_t last_good_iter = 0;
    std::uint32_t last_good_j = 0;
    std::uint64_t errors = 0;
    std::vector<std::uint64_t> D;
    std::vector<std::uint64_t> last_good_D;
    std::vector<std::uint64_t> last_good_state;

    static std::uint32_t choose_time_aligned_block(std::uint32_t total_iters,
                                                    std::uint32_t desired_B,
                                                    std::uint32_t root_B) {
        if (total_iters <= 1u) return 1u;
        desired_B = std::max<std::uint32_t>(desired_B, root_B);
        desired_B = std::min<std::uint32_t>(desired_B, total_iters);
        if (desired_B >= total_iters) return total_iters;
        const std::uint32_t lo = std::max<std::uint32_t>(root_B, desired_B - desired_B / 4u);
        const std::uint32_t hi = std::min<std::uint32_t>(total_iters, desired_B + desired_B / 2u + 1u);
        std::uint32_t best_B = desired_B;
        double best_score = 1e300;
        const double desired = static_cast<double>(desired_B);
        for (std::uint32_t cand = lo; cand <= hi; ++cand) {
            std::uint32_t first = total_iters % cand;
            if (first == 0u) first = cand;
            if (first < desired_B / 2u) continue;
            const double score = std::abs(static_cast<double>(first) - desired)
                               + 0.05 * std::abs(static_cast<double>(cand) - desired);
            if (score < best_score) {
                best_score = score;
                best_B = cand;
            }
            if (cand == std::numeric_limits<std::uint32_t>::max()) break;
        }
        return best_B;
    }

    void init(const ibdwt::Layout& layout, std::uint32_t total_iters) {
        enabled = g_runtime.gerbicz_enabled;
        if (!enabled) return;
        double ips_guess = g_runtime.gerbicz_estimated_ips;
        if (ips_guess <= 0.0) {
            ips_guess = (total_iters >= 100000000u) ? 510.0 :
                        (total_iters >= 10000000u)  ? 900.0 :
                        (total_iters >= 1000000u)   ? 2500.0 : 8000.0;
        }
        const double root = std::sqrt(static_cast<double>(std::max<std::uint32_t>(1u, total_iters)));
        const std::uint32_t root_B = std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::ceil(root)));
        const double target_seconds = g_runtime.gerbicz_user_seconds ? std::max(1.0, g_runtime.gerbicz_target_seconds) : std::max(BANANANTT_DEFAULT_GERBICZ_MIN_SECONDS, g_runtime.gerbicz_target_seconds);
        if (g_runtime.gerbicz_block) {
            B = std::min<std::uint32_t>(std::max<std::uint32_t>(1u, g_runtime.gerbicz_block), total_iters ? total_iters : 1u);
        } else {
            const double desired_boundary_seconds = std::max(0.05, g_runtime.gerbicz_boundary_seconds);
            double desired_raw = std::ceil(ips_guess * desired_boundary_seconds);
            if (desired_raw < static_cast<double>(root_B)) desired_raw = static_cast<double>(root_B);
            std::uint32_t max_auto_B = total_iters;
            if (total_iters >= 8192u) max_auto_B = std::max<std::uint32_t>(root_B, total_iters / 8u);
            if (desired_raw > static_cast<double>(max_auto_B)) desired_raw = static_cast<double>(max_auto_B);
            B = choose_time_aligned_block(total_iters, static_cast<std::uint32_t>(desired_raw), root_B);
        }
        if (total_iters > 0u) B = std::min<std::uint32_t>(B, total_iters);
        r = total_iters % B;
        if (r == 0u) r = B;
        if (g_runtime.gerbicz_checklevel) {
            checklevel = std::max<std::uint32_t>(1u, g_runtime.gerbicz_checklevel);
        } else {
            const double blocks_per_check = (ips_guess * target_seconds) / std::max<double>(1.0, static_cast<double>(B));
            checklevel = std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::ceil(blocks_per_check)));
        }
        D = ibdwt::from_small(1, layout);
        last_good_D = D;
        last_good_state = ibdwt::from_small(3, layout);
        last_good_iter = 0;
        last_good_j = total_iters ? (total_iters - 1u) : 0u;
        errors = 0;
        g_runtime.gerbicz_block = B;
        g_runtime.gerbicz_checklevel = checklevel;
        const double ips = ips_guess;
        const double boundary_seconds = static_cast<double>(r) / std::max(ips, 1.0);
        const double steady_boundary_seconds = static_cast<double>(B) / std::max(ips, 1.0);
        const double full_check_seconds = steady_boundary_seconds * static_cast<double>(checklevel);
        std::cout << "[Gerbicz Li] enabled: B=" << B
                  << " r=" << r
                  << " checklevel=" << checklevel
                  << " first_boundary≈" << std::fixed << std::setprecision(1) << boundary_seconds << "s"
                  << " next_boundaries≈" << steady_boundary_seconds << "s"
                  << " full_check≈" << full_check_seconds << "s"
                  << " boundary_target≈" << g_runtime.gerbicz_boundary_seconds << "s"
                  << " backend=" << (g_runtime.gerbicz_gpu_verify ? "gpu-fullcheck+gpu-D-update" : "host-exact-GMP") << std::endl;
    }

    bool boundary(std::uint32_t iter_done, std::uint32_t j_remaining, const std::vector<std::uint64_t>& state,
                  const ibdwt::Layout& layout, bool final_boundary) {
        auto host_full_check = [&](const std::vector<std::uint64_t>& D_before,
                                   std::uint32_t block, std::uint32_t rr,
                                   const ibdwt::Layout& lay) {
            std::vector<std::uint64_t> check = D_before;
            const std::uint32_t first_squares = block - rr;
            for (std::uint32_t z = 0; z < first_squares; ++z) {
                check = ibdwt::square_mod_mersenne_exact_digits(check, lay);
            }
            check = ibdwt::mul_small_mod_mersenne_exact_digits(check, 3u, lay);
            for (std::uint32_t z = 0; z < rr; ++z) {
                check = ibdwt::square_mod_mersenne_exact_digits(check, lay);
            }
            return check;
        };
        auto host_d_update = [&](const std::vector<std::uint64_t>& a,
                                 const std::vector<std::uint64_t>& b,
                                 const ibdwt::Layout& lay) {
            return ibdwt::mul_mod_mersenne_exact_digits(a, b, lay);
        };
        return boundary_with_checker_and_update(iter_done, j_remaining, state, layout, final_boundary, host_full_check, host_d_update);
    }

    template <class FullCheckFn>
    bool boundary_with_checker(std::uint32_t iter_done, std::uint32_t j_remaining,
                               const std::vector<std::uint64_t>& state,
                               const ibdwt::Layout& layout, bool final_boundary,
                               FullCheckFn&& full_check) {
        auto host_d_update = [&](const std::vector<std::uint64_t>& a,
                                 const std::vector<std::uint64_t>& b,
                                 const ibdwt::Layout& lay) {
            return ibdwt::mul_mod_mersenne_exact_digits(a, b, lay);
        };
        return boundary_with_checker_and_update(iter_done, j_remaining, state, layout, final_boundary, std::forward<FullCheckFn>(full_check), host_d_update);
    }

    template <class FullCheckFn, class DUpdateFn>
    bool boundary_with_checker_and_update(std::uint32_t iter_done, std::uint32_t j_remaining,
                               const std::vector<std::uint64_t>& state,
                               const ibdwt::Layout& layout, bool final_boundary,
                               FullCheckFn&& full_check,
                               DUpdateFn&& d_update) {
        if (!enabled) return true;
        const std::vector<std::uint64_t> D_before = D;
        const auto d_update_t0 = std::chrono::steady_clock::now();
        if (g_runtime.gerbicz_verbose) {
            std::cout << "[Gerbicz Li] boundary: iter=" << iter_done
                      << " j=" << j_remaining
                      << " updating D *= state..." << std::flush;
        }
        D = d_update(D, state, layout);
        const double d_update_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - d_update_t0).count();
        if (g_runtime.gerbicz_verbose) {
            std::cout << " done in " << std::fixed << std::setprecision(2) << d_update_s << "s" << std::endl;
        }
        ++checkpass;
        const bool condcheck = (checkpass >= checklevel) || final_boundary;
        if (!condcheck) return true;

        std::cout << "[Gerbicz Li] full Li check start..." << std::endl;
        const auto full_t0 = std::chrono::steady_clock::now();
        std::vector<std::uint64_t> check = full_check(D_before, B, r, layout);
        const double full_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - full_t0).count();
        std::cout << "[Gerbicz Li] full Li check computed in " << std::fixed << std::setprecision(2) << full_s << "s" << std::endl;
        ibdwt::canonicalize_zero(check, layout);
        ibdwt::canonicalize_zero(D, layout);

        checkpass = 0;
        if (check == D) {
            last_good_state = state;
            last_good_D = D;
            last_good_iter = iter_done;
            last_good_j = j_remaining;
            std::cout << "[Gerbicz Li] Check passed! iter=" << iter_done
                      << (g_runtime.gerbicz_gpu_verify ? " [gpu]" : " [host]") << std::endl;
            return true;
        }
        ++errors;
        ++g_runtime.gerbicz_errors;
        std::cout << "[Gerbicz Li] Mismatch\n"
                  << "[Gerbicz Li] Check FAILED! iter=" << iter_done << "\n"
                  << "[Gerbicz Li] Restore iter=" << last_good_iter << " (j=" << last_good_j << ")\n";
        D = last_good_D;
        return false;
    }
};

struct BananaBackupState {
    bool valid = false;
    bool has_gerbicz = false;
    std::uint32_t exponent = 0;
    std::uint32_t iter = 0;
    std::uint32_t n = 0;
    std::uint32_t odd = 0;
    std::uint32_t ln = 0;
    std::uint32_t B = 0;
    std::uint32_t r = 0;
    std::uint32_t checklevel = 0;
    std::uint32_t checkpass = 0;
    std::uint32_t last_good_iter = 0;
    std::uint32_t last_good_j = 0;
    std::uint64_t gerbicz_errors = 0;
    std::vector<std::uint64_t> state;
    std::vector<std::uint64_t> D;
    std::vector<std::uint64_t> last_good_D;
    std::vector<std::uint64_t> last_good_state;
};

static std::string default_bananantt_backup_path(std::uint32_t p) {
    if (!g_runtime.backup_path.empty()) return g_runtime.backup_path;
    std::filesystem::path dir = g_runtime.backup_dir.empty() ? std::filesystem::path("save") : std::filesystem::path(g_runtime.backup_dir);
    if (!g_runtime.output_dir.empty() && dir.is_relative()) dir = std::filesystem::path(g_runtime.output_dir) / dir;
    return (dir / ("M" + std::to_string(p) + ".bananantt.chk")).string();
}

static std::string default_bananantt_resume_path(std::uint32_t p) {
    if (!g_runtime.resume_path.empty()) return g_runtime.resume_path;
    return default_bananantt_backup_path(p);
}

static void backup_write_u32(std::ostream& os, std::uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void backup_write_u64(std::ostream& os, std::uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static bool backup_read_u32(std::istream& is, std::uint32_t& v) { return bool(is.read(reinterpret_cast<char*>(&v), sizeof(v))); }
static bool backup_read_u64(std::istream& is, std::uint64_t& v) { return bool(is.read(reinterpret_cast<char*>(&v), sizeof(v))); }

static void backup_write_vec(std::ostream& os, const std::vector<std::uint64_t>& v) {
    backup_write_u32(os, static_cast<std::uint32_t>(v.size()));
    if (!v.empty()) os.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(std::uint64_t)));
}

static bool backup_read_vec(std::istream& is, std::vector<std::uint64_t>& v, std::uint32_t expected_n) {
    std::uint32_t n = 0;
    if (!backup_read_u32(is, n)) return false;
    if (n != expected_n) return false;
    v.assign(n, 0);
    if (n) return bool(is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(std::uint64_t))));
    return true;
}

static bool write_bananantt_backup_file(const std::string& path,
                                        std::uint32_t p,
                                        std::uint32_t iter_done,
                                        const ibdwt::Layout& layout,
                                        const std::vector<std::uint64_t>& state,
                                        const GerbiczLiHostChecker* gerbicz) {
    if (path.empty()) return false;
    std::filesystem::path out(path);
    if (!out.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(out.parent_path(), ec);
    }
    std::filesystem::path tmp = out;
    tmp += ".tmp";
    std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    const char magic[16] = {'B','A','N','A','N','A','C','H','K','P','T','0','0','2','\0','\0'};
    os.write(magic, sizeof(magic));
    backup_write_u32(os, 2u);
    backup_write_u32(os, p);
    backup_write_u32(os, iter_done);
    backup_write_u32(os, layout.n);
    backup_write_u32(os, layout.odd);
    backup_write_u32(os, layout.ln);
    const bool has_g = gerbicz && gerbicz->enabled;
    backup_write_u32(os, has_g ? 1u : 0u);
    backup_write_u32(os, has_g ? gerbicz->B : 0u);
    backup_write_u32(os, has_g ? gerbicz->r : 0u);
    backup_write_u32(os, has_g ? gerbicz->checklevel : 0u);
    backup_write_u32(os, has_g ? gerbicz->checkpass : 0u);
    backup_write_u32(os, has_g ? gerbicz->last_good_iter : 0u);
    backup_write_u32(os, has_g ? gerbicz->last_good_j : 0u);
    backup_write_u64(os, has_g ? gerbicz->errors : 0u);
    backup_write_vec(os, state);
    backup_write_vec(os, has_g ? gerbicz->D : std::vector<std::uint64_t>{});
    backup_write_vec(os, has_g ? gerbicz->last_good_D : std::vector<std::uint64_t>{});
    backup_write_vec(os, has_g ? gerbicz->last_good_state : std::vector<std::uint64_t>{});
    os.close();
    if (!os) return false;
    std::error_code ec;
    std::filesystem::rename(tmp, out, ec);
    if (ec) {
        std::filesystem::remove(out, ec);
        ec.clear();
        std::filesystem::rename(tmp, out, ec);
    }
    return !ec;
}

static bool read_bananantt_backup_file(const std::string& path,
                                       std::uint32_t p,
                                       const ibdwt::Layout& layout,
                                       BananaBackupState& b) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    char magic[16] = {};
    if (!is.read(magic, sizeof(magic))) return false;
    const char expect[12] = {'B','A','N','A','N','A','C','H','K','P','T','0'};
    if (std::memcmp(magic, expect, sizeof(expect)) != 0) return false;
    std::uint32_t version=0, has_g=0;
    if (!backup_read_u32(is, version) || version != 2u) return false;
    if (!backup_read_u32(is, b.exponent) || b.exponent != p) return false;
    if (!backup_read_u32(is, b.iter)) return false;
    if (!backup_read_u32(is, b.n) || b.n != layout.n) return false;
    if (!backup_read_u32(is, b.odd) || b.odd != layout.odd) return false;
    if (!backup_read_u32(is, b.ln) || b.ln != layout.ln) return false;
    if (!backup_read_u32(is, has_g)) return false;
    b.has_gerbicz = has_g != 0u;
    if (!backup_read_u32(is, b.B)) return false;
    if (!backup_read_u32(is, b.r)) return false;
    if (!backup_read_u32(is, b.checklevel)) return false;
    if (!backup_read_u32(is, b.checkpass)) return false;
    if (!backup_read_u32(is, b.last_good_iter)) return false;
    if (!backup_read_u32(is, b.last_good_j)) return false;
    if (!backup_read_u64(is, b.gerbicz_errors)) return false;
    if (!backup_read_vec(is, b.state, layout.n)) return false;
    if (b.has_gerbicz) {
        if (!backup_read_vec(is, b.D, layout.n)) return false;
        if (!backup_read_vec(is, b.last_good_D, layout.n)) return false;
        if (!backup_read_vec(is, b.last_good_state, layout.n)) return false;
    } else {
        std::uint32_t z=0;
        if (!backup_read_u32(is, z) || z != 0u) return false;
        if (!backup_read_u32(is, z) || z != 0u) return false;
        if (!backup_read_u32(is, z) || z != 0u) return false;
    }
    b.valid = true;
    return true;
}

static void restore_gerbicz_from_backup(GerbiczLiHostChecker& g, const BananaBackupState& b) {
    if (!b.valid || !b.has_gerbicz || !g.enabled) return;
    g.B = b.B;
    g.r = b.r;
    g.checklevel = std::max<std::uint32_t>(1u, b.checklevel);
    g.checkpass = b.checkpass;
    g.last_good_iter = b.last_good_iter;
    g.last_good_j = b.last_good_j;
    g.errors = b.gerbicz_errors;
    g.D = b.D;
    g.last_good_D = b.last_good_D;
    g.last_good_state = b.last_good_state;
    g_runtime.gerbicz_block = g.B;
    g_runtime.gerbicz_checklevel = g.checklevel;
    g_runtime.gerbicz_errors = g.errors;
}

static void maybe_write_runtime_backup(const char* reason,
                                       clwrap::GpuPrp& gpu61,
                                       clwrap::GpuPrp& gpu31,
                                       const ibdwt::Layout& layout,
                                       std::uint32_t p,
                                       std::uint32_t iter_done,
                                       GerbiczLiHostChecker* gerbicz) {
    if (!g_runtime.backup_enabled) return;
    clwrap::finish_crt_queues(gpu61, gpu31);
    std::vector<std::uint64_t> state = clwrap::read_digits(gpu61);
    ibdwt::canonicalize_zero(state, layout);
    const std::string path = default_bananantt_backup_path(p);
    if (write_bananantt_backup_file(path, p, iter_done, layout, state, gerbicz)) {
        std::cout << "backup saved: " << path << " iter=" << iter_done;
        if (reason && *reason) std::cout << " [" << reason << "]";
        std::cout << std::endl;
    } else {
        std::cerr << "warning: could not write backup: " << path << std::endl;
    }
}

static void maybe_remove_runtime_backup(std::uint32_t p) {
    if (!g_runtime.backup_enabled) return;
    const std::string path = default_bananantt_backup_path(p);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

static void maybe_inject_error_after_iter(clwrap::GpuPrp& gpu, std::uint32_t iter_done) {
    if (!g_runtime.error_iter || g_runtime.error_injected || iter_done != g_runtime.error_iter) return;
    clwrap::check(clFinish(gpu.queue), "clFinish(before error injection)");
    const std::size_t idx = std::min<std::size_t>(g_runtime.error_limb, gpu.n ? gpu.n - 1u : 0u);
    std::uint64_t limb = 0;
    clwrap::check(clEnqueueReadBuffer(gpu.queue, gpu.bufDigits, CL_TRUE,
                                      idx * sizeof(std::uint64_t), sizeof(std::uint64_t),
                                      &limb, 0, nullptr, nullptr), "read injected limb");
    limb += g_runtime.error_delta ? g_runtime.error_delta : 1u;
    clwrap::check(clEnqueueWriteBuffer(gpu.queue, gpu.bufDigits, CL_TRUE,
                                       idx * sizeof(std::uint64_t), sizeof(std::uint64_t),
                                       &limb, 0, nullptr, nullptr), "write injected limb");
    g_runtime.error_injected = true;
    std::cout << "Injected error at iteration " << iter_done
              << " limb=" << idx << " delta=" << (g_runtime.error_delta ? g_runtime.error_delta : 1u)
              << std::endl;
}


static __int128 floor_div_pow2_i128(__int128 v, unsigned shift) {
    const __int128 base = (__int128(1) << shift);
    if (v >= 0) return v / base;
    return -(((-v) + base - 1) / base);
}

static std::vector<std::uint64_t> normalize_signed_digits_mod(std::vector<__int128> acc,
                                                              const ibdwt::Layout& layout) {
    if (acc.size() != layout.n) throw std::runtime_error("digit normalize: size mismatch");
    std::vector<std::uint64_t> out(layout.n, 0);
    __int128 carry = 0;
    for (unsigned pass = 0; pass < 16; ++pass) {
        carry = 0;
        for (std::size_t i = 0; i < layout.n; ++i) {
            const unsigned w = layout.digit_width[i];
            const __int128 base = (__int128(1) << w);
            const __int128 v = acc[i] + carry;
            const __int128 q = floor_div_pow2_i128(v, w);
            const __int128 rem = v - q * base;
            out[i] = static_cast<std::uint64_t>(rem);
            acc[i] = rem;
            carry = q;
        }
        if (carry == 0) break;
        acc[0] += carry;
    }
    if (carry != 0) throw std::runtime_error("digit normalize: carry did not settle");
    ibdwt::canonicalize_zero(out, layout);
    return out;
}

static std::vector<std::uint64_t> add_mod_digits_linear(const std::vector<std::uint64_t>& a,
                                                        const std::vector<std::uint64_t>& b,
                                                        const ibdwt::Layout& layout) {
    if (a.size() != layout.n || b.size() != layout.n) throw std::runtime_error("digit add: size mismatch");
    std::vector<__int128> acc(layout.n);
    for (std::size_t i = 0; i < layout.n; ++i) acc[i] = __int128(a[i]) + __int128(b[i]);
    return normalize_signed_digits_mod(std::move(acc), layout);
}

static std::vector<std::uint64_t> half_mod_digits_linear(const std::vector<std::uint64_t>& x,
                                                         const ibdwt::Layout& layout) {
    if (x.size() != layout.n) throw std::runtime_error("digit half: size mismatch");
    std::vector<std::uint64_t> out(layout.n, 0);
    const bool odd = !x.empty() && (x[0] & 1u);
    std::uint64_t carry = 0;
    for (std::size_t ri = 0; ri < layout.n; ++ri) {
        const std::size_t i = layout.n - 1u - ri;
        const unsigned w = layout.digit_width[i];
        out[i] = (x[i] >> 1u) | (carry << (w - 1u));
        carry = x[i] & 1u;
    }
    if (odd && layout.n) {
        const unsigned w = layout.digit_width.back();
        out.back() += (std::uint64_t(1) << (w - 1u));
    }
    ibdwt::canonicalize_zero(out, layout);
    return out;
}

static std::vector<std::uint64_t> mul_from_square_identity_digits(const std::vector<std::uint64_t>& sq_sum,
                                                                  const std::vector<std::uint64_t>& sq_a,
                                                                  const std::vector<std::uint64_t>& sq_b,
                                                                  const ibdwt::Layout& layout) {
    if (sq_sum.size() != layout.n || sq_a.size() != layout.n || sq_b.size() != layout.n) {
        throw std::runtime_error("square identity multiply: size mismatch");
    }
    std::vector<__int128> acc(layout.n);
    for (std::size_t i = 0; i < layout.n; ++i) acc[i] = __int128(sq_sum[i]) - __int128(sq_a[i]) - __int128(sq_b[i]);
    auto two_ab = normalize_signed_digits_mod(std::move(acc), layout);
    return half_mod_digits_linear(two_ab, layout);
}

static void gerbicz_gpu_square_once_crt(clwrap::GpuPrp& gpu61,
                                            clwrap::GpuPrp& gpu31,
                                            clwrap::CrtFusedKernels& crt_fused,
                                            const clwrap::CarryConfig& carry_cfg,
                                            cl_uint center_max,
                                            bool use_crt_defused_fast,
                                            bool use_crt_fused_pipeline,
                                            bool crt_split_center,
                                            bool crt_fused_center_lockstep) {
    clwrap::g_crt_mixed_skip_pack_this_square = false;
    clwrap::g_crt_mixed_carry_pack_next_request = false;
    clwrap::g_crt_mixed_carry_pack_next_done = false;
    if (use_crt_defused_fast) {
        if (!clwrap::enqueue_square_mod_crt_defused_fast(gpu61, gpu31, crt_fused)) {
            throw std::runtime_error("Gerbicz GPU full-check: CRT defused-fast pipeline rejected this transform");
        }
    } else if (use_crt_fused_pipeline) {
        if (!clwrap::enqueue_square_mod_crt_fused_gpuowl_like(gpu61, gpu31, crt_fused,
                                                              crt_split_center,
                                                              crt_fused_center_lockstep)) {
            throw std::runtime_error("Gerbicz GPU full-check: CRT fused pipeline rejected this transform");
        }
    } else {
        clwrap::enqueue_square_mod(gpu31, center_max);
        cl_event gf31_square_done = clwrap::enqueue_queue_marker(gpu31, "gerbicz crt gf31 square marker");
        clwrap::set_pending_wait_event(gpu61, gf31_square_done);
        if (parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false) && gpu31.queue != gpu61.queue) clFlush(gpu31.queue);
        clwrap::enqueue_square_mod(gpu61, center_max);
    }
    clwrap::enqueue_crt_garner_carry_gpu(gpu61, gpu31, carry_cfg, true);
}


static std::vector<std::uint64_t> gerbicz_gpu_square_digits_crt(
    clwrap::GpuPrp& gpu61,
    clwrap::GpuPrp& gpu31,
    clwrap::CrtFusedKernels& crt_fused,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    bool use_crt_defused_fast,
    bool use_crt_fused_pipeline,
    bool crt_split_center,
    bool crt_fused_center_lockstep,
    const std::vector<std::uint64_t>& input) {
    clwrap::upload_digits(gpu61, input);
    gerbicz_gpu_square_once_crt(gpu61, gpu31, crt_fused, carry_cfg, center_max,
                                use_crt_defused_fast, use_crt_fused_pipeline,
                                crt_split_center, crt_fused_center_lockstep);
    clwrap::finish_crt_queues(gpu61, gpu31);
    return clwrap::read_digits(gpu61);
}

static std::vector<std::uint64_t> gerbicz_gpu_d_update_crt(
    clwrap::GpuPrp& gpu61,
    clwrap::GpuPrp& gpu31,
    clwrap::CrtFusedKernels& crt_fused,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    bool use_crt_defused_fast,
    bool use_crt_fused_pipeline,
    bool crt_split_center,
    bool crt_fused_center_lockstep,
    const ibdwt::Layout& layout,
    const std::vector<std::uint64_t>& D,
    const std::vector<std::uint64_t>& state) {
    clwrap::finish_crt_queues(gpu61, gpu31);
    if (gpu31.queue != gpu61.queue) clwrap::release_pending_wait_event(gpu31);
    clwrap::release_pending_wait_event(gpu61);
    auto sq_D = gerbicz_gpu_square_digits_crt(gpu61, gpu31, crt_fused, carry_cfg, center_max,
                                              use_crt_defused_fast, use_crt_fused_pipeline,
                                              crt_split_center, crt_fused_center_lockstep, D);
    auto sq_state = gerbicz_gpu_square_digits_crt(gpu61, gpu31, crt_fused, carry_cfg, center_max,
                                                  use_crt_defused_fast, use_crt_fused_pipeline,
                                                  crt_split_center, crt_fused_center_lockstep, state);
    auto sum = add_mod_digits_linear(D, state, layout);
    auto sq_sum = gerbicz_gpu_square_digits_crt(gpu61, gpu31, crt_fused, carry_cfg, center_max,
                                                use_crt_defused_fast, use_crt_fused_pipeline,
                                                crt_split_center, crt_fused_center_lockstep, sum);
    auto out = mul_from_square_identity_digits(sq_sum, sq_D, sq_state, layout);
    clwrap::upload_digits(gpu61, state);
    clwrap::check(clFinish(gpu61.queue), "clFinish(Gerbicz GPU restore after D update)");
    if (gpu31.queue != gpu61.queue) clwrap::release_pending_wait_event(gpu31);
    clwrap::release_pending_wait_event(gpu61);
    return out;
}

static std::vector<std::uint64_t> gerbicz_gpu_full_check_crt(
    clwrap::GpuPrp& gpu61,
    clwrap::GpuPrp& gpu31,
    clwrap::CrtFusedKernels& crt_fused,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    bool use_crt_defused_fast,
    bool use_crt_fused_pipeline,
    bool crt_split_center,
    bool crt_fused_center_lockstep,
    const ibdwt::Layout& layout,
    const std::vector<std::uint64_t>& start_D,
    const std::vector<std::uint64_t>& restore_state,
    std::uint32_t B,
    std::uint32_t r) {
    (void)layout;
    clwrap::finish_crt_queues(gpu61, gpu31);
    if (gpu31.queue != gpu61.queue) clwrap::release_pending_wait_event(gpu31);
    clwrap::release_pending_wait_event(gpu61);

    clwrap::upload_digits(gpu61, start_D);
    clwrap::check(clFinish(gpu61.queue), "clFinish(Gerbicz GPU upload D)");

    const std::uint32_t first_squares = B - r;
    for (std::uint32_t z = 0; z < first_squares; ++z) {
        if (g_stop_requested.load(std::memory_order_relaxed)) throw InterruptedRun();
        gerbicz_gpu_square_once_crt(gpu61, gpu31, crt_fused, carry_cfg, center_max,
                                    use_crt_defused_fast, use_crt_fused_pipeline,
                                    crt_split_center, crt_fused_center_lockstep);
        if (g_runtime.gerbicz_progress && (((z + 1u) % 2048u) == 0u || (z + 1u) == first_squares)) {
            clwrap::check(clFlush(gpu61.queue), "clFlush(Gerbicz GPU first leg progress)");
            std::cout << "[Gerbicz Li] GPU first leg " << (z + 1u) << "/" << first_squares << std::endl;
        }
    }

    clwrap::enqueue_mul_small(gpu61, 3u);
    clwrap::enqueue_carry(gpu61, carry_cfg);
    clwrap::check(clFinish(gpu61.queue), "clFinish(Gerbicz GPU mul3 carry)");

    for (std::uint32_t z = 0; z < r; ++z) {
        if (g_stop_requested.load(std::memory_order_relaxed)) throw InterruptedRun();
        gerbicz_gpu_square_once_crt(gpu61, gpu31, crt_fused, carry_cfg, center_max,
                                    use_crt_defused_fast, use_crt_fused_pipeline,
                                    crt_split_center, crt_fused_center_lockstep);
        if (g_runtime.gerbicz_progress && (((z + 1u) % 2048u) == 0u || (z + 1u) == r)) {
            clwrap::check(clFlush(gpu61.queue), "clFlush(Gerbicz GPU second leg progress)");
            std::cout << "[Gerbicz Li] GPU second leg " << (z + 1u) << "/" << r << std::endl;
        }
    }

    clwrap::finish_crt_queues(gpu61, gpu31);
    std::vector<std::uint64_t> check = clwrap::read_digits(gpu61);

    clwrap::upload_digits(gpu61, restore_state);
    clwrap::check(clFinish(gpu61.queue), "clFinish(Gerbicz GPU restore current state)");
    if (gpu31.queue != gpu61.queue) clwrap::release_pending_wait_event(gpu31);
    clwrap::release_pending_wait_event(gpu61);
    return check;
}

static bool prp_mersenne_pow2_base3_gpu(std::uint32_t p, bool verbose, clwrap::GpuPrp& gpu, const clwrap::CarryConfig& carry_cfg, cl_uint center_max = 0, std::uint32_t profile_every = 0, std::uint32_t max_iters = 0) {
    if (p < 2) throw std::runtime_error("exponent must be >= 2");
    if (p == 2) return true;

    const ibdwt::Layout layout = ibdwt::make_layout(p);
    if (layout.n != gpu.n) throw std::runtime_error("layout/GPU size mismatch");

    const std::vector<std::uint64_t> init = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu, init);

    const auto t0 = std::chrono::steady_clock::now();
    constexpr std::uint32_t report_interval = 1000;
    const std::uint32_t run_iters = (max_iters && max_iters < p) ? max_iters : p;
    const bool full_run = (run_iters == p);
    const std::uint32_t effective_profile_every = gpu.profile_kernels ? (profile_every ? profile_every : report_interval) : 0;
    const bool progress_finish = parse_bool_env("PRMERS_PROGRESS_FINISH", gpu.profile_kernels);
    const bool periodic_flush = parse_bool_env("PRMERS_PERIODIC_FLUSH", false);
    for (std::uint32_t iter = 0; iter < run_iters; ++iter) {
        clwrap::enqueue_square_mod(gpu, center_max);
        clwrap::enqueue_carry(gpu, carry_cfg);

        const bool do_report = verbose && ((iter + 1) % report_interval == 0 || iter + 1 == run_iters);
        const bool do_profile_report = effective_profile_every && (((iter + 1) % effective_profile_every) == 0 || iter + 1 == run_iters);
        const bool stop_requested = g_stop_requested.load(std::memory_order_relaxed);

        if (do_report || do_profile_report || stop_requested) {
            const bool need_finish_now = stop_requested || do_profile_report || progress_finish;
            if (need_finish_now) {
                clwrap::check(clFinish(gpu.queue), stop_requested ? "clFinish(interrupt)" : "clFinish(progress)");
                clwrap::profile_flush_pending(gpu);
            }
            const auto now = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(now - t0).count();
            const double rate = static_cast<double>(iter + 1) / std::max(sec, 1e-9);
            if (do_report || stop_requested) {
                std::cout << "iter " << (iter + 1) << "/" << p
                          << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * double(iter + 1) / double(p)) << "%), elapsed "
                          << std::setprecision(2) << sec << " s, it/s " << std::setprecision(1) << rate << "\n";
            }
            if (do_profile_report || stop_requested) {
                std::ostringstream title;
                title << "Kernel profile summary at iter " << (iter + 1);
                clwrap::profile_print_summary(gpu, title.str());
            }
            if (do_report || do_profile_report || stop_requested) {
                clwrap::print_carry_stats_if_changed(gpu, iter + 1);
            }
            if (stop_requested) throw InterruptedRun();
        } else if (periodic_flush && (((iter + 1) & 255u) == 0u)) {
            clFlush(gpu.queue);
        }
    }

    clwrap::check(clFinish(gpu.queue), "clFinish(final)");
    const auto done = std::chrono::steady_clock::now();
    const double completed_sec = std::chrono::duration<double>(done - t0).count();
    if (max_iters) {
        const double completed_rate = static_cast<double>(run_iters) / std::max(completed_sec, 1e-9);
        std::cout << "completed " << run_iters << " iterations in "
                  << std::fixed << std::setprecision(3) << completed_sec << " s, final it/s "
                  << std::setprecision(1) << completed_rate << "\n";
    }
    clwrap::profile_flush_pending(gpu);
    clwrap::profile_print_summary(gpu, "Kernel profile summary (final)");
    if (!full_run) return false;
    std::vector<std::uint64_t> r = clwrap::read_digits(gpu);
    ibdwt::canonicalize_zero(r, layout);
    return ibdwt::equals_small(r, layout, 9);
}

static std::uint64_t modinv_u64(std::uint64_t a, std::uint64_t m) {
    long long t = 0, new_t = 1;
    long long r = static_cast<long long>(m);
    long long new_r = static_cast<long long>(a % m);
    while (new_r != 0) {
        const long long q = r / new_r;
        const long long tmp_t = t - q * new_t;
        t = new_t;
        new_t = tmp_t;
        const long long tmp_r = r - q * new_r;
        r = new_r;
        new_r = tmp_r;
    }
    if (r != 1) throw std::runtime_error("CRT inverse does not exist");
    if (t < 0) t += static_cast<long long>(m);
    return static_cast<std::uint64_t>(t);
}

static void crt_garner_carry_cpu(
    const std::vector<std::uint64_t>& r61,
    const std::vector<std::uint64_t>& r31,
    const ibdwt::Layout& layout,
    std::vector<std::uint64_t>& digits)
{
    constexpr std::uint64_t M61 = (1ull << 61) - 1ull;
    constexpr std::uint64_t M31 = (1ull << 31) - 1ull;
    static const std::uint64_t inv_m61_mod_m31 = modinv_u64(M61 % M31, M31);

    const std::size_t n = layout.n;
    std::vector<__uint128_t> coeff(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t a = r61[i] % M61;
        const std::uint64_t b = r31[i] % M31;
        const std::uint64_t a31 = a % M31;
        const std::uint64_t diff = (b + M31 - a31) % M31;
        const std::uint64_t t = static_cast<std::uint64_t>((static_cast<__uint128_t>(diff) * inv_m61_mod_m31) % M31);
        coeff[i] = static_cast<__uint128_t>(a) + static_cast<__uint128_t>(M61) * t;
    }

    digits.assign(n, 0);
    __uint128_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned w = layout.digit_width[i];
        const __uint128_t total = coeff[i] + carry;
        const __uint128_t mask = (static_cast<__uint128_t>(1) << w) - 1;
        digits[i] = static_cast<std::uint64_t>(total & mask);
        carry = total >> w;
    }

    
    for (unsigned pass = 0; carry != 0 && pass < 8; ++pass) {
        for (std::size_t i = 0; i < n && carry != 0; ++i) {
            const unsigned w = layout.digit_width[i];
            const __uint128_t total = static_cast<__uint128_t>(digits[i]) + carry;
            const __uint128_t mask = (static_cast<__uint128_t>(1) << w) - 1;
            digits[i] = static_cast<std::uint64_t>(total & mask);
            carry = total >> w;
        }
    }
    if (carry != 0) throw std::runtime_error("CRT CPU carry did not converge");
}


static bool prp_mersenne_pow2_base3_gpu_crt_garner(
    std::uint32_t p,
    bool verbose,
    clwrap::GpuPrp& gpu61,
    clwrap::GpuPrp& gpu31,
    const ibdwt::Layout& layout,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    cl_uint profile_every,
    std::uint32_t max_iters,
    bool crt_split_center,
    bool crt_fused_center_lockstep,
    bool crt_defused_fast)
{
    std::vector<std::uint64_t> digits = ibdwt::from_small(3, layout);
    std::uint32_t start_iter = 0;
    BananaBackupState resume_backup;
    if (g_runtime.resume_enabled) {
        const std::string resume_path = default_bananantt_resume_path(p);
        if (read_bananantt_backup_file(resume_path, p, layout, resume_backup)) {
            if (max_iters && resume_backup.iter > max_iters) {
                std::cout << "backup ignored: iter=" << resume_backup.iter << " is beyond --iters " << max_iters << "\n";
            } else {
                digits = resume_backup.state;
                start_iter = resume_backup.iter;
                std::cout << "backup restored: " << resume_path << " iter=" << start_iter << "\n";
            }
        }
    }
    clwrap::upload_digits(gpu61, digits);
    gpu31.crtInputDigits = gpu61.bufDigits;
    
    clwrap::CrtFusedKernels crt_fused(gpu61.program);
    const bool use_crt_mixed_odd = (clwrap::g_crt_odd_radix > 1u) &&
                                   crt_fused.mixed_odd_available() &&
                                   !clwrap::g_force_strict_reference;
    const bool use_crt_defused_fast = use_crt_mixed_odd ||
                                      (crt_defused_fast && crt_fused.defused_fast_available() &&
                                       !clwrap::g_force_strict_reference && layout.n >= 512u &&
                                       (clwrap::g_crt_center_chunk == 512u));
    const bool use_crt_fused_pipeline = !use_crt_defused_fast && crt_fused.available() &&
                                        !clwrap::g_force_strict_reference && !clwrap::g_disable_crt_fused_pipeline;

    if (clwrap::g_crt_center_mode == "halfreal") {
        if (!use_crt_defused_fast) {
            std::cerr << "--crt-center-mode halfreal requires the defused CRT path.\n";
            return false;
        }
        if (!clwrap::validate_crt_halfreal_one_square(gpu61, gpu31, carry_cfg, crt_fused)) {
            std::cerr << "Stopping before the PRP loop because halfreal probe/validation failed.\n";
            return false;
        }
    }

    if (use_crt_mixed_odd) {
        const bool mixed_precrt_requested = parse_bool_env("PRMERS_CRT_MIXED_PRECRT_COEFFHI", true);
        const bool mixed_precrt_split_requested = parse_bool_env("PRMERS_CRT_MIXED_PRECRT_SPLIT", false);
        const bool mixed_precrt_possible = mixed_precrt_requested && crt_fused.mixed_odd_precrt_coeffhi_available() &&
                                           gpu31.bufDigits32 && gpu61.bufWidthMask32;
        const bool mixed_precrt_split_possible = mixed_precrt_split_requested && crt_fused.mixed_residues_to_coeffhi &&
                                                 gpu31.bufDigits32 && gpu61.bufWidthMask32;
        const bool mixed_shift_lut_enabled = parse_bool_env("PRMERS_CRT_MIXED_SHIFT_LUT", clwrap::g_crt_odd_radix == 9u) &&
                                             crt_fused.mixed_odd_shift_lut_available();
        const bool mixed_tile14_shift_enabled = mixed_shift_lut_enabled &&
                                                parse_bool_env("PRMERS_CRT_MIXED_TILE14_SHIFT", true) &&
                                                clwrap::g_crt_odd_radix == 9u &&
                                                crt_fused.mixed_odd_pack_tile14_shift_available() &&
                                                crt_fused.mixed_odd_inv_precrt_coeffhi_tile14_shift;
        const bool mixed_tile14_enabled = !mixed_shift_lut_enabled &&
                                          parse_bool_env("PRMERS_CRT_MIXED_TILE14", true) &&
                                          clwrap::g_crt_odd_radix == 9u &&
                                          crt_fused.mixed_pack_odd_fwd_tile14_61 &&
                                          crt_fused.mixed_pack_odd_fwd_tile14_31 &&
                                          crt_fused.mixed_odd_inv_precrt_coeffhi_tile14;
        const bool mixed_precrt_tile14_shift_enabled = mixed_tile14_shift_enabled && mixed_precrt_possible;
        const bool mixed_precrt_garner64_enabled = mixed_precrt_tile14_shift_enabled &&
                                                   parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER", true) &&
                                                   (crt_fused.mixed_odd9_precrt_garner9seg30_pair_available() || crt_fused.mixed_odd9_precrt_garner9seg_available() || crt_fused.mixed_odd9_precrt_garner64_available());
        const bool mixed_precrt_tile14_enabled = !mixed_precrt_tile14_shift_enabled && mixed_tile14_enabled && mixed_precrt_possible;
        const bool mixed_precrt_tile7_enabled = !mixed_precrt_tile14_shift_enabled &&
                                                 !mixed_precrt_tile14_enabled &&
                                                 parse_bool_env("PRMERS_CRT_MIXED_PRECRT_TILE7", true) &&
                                                 mixed_precrt_possible &&
                                                 !mixed_shift_lut_enabled &&
                                                 crt_fused.mixed_odd_inv_precrt_coeffhi_tile7;
        const bool mixed_precrt_outpar_enabled = parse_bool_env("PRMERS_CRT_MIXED_PRECRT_OUTPAR", false) &&
                                                  mixed_precrt_possible &&
                                                  !mixed_precrt_tile14_shift_enabled &&
                                                  !mixed_precrt_tile14_enabled &&
                                                  !mixed_precrt_tile7_enabled &&
                                                  !mixed_shift_lut_enabled &&
                                                  crt_fused.mixed_odd_inv_precrt_coeffhi_outpar;
        const bool mixed_pack_tile14_shift_enabled = mixed_tile14_shift_enabled;
        const bool mixed_pack_tile14_enabled = !mixed_pack_tile14_shift_enabled && mixed_tile14_enabled && !mixed_shift_lut_enabled &&
                                             crt_fused.mixed_odd_pack_tile14_available();
        const bool mixed_pack_tile7_enabled = !mixed_pack_tile14_shift_enabled &&
                                             !mixed_pack_tile14_enabled &&
                                             parse_bool_env("PRMERS_CRT_MIXED_PACK_TILE7", true) &&
                                             !mixed_shift_lut_enabled &&
                                             clwrap::g_crt_odd_radix == 9u &&
                                             crt_fused.mixed_odd_pack_tile7_available();
        const bool mixed_pack_tile14_lmat_both_enabled = mixed_tile14_shift_enabled &&
                                                         parse_bool_env("PRMERS_CRT_MIXED_TILE14_LMAT", true) &&
                                                         parse_bool_env("PRMERS_CRT_MIXED_FUSE_PACK_BOTH", true) &&
                                                         crt_fused.mixed_odd_pack_tile14_shift_lmat_both_available();
        const bool mixed_pack_both_enabled = parse_bool_env("PRMERS_CRT_MIXED_FUSE_PACK_BOTH", true) &&
                                             (mixed_pack_tile14_lmat_both_enabled ||
                                              (!mixed_pack_tile14_shift_enabled &&
                                               ((mixed_shift_lut_enabled && crt_fused.mixed_pack_odd_fwd_both_shift != nullptr) ||
                                                (mixed_pack_tile7_enabled && crt_fused.mixed_odd_pack_tile7_both_available()))));
        const std::string mixed_fuse_both_mode = clwrap::g_crt_mixed_row_fuse_both;
        const char* pipe_center_global0 = std::getenv("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS");
        const char* pipe_center_legacy0 = std::getenv("PRMERS_CRT_MIXED_CENTER512_SINGLE_LDS");
        const bool pipe_center_global_set0 = (pipe_center_global0 && *pipe_center_global0) ||
                                             (pipe_center_legacy0 && *pipe_center_legacy0);
        const bool pipe_center_global_val0 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS",
                                                            parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SINGLE_LDS", true));
        const bool pipe_center_1lds_any0 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61",
                                                          pipe_center_global_set0 ? pipe_center_global_val0 : true) ||
                                           parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31", false);
        const bool pipe_stage_global0 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS", false);
        const char* pipe_stage_global_env0 = std::getenv("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS");
        const bool pipe_stage_global_set0 = pipe_stage_global_env0 && *pipe_stage_global_env0;
        const bool pipe_stage_1lds_any0 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61",
                                                         pipe_stage_global_set0 ? pipe_stage_global0 : true) ||
                                          parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31",
                                                         pipe_stage_global_set0 ? pipe_stage_global0 : false);
        const bool mixed_fuse_center_explicit = parse_bool_env("PRMERS_CRT_MIXED_FUSE_CENTER_BOTH", false) ||
                                                mixed_fuse_both_mode == "center" || mixed_fuse_both_mode == "all" || mixed_fuse_both_mode == "force";
        const bool mixed_fuse_stage_explicit = parse_bool_env("PRMERS_CRT_MIXED_FUSE_LDS_BOTH", false) ||
                                               mixed_fuse_both_mode == "stage" || mixed_fuse_both_mode == "all" || mixed_fuse_both_mode == "force";
        const bool pipe_center_override0 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS", false) ||
                                           parse_bool_env("PRMERS_CRT_MIXED_FUSE_OVERRIDES_SINGLE_LDS", false);
        const bool pipe_stage_override0 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS", false) ||
                                          parse_bool_env("PRMERS_CRT_MIXED_FUSE_OVERRIDES_SINGLE_LDS", false);
        const bool mixed_fuse_center_wanted = (mixed_fuse_center_explicit ||
                                              (mixed_fuse_both_mode == "auto" && !pipe_center_1lds_any0 && crt_fused.mixed_odd_center_both_available())) &&
                                             (!pipe_center_1lds_any0 || pipe_center_override0);
        const bool mixed_fuse_stage_wanted = (mixed_fuse_stage_explicit ||
                                             (mixed_fuse_both_mode == "auto" && !pipe_stage_1lds_any0 && crt_fused.mixed_odd_lds_any_both_available())) &&
                                            (!pipe_stage_1lds_any0 || pipe_stage_override0);
        const bool mixed_center_both_enabled = mixed_fuse_center_wanted && crt_fused.mixed_odd_center_both_available();
        const bool mixed_lds_both_enabled = mixed_fuse_stage_wanted &&
                                            (crt_fused.mixed_odd_lds_any_both_available() || crt_fused.mixed_odd_lds512_both_available());
        const cl_uint mixed_odd = clwrap::g_crt_odd_radix;
        const cl_uint mixed_pow2_n = (mixed_odd > 1u && (gpu61.n % mixed_odd) == 0u) ? (gpu61.n / mixed_odd) : 0u;
        const cl_uint mixed_row_m = mixed_pow2_n >> 1;
        const bool mixed_row_force_generic = (clwrap::g_crt_mixed_row_core == "generic");
        const bool mixed_row_force_lds = (clwrap::g_crt_mixed_row_core == "lds");
        const bool mixed_row_force_lds512 = (clwrap::g_crt_mixed_row_core == "lds512");
        const bool mixed_row_force_lds1024 = (clwrap::g_crt_mixed_row_core == "lds1024");
        const bool mixed_row_center_512 = (clwrap::g_crt_center_chunk == 512u);
        const bool mixed_row_center_1024 = (clwrap::g_crt_center_chunk == 1024u);
        const cl_uint mixed_row_center = mixed_row_force_lds512 ? 512u :
                                         (mixed_row_force_lds1024 ? 1024u : clwrap::g_crt_center_chunk);
        const auto mixed_valid_pow2_lds = [](cl_uint v) {
            return v == 8u || v == 16u || v == 32u || v == 64u ||
                   v == 128u || v == 256u || v == 512u || v == 1024u;
        };
        const bool mixed_row_center_valid = mixed_valid_pow2_lds(mixed_row_center);
        const bool mixed_row_force_any_lds = mixed_row_force_lds || mixed_row_force_lds512 || mixed_row_force_lds1024;
        const bool mixed_row_auto_lds = (clwrap::g_crt_mixed_row_core == "auto") &&
                                        (mixed_row_center_512 || mixed_row_center_1024) &&
                                        clwrap::g_crt_lds_stage >= 8u;
        const bool mixed_row_lds_any_enabled =
            !parse_bool_env("PRMERS_CRT_MIXED_LDS_DISABLE", false) &&
            !parse_bool_env("PRMERS_CRT_MIXED_LDS512_DISABLE", false) &&
            !mixed_row_force_generic &&
            mixed_row_center_valid && mixed_row_m >= mixed_row_center &&
            (mixed_row_force_any_lds || mixed_row_auto_lds) &&
            (clwrap::crt_halfreal_effective_flags61() & 16u) &&
            (clwrap::crt_halfreal_effective_flags31() & 16u) &&
            crt_fused.mixed_odd_lds_any_available();
        const bool mixed_row_lds1024_enabled = mixed_row_lds_any_enabled &&
                                               mixed_row_center == 1024u &&
                                               crt_fused.mixed_odd_lds1024_available();
        const bool mixed_row_lds512_enabled = mixed_row_lds_any_enabled &&
                                              !mixed_row_lds1024_enabled &&
                                              mixed_row_center == 512u &&
                                              crt_fused.mixed_odd_lds512_available();
        const std::string mixed_row_note = mixed_row_lds_any_enabled
            ? ((mixed_row_force_any_lds ? "forced-lds" : "auto-lds") + std::string("-center") + std::to_string(mixed_row_center))
            : (mixed_row_force_generic ? "forced-generic" : "generic-fallback");
        std::string mixed_row_plan;
        if (mixed_row_lds_any_enabled) {
            const cl_uint row_center = mixed_row_center;
            const cl_uint requested_stage_cap = (clwrap::g_crt_lds_stage >= 8u) ? clwrap::g_crt_lds_stage : row_center;
            const cl_uint stage_cap = std::max<cl_uint>(2u, std::min<cl_uint>(requested_stage_cap, mixed_row_m));
            cl_uint cur = mixed_row_m;
            mixed_row_plan = "fwd=";
            bool first = true;
            while (cur > row_center) {
                cl_uint radix = clwrap::floor_pow2_leq(std::min<cl_uint>(stage_cap, cur / row_center));
                if (radix < 2u) radix = 2u;
                if (!first) mixed_row_plan += "+";
                mixed_row_plan += std::to_string(radix);
                first = false;
                cur /= radix;
            }
            if (first) mixed_row_plan += "none";
            mixed_row_plan += ",center=" + std::to_string(row_center) + ",inv=reverse";
        } else {
            mixed_row_plan = "generic-radix2";
        }
        const std::string mixed_row_core_text = mixed_row_lds_any_enabled
            ? ("LDS" + std::to_string(mixed_row_center))
            : std::string("generic-radix2");
        const char* print_center_global = std::getenv("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS");
        const char* print_center_legacy = std::getenv("PRMERS_CRT_MIXED_CENTER512_SINGLE_LDS");
        const bool print_center_global_set = (print_center_global && *print_center_global) ||
                                             (print_center_legacy && *print_center_legacy);
        const bool print_center_global_val = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS",
                                                            parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SINGLE_LDS", true));
        const bool print_center_1lds_61 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61",
                                                         print_center_global_set ? print_center_global_val : true);
        const bool print_center_1lds_31 = parse_bool_env("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31", false);
        const char* print_stage_global = std::getenv("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS");
        const bool print_stage_global_set = print_stage_global && *print_stage_global;
        const bool print_stage_global_val = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS", false);
        const bool print_stage_1lds_61 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61",
                                                        print_stage_global_set ? print_stage_global_val : true);
        const bool print_stage_1lds_31 = parse_bool_env("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31",
                                                        print_stage_global_set ? print_stage_global_val : false);
        std::cout << "CRT pipeline: mixed odd CRT/PFA row half-real; odd=" << clwrap::g_crt_odd_radix
                  << ", row-core=" << mixed_row_core_text
                  << ", row-core-request=" << clwrap::g_crt_mixed_row_core
                  << ", rows use power-of-two half-real NTT, Garner/carry stays fused, queues="
                  << ((gpu31.queue != gpu61.queue) ? "async" : "shared")
                  << ", profile-queue=" << ((gpu61.profile_kernels || gpu31.profile_kernels) ? "on" : "off")
                  << ", host-sync=" << ((gpu61.profile_kernels || gpu31.profile_kernels) ? "profile-reports" : "final-only")
                  << ", tail=" << (mixed_precrt_split_possible ? "split-preCRT-coeffhi" : (mixed_precrt_possible ? "preCRT-coeffhi-anybase" : "old-unpack+Garner"))
                  << ", tile14=" << (mixed_tile14_enabled ? "on" : "off")
                  << ", tile14-shift=" << (mixed_tile14_shift_enabled ? "on" : "off")
                  << ", precrt-tile14=" << (mixed_precrt_tile14_enabled ? "on" : "off")
                  << ", precrt-tile14-shift=" << (mixed_precrt_tile14_shift_enabled ? "on" : "off")
                  << ", precrt-garner64=" << (mixed_precrt_garner64_enabled ? "on" : "off")
                  << ", precrt-garner-pair30=" << ((mixed_precrt_garner64_enabled && parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30", true)) ? "on" : "off")
                  << ", precrt-garner-pair30-smat=" << ((mixed_precrt_garner64_enabled && parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30", true) && parse_bool_env("PRMERS_CRT_MIXED_FUSE_PRECRT_GARNER_PAIR30_SMAT", true)) ? "on" : "off")
                  << ", carry-pack-next-lds=" << ((parse_bool_env("PRMERS_CRT_MIXED_PREPACK_NEXT", false) && parse_bool_env("PRMERS_CRT_MIXED_CARRY_PACK_NEXT_LDS", false)) ? "on" : "off")
                  << ", precrt-tile7=" << (mixed_precrt_tile7_enabled ? "on" : "off")
                  << ", precrt-outpar=" << (mixed_precrt_outpar_enabled ? "on" : "off")
                  << ", pack-tile14=" << (mixed_pack_tile14_enabled ? "on" : "off")
                  << ", pack-tile14-shift=" << (mixed_pack_tile14_shift_enabled ? "on" : "off")
                  << ", pack-tile14-lmat-61x31=" << (mixed_pack_tile14_lmat_both_enabled ? "on" : "off")
                  << ", pack-tile28-lmat-61x31=" << ((mixed_pack_tile14_lmat_both_enabled && parse_bool_env("PRMERS_CRT_MIXED_PACK_TILE28_61X31", true)) ? "on" : "off")
                  << ", prepack-next=" << ((layout.odd > 1u && parse_bool_env("PRMERS_CRT_MIXED_PREPACK_NEXT", false)) ? "on" : "off")
                  << ", pack-tile7=" << (mixed_pack_tile7_enabled ? "on" : "off")
                  << ", shift-lut=" << (mixed_shift_lut_enabled ? "on" : "off")
                  << ", lds-stage=" << clwrap::g_crt_lds_stage
                  << ", lds-square=" << clwrap::g_crt_center_chunk
                  << ", row-stage-plan=" << mixed_row_plan
                  << ", fast-row-note=" << mixed_row_note
                  << ", fuse-pack-both=" << (mixed_pack_both_enabled ? "on" : "off")
                  << ", row-fuse-both-mode=" << mixed_fuse_both_mode
                  << ", fuse-lds-both=" << (mixed_lds_both_enabled ? "on" : "off")
                  << ", fuse-center-both=" << (mixed_center_both_enabled ? "on" : "off")
                  << ", center-single-lds=61:" << (print_center_1lds_61 ? "on" : "off")
                  << ",31:" << (print_center_1lds_31 ? "on" : "off")
                  << ", center-split-f48=61:" << ((print_center_1lds_61 && parse_bool_env("PRMERS_CRT_MIXED_CENTER_SPLIT_F48_61",
                                                                    parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SPLIT_F48_61", false))) ? "on" : "off")
                  << ",31:" << ((print_center_1lds_31 && parse_bool_env("PRMERS_CRT_MIXED_CENTER_SPLIT_F48_31",
                                                parse_bool_env("PRMERS_CRT_MIXED_CENTER512_SPLIT_F48_31", false))) ? "on" : "off")
                  << ", center-f48-delayed-scale=61:" << (mixed_center_f48_delayed_scale_61() ? "on" : "off")
                  << ",31:" << (mixed_center_f48_delayed_scale_31() ? "on" : "off")
                  << ", center-f48-twin-sym=61:" << (mixed_center_f48_twin_symmetry_61() ? "on" : "off")
                  << ",31:" << (mixed_center_f48_twin_symmetry_31() ? "on" : "off")
                  << ", center-regA-f48=61:" << ((print_center_1lds_61 && (parse_bool_env("PRMERS_CRT_MIXED_CENTER_REGA_61", true) || parse_bool_env("PRMERS_CRT_MIXED_CENTER_PRIVATE_A_61", false))) ? "on" : "off")
                  << ", center-twinline-f48=61:" << ((print_center_1lds_61 && parse_bool_env("PRMERS_CRT_MIXED_CENTER_TWINLINE_61", true)) ? "on" : "off")
                  << ", center-regA-f48=31:" << ((print_center_1lds_31 && clwrap::crt_halfreal_effective_flags31() == 48u && parse_bool_env("PRMERS_CRT_MIXED_CENTER_REGA_31", true)) ? "on" : "off")
                  << ", center-twinline-f48=31:" << ((print_center_1lds_31 && clwrap::crt_halfreal_effective_flags31() == 48u && parse_bool_env("PRMERS_CRT_MIXED_CENTER_TWINLINE_31", true)) ? "on" : "off")
                  << ", stage-single-lds=61:" << (print_stage_1lds_61 ? "on" : "off")
                  << ",31:" << (print_stage_1lds_31 ? "on" : "off")
                  << ", digit-width-base=" << gpu61.min_digit_width << "\n";
    } else if (use_crt_defused_fast) {
        const char* sched_name = "grouped";
        if (clwrap::g_crt_defused_schedule == 1) sched_name = "interleave";
        else if (clwrap::g_crt_defused_schedule == 2) sched_name = "gf31-first";
        else if (clwrap::g_crt_defused_schedule == 3) sched_name = "gf61-dominant";
        else if (clwrap::g_crt_defused_schedule == 4) sched_name = "serial-gf31-first";
        else if (clwrap::g_crt_defused_schedule == 5) sched_name = "gf61-forward-first";
        else if (clwrap::g_crt_defused_schedule == 6) sched_name = "hybrid-fwd8";
        std::cout << "CRT pipeline: defused-fast NTT engines; GF61/GF31 run separately with radix8-preferred + LDS square512, Garner/carry stays fused, queues="
                  << ((gpu31.queue != gpu61.queue) ? "async" : "shared")
                  << ", schedule=" << sched_name
                  << ", edge-fuse=" << (clwrap::g_crt_defused_edge_fuse == 0 ? "off" : (clwrap::g_crt_defused_edge_fuse == 1 ? "forward" : (clwrap::g_crt_defused_edge_fuse == 2 ? "tail" : "both")))
                  << ", edge-radix=" << clwrap::g_crt_defused_edge_radix
                  << ", edge-mode=" << (clwrap::g_crt_defused_edge_mode == 2 ? "generic" : (clwrap::g_crt_defused_edge_mode == 1 ? "legacy" : "auto"))
                  << ", center-mode=" << clwrap::g_crt_center_mode;
        if (clwrap::g_crt_center_mode == "halfreal") {
            const bool half_tail_precrt_print = parse_bool_env("PRMERS_CRT_HALFREAL_TAIL_PRECRT", true) &&
                crt_fused.halfreal_tail_lds512_precrt && gpu61.min_digit_width == 32u && gpu61.bufWidthMask32;
            const bool half_head_precrt_print = parse_bool_env("PRMERS_CRT_HALFREAL_HEAD_PRECRT", false) &&
                crt_fused.halfreal_head_lds512_precrt;
            const char* head_stage_print = half_head_precrt_print ? "headLDS(preCRT-pack)" : "headLDS(pack+weight)";
            std::cout << ", flags=" << clwrap::crt_halfreal_flags_desc()
                      << ", stages=" << (parse_bool_env("PRMERS_CRT_HALFREAL_LDS_PAIR", true) ?
                          (std::string("halfreal-") + head_stage_print + "+residual-global+LDS512-pair(fused)+" +
                           (half_tail_precrt_print ? "tailLDS(preCRT-unpack)" : "tailLDS(unpack+unweight)")) :
                          (std::string("halfreal-") + head_stage_print + "+residual-global+LDS512-center+" +
                           (half_tail_precrt_print ? "tailLDS(preCRT-unpack)" : "tailLDS(unpack+unweight)"))) << "\n";
        } else {
            std::cout << ", stages=global-radix8-to-512+local-square\n";
        }
    } else if (use_crt_fused_pipeline) {
        const bool crt_radix8_active = clwrap::g_crt_radix8_global && crt_fused.fwd_r8 && crt_fused.inv_r8 && crt_fused.fwd_r2 && crt_fused.inv_r2 && layout.n >= 2048u;
        const bool crt_strided_lds_allowed = parse_bool_env("PRMERS_CRT_ALLOW_STRIDED_LDS_STAGE", false);
        const bool crt_multi_lds_active = crt_strided_lds_allowed && crt_radix8_active && crt_fused.fwd_lds_stage && crt_fused.inv_lds_stage && clwrap::g_crt_lds_stage >= 16u;
        const std::string center_text = std::to_string(clwrap::g_crt_center_chunk);
        const bool crt_reglds_center = parse_bool_env("PRMERS_CRT_USE_REGLDS_CENTER512", false) &&
                                       (clwrap::g_crt_center_chunk == 512u && crt_fused.center512_reglds);
        const std::string center_mode = crt_split_center ? (" with validated split center" + center_text)
            : (crt_fused_center_lockstep ? " with LDS square256 lockstep"
               : (crt_reglds_center ? " with regLDS square512" : (" with LDS square" + center_text + " dualwave")));
        std::cout << "CRT pipeline: fused GF61+GF31 stages"
                  << center_mode
                  << (crt_radix8_active ?
                      (crt_multi_lds_active ? (", experimental-multi-LDS-stage-max=" + std::to_string(clwrap::g_crt_lds_stage) + ", lds-tile=" + std::to_string(clwrap::g_crt_lds_tile) + ", head-radix8=" + std::to_string(clwrap::g_crt_head_radix8)) : (", local-square-entry=global-radix8-to-" + center_text))
                      : ", bridge=")
                  << (crt_radix8_active ? "" : ((crt_fused.boundary1024_available() && std::getenv("PRMERS_CRT_BOUNDARY512") == nullptr && ((layout.ln & 1u) == 0u) && layout.n >= 16384u) ? "1024" : "512"))
                  << ", queues=" << ((gpu31.queue != gpu61.queue) ? "async" : "shared")
                  << ", stages=" << (crt_radix8_active ? "lds-pow2+local-square" : "radix4") << "\n";
    } else {
        std::cout << "CRT pipeline: reference two-engine bridge.\n";
    }

    const auto t0 = std::chrono::steady_clock::now();
    constexpr std::uint32_t report_interval = 1000;
    const std::uint32_t run_iters = (max_iters && max_iters < p) ? max_iters : p;
    const bool full_run = (run_iters == p);
    const std::uint32_t effective_profile_every = (gpu61.profile_kernels || gpu31.profile_kernels)
        ? (profile_every ? profile_every : report_interval)
        : 0;
    const bool crt_progress_finish = parse_bool_env("PRMERS_CRT_PROGRESS_FINISH",
                                                   (gpu61.profile_kernels || gpu31.profile_kernels));
    const bool crt_periodic_flush = parse_bool_env("PRMERS_CRT_PERIODIC_FLUSH", false);
    const bool crt_mixed_prepack_next = use_crt_defused_fast && layout.odd > 1u &&
        parse_bool_env("PRMERS_CRT_MIXED_PREPACK_NEXT", false);
    const bool crt_mixed_carry_pack_next_lds = crt_mixed_prepack_next &&
        parse_bool_env("PRMERS_CRT_MIXED_CARRY_PACK_NEXT_LDS", false);
    bool crt_mixed_prepack_ready = false;

    GerbiczLiHostChecker gerbicz;
    gerbicz.init(layout, run_iters);
    if (start_iter && resume_backup.valid) restore_gerbicz_from_backup(gerbicz, resume_backup);

    auto last_backup_time = std::chrono::steady_clock::now();
    std::uint32_t last_backup_iter = start_iter;
    auto estimate_initial_ips = [&]() {
        if (g_runtime.gerbicz_estimated_ips > 0.0) return g_runtime.gerbicz_estimated_ips;
        return (run_iters >= 100000000u) ? 510.0 :
               (run_iters >= 10000000u)  ? 900.0 :
               (run_iters >= 1000000u)   ? 2500.0 : 8000.0;
    };
    std::uint32_t guard_depth = g_runtime.queue_guard_auto ? bananantt_auto_queue_guard(estimate_initial_ips(), g_runtime.queue_guard_seconds) : g_runtime.queue_guard_depth;
    std::uint32_t next_guard_iter = guard_depth ? (start_iter + guard_depth) : 0u;
    if (g_runtime.backup_enabled) {
        std::cout << "backup enabled: " << default_bananantt_backup_path(p)
                  << " every " << std::fixed << std::setprecision(0) << g_runtime.backup_every_seconds
                  << "s" << (g_runtime.backup_every_iters ? (" or " + std::to_string(g_runtime.backup_every_iters) + " iters") : "")
                  << "; resume=" << (g_runtime.resume_enabled ? "on" : "off") << std::endl;
    }
    if (guard_depth) {
        if (g_runtime.queue_guard_auto) {
            std::cout << "OpenCL queue guard: auto clFinish about every " << std::fixed << std::setprecision(1)
                      << g_runtime.queue_guard_seconds << "s, initial depth=" << guard_depth
                      << "; use --queue-guard 0 for pure bench.\n";
        } else {
            std::cout << "OpenCL queue guard: clFinish every " << guard_depth
                      << " iterations; use --queue-guard auto or --queue-guard 0.\n";
        }
    }

    for (std::uint32_t iter = start_iter; iter < run_iters; ++iter) {
        if (g_stop_requested.load(std::memory_order_relaxed)) {
            if (g_runtime.save_on_interrupt) maybe_write_runtime_backup("interrupt", gpu61, gpu31, layout, p, iter, &gerbicz);
            throw InterruptedRun();
        }

        clwrap::g_crt_mixed_skip_pack_this_square = crt_mixed_prepack_next && crt_mixed_prepack_ready;
        if (use_crt_defused_fast) {
            if (!clwrap::enqueue_square_mod_crt_defused_fast(gpu61, gpu31, crt_fused)) {
                clwrap::g_crt_mixed_skip_pack_this_square = false;
                throw std::runtime_error("CRT defused-fast pipeline rejected this transform");
            }
        } else if (use_crt_fused_pipeline) {
            if (!clwrap::enqueue_square_mod_crt_fused_gpuowl_like(gpu61, gpu31, crt_fused, crt_split_center, crt_fused_center_lockstep)) {
                throw std::runtime_error("CRT fused pipeline rejected this transform");
            }
        } else {
            
            
            clwrap::enqueue_square_mod(gpu31, center_max);
            cl_event gf31_square_done = clwrap::enqueue_queue_marker(gpu31, "crt gf31 square marker");
            clwrap::set_pending_wait_event(gpu61, gf31_square_done);
            if (parse_bool_env("PRMERS_CRT_ALLOW_HOST_FLUSH", false) && gpu31.queue != gpu61.queue) clFlush(gpu31.queue);
            clwrap::enqueue_square_mod(gpu61, center_max);
        }
        clwrap::g_crt_mixed_skip_pack_this_square = false;
        clwrap::g_crt_mixed_carry_pack_next_request = crt_mixed_carry_pack_next_lds && ((iter + 1u) < run_iters);
        clwrap::g_crt_mixed_carry_pack_next_done = false;
        clwrap::enqueue_crt_garner_carry_gpu(gpu61, gpu31, carry_cfg, true);
        clwrap::g_crt_mixed_carry_pack_next_request = false;
        if (crt_mixed_prepack_next && (iter + 1u) < run_iters) {
            if (crt_mixed_carry_pack_next_lds && clwrap::g_crt_mixed_carry_pack_next_done) {
                crt_mixed_prepack_ready = true;
            } else {
                if (!clwrap::enqueue_crt_mixed_pack_next_after_carry(gpu61, gpu31, crt_fused)) {
                    throw std::runtime_error("PRMERS_CRT_MIXED_PREPACK_NEXT requested but pack-next tile28 61x31 path is unavailable");
                }
                crt_mixed_prepack_ready = true;
            }
        } else {
            crt_mixed_prepack_ready = false;
        }

        const std::uint32_t iter_done_now = iter + 1u;
        maybe_inject_error_after_iter(gpu61, iter_done_now);
        if (gerbicz.enabled) {
            const std::uint32_t j_remaining_now = run_iters - iter_done_now;
            const bool gerbicz_boundary = ((j_remaining_now != 0u && (j_remaining_now % gerbicz.B) == 0u) || iter_done_now == run_iters);
            if (gerbicz_boundary) {
                clwrap::check(clFinish(gpu61.queue), "clFinish(Gerbicz-Li boundary)");
                std::vector<std::uint64_t> gli_state = clwrap::read_digits(gpu61);
                ibdwt::canonicalize_zero(gli_state, layout);
                bool gerbicz_ok = true;
                if (g_runtime.gerbicz_gpu_verify) {
                    auto gpu_full_check = [&](const std::vector<std::uint64_t>& D_before,
                                              std::uint32_t block, std::uint32_t rr,
                                              const ibdwt::Layout& lay) {
                        if (g_runtime.gerbicz_verbose) {
                            std::cout << "[Gerbicz Li] GPU full check start: B=" << block
                                      << " r=" << rr << " iter=" << iter_done_now << std::endl;
                        }
                        return gerbicz_gpu_full_check_crt(gpu61, gpu31, crt_fused, carry_cfg,
                                                          center_max, use_crt_defused_fast,
                                                          use_crt_fused_pipeline,
                                                          crt_split_center,
                                                          crt_fused_center_lockstep,
                                                          lay, D_before, gli_state, block, rr);
                    };
                    auto gpu_d_update = [&](const std::vector<std::uint64_t>& a,
                                            const std::vector<std::uint64_t>& b,
                                            const ibdwt::Layout& lay) {
                        if (g_runtime.gerbicz_verbose) std::cout << " [gpu3sq]" << std::flush;
                        return gerbicz_gpu_d_update_crt(gpu61, gpu31, crt_fused, carry_cfg,
                                                        center_max, use_crt_defused_fast,
                                                        use_crt_fused_pipeline,
                                                        crt_split_center,
                                                        crt_fused_center_lockstep,
                                                        lay, a, b);
                    };
                    gerbicz_ok = gerbicz.boundary_with_checker_and_update(iter_done_now, j_remaining_now,
                                                               gli_state, layout,
                                                               iter_done_now == run_iters,
                                                               gpu_full_check,
                                                               gpu_d_update);
                } else {
                    gerbicz_ok = gerbicz.boundary(iter_done_now, j_remaining_now, gli_state,
                                                  layout, iter_done_now == run_iters);
                }
                if (!gerbicz_ok) {
                    clwrap::upload_digits(gpu61, gerbicz.last_good_state);
                    clwrap::check(clFinish(gpu61.queue), "clFinish(Gerbicz-Li restore)");
                    crt_mixed_prepack_ready = false;
                    if (gpu31.queue != gpu61.queue) clwrap::release_pending_wait_event(gpu31);
                    iter = (gerbicz.last_good_iter == 0u) ? std::numeric_limits<std::uint32_t>::max()
                                                          : (gerbicz.last_good_iter - 1u);
                    continue;
                }
            }
        }

        const bool do_report = verbose && ((iter + 1) % report_interval == 0 || iter + 1 == run_iters);
        const bool do_profile_report = effective_profile_every && (((iter + 1) % effective_profile_every) == 0 || iter + 1 == run_iters);
        if (do_report || do_profile_report) {
            const bool need_finish_now = do_profile_report || crt_progress_finish;
            if (need_finish_now) {
                clwrap::check(clFinish(gpu61.queue), "clFinish(crt report)");
                clwrap::profile_flush_pending(gpu61);
                clwrap::profile_flush_pending(gpu31);
            }
        }
        if (do_report) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            const double ips = static_cast<double>((iter + 1u) - start_iter) / std::max(elapsed, 1e-9);
            std::cout << "iter " << (iter + 1) << "/" << p << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * static_cast<double>(iter + 1) / p) << "%), elapsed "
                      << std::setprecision(2) << elapsed << " s, it/s " << std::setprecision(1) << ips
                      << " [mixed CRT/PFA half-real]\n";
        }
        if (do_profile_report) {
            clwrap::profile_print_summary(gpu61, "CRT kernel profile summary at iter " + std::to_string(iter + 1));
            if (clwrap::profile_has_data(gpu31))
                clwrap::profile_print_summary(gpu31, "CRT GF31 auxiliary profile at iter " + std::to_string(iter + 1));
        }

        const std::uint32_t iter_done = iter + 1u;
        const std::uint32_t proof_power_eff = g_runtime.proof_power ? g_runtime.proof_power : best_proof_power_banana(p);
        const bool want_res64_now = g_runtime.res64_every && ((iter_done % g_runtime.res64_every) == 0u || iter_done == run_iters);
        const bool want_proof_now = g_runtime.proof_checkpoints && proof_is_in_points_banana(p, proof_power_eff, iter_done);
        const bool want_gerbicz_note = g_runtime.gerbicz_interval && ((iter_done % g_runtime.gerbicz_interval) == 0u || iter_done == run_iters);
        if (want_res64_now || want_proof_now || want_gerbicz_note) {
            clwrap::check(clFinish(gpu61.queue), "clFinish(optional residue/proof checkpoint)");
            std::vector<std::uint64_t> chk = clwrap::read_digits(gpu61);
            ibdwt::canonicalize_zero(chk, layout);
            const std::uint64_t r64 = residue64_from_digits(chk, layout);
            g_runtime.last_iter = iter_done;
            g_runtime.last_res64 = r64;
            if (want_res64_now || want_gerbicz_note) {
                std::cout << "res64 iter " << iter_done << ": " << hex64(r64);
                if (want_gerbicz_note) std::cout << "  [Gerbicz checkpoint hook]";
                std::cout << "\n";
            }
            if (want_proof_now) write_bananantt_checkpoint(p, iter_done, chk, layout);
        } else if (crt_periodic_flush && (((iter + 1) & 255u) == 0u)) {
            clFlush(gpu61.queue);
            if (gpu31.queue != gpu61.queue) clFlush(gpu31.queue);
        }

        const std::uint32_t completed_now = iter + 1u;
        if (guard_depth && completed_now >= next_guard_iter) {
            clwrap::finish_crt_queues(gpu61, gpu31);
            if (g_stop_requested.load(std::memory_order_relaxed)) {
                if (g_runtime.save_on_interrupt) maybe_write_runtime_backup("interrupt", gpu61, gpu31, layout, p, completed_now, &gerbicz);
                throw InterruptedRun();
            }
            if (g_runtime.queue_guard_auto) {
                const auto guard_now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(guard_now - t0).count();
                const double ips = static_cast<double>(completed_now - start_iter) / std::max(elapsed, 1e-9);
                guard_depth = bananantt_auto_queue_guard(ips, g_runtime.queue_guard_seconds);
            }
            next_guard_iter = completed_now + guard_depth;
        }
        if (g_runtime.backup_enabled && completed_now < run_iters) {
            const auto backup_now = std::chrono::steady_clock::now();
            const double since_backup = std::chrono::duration<double>(backup_now - last_backup_time).count();
            const bool by_time = g_runtime.backup_every_seconds > 0.0 && since_backup >= g_runtime.backup_every_seconds;
            const bool by_iter = g_runtime.backup_every_iters && (completed_now - last_backup_iter) >= g_runtime.backup_every_iters;
            if (by_time || by_iter) {
                maybe_write_runtime_backup(by_time ? "periodic-time" : "periodic-iter", gpu61, gpu31, layout, p, completed_now, &gerbicz);
                last_backup_time = std::chrono::steady_clock::now();
                last_backup_iter = completed_now;
            }
        }
    }

    clwrap::check(clFinish(gpu61.queue), "clFinish(crt final)");
    const auto done = std::chrono::steady_clock::now();
    const double completed_sec = std::chrono::duration<double>(done - t0).count();
    if (max_iters) {
        const double completed_rate = static_cast<double>(run_iters - start_iter) / std::max(completed_sec, 1e-9);
        std::cout << "completed " << run_iters << " iterations in "
                  << std::fixed << std::setprecision(3) << completed_sec << " s, final it/s "
                  << std::setprecision(1) << completed_rate << " [mixed CRT/PFA half-real]\n";
    }
    clwrap::profile_flush_pending(gpu61);
    clwrap::profile_flush_pending(gpu31);
    clwrap::release_pending_wait_event(gpu31);
    clwrap::profile_print_summary(gpu61, "CRT kernel profile summary (final)");
    if (clwrap::profile_has_data(gpu31)) clwrap::profile_print_summary(gpu31, "CRT GF31 auxiliary profile (final)");

    if (!full_run) {
        if (g_runtime.json_enabled || g_runtime.res64_every || g_runtime.proof_checkpoints) {
            std::vector<std::uint64_t> out = clwrap::read_digits(gpu61);
            ibdwt::canonicalize_zero(out, layout);
            const std::uint64_t r64 = residue64_from_digits(out, layout);
            g_runtime.last_iter = run_iters;
            g_runtime.last_res64 = r64;
            if (g_runtime.proof_checkpoints) write_bananantt_checkpoint(p, run_iters, out, layout);
            write_bananantt_json_from_digits(p, "benchmark-stopped", out, layout);
        }
        return false;
    }
    std::vector<std::uint64_t> out = clwrap::read_digits(gpu61);
    ibdwt::canonicalize_zero(out, layout);
    const std::uint64_t r64 = residue64_from_digits(out, layout);
    g_runtime.last_iter = run_iters;
    g_runtime.last_res64 = r64;
    if (g_runtime.proof_checkpoints) write_bananantt_checkpoint(p, run_iters, out, layout);
    const bool prp_ok = ibdwt::equals_small(out, layout, 9);
    write_bananantt_json_from_digits(p, prp_ok ? "PRP" : "composite-or-error", out, layout);
    maybe_remove_runtime_backup(p);
    return prp_ok;
}


static bool prp_mersenne_pow2_base3_gpu_crt_cpu_garner(
    std::uint32_t p,
    bool verbose,
    clwrap::GpuPrp& gpu61,
    clwrap::GpuPrp& gpu31,
    const ibdwt::Layout& layout,
    cl_uint center_max,
    cl_uint profile_every,
    std::uint32_t max_iters = 0)
{
    std::vector<std::uint64_t> digits = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu61, digits);
    clwrap::upload_digits(gpu31, digits);
    gpu31.crtInputDigits = nullptr;
    clFinish(gpu61.queue);
    clFinish(gpu31.queue);

    const auto t0 = std::chrono::steady_clock::now();
    constexpr std::uint32_t report_interval = 1000;
    const std::uint32_t run_iters = (max_iters && max_iters < p) ? max_iters : p;
    const bool full_run = (run_iters == p);
    const std::uint32_t effective_profile_every = (gpu61.profile_kernels || gpu31.profile_kernels)
        ? (profile_every ? profile_every : report_interval)
        : 0;

    for (std::uint32_t iter = 0; iter < run_iters; ++iter) {
        if (g_stop_requested.load(std::memory_order_relaxed)) throw InterruptedRun();

        clwrap::enqueue_square_mod(gpu61, center_max);
        clwrap::enqueue_square_mod(gpu31, center_max);
        clFinish(gpu61.queue);
        clFinish(gpu31.queue);

        const std::vector<std::uint64_t> r61 = clwrap::read_digits(gpu61);
        const std::vector<std::uint64_t> r31 = clwrap::read_digits(gpu31);
        crt_garner_carry_cpu(r61, r31, layout, digits);

        clwrap::upload_digits(gpu61, digits);
        clwrap::upload_digits(gpu31, digits);

        const bool do_report = verbose && ((iter + 1) % report_interval == 0 || iter + 1 == run_iters);
        const bool do_profile_report = effective_profile_every && (((iter + 1) % effective_profile_every) == 0 || iter + 1 == run_iters);
        if (do_report || do_profile_report) {
            clFinish(gpu61.queue);
            clFinish(gpu31.queue);
            clwrap::profile_flush_pending(gpu61);
            clwrap::profile_flush_pending(gpu31);
        }
        if (do_report) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            const double ips = static_cast<double>(iter + 1) / std::max(elapsed, 1e-9);
            std::cout << "iter " << (iter + 1) << "/" << p << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * static_cast<double>(iter + 1) / p) << "%), elapsed "
                      << std::setprecision(2) << elapsed << " s, it/s " << std::setprecision(1) << ips
                      << " [CRT CPU-Garner debug]\n";
        }
        if (do_profile_report) {
            clwrap::profile_print_summary(gpu61, "GF61 kernel profile summary at iter " + std::to_string(iter + 1));
            clwrap::profile_print_summary(gpu31, "GF31 kernel profile summary at iter " + std::to_string(iter + 1));
        }
    }

    clFinish(gpu61.queue);
    clFinish(gpu31.queue);
    clwrap::profile_flush_pending(gpu61);
    clwrap::profile_flush_pending(gpu31);
    clwrap::profile_print_summary(gpu61, "GF61 kernel profile summary (final)");
    clwrap::profile_print_summary(gpu31, "GF31 kernel profile summary (final)");
    if (!full_run) return false;
    ibdwt::canonicalize_zero(digits, layout);
    return ibdwt::equals_small(digits, layout, 9);
}

static void selftest(const clwrap::DeviceInfo& dev, const std::string& kernel_path, cl_uint carry_block_override, cl_uint carry_items_override) {
    struct Case { std::uint32_t p; bool expect; };
    const std::vector<Case> cases = {
        {3, true}, {5, true}, {7, true}, {11, false}, {13, true}, {17, true},
        {19, true}, {23, false}, {31, true}, {61, true}, {89, true}, {107, true}, {127, true}
    };
    for (const auto& c : cases) {
        const auto layout = ibdwt::make_layout(c.p);
        auto gpu = clwrap::make_gpu(dev, kernel_path, layout, false);
        const auto carry_cfg = clwrap::choose_carry_config(dev, layout.n, carry_block_override, carry_items_override);
        const bool got = prp_mersenne_pow2_base3_gpu(c.p, false, gpu, carry_cfg, 0, 0);
        if (got != c.expect) {
            std::ostringstream oss;
            oss << "In-place fused GPU PRP self-test failed at p=" << c.p << ": got=" << got << ", expected=" << c.expect;
            throw std::runtime_error(oss.str());
        }
    }
}


static std::string hex64(std::uint64_t x) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << x;
    return os.str();
}

static std::string hex64_plain_upper(std::uint64_t x) {
    std::ostringstream os;
    os << std::uppercase << std::hex << std::setw(16) << std::setfill('0') << x;
    return os.str();
}

static std::string json_escape_banana(const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default: o << c; break;
        }
    }
    o << '"';
    return o.str();
}

static std::string lower_hex(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::uint64_t residue64_from_digits(const std::vector<std::uint64_t>& digits, const ibdwt::Layout& layout) {
    std::uint64_t r = 0;
    std::uint32_t bit = 0;
    const std::size_t m = std::min(digits.size(), layout.digit_width.size());
    for (std::size_t i = 0; i < m && bit < 64u; ++i) {
        const std::uint32_t w = layout.digit_width[i];
        const std::uint64_t mask = (w >= 64u) ? ~0ull : ((1ull << w) - 1ull);
        r |= (digits[i] & mask) << bit;
        bit += w;
    }
    return r;
}

static std::vector<std::uint32_t> residue_words32_from_digits(const std::vector<std::uint64_t>& digits, const ibdwt::Layout& layout) {
    const std::size_t words = (layout.p + 31u) / 32u;
    std::vector<std::uint32_t> out(words, 0u);
    std::uint64_t bitpos = 0;
    const std::size_t m = std::min(digits.size(), layout.digit_width.size());
    for (std::size_t i = 0; i < m; ++i) {
        const std::uint32_t w = layout.digit_width[i];
        std::uint64_t v = digits[i] & ((w >= 64u) ? ~0ull : ((1ull << w) - 1ull));
        std::uint32_t remain = w;
        while (remain && bitpos < layout.p) {
            const std::size_t wi = static_cast<std::size_t>(bitpos >> 5);
            const std::uint32_t off = static_cast<std::uint32_t>(bitpos & 31u);
            const std::uint32_t take = std::min<std::uint32_t>(remain, 32u - off);
            const std::uint32_t chunk_mask = (take == 32u) ? 0xffffffffu : ((1u << take) - 1u);
            out[wi] |= static_cast<std::uint32_t>(v & chunk_mask) << off;
            v >>= take;
            bitpos += take;
            remain -= take;
        }
    }
    return out;
}

static inline std::uint32_t mod3_words_banana(const std::vector<std::uint32_t>& W) {
    std::uint32_t r = 0;
    for (std::uint32_t w : W) r = (r + (w % 3u)) % 3u;
    return r;
}

static inline void div3_words_banana(std::uint32_t exponent, std::vector<std::uint32_t>& W) {
    if (W.empty()) return;
    std::uint32_t r = (3u - mod3_words_banana(W)) % 3u;
    const int top_bits = static_cast<int>(exponent % 32u);
    {
        const std::uint64_t t = (static_cast<std::uint64_t>(r) << top_bits) + W.back();
        W.back() = static_cast<std::uint32_t>(t / 3u);
        r = static_cast<std::uint32_t>(t % 3u);
    }
    for (auto it = W.rbegin() + 1; it != W.rend(); ++it) {
        const std::uint64_t t = (static_cast<std::uint64_t>(r) << 32u) + *it;
        *it = static_cast<std::uint32_t>(t / 3u);
        r = static_cast<std::uint32_t>(t % 3u);
    }
}

static inline void prp3_div9_words_banana(std::uint32_t exponent, std::vector<std::uint32_t>& W) {
    div3_words_banana(exponent, W);
    div3_words_banana(exponent, W);
}

static std::string format_res64_words_plain_upper(const std::vector<std::uint32_t>& W) {
    const std::uint64_t r64 = (static_cast<std::uint64_t>(W.size() > 1 ? W[1] : 0u) << 32u) |
                              static_cast<std::uint64_t>(W.empty() ? 0u : W[0]);
    return hex64_plain_upper(r64);
}

static std::string format_res2048_words_lower(const std::vector<std::uint32_t>& W) {
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0');
    for (int i = 63; i >= 0; --i) {
        const std::uint32_t w = (static_cast<std::size_t>(i) < W.size()) ? W[static_cast<std::size_t>(i)] : 0u;
        oss << std::setw(8) << w;
    }
    return oss.str();
}

static bool words_equal_one(const std::vector<std::uint32_t>& W) {
    if (W.empty() || W[0] != 1u) return false;
    for (std::size_t i = 1; i < W.size(); ++i) if (W[i] != 0u) return false;
    return true;
}

static std::uint32_t best_proof_power_banana(std::uint32_t E) {
    if (E == 0u) return 6u;
    int power = 10 + static_cast<int>(std::floor(std::log2(static_cast<double>(E) / 60e6) / 2.0));
    power = std::max(power, 2);
    power = std::min(power, 12);
    return static_cast<std::uint32_t>(power);
}

static bool proof_is_in_points_banana(std::uint32_t E, std::uint32_t npower, std::uint32_t k) {
    if (k == E) return true;
    std::uint32_t start = 0;
    for (std::uint32_t p = 0, span = (E + 1u) / 2u; p < npower; ++p, span = (span + 1u) / 2u) {
        if (k > start + span) start += span;
        else if (k == start + span) return true;
    }
    return false;
}

static std::string current_timestamp_utc_banana() {
    const std::time_t now = std::time(nullptr);
    std::tm timeinfo{};
#if defined(_WIN32)
    gmtime_s(&timeinfo, &now);
#else
    std::tm* tmp = std::gmtime(&now);
    if (tmp) timeinfo = *tmp;
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
    return std::string(buf);
}

static std::string os_name_banana() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown";
#endif
}

static std::string os_arch_banana() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

static std::string shell_quote_banana(const std::string& path) {
    std::string q = "'";
    for (char c : path) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    q += "'";
    return q;
}

static std::string md5_file_hex_banana(const std::string& path) {
    if (path.empty()) return "";
    const std::string cmd = "md5sum " + shell_quote_banana(path) + " 2>/dev/null";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return "";
    char buf[128] = {0};
    std::string out;
    if (fgets(buf, sizeof(buf), fp)) out = buf;
    pclose(fp);
    if (out.size() < 32) return "";
    std::string h = out.substr(0, 32);
    for (char c : h) if (!std::isxdigit(static_cast<unsigned char>(c))) return "";
    return lower_hex(h);
}

static void ensure_parent_dir_exists(const std::string& path) {
    std::filesystem::path p(path);
    std::filesystem::path parent = p.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
}

static void ensure_dir_exists(const std::string& dir) {
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
}

static std::string default_json_path(std::uint32_t p) {
    const std::string name = "prmers_bananantt_M" + std::to_string(p) + ".json";
    if (g_runtime.output_dir.empty()) return name;
    return (std::filesystem::path(g_runtime.output_dir) / name).string();
}

static std::string default_proof_dir(std::uint32_t p) {
    // Same tree/checkpoint directory shape as PrMers: <exponent>/proof/...
    const std::filesystem::path rel = std::filesystem::path(std::to_string(p)) / "proof";
    if (g_runtime.output_dir.empty()) return rel.string();
    return (std::filesystem::path(g_runtime.output_dir) / rel).string();
}

static std::uint32_t bananantt_crc32(const void* data, std::size_t len) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < len; ++i) {
        crc ^= p[i];
        for (int j = 0; j < 8; ++j) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static void write_bananantt_checkpoint(std::uint32_t p, std::uint32_t iter,
                                       const std::vector<std::uint64_t>& digits,
                                       const ibdwt::Layout& layout) {
    if (!g_runtime.proof_checkpoints) return;
    std::string dir = g_runtime.proof_dir.empty() ? default_proof_dir(p) : g_runtime.proof_dir;
    ensure_dir_exists(dir);
    const std::string path = dir + "/M" + std::to_string(p) + "_iter_" + std::to_string(iter) + ".chk";
    const auto words = residue_words32_from_digits(digits, layout);

    // PrMers ProofSet-compatible checkpoint: filename is just the iteration,
    // content is CRC32 followed by packed little-endian uint32 residue words.
    {
        const std::string prm_path = dir + "/" + std::to_string(iter);
        std::ofstream prm(prm_path, std::ios::binary);
        if (!prm) throw std::runtime_error("cannot write PrMers-style checkpoint: " + prm_path);
        const std::uint32_t crc = bananantt_crc32(words.data(), words.size() * sizeof(std::uint32_t));
        prm.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
        prm.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
        g_runtime.last_proof_file = prm_path;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write proof checkpoint: " + path);
    out << "BANANANTT_PRP_CHECKPOINT_V1\n";
    out << "p=" << p << "\n";
    out << "iter=" << iter << "\n";
    out << "words32=" << words.size() << "\n";
    out << "crc32=0x" << std::hex << std::setw(8) << std::setfill('0')
        << bananantt_crc32(words.data(), words.size() * sizeof(std::uint32_t)) << std::dec << "\n";
    out << "res64=" << hex64(residue64_from_digits(digits, layout)) << "\n";
    out << "binary_words_le_after_this_line\n";
    out.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
}

static void write_bananantt_json(std::uint32_t p, const std::string& mode, const std::string& status,
                                 std::uint32_t iter, std::uint64_t res64, double seconds,
                                 const ibdwt::Layout& layout) {
    (void)mode; (void)iter; (void)seconds;
    std::vector<std::uint32_t> words((layout.p + 31u) / 32u, 0u);
    if (!words.empty()) words[0] = static_cast<std::uint32_t>(res64 & 0xffffffffu);
    if (words.size() > 1) words[1] = static_cast<std::uint32_t>(res64 >> 32u);
    prp3_div9_words_banana(p, words);
    const std::string st = (status == "benchmark-stopped") ? "U" : (words_equal_one(words) ? "P" : "C");
    const std::string res64_prp = format_res64_words_plain_upper(words);
    const std::string res2048 = format_res2048_words_lower(words);
    const std::string timestamp = current_timestamp_utc_banana();
    const std::uint32_t proof_power = g_runtime.proof_power ? g_runtime.proof_power : best_proof_power_banana(p);
    const bool have_proof = g_runtime.proof_checkpoints && !g_runtime.last_proof_file.empty();
    const std::string proof_md5 = have_proof ? md5_file_hex_banana(g_runtime.last_proof_file) : "";
    std::ostringstream prefix;
    prefix << "{\"status\":" << json_escape_banana(st)
           << ",\"exponent\":" << p
           << ",\"worktype\":\"PRP-3\""
           << ",\"res64\":" << json_escape_banana(res64_prp)
           << ",\"res2048\":" << json_escape_banana(res2048)
           << ",\"residue-type\":1"
           << ",\"errors\":{\"gerbicz\":" << static_cast<unsigned long long>(g_runtime.gerbicz_errors) << "}"
           << ",\"shift-count\":0"
           << ",\"fft-length\":" << layout.n;
    if (have_proof) {
        prefix << ",\"proof\":{\"version\":2,\"power\":" << proof_power
               << ",\"hashsize\":64,\"md5\":" << json_escape_banana(proof_md5) << "}";
    }
    prefix << ",\"program\":{\"name\":" << json_escape_banana(BANANANTT_PROGRAM_NAME)
           << ",\"version\":" << json_escape_banana(BANANANTT_PROGRAM_VERSION)
           << ",\"port\":" << BANANANTT_PROGRAM_PORT << "}"
           << ",\"os\":{\"os\":" << json_escape_banana(os_name_banana())
           << ",\"architecture\":" << json_escape_banana(os_arch_banana()) << "}"
           << ",\"timestamp\":" << json_escape_banana(timestamp);
    std::ostringstream canon;
    canon << p << ";PRP;;;" << lower_hex(res64_prp) << ";" << lower_hex(res2048)
          << ";0_3_1;" << layout.n << ";gerbicz:" << g_runtime.gerbicz_errors << ";"
          << BANANANTT_PROGRAM_NAME << ";" << BANANANTT_PROGRAM_VERSION << ";;;"
          << os_name_banana() << ";" << os_arch_banana() << ";" << timestamp;
    const std::string canon_str = canon.str();
    const std::uint32_t crc = bananantt_crc32(canon_str.data(), canon_str.size());
    std::ostringstream checksum;
    checksum << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;
    const std::string json = prefix.str() + ",\"checksum\":{\"version\":1,\"checksum\":\"" + checksum.str() + "\"}}";
    if (g_runtime.json_enabled) {
        const std::string path = g_runtime.json_path.empty() ? default_json_path(p) : g_runtime.json_path;
        ensure_parent_dir_exists(path);
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write JSON output: " + path);
        out << json << "\n";
        std::cout << "JSON written: " << path << "\n";
    }
    if (g_runtime.append_results && (st == "P" || st == "C")) {
        std::string rpath = g_runtime.results_path.empty() ? "./results.txt" : g_runtime.results_path;
        if (!g_runtime.output_dir.empty() && rpath == "./results.txt") rpath = (std::filesystem::path(g_runtime.output_dir) / "results.txt").string();
        ensure_parent_dir_exists(rpath);
        std::ofstream r(rpath, std::ios::app);
        if (!r) throw std::runtime_error("cannot append results file: " + rpath);
        r << json << "\n";
        std::cout << "Result appended: " << rpath << "\n";
    }
}

static void write_bananantt_json_from_digits(std::uint32_t p, const std::string& status,
                                             const std::vector<std::uint64_t>& digits,
                                             const ibdwt::Layout& layout) {
    std::vector<std::uint32_t> words = residue_words32_from_digits(digits, layout);
    prp3_div9_words_banana(p, words);
    const bool residue_one = words_equal_one(words);
    const std::string st = (status == "benchmark-stopped") ? "U" : (residue_one ? "P" : "C");
    const std::string res64_prp = format_res64_words_plain_upper(words);
    const std::string res2048 = format_res2048_words_lower(words);
    const std::string timestamp = current_timestamp_utc_banana();
    const std::uint32_t proof_power = g_runtime.proof_power ? g_runtime.proof_power : best_proof_power_banana(p);
    const bool have_proof = g_runtime.proof_checkpoints && !g_runtime.last_proof_file.empty();
    const std::string proof_md5 = have_proof ? md5_file_hex_banana(g_runtime.last_proof_file) : "";
    std::ostringstream prefix;
    prefix << "{\"status\":" << json_escape_banana(st)
           << ",\"exponent\":" << p
           << ",\"worktype\":\"PRP-3\""
           << ",\"res64\":" << json_escape_banana(res64_prp)
           << ",\"res2048\":" << json_escape_banana(res2048)
           << ",\"residue-type\":1"
           << ",\"errors\":{\"gerbicz\":" << static_cast<unsigned long long>(g_runtime.gerbicz_errors) << "}"
           << ",\"shift-count\":0"
           << ",\"fft-length\":" << layout.n;
    if (have_proof) {
        prefix << ",\"proof\":{\"version\":2,\"power\":" << proof_power
               << ",\"hashsize\":64,\"md5\":" << json_escape_banana(proof_md5) << "}";
    }
    prefix << ",\"program\":{\"name\":" << json_escape_banana(BANANANTT_PROGRAM_NAME)
           << ",\"version\":" << json_escape_banana(BANANANTT_PROGRAM_VERSION)
           << ",\"port\":" << BANANANTT_PROGRAM_PORT << "}"
           << ",\"os\":{\"os\":" << json_escape_banana(os_name_banana())
           << ",\"architecture\":" << json_escape_banana(os_arch_banana()) << "}"
           << ",\"timestamp\":" << json_escape_banana(timestamp);
    std::ostringstream canon;
    canon << p << ";PRP;;;" << lower_hex(res64_prp) << ";" << lower_hex(res2048)
          << ";0_3_1;" << layout.n << ";gerbicz:" << g_runtime.gerbicz_errors << ";"
          << BANANANTT_PROGRAM_NAME << ";" << BANANANTT_PROGRAM_VERSION << ";;;"
          << os_name_banana() << ";" << os_arch_banana() << ";" << timestamp;
    const std::string canon_str = canon.str();
    const std::uint32_t crc = bananantt_crc32(canon_str.data(), canon_str.size());
    std::ostringstream checksum;
    checksum << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc;
    const std::string json = prefix.str() + ",\"checksum\":{\"version\":1,\"checksum\":\"" + checksum.str() + "\"}}";
    if (g_runtime.json_enabled) {
        const std::string path = g_runtime.json_path.empty() ? default_json_path(p) : g_runtime.json_path;
        ensure_parent_dir_exists(path);
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cannot write JSON output: " + path);
        out << json << "\n";
        std::cout << "JSON written: " << path << "\n";
    }
    if (g_runtime.append_results && (st == "P" || st == "C")) {
        std::string rpath = g_runtime.results_path.empty() ? "./results.txt" : g_runtime.results_path;
        if (!g_runtime.output_dir.empty() && rpath == "./results.txt") rpath = (std::filesystem::path(g_runtime.output_dir) / "results.txt").string();
        ensure_parent_dir_exists(rpath);
        std::ofstream r(rpath, std::ios::app);
        if (!r) throw std::runtime_error("cannot append results file: " + rpath);
        r << json << "\n";
        std::cout << "Result appended: " << rpath << "\n";
    }
}



}

struct Options {
    std::uint32_t exponent = 0;
    bool verbose = true;
    bool selftest_only = false;
    int device_index = 0;
    std::string kernel_path = "prmers_opencl_prp.cl";
    cl_uint carry_block = 0;
    cl_uint carry_items = 0;
    cl_uint center_max = 0;
    bool center_max_user = false;
    bool autotune_center = false;
    cl_uint autotune_center_iters = 300;
    bool profile_kernels = false;
    bool prefer_radix4x2 = true;
    bool planner_debug = false;
    bool local_block_lds_disabled = false;
    bool unsafe_local_block_512 = false;
    cl_uint local_block_lds = 0;
    std::uint32_t profile_every = 0;
    std::uint32_t max_iters = 0;
    std::string modulus_mode = "crt";
    std::uint32_t headroom_bits = 2;
    bool crt_async_queues = true;
    bool crt_fused_pipeline = true;
    bool crt_defused_fast = true;
    int crt_defused_schedule = 0;
    int crt_fwd8_61_wg = 64;
    int crt_defused_edge_fuse = 0;
    uint32_t crt_edge_radix = 4u;
    uint32_t crt_odd_radix = 0u;
    bool crt_odd_radix_auto = true;
    std::string crt_mixed_row_core = "auto";
    std::string crt_mixed_row_fuse_both = "off";
    int crt_defused_edge_mode = 0;
    bool user_crt_queue = false;
    bool user_crt_schedule = false;
    bool user_crt_local_stage = false;
    bool user_crt_local_square = false;
    bool user_crt_edge_radix = false;
    bool user_crt_edge_mode = false;
    bool user_crt_edge_fuse = false;
    std::string crt_center_mode = "halfreal";
    bool crt_halfreal_validate = false;
    uint32_t crt_halfreal_validate_iters = 1;
    bool crt_halfreal_validate_random = false;
    bool crt_mixed_gpu_reference = false;
    std::size_t crt_halfreal_dump_count = 32;
    std::string crt_halfreal_dump_prefix = "halfreal_debug";
    int crt_halfreal_flags61 = 48;
    int crt_halfreal_flags31 = 48;
    bool crt_halfreal_autoprobe = false;
    bool crt_halfreal_probe_exhaustive = true;
    bool crt_startup_autotune = false;
    int crt_autotune_iters = 1000;
    bool crt_autotune_wide = false;
    bool crt_split_center = false;
    bool crt_fused_center_lockstep = false;
    bool crt_radix8_global = true;
    cl_uint crt_center_chunk = 512u;
    cl_uint crt_lds_stage = 0u;
    cl_uint crt_lds_tile = 2u;
    cl_uint crt_head_radix8 = 0u;
    std::string single_center_mode = "normal";

    std::string config_path;
    std::string worktodo_path = "worktodo.txt";
    bool no_worktodo = false;
    std::uint32_t res64_every = 0;
    bool json_enabled = true;
    std::string json_path;
    std::string output_dir;
    std::string results_path = "./results.txt";
    bool append_results = true;
    bool proof_checkpoints = false;
    std::uint32_t proof_power = 0;
    std::string proof_dir;
    std::uint32_t gerbicz_interval = 0;
    bool gerbicz_enabled = true;
    bool gerbicz_gpu_verify = true;
    bool gerbicz_user_checklevel = false;
    std::uint32_t gerbicz_checklevel = 0;
    std::uint32_t gerbicz_block = 0;
    double gerbicz_target_seconds = BANANANTT_DEFAULT_GERBICZ_TARGET_SECONDS;
    double gerbicz_estimated_ips = 0.0;
    bool gerbicz_user_seconds = false;
    double gerbicz_boundary_seconds = BANANANTT_DEFAULT_GERBICZ_BOUNDARY_SECONDS;
    bool gerbicz_verbose = false;
    bool gerbicz_progress = false;
    std::uint32_t error_iter = 0;
    std::uint32_t error_limb = 0;
    std::uint64_t error_delta = 1;
    bool backup_enabled = true;
    bool resume_enabled = true;
    bool save_on_interrupt = true;
    std::string backup_path;
    std::string resume_path;
    std::string backup_dir = "save";
    std::uint32_t backup_every_iters = 0;
    double backup_every_seconds = 300.0;
    std::uint32_t queue_guard_depth = 0;
    bool queue_guard_auto = true;
    double queue_guard_seconds = BANANANTT_DEFAULT_QUEUE_GUARD_SECONDS;
};


static std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool parse_bool_text(const std::string& v, bool defv=false) {
    const std::string x = lower_copy(trim_copy(v));
    if (x == "1" || x == "true" || x == "yes" || x == "on") return true;
    if (x == "0" || x == "false" || x == "no" || x == "off") return false;
    return defv;
}

static void apply_config_key_value(Options& opt, std::string key, std::string val) {
    key = lower_copy(trim_copy(key));
    val = trim_copy(val);
    if (key.empty()) return;
    if (key == "p" || key == "exponent") opt.exponent = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "device") opt.device_index = std::stoi(val);
    else if (key == "kernel") opt.kernel_path = val;
    else if (key == "modulus" || key == "field") opt.modulus_mode = lower_copy(val);
    else if (key == "iters" || key == "max_iters" || key == "benchmark_iters") opt.max_iters = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "quiet") opt.verbose = !parse_bool_text(val, false);
    else if (key == "profile_kernels" || key == "profile") opt.profile_kernels = parse_bool_text(val, false);
    else if (key == "profile_every") opt.profile_every = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "crt_odd_radix") opt.crt_odd_radix = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "crt_mixed_row_core") opt.crt_mixed_row_core = val;
    else if (key == "crt_mixed_row_stage" || key == "crt_lds_stage") { opt.crt_lds_stage = static_cast<cl_uint>(std::stoul(val)); opt.user_crt_local_stage = true; }
    else if (key == "crt_mixed_row_center" || key == "crt_local_square") { opt.crt_center_chunk = static_cast<cl_uint>(std::stoul(val)); opt.user_crt_local_square = true; }
    else if (key == "crt_mixed_row_fuse_both") opt.crt_mixed_row_fuse_both = val;
    else if (key == "res64_every" || key == "res64_interval") opt.res64_every = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "json") opt.json_enabled = parse_bool_text(val, true);
    else if (key == "no_json") opt.json_enabled = !parse_bool_text(val, true);
    else if (key == "json_path" || key == "json_file" || key == "json_out") { opt.json_enabled = true; opt.json_path = val; }
    else if (key == "output_dir" || key == "save_path") opt.output_dir = val;
    else if (key == "results_path" || key == "results_file") opt.results_path = val;
    else if (key == "append_results") opt.append_results = parse_bool_text(val, true);
    else if (key == "no_results") opt.append_results = !parse_bool_text(val, true);
    else if (key == "proof" || key == "proof_checkpoints") opt.proof_checkpoints = parse_bool_text(val, true);
    else if (key == "proof_power") opt.proof_power = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "proof_dir") { opt.proof_checkpoints = true; opt.proof_dir = val; }
    else if (key == "gerbicz" || key == "gerbiczli") opt.gerbicz_enabled = parse_bool_text(val, true);
    else if (key == "gerbicz_gpu" || key == "gerbicz_gpu_verify") opt.gerbicz_gpu_verify = parse_bool_text(val, true);
    else if (key == "gerbicz_backend") { auto b = lower_copy(val); opt.gerbicz_gpu_verify = (b != "host" && b != "cpu" && b != "gmp"); }
    else if (key == "gerbicz_interval" || key == "gerbicz_checklevel" || key == "checklevel") { opt.gerbicz_enabled = true; opt.gerbicz_user_checklevel = true; opt.gerbicz_checklevel = static_cast<std::uint32_t>(std::stoul(val)); }
    else if (key == "gerbicz_b" || key == "gerbicz_block") { opt.gerbicz_enabled = true; opt.gerbicz_block = static_cast<std::uint32_t>(std::stoul(val)); }
    else if (key == "gerbicz_target_seconds" || key == "gerbicz_seconds") { opt.gerbicz_enabled = true; opt.gerbicz_user_seconds = true; opt.gerbicz_target_seconds = std::stod(val); }
    else if (key == "gerbicz_estimate_it_s" || key == "gerbicz_estimated_ips") { opt.gerbicz_enabled = true; opt.gerbicz_estimated_ips = std::stod(val); }
    else if (key == "gerbicz_boundary_seconds") { opt.gerbicz_enabled = true; opt.gerbicz_boundary_seconds = std::stod(val); }
    else if (key == "gerbicz_verbose") opt.gerbicz_verbose = parse_bool_text(val, true);
    else if (key == "gerbicz_progress") opt.gerbicz_progress = parse_bool_text(val, true);
    else if (key == "erroriter" || key == "error_iter") opt.error_iter = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "error_limb") opt.error_limb = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "error_delta") opt.error_delta = static_cast<std::uint64_t>(std::stoull(val));
    else if (key == "backup" || key == "save" || key == "checkpoint") opt.backup_enabled = parse_bool_text(val, true);
    else if (key == "resume") opt.resume_enabled = parse_bool_text(val, true);
    else if (key == "save_on_interrupt") opt.save_on_interrupt = parse_bool_text(val, true);
    else if (key == "backup_file" || key == "save_file" || key == "checkpoint_file") opt.backup_path = val;
    else if (key == "resume_file") { opt.resume_enabled = true; opt.resume_path = val; }
    else if (key == "backup_dir" || key == "save_dir") opt.backup_dir = val;
    else if (key == "backup_every" || key == "backup_every_iters" || key == "save_every") opt.backup_every_iters = static_cast<std::uint32_t>(std::stoul(val));
    else if (key == "backup_seconds" || key == "backup_every_seconds" || key == "save_seconds") opt.backup_every_seconds = std::stod(val);
    else if (key == "queue_guard" || key == "queue_guard_depth" || key == "cl_queue_guard") { auto qv = lower_copy(val); if (qv == "auto") { opt.queue_guard_auto = true; opt.queue_guard_depth = 0; } else { opt.queue_guard_auto = false; opt.queue_guard_depth = static_cast<std::uint32_t>(std::stoul(val)); } }
    else if (key == "queue_guard_seconds" || key == "cl_queue_guard_seconds") { opt.queue_guard_auto = true; opt.queue_guard_seconds = std::stod(val); }
    else if (key == "worktodo") opt.worktodo_path = val;
    else if (key == "no_worktodo") opt.no_worktodo = parse_bool_text(val, true);
}

static void apply_config_file(Options& opt, const std::string& path) {
    if (path.empty()) return;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config file: " + path);
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t hash = line.find_first_of("#;");
        if (hash != std::string::npos) line.resize(hash);
        line = trim_copy(line);
        if (line.empty()) continue;
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) eq = line.find(':');
        if (eq == std::string::npos) continue;
        apply_config_key_value(opt, line.substr(0, eq), line.substr(eq + 1));
    }
}

static bool looks_like_prp_worktodo_line(const std::string& line) {
    const std::string l = lower_copy(line);
    return l.find("prp") != std::string::npos && l.find("ecm") == std::string::npos && l.find("pminus1") == std::string::npos;
}

static std::uint32_t parse_first_prp_from_worktodo(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    std::string line;
    while (std::getline(f, line)) {
        std::string no_comment = line;
        const std::size_t hash = no_comment.find_first_of("#;");
        if (hash != std::string::npos) no_comment.resize(hash);
        if (!looks_like_prp_worktodo_line(no_comment)) continue;
        std::vector<std::uint64_t> nums;
        std::uint64_t cur = 0; bool in = false;
        for (char c : no_comment) {
            if (std::isdigit(static_cast<unsigned char>(c))) { in = true; cur = cur * 10u + static_cast<unsigned>(c - '0'); }
            else { if (in) { nums.push_back(cur); cur = 0; in = false; } }
        }
        if (in) nums.push_back(cur);
        // Worktodo PRP lines can contain small flags before p.  The exponent is normally the largest useful number.
        std::uint64_t best = 0;
        for (std::uint64_t v : nums) if (v > best && v <= 0xffffffffULL) best = v;
        if (best >= 3) return static_cast<std::uint32_t>(best);
    }
    return 0;
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-config" || a == "--cfg" || a == "-cfg") && i + 1 < argc) {
            opt.config_path = argv[++i];
            apply_config_file(opt, opt.config_path);
        }
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--modulus" || arg == "--field") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.modulus_mode = argv[++i];
            if (opt.modulus_mode == "m61" || opt.modulus_mode == "gf61" || opt.modulus_mode == "GF61") opt.modulus_mode = "gf61";
            else if (opt.modulus_mode == "m31" || opt.modulus_mode == "gf31" || opt.modulus_mode == "GF31") opt.modulus_mode = "gf31";
            else if (opt.modulus_mode == "crt" || opt.modulus_mode == "gf61xgf31" || opt.modulus_mode == "GF61xGF31") opt.modulus_mode = "crt";
            else if (opt.modulus_mode == "crt-cpu" || opt.modulus_mode == "cpu-crt") opt.modulus_mode = "crt-cpu";
            else if (opt.modulus_mode == "best" || opt.modulus_mode == "auto") opt.modulus_mode = "best";
            else throw std::runtime_error("--modulus must be gf61, gf31, crt, crt-cpu or best");
        } else if (arg == "--quiet") {
            opt.verbose = false;
        } else if (arg == "--selftest") {
            opt.selftest_only = true;
        } else if (arg == "--device") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --device");
            opt.device_index = std::stoi(argv[++i]);
        } else if (arg == "--kernel") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --kernel");
            opt.kernel_path = argv[++i];
        } else if (arg == "--config" || arg == "-config" || arg == "--cfg" || arg == "-cfg") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.config_path = argv[++i];
        } else if (arg == "--worktodo" || arg == "-worktodo") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.worktodo_path = argv[++i];
        } else if (arg == "--no-worktodo") {
            opt.no_worktodo = true;
        } else if (arg == "--res64-every" || arg == "--res64-display" || arg == "--res64-interval") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.res64_every = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--json") {
            opt.json_enabled = true;
        } else if (arg == "--no-json") {
            opt.json_enabled = false;
        } else if (arg == "--json-file" || arg == "--json-out") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.json_enabled = true;
            opt.json_path = argv[++i];
        } else if (arg == "--output-dir" || arg == "--save-path") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.output_dir = argv[++i];
        } else if (arg == "--results-file" || arg == "--results-path") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.results_path = argv[++i];
        } else if (arg == "--no-results") {
            opt.append_results = false;
        } else if (arg == "--proof" || arg == "--prp-proof" || arg == "--proof-checkpoints") {
            opt.proof_checkpoints = true;
        } else if (arg == "--proof-power") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --proof-power");
            opt.proof_checkpoints = true;
            opt.proof_power = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--proof-dir") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --proof-dir");
            opt.proof_checkpoints = true;
            opt.proof_dir = argv[++i];
        } else if (arg == "--gerbicz" || arg == "--gerbicz-li" || arg == "--gerbiczli") {
            opt.gerbicz_enabled = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opt.gerbicz_user_checklevel = true;
                opt.gerbicz_checklevel = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            }
        } else if (arg == "--no-gerbicz" || arg == "-gerbiczli") {
            opt.gerbicz_enabled = false;
        } else if (arg == "--gerbicz-checklevel" || arg == "-checklevel") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.gerbicz_enabled = true;
            opt.gerbicz_user_checklevel = true;
            opt.gerbicz_checklevel = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--gerbicz-seconds" || arg == "--gerbicz-target-seconds") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.gerbicz_enabled = true;
            opt.gerbicz_user_seconds = true;
            opt.gerbicz_target_seconds = std::stod(argv[++i]);
        } else if (arg == "--gerbicz-backend") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            std::string b = lower_copy(argv[++i]);
            opt.gerbicz_enabled = true;
            opt.gerbicz_gpu_verify = (b != "host" && b != "cpu" && b != "gmp");
        } else if (arg == "--gerbicz-gpu") {
            opt.gerbicz_enabled = true;
            opt.gerbicz_gpu_verify = true;
        } else if (arg == "--gerbicz-host") {
            opt.gerbicz_enabled = true;
            opt.gerbicz_gpu_verify = false;
        } else if (arg == "--gerbicz-estimate-it-s" || arg == "--gerbicz-estimated-ips") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.gerbicz_enabled = true;
            opt.gerbicz_estimated_ips = std::stod(argv[++i]);
        } else if (arg == "--gerbicz-boundary-seconds") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.gerbicz_enabled = true;
            opt.gerbicz_boundary_seconds = std::stod(argv[++i]);
        } else if (arg == "--gerbicz-verbose") {
            opt.gerbicz_verbose = true;
        } else if (arg == "--gerbicz-progress") {
            opt.gerbicz_progress = true;
        } else if (arg == "--gerbicz-b" || arg == "--gerbicz-block") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.gerbicz_enabled = true;
            opt.gerbicz_block = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--error-iter" || arg == "--erroriter" || arg == "-erroriter") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.error_iter = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--error-limb") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.error_limb = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--error-delta") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.error_delta = static_cast<std::uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--no-backup" || arg == "--no-save" || arg == "--no-checkpoint") {
            opt.backup_enabled = false;
        } else if (arg == "--backup" || arg == "--save" || arg == "--checkpoint") {
            opt.backup_enabled = true;
        } else if (arg == "--no-resume") {
            opt.resume_enabled = false;
        } else if (arg == "--resume") {
            opt.resume_enabled = true;
        } else if (arg == "--save-file" || arg == "--backup-file" || arg == "--checkpoint-file") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.backup_enabled = true;
            opt.backup_path = argv[++i];
        } else if (arg == "--resume-file") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --resume-file");
            opt.resume_enabled = true;
            opt.resume_path = argv[++i];
        } else if (arg == "--backup-dir" || arg == "--save-dir") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.backup_dir = argv[++i];
        } else if (arg == "--backup-every" || arg == "--save-every" || arg == "--backup-every-iters") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.backup_every_iters = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--backup-seconds" || arg == "--save-seconds" || arg == "--backup-every-seconds") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.backup_every_seconds = std::stod(argv[++i]);
        } else if (arg == "--no-save-on-interrupt") {
            opt.save_on_interrupt = false;
        } else if (arg == "--queue-guard" || arg == "--cl-queue-guard") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            std::string qv = lower_copy(argv[++i]);
            if (qv == "auto") {
                opt.queue_guard_auto = true;
                opt.queue_guard_depth = 0;
            } else {
                opt.queue_guard_auto = false;
                opt.queue_guard_depth = static_cast<std::uint32_t>(std::stoul(qv));
            }
        } else if (arg == "--queue-guard-seconds" || arg == "--cl-queue-guard-seconds") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.queue_guard_auto = true;
            opt.queue_guard_seconds = std::stod(argv[++i]);
        } else if (arg == "--carry-block") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --carry-block");
            opt.carry_block = static_cast<cl_uint>(std::stoul(argv[++i]));
        } else if (arg == "--carry-items") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --carry-items");
            opt.carry_items = static_cast<cl_uint>(std::stoul(argv[++i]));
        } else if (arg == "--center-max") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --center-max");
            opt.center_max = static_cast<cl_uint>(std::stoul(argv[++i]));
            opt.center_max_user = true;
        } else if (arg == "--center-autotune") {
            opt.autotune_center = true;
        } else if (arg == "--no-center-autotune") {
            opt.autotune_center = false;
        } else if (arg == "--center-autotune-iters") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --center-autotune-iters");
            opt.autotune_center_iters = static_cast<cl_uint>(std::stoul(argv[++i]));
            if (opt.autotune_center_iters == 0u) opt.autotune_center_iters = 1u;
        } else if (arg == "--crt-shared-queue" || arg == "--crt-single-queue" || arg == "--crt-defused-single-queue") {
            
            
            opt.crt_async_queues = false;
            opt.user_crt_queue = true;
        } else if (arg == "--crt-async-queues" || arg == "--crt-two-queues") {
            opt.crt_async_queues = true;
            opt.user_crt_queue = true;
        } else if (arg == "--no-crt-fused" || arg == "--crt-reference") {
            opt.crt_fused_pipeline = false;
        } else if (arg == "--crt-defused-ntt" || arg == "--crt-defused-fast") {
            opt.crt_defused_fast = true;
        } else if (arg == "--crt-no-defused-ntt" || arg == "--crt-fused-ntt") {
            opt.crt_defused_fast = false;
        } else if (arg == "--crt-defused-schedule") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-defused-schedule");
            std::string v = argv[++i];
            if (v == "grouped" || v == "concurrent") opt.crt_defused_schedule = 0;
            else if (v == "interleave" || v == "interleaved") opt.crt_defused_schedule = 1;
            else if (v == "gf31-first") opt.crt_defused_schedule = 2;
            else if (v == "gf61-dominant" || v == "gf61-main" || v == "gf61-first") opt.crt_defused_schedule = 3;
            else if (v == "serial-gf31-first") opt.crt_defused_schedule = 4;
            else if (v == "gf61-forward-first" || v == "gf61-fwd-first") opt.crt_defused_schedule = 5;
            else if (v == "hybrid-fwd8" || v == "hybrid" || v == "gf61-top-fwd8") opt.crt_defused_schedule = 6;
            else throw std::runtime_error("--crt-defused-schedule must be grouped, interleave, gf31-first, gf61-dominant, serial-gf31-first, gf61-forward-first, or hybrid-fwd8");
            opt.user_crt_schedule = true;
        } else if (arg == "--crt-fwd8-61-wg") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-fwd8-61-wg");
            opt.crt_fwd8_61_wg = std::stoi(argv[++i]);
            if (opt.crt_fwd8_61_wg != 64 && opt.crt_fwd8_61_wg != 128) throw std::runtime_error("--crt-fwd8-61-wg must be 64 or 128");
        } else if (arg == "--crt-defused-edge-fuse") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-defused-edge-fuse");
            std::string v = argv[++i];
            if (v == "off" || v == "0" || v == "none") opt.crt_defused_edge_fuse = 0;
            else if (v == "forward" || v == "fwd") opt.crt_defused_edge_fuse = 1;
            else if (v == "tail" || v == "last") opt.crt_defused_edge_fuse = 2;
            else if (v == "both" || v == "on" || v == "1") opt.crt_defused_edge_fuse = 3;
            else throw std::runtime_error("--crt-defused-edge-fuse must be off, forward, tail, or both");
            opt.user_crt_edge_fuse = true;
        } else if (arg == "--crt-edge-radix" || arg == "--crt-defused-edge-radix") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-edge-radix");
            opt.crt_edge_radix = static_cast<uint32_t>(std::stoul(argv[++i]));
            opt.user_crt_edge_radix = true;
            if (!(opt.crt_edge_radix == 2u || opt.crt_edge_radix == 4u || opt.crt_edge_radix == 8u || opt.crt_edge_radix == 16u)) {
                throw std::runtime_error("--crt-edge-radix must be 2, 4, 8, or 16");
            }
        } else if (arg == "--crt-odd-radix" || arg == "--crt-pfa-odd" || arg == "--crt-mixed-odd") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-odd-radix");
            const std::string v = argv[++i];
            if (v == "auto" || v == "default") {
                opt.crt_odd_radix = 0u;
                opt.crt_odd_radix_auto = true;
            } else {
                opt.crt_odd_radix_auto = false;
                if (v == "off" || v == "none" || v == "1") opt.crt_odd_radix = 1u;
                else opt.crt_odd_radix = static_cast<uint32_t>(std::stoul(v));
                if (!(opt.crt_odd_radix == 1u || opt.crt_odd_radix == 3u || opt.crt_odd_radix == 9u)) {
                    throw std::runtime_error("--crt-odd-radix must be auto, 1/off, 3 or 9 in this GPU prototype");
                }
            }
            if (opt.crt_odd_radix > 1u) {
                opt.crt_center_mode = "halfreal";
                opt.crt_defused_fast = true;
            }
        } else if (arg == "--crt-mixed-row-core" || arg == "--crt-odd-row-core") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.crt_mixed_row_core = argv[++i];
            if (opt.crt_mixed_row_core == "radix2" || opt.crt_mixed_row_core == "generic-radix2" || opt.crt_mixed_row_core == "slow") opt.crt_mixed_row_core = "generic";
            if (opt.crt_mixed_row_core == "fast") opt.crt_mixed_row_core = "lds";
            if (!(opt.crt_mixed_row_core == "auto" || opt.crt_mixed_row_core == "lds" || opt.crt_mixed_row_core == "lds512" || opt.crt_mixed_row_core == "lds1024" || opt.crt_mixed_row_core == "generic")) {
                throw std::runtime_error("--crt-mixed-row-core must be auto, lds, lds512, lds1024 or generic");
            }
        } else if (arg == "--crt-mixed-row-generic" || arg == "--crt-odd-row-generic") {
            opt.crt_mixed_row_core = "generic";
        } else if (arg == "--crt-mixed-row-lds" || arg == "--crt-odd-row-lds") {
            opt.crt_mixed_row_core = "lds";
        } else if (arg == "--crt-mixed-row-lds512" || arg == "--crt-odd-row-lds512") {
            opt.crt_mixed_row_core = "lds512";
        } else if (arg == "--crt-mixed-row-lds1024" || arg == "--crt-odd-row-lds1024") {
            opt.crt_mixed_row_core = "lds1024";
        } else if (arg == "--crt-mixed-row-fuse-both" || arg == "--crt-odd-row-fuse-both") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.crt_mixed_row_fuse_both = argv[++i];
            if (opt.crt_mixed_row_fuse_both == "0" || opt.crt_mixed_row_fuse_both == "none") opt.crt_mixed_row_fuse_both = "off";
            if (opt.crt_mixed_row_fuse_both == "1" || opt.crt_mixed_row_fuse_both == "on" || opt.crt_mixed_row_fuse_both == "yes") opt.crt_mixed_row_fuse_both = "all";
            if (!(opt.crt_mixed_row_fuse_both == "off" || opt.crt_mixed_row_fuse_both == "auto" ||
                  opt.crt_mixed_row_fuse_both == "center" || opt.crt_mixed_row_fuse_both == "stage" ||
                  opt.crt_mixed_row_fuse_both == "all" || opt.crt_mixed_row_fuse_both == "force")) {
                throw std::runtime_error("--crt-mixed-row-fuse-both must be off, auto, center, stage, all, or force");
            }
        } else if (arg == "--crt-mixed-row-fuse-both-center") {
            opt.crt_mixed_row_fuse_both = "center";
        } else if (arg == "--crt-mixed-row-fuse-both-stage") {
            opt.crt_mixed_row_fuse_both = "stage";
        } else if (arg == "--crt-mixed-row-fuse-both-all") {
            opt.crt_mixed_row_fuse_both = "all";
        } else if (arg == "--crt-mixed-row-fuse-both-force") {
            opt.crt_mixed_row_fuse_both = "force";
        } else if (arg == "--crt-edge-mode" || arg == "--crt-defused-edge-mode") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-edge-mode");
            std::string v = argv[++i];
            if (v == "auto" || v == "default") opt.crt_defused_edge_mode = 0;
            else if (v == "legacy" || v == "split" || v == "specialized") opt.crt_defused_edge_mode = 1;
            else if (v == "generic" || v == "on") opt.crt_defused_edge_mode = 2;
            else throw std::runtime_error("--crt-edge-mode must be auto, legacy/split, or generic");
            opt.user_crt_edge_mode = true;
        } else if (arg == "--crt-center-mode") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-center-mode");
            opt.crt_center_mode = argv[++i];
            if (!(opt.crt_center_mode == "normal" || opt.crt_center_mode == "classic" || opt.crt_center_mode == "halfreal")) {
                throw std::runtime_error("--crt-center-mode must be normal/classic or halfreal");
            }
        } else if (arg == "--single-center-mode" || arg == "--field-center-mode") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --single-center-mode");
            opt.single_center_mode = argv[++i];
            if (opt.single_center_mode == "classic") opt.single_center_mode = "normal";
            if (!(opt.single_center_mode == "normal" || opt.single_center_mode == "halfreal" || opt.single_center_mode == "auto")) {
                throw std::runtime_error("--single-center-mode must be normal/classic, halfreal or auto");
            }
        } else if (arg == "--single-halfreal" || arg == "--field-halfreal") {
            opt.single_center_mode = "halfreal";
        } else if (arg == "--single-normal" || arg == "--field-normal") {
            opt.single_center_mode = "normal";
        } else if (arg == "--crt-halfreal-validate" || arg == "--crt-halfreal-debug") {
            opt.crt_halfreal_validate = true;
        } else if (arg == "--crt-halfreal-validate-iters") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-validate-iters");
            opt.crt_halfreal_validate = true;
            opt.crt_halfreal_validate_iters = std::max(1u, (unsigned)std::stoul(argv[++i]));
        } else if (arg == "--crt-halfreal-validate-random") {
            opt.crt_halfreal_validate = true;
            opt.crt_halfreal_validate_random = true;
        } else if (arg == "--crt-mixed-gpu-reference" || arg == "--crt-mixed-gpu-validate" || arg == "--crt-halfreal-gpu-reference") {
            opt.crt_halfreal_validate = true;
            opt.crt_mixed_gpu_reference = true;
        } else if (arg == "--crt-halfreal-dump") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-dump");
            opt.crt_halfreal_validate = true;
            opt.crt_halfreal_dump_prefix = argv[++i];
        } else if (arg == "--crt-halfreal-dump-prefix") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-dump-prefix");
            opt.crt_halfreal_validate = true;
            opt.crt_halfreal_dump_prefix = argv[++i];
        } else if (arg == "--crt-halfreal-dump-count" || arg == "--crt-halfreal-dump-n") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-dump-count");
            opt.crt_halfreal_validate = true;
            opt.crt_halfreal_dump_count = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--crt-halfreal-probe") {
            opt.crt_halfreal_autoprobe = true;
            opt.crt_halfreal_probe_exhaustive = true;
            opt.crt_halfreal_flags61 = -1;
            opt.crt_halfreal_flags31 = -1;
        } else if (arg == "--crt-halfreal-probe-fast") {
            opt.crt_halfreal_autoprobe = true;
            opt.crt_halfreal_probe_exhaustive = false;
            opt.crt_halfreal_flags61 = -1;
            opt.crt_halfreal_flags31 = -1;
        } else if (arg == "--crt-halfreal-no-autoprobe") {
            opt.crt_halfreal_autoprobe = false;
        } else if (arg == "--crt-halfreal-flags") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-flags");
            const int f = std::stoi(argv[++i]);
            if (f < 0 || f > 63) throw std::runtime_error("--crt-halfreal-flags must be in 0..63");
            opt.crt_halfreal_flags61 = f;
            opt.crt_halfreal_flags31 = f;
            opt.crt_halfreal_autoprobe = false;
        } else if (arg == "--crt-halfreal-flags61") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-flags61");
            const int f = std::stoi(argv[++i]);
            if (f < 0 || f > 63) throw std::runtime_error("--crt-halfreal-flags61 must be in 0..63");
            opt.crt_halfreal_flags61 = f;
        } else if (arg == "--crt-halfreal-flags31") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for --crt-halfreal-flags31");
            const int f = std::stoi(argv[++i]);
            if (f < 0 || f > 63) throw std::runtime_error("--crt-halfreal-flags31 must be in 0..63");
            opt.crt_halfreal_flags31 = f;
        } else if (arg == "--crt-edge-generic" || arg == "--crt-defused-edge-generic") {
            opt.crt_defused_edge_mode = 2;
            opt.user_crt_edge_mode = true;
        } else if (arg == "--crt-edge-legacy" || arg == "--crt-edge-split") {
            opt.crt_defused_edge_mode = 1;
            opt.user_crt_edge_mode = true;
        } else if (arg == "--crt-split-center" || arg == "--crt-no-fused-center") {
            opt.crt_split_center = true;
        } else if (arg == "--crt-fused-center-experimental") {
            opt.crt_split_center = false;
            opt.crt_fused_center_lockstep = false;
        } else if (arg == "--crt-fused-center-lockstep") {
            opt.crt_split_center = false;
            opt.crt_fused_center_lockstep = true;
        } else if (arg == "--crt-radix8" || arg == "--crt-radix8-global") {
            opt.crt_radix8_global = true;
        } else if (arg == "--crt-radix4" || arg == "--crt-no-radix8") {
            opt.crt_radix8_global = false;
        } else if (arg == "--crt-center" || arg == "--crt-local-square" || arg == "--crt-lds-square" || arg == "--crt-mixed-row-center" || arg == "--crt-mixed-row-square") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.crt_center_chunk = static_cast<cl_uint>(std::stoul(argv[++i]));
            opt.user_crt_local_square = true;
            if (opt.crt_center_chunk != 8u && opt.crt_center_chunk != 16u && opt.crt_center_chunk != 32u && opt.crt_center_chunk != 64u && opt.crt_center_chunk != 128u && opt.crt_center_chunk != 256u && opt.crt_center_chunk != 512u && opt.crt_center_chunk != 1024u) {
                throw std::runtime_error("--crt-local-square must be 8, 16, 32, 64, 128, 256, 512 or 1024");
            }
        } else if (arg == "--crt-lds-stage" || arg == "--crt-local-stage-max" || arg == "--crt-mixed-row-stage") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --crt-lds-stage");
            opt.crt_lds_stage = static_cast<cl_uint>(std::stoul(argv[++i]));
            opt.user_crt_local_stage = true;
            if (opt.crt_lds_stage != 0u && opt.crt_lds_stage != 8u && opt.crt_lds_stage != 16u && opt.crt_lds_stage != 32u && opt.crt_lds_stage != 64u &&
                opt.crt_lds_stage != 128u && opt.crt_lds_stage != 256u && opt.crt_lds_stage != 512u && opt.crt_lds_stage != 1024u) {
                throw std::runtime_error("--crt-local-stage-max must be 0, 8, 16, 32, 64, 128, 256, 512 or 1024");
            }
        } else if (arg == "--crt-lds-tile" || arg == "--crt-local-stage-tile") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.crt_lds_tile = static_cast<cl_uint>(std::stoul(argv[++i]));
            if (opt.crt_lds_tile != 1u && opt.crt_lds_tile != 2u) {
                throw std::runtime_error("--crt-lds-tile must be 1 or 2");
            }
        } else if (arg == "--crt-head-radix8") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --crt-head-radix8");
            opt.crt_head_radix8 = static_cast<cl_uint>(std::stoul(argv[++i]));
            if (opt.crt_head_radix8 > 1u) throw std::runtime_error("--crt-head-radix8 must be 0 or 1");
        } else if (arg == "--crt-startup-autotune" || arg == "--crt-autotune-startup") {
            opt.crt_startup_autotune = true;
        } else if (arg == "--crt-no-startup-autotune") {
            opt.crt_startup_autotune = false;
        } else if (arg == "--crt-autotune-iters") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --crt-autotune-iters");
            opt.crt_autotune_iters = std::max(1000, std::stoi(argv[++i]));
        } else if (arg == "--crt-autotune-wide") {
            opt.crt_autotune_wide = true;
        } else if (arg == "--profile-kernels") {
            opt.profile_kernels = true;
        } else if (arg == "--prefer-radix4") {
            opt.prefer_radix4x2 = false;
        } else if (arg == "--prefer-radix4x2") {
            opt.prefer_radix4x2 = true;
        } else if (arg == "--planner-debug") {
            opt.planner_debug = true;
        } else if (arg == "--no-local-block-lds") {
            opt.local_block_lds_disabled = true;
        } else if (arg == "--unsafe-local-block-lds-512") {
            opt.unsafe_local_block_512 = true;
            opt.local_block_lds = 512u;
        } else if (arg == "--local-block-lds") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --local-block-lds");
            const std::string v = argv[++i];
            if (v == "auto" || v == "0") {
                opt.local_block_lds = 0;
            } else {
                opt.local_block_lds = static_cast<cl_uint>(std::stoul(v));
                if (opt.local_block_lds != 512u && opt.local_block_lds != 1024u && opt.local_block_lds != 2048u) {
                    throw std::runtime_error("--local-block-lds must be auto, 512, 1024 or 2048");
                }
            }
        } else if (arg == "--profile-every") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --profile-every");
            opt.profile_every = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--iters" || arg == "--max-iters" || arg == "--benchmark-iters") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            opt.max_iters = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            if (opt.max_iters == 0u) throw std::runtime_error(arg + " must be > 0");
        } else if (arg == "--headroom-bits") {
            if (i + 1 >= argc) throw std::runtime_error("missing value after --headroom-bits");
            opt.headroom_bits = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            if (opt.headroom_bits > 8u) throw std::runtime_error("--headroom-bits must be between 0 and 8");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  ./prmers_opencl_prp <exponent_p> [--modulus gf61|gf31|crt|best] [--device k] [--quiet]\n"
                << "  ./prmers_opencl_prp --selftest [--modulus gf61|gf31] [--device k]\n\n"
                << "Modes:\n"
                << "  gf61 : GF(M61^2), safest single-field path.\n"
                << "  gf31 : GF(M31^2), only when coefficient recovery fits in 31 bits.\n"
                << "  crt  : GF(M61^2) x GF(M31^2), GPU Garner carry, smaller transform when possible.\n"
                << "  best : selects gf61 or crt from the transform sizes.\n\n"
                << "Defaults: CRT defused NTT + fused center are enabled; known AMD/NVIDIA presets are applied unless overridden.\n"
                << "Autotune: --crt-startup-autotune [--crt-autotune-iters N] tests CRT plans at startup; command-line knobs stay priority.\n"
                << "Tuning: --center-max N, --center-autotune, --single-center-mode normal|halfreal|auto, --local-block-lds 512|1024|2048, --no-local-block-lds.\n"
                << "CRT:    --crt-async-queues/--crt-two-queues default, --crt-shared-queue or --crt-single-queue for debug/contention tests.\n"
                << "        --crt-radix8 default, --crt-radix4 disables global radix-8 test path.\n"
                << "        --crt-local-square/--crt-mixed-row-center 8|16|32|64|128|256|512|1024 selects the requested center block for mixed odd LDS tests.\n"
                << "        --crt-defused-ntt is now default; --crt-no-defused-ntt/--crt-fused-ntt reverts to fused NTT.\n"
                << "        --crt-edge-radix 2|4|8|16 changes first weighted and last unweighted edge radix; default 4.\n"
                << "        --crt-odd-radix auto|1|3|9 enables CRT/PFA odd x 2^m half-real rows; 9 is the tuned mixed path.\n"
                << "        odd=9 uses the 2D mixed CRT half-real scheme and a DFT3x3 odd transform.\n"
                << "        --crt-edge-mode auto|legacy|generic selects old radix4 split/fuse path or generic edge kernels.\n"
                << "        --crt-center is kept as an alias; --crt-split-center keeps the older split GF61/GF31 center.\n"
                << "        --crt-lds-stage/--crt-mixed-row-stage 8|16|32|64|128|256|512|1024 forces the LDS stage before the row center for tests.\n"
                << "        --crt-mixed-row-core auto|lds|lds512|lds1024|generic selects the odd radix row path; lds uses --crt-mixed-row-center.\n"
                << "        --crt-mixed-row-fuse-both off|auto|center|stage|all|force optionally runs GF61+GF31 in the same LDS center/stage kernels.\n"
                << "        --crt-mixed-gpu-reference validates mixed odd LDS/fused paths against the generic mixed GPU path instead of exact CPU.\n"
                << "        --crt-startup-autotune [--crt-autotune-iters N] tests CRT plans at startup; --crt-autotune-wide adds more candidates.\n"
                << "I/O:    --config FILE reads simple key=value options before CLI overrides.\n"
                << "        --worktodo FILE loads the first PRP assignment if no exponent is given; --no-worktodo disables this.\n"
                << "        --json or --json-file FILE writes a small run/result JSON file.\n"
                << "        --res64-every N prints res64 every N iterations; this intentionally reads back from the GPU.\n"
                << "        --proof/--proof-dir DIR [--proof-power k] writes experimental residue checkpoints every 2^k iterations.\n"
                << "        --gerbicz/--gerbicz-li [N] enables true Gerbicz-Li; N is checklevel in block boundaries.\n"
                << "        --gerbicz-backend gpu|host selects GPU full-check or GMP host full-check; default gpu.\n"
                << "        --gerbicz-seconds S targets a full Li check about every S seconds; default 600.\n"
                << "        --gerbicz-boundary-seconds S targets quiet D updates about every S seconds when B is automatic; default 2.\n"
                << "        --gerbicz-verbose shows every D update; --gerbicz-progress shows full-check leg progress.\n"
                << "        --gerbicz-b B forces B; --no-gerbicz or -gerbiczli disables it.\n"
                << "        --error-iter I [--error-limb L] [--error-delta D] injects a state error for Gerbicz testing.\n"
                << "        --queue-guard auto|N and --queue-guard-seconds S control clFinish cadence; default auto, 2s.\n"
                << "Debug:  --profile-kernels [--profile-every N], --iters N, --planner-debug, --headroom-bits N.\n"
                << "Sync:   default benchmark path avoids hot-loop clFlush/clFinish; profiling queues are enabled only with --profile-kernels.\n";
            std::exit(0);
        } else if (!arg.empty() && arg[0] != '-') {
            opt.exponent = static_cast<std::uint32_t>(std::stoul(arg));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opt.modulus_mode == "m61" || opt.modulus_mode == "gf61" || opt.modulus_mode == "GF61") opt.modulus_mode = "gf61";
    else if (opt.modulus_mode == "m31" || opt.modulus_mode == "gf31" || opt.modulus_mode == "GF31") opt.modulus_mode = "gf31";
    else if (opt.modulus_mode == "crt" || opt.modulus_mode == "gf61xgf31" || opt.modulus_mode == "GF61xGF31") opt.modulus_mode = "crt";
    else if (opt.modulus_mode == "crt-cpu" || opt.modulus_mode == "cpu-crt") opt.modulus_mode = "crt-cpu";
    else if (opt.modulus_mode == "best" || opt.modulus_mode == "auto") opt.modulus_mode = "best";

    if (opt.exponent == 0 && !opt.no_worktodo) {
        const std::uint32_t wp = parse_first_prp_from_worktodo(opt.worktodo_path);
        if (wp != 0) {
            opt.exponent = wp;
            std::cout << "worktodo: selected PRP exponent p=" << wp << " from " << opt.worktodo_path << "\n";
        }
    }

    return opt;
}

static const char* yesno(bool v) { return v ? "on" : "off"; }


static std::string crt_schedule_name(int v) {
    switch (v) {
        case 0: return "grouped";
        case 1: return "interleave";
        case 2: return "gf31-first";
        case 3: return "gf61-dominant";
        case 4: return "serial-gf31-first";
        case 5: return "gf61-forward-first";
        case 6: return "hybrid-fwd8";
        default: return "unknown";
    }
}

static std::string crt_edge_mode_name(int v) {
    switch (v) {
        case 0: return "auto";
        case 1: return "legacy";
        case 2: return "generic";
        default: return "unknown";
    }
}

static std::string crt_edge_fuse_name(int v) {
    switch (v) {
        case 0: return "off";
        case 1: return "forward";
        case 2: return "tail";
        case 3: return "both";
        default: return "unknown";
    }
}

static bool device_name_has(const std::string& name, const std::string& needle) {
    std::string a = name, b = needle;
    std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return a.find(b) != std::string::npos;
}

static void apply_crt_device_preset(Options& opt, const std::string& device_name) {
    
    
    const bool is_nvidia = device_name_has(device_name, "nvidia") || device_name_has(device_name, "geforce") || device_name_has(device_name, "rtx");
    const bool is_amd = device_name_has(device_name, "amd") || device_name_has(device_name, "gfx") || device_name_has(device_name, "radeon");

    if (!opt.user_crt_queue) opt.crt_async_queues = true;
    if (!opt.user_crt_schedule) opt.crt_defused_schedule = 0;
    if (!opt.user_crt_local_square) opt.crt_center_chunk = 512;
    if (!opt.user_crt_edge_radix) opt.crt_edge_radix = 4;
    if (!opt.user_crt_edge_mode) opt.crt_defused_edge_mode = 1;

    if (is_nvidia) {
        if (!opt.user_crt_local_stage) opt.crt_lds_stage = 512;
        if (!opt.user_crt_edge_fuse) opt.crt_defused_edge_fuse = 3;
    } else if (is_amd) {
        if (!opt.user_crt_local_stage) opt.crt_lds_stage = 0;
        if (!opt.user_crt_edge_fuse) opt.crt_defused_edge_fuse = 0;
    } else {
        if (!opt.user_crt_local_stage) opt.crt_lds_stage = 0;
        if (!opt.user_crt_edge_fuse) opt.crt_defused_edge_fuse = 0;
    }
}

static void print_crt_transform_plan(const Options& opt, const ibdwt::Layout& layout) {
    std::cout << "\n=== CRT transform plan ===\n";
    std::cout << "N=" << layout.n;
    if (layout.odd > 1u) std::cout << " = " << layout.odd << "*2^" << layout.ln << " (row real=" << layout.pow2_n << ", row complex=" << (layout.pow2_n / 2u) << ")";
    else std::cout << " (ln=" << layout.ln << ")";
    std::cout << ", defused-ntt=" << yesno(opt.crt_defused_fast)
              << ", queues=" << (opt.crt_async_queues ? "two" : "single")
              << ", schedule=" << crt_schedule_name(opt.crt_defused_schedule) << "\n";
    std::cout << "edge: radix=" << opt.crt_edge_radix
              << ", mode=" << crt_edge_mode_name(opt.crt_defused_edge_mode)
              << ", fuse=" << crt_edge_fuse_name(opt.crt_defused_edge_fuse) << "\n";
    std::cout << "local: center-square=" << opt.crt_center_chunk
              << ", intermediate-lds-stage=" << opt.crt_lds_stage
              << ", lds-tile=" << opt.crt_lds_tile
              << ", center-mode=" << opt.crt_center_mode << "\n";
    if (opt.crt_center_mode == "halfreal") {
        std::cout << "halfreal: autoprobe=" << yesno(opt.crt_halfreal_autoprobe)
                  << ", flags61=" << (opt.crt_halfreal_flags61 < 0 ? std::string("auto") : std::to_string(opt.crt_halfreal_flags61))
                  << ", flags31=" << (opt.crt_halfreal_flags31 < 0 ? std::string("auto") : std::to_string(opt.crt_halfreal_flags31))
                  << ", lds512=on" << "\n";
        if (layout.odd > 1u) {
            std::cout << "mixed odd: CRT/PFA, odd=" << layout.odd << ", power2 axis=2^" << layout.ln
                      << ", sequence=pack+oddDFT -> row halfreal -> oddIDFT+unweight -> Garner/carry\n";
        } else {
            std::cout << "sequence: pack -> halfreal NTT -> center -> inverse -> unpack -> Garner/carry\n";
        }
    } else {
        std::cout << "logical sequence: edge -> global radix8 stages -> ";
        if (opt.crt_lds_stage > 0) std::cout << "LDS stage" << opt.crt_lds_stage << " -> ";
        if (opt.crt_center_chunk > 0) std::cout << "LDS square" << opt.crt_center_chunk << " -> ";
        std::cout << "inverse stages -> Garner/carry\n";
    }
    std::cout << "==========================\n";
}

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static double run_child_bench(const std::string& cmd) {
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return 0.0;
    char buf[512];
    double best = 0.0;
    while (fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        std::size_t p = line.find("it/s");
        if (p != std::string::npos) {
            
            p += 4;
            while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) ++p;
            try { best = std::stod(line.substr(p)); } catch (...) {}
        }
    }
    pclose(fp);
    return best;
}

static void maybe_run_crt_startup_autotune(Options& opt, const char* argv0) {
    if (!opt.crt_startup_autotune || opt.modulus_mode != "crt") return;

    struct Cand { bool twoq; int sched; std::uint32_t lds; std::uint32_t edge; int mode; int fuse; };
    std::vector<bool> queues = opt.user_crt_queue ? std::vector<bool>{opt.crt_async_queues} : std::vector<bool>{true, false};
    std::vector<int> scheds = opt.user_crt_schedule ? std::vector<int>{opt.crt_defused_schedule} : std::vector<int>{0, 1};
    if (opt.crt_autotune_wide && !opt.user_crt_schedule) scheds = {0, 1, 2, 3, 5, 6};
    std::vector<std::uint32_t> ldss = opt.user_crt_local_stage ? std::vector<std::uint32_t>{opt.crt_lds_stage} : std::vector<std::uint32_t>{0, 512};
    std::vector<std::uint32_t> edges = opt.user_crt_edge_radix ? std::vector<std::uint32_t>{opt.crt_edge_radix} : std::vector<std::uint32_t>{4};
    if (opt.crt_autotune_wide && !opt.user_crt_edge_radix) edges = {4, 8, 16};
    std::vector<int> modes = opt.user_crt_edge_mode ? std::vector<int>{opt.crt_defused_edge_mode} : std::vector<int>{1};
    if (opt.crt_autotune_wide && !opt.user_crt_edge_mode) modes = {1, 2};
    std::vector<int> fuses = opt.user_crt_edge_fuse ? std::vector<int>{opt.crt_defused_edge_fuse} : std::vector<int>{0, 3};

    double best = -1.0;
    Cand bestc{opt.crt_async_queues, opt.crt_defused_schedule, opt.crt_lds_stage, opt.crt_edge_radix, opt.crt_defused_edge_mode, opt.crt_defused_edge_fuse};
    int idx = 0;
    std::cout << "CRT startup autotune: " << opt.crt_autotune_iters << " iterations per candidate" << (opt.crt_autotune_wide ? " (wide)" : "") << "\n";
    for (bool q : queues) for (int sc : scheds) for (auto lds : ldss) for (auto er : edges) for (int em : modes) for (int fu : fuses) {
        if (er != 4 && fu != 0) continue;
        std::ostringstream cmd;
        cmd << shell_quote(argv0) << " " << opt.exponent
            << " --modulus crt --device " << opt.device_index
            << " --crt-no-startup-autotune --iters " << opt.crt_autotune_iters
            << " --crt-defused-ntt --crt-fused-center-experimental"
            << (q ? " --crt-two-queues" : " --crt-single-queue")
            << " --crt-defused-schedule " << crt_schedule_name(sc)
            << " --crt-local-square 512 --crt-local-stage-max " << lds
            << " --crt-edge-radix " << er
            << " --crt-edge-mode " << crt_edge_mode_name(em)
            << " --crt-defused-edge-fuse " << crt_edge_fuse_name(fu)
            << " 2>/dev/null";
        double speed = run_child_bench(cmd.str());
        std::cout << "  [" << idx++ << "] " << speed << " it/s  "
                  << (q ? "two-queues" : "single-queue")
                  << ", schedule=" << crt_schedule_name(sc)
                  << ", lds-stage=" << lds
                  << ", edge-radix=" << er
                  << ", edge-mode=" << crt_edge_mode_name(em)
                  << ", edge-fuse=" << crt_edge_fuse_name(fu) << "\n";
        if (speed > best) { best = speed; bestc = {q, sc, lds, er, em, fu}; }
    }
    if (best > 0) {
        if (!opt.user_crt_queue) opt.crt_async_queues = bestc.twoq;
        if (!opt.user_crt_schedule) opt.crt_defused_schedule = bestc.sched;
        if (!opt.user_crt_local_stage) opt.crt_lds_stage = bestc.lds;
        if (!opt.user_crt_edge_radix) opt.crt_edge_radix = bestc.edge;
        if (!opt.user_crt_edge_mode) opt.crt_defused_edge_mode = bestc.mode;
        if (!opt.user_crt_edge_fuse) opt.crt_defused_edge_fuse = bestc.fuse;
        std::cout << "CRT startup autotune selected: " << best << " it/s  "
                  << (opt.crt_async_queues ? "two-queues" : "single-queue")
                  << ", schedule=" << crt_schedule_name(opt.crt_defused_schedule)
                  << ", lds-stage=" << opt.crt_lds_stage
                  << ", edge-radix=" << opt.crt_edge_radix
                  << ", edge-mode=" << crt_edge_mode_name(opt.crt_defused_edge_mode)
                  << ", edge-fuse=" << crt_edge_fuse_name(opt.crt_defused_edge_fuse) << "\n";
    }
}

struct PlannerTry {
    bool ok = false;
    ibdwt::Layout layout{};
    std::string error;
};

static PlannerTry try_make_planner_layout(std::uint32_t p, unsigned capacity_bits, unsigned shift_mod_bits, unsigned headroom_bits) {
    PlannerTry r;
    try {
        ibdwt::configure_capacity(capacity_bits, shift_mod_bits, headroom_bits);
        r.layout = ibdwt::make_layout(p);
        r.ok = true;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

static void print_one_planner_lane(const char* name, const PlannerTry& t) {
    if (t.ok) std::cerr << name << " ln=" << t.layout.ln << " N=" << t.layout.n;
    else std::cerr << name << " unavailable";
}

static std::string choose_best_modulus_for_current_bridge(std::uint32_t p, unsigned headroom_bits) {
    const PlannerTry l61 = try_make_planner_layout(p, 61, 61, headroom_bits);
    const PlannerTry lcrt = try_make_planner_layout(p, 92, 61, headroom_bits);
    if (l61.ok && lcrt.ok) return (lcrt.layout.n < l61.layout.n) ? "crt" : "gf61";
    if (l61.ok) return "gf61";
    if (lcrt.ok) return "crt";
    throw std::runtime_error("unable to find safe transform size for GF61 or CRT combined modulus");
}

static void print_modulus_planner_note(std::uint32_t p, const std::string& requested, const std::string& selected, unsigned headroom_bits) {
    const PlannerTry l61 = try_make_planner_layout(p, 61, 61, headroom_bits);
    const PlannerTry l31 = try_make_planner_layout(p, 31, 31, headroom_bits);
    const PlannerTry lcrt = try_make_planner_layout(p, 92, 61, headroom_bits);
    std::cerr << "plan: ";
    print_one_planner_lane("gf61", l61);
    std::cerr << ", ";
    print_one_planner_lane("gf31", l31);
    std::cerr << ", ";
    print_one_planner_lane("crt", lcrt);
    std::cerr << "; selected=" << selected << "\n";
    if (requested == "best" || requested == "auto") std::cerr << "best selected " << selected << "\n";
}

static void ensure_selected_modulus_available(std::uint32_t p, const std::string& mode, unsigned headroom_bits) {
    PlannerTry lane;
    if (mode == "gf61") lane = try_make_planner_layout(p, 61, 61, headroom_bits);
    else if (mode == "gf31") lane = try_make_planner_layout(p, 31, 31, headroom_bits);
    else if (mode == "crt" || mode == "crt-cpu") lane = try_make_planner_layout(p, 92, 61, headroom_bits);
    else return;
    if (lane.ok) return;

    std::ostringstream oss;
    oss << "selected modulus '" << mode << "' has no safe transform for p=" << p
        << " with headroom-bits=" << headroom_bits;
    if (mode == "gf31") oss << "; GF31 alone is capacity-limited here, use --modulus crt or --modulus gf61";
    else if (mode == "gf61") oss << "; try --modulus crt if the product field has enough capacity";
    throw std::runtime_error(oss.str());
}

static cl_uint default_center_max_for_plan(const Options& opt, const ibdwt::Layout& layout) {
    if (opt.center_max_user) return opt.center_max;
    if (opt.modulus_mode == "crt" || opt.modulus_mode == "crt-cpu") return (layout.n <= 8192u) ? 16u : 8u;
    if (opt.modulus_mode == "gf31") return 8u;
    
    
    return 0u;
}

static cl_uint default_local_block_lds_for_plan(const Options& opt, const ibdwt::Layout& layout, const clwrap::GpuPrp&) {
    if (opt.local_block_lds != 0u) return opt.local_block_lds;
    if (layout.n < 512u) return 0u;
    if (opt.modulus_mode == "crt" || opt.modulus_mode == "crt-cpu") return (layout.n <= 8192u) ? 512u : 1024u;
    if (opt.modulus_mode == "gf61") return (layout.n >= 1048576u) ? 2048u : 0u;
    return 512u;
}

static cl_uint env_u32_or_default_main(const char* name, cl_uint defv, cl_uint minv, cl_uint maxv) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defv;
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v < minv || v > maxv) return defv;
    return static_cast<cl_uint>(v);
}

static std::string resolve_single_center_mode_auto(const Options& opt, const ibdwt::Layout& layout) {
    if (opt.single_center_mode != "auto") return opt.single_center_mode;
    if (opt.modulus_mode == "gf31") return "normal";
    if (opt.modulus_mode == "gf61") {
        const cl_uint min_n = env_u32_or_default_main("PRMERS_SINGLE_HALFREAL_AUTO_MIN_N", 65536u, 512u, 1u << 30);
        return (layout.n >= min_n) ? "halfreal" : "normal";
    }
    return "normal";
}

static uint32_t resolve_crt_odd_radix_auto(const Options& opt, std::uint32_t p) {
    if (opt.crt_odd_radix != 0u) return opt.crt_odd_radix;
    const auto base = ibdwt::make_layout(p);
    uint32_t selected = 1u;
    try {
        const auto odd9 = ibdwt::make_layout_mixed(p, 9u);
        const bool small_medium_win = (base.ln >= 13u && base.ln <= 15u);
        const bool strong_size_win = (odd9.n * std::size_t(4) <= base.n * std::size_t(3));
        if (small_medium_win || strong_size_win) selected = 9u;
    } catch (...) {
        selected = 1u;
    }
    return selected;
}

static bool path_matches_strict_reference(
    std::uint32_t p,
    const ibdwt::Layout& layout,
    clwrap::GpuPrp& gpu,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    cl_uint local_override,
    bool local_disabled,
    unsigned iterations);

static cl_uint autotune_center_max_gpu(std::uint32_t p, clwrap::GpuPrp& gpu, const clwrap::CarryConfig& carry_cfg, cl_uint requested_iters)
{
    const cl_uint candidates[] = {8u, 16u, 32u};
    cl_uint bench_iters = requested_iters;
    if (bench_iters < 200u) bench_iters = 200u;
    if (bench_iters > 500u) bench_iters = 500u;
    if (p < bench_iters) bench_iters = std::max<cl_uint>(1u, p);

    const ibdwt::Layout layout = ibdwt::make_layout(p);
    const unsigned validate_iters = (gpu.n <= 4096) ? 128u : (gpu.n <= 65536 ? 64u : 32u);

    const bool save_local_disabled = clwrap::g_local_block_lds_disabled;
    const cl_uint save_local_override = clwrap::g_local_block_lds_override;
    const bool save_strict_reference = clwrap::g_force_strict_reference;

    double best_seconds = std::numeric_limits<double>::infinity();
    cl_uint best = 8u;
    bool any_valid = false;

    std::cout << "autotune center-max:";
    for (cl_uint c : candidates) {
        bool ok = false;
        try {
            ok = path_matches_strict_reference(p, layout, gpu, carry_cfg, c, 0u, true, validate_iters);
        } catch (...) {
            ok = false;
        }
        if (!ok) {
            std::cout << " " << c << "=reject";
            continue;
        }

        any_valid = true;
        clwrap::g_local_block_lds_disabled = true;
        clwrap::g_local_block_lds_override = 0;
        clwrap::g_force_strict_reference = false;

        std::vector<std::uint64_t> digits = ibdwt::from_small(3, layout);
        clwrap::upload_digits(gpu, digits);
        clFinish(gpu.queue);

        const auto t0 = std::chrono::steady_clock::now();
        for (cl_uint i = 0; i < bench_iters; ++i) {
            clwrap::enqueue_square_mod(gpu, c);
            clwrap::enqueue_carry(gpu, carry_cfg);
        }
        clFinish(gpu.queue);
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double ips = seconds > 0.0 ? static_cast<double>(bench_iters) / seconds : 0.0;
        std::cout << " " << c << "=" << std::fixed << std::setprecision(1) << ips << " it/s";
        if (seconds < best_seconds) {
            best_seconds = seconds;
            best = c;
        }
    }

    if (!any_valid) {
        best = 8u;
        std::cout << " strict-fallback";
    }
    std::cout << " -> " << best << "\n";

    clwrap::g_local_block_lds_disabled = save_local_disabled;
    clwrap::g_local_block_lds_override = save_local_override;
    clwrap::g_force_strict_reference = save_strict_reference;

    std::vector<std::uint64_t> digits = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu, digits);
    clFinish(gpu.queue);
    return best;
}

struct LocalBlockFlagGuard {
    bool disabled;
    cl_uint override_value;
    bool strict_reference;
    LocalBlockFlagGuard()
        : disabled(clwrap::g_local_block_lds_disabled),
          override_value(clwrap::g_local_block_lds_override),
          strict_reference(clwrap::g_force_strict_reference) {}
    ~LocalBlockFlagGuard() {
        clwrap::g_local_block_lds_disabled = disabled;
        clwrap::g_local_block_lds_override = override_value;
        clwrap::g_force_strict_reference = strict_reference;
    }
};

static bool local_block_candidate_available(clwrap::GpuPrp& gpu, cl_uint candidate)
{
    LocalBlockFlagGuard guard;
    clwrap::g_local_block_lds_disabled = false;
    clwrap::g_local_block_lds_override = candidate;
    clwrap::g_force_strict_reference = false;
    const clwrap::BridgeKernelConfig plan = clwrap::choose_local_block_lds_kernel(gpu);
    return plan.enabled && plan.outer_chunk == candidate;
}

static std::vector<std::uint64_t> local_block_prefix_snapshot(
    std::uint32_t p,
    const ibdwt::Layout& layout,
    clwrap::GpuPrp& gpu,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    bool local_disabled,
    cl_uint local_override,
    bool strict_reference,
    cl_uint requested_iters)
{
    LocalBlockFlagGuard guard;
    clwrap::g_local_block_lds_disabled = local_disabled;
    clwrap::g_local_block_lds_override = local_override;
    clwrap::g_force_strict_reference = strict_reference;

    const bool save_profile = gpu.profile_kernels;
    gpu.profile_kernels = false;

    std::vector<std::uint64_t> digits = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu, digits);
    clFinish(gpu.queue);

    const cl_uint iters = std::max<cl_uint>(1u, std::min<cl_uint>(requested_iters, p));
    for (cl_uint i = 0; i < iters; ++i) {
        clwrap::enqueue_square_mod(gpu, center_max);
        clwrap::enqueue_carry(gpu, carry_cfg);
    }
    clFinish(gpu.queue);

    std::vector<std::uint64_t> out = clwrap::read_digits(gpu);
    gpu.profile_kernels = save_profile;
    clwrap::profile_flush_pending(gpu);
    gpu.profile_order.clear();
    gpu.profile_totals.clear();
    return out;
}

static bool path_matches_strict_reference(
    std::uint32_t p,
    const ibdwt::Layout& layout,
    clwrap::GpuPrp& gpu,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    cl_uint local_override,
    bool local_disabled,
    unsigned iterations)
{
    const std::vector<std::uint64_t> baseline = local_block_prefix_snapshot(
        p, layout, gpu, carry_cfg, 0u, true, 0u, true, static_cast<cl_uint>(iterations));
    const std::vector<std::uint64_t> test = local_block_prefix_snapshot(
        p, layout, gpu, carry_cfg, center_max, local_disabled, local_override, false, static_cast<cl_uint>(iterations));
    return baseline == test;
}

static double local_block_bench_ips(
    std::uint32_t p,
    const ibdwt::Layout& layout,
    clwrap::GpuPrp& gpu,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    cl_uint local_override,
    cl_uint requested_iters)
{
    LocalBlockFlagGuard guard;
    clwrap::g_local_block_lds_disabled = false;
    clwrap::g_local_block_lds_override = local_override;

    const bool save_profile = gpu.profile_kernels;
    gpu.profile_kernels = false;

    std::vector<std::uint64_t> digits = ibdwt::from_small(3, layout);
    clwrap::upload_digits(gpu, digits);
    clFinish(gpu.queue);

    const cl_uint iters = std::max<cl_uint>(1u, std::min<cl_uint>(requested_iters, p));
    const auto t0 = std::chrono::steady_clock::now();
    for (cl_uint i = 0; i < iters; ++i) {
        clwrap::enqueue_square_mod(gpu, center_max);
        clwrap::enqueue_carry(gpu, carry_cfg);
    }
    clFinish(gpu.queue);
    const auto t1 = std::chrono::steady_clock::now();

    gpu.profile_kernels = save_profile;
    clwrap::profile_flush_pending(gpu);
    gpu.profile_order.clear();
    gpu.profile_totals.clear();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    return seconds > 0.0 ? static_cast<double>(iters) / seconds : 0.0;
}

static cl_uint validate_and_select_local_block_lds(
    std::uint32_t p,
    const ibdwt::Layout& layout,
    clwrap::GpuPrp& gpu,
    const clwrap::CarryConfig& carry_cfg,
    cl_uint center_max,
    const Options& opt)
{
    const unsigned validate_iters = (layout.n <= 4096) ? 128u : (layout.n <= 65536 ? 64u : 32u);

    auto accept_candidate = [&](cl_uint candidate) -> bool {
        const bool disable_local = (candidate == 0u);
        if (!disable_local && !local_block_candidate_available(gpu, candidate)) return false;
        try {
            return path_matches_strict_reference(p, layout, gpu, carry_cfg, center_max,
                                                 disable_local ? 0u : candidate,
                                                 disable_local,
                                                 validate_iters);
        } catch (...) {
            return false;
        }
    };

    auto select_candidate = [&](cl_uint candidate) -> cl_uint {
        if (candidate == 0u) {
            clwrap::g_local_block_lds_disabled = true;
            clwrap::g_local_block_lds_override = 0;
        } else {
            clwrap::g_local_block_lds_disabled = false;
            clwrap::g_local_block_lds_override = candidate;
        }
        return candidate;
    };

    if (opt.local_block_lds_disabled) return select_candidate(0u);

    if (opt.local_block_lds != 0u || opt.unsafe_local_block_512) {
        const cl_uint requested = opt.unsafe_local_block_512 ? 512u : opt.local_block_lds;
        if (!local_block_candidate_available(gpu, requested)) {
            throw std::runtime_error("requested local-block-lds size is not available for this transform/device");
        }
        if (!accept_candidate(requested)) {
            throw std::runtime_error("requested local-block-lds size failed strict prefix validation");
        }
        return select_candidate(requested);
    }

    std::vector<cl_uint> candidates;
    const cl_uint preferred = default_local_block_lds_for_plan(opt, layout, gpu);
    if (preferred != 0u) candidates.push_back(preferred);

    if (opt.modulus_mode == "gf61") {
        
        
        if (preferred != 0u) {
            for (cl_uint c : {2048u, 1024u, 512u}) {
                if (std::find(candidates.begin(), candidates.end(), c) == candidates.end()) candidates.push_back(c);
            }
        }
    } else if (opt.modulus_mode == "gf31") {
        for (cl_uint c : {512u, 1024u, 2048u}) {
            if (std::find(candidates.begin(), candidates.end(), c) == candidates.end()) candidates.push_back(c);
        }
    } else {
        for (cl_uint c : {1024u, 512u, 2048u}) {
            if (std::find(candidates.begin(), candidates.end(), c) == candidates.end()) candidates.push_back(c);
        }
    }
    candidates.push_back(0u);

    for (cl_uint candidate : candidates) {
        if (accept_candidate(candidate)) return select_candidate(candidate);
    }

    throw std::runtime_error("no deterministic local-block-lds plan matched the strict prefix reference");
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        const std::string requested_modulus_mode = opt.modulus_mode;
        if (opt.modulus_mode == "best") {
            opt.modulus_mode = choose_best_modulus_for_current_bridge(opt.exponent, opt.headroom_bits);
        }
        if (!opt.selftest_only && opt.exponent != 0) ensure_selected_modulus_available(opt.exponent, opt.modulus_mode, opt.headroom_bits);
        opt.kernel_path = clwrap::resolve_kernel_path(opt.kernel_path, (argc > 0 ? argv[0] : nullptr));
        if (opt.modulus_mode == "gf61") {
            gf61::configure_field(61);
            ibdwt::configure_capacity(61, 61, opt.headroom_bits);
        } else if (opt.modulus_mode == "gf31") {
            gf61::configure_field(31);
            ibdwt::configure_capacity(31, 31, opt.headroom_bits);
        } else if (opt.modulus_mode == "crt" || opt.modulus_mode == "crt-cpu") {
            
            gf61::configure_field(61);
            ibdwt::configure_capacity(92, 61, opt.headroom_bits);
        }
        if (!opt.selftest_only) {
            print_modulus_planner_note(opt.exponent, requested_modulus_mode, opt.modulus_mode, opt.headroom_bits);
            if (opt.modulus_mode == "gf61") { gf61::configure_field(61); ibdwt::configure_capacity(61, 61, opt.headroom_bits); }
            else if (opt.modulus_mode == "gf31") { gf61::configure_field(31); ibdwt::configure_capacity(31, 31, opt.headroom_bits); }
            else if (opt.modulus_mode == "crt" || opt.modulus_mode == "crt-cpu") { gf61::configure_field(61); ibdwt::configure_capacity(92, 61, opt.headroom_bits); }
        }
        const auto devices = clwrap::list_devices();
        if (opt.device_index < 0 || static_cast<std::size_t>(opt.device_index) >= devices.size()) {
            throw std::runtime_error("invalid device index");
        }
        const auto& dev = devices[static_cast<std::size_t>(opt.device_index)];
        std::cout << "Using OpenCL device: [" << opt.device_index << "] " << dev.name << "\n";
        std::cout << "Using kernel source: " << opt.kernel_path << "\n";

        if (opt.modulus_mode == "crt" || opt.modulus_mode == "crt-cpu") {
            if (opt.selftest_only) {
                throw std::runtime_error("--selftest --modulus crt is not wired yet; use a small exponent test such as: ./prmers_opencl_prp 127 --modulus crt");
            }
            if (opt.exponent == 0) throw std::runtime_error("missing exponent p");

            ibdwt::configure_capacity(92, 61, opt.headroom_bits);
            const bool crt_odd_was_auto = opt.crt_odd_radix_auto || opt.crt_odd_radix == 0u;
            opt.crt_odd_radix = resolve_crt_odd_radix_auto(opt, opt.exponent);
            if (crt_odd_was_auto) {
                std::cout << "CRT odd-radix auto selected " << (opt.crt_odd_radix > 1u ? std::to_string(opt.crt_odd_radix) : std::string("off")) << "\n";
            }
            if (opt.crt_odd_radix > 1u) {
                opt.crt_center_mode = "halfreal";
                opt.crt_defused_fast = true;
            }
            const auto layout = ibdwt::make_layout_mixed(opt.exponent, opt.crt_odd_radix);
            apply_crt_device_preset(opt, dev.name);
            maybe_run_crt_startup_autotune(opt, argv[0]);
            std::signal(SIGINT, handle_sigint);
            g_stop_requested.store(false, std::memory_order_relaxed);

            clwrap::g_planner_debug = opt.planner_debug;
            clwrap::g_local_block_lds_disabled = opt.local_block_lds_disabled;
            clwrap::g_local_block_lds_override = opt.local_block_lds;
            clwrap::g_disable_crt_fused_pipeline = !opt.crt_fused_pipeline;
            clwrap::g_crt_radix8_global = opt.crt_radix8_global;
            clwrap::g_crt_center_chunk = opt.crt_center_chunk;
            clwrap::g_crt_lds_stage = opt.crt_lds_stage;
            clwrap::g_crt_lds_tile = opt.crt_lds_tile;
            clwrap::g_crt_head_radix8 = opt.crt_head_radix8;
            clwrap::g_crt_defused_schedule = opt.crt_defused_schedule;
            clwrap::g_crt_fwd8_61_wg = opt.crt_fwd8_61_wg;
            clwrap::g_crt_defused_edge_fuse = opt.crt_defused_edge_fuse;
            clwrap::g_crt_defused_edge_radix = opt.crt_edge_radix;
            clwrap::g_crt_odd_radix = opt.crt_odd_radix;
            clwrap::g_crt_mixed_row_core = opt.crt_mixed_row_core;
            clwrap::g_crt_mixed_row_fuse_both = opt.crt_mixed_row_fuse_both;
            clwrap::g_crt_defused_edge_mode = opt.crt_defused_edge_mode;
            clwrap::g_crt_center_mode = opt.crt_center_mode;
            clwrap::g_crt_halfreal_validate = opt.crt_halfreal_validate;
            clwrap::g_crt_halfreal_validate_iters = opt.crt_halfreal_validate_iters;
            clwrap::g_crt_halfreal_validate_random = opt.crt_halfreal_validate_random;
            clwrap::g_crt_mixed_gpu_reference = opt.crt_mixed_gpu_reference;
            clwrap::g_crt_halfreal_dump_count = opt.crt_halfreal_dump_count;
            clwrap::g_crt_halfreal_dump_prefix = opt.crt_halfreal_dump_prefix;
            clwrap::g_crt_halfreal_flags61 = opt.crt_halfreal_flags61;
            clwrap::g_crt_halfreal_flags31 = opt.crt_halfreal_flags31;
            clwrap::g_crt_halfreal_autoprobe = opt.crt_halfreal_autoprobe;
            clwrap::g_crt_halfreal_probe_exhaustive = opt.crt_halfreal_probe_exhaustive;

            gf61::configure_field(61);
            ibdwt::configure_capacity(92, 61, opt.headroom_bits);
            auto gpu61 = clwrap::make_gpu(dev, opt.kernel_path, layout, opt.profile_kernels, opt.prefer_radix4x2);

            gf61::configure_field(31);
            ibdwt::configure_capacity(92, 31, opt.headroom_bits);
            cl_command_queue crt_shared_queue = opt.crt_async_queues ? nullptr : gpu61.queue;
            auto gpu31 = clwrap::make_gpu(dev, opt.kernel_path, layout, opt.profile_kernels, opt.prefer_radix4x2,
                                          gpu61.context, crt_shared_queue);

            auto carry_cfg_crt = clwrap::choose_crt_carry_config(dev, layout.n, opt.carry_block, opt.carry_items);

            
            gf61::configure_field(61);
            ibdwt::configure_capacity(92, 61, opt.headroom_bits);
            if (opt.autotune_center && !opt.center_max_user) {
                opt.center_max = autotune_center_max_gpu(opt.exponent, gpu61, carry_cfg_crt, opt.autotune_center_iters);
            } else {
                opt.center_max = default_center_max_for_plan(opt, layout);
            }
            if (layout.odd == 1u) {
                validate_and_select_local_block_lds(opt.exponent, layout, gpu61, carry_cfg_crt, opt.center_max, opt);

                const unsigned selected_check_iters = (layout.n <= 4096) ? 128u : (layout.n <= 65536 ? 64u : 32u);
                if (!path_matches_strict_reference(opt.exponent, layout, gpu61, carry_cfg_crt, opt.center_max,
                                                   clwrap::g_local_block_lds_override,
                                                   clwrap::g_local_block_lds_disabled,
                                                   selected_check_iters)) {
                    throw std::runtime_error("CRT GF61 deterministic plan failed strict prefix validation; try --no-local-block-lds or --center-max 0");
                }
                gf61::configure_field(31);
                ibdwt::configure_capacity(92, 31, opt.headroom_bits);
                if (!path_matches_strict_reference(opt.exponent, layout, gpu31, carry_cfg_crt, opt.center_max,
                                                   clwrap::g_local_block_lds_override,
                                                   clwrap::g_local_block_lds_disabled,
                                                   selected_check_iters)) {
                    throw std::runtime_error("CRT GF31 deterministic plan failed strict prefix validation; try --crt-split-center, --no-local-block-lds or --center-max 0");
                }
                gf61::configure_field(61);
                ibdwt::configure_capacity(92, 61, opt.headroom_bits);
            } else {
                clwrap::g_local_block_lds_override = 0;
                clwrap::g_local_block_lds_disabled = true;
                std::cout << "mixed CRT/PFA odd-radix path: strict classic-prefix validator skipped (different digit order); using row half-real core.\n";
                gf61::configure_field(61);
                ibdwt::configure_capacity(92, 61, opt.headroom_bits);
            }

            std::cout << "p=" << opt.exponent << ", ln=" << layout.ln;
            if (layout.odd > 1u) std::cout << ", transform=" << layout.odd << "*2^" << layout.ln << " = " << layout.n
                                           << ", storage=" << (layout.n / 2u) << " complex values";
            else std::cout << ", transform=" << layout.n;
            std::cout << " (GF(M61^2)xGF(M31^2), ";
            if (layout.odd > 1u) std::cout << "mixed CRT/PFA odd half-real rows, ";
            if (opt.modulus_mode == "crt") std::cout << "CRT Garner GPU bridge";
            else std::cout << "CRT Garner CPU bridge";
            std::cout << ", no bit-reverse kernel)\n";
            const cl_uint crt_garner_items = crt_tune::garner_items(static_cast<cl_uint>(layout.n), gpu61.min_digit_width, static_cast<cl_uint>(carry_cfg_crt.items_per_worker));
            const cl_uint crt_segments = static_cast<cl_uint>((layout.n + std::size_t(crt_garner_items) - 1u) / std::size_t(crt_garner_items));
            const cl_uint crt_garner_local = crt_tune::garner_local(crt_segments);
            const cl_uint crt_bits_per_segment = std::max<cl_uint>(1u, crt_garner_items * gpu61.min_digit_width);
            const cl_uint crt_passes = std::max<cl_uint>(1u, std::min<cl_uint>(8u, (128u + crt_bits_per_segment - 1u) / crt_bits_per_segment));
            if (layout.odd > 1u) {
                std::cout << "mixed CRT/PFA path: odd=" << layout.odd
                          << ", pow2-axis=2^" << layout.ln
                          << ", row-real=" << layout.pow2_n
                          << ", row-complex=" << (layout.pow2_n / 2u)
                          << ", storage=" << (layout.n / 2u)
                          << ", kernels=fused(pack+oddDFT)+row LDS512 halfreal+fused(oddIDFT+unpack) (fallback radix2 below 512)\n";
            }
            std::cout << "carry config CRT: block=" << carry_cfg_crt.block_size
                      << ", items/worker=" << carry_cfg_crt.items_per_worker
                      << ", local=" << carry_cfg_crt.local_size
                      << ", garner-items=" << crt_garner_items
                      << ", garner-local=" << crt_garner_local
                      << ", segments=" << crt_segments
                      << ", parallel-passes=" << crt_passes
                      << ", queues=" << (opt.crt_async_queues ? "async-event" : "shared-serial")
                      << ", profile-queue=" << (opt.profile_kernels ? "on" : "off")
                      << ", host-sync=" << (opt.profile_kernels ? "profile-reports" : "final-only")
                      << ", center-max=" << opt.center_max
                      << ", local-block-lds=" << clwrap::describe_local_block_lds_choice(gpu61)
                      << ", garner=shift-sub-M61"
                      << ", global-stages=" << ((opt.crt_radix8_global) ? "radix8-preferred" : "radix4")
                      << ", crt-local-square=" << opt.crt_center_chunk
                      << ", crt-local-stage-max=" << opt.crt_lds_stage
                      << ", crt-lds-tile=" << opt.crt_lds_tile
                      << ", crt-head-radix8=" << opt.crt_head_radix8
                      << ", crt-edge-radix=" << opt.crt_edge_radix
                      << ", crt-edge-mode=" << (opt.crt_defused_edge_mode == 2 ? "generic" : (opt.crt_defused_edge_mode == 1 ? "legacy" : "auto"))
                      << ", crt-odd-radix=" << opt.crt_odd_radix
                      << ", cleanup="
                      << ((std::getenv("PRMERS_CRT_SERIAL_CLEANUP") != nullptr) ? "serial-forced" :
                          ((std::getenv("PRMERS_CRT_PARALLEL_CLEANUP") != nullptr) ? "parallel-env+serial-guard" : "serial-guard"))
                      << "\n";

            print_crt_transform_plan(opt, layout);

            g_runtime.exponent = opt.exponent;
            g_runtime.mode_label = (layout.odd > 1u) ? "BananaNTT mixed CRT/PFA half-real" : "PrMers CRT";
            g_runtime.res64_every = opt.res64_every;
            g_runtime.json_enabled = opt.json_enabled;
            g_runtime.json_path = opt.json_path;
            g_runtime.output_dir = opt.output_dir;
            g_runtime.results_path = opt.results_path;
            g_runtime.append_results = opt.append_results;
            g_runtime.proof_checkpoints = opt.proof_checkpoints;
            g_runtime.proof_power = opt.proof_power;
            g_runtime.proof_dir = opt.proof_dir;
            g_runtime.gerbicz_interval = opt.gerbicz_interval;
            g_runtime.gerbicz_enabled = opt.gerbicz_enabled;
            g_runtime.gerbicz_gpu_verify = opt.gerbicz_gpu_verify;
            g_runtime.gerbicz_user_checklevel = opt.gerbicz_user_checklevel;
            g_runtime.gerbicz_checklevel = opt.gerbicz_checklevel;
            g_runtime.gerbicz_block = opt.gerbicz_block;
            g_runtime.gerbicz_target_seconds = opt.gerbicz_target_seconds;
            g_runtime.gerbicz_estimated_ips = opt.gerbicz_estimated_ips;
            g_runtime.gerbicz_user_seconds = opt.gerbicz_user_seconds;
            g_runtime.gerbicz_boundary_seconds = opt.gerbicz_boundary_seconds;
            g_runtime.gerbicz_verbose = opt.gerbicz_verbose;
            g_runtime.gerbicz_progress = opt.gerbicz_progress;
            g_runtime.gerbicz_errors = 0;
            g_runtime.last_proof_file.clear();
            g_runtime.error_iter = opt.error_iter;
            g_runtime.error_limb = opt.error_limb;
            g_runtime.error_delta = opt.error_delta;
            g_runtime.error_injected = false;
            g_runtime.backup_enabled = opt.backup_enabled;
            g_runtime.resume_enabled = opt.resume_enabled;
            g_runtime.save_on_interrupt = opt.save_on_interrupt;
            g_runtime.backup_path = opt.backup_path;
            g_runtime.resume_path = opt.resume_path;
            g_runtime.backup_dir = opt.backup_dir;
            g_runtime.backup_every_iters = opt.backup_every_iters;
            g_runtime.backup_every_seconds = opt.backup_every_seconds;
            g_runtime.queue_guard_depth = opt.queue_guard_depth;
            g_runtime.queue_guard_auto = opt.queue_guard_auto;
            g_runtime.queue_guard_seconds = opt.queue_guard_seconds;
            if (g_runtime.res64_every)
                std::cout << "optional res64 display every " << g_runtime.res64_every << " iteration(s); this reads data back from the GPU.\n";
            if (g_runtime.proof_checkpoints)
                std::cout << "experimental PRP checkpoint output enabled; this writes residue checkpoints, not a PrimeNet upload.\n";
            if (g_runtime.gerbicz_enabled)
                std::cout << "Gerbicz-Li enabled by default: Li relation, backend="
                          << (g_runtime.gerbicz_gpu_verify ? "gpu-fullcheck+gpu-D-update" : "host-exact-GMP")
                          << "; use --gerbicz-host for GMP full-check or --no-gerbicz/-gerbiczli to disable.\n";
            if (g_runtime.error_iter)
                std::cout << "error injection armed at iteration " << g_runtime.error_iter << " limb=" << g_runtime.error_limb << " delta=" << g_runtime.error_delta << "\n";

            bool prp = false;
            if (opt.modulus_mode == "crt") {
                prp = mersenne_prp::prp_mersenne_pow2_base3_gpu_crt_garner(
                    opt.exponent, opt.verbose, gpu61, gpu31, layout, carry_cfg_crt, opt.center_max, opt.profile_every, opt.max_iters, opt.crt_split_center, opt.crt_fused_center_lockstep, opt.crt_defused_fast);
            } else {
                std::cout << "WARNING: --modulus crt-cpu keeps the old validation path and performs Garner+carry on CPU after each iteration.\n";
                prp = mersenne_prp::prp_mersenne_pow2_base3_gpu_crt_cpu_garner(
                    opt.exponent, opt.verbose, gpu61, gpu31, layout, opt.center_max, opt.profile_every, opt.max_iters);
            }
            if (opt.max_iters && opt.max_iters < opt.exponent) {
                std::cout << "benchmark stopped after " << opt.max_iters << " iterations; no PRP result computed.\n";
                return 0;
            }
            std::cout << "2^" << opt.exponent << " - 1 is " << (prp ? "PRP" : "composite") << "\n";
            return prp ? 0 : 1;
        }

        if (opt.selftest_only) {
            const auto carry_cfg = clwrap::choose_carry_config(dev, ibdwt::make_layout(127u).n, opt.carry_block, opt.carry_items);
            std::cout << "carry config: block=" << carry_cfg.block_size << ", items/worker=" << carry_cfg.items_per_worker
                      << ", local=" << carry_cfg.local_size << ", blocks(for selftest max size)=" << carry_cfg.num_blocks << "\n";
            mersenne_prp::selftest(dev, opt.kernel_path, opt.carry_block, opt.carry_items);
            std::cout << "In-place OpenCL " << gf61::FIELD_NAME << " PRP self-tests: PASS\n";
            return 0;
        }

        if (opt.exponent == 0) throw std::runtime_error("missing exponent p");
        const auto layout = ibdwt::make_layout(opt.exponent);
        opt.single_center_mode = resolve_single_center_mode_auto(opt, layout);
        std::signal(SIGINT, handle_sigint);
        g_stop_requested.store(false, std::memory_order_relaxed);
        auto gpu = clwrap::make_gpu(dev, opt.kernel_path, layout, opt.profile_kernels, opt.prefer_radix4x2);
        clwrap::g_planner_debug = opt.planner_debug;
        clwrap::g_local_block_lds_disabled = opt.local_block_lds_disabled;
        clwrap::g_local_block_lds_override = opt.local_block_lds;
        clwrap::g_single_center_mode = opt.single_center_mode;
        clwrap::g_crt_halfreal_flags61 = opt.crt_halfreal_flags61;
        clwrap::g_crt_halfreal_autoprobe = false;
        const auto run_carry_cfg = clwrap::choose_carry_config(dev, layout.n, opt.carry_block, opt.carry_items);
        if (opt.autotune_center && !opt.center_max_user && !opt.selftest_only) {
            opt.center_max = autotune_center_max_gpu(opt.exponent, gpu, run_carry_cfg, opt.autotune_center_iters);
        } else {
            opt.center_max = default_center_max_for_plan(opt, layout);
        }
        validate_and_select_local_block_lds(opt.exponent, layout, gpu, run_carry_cfg, opt.center_max, opt);

        const unsigned selected_check_iters = (layout.n <= 4096) ? 128u : (layout.n <= 65536 ? 64u : 32u);
        if (!path_matches_strict_reference(opt.exponent, layout, gpu, run_carry_cfg, opt.center_max,
                                           clwrap::g_local_block_lds_override,
                                           clwrap::g_local_block_lds_disabled,
                                           selected_check_iters)) {
            throw std::runtime_error("deterministic plan failed strict prefix validation; try --no-local-block-lds or --center-max 0");
        }

        std::cout << "p=" << opt.exponent << ", ln=" << layout.ln
                  << ", transform=" << layout.n << " (" << gf61::FIELD_NAME << ", in-place field buffer, no bit-reverse kernel)\n";
        const std::string lds_choice = clwrap::describe_local_block_lds_choice(gpu);
        std::cout << "device caps: local_mem=" << (gpu.local_mem_size / 1024u) << " KiB"
                  << ", max_wg=" << gpu.max_work_group_size << "\n";
        std::cout << "carry config: block=" << run_carry_cfg.block_size << ", items/worker=" << run_carry_cfg.items_per_worker
                  << ", local=" << run_carry_cfg.local_size << ", blocks=" << run_carry_cfg.num_blocks
                  << ", center-max=" << opt.center_max
                  << ", single-center-mode=" << opt.single_center_mode
                  << ", local-block-lds=" << lds_choice
                  << ", headroom-bits=" << opt.headroom_bits
                  << ", max-digit-width=" << ibdwt::max_digit_width_for_log2(opt.exponent, layout.ln)
                  << (opt.unsafe_local_block_512 ? " (unsafe-512)" : "")
                  << "\n";
        const bool prp = mersenne_prp::prp_mersenne_pow2_base3_gpu(opt.exponent, opt.verbose, gpu, run_carry_cfg, opt.center_max, opt.profile_every, opt.max_iters);
        if (opt.max_iters && opt.max_iters < opt.exponent) {
            std::cout << "benchmark stopped after " << opt.max_iters << " iterations; no PRP result computed.\n";
            return 0;
        }
        std::cout << "2^" << opt.exponent << " - 1 is " << (prp ? "PRP" : "composite") << "\n";
        return prp ? 0 : 1;
    } catch (const InterruptedRun&) {
        std::cerr << "Run interrupted.\n";
        return 130;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
