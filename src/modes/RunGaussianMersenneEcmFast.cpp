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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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

constexpr const char* GM_ECM_FAST_RELEASE = "v99.97";
constexpr std::array<char, 8> GM_NAF_MAGIC{{'P','R','G','M','N','A','F','1'}};
constexpr std::uint32_t GM_NAF_CHECKPOINT_VERSION = 1;

struct GmFastTarget {
    std::string family = "GM";
    std::uint32_t p = 0;
    std::uint32_t lift = 0;
    std::uint64_t middle = 0;
    int chi = 0;
    mpz_class n;
};

std::uint64_t add_mod_u64_fast(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
    return a >= mod - b ? a - (mod - b) : a + b;
}

std::uint64_t mul_mod_u64_fast(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
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
        if ((b & 1U) != 0) r = add_mod_u64_fast(r, a, mod);
        b >>= 1U;
        if (b != 0) a = add_mod_u64_fast(a, a, mod);
    }
    return r;
#endif
}

std::uint64_t pow_mod_u64_fast(std::uint64_t a, std::uint64_t e, std::uint64_t mod) {
    std::uint64_t r = 1 % mod;
    while (e != 0) {
        if ((e & 1U) != 0) r = mul_mod_u64_fast(r, a, mod);
        e >>= 1U;
        if (e != 0) a = mul_mod_u64_fast(a, a, mod);
    }
    return r;
}

bool is_prime_u64_fast(std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint32_t p : {2U,3U,5U,7U,11U,13U,17U,19U,23U,29U,31U,37U}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    std::uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1U) == 0) { d >>= 1U; ++s; }
    constexpr std::array<std::uint64_t, 7> bases{{2ULL, 325ULL, 9375ULL, 28178ULL,
                                                   450775ULL, 9780504ULL, 1795265022ULL}};
    for (std::uint64_t a : bases) {
        if (a % n == 0) continue;
        std::uint64_t x = pow_mod_u64_fast(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_u64_fast(x, x, n);
            if (x == n - 1) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}

GmFastTarget make_fast_target(std::uint64_t p64, const std::string& family) {
    if (p64 < 3 || (p64 & 1ULL) == 0 || !is_prime_u64_fast(p64))
        throw std::runtime_error("Gaussian ECM requires an odd prime exponent p >= 3");
    if (p64 > std::numeric_limits<std::uint32_t>::max() / 4ULL)
        throw std::runtime_error("Gaussian ECM requires 4p <= 2^32-1");
    if (family != "GM" && family != "GQ")
        throw std::runtime_error("Gaussian ECM target family must be GM or GQ");

    GmFastTarget t;
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

mpz_class mod_positive_fast(mpz_class x, const mpz_class& n) {
    x %= n;
    if (x < 0) x += n;
    return x;
}

mpz_class gcd_fast(const mpz_class& a, const mpz_class& n) {
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), n.get_mpz_t());
    return g;
}

bool proper_factor_fast(const mpz_class& g, const mpz_class& n) {
    return g > 1 && g < n;
}

mpz_class project_reg_fast(engine* eng, engine::Reg reg, const mpz_class& n) {
    mpz_t z;
    mpz_init(z);
    eng->get_mpz(z, reg);
    mpz_class out(z);
    mpz_clear(z);
    return mod_positive_fast(out, n);
}

void set_reg_fast(engine* eng, engine::Reg reg, const mpz_class& value, const mpz_class& n) {
    mpz_class v = mod_positive_fast(value, n);
    mpz_t z;
    mpz_init_set(z, v.get_mpz_t());
    eng->set_mpz(reg, z);
    mpz_clear(z);
}

std::uint64_t splitmix64_fast(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

std::string low_hex_fast(const mpz_class& x, unsigned bits = 64) {
    mpz_class low;
    mpz_fdiv_r_2exp(low.get_mpz_t(), x.get_mpz_t(), bits);
    std::string s = low.get_str(16);
    const std::size_t width = bits / 4;
    if (s.size() < width) s.insert(0, width - s.size(), '0');
    return s;
}

struct TeCurveSetup {
    mpz_class a;
    mpz_class d;
    mpz_class x;
    mpz_class y;
    mpz_class factor;
    bool ok = false;
};

bool invert_or_factor(const mpz_class& value,
                      const mpz_class& n,
                      mpz_class& inverse,
                      mpz_class& factor) {
    const mpz_class v = mod_positive_fast(value, n);
    if (mpz_invert(inverse.get_mpz_t(), v.get_mpz_t(), n.get_mpz_t()) != 0) return true;
    factor = gcd_fast(v, n);
    return false;
}

// Convert the exact same Suyama curve/point used by the legacy GM Montgomery
// ladder into a twisted-Edwards model.  All inversions happen modulo the
// selected GM/GQ norm on the CPU.  The GPU then evaluates polynomial Edwards
// formulas in the 2^(4p)-1 lift, whose projection modulo the selected norm is
// exact because the norm divides the lift modulus.
TeCurveSetup make_suyama_twisted_edwards(const mpz_class& n, std::uint64_t sigma64) {
    TeCurveSetup out;
    mpz_class sigma;
    mpz_import(sigma.get_mpz_t(), 1, 1, sizeof(sigma64), 0, 0, &sigma64);
    sigma %= n;
    if (sigma < 6) sigma += 6;

    const mpz_class u = mod_positive_fast(sigma * sigma - 5, n);
    const mpz_class v = mod_positive_fast(4 * sigma, n);
    const mpz_class u2 = mod_positive_fast(u * u, n);
    const mpz_class u3 = mod_positive_fast(u2 * u, n);
    const mpz_class v2 = mod_positive_fast(v * v, n);
    const mpz_class v3 = mod_positive_fast(v2 * v, n);
    const mpz_class vu = mod_positive_fast(v - u, n);

    mpz_class inv, factor;

    // A+2 = (v-u)^3(3u+v)/(4u^3v)
    const mpz_class den_a = mod_positive_fast(4 * u3 * v, n);
    if (!invert_or_factor(den_a, n, inv, factor)) {
        if (proper_factor_fast(factor, n)) out.factor = factor;
        return out;
    }
    const mpz_class aplus2 = mod_positive_fast(vu * vu * vu * (3 * u + v) * inv, n);

    // B = u/v in the Montgomery model used by the Edwards conversion.
    if (!invert_or_factor(v, n, inv, factor)) {
        if (proper_factor_fast(factor, n)) out.factor = factor;
        return out;
    }
    const mpz_class Bm = mod_positive_fast(u * inv, n);
    if (!invert_or_factor(Bm, n, inv, factor)) {
        if (proper_factor_fast(factor, n)) out.factor = factor;
        return out;
    }
    out.a = mod_positive_fast(aplus2 * inv, n);
    out.d = mod_positive_fast((aplus2 - 4) * inv, n);

    // Starting point in twisted-Edwards affine coordinates.  These formulas
    // are the same conversion already used by PrMers' regular Mersenne ECM.
    const mpz_class sp = mod_positive_fast(sigma * sigma + 5, n);
    const mpz_class den_x = mod_positive_fast((u - v) * (u + v) * sp, n);
    if (!invert_or_factor(den_x, n, inv, factor)) {
        if (proper_factor_fast(factor, n)) out.factor = factor;
        return out;
    }
    out.x = mod_positive_fast(u2 * v * inv, n);

    const mpz_class den_y = mod_positive_fast(u3 + v3, n);
    if (!invert_or_factor(den_y, n, inv, factor)) {
        if (proper_factor_fast(factor, n)) out.factor = factor;
        return out;
    }
    out.y = mod_positive_fast((u3 - v3) * inv, n);

    // Cheap CPU sanity check of a*x^2 + y^2 = 1 + d*x^2*y^2.
    const mpz_class x2 = mod_positive_fast(out.x * out.x, n);
    const mpz_class y2 = mod_positive_fast(out.y * out.y, n);
    const mpz_class lhs = mod_positive_fast(out.a * x2 + y2, n);
    const mpz_class rhs = mod_positive_fast(1 + out.d * x2 * y2, n);
    if (lhs != rhs) return out;

    out.ok = true;
    return out;
}

std::vector<short> naf_digits(const mpz_class& scalar) {
    std::vector<short> naf;
    mpz_class e = scalar;
    while (e != 0) {
        short digit = 0;
        if (mpz_odd_p(e.get_mpz_t())) {
            const unsigned long limb0 = mpz_getlimbn(e.get_mpz_t(), 0);
            digit = ((limb0 & 3UL) == 1UL) ? 1 : -1;
            if (digit > 0) e -= 1;
            else e += 1;
        }
        naf.push_back(digit);
        mpz_fdiv_q_2exp(e.get_mpz_t(), e.get_mpz_t(), 1);
    }
    while (!naf.empty() && naf.back() == 0) naf.pop_back();
    return naf;
}

struct TeRegs {
    static constexpr engine::Reg RZ = 1;
    static constexpr engine::Reg RX = 3;
    static constexpr engine::Reg RY = 4;
    static constexpr engine::Reg RT = 5;
    static constexpr engine::Reg PX = 6;
    static constexpr engine::Reg PY = 7;
    static constexpr engine::Reg PT = 9;
    static constexpr engine::Reg MA = 43;
    static constexpr engine::Reg MD = 45;
    static constexpr engine::Reg MPT = 46;
    static constexpr engine::Reg A = 16;
    static constexpr engine::Reg D = 29;
    static constexpr engine::Reg NX = 47;
    static constexpr engine::Reg NY = 48;
    static constexpr engine::Reg NT = 49;
    static constexpr engine::Reg MNT = 50;
    static constexpr std::size_t count = 51;
};

void te_restore_multiplicands(engine* eng) {
    eng->set_multiplicand(TeRegs::MA, TeRegs::A);
    eng->set_multiplicand(TeRegs::MD, TeRegs::D);
    eng->set_multiplicand(TeRegs::MPT, TeRegs::PT);
    eng->set_multiplicand(TeRegs::MNT, TeRegs::NT);
}

void te_init(engine* eng, const TeCurveSetup& c, const mpz_class& n) {
    for (std::size_t reg = 0; reg < TeRegs::count; ++reg) eng->set(static_cast<engine::Reg>(reg), 0u);

    set_reg_fast(eng, TeRegs::A, c.a, n);
    set_reg_fast(eng, TeRegs::D, c.d, n);
    set_reg_fast(eng, TeRegs::PX, c.x, n);
    set_reg_fast(eng, TeRegs::PY, c.y, n);
    const mpz_class t = mod_positive_fast(c.x * c.y, n);
    set_reg_fast(eng, TeRegs::PT, t, n);
    set_reg_fast(eng, TeRegs::NX, n - c.x, n);
    set_reg_fast(eng, TeRegs::NY, c.y, n);
    set_reg_fast(eng, TeRegs::NT, n - t, n);

    eng->set(TeRegs::RZ, 1u);
    te_restore_multiplicands(eng);
}

// Extended twisted-Edwards doubling, copied from the proven regular ECM path.
void te_double(engine* eng) {
    eng->set_multiplicand(11, TeRegs::RZ);
    eng->mul(TeRegs::RT, 11);
    eng->add(TeRegs::RT, TeRegs::RT);
    eng->square_mul(TeRegs::RZ);
    eng->add(TeRegs::RZ, TeRegs::RZ);
    eng->square_mul(TeRegs::RX);
    eng->square_mul(TeRegs::RY);
    eng->mul(TeRegs::RX, TeRegs::MA);
    eng->addsub(23, 25, TeRegs::RX, TeRegs::RY);
    eng->copy(24, 23);
    eng->sub_reg(24, TeRegs::RZ);
    eng->set_multiplicand(11, 24);
    eng->copy(TeRegs::RX, TeRegs::RT);
    eng->mul(TeRegs::RX, 11);
    eng->copy(TeRegs::RZ, 23);
    eng->mul(TeRegs::RZ, 11);
    eng->set_multiplicand(11, 25);
    eng->copy(TeRegs::RY, 23);
    eng->mul(TeRegs::RY, 11);
    eng->mul(TeRegs::RT, 11);
}

// R <- R + P, where P is the fixed positive starting point.
void te_add_pos(engine* eng) {
    eng->addsub(34, 35, TeRegs::RY, TeRegs::RX);
    eng->addsub(36, 37, TeRegs::PY, TeRegs::PX);
    eng->copy(30, TeRegs::RX);
    eng->set_multiplicand(11, TeRegs::PX);
    eng->mul_copy(30, 11, 39);
    eng->copy(31, TeRegs::RY);
    eng->set_multiplicand(11, TeRegs::PY);
    eng->mul(31, 11);
    eng->copy(32, TeRegs::RT);
    eng->mul(32, TeRegs::MPT);
    eng->mul(32, TeRegs::MD);
    eng->addsub(42, 41, TeRegs::RZ, 32);
    eng->copy(38, 34);
    eng->set_multiplicand(11, 36);
    eng->mul(38, 11);
    eng->sub_reg(38, 30);
    eng->sub_reg(38, 31);
    eng->mul(39, TeRegs::MA);
    eng->copy(40, 31);
    eng->sub_reg(40, 39);
    eng->copy(TeRegs::RX, 38);
    eng->set_multiplicand(11, 41);
    eng->mul(TeRegs::RX, 11);
    eng->copy(TeRegs::RZ, 42);
    eng->mul(TeRegs::RZ, 11);
    eng->set_multiplicand(11, 40);
    eng->copy(TeRegs::RY, 42);
    eng->mul(TeRegs::RY, 11);
    eng->copy(TeRegs::RT, 38);
    eng->mul(TeRegs::RT, 11);
}

// R <- R - P, using the fixed negative point (-x,y,-t).
void te_add_neg(engine* eng) {
    eng->addsub(34, 35, TeRegs::RY, TeRegs::RX);
    eng->addsub(36, 37, TeRegs::NY, TeRegs::NX);
    eng->copy(30, TeRegs::RX);
    eng->set_multiplicand(11, TeRegs::NX);
    eng->mul_copy(30, 11, 39);
    eng->copy(31, TeRegs::RY);
    eng->set_multiplicand(11, TeRegs::NY);
    eng->mul(31, 11);
    eng->copy(32, TeRegs::RT);
    eng->mul(32, TeRegs::MNT);
    eng->mul(32, TeRegs::MD);
    eng->addsub(42, 41, TeRegs::RZ, 32);
    eng->copy(38, 34);
    eng->set_multiplicand(11, 36);
    eng->mul(38, 11);
    eng->sub_reg(38, 30);
    eng->sub_reg(38, 31);
    eng->mul(39, TeRegs::MA);
    eng->copy(40, 31);
    eng->sub_reg(40, 39);
    eng->copy(TeRegs::RX, 38);
    eng->set_multiplicand(11, 41);
    eng->mul(TeRegs::RX, 11);
    eng->copy(TeRegs::RZ, 42);
    eng->mul(TeRegs::RZ, 11);
    eng->set_multiplicand(11, 40);
    eng->copy(TeRegs::RY, 42);
    eng->mul(TeRegs::RY, 11);
    eng->copy(TeRegs::RT, 38);
    eng->mul(TeRegs::RT, 11);
}

void te_set_from_top_digit(engine* eng, short top) {
    if (top < 0) {
        eng->copy(TeRegs::RX, TeRegs::NX);
        eng->copy(TeRegs::RY, TeRegs::NY);
        eng->copy(TeRegs::RT, TeRegs::NT);
    } else {
        eng->copy(TeRegs::RX, TeRegs::PX);
        eng->copy(TeRegs::RY, TeRegs::PY);
        eng->copy(TeRegs::RT, TeRegs::PT);
    }
    eng->set(TeRegs::RZ, 1u);
}

struct NafCheckpointHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t p;
    std::uint32_t lift;
    std::uint32_t curve;
    std::uint64_t B1;
    std::uint64_t sigma;
    std::uint64_t total_steps;
    std::uint64_t done_steps;
    double elapsed;
    std::uint64_t checkpoint_bytes;
};

bool load_naf_checkpoint(const std::filesystem::path& path,
                         engine* eng,
                         const GmFastTarget& t,
                         std::uint64_t B1,
                         std::uint32_t curve,
                         std::uint64_t sigma,
                         std::uint64_t total_steps,
                         std::uint64_t& done_steps,
                         double& restored) {
    File f(path.string());
    if (!f.exists()) return false;
    NafCheckpointHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (!std::equal(GM_NAF_MAGIC.begin(), GM_NAF_MAGIC.end(), h.magic) ||
        h.version != GM_NAF_CHECKPOINT_VERSION || h.p != t.p || h.lift != t.lift ||
        h.curve != curve || h.B1 != B1 || h.sigma != sigma || h.total_steps != total_steps ||
        h.done_steps > total_steps || h.checkpoint_bytes != eng->get_checkpoint_size()) return false;
    std::vector<char> data(eng->get_checkpoint_size());
    if (!f.read(data.data(), data.size()) || !f.check_crc32() || !eng->set_checkpoint(data)) return false;
    done_steps = h.done_steps;
    restored = h.elapsed;
    te_restore_multiplicands(eng);
    return true;
}

void save_naf_checkpoint(const std::filesystem::path& path,
                         engine* eng,
                         const GmFastTarget& t,
                         std::uint64_t B1,
                         std::uint32_t curve,
                         std::uint64_t sigma,
                         std::uint64_t total_steps,
                         std::uint64_t done_steps,
                         double elapsed) {
    eng->sync();
    NafCheckpointHeader h{};
    std::copy(GM_NAF_MAGIC.begin(), GM_NAF_MAGIC.end(), h.magic);
    h.version = GM_NAF_CHECKPOINT_VERSION;
    h.p = t.p;
    h.lift = t.lift;
    h.curve = curve;
    h.B1 = B1;
    h.sigma = sigma;
    h.total_steps = total_steps;
    h.done_steps = done_steps;
    h.elapsed = elapsed;
    h.checkpoint_bytes = eng->get_checkpoint_size();

    std::vector<char> data(eng->get_checkpoint_size());
    if (!eng->get_checkpoint(data)) throw std::runtime_error("cannot capture GM ECM NAF checkpoint");

    const std::filesystem::path new_path = path.string() + ".new";
    const std::filesystem::path old_path = path.string() + ".old";
    {
        File f(new_path.string(), "wb");
        if (!f.write(reinterpret_cast<const char*>(&h), sizeof(h)) ||
            !f.write(data.data(), data.size()))
            throw std::runtime_error("cannot write GM ECM NAF checkpoint");
        f.write_crc32();
    }
    std::error_code ec;
    std::filesystem::remove(old_path, ec);
    ec.clear();
    if (std::filesystem::exists(path)) std::filesystem::rename(path, old_path, ec);
    ec.clear();
    std::filesystem::rename(new_path, path, ec);
    if (ec) throw std::runtime_error("cannot install GM ECM NAF checkpoint: " + ec.message());
}

void clear_naf_checkpoint(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".old", ec);
    std::filesystem::remove(path.string() + ".new", ec);
}

std::string json_escape_fast(const std::string& s) {
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

void write_fast_result(const std::filesystem::path& dir,
                       const GmFastTarget& t,
                       std::uint64_t B1,
                       std::uint64_t curves,
                       const std::optional<std::uint64_t>& curve,
                       const std::optional<std::uint64_t>& sigma,
                       const std::optional<mpz_class>& factor,
                       const std::string& backend,
                       int device,
                       double elapsed) {
    std::filesystem::create_directories(dir);
    const std::string prefix = t.family == "GQ" ? "gq" : "gm";
    const std::filesystem::path result = dir / (prefix + "_ecm_p" + std::to_string(t.p) + "_result.json");
    std::ostringstream j;
    j << std::fixed << std::setprecision(3);
    j << "{\n"
      << "  \"schema_version\": 2,\n"
      << "  \"program\": \"PrMers\",\n"
      << "  \"program_version\": \"" << GM_ECM_FAST_RELEASE << "\",\n"
      << "  \"program_build\": \"" << json_escape_fast(core::PRMERS_VERSION) << "\",\n"
      << "  \"family\": \"gaussian-pair\",\n"
      << "  \"target_family\": \"" << t.family << "\",\n"
      << "  \"mode\": \"gm-ecm\",\n"
      << "  \"engine\": \"twisted-edwards-naf\",\n"
      << "  \"outcome\": \"" << (factor ? "factor" : "no-factor") << "\",\n"
      << "  \"stage\": 1,\n"
      << "  \"exponent\": " << t.p << ",\n"
      << "  \"B1\": \"" << B1 << "\",\n"
      << "  \"B2\": null,\n"
      << "  \"curves\": " << curves << ",\n"
      << "  \"curve\": " << (curve ? std::to_string(*curve) : "null") << ",\n"
      << "  \"sigma\": " << (sigma ? ("\"" + std::to_string(*sigma) + "\"") : "null") << ",\n"
      << "  \"factor\": " << (factor ? ("\"" + factor->get_str() + "\"") : "null") << ",\n"
      << "  \"backend\": \"" << json_escape_fast(backend) << "\",\n"
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

} // namespace

namespace core {

// v99.97 wrapper.  The v99.96 implementation is compiled unchanged under the
// runGaussianMersenneECMLegacy() symbol.  Without -edwards, behavior is exactly
// legacy.  -edwards enables the Stage-1 twisted-Edwards/NAF accelerator while
// preserving the same Suyama sigma/curve stream.
int App::runGaussianMersenneECM() {
    if (!options.edwards) return runGaussianMersenneECMLegacy();

    // v99.97 deliberately accelerates Stage 1 first.  Stage 2 and optional
    // torsion families stay on the proven legacy path until their independent
    // regression suite is complete.
    if (options.B2 > options.B1 && options.B2 != 0) {
        std::cout << "[GM ECM v99.97] -edwards accelerator is Stage1-only; "
                     "falling back to legacy for B2>B1.\n";
        return runGaussianMersenneECMLegacy();
    }
    if (!options.notorsion || options.torsion16) {
        std::cout << "[GM ECM v99.97] GM torsion acceleration is not enabled in this release; "
                     "falling back to legacy Suyama ladder.\n";
        return runGaussianMersenneECMLegacy();
    }

    std::string family = options.gm_family;
    std::transform(family.begin(), family.end(), family.begin(),
                   [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
    if (family == "BOTH") {
        const std::string saved = options.gm_family;
        options.gm_family = "GM";
        const int gm_rc = runGaussianMersenneECM();
        if (interrupted) { options.gm_family = saved; return gm_rc; }
        options.gm_family = "GQ";
        const int gq_rc = runGaussianMersenneECM();
        options.gm_family = saved;
        if (gm_rc == 2 || gq_rc == 2) return 2;
        return (gm_rc == 0 && gq_rc == 0) ? 0 : 1;
    }

    GmFastTarget t;
    try { t = make_fast_target(options.exponent, family); }
    catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 2; }

    const std::uint64_t B1 = options.B1 != 0 ? options.B1 : 50000ULL;
    const std::uint64_t curves = options.K != 0 ? options.K : (options.nmax != 0 ? options.nmax : 20ULL);
    const std::filesystem::path save_dir = options.save_path.empty() ? "." : options.save_path;
    std::filesystem::create_directories(save_dir);
    const auto job_start = Clock::now();
    auto job_elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - job_start).count(); };

    std::cout << "Gaussian pair ECM factoring\n"
              << "  p              : " << t.p << "\n"
              << "  target family  : " << t.family << "\n"
              << "  lift exponent  : " << t.lift << "\n"
              << "  B1 / B2        : " << B1 << " / " << options.B2 << "\n"
              << "  curves         : " << curves << "\n"
              << "  Stage1 engine  : twisted Edwards + signed NAF (v99.97 opt-in)\n"
              << "  curve stream   : exact legacy Suyama sigma sequence\n"
              << "  legacy fallback: omit -edwards\n";

    const mpz_class K = buildE(B1);
    const std::vector<short> naf = naf_digits(K);
    if (naf.empty()) {
        std::cerr << "Invalid empty Stage1 exponent.\n";
        return 2;
    }
    const std::uint64_t total_steps = naf.size() - 1;
    const std::uint64_t additions = static_cast<std::uint64_t>(
        std::count_if(naf.begin(), naf.end() - 1, [](short d){ return d != 0; }));
    std::cout << "  K bits         : " << mpz_sizeinbase(K.get_mpz_t(), 2) << "\n"
              << "  NAF steps      : " << total_steps << "\n"
              << "  signed adds    : " << additions << "\n";

    std::unique_ptr<engine> eng(engine::create_gpu(t.lift, TeRegs::count,
                                                    static_cast<std::size_t>(options.device_id), true));
    if (!eng) {
        std::cerr << "[GM ECM NAF] GPU engine unavailable.\n";
        return 2;
    }
    const std::string backend = eng->is_aevum_backend() ? "Aevum" : "Marin";
    std::cout << "  backend        : " << backend << "\n"
              << "  transform      : " << eng->get_size() << " words\n"
              << "  registers      : " << TeRegs::count << "\n";

    const std::uint64_t base_seed = options.curve_seed != 0
        ? options.curve_seed : 0x474d45434d763938ULL;

    for (std::uint64_t curve = 0; curve < curves; ++curve) {
        std::uint64_t sigma = 0;
        if (!options.sigma.empty() && curve == 0) {
            try { sigma = std::stoull(options.sigma); }
            catch (...) {
                std::cerr << "Invalid -sigma for Gaussian ECM; expected unsigned 64-bit integer.\n";
                return 2;
            }
        } else {
            sigma = 6 + (splitmix64_fast(base_seed + curve) % 0x7ffffffffffffff0ULL);
        }

        std::cout << "\n[GM ECM] curve " << (curve + 1) << "/" << curves
                  << " sigma=" << sigma << "\n";

        TeCurveSetup setup = make_suyama_twisted_edwards(t.n, sigma);
        if (proper_factor_fast(setup.factor, t.n)) {
            std::cout << ">>> Gaussian pair ECM setup factor: " << setup.factor << "\n";
            write_fast_result(save_dir, t, B1, curves, curve + 1, sigma,
                              setup.factor, backend, options.device_id, job_elapsed());
            return 0;
        }
        if (!setup.ok) {
            std::cout << "[GM ECM NAF] singular/noninvertible curve setup; next curve.\n";
            continue;
        }

        te_init(eng.get(), setup, t.n);
        te_set_from_top_digit(eng.get(), naf.back());

        const std::filesystem::path ckpt = save_dir /
            ((t.family == "GQ" ? "gq" : "gm") + std::string("_ecm_p") +
             std::to_string(t.p) + "_c" + std::to_string(curve) + "_stage1_naf.ckpt");

        std::uint64_t done = 0;
        double restored = 0.0;
        if (options.resume) {
            if (!load_naf_checkpoint(ckpt, eng.get(), t, B1,
                                     static_cast<std::uint32_t>(curve), sigma,
                                     total_steps, done, restored)) {
                load_naf_checkpoint(ckpt.string() + ".old", eng.get(), t, B1,
                                    static_cast<std::uint32_t>(curve), sigma,
                                    total_steps, done, restored);
            }
            if (done != 0) {
                std::cout << "[GM ECM NAF] resuming curve " << (curve + 1)
                          << " at step " << done << "/" << total_steps << "\n";
            }
        }

        const auto curve_start = Clock::now();
        auto elapsed = [&]() {
            return restored + std::chrono::duration<double>(Clock::now() - curve_start).count();
        };
        const int backup_period = options.backup_interval > 0 ? options.backup_interval : 120;
        auto last_display = Clock::now();
        auto last_backup = Clock::now();

        std::vector<char> replay_start;
        const std::uint64_t replay_done_start = done;
        if (options.gm_safe_replay) {
            eng->sync();
            replay_start.resize(eng->get_checkpoint_size());
            if (!eng->get_checkpoint(replay_start))
                throw std::runtime_error("cannot capture GM ECM NAF replay start");
        }

        auto run_segment = [&](std::uint64_t& step,
                               bool allow_checkpoint,
                               bool show_progress) -> bool {
            while (step < total_steps) {
                if (interrupted) {
                    if (allow_checkpoint) {
                        save_naf_checkpoint(ckpt, eng.get(), t, B1,
                                            static_cast<std::uint32_t>(curve), sigma,
                                            total_steps, step, elapsed());
                        std::cout << "Interrupted; GM ECM NAF checkpoint saved at "
                                  << step << "/" << total_steps << ".\n";
                    }
                    return false;
                }

                te_double(eng.get());
                const short digit = naf[naf.size() - 2 - step];
                if (digit > 0) te_add_pos(eng.get());
                else if (digit < 0) te_add_neg(eng.get());
                ++step;

                const auto now = Clock::now();
                if (show_progress && (now - last_display >= std::chrono::seconds(10) || step == total_steps)) {
                    eng->sync();
                    const double pct = total_steps == 0 ? 100.0
                        : 100.0 * static_cast<double>(step) / static_cast<double>(total_steps);
                    std::cout << std::fixed << std::setprecision(2)
                              << "GM ECM Stage 1 NAF curve " << (curve + 1) << ": "
                              << pct << "% | " << step << "/" << total_steps
                              << " NAF steps | elapsed " << elapsed() << " s\n";
                    last_display = now;
                }
                if (allow_checkpoint && now - last_backup >= std::chrono::seconds(backup_period)) {
                    save_naf_checkpoint(ckpt, eng.get(), t, B1,
                                        static_cast<std::uint32_t>(curve), sigma,
                                        total_steps, step, elapsed());
                    last_backup = now;
                }
            }
            return true;
        };

        if (!run_segment(done, true, true)) return 0;

        if (options.gm_safe_replay) {
            eng->sync();
            std::vector<char> first_x, first_y, first_z, first_t;
            if (!eng->get_data(first_x, TeRegs::RX) || !eng->get_data(first_y, TeRegs::RY) ||
                !eng->get_data(first_z, TeRegs::RZ) || !eng->get_data(first_t, TeRegs::RT) ||
                !eng->set_checkpoint(replay_start))
                throw std::runtime_error("cannot prepare GM ECM NAF safe replay");
            te_restore_multiplicands(eng.get());
            std::uint64_t replay_done = replay_done_start;
            if (!run_segment(replay_done, false, false)) return 0;
            eng->sync();
            std::vector<char> second_x, second_y, second_z, second_t;
            if (!eng->get_data(second_x, TeRegs::RX) || !eng->get_data(second_y, TeRegs::RY) ||
                !eng->get_data(second_z, TeRegs::RZ) || !eng->get_data(second_t, TeRegs::RT) ||
                first_x != second_x || first_y != second_y ||
                first_z != second_z || first_t != second_t)
                throw std::runtime_error("Gaussian ECM NAF safe replay mismatch");
            std::cout << "[GM ECM NAF safe replay] Stage 1 coordinates verified.\n";
        }

        clear_naf_checkpoint(ckpt);
        eng->sync();

        const mpz_class Tfin = project_reg_fast(eng.get(), TeRegs::RT, t.n);
        mpz_class g = gcd_fast(Tfin, t.n);
        if (proper_factor_fast(g, t.n)) {
            std::cout << ">>> Gaussian pair ECM Stage 1 factor: " << g << "\n";
            write_fast_result(save_dir, t, B1, curves, curve + 1, sigma,
                              g, backend, options.device_id, job_elapsed());
            return 0;
        }
        if (g == t.n) {
            std::cout << "[GM ECM NAF] Stage 1 gcd=target; curve is fully killed/trivial.\n";
            continue;
        }

        // Convert the Edwards result back to Montgomery x for a comparable
        // residue fingerprint and to catch a denominator factor if present.
        const mpz_class Zv = project_reg_fast(eng.get(), TeRegs::RZ, t.n);
        const mpz_class Yv = project_reg_fast(eng.get(), TeRegs::RY, t.n);
        const mpz_class den = mod_positive_fast(Zv - Yv, t.n);
        mpz_class inv_den, conversion_factor;
        mpz_class x_mont = 0;
        if (invert_or_factor(den, t.n, inv_den, conversion_factor)) {
            x_mont = mod_positive_fast((Zv + Yv) * inv_den, t.n);
        } else if (proper_factor_fast(conversion_factor, t.n)) {
            std::cout << ">>> Gaussian pair ECM Stage 1 factor: " << conversion_factor << "\n";
            write_fast_result(save_dir, t, B1, curves, curve + 1, sigma,
                              conversion_factor, backend, options.device_id, job_elapsed());
            return 0;
        }

        std::cout << "[GM ECM] Stage 1 no factor | x low64=0x"
                  << low_hex_fast(x_mont) << " | elapsed="
                  << std::fixed << std::setprecision(2) << elapsed() << " s\n";
    }

    std::cout << "No Gaussian pair ECM factor found in " << curves << " curve(s).\n";
    write_fast_result(save_dir, t, B1, curves, std::nullopt, std::nullopt,
                      std::nullopt, backend, options.device_id, job_elapsed());
    return 1;
}

} // namespace core
