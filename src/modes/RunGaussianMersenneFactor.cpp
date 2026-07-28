#include "core/App.hpp"
#include "core/AlgoUtils.hpp"
#include "core/Version.hpp"
#include "marin/engine.h"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"

#include <gmpxx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using core::algo::buildE;
using core::algo::interrupted;

constexpr std::array<char, 8> GMF_MAGIC{{'P','R','G','M','F','A','C','T'}};
constexpr std::uint32_t GMF_VERSION = 3;
constexpr const char* GM_RELEASE = "v99.89";

struct GmTarget {
    std::uint32_t p = 0;
    std::uint32_t lift = 0;
    std::uint64_t middle = 0;
    int chi = 0;
    mpz_class n;
    std::uint64_t digits = 0;
};

std::uint64_t mul_mod_u64(std::uint64_t a, std::uint64_t b, std::uint64_t mod) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % mod);
}

std::uint64_t pow_mod_u64(std::uint64_t a, std::uint64_t e, std::uint64_t mod) {
    std::uint64_t r = 1 % mod;
    while (e != 0) {
        if ((e & 1U) != 0) r = mul_mod_u64(r, a, mod);
        e >>= 1U;
        if (e != 0) a = mul_mod_u64(a, a, mod);
    }
    return r;
}

bool is_prime_u64(std::uint64_t n) {
    if (n < 2) return false;
    for (const std::uint32_t p : {2U,3U,5U,7U,11U,13U,17U,19U,23U,29U,31U,37U}) {
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
        std::uint64_t x = pow_mod_u64(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

GmTarget make_target(std::uint64_t p64) {
    if (p64 < 3 || (p64 & 1ULL) == 0 || !is_prime_u64(p64)) {
        throw std::runtime_error("Gaussian-Mersenne factoring requires an odd prime exponent p >= 3");
    }
    if (p64 > std::numeric_limits<std::uint32_t>::max() / 4ULL) {
        throw std::runtime_error("Gaussian-Mersenne lift requires 4p <= 2^32-1");
    }
    GmTarget t;
    t.p = static_cast<std::uint32_t>(p64);
    t.lift = static_cast<std::uint32_t>(4ULL * p64);
    t.middle = (p64 + 1) / 2;
    const std::uint64_t r = p64 & 7ULL;
    t.chi = (r == 1 || r == 7) ? 1 : -1;
    t.n = mpz_class(1) << p64;
    const mpz_class mid = mpz_class(1) << t.middle;
    if (t.chi > 0) t.n -= mid;
    else t.n += mid;
    t.n += 1;
    t.digits = static_cast<std::uint64_t>(std::floor(p64 * std::log10(2.0))) + 1;
    return t;
}


std::uint64_t find_admissible_small_factor(const GmTarget& t, std::uint64_t limit) {
    if (limit < 5) return 0;
    if (mpz_divisible_ui_p(t.n.get_mpz_t(), 5) && mpz_cmp_ui(t.n.get_mpz_t(), 5) != 0) return 5;
    const unsigned __int128 step128 = static_cast<unsigned __int128>(4) * t.p;
    if (step128 > limit) return 0;
    const std::uint64_t step = static_cast<std::uint64_t>(step128);
    const std::uint64_t max_k = (limit - 1) / step;
    if (max_k > 100'000'000ULL) {
        throw std::runtime_error("-gm-sieve would inspect more than 100000000 admissible factors; lower the bound");
    }
    for (std::uint64_t k = 1; k <= max_k; ++k) {
        const unsigned __int128 q128 = static_cast<unsigned __int128>(step) * k + 1;
        if (q128 > limit || q128 > std::numeric_limits<std::uint64_t>::max()) break;
        const std::uint64_t q = static_cast<std::uint64_t>(q128);
        if (!is_prime_u64(q)) continue;
        const std::uint64_t a = pow_mod_u64(2, t.p, q);
        const std::uint64_t b = pow_mod_u64(2, t.middle, q);
        const std::uint64_t signed_b = t.chi > 0 ? b : (b == 0 ? 0 : q - b);
        if ((a + q - signed_b + 1) % q == 0 && mpz_cmp_ui(t.n.get_mpz_t(), q) != 0) return q;
    }
    return 0;
}

mpz_class mod_positive(mpz_class x, const mpz_class& n) {
    x %= n;
    if (x < 0) x += n;
    return x;
}

mpz_class project_reg(engine* eng, engine::Reg reg, const mpz_class& n) {
    mpz_t z;
    mpz_init(z);
    eng->get_mpz(z, reg);
    mpz_class out(z);
    mpz_clear(z);
    return mod_positive(out, n);
}

void set_reg_mpz(engine* eng, engine::Reg reg, const mpz_class& value, const mpz_class& n) {
    mpz_class v = mod_positive(value, n);
    mpz_t z;
    mpz_init_set(z, v.get_mpz_t());
    eng->set_mpz(reg, z);
    mpz_clear(z);
}

mpz_class proper_gcd(const mpz_class& value, const mpz_class& n) {
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), value.get_mpz_t(), n.get_mpz_t());
    return g;
}

bool is_proper_factor(const mpz_class& g, const mpz_class& n) {
    return g > 1 && g < n;
}

std::string low_hex(const mpz_class& x, unsigned bits) {
    mpz_class low;
    mpz_fdiv_r_2exp(low.get_mpz_t(), x.get_mpz_t(), bits);
    std::string s = low.get_str(16);
    const std::size_t width = bits / 4;
    if (s.size() < width) s.insert(0, width - s.size(), '0');
    return s;
}

std::string format_hms(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) return "--:--:--";
    const std::uint64_t total = static_cast<std::uint64_t>(seconds + 0.5);
    const std::uint64_t hours = total / 3600;
    const std::uint64_t minutes = (total % 3600) / 60;
    const std::uint64_t secs = total % 60;
    std::ostringstream out;
    out << std::setfill('0') << std::setw(2) << hours << ':'
        << std::setw(2) << minutes << ':' << std::setw(2) << secs
        << std::setfill(' ');
    return out.str();
}

std::vector<std::uint32_t> simple_primes(std::uint64_t limit) {
    if (limit < 2) return {};
    if (limit > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("this Gaussian factoring build currently requires B2 <= 2^32-1");
    }
    const std::uint32_t n = static_cast<std::uint32_t>(limit);
    std::vector<bool> prime(static_cast<std::size_t>(n) + 1, true);
    prime[0] = false;
    if (n >= 1) prime[1] = false;
    for (std::uint32_t q = 2; static_cast<std::uint64_t>(q) * q <= n; ++q) {
        if (!prime[q]) continue;
        for (std::uint64_t m = static_cast<std::uint64_t>(q) * q; m <= n; m += q) {
            prime[static_cast<std::size_t>(m)] = false;
        }
    }
    std::vector<std::uint32_t> out;
    for (std::uint32_t q = 2; q <= n; ++q) if (prime[q]) out.push_back(q);
    return out;
}

std::vector<std::uint32_t> primes_in_range(std::uint64_t low, std::uint64_t high) {
    if (high <= low || high < 2) return {};
    if (high > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Gaussian Stage 2 currently requires B2 <= 2^32-1");
    }
    const std::uint64_t root = static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(high))) + 1;
    const std::vector<std::uint32_t> base = simple_primes(root);
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
            for (std::uint64_t m = first; m <= seg_high; m += q) {
                prime[static_cast<std::size_t>(m - seg_low)] = false;
            }
        }
        for (std::uint64_t n = seg_low; n <= seg_high; ++n) {
            if (prime[static_cast<std::size_t>(n - seg_low)]) out.push_back(static_cast<std::uint32_t>(n));
        }
        if (seg_high == high) break;
        seg_low = seg_high + 1;
    }
    return out;
}

mpz_class product_range(const std::vector<std::uint32_t>& primes, std::size_t first, std::size_t last) {
    mpz_class q = 1;
    for (std::size_t i = first; i < last; ++i) q *= primes[i];
    return q;
}

std::size_t choose_chunk_end(const std::vector<std::uint32_t>& primes,
                             std::size_t first,
                             std::uint64_t bit_limit) {
    if (first >= primes.size()) return first;
    mpz_class q = 1;
    std::size_t i = first;
    for (; i < primes.size(); ++i) {
        q *= primes[i];
        if (i + 1 > first && mpz_sizeinbase(q.get_mpz_t(), 2) >= bit_limit) {
            ++i;
            break;
        }
    }
    return std::min(i, primes.size());
}

void print_target(const char* label, const GmTarget& t) {
    std::cout << label << "\n"
              << "  p              : " << t.p << "\n"
              << "  G_p            : 2^" << t.p << (t.chi > 0 ? " - " : " + ")
              << "2^" << t.middle << " + 1\n"
              << "  decimal digits : " << t.digits << "\n"
              << "  exact lift     : G_p | 2^(2p)+1 | 2^(4p)-1\n"
              << "  lift exponent  : " << t.lift << "\n";
}

void write_json_result(const std::filesystem::path& dir,
                       const std::string& filename,
                       const std::string& json) {
    std::filesystem::create_directories(dir);
    {
        std::ofstream out(dir / filename);
        out << json << '\n';
    }
    {
        std::ofstream out(dir / "results.txt", std::ios::app);
        out << json << '\n';
    }
    std::cout << "Result file: " << (dir / filename) << "\n";
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char raw : value) {
        const unsigned char c = static_cast<unsigned char>(raw);
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string iso8601_utc_now() {
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

std::string opencl_device_name(std::size_t requested) {
    try {
        cl_uint platform_count = 0;
        if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0)
            return "unknown";
        std::vector<cl_platform_id> platforms(platform_count);
        if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS)
            return "unknown";

        auto enumerate = [&](cl_device_type type) {
            std::vector<cl_device_id> devices;
            for (cl_platform_id platform_id : platforms) {
                cl_uint count = 0;
                const cl_int rc = clGetDeviceIDs(platform_id, type, 0, nullptr, &count);
                if (rc != CL_SUCCESS || count == 0) continue;
                const std::size_t old = devices.size();
                devices.resize(old + count);
                if (clGetDeviceIDs(platform_id, type, count, devices.data() + old, nullptr) != CL_SUCCESS)
                    devices.resize(old);
            }
            return devices;
        };

        std::vector<cl_device_id> devices = enumerate(CL_DEVICE_TYPE_GPU);
        if (devices.empty()) devices = enumerate(CL_DEVICE_TYPE_ALL);
        if (requested >= devices.size()) return "device " + std::to_string(requested);

        std::size_t bytes = 0;
        if (clGetDeviceInfo(devices[requested], CL_DEVICE_NAME, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0)
            return "device " + std::to_string(requested);
        std::string name(bytes, '\0');
        if (clGetDeviceInfo(devices[requested], CL_DEVICE_NAME, bytes, name.data(), nullptr) != CL_SUCCESS)
            return "device " + std::to_string(requested);
        if (!name.empty() && name.back() == '\0') name.pop_back();
        return name.empty() ? ("device " + std::to_string(requested)) : name;
    } catch (...) {
        return "device " + std::to_string(requested);
    }
}

std::string nullable_json_string(const std::optional<std::string>& value) {
    if (!value) return "null";
    return "\"" + json_escape(*value) + "\"";
}

std::string nullable_json_u64(const std::optional<std::uint64_t>& value) {
    return value ? std::to_string(*value) : "null";
}

std::string gm_result_json(const std::string& mode,
                           const std::string& outcome,
                           int stage,
                           const GmTarget& target,
                           const std::optional<std::uint64_t>& B1,
                           const std::optional<std::uint64_t>& B2,
                           const std::optional<std::uint64_t>& curves,
                           const std::optional<std::uint64_t>& curve,
                           const std::optional<std::string>& sigma,
                           const std::optional<std::string>& factor,
                           const std::string& backend,
                           const std::string& device,
                           double elapsed_seconds,
                           const std::optional<std::string>& source = std::nullopt) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"program\": \"PrMers\",\n"
         << "  \"program_version\": \"" << GM_RELEASE << "\",\n"
         << "  \"program_build\": \"" << json_escape(core::PRMERS_VERSION) << "\",\n"
         << "  \"family\": \"gaussian-mersenne\",\n"
         << "  \"mode\": \"" << json_escape(mode) << "\",\n"
         << "  \"outcome\": \"" << json_escape(outcome) << "\",\n"
         << "  \"stage\": " << stage << ",\n"
         << "  \"exponent\": " << target.p << ",\n"
         << "  \"B1\": "
         << (B1 ? nullable_json_string(std::to_string(*B1)) : "null") << ",\n"
         << "  \"B2\": "
         << (B2 ? nullable_json_string(std::to_string(*B2)) : "null") << ",\n"
         << "  \"curves\": " << nullable_json_u64(curves) << ",\n"
         << "  \"curve\": " << nullable_json_u64(curve) << ",\n"
         << "  \"sigma\": " << nullable_json_string(sigma) << ",\n"
         << "  \"factor\": " << nullable_json_string(factor) << ",\n"
         << "  \"factor_source\": " << nullable_json_string(source) << ",\n"
         << "  \"backend\": \"" << json_escape(backend) << "\",\n"
         << "  \"device\": \"" << json_escape(device) << "\",\n"
         << "  \"elapsed_seconds\": " << std::max(0.0, elapsed_seconds) << ",\n"
         << "  \"timestamp\": \"" << iso8601_utc_now() << "\"\n"
         << "}";
    return json.str();
}

struct FactorCheckpointHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t mode;        // 1=P-1, 2=ECM
    std::uint32_t phase;       // 1=stage1, 2=stage2
    std::uint32_t p;
    std::uint32_t lift;
    std::uint32_t base;
    std::uint32_t curve;
    std::uint64_t B1;
    std::uint64_t B2;
    std::uint64_t token;       // remaining bits or next prime index
    std::uint64_t scalar_bits;
    std::uint64_t sigma;
    double elapsed;
    std::uint64_t checkpoint_bytes;
};

bool matching_header(const FactorCheckpointHeader& h,
                     std::uint32_t mode,
                     std::uint32_t phase,
                     const GmTarget& t,
                     std::uint64_t B1,
                     std::uint64_t B2,
                     std::uint64_t scalar_bits,
                     std::uint32_t base,
                     std::uint32_t curve,
                     std::uint64_t sigma,
                     std::size_t bytes) {
    return std::equal(GMF_MAGIC.begin(), GMF_MAGIC.end(), h.magic) &&
           h.version >= 2 && h.version <= GMF_VERSION &&
           h.mode == mode && h.phase == phase &&
           h.p == t.p && h.lift == t.lift && h.B1 == B1 && h.B2 == B2 &&
           h.scalar_bits == scalar_bits && h.base == base && h.curve == curve &&
           h.sigma == sigma && h.checkpoint_bytes == bytes;
}

bool load_factor_checkpoint(const std::filesystem::path& path,
                            engine* eng,
                            std::uint32_t mode,
                            std::uint32_t phase,
                            const GmTarget& t,
                            std::uint64_t B1,
                            std::uint64_t B2,
                            std::uint64_t scalar_bits,
                            std::uint32_t base,
                            std::uint32_t curve,
                            std::uint64_t sigma,
                            std::uint64_t& token,
                            double& elapsed) {
    File f(path.string());
    if (!f.exists()) return false;
    FactorCheckpointHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (!matching_header(h, mode, phase, t, B1, B2, scalar_bits, base, curve, sigma,
                         eng->get_checkpoint_size())) return false;
    std::vector<char> data(eng->get_checkpoint_size());
    if (!f.read(data.data(), data.size()) || !f.check_crc32() || !eng->set_checkpoint(data)) return false;
    token = h.token;
    elapsed = h.elapsed;
    if (h.version != GMF_VERSION) {
        std::cout << "Loaded legacy Gaussian factoring checkpoint v" << h.version
                  << "; it will be upgraded to v" << GMF_VERSION << " on the next save.\n";
    }
    return true;
}

void save_factor_checkpoint(const std::filesystem::path& path,
                            engine* eng,
                            std::uint32_t mode,
                            std::uint32_t phase,
                            const GmTarget& t,
                            std::uint64_t B1,
                            std::uint64_t B2,
                            std::uint64_t scalar_bits,
                            std::uint32_t base,
                            std::uint32_t curve,
                            std::uint64_t sigma,
                            std::uint64_t token,
                            double elapsed) {
    eng->sync();
    FactorCheckpointHeader h{};
    std::copy(GMF_MAGIC.begin(), GMF_MAGIC.end(), h.magic);
    h.version = GMF_VERSION;
    h.mode = mode;
    h.phase = phase;
    h.p = t.p;
    h.lift = t.lift;
    h.base = base;
    h.curve = curve;
    h.B1 = B1;
    h.B2 = B2;
    h.token = token;
    h.scalar_bits = scalar_bits;
    h.sigma = sigma;
    h.elapsed = elapsed;
    h.checkpoint_bytes = eng->get_checkpoint_size();

    std::vector<char> data(eng->get_checkpoint_size());
    if (!eng->get_checkpoint(data)) throw std::runtime_error("cannot read Gaussian factoring checkpoint");

    const std::filesystem::path new_path = path.string() + ".new";
    const std::filesystem::path old_path = path.string() + ".old";
    {
        File f(new_path.string(), "wb");
        if (!f.write(reinterpret_cast<const char*>(&h), sizeof(h)) ||
            !f.write(data.data(), data.size())) {
            throw std::runtime_error("cannot write Gaussian factoring checkpoint");
        }
        f.write_crc32();
    }
    std::error_code ec;
    std::filesystem::remove(old_path, ec);
    ec.clear();
    if (std::filesystem::exists(path)) std::filesystem::rename(path, old_path, ec);
    ec.clear();
    std::filesystem::rename(new_path, path, ec);
    if (ec) throw std::runtime_error("cannot install Gaussian factoring checkpoint: " + ec.message());
}

void clear_checkpoint(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".old", ec);
    std::filesystem::remove(path.string() + ".new", ec);
}

// Compute dst <- base^exponent in the lifted Mersenne ring.  The base is a
// small integer and Aevum/Marin can fuse the multiply into square_mul.
bool pow_small_base(engine* eng,
                    engine::Reg dst,
                    const mpz_class& exponent,
                    std::uint32_t base,
                    bool safe_replay,
                    engine::Reg start_reg,
                    engine::Reg verify_reg,
                    std::uint64_t replay_block,
                    std::uint64_t& remaining,
                    const std::function<void(std::uint64_t)>& checkpoint,
                    const std::function<double()>& elapsed,
                    const std::string& progress_label) {
    const std::uint64_t total = static_cast<std::uint64_t>(mpz_sizeinbase(exponent.get_mpz_t(), 2));
    if (remaining == 0 || remaining > total) {
        eng->set(dst, 1);
        remaining = total;
    }
    auto apply = [&](engine::Reg reg, std::uint64_t bit_index) {
        if (mpz_tstbit(exponent.get_mpz_t(), bit_index)) eng->square_mul(reg, base);
        else eng->square_mul(reg);
    };

    auto last_display = Clock::now();
    auto last_backup = Clock::now();
    while (remaining > 0) {
        if (interrupted) {
            checkpoint(remaining);
            std::cout << "Interrupted; checkpoint saved with " << remaining << " bits remaining.\n";
            return false;
        }
        const std::uint64_t block = safe_replay ? std::min(remaining, replay_block) : 1ULL;
        if (safe_replay) eng->copy(start_reg, dst);
        for (std::uint64_t j = 0; j < block; ++j) apply(dst, remaining - 1 - j);
        if (safe_replay) {
            eng->copy(verify_reg, start_reg);
            for (std::uint64_t j = 0; j < block; ++j) apply(verify_reg, remaining - 1 - j);
            if (!eng->is_equal(dst, verify_reg)) {
                std::cout << "[GM factoring safe replay] mismatch; restoring block.\n";
                eng->copy(dst, start_reg);
                continue;
            }
        }
        remaining -= block;

        const auto now = Clock::now();
        if (now - last_display >= std::chrono::seconds(10) || remaining == 0) {
            eng->sync();
            const double pct = total == 0 ? 100.0 : 100.0 * static_cast<double>(total - remaining) / total;
            std::cout << std::fixed << std::setprecision(2)
                      << progress_label << ": " << pct << "% | " << (total - remaining)
                      << "/" << total << " bits | elapsed " << elapsed() << " s\n";
            last_display = now;
        }
        if (now - last_backup >= std::chrono::seconds(60)) {
            checkpoint(remaining);
            last_backup = now;
        }
    }
    return true;
}

struct Pm1WindowRegs {
    engine::Reg base = 3;
    engine::Reg base2 = 4;
    engine::Reg mbase2 = 5;
    engine::Reg odd = 6;
    std::array<engine::Reg, 8> modd{{7, 8, 9, 10, 11, 12, 13, 14}};
    static constexpr unsigned width = 4;
    static constexpr std::size_t count = 15;
};

// Left-to-right sliding-window exponentiation.  Stage-2 exponents are dense
// products of primes, so width 4 reduces the expected multiply count from
// about one per two bits to one per five bits while retaining the existing
// Aevum prepared-multiplicand kernels.
bool pow_window_base(engine* eng,
                     engine::Reg dst,
                     engine::Reg source,
                     const mpz_class& exponent,
                     const Pm1WindowRegs& r,
                     const std::function<void(std::uint64_t, std::uint64_t, double)>& progress) {
    eng->copy(r.base, source);
    eng->copy(r.base2, r.base);
    eng->square_mul(r.base2);
    eng->set_multiplicand(r.mbase2, r.base2);
    eng->copy(r.odd, r.base);
    for (std::size_t j = 0; j < r.modd.size(); ++j) {
        eng->set_multiplicand(r.modd[j], r.odd);
        if (j + 1 < r.modd.size()) eng->mul(r.odd, r.mbase2);
    }

    eng->set(dst, 1);
    const std::uint64_t total_bits =
        static_cast<std::uint64_t>(mpz_sizeinbase(exponent.get_mpz_t(), 2));
    std::int64_t i = static_cast<std::int64_t>(total_bits) - 1;
    std::uint64_t last_sync_done = 0;
    auto started = Clock::now();
    auto last_report = started;
    constexpr std::uint64_t sync_interval_bits = 4096;

    auto maybe_report = [&](bool force) -> bool {
        const std::uint64_t remaining = i >= 0 ? static_cast<std::uint64_t>(i + 1) : 0;
        const std::uint64_t done = total_bits - remaining;
        if (!force && done - last_sync_done < sync_interval_bits) return true;
        eng->sync();
        last_sync_done = done;
        const auto now = Clock::now();
        const double seconds = std::chrono::duration<double>(now - started).count();
        if (force || now - last_report >= std::chrono::seconds(5)) {
            progress(done, total_bits, seconds > 0.0 ? done / seconds : 0.0);
            last_report = now;
        }
        return !interrupted;
    };

    while (i >= 0) {
        if (mpz_tstbit(exponent.get_mpz_t(), static_cast<mp_bitcnt_t>(i)) == 0) {
            eng->square_mul(dst);
            --i;
            if (!maybe_report(false)) return false;
            continue;
        }
        std::int64_t low = std::max<std::int64_t>(0, i - static_cast<std::int64_t>(Pm1WindowRegs::width) + 1);
        while (low < i && mpz_tstbit(exponent.get_mpz_t(), static_cast<mp_bitcnt_t>(low)) == 0) ++low;
        unsigned value = 0;
        for (std::int64_t bit = i; bit >= low; --bit) {
            value = (value << 1U) |
                    static_cast<unsigned>(mpz_tstbit(exponent.get_mpz_t(), static_cast<mp_bitcnt_t>(bit)) != 0);
        }
        for (std::int64_t bit = low; bit <= i; ++bit) eng->square_mul(dst);
        eng->mul(dst, r.modd[(value - 1U) / 2U]);
        i = low - 1;
        if (!maybe_report(false)) return false;
    }
    return maybe_report(true);
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

struct CurveSetup {
    mpz_class x_affine;
    mpz_class a24;
    mpz_class factor;
    bool ok = false;
};

CurveSetup make_suyama_curve(const mpz_class& n, std::uint64_t sigma64) {
    CurveSetup out;
    mpz_class sigma = sigma64;
    sigma %= n;
    if (sigma < 6) sigma += 6;
    mpz_class u = mod_positive(sigma * sigma - 5, n);
    mpz_class v = mod_positive(4 * sigma, n);
    mpz_class x = mod_positive(u * u * u, n);
    mpz_class z = mod_positive(v * v * v, n);

    mpz_class g = proper_gcd(z, n);
    if (is_proper_factor(g, n)) { out.factor = g; return out; }
    mpz_class invz;
    if (mpz_invert(invz.get_mpz_t(), z.get_mpz_t(), n.get_mpz_t()) == 0) {
        out.factor = proper_gcd(z, n);
        return out;
    }
    out.x_affine = mod_positive(x * invz, n);

    const mpz_class vu = mod_positive(v - u, n);
    const mpz_class numerator = mod_positive(vu * vu * vu * (3 * u + v), n);
    const mpz_class denominator = mod_positive(16 * x * v, n);
    g = proper_gcd(denominator, n);
    if (is_proper_factor(g, n)) { out.factor = g; return out; }
    mpz_class invden;
    if (mpz_invert(invden.get_mpz_t(), denominator.get_mpz_t(), n.get_mpz_t()) == 0) {
        out.factor = proper_gcd(denominator, n);
        return out;
    }
    out.a24 = mod_positive(numerator * invden, n);
    out.ok = true;
    return out;
}

struct MontgomeryRegs {
    engine::Reg xa = 0, za = 1, xb = 2, zb = 3;
    engine::Reg a24 = 4, xdiff = 5, mxdiff = 6, ma24 = 7;
    engine::Reg A = 8, B = 9, C = 10, D = 11;
    engine::Reg mA = 12, mB = 13;
    engine::Reg DA = 14, CB = 15, sum = 16, diff = 17;
    engine::Reg AA = 18, BB = 19, mBB = 20;
    engine::Reg E = 21, tmp = 22, mTmp = 23;
    static constexpr std::size_t count = 24;
};

void montgomery_set_constants(engine* eng,
                              const MontgomeryRegs& r,
                              const mpz_class& xdiff,
                              const mpz_class& a24,
                              const mpz_class& n) {
    set_reg_mpz(eng, r.xdiff, xdiff, n);
    set_reg_mpz(eng, r.a24, a24, n);
    eng->set_multiplicand(r.mxdiff, r.xdiff);
    eng->set_multiplicand(r.ma24, r.a24);
}

void montgomery_double(engine* eng, const MontgomeryRegs& r,
                       engine::Reg x, engine::Reg z) {
    eng->addsub(r.A, r.B, x, z);
    eng->copy(r.AA, r.A);
    eng->square_mul(r.AA);
    eng->copy(r.BB, r.B);
    eng->square_mul(r.BB);
    eng->copy(x, r.AA);
    eng->set_multiplicand(r.mBB, r.BB);
    eng->mul(x, r.mBB);
    eng->copy(r.E, r.AA);
    eng->sub_reg(r.E, r.BB);
    eng->copy(r.tmp, r.E);
    eng->mul(r.tmp, r.ma24);
    eng->add(r.tmp, r.BB);
    eng->set_multiplicand(r.mTmp, r.tmp);
    eng->copy(z, r.E);
    eng->mul(z, r.mTmp);
}

// Simultaneously doubles (x2:z2) and replaces (x3:z3) by their sum.  The
// differential point is the fixed affine x-coordinate held in r.mxdiff.
void montgomery_dbladd(engine* eng, const MontgomeryRegs& r,
                       engine::Reg x2, engine::Reg z2,
                       engine::Reg x3, engine::Reg z3) {
    eng->addsub(r.A, r.B, x2, z2);
    eng->addsub(r.C, r.D, x3, z3);

    eng->set_multiplicand(r.mA, r.A);
    eng->copy(r.DA, r.D);
    eng->mul(r.DA, r.mA);
    eng->set_multiplicand(r.mB, r.B);
    eng->copy(r.CB, r.C);
    eng->mul(r.CB, r.mB);

    eng->addsub(r.sum, r.diff, r.DA, r.CB);
    eng->copy(x3, r.sum);
    eng->square_mul(x3);
    eng->copy(z3, r.diff);
    eng->square_mul(z3);
    eng->mul(z3, r.mxdiff);

    eng->copy(r.AA, r.A);
    eng->square_mul(r.AA);
    eng->copy(r.BB, r.B);
    eng->square_mul(r.BB);
    eng->copy(x2, r.AA);
    eng->set_multiplicand(r.mBB, r.BB);
    eng->mul(x2, r.mBB);
    eng->copy(r.E, r.AA);
    eng->sub_reg(r.E, r.BB);
    eng->copy(r.tmp, r.E);
    eng->mul(r.tmp, r.ma24);
    eng->add(r.tmp, r.BB);
    eng->set_multiplicand(r.mTmp, r.tmp);
    eng->copy(z2, r.E);
    eng->mul(z2, r.mTmp);
}

void montgomery_init_ladder(engine* eng,
                            const MontgomeryRegs& r,
                            const mpz_class& x,
                            const mpz_class& a24,
                            const mpz_class& n) {
    montgomery_set_constants(eng, r, x, a24, n);
    set_reg_mpz(eng, r.xa, x, n);
    eng->set(r.za, 1);
    eng->copy(r.xb, r.xa);
    eng->copy(r.zb, r.za);
    montgomery_double(eng, r, r.xb, r.zb);
}

// Scalar must be positive.  Registers contain P and 2P on entry.  The leading
// one bit is already represented by P, so processing starts at bit bits-2.
bool montgomery_ladder(engine* eng,
                       const MontgomeryRegs& r,
                       const mpz_class& scalar,
                       std::uint64_t& remaining,
                       const std::function<void(std::uint64_t)>& checkpoint,
                       const std::function<double()>& elapsed,
                       const std::string& label) {
    const std::uint64_t bits = static_cast<std::uint64_t>(mpz_sizeinbase(scalar.get_mpz_t(), 2));
    if (bits <= 1) { remaining = 0; return true; }
    const std::uint64_t work = bits - 1;
    if (remaining == 0 || remaining > work) remaining = work;
    auto last_display = Clock::now();
    auto last_backup = Clock::now();

    while (remaining > 0) {
        if (interrupted) {
            checkpoint(remaining);
            std::cout << "Interrupted; ECM checkpoint saved with " << remaining << " ladder bits remaining.\n";
            return false;
        }
        const std::uint64_t bit_index = remaining - 1;
        if (mpz_tstbit(scalar.get_mpz_t(), bit_index)) {
            montgomery_dbladd(eng, r, r.xb, r.zb, r.xa, r.za);
        } else {
            montgomery_dbladd(eng, r, r.xa, r.za, r.xb, r.zb);
        }
        --remaining;

        const auto now = Clock::now();
        if (now - last_display >= std::chrono::seconds(10) || remaining == 0) {
            eng->sync();
            const double pct = 100.0 * static_cast<double>(work - remaining) / work;
            std::cout << std::fixed << std::setprecision(2)
                      << label << ": " << pct << "% | " << (work - remaining)
                      << "/" << work << " ladder bits | elapsed " << elapsed() << " s\n";
            last_display = now;
        }
        if (now - last_backup >= std::chrono::seconds(60)) {
            checkpoint(remaining);
            last_backup = now;
        }
    }
    return true;
}

struct PointProjection {
    mpz_class x;
    mpz_class z;
    mpz_class factor;
    bool normalized = false;
};

PointProjection project_point(engine* eng,
                              const MontgomeryRegs& r,
                              const mpz_class& n) {
    PointProjection out;
    out.x = project_reg(eng, r.xa, n);
    out.z = project_reg(eng, r.za, n);
    mpz_class g = proper_gcd(out.z, n);
    if (is_proper_factor(g, n)) { out.factor = g; return out; }
    mpz_class inv;
    if (mpz_invert(inv.get_mpz_t(), out.z.get_mpz_t(), n.get_mpz_t()) == 0) {
        out.factor = proper_gcd(out.z, n);
        return out;
    }
    out.x = mod_positive(out.x * inv, n);
    out.z = 1;
    out.normalized = true;
    return out;
}

} // namespace

namespace core {

int App::runGaussianMersennePM1() {
    GmTarget t;
    try { t = make_target(options.exponent); }
    catch (const std::exception& ex) { std::cerr << ex.what() << "\n"; return 2; }

    const std::uint64_t B1 = options.B1 != 0 ? options.B1 : 100000ULL;
    const std::uint64_t B2 = options.B2;
    const std::uint32_t base = options.gm_base != 0 ? options.gm_base : 3U;
    const std::uint64_t chunk_bits = options.gm_factor_chunk_bits != 0
        ? options.gm_factor_chunk_bits : 262144ULL;
    const std::filesystem::path save_dir = options.save_path.empty() ? "." : options.save_path;
    std::filesystem::create_directories(save_dir);
    const auto job_clock = Clock::now();
    auto job_elapsed = [&]() {
        return std::chrono::duration<double>(Clock::now() - job_clock).count();
    };
    const std::string device_name = opencl_device_name(static_cast<std::size_t>(options.device_id));

    print_target("Gaussian-Mersenne P-1 factoring", t);
    if (options.gm_sieve_limit != 0) {
        std::cout << "  admissible sieve: q=4kp+1 through " << options.gm_sieve_limit << "\n";
        const std::uint64_t sf = find_admissible_small_factor(t, options.gm_sieve_limit);
        if (sf != 0) {
            std::cout << ">>> Gaussian-Mersenne admissible-sieve factor: " << sf << "\n";
            write_json_result(
                save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
                gm_result_json("gm-pm1", "factor", 0, t, B1,
                               B2 > B1 ? std::optional<std::uint64_t>(B2) : std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt,
                               std::to_string(sf), "CPU sieve", device_name, job_elapsed(),
                               std::string("q=4kp+1 sieve")));
            return 0;
        }
    }
    std::cout << "  B1 / B2        : " << B1 << " / " << B2 << "\n"
              << "  base           : " << base << "\n"
              << "  optimization   : exponent includes guaranteed factor 4p of every q-1 (q | G_p, q != 5)\n"
              << "  projection     : CPU reduction/GCD modulo G_p only at stage boundaries\n";

    mpz_class base_gcd;
    mpz_gcd_ui(base_gcd.get_mpz_t(), t.n.get_mpz_t(), base);
    if (is_proper_factor(base_gcd, t.n)) {
        std::cout << "P-1 base gcd found factor " << base_gcd << "\n";
        write_json_result(
            save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
            gm_result_json("gm-pm1", "factor", 0, t, B1,
                           B2 > B1 ? std::optional<std::uint64_t>(B2) : std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, base_gcd.get_str(),
                           "CPU gcd", device_name, job_elapsed(), std::string("base gcd")));
        return 0;
    }

    const mpz_class smooth = buildE(B1);
    const mpz_class structural = mpz_class(4) * t.p;
    mpz_class exponent;
    mpz_lcm(exponent.get_mpz_t(), smooth.get_mpz_t(), structural.get_mpz_t());
    const std::uint64_t exponent_bits = static_cast<std::uint64_t>(mpz_sizeinbase(exponent.get_mpz_t(), 2));
    std::cout << "  stage1 bits    : " << exponent_bits << "\n";

    constexpr engine::Reg RSTATE = 0;
    constexpr engine::Reg RSTART = 1;
    constexpr engine::Reg RVERIFY = 2;
    const Pm1WindowRegs window_regs;
    std::unique_ptr<engine> eng(engine::create_gpu(t.lift, Pm1WindowRegs::count,
                                                    static_cast<std::size_t>(options.device_id), true));
    const std::string backend = eng->is_aevum_backend() ? "Aevum" : "Marin";
    // Checkpoints contain every allocated register. Initializing the scratch
    // registers avoids harmless Aevum "Read ZERO" diagnostics during the
    // first periodic Stage-1 checkpoint.
    for (std::size_t reg = 0; reg < Pm1WindowRegs::count; ++reg) eng->set(reg, 1);
    std::cout << "  backend        : " << backend << "\n"
              << "  transform      : " << eng->get_size() << " words\n"
              << "  safe replay    : " << (options.gm_safe_replay ? "enabled" : "disabled") << "\n"
              << "  Stage 2 power  : width-4 sliding window with 8 prepared odd powers\n";

    const std::filesystem::path s1_ckpt = save_dir / ("gm_pm1_p" + std::to_string(t.p) + "_stage1.ckpt");
    std::uint64_t remaining = exponent_bits;
    double restored = 0.0;
    if (options.resume) {
        if (!load_factor_checkpoint(s1_ckpt, eng.get(), 1, 1, t, B1, B2, exponent_bits,
                                    base, 0, 0, remaining, restored)) {
            load_factor_checkpoint(s1_ckpt.string() + ".old", eng.get(), 1, 1, t, B1, B2,
                                   exponent_bits, base, 0, 0, remaining, restored);
        }
    }
    // Only RSTATE is semantically live at a P-1 checkpoint boundary. Reset
    // legacy v2 scratch registers so their former zero values cannot trigger
    // Aevum Read ZERO diagnostics on the next checkpoint export.
    for (std::size_t reg = 1; reg < Pm1WindowRegs::count; ++reg) eng->set(reg, 1);

    const auto start = Clock::now();
    auto elapsed = [&]() { return restored + std::chrono::duration<double>(Clock::now() - start).count(); };
    auto save_s1 = [&](std::uint64_t rem) {
        save_factor_checkpoint(s1_ckpt, eng.get(), 1, 1, t, B1, B2, exponent_bits,
                               base, 0, 0, rem, elapsed());
    };

    if (remaining == exponent_bits) eng->set(RSTATE, 1);
    const std::uint64_t replay = options.gm_replay_block != 0
        ? options.gm_replay_block : std::max<std::uint64_t>(64, static_cast<std::uint64_t>(std::sqrt(exponent_bits)));
    if (!pow_small_base(eng.get(), RSTATE, exponent, base, options.gm_safe_replay,
                        RSTART, RVERIFY, replay, remaining, save_s1, elapsed, "GM P-1 Stage 1")) {
        return 0;
    }
    clear_checkpoint(s1_ckpt);
    eng->sync();

    mpz_class h = project_reg(eng.get(), RSTATE, t.n);
    mpz_class g = proper_gcd(mod_positive(h - 1, t.n), t.n);
    std::cout << "Stage 1 residue low64: 0x" << low_hex(h, 64) << "\n";
    if (is_proper_factor(g, t.n)) {
        std::cout << ">>> Gaussian-Mersenne P-1 Stage 1 factor: " << g << "\n";
        write_json_result(
            save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
            gm_result_json("gm-pm1", "factor", 1, t, B1, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, g.get_str(),
                           backend, device_name, job_elapsed()));
        if (B2 <= B1 || !options.pm1_continue_stage2_after_factor) return 0;
        std::cout << "Continuing Stage 2 by explicit -pm1-continue-stage2-after-factor.\n";
    } else if (g == t.n) {
        std::cout << "Stage 1 gcd=G_p (all remaining factors were killed); retry with a smaller B1 or another base to isolate one.\n";
        if (B2 <= B1) return 1;
    } else {
        std::cout << "No Gaussian-Mersenne P-1 Stage 1 factor.\n";
    }

    if (B2 <= B1) {
        std::cout << "Stage 2 disabled (B2 <= B1).\n";
        write_json_result(
            save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
            gm_result_json("gm-pm1", "no-factor", 1, t, B1, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                           backend, device_name, job_elapsed()));
        return 1;
    }

    std::vector<std::uint32_t> s2primes = primes_in_range(B1, B2);
    s2primes.erase(std::remove(s2primes.begin(), s2primes.end(), t.p), s2primes.end());
    std::cout << "Stage 2 product-exponent primes: " << s2primes.size()
              << " | chunk target " << chunk_bits << " bits\n";
    if (s2primes.empty()) {
        write_json_result(
            save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
            gm_result_json("gm-pm1", "no-factor", 2, t, B1, B2,
                           std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                           backend, device_name, job_elapsed()));
        return 1;
    }

    const std::filesystem::path s2_ckpt = save_dir / ("gm_pm1_p" + std::to_string(t.p) + "_stage2.ckpt");
    std::size_t prime_index = 0;
    double s2_restored = 0.0;
    std::uint64_t token = 0;
    if (options.resume && load_factor_checkpoint(s2_ckpt, eng.get(), 1, 2, t, B1, B2,
                                                  s2primes.size(), base, 0, 0,
                                                  token, s2_restored)) {
        prime_index = static_cast<std::size_t>(std::min<std::uint64_t>(token, s2primes.size()));
        std::cout << "Resuming Stage 2 at prime index " << prime_index << "/" << s2primes.size() << "\n";
    }
    for (std::size_t reg = 1; reg < Pm1WindowRegs::count; ++reg) eng->set(reg, 1);
    const auto s2_start = Clock::now();
    auto s2_elapsed = [&]() { return s2_restored + std::chrono::duration<double>(Clock::now() - s2_start).count(); };

    std::size_t chunk_no = 0;
    while (prime_index < s2primes.size()) {
        if (interrupted) {
            save_factor_checkpoint(s2_ckpt, eng.get(), 1, 2, t, B1, B2, s2primes.size(),
                                   base, 0, 0, prime_index, s2_elapsed());
            std::cout << "Interrupted at a clean Stage 2 chunk boundary.\n";
            return 0;
        }
        const std::size_t end = choose_chunk_end(s2primes, prime_index, chunk_bits);
        const mpz_class qprod = product_range(s2primes, prime_index, end);
        ++chunk_no;
        std::cout << "[GM P-1 Stage 2] chunk " << chunk_no << " primes "
                  << s2primes[prime_index] << ".." << s2primes[end - 1]
                  << " | count=" << (end - prime_index)
                  << " | bits=" << mpz_sizeinbase(qprod.get_mpz_t(), 2) << "\n";

        const std::size_t chunk_begin = prime_index;
        const std::size_t chunk_prime_count = end - chunk_begin;
        eng->copy(RSTART, RSTATE);
        auto report_stage2 = [&](std::uint64_t done_bits, std::uint64_t total_bits, double bit_ips) {
            const double chunk_fraction = total_bits == 0 ? 1.0
                : static_cast<double>(done_bits) / static_cast<double>(total_bits);
            const double overall_done = static_cast<double>(chunk_begin) +
                static_cast<double>(chunk_prime_count) * chunk_fraction;
            const double overall_fraction = s2primes.empty() ? 1.0
                : overall_done / static_cast<double>(s2primes.size());
            const double spent = s2_elapsed();
            const double eta = overall_fraction > 0.0
                ? spent * (1.0 - overall_fraction) / overall_fraction
                : std::numeric_limits<double>::infinity();
            std::cout << std::fixed << std::setprecision(2)
                      << "GM P-1 Stage 2: " << (100.0 * overall_fraction) << "%"
                      << " | primes " << static_cast<std::uint64_t>(overall_done)
                      << "/" << s2primes.size()
                      << " | chunk " << chunk_no << " " << (100.0 * chunk_fraction) << "%"
                      << " | bits " << done_bits << "/" << total_bits
                      << " | bit-IPS " << bit_ips
                      << " | elapsed " << spent << " s"
                      << " | ETA " << format_hms(eta) << "\n";
        };
        if (!pow_window_base(eng.get(), RSTATE, RSTATE, qprod, window_regs, report_stage2)) {
            eng->copy(RSTATE, RSTART);
            save_factor_checkpoint(s2_ckpt, eng.get(), 1, 2, t, B1, B2, s2primes.size(),
                                   base, 0, 0, prime_index, s2_elapsed());
            std::cout << "Interrupted inside Stage 2 chunk; restored its start and saved a clean checkpoint.\n";
            return 0;
        }
        if (options.gm_safe_replay) {
            eng->copy(RVERIFY, RSTART);
            auto silent_progress = [](std::uint64_t, std::uint64_t, double) {};
            if (!pow_window_base(eng.get(), RVERIFY, RSTART, qprod, window_regs, silent_progress)) {
                eng->copy(RSTATE, RSTART);
                save_factor_checkpoint(s2_ckpt, eng.get(), 1, 2, t, B1, B2, s2primes.size(),
                                       base, 0, 0, prime_index, s2_elapsed());
                return 0;
            }
            if (!eng->is_equal(RSTATE, RVERIFY)) {
                std::cout << "[GM P-1 safe replay] Stage 2 mismatch; restoring chunk.\n";
                eng->copy(RSTATE, RSTART);
                continue;
            }
        }
        eng->sync();
        h = project_reg(eng.get(), RSTATE, t.n);
        g = proper_gcd(mod_positive(h - 1, t.n), t.n);
        prime_index = end;
        save_factor_checkpoint(s2_ckpt, eng.get(), 1, 2, t, B1, B2, s2primes.size(),
                               base, 0, 0, prime_index, s2_elapsed());
        std::cout << std::fixed << std::setprecision(2)
                  << "[GM P-1 Stage 2] " << (100.0 * prime_index / s2primes.size())
                  << "% | residue low64=0x" << low_hex(h, 64)
                  << " | elapsed=" << s2_elapsed() << " s\n";
        if (is_proper_factor(g, t.n)) {
            clear_checkpoint(s2_ckpt);
            std::cout << ">>> Gaussian-Mersenne P-1 Stage 2 factor: " << g << "\n";
            write_json_result(
                save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
                gm_result_json("gm-pm1", "factor", 2, t, B1, B2,
                               std::nullopt, std::nullopt, std::nullopt, g.get_str(),
                               backend, device_name, job_elapsed()));
            return 0;
        }
        if (g == t.n) {
            std::cout << "Stage 2 gcd=G_p; reduce -gm-factor-chunk-bits to isolate a factor.\n";
            clear_checkpoint(s2_ckpt);
            return 1;
        }
    }

    clear_checkpoint(s2_ckpt);
    std::cout << "No Gaussian-Mersenne P-1 factor through B2=" << B2 << ".\n";
    write_json_result(
        save_dir, "gm_pm1_p" + std::to_string(t.p) + "_result.json",
        gm_result_json("gm-pm1", "no-factor", 2, t, B1, B2,
                       std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                       backend, device_name, job_elapsed()));
    return 1;
}

int App::runGaussianMersenneECM() {
    GmTarget t;
    try { t = make_target(options.exponent); }
    catch (const std::exception& ex) { std::cerr << ex.what() << "\n"; return 2; }

    const std::uint64_t B1 = options.B1 != 0 ? options.B1 : 50000ULL;
    const std::uint64_t B2 = options.B2;
    const std::uint64_t curves = options.K != 0 ? options.K : (options.nmax != 0 ? options.nmax : 20ULL);
    const std::uint64_t chunk_bits = options.gm_factor_chunk_bits != 0
        ? options.gm_factor_chunk_bits : 131072ULL;
    const std::filesystem::path save_dir = options.save_path.empty() ? "." : options.save_path;
    std::filesystem::create_directories(save_dir);
    const auto job_clock = Clock::now();
    auto job_elapsed = [&]() {
        return std::chrono::duration<double>(Clock::now() - job_clock).count();
    };
    const std::string device_name = opencl_device_name(static_cast<std::size_t>(options.device_id));

    print_target("Gaussian-Mersenne ECM factoring", t);
    if (options.gm_sieve_limit != 0) {
        std::cout << "  admissible sieve: q=4kp+1 through " << options.gm_sieve_limit << "\n";
        const std::uint64_t sf = find_admissible_small_factor(t, options.gm_sieve_limit);
        if (sf != 0) {
            std::cout << ">>> Gaussian-Mersenne admissible-sieve factor: " << sf << "\n";
            write_json_result(
                save_dir, "gm_ecm_p" + std::to_string(t.p) + "_result.json",
                gm_result_json("gm-ecm", "factor", 0, t, B1,
                               B2 > B1 ? std::optional<std::uint64_t>(B2) : std::nullopt,
                               curves, std::nullopt, std::nullopt, std::to_string(sf),
                               "CPU sieve", device_name, job_elapsed(),
                               std::string("q=4kp+1 sieve")));
            return 0;
        }
    }
    std::cout << "  B1 / B2        : " << B1 << " / " << B2 << "\n"
              << "  curves         : " << curves << "\n"
              << "  curve family   : Suyama Montgomery, projective x/z\n"
              << "  projection     : exact CPU reduction modulo G_p after Stage 1 and each Stage 2 chunk\n"
              << "  inversions     : only modulo G_p on CPU; never modulo the lifted cofactor ring\n";

    const mpz_class K = buildE(B1);
    const std::uint64_t kbits = static_cast<std::uint64_t>(mpz_sizeinbase(K.get_mpz_t(), 2));
    std::cout << "  Stage 1 bits   : " << kbits << "\n";
    std::vector<std::uint32_t> s2primes;
    if (B2 > B1) s2primes = primes_in_range(B1, B2);

    const MontgomeryRegs r;
    std::unique_ptr<engine> eng(engine::create_gpu(t.lift, MontgomeryRegs::count,
                                                    static_cast<std::size_t>(options.device_id), true));
    const std::string backend = eng->is_aevum_backend() ? "Aevum" : "Marin";
    std::cout << "  backend        : " << backend << "\n"
              << "  transform      : " << eng->get_size() << " words\n"
              << "  registers      : " << MontgomeryRegs::count << "\n"
              << "  safe replay    : " << (options.gm_safe_replay ? "full Stage 1 + every Stage 2 chunk" : "disabled") << "\n";

    const std::uint64_t base_seed = options.seed != 0 ? options.seed : 0x474d45434d763938ULL;
    for (std::uint64_t curve = 0; curve < curves; ++curve) {
        std::uint64_t sigma = 0;
        if (!options.sigma.empty() && curve == 0) {
            try { sigma = std::stoull(options.sigma); }
            catch (...) { std::cerr << "Invalid -sigma for Gaussian ECM; expected an unsigned 64-bit integer.\n"; return 2; }
        } else {
            sigma = 6 + (splitmix64(base_seed + curve) % 0x7ffffffffffffff0ULL);
        }
        std::cout << "\n[GM ECM] curve " << (curve + 1) << "/" << curves << " sigma=" << sigma << "\n";
        CurveSetup setup = make_suyama_curve(t.n, sigma);
        if (is_proper_factor(setup.factor, t.n)) {
            std::cout << ">>> Gaussian-Mersenne ECM setup factor: " << setup.factor << "\n";
            write_json_result(
                save_dir, "gm_ecm_p" + std::to_string(t.p) + "_result.json",
                gm_result_json("gm-ecm", "factor", 0, t, B1,
                               B2 > B1 ? std::optional<std::uint64_t>(B2) : std::nullopt,
                               curves, curve + 1, std::to_string(sigma), setup.factor.get_str(),
                               backend, device_name, job_elapsed(), std::string("curve setup gcd")));
            return 0;
        }
        if (!setup.ok) {
            std::cout << "[GM ECM] singular setup; trying next curve.\n";
            continue;
        }

        montgomery_init_ladder(eng.get(), r, setup.x_affine, setup.a24, t.n);
        const std::filesystem::path ckpt = save_dir / ("gm_ecm_p" + std::to_string(t.p) + "_c" +
                                                         std::to_string(curve) + "_stage1.ckpt");
        std::uint64_t remaining = kbits > 0 ? kbits - 1 : 0;
        double restored = 0.0;
        if (options.resume) {
            load_factor_checkpoint(ckpt, eng.get(), 2, 1, t, B1, B2, kbits,
                                   0, static_cast<std::uint32_t>(curve), sigma, remaining, restored);
        }
        const auto curve_start = Clock::now();
        auto elapsed = [&]() { return restored + std::chrono::duration<double>(Clock::now() - curve_start).count(); };
        auto save_curve = [&](std::uint64_t rem) {
            save_factor_checkpoint(ckpt, eng.get(), 2, 1, t, B1, B2, kbits,
                                   0, static_cast<std::uint32_t>(curve), sigma, rem, elapsed());
        };
        std::vector<char> safe_stage1_start;
        std::uint64_t safe_stage1_remaining = remaining;
        if (options.gm_safe_replay) {
            eng->sync();
            safe_stage1_start.resize(eng->get_checkpoint_size());
            if (!eng->get_checkpoint(safe_stage1_start)) {
                throw std::runtime_error("cannot capture Gaussian ECM Stage 1 replay checkpoint");
            }
        }
        if (!montgomery_ladder(eng.get(), r, K, remaining, save_curve, elapsed,
                               "GM ECM Stage 1 curve " + std::to_string(curve + 1))) {
            return 0;
        }
        if (options.gm_safe_replay) {
            eng->sync();
            std::vector<char> first_x, first_z;
            if (!eng->get_data(first_x, r.xa) || !eng->get_data(first_z, r.za) ||
                !eng->set_checkpoint(safe_stage1_start)) {
                throw std::runtime_error("cannot prepare Gaussian ECM Stage 1 replay");
            }
            std::uint64_t replay_remaining = safe_stage1_remaining;
            auto no_checkpoint = [&](std::uint64_t) {};
            if (!montgomery_ladder(eng.get(), r, K, replay_remaining, no_checkpoint, elapsed,
                                   "GM ECM Stage 1 replay curve " + std::to_string(curve + 1))) {
                return 0;
            }
            eng->sync();
            std::vector<char> second_x, second_z;
            if (!eng->get_data(second_x, r.xa) || !eng->get_data(second_z, r.za) ||
                first_x != second_x || first_z != second_z) {
                throw std::runtime_error("Gaussian ECM Stage 1 safe replay mismatch");
            }
            std::cout << "[GM ECM safe replay] Stage 1 coordinates verified.\n";
        }
        clear_checkpoint(ckpt);
        eng->sync();
        PointProjection point = project_point(eng.get(), r, t.n);
        if (is_proper_factor(point.factor, t.n)) {
            std::cout << ">>> Gaussian-Mersenne ECM Stage 1 factor: " << point.factor << "\n";
            write_json_result(
                save_dir, "gm_ecm_p" + std::to_string(t.p) + "_result.json",
                gm_result_json("gm-ecm", "factor", 1, t, B1, std::nullopt,
                               curves, curve + 1, std::to_string(sigma), point.factor.get_str(),
                               backend, device_name, job_elapsed()));
            return 0;
        }
        if (!point.normalized) {
            std::cout << "[GM ECM] Stage 1 produced a singular/trivial point; next curve.\n";
            continue;
        }
        std::cout << "[GM ECM] Stage 1 no factor | x low64=0x" << low_hex(point.x, 64)
                  << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed() << " s\n";

        if (s2primes.empty()) continue;
        const std::filesystem::path s2_ckpt = save_dir / ("gm_ecm_p" + std::to_string(t.p) + "_c" +
                                                            std::to_string(curve) + "_stage2.ckpt");
        std::size_t index = 0;
        double s2_restored = 0.0;
        std::uint64_t s2_token = 0;
        if (options.resume && load_factor_checkpoint(s2_ckpt, eng.get(), 2, 2, t, B1, B2,
                                                      s2primes.size(), 0,
                                                      static_cast<std::uint32_t>(curve), sigma,
                                                      s2_token, s2_restored)) {
            index = static_cast<std::size_t>(std::min<std::uint64_t>(s2_token, s2primes.size()));
            eng->sync();
            point = project_point(eng.get(), r, t.n);
            if (is_proper_factor(point.factor, t.n)) {
                std::cout << ">>> Gaussian-Mersenne ECM Stage 2 checkpoint contains factor: "
                          << point.factor << "\n";
                write_json_result(
                    save_dir, "gm_ecm_p" + std::to_string(t.p) + "_result.json",
                    gm_result_json("gm-ecm", "factor", 2, t, B1, B2, curves,
                                   curve + 1, std::to_string(sigma), point.factor.get_str(),
                                   backend, device_name, job_elapsed(),
                                   std::string("stage2 checkpoint projection")));
                clear_checkpoint(s2_ckpt);
                return 0;
            }
            if (!point.normalized) {
                clear_checkpoint(s2_ckpt);
                std::cout << "[GM ECM] unusable Stage 2 checkpoint; restarting this curve Stage 2.\n";
                index = 0;
                point.x = setup.x_affine;
                point.z = 1;
                point.normalized = true;
                montgomery_init_ladder(eng.get(), r, point.x, setup.a24, t.n);
                std::uint64_t restart_remaining = kbits > 0 ? kbits - 1 : 0;
                auto no_checkpoint = [&](std::uint64_t) {};
                if (!montgomery_ladder(eng.get(), r, K, restart_remaining, no_checkpoint, elapsed,
                                       "GM ECM Stage 1 restart curve " + std::to_string(curve + 1))) return 0;
                point = project_point(eng.get(), r, t.n);
            } else {
                std::cout << "Resuming ECM Stage 2 at prime index " << index << "/" << s2primes.size() << "\n";
            }
        }
        const auto stage2_clock = Clock::now();
        auto total_s2_elapsed = [&]() {
            return s2_restored + std::chrono::duration<double>(Clock::now() - stage2_clock).count();
        };
        std::size_t chunk_no = 0;
        while (index < s2primes.size()) {
            if (interrupted) {
                save_factor_checkpoint(s2_ckpt, eng.get(), 2, 2, t, B1, B2, s2primes.size(),
                                       0, static_cast<std::uint32_t>(curve), sigma,
                                       index, total_s2_elapsed());
                std::cout << "Interrupted at a clean ECM Stage 2 chunk boundary; checkpoint saved.\n";
                return 0;
            }
            const std::size_t end = choose_chunk_end(s2primes, index, chunk_bits);
            const mpz_class qprod = product_range(s2primes, index, end);
            ++chunk_no;
            std::cout << "[GM ECM Stage 2] chunk " << chunk_no << " primes "
                      << s2primes[index] << ".." << s2primes[end - 1]
                      << " | count=" << (end - index)
                      << " | bits=" << mpz_sizeinbase(qprod.get_mpz_t(), 2) << "\n";

            // Re-embed the exact projected Stage-1/current point.  This avoids
            // every inversion in the lifted ring and makes each chunk a clean
            // homomorphic scalar multiplication modulo G_p.
            montgomery_init_ladder(eng.get(), r, point.x, setup.a24, t.n);
            std::vector<char> safe_chunk_start;
            if (options.gm_safe_replay) {
                eng->sync();
                safe_chunk_start.resize(eng->get_checkpoint_size());
                if (!eng->get_checkpoint(safe_chunk_start)) {
                    throw std::runtime_error("cannot capture Gaussian ECM Stage 2 replay checkpoint");
                }
            }
            const std::uint64_t chunk_work = mpz_sizeinbase(qprod.get_mpz_t(), 2) - 1;
            std::uint64_t s2_remaining = chunk_work;
            const auto s2_start = Clock::now();
            auto s2_elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - s2_start).count(); };
            auto no_ckpt = [&](std::uint64_t) {};
            if (!montgomery_ladder(eng.get(), r, qprod, s2_remaining, no_ckpt, s2_elapsed,
                                   "GM ECM Stage 2 curve " + std::to_string(curve + 1))) {
                return 0;
            }
            if (options.gm_safe_replay) {
                eng->sync();
                std::vector<char> first_x, first_z;
                if (!eng->get_data(first_x, r.xa) || !eng->get_data(first_z, r.za) ||
                    !eng->set_checkpoint(safe_chunk_start)) {
                    throw std::runtime_error("cannot prepare Gaussian ECM Stage 2 replay");
                }
                std::uint64_t replay_remaining = chunk_work;
                if (!montgomery_ladder(eng.get(), r, qprod, replay_remaining, no_ckpt, s2_elapsed,
                                       "GM ECM Stage 2 replay curve " + std::to_string(curve + 1))) return 0;
                eng->sync();
                std::vector<char> second_x, second_z;
                if (!eng->get_data(second_x, r.xa) || !eng->get_data(second_z, r.za) ||
                    first_x != second_x || first_z != second_z) {
                    throw std::runtime_error("Gaussian ECM Stage 2 safe replay mismatch");
                }
                std::cout << "[GM ECM safe replay] Stage 2 chunk verified.\n";
            }
            eng->sync();
            point = project_point(eng.get(), r, t.n);
            index = end;
            save_factor_checkpoint(s2_ckpt, eng.get(), 2, 2, t, B1, B2, s2primes.size(),
                                   0, static_cast<std::uint32_t>(curve), sigma,
                                   index, total_s2_elapsed());
            if (is_proper_factor(point.factor, t.n)) {
                std::cout << ">>> Gaussian-Mersenne ECM Stage 2 factor: " << point.factor << "\n";
                write_json_result(
                    save_dir, "gm_ecm_p" + std::to_string(t.p) + "_result.json",
                    gm_result_json("gm-ecm", "factor", 2, t, B1, B2, curves,
                                   curve + 1, std::to_string(sigma), point.factor.get_str(),
                                   backend, device_name, job_elapsed()));
                clear_checkpoint(s2_ckpt);
                return 0;
            }
            if (!point.normalized) {
                std::cout << "[GM ECM] Stage 2 singular/trivial point; abandoning this curve.\n";
                break;
            }
            std::cout << std::fixed << std::setprecision(2)
                      << "[GM ECM Stage 2] " << (100.0 * index / s2primes.size())
                      << "% | x low64=0x" << low_hex(point.x, 64)
                      << " | elapsed=" << total_s2_elapsed() << " s\n";
        }
        clear_checkpoint(s2_ckpt);
    }

    std::cout << "No Gaussian-Mersenne ECM factor found in " << curves << " curve(s).\n";
    write_json_result(
        save_dir, "gm_ecm_p" + std::to_string(t.p) + "_result.json",
        gm_result_json("gm-ecm", "no-factor", B2 > B1 ? 2 : 1, t, B1,
                       B2 > B1 ? std::optional<std::uint64_t>(B2) : std::nullopt,
                       curves, std::nullopt, std::nullopt, std::nullopt,
                       backend, device_name, job_elapsed()));
    return 1;
}

} // namespace core
