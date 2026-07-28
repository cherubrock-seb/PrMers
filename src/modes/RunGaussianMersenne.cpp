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
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr const char* GM_RELEASE = "v99.91";

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

std::string gm_test_result_json(const std::string& mode,
                                const std::string& outcome,
                                std::uint32_t exponent,
                                std::uint64_t digits,
                                std::uint32_t lift_exponent,
                                const std::optional<std::uint32_t>& base,
                                const std::optional<int>& jacobi,
                                const std::optional<std::string>& factor,
                                const std::optional<std::string>& factor_source,
                                const std::string& backend,
                                const std::string& device,
                                double elapsed_seconds,
                                const std::optional<std::string>& res64 = std::nullopt,
                                const std::optional<std::string>& res2048 = std::nullopt) {
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
         << "  \"stage\": 0,\n"
         << "  \"exponent\": " << exponent << ",\n"
         << "  \"B1\": null,\n"
         << "  \"B2\": null,\n"
         << "  \"curves\": null,\n"
         << "  \"sigma\": null,\n"
         << "  \"factor\": " << nullable_json_string(factor) << ",\n"
         << "  \"factor_source\": " << nullable_json_string(factor_source) << ",\n"
         << "  \"base\": " << (base ? std::to_string(*base) : "null") << ",\n"
         << "  \"jacobi\": " << (jacobi ? std::to_string(*jacobi) : "null") << ",\n"
         << "  \"digits\": " << digits << ",\n"
         << "  \"lift_exponent\": " << lift_exponent << ",\n"
         << "  \"res64\": " << nullable_json_string(res64) << ",\n"
         << "  \"res2048\": " << nullable_json_string(res2048) << ",\n"
         << "  \"backend\": \"" << json_escape(backend) << "\",\n"
         << "  \"device\": \"" << json_escape(device) << "\",\n"
         << "  \"elapsed_seconds\": " << std::max(0.0, elapsed_seconds) << ",\n"
         << "  \"timestamp\": \"" << iso8601_utc_now() << "\"\n"
         << "}";
    return json.str();
}

constexpr std::uint32_t GM_CHECKPOINT_VERSION = 1;
constexpr std::array<char, 8> GM_CHECKPOINT_MAGIC{{'P','R','G','M','L','I','F','T'}};

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
    // Deterministic for all 64-bit integers.
    constexpr std::array<std::uint64_t, 7> bases{{2ULL, 325ULL, 9375ULL, 28178ULL,
                                                   450775ULL, 9780504ULL, 1795265022ULL}};
    for (std::uint64_t a : bases) {
        if (a % n == 0) continue;
        std::uint64_t x = pow_mod_u64(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n - 1) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}

int legendre_two_for_odd_prime(std::uint64_t p) {
    const std::uint64_t r = p & 7U;
    return (r == 1 || r == 7) ? 1 : -1;
}

mpz_class gaussian_mersenne_norm(std::uint64_t p, int chi) {
    if (p == 2) return mpz_class(5);
    const std::uint64_t m = (p + 1) / 2;
    mpz_class n = mpz_class(1) << p;
    const mpz_class middle = mpz_class(1) << m;
    if (chi > 0) n -= middle;
    else n += middle;
    n += 1;
    return n;
}

std::uint64_t find_small_factor(std::uint64_t p, int chi, std::uint64_t limit, const mpz_class& n) {
    if (limit < 5) return 0;
    const std::uint64_t m = (p + 1) / 2;

    // From 2^(2p) == -1 (mod q), every prime factor q != 5 has
    // ord_q(2)=4p and therefore q == 1 (mod 4p).  Enumerating only those
    // candidates is dramatically faster than sieving every prime.
    if (mpz_divisible_ui_p(n.get_mpz_t(), 5) && mpz_cmp_ui(n.get_mpz_t(), 5) != 0) return 5;

    const unsigned __int128 step128 = static_cast<unsigned __int128>(4) * p;
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
        const std::uint64_t a = pow_mod_u64(2, p, q);
        const std::uint64_t b = pow_mod_u64(2, m, q);
        const std::uint64_t signed_b = chi > 0 ? b : (b == 0 ? 0 : q - b);
        const std::uint64_t residue = (a + q - signed_b + 1) % q;
        if (residue == 0 && mpz_cmp_ui(n.get_mpz_t(), q) != 0) return q;
    }
    return 0;
}

struct BaseSelection {
    std::uint32_t base = 0;
    std::uint64_t factor = 0;
    int jacobi = 0;
};

BaseSelection choose_proth_base(const mpz_class& n, std::uint32_t requested) {
    auto inspect = [&](std::uint32_t a) -> BaseSelection {
        mpz_class g;
        mpz_gcd_ui(g.get_mpz_t(), n.get_mpz_t(), a);
        if (g > 1 && g < n) return {a, g.get_ui(), 0};
        return {a, 0, mpz_jacobi(mpz_class(a).get_mpz_t(), n.get_mpz_t())};
    };

    if (requested != 0) return inspect(requested);

    constexpr std::array<std::uint32_t, 25> preferred{{
        3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101
    }};
    for (std::uint32_t a : preferred) {
        BaseSelection b = inspect(a);
        if (b.factor != 0 || b.jacobi == -1) return b;
    }
    for (std::uint32_t a = 103; a < 1'000'000; a += 2) {
        BaseSelection b = inspect(a);
        if (b.factor != 0 || b.jacobi == -1) return b;
    }
    throw std::runtime_error("could not find a small Jacobi -1 Proth base");
}

std::string hex_low(const mpz_class& x, unsigned bits) {
    mpz_class low;
    mpz_fdiv_r_2exp(low.get_mpz_t(), x.get_mpz_t(), bits);
    std::string s = low.get_str(16);
    const std::size_t width = bits / 4;
    if (s.size() < width) s.insert(0, width - s.size(), '0');
    return s;
}

struct GmCheckpointHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t p;
    std::uint32_t lift_exponent;
    std::uint32_t base;
    std::int32_t chi;
    std::uint32_t prp_only;
    std::uint32_t safe_replay;
    std::uint64_t next_operation;
    std::uint64_t total_operations;
    double elapsed_seconds;
    std::uint64_t checkpoint_bytes;
};

bool checkpoint_header_matches(const GmCheckpointHeader& h,
                               std::uint32_t p,
                               std::uint32_t lift,
                               std::uint32_t base,
                               int chi,
                               bool prp_only,
                               bool safe_replay,
                               std::uint64_t total_ops,
                               std::size_t bytes) {
    return std::equal(GM_CHECKPOINT_MAGIC.begin(), GM_CHECKPOINT_MAGIC.end(), h.magic) &&
           h.version == GM_CHECKPOINT_VERSION && h.p == p && h.lift_exponent == lift &&
           h.base == base && h.chi == chi && h.prp_only == static_cast<std::uint32_t>(prp_only) &&
           h.safe_replay == static_cast<std::uint32_t>(safe_replay) &&
           h.total_operations == total_ops && h.checkpoint_bytes == bytes &&
           h.next_operation <= total_ops;
}

} // namespace

namespace core {

int App::runGaussianMersenne() {
    const std::uint64_t p64 = options.exponent;
    if (p64 < 2) {
        std::cerr << "Gaussian-Mersenne exponent must be at least 2.\n";
        return 2;
    }
    if (!is_prime_u64(p64)) {
        std::cout << "Exponent p=" << p64 << " is composite, therefore the Gaussian-Mersenne norm is composite.\n";
        return 1;
    }
    if (p64 == 2) {
        std::cout << "G_2 = Norm((1+i)^2-1) = 5 is prime.\n";
        return 0;
    }
    if (p64 > std::numeric_limits<std::uint32_t>::max() / 4ULL) {
        std::cerr << "Gaussian-Mersenne Aevum lift requires 4p <= 2^32-1; maximum p is "
                  << (std::numeric_limits<std::uint32_t>::max() / 4ULL) << ".\n";
        return 2;
    }

    const std::uint32_t p = static_cast<std::uint32_t>(p64);
    const std::uint32_t lift_exponent = static_cast<std::uint32_t>(4ULL * p64);
    const std::uint64_t m = (p64 + 1) / 2;
    const int chi = legendre_two_for_odd_prime(p64);
    const mpz_class n = gaussian_mersenne_norm(p64, chi);
    const std::uint64_t decimal_digits = static_cast<std::uint64_t>(std::floor(p64 * std::log10(2.0))) + 1;
    const std::filesystem::path save_dir = options.save_path.empty() ? "." : options.save_path;
    std::filesystem::create_directories(save_dir);
    const auto job_clock = clock_type::now();
    auto job_elapsed = [&]() {
        return std::chrono::duration<double>(clock_type::now() - job_clock).count();
    };
    const std::string device_name = opencl_device_name(static_cast<std::size_t>(options.device_id));

    auto write_result = [&](const std::string& filename, const std::string& json) {
        const std::filesystem::path path = save_dir / filename;
        {
            std::ofstream out(path);
            out << json << '\n';
        }
        {
            std::ofstream out(save_dir / "results.txt", std::ios::app);
            out << json << '\n';
        }
        std::cout << "Result file: " << path << "\n";
    };

    std::cout << "Gaussian-Mersenne norm test\n"
              << "  p             : " << p << "\n"
              << "  G_p           : 2^" << p << (chi > 0 ? " - " : " + ")
              << "2^" << m << " + 1\n"
              << "  decimal digits: " << decimal_digits << "\n"
              << "  exact lift    : G_p divides 2^(2p)+1 and therefore 2^(4p)-1\n"
              << "  Aevum exponent: " << lift_exponent << "\n";

    if (options.gm_sieve_limit != 0) {
        std::cout << "Small-factor sieve through " << options.gm_sieve_limit << "..." << std::flush;
        const std::uint64_t factor = find_small_factor(p64, chi, options.gm_sieve_limit, n);
        if (factor != 0) {
            std::cout << " factor " << factor << " found.\n";
            write_result(
                "gm_factor_p" + std::to_string(p) + "_result.json",
                gm_test_result_json(options.gm_prp_only ? "gm-prp" : "gm-proth", "factor",
                                    p, decimal_digits, lift_exponent, std::nullopt, std::nullopt,
                                    std::to_string(factor), std::string("q=4kp+1 sieve"),
                                    "CPU sieve", device_name, job_elapsed()));
            return 1;
        }
        std::cout << " no factor found.\n";
    }

    BaseSelection selected;
    if (options.gm_prp_only) {
        selected.base = options.gm_base != 0 ? options.gm_base : 3U;
        mpz_class g;
        mpz_gcd_ui(g.get_mpz_t(), n.get_mpz_t(), selected.base);
        if (g > 1 && g < n) {
            std::cout << "Base gcd found factor " << g << ".\n";
            write_result(
                "gm_factor_p" + std::to_string(p) + "_result.json",
                gm_test_result_json("gm-prp", "factor", p, decimal_digits, lift_exponent,
                                    selected.base, std::nullopt, g.get_str(),
                                    std::string("base gcd"), "CPU gcd", device_name, job_elapsed()));
            return 1;
        }
        selected.jacobi = mpz_jacobi(mpz_class(selected.base).get_mpz_t(), n.get_mpz_t());
    } else {
        selected = choose_proth_base(n, options.gm_base);
        if (selected.factor != 0) {
            std::cout << "Proth base selection found factor " << selected.factor << ".\n";
            write_result(
                "gm_factor_p" + std::to_string(p) + "_result.json",
                gm_test_result_json("gm-proth", "factor", p, decimal_digits, lift_exponent,
                                    selected.base, std::nullopt, std::to_string(selected.factor),
                                    std::string("Proth base gcd"), "CPU gcd", device_name, job_elapsed()));
            return 1;
        }
        if (selected.jacobi != -1) {
            std::cerr << "Deterministic Proth mode requires Jacobi(base/G_p) = -1; base "
                      << selected.base << " has Jacobi " << selected.jacobi << ".\n";
            return 2;
        }
    }

    std::cout << "  mode          : " << (options.gm_prp_only ? "Fermat PRP" : "deterministic Proth proof") << "\n"
              << "  base          : " << selected.base << " (Jacobi " << selected.jacobi << ")\n";

    const std::uint64_t phase_a = chi > 0 ? (m - 2) : (m - 1);
    const std::uint64_t phase_mul = chi > 0 ? 0 : 1;
    const std::uint64_t phase_b = m - 1;
    const std::uint64_t euler_ops = phase_a + phase_mul + phase_b;
    const std::uint64_t total_ops = euler_ops + (options.gm_prp_only ? 1 : 0);

    if (options.gm_cpu) {
        std::cout << "CPU GMP reference path selected.\n";
        mpz_class exponent = (n - 1) / 2;
        if (options.gm_prp_only) exponent *= 2;
        mpz_class residue;
        const auto t0 = clock_type::now();
        mpz_powm(residue.get_mpz_t(), mpz_class(selected.base).get_mpz_t(), exponent.get_mpz_t(), n.get_mpz_t());
        const double elapsed = std::chrono::duration<double>(clock_type::now() - t0).count();
        const bool pass = options.gm_prp_only ? residue == 1 : residue == n - 1;
        std::cout << "CPU residue low64 = 0x" << hex_low(residue, 64) << "\n";
        std::cout << "G_" << p << (pass ? (options.gm_prp_only ? " is a probable prime" : " is prime by Proth") : " is composite")
                  << ", time = " << std::fixed << std::setprecision(2) << elapsed << " s.\n";
        const std::string cpu_res64 = hex_low(residue, 64);
        const std::string cpu_res2048 = hex_low(residue, 2048);
        const std::string outcome = pass
            ? (options.gm_prp_only ? "probable-prime" : "prime")
            : "composite";
        write_result(
            "gm_" + std::string(options.gm_prp_only ? "prp" : "proth") + "_p" + std::to_string(p) + "_result.json",
            gm_test_result_json(options.gm_prp_only ? "gm-prp" : "gm-proth", outcome,
                                p, decimal_digits, lift_exponent, selected.base, selected.jacobi,
                                std::nullopt, std::nullopt, "GMP", "CPU",
                                elapsed, cpu_res64, cpu_res2048));
        return pass ? 0 : 1;
    }

    // Register layout is local to this mode. Existing Mersenne modes and kernels
    // are not changed. Arithmetic occurs modulo M_(4p), then the final residue is
    // projected modulo the exact factor G_p.
    constexpr engine::Reg RSTATE = 0;
    constexpr engine::Reg RBASE = 1;
    constexpr engine::Reg RBASE_PREP = 2;
    constexpr engine::Reg RSTART = 3;
    constexpr engine::Reg RVERIFY = 4;
    const std::size_t register_count = options.gm_safe_replay ? 5U : (chi < 0 ? 3U : 1U);

    std::unique_ptr<engine> eng(engine::create_gpu(lift_exponent, register_count,
                                                    static_cast<std::size_t>(options.device_id), true));
    std::cout << "  backend       : " << (eng->is_aevum_backend() ? "Aevum" : "Marin") << "\n"
              << "  transform     : " << eng->get_size() << " words\n"
              << "  registers     : " << register_count << "\n";
    if (!eng->is_aevum_backend()) {
        std::cout << "  note          : use -aevum to force the Aevum plugin for record-oriented runs.\n";
    }
    if (options.gm_safe_replay) {
        std::cout << "  safety        : full block replay and register comparison (strong, roughly 2x arithmetic)\n";
    } else {
        std::cout << "  safety        : Aevum redundant arithmetic/roundoff checks plus periodic CRC checkpoints\n"
                  << "                  use -gm-safe for independent full block replay\n";
    }

    eng->set(RSTATE, selected.base);
    if (chi < 0) {
        eng->set(RBASE, selected.base);
        eng->set_multiplicand(RBASE_PREP, RBASE);
    }
    if (options.gm_safe_replay) {
        eng->copy(RSTART, RSTATE);
        eng->copy(RVERIFY, RSTATE);
    }

    auto apply_operation = [&](engine::Reg reg, std::uint64_t op) {
        if (options.gm_prp_only && op == euler_ops) {
            eng->square_mul(reg);
            return;
        }
        if (chi > 0) {
            if (op < phase_a) eng->square_mul(reg, selected.base);
            else eng->square_mul(reg);
        } else {
            if (op < phase_a) eng->square_mul(reg);
            else if (op == phase_a) eng->mul(reg, RBASE_PREP);
            else eng->square_mul(reg);
        }
    };

    const std::string mode_tag = options.gm_prp_only ? "prp" : "proth";
    const std::filesystem::path checkpoint_path = save_dir / ("gm_" + mode_tag + "_p" + std::to_string(p) + ".ckpt");

    auto load_checkpoint = [&](const std::filesystem::path& path, std::uint64_t& next, double& elapsed) -> bool {
        File f(path.string());
        if (!f.exists()) return false;
        GmCheckpointHeader h{};
        if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
        const std::size_t bytes = eng->get_checkpoint_size();
        if (!checkpoint_header_matches(h, p, lift_exponent, selected.base, chi,
                                       options.gm_prp_only, options.gm_safe_replay,
                                       total_ops, bytes)) {
            std::cout << "[GM checkpoint] Ignoring incompatible " << path << ".\n";
            return false;
        }
        std::vector<char> data(bytes);
        if (!f.read(data.data(), data.size()) || !f.check_crc32() || !eng->set_checkpoint(data)) return false;
        next = h.next_operation;
        elapsed = h.elapsed_seconds;
        return true;
    };

    auto save_checkpoint = [&](std::uint64_t next, double elapsed) {
        eng->sync();
        const std::filesystem::path new_path = checkpoint_path.string() + ".new";
        const std::filesystem::path old_path = checkpoint_path.string() + ".old";
        GmCheckpointHeader h{};
        std::copy(GM_CHECKPOINT_MAGIC.begin(), GM_CHECKPOINT_MAGIC.end(), h.magic);
        h.version = GM_CHECKPOINT_VERSION;
        h.p = p;
        h.lift_exponent = lift_exponent;
        h.base = selected.base;
        h.chi = chi;
        h.prp_only = static_cast<std::uint32_t>(options.gm_prp_only);
        h.safe_replay = static_cast<std::uint32_t>(options.gm_safe_replay);
        h.next_operation = next;
        h.total_operations = total_ops;
        h.elapsed_seconds = elapsed;
        h.checkpoint_bytes = eng->get_checkpoint_size();
        std::vector<char> data(static_cast<std::size_t>(h.checkpoint_bytes));
        if (!eng->get_checkpoint(data)) throw std::runtime_error("cannot read Gaussian-Mersenne engine checkpoint");
        {
            File f(new_path.string(), "wb");
            if (!f.write(reinterpret_cast<const char*>(&h), sizeof(h)) ||
                !f.write(data.data(), data.size())) {
                throw std::runtime_error("cannot write Gaussian-Mersenne checkpoint");
            }
            f.write_crc32();
        }
        std::error_code ec;
        std::filesystem::remove(old_path, ec);
        if (std::filesystem::exists(checkpoint_path)) std::filesystem::rename(checkpoint_path, old_path, ec);
        ec.clear();
        std::filesystem::rename(new_path, checkpoint_path, ec);
        if (ec) throw std::runtime_error("cannot atomically install Gaussian-Mersenne checkpoint: " + ec.message());
    };

    std::uint64_t next_op = 0;
    double restored_elapsed = 0.0;
    if (options.resume) {
        if (!load_checkpoint(checkpoint_path, next_op, restored_elapsed)) {
            load_checkpoint(checkpoint_path.string() + ".old", next_op, restored_elapsed);
        }
    }
    if (next_op != 0) {
        if (chi < 0) eng->set_multiplicand(RBASE_PREP, RBASE);
        std::cout << "Resuming at operation " << next_op << " / " << total_ops << ".\n";
    }

    const std::uint64_t replay_block = options.gm_replay_block != 0
        ? options.gm_replay_block
        : std::max<std::uint64_t>(64, static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(total_ops))));
    const auto run_start = clock_type::now();
    auto last_backup = run_start;
    auto last_display = run_start;
    std::uint64_t display_start_op = next_op;
    double display_start_elapsed = restored_elapsed;
    std::uint64_t error_count = 0;
    bool error_injected = false;

    auto elapsed_now = [&]() {
        return restored_elapsed + std::chrono::duration<double>(clock_type::now() - run_start).count();
    };

    while (next_op < total_ops) {
        if (core::algo::interrupted) {
            save_checkpoint(next_op, elapsed_now());
            std::cout << "Interrupted; checkpoint saved at operation " << next_op << ".\n";
            return 0;
        }

        const std::uint64_t block_end = options.gm_safe_replay
            ? std::min(total_ops, next_op + replay_block)
            : next_op + 1;

        if (options.gm_safe_replay) {
            eng->copy(RSTART, RSTATE);
            for (std::uint64_t op = next_op; op < block_end; ++op) {
                apply_operation(RSTATE, op);
                if (options.erroriter != 0 && op + 1 == options.erroriter && !error_injected) {
                    eng->sub(RSTATE, 1);
                    error_injected = true;
                }
            }
            eng->copy(RVERIFY, RSTART);
            for (std::uint64_t op = next_op; op < block_end; ++op) apply_operation(RVERIFY, op);
            if (!eng->is_equal(RSTATE, RVERIFY)) {
                ++error_count;
                std::cout << "[GM safe replay] mismatch in block [" << next_op << "," << block_end
                          << "); restoring and retrying.\n";
                eng->copy(RSTATE, RSTART);
                continue;
            }
            next_op = block_end;
        } else {
            apply_operation(RSTATE, next_op);
            if (options.erroriter != 0 && next_op + 1 == options.erroriter && !error_injected) {
                eng->sub(RSTATE, 1);
                error_injected = true;
            }
            ++next_op;
        }

        if (options.iterforce != 0 && next_op % options.iterforce == 0) eng->sync();

        const auto now = clock_type::now();
        if (now - last_display >= std::chrono::seconds(10) || next_op == total_ops) {
            eng->sync();
            const double elapsed = elapsed_now();
            const double interval = elapsed - display_start_elapsed;
            const double ips = interval > 0 ? static_cast<double>(next_op - display_start_op) / interval : 0.0;
            const double eta = ips > 0 ? static_cast<double>(total_ops - next_op) / ips : 0.0;
            const double pct = total_ops != 0 ? 100.0 * static_cast<double>(next_op) / static_cast<double>(total_ops) : 100.0;
            std::cout << std::fixed << std::setprecision(2)
                      << "Progress: " << pct << "% | op " << next_op << "/" << total_ops
                      << " | IPS " << ips << " | elapsed " << elapsed << "s | ETA " << eta << "s\n";
            if (guiServer_) {
                guiServer_->setProgress(next_op, total_ops, "Gaussian-Mersenne");
                guiServer_->appendLog("Gaussian-Mersenne progress " + std::to_string(pct) + "%");
            }
            display_start_op = next_op;
            display_start_elapsed = elapsed;
            last_display = now;
        }
        if (now - last_backup >= std::chrono::seconds(options.backup_interval)) {
            save_checkpoint(next_op, elapsed_now());
            std::cout << "[GM checkpoint] saved at operation " << next_op << ".\n";
            last_backup = now;
        }
    }

    eng->sync();
    mpz_t raw;
    mpz_init(raw);
    eng->get_mpz(raw, RSTATE);
    mpz_class lifted_residue(raw);
    mpz_clear(raw);
    mpz_class residue = lifted_residue % n;
    if (residue < 0) residue += n;

    const bool pass = options.gm_prp_only ? residue == 1 : residue == n - 1;
    const double elapsed = elapsed_now();
    const std::string status = pass
        ? (options.gm_prp_only ? "probable-prime" : "prime")
        : "composite";
    const std::string res64 = hex_low(residue, 64);
    const std::string res2048 = hex_low(residue, 2048);

    std::cout << "Final residue modulo G_p low64: 0x" << res64 << "\n";
    if (pass && !options.gm_prp_only) {
        std::cout << "G_" << p << " is PRIME by Proth's theorem.\n";
    } else if (pass) {
        std::cout << "G_" << p << " is a base-" << selected.base << " probable prime.\n";
    } else {
        std::cout << "G_" << p << " is composite.\n";
    }
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << elapsed
              << " s; replay mismatches: " << error_count << ".\n";

    write_result(
        "gm_" + mode_tag + "_p" + std::to_string(p) + "_result.json",
        gm_test_result_json(options.gm_prp_only ? "gm-prp" : "gm-proth", status,
                            p, decimal_digits, lift_exponent, selected.base, selected.jacobi,
                            std::nullopt, std::nullopt,
                            eng->is_aevum_backend() ? "Aevum" : "Marin", device_name,
                            elapsed, res64, res2048));

    std::error_code ec;
    std::filesystem::remove(checkpoint_path, ec);
    std::filesystem::remove(checkpoint_path.string() + ".old", ec);
    std::filesystem::remove(checkpoint_path.string() + ".new", ec);
    return pass ? 0 : 1;
}

} // namespace core
