#include "core/Bench3.hpp"

#include "aevum/EngineAevum.hpp"
#include "core/Version.hpp"
#include "io/CliParser.hpp"
#include "marin/engine.h"
#include "opencl/Context.hpp"
#include "util/SelfExe.hpp"
#include "util/WinCmdLine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gmpxx.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#ifndef PRMERS_GIT_SHA
#define PRMERS_GIT_SHA unknown
#endif
#define PRMERS_STRINGIFY_IMPL(value) #value
#define PRMERS_STRINGIFY(value) PRMERS_STRINGIFY_IMPL(value)

namespace core::bench3 {
namespace {

constexpr const char* kSchemaVersion = "bench3.v1";
constexpr const char* kPlanCheckpointVersion = "bench3.plan.v1";
constexpr std::size_t kPrpRegisterCount = 8;
volatile std::sig_atomic_t g_stop = 0;

void signal_handler(int) { g_stop = 1; }

struct ModeConfig {
    std::string name;
    std::size_t warmup = 0;
    std::size_t timed_iterations = 0;
    std::size_t repetitions = 0;
    std::size_t exact_iterations = 0;
    std::vector<std::uint32_t> exponents;
};

struct DeviceInfo {
    std::string vendor = "unknown";
    std::string name = "unknown";
    std::string driver = "unknown";
    std::string runtime = "unknown";
    std::uint32_t compute_units = 0;
    std::uint32_t clock_mhz = 0;
    std::uint64_t global_mem_bytes = 0;
    std::uint64_t local_mem_bytes = 0;
    std::size_t max_workgroup = 0;
};

struct ProductionTiming {
    std::uint64_t block = 0;
    std::uint64_t checkpasslevel = 0;
    std::uint64_t full_check_interval = 0;
    double boundary_us = 0.0;
    double full_check_us = 0.0;
    double amortized_us_per_iter = 0.0;
    double production_us_per_iter = 0.0;
    bool exact = false;
};

struct PlanResult {
    std::string status = "PASS";
    std::string error;
    std::string request;
    std::string resolved_plan;
    std::string active_plan;
    std::string backend;
    bool auto_baseline = false;
    bool legal = true;
    bool duplicate = false;
    bool exact_match = false;
    bool production_probe_exact = false;
    bool best_exact = false;
    std::string residue_hash;

    std::size_t transform_words = 0;
    double transform_mwords = 0.0;
    double bpw = 0.0;
    double setup_ms = 0.0;
    double us_median = 0.0;
    double us_min = 0.0;
    double us_max = 0.0;
    double production_us_per_iter = 0.0;
    double delta_vs_auto_pct = 0.0;
};

struct PointResult {
    std::uint32_t exponent = 0;
    std::size_t device_index = 0;
    DeviceInfo device;
    std::vector<PlanResult> plans;
};

void write_plan_progress(const std::filesystem::path& path,
                         const std::string& phase,
                         double elapsed_s);

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<unsigned>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string cl_string(cl_device_id device, cl_device_info what) {
    std::size_t bytes = 0;
    if (clGetDeviceInfo(device, what, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0)
        return "unknown";
    std::string value(bytes, '\0');
    if (clGetDeviceInfo(device, what, bytes, value.data(), nullptr) != CL_SUCCESS)
        return "unknown";
    while (!value.empty() && value.back() == '\0') value.pop_back();
    for (char& c : value)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    return value;
}

template <typename T>
T cl_scalar(cl_device_id device, cl_device_info what) {
    T value{};
    if (clGetDeviceInfo(device, what, sizeof(value), &value, nullptr) != CL_SUCCESS)
        return T{};
    return value;
}

DeviceInfo device_info(std::size_t device_index) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0)
        throw std::runtime_error("No OpenCL platform found");

    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    std::vector<cl_device_id> devices;
    for (const auto platform : platforms) {
        cl_uint count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) != CL_SUCCESS || count == 0)
            continue;
        const std::size_t old = devices.size();
        devices.resize(old + count);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data() + old, nullptr);
    }

    if (device_index >= devices.size())
        throw std::runtime_error("OpenCL GPU device index is out of range");

    const auto d = devices[device_index];
    DeviceInfo info;
    info.vendor = cl_string(d, CL_DEVICE_VENDOR);
    info.name = cl_string(d, CL_DEVICE_NAME);
    info.driver = cl_string(d, CL_DRIVER_VERSION);
    info.runtime = cl_string(d, CL_DEVICE_VERSION);
    info.compute_units = cl_scalar<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS);
    info.clock_mhz = cl_scalar<cl_uint>(d, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    info.global_mem_bytes = static_cast<std::uint64_t>(cl_scalar<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE));
    info.local_mem_bytes = static_cast<std::uint64_t>(cl_scalar<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE));
    info.max_workgroup = cl_scalar<std::size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    return info;
}

std::vector<std::uint32_t> full_grid() {
    return {
         37156667u, 58000003u, 77232917u, 82589933u, 100000007u,
        130000007u, 145000007u, 150000007u, 160000003u, 170000003u,
        175000003u, 180000017u, 196999969u, 197000003u, 200000033u,
        210000017u, 220000013u, 250000013u, 300000007u, 400000009u,
        500000003u, 600000001u
    };
}

ModeConfig mode_config(const std::string& mode) {
    if (mode == "quick")
        return {"quick", 16, 64, 3, 64,
                {145000007u, 170000003u, 197000003u, 210000017u, 500000003u}};
    if (mode == "standard")
        return {"standard", 24, 128, 3, 96,
                {77232917u, 130000007u, 145000007u, 170000003u,
                 196999969u, 197000003u, 210000017u, 220000013u,
                 300000007u, 500000003u}};
    if (mode == "dense")
        return {"dense", 48, 256, 5, 128, full_grid()};
    if (mode == "full")
        return {"full", 96, 512, 7, 192, full_grid()};
    throw std::runtime_error(
        "invalid -bench3-mode '" + mode +
        "' (expected quick, standard, dense or full)");
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n == 0) return 0.0;
    if (n & 1u) return values[n / 2];
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

std::string compiler_string() {
#ifdef __clang__
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("msvc ") + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

std::string residue_hash(const mpz_class& value) {
    const std::string text = value.get_str(16);
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << h;
    return out.str();
}

mpz_class exact_residue(engine& eng, std::size_t iterations) {
    eng.set(0, 3u);
    for (std::size_t i = 0; i < iterations; ++i) {
        if (g_stop) throw std::runtime_error("bench3 interrupted");
        eng.square_mul(0);
    }
    eng.sync();
    mpz_t z;
    mpz_init(z);
    eng.get_mpz(z, 0);
    mpz_class result(z);
    mpz_clear(z);
    return result;
}

ProductionTiming production_timing(engine& eng,
                                   std::uint32_t exponent,
                                   double square_us_per_iter) {
    const engine::Reg R0 = 0, R1 = 1, R2 = 2, R3 = 3;
    const engine::Reg R4 = 4, R5 = 5, RBASE = 6, RTMP = 7;

    ProductionTiming out;
    const std::uint64_t legacy_B =
        static_cast<std::uint64_t>(std::sqrt(static_cast<double>(exponent)));
    out.block = std::min<std::uint64_t>(1000u, legacy_B);
    if (out.block == 0) throw std::runtime_error("bench3 Gerbicz block is zero");

    out.checkpasslevel =
        static_cast<std::uint64_t>((1000.0 * 600.0) / static_cast<double>(out.block));
    if (out.checkpasslevel == 0) out.checkpasslevel = 1;
    out.full_check_interval = out.block * out.checkpasslevel;

    eng.set(R1, 1u);
    eng.set(R0, 3u);
    eng.copy(R4, R0);
    eng.copy(R5, R1);
    eng.set(RBASE, 3u);
    eng.set_multiplicand(RTMP, RBASE);

    const std::uint64_t modB = exponent % out.block == 0 ? out.block : exponent % out.block;
    for (std::uint64_t z = 0; z < modB; ++z) eng.square_mul(R0);
    eng.sync();

    const auto boundary_begin = std::chrono::steady_clock::now();
    eng.copy(R3, R1);
    eng.set_multiplicand(R2, R0);
    eng.mul(R1, R2);
    eng.sync();
    const auto boundary_end = std::chrono::steady_clock::now();

    out.boundary_us =
        std::chrono::duration<double, std::micro>(boundary_end - boundary_begin).count();

    const auto full_begin = std::chrono::steady_clock::now();
    const std::uint64_t loop_count = out.block > modB ? out.block - modB - 1 : 0;
    for (std::uint64_t z = 0; z < loop_count; ++z) eng.square_mul(R3);

    if (exponent % out.block == 0) eng.mul(R3, RTMP);
    else eng.square_mul(R3, 3u);

    for (std::uint64_t z = 0; z < modB; ++z) eng.square_mul(R3);

    mpz_t z0, z1;
    mpz_inits(z0, z1, nullptr);
    eng.get_mpz(z0, R3);
    eng.get_mpz(z1, R1);
    const mpz_class Mp = (mpz_class(1) << exponent) - 1;
    out.exact = ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);
    mpz_clears(z0, z1, nullptr);
    if (!out.exact) throw std::runtime_error("bench3 production PRP Gerbicz probe mismatch");

    eng.copy(R4, R0);
    eng.copy(R5, R1);
    eng.sync();

    const auto full_end = std::chrono::steady_clock::now();
    out.full_check_us =
        std::chrono::duration<double, std::micro>(full_end - full_begin).count();

    out.amortized_us_per_iter =
        out.boundary_us / static_cast<double>(out.block) +
        out.full_check_us / static_cast<double>(out.full_check_interval);
    out.production_us_per_iter = square_us_per_iter + out.amortized_us_per_iter;
    return out;
}

PlanResult measure_plan(std::uint32_t exponent,
                        std::size_t device_index,
                        const ModeConfig& cfg,
                        bool auto_baseline,
                        const std::string& request,
                        const std::string& resolved_plan,
                        const mpz_class* auto_residue,
                        mpz_class* measured_residue,
                        const std::filesystem::path* progress_path = nullptr) {
    PlanResult r;
    const auto plan_begin = std::chrono::steady_clock::now();
    const auto progress = [&](const std::string& phase) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - plan_begin).count();
        std::ostringstream line;
        line << std::fixed << std::setprecision(3)
             << "BENCH3_PROGRESS p=" << exponent
             << " request=" << request
             << " phase=" << phase
             << " plan=" << (resolved_plan.empty() ? "AUTO" : resolved_plan)
             << " elapsed_s=" << elapsed;
        std::cout << line.str() << std::endl;
        if (progress_path)
            write_plan_progress(*progress_path, phase, elapsed);
    };

    r.auto_baseline = auto_baseline;
    r.request = request;
    r.resolved_plan = resolved_plan;

    const auto setup_begin = std::chrono::steady_clock::now();
    progress("setup_begin");
    try {
        if (auto_baseline) {
            engine::configure_gpu_backend(
                engine::gpu_backend::auto_select, "", engine::gpu_workload::prp);
        } else {
            engine::configure_gpu_backend(
                engine::gpu_backend::aevum, resolved_plan, engine::gpu_workload::prp);
        }

        std::unique_ptr<engine> eng(
            engine::create_gpu(exponent, kPrpRegisterCount, device_index, false));
        if (!eng) throw std::runtime_error("engine::create_gpu returned null");
        eng->sync();
        progress("setup_done");

        r.setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - setup_begin).count();
        r.backend = eng->is_aevum_backend() ? "Aevum" : "Marin";
        r.transform_words = eng->get_size();
        r.transform_mwords =
            static_cast<double>(r.transform_words) / (1024.0 * 1024.0);
        if (r.transform_words != 0)
            r.bpw = static_cast<double>(exponent) /
                    static_cast<double>(r.transform_words);

        if (eng->is_aevum_backend()) {
            r.active_plan = aevum_engine_active_plan(eng.get());
            if (r.active_plan.empty()) r.active_plan = resolved_plan;
        } else {
            r.active_plan = "marin-native";
        }

        progress("exact_begin");
        const mpz_class residue = exact_residue(*eng, cfg.exact_iterations);
        r.residue_hash = residue_hash(residue);
        if (measured_residue) *measured_residue = residue;
        progress("exact_done");

        r.exact_match = auto_baseline ||
                        (auto_residue != nullptr && residue == *auto_residue);
        if (!r.exact_match) {
            r.status = "EXACT_MISMATCH";
            r.error = "full deterministic residue differs from AUTO";
            progress("exact_mismatch");
            return r;
        }

        progress("warmup_begin");
        eng->set(0, 3u);
        for (std::size_t i = 0; i < cfg.warmup; ++i) {
            if (g_stop) throw std::runtime_error("bench3 interrupted");
            eng->square_mul(0);
        }
        eng->sync();
        progress("warmup_done");

        std::vector<double> samples;
        samples.reserve(cfg.repetitions);
        for (std::size_t rep = 0; rep < cfg.repetitions; ++rep) {
            progress("timing_rep_" + std::to_string(rep + 1) + "_begin");
            eng->set(0, 3u);
            eng->sync();
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < cfg.timed_iterations; ++i) {
                if (g_stop) throw std::runtime_error("bench3 interrupted");
                eng->square_mul(0);
            }
            eng->sync();
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count() /
                static_cast<double>(cfg.timed_iterations));
            progress("timing_rep_" + std::to_string(rep + 1) + "_done");
        }

        r.us_median = median(samples);
        r.us_min = *std::min_element(samples.begin(), samples.end());
        r.us_max = *std::max_element(samples.begin(), samples.end());

        progress("production_probe_begin");
        const ProductionTiming prod =
            production_timing(*eng, exponent, r.us_median);
        r.production_probe_exact = prod.exact;
        r.production_us_per_iter = prod.production_us_per_iter;
        progress("production_probe_done");
    } catch (const std::exception& e) {
        if (g_stop) throw;
        progress("error");
        const std::string detail = e.what();
        if (!auto_baseline &&
            detail.find("small-factor capacity exceeded") != std::string::npos) {
            r.status = "SKIPPED_ILLEGAL";
            r.legal = false;
            r.production_probe_exact = false;
            r.production_us_per_iter = 0.0;
            r.error = "PRP factor-safe rejection: " + detail;
        } else if (!auto_baseline &&
                   detail.find("workgroup size too large for kernel") != std::string::npos) {
            r.status = "SKIPPED_ILLEGAL";
            r.legal = false;
            r.production_probe_exact = false;
            r.production_us_per_iter = 0.0;
            r.error = "device workgroup-limit rejection: " + detail;
        } else {
            r.status = "ERROR";
            r.error = detail;
        }
        r.setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - setup_begin).count();
    }
    return r;
}

std::vector<std::string> candidate_requests() {
    return {
        "pow2:auto",
        "pfa:auto",
        "pfa:3",
        "pfa7:1:512:7:512:202",
        "pfa:9"
    };
}

std::string checkpoint_context(std::uint32_t exponent,
                               std::size_t device_index,
                               const ModeConfig& cfg) {
    const DeviceInfo info = device_info(device_index);
    std::ostringstream o;
    o << kSchemaVersion << '|'
      << PRMERS_STRINGIFY(PRMERS_GIT_SHA) << '|'
      << cfg.name << '|'
      << exponent << '|'
      << device_index << '|'
      << info.vendor << '|'
      << info.name << '|'
      << info.driver << '|'
      << info.runtime << '|'
      << cfg.warmup << '|'
      << cfg.timed_iterations << '|'
      << cfg.repetitions << '|'
      << cfg.exact_iterations;
    return o.str();
}

std::filesystem::path partial_point_path(const std::filesystem::path& root,
                                         std::size_t device,
                                         std::uint32_t exponent) {
    std::ostringstream name;
    name << "d" << device << "_p" << std::setw(10) << std::setfill('0')
         << exponent;
    return root / name.str();
}

std::filesystem::path plan_checkpoint_path(const std::filesystem::path& dir,
                                           std::size_t index) {
    std::ostringstream name;
    name << "plan_" << std::setw(2) << std::setfill('0') << index << ".state";
    return dir / name.str();
}

std::filesystem::path plan_progress_path(const std::filesystem::path& dir,
                                         std::size_t index) {
    std::ostringstream name;
    name << "plan_" << std::setw(2) << std::setfill('0') << index << ".progress";
    return dir / name.str();
}

void write_plan_progress(const std::filesystem::path& path,
                         const std::string& phase,
                         double elapsed_s) {
    try {
        std::filesystem::create_directories(path.parent_path());
        auto tmp = path;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) return;
            out << std::quoted(phase) << ' '
                << std::setprecision(17) << elapsed_s << '\n';
            out.flush();
            if (!out) return;
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
        if (ec) std::filesystem::remove(tmp, ec);
    } catch (...) {
        // Progress reporting must never make a benchmark fail.
    }
}

bool read_plan_progress(const std::filesystem::path& path,
                        std::string& phase,
                        double& elapsed_s) {
    std::ifstream in(path);
    if (!in) return false;
    return static_cast<bool>(in >> std::quoted(phase) >> elapsed_s);
}

void atomic_write_plan_checkpoint(const std::filesystem::path& path,
                                  const std::string& context,
                                  const PlanResult& r) {
    std::filesystem::create_directories(path.parent_path());
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create Bench3 plan checkpoint");
        out << kPlanCheckpointVersion << '\n'
            << std::quoted(context) << '\n'
            << std::quoted(r.status) << '\n'
            << std::quoted(r.error) << '\n'
            << std::quoted(r.request) << '\n'
            << std::quoted(r.resolved_plan) << '\n'
            << std::quoted(r.active_plan) << '\n'
            << std::quoted(r.backend) << '\n'
            << std::quoted(r.residue_hash) << '\n'
            << r.auto_baseline << ' ' << r.legal << ' ' << r.duplicate << ' '
            << r.exact_match << ' ' << r.production_probe_exact << '\n'
            << r.transform_words << '\n'
            << std::setprecision(17)
            << r.transform_mwords << ' ' << r.bpw << ' ' << r.setup_ms << ' '
            << r.us_median << ' ' << r.us_min << ' ' << r.us_max << ' '
            << r.production_us_per_iter << '\n';
        out.flush();
        if (!out) throw std::runtime_error("cannot flush Bench3 plan checkpoint");
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp);
        throw std::runtime_error("cannot publish Bench3 plan checkpoint");
    }
}

bool read_plan_checkpoint(const std::filesystem::path& path,
                          const std::string& expected_context,
                          PlanResult& r) {
    std::ifstream in(path);
    if (!in) return false;

    std::string version;
    if (!std::getline(in, version) || version != kPlanCheckpointVersion)
        return false;

    std::string context;
    if (!(in >> std::quoted(context)) || context != expected_context)
        return false;

    PlanResult loaded;
    if (!(in >> std::quoted(loaded.status)
             >> std::quoted(loaded.error)
             >> std::quoted(loaded.request)
             >> std::quoted(loaded.resolved_plan)
             >> std::quoted(loaded.active_plan)
             >> std::quoted(loaded.backend)
             >> std::quoted(loaded.residue_hash)
             >> loaded.auto_baseline
             >> loaded.legal
             >> loaded.duplicate
             >> loaded.exact_match
             >> loaded.production_probe_exact
             >> loaded.transform_words
             >> loaded.transform_mwords
             >> loaded.bpw
             >> loaded.setup_ms
             >> loaded.us_median
             >> loaded.us_min
             >> loaded.us_max
             >> loaded.production_us_per_iter))
        return false;

    if (loaded.status == "ERROR" || loaded.status == "TIMEOUT")
        return false;
    r = std::move(loaded);
    return true;
}

void atomic_write_mpz(const std::filesystem::path& path,
                      const mpz_class& value) {
    std::filesystem::create_directories(path.parent_path());
    auto tmp = path;
    tmp += ".tmp";
    const std::string tmp_name = tmp.string();
    std::FILE* out = std::fopen(tmp_name.c_str(), "wb");
    if (!out) throw std::runtime_error("cannot create Bench3 AUTO residue checkpoint");

    const std::size_t bytes = mpz_out_raw(out, value.get_mpz_t());
    const int flush_rc = std::fflush(out);
    const int close_rc = std::fclose(out);
    if (bytes == 0 || flush_rc != 0 || close_rc != 0) {
        std::filesystem::remove(tmp);
        throw std::runtime_error("cannot write Bench3 AUTO residue checkpoint");
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp);
        throw std::runtime_error("cannot publish Bench3 AUTO residue checkpoint");
    }
}

bool read_mpz_checkpoint(const std::filesystem::path& path,
                         mpz_class& value) {
    const std::string name = path.string();
    std::FILE* in = std::fopen(name.c_str(), "rb");
    if (!in) return false;
    const std::size_t bytes = mpz_inp_raw(value.get_mpz_t(), in);
    const int trailing = std::fgetc(in);
    const bool ok = bytes != 0 && trailing == EOF && std::ferror(in) == 0;
    std::fclose(in);
    return ok;
}

void checkpoint_plan(const std::filesystem::path& partial_dir,
                     const std::string& context,
                     std::size_t index,
                     std::uint32_t exponent,
                     const PlanResult& row) {
    if (row.status == "ERROR") return;
    const auto path = plan_checkpoint_path(partial_dir, index);
    atomic_write_plan_checkpoint(path, context, row);
    std::cout << "BENCH3_PLAN_CHECKPOINT p=" << exponent
              << " index=" << index
              << " request=" << row.request
              << " status=" << row.status
              << " path=" << path.string() << std::endl;
}

struct PlanWorkerProcessResult {
    bool timed_out = false;
    bool interrupted = false;
    int exit_code = -1;
    std::string error;
};

std::string bench3_self_executable() {
#ifdef _WIN32
    std::vector<char> path(32768, '\0');
    const DWORD n = GetModuleFileNameA(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (n == 0 || n >= path.size()) return {};
    return std::string(path.data(), n);
#else
    return util::selfExecutablePath();
#endif
}

std::vector<std::string> worker_arguments(
        const ModeConfig& cfg,
        std::size_t device_index,
        std::uint32_t exponent,
        std::size_t plan_index,
        const std::string& request,
        const std::string& resolved_plan,
        const std::filesystem::path& partial_dir) {
    const std::string exe = bench3_self_executable();
    if (exe.empty()) throw std::runtime_error("cannot determine Bench3 worker executable");

    std::vector<std::string> args{
        exe,
        "-bench3",
        "-bench3-mode", cfg.name,
        "-d", std::to_string(device_index),
        "-bench3-worker",
        "-bench3-worker-exp", std::to_string(exponent),
        "-bench3-worker-index", std::to_string(plan_index),
        "-bench3-worker-request", request,
        "-bench3-worker-partial", partial_dir.string()
    };
    if (!resolved_plan.empty()) {
        args.push_back("-bench3-worker-plan");
        args.push_back(resolved_plan);
    }
    return args;
}

#ifndef _WIN32
void terminate_posix_worker(pid_t pid) {
    if (pid <= 0) return;
    ::kill(pid, SIGTERM);
    for (int attempt = 0; attempt < 20; ++attempt) {
        int status = 0;
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid || waited < 0) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}
#endif

PlanWorkerProcessResult run_plan_worker_process(
        const io::CliOptions& options,
        const ModeConfig& cfg,
        std::size_t device_index,
        std::uint32_t exponent,
        std::size_t plan_index,
        const std::string& request,
        const std::string& resolved_plan,
        const std::filesystem::path& partial_dir) {
    PlanWorkerProcessResult result;
    const auto args = worker_arguments(
        cfg, device_index, exponent, plan_index,
        request, resolved_plan, partial_dir);
    const auto started = std::chrono::steady_clock::now();
    const auto progress_path = plan_progress_path(partial_dir, plan_index);
    {
        std::error_code ec;
        std::filesystem::remove(progress_path, ec);
    }
    auto next_heartbeat = started + std::chrono::seconds(10);
    const auto maybe_heartbeat = [&]() {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_heartbeat) return;
        next_heartbeat = now + std::chrono::seconds(10);

        const double elapsed = std::chrono::duration<double>(now - started).count();
        std::string phase = "starting";
        double phase_elapsed = 0.0;
        read_plan_progress(progress_path, phase, phase_elapsed);

        std::cout << std::fixed << std::setprecision(3)
                  << "BENCH3_PLAN_HEARTBEAT p=" << exponent
                  << " index=" << plan_index
                  << " request=" << request
                  << " phase=" << phase
                  << " worker_elapsed_s=" << phase_elapsed
                  << " elapsed_s=" << elapsed;
        if (options.bench3_plan_timeout_sec == 0) {
            std::cout << " timeout_remaining_s=disabled";
        } else {
            const double remaining = std::max(
                0.0,
                static_cast<double>(options.bench3_plan_timeout_sec) - elapsed);
            std::cout << " timeout_remaining_s=" << remaining;
        }
        std::cout << std::endl;
    };

    std::cout << "BENCH3_PLAN_WORKER_START p=" << exponent
              << " index=" << plan_index
              << " request=" << request
              << " plan=" << (resolved_plan.empty() ? "AUTO" : resolved_plan)
              << " timeout_s=" << options.bench3_plan_timeout_sec
              << std::endl;

#ifdef _WIN32
    std::string command = util::winCommandLine(args);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(
            args.front().c_str(),
            command.data(),
            nullptr,
            nullptr,
            FALSE,
            0,
            nullptr,
            nullptr,
            &si,
            &pi)) {
        result.error = "CreateProcess failed with error " +
                       std::to_string(static_cast<unsigned long>(GetLastError()));
        return result;
    }
    CloseHandle(pi.hThread);

    for (;;) {
        const DWORD wait_rc = WaitForSingleObject(pi.hProcess, 100);
        if (wait_rc == WAIT_OBJECT_0) break;
        if (wait_rc != WAIT_TIMEOUT) {
            result.error = "WaitForSingleObject failed with error " +
                           std::to_string(static_cast<unsigned long>(GetLastError()));
            TerminateProcess(pi.hProcess, 125);
            WaitForSingleObject(pi.hProcess, 5000);
            break;
        }

        if (g_stop) {
            result.interrupted = true;
            TerminateProcess(pi.hProcess, 130);
            WaitForSingleObject(pi.hProcess, 5000);
            break;
        }

        maybe_heartbeat();

        if (options.bench3_plan_timeout_sec != 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - started).count();
            if (elapsed >= options.bench3_plan_timeout_sec) {
                result.timed_out = true;
                TerminateProcess(pi.hProcess, 124);
                WaitForSingleObject(pi.hProcess, 5000);
                break;
            }
        }
    }

    DWORD exit_code = 0;
    if (GetExitCodeProcess(pi.hProcess, &exit_code)) {
        result.exit_code = static_cast<int>(exit_code);
        constexpr DWORD kWindowsConsoleControlExit = 0xC000013Au;
        if (!result.timed_out && exit_code == kWindowsConsoleControlExit) {
            result.interrupted = true;
            result.error =
                "worker terminated by Windows console control event (0xC000013A)";
        }
    } else if (result.error.empty()) {
        result.error = "GetExitCodeProcess failed";
    }
    CloseHandle(pi.hProcess);
#else
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args)
        argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int spawn_rc = posix_spawn(
        &pid, args.front().c_str(), nullptr, nullptr, argv.data(), environ);
    if (spawn_rc != 0) {
        result.error = std::string("posix_spawn failed: ") + std::strerror(spawn_rc);
        return result;
    }

    int status = 0;
    for (;;) {
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
            else if (WIFSIGNALED(status)) result.exit_code = 128 + WTERMSIG(status);
            else result.exit_code = 125;
            break;
        }
        if (waited < 0) {
            if (errno == EINTR) continue;
            result.error = std::string("waitpid failed: ") + std::strerror(errno);
            terminate_posix_worker(pid);
            break;
        }

        if (g_stop) {
            result.interrupted = true;
            terminate_posix_worker(pid);
            result.exit_code = 130;
            break;
        }

        maybe_heartbeat();

        if (options.bench3_plan_timeout_sec != 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - started).count();
            if (elapsed >= options.bench3_plan_timeout_sec) {
                result.timed_out = true;
                terminate_posix_worker(pid);
                result.exit_code = 124;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << std::fixed << std::setprecision(3)
              << "BENCH3_PLAN_WORKER_EXIT p=" << exponent
              << " index=" << plan_index
              << " request=" << request
              << " exit_code=" << result.exit_code
              << " timeout=" << (result.timed_out ? "yes" : "no")
              << " interrupted=" << (result.interrupted ? "yes" : "no")
              << " elapsed_s=" << elapsed
              << std::endl;
    {
        std::error_code ec;
        std::filesystem::remove(progress_path, ec);
        auto tmp = progress_path;
        tmp += ".tmp";
        std::filesystem::remove(tmp, ec);
    }
    return result;
}

int run_plan_worker(const io::CliOptions& options,
                    const ModeConfig& cfg,
                    std::size_t device_index) {
    if (!options.bench3_worker ||
        options.bench3_worker_exponent == 0 ||
        options.bench3_worker_request.empty() ||
        options.bench3_worker_partial_dir.empty()) {
        std::cerr << "invalid internal Bench3 worker request\n";
        return 2;
    }

    const auto exponent = options.bench3_worker_exponent;
    const auto plan_index =
        static_cast<std::size_t>(options.bench3_worker_index);
    const auto partial_dir =
        std::filesystem::path(options.bench3_worker_partial_dir);
    const std::string context =
        checkpoint_context(exponent, device_index, cfg);
    const auto progress_path = plan_progress_path(partial_dir, plan_index);
    {
        std::error_code ec;
        std::filesystem::remove(progress_path, ec);
    }

    std::cout << "BENCH3_WORKER p=" << exponent
              << " index=" << plan_index
              << " request=" << options.bench3_worker_request
              << " plan=" << (options.bench3_worker_resolved_plan.empty()
                                  ? "AUTO"
                                  : options.bench3_worker_resolved_plan)
              << std::endl;

    PlanResult row;
    if (options.bench3_worker_request == "AUTO") {
        mpz_class auto_residue;
        row = measure_plan(
            exponent, device_index, cfg, true, "AUTO", "",
            nullptr, &auto_residue, &progress_path);
        if (row.status == "PASS")
            atomic_write_mpz(partial_dir / "auto-residue.raw", auto_residue);
    } else {
        mpz_class auto_residue;
        if (!read_mpz_checkpoint(
                partial_dir / "auto-residue.raw", auto_residue)) {
            std::cerr << "Bench3 worker cannot read authoritative AUTO residue\n";
            return 3;
        }
        row = measure_plan(
            exponent, device_index, cfg, false,
            options.bench3_worker_request,
            options.bench3_worker_resolved_plan,
            &auto_residue, nullptr, &progress_path);
    }

    checkpoint_plan(
        partial_dir, context, plan_index, exponent, row);
    std::cout << "BENCH3_WORKER_RESULT p=" << exponent
              << " index=" << plan_index
              << " request=" << row.request
              << " status=" << row.status
              << std::endl;
    return row.status == "ERROR" ? 4 : 0;
}

PlanResult measure_plan_isolated(
        const io::CliOptions& options,
        const ModeConfig& cfg,
        std::size_t device_index,
        std::uint32_t exponent,
        std::size_t plan_index,
        const std::string& request,
        const std::string& resolved_plan,
        const std::filesystem::path& partial_dir) {
    const std::string context =
        checkpoint_context(exponent, device_index, cfg);
    const auto checkpoint =
        plan_checkpoint_path(partial_dir, plan_index);

    const PlanWorkerProcessResult process = run_plan_worker_process(
        options, cfg, device_index, exponent, plan_index,
        request, resolved_plan, partial_dir);

    if (process.interrupted) {
        std::cout << "BENCH3_PLAN_INTERRUPTED p=" << exponent
                  << " index=" << plan_index
                  << " request=" << request;
        if (!process.error.empty())
            std::cout << " reason=" << process.error;
        std::cout << std::endl;
        throw std::runtime_error("bench3 interrupted");
    }

    if (process.timed_out) {
        PlanResult row;
        row.status = "TIMEOUT";
        row.error = "isolated plan exceeded " +
                    std::to_string(options.bench3_plan_timeout_sec) +
                    " seconds";
        row.request = request;
        row.resolved_plan = resolved_plan;
        row.active_plan = resolved_plan;
        row.auto_baseline = request == "AUTO";
        checkpoint_plan(
            partial_dir, context, plan_index, exponent, row);
        std::cout << "BENCH3_PLAN_TIMEOUT p=" << exponent
                  << " index=" << plan_index
                  << " request=" << request
                  << " timeout_s=" << options.bench3_plan_timeout_sec
                  << std::endl;
        return row;
    }

    if (!process.error.empty() || process.exit_code != 0) {
        PlanResult row;
        row.status = "ERROR";
        row.request = request;
        row.resolved_plan = resolved_plan;
        row.active_plan = resolved_plan;
        row.auto_baseline = request == "AUTO";
        row.error = !process.error.empty()
            ? process.error
            : "isolated plan worker exited with code " +
              std::to_string(process.exit_code);
        return row;
    }

    PlanResult row;
    if (!read_plan_checkpoint(checkpoint, context, row)) {
        row.status = "ERROR";
        row.request = request;
        row.resolved_plan = resolved_plan;
        row.active_plan = resolved_plan;
        row.auto_baseline = request == "AUTO";
        row.error = "isolated plan worker exited without a valid checkpoint";
    }
    return row;
}

PointResult benchmark_point(std::uint32_t exponent,
                            std::size_t device_index,
                            const DeviceInfo& info,
                            const ModeConfig& cfg,
                            const io::CliOptions& options,
                            const std::filesystem::path& partial_dir) {
    PointResult point;
    point.exponent = exponent;
    point.device_index = device_index;
    point.device = info;

    const std::string context = checkpoint_context(exponent, device_index, cfg);
    const auto auto_plan_path = plan_checkpoint_path(partial_dir, 0);
    const auto auto_residue_path = partial_dir / "auto-residue.raw";

    mpz_class auto_residue;
    PlanResult baseline;
    bool resumed_baseline =
        read_plan_checkpoint(auto_plan_path, context, baseline) &&
        baseline.status == "PASS" &&
        baseline.auto_baseline &&
        baseline.request == "AUTO" &&
        read_mpz_checkpoint(auto_residue_path, auto_residue);

    if (resumed_baseline) {
        std::cout << "BENCH3_PLAN_RESUME p=" << exponent
                  << " index=0 request=AUTO status=" << baseline.status
                  << std::endl;
    } else {
        baseline = measure_plan_isolated(
            options, cfg, device_index, exponent, 0,
            "AUTO", "", partial_dir);
        if (baseline.status == "PASS" &&
            !read_mpz_checkpoint(auto_residue_path, auto_residue)) {
            baseline.status = "ERROR";
            baseline.error = "AUTO worker checkpoint is missing its exact residue";
        }
    }

    point.plans.push_back(baseline);
    if (baseline.status != "PASS") return point;

    std::set<std::string> seen;
    seen.insert(baseline.active_plan);

    const auto requests = candidate_requests();
    for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
        if (g_stop) throw std::runtime_error("bench3 interrupted");

        const std::string& request = requests[request_index];
        const std::size_t plan_index = request_index + 1;
        const auto checkpoint = plan_checkpoint_path(partial_dir, plan_index);

        PlanResult row;
        if (read_plan_checkpoint(checkpoint, context, row) &&
            row.request == request) {
            std::cout << "BENCH3_PLAN_RESUME p=" << exponent
                      << " index=" << plan_index
                      << " request=" << request
                      << " status=" << row.status << std::endl;
            if (!row.resolved_plan.empty() && row.status != "SKIPPED_DUPLICATE")
                seen.insert(row.resolved_plan);
            point.plans.push_back(row);
            continue;
        }

        std::size_t transform = 0;
        std::string resolved;
        std::string reason;
        row.request = request;
        row.auto_baseline = false;

        if (!aevum_engine_resolve_fft(
                exponent, request, &transform, &resolved, &reason)) {
            row.status = "SKIPPED_ILLEGAL";
            row.legal = false;
            row.error = reason.empty() ? "Aevum resolver rejected plan" : reason;
            checkpoint_plan(partial_dir, context, plan_index, exponent, row);
            point.plans.push_back(row);
            continue;
        }

        row.resolved_plan = resolved;
        if (seen.count(resolved) != 0) {
            row.status = "SKIPPED_DUPLICATE";
            row.duplicate = true;
            row.active_plan = resolved;
            row.transform_words = transform;
            checkpoint_plan(partial_dir, context, plan_index, exponent, row);
            point.plans.push_back(row);
            continue;
        }
        seen.insert(resolved);

        row = measure_plan_isolated(
            options, cfg, device_index, exponent, plan_index,
            request, resolved, partial_dir);
        point.plans.push_back(row);
    }

    const double auto_us = point.plans.front().production_us_per_iter;
    double best = auto_us;
    std::size_t best_index = 0;

    for (auto& row : point.plans) {
        row.best_exact = false;
        row.delta_vs_auto_pct = 0.0;
    }

    for (std::size_t i = 0; i < point.plans.size(); ++i) {
        auto& row = point.plans[i];
        if (row.status == "PASS" && row.exact_match &&
            row.production_probe_exact && row.production_us_per_iter > 0.0) {
            if (auto_us > 0.0)
                row.delta_vs_auto_pct =
                    100.0 * (auto_us / row.production_us_per_iter - 1.0);
            if (row.production_us_per_iter < best) {
                best = row.production_us_per_iter;
                best_index = i;
            }
        }
    }
    point.plans[best_index].best_exact = true;
    return point;
}

std::string row_json(const PlanResult& r) {
    std::ostringstream o;
    o << std::setprecision(12)
      << "{"
      << "\"status\":\"" << json_escape(r.status) << "\","
      << "\"error\":\"" << json_escape(r.error) << "\","
      << "\"request\":\"" << json_escape(r.request) << "\","
      << "\"resolved_plan\":\"" << json_escape(r.resolved_plan) << "\","
      << "\"active_plan\":\"" << json_escape(r.active_plan) << "\","
      << "\"backend\":\"" << json_escape(r.backend) << "\","
      << "\"auto_baseline\":" << (r.auto_baseline ? "true" : "false") << ","
      << "\"legal\":" << (r.legal ? "true" : "false") << ","
      << "\"duplicate\":" << (r.duplicate ? "true" : "false") << ","
      << "\"exact_match\":" << (r.exact_match ? "true" : "false") << ","
      << "\"production_probe_exact\":" << (r.production_probe_exact ? "true" : "false") << ","
      << "\"best_exact\":" << (r.best_exact ? "true" : "false") << ","
      << "\"residue_hash\":\"" << json_escape(r.residue_hash) << "\","
      << "\"transform_words\":" << r.transform_words << ","
      << "\"transform_mwords\":" << r.transform_mwords << ","
      << "\"bpw\":" << r.bpw << ","
      << "\"setup_ms\":" << r.setup_ms << ","
      << "\"us_per_iter_median\":" << r.us_median << ","
      << "\"us_per_iter_min\":" << r.us_min << ","
      << "\"us_per_iter_max\":" << r.us_max << ","
      << "\"production_prp_us_per_iter\":" << r.production_us_per_iter << ","
      << "\"delta_vs_auto_pct\":" << r.delta_vs_auto_pct
      << "}";
    return o.str();
}

std::string point_json(const PointResult& p, const ModeConfig& cfg) {
    std::ostringstream o;
    o << std::setprecision(12)
      << "{"
      << "\"schema_version\":\"" << kSchemaVersion << "\","
      << "\"record_type\":\"bench3_point\","
      << "\"mode\":\"" << json_escape(cfg.name) << "\","
      << "\"exponent\":" << p.exponent << ","
      << "\"device_index\":" << p.device_index << ","
      << "\"device_vendor\":\"" << json_escape(p.device.vendor) << "\","
      << "\"device_name\":\"" << json_escape(p.device.name) << "\","
      << "\"driver_version\":\"" << json_escape(p.device.driver) << "\","
      << "\"runtime_version\":\"" << json_escape(p.device.runtime) << "\","
      << "\"prmers_version\":\"" << json_escape(core::PRMERS_VERSION) << "\","
      << "\"source_sha\":\"" << json_escape(PRMERS_STRINGIFY(PRMERS_GIT_SHA)) << "\","
      << "\"compiler\":\"" << json_escape(compiler_string()) << "\","
      << "\"warmup_iterations\":" << cfg.warmup << ","
      << "\"timed_iterations_per_rep\":" << cfg.timed_iterations << ","
      << "\"repetitions\":" << cfg.repetitions << ","
      << "\"exact_iterations\":" << cfg.exact_iterations << ","
      << "\"plans\":[";
    for (std::size_t i = 0; i < p.plans.size(); ++i) {
        if (i) o << ',';
        o << row_json(p.plans[i]);
    }
    o << "]}";
    return o.str();
}

std::string csv_header() {
    return "schema_version,mode,exponent,device_index,device_vendor,device_name,"
           "status,error,request,resolved_plan,active_plan,backend,auto_baseline,"
           "legal,duplicate,exact_match,production_probe_exact,best_exact,residue_hash,"
           "transform_words,transform_mwords,bpw,setup_ms,us_per_iter_median,"
           "us_per_iter_min,us_per_iter_max,production_prp_us_per_iter,delta_vs_auto_pct";
}

std::string row_csv(const PointResult& p, const PlanResult& r, const ModeConfig& cfg) {
    std::ostringstream o;
    o << std::setprecision(12)
      << kSchemaVersion << ',' << csv_escape(cfg.name) << ','
      << p.exponent << ',' << p.device_index << ','
      << csv_escape(p.device.vendor) << ',' << csv_escape(p.device.name) << ','
      << csv_escape(r.status) << ',' << csv_escape(r.error) << ','
      << csv_escape(r.request) << ',' << csv_escape(r.resolved_plan) << ','
      << csv_escape(r.active_plan) << ',' << csv_escape(r.backend) << ','
      << (r.auto_baseline ? 1 : 0) << ',' << (r.legal ? 1 : 0) << ','
      << (r.duplicate ? 1 : 0) << ',' << (r.exact_match ? 1 : 0) << ','
      << (r.production_probe_exact ? 1 : 0) << ',' << (r.best_exact ? 1 : 0) << ','
      << csv_escape(r.residue_hash) << ',' << r.transform_words << ','
      << r.transform_mwords << ',' << r.bpw << ',' << r.setup_ms << ','
      << r.us_median << ',' << r.us_min << ',' << r.us_max << ','
      << r.production_us_per_iter << ',' << r.delta_vs_auto_pct;
    return o.str();
}

std::string row_text(std::uint32_t exponent, const PlanResult& r) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3)
      << "BENCH3 p=" << exponent
      << " request=" << r.request
      << " status=" << r.status
      << " backend=" << (r.backend.empty() ? "n/a" : r.backend)
      << " plan=" << (r.active_plan.empty() ? r.resolved_plan : r.active_plan)
      << " exact=" << (r.exact_match ? "yes" : "no")
      << " prod_exact=" << (r.production_probe_exact ? "yes" : "no")
      << " median_us=" << r.us_median
      << " prod_us=" << r.production_us_per_iter
      << " delta_auto_pct=" << r.delta_vs_auto_pct
      << " best=" << (r.best_exact ? "yes" : "no")
      << " residue=" << (r.residue_hash.empty() ? "n/a" : r.residue_hash);
    if (!r.error.empty()) o << " error=" << r.error;
    return o.str();
}

std::filesystem::path record_path(const std::filesystem::path& dir,
                                  std::size_t device,
                                  std::uint32_t exponent) {
    std::ostringstream name;
    name << "d" << device << "_p" << std::setw(10) << std::setfill('0')
         << exponent << ".record";
    return dir / name.str();
}

void atomic_write_record(const std::filesystem::path& path,
                         const std::string& json,
                         const std::vector<std::string>& csv,
                         const std::vector<std::string>& text) {
    std::filesystem::create_directories(path.parent_path());
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create bench3 record");
        out << "JSON\t" << json << '\n';
        for (const auto& row : csv) out << "CSV\t" << row << '\n';
        for (const auto& row : text) out << "TEXT\t" << row << '\n';
        out.flush();
        if (!out) throw std::runtime_error("cannot flush bench3 record");
    }
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(tmp);
        return;
    }
    std::filesystem::rename(tmp, path);
}

void replace_file(const std::filesystem::path& path, const std::string& content) {
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create " + tmp.string());
        out << content;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("cannot replace " + path.string());
}

void rebuild_outputs(const std::filesystem::path& out_dir,
                     const std::filesystem::path& record_dir,
                     const ModeConfig& cfg,
                     std::size_t device_index) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::exists(record_dir)) {
        for (const auto& e : std::filesystem::directory_iterator(record_dir))
            if (e.is_regular_file() && e.path().extension() == ".record")
                files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    std::vector<std::string> json_rows, csv_rows, text_rows;
    for (const auto& path : files) {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("JSON\t", 0) == 0) json_rows.push_back(line.substr(5));
            else if (line.rfind("CSV\t", 0) == 0) csv_rows.push_back(line.substr(4));
            else if (line.rfind("TEXT\t", 0) == 0) text_rows.push_back(line.substr(5));
        }
    }

    {
        std::ostringstream o;
        for (const auto& row : json_rows) o << row << '\n';
        replace_file(out_dir / "bench3.jsonl", o.str());
    }
    {
        std::ostringstream o;
        o << "[\n";
        for (std::size_t i = 0; i < json_rows.size(); ++i) {
            o << "  " << json_rows[i];
            if (i + 1 != json_rows.size()) o << ',';
            o << '\n';
        }
        o << "]\n";
        replace_file(out_dir / "bench3.json", o.str());
    }
    {
        std::ostringstream o;
        o << csv_header() << '\n';
        for (const auto& row : csv_rows) o << row << '\n';
        replace_file(out_dir / "bench3.csv", o.str());
    }
    {
        std::ostringstream o;
        o << "PrMers -bench3 schema=" << kSchemaVersion
          << " mode=" << cfg.name
          << " device=" << device_index
          << " completed_points=" << json_rows.size() << '\n';
        for (const auto& row : text_rows) o << row << '\n';
        replace_file(out_dir / "bench3.txt", o.str());
    }
}

} // namespace

int run(const io::CliOptions& options) {
    const ModeConfig cfg = mode_config(options.bench3_mode);
    if (options.device_id < 0) {
        std::cerr << "-bench3 requires a non-negative -d device index\n";
        return 2;
    }

    const std::size_t device_index =
        static_cast<std::size_t>(options.device_id);
    const std::filesystem::path out_dir =
        options.bench3_output.empty()
            ? std::filesystem::path("bench3-results")
            : std::filesystem::path(options.bench3_output);
    const auto record_dir = out_dir / "records";
    const auto partial_root = out_dir / "partial";

    std::signal(SIGINT, signal_handler);
#ifdef SIGTERM
    std::signal(SIGTERM, signal_handler);
#endif

    if (options.bench3_worker)
        return run_plan_worker(options, cfg, device_index);

    if (!options.bench3_resume) {
        std::error_code ec;
        std::filesystem::remove_all(record_dir, ec);
        std::filesystem::remove_all(partial_root, ec);
        std::filesystem::remove(out_dir / "bench3.jsonl", ec);
        std::filesystem::remove(out_dir / "bench3.json", ec);
        std::filesystem::remove(out_dir / "bench3.csv", ec);
        std::filesystem::remove(out_dir / "bench3.txt", ec);
    }
    std::filesystem::create_directories(record_dir);

    const DeviceInfo info = device_info(device_index);
    std::cout << "PrMers -bench3 schema=" << kSchemaVersion
              << " mode=" << cfg.name
              << " device=" << device_index
              << " GPU=\"" << info.vendor << ' ' << info.name << "\""
              << " points=" << cfg.exponents.size()
              << " resume=" << (options.bench3_resume ? "on" : "off")
              << " plan_timeout_s=" << options.bench3_plan_timeout_sec
              << " out=" << out_dir.string() << '\n';

    for (std::size_t i = 0; i < cfg.exponents.size(); ++i) {
        if (g_stop) {
            rebuild_outputs(out_dir, record_dir, cfg, device_index);
            return 130;
        }

        const auto exponent = cfg.exponents[i];
        const auto path = record_path(record_dir, device_index, exponent);
        const auto partial_dir =
            partial_point_path(partial_root, device_index, exponent);
        if (options.bench3_resume && std::filesystem::exists(path)) {
            std::error_code ec;
            std::filesystem::remove_all(partial_dir, ec);
            std::cout << "BENCH3_RESUME_SKIP [" << (i + 1) << '/'
                      << cfg.exponents.size() << "] p=" << exponent << '\n';
            continue;
        }

        std::cout << "BENCH3_POINT [" << (i + 1) << '/'
                  << cfg.exponents.size() << "] p=" << exponent
                  << " AUTO+bounded-alternatives" << std::endl;

        PointResult point;
        try {
            point = benchmark_point(
                exponent, device_index, info, cfg, options, partial_dir);
        } catch (const std::exception& e) {
            if (g_stop) {
                rebuild_outputs(out_dir, record_dir, cfg, device_index);
                std::cerr << e.what() << "; completed Bench3 points remain resumable\n";
                return 130;
            }
            throw;
        }

        std::vector<std::string> csv;
        std::vector<std::string> text;
        for (const auto& row : point.plans) {
            csv.push_back(row_csv(point, row, cfg));
            text.push_back(row_text(exponent, row));
        }

        const bool retryable_point = std::any_of(
            point.plans.begin(), point.plans.end(),
            [](const PlanResult& row) {
                return row.status == "TIMEOUT" || row.status == "ERROR";
            });

        if (retryable_point) {
            std::filesystem::create_directories(partial_dir);

            replace_file(
                partial_dir / "point-status.json",
                point_json(point, cfg) + "\n");

            {
                std::ostringstream partial_csv;
                partial_csv << csv_header() << '\n';
                for (const auto& row : csv) partial_csv << row << '\n';
                replace_file(
                    partial_dir / "point-status.csv",
                    partial_csv.str());
            }
            {
                std::ostringstream partial_text;
                partial_text << "PrMers -bench3 partial point"
                             << " schema=" << kSchemaVersion
                             << " mode=" << cfg.name
                             << " device=" << device_index
                             << " exponent=" << exponent << '\n';
                for (const auto& row : text) partial_text << row << '\n';
                replace_file(
                    partial_dir / "point-status.txt",
                    partial_text.str());
            }

            rebuild_outputs(out_dir, record_dir, cfg, device_index);
            std::cout << "BENCH3_POINT_PARTIAL p=" << exponent
                      << " retryable=yes"
                      << " path=" << partial_dir.string()
                      << std::endl;
        } else {
            atomic_write_record(path, point_json(point, cfg), csv, text);
            rebuild_outputs(out_dir, record_dir, cfg, device_index);
            {
                std::error_code ec;
                std::filesystem::remove_all(partial_dir, ec);
            }
        }

        for (const auto& row : text) std::cout << row << '\n';
    }

    rebuild_outputs(out_dir, record_dir, cfg, device_index);
    std::cout << "BENCH3_COMPLETE schema=" << kSchemaVersion
              << " mode=" << cfg.name
              << " device=" << device_index
              << " json=" << (out_dir / "bench3.json").string()
              << " csv=" << (out_dir / "bench3.csv").string()
              << " text=" << (out_dir / "bench3.txt").string()
              << '\n';
    return 0;
}

} // namespace core::bench3
