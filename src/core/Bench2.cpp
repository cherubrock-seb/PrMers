#include "core/Bench2.hpp"

#include "aevum/EngineAevum.hpp"
#include "core/Version.hpp"
#include "marin/engine.h"
#include "opencl/Context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmpxx.h>

#ifndef PRMERS_GIT_SHA
#define PRMERS_GIT_SHA unknown
#endif
#define PRMERS_STRINGIFY_IMPL(value) #value
#define PRMERS_STRINGIFY(value) PRMERS_STRINGIFY_IMPL(value)

namespace core::bench2 {
namespace {

constexpr const char* kSchemaVersion = "bench2.v1";
constexpr std::size_t kPrpRegisterCount = 8;

volatile std::sig_atomic_t g_stop = 0;

void signal_handler(int) {
    g_stop = 1;
}

struct ModeConfig {
    std::string name;
    std::size_t warmup = 0;
    std::size_t timed_iterations = 0;
    std::size_t repetitions = 0;
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

struct Record {
    std::string status = "PASS";
    std::string error;

    std::uint32_t exponent = 0;
    std::size_t device_index = 0;

    std::string backend;
    std::string active_plan;
    std::string selection_source =
        "engine::configure_gpu_backend(auto_select,PRP)+engine::create_gpu";
    std::string selection_reason = "production-auto";

    std::size_t transform_words = 0;
    double transform_mwords = 0.0;
    double bpw = 0.0;

    double setup_ms = 0.0;
    double us_median = 0.0;
    double us_min = 0.0;
    double us_max = 0.0;
    double dispersion_pct = 0.0;
    double iterations_per_second = 0.0;
    double estimated_full_prp_seconds = 0.0;

    std::uint64_t gerbicz_block = 0;
    std::uint64_t gerbicz_checkpasslevel = 0;
    std::uint64_t gerbicz_full_check_interval = 0;
    double gerbicz_boundary_us = 0.0;
    double gerbicz_full_check_us = 0.0;
    double gerbicz_amortized_us_per_iter = 0.0;
    double production_prp_us_per_iter = 0.0;
    double production_prp_iterations_per_second = 0.0;
    double production_prp_estimated_seconds = 0.0;
    bool production_prp_probe_exact = false;

    std::size_t register_bytes = 0;
    std::size_t checkpoint_bytes = 0;

    DeviceInfo device;
};

std::string getenv_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : std::string(fallback);
}

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
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos)
        return value;
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value) {
        if (c == '"') escaped.push_back('"');
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

std::string cl_string(cl_device_id device, cl_device_info what) {
    std::size_t bytes = 0;
    if (clGetDeviceInfo(device, what, 0, nullptr, &bytes) != CL_SUCCESS ||
        bytes == 0)
        return "unknown";
    std::string value(bytes, '\0');
    if (clGetDeviceInfo(device, what, bytes, value.data(), nullptr) != CL_SUCCESS)
        return "unknown";
    while (!value.empty() && value.back() == '\0') value.pop_back();
    for (char& c : value) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    return value;
}

template <typename T>
T cl_scalar(cl_device_id device, cl_device_info what) {
    T value{};
    if (clGetDeviceInfo(device, what, sizeof(value), &value, nullptr) != CL_SUCCESS)
        return T{};
    return value;
}

DeviceInfo device_info(const std::size_t device_index) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS ||
        platform_count == 0)
        throw std::runtime_error("No OpenCL platform found");

    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    std::vector<cl_device_id> devices;
    for (const cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) !=
                CL_SUCCESS ||
            device_count == 0)
            continue;

        const std::size_t old_size = devices.size();
        devices.resize(old_size + device_count);
        clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, device_count,
            devices.data() + old_size, nullptr);
    }

    if (device_index >= devices.size())
        throw std::runtime_error("OpenCL GPU device index is out of range");

    const cl_device_id device = devices[device_index];

    DeviceInfo info;
    info.vendor = cl_string(device, CL_DEVICE_VENDOR);
    info.name = cl_string(device, CL_DEVICE_NAME);
    info.driver = cl_string(device, CL_DRIVER_VERSION);
    info.runtime = cl_string(device, CL_DEVICE_VERSION);
    info.compute_units =
        cl_scalar<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    info.clock_mhz =
        cl_scalar<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    info.global_mem_bytes =
        static_cast<std::uint64_t>(
            cl_scalar<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE));
    info.local_mem_bytes =
        static_cast<std::uint64_t>(
            cl_scalar<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE));
    info.max_workgroup =
        cl_scalar<std::size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    return info;
}

std::vector<std::uint32_t> full_grid() {
    // Stable campaign grid plus the known ~197M selector boundary neighborhood.
    return {
         37156667u,
         58000003u,
         77232917u,
         82589933u,
        100000007u,
        130000007u,
        145000007u,
        150000007u,
        160000003u,
        170000003u,
        175000003u,
        180000017u,
        190000003u,
        196999937u,
        196999969u,
        197000003u,
        200000033u,
        210000017u,
        220000013u,
        230000003u,
        250000013u,
        280000027u,
        300000007u,
        320000077u,
        340000019u,
        360000019u,
        400000009u,
        500000003u,
        600000001u
    };
}

ModeConfig mode_config(const std::string& mode) {
    if (mode == "quick") {
        return {
            "quick", 24, 64, 3,
            {
                 37156667u,
                 77232917u,
                130000007u,
                170000003u,
                196999969u,
                197000003u,
                210000017u,
                220000013u,
                300000007u,
                600000001u
            }
        };
    }

    if (mode == "standard") {
        return {
            "standard", 48, 192, 5,
            {
                 37156667u,
                 58000003u,
                 77232917u,
                 82589933u,
                100000007u,
                130000007u,
                145000007u,
                150000007u,
                160000003u,
                170000003u,
                180000017u,
                196999969u,
                197000003u,
                200000033u,
                210000017u,
                220000013u,
                250000013u,
                300000007u,
                400000009u,
                600000001u
            }
        };
    }

    if (mode == "dense") {
        return {"dense", 64, 384, 5, full_grid()};
    }

    if (mode == "full") {
        return {"full", 128, 1024, 7, full_grid()};
    }

    throw std::runtime_error(
        "invalid -bench2-mode '" + mode +
        "' (expected quick, standard, dense or full)");
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1u) != 0u) return values[n / 2];
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

struct ProductionPrpTiming {
    std::uint64_t block = 0;
    std::uint64_t checkpasslevel = 0;
    std::uint64_t full_check_interval = 0;
    double boundary_us = 0.0;
    double full_check_us = 0.0;
    double amortized_us_per_iter = 0.0;
    double production_us_per_iter = 0.0;
    double production_iterations_per_second = 0.0;
    double production_estimated_seconds = 0.0;
    bool exact = false;
};

ProductionPrpTiming measure_production_prp_timing(
    engine& eng,
    const std::uint32_t exponent,
    const double square_us_per_iter) {
    const engine::Reg R0 = 0, R1 = 1, R2 = 2, R3 = 3;
    const engine::Reg R4 = 4, R5 = 5, RBASE = 6, RTMP = 7;

    ProductionPrpTiming out;

    const std::uint64_t legacy_B =
        static_cast<std::uint64_t>(
            std::sqrt(static_cast<double>(exponent)));
    out.block = std::min<std::uint64_t>(1000u, legacy_B);
    if (out.block == 0)
        throw std::runtime_error(
            "bench2 production PRP Gerbicz block is zero");

    out.checkpasslevel =
        static_cast<std::uint64_t>(
            (1000.0 * 600.0) / static_cast<double>(out.block));
    if (out.checkpasslevel == 0) out.checkpasslevel = 1;
    out.full_check_interval = out.block * out.checkpasslevel;

    // Exact fresh ordinary-PRP state.  Advancing by modB iterations reaches
    // the first real production Gerbicz boundary; forcing a check there has
    // the same replay/readback operation sequence as the periodic full check.
    eng.set(R1, 1u);
    eng.set(R0, 3u);
    eng.copy(R4, R0);
    eng.copy(R5, R1);
    eng.set(RBASE, 3u);
    eng.set_multiplicand(RTMP, RBASE);

    const std::uint64_t modB =
        exponent % out.block == 0 ? out.block : exponent % out.block;
    for (std::uint64_t z = 0; z < modB; ++z)
        eng.square_mul(R0);
    eng.sync();

    const auto boundary_begin = std::chrono::steady_clock::now();
    eng.copy(R3, R1);
    eng.set_multiplicand(R2, R0);
    eng.mul(R1, R2);
    eng.sync();
    const auto boundary_end = std::chrono::steady_clock::now();

    out.boundary_us =
        std::chrono::duration<double, std::micro>(
            boundary_end - boundary_begin).count();

    const auto full_begin = std::chrono::steady_clock::now();

    const std::uint64_t loop_count =
        out.block > modB ? out.block - modB - 1 : 0;
    for (std::uint64_t z = 0; z < loop_count; ++z)
        eng.square_mul(R3);

    if (exponent % out.block == 0)
        eng.mul(R3, RTMP);
    else
        eng.square_mul(R3, 3u);

    for (std::uint64_t z = 0; z < modB; ++z)
        eng.square_mul(R3);

    mpz_t z0, z1;
    mpz_inits(z0, z1, nullptr);
    eng.get_mpz(z0, R3);
    eng.get_mpz(z1, R1);

    const mpz_class Mp = (mpz_class(1) << exponent) - 1;
    out.exact =
        ((mpz_class)z0 % Mp) == ((mpz_class)z1 % Mp);

    mpz_clears(z0, z1, nullptr);

    if (!out.exact)
        throw std::runtime_error(
            "bench2 production PRP Gerbicz probe mismatch");

    // Successful production-check path.
    eng.copy(R4, R0);
    eng.copy(R5, R1);
    eng.sync();

    const auto full_end = std::chrono::steady_clock::now();
    out.full_check_us =
        std::chrono::duration<double, std::micro>(
            full_end - full_begin).count();

    out.amortized_us_per_iter =
        out.boundary_us / static_cast<double>(out.block) +
        out.full_check_us /
            static_cast<double>(out.full_check_interval);

    out.production_us_per_iter =
        square_us_per_iter + out.amortized_us_per_iter;

    if (out.production_us_per_iter > 0.0) {
        out.production_iterations_per_second =
            1'000'000.0 / out.production_us_per_iter;
        out.production_estimated_seconds =
            out.production_us_per_iter *
            static_cast<double>(exponent) / 1'000'000.0;
    }

    return out;
}

std::string csv_header() {
    return
        "schema_version,status,error,mode,exponent,device_index,"
        "device_vendor,device_name,driver_version,runtime_version,"
        "compute_units,clock_mhz,global_mem_bytes,local_mem_bytes,max_workgroup,"
        "prmers_version,source_sha,compiler,build_flags,workload,register_count,"
        "backend,selection_source,selection_reason,active_plan,"
        "transform_words,transform_mwords,bpw,setup_ms,warmup_iterations,"
        "timed_iterations_per_rep,repetitions,us_per_iter_median,"
        "us_per_iter_min,us_per_iter_max,dispersion_pct,iterations_per_second,"
        "estimated_full_prp_seconds,register_bytes,checkpoint_bytes,"
        "gerbicz_block,gerbicz_checkpasslevel,gerbicz_full_check_interval,"
        "gerbicz_boundary_us,gerbicz_full_check_us,"
        "gerbicz_amortized_us_per_iter,production_prp_us_per_iter,"
        "production_prp_iterations_per_second,"
        "production_prp_estimated_seconds,production_prp_probe_exact,"
        "timing_scope,queue_sync,exactness_state,profile_state";
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

std::string record_json(const Record& r, const ModeConfig& cfg) {
    const std::string source_sha =
        getenv_or(
            "PRMERS_BENCH2_SOURCE_SHA", PRMERS_STRINGIFY(PRMERS_GIT_SHA));
    const std::string build_flags =
        getenv_or("PRMERS_BENCH2_BUILD_FLAGS", "project-default");

    std::ostringstream o;
    o << std::setprecision(12);
    o << "{"
      << "\"schema_version\":\"" << kSchemaVersion << "\","
      << "\"record_type\":\"benchmark_point\","
      << "\"status\":\"" << json_escape(r.status) << "\","
      << "\"error\":\"" << json_escape(r.error) << "\","
      << "\"mode\":\"" << json_escape(cfg.name) << "\","
      << "\"exponent\":" << r.exponent << ","
      << "\"device_index\":" << r.device_index << ","
      << "\"device_vendor\":\"" << json_escape(r.device.vendor) << "\","
      << "\"device_name\":\"" << json_escape(r.device.name) << "\","
      << "\"driver_version\":\"" << json_escape(r.device.driver) << "\","
      << "\"runtime_version\":\"" << json_escape(r.device.runtime) << "\","
      << "\"compute_units\":" << r.device.compute_units << ","
      << "\"clock_mhz\":" << r.device.clock_mhz << ","
      << "\"global_mem_bytes\":" << r.device.global_mem_bytes << ","
      << "\"local_mem_bytes\":" << r.device.local_mem_bytes << ","
      << "\"max_workgroup\":" << r.device.max_workgroup << ","
      << "\"prmers_version\":\"" << json_escape(core::PRMERS_VERSION) << "\","
      << "\"source_sha\":\"" << json_escape(source_sha) << "\","
      << "\"compiler\":\"" << json_escape(compiler_string()) << "\","
      << "\"build_flags\":\"" << json_escape(build_flags) << "\","
      << "\"workload\":\"PRP\","
      << "\"register_count\":" << kPrpRegisterCount << ","
      << "\"backend\":\"" << json_escape(r.backend) << "\","
      << "\"selection_source\":\"" << json_escape(r.selection_source) << "\","
      << "\"selection_reason\":\"" << json_escape(r.selection_reason) << "\","
      << "\"active_plan\":\"" << json_escape(r.active_plan) << "\","
      << "\"transform_words\":" << r.transform_words << ","
      << "\"transform_mwords\":" << r.transform_mwords << ","
      << "\"bpw\":" << r.bpw << ","
      << "\"setup_ms\":" << r.setup_ms << ","
      << "\"warmup_iterations\":" << cfg.warmup << ","
      << "\"timed_iterations_per_rep\":" << cfg.timed_iterations << ","
      << "\"repetitions\":" << cfg.repetitions << ","
      << "\"us_per_iter_median\":" << r.us_median << ","
      << "\"us_per_iter_min\":" << r.us_min << ","
      << "\"us_per_iter_max\":" << r.us_max << ","
      << "\"dispersion_pct\":" << r.dispersion_pct << ","
      << "\"iterations_per_second\":" << r.iterations_per_second << ","
      << "\"estimated_full_prp_seconds\":" << r.estimated_full_prp_seconds << ","
      << "\"register_bytes\":" << r.register_bytes << ","
      << "\"checkpoint_bytes\":" << r.checkpoint_bytes << ","
      << "\"gerbicz_block\":" << r.gerbicz_block << ","
      << "\"gerbicz_checkpasslevel\":" << r.gerbicz_checkpasslevel << ","
      << "\"gerbicz_full_check_interval\":" << r.gerbicz_full_check_interval << ","
      << "\"gerbicz_boundary_us\":" << r.gerbicz_boundary_us << ","
      << "\"gerbicz_full_check_us\":" << r.gerbicz_full_check_us << ","
      << "\"gerbicz_amortized_us_per_iter\":"
      << r.gerbicz_amortized_us_per_iter << ","
      << "\"production_prp_us_per_iter\":"
      << r.production_prp_us_per_iter << ","
      << "\"production_prp_iterations_per_second\":"
      << r.production_prp_iterations_per_second << ","
      << "\"production_prp_estimated_seconds\":"
      << r.production_prp_estimated_seconds << ","
      << "\"production_prp_probe_exact\":"
      << (r.production_prp_probe_exact ? "true" : "false") << ","
      << "\"timing_scope\":\"steady-state square hot path; setup/JIT/backend probe excluded\","
      << "\"queue_sync\":\"engine::sync after warmup and every timed batch\","
      << "\"exactness_state\":\"NOT_VALIDATED_IN_BENCH2_TIMING\","
      << "\"profile_state\":\"NOT_COLLECTED_IN_TIMING_PHASE\""
      << "}";
    return o.str();
}

std::string record_csv(const Record& r, const ModeConfig& cfg) {
    const std::string source_sha =
        getenv_or(
            "PRMERS_BENCH2_SOURCE_SHA", PRMERS_STRINGIFY(PRMERS_GIT_SHA));
    const std::string build_flags =
        getenv_or("PRMERS_BENCH2_BUILD_FLAGS", "project-default");

    std::ostringstream o;
    o << std::setprecision(12);
    o
      << csv_escape(kSchemaVersion) << ','
      << csv_escape(r.status) << ','
      << csv_escape(r.error) << ','
      << csv_escape(cfg.name) << ','
      << r.exponent << ','
      << r.device_index << ','
      << csv_escape(r.device.vendor) << ','
      << csv_escape(r.device.name) << ','
      << csv_escape(r.device.driver) << ','
      << csv_escape(r.device.runtime) << ','
      << r.device.compute_units << ','
      << r.device.clock_mhz << ','
      << r.device.global_mem_bytes << ','
      << r.device.local_mem_bytes << ','
      << r.device.max_workgroup << ','
      << csv_escape(core::PRMERS_VERSION) << ','
      << csv_escape(source_sha) << ','
      << csv_escape(compiler_string()) << ','
      << csv_escape(build_flags) << ','
      << "PRP,"
      << kPrpRegisterCount << ','
      << csv_escape(r.backend) << ','
      << csv_escape(r.selection_source) << ','
      << csv_escape(r.selection_reason) << ','
      << csv_escape(r.active_plan) << ','
      << r.transform_words << ','
      << r.transform_mwords << ','
      << r.bpw << ','
      << r.setup_ms << ','
      << cfg.warmup << ','
      << cfg.timed_iterations << ','
      << cfg.repetitions << ','
      << r.us_median << ','
      << r.us_min << ','
      << r.us_max << ','
      << r.dispersion_pct << ','
      << r.iterations_per_second << ','
      << r.estimated_full_prp_seconds << ','
      << r.register_bytes << ','
      << r.checkpoint_bytes << ','
      << r.gerbicz_block << ','
      << r.gerbicz_checkpasslevel << ','
      << r.gerbicz_full_check_interval << ','
      << r.gerbicz_boundary_us << ','
      << r.gerbicz_full_check_us << ','
      << r.gerbicz_amortized_us_per_iter << ','
      << r.production_prp_us_per_iter << ','
      << r.production_prp_iterations_per_second << ','
      << r.production_prp_estimated_seconds << ','
      << (r.production_prp_probe_exact ? 1 : 0) << ','
      << csv_escape("steady-state square hot path; setup/JIT/backend probe excluded") << ','
      << csv_escape("engine::sync after warmup and every timed batch") << ','
      << csv_escape("NOT_VALIDATED_IN_BENCH2_TIMING") << ','
      << csv_escape("NOT_COLLECTED_IN_TIMING_PHASE");
    return o.str();
}

std::string record_text(const Record& r, const ModeConfig& cfg) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(3)
      << "BENCH2"
      << " schema=" << kSchemaVersion
      << " mode=" << cfg.name
      << " status=" << r.status
      << " p=" << r.exponent
      << " device=" << r.device_index
      << " backend=" << (r.backend.empty() ? "unknown" : r.backend)
      << " plan=" << (r.active_plan.empty() ? "unknown" : r.active_plan)
      << " words=" << r.transform_words
      << " Mwords=" << r.transform_mwords
      << " bpw=" << r.bpw
      << " setup_ms=" << r.setup_ms
      << " median_us=" << r.us_median
      << " min_us=" << r.us_min
      << " max_us=" << r.us_max
      << " dispersion_pct=" << r.dispersion_pct
      << " iter_s=" << r.iterations_per_second
      << " ETA=" << r.estimated_full_prp_seconds << "s"
      << " prod_us=" << r.production_prp_us_per_iter
      << " prod_iter_s=" << r.production_prp_iterations_per_second
      << " prod_ETA=" << r.production_prp_estimated_seconds << "s"
      << " gl_B=" << r.gerbicz_block
      << " gl_exact=" << (r.production_prp_probe_exact ? "yes" : "no");
    if (!r.error.empty()) o << " error=" << r.error;
    return o.str();
}

std::filesystem::path record_path(const std::filesystem::path& record_dir,
                                  const std::size_t device,
                                  const std::uint32_t exponent) {
    std::ostringstream name;
    name << "d" << device << "_p" << std::setw(10) << std::setfill('0')
         << exponent << ".record";
    return record_dir / name.str();
}

void atomic_write_record(const std::filesystem::path& path,
                         const std::string& json,
                         const std::string& csv,
                         const std::string& text) {
    std::filesystem::create_directories(path.parent_path());

    auto tmp = path;
    tmp += ".tmp";

    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot create bench2 record " + tmp.string());
        out << "JSON\t" << json << '\n'
            << "CSV\t" << csv << '\n'
            << "TEXT\t" << text << '\n';
        out.flush();
        if (!out)
            throw std::runtime_error("cannot flush bench2 record " + tmp.string());
    }

    if (std::filesystem::exists(path)) {
        std::filesystem::remove(tmp);
        return;
    }

    std::filesystem::rename(tmp, path);
}

void rebuild_aggregate_outputs(const std::filesystem::path& out_dir,
                               const std::filesystem::path& record_dir,
                               const ModeConfig& cfg,
                               const std::size_t device_index) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::exists(record_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(record_dir)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".record")
                files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    std::vector<std::string> json_rows;
    std::vector<std::string> csv_rows;
    std::vector<std::string> text_rows;

    for (const auto& path : files) {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("JSON\t", 0) == 0)
                json_rows.push_back(line.substr(5));
            else if (line.rfind("CSV\t", 0) == 0)
                csv_rows.push_back(line.substr(4));
            else if (line.rfind("TEXT\t", 0) == 0)
                text_rows.push_back(line.substr(5));
        }
    }

    const auto replace_file = [](const std::filesystem::path& path,
                                 const std::string& content) {
        auto tmp = path;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) throw std::runtime_error("cannot create " + tmp.string());
            out << content;
            out.flush();
            if (!out) throw std::runtime_error("cannot flush " + tmp.string());
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
        if (ec) throw std::runtime_error(
            "cannot replace " + path.string() + ": " + ec.message());
    };

    {
        std::ostringstream out;
        for (const auto& row : json_rows) out << row << '\n';
        replace_file(out_dir / "bench2.jsonl", out.str());
    }

    {
        std::ostringstream out;
        out << "[\n";
        for (std::size_t i = 0; i < json_rows.size(); ++i) {
            out << "  " << json_rows[i];
            if (i + 1 != json_rows.size()) out << ',';
            out << '\n';
        }
        out << "]\n";
        replace_file(out_dir / "bench2.json", out.str());
    }

    {
        std::ostringstream out;
        out << csv_header() << '\n';
        for (const auto& row : csv_rows) out << row << '\n';
        replace_file(out_dir / "bench2.csv", out.str());
    }

    {
        std::ostringstream out;
        out << "PrMers -bench2\n"
            << "schema=" << kSchemaVersion
            << " mode=" << cfg.name
            << " device=" << device_index
            << " records=" << text_rows.size() << '\n';
        for (const auto& row : text_rows) out << row << '\n';
        replace_file(out_dir / "bench2.txt", out.str());
    }
}

Record benchmark_point(const std::uint32_t exponent,
                       const std::size_t device_index,
                       const DeviceInfo& info,
                       const ModeConfig& cfg) {
    Record r;
    r.exponent = exponent;
    r.device_index = device_index;
    r.device = info;

    const auto setup_begin = std::chrono::steady_clock::now();

    try {
        engine::configure_gpu_backend(
            engine::gpu_backend::auto_select,
            "",
            engine::gpu_workload::prp);

        std::unique_ptr<engine> eng(
            engine::create_gpu(
                exponent,
                kPrpRegisterCount,
                device_index,
                false));

        if (!eng)
            throw std::runtime_error("production engine::create_gpu returned null");

        eng->sync();

        const auto setup_end = std::chrono::steady_clock::now();
        r.setup_ms =
            std::chrono::duration<double, std::milli>(
                setup_end - setup_begin).count();

        r.backend = eng->is_aevum_backend() ? "Aevum" : "Marin";
        r.selection_reason =
            std::string("production auto selected ") + r.backend;
        r.transform_words = eng->get_size();
        r.transform_mwords =
            static_cast<double>(r.transform_words) / (1024.0 * 1024.0);
        if (r.transform_words != 0)
            r.bpw =
                static_cast<double>(exponent) /
                static_cast<double>(r.transform_words);

        if (eng->is_aevum_backend()) {
            r.active_plan = aevum_engine_active_plan(eng.get());
            if (r.active_plan.empty())
                r.active_plan = "plugin-active-plan-unavailable";
        } else {
            r.active_plan = "marin-native";
        }

        r.register_bytes = eng->get_register_data_size();
        r.checkpoint_bytes = eng->get_checkpoint_size();

        eng->set(0, 3u);
        for (std::size_t i = 0; i < cfg.warmup; ++i) {
            if (g_stop) throw std::runtime_error("bench2 interrupted");
            eng->square_mul(0);
        }
        eng->sync();

        std::vector<double> samples;
        samples.reserve(cfg.repetitions);

        for (std::size_t rep = 0; rep < cfg.repetitions; ++rep) {
            if (g_stop) throw std::runtime_error("bench2 interrupted");

            eng->set(0, 3u);
            eng->sync();

            const auto begin = std::chrono::steady_clock::now();

            for (std::size_t i = 0; i < cfg.timed_iterations; ++i) {
                if (g_stop) throw std::runtime_error("bench2 interrupted");
                eng->square_mul(0);
            }

            eng->sync();

            const auto end = std::chrono::steady_clock::now();

            const double us =
                std::chrono::duration<double, std::micro>(
                    end - begin).count() /
                static_cast<double>(cfg.timed_iterations);
            samples.push_back(us);
        }

        r.us_median = median(samples);
        r.us_min = *std::min_element(samples.begin(), samples.end());
        r.us_max = *std::max_element(samples.begin(), samples.end());
        if (r.us_median > 0.0) {
            r.dispersion_pct =
                (r.us_max - r.us_min) / r.us_median * 100.0;
            r.iterations_per_second = 1'000'000.0 / r.us_median;
            r.estimated_full_prp_seconds =
                r.us_median * static_cast<double>(exponent) / 1'000'000.0;
        }

        const ProductionPrpTiming production =
            measure_production_prp_timing(
                *eng, exponent, r.us_median);

        r.gerbicz_block = production.block;
        r.gerbicz_checkpasslevel = production.checkpasslevel;
        r.gerbicz_full_check_interval =
            production.full_check_interval;
        r.gerbicz_boundary_us = production.boundary_us;
        r.gerbicz_full_check_us = production.full_check_us;
        r.gerbicz_amortized_us_per_iter =
            production.amortized_us_per_iter;
        r.production_prp_us_per_iter =
            production.production_us_per_iter;
        r.production_prp_iterations_per_second =
            production.production_iterations_per_second;
        r.production_prp_estimated_seconds =
            production.production_estimated_seconds;
        r.production_prp_probe_exact = production.exact;
    } catch (const std::exception& e) {
        if (g_stop) throw;
        r.status = "SKIPPED";
        r.error = e.what();
        const auto setup_end = std::chrono::steady_clock::now();
        r.setup_ms =
            std::chrono::duration<double, std::milli>(
                setup_end - setup_begin).count();
    }

    return r;
}

} // namespace

int run(const io::CliOptions& options) {
    const ModeConfig cfg = mode_config(options.bench2_mode);

    if (options.device_id < 0) {
        std::cerr << "-bench2 requires a non-negative -d device index\n";
        return 2;
    }

    const std::size_t device_index =
        static_cast<std::size_t>(options.device_id);

    std::filesystem::path out_dir =
        options.bench2_output.empty()
            ? std::filesystem::path("bench2-results")
            : std::filesystem::path(options.bench2_output);

    const std::filesystem::path record_dir = out_dir / "records";

    if (!options.bench2_resume) {
        std::error_code ec;
        std::filesystem::remove_all(record_dir, ec);
        std::filesystem::remove(out_dir / "bench2.jsonl", ec);
        std::filesystem::remove(out_dir / "bench2.json", ec);
        std::filesystem::remove(out_dir / "bench2.csv", ec);
        std::filesystem::remove(out_dir / "bench2.txt", ec);
    }

    std::filesystem::create_directories(record_dir);

    const DeviceInfo info = device_info(device_index);

    std::cout
        << "PrMers -bench2"
        << " schema=" << kSchemaVersion
        << " mode=" << cfg.name
        << " device=" << device_index
        << " GPU=\"" << info.vendor << ' ' << info.name << "\""
        << " points=" << cfg.exponents.size()
        << " resume=" << (options.bench2_resume ? "on" : "off")
        << " out=" << out_dir.string()
        << '\n';

    std::signal(SIGINT, signal_handler);
#ifdef SIGTERM
    std::signal(SIGTERM, signal_handler);
#endif

    for (std::size_t i = 0; i < cfg.exponents.size(); ++i) {
        if (g_stop) {
            rebuild_aggregate_outputs(out_dir, record_dir, cfg, device_index);
            std::cerr << "bench2 interrupted; completed records are resumable from "
                      << record_dir.string() << '\n';
            return 130;
        }

        const std::uint32_t exponent = cfg.exponents[i];
        const auto path = record_path(record_dir, device_index, exponent);

        if (options.bench2_resume && std::filesystem::exists(path)) {
            std::cout << "BENCH2_RESUME_SKIP"
                      << " [" << (i + 1) << '/' << cfg.exponents.size() << ']'
                      << " p=" << exponent << '\n';
            continue;
        }

        std::cout << "BENCH2_POINT"
                  << " [" << (i + 1) << '/' << cfg.exponents.size() << ']'
                  << " p=" << exponent
                  << " setup+measure"
                  << std::endl;

        Record record;
        try {
            record = benchmark_point(exponent, device_index, info, cfg);
        } catch (const std::exception& e) {
            if (g_stop) {
                rebuild_aggregate_outputs(
                    out_dir, record_dir, cfg, device_index);
                std::cerr << e.what()
                          << "; completed records remain resumable\n";
                return 130;
            }
            throw;
        }

        const std::string json = record_json(record, cfg);
        const std::string csv = record_csv(record, cfg);
        const std::string text = record_text(record, cfg);

        atomic_write_record(path, json, csv, text);
        rebuild_aggregate_outputs(out_dir, record_dir, cfg, device_index);

        std::cout << text << std::endl;
    }

    rebuild_aggregate_outputs(out_dir, record_dir, cfg, device_index);

    std::cout
        << "BENCH2_COMPLETE"
        << " schema=" << kSchemaVersion
        << " mode=" << cfg.name
        << " device=" << device_index
        << " json=" << (out_dir / "bench2.json").string()
        << " csv=" << (out_dir / "bench2.csv").string()
        << " text=" << (out_dir / "bench2.txt").string()
        << '\n';

    return 0;
}

} // namespace core::bench2
