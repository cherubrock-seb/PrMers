/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "marin/engine_gpu.h"
#include "aevum/AutoPolicy.hpp"
#include "aevum/EngineAevum.hpp"
#include "marin/ibdwt.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::mutex backend_mutex;
engine::gpu_backend backend = engine::gpu_backend::marin;
engine::gpu_workload workload = engine::gpu_workload::generic;
std::string aevum_fft_spec;

std::string aevum_radix1k_policy_detail() {
    const char* value = std::getenv("AEVUM_RADIX1K");
    if (value && std::string(value) == "8")
        return "radix1k=8 explicit-override";
    return "radix1k=4 safe-default";
}

enum class BackendProbeMode { off, automatic, retune };

BackendProbeMode backend_probe_mode() {
    const char* raw = std::getenv("PRMERS_BACKEND_AUTO");
    if (!raw || !*raw) return BackendProbeMode::automatic;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "0" || value == "off") return BackendProbeMode::off;
    if (value == "retune" || value == "force-retune" || value == "force")
        return BackendProbeMode::retune;
    return BackendProbeMode::automatic;
}

double backend_probe_min_speedup() {
    constexpr double fallback = 1.03;
    const char* raw = std::getenv("PRMERS_BACKEND_AUTO_MIN_SPEEDUP");
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || value < 1.0 || value > 2.0) return fallback;
    return value;
}

std::string cl_string(cl_device_id device, cl_device_info what) {
    std::size_t bytes = 0;
    if (clGetDeviceInfo(device, what, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0)
        return "unknown";
    std::string value(bytes, '\0');
    if (clGetDeviceInfo(device, what, bytes, value.data(), nullptr) != CL_SUCCESS)
        return "unknown";
    while (!value.empty() && value.back() == '\0') value.pop_back();
    for (char& c : value) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return value;
}

std::string backend_probe_key(const std::uint32_t exponent,
                              const std::size_t device) {
    try {
        ocl::platform platform;
        if (device >= platform.get_device_count()) return {};
        const cl_device_id d = platform.get_device(device);
        std::ostringstream out;
        out << "schema=2"
            << "|prmers=" << core::PRMERS_VERSION
            << "|p=" << exponent
            << "|device-index=" << device
            << "|vendor=" << cl_string(d, CL_DEVICE_VENDOR)
            << "|name=" << cl_string(d, CL_DEVICE_NAME)
            << "|driver=" << cl_string(d, CL_DRIVER_VERSION)
            << "|runtime=" << cl_string(d, CL_DEVICE_VERSION);
        static const char* flags[] = {
            "AEVUM_RADIX1K", "AEVUM_TYPE4_MULTI_Q", "AEVUM_PRP_MIDDLE1",
            "AEVUM_REG_LEAD_CACHE", "AEVUM_TUNE_DIR"
        };
        for (const char* name : flags) {
            if (const char* value = std::getenv(name); value && *value)
                out << '|' << name << '=' << value;
        }
        return out.str();
    } catch (...) {
        return {};
    }
}

std::uint64_t fnv1a64(const std::string& text) {
    std::uint64_t h = 1469598103934665603ull;
    for (char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

std::filesystem::path backend_probe_cache_path(const std::string& key) {
    std::filesystem::path dir;
    if (const char* explicit_dir = std::getenv("PRMERS_BACKEND_AUTO_CACHE_DIR");
        explicit_dir && *explicit_dir) {
        dir = explicit_dir;
    } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        dir = std::filesystem::path(xdg) / "prmers" / "backend-auto-v1";
    } else if (const char* home = std::getenv("HOME"); home && *home) {
        dir = std::filesystem::path(home) / ".cache" / "prmers" / "backend-auto-v1";
    } else {
        dir = ".prmers-backend-auto-v1";
    }
    std::ostringstream name;
    name << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(key);
    return dir / (name.str() + ".txt");
}

struct BackendProbeRecord {
    bool valid = false;
    bool exact = false;
    bool use_aevum = false;
    std::string plan;
    std::size_t transform = 0;
    double marin_ns = 0.0;
    double aevum_ns = 0.0;
};

BackendProbeRecord load_backend_probe_record(const std::string& key) {
    BackendProbeRecord record;
    if (key.empty()) return record;
    try {
        std::ifstream in(backend_probe_cache_path(key));
        std::string schema, stored_key, exact, winner, transform, marin_ns, aevum_ns;
        if (!std::getline(in, schema) || schema != "2") return record;
        if (!std::getline(in, stored_key) || stored_key != key) return record;
        if (!std::getline(in, exact)) return record;
        if (!std::getline(in, winner)) return record;
        if (!std::getline(in, record.plan)) return record;
        if (!std::getline(in, transform)) return record;
        if (!std::getline(in, marin_ns)) return record;
        if (!std::getline(in, aevum_ns)) return record;
        record.exact = exact == "1";
        record.use_aevum = winner == "Aevum";
        record.transform = static_cast<std::size_t>(std::stoull(transform));
        record.marin_ns = std::stod(marin_ns);
        record.aevum_ns = std::stod(aevum_ns);
        record.valid = record.marin_ns > 0.0 &&
                       (!record.exact || record.aevum_ns > 0.0);
    } catch (...) {
        return {};
    }
    return record;
}

void store_backend_probe_record(const std::string& key,
                                const BackendProbeRecord& record) {
    if (key.empty()) return;
    try {
        const auto path = backend_probe_cache_path(key);
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) return;
        auto temp = path;
        temp += ".tmp";
        {
            std::ofstream out(temp, std::ios::trunc);
            if (!out) return;
            out << "2\n"
                << key << '\n'
                << (record.exact ? "1" : "0") << '\n'
                << (record.use_aevum ? "Aevum" : "Marin") << '\n'
                << record.plan << '\n'
                << record.transform << '\n'
                << std::setprecision(17) << record.marin_ns << '\n'
                << std::setprecision(17) << record.aevum_ns << '\n';
            out.flush();
            if (!out) return;
        }
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temp, path, ec);
        if (ec) std::filesystem::remove(temp);
    } catch (...) {
    }
}

void prp_operation_probe_state(engine& eng, const std::uint32_t seed) {
    const engine::Reg R0 = 0, R1 = 1, R2 = 2, R3 = 3;
    const engine::Reg R4 = 4, R5 = 5, RBASE = 6, RTMP = 7;

    // Mirror the ordinary PRP register layout and the important Gerbicz
    // operation mix from RunPrpOrLlMarin, but in a short deterministic probe.
    eng.set(R1, 1u);
    eng.set(R0, seed);
    eng.copy(R4, R0);
    eng.copy(R5, R1);
    eng.set(RBASE, 3u);
    eng.set_multiplicand(RTMP, RBASE);

    for (unsigned block = 0; block < 8; ++block) {
        for (unsigned i = 0; i < 256; ++i) eng.square_mul(R0);

        eng.copy(R3, R1);
        eng.set_multiplicand(R2, R0);
        eng.mul(R1, R2);

        // Exercise both Gerbicz replay tails used by production PRP:
        // prepared multiply-by-base and fused square_mul(...,3).
        for (unsigned i = 0; i < 31; ++i) eng.square_mul(R3);
        if ((block & 1u) != 0u) eng.mul(R3, RTMP);
        else eng.square_mul(R3, 3u);
        for (unsigned i = 0; i < 17; ++i) eng.square_mul(R3);

        eng.copy(R4, R0);
        eng.copy(R5, R1);
    }
    eng.sync();
}

bool equal_register(engine& lhs, engine& rhs, const engine::Reg reg) {
    mpz_t a, b;
    mpz_inits(a, b, nullptr);
    lhs.get_mpz(a, reg);
    rhs.get_mpz(b, reg);
    const bool equal = mpz_cmp(a, b) == 0;
    mpz_clears(a, b, nullptr);
    return equal;
}

bool exact_prp_operation_probe(engine& marin, engine& aevum) {
    for (const std::uint32_t seed : {3u, 5u}) {
        prp_operation_probe_state(marin, seed);
        prp_operation_probe_state(aevum, seed);
        for (const engine::Reg reg : {engine::Reg(0), engine::Reg(1),
                                      engine::Reg(3), engine::Reg(4),
                                      engine::Reg(5)}) {
            if (!equal_register(marin, aevum, reg)) return false;
        }
    }
    return true;
}

double timed_square_sample(engine& eng) {
    constexpr unsigned warmup = 64;
    constexpr unsigned iterations = 512;
    eng.set(0, 3u);
    for (unsigned i = 0; i < warmup; ++i) eng.square_mul(0);
    eng.sync();
    const auto begin = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) eng.square_mul(0);
    eng.sync();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(end - begin).count() /
           static_cast<double>(iterations);
}

double median3(std::array<double, 3> values) {
    std::sort(values.begin(), values.end());
    return values[1];
}

double measured_square_ns(engine& eng) {
    std::array<double, 3> samples{};
    for (double& value : samples) value = timed_square_sample(eng);
    return median3(samples);
}

struct MeasuredBackendResult {
    std::unique_ptr<engine> selected;
    bool use_aevum = false;
    bool exact = false;
    std::string plan;
    std::size_t transform = 0;
    double marin_ns = 0.0;
    double aevum_ns = 0.0;
    std::string detail;
};

MeasuredBackendResult measured_lowrange_prp_backend(
    const std::uint32_t exponent,
    const std::size_t register_count,
    const std::size_t device,
    const bool verbose) {

    constexpr std::size_t quarantine_words = 524288u;
    const BackendProbeMode mode = backend_probe_mode();
    const double min_speedup = backend_probe_min_speedup();
    const std::string key = backend_probe_key(exponent, device);

    if (mode == BackendProbeMode::off) {
        MeasuredBackendResult out;
        out.selected = std::make_unique<engine_gpu>(
            exponent, register_count, device, verbose);
        out.detail = "runtime-compare=off => Marin";
        return out;
    }

    if (mode != BackendProbeMode::retune) {
        const BackendProbeRecord cached = load_backend_probe_record(key);
        if (cached.valid) {
            if (cached.use_aevum && cached.exact &&
                cached.transform > quarantine_words && !cached.plan.empty()) {
                std::size_t resolved_size = 0;
                std::string resolved_plan, reason;
                if (aevum_engine_resolve_fft(
                        exponent, cached.plan, &resolved_size, &resolved_plan, &reason) &&
                    resolved_size == cached.transform &&
                    resolved_size > quarantine_words) {
                    try {
                        MeasuredBackendResult out;
                        out.selected.reset(create_aevum_engine(
                            exponent, register_count, device, verbose,
                            cached.plan,
                            static_cast<std::uint32_t>(engine::gpu_workload::prp)));
                        out.use_aevum = true;
                        out.exact = true;
                        out.plan = cached.plan;
                        out.transform = cached.transform;
                        out.marin_ns = cached.marin_ns;
                        out.aevum_ns = cached.aevum_ns;
                        std::ostringstream detail;
                        detail << "cache=hit, exact=yes, winner=Aevum"
                               << ", plan=" << cached.plan
                               << ", Marin=" << std::fixed << std::setprecision(1)
                               << cached.marin_ns << " ns/it"
                               << ", Aevum=" << cached.aevum_ns << " ns/it";
                        out.detail = detail.str();
                        return out;
                    } catch (...) {
                    }
                }
            } else {
                MeasuredBackendResult out;
                out.selected = std::make_unique<engine_gpu>(
                    exponent, register_count, device, verbose);
                out.exact = cached.exact;
                out.plan = cached.plan;
                out.transform = cached.transform;
                out.marin_ns = cached.marin_ns;
                out.aevum_ns = cached.aevum_ns;
                std::ostringstream detail;
                detail << "cache=hit, winner=Marin"
                       << ", exact=" << (cached.exact ? "yes" : "no");
                if (!cached.plan.empty()) detail << ", best-Aevum-plan=" << cached.plan;
                if (cached.marin_ns > 0.0)
                    detail << ", Marin=" << std::fixed << std::setprecision(1)
                           << cached.marin_ns << " ns/it";
                if (cached.aevum_ns > 0.0)
                    detail << ", Aevum=" << cached.aevum_ns << " ns/it";
                out.detail = detail.str();
                return out;
            }
        }
    }

    static const std::array<const char*, 2> requested_plans = {
        "1:512:4:256:101",
        "1:512:4:256:202"
    };

    auto marin = std::make_unique<engine_gpu>(
        exponent, register_count, device, false);
    const double marin_ns = measured_square_ns(*marin);

    bool any_exact = false;
    double best_aevum_ns = 0.0;
    std::string best_plan;
    std::size_t best_transform = 0;

    for (const char* requested : requested_plans) {
        std::size_t transform = 0;
        std::string plan, reason;
        if (!aevum_engine_resolve_fft(
                exponent, requested, &transform, &plan, &reason) ||
            transform <= quarantine_words)
            continue;

        try {
            std::unique_ptr<engine> candidate(create_aevum_engine(
                exponent, register_count, device, false, plan,
                static_cast<std::uint32_t>(engine::gpu_workload::prp)));

            if (!exact_prp_operation_probe(*marin, *candidate)) {
                std::cout << "[Backend Auto Probe] PRP: rejected Aevum plan "
                          << plan << " (word-exact differential mismatch)." << std::endl;
                continue;
            }

            any_exact = true;
            const double aevum_ns = measured_square_ns(*candidate);
            std::cout << "[Backend Auto Probe] PRP: plan " << plan
                      << " exact, Marin=" << std::fixed << std::setprecision(1)
                      << marin_ns << " ns/it, Aevum=" << aevum_ns << " ns/it."
                      << std::endl;
            if (best_aevum_ns == 0.0 || aevum_ns < best_aevum_ns) {
                best_aevum_ns = aevum_ns;
                best_plan = plan;
                best_transform = transform;
            }
        } catch (const std::exception& e) {
            std::cout << "[Backend Auto Probe] PRP: rejected Aevum plan "
                      << requested << " (" << e.what() << ")." << std::endl;
        }
    }

    BackendProbeRecord record;
    record.exact = any_exact;
    record.plan = best_plan;
    record.transform = best_transform;
    record.marin_ns = marin_ns;
    record.aevum_ns = best_aevum_ns;

    const double speedup =
        (any_exact && best_aevum_ns > 0.0) ? marin_ns / best_aevum_ns : 0.0;
    record.use_aevum = any_exact && speedup >= min_speedup;
    record.valid = true;
    store_backend_probe_record(key, record);

    marin.reset();

    MeasuredBackendResult out;
    out.use_aevum = record.use_aevum;
    out.exact = record.exact;
    out.plan = record.plan;
    out.transform = record.transform;
    out.marin_ns = record.marin_ns;
    out.aevum_ns = record.aevum_ns;

    if (record.use_aevum) {
        try {
            out.selected.reset(create_aevum_engine(
                exponent, register_count, device, verbose, record.plan,
                static_cast<std::uint32_t>(engine::gpu_workload::prp)));
        } catch (const std::exception& e) {
            out.use_aevum = false;
            out.selected = std::make_unique<engine_gpu>(
                exponent, register_count, device, verbose);
            out.detail = std::string("probe winner Aevum became unavailable; Marin fallback: ") +
                         e.what();
            return out;
        }
    } else {
        out.selected = std::make_unique<engine_gpu>(
            exponent, register_count, device, verbose);
    }

    std::ostringstream detail;
    detail << "cache=miss"
           << ", exact=" << (record.exact ? "yes" : "no")
           << ", winner=" << (record.use_aevum ? "Aevum" : "Marin");
    if (!record.plan.empty()) detail << ", best-Aevum-plan=" << record.plan;
    detail << ", Marin=" << std::fixed << std::setprecision(1)
           << record.marin_ns << " ns/it";
    if (record.aevum_ns > 0.0) {
        detail << ", Aevum=" << record.aevum_ns << " ns/it"
               << ", speedup=" << std::setprecision(3) << speedup << "x"
               << ", threshold=" << min_speedup << "x";
    }
    out.detail = detail.str();
    return out;
}
}

void engine::configure_gpu_backend(const gpu_backend selected,
                                   const std::string& fft_spec,
                                   const gpu_workload selected_workload) {
    std::lock_guard<std::mutex> lock(backend_mutex);
    backend = selected;
    workload = selected_workload;
    aevum_fft_spec = fft_spec;
}

engine::gpu_backend engine::configured_gpu_backend() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    return backend;
}

const char* engine::configured_gpu_backend_name() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend == gpu_backend::aevum) return "Aevum";
    if (backend == gpu_backend::auto_select) return "Auto";
    return "Marin";
}

std::string engine::configured_aevum_fft_spec() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    return aevum_fft_spec;
}

engine* engine::create_gpu(const uint32_t p, const size_t reg_count, const size_t device, const bool verbose) {
    gpu_backend selected;
    gpu_workload selected_workload;
    std::string fft_spec;
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        selected = backend;
        selected_workload = workload;
        fft_spec = aevum_fft_spec;
    }

    const gpu_backend configured = selected;
    const std::size_t marin_transform = ibdwt::transform_size(p);
    auto publish = [&](const std::string& mode,
                       const std::string& active,
                       const std::string& detail,
                       const std::size_t aevum_transform,
                       const std::string& resolved_fft) {
        if (auto gui = ui::WebGuiServer::instance()) {
            gui->setBackendInfo(mode, active, aevum_workload_name(selected_workload), detail,
                                static_cast<uint64_t>(aevum_transform),
                                static_cast<uint64_t>(marin_transform), resolved_fft);
        }
    };

    AevumAutoDecision decision;
    if (selected == gpu_backend::auto_select) {
        decision = aevum_auto_decide(p, reg_count, selected_workload, fft_spec);

        if (decision.runtime_compare) {
            MeasuredBackendResult measured =
                measured_lowrange_prp_backend(p, reg_count, device, verbose);
            const std::string auto_detail =
                decision.detail + " | " + measured.detail +
                (measured.use_aevum ? " | " + aevum_radix1k_policy_detail() : "");
            std::cout << "[Backend Auto] " << aevum_workload_name(selected_workload)
                      << ": " << (measured.use_aevum ? "Aevum" : "Marin")
                      << " selected (" << auto_detail << ")." << std::endl;
            publish("Auto", measured.use_aevum ? "Aevum" : "Marin", auto_detail,
                    measured.transform, measured.plan);
            return measured.selected.release();
        }

        const std::string auto_detail = decision.use_aevum
            ? decision.detail + " | " + aevum_radix1k_policy_detail()
            : decision.detail;
        std::cout << "[Backend Auto] " << aevum_workload_name(selected_workload) << ": "
                  << (decision.use_aevum ? "Aevum" : "Marin")
                  << " selected (" << auto_detail << ")." << std::endl;
        selected = decision.use_aevum ? gpu_backend::aevum : gpu_backend::marin;
        publish("Auto", decision.use_aevum ? "Aevum" : "Marin", auto_detail,
                decision.aevum_transform, decision.fft_spec);
    }

    // Keep ordinary issue-#36 plugin-auto requests empty all the way into
    // Aevum. Only an explicit boundary-bridge decision is materialized as a
    // runtime FFT specification.
    std::string runtime_fft_spec = fft_spec;
    if (configured == gpu_backend::auto_select &&
        fft_spec.empty() &&
        decision.force_fft_spec) {
        runtime_fft_spec = decision.fft_spec;
    }

#if defined(__APPLE__)
    if (selected == gpu_backend::aevum &&
        selected_workload != gpu_workload::prp &&
        selected_workload != gpu_workload::ll) {
        const std::string reason =
            "Apple OpenCL 1.2 Aevum is currently validated only for PRP/LL stock FFT3161; "
            "ECM and P-1 use Marin because Aevum mixed/prepared multiplication failed invariant/Gerbicz validation";
        publish(configured == gpu_backend::auto_select ? "Auto" : "Forced Aevum rejected",
                configured == gpu_backend::auto_select ? "Marin" : "Unavailable",
                reason, 0, "");
        if (configured == gpu_backend::auto_select) {
            std::cout << "[Backend Auto] " << aevum_workload_name(selected_workload)
                      << ": Marin selected (" << reason << ")." << std::endl;
            return new engine_gpu(p, reg_count, device, verbose);
        }
        throw std::runtime_error(reason);
    }
#endif

    if (selected == gpu_backend::aevum) {
        if (configured != gpu_backend::auto_select) {
            std::cout << "[Backend Aevum] " << aevum_workload_name(selected_workload)
                      << ": forced by -aevum" << std::endl;
        }
        if (selected_workload == gpu_workload::pm1_ultralowmem) {
            throw std::runtime_error(
                "Aevum is incompatible with the one-register P-1 ultra-low-memory fast3 algorithm");
        }
        std::string resolved_fft = runtime_fft_spec;
        std::size_t resolved_transform = decision.aevum_transform;
        if (runtime_fft_spec.empty()) {
            std::string reason;
            if (resolved_transform == 0 &&
                !aevum_engine_resolve_auto_fft(p, &resolved_transform, &resolved_fft, &reason)) {
                if (configured == gpu_backend::auto_select) {
                    // Defensive only: the automatic policy normally rejects Aevum
                    // before reaching this branch when no FFT3161 plan exists.
                    std::cout << "[Backend Auto] " << aevum_workload_name(selected_workload)
                              << ": Marin selected (" << reason << ")." << std::endl;
                    publish("Auto", "Marin", reason, 0, "");
                    return new engine_gpu(p, reg_count, device, verbose);
                }
                publish("Forced Aevum rejected", "Unavailable", reason, 0, "");
                throw std::runtime_error(
                    std::string("Forced Aevum request cannot be satisfied for exponent ") +
                    std::to_string(p) + ": " + reason);
            }
        }

        engine* created = create_aevum_engine(p, reg_count, device, verbose, runtime_fft_spec, static_cast<std::uint32_t>(selected_workload));

        // The plugin chooses the plan when the engine is created (runtime autotune, tune.txt replay, device
        // profiles), which can differ from the policy's resolver preview that was logged and published above.
        // Report the plan that actually runs.
        const std::string active_plan = aevum_engine_active_plan(created);
        if (!active_plan.empty()) {
            resolved_fft = active_plan;
            if (configured == gpu_backend::auto_select && active_plan != decision.fft_spec) {
                std::cout << "[Backend Auto] " << aevum_workload_name(selected_workload)
                          << ": Aevum created FFT " << active_plan
                          << " (policy preview was " << decision.fft_spec << ")." << std::endl;
                publish("Auto", "Aevum",
                        "created-FFT=" + active_plan + ", transform=" + std::to_string(created->get_size()) +
                        " | policy preview: " + decision.detail + " | " + aevum_radix1k_policy_detail(),
                        created->get_size(), active_plan);
            }
        }

        // v100.17 safety quarantine for forced ordinary PRP as well as AUTO.
        // AUTO is rejected earlier by AutoPolicy; this guard prevents an
        // explicit -aevum / <=512K plan from accidentally producing a result
        // in the externally reproduced unsafe family. A deliberately larger
        // -aevum-fft plan remains available for diagnostic comparison.
        constexpr std::size_t kOrdinaryPrpQuarantineWords = 524288u;
        if (configured != gpu_backend::auto_select &&
            selected_workload == gpu_workload::prp &&
            reg_count == 8u &&
            created->get_size() <= kOrdinaryPrpQuarantineWords) {
            const std::size_t unsafe_transform = created->get_size();
            const std::string reason =
                "Aevum ordinary PRP <=512K is temporarily quarantined after an externally reproduced v100.16 residue mismatch; "
                "use -engine-marin, or a larger explicit -aevum-fft plan for diagnostic testing";
            delete created;
            publish("Forced Aevum rejected", "Unavailable", reason,
                    unsafe_transform, resolved_fft.empty() ? runtime_fft_spec : resolved_fft);
            throw std::runtime_error(reason);
        }

        if (configured != gpu_backend::auto_select) {
            if (resolved_transform == 0 || !active_plan.empty()) resolved_transform = created->get_size();
            publish("Forced Aevum", "Aevum",
                    "selected by -aevum | " + aevum_radix1k_policy_detail(),
                    resolved_transform, resolved_fft);
        }
        return created;
    }

    if (configured != gpu_backend::auto_select) {
        std::cout << "[Backend Marin] " << aevum_workload_name(selected_workload)
                  << ": selected by -engine-marin, compatibility route, or platform policy" << std::endl;
        publish("Marin", "Marin", "selected by -engine-marin, compatibility route, or platform policy", 0, "");
    }
    return new engine_gpu(p, reg_count, device, verbose);
}
