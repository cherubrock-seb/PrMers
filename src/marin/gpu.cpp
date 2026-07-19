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

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace {
std::mutex backend_mutex;
engine::gpu_backend backend = engine::gpu_backend::marin;
engine::gpu_workload workload = engine::gpu_workload::generic;
std::string aevum_fft_spec;
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
        std::cout << "[Backend Auto] " << aevum_workload_name(selected_workload) << ": "
                  << (decision.use_aevum ? "Aevum" : "Marin")
                  << " selected (" << decision.detail << ")." << std::endl;
        selected = decision.use_aevum ? gpu_backend::aevum : gpu_backend::marin;
        publish("Auto", decision.use_aevum ? "Aevum" : "Marin", decision.detail,
                decision.aevum_transform, decision.fft_spec);
    }

    if (selected == gpu_backend::aevum) {
        if (configured != gpu_backend::auto_select) {
            std::cout << "[Backend Aevum] " << aevum_workload_name(selected_workload)
                      << ": forced by -aevum" << std::endl;
        }
        if (selected_workload == gpu_workload::pm1_ultralowmem) {
            throw std::runtime_error(
                "Aevum is incompatible with the one-register P-1 ultra-low-memory fast3 algorithm");
        }
        std::string resolved_fft = fft_spec;
        std::size_t resolved_transform = decision.aevum_transform;
        if (fft_spec.empty()) {
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
        engine* created = create_aevum_engine(p, reg_count, device, verbose, fft_spec);
        if (configured != gpu_backend::auto_select) {
            if (resolved_transform == 0) resolved_transform = created->get_size();
            publish("Forced Aevum", "Aevum", "selected by -aevum", resolved_transform, resolved_fft);
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
