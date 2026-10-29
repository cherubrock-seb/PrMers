#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

class engine;

bool aevum_engine_supports_auto_fft(uint32_t exponent, std::string* reason = nullptr);
bool aevum_engine_resolve_auto_fft(uint32_t exponent,
                                    std::size_t* transform_size,
                                    std::string* resolved_spec = nullptr,
                                    std::string* reason = nullptr);
bool aevum_engine_resolve_factor_safe_fft(uint32_t exponent,
                                           uint32_t factor,
                                           std::size_t* transform_size,
                                           std::string* resolved_spec = nullptr,
                                           std::string* reason = nullptr);
bool aevum_engine_resolve_fft(uint32_t exponent,
                               const std::string& requested_spec,
                               std::size_t* transform_size,
                               std::string* resolved_spec = nullptr,
                               std::string* reason = nullptr);

engine* create_aevum_engine(uint32_t exponent,
                            std::size_t register_count,
                            std::size_t device,
                            bool verbose,
                            const std::string& fft_spec,
                            std::uint32_t workload);

// FFT plan an engine returned by create_aevum_engine() actually runs, as reported by the plugin.  Empty if `eng`
// is not an Aevum engine or the plugin cannot report it.  This can differ from the plan the resolver predicted
// before creation, because autotune, tune.txt replay and device profiles are applied when the engine is created.
std::string aevum_engine_active_plan(const engine* eng);

// Identity of the Aevum plugin PrMers loads (version string, path, size and modification time of the library, and of
// the CUDA plugin next to it when present).  A plugin swapped through AEVUM_ENGINE_LIB, or rebuilt in place, gives a
// different string.  "unavailable" if no plugin can be loaded.
std::string aevum_plugin_identity();
