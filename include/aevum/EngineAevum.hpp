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

// "radix1k=<4|8> <source>" for the 1K radix the Aevum plugin uses on PrMers device `device` (see
// aevum/Radix1kPolicy.hpp).  `cuda_plugin` is true when the CUDA plugin runs the engine, which is always NVIDIA.
std::string aevum_radix1k_policy_detail(std::size_t device, bool cuda_plugin = false);
