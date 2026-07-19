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
bool aevum_engine_resolve_fft(uint32_t exponent,
                               const std::string& requested_spec,
                               std::size_t* transform_size,
                               std::string* resolved_spec = nullptr,
                               std::string* reason = nullptr);

engine* create_aevum_engine(uint32_t exponent,
                            std::size_t register_count,
                            std::size_t device,
                            bool verbose,
                            const std::string& fft_spec);
