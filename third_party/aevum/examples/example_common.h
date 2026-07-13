#pragma once

#include "../src/EngineApi.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

inline void require_ok(int ok, const char* operation) {
    if (ok) return;
    const char* error = aevum_engine_last_error();
    throw std::runtime_error(std::string(operation) + ": " + (error ? error : "unknown error"));
}

inline uint32_t read_small(aevum_engine_handle engine, size_t reg) {
    std::vector<uint32_t> words(aevum_engine_word_count(engine));
    require_ok(aevum_engine_get_words(engine, reg, words.data(), words.size()), "get_words");
    for (size_t i = 1; i < words.size(); ++i) {
        if (words[i] != 0) throw std::runtime_error("value does not fit in one word");
    }
    return words.empty() ? 0 : words[0];
}

inline uint32_t parse_device(int argc, char** argv) {
    return argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 0;
}

inline const char* parse_tune_dir(int argc, char** argv) {
    return argc > 2 ? argv[2] : ".";
}
