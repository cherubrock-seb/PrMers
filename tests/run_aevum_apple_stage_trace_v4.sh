#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
DEVICE="${2:-0}"
STRICT="${3:-0}"
TMPBASE="${TMPDIR:-/tmp}"
SRC="$TMPBASE/aevum_apple_stage_trace_v4.cpp"
BIN="$TMPBASE/aevum_apple_stage_trace_v4"
LOG="$ROOT/aevum-apple-stage-trace-${STRICT}.log"

if [[ -f "$ROOT/third_party/aevum/build-engine/libaevum_engine.so" ]]; then
  AEVUM_ROOT="$ROOT/third_party/aevum"
elif [[ -f "$ROOT/build-engine/libaevum_engine.so" ]]; then
  AEVUM_ROOT="$ROOT"
else
  echo "Missing Aevum engine below: $ROOT" >&2
  exit 2
fi
ENGINE="$AEVUM_ROOT/build-engine/libaevum_engine.so"
TUNE="$AEVUM_ROOT"

if [[ "$STRICT" == "1" ]]; then
  export AEVUM_APPLE_STAGE_FINISH=1
else
  unset AEVUM_APPLE_STAGE_FINISH || true
fi

cat > "$SRC" <<'CPP'
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <class T>
T sym(void* lib, const char* name) {
    void* p = dlsym(lib, name);
    if (!p) throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<T>(p);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: " << argv[0]
                  << " /path/to/libaevum_engine.so [device=0] [tune_dir=. ]\n";
        return 2;
    }

    const char* library = argv[1];
    const uint32_t device = argc >= 3 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 0;
    const char* tune_dir = argc >= 4 ? argv[3] : ".";

    void* lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(dlerror());

    using handle_t = void*;
    using create_t = handle_t (*)(uint32_t, size_t, uint32_t, int, const char*, const char*);
    using destroy_t = void (*)(handle_t);
    using words_t = size_t (*)(handle_t);
    using set_t = int (*)(handle_t, size_t, uint32_t);
    using prepare_t = int (*)(handle_t, size_t, size_t);
    using square_t = int (*)(handle_t, size_t, uint32_t);
    using mul_t = int (*)(handle_t, size_t, size_t, uint32_t);
    using get_t = int (*)(handle_t, size_t, uint32_t*, size_t);
    using trace_t = int (*)(handle_t, size_t, uint64_t*, size_t);
    using version_t = const char* (*)();
    using error_t = const char* (*)();

    const auto create = sym<create_t>(lib, "aevum_engine_create");
    const auto destroy = sym<destroy_t>(lib, "aevum_engine_destroy");
    const auto word_count = sym<words_t>(lib, "aevum_engine_word_count");
    const auto set_u32 = sym<set_t>(lib, "aevum_engine_set_u32");
    const auto prepare = sym<prepare_t>(lib, "aevum_engine_prepare");
    const auto square_mul = sym<square_t>(lib, "aevum_engine_square_mul");
    const auto mul = sym<mul_t>(lib, "aevum_engine_mul");
    const auto get_words = sym<get_t>(lib, "aevum_engine_get_words");
    const auto debug_trace = sym<trace_t>(lib, "aevum_engine_debug_square_trace");
    const auto version = sym<version_t>(lib, "aevum_engine_version");
    const auto last_error = sym<error_t>(lib, "aevum_engine_last_error");

    handle_t h = create(859553u, 4, device, 1, "", tune_dir);
    if (!h) throw std::runtime_error(last_error());

    std::vector<uint32_t> words(word_count(h));
    auto require = [&](bool ok, const char* op) {
        if (!ok) throw std::runtime_error(std::string(op) + ": " + last_error());
    };
    auto check = [&](size_t reg, uint32_t expected, const char* label) {
        std::fill(words.begin(), words.end(), 0);
        require(get_words(h, reg, words.data(), words.size()) != 0, "get_words");
        bool good = words[0] == expected;
        for (size_t i = 1; i < words.size(); ++i) good = good && words[i] == 0;
        std::cout << (good ? "PASS" : "FAIL") << "  " << std::left << std::setw(34)
                  << label << " expected=" << expected << " actual_low=" << words[0] << "\n";
        if (!good) {
            std::cout << "FIRST_DIVERGENCE=" << label << "\n";
            destroy(h);
            dlclose(lib);
            std::exit(1);
        }
    };

    std::cout << "Aevum version: " << version() << "\n";
    std::cout << "TRACE_MODE=" << (std::getenv("AEVUM_APPLE_STAGE_FINISH") ? "STRICT_FINISH" : "NORMAL") << "\n";

    require(set_u32(h, 0, 5) != 0, "set 5 for trace");
    std::array<uint64_t, 12> trace{};
    require(debug_trace(h, 0, trace.data(), trace.size()) != 0, "debug_square_trace");
    static const char* labels[12] = {
        "TRACE_FFTP_GF31", "TRACE_FFTP_GF61",
        "TRACE_MIDDLEIN_GF31", "TRACE_MIDDLEIN_GF61",
        "TRACE_TAILSQUARE_GF31", "TRACE_TAILSQUARE_GF61",
        "TRACE_MIDDLEOUT_GF31", "TRACE_MIDDLEOUT_GF61",
        "TRACE_FFTW_GF31", "TRACE_FFTW_GF61",
        "TRACE_FINAL_WORD_HASH", "TRACE_FINAL_LOW"
    };
    for (size_t i = 0; i < trace.size(); ++i) {
        std::cout << labels[i] << "=0x" << std::hex << std::setw(16)
                  << std::setfill('0') << trace[i] << std::dec << std::setfill(' ') << "\n";
    }

    require(set_u32(h, 0, 5) != 0, "set 5");
    require(square_mul(h, 0, 1) != 0, "square 5");
    check(0, 25, "square: 5^2");

    require(set_u32(h, 1, 3) != 0, "set base 3");
    require(prepare(h, 1, 1) != 0, "prepare base 3");
    require(set_u32(h, 0, 5) != 0, "set 5 again");
    require(mul(h, 0, 1, 1) != 0, "prepared mul 5*3");
    check(0, 15, "prepared multiply: 5*3");

    require(set_u32(h, 0, 5) != 0, "set 5 step");
    require(square_mul(h, 0, 1) != 0, "square step");
    require(mul(h, 0, 1, 1) != 0, "prepared mul step");
    check(0, 75, "P-1 step: 5^2*3");

    require(set_u32(h, 2, 7) != 0, "set base 7");
    require(prepare(h, 2, 2) != 0, "prepare base 7");
    require(set_u32(h, 0, 5) != 0, "set 5 third");
    require(mul(h, 0, 2, 1) != 0, "prepared mul 5*7");
    check(0, 35, "prepared multiply: 5*7");

    require(set_u32(h, 0, 3) != 0, "set 3");
    require(square_mul(h, 0, 3) != 0, "square_mul factor 3");
    check(0, 27, "square_mul small factor: 3^2*3");

    destroy(h);
    dlclose(lib);
    std::cout << "ALL_AEVUM_APPLE_STAGE_TRACE_PROBES_PASSED\n";
    return 0;
}
CPP

c++ -std=c++20 -O2 "$SRC" -o "$BIN"
rm -rf "$ROOT/.aevum-kernel-cache"
cd "$ROOT"
"$BIN" "$ENGINE" "$DEVICE" "$TUNE" 2>&1 | tee "$LOG"
