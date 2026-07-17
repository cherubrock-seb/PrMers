#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
DEVICE="${2:-0}"
ENGINE="$ROOT/third_party/aevum/build-engine/libaevum_engine.so"
TUNE="$ROOT/third_party/aevum"
if [[ ! -f "$ENGINE" ]]; then
  ENGINE="$ROOT/build-engine/libaevum_engine.so"
  TUNE="$ROOT"
fi
LOG="$ROOT/aevum-apple-plane-isolation.log"
TMPBASE="${TMPDIR:-/tmp}"
SRC="$TMPBASE/aevum_apple_plane_isolation.cpp"
BIN="$TMPBASE/aevum_apple_plane_isolation"

if [[ ! -f "$ENGINE" ]]; then
  echo "Missing engine: $ENGINE" >&2
  exit 2
fi

cat > "$SRC" <<'CPP'
#include <dlfcn.h>
#include <algorithm>
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
    using square_t = int (*)(handle_t, size_t, uint32_t);
    using get_t = int (*)(handle_t, size_t, uint32_t*, size_t);
    using version_t = const char* (*)();
    using error_t = const char* (*)();

    const auto create = sym<create_t>(lib, "aevum_engine_create");
    const auto destroy = sym<destroy_t>(lib, "aevum_engine_destroy");
    const auto word_count = sym<words_t>(lib, "aevum_engine_word_count");
    const auto set_u32 = sym<set_t>(lib, "aevum_engine_set_u32");
    const auto square_mul = sym<square_t>(lib, "aevum_engine_square_mul");
    const auto get_words = sym<get_t>(lib, "aevum_engine_get_words");
    const auto version = sym<version_t>(lib, "aevum_engine_version");
    const auto last_error = sym<error_t>(lib, "aevum_engine_last_error");

    struct Mode { const char* name; const char* spec; };
    const Mode modes[] = {
        {"HYBRID_GF31_GF61", "1:256:2:256:101"},
        {"GF31_ONLY",        "52:256:2:256:101"},
        {"GF61_ONLY",        "3:256:2:256:101"},
    };

    std::cout << "Aevum version: " << version() << "\n";
    int passed_modes = 0;

    for (const auto& mode : modes) {
        std::cout << "\nMODE=" << mode.name << " SPEC=" << mode.spec << "\n";
        handle_t h = create(859553u, 2, device, 1, mode.spec, tune_dir);
        if (!h) {
            std::cout << "MODE_CREATE_FAIL=" << mode.name << " ERROR=" << last_error() << "\n";
            continue;
        }
        std::vector<uint32_t> words(word_count(h));
        bool mode_ok = true;

        auto read_value = [&](uint32_t expected, const char* label) {
            std::fill(words.begin(), words.end(), 0);
            if (!get_words(h, 0, words.data(), words.size())) {
                std::cout << "FAIL  " << mode.name << " " << label
                          << " get_words error=" << last_error() << "\n";
                mode_ok = false;
                return;
            }
            bool good = words[0] == expected;
            for (size_t i = 1; i < words.size(); ++i) good = good && words[i] == 0;
            std::cout << (good ? "PASS" : "FAIL") << "  " << std::left << std::setw(24)
                      << mode.name << " " << std::setw(18) << label
                      << " expected=" << expected << " actual_low=" << words[0] << "\n";
            mode_ok = mode_ok && good;
        };

        const uint32_t values[] = {0, 1, 2, 3, 5};
        for (uint32_t v : values) {
            if (!set_u32(h, 0, v)) {
                std::cout << "FAIL  " << mode.name << " set " << v
                          << " error=" << last_error() << "\n";
                mode_ok = false;
                break;
            }
            const std::string roundtrip_label = std::string("roundtrip ") + std::to_string(v);
            read_value(v, roundtrip_label.c_str());
            if (!square_mul(h, 0, 1)) {
                std::cout << "FAIL  " << mode.name << " square " << v
                          << " error=" << last_error() << "\n";
                mode_ok = false;
                break;
            }
            const std::string square_label = std::string("square ") + std::to_string(v);
            read_value(v * v, square_label.c_str());
        }

        std::cout << "MODE_RESULT=" << mode.name << ':' << (mode_ok ? "PASS" : "FAIL") << "\n";
        if (mode_ok) ++passed_modes;
        destroy(h);
    }

    std::cout << "\nPLANE_ISOLATION_PASSED_MODES=" << passed_modes << "/3\n";
    if (passed_modes == 3) std::cout << "PLANE_DIAGNOSIS=ALL_PATHS_PASS\n";
    else if (passed_modes == 2) std::cout << "PLANE_DIAGNOSIS=ONE_PATH_FAILS\n";
    else if (passed_modes == 1) std::cout << "PLANE_DIAGNOSIS=ONLY_ONE_PATH_PASSES\n";
    else std::cout << "PLANE_DIAGNOSIS=COMMON_PIPELINE_OR_BOTH_PLANES_FAIL\n";

    dlclose(lib);
    return passed_modes == 3 ? 0 : 1;
}
CPP

c++ -std=c++20 -O2 "$SRC" -o "$BIN"
rm -rf "$ROOT/.aevum-kernel-cache"
cd "$ROOT"
export AEVUM_APPLE_DIAGNOSTIC_PLANES=1
unset AEVUM_APPLE_STAGE_FINISH || true
set +e
"$BIN" "$ENGINE" "$DEVICE" "$TUNE" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
