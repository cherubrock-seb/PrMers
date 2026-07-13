#include "../src/EngineApi.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <class T>
T load_symbol(void* lib, const char* name) {
    void* symbol = dlsym(lib, name);
    if (!symbol) throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<T>(symbol);
}

static void check_small(const std::vector<uint32_t>& words, uint32_t expected, const char* label) {
    if (words.empty() || words[0] != expected) {
        throw std::runtime_error(std::string(label) + " low word mismatch");
    }
    for (size_t i = 1; i < words.size(); ++i) {
        if (words[i] != 0) throw std::runtime_error(std::string(label) + " high word mismatch");
    }
}

int main(int argc, char** argv) {
    const char* library = argc > 1 ? argv[1] : "build-engine/libaevum_engine.so";
    const uint32_t device = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 0;
    const char* tune_dir = argc > 3 ? argv[3] : ".";

    void* lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(dlerror());

    using create_t = aevum_engine_handle (*)(uint32_t, size_t, uint32_t, int, const char*, const char*);
    using destroy_t = void (*)(aevum_engine_handle);
    using words_t = size_t (*)(aevum_engine_handle);
    using set_t = int (*)(aevum_engine_handle, size_t, uint32_t);
    using prepare_t = int (*)(aevum_engine_handle, size_t, size_t);
    using square_t = int (*)(aevum_engine_handle, size_t, uint32_t);
    using mul_t = int (*)(aevum_engine_handle, size_t, size_t, uint32_t);
    using get_t = int (*)(aevum_engine_handle, size_t, uint32_t*, size_t);
    using error_t = const char* (*)();

    const auto create = load_symbol<create_t>(lib, "aevum_engine_create");
    const auto destroy = load_symbol<destroy_t>(lib, "aevum_engine_destroy");
    const auto word_count = load_symbol<words_t>(lib, "aevum_engine_word_count");
    const auto set_u32 = load_symbol<set_t>(lib, "aevum_engine_set_u32");
    const auto prepare = load_symbol<prepare_t>(lib, "aevum_engine_prepare");
    const auto square_mul = load_symbol<square_t>(lib, "aevum_engine_square_mul");
    const auto mul = load_symbol<mul_t>(lib, "aevum_engine_mul");
    const auto get_words = load_symbol<get_t>(lib, "aevum_engine_get_words");
    const auto last_error = load_symbol<error_t>(lib, "aevum_engine_last_error");

    aevum_engine_handle handle = create(1362763u, 2, device, 1, "", tune_dir);
    if (!handle) throw std::runtime_error(last_error());

    try {
        std::vector<uint32_t> words(word_count(handle));

        if (!set_u32(handle, 0, 3) || !square_mul(handle, 0, 3) ||
            !get_words(handle, 0, words.data(), words.size())) {
            throw std::runtime_error(last_error());
        }
        check_small(words, 27, "square_mul factor 3");

        if (!set_u32(handle, 1, 5) || !prepare(handle, 1, 1) ||
            !set_u32(handle, 0, 7) || !mul(handle, 0, 1, 2) ||
            !get_words(handle, 0, words.data(), words.size())) {
            throw std::runtime_error(last_error());
        }
        check_small(words, 70, "mul factor 2");
    } catch (...) {
        destroy(handle);
        dlclose(lib);
        throw;
    }

    destroy(handle);
    dlclose(lib);
    std::cout << "Aevum GPU small-factor tests passed" << std::endl;
    return 0;
}
