#include "../src/EngineApi.h"

#include <dlfcn.h>

#include <array>
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
    using create_ex_t = aevum_engine_handle (*)(uint32_t, size_t, uint32_t, int, const char*, const char*, uint32_t);
    using destroy_t = void (*)(aevum_engine_handle);
    using words_t = size_t (*)(aevum_engine_handle);
    using transform_t = size_t (*)(aevum_engine_handle);
    using plan_t = int (*)(aevum_engine_handle, char*, size_t);
    using sync_t = int (*)(aevum_engine_handle);
    using set_t = int (*)(aevum_engine_handle, size_t, uint32_t);
    using prepare_t = int (*)(aevum_engine_handle, size_t, size_t);
    using square_t = int (*)(aevum_engine_handle, size_t, uint32_t);
    using mul_t = int (*)(aevum_engine_handle, size_t, size_t, uint32_t);
    using get_t = int (*)(aevum_engine_handle, size_t, uint32_t*, size_t);
    using error_t = const char* (*)();

    const auto create = load_symbol<create_t>(lib, "aevum_engine_create");
    const auto create_ex = load_symbol<create_ex_t>(lib, "aevum_engine_create_ex");
    const auto destroy = load_symbol<destroy_t>(lib, "aevum_engine_destroy");
    const auto word_count = load_symbol<words_t>(lib, "aevum_engine_word_count");
    const auto transform_size = load_symbol<transform_t>(lib, "aevum_engine_transform_size");
    const auto plan_spec = load_symbol<plan_t>(lib, "aevum_engine_plan_spec");
    const auto sync = load_symbol<sync_t>(lib, "aevum_engine_sync");
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

    // Ordinary production PRP also executes square_mul(..., 3) in its
    // Gerbicz full-check replay. M82589933 reproduces the selector boundary
    // where the old PRP AUTO path admitted a 2M transform with insufficient
    // factor-3 arithmetic headroom.
    handle = create_ex(
        82589933u, 8, device, 1, "", tune_dir, AEVUM_WORKLOAD_PRP);
    if (!handle) throw std::runtime_error(last_error());

    try {
        const size_t transform = transform_size(handle);
        std::array<char, 96> plan{};
        if (!plan_spec(handle, plan.data(), plan.size()))
            throw std::runtime_error("PRP active plan unavailable");
        if (transform <= 2097152u)
            throw std::runtime_error(
                "PRP factor-3 capacity guard did not promote the unsafe 2M plan");

        std::vector<uint32_t> words(word_count(handle));
        if (!set_u32(handle, 0, 3) ||
            !square_mul(handle, 0, 3) ||
            !sync(handle) ||
            !get_words(handle, 0, words.data(), words.size())) {
            throw std::runtime_error(last_error());
        }
        check_small(words, 27, "PRP Gerbicz square_mul factor 3");

        std::cout << "Aevum PRP factor-3 capacity"
                  << " exponent=82589933"
                  << " transform=" << transform
                  << " plan=" << plan.data()
                  << std::endl;
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
