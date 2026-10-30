// Creates a PRP engine through the public API, squares 3 a few times and prints the low 64 bits of the result.
// usage: engine_prp_residue_driver <libaevum_engine.so> <device> <exponent> <fft-spec> <tune-dir> <squarings>
// Exit status: 0 printed a residue, 2 the engine could not be created, 3 an engine call failed.
#include "../src/EngineApi.h"

#include <dlfcn.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <class T>
static T sym(void* lib, const char* name) {
    void* s = dlsym(lib, name);
    if (!s) {
        std::fprintf(stderr, "missing symbol %s\n", name);
        std::exit(4);
    }
    return reinterpret_cast<T>(s);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s lib device exponent fft-spec tune-dir squarings\n", argv[0]);
        return 4;
    }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "%s\n", dlerror());
        return 4;
    }
    const auto device = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    const auto exponent = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    const unsigned squarings = static_cast<unsigned>(std::strtoul(argv[6], nullptr, 10));

    const auto create_ex = sym<aevum_engine_handle (*)(uint32_t, size_t, uint32_t, int, const char*, const char*,
                                                       uint32_t)>(lib, "aevum_engine_create_ex");
    const auto destroy = sym<void (*)(aevum_engine_handle)>(lib, "aevum_engine_destroy");
    const auto word_count = sym<size_t (*)(aevum_engine_handle)>(lib, "aevum_engine_word_count");
    const auto set_u32 = sym<int (*)(aevum_engine_handle, size_t, uint32_t)>(lib, "aevum_engine_set_u32");
    const auto square_mul = sym<int (*)(aevum_engine_handle, size_t, uint32_t)>(lib, "aevum_engine_square_mul");
    const auto get_words = sym<int (*)(aevum_engine_handle, size_t, uint32_t*, size_t)>(lib, "aevum_engine_get_words");
    const auto last_error = sym<const char* (*)()>(lib, "aevum_engine_last_error");

    aevum_engine_handle h = create_ex(exponent, 2, device, 1, argv[4], argv[5], AEVUM_WORKLOAD_PRP);
    if (!h) {
        std::printf("CREATE_FAILED %s\n", last_error());
        return 2;
    }
    std::vector<uint32_t> words(word_count(h));
    bool ok = set_u32(h, 0, 3);
    for (unsigned i = 0; ok && i < squarings; ++i) ok = square_mul(h, 0, 1);
    ok = ok && get_words(h, 0, words.data(), words.size());
    if (!ok) {
        std::printf("ENGINE_FAILED %s\n", last_error());
        destroy(h);
        return 3;
    }
    const uint64_t low = words.size() > 1 ? (uint64_t(words[1]) << 32 | words[0]) : words[0];
    std::printf("RESIDUE %016" PRIx64 "\n", low);
    destroy(h);
    return 0;
}
