// aevum_engine_equal must compare residues, not raw balanced digits: add,
// sub_reg, sub_u32 and small-factor multiplies leave digits unnormalised, and
// an equality test right after them must still see equal values as equal.
// Device runs on a laptop must use PoCL (OCL_ICD_VENDORS=.../pocl.icd).
//
// usage: aevum-engine-equal-canonical-gpu-test <libaevum_engine.so> [device] [tune_dir] [fft_spec] [exponent]
#include "../src/EngineApi.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Words = std::vector<uint32_t>;

template <class T>
T load_symbol(void* lib, const char* name) {
    void* symbol = dlsym(lib, name);
    if (!symbol) throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<T>(symbol);
}

// Host arithmetic modulo 2^E - 1 on little-endian 32-bit words.
struct Mersenne {
    uint32_t E;
    size_t n;
    uint32_t topMask;

    explicit Mersenne(uint32_t e) : E(e), n((e - 1) / 32 + 1), topMask(e % 32 ? (uint32_t(1) << (e % 32)) - 1 : ~0u) {}

    bool isModulus(const Words& a) const {
        for (size_t i = 0; i + 1 < n; ++i) if (a[i] != ~0u) return false;
        return a[n - 1] == topMask;
    }

    void canon(Words& a) const {
        if (isModulus(a)) a.assign(n, 0);
    }

    Words add(const Words& a, const Words& b) const {
        Words r(n);
        uint64_t c = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t s = uint64_t(a[i]) + b[i] + c;
            r[i] = uint32_t(s);
            c = s >> 32;
        }
        // Bits at and above E wrap around to bit 0 (2^E == 1).
        const uint32_t topBits = E % 32;
        uint64_t wrap = topBits ? (uint64_t(r[n - 1] >> topBits) | (c << (32 - topBits))) : c;
        r[n - 1] &= topMask;
        for (size_t i = 0; i < n && wrap; ++i) {
            uint64_t s = uint64_t(r[i]) + wrap;
            r[i] = uint32_t(s);
            wrap = s >> 32;
        }
        canon(r);
        return r;
    }

    Words neg(const Words& a) const {
        Words r(n);
        for (size_t i = 0; i < n; ++i) r[i] = ~a[i];
        r[n - 1] &= topMask;
        canon(r);
        return r;
    }

    Words sub(const Words& a, const Words& b) const { return add(a, neg(b)); }

    Words small(uint32_t v) const {
        Words r(n, 0);
        r[0] = v;
        return r;
    }

    Words random(std::mt19937_64& rng) const {
        Words r(n);
        for (auto& w : r) w = uint32_t(rng());
        r[n - 1] &= topMask;
        canon(r);
        return r;
    }
};

int main(int argc, char** argv) {
    const char* library = argc > 1 ? argv[1] : "build-engine/libaevum_engine.so";
    const uint32_t device = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 0;
    const char* tune_dir = argc > 3 ? argv[3] : ".";
    const char* fft_spec = argc > 4 ? argv[4] : "";
    const uint32_t exponent = argc > 5 ? static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 1362763u;

    void* lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(dlerror());

    using create_t = aevum_engine_handle (*)(uint32_t, size_t, uint32_t, int, const char*, const char*);
    using destroy_t = void (*)(aevum_engine_handle);
    using words_t = size_t (*)(aevum_engine_handle);
    using plan_t = int (*)(aevum_engine_handle, char*, size_t);
    using set_t = int (*)(aevum_engine_handle, size_t, uint32_t);
    using setw_t = int (*)(aevum_engine_handle, size_t, const uint32_t*, size_t);
    using getw_t = int (*)(aevum_engine_handle, size_t, uint32_t*, size_t);
    using binary_t = int (*)(aevum_engine_handle, size_t, size_t);
    using square_t = int (*)(aevum_engine_handle, size_t, uint32_t);
    using equal_t = int (*)(aevum_engine_handle, size_t, size_t, int*);
    using error_t = const char* (*)();

    const auto create = load_symbol<create_t>(lib, "aevum_engine_create");
    const auto destroy = load_symbol<destroy_t>(lib, "aevum_engine_destroy");
    const auto word_count = load_symbol<words_t>(lib, "aevum_engine_word_count");
    const auto plan_spec = load_symbol<plan_t>(lib, "aevum_engine_plan_spec");
    const auto set_u32 = load_symbol<set_t>(lib, "aevum_engine_set_u32");
    const auto set_words = load_symbol<setw_t>(lib, "aevum_engine_set_words");
    const auto get_words = load_symbol<getw_t>(lib, "aevum_engine_get_words");
    const auto copy = load_symbol<binary_t>(lib, "aevum_engine_copy");
    const auto add = load_symbol<binary_t>(lib, "aevum_engine_add");
    const auto sub_reg = load_symbol<binary_t>(lib, "aevum_engine_sub_reg");
    const auto sub_u32 = load_symbol<set_t>(lib, "aevum_engine_sub_u32");
    const auto square_mul = load_symbol<square_t>(lib, "aevum_engine_square_mul");
    const auto equal = load_symbol<equal_t>(lib, "aevum_engine_equal");
    const auto last_error = load_symbol<error_t>(lib, "aevum_engine_last_error");

    aevum_engine_handle h = create(exponent, 6, device, 1, fft_spec, tune_dir);
    if (!h) throw std::runtime_error(last_error());

    int failures = 0;
    auto ok = [&](int rc) { if (!rc) throw std::runtime_error(last_error()); };
    auto expect = [&](size_t a, size_t b, bool want, const char* label) {
        int eq = -1;
        ok(equal(h, a, b, &eq));
        const bool pass = (eq == 1) == want;
        std::cout << (pass ? "PASS " : "FAIL ") << label << " (equal=" << eq << ")" << std::endl;
        if (!pass) ++failures;
    };

    try {
        const Mersenne m(exponent);
        if (word_count(h) != m.n) throw std::runtime_error("unexpected word count");
        char plan[128] = {};
        ok(plan_spec(h, plan, sizeof(plan)));
        std::cout << "exponent=" << exponent << " plan=" << plan << std::endl;

        std::mt19937_64 rng(0x5eed1234u);
        const Words A = m.random(rng);
        const Words B = m.random(rng);
        auto put = [&](size_t r, const Words& w) { ok(set_words(h, r, w.data(), w.size())); };

        // 0: A + B through add, 1: canonical A + B.
        put(0, A); put(2, B); ok(add(h, 0, 2));
        put(1, m.add(A, B));
        expect(0, 1, true, "add then equal (canonical rhs)");
        expect(1, 0, true, "add then equal (operands swapped)");

        // A + B + 1 must still differ.
        put(1, m.add(m.add(A, B), m.small(1)));
        expect(0, 1, false, "add then equal against value + 1");

        // sub_reg: A - B.
        put(0, A); ok(sub_reg(h, 0, 2));
        put(1, m.sub(A, B));
        expect(0, 1, true, "sub_reg then equal (canonical rhs)");

        // A - A == 0, against set_u32(0) and against a stored 2^E - 1 (also zero).
        put(0, A); put(2, A); ok(sub_reg(h, 0, 2));
        ok(set_u32(h, 1, 0));
        expect(0, 1, true, "A - A equals 0");
        Words modulus(m.n, ~0u);
        modulus[m.n - 1] = m.topMask;
        put(1, modulus);
        expect(0, 1, true, "A - A equals a stored 2^E - 1");

        // sub_u32 across a borrow: (A + 5) - 7 == A - 2.
        put(0, m.add(A, m.small(5))); ok(sub_u32(h, 0, 7));
        put(1, m.sub(A, m.small(2)));
        expect(0, 1, true, "sub_u32 then equal");

        // set_u32 with the top bit set vs the same value from set_words.
        ok(set_u32(h, 0, 0xfffffff0u));
        put(1, m.small(0xfffffff0u));
        expect(0, 1, true, "set_u32(0xfffffff0) equals set_words");

        // (A + B)^2 + (A - B)^2 == 2A^2 + 2B^2, each side summed with add after
        // squaring add/sub_reg outputs.  Only squares are used: on PoCL the
        // tailMul kernel takes longer to JIT than this test should run.
        put(0, A); put(2, B); ok(add(h, 0, 2)); ok(square_mul(h, 0, 1));
        put(3, A); ok(sub_reg(h, 3, 2)); ok(square_mul(h, 3, 1)); ok(add(h, 0, 3));
        put(1, A); ok(square_mul(h, 1, 1)); ok(copy(h, 5, 1)); ok(add(h, 1, 5));
        put(3, B); ok(square_mul(h, 3, 1)); ok(copy(h, 5, 3)); ok(add(h, 3, 5)); ok(add(h, 1, 3));
        expect(0, 1, true, "(A+B)^2 + (A-B)^2 equals 2A^2 + 2B^2");

        // Cross-check the same values on the canonical export.
        Words w0(m.n), w1(m.n);
        ok(get_words(h, 0, w0.data(), w0.size()));
        ok(get_words(h, 1, w1.data(), w1.size()));
        if (w0 != w1) { std::cout << "FAIL exported words differ" << std::endl; ++failures; }

        // Both sides out of a carry kernel still compare equal (GPU fast path).
        put(5, A); ok(square_mul(h, 5, 1));
        put(4, A); ok(square_mul(h, 4, 1));
        expect(5, 4, true, "A^2 equals A^2");

        // square_mul(., 3) (small-factor adds) equals A^2 + A^2 + A^2.
        put(5, A); ok(square_mul(h, 5, 3));
        ok(copy(h, 3, 4)); ok(add(h, 3, 4)); ok(add(h, 3, 4));
        expect(5, 3, true, "square_mul(A, 3) equals 3 * A^2 via add");

        // A truly different value after add must still be unequal.
        put(0, A); put(2, B); ok(add(h, 0, 2));
        put(1, A);
        expect(0, 1, false, "A + B differs from A");
    } catch (...) {
        destroy(h);
        dlclose(lib);
        throw;
    }

    destroy(h);
    dlclose(lib);
    if (failures) {
        std::cout << failures << " Aevum engine equal test(s) failed" << std::endl;
        return 1;
    }
    std::cout << "Aevum engine equal tests passed" << std::endl;
    return 0;
}
