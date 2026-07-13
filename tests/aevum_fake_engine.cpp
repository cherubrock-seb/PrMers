#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" {

struct FakeRuntime {
    std::uint32_t exponent;
    std::uint64_t modulus;
    std::vector<std::uint64_t> regs;
};

static thread_local std::string last_error;

static FakeRuntime* checked(void* handle, std::size_t reg) {
    auto* rt = static_cast<FakeRuntime*>(handle);
    if (!rt || reg >= rt->regs.size()) {
        last_error = "invalid fake Aevum register";
        return nullptr;
    }
    return rt;
}

const char* aevum_engine_version() { return "fake-reg-test"; }
const char* aevum_engine_last_error() { return last_error.c_str(); }
int aevum_engine_resolve_fft(std::uint32_t exponent, const char*, char* output, std::size_t output_size) {
    if (exponent < 3 || !output || output_size < 8) {
        last_error = "no fake FFT";
        return 0;
    }
    const char spec[] = "1:256:2:256:101";
    if (output_size < sizeof(spec)) return 0;
    std::copy(spec, spec + sizeof(spec), output);
    return 1;
}

void* aevum_engine_create(std::uint32_t exponent, std::size_t register_count,
                          std::uint32_t, int, const char*, const char*) {
    if (exponent < 3 || exponent > 62 || register_count == 0) return nullptr;
    auto* rt = new FakeRuntime;
    rt->exponent = exponent;
    rt->modulus = (std::uint64_t(1) << exponent) - 1;
    rt->regs.assign(register_count, 0);
    return rt;
}

void aevum_engine_destroy(void* handle) { delete static_cast<FakeRuntime*>(handle); }
std::size_t aevum_engine_transform_size(void*) { return 8; }
std::size_t aevum_engine_word_count(void*) { return 1; }
int aevum_engine_sync(void*) { return 1; }

int aevum_engine_set_u32(void* handle, std::size_t dst, std::uint32_t value) {
    auto* rt = checked(handle, dst); if (!rt) return 0;
    rt->regs[dst] = value % rt->modulus; return 1;
}
int aevum_engine_set_words(void* handle, std::size_t dst, const std::uint32_t* words, std::size_t count) {
    auto* rt = checked(handle, dst); if (!rt || !words || count != 1) return 0;
    rt->regs[dst] = words[0] % rt->modulus; return 1;
}
int aevum_engine_get_words(void* handle, std::size_t src, std::uint32_t* words, std::size_t count) {
    auto* rt = checked(handle, src); if (!rt || !words || count != 1) return 0;
    words[0] = static_cast<std::uint32_t>(rt->regs[src]); return 1;
}
int aevum_engine_copy(void* handle, std::size_t dst, std::size_t src) {
    auto* a = checked(handle, dst); auto* b = checked(handle, src); if (!a || !b) return 0;
    a->regs[dst] = a->regs[src]; return 1;
}
int aevum_engine_prepare(void* handle, std::size_t dst, std::size_t src) {
    return aevum_engine_copy(handle, dst, src);
}
int aevum_engine_square_mul(void* handle, std::size_t reg, std::uint32_t factor) {
    auto* rt = checked(handle, reg); if (!rt) return 0;
    const unsigned __int128 v = static_cast<unsigned __int128>(rt->regs[reg]) * rt->regs[reg] * factor;
    rt->regs[reg] = static_cast<std::uint64_t>(v % rt->modulus); return 1;
}
int aevum_engine_mul(void* handle, std::size_t dst, std::size_t src, std::uint32_t factor) {
    auto* rt = checked(handle, dst); if (!rt || !checked(handle, src)) return 0;
    const unsigned __int128 v = static_cast<unsigned __int128>(rt->regs[dst]) * rt->regs[src] * factor;
    rt->regs[dst] = static_cast<std::uint64_t>(v % rt->modulus); return 1;
}
int aevum_engine_add(void* handle, std::size_t dst, std::size_t src) {
    auto* rt = checked(handle, dst); if (!rt || !checked(handle, src)) return 0;
    rt->regs[dst] = (rt->regs[dst] + rt->regs[src]) % rt->modulus; return 1;
}
int aevum_engine_sub_reg(void* handle, std::size_t dst, std::size_t src) {
    auto* rt = checked(handle, dst); if (!rt || !checked(handle, src)) return 0;
    rt->regs[dst] = (rt->regs[dst] + rt->modulus - rt->regs[src]) % rt->modulus; return 1;
}
int aevum_engine_sub_u32(void* handle, std::size_t dst, std::uint32_t value) {
    auto* rt = checked(handle, dst); if (!rt) return 0;
    const std::uint64_t v = value % rt->modulus;
    rt->regs[dst] = (rt->regs[dst] + rt->modulus - v) % rt->modulus; return 1;
}

int aevum_engine_equal(void* handle, std::size_t lhs, std::size_t rhs, int* equal_out) {
    auto* rt = checked(handle, lhs); if (!rt || !checked(handle, rhs) || !equal_out) return 0;
    *equal_out = rt->regs[lhs] == rt->regs[rhs] ? 1 : 0;
    return 1;
}

}
