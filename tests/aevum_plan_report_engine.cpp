// Fake Aevum plugin for test_aevum_reported_plan: the resolver previews one plan for the native PRP policy
// ("native-prp:auto") while an engine created without an explicit plan ends up on another, as the real plugin
// does when runtime autotune, tune.txt replay or a device profile picks the plan at creation.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_error;

const char* const kPreview = "4:512:8:512:202";   // what the device-neutral native PRP resolver reports (4M)
const char* const kCreated = "1:1K:16:256:101";   // what an engine created without a plan runs (8M)

struct Fake {
    std::string plan;
};

std::size_t transform_of(const std::string& plan) {
    return plan == kPreview ? 4194304u : 8388608u;
}
}

#if defined(_WIN32)
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

FAKE_EXPORT const char* aevum_engine_version() { return "fake-plan-report"; }
FAKE_EXPORT const char* aevum_engine_last_error() { return g_error.c_str(); }

FAKE_EXPORT int aevum_engine_resolve_fft(std::uint32_t exponent, const char* fft_spec,
                                         char* output, std::size_t output_size) {
    g_error.clear();
    if (!output || exponent < 3) return 0;
    const std::string req = fft_spec ? fft_spec : "";
    const std::string spec = req == "native-prp:auto" ? kPreview : req.empty() ? kCreated : req;
    if (spec.size() + 1 > output_size) return 0;
    std::memcpy(output, spec.c_str(), spec.size() + 1);
    return 1;
}

FAKE_EXPORT void* aevum_engine_create_ex(std::uint32_t, std::size_t, std::uint32_t, int, const char* fft_spec,
                                         const char*, std::uint32_t) {
    return new Fake{fft_spec && *fft_spec ? fft_spec : kCreated};
}
FAKE_EXPORT void* aevum_engine_create(std::uint32_t e, std::size_t r, std::uint32_t d, int v, const char* s, const char* t) {
    return aevum_engine_create_ex(e, r, d, v, s, t, 0);
}
FAKE_EXPORT void aevum_engine_destroy(void* handle) { delete static_cast<Fake*>(handle); }
FAKE_EXPORT std::size_t aevum_engine_transform_size(void* handle) { return transform_of(static_cast<Fake*>(handle)->plan); }
FAKE_EXPORT std::size_t aevum_engine_word_count(void*) { return 8; }
FAKE_EXPORT int aevum_engine_plan_spec(void* handle, char* output, std::size_t output_size) {
    const std::string& plan = static_cast<Fake*>(handle)->plan;
    if (plan.size() + 1 > output_size) return 0;
    std::memcpy(output, plan.c_str(), plan.size() + 1);
    return 1;
}
FAKE_EXPORT int aevum_engine_sync(void*) { return 1; }
FAKE_EXPORT int aevum_engine_set_u32(void*, std::size_t, std::uint32_t) { return 0; }
FAKE_EXPORT int aevum_engine_set_words(void*, std::size_t, const std::uint32_t*, std::size_t) { return 0; }
FAKE_EXPORT int aevum_engine_get_words(void*, std::size_t, std::uint32_t*, std::size_t) { return 0; }
FAKE_EXPORT int aevum_engine_copy(void*, std::size_t, std::size_t) { return 0; }
FAKE_EXPORT int aevum_engine_prepare(void*, std::size_t, std::size_t) { return 0; }
FAKE_EXPORT int aevum_engine_square_mul(void*, std::size_t, std::uint32_t) { return 0; }
FAKE_EXPORT int aevum_engine_mul(void*, std::size_t, std::size_t, std::uint32_t) { return 0; }
FAKE_EXPORT int aevum_engine_add(void*, std::size_t, std::size_t) { return 0; }
FAKE_EXPORT int aevum_engine_sub_reg(void*, std::size_t, std::size_t) { return 0; }
FAKE_EXPORT int aevum_engine_sub_u32(void*, std::size_t, std::uint32_t) { return 0; }
FAKE_EXPORT int aevum_engine_equal(void*, std::size_t, std::size_t, int*) { return 0; }

}
