#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_error;

std::string resolve(std::uint32_t exponent, const char* requested) {
    const std::string req = requested ? requested : "";
    if (!req.empty()) {
        if (req == "pfa:auto") {
            return exponent >= 170000000u
                ? "pfa9:4:512:9:512:202"
                : "pfa3:1:1K:3:512:101";
        }
        if (req == "throughput:auto" || req == "throughput:prp") {
            if (exponent < 20000000u) return "4:256:4:256:202";
            return "4:512:8:512:202";
        }
        if (req == "throughput:ll") return "4:1K:2:1K:202";
        if (req == "throughput:pm1") {
            if (exponent < 20000000u) return "4:256:4:256:202";
            return "4:256:16:256:202";
        }
        if (req == "throughput:ecm") {
            if (exponent < 20000000u) return "1:256:4:256:101";
            return "1:1K:4:256:101";
        }
        // Exact manual FFT/PFA requests are passed through unchanged.  This
        // keeps the host policy test focused on precedence and transform-size
        // decisions; the real FFT resolver has separate source/unit tests.
        return req;
    }

    // Deterministic device-neutral native-AUTO stand-in matching the transform
    // bands exercised by test_aevum_auto_policy.cpp.  No OpenCL runtime or GPU
    // is required for this host policy test.
    if (exponent < 3000000u) return "1:256:2:256:101";      // 256K
    if (exponent < 30000000u) return "1:256:4:256:101";     // 512K
    if (exponent < 170000000u) return "1:512:8:512:202";    // 4M
    return "1:1K:8:512:202";                                // 8M
}
}

#if defined(_WIN32)
#define POLICY_EXPORT __declspec(dllexport)
#else
#define POLICY_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

POLICY_EXPORT const char* aevum_engine_version() { return "host-policy-resolver"; }
POLICY_EXPORT const char* aevum_engine_last_error() { return g_error.c_str(); }

POLICY_EXPORT int aevum_engine_resolve_fft(std::uint32_t exponent, const char* fft_spec,
                                           char* output, std::size_t output_size) {
    g_error.clear();
    if (!output || output_size == 0 || exponent < 3) {
        g_error = "invalid host-policy resolver request";
        return 0;
    }
    const std::string request = fft_spec ? fft_spec : "";

#if defined(__APPLE__)
    if (request == "pfa:auto" ||
        request.rfind("pfa3:", 0) == 0 ||
        request.rfind("pfa9:", 0) == 0 ||
        request.rfind("4:", 0) == 0) {
        g_error =
            "Apple OpenCL 1.2 supports only stock FFT3161 Aevum plans";
        return 0;
    }

    const char* effective_request =
        (request == "pow2:auto" ||
         request.rfind("throughput:", 0) == 0)
            ? nullptr
            : fft_spec;
#else
    const char* effective_request = fft_spec;
#endif

    const std::string spec = resolve(exponent, effective_request);
    if (spec.size() + 1 > output_size) {
        g_error = "host-policy resolver output buffer too small";
        return 0;
    }
    std::memcpy(output, spec.c_str(), spec.size() + 1);
    return 1;
}

// EngineAevum loads the stable plugin ABI eagerly.  The policy test invokes
// only aevum_engine_resolve_fft; these entries intentionally remain inert.
POLICY_EXPORT void* aevum_engine_create(std::uint32_t, std::size_t, std::uint32_t, int, const char*, const char*) { return nullptr; }
POLICY_EXPORT void* aevum_engine_create_ex(std::uint32_t, std::size_t, std::uint32_t, int, const char*, const char*, std::uint32_t) { return nullptr; }
POLICY_EXPORT void aevum_engine_destroy(void*) {}
POLICY_EXPORT std::size_t aevum_engine_transform_size(void*) { return 0; }
POLICY_EXPORT std::size_t aevum_engine_word_count(void*) { return 0; }
POLICY_EXPORT int aevum_engine_sync(void*) { return 0; }
POLICY_EXPORT int aevum_engine_set_u32(void*, std::size_t, std::uint32_t) { return 0; }
POLICY_EXPORT int aevum_engine_set_words(void*, std::size_t, const std::uint32_t*, std::size_t) { return 0; }
POLICY_EXPORT int aevum_engine_get_words(void*, std::size_t, std::uint32_t*, std::size_t) { return 0; }
POLICY_EXPORT int aevum_engine_copy(void*, std::size_t, std::size_t) { return 0; }
POLICY_EXPORT int aevum_engine_prepare(void*, std::size_t, std::size_t) { return 0; }
POLICY_EXPORT int aevum_engine_square_mul(void*, std::size_t, std::uint32_t) { return 0; }
POLICY_EXPORT int aevum_engine_mul(void*, std::size_t, std::size_t, std::uint32_t) { return 0; }
POLICY_EXPORT int aevum_engine_add(void*, std::size_t, std::size_t) { return 0; }
POLICY_EXPORT int aevum_engine_sub_reg(void*, std::size_t, std::size_t) { return 0; }
POLICY_EXPORT int aevum_engine_sub_u32(void*, std::size_t, std::uint32_t) { return 0; }
POLICY_EXPORT int aevum_engine_equal(void*, std::size_t, std::size_t, int*) { return 0; }

}
