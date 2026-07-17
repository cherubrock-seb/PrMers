/*
Aevum register API adaptation, Copyright 2026 cherubrock-seb.
Derived from GPUOwl/PRPLL by Mihai Preda and George Woltman.
Licensed under GNU GPL version 3. See LICENSE and UPSTREAM.md.
*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(AEVUM_ENGINE_LIBRARY)
    #define AEVUM_ENGINE_API __declspec(dllexport)
  #else
    #define AEVUM_ENGINE_API __declspec(dllimport)
  #endif
#else
  #define AEVUM_ENGINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* aevum_engine_handle;

AEVUM_ENGINE_API const char* aevum_engine_version(void);
AEVUM_ENGINE_API const char* aevum_engine_last_error(void);
AEVUM_ENGINE_API int aevum_engine_resolve_fft(
    uint32_t exponent,
    const char* fft_spec,
    char* output,
    size_t output_size);

AEVUM_ENGINE_API aevum_engine_handle aevum_engine_create(
    uint32_t exponent,
    size_t register_count,
    uint32_t device,
    int verbose,
    const char* fft_spec,
    const char* tune_dir);

AEVUM_ENGINE_API void aevum_engine_destroy(aevum_engine_handle handle);
AEVUM_ENGINE_API size_t aevum_engine_transform_size(aevum_engine_handle handle);
AEVUM_ENGINE_API size_t aevum_engine_word_count(aevum_engine_handle handle);
AEVUM_ENGINE_API int aevum_engine_sync(aevum_engine_handle handle);

AEVUM_ENGINE_API int aevum_engine_set_u32(aevum_engine_handle handle, size_t dst, uint32_t value);
AEVUM_ENGINE_API int aevum_engine_set_words(aevum_engine_handle handle, size_t dst, const uint32_t* words, size_t count);
AEVUM_ENGINE_API int aevum_engine_get_words(aevum_engine_handle handle, size_t src, uint32_t* words, size_t count);
AEVUM_ENGINE_API int aevum_engine_copy(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_prepare(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_square_mul(aevum_engine_handle handle, size_t reg, uint32_t factor);
AEVUM_ENGINE_API int aevum_engine_mul(aevum_engine_handle handle, size_t dst, size_t src, uint32_t factor);
AEVUM_ENGINE_API int aevum_engine_add(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_sub_reg(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_sub_u32(aevum_engine_handle handle, size_t dst, uint32_t value);
AEVUM_ENGINE_API int aevum_engine_equal(aevum_engine_handle handle, size_t lhs, size_t rhs, int* equal_out);

// Diagnostic only: hashes GF31/GF61 after fftP, middle-in, tail-square,
// middle-out and fftW, followed by the final word-buffer hash and low word.
// trace_count must be at least 12. The source register is preserved.
AEVUM_ENGINE_API int aevum_engine_debug_square_trace(
    aevum_engine_handle handle, size_t src, uint64_t* trace, size_t trace_count);

#ifdef __cplusplus
}
#endif
