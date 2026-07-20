#include "aevum/EngineAevum.hpp"
#include "marin/engine.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/dyld.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#ifndef AEVUM_ENGINE_DEFAULT_LIB
#define AEVUM_ENGINE_DEFAULT_LIB "/usr/local/lib/prmers/libaevum_engine.so"
#endif
#ifndef AEVUM_ENGINE_DEFAULT_TUNE_DIR
#define AEVUM_ENGINE_DEFAULT_TUNE_DIR "/usr/local/share/prmers/aevum"
#endif

namespace {

using Handle = void*;

#if defined(_WIN32)
using NativeLibrary = HMODULE;

std::string library_error_text() {
    const DWORD code = GetLastError();
    if (code == 0) return "unknown Windows loader error";
    char* buffer = nullptr;
    const DWORD size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                      FORMAT_MESSAGE_FROM_SYSTEM |
                                      FORMAT_MESSAGE_IGNORE_INSERTS,
                                      nullptr, code, 0,
                                      reinterpret_cast<char*>(&buffer), 0, nullptr);
    std::string message = size && buffer ? std::string(buffer, size) : "Windows loader error " + std::to_string(code);
    if (buffer) LocalFree(buffer);
    while (!message.empty() && (message.back() == '\r' || message.back() == '\n')) message.pop_back();
    return message;
}

NativeLibrary open_library(const std::filesystem::path& path) {
    SetLastError(0);
    return LoadLibraryW(path.wstring().c_str());
}

void close_library(NativeLibrary library) {
    if (library) FreeLibrary(library);
}
#else
using NativeLibrary = void*;

std::string library_error_text() {
    const char* error = dlerror();
    return error ? error : "unknown dynamic loader error";
}

NativeLibrary open_library(const std::filesystem::path& path) {
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    dlerror();
    return dlopen(path.c_str(), flags);
}

void close_library(NativeLibrary library) {
    if (library) dlclose(library);
}
#endif

struct Api {
    using version_fn = const char* (*)();
    using error_fn = const char* (*)();
    using resolve_fn = int (*)(uint32_t, const char*, char*, std::size_t);
    using create_fn = Handle (*)(uint32_t, std::size_t, uint32_t, int, const char*, const char*);
    using destroy_fn = void (*)(Handle);
    using size_fn = std::size_t (*)(Handle);
    using sync_fn = int (*)(Handle);
    using set_u32_fn = int (*)(Handle, std::size_t, uint32_t);
    using set_words_fn = int (*)(Handle, std::size_t, const uint32_t*, std::size_t);
    using get_words_fn = int (*)(Handle, std::size_t, uint32_t*, std::size_t);
    using copy_fn = int (*)(Handle, std::size_t, std::size_t);
    using prepare_fn = int (*)(Handle, std::size_t, std::size_t);
    using square_fn = int (*)(Handle, std::size_t, uint32_t);
    using mul_fn = int (*)(Handle, std::size_t, std::size_t, uint32_t);
    using binary_fn = int (*)(Handle, std::size_t, std::size_t);
    using sub_u32_fn = int (*)(Handle, std::size_t, uint32_t);
    using equal_fn = int (*)(Handle, std::size_t, std::size_t, int*);

    NativeLibrary library = nullptr;
    std::string path;
    version_fn version = nullptr;
    error_fn last_error = nullptr;
    resolve_fn resolve_fft = nullptr;
    create_fn create = nullptr;
    destroy_fn destroy = nullptr;
    size_fn transform_size = nullptr;
    size_fn word_count = nullptr;
    sync_fn sync = nullptr;
    set_u32_fn set_u32 = nullptr;
    set_words_fn set_words = nullptr;
    get_words_fn get_words = nullptr;
    copy_fn copy = nullptr;
    prepare_fn prepare = nullptr;
    square_fn square_mul = nullptr;
    mul_fn mul = nullptr;
    binary_fn add = nullptr;
    binary_fn sub_reg = nullptr;
    sub_u32_fn sub_u32 = nullptr;
    equal_fn equal = nullptr;

    ~Api() {
        close_library(library);
    }

    template <class T>
    T load_symbol(const char* name) {
#if defined(_WIN32)
        SetLastError(0);
        FARPROC symbol = GetProcAddress(library, name);
        if (!symbol) {
            throw std::runtime_error(std::string("Aevum plugin missing symbol ") + name + ": " + library_error_text());
        }
        return reinterpret_cast<T>(symbol);
#else
        dlerror();
        void* symbol = dlsym(library, name);
        const char* error = dlerror();
        if (error || !symbol) {
            throw std::runtime_error(std::string("Aevum plugin missing symbol ") + name +
                                     (error ? std::string(": ") + error : std::string()));
        }
        return reinterpret_cast<T>(symbol);
#endif
    }

    static std::filesystem::path executable_dir() {
#if defined(_WIN32)
        std::vector<wchar_t> buffer(32768);
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0 || static_cast<std::size_t>(length) >= buffer.size()) {
            return std::filesystem::current_path();
        }
        return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
#elif defined(__APPLE__)
        uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        if (size == 0) return std::filesystem::current_path();
        std::vector<char> buffer(static_cast<std::size_t>(size) + 1, '\0');
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) return std::filesystem::current_path();
        std::error_code ec;
        const auto resolved = std::filesystem::weakly_canonical(std::filesystem::path(buffer.data()), ec);
        return (ec ? std::filesystem::path(buffer.data()) : resolved).parent_path();
#else
        std::array<char, 4096> buffer{};
        const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
        if (length <= 0) return std::filesystem::current_path();
        buffer[static_cast<std::size_t>(length)] = '\0';
        return std::filesystem::path(buffer.data()).parent_path();
#endif
    }

    std::filesystem::path tune_dir() const {
        if (const char* env = std::getenv("AEVUM_TUNE_DIR")) return std::filesystem::absolute(env);
        const auto exe = executable_dir();
        const auto exe_tune = exe / "third_party/aevum/tune.txt";
        if (std::filesystem::exists(exe_tune)) return exe_tune.parent_path();
        const auto adjacent_tune = exe / "tune.txt";
        if (std::filesystem::exists(adjacent_tune)) return adjacent_tune.parent_path();
        const auto source_tune = std::filesystem::path(path).parent_path().parent_path() / "tune.txt";
        if (std::filesystem::exists(source_tune)) return source_tune.parent_path();
        return AEVUM_ENGINE_DEFAULT_TUNE_DIR;
    }

    Api() {
        std::vector<std::filesystem::path> candidates;
        if (const char* env = std::getenv("AEVUM_ENGINE_LIB")) candidates.emplace_back(env);
        const auto add_candidates = [&](const std::filesystem::path& root) {
            candidates.emplace_back(root / "third_party/aevum/build-engine/libaevum_engine.so");
            candidates.emplace_back(root / "third_party/aevum/build-engine/libaevum_engine.dylib");
            candidates.emplace_back(root / "third_party/aevum/build-engine/aevum_engine.dll");
            candidates.emplace_back(root / "third_party/aevum/build-engine/libaevum_engine.dll");
            candidates.emplace_back(root / "libaevum_engine.so");
            candidates.emplace_back(root / "libaevum_engine.dylib");
            candidates.emplace_back(root / "aevum_engine.dll");
            candidates.emplace_back(root / "libaevum_engine.dll");
        };
        add_candidates(executable_dir());
        add_candidates(std::filesystem::current_path());
        candidates.emplace_back(AEVUM_ENGINE_DEFAULT_LIB);

        std::string errors;
        for (const auto& candidate : candidates) {
            if (candidate.empty()) continue;
            library = open_library(candidate);
            if (library) {
                path = candidate.string();
                break;
            }
            errors += candidate.string() + ": " + library_error_text() + "\n";
        }
        if (!library) {
            throw std::runtime_error("Cannot load the Aevum engine plugin. Build it with "
                                     "./build_with_aevum_engine.sh or set AEVUM_ENGINE_LIB.\n" + errors);
        }

        version = load_symbol<version_fn>("aevum_engine_version");
        last_error = load_symbol<error_fn>("aevum_engine_last_error");
        resolve_fft = load_symbol<resolve_fn>("aevum_engine_resolve_fft");
        create = load_symbol<create_fn>("aevum_engine_create");
        destroy = load_symbol<destroy_fn>("aevum_engine_destroy");
        transform_size = load_symbol<size_fn>("aevum_engine_transform_size");
        word_count = load_symbol<size_fn>("aevum_engine_word_count");
        sync = load_symbol<sync_fn>("aevum_engine_sync");
        set_u32 = load_symbol<set_u32_fn>("aevum_engine_set_u32");
        set_words = load_symbol<set_words_fn>("aevum_engine_set_words");
        get_words = load_symbol<get_words_fn>("aevum_engine_get_words");
        copy = load_symbol<copy_fn>("aevum_engine_copy");
        prepare = load_symbol<prepare_fn>("aevum_engine_prepare");
        square_mul = load_symbol<square_fn>("aevum_engine_square_mul");
        mul = load_symbol<mul_fn>("aevum_engine_mul");
        add = load_symbol<binary_fn>("aevum_engine_add");
        sub_reg = load_symbol<binary_fn>("aevum_engine_sub_reg");
        sub_u32 = load_symbol<sub_u32_fn>("aevum_engine_sub_u32");
        equal = load_symbol<equal_fn>("aevum_engine_equal");
    }
};

Api& api() {
    static Api instance;
    return instance;
}

class engine_aevum final : public engine {
public:
    engine_aevum(uint32_t exponent, std::size_t register_count, std::size_t device, bool verbose, const std::string& fft_spec)
        : exponent_(exponent), register_count_(register_count), api_(api()) {
        const std::string tune_dir = api_.tune_dir().string();
        handle_ = api_.create(exponent, register_count, static_cast<uint32_t>(device), verbose ? 1 : 0,
                              fft_spec.empty() ? nullptr : fft_spec.c_str(), tune_dir.c_str());
        if (!handle_) fail("create");
        transform_size_ = api_.transform_size(handle_);
        word_count_ = api_.word_count(handle_);
        if (transform_size_ == 0 || word_count_ == 0) {
            api_.destroy(handle_);
            handle_ = nullptr;
            throw std::runtime_error("Aevum plugin returned an invalid transform");
        }
        std::cout << "[Backend Aevum] engine::Reg adapter active, GF(M31^2) x GF(M61^2)"
                  << " | transform=" << transform_size_
                  << " | regs=" << register_count_
                  << " | plugin=" << api_.path
                  << " | tune=" << api_.tune_dir().string()
                  << " | version=" << (api_.version ? api_.version() : "unknown") << std::endl;
    }

    ~engine_aevum() override { release_gpu_resources_for_lowmem_handoff(); }

    bool is_aevum_backend() const override { return true; }

    void release_gpu_resources_for_lowmem_handoff() override {
        if (handle_) {
            api_.sync(handle_);
            api_.destroy(handle_);
            handle_ = nullptr;
        }
    }

    std::size_t get_size() const override { return transform_size_; }

    void sync() const override {
        require(api_.sync(handle_), "sync");
    }

    void set(const Reg dst, const uint32 a) const override {
        check_reg(dst);
        require(api_.set_u32(handle_, dst, a), "set_u32");
    }

    void copy(const Reg dst, const Reg src) const override {
        check_reg(dst);
        check_reg(src);
        require(api_.copy(handle_, dst, src), "copy");
    }

    void square_mul(const Reg src, const uint32 a = 1) const override {
        check_reg(src);
        require(api_.square_mul(handle_, src, a), "square_mul");
    }

    void set_multiplicand(const Reg dst, const Reg src) const override {
        check_reg(dst);
        check_reg(src);
        require(api_.prepare(handle_, dst, src), "prepare");
    }

    void set_multiplicand2(const Reg dst, const Reg src) const override {
        set_multiplicand(dst, src);
    }

    void mul(const Reg dst, const Reg src, const uint32 a = 1) const override {
        check_reg(dst);
        check_reg(src);
        require(api_.mul(handle_, dst, src, a), "mul");
    }

    void sub(const Reg src, const uint32 a) const override {
        check_reg(src);
        require(api_.sub_u32(handle_, src, a), "sub_u32");
    }

    void add(const Reg dst, const Reg src) const override {
        check_reg(dst);
        check_reg(src);
        require(api_.add(handle_, dst, src), "add");
    }

    void sub_reg(const Reg dst, const Reg src) const override {
        check_reg(dst);
        check_reg(src);
        require(api_.sub_reg(handle_, dst, src), "sub_reg");
    }

    bool is_equal(const Reg lhs, const Reg rhs) const override {
        check_reg(lhs);
        check_reg(rhs);
        int equal = 0;
        require(api_.equal(handle_, lhs, rhs, &equal), "equal");
        return equal != 0;
    }

    std::size_t get_register_data_size() const override {
        return word_count_ * sizeof(uint32_t);
    }

    bool get_data(std::vector<char>& data, const Reg src) const override {
        if (data.size() != get_register_data_size()) return false;
        std::vector<uint32_t> words(word_count_);
        if (!api_.get_words(handle_, src, words.data(), words.size())) return false;
        std::memcpy(data.data(), words.data(), data.size());
        return true;
    }

    bool set_data(const Reg dst, const std::vector<char>& data) const override {
        if (data.size() != get_register_data_size()) return false;
        std::vector<uint32_t> words(word_count_);
        std::memcpy(words.data(), data.data(), data.size());
        return api_.set_words(handle_, dst, words.data(), words.size()) != 0;
    }

    std::size_t get_checkpoint_size() const override {
        return register_count_ * get_register_data_size();
    }

    bool get_checkpoint(std::vector<char>& data) const override {
        if (data.size() != get_checkpoint_size()) return false;
        const std::size_t bytes = get_register_data_size();
        std::vector<char> one(bytes);
        for (std::size_t i = 0; i < register_count_; ++i) {
            if (!get_data(one, static_cast<Reg>(i))) return false;
            std::memcpy(data.data() + i * bytes, one.data(), bytes);
        }
        return true;
    }

    bool set_checkpoint(const std::vector<char>& data) const override {
        if (data.size() != get_checkpoint_size()) return false;
        const std::size_t bytes = get_register_data_size();
        std::vector<char> one(bytes);
        for (std::size_t i = 0; i < register_count_; ++i) {
            std::memcpy(one.data(), data.data() + i * bytes, bytes);
            if (!set_data(static_cast<Reg>(i), one)) return false;
        }
        return true;
    }

    void get_mpz(mpz_t& z, const Reg src) const override {
        check_reg(src);
        std::vector<uint32_t> words(word_count_, 0);
        require(api_.get_words(handle_, src, words.data(), words.size()), "get_words");
        std::size_t used = words.size();
        while (used != 0 && words[used - 1] == 0u) --used;
        if (used == 0) mpz_set_ui(z, 0);
        else mpz_import(z, used, -1, sizeof(uint32_t), 0, 0, words.data());
    }

    void set_mpz(const Reg dst, const mpz_t& z) const override {
        check_reg(dst);
        mpz_t modulus, reduced;
        mpz_init(modulus);
        mpz_init(reduced);
        mpz_set_ui(modulus, 0);
        mpz_setbit(modulus, exponent_);
        mpz_sub_ui(modulus, modulus, 1);
        mpz_mod(reduced, z, modulus);

        std::vector<uint32_t> words(word_count_, 0);
        std::size_t count = 0;
        mpz_export(words.data(), &count, -1, sizeof(uint32_t), 0, 0, reduced);
        mpz_clear(reduced);
        mpz_clear(modulus);
        require(api_.set_words(handle_, dst, words.data(), words.size()), "set_mpz");
    }

protected:
    void get(uint64* const encoded, const std::size_t src) const override {
        check_reg(src);
        std::vector<uint32_t> words(word_count_, 0);
        require(api_.get_words(handle_, src, words.data(), words.size()), "get_words");
        std::size_t bit_pos = 0;
        for (std::size_t k = 0; k < transform_size_; ++k) {
            const uint32_t width = digit_width(k);
            const std::size_t wi = bit_pos / 32;
            const uint32_t shift = static_cast<uint32_t>(bit_pos % 32);
            uint64_t chunk = wi < words.size() ? words[wi] : 0;
            if (shift != 0 && wi + 1 < words.size()) chunk |= uint64_t(words[wi + 1]) << 32;
            const uint64_t mask = width == 32 ? std::numeric_limits<uint32_t>::max() : ((uint64_t(1) << width) - 1);
            const uint32_t value = static_cast<uint32_t>((chunk >> shift) & mask);
            encoded[k] = uint64_t(value) | (uint64_t(width) << 32);
            bit_pos += width;
        }
    }

    void set(const std::size_t dst, uint64* const encoded) const override {
        check_reg(dst);
        std::vector<uint32_t> words(word_count_, 0);
        std::size_t bit_pos = 0;
        for (std::size_t k = 0; k < transform_size_; ++k) {
            const uint32_t width = static_cast<uint32_t>(encoded[k] >> 32);
            const uint32_t expected = digit_width(k);
            if (width != expected) throw std::runtime_error("Aevum digit width mismatch");
            const uint64_t mask = width == 32 ? std::numeric_limits<uint32_t>::max() : ((uint64_t(1) << width) - 1);
            const uint64_t value = uint32_t(encoded[k]) & mask;
            const std::size_t wi = bit_pos / 32;
            const uint32_t shift = static_cast<uint32_t>(bit_pos % 32);
            words[wi] |= static_cast<uint32_t>(value << shift);
            if (shift != 0 && shift + width > 32 && wi + 1 < words.size()) {
                words[wi + 1] |= static_cast<uint32_t>(value >> (32 - shift));
            }
            bit_pos += width;
        }
        require(api_.set_words(handle_, dst, words.data(), words.size()), "set_digits");
    }

private:
    uint32_t digit_width(std::size_t k) const {
        const uint64_t n = transform_size_;
        const uint64_t step = n - (uint64_t(exponent_) % n);
        const uint64_t extra = (step * static_cast<uint64_t>(k)) % n;
        return static_cast<uint32_t>(uint64_t(exponent_) / n + (extra + step < n ? 1 : 0));
    }

    void check_reg(std::size_t reg) const {
        if (!handle_) throw std::runtime_error("Aevum engine has been released");
        if (reg >= register_count_) throw std::runtime_error("Aevum engine register out of range");
    }

    [[noreturn]] void fail(const char* operation) const {
        const char* detail = api_.last_error ? api_.last_error() : nullptr;
        throw std::runtime_error(std::string("Aevum ") + operation + " failed" +
                                 (detail && *detail ? std::string(": ") + detail : std::string()));
    }

    void require(int ok, const char* operation) const {
        if (!ok) fail(operation);
    }

    uint32_t exponent_;
    std::size_t register_count_;
    Api& api_;
    Handle handle_ = nullptr;
    std::size_t transform_size_ = 0;
    std::size_t word_count_ = 0;
};

} // namespace


namespace {

std::size_t parse_fft_dimension(const std::string& text) {
    if (text.empty()) throw std::runtime_error("empty Aevum FFT dimension");
    std::size_t multiplier = 1;
    std::string digits = text;
    const char suffix = digits.back();
    if (suffix == 'K' || suffix == 'k') { multiplier = 1024; digits.pop_back(); }
    else if (suffix == 'M' || suffix == 'm') { multiplier = 1024 * 1024; digits.pop_back(); }
    std::size_t used = 0;
    const unsigned long long value = std::stoull(digits, &used, 10);
    if (used != digits.size() || value == 0) throw std::runtime_error("invalid Aevum FFT dimension");
    if (value > std::numeric_limits<std::size_t>::max() / multiplier) throw std::runtime_error("Aevum FFT dimension overflow");
    return static_cast<std::size_t>(value) * multiplier;
}

std::size_t transform_size_from_spec(const std::string& spec) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (start <= spec.size()) {
        const std::size_t pos = spec.find(':', start);
        fields.emplace_back(spec.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    std::size_t offset = 0;
    if (!fields.empty() && (fields[0] == "pfa3" || fields[0] == "pfa9" || fields[0] == "pfa9fast" || fields[0] == "pfa9full")) offset = 1;
    const bool supported_type = fields.size() >= offset + 4 &&
                                (fields[offset] == "1" || fields[offset] == "4");
    if (!supported_type)
        throw std::runtime_error("invalid Aevum FFT3161/FFT323161 spec");
    const bool explicit_pfa9 = spec.rfind("pfa9:", 0) == 0 || spec.rfind("pfa9fast:", 0) == 0 || spec.rfind("pfa9full:", 0) == 0;
    if (fields[offset] == "4" && !explicit_pfa9)
        throw std::runtime_error("Aevum FFT323161 requires explicit pfa9, pfa9fast, or pfa9full plan");
    const std::size_t width = parse_fft_dimension(fields[offset + 1]);
    const std::size_t middle = parse_fft_dimension(fields[offset + 2]);
    const std::size_t height = parse_fft_dimension(fields[offset + 3]);
    if (width > std::numeric_limits<std::size_t>::max() / middle) throw std::runtime_error("Aevum FFT size overflow");
    const std::size_t half = width * middle;
    if (half > std::numeric_limits<std::size_t>::max() / height / 2) throw std::runtime_error("Aevum FFT size overflow");
    return 2 * half * height;
}

}

bool aevum_engine_resolve_auto_fft(uint32_t exponent,
                                    std::size_t* transform_size,
                                    std::string* resolved_spec,
                                    std::string* reason) {
    try {
        auto& loaded = api();
        std::array<char, 96> resolved{};
        if (!loaded.resolve_fft(exponent, nullptr, resolved.data(), resolved.size())) {
            if (reason) {
                const char* detail = loaded.last_error ? loaded.last_error() : nullptr;
                *reason = detail && *detail ? detail : "no admissible FFT3161 plan";
            }
            return false;
        }
        const std::string spec(resolved.data());
        const std::size_t size = transform_size_from_spec(spec);
        if (transform_size) *transform_size = size;
        if (resolved_spec) *resolved_spec = spec;
        return true;
    } catch (const std::exception& e) {
        if (reason) *reason = std::string("Aevum engine plugin unavailable: ") + e.what();
        return false;
    }
}

bool aevum_engine_resolve_fft(uint32_t exponent,
                               const std::string& requested_spec,
                               std::size_t* transform_size,
                               std::string* resolved_spec,
                               std::string* reason) {
    try {
        auto& loaded = api();
        std::array<char, 96> resolved{};
        const char* request = requested_spec.empty() ? nullptr : requested_spec.c_str();
        if (!loaded.resolve_fft(exponent, request, resolved.data(), resolved.size())) {
            if (reason) {
                const char* detail = loaded.last_error ? loaded.last_error() : nullptr;
                *reason = detail && *detail ? detail : "no admissible FFT3161 plan";
            }
            return false;
        }
        const std::string spec(resolved.data());
        const std::size_t size = transform_size_from_spec(spec);
        if (transform_size) *transform_size = size;
        if (resolved_spec) *resolved_spec = spec;
        return true;
    } catch (const std::exception& e) {
        if (reason) *reason = std::string("Aevum engine plugin unavailable: ") + e.what();
        return false;
    }
}

bool aevum_engine_supports_auto_fft(uint32_t exponent, std::string* reason) {
    return aevum_engine_resolve_auto_fft(exponent, nullptr, nullptr, reason);
}

engine* create_aevum_engine(uint32_t exponent,
                            std::size_t register_count,
                            std::size_t device,
                            bool verbose,
                            const std::string& fft_spec) {
    return new engine_aevum(exponent, register_count, device, verbose, fft_spec);
}
