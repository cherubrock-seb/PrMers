#include <dlfcn.h>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Handle = void*;

struct Api {
  void* so{};
  const char* (*last_error)(){};
  int (*resolve)(uint32_t, const char*, char*, size_t){};
  Handle (*create)(uint32_t, size_t, uint32_t, int, const char*, const char*){};
  void (*destroy)(Handle){};
  size_t (*transform_size)(Handle){};
  size_t (*word_count)(Handle){};
  int (*set_u32)(Handle, size_t, uint32_t){};
  int (*get_words)(Handle, size_t, uint32_t*, size_t){};
  int (*prepare)(Handle, size_t, size_t){};
  int (*square_mul)(Handle, size_t, uint32_t){};
  int (*mul)(Handle, size_t, size_t, uint32_t){};

  template <typename T> T sym(const char* name) {
    void* p = dlsym(so, name);
    if (!p) throw std::runtime_error(std::string("missing symbol ") + name);
    return reinterpret_cast<T>(p);
  }

  explicit Api(const char* path) {
    so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!so) throw std::runtime_error(dlerror());
    last_error = sym<decltype(last_error)>("aevum_engine_last_error");
    resolve = sym<decltype(resolve)>("aevum_engine_resolve_fft");
    create = sym<decltype(create)>("aevum_engine_create");
    destroy = sym<decltype(destroy)>("aevum_engine_destroy");
    transform_size = sym<decltype(transform_size)>("aevum_engine_transform_size");
    word_count = sym<decltype(word_count)>("aevum_engine_word_count");
    set_u32 = sym<decltype(set_u32)>("aevum_engine_set_u32");
    get_words = sym<decltype(get_words)>("aevum_engine_get_words");
    prepare = sym<decltype(prepare)>("aevum_engine_prepare");
    square_mul = sym<decltype(square_mul)>("aevum_engine_square_mul");
    mul = sym<decltype(mul)>("aevum_engine_mul");
  }

  ~Api() { if (so) dlclose(so); }
};

static void require(Api& api, int rc, const char* what) {
  if (!rc) throw std::runtime_error(std::string(what) + ": " +
                                    (api.last_error() ? api.last_error() : "unknown"));
}

struct Run {
  std::string resolved;
  size_t transform{};
  std::vector<std::vector<uint32_t>> outputs;
};

static uint64_t hash_words(const std::vector<uint32_t>& words) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t value : words) { h ^= value; h *= 1099511628211ULL; }
  return h;
}

static Run execute(Api& api, uint32_t exponent, uint32_t device,
                   const char* spec, unsigned iterations) {
  char resolved[256]{};
  require(api, api.resolve(exponent, spec, resolved, sizeof(resolved)), "resolve");
  Handle h = api.create(exponent, 2, device, 1, spec, ".");
  if (!h) throw std::runtime_error(api.last_error() ? api.last_error() : "create failed");

  Run result;
  result.resolved = resolved;
  result.transform = api.transform_size(h);
  const size_t count = api.word_count(h);
  auto capture = [&](size_t reg) {
    std::vector<uint32_t> words(count);
    require(api, api.get_words(h, reg, words.data(), words.size()), "get_words");
    result.outputs.push_back(std::move(words));
  };

  require(api, api.set_u32(h, 0, 3), "set square seed");
  for (unsigned i = 0; i < iterations; ++i) {
    require(api, api.square_mul(h, 0, (i & 1u) ? 3u : 1u), "square_mul");
    capture(0);
  }

  require(api, api.set_u32(h, 0, 7), "set mul destination");
  require(api, api.set_u32(h, 1, 5), "set mul source");
  require(api, api.prepare(h, 1, 1), "prepare");
  require(api, api.mul(h, 0, 1, 3), "mul");
  capture(0);

  api.destroy(h);
  return result;
}

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: " << argv[0]
                << " libaevum_engine.so device exponent [iterations]\n";
      return 2;
    }
    Api api(argv[1]);
    const uint32_t device = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t exponent = static_cast<uint32_t>(std::stoul(argv[3]));
    const unsigned iterations = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 2u;

    constexpr const char* paired = "pfa9:1:512:9:512:202";
    constexpr const char* hybrid = "pfa9full:4:512:9:512:202";
    Run left = execute(api, exponent, device, paired, iterations);
    Run right = execute(api, exponent, device, hybrid, iterations);

    if (left.transform != right.transform)
      throw std::runtime_error("transform-size mismatch");
    if (left.outputs.size() != right.outputs.size())
      throw std::runtime_error("output-count mismatch");

    for (size_t output = 0; output < left.outputs.size(); ++output) {
      if (left.outputs[output] != right.outputs[output]) {
        size_t word = 0;
        while (word < left.outputs[output].size() &&
               left.outputs[output][word] == right.outputs[output][word]) ++word;
        std::cerr << "MISMATCH output=" << output << " word=" << word
                  << " type1_hash=0x" << std::hex << hash_words(left.outputs[output])
                  << " type4_hash=0x" << hash_words(right.outputs[output]) << std::dec << "\n";
        const size_t begin = word > 3 ? word - 3 : 0;
        const size_t end = std::min(word + 5, left.outputs[output].size());
        for (size_t i = begin; i < end; ++i)
          std::cerr << "  word[" << i << "] type1=" << left.outputs[output][i]
                    << " type4=" << right.outputs[output][i]
                    << (i == word ? "  <-- first" : "") << "\n";
        return 1;
      }
      std::cout << "exact output " << output << " OK hash=0x" << std::hex
                << hash_words(left.outputs[output]) << std::dec << "\n";
    }

    std::cout << "type1=" << left.resolved << " size=" << left.transform << "\n";
    std::cout << "type4=" << right.resolved << " size=" << right.transform << "\n";
    std::cout << "FFT323161 PFA9 DIFFERENTIAL TEST PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
