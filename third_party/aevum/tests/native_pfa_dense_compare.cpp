// Native Good-Thomas PFA GPU differential test with dense random residues.
//
// A pfa3:1 / pfa9:1 plan is compared word-for-word against a power-of-two
// FFT3161 plan on dense, seeded random residues: several squarings, a generic
// multiply (tailMul) and a prepared multiply (tailMulLow).
//
// Dense input is essential.  The former PFA check squared set_u32(3), whose
// transform spectrum is constant; an omitted twiddle that both the forward and
// the inverse transform skip is invisible on such input, while every word of
// a dense residue comes out wrong.
//
// usage: native-pfa-dense-compare libaevum_engine.so device exponent
//                                 reference_spec candidate_spec [iterations] [seed]
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
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
  int (*set_words)(Handle, size_t, const uint32_t*, size_t){};
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
    set_words = sym<decltype(set_words)>("aevum_engine_set_words");
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
  std::vector<std::string> labels;
};

static uint64_t hash_words(const std::vector<uint32_t>& words) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t value : words) { h ^= value; h *= 1099511628211ULL; }
  return h;
}

// Uniformly random E-bit residue (the engine masks the top word to E bits).
static std::vector<uint32_t> dense_words(std::mt19937& rng, uint32_t exponent, size_t count) {
  std::vector<uint32_t> words(count);
  for (uint32_t& w : words) w = rng();
  if (exponent % 32) words.back() &= (uint32_t(1) << (exponent % 32)) - 1;
  return words;
}

static Run execute(Api& api, uint32_t exponent, uint32_t device,
                   const char* spec, unsigned iterations, unsigned seed) {
  char resolved[256]{};
  require(api, api.resolve(exponent, spec, resolved, sizeof(resolved)), "resolve");
  Handle h = api.create(exponent, 2, device, 1, spec, ".");
  if (!h) throw std::runtime_error(std::string("create ") + spec + ": " +
                                   (api.last_error() ? api.last_error() : "failed"));

  Run result;
  result.resolved = resolved;
  result.transform = api.transform_size(h);
  const size_t count = api.word_count(h);
  auto capture = [&](size_t reg, const std::string& label) {
    std::vector<uint32_t> words(count);
    require(api, api.get_words(h, reg, words.data(), words.size()), "get_words");
    result.outputs.push_back(std::move(words));
    result.labels.push_back(label);
  };

  // The same seed produces the same residues for both plans.
  std::mt19937 rng(seed);
  std::vector<uint32_t> x = dense_words(rng, exponent, count);
  require(api, api.set_words(h, 0, x.data(), x.size()), "set dense square input");
  for (unsigned i = 0; i < iterations; ++i) {
    require(api, api.square_mul(h, 0, (i & 1u) ? 3u : 1u), "square_mul");
    capture(0, "square " + std::to_string(i + 1));
  }

  // Generic multiply by a dense operand (tailMul).
  std::vector<uint32_t> y = dense_words(rng, exponent, count);
  require(api, api.set_words(h, 1, y.data(), y.size()), "set dense mul source");
  require(api, api.mul(h, 0, 1, 1), "mul");
  capture(0, "generic mul");

  // Prepared multiply by another dense operand (tailMulLow).
  std::vector<uint32_t> z = dense_words(rng, exponent, count);
  require(api, api.set_words(h, 1, z.data(), z.size()), "set dense prepared source");
  require(api, api.prepare(h, 1, 1), "prepare");
  require(api, api.mul(h, 0, 1, 3), "mul prepared");
  capture(0, "prepared mul");

  // One more squaring after the multiplies.
  require(api, api.square_mul(h, 0, 1), "square_mul final");
  capture(0, "final square");

  // Consecutive squarings without an intervening read: the engine keeps one
  // operation pending and executes it with a retained width transform, so a
  // PFA plan with a lead cache runs fftW + carryA + fftPCarryB (lazy carryB
  // inside the Good-Thomas gather) instead of the canonical carry sequence.
  for (unsigned i = 0; i < 4; ++i)
    require(api, api.square_mul(h, 0, 1), "square_mul chain");
  capture(0, "square chain x4");

  api.destroy(h);
  return result;
}

int main(int argc, char** argv) {
  try {
    if (argc < 6) {
      std::cerr << "usage: " << argv[0]
                << " libaevum_engine.so device exponent reference_spec candidate_spec"
                   " [iterations] [seed]\n";
      return 2;
    }
    Api api(argv[1]);
    const uint32_t device = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t exponent = static_cast<uint32_t>(std::stoul(argv[3]));
    const char* reference = argv[4];
    const char* candidate = argv[5];
    const unsigned iterations = argc > 6 ? static_cast<unsigned>(std::stoul(argv[6])) : 3u;
    const unsigned seed = argc > 7 ? static_cast<unsigned>(std::stoul(argv[7])) : 20261002u;

    Run left = execute(api, exponent, device, reference, iterations, seed);
    Run right = execute(api, exponent, device, candidate, iterations, seed);

    if (left.outputs.size() != right.outputs.size())
      throw std::runtime_error("output-count mismatch");

    for (size_t output = 0; output < left.outputs.size(); ++output) {
      if (left.outputs[output] != right.outputs[output]) {
        size_t word = 0, differing = 0;
        while (word < left.outputs[output].size() &&
               left.outputs[output][word] == right.outputs[output][word]) ++word;
        for (size_t i = 0; i < left.outputs[output].size(); ++i)
          differing += left.outputs[output][i] != right.outputs[output][i];
        std::cerr << "MISMATCH output=" << output << " (" << left.labels[output] << ")"
                  << " first_word=" << word
                  << " differing_words=" << differing << "/" << left.outputs[output].size()
                  << " reference_hash=0x" << std::hex << hash_words(left.outputs[output])
                  << " candidate_hash=0x" << hash_words(right.outputs[output]) << std::dec << "\n";
        const size_t begin = word > 3 ? word - 3 : 0;
        const size_t end = std::min(word + 5, left.outputs[output].size());
        for (size_t i = begin; i < end; ++i)
          std::cerr << "  word[" << i << "] " << reference << "=" << left.outputs[output][i]
                    << " " << candidate << "=" << right.outputs[output][i]
                    << (i == word ? "  <-- first" : "") << "\n";
        return 1;
      }
      std::cout << "exact output " << output << " (" << left.labels[output] << ") OK hash=0x"
                << std::hex << hash_words(left.outputs[output]) << std::dec << "\n";
    }

    std::cout << "reference=" << left.resolved << " size=" << left.transform << "\n";
    std::cout << "candidate=" << right.resolved << " size=" << right.transform
              << " bits/word=" << double(exponent) / double(right.transform) << "\n";
    std::cout << "NATIVE AEVUM PFA DENSE DIFFERENTIAL TEST PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
