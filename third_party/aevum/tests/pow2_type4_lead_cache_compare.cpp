#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Handle = void*;

struct Api {
  void* so{};
  const char* (*last_error)(){};
  Handle (*create)(uint32_t, size_t, uint32_t, int, const char*, const char*){};
  void (*destroy)(Handle){};
  size_t (*transform_size)(Handle){};
  size_t (*word_count)(Handle){};
  int (*sync)(Handle){};
  int (*set_u32)(Handle, size_t, uint32_t){};
  int (*get_words)(Handle, size_t, uint32_t*, size_t){};
  int (*copy)(Handle, size_t, size_t){};
  int (*square_mul)(Handle, size_t, uint32_t){};

  template <typename T> T sym(const char* name) {
    void* p = dlsym(so, name);
    if (!p) throw std::runtime_error(std::string("missing symbol ") + name);
    return reinterpret_cast<T>(p);
  }

  explicit Api(const char* path) {
    so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!so) throw std::runtime_error(dlerror());
    last_error = sym<decltype(last_error)>("aevum_engine_last_error");
    create = sym<decltype(create)>("aevum_engine_create");
    destroy = sym<decltype(destroy)>("aevum_engine_destroy");
    transform_size = sym<decltype(transform_size)>("aevum_engine_transform_size");
    word_count = sym<decltype(word_count)>("aevum_engine_word_count");
    sync = sym<decltype(sync)>("aevum_engine_sync");
    set_u32 = sym<decltype(set_u32)>("aevum_engine_set_u32");
    get_words = sym<decltype(get_words)>("aevum_engine_get_words");
    copy = sym<decltype(copy)>("aevum_engine_copy");
    square_mul = sym<decltype(square_mul)>("aevum_engine_square_mul");
  }
  ~Api() { if (so) dlclose(so); }
};

static void require(Api& api, int rc, const char* what) {
  if (!rc) throw std::runtime_error(std::string(what) + ": " +
                                    (api.last_error() ? api.last_error() : "unknown"));
}

static uint64_t hash_words(const std::vector<uint32_t>& words) {
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t value : words) { h ^= value; h *= 1099511628211ULL; }
  return h;
}

struct Run {
  size_t transform{};
  std::vector<std::vector<uint32_t>> snapshots;
};

static Run execute(Api& api, uint32_t exponent, uint32_t device, bool cache) {
#if defined(_WIN32)
  _putenv_s("AEVUM_REG_LEAD_CACHE", cache ? "1" : "0");
#else
  setenv("AEVUM_REG_LEAD_CACHE", cache ? "1" : "0", 1);
#endif
  constexpr const char* spec = "4:512:8:512:202";
  Handle h = api.create(exponent, 2, device, 1, spec, ".");
  if (!h) throw std::runtime_error(api.last_error() ? api.last_error() : "create failed");

  Run run;
  run.transform = api.transform_size(h);
  if (run.transform != 4194304u) throw std::runtime_error("unexpected power-of-two type-4 transform size");
  const size_t count = api.word_count(h);

  auto capture = [&](size_t reg) {
    std::vector<uint32_t> words(count);
    require(api, api.get_words(h, reg, words.data(), words.size()), "get_words");
    run.snapshots.push_back(std::move(words));
  };
  auto squares = [&](unsigned count_) {
    for (unsigned i = 0; i < count_; ++i)
      require(api, api.square_mul(h, 0, 1), "square_mul");
  };

  require(api, api.set_u32(h, 0, 3), "set seed");
  squares(1); capture(0);       // pending-only flush
  squares(2); capture(0);       // NONE->WIDTH, WIDTH->NONE
  squares(17); capture(0);      // long hot chain
  squares(33); require(api, api.sync(h), "sync"); capture(0); // explicit sync flush
  squares(9); require(api, api.copy(h, 1, 0), "copy"); capture(0); capture(1);
  require(api, api.square_mul(h, 0, 3), "square_mul factor3"); capture(0);

  api.destroy(h);
  return run;
}

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: " << argv[0] << " libaevum_engine.so device exponent\n";
      return 2;
    }
    Api api(argv[1]);
    const uint32_t device = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t exponent = static_cast<uint32_t>(std::stoul(argv[3]));
    Run cached = execute(api, exponent, device, true);
    Run canonical = execute(api, exponent, device, false);
    if (cached.transform != canonical.transform || cached.snapshots.size() != canonical.snapshots.size())
      throw std::runtime_error("run shape mismatch");
    for (size_t i = 0; i < cached.snapshots.size(); ++i) {
      if (cached.snapshots[i] != canonical.snapshots[i]) {
        size_t word = 0;
        while (word < cached.snapshots[i].size() &&
               cached.snapshots[i][word] == canonical.snapshots[i][word]) ++word;
        std::cerr << "MISMATCH snapshot=" << i << " word=" << word
                  << " cached=0x" << std::hex << hash_words(cached.snapshots[i])
                  << " canonical=0x" << hash_words(canonical.snapshots[i]) << std::dec << "\n";
        return 1;
      }
      std::cout << "snapshot " << i << " exact hash=0x" << std::hex
                << hash_words(cached.snapshots[i]) << std::dec << "\n";
    }
    std::cout << "POWER2 FFT323161 LEAD-CACHE DIFFERENTIAL TEST PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
