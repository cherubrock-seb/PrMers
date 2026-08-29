#include <dlfcn.h>
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
  Handle (*create)(uint32_t,size_t,uint32_t,int,const char*,const char*){};
  void (*destroy)(Handle){};
  size_t (*transform_size)(Handle){};
  size_t (*word_count)(Handle){};
  int (*sync)(Handle){};
  int (*set_u32)(Handle,size_t,uint32_t){};
  int (*get_words)(Handle,size_t,uint32_t*,size_t){};
  int (*copy)(Handle,size_t,size_t){};
  int (*square_mul)(Handle,size_t,uint32_t){};

  template<class T> T sym(const char* n) {
    void* p = dlsym(so,n);
    if (!p) throw std::runtime_error(std::string("missing symbol ") + n);
    return reinterpret_cast<T>(p);
  }
  explicit Api(const char* p) {
    so = dlopen(p, RTLD_NOW | RTLD_LOCAL);
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
  ~Api(){ if (so) dlclose(so); }
};

static void req(Api& a, int rc, const char* what) {
  if (!rc) throw std::runtime_error(std::string(what) + ": " + (a.last_error() ? a.last_error() : "unknown"));
}
static uint64_t hash_words(const std::vector<uint32_t>& v) {
  uint64_t h=1469598103934665603ULL;
  for (auto x:v) { h ^= x; h *= 1099511628211ULL; }
  return h;
}
static void env01(const char* k, bool v) { setenv(k, v ? "1" : "0", 1); }

struct Run { size_t transform{}; std::vector<std::vector<uint32_t>> snaps; };

static Run execute(Api& a, uint32_t exponent, uint32_t device, bool resident) {
  env01("AEVUM_TYPE4_MULTI_Q", true);
  env01("AEVUM_REG_LEAD_CACHE", resident);
  env01("AEVUM_PFA_RESIDENT", resident);
  env01("AEVUM_PFA_LEAD_BRIDGE", false);

  constexpr const char* spec = "pfa9full:4:512:9:512:202";
  Handle h = a.create(exponent, 2, device, 1, spec, ".");
  if (!h) throw std::runtime_error(a.last_error() ? a.last_error() : "create failed");

  Run r;
  r.transform = a.transform_size(h);
  if (r.transform != 4718592u) throw std::runtime_error("unexpected PFA9 transform size");
  const size_t wc = a.word_count(h);

  auto capture = [&](size_t reg) {
    std::vector<uint32_t> w(wc);
    req(a, a.get_words(h, reg, w.data(), w.size()), "get_words");
    r.snaps.push_back(std::move(w));
  };
  auto squares = [&](unsigned n) {
    for (unsigned i=0; i<n; ++i) req(a, a.square_mul(h, 0, 1), "square_mul");
  };

  req(a, a.set_u32(h, 0, 3), "seed");

  // Short boundary: no resident state is retained yet.
  squares(1); capture(0);
  // Exercise the first canonical -> resident -> canonical transition.
  squares(2); capture(0);
  // Long resident chains.
  squares(17); capture(0);
  squares(64); capture(0);
  squares(257); capture(0);
  // Explicit synchronization must flush the final pending resident square.
  squares(33); req(a, a.sync(h), "sync"); capture(0);
  // Copy is an external canonical boundary and both registers must match.
  squares(19); req(a, a.copy(h,1,0), "copy"); capture(0); capture(1);
  // Non-unit factor forces the fallback canonical arithmetic boundary.
  req(a, a.square_mul(h,0,3), "square_mul factor3"); capture(0);

  a.destroy(h);
  return r;
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

    Run resident = execute(api, exponent, device, true);
    Run canonical = execute(api, exponent, device, false);
    if (resident.transform != canonical.transform || resident.snaps.size() != canonical.snaps.size())
      throw std::runtime_error("run shape mismatch");

    for (size_t i=0; i<resident.snaps.size(); ++i) {
      if (resident.snaps[i] != canonical.snaps[i]) {
        size_t w=0;
        while (w<resident.snaps[i].size() && resident.snaps[i][w]==canonical.snaps[i][w]) ++w;
        std::cerr << "MISMATCH snapshot=" << i << " word=" << w
                  << " resident=0x" << std::hex << hash_words(resident.snaps[i])
                  << " canonical=0x" << hash_words(canonical.snaps[i]) << std::dec << "\n";
        return 1;
      }
      std::cout << "snapshot " << i << " exact hash=0x" << std::hex
                << hash_words(resident.snaps[i]) << std::dec << "\n";
    }

    std::cout << "PFA9 FFT323161 PFA-RESIDENT DIFFERENTIAL TEST PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
