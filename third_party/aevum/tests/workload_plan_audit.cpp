#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Handle = void*;

struct Api {
  void* so{};
  const char* (*last_error)(){};
  int (*resolve)(std::uint32_t, const char*, char*, std::size_t){};
  Handle (*create)(std::uint32_t, std::size_t, std::uint32_t, int,
                   const char*, const char*){};
  void (*destroy)(Handle){};
  std::size_t (*transform_size)(Handle){};
  std::size_t (*word_count)(Handle){};
  int (*sync)(Handle){};
  int (*set_u32)(Handle, std::size_t, std::uint32_t){};
  int (*get_words)(Handle, std::size_t, std::uint32_t*, std::size_t){};
  int (*copy)(Handle, std::size_t, std::size_t){};
  int (*prepare)(Handle, std::size_t, std::size_t){};
  int (*square_mul)(Handle, std::size_t, std::uint32_t){};
  int (*mul)(Handle, std::size_t, std::size_t, std::uint32_t){};
  int (*add)(Handle, std::size_t, std::size_t){};
  int (*sub_reg)(Handle, std::size_t, std::size_t){};

  template <typename T> T sym(const char* name) {
    void* ptr = dlsym(so, name);
    if (!ptr) throw std::runtime_error(std::string("missing symbol ") + name);
    return reinterpret_cast<T>(ptr);
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
    sync = sym<decltype(sync)>("aevum_engine_sync");
    set_u32 = sym<decltype(set_u32)>("aevum_engine_set_u32");
    get_words = sym<decltype(get_words)>("aevum_engine_get_words");
    copy = sym<decltype(copy)>("aevum_engine_copy");
    prepare = sym<decltype(prepare)>("aevum_engine_prepare");
    square_mul = sym<decltype(square_mul)>("aevum_engine_square_mul");
    mul = sym<decltype(mul)>("aevum_engine_mul");
    add = sym<decltype(add)>("aevum_engine_add");
    sub_reg = sym<decltype(sub_reg)>("aevum_engine_sub_reg");
  }

  ~Api() {
    if (so) dlclose(so);
  }
};

static void require(Api& api, int rc, const char* operation) {
  if (rc) return;
  throw std::runtime_error(std::string(operation) + ": " +
                           (api.last_error() ? api.last_error() : "unknown"));
}

static std::uint64_t hash_words(const std::vector<std::uint32_t>& words) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint32_t word : words) {
    hash ^= word;
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct Run {
  std::string requested;
  std::string resolved;
  std::string workload;
  std::size_t transform{};
  std::size_t word_count{};
  unsigned iterations{};
  double seconds{};
  std::vector<std::vector<std::uint32_t>> outputs;
};

static std::size_t register_count(const std::string& workload) {
  if (workload == "prp" || workload == "ll") return 8;
  if (workload == "pm1") return 11;
  if (workload == "ecm") return 51;
  throw std::runtime_error("workload must be prp, ll, pm1, or ecm");
}

static void initialize(Api& api, Handle handle, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t value = static_cast<std::uint32_t>(3 + (index * 17) % 251);
    require(api, api.set_u32(handle, index, value), "set_u32");
  }
}

static void step_prp(Api& api, Handle handle, unsigned iteration) {
  const std::uint32_t factor = (iteration % 127u == 126u) ? 3u : 1u;
  require(api, api.square_mul(handle, 0, factor), "prp square_mul");
  if ((iteration & 255u) == 255u)
    require(api, api.copy(handle, 1, 0), "prp checkpoint copy");
}

static void step_pm1(Api& api, Handle handle, unsigned iteration) {
  // A fixed non-trivial bit pattern exercises the ordinary square plus base-3
  // multiply path used by P-1 Stage 1.  Periodic prepare/mul/copy calls model
  // the Gerbicz checkpoint work without changing the deterministic result.
  const std::uint32_t factor = ((iteration * 0x9e3779b9u) >> 31u) ? 3u : 1u;
  require(api, api.square_mul(handle, 0, factor), "pm1 square_mul");
  if ((iteration & 63u) == 63u) {
    require(api, api.copy(handle, 1, 0), "pm1 copy");
    require(api, api.prepare(handle, 2, 1), "pm1 prepare");
    require(api, api.copy(handle, 3, 0), "pm1 destination copy");
    require(api, api.mul(handle, 3, 2, 1), "pm1 prepared mul");
    require(api, api.copy(handle, 4, 3), "pm1 result copy");
  }
}

static void step_ecm(Api& api, Handle handle, unsigned iteration) {
  // Use 16 canonical point-like registers, 16 work registers and 16 prepared
  // operands.  Cycling through more operands than the default prepared cache
  // deliberately exercises eviction/rebuild behavior used by ECM.
  const std::size_t a = iteration % 16u;
  const std::size_t b = (iteration * 5u + 3u) % 16u;
  const std::size_t work = 16u + (iteration % 16u);
  const std::size_t prepared = 32u + (iteration % 16u);

  require(api, api.copy(handle, work, a), "ecm copy");
  require(api, api.add(handle, work, b), "ecm add");
  require(api, api.square_mul(handle, work, 1), "ecm square");
  require(api, api.prepare(handle, prepared, b), "ecm prepare");
  require(api, api.mul(handle, work, prepared, 1), "ecm prepared mul");
  require(api, api.sub_reg(handle, work, a), "ecm subtract");
  require(api, api.add(handle, work, b), "ecm final add");
  require(api, api.copy(handle, a, work), "ecm point update");
}

static Run execute(Api& api, std::uint32_t exponent, std::uint32_t device,
                   const std::string& workload, const std::string& spec,
                   unsigned iterations, const char* tune_dir) {
  char resolved[256]{};
  require(api, api.resolve(exponent, spec.c_str(), resolved, sizeof(resolved)),
          "resolve FFT");

  const std::size_t registers = register_count(workload);
  Handle handle = api.create(exponent, registers, device, 0, spec.c_str(), tune_dir);
  if (!handle)
    throw std::runtime_error(api.last_error() ? api.last_error() : "create failed");

  Run result;
  result.requested = spec;
  result.resolved = resolved;
  result.workload = workload;
  result.transform = api.transform_size(handle);
  result.word_count = api.word_count(handle);
  result.iterations = iterations;

  try {
    initialize(api, handle, registers);
    require(api, api.sync(handle), "initial sync");

    const auto begin = std::chrono::steady_clock::now();
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
      if (workload == "prp" || workload == "ll")
        step_prp(api, handle, iteration);
      else if (workload == "pm1")
        step_pm1(api, handle, iteration);
      else
        step_ecm(api, handle, iteration);
    }
    require(api, api.sync(handle), "final sync");
    result.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();

    const std::size_t captures = workload == "ecm" ? 4u : 1u;
    for (std::size_t reg = 0; reg < captures; ++reg) {
      std::vector<std::uint32_t> words(result.word_count);
      require(api, api.get_words(handle, reg, words.data(), words.size()),
              "get_words");
      result.outputs.push_back(std::move(words));
    }
  } catch (...) {
    api.destroy(handle);
    throw;
  }

  api.destroy(handle);
  return result;
}

static void print_run(const Run& run) {
  std::uint64_t combined = 1469598103934665603ULL;
  for (const auto& output : run.outputs) {
    combined ^= hash_words(output);
    combined *= 1099511628211ULL;
  }
  const double iterations_per_second =
      run.iterations / std::max(run.seconds, 1.0e-12);
  std::cout << "PLAN"
            << " workload=" << run.workload
            << " requested=" << run.requested
            << " resolved=" << run.resolved
            << " transform=" << run.transform
            << " iterations=" << run.iterations
            << " seconds=" << std::fixed << std::setprecision(6) << run.seconds
            << " iter_per_s=" << std::setprecision(3) << iterations_per_second
            << " hash=0x" << std::hex << combined << std::dec << '\n';
}

int main(int argc, char** argv) {
  try {
    if (argc < 8) {
      std::cerr << "usage: " << argv[0]
                << " libaevum_engine.so device exponent workload spec-a spec-b iterations [tune-dir]\n";
      return 2;
    }

    Api api(argv[1]);
    const std::uint32_t device = static_cast<std::uint32_t>(std::stoul(argv[2]));
    const std::uint32_t exponent = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const std::string workload = argv[4];
    const std::string left_spec = argv[5];
    const std::string right_spec = argv[6];
    const unsigned iterations = static_cast<unsigned>(std::stoul(argv[7]));
    const char* tune_dir = argc > 8 ? argv[8] : ".";

    Run left = execute(api, exponent, device, workload, left_spec, iterations, tune_dir);
    Run right = execute(api, exponent, device, workload, right_spec, iterations, tune_dir);
    print_run(left);
    print_run(right);

    if (left.outputs.size() != right.outputs.size())
      throw std::runtime_error("output-count mismatch");
    for (std::size_t output = 0; output < left.outputs.size(); ++output) {
      if (left.outputs[output] == right.outputs[output]) continue;
      std::size_t word = 0;
      while (word < left.outputs[output].size() &&
             left.outputs[output][word] == right.outputs[output][word])
        ++word;
      std::cerr << "MISMATCH workload=" << workload
                << " output=" << output << " word=" << word
                << " left_hash=0x" << std::hex << hash_words(left.outputs[output])
                << " right_hash=0x" << hash_words(right.outputs[output])
                << std::dec << '\n';
      return 1;
    }

    std::cout << "AEVUM WORKLOAD PLAN DIFFERENTIAL PASSED workload=" << workload
              << " left=" << left.resolved << " right=" << right.resolved << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
