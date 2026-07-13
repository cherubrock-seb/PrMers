#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) throw std::runtime_error(dlerror());

  const char* names[] = {
      "aevum_engine_version",
      "aevum_engine_last_error",
      "aevum_engine_resolve_fft",
      "aevum_engine_create",
      "aevum_engine_destroy",
      "aevum_engine_transform_size",
      "aevum_engine_word_count",
      "aevum_engine_sync",
      "aevum_engine_set_u32",
      "aevum_engine_set_words",
      "aevum_engine_get_words",
      "aevum_engine_copy",
      "aevum_engine_prepare",
      "aevum_engine_square_mul",
      "aevum_engine_mul",
      "aevum_engine_add",
      "aevum_engine_sub_reg",
      "aevum_engine_sub_u32",
      "aevum_engine_equal",
  };

  for (const char* name : names) {
    if (!dlsym(lib, name)) throw std::runtime_error(std::string("missing symbol: ") + name);
  }

  using version_fn = const char* (*)();
  using error_fn = const char* (*)();
  using resolve_fn = int (*)(uint32_t, const char*, char*, size_t);
  auto version = reinterpret_cast<version_fn>(dlsym(lib, "aevum_engine_version"));
  auto error = reinterpret_cast<error_fn>(dlsym(lib, "aevum_engine_last_error"));
  auto resolve = reinterpret_cast<resolve_fn>(dlsym(lib, "aevum_engine_resolve_fft"));

  char small[64]{};
  if (resolve(216091u, nullptr, small, sizeof(small))) {
    throw std::runtime_error(std::string("undersized exponent unexpectedly resolved to ") + small);
  }
  std::cout << "Aevum FFT 216091 -> Marin fallback: " << error() << std::endl;

  for (uint32_t exponent : std::vector<uint32_t>{1362763u, 136279841u, 2147483647u}) {
    char spec[64]{};
    if (!resolve(exponent, nullptr, spec, sizeof(spec))) {
      throw std::runtime_error(std::string("auto FFT failed: ") + error());
    }
    std::vector<std::string> parts;
    std::string item;
    std::istringstream in(spec);
    while (std::getline(in, item, ':')) parts.push_back(item);
    if (parts.size() < 5) throw std::runtime_error(std::string("bad resolved FFT: ") + spec);
    const unsigned type = static_cast<unsigned>(std::stoul(parts[0]));
    const unsigned middle = static_cast<unsigned>(std::stoul(parts[2]));
    const unsigned width = static_cast<unsigned>(std::stoul(parts[1]));
    const unsigned height = static_cast<unsigned>(std::stoul(parts[3]));
    const uint64_t words = uint64_t(width) * middle * height * 2;
    const double bpw = exponent / double(words);
    if (type != 1 || middle < 2 || (middle & (middle - 1)) != 0 || bpw < 3.0) {
      throw std::runtime_error(std::string("invalid FFT3161 plan: ") + spec);
    }
    std::cout << "Aevum FFT " << exponent << " -> " << spec << std::endl;
  }

  char invalid[64]{};
  if (resolve(216091u, "1:1024:13:256", invalid, sizeof(invalid))) {
    throw std::runtime_error("invalid NTT middle was accepted");
  }

  std::cout << "Aevum engine API loaded, version=" << version() << std::endl;
  dlclose(lib);
  return 0;
}
