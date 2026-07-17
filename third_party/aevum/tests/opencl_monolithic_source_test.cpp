#include "../src/OpenCLSourceBuilder.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static std::string readFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int main(int argc, char** argv) {
  {
    const std::vector<std::pair<std::string, std::string>> files{
        {"root.cl", "#include \"once.cl\"\n#include \"once.cl\"\n#include \"repeat.cl\"\n#include \"repeat.cl\"\nkernel void k() {}\n"},
        {"once.cl", "#pragma once\nint once_symbol;\n"},
        {"repeat.cl", "int repeat_symbol;\n"},
    };
    const std::string out = buildMonolithicOpenCLSource("root.cl", files);
    assert(out.find("#include \"") == std::string::npos);
    assert(out.find("once_symbol") == out.rfind("once_symbol"));
    assert(out.find("repeat_symbol") != out.rfind("repeat_symbol"));
    assert(out.find("kernel void k") != std::string::npos);
  }

  std::vector<std::pair<std::string, std::string>> bundled;
  for (const auto& entry : fs::directory_iterator("src/cl")) {
    if (entry.is_regular_file() && entry.path().extension() == ".cl") {
      bundled.push_back({entry.path().filename().string(), readFile(entry.path())});
    }
  }
  assert(!bundled.empty());

  if (argc == 3 && std::string(argv[1]) == "--dump-fftp") {
    std::ofstream out(argv[2], std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + std::string(argv[2]));
    out << buildMonolithicOpenCLSource("fftp.cl", bundled);
  } else if (argc == 4 && std::string(argv[1]) == "--dump-root") {
    std::ofstream out(argv[3], std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + std::string(argv[3]));
    out << buildMonolithicOpenCLSource(argv[2], bundled);
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0]
              << " [--dump-fftp output.cl] [--dump-root root.cl output.cl]\n";
    return 2;
  }

  size_t expanded = 0;
  for (const auto& [name, source] : bundled) {
    const std::string out = buildMonolithicOpenCLSource(name, bundled);
    assert(!out.empty());
    assert(out.find("#include \"") == std::string::npos);
    assert(out.find("#line 1 \"" + name + "\"") != std::string::npos);
    ++expanded;
  }

  std::cout << "Aevum monolithic OpenCL source test passed for " << expanded << " bundled files\n";
}
