#include "OpenCLSourceBuilder.h"

#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

const std::string* findSource(
    std::string_view name,
    const std::vector<std::pair<std::string, std::string>>& files) {
  for (const auto& [fileName, source] : files) {
    if (fileName == name) return &source;
  }
  return nullptr;
}

std::string trimLeft(std::string_view line) {
  size_t pos = 0;
  while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
  return std::string(line.substr(pos));
}

bool containsPragmaOnce(const std::string& source) {
  size_t begin = 0;
  while (begin <= source.size()) {
    size_t end = source.find('\n', begin);
    if (end == std::string::npos) end = source.size();
    if (trimLeft(std::string_view(source).substr(begin, end - begin)) == "#pragma once") return true;
    if (end == source.size()) break;
    begin = end + 1;
  }
  return false;
}

bool quotedInclude(std::string_view line, std::string& includeName) {
  std::string t = trimLeft(line);
  constexpr std::string_view prefix = "#include";
  if (!t.starts_with(prefix)) return false;
  size_t pos = prefix.size();
  while (pos < t.size() && std::isspace(static_cast<unsigned char>(t[pos]))) ++pos;
  if (pos >= t.size() || t[pos] != '"') return false;
  size_t close = t.find('"', pos + 1);
  if (close == std::string::npos) return false;
  includeName = t.substr(pos + 1, close - pos - 1);
  return true;
}

std::string expand(
    std::string_view name,
    const std::vector<std::pair<std::string, std::string>>& files,
    std::unordered_set<std::string>& onceFiles,
    std::unordered_set<std::string>& active) {
  const std::string* source = findSource(name, files);
  if (!source) throw std::runtime_error("Bundled OpenCL source not found: " + std::string(name));

  const bool pragmaOnce = containsPragmaOnce(*source);
  if (pragmaOnce && onceFiles.contains(std::string(name))) return {};
  if (active.contains(std::string(name))) {
    throw std::runtime_error("Recursive bundled OpenCL include: " + std::string(name));
  }
  if (pragmaOnce) onceFiles.insert(std::string(name));
  active.insert(std::string(name));

  std::string out;
  out.reserve(source->size() + 128);
  out += "\n#line 1 \"" + std::string(name) + "\"\n";

  size_t begin = 0;
  size_t lineNumber = 1;
  while (begin <= source->size()) {
    size_t end = source->find('\n', begin);
    if (end == std::string::npos) end = source->size();
    std::string_view line(*source);
    line = line.substr(begin, end - begin);

    std::string includeName;
    const std::string trimmed = trimLeft(line);
    if (trimmed == "#pragma once") {
      out += "// #pragma once expanded by Aevum\n";
    } else if (quotedInclude(line, includeName) && findSource(includeName, files)) {
      out += expand(includeName, files, onceFiles, active);
      out += "\n#line " + std::to_string(lineNumber + 1) + " \"" + std::string(name) + "\"\n";
    } else {
      out.append(line.data(), line.size());
      out.push_back('\n');
    }

    if (end == source->size()) break;
    begin = end + 1;
    ++lineNumber;
  }

  active.erase(std::string(name));
  return out;
}

}  // namespace

std::string buildMonolithicOpenCLSource(
    std::string_view rootFile,
    const std::vector<std::pair<std::string, std::string>>& files) {
  std::unordered_set<std::string> onceFiles;
  std::unordered_set<std::string> active;
  return expand(rootFile, files, onceFiles, active);
}
