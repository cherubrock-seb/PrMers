#pragma once
#include <cctype>
#include <sstream>
#include <string>

// Rewrite only the named, bounded CUDA entry. CUDA 13/PTX 9.0 introduced the
// function-scoped pragma; do not relabel an older PTX version to force it.
inline bool aevumSmemSpillPtx(const std::string& ptx, const std::string& name,
                            std::string& candidate) {
  const auto version = ptx.find(".version ");
  if (version == std::string::npos) return false;
  std::istringstream vs(ptx.substr(version + 9));
  unsigned major = 0, minor = 0;
  char dot = 0;
  if (!(vs >> major >> dot >> minor) || dot != '.' || major < 9) return false;
  auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)); };
  auto ident = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' || c == '.';
  };
  for (size_t pos = 0; (pos = ptx.find(".entry", pos)) != std::string::npos;) {
    size_t cur = pos + 6;
    pos = cur;
    if (cur == ptx.size() || !space(ptx[cur])) continue;
    while (cur < ptx.size() && space(ptx[cur])) ++cur;
    const size_t begin = cur;
    while (cur < ptx.size() && ident(ptx[cur])) ++cur;
    if (ptx.substr(begin, cur - begin) != name) continue;
    while (cur < ptx.size() && space(ptx[cur])) ++cur;
    if (cur == ptx.size() || ptx[cur] != '(') return false;
    unsigned depth = 1;
    for (++cur; cur < ptx.size() && depth; ++cur) {
      if (ptx[cur] == '(') ++depth;
      if (ptx[cur] == ')') --depth;
    }
    if (depth) return false;
    const auto body = ptx.find('{', cur);
    const auto semicolon = ptx.find(';', cur);
    const auto bound = ptx.find(".maxntid", cur);
    if (body == std::string::npos ||
        (semicolon != std::string::npos && semicolon < body) ||
        bound == std::string::npos || bound > body) return false;
    size_t end = body + 1;
    depth = 1;
    for (; end < ptx.size() && depth; ++end) {
      if (ptx[end] == '{') ++depth;
      if (ptx[end] == '}') --depth;
    }
    if (depth) return false;
    const std::string pragma = "\n.pragma \"enable_smem_spilling\";";
    candidate = ptx;
    if (ptx.substr(body + 1, end - body - 2).find("enable_smem_spilling") == std::string::npos)
      candidate.insert(body + 1, pragma);
    return true;
  }
  return false;
}
