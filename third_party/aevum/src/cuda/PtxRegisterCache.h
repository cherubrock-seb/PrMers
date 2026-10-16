#pragma once

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

// PRPLL f222e885 separates frontend compilation from per-entry register limits.
// Keep OFF options byte-for-byte intact and put neutral ON images in a distinct
// cache namespace, so a previously limited PTX can never masquerade as neutral.
struct AevumPtxRegisterLimit {
  std::string compileArgs;
  bool enabled{false};
  int maxRegs{0}, minBlocks{0};
};

namespace aevum_ptx_detail {
inline int positive(const std::string& value, int maximum) {
  if (value.empty()) throw std::runtime_error("empty PTX register limit");
  int result = 0;
  for (unsigned char c : value) {
    if (c < '0' || c > '9' || result > maximum / 10)
      throw std::runtime_error("invalid PTX register limit");
    result = result * 10 + (c - '0');
    if (result > maximum) throw std::runtime_error("PTX register limit out of range");
  }
  if (!result) throw std::runtime_error("PTX register limit must be positive");
  return result;
}
inline bool space(char c) { return std::isspace(static_cast<unsigned char>(c)); }
inline bool identifier(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
// Mask comments and strings without moving offsets. A debug filename/comment
// containing '.entry name(' must never select an unrelated kernel.
inline std::string codeOnly(const std::string& ptx) {
  std::string code = ptx;
  for (size_t i = 0; i < code.size();) {
    if (code.compare(i, 2, "//") == 0) {
      const size_t end = code.find('\n', i);
      const size_t stop = end == std::string::npos ? code.size() : end;
      code.replace(i, stop - i, stop - i, ' '); i = stop;
    } else if (code.compare(i, 2, "/*") == 0) {
      const size_t end = code.find("*/", i + 2);
      if (end == std::string::npos) throw std::runtime_error("unterminated PTX comment");
      code.replace(i, end + 2 - i, end + 2 - i, ' '); i = end + 2;
    } else if (code[i] == '"') {
      const size_t start = i++;
      while (i < code.size() && code[i] != '"') {
        if (code[i] == '\\' && i + 1 < code.size()) ++i;
        ++i;
      }
      if (i == code.size()) throw std::runtime_error("unterminated PTX string");
      ++i; code.replace(start, i - start, i - start, ' ');
    } else ++i;
  }
  return code;
}
}

inline AevumPtxRegisterLimit aevumTakePtxRegisterLimit(const std::string& args) {
  AevumPtxRegisterLimit result{args};
  std::istringstream scan(args);
  for (std::string token; scan >> token;) {
    if (token.rfind("--aevum-reg-ptx", 0) != 0) continue;
    if (token != "--aevum-reg-ptx=1" || result.enabled)
      throw std::runtime_error("invalid or duplicate PTX register-cache marker");
    result.enabled = true;
  }
  if (!result.enabled) return result;
  result.compileArgs.clear();
  std::istringstream input(args);
  for (std::string token; input >> token;) {
    if (token == "--aevum-reg-ptx=1") continue;
    const std::string maxPrefix = "--maxrregcount=", minPrefix = "-DCUDA_MIN_BLOCKS=";
    if (token.rfind(maxPrefix, 0) == 0 || token.rfind(minPrefix, 0) == 0) {
      if (result.maxRegs || result.minBlocks)
        throw std::runtime_error("multiple PTX register limits");
      if (token.rfind(maxPrefix, 0) == 0)
        result.maxRegs = aevum_ptx_detail::positive(token.substr(maxPrefix.size()), 255);
      else
        result.minBlocks = aevum_ptx_detail::positive(token.substr(minPrefix.size()), 16);
    } else {
      if (token.rfind("--maxrregcount", 0) == 0 || token.rfind("-DCUDA_MIN_BLOCKS", 0) == 0)
        throw std::runtime_error("malformed PTX register limit option");
      result.compileArgs += token + ' ';
    }
  }
  if (!result.maxRegs && !result.minBlocks)
    throw std::runtime_error("PTX register cache requires a resolved positive limit");
  result.compileArgs += "-DAEVUM_CUDA_REG_PTX_BASE=1 ";
  return result;
}

inline std::string aevumPtxWithRegisterLimit(const std::string& ptx, const std::string& name,
                                           int maxRegs, int minBlocks) {
  using namespace aevum_ptx_detail;
  if (name.empty() || maxRegs < 0 || maxRegs > 255 || minBlocks < 0 || minBlocks > 16 ||
      (maxRegs > 0) == (minBlocks > 0))
    throw std::runtime_error("invalid named PTX register-limit request");
  const std::string code = codeOnly(ptx);
  size_t entry = std::string::npos;
  for (size_t pos = 0; (pos = code.find(".entry", pos)) != std::string::npos; pos += 6) {
    if ((pos && identifier(code[pos - 1])) || pos + 6 == code.size() || !space(code[pos + 6])) continue;
    size_t start = pos + 6;
    while (start < code.size() && space(code[start])) ++start;
    size_t end = start;
    while (end < code.size() && identifier(code[end])) ++end;
    if (code.substr(start, end - start) != name) continue;
    if (entry != std::string::npos) throw std::runtime_error("duplicate named PTX entry");
    entry = end;
  }
  if (entry == std::string::npos) throw std::runtime_error("named PTX entry not found");
  while (entry < code.size() && space(code[entry])) ++entry;
  if (entry == code.size() || code[entry] != '(') throw std::runtime_error("malformed PTX entry parameters");
  const size_t paramsEnd = code.find(')', entry);
  const size_t body = code.find('{', entry);
  if (paramsEnd == std::string::npos || body == std::string::npos || paramsEnd > body)
    throw std::runtime_error("missing PTX entry body");
  const std::string header = code.substr(paramsEnd + 1, body - paramsEnd - 1);
  if (header.find(".entry") != std::string::npos || header.find(".func") != std::string::npos ||
      header.find(".maxnreg") != std::string::npos || header.find(".minnctapersm") != std::string::npos ||
      header.find(".maxnctapersm") != std::string::npos)
    throw std::runtime_error("PTX register cache requires an unbounded entry");
  const size_t maxntid = header.find(".maxntid");
  if (maxntid == std::string::npos || maxntid + 8 == header.size() || !space(header[maxntid + 8]))
    throw std::runtime_error("PTX entry lacks its workgroup bound");
  const std::string directive = maxRegs ? ".maxnreg " + std::to_string(maxRegs) + "\n" :
      ".minnctapersm " + std::to_string(minBlocks) + "\n";
  std::string result = ptx;
  result.insert(paramsEnd + 1 + maxntid, directive);
  return result;
}
