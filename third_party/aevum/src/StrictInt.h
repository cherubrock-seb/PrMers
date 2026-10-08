// Strict integer parsing for -use values.  atoi() maps "garbage" to 0 and "3x" to 3, so a mistyped -use value would
// pass a range check that was meant to refuse it (and the same text is forwarded to the kernels as a -D define).

#pragma once

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>

// Parse `text` as a decimal int: an optional sign followed by digits, nothing else (no leading/trailing space or
// text, no hex, no empty string), and the value must fit in an int.  Returns false and leaves `out` unchanged if not.
inline bool parseStrictInt(const std::string& text, int& out) {
  if (text.empty() || std::isspace(static_cast<unsigned char>(text[0]))) return false;
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return false;
  out = int(v);
  return true;
}
