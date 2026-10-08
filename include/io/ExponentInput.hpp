// io/ExponentInput.hpp
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>

namespace io {

// Largest exponent PrMers accepts (command line and interactive prompt alike).
constexpr uint64_t kMaxExponent = 5650242869ULL;
// Every engine and driver holds the exponent in 32 bits; a larger value would be silently truncated
// and a different (smaller) number tested.
constexpr uint64_t kMaxEngineExponent = std::numeric_limits<uint32_t>::max();

// Empty when `exponent` (the exponent actually run, i.e. after any -wagstaff doubling) is within
// both limits, otherwise the error message to print. The command line and the -wagstaff worktodo
// path both go through this so they accept and reject the same values.
inline std::string exponentLimitError(uint64_t exponent, bool wagstaff = false) {
    const std::string suffix = wagstaff ? " (twice the requested Wagstaff exponent)" : "";
    if (exponent > kMaxExponent)
        return "Error: Exponent must be <= " + std::to_string(kMaxExponent) + ". Given: " +
               std::to_string(exponent) + suffix;
    if (exponent > kMaxEngineExponent)
        return "Error: Exponent must be <= " + std::to_string(kMaxEngineExponent) +
               " (the largest exponent the engines support). Given: " + std::to_string(exponent) + suffix;
    return std::string();
}

// Parse an exponent typed at the interactive prompt: plain decimal digits (surrounding blanks allowed)
// with 2 <= value <= kMaxExponent. A negative, zero, garbage or oversized answer is rejected instead of
// wrapping to an enormous unsigned value that would ask for an impossible transform size.
inline bool parseExponentAnswer(const std::string& text, uint64_t& exponent) {
    const size_t a = text.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return false;
    const size_t b = text.find_last_not_of(" \t\r\n");
    const std::string s = text.substr(a, b - a + 1);
    if (s.size() > 19 || !std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
        return false;
    const unsigned long long v = std::stoull(s);
    if (v < 2 || v > kMaxExponent) return false;
    exponent = v;
    return true;
}

} // namespace io
