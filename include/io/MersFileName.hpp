// io/MersFileName.hpp
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>

namespace io {

// Parse the name of a P-1 stage 1 state file, "<p>pm<B1>.mers" (for example "127pm5000.mers"), into the
// exponent p (1 .. 2^32 - 1) and the bound B1. Returns false when the name does not have that form:
// both numbers must be plain decimal digits, so the std::stoul/stoull calls that follow cannot throw.
inline bool parseMersFileName(const std::string& fname, uint32_t& p, uint64_t& B1) {
    const size_t pos_pm = fname.find("pm");
    const size_t pos_dot = fname.rfind('.');
    if (pos_pm == std::string::npos || pos_dot == std::string::npos || pos_pm == 0 || pos_pm + 2 >= pos_dot)
        return false;
    const std::string p_str = fname.substr(0, pos_pm);
    const std::string b1_str = fname.substr(pos_pm + 2, pos_dot - (pos_pm + 2));
    auto digits = [](const std::string& s, size_t maxLen) {
        return !s.empty() && s.size() <= maxLen &&
               std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    };
    if (!digits(p_str, 10) || !digits(b1_str, 19)) return false;
    const unsigned long long pv = std::stoull(p_str);
    if (pv == 0 || pv > std::numeric_limits<uint32_t>::max()) return false;
    p = static_cast<uint32_t>(pv);
    B1 = std::stoull(b1_str);
    return true;
}

} // namespace io
