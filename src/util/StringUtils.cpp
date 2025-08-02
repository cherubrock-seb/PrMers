#include "util/StringUtils.hpp"

namespace util {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t i = 0, j;
    while ((j = s.find(sep, i)) != std::string::npos) {
        out.push_back(s.substr(i, j-i));
        i = j+1;
    }
    out.push_back(s.substr(i));
    return out;
}

} // namespace util
