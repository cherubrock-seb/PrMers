#include "io/WorktodoParser.hpp"
#include <fstream>
#include <sstream>

namespace io {

WorktodoParser::WorktodoParser(const std::string& filename)
    : filename_(filename) {}

std::optional<WorktodoEntry> WorktodoParser::parse() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        return std::nullopt;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // skip empty or comment lines
        }

        std::istringstream iss(line);
        WorktodoEntry entry{};
        std::string type;
        iss >> type;

        if (type == "P" || type == "PRP") {
            entry.prpTest = true;
            entry.llTest = false;
        } else if (type == "F" || type == "LL") {
            entry.prpTest = false;
            entry.llTest = true;
        } else {
            continue; // unknown line, skip
        }

        iss >> entry.exponent;

        if (entry.exponent > 0) {
            return entry;
        }
    }

    return std::nullopt;
}

} // namespace io
