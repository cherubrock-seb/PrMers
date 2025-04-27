// io/WorktodoParser.cpp
#include "io/WorktodoParser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <vector>

namespace io {

WorktodoParser::WorktodoParser(const std::string& filename)
  : filename_(filename)
{}


static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t i = 0, j;
    while ((j = s.find(sep, i)) != std::string::npos) {
        out.push_back(s.substr(i, j-i));
        i = j+1;
    }
    out.push_back(s.substr(i));
    return out;
}
static bool isHex(const std::string& s) {
    if (s.size() != 32) return false;
    for (char c : s)
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

std::optional<WorktodoEntry> WorktodoParser::parse() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << filename_ << "\n";
        return std::nullopt;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto top = split(line, '=');
        if (top.size() < 2) continue;

        bool isPRP  = (top[0] == "PRP" || top[0] == "PRPDC");
        bool isLL   = (top[0] == "Test" || top[0] == "DoubleCheck");
        if (!(isPRP || isLL)) continue;

        auto parts = split(top[1], ',');
        if (!parts.empty() && (parts[0].empty() || parts[0] == "N/A"))
            parts.erase(parts.begin());

        std::string aid;
        if (!parts.empty() && isHex(parts[0])) {
            aid = parts[0];
            parts.erase(parts.begin());
        }

        if (parts.size() < 4
            || parts[0] != "1"
            || parts[1] != "2"
            || parts[3] != "-1")
            continue;

        try {
            uint32_t exp = static_cast<uint32_t>(std::stoul(parts[2]));
            if (exp == 0) continue;

            WorktodoEntry entry;
            entry.prpTest   = isPRP;
            entry.llTest    = isLL;
            entry.exponent  = exp;
            entry.rawLine   = line;
            entry.aid       = aid;
            std::cout << "Loaded entry: "
                      << (entry.prpTest ? "PRP" : "LL")
                      << " exponent=" << entry.exponent
                      << (aid.empty() ? "" : " (AID=" + aid + ")")
                      << "\n";
            return entry;
        }
        catch (...) {
            continue;
        }
    }

    std::cerr << "No valid entry found in " << filename_ << "\n";
    return std::nullopt;
}

bool WorktodoParser::removeFirstProcessed() {
    std::ifstream inFile(filename_);
    std::ofstream tempFile(filename_ + ".tmp");
    std::ofstream saveFile("worktodo_save.txt", std::ios::app);

    if (!inFile || !tempFile || !saveFile)
        return false;

    std::string line;
    bool skipped = false;

    while (std::getline(inFile, line)) {
        if (!skipped && !line.empty()) {
            skipped = true;
            saveFile << line << "\n";
            continue;
        }
        tempFile << line << "\n";
    }

    inFile.close();
    tempFile.close();
    saveFile.close();

    std::remove(filename_.c_str());
    std::rename((filename_ + ".tmp").c_str(), filename_.c_str());

    return skipped;
}



} // namespace io
