// io/WorktodoParser.hpp
#pragma once
#include <optional>
#include <string>
#include <cstdint>
#include <vector>

namespace io {

struct WorktodoEntry {
    bool prpTest;
    bool llTest;
    uint32_t exponent;
    std::string aid;
    std::string rawLine;  
    // Cofactor test support
    std::vector<std::string> knownFactors;  // List of known factors as strings
    uint32_t residueType = 1;               // Default Type 1, Type 5 for cofactors
};

class WorktodoParser {
public:
    explicit WorktodoParser(const std::string& filename);
    std::optional<WorktodoEntry> parse();
    bool removeFirstProcessed();  // supprime la 1ʳᵉ entrée non vide et la sauvegarde

private:
    std::string filename_;
};

} // namespace io
