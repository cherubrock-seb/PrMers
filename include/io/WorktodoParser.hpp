// io/WorktodoParser.hpp
#pragma once
#include <optional>
#include <string>
#include <cstdint>
#include <vector>

namespace io {

struct WorktodoEntry {
    bool prpTest   = false;
    bool llTest    = false;
    bool pm1Test   = false; 
    uint32_t exponent = 0;
    std::string aid;
    std::string rawLine;  
    std::vector<std::string> knownFactors;  
    uint32_t residueType = 1;               
    uint64_t B1 = 0;                       
    uint64_t B2 = 0;
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
