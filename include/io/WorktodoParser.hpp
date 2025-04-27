// io/WorktodoParser.hpp
#pragma once
#include <optional>
#include <string>
#include <cstdint>

namespace io {

struct WorktodoEntry {
    bool prpTest;
    bool llTest;
    uint32_t exponent;
    std::string aid;
    std::string rawLine;  
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
