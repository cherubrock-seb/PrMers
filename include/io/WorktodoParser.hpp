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
    bool ecmTest   = false; 
    bool doubleCheck = false;        // Prime95 DoubleCheck= LL worktodo entry
    bool gaussianMersenne = false;   // Native PrMers Gaussian-Mersenne worktodo entry
    bool gmPrpOnly = false;          // GMPRP rather than deterministic GMPROTH
    bool gmPipeline = false;         // GMCHAIN conditional P-1 -> ECM -> Proth
    bool pminus1ed = true;           // Prime95 Test/DoubleCheck third field; informational for now
    uint32_t exponent = 0;
    std::string aid;
    std::string rawLine;  
    std::vector<std::string> knownFactors;  
    double sieveDepth = 0.0;                // Prime95 Pminus1 how_far_factored (TF depth, e.g. 79 means 2^79)
    uint64_t B2Start = 0;                   // Prime95 Pminus1 optional Stage 2 start bound
    uint32_t residueType = 1;               
    uint64_t B1 = 0;                       
    uint64_t B2 = 0;
    uint64_t curves = 0;
    uint32_t gmBase = 0;
    uint64_t gmSieveLimit = 1'000'000ULL;
    uint64_t gmFactorChunkBits = 0;
    uint64_t gmEcmB1 = 0;
    uint64_t gmEcmB2 = 0;
    uint64_t gmEcmCurves = 0;
    std::string sigma;
};

class WorktodoParser {
public:
    explicit WorktodoParser(const std::string& filename);
    std::optional<WorktodoEntry> parse();
    bool removeFirstProcessed();  // legacy: removes the first actionable entry and archives it
    bool removeProcessedLine(const std::string& rawLine); // exact-line removal for native queues

private:
    std::string filename_;
};

} // namespace io
