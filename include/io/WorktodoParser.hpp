// io/WorktodoParser.hpp
#pragma once
#include <mutex>
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
    bool gmPipeline = false;         // GMCHAIN conditional P-1 -> optional ECM
    std::string gmFamily = "GM";      // GM, GQ, or BOTH
    bool gmPipelineProth = true;     // legacy default; false stops after factoring
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

// Exponent to test for a worktodo entry when -wagstaff is given, or 0 when
// the entry cannot be a Wagstaff test. -wagstaff tests (2^p+1)/3 by running
// the PRP on 2^(2p) - 1, so a PRP entry's p is doubled exactly as the command
// line exponent is. Other entry types and cofactor entries (the known factors
// divide 2^p - 1, not (2^p+1)/3) are not Wagstaff work.
inline uint64_t wagstaffExponentForEntry(const WorktodoEntry& e) {
    if (!e.prpTest || e.gaussianMersenne || !e.knownFactors.empty()) return 0;
    return 2ULL * e.exponent;
}

class WorktodoParser {
public:
    explicit WorktodoParser(const std::string& filename);
    std::optional<WorktodoEntry> parse();
    // True when `line` alone is an entry parse() would run (the check the GUI applies before appending a
    // line to worktodo). Comments, blank lines, unsupported keywords and malformed entries are rejected,
    // and so is any text containing a line break.
    static bool isValidEntryLine(const std::string& line);
    // Append one line to a worktodo file (the GUI "Append & Run" path). If the file does not end in a
    // newline (hand-written files and many editors leave none), start a new line first so the new entry
    // is not glued onto the last one. Serialised with removeProcessedLine(), so an append cannot be lost to a
    // concurrent rewrite of the file. Returns false when the file cannot be written.
    static bool appendLine(const std::string& path, const std::string& line);
    // Remove the line that was actually run (WorktodoEntry::rawLine) and archive it to worktodo_save.txt.
    // parse() skips lines it cannot run, so "the first actionable line" is not necessarily that line.
    bool removeProcessedLine(const std::string& rawLine);
    // True when parse() would return an entry for this file: a dry run of parse() with no output.
    // Blank lines, '#' and ';' comments, unsupported keywords and lines parse() rejects (malformed
    // fields, a Pfactor with no bounded P-1 work, invalid known factors, glued entries) do not count,
    // so "restart for the next entry" is only taken when the restarted process will find one.
    // A GMTF= line counts when it is the first actionable line: the Gaussian trial-factoring
    // pre-parser runs it on the restart.
    static bool hasPendingEntry(const std::string& filename);
    // Take the lock appendLine()/removeProcessedLine() hold while they write. restart_self holds it
    // through the exec, so a GUI append on another thread is never cut off half-written (a truncated
    // line can parse as a different exponent). The lock is never released if the restart succeeds.
    static std::unique_lock<std::mutex> lockFileWrites();

private:
    std::string filename_;
    bool quiet_ = false;   // dry run: parse() prints nothing
    const std::string* text_ = nullptr;   // isValidEntryLine(): parse this text instead of the file
};

} // namespace io
