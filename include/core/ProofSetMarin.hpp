#pragma once

#include "core/ProofMarin.hpp"
#include <cstdint>
#include <vector>
#include <filesystem>
#include <map>
#include <string>

namespace core {

class WordsMarin {
public:
    WordsMarin();
    explicit WordsMarin(const std::vector<uint64_t>& v);

    const std::vector<uint64_t>& data() const noexcept;
    std::vector<uint64_t>& data() noexcept;

    static WordsMarin fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);

private:
    std::vector<uint64_t> data_;
};

class ProofSetMarin {
public:
    const uint32_t E;     // exponent
    uint32_t power;       // proof power level (see setPower)
    const std::vector<std::string> knownFactors; // known factors (for cofactor tests)

    ProofSetMarin(uint32_t exponent, uint32_t proofLevel, std::vector<std::string> factors = {});

    bool shouldCheckpoint(uint32_t iter) const;
    // Lower the power residues are saved for (a resumed test may lack the
    // residues of earlier points). The points of a lower power are a subset
    // of those of the original power.
    void setPower(uint32_t newPower);
    void save(uint32_t iter, const std::vector<uint32_t>& words);
    std::vector<uint32_t> load(uint32_t iter) const;

    static WordsMarin fromUint64(const std::vector<uint64_t>& host, uint32_t exponent);
    static uint32_t bestPower(uint32_t E);
    static bool isInPoints(uint32_t E, uint32_t power, uint32_t k);
    static std::filesystem::path proofPath(uint32_t E);
    // Remove the saved proof residues of exponent E (<E>/proof, and <E> when
    // that leaves it empty). Call when the test is over and the proof has
    // been made or given up; a test that can still be resumed needs them.
    static void clearResidues(uint32_t E);

    // What to do with the residues once a PRP test has finished. They are
    // deleted only when the result is saved, the worktodo entry (if any) is
    // retired and any requested proof has been made (and verified, unless
    // -noverify); otherwise they are kept, because a failed or interrupted
    // proof can still be retried from them, and a result that could not be
    // saved or an entry still in the worktodo means the test is run again.
    enum class ResidueAction {
        NotApplicable,       // not a Mersenne PRP test: nothing to do
        Clear,
        KeepResultNotSaved,
        KeepEntryNotRetired,
        KeepProofFailed
    };
    static ResidueAction residueAction(bool isPrp, bool wagstaff,
                                       bool proofRequested, bool proofCompleted,
                                       bool resultSaved, bool entryRetired);
    // Message for the two Keep actions: where the residues are and that they
    // can be deleted by hand.
    static std::string residuesKeptMessage(uint32_t E, ResidueAction action);
    static double diskUsageGB(uint32_t E, uint32_t power);
    // Checkpoint iterations of a proof of this power, ascending, ending with E.
    static std::vector<uint32_t> proofPoints(uint32_t E, uint32_t power);
    // True when every residue a proof of this power needs from iterations up
    // to currentK is on disk and reads back intact (size and CRC).
    static bool canDo(uint32_t E, uint32_t power, uint32_t currentK);
    // The highest power <= power that canDo for a test resumed at currentK,
    // or 0 when no proof is possible.
    static uint32_t effectivePower(uint32_t E, uint32_t power, uint32_t currentK);
    
    // Core proof generation algorithm
    ProofMarin computeProof() const;

private:
    std::vector<uint32_t> points; // checkpoint iteration points
    
    static bool fileExists(uint32_t E, uint32_t k);
    // canDo with the result of each residue check remembered in `checked`,
    // so that effectivePower reads each residue at most once.
    static bool canDo(uint32_t E, uint32_t power, uint32_t currentK,
                      std::map<uint32_t, bool>& checked);
    static std::vector<uint32_t> loadResidue(uint32_t E, uint32_t iter);
};

} // namespace core