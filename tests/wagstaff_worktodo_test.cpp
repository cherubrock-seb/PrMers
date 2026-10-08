#include "io/CliParser.hpp"
#include "io/ExponentInput.hpp"
#include "io/WorktodoParser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

// Parse a single worktodo line and return the Wagstaff exponent for it.
uint64_t wagstaffFor(const std::string& line, bool* parsed = nullptr) {
    const auto path = std::filesystem::temp_directory_path() / "prmers_wagstaff_worktodo_test.txt";
    {
        std::ofstream out(path);
        out << line << "\n";
    }
    io::WorktodoParser parser(path.string());
    const auto entry = parser.parse();
    std::filesystem::remove(path);
    if (parsed) *parsed = entry.has_value();
    return entry ? io::wagstaffExponentForEntry(*entry) : 0;
}

} // namespace

int main() {
    bool parsed = false;

    check(wagstaffFor("PRP=1,2,1000003,-1,76,0,3,1", &parsed) == 2000006ULL && parsed,
          "PRP entry doubles its exponent");
    check(wagstaffFor("PRP=1,2,86243,-1", &parsed) == 172486ULL && parsed,
          "short PRP entry doubles its exponent");
    // Large enough that 2p no longer fits in 32 bits.
    check(wagstaffFor("PRP=1,2,2500000001,-1,76,0", &parsed) == 5000000002ULL && parsed,
          "doubling does not wrap at 32 bits");
    check(wagstaffFor("PRP=1,2,11,-1,76,0,\"23\"", &parsed) == 0ULL && parsed,
          "cofactor PRP entry is not Wagstaff work");
    check(wagstaffFor("Test=1000003,76,0", &parsed) == 0ULL && parsed,
          "LL entry is not Wagstaff work");
    check(wagstaffFor("Pminus1=1,2,1000003,-1,100000,1000000,76", &parsed) == 0ULL && parsed,
          "P-1 entry is not Wagstaff work");

    // The doubled worktodo exponent is held to the same limit as the command line's -wagstaff exponent
    // (CliParser checks it after doubling): the largest accepted p and the first rejected p.
    {
        const uint64_t limit = io::kMaxEngineExponent < io::kMaxExponent ? io::kMaxEngineExponent : io::kMaxExponent;
        const uint64_t largestP = limit / 2;    // 2p <= limit
        const uint64_t firstBadP = largestP + 1;
        check(io::exponentLimitError(2 * largestP, true).empty(), "largest Wagstaff exponent is accepted");
        check(!io::exponentLimitError(2 * firstBadP, true).empty(), "first Wagstaff exponent past the limit is rejected");
        check(io::exponentLimitError(limit).empty() && !io::exponentLimitError(limit + 1).empty(),
              "limit is exact for a plain exponent");
        check(!io::exponentLimitError(io::kMaxExponent + 1).empty(), "global exponent limit is enforced");

        // Through the worktodo parser and the doubling: p = 2^31 - 1 doubles to 2^32 - 2 (accepted);
        // p = 2^31 doubles to 2^32, which the 32-bit engines would truncate to 0 (rejected);
        // p = 3000000000 doubles to 6000000000, which `-wagstaff 3000000000` on the command line
        // already rejects.
        auto limitErrorFor = [&](const std::string& line) {
            return io::exponentLimitError(wagstaffFor(line), true);
        };
        check(wagstaffFor("PRP=1,2,2147483647,-1") == 4294967294ULL && limitErrorFor("PRP=1,2,2147483647,-1").empty(),
              "p = 2^31 - 1 is accepted");
        check(wagstaffFor("PRP=1,2,2147483648,-1") == 4294967296ULL && !limitErrorFor("PRP=1,2,2147483648,-1").empty(),
              "p = 2^31 is rejected");
        check(!limitErrorFor("PRP=1,2,3000000000,-1").empty(), "p = 3000000000 is rejected");
        check(!limitErrorFor("PRP=1,2,2500000001,-1,76,0").empty(), "p = 2500000001 is rejected");
        check(!limitErrorFor("PRP=1,2,4294967295,-1").empty(), "p = 2^32 - 1 is rejected");
        check(largestP == 2147483647ULL, "largest accepted worktodo p is 2^31 - 1");
    }

    // Dropping -wagstaff for an entry that is not Wagstaff work gives the Gerbicz-Li check and proof
    // generation back (-wagstaff switched both off), unless the command line had turned them off.
    {
        io::CliOptions o;
        o.wagstaff = true;
        o.gerbiczli_before_wagstaff = true;
        o.proof_before_wagstaff = true;
        o.gerbiczli = false;   // as forced by -wagstaff
        o.proof = false;
        io::dropWagstaff(o);
        check(!o.wagstaff && o.gerbiczli && o.proof, "dropping -wagstaff restores Gerbicz-Li and proof");

        io::CliOptions e;
        e.wagstaff = true;
        e.gerbiczli_before_wagstaff = false;   // -gerbiczli was given explicitly
        e.proof_before_wagstaff = false;       // -proof 0 was given explicitly
        e.gerbiczli = false;
        e.proof = false;
        io::dropWagstaff(e);
        check(!e.wagstaff && !e.gerbiczli && !e.proof, "dropping -wagstaff keeps explicit -gerbiczli / -proof 0");
    }

    if (failures) return 1;
    std::cout << "wagstaff worktodo test passed\n";
    return 0;
}
