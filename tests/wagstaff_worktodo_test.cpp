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

    if (failures) return 1;
    std::cout << "wagstaff worktodo test passed\n";
    return 0;
}
