// Worktodo exponent field validation for the k,b,n,c line forms.
//
// The exponent is stored in 32 bits. A value that does not fit, or one with
// trailing junk, used to be silently truncated or cut short (for example
// 4294967311 became 15 and 127abc became 127), so the wrong number was tested.
// Such lines must be skipped instead.

#include "io/WorktodoParser.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace {

int failures = 0;

std::optional<io::WorktodoEntry> parseFirst(const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / "prmers_exponent_range_test.txt";
    {
        std::ofstream out(path);
        out << text << "\n";
    }
    io::WorktodoParser parser(path.string());
    auto entry = parser.parse();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return entry;
}

// The first line must be skipped, so the sentinel line after it is returned.
void expectSkipped(const std::string& line) {
    const auto entry = parseFirst(line + "\nPRP=1,2,131,-1");
    if (!entry || entry->exponent != 131U) {
        std::cerr << "FAIL expected skip: " << line << " -> "
                  << (entry ? "exponent " + std::to_string(entry->exponent) : "no entry") << "\n";
        ++failures;
    } else {
        std::cout << "PASS skipped: " << line << "\n";
    }
}

void expectExponent(const std::string& line, uint32_t exponent) {
    const auto entry = parseFirst(line);
    if (!entry || entry->exponent != exponent) {
        std::cerr << "FAIL expected exponent " << exponent << ": " << line << " -> "
                  << (entry ? "exponent " + std::to_string(entry->exponent) : "no entry") << "\n";
        ++failures;
    } else {
        std::cout << "PASS exponent " << exponent << ": " << line << "\n";
    }
}

} // namespace

int main() {
    // 4294967311 = 2^32 + 15 and 4425967297 = 2^32 + 131000001.
    expectSkipped("PRP=1,2,4294967311,-1");
    expectSkipped("PRP=N/A,1,2,4294967311,-1,70,0");
    expectSkipped("Test=1,2,4294967311,-1");
    expectSkipped("Pminus1=1,2,4294967311,-1,100000,1000000");
    expectSkipped("Pfactor=N/A,1,2,4425967297,-1,77,1");
    expectSkipped("ECM2=1,2,4294967311,-1,1000,10000,1");
    expectSkipped("PRP=1,2,18446744073709551631,-1");  // 2^64 + 15

    // Trailing junk, signs and empty fields are not exponents.
    expectSkipped("PRP=1,2,127abc,-1");
    expectSkipped("PRP=1,2,12 7,-1");
    expectSkipped("PRP=1,2,+127,-1");
    expectSkipped("PRP=1,2,-127,-1");
    expectSkipped("PRP=1,2,0,-1");
    expectSkipped("PRP=1,2,,-1");
    expectSkipped("Pminus1=1,2,127abc,-1,100000,1000000");
    expectSkipped("Pfactor=N/A,1,2,127abc,-1,77,1");
    expectSkipped("ECM2=1,2,127abc,-1,1000,10000,1");

    // The largest 32-bit exponent and surrounding whitespace are accepted.
    expectExponent("PRP=1,2,4294967295,-1", 4294967295U);
    expectExponent("PRP=1,2, 127 ,-1", 127U);
    expectExponent("PRP=1,2,00127,-1", 127U);

    // Leading zeros do not count toward the value, however many there are.
    expectExponent("PRP=1,2,00000000000000000127,-1", 127U);
    expectExponent("PRP=1,2,000000004294967295,-1", 4294967295U);
    expectExponent("Pminus1=1,2,00000000000000000127,-1,100000,1000000", 127U);
    expectExponent("Pfactor=N/A,1,2,00000000004294967295,-1,77,1", 4294967295U);
    expectExponent("ECM2=1,2,00000000000000000127,-1,1000,10000,1", 127U);
    expectSkipped("PRP=1,2,00000004294967296,-1");  // 2^32 with leading zeros
    expectSkipped("PRP=1,2,000000000000000000000,-1");
    expectSkipped("PRP=1,2,99999999999999999999999,-1");
    expectExponent("Pminus1=1,2,4294967295,-1,100000,1000000", 4294967295U);
    expectExponent("Pfactor=N/A,1,2,4294967295,-1,77,1", 4294967295U);
    expectExponent("ECM2=1,2,4294967295,-1,1000,10000,1", 4294967295U);

    if (failures != 0) {
        std::cerr << failures << " worktodo exponent range check(s) failed\n";
        return 1;
    }
    std::cout << "Worktodo exponent range test passed\n";
    return 0;
}
