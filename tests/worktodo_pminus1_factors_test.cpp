// Pminus1 known-factor parsing.
//
// Prime95 writes the known factors as one quoted comma-separated list
// ("f1,f2"). The web GUI writes each factor as its own quoted field
// ("f1","f2"). Both forms must yield every factor; the separately quoted form
// used to keep only the first one, so P-1 ran against the wrong cofactor.

#include "io/WorktodoParser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::optional<io::WorktodoEntry> parseLine(const std::string& line) {
    const auto path = std::filesystem::temp_directory_path() / "prmers_pm1_factors_test.txt";
    {
        std::ofstream out(path);
        out << line << "\n";
    }
    io::WorktodoParser parser(path.string());
    auto entry = parser.parse();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return entry;
}

std::string join(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    return s + "]";
}

void expectFactors(const std::string& line, const std::vector<std::string>& factors) {
    const auto entry = parseLine(line);
    if (!entry || !entry->pm1Test || entry->exponent != 127U || entry->B1 != 1000ULL ||
        entry->B2 != 50000ULL || entry->knownFactors != factors) {
        std::cerr << "FAIL " << line << ": ";
        if (!entry) std::cerr << "no entry\n";
        else std::cerr << "factors=" << join(entry->knownFactors)
                       << " expected " << join(factors) << "\n";
        ++failures;
    } else {
        std::cout << "PASS " << line << " -> " << join(entry->knownFactors) << "\n";
    }
}

} // namespace

int main() {
    const std::string aid = "0123456789ABCDEF0123456789ABCDEF";

    // One quoted list (Prime95 form).
    expectFactors("Pminus1=1,2,127,-1,1000,50000,\"2554,5108\"", {"2554", "5108"});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,\"2554\"", {"2554"});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,\"2554, 5108\"", {"2554", "5108"});

    // Separately quoted factors (web GUI form).
    expectFactors("Pminus1=1,2,127,-1,1000,50000,\"2554\",\"5108\"", {"2554", "5108"});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,\"2554\",\"5108\",\"7\"",
                  {"2554", "5108", "7"});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,\"2554,5108\",\"7\"", {"2554", "5108", "7"});
    expectFactors("Pminus1=" + aid + ",1,2,127,-1,1000,50000,\"2554\",\"5108\"",
                  {"2554", "5108"});

    // With the optional how_far_factored and B2_start fields in front.
    expectFactors("Pminus1=1,2,127,-1,1000,50000,70,\"2554\",\"5108\"", {"2554", "5108"});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,70,500,\"2554\",\"5108\"", {"2554", "5108"});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,70,500,\"2554,5108\"", {"2554", "5108"});

    // No factors.
    expectFactors("Pminus1=1,2,127,-1,1000,50000", {});
    expectFactors("Pminus1=1,2,127,-1,1000,50000,70,500", {});

    // The optional fields in front still parse next to the factors.
    {
        const auto entry =
            parseLine("Pminus1=1,2,127,-1,1000,50000,70,500,\"2554\",\"5108\"");
        if (!entry || entry->sieveDepth != 70.0 || entry->B2Start != 500ULL) {
            std::cerr << "FAIL how_far_factored/B2_start not kept next to factors\n";
            ++failures;
        } else {
            std::cout << "PASS sieveDepth=70 B2Start=500 kept next to factors\n";
        }
    }

    if (failures != 0) {
        std::cerr << failures << " worktodo Pminus1 factor check(s) failed\n";
        return 1;
    }
    std::cout << "Worktodo Pminus1 factors test passed\n";
    return 0;
}
