// Known-factor parsing for PRP, ECM2 and Pfactor worktodo lines.
//
// Prime95 writes the known factors as one quoted comma-separated list
// ("f1,f2"); the web GUI and hand-written lines may quote each factor on its
// own ("f1","f2"). Both forms must keep every factor. The separately quoted
// form used to keep only the last one, so a cofactor PRP ran on
// M_p / f_last instead of M_p / (f1 * f2 * ...).
//
// M29 = 233 * 1103 * 2089.

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
    const auto path = std::filesystem::temp_directory_path() / "prmers_quoted_factors_test.txt";
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
    if (!entry || entry->knownFactors != factors) {
        std::cerr << "FAIL " << line << ": ";
        if (!entry) std::cerr << "no entry\n";
        else std::cerr << "factors=" << join(entry->knownFactors)
                       << " expected " << join(factors) << "\n";
        ++failures;
    } else {
        std::cout << "PASS " << line << " -> " << join(entry->knownFactors) << "\n";
    }
}

void expectRejected(const std::string& line) {
    const auto entry = parseLine(line);
    if (entry) {
        std::cerr << "FAIL " << line << ": accepted with factors=" << join(entry->knownFactors)
                  << ", expected the line to be rejected\n";
        ++failures;
    } else {
        std::cout << "PASS " << line << " -> rejected\n";
    }
}

} // namespace

int main() {
    const std::vector<std::string> all = {"233", "1103", "2089"};

    // PRP: one quoted list, separately quoted factors, and a mix.
    expectFactors("PRP=1,2,29,-1,76,2,3,5,\"233,1103,2089\"", all);
    expectFactors("PRP=1,2,29,-1,76,2,3,5,\"233\",\"1103\",\"2089\"", all);
    expectFactors("PRP=1,2,29,-1,76,2,3,5,\"233,1103\",\"2089\"", all);
    expectFactors("PRP=1,2,29,-1,76,2,3,5,\"2089\"", {"2089"});
    expectFactors("PRP=1,2,29,-1,76,2,3,5,\"233\", \"1103\"", {"233", "1103"});
    expectFactors("PRP=1,2,29,-1,76,2", {});
    {
        const auto entry = parseLine("PRP=1,2,29,-1,76,2,3,5,\"233\",\"1103\",\"2089\"");
        if (!entry || !entry->prpTest || entry->residueType != 5U || entry->exponent != 29U) {
            std::cerr << "FAIL cofactor PRP entry fields\n";
            ++failures;
        }
    }
    // A factor that does not divide M_p is still rejected, wherever it is.
    expectRejected("PRP=1,2,29,-1,76,2,3,5,\"233\",\"1105\"");
    // A quoted factor followed by an unquoted one is malformed.
    expectRejected("PRP=1,2,29,-1,76,2,3,5,\"233\",1103");

    // ECM2.
    expectFactors("ECM2=1,2,29,-1,1000,50000,10,\"233,1103\"", {"233", "1103"});
    expectFactors("ECM2=1,2,29,-1,1000,50000,10,\"233\",\"1103\"", {"233", "1103"});
    expectFactors("ECM2=1,2,29,-1,1000,50000,10,\"233\",\"1103\",\"2089\"", all);
    expectFactors("ECM2=1,2,29,-1,1000,50000,10,233", {"233"});
    expectFactors("ECM2=1,2,29,-1,1000,50000,10", {});
    expectRejected("ECM2=1,2,29,-1,1000,50000,10,\"233\",1103");
    expectRejected("ECM2=1,2,29,-1,1000,50000,10,\"233\",\"1105\"");

    // Pfactor (tests_saved large enough that B1 is bounded and non-zero).
    expectFactors("Pfactor=1,2,20000003,-1,70,2,\"233,1103\"", {"233", "1103"});
    expectFactors("Pfactor=1,2,20000003,-1,70,2,\"233\",\"1103\"", {"233", "1103"});
    expectFactors("Pfactor=1,2,20000003,-1,70,2,\"233\",\"1103\",\"2089\"", all);
    expectFactors("Pfactor=1,2,20000003,-1,70,2,233", {"233"});
    expectFactors("Pfactor=1,2,20000003,-1,70,2", {});
    expectRejected("Pfactor=1,2,20000003,-1,70,2,\"233\",1103");

    if (failures != 0) {
        std::cerr << failures << " worktodo quoted-factor check(s) failed\n";
        return 1;
    }
    std::cout << "Worktodo quoted factors test passed\n";
    return 0;
}
