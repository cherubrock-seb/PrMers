#include "io/WorktodoParser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "prmers_gm_worktodo_test.txt";
    {
        std::ofstream out(path);
        out << "# preserved comment\n";
        out << "; preserved semicolon comment\n";
        out << "GMCHAIN=45951761,100000,1000000,2000,0,2,1000000000000,262144\n";
        out << "GMPROTH=45951781,0\n";
    }

    io::WorktodoParser parser(path.string());
    auto entry = parser.parse();
    if (!entry || !entry->gaussianMersenne || !entry->gmPipeline ||
        entry->exponent != 45951761U || entry->B1 != 100000ULL ||
        entry->B2 != 1000000ULL || entry->gmEcmB1 != 2000ULL ||
        entry->gmEcmB2 != 0ULL || entry->gmEcmCurves != 2ULL ||
        entry->gmSieveLimit != 1000000000000ULL ||
        entry->gmFactorChunkBits != 262144ULL) {
        std::cerr << "GMCHAIN parse mismatch\n";
        return 1;
    }

    if (!parser.removeFirstProcessed()) {
        std::cerr << "failed to remove completed entry\n";
        return 1;
    }

    std::ifstream remaining(path);
    const std::string text((std::istreambuf_iterator<char>(remaining)), {});
    if (text.find("# preserved comment") == std::string::npos ||
        text.find("; preserved semicolon comment") == std::string::npos ||
        text.find("GMCHAIN=") != std::string::npos ||
        text.find("GMPROTH=45951781,0") == std::string::npos) {
        std::cerr << "worktodo removal did not preserve comments/next entry\n";
        return 1;
    }

    auto next = parser.parse();
    if (!next || !next->gaussianMersenne || next->gmPipeline ||
        next->gmPrpOnly || next->exponent != 45951781U) {
        std::cerr << "GMPROTH parse mismatch\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::cout << "Gaussian worktodo parser test passed\n";
    return 0;
}
