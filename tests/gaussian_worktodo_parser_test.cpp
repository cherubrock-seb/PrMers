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
        out << "GMCHAIN=45951761,100000,1000000,2000,0,2,1000000000000,262144,proth,BOTH\n";
        out << "GMCHAIN=45951771,100000,1000000,2000,250000,2,0,262144,factor,GQ\n";
        out << "GMPROTH=45951781,0,GM\n";
    }

    io::WorktodoParser parser(path.string());
    auto entry = parser.parse();
    if (!entry || !entry->gaussianMersenne || !entry->gmPipeline ||
        entry->exponent != 45951761U || entry->B1 != 100000ULL ||
        entry->B2 != 1000000ULL || entry->gmEcmB1 != 2000ULL ||
        entry->gmEcmB2 != 0ULL || entry->gmEcmCurves != 2ULL ||
        entry->gmSieveLimit != 1000000000000ULL ||
        entry->gmFactorChunkBits != 262144ULL || !entry->gmPipelineProth || entry->gmFamily != "BOTH") {
        std::cerr << "GMCHAIN parse mismatch\n";
        return 1;
    }

    if (!parser.removeProcessedLine(entry->rawLine)) {
        std::cerr << "failed to remove completed entry\n";
        return 1;
    }

    std::ifstream remaining(path);
    const std::string text((std::istreambuf_iterator<char>(remaining)), {});
    if (text.find("# preserved comment") == std::string::npos ||
        text.find("; preserved semicolon comment") == std::string::npos ||
        text.find("GMCHAIN=45951761") != std::string::npos ||
        text.find("GMCHAIN=45951771") == std::string::npos ||
        text.find("GMPROTH=45951781,0") == std::string::npos) {
        std::cerr << "worktodo removal did not preserve comments/next entry\n";
        return 1;
    }

    auto factor_only = parser.parse();
    if (!factor_only || !factor_only->gaussianMersenne ||
        !factor_only->gmPipeline || factor_only->gmPipelineProth ||
        factor_only->exponent != 45951771U || factor_only->gmFamily != "GQ") {
        std::cerr << "factor-only GMCHAIN parse mismatch\n";
        return 1;
    }

    if (!parser.removeProcessedLine(factor_only->rawLine)) {
        std::cerr << "failed to remove factor-only entry\n";
        return 1;
    }

    auto next = parser.parse();
    if (!next || !next->gaussianMersenne || next->gmPipeline ||
        next->gmPrpOnly || next->exponent != 45951781U || next->gmFamily != "GM") {
        std::cerr << "GMPROTH parse mismatch\n";
        return 1;
    }

    // parse() skips lines it cannot run; completing the entry it returned must remove that entry, not
    // the first actionable line of the file.
    const auto mixed = std::filesystem::temp_directory_path() / "prmers_mixed_worktodo_test.txt";
    {
        std::ofstream out(mixed);
        out << "Factor=N/A,1279,60,61\n";
        out << "PRP=1,2,127,-1\n";
        out << "PRP=1,2,521,-1\n";
    }
    io::WorktodoParser mixedParser(mixed.string());
    auto prp = mixedParser.parse();
    if (!prp || !prp->prpTest || prp->exponent != 127U) {
        std::cerr << "mixed worktodo parse mismatch\n";
        return 1;
    }
    if (!mixedParser.removeProcessedLine(prp->rawLine)) {
        std::cerr << "failed to remove the PRP entry that ran\n";
        return 1;
    }
    {
        std::ifstream in(mixed);
        const std::string text((std::istreambuf_iterator<char>(in)), {});
        if (text != "Factor=N/A,1279,60,61\nPRP=1,2,521,-1\n") {
            std::cerr << "mixed worktodo removal removed the wrong line:\n" << text;
            return 1;
        }
    }
    auto prp2 = mixedParser.parse();
    if (!prp2 || prp2->exponent != 521U) {
        std::cerr << "mixed worktodo did not advance to the next entry\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(mixed, ec);
    std::cout << "Gaussian worktodo parser test passed\n";
    return 0;
}
