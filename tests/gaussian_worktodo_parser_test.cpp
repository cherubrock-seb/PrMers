#include "io/WorktodoParser.hpp"
#include "math/Pm1Bounds.hpp"

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

    // Pfactor=[AID,]k,b,n,c,how_far_factored,tests_saved: fields 4 and 5 are the TF depth and the tests saved,
    // not B1/B2. Malformed lines are skipped; tests_saved = 0 still gives bounds.
    const auto pf = std::filesystem::temp_directory_path() / "prmers_pfactor_worktodo_test.txt";
    {
        std::ofstream out(pf);
        out << "Pfactor=0123456789ABCDEF0123456789ABCDEF,1,2,1277,-1,76,2\n";
        out << "Pfactor=N/A,1,2,130000001,-1,77,1\n";
        out << "Pfactor=N/A,1,2,130000001,-1,77,2\n";
        out << "Pfactor=N/A,1,2,130000001,-1,77,0\n";
        out << "Pfactor=N/A,1,2,130000001,-1,abc,1\n";
        out << "Pfactor=N/A,1,2,130000001,-1,77,nan\n";
    }
    struct PfCase { uint32_t exponent; double tf; uint64_t B1, B2; std::string aid; };
    auto chosen = [](uint32_t p, double tf, double saved) { return math::choosePm1Bounds(p, tf, saved); };
    const PfCase pfCases[] = {
        {1277u, 76, chosen(1277u, 76, 2).B1, chosen(1277u, 76, 2).B2, "0123456789ABCDEF0123456789ABCDEF"},
        {130000001u, 77, chosen(130000001u, 77, 1).B1, chosen(130000001u, 77, 1).B2, ""},
        {130000001u, 77, chosen(130000001u, 77, 2).B1, chosen(130000001u, 77, 2).B2, ""},
        {130000001u, 77, chosen(130000001u, 77, 0).B1, chosen(130000001u, 77, 0).B2, ""},
    };
    io::WorktodoParser pfParser(pf.string());
    for (const auto& c : pfCases) {
        auto e = pfParser.parse();
        if (!e || !e->pm1Test || e->exponent != c.exponent || e->sieveDepth != c.tf ||
            e->B1 != c.B1 || e->B2 != c.B2 || e->aid != c.aid) {
            std::cerr << "Pfactor parse mismatch for exponent " << c.exponent << ": got B1="
                      << (e ? e->B1 : 0) << " B2=" << (e ? e->B2 : 0) << "\n";
            return 1;
        }
        if (!pfParser.removeProcessedLine(e->rawLine)) {
            std::cerr << "failed to remove Pfactor entry\n";
            return 1;
        }
    }
    if (pfParser.parse()) {
        std::cerr << "malformed Pfactor lines were not skipped\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(mixed, ec);
    std::filesystem::remove(pf, ec);
    std::cout << "Gaussian worktodo parser test passed\n";
    return 0;
}
