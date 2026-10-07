// Host checks for small worktodo/log fixes:
//  - WorktodoParser::hasPendingEntry ignores comments, blanks and unsupported lines
//  - the PRP line the GUI generates (tf/tests_saved padding) keeps the residue type
//  - removeProcessedLine replaces the file and leaves no .tmp behind
//  - Logger::logStart prints a 64-bit exponent correctly
#include "core/Logger.hpp"
#include "io/WorktodoParser.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

static void write(const fs::path& p, const std::string& text) {
    std::ofstream out(p);
    out << text;
}

static std::string slurp(const fs::path& p) {
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "prmers_worktodo_small_items";
    fs::remove_all(dir);
    fs::create_directories(dir);
    fs::current_path(dir);  // worktodo_save.txt and the log go to the cwd

    // hasPendingEntry
    {
        const fs::path wt = dir / "pending.txt";
        expect(!io::WorktodoParser::hasPendingEntry((dir / "missing.txt").string()), "missing file");
        write(wt, "# comment\n; semicolon comment\n   \n\t\nFactor=N/A,1,2,61,-1,60,61\nnot a line\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()),
               "only comments, blanks and unsupported lines -> nothing pending");
        write(wt, "; comment\n  prp=1,2,61,-1\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "indented lower-case PRP is pending");
        write(wt, "GMCHAIN=45951761,100000,1000000,2000,0,2,1000000,262144,proth,BOTH\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "GM entry is pending");
    }

    // PRP residue type: Prime95 order is k,b,n,c,tf,tests_saved,base,residue_type.
    {
        const fs::path wt = dir / "prp.txt";
        write(wt, "PRP=1,2,61,-1,3,4\n");  // tf=3, tests_saved=4: no residue type
        auto e = io::WorktodoParser(wt.string()).parse();
        expect(e && e->residueType == 1, "two trailing ints are tf/tests_saved");
        write(wt, "PRP=1,2,61,-1,0,0,3,4\n");
        e = io::WorktodoParser(wt.string()).parse();
        expect(e && e->residueType == 4, "padded base/residue_type keeps residue type");
    }

    // removeProcessedLine
    {
        const fs::path wt = dir / "queue.txt";
        write(wt, "# keep\nPRP=1,2,61,-1\nPRP=1,2,89,-1\n");
        io::WorktodoParser p(wt.string());
        auto e = p.parse();
        expect(e && e->exponent == 61, "first entry is 61");
        expect(e && p.removeProcessedLine(e->rawLine), "removeProcessedLine succeeds");
        expect(slurp(wt) == "# keep\nPRP=1,2,89,-1\n", "worktodo keeps the other lines");
        expect(slurp(dir / "worktodo_save.txt") == "PRP=1,2,61,-1\n", "removed line archived");
        expect(!fs::exists(wt.string() + ".tmp"), "no .tmp left behind");
        expect(!p.removeProcessedLine("PRP=1,2,1000003,-1"), "unknown line is not removed");
        expect(!fs::exists(wt.string() + ".tmp"), "no .tmp left after a miss");
    }

    // Logger: exponent is uint64_t
    {
        io::CliOptions o;
        o.exponent = 4294967297ULL;  // 2^32 + 1, wrong with %u
        o.mode = "prp";
        core::Logger log((dir / "log.txt").string());
        log.logStart(o);
        log.flush_log();
        expect(slurp(dir / "log.txt").find("exponent=4294967297,") != std::string::npos,
               "logStart prints a 64-bit exponent");
    }

    fs::current_path(fs::temp_directory_path());
    fs::remove_all(dir);
    if (failures == 0) std::cout << "worktodo small items test passed\n";
    return failures == 0 ? 0 : 1;
}
