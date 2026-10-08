// Host checks for small worktodo/log fixes:
//  - WorktodoParser::hasPendingEntry ignores comments, blanks and unsupported lines, and lines
//    parse() would skip (malformed fields, no-bounds Pfactor, bad known factors, glued lines)
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
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

static void expect(bool ok, const std::string& what) { expect(ok, what.c_str()); }

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

        // A GMTF line is run by the Gaussian trial-factoring pre-parser, and only from the top.
        write(wt, "GMTF=1009,20,24\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "GMTF line on top is pending");
        write(wt, "# c\n  gmtf=1009,20,24,BOTH,65536,1000\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "indented lower-case GMTF is pending");
        write(wt, "GMTF=1009,20\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "GMTF with too few fields is not pending");
        write(wt, "GMTF=1009,20,24,BOTH,65536,1000,9\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "GMTF with too many fields is not pending");
        write(wt, "GMTF=\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "empty GMTF is not pending");
        write(wt, "GMTF\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "GMTF without '=' is not pending");
        write(wt, "Pfactor=N/A,1,2,127,-1,70,0\nGMTF=1009,20,24\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()),
               "a GMTF line behind an unrunnable line is not on top, so not pending");

        // A line with a supported keyword that parse() skips is not pending work: the restart would
        // find no entry and land in the interactive prompt (or, under -gui, run exponent 0).
        write(wt, "PRP=1,2,127,-1PRP=1,2,521,-1\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "glued lines are not pending");
        write(wt, "Pfactor=N/A,1,2,127,-1,70,0\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "Pfactor with no bounded P-1 work is not pending");
        write(wt, "PRP=1,2,29,-1,76,2,3,5,\"1105\"\n");
        expect(!io::WorktodoParser::hasPendingEntry(wt.string()), "cofactor PRP with a bad factor is not pending");
        write(wt, "PRP=1,2,127,-1PRP=1,2,521,-1\n; comment\nPRP=1,2,521,-1\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "a runnable line after a skipped one is pending");
        write(wt, "Pfactor=N/A,1,2,20000003,-1,70,2\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "Pfactor with bounds is pending");
        write(wt, "PRP=1,2,29,-1,76,2,3,5,\"233,1103,2089\"\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "cofactor PRP is pending");

        // Malformed lines of every supported keyword: hasPendingEntry must agree with parse(). A line
        // parse() cannot turn into an entry (garbage, a bad exponent, missing fields) is not pending,
        // for the same reason as above.
        const std::vector<std::string> malformed = {
            "PRP=garbage",
            "PRP=1,2,abc,-1",
            "PRP=1,2,,-1",
            "PRP=1,2,127",
            "PRPDC=garbage",
            "PRPDC=1,2,abc,-1",
            "Test=garbage",
            "Test=1,2,abc,-1",
            "DoubleCheck=garbage",
            "DoubleCheck=abc",
            "DoubleCheck=N/A,abc,70,1",
            "DoubleCheck=",
            "Pminus1=garbage",
            "Pminus1=1,2,abc,-1,100000,1000000",
            "Pminus1=1,2,127,-1,abc,1000000",
            "Pfactor=garbage",
            "Pfactor=N/A,1,2,abc,-1,77,1",
            "Pfactor=N/A,1,2,127,-1,abc,1",
            "ECM2=garbage",
            "ECM2=1,2,abc,-1,1000,10000,1",
            "ECM2=1,2,127,-1,abc,10000,1",
            "GMPRP=garbage",
            "GMPROTH=garbage",
            "GMTEST=garbage",
            "GMPMINUS1=garbage",
            "GMPM1=garbage",
            "GMECM=garbage",
            "GMCHAIN=garbage",
            "GMCAMPAIGN=garbage",
            "PRP=",
        };
        for (const std::string& line : malformed) {
            write(wt, line + "\n");
            bool pending = true;
            bool parsed = true;
            try {
                pending = io::WorktodoParser::hasPendingEntry(wt.string());
                parsed = io::WorktodoParser(wt.string()).parse().has_value();
            } catch (const std::exception& e) {
                expect(false, "malformed line must not throw: " + line + " (" + e.what() + ")");
                continue;
            }
            expect(!parsed, "parse() yields no entry for: " + line);
            expect(!pending, "malformed supported-keyword line is not pending: " + line);
        }

        // A malformed line before a runnable one does not hide it, and the answer follows parse().
        write(wt, "PRP=garbage\nDoubleCheck=bad\nECM2=1,2,abc,-1,1000,10000,1\nPRP=1,2,521,-1\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "runnable line after malformed ones is pending");
        write(wt, "PRP=1,2,521,-1\nPRP=garbage\n");
        expect(io::WorktodoParser::hasPendingEntry(wt.string()), "runnable line before a malformed one is pending");
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
