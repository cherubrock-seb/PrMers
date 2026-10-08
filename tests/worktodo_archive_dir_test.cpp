// Host checks: the archive of finished worktodo entries (worktodo_save.txt) lives in the same
// directory as the worktodo file in use, not in the current directory.
#include "io/WorktodoParser.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

static void write(const fs::path& p, const std::string& text) {
    std::ofstream out(p, std::ios::binary);
    out << text;
}

static std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}

// Run the first entry of `wt` through parse()/removeProcessedLine() the way the modes do.
static bool finishFirst(const std::string& wt) {
    io::WorktodoParser p(wt);
    auto e = p.parse();
    return e && p.removeProcessedLine(e->rawLine);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "prmers_worktodo_archive_dir";
    fs::remove_all(root);
    fs::create_directories(root / "cwd");
    fs::create_directories(root / "wt" / "nested");
    fs::current_path(root / "cwd");

    // archivePathFor: pure path logic
    expect(io::WorktodoParser::archivePathFor("worktodo.txt") == "worktodo_save.txt", "no directory part -> cwd");
    expect(io::WorktodoParser::archivePathFor("./worktodo.txt") == "./worktodo_save.txt", "dot-relative");
    expect(io::WorktodoParser::archivePathFor("sub/w.txt") == (fs::path("sub") / "worktodo_save.txt").string(),
           "relative with directory");
    expect(io::WorktodoParser::archivePathFor("/abs/dir/w.txt") == "/abs/dir/worktodo_save.txt", "absolute path");
    expect(io::WorktodoParser::archivePathFor("../w.txt") == (fs::path("..") / "worktodo_save.txt").string(),
           "parent-relative");
    expect(io::WorktodoParser("/abs/dir/w.txt").archivePath() == "/abs/dir/worktodo_save.txt", "member form");

    // Absolute path in another directory: archive goes beside it, not into the cwd.
    {
        const fs::path wt = root / "wt" / "worktodo.txt";
        write(wt, "PRP=1,2,61,-1\nPRP=1,2,89,-1\n");
        expect(finishFirst(wt.string()), "absolute: removeProcessedLine succeeds");
        expect(slurp(root / "wt" / "worktodo_save.txt") == "PRP=1,2,61,-1\n", "absolute: archive beside worktodo");
        expect(!fs::exists(root / "cwd" / "worktodo_save.txt"), "absolute: nothing archived in the cwd");
        expect(slurp(wt) == "PRP=1,2,89,-1\n", "absolute: worktodo keeps the other line");
        // A second finished entry appends.
        expect(finishFirst(wt.string()), "absolute: second removal succeeds");
        expect(slurp(root / "wt" / "worktodo_save.txt") == "PRP=1,2,61,-1\nPRP=1,2,89,-1\n", "absolute: archive appends");
    }

    // Relative path with a directory component, resolved against the cwd.
    {
        fs::create_directories("rel/deeper");
        write("rel/deeper/w.txt", "PRP=1,2,127,-1\n");
        expect(finishFirst("rel/deeper/w.txt"), "relative: removeProcessedLine succeeds");
        expect(slurp("rel/deeper/worktodo_save.txt") == "PRP=1,2,127,-1\n", "relative: archive beside worktodo");
        expect(!fs::exists("worktodo_save.txt"), "relative: nothing archived in the cwd");
        expect(slurp("rel/deeper/w.txt").empty(), "relative: worktodo drained");
        expect(!fs::exists("rel/deeper/w.txt.tmp"), "relative: no .tmp left");
    }

    // Path with no directory component: the cwd, as before.
    {
        write("worktodo.txt", "PRP=1,2,521,-1\n");
        expect(finishFirst("worktodo.txt"), "bare name: removeProcessedLine succeeds");
        expect(slurp("worktodo_save.txt") == "PRP=1,2,521,-1\n", "bare name: archive in the cwd");
        fs::remove("worktodo_save.txt");
    }

    // Dot-relative and ".." paths.
    {
        write("./dot.txt", "PRP=1,2,607,-1\n");
        expect(finishFirst("./dot.txt"), "dot: removeProcessedLine succeeds");
        expect(slurp("worktodo_save.txt") == "PRP=1,2,607,-1\n", "dot: archive in the cwd");
        fs::remove("worktodo_save.txt");
        fs::current_path(root / "wt" / "nested");
        write("../up.txt", "PRP=1,2,1279,-1\n");
        expect(finishFirst("../up.txt"), "dotdot: removeProcessedLine succeeds");
        expect(slurp(root / "wt" / "worktodo_save.txt").find("PRP=1,2,1279,-1\n") != std::string::npos,
               "dotdot: archive in the parent directory");
        expect(!fs::exists("worktodo_save.txt"), "dotdot: nothing archived in the cwd");
        fs::current_path(root / "cwd");
    }

    // A hand-edited archive with no trailing newline: the new line must not be glued onto it.
    {
        fs::create_directories("glue");
        write("glue/w.txt", "PRP=1,2,2203,-1\n");
        write("glue/worktodo_save.txt", "PRP=1,2,3,-1");
        expect(finishFirst("glue/w.txt"), "glue: removeProcessedLine succeeds");
        expect(slurp("glue/worktodo_save.txt") == "PRP=1,2,3,-1\nPRP=1,2,2203,-1\n", "glue: newline inserted");
    }

    // Failures keep the entry: nothing is dropped without being archived, no .tmp is left behind.
    {
        fs::create_directories("fail");
        // archive path is a directory
        write("fail/w.txt", "PRP=1,2,2281,-1\n");
        fs::create_directories("fail/worktodo_save.txt");
        expect(!finishFirst("fail/w.txt"), "archive is a directory: removal fails");
        expect(slurp("fail/w.txt") == "PRP=1,2,2281,-1\n", "archive is a directory: worktodo untouched");
        expect(!fs::exists("fail/w.txt.tmp"), "archive is a directory: no .tmp left");
        fs::remove("fail/worktodo_save.txt");

        // missing worktodo
        io::WorktodoParser missing("fail/absent.txt");
        expect(!missing.removeProcessedLine("PRP=1,2,61,-1"), "missing worktodo: removal fails");
        expect(!fs::exists("fail/worktodo_save.txt"), "missing worktodo: no archive created");
        expect(!fs::exists("fail/absent.txt.tmp"), "missing worktodo: no .tmp created");

        // line not present: no archive created, worktodo untouched
        io::WorktodoParser p("fail/w.txt");
        expect(!p.removeProcessedLine("PRP=1,2,61,-1"), "unknown line: removal fails");
        expect(!fs::exists("fail/worktodo_save.txt"), "unknown line: no archive created");
        expect(slurp("fail/w.txt") == "PRP=1,2,2281,-1\n", "unknown line: worktodo untouched");

        if (geteuid() != 0) {
            // read-only archive file
            write("fail/worktodo_save.txt", "old\n");
            chmod("fail/worktodo_save.txt", 0444);
            expect(!finishFirst("fail/w.txt"), "read-only archive: removal fails");
            expect(slurp("fail/w.txt") == "PRP=1,2,2281,-1\n", "read-only archive: entry kept in worktodo");
            expect(slurp("fail/worktodo_save.txt") == "old\n", "read-only archive: archive untouched");
            expect(!fs::exists("fail/w.txt.tmp"), "read-only archive: no .tmp left");
            chmod("fail/worktodo_save.txt", 0644);
            fs::remove("fail/worktodo_save.txt");

            // read-only directory (cannot create .tmp or the archive): clear failure, nothing lost
            chmod("fail", 0555);
            expect(!finishFirst("fail/w.txt"), "read-only directory: removal fails");
            expect(slurp("fail/w.txt") == "PRP=1,2,2281,-1\n", "read-only directory: entry kept");
            expect(!fs::exists("fail/worktodo_save.txt"), "read-only directory: no archive created anywhere");
            expect(!fs::exists("worktodo_save.txt"), "read-only directory: no silent fallback to the cwd");
            chmod("fail", 0755);
        } else {
            std::cout << "running as root: skipping read-only checks\n";
        }
    }

    fs::current_path(fs::temp_directory_path());
    fs::remove_all(root);
    if (failures == 0) std::cout << "worktodo archive dir test passed\n";
    return failures == 0 ? 0 : 1;
}
