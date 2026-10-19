// WorktodoParser::removeProcessedLine copies the worktodo file without the finished entry and renames the copy
// over it.  The copy loop (std::getline) also stops on a read error, and that was taken as the end of the file:
// an I/O error part way through left a copy holding only the lines read so far, and the rename cut every later
// entry from the worktodo file.  A failed read must now leave the file exactly as it was.
//
// read() is defined here: it forwards to the C library, except that while a case is armed, the reads of the
// worktodo file fail with EIO from a chosen call on.  usage: worktodo-read-error-test <scratch dir>
#include "io/WorktodoParser.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {
std::string g_target;   // canonical path of the file whose reads fail
int g_fail_from = 0;    // 1-based read call on that file from which reads fail; 0: none
int g_calls = 0;
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

void spit(const std::filesystem::path& p, const std::string& text) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << text;
}
}  // namespace

extern "C" ssize_t read(int fd, void* buf, size_t count) {
    static auto real = reinterpret_cast<ssize_t (*)(int, void*, size_t)>(dlsym(RTLD_NEXT, "read"));
    if (g_fail_from > 0 && !g_target.empty()) {
        char link[PATH_MAX];
        const std::string self = "/proc/self/fd/" + std::to_string(fd);
        const ssize_t n = readlink(self.c_str(), link, sizeof(link) - 1);
        if (n > 0 && std::string(link, static_cast<size_t>(n)) == g_target && ++g_calls >= g_fail_from) {
            errno = EIO;
            return -1;
        }
    }
    return real(fd, buf, count);
}

namespace {
// A worktodo file well over one stdio buffer, so that a read fails part way through.
std::string makeWorktodo(int lines) {
    std::string text;
    for (int i = 0; i < lines; ++i) {
        std::ostringstream l;
        l << "PRP=" << std::uppercase << std::hex << (0x10000000u + static_cast<unsigned>(i)) << "0123456789ABCDEF0123456789AB,1,2,"
          << std::dec << (100000007 + 2 * i) << ",-1,76,0\n";
        text += l.str();
    }
    return text;
}

std::string lineAt(const std::string& text, int index) {
    std::istringstream in(text);
    std::string l;
    for (int i = 0; i <= index && std::getline(in, l); ++i) {}
    return l;
}

// remove line <index> while reads fail from call <failFrom> on; expect the file unchanged when a read failed
void runCase(const std::filesystem::path& dir, const char* name, int index, int failFrom, bool expectRemoved) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto wt = dir / "worktodo.txt";
    const std::string before = makeWorktodo(1000);  // about 60 KB
    spit(wt, before);
    io::WorktodoParser parser(wt.string());

    g_target = std::filesystem::canonical(wt).string();
    g_calls = 0;
    g_fail_from = failFrom;
    const bool ok = parser.removeProcessedLine(lineAt(before, index));
    g_fail_from = 0;
    const std::string after = slurp(wt);
    const std::string tag = std::string(name) + ": ";

    if (expectRemoved) {
        expect(ok, tag + "removal failed");
        expect(after.size() + lineAt(before, index).size() + 1 == before.size(), tag + "wrong size after removal");
        expect(after.find(lineAt(before, index)) == std::string::npos, tag + "line still present");
    } else {
        expect(!ok, tag + "removal reported success after a read error");
        expect(after == before, tag + "worktodo changed (" + std::to_string(before.size()) + " -> " +
                                    std::to_string(after.size()) + " bytes)");
        expect(!std::filesystem::exists(dir / "worktodo_save.txt"), tag + "entry archived after a read error");
    }
    expect(failFrom == 0 || g_calls >= failFrom, tag + "the failing read was not reached");
    expect(!std::filesystem::exists(dir / "worktodo.txt.tmp"), tag + "temporary file left behind");
    std::cout << (failures ? "" : "ok   ") << name << "\n";
}
}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path base = argc > 1 ? argv[1] : "worktodo-read-error";
    runCase(base / "a", "no error: the line is removed", 3, 0, true);
    runCase(base / "b", "error after the first buffer, line in it", 3, 2, false);
    runCase(base / "c", "error after the first buffer, line after it", 900, 2, false);
    runCase(base / "d", "error on the first read", 0, 1, false);
    runCase(base / "e", "error near the end", 999, 6, false);

    // No error, last line without a newline: still a complete read, the line is removed.
    {
        const auto dir = base / "f";
        std::filesystem::create_directories(dir);
        const auto wt = dir / "worktodo.txt";
        spit(wt, "PRP=1,2,127,-1\nPRP=1,2,521,-1");
        io::WorktodoParser parser(wt.string());
        expect(parser.removeProcessedLine("PRP=1,2,521,-1"), "no final newline: removal failed");
        expect(slurp(wt) == "PRP=1,2,127,-1\n", "no final newline: wrong content " + slurp(wt));
        std::cout << "ok   no final newline\n";
    }
    // The worktodo path is a directory: nothing is created or changed.
    {
        const auto dir = base / "g";
        std::filesystem::create_directories(dir / "worktodo.txt");
        io::WorktodoParser parser((dir / "worktodo.txt").string());
        expect(!parser.removeProcessedLine("PRP=1,2,127,-1"), "directory: removal reported success");
        expect(!std::filesystem::exists(dir / "worktodo.txt.tmp"), "directory: temporary file left behind");
        expect(std::filesystem::is_directory(dir / "worktodo.txt"), "directory: replaced");
        std::cout << "ok   worktodo path is a directory\n";
    }
    std::filesystem::remove_all(base);
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Worktodo read error test: PASS\n";
    return 0;
}
