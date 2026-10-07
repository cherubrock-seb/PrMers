// WorktodoParser::appendLine (GUI "Append & Run").
//
// A worktodo file written by hand, with `echo -n`, or by many editors has no
// trailing newline. Appending "line\n" to it used to glue the new entry onto the
// last one ("PRP=1,2,127,-1PRP=1,2,521,-1"); parse() rejects that line, so both
// entries were lost and the queue could not continue.

#include "io/WorktodoParser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;
const std::string kPath = "prmers_append_test_worktodo.txt";

void write(const std::string& content) {
    std::ofstream out(kPath, std::ios::binary | std::ios::trunc);
    out << content;
}

std::string slurp() {
    std::ifstream in(kPath, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void expectFile(const std::string& what, const std::string& want) {
    const std::string got = slurp();
    if (got != want) {
        std::cerr << "FAIL " << what << ": file is [" << got << "], expected [" << want << "]\n";
        ++failures;
    } else {
        std::cout << "PASS " << what << "\n";
    }
}

} // namespace

int main() {
    std::filesystem::remove(kPath);

    // Missing file: created with the new line.
    io::WorktodoParser::appendLine(kPath, "PRP=1,2,521,-1");
    expectFile("append creates the file", "PRP=1,2,521,-1\n");

    // File ending in a newline: no blank line in between.
    write("PRP=1,2,127,-1\n");
    io::WorktodoParser::appendLine(kPath, "PRP=1,2,521,-1");
    expectFile("append after a terminated line", "PRP=1,2,127,-1\nPRP=1,2,521,-1\n");

    // File without a trailing newline: the new entry starts its own line.
    write("PRP=1,2,127,-1");
    io::WorktodoParser::appendLine(kPath, "PRP=1,2,521,-1");
    expectFile("append after an unterminated line", "PRP=1,2,127,-1\nPRP=1,2,521,-1\n");

    // CRLF file: already terminated.
    write("PRP=1,2,127,-1\r\n");
    io::WorktodoParser::appendLine(kPath, "PRP=1,2,521,-1");
    expectFile("append after a CRLF-terminated line", "PRP=1,2,127,-1\r\nPRP=1,2,521,-1\n");

    // Empty file.
    write("");
    io::WorktodoParser::appendLine(kPath, "PRP=1,2,521,-1");
    expectFile("append to an empty file", "PRP=1,2,521,-1\n");

    // Both entries are runnable afterwards: the first is parsed and removed, then the second.
    write("PRP=1,2,127,-1");
    io::WorktodoParser::appendLine(kPath, "PRP=1,2,521,-1");
    {
        io::WorktodoParser parser(kPath);
        const auto first = parser.parse();
        if (!first || first->exponent != 127U) {
            std::cerr << "FAIL first entry after append\n";
            ++failures;
        } else if (!parser.removeProcessedLine(first->rawLine)) {
            std::cerr << "FAIL removeProcessedLine\n";
            ++failures;
        } else {
            const auto second = parser.parse();
            if (!second || second->exponent != 521U) {
                std::cerr << "FAIL second entry after append\n";
                ++failures;
            } else {
                std::cout << "PASS both appended entries parse in order\n";
            }
        }
    }

    // Unwritable location.
    if (io::WorktodoParser::appendLine("no_such_directory/worktodo.txt", "PRP=1,2,521,-1")) {
        std::cerr << "FAIL append to a missing directory reported success\n";
        ++failures;
    } else {
        std::cout << "PASS append failure is reported\n";
    }

    std::filesystem::remove(kPath);
    std::filesystem::remove("worktodo_save.txt");
    if (failures != 0) {
        std::cerr << failures << " worktodo append check(s) failed\n";
        return 1;
    }
    std::cout << "Worktodo append test passed\n";
    return 0;
}
