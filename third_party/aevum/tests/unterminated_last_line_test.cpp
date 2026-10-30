// tune.txt and config.txt are edited by hand, and some editors (Notepad) leave no newline after the last line.
// File::readLine() throws ReadError on such a line, which is right for worktodo files (lines are appended to them and
// deleted by exact match) but made a hand-edited tune.txt fail every engine create, and a config.txt stop startup.
// (Args::readConfig makes the same one-line opt-in; Args::parse drags in the device layer, so it is not linked here.)
#include "Args.h"
#include "File.h"
#include "TuneEntry.h"

#include <cstdarg>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

// Minimal host-only stubs needed by FFTConfig.cpp (see type4_pfa9_plan_test.cpp).
void log(const char*, ...) {}
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delimiter)) result.push_back(part);
  return result;
}

static int failures = 0;
static void require(bool ok, const std::string& what) {
  if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}

static void writeText(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
}

// Number of tune entries read, or -1 if reading threw.
static long tuneEntries(const fs::path& dir, const std::string& text) {
  writeText(dir / "tune.txt", text);
  Args args(true);
  args.masterDir = dir;
  try {
    return long(TuneEntry::readTuneFile(args).size());
  } catch (...) {
    return -1;
  }
}

int main() {
  fs::path dir = fs::temp_directory_path() / ("aevum-unterminated-" + std::to_string(getpid()));
  fs::create_directories(dir);

  const std::string a = "1.0 1:256:2:256:101";
  const std::string b = "2.0 1:512:2:512:101";
  require(tuneEntries(dir, a + "\n") == 1, "tune.txt with a final newline");
  require(tuneEntries(dir, a) == 1, "tune.txt whose only line has no final newline");
  require(tuneEntries(dir, a + "\n" + b) == 2, "tune.txt whose last line has no final newline");
  require(tuneEntries(dir, a + "\n" + b + "\n") == 2, "two-line tune.txt with a final newline");
  require(tuneEntries(dir, "") == 0, "empty tune.txt");
  require(tuneEntries(dir, a + "\n   ") == 1, "tune.txt ending in blanks with no newline");
  require(tuneEntries(dir, a + "\r\n" + b) == 2, "CRLF tune.txt whose last line has no final newline");

  // Everything else (worktodo) keeps the strict check.
  {
    writeText(dir / "strict.txt", "PRP=1,2,127,-1\nPRP=1,2,8191,-1");
    File f = File::openRead(dir / "strict.txt");
    bool threw = false;
    try { for (const std::string& line : f) { (void) line; } } catch (const ReadError&) { threw = true; }
    require(threw, "a file that did not opt in must still refuse an unterminated last line");
  }
  // The opt-in survives a move (File is returned by value from openRead and moved around).
  {
    writeText(dir / "moved.txt", "x\ny");
    File f = File::openRead(dir / "moved.txt");
    f.allowUnterminatedLastLine();
    File g = std::move(f);
    std::vector<std::string> lines;
    try { for (const std::string& line : g) lines.push_back(line); } catch (...) { lines.clear(); }
    require(lines.size() == 2 && lines[1] == "y\n", "opt-in must survive a move; the last line gets its newline");
  }

  fs::remove_all(dir);
  if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
  std::cout << "unterminated last line test ok\n";
  return 0;
}
