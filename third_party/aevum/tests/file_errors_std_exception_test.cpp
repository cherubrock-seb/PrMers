// Host-only test: CRCError, ReadError and WriteError are std::exceptions, so the top-level handlers (main(), gpuWorker(),
// Background::run()) log them instead of letting them reach std::terminate.

#include "Background.h"
#include "File.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <type_traits>
#include <unistd.h>

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

static_assert(std::is_base_of_v<std::exception, CRCError>);
static_assert(std::is_base_of_v<std::exception, ReadError>);
static_assert(std::is_base_of_v<std::exception, WriteError>);
static_assert(std::is_base_of_v<FileError, CRCError>);

template<class F> static bool caughtAsStdException(F f, const std::string& expectName) {
  try {
    f();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(expectName) != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

int main() {
  char dirTemplate[] = "file_errors_XXXXXX";
  if (!mkdtemp(dirTemplate)) { std::perror("mkdtemp"); return 2; }
  std::string dir = dirTemplate;

  check(caughtAsStdException([] { throw ReadError{"r.txt"}; }, "r.txt"), "ReadError caught as std::exception");
  check(caughtAsStdException([] { throw WriteError{"w.txt"}; }, "w.txt"), "WriteError caught as std::exception");
  check(caughtAsStdException([] { throw CRCError{"c.txt"}; }, "c.txt"), "CRCError caught as std::exception");

  // `name` stays available to catch-by-type sites.
  try { throw CRCError{"named"}; } catch (const CRCError& e) { check(e.name == "named", "name member kept"); }

  // A worktodo-style file whose last line has no newline: File::readLine throws ReadError.
  {
    std::string path = dir + "/worktodo.txt";
    File::append(path, "PRP=N/A,1,2,9689,-1,75,0");
    File f = File::openRead(path);
    check(caughtAsStdException([&] { f.readLine(); }, "worktodo.txt"), "unterminated line is a std::exception");
  }

  // A corrupt checksum: File::readChecked throws CRCError.
  {
    std::string path = dir + "/bad.crc";
    File::openWrite(path).write(std::string("\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c", 12));
    File f = File::openRead(path);
    check(caughtAsStdException([&] { f.readChecked<u32>(2); }, "bad.crc"), "bad CRC is a std::exception");
  }

  // Background::run() must survive a file error thrown by a task (a full disk while saving) and run the next task.
  {
    Background background;
    int ran = 0;
    background([] { throw WriteError{"savefile"}; });
    background([&ran] { ran = 1; });
    background.waitEmpty();
    check(ran == 1, "Background keeps running after a WriteError");
  }

  std::string cmd = "rm -rf '" + dir + "'";
  if (std::system(cmd.c_str())) { std::perror("cleanup"); }

  if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
  std::puts("file_errors_std_exception_test: ok");
  return 0;
}
