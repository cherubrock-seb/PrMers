// -verbose takes an optional level.  Gpu runs the developer self-tests (selftestTrig: on AMD, inline-asm timing kernels)
// only at -verbose 99 and above, not on every -verbose run, nor for an engine created with verbose = 1 (PrMers).
#include "Args.h"
#include "Proof.h"
#include "clwrap.h"
#include "gpuid.h"

#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Host-only stubs: Args::parse references the device layer (-h, -uid, -pci) and the proof size estimate (-h).
void log(const char*, ...) {}
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delimiter)) result.push_back(part);
  return result;
}
const char* VERSION = "test";
double ProofSet::diskUsageGB(u64, u32) { return 0; }
vector<cl_device_id> getAllDeviceIDs() { return {}; }
string getDeviceName(cl_device_id) { return {}; }
string getBoardName(cl_device_id) { return {}; }
string getDriverVersion(cl_device_id) { return {}; }
string getBdfFromDevice(cl_device_id) { return {}; }
std::string getUidFromBdf(const std::string&) { return {}; }
int getPosFromBdf(const std::string&) { return 0; }
int getPosFromUid(const std::string&) { return 0; }

static int failures = 0;
static void require(bool ok, const std::string& what) {
  if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}

// The verbose level after parsing `line`, or -1 if parsing threw.
static int levelOf(const std::string& line, std::string* user = nullptr) {
  Args args(true);
  try {
    args.parse(line);
  } catch (...) {
    return -1;
  }
  if (args.verbose != prpll_verbose) { require(false, "prpll_verbose must follow args.verbose for '" + line + "'"); }
  if (user) *user = args.user;
  return args.verbose;
}

int main() {
  require(Args(true).verbose == 0, "verbose is off by default");
  require(levelOf("-verbose") == 1, "-verbose alone is level 1");
  require(levelOf("-v") == 1, "-v alone is level 1");
  require(levelOf("-verbose 2") == 2, "-verbose 2");
  require(levelOf("-verbose 0") == 0, "-verbose 0 turns it off");
  require(levelOf("-verbose 99") == 99, "-verbose 99");
  std::string user;
  require(levelOf("-verbose -user tester", &user) == 1 && user == "tester", "-verbose followed by another option");
  require(levelOf("-verbose abc") == -1, "-verbose with a non-numeric level is refused");
  require(levelOf("-verbose 3x") == -1, "-verbose with trailing text is refused");

  // Only an explicit high level runs the self-tests; plain -verbose and an engine's verbose flag do not.
  require(levelOf("-verbose") < Args::VERBOSE_SELFTEST, "plain -verbose must not run the self-tests");
  require(levelOf("-verbose 99") >= Args::VERBOSE_SELFTEST, "-verbose 99 runs the self-tests");
  Args engine(true);
  engine.verbose = true;   // EngineApi: args_.verbose = verbose (a bool)
  require(engine.verbose < Args::VERBOSE_SELFTEST, "an engine created with verbose = 1 must not run the self-tests");

  if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
  std::cout << "verbose level test ok\n";
  return 0;
}
