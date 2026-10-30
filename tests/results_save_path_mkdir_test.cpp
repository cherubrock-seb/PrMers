// The result files (<p>_<mode>_result.json and results.txt) go under the save path (-f).  A run that ends before anything
// else created that directory (a short P-1, ECM or LL-SAFE job) must still write them: WorktodoManager creates it.
#include "io/WorktodoManager.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const fs::path base = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "prmers-results-mkdir-test";
    fs::remove_all(base);
    io::CliOptions opts;
    opts.save_path = (base / "not" / "yet").string();
    io::WorktodoManager wm(opts);
    int failures = 0;
    if (!wm.saveIndividualJson(269, "pm1_stage1", "{\"exponent\":269}")) { std::puts("FAIL: individual JSON not written"); ++failures; }
    if (!wm.appendToResultsTxt("{\"exponent\":269}")) { std::puts("FAIL: results.txt not written"); ++failures; }
    std::ifstream in(base / "not" / "yet" / "results.txt");
    std::stringstream ss; ss << in.rdbuf();
    if (ss.str() != "{\"exponent\":269}\n") { std::puts("FAIL: results.txt content"); ++failures; }
    if (!fs::exists(base / "not" / "yet" / "269_pm1_stage1_result.json")) { std::puts("FAIL: no individual JSON file"); ++failures; }
    fs::remove_all(base);
    if (failures) return 1;
    std::puts("results save-path mkdir test passed");
    return 0;
}
