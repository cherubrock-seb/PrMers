#include "io/WorktodoManager.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main() {
    const fs::path base = fs::temp_directory_path() / "prmers_worktodo_manager_test";
    fs::remove_all(base);
    fs::create_directories(base);

    io::CliOptions good;
    good.save_path = (base / "save").string();
    fs::create_directories(good.save_path);
    io::WorktodoManager okWm(good);
    if (!okWm.saveIndividualJson(127, "prp", "{}") || !okWm.appendToResultsTxt("{}")) {
        std::cerr << "writable save path must report success\n";
        return 1;
    }
    std::ifstream res(fs::path(good.save_path) / "results.txt");
    std::string line;
    if (!std::getline(res, line) || line != "{}") {
        std::cerr << "results.txt content mismatch\n";
        return 1;
    }

    io::CliOptions bad;
    bad.save_path = (base / "missing" / "dir").string();
    io::WorktodoManager badWm(bad);
    if (badWm.saveIndividualJson(127, "prp", "{}")) {
        std::cerr << "saveIndividualJson must report failure for an unwritable path\n";
        return 1;
    }
    if (badWm.appendToResultsTxt("{}")) {
        std::cerr << "appendToResultsTxt must report failure for an unwritable path\n";
        return 1;
    }

    fs::remove_all(base);
    std::cout << "worktodo manager save-result test passed\n";
    return 0;
}
