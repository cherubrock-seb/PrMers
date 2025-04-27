// src/io/WorktodoManager.cpp
#include "io/WorktodoManager.hpp"
#include "io/CliParser.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace io {

WorktodoManager::WorktodoManager(const io::CliOptions& opts)
  : options_(opts)
{}

void WorktodoManager::saveIndividualJson(uint32_t p,
                                         const std::string& mode,
                                         const std::string& jsonResult) const
{
    // ex: ./save/100003_prp_result.json
    std::string file = options_.save_path + "/"
                     + std::to_string(p) + "_" + mode + "_result.json";
    std::ofstream out(file);
    if (!out) {
        std::cerr << "Cannot open " << file << " for writing JSON\n";
        return;
    }
    out << jsonResult;
    std::cout << "JSON result written to: " << file << "\n";
}

void WorktodoManager::appendToResultsTxt(const std::string& jsonResult) const
{
    // ex: ./save/results.txt
    std::string resultPath = options_.save_path + "/results.txt";
    std::ofstream resOut(resultPath, std::ios::app);
    if (resOut) {
        resOut << jsonResult << "\n";
        std::cout << "Result appended to: " << resultPath << "\n";
    } else {
        std::cerr << "Cannot open " << resultPath << " for appending\n";
    }
}

} // namespace core
