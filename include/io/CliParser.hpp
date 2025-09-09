// CliParser.hpp
#ifndef IO_CLIPARSER_HPP
#define IO_CLIPARSER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace io {

struct CliOptions {
    uint64_t exponent = 0;
    uint64_t iterforce = 0;
    uint64_t iterforce2 = 0;
    bool wagstaff = false;
    int device_id = 0;
    bool tune = false;
    std::string mode = "prp";                // "prp" ou "ll"
    std::string filemers = "";
    std::string filep95 = "";
    bool exportp95 = false;
    bool resume95 = false;
    bool exportmers = false; 
    bool marin = true;
    bool bench = false;
    bool profiling = false;
    bool debug = false;
    bool gerbiczli = true;
    uint64_t B1 = 10000;
    uint64_t B2 = 0;
    uint64_t checklevel = 0;
    uint64_t gerbicz_error_count = 0;
    uint64_t erroriter = 0;
    bool proof = true;
    bool resume = false;
    bool submit = false;
    uint64_t chunk256 = 4;
    int localCarryPropagationDepth = 8;
    int enqueue_max = 0;
    int backup_interval = 3000;
    std::string save_path = ".";
    std::string user;
    std::string password;
    std::string computer_name;
    std::string config_path;
    std::string worktodo_path = "worktodo.txt";
    int max_local_size1 = 0;
    int max_local_size2 = 0;
    int max_local_size3 = 0;
    int max_local_size5 = 0;
    bool noAsk = false;
    std::string kernel_path;
    std::string output_path;
    std::string build_options = "";
    uint32_t proofPower = 1;
    bool manual_proofPower = false; // Track if proof power is set manually
    std::string proofFile = "";
    int portCode = 8;
    std::string osName = "Linux";
    std::string osVersion = "14.0";
    std::string osArch = "x86_64";
    std::string aid = "";
    std::string uid = "";
    int res64_display_interval = 0;
    bool cl_queue_throttle_active = false;
    std::vector<std::string> knownFactors;
};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
};

} // namespace io

#endif // IO_CLIPARSER_HPP
