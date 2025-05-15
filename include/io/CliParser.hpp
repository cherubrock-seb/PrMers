// CliParser.hpp
#ifndef IO_CLIPARSER_HPP
#define IO_CLIPARSER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace io {

struct CliOptions {
    uint32_t exponent = 0;
    uint32_t iterforce = 0;
    int device_id = 0;
    std::string mode = "prp";                // "prp" ou "ll"
    bool profiling = false;
    bool debug = false;
    bool gerbiczli = false;
    bool proof = true;
    bool submit = true;
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
    bool noAsk = false;
    std::string kernel_path;
    std::string output_path;
    std::string build_options = "";
    int proofPower = 1;
    std::string proofFile = "";
    int portCode = 8;
    std::string osName = "Linux";
    std::string osVersion = "14.0";
    std::string osArch = "x86_64";
    std::string aid = "";
    std::string uid = "";
    int waitPercentageFactor = 100;
    int res64_display_interval = 100000;
    bool cl_queue_throttle_active = false;
};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
};

} // namespace io

#endif // IO_CLIPARSER_HPP
