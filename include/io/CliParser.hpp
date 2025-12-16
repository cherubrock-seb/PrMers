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
    uint64_t llsafe_block = 0;
    uint64_t B3 = 0;
    bool marin = true;
    bool bench = false;
    bool profiling = false;
    bool debug = false;
    bool verify = true;
    bool gerbiczli = true;
    uint64_t B1 = 0;
    uint64_t memlim = 85;
    uint64_t B1old = 0;
    uint64_t B1_new = 0;
    uint64_t B2 = 0;
    uint64_t B4 = 1;
    uint64_t checklevel = 0;
    uint64_t gerbicz_error_count = 0;
    uint64_t erroriter = 0;
    bool proof = true;
    bool edwards = false;
    uint32_t ecm_check_interval = 300;
    bool compute_edwards = false;
    bool torsion16 = false;
    bool notorsion = false;
    uint64_t sigma = 0ULL;
    uint64_t seed = 0ULL;
    bool resume = true;
    bool submit = false;
    uint64_t chunk256 = 4;
    uint64_t K = 0;
    uint64_t nmax = 0;
    bool bsgs = false;
    uint64_t brent = 0; 
    int localCarryPropagationDepth = 8;
    int enqueue_max = 0;
    int backup_interval = 300;
    std::string save_path = ".";
    std::string user;
    std::string password;
    std::string computer_name;
    uint64_t sigma192 = 0ULL;
    std::string config_path;
    std::string worktodo_path = "worktodo.txt";
    std::string pm1_extend_save_path = "";
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
    std::string sigma_hex = "";
    uint64_t curve_seed = 0ULL;
    uint64_t base_seed = 0ULL;
    std::string uid = "";
    int res64_display_interval = 0;
    bool cl_queue_throttle_active = false;
    std::vector<std::string> knownFactors;
    bool gui = false;
    int http_port = 3131;
    std::string http_host = "localhost";
    bool ipv4 = true;
    uint64_t max_e_bits = 268'435'456ULL;
    uint64_t curves_tested_for_found = 0;
    int invarianterror = 0;

};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
};

} // namespace io

#endif // IO_CLIPARSER_HPP
