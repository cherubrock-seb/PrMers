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
    bool torus = false;
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
    bool aevum = false;
    bool aevum_auto = true;
    bool force_engine_marin = false;
    std::string aevum_fft_spec = "";
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
    double sieveDepth = 0.0;
    uint64_t B2Start = 0;
    uint64_t B4 = 1;
    uint64_t checklevel = 0;
    uint64_t gerbicz_error_count = 0;
    uint64_t erroriter = 0;
    bool proof = true;
    bool edwards = false;
    uint32_t ecm_check_interval = 600;
    bool compute_edwards = true;
    bool torsion16 = false;
    bool family_iv_163 = false;
    bool notorsion = true;
    std::string sigma;
    uint64_t seed = 0ULL;
    uint64_t tbits = 500000ULL;
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
    bool p95stage2 = false;
    bool pm1_lowmem = false;
    bool pm1_ultralowmem = false;
    // True low-memory P-1 Stage 2 from an existing Stage-1 resume.
    // Uses 2 GPU registers and computes H^Q where Q=prod primes(stage2Low,B2].
    // B1 remains the Stage-1 resume bound; B2Start optionally selects stage2Low
    // for split ranges such as B1=1500000, B2Start=2000000, B2=2500000.
    // This is intended for 10GB GPUs such as RTX 3080 and does not recompute from base 3.
    bool pm1_s2_resume2reg = false;
    bool pm1_vtrace = false;          // Accepted legacy flag; normal-memory P-1 Stage 2 uses V-trace by default
    bool pm1_vtrace_off = false;      // Disable default V-trace Stage 2 and use the classic BSGS path
    uint64_t pm1_vtrace_D = 0;        // Optional giant-step D override; default is auto-D, fallback 630
    bool pm1_vtrace_auto_d = false;   // Explicit auto-select D for V-trace from a small candidate set
    bool pm1_vtrace_auto_d_aggressive = false; // Aggressive auto-D profile; raises default register cap to 8192
    bool pm1_vtrace_deep_d_auto = false; // Primorial-aware deep auto-D profile, default normal-memory policy in v61
    uint64_t pm1_vtrace_max_regs = 0; // Optional auto-D register cap; default 4096, or 8192 with aggressive/deep auto-D
    bool pm1_vtrace_auto_batch = true;  // v85: default; allow auto-D to consider baby batching, selected only when cost model wins
    uint64_t pm1_vtrace_max_batches = 4; // v78: cap automatic baby-window passes while scoring D candidates when auto-batch is enabled
    uint64_t pm1_vtrace_baby_batch = 0; // v72: optional active baby count override; env PRMERS_PM1_VTRACE_BABY_BATCH still supported
    bool pm1_vtrace_negadd_off = false; // Disable negative-baby/add term builder and use the older copy+sub_reg path
    bool pm1_vtrace_pair95 = false;     // v97: explicitly enable Pair95 irregular prime pairing (default when dense map is available)
    bool pm1_vtrace_pair95_off = false; // v97: disable default Pair95 and use classic V-trace pairing
    uint64_t pm1_vtrace_pair95_L = 0;   // v97: optional Pair95 level count override; 0 = auto
    bool pm1_vtrace_product_tree = false; // Experimental v62 bucket-local product-tree Stage 2 accumulation (opt-in)
    uint32_t pm1_vtrace_product_tree_width = 16; // Scratch fan-in/chunk width for product-tree experiment
    bool pm1_no_stage1_gcd = false;
    std::string p95path;
    int max_local_size1 = 0;
    int max_local_size2 = 0;
    int max_local_size3 = 0;
    int max_local_size5 = 0;
    bool noAsk = false;
    bool doubleCheck = false;        // Worktodo DoubleCheck= task; still runs LL but preserves task kind
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
    std::vector<std::string> knownFactors_start;
    bool gui = false;
    int http_port = 3131;
    std::string http_host = "localhost";
    bool ipv4 = true;
    uint64_t max_e_bits = 268'435'456ULL;
    uint64_t curves_tested_for_found = 0;
    int invarianterror = 0;
    uint32_t ecm_progress_interval_ms = 2000;
    bool s3only = false;
    bool s4only = false;
};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
};

} // namespace io

#endif // IO_CLIPARSER_HPP
