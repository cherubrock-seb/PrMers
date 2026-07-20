// CliParser.cpp
/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
#include "io/CliParser.hpp"
#include "util/StringUtils.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "util/PathUtils.hpp"
#include <filesystem>
#include <stdexcept>
#include "opencl/Context.hpp"
#include "core/Version.hpp"

// Forward-declare the usage function (defined elsewhere, e.g. in your main host file)
extern void printUsage(const char* progName);

namespace io {
// -----------------------------------------------------------------------------
// Usage printing helper (updated with new backup and path options)
// -----------------------------------------------------------------------------
void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <p> [-d <device_id>] [-c <localCarryPropagationDepth>]" << std::endl;
    std::cout << "              [-profile] [-prp|-ll] [-t <backup_interval>] [-f <path>] [-computer <name>]" << std::endl;
    std::cout << "              [--noask] [-user <username>]" << std::endl;
    std::cout << "              [-enqueue_max <value>] [-worktodo <path>] [-config <path>] [-iterforce] " << std::endl;
    std::cout << "               " << std::endl;
    std::cout << std::endl;
    std::cout << "  <p>       : Exponent to test (required unless -worktodo is used)" << std::endl;
    std::cout << "  -d <device_id>       : (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -c <depth>           : (Optional) Set local carry propagation depth (default: 8)" << std::endl;
    //std::cout << "  -profile             : (Optional) Enable kernel execution profiling" << std::endl;
    std::cout << "  -prp                 : (Optional) Run in PRP mode (default). Uses initial value 3; final result must equal 9" << std::endl;
    std::cout << "  -ll                  : (Optional) Run in Lucas-Lehmer SAFE mode (with Gerbicz-Li). Uses initial value 4 and p-2 iterations" << std::endl;
    std::cout << "  -llunsafe            : (Optional) Run in Lucas-Lehmer UNSAFE mode (no Gerbicz-Li). Uses initial value 4 and p-2 iterations" << std::endl;
    std::cout << "  -llsafe2             : (Optional) Run on GPU in Lucas-Lehmer Doubling Safe mode ( LL Doubling error check by block)" << std::endl;
    std::cout << "  -llsafeb             : (Optional) override length for block verification in llsafe Doubling mode by default exponent/sqrt(exponent) " << std::endl;
    //std::cout << "  -llsafecpu           : (Optional) Run on CPU in Lucas-Lehmer Safe mode (matrix squaring with Gerbicz-Li check)" << std::endl;
    std::cout << "  -factors <factor1,factor2,...> : (Optional) Specify known factors to run PRP test on the Mersenne cofactor" << std::endl;
    std::cout << "  -pm1                 : (Optional) Run factoring P-1" << std::endl;
    std::cout << "  -b1                  : (Optional) B1 for factoring P-1" << std::endl;
    std::cout << "  -b2                  : (Optional) B2 for factoring P-1" << std::endl;
    std::cout << "  -b1old               : (Optional) Continue P-1 factoring stage 1 from a previous run. "
             "Loads a file named 'resume_p[Exponent]_B1_[oldB1].save (GMP-ECM) or .p95 (if .save not found) '. "
             "Use -b1 to specify the new B1 bound for extension." << std::endl;
    std::cout << "  -K <value>           : Exponent K for the n^K variant of P-1 stage 2" << std::endl;
    std::cout << "  -nmax <value>        : Maximum value of n for the n^K variant of P-1 stage 2" << std::endl;
    std::cout << "  -t <seconds>         : (Optional) Specify backup interval in seconds (default: 120)" << std::endl;
    std::cout << "  -f <path>            : (Optional) Specify path for saving/loading checkpoint files (default: current directory)" << std::endl;
    //std::cout << "  -l1 <value>          : (Optional) Force local size max for NTT kernels" << std::endl;
    //std::cout << "  -l5 <value>          : (Optional) Force local size max for NTT kernels radix 5" << std::endl;
    //std::cout << "  -l2 <value>          : (Optional) Force local size for 2-step radix-16 NTT kernel" << std::endl;
    //std::cout << "  -l3 <value>          : (Optional) Force local size for mixed radix NTT kernel" << std::endl;
    //std::cout << "  -submit              : (Optional) activate the possibility to send results to PrimeNet (prompt or autosend)" << std::endl;
    std::cout << "  --noask              : (Optional) Automatically send results to PrimeNet without prompting" << std::endl;
    //std::cout << "  -user <username>     : (Optional) PrimeNet username to auto-fill during result submission" << std::endl;
    //std::cout << "  -password <password> : (Optional) PrimeNet password to autosubmit the result without prompt (used only when -no-ask is set)" << std::endl;
    std::cout << "  -computer <name>     : (Optional) PrimeNet computer name to auto-fill the result submission" << std::endl;
    std::cout << "  -worktodo <path>     : (Optional) Load exponent from specified worktodo.txt (default: ./worktodo.txt)" << std::endl;
    std::cout << "  -config <path>       : (Optional) Load config file from specified path" << std::endl;
    std::cout << "  -proof <level>       : (Optional) Set proof power between 1 and 12 or 0 to disable proof generation (default: optimal proof power selected automatically)" << std::endl;
    std::cout << "  -noverify            : (Optional) Skip verification of the generated PRP proof (useful for benchmarking or when verification will be done separately)" << std::endl;
    std::cout << "  -erroriter <iter>    : (Optional) injects an error at iteration <iter> to test Gerbicz-Li error detection mechanism." << std::endl;
    std::cout << "  -iterforce <iter>    : (Optional) forces a GPU queue synchronization (clFinish) every <iter> iterations to improve stability or allow interruption checks." << std::endl;
    std::cout << "  -iterforce2 <iter>   : (Optional) forces a GPU queue synchronization in P-1 stage 2 (clFinish) every <iter> iterations to improve stability or allow interruption checks." << std::endl;
    std::cout << "  -gerbiczli           : (Optional) deactivate gerbicz li error check" << std::endl;
    std::cout << "  -pm1-lowmem          : (Optional) P-1 stage 1 low-memory mode: 3 GPU registers, Gerbicz-Li disabled" << std::endl;
    std::cout << "  -pm1-ultralowmem     : (Optional) P-1 stage 1 ultra-low-memory mode: 1 GPU register, fast3 only, Gerbicz-Li disabled" << std::endl;
    std::cout << "  -pm1-s2-resume2reg   : (Optional) P-1 Stage 2 ultra-low-memory true resume mode: load resume_p...B1_<B1>.p95/.save and compute H^prod(primes) with 2 GPU registers" << std::endl;
    std::cout << "  -pm1-vtrace          : P-1 Stage 2 scalar trace BSGS (default for normal-memory Stage 2, primorial-aware auto-D)" << std::endl;
    std::cout << "  -pm1-vtrace-off      : Disable default V-trace and use the previous classic Stage 2 BSGS path" << std::endl;
    std::cout << "  -pm1-vtrace-d <D>    : (Optional) Force V-trace D. Use an even primorial/highly-composite D such as 4620/30030" << std::endl;
    std::cout << "  -pm1-vtrace-auto-d   : (Optional) Explicitly auto-select D for V-trace under the default primorial-aware register cap" << std::endl;
    std::cout << "  -pm1-vtrace-deep-d auto : (Optional) Explicit deep primorial-aware auto-D profile; current default is auto" << std::endl;
    std::cout << "  -pm1-vtrace-auto-d-aggressive : (Optional) auto-select D with a larger default cap (8192 regs) for normal-size Mersennes" << std::endl;
    std::cout << "  -pm1-vtrace-max-regs <N> : (Optional) register cap for V-trace auto-D, default 4096 or 8192 with aggressive/deep auto-D" << std::endl;
    std::cout << "  -pm1-vtrace-auto-batch : Explicitly allow auto-D baby-window batching (v85: already default; only used when predicted faster)" << std::endl;
    std::cout << "  -pm1-vtrace-baby-batch <N> : (Optional) force V-trace active baby traces per pass. Keeps D large but scans several baby windows" << std::endl;
    std::cout << "  -pm1-vtrace-max-batches <N> : (Optional) cap auto-D baby-window passes, default 4" << std::endl;
    std::cout << "  -pm1-vtrace-no-auto-batch : Disable default integrated D+batch scoring; useful for flat/full-slab regression tests" << std::endl;
    std::cout << "  -pm1-vtrace-pair95 : Enable Pair95 irregular prime pairing explicitly (v97: default when dense map is available)" << std::endl;
    std::cout << "  -pm1-vtrace-pair95-off : Disable default Pair95 and use classic V-trace pairing" << std::endl;
    std::cout << "  -pm1-vtrace-pair95-l <N> : Force Pair95 irregular level count; 0/omitted = auto, common values 2 or 3" << std::endl;
    std::cout << "  -pm1-vtrace-product-tree : (Experimental) Use v62 bucket-local product-tree accumulation, opt-in" << std::endl;
    std::cout << "  -pm1-vtrace-product-tree-width <N> : (Experimental) Product-tree scratch/chunk width, default 16" << std::endl;
    std::cout << "  -b2start <value>     : (Optional) Stage 2 lower bound/start for split ranges. With -pm1-s2-resume2reg, -b1 remains the Stage-1 resume bound and primes in (-b2start,-b2] are tested" << std::endl;
    std::cout << "  -nogcd-stage1        : (Optional) skip the ordinary P-1 Stage 1 GCD after writing PM1 resume/checkpoint; useful before Stage 2" << std::endl;
    std::cout << "  -checklevel <value>  : (Optional) Will force gerbicz check every B*<value> by default check is done every 10 min and at the end." << std::endl;
    std::cout << "  -wagstaff            : (Optional) will check PRP if (2^p + 1)/3 is probably prime" << std::endl;
    std::cout << "  -ecm -b1 <B1> [-b2 <B2>] -K <curves> : Run ECM factoring with bounds B1 [and optional B2], on given number of curves" << std::endl;
    
    std::cout << "  -montgomery          : (Optional) compute in Montgomery and use Montgomery (compute done in montgomery)" << std::endl;
    std::cout << "  -edwards             : (Optional) compute in Montgomery and use (twisted) Edwards curve converted to Montgomery (compute done in Montgomery)" << std::endl;
    std::cout << "  -ced                 : (Optional) compute in Twisted Edwards (by default) and use (twisted) Edwards curves (notorsion twisted or torsion 2x8 possible no twist a=1) " << std::endl;
    std::cout << "  -cmont               : (Optional) compute in Montgomery (Twisted Edwards by default) and use (twisted) Edwards curves (notorsion twisted or torsion 2x8 possible no twist a=1) " << std::endl;
    std::cout << "  -seed                : (Optional) force a curve seed" << std::endl;
    std::cout << "  -sigma               : (Optional) force a curve sigma in Montgomery (notorsion mode)" << std::endl;
    std::cout << "  -torsion8            : (Optional) use torsion-8" << std::endl;
    std::cout << "  -torsion16           : (Optional) use torsion-16" << std::endl;
    std::cout << "  -notorsion           : (Optional) use no torsion instead of default torsion-16" << std::endl;
    std::cout << "  -iv163               : (Optional) use family_iv_163 curves (Gélin-Kleinjung-Lenstra) gives 16/3 average v2 (around order 32 point)" << std::endl;
    
    std::cout << "  -ecm_check_interval <value> : ECM Error Check interval in seconds (300s by default)" << std::endl;
    std::cout << "  -ecm_progress_ms <value>    : ECM progress update interval in milliseconds (default: 2000 ms)" << std::endl;
    std::cout << "  -p95path <path>      : (Optional) Prime95/mprime directory for ECM Stage2 handoff; enables Prime95 Stage2" << std::endl;
    std::cout << "  -nop95stage2         : (Optional) Disable Prime95 Stage2 handoff even if -p95path is set" << std::endl;
    //std::cout << "  -brent [<d>]         : (Optional) use Brent-Suyama variant with default or specified degree d (e.g., -brent 6)" << std::endl;
    //std::cout << "  -bsgs                : (Optional) enable batching of multipliers in ECM stage 2 to reduce ladder calls" << std::endl;
    std::cout << "  Backend selection (default: automatic Marin/Aevum):" << std::endl;
    std::cout << "  -aevum               : Strictly force Aevum; exit with an error when no supported Aevum plan is available" << std::endl;
    std::cout << "  -engine-marin        : Force the Marin engine::Reg backend" << std::endl;
    std::cout << "  -aevum-auto          : Explicitly select automatic Marin/Aevum mode (macOS still defaults to Marin unless -aevum is used)" << std::endl;
    std::cout << "  -marin               : Legacy internal PrMers NTT path (not supported with -llunsafe)" << std::endl;
    std::cout << "  -aevum-fft <spec>    : Force an Aevum plan; pfa9:4 is capacity-adaptive, pfa9full:4 forces all three planes" << std::endl;
    std::cout << "  -pfa9-type4          : Force type-4 policy; automatically elides redundant FP32 when exact FFT3161 is sufficient" << std::endl;
    std::cout << "  -pfa9-type4-full     : Diagnostic only: force full FP32+GF31+GF61 PFA9 plan" << std::endl;
    std::cout << "  -pfa [3|9]           : Enable/force native Aevum Good-Thomas PFA (auto when omitted)" << std::endl;
    std::cout << "  -pfa3 / -pfa9        : Force native Aevum PFA radix 3 or radix 9" << std::endl;
    std::cout << "  -pfa-off             : Keep the stock power-of-two Aevum plan" << std::endl;
    std::cout << "  Auto policy env      : AEVUM_AUTO_MAX_RATIO or workload-specific AEVUM_AUTO_PM1_STAGE1_MAX_RATIO, AEVUM_AUTO_PM1_STAGE2_MAX_RATIO, AEVUM_AUTO_ECM_MAX_RATIO" << std::endl;
    std::cout << "  -resume              : (Optional) write GMP-ECM and Prime 95 resume file after P-1 stage 1" << std::endl;
    //std::cout << "  -p95                 : (Optional) write Prime 95 resume file after P-1 stage 1" << std::endl;
    std::cout << "  -res64_display_interval <N> : (Optional) (only in -marin mode) Display Res64 every N iterations (0 = disabled or > 0, default = 100000)" << std::endl;
    std::cout << "  -bench               : (Optional) run benchmark on all NTT transform sizes" << std::endl;
   // std::cout << "  -chunk256 <1..4>     : (Optional) cap for CHUNK256; lower can help on Radeon VII/GCN (default: auto)" << std::endl;
    std::cout << "  -filemers <path>     : (Optional) Export .mers file to GMP-ECM .save format using stored state" << std::endl;
    //std::cout << "  -filep95 <path>      : (Optional) Export .mers file to Prime95 .p95 format using stored state" << std::endl;
    std::cout << "  -gui                  : (Optional) Enable the embedded web GUI accessible via your browser" << std::endl;
    std::cout << "  -http <port>          : (Optional) Specify the HTTP port for the GUI server (default: 3131)" << std::endl;
    std::cout << "  -host <ip|0.0.0.0|localhost> : (Optional) Specify the HTTP host for the GUI server (default: 127.0.0.1)" << std::endl;
    //std::cout << "  -ipv4                 : (Optional) Set the HTTP host to the first IPv4 interface" << std::endl;
    
    std::cout << "  -maxe <value>         : (Optional) Max bits for each E chunk (in MiB). If set to 0, defaults to 10000 bits. Example: -maxe 64 -> 64 MiB = 536870912 bits. By default if no -maxe you it is set to 32 Mib." << std::endl;
    std::cout << "  -memtest              : GPU Memory & Stability test (OpenCL)" << std::endl;
    std::cout << "  -memlim <percent>     : (Optional) Fraction percentage of memory used (used precompute stage 2 p-1)" << std::endl;

    //std::cout << "  -throttle_low        : (Optional) Enable CL_QUEUE_THROTTLE_LOW_KHR if OpenCL >= 2.2 (default: disabled)" << std::endl;
    //std::cout << "  -tune               : (Optional) Automatically determine the best pacing (iterForce) and how often to call clFinish() to synchronize kernels (default: disabled)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:\n  " << progName << " 127 mad -c 16 -profile -ll -t 120 -f /my/backup/path \\\n"
              << "            -l1 256 -l2 128 -l3 64 --noask -user myaccountname -enqueue_max 65536 \\\n"
              << "            -worktodo ./mydir/worktodo.txt -config ./mydir/settings.cfg -proof" << std::endl;
    prmers::ocl::Context::listAllOpenCLDevices();
}

static uint64_t to_u64(const char* s){
    char* end=nullptr;
    errno=0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if(errno || end==s) return 0ULL;
    return static_cast<uint64_t>(v);
}

CliOptions CliParser::parse(int argc, char** argv ) {
    // Early check for -h, --help or -help
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0
         || std::strcmp(argv[i], "--help") == 0
         || std::strcmp(argv[i], "-help") == 0)
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0
        || std::strcmp(argv[i], "--version") == 0
        || std::strcmp(argv[i], "-version") == 0)
        {
            std::cout << "prmers Release v" << core::PRMERS_VERSION << "\n";
            std::exit(EXIT_SUCCESS);
        }
    }
    CliOptions opts;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            opts.device_id = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-prp") == 0) {
            opts.mode = "prp";
        }
        else if (std::strcmp(argv[i], "-llunsafe") == 0) {
            opts.mode = "ll";
            opts.proof = false;
        }
        /*else if (std::strcmp(argv[i], "-llsafecpu") == 0) {
            opts.mode = "llsafecpu";
            opts.proof = false;
        }*/
        else if (std::strcmp(argv[i], "-ll") == 0) {
            opts.mode = "llsafe";
            opts.proof = false;
        }
        else if (std::strcmp(argv[i], "-llsafe2") == 0) {
            opts.mode = "llsafe2";
            opts.proof = false;
        }
        else if (std::strcmp(argv[i], "-pm1") == 0) {
            opts.mode = "pm1";
            //opts.marin = false;
            opts.proof = false;
        }
        else if (std::strcmp(argv[i], "-torus") == 0) {
            opts.torus = true;
        }
        else if (std::strcmp(argv[i], "-ecm") == 0) {
            opts.mode = "ecm";
            //opts.marin = false;
            //opts.proof = false;
        }
        else if (std::strcmp(argv[i], "-bsgs") == 0) {
            opts.bsgs = true;
        }
        else if (std::strcmp(argv[i], "-brent") == 0 && i + 1 < argc) {
            opts.brent = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-profile") == 0) {
            opts.profiling = true;
        }
        else if (std::strcmp(argv[i], "-debug") == 0) {
            opts.debug = true;
        }
        else if (std::strcmp(argv[i], "-marin") == 0) {
            opts.marin = false;
            opts.aevum = false;
            opts.aevum_auto = false;
            opts.force_engine_marin = false;
        }
        else if (std::strcmp(argv[i], "-engine-marin") == 0 || std::strcmp(argv[i], "-backend-marin") == 0) {
            opts.marin = true;
            opts.aevum = false;
            opts.aevum_auto = false;
            opts.force_engine_marin = true;
        }
        else if (std::strcmp(argv[i], "-aevum") == 0) {
            opts.aevum = true;
            opts.aevum_auto = false;
            opts.force_engine_marin = false;
            opts.marin = true;
        }
        else if (std::strcmp(argv[i], "-aevum-auto") == 0 || std::strcmp(argv[i], "-backend-auto") == 0) {
            opts.aevum = false;
            opts.aevum_auto = true;
            opts.force_engine_marin = false;
            opts.marin = true;
        }
        else if (std::strcmp(argv[i], "-aevum-fft") == 0 && i + 1 < argc) {
            opts.aevum = true;
            opts.aevum_auto = false;
            opts.force_engine_marin = false;
            opts.marin = true;
            opts.aevum_fft_spec = argv[++i];
        }
        else if (std::strcmp(argv[i], "-pfa9-type4") == 0 ||
                 std::strcmp(argv[i], "-pfa9-type4-fast") == 0 ||
                 std::strcmp(argv[i], "-pfa9-fft323161") == 0) {
            opts.aevum = true;
            opts.aevum_auto = false;
            opts.force_engine_marin = false;
            opts.marin = true;
            opts.aevum_pfa_radix = 9;
            opts.aevum_pfa_off = false;
            opts.aevum_fft_spec = "pfa9:4:512:9:512:202";
        }
        else if (std::strcmp(argv[i], "-pfa9-type4-full") == 0) {
            opts.aevum = true;
            opts.aevum_auto = false;
            opts.force_engine_marin = false;
            opts.marin = true;
            opts.aevum_pfa_radix = 9;
            opts.aevum_pfa_off = false;
            opts.aevum_fft_spec = "pfa9full:4:512:9:512:202";
        }
        else if (std::strcmp(argv[i], "-pfa-off") == 0 ||
                 std::strcmp(argv[i], "-no-pfa") == 0) {
            opts.aevum_pfa_radix = 0;
            opts.aevum_pfa_off = true;
            opts.aevum_fft_spec.clear();
        }
        else if (std::strcmp(argv[i], "-pfa") == 0 ||
                 std::strcmp(argv[i], "-pfa-auto") == 0 ||
                 std::strncmp(argv[i], "-pfa=", 5) == 0 ||
                 std::strcmp(argv[i], "-pfa3") == 0 ||
                 std::strcmp(argv[i], "-pfa9") == 0) {
            int radix = -1;
            if (std::strcmp(argv[i], "-pfa3") == 0) radix = 3;
            else if (std::strcmp(argv[i], "-pfa9") == 0) radix = 9;
            else if (std::strncmp(argv[i], "-pfa=", 5) == 0) {
                const char* value = argv[i] + 5;
                if (std::strcmp(value, "3") == 0) radix = 3;
                else if (std::strcmp(value, "9") == 0) radix = 9;
                else if (std::strcmp(value, "auto") != 0) {
                    throw std::runtime_error("-pfa accepts only auto, 3, or 9");
                }
            } else if (i + 1 < argc &&
                       (std::strcmp(argv[i + 1], "3") == 0 || std::strcmp(argv[i + 1], "9") == 0)) {
                radix = std::atoi(argv[++i]);
            }
            opts.aevum = true;
            opts.aevum_auto = false;
            opts.force_engine_marin = false;
            opts.marin = true;
            opts.aevum_pfa_radix = radix;
            opts.aevum_pfa_off = false;
            opts.aevum_fft_spec = radix == 3 ? "pfa:3" : radix == 9 ? "pfa:9" : "pfa:auto";
        }
        else if (std::strcmp(argv[i], "-s3") == 0) {
            opts.s3only = true;
        }
        else if (std::strcmp(argv[i], "-s4") == 0) {
            opts.s4only = true;
        }
        else if (std::strcmp(argv[i], "-montgomery") == 0) {
            opts.edwards = false;
        }
        else if (std::strcmp(argv[i], "-edwards") == 0) {
            opts.edwards = true;
        }
        else if (std::strcmp(argv[i], "-ced") == 0) {
            opts.compute_edwards = true;
           // opts.torsion16 = true;
            //opts.notorsion = false;
        }
        else if (std::strcmp(argv[i], "-cmont") == 0) {
            opts.compute_edwards = false;
           // opts.torsion16 = true;
            //opts.notorsion = false;
        }
        else if (std::strcmp(argv[i], "-torsion8") == 0) {
            opts.torsion16 = false;
            opts.notorsion = false;
        }
        else if (std::strcmp(argv[i], "-torsion16") == 0) {
            opts.torsion16 = true;
            opts.notorsion = false;
        }
        else if (std::strcmp(argv[i], "-iv163") == 0) {
            opts.family_iv_163 = true;
            opts.notorsion = false;
        }
        else if (std::strcmp(argv[i], "-notorsion") == 0) {
            opts.notorsion = true;
        }
        else if (std::strcmp(argv[i], "-submit") == 0) {
            opts.submit = true;
        }
        else if (std::strcmp(argv[i], "-memtest") == 0) {
            opts.mode = "memtest";
            opts.exponent = 127;
        }
        else if (std::strcmp(argv[i], "-bench") == 0) {
            opts.bench = true;
            opts.exponent = 127;
        }
        else if (std::strcmp(argv[i], "-gui") == 0) {
            opts.gui = true;
        }
        else if (std::strcmp(argv[i], "-p95path") == 0 && i + 1 < argc) {
            opts.p95path = argv[++i];
            opts.p95stage2 = true;
        }
        else if (std::strcmp(argv[i], "-nop95stage2") == 0) {
            opts.p95stage2 = false;
        }
        else if (std::strcmp(argv[i], "-http") == 0 && i + 1 < argc) {
            opts.http_port = to_u64(argv[++i]);
            opts.gui = true;
        }
        else if (std::strcmp(argv[i], "-host") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.http_host = argv[++i];
            } else {
                std::cerr << "Error: missing value for -host\n";
                std::exit(1);
            }
        }
        else if (std::strcmp(argv[i], "-ipv4") == 0) {
            opts.ipv4 = true;
        }
        else if (std::strcmp(argv[i], "-throttle_low") == 0) {
            opts.cl_queue_throttle_active = true;
        }
        else if (std::strcmp(argv[i], "-proof") == 0 && i + 1 < argc) {
            int level = to_u64(argv[++i]);
            if (level == 0) {
                opts.proof = false;
            } else if (1 <= level && level <= 12) {
                opts.proofPower = static_cast<uint32_t>(level);
                opts.manual_proofPower = true;
            } else {
                std::cerr << "Error: -proof level must be between 0 and 12. Given: " << level << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            opts.localCarryPropagationDepth = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            opts.backup_interval = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            opts.save_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "-filemers") == 0 && i + 1 < argc) {
            opts.filemers = argv[++i];
            std::string fname = std::filesystem::path(opts.filemers).filename().string();
            size_t pos_pm = fname.find("pm1");
            size_t pos_dot = fname.rfind('.');
            if (pos_pm == std::string::npos || pos_dot == std::string::npos || pos_pm >= pos_dot){
                std::cerr << "Invalid filename format, expected <p>pm<B1>.mers\n";
                std::exit(EXIT_FAILURE);
            }
            std::string p_str  = fname.substr(0, pos_pm);

            uint32_t p  = std::stoul(p_str);
            opts.exportmers = true;
            opts.exponent = p;
        }
        else if (std::strcmp(argv[i], "-b1") == 0 && i + 1 < argc) {
            opts.B1 = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-b1old") == 0 && i + 1 < argc) {
            opts.B1old = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-b2") == 0 && i + 1 < argc) {
            opts.B2 = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if ((std::strcmp(argv[i], "-b2start") == 0 ||
                  std::strcmp(argv[i], "--b2start") == 0 ||
                  std::strcmp(argv[i], "-s2from") == 0 ||
                  std::strcmp(argv[i], "--s2from") == 0 ||
                  std::strcmp(argv[i], "-stage2start") == 0 ||
                  std::strcmp(argv[i], "--stage2start") == 0) && i + 1 < argc) {
            opts.B2Start = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-b3") == 0 && i + 1 < argc) {
            opts.B3 = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-b4") == 0 && i + 1 < argc) {
            opts.B4 = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            opts.K = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-nmax") == 0 && i + 1 < argc) {
            opts.nmax = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-memlim") == 0 && i + 1 < argc) {
            opts.memlim = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            opts.curve_seed = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-sigma") == 0 && i + 1 < argc) {
            opts.sigma = argv[i + 1];
            opts.K = 1;
            ++i;
        }
        else if (std::strcmp(argv[i], "-tbits") == 0 && i + 1 < argc) {
            opts.tbits = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-erroriter") == 0 && i + 1 < argc) {
            opts.erroriter = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-ecm_check_interval") == 0 && i + 1 < argc) {
            opts.ecm_check_interval = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-ecm_progress_ms") == 0 && i + 1 < argc) {
            opts.ecm_progress_interval_ms = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        
        else if (std::strcmp(argv[i], "-llsafeb") == 0 && i + 1 < argc) {
            opts.llsafe_block = std::strtoull(argv[i + 1], nullptr, 10);  // base 10
            ++i;
        }
        else if (std::strcmp(argv[i], "-l1") == 0 && i + 1 < argc) {
            opts.max_local_size1 = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-checklevel") == 0 && i + 1 < argc) {
            opts.checklevel = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-chunk256") == 0 && i + 1 < argc) {
            opts.chunk256 = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-l5") == 0 && i + 1 < argc) {
            opts.max_local_size5 = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-iterforce") == 0 && i + 1 < argc) {
            opts.iterforce = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-iterforce2") == 0 && i + 1 < argc) {
            opts.iterforce2 = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-maxe") == 0 && i + 1 < argc) {
            uint64_t mb = std::strtoull(argv[i + 1], nullptr, 10);
            opts.max_e_bits = (mb == 0 ? 10000ULL : (mb << 23));
            ++i;
        }
        
        else if (std::strcmp(argv[i], "-l2") == 0 && i + 1 < argc) {
            opts.max_local_size2 = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-l3") == 0 && i + 1 < argc) {
            opts.max_local_size3 = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-enqueue_max") == 0 && i + 1 < argc) {
            opts.enqueue_max = to_u64(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-res64_display_interval") == 0 && i + 1 < argc) {
            int v = to_u64(argv[++i]);
            if (v < 0) {
                std::cerr << "Error: -res64_display_interval must be 0 (to disable) or > 0\n";
                std::exit(EXIT_FAILURE);
            }
            opts.res64_display_interval = v;
        }
        else if (std::strcmp(argv[i], "-user") == 0 && i + 1 < argc) {
            opts.user = argv[++i];
        }
        else if (std::strcmp(argv[i], "-password") == 0 && i + 1 < argc) {
            opts.password = argv[++i];
        }
        else if (std::strcmp(argv[i], "-computer") == 0 && i + 1 < argc) {
            opts.computer_name = argv[++i];
        }
        else if (std::strcmp(argv[i], "--noask") == 0 || std::strcmp(argv[i], "-noask") == 0) {
            opts.noAsk = true;
        }
        else if (std::strcmp(argv[i], "-wagstaff") == 0) {
            opts.wagstaff = true;
        }
        else if (std::strcmp(argv[i], "-resume") == 0) {
            opts.resume = true;
        }
        else if (std::strcmp(argv[i], "-p95") == 0) {
            opts.resume95 = true;
        }
        else if (std::strcmp(argv[i], "-noverify") == 0) {
            opts.verify = false;
        }
        else if (std::strcmp(argv[i], "-tune") == 0) {
            opts.tune = true;
        }
        else if (std::strcmp(argv[i], "-worktodo") == 0 && i + 1 < argc) {
            opts.worktodo_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "-config") == 0 && i + 1 < argc) {
            opts.config_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "-kernelpath") == 0 && i + 1 < argc) {
            opts.kernel_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "-gerbiczli") == 0 || std::strcmp(argv[i], "-gerbiczli") == 0) {
            opts.gerbiczli = false;
        }
        else if (std::strcmp(argv[i], "-pm1-lowmem") == 0 ||
                 std::strcmp(argv[i], "--pm1-lowmem") == 0 ||
                 std::strcmp(argv[i], "-pm1lowmem") == 0 ||
                 std::strcmp(argv[i], "-lowmem") == 0) {
            opts.pm1_lowmem = true;
            opts.gerbiczli = false;
        }
        else if (std::strcmp(argv[i], "-pm1-ultralowmem") == 0 ||
                 std::strcmp(argv[i], "--pm1-ultralowmem") == 0 ||
                 std::strcmp(argv[i], "-pm1ultralowmem") == 0 ||
                 std::strcmp(argv[i], "-pm1-1reg") == 0) {
            opts.pm1_lowmem = true;
            opts.pm1_ultralowmem = true;
            opts.gerbiczli = false;
        }
        else if (std::strcmp(argv[i], "-pm1-s2-resume2reg") == 0 ||
                 std::strcmp(argv[i], "--pm1-s2-resume2reg") == 0 ||
                 std::strcmp(argv[i], "-pm1s2resume2reg") == 0 ||
                 std::strcmp(argv[i], "-pm1-stage2-2reg") == 0) {
            opts.pm1_lowmem = true;
            opts.pm1_ultralowmem = true;
            opts.pm1_s2_resume2reg = true;
            opts.gerbiczli = false;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-off") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-off") == 0 ||
                 std::strcmp(argv[i], "-pm1-stage2-classic") == 0 ||
                 std::strcmp(argv[i], "-vtrace-off") == 0) {
            opts.pm1_vtrace_off = true;
            opts.pm1_vtrace = false;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace") == 0 ||
                 std::strcmp(argv[i], "-pm1-stage2-vtrace") == 0 ||
                 std::strcmp(argv[i], "-vtrace") == 0) {
            // Kept for compatibility: V-trace is now the default normal-memory Stage 2.
            opts.pm1_vtrace = true;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-d") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-d") == 0 ||
                  std::strcmp(argv[i], "-vtrace-d") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_D = std::strtoull(argv[i + 1], nullptr, 10);
            ++i;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-auto-d") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-auto-d") == 0 ||
                 std::strcmp(argv[i], "-vtrace-auto-d") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_auto_d = true;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-auto-d-aggressive") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-auto-d-aggressive") == 0 ||
                 std::strcmp(argv[i], "-vtrace-auto-d-aggressive") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_auto_d = true;
            opts.pm1_vtrace_auto_d_aggressive = true;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-deep-d") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-deep-d") == 0 ||
                  std::strcmp(argv[i], "-vtrace-deep-d") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            const char* val = argv[i + 1];
            if (std::strcmp(val, "auto") == 0 || std::strcmp(val, "AUTO") == 0) {
                opts.pm1_vtrace_auto_d = true;
                opts.pm1_vtrace_deep_d_auto = true;
            } else {
                opts.pm1_vtrace_D = std::strtoull(val, nullptr, 10);
            }
            ++i;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-product-tree") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-product-tree") == 0 ||
                 std::strcmp(argv[i], "-vtrace-product-tree") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_product_tree = true;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-product-tree-width") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-product-tree-width") == 0 ||
                  std::strcmp(argv[i], "-vtrace-product-tree-width") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_product_tree = true;
            unsigned long long w = std::strtoull(argv[i + 1], nullptr, 10);
            if (w < 2ULL) w = 2ULL;
            if (w > 64ULL) w = 64ULL;
            opts.pm1_vtrace_product_tree_width = static_cast<uint32_t>(w);
            ++i;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-max-regs") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-max-regs") == 0 ||
                  std::strcmp(argv[i], "-vtrace-max-regs") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_auto_d = true;
            opts.pm1_vtrace_max_regs = std::strtoull(argv[i + 1], nullptr, 10);
            ++i;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-auto-batch") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-auto-batch") == 0 ||
                 std::strcmp(argv[i], "-vtrace-auto-batch") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_auto_batch = true;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-baby-batch") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-baby-batch") == 0 ||
                  std::strcmp(argv[i], "-vtrace-baby-batch") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_baby_batch = std::strtoull(argv[i + 1], nullptr, 10);
            ++i;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-max-batches") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-max-batches") == 0 ||
                  std::strcmp(argv[i], "-vtrace-max-batches") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_max_batches = std::strtoull(argv[i + 1], nullptr, 10);
            if (opts.pm1_vtrace_max_batches == 0) opts.pm1_vtrace_max_batches = 1;
            ++i;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-no-auto-batch") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-no-auto-batch") == 0 ||
                 std::strcmp(argv[i], "-vtrace-no-auto-batch") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_auto_batch = false;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-negadd-off") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-negadd-off") == 0 ||
                 std::strcmp(argv[i], "-vtrace-negadd-off") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_negadd_off = true;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-pair95") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-pair95") == 0 ||
                 std::strcmp(argv[i], "-vtrace-pair95") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_pair95 = true;
            opts.pm1_vtrace_pair95_off = false;
        }
        else if (std::strcmp(argv[i], "-pm1-vtrace-pair95-off") == 0 ||
                 std::strcmp(argv[i], "--pm1-vtrace-pair95-off") == 0 ||
                 std::strcmp(argv[i], "-vtrace-pair95-off") == 0) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_pair95_off = true;
            opts.pm1_vtrace_pair95 = false;
        }
        else if ((std::strcmp(argv[i], "-pm1-vtrace-pair95-l") == 0 ||
                  std::strcmp(argv[i], "--pm1-vtrace-pair95-l") == 0 ||
                  std::strcmp(argv[i], "-vtrace-pair95-l") == 0) && i + 1 < argc) {
            opts.pm1_vtrace = true;
            opts.pm1_vtrace_pair95 = true;
            opts.pm1_vtrace_pair95_L = std::strtoull(argv[i + 1], nullptr, 10);
            ++i;
        }
        else if (std::strcmp(argv[i], "-nogcd-stage1") == 0 ||
                 std::strcmp(argv[i], "--nogcd-stage1") == 0 ||
                 std::strcmp(argv[i], "-no-gcd-stage1") == 0 ||
                 std::strcmp(argv[i], "-nogcdstage1") == 0) {
            opts.pm1_no_stage1_gcd = true;
        }
        else if (strcmp(argv[i], "-factors") == 0 && i + 1 < argc) {
            opts.knownFactors = util::split(argv[++i], ',');
            //opts.knownFactors_start = util::split(argv[++i], ',');
            opts.knownFactors_start.assign(opts.knownFactors.begin(), opts.knownFactors.end());
        }
        
        else if (argv[i][0] != '-') {
            if (opts.exponent == 0) {
                opts.exponent = std::strtoull(argv[i], nullptr, 10);
            } else {
                std::cerr << "Warning: ignoring extra positional argument '"
                        << argv[i] << "'\n";
            }
        }
        else {
            std::cerr << "Warning: Unknown option '" << argv[i] << "'\n";
        }
    }
    if(opts.wagstaff){
        //std::cout << "[WAGSTAFF MODE] This test will check if (2^)" << options.exponent << " + 1)/3 is PRP prime" << std::endl;
        //p  = p*2;
        opts.exponent = 2*opts.exponent;
        opts.mode = "prp";
        opts.gerbiczli = false;
        opts.proof = false;
    }
    if(opts.mode == "ll"){
        opts.erroriter = 0;
    }
    
    // Check that LL test is not used for Mersenne cofactors
    if (opts.mode == "ll" && !opts.knownFactors.empty()) {
        std::cerr << "Error: Lucas-Lehmer test cannot be used on Mersenne cofactors." << std::endl;
        std::cerr << "Use PRP test for Mersenne cofactors instead." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    /*if(opts.mode == "pm1" && opts.B2>0){
        opts.marin = false;
    }*/
    if(opts.iterforce == 0){
        opts.iterforce = 500;
    }
    if(opts.iterforce2 == 0){
        opts.iterforce2 = 10;
    }
    if(opts.enqueue_max == 0){
        opts.enqueue_max = opts.iterforce*64;
    }

/*    if (opts.exponent == 0) {
        std::cerr << "Error: No exponent provided.\n";
        std::exit(EXIT_FAILURE);
    }*/
    constexpr uint64_t MAX_EXPONENT = 5650242869UL;
    if (opts.exponent > MAX_EXPONENT) {
        std::cerr << "Error: Exponent must be <= " << MAX_EXPONENT
                  << ". Given: " << opts.exponent << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (opts.kernel_path.empty()) {
        std::string execDir = util::getExecutableDir();
        std::string kernelFile = execDir + "/prmers.cl";

        if (std::filesystem::exists(kernelFile)) {
            opts.kernel_path = kernelFile;
        } else {
            kernelFile = execDir + "/kernels/prmers.cl";
            if (std::filesystem::exists(kernelFile)) {
                opts.kernel_path = kernelFile;
            } else {
                kernelFile = std::string(KERNEL_PATH) + "prmers.cl";
                if (std::filesystem::exists(kernelFile)) {
                    opts.kernel_path = kernelFile;
                } else {
                    std::cerr << "Error: Cannot find kernel file 'prmers.cl'\n";
                    std::exit(1);
                }
            }
        }
    }

    unsigned int detectedPort = 0;
    #if defined(__APPLE__)
        #if defined(__x86_64__)
            detectedPort = 10; // macOS 64-bit
        #else
            detectedPort = 9;  // macOS 32-bit
        #endif
    #elif defined(__linux__)
        #if defined(__x86_64__)
            detectedPort = 8;  // Linux 64-bit
        #else
            detectedPort = 2;  // Linux 32-bit
        #endif
    #elif defined(_WIN64)
        detectedPort = 4;      // Windows 64-bit
    #elif defined(_WIN32)
        detectedPort = 1;      // Windows 32-bit
    #else
        detectedPort = 14;     // Unix 64-bit (fallback)
    #endif
    opts.portCode = static_cast<int>(detectedPort);

    #if defined(_WIN32)    \
    || defined(__MINGW32__)  \
    || defined(__MINGW64__)  \
    || defined(__CYGWIN__)
        opts.osName = "Windows";
        opts.osVersion = "";
    #elif __APPLE__
        opts.osName = "macOS";
        opts.osVersion = "";
    #else
        opts.osName = "Linux";
        opts.osVersion = "";
    #endif

    #if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    opts.osArch = "x86_64";
    #elif defined(__i386__) || defined(__i386) || defined(i386) || defined(_M_IX86)
    opts.osArch = "x86_32";
    #elif defined(__aarch64__) || defined(_M_ARM64)
    opts.osArch = "arm64";
    #elif defined(_M_ARM64EC)
    opts.osArch = "arm64ec";
    #elif defined(__arm__) || defined(_M_ARM)
    #if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7S__)
        opts.osArch = "armv7";
    #elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)
        opts.osArch = "armv6";
    #else
        opts.osArch = "arm32";
    #endif
    #elif defined(__riscv)
    #if defined(__riscv_xlen) && (__riscv_xlen == 64)
        opts.osArch = "riscv64";
    #else
        opts.osArch = "riscv32";
    #endif
    #elif defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
    #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        opts.osArch = "ppc64le";
    #else
        opts.osArch = "ppc64";
    #endif
    #elif defined(__powerpc__) || defined(__ppc__) || defined(_M_PPC)
    opts.osArch = "ppc32";
    #elif defined(__mips64) || defined(__mips64__) || defined(__MIPS64__)
    #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        opts.osArch = "mips64el";
    #else
        opts.osArch = "mips64";
    #endif
    #elif defined(__mips__) || defined(__mips) || defined(__MIPS__)
    #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        opts.osArch = "mipsel";
    #else
        opts.osArch = "mips";
    #endif
    #elif defined(__s390x__)
    opts.osArch = "s390x";
    #elif defined(__s390__)
    opts.osArch = "s390";
    #elif defined(__wasm64__)
    opts.osArch = "wasm64";
    #elif defined(__wasm32__)
    opts.osArch = "wasm32";
    #else
    opts.osArch = "unknown";
    #endif

    if (opts.p95path.empty()) {
        opts.p95stage2 = false;
    }
    return opts;
}

} // namespace io