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
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "util/PathUtils.hpp"
#include <filesystem>
#include "opencl/Context.hpp"

// Forward-declare the usage function (defined elsewhere, e.g. in your main host file)
extern void printUsage(const char* progName);

namespace io {
// -----------------------------------------------------------------------------
// Usage printing helper (updated with new backup and path options)
// -----------------------------------------------------------------------------
void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <p> [-d <device_id>] [-O <options>] [-c <localCarryPropagationDepth>]" << std::endl;
    std::cout << "              [-profile] [-prp|-ll] [-t <backup_interval>] [-f <path>] [-computer <name>]" << std::endl;
    std::cout << "              [--noask] [-user <username>]" << std::endl;
    std::cout << "              [-enqueue_max <value>] [-worktodo <path>] [-config <path>] [-iterforce] [-res64_display_interval] " << std::endl;
    std::cout << "               " << std::endl;
    std::cout << std::endl;
    std::cout << "  <p>       : Exponent to test (required unless -worktodo is used)" << std::endl;
    std::cout << "  -d <device_id>       : (Optional) Specify OpenCL device ID (default: 0)" << std::endl;
    std::cout << "  -O <options>         : (Optional) Enable OpenCL optimization flags (e.g., fastmath, mad, unsafe, nans, optdisable)" << std::endl;
    std::cout << "  -c <depth>           : (Optional) Set local carry propagation depth (default: 8)" << std::endl;
    std::cout << "  -profile             : (Optional) Enable kernel execution profiling" << std::endl;
    std::cout << "  -prp                 : (Optional) Run in PRP mode (default). Uses initial value 3; final result must equal 9" << std::endl;
    std::cout << "  -ll                  : (Optional) Run in Lucas-Lehmer mode. Uses initial value 4 and p-2 iterations" << std::endl;
    std::cout << "  -t <seconds>         : (Optional) Specify backup interval in seconds (default: 120)" << std::endl;
    std::cout << "  -f <path>            : (Optional) Specify path for saving/loading checkpoint files (default: current directory)" << std::endl;
    std::cout << "  -l1 <value>          : (Optional) Force local size max for NTT kernels" << std::endl;
    //std::cout << "  -l2 <value>          : (Optional) Force local size for 2-step radix-16 NTT kernel" << std::endl;
    //std::cout << "  -l3 <value>          : (Optional) Force local size for mixed radix NTT kernel" << std::endl;
    std::cout << "  --noask              : (Optional) Automatically send results to PrimeNet without prompting" << std::endl;
    std::cout << "  -user <username>     : (Optional) PrimeNet username to auto-fill during result submission" << std::endl;
    std::cout << "  -password <password> : (Optional) PrimeNet password to autosubmit the result without prompt (used only when -no-ask is set)" << std::endl;
    std::cout << "  -computer <name>     : (Optional) PrimeNet computer name to auto-fill the result submission" << std::endl;
    std::cout << "  -worktodo <path>     : (Optional) Load exponent from specified worktodo.txt (default: ./worktodo.txt)" << std::endl;
    std::cout << "  -config <path>       : (Optional) Load config file from specified path" << std::endl;
    //std::cout << "  -proof               : (Optional) Disable proof generation (by default a proof is created if PRP test passes)" << std::endl;
    std::cout << "  -iterforce <iter>    : (Optional) force a display every <iter>" << std::endl;
    std::cout << "  -res64_display_interval <N> : (Optional) Display Res64 every N iterations (0 = disabled or > 0, default = 100000)" << std::endl;
    //std::cout << "  -throttle_low        : (Optional) Enable CL_QUEUE_THROTTLE_LOW_KHR if OpenCL >= 2.2 (default: disabled)" << std::endl;
    std::cout << "  -tune               : (Optional) Automatically determine the best pacing (iterForce) and how often to call clFinish() to synchronize kernels (default: disabled)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:\n  " << progName << " 127 -O fastmath mad -c 16 -profile -ll -t 120 -f /my/backup/path \\\n"
              << "            -l1 256 -l2 128 -l3 64 --noask -user myaccountname -enqueue_max 65536 \\\n"
              << "            -worktodo ./mydir/worktodo.txt -config ./mydir/settings.cfg -proof" << std::endl;
    opencl::Context::listAllOpenCLDevices();
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
            std::cout << "prmers Release v4.0.51-alpha\n";
            std::exit(EXIT_SUCCESS);
        }
    }
    CliOptions opts;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            opts.device_id = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-prp") == 0) {
            opts.mode = "prp";
        }
        else if (std::strcmp(argv[i], "-O") == 0 && i + 1 < argc) {
            opts.build_options.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string opt = argv[++i];
                if (opt == "fastmath")     opts.build_options += " -cl-fast-relaxed-math";
                else if (opt == "mad")     opts.build_options += " -cl-mad-enable";
                else if (opt == "unsafe")  opts.build_options += " -cl-unsafe-math-optimizations";
                else if (opt == "nans")    opts.build_options += " -cl-no-signed-zeros";
                else if (opt == "optdisable") opts.build_options += " -cl-opt-disable";
                else {
                    std::cerr << "Warning: unrecognized optimization flag '" << opt << "'\n";
                 }

            }
        }
        else if (std::strcmp(argv[i], "-ll") == 0) {
            opts.mode = "ll";
            opts.proof = false;
        }
        else if (std::strcmp(argv[i], "-profile") == 0) {
            opts.profiling = true;
        }
        else if (std::strcmp(argv[i], "-debug") == 0) {
            opts.debug = false;
        }
        
        else if (std::strcmp(argv[i], "-throttle_low") == 0) {
            opts.cl_queue_throttle_active = true;
        }
        else if (std::strcmp(argv[i], "-proof") == 0) {
            opts.proof = false;
        }
        else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            opts.localCarryPropagationDepth = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            opts.backup_interval = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            opts.save_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "-l1") == 0 && i + 1 < argc) {
            opts.max_local_size1 = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-iterforce") == 0 && i + 1 < argc) {
            opts.iterforce = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-l2") == 0 && i + 1 < argc) {
            opts.max_local_size2 = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-l3") == 0 && i + 1 < argc) {
            opts.max_local_size3 = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-enqueue_max") == 0 && i + 1 < argc) {
            opts.enqueue_max = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-res64_display_interval") == 0 && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
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
            opts.gerbiczli = true;
        }
        else if (argv[i][0] != '-') {
            if (opts.exponent == 0) {
                opts.exponent = static_cast<uint32_t>(std::atoi(argv[i]));
            } else {
                std::cerr << "Warning: ignoring extra positional argument '"
                        << argv[i] << "'\n";
            }
        }
        else {
            std::cerr << "Warning: Unknown option '" << argv[i] << "'\n";
        }
    }
    if(opts.iterforce == 0){
        opts.iterforce = 500;
    }
    if(opts.enqueue_max == 0){
        opts.enqueue_max = opts.iterforce*64;
    }

/*    if (opts.exponent == 0) {
        std::cerr << "Error: No exponent provided.\n";
        std::exit(EXIT_FAILURE);
    }*/
    constexpr uint32_t MAX_EXPONENT = 1207959503u;
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

    #if defined(__x86_64__)
        opts.osArch = "x86_64";
    #elif defined(__aarch64__)
        opts.osArch = "arm64";
    #else
        opts.osArch = "unknown";
    #endif
    return opts;
}

} // namespace io