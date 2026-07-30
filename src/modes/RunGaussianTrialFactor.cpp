#include "modes/GaussianTrialFactor.hpp"

#include "core/Version.hpp"
#include "opencl/Context.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace core {
namespace {

constexpr std::uint32_t FAMILY_GM = 1U;
constexpr std::uint32_t FAMILY_GQ = 2U;
constexpr std::uint32_t FAMILY_BOTH = FAMILY_GM | FAMILY_GQ;
constexpr std::size_t MAX_FACTORS = 64;

struct TfRequest {
    std::uint64_t exponent = 0;
    unsigned fromBits = 0;
    unsigned toBits = 0;
    std::uint32_t familyBits = FAMILY_BOTH;
    std::uint64_t chunkSpan = 4'194'304ULL;
    std::uint32_t sievePrime = 65'536U;
    int device = 0;
    fs::path outputDirectory = ".";
    fs::path worktodoPath = "worktodo.txt";
    std::string rawWorktodoLine;
};

struct DeviceFactor {
    cl_ulong factor;
    cl_uint familyBits;
    cl_uint reserved;
};

struct FoundFactor {
    std::uint64_t factor = 0;
    std::uint32_t familyBits = 0;
};

void checkCl(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with OpenCL error " +
                                 std::to_string(error));
    }
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

std::vector<std::string> splitCsv(const std::string& value) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) parts.push_back(trim(part));
    return parts;
}

std::uint64_t parseU64(const std::string& text, const char* name) {
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(text, &consumed, 10);
    } catch (...) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::uint32_t parseFamily(const std::string& text) {
    const std::string family = upper(trim(text));
    if (family == "GM") return FAMILY_GM;
    if (family == "GQ") return FAMILY_GQ;
    if (family == "BOTH" || family == "PAIR" || family == "GM+GQ") return FAMILY_BOTH;
    throw std::runtime_error("Gaussian TF family must be GM, GQ, or BOTH");
}

std::string familyName(std::uint32_t bits) {
    if (bits == FAMILY_GM) return "GM";
    if (bits == FAMILY_GQ) return "GQ";
    return "BOTH";
}

std::vector<std::string> effectiveArguments(int argc, char** argv) {
    std::vector<std::string> output;
    output.emplace_back(argc > 0 && argv[0] ? argv[0] : "prmers");
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i] ? argv[i] : "";
        if (argument == "-config" && i + 1 < argc) {
            std::ifstream config(argv[++i]);
            if (!config) throw std::runtime_error("Unable to open PrMers config file");
            std::string line;
            while (std::getline(config, line)) {
                const auto comment = line.find('#');
                if (comment != std::string::npos) line.resize(comment);
                std::istringstream tokens(line);
                std::string token;
                while (tokens >> token) output.push_back(token);
            }
        } else {
            output.push_back(argument);
        }
    }
    return output;
}

std::optional<TfRequest> parseDirectRequest(const std::vector<std::string>& args) {
    TfRequest request;
    bool selected = false;
    bool exponentSet = false;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "-gm-tf" || arg == "--gm-tf") {
            if (i + 2 >= args.size()) {
                throw std::runtime_error("-gm-tf requires FROM_BITS and TO_BITS");
            }
            request.fromBits = static_cast<unsigned>(parseU64(args[++i], "TF lower bit"));
            request.toBits = static_cast<unsigned>(parseU64(args[++i], "TF upper bit"));
            selected = true;
        } else if ((arg == "-gm-family" || arg == "--gm-family") && i + 1 < args.size()) {
            request.familyBits = parseFamily(args[++i]);
        } else if ((arg == "-gm-tf-chunk" || arg == "--gm-tf-chunk") && i + 1 < args.size()) {
            request.chunkSpan = parseU64(args[++i], "TF chunk span");
        } else if ((arg == "-gm-tf-sieve" || arg == "--gm-tf-sieve") && i + 1 < args.size()) {
            request.sievePrime = static_cast<std::uint32_t>(parseU64(args[++i], "TF sieve prime"));
        } else if (arg == "-d" && i + 1 < args.size()) {
            request.device = static_cast<int>(parseU64(args[++i], "device"));
        } else if (arg == "-f" && i + 1 < args.size()) {
            request.outputDirectory = args[++i];
        } else if (arg == "-worktodo" && i + 1 < args.size()) {
            request.worktodoPath = args[++i];
        } else if (!arg.empty() && arg[0] != '-' && !exponentSet) {
            request.exponent = parseU64(arg, "Gaussian exponent");
            exponentSet = true;
        }
    }

    if (!selected) return std::nullopt;
    if (!exponentSet) throw std::runtime_error("Gaussian TF requires the exponent p");
    return request;
}

bool hasExplicitNonTfWork(const std::vector<std::string>& args) {
    static const std::unordered_set<std::string> modes{
        "-gm", "--gm", "-gm-proth", "--gm-proth",
        "-gm-prp", "--gm-prp", "-gm-pm1", "--gm-pm1",
        "-gm-ecm", "--gm-ecm", "-prp", "-ll", "-llunsafe",
        "-llsafe2", "-pm1", "-ecm", "-bench", "-memtest",
        "-v", "--version", "-version", "-h", "--help", "-help"
    };

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (modes.contains(arg)) return true;

        if (arg == "-d" || arg == "-f" || arg == "-worktodo" ||
            arg == "-gm-family" || arg == "--gm-family" ||
            arg == "-gm-tf-chunk" || arg == "--gm-tf-chunk" ||
            arg == "-gm-tf-sieve" || arg == "--gm-tf-sieve") {
            if (i + 1 < args.size()) ++i;
            continue;
        }

        if (!arg.empty() && arg[0] != '-') return true;
    }
    return false;
}

std::optional<TfRequest> parseWorktodoRequest(const std::vector<std::string>& args) {
    TfRequest defaults;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "-worktodo" && i + 1 < args.size()) defaults.worktodoPath = args[++i];
        else if (args[i] == "-d" && i + 1 < args.size()) defaults.device = static_cast<int>(parseU64(args[++i], "device"));
        else if (args[i] == "-f" && i + 1 < args.size()) defaults.outputDirectory = args[++i];
    }

    std::ifstream input(defaults.worktodoPath);
    if (!input) return std::nullopt;
    std::string line;
    while (std::getline(input, line)) {
        const std::string clean = trim(line);
        if (clean.empty() || clean[0] == '#' || clean[0] == ';') continue;
        if (upper(clean.substr(0, std::min<std::size_t>(5, clean.size()))) != "GMTF=") continue;
        const auto parts = splitCsv(clean.substr(5));
        if (parts.size() < 3 || parts.size() > 6) {
            throw std::runtime_error(
                "GMTF format is GMTF=p,from_bits,to_bits[,GM|GQ|BOTH[,chunk_span[,sieve_prime]]]");
        }
        TfRequest request = defaults;
        request.exponent = parseU64(parts[0], "Gaussian exponent");
        request.fromBits = static_cast<unsigned>(parseU64(parts[1], "TF lower bit"));
        request.toBits = static_cast<unsigned>(parseU64(parts[2], "TF upper bit"));
        if (parts.size() >= 4 && !parts[3].empty()) request.familyBits = parseFamily(parts[3]);
        if (parts.size() >= 5 && !parts[4].empty()) request.chunkSpan = parseU64(parts[4], "TF chunk span");
        if (parts.size() >= 6 && !parts[5].empty()) {
            request.sievePrime = static_cast<std::uint32_t>(parseU64(parts[5], "TF sieve prime"));
        }
        request.rawWorktodoLine = clean;
        return request;
    }
    return std::nullopt;
}

void validateRequest(const TfRequest& request) {
    if (request.exponent < 3 || (request.exponent & 1ULL) == 0ULL) {
        throw std::runtime_error("Gaussian TF requires an odd exponent p >= 3");
    }
    if (request.exponent > std::numeric_limits<std::uint32_t>::max() / 4ULL) {
        throw std::runtime_error("Gaussian TF requires 4p <= 2^32-1");
    }
    if (request.fromBits < 8 || request.fromBits >= 64 ||
        request.toBits <= request.fromBits || request.toBits > 64) {
        throw std::runtime_error("Gaussian TF requires 8 <= FROM_BITS < TO_BITS <= 64");
    }
    if (request.chunkSpan < 1024 || request.chunkSpan > 268'435'456ULL) {
        throw std::runtime_error("Gaussian TF chunk span must be between 1024 and 268435456");
    }
    if (request.sievePrime < 97 || request.sievePrime > 2'000'000U) {
        throw std::runtime_error("Gaussian TF sieve prime must be between 97 and 2000000");
    }
}

std::vector<std::uint32_t> smallPrimes(std::uint32_t limit) {
    std::vector<bool> composite(static_cast<std::size_t>(limit) + 1, false);
    std::vector<std::uint32_t> primes;
    for (std::uint32_t value = 2; value <= limit; ++value) {
        if (composite[value]) continue;
        primes.push_back(value);
        if (static_cast<std::uint64_t>(value) * value <= limit) {
            for (std::uint64_t multiple = static_cast<std::uint64_t>(value) * value;
                 multiple <= limit; multiple += value) {
                composite[static_cast<std::size_t>(multiple)] = true;
            }
        }
    }
    return primes;
}

std::uint64_t modPowSmall(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    std::uint64_t result = 1;
    base %= modulus;
    while (exponent != 0) {
        if (exponent & 1ULL) result = (result * base) % modulus;
        exponent >>= 1ULL;
        if (exponent != 0) base = (base * base) % modulus;
    }
    return result;
}

std::vector<std::uint64_t> sieveKRange(
    std::uint64_t begin,
    std::uint64_t end,
    std::uint64_t step,
    const std::vector<std::uint32_t>& primes)
{
    const std::uint64_t size64 = end - begin + 1;
    if (size64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("TF sieve chunk is too large for this host");
    }
    const std::size_t size = static_cast<std::size_t>(size64);
    std::vector<std::uint8_t> composite(size, 0);

    for (const std::uint32_t prime : primes) {
        if (prime == 2) continue;
        const std::uint64_t stepMod = step % prime;
        if (stepMod == 0) continue;
        const std::uint64_t inverse = modPowSmall(stepMod, prime - 2ULL, prime);
        const std::uint64_t bad = ((prime - 1ULL) * inverse) % prime;
        const std::uint64_t beginMod = begin % prime;
        std::uint64_t offset = bad >= beginMod ? bad - beginMod : bad + prime - beginMod;
        for (std::uint64_t index = offset; index < size64; index += prime) {
            const std::uint64_t k = begin + index;
            const std::uint64_t q = step * k + 1ULL;
            if (q != prime) composite[static_cast<std::size_t>(index)] = 1;
        }
    }

    std::vector<std::uint64_t> candidates;
    candidates.reserve(size / 8 + 16);
    for (std::size_t index = 0; index < size; ++index) {
        if (!composite[index]) candidates.push_back(begin + static_cast<std::uint64_t>(index));
    }
    return candidates;
}

std::string readTextFile(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open OpenCL kernel: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

fs::path kernelPath() {
    if (const char* environment = std::getenv("PRMERS_KERNEL_PATH")) {
        fs::path path(environment);
        if (fs::is_directory(path)) path /= "gm_trial_factor.cl";
        if (fs::exists(path)) return path;
    }
    const fs::path local = fs::path("kernels") / "gm_trial_factor.cl";
    if (fs::exists(local)) return local;
#ifdef KERNEL_PATH
    const fs::path installed = fs::path(KERNEL_PATH) / "gm_trial_factor.cl";
    if (fs::exists(installed)) return installed;
#endif
    throw std::runtime_error("gm_trial_factor.cl was not found; run from the PrMers directory or set PRMERS_KERNEL_PATH");
}

std::string deviceName(cl_device_id device) {
    std::size_t size = 0;
    checkCl(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size), "clGetDeviceInfo(size)");
    std::string name(size, '\0');
    checkCl(clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr), "clGetDeviceInfo(name)");
    while (!name.empty() && name.back() == '\0') name.pop_back();
    return name;
}

std::string isoUtcNow() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(c) << std::dec;
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

std::uint64_t lowerForBits(unsigned bits) {
    return std::uint64_t{1} << bits;
}

std::uint64_t upperForBits(unsigned bits) {
    if (bits == 64) return std::numeric_limits<std::uint64_t>::max();
    return (std::uint64_t{1} << bits) - 1ULL;
}

std::uint64_t ceilDiv(std::uint64_t numerator, std::uint64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0 ? 1ULL : 0ULL);
}

fs::path checkpointPath(const TfRequest& request) {
    std::ostringstream name;
    name << "gm_tf_p" << request.exponent << '_' << request.fromBits << '_'
         << request.toBits << '_' << familyName(request.familyBits) << ".checkpoint";
    return request.outputDirectory / name.str();
}

fs::path resultPath(const TfRequest& request) {
    std::ostringstream name;
    name << "gm_tf_p" << request.exponent << '_' << request.fromBits << '_'
         << request.toBits << '_' << familyName(request.familyBits) << "_result.json";
    return request.outputDirectory / name.str();
}

void saveCheckpoint(const fs::path& path, std::uint64_t nextK) {
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) throw std::runtime_error("Unable to write TF checkpoint");
        output << nextK << '\n';
    }

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        // std::filesystem::rename cannot replace an existing destination on
        // Windows. Keep the POSIX atomic fast path, then use a portable
        // replace fallback for subsequent checkpoints on Windows.
        std::error_code ignored;
        fs::remove(path, ignored);
        error.clear();
        fs::rename(temporary, path, error);
        if (error) {
            throw std::runtime_error(
                "Unable to replace TF checkpoint: " + error.message());
        }
    }
}

std::optional<std::uint64_t> loadCheckpoint(const fs::path& path) {
    std::ifstream input(path);
    std::uint64_t value = 0;
    if (input >> value) return value;
    return std::nullopt;
}

bool hasFamily(const std::vector<FoundFactor>& factors, std::uint32_t familyBit) {
    return std::any_of(factors.begin(), factors.end(), [familyBit](const FoundFactor& factor) {
        return (factor.familyBits & familyBit) != 0;
    });
}

bool targetSatisfied(const TfRequest& request, const std::vector<FoundFactor>& factors) {
    if (request.familyBits == FAMILY_GM) return hasFamily(factors, FAMILY_GM);
    if (request.familyBits == FAMILY_GQ) return hasFamily(factors, FAMILY_GQ);
    return hasFamily(factors, FAMILY_GM) && hasFamily(factors, FAMILY_GQ);
}

void addFound(std::vector<FoundFactor>& factors, const DeviceFactor& value) {
    for (const std::uint32_t bit : {FAMILY_GM, FAMILY_GQ}) {
        if ((value.familyBits & bit) == 0) continue;
        const bool exists = std::any_of(factors.begin(), factors.end(), [&](const FoundFactor& existing) {
            return existing.factor == value.factor && existing.familyBits == bit;
        });
        if (!exists) factors.push_back({static_cast<std::uint64_t>(value.factor), bit});
    }
}

void writeResult(
    const TfRequest& request,
    const std::vector<FoundFactor>& factors,
    const std::string& device,
    double elapsed,
    std::uint64_t testedCandidates,
    std::uint64_t completedK)
{
    fs::create_directories(request.outputDirectory);
    const fs::path path = resultPath(request);
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("Unable to write Gaussian TF JSON result");

    const bool found = !factors.empty();
    output << "{\n"
           << "  \"schema_version\": 2,\n"
           << "  \"program\": \"PrMers\",\n"
           << "  \"program_version\": \"" << jsonEscape(PRMERS_VERSION) << "\",\n"
           << "  \"family\": \"gaussian-pair\",\n"
           << "  \"mode\": \"gm-tf\",\n"
           << "  \"outcome\": \"" << (found ? "factor" : "no-factor") << "\",\n"
           << "  \"stage\": null,\n"
           << "  \"exponent\": " << request.exponent << ",\n"
           << "  \"target_family\": \"" << familyName(request.familyBits) << "\",\n"
           << "  \"tf_from_bits\": " << request.fromBits << ",\n"
           << "  \"tf_to_bits\": " << request.toBits << ",\n"
           << "  \"tf_sieve_prime\": " << request.sievePrime << ",\n"
           << "  \"tf_chunk_candidates\": " << request.chunkSpan << ",\n"
           << "  \"tested_candidates\": " << testedCandidates << ",\n"
           << "  \"completed_k\": \"" << completedK << "\",\n"
           << "  \"B1\": null,\n"
           << "  \"B2\": null,\n"
           << "  \"base\": null,\n"
           << "  \"curves\": null,\n"
           << "  \"sigma\": null,\n"
           << "  \"sieve_limit\": null,\n"
           << "  \"chunk_bits\": null,\n";

    if (found) output << "  \"factor\": \"" << factors.front().factor << "\",\n";
    else output << "  \"factor\": null,\n";

    output << "  \"factors\": [";
    for (std::size_t index = 0; index < factors.size(); ++index) {
        if (index != 0) output << ',';
        output << "\n    {\"family\": \"" << familyName(factors[index].familyBits)
               << "\", \"factor\": \"" << factors[index].factor << "\"}";
    }
    if (!factors.empty()) output << '\n' << "  ";
    output << "],\n"
           << "  \"backend\": \"OpenCL-GPU-TF\",\n"
           << "  \"device\": \"" << jsonEscape(device) << "\",\n"
           << "  \"elapsed_seconds\": " << std::fixed << std::setprecision(6) << elapsed << ",\n"
           << "  \"residue\": null,\n"
           << "  \"proof_sha256\": null,\n"
           << "  \"timestamp\": \"" << isoUtcNow() << "\"\n"
           << "}\n";

    std::cout << "Gaussian TF result: " << path << '\n';
}

int runTrialFactor(const TfRequest& request) {
    validateRequest(request);
    fs::create_directories(request.outputDirectory);

    const std::uint64_t step = 4ULL * request.exponent;
    const std::uint64_t lower = lowerForBits(request.fromBits);
    const std::uint64_t upper = upperForBits(request.toBits);
    const std::uint64_t firstK = std::max<std::uint64_t>(1, ceilDiv(lower - 1ULL, step));
    const std::uint64_t lastK = (upper - 1ULL) / step;
    if (firstK > lastK) throw std::runtime_error("The requested bit interval contains no q=4kp+1 candidates");

    const fs::path checkpoint = checkpointPath(request);
    std::uint64_t nextK = firstK;
    if (const auto saved = loadCheckpoint(checkpoint); saved && *saved >= firstK && *saved <= lastK + 1ULL) {
        nextK = *saved;
        std::cout << "Resuming Gaussian TF at k=" << nextK << '\n';
    }

    prmers::ocl::Context context(request.device, 0, false, false);
    const std::string source = readTextFile(kernelPath());
    const char* sourcePointer = source.c_str();
    const std::size_t sourceSize = source.size();
    cl_int error = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        context.getContext(), 1, &sourcePointer, &sourceSize, &error);
    checkCl(error, "clCreateProgramWithSource");

    const cl_device_id buildDevice = context.getDevice();
    error = clBuildProgram(program, 1, &buildDevice, "-cl-std=CL1.2", nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::size_t logSize = 0;
        clGetProgramBuildInfo(program, context.getDevice(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, context.getDevice(), CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("Gaussian TF OpenCL build failed:\n" + log);
    }

    cl_kernel kernel = clCreateKernel(program, "gm_trial_factor", &error);
    checkCl(error, "clCreateKernel(gm_trial_factor)");

    const std::string gpuName = deviceName(context.getDevice());
    const auto primes = smallPrimes(request.sievePrime);
    std::vector<FoundFactor> found;
    std::uint64_t testedCandidates = 0;
    const auto started = std::chrono::steady_clock::now();

    std::cout << "Gaussian pair GPU trial factoring\n"
              << "  p              : " << request.exponent << '\n'
              << "  family         : " << familyName(request.familyBits) << '\n'
              << "  q bits         : [" << request.fromBits << ", " << request.toBits << ")\n"
              << "  q form         : 4*k*p + 1\n"
              << "  GPU            : " << gpuName << '\n'
              << "  sieve primes   : <= " << request.sievePrime << '\n'
              << "  raw k chunk    : " << request.chunkSpan << '\n';

    while (nextK <= lastK && !targetSatisfied(request, found)) {
        const std::uint64_t remaining = lastK - nextK + 1ULL;
        const std::uint64_t span = std::min(request.chunkSpan, remaining);
        const std::uint64_t chunkEnd = nextK + span - 1ULL;
        auto candidates = sieveKRange(nextK, chunkEnd, step, primes);

        if (!candidates.empty()) {
            cl_mem candidatesBuffer = clCreateBuffer(
                context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                candidates.size() * sizeof(std::uint64_t), candidates.data(), &error);
            checkCl(error, "clCreateBuffer(candidates)");

            cl_uint zero = 0;
            cl_mem countBuffer = clCreateBuffer(
                context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                sizeof(zero), &zero, &error);
            checkCl(error, "clCreateBuffer(factor_count)");

            std::array<DeviceFactor, MAX_FACTORS> deviceFactors{};
            cl_mem factorsBuffer = clCreateBuffer(
                context.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                sizeof(deviceFactors), deviceFactors.data(), &error);
            checkCl(error, "clCreateBuffer(factors)");

            const cl_ulong candidateCount = static_cast<cl_ulong>(candidates.size());
            const cl_ulong stepArg = static_cast<cl_ulong>(step);
            const cl_ulong middle = static_cast<cl_ulong>((request.exponent + 1ULL) / 2ULL);
            const std::uint64_t residue = request.exponent & 7ULL;
            const cl_int epsilon = (residue == 1ULL || residue == 7ULL) ? 1 : -1;
            const cl_uint family = request.familyBits;

            checkCl(clSetKernelArg(kernel, 0, sizeof(candidatesBuffer), &candidatesBuffer), "clSetKernelArg(0)");
            checkCl(clSetKernelArg(kernel, 1, sizeof(candidateCount), &candidateCount), "clSetKernelArg(1)");
            checkCl(clSetKernelArg(kernel, 2, sizeof(stepArg), &stepArg), "clSetKernelArg(2)");
            checkCl(clSetKernelArg(kernel, 3, sizeof(middle), &middle), "clSetKernelArg(3)");
            checkCl(clSetKernelArg(kernel, 4, sizeof(epsilon), &epsilon), "clSetKernelArg(4)");
            checkCl(clSetKernelArg(kernel, 5, sizeof(family), &family), "clSetKernelArg(5)");
            checkCl(clSetKernelArg(kernel, 6, sizeof(countBuffer), &countBuffer), "clSetKernelArg(6)");
            checkCl(clSetKernelArg(kernel, 7, sizeof(factorsBuffer), &factorsBuffer), "clSetKernelArg(7)");

            std::size_t local = std::min<std::size_t>(256, context.getMaxWorkGroupSize());
            if (local == 0) local = 1;
            const std::size_t global = ((candidates.size() + local - 1) / local) * local;
            checkCl(clEnqueueNDRangeKernel(
                context.getQueue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(gm_trial_factor)");
            checkCl(clFinish(context.getQueue()), "clFinish(gm_trial_factor)");

            cl_uint count = 0;
            checkCl(clEnqueueReadBuffer(
                context.getQueue(), countBuffer, CL_TRUE, 0, sizeof(count), &count, 0, nullptr, nullptr),
                "clEnqueueReadBuffer(factor_count)");
            const std::size_t returned = std::min<std::size_t>(count, MAX_FACTORS);
            if (returned != 0) {
                checkCl(clEnqueueReadBuffer(
                    context.getQueue(), factorsBuffer, CL_TRUE, 0,
                    returned * sizeof(DeviceFactor), deviceFactors.data(), 0, nullptr, nullptr),
                    "clEnqueueReadBuffer(factors)");
                for (std::size_t index = 0; index < returned; ++index) {
                    addFound(found, deviceFactors[index]);
                    std::cout << "  factor found   : " << deviceFactors[index].factor
                              << " (" << familyName(deviceFactors[index].familyBits) << ")\n";
                }
            }

            testedCandidates += static_cast<std::uint64_t>(candidates.size());
            clReleaseMemObject(factorsBuffer);
            clReleaseMemObject(countBuffer);
            clReleaseMemObject(candidatesBuffer);
        }

        nextK = chunkEnd + 1ULL;
        saveCheckpoint(checkpoint, nextK);
        const long double completed = static_cast<long double>(nextK - firstK);
        const long double total = static_cast<long double>(lastK - firstK + 1ULL);
        std::cout << "  progress       : " << std::fixed << std::setprecision(2)
                  << static_cast<double>(100.0L * completed / total) << "%"
                  << " | tested after sieve: " << testedCandidates << '\n';
    }

    const auto finished = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(finished - started).count();
    writeResult(request, found, gpuName, elapsed, testedCandidates,
                nextK == 0 ? lastK : std::min<std::uint64_t>(nextK, lastK + std::uint64_t{1}));
    std::error_code ignored;
    fs::remove(checkpoint, ignored);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return 0;
}

} // namespace

std::optional<int> tryRunGaussianTrialFactor(int argc, char** argv) {
    const auto args = effectiveArguments(argc, argv);
    if (auto direct = parseDirectRequest(args)) return runTrialFactor(*direct);
    if (hasExplicitNonTfWork(args)) return std::nullopt;
    if (auto worktodo = parseWorktodoRequest(args)) return runTrialFactor(*worktodo);
    return std::nullopt;
}

} // namespace core
