// Host test: CliParser must reject exponents the engines cannot hold.
//
// The Marin/Aevum drivers, the proof managers and the JSON writer keep the
// exponent in 32 bits, so a command-line exponent above 2^32 - 1 used to be
// accepted (the limit was the PrimeNet range, 5650242869) and then truncated:
// 4294967357 = 2^32 + 61 tested M61.  No GPU is used; CliParser::parse exits
// for a rejected exponent and returns for an accepted one.
//
// Usage: cli_exponent_range_test accepted <exponent> [args...]
//        cli_exponent_range_test rejected <exponent> [args...]
// "accepted" parses and prints the exponent; "rejected" is expected to make the
// parser exit with a failure and an error message.

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "io/CliParser.hpp"

#include "opencl/Context.hpp"

// printUsage references this; the test never lists devices, so no OpenCL is needed.
void prmers::ocl::Context::listAllOpenCLDevices() {}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " accepted|rejected <exponent> [args...]\n";
        return 2;
    }
    std::vector<std::string> store{"prmers"};
    for (int i = 2; i < argc; ++i) store.emplace_back(argv[i]);
    std::vector<char*> args;
    for (auto& s : store) args.push_back(s.data());
    args.push_back(nullptr);
    io::CliOptions o = io::CliParser::parse(static_cast<int>(store.size()), args.data());
    // Only reached when the parser accepted the exponent.
    std::cout << "parsed exponent " << o.exponent << std::endl;
    return std::strcmp(argv[1], "accepted") == 0 ? 0 : 3;
}
