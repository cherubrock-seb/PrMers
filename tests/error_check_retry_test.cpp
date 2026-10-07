#include "core/ErrorCheckRetry.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL line " << __LINE__ << ": " #cond << std::endl;    \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static std::string read_file(const std::string& path) {
    std::ifstream in(path);
    CHECK(in.good());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Every error-check failure branch must bound its retries with ErrorCheckRetry.
static void check_driver(const std::string& root, const std::string& rel, size_t branches) {
    const std::string src = read_file(root + "/" + rel);
    const std::string needle = "errcheck_retry.failed()";
    size_t n = 0;
    for (size_t pos = src.find(needle); pos != std::string::npos; pos = src.find(needle, pos + 1)) ++n;
    if (n != branches) {
        std::cerr << "FAIL " << rel << ": expected " << branches << " bounded retry branches, found " << n << std::endl;
        std::exit(1);
    }
    CHECK(src.find("errcheck_retry.passed()") != std::string::npos);
}

int main(int argc, char** argv) {
    // Three failures in a row, each restoring the same verified state, stop the run.
    {
        core::ErrorCheckRetry r;
        CHECK(!r.failed());
        CHECK(!r.failed());
        CHECK(r.failed());
        CHECK(r.streak() == 3);
        const std::string msg = r.reason(42, "hint text");
        CHECK(msg.find("3 times in a row") != std::string::npos);
        CHECK(msg.find("iteration 42") != std::string::npos);
        CHECK(msg.find("hint text") != std::string::npos);
    }
    // A passed check in between moves the verified state, so the streak restarts.
    {
        core::ErrorCheckRetry r;
        CHECK(!r.failed());
        CHECK(!r.failed());
        r.passed();
        CHECK(r.streak() == 0);
        CHECK(!r.failed());
        CHECK(!r.failed());
        CHECK(r.failed());
    }
    // A single injected fault always recovers.
    {
        core::ErrorCheckRetry r;
        CHECK(!r.failed());
        r.passed();
        CHECK(!r.failed());
        r.passed();
    }
    // A custom limit is honoured.
    {
        core::ErrorCheckRetry r(1);
        CHECK(r.failed());
    }

    // The drivers use it: LL-SAFE2, LL-SAFE and the legacy PRP loop.
    if (argc > 1) {
        const std::string root = argv[1];
        check_driver(root, "src/modes/RunLlSafeMarin.cpp", 2);
        check_driver(root, "src/modes/RunPrpOrLl.cpp", 1);
    }

    std::cout << "error check retry test: PASS" << std::endl;
    return 0;
}
