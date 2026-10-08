// Input validation helpers of the CLI: the -filemers file name and the interactive exponent prompt.
//
// - "<p>pm<B1>.mers": the CLI used to require the name to contain "pm1" (so 127pm5000.mers was refused
//   while 127pm1000.mers passed by accident), and a non-numeric prefix ("abcpm1.mers") reached
//   std::stoul, which threw out of CliParser::parse.
// - The interactive prompt took any int: a negative answer wrapped to an enormous unsigned exponent
//   and bypassed the MAX_EXPONENT check, and 0 or 1 were accepted.

#include "io/ExponentInput.hpp"
#include "io/MersFileName.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

void expectName(const std::string& name, bool valid, uint32_t p = 0, uint64_t b1 = 0) {
    uint32_t gotP = 0;
    uint64_t gotB1 = 0;
    const bool ok = io::parseMersFileName(name, gotP, gotB1);
    check(ok == valid, "file name '" + name + "' " + (valid ? "must be accepted" : "must be rejected"));
    if (ok && valid) check(gotP == p && gotB1 == b1, "file name '" + name + "' parsed to the wrong p/B1");
}

void expectAnswer(const std::string& text, bool valid, uint64_t exponent = 0) {
    uint64_t got = 0;
    const bool ok = io::parseExponentAnswer(text, got);
    check(ok == valid, "answer '" + text + "' " + (valid ? "must be accepted" : "must be rejected"));
    if (ok && valid) check(got == exponent, "answer '" + text + "' parsed to the wrong exponent");
}

} // namespace

int main() {
    expectName("127pm5000.mers", true, 127, 5000);
    expectName("127pm1000.mers", true, 127, 1000);
    expectName("1000003pm1000000.mers", true, 1000003, 1000000);
    expectName("4294967295pm1.mers", true, 4294967295U, 1);
    expectName("abcpm1.mers", false);
    expectName("pm1000.mers", false);
    expectName("127pm.mers", false);
    expectName("127pmabc.mers", false);
    expectName("127pm5000", false);
    expectName("127.mers", false);
    expectName("0pm1000.mers", false);
    expectName("4294967296pm1000.mers", false);
    expectName("99999999999pm1000.mers", false);
    expectName("127pm5000.1.mers", false);

    expectAnswer("21701", true, 21701);
    expectAnswer("  127\t", true, 127);
    expectAnswer("2", true, 2);
    expectAnswer("5650242869", true, 5650242869ULL);
    expectAnswer("-5", false);
    expectAnswer("0", false);
    expectAnswer("1", false);
    expectAnswer("", false);
    expectAnswer("abc", false);
    expectAnswer("12x", false);
    expectAnswer("5650242870", false);
    expectAnswer("99999999999", false);
    expectAnswer("99999999999999999999", false);

    // App runs a prompt answer through the same limit check as the command line (and doubles it
    // first for -wagstaff): the prompt accepts up to kMaxExponent, but the engines hold the
    // exponent in 32 bits, so anything above 2^32 - 1 must still be refused.
    auto promptLimitError = [](const std::string& text, bool wagstaff) {
        uint64_t got = 0;
        if (!io::parseExponentAnswer(text, got)) return std::string("unparsed");
        return io::exponentLimitError(wagstaff ? 2 * got : got, wagstaff);
    };
    check(promptLimitError("4294967295", false).empty(), "prompt answer 2^32 - 1 is accepted");
    check(promptLimitError("4294967296", false).find("<= 4294967295") != std::string::npos,
          "prompt answer 2^32 is refused with the engine limit");
    check(promptLimitError("5650242869", false).find("<= 4294967295") != std::string::npos,
          "prompt answer kMaxExponent is refused with the engine limit");
    check(promptLimitError("2147483647", true).empty(), "Wagstaff prompt answer 2^31 - 1 is accepted");
    check(promptLimitError("2147483648", true).find("twice the requested Wagstaff exponent") != std::string::npos,
          "Wagstaff prompt answer 2^31 is refused");
    check(promptLimitError("5650242869", true).find("<= 5650242869") != std::string::npos,
          "Wagstaff prompt answer kMaxExponent is refused with the global limit");
    check(io::exponentLimitError(io::kMaxExponent + 1).find("<= 5650242869. Given: 5650242870") != std::string::npos,
          "global limit message is unchanged");
    check(io::exponentLimitError(0).empty() && io::exponentLimitError(2).empty(), "small exponents pass the limit check");

    if (failures != 0) {
        std::cerr << failures << " CLI input validation check(s) failed\n";
        return 1;
    }
    std::cout << "CLI input validation test passed\n";
    return 0;
}
