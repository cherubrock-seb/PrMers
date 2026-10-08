// JsonBuilder::computeRes64 / computeRes64Iter / computeRes2048 for exponents of at most 32 bits.
//
// compactBits() returns (E - 1) / 32 + 1 words, which is one word for E <= 32.
// The res64 helpers read words[1] unconditionally, and computeRes2048 reads 64
// words, so these exponents read past the end of the vector (garbage digits in
// the displayed Res64, heap over-read). Build with -fsanitize=address to see
// the over-read directly.

#include "io/JsonBuilder.hpp"
#include "io/CliParser.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expectEq(const std::string& got, const std::string& want, const std::string& what) {
    if (got != want) {
        std::cerr << "FAIL " << what << ": got " << got << ", expected " << want << "\n";
        ++failures;
    } else {
        std::cout << "PASS " << what << " = " << got << "\n";
    }
}

} // namespace

int main() {
    // 2^31 - 1 residue 0x12345678 held in a single 31-bit digit.
    const std::vector<uint64_t> x = {0x12345678ULL};
    const std::vector<int> width = {31};

    io::CliOptions opts;
    opts.exponent = 31;

    opts.mode = "ll";
    expectEq(io::JsonBuilder::computeRes64(x, opts, width, 0.0, 0), "0000000012345678", "LL res64, p=31");
    expectEq(io::JsonBuilder::computeRes64Iter(x, opts, width, 0.0, 0), "0000000012345678", "res64 iter, p=31");

    const std::string r2048 = io::JsonBuilder::computeRes2048(x, opts, width, 0.0, 0);
    expectEq(std::to_string(r2048.size()), "512", "res2048 length, p=31");
    expectEq(r2048.substr(r2048.size() - 16), "0000000012345678", "res2048 low 64 bits, p=31");
    expectEq(r2048.substr(0, 16), "0000000000000000", "res2048 high bits are zero, p=31");

    // A multi-word exponent still reads both words.
    const std::vector<uint64_t> y = {0xFFFFFFFFULL, 0x1ULL};
    const std::vector<int> widthY = {32, 32};
    opts.exponent = 64;
    expectEq(io::JsonBuilder::computeRes64(y, opts, widthY, 0.0, 0), "00000001FFFFFFFF", "res64, p=64");

    if (failures != 0) {
        std::cerr << failures << " res64 small-exponent check(s) failed\n";
        return 1;
    }
    std::cout << "JsonBuilder res64 small-exponent test passed\n";
    return 0;
}
