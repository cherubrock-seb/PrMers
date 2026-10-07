// The submission JSON of a P-1 run must say which stage-2 range was searched.
// A low-memory stage 2 started with -b2start searches (B2Start, B2], but the
// JSON only carried "b1" and "b2", which claims the whole (B1, B2].
#include "io/CliParser.hpp"
#include "io/JsonBuilder.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

static int failures = 0;

static void expect(bool ok, const std::string& what, const std::string& json) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n  " << json << "\n";
        ++failures;
    }
}

static bool has(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

static std::string make(uint64_t b1, uint64_t b2, uint64_t b2start, bool lowmem, const char* mode = "pm1") {
    io::CliOptions o;
    o.mode = mode;
    o.exponent = 269;
    o.B1 = b1;
    o.B2 = b2;
    o.B2Start = b2start;
    o.pm1_lowmem = lowmem;
    return io::JsonBuilder::generate(o, 16, false, "0000000000000000", "");
}

int main() {
    // The stage-2 range does not start at B1: report where it starts.
    std::string j = make(4, 2141, 100, true);
    expect(has(j, "\"b1\":4,\"b2\":2141,\"b2-start\":100,"), "low-memory -b2start is reported", j);

    // No -b2start, or one at or below B1: the range is (B1, B2].
    j = make(4, 2141, 0, true);
    expect(has(j, "\"b1\":4,\"b2\":2141,") && !has(j, "b2-start"), "no b2-start without -b2start", j);
    j = make(4, 2141, 4, true);
    expect(!has(j, "b2-start"), "no b2-start when it equals B1", j);

    // Stage-2 paths that ignore -b2start search (B1, B2] whatever it says.
    j = make(4, 2141, 100, false);
    expect(has(j, "\"b2\":2141,") && !has(j, "b2-start"), "no b2-start when the stage 2 ignores it", j);

    // Stage 1 only.
    j = make(4, 0, 100, true);
    expect(!has(j, "\"b2\"") && !has(j, "b2-start"), "no b2-start without a stage 2", j);

    if (failures) return EXIT_FAILURE;
    std::cout << "pm1 b2start json test passed\n";
    return EXIT_SUCCESS;
}
