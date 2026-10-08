// Host test for include/math/EcmKnownFactors.hpp: with -ecm-continue-after-factor a
// later curve can find a product of an already known factor and a new one; only the
// new part may be reported.
#include "math/EcmKnownFactors.hpp"

#include <iostream>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++failures; std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; } } while (0)

// strip() on a value given as text: returns the "known" flag and leaves the result in g.
static bool run(const char* g_text, const std::vector<std::string>& known, mpz_class& g) {
    g = mpz_class(g_text);
    return ecm_known::strip(g, known);
}

int main() {
    mpz_class g;

    // 2^41 - 1 = 13367 * 164511353.
    const mpz_class p1 = 13367, p2 = 164511353;
    CHECK(p1 * p2 == (mpz_class(1) << 41) - 1);

    // Exact match: known, g unchanged (it is shown as the known factor).
    CHECK(run("13367", {"13367"}, g) && g == p1);
    // Nothing known, or something unrelated: new, unchanged.
    CHECK(!run("13367", {}, g) && g == p1);
    CHECK(!run("13367", {"164511353"}, g) && g == p1);

    // The bug: p1 was reported earlier, a later curve finds p1 * p2.
    CHECK(!run("2199023255551", {"13367"}, g) && g == p2);
    CHECK(!ecm_known::strip(g = p1 * p2, {"164511353"}) && g == p1);
    // Both known: nothing new, g unchanged.
    CHECK(ecm_known::strip(g = p1 * p2, {"13367", "164511353"}) && g == p1 * p2);
    // Order of the known list does not matter.
    CHECK(ecm_known::strip(g = p1 * p2, {"164511353", "13367"}));

    // A known factor that divides g more than once is removed completely.
    CHECK(ecm_known::strip(g = p1 * p1, {"13367"}));
    CHECK(!ecm_known::strip(g = p1 * p1 * p2, {"13367"}) && g == p2);
    // A composite known factor (the product of two reported factors).
    CHECK(!ecm_known::strip(g = p1 * p2 * 3, {"2199023255551"}) && g == 3);
    // A composite known factor does not hide a new prime that is only a part of it.
    CHECK(!ecm_known::strip(g = p2, {"2199023255551"}) && g == p2);

    // Known factors of the wrong shape are ignored, never fatal.
    const std::vector<std::string> garbage = {
        "", " ", "abc", "12abc", "0", "1", "-1", "+", "-", "0x", "0xzz", "1e5", "13 367",
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    };
    CHECK(!run("13367", garbage, g) && g == p1);
    // ... and do not stop a good entry that follows them.
    std::vector<std::string> mixed = garbage;
    mixed.push_back("13367");
    CHECK(run("13367", mixed, g) && g == p1);

    // Spelling of a known factor: white space, sign, leading zeros, 0x prefix.
    CHECK(run("13367", {" 13367 "}, g));
    CHECK(run("13367", {"\t13367\n"}, g));
    CHECK(run("13367", {"+13367"}, g));
    CHECK(run("13367", {"-13367"}, g));
    CHECK(run("13367", {"0013367"}, g));          // decimal, not octal
    CHECK(run("13367", {"013367"}, g));
    CHECK(run("13367", {"0x3437"}, g));           // 13367 = 0x3437
    CHECK(run("13367", {"0X3437"}, g));

    // Degenerate g: nothing to report.
    CHECK(ecm_known::strip(g = 0, {"13367"}));
    CHECK(ecm_known::strip(g = 1, {}));
    CHECK(ecm_known::strip(g = -5, {"13367"}));

    // Boundaries around 2^32 and 2^64 (known factors wider than an LLP64 unsigned long).
    const mpz_class big = (mpz_class(1) << 64) + 13;     // only used as a divisor
    CHECK(ecm_known::strip(g = big, {big.get_str()}));
    CHECK(!ecm_known::strip(g = big * 5, {big.get_str()}) && g == 5);
    const mpz_class b32 = (mpz_class(1) << 32);
    const mpz_class b32p15 = b32 + 15, b32m1 = b32 - 1;
    CHECK(ecm_known::strip(g = b32 + 15, {b32p15.get_str()}));
    CHECK(!ecm_known::strip(g = (b32 - 1) * 7, {b32m1.get_str()}) && g == 7);
    CHECK(!ecm_known::strip(g = (b32 - 1) * 7, {std::to_string(0xFFFFFFFFULL)}) && g == 7);

    // A long known list stays instant.
    std::vector<std::string> many;
    for (int i = 2; i < 20000; ++i) many.push_back(std::to_string(i * 2 + 1000003));
    many.push_back("13367");
    CHECK(!ecm_known::strip(g = p1 * p2, many) && g == p2);

    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "ecm known factors test passed\n";
    return 0;
}
