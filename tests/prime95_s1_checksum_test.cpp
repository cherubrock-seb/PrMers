// A Prime95 stage-1 (.p95) file whose checksum or magic number is wrong must be
// rejected, not loaded with a warning: it carries the P-1 residue that an
// extension or a stage-2 resume continues from.
#include "core/AlgoUtils.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using namespace core::algo;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond " (line " << __LINE__ << ")\n"; ++failures; } } while (0)

static std::vector<char> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static void spit(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), (std::streamsize)bytes.size());
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : ".";
    const std::string good = dir + "/good.p95";
    const std::string bad  = dir + "/bad.p95";

    std::vector<uint8_t> data(64);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t)(i * 37u + 11u);
    CHECK(write_prime95_s1_from_bytes(good, 8191u, 100000u, data, "", ""));

    uint32_t p = 0;
    uint64_t B1 = 0;
    std::vector<uint8_t> back;
    CHECK(read_prime95_s1_to_bytes(good, p, B1, back));
    CHECK(p == 8191u);
    CHECK(B1 == 100000u);
    CHECK(back == data);

    // Header layout: magic(4) ver(4) f64(8) type(4) p(4) n(4) 'S''1'(2) rsv(2) zero(8) f64(8)
    // chk(4) ... so the checksum word starts at byte 48; the residue data starts at byte 80.
    const std::vector<char> orig = slurp(good);
    CHECK(orig.size() > 80u + data.size());

    // One damaged residue byte.
    std::vector<char> v = orig;
    v[80 + 5] = (char)(v[80 + 5] ^ 0x40);
    spit(bad, v);
    CHECK(!read_prime95_s1_to_bytes(bad, p, B1, back));

    // Damaged checksum word.
    v = orig;
    v[48] = (char)(v[48] ^ 0x01);
    spit(bad, v);
    CHECK(!read_prime95_s1_to_bytes(bad, p, B1, back));

    // Wrong magic number.
    v = orig;
    v[0] = (char)(v[0] ^ 0x01);
    spit(bad, v);
    CHECK(!read_prime95_s1_to_bytes(bad, p, B1, back));

    // Truncated data is still rejected.
    v = orig;
    v.resize(80 + data.size() - 4);
    spit(bad, v);
    CHECK(!read_prime95_s1_to_bytes(bad, p, B1, back));

    std::remove(good.c_str());
    std::remove(bad.c_str());
    if (failures) return 1;
    std::cout << "prime95 S1 checksum test passed\n";
    return 0;
}
