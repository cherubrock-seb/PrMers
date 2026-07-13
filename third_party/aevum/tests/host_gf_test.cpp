#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

using u64 = std::uint64_t;
using u128 = unsigned __int128;

static constexpr u64 M31 = (u64{1} << 31) - 1;
static constexpr u64 M61 = (u64{1} << 61) - 1;

static u64 red31(u128 x) {
    u64 r = u64(x & M31) + u64(x >> 31);
    r = (r & M31) + (r >> 31);
    r = (r & M31) + (r >> 31);
    return r >= M31 ? r - M31 : r;
}

static u64 red61(u128 x) {
    u64 r = u64(x & M61) + u64(x >> 61);
    r = (r & M61) + (r >> 61);
    return r >= M61 ? r - M61 : r;
}

static u64 addp(u64 a, u64 b, u64 p) {
    u64 s = a + b;
    if (s >= p || s < a) s -= p;
    return s;
}

static u64 subp(u64 a, u64 b, u64 p) { return a >= b ? a - b : a + p - b; }

struct Gf2 { u64 a, b; };

static Gf2 mul31(Gf2 x, Gf2 y) {
    u64 ac = red31(u128(x.a) * y.a);
    u64 bd = red31(u128(x.b) * y.b);
    u64 ad = red31(u128(x.a) * y.b);
    u64 bc = red31(u128(x.b) * y.a);
    return {subp(ac, bd, M31), addp(ad, bc, M31)};
}

static Gf2 mul61(Gf2 x, Gf2 y) {
    u64 ac = red61(u128(x.a) * y.a);
    u64 bd = red61(u128(x.b) * y.b);
    u64 ad = red61(u128(x.a) * y.b);
    u64 bc = red61(u128(x.b) * y.a);
    return {subp(ac, bd, M61), addp(ad, bc, M61)};
}

static u64 inv_mod(u64 a, u64 p) {
    u64 e = p - 2, r = 1;
    auto mul = [p](u64 x, u64 y) -> u64 {
        return p == M31 ? red31(u128(x) * y) : red61(u128(x) * y);
    };
    while (e) {
        if (e & 1) r = mul(r, a);
        a = mul(a, a);
        e >>= 1;
    }
    return r;
}

static u128 crt(u64 a31, u64 a61) {
    static const u64 inv31mod61 = inv_mod(M31, M61);
    u64 delta = subp(a61, a31, M61);
    u64 t = red61(u128(delta) * inv31mod61);
    return u128(a31) + u128(M31) * t;
}

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

int main() {
    std::mt19937_64 rng(0xAE7A3161ULL);
    for (int i = 0; i < 200000; ++i) {
        u64 a31 = rng() % M31, b31 = rng() % M31;
        u64 a61 = rng() % M61, b61 = rng() % M61;
        require(red31(u128(a31) * b31) == u64((u128(a31) * b31) % M31), "M31 multiply");
        require(red61(u128(a61) * b61) == u64((u128(a61) * b61) % M61), "M61 multiply");

        Gf2 x31{a31, b31}, y31{rng() % M31, rng() % M31};
        Gf2 z31 = mul31(x31, y31);
        u64 e31a = u64((u128(x31.a) * y31.a + u128(M31) - (u128(x31.b) * y31.b) % M31) % M31);
        u64 e31b = u64((u128(x31.a) * y31.b + u128(x31.b) * y31.a) % M31);
        require(z31.a == e31a && z31.b == e31b, "GF31 multiply");

        Gf2 x61{a61, b61}, y61{rng() % M61, rng() % M61};
        Gf2 z61 = mul61(x61, y61);
        u64 e61a = u64((u128(x61.a) * y61.a + u128(M61) - (u128(x61.b) * y61.b) % M61) % M61);
        u64 e61b = u64((u128(x61.a) * y61.b + u128(x61.b) * y61.a) % M61);
        require(z61.a == e61a && z61.b == e61b, "GF61 multiply");

        u128 c = crt(a31, a61);
        require(u64(c % M31) == a31, "CRT M31");
        require(u64(c % M61) == a61, "CRT M61");
    }
    std::cout << "Aevum host arithmetic tests passed\n";
    return 0;
}
