#pragma once
#include <cstdint>

#define MOD_P 0xffffffff00000001ULL

namespace math {


enum class ModMode { Solinas, GF31, GF61 };

struct gf31_2 { uint64_t a; uint64_t b; };
struct gf61_2 { uint64_t a; uint64_t b; };

class Mod64 {
public:
    static void setMode(ModMode m);
    static ModMode mode();

    static uint64_t mulModP(uint64_t a, uint64_t b);
    static uint64_t powModP(uint64_t base, uint64_t exp);
    static uint64_t invModP(uint64_t x);

    static uint64_t add31(uint64_t a, uint64_t b);
    static uint64_t sub31(uint64_t a, uint64_t b);
    static uint64_t mul31(uint64_t a, uint64_t b);
    static uint64_t pow31(uint64_t base, uint64_t exp);
    static uint64_t inv31(uint64_t x);

    static gf31_2 add31_2(gf31_2 x, gf31_2 y);
    static gf31_2 sub31_2(gf31_2 x, gf31_2 y);
    static gf31_2 mul31_2(gf31_2 x, gf31_2 y);
    static gf31_2 pow31_2(gf31_2 x, uint64_t e);
    static gf31_2 inv31_2(gf31_2 x);

    static uint64_t add61(uint64_t a, uint64_t b);
    static uint64_t sub61(uint64_t a, uint64_t b);
    static uint64_t mul61(uint64_t a, uint64_t b);
    static uint64_t pow61(uint64_t base, uint64_t exp);
    static uint64_t inv61(uint64_t x);

    static gf61_2 add61_2(gf61_2 x, gf61_2 y);
    static gf61_2 sub61_2(gf61_2 x, gf61_2 y);
    static gf61_2 mul61_2(gf61_2 x, gf61_2 y);
    static gf61_2 pow61_2(gf61_2 x, uint64_t e);
    static gf61_2 inv61_2(gf61_2 x);
};

uint64_t mulModP(uint64_t a, uint64_t b);
uint64_t powModP(uint64_t base, uint64_t exp);
uint64_t invModP(uint64_t x);

}
