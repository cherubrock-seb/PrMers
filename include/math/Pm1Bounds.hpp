#pragma once
#include <cstdint>

namespace math {

// Dickman's rho function: the probability that a random integer near x is x^(1/u)-smooth.
double dickmanRho(double u);

struct Pm1Probability {
    double stage1 = 0.0;   // probability that stage 1 alone finds a factor
    double stage2 = 0.0;   // additional probability from stage 2
    double total() const { return stage1 + stage2; }
};

// Probability that P-1 with bounds B1, B2 finds a factor of 2^p - 1, given that no factor below
// 2^tfBits exists. Uses the usual heuristics: a factor between 2^b and 2^(b+db) exists with
// probability db/b, a factor q = 2kp + 1 is found when k is B1-smooth (stage 1) or B1-smooth apart
// from one prime in (B1, B2] (stage 2), and smoothness follows Dickman's rho.
Pm1Probability pm1Probability(uint32_t p, double tfBits, double B1, double B2);

struct Pm1Bounds {
    uint64_t B1 = 0;
    uint64_t B2 = 0;
    Pm1Probability probability;
    double cost = 0.0;      // expected P-1 cost, in squarings mod 2^p - 1
    double gain = 0.0;      // expected squarings saved, net of the P-1 cost
};

// Cost of one stage-2 prime, in squarings. Measured for PrMers' default V-trace stage 2, which pairs
// primes so that about 0.7 terms are needed per prime, each costing about 2.2 squarings.
constexpr double kPm1Stage2CostPerPrime = 1.5;

// Success probability targeted when no bounds pay for themselves; roughly what a wavefront PRP
// assignment's optimal bounds give.
constexpr double kPm1FallbackSuccess = 0.03;

// Bounds maximising the expected saving of a P-1 run before a primality test, as Prime95 does for
// Pfactor= assignments: probability * testsSaved * p, minus the expected P-1 cost
// (1.4427 * B1 squarings for stage 1, plus stage2CostPerPrime per prime in (B1, B2] when stage 1 finds
// nothing). Always returns bounds, since some users run P-1 to find factors even when a primality test
// would be cheaper: tests_saved below 1 is treated as 1, and when no bounds give a positive expected
// saving (small exponents, where a primality test is very cheap) it returns the cheapest bounds whose
// estimated success reaches kPm1FallbackSuccess instead. gain <= 0 tells the caller which case applied.
// Input is sanitised: p < 3 gives B1 = B2 = 0; tfBits is raised to log2(2p + 1) (the smallest possible factor)
// and capped at 128; testsSaved is clamped to [1, 10]; a non-positive stage-2 cost uses the default.
Pm1Bounds choosePm1Bounds(uint32_t p, double tfBits, double testsSaved,
                          double stage2CostPerPrime = kPm1Stage2CostPerPrime);

} // namespace math
