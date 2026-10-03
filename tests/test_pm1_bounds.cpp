#include "math/Pm1Bounds.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

static void expectNear(double actual, double expected, double relTol, const std::string& label) {
    if (!(std::fabs(actual - expected) <= relTol * std::fabs(expected))) {
        throw std::runtime_error(label + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

static void expect(bool ok, const std::string& label) {
    if (!ok) throw std::runtime_error(label);
}

int main() {
    // Dickman rho at integer points (published values).
    expectNear(math::dickmanRho(2.0), 0.3068528194, 1e-9, "rho(2)");
    expectNear(math::dickmanRho(3.0), 0.0486083883, 1e-5, "rho(3)");
    expectNear(math::dickmanRho(4.0), 4.910925648e-3, 1e-5, "rho(4)");
    expectNear(math::dickmanRho(5.0), 3.547247006e-4, 1e-5, "rho(5)");
    expectNear(math::dickmanRho(6.0), 1.964969635e-5, 1e-5, "rho(6)");
    expectNear(math::dickmanRho(8.0), 3.232069304e-8, 1e-5, "rho(8)");
    expectNear(math::dickmanRho(10.0), 2.770171838e-11, 1e-4, "rho(10)");

    // Success probabilities, against an independent Dickman-rho P-1 calculator.
    struct Case { uint32_t p; double tf, B1, B2, total, stage1; };
    const Case cases[] = {
        {130000003u, 77, 1000000, 30000000, 0.03997, 0.01594},
        {120000007u, 76,  500000, 15000000, 0.03490, 0.01300},
        { 60000011u, 74,  300000,  6000000, 0.03007, 0.01164},
        { 10000019u, 69,  100000,  2000000, 0.02638, 0.00922},
        {  1000003u, 60,   20000,  2000000, 0.03887, 0.00838},
    };
    for (const auto& c : cases) {
        const auto pr = math::pm1Probability(c.p, c.tf, c.B1, c.B2);
        const std::string label = "p=" + std::to_string(c.p);
        expectNear(pr.total(), c.total, 0.03, label + " total");
        expectNear(pr.stage1, c.stage1, 0.03, label + " stage1");
    }

    // Bound selection: a wavefront PRP assignment gets sensible, rounded bounds that pay off.
    const auto b = math::choosePm1Bounds(130000003u, 77, 1.0);
    std::printf("p=130000003 tf=77 tests_saved=1: B1=%llu B2=%llu P=%.2f%% cost=%.0f gain=%.0f\n",
                (unsigned long long)b.B1, (unsigned long long)b.B2, b.probability.total() * 100, b.cost, b.gain);
    expect(b.B1 >= 100000 && b.B1 <= 10000000, "wavefront B1 in a plausible range");
    expect(b.B2 > b.B1, "wavefront B2 above B1");
    expect(b.gain > 0, "positive expected saving");

    // More tests saved justifies at least as much P-1 effort; deeper TF justifies no more.
    const auto b2 = math::choosePm1Bounds(130000003u, 77, 2.0);
    expect(b2.B1 >= b.B1 && b2.probability.total() >= b.probability.total(), "tests_saved=2 searches harder");
    const auto b3 = math::choosePm1Bounds(130000003u, 82, 1.0);
    expect(b3.probability.total() <= b.probability.total(), "deeper TF lowers the success probability");

// tests_saved=0 is optimised exactly like one saved test.
const auto zero = math::choosePm1Bounds(130000003u, 77, 0.0);
expect(zero.B1 == b.B1 && zero.B2 == b.B2,
"tests_saved=0 gets the same bounds as tests_saved=1");

// No-regret fallback: no negative-gain fallback may cost more than all
// requested primality-test work combined.
const auto tinyOne = math::choosePm1Bounds(1277u, 76, 1.0);
expect(tinyOne.B1 == 0 && tinyOne.B2 == 0,
       "p=1277 saved=1: B1=1000 cannot fit the absolute saved-work budget");

const auto tiny = math::choosePm1Bounds(1277u, 76, 2.0);
std::printf("p=1277 tf=76 tests_saved=2: B1=%llu B2=%llu P=%.6f%% cost=%.0f gain=%.0f\n",
            (unsigned long long)tiny.B1,
            (unsigned long long)tiny.B2,
            tiny.probability.total() * 100,
            tiny.cost,
            tiny.gain);

expect(tiny.gain <= 0.0,
       "p=1277 saved=2: normal optimizer has no positive-gain bounds");
expect(tiny.B1 >= 1000,
       "p=1277 saved=2: a bounded fallback fits");
expect(tiny.cost <= 2.0 * 1277.0,
       "p=1277 saved=2: fallback stays inside absolute saved-work budget");

const auto small = math::choosePm1Bounds(2000003u, 65, 1.0);
if (small.gain <= 0.0) {
    expect(small.cost <= 2000003.0,
           "p=2M fallback stays inside absolute saved-work budget");
    expect(small.probability.total() >= 0.9 * math::kPm1FallbackSuccess,
           "p=2M still reaches approximately the historical fallback target");
}

    // Extreme and invalid input never crashes and gives sane bounds.
    const double inf = HUGE_VAL, nan = std::nan("");
    for (const double tf : {-5.0, 0.0, 10.0, 200.0, inf, nan}) {
        for (const double ts : {-3.0, 0.0, 1e9, inf, nan}) {
            const auto e = math::choosePm1Bounds(130000001u, tf, ts);
            expect(e.B1 >= 1000 && e.B1 <= 10000000000ULL, "extreme input: B1 in range");
            expect(e.B2 == 0 || e.B2 > e.B1, "extreme input: B2 above B1 or stage 1 only");
            expect(std::isfinite(e.probability.total()) && e.probability.total() >= 0.0 && e.probability.total() <= 1.0,
                   "extreme input: probability in [0, 1]");
        }
    }
    expect(math::choosePm1Bounds(2u, 70, 1).B1 == 0, "p < 3 gives no bounds");
    const auto huge = math::choosePm1Bounds(4294967291u, 90, 1);
    expect(huge.B1 > 0 && huge.B2 > huge.B1, "largest 32-bit prime exponent gets bounds");

    std::printf("P-1 bounds tests passed\n");
    return 0;
}
