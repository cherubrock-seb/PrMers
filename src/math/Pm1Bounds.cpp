#include "math/Pm1Bounds.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace math {

namespace {

constexpr double kRhoStep = 1.0 / 1024;
constexpr double kRhoMax = 40.0;

// rho(u) = 1 on [0, 1], 1 - ln u on [1, 2], and beyond that the identity u rho(u) = integral_{u-1}^{u} rho(t) dt,
// with the trapezoidal rule over the window (solved for the unknown endpoint rho(u)). Unlike stepping
// rho'(u) = -rho(u - 1) / u forward, this keeps the error relative to rho, which falls to ~1e-30 at u = 25.
const std::vector<double>& rhoTable() {
    static const std::vector<double> table = [] {
        const size_t n = static_cast<size_t>(kRhoMax / kRhoStep) + 1;
        const size_t one = static_cast<size_t>(1.0 / kRhoStep);
        std::vector<double> r(n);
        for (size_t i = 0; i < n; ++i) {
            const double u = static_cast<double>(i) * kRhoStep;
            if (u <= 1.0) r[i] = 1.0;
            else if (u <= 2.0) r[i] = 1.0 - std::log(u);
            else {
                double interior = 0.0;
                for (size_t j = i - one + 1; j < i; ++j) interior += r[j];
                r[i] = kRhoStep * (0.5 * r[i - one] + interior) / (u - 0.5 * kRhoStep);
            }
        }
        return r;
    }();
    return table;
}

// Probability that k, with log k = alpha log B1, is B1-smooth apart from one prime in (B1, B2]:
// integral_1^{min(beta, alpha)} rho(alpha - x) / x dx, with beta = log B2 / log B1.
double semismooth(double alpha, double beta) {
    const double top = std::min(beta, alpha);
    if (top <= 1.0) return 0.0;
    constexpr int kSteps = 32;                       // even, for Simpson
    const double h = (top - 1.0) / kSteps;
    double sum = 0.0;
    for (int i = 0; i <= kSteps; ++i) {
        const double x = 1.0 + i * h;
        const double w = (i == 0 || i == kSteps) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        sum += w * dickmanRho(alpha - x) / x;
    }
    return sum * h / 3.0;
}

// Number of primes up to x, from the asymptotic expansion of the logarithmic integral.
double primePi(double x) {
    if (x < 3.0) return x < 2.0 ? 0.0 : 1.0;
    const double l = std::log(x);
    return x / l * (1.0 + 1.0 / l + 2.0 / (l * l) + 6.0 / (l * l * l));
}

// Round to two significant digits, as hand-picked bounds usually are.
uint64_t roundBound(double x) {
    if (x < 100.0) return static_cast<uint64_t>(std::llround(x));
    const double scale = std::pow(10.0, std::floor(std::log10(x)) - 1.0);
    return static_cast<uint64_t>(std::llround(x / scale) * scale);
}

struct Evaluated {
    double gain;
    Pm1Probability prob;
    double cost;
};

Evaluated evaluate(uint32_t p, double tfBits, double testsSaved, double stage2Cost, double B1, double B2) {
    Evaluated e;
    e.prob = pm1Probability(p, tfBits, B1, B2);
    const double stage1 = 1.4427 * B1;                 // log2 of the stage-1 exponent, about B1 / ln 2
    const double stage2 = B2 > B1 ? stage2Cost * (primePi(B2) - primePi(B1)) : 0.0;
    e.cost = stage1 + (1.0 - e.prob.stage1) * stage2;
    e.gain = e.prob.total() * testsSaved * static_cast<double>(p) - e.cost;
    return e;
}

} // namespace

double dickmanRho(double u) {
    if (u <= 1.0) return 1.0;
    if (u >= kRhoMax) return 0.0;
    const auto& r = rhoTable();
    const double pos = u / kRhoStep;
    const size_t i = static_cast<size_t>(pos);
    const double f = pos - static_cast<double>(i);
    return r[i] * (1.0 - f) + r[i + 1] * f;
}

Pm1Probability pm1Probability(uint32_t p, double tfBits, double B1, double B2) {
    Pm1Probability out;
    if (B1 < 2.0 || p < 2) return out;
    B2 = std::max(B2, B1);
    const double log2B1 = std::log2(B1);
    const double beta = std::log2(B2) / log2B1;
    const double kOffset = std::log2(static_cast<double>(p)) + 1.0;   // q = 2kp + 1, so log2 k ~ log2 q - log2(2p)
    constexpr double kSlice = 0.25;                                    // bits per slice of factor sizes

    double none1 = 1.0;   // probability that stage 1 finds no factor
    double none = 1.0;    // probability that neither stage finds a factor
    for (double b = tfBits + kSlice / 2; ; b += kSlice) {
        const double alpha = (b - kOffset) / log2B1;
        if (alpha >= kRhoMax) break;
        const double density = kSlice / b;
        const double s1 = dickmanRho(alpha);
        const double s2 = semismooth(alpha, beta);
        none1 *= 1.0 - density * s1;
        none *= 1.0 - density * (s1 + s2);
        if (alpha > beta + 2.0 && density * (s1 + s2) < 1e-15) break;
    }
    out.stage1 = 1.0 - none1;
    out.stage2 = none1 - none;
    return out;
}

Pm1Bounds choosePm1Bounds(uint32_t p, double tfBits, double testsSaved, double stage2CostPerPrime) {
    Pm1Bounds best;
    // Guard extreme or invalid input: every Mersenne factor is at least 2p + 1, factors above 2^128 add nothing
    // measurable, and tests_saved is optimised as at least one and at most ten tests.
    if (p < 3) return best;
    const double minTf = std::log2(2.0 * static_cast<double>(p) + 1.0);
    if (!std::isfinite(tfBits) || tfBits < minTf) tfBits = minTf;
    tfBits = std::min(tfBits, 128.0);
    if (!std::isfinite(testsSaved)) testsSaved = 1.0;
    testsSaved = std::min(std::max(testsSaved, 1.0), 10.0);
    if (!std::isfinite(stage2CostPerPrime) || stage2CostPerPrime <= 0.0) stage2CostPerPrime = kPm1Stage2CostPerPrime;

    // For each B1, find the best B2 / B1 ratio (up to 1000) by golden-section search on its logarithm.
    // B1 is searched coarse to fine: every 0.2 decade over the whole range, then every 0.05 decade
    // around the best coarse point.
    double bestGain = -HUGE_VAL, bestB1 = 0.0, bestB2 = 0.0;
    auto tryB1 = [&](double l1) {
        const double B1 = std::pow(10.0, l1);
        auto gainAt = [&](double logRatio) {
            return evaluate(p, tfBits, testsSaved, stage2CostPerPrime, B1, B1 * std::pow(10.0, logRatio)).gain;
        };
        constexpr double kPhi = 0.6180339887498949;
        double lo = 0.0, hi = 3.0;
        double x1 = hi - kPhi * (hi - lo), x2 = lo + kPhi * (hi - lo);
        double g1 = gainAt(x1), g2 = gainAt(x2);
        for (int it = 0; it < 20; ++it) {
            if (g1 < g2) { lo = x1; x1 = x2; g1 = g2; x2 = lo + kPhi * (hi - lo); g2 = gainAt(x2); }
            else         { hi = x2; x2 = x1; g2 = g1; x1 = hi - kPhi * (hi - lo); g1 = gainAt(x1); }
        }
        const double logRatio = g1 > g2 ? x1 : x2;
        // Stage 1 alone is a candidate too (ratio 1).
        const double gStage1Only = gainAt(0.0);
        const double g = std::max(std::max(g1, g2), gStage1Only);
        if (g > bestGain) {
            bestGain = g;
            bestB1 = B1;
            bestB2 = g == gStage1Only ? B1 : B1 * std::pow(10.0, logRatio);
        }
    };
    const double maxLog10B1 = std::max(3.0, std::log10(static_cast<double>(p)) + 1.0);
    for (double l1 = 3.0; l1 <= maxLog10B1 + 1e-9; l1 += 0.2) tryB1(l1);
    const double coarse = std::log10(bestB1);
    for (double l1 = std::max(3.0, coarse - 0.2); l1 <= std::min(maxLog10B1, coarse + 0.2) + 1e-9; l1 += 0.05) tryB1(l1);

    if (bestGain <= 0.0) {
        // Nothing pays off: take the cheapest bounds that reach the fallback success probability. Success
        // grows with both bounds, so for each B1 bisect for the smallest B2 / B1 ratio that reaches it.
        double bestCost = HUGE_VAL, bestP = -1.0;
        for (double l1 = 3.0; l1 <= 10.0 + 1e-9; l1 += 0.05) {
            const double B1 = std::pow(10.0, l1);
            auto probAt = [&](double logRatio) {
                return pm1Probability(p, tfBits, B1, B1 * std::pow(10.0, logRatio)).total();
            };
            const double pMax = probAt(3.0);
            if (pMax < kPm1FallbackSuccess) {
                // Unreachable at this B1; remember the most likely bounds in case it is unreachable everywhere.
                if (bestCost == HUGE_VAL && pMax > bestP) { bestP = pMax; bestB1 = B1; bestB2 = B1 * 1000.0; }
                continue;
            }
            double lo = 0.0, hi = 3.0;
            if (probAt(lo) < kPm1FallbackSuccess) {
                for (int it = 0; it < 20; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    (probAt(mid) >= kPm1FallbackSuccess ? hi : lo) = mid;
                }
            } else {
                hi = 0.0;
            }
            const double B2 = B1 * std::pow(10.0, hi);
            const double cost = evaluate(p, tfBits, testsSaved, stage2CostPerPrime, B1, B2).cost;
            if (cost < bestCost) { bestCost = cost; bestB1 = B1; bestB2 = B2; }
        }
    }

    best.B1 = roundBound(bestB1);
    best.B2 = roundBound(bestB2);
    if (best.B2 <= best.B1) best.B2 = 0;              // stage 1 only
    const Evaluated e = evaluate(p, tfBits, testsSaved, stage2CostPerPrime,
                                 static_cast<double>(best.B1),
                                 static_cast<double>(best.B2 ? best.B2 : best.B1));
    best.probability = e.prob;
    best.cost = e.cost;
    best.gain = e.gain;
    return best;
}

} // namespace math
