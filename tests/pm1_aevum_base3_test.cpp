#include "math/Pm1Base3.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

// Exact small-ring adapter for the production helper. This does not model GPU
// kernels or floating-point transforms. Moduli below 2^31 keep 3*x*x in uint64_t.
struct ModularEngine {
    using Reg = std::size_t;
    std::uint64_t modulus;
    mutable std::array<std::uint64_t, 11> words{};
    mutable unsigned square_calls = 0;
    bool factor3_safe = true;

    explicit ModularEngine(std::uint64_t m) : modulus(m)
    {
        for (Reg r = 0; r < words.size(); ++r) words[r] = (r + 1) % m;
    }

    void set(Reg r, std::uint32_t value) const { words.at(r) = value % modulus; }
    void square_mul(Reg r, std::uint32_t factor = 1) const
    {
        ++square_calls;
        if (factor == 3 && !factor3_safe)
            throw std::runtime_error("small-factor capacity exceeded");
        const auto value = words.at(r);
        words.at(r) = factor * value * value % modulus;
    }
    bool is_equal(Reg a, Reg b) const { return words.at(a) == words.at(b); }
};

void flags_and_eligibility()
{
    using namespace prmers::pm1;
    require(!parse_aevum_base3_flag(nullptr), "unset must preserve the default");
    require(!parse_aevum_base3_flag("0"), "zero must disable");
    require(parse_aevum_base3_flag("1"), "one must enable");
    for (const char* value : {"", "true", "01", "2", "-1", "1 ", " 1"}) {
        bool rejected = false;
        try { (void)parse_aevum_base3_flag(value); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid flag accepted");
    }
    for (bool requested : {false, true})
        for (bool aevum : {false, true})
            for (bool first : {false, true}) {
                require(aevum_base3_candidate(requested, aevum, first, 0) ==
                            (requested && aevum && first),
                        "single-chunk eligibility is wrong");
                for (std::uint64_t next : {3u, 101u, 4294967295u})
                    require(!aevum_base3_candidate(requested, aevum, first, next),
                            "multi-chunk run admitted");
            }
}

void base_and_capacity_gates()
{
    using prmers::pm1::check_base3;
    ModularEngine e(127);
    e.set(10, 3);
    const auto before = e.words;
    require(check_base3(e, 10, 5), "base three rejected");
    for (std::size_t r = 0; r < e.words.size(); ++r)
        if (r != 5) require(e.words[r] == before[r], "probe changed live state");
    require(e.square_calls == 1, "probe did not exercise the factor guard");

    e.set(10, 7);
    require(!check_base3(e, 10, 5), "non-three checkpoint base accepted");
    e.set(10, 3);
    e.factor3_safe = false;
    const auto live = e.words[0];
    bool rejected = false;
    try { (void)check_base3(e, 10, 5); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "capacity failure was swallowed");
    require(e.words[0] == live && e.words[10] == 3, "capacity probe changed state/base");
}

void bit_steps(std::uint64_t modulus)
{
    for (std::uint64_t value = 0; value < modulus && value < 256; ++value) {
        for (bool bit : {false, true}) {
            ModularEngine e(modulus);
            e.words[0] = value;
            const auto before = e.words;
            // Independent general square followed by multiplication by base 3.
            auto expected = value * value % modulus;
            if (bit) expected = expected * 3 % modulus;
            prmers::pm1::base3_bit(e, 0, bit);
            require(e.words[0] == expected, "bit step differs from general arithmetic");
            require(e.square_calls == 1, "bit step issued more than one square");
            for (std::size_t r = 1; r < e.words.size(); ++r)
                require(e.words[r] == before[r], "bit step changed another register");
        }
    }
    // Upper-bound values exercise reduction of 3*x*x without overflowing u64.
    for (const auto value : {modulus - 1, modulus - 2, modulus / 2}) {
        ModularEngine e(modulus);
        e.words[0] = value;
        const auto expected = (value * value % modulus) * 3 % modulus;
        prmers::pm1::base3_bit(e, 0, true);
        require(e.words[0] == expected, "upper-bound bit step differs");
    }
}

std::uint64_t reference_power(std::uint64_t exponent, std::uint64_t modulus)
{
    // Right-to-left exponentiation is independent of the production bit order.
    std::uint64_t result = 1, base = 3 % modulus;
    while (exponent) {
        if (exponent & 1) result = result * base % modulus;
        base = base * base % modulus;
        exponent >>= 1;
    }
    return result;
}

void exponent_and_resume(std::uint64_t modulus)
{
    for (const std::uint64_t exponent : {1u, 2u, 3u, 17u, 255u, 538u, 65535u}) {
        for (unsigned cut = 0; cut <= 16; ++cut) {
            for (bool fast_first : {false, true}) {
                ModularEngine e(modulus);
                e.words[0] = 1;
                for (unsigned position = 0; position < 16; ++position) {
                    const bool bit = ((exponent >> (15 - position)) & 1) != 0;
                    const bool fast = position < cut ? fast_first : !fast_first;
                    if (fast) prmers::pm1::base3_bit(e, 0, bit);
                    else {
                        e.square_mul(0);
                        if (bit) e.words[0] = e.words[0] * 3 % modulus;
                    }
                    if (position == cut) {
                        // Canonical state may cross the mode boundary unchanged.
                        ModularEngine restored(modulus);
                        restored.words = e.words;
                        e = restored;
                    }
                }
                require(e.words[0] == reference_power(exponent, modulus),
                        "mode transition changed the exponentiation");
            }
        }
    }
}

} // namespace

int main()
{
    flags_and_eligibility();
    base_and_capacity_gates();
    for (const std::uint64_t modulus : {3u, 7u, 15u, 31u, 127u, 2147483647u}) {
        bit_steps(modulus);
        exponent_and_resume(modulus);
    }
    std::cout << "PM1_AEVUM_BASE3=PASS\n";
}
