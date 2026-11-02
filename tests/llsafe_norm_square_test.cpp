#include "math/LlSafeNormSquare.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

// Small exact modular adapter for the production schedule, not a GPU model.
// Prepared operands are tagged separately so reading one as a canonical
// register (or multiplying by an unprepared register) fails the test.
struct ModularEngine {
    using Reg = std::size_t;
    std::uint64_t modulus;
    mutable std::array<std::uint64_t, 18> words{};
    mutable std::array<bool, 18> prepared{};
    mutable unsigned calls = 0, squares = 0, products = 0, preparations = 0;

    explicit ModularEngine(std::uint64_t m) : modulus(m)
    {
        for (Reg i = 0; i < words.size(); ++i) words[i] = (i + 5) % m;
    }

    std::uint64_t get(Reg r) const
    {
        require(!prepared.at(r), "expected a canonical register");
        return words.at(r);
    }

    void put(Reg r, std::uint64_t value) const
    {
        words.at(r) = value % modulus;
        prepared.at(r) = false;
    }

    void set(Reg r, std::uint32_t value) const
    {
        ++calls;
        put(r, value);
    }

    void copy(Reg dst, Reg src) const
    {
        ++calls;
        put(dst, get(src));
    }

    void square_mul(Reg r, std::uint32_t factor = 1) const
    {
        ++calls;
        ++squares;
        const auto value = get(r);
        put(r, factor * value * value);
    }

    void set_multiplicand(Reg dst, Reg src) const
    {
        ++calls;
        ++preparations;
        put(dst, get(src));
        prepared.at(dst) = true;
    }

    void mul(Reg dst, Reg src, std::uint32_t factor = 1) const
    {
        ++calls;
        ++products;
        require(prepared.at(src), "multiplication needs a prepared operand");
        put(dst, factor * get(dst) * words.at(src));
    }

    void sub(Reg r, std::uint32_t value) const
    {
        ++calls;
        put(r, get(r) + modulus - value % modulus);
    }

    void sub_reg(Reg dst, Reg src) const
    {
        ++calls;
        put(dst, get(dst) + modulus - get(src));
    }

    bool is_equal(Reg lhs, Reg rhs) const
    {
        ++calls;
        return get(lhs) == get(rhs);
    }

    void reset_counts() const
    {
        calls = squares = products = preparations = 0;
    }
};

bool reference_norm(std::uint64_t a, std::uint64_t b, std::uint64_t m)
{
    return (a * a % m + m - 3 * b * b % m) % m == 1;
}

std::array<std::uint64_t, 2> reference_square(
    std::uint64_t a, std::uint64_t b, std::uint64_t m)
{
    return {(a * a + 3 * b * b) % m, 2 * a * b % m};
}

void check_all_pairs(std::uint64_t m)
{
    for (std::uint64_t a = 0; a < m; ++a) {
        for (std::uint64_t b = 0; b < m; ++b) {
            ModularEngine e(m);
            e.put(0, a);
            e.put(1, b);
            // The norm check reuses the prepared slot as its canonical one.
            e.set_multiplicand(14, 0);
            const auto before_norm = e.words;
            const bool unit = prmers::llsafe::has_unit_norm(e, 0, 1, 12, 13, 14);
            require(unit == reference_norm(a, b, m), "incorrect norm predicate");
            for (std::size_t i = 0; i < e.words.size(); ++i) {
                if (i != 12 && i != 13 && i != 14)
                    require(e.words[i] == before_norm[i], "norm check changed live state");
            }
            require(!e.prepared[14], "norm constant remained prepared");
            if (!unit) continue;

            const auto before_square = e.words;
            const auto expected = reference_square(a, b, m);
            e.reset_counts();
            prmers::llsafe::square_unit_norm(e, 0, 1, 14);
            require(e.get(0) == expected[0] && e.get(1) == expected[1],
                    "unit-norm square differs from general pair square");
            require(reference_norm(e.get(0), e.get(1), m), "norm was not preserved");
            require(e.calls == 4 && e.squares == 1 && e.products == 1 &&
                    e.preparations == 1, "unexpected hot-path arithmetic cost");
            require(e.prepared[14] && e.words[14] == a,
                    "prepared operand did not retain the old A");
            for (std::size_t i = 2; i < e.words.size(); ++i) {
                if (i != 14)
                    require(e.words[i] == before_square[i], "square changed unrelated state");
            }
        }
    }
}

void check_chains(std::uint64_t m)
{
    ModularEngine e(m);
    e.put(0, 2);
    e.put(1, 1);
    auto expected = std::array<std::uint64_t, 2>{2 % m, 1};
    for (unsigned iteration = 0; iteration < 64; ++iteration) {
        expected = reference_square(expected[0], expected[1], m);
        prmers::llsafe::square_unit_norm(e, 0, 1, 14);
        require(e.get(0) == expected[0] && e.get(1) == expected[1],
                "repeated specialized squares diverged");
        require(prmers::llsafe::has_unit_norm(e, 0, 1, 12, 13, 14),
                "valid chain failed the boundary norm check");
    }

    // Check both checkpoint pairs independently, using the same scratch slots.
    e.put(6, 2);
    e.put(7, 1);
    require(prmers::llsafe::has_unit_norm(e, 6, 7, 12, 13, 14),
            "valid rollback pair rejected");
    e.put(6, 0);
    e.put(7, 0);
    require(!prmers::llsafe::has_unit_norm(e, 6, 7, 12, 13, 14),
            "invalid rollback pair accepted");
    e.put(0, 0);
    e.put(1, 0);
    require(!prmers::llsafe::has_unit_norm(e, 0, 1, 12, 13, 14),
            "invalid live pair accepted");
}

void check_flag()
{
    using prmers::llsafe::parse_norm_square_flag;
    require(!parse_norm_square_flag(nullptr), "unset flag must be disabled");
    require(!parse_norm_square_flag("0"), "zero flag must be disabled");
    require(parse_norm_square_flag("1"), "one flag must enable the candidate");
    for (const char* invalid : {"", "true", "01", "2", "-1", "1 ", " 1"}) {
        bool rejected = false;
        try { (void)parse_norm_square_flag(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid flag accepted");
    }
}

} // namespace

int main()
{
    check_flag();
    // Include composite Mersenne rings and modulus 3: the identity must not
    // assume a field or divide by 2, 3, A or B. Exhaustive pairs include A=0.
    for (const std::uint64_t m : {3u, 7u, 15u, 31u, 63u, 127u}) {
        check_all_pairs(m);
        check_chains(m);
    }
    std::cout << "LLSAFE_NORM_SQUARE=PASS\n";
}
