#pragma once

#include <stdexcept>
#include <string_view>

namespace prmers::llsafe {

// Explicit opt-in only. Reject misspellings instead of silently selecting a path.
inline bool parse_norm_square_flag(const char* value)
{
    if (value == nullptr || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("PRMERS_LLSAFE_NORM_SQUARE must be 0 or 1");
}

// Square A + B*sqrt(3) when A*A - 3*B*B == 1 modulo 2^p - 1.
// Then A' = 2*A*A - 1 and B' = 2*A*B. No primality or division is required.
// a, b and multiplicand must be distinct. Preparing the old A before changing
// it preserves the operand needed by B; only these three registers are written.
template<class Engine>
void square_unit_norm(const Engine& eng, typename Engine::Reg a,
                      typename Engine::Reg b, typename Engine::Reg multiplicand)
{
    eng.set_multiplicand(multiplicand, a);
    eng.square_mul(a, 2u);
    eng.sub(a, 1u);
    eng.mul(b, multiplicand, 2u);
}

// Check the specialization's precondition using general arithmetic. All five
// registers must be distinct. Only scratch_a, scratch_b and one are written;
// one may previously have held a prepared multiplicand.
template<class Engine>
bool has_unit_norm(const Engine& eng, typename Engine::Reg a,
                   typename Engine::Reg b, typename Engine::Reg scratch_a,
                   typename Engine::Reg scratch_b, typename Engine::Reg one)
{
    eng.copy(scratch_a, a);
    eng.copy(scratch_b, b);
    eng.square_mul(scratch_a);
    eng.square_mul(scratch_b, 3u);
    eng.sub_reg(scratch_a, scratch_b);
    eng.set(one, 1u);
    return eng.is_equal(scratch_a, one);
}

} // namespace prmers::llsafe
