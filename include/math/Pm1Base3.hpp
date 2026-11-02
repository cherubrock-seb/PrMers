#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace prmers::pm1 {

inline bool parse_aevum_base3_flag(const char* value)
{
    if (value == nullptr || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("PRMERS_PM1_AEVUM_BASE3 must be 0 or 1");
}

// Later chunks exponentiate the preceding residue, not the integer 3.
// The extension loop is separate and must not use this specialization.
inline bool aevum_base3_candidate(bool requested, bool aevum,
                                  bool first_chunk, std::uint64_t next_start)
{
    return requested && aevum && first_chunk && next_start == 0;
}

// base and scratch must be distinct canonical registers. Exercise the existing
// factor-3 capacity guard on scratch before touching the live result. Exceptions
// propagate: an unsafe explicit FFT or a device failure must not be hidden.
// Equality also validates the base restored from a checkpoint.
template<class Engine>
bool check_base3(const Engine& eng, typename Engine::Reg base,
                 typename Engine::Reg scratch)
{
    eng.set(scratch, 1u);
    eng.square_mul(scratch, 3u);
    return eng.is_equal(base, scratch);
}

// Exact binary-exponent step x <- x*x*3^bit (mod 2^p-1), with base 3 verified
// before the loop. Scaling uses the engine's existing small-factor path;
// no prepared multiplication or new arithmetic representation is needed.
template<class Engine>
void base3_bit(const Engine& eng, typename Engine::Reg state, bool bit)
{
    eng.square_mul(state, bit ? 3u : 1u);
}

} // namespace prmers::pm1
