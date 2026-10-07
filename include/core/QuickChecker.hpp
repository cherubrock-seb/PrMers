// core/QuickChecker.hpp
#ifndef CORE_QUICKCHECKER_HPP
#define CORE_QUICKCHECKER_HPP

#include <cstdint>
#include <optional>

#include "io/CliParser.hpp"

namespace core {
class QuickChecker {
public:
    // Table lookup of the known Mersenne primes 2^p - 1 for p < 127.
    static std::optional<int> run(uint64_t p);

    // Shortcut for a command-line test of a plain Mersenne number 2^p - 1.
    // Returns nullopt (run the real test) when the table does not describe the
    // number being tested: Wagstaff (opts.exponent holds 2p there), cofactors
    // (known factors divided out), and worktodo entries, which must go through
    // the normal result/removal path.
    static std::optional<int> run(const io::CliOptions& opts, bool fromWorktodo);
};

} // namespace core

#endif // CORE_QUICKCHECKER_HPP
