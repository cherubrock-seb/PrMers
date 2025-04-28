#ifndef MATH_GERBICZ_LI_CHECKER_HPP
#define MATH_GERBICZ_LI_CHECKER_HPP

#include <vector>
#include <cstdint>

namespace math {

class GerbiczLiChecker {
public:

    GerbiczLiChecker(uint64_t a, uint64_t d0, size_t blockSize);

    size_t getBlockSize() const;

    bool check(const std::vector<uint64_t>& hostX,
               const std::vector<uint64_t>& hostD,
               const std::vector<uint64_t>& xPrev,
               const std::vector<uint64_t>& dPrev) const;

private:
    uint64_t a_;  
    uint64_t d0_;
    size_t   B_;
};

} // namespace math

#endif // MATH_GERBICZ_LI_CHECKER_HPP
