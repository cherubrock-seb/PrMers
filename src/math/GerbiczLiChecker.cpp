#include "math/GerbiczLiChecker.hpp"

namespace math {

GerbiczLiChecker::GerbiczLiChecker(uint64_t a,
                                   uint64_t d0,
                                   size_t blockSize)
  : B_(blockSize)
{}

size_t GerbiczLiChecker::getBlockSize() const {
    return B_;
}

bool GerbiczLiChecker::check(const std::vector<uint64_t>& hostX,
                              const std::vector<uint64_t>& hostD,
                              const std::vector<uint64_t>& xPrev,
                              const std::vector<uint64_t>& dPrev) const
{
    return hostX == xPrev
        && hostD == dPrev;
}

} // namespace math
