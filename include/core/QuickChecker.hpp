// core/QuickChecker.hpp
#ifndef CORE_QUICKCHECKER_HPP
#define CORE_QUICKCHECKER_HPP

#include <cstdint>
#include <optional>

namespace core {
class QuickChecker {
public:
    static std::optional<int> run(uint64_t p);
};

} // namespace core

#endif // CORE_QUICKCHECKER_HPP
