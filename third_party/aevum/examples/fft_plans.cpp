#include "../src/EngineApi.h"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
    constexpr std::array<uint32_t, 4> exponents = {
        1362763u,
        16279841u,
        136279841u,
        2147483647u
    };

    for (uint32_t exponent : exponents) {
        char spec[128]{};
        if (!aevum_engine_resolve_fft(exponent, "", spec, sizeof(spec))) {
            std::cout << "M" << exponent << ": unavailable: "
                      << aevum_engine_last_error() << '\n';
            continue;
        }
        std::cout << "M" << exponent << ": " << spec << '\n';
    }
    return 0;
}
