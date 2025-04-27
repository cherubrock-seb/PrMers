// core/QuickChecker.cpp
/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
#include "core/QuickChecker.hpp"
#include <iostream>
#include <map>

namespace core {

std::optional<int> QuickChecker::run(uint32_t p) {
    if (p >= 89) return std::nullopt;

    static const std::map<uint32_t,bool> known = {
        {2,true},{3,true},{5,true},{7,true},
        {13,true},{17,true},{19,true},
        {31,true},{61,true},{89,true}
    };
    bool isPrime = (known.find(p) != known.end());
    std::cout << "\nKernel execution time: 0.0 seconds\n"
              << "Iterations per second: ∞ (simulated)\n\n"
              << "M" << p << (isPrime ? " is prime!" : " is composite.") << std::endl;
    return isPrime ? 0 : 1;
}

} // namespace core
