#pragma once

// This header is force-included only while compiling the v99.96 Gaussian
// factoring translation unit.  App.hpp is parsed first with its normal class
// definition, then the implementation symbol is renamed.  This avoids touching
// RunGaussianMersenneFactor.cpp and keeps the old ECM implementation available
// as an exact fallback.
#include "core/App.hpp"
#define runGaussianMersenneECM runGaussianMersenneECMLegacy
