#pragma once

// Force-included only while compiling RunGaussianMersenneFactor.cpp.
// Parse App.hpp first with both public and legacy declarations, then rename only
// the implementation/calls in the legacy translation unit.
#include "core/App.hpp"
#define runGaussianMersennePM1 runGaussianMersennePM1Legacy
