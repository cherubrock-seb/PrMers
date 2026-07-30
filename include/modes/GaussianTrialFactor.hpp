#pragma once

#include <optional>

namespace core {

// Handles the lightweight Gaussian pair trial-factoring mode before App
// allocates the large NTT/FFT state. Returns std::nullopt when the command or
// worktodo does not request GMTF work.
std::optional<int> tryRunGaussianTrialFactor(int argc, char** argv);

} // namespace core
