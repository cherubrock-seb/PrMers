#pragma once
#include <string>

namespace core {

// The legacy internal PrMers NTT path (selected by -marin, which clears options.marin) is not validated
// for Lucas-Lehmer. Returns the rejection text when `mode` would run LL on it, or an empty string.
// Used for the command line (main.cpp) and for a mode that came from a worktodo LL entry (App.cpp), so
// both sources follow the same rule. -allow-unvalidated-legacy-ll (`allowUnvalidated`) is the explicit
// opt-in that lets the run through; legacyLlWarning() is what to print when it does.
inline bool legacyLlOnLegacyPath(const std::string& mode, bool marinEngine) {
    return mode == "ll" && !marinEngine;
}

inline std::string legacyLlRejection(const std::string& mode, bool marinEngine, bool allowUnvalidated = false) {
    if (legacyLlOnLegacyPath(mode, marinEngine) && !allowUnvalidated) {
        return "[Backend Compatibility] -llunsafe cannot use the legacy internal "
               "PrMers NTT backend selected by -marin because that path is not "
               "validated for Lucas-Lehmer. Use automatic mode, -engine-marin, "
               "or -aevum, or pass -allow-unvalidated-legacy-ll to run it anyway.";
    }
    return std::string();
}

// Non-empty when the opt-in is what lets `mode` run Lucas-Lehmer on the legacy path.
inline std::string legacyLlWarning(const std::string& mode, bool marinEngine, bool allowUnvalidated) {
    if (legacyLlOnLegacyPath(mode, marinEngine) && allowUnvalidated) {
        return "Warning: -allow-unvalidated-legacy-ll: running Lucas-Lehmer on the legacy internal "
               "PrMers NTT backend (-marin), which is not validated for Lucas-Lehmer.";
    }
    return std::string();
}

} // namespace core
