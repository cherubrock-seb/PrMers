#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Expand bundled quoted #include directives into one source string.
// Files containing #pragma once are emitted once; intentionally repeatable
// fragments (for example carryinc.cl) remain repeatable.
std::string buildMonolithicOpenCLSource(
    std::string_view rootFile,
    const std::vector<std::pair<std::string, std::string>>& files);
