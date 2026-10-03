// src/io/WorktodoManager.hpp
#pragma once

#include "io/CliParser.hpp"
#include <string>

namespace io {

class WorktodoManager {
public:
    explicit WorktodoManager(const io::CliOptions& opts);

    // Both return true only when the result was fully written.
    bool saveIndividualJson(uint32_t p,
                            const std::string& mode,
                            const std::string& jsonResult) const;

    bool appendToResultsTxt(const std::string& jsonResult) const;

private:
    const io::CliOptions& options_;
};

} // namespace io
