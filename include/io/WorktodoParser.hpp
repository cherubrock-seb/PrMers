#ifndef WORKTODO_PARSER_HPP
#define WORKTODO_PARSER_HPP

#include <string>
#include <optional>

namespace io {

struct WorktodoEntry {
    unsigned int exponent;
    bool prpTest;
    bool llTest;
};

class WorktodoParser {
public:
    explicit WorktodoParser(const std::string& filename);
    std::optional<WorktodoEntry> parse();

private:
    std::string filename_;
};

} // namespace io

#endif // WORKTODO_PARSER_HPP
