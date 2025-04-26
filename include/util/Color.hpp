#ifndef UTIL_COLOR_HPP
#define UTIL_COLOR_HPP

namespace util {
    namespace Color {
        constexpr const char* RESET       = "\033[0m";
        constexpr const char* RED         = "\033[31m";
        constexpr const char* GREEN       = "\033[32m";
        constexpr const char* YELLOW      = "\033[33m";
        constexpr const char* BLUE        = "\033[34m";
        constexpr const char* MAGENTA     = "\033[35m";
        constexpr const char* CYAN        = "\033[36m";
        constexpr const char* WHITE       = "\033[37m";
        constexpr const char* BOLD        = "\033[1m";
        constexpr const char* UNDERLINE   = "\033[4m";

        constexpr const char* BOLD_RED    = "\033[1;31m";
        constexpr const char* BOLD_GREEN  = "\033[1;32m";
        constexpr const char* BOLD_YELLOW = "\033[1;33m";
        constexpr const char* BOLD_BLUE   = "\033[1;34m";
        constexpr const char* BOLD_MAGENTA= "\033[1;35m";
        constexpr const char* BOLD_CYAN   = "\033[1;36m";
        constexpr const char* BOLD_WHITE  = "\033[1;37m";
    }
}

#endif // UTIL_COLOR_HPP
