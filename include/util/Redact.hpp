#pragma once
#include <regex>
#include <string>
#include <vector>

namespace util {

// Options whose value must never be echoed to the console, prmers.log or the GUI.
inline bool isSecretOption(const std::string& arg) {
    return arg == "-password";
}

// Copy of args with the value following each secret option replaced by "********".
inline std::vector<std::string> redactSecretArgs(std::vector<std::string> args) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (isSecretOption(args[i])) args[++i] = "********";
    }
    return args;
}

// Same for whitespace-separated option text such as a settings file.
inline std::string redactSecretText(const std::string& text) {
    static const std::regex re(R"((^|\s)(-password)(\s+)\S+)");
    return std::regex_replace(text, re, "$1$2$3********");
}

// Remove secret options and their values from option text.
inline std::string stripSecretText(const std::string& text) {
    static const std::regex re(R"((^|\s)-password\s+\S+)");
    return std::regex_replace(text, re, "$1");
}

} // namespace util
