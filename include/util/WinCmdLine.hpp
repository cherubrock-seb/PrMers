#pragma once
// Windows command-line construction that round-trips through CommandLineToArgvW and the
// MSVCRT/MinGW argv parser. Pure C++ (no windows.h) so it can be unit-tested on any host.
#include <string>
#include <vector>

namespace util {

// Quote one argument (not argv[0]) so the child's CommandLineToArgvW returns it unchanged.
//  - An argument with no space, tab, newline, vertical tab or double quote and not empty is left alone.
//  - Otherwise it is wrapped in double quotes; backslashes that precede a double quote (or the
//    closing quote) are doubled, and each embedded double quote gets one more backslash.
template <class CharT>
std::basic_string<CharT> winQuoteArg(const std::basic_string<CharT>& arg) {
    using S = std::basic_string<CharT>;
    const auto special = [](CharT c) {
        return c == CharT(' ') || c == CharT('\t') || c == CharT('\n') || c == CharT('\v') || c == CharT('"');
    };
    bool plain = !arg.empty();
    for (CharT c : arg) {
        if (special(c)) { plain = false; break; }
    }
    if (plain) return arg;

    S out;
    out.reserve(arg.size() + 2);
    out.push_back(CharT('"'));
    for (auto it = arg.begin();; ++it) {
        size_t backslashes = 0;
        while (it != arg.end() && *it == CharT('\\')) { ++it; ++backslashes; }
        if (it == arg.end()) {
            out.append(backslashes * 2, CharT('\\'));   // trailing backslashes precede the closing quote
            break;
        }
        if (*it == CharT('"')) {
            out.append(backslashes * 2 + 1, CharT('\\'));
            out.push_back(CharT('"'));
        } else {
            out.append(backslashes, CharT('\\'));
            out.push_back(*it);
        }
    }
    out.push_back(CharT('"'));
    return out;
}

// argv[0] (the program path) is parsed differently: a quoted first token ends at the next quote
// and backslashes are literal. Wrap it in quotes; a path cannot contain a double quote on Windows.
template <class CharT>
std::basic_string<CharT> winQuoteProgram(const std::basic_string<CharT>& path) {
    std::basic_string<CharT> out;
    out.reserve(path.size() + 2);
    out.push_back(CharT('"'));
    for (CharT c : path) {
        if (c != CharT('"')) out.push_back(c);
    }
    out.push_back(CharT('"'));
    return out;
}

// "prog" arg1 arg2 ... for CreateProcess; args[0] is the program, the rest are arguments.
template <class CharT>
std::basic_string<CharT> winCommandLine(const std::vector<std::basic_string<CharT>>& args) {
    std::basic_string<CharT> out;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i != 0) out.push_back(CharT(' '));
        out += i == 0 ? winQuoteProgram(args[i]) : winQuoteArg(args[i]);
    }
    return out;
}

}  // namespace util
