#pragma once
// Reading settings.cfg, and the rules for the part of it the web GUI may write.
//
// settings.cfg is option text read like a command line: whitespace-separated tokens, a token starting
// with '#' starts a comment that runs to the end of the line. The GUI writes its options below a marker
// line; everything above the marker is written by hand (the GUI copies it unchanged on every save).
//
// Options that name a file or directory (-worktodo, -f, -config, -kernelpath, ...) or change where the
// GUI listens (-host, -http, -ipv4) are "locked": the GUI never writes them, the server rejects settings
// text that contains them, and the loader ignores them below the marker. Above the marker, and in a
// file with no marker, they are honoured as before.
//
// The server-side check and the loader share configLineTokens(), so the check sees exactly the tokens
// the loader passes to the option parser.

#include <cstddef>
#include <istream>
#include <locale>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace util {

inline const std::string& guiSettingsMarker() {
    static const std::string m =
        "# --- Written by the PrMers GUI below this line. Path and GUI network options here are ignored. ---";
    return m;
}

inline constexpr std::size_t kMaxGuiSettingsBytes = 16 * 1024;

inline std::string trimConfigLine(const std::string& line) {
    const auto a = line.find_first_not_of(" \t\r\n\f\v");
    if (a == std::string::npos) return std::string();
    const auto b = line.find_last_not_of(" \t\r\n\f\v");
    return line.substr(a, b - a + 1);
}

inline bool isGuiSettingsMarker(const std::string& line) { return trimConfigLine(line) == guiSettingsMarker(); }

// Tokens of one line, as every settings.cfg reader splits it (whitespace in the classic locale).
inline std::vector<std::string> configLineTokens(const std::string& line) {
    std::istringstream iss(line);
    iss.imbue(std::locale::classic());
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) {
        if (tok[0] == '#') break;
        out.push_back(tok);
    }
    return out;
}

// The option name a token would mean to a lenient parser: text before any NUL (the option parser
// compares C strings), surrounding quotes, all leading dashes and anything from '=' removed, lower case.
// Empty when the token is not an option. Deliberately broader than CliParser (which only knows the
// exact single-dash spelling), so that a future alias or "--opt=value" form cannot slip through.
inline std::string normalizedOptionName(const std::string& token) {
    std::string s = token.substr(0, token.find('\0'));
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == '"' || s[a] == '\'')) ++a;
    while (b > a && (s[b - 1] == '"' || s[b - 1] == '\'')) --b;
    s = s.substr(a, b - a);
    if (s.empty() || s[0] != '-') return std::string();
    const auto d = s.find_first_not_of('-');
    if (d == std::string::npos) return std::string();
    s = s.substr(d);
    s = s.substr(0, s.find('='));
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

struct GuiLockedOption {
    const char* name;     // without the leading dash
    bool takesValue;
};

// Locked option this token names, or nullptr.
inline const GuiLockedOption* guiLockedOption(const std::string& token) {
    static const GuiLockedOption kLocked[] = {
        {"worktodo", true},     // worktodo file (the GUI appends to it)
        {"config", true},       // settings file (the GUI saves to it)
        {"f", true},            // save/checkpoint directory, results.txt
        {"kernelpath", true},   // OpenCL kernel source
        {"kernel_path", true},
        {"kernel-path", true},
        {"output", true},       // not parsed today; a path by name
        {"filemers", true},     // .mers file to read and export
        {"p95path", true},      // Prime95 file location
        {"host", true},         // GUI bind address
        {"http", true},         // GUI port
        {"ipv4", false},        // GUI on the first LAN address
    };
    const std::string name = normalizedOptionName(token);
    if (name.empty()) return nullptr;
    for (const auto& o : kLocked)
        if (name == o.name) return &o;
    return nullptr;
}

// True when the locked option in `token` takes its value from the next token.
inline bool lockedOptionConsumesNext(const std::string& token, const GuiLockedOption& o) {
    return o.takesValue && token.find('=') == std::string::npos;
}

struct ConfigArgs {
    std::vector<std::string> args;      // tokens for the option parser
    std::vector<std::string> ignored;   // locked options (and values) dropped from the GUI section
};

// Read a settings file the way PrMers applies it: every token above the GUI marker; below it,
// everything except locked options and their values.
inline ConfigArgs readConfigArgs(std::istream& in) {
    ConfigArgs out;
    bool gui = false;
    bool dropNext = false;
    std::string line;
    while (std::getline(in, line)) {
        if (!gui && isGuiSettingsMarker(line)) {
            gui = true;
            // A hand-written locked option left without a value must not take its value from the GUI.
            if (!out.args.empty()) {
                const GuiLockedOption* o = guiLockedOption(out.args.back());
                if (o && lockedOptionConsumesNext(out.args.back(), *o)) {
                    out.ignored.push_back(out.args.back());
                    out.args.pop_back();
                }
            }
            continue;
        }
        for (const std::string& tok : configLineTokens(line)) {
            if (!gui) { out.args.push_back(tok); continue; }
            if (dropNext) { dropNext = false; out.ignored.push_back(tok); continue; }
            if (const GuiLockedOption* o = guiLockedOption(tok)) {
                out.ignored.push_back(tok);
                dropNext = lockedOptionConsumesNext(tok, *o);
                continue;
            }
            out.args.push_back(tok);
        }
    }
    return out;
}

inline ConfigArgs readConfigArgsFromText(const std::string& text) {
    std::istringstream in(text);
    return readConfigArgs(in);
}

namespace detail {
inline std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> lines;
    std::string line;
    std::istringstream in(text);
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}
}  // namespace detail

// Option text for the GUI's settings box: the GUI section of the file (the whole file when it has no
// marker), without locked options. One line of space-separated tokens per non-empty source line.
inline std::string guiEditableSettings(const std::string& fileText) {
    const auto lines = detail::splitLines(fileText);
    std::size_t start = 0;
    for (std::size_t i = 0; i < lines.size(); ++i)
        if (isGuiSettingsMarker(lines[i])) { start = i + 1; break; }
    std::string out;
    bool dropNext = false;
    for (std::size_t i = start; i < lines.size(); ++i) {
        std::string kept;
        for (const std::string& tok : configLineTokens(lines[i])) {
            if (dropNext) { dropNext = false; continue; }
            if (const GuiLockedOption* o = guiLockedOption(tok)) { dropNext = lockedOptionConsumesNext(tok, *o); continue; }
            if (!kept.empty()) kept += ' ';
            kept += tok;
        }
        if (!kept.empty()) out += kept + "\n";
    }
    return out;
}

// The hand-written part a GUI save keeps: the lines above the marker, unchanged; for a file with no
// marker yet, its locked options with their values (one per line), so paths set by hand survive.
inline std::string handWrittenSettings(const std::string& fileText) {
    const auto lines = detail::splitLines(fileText);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (isGuiSettingsMarker(lines[i])) {
            std::string out;
            for (std::size_t j = 0; j < i; ++j) out += lines[j] + "\n";
            return out;
        }
    }
    std::vector<std::string> toks;
    for (const auto& l : lines)
        for (auto& t : configLineTokens(l)) toks.push_back(t);
    std::string out;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const GuiLockedOption* o = guiLockedOption(toks[i]);
        if (!o) continue;
        std::string entry = toks[i];
        if (lockedOptionConsumesNext(toks[i], *o)) {
            if (i + 1 >= toks.size()) break;   // no value: dropping it is what the loader would see anyway
            entry += " " + toks[++i];
        }
        out += entry + "\n";
    }
    return out;
}

// Check settings text posted by the GUI. Returns an empty string and sets `cleaned` (the tokens to write
// below the marker, without comments or -password) when it is acceptable; otherwise a message for the user.
inline std::string checkGuiSettingsText(const std::string& body, std::string& cleaned) {
    cleaned.clear();
    if (body.size() > kMaxGuiSettingsBytes)
        return "Settings text is too long (limit " + std::to_string(kMaxGuiSettingsBytes) + " bytes).";
    for (char ch : body) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if ((c < 0x20 && c != '\n' && c != '\r' && c != '\t') || c == 0x7f) {
            std::ostringstream m;
            m << "Settings text contains a control character (0x" << std::hex << static_cast<int>(c) << ").";
            return m.str();
        }
    }
    std::set<std::string> locked;
    auto note = [&](const std::string& tok) {
        if (const GuiLockedOption* o = guiLockedOption(tok)) locked.insert(std::string("-") + o->name);
    };
    // The loader's tokens, comments included (a locked option there is refused too), and, to be safe
    // against any other notion of whitespace, the text split at every byte that is not printable ASCII.
    {
        std::istringstream iss(body);
        iss.imbue(std::locale::classic());
        std::string tok;
        while (iss >> tok) note(tok);
        std::string piece;
        for (char ch : body) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (c > 0x20 && c < 0x7f) piece += ch;
            else { note(piece); piece.clear(); }
        }
        note(piece);
    }
    if (!locked.empty()) {
        std::string names;
        for (const auto& n : locked) names += (names.empty() ? "" : ", ") + n;
        return "These options set file or directory paths or the GUI's network address and cannot be changed "
               "from the GUI: " + names + ". Pass them on the command line or edit the settings file by hand "
               "(above the GUI marker line).";
    }
    bool skipNext = false;
    for (const std::string& line : detail::splitLines(body)) {
        if (isGuiSettingsMarker(line)) continue;
        std::string kept;
        for (const std::string& tok : configLineTokens(line)) {
            if (skipNext) { skipNext = false; continue; }
            if (tok == "-password") { skipNext = true; continue; }   // never stored from the browser
            if (!kept.empty()) kept += ' ';
            kept += tok;
        }
        if (!kept.empty()) cleaned += kept + "\n";
    }
    return std::string();
}

// The new file content for a GUI save: the hand-written part of `existingFile`, the marker, `cleaned`.
inline std::string composeGuiSettingsFile(const std::string& existingFile, const std::string& cleaned) {
    return handWrittenSettings(existingFile) + guiSettingsMarker() + "\n" + cleaned;
}

}  // namespace util
