// Checks util::winCommandLine / winQuoteArg: exact strings for the tricky cases, and a round trip
// through an argv parser that follows the CommandLineToArgvW / MSVCRT rules.
//
// On Windows (or MinGW under Wine) it additionally round-trips through the real
// CommandLineToArgvW and through a real child process started with CreateProcessA, which exercises
// the C runtime's own argv parsing.
#include "util/WinCmdLine.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using Args = std::vector<std::string>;

// Argument parser for arguments after argv[0], following the documented CommandLineToArgvW rules:
// 2n backslashes + quote -> n backslashes, quote toggles; 2n+1 backslashes + quote -> n backslashes
// and a literal quote; backslashes not followed by a quote are literal; inside a quoted section
// a doubled quote yields one literal quote.
static Args parseArgs(const std::string& s) {
    Args out;
    size_t i = 0;
    const size_t n = s.size();
    while (true) {
        while (i < n && (s[i] == ' ' || s[i] == '\t')) ++i;
        if (i >= n) break;
        std::string cur;
        bool inQuotes = false;
        while (i < n && (inQuotes || (s[i] != ' ' && s[i] != '\t'))) {
            size_t bs = 0;
            while (i < n && s[i] == '\\') { ++bs; ++i; }
            if (i < n && s[i] == '"') {
                cur.append(bs / 2, '\\');
                if (bs % 2) {
                    cur.push_back('"');
                } else if (inQuotes && i + 1 < n && s[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    inQuotes = !inQuotes;
                }
                ++i;
            } else {
                cur.append(bs, '\\');
                if (bs == 0 && i < n) cur.push_back(s[i++]);   // with backslashes, re-check for a separator first
            }
        }
        out.push_back(cur);
    }
    return out;
}

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

static std::string show(const std::string& s) {
    std::string o = "[";
    o += s;
    o += "]";
    return o;
}

static void expectQuoted(const std::string& arg, const std::string& want) {
    const std::string got = util::winQuoteArg(arg);
    check(got == want, "winQuoteArg(" + show(arg) + ") = " + show(got) + ", expected " + show(want));
}

#ifdef _WIN32
static std::string toHex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string o;
    for (unsigned char c : s) { o.push_back(d[c >> 4]); o.push_back(d[c & 15]); }
    return o;
}

static int childMain(int argc, char** argv) {
    // argv[1] = "--child", argv[2] = output file, argv[3..] = arguments under test
    std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
    out << (argc - 3) << "\n";
    for (int i = 3; i < argc; ++i) out << toHex(argv[i]) << "\n";
    return 0;
}

static Args realParse(const std::string& cmd) {
    int n = 0;
    std::wstring w(cmd.begin(), cmd.end());
    LPWSTR* v = CommandLineToArgvW(w.c_str(), &n);
    Args out;
    for (int i = 0; i < n; ++i) {
        std::wstring a = v[i];
        out.emplace_back(a.begin(), a.end());
    }
    LocalFree(v);
    return out;
}

static bool childRoundTrip(const std::string& self, const Args& args, Args& got) {
    const std::string outFile = "wincmdline_child_out.txt";
    std::remove(outFile.c_str());
    Args full = {self, "--child", outFile};
    full.insert(full.end(), args.begin(), args.end());
    std::string cmd = util::winCommandLine(full);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');
    if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return false;
    WaitForSingleObject(pi.hProcess, 20000);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    std::ifstream in(outFile, std::ios::binary);
    int n = -1;
    if (!(in >> n)) return false;
    got.clear();
    std::string line;
    std::getline(in, line);
    for (int i = 0; i < n; ++i) {
        std::getline(in, line);
        std::string a;
        for (size_t k = 0; k + 1 < line.size(); k += 2) a.push_back(static_cast<char>(std::stoi(line.substr(k, 2), nullptr, 16)));
        got.push_back(a);
    }
    return true;
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc >= 3 && std::string(argv[1]) == "--child") return childMain(argc, argv);
#else
    (void)argc;
    (void)argv;
#endif

    // Exact output.
    expectQuoted("", "\"\"");
    expectQuoted("abc", "abc");
    expectQuoted("-d", "-d");
    expectQuoted("a b", "\"a b\"");
    expectQuoted("a\tb", "\"a\tb\"");
    expectQuoted("say \"hi\"", "\"say \\\"hi\\\"\"");
    expectQuoted("\"", "\"\\\"\"");
    expectQuoted("C:\\dir\\file", "C:\\dir\\file");               // backslashes alone need no quoting
    expectQuoted("C:\\dir\\", "C:\\dir\\");                       // trailing backslash, no space: untouched
    expectQuoted("C:\\Program Files\\x\\", "\"C:\\Program Files\\x\\\\\"");   // trailing backslash doubled before closing quote
    expectQuoted("a\\\"b", "\"a\\\\\\\"b\"");                      // backslash + quote: 2n+1 backslashes then the quote
    expectQuoted("a b\\\\", "\"a b\\\\\\\\\"");
    expectQuoted("\\\\server\\share name\\", "\"\\\\server\\share name\\\\\"");
    check(util::winQuoteProgram(std::string("C:\\Program Files\\prmers.exe")) == "\"C:\\Program Files\\prmers.exe\"", "program path");
    check(util::winCommandLine(Args{"C:\\p q\\prmers.exe", "-worktodo", "my file.txt", ""}) ==
              "\"C:\\p q\\prmers.exe\" -worktodo \"my file.txt\" \"\"",
          "command line");
    check(util::winQuoteArg(std::wstring(L"a b")) == L"\"a b\"", "wide string");

    // Round trips.
    const Args cases = {
        "", " ", "a", "a b", "  lead", "trail  ", "tab\there", "new\nline", "\"", "\"\"", "\"quoted\"",
        "a\"b", "a\\\"b", "a\\\\\"b", "a b\"", "a b\\", "a b\\\\", "a b\\\\\\", "\\", "\\\\", "\\\\\\",
        "C:\\Program Files (x86)\\Prime95\\", "C:\\Program Files (x86)\\Prime95\\prime95.exe",
        "\\\\server\\share dir\\sub dir\\", "-user", "name with \"quotes\" and \\backslashes\\ ", "ends with quote\"",
        "ends with backslash-quote \\\"", "^&|<>%PATH%", "100%", "a=b c=d", "'single quoted'", "x\\ y", "\"\\ \"",
    };
    for (const std::string& a : cases) {
        const std::string line = "prog.exe " + util::winQuoteArg(a);
        const Args parsed = parseArgs(line.substr(std::string("prog.exe").size()));
        check(parsed.size() == 1 && parsed[0] == a, "round trip of " + show(a) + " via " + show(line));
    }
    // All arguments together, including empty ones, keep their boundaries.
    {
        std::vector<std::string> full{"C:\\Program Files\\prmers\\prmers.exe"};
        full.insert(full.end(), cases.begin(), cases.end());
        const std::string line = util::winCommandLine(full);
        const std::string rest = line.substr(util::winQuoteProgram(full[0]).size());
        const Args back = parseArgs(rest);
        check(back == cases, "full command line round trip");
        if (back != cases) {
            std::cerr << "  got " << back.size() << " args, want " << cases.size() << "\n";
            for (size_t i = 0; i < back.size() && i < cases.size(); ++i)
                if (back[i] != cases[i]) { std::cerr << "  arg " << i << ": got " << show(back[i]) << " want " << show(cases[i]) << "\n"; break; }
        }
#ifdef _WIN32
        const Args real = realParse(line);
        Args expect = full;
        check(real == expect, "CommandLineToArgvW round trip");
        if (real != expect) {
            for (size_t i = 0; i < real.size() && i < expect.size(); ++i)
                if (real[i] != expect[i]) std::cerr << "  arg " << i << ": got " << show(real[i]) << " want " << show(expect[i]) << "\n";
        }
        Args got;
        const bool ran = childRoundTrip(argv[0], cases, got);
        check(ran, "child process ran");
        if (ran) {
            check(got == cases, "child process (CRT argv) round trip");
            if (got != cases) {
                for (size_t i = 0; i < got.size() && i < cases.size(); ++i)
                    if (got[i] != cases[i]) std::cerr << "  arg " << i << ": got " << show(got[i]) << " want " << show(cases[i]) << "\n";
            }
        }
#endif
    }

    // The old implementation (escape only the quotes) fails on trailing backslashes before the closing quote.
    {
        std::string old = "\"";
        for (char ch : std::string("C:\\Program Files\\x\\")) { if (ch == '"') old += '\\'; old.push_back(ch); }
        old += "\"";
        check(parseArgs(old) != Args{"C:\\Program Files\\x\\"}, "sanity: old quoting is really broken for trailing backslash");
    }

    if (g_failures) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Windows command-line quoting test passed\n";
    return 0;
}
