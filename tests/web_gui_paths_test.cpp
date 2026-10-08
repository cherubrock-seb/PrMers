// The web GUI must not be able to change file/directory paths or the GUI's network options:
// - POST /api/save-settings refuses path options in every spelling (quoting, --opt=value, case, split
//   across lines, hidden behind a NUL or a non-ASCII space) and keeps the hand-written part of the file;
// - the settings loader ignores such options below the GUI marker line;
// - POST /api/append-worktodo only accepts runnable worktodo entries (the real WorktodoParser check).
#include "io/WorktodoParser.hpp"
#include "ui/WebGuiServer.hpp"
#include "util/GuiSettings.hpp"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static int g_failures = 0;
static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) ++g_failures;
}

static std::string readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string joined(const std::vector<std::string>& v) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : " ") + s;
    return out;
}

static bool refused(const std::string& body, const std::string& mustMention = std::string()) {
    std::string cleaned;
    const std::string err = util::checkGuiSettingsText(body, cleaned);
    return !err.empty() && (mustMention.empty() || err.find(mustMention) != std::string::npos);
}

static bool accepted(const std::string& body, std::string* cleanedOut = nullptr) {
    std::string cleaned;
    const bool ok = util::checkGuiSettingsText(body, cleaned).empty();
    if (cleanedOut) *cleanedOut = cleaned;
    return ok;
}

static void testValidator() {
    std::cout << "settings text check\n";
    const char* lockedNames[] = {"-worktodo", "-config", "-f", "-kernelpath", "-kernel_path", "-kernel-path",
                                 "-output", "-filemers", "-p95path", "-host", "-http"};
    for (const char* n : lockedNames) {
        check(refused(std::string("-d 0 ") + n + " /home/user/.bashrc", n), std::string("refuses ") + n);
        check(refused(std::string("-") + n + "=/x"), std::string("refuses -") + n + "=/x");
        check(refused(std::string(n) + "=/x"), std::string("refuses ") + n + "=/x");
    }
    check(refused("-ipv4", "-ipv4"), "refuses -ipv4");
    check(refused("-WORKTODO x"), "refuses upper case");
    check(refused("-WorkToDo x"), "refuses mixed case");
    check(refused("\"-worktodo\" x"), "refuses double-quoted option");
    check(refused("'-f' x"), "refuses single-quoted option");
    check(refused("\"--worktodo=x\""), "refuses quoted --opt=value");
    check(refused("---worktodo x"), "refuses extra dashes");
    check(refused("-d 0\n-worktodo\n/home/user/.bashrc\n"), "refuses option and value on separate lines");
    check(refused("-d 0\r\n-f\r\n/tmp\r\n"), "refuses CRLF-separated option");
    check(refused("-d 0 -d 1 -f /a -f /b"), "refuses duplicated path options");
    check(refused("-user -f"), "refuses a locked name used as a value");
    check(refused("# -worktodo x"), "refuses a locked option inside a comment");
    check(refused("-d 0 #c\n-f x"), "refuses after a comment");
    check(refused(std::string("-worktodo\0x /etc/passwd", 23), "control"), "refuses NUL (C-string parser would see -worktodo)");
    check(refused("-d\x01" "0"), "refuses \\x01");
    check(refused("-d 0\x7f"), "refuses DEL");
    check(refused("-d 0\v-f x"), "refuses vertical tab");
    check(refused("-d 0\f-f x"), "refuses form feed");
    check(refused("-d 0\xC2\xA0-f\xC2\xA0/x"), "refuses -f behind non-breaking spaces");
    check(refused("-d 0\xE3\x80\x80-worktodo\xE3\x80\x80x"), "refuses -worktodo behind ideographic spaces");
    check(refused(std::string(util::kMaxGuiSettingsBytes + 1, ' '), "too long"), "refuses oversized body");
    check(accepted(std::string(util::kMaxGuiSettingsBytes - 6, ' ') + "-d 0\n"), "accepts body at the size limit");
    check(accepted("\xE2\x88\x92" "f /x"), "accepts U+2212 minus (CliParser does not read it as -f)");
    check(accepted("-user J\xC3\xB6rg -computer box"), "accepts UTF-8 in values");
    check(accepted("-d 0 -d 1 -t 60 -t 120"), "accepts duplicated non-path options");
    check(accepted("-build \"-cl-fast-relaxed-math\""), "accepts -build (not a path option)");
    check(accepted("-fft 1 -filemersx"), "does not match option prefixes");
    check(accepted(""), "accepts empty text");
    std::string cleaned;
    check(accepted("-d 0 -password hunter2 -t 60\n# comment\n\n-user me -password\nsecret\n", &cleaned) &&
              cleaned == "-d 0 -t 60\n-user me\n",
          "strips -password, comments and blank lines (" + cleaned + ")");
    check(accepted(util::guiSettingsMarker() + "\n-d 1\n", &cleaned) && cleaned == "-d 1\n", "drops a posted marker line");
}

static void testLoader() {
    std::cout << "settings loader\n";
    const std::string M = util::guiSettingsMarker() + "\n";
    auto args = [](const std::string& t) { return joined(util::readConfigArgsFromText(t).args); };
    check(args("-d 0 -worktodo w.txt\n-f /data # trailing\n# full line\n") == "-d 0 -worktodo w.txt -f /data",
          "file without marker: everything honoured, comments skipped");
    check(args("-f /data\n" + M + "-d 1 -worktodo /home/u/.bashrc -t 60\n") == "-f /data -d 1 -t 60",
          "GUI section: -worktodo and its value ignored");
    check(args(M + "-d 1 -f\n/etc\n-t 5\n") == "-d 1 -t 5", "GUI section: value on the next line ignored too");
    check(args(M + "-f=/etc --worktodo=x -KERNELPATH k \"-config\" c -ipv4 -host 0.0.0.0 -http 80 -d 2\n") == "-d 2",
          "GUI section: every spelling ignored");
    check(args(M + std::string("-worktodo\0z /x -d 3\n", 20)) == "-d 3", "GUI section: NUL-suffixed option ignored");
    check(args("-d 0 -f\n" + M + "/etc -t 1\n") == "-d 0 /etc -t 1",
          "hand-written -f without a value cannot take it from the GUI section");
    check(args("-user\n" + M + "-d 0\n") == "-user -d 0", "a non-path dangling option is left alone");
    check(args(M + "-d 1\n" + M + "-f /x\n") == "-d 1", "a second marker does not end the GUI section");
    check(args("-d 0\r\n" + util::guiSettingsMarker() + "\r\n-f /x\r\n-t 9\r\n") == "-d 0 -t 9", "CRLF file");
    const auto r = util::readConfigArgsFromText(M + "-f /x -d 1\n");
    check(joined(r.ignored) == "-f /x", "ignored options are reported");
}

static void testCompose() {
    std::cout << "settings save/load round trip\n";
    const std::string hand = "-d 0 -worktodo mine.txt -t 60\n-f\n/data/prmers\n";
    check(util::guiEditableSettings(hand) == "-d 0 -t 60\n", "editable view omits path options");
    check(util::handWrittenSettings(hand) == "-worktodo mine.txt\n-f /data/prmers\n", "kept part: path options with values");
    const std::string first = util::composeGuiSettingsFile(hand, "-d 1 -t 120\n");
    check(joined(util::readConfigArgsFromText(first).args) == "-worktodo mine.txt -f /data/prmers -d 1 -t 120",
          "first GUI save keeps hand-written paths");
    const std::string handEdited = "# my paths\n-worktodo mine.txt -config x.cfg\n" + util::guiSettingsMarker() + "\n-d 1\n";
    const std::string second = util::composeGuiSettingsFile(handEdited, "-d 2\n");
    check(second == "# my paths\n-worktodo mine.txt -config x.cfg\n" + util::guiSettingsMarker() + "\n-d 2\n",
          "later GUI saves copy the part above the marker unchanged");
    check(util::guiEditableSettings(second) == "-d 2\n", "editable view is the GUI section");
    check(util::guiEditableSettings(util::guiSettingsMarker() + "\n-d 2 -f /x\n") == "-d 2\n",
          "editable view hides a hand-added path option below the marker");
    check(util::handWrittenSettings("-d 0 -f") == "", "dangling locked option is not kept");
}

static void testWorktodoLine() {
    std::cout << "worktodo line check\n";
    using io::WorktodoParser;
    check(WorktodoParser::isValidEntryLine("PRP=1,2,127,-1"), "PRP accepted");
    check(WorktodoParser::isValidEntryLine("Test=1,2,127,-1"), "Test accepted");
    check(WorktodoParser::isValidEntryLine("Pminus1=1,2,1277,-1,1000,50000"), "Pminus1 accepted");
    check(!WorktodoParser::isValidEntryLine(""), "empty refused");
    check(!WorktodoParser::isValidEntryLine("# PRP=1,2,127,-1"), "comment refused");
    check(!WorktodoParser::isValidEntryLine("hello world"), "text refused");
    check(!WorktodoParser::isValidEntryLine("Factor=1,2,127,-1"), "unsupported keyword refused");
    check(!WorktodoParser::isValidEntryLine("PRP=1,2,127,-1\nPRP=1,2,521,-1"), "two lines refused");
    check(!WorktodoParser::isValidEntryLine("PRP=1,2,127,-1\rexport X=1"), "embedded CR refused");
}

// ---- HTTP ---------------------------------------------------------------------------------------------
static int g_port = 0;
static std::string g_auth;

static int request(const std::string& raw, std::string* bodyOut = nullptr) {
    int s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(g_port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { ::close(s); throw std::runtime_error("connect"); }
    size_t off = 0;
    while (off < raw.size()) {
        const ssize_t n = send(s, raw.data() + off, raw.size() - off, 0);
        if (n <= 0) break;
        off += static_cast<size_t>(n);
    }
    std::string resp;
    char buf[4096];
    for (;;) {
        const ssize_t n = recv(s, buf, sizeof(buf), 0);
        if (n <= 0) break;
        resp.append(buf, buf + n);
    }
    ::close(s);
    if (resp.rfind("HTTP/1.1 ", 0) != 0) return 0;
    if (bodyOut) {
        const auto p = resp.find("\r\n\r\n");
        *bodyOut = p == std::string::npos ? std::string() : resp.substr(p + 4);
    }
    return std::stoi(resp.substr(9, 3));
}

static int get(const std::string& path, std::string* body = nullptr) {
    return request("GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(g_port) + "\r\n" + g_auth + "\r\n", body);
}

static int post(const std::string& path, const std::string& payload, std::string* body = nullptr) {
    return request("POST " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(g_port) + "\r\n" + g_auth +
                   "Content-Type: text/plain\r\nContent-Length: " + std::to_string(payload.size()) + "\r\n\r\n" + payload, body);
}

static void testHttp() {
    std::cout << "HTTP\n";
    const std::string cfgPath = "gui_paths_settings.cfg";
    const std::string handWritten = "-d 0 -worktodo gui_paths_worktodo.txt -t 600\n";
    std::ofstream(cfgPath) << handWritten;
    std::ofstream("gui_paths_worktodo.txt") << "PRP=1,2,127,-1\n";

    ui::WebGuiConfig cfg;
    cfg.port = 0;
    cfg.bind_host = "127.0.0.1";
    cfg.config_path = cfgPath;
    cfg.results_path = "gui_paths_results.txt";
    cfg.worktodo_path = "gui_paths_worktodo.txt";
    cfg.save_path = "/srv/prmers-save";
    cfg.kernel_path = "/opt/prmers/kernels/prmers.cl";
    cfg.worktodo_line_ok = [](const std::string& l) { return io::WorktodoParser::isValidEntryLine(l); };
    std::mutex m;
    std::vector<std::string> submitted;
    ui::WebGuiServer server(cfg, [&](const std::string& s) { std::lock_guard<std::mutex> lk(m); submitted.push_back(s); });
    if (!server.start()) throw std::runtime_error("GUI did not start");
    const std::string url = server.url();
    g_port = std::stoi(url.substr(url.rfind(':') + 1));
    g_auth = "X-PrMers-Token: " + url.substr(url.find("token=") + 6) + "\r\n";

    std::string body;
    // The attack: point -worktodo at another file, then append to it after the restart.
    check(post("/api/save-settings", "-d 0 -worktodo /home/user/.bashrc", &body) == 400 &&
              body.find("-worktodo") != std::string::npos,
          "save-settings refuses -worktodo with a message naming it: " + body);
    check(readAll(cfgPath) == handWritten, "refused save leaves the file unchanged");
    check(post("/api/save-settings", "-d 0\n--f=/tmp/evil -kernelpath /tmp/k.cl -config /tmp/x.cfg -host 0.0.0.0", &body) == 400 &&
              body.find("-config") != std::string::npos && body.find("-f") != std::string::npos &&
              body.find("-host") != std::string::npos && body.find("-kernelpath") != std::string::npos,
          "save-settings lists every refused option: " + body);
    check(post("/api/save-settings", std::string("-worktodo\0x /etc/passwd", 23), &body) == 400, "save-settings refuses NUL");
    check(post("/api/save-settings", std::string(util::kMaxGuiSettingsBytes + 1, 'a'), &body) == 400, "save-settings refuses oversized text");
    check(readAll(cfgPath) == handWritten, "file still unchanged");

    check(get("/api/load-settings", &body) == 200 && body == "-d 0 -t 600\n", "load-settings omits path options: " + body);
    check(post("/api/save-settings", body + "-t 300 -password hunter2\n") == 200, "save-settings accepts the edited text");
    const std::string saved = readAll(cfgPath);
    check(saved.find(util::guiSettingsMarker()) != std::string::npos, "saved file has the GUI marker");
    check(saved.find("hunter2") == std::string::npos, "password not stored");
    check(joined(util::readConfigArgsFromText(saved).args) == "-worktodo gui_paths_worktodo.txt -d 0 -t 600 -t 300",
          "hand-written -worktodo survives a GUI save: " + joined(util::readConfigArgsFromText(saved).args));
    // A file edited by hand to add a path option below the marker: the loader ignores it.
    std::ofstream(cfgPath, std::ios::app) << "-worktodo /home/user/.bashrc\n";
    const auto loaded = util::readConfigArgsFromText(readAll(cfgPath));
    check(joined(loaded.ignored) == "-worktodo /home/user/.bashrc", "loader ignores a path option below the marker");

    check(get("/api/paths", &body) == 200 && body.find("\"worktodo\":\"gui_paths_worktodo.txt\"") != std::string::npos &&
              body.find("/srv/prmers-save") != std::string::npos && body.find("/opt/prmers/kernels/prmers.cl") != std::string::npos,
          "paths endpoint reports the current paths: " + body);
    const std::string token = url.substr(url.find("token=") + 6);
    check(get("/?token=" + token, &body) == 200 && body.find("id=opt_wt") == std::string::npos &&
              body.find("id=opt_f ") == std::string::npos && body.find("id=opt_kpath") == std::string::npos &&
              body.find("id=opt_out") == std::string::npos && body.find("id=path_worktodo") != std::string::npos &&
              body.find("'-worktodo'") == std::string::npos && body.find("'-kernelpath'") == std::string::npos,
          "page shows paths read-only and generates no path options");
    // The current worktodo is shown separately: Append & Run must not post the whole file back.
    check(body.find("id=wtcur") != std::string::npos && body.find("$('#wt').value=t") == std::string::npos,
          "page does not preload the worktodo into the append box");
    if (const char* dump = std::getenv("GUI_PAGE_DUMP")) std::ofstream(dump) << body;

    // append-worktodo
    auto nSubmitted = [&]() { std::lock_guard<std::mutex> lk(m); return submitted.size(); };
    check(post("/api/append-worktodo", "PRP=1,2,127,-1") == 200 && nSubmitted() == 1 && submitted.back() == "PRP=1,2,127,-1",
          "append accepts a PRP entry");
    check(post("/api/append-worktodo", "\r\n  PRP=1,2,521,-1  \r\n\r\nTest=1,2,607,-1\r\n") == 200 && nSubmitted() == 2 &&
              submitted.back() == "PRP=1,2,521,-1\nTest=1,2,607,-1",
          "append accepts CRLF lines and trims them");
    const size_t before = nSubmitted();
    struct Bad { std::string body; const char* what; };
    const Bad bad[] = {
        {"hello", "free text"},
        {"PRP=1,2,127,-1\nexport PATH=/tmp:$PATH", "a valid line followed by shell text"},
        {"# PRP=1,2,127,-1", "a comment"},
        {"Factor=1,2,127,-1", "an unsupported keyword"},
        {"PRP=1,2,127,-1\rTest=1,2,521,-1", "an embedded CR"},
        {"PRP=1,2,127,-1\t", "a tab"},
        {std::string("PRP=1,2,127,-1\0x", 16), "a NUL"},
        {"PRP=1,2,127,-1 \xC3\xA9", "non-ASCII"},
        {"\n \r\n", "only blank lines"},
        {std::string(8 * 1024 + 1, ' '), "an oversized body"},
        {"PRP=1,2," + std::string(1100, '7') + ",-1", "an overlong line"},
    };
    for (const auto& b : bad) check(post("/api/append-worktodo", b.body, &body) == 400, std::string("append refuses ") + b.what + ": " + body);
    std::string many;
    for (int i = 0; i < 65; ++i) many += "PRP=1,2,127,-1\n";
    check(post("/api/append-worktodo", many, &body) == 400, "append refuses more than 64 lines: " + body);
    check(nSubmitted() == before, "nothing refused reached the worktodo");

    server.stop();
    std::remove(cfgPath.c_str());
    std::remove("gui_paths_worktodo.txt");
}

int main() {
    testValidator();
    testLoader();
    testCompose();
    testWorktodoLine();
    testHttp();
    if (g_failures) {
        std::cout << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "Web GUI path lock test passed\n";
    return 0;
}
