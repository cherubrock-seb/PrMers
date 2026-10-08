#include "ui/WebGuiServer.hpp"
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <csignal>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <random>
#include "util/GuiSettings.hpp"
#include "util/Redact.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
#ifndef _WIN32
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#endif
#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#endif

namespace ui {

static std::shared_ptr<WebGuiServer> g_instance;

static constexpr size_t kMaxHeaderBytes = 16 * 1024;
static constexpr size_t kMaxBodyBytes = 1024 * 1024;
static constexpr int kMaxConnections = 32;
static constexpr int kSocketTimeoutSeconds = 10;
// /api/append-worktodo limits: a few entries per request, each a normal worktodo line.
static constexpr size_t kMaxAppendBytes = 8 * 1024;
static constexpr size_t kMaxAppendLines = 64;
static constexpr size_t kMaxAppendLineBytes = 1024;

static bool validToken(const std::string& t) {
    if (t.size() < 16 || t.size() > 128) return false;
    for (char c : t) if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') return false;
    return true;
}

static std::string makeToken() {
    std::random_device rd;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 4; ++i) oss << std::setw(8) << static_cast<uint32_t>(rd());
    return oss.str();
}

static std::string toLower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::string percentDecode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') out += ' ';
        else if (s[i] == '%' && i + 2 < s.size() && std::isxdigit((unsigned char)s[i+1]) && std::isxdigit((unsigned char)s[i+2])) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else out += s[i];
    }
    return out;
}

static std::string queryParam(const std::string& query, const std::string& key) {
    std::istringstream iss(query);
    std::string kv;
    while (std::getline(iss, kv, '&')) {
        auto eq = kv.find('=');
        if (eq != std::string::npos && kv.substr(0, eq) == key) return percentDecode(kv.substr(eq + 1));
    }
    return {};
}

// Host part of a Host header or authority: "127.0.0.1:3131" -> "127.0.0.1", "[::1]:3131" -> "[::1]".
static std::string hostPart(const std::string& authority) {
    if (!authority.empty() && authority[0] == '[') {
        auto e = authority.find(']');
        return e == std::string::npos ? authority : authority.substr(0, e + 1);
    }
    return authority.substr(0, authority.find(':'));
}

static void setSocketTimeouts(int fd) {
#ifdef _WIN32
    u_long nb = 0; ioctlsocket((SOCKET)fd, FIONBIO, &nb);   // accepted sockets inherit FIONBIO from the listener
    DWORD ms = kSocketTimeoutSeconds * 1000;
    setsockopt((SOCKET)fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof(ms));
    setsockopt((SOCKET)fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof(ms));
#else
    // Linux does not pass O_NONBLOCK from the (non-blocking) listening socket to the accepted one, but
    // macOS and the BSDs do. A non-blocking accepted socket makes recv() fail with EAGAIN whenever the
    // request bytes have not arrived yet, which readRequest() treats as the end of the stream.
    const int fl = fcntl(fd, F_GETFL, 0);
    if (fl != -1 && (fl & O_NONBLOCK)) fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
    timeval tv{}; tv.tv_sec = kSocketTimeoutSeconds;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

WebGuiServer::WebGuiServer(const WebGuiConfig& cfg, SubmitFn onSubmit, StopFn onStop)
: cfg_(cfg), onSubmit_(std::move(onSubmit)), onStop_(std::move(onStop)) {
    // Reuse the token of the process we were restarted from (restart_self after "Append & Run"),
    // so an open browser tab keeps working; otherwise make a new one.
    const char* inherited = std::getenv("PRMERS_GUI_TOKEN");
    token_ = (inherited && validToken(inherited)) ? std::string(inherited) : makeToken();
#ifdef _WIN32
    _putenv_s("PRMERS_GUI_TOKEN", token_.c_str());
#else
    setenv("PRMERS_GUI_TOKEN", token_.c_str(), 1);
#endif
}
WebGuiServer::~WebGuiServer() { stop(); }

std::shared_ptr<WebGuiServer> WebGuiServer::instance() { return g_instance; }
void WebGuiServer::setInstance(std::shared_ptr<WebGuiServer> s) { g_instance = std::move(s); }

static std::string firstLanIPv4() {
#ifndef _WIN32
    ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) return {};
    std::string ip;
    for (auto* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
        if (!ifa || !ifa->ifa_addr) continue;
        if (ifa->ifa_addr->sa_family != AF_INET) continue;
        if (ifa->ifa_flags & IFF_LOOPBACK) continue;
        char host[NI_MAXHOST];
        if (getnameinfo(ifa->ifa_addr, sizeof(sockaddr_in), host, sizeof(host), nullptr, 0, NI_NUMERICHOST) == 0) {
            ip = host; break;
        }
    }
    freeifaddrs(ifaddr);
    return ip;
#else
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname)) != 0) return {};
    addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(hostname, nullptr, &hints, &res) != 0) return {};
    std::string ip;
    for (auto* p = res; p; p = p->ai_next) {
        auto* sa = (sockaddr_in*)p->ai_addr;
        uint32_t a = ntohl(sa->sin_addr.s_addr);
        if ((a >> 24) == 127) continue; // skip loopback
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf))) { ip = buf; break; }
    }
    freeaddrinfo(res);
    return ip;
#endif
}


bool WebGuiServer::start() {
    if (running_) return true;
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    std::string host;
    if (cfg_.lanipv4) {
        host = firstLanIPv4();
    } else if (!cfg_.advertise_host.empty()) {
        host = cfg_.advertise_host;
    } else if (cfg_.bind_host.empty() || cfg_.bind_host == "0.0.0.0") {
        host = "127.0.0.1";
    } else {
        host = cfg_.bind_host;
    }
    url_ = "http://" + host + ":" + std::to_string(cfg_.port) + "/";


    listen_fd_ = createListenSocket(host, cfg_.port, cfg_.port);
    if (listen_fd_ < 0) {
#ifdef _WIN32
        std::cerr << "Error: cannot start the GUI on " << host << ":" << cfg_.port
                  << " (socket error " << WSAGetLastError() << ")" << std::endl;
#else
        std::cerr << "Error: cannot start the GUI on " << host << ":" << cfg_.port
                  << ": " << std::strerror(errno) << std::endl;
#endif
        return false;
    }
    url_ = "http://" + host + ":" + std::to_string(cfg_.port) + "/";
    //std::string host = firstLanIPv4();
    if (host.empty()) host = "127.0.0.1";    // fallback
    url_ = std::string("http://") + host + ":" + std::to_string(cfg_.port) + "/?token=" + token_;

    running_ = true;
    thr_ = std::thread([this]{ run(); });
    return true;
}


void WebGuiServer::stop() {
    if (!running_) return;
    running_ = false;
    closeListen();
    if (thr_.joinable()) thr_.join();
#ifdef _WIN32
    WSACleanup();
#endif
}

std::string WebGuiServer::url() const { return url_; }

void WebGuiServer::setStatus(const std::string& s) {
    std::lock_guard<std::mutex> lk(mtx_);
    st_.status = s;
    if (st_.logs.size() > 2000) st_.logs.pop_front();
    st_.logs.push_back(s);
}

void WebGuiServer::setProgress(uint64_t current, uint64_t total, const std::string& res64) {
    std::lock_guard<std::mutex> lk(mtx_);
    st_.cur = current;
    st_.tot = total;
    st_.res64 = res64;
}

void WebGuiServer::setBackendInfo(const std::string& mode,
                                  const std::string& active,
                                  const std::string& workload,
                                  const std::string& detail,
                                  uint64_t aevum_transform,
                                  uint64_t marin_transform,
                                  const std::string& fft_spec) {
    std::lock_guard<std::mutex> lk(mtx_);
    const bool changed = st_.backend_mode != mode || st_.backend_active != active ||
                         st_.backend_workload != workload || st_.backend_detail != detail ||
                         st_.backend_fft != fft_spec ||
                         st_.backend_aevum_transform != aevum_transform ||
                         st_.backend_marin_transform != marin_transform;
    st_.backend_mode = mode;
    st_.backend_active = active;
    st_.backend_workload = workload;
    st_.backend_detail = detail;
    st_.backend_fft = fft_spec;
    st_.backend_aevum_transform = aevum_transform;
    st_.backend_marin_transform = marin_transform;
    if (changed) {
        if (st_.logs.size() > 2000) st_.logs.pop_front();
        std::ostringstream line;
        line << "[Backend UI] " << mode << " -> " << active;
        if (!workload.empty()) line << " | " << workload;
        if (!detail.empty()) line << " | " << detail;
        st_.logs.push_back(line.str());
    }
}

void WebGuiServer::appendLog(const std::string& line) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (st_.logs.size() > 2000) st_.logs.pop_front();
    st_.logs.push_back(line);
}

void WebGuiServer::run() {
#ifdef _WIN32
    u_long nb = 1; ioctlsocket((SOCKET)listen_fd_, FIONBIO, &nb);
    while (running_) {
        fd_set rfds; FD_ZERO(&rfds); FD_SET((SOCKET)listen_fd_, &rfds);
        timeval tv; tv.tv_sec = 0; tv.tv_usec = 200000;
        int r = select((int)listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
        if (r <= 0) continue;
        for (;;) {
            SOCKET cfd = accept((SOCKET)listen_fd_, nullptr, nullptr);
            if (cfd == INVALID_SOCKET) {
                int e = WSAGetLastError();
                if (e == WSAEWOULDBLOCK || e == WSAEINTR) break;
                break;
            }
            if (active_connections_.load() >= kMaxConnections) { closesocket(cfd); continue; }
            setSocketTimeouts((int)cfd);
            ++active_connections_;
            std::thread([this, cfd]{ serveOne((int)cfd); closesocket(cfd); --active_connections_; }).detach();
        }
    }
#else
    struct pollfd pfd; pfd.fd = listen_fd_; pfd.events = POLLIN; pfd.revents = 0;
    while (running_) {
        int r = ::poll(&pfd, 1, 200);
        if (r < 0) { if (errno == EINTR) continue; else break; }
        if (r == 0) continue;
        for (;;) {
            int cfd = ::accept(listen_fd_, nullptr, nullptr);
            if (cfd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) break;
                break;
            }
            if (active_connections_.load() >= kMaxConnections) { ::close(cfd); continue; }
            setSocketTimeouts(cfd);
            ++active_connections_;
            std::thread([this, cfd]{ serveOne(cfd); ::close(cfd); --active_connections_; }).detach();
        }
    }
#endif
}


void WebGuiServer::closeListen() {
#ifdef _WIN32
    if (listen_fd_ != -1) { closesocket((SOCKET)listen_fd_); listen_fd_ = -1; }
#else
    if (listen_fd_ != -1) { ::close(listen_fd_); listen_fd_ = -1; }
#endif
}
int WebGuiServer::createListenSocket(const std::string& bind_host, int port, int& out_port) {
    int fd;
#ifdef _WIN32
    fd = (int)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == (int)INVALID_SOCKET) return -1;
    { BOOL yes = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes)); }
#else
    fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, (char*)&yes, sizeof(yes));
#endif
    int fl = fcntl(fd, F_GETFD, 0); if (fl != -1) fcntl(fd, F_SETFD, fl | FD_CLOEXEC);
    int fl2 = fcntl(fd, F_GETFL, 0); if (fl2 != -1) fcntl(fd, F_SETFL, fl2 | O_NONBLOCK);
#endif

    sockaddr_in addr; std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);

    in_addr ip{}; std::string h = bind_host;
    if (h.empty() || h == "localhost") h = "127.0.0.1";
    if (h == "0.0.0.0") ip.s_addr = htonl(INADDR_ANY);
#ifdef _WIN32
    else if (InetPtonA(AF_INET, h.c_str(), &ip) != 1) ip.s_addr = htonl(INADDR_LOOPBACK);
#else
    else if (inet_pton(AF_INET, h.c_str(), &ip) != 1) ip.s_addr = htonl(INADDR_LOOPBACK);
#endif
    if (h != "0.0.0.0" && ip.s_addr == htonl(INADDR_LOOPBACK) && h != "127.0.0.1") {
        // Not an IPv4 address literal: only a numeric address is understood here, so the GUI listens on
        // the loopback interface only, whatever name the URL advertises.
        std::cerr << "Warning: GUI host '" << h << "' is not an IPv4 address; listening on 127.0.0.1 only." << std::endl;
    }
    addr.sin_addr = ip;

    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { 
#ifdef _WIN32
        closesocket((SOCKET)fd);
#else
        const int savedErrno = errno;
        ::close(fd);
        errno = savedErrno;
#endif
        return -1; 
    }
    if (::listen(fd, 64) < 0) { 
#ifdef _WIN32
        closesocket((SOCKET)fd);
#else
        const int savedErrno = errno;
        ::close(fd);
        errno = savedErrno;
#endif
        return -1; 
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(fd, (sockaddr*)&addr, &len) == 0) out_port = ntohs(addr.sin_port); else out_port = port;
    return fd;
}



std::string WebGuiServer::headerValue(const std::string& headers, const std::string& name) {
    const std::string want = toLower(name);
    size_t pos = 0;
    while (pos < headers.size()) {
        size_t eol = headers.find("\r\n", pos);
        if (eol == std::string::npos) eol = headers.size();
        if (eol == pos) break;                                  // blank line: end of headers
        const std::string line = headers.substr(pos, eol - pos);
        const size_t colon = line.find(':');
        if (colon != std::string::npos && toLower(line.substr(0, colon)) == want) {
            std::string v = line.substr(colon + 1);
            const size_t a = v.find_first_not_of(" \t");
            const size_t b = v.find_last_not_of(" \t");
            return a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
        }
        pos = eol + 2;
    }
    return {};
}

int WebGuiServer::readRequest(int fd, std::string& method, std::string& path, std::string& body, std::string& headers) {
    std::string req;
    char buf[4096];
    size_t hend = std::string::npos;
    for (;;) {
#ifdef _WIN32
        int n = recv(fd, buf, sizeof(buf), 0);
#else
        int n = (int)::recv(fd, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        req.append(buf, buf + n);
        hend = req.find("\r\n\r\n");
        if (hend != std::string::npos) break;
        if (req.size() > kMaxHeaderBytes) return 431;
    }
    if (hend == std::string::npos) return req.empty() ? -1 : 400;
    if (hend > kMaxHeaderBytes) return 431;
    size_t pos = req.find("\r\n");
    std::string start = req.substr(0, pos);
    headers = req.substr(pos + 2, hend + 4 - (pos + 2));
    size_t p2 = start.find(' ');
    if (p2 == std::string::npos) return 400;
    size_t p3 = start.find(' ', p2 + 1);
    if (p3 == std::string::npos) return 400;
    method = start.substr(0, p2);
    path = start.substr(p2 + 1, p3 - p2 - 1);

    size_t contentLength = 0;
    const std::string cl = headerValue(headers, "Content-Length");
    if (!cl.empty()) {
        if (cl.size() > 9 || cl.find_first_not_of("0123456789") != std::string::npos) return 400;
        contentLength = (size_t)std::stoul(cl);
        if (contentLength > kMaxBodyBytes) return 413;
    }
    std::string tail = req.substr(hend + 4);
    if (tail.size() > contentLength) tail.resize(contentLength);
    while (tail.size() < contentLength) {
        const size_t to_read = std::min(contentLength - tail.size(), sizeof(buf));
#ifdef _WIN32
        int n = ::recv(fd, buf, (int)to_read, 0);
#else
        ssize_t n = ::recv(fd, buf, to_read, 0);
#endif
        if (n <= 0) return -1;
        tail.append(buf, buf + n);
    }
    body = tail;
    return 0;
}

bool WebGuiServer::hostAllowed(const std::string& hostHeader) const {
    // Reject DNS names other than localhost and the configured host, so a rebinding attack (an attacker's
    // name that resolves to 127.0.0.1) cannot reach the API. Numeric IPv4 addresses are always fine.
    const std::string h = toLower(hostPart(hostHeader));
    if (h.empty()) return false;
    if (h == "localhost" || h == "[::1]") return true;
    in_addr a{};
#ifdef _WIN32
    if (InetPtonA(AF_INET, h.c_str(), &a) == 1) return true;
#else
    if (inet_pton(AF_INET, h.c_str(), &a) == 1) return true;
#endif
    return (!cfg_.bind_host.empty() && h == toLower(cfg_.bind_host)) ||
           (!cfg_.advertise_host.empty() && h == toLower(cfg_.advertise_host));
}

bool WebGuiServer::originAllowed(const std::string& origin) const {
    for (const char* scheme : {"http://", "https://"}) {
        const std::string sc = scheme;
        if (origin.rfind(sc, 0) == 0) return hostAllowed(origin.substr(sc.size()));
    }
    return false;                                               // includes "null"
}

bool WebGuiServer::tokenMatches(const std::string& candidate) const {
    if (candidate.size() != token_.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < candidate.size(); ++i) diff |= (unsigned char)(candidate[i] ^ token_[i]);
    return diff == 0;
}

bool WebGuiServer::resolveResultsPath(const std::string& requested, std::string& resolved) const {
    // The results view may show other .txt/.json files next to the configured results file, nothing else.
    namespace fs = std::filesystem;
    if (requested.empty()) { resolved = cfg_.results_path; return true; }
    std::error_code ec;
    const fs::path base = fs::weakly_canonical(fs::absolute(cfg_.results_path, ec), ec).parent_path();
    if (ec) return false;
    fs::path cand = requested;
    if (cand.is_relative()) cand = base / cand;
    cand = fs::weakly_canonical(cand, ec);
    if (ec || cand.parent_path() != base) return false;
    const std::string ext = toLower(cand.extension().string());
    if (ext != ".txt" && ext != ".json") return false;
    resolved = cand.string();
    return true;
}

bool WebGuiServer::sendAll(int fd, const char* data, size_t len) {
    size_t off = 0;
#ifdef _WIN32
    while (off < len) {
        int n = send(fd, data + off, (int)(len - off), 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
#else
    while (off < len) {
        int n = (int)::send(fd, data + off, len - off, 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
#endif
}

void WebGuiServer::serveOne(int fd) {
    std::string method, path, body, headers;
    const int rc = readRequest(fd, method, path, body, headers);
    if (rc < 0) return;
    std::string resp;
    if (rc != 0) {
        resp = httpError(rc, "bad-request");
        sendAll(fd, resp.data(), resp.size());
        return;
    }
    const auto qpos = path.find('?');
    const std::string route = path.substr(0, qpos);
    const std::string query = qpos == std::string::npos ? std::string() : path.substr(qpos + 1);
    const std::string origin = headerValue(headers, "Origin");

    if (!hostAllowed(headerValue(headers, "Host")) || (!origin.empty() && !originAllowed(origin))) {
        resp = httpError(403, "forbidden-host");
    } else if (route.rfind("/api/", 0) == 0 && !tokenMatches(headerValue(headers, "X-PrMers-Token"))) {
        resp = httpError(401, "missing-or-invalid-token");
    } else if (method == "GET" && (route == "/" || route == "/index.html")) {
        if (tokenMatches(queryParam(query, "token"))) {
            resp = httpOk("text/html; charset=utf-8", htmlPage());
        } else {
            resp = httpError(401, "Open the GUI with the URL PrMers prints at startup; it includes ?token=...");
        }
    } else if (method == "GET" && route == "/api/state") {
        resp = httpOk("application/json", handleStateJson());
    } else if (method == "GET" && route == "/api/results") {
        size_t limit = 100;
        const std::string lim = queryParam(query, "limit");
        if (!lim.empty()) {
            size_t n = (size_t)std::strtoul(lim.c_str(), nullptr, 10);
            if (n > 0) limit = n;
        }
        std::string resultsPath;
        if (resolveResultsPath(queryParam(query, "path"), resultsPath)) {
            resp = httpOk("application/json", handleResultsJson(limit, resultsPath));
        } else {
            resp = httpError(400, "path-not-allowed");
        }
    } else if (method == "GET" && route == "/api/load-settings") {
        resp = httpOk("text/plain; charset=utf-8", handleLoadSettings());
    } else if (method == "GET" && route == "/api/load-worktodo") {
        resp = httpOk("text/plain; charset=utf-8", handleLoadWorktodo());
    } else if (method == "GET" && route == "/api/paths") {
        resp = httpOk("application/json", handlePathsJson());
    } else if (method == "POST" && route == "/api/save-settings") {
        int status = 400;
        const std::string err = handleSaveSettings(body, status);
        resp = err.empty() ? httpOk("application/json", "{\"ok\":true}") : httpError(status, err);
    } else if (method == "POST" && route == "/api/append-worktodo") {
        std::string lines;
        const std::string err = checkWorktodoAppend(body, lines);
        if (!err.empty()) {
            resp = httpBadRequest(err);
        } else {
            if (onSubmit_) onSubmit_(lines);
            resp = httpOk("application/json", "{\"ok\":true}");
        }
    } else if (method == "POST" && route == "/api/stop") {
        bool ok = handleStop();
        resp = ok ? httpOk("application/json", "{\"ok\":true}") : httpBadRequest("stop-failed");
    } else {
        resp = httpNotFound();
    }
    sendAll(fd, resp.data(), resp.size());
}

std::string WebGuiServer::stateJson() {
    return handleStateJson();
}

std::string WebGuiServer::handleStateJson() {
    State s;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        s = st_;
    }
    double pct = 0.0;
    if (s.tot > 0) pct = 100.0 * (double)s.cur / (double)s.tot;
    std::ostringstream oss;
    oss << "{";
    oss << "\"status\":\"" << jsonEscape(s.status) << "\",";
    oss << "\"current\":" << s.cur << ",";
    oss << "\"total\":" << s.tot << ",";
    oss << "\"percent\":" << (int)(pct+0.5) << ",";
    oss << "\"res64\":\"" << jsonEscape(s.res64) << "\",";
    oss << "\"backend_mode\":\"" << jsonEscape(s.backend_mode) << "\",";
    oss << "\"backend_active\":\"" << jsonEscape(s.backend_active) << "\",";
    oss << "\"backend_workload\":\"" << jsonEscape(s.backend_workload) << "\",";
    oss << "\"backend_detail\":\"" << jsonEscape(s.backend_detail) << "\",";
    oss << "\"backend_fft\":\"" << jsonEscape(s.backend_fft) << "\",";
    oss << "\"backend_aevum_transform\":" << s.backend_aevum_transform << ",";
    oss << "\"backend_marin_transform\":" << s.backend_marin_transform << ",";
    oss << "\"logs\":[";
    bool first = true;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& l : st_.logs) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << jsonEscape(l) << "\"";
        }
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

std::vector<std::string> WebGuiServer::tailLines(const std::string& path, size_t limit) {
    std::ifstream f(path);
    std::vector<std::string> lines;
    if (!f.is_open()) return lines;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    if (lines.size() > limit) {
        return std::vector<std::string>(lines.end() - (long)limit, lines.end());
    }
    return lines;
}

std::string WebGuiServer::handleResultsJson(size_t limit, const std::string& pathOverride) {
    std::string p = pathOverride.empty() ? cfg_.results_path : pathOverride;
    auto lines = tailLines(p, limit);
    std::ostringstream oss;
    oss << "{\"lines\":[";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) oss << ",";
        oss << "\"" << jsonEscape(lines[i]) << "\"";
    }
    oss << "]}";
    return oss.str();
}

std::string WebGuiServer::readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Write a sibling temporary file and rename it over `path`, so a failed write (full disk, I/O error)
// leaves the old file intact instead of a truncated one. A symlink is followed: its target is replaced.
bool WebGuiServer::writeFileReplacing(const std::string& path, const std::string& data) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path target = path;
    if (fs::is_symlink(target, ec)) {
        const fs::path resolved = fs::canonical(target, ec);
        if (ec) return false;
        target = resolved;
    }
    const fs::path tmp = target.string() + ".gui-tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return false;
        f.write(data.data(), (std::streamsize)data.size());
        f.close();
        if (!f) { fs::remove(tmp, ec); return false; }
    }
    fs::rename(tmp, target, ec);
    if (ec) { std::error_code ec2; fs::remove(tmp, ec2); return false; }
    return true;
}

std::string WebGuiServer::handleLoadSettings() {
    // Only the part the GUI may edit: path and network options are shown read-only (/api/paths).
    return util::redactSecretText(util::guiEditableSettings(readFile(cfg_.config_path)));
}

std::string WebGuiServer::handleSaveSettings(const std::string& body, int& status) {
    // The GUI must not be able to change where PrMers reads or writes files (worktodo, save directory,
    // kernels, the settings file itself) or where the GUI listens: a token holder could otherwise point
    // -worktodo at any file and append to it. Such options are refused here, the hand-written part of the
    // file is carried over unchanged, and the loader ignores them below the GUI marker line anyway.
    // A PrimeNet password is never stored from the browser either.
    std::string cleaned;
    const std::string err = util::checkGuiSettingsText(body, cleaned);
    if (!err.empty()) { status = 400; return err; }
    std::lock_guard<std::mutex> lk(settingsMtx_);
    const std::string content = util::composeGuiSettingsFile(readFile(cfg_.config_path), cleaned);
    if (!writeFileReplacing(cfg_.config_path, content)) {
        status = 500;
        return "Could not write the settings file.";
    }
    return std::string();
}

std::string WebGuiServer::checkWorktodoAppend(const std::string& body, std::string& lines) const {
    lines.clear();
    if (body.size() > kMaxAppendBytes)
        return "Worktodo text is too long (limit " + std::to_string(kMaxAppendBytes) + " bytes).";
    size_t count = 0, lineNo = 0, pos = 0;
    while (pos <= body.size()) {
        size_t eol = body.find('\n', pos);
        if (eol == std::string::npos) eol = body.size();
        std::string line = body.substr(pos, eol - pos);
        pos = eol + 1;
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        for (char ch : line) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (c < 0x20 || c > 0x7e)
                return "Line " + std::to_string(lineNo) + " contains a control or non-ASCII character.";
        }
        const size_t a = line.find_first_not_of(' ');
        if (a == std::string::npos) continue;                  // blank line
        line = line.substr(a, line.find_last_not_of(' ') - a + 1);
        if (line.size() > kMaxAppendLineBytes)
            return "Line " + std::to_string(lineNo) + " is too long.";
        if (++count > kMaxAppendLines)
            return "Too many lines (limit " + std::to_string(kMaxAppendLines) + ").";
        if (!cfg_.worktodo_line_ok || !cfg_.worktodo_line_ok(line))
            return "Line " + std::to_string(lineNo) + " is not a supported worktodo entry: " + line;
        if (!lines.empty()) lines += '\n';
        lines += line;
    }
    if (lines.empty()) return "empty";
    return std::string();
}

std::string WebGuiServer::handlePathsJson() const {
    std::ostringstream oss;
    oss << "{\"worktodo\":\"" << jsonEscape(cfg_.worktodo_path) << "\","
        << "\"settings\":\"" << jsonEscape(cfg_.config_path) << "\","
        << "\"results\":\"" << jsonEscape(cfg_.results_path) << "\","
        << "\"save\":\"" << jsonEscape(cfg_.save_path) << "\","
        << "\"kernel\":\"" << jsonEscape(cfg_.kernel_path) << "\"}";
    return oss.str();
}

std::string WebGuiServer::handleLoadWorktodo() {
    return readFile(cfg_.worktodo_path);
}

bool WebGuiServer::handleStop() {
    appendLog("Stop requested");
    setStatus("Stop requested");
#ifdef _WIN32
    if (!GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0)) {
        std::raise(SIGINT);
    }
#else
    std::raise(SIGINT);
#endif
    if (onStop_) onStop_();
    return true;
}


std::string WebGuiServer::htmlPage() {
    return
"<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
"<meta name=\"referrer\" content=\"no-referrer\">"
"<title>PrMers</title>"
"<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;margin:0;padding:0;background:#0b0b0e;color:#e6e6ea}"
".bar{background:#16161d;padding:12px 16px;display:flex;gap:16px;align-items:center;position:sticky;top:0;border-bottom:1px solid #242433;z-index:30}"
".url{opacity:.8}.stat{margin-left:auto;opacity:.8}"
".wrap{padding:16px;max-width:900px;margin:0 auto}"
".card{background:#111118;border:1px solid #26263a;border-radius:12px;padding:16px;margin-bottom:16px}"
".grid{display:grid;grid-template-columns:1fr;gap:8px}"
"@media(min-width:720px){.grid-2{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:8px}}"
"label{font-size:12px;opacity:.9}"
"input,select,textarea{width:100%;background:#0d0d14;color:#e6e6ea;border:1px solid #26263a;border-radius:10px;padding:10px}"
"textarea{resize:vertical}"
".progress{height:10px;background:#1c1c29;border-radius:999px;overflow:hidden}"
".fill{height:100%;width:0%}"
"button{background:#3d5afe;border:none;color:white;padding:10px 14px;border-radius:10px;cursor:pointer}"
".mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;white-space:pre-wrap}"
".muted{opacity:.8}"
".rowbtn{display:flex;gap:8px;flex-wrap:wrap}"
"#logs{max-height:160px;overflow:auto}"
".right{margin-left:auto;display:flex;gap:8px;align-items:center}"
".btn-red{background:#e53935}"
".pill{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;background:#202033;border:1px solid #343451;font-size:12px;font-weight:600}"
".backend-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px}"
".kv{min-width:0}.kv .k{font-size:11px;opacity:.65;text-transform:uppercase;letter-spacing:.05em}.kv .v{margin-top:2px;overflow-wrap:anywhere}"
"@media(max-width:600px){.backend-grid{grid-template-columns:1fr}.bar{gap:8px;flex-wrap:wrap}.stat{margin-left:0}}"
"</style></head><body>"
"<div class=bar><div><a href='https://github.com/cherubrock-seb/PrMers' target=_blank rel=noopener noreferrer style='display:inline-flex;align-items:center;gap:8px;text-decoration:none;color:inherit'><svg width='18' height='18' viewBox='0 0 24 24' fill='none' aria-hidden='true'><circle cx='12' cy='12' r='9' stroke='#3d5afe' stroke-width='2'/><path d='M7 15V9l5 6 5-6v6' stroke='#00e5ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg><b>PrMers</b></a></div><div class=url id=url></div><div class=pill id=backendBadge>Backend pending</div><div class=stat id=stat></div></div>"
"<div class=wrap>"
"<div class='card' id=backendCard>"
"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'><div style='font-weight:600'>Computation engine</div><div class=pill id=backendActive>Pending</div></div>"
"<div class=backend-grid>"
"<div class=kv><div class=k>Selection mode</div><div class=v id=backendMode>—</div></div>"
"<div class=kv><div class=k>Workload</div><div class=v id=backendWorkload>—</div></div>"
"<div class=kv><div class=k>Aevum transform</div><div class=v id=backendAevum>—</div></div>"
"<div class=kv><div class=k>Marin transform</div><div class=v id=backendMarin>—</div></div>"
"<div class=kv><div class=k>FFT3161 plan</div><div class=v id=backendFft>—</div></div>"
"<div class=kv><div class=k>Decision</div><div class=v id=backendDetail>Waiting for engine creation</div></div>"
"</div></div>"
"<div class='card'>"
"<div style='font-weight:600;margin-bottom:8px'>Logs</div>"
"<div id=logs class='mono'></div>"
"</div>"
"<div class='card'>"
"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'><div>Status</div><div class='right'><div class=muted id=res64></div><button id=stop class='btn-red'>Stop</button></div></div>"
"<div class=progress><div class=fill id=fill style='background:linear-gradient(90deg,#3d5afe,#00e5ff)'></div></div>"
"<div class=muted style='margin-top:8px' id=prog></div>"
"</div>"
"<div class=card>"
"<div style='font-weight:600;margin-bottom:8px'>Worktodo</div>"
"<div class='grid-2'>"
"<div><label>Mode</label><select id=mode><option value=prp>PRP</option><option value=ll>LL</option><option value=llsafe2>LL-SAFE2</option><option value=pm1>P-1</option><option value=ecm>ECM</option></select></div>"
"<div><label>Exponent</label><div style='display:flex;gap:8px;align-items:center'><input id=exp type=number min=2 placeholder='139700819'><a id=expopen target=_blank class=muted>Open on mersenne.ca</a></div></div>"
"<div><label>B1 (P-1 / ECM)</label><input id=b1 type=number min=0 placeholder='500000'></div>"
"<div><label>B2 (P-1 / ECM)</label><input id=b2 type=text placeholder='0 or 1000000'></div>"
"<div><label>Curves / K (ECM)</label><input id=curves type=number min=1 value=1></div>"
"<div><label>Known factors (comma)</label><input id=factors type=text placeholder='36357263,145429049'></div>"
"<div><label>Base,residue_type (PRP)</label><input id=basert type=text placeholder='3,1'></div>"
"</div>"
"<div class=rowbtn style='margin-top:8px'>"
"<button id=buildwt>Build line</button>"
"<button id=appendrun>Append & Run</button>"
"</div>"
"<div class=muted id=wtmsg style='margin-top:8px'></div>"
"<textarea id=wt class=mono style='margin-top:8px;height:140px' placeholder='PRP=1,2,p,-1'></textarea>"
"</div>"
"<div class=card>"
"<div style='font-weight:600;margin-bottom:8px'>Results</div>"
"<div class='grid-2'>"
"<div><label>Results path</label><input id=respath type=text value=''></div>"
"<div><label>Last N lines</label><div style='display:flex;gap:8px;align-items:center'><input id=reslimit type=number value=50 style='width:140px'><button id=refreshres>Refresh</button></div></div>"
"</div>"
"<pre id=reslist class='mono' style='margin-top:8px;max-height:420px;overflow:auto'></pre>"
"</div>"
"<div class=card>"
"<div style='font-weight:600;margin-bottom:8px'>Settings</div>"
"<div class='grid-2'>"
"<div><label>OpenCL device id</label><input id=opt_d type=number value=0></div>"
"<div><label>Computation backend</label><select id=opt_backend><option value=auto>Auto (recommended)</option><option value=aevum>Force Aevum</option><option value=marin>Force Marin engine::Reg</option><option value=internal>Internal PrMers NTT</option></select></div>"
"<div><label>Forced Aevum FFT3161 plan</label><input id=opt_afft type=text placeholder='auto or 1:1K:8:256'></div>"
"<div><label>Backup interval (s)</label><input id=opt_t type=number value=120></div>"
"<div><label>Enqueue max</label><input id=opt_eq type=number value=0></div>"
"<div><label>Build options</label><input id=opt_build type=text placeholder=''></div>"
"<div><label>Proof power (0..12)</label><input id=opt_proof type=number min=0 max=12 value=0></div>"
"<div><label>Res64 display interval</label><input id=opt_r64i type=number value=100000></div>"
"<div><label>Error iter</label><input id=opt_err type=number value=0></div>"
"<div><label>iterforce</label><input id=opt_if type=number value=0></div>"
"<div><label>iterforce2</label><input id=opt_if2 type=number value=0></div>"
"<div><label>LL-safe block</label><input id=opt_llb type=number value=0></div>"
"<div><label>Local size r2</label><input id=opt_l1 type=number value=0></div>"
"<div><label>Local size r5</label><input id=opt_l5 type=number value=0></div>"
"<div><label>User</label><input id=opt_user type=text></div>"
"<div><label>Computer</label><input id=opt_comp type=text></div>"
"<div><label>Submit</label><select id=opt_submit><option value=0>No</option><option value=1>Yes</option></select></div>"
"<div><label>NoAsk</label><select id=opt_noask><option value=0>No</option><option value=1>Yes</option></select></div>"
"<div><label>Wagstaff</label><select id=opt_wag><option value=0>No</option><option value=1>Yes</option></select></div>"
"<div><label>Throttle Low</label><select id=opt_th><option value=0>No</option><option value=1>Yes</option></select></div>"
"</div>"
"<div class=rowbtn style='margin-top:8px'>"
"<button id=gensettings>Generate settings</button>"
"<button id=savesettings>Save settings.cfg</button>"
"<button id=loadsettings>Load settings.cfg</button>"
"</div>"
"<div class=muted id=settingsmsg style='margin-top:8px'></div>"
"<textarea id=settingstxt class=mono style='margin-top:8px;height:160px' placeholder='-d 0 -prp -t 120 ...'></textarea>"
"</div>"
"<div class=card>"
"<div style='font-weight:600;margin-bottom:8px'>Paths</div>"
"<div class=muted style='margin-bottom:8px'>Read-only here. Set them on the command line (-worktodo, -f, -kernelpath, -config) or by editing the settings file by hand, above the GUI marker line.</div>"
"<div class=backend-grid>"
"<div class=kv><div class=k>Worktodo</div><div class='v mono' id=path_worktodo>—</div></div>"
"<div class=kv><div class=k>Settings file</div><div class='v mono' id=path_settings>—</div></div>"
"<div class=kv><div class=k>Save directory</div><div class='v mono' id=path_save>—</div></div>"
"<div class=kv><div class=k>Results</div><div class='v mono' id=path_results>—</div></div>"
"<div class=kv><div class=k>Kernel</div><div class='v mono' id=path_kernel>—</div></div>"
"</div></div>"
"</div>"
"<script>"
"const TOKEN='" + token_ + "';"
"function api(u,o){o=o||{};o.headers=Object.assign({},o.headers||{},{'X-PrMers-Token':TOKEN});return fetch(u,o);}"
"const $=q=>document.querySelector(q);"
"$('#url').textContent=window.location.origin+'/';"
"const statEl=$('#stat');const resEl=$('#res64');const progEl=$('#prog');const fill=$('#fill');const logs=$('#logs');const backendBadge=$('#backendBadge');"
"let disconnected=false, tries=0;"
"function setDisconnected(on){"
"  disconnected=on;"
"  if(on){ statEl.textContent='Reconnecting…'; }"
"}"
"async function pull(){"
"  try{"
"    const r=await api('/api/state',{cache:'no-store'});"
"    if(!r.ok) throw new Error('http');"
"    const j=await r.json();"
"    resEl.textContent=j.res64||'';"
"    fill.style.width=(j.percent||0)+'%';"
"    progEl.textContent=(j.current||0)+' / '+(j.total||0)+'  ('+(j.percent||0)+'%)';"
"    statEl.textContent=j.status||'';"
"    const active=j.backend_active||'Pending';const mode=j.backend_mode||'—';const wl=j.backend_workload||'—';"
"    backendBadge.textContent=(mode==='Auto'?('Auto → '+active):active);const bc=(active||'').toLowerCase().includes('aevum')?'aevum':((active||'').toLowerCase().includes('marin')?'marin':((active||'').toLowerCase().includes('internal')?'internal':''));backendBadge.className='pill '+bc;$('#backendActive').textContent=active;$('#backendActive').className='pill '+bc;$('#backendMode').textContent=mode;$('#backendWorkload').textContent=wl;"
"    $('#backendAevum').textContent=j.backend_aevum_transform?Number(j.backend_aevum_transform).toLocaleString()+' words':'—';$('#backendMarin').textContent=j.backend_marin_transform?Number(j.backend_marin_transform).toLocaleString()+' words':'—';$('#backendFft').textContent=j.backend_fft||'—';$('#backendDetail').textContent=j.backend_detail||'—';"
"    logs.innerHTML=(j.logs||[]).slice(-1000).map(x=>x.replace(/&/g,'&amp;').replace(/</g,'&lt;')).join('\\n');"
"    logs.scrollTop=logs.scrollHeight;"
"    if(disconnected){ setDisconnected(false); tries=0; }"
"  }catch(e){"
"    if(!disconnected) setDisconnected(true);"
"    tries++;"
"  }"
"}"
"setInterval(pull,1000); pull();"
"$('#stop').onclick=async()=>{try{await api('/api/stop',{method:'POST'});}catch(e){}};"
"function buildWorktodo(){const m=$('#mode').value;const p=parseInt($('#exp').value||'0');const b1=$('#b1').value.trim();const b2=$('#b2').value.trim();const curves=Math.max(1,parseInt($('#curves').value||'1'));const factors=($('#factors').value||'').split(',').map(s=>s.trim()).filter(Boolean);const basert=($('#basert').value||'').split(',').map(s=>s.trim()).filter(Boolean);let line='';if(m==='prp'){line=`PRP=1,2,${p},-1`;if(basert.length===2)line+=`,0,0,`+basert[0]+`,`+basert[1];if(factors.length)line+=`,\"`+factors.join(',')+`\"`;}else if(m==='ll'){line=`Test=1,2,${p},-1`;}else if(m==='llsafe2'){line=`DoubleCheck=1,2,${p},-1`;}else if(m==='pm1'){let B1=(b1||'0');let B2=(b2||'0');line=`Pminus1=1,2,${p},-1,${B1},${B2}`;if(factors.length)line+=`,`+factors.map(s=>`\"${s}\"`).join(',');}else if(m==='ecm'){let B1=(b1||'0');let B2=(b2||'0');line=`ECM2=1,2,${p},-1,${B1},${B2},${curves}`;if(factors.length)line+=`,\"`+factors.join(',')+`\"`;}return line;}"
"$('#buildwt').onclick=()=>{$('#wt').value=buildWorktodo();};"
"$('#appendrun').onclick=async()=>{await postText('/api/append-worktodo',$('#wt').value,$('#wtmsg'),'Appended.');};"
"function genSettings(){const parts=[];const d=$('#opt_d').value;parts.push('-d',d);const m=$('#mode').value;parts.push(m==='prp'?'-prp':m==='ll'?'-ll':m==='llsafe2'?'-llsafe2':m==='ecm'?'-ecm':'-pm1');const be=$('#opt_backend').value;const afft=($('#opt_afft').value||'').trim();if(be==='aevum'){parts.push('-aevum');if(afft)parts.push('-aevum-fft',afft);}else if(be==='marin')parts.push('-engine-marin');else if(be==='internal')parts.push('-marin');else parts.push('-aevum-auto');if(m==='pm1'||m==='ecm'){const b1=($('#b1').value||'').trim();const b2=($('#b2').value||'').trim();if(b1)parts.push('-b1',b1);if(b2)parts.push('-b2',b2);}if(m==='ecm'){const cv=Math.max(1,parseInt($('#curves').value||'1'));parts.push('-K',String(cv));}const t=$('#opt_t').value;if(t)parts.push('-t',t);const l1=$('#opt_l1').value;if(l1&&parseInt(l1))parts.push('-l1',l1);const l5=$('#opt_l5').value;if(l5&&parseInt(l5))parts.push('-l5',l5);const eq=$('#opt_eq').value;if(eq&&parseInt(eq))parts.push('-enqueue_max',eq);const bo=$('#opt_build').value;if(bo)parts.push('-build',`\"${bo}\"`);const proof=$('#opt_proof').value;if(proof!==''&&proof!==null)parts.push('-proof',proof);const r64=$('#opt_r64i').value;if(r64&&parseInt(r64)>=0)parts.push('-res64_display_interval',r64);const err=$('#opt_err').value;if(err&&parseInt(err))parts.push('-erroriter',err);const iff=$('#opt_if').value;if(iff&&parseInt(iff))parts.push('-iterforce',iff);const iff2=$('#opt_if2').value;if(iff2&&parseInt(iff2))parts.push('-iterforce2',iff2);const llb=$('#opt_llb').value;if(llb&&parseInt(llb))parts.push('-llsafeb',llb);const wag=$('#opt_wag').value;if(wag==='1')parts.push('-wagstaff');const th=$('#opt_th').value;if(th==='1')parts.push('-throttle_low');const sub=$('#opt_submit').value;if(sub==='1')parts.push('-submit');const na=$('#opt_noask').value;if(na==='1')parts.push('--noask');const user=$('#opt_user').value;if(user)parts.push('-user',user);const comp=$('#opt_comp').value;if(comp)parts.push('-computer',comp);return parts.join(' ');} "
"$('#gensettings').onclick=()=>{$('#settingstxt').value=genSettings();};"
"async function postText(u,t,msgEl,okText){msgEl.textContent='';try{const r=await api(u,{method:'POST',headers:{'Content-Type':'text/plain'},body:t});let j={};try{j=await r.json();}catch(e){}msgEl.textContent=r.ok?okText:('Error: '+(j.error||('HTTP '+r.status)));return r.ok;}catch(e){msgEl.textContent='Error: '+e;return false;}}"
"$('#savesettings').onclick=async()=>{await postText('/api/save-settings',$('#settingstxt').value,$('#settingsmsg'),'Saved; used from the next start.');};"
"$('#loadsettings').onclick=async()=>{const r=await api('/api/load-settings');const t=await r.text();$('#settingstxt').value=t;};"
"async function refreshResults(){const n=parseInt($('#reslimit').value||'50');const p=$('#respath').value||'';const r=await api('/api/results?limit='+n+(p?('&path='+encodeURIComponent(p)):''));const j=await r.json();const html=(j.lines||[]).map(x=>{let o=null;try{o=JSON.parse(x);}catch(e){}if(!o)return x;const ts=o.timestamp||'';const st=o.status||'';const wt=o.worktype||o.program?.name||'';const e=o.exponent||'';const r64=o.res64||'';return `[${ts}] ${st} e=${e} ${wt} res64=${r64}`;}).join('\\n');$('#reslist').textContent=html;}"
"$('#refreshres').onclick=refreshResults;"
"function updateExpLink(){const e=$('#exp').value||'';const u=e?('https://www.mersenne.ca/exponent/'+e):'https://www.mersenne.ca';$('#expopen').href=u;}"
"$('#exp').addEventListener('input',updateExpLink);updateExpLink();"
"async function loadWorktodo(){try{const r=await api('/api/load-worktodo');const t=await r.text();if(t)$('#wt').value=t;}catch(e){}}"
"async function loadPaths(){try{const r=await api('/api/paths');const j=await r.json();for(const k of ['worktodo','settings','save','results','kernel']){$('#path_'+k).textContent=j[k]||'—';}}catch(e){}}"
"(async()=>{loadPaths();try{const r=await api('/api/load-settings');const t=await r.text();if(t)$('#settingstxt').value=t;}catch(e){};refreshResults();loadWorktodo();})();"
"</script>"
"</body></html>";
}


std::string WebGuiServer::httpOk(const std::string& contentType, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n";
    oss << "Content-Type: " << contentType << "\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n\r\n";
    oss << body;
    return oss.str();
}

std::string WebGuiServer::httpBadRequest(const std::string& msg) {
    std::string body = "{\"error\":\"" + jsonEscape(msg) + "\"}";
    std::ostringstream oss;
    oss << "HTTP/1.1 400 Bad Request\r\n";
    oss << "Content-Type: application/json\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n\r\n";
    oss << body;
    return oss.str();
}

std::string WebGuiServer::httpError(int code, const std::string& msg) {
    const char* reason = code == 400 ? "Bad Request" : code == 401 ? "Unauthorized" : code == 403 ? "Forbidden"
                       : code == 413 ? "Payload Too Large" : code == 431 ? "Request Header Fields Too Large"
                       : code == 500 ? "Internal Server Error" : "Error";
    std::string body = "{\"error\":\"" + jsonEscape(msg) + "\"}";
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << " " << reason << "\r\n";
    oss << "Content-Type: application/json\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n\r\n";
    oss << body;
    return oss.str();
}

std::string WebGuiServer::httpNotFound() {
    std::string body = "Not Found";
    std::ostringstream oss;
    oss << "HTTP/1.1 404 Not Found\r\n";
    oss << "Content-Type: text/plain\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n\r\n";
    oss << body;
    return oss.str();
}

std::string WebGuiServer::jsonEscape(const std::string& s) {
    std::string o; o.reserve(s.size()+8);
    for (char c : s) {
        switch(c){
            case '\\': o += "\\\\"; break;
            case '\"': o += "\\\""; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c); o += buf; }
                else o += c;
        }
    }
    return o;
}

}
