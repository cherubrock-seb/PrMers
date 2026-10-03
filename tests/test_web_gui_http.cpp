// End-to-end HTTP checks of the GUI server's access control and request limits, over real sockets.
#include "ui/WebGuiServer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
static void closeSocket(socket_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static void closeSocket(socket_t s) { ::close(s); }
#endif

static int g_port = 0;

// Send a raw request and return the HTTP status code (0 if the connection was closed without a reply).
static int request(const std::string& raw, std::string* bodyOut = nullptr) {
    socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(g_port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { closeSocket(s); throw std::runtime_error("connect failed"); }
    size_t off = 0;
    while (off < raw.size()) {
        const int n = send(s, raw.data() + off, static_cast<int>(raw.size() - off), 0);
        if (n <= 0) break;
        off += static_cast<size_t>(n);
    }
    std::string resp;
    char buf[4096];
    for (;;) {
        const int n = recv(s, buf, sizeof(buf), 0);
        if (n <= 0) break;
        resp.append(buf, buf + n);
    }
    closeSocket(s);
    if (resp.rfind("HTTP/1.1 ", 0) != 0) return 0;
    if (bodyOut) {
        const auto p = resp.find("\r\n\r\n");
        *bodyOut = p == std::string::npos ? std::string() : resp.substr(p + 4);
    }
    return std::stoi(resp.substr(9, 3));
}

static std::string get(const std::string& path, const std::string& extraHeaders) {
    return "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(g_port) + "\r\n" + extraHeaders + "\r\n";
}

static void expect(int got, int want, const std::string& what) {
    if (got != want) throw std::runtime_error(what + ": HTTP " + std::to_string(got) + ", expected " + std::to_string(want));
    std::cout << "  " << what << ": " << got << "\n";
}

int main() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    {
        std::ofstream("gui_http_settings.cfg") << "-d 0 -password hunter2 -t 600\n";
        std::ofstream("gui_http_results.txt") << "{\"status\":\"C\",\"exponent\":127}\n";
    }
    ui::WebGuiConfig cfg;
    cfg.port = 0;                                   // any free port
    cfg.bind_host = "127.0.0.1";
    cfg.config_path = "gui_http_settings.cfg";
    cfg.results_path = "gui_http_results.txt";
    cfg.worktodo_path = "gui_http_worktodo.txt";
    ui::WebGuiServer server(cfg, [](const std::string&) {});
    server.start();

    const std::string url = server.url();
    const auto colon = url.rfind(':');
    g_port = std::stoi(url.substr(colon + 1));
    const auto tpos = url.find("token=");
    if (g_port <= 0 || tpos == std::string::npos) throw std::runtime_error("unexpected GUI URL " + url);
    const std::string token = url.substr(tpos + 6);
    const std::string auth = "X-PrMers-Token: " + token + "\r\n";
    std::cout << "GUI HTTP test on port " << g_port << "\n";

    expect(request(get("/", "")), 401, "page without token");
    expect(request(get("/?token=" + token, "")), 200, "page with token");
    expect(request(get("/api/state", "")), 401, "API without token");
    expect(request(get("/api/state", "X-PrMers-Token: 0123456789abcdef0123456789abcdef\r\n")), 401, "API with wrong token");
    expect(request(get("/api/state", auth)), 200, "API with token");
    expect(request(get("/api/state", "x-prmers-token: " + token + "\r\n")), 200, "token header name is case-insensitive");
    expect(request("GET /api/state HTTP/1.1\r\nHost: evil.example:" + std::to_string(g_port) + "\r\n" + auth + "\r\n"), 403, "foreign Host");
    expect(request(get("/api/state", auth + "Origin: http://evil.example\r\n")), 403, "foreign Origin");
    expect(request(get("/api/results?path=gui_http_settings.cfg", auth)), 400, "results path outside .txt/.json");
    expect(request(get("/api/results?path=..%2Fx.txt", auth)), 400, "results path outside the results directory");
    expect(request(get("/api/results", auth)), 200, "results");

    std::string body;
    expect(request(get("/api/load-settings", auth), &body), 200, "load settings");
    if (body.find("hunter2") != std::string::npos || body.find("-password ********") == std::string::npos)
        throw std::runtime_error("load-settings did not mask the password: " + body);

    expect(request("POST /api/save-settings HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(g_port) + "\r\n" + auth +
                   "Content-Length: 2000000\r\n\r\n"), 413, "oversized body");
    expect(request("POST /api/save-settings HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(g_port) + "\r\n" + auth +
                   "Content-Length: -1\r\n\r\n"), 400, "negative Content-Length");
    expect(request("GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Junk: " + std::string(20000, 'a') + "\r\n\r\n"), 431, "oversized headers");

    server.stop();
    std::remove("gui_http_settings.cfg");
    std::remove("gui_http_results.txt");
    std::cout << "Web GUI HTTP test passed\n";
    return 0;
}
