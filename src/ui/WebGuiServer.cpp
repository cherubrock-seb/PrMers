#include "ui/WebGuiServer.hpp"
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <csignal>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib,"Ws2_32.lib")
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

namespace ui {

static std::shared_ptr<WebGuiServer> g_instance;

WebGuiServer::WebGuiServer(const WebGuiConfig& cfg, SubmitFn onSubmit, StopFn onStop)
: cfg_(cfg), onSubmit_(std::move(onSubmit)), onStop_(std::move(onStop)) {}
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


void WebGuiServer::start() {
    if (running_) return;
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
    if (listen_fd_ < 0) return;
    url_ = "http://" + host + ":" + std::to_string(cfg_.port) + "/";
    //std::string host = firstLanIPv4();
    if (host.empty()) host = "127.0.0.1";    // fallback
    url_ = std::string("http://") + host + ":" + std::to_string(cfg_.port) + "/";

    running_ = true;
    thr_ = std::thread([this]{ run(); });
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

void WebGuiServer::appendLog(const std::string& line) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (st_.logs.size() > 2000) st_.logs.pop_front();
    st_.logs.push_back(line);
}

void WebGuiServer::run() {
    while (running_) {
#ifdef _WIN32
        SOCKET cfd = accept((SOCKET)listen_fd_, nullptr, nullptr);
        if (cfd == INVALID_SOCKET) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
        std::thread([this, cfd]{
            serveOne((int)cfd);
            closesocket(cfd);
        }).detach();
#else
        int cfd = ::accept(listen_fd_, nullptr, nullptr);
        if (cfd < 0) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
        std::thread([this, cfd]{
            serveOne(cfd);
            ::close(cfd);
        }).detach();
#endif
    }
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
#endif

    sockaddr_in addr; std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);

    in_addr ip{};
    std::string h = bind_host;
    if (h.empty() || h == "localhost") h = "127.0.0.1";
    if (h == "0.0.0.0") ip.s_addr = htonl(INADDR_ANY);
#ifdef _WIN32
    else if (InetPtonA(AF_INET, h.c_str(), &ip) != 1) ip.s_addr = htonl(INADDR_LOOPBACK);
#else
    else if (inet_pton(AF_INET, h.c_str(), &ip) != 1) ip.s_addr = htonl(INADDR_LOOPBACK);
#endif
    addr.sin_addr = ip;

    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { 
        return -1; 
    }
    if (::listen(fd, 16) < 0) { 
        return -1; 
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(fd, (sockaddr*)&addr, &len) == 0) out_port = ntohs(addr.sin_port); else out_port = port;
    return fd;
}



bool WebGuiServer::readRequest(int fd, std::string& method, std::string& path, std::string& body, std::string& headers) {
    std::string req;
    char buf[4096];
    for (;;) {
#ifdef _WIN32
        int n = recv(fd, buf, sizeof(buf), 0);
#else
        int n = (int)::recv(fd, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        req.append(buf, buf + n);
        if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    size_t pos = req.find("\r\n");
    if (pos == std::string::npos) return false;
    std::string start = req.substr(0, pos);
    headers = req.substr(pos + 2);
    size_t p2 = start.find(' ');
    if (p2 == std::string::npos) return false;
    size_t p3 = start.find(' ', p2 + 1);
    if (p3 == std::string::npos) return false;
    method = start.substr(0, p2);
    path = start.substr(p2 + 1, p3 - p2 - 1);
    size_t hend = headers.find("\r\n\r\n");
    if (hend == std::string::npos) return false;
    std::string hs = headers.substr(0, hend);
    size_t clpos = hs.find("Content-Length:");
    size_t contentLength = 0;
    if (clpos != std::string::npos) {
        size_t eol = hs.find("\r\n", clpos);
        std::string v = hs.substr(clpos + 15, eol - (clpos + 15));
        size_t a = v.find_first_not_of(" \t");
        if (a != std::string::npos) v = v.substr(a);
        contentLength = (size_t)atoll(v.c_str());
    }
    std::string tail = headers.substr(hend + 4);
    if (contentLength > tail.size()) {
        size_t need = contentLength - tail.size();
        std::string more;
        while (need > 0) {
#ifdef _WIN32
            int n = recv(fd, buf, (int)std::min(need,(size_t)sizeof(buf)), 0);
#else
            int n = (int)::recv(fd, buf, (int)std::min(need,(size_t)sizeof(buf)), 0);
#endif
            if (n <= 0) break;
            more.append(buf, buf + n);
            need -= (size_t)n;
        }
        tail += more;
    }
    body = tail;
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
    if (!readRequest(fd, method, path, body, headers)) return;
    std::string resp;
    if (method == "GET" && (path == "/" || path == "/index.html")) {
        resp = httpOk("text/html; charset=utf-8", htmlPage());
    } else if (method == "GET" && path == "/api/state") {
        resp = httpOk("application/json", handleStateJson());
    } else if (method == "GET" && path.rfind("/api/results", 0) == 0) {
        size_t limit = 100;
        std::string overridePath;
        auto qpos = path.find('?');
        if (qpos != std::string::npos) {
            auto qs = path.substr(qpos + 1);
            std::istringstream iss(qs);
            std::string kv;
            while (std::getline(iss, kv, '&')) {
                auto eq = kv.find('=');
                if (eq == std::string::npos) continue;
                auto k = kv.substr(0, eq);
                auto v = kv.substr(eq + 1);
                if (k == "limit") {
                    size_t n = (size_t)std::strtoul(v.c_str(), nullptr, 10);
                    if (n > 0) limit = n;
                } else if (k == "path") {
                    for (auto& c : v) if (c == '+') c = ' ';
                    overridePath = v;
                }
            }
        }
        resp = httpOk("application/json", handleResultsJson(limit, overridePath));
    } else if (method == "GET" && path == "/api/load-settings") {
        resp = httpOk("text/plain; charset=utf-8", handleLoadSettings());
    } else if (method == "GET" && path == "/api/load-worktodo") {
        resp = httpOk("text/plain; charset=utf-8", handleLoadWorktodo());
    } else if (method == "POST" && path == "/api/save-settings") {
        bool ok = handleSaveSettings(body);
        resp = ok ? httpOk("application/json", "{\"ok\":true}") : httpBadRequest("write-failed");
    } else if (method == "POST" && path == "/api/append-worktodo") {
        std::string line = body;
        size_t p = line.find_first_not_of("\r\n ");
        if (p != std::string::npos) line = line.substr(p);
        while (!line.empty() && (line.back()=='\r' || line.back()=='\n')) line.pop_back();
        if (line.empty()) {
            resp = httpBadRequest("empty");
        } else {
            if (onSubmit_) onSubmit_(line);
            resp = httpOk("application/json", "{\"ok\":true}");
        }
        } else if (method == "POST" && path == "/api/stop") {
        bool ok = handleStop();
        resp = ok ? httpOk("application/json", "{\"ok\":true}") : httpBadRequest("stop-failed");
    } else {
        resp = httpNotFound();
    }
    sendAll(fd, resp.data(), resp.size());
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

bool WebGuiServer::writeFile(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f.write(data.data(), (std::streamsize)data.size());
    return (bool)f;
}

std::string WebGuiServer::handleLoadSettings() {
    return readFile(cfg_.config_path);
}

bool WebGuiServer::handleSaveSettings(const std::string& body) {
    return writeFile(cfg_.config_path, body);
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
"</style></head><body>"
"<div class=bar><div><a href='https://github.com/cherubrock-seb/PrMers' target=_blank rel=noopener noreferrer style='display:inline-flex;align-items:center;gap:8px;text-decoration:none;color:inherit'><svg width='18' height='18' viewBox='0 0 24 24' fill='none' aria-hidden='true'><circle cx='12' cy='12' r='9' stroke='#3d5afe' stroke-width='2'/><path d='M7 15V9l5 6 5-6v6' stroke='#00e5ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg><b>PrMers</b></a></div><div class=url id=url></div><div class=stat id=stat></div></div>"
"<div class=wrap>"
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
"<div><label>Mode</label><select id=mode><option value=prp>PRP</option><option value=ll>LL</option><option value=llsafe>LL-SAFE</option><option value=pm1>P-1</option></select></div>"
"<div><label>Exponent</label><div style='display:flex;gap:8px;align-items:center'><input id=exp type=number min=2 placeholder='139700819'><a id=expopen target=_blank class=muted>Open on mersenne.ca</a></div></div>"
"<div><label>B1 (P-1)</label><input id=b1 type=number min=0 placeholder='500000'></div>"
"<div><label>B2 (P-1)</label><input id=b2 type=text placeholder='0 or 1.3'></div>"
"<div><label>Known factors (comma)</label><input id=factors type=text placeholder='36357263,145429049'></div>"
"<div><label>Base,residue_type (PRP)</label><input id=basert type=text placeholder='3,1'></div>"
"</div>"
"<div class=rowbtn style='margin-top:8px'>"
"<button id=buildwt>Build line</button>"
"<button id=appendrun>Append & Run</button>"
"</div>"
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
"<div><label>Backup interval (s)</label><input id=opt_t type=number value=120></div>"
"<div><label>Save path</label><input id=opt_f type=text value='.'></div>"
"<div><label>Enqueue max</label><input id=opt_eq type=number value=0></div>"
"<div><label>Kernel path</label><input id=opt_kpath type=text value='ocl/kernel.cl'></div>"
"<div><label>Build options</label><input id=opt_build type=text placeholder=''></div>"
"<div><label>Proof power (0..12)</label><input id=opt_proof type=number min=0 max=12 value=0></div>"
"<div><label>Res64 display interval</label><input id=opt_r64i type=number value=100000></div>"
"<div><label>Error iter</label><input id=opt_err type=number value=0></div>"
"<div><label>iterforce</label><input id=opt_if type=number value=0></div>"
"<div><label>iterforce2</label><input id=opt_if2 type=number value=0></div>"
"<div><label>LL-safe block</label><input id=opt_llb type=number value=0></div>"
"<div><label>Local size r2</label><input id=opt_l1 type=number value=0></div>"
"<div><label>Local size r5</label><input id=opt_l5 type=number value=0></div>"
"<div><label>Worktodo path</label><input id=opt_wt type=text value='./worktodo.txt'></div>"
"<div><label>Output dir</label><input id=opt_out type=text value='.'></div>"
"<div><label>User</label><input id=opt_user type=text></div>"
"<div><label>Password</label><input id=opt_pass type=password></div>"
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
"<textarea id=settingstxt class=mono style='margin-top:8px;height:160px' placeholder='-d 0 -prp -t 120 -f . ...'></textarea>"
"</div>"
"</div>"
"<script>"
"const $=q=>document.querySelector(q);"
"$('#url').textContent=window.location.href;"
"const statEl=$('#stat');const resEl=$('#res64');const progEl=$('#prog');const fill=$('#fill');const logs=$('#logs');"
"let disconnected=false, tries=0;"
"function setDisconnected(on){"
"  disconnected=on;"
"  if(on){ statEl.textContent='Reconnecting…'; }"
"}"
"async function pull(){"
"  try{"
"    const r=await fetch('/api/state',{cache:'no-store'});"
"    if(!r.ok) throw new Error('http');"
"    const j=await r.json();"
"    resEl.textContent=j.res64||'';"
"    fill.style.width=(j.percent||0)+'%';"
"    progEl.textContent=(j.current||0)+' / '+(j.total||0)+'  ('+(j.percent||0)+'%)';"
"    statEl.textContent=j.status||'';"
"    logs.innerHTML=(j.logs||[]).slice(-1000).map(x=>x.replace(/&/g,'&amp;').replace(/</g,'&lt;')).join('\\n');"
"    logs.scrollTop=logs.scrollHeight;"
"    if(disconnected){ setDisconnected(false); tries=0; }"
"  }catch(e){"
"    if(!disconnected) setDisconnected(true);"
"    tries++;"
"  }"
"}"
"setInterval(pull,1000); pull();"
"$('#stop').onclick=async()=>{try{await fetch('/api/stop',{method:'POST'});}catch(e){}};"
"function buildWorktodo(){const m=$('#mode').value;const p=parseInt($('#exp').value||'0');const b1=$('#b1').value.trim();const b2=$('#b2').value.trim();const factors=($('#factors').value||'').split(',').map(s=>s.trim()).filter(Boolean);const basert=($('#basert').value||'').split(',').map(s=>s.trim()).filter(Boolean);let line='';if(m==='prp'){line=`PRP=1,2,${p},-1`;if(basert.length===2)line+=`,`+basert[0]+`,`+basert[1];if(factors.length)line+=`,\"`+factors.join(',')+`\"`;}else if(m==='ll'){line=`Test=1,2,${p},-1`;}else if(m==='llsafe'){line=`DoubleCheck=1,2,${p},-1`;}else if(m==='pm1'){let B1=(b1||'0');let B2=(b2||'0');line=`Pminus1=1,2,${p},-1,${B1},${B2}`;if(factors.length)line+=`,`+factors.map(s=>`\"${s}\"`).join(',');}return line;}"
"$('#buildwt').onclick=()=>{$('#wt').value=buildWorktodo();};"
"$('#appendrun').onclick=async()=>{const t=$('#wt').value;try{await fetch('/api/append-worktodo',{method:'POST',headers:{'Content-Type':'text/plain'},body:t});}catch(e){}};"
"function genSettings(){const parts=[];const d=$('#opt_d').value;parts.push('-d',d);const m=$('#mode').value;parts.push(m==='prp'?'-prp':m==='ll'?'-ll':m==='llsafe'?'-llsafe':'-pm1');const t=$('#opt_t').value;if(t)parts.push('-t',t);const f=$('#opt_f').value;if(f)parts.push('-f',f);const l1=$('#opt_l1').value;if(l1&&parseInt(l1))parts.push('-l1',l1);const l5=$('#opt_l5').value;if(l5&&parseInt(l5))parts.push('-l5',l5);const eq=$('#opt_eq').value;if(eq&&parseInt(eq))parts.push('-enqueue_max',eq);const kp=$('#opt_kpath').value;if(kp)parts.push('-kernel_path',kp);const bo=$('#opt_build').value;if(bo)parts.push('-build',`\"${bo}\"`);const proof=$('#opt_proof').value;if(proof!==''&&proof!==null)parts.push('-proof',proof);const r64=$('#opt_r64i').value;if(r64&&parseInt(r64)>=0)parts.push('-res64_display_interval',r64);const err=$('#opt_err').value;if(err&&parseInt(err))parts.push('-erroriter',err);const iff=$('#opt_if').value;if(iff&&parseInt(iff))parts.push('-iterforce',iff);const iff2=$('#opt_if2').value;if(iff2&&parseInt(iff2))parts.push('-iterforce2',iff2);const llb=$('#opt_llb').value;if(llb&&parseInt(llb))parts.push('-llsafeb',llb);const wt=$('#opt_wt').value;if(wt)parts.push('-worktodo',wt);const out=$('#opt_out').value;if(out)parts.push('-output',out);const wag=$('#opt_wag').value;if(wag==='1')parts.push('-wagstaff');const th=$('#opt_th').value;if(th==='1')parts.push('-throttle_low');const sub=$('#opt_submit').value;if(sub==='1')parts.push('-submit');const na=$('#opt_noask').value;if(na==='1')parts.push('--noask');const user=$('#opt_user').value;if(user)parts.push('-user',user);const pass=$('#opt_pass').value;if(pass)parts.push('-password',pass);const comp=$('#opt_comp').value;if(comp)parts.push('-computer',comp);return parts.join(' ');} "
"$('#gensettings').onclick=()=>{$('#settingstxt').value=genSettings();};"
"$('#savesettings').onclick=async()=>{const txt=$('#settingstxt').value;await fetch('/api/save-settings',{method:'POST',headers:{'Content-Type':'text/plain'},body:txt});};"
"$('#loadsettings').onclick=async()=>{const r=await fetch('/api/load-settings');const t=await r.text();$('#settingstxt').value=t;};"
"async function refreshResults(){const n=parseInt($('#reslimit').value||'50');const p=$('#respath').value||'';const r=await fetch('/api/results?limit='+n+(p?('&path='+encodeURIComponent(p)):''));const j=await r.json();const html=(j.lines||[]).map(x=>{let o=null;try{o=JSON.parse(x);}catch(e){}if(!o)return x;const ts=o.timestamp||'';const st=o.status||'';const wt=o.worktype||o.program?.name||'';const e=o.exponent||'';const r64=o.res64||'';return `[${ts}] ${st} e=${e} ${wt} res64=${r64}`;}).join('\\n');$('#reslist').textContent=html;}"
"$('#refreshres').onclick=refreshResults;"
"function updateExpLink(){const e=$('#exp').value||'';const u=e?('https://www.mersenne.ca/exponent/'+e):'https://www.mersenne.ca';$('#expopen').href=u;}"
"$('#exp').addEventListener('input',updateExpLink);updateExpLink();"
"async function loadWorktodo(){try{const r=await fetch('/api/load-worktodo');const t=await r.text();if(t)$('#wt').value=t;}catch(e){}}"
"(async()=>{try{const r=await fetch('/api/load-settings');const t=await r.text();if(t)$('#settingstxt').value=t;}catch(e){};refreshResults();loadWorktodo();})();"
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
