#pragma once
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <vector>
#include <memory>

namespace ui {

struct WebGuiConfig {
    int port = 0;
    std::string worktodo_path;
    std::string config_path;
    std::string results_path;
    // Shown read-only in the page with the paths above; the GUI cannot change any of them.
    std::string save_path;
    std::string kernel_path;
    // Accepts one line for /api/append-worktodo (a runnable worktodo entry). Unset: appends are refused.
    std::function<bool(const std::string&)> worktodo_line_ok;
    std::string bind_host;
    std::string advertise_host;
    bool lanipv4 = false;
};

class WebGuiServer {
public:
    using SubmitFn = std::function<void(const std::string&)>;
    using StopFn = std::function<void()>;
    WebGuiServer(const WebGuiConfig& cfg, SubmitFn onSubmit, StopFn onStop = {});
    ~WebGuiServer();
    // Returns false (after printing the reason to stderr) when the listening socket cannot be set up.
    bool start();
    void stop();
    std::string url() const;
    static std::shared_ptr<WebGuiServer> instance();
    static void setInstance(std::shared_ptr<WebGuiServer> s);
    void setStatus(const std::string& s);
    void setProgress(uint64_t current, uint64_t total, const std::string& res64);
    void setBackendInfo(const std::string& mode,
                        const std::string& active,
                        const std::string& workload,
                        const std::string& detail,
                        uint64_t aevum_transform = 0,
                        uint64_t marin_transform = 0,
                        const std::string& fft_spec = {});
    void appendLog(const std::string& line);
    std::string stateJson();
private:
    struct State {
        std::string status;
        uint64_t cur = 0;
        uint64_t tot = 0;
        std::string res64;
        std::string backend_mode;
        std::string backend_active;
        std::string backend_workload;
        std::string backend_detail;
        std::string backend_fft;
        uint64_t backend_aevum_transform = 0;
        uint64_t backend_marin_transform = 0;
        std::deque<std::string> logs;
    };
    WebGuiConfig cfg_;
    SubmitFn onSubmit_;
    StopFn onStop_;
    mutable std::mutex mtx_;
    std::mutex settingsMtx_;   // one settings save (read, merge, replace) at a time
    State st_;
    std::atomic<bool> running_{false};
    std::thread thr_;
    int listen_fd_ = -1;
    int bound_port_ = 3131;
    std::string url_;
    // Per-run access token. Every /api/* request must carry it in X-PrMers-Token, and the page itself is
    // only served for /?token=<token>. Inherited through PRMERS_GUI_TOKEN so a restart keeps it.
    std::string token_;
    std::atomic<int> active_connections_{0};
    void run();
    void closeListen();
    void serveOne(int fd);
    // Returns 0 on success, an HTTP status code for a request to reject, or -1 to drop the connection.
    static int readRequest(int fd, std::string& method, std::string& path, std::string& body, std::string& headers);
    static std::string headerValue(const std::string& headers, const std::string& name);
    bool hostAllowed(const std::string& hostHeader) const;
    bool originAllowed(const std::string& origin) const;
    bool tokenMatches(const std::string& candidate) const;
    bool resolveResultsPath(const std::string& requested, std::string& resolved) const;
    static bool sendAll(int fd, const char* data, size_t len);
    static int createListenSocket(const std::string& bind_host, int port, int& out_port);
    std::string handleStateJson();
    std::string handleResultsJson(size_t limit, const std::string& pathOverride);
    std::string handleLoadSettings();
    // Empty on success, otherwise the reason (sets `status` to the HTTP code to answer with).
    std::string handleSaveSettings(const std::string& body, int& status);
    // Empty and `lines` set to the lines to append when the body is acceptable, otherwise the reason.
    std::string checkWorktodoAppend(const std::string& body, std::string& lines) const;
    std::string handlePathsJson() const;
    std::string handleLoadWorktodo();
    bool handleStop();
    std::string htmlPage();
    std::string httpOk(const std::string& contentType, const std::string& body);
    std::string httpBadRequest(const std::string& msg);
    std::string httpNotFound();
    std::string httpError(int code, const std::string& msg);
    static std::string jsonEscape(const std::string& s);
    static std::string readFile(const std::string& path);
    static bool writeFileReplacing(const std::string& path, const std::string& data);
    static std::vector<std::string> tailLines(const std::string& path, size_t limit);
};

}
