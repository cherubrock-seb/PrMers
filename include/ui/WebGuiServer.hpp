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
};

class WebGuiServer {
public:
    using SubmitFn = std::function<void(const std::string&)>;
    using StopFn = std::function<void()>;
    WebGuiServer(const WebGuiConfig& cfg, SubmitFn onSubmit, StopFn onStop = {});
    ~WebGuiServer();
    void start();
    void stop();
    std::string url() const;
    static std::shared_ptr<WebGuiServer> instance();
    static void setInstance(std::shared_ptr<WebGuiServer> s);
    void setStatus(const std::string& s);
    void setProgress(uint64_t current, uint64_t total, const std::string& res64);
    void appendLog(const std::string& line);
private:
    struct State {
        std::string status;
        uint64_t cur = 0;
        uint64_t tot = 0;
        std::string res64;
        std::deque<std::string> logs;
    };
    WebGuiConfig cfg_;
    SubmitFn onSubmit_;
    StopFn onStop_;
    mutable std::mutex mtx_;
    State st_;
    std::atomic<bool> running_{false};
    std::thread thr_;
    int listen_fd_ = -1;
    int bound_port_ = 0;
    std::string url_;
    void run();
    void closeListen();
    void serveOne(int fd);
    static bool readRequest(int fd, std::string& method, std::string& path, std::string& body, std::string& headers);
    static bool sendAll(int fd, const char* data, size_t len);
    static int createListenSocket(int port, int& out_port);
    std::string handleStateJson();
    std::string handleResultsJson(size_t limit, const std::string& pathOverride);
    std::string handleLoadSettings();
    bool handleSaveSettings(const std::string& body);
    std::string handleLoadWorktodo();
    bool handleStop();
    std::string htmlPage();
    std::string httpOk(const std::string& contentType, const std::string& body);
    std::string httpBadRequest(const std::string& msg);
    std::string httpNotFound();
    static std::string jsonEscape(const std::string& s);
    static std::string readFile(const std::string& path);
    static bool writeFile(const std::string& path, const std::string& data);
    static std::vector<std::string> tailLines(const std::string& path, size_t limit);
};

}
