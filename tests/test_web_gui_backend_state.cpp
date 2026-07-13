#include "ui/WebGuiServer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

static void require_contains(const std::string& text, const std::string& token) {
    if (text.find(token) == std::string::npos) {
        throw std::runtime_error("missing token in GUI state: " + token + "\n" + text);
    }
}

int main() {
    ui::WebGuiConfig cfg;
    cfg.port = 0;
    cfg.worktodo_path = "worktodo.txt";
    cfg.config_path = "settings.cfg";
    cfg.results_path = "results.txt";
    ui::WebGuiServer server(cfg, [](const std::string&){});
    server.setStatus("Running P-1 Stage 1");
    server.setProgress(927, 1442127, "Gerbicz OK");
    server.setBackendInfo("Auto", "Aevum", "P-1",
                          "profile=P-1 Stage 1, ratio=0.50, limit=0.75",
                          4194304, 8388608, "1:1K:8:256:101");
    const std::string json = server.stateJson();
    require_contains(json, "\"backend_mode\":\"Auto\"");
    require_contains(json, "\"backend_active\":\"Aevum\"");
    require_contains(json, "\"backend_workload\":\"P-1\"");
    require_contains(json, "\"backend_aevum_transform\":4194304");
    require_contains(json, "\"backend_marin_transform\":8388608");
    require_contains(json, "1:1K:8:256:101");
    require_contains(json, "\"current\":927");
    std::cout << "Web GUI backend state test passed" << std::endl;
    return 0;
}
