// When the AUTO backend picks Aevum and the plugin cannot create its engine (for example its working directories
// cannot be created), PrMers must fall back to Marin with a warning instead of aborting the run.  A forced -aevum
// must still fail, naming the plugin's reason.  The fake plugin (aevum_plan_report_engine.cpp) fails engine
// creation with FAKE_AEVUM_CREATE_ERROR as its error text.
#include "marin/engine.h"
#include "ui/WebGuiServer.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

const char* const kReason = "filesystem error: cannot create directory: Read-only file system [.aevum-proof]";

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct Attempt {
    std::unique_ptr<engine> eng;
    std::string out;
    std::string error;
};

Attempt create(engine::gpu_backend backend) {
    engine::configure_gpu_backend(backend, "", engine::gpu_workload::prp);
    Attempt a;
    std::ostringstream captured;
    std::streambuf* original = std::cout.rdbuf(captured.rdbuf());
    try {
        a.eng.reset(engine::create_gpu(180000000u, 8, 0, false));
    } catch (const std::exception& e) {
        a.error = e.what();
    }
    std::cout.rdbuf(original);
    a.out = captured.str();
    std::cout << a.out;
    return a;
}

int run() {
    setenv("AEVUM_AUTOTUNE", "off", 1);
    setenv("FAKE_AEVUM_CREATE_ERROR", kReason, 1);
    auto gui = std::make_shared<ui::WebGuiServer>(ui::WebGuiConfig{}, [](const std::string&) {});
    ui::WebGuiServer::setInstance(gui);

    // AUTO: the policy selects Aevum, creation fails, Marin takes over.  Building the Marin engine needs an
    // OpenCL device; without one it may throw too, but never with the Aevum failure.
    {
        const Attempt a = create(engine::gpu_backend::auto_select);
        require(a.out.find("[Backend Auto] PRP: Aevum selected") != std::string::npos,
                "the auto policy should have selected Aevum first:\n" + a.out);
        require(a.out.find(std::string("Marin fallback: Aevum create failed: ") + kReason) != std::string::npos,
                "AUTO must warn that it falls back to Marin, with the plugin's reason:\n" + a.out);
        require(a.error.find("Aevum create failed") == std::string::npos,
                "AUTO must not abort on an Aevum creation failure: " + a.error);
        require(!a.eng || !a.eng->is_aevum_backend(), "AUTO fallback must run Marin");
        const std::string state = gui->stateJson();
        require(state.find("\"backend_active\":\"Marin\"") != std::string::npos,
                "GUI must show Marin as the active backend: " + state);
    }

    // Forced -aevum: no fallback, a clear error that carries the plugin's reason.
    {
        const Attempt a = create(engine::gpu_backend::aevum);
        require(!a.eng, "forced Aevum must not silently run another engine");
        require(a.error.find("Forced Aevum engine could not be created") != std::string::npos &&
                    a.error.find(kReason) != std::string::npos,
                "forced Aevum must fail naming the plugin's reason: " + a.error);
        const std::string state = gui->stateJson();
        require(state.find("\"backend_mode\":\"Forced Aevum rejected\"") != std::string::npos,
                "GUI must show the forced Aevum rejection: " + state);
    }

    ui::WebGuiServer::setInstance(nullptr);
    std::cout << "Aevum create-fallback test passed" << std::endl;
    return 0;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
