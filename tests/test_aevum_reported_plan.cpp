// The FFT plan PrMers reports (console and GUI) must be the plan the Aevum plugin created.
//
// With runtime autotune off the auto policy previews the native PRP geometry ("native-prp:auto"), but an engine
// created with an empty plan is chosen by the plugin, which may pick another one.  The fake plugin
// (aevum_plan_report_engine.cpp) models that: it previews 4:512:8:512:202 and creates 1:1K:16:256:101.
#include "marin/engine.h"
#include "ui/WebGuiServer.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

static void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

static int run() {
    constexpr const char* created_plan = "1:1K:16:256:101";
    constexpr const char* preview_plan = "4:512:8:512:202";
    constexpr std::uint32_t exponent = 180000000u;

    setenv("AEVUM_AUTOTUNE", "off", 1);
    engine::configure_gpu_backend(engine::gpu_backend::auto_select, "", engine::gpu_workload::prp);

    auto gui = std::make_shared<ui::WebGuiServer>(ui::WebGuiConfig{}, [](const std::string&) {});
    ui::WebGuiServer::setInstance(gui);

    std::ostringstream captured;
    std::streambuf* original = std::cout.rdbuf(captured.rdbuf());
    std::unique_ptr<engine> eng;
    try {
        eng.reset(engine::create_gpu(exponent, 8, 0, false));
    } catch (...) {
        std::cout.rdbuf(original);
        throw;
    }
    std::cout.rdbuf(original);
    const std::string out = captured.str();
    std::cout << out;

    require(eng && eng->is_aevum_backend(), "auto policy should have selected the Aevum engine");
    require(eng->get_size() == 8388608u, "the fake engine runs an 8M transform");

    // The engine reports the plan it runs.
    require(out.find(std::string("plan=") + created_plan) != std::string::npos,
            "the engine line must name the plan the plugin created");

#if !defined(__APPLE__)
    // The policy preview differs from what runs; the mismatch must be called out, not hidden.
    require(out.find(std::string("Aevum created FFT ") + created_plan + " (policy preview was " + preview_plan + ")") != std::string::npos,
            "a created plan that differs from the preview must be reported");
#endif

    // The GUI shows the created plan and its transform size, not the preview.
    const std::string state = gui->stateJson();
    require(state.find(std::string("\"backend_fft\":\"") + created_plan + "\"") != std::string::npos,
            "GUI backend_fft must be the created plan: " + state);
    require(state.find("\"backend_aevum_transform\":8388608") != std::string::npos,
            "GUI Aevum transform must be the created plan's size: " + state);

    ui::WebGuiServer::setInstance(nullptr);
    std::cout << "Aevum reported-plan test passed" << std::endl;
    return 0;
}

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
