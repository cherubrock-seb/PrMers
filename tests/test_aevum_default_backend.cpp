#include "io/CliParser.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    io::CliOptions options;
    if (!options.marin) throw std::runtime_error("Marin engine path must remain available");
    if (options.aevum) throw std::runtime_error("Aevum must not be forced by default");
    if (!options.aevum_auto) throw std::runtime_error("automatic backend selection must be the default");
    if (options.force_engine_marin) throw std::runtime_error("Marin must not be forced by default");
    std::cout << "Default backend policy test passed" << std::endl;
    return 0;
}
