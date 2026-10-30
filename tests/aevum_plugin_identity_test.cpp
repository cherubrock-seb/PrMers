// Prints the identity of the Aevum plugin loaded through AEVUM_ENGINE_LIB; the backend probe cache key includes it.
#include "aevum/EngineAevum.hpp"

#include <iostream>

int main() {
    std::cout << aevum_plugin_identity() << std::endl;
    return 0;
}
