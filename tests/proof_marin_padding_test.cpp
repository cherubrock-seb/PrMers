#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/ProofMarin.hpp"

int main() {
    constexpr uint32_t E = 191;

    std::vector<uint32_t> shortWords{
        0x89abcdefu,
        0x01234567u
    };

    std::vector<uint32_t> fullWords = shortWords;
    fullWords.resize((E + 31u) / 32u, 0u);

    if (core::ProofMarin::hashWords(E, shortWords) !=
        core::ProofMarin::hashWords(E, fullWords)) {
        std::cerr << "zero-padding hash mismatch\n";
        return 1;
    }

    const std::array<uint64_t, 4> prefix{
        1u, 2u, 3u, 4u
    };

    if (core::ProofMarin::hashWords(E, prefix, shortWords) !=
        core::ProofMarin::hashWords(E, prefix, fullWords)) {
        std::cerr << "prefixed zero-padding hash mismatch\n";
        return 2;
    }

    const auto stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();

    const auto file =
        std::filesystem::temp_directory_path() /
        ("prmers-proof-marin-" +
         std::to_string(stamp) + ".proof");

    core::ProofMarin proof(
        E,
        fullWords,
        std::vector<std::vector<uint32_t>>{fullWords},
        std::vector<std::string>{
            "7801609129",
            "134921946097"
        });

    proof.save(file);
    auto loaded = core::ProofMarin::load(file);
    std::filesystem::remove(file);

    if (loaded.E != E ||
        loaded.B != fullWords ||
        loaded.middles.size() != 1 ||
        loaded.middles[0] != fullWords ||
        loaded.knownFactors.size() != 2) {
        std::cerr << "proof save/load mismatch\n";
        return 3;
    }

    std::cout
        << "ProofMarin canonical padding regression: PASS\n";
    return 0;
}
