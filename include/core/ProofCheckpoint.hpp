// core/ProofCheckpoint.hpp
//
// Shared by the legacy (ProofSet) and Marin (ProofSetMarin) proof sets:
// writing one proof checkpoint residue with a read-back check.
#pragma once

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace core {

// Thrown when a proof checkpoint residue could not be written or does not read
// back as written, even after a retry. The PRP itself is unaffected; callers
// stop collecting residues and report the result without a proof.
class ProofCheckpointError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Ask the OS to push a file written through a stream to the disk, so that a
// write error is reported here and not after the test is over. Best effort:
// returns false only when the file could be opened but not synced.
inline bool syncFileToDisk(const std::filesystem::path& path) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return true;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
#else
    (void)path;
    return true;
#endif
}

// Save `words` as the residue of iteration `iter` and read it back, comparing
// with what was written. On any failure the residue is written once more; a
// second failure throws ProofCheckpointError. `Set` provides
// save(iter, words) and load(iter).
template <class Set>
void saveProofCheckpointVerified(Set& set, uint32_t iter,
                                 const std::vector<uint32_t>& words) {
    std::string lastError;
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            set.save(iter, words);
            const std::vector<uint32_t> loaded = set.load(iter);
            if (loaded.size() != words.size()) {
                throw std::runtime_error(
                    "size mismatch (" + std::to_string(words.size()) + " vs " +
                    std::to_string(loaded.size()) + ")");
            }
            for (size_t i = 0; i < words.size(); ++i) {
                if (words[i] != loaded[i]) {
                    throw std::runtime_error(
                        "data mismatch at word " + std::to_string(i));
                }
            }
            return;
        } catch (const std::exception& e) {
            lastError = e.what();
            std::cerr << "Warning: proof checkpoint at iteration " << iter
                      << " failed: " << lastError
                      << (attempt == 0 ? "; writing it again" : "")
                      << std::endl;
        }
    }
    throw ProofCheckpointError(
        "proof checkpoint at iteration " + std::to_string(iter) +
        " could not be written and verified: " + lastError);
}

} // namespace core
