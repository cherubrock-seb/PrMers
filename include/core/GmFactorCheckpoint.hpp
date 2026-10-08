#pragma once

// Checkpoint file shared by the Gaussian-Mersenne P-1/ECM factoring drivers
// (the legacy product-exponent drivers and the P-1 V-trace Stage 1).
//
// The file is a small header followed by the engine checkpoint of an engine
// with the same register count that wrote it, so every writer of a given file
// kind must allocate the same number of registers.  The P-1 drivers use
// PM1_WINDOW_REGS registers.  `Target` is any type with `p` and `lift` members.

#include "marin/engine.h"
#include "marin/file.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace core::gm_factor_ckpt {

inline constexpr std::array<char, 8> GMF_MAGIC{{'P','R','G','M','F','A','C','T'}};
inline constexpr std::uint32_t GMF_VERSION = 3;

// Register count of the engine behind the P-1 Stage 1/Stage 2 checkpoints.
inline constexpr std::size_t PM1_WINDOW_REGS = 15;

struct FactorCheckpointHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t mode;        // 1=P-1, 2=ECM
    std::uint32_t phase;       // 1=stage1, 2=stage2
    std::uint32_t p;
    std::uint32_t lift;
    std::uint32_t base;
    std::uint32_t curve;
    std::uint64_t B1;
    std::uint64_t B2;
    std::uint64_t token;       // remaining bits or next prime index
    std::uint64_t scalar_bits;
    std::uint64_t sigma;
    double elapsed;
    std::uint64_t checkpoint_bytes;
};

template <class Target>
inline bool matching_header(const FactorCheckpointHeader& h,
                     std::uint32_t mode,
                     std::uint32_t phase,
                     const Target& t,
                     std::uint64_t B1,
                     std::uint64_t B2,
                     std::uint64_t scalar_bits,
                     std::uint32_t base,
                     std::uint32_t curve,
                     std::uint64_t sigma,
                     std::size_t bytes) {
    return std::equal(GMF_MAGIC.begin(), GMF_MAGIC.end(), h.magic) &&
           h.version >= 2 && h.version <= GMF_VERSION &&
           h.mode == mode && h.phase == phase &&
           h.p == t.p && h.lift == t.lift && h.B1 == B1 && h.B2 == B2 &&
           h.scalar_bits == scalar_bits && h.base == base && h.curve == curve &&
           h.sigma == sigma && h.checkpoint_bytes == bytes;
}

template <class Target>
inline bool load_factor_checkpoint(const std::filesystem::path& path,
                            engine* eng,
                            std::uint32_t mode,
                            std::uint32_t phase,
                            const Target& t,
                            std::uint64_t B1,
                            std::uint64_t B2,
                            std::uint64_t scalar_bits,
                            std::uint32_t base,
                            std::uint32_t curve,
                            std::uint64_t sigma,
                            std::uint64_t& token,
                            double& elapsed) {
    File f(path.string());
    if (!f.exists()) return false;
    FactorCheckpointHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (!matching_header(h, mode, phase, t, B1, B2, scalar_bits, base, curve, sigma,
                         eng->get_checkpoint_size())) return false;
    std::vector<char> data(eng->get_checkpoint_size());
    if (!f.read(data.data(), data.size()) || !f.check_crc32() || !eng->set_checkpoint(data)) return false;
    token = h.token;
    elapsed = h.elapsed;
    if (h.version != GMF_VERSION) {
        std::cout << "Loaded legacy Gaussian factoring checkpoint v" << h.version
                  << "; it will be upgraded to v" << GMF_VERSION << " on the next save.\n";
    }
    return true;
}

// Writes `header` + `data` + CRC to `<path>.new` and installs it as `path`, keeping
// the previous file as `<path>.old`.  Throws std::runtime_error when the file
// cannot be created or completely written (including a failed flush or close,
// where a full disk is reported) or installed; the previous checkpoint is then
// untouched and the partial `.new` file is removed.  The layout and CRC are the
// ones `File` reads back, so every Gaussian-Mersenne checkpoint format that is a
// fixed header followed by the engine state can use it.
inline void write_checkpoint_blob(const std::filesystem::path& path,
                                  const void* header,
                                  std::size_t header_bytes,
                                  const std::vector<char>& data) {
    const std::filesystem::path new_path = path.string() + ".new";
    const std::filesystem::path old_path = path.string() + ".old";
    std::FILE* f = std::fopen(new_path.string().c_str(), "wb");
    if (f == nullptr) {
        throw std::runtime_error("cannot create " + new_path.string() + ": " + std::strerror(errno));
    }
    std::uint32_t crc = File::rc_crc32(0, reinterpret_cast<const char*>(header), header_bytes);
    crc = File::rc_crc32(crc, data.data(), data.size());
    const std::uint32_t tail = ~crc ^ 0xa23777acU;
    errno = 0;
    bool ok = std::fwrite(header, header_bytes, 1, f) == 1 &&
              (data.empty() || std::fwrite(data.data(), 1, data.size(), f) == data.size()) &&
              std::fwrite(&tail, sizeof(tail), 1, f) == 1 &&
              std::fflush(f) == 0 && std::ferror(f) == 0;
    const int write_errno = errno;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) {
        std::error_code rec;
        std::filesystem::remove(new_path, rec);
        throw std::runtime_error("cannot write checkpoint " + new_path.string() +
                                 (write_errno != 0 ? std::string(": ") + std::strerror(write_errno) : std::string()));
    }
    std::error_code ec;
    std::filesystem::remove(old_path, ec);
    ec.clear();
    if (std::filesystem::exists(path)) std::filesystem::rename(path, old_path, ec);
    ec.clear();
    std::filesystem::rename(new_path, path, ec);
    if (ec) {
        std::error_code rec;
        std::filesystem::remove(new_path, rec);
        throw std::runtime_error("cannot install checkpoint " + path.string() + ": " + ec.message());
    }
}

inline void write_checkpoint_file(const std::filesystem::path& path,
                                  const FactorCheckpointHeader& h,
                                  const std::vector<char>& data) {
    write_checkpoint_blob(path, &h, sizeof(h), data);
}

template <class Target>
inline void save_factor_checkpoint(const std::filesystem::path& path,
                            engine* eng,
                            std::uint32_t mode,
                            std::uint32_t phase,
                            const Target& t,
                            std::uint64_t B1,
                            std::uint64_t B2,
                            std::uint64_t scalar_bits,
                            std::uint32_t base,
                            std::uint32_t curve,
                            std::uint64_t sigma,
                            std::uint64_t token,
                            double elapsed) {
    eng->sync();
    FactorCheckpointHeader h{};
    std::copy(GMF_MAGIC.begin(), GMF_MAGIC.end(), h.magic);
    h.version = GMF_VERSION;
    h.mode = mode;
    h.phase = phase;
    h.p = t.p;
    h.lift = t.lift;
    h.base = base;
    h.curve = curve;
    h.B1 = B1;
    h.B2 = B2;
    h.token = token;
    h.scalar_bits = scalar_bits;
    h.sigma = sigma;
    h.elapsed = elapsed;
    h.checkpoint_bytes = eng->get_checkpoint_size();

    std::vector<char> data(eng->get_checkpoint_size());
    if (!eng->get_checkpoint(data)) throw std::runtime_error("cannot read Gaussian factoring checkpoint");
    write_checkpoint_file(path, h, data);
}

// Runs `save` (a call to save_factor_checkpoint); a failure is reported as a
// warning and returns false instead of ending the run.  A failed save leaves
// the previous checkpoint in place, so the computation can go on and try again
// at the next backup.  Returns true when the checkpoint was written.
template <class Save>
inline bool guarded_save(const std::filesystem::path& path, Save&& save) {
    try {
        save();
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[GM] Warning: checkpoint " << path.string() << " was not saved: " << ex.what()
                  << ". Continuing; the previous checkpoint, if any, is kept.\n";
        return false;
    }
}

// save_factor_checkpoint that reports a failure instead of throwing.
template <class Target>
inline bool try_save_factor_checkpoint(const std::filesystem::path& path,
                                engine* eng,
                                std::uint32_t mode,
                                std::uint32_t phase,
                                const Target& t,
                                std::uint64_t B1,
                                std::uint64_t B2,
                                std::uint64_t scalar_bits,
                                std::uint32_t base,
                                std::uint32_t curve,
                                std::uint64_t sigma,
                                std::uint64_t token,
                                double elapsed) {
    return guarded_save(path, [&]() {
        save_factor_checkpoint(path, eng, mode, phase, t, B1, B2, scalar_bits, base, curve,
                               sigma, token, elapsed);
    });
}

inline void clear_checkpoint(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".old", ec);
    std::filesystem::remove(path.string() + ".new", ec);
}


} // namespace core::gm_factor_ckpt
