// include/core/CheckpointLength.hpp
//
// Host-only helper for checkpoint loaders that read a length-prefixed string
// (a uint32 length followed by that many hex digits).
//
// The CRC of a checkpoint is only checked at the very end of the file, after
// these fields.  A damaged or foreign file can therefore carry any 32-bit
// length, and sizing a std::string from it before the CRC is known zero-fills
// up to 4 GiB or throws bad_alloc out of the loader, so a bad checkpoint kills
// the run instead of being rejected like any other CRC failure.  A length can
// never exceed the size of the file that holds the string, so the helper
// rejects any larger value before it allocates anything.  The on-disk format
// is unchanged.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace core {
namespace ckptlen {

// Size of `path` in bytes, or 0 when it cannot be determined (every non-empty
// field is then rejected, i.e. the file is treated as unreadable).
inline uint64_t fileSizeOf(const std::string& path) {
    std::error_code ec;
    const std::uintmax_t n = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(n);
}

// Reads  uint32 length | length bytes  into `out`.  F is any type with
// `bool read(char*, size_t)` (the marin File class).  Returns false (leaving
// `out` empty and allocating nothing) for a short read or when the length is
// larger than `fileSize`; the caller then treats the checkpoint as unreadable.
template <class F>
inline bool readLenPrefixed(F& f, uint64_t fileSize, std::string& out) {
    out.clear();
    uint32_t len = 0;
    if (!f.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
    if (static_cast<uint64_t>(len) > fileSize) return false;
    if (len == 0) return true;
    out.resize(len);
    if (!f.read(&out[0], len)) { out.clear(); return false; }
    return true;
}

}  // namespace ckptlen
}  // namespace core
