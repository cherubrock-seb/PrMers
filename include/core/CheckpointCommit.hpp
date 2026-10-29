// include/core/CheckpointCommit.hpp
//
// Host-only helper that publishes a freshly written checkpoint.
//
// A checkpoint writer saves to `<ckpt>.new` and then swaps it over `<ckpt>`.
// The old stage-1 P-1 / ECM writers ignored a failed final write, flush or
// close (marin `File` removes such a file), then ran
//   remove(.old); rename(ckpt, .old); rename(.new, ckpt); remove(.old)
// without looking at any result.  With the `.new` file gone the second rename
// fails and the last remove deleted the only good checkpoint.
//
// Writer side:  `ok = f.write_crc32() && f.close();`  and stop when !ok.
// Commit side:  commit(ckpt) below.  `<ckpt>` is replaced only after
// `<ckpt>.new` is known to exist; `<ckpt>.old` is removed only after the swap
// succeeded; when the second rename fails the previous checkpoint is moved
// back.  The on-disk checkpoint format is not touched.
#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace core {
namespace ckptcommit {

// `renameFn(from, to, ec)` is fs::rename in production; tests inject failures.
template <class RenameFn>
inline bool commitWith(const std::string& ckpt, bool keepOld, RenameFn renameFn) {
    namespace fs = std::filesystem;
    const std::string newf = ckpt + ".new", oldf = ckpt + ".old";
    std::error_code ec;
    if (!fs::is_regular_file(newf, ec) || ec) return false;   // a directory must never become `<ckpt>`
    const bool hadCkpt = fs::exists(ckpt, ec) && !ec;
    ec.clear();
    if (hadCkpt) {
        fs::remove(oldf, ec);
        ec.clear();
        renameFn(ckpt, oldf, ec);
        if (ec) {                       // `<ckpt>` is still in place
            std::error_code ec2;
            fs::remove(newf, ec2);
            return false;
        }
    }
    renameFn(newf, ckpt, ec);
    if (ec) {
        std::error_code ec2;
        if (hadCkpt) fs::rename(oldf, ckpt, ec2);   // put the previous checkpoint back
        fs::remove(newf, ec2);
        return false;
    }
    if (hadCkpt && !keepOld) {
        std::error_code ec2;
        fs::remove(oldf, ec2);
    }
    return true;
}

// Replaces `<ckpt>` by `<ckpt>.new`.  Returns false, leaving `<ckpt>` as it was
// and no `<ckpt>.new` behind, when `<ckpt>.new` is missing or a rename fails.
// keepOld = false removes `<ckpt>.old` after a successful swap (stage 1);
// keepOld = true keeps the previous checkpoint as `<ckpt>.old` (stage 2).
inline bool commit(const std::string& ckpt, bool keepOld = false) {
    return commitWith(ckpt, keepOld,
        [](const std::string& from, const std::string& to, std::error_code& ec) {
            std::filesystem::rename(from, to, ec);
        });
}

}  // namespace ckptcommit
}  // namespace core
