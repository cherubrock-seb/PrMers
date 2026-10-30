// core/CheckpointLocation.hpp
//
// Where a run keeps one checkpoint family (<name>, <name>.old, <name>.new). The checkpoint
// lives under the save path (-f), next to the other state of the run, or in the working
// directory when -f is empty. Earlier versions kept it in the working directory whatever -f
// said, so a run that finds nothing under -f but a checkpoint of that name in the working
// directory resumes from it (adoptLegacy) and removes it once its own is written or the job is
// done (removeLegacy). The same rules as the PRP/LL checkpoint and the proof files.
#pragma once

#include "core/ProofLocation.hpp"

#include <filesystem>
#include <string>
#include <system_error>

namespace core {

class CheckpointLocation {
public:
    CheckpointLocation(const std::string& savePath, const std::string& name)
        : name_(name),
          dir_(savePath.empty() ? std::filesystem::path(".") : std::filesystem::path(savePath)),
          dirIsCwd_(ProofLocation::samePlace(dir_, ".")),
          file_(dirIsCwd_ ? name : (dir_ / name).string()) {}

    // The path to read and write (the bare name when the save path is the working directory).
    const std::string& file() const { return file_; }
    // The bare name: where earlier versions kept the checkpoint, relative to the working directory.
    const std::string& legacyFile() const { return name_; }
    const std::filesystem::path& dir() const { return dir_; }
    bool underSavePath() const { return !dirIsCwd_; }

    // Call before writing: the save path is created when the first checkpoint is saved.
    void prepareWrite() const {
        if (dirIsCwd_) return;
        std::error_code ec;
        std::filesystem::create_directories(dir_, ec);
    }

    // True when neither the checkpoint nor its .old exists under the save path, so that one of
    // the same name in the working directory (an older version's) may be read instead.
    bool legacyCandidate() const {
        if (dirIsCwd_) return false;
        std::error_code ec;
        return !std::filesystem::exists(file_, ec) && !std::filesystem::exists(file_ + ".old", ec);
    }

    // The run resumed from the old location: remove it once the run's own checkpoint is
    // written, or at the end of the job.
    void adoptLegacy() { legacyPending_ = true; }
    bool legacyPending() const { return legacyPending_; }
    void removeLegacy() {
        if (!legacyPending_) return;
        std::error_code ec;
        std::filesystem::remove(name_, ec);
        std::filesystem::remove(name_ + ".old", ec);
        std::filesystem::remove(name_ + ".new", ec);
        legacyPending_ = false;
    }

    // Removes the checkpoint family: the one under the save path and any old-location copy
    // this run adopted.
    void removeAll() {
        std::error_code ec;
        std::filesystem::remove(file_, ec);
        std::filesystem::remove(file_ + ".old", ec);
        std::filesystem::remove(file_ + ".new", ec);
        removeLegacy();
    }

    // The note a run prints when it resumes from the old location.
    std::string legacyNote() const {
        return "[Checkpoint] No checkpoint under " + dir_.string() + "; using " + name_ +
               " from the working directory (an older version kept it there).";
    }

private:
    std::string name_;
    std::filesystem::path dir_;
    bool dirIsCwd_;
    std::string file_;
    bool legacyPending_ = false;
};

}  // namespace core
