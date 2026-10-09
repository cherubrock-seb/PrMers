// include/core/BackupManager.hpp
#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdint>
#include <string>
#include <vector>
#include <gmpxx.h>

namespace core {

// Legacy (-marin) NTT checkpoints.
//
// A checkpoint is a set of files: the residue (.mers), the iteration (.loop) and, with Gerbicz-Li
// checking, .bufd/.lbufd/.gli/.isav/.jsav; P-1 stage 2 keeps .hq/.q/.loop2. Their formats are
// unchanged (the .mers is a raw image of the digit vector that -filemers also reads, the .loop a
// decimal count). A set is saved as follows:
//   1. every file is written to "<file>.new", flushed, checked and synced;
//   2. a manifest "<base>.state" ("<b2 base>.state" for stage 2) listing the size and CRC-32 of
//      every file of the set is written the same way; the previous manifest is copied to
//      "<base>.state.old" and the new one renamed over the old: this is the commit;
//   3. each "<file>.new" is renamed over "<file>", the previous file becoming "<file>.old".
// A kill at any point leaves a manifest whose files can each be found, by size and CRC, as
// "<file>", "<file>.new" or "<file>.old"; the reader uses only a complete matching set, tries
// "<base>.state.old" next, and otherwise starts over with a warning. Files written by earlier
// versions have no manifest and are read as before (no file is changed before the first commit).
class BackupManager {
public:
    // The Gerbicz-Li buffers saved with the state (a -marin PRP with Gerbicz-Li checking).
    struct GerbiczLiBuffers {
        cl_mem correct;     // last verified state            -> .gli
        cl_mem bufd;        // running product                -> .bufd
        cl_mem lastBufd;    // product at the last check      -> .lbufd
        uint64_t itersave;  // iteration of the last check    -> .isav
        uint64_t jsave;     // j at the last check            -> .jsav
    };
    struct GerbiczLiHost {
        std::vector<uint64_t> correct, bufd, lastBufd;
        uint64_t itersave = 0, jsave = 0;
    };

    BackupManager(cl_command_queue queue,
                  unsigned interval,
                  size_t vectorSize,
                  const std::string& savePath,
                  unsigned exponent,
                  const std::string& mode,
                  const uint64_t b1,
                  const uint64_t b2,
                  bool wagstaff,
                  bool marin
                  );

    // read the saved set into x (and keep its Gerbicz-Li part for the calls below); return the
    // resume iteration, 0 for a fresh start
    uint64_t loadState(std::vector<uint64_t>& x);
    void loadGerbiczLiBufDState(std::vector<uint64_t>& x);
    void loadGerbiczLiCorrectState(std::vector<uint64_t>& x);
    void loadGerbiczLiCorrectBufDState(std::vector<uint64_t>& x);
    uint64_t loadGerbiczIterSave();
    uint64_t loadGerbiczJSave();
    
    uint64_t loadStatePM1S2(cl_mem hqBuf, cl_mem qBuf, size_t bytes);
    void     saveStatePM1S2(cl_mem hqBuf, cl_mem qBuf, uint64_t idx, size_t bytes);
    // read back from the device and save the set at iteration iter (.loop records iter + 1),
    // with the Gerbicz-Li buffers when gl is given
    void saveState(cl_mem buffer, uint64_t iter, const mpz_class* E_ptr = nullptr,
                   const GerbiczLiBuffers* gl = nullptr);

    // Host forms of the saves (also used by the tests); return false if the set was not committed.
    bool saveStateHost(const std::vector<uint64_t>& x, uint64_t iter, const GerbiczLiHost* gl = nullptr);
    bool saveStatePM1S2Host(const std::vector<uint64_t>& hq, const std::vector<uint64_t>& q, uint64_t idx);
    // Stage 2 resume from host vectors (also used by the tests); returns the resume index or 0.
    uint64_t loadStatePM1S2Host(std::vector<uint64_t>& hq, std::vector<uint64_t>& q);

    mpz_class loadExponent() const;

    void clearState() const;

    // A component of a checkpoint set: the file suffix (without the dot) and its contents.
    struct Component { std::string suffix; std::string bytes; };
private:
    bool saveSet(const std::string& prefix, const std::vector<Component>& parts, uint64_t loopValue);
    // 1: a complete set was loaded into parts; 0: no manifest (files of an earlier version);
    // -1: a manifest exists but no complete set matches it
    int loadSet(const std::string& prefix, const char* loopSuffix, std::vector<Component>& parts,
                uint64_t& loopValue);

    cl_command_queue queue_;
    unsigned         backupInterval_;
    size_t           vectorSize_;
    std::string      savePath_;
    unsigned         exponent_;
    std::string      mode_;
    std::string      basePrefix_;
    std::string      b2Prefix_;
    std::string      mersFilename_;
    std::string      loopFilename_;
    std::string      GerbiczLiBufDFilename_;
    std::string      GerbiczLiCorrectBufFilename_;
    std::string      GerbiczLiIterSaveFilename_;
    std::string      GerbiczLiJSaveFilename_;
    std::string      GerbiczLiLastBufDFilename_;
    std::string      exponentFilename_;
    uint64_t b1_;
    uint64_t b2_;
    bool             wagstaff_;
    bool             marin_;
    bool             stateDiscarded_ = false; // loadState rejected an unusable state file
    // loadState read a set through its manifest: the Gerbicz-Li part below comes from that set
    // (a part the set does not hold keeps its default) instead of from the files.
    bool             setLoaded_ = false;
    bool             glHasBufd_ = false, glHasCorrect_ = false, glHasLastBufd_ = false;
    std::vector<uint64_t> glBufd_, glCorrect_, glLastBufd_;
    uint64_t         glIterSave_ = 0, glJSave_ = 0;
    std::string hqFilename_, qFilename_, loop2Filename_;

};

} // namespace core
