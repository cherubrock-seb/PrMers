// src/core/BackupManager.cpp
/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software.
 */
#include "core/BackupManager.hpp"
#include "core/ProofCheckpoint.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <gmpxx.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <chrono>
#include <limits>
#include <ios>
#include <sstream>

namespace core {

BackupManager::BackupManager(cl_command_queue queue,
                             unsigned interval,
                             size_t vectorSize,
                             const std::string& savePath,
                             unsigned exponent,
                             const std::string& mode,
                             const uint64_t b1,
                             const uint64_t b2,
                             bool wagstaff,
                             bool marin)
  : queue_(queue)
  , backupInterval_(interval)
  , vectorSize_(vectorSize)
  , savePath_(savePath.empty() ? "." : savePath)
  , exponent_(exponent)
  , mode_(mode)
  , b1_(b1)
  , b2_(b2)
  , wagstaff_(wagstaff)
  , marin_(marin)
{
    if(!marin_){
        std::filesystem::create_directories(savePath_);
        auto base = std::to_string(exponent_) + mode_;
        if(b1_>0){
            base = std::to_string(exponent_) + mode_ + std::to_string(b1_);
        }
        if(wagstaff_){
            base = base + "_wagstaff";
        }
        basePrefix_ = savePath_ + "/" + base;
        GerbiczLiBufDFilename_ = basePrefix_ + ".bufd";
        GerbiczLiCorrectBufFilename_ = basePrefix_ + ".gli";
        GerbiczLiIterSaveFilename_ = basePrefix_ + ".isav";
        GerbiczLiJSaveFilename_ = basePrefix_ + ".jsav";
        GerbiczLiLastBufDFilename_ = basePrefix_ + ".lbufd";
        mersFilename_ = basePrefix_ + ".mers";
        loopFilename_ = basePrefix_ + ".loop";
        exponentFilename_ = basePrefix_ + ".exponent";
        if(b2_>0){
            base = std::to_string(exponent_) + mode_ + std::to_string(b1_) + "_" + (std::to_string(b2_));
            b2Prefix_     = savePath_ + "/" + base;
            hqFilename_   = b2Prefix_ + ".hq";
            qFilename_    = b2Prefix_ + ".q";
            loop2Filename_= b2Prefix_ + ".loop2";
        }
    }

}

static inline std::streamsize ss_from_size(size_t n) {
    using S = std::streamsize;
    const size_t Smax = static_cast<size_t>(std::numeric_limits<S>::max());
    return static_cast<S>(n > Smax ? Smax : n);
}


// Read exactly `words` 64-bit words from `file`. Legacy checkpoints are raw images of the transform-size
// digit vector, so a file of any other length was written for a different transform size (or truncated or
// extended) and must not be used. Returns false if the file is missing, has any other size, or is unreadable.
static bool readExactWords(const std::string& file, uint64_t* data, size_t words, const char* what) {
    std::error_code ec;
    if (!std::filesystem::exists(file, ec)) return false;
    const uintmax_t want = static_cast<uintmax_t>(words) * sizeof(uint64_t);
    const uintmax_t have = std::filesystem::file_size(file, ec);
    if (ec || have != want) {
        std::cerr << "Warning: " << what << " " << file << " has "
                  << (ec ? std::string("an unreadable size") : std::to_string(have) + " bytes")
                  << " but this transform size needs " << want << " bytes — ignoring it\n";
        return false;
    }
    std::ifstream in(file, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(data), ss_from_size(static_cast<size_t>(want)));
    return in.gcount() == ss_from_size(static_cast<size_t>(want));
}

namespace {

// ---- checkpoint sets -------------------------------------------------------------------------------

const char* const kManifestMagic = "prmers-legacy-state";
const unsigned kManifestVersion = 1;
const uintmax_t kMaxTextPart = 64;          // .loop/.loop2/.isav/.jsav hold one decimal number
const uintmax_t kMaxManifest = 64 * 1024;

uint32_t crc32(const void* data, size_t n) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::string hex32(uint32_t v) {
    std::ostringstream o;
    o << std::hex;
    o.width(8);
    o.fill('0');
    o << v;
    return o.str();
}

bool isWordsPart(const std::string& suffix) {
    return suffix == "mers" || suffix == "bufd" || suffix == "lbufd" || suffix == "gli" ||
           suffix == "hq" || suffix == "q";
}
bool isTextPart(const std::string& suffix) {
    return suffix == "loop" || suffix == "loop2" || suffix == "isav" || suffix == "jsav";
}

// Write `bytes` to `path`, check every step and push the file to the disk.
bool writeDurable(const std::string& path, const std::string& bytes) {
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(bytes.data(), ss_from_size(bytes.size()));
        out.flush();
        if (!out) return false;
        out.close();
        if (out.fail()) return false;
    }
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec || size != bytes.size()) return false;
    return syncFileToDisk(path);
}

// Read a whole regular file of at most maxBytes bytes.
bool readSmall(const std::string& path, std::string& out, uintmax_t maxBytes) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return false;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec || size > maxBytes) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(static_cast<size_t>(size), '\0');
    if (size) in.read(&out[0], ss_from_size(static_cast<size_t>(size)));
    return static_cast<uintmax_t>(in.gcount()) == size;
}

// Make the renames in the directory of `file` durable (best effort).
void syncDirectory(const std::string& file) {
#ifndef _WIN32
    std::filesystem::path dir = std::filesystem::path(file).parent_path();
    if (dir.empty()) dir = ".";
    const int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd >= 0) {
        (void)::fsync(fd);
        ::close(fd);
    }
#else
    (void)file;
#endif
}

// A decimal number and nothing else (surrounding whitespace allowed).
bool parseDecimal(const std::string& text, uint64_t& v) {
    std::istringstream in(text);
    uint64_t t = 0;
    if (!(in >> t)) return false;
    in >> std::ws;
    if (!in.eof()) return false;
    v = t;
    return true;
}

std::string wordsToBytes(const std::vector<uint64_t>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(uint64_t));
}

bool bytesToWords(const std::string& b, std::vector<uint64_t>& v) {
    if (b.size() != v.size() * sizeof(uint64_t)) return false;
    if (!b.empty()) std::memcpy(v.data(), b.data(), b.size());
    return true;
}

const BackupManager::Component* findPart(const std::vector<BackupManager::Component>& parts,
                                         const std::string& suffix) {
    for (const auto& p : parts) if (p.suffix == suffix) return &p;
    return nullptr;
}

}  // namespace

bool BackupManager::saveSet(const std::string& prefix, const std::vector<Component>& parts,
                            uint64_t loopValue) {
    const std::string manifest = prefix + ".state";
    auto fail = [&](const std::string& what) {
        std::cerr << "Error saving state: " << what
                  << " — the previous saved state is kept" << std::endl;
        std::error_code ec;
        for (const auto& p : parts) std::filesystem::remove(prefix + "." + p.suffix + ".new", ec);
        std::filesystem::remove(manifest + ".new", ec);
        return false;
    };

    // 1. Every file of the set as "<file>.new": nothing the current manifest names is touched.
    std::ostringstream m;
    m << kManifestMagic << ' ' << kManifestVersion << '\n'
      << "exponent " << exponent_ << '\n'
      << "words " << vectorSize_ << '\n'
      << "iteration " << loopValue << '\n';
    for (const auto& p : parts) {
        const std::string tmp = prefix + "." + p.suffix + ".new";
        if (!writeDurable(tmp, p.bytes)) return fail("could not write " + tmp);
        m << "file " << p.suffix << ' ' << p.bytes.size() << ' '
          << hex32(crc32(p.bytes.data(), p.bytes.size())) << '\n';
    }
    std::string text = m.str();
    text += "end " + hex32(crc32(text.data(), text.size())) + "\n";

    // 2. The manifest: keep a copy of the current one as ".old", then rename the new one over it.
    // This rename is the commit.
    if (!writeDurable(manifest + ".new", text)) return fail("could not write " + manifest + ".new");
    std::error_code ec;
    std::string previous;
    if (readSmall(manifest, previous, kMaxManifest)) {
        if (writeDurable(manifest + ".old.new", previous)) {
            std::filesystem::rename(manifest + ".old.new", manifest + ".old", ec);
            if (ec) std::filesystem::remove(manifest + ".old.new", ec);
        }
    }
    ec.clear();
    std::filesystem::rename(manifest + ".new", manifest, ec);
    if (ec) return fail("could not rename " + manifest + ".new: " + ec.message());
    syncDirectory(manifest);

    // 3. Each new file takes its place; the one it replaces becomes ".old" (the set that
    // "<base>.state.old" names). A file this leaves as ".new" is still found on resume.
    bool placed = true;
    for (const auto& p : parts) {
        const std::string file = prefix + "." + p.suffix;
        ec.clear();
        if (std::filesystem::exists(file, ec)) {
            std::filesystem::rename(file, file + ".old", ec);
            if (ec) { placed = false; continue; }
        }
        std::filesystem::rename(file + ".new", file, ec);
        if (ec) placed = false;
    }
    syncDirectory(manifest);
    if (!placed)
        std::cerr << "Warning: some state files under " << prefix
                  << " are still named .new; they will be found on resume" << std::endl;
    return true;
}

int BackupManager::loadSet(const std::string& prefix, const char* loopSuffix,
                           std::vector<Component>& parts, uint64_t& loopValue) {
    const std::string manifest = prefix + ".state";
    std::error_code ec;
    if (!std::filesystem::exists(manifest, ec) && !std::filesystem::exists(manifest + ".old", ec))
        return 0;

    for (const std::string& cand : {manifest, manifest + ".old"}) {
        if (!std::filesystem::exists(cand, ec)) continue;
        auto reject = [&](const std::string& why) {
            std::cerr << "Warning: saved state " << cand << " is not usable: " << why << std::endl;
        };
        std::string text;
        if (!readSmall(cand, text, kMaxManifest)) { reject("unreadable"); continue; }
        const size_t endPos = text.rfind("end ");
        if (endPos == std::string::npos || (endPos != 0 && text[endPos - 1] != '\n')) {
            reject("incomplete");
            continue;
        }
        {
            std::istringstream e(text.substr(endPos + 4));
            std::string crc, rest;
            e >> crc;
            if (crc != hex32(crc32(text.data(), endPos)) || (e >> rest)) {
                reject("checksum mismatch");
                continue;
            }
        }
        std::istringstream in(text.substr(0, endPos));
        std::string magic;
        unsigned version = 0;
        if (!(in >> magic >> version) || magic != kManifestMagic) { reject("not a state manifest"); continue; }
        if (version != kManifestVersion) {
            reject("format version " + std::to_string(version) + " (written by another version of PrMers)");
            continue;
        }
        std::string key;
        uint64_t exponent = 0, words = 0, iteration = 0;
        bool haveIter = false, bad = false;
        struct Entry { std::string suffix; uint64_t size; std::string crc; };
        std::vector<Entry> entries;
        while (!bad && in >> key) {
            if (key == "exponent") { if (!(in >> exponent)) bad = true; }
            else if (key == "words") { if (!(in >> words)) bad = true; }
            else if (key == "iteration") { if (!(in >> iteration)) bad = true; else haveIter = true; }
            else if (key == "file") {
                Entry en;
                if (!(in >> en.suffix >> en.size >> en.crc)) { bad = true; break; }
                if (!isWordsPart(en.suffix) && !isTextPart(en.suffix)) { bad = true; break; }
                if (isTextPart(en.suffix) && en.size > kMaxTextPart) { bad = true; break; }
                for (const auto& o : entries) if (o.suffix == en.suffix) bad = true;
                entries.push_back(en);
            } else { bad = true; }
        }
        if (bad || !haveIter) { reject("malformed"); continue; }
        if (exponent != exponent_ || words != vectorSize_) {
            reject("written for exponent " + std::to_string(exponent) + " with " + std::to_string(words) +
                   " words, not " + std::to_string(exponent_) + " with " + std::to_string(vectorSize_));
            continue;
        }
        const uint64_t wordsBytes = static_cast<uint64_t>(vectorSize_) * sizeof(uint64_t);
        if (std::any_of(entries.begin(), entries.end(),
                        [&](const Entry& en) { return isWordsPart(en.suffix) && en.size != wordsBytes; })) {
            reject("a buffer of the wrong size");
            continue;
        }
        if (!std::any_of(entries.begin(), entries.end(), [&](const Entry& en) { return en.suffix == loopSuffix; })) {
            reject(std::string("no .") + loopSuffix + " in the set");
            continue;
        }

        // Each file of the set, by size and checksum, under its own name or as ".new"/".old".
        std::vector<Component> found;
        std::string missing;
        bool moved = false;
        for (const auto& en : entries) {
            const std::string file = prefix + "." + en.suffix;
            bool ok = false;
            for (const std::string& f : {file, file + ".new", file + ".old"}) {
                std::string bytes;
                if (!std::filesystem::is_regular_file(f, ec)) continue;
                const uintmax_t size = std::filesystem::file_size(f, ec);
                if (ec || size != en.size) continue;
                if (!readSmall(f, bytes, en.size)) continue;
                if (hex32(crc32(bytes.data(), bytes.size())) != en.crc) continue;
                found.push_back({en.suffix, std::move(bytes)});
                ok = true;
                moved = moved || f != file;
                break;
            }
            if (!ok) { missing = file; break; }
        }
        if (!missing.empty()) {
            reject("no copy of " + missing + " matches it (an interrupted or damaged save)");
            continue;
        }
        uint64_t loop = 0;
        if (!parseDecimal(findPart(found, loopSuffix)->bytes, loop) || loop != iteration) {
            reject(std::string("its .") + loopSuffix + " does not hold iteration " + std::to_string(iteration));
            continue;
        }
        if (cand != manifest)
            std::cerr << "Warning: using the previous saved state " << cand << std::endl;
        else if (moved)
            std::cout << "Saved state " << cand << " completed from its .new/.old files (an interrupted save)" << std::endl;
        parts = std::move(found);
        loopValue = loop;
        return 1;
    }
    std::cerr << "Warning: no complete saved state under " << prefix
              << " — refusing to resume from an inconsistent set of state files" << std::endl;
    return -1;
}


uint64_t BackupManager::loadStatePM1S2Host(std::vector<uint64_t>& hq, std::vector<uint64_t>& q)
{
    uint64_t resume = 0;
    if (marin_) return 0;
    std::vector<Component> parts;
    const int r = loadSet(b2Prefix_, "loop2", parts, resume);
    if (r > 0) {
        if (resume == 0) return 0;
        const Component* h = findPart(parts, "hq");
        const Component* qq = findPart(parts, "q");
        if (!h || !qq || !bytesToWords(h->bytes, hq) || !bytesToWords(qq->bytes, q)) {
            std::cerr << "Warning: the saved stage-2 state has no buffers — starting stage 2 from the beginning\n";
            return 0;
        }
        std::cout << "Stage-2 resume at iteration " << resume << std::endl;
        return resume;
    }
    if (r < 0) {
        std::cerr << "Warning: starting stage 2 from the beginning\n";
        return 0;
    }

    // No manifest: the files of an earlier version, read as that version did.
    std::ifstream loopIn(loop2Filename_);
    if (loopIn >> resume && resume > 0) {
        std::cout << "Stage-2 resume at iteration " << resume << std::endl;
        // Both buffers must be restored from files of the current transform size; otherwise
        // the loop counter is meaningless and stage 2 restarts from its beginning.
        if (!readExactWords(hqFilename_, hq.data(), hq.size(), "stage-2 state") ||
            !readExactWords(qFilename_,  q.data(),  q.size(), "stage-2 state")) {
            std::cerr << "Warning: could not read complete stage-2 buffers — ignoring "
                      << loop2Filename_ << " and starting stage 2 from the beginning\n";
            return 0;
        }
    } else {
        resume = 0;
    }
    return resume;
}

uint64_t BackupManager::loadStatePM1S2(cl_mem hqBuf,
                                       cl_mem qBuf,
                                       size_t bytes)
{
    if (marin_) return 0;
    const size_t words = bytes / sizeof(uint64_t);
    std::vector<uint64_t> hq(words), q(words);
    const uint64_t resume = loadStatePM1S2Host(hq, q);
    if (resume > 0) {
        clEnqueueWriteBuffer(queue_, hqBuf, CL_TRUE, 0, bytes, hq.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue_, qBuf, CL_TRUE, 0, bytes, q.data(), 0, nullptr, nullptr);
        std::cout << "Stage-2 buffers restored" << std::endl;
    }
    return resume;
}

// A Gerbicz-Li buffer: from the set loadState read, from the file of an earlier version, or the
// default when the state was discarded or holds no such buffer.
static void loadGerbiczLiPart(std::vector<uint64_t>& x, uint64_t defaultWord, bool marin, bool discarded,
                              bool setLoaded, bool has, const std::vector<uint64_t>& fromSet,
                              const std::string& file, const char* name) {
    if (marin) return;
    if (!discarded && setLoaded && has && fromSet.size() == x.size()) {
        x = fromSet;
        std::cout << "Loaded " << name << " from the saved state" << std::endl;
        return;
    }
    if (!discarded && !setLoaded) {
        std::vector<uint64_t> tmp(x.size());
        if (readExactWords(file, tmp.data(), tmp.size(), "Gerbicz-Li state")) {
            x = std::move(tmp);
            std::cout << "Loaded " << name << " from " << std::filesystem::absolute(file) << std::endl;
            return;
        }
    }
    x.assign(x.size(), 0ULL);
    if (!x.empty()) x[0] = defaultWord;
    std::cout << "No " << name << " file found at " << std::filesystem::absolute(file) << std::endl;
}

void BackupManager::loadGerbiczLiBufDState(std::vector<uint64_t>& x) {
    loadGerbiczLiPart(x, 1ULL, marin_, stateDiscarded_, setLoaded_, glHasBufd_, glBufd_,
                      GerbiczLiBufDFilename_, "GerbiczLiBufD");
}

void BackupManager::loadGerbiczLiCorrectState(std::vector<uint64_t>& x) {
    loadGerbiczLiPart(x, 3ULL, marin_, stateDiscarded_, setLoaded_, glHasCorrect_, glCorrect_,
                      GerbiczLiCorrectBufFilename_, "GerbiczLiCorrectBuf");
}

void BackupManager::loadGerbiczLiCorrectBufDState(std::vector<uint64_t>& x) {
    loadGerbiczLiPart(x, 1ULL, marin_, stateDiscarded_, setLoaded_, glHasLastBufd_, glLastBufd_,
                      GerbiczLiLastBufDFilename_, "GerbiczLiLastBufD");
}

uint64_t core::BackupManager::loadGerbiczIterSave() {
    uint64_t v = 0;
    if (marin_ || stateDiscarded_) return 0;
    if (setLoaded_) return glIterSave_;
    std::ifstream in(GerbiczLiIterSaveFilename_);
    if (in) {
        in >> v;
        std::cout << "Loaded GerbiczLiIterSave: " << v << " from " << std::filesystem::absolute(GerbiczLiIterSaveFilename_) << std::endl;
    } else {
        std::cout << "No GerbiczLiIterSave file found at " << std::filesystem::absolute(GerbiczLiIterSaveFilename_) << std::endl;
    }
    return v;
}

uint64_t core::BackupManager::loadGerbiczJSave() {
    uint64_t v = 0;
    if (marin_ || stateDiscarded_) return 0;
    if (setLoaded_) return glJSave_;
    std::ifstream in(GerbiczLiJSaveFilename_);
    if (in) {
        in >> v;
        std::cout << "Loaded GerbiczLiJSave: " << v << " from " << std::filesystem::absolute(GerbiczLiJSaveFilename_) << std::endl;
    } else {
        std::cout << "No GerbiczLiJSave file found at " << std::filesystem::absolute(GerbiczLiJSaveFilename_) << std::endl;
    }
    return v;
}

bool BackupManager::saveStatePM1S2Host(const std::vector<uint64_t>& hq, const std::vector<uint64_t>& q,
                                       uint64_t idx)
{
    if (marin_) return false;
    const std::vector<Component> parts = {
        {"hq", wordsToBytes(hq)},
        {"q", wordsToBytes(q)},
        {"loop2", std::to_string(idx + 1)},
    };
    if (!saveSet(b2Prefix_, parts, idx + 1)) return false;
    std::cout << "Stage-2 backup saved at iteration " << idx + 1 << std::endl;
    return true;
}

void BackupManager::saveStatePM1S2(cl_mem hqBuf,
                                   cl_mem qBuf,
                                   uint64_t idx,
                                   size_t bytes)
{
    if(!marin_){
        std::vector<uint64_t> hq(bytes / sizeof(uint64_t)), q(bytes / sizeof(uint64_t));
        clEnqueueReadBuffer(queue_, hqBuf, CL_TRUE, 0, bytes, hq.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue_, qBuf, CL_TRUE, 0, bytes, q.data(), 0, nullptr, nullptr);
        saveStatePM1S2Host(hq, q, idx);
    }
}


uint64_t BackupManager::loadState(std::vector<uint64_t>& x) {
    uint64_t resume = 0;
    setLoaded_ = false;
    glHasBufd_ = glHasCorrect_ = glHasLastBufd_ = false;
    glIterSave_ = glJSave_ = 0;
    auto fresh = [&]() {
        x.assign(x.size(), 0ULL);
        if (!x.empty()) x[0] = (mode_ == "prp") ? 3ULL : 4ULL;
    };

    // 1) Debug : afficher le chemin absolu du .loop
    auto absLoop = std::filesystem::absolute(loopFilename_);
    std::cout << "Looking for loop file at " << absLoop << std::endl;

    // 2) A set saved with a manifest: all of its files, or none of them.
    if (!marin_) {
        std::vector<Component> parts;
        const int r = loadSet(basePrefix_, "loop", parts, resume);
        if (r < 0) {
            std::cerr << "Warning: starting from iteration 0" << std::endl;
            stateDiscarded_ = true;
            fresh();
            return 0;
        }
        if (r > 0) {
            setLoaded_ = true;
            if (resume == 0) {
                std::cout << "No valid loop file, initializing fresh state\n";
                fresh();
                return 0;
            }
            const Component* mers = findPart(parts, "mers");
            if (!mers || !bytesToWords(mers->bytes, x)) {
                std::cerr << "Warning: the saved state has no residue — starting from iteration 0\n";
                stateDiscarded_ = true;
                fresh();
                return 0;
            }
            auto take = [&](const char* suffix, bool& has, std::vector<uint64_t>& v) {
                const Component* c = findPart(parts, suffix);
                v.assign(x.size(), 0ULL);
                has = c && bytesToWords(c->bytes, v);
            };
            take("bufd", glHasBufd_, glBufd_);
            take("gli", glHasCorrect_, glCorrect_);
            take("lbufd", glHasLastBufd_, glLastBufd_);
            if (const Component* c = findPart(parts, "isav")) parseDecimal(c->bytes, glIterSave_);
            if (const Component* c = findPart(parts, "jsav")) parseDecimal(c->bytes, glJSave_);
            std::cout << "Resuming from iteration " << resume
                      << " based on " << absLoop << std::endl;
            std::cout << "Loaded state from "
                      << std::filesystem::absolute(mersFilename_)
                      << std::endl;
            return resume;
        }
    }

    // 3) No manifest: the files of an earlier version, read as that version did.
    std::ifstream loopIn(loopFilename_);
    if (loopIn >> resume && resume > 0) {
        std::cout << "Resuming from iteration " << resume
                  << " based on " << absLoop << std::endl;

        // Charger le vecteur binaire. Without a complete state file of exactly the current
        // transform size the loop counter is meaningless: start over rather than resume from a
        // zero, truncated or differently sized state.
        const bool loaded = readExactWords(mersFilename_, x.data(), x.size(), "state file");
        if (loaded) {
            std::cout << "Loaded state from "
                      << std::filesystem::absolute(mersFilename_)
                      << std::endl;
        } else {
            std::cerr << "Warning: could not read a complete state from "
                      << std::filesystem::absolute(mersFilename_)
                      << " — ignoring the loop file and starting from iteration 0\n";
            resume = 0;
            stateDiscarded_ = true;
            fresh();
        }
    }
    else {
        // 4) Pas de fichier valide → nouvelle session
        std::cout << "No valid loop file, initializing fresh state\n";
        resume = 0;
        fresh();
    }

    return resume;
}


bool BackupManager::saveStateHost(const std::vector<uint64_t>& x, uint64_t iter, const GerbiczLiHost* gl) {
    if (marin_) return false;
    std::vector<Component> parts;
    parts.push_back({"mers", wordsToBytes(x)});
    if (gl) {
        parts.push_back({"bufd", wordsToBytes(gl->bufd)});
        parts.push_back({"lbufd", wordsToBytes(gl->lastBufd)});
        parts.push_back({"gli", wordsToBytes(gl->correct)});
        parts.push_back({"isav", std::to_string(gl->itersave)});
        parts.push_back({"jsav", std::to_string(gl->jsave)});
    }
    parts.push_back({"loop", std::to_string(iter + 1)});
    if (!saveSet(basePrefix_, parts, iter + 1)) {
        std::cerr << "Error saving state to " << mersFilename_ << std::endl;
        return false;
    }
    std::cout << "\nState saved to " << mersFilename_ << std::endl;
    std::cout << "Loop iteration saved to " << loopFilename_ << std::endl;
    return true;
}

void BackupManager::saveState(cl_mem buffer, uint64_t iter, const mpz_class* E_ptr, const GerbiczLiBuffers* gl) {
    if (marin_) return;
    const size_t bytes = vectorSize_ * sizeof(uint64_t);
    std::vector<uint64_t> x(vectorSize_);
    clEnqueueReadBuffer(queue_, buffer, CL_TRUE, 0, bytes, x.data(), 0, nullptr, nullptr);
    GerbiczLiHost host;
    if (gl) {
        host.bufd.resize(vectorSize_);
        host.lastBufd.resize(vectorSize_);
        host.correct.resize(vectorSize_);
        clEnqueueReadBuffer(queue_, gl->bufd, CL_TRUE, 0, bytes, host.bufd.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue_, gl->lastBufd, CL_TRUE, 0, bytes, host.lastBufd.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue_, gl->correct, CL_TRUE, 0, bytes, host.correct.data(), 0, nullptr, nullptr);
        host.itersave = gl->itersave;
        host.jsave = gl->jsave;
    }
    saveStateHost(x, iter, gl ? &host : nullptr);

   if (mode_ == "pm1" && E_ptr != nullptr) {
        std::atomic<bool> done{false};
        std::thread spinner([&]{
            const char seq[] = {'|','/','-','\\'};
            int i = 0;
            while (!done) {
                std::cout << "\rSaving exponent to " << exponentFilename_ << " "
                        << seq[i++ % 4] << std::flush;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        // E depends only on p and B1 (both in the file name), so it is not part of a set; it is
        // written whole or not at all.
        std::ostringstream e;
        e << *E_ptr;
        const std::string tmp = exponentFilename_ + ".new";
        bool ok = writeDurable(tmp, e.str());
        if (ok) {
            std::error_code ec;
            std::filesystem::rename(tmp, exponentFilename_, ec);
            ok = !ec;
        }
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
        }
        done = true;
        spinner.join();
        if (ok) std::cout << "\rSaved exponent to " << exponentFilename_ << "    \n";
        else std::cerr << "Error saving exponent to " << exponentFilename_ << std::endl;
    }

}


mpz_class BackupManager::loadExponent() const {
    mpz_class result{0};
    std::ifstream expIn(exponentFilename_);
    if (expIn) {
        expIn >> result;
        std::cout << "Loaded exponent value from " << exponentFilename_ << std::endl;
    } else {
        std::cout << "No exponent file found at " << exponentFilename_
                  << " — defaulting to 0" << std::endl;
    }
    return result;
}


void BackupManager::clearState() const {
    if(!marin_){
        std::error_code ec;
        auto rm1 = [&](const std::string& f) {
            if (std::filesystem::exists(f, ec)) {
                std::filesystem::remove(f, ec);
                std::cout << "Removed file: " << f << std::endl;
            }
        };
        // with the ".new" and ".old" copies a save keeps
        auto rm = [&](const std::string& f) {
            if (f.empty()) return;
            rm1(f);
            rm1(f + ".new");
            rm1(f + ".old");
        };
        // The manifests first: what is left after a kill here is read as before (an earlier
        // version's files), and the .loop goes next.
        if (!basePrefix_.empty()) { rm(basePrefix_ + ".state"); rm1(basePrefix_ + ".state.old.new"); }
        if (!b2Prefix_.empty()) { rm(b2Prefix_ + ".state"); rm1(b2Prefix_ + ".state.old.new"); }
        rm(loopFilename_);
        rm(loop2Filename_);
        rm(mersFilename_);
        rm(exponentFilename_);
        rm(hqFilename_);
        rm(qFilename_);
        rm(GerbiczLiBufDFilename_);
        rm(GerbiczLiCorrectBufFilename_);
        rm(GerbiczLiIterSaveFilename_);
        rm(GerbiczLiJSaveFilename_);
        rm(GerbiczLiLastBufDFilename_);
    }
}


} // namespace core
