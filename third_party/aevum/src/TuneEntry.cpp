#include "TuneEntry.h"
#include "Args.h"
#include "CycleFile.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <string>

// v100.10 tune-format compatibility.
//
// There are two materially different unprefixed tune formats in the wild:
//
//   old GPUOwl/PRPLL FP64:  W:M:H:<single digit 0..3>[:carry]
//   current WMH NTT:        W:M:H:<three digits WMH>[:carry]
//
// The old table bundled with Aevum is the first format.  It is NOT safe to
// relabel those timings as FFT3161.  In June 2025 upstream changed the variant
// encoding from 0..3 to three-digit WMH (000/101/202/...), which gives us an
// unambiguous compatibility boundary.
//
// Explicit current Aevum records always use "1:" and remain authoritative.

namespace {

bool validCurrentVariant(const std::string& text) {
  if (text.size() != 3 ||
      !std::all_of(text.begin(), text.end(),
                   [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return false;
  }
  const unsigned w = unsigned(text[0] - '0');
  const unsigned m = unsigned(text[1] - '0');
  const unsigned h = unsigned(text[2] - '0');
  return w < N_VARIANT_W && m < N_VARIANT_M && h < N_VARIANT_H;
}

bool legacySingleDigitVariant(const std::string& text) {
  return text.size() == 1 && text[0] >= '0' && text[0] <= '3';
}

bool validCarry(const std::string& text) {
  return text == "0" || text == "1";
}

std::string normalizeNativeAevumTuneSpec(const std::string& input,
                                         bool& legacy_fp64,
                                         bool& normalized_unprefixed) {
  legacy_fp64 = false;
  normalized_unprefixed = false;

  auto fields = split(input, ':');
  if (fields.empty()) return {};

  // Explicit current FFT3161:
  // 1:W:M:H:WMH[:carry]
  if (fields[0] == "1") {
    if (fields.size() != 5 && fields.size() != 6) return {};
    if (!validCurrentVariant(fields[4])) return {};
    if (fields.size() == 6 && !validCarry(fields[5])) return {};
    return input;
  }

  // Explicit other families never belong in the stock Type1 tune envelope.
  if (fields[0] == "0" || fields[0] == "2" || fields[0] == "3" ||
      fields[0] == "4" || fields[0] == "50" || fields[0] == "51" ||
      fields[0] == "52" || fields[0] == "53") {
    return {};
  }

  // Unprefixed current WMH NTT:
  // W:M:H:WMH[:carry]
  if (fields.size() == 4 || fields.size() == 5) {
    if (validCurrentVariant(fields[3])) {
      if (fields.size() == 5 && !validCarry(fields[4])) return {};
      normalized_unprefixed = true;
      return "1:" + input;
    }

    // Historical FP64 table:
    // W:M:H:<0..3>[:carry]
    if (legacySingleDigitVariant(fields[3]) &&
        (fields.size() == 4 || validCarry(fields[4]))) {
      legacy_fp64 = true;
      return {};
    }
  }

  return {};
}

} // namespace

// Returns whether *results* was updated.
bool TuneEntry::update(vector<TuneEntry>& results) const {
  u64 maxExp = fft.maxExp();
  [[maybe_unused]] bool didErase = false;

  int i{};
  for (i = int(results.size()) - 1; i >= 0 && results[i].cost > cost; --i) {
    if (results[i].fft.maxExp() <= maxExp) {
      results.erase(std::next(results.begin(), i));
      didErase = true;
    }
  }

  if (i >= 0 && results[i].fft.maxExp() >= maxExp) {
    assert(!didErase);
    return false;
  }

  results.insert(std::next(results.begin(), i + 1), *this);
  return true;
}

// Returns whether entry *e* represents an improvement over *results*.
bool TuneEntry::willUpdate(const vector<TuneEntry>& results) const {
  u64 maxExp = fft.maxExp();
  for (const auto& r : results) {
    if (r.cost > cost) {
      break;
    } else if (r.fft.maxExp() >= maxExp) {
      return false;
    }
  }
  return true;
}

vector<TuneEntry> TuneEntry::readTuneFile(const Args& args) {
  // masterDir is the explicit engine/tune root supplied by PrMers or
  // AEVUM_TUNE_DIR. Prefer it over an accidental tune.txt in cwd.
  fs::path tuneFile = args.masterDir / "tune.txt";
  if (!fs::exists(tuneFile)) tuneFile = "tune.txt";

  vector<TuneEntry> results;
  File fi = File::openRead(tuneFile);
  if (!fi) return {};

  u64 prevMaxExp{};
  double prevCost{};
  unsigned legacyFp64Count = 0;
  unsigned incompatibleCount = 0;
  unsigned normalizedCount = 0;

  for (const string& line : fi) {
    char specBuf[64]{};
    double cost{};
    if (sscanf(line.c_str(), "%lf %63s", &cost, specBuf) < 2) {
      if (args.verbose) log("tune.txt line '%s' ignored: malformed record\n", line.c_str());
      continue;
    }

    bool legacy_fp64 = false;
    bool normalized_unprefixed = false;
    const std::string normalized =
        normalizeNativeAevumTuneSpec(specBuf, legacy_fp64, normalized_unprefixed);

    if (legacy_fp64) {
      ++legacyFp64Count;
      continue;
    }
    if (normalized.empty()) {
      ++incompatibleCount;
      continue;
    }

    try {
      FFTConfig fft{normalized};
      const u64 maxExp = fft.maxExp();

      // A stale/mixed file is input data, not a reason to abort the engine.
      if ((!results.empty() && cost < prevCost) ||
          (!results.empty() && maxExp <= prevMaxExp)) {
        ++incompatibleCount;
        if (args.verbose) log("Ignoring out-of-order Aevum tune entry '%s'\n", specBuf);
        continue;
      }

      prevCost = cost;
      prevMaxExp = maxExp;
      results.push_back({cost, fft});
      if (normalized_unprefixed) ++normalizedCount;
    } catch (...) {
      ++incompatibleCount;
      if (args.verbose) log("Ignoring invalid Aevum tune entry '%s'\n", specBuf);
    }
  }

  if (args.verbose && legacyFp64Count) {
    log("Aevum tune compatibility: ignored %u legacy GPUOwl/PRPLL FP64 tune entries "
        "(single-digit variant 0..3; not FFT3161 timings).\n",
        legacyFp64Count);
  }
  if (args.verbose && normalizedCount) {
    log("Aevum tune compatibility: normalized %u unprefixed current WMH NTT entries "
        "to explicit FFT3161 type 1.\n",
        normalizedCount);
  }
  if (args.verbose && incompatibleCount) {
    log("Aevum tune compatibility: ignored %u incompatible/invalid tune entries.\n",
        incompatibleCount);
  }
  if (args.verbose && !results.empty()) {
    log("Read %u native Aevum FFT3161 tune entries from %s\n",
        u32(results.size()), tuneFile.string().c_str());
  }
  return results;
}

void TuneEntry::writeTuneFile(const vector<TuneEntry>& results) {
  [[maybe_unused]] u64 prevMaxExp{};
  [[maybe_unused]] double prevCost{};
  CycleFile tune{"tune.txt"};
  for (const TuneEntry& r : results) {
    u64 maxExp = r.fft.maxExp();
    assert(r.cost >= prevCost && maxExp > prevMaxExp);
    prevCost = r.cost;
    prevMaxExp = maxExp;
    tune->printf("%6.1f %14s # %" PRIu64 "\n",
                 r.cost, r.fft.spec().c_str(), maxExp);
  }
}
