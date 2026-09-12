#pragma once
#include "FFTConfig.h"
#include "RuntimeAutotune.h"
#include <map>
#include <string>
#include <utility>
#include <vector>
namespace aevum_prp_use {
enum class Decision { Positive, Negative, Deferred };
struct SearchProgress {
  unsigned planned{}, screened{}, finalists{}, finalized{};
  bool interrupted{};
  Decision decision(bool has_winner) const;
};
struct ResumeState {
  unsigned planned{};
  unsigned next_screen{};
  unsigned next_finalist{};
  std::vector<std::pair<std::string,double>> top;
  std::string winner;
  double winning_gain{1.0};
};
bool persistDecision(const std::string& key, Decision decision,
                     const std::string& profile, double gain, uint64_t milliseconds);
std::vector<KeyVal> parse(const std::string& text);
std::string normalize(const std::vector<KeyVal>& values);
std::vector<std::string> candidates(bool nvidia_asm);
aevum_autotune::Mode mode();
std::string key(const std::string& identity,const std::string& shape,uint32_t p,const std::string& flags);
std::filesystem::path path(const std::string& key);
std::filesystem::path progressPath(const std::string& key);
std::optional<aevum_autotune::Record> load(const std::string& key);
void store(const std::string& key,const std::string& profile,double gain,uint64_t milliseconds);
std::optional<ResumeState> loadProgress(const std::string& key);
void storeProgress(const std::string& key,const ResumeState& state);
void clearProgress(const std::string& key);
}
