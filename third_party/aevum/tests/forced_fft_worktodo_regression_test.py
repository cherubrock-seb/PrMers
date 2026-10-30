#!/usr/bin/env python3

import os
import pathlib
import platform
import subprocess
import tempfile
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
CXX = os.environ.get("CXX", "c++")

HARNESS = r'''
#include "Task.h"
#include "Worktodo.h"
#include "Args.h"
#include "FFTConfig.h"
#include "CycleFile.h"
#include "Proof.h"
#include "clwrap.h"
#include "gpuid.h"
#include "version.h"

#include <cstdarg>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifdef _WIN32
static int setenv(const char* name, const char* value, int) {
return _putenv_s(name, value);
}
static int unsetenv(const char* name) {
return _putenv_s(name, "");
}
#endif

void log(const char*, ...) {}

const char *VERSION = "forced-fft-regression-test";

CycleFile::CycleFile(const fs::path& path) : name(path), f{} {}
CycleFile::~CycleFile() = default;
File* CycleFile::operator->() { return nullptr; }
File& CycleFile::operator*() {
  throw std::runtime_error("unused CycleFile stub");
}
void CycleFile::reset() { f.reset(); }

u32 ProofSet::bestPower(u64) { return 0; }
double ProofSet::diskUsageGB(u64, u32) { return 0.0; }

std::vector<cl_device_id> getAllDeviceIDs() { return {}; }
std::string getBdfFromDevice(cl_device_id) { return {}; }
std::string getBoardName(cl_device_id) { return {}; }
std::string getDeviceName(cl_device_id) { return {}; }
std::string getDriverVersion(cl_device_id) { return {}; }
std::string getDriverVersionByPos(int) { return {}; }

std::string getUidFromBdf(const std::string&) { return {}; }
std::string getUidFromPos(int) { return {}; }
int getPosFromBdf(const std::string&) { return -1; }
int getPosFromUid(const std::string&) { return -1; }

std::vector<std::string> split(
    const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delimiter))
    result.push_back(part);
  return result;
}

static void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

static Task selectLocal(
    const fs::path& dir,
    const std::string& fftSpec,
    const std::string& lines) {
  fs::remove_all(dir);
  fs::create_directories(dir);

  const fs::path saved = fs::current_path();
  fs::current_path(dir);

  {
    std::ofstream file("worktodo-0.txt", std::ios::binary);
    file << lines;
  }

  Args args(true);
  args.fftSpec = fftSpec;

  auto task = Worktodo::getTask(args, 0);

  fs::current_path(saved);

  if (!task)
    throw std::runtime_error(
        "no runnable worktodo task selected");

  return *task;
}

int main() {
  unsetenv("AEVUM_ALLOW_OVERCAPACITY_PLAN");

  Args args(true);

  const std::string large = "1:512:16:512:202";

  FFTConfig valid =
      FFTConfig::bestFit(args, 100000007u, large);

  require(
      valid.spec() == large,
      "valid forced FFT changed");

  bool tooLargeRejected = false;

  try {
    (void) FFTConfig::bestFit(
        args, 1000003u, large);
  } catch (const std::runtime_error& e) {
    tooLargeRejected =
        std::string(e.what()).find(
            "too large for exponent") != std::string::npos;
  }

  require(
      tooLargeRejected,
      "too-large forced FFT was not rejected before GPU creation");

#if !defined(__APPLE__)
  for (const std::string adaptivePfa : {
           "pfa9:4:512:9:512:202",
           "pfa9fast:4:512:9:512:202"
       }) {
    bool adaptiveTooLargeRejected = false;

    try {
      (void) FFTConfig::bestFit(
          args, 1000003u, adaptivePfa);
    } catch (const std::runtime_error& e) {
      adaptiveTooLargeRejected =
          std::string(e.what()).find(
              "too large for exponent") != std::string::npos;
    }

    require(
        adaptiveTooLargeRejected,
        "adaptive PFA9 Type4 bypassed the minimum BPW gate");
  }

  FFTConfig validAdaptivePfa =
      FFTConfig::bestFit(
          args, 20000003u,
          "pfa9:4:512:9:512:202");

  require(
      validAdaptivePfa.isPfa() &&
      validAdaptivePfa.shape.fft_type == FFT3161 &&
      validAdaptivePfa.adaptive_type4_elided,
      "valid adaptive PFA9 Type4 behavior changed");
#else
  for (const std::string adaptivePfa : {
           "pfa9:4:512:9:512:202",
           "pfa9fast:4:512:9:512:202"
       }) {
    bool applePfaRejected = false;

    try {
      (void) FFTConfig::bestFit(
          args, 1000003u, adaptivePfa);
    } catch (const std::runtime_error& e) {
      applePfaRejected =
          std::string(e.what()).find(
              "Apple OpenCL 1.2 supports only stock FFT3161") !=
          std::string::npos;
    }

    require(
        applePfaRejected,
        "Apple explicit PFA rejection behavior changed");
  }
#endif

  const std::string small = "1:256:2:256:101";

  bool tooSmallRejected = false;

  try {
    (void) FFTConfig::bestFit(
        args, 500000003u, small);
  } catch (const std::runtime_error& e) {
    tooSmallRejected =
        std::string(e.what()).find(
            "too small for exponent") != std::string::npos;
  }

  require(
      tooSmallRejected,
      "existing explicit maxExp rejection was weakened");

  FFTConfig automatic =
      FFTConfig::bestFit(args, 1000003u, "");

  const double autoBpw =
      1000003.0 / double(automatic.size());

  require(
      autoBpw >= automatic.minBpw(),
      "AUTO selected an FFT too large for the exponent");

  require(
      automatic.maxExp() * args.fftOverdrive >= 1000003u,
      "AUTO selected an FFT too small for the exponent");

  const fs::path root =
      fs::temp_directory_path() /
      "aevum-forced-fft-worktodo-regression";

  const Task tooLarge = selectLocal(
      root / "too-large",
      large,
      "PRP=N/A,1,2,1000003,-1,77,0\n"
      "PRP=N/A,1,2,100000007,-1,77,0\n");

  require(
      tooLarge.kind == Task::PRP &&
      tooLarge.exponent == 100000007u,
      "too-large forced FFT line blocked following PRP");

  const Task tooSmall = selectLocal(
      root / "too-small",
      small,
      "Test=1,2,500000003,-1\n"
      "Test=1,2,1000003,-1\n");

  require(
      tooSmall.kind == Task::LL &&
      tooSmall.exponent == 1000003u,
      "too-small forced FFT line blocked following LL");

  const Task selector = selectLocal(
      root / "pseudo-selector",
      "native-prp:auto",
      "PRP=N/A,1,2,1000003,-1,77,0\n");

  require(
      selector.kind == Task::PRP &&
      selector.exponent == 1000003u,
      "pseudo-selector behavior changed");

  setenv("AEVUM_ALLOW_OVERCAPACITY_PLAN", "1", 1);

  const Task override = selectLocal(
      root / "override",
      small,
      "PRP=N/A,1,2,500000003,-1,77,0\n");

  require(
      override.kind == Task::PRP &&
      override.exponent == 500000003u,
      "AEVUM_ALLOW_OVERCAPACITY_PLAN behavior changed");

  unsetenv("AEVUM_ALLOW_OVERCAPACITY_PLAN");
  fs::remove_all(root);

  std::cout
      << "forced FFT capacity/worktodo regression: OK"
      << std::endl;

  return 0;
}
'''

SOURCES = [
    "src/Worktodo.cpp",
    "src/PrpUseTune.cpp",
    "src/RuntimeAutotune.cpp",
    "src/FFTConfig.cpp",
    "src/Args.cpp",
    "src/TuneEntry.cpp",
    "src/common.cpp",
    "src/fs.cpp",
    "src/File.cpp",
    "src/timeutil.cpp",
]

with tempfile.TemporaryDirectory(
    prefix="aevum-forced-fft-regression-"
) as tmp:
    tmp_path = pathlib.Path(tmp)
    source = tmp_path / "forced_fft_regression.cpp"
    binary = tmp_path / (
        "forced_fft_regression.exe"
        if os.name == "nt"
        else "forced_fft_regression"
    )

    source.write_text(
        textwrap.dedent(HARNESS),
        encoding="utf-8",
    )

    command = [
        CXX,
        "-O2",
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-ffunction-sections",
        "-fdata-sections",
        "-Isrc",
        str(source),
        *SOURCES,
    ]

    if platform.system() == "Darwin":
        command.append("-Wl,-dead_strip")
    else:
        command.append("-Wl,--gc-sections")

    command += ["-o", str(binary)]

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )

    subprocess.run(
        [str(binary)],
        cwd=ROOT,
        check=True,
    )