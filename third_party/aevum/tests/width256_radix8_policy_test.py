"""POSIX header-policy gate with a BPW-free constructor stub; no engine/GPU test."""
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = (Path(__file__).resolve().parents[1] / "src/FFTConfig.h").read_text()
DRIVER = "#include \"FFTConfig.h\"\n#include <cassert>\n#include <iostream>\n\n// BPW tables and device construction are outside this header-policy gate.\nFFTShape::FFTShape(FFT_TYPES t, u32 w, u32 m, u32 h) :\n  fft_type(t), width(w), middle(m), height(h) {}\n\nint main() {\n  unsetenv(\"AEVUM_RADIX8_WIDTH256\");\n  setenv(\"AEVUM_RADIX1K\", \"4\", 1);\n  FFTShape baseline(FFT3161, 256, 16, 512);\n  assert(baseline.nW() == 4 && baseline.nH() == 8);\n  setenv(\"AEVUM_RADIX8_WIDTH256\", \"0\", 1);\n  FFTShape zero(FFT3161, 256, 16, 512);\n  assert(zero.nW() == 4);\n  setenv(\"AEVUM_RADIX8_WIDTH256\", \"1\", 1);\n  FFTShape enabled(FFT3161, 256, 16, 512);\n  assert(enabled.nW() == 8 && enabled.nH() == 8);\n  FFTShape copy = enabled;\n  FFTShape assigned;\n  assigned = enabled;\n  setenv(\"AEVUM_RADIX8_WIDTH256\", \"0\", 1);\n  FFTShape later(FFT3161, 256, 16, 512);\n  assert(later.nW() == 4 && baseline.nW() == 4);\n  assert(enabled.nW() == 8 && copy.nW() == 8 && assigned.nW() == 8);\n  for (const char* invalid : {\"\", \"2\", \"-1\", \"true\", \"01\", \" 1\", \"1 \", \"x\"}) {\n    setenv(\"AEVUM_RADIX8_WIDTH256\", invalid, 1);\n    bool rejected = false;\n    try { FFTShape bad; } catch (const std::runtime_error&) { rejected = true; }\n    assert(rejected);\n  }\n  setenv(\"AEVUM_RADIX8_WIDTH256\", \"1\", 1);\n  FFTShape height(FFT3161, 512, 8, 256);\n  FFTShape oneK(FFT3161, 1024, 8, 512);\n  assert(height.nW() == 8 && height.nH() == 4);\n  assert(oneK.nW() == 4);\n  std::cout << \"width256 header policy, copy isolation and strict values verified; GPU validation pending\\n\";\n}\n"
COMMON = "#pragma once\n#include <cstdint>\n#include <string>\n#include <vector>\n#include <array>\n#include <algorithm>\nusing u32 = std::uint32_t;\nusing u64 = std::uint64_t;\nusing std::string; using std::vector; using std::array; using std::max_element; using std::to_string;\n"
compiler = shutil.which("g++") or shutil.which("c++")
if not compiler:
    raise SystemExit("C++ compiler required")
with tempfile.TemporaryDirectory(prefix="width256-policy-") as tmp:
    root = Path(tmp)
    (root / "FFTConfig.h").write_text(SOURCE)
    (root / "common.h").write_text(COMMON)
    (root / "probe.cpp").write_text(DRIVER)
    binary = root / "probe"
    subprocess.run([compiler, "-std=c++17", str(root / "probe.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
