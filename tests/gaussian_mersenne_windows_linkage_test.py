#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cmake = (root / "CMakeLists.txt").read_text()
makefile = (root / "Makefile").read_text()
rename = (root / "include/core/GmEcmLegacyRename.hpp").read_text()
factor = (root / "src/modes/RunGaussianMersenneFactor.cpp").read_text()
fast = (root / "src/modes/RunGaussianMersenneEcmFast.cpp").read_text()
opt = (root / "src/modes/RunGaussianMersenneEcmOptimized.cpp").read_text()
app = (root / "include/core/App.hpp").read_text()

# The v99.97+ architecture deliberately keeps the old implementation in the
# original translation unit but renames its method at compile time. Unix Make
# and CMake/MSVC MUST apply the same forced-include header to that TU.
assert "RunGaussianMersenneFactor.o: CPPFLAGS += -include $(INC_DIR)/core/GmEcmLegacyRename.hpp" in makefile
assert "src/modes/RunGaussianMersenneFactor.cpp" in cmake
assert "GmEcmLegacyRename.hpp" in cmake
assert "if(MSVC)" in cmake
assert "/FI${PROJECT_SOURCE_DIR}/include/core/GmEcmLegacyRename.hpp" in cmake
assert "-include;${PROJECT_SOURCE_DIR}/include/core/GmEcmLegacyRename.hpp" in cmake

# Header order matters: App.hpp must be parsed before the method-token remap.
inc = rename.index('#include "core/App.hpp"')
define = rename.index('#define runGaussianMersenneECM runGaussianMersenneECMLegacy')
assert inc < define

# Linkage contract: one source owns the historical implementation token and
# one source owns the public wrapper. The forced include changes only the old
# translation unit to the Legacy symbol.
assert "int App::runGaussianMersenneECM()" in factor
assert "int App::runGaussianMersenneECM()" in fast
assert "runGaussianMersenneECMLegacy();" in fast
assert "int runGaussianMersenneECMLegacy();" in app
assert "int runGaussianMersenneECMOptimized();" in app

# Audit the new v99.97/v99.98 translation units themselves for the x64 MSVC
# modular-multiply branch. Do not permit an unconditional __int128 path.
for source in (fast, opt):
    assert "#if defined(_MSC_VER) && defined(_M_X64)" in source
    assert "#include <intrin.h>" in source
    assert "_umul128" in source
    assert "_udiv128" in source
    assert "#elif defined(__SIZEOF_INT128__)" in source

print("Gaussian-Mersenne CMake/MSVC legacy-linkage audit passed")
