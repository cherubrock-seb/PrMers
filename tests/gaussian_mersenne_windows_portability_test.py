#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]

gm = (root / "src/modes/RunGaussianMersenne.cpp").read_text()
factor = (
    root / "src/modes/RunGaussianMersenneFactor.cpp"
).read_text()
cli = (root / "src/io/CliParser.cpp").read_text()

for source in (gm, factor):
    assert "#if defined(_MSC_VER) && defined(_M_X64)" in source
    assert "#elif defined(__SIZEOF_INT128__)" in source
    assert "#include <intrin.h>" in source
    assert "_umul128" in source
    assert "_udiv128" in source
    assert "add_mod_u64" in source
    assert "const unsigned __int128 step128" not in source
    assert "const unsigned __int128 q128" not in source

assert "static bool parse_cli_tail_option" in cli
assert "parse_cli_tail_option(opts, i, argc, argv)" in cli

print("Gaussian-Mersenne Windows/MSVC portability audit passed")
