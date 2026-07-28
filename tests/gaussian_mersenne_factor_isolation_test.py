#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "src/modes/RunGaussianMersenneFactor.cpp").read_text()
parser = (root / "src/io/CliParser.cpp").read_text()
app = (root / "src/core/App.cpp").read_text()
header = (root / "include/core/App.hpp").read_text()
options = (root / "include/io/CliParser.hpp").read_text()
version = (root / "include/core/Version.hpp").read_text()

for token in (
    "runGaussianMersennePM1",
    "runGaussianMersenneECM",
    "G_p | 2^(2p)+1 | 2^(4p)-1",
    "make_suyama_curve",
    "montgomery_ladder",
    "project_reg",
    "gm_factor_chunk_bits",
    "GM P-1 Stage 2:",
    "gm_result_json",
    r'\"schema_version\": 1',
    "h.version >= 2 && h.version <= GMF_VERSION",
):
    assert token in source or token in parser or token in app or token in header or token in options

assert 'opts.mode = "gm-pm1"' in parser
assert 'opts.mode = "gm-ecm"' in parser
assert 'options.mode == "gm-pm1"' in app
assert 'options.mode == "gm-ecm"' in app
assert 'options.mode == "gm-chain"' in app
assert "removeProcessedLine(activeWorktodoRawLine_)" in app
assert "4.20.79-alpha-v99.90-gaussian-macos-u64-fix" in version

# GMP C++ has no unambiguous unsigned-long-long constructor on every ABI.
assert "mpz_class sigma = sigma64;" not in source
assert "mpz_import(" in source
assert "sizeof(sigma64)" in source

# Dedicated opt-in dispatch: ordinary modes remain on their historical functions.
assert 'if(options.mode == "ecm")' in app
assert 'else if (options.mode == "pm1"' in app
assert source.count("engine::create_gpu(t.lift") == 2
assert "2^(4p)-1" in source

# No native sparse-modulus kernel was added: the implementation is host-side
# orchestration over the existing engine API, deliberately isolating Aevum.
assert not (root / "third_party/aevum/src/GaussianMersenne.cpp").exists()
assert not (root / "kernels/gaussian_mersenne.cl").exists()

print("Gaussian-Mersenne factoring dispatch/isolation checks passed")
