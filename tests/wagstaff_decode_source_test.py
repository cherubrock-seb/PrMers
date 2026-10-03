from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

run = (ROOT / "src/modes/RunPrpOrLlMarin.cpp").read_text()

# The Wagstaff verdict must be read from the engine that ran the test. The
# legacy Precompute digit widths describe a different transform than the Marin
# or Aevum engine and decode the residue incorrectly whenever they differ.
start = run.index("if (options.wagstaff) {\n            mpz_class Mp")
end = run.index('"Not a Wagstaff PRP', start)
block = run[start:end]

assert "eng->get_mpz(" in block, "Wagstaff result must be read with eng->get_mpz"
assert "vectToMpz" not in block, "Wagstaff result must not use Precompute digit widths"
assert "getDigitWidth" not in block, "Wagstaff result must not use Precompute digit widths"

print("Wagstaff engine-decode source regression: PASS")
