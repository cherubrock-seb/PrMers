from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

run = (ROOT / "src/modes/RunPrpOrLl.cpp").read_text()
cli = (ROOT / "include/io/CliParser.hpp").read_text()

# options.proof is a bool: casting the proof power to its type collapses
# every power to 0 or 1 and PrimeNet is told "power":1.
assert "bool proof = true;" in cli
assert "static_cast<decltype(options.proof)>" not in run
assert "options.proofPower = static_cast<decltype(options.proofPower)>(proofPower);" in run

print("Legacy proof power report source regression: PASS")
