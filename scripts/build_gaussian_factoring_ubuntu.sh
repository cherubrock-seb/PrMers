#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-$(nproc)}"
cd "$ROOT"

echo "[1/6] Gaussian-Mersenne mathematics and isolation tests"
make test-gm

echo "[2/6] Aevum host policy/arithmetic tests"
make -C third_party/aevum test-host

echo "[3/6] Building embedded Aevum engine"
make -C third_party/aevum engine-lib -j"$JOBS"

echo "[4/6] Building PrMers"
make -j"$JOBS"

echo "[5/6] Verifying binary, plugin and new CLI"
test -x ./prmers
test -f ./third_party/aevum/build-engine/libaevum_engine.so
HELP="$(./prmers -h 2>&1 || true)"
grep -q -- '-gm-pm1' <<<"$HELP"
grep -q -- '-gm-ecm' <<<"$HELP"
grep -q -- '-gm-factor-chunk-bits' <<<"$HELP"
grep -q -- 'GMCHAIN=' <<<"$HELP"

echo "[6/6] Verifying campaign tools"
python3 tests/test_gaussian_worktodo_generator.py
bash tests/test_gaussian_worktodo_parser.sh

cat <<TXT

Build complete.
Binary : $ROOT/prmers
Aevum  : $ROOT/third_party/aevum/build-engine/libaevum_engine.so

Run the GPU validation suite:
  ./scripts/run_gaussian_factoring_validation.sh 1
TXT
