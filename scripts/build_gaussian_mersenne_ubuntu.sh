#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-$(nproc)}"
cd "$ROOT"

echo "[1/4] Source-level Gaussian-Mersenne tests"
make test-gm

echo "[2/4] Building embedded Aevum engine"
make -C third_party/aevum engine-lib -j"$JOBS"

echo "[3/4] Building PrMers"
make -j"$JOBS"

echo "[4/4] Verifying local plugin and CLI"
test -x ./prmers
test -f ./third_party/aevum/build-engine/libaevum_engine.so
./prmers -h 2>&1 | grep -q -- '-gm-proth'

cat <<EOF

Build complete.
Binary : $ROOT/prmers
Aevum  : $ROOT/third_party/aevum/build-engine/libaevum_engine.so

Quick checks (replace device 1 if needed):
  ./prmers 7  -gm -aevum -gm-sieve 0 -d 1
  ./prmers 13 -gm -aevum -gm-sieve 1000000 -d 1
  ./prmers 113 -gm -aevum -gm-safe -gm-sieve 0 -d 1
EOF
