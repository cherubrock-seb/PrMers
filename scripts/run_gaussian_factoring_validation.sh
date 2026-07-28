#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-1}"
OUT="${GM_FACTOR_TEST_DIR:-$ROOT/gm-factor-validation-v9989}"
cd "$ROOT"
rm -rf "$OUT"
mkdir -p "$OUT"

run_factor() {
  local name="$1" factor="$2"; shift 2
  local log="$OUT/$name.log"
  echo "=== $name ==="
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "Validation command failed with rc=$rc: $*" >&2
    exit 1
  fi
  grep -Eq "factor(:| ) *$factor([^0-9]|$)" "$log" || {
    echo "Expected factor $factor not found in $log" >&2
    exit 1
  }
}

validate_json() {
  local file="$1" mode="$2" outcome="$3" stage="$4" factor="${5:-}"
  python3 - "$file" "$mode" "$outcome" "$stage" "$factor" <<'PY'
import json, sys
from pathlib import Path
path, mode, outcome, stage, factor = sys.argv[1:]
data = json.loads(Path(path).read_text())
required = {
    "schema_version", "program", "program_version", "family", "mode",
    "outcome", "stage", "exponent", "B1", "B2", "curves", "sigma",
    "factor", "backend", "device", "elapsed_seconds", "timestamp",
}
missing = sorted(required - data.keys())
assert not missing, f"missing JSON fields: {missing}"
assert data["schema_version"] == 1
assert data["program"] == "PrMers"
assert data["program_version"] == "v99.89"
assert data["family"] == "gaussian-mersenne"
assert data["mode"] == mode
assert data["outcome"] == outcome
assert data["stage"] == int(stage)
assert isinstance(data["elapsed_seconds"], (int, float))
assert data["timestamp"].endswith("Z")
if factor:
    assert data["factor"] == factor
print(f"JSON schema OK: {path}")
PY
}

make test-gm

run_factor pm1-stage1 53 \
  ./prmers 13 -gm-pm1 -b1 2 -gm-sieve 0 -aevum -d "$DEVICE" -f "$OUT/pm1-stage1"
validate_json "$OUT/pm1-stage1/gm_pm1_p13_result.json" gm-pm1 factor 1 53

run_factor pm1-stage2 277 \
  ./prmers 23 -gm-pm1 -b1 2 -b2 3 -gm-sieve 0 -aevum -d "$DEVICE" -f "$OUT/pm1-stage2"
grep -q 'GM P-1 Stage 2:' "$OUT/pm1-stage2.log" || {
  echo "Stage 2 progress line missing" >&2
  exit 1
}
validate_json "$OUT/pm1-stage2/gm_pm1_p23_result.json" gm-pm1 factor 2 277

run_factor ecm-stage1 137 \
  ./prmers 17 -gm-ecm -b1 50 -K 1 -sigma 7 -gm-sieve 0 -aevum -d "$DEVICE" -f "$OUT/ecm-stage1"
validate_json "$OUT/ecm-stage1/gm_ecm_p17_result.json" gm-ecm factor 1 137

run_factor ecm-stage2-safe 137 \
  ./prmers 17 -gm-ecm -b1 2 -b2 3 -K 1 -sigma 14 -gm-safe -gm-sieve 0 \
    -aevum -d "$DEVICE" -f "$OUT/ecm-stage2-safe"
validate_json "$OUT/ecm-stage2-safe/gm_ecm_p17_result.json" gm-ecm factor 2 137

# Native conditional worktodo: P-1 finds 53, so ECM/Proth are skipped and the
# line is archived/removed.
WT="$OUT/worktodo-gm.txt"
printf '# validation comment\nGMCHAIN=13,2,0,0,0,0,0,1024\n' > "$WT"
run_factor worktodo-chain 53 \
  ./prmers -worktodo "$WT" -aevum -d "$DEVICE" -r -f "$OUT/worktodo-chain"
if grep -Eq '^[[:space:]]*GMCHAIN=' "$WT"; then
  echo "Completed GMCHAIN line was not removed" >&2
  exit 1
fi
validate_json "$OUT/worktodo-chain/gm_pm1_p13_result.json" gm-pm1 factor 1 53

echo "Gaussian-Mersenne v99.89 validation passed on OpenCL device $DEVICE"
