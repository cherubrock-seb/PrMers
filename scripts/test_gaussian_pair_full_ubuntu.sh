#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${DEVICE:-0}"
JOBS="${JOBS:-$(nproc)}"
OUT="${GM_PAIR_TEST_DIR:-$ROOT/gm-pair-full-validation-v9995}"
AEVUM_SMOKE_P="${AEVUM_SMOKE_P:-3704053}"

cd "$ROOT"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake pkg-config python3 \
    libgmp-dev libcurl4-openssl-dev \
    ocl-icd-opencl-dev opencl-headers clinfo
fi

command -v c++ >/dev/null
command -v make >/dev/null
command -v python3 >/dev/null
command -v clinfo >/dev/null

clinfo -l
export PYTHONDONTWRITEBYTECODE=1

make clean-all
make -j"$JOBS" KERNEL_PATH=./kernels/
make test-gm
make test-aevum-source
make test-aevum-host
make test-aevum-auto
./prmers -v

rm -rf "$OUT"
mkdir -p "$OUT"

run_result() {
  local name="$1"; shift
  local log="$OUT/$name.log"
  echo "=== $name ==="
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -gt 1 ]]; then
    echo "Command failed with rc=$rc: $*" >&2
    exit 1
  fi
  echo "Result exit code: $rc"
}

# 1. Direct OpenCL TF: one GPU pass classifies both families.
run_result tf \
  ./prmers 19 -gm-tf 8 20 -gm-family BOTH \
    -gm-tf-chunk 65536 -gm-tf-sieve 997 \
    -d "$DEVICE" -f "$OUT/tf"

# 2. PRP and Proth lanes. The GQ side of Proth is honestly a Fermat PRP.
run_result prp \
  ./prmers 13 -gm-prp -gm-family BOTH -gm-sieve 0 \
    -aevum-auto -d "$DEVICE" -f "$OUT/prp"

run_result proth \
  ./prmers 13 -gm-proth -gm-family BOTH -gm-sieve 0 \
    -aevum-auto -d "$DEVICE" -f "$OUT/proth"

# 3. P-1 and ECM through the ordinary workload-aware engine policy.
run_result pm1 \
  ./prmers 13 -gm-pm1 -gm-family BOTH -b1 2 -b2 3 \
    -gm-sieve 0 -aevum-auto -d "$DEVICE" -f "$OUT/pm1"

run_result ecm \
  ./prmers 17 -gm-ecm -gm-family BOTH -b1 50 -K 1 -sigma 7 \
    -gm-sieve 0 -aevum-auto -d "$DEVICE" -f "$OUT/ecm"

# 4. Full conditional chain. GM should stop after P-1 factor 53. GQ should
# continue through ECM to the final Fermat PRP.
cat > "$OUT/worktodo.txt" <<'WT'
GMCHAIN=13,2,3,2,3,1,0,1024,proth,BOTH
WT
run_result chain \
  ./prmers -worktodo "$OUT/worktodo.txt" -aevum-auto -d "$DEVICE" \
    -f "$OUT/chain"

# 5. Real Aevum-capable Gaussian smoke cases.  The small p=13/17 cases
# above validate arithmetic and JSON quickly, but they are intentionally below
# the Aevum planning range.  p=3,704,053 is a known GM prime listed by GMNet
# and PrimePages.  Its exact lift 4p=14,816,212 selects a 524,288-word Aevum
# transform while B1=2 / one ECM curve keep the physical test short.
run_result aevum-auto-pm1 \
  ./prmers "$AEVUM_SMOKE_P" -gm-pm1 -gm-family GM -b1 2 -b2 3 \
    -gm-sieve 0 -aevum-auto -d "$DEVICE" -f "$OUT/aevum-auto-pm1"

run_result aevum-auto-ecm \
  ./prmers "$AEVUM_SMOKE_P" -gm-ecm -gm-family GM -b1 2 -K 1 -sigma 7 \
    -gm-sieve 0 -aevum-auto -d "$DEVICE" -f "$OUT/aevum-auto-ecm"

# 6. Explicit backend overrides remain available on the same real-size case.
run_result forced-aevum \
  ./prmers "$AEVUM_SMOKE_P" -gm-pm1 -gm-family GM -b1 2 -b2 3 \
    -gm-sieve 0 -aevum -d "$DEVICE" -f "$OUT/forced-aevum"
run_result forced-marin \
  ./prmers "$AEVUM_SMOKE_P" -gm-pm1 -gm-family GM -b1 2 -b2 3 \
    -gm-sieve 0 -engine-marin -d "$DEVICE" -f "$OUT/forced-marin"

python3 - "$OUT" "$AEVUM_SMOKE_P" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
smoke_p = int(sys.argv[2])

def load(relative):
    path = root / relative
    assert path.is_file(), path
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2, path
    assert data["family"] == "gaussian-pair", path
    return data

# TF physical result.
tf = load("tf/gm_tf_p19_8_20_BOTH_result.json")
assert tf["target_family"] == "BOTH"
assert tf["backend"] == "OpenCL-GPU-TF"
found = {(item["family"], item["factor"]) for item in tf["factors"]}
assert ("GM", "525313") in found, found
assert ("GQ", "457") in found, found

# Separate family JSON for every non-TF pair mode.
for mode in ("prp", "proth"):
    gm = load(f"{mode}/gm_{mode}_p13_result.json")
    gq = load(f"{mode}/gq_{mode}_p13_result.json")
    assert gm["target_family"] == "GM"
    assert gq["target_family"] == "GQ"
    assert gm["backend"] in ("Aevum", "Marin")
    assert gq["backend"] in ("Aevum", "Marin")

assert load("proth/gm_proth_p13_result.json")["test_method"] == "proth"
gq_proth = load("proth/gq_proth_p13_result.json")
assert gq_proth["test_method"] == "fermat-prp"
assert gq_proth["outcome"] == "probable-prime"

pm1_gm = load("pm1/gm_pm1_p13_result.json")
pm1_gq = load("pm1/gq_pm1_p13_result.json")
assert pm1_gm["factor"] == "53", pm1_gm
assert pm1_gq["outcome"] == "no-factor", pm1_gq

assert load("ecm/gm_ecm_p17_result.json")["factor"] == "137"
assert load("ecm/gq_ecm_p17_result.json")["outcome"] == "no-factor"

assert load("chain/gm_pm1_p13_result.json")["factor"] == "53"
assert load("chain/gq_pm1_p13_result.json")["outcome"] == "no-factor"
assert load("chain/gq_ecm_p13_result.json")["outcome"] == "no-factor"
chain_final = load("chain/gq_proth_p13_result.json")
assert chain_final["outcome"] == "probable-prime"
assert chain_final["test_method"] == "fermat-prp"

# Real-size backend smoke results.  These are no-factor runs on a known GM
# prime, so the important contract is the selected physical backend.
auto_pm1 = load(f"aevum-auto-pm1/gm_pm1_p{smoke_p}_result.json")
auto_ecm = load(f"aevum-auto-ecm/gm_ecm_p{smoke_p}_result.json")
forced_aevum = load(f"forced-aevum/gm_pm1_p{smoke_p}_result.json")
forced_marin = load(f"forced-marin/gm_pm1_p{smoke_p}_result.json")
assert auto_pm1["backend"] == "Aevum", auto_pm1
assert auto_ecm["backend"] == "Aevum", auto_ecm
assert forced_aevum["backend"] == "Aevum", forced_aevum
assert forced_marin["backend"] == "Marin", forced_marin

print("Full Gaussian pair JSON validation passed")
PY

# Workload policy and explicit overrides must be visible in logs.
grep -Eq '\[Backend Auto\] (PRP|P-1|ECM):' "$OUT/prp.log" "$OUT/pm1.log" "$OUT/ecm.log"
grep -F '[Gaussian backend policy] P-1 workload=P-1' "$OUT/chain.log"
grep -F '[Gaussian backend policy] ECM workload=ECM' "$OUT/chain.log"
grep -F '[Gaussian backend policy] GQ Fermat PRP workload=PRP' "$OUT/chain.log"
grep -F '[Backend Auto] P-1: Aevum selected' "$OUT/aevum-auto-pm1.log"
grep -F '[Backend Auto] ECM: Aevum selected' "$OUT/aevum-auto-ecm.log"
grep -F '[Backend Aevum]' "$OUT/forced-aevum.log"
grep -F '[Backend Marin]' "$OUT/forced-marin.log"
if grep -Eq '\[Backend (Aevum|Marin|Auto)\]' "$OUT/tf.log"; then
  echo "TF unexpectedly entered the Aevum/Marin engine policy" >&2
  exit 1
fi

# make clean-all removes tracked dependency files in historical snapshots.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git ls-files -d -z | xargs -0 -r git restore --
fi

echo "PrMers v99.95 full Gaussian pair Ubuntu validation passed on device $DEVICE."
