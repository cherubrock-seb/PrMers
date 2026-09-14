#!/usr/bin/env bash
# PrMers v100.13 RC1 real-GPU gate.
# Usage:
#   DEVICE=2 TARGET=rtx3080 bash scripts/bench_v10013_factoring.sh
#   DEVICE=0 TARGET=radeonVII bash scripts/bench_v10013_factoring.sh
#
# The primary gate compares the exact same GM workload with:
#   OLD = v100.12 product-exponent Stage 2
#   NEW = v100.13 V-trace-lift Stage 2
# in alternating order after a warmup.  Optional ordinary Mersenne P-1 and ECM
# Aevum-vs-Marin probes are enabled with BENCH_MERSENNE=1 / BENCH_ECM=1.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-0}"
TARGET="${TARGET:-gpu}"
P="${P:-21000041}"
B1="${B1:-20000}"
B2="${B2:-40000}"
OUT="${OUT:-$ROOT/v10013-bench-${TARGET}-d${DEVICE}-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUT"

export AEVUM_ENGINE_LIB="${AEVUM_ENGINE_LIB:-$ROOT/third_party/aevum/build-engine/libaevum_engine.so}"

echo "=== BUILD ==="
if ! make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" KERNEL_PATH=./kernels/ >"$OUT/build.log" 2>&1; then
  echo "BUILD FAIL: $OUT/build.log"
  exit 2
fi
echo "BUILD OK | commit=$(git rev-parse --short HEAD) | device=$DEVICE | target=$TARGET"

now_ns() {
  python3 - <<'PY'
import time
print(time.time_ns())
PY
}

elapsed_s() {
  python3 - "$1" "$2" <<'PY'
import sys
a,b=map(int,sys.argv[1:])
print(f"{(b-a)/1e9:.6f}")
PY
}

run_gm() {
  local mode="$1" rep="$2"
  local d="$OUT/${mode}-${rep}"
  mkdir -p "$d"
  local log="$d/run.log"
  local t0 t1 rc wall
  t0="$(now_ns)"
  if [[ "$mode" == "OLD" ]]; then
    PRMERS_GM_PM1_PRODUCT_STAGE2=1 \
      ./prmers "$P" -gm-pm1 -b1 "$B1" -b2 "$B2" -gm-base 3 -gm-sieve 0 \
      -aevum -d "$DEVICE" -f "$d" >"$log" 2>&1
    rc=$?
  else
    PRMERS_GM_PM1_PRODUCT_STAGE2=0 \
      ./prmers "$P" -gm-pm1 -b1 "$B1" -b2 "$B2" -gm-base 3 -gm-sieve 0 \
      -aevum -d "$DEVICE" -f "$d" >"$log" 2>&1
    rc=$?
  fi

  if [[ "$mode" == "NEW" ]]; then
    if ! grep -q 'GM P-1 V-trace Stage 2' "$log"; then
      echo "NEW rep=$rep DID NOT EXECUTE V-TRACE"
      grep -E 'v100.13 fast V-trace bypassed|legacy checkpoint|falling back|Falling back|product-exponent|GM-PM1-VTRACE' "$log" | tail -20 || true
      return 9
    fi
  else
    if ! grep -q 'Stage 2 product-exponent' "$log"; then
      echo "OLD rep=$rep DID NOT EXECUTE PRODUCT-EXPONENT"
      return 9
    fi
  fi
  t1="$(now_ns)"
  wall="$(elapsed_s "$t0" "$t1")"
  if [[ "$rc" -gt 1 ]]; then
    echo "$mode rep=$rep FAIL rc=$rc wall=${wall}s"
    tail -40 "$log"
    return "$rc"
  fi
  printf '%s,%s,%s,%s\n' "$mode" "$rep" "$wall" "$rc" >>"$OUT/gm-times.csv"
  local marker
  marker="$(grep -E 'GM P-1 V-trace Stage 2 complete|GM P-1 Stage 2: 100\.00%|Stage 2 product-exponent|backend=' "$log" | tail -3 | tr '\n' ' ' || true)"
  echo "$mode rep=$rep rc=$rc wall=${wall}s | $marker"
}

echo
echo "=== WARMUP p=$P B1=$B1 B2=$B2 ==="
# Short warmups compile/tune the exact transform and both execution paths.
for mode in NEW OLD; do
  d="$OUT/warm-$mode"; mkdir -p "$d"
  if [[ "$mode" == OLD ]]; then
    PRMERS_GM_PM1_PRODUCT_STAGE2=1 ./prmers "$P" -gm-pm1 -b1 1000 -b2 2000 \
      -gm-base 3 -gm-sieve 0 -aevum -d "$DEVICE" -f "$d" >"$d/run.log" 2>&1 || true
  else
    PRMERS_GM_PM1_PRODUCT_STAGE2=0 \
      ./prmers "$P" -gm-pm1 -b1 1000 -b2 2000 \
      -gm-base 3 -gm-sieve 0 -aevum -d "$DEVICE" -f "$d" >"$d/run.log" 2>&1 || true
  fi
done

echo
echo "=== GM A/B: OLD NEW NEW OLD ==="
: >"$OUT/gm-times.csv"
echo "mode,rep,wall_s,rc" >"$OUT/gm-times.csv"
run_gm OLD 1 || exit 2
run_gm NEW 1 || exit 2
run_gm NEW 2 || exit 2
run_gm OLD 2 || exit 2

python3 - "$OUT/gm-times.csv" <<'PY'
import csv, statistics, sys
rows=list(csv.DictReader(open(sys.argv[1])))
vals={}
for r in rows:
    vals.setdefault(r["mode"],[]).append(float(r["wall_s"]))
old=statistics.median(vals["OLD"])
new=statistics.median(vals["NEW"])
print("\n========== GM RESULT ==========")
print(f"OLD product-exp median : {old:.3f}s")
print(f"NEW V-trace median     : {new:.3f}s")
print(f"TOTAL speedup          : {old/new:.3f}x")
print(f"wall reduction         : {(1-new/old)*100:.1f}%")
print("================================")
PY

# Ordinary Mersenne V-trace already contains Aevum-aware prepared-cache code.
# This optional gate verifies real-hardware correctness/performance versus Marin.
if [[ "${BENCH_MERSENNE:-0}" == "1" ]]; then
  MP="${MP:-23000009}"
  MB1="${MB1:-5000}"
  MB2="${MB2:-10000}"
  echo
  echo "=== MERSENNE P-1 VTRACE AEVUM vs MARIN p=$MP ==="
  for be in MARIN AEVUM; do
    rm -f "pm1_m_${MP}.ckpt"* "pm1_s2_"*"${MP}"*.ckpt* 2>/dev/null || true
    log="$OUT/mersenne-${be}.log"
    t0="$(now_ns)"
    if [[ "$be" == MARIN ]]; then
      ./prmers "$MP" -pm1 -b1 "$MB1" -b2 "$MB2" -engine-marin -d "$DEVICE" >"$log" 2>&1
    else
      ./prmers "$MP" -pm1 -b1 "$MB1" -b2 "$MB2" -aevum -d "$DEVICE" >"$log" 2>&1
    fi
    rc=$?
    t1="$(now_ns)"; wall="$(elapsed_s "$t0" "$t1")"
    if [[ "$rc" -gt 1 ]]; then echo "$be FAIL rc=$rc"; tail -40 "$log"; exit 2; fi
    echo "$be rc=$rc wall=${wall}s | $(grep -E '\[Backend|V-trace|factor stage|No P-1' "$log" | tail -4 | tr '\n' ' ')"
  done
fi

# ECM is intentionally not force-promoted by this RC.  The existing workload
# auto-policy can select Aevum when its transform advantage is large enough.
# This optional gate measures whether the threshold can be relaxed on this GPU.
if [[ "${BENCH_ECM:-0}" == "1" ]]; then
  EP="${EP:-23000009}"
  EB1="${EB1:-1000}"
  EB2="${EB2:-10000}"
  EK="${EK:-1}"
  echo
  echo "=== MERSENNE ECM AEVUM vs MARIN p=$EP ==="
  for be in MARIN AEVUM AUTO; do
    log="$OUT/ecm-${be}.log"
    t0="$(now_ns)"
    case "$be" in
      MARIN) ./prmers "$EP" -ecm -b1 "$EB1" -b2 "$EB2" -K "$EK" -engine-marin -d "$DEVICE" >"$log" 2>&1 ;;
      AEVUM) ./prmers "$EP" -ecm -b1 "$EB1" -b2 "$EB2" -K "$EK" -aevum -d "$DEVICE" >"$log" 2>&1 ;;
      AUTO)  ./prmers "$EP" -ecm -b1 "$EB1" -b2 "$EB2" -K "$EK" -d "$DEVICE" >"$log" 2>&1 ;;
    esac
    rc=$?
    t1="$(now_ns)"; wall="$(elapsed_s "$t0" "$t1")"
    if [[ "$rc" -gt 1 ]]; then echo "$be FAIL rc=$rc"; tail -40 "$log"; exit 2; fi
    echo "$be rc=$rc wall=${wall}s | $(grep -E '\[Backend|ECM.*Elapsed|factor|No factor' "$log" | tail -4 | tr '\n' ' ')"
  done
fi

echo
echo "RESULT DIR: $OUT"
