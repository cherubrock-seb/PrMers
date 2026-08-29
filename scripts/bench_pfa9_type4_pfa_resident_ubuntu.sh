#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-1}"
SECONDS="${2:-75}"
EXPONENT="${3:-175000039}"
ROUNDS="${4:-1}"
cd "$ROOT"

make -j"$(nproc)" KERNEL_PATH=./kernels/
echo '===== EXACT GPU VALIDATION ====='
./third_party/aevum/scripts/test_pfa9_type4_pfa_resident_ubuntu.sh "$DEVICE" "$EXPONENT"

OUT="$ROOT/bench-pfa9-resident-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
run_one() {
  local mode="$1" round="$2"; shift 2
  local dir="$OUT/$mode/r$round"; mkdir -p "$dir"
  echo; echo "===== $mode round $round ====="
  set +e
  timeout -k 8s -s INT "${SECONDS}s" "$@" -f "$dir" 2>&1 | tee "$dir/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 && $rc -ne 137 ]]; then
    echo "FAILED mode=$mode round=$round rc=$rc" >&2; exit "$rc"
  fi
}

for r in $(seq 1 "$ROUNDS"); do
  run_one pow2 "$r" env \
    AEVUM_REG_LEAD_CACHE=1 AEVUM_PFA_RESIDENT=0 AEVUM_PFA_LEAD_BRIDGE=0 AEVUM_TYPE4_MULTI_Q=1 \
    ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft 4:512:8:512:202 -proof 0

  run_one pfa9full-canonical "$r" env \
    AEVUM_REG_LEAD_CACHE=0 AEVUM_PFA_RESIDENT=0 AEVUM_PFA_LEAD_BRIDGE=0 AEVUM_TYPE4_MULTI_Q=1 \
    ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9full:4:512:9:512:202 -proof 0

  run_one pfa9full-resident "$r" env \
    AEVUM_REG_LEAD_CACHE=1 AEVUM_PFA_RESIDENT=1 AEVUM_PFA_LEAD_BRIDGE=0 AEVUM_TYPE4_MULTI_Q=1 \
    ./prmers "$EXPONENT" -d "$DEVICE" -aevum-fft pfa9full:4:512:9:512:202 -proof 0
done

python3 - "$OUT" <<'PY'
from pathlib import Path
import re, statistics, sys
root=Path(sys.argv[1]); prime95=1_000_000/963.9
modes=['pow2','pfa9full-canonical','pfa9full-resident']; vals={}
for mode in modes:
    samples=[]
    for log in sorted((root/mode).glob('r*/run.log')):
        x=[float(v) for v in re.findall(r'IPS:\s*([0-9]+(?:\.[0-9]+)?)',log.read_text(errors='ignore'))]
        x=[v for v in x if v>0]
        # Drop the first non-zero sample because kernel JIT/startup contaminates it.
        if len(x)>1: x=x[1:]
        samples += x
    vals[mode]=statistics.median(samples) if samples else float('nan')
pow2=vals['pow2']
print('\n================ PERFORMANCE SUMMARY ================')
print(f'Prime95 July reference : {prime95:.2f} IPS (963.9 us/iter)')
for m in modes:
    v=vals[m]
    if v!=v: print(f'{m:22s}: no IPS sample'); continue
    print(f'{m:22s}: {v:9.2f} IPS | vs Prime95 {(v/prime95-1)*100:+6.2f}% | vs pow2 {(v/pow2-1)*100:+6.2f}%')
if all(vals[m]==vals[m] for m in modes):
    r=vals['pfa9full-resident']; c=vals['pfa9full-canonical']
    print(f'resident gain vs same PFA canonical: {(r/c-1)*100:+.2f}%')
    if r > pow2:
        print('RESULT: PFA9 RESIDENT BEATS CURRENT POW2 AEVUM on this run.')
    elif r > c:
        print('RESULT: resident layout is a real PFA speedup, but does not yet beat pow2.')
    else:
        print('RESULT: resident layout is correct, but does not improve PFA throughput on this run.')
print('Logs:',root)
PY

echo; echo '===== KEY MARKERS ====='
grep -RHE 'FFT:|resident-word|PFA-RESIDENT|concurrent GF61|Progress:' "$OUT"/*/r*/run.log | tail -90 || true
