#!/usr/bin/env bash

set -euo pipefail

PRMERS_BIN="${PRMERS:-./prmers}"
K=50

if [[ ! -x "$PRMERS_BIN" ]]; then
  echo "Error: prmers binary not found or not executable: '$PRMERS_BIN'"
  echo "Set PRMERS=/path/to/prmers or place the binary in the current directory."
  exit 1
fi

exps=(193 199 1999 2027 2029 20849 20857 20887)
b1s=(100 1000 10000 100000 500000)

combo_names=(
  "edwards_t8"
  "edwards_t16"
  "edwards_notorsion"
  "montgomery_notorsion"
  "montgomery_t8"
  "montgomery_t16"
)
combo_flags=(
  ""
  "-torsion16"
  "-notorsion"
  "-montgomery -notorsion"
  "-montgomery"
  "-montgomery -torsion16"
)

timestamp="$(date +%Y%m%d_%H%M%S)"
outdir="ecm_runs_${timestamp}"
mkdir -p "$outdir"

summary="$outdir/summary.csv"
echo "exponent,B1,combo,exit_code,duration_s,logfile" > "$summary"

echo "=== ECM MATRIX START ==="
echo "Binary: $PRMERS_BIN"
echo "Output folder: $outdir"
echo "Combos:"
for i in "${!combo_names[@]}"; do
  printf "  - %s : %s\n" "${combo_names[$i]}" "${combo_flags[$i]}"
done
echo

DRY_RUN="${DRY_RUN:-0}"

for p in "${exps[@]}"; do
  for b1 in "${b1s[@]}"; do
    for i in "${!combo_names[@]}"; do
      name="${combo_names[$i]}"
      flags="${combo_flags[$i]}"
      logfile="$outdir/p${p}_b1${b1}_${name}.log"
      cmd=( "$PRMERS_BIN" "$p" -ecm -b1 "$b1" -K "$K" )
      if [[ -n "$flags" ]]; then
        extra=( $flags )
        cmd+=( "${extra[@]}" )
      fi

      echo "[$(date '+%F %T')] RUN: ${cmd[*]}"
      echo "Command: ${cmd[*]}" > "$logfile"

      start=$(date +%s)
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "(dry-run) ${cmd[*]}"
        rc=0
        dur=0
      else
        set +e
        "${cmd[@]}" 2>&1 | tee -a "$logfile"
        rc=$?
        set -e
        end=$(date +%s)
        dur=$(( end - start ))
      fi

      echo "${p},${b1},${name},${rc},${dur},${logfile}" >> "$summary"
      echo "=> DONE: p=${p}, B1=${b1}, combo=${name}, rc=${rc}, duration=${dur}s"
      echo
    done
  done
done

echo "=== COMPLETE ==="
echo "Summary CSV: $summary"
