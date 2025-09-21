#!/usr/bin/env bash
set -euo pipefail

PRMERS="./prmers"
PATTERN='\[Gerbicz Li] Check passed'
LOGDIR="./logs_gerbicz"
mkdir -p "$LOGDIR"

EXTRA_ARGS=("$@")

EXPS=(
  127
  1279
  2203
  9941
  44497
  756839
  3021377
  37156667
  57885161
  77232917
  82589933
  136279841
  146410013
  161051017
  177156127
  180000017
  200000033
  220000013
  250000013
  280000027
  300000007
  320000077
  340000019
  360000019
  400000009
  500000003
  600000001
)

cleanup_children() {
  local pids=("$@")
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap 'cleanup_children $(jobs -pr) || true' INT TERM EXIT

run_one() {
  local exp="$1"
  local log="$LOGDIR/run_${exp}.log"
  : > "$log"

  echo "==> start exp=$exp | log=$log"
  if command -v stdbuf >/dev/null 2>&1; then
    (stdbuf -oL -eL "$PRMERS" "$exp" -checklevel 1 "${EXTRA_ARGS[@]}" 2>&1) >"$log" &
  else
    ("$PRMERS" "$exp" -checklevel 1 "${EXTRA_ARGS[@]}" 2>&1) >"$log" &
  fi
  local app_pid=$!

  ( tail -n +1 -F "$log" | tee /dev/stderr | grep -m1 -q "$PATTERN" ) &
  local watch_pid=$!

  local status=0
  if wait "$watch_pid"; then
    echo "==> detected Gerbicz pass for exp=$exp, stopping process..."
    kill -TERM "$app_pid" 2>/dev/null || true
    ( sleep 2; kill -KILL "$app_pid" 2>/dev/null || true ) &
    wait "$app_pid" || status=$?
  else
    wait "$app_pid" || status=$?
  fi

  echo "==> done exp=$exp (status=$status)"
  return 0
}

for exp in "${EXPS[@]}"; do
  run_one "$exp"
done

echo "==> all runs finished"
