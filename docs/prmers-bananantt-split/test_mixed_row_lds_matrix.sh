#!/usr/bin/env bash
set -euo pipefail

P=${P:-3021377}
DEVICE=${DEVICE:-1}
ODD=${ODD:-9}
ITERS=${ITERS:-1}
VAL_ITERS=${VAL_ITERS:-8}
PROFILE=${PROFILE:-0}
BIN=${BIN:-./prmers_opencl_prp}
MAX_CASES=${MAX_CASES:-0}
START_CASE=${START_CASE:-1}
STOP_ON_FAIL=${STOP_ON_FAIL:-0}
LOGDIR=${LOGDIR:-mixed_row_lds_matrix_logs}
SUMMARY=${SUMMARY:-$LOGDIR/summary.tsv}
STAGES=${STAGES:-"8 16 32 64 128 256 512 1024"}
CENTERS=${CENTERS:-"8 16 32 64 128 256 512 1024"}

mkdir -p "$LOGDIR"
printf "stage\tcenter\tstatus\tlog\n" > "$SUMMARY"

case_no=0
ok_count=0
fail_count=0
skip_count=0

for STAGE in $STAGES; do
  for CENTER in $CENTERS; do
    case_no=$((case_no + 1))
    if (( case_no < START_CASE )); then
      skip_count=$((skip_count + 1))
      continue
    fi
    if (( MAX_CASES > 0 && case_no >= START_CASE + MAX_CASES )); then
      break 2
    fi

    LOG="$LOGDIR/p${P}_odd${ODD}_stage${STAGE}_center${CENTER}.log"
    echo "=== case #$case_no: stage=$STAGE center=$CENTER p=$P odd=$ODD ==="

    CMD=("$BIN" "$P"
      --modulus crt
      --crt-odd-radix "$ODD"
      --crt-center-mode halfreal
      --crt-halfreal-no-autoprobe
      --crt-halfreal-flags 48
      --crt-halfreal-validate-random
      --crt-halfreal-validate-iters "$VAL_ITERS"
      --crt-mixed-row-core lds
      --crt-mixed-row-stage "$STAGE"
      --crt-mixed-row-center "$CENTER"
      --device "$DEVICE"
      --iters "$ITERS")

    if (( PROFILE != 0 )); then
      CMD+=(--profile-kernels)
    fi

    printf 'cmd:' > "$LOG"
    printf ' %q' "${CMD[@]}" >> "$LOG"
    printf '\n' >> "$LOG"

    if "${CMD[@]}" 2>&1 | tee -a "$LOG"; then
      echo "OK stage=$STAGE center=$CENTER"
      printf "%s\t%s\tOK\t%s\n" "$STAGE" "$CENTER" "$LOG" >> "$SUMMARY"
      ok_count=$((ok_count + 1))
    else
      echo "FAIL stage=$STAGE center=$CENTER, log=$LOG" >&2
      printf "%s\t%s\tFAIL\t%s\n" "$STAGE" "$CENTER" "$LOG" >> "$SUMMARY"
      fail_count=$((fail_count + 1))
      if (( STOP_ON_FAIL != 0 )); then
        exit 1
      fi
    fi
  done
done

echo "summary: ok=$ok_count fail=$fail_count skipped=$skip_count logs=$LOGDIR summary=$SUMMARY"
if (( fail_count != 0 )); then
  exit 1
fi
