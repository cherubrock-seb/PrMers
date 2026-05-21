#!/usr/bin/env bash
set -euo pipefail

P=${P:-3021377}
DEVICE=${DEVICE:-1}
ODD=${ODD:-9}
ITERS=${ITERS:-1}
VAL_ITERS=${VAL_ITERS:-8}
# VALIDATE:
#   auto = enable only for small P where exact CPU reference is practical
#   1    = force validation
#   0    = benchmark/run without the halfreal validator
VALIDATE=${VALIDATE:-auto}
CPU_REF_MAX_P=${CPU_REF_MAX_P:-5000000}
PROFILE=${PROFILE:-1}
PROFILE_VALIDATOR=${PROFILE_VALIDATOR:-0}
GPU_REF=${GPU_REF:-0}
# Single-LDS center policy.  Empty global keeps program default.
# Current v23 program default is GF61=1, GF31=0.
SINGLE_LDS_CENTER=${SINGLE_LDS_CENTER:-}
SINGLE_LDS_CENTER_61=${SINGLE_LDS_CENTER_61:-}
SINGLE_LDS_CENTER_31=${SINGLE_LDS_CENTER_31:-}
# Stage single-LDS policy. v23 has fixed-size one-LDS kernels for 8..1024.
# Current v23 program default is GF61=1, GF31=0. These flags also prevent
# automatic fused 61x31 LDS stage unless override is set.
SINGLE_LDS_STAGE=${SINGLE_LDS_STAGE:-}
SINGLE_LDS_STAGE_61=${SINGLE_LDS_STAGE_61:-}
SINGLE_LDS_STAGE_31=${SINGLE_LDS_STAGE_31:-}
FUSE_OVERRIDES_SINGLE_LDS=${FUSE_OVERRIDES_SINGLE_LDS:-0}
CENTER_FUSE_OVERRIDES_SINGLE_LDS=${CENTER_FUSE_OVERRIDES_SINGLE_LDS:-$FUSE_OVERRIDES_SINGLE_LDS}
STAGE_FUSE_OVERRIDES_SINGLE_LDS=${STAGE_FUSE_OVERRIDES_SINGLE_LDS:-$FUSE_OVERRIDES_SINGLE_LDS}
BIN=${BIN:-./prmers_opencl_prp}
MAX_CASES=${MAX_CASES:-0}
START_CASE=${START_CASE:-1}
STOP_ON_FAIL=${STOP_ON_FAIL:-0}
LOGDIR=${LOGDIR:-mixed_row_lds_matrix_logs}
SUMMARY=${SUMMARY:-$LOGDIR/summary.tsv}
DETAIL=${DETAIL:-$LOGDIR/summary_detail.tsv}
KERNELS=${KERNELS:-$LOGDIR/kernel_profile.tsv}
COMBINED=${COMBINED:-$LOGDIR/all_tests_combined.log}
STAGES=${STAGES:-"8 16 32 64 128 256 512 1024"}
CENTERS=${CENTERS:-"8 16 32 64 128 256 512 1024"}
FUSE_BOTHS=${FUSE_BOTHS:-"off all"}

mkdir -p "$LOGDIR"
printf "case\tfuse\tstage\tcenter\tstatus\tlog\n" > "$SUMMARY"
: > "$COMBINED"

case_no=0
ok_count=0
fail_count=0
skip_count=0

should_validate=0
if [[ "$VALIDATE" == "1" || "$VALIDATE" == "yes" || "$VALIDATE" == "true" ]]; then
  should_validate=1
elif [[ "$VALIDATE" == "0" || "$VALIDATE" == "no" || "$VALIDATE" == "false" ]]; then
  should_validate=0
else
  if (( VAL_ITERS > 0 && (P <= CPU_REF_MAX_P || GPU_REF != 0) )); then
    should_validate=1
  else
    should_validate=0
  fi
fi

{
  echo "# P=$P DEVICE=$DEVICE ODD=$ODD ITERS=$ITERS PROFILE=$PROFILE PROFILE_VALIDATOR=$PROFILE_VALIDATOR"
  echo "# VALIDATE=$VALIDATE VAL_ITERS=$VAL_ITERS CPU_REF_MAX_P=$CPU_REF_MAX_P GPU_REF=$GPU_REF should_validate=$should_validate"
  echo "# SINGLE_LDS_CENTER=$SINGLE_LDS_CENTER SINGLE_LDS_CENTER_61=$SINGLE_LDS_CENTER_61 SINGLE_LDS_CENTER_31=$SINGLE_LDS_CENTER_31"
  echo "# SINGLE_LDS_STAGE=$SINGLE_LDS_STAGE SINGLE_LDS_STAGE_61=$SINGLE_LDS_STAGE_61 SINGLE_LDS_STAGE_31=$SINGLE_LDS_STAGE_31"
  echo "# CENTER_FUSE_OVERRIDES_SINGLE_LDS=$CENTER_FUSE_OVERRIDES_SINGLE_LDS STAGE_FUSE_OVERRIDES_SINGLE_LDS=$STAGE_FUSE_OVERRIDES_SINGLE_LDS"
  echo "# STAGES=$STAGES"
  echo "# CENTERS=$CENTERS"
  echo "# FUSE_BOTHS=$FUSE_BOTHS"
  echo
} | tee -a "$COMBINED" >/dev/null

for FUSE in $FUSE_BOTHS; do
  for STAGE in $STAGES; do
    for CENTER in $CENTERS; do
      case_no=$((case_no + 1))
      if (( case_no < START_CASE )); then
        skip_count=$((skip_count + 1))
        continue
      fi
      if (( MAX_CASES > 0 && case_no >= START_CASE + MAX_CASES )); then
        break 3
      fi

      LOG="$LOGDIR/p${P}_odd${ODD}_fuse${FUSE}_stage${STAGE}_center${CENTER}.log"
      echo "=== case #$case_no: fuse=$FUSE stage=$STAGE center=$CENTER p=$P odd=$ODD validate=$VALIDATE gpu_ref=$GPU_REF ===" | tee -a "$COMBINED"

      # Hardware limit: fused 61x31 pair-center needs four LDS arrays.
      # center=1024 exceeds the NVIDIA 48 KiB local memory limit, so force mode
      # is intentionally unsupported there. Other modes may fall back cleanly.
      if [[ "$FUSE" == "force" && "$CENTER" -gt 512 ]]; then
        echo "SKIP unsupported: fuse=force center=$CENTER exceeds fused 61x31 center LDS limit" | tee -a "$COMBINED"
        printf "%s\t%s\t%s\t%s\tSKIP_UNSUPPORTED\t%s\n" "$case_no" "$FUSE" "$STAGE" "$CENTER" "$LOG" >> "$SUMMARY"
        {
          echo "cmd: skipped before run"
          echo "reason: fuse=force center=$CENTER exceeds fused 61x31 center LDS limit"
          echo "__STATUS__=SKIP_UNSUPPORTED"
        } > "$LOG"
        skip_count=$((skip_count + 1))
        echo "" | tee -a "$COMBINED"
        continue
      fi

      ENVARGS=()
      [[ -n "$SINGLE_LDS_CENTER" ]] && ENVARGS+=("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS=$SINGLE_LDS_CENTER")
      [[ -n "$SINGLE_LDS_CENTER_61" ]] && ENVARGS+=("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_61=$SINGLE_LDS_CENTER_61")
      [[ -n "$SINGLE_LDS_CENTER_31" ]] && ENVARGS+=("PRMERS_CRT_MIXED_CENTER_SINGLE_LDS_31=$SINGLE_LDS_CENTER_31")
      [[ -n "$SINGLE_LDS_STAGE" ]] && ENVARGS+=("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS=$SINGLE_LDS_STAGE")
      [[ -n "$SINGLE_LDS_STAGE_61" ]] && ENVARGS+=("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_61=$SINGLE_LDS_STAGE_61")
      [[ -n "$SINGLE_LDS_STAGE_31" ]] && ENVARGS+=("PRMERS_CRT_MIXED_STAGE_SINGLE_LDS_31=$SINGLE_LDS_STAGE_31")
      ENVARGS+=("PRMERS_CRT_MIXED_CENTER_FUSE_OVERRIDES_SINGLE_LDS=$CENTER_FUSE_OVERRIDES_SINGLE_LDS")
      ENVARGS+=("PRMERS_CRT_MIXED_STAGE_FUSE_OVERRIDES_SINGLE_LDS=$STAGE_FUSE_OVERRIDES_SINGLE_LDS")

      CMD=(env "${ENVARGS[@]}" "$BIN" "$P"
        --modulus crt
        --crt-odd-radix "$ODD"
        --crt-center-mode halfreal
        --crt-halfreal-no-autoprobe
        --crt-halfreal-flags 48
        --crt-mixed-row-core lds
        --crt-mixed-row-stage "$STAGE"
        --crt-mixed-row-center "$CENTER"
        --crt-mixed-row-fuse-both "$FUSE"
        --device "$DEVICE"
        --iters "$ITERS")

      if (( should_validate != 0 )); then
        CMD+=(--crt-halfreal-validate-random --crt-halfreal-validate-iters "$VAL_ITERS")
        if (( GPU_REF != 0 )); then
          CMD+=(--crt-mixed-gpu-reference)
        fi
      fi

      if (( PROFILE != 0 )); then
        CMD+=(--profile-kernels)
      fi
      if (( PROFILE_VALIDATOR != 0 )); then
        CMD+=(--profile-validator)
      fi

      printf 'cmd:' > "$LOG"
      printf ' %q' "${CMD[@]}" >> "$LOG"
      printf '\n' >> "$LOG"
      cat "$LOG" >> "$COMBINED"

      set +e
      "${CMD[@]}" 2>&1 | tee -a "$LOG" | tee -a "$COMBINED"
      rc=${PIPESTATUS[0]}
      set -e

      # Some validation-stop paths return rc=0 but did not actually run the benchmark.
      # Important: with --iters 1 the program normally prints
      # "benchmark stopped after 1 iterations; no PRP result computed".
      # That is NOT a failure. It only means no full PRP residue was produced.
      # A real no-run is when the program explicitly stopped before the PRP loop,
      # or when validation could not run because CPU reference was refused.
      bad_success=0
      if grep -Eqi 'Stopping before the PRP loop|validator requires exact CPU reference' "$LOG"; then
        bad_success=1
      fi

      if (( rc == 0 && bad_success == 0 )); then
        echo "__STATUS__=OK" >> "$LOG"
        echo "OK fuse=$FUSE stage=$STAGE center=$CENTER" | tee -a "$COMBINED"
        printf "%s\t%s\t%s\t%s\tOK\t%s\n" "$case_no" "$FUSE" "$STAGE" "$CENTER" "$LOG" >> "$SUMMARY"
        ok_count=$((ok_count + 1))
      else
        if (( bad_success != 0 )); then
          echo "__STATUS__=FAIL_NORUN rc=$rc" >> "$LOG"
          echo "FAIL_NORUN fuse=$FUSE stage=$STAGE center=$CENTER, validation stopped before PRP loop, log=$LOG" | tee -a "$COMBINED" >&2
          printf "%s\t%s\t%s\t%s\tFAIL_NORUN\t%s\n" "$case_no" "$FUSE" "$STAGE" "$CENTER" "$LOG" >> "$SUMMARY"
        else
          echo "__STATUS__=FAIL rc=$rc" >> "$LOG"
          echo "FAIL fuse=$FUSE stage=$STAGE center=$CENTER, log=$LOG" | tee -a "$COMBINED" >&2
          printf "%s\t%s\t%s\t%s\tFAIL\t%s\n" "$case_no" "$FUSE" "$STAGE" "$CENTER" "$LOG" >> "$SUMMARY"
        fi
        fail_count=$((fail_count + 1))
        if (( STOP_ON_FAIL != 0 )); then
          exit 1
        fi
      fi
      echo "" | tee -a "$COMBINED"
    done
  done
done

python3 - "$LOGDIR" "$SUMMARY" "$DETAIL" "$KERNELS" <<'PY'
import csv
import pathlib
import re
import sys

logdir = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
detail_path = pathlib.Path(sys.argv[3])
kernels_path = pathlib.Path(sys.argv[4])

with summary_path.open(newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

kernel_re = re.compile(r"^\s{2}(.+?):\s+([0-9.]+)\s+ms total \(([0-9.]+)%.*, launches=([0-9]+)\)")
iter_re = re.compile(r"it/s\s+([0-9.]+)")
elapsed_re = re.compile(r"elapsed\s+([0-9.]+)\s+s")
bad_re = re.compile(r"Stopping before the PRP loop|validator requires exact CPU reference", re.I)

with detail_path.open('w', newline='') as df, kernels_path.open('w', newline='') as kf:
    dw = csv.writer(df, delimiter='\t')
    kw = csv.writer(kf, delimiter='\t')
    dw.writerow(['case','fuse','stage','center','status','it_s','elapsed_s','validation','ran_prp_loop','top_kernel','top_ms','top_pct','log'])
    kw.writerow(['case','fuse','stage','center','status','kernel','ms','pct','launches','log'])
    for row in rows:
        log_path = pathlib.Path(row['log'])
        text = log_path.read_text(errors='replace') if log_path.exists() else ''
        it_s = ''
        elapsed_s = ''
        for m in iter_re.finditer(text): it_s = m.group(1)
        for m in elapsed_re.finditer(text): elapsed_s = m.group(1)
        ran_prp_loop = '0' if bad_re.search(text) else ('1' if re.search(r'iter\s+[0-9]+/', text) or it_s or 'kernel profile summary' in text else '')
        vlines = [ln.strip() for ln in text.splitlines() if re.search(r'validat|mismatch|FAILED|FAIL|no PRP|Stopping before|\bOK\b', ln, re.I)]
        validation = ' | '.join(vlines[-6:])[:800]
        krows = []
        for ln in text.splitlines():
            m = kernel_re.match(ln)
            if m:
                krows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
        top = krows[0] if krows else ('','','','')
        dw.writerow([row['case'], row['fuse'], row['stage'], row['center'], row['status'], it_s, elapsed_s,
                     validation, ran_prp_loop, top[0], top[1], top[2], row['log']])
        for name, ms, pct, launches in krows:
            kw.writerow([row['case'], row['fuse'], row['stage'], row['center'], row['status'], name, ms, pct, launches, row['log']])
PY

echo "summary: ok=$ok_count fail=$fail_count skipped=$skip_count"
echo "logs: $LOGDIR"
echo "combined: $COMBINED"
echo "summary: $SUMMARY"
echo "detail: $DETAIL"
echo "kernels: $KERNELS"
if (( fail_count != 0 )); then
  exit 1
fi
