#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-${PRMERS_TEST_DEVICE:-0}}"
PROFILE="${2:-${PRMERS_MATRIX_PROFILE:-standard}}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="${PRMERS_MATRIX_DIR:-$ROOT/tests/backend-validation-$STAMP}"
SHORT="${PRMERS_MATRIX_SHORT_SECONDS:-35}"
MEDIUM="${PRMERS_MATRIX_MEDIUM_SECONDS:-90}"
LONG="${PRMERS_MATRIX_LONG_SECONDS:-180}"
COMPLETE="${PRMERS_MATRIX_COMPLETE_SECONDS:-900}"
ECM_SEED="${PRMERS_MATRIX_ECM_SEED:-123456789}"
CASE_FILTER="${PRMERS_MATRIX_CASE_FILTER:-}"
KILL_AFTER="${PRMERS_MATRIX_KILL_AFTER_SECONDS:-25}"
SUMMARY="$BASE/summary.tsv"
COMBINED="$BASE/combined.log"
SYSTEM="$BASE/system.txt"
COMPARISONS="$BASE/comparisons.tsv"
ERRORS="$BASE/errors.tsv"
MANIFEST="$BASE/report-manifest.txt"
FAILURES=0
PASSES=0
SKIPS=0

usage() {
  cat <<USAGE
Usage: $0 [device-id] [quick|standard|full]

Environment:
  PRMERS_BIN=/path/to/prmers
  PRMERS_MATRIX_DIR=/path/to/output
  PRMERS_MATRIX_SHORT_SECONDS=$SHORT
  PRMERS_MATRIX_MEDIUM_SECONDS=$MEDIUM
  PRMERS_MATRIX_LONG_SECONDS=$LONG
  PRMERS_MATRIX_COMPLETE_SECONDS=$COMPLETE
  PRMERS_MATRIX_ECM_SEED=$ECM_SEED
  PRMERS_MATRIX_CASE_FILTER=regex-or-case-name

The script keeps running after individual failures and always creates one
.tar.gz report containing commands, logs, system information, a TSV summary,
and cross-backend result comparisons.
USAGE
}

if [[ "$PROFILE" != quick && "$PROFILE" != standard && "$PROFILE" != full ]]; then
  usage >&2
  exit 2
fi
if [[ ! -x "$BIN" ]]; then
  echo "Missing executable: $BIN" >&2
  exit 2
fi
for tool in timeout python3 tar sha256sum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required command not found: $tool" >&2
    exit 2
  fi
done

rm -rf "$BASE"
mkdir -p "$BASE/logs"
printf 'name\tgroup\tmode\texponent\trequest\texpected\tactual\texit\tstatus\tseconds\tmax_ips\ttransform\tresult_status\tres64\tfactor\tchecks\tnote\n' > "$SUMMARY"
: > "$COMBINED"

{
  echo "PrMers backend validation matrix"
  echo "timestamp=$STAMP"
  echo "profile=$PROFILE"
  echo "device=$DEVICE"
  echo "short_seconds=$SHORT"
  echo "medium_seconds=$MEDIUM"
  echo "long_seconds=$LONG"
  echo "complete_seconds=$COMPLETE"
  echo "ecm_seed=$ECM_SEED"
  echo "case_filter=${CASE_FILTER:-all}"
  echo "binary=$BIN"
  "$BIN" -v 2>&1 || true
  echo "git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo archive)"
  echo "git_describe=$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo archive)"
  echo "uname=$(uname -a)"
  command -v lsb_release >/dev/null 2>&1 && lsb_release -a 2>/dev/null || true
  command -v clinfo >/dev/null 2>&1 && clinfo -l 2>&1 || true
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>&1 || true
  echo "aevum_library=$ROOT/third_party/aevum/build-engine/libaevum_engine.so"
  if [[ -f "$ROOT/third_party/aevum/build-engine/libaevum_engine.so" ]]; then
    file "$ROOT/third_party/aevum/build-engine/libaevum_engine.so" 2>/dev/null || true
    sha256sum "$ROOT/third_party/aevum/build-engine/libaevum_engine.so" 2>/dev/null || true
  fi
} > "$SYSTEM"

extract_actual_backend() {
  local log="$1"
  if grep -q '\[Backend Aevum\] engine::Reg adapter active' "$log"; then echo Aevum; return; fi
  if grep -q '\[Backend Auto\].*: Aevum selected' "$log"; then echo Aevum-selected; return; fi
  if grep -q '\[Backend Auto\].*: Marin selected' "$log"; then echo Marin-selected; return; fi
  if grep -q '\[Backend Marin\]' "$log"; then echo Marin; return; fi
  if grep -q 'using Aevum engine' "$log"; then echo Aevum; return; fi
  if grep -q 'using Marin engine' "$log"; then echo Marin; return; fi
  if grep -q 'Sampling 100 iterations for IPS estimation' "$log"; then echo Internal-NTT; return; fi
  if grep -Eq 'Forced Aevum rejected|cannot be forced to Aevum|Forced Aevum request cannot be satisfied|not validated for Lucas-Lehmer' "$log"; then echo Rejected; return; fi
  echo Unknown
}

extract_max_ips() {
  local log="$1"
  { grep -Eo 'IPS: [0-9]+([.][0-9]+)?|it/s [0-9]+([.][0-9]+)?' "$log" || true; } \
    | grep -Eo '[0-9]+([.][0-9]+)?' \
    | awk 'BEGIN{m=0} {if($1>m)m=$1} END{printf "%.2f",m}'
}

extract_transform() {
  local log="$1" v
  v="$(grep -Eo 'transform=[0-9]+|Transform size=[0-9]+|Transform Size = [0-9]+' "$log" | tail -1 | grep -Eo '[0-9]+' || true)"
  echo "${v:-0}"
}

extract_json_value() {
  local key="$1" log="$2"
  grep -Eo '"'"$key"'":"[^"]*"' "$log" | tail -1 | sed -E 's/^"[^"]+":"(.*)"$/\1/' || true
}

extract_factor() {
  local log="$1" value
  value="$(extract_json_value factor "$log")"
  if [[ -z "$value" ]]; then
    value="$(grep -Ei 'factor[^0-9]{0,30}[0-9]{6,}' "$log" | tail -1 | grep -Eo '[0-9]{6,}' | tail -1 || true)"
  fi
  echo "$value"
}

case_group() {
  case "$1" in
    prp-small-*) echo prp-small ;;
    prp-medium-*) echo prp-medium ;;
    prp-large-*) echo prp-large ;;
    ll-safe-small-*) echo ll-safe-small ;;
    ll-safe-medium-*) echo ll-safe-medium ;;
    ll-safe-large-*) echo ll-safe-large ;;
    ll-unsafe-*) echo ll-unsafe ;;
    ll-safe2-*) echo ll-safe2 ;;
    pm1-small-*) echo pm1-small ;;
    pm1-medium-*) echo pm1-medium ;;
    pm1-large-*) echo pm1-large ;;
    pm1-lowmem-tiny-*) echo pm1-lowmem-tiny ;;
    pm1-lowmem-medium-*) echo pm1-lowmem-medium ;;
    pm1-lowmem-large-*) echo pm1-lowmem-large ;;
    pm1-ultralow-*) echo pm1-ultralow ;;
    ecm-tiny-*) echo ecm-tiny ;;
    ecm-medium-*) echo ecm-medium ;;
    ecm-large-*) echo ecm-large ;;
    *) echo "$1" ;;
  esac
}

quick_case() {
  case "$1" in
    prp-small-auto|prp-small-marin|prp-small-internal|prp-medium-aevum|\
    ll-safe-small-auto|ll-safe-small-marin|ll-safe-small-aevum-reject|\
    ll-safe-medium-aevum|ll-safe-large-auto|\
    pm1-small-vtrace-auto|pm1-lowmem-tiny-auto|pm1-lowmem-medium-aevum|\
    pm1-ultralow-auto|pm1-ultralow-aevum-reject|\
    ecm-tiny-edwards-auto|ecm-medium-aevum|ecm-large-auto) return 0 ;;
    *) return 1 ;;
  esac
}

# run_case name mode exponent request expected-regex timeout allowed-rcs required-regex -- args...
run_case() {
  local name="$1" mode="$2" exponent="$3" request="$4" expected="$5" seconds="$6" allowed="$7" required="$8"
  shift 8
  [[ "${1:-}" == -- ]] && shift
  local group
  group="$(case_group "$name")"

  if [[ -n "$CASE_FILTER" ]] && [[ ! "$name" =~ $CASE_FILTER ]]; then
    SKIPS=$((SKIPS+1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\tSKIPPED\t-\tSKIP\t0\t0\t0\t\t\t\t0\tcase-filter\n' \
      "$name" "$group" "$mode" "$exponent" "$request" "$expected" >> "$SUMMARY"
    return 0
  fi
  if [[ "$PROFILE" == quick ]] && ! quick_case "$name"; then
    SKIPS=$((SKIPS+1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\tSKIPPED\t-\tSKIP\t0\t0\t0\t\t\t\t0\tquick-profile\n' \
      "$name" "$group" "$mode" "$exponent" "$request" "$expected" >> "$SUMMARY"
    return 0
  fi

  local dir="$BASE/logs/$name" log="$BASE/logs/$name/run.log" cmdfile="$BASE/logs/$name/command.txt"
  mkdir -p "$dir"
  printf '%q ' "$BIN" "$@" -d "$DEVICE" --noask -f "$dir" > "$cmdfile"
  echo >> "$cmdfile"

  echo "===== $name =====" | tee -a "$COMBINED"
  cat "$cmdfile" | tee -a "$COMBINED"

  local t0 t1 rc elapsed actual maxips transform result_status res64 factor checks status note
  t0="$(date +%s)"
  set +e
  (
    cd "$dir"
    timeout --signal=INT --kill-after="${KILL_AFTER}s" "${seconds}s" \
      "$BIN" "$@" -d "$DEVICE" --noask -f "$dir"
  ) > "$log" 2>&1
  rc=$?
  set +e
  t1="$(date +%s)"
  elapsed=$((t1-t0))
  cat "$log" >> "$COMBINED"

  actual="$(extract_actual_backend "$log")"
  maxips="$(extract_max_ips "$log")"
  transform="$(extract_transform "$log")"
  result_status="$(extract_json_value status "$log")"
  res64="$(extract_json_value res64 "$log")"
  factor="$(extract_factor "$log")"
  checks="$(grep -Eci 'Check passed|Check OK|invariant OK|check_on_curve=OK' "$log" || true)"
  status=PASS
  note=""

  if grep -Eqi 'Segmentation fault|core dumped|Memory access fault|CL_BUILD_PROGRAM_FAILURE|terminate called after throwing|double free|invalid pointer|Gerbicz[^[:cntrl:]]*(FAIL|failed)|Check failed|invariant FAIL' "$log"; then
    status=FAIL; note=crash-or-correctness-signature
  elif ! grep -Eq "$expected" "$log"; then
    status=FAIL; note=expected-backend-pattern-missing
  elif [[ -n "$required" ]] && ! grep -Eq "$required" "$log"; then
    status=FAIL; note=required-result-pattern-missing
  elif [[ ",${allowed}," != *",${rc},"* ]]; then
    status=FAIL; note=unexpected-exit
  elif [[ "$rc" == 137 ]]; then
    if grep -Eq 'Progress:|Check passed|Check OK|invariant OK|check_on_curve=OK|Stage1|Chunk 1/1' "$log"; then
      note=timeout-kill-after-useful-progress
    else
      status=FAIL; note=timeout-kill-before-useful-progress
    fi
  fi

  if [[ "$status" == PASS ]]; then
    PASSES=$((PASSES+1))
  else
    FAILURES=$((FAILURES+1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$group" "$mode" "$exponent" "$request" "$expected" "$actual" "$rc" "$status" \
    "$elapsed" "${maxips:-0}" "$transform" "$result_status" "$res64" "$factor" "$checks" "$note" >> "$SUMMARY"
  echo "[$status] $name actual=$actual rc=$rc elapsed=${elapsed}s max_ips=${maxips:-0} transform=$transform result=$result_status res64=$res64 factor=$factor checks=$checks $note" | tee -a "$COMBINED"
}

# Exit 1 means a valid completed no-factor/composite result for several modes.
COMPLETE_RC="0,1"
SMOKE_RC="0,1,124,130,137"
REJECT_RC="2"

# -------------------- PRP --------------------
run_case prp-small-auto PRP 216091 auto '\[Backend Auto\] PRP: Marin selected' "$MEDIUM" "$COMPLETE_RC" '"status":"P"|probably prime|probable prime' -- \
  216091 -prp -proof 0
run_case prp-small-marin PRP 216091 forced-marin '\[Backend Marin\] PRP:' "$MEDIUM" "$COMPLETE_RC" '"status":"P"|probably prime|probable prime' -- \
  216091 -prp -proof 0 -engine-marin
run_case prp-small-internal PRP 216091 internal-ntt 'Sampling 100 iterations for IPS estimation' "$MEDIUM" "$COMPLETE_RC" '"status":"P"|probably prime|probable prime' -- \
  216091 -prp -proof 0 -marin
run_case prp-medium-auto PRP 1362763 auto '\[Backend Auto\] PRP: Marin selected' "$SHORT" "$SMOKE_RC" 'Progress:' -- \
  1362763 -prp -proof 0
run_case prp-medium-aevum PRP 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$SHORT" "$SMOKE_RC" 'PRP on .* using Aevum engine' -- \
  1362763 -prp -proof 0 -aevum
run_case prp-medium-marin PRP 1362763 forced-marin '\[Backend Marin\] PRP:' "$SHORT" "$SMOKE_RC" 'PRP on .* using Marin engine' -- \
  1362763 -prp -proof 0 -engine-marin
run_case prp-large-auto PRP 136279841 auto '\[Backend Auto\] PRP: Aevum selected' "$SHORT" "$SMOKE_RC" 'PRP on .* using Aevum engine' -- \
  136279841 -prp -proof 0

# -------------------- Lucas-Lehmer --------------------
run_case ll-safe-small-auto LL-safe 216091 auto '\[Backend Auto\] LL: Marin selected' "$MEDIUM" "$COMPLETE_RC" 'is prime|is composite' -- \
  216091 -ll
run_case ll-safe-small-marin LL-safe 216091 forced-marin '\[Backend Marin\] LL:' "$MEDIUM" "$COMPLETE_RC" 'is prime|is composite' -- \
  216091 -ll -engine-marin
run_case ll-safe-small-aevum-reject LL-safe 216091 forced-aevum 'Forced Aevum request cannot be satisfied' "$SHORT" "$REJECT_RC" 'No admissible FFT3161 plan|cannot be satisfied' -- \
  216091 -ll -aevum
run_case ll-safe-medium-aevum LL-safe 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$SHORT" "$SMOKE_RC" 'LL-SAFE on .* using Aevum engine' -- \
  1362763 -ll -aevum
run_case ll-safe-medium-marin LL-safe 1362763 forced-marin '\[Backend Marin\] LL:' "$SHORT" "$SMOKE_RC" 'LL-SAFE on .* using Marin engine' -- \
  1362763 -ll -engine-marin
run_case ll-safe-large-auto LL-safe 136279841 auto '\[Backend Auto\] LL: Aevum selected' "$SHORT" "$SMOKE_RC" 'LL-SAFE on .* using Aevum engine' -- \
  136279841 -ll
run_case ll-unsafe-medium-aevum LL-unsafe 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$SHORT" "$SMOKE_RC" 'LL-UNSAFE on .* using Aevum engine' -- \
  1362763 -llunsafe -aevum
run_case ll-unsafe-medium-marin LL-unsafe 1362763 forced-marin '\[Backend Marin\] LL:' "$SHORT" "$SMOKE_RC" 'LL-UNSAFE on .* using Marin engine' -- \
  1362763 -llunsafe -engine-marin
run_case ll-unsafe-small-internal-reject LL-unsafe 216091 internal-ntt-rejected 'legacy internal PrMers NTT backend' "$SHORT" "$REJECT_RC" 'not validated for Lucas-Lehmer' -- \
  216091 -llunsafe -marin
run_case ll-safe2-medium-aevum LL-safe2 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$SHORT" "$SMOKE_RC" 'LL-SAFE2 on .* using Aevum engine' -- \
  1362763 -llsafe2 -aevum
run_case ll-safe2-medium-marin LL-safe2 1362763 forced-marin '\[Backend Marin\] LL:' "$SHORT" "$SMOKE_RC" 'LL-SAFE2 on .* using Marin engine' -- \
  1362763 -llsafe2 -engine-marin
run_case ll-safe2-large-auto LL-safe2 136279841 auto '\[Backend Auto\] LL: Aevum selected' "$SHORT" "$SMOKE_RC" 'LL-SAFE2 on .* using Aevum engine' -- \
  136279841 -llsafe2

# -------------------- P-1 normal and low-memory --------------------
run_case pm1-small-vtrace-auto P-1 367 auto '\[Backend Auto\] P-1: Marin selected' "$MEDIUM" "$COMPLETE_RC" 'No P-1|P-1.*factor|factor.*P-1' -- \
  367 -pm1 -b1 11981 -b2 38971
run_case pm1-small-classic-auto P-1 367 auto '\[Backend Auto\] P-1: Marin selected' "$MEDIUM" "$COMPLETE_RC" 'No P-1|P-1.*factor|factor.*P-1' -- \
  367 -pm1 -b1 11981 -b2 38971 -pm1-vtrace-off
run_case pm1-medium-aevum P-1 16279841 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$MEDIUM" "$COMPLETE_RC" 'Gerbicz Li.*passed|No P-1|P-1.*factor|factor.*P-1' -- \
  16279841 -pm1 -b1 10000 -aevum
run_case pm1-medium-marin P-1 16279841 forced-marin '\[Backend Marin\] P-1:' "$MEDIUM" "$COMPLETE_RC" 'Gerbicz Li.*passed|No P-1|P-1.*factor|factor.*P-1' -- \
  16279841 -pm1 -b1 10000 -engine-marin
run_case pm1-large-auto P-1 136279841 auto '\[Backend Auto\] P-1: Aevum selected' "$SHORT" "$SMOKE_RC" 'Gerbicz Li.*passed|Progress:' -- \
  136279841 -pm1 -b1 1000000
run_case pm1-lowmem-tiny-auto P-1-lowmem 367 auto '\[Backend Auto\] P-1 low-memory: Marin selected' "$MEDIUM" "$COMPLETE_RC" 'Low-memory Stage 1 enabled' -- \
  367 -pm1 -b1 1000 -b2 10000 -pm1-lowmem
run_case pm1-lowmem-tiny-marin P-1-lowmem 367 forced-marin '\[Backend Marin\] P-1 low-memory:' "$MEDIUM" "$COMPLETE_RC" 'Low-memory Stage 1 enabled' -- \
  367 -pm1 -b1 1000 -b2 10000 -pm1-lowmem -engine-marin
run_case pm1-lowmem-medium-aevum P-1-lowmem 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$MEDIUM" "$SMOKE_RC" 'Low-memory Stage 1 enabled' -- \
  1362763 -pm1 -b1 1000 -b2 5000 -pm1-lowmem -aevum
run_case pm1-lowmem-large-auto P-1-lowmem 136279841 auto '\[Backend Auto\] P-1 low-memory: Aevum selected' "$SHORT" "$SMOKE_RC" 'Low-memory Stage 1 enabled' -- \
  136279841 -pm1 -b1 1000 -b2 5000 -pm1-lowmem
run_case pm1-ultralow-auto P-1-ultralow 2147483647 auto '\[Backend Auto\] P-1 ultra-low-memory: Marin selected' "$LONG" "$SMOKE_RC" 'fast3-only path' -- \
  2147483647 -pm1 -b1 100 -b2 5000 -pm1-ultralowmem -nogcd-stage1
run_case pm1-ultralow-marin P-1-ultralow 2147483647 forced-marin '\[Backend Marin\] P-1 ultra-low-memory:' "$LONG" "$SMOKE_RC" 'fast3-only path' -- \
  2147483647 -pm1 -b1 100 -b2 5000 -pm1-ultralowmem -nogcd-stage1 -engine-marin
run_case pm1-ultralow-aevum-reject P-1-ultralow 2147483647 forced-aevum 'Forced Aevum rejected|-pm1-ultralowmem is a Marin fast3-only' "$SHORT" "$REJECT_RC" 'cannot be forced to Aevum' -- \
  2147483647 -pm1 -b1 100 -b2 5000 -pm1-ultralowmem -nogcd-stage1 -aevum

# -------------------- ECM --------------------
run_case ecm-tiny-edwards-auto ECM-Edwards 701 auto '\[Backend Auto\] ECM: Marin selected' "$MEDIUM" "$COMPLETE_RC" '\[ECM\] No factor found|Factor ECM' -- \
  701 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -seed "$ECM_SEED" -ecm_progress_ms 500
run_case ecm-tiny-montgomery-auto ECM-Montgomery 701 auto '\[Backend Auto\] ECM: Marin selected' "$MEDIUM" "$COMPLETE_RC" '\[ECM\] No factor found|Factor ECM' -- \
  701 -ecm -b1 100 -b2 1000 -K 1 -montgomery -seed "$ECM_SEED" -ecm_progress_ms 500
run_case ecm-medium-auto ECM-Edwards 1362763 auto '\[Backend Auto\] ECM: Marin selected' "$MEDIUM" "$COMPLETE_RC" '\[ECM\] No factor found|Factor ECM' -- \
  1362763 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -seed "$ECM_SEED" -ecm_progress_ms 500
run_case ecm-medium-aevum ECM-Edwards 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$MEDIUM" "$SMOKE_RC" 'check_on_curve=OK|Stage1' -- \
  1362763 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -aevum -seed "$ECM_SEED" -ecm_progress_ms 500
run_case ecm-medium-torsion16-aevum ECM-TE16 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$MEDIUM" "$SMOKE_RC" 'torsion=16|check_on_curve=OK|Stage1' -- \
  1362763 -ecm -b1 100 -b2 1000 -K 1 -edwards -torsion16 -ced -aevum -seed "$ECM_SEED" -ecm_progress_ms 500
run_case ecm-large-auto ECM-Edwards 136279841 auto '\[Backend Auto\] ECM: Aevum selected' "$MEDIUM" "$SMOKE_RC" 'check_on_curve=OK|Stage1' -- \
  136279841 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -seed "$ECM_SEED" -ecm_progress_ms 1000
run_case ecm-large-marin ECM-Edwards 136279841 forced-marin '\[Backend Marin\] ECM:' "$MEDIUM" "$SMOKE_RC" 'check_on_curve=OK|Stage1' -- \
  136279841 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -engine-marin -seed "$ECM_SEED" -ecm_progress_ms 1000
run_case ecm-large-aevum ECM-Edwards 136279841 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$MEDIUM" "$SMOKE_RC" 'check_on_curve=OK|Stage1' -- \
  136279841 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -aevum -seed "$ECM_SEED" -ecm_progress_ms 1000

if [[ "$PROFILE" == full ]]; then
  # Complete medium-size pairs.  These are intentionally duplicated from the
  # smoke cases with a longer timeout so the report can compare final residues,
  # factors and result JSON across real Aevum and Marin executions.
  run_case prp-medium-aevum-complete PRP 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$COMPLETE" "$COMPLETE_RC" '"status":"P"|probably prime|probable prime|composite' -- \
    1362763 -prp -proof 0 -aevum
  run_case prp-medium-marin-complete PRP 1362763 forced-marin '\[Backend Marin\] PRP:' "$COMPLETE" "$COMPLETE_RC" '"status":"P"|probably prime|probable prime|composite' -- \
    1362763 -prp -proof 0 -engine-marin

  run_case ll-safe-medium-aevum-complete LL-safe 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$COMPLETE" "$COMPLETE_RC" 'is prime|is composite' -- \
    1362763 -ll -aevum
  run_case ll-safe-medium-marin-complete LL-safe 1362763 forced-marin '\[Backend Marin\] LL:' "$COMPLETE" "$COMPLETE_RC" 'is prime|is composite' -- \
    1362763 -ll -engine-marin

  run_case ll-unsafe-medium-aevum-complete LL-unsafe 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$COMPLETE" "$COMPLETE_RC" 'is prime|is composite' -- \
    1362763 -llunsafe -aevum
  run_case ll-unsafe-medium-marin-complete LL-unsafe 1362763 forced-marin '\[Backend Marin\] LL:' "$COMPLETE" "$COMPLETE_RC" 'is prime|is composite' -- \
    1362763 -llunsafe -engine-marin

  run_case ll-safe2-medium-aevum-complete LL-safe2 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$COMPLETE" "$COMPLETE_RC" 'is prime|is composite' -- \
    1362763 -llsafe2 -aevum
  run_case ll-safe2-medium-marin-complete LL-safe2 1362763 forced-marin '\[Backend Marin\] LL:' "$COMPLETE" "$COMPLETE_RC" 'is prime|is composite' -- \
    1362763 -llsafe2 -engine-marin

  run_case pm1-lowmem-medium-aevum-complete P-1-lowmem 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$COMPLETE" "$COMPLETE_RC" 'No P-1|P-1.*factor|factor.*P-1' -- \
    1362763 -pm1 -b1 1000 -b2 5000 -pm1-lowmem -aevum
  run_case pm1-lowmem-medium-marin-complete P-1-lowmem 1362763 forced-marin '\[Backend Marin\] P-1 low-memory:' "$COMPLETE" "$COMPLETE_RC" 'No P-1|P-1.*factor|factor.*P-1' -- \
    1362763 -pm1 -b1 1000 -b2 5000 -pm1-lowmem -engine-marin

  run_case ecm-medium-aevum-complete ECM-Edwards 1362763 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$COMPLETE" "$COMPLETE_RC" '\[ECM\] No factor found|Factor ECM' -- \
    1362763 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -aevum -seed "$ECM_SEED" -ecm_progress_ms 500
  run_case ecm-medium-marin-complete ECM-Edwards 1362763 forced-marin '\[Backend Marin\] ECM:' "$COMPLETE" "$COMPLETE_RC" '\[ECM\] No factor found|Factor ECM' -- \
    1362763 -ecm -b1 100 -b2 1000 -K 1 -edwards -notorsion -engine-marin -seed "$ECM_SEED" -ecm_progress_ms 500

  run_case pm1-pair95-medium P-1-pair95 1362763 auto '\[Backend Auto\] P-1: Marin selected' "$LONG" "$SMOKE_RC" 'PAIR95|Pair95|Stage 2' -- \
    1362763 -pm1 -b1 29 -b2 6910159 -pm1-vtrace-pair95 -pm1-vtrace-pair95-l 3
  run_case ecm-large-montgomery-aevum ECM-Montgomery 136279841 forced-aevum '\[Backend Aevum\] engine::Reg adapter active' "$LONG" "$SMOKE_RC" 'Stage1|Montgomery' -- \
    136279841 -ecm -b1 100 -b2 1000 -K 1 -montgomery -aevum -seed "$ECM_SEED" -ecm_progress_ms 1000
fi

python3 - "$SUMMARY" "$COMPARISONS" <<'PYCOMPARE'
import csv, sys
summary, output = sys.argv[1:]
with open(summary, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
by_name = {r['name']: r for r in rows if r.get('status') != 'SKIP'}
groups = {
    'prp-small-result': (['prp-small-auto', 'prp-small-marin', 'prp-small-internal'], ['result_status', 'res64']),
    'll-safe-small-result': (['ll-safe-small-auto', 'll-safe-small-marin'], ['result_status', 'res64']),
    'pm1-small-result': (['pm1-small-vtrace-auto', 'pm1-small-classic-auto'], ['result_status', 'factor']),
    'ecm-tiny-completion': (['ecm-tiny-edwards-auto', 'ecm-tiny-montgomery-auto'], ['result_status']),
    'prp-medium-aevum-vs-marin': (['prp-medium-marin-complete', 'prp-medium-aevum-complete'], ['result_status', 'res64']),
    'll-safe-medium-aevum-vs-marin': (['ll-safe-medium-marin-complete', 'll-safe-medium-aevum-complete'], ['result_status', 'res64']),
    'll-unsafe-medium-aevum-vs-marin': (['ll-unsafe-medium-marin-complete', 'll-unsafe-medium-aevum-complete'], ['result_status', 'res64']),
    'll-safe2-medium-aevum-vs-marin': (['ll-safe2-medium-marin-complete', 'll-safe2-medium-aevum-complete'], ['result_status', 'res64']),
    'pm1-lowmem-medium-aevum-vs-marin': (['pm1-lowmem-medium-marin-complete', 'pm1-lowmem-medium-aevum-complete'], ['result_status', 'factor']),
    'ecm-medium-aevum-vs-marin': (['ecm-medium-marin-complete', 'ecm-medium-aevum-complete'], ['result_status', 'factor']),
}
with open(output, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['group', 'baseline', 'candidate', 'fields', 'baseline_value', 'candidate_value', 'match',
                'baseline_ips', 'candidate_ips', 'candidate_over_baseline_ips', 'note'])
    for group, (names, fields) in groups.items():
        available = [by_name[n] for n in names if n in by_name]
        if len(available) < 2:
            w.writerow([group, '', '', ','.join(fields), '', '', 'SKIP', '', '', '', 'not enough executed rows'])
            continue
        base = available[0]
        base_value = '|'.join(base.get(k, '') for k in fields)
        for candidate in available[1:]:
            candidate_value = '|'.join(candidate.get(k, '') for k in fields)
            complete = bool(base.get('result_status')) and bool(candidate.get('result_status'))
            match = 'YES' if complete and base_value == candidate_value else ('NO' if complete else 'INCOMPLETE')
            try:
                bips = float(base.get('max_ips') or 0)
                cips = float(candidate.get('max_ips') or 0)
                ratio = f'{cips / bips:.4f}' if bips > 0 and cips > 0 else ''
            except ValueError:
                bips = cips = 0
                ratio = ''
            note = ''
            if match == 'NO': note = 'final result mismatch'
            elif match == 'INCOMPLETE': note = 'one or both cases timed out or produced no final JSON'
            w.writerow([group, base['name'], candidate['name'], ','.join(fields), base_value, candidate_value,
                        match, f'{bips:.2f}', f'{cips:.2f}', ratio, note])
PYCOMPARE

awk -F '\t' 'NR==1 || $9=="FAIL"' "$SUMMARY" > "$ERRORS"
{
  echo "PrMers backend validation report manifest"
  echo "generated=$STAMP"
  echo "profile=$PROFILE"
  echo "device=$DEVICE"
  echo "binary=$BIN"
  echo "summary=summary.tsv"
  echo "comparisons=comparisons.tsv"
  echo "errors=errors.tsv"
  echo "system=system.txt"
  echo "combined_log=combined.log"
  echo "case_logs=logs/<case>/run.log"
  echo "case_commands=logs/<case>/command.txt"
  echo "passes=$PASSES"
  echo "failures=$FAILURES"
  echo "skips=$SKIPS"
} > "$MANIFEST"

{
  echo
  echo "passes=$PASSES"
  echo "failures=$FAILURES"
  echo "skips=$SKIPS"
  echo "comparisons=$COMPARISONS"
} | tee -a "$COMBINED"

ARCHIVE="${BASE}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$BASE")" "$(basename "$BASE")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
echo "Report: $ARCHIVE"
echo "Checksum: ${ARCHIVE}.sha256"
echo "Summary: $SUMMARY"
echo "Comparisons: $COMPARISONS"
echo "Errors: $ERRORS"
echo "Manifest: $MANIFEST"

if (( FAILURES > 0 )); then
  exit 1
fi
exit 0
