# chainpm1.sh
# Chain P-1 stage 1 by B1 steps. First: -b1 <start>, then: <RESUME_FLAG> <curr> -b1 <next>.
# Stop if "P-1 factor stage 1 found:" appears.
# Usage: ./chainpm1.sh <exp> <start_B1> <step> [max_B1]

#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <exponent> <start_B1> <step> [max_B1]" >&2
  exit 1
fi

PROG="${PRMERS_BIN:-./prmers}"
RESUME_FLAG="${RESUME_FLAG:--b1old}"

p="$1"
curr="$2"
step="$3"
max="${4:-}"

log="pm1_p${p}_B1_${curr}.log"
echo "[INFO] Run: ${PROG} ${p} -pm1 -b1 ${curr}"
set +e
"${PROG}" "${p}" -pm1 -b1 "${curr}" | tee "${log}"
set -e

if grep -q "P-1 factor stage 1 found:" "${log}"; then
  factor=$(sed -n 's/.*P-1 factor stage 1 found: \([0-9][0-9]*\).*/\1/p' "${log}" | head -n1)
  echo "[FOUND] Factor ${factor} at B1=${curr}"
  exit 0
fi

while :; do
  next=$(( curr + step ))
  if [ -n "${max}" ] && [ "${next}" -gt "${max}" ]; then
    echo "[STOP] Reached max_B1=${max} (next=${next}). No factor found."
    exit 1
  fi

  log="pm1_p${p}_B1_${curr}_to_${next}.log"
  echo "[INFO] Run: ${PROG} ${p} -pm1 ${RESUME_FLAG} ${curr} -b1 ${next}"
  set +e
  "${PROG}" "${p}" -pm1 "${RESUME_FLAG}" "${curr}" -b1 "${next}" | tee "${log}"
  set -e

  if grep -q "P-1 factor stage 1 found:" "${log}"; then
    factor=$(sed -n 's/.*P-1 factor stage 1 found: \([0-9][0-9]*\).*/\1/p' "${log}" | head -n1)
    echo "[FOUND] Factor ${factor} between B1=${curr} and B1=${next}"
    exit 0
  fi

  curr=${next}
done
