# chainpm1old.sh
# Extend P-1 stage 1 from existing .save/.p95 at <old_B1>: <RESUME_FLAG> <old_B1> -b1 <old_B1+step>.
# Stop if "P-1 factor stage 1 found:" appears.
# Usage: ./chainpm1old.sh <exp> <old_B1> <step>

#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <exponent> <old_B1> <step>" >&2
  exit 1
fi

PROG="${PRMERS_BIN:-./prmers}"
RESUME_FLAG="${RESUME_FLAG:--b1old}"

p="$1"
curr="$2"
step="$3"

while :; do
  next=$(( curr + step ))
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
