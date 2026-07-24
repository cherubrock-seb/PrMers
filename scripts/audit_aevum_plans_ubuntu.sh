#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=(
  --root "$ROOT"
  --device "${PRMERS_DEVICE:-1}"
  --profile "${PRMERS_PLAN_AUDIT_PROFILE:-standard}"
  --seconds "${PRMERS_PLAN_AUDIT_SECONDS:-45}"
)
if [[ -n "${PRMERS_PLAN_AUDIT_REPEATS:-}" ]]; then
  ARGS+=(--repeats "$PRMERS_PLAN_AUDIT_REPEATS")
fi
if [[ "${PRMERS_PLAN_AUDIT_STRICT_POLICY:-0}" == "1" ]]; then
  ARGS+=(--strict-policy)
fi
exec python3 -u "$ROOT/scripts/audit_aevum_plans.py" "${ARGS[@]}" "$@"
