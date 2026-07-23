#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -u "$ROOT/scripts/audit_aevum_plans.py" \
  --root "$ROOT" \
  --device "${PRMERS_DEVICE:-1}" \
  --profile "${PRMERS_PLAN_AUDIT_PROFILE:-standard}" \
  --seconds "${PRMERS_PLAN_AUDIT_SECONDS:-45}" "$@"
