#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$SCRIPT_DIR/validate_v9985.py" \
  --platform macos \
  --base "${PRMERS_BASE:-$HOME/Downloads}" \
  --device "${PRMERS_DEVICE:-0}" \
  --profile "${PRMERS_VALIDATION_PROFILE:-standard}" "$@"
