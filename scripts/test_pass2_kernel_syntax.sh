#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLANG_REAL="${CLANG:-$(command -v clang || true)}"
[[ -n "$CLANG_REAL" ]] || { echo "Enabled GF61 kernel syntax NOT tested: clang unavailable" >&2; exit 2; }
export AEVUM_TEST_CLANG_REAL="$CLANG_REAL"
WRAPPER="$(mktemp)"
trap 'rm -f "$WRAPPER"' EXIT
cat > "$WRAPPER" <<'EOF'
#!/usr/bin/env bash
exec "$AEVUM_TEST_CLANG_REAL" -DAEVUM_GF61_LIMB32=1 -DWMUL="${AEVUM_TEST_WMUL:-2}" "$@"
EOF
chmod +x "$WRAPPER"
# Both power-of-two types and PFA include the changed scalar/complex products.
for WMUL in 1 2 4; do
  AEVUM_TEST_WMUL="$WMUL" CLANG="$WRAPPER" bash "$ROOT/third_party/aevum/tests/pow2_type4_opencl_syntax.sh"
done
CLANG="$WRAPPER" bash "$ROOT/third_party/aevum/tests/native_pfa_opencl_syntax.sh"
