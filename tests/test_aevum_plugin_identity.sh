#!/usr/bin/env bash
# The low-range backend probe cache key includes the Aevum plugin identity, so a cached probe result is not
# reused after the plugin is swapped (AEVUM_ENGINE_LIB), rebuilt in place, or when no plugin loads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/tests/build-aevum-plugin-identity"
CXX="${CXX:-c++}"
rm -rf "$BUILD"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

# The key is built in src/marin/gpu.cpp from the identity.
grep -q '<< "|aevum-plugin=" << aevum_plugin_identity()' "$ROOT/src/marin/gpu.cpp"

if [[ "$(uname -s)" == Darwin ]]; then
  SHARED=(-dynamiclib)
  LIBS=(-pthread -framework OpenCL -lgmpxx -lgmp)
else
  SHARED=(-shared)
  LIBS=(-pthread -ldl -lOpenCL -lgmpxx -lgmp)
fi
for v in A B; do
  "$CXX" -std=c++20 -O2 -fPIC "${SHARED[@]}" -DFAKE_ENGINE_VERSION="\"fake-$v\"" \
    "$ROOT/tests/aevum_fake_engine.cpp" -o "$BUILD/libaevum_engine_$v.so"
done
cp "$BUILD/libaevum_engine_A.so" "$BUILD/libaevum_engine_A2.so"   # same version, different file
"$CXX" -std=c++20 -O2 -I"$ROOT/include" -I"$ROOT/include/marin" \
  "$ROOT/tests/aevum_plugin_identity_test.cpp" \
  "$ROOT/src/aevum/EngineAevum.cpp" "$ROOT/src/aevum/AutoPolicy.cpp" \
  "$ROOT/src/marin/gpu.cpp" "$ROOT/src/ui/WebGuiServer.cpp" \
  -DAEVUM_ENGINE_DEFAULT_LIB=\"/nonexistent/libaevum_engine.so\" \
  "${LIBS[@]}" -o "$BUILD/identity"

ident() { AEVUM_ENGINE_LIB="$1" "$BUILD/identity"; }
a="$(ident "$BUILD/libaevum_engine_A.so")"
a_again="$(ident "$BUILD/libaevum_engine_A.so")"
a2="$(ident "$BUILD/libaevum_engine_A2.so")"
b="$(ident "$BUILD/libaevum_engine_B.so")"
echo "A=$a"; echo "B=$b"
[ "$a" = "$a_again" ] || { echo "FAIL: identity of the same plugin is not stable"; exit 1; }
[ "$a" != "$b" ] || { echo "FAIL: plugins with different versions have the same identity"; exit 1; }
[ "$a" != "$a2" ] || { echo "FAIL: a different plugin file with the same version has the same identity"; exit 1; }
echo "$a" | grep -q 'version=fake-A' || { echo "FAIL: identity lacks the plugin version"; exit 1; }

# Rebuilt in place: same path and version, new modification time.
touch -d '2001-02-03 04:05:06' "$BUILD/libaevum_engine_A.so"
a_touched="$(ident "$BUILD/libaevum_engine_A.so")"
[ "$a" != "$a_touched" ] || { echo "FAIL: identity does not change when the plugin file is replaced"; exit 1; }

# No plugin loads (run where no default plugin is found): a fixed marker, no crash.
none="$(cd "$BUILD" && AEVUM_ENGINE_LIB="$BUILD/missing.so" "$BUILD/identity")"
[ "$none" = "unavailable" ] || { echo "FAIL: expected 'unavailable' without a plugin, got '$none'"; exit 1; }
echo "AEVUM_PLUGIN_IDENTITY=PASS"
