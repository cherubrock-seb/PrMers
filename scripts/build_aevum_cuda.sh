#!/bin/sh
# Optional native CUDA build. Keep CUDA objects separate from the OpenCL plugin.
set -eu
mode=${1:-auto}
case "$mode" in auto|required|off) ;; *) echo "[Aevum] Invalid CUDA build mode: $mode" >&2; exit 2;; esac
case "$(uname -s)" in
  Darwin) echo "[Aevum] macOS: OpenCL-only (no NVIDIA CUDA support)."; exit 0;;
  *MINGW*|*MSYS*|*CYGWIN*) windows=1;;
  *) windows=0;;
esac
if [ "$mode" = off ]; then
  echo "[Aevum] CUDA auto-build disabled (AEVUM_CUDA_MODE=off): OpenCL-only."
  exit 0
fi
hint() {
  echo "[Aevum] CUDA development SDK not found (headers, NVRTC and driver link library). OpenCL-only build remains functional."
  if [ "$windows" = 1 ]; then
    echo "[Aevum] Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads and rerun make."
  else
    echo "[Aevum] Ubuntu/Debian (after enabling NVIDIA's CUDA apt repository): sudo apt update && sudo apt install cuda-toolkit"
    echo "[Aevum] Instructions: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
  fi
  if [ "$mode" = required ]; then exit 1; fi
  exit 0
}
normalize() {
  case "$1" in
    [A-Za-z]:\\*|[A-Za-z]:/*)
      if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s\n' "$1"; fi;;
    *) printf '%s\n' "$1";;
  esac
}
nvcc_root=
if command -v nvcc >/dev/null 2>&1; then
  nvcc_root=$(dirname "$(dirname "$(command -v nvcc)")")
fi
if [ -n "${AEVUM_CUDA_ROOT:-}" ]; then
  set -- "$AEVUM_CUDA_ROOT"
elif [ -n "${CUDA_HOME:-}" ]; then
  set -- "$CUDA_HOME"
elif [ -n "${CUDA_PATH:-}" ]; then
  set -- "$CUDA_PATH"
else
  set -- "$nvcc_root" /usr/local/cuda /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr
fi
sdk_root=; sdk_inc=; driver_dir=; driver_lib=; nvrtc_dir=; nvrtc_lib=
for candidate do
  [ -n "$candidate" ] || continue
  root=$(normalize "$candidate")
  inc=
  for d in "$root/include" "$root/targets/x86_64-linux/include"; do
    if [ -r "$d/cuda.h" ] && [ -r "$d/nvrtc.h" ]; then inc=$d; break; fi
  done
  [ -n "$inc" ] || continue
  nd=; nl=; dd=; dl=
  if [ "$windows" = 1 ]; then
    for d in "$root/lib/x64" "$root/lib64"; do
      if [ -r "$d/cuda.lib" ] && [ -r "$d/nvrtc.lib" ]; then
        nd=$d; dd=$d; nl=nvrtc.lib; dl=cuda.lib; break
      fi
    done
  else
    for d in "$root/lib64" "$root/targets/x86_64-linux/lib" "$root/lib" /usr/lib/x86_64-linux-gnu /usr/local/lib; do
      for lib in libnvrtc.so libnvrtc.so.13 libnvrtc.so.12; do
        if [ -r "$d/$lib" ]; then nd=$d; nl=$lib; break; fi
      done
      [ -z "$nl" ] || break
    done
    # Real driver libcuda.so.1 is valid in Kaggle; SDK stubs work on
    # machines without installed NVIDIA drivers.
    for d in "$root/lib64/stubs" "$root/targets/x86_64-linux/lib/stubs" "$root/lib/stubs" /usr/lib/x86_64-linux-gnu/stubs /usr/local/nvidia/lib64 /usr/lib/x86_64-linux-gnu; do
      for lib in libcuda.so libcuda.so.1; do
        if [ -r "$d/$lib" ]; then dd=$d; dl=$lib; break; fi
      done
      [ -z "$dl" ] || break
    done
  fi
  if [ -n "$nl" ] && [ -n "$dl" ]; then
    sdk_root=$root; sdk_inc=$inc; nvrtc_dir=$nd; nvrtc_lib=$nl; driver_dir=$dd; driver_lib=$dl
    break
  fi
done
if [ -z "$sdk_root" ]; then hint; fi
cuda_flags=${AEVUM_CUDA_FLAGS:-"-DCUDA_BACKEND -Isrc/cuda -I\"$sdk_inc\""}
cuda_libs=${AEVUM_CUDA_LIBS:-"-L\"$driver_dir\" -l:$driver_lib -L\"$nvrtc_dir\" -l:$nvrtc_lib"}
vendor=${AEVUM_VENDOR_DIR:-third_party/aevum}
src="$vendor/build-engine-cuda/libaevum_engine.so"
if [ "$windows" = 1 ]; then dest="$vendor/build-engine/aevum_engine_cuda.dll"
else dest="$vendor/build-engine/libaevum_engine_cuda.so"; fi
echo "[Aevum] CUDA SDK found: $sdk_root (NVRTC=$nvrtc_lib, driver link=$driver_lib)"
echo "[Aevum] Building optional CUDA plugin in separate directory (OpenCL untouched)."
if [ "$windows" = 1 ]; then
  if "${AEVUM_MAKE:-make}" -C "$vendor" CUDA=1 ENGINE_BIN=build-engine-cuda \
    "CXX=g++" "DL_LIBS=" "ENGINE_LINK_FLAGS=-shared -static-libstdc++ -static-libgcc" \
    "CUDAFLAGS=$cuda_flags" "OPENCL_LIBS=$cuda_libs" engine-lib; then :;
  else
    echo "[Aevum] WARNING: CUDA build failed; retaining OpenCL." >&2
    if [ "$mode" = required ]; then exit 1; fi
    exit 0
  fi
else
  if "${AEVUM_MAKE:-make}" -C "$vendor" CUDA=1 ENGINE_BIN=build-engine-cuda \
    "CUDAFLAGS=$cuda_flags" "OPENCL_LIBS=$cuda_libs" engine-lib; then :;
  else
    echo "[Aevum] WARNING: CUDA build failed; retaining OpenCL." >&2
    if [ "$mode" = required ]; then exit 1; fi
    exit 0
  fi
fi
if [ ! -s "$src" ]; then
  echo "[Aevum] WARNING: CUDA build produced no library; retaining OpenCL." >&2
  if [ "$mode" = required ]; then exit 1; fi
  exit 0
fi
mkdir -p "$(dirname "$dest")"
if [ ! -s "$dest" ] || [ "$src" -nt "$dest" ]; then
  cp -p "$src" "$dest"
  echo "[Aevum] CUDA plugin ready: $dest"
else
  echo "[Aevum] CUDA plugin up to date: $dest"
fi
