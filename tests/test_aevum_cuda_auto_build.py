#!/usr/bin/env python3
"""Host-only regression tests for optional CUDA SDK discovery and build selection.

Uses disposable fake SDKs and a fake sub-make. No CUDA Toolkit, GPU, compiler,
network access, or modification of checked-in Aevum files is needed.
"""
import os
from pathlib import Path
import subprocess
import tempfile

SCRIPT = Path("scripts/build_aevum_cuda.sh").resolve()


def invoke(mode, root, vendor, fake_make, log, *, windows=False, failure=False):
    env = dict(os.environ)
    env.update(
        AEVUM_CUDA_ROOT=str(root),
        AEVUM_VENDOR_DIR=str(vendor),
        AEVUM_MAKE=str(fake_make),
        AEVUM_TEST_CALL_LOG=str(log),
        AEVUM_TEST_BUILD_FAIL="1" if failure else "0",
    )
    env.pop("AEVUM_CUDA_LIBS", None)
    env.pop("AEVUM_CUDA_FLAGS", None)
    # Simulate target OS explicitly so the host regression test runs on
    # Windows/MSYS and Linux regardless of the actual CI runner.
    fake_bin = fake_make.parent / ("fake-bin-windows" if windows else "fake-bin-linux")
    fake_bin.mkdir(exist_ok=True)
    uname = fake_bin / "uname"
    uname.write_text("#!/bin/sh\\nprintf 'MINGW64_NT\\n'\\n" if windows
                     else "#!/bin/sh\\nprintf 'Linux\\n'\\n")
    uname.chmod(0o755)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    return subprocess.run(["sh", str(SCRIPT), mode], env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


with tempfile.TemporaryDirectory(prefix="prmers-cuda-autobuild-") as td:
    temp = Path(td)
    vendor = temp / "vendor"
    (vendor / "build-engine").mkdir(parents=True)
    opencl = vendor / "build-engine" / "libaevum_engine.so"
    opencl.write_bytes(b"OPENCL_PRESERVED")
    root = temp / "missing-sdk"
    fake_make = temp / "fake-make"
    log = temp / "calls"
    fake_make.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$AEVUM_TEST_CALL_LOG"\n'
        'if [ "$AEVUM_TEST_BUILD_FAIL" = 1 ]; then exit 42; fi\n'
        'mkdir -p "$2/build-engine-cuda"\n'
        'printf "CUDA_PLUGIN" > "$2/build-engine-cuda/libaevum_engine.so"\n'
    )
    fake_make.chmod(0o755)
    r = invoke("auto", root, vendor, fake_make, log)
    assert r.returncode == 0 and "OpenCL-only" in r.stdout, r.stdout
    assert not log.exists(), "No SDK must not trigger CUDA compilation"
    r = invoke("required", root, vendor, fake_make, log)
    assert r.returncode != 0 and "CUDA development SDK not found" in r.stdout, r.stdout
    r = invoke("off", root, vendor, fake_make, log)
    assert r.returncode == 0 and not log.exists(), r.stdout

    inc = root / "include"
    lib = root / "lib64"
    stub = lib / "stubs"
    for d in (inc, lib, stub):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("cuda.h", "nvrtc.h"):
        (inc / name).write_text("/* fake header */")
    (lib / "libnvrtc.so").write_text("fake")
    (stub / "libcuda.so").write_text("fake")
    r = invoke("auto", root, vendor, fake_make, log)
    assert r.returncode == 0 and "CUDA plugin ready:" in r.stdout, r.stdout
    assert opencl.read_bytes() == b"OPENCL_PRESERVED", "OpenCL plugin mutated"
    cuda = vendor / "build-engine" / "libaevum_engine_cuda.so"
    assert cuda.read_bytes() == b"CUDA_PLUGIN"
    assert "ENGINE_BIN=build-engine-cuda" in log.read_text()
    assert "libnvrtc.so" in log.read_text()
    assert "libcuda.so" in log.read_text()
    r = invoke("required", root, vendor, fake_make, log)
    assert r.returncode == 0, r.stdout
    r = invoke("auto", root, vendor, fake_make, log, failure=True)
    assert r.returncode == 0 and "retaining OpenCL" in r.stdout, r.stdout
    r = invoke("required", root, vendor, fake_make, log, failure=True)
    assert r.returncode != 0, r.stdout

    # Simulated Windows/MSYS path with native CUDA import libraries.
    win_sdk = temp / "windows SDK"
    (win_sdk / "include").mkdir(parents=True)
    (win_sdk / "lib" / "x64").mkdir(parents=True)
    for name in ("cuda.h", "nvrtc.h"):
        (win_sdk / "include" / name).write_text("header")
    for name in ("cuda.lib", "nvrtc.lib"):
        (win_sdk / "lib" / "x64" / name).write_text("lib")
    r = invoke("required", win_sdk, vendor, fake_make, log, windows=True)
    assert r.returncode == 0, r.stdout
    assert (vendor / "build-engine" / "aevum_engine_cuda.dll").read_bytes() == b"CUDA_PLUGIN"
    assert "nvrtc.lib" in log.read_text()
    assert opencl.read_bytes() == b"OPENCL_PRESERVED"

print("AEVUM_CUDA_AUTO_BUILD_HOST_TEST=PASS")
