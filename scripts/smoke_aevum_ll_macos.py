#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run_case(
    root: Path,
    binary: Path,
    library: Path,
    device: int,
    mode: str,
    backend: str,
    timeout_s: int,
    force_auto_ratio: bool = False,
) -> None:
    mode_label = "ll-safe" if mode == "-ll" else "ll-unsafe"
    backend_label = "auto" if backend == "-aevum-auto" else "forced"
    label = f"{mode_label}-{backend_label}"
    out_dir = root / f".aevum-macos-{label}-smoke"
    log_path = root / f".aevum-macos-{label}-smoke.log"
    shutil.rmtree(out_dir, ignore_errors=True)
    if log_path.exists():
        log_path.unlink()

    cmd = [str(binary), "1362763", mode, backend, "-d", str(device),
           "--noask", "-f", str(out_dir)]
    env = os.environ.copy()
    env["AEVUM_ENGINE_LIB"] = str(library)
    env["AEVUM_PREPARED_CACHE"] = "0"
    env["AEVUM_ENGINE_API_TRACE"] = "1"
    env["AEVUM_ENGINE_SYNC_EACH_OP"] = "1"
    if force_auto_ratio:
        env["AEVUM_AUTO_LL_MAX_RATIO"] = "10"

    required_markers = ["Loaded fftWGF61WidthFinalApple"]
    if mode == "-ll":
        required_markers.extend([
            "Loaded tailMulGF61LoadScalarApple",
            "Loaded tailMulGF61FftRadixApple",
            "Loaded tailMulGF61FftTwiddleApple",
            "Loaded tailMulGF61FftShuffleApple",
            "Loaded tailMulGF61FftFinalApple",
            "Loaded tailMulGF61ReverseStockApple",
            "Loaded tailMulGF61PairStockApple",
        ])

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, text=True)

    deadline = time.monotonic() + timeout_s
    text = ""
    failure = None
    try:
        while time.monotonic() < deadline:
            time.sleep(1)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if "-cl-std=CL2.0" in text:
                raise RuntimeError(f"{label}: Apple build still selected OpenCL C 2.0")
            lowered = text.lower()
            if ("invalid_kernel" in lowered or "error: aevum" in lowered or
                    "gpu read failed" in lowered or "gpu double-read mismatch" in lowered or
                    "persistent double-read errors" in lowered):
                raise RuntimeError(f"{label}: runtime failure detected in log")

            iterations = [int(value) for value in re.findall(r"Iter:\s*(\d+)", text)]
            max_iter = max(iterations, default=0)
            markers_ready = all(marker in text for marker in required_markers)
            if max_iter >= 1 and markers_ready:
                break

            rc = proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"{label}: process exited before Iter >= 1 and required kernels, rc={rc}")
        else:
            missing = [marker for marker in required_markers if marker not in text]
            iterations = [int(value) for value in re.findall(r"Iter:\s*(\d+)", text)]
            raise RuntimeError(
                f"{label}: timeout before real iteration; max_iter={max(iterations, default=0)}, "
                f"missing={missing}")
    except Exception as exc:
        failure = exc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"===== {label} log: {log_path} =====")
    print(text)
    print(f"===== end {label} log =====")
    if failure is not None:
        raise failure
    if "-cl-std=CL1.2" not in text:
        raise RuntimeError(f"{label}: log did not confirm OpenCL C 1.2")
    if "[Backend Aevum] engine::Reg adapter active" not in text:
        raise RuntimeError(f"{label}: Aevum adapter was not activated")
    if force_auto_ratio and "[Backend Auto] LL: Aevum selected" not in text:
        raise RuntimeError(f"{label}: LL auto-selection did not choose Aevum")
    iterations = [int(value) for value in re.findall(r"Iter:\s*(\d+)", text)]
    if max(iterations, default=0) < 1:
        raise RuntimeError(f"{label}: no completed LL iteration was observed")
    for marker in required_markers:
        if marker not in text:
            raise RuntimeError(f"{label}: required marker missing: {marker}")
    lowered = text.lower()
    if ("zsh: abort" in lowered or "abort trap" in lowered or
            "core dumped" in lowered or "assertion failed" in lowered):
        raise RuntimeError(f"{label}: abort detected")
    print(f"{label}: Aevum macOS smoke passed at Iter >= 1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    binary = root / "prmers"
    library = root / "third_party/aevum/build-engine/libaevum_engine.so"
    if not binary.is_file() or not library.is_file():
        parser.error("prmers or libaevum_engine.so is missing")

    shutil.rmtree(root / ".aevum-kernel-cache", ignore_errors=True)
    run_case(root, binary, library, args.device, "-llunsafe", "-aevum", args.timeout)
    run_case(root, binary, library, args.device, "-ll", "-aevum", args.timeout)
    run_case(root, binary, library, args.device, "-ll", "-aevum-auto", args.timeout,
             force_auto_ratio=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Aevum macOS LL smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
