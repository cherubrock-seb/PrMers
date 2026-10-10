#!/usr/bin/env python3
"""Collect Bench2 kernel resources separately from throughput timing."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator

PROFILE_SCHEMA = "bench2.profile.v1"
TIMING_SCHEMA = "bench2.v1"
AEVUM_WORKLOAD_PRP = 1

RESOURCE_RE = re.compile(
    r"AEVUM_RESOURCE name=(\S+) wg=(\d+) local_bytes=(\d+) "
    r"private_bytes=(\d+) preferred=(\d+) rc=(-?\d+),(-?\d+),(-?\d+)"
)
PROFILE_RE = re.compile(
    r"AEVUM_PROFILE name=(\S+) calls=(\d+) exec_ns=(-?\d+)"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_record(path: Path) -> dict[str, Any]:
    payload = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("JSON\t"):
            payload = json.loads(line.split("\t", 1)[1])
            break
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON row missing from Bench2 record: {path}")
    if payload.get("schema_version") != TIMING_SCHEMA:
        raise RuntimeError(f"unsupported timing schema: {payload.get('schema_version')!r}")
    if payload.get("status") != "PASS" or payload.get("backend") != "Aevum":
        raise RuntimeError("profiling requires a PASS Aevum timing record")
    plan = str(payload.get("active_plan") or "")
    if not plan or plan == "plugin-active-plan-unavailable":
        raise RuntimeError("timing record has no usable active Aevum plan")
    for key in ("exponent", "device_index", "transform_words", "register_count"):
        if key not in payload:
            raise RuntimeError(f"timing record missing required field {key}")
    return payload


def parse_native_log(text: str) -> dict[str, Any]:
    kernels: dict[str, dict[str, Any]] = {}
    for m in RESOURCE_RE.finditer(text):
        name, wg, local_b, private_b, preferred, rc_l, rc_p, rc_w = m.groups()
        row = kernels.setdefault(name, {"kernel_name": name})
        row.update({
            "workgroup_size": int(wg),
            "local_memory_bytes": int(local_b),
            "private_memory_bytes": int(private_b),
            "preferred_workgroup_multiple": int(preferred),
            "query_status": {
                "local_memory": int(rc_l),
                "private_memory": int(rc_p),
                "preferred_multiple": int(rc_w),
            },
        })
    for m in PROFILE_RE.finditer(text):
        name, calls, exec_ns = m.groups()
        row = kernels.setdefault(name, {"kernel_name": name})
        row["profile_calls"] = int(calls)
        row["profile_exec_ns"] = int(exec_ns)
    rows = [kernels[name] for name in sorted(kernels)]
    return {
        "kernels": rows,
        "resource_rows": sum("workgroup_size" in row for row in rows),
        "profile_rows": sum("profile_calls" in row for row in rows),
    }


@contextlib.contextmanager
def capture_native_output(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_out, saved_err = os.dup(1), os.dup(2)
    with path.open("wb", buffering=0) as out:
        try:
            os.dup2(out.fileno(), 1)
            os.dup2(out.fileno(), 2)
            yield
        finally:
            try:
                ctypes.CDLL(None).fflush(None)
            except Exception:
                pass
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def load_api(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.aevum_engine_version.restype = ctypes.c_char_p
    lib.aevum_engine_last_error.restype = ctypes.c_char_p
    lib.aevum_engine_create_ex.argtypes = [
        ctypes.c_uint32, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
    ]
    lib.aevum_engine_create_ex.restype = ctypes.c_void_p
    lib.aevum_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.aevum_engine_transform_size.argtypes = [ctypes.c_void_p]
    lib.aevum_engine_transform_size.restype = ctypes.c_size_t
    lib.aevum_engine_plan_spec.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.aevum_engine_plan_spec.restype = ctypes.c_int
    lib.aevum_engine_sync.argtypes = [ctypes.c_void_p]
    lib.aevum_engine_sync.restype = ctypes.c_int
    lib.aevum_engine_profile_report.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aevum_engine_profile_report.restype = ctypes.c_int
    lib.aevum_engine_set_u32.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    lib.aevum_engine_set_u32.restype = ctypes.c_int
    lib.aevum_engine_square_mul.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    lib.aevum_engine_square_mul.restype = ctypes.c_int
    return lib


def last_error(lib) -> str:
    raw = lib.aevum_engine_last_error()
    return raw.decode(errors="replace") if raw else ""


def require(lib, result: int, operation: str) -> None:
    if not result:
        message = last_error(lib)
        raise RuntimeError(f"Aevum {operation} failed" + (f": {message}" if message else ""))


def architecture_template(vendor: str) -> dict[str, Any]:
    upper = vendor.upper()
    return {
        "nvidia": {
            "state": "not_collected" if "NVIDIA" in upper else "not_applicable",
            "registers_per_thread": None,
            "spill_store_bytes": None,
            "spill_load_bytes": None,
            "stack_frame_bytes": None,
            "compiler_source": None,
        },
        "amd": {
            "state": "not_collected" if "AMD" in upper else "not_applicable",
            "vgpr_count": None,
            "sgpr_count": None,
            "private_segment_bytes": None,
            "group_segment_bytes": None,
            "code_object_source": None,
        },
    }


def collect(record: dict[str, Any], engine: Path, tune: Path, out: Path,
            iterations: int, work: Path) -> dict[str, Any]:
    if iterations < 1:
        raise RuntimeError("--iterations must be >= 1")
    exponent = int(record["exponent"])
    device = int(record["device_index"])
    registers = int(record["register_count"])
    expected_words = int(record["transform_words"])
    expected_plan = str(record["active_plan"])

    for key in list(os.environ):
        if key.startswith("AEVUM_"):
            os.environ.pop(key, None)
    os.environ["AEVUM_PROFILE_KERNELS"] = "1"
    os.environ["AEVUM_TUNE_DIR"] = str(tune.resolve())

    work.mkdir(parents=True, exist_ok=True)
    (work / ".aevum-kernel-cache").mkdir(exist_ok=True)
    native_log = out.with_suffix(".native.log")
    lib = load_api(engine)
    handle = None
    original_cwd = Path.cwd()

    with capture_native_output(native_log):
        try:
            os.chdir(work)
            handle = lib.aevum_engine_create_ex(
                exponent, registers, device, 0, expected_plan.encode(),
                str(tune.resolve()).encode(), AEVUM_WORKLOAD_PRP,
            )
            if not handle:
                raise RuntimeError(f"Aevum create_ex failed: {last_error(lib)}")
            words = int(lib.aevum_engine_transform_size(handle))
            plan_buf = ctypes.create_string_buffer(128)
            require(lib, lib.aevum_engine_plan_spec(handle, plan_buf, len(plan_buf)), "plan_spec")
            plan = plan_buf.value.decode(errors="replace")
            if words != expected_words:
                raise RuntimeError(f"profile transform mismatch: {words} != {expected_words}")
            if plan != expected_plan:
                raise RuntimeError(f"profile plan mismatch: {plan!r} != {expected_plan!r}")
            require(lib, lib.aevum_engine_set_u32(handle, 0, 3), "set_u32")
            for _ in range(iterations):
                require(lib, lib.aevum_engine_square_mul(handle, 0, 1), "square_mul")
            require(lib, lib.aevum_engine_sync(handle), "sync")
            require(lib, lib.aevum_engine_profile_report(handle, 1), "profile_report")
        finally:
            if handle:
                lib.aevum_engine_destroy(handle)
            os.chdir(original_cwd)

    parsed = parse_native_log(native_log.read_text(errors="replace"))
    if parsed["resource_rows"] == 0:
        raise RuntimeError("profile emitted no AEVUM_RESOURCE rows")
    if parsed["profile_rows"] == 0:
        raise RuntimeError("profile emitted no AEVUM_PROFILE rows")
    version_raw = lib.aevum_engine_version()
    version = version_raw.decode(errors="replace") if version_raw else "unknown"
    return {
        "schema_version": PROFILE_SCHEMA,
        "record_type": "bench2_profile",
        "timing_isolation": True,
        "timing_scope": "separate diagnostic engine; never part of bench2.v1 timing",
        "source_record": {
            "path": str(record["_record_path"]),
            "sha256": record["_record_sha256"],
            "schema_version": record["schema_version"],
            "source_sha": record.get("source_sha"),
        },
        "engine": {
            "library": str(engine.resolve()),
            "library_sha256": sha256_file(engine),
            "version": version,
        },
        "point": {
            "exponent": exponent,
            "device_index": device,
            "device_vendor": record.get("device_vendor"),
            "device_name": record.get("device_name"),
            "runtime_version": record.get("runtime_version"),
            "register_count": registers,
            "active_plan": expected_plan,
            "transform_words": expected_words,
            "profile_iterations": iterations,
        },
        "kernel_resources": parsed["kernels"],
        "resource_row_count": parsed["resource_rows"],
        "profile_row_count": parsed["profile_rows"],
        "architecture_resources": architecture_template(str(record.get("device_vendor") or "")),
        "native_log": str(native_log.resolve()),
        "kernel_cache_dir": str((work / ".aevum-kernel-cache").resolve()),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(payload, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--engine-lib", required=True, type=Path)
    parser.add_argument("--tune-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    record_path = args.record.resolve()
    engine = args.engine_lib.resolve()
    tune = args.tune_dir.resolve()
    out = args.out.resolve()
    if not record_path.is_file() or not engine.is_file() or not tune.is_dir():
        raise RuntimeError("record, engine library or tune directory is missing")

    record = read_record(record_path)
    record["_record_path"] = str(record_path)
    record["_record_sha256"] = sha256_file(record_path)
    work = args.work_dir.resolve() if args.work_dir else out.parent / (
        f"work-d{int(record['device_index'])}-p{int(record['exponent'])}"
    )
    result = collect(record, engine, tune, out, args.iterations, work)
    atomic_json(out, result)
    print(
        "BENCH2_PROFILE"
        f" schema={PROFILE_SCHEMA}"
        f" device={result['point']['device_index']}"
        f" p={result['point']['exponent']}"
        f" plan={result['point']['active_plan']}"
        f" resources={result['resource_row_count']}"
        f" profiles={result['profile_row_count']}"
        f" out={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
