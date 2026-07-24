#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

FATAL = [
    r"INVALID_KERNEL", r"COMPILE_PROGRAM_FAILURE", r"Segmentation fault",
    r"Abort trap", r"double free", r"\[Gerbicz Li\]\s*Mismatch",
    r"\[ECM\]\s*invariant FAIL", r"ERROR:",
]


@dataclass
class Row:
    workload: str
    exponent: int
    requested: str
    resolved: str = ""
    family: str = ""
    transform: str = ""
    exact: str = "n/a"
    synthetic_rate: str = ""
    actual_metric: str = ""
    state_sha256: str = ""
    status: str = "PENDING"
    note: str = ""


def qcmd(cmd: Sequence[str]) -> str:
    return " ".join(repr(str(x)) for x in cmd)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_logged(cmd: Sequence[str], cwd: Path, log: Path, timeout: int,
               env: dict[str, str] | None = None) -> tuple[int, float, str]:
    cwd.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    start = time.monotonic()
    with log.open("w", encoding="utf-8", errors="replace") as out:
        out.write("COMMAND: " + qcmd(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen([str(x) for x in cmd], cwd=cwd, stdout=out,
                                stderr=subprocess.STDOUT, env=merged,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            rc = 124
    elapsed = time.monotonic() - start
    text = log.read_text(encoding="utf-8", errors="replace")
    print(f"[RUN] rc={rc} elapsed={elapsed:.2f}s log={log.name} :: {qcmd(cmd)}", flush=True)
    return rc, elapsed, text


def last_match(text: str, patterns: Iterable[str], default: str = "") -> str:
    found: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I | re.M):
            found.append((match.start(), match.group(1)))
    return max(found, default=(-1, default))[1]


def has_fatal(text: str) -> str:
    return next((p for p in FATAL if re.search(p, text, re.I | re.M)), "")


def family(spec: str) -> str:
    if spec.startswith("4:"):
        return "Type4 FFT323161"
    if spec.startswith("pfa3:"):
        return "PFA3"
    if spec.startswith("pfa9:") or spec.startswith("pfa9full:"):
        return "PFA9"
    return "Type1 FFT3161"


def normalize_scalar(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("0x"):
        digits = value[2:].lower().lstrip("0") or "0"
        return "0x" + digits
    if re.fullmatch(r"[+-]?[0-9]+", value):
        try:
            return str(int(value))
        except ValueError:
            return value
    return value


def parse_resume_record(line: str) -> str:
    fields: dict[str, str] = {}
    for part in line.strip().split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        normalized_key = key.strip().upper()
        normalized_value = normalize_scalar(value)
        if normalized_key in {"X", "X0", "Y", "Y0"} and not normalized_value.startswith("0x"):
            try:
                normalized_value = "0x" + format(int(normalized_value), "x")
            except ValueError:
                pass
        fields[normalized_key] = normalized_value

    method = fields.get("METHOD", "")
    if not method:
        return ""

    preferred = {
        "P-1": ("METHOD", "B1", "N", "X", "X0", "Y", "Y0", "CHECKSUM"),
        "ECM": ("METHOD", "SIGMA", "A", "B1", "N", "X", "X0", "Y0", "CHECKSUM"),
    }
    keys = preferred.get(method, tuple(sorted(
        key for key in fields if key not in {"PROGRAM", "TIME", "WHO"})))
    return ";".join(f"{key}={fields[key]}" for key in keys if key in fields)


def canonical_state(case_dir: Path) -> tuple[str, str]:
    records: set[str] = set()
    candidates = sorted(case_dir.glob("*.p95"))
    candidates += sorted(case_dir.glob("*.save"))
    for path in candidates:
        try:
            data = path.read_bytes()
            if b"\0" in data[:4096]:
                continue
            text = data.decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            record = parse_resume_record(line)
            if record:
                records.add(record)

    canonical = "\n".join(sorted(records))
    if not canonical:
        return "", ""
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (case_dir / "canonical-state.txt").write_text(canonical + "\n", encoding="utf-8")
    return digest, canonical


def make_cache_link(case_dir: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    link = case_dir / ".aevum-kernel-cache"
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
    if not link.exists():
        link.symlink_to(cache, target_is_directory=True)


def discover_plans(binary: Path, device: int, exponent: int, base: Path) -> list[str]:
    case = base / f"discover-{exponent}"
    case.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), str(exponent), "-prp", "-proof", "0", "-aevum",
           "-aevum-fft", "throughput:auto", "-d", str(device), "--noask",
           "-f", str(case)]
    _, _, text = run_logged(cmd, case, case / "discover.log", 12)
    specs: list[str] = []
    patterns = [
        r"Aevum auto FFT:\s*([^\s]+)\s+for exponent",
        r"Aevum throughput auto:\s*([^\s]+)\s+selected",
        r"Aevum throughput candidate:\s*([^,\s]+)",
        r"Aevum native PFA:\s*([^\s]+)\s+selected",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            spec = match.group(1).strip()
            if spec and spec not in specs:
                specs.append(spec)
    if not specs:
        raise RuntimeError(f"No Aevum plan was discovered for exponent {exponent}; see {case / 'discover.log'}")
    return specs


def parse_plan_line(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("workload", "requested", "resolved", "transform", "iterations", "seconds", "iter_per_s", "hash"):
        match = re.search(rf"(?:^|\s){key}=([^\s]+)", line)
        if match:
            result[key] = match.group(1)
    return result


def resolve_with_library(lib: Path, exponent: int, spec: str) -> str:
    library = ctypes.CDLL(str(lib))
    resolve = library.aevum_engine_resolve_fft
    resolve.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    resolve.restype = ctypes.c_int
    last_error = library.aevum_engine_last_error
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    buffer = ctypes.create_string_buffer(256)
    rc = resolve(exponent, spec.encode("utf-8"), buffer, len(buffer))
    if not rc:
        detail = last_error()
        raise RuntimeError(
            f"cannot resolve {spec} for exponent {exponent}: "
            f"{detail.decode(errors='replace') if detail else 'unknown error'}")
    return buffer.value.decode("utf-8")


def synthetic_compare(tool: Path, lib: Path, device: int, exponent: int,
                      workload: str, stock: str, candidate: str,
                      iterations: int, root: Path, cache: Path) -> tuple[bool, dict[str, str], str]:
    case = root / "synthetic" / workload / slug(candidate)
    case.mkdir(parents=True, exist_ok=True)
    make_cache_link(case, cache)
    cmd = [str(tool), str(lib), str(device), str(exponent), workload,
           stock, candidate, str(iterations), str(lib.parent.parent)]
    rc, _, text = run_logged(cmd, case, case / "run.log", 900)
    lines = [parse_plan_line(line) for line in text.splitlines() if line.startswith("PLAN ")]
    candidate_line = lines[-1] if lines else {}
    ok = rc == 0 and "AEVUM WORKLOAD PLAN DIFFERENTIAL PASSED" in text and not has_fatal(text)
    return ok, candidate_line, has_fatal(text)


def parse_ips(text: str) -> float | None:
    values: list[float] = []
    for pattern in (r"IPS:\s*([0-9]+(?:\.[0-9]+)?)", r"it/s\s+([0-9]+(?:\.[0-9]+)?)"):
        values.extend(float(m.group(1)) for m in re.finditer(pattern, text, re.I))
    values = [v for v in values if v > 0]
    return values[-1] if values else None


def actual_smoke(binary: Path, device: int, exponent: int, workload: str,
                 spec: str, seconds: int, root: Path, cache: Path,
                 repeat: int = 0) -> tuple[bool, str, str, str]:
    case = root / "actual" / workload / slug(spec) / f"repeat-{repeat + 1}"
    case.mkdir(parents=True, exist_ok=True)
    make_cache_link(case, cache)
    mode_args = ["-prp", "-proof", "0"] if workload == "prp" else ["-ll"]
    cmd = [str(binary), str(exponent), *mode_args, "-aevum", "-aevum-fft", spec,
           "-d", str(device), "--noask", "-f", str(case)]
    rc, _, text = run_logged(cmd, case, case / "run.log", seconds)
    fatal = has_fatal(text)
    ips = parse_ips(text)
    resolved = last_match(text, [r"FFT:\s*[^\n]*?\s([0-9]+:[^\s]+)\s*\(",
                                 r"Aevum throughput auto:\s*([^\s]+)\s+selected"], spec)
    ok = rc in {0, 1, 124, 130} and not fatal and ips is not None
    return ok, f"{ips:.3f} IPS" if ips is not None else "", resolved, fatal


def actual_finite(binary: Path, device: int, workload: str, spec: str,
                  root: Path, cache: Path, timeout: int,
                  repeat: int = 0) -> tuple[bool, str, str, str, str, str]:
    case = root / "actual" / workload / slug(spec) / f"repeat-{repeat + 1}"
    case.mkdir(parents=True, exist_ok=True)
    make_cache_link(case, cache)
    if workload == "pm1":
        exponent = 55050557
        args = [str(exponent), "-pm1", "-b1", "191"]
    elif workload == "ecm":
        exponent = 55050557
        args = [str(exponent), "-ecm", "-b1", "50", "-K", "1",
                "-edwards", "-notorsion", "-seed", "123456789"]
    elif workload == "ecm-stage2":
        exponent = 55050557
        args = [str(exponent), "-ecm", "-b1", "50", "-b2", "1000", "-K", "1",
                "-edwards", "-notorsion", "-seed", "123456789"]
    else:
        raise ValueError(workload)
    cmd = [str(binary), *args, "-aevum", "-aevum-fft", spec,
           "-d", str(device), "--noask", "-f", str(case)]
    rc, wall, text = run_logged(cmd, case, case / "run.log", timeout)
    fatal = has_fatal(text)
    internal = last_match(text, [
        r"Elapsed time =\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        r"Stage1 elapsed=([0-9]+(?:\.[0-9]+)?)\s*s",
        r"Elapsed time \(stage 2[^\)]*\) =\s*([0-9]+(?:\.[0-9]+)?)\s*s",
    ])
    metric = f"{float(internal):.3f}s internal" if internal else f"{wall:.3f}s wall"
    resolved = last_match(text, [r"FFT:\s*[^\n]*?\s([0-9]+:[^\s]+)\s*\(",
                                 r"Aevum throughput auto:\s*([^\s]+)\s+selected"], spec)
    state, canonical = canonical_state(case)
    ok = rc in {0, 1} and not fatal and bool(state)
    return ok, metric, resolved, state, canonical, fatal


def metric_number(value: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    return float(match.group(1)) if match else None


def write_summary(rows: list[Row], output: Path) -> None:
    fields = list(asdict(rows[0]).keys()) if rows else [f.name for f in Row.__dataclass_fields__.values()]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Word-exact Aevum workload plan audit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", type=int, default=int(os.environ.get("PRMERS_DEVICE", "1")))
    parser.add_argument("--profile", choices=("quick", "standard", "full"),
                        default=os.environ.get("PRMERS_PLAN_AUDIT_PROFILE", "standard"))
    parser.add_argument("--seconds", type=int, default=int(os.environ.get("PRMERS_PLAN_AUDIT_SECONDS", "45")))
    parser.add_argument("--large-exponent", type=int, default=175000039)
    parser.add_argument("--factoring-exponent", type=int, default=55050557)
    parser.add_argument("--plans", help="comma-separated plan list used for both exponent groups; prefer the workload-specific options")
    parser.add_argument("--large-plans", help="comma-separated explicit plans for PRP/LL at --large-exponent")
    parser.add_argument("--factoring-plans", help="comma-separated explicit plans for P-1/ECM at --factoring-exponent")
    parser.add_argument("--extra-large-exponents",
                        help="comma-separated extra PRP/LL exponents for synthetic exactness")
    parser.add_argument("--extra-factoring-exponents",
                        help="comma-separated extra P-1/ECM exponents for synthetic exactness")
    parser.add_argument("--repeats", type=int,
                        help="measurement repetitions (quick=1, standard=2, full=3)")
    parser.add_argument(
        "--strict-policy", action="store_true",
        default=os.environ.get("PRMERS_PLAN_AUDIT_STRICT_POLICY", "0") == "1",
        help="fail when a built-in workload selector does not match the measured exact winner")
    ns = parser.parse_args()
    repeats = ns.repeats if ns.repeats is not None else {
        "quick": 1, "standard": 2, "full": 3
    }[ns.profile]
    if repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    root = ns.root.resolve()
    binary = root / "prmers"
    aevum = root / "third_party" / "aevum"
    lib = aevum / "build-engine" / "libaevum_engine.so"
    tool = aevum / "build-tests" / "aevum-workload-plan-audit"
    if not binary.is_file():
        raise SystemExit(f"PrMers binary not found: {binary}")

    report = root / f"aevum-plan-audit-{time.strftime('%Y%m%d-%H%M%S')}"
    report.mkdir()
    print(f"Report: {report}", flush=True)
    rc, _, _ = run_logged(["make", "-C", str(aevum), "workload-plan-audit-build"],
                          root, report / "build-audit.log", 1800)
    if rc != 0 or not lib.is_file() or not tool.is_file():
        raise SystemExit(f"Aevum workload audit build failed; see {report / 'build-audit.log'}")

    if platform.system() == "Darwin":
        print("Apple OpenCL 1.2: Type4/PFA and Aevum ECM/P-1 are intentionally disabled. "
              "Use this audit on Linux for cross-plan comparisons.", flush=True)

    def parse_plan_list(value: str | None) -> list[str] | None:
        return [item.strip() for item in value.split(",") if item.strip()] if value else None

    def parse_exponent_list(value: str | None, defaults: list[int]) -> list[int]:
        if value is None:
            return defaults
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    default_extra_large = {
        "quick": [],
        "standard": [142606549],
        "full": [95000011, 142606549],
    }[ns.profile]
    default_extra_factoring = {
        "quick": [],
        "standard": [95000011],
        "full": [31284263, 95000011],
    }[ns.profile]
    extra_large_exponents = parse_exponent_list(
        ns.extra_large_exponents, default_extra_large)
    extra_factoring_exponents = parse_exponent_list(
        ns.extra_factoring_exponents, default_extra_factoring)

    common_plans = parse_plan_list(ns.plans)
    explicit_large = parse_plan_list(ns.large_plans) or common_plans
    explicit_factoring = parse_plan_list(ns.factoring_plans) or common_plans
    large_plans = explicit_large or discover_plans(binary, ns.device, ns.large_exponent, report)
    factoring_plans = explicit_factoring or discover_plans(binary, ns.device, ns.factoring_exponent, report)
    print(f"Large candidates: {large_plans}", flush=True)
    print(f"Factoring candidates: {factoring_plans}", flush=True)

    def stock(plans: list[str]) -> str:
        return next((p for p in plans if family(p) == "Type1 FFT3161"), plans[0])

    rows: list[Row] = []
    exact_by_key: dict[tuple[str, str], bool] = {}
    cache_root = report / "kernel-cache"

    synthetic_jobs = [
        ("prp", ns.large_exponent, large_plans, 1024 if ns.profile == "quick" else 2048),
        ("ll", ns.large_exponent, large_plans, 1024 if ns.profile == "quick" else 2048),
        ("pm1", ns.factoring_exponent, factoring_plans, 256 if ns.profile == "quick" else 768),
        ("ecm", ns.factoring_exponent, factoring_plans, 16 if ns.profile == "quick" else 48),
    ]
    for workload, exponent, plans, iterations in synthetic_jobs:
        base_plan = stock(plans)
        for candidate in plans:
            row = Row(workload=f"synthetic-{workload}", exponent=exponent,
                      requested=candidate, family=family(candidate))
            if candidate == base_plan:
                row.exact = "baseline"
                row.status = "PASS"
                exact_by_key[(workload, candidate)] = True
                rows.append(row)
                continue
            cache = cache_root / slug(candidate)
            ok, parsed, fatal = synthetic_compare(tool, lib, ns.device, exponent,
                                                  workload, base_plan, candidate,
                                                  iterations, report, cache)
            row.resolved = parsed.get("resolved", candidate)
            row.transform = parsed.get("transform", "")
            row.synthetic_rate = parsed.get("iter_per_s", "")
            row.exact = "yes" if ok else "no"
            row.status = "PASS" if ok else "REJECTED"
            row.note = fatal or ("word-exact against Type1" if ok else "word differential mismatch")
            exact_by_key[(workload, candidate)] = ok
            rows.append(row)


    # Cross-size synthetic exactness matrix. These runs are intentionally
    # shorter than the primary measurements; they verify that a workload
    # selector is not exact only at the calibration exponent.
    extra_groups = [
        (extra_large_exponents, ("prp", "ll"), 512),
        (extra_factoring_exponents, ("pm1", "ecm"), 192),
    ]
    for exponents, workloads, iterations in extra_groups:
        for exponent in exponents:
            plans = discover_plans(binary, ns.device, exponent, report)
            base_plan = stock(plans)
            print(f"Extra exactness exponent {exponent}: {plans}", flush=True)
            for workload in workloads:
                workload_iterations = max(12, iterations // 8) if workload == "ecm" else iterations
                for candidate in plans:
                    row = Row(
                        workload=f"synthetic-{workload}-extra",
                        exponent=exponent,
                        requested=candidate,
                        family=family(candidate),
                    )
                    if candidate == base_plan:
                        row.exact = "baseline"
                        row.status = "PASS"
                        rows.append(row)
                        continue
                    ok, parsed, fatal = synthetic_compare(
                        tool, lib, ns.device, exponent, workload,
                        base_plan, candidate, workload_iterations,
                        report / f"extra-{exponent}", cache_root / slug(candidate))
                    row.resolved = parsed.get("resolved", candidate)
                    row.transform = parsed.get("transform", "")
                    row.synthetic_rate = parsed.get("iter_per_s", "")
                    row.exact = "yes" if ok else "no"
                    row.status = "PASS" if ok else "REJECTED"
                    row.note = fatal or (
                        "word-exact against Type1"
                        if ok else "word differential mismatch")
                    rows.append(row)

    # PRP/LL actual hot-loop measurements.  Candidate plans that failed the
    # matching differential are not allowed to win the recommendation.
    for workload in ("prp", "ll"):
        for candidate in large_plans:
            row = Row(workload=workload, exponent=ns.large_exponent,
                      requested=candidate, family=family(candidate))
            exact = exact_by_key.get((workload, candidate), candidate == stock(large_plans))
            if not exact:
                row.exact = "no"
                row.status = "SKIP"
                row.note = "synthetic differential failed"
                rows.append(row)
                continue
            measurements: list[float] = []
            resolved_values: list[str] = []
            errors: list[str] = []
            for repeat in range(repeats):
                ok, metric, resolved, fatal = actual_smoke(
                    binary, ns.device, ns.large_exponent, workload, candidate,
                    max(20, ns.seconds if ns.profile != "quick" else 25), report,
                    cache_root / slug(candidate), repeat)
                value = metric_number(metric)
                if ok and value is not None:
                    measurements.append(value)
                    resolved_values.append(resolved)
                else:
                    errors.append(fatal or f"repeat {repeat + 1} failed")
            row.resolved = resolved_values[-1] if resolved_values else candidate
            row.transform = ""
            row.exact = "yes"
            if measurements:
                row.actual_metric = f"{statistics.median(measurements):.3f} IPS median/{len(measurements)}"
            row.status = "PASS" if len(measurements) == repeats else "FAIL"
            row.note = "; ".join(errors)
            rows.append(row)

    # Finite P-1/ECM measurements with deterministic output-state hashes.
    finite = ["pm1", "ecm"] + (["ecm-stage2"] if ns.profile == "full" else [])
    for workload in finite:
        synthetic_workload = "ecm" if workload.startswith("ecm") else "pm1"
        for candidate in factoring_plans:
            row = Row(workload=workload, exponent=ns.factoring_exponent,
                      requested=candidate, family=family(candidate))
            exact = exact_by_key.get((synthetic_workload, candidate),
                                     candidate == stock(factoring_plans))
            if not exact:
                row.exact = "no"
                row.status = "SKIP"
                row.note = "synthetic differential failed"
                rows.append(row)
                continue
            timeout = 240 if workload == "pm1" else 420
            measurements: list[float] = []
            resolved_values: list[str] = []
            states: list[str] = []
            canonicals: list[str] = []
            errors: list[str] = []
            used_internal = False
            for repeat in range(repeats):
                ok, metric, resolved, state, canonical, fatal = actual_finite(
                    binary, ns.device, workload, candidate, report,
                    cache_root / slug(candidate), timeout, repeat)
                value = metric_number(metric)
                used_internal = used_internal or "internal" in metric
                if ok and value is not None:
                    measurements.append(value)
                    resolved_values.append(resolved)
                    states.append(state)
                    canonicals.append(canonical)
                else:
                    errors.append(fatal or f"repeat {repeat + 1} failed")
            row.resolved = resolved_values[-1] if resolved_values else candidate
            row.exact = "yes"
            if measurements:
                unit = "s internal" if used_internal else "s wall"
                row.actual_metric = f"{statistics.median(measurements):.3f}{unit} median/{len(measurements)}"
            if states:
                row.state_sha256 = states[0]
            if len(set(states)) > 1 or len(set(canonicals)) > 1:
                errors.append("canonical state differs between repetitions")
            row.status = "PASS" if len(measurements) == repeats and not errors else "FAIL"
            row.note = "; ".join(errors)
            rows.append(row)

    # Enforce identical finite-mode states.  A fast plan cannot be recommended
    # when it does not produce the same Stage 1 handoff as the Type1 baseline.
    for workload in finite:
        valid = [r for r in rows if r.workload == workload and r.status == "PASS" and r.state_sha256]
        if not valid:
            continue
        baseline = next((r.state_sha256 for r in valid if r.family == "Type1 FFT3161"), valid[0].state_sha256)
        for row in valid:
            if row.state_sha256 != baseline:
                row.status = "FAIL"
                row.exact = "no"
                row.note = "canonical mathematical state differs from Type1 baseline"

    summary = report / "summary.tsv"
    write_summary(rows, summary)

    print("\nRecommended plans (only exact PASS rows are eligible):", flush=True)
    recommendations: list[str] = []
    winners: dict[str, Row] = {}
    for workload in ("prp", "ll", "pm1", "ecm", "ecm-stage2"):
        candidates = [r for r in rows if r.workload == workload and r.status == "PASS" and r.actual_metric]
        if not candidates:
            continue
        if workload in {"prp", "ll"}:
            winner = max(candidates, key=lambda r: metric_number(r.actual_metric) or -1.0)
        else:
            winner = min(candidates, key=lambda r: metric_number(r.actual_metric) or 1.0e99)
        line = (f"{workload}: {winner.requested} -> {winner.resolved or winner.requested} "
                f"[{winner.family}] {winner.actual_metric}")
        recommendations.append(line)
        winners[workload] = winner
        print("  " + line, flush=True)

    (report / "recommendations.txt").write_text("\n".join(recommendations) + "\n")

    env_names = {
        "prp": "PRMERS_AEVUM_PRP_FFT",
        "ll": "PRMERS_AEVUM_LL_FFT",
        "pm1": "PRMERS_AEVUM_PM1_FFT",
        "ecm": "PRMERS_AEVUM_ECM_FFT",
    }
    env_lines = [
        f"export {env_names[workload]}='{winner.resolved or winner.requested}'"
        for workload, winner in winners.items() if workload in env_names
    ]
    (report / "recommended-workload-plans.env").write_text(
        "\n".join(env_lines) + ("\n" if env_lines else ""), encoding="utf-8")

    selector_specs = {
        "prp": ("throughput:prp", ns.large_exponent),
        "ll": ("throughput:ll", ns.large_exponent),
        "pm1": ("throughput:pm1", ns.factoring_exponent),
        "ecm": ("throughput:ecm", ns.factoring_exponent),
    }
    policy_lines = ["workload\tselector\tresolved\tmeasured_winner\tmatch"]
    policy_mismatches = 0
    print("\nBuilt-in workload selector audit:", flush=True)
    for workload, (selector, exponent) in selector_specs.items():
        if workload not in winners:
            continue
        resolved = resolve_with_library(lib, exponent, selector)
        measured = winners[workload].resolved or winners[workload].requested
        match = resolved == measured
        policy_mismatches += 0 if match else 1
        policy_lines.append(
            f"{workload}\t{selector}\t{resolved}\t{measured}\t"
            f"{'yes' if match else 'no'}")
        print(f"  {workload}: {selector} -> {resolved}; measured={measured}; "
              f"match={'yes' if match else 'NO'}", flush=True)
    (report / "policy.tsv").write_text("\n".join(policy_lines) + "\n", encoding="utf-8")

    failures = sum(1 for row in rows if row.status == "FAIL")
    if ns.strict_policy:
        failures += policy_mismatches
    rejected = sum(1 for row in rows if row.status == "REJECTED")
    print(f"Summary: {summary}", flush=True)
    print(f"Rejected candidates: {rejected}", flush=True)
    print(f"Policy mismatches: {policy_mismatches}", flush=True)
    print(f"Failures: {failures}", flush=True)
    print("Note: PrMers P-1 Stage 2 V-trace/classic BSGS use Marin; no Aevum FFT plan is selected there.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
