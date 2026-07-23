#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import re
import signal
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(case_dir: Path) -> str:
    candidates = sorted(case_dir.glob("*.p95")) + sorted(case_dir.glob("*.save"))
    if not candidates:
        return ""
    # Prime95-format files are preferred because they are normally stable and
    # contain the exact canonical state used for Stage 2 handoff.
    chosen = next((p for p in candidates if p.suffix == ".p95"), candidates[0])
    return sha256_file(chosen)


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
                 spec: str, seconds: int, root: Path, cache: Path) -> tuple[bool, str, str, str]:
    case = root / "actual" / workload / slug(spec)
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
                  root: Path, cache: Path, timeout: int) -> tuple[bool, str, str, str, str]:
    case = root / "actual" / workload / slug(spec)
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
    state = state_hash(case)
    ok = rc in {0, 1} and not fatal and bool(state)
    return ok, metric, resolved, state, fatal


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
    ns = parser.parse_args()

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
            row.status = "PASS" if ok else "FAIL"
            row.note = fatal or ("word-exact against Type1" if ok else "differential failed")
            exact_by_key[(workload, candidate)] = ok
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
            ok, metric, resolved, fatal = actual_smoke(
                binary, ns.device, ns.large_exponent, workload, candidate,
                max(20, ns.seconds if ns.profile != "quick" else 25), report,
                cache_root / slug(candidate))
            row.resolved = resolved
            row.transform = ""
            row.exact = "yes"
            row.actual_metric = metric
            row.status = "PASS" if ok else "FAIL"
            row.note = fatal
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
            ok, metric, resolved, state, fatal = actual_finite(
                binary, ns.device, workload, candidate, report,
                cache_root / slug(candidate), timeout)
            row.resolved = resolved
            row.exact = "yes"
            row.actual_metric = metric
            row.state_sha256 = state
            row.status = "PASS" if ok else "FAIL"
            row.note = fatal
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
                row.note = "actual state hash differs from Type1 baseline"

    summary = report / "summary.tsv"
    write_summary(rows, summary)

    print("\nRecommended plans (only exact PASS rows are eligible):", flush=True)
    recommendations: list[str] = []
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
        print("  " + line, flush=True)

    (report / "recommendations.txt").write_text("\n".join(recommendations) + "\n")
    failures = sum(1 for row in rows if row.status == "FAIL")
    print(f"Summary: {summary}", flush=True)
    print(f"Failures: {failures}", flush=True)
    print("Note: PrMers P-1 Stage 2 V-trace/classic BSGS use Marin; no Aevum FFT plan is selected there.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
