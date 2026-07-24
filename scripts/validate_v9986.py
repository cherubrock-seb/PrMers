#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import platform
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

VERSION = "4.20.75-alpha-v99.86-workload-plan-policy-audit-fix"
ZIP_NAME = "PrMers-v99.86-WorkloadPlanPolicyAuditFix.zip"

FATAL_RUNTIME = [
    r"INVALID_KERNEL",
    r"COMPILE_PROGRAM_FAILURE",
    r"Segmentation fault",
    r"Abort trap",
    r"double free",
    r"\[Gerbicz Li\]\s*Mismatch",
    r"Gerbicz[^\n]*(?:FAIL|failed)",
    r"\[ECM\]\s*invariant FAIL",
]


@dataclass
class Case:
    name: str
    exponent: int
    seconds: int
    args: list[str]
    required: list[str]
    forbidden: list[str]
    allowed_rc: set[int]
    note: str = ""
    expected_factor: str = ""


def qcmd(cmd: Sequence[str]) -> str:
    return " ".join(repr(str(x)) for x in cmd)


def tail_text(path: Path, lines: int = 35) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-lines:])
    except Exception as exc:
        return f"<cannot read {path}: {exc}>"


def run_logged(cmd: Sequence[str], cwd: Path, log: Path, timeout: int | None = None,
               env: dict[str, str] | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)

    print(f"[RUN] {qcmd(cmd)}", flush=True)
    print(f"      cwd={cwd}", flush=True)
    print(f"      log={log}", flush=True)
    start = time.monotonic()
    stop_heartbeat = threading.Event()

    with log.open("w", encoding="utf-8", errors="replace") as out:
        out.write("COMMAND: " + qcmd(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd], cwd=str(cwd), stdout=out,
            stderr=subprocess.STDOUT, env=merged, start_new_session=True,
        )

        def heartbeat() -> None:
            while not stop_heartbeat.wait(10):
                elapsed = int(time.monotonic() - start)
                print(f"[WORKING] elapsed={elapsed}s pid={proc.pid} log={log}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            elapsed = int(time.monotonic() - start)
            print(f"[TIMEOUT] {elapsed}s reached; stopping pid={proc.pid}", flush=True)
            out.write(f"\n[VALIDATOR] timeout after {timeout}s; sending SIGINT\n")
            out.flush()
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                out.write("[VALIDATOR] grace period expired; sending SIGKILL\n")
                out.flush()
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            rc = 124
        finally:
            stop_heartbeat.set()
            thread.join(timeout=1)

    elapsed = time.monotonic() - start
    if not timed_out:
        print(f"[DONE] rc={rc} elapsed={elapsed:.1f}s log={log}", flush=True)
    return rc


def grep_all(text: str, patterns: Iterable[str]) -> tuple[bool, str]:
    for pattern in patterns:
        if not re.search(pattern, text, re.I | re.M):
            return False, pattern
    return True, ""


def extract(text: str, pattern: str, default: str = "") -> str:
    matches = list(re.finditer(pattern, text, re.I | re.M))
    return matches[-1].group(1).strip() if matches else default


def detect_backend(text: str) -> str:
    if re.search(r"validated only for PRP/LL stock FFT3161", text):
        return "Rejected-Apple-safety"
    if re.search(r"supports only stock FFT3161|Forced Aevum request cannot be satisfied", text):
        return "Rejected"
    if re.search(r"\[Backend Aevum\] engine::Reg adapter active", text):
        return "Aevum"
    if re.search(r"\[Backend Auto\].*: Aevum selected", text):
        return "Aevum-selected"
    if re.search(r"\[Backend macOS\] Marin selected", text):
        return "Marin-macOS-default"
    if re.search(r"\[Backend Auto\].*: Marin selected", text):
        return "Marin-selected"
    if re.search(r"\[Backend Marin\]", text):
        return "Marin"
    return "Unknown"


def detect_family(text: str) -> str:
    value = extract(text, r"family=([^,\n\)]+)")
    if value:
        return value
    if re.search(r"PFA_RADIX=[39]u|FFT:.*pfa[39]", text):
        return "PFA"
    if re.search(r"FFT_TYPE=4", text):
        return "Type4 FFT323161"
    if re.search(r"FFT_TYPE=1", text):
        return "Type1 FFT3161"
    return ""


def detect_fft(text: str) -> str:
    for pattern in (
        r"requested-plan=([^|\n]+)",
        r"FFT=([^\)\n,]+)",
        r"FFT:\s*[^\n]*?\s([0-9]+:[^\s]+)",
    ):
        value = extract(text, pattern)
        if value:
            return value
    return ""


def detect_transform(text: str) -> str:
    for pattern in (
        r"\[Backend Aevum\][^\n]*transform=([0-9]+)",
        r"Transform size=([0-9]+)",
        r"Transform Size = ([0-9]+)",
        r'"fft-length":([0-9]+)',
    ):
        value = extract(text, pattern)
        if value:
            return value
    return ""


def detect_factor(text: str) -> str:
    patterns = (
        r"P-1 factor stage [12] found:\s*([0-9]+)",
        r"\| factor=([0-9]+)",
        r'"factors":\["([0-9]+)"',
        r'"factor":"([0-9]+)"',
    )
    found: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I | re.M):
            found.append((match.start(), match.group(1)))
    return max(found, default=(-1, ""))[1]


def verify_factor(exponent: int, factor: str) -> bool:
    if not factor:
        return True
    try:
        f = int(factor)
        return f > 1 and pow(2, exponent, f) == 1
    except ValueError:
        return False


def install_deps(target: str, log: Path) -> int:
    if os.environ.get("PRMERS_INSTALL_DEPS", "0") != "1":
        log.write_text("Dependency installation skipped (set PRMERS_INSTALL_DEPS=1 to enable).\n")
        return 0
    if target == "ubuntu":
        rc = run_logged(["sudo", "apt-get", "update"], Path.cwd(), log.with_name("apt-update.log"), 600)
        if rc:
            return rc
        return run_logged([
            "sudo", "apt-get", "install", "-y", "build-essential", "python3", "unzip",
            "libgmp-dev", "libcurl4-openssl-dev", "ocl-icd-opencl-dev", "clinfo",
        ], Path.cwd(), log, 900)
    return run_logged(["brew", "install", "gmp", "curl"], Path.cwd(), log, 900)


def build_command(target: str) -> tuple[list[str], dict[str, str]]:
    jobs = str(os.cpu_count() or 4)
    if target == "macos":
        gmp = subprocess.check_output(["brew", "--prefix", "gmp"], text=True).strip()
        return ([
            "make", f"-j{jobs}", "CXX=c++",
            f"CXXFLAGS=-I{gmp}/include -std=c++20 -O3 -Wall -DHAS_CURL=1",
            f"LDFLAGS=-L{gmp}/lib -lgmpxx -lgmp -lcurl",
            "KERNEL_PATH=./kernels/", "MACOSX_DEPLOYMENT_TARGET=12.0",
        ], {"MACOSX_DEPLOYMENT_TARGET": "12.0"})
    return ([
        "make", f"-j{jobs}", "CXX=g++",
        "CXXFLAGS=-std=c++20 -O3 -Wall -DHAS_CURL=1",
        "LDFLAGS=-lOpenCL -ldl -lgmpxx -lgmp -lcurl",
        "KERNEL_PATH=./kernels/",
    ], {})


def common_cases(target: str, profile: str) -> list[Case]:
    smoke = {0, 1, 124, 130}
    complete = {0, 1}
    fatal = list(FATAL_RUNTIME)
    cases: list[Case] = [
        Case("prp-small-complete", 216091, 180,
             ["216091", "-prp", "-proof", "0"],
             [r"probably prime|\"status\":\"P\""], fatal, complete,
             "Complete PRP reference result"),
        Case("ll-small-complete", 216091, 180,
             ["216091", "-ll"], [r"is prime"], fatal, complete,
             "Complete Lucas-Lehmer reference result"),
    ]

    if target == "ubuntu":
        cases += [
            Case("prp-large-auto-workload-plan", 175000039, 50,
                 ["175000039", "-prp", "-proof", "0"],
                 [r"\[Backend Auto\] PRP: Aevum selected",
                  r"Aevum throughput auto:\s*4:512:8:512:202\s+selected",
                  r"selector prp", r"Progress:"],
                 fatal, smoke, "PRP workload selector/progress smoke; timeout is expected"),
            Case("ll-large-auto-workload-plan", 175000039, 50,
                 ["175000039", "-ll"],
                 [r"\[Backend Auto\] LL: Aevum selected",
                  r"Aevum throughput auto:\s*4:1K:2:1K:202\s+selected",
                  r"selector ll", r"Progress:"],
                 fatal, smoke, "LL workload selector/progress smoke; timeout is expected"),
            Case("ecm-auto-workload-plan", 55050557, 150,
                 ["55050557", "-ecm", "-b1", "50", "-K", "1", "-seed", "123456789", "-ecm_progress_ms", "500"],
                 [r"\[Backend Auto\] ECM: Aevum selected",
                  r"Aevum throughput auto:\s*1:1K:4:256:101\s+selected",
                  r"selector ecm", r"family=Type1 FFT3161", r"Stage1"],
                 fatal, complete, "ECM remains on measured-safe Type1"),
            Case("pm1-auto-workload-plan", 55050557, 120,
                 ["55050557", "-pm1", "-b1", "191"],
                 [r"\[Backend Auto\] P-1: Aevum selected",
                  r"Aevum throughput auto:\s*4:256:16:256:202\s+selected",
                  r"selector pm1", r"family=Type4 FFT323161",
                  r"Gerbicz Li.*passed|No P-1|P-1 factor"],
                 fatal, complete, "P-1 Stage 1 uses measured faster exact Type4"),
        ]
    else:
        cases += [
            Case("prp-large-auto-marin", 175000039, 35,
                 ["175000039", "-prp", "-proof", "0"],
                 [r"\[Backend macOS\] Marin selected", r"Progress:"], fatal, smoke),
            Case("ll-large-auto-marin", 175000039, 35,
                 ["175000039", "-ll"],
                 [r"\[Backend macOS\] Marin selected", r"Progress:"], fatal, smoke),
            Case("prp-forced-aevum-stock-type1", 55050557, 120,
                 ["55050557", "-prp", "-proof", "0", "-aevum"],
                 [r"PRP/LL uses staged stock Type1 FFT3161", r"\[Backend Aevum\] engine::Reg adapter active", r"FFT_TYPE=1", r"Progress:"],
                 fatal, smoke),
            Case("ll-forced-aevum-stock-type1", 5690197, 120,
                 ["5690197", "-ll", "-aevum"],
                 [r"PRP/LL uses staged stock Type1 FFT3161", r"\[Backend Aevum\] engine::Reg adapter active", r"FFT_TYPE=1", r"Progress:"],
                 fatal, smoke),
            Case("apple-type4-clean-reject", 175000039, 30,
                 ["175000039", "-aevum", "-aevum-fft", "4:512:8:512:202"],
                 [r"Apple OpenCL 1.2 supports only stock FFT3161"],
                 [r"INVALID_KERNEL", r"COMPILE_PROGRAM_FAILURE"], {1, 2}),
            Case("ecm-auto-marin", 55050557, 100,
                 ["55050557", "-ecm", "-b1", "50", "-K", "1", "-seed", "123456789"],
                 [r"\[Backend macOS\] Marin selected", r"Stage1"], fatal, complete),
            Case("pm1-auto-marin", 55050557, 100,
                 ["55050557", "-pm1", "-b1", "191"],
                 [r"\[Backend macOS\] Marin selected", r"No P-1|P-1 factor"], fatal, complete),
            Case("ecm-forced-aevum-clean-reject", 55050557, 45,
                 ["55050557", "-ecm", "-b1", "5000", "-K", "1", "-aevum"],
                 [r"validated only for PRP/LL stock FFT3161", r"mixed/prepared multiplication failed invariant/Gerbicz validation"],
                 [r"INVALID_KERNEL", r"invariant FAIL"], {1, 2},
                 "Safety rejection replaces the numerically invalid Apple ECM path"),
            Case("pm1-forced-aevum-clean-reject", 55050557, 45,
                 ["55050557", "-pm1", "-b1", "191", "-aevum"],
                 [r"validated only for PRP/LL stock FFT3161", r"mixed/prepared multiplication failed invariant/Gerbicz validation"],
                 [r"INVALID_KERNEL", r"Gerbicz Li.*Mismatch"], {1, 2},
                 "Safety rejection replaces the numerically invalid Apple P-1 path"),
        ]

    ecm_seed = "9295750182886826573"
    cases += [
        Case("ecm-stage1-stop-default", 569, 180,
             ["569", "-ecm", "-b1", "20000", "-K", "2", "-edwards", "-notorsion", "-seed", ecm_seed],
             [r"factor=879466275147241|\"879466275147241\"", r"New factor found; stopping ECM by default"],
             fatal + [r"Curve 2/2"], complete, expected_factor="879466275147241"),
        Case("ecm-stage2-factor", 569, 180,
             ["569", "-ecm", "-b1", "50", "-b2", "20000", "-K", "1", "-edwards", "-notorsion", "-seed", ecm_seed],
             [r"Stage2", r"factor=15854617|\"15854617\""], fatal, complete,
             expected_factor="15854617"),
        Case("pm1-stage1-stop-default", 55050557, 240,
             ["55050557", "-pm1", "-b1", "5000", "-b2", "10000"],
             [r"P-1 factor stage 1 found:\s*913366141713143", r"Stage 2 skipped by default"],
             fatal + [r"Start a P-1 factoring : Stage 2(?: V-trace)? Bounds"], complete,
             expected_factor="913366141713143"),
        Case("pm1-stage2-vtrace", 569, 180,
             ["569", "-pm1", "-b1", "9", "-b2", "677", "-pm1-vtrace"],
             [r"Stage 2 V-trace Bounds", r"P-1 factor stage 2 found:\s*55470673"],
             fatal, complete, expected_factor="55470673",
             note="Actual Stage 2 factor; cannot pass from a Stage 1 factor"),
        Case("pm1-stage2-classic", 569, 180,
             ["569", "-pm1", "-b1", "9", "-b2", "677", "-pm1-vtrace-off"],
             [r"Stage 2 BSGS", r"\[PM1-CLASSIC\] canonical giant/baby multiplication enabled",
              r"P-1 factor stage 2 found:\s*55470673"],
             fatal, complete, expected_factor="55470673",
             note="Direct regression for classic BSGS"),
    ]

    if profile in {"standard", "full"}:
        cases += [
            Case("ecm-stage1-continue-override", 569, 300,
                 ["569", "-ecm", "-b1", "20000", "-K", "2", "-edwards", "-notorsion", "-seed", ecm_seed,
                  "-ecm-continue-after-factor"],
                 [r"factor=879466275147241|\"879466275147241\"", r"Continuing with later curves", r"Curve 2/2"],
                 fatal, complete,
                 note="Curve 1 uses the supplied seed; curve 2 uses the deterministic derived seed"),
            Case("pm1-stage1-continue-override", 55050557, 300,
                 ["55050557", "-pm1", "-b1", "5000", "-b2", "10000", "-pm1-continue-stage2-after-factor"],
                 [r"P-1 factor stage 1 found:\s*913366141713143", r"Start a P-1 factoring : Stage 2(?: V-trace)? Bounds"],
                 fatal, complete, expected_factor="913366141713143"),
        ]

    if profile == "full":
        cases += [
            Case("prp-medium-forced-aevum", 1362763, 80,
                 ["1362763", "-prp", "-proof", "0", "-aevum"],
                 [r"\[Backend Aevum\] engine::Reg adapter active", r"Progress:"], fatal, smoke),
            Case("ll-medium-forced-aevum", 1362763, 80,
                 ["1362763", "-ll", "-aevum"],
                 [r"\[Backend Aevum\] engine::Reg adapter active", r"Progress:"], fatal, smoke),
            Case("ecm-montgomery-smoke", 1362763, 150,
                 ["1362763", "-ecm", "-b1", "100", "-b2", "1000", "-K", "1", "-montgomery", "-seed", "123456789"],
                 [r"Montgomery|Stage1"], fatal, smoke),
        ]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("ubuntu", "macos"), required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--profile", choices=("quick", "standard", "full"),
                        default=os.environ.get("PRMERS_VALIDATION_PROFILE", "standard"))
    ns = parser.parse_args()

    base = (ns.base or (Path.home() / ("mgpu" if ns.platform == "ubuntu" else "Downloads"))).expanduser().resolve()
    device = ns.device if ns.device is not None else int(os.environ.get("PRMERS_DEVICE", "1" if ns.platform == "ubuntu" else "0"))
    here = Path(__file__).resolve().parent
    zip_path = ns.zip_path
    if zip_path is None:
        zip_path = next((p for p in (here / ZIP_NAME, base / ZIP_NAME, here.parent / ZIP_NAME) if p.is_file()), None)
    if zip_path is None or not Path(zip_path).is_file():
        print(f"ERROR: {ZIP_NAME} not found beside the script or in {base}", file=sys.stderr)
        return 2
    zip_path = Path(zip_path).resolve()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    work = base / f"PrMers-v99.86-validation-{ns.platform}-{stamp}"
    report = base / f"PrMers-v99.86-report-{ns.platform}-{stamp}"
    extract_dir = work / "source"
    report.mkdir(parents=True)
    extract_dir.mkdir(parents=True)

    print(f"Candidate: {zip_path}", flush=True)
    print(f"Work: {work}", flush=True)
    print(f"Report: {report}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    roots = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        print("ERROR: source ZIP must contain exactly one top-level directory", file=sys.stderr)
        return 2
    root = roots[0]

    if install_deps(ns.platform, report / "dependencies.log") != 0:
        print("Dependency installation failed; see dependencies.log", file=sys.stderr)
        return 2

    run_logged(["make", "clean-all"], root, report / "build-clean.log", 300)
    build_cmd, build_env = build_command(ns.platform)
    build_log = report / "build.log"
    build_rc = run_logged(build_cmd, root, build_log, 1800, build_env)
    if build_rc != 0:
        print("BUILD FAILED\n" + tail_text(build_log), file=sys.stderr)
        return 2

    gates: list[tuple[str, list[str], int]] = [
        ("version", [str(root / "prmers"), "-v"], 30),
        ("stable-source", ["python3", "tests/stable_backend_stop_bsgs_apple_source_test.py"], 60),
        ("aevum-source", ["make", "test-aevum-source"], 300),
        ("aevum-host", ["make", "test-aevum-host"], 600),
        ("backend-compat", ["make", "test-backend-compat"], 300),
        ("engine-api", ["make", "-C", "third_party/aevum", "test-engine-api"], 600),
    ]
    if ns.platform == "ubuntu" and ns.profile == "full":
        gates.append((
            "workload-plan-audit-quick",
            ["python3", "scripts/audit_aevum_plans.py",
             "--profile", "quick", "--seconds", "25", "--repeats", "1",
             "--strict-policy", "--device", str(device)],
            3600,
        ))
    failures = 0
    for name, command, limit in gates:
        log = report / "gates" / f"{name}.log"
        rc = run_logged(command, root, log, limit, build_env)
        if rc:
            failures += 1
            print(f"[FAIL] gate {name} rc={rc}\n----- log tail -----\n{tail_text(log)}\n--------------------", flush=True)
        else:
            print(f"[PASS] gate {name}", flush=True)

    rows: list[dict[str, str]] = []
    for case in common_cases(ns.platform, ns.profile):
        case_dir = report / "cases" / case.name
        case_dir.mkdir(parents=True)
        command = [str(root / "prmers"), *case.args, "-d", str(device), "--noask", "-f", str(case_dir)]
        log = case_dir / "run.log"
        rc = run_logged(command, case_dir, log, case.seconds, build_env)
        text = log.read_text(encoding="utf-8", errors="replace")
        required_ok, missing = grep_all(text, case.required)
        forbidden_hit = next((p for p in case.forbidden if re.search(p, text, re.I | re.M)), "")
        factor = detect_factor(text)
        factor_ok = verify_factor(case.exponent, factor)
        expected_ok = not case.expected_factor or factor == case.expected_factor
        rc_ok = rc in case.allowed_rc

        status, reason = "PASS", ""
        if not rc_ok:
            status, reason = "FAIL", f"unexpected rc={rc}"
        elif not required_ok:
            status, reason = "FAIL", f"missing pattern: {missing}"
        elif forbidden_hit:
            status, reason = "FAIL", f"forbidden pattern: {forbidden_hit}"
        elif not factor_ok:
            status, reason = "FAIL", "reported factor does not divide M_p"
        elif not expected_ok:
            status, reason = "FAIL", f"expected factor {case.expected_factor}, got {factor or 'none'}"

        if status == "FAIL":
            failures += 1
        row = {
            "name": case.name, "status": status, "rc": str(rc),
            "backend": detect_backend(text), "family": detect_family(text),
            "fft": detect_fft(text), "transform": detect_transform(text),
            "factor": factor,
            "factor_verified": "yes" if factor and factor_ok else ("no" if factor else "n/a"),
            "seconds_limit": str(case.seconds), "reason": reason, "note": case.note,
        }
        rows.append(row)
        print(f"[{status}] {case.name}: rc={rc} backend={row['backend']} family={row['family']} "
              f"fft={row['fft']} transform={row['transform']} factor={factor or '-'} {reason}", flush=True)
        if status == "FAIL":
            print(f"----- {case.name} log tail -----\n{tail_text(log)}\n--------------------------------", flush=True)

    summary = report / "summary.tsv"
    fields = ["name", "status", "rc", "backend", "family", "fft", "transform", "factor",
              "factor_verified", "seconds_limit", "reason", "note"]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with (report / "system.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"validator_platform={ns.platform}\nprofile={ns.profile}\ndevice={device}\n")
        handle.write(f"candidate_zip={zip_path}\nversion={VERSION}\n")
        handle.write(f"python={sys.version}\nplatform={platform.platform()}\n")
        for command in (["uname", "-a"], [str(root / "prmers"), "-v"]):
            try:
                handle.write("$ " + " ".join(command) + "\n")
                handle.write(subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.STDOUT) + "\n")
            except Exception as exc:
                handle.write(f"ERROR {exc}\n")

    archive_path = report.parent / f"{report.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(report, arcname=report.name)
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = archive_path.parent / f"{archive_path.name}.sha256"
    checksum.write_text(f"{digest.hexdigest()}  {archive_path.name}\n")

    print(f"Summary: {summary}", flush=True)
    print(f"Report archive: {archive_path}", flush=True)
    print(f"Checksum: {checksum}", flush=True)
    print(f"Failures: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
