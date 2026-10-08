#!/usr/bin/env python3
"""A Montgomery (Suyama) ECM run interrupted in stage 2 must resume stage 2.

M1693 with seed 1, B1=3, B2=20000 finds the factor 10159 in stage 2.  The test
interrupts that run as soon as stage 1 is done, runs it again with the same
arguments, and checks that the second run resumes the saved stage-2 state (it
does not redo stage 1) and still finds 10159.

The interrupt is sent on the end-of-stage-1 line, not on a stage-2 progress
percentage (a short stage 2 can print its first progress line only once it is
complete), and B2 is large enough that stage 2 is still running when the
signal lands.  A stage-2 checkpoint is written when the signal is seen inside
the chunk; a run so short that the whole chunk finished first would write none.

Needs an OpenCL GPU; set PRMERS_BIN to the binary (default ./prmers).  Skips
when the binary or a device is missing.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
binary = Path(os.environ.get('PRMERS_BIN', root / 'prmers')).resolve()
if not binary.exists():
    print(f'SKIP: {binary} not found')
    sys.exit(0)

ARGS = ['1693', '-ecm', '-cmont', '-notorsion', '-b1', '3', '-b2', '20000', '-K', '1', '-seed', '1']


def run(work, trigger=None):
    """Run ECM in work; send SIGINT once the regex trigger appears in the output.
    Returns (output, signal_sent, returncode)."""
    env = dict(os.environ, AEVUM_CARRY_WMUL='1', AEVUM_AUTOTUNE='off')
    proc = subprocess.Popen([str(binary)] + ARGS, cwd=work, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = b''
    sent = False
    deadline = time.time() + 300
    fd = proc.stdout.fileno()
    while time.time() < deadline:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        out += chunk
        if trigger and not sent and re.search(trigger, out):
            proc.send_signal(signal.SIGINT)
            sent = True
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return out.decode(errors='replace'), sent, proc.returncode


STAGE1_DONE = rb'Stage1 elapsed='
CKPT = 'ecm2_m_1693_c0.ckpt'


def interrupted_in_stage2(work):
    out, sent, _ = run(work, STAGE1_DONE)
    if 'Stage1 start' not in out:
        print('SKIP: ECM did not start (no usable OpenCL device?)')
        print(out[-400:])
        sys.exit(0)
    assert sent, 'stage 1 never finished in the interrupted run'
    assert 'Interrupted at Stage2' in out or 'checkpoint saved inside current Stage2 chunk' in out, \
        'the interrupted run finished stage 2 before the interrupt was seen; raise B2'
    assert (work / CKPT).exists(), 'no stage-2 checkpoint was written'
    return out


def fresh_dir(tmp, name):
    work = Path(tmp) / name
    work.mkdir()
    if (root / 'kernels').exists():
        (work / 'kernels').symlink_to(root / 'kernels')
    return work


def check_found_without_crash(out, rc, what):
    assert rc == 0, f'{what}: exit status {rc}'
    assert re.search(r'factor=10159', out), f'{what}: the factor was lost'


with tempfile.TemporaryDirectory() as tmp:
    # 1. Interrupt in stage 2, rerun: stage 2 resumes, stage 1 is not redone.
    work = fresh_dir(tmp, 'resume')
    interrupted_in_stage2(work)
    second, _, rc = run(work)
    assert 'Resuming Stage2' in second, 'second run did not resume stage 2'
    assert 'Stage1 start' not in second, 'second run redid stage 1'
    check_found_without_crash(second, rc, 'resume')

    # 2. Interrupt again right after resuming, then finish: the checkpoint written
    # by a resumed run must itself be resumable.
    work = fresh_dir(tmp, 'twice')
    interrupted_in_stage2(work)
    again, sent, _ = run(work, rb'Resuming Stage2')
    assert sent and 'Resuming Stage2' in again, 'twice: second run did not resume stage 2'
    assert (work / CKPT).exists(), 'twice: the resumed, re-interrupted run left no checkpoint'
    third, _, rc = run(work)
    assert 'Resuming Stage2' in third, 'twice: third run did not resume stage 2'
    assert 'Stage1 start' not in third, 'twice: third run redid stage 1'
    check_found_without_crash(third, rc, 'twice')

    # 3. A truncated or damaged stage-2 checkpoint must be ignored in full, not
    # crash the run, resume from a stale position, or lose the factor (the curve
    # is simply redone).
    ckpt_variants = {
        'cut-early': lambda data: data[:100],
        'cut-half': lambda data: data[:len(data) // 2],
        'garbage': lambda data: bytes((i * 131 + 7) & 0xFF for i in range(len(data))),
        'flipped-tail': lambda data: data[:-16] + bytes(b ^ 0xFF for b in data[-16:]),
    }
    base = fresh_dir(tmp, 'damaged-base')
    interrupted_in_stage2(base)
    good = (base / CKPT).read_bytes()
    for name, damage in ckpt_variants.items():
        work = fresh_dir(tmp, name)
        for f in base.iterdir():
            if f.is_file() and not f.is_symlink():
                shutil.copy(f, work / f.name)
        (work / CKPT).write_bytes(damage(good))
        out, _, rc = run(work)
        assert 'Resuming Stage2' not in out, f'{name}: a damaged checkpoint was resumed'
        check_found_without_crash(out, rc, name)
print('PrMers ECM Montgomery stage-2 resume test passed')
