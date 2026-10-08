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


def run(work, interrupt_at_stage2):
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
        if interrupt_at_stage2 and not sent and re.search(rb'Stage1 elapsed=', out):
            proc.send_signal(signal.SIGINT)
            sent = True
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return out.decode(errors='replace'), sent


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    if (root / 'kernels').exists():
        (work / 'kernels').symlink_to(root / 'kernels')
    first, sent = run(work, True)
    if 'Stage1 start' not in first:
        print('SKIP: ECM did not start (no usable OpenCL device?)')
        print(first[-400:])
        sys.exit(0)
    assert sent, 'stage 2 never started in the first run'
    assert 'Interrupted at Stage2' in first or 'checkpoint saved inside current Stage2 chunk' in first, \
        'the first run finished stage 2 before the interrupt was seen; raise B2'
    assert list(work.glob('ecm2_m_1693_c0.ckpt')), 'no stage-2 checkpoint was written'
    second, _ = run(work, False)
    assert 'Resuming Stage2' in second, 'second run did not resume stage 2'
    assert 'Stage1 start' not in second, 'second run redid stage 1'
    assert re.search(r'factor=10159', second), 'resumed stage 2 lost the factor'
print('PrMers ECM Montgomery stage-2 resume test passed')
