#!/usr/bin/env python3
"""A Montgomery (Suyama) ECM run interrupted in stage 2 must resume stage 2.

M1693 with seed 1, B1=3, B2=2000 finds the factor 10159 in stage 2.  The test
interrupts that run mid stage 2, runs it again with the same arguments, and
checks that the second run resumes the saved stage-2 state (it does not redo
stage 1) and still finds 10159.

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

ARGS = ['1693', '-ecm', '-cmont', '-notorsion', '-b1', '3', '-b2', '2000', '-K', '1', '-seed', '1']


def run(work, interrupt_at_stage2):
    env = dict(os.environ, AEVUM_CARRY_WMUL='1', AEVUM_AUTOTUNE='off')
    proc = subprocess.Popen([str(binary)] + ARGS, cwd=work, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = b''
    sent = False
    deadline = time.time() + 100
    fd = proc.stdout.fileno()
    while time.time() < deadline:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        out += chunk
        if interrupt_at_stage2 and not sent and re.search(rb'Stage2 [1-9][0-9.]*%', out):
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
    assert list(work.glob('ecm2_m_1693_c0.ckpt')), 'no stage-2 checkpoint was written'
    second, _ = run(work, False)
    assert 'Resuming Stage2' in second, 'second run did not resume stage 2'
    assert 'Stage1 start' not in second, 'second run redid stage 1'
    assert re.search(r'factor=10159', second), 'resumed stage 2 lost the factor'
print('PrMers ECM Montgomery stage-2 resume test passed')
