#!/usr/bin/env python3
"""ECM resume lines must be self-consistent when known factors are supplied.

With known factors the residues X and A are computed modulo the cofactor, so
the N field must name the cofactor and CHECKSUM (B1 * A|SIGMA * N * X, each
reduced mod 4294967291, as in GMP-ECM) must be computed from that same N.

Runs the real binary on a small Mersenne number with one known factor
(M1693 = 10159 * cofactor).  Needs an OpenCL GPU; set PRMERS_BIN to the
binary (default ./prmers).  Skips when the binary or a device is missing.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
binary = Path(os.environ.get('PRMERS_BIN', root / 'prmers')).resolve()
if not binary.exists():
    print(f'SKIP: {binary} not found')
    sys.exit(0)

MOD = 4294967291
P = 1693
KNOWN = 10159


def eval_n(text):
    m = re.fullmatch(r'2\^(\d+)-1', text)
    return (1 << int(m.group(1))) - 1 if m else int(text, 0)


def check_file(path, expect_cofactor):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert lines, f'{path.name} is empty'
    for line in lines:
        f = dict(x.strip().split('=', 1) for x in line.split(';') if '=' in x)
        n = eval_n(f['N'])
        assert n == expect_cofactor, f'{path.name}: N field is not the cofactor'
        x = int(f['X'], 16)
        assert x < n, f'{path.name}: X is not reduced mod N'
        a = int(f['SIGMA']) if 'SIGMA' in f else int(f['A'])
        chk = int(f['B1']) * (a % MOD) * (n % MOD) * (x % MOD) % MOD
        assert chk == int(f['CHECKSUM']), f'{path.name}: CHECKSUM does not match N/X/A in the line'


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    if (root / 'kernels').exists():
        (work / 'kernels').symlink_to(root / 'kernels')
    env = dict(os.environ, AEVUM_CARRY_WMUL='1', AEVUM_AUTOTUNE='off')
    cmd = [str(binary), str(P), '-ecm', '-cmont', '-notorsion', '-b1', '650', '-b2', '0',
           '-K', '1', '-resume', '-factors', str(KNOWN)]
    res = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=120)
    saves = list(work.glob('resume_p*_ECM_B1_650.save'))
    if not saves:
        print('SKIP: no resume file written (no usable OpenCL device?)')
        print(res.stdout[-400:], res.stderr[-400:])
        sys.exit(0)
    cofactor = ((1 << P) - 1) // KNOWN
    check_file(saves[0], cofactor)
    check_file(next(work.glob('resume_p*_ECM_B1_650.p95')), cofactor)
print('PrMers ECM resume-line checksum test passed')
