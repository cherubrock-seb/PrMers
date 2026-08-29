#!/usr/bin/env python3
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
needles = {
    "src/cl/carryutil.cl": ["pfaResidentScalarIndex", "residentPair * 2u + wordHalf"],
    "src/cl/fftp.cl": ["#if PFA_RESIDENT", "residentPair = in[g * WIDTH + G_W * i + me]"],
    "src/cl/carry.cl": ["outResident[pfaResidentScalarIndex(logical0)]", "Word2 result = weightAndCarryPair"],
    "src/cl/carryb.cl": ["P(Word) ioResident", "pfaResidentScalarIndex(logical0 + 1u)"],
    "src/Gpu.h": ["kfftPResident", "kCarryAResident", "pfa_resident_enabled"],
    "src/Gpu.cpp": ["AEVUM_PFA_RESIDENT", "fftPResident(buf1, in)", "carryAResident(out, buf3)", "carryBResident(out)"],
    "src/EngineApi.cpp": ["pfa_resident_requested", "PFA9 resident-word chain enabled"],
}
for rel, pats in needles.items():
    text = (root / rel).read_text()
    for pat in pats:
        assert pat in text, f"missing {pat!r} in {rel}"

# CPU proof-of-layout sanity for the exact 4.50M PFA9 plan.
R=9; L=524288; H=512; W=512; LINV=5; N=R*L

def logical(row,binary):
    delta=(row + R - binary % R) % R
    return binary + L * ((delta * LINV) % R)

def resident(n):
    binary=n % L; row=n % R; half=binary & 1; pair=binary >> 1
    q=pair // H; y=pair - q*H
    return ((row*H+y)*W+q)*2+half

# Validate exactly the coordinates used by the kernel for deterministic/random lanes.
rng=random.Random(0x32316109)
for _ in range(100000):
    row=rng.randrange(R); y=rng.randrange(H); me=rng.randrange(64); i=rng.randrange(8)
    binary_pair=me*H+y + i*(L//(2*8))
    for half in (0,1):
        b=2*binary_pair+half
        n=logical(row,b)
        expected=((row*H+y)*W+(me+i*64))*2+half
        assert resident(n)==expected, (row,y,me,i,half,n,resident(n),expected)
print("PFA9 FFT323161 PFA-RESIDENT SOURCE/LAYOUT AUDIT PASSED")
