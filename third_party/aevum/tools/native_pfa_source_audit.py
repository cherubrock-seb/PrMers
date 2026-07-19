#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
required={
 'src/FFTConfig.h':['pfa_radix','isPfa'],
 'src/FFTConfig.cpp':['pfa:auto','pfa3:','pfa9:'],
 'src/Gpu.cpp':['PFA_BINARY_LENGTH','PFA_LOGICAL_STEP','PFA_LOG2_ROOT_TWO31','PFA_LOG2_ROOT_TWO61','Aevum native PFA active','former\n  // pfaUnpack'],
 'src/cl/base.cl':['M31_LOG2_ROOT_TWO','M61_LOG2_ROOT_TWO'],
 'src/cl/fftp.cl':['#if !PFA_RADIX','pfaLogicalIndex','pfaLoadCanonicalWord'],
 'src/cl/fft-middle.cl':['pfaForwardMiddle','pfaInverseMiddle','pfaMulScalar(sub(a1, a2), w)'],
 'src/cl/fftw.cl':['pfaWLogicalIndex','pfaWCanonicalPairIndex','outScalar31','outScalar61','PFA_LOGICAL_STEP'],
 'src/cl/pfaunpack.cl':['pfaUnpack','pfaSourcePairIndex','const Z31 even31','out31[dst] = U2(odd31, even31)'],
}
for rel,needles in required.items():
    text=(root/rel).read_text(errors='strict')
    for needle in needles:
        assert needle in text, f'{needle!r} missing from {rel}'

for rel in ('src/cl/fftp.cl','src/cl/carry.cl','src/cl/carryfused.cl'):
    text=(root/rel).read_text(errors='strict')
    assert '((1ULL << 30) / NWORDS)' not in text, f'inline power-of-two-only M31 weighting remains in {rel}'
    assert '((1ULL << 60) / NWORDS)' not in text, f'inline power-of-two-only M61 weighting remains in {rel}'

for p in root.rglob('*'):
    if p.is_file() and '.git' not in p.parts and p.stat().st_size<8_000_000:
        text=p.read_text(errors='ignore').lower()
        forbidden='prmers_'+'opencl_'+'prp'
        assert forbidden not in text, f'forbidden standalone runner reference in {p}'

gpu_cpp=(root/'src/Gpu.cpp').read_text(errors='strict')
gpu_h=(root/'src/Gpu.h').read_text(errors='strict')
assert 'kPfaUnpack' not in gpu_cpp and 'kPfaUnpack' not in gpu_h, 'obsolete pfaUnpack runtime kernel remains'
assert 'const Z31 w2 = mul(w, w)' not in (root/'src/cl/fft-middle.cl').read_text(), 'old four-multiply DFT3 remains'

bundle=(root/'src/bundle.cpp').read_text(errors='ignore')
assert 'pfaunpack.cl' in bundle and 'pfaUnpack' in bundle
print('Native PFA source audit passed: fast DFT3, fused fftW scatter, no runtime pfaUnpack')

fftconfig=(root/'src/FFTConfig.cpp').read_text(errors='strict')
assert 'Native PFA is initially enabled' not in fftconfig, 'macOS PFA gate remains'
