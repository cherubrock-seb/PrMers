#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cpp=(root/'src/io/CliParser.cpp').read_text()
hpp=(root/'include/io/CliParser.hpp').read_text()
assert '"-pfa"' in cpp and '"pfa:auto"' in cpp
assert '"pfa:3"' in cpp and '"pfa:9"' in cpp
assert 'aevum_pfa_radix' in hpp
assert '"-pfa9-type4"' in cpp
assert '"pfa9full:4:512:9:512:202"' in cpp
assert '"-pfa9-type4-full"' in cpp
assert '"pfa9:4:512:9:512:202"' in cpp
adapter=(root/'src/aevum/EngineAevum.cpp').read_text()
assert 'fields[offset] == "1" || fields[offset] == "4"' in adapter
assert 'Aevum FFT323161 requires explicit pfa9, pfa9fast, or pfa9full plan' in adapter
for p in root.rglob('*'):
    if p.is_file() and p.name != 'MANIFEST_NATIVE_PFA.json' and 'third_party' not in p.parts and 'docs' not in p.parts and '__pycache__' not in p.parts and p.stat().st_size<8_000_000:
        forbidden='prmers_'+'opencl_'+'prp'
        assert forbidden not in p.read_text(errors='ignore'), f'old standalone runner reference: {p}'
print('PrMers native PFA CLI source test passed')

app=(root/'src/core/App.cpp').read_text()
assert '-pfa-off' in cpp
assert 'aevum_pfa_off' in hpp
assert 'o.aevum_fft_spec = "pfa:auto"' in app

policy=(root/'src/aevum/AutoPolicy.cpp').read_text()
gpu=(root/'src/marin/gpu.cpp').read_text()
assert 'aevum_engine_resolve_fft(exponent, fft_spec' in policy
assert 'aevum_auto_decide(p, reg_count, selected_workload, fft_spec)' in gpu
