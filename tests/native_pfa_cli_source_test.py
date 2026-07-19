#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cpp=(root/'src/io/CliParser.cpp').read_text()
hpp=(root/'include/io/CliParser.hpp').read_text()
assert '"-pfa"' in cpp and '"pfa:auto"' in cpp
assert '"pfa:3"' in cpp and '"pfa:9"' in cpp
assert 'aevum_pfa_radix' in hpp
for p in root.rglob('*'):
    if p.is_file() and 'third_party' not in p.parts and p.stat().st_size<8_000_000:
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
