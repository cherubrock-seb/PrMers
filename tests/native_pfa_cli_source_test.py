#!/usr/bin/env python3
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
cpp=(root/'src/io/CliParser.cpp').read_text()
hpp=(root/'include/io/CliParser.hpp').read_text()
assert '"-pfa"' in cpp and '"pfa:auto"' in cpp
assert '"pfa:3"' in cpp and '"pfa:9"' in cpp
assert '"-pfa7"' in cpp and '"pfa7:1:512:7:512:202"' in cpp
assert '-pfa accepts only auto, 3, 7, or 9' in cpp
assert 'aevum_pfa_radix' in hpp and '3, 7, or 9' in hpp
assert '"-pfa9-type4"' in cpp
assert '"pfa9full:4:512:9:512:202"' in cpp
assert '"-pfa9-type4-full"' in cpp
assert '"pfa9:4:512:9:512:202"' in cpp
adapter=(root/'src/aevum/EngineAevum.cpp').read_text()
assert 'fields[0] == "pfa7"' in adapter
policy=(root/'src/aevum/AutoPolicy.cpp').read_text()
assert 'spec.rfind("pfa7:", 0)' in policy
fft=(root/'third_party/aevum/src/FFTConfig.cpp').read_text()
assert 'spec == "pfa:auto" || spec == "pfa:3" || spec == "pfa:9"' in fft
assert 'fields[offset] == "1" || fields[offset] == "4"' in adapter
assert 'Aevum FFT323161 requires explicit pfa9' not in adapter
assert '4:512:8:512:202' in (root/'README_POW2_TYPE4_LEAD_CACHE.md').read_text()
for p in root.rglob('*'):
    if p.is_file() and p.name != 'MANIFEST_NATIVE_PFA.json' and '.git' not in p.parts and 'third_party' not in p.parts and 'docs' not in p.parts and '__pycache__' not in p.parts and p.stat().st_size<8_000_000:
        forbidden='prmers_'+'opencl_'+'prp'
        assert forbidden not in p.read_text(errors='ignore'), f'old standalone runner reference: {p}'
print('PrMers native PFA CLI source test passed')

app=(root/'src/core/App.cpp').read_text()
assert '-pfa-off' in cpp
assert 'aevum_pfa_off' in hpp
assert 'fallback = ""' in app
assert app.count('? plan_override : "";') >= 2
assert '"throughput:prp"' not in app
assert '"throughput:ll"' not in app
assert '"throughput:pm1"' not in app
assert '"throughput:ecm"' not in app

policy=(root/'src/aevum/AutoPolicy.cpp').read_text()
gpu=(root/'src/marin/gpu.cpp').read_text()
assert re.search(r'aevum_engine_resolve_fft\s*\(\s*exponent\s*,\s*fft_spec\s*,', policy)
assert 'aevum_auto_decide(p, reg_count, selected_workload, fft_spec)' in gpu
assert '4:1K:8:256:101' in policy
assert 'boundary-bridge=1' in policy
assert 'decision.force_fft_spec' in gpu
assert 'runtime_fft_spec' in gpu
