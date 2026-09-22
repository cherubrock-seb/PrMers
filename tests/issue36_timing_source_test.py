#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
h=(root/'third_party/aevum/src/EngineApi.h').read_text()
c=(root/'third_party/aevum/src/EngineApi.cpp').read_text()
bench=(root/'scripts/aevum_issue36_timing.cpp').read_text()
sh=(root/'scripts/bench_issue36_timing_decomposition.sh').read_text()
for needle in ('aevum_engine_timing_stats','aevum_engine_timing_reset','aevum_engine_timing_get'):
    assert needle in h, needle
    assert needle in c, needle
assert 'timing_enabled_ = envExactly("AEVUM_TIMING_DECOMP", "1")' in c
for needle in ('pending_flush_ns','queue_sync_ns','copy_ns','equal_ns','readback_ns'):
    assert needle in c and needle in bench, needle
assert 'exact_timing_on_vs_off' in bench
assert 'control_long_us_per_iter' in bench
assert 'estimated_amortized_ns_per_prp_iteration' in bench
assert '1000.0*600.0' in bench  # matches current RunPrpOrLlMarin checkpasslevel_auto
assert 'proof_single_register_proxy_wall_ns' in bench
for label in ('A_raw_square_hot_path','B_pending_flush','C_gerbicz','D_export','E_queue_waits','F_progress_io_probe','G_short_vs_long'):
    assert label in bench, label
assert '1:512:8:512:202' in sh
assert 'INPLACE=1,LOADS=10040,MODM31=2,STORES=22' in sh
assert 'env -u AEVUM_TUNE_DIR' in sh
assert 'gpu-clock-thermal.csv' in sh
assert 'AEVUM_NVIDIA_SMI_ID' in sh
print('issue36 timing instrumentation source contract: PASS')
