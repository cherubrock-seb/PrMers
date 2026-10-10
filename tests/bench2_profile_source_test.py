#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench2_profile.py"
spec = importlib.util.spec_from_file_location("bench2_profile", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    record = root / "d2_p0210000017.record"
    payload = {
        "schema_version": "bench2.v1",
        "status": "PASS",
        "backend": "Aevum",
        "exponent": 210000017,
        "device_index": 2,
        "device_vendor": "NVIDIA Corporation",
        "device_name": "NVIDIA GeForce RTX 3080",
        "runtime_version": "OpenCL 3.0 CUDA",
        "register_count": 8,
        "active_plan": "1:1K:16:256:101",
        "transform_words": 8388608,
        "source_sha": "0123456789abcdef",
    }
    record.write_text(
        "JSON\t" + json.dumps(payload) + "\n"
        "CSV\tignored\n"
        "TEXT\tignored\n"
    )
    parsed_record = module.read_record(record)
    assert parsed_record["exponent"] == 210000017
    assert parsed_record["active_plan"] == "1:1K:16:256:101"

sample = "\n".join([
    "AEVUM_RESOURCE name=kfftP wg=256 local_bytes=8304 private_bytes=0 preferred=32 rc=0,0,0",
    "AEVUM_RESOURCE name=kCarryFused wg=512 local_bytes=16592 private_bytes=0 preferred=32 rc=0,0,0",
    "AEVUM_PROFILE name=kfftP calls=256 exec_ns=12345678",
    "AEVUM_PROFILE name=kCarryFused calls=256 exec_ns=23456789",
])
parsed = module.parse_native_log(sample)
assert parsed["resource_rows"] == 2
assert parsed["profile_rows"] == 2
assert [row["kernel_name"] for row in parsed["kernels"]] == ["kCarryFused", "kfftP"]
kfft = next(row for row in parsed["kernels"] if row["kernel_name"] == "kfftP")
assert kfft["workgroup_size"] == 256
assert kfft["local_memory_bytes"] == 8304
assert kfft["private_memory_bytes"] == 0
assert kfft["preferred_workgroup_multiple"] == 32
assert kfft["query_status"] == {
    "local_memory": 0,
    "private_memory": 0,
    "preferred_multiple": 0,
}
assert kfft["profile_calls"] == 256
assert kfft["profile_exec_ns"] == 12345678

source = SCRIPT.read_text()
assert "AEVUM_PROFILE_KERNELS" in source
assert "aevum_engine_profile_report" in source
assert "bench2.profile.v1" in source
assert "timing_isolation" in source
assert "ptxas" not in source
assert "ncu" not in source.lower()

print("bench2 profile source regression: PASS")
