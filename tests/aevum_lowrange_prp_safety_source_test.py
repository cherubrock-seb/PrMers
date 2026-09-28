from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
policy = (ROOT / "src/aevum/AutoPolicy.cpp").read_text(encoding="utf-8")
gpu = (ROOT / "src/marin/gpu.cpp").read_text(encoding="utf-8")

# AUTO: ordinary 8-register PRP <=512K must fall back to Marin.
assert "kOrdinaryPrpQuarantineWords = 524288u" in policy
assert "workload == engine::gpu_workload::prp" in policy
assert "register_count == 8u" in policy
assert "result.aevum_transform <= kOrdinaryPrpQuarantineWords" in policy
assert "safety=Marin-only" in policy

# Forced Aevum: do not allow the same unsafe transform family to emit a result.
# A larger explicit -aevum-fft remains possible because the guard is based on
# the resolved engine size, not the exponent alone.
assert "selected_workload == gpu_workload::prp" in gpu
assert "reg_count == 8u" in gpu
assert "created->get_size() <= kOrdinaryPrpQuarantineWords" in gpu
assert "Aevum ordinary PRP <=512K is temporarily quarantined" in gpu
assert "larger explicit -aevum-fft plan for diagnostic testing" in gpu

print("Aevum low-range ordinary PRP safety source regression: PASS")
