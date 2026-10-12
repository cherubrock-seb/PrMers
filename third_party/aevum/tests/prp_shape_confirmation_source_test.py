from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
api = (ROOT / "src/EngineApi.cpp").read_text()

required = [
    "struct PlanProductionConfirmation",
    "confirmPrpPlanProduction",
    "PrpUseProbe native_probe",
    "PrpUseProbe candidate_probe",
    "native_probe.sequence(64)",
    "candidate_probe.sequence(64)",
    "probe.sequence(256)",
    "shape-confirm=prp-square-v1",
    "production-confirm candidate=",
    "out.wins >= 2",
    "out.worst >= 0.985",
]

for marker in required:
    assert marker in api, marker

assert api.index("class PrpUseProbe") < api.index("confirmPrpPlanProduction")
assert api.index("confirmPrpPlanProduction") < api.index("production-confirm candidate=")

print("prp_shape_confirmation_source_test: OK")
