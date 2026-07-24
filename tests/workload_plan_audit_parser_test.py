#!/usr/bin/env python3
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("audit_aevum_plans", ROOT / "scripts/audit_aevum_plans.py")
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

pm1_a = (
    "METHOD=P-1; B1=191; N=2^55050557-1; X=0x000abc; CHECKSUM=17; "
    "PROGRAM=PrMers old; X0=0x3; Y=0x0; Y0=0x0; WHO=alice; TIME=now;"
)
pm1_b = (
    "METHOD=P-1; B1=191; N=2^55050557-1; X=0xabc; CHECKSUM=17; "
    "PROGRAM=PrMers new; X0=0x03; Y=0; Y0=0; WHO=bob; TIME=later;"
)
assert MODULE.parse_resume_record(pm1_a) == MODULE.parse_resume_record(pm1_b)

ecm_a = (
    "METHOD=ECM; SIGMA=123456789; B1=50; N=2^55050557-1; X=0x001234; "
    "CHECKSUM=19; PROGRAM=PrMers old; X0=0x0; Y0=0x0; TIME=one;"
)
ecm_b = (
    "METHOD=ECM; SIGMA=123456789; B1=50; N=2^55050557-1; X=0x1234; "
    "CHECKSUM=19; PROGRAM=PrMers new; X0=0; Y0=0; TIME=two;"
)
assert MODULE.parse_resume_record(ecm_a) == MODULE.parse_resume_record(ecm_b)
assert MODULE.parse_resume_record(ecm_a) != MODULE.parse_resume_record(ecm_b.replace("X=0x1234", "X=0x1235"))

print("Aevum workload audit canonical resume parser test passed")
