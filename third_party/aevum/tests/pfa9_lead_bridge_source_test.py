#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fftp = (root / "src/cl/fftp.cl").read_text()
gpu = (root / "src/Gpu.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()
header = (root / "src/Gpu.h").read_text()

required = {
    "fftp carry-aware gather": "fftPCarryB" in fftp and "pfaLoadCanonicalWordCarried" in fftp,
    "GPU dispatch": "void Gpu::fftPCarryB" in gpu and "kfftPCarryB" in gpu,
    "PFA retained lead boundary": "if (fft.isPfa())" in gpu and "fftPCarryB(buf1, out);" in gpu,
    "PFA9 support gate": "fft.pfa_radix == 9" in gpu,
    "API opt-in gate": "AEVUM_PFA_LEAD_BRIDGE" in api and "Aevum PFA9 lead bridge enabled" in api,
    "kernel member": "Kernel kfftPCarryB" in header,
}
missing = [name for name, ok in required.items() if not ok]
if missing:
    raise SystemExit("PFA9 lead bridge source audit failed: " + ", ".join(missing))
print("Aevum PFA9 lead-bridge scheduling/source test passed")
