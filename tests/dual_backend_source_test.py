#!/usr/bin/env python3
"""GPU-less source gates for unified CUDA/OpenCL arbitration.

These are structural regression guards, not substitutes for exact GPU tests.
"""
from pathlib import Path

src = Path("src/aevum/EngineAevum.cpp").read_text()
make = Path("Makefile").read_text()
workflow = Path(".github/workflows/unified-cuda-opencl-release.yml").read_text()
mac = Path(".github/workflows/build_mac_os.yml").read_text()

for token in (
    "mapped_cuda_device(device)",
    'symbol("cuInit")',
    'symbol("cuDeviceGetCount")',
    'symbol("cuDeviceGetPCIBusId")',
    "clGetDeviceInfo(gpu, pci_bus_nv",
    "clGetDeviceInfo(gpu, pci_slot_nv",
    "nvidia_count == 1",
    "return std::nullopt;",
    "AEVUM_ENGINE_LIB",
    "AEVUM_BACKEND",
    "libaevum_engine_cuda.so",
    "aevum_engine_cuda.dll",
    "std::filesystem::is_regular_file(cuda_path)",
    "cache.find(key)",
    "workload == 3u || workload == 4u || workload == 6u",
    "engine_api.mul(handle, 0, 2, 1)",
    "engine_api.add(handle, 0, 1)",
    "engine_api.prepare(handle, 2, 1)",
    "cuda_result.residue == ocl_result.residue",
    "cuda_result.us_per_operation < 0.97 * ocl_result.us_per_operation",
    'selected=" << (use_cuda ? "CUDA" : "OpenCL")',
    "using OpenCL:",
    'backend=" << (api_.is_cuda ? "CUDA" : "OpenCL")',
    "api_.is_cuda ? choice_.cuda_ordinal",
):
    assert token in src, f"Missing safe CUDA/OpenCL selector element: {token}"

assert "workload != 1u" not in src, "PRP-only backend gate reintroduced"
assert "one_nvidia_gpu(device)" not in src, "One-GPU backend gate reintroduced"
assert "aevum-dual:" in make and "CUDA=1 engine-lib" in make
assert "libaevum_engine_cuda.so $(DESTDIR)$(PREFIX)/lib/prmers/" in make
assert "make aevum-dual" in workflow
assert "uses: ./.github/workflows/build_mac_os.yml" in workflow
assert "needs: [linux, windows, macos]" in workflow
assert "workflow_call:" in mac
assert "prmers-macos-release-packages" in workflow
print("DUAL_MULTIMODE_SOURCE_GATE=PASS")
