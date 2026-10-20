#!/usr/bin/env python3
"""GPU-less checks for the optional CUDA/OpenCL loader contract."""
from pathlib import Path
src = Path("src/aevum/EngineAevum.cpp").read_text()
make = Path("Makefile").read_text()
assert 'if (!one_nvidia_gpu(device)) return opencl;' in src
assert 'AEVUM_ENGINE_LIB' in src and 'AEVUM_BACKEND' in src
assert 'libaevum_engine_cuda.so' in src and 'aevum_engine_cuda.dll' in src
assert 'std::filesystem::is_regular_file(cuda_path)' in src
assert 'cuda_result.residue == ocl_result.residue' in src
assert 'cuda_result.us_per_square < 0.97 * ocl_result.us_per_square' in src
assert 'cache[key] = false;' in src and 'using OpenCL:' in src
assert 'backend=" << (api_.is_cuda ? "CUDA" : "OpenCL")' in src
assert 'aevum-dual:' in make and 'CUDA=1 engine-lib' in make
print("DUAL_BACKEND_SOURCE_GATE=PASS")
