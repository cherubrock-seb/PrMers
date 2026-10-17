#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
shim = (root / "src/cuda/clwrap_cuda.cpp").read_text()
api = (root / "src/EngineApi.cpp").read_text()

required = [
    "static vector<CUcontext> g_cudaContexts;",
    "static void rememberCudaContext(CUcontext ctx)",
    "static void forgetCudaContext(CUcontext ctx)",
    "g_cudaContexts.push_back(ctx);",
    "g_cudaContexts.erase(it);",
    "g_cudaContext = g_cudaContexts.empty() ? nullptr : g_cudaContexts.back();",
    "rememberCudaContext(ctx->ctx);",
    "const CUcontext dead = ctx->ctx;",
    "forgetCudaContext(dead);",
    "cuCtxDestroy(dead);",
    "if (g_cudaContext) cuCtxSetCurrent(g_cudaContext);",
]

missing = [item for item in required if item not in shim]
assert not missing, "missing CUDA context lifecycle source: " + ", ".join(missing)

release = shim.index("int clReleaseContext(cl_context ctx)")
forget = shim.index("forgetCudaContext(dead);", release)
destroy = shim.index("cuCtxDestroy(dead);", release)
restore = shim.index("cuCtxSetCurrent(g_cudaContext);", release)
assert forget < destroy < restore, "context release must forget, destroy, then restore"

assert "struct TuneContext" in api
assert "shared.context = context.get();" in api

print("CUDA autotune context lifecycle source test PASS")
