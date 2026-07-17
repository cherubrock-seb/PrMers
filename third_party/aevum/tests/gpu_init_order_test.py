#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
h = (root / "src/Gpu.h").read_text(encoding="utf-8")
cpp = (root / "src/EngineApi.cpp").read_text(encoding="utf-8")
compiler = h.find("KernelCompiler compiler;")
if compiler < 0:
    raise SystemExit("KernelCompiler member not found")
for declaration in (
    "bool tail_single_wide{false};",
    "bool tail_single_kernel{true};",
    "u32 in_place{0};",
    "u32 wmul{2};",
    "u32 pad_size{0};",
):
    pos = h.find(declaration)
    if pos < 0:
        raise SystemExit(f"missing initialized Gpu member: {declaration}")
    if pos > compiler:
        raise SystemExit(f"Gpu member is declared after KernelCompiler: {declaration}")

# v99.49 intentionally restores the exact GitHub prepared-cache default on all
# platforms. Apple compatibility is implemented inside the prepared transform
# routing, not by disabling the cache.
prepared_default = cpp.find("size_t prepared_count = std::min<size_t>(register_count_, 2);")
prepared_env = cpp.find('std::getenv("AEVUM_PREPARED_CACHE")', prepared_default)
prepared_alloc = cpp.find("prepared_buffers_ = gpu_->makeTransformBufVector", prepared_default)
if min(prepared_default, prepared_env, prepared_alloc) < 0 or not (prepared_default < prepared_env < prepared_alloc):
    raise SystemExit("GitHub prepared-cache default/order is missing")
if "size_t prepared_count = 0;" in cpp:
    raise SystemExit("Apple prepared cache must not be disabled")

print("Aevum GPU initialization-order and GitHub prepared-cache policy test passed")
