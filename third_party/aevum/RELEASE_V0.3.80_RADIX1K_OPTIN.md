# Aevum v0.3.80 candidate — 1K radix-8 opt-in

This candidate synchronizes the validated gpuowl 6cf0dc 1K radix-8 engine
path from PrMers v100.08 while preserving standalone Aevum's PFA resident-v8
work.

Safety policy:

- 1K radix-4 remains the default;
- `AEVUM_RADIX1K=8` explicitly enables the validated radix-8 implementation;
- `AEVUM_RADIX1K=4` explicitly retains radix-4;
- any other non-empty value is rejected by the engine API;
- 256 remains radix-4;
- no NVIDIA/AMD vendor heuristic is used.

The candidate is intentionally not tagged as a standalone release until the
same real-GPU A/B validation used for PrMers passes on Radeon VII and RTX 3080.
