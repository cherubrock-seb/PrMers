# RTX 5090 Pass-5 reporter test

Place the source ZIP in `~/mgpu`. Device 0 is used below.

```bash
cd ~/mgpu
sudo apt-get update
sudo apt-get install -y build-essential libgmp-dev ocl-icd-opencl-dev clinfo clang python3 zip unzip
PASS5_RUN=$(mktemp -d "$PWD/pass5-rtx5090-XXXXXX")
unzip -q prmers-aevum-optimized-pass5-prp-max.zip -d "$PASS5_RUN"
cd "$PASS5_RUN/prmers-aevum-optimized-pass5-prp-max"
AEVUM_TARGET=rtx5090 AEVUM_DEVICES=0 bash scripts/bench_aevum_pass5.sh --reporter
```

This single short campaign runs:
- p=21000029 word regression, including the historical profile.
- p=147800003 cold implementation tune plus cache hit; shape must remain `1:512:8:512:202`.
- p=180000007 cold shape/use tune plus cache hit; expected shape `4:512:8:512:202`.
- p=196999969 non-regression plus cache hit; expected shape `4:512:8:512:202`.

Each exponent gets same-shape Pass-4/Pass-5 alternating A/B with exact words, isolated startup timing and separate kernel profiling. Cold means fresh plan/use records; an existing OpenCL compiler cache may remain warm. If cold compilation prevents use confirmation, an exact fast historical vector triggers one warm retry.

The driver removes inherited manual overrides for runtime tests. It creates a fresh isolated cache; do not add manual plan-directory overrides. The carry-epoch experiment is excluded from this short reporter run.

Return only the printed `pass5-rtx5090-*/tuning-output.zip`. Summary JSON records failures and continues instead of retaining mismatched or regressing profiles. No new speedup is assumed in advance.
