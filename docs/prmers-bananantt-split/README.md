# BananaNTT v78

BananaNTT is an experimental OpenCL PRP/NTT branch derived from the PrMers work.  It keeps the PrMers-style command line, worktodo handling, JSON output, result append file, backup/restore and Gerbicz-Li checking, but the odd radix 9 CRT/PFA half-real path is now the default CRT path when available.

The normal simple run is:

```bash
./prmers_opencl_prp 142606357 --device 1
```

By default this uses:

- CRT with GF(M61^2) and GF(M31^2)
- odd radix 9 mixed CRT/PFA row half-real path
- LDS512 row center when it is the best known safe default
- OpenCL binary cache in `.ocl_cache/`
- Gerbicz-Li enabled with GPU D update and GPU full check
- backup in `save/M<p>.bananantt.chk`
- restore enabled if a matching backup exists
- PrMers-compatible JSON format
- append of final P/C results to `./results.txt`

## Build

```bash
cd ~/mgpu
rm -rf odd9_release_v78_bananantt_autotune_x8
unzip odd9_release_v78_bananantt_autotune_x8.zip
cd odd9_release_v78_bananantt_autotune_x8

g++ -O3 -std=c++20 -march=native -pthread prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

## Quick validation

```bash
./prmers_opencl_prp 216091 \
  --device 1 \
  --iters 1 \
  --crt-halfreal-no-autoprobe \
  --crt-halfreal-validate-random \
  --crt-halfreal-validate-iters 1 \
  --crt-mixed-gpu-reference
```

## Benchmark without safety overhead

For a pure speed test, disable Gerbicz, backup, restore and startup autotune:

```bash
./prmers_opencl_prp 9437189 \
  --device 1 \
  --iters 5000 \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --no-startup-autotune \
  --quiet
```

With kernel profile:

```bash
./prmers_opencl_prp 9437189 \
  --device 1 \
  --iters 1000 \
  --no-gerbicz \
  --no-backup \
  --no-resume \
  --no-startup-autotune \
  --profile-kernels
```

## Startup autotune

For real runs without `--iters`, startup autotune is enabled by default.  It runs a short set of child benchmarks and selects the fastest mixed row center/stage combination for the detected GPU.  The output shows each candidate so the program does not look stuck.

Disable it:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-startup-autotune
```

Force it during a benchmark:

```bash
./prmers_opencl_prp 9437189 --device 1 --iters 5000 --startup-autotune
```

Longer/wider test:

```bash
./prmers_opencl_prp 9437189 \
  --device 1 \
  --startup-autotune \
  --crt-autotune-iters 3000 \
  --crt-autotune-wide
```

Force row parameters manually:

```bash
./prmers_opencl_prp 9437189 \
  --device 1 \
  --crt-mixed-row-center 512 \
  --crt-mixed-row-stage 512 \
  --no-startup-autotune
```

Useful row sizes to compare are 256, 512 and 1024.  The 512 kernels are still the main optimized path.  v78 also adds the small radix 8/16/32 x8 path, selected by default before x4/x2.

Disable small radix variants individually:

```bash
PRMERS_CRT_MIXED_SMALL32_BATCH8=0 ./prmers_opencl_prp 9437189 --device 1 --iters 1000 --profile-kernels --no-gerbicz --no-startup-autotune
PRMERS_CRT_MIXED_SMALL32_X4=0 ./prmers_opencl_prp 9437189 --device 1 --iters 1000 --profile-kernels --no-gerbicz --no-startup-autotune
PRMERS_CRT_MIXED_SMALL32_X2=0 ./prmers_opencl_prp 9437189 --device 1 --iters 1000 --profile-kernels --no-gerbicz --no-startup-autotune
```

## Gerbicz-Li

Gerbicz-Li is on by default for full runs.  The default target is designed to keep the boundary update cheap and the expensive full check rare.  For debugging or forced error tests, use shorter settings.

Normal run with Gerbicz:

```bash
./prmers_opencl_prp 10000019 --device 1 --gerbicz-seconds 600 --gerbicz-boundary-seconds 2
```

Inject an error and detect it sooner:

```bash
./prmers_opencl_prp 10000019 \
  --device 1 \
  --error-iter 2000 \
  --gerbicz-seconds 30 \
  --gerbicz-boundary-seconds 1 \
  --res64-every 1000 \
  --no-resume
```

Host/GMP check mode is slower but useful as an independent check:

```bash
./prmers_opencl_prp 216091 --device 1 --gerbicz-host --gerbicz-seconds 20 --no-resume
```

Disable Gerbicz:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-gerbicz
```

## Backup and restore

Default backup path:

```text
save/M<p>.bananantt.chk
```

Disable restore for a clean run:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-resume
```

Disable backup:

```bash
./prmers_opencl_prp 142606357 --device 1 --no-backup
```

Change backup location:

```bash
./prmers_opencl_prp 142606357 --device 1 --backup-dir my_save
```

Backup is saved on Ctrl-C when possible.

## Output files

Default JSON:

```text
prmers_bananantt_M<p>.json
```

Default final result append file:

```text
./results.txt
```

Change output directory:

```bash
./prmers_opencl_prp 756839 --device 1 --output-dir out
```

Change JSON file:

```bash
./prmers_opencl_prp 756839 --device 1 --json-file out/M756839.json
```

Change result append file:

```bash
./prmers_opencl_prp 756839 --device 1 --results-file out/results.txt
```

Disable JSON or result append:

```bash
./prmers_opencl_prp 756839 --device 1 --no-json
./prmers_opencl_prp 756839 --device 1 --no-results
```

The JSON format follows the PrMers-style object:

```json
{"status":"P","exponent":756839,"worktype":"PRP-3","res64":"0000000000000001","res2048":"...","residue-type":1,"errors":{"gerbicz":0},"shift-count":0,"fft-length":40960,"proof":{"version":2,"power":6,"hashsize":64,"md5":"..."},"program":{"name":"banana","version":"0.79.00-alpha","port":8},"os":{"os":"Linux","architecture":"x86_64"},"timestamp":"2026-05-23 20:09:59","checksum":{"version":1,"checksum":"..."}}
```

## OpenCL binary cache

The first run builds the kernels and writes binary cache files in `.ocl_cache/`.  Next runs reload the binary cache when the device, program version, source and build options match.

Force rebuild by deleting the cache:

```bash
rm -rf .ocl_cache
```

Change cache directory:

```bash
PRMERS_OCL_CACHE_DIR=/tmp/bananantt_ocl_cache ./prmers_opencl_prp 216091 --device 1
```

Disable cache:

```bash
PRMERS_OCL_BINARY_CACHE=0 ./prmers_opencl_prp 216091 --device 1
```

## worktodo

If no exponent is given, the program tries to read `worktodo.txt`.

```bash
./prmers_opencl_prp --device 1
```

Use another worktodo file:

```bash
./prmers_opencl_prp --device 1 --worktodo ./worktodo.txt
```

Ignore worktodo:

```bash
./prmers_opencl_prp --device 1 --no-worktodo 216091
```

## Notes

This is still BananaNTT v1 experimental code.  Keep PrMers as the PrimeNet-compatible reference path until the proof/result workflow has been validated on long runs.

## Small stage naming

`row-stage-plan=fwd=32` means the mathematical LDS stage is still radix 32.

`batch8` means eight independent radix-32 row transforms are packed in one kernel launch/workgroup family to improve occupancy and reduce overhead. It does not mean the stage became radix 8.

Typical profile labels are now:

```text
crt_mixed_lds32_forward_batch8_61
crt_mixed_lds32_inverse_batch8_61
crt_mixed_lds32_forward_batch8_31
crt_mixed_lds32_inverse_batch8_31
```

Disable it with:

```bash
PRMERS_CRT_MIXED_SMALL32_BATCH8=0 ./prmers_opencl_prp ...
```
