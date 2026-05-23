# BananaNTT 0.74.00-alpha

GPU PRP prototype for Mersenne numbers using the mixed CRT/PFA odd-radix path.

## Build

```bash
cd ~/mgpu
rm -rf odd9_release_v74_bananantt_ocl_cache
unzip odd9_release_v74_bananantt_ocl_cache.zip
cd odd9_release_v74_bananantt_ocl_cache

g++ -O3 -std=c++20 -march=native -pthread prmers_opencl_prp.cpp \
  -o prmers_opencl_prp $(pkg-config --cflags --libs OpenCL) -lgmp
```

## OpenCL binary cache

The first run compiles the OpenCL program and saves a binary in `.ocl_cache/`.
Next runs with the same GPU, driver, kernel source and build options load the binary cache.

```bash
./prmers_opencl_prp 10000019 --device 0 --iters 1000 --no-resume
./prmers_opencl_prp 10000019 --device 0 --iters 1000 --no-resume
```

Controls:

```text
PRMERS_OCL_BINARY_CACHE=0       disable host binary cache
PRMERS_OCL_CACHE_DIR=/path      choose cache directory
PRMERS_SHOW_OCL_BUILD=0         hide build/cache messages
PRMERS_OCL_BUILD_SPINNER=0      keep simple build messages
```

To reset the cache:

```bash
rm -rf .ocl_cache
```

## Default run

```bash
./prmers_opencl_prp 142606357 --device 1
```

Defaults:

- CRT mixed odd-radix auto path
- odd radix 9 when selected by the planner
- half-real row core
- Gerbicz-Li enabled
- backup and resume enabled
- JSON result enabled
- append result to `./results.txt`
- adaptive queue guard enabled
- OpenCL binary cache enabled

## Gerbicz-Li

Default Gerbicz is quiet. It prints the setup line and only prints full checks, failures, restore, backup and final status.

Useful commands:

```bash
./prmers_opencl_prp 216091 --device 1 --iters 50000 --no-resume
```

```bash
./prmers_opencl_prp 216091 --device 1 --iters 50000 --gerbicz-seconds 20 --no-resume
```

Fast injected-error test on M10000019:

```bash
./prmers_opencl_prp 10000019 --device 0 \
  --iters 8000 \
  --no-resume \
  --no-backup \
  --gerbicz-b 1024 \
  --gerbicz-checklevel 1 \
  --error-iter 2000 \
  --error-limb 0 \
  --error-delta 1
```

More visible diagnostics:

```bash
./prmers_opencl_prp 10000019 --device 0 \
  --iters 8000 \
  --no-resume \
  --no-backup \
  --gerbicz-b 1024 \
  --gerbicz-checklevel 1 \
  --error-iter 2000 \
  --error-limb 0 \
  --error-delta 1 \
  --gerbicz-progress
```

Main controls:

```text
--gerbicz-seconds S            target full Li check spacing
--gerbicz-boundary-seconds S   target D update spacing when B is automatic
--gerbicz-b B                  force Li block size
--gerbicz-checklevel N         force full check every N boundaries
--gerbicz-verbose              print every boundary update
--gerbicz-progress             print full-check leg progress
--gerbicz-host                 use host GMP checker
--no-gerbicz                   disable Gerbicz-Li
```

For speed tests, avoid `--res64-every` unless needed. It forces regular readbacks and synchronization.

## Backup and resume

```bash
./prmers_opencl_prp 142606357 --device 1 --backup-seconds 120
```

Resume is automatic from `save/M<p>.bananantt.chk`.

Disable resume for clean tests:

```bash
./prmers_opencl_prp 216091 --device 1 --no-resume
```

Disable backup and resume for pure benchmarks:

```bash
./prmers_opencl_prp 142606357 --device 1 --iters 30000 --quiet --no-backup --no-resume
```

## Queue guard

The default is adaptive:

```text
--queue-guard auto --queue-guard-seconds 2
```

This limits Ctrl-C latency without forcing a fixed `clFinish` every 256 iterations.

Pure benchmark:

```bash
./prmers_opencl_prp 142606357 --device 1 --iters 30000 --quiet --no-gerbicz --no-backup --no-resume --queue-guard 0
```

## Output files

Default files:

```text
./prmers_bananantt_M<p>.json
./results.txt
./save/M<p>.bananantt.chk
./<p>/proof/
```

Path controls:

```text
--output-dir DIR
--json-file FILE
--results-file FILE
--backup-dir DIR
--save-file FILE
--resume-file FILE
--proof-dir DIR
```

The JSON follows the PrMers-style result shape with `program.name = "banana"` and `program.version = "0.74.00-alpha"`.
