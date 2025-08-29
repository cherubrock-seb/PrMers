PrMers: GPU-accelerated Mersenne Primality Testing
==================================================

PrMers is a high-performance GPU application for Lucas–Lehmer (LL), PRP, and P-1
testing of Mersenne numbers. It uses OpenCL and integer NTT/IBDWT kernels and is
built for long, reliable runs with checkpointing and PrimeNet submission.

Backend
-------
- Default backend: Marin (https://github.com/galloty/marin)
  - Efficient modular exponentiation with Gerbicz–Li error checking.
  - Fast Mersenne-mod multiplication via IDBWT over Z/(2^64 - 2^32 + 1)Z.
- You can disable Marin and use the legacy internal backend with: -marin

Key Features
------------
- OpenCL GPU acceleration (OpenCL 1.2+; 2.0 recommended)
- LL and PRP for Mersenne (and PRP of cofactors with known factors)
- P-1 factoring (stage-1 and stage-2)
- Gerbicz–Li timed validation checkpoints (PRP)
- Automatic disk checkpoints with deterministic resume
- PrimeNet result submission (JSON + optional auto-submit)
- Cross-platform builds (Linux, macOS, Windows)

Google Colab (demo)
-------------------
You can test **PrMers** directly in your browser with GPU acceleration by opening the interactive notebook below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/cherubrock-seb/PrMers/blob/main/prmers.ipynb)

https://colab.research.google.com/github/cherubrock-seb/PrMers/blob/main/prmers.ipynb


Performance (as measured)
-------------------------
GeForce RTX 5090
  | Exponent  | Iter/s  | ETA            |
  |-----------|---------|----------------|
  | 136279841 | 2730.69 | 1d 18h 31m 29s |
  | 82589933  | 2647.09 | 0d 8h 36m 31s  |
  | 74207281  | 2658.54 | 0d 7h 41m 33s  |
  | 57885161  | 2730.69 | 0d 5h 51m 42s  |

Radeon VII
  | Exponent  | Iter/s  | ETA            |
  |-----------|---------|----------------|
  | 136279841 | 290.77  | 5d 10h 10m 27s |
  | 82589933  | 544.85  | 1d 18h 5m 45s  |
  | 74207281  | 544.38  | 1d 15h 46m 49s |
  | 57885161  | 552.32  | 1d 5h 6m 18s   |

MacBook Air 2022 (Apple M2)
  | Exponent  | Iter/s | ETA             |
  |-----------|--------|-----------------|
  | 136279841 | 31.16  | 50d 14h 57m 35s |
  | 82589933  | 50.88  | 18d 18h 52m 10s |
  | 77232917  | 51.17  | 17d 11h 16m 9s  |
  | 74207281  | 51.15  | 16d 18h 58m 41s |
  | 57885161  | 51.00  | 13d 3h 16m 30s  |

Intel HD Graphics (OpenCL 1.2 legacy)
  | Exponent  | Iter/s | ETA             |
  |-----------|--------|-----------------|
  | 136279841 | 2.46   | 637d 18h 5m 0s  |
  | 1257787   | 245.44 | 0d 1h 24m 0s    |
  | 756839    | 249.23 | 0d 0h 49m 53s   |

Requirements
------------
- OpenCL drivers for your GPU
- C++20 compiler
- libcurl (for PrimeNet)
- Optional: GMP for some CPU-side helpers

Quick Start
-----------
1) Build from source
   Linux/macOS (example):
     make
   Windows (recommended): CMake + vcpkg + Visual Studio
     cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
     cmake --build build --config Release

2) Run a PRP test
   ./prmers 136279841
   (Default: PRP mode, Marin backend on)

3) Disable Marin (use legacy internal backend)
   ./prmers 136279841 -marin

4) P-1 stage-1 + stage-2
   ./prmers 367 -pm1 -b1 11981 -b2 38971

5) Use worktodo.txt or a config file
   ./prmers -worktodo ./worktodo.txt
   ./prmers -config ./settings.cfg

Command-Line Options (selected)
-------------------------------
<p>                 exponent to test (Mersenne p)
-d <id>            OpenCL device id
-prp               force PRP mode (default)
-ll                Lucas–Lehmer mode
-pm1               P-1 factoring; use -b1 <B1> and optional -b2 <B2>
-factors <csv>     known factors, test the remaining Mersenne cofactor
-t <sec>           checkpoint interval (default 60)
-f <path>          checkpoint directory (default .)
-proof <k>         set proof power (1..12) or 0 to disable
-user <name>       PrimeNet username (for submission)
-computer <name>   PrimeNet computer name
--noask            auto-submit results (requires -user and -password)
-password <pwd>    PrimeNet password (only with --noask)
-worktodo [path]   load exponent from PRP= line in worktodo file
-config <path>     load options from a .cfg file
-res64_display_interval <n>  print residues every n iterations
-erroriter <i>     inject fault at iteration i (PRP Gerbicz–Li testing)
-marin             disable the Marin backend (use legacy NTT backend)

Gerbicz–Li (PRP)
----------------
- Time-based verification every ~T seconds (default T=600).
- Two rolling products; on mismatch the run restores to the last verified state.
- Files: .bufd, .lbufd, .gli, .isav, .jsav ensure deterministic recovery.
Reference: D. Li, Y. Gallot, "An Efficient Modular Exponentiation Proof Scheme", arXiv:2209.15623

P-1 Factoring (overview)
------------------------
Stage-1:
  choose B1, build E=lcm(1..B1), compute x=3^(E·2p) mod (2^p-1), factor=gcd(x-1,2^p-1)
Stage-2:
  search primes q in (B1,B2] using cached powers; final gcd reveals a factor if present.

worktodo.txt and Config
-----------------------
- worktodo.txt: supports PRP= lines; the program extracts n (the Mersenne exponent).
- -config loads a file containing command-line flags (one line, space-separated).
- Precedence: explicit CLI > config > worktodo.

PrimeNet Submission
-------------------
- On completion, a JSON is written (res64, res2048, meta).
- You may submit automatically with --noask -user <name> -password <pwd>.
- Manual submission remains available; unsent results are detected on next run.

Proofs (experimental)
---------------------
- PRP proof generation similar in spirit to gpuowl; still under stabilization.

Build Notes
-----------
Linux:
  sudo apt-get update
  sudo apt-get install -y ocl-icd-opencl-dev opencl-headers libcurl4-openssl-dev g++ make
macOS:
  Xcode CLT, curl available; OpenCL is preinstalled on supported versions.
Windows (CMake + vcpkg recommended):
  vcpkg install curl:x64-windows
  cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release

Uninstall / Clean
-----------------
sudo make uninstall
make clean

Sample Usage
------------
PRP with residues every 1000 iterations, inject a test error:
  ./prmers 6972593 -res64_display_interval 1000 -erroriter 19500
Resume from checkpoints automatically on restart.

Resources and Credits
---------------------
- GIMPS: https://www.mersenne.org
- Mersenne Forum: https://www.mersenneforum.org
- Marin backend: https://github.com/galloty/marin
- Darren Li, Yves Gallot (Gerbicz–Li scheme): https://arxiv.org/abs/2209.15623
- GPUOwl: https://github.com/preda/gpuowl
- Genefer: https://github.com/galloty/genefer22
- Nick Craig-Wood IOCCC details on modular arithmetic and transforms.
- Author: Cherubrock
