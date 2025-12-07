PrMers: GPU-accelerated Mersenne Primality Testing
==================================================
https://github.com/cherubrock-seb/PrMers/

PrMers is a high-performance GPU application for Lucas–Lehmer (LL), PRP, P‑1 and ECM
testing of Mersenne numbers. It uses OpenCL and an integer NTT / IBDWT engine modulo

    p = 2^64 − 2^32 + 1

and is designed for long, reliable runs with checkpointing and optional PrimeNet
submission.

The project also supports PRP tests of cofactors and Wagstaff numbers, and includes
a web-based GUI and a GPU VRAM tester.


Overview of Algorithms and Backends
-----------------------------------

PrMers has two main computational backends:

- Marin backend (default)
  - External library: https://github.com/galloty/marin
  - Efficient modular exponentiation with Gerbicz–Li style error checking (in PRP and LL safe).
  - Uses IBDWT-style transforms over Z / (2^64 − 2^32 + 1) Z.
  - Supports PRP, LL, P‑1 and ECM on Mersenne numbers.

- Internal NTT backend
  - Integer NTT / IBDWT implementation inside PrMers.
  - Used when the Marin backend is disabled.
  - Option for experimentation and comparison.

Select backend:
- Default: Marin backend enabled.
- `-marin`: disable Marin and use the internal NTT backend instead.


Supported Modes
---------------

Numbers
- **Mersenne**:            N = 2^p − 1
- **Wagstaff**:            W = (2^p + 1) / 3  (via `-wagstaff`)
- **Cofactors**:           N / product(known factors), PRP‑tested as generic integer

Main modes
- **PRP (default)**
  - Probable-prime test with Gerbicz–Li error checking.
  - Works for Mersenne, Wagstaff and cofactors.
  - Produces Res64 and full residue (optionally a proof).

- **Lucas–Lehmer (LL)**
  - Three LL modes exist (GPU). See the dedicated section *“Lucas–Lehmer modes and safety”* below.
    - `-ll`        → **LL (safe)** (default LL mode)
    - `-llunsafe`  → **LL (classic/unsafe)**
    - `-llsafe2`   → **LL (safe, “doubling” variant)**

- **P‑1 factoring**
  - Stage 1 and Stage 2 on N = 2^p − 1.
  - Stage is error checked with Gerbicz–Li.
  - Targets factors q of N such that q − 1 is B1‑smooth (with Stage 2 extension).
  - **GPU (Marin) only.** Stage 2 supports both the classic prime-sweep and an **n^K (Crandall) variant** (see `-K` and `-nmax`).
  - **Interoperability & resume files** (after Stage 1 or Stage 2, see `-resume` / `-p95`):
    - Export a **GMP‑ECM `.save`** resume and/or a **Prime95 `.p95`** resume.
    - You can **extend Stage 1** from an existing **`.save` or `.p95`** using `-b1old <B1old>` (auto‑detects the matching file).

- **ECM**
  - Elliptic Curve Method on N = 2^p − 1.
  - Multiple curve models (Edwards/Montgomery) and torsion variants.

- **Wagstaff**
  - With `-wagstaff`, runs PRP/ECM/P‑1 on W = (2^p + 1) / 3.


Requirements
------------

- OpenCL 1.2 runtime (OpenCL 2.0 recommended) and a supported GPU.
- C++20 compiler (g++/clang++ on Linux/macOS; MSVC or MinGW on Windows).
- libcurl (PrimeNet HTTP client).
- GMP (ECM and CPU‑side helpers).

Debian/Ubuntu packages:

    sudo apt-get update
    sudo apt-get install -y g++ make \
        ocl-icd-opencl-dev opencl-headers \
        libcurl4-openssl-dev \
        libgmp-dev


Building from Source
--------------------

Clone the repository:

    git clone https://github.com/cherubrock-seb/PrMers.git
    cd PrMers

Linux / macOS (Makefile)
- Build:

    make -j$(nproc)

- Install executable and kernels:

    sudo make install

This installs:
- Executable:   /usr/local/bin/prmers
- Kernels:      /usr/local/share/prmers/

The binary embeds a `KERNEL_PATH` pointing to the installation directory so that
PrMers can find its OpenCL kernels after installation.

Windows with CMake + vcpkg (recommended)
- Install CMake, Visual Studio and vcpkg.
- From the PrMers directory:

    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg install curl:x64-windows
    cd ..

    cmake -S . -B build ^
      -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake ^
      -DCMAKE_BUILD_TYPE=Release

    cmake --build build --config Release

Copy the required DLLs from `vcpkg\installed\x64-windows\bin` next to `prmers.exe`
or add that directory to your PATH.

Windows with MSYS2 / MinGW (UCRT64)
- In MSYS2 UCRT64 shell:

    pacman -Syu
    pacman -S --noconfirm make \
        mingw-w64-ucrt-x86_64-gcc \
        mingw-w64-ucrt-x86_64-opencl-headers \
        mingw-w64-ucrt-x86_64-opencl-icd-loader \
        mingw-w64-ucrt-x86_64-gmp \
        mingw-w64-ucrt-x86_64-curl

- Build:

    make -j$(nproc)

Prebuilt binaries
- Linux / Windows / macOS builds are on GitHub Releases:
  https://github.com/cherubrock-seb/PrMers/releases

macOS notes
- OpenCL is present on supported macOS versions.
- On first run, Gatekeeper may block the binary (not notarized). Allow it in
  *System Settings → Security & Privacy*, then run again.


Quick Start
-----------

Basic PRP on a Mersenne exponent:

    ./prmers 136279841

- Mode: PRP (default, Marin backend).
- Checkpoint every 120 s (default).
- Results go to `results.txt` and `<exponent>_prp_result.json`.

Lucas–Lehmer test (safe mode):

    ./prmers 127 -ll

P‑1 (stage 1 and stage 2):

    ./prmers 367 -pm1 -b1 11981 -b2 38971

Export Stage‑1 resume to GMP‑ECM `.save` and Prime95 `.p95`:

    ./prmers 367 -pm1 -b1 11981 -resume      # both .save and .p95
    ./prmers 367 -pm1 -b1 11981 -p95         # .p95 only

Extend Stage‑1 from a previous B1 using `.save` or `.p95` (auto‑detected):

    ./prmers 367 -pm1 -b1 38971 -b1old 11981

ECM on a Mersenne number:

    ./prmers 701 -ecm -b1 6000 -K 8
    ./prmers 701 -ecm -b1 6000 -b2 33333 -K 8

Test a Wagstaff number W = (2^p + 1) / 3:

    ./prmers 100003 -wagstaff

Use worktodo.txt:

    ./prmers -worktodo ./worktodo.txt

Use a config file:

    ./prmers -config ./settings.cfg

Disable Marin and use the internal backend:

    ./prmers 136279841 -marin


Lucas–Lehmer modes and safety
-----------------------------

PrMers implements three LL variants:

1) **LL (safe)** - `-ll`  
   - Uses the split representation *s = a + b√3* so each squaring is computed as:
     (a + b√3)^2 = (a^2 + 3b^2) + (2ab)√3.  
     Implemented as four transforms per iteration:  
     A=T(a), B=T(b), invT(A^2 + T(3)·B^2), 2·invT(A·B) (start from (a,b)=(2,1); final check is (−1,0)).  
   - **Error checking:** protected by Gerbicz–Li style verification with periodic
     roll‑back/restore to the last verified checkpoint. Default check cadence is
     ~10 minutes; tune with `-checklevel <k>` (higher = more frequent). Disable
     with `-gerbiczli` (not recommended except for benchmarking).
   - **Speed:** safer but slower than the classic LL (more transforms per step).
   - **When to use:** when you need a reliable LL run on GPU.

2) **LL (classic / unsafe)** - `-llunsafe`  
   - Classical recurrence S_0=4; S_{i+1}=S_i^2−2 modulo M_p.  
   - **Error checking:** *none*. Fastest LL but susceptible to silent errors on
     marginal hardware or aggressive overclocks.
   - **When to use:** quick checks / debugging. Prefer PRP or LL safe for proofs.

3) **LL (safe2, doubling)** - `-llsafe2` [optional `-llsafeb <B>`]  
   - Block‑doubling consistency check variant. Work is split into blocks of size *B*
     (by default B≈⌊√p⌋); at block boundaries a
     doubling identity is used to verify progress and roll back on mismatch.
   - **Error checking:** periodic; lighter‑weight than full GL but still robust.
   - **Tuning:** set block size with `-llsafeb <B>` (auto if omitted).

**Notes**
- LL modes are only for genuine Mersenne numbers. For cofactors, use PRP.
- You can inject an error to test detection/restart with `-erroriter <i>`.
- `-res64_display_interval N` prints Res64 every N iterations (0 disables).


Command Line Options (summary)
------------------------------

For the full list, run:

    ./prmers -h

Common options

Positional
- `<p>`                   Exponent of the Mersenne or Wagstaff number.

Device and performance
- `-d <id>`               OpenCL device id (default: 0).
- `-O <flags>`            OpenCL compiler opts, e.g. `fastmath mad`.
- `-c <depth>`            Local carry propagation depth.
- `-profile`              Enable kernel profiling.
- `-memtest`              Run GPU VRAM test.

Modes
- `-prp`                  Force PRP (default).
- `-ll`                   LL safe (a + b√3 with GL checks).
- `-llunsafe`             LL classic (no checks).
- `-llsafe2`              LL safe “doubling” variant (block checks).
- `-wagstaff`             Test W = (2^p + 1)/3 instead of 2^p − 1.

LL safety / diagnostics
- `-checklevel <k>`       Force GL check every ~B×k iters (B≈√p by default).
- `-gerbiczli`            Disable Gerbicz–Li checks (PRP/LL‑safe). Not recommended.
- `-llsafeb <B>`          Block size for `-llsafe2` (default ≈ √p).
- `-erroriter <i>`        Inject an error at iteration *i* to test detection.
- `-res64_display_interval <N>`  Show Res64 every N iterations (0=off).

P‑1
- `-pm1`                  P‑1 factoring mode.
- `-b1 <B1>`              Stage 1 bound.
- `-b2 <B2>`              Stage 2 bound.
- `-b1old <B1old>`        Extend an existing Stage 1 run (auto‑loads `.save` or `.p95`).
- `-resume`               After Stage 1/2, write GMP‑ECM `.save` and Prime95 `.p95` resumes.
- `-p95`                  After Stage 1/2, write Prime95 `.p95` only.
- `-filemers <path>`      Convert `<p>pm<B1>.mers` → GMP‑ECM `.save` (helper).
- `-K <K>`                Enable n^K (Crandall) Stage‑2 variant with K powers.
- `-nmax <n>`             Upper bound for the n^K variant.

ECM
- `-ecm`                  ECM on 2^p − 1.
- `-b1 <B1>`              Stage 1 bound.
- `-b2 <B2>`              Stage 2 bound (optional).
- `-K <curves>`           Number of curves.
- `-montgomery`           Use a Montgomery model.
- `-torsion16`            Force torsion‑16 (or `-notorsion` to disable).

Checkpoints and backup
- `-t <sec>`              Checkpoint interval (default: 120s).
- `-f <path>`             Directory for checkpoints (default: current).

PrimeNet / JSON
- `-submit`               Enable auto‑submission to mersenne.org.
- `--noask`               Submit without prompting.
- `-user <name>`          PrimeNet username.
- `-password <pwd>`       PrimeNet password (with `--noask`).
- `-computer <name>`      Computer name (PrimeNet).

Worktodo / config
- `-worktodo [path]`      Read GIMPS‑style `worktodo.txt` (first PRP= line).
- `-config <path>`        Read options from a config file.

Backend / expert
- `-marin`                Disable Marin; use internal NTT backend.
- More expert flags exist (local sizes, enqueue caps, etc.). See `-h`.


Web-based GUI
-------------

Start:

    ./prmers -gui -http 3131

- Default host: first non‑loopback IPv4; default port: 3131.
- Override host/port with `-host` and `-http`.
- Then open the printed URL, e.g. http://127.0.0.1:3131/

The GUI lets you:
- Monitor progress, residues and logs in real time.
- Build and edit `worktodo.txt` entries.
- Inspect results and settings.


worktodo.txt
------------

PrMers understands GIMPS‑style `PRP=` lines. Example:

    PRP=DEADBEEFCAFEBABEDEADBEEFCAFEBABE,1,2,197493337,-1,76,0;

This is k*b^n+c with k=1, b=2, n=197493337, c=−1 → the Mersenne 2^197493337−1.

Usage:

    ./prmers -worktodo
    ./prmers -worktodo ./worktodo.txt

- If no exponent is on the CLI and a valid `PRP=` is found, that exponent is used.
- Only the first valid `PRP=` line is read currently.


Gerbicz–Li Error Checking (PRP & LL safe)
-----------------------------------------

Principle
- Split long exponentiations into blocks of size B≈√p.
- Maintain a *current* rolling product and a *reference* product from the last
  verified checkpoint.
- Every ~10 minutes (tunable via `-checklevel`), recompute the theoretical product
  from the stored checkpoint, compare, and either advance the “last‑correct”
  marker or roll back and replay.

Persistence
- Files `.bufd`, `.lbufd`, `.gli`, `.isav`, `.jsav` and the state file contain
  everything needed for deterministic restart after a mismatch, crash, or Ctrl‑C.

Testing
- Inject an error to validate recovery:

    ./prmers 6972593 -erroriter 19500

Disabling
- Use `-gerbiczli` to disable (benchmarks only; not recommended for production).


P‑1 Factoring
-------------

Target: N = 2^p − 1 with q | N such that q = 2kp + 1.

Stage 1
- Choose B1. Let E = lcm(1,…,B1). Compute x = 3^(E·2p) mod N and g = gcd(x − 1, N).
- If 1 < g < N, a non‑trivial factor is found.
- **Export resumes** (optional): with `-resume` PrMers writes `resume_p<p>_B1_<B1>.save` (GMP‑ECM) and `resume_p<p>_B1_<B1>.p95` (Prime95). With `-p95`, write only the `.p95`.

Example:

    ./prmers 541 -pm1 -b1 8099
    ./prmers 541 -pm1 -b1 8099 -resume

**Extend Stage 1**
- To extend from B1old to a higher B1 using an existing `.save` or `.p95`:
  
    ./prmers 541 -pm1 -b1 20000 -b1old 8099

  The program auto‑detects the matching `resume_p<p>_B1_<B1old>.save` or `.p95` in the current directory.

Stage 2
- Choose B2 > B1.
- From Stage 1’s state H, compute

  Q = ∏_{q ∈ (B1,B2]} (H^q − 1) mod N,

  with standard optimizations (prime gaps, cached powers).
- **n^K (Crandall) variant**: enable with `-K <K>` and optionally bound exponents with `-nmax <n>`.
- **Export resumes** (optional): with `-resume`, PrMers writes `resume_p<p>_B1_<B1>_B2_<B2>.save` and `.p95` after Stage 2.

Example:

    ./prmers 367 -pm1 -b1 11981 -b2 38971
    ./prmers 367 -pm1 -b1 11981 -b2 38971 -resume
    ./prmers 367 -pm1 -b1 11981 -b2 38971 -K 8 -nmax 200000   # n^K variant

Implementations
- **GPU P‑1 using the Marin backend** (Stage 1 and Stage 2, including n^K).


ECM on Mersenne Numbers
-----------------------

    ./prmers p -ecm -b1 B1 -b2 B2 -K curves [curve options]

- Stage 1 bound B1, optional Stage 2 bound B2, K curves.
- Defaults: Edwards curve with torsion optimizations.
- Options: `-montgomery`, `-torsion16`, `-notorsion`, `-seed <val>`.
- Interoperability: P‑1 resumes use **GMP‑ECM**’s `.save` textual format in addition to Prime95’s `.p95` when requested via `-resume`.

GPU Memory Test (-memtest)
--------------------------

    ./prmers -memtest
    ./prmers -memtest -d 2

- Scans as much VRAM as possible (subject to device limits).
- Patterns include address‑derived values, inversion toggles, and modulo‑stride
  sequences with multiple offsets.
- Reports coverage, traffic, bandwidth and any detected errors.


NTT Transform Sizes
-------------------

For a given exponent p, PrMers chooses an NTT/IBDWT size N:

| Exponent p range | N | Structure |
|---|---:|---|
| 3–113 | 4 | 2^2 |
| 127–239 | 8 | 2^3 |
| 241–463 | 16 | 2^4 |
| 467–919 | 32 | 2^5 |
| 929–1153 | 40 | 5·2^3 |
| 1163–1789 | 64 | 2^6 |
| 1801–2239 | 80 | 5·2^4 |
| 2243–3583 | 128 | 2^7 |
| 3593–4463 | 160 | 5·2^5 |
| 4481–6911 | 256 | 2^8 |
| 6917–8629 | 320 | 5·2^6 |
| 8641–13807 | 512 | 2^9 |
| 13829–17257 | 640 | 5·2^7 |
| 17291–26597 | 1024 | 2^10 |
| 26627–33247 | 1280 | 5·2^8 |
| 33287–53239 | 2048 | 2^11 |
| 53267–66553 | 2560 | 5·2^9 |
| 66569–102397 | 4096 | 2^12 |
| 102407–127997 | 5120 | 5·2^10 |
| 128021–204797 | 8192 | 2^13 |
| 204803–255989 | 10240 | 5·2^11 |
| 256019–393209 | 16384 | 2^14 |
| 393241–491503 | 20480 | 5·2^12 |
| 491527–786431 | 32768 | 2^15 |
| 786433–982981 | 40960 | 5·2^13 |
| 983063–1507321 | 65536 | 2^16 |
| 1507369–1884133 | 81920 | 5·2^14 |
| 1884193–3014653 | 131072 | 2^17 |
| 3014659–3768311 | 163840 | 5·2^15 |
| 3768341–5767129 | 262144 | 2^18 |
| 5767169–7208951 | 327680 | 5·2^16 |
| 7208977–11534329 | 524288 | 2^19 |
| 11534351–14417881 | 655360 | 5·2^17 |
| 14417927–22020091 | 1048576 | 2^20 |
| 22020127–27525109 | 1310720 | 5·2^18 |
| 27525131–44040187 | 2097152 | 2^21 |
| 44040253–55050217 | 2621440 | 5·2^19 |
| 55050253–83886053 | 4194304 | 2^22 |
| 83886091–104857589 | 5242880 | 5·2^20 |
| 104857601–167772107 | 8388608 | 2^23 |
| 167772161–209715199 | 10485760 | 5·2^21 |
| 209715263–318767093 | 16777216 | 2^24 |
| 318767107–398458859 | 20971520 | 5·2^22 |
| 398458889–637534199 | 33554432 | 2^25 |
| 637534277–796917757 | 41943040 | 5·2^23 |
| 796917763–1207959503 | 67108864 | 2^26 |
| 1207959559–1509949421 | 83886080 | 5·2^24 |


Benchmarks
----------

PrMers performance depends on

- GPU model
- clock rates and power limits
- OpenCL driver
- code version and options

Numbers below are approximate and obtained on specific setups. Treat them as
order‑of‑magnitude guidance only. For more detail, see the Mersenne Forum thread:

    https://www.mersenneforum.org/node/1086124/page3

### Quick overview (PRP on Mersenne exponents, Marin backend)

PRP throughput for p ≈ 136,279,841 (PRP mode, Marin, auto NTT).

| GPU                                       | User / system                | PRMERS_SCORE | Iter/s @ p ≈ 1.36e8 | Approx PRP ETA | Notes                                  |
|-------------------------------------------|------------------------------|--------------|---------------------|----------------|----------------------------------------|
| NVIDIA GeForce RTX 5090                   | Resolver (vast.ai)           | n/a          | ≈ 2230              | ≈ 17 h         | High‑end NVIDIA Ada/Blackwell          |
| NVIDIA GeForce RTX 4090                   | Resolver                     | 100.00/100   | ≈ 1225              | ≈ 31 h         | Reference 100/100 score                |
| NVIDIA GeForce RTX 5070 Laptop            | beepthebee                   | 62.69/100    | ≈ 356               | ≈ 4.4–4.6 days | +200 MHz core, +500 MHz VRAM (OC)      |
| NVIDIA GeForce RTX 4060 Ti                | Lorenzo                      | 69.14/100    | ≈ 318               | ≈ 5 days       | Desktop midrange                       |
| NVIDIA GeForce RTX 4070 Laptop GPU        | Phantomas                    | 52.24/100    | ≈ 255               | ≈ 6 days       | Gaming laptop GPU                      |
| NVIDIA GeForce RTX 2060                   | hwt; Artoria2e5              | 45.76/100    | ≈ 240–259           | ≈ 5.9–6.7 days | Undervolt / power cap in some reports  |
| NVIDIA GeForce GTX 1660 Ti                | Phantomas (MSI GL73)         | n/a          | ≈ 234               | ≈ 6.8 days     | Older Turing GPU                       |
| AMD Radeon VII                            | cherubrock (author)          | 50.57/100    | ≈ 350               | ≈ 4.5 days     | Reference dev card                     |
| Apple M4 Pro (Mac mini / MacBook)         | wigglefruit                  | 30.29/100    | ≈ 164               | ≈ 9.6 days     | Apple silicon, 18‑core GPU             |
| Apple M2 (MacBook Air, 8 GB unified RAM)  | cherubrock (author)          | n/a          | ≈ 25                | ≈ 62 days      | Thin‑and‑light laptop                  |

All runs above:
- use the Marin backend in PRP mode;
- let PrMers choose NTT sizes automatically;
- were executed with reasonably tuned (not extreme) power settings.

Your results will vary with clocks, thermals, drivers and PrMers version.

PrimeNet Integration
--------------------

When a test finishes, PrMers writes a JSON file containing status, exponent,
worktype (PRP‑3, PM1, ECM, LL, …), residues (res64, res2048), errors (Gerbicz),
transform size, program/version, OS and checksum. Example filename:

    100003_prp_result.json

Results are also appended to `results.txt`.

If `-submit` and proper credentials are provided, PrMers can log in to
mersenne.org and send results automatically, e.g.:

    ./prmers -worktodo ./worktodo.txt \
             -submit --noask \
             -user my_login \
             -password my_password

If a result JSON remains unsent (e.g. after exit), PrMers will detect it at the
next startup and ask whether to submit.


Cleaning and Uninstall
----------------------

Clean build artifacts:

    make clean

Uninstall installed files:

    sudo make uninstall


Backend and code
----------------

- Marin backend by Yves Gallot  
  - https://github.com/galloty/marin

- Integer NTT / IBDWT techniques  
  - Based on ideas discussed by Nick Craig-Wood and others in the context of
    modular arithmetic for Mersenne numbers. In particular, NTT and IBDWT
    using modular arithmetic modulo 2^64 - 2^32 + 1.

- Gerbicz-Li proof scheme  
  - Used for PRP error checking in PrMers (see the paper in the "Must read papers"
    section below).

Related inspiration
-------------------

- GPUOwl (Preda)  
- Genefer22 (Yves Gallot)  
- GIMPS and the Mersenne Forum community  
- GMP-ECM and related work on elliptic curve factoring:  
  - https://gitlab.inria.fr/zimmerma/ecm

- Repositories by Yves Gallot containing many useful resources:  
  - https://github.com/galloty  
  - https://github.com/galloty/f12ecm  
  - https://github.com/galloty/FastMultiplication

- Work by Nick Craig-Wood:  
  - IOCCC 2012 entry: https://github.com/ncw/ioccc2012  
  - Armprime project: https://github.com/ncw/  
  - ARM Prime Math (background on the math behind Armprime):  
    https://www.craig-wood.com/nick/armprime/math/

Must read papers
----------------

### Multiplication by FFT

- Discrete Weighted Transforms and Large Integer Arithmetic  
  Richard Crandall and Barry Fagin, 1994  
  https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/S0025-5718-1994-1185244-1.pdf

- Rapid Multiplication Modulo the Sum And Difference of Highly Composite Numbers  
  Colin Percival, 2002  
  https://www.daemonology.net/papers/fft.pdf

### P-1 factoring

- An FFT Extension to the P-1 Factoring Algorithm  
  Peter L. Montgomery and Robert D. Silverman, 1990  
  https://www.ams.org/journals/mcom/1990-54-190/S0025-5718-1990-1011444-3/S0025-5718-1990-1011444-3.pdf

- Improved Stage 2 to P+/-1 Factoring Algorithms  
  Peter L. Montgomery and Alexander Kruppa, 2008  
  https://inria.hal.science/inria-00188192v3/document

### Proof schemes (Gerbicz-Li)

- An Efficient Modular Exponentiation Proof Scheme  
  Darren Li, Yves Gallot, 2022–2023  
  arXiv: https://arxiv.org/abs/2209.15623  

  Presents an efficient proof scheme for left-to-right modular exponentiation,
  generalizing the Gerbicz-Pietrzak approach to arbitrary exponents. It allows
  an = r (mod m) to be proven with overhead negligible compared to the
  exponentiation itself and has been deployed at PrimeGrid to validate long
  runs.

Author
------

Author of PrMers:

- cherubrock (Sebastien), with contributions and feedback from users on
  mersenneforum.org and GitHub.

For bug reports, feature requests, or contributions, please use:

    https://github.com/cherubrock-seb/PrMers/issues


    ### Example PRP throughput (Marin backend, PRP mode)

Below are some concrete examples for different GPUs and exponents. All are for
Mersenne numbers M_p = 2^p − 1 in PRP mode.

#### NVIDIA GeForce RTX 5090 (Resolver, vast.ai instance)

Transform sizes were chosen automatically by PrMers.

- p = 57 885 161, NTT size 8  
  - About 2350 iter/s, ETA around 6 h 50 min.
- p = 74 207 281, NTT size 8  
  - About 2230 iter/s, ETA around 9 h 15 min.
- p = 82 589 933, NTT size 8  
  - About 1970 iter/s, ETA around 11 h 40 min.
- p = 136 279 841, NTT size 8  
  - About 2230 iter/s, ETA around 17 h.

#### AMD Radeon VII (cherubrock, local dev machine)

- p = 57 885 161, NTT size 8  
  - About 510 iter/s, ETA around 31 h.
- p = 74 207 281, NTT size 8  
  - About 436 iter/s, ETA around 48 h.
- p = 82 589 933, NTT size 8  
  - About 402 iter/s, ETA around 52 h.
- p = 136 279 841, NTT size 8  
  - About 350 iter/s, ETA around 4.5 days.

#### NVIDIA GeForce RTX 4090 (Resolver)

- p = 57 885 161, NTT size 8  
  - About 1030 iter/s, ETA around 15 h.
- p = 74 207 281, NTT size 8  
  - About 910 iter/s, ETA around 22 h.
- p = 82 589 933, NTT size 8  
  - About 840 iter/s, ETA around 27 h.
- p = 136 279 841, NTT size 8  
  - About 1225 iter/s, ETA around 31 h.

#### NVIDIA GeForce RTX 4060 Ti (Lorenzo)

- p = 57 885 161, NTT size 8  
  - About 420 iter/s, ETA around 37 h.
- p = 74 207 281, NTT size 8  
  - About 366 iter/s, ETA around 55 h.
- p = 82 589 933, NTT size 8  
  - About 337 iter/s, ETA around 59 h.
- p = 136 279 841, NTT size 8  
  - About 318 iter/s, ETA just under 5 days.

#### NVIDIA GeForce RTX 4070 Laptop GPU (Phantomas)

- p = 57 885 161, NTT size 8  
  - About 370 iter/s, ETA around 42 h.
- p = 74 207 281, NTT size 8  
  - About 320 iter/s, ETA around 63 h.
- p = 82 589 933, NTT size 8  
  - About 283 iter/s, ETA around 71 h.
- p = 136 279 841, NTT size 8  
  - About 255 iter/s, ETA a bit over 6 days.

#### NVIDIA GeForce GTX 1660 Ti (Phantomas, MSI GL73 notebook)

- p = 57 885 161, NTT size 8  
  - About 330 iter/s, ETA around 47 h.
- p = 74 207 281, NTT size 8  
  - About 288 iter/s, ETA around 69 h.
- p = 82 589 933, NTT size 8  
  - About 262 iter/s, ETA around 76 h.
- p = 136 279 841, NTT size 8  
  - About 234 iter/s, ETA around 6.8 days.

#### NVIDIA GeForce RTX 2060 (hwt; Artoria2e5)

Typical ranges seen (power-capped / undervolted in some runs):

- p = 57 885 161  
  - ≈ 491–502 iter/s, ETA ≈ 1 d 7 h – 1 d 20 h.
- p = 74 207 281  
  - ≈ 499 iter/s, ETA ≈ 1 d 16 h.
- p = 82 589 933  
  - ≈ 499–502 iter/s, ETA ≈ 1 d 15 h – 1 d 20 h.
- p = 136 279 841  
  - ≈ 240–259 iter/s, ETA ≈ 5 d 21 h – 6 d 18 h.

#### NVIDIA GeForce RTX 5070 Laptop (beepthebee)

- p = 57 885 161, NTT size 8  
  - About 858 iter/s, ETA around 18 h 45 min.
- p = 74 207 281, NTT size 8  
  - About 882 iter/s, ETA around 1 d 0 h.
- p = 82 589 933, NTT size 8  
  - About 875 iter/s, ETA around 1 d 2 h.
- p = 136 279 841, NTT size 8  
  - About 356 iter/s, ETA around 4 d 10 h.

#### Apple M4 Pro (wigglefruit)

- p = 57 885 161, NTT size 8  
  - About 264 iter/s, ETA around 58 h.
- p = 74 207 281, NTT size 8  
  - About 231 iter/s, ETA around 87 h.
- p = 82 589 933, NTT size 8  
  - About 213 iter/s, ETA around 94 h.
- p = 136 279 841, NTT size 8  
  - About 164 iter/s, ETA around 9.6 days.

#### Apple M2 (MacBook Air 8 GB, cherubrock)

- p = 57 885 161, NTT size 8  
  - About 42 iter/s, ETA around 15 h.
- p = 74 207 281, NTT size 8  
  - About 38 iter/s, ETA around 25 h.
- p = 82 589 933, NTT size 8  
  - About 32 iter/s, ETA around 29 h.
- p = 136 279 841, NTT size 8  
  - About 25 iter/s, ETA around 62 days.
