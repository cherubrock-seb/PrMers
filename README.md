PrMers: GPU-accelerated Mersenne Primality Testing
==================================================

PrMers is a high-performance GPU application for Lucas-Lehmer (LL), PRP, P-1 and ECM
testing of Mersenne numbers. It uses OpenCL and an integer NTT / IBDWT engine modulo

    p = 2^64 - 2^32 + 1

and is designed for long, reliable runs with checkpointing and optional PrimeNet
submission.

The project also supports PRP tests of cofactors and Wagstaff numbers, and includes
a web-based GUI and a GPU VRAM tester.


Overview of Algorithms and Backends
-----------------------------------

PrMers has two main computational backends:

- Marin backend (default)
  - External library: https://github.com/galloty/marin
  - Efficient modular exponentiation with Gerbicz-Li style error checking (in PRP).
  - Uses IBDWT-style transforms over Z / (2^64 - 2^32 + 1) Z.
  - Supports PRP, LL, P-1, and ECM on Mersenne numbers.

- Internal NTT backend
  - Integer NTT / IBDWT implementation inside PrMers.
  - Used when the Marin backend is disabled.
  - Option mostly for experimentation and comparison.

The backend is controlled by:

- Default: Marin backend enabled.
- `-marin`: disable Marin and use the internal NTT backend instead.


Supported Modes
---------------

Mersenne and related numbers
- Mersenne numbers:    N = 2^p - 1
- Wagstaff numbers:    W = (2^p + 1) / 3  (optional, via flag)
- Cofactors:           N / product(known factors), PRP-tested as generic integer

Main computational modes
- PRP (default)
  - Probable-prime test with Gerbicz-Li error checking.
  - Supports Mersenne, Wagstaff, and cofactors.
  - Final residue is reported (Res64 and full residue) and can be used as proof input.

- Lucas-Lehmer (LL)
  - Classical LL for Mersenne numbers.
  - Several LL modes exist internally (GPU-safe, GPU-unsafe / debugging, CPU-safe),
    mainly selected via `-ll`, `-llunsafe`, `-llsafe2`. See `./prmers -h` for details.

- P-1 factoring
  - Stage 1 and Stage 2, with optional Stage 3.
  - Targets factors q of 2^p - 1 with q - 1 having small prime factors.
  - Supports various implementations (Marin-based GPU, GMP-based CPU polynomial code).

- ECM
  - Elliptic Curve Method on Mersenne numbers N = 2^p - 1.
  - Supports multiple curve models (Edwards, Montgomery) and torsion variants.

- Wagstaff
  - With `-wagstaff`, the program tests the Wagstaff number (2^p + 1)/3 instead of Mersenne 2^p - 1.
  - All the same modes (PRP, P-1, ECM) can be applied to the Wagstaff modulus.


Requirements
------------

- OpenCL 1.2 runtime (OpenCL 2.0 recommended) and a supported GPU.
- C++20 compiler (g++ or clang++ on Linux/macOS; MSVC or MinGW on Windows).
- libcurl (PrimeNet HTTP client).
- GMP (used for CPU-side helpers, ECM, and some P-1 polynomial code).

Typical packages (Linux, Debian / Ubuntu):

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
- Simple build:

    make -j$(nproc)

- Install executable and kernels:

    sudo make install

This installs:
- Executable:      /usr/local/bin/prmers
- Kernel files:    /usr/local/share/prmers/

The build embeds a `KERNEL_PATH` pointing to the installation directory so that
PrMers can find its OpenCL kernels after installation.

Windows with CMake + vcpkg (recommended)
- Install CMake, Visual Studio, and vcpkg.
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

Windows with MSYS2 / MinGW
- Install MSYS2 (UCRT64).
- In the MSYS2 UCRT64 shell:

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
- Precompiled binaries for Linux, Windows and macOS are published on the
  GitHub Releases page:
    https://github.com/cherubrock-seb/PrMers/releases

macOS notes
- OpenCL is already present on supported macOS versions.
- On first run, Gatekeeper may block the binary because it is not notarized.
  Use "System Settings -> Security & Privacy" to allow it, then run again.


Quick Start
-----------

Basic PRP on a Mersenne exponent:

    ./prmers 136279841

- Mode: PRP (default).
- Backend: Marin (default).
- Checkpoint interval: 120 s (default).
- Results are appended to `results.txt` and to a JSON file
  `<exponent>_prp_result.json`.

Lucas-Lehmer test:

    ./prmers 127 -ll

P-1, stage 1 and stage 2:

    ./prmers 367 -pm1 -b1 11981 -b2 38971

ECM on a Mersenne number:

    ./prmers 701 -ecm -b1 6000 -K 8
    ./prmers 701 -ecm -b1 6000 -b2 33333 -K 8

Test a Wagstaff number W = (2^p + 1)/3:

    ./prmers 100003 -wagstaff

Use worktodo.txt:

    ./prmers -worktodo ./worktodo.txt

Use a config file:

    ./prmers -config ./settings.cfg

Disable Marin and use the internal backend:

    ./prmers 136279841 -marin


Command Line Options (summary)
------------------------------

For the full list, run:

    ./prmers -h

Below is a summary of commonly used options.

Positional
- `<p>`                   Exponent of the Mersenne or Wagstaff number.

Device and performance
- `-d <id>`               OpenCL device id (default: 0).
- `-O <flags>`            OpenCL compiler options, e.g. "fastmath mad".
- `-c <depth>`            Local carry propagation depth.
- `-profile`              Enable kernel profiling.
- `-memtest`              Run GPU VRAM test instead of a Mersenne computation.

Modes
- `-prp`                  Force PRP mode (default).
- `-ll`                   Lucas-Lehmer mode (GPU).
- `-llunsafe`             LL mode with relaxed safety (mostly for debugging).
- `-llsafe2`              Alternative LL mode (safer, different kernel).
- `-wagstaff`             Test the Wagstaff number (2^p + 1)/3.

P-1
- `-pm1`                  P-1 factoring mode.
- `-b1 <B1>`              Stage 1 bound.
- `-b2 <B2>`              Stage 2 bound.
- `-b1old <B1old>`        Extend an existing stage 1 run (resume / extend).
- Additional P-1 options exist for alternative implementations (e.g. GMP-based
  polynomial or gwnum-based P-1). See `./prmers -h` for these advanced flags.

ECM
- `-ecm`                  ECM on 2^p - 1.
- `-b1 <B1>`              ECM stage 1 bound.
- `-b2 <B2>`              ECM stage 2 bound (optional).
- `-K <curves>`           Number of curves for ECM.
- `-montgomery`           Use Montgomery curve model.
- `-torsion16`            Force torsion 16.
- `-notorsion`            Disable torsion optimization.
(Defaults and exact combinations are documented in `./prmers -h`.)

Checkpoints and backup
- `-t <sec>`              Checkpoint interval in seconds (default: 120).
- `-f <path>`             Directory that stores checkpoints (default: current).
- The program saves and resumes `.mers`, `.loop`, `.bufd`, `.lbufd`, `.gli`,
  `.isav`, `.jsav` and other state files as needed.

PrimeNet and JSON
- `-submit`               Enable PrimeNet submission support in this run.
- `--noask`               Auto-submit results without prompting.
- `-user <name>`          PrimeNet username.
- `-password <pwd>`       PrimeNet password (only with `--noask`).
- `-computer <name>`      Computer name for PrimeNet.
- Result JSON files are written next to the executable, for example:
    `100003_prp_result.json`

Worktodo and config
- `-worktodo [path]`      Read assignment from `worktodo.txt` style file.
- `-config <path>`        Read additional command line options from a config file.
- Precedence: explicit CLI options override config; config overrides worktodo.

Backend and internal options
- `-marin`                Disable Marin backend, use internal NTT backend.
- Several additional expert options exist (fine control of local sizes,
  enqueue limits, debug injection via `-erroriter`, etc.). Refer to `-h`.


Web-based GUI
-------------

PrMers includes a small HTTP server with a web GUI.

Start with:

    ./prmers -gui -http 3131

- Default host: first non-loopback IPv4 address.
- Default port: 3131.

You can override host and port, for example:

    ./prmers -gui -http 3131 -host 127.0.0.1
    ./prmers -gui -http 3131 -host localhost
    ./prmers -gui -http 8080 -host 192.168.1.99

Then open the browser at the displayed URL, for example:

    http://127.0.0.1:3131/

The GUI allows you to:
- Monitor progress, residues and logs in real time.
- Build `worktodo` lines from a form.
- Edit and append `worktodo.txt`.
- Inspect results and settings.


worktodo.txt
------------

PrMers can read assignments from a GIMPS-like `worktodo.txt` file.

Supported format
- Lines starting with `PRP=` are supported.
- Example:

    PRP=DEADBEEFCAFEBABEDEADBEEFCAFEBABE,1,2,197493337,-1,76,0;

This line represents the number k*b^n + c with k=1, b=2, n=197493337, c=-1,
that is the Mersenne number 2^197493337 - 1. PrMers extracts n and uses it as
the exponent.

Usage

    ./prmers -worktodo
    ./prmers -worktodo ./worktodo.txt

- If no exponent is given on the command line and a valid PRP= line is found,
  the exponent is taken from the worktodo file.
- If neither a positional exponent nor a valid worktodo entry is found, the
  program asks for an exponent interactively.

Only the first valid `PRP=` line is used at the moment.


Configuration Files
-------------------

Instead of a long command line, you can provide a config file:

Example `settings.cfg`:

    -d 1
    -O fastmath mad
    -c 16
    -profile
    -ll
    -t 300
    -f /home/user/checkpoints
    --noask
    -user myusername
    -worktodo ./tasks/worktodo.txt

Then run:

    ./prmers -config ./settings.cfg

Rules:
- Arguments in the config file are exactly as they would appear on the command line.
- The positional exponent given on the CLI (if any) has precedence.
- Then come CLI flags, then config flags, then worktodo.


Gerbicz-Li Error Checking (PRP)
-------------------------------

PrMers implements Gerbicz-Li style error checking in PRP mode
(see D. Li, Y. Gallot, "An Efficient Modular Exponentiation Proof Scheme").

Principle
- The long exponentiation is split into blocks of size B.
- Two rolling products are maintained:
  - a "current" product for the present progress;
  - a "reference" product stored at the last verified checkpoint.
- After about T seconds of work (T defaults to 600 seconds), the code performs
  a verification step:
  - recompute the theoretical product from the last checkpoint;
  - compare with the current product.
- On match, the last-correct marker is advanced.
- On mismatch, state is restored from the last checkpoint and work is repeated.

Persistence
- Files `.bufd`, `.lbufd`, `.gli`, `.isav`, `.jsav` and the main state file
  contain all data needed for deterministic restart after:
  - a mismatch;
  - a crash;
  - a manual interrupt.

Testing the checker
- You can inject a deliberate error at iteration `i` with:

    ./prmers 6972593 -erroriter 19500

- The program should detect the mismatch, roll back, and continue from the last
  verified point.

Disabling
- You can disable the Gerbicz-Li protection with `-gerbiczli` (for benchmarks).
  Use with care; this removes protection against silent errors.


P-1 Factoring
-------------

PrMers supports P-1 factoring of Mersenne numbers:

- N = 2^p - 1
- A factor q of N must satisfy q = 2kp + 1.
- P-1 will find q if k is B1-smooth (and possibly with one extra prime in stage 2).

Stage 1
- Choose a bound B1.
- Let E = lcm(1, 2, ..., B1).
- Compute x = 3^(E * 2p) mod N.
- Compute g = gcd(x - 1, N).
- If 1 < g < N, g is a non-trivial factor.

Example:

    ./prmers 541 -pm1 -b1 8099

Stage 2
- Choose B2 > B1.
- Starting from the end of stage 1, compute:

  Q = product over primes q in (B1, B2] of (H^q - 1) mod N,

  where H is the stage 1 state.
- Various optimizations cache small powers of H and use prime differences.

Example:

    ./prmers 367 -pm1 -b1 11981 -b2 38971

PrMers includes several implementations:
- GPU P-1 using the Marin backend.
- CPU / GMP-based polynomial P-1, including polynomial product and BSGS style
  stage 2, intended for future optimization.
Exact selection of the implementation is controlled by internal options and
additional flags (see the full help output).


ECM on Mersenne Numbers
-----------------------

ECM mode:

    ./prmers p -ecm -b1 B1 -b2 B2 -K curves [curve options]

- Target: N = 2^p - 1.
- Stage 1 bound: B1.
- Stage 2 bound: B2 (optional).
- Number of curves: K.

Curve options
- Default: Edwards curve, torsion 8 (fast on Mersenne numbers).
- `-montgomery`      Use a Montgomery model.
- `-torsion16`       Use torsion 16 when supported.
- `-notorsion`       Disable torsion optimizations.

Example:

    ./prmers 701 -ecm -b1 6000 -K 8
    ./prmers 701 -ecm -b1 6000 -b2 33333 -K 8
    ./prmers 701 -ecm -b1 6000 -K 8 -montgomery
    ./prmers 701 -ecm -b1 6000 -K 8 -torsion16


GPU Memory Test (-memtest)
--------------------------

PrMers includes a GPU VRAM tester to detect unstable cells, lanes, or address
decode issues.

Usage:

    ./prmers -memtest
    ./prmers -memtest -d 2

Behavior
- Scans as much VRAM as possible, subject to the device max allocation size.
- Runs several patterns:
  - address pattern (write address-derived value, then read back);
  - inversion toggles (x / ~x over many passes);
  - modulo-stride patterns with multiple offsets.

The final report shows:
- tested VRAM coverage;
- total traffic (GB read / written);
- aggregated bandwidth numbers;
- any detected errors.


NTT Transform Sizes
-------------------

For a given Mersenne exponent p, PrMers chooses an NTT / IBDWT transform size N.
The table below summarizes the mapping currently used in the code (p ranges
inclusive):

| Exponent p range | Transform size N | N structure |
|------------------|------------------|------------|
| 3-113            | 4                | 2^2        |
| 127-239          | 8                | 2^3        |
| 241-463          | 16               | 2^4        |
| 467-919          | 32               | 2^5        |
| 929-1153         | 40               | 5*2^3      |
| 1163-1789        | 64               | 2^6        |
| 1801-2239        | 80               | 5*2^4      |
| 2243-3583        | 128              | 2^7        |
| 3593-4463        | 160              | 5*2^5      |
| 4481-6911        | 256              | 2^8        |
| 6917-8629        | 320              | 5*2^6      |
| 8641-13807       | 512              | 2^9        |
| 13829-17257      | 640              | 5*2^7      |
| 17291-26597      | 1024             | 2^10       |
| 26627-33247      | 1280             | 5*2^8      |
| 33287-53239      | 2048             | 2^11       |
| 53267-66553      | 2560             | 5*2^9      |
| 66569-102397     | 4096             | 2^12       |
| 102407-127997    | 5120             | 5*2^10     |
| 128021-204797    | 8192             | 2^13       |
| 204803-255989    | 10240            | 5*2^11     |
| 256019-393209    | 16384            | 2^14       |
| 393241-491503    | 20480            | 5*2^12     |
| 491527-786431    | 32768            | 2^15       |
| 786433-982981    | 40960            | 5*2^13     |
| 983063-1507321   | 65536            | 2^16       |
| 1507369-1884133  | 81920            | 5*2^14     |
| 1884193-3014653  | 131072           | 2^17       |
| 3014659-3768311  | 163840           | 5*2^15     |
| 3768341-5767129  | 262144           | 2^18       |
| 5767169-7208951  | 327680           | 5*2^16     |
| 7208977-11534329 | 524288           | 2^19       |
| 11534351-14417881| 655360           | 5*2^17     |
| 14417927-22020091| 1048576          | 2^20       |
| 22020127-27525109| 1310720          | 5*2^18     |
| 27525131-44040187| 2097152          | 2^21       |
| 44040253-55050217| 2621440          | 5*2^19     |
| 55050253-83886053| 4194304          | 2^22       |
| 83886091-104857589| 5242880         | 5*2^20     |
| 104857601-167772107| 8388608        | 2^23       |
| 167772161-209715199| 10485760       | 5*2^21     |
| 209715263-318767093| 16777216       | 2^24       |
| 318767107-398458859| 20971520       | 5*2^22     |
| 398458889-637534199| 33554432       | 2^25       |
| 637534277-796917757| 41943040       | 5*2^23     |
| 796917763-1207959503| 67108864      | 2^26       |
| 1207959559-1509949421| 83886080     | 5*2^24     |

These values are used internally to select `TRANSFORM_SIZE_N` and related kernel
parameters for each exponent range.

Benchmarks
----------

PrMers performance depends on:

- GPU model
- clock rates and power limits
- OpenCL driver
- code version and options

The numbers below are approximate and were obtained on specific setups. They
are intended as order-of-magnitude guidance only.

For more detail, see the Mersenne Forum
discussion:

    https://www.mersenneforum.org/node/1086124/page3

### Quick overview (PRP on Mersenne exponents, Marin backend)

The table below compares the PRP throughput for a single large Mersenne
exponent (p = 136 279 841) in PRP mode. Values are approximate.

| GPU                                       | User / system                | PRMERS_SCORE | Iter/s @ p ≈ 1.36e8 | Approx PRP ETA | Notes                                  |
|-------------------------------------------|------------------------------|--------------|---------------------|----------------|----------------------------------------|
| NVIDIA GeForce RTX 5090                   | Resolver (vast.ai)           | n/a          | ≈ 2230              | ≈ 17 h         | High-end NVIDIA Ada/Blackwell          |
| NVIDIA GeForce RTX 4090                   | Resolver                     | 100.00/100   | ≈ 1225              | ≈ 31 h         | Reference 100/100 score                |
| NVIDIA GeForce RTX 5070 Laptop            | beepthebee                   | 62.69/100    | ≈ 356               | ≈ 4.4–4.6 days | +200 MHz core, +500 MHz VRAM (OC)      |
| NVIDIA GeForce RTX 4060 Ti                | Lorenzo                      | 69.14/100    | ≈ 318               | ≈ 5 days       | Desktop midrange                       |
| NVIDIA GeForce RTX 4070 Laptop GPU        | Phantomas                    | 52.24/100    | ≈ 255               | ≈ 6 days       | Gaming laptop GPU                      |
| NVIDIA GeForce RTX 2060                   | hwt, Artoria2e5              | 45.76/100    | ≈ 240–259           | ≈ 5.9–6.7 days | Undervolt / power cap in some reports  |
| NVIDIA GeForce GTX 1660 Ti                | Phantomas (MSI GL73)         | n/a          | ≈ 234               | ≈ 6.8 days     | Older Turing GPU                       |
| AMD Radeon VII                            | cherubrock (author)          | 50.57/100    | ≈ 350               | ≈ 4.5 days     | Reference dev card                     |
| Apple M4 Pro (Mac mini / MacBook)         | wigglefruit                  | 30.29/100    | ≈ 164               | ≈ 9.6 days     | Apple silicon, 18-core GPU             |
| Apple M2 (MacBook Air, 8 GB unified RAM)  | cherubrock (author)          | n/a          | ≈ 25                | ≈ 62 days      | Thin-and-light laptop                  |

All runs above:

- use the Marin backend in PRP mode;
- let PrMers choose NTT sizes automatically;
- were executed with reasonably tuned, but not extreme, power settings.

Your own results will differ depending on clocks, thermals, drivers, and PrMers
version.

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

These values may change between PrMers versions or with different OpenCL drivers and power
limits; treat them as indicative rather than absolute.


PrimeNet Integration
--------------------

When a test finishes, PrMers writes a JSON file containing:
- status (C/F);
- exponent;
- worktype (PRP-3, PM1, ECM, etc.);
- residues (res64, res2048);
- errors (Gerbicz);
- FFT length / transform size;
- program name and version;
- operating system;
- checksum.

Example filename:

    100003_prp_result.json

Results are also appended to `results.txt`.

If `-submit` and the required credentials are provided, PrMers can log in to
mersenne.org and send the results automatically:

    ./prmers -worktodo ./worktodo.txt \
             -submit --noask \
             -user my_login \
             -password my_password

If a result JSON file remains unsent (for example if the program was closed),
PrMers will detect it at next startup and ask whether to submit it.


Cleaning and Uninstall
----------------------

Clean build artifacts:

    make clean

Uninstall (installed files only):

    sudo make uninstall


Credits
-------

PrMers brings together work from several sources.

Credits
-------

PrMers brings together work from several sources.

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