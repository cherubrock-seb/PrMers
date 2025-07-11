# PrMers: GPU-accelerated Mersenne Primality Test

PrMers is a high-performance GPU application for testing the primality of Mersenne numbers using the Lucas–Lehmer and PRP (Probable Prime) algorithms. It leverages OpenCL and Number Theoretic Transforms (NTT) for fast large-integer arithmetic, and is optimized for long-running computations.

Key features:
- ⚡ GPU acceleration using OpenCL (including legacy OpenCL 1.2 devices)
- 🔁 Automatic checkpointing with resume support
- 📤 Submit results directly to your [mersenne.org](https://www.mersenne.org/) PrimeNet account
- 🖥️ Runs on Linux, macOS, and Windows (build from source or use precompiled binaries)
- 📊 Benchmark output for performance comparison across devices

`prmers` supports [P−1 factoring](https://en.wikipedia.org/wiki/Pollard%27s_p_%E2%88%92_1_algorithm) stage 1, a powerful algorithm to find a prime factor \( q \) of a Mersenne number \( 2^p - 1 \), provided that \( q - 1 \) is composed entirely of small prime factors.
See below

## 🚀 Try the PrMers Demo on Google Colab

You can test **PrMers** directly in your browser with GPU acceleration by opening the interactive notebook below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/cherubrock-seb/PrMers/blob/main/prmers.ipynb)


### 📈 Sample Performance Results

#### 🔥 Radeon VII GPU

```
| Exponent  | Iter/s  | ETA             |
|-----------|---------|-----------------|
| 136279841 | 290.77  | 5d 10h 10m 27s  |
| 82589933  | 544.85  | 1d 18h 5m 45s   |
| 74207281  | 544.38  | 1d 15h 46m 49s  |
| 57885161  | 552.32  | 1d 5h 6m 18s    |
```

#### 🍏 MacBook Air 2022 (Apple M2)

```
| Exponent  | Iter/s  | ETA              |
|-----------|---------|------------------|
| 136279841 | 31.16   | 50d 14h 57m 35s  |
| 82589933  | 50.88   | 18d 18h 52m 10s  |
| 77232917  | 51.17   | 17d 11h 16m 9s   |
| 74207281  | 51.15   | 16d 18h 58m 41s  |
| 57885161  | 51.00   | 13d 3h 16m 30s   |
```

#### 🧊 INTEL(R) HD Graphics Family Integrated (113 MB) (OpenCL 1.2, legacy GPU - Windows) 


```
| Exponent  | Iter/s  | ETA              |
|-----------|---------|------------------|
| 136279841 | 2.46    | 637d 18h 5m 0s   |
| 1257787   | 245.44  | 0d 1h 24m 0s     |
| 756839    | 249.23  | 0d 0h 49m 53s    |
```

For a full table of benchmark results, see the section below 👇



## Features

- GPU-accelerated Mersenne prime testing using OpenCL
- Implementation of NTT / Lucas–Lehmer algorithms / PRP / IBDWT
- Automatic backup of computation state with resume support
- Command-line options for performance tuning and debugging

## Requirements

PrMers runs on both **Linux** and **Windows** systems with OpenCL support.

To build PrMers with GMP support you must install the GMP library on your system before compiling:

• macOS (Homebrew):  
  brew install gmp

• Ubuntu/Debian:  
  sudo apt-get update  
  sudo apt-get install -y libgmp-dev

• Windows (vcpkg):  
  git clone https://github.com/microsoft/vcpkg.git  
  cd vcpkg  
  .\bootstrap-vcpkg.bat  
  .\vcpkg install gmp:x64-windows


### ✅ Common Requirements
- A GPU supporting **OpenCL 1.2** or higher (**OpenCL 2.0** recommended)
- OpenCL-compatible drivers installed
- Libcurl

### 🐧 On Linux
- `g++` with C++20 support (e.g., GNU g++ 10+)
- OpenCL development libraries:
  - `ocl-icd-opencl-dev`
  - `opencl-headers`
- Git (to clone the repository)
- libcurl4-openssl-dev

Example on ubuntu : 
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers libcurl4-openssl-dev

### 🪟 On Windows
- A compiler supporting C++20 (e.g., MSVC or MinGW64)
- OpenCL SDK (e.g., [Khronos OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK))
- CMake (if building via CMake)
- Git for Windows (optional, if cloning directly)
- Libcurl

### 🍎 On macOS
- Xcode (Command Line Tools)
- OpenCL is **preinstalled** (no setup required)
- `g++` or `clang++` with C++20 support
- Compatible with macOS Big Sur or later
- libcurl

> ✅ macOS builds are supported and automatically generated in each release.

Alternatively, **precompiled binaries** are available from the [Releases](https://github.com/cherubrock-seb/PrMers/releases) page.



## P-1 Factoring with `prmers`


`prmers` supports [P−1 factoring](https://en.wikipedia.org/wiki/Pollard%27s_p_%E2%88%92_1_algorithm) stage 1, a powerful algorithm to find a prime factor \( q \) of a Mersenne number \( 2^p - 1 \), provided that \( q - 1 \) is composed entirely of small prime factors.

A factor \( q \) of \( 2^p - 1 \) must satisfy \( q \equiv 1 \mod 2p \), i.e. \( q = 2kp + 1 \) for some integer \( k \). The P−1 method will find such a factor if \( k \) is B1-smooth — meaning all its prime divisors are less than a chosen bound \( B_1 \).

The algorithm proceeds as follows:

1. Choose a smoothness bound \( B_1 \)
2. Build an exponent \( E = \text{lcm}(1, 2, \ldots, B_1) \)
3. Compute \( x = 3^{E \cdot 2p} \mod (2^p - 1) \)
4. Compute \( \gcd(x - 1, 2^p - 1) \)

If the result is a nontrivial factor, the algorithm succeeds.

To run a stage-1 P−1 factoring test on exponent 541 with bound B1 = 8099:

```bash
./prmers 541 -pm1 -b1 8099
```

Sample output if a factor is found:

```
Start a P-1 factoring stage 1 up to B1=8099
GCD(x - 1, 2^541 - 1) = 4312790327
P-1 factor stage 1 found: 4312790327
```

If no factor is found up to the chosen bound, the program reports:
```
No P-1 factor up to B1=8099
```

This stage-1 P−1 test is GPU-accelerated using optimized NTT-based exponentiation modulo \( 2^n - 1 \).



## ⚙️ Example Execution and Submission

Below is a typical run of `PrMers`.

```
sebastien@cherubrock:~/prmers$ ./prmers 100003
No valid entry found in worktodo.txt
GPU Vendor: ADVANCED MICRO DEVICES, INC.
Device on-device queue preferred=262144  max=8388608
Queue size = 16384
Max CL_DEVICE_MAX_WORK_GROUP_SIZE = 256
Max CL_DEVICE_MAX_WORK_ITEM_SIZES = 1024, 1024, 1024
Max CL_DEVICE_LOCAL_MEM_SIZE = 65536 bytes
Transform Size = 4096
No valid entry found in worktodo.txt
Building OpenCL program with options:  -DWG_SIZE=16 -DLOCAL_PROPAGATION_DEPTH=8 -DCARRY_WORKER=512 -DLOCAL_PROPAGATION_DEPTH_DIV4=2 -DLOCAL_PROPAGATION_DEPTH_DIV4_MIN=1 -DLOCAL_PROPAGATION_DEPTH_DIV2=2 -DLOCAL_PROPAGATION_DEPTH_DIV2_MIN=3 -DWORKER_NTT=1024 -DWORKER_NTT_2_STEPS=256 -DMODULUS_P=100003 -DTRANSFORM_SIZE_N=4096 -DLOCAL_SIZE=256 -DLOCAL_SIZE2=256 -DLOCAL_SIZE3=256
OpenCL program built successfully from: /home/sebastien/prmers/kernels/prmers.cl
PrMers : GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas Lehmer)
Testing exponent : 100003
Device OpenCL ID : 0
Mode : PRP
Backup interval : 30 s
Save/Load path: .
Looking for loop file at "/home/sebastien/prmers/./100003prp.loop"
No valid loop file, initializing fresh state
Initial x[0] set to 3 (PRP mode)
Progress: 0.00% | Exp: 100003 | Iter: 0 | Elapsed: 0.00s | IPS: 0.00 | ETA: 0d 0h 0m 0s | RES64: 
Progress: 8.19% | Exp: 100003 | Iter: 8194 | Elapsed: 2.15s | IPS: 3812.20 | ETA: 0d 0h 0m 24s | RES64: FE049C7B28CF9800
Progress: 16.39% | Exp: 100003 | Iter: 16389 | Elapsed: 4.30s | IPS: 3812.10 | ETA: 0d 0h 0m 21s | RES64: CBE8841CD0DA8728
Progress: 24.58% | Exp: 100003 | Iter: 24584 | Elapsed: 6.44s | IPS: 3812.38 | ETA: 0d 0h 0m 19s | RES64: 6E9F608CA08C14A4
Progress: 32.78% | Exp: 100003 | Iter: 32779 | Elapsed: 8.59s | IPS: 3812.66 | ETA: 0d 0h 0m 17s | RES64: ED1AEED6832C638B
Progress: 40.97% | Exp: 100003 | Iter: 40974 | Elapsed: 10.74s | IPS: 3813.06 | ETA: 0d 0h 0m 15s | RES64: 6A03925E314AFFFF
Progress: 49.17% | Exp: 100003 | Iter: 49169 | Elapsed: 12.84s | IPS: 3814.62 | ETA: 0d 0h 0m 13s | RES64: BF436FB1EE7DA640
Progress: 57.36% | Exp: 100003 | Iter: 57364 | Elapsed: 14.99s | IPS: 3815.80 | ETA: 0d 0h 0m 11s | RES64: D72B40192A12F3D0
Progress: 65.56% | Exp: 100003 | Iter: 65559 | Elapsed: 17.13s | IPS: 3816.84 | ETA: 0d 0h 0m 9s | RES64: E04C07F04F534E24
Progress: 73.75% | Exp: 100003 | Iter: 73754 | Elapsed: 19.28s | IPS: 3817.66 | ETA: 0d 0h 0m 6s | RES64: 03AAC604F7FBB83A
Progress: 81.95% | Exp: 100003 | Iter: 81949 | Elapsed: 21.42s | IPS: 3818.39 | ETA: 0d 0h 0m 4s | RES64: 39198F5DCB0F078C
Progress: 90.14% | Exp: 100003 | Iter: 90144 | Elapsed: 23.57s | IPS: 3819.00 | ETA: 0d 0h 0m 2s | RES64: 64D9882C2A10ECB1
Progress: 98.34% | Exp: 100003 | Iter: 98339 | Elapsed: 25.72s | IPS: 3819.39 | ETA: 0d 0h 0m 0s | RES64: C2C6ED57A9EAC999
Progress: 100.00% | Exp: 100003 | Iter: 100003 | Elapsed: 26.16s | IPS: 3819.71 | ETA: 0d 0h 0m 0s | RES64: 1CF45E9503C71FD6

Warning: Cannot open file for MD5: 

M100003 PRP test: composite.

Manual submission JSON:
{"status":"C","exponent":100003,"worktype":"PRP-3","res64":"1CF45E9503C71FD6","res2048":"af262d00ed00a05d53e99d0e0e451b12405ddabe139fe8396a4c520b505bb65bed1609d3c8ef23bbb1d0f8140a6bcdd2c67f9c8aa3bd0e6eeb3e8e79db904810c88de09820557176b389290f84f18424efa6a59fb9f132a74f53a83ba6e2f508c617a5e1451c3ee08d179e6614026f973d1900602f2068a08894cd81ed5035de9ded85909b1ee6ff4dc723118b79d3f940272ae1066aebe27c86338ad7edf70e76c0e8abf3e985b73db2a06f1b742a9a908728be2bd4b7daa2d6aafc11bacaaa40944e9a66b039cb0deaaa8e5e357cd54b81b3ec6661d55e48bacb994bfd3cbb33f3f01d82347fa00578ec86c4cd7eb568a1463cf3e38dae1cf45e9503c71fd6","residue-type":1,"errors":{"gerbicz":0},"fft-length":4096,"shift-count":0,"program":{"name":"prmers","version":"0.1.0","port":8},"os":{"os":"Linux","architecture":"x86_64"},"user":"cherubrock","timestamp":"2025-04-27 22:01:04","checksum":{"version":1,"checksum":"445C4880"}}

Total elapsed time: 26.16 seconds.
JSON result written to: ./100003_prp_result.json
Result appended to: ./results.txt
```



**No result is lost**, and every completed test can be credited to your PrimeNet account.

# 📄 worktodo.txt support

This program now supports reading assignments from a `worktodo.txt` file, similar to how Prime95 or other GIMPS tools operate.

## ✅ Supported format

Only lines beginning with the `PRP=` prefix are currently supported.

Each `PRP=` line follows this format:

PRP=assignment_id,k,b,n,c[,how_far_factored,tests_saved][,known_factors]

Where:
- `k × bⁿ + c` defines the number to test.
- For Mersenne numbers, this is typically: `k=1`, `b=2`, `c=-1`.
- `n` is the exponent (this is the value your program will extract and test).
- Other fields are optional and currently ignored.

### 📌 Example
PRP=DEADBEEFCAFEBABEDEADBEEFCAFEBABE,1,2,197493337,-1,76,0;

This line instructs the program to test the Mersenne number \( 2^{197493337} - 1 \).

## 🔍 Usage

You can either:
- Provide the exponent manually on the command line:

```bash
./prmers 197493337
```

Or use a worktodo.txt file (default path: ./worktodo.txt):

./prmers -worktodo

Or specify a custom path to your worktodo file:

./prmers -worktodo /path/to/worktodo.txt

If no exponent is passed and no -worktodo is provided, the program will prompt you to enter one interactively.

Note: Only the first valid PRP= line is used for now. Multi-line batching may be added later.


## 🔧 Using a Configuration File (`-config`)

You can run `prmers` using a configuration file to avoid passing long arguments on the command line every time.

### 📄 Example `setting.cfg`

```txt
-d 1
-O fastmath mad
-c 16
-profile
-ll
-t 300
-f /home/user/checkpoints
-l1 256
-l2 128
-l3 64
--noask
-user myusername
-worktodo ./tasks/worktodo.txt
```

### 🚀 How to use

You can launch `prmers` with this configuration file using:

```bash
./prmers -config ./setting.cfg
```

This will load all the parameters listed in `setting.cfg` as if you had typed them directly on the command line.

### 📌 Supported use cases

- ✅ If an exponent is **given on the command line**, it takes priority:
  
  ```bash
  ./prmers 333 -config ./setting.cfg
  ```

- ✅ If **no exponent is given**, but `-worktodo <path>` is present (in config or command line), the exponent is read from the `worktodo.txt` file:

  ```bash
  ./prmers -config ./setting.cfg
  # → uses the exponent found in tasks/worktodo.txt
  ```

- ✅ If no exponent and no valid `worktodo.txt` entry are found, the program will **prompt you to enter the exponent manually**.

### ⚠️ Notes

- The config file must contain arguments **exactly as they would be typed on the command line**, space-separated.
- You can still override values from the config file by passing extra arguments on the command line.


## 🔐 Submitting Results to PrimeNet

PrMers supports direct submission of results to [PrimeNet](https://www.mersenne.org) using your personal account. If you don’t have an account yet, it’s free and quick to create one at:  
👉 **https://www.mersenne.org**

After a test completes, PrMers will prompt you to submit your result to PrimeNet:

```
✅ JSON result written to: ./86243_prp_result.json
Do you want to send the result to PrimeNet (https://www.mersenne.org) ? (y/n): y
Enter your PrimeNet username (Don't have an account? Create one at https://www.mersenne.org)
Enter your PrimeNet username : cherubrock
Enter your PrimeNet password: 
[TRACE] Sending login with user: cherubrock
[TRACE] Login response size: 32966 bytes...
[TRACE] Server response size: 13644 bytes
✅ Server response:...

📝 Parsed PrimeNet Result Summary:
 Manually check in your results Found 1 lines to process at 2025-04-11T11:13:30 Results for M 86 243 ignored, it is a known Mersenne Prime and no further testing is required. 
Done processing: 
* Parsed 1 lines.
* Found 0 datestamps.
GHz-days Qty Work Submitted Accepted Average 
1 PRP (Probable Prime): PRIME 0.000 - 0.000 
1 - all - 0.000 
Did not understand 0 lines. 
Recognized, but ignored 0/1 of the remaining lines. 
Skipped 0 lines already in the database. 
Accepted 1 lines.
```

Once accepted, the result is marked as sent:

```
✅ Result successfully sent to PrimeNet.
```

If for any reason a result was not submitted (e.g. skipped, disconnected, closed), PrMers will automatically prompt you to re-submit the result later on launch.

```bash
Found unsent result: ./86243_prp_result.json
Do you want to submit it to PrimeNet now? (y/n)
```

This ensures **you never lose credit** for a completed computation.


## 💾 Download Precompiled Binaries (Linux, Windows & macOS)

You can download **precompiled binaries** for **Linux**, **Windows**, and **macOS** directly from the [Releases page](https://github.com/cherubrock-seb/PrMers/releases).

Each release contains:
- A compiled executable: `prmers` (Linux/macOS) or `prmers.exe` (Windows)
- Required OpenCL kernel files inside a `kernels/` folder
- A SHA256 checksum file to verify integrity

All releases are built automatically using **GitHub Actions**, a continuous integration system provided by GitHub. This ensures that the binaries are **consistently built** from the source code in the repository.

You can view the exact build process and logs by visiting:
👉 [GitHub Actions for PrMers](https://github.com/cherubrock-seb/PrMers/actions)

This setup guarantees that even users without development tools can safely download and verify that the binaries match the public source code.

### 🍏 macOS: Allow Execution of the Binary

On macOS, running unsigned executables downloaded from the internet will trigger **Gatekeeper**, blocking the file by default.

If you see this error:

> “prmers” cannot be opened because the developer cannot be verified.

Follow these steps to allow it:

1. Attempt to run `prmers` once (via Terminal or double-click).
2. Open **System Preferences → Security & Privacy → General**.
3. At the bottom, you will see a message:
   > “prmers” was blocked from use because it is not from an identified developer.
4. Click **"Allow Anyway"**.
5. Run the file again from Terminal:
   ```bash
   ./prmers
   ```
6. A dialog will appear asking for confirmation. Click **"Open"**.

✅ The binary will now execute normally in the future.

#### (Optional) Local code signing

If you're a developer or wish to suppress some security warnings (e.g., `killed: 9`), you can locally self-sign the binary:

```bash
codesign -s - --deep --force --timestamp --options runtime ./prmers
```

> Note: This is *not* a notarized signature, but it may prevent runtime security blocks.

To distribute a notarized macOS app without this issue, an Apple Developer Account is required.

## Installation from sources

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cherubrock-seb/PrMers.git
   cd PrMers
2. **Compile the software:**
   ```bash
   make
   ```
3. **Install the executable and kernel file:**
The following command installs the executable to /usr/local/bin and the OpenCL kernel file to /usr/local/share/prmers.
  ```bash
   sudo make install
  ```
The Makefile compiles the executable with a compile-time macro (KERNEL_PATH) so that, after installation, PrMers automatically finds its kernel file.

## Compilation
Once installed, you can run PrMers directly from the command line. The basic syntax is:
  ```bash
prmers <p> [options]
  ```
for example 
  ```bash
prmers 127 -O fastmath mad -c 16 -profile -ll -t 120 -f /your/backup/path
  ```

# 🛠️ Building PrMers on Windows (Manual Instructions)

This guide explains how to **build PrMers manually on Windows**, with different options.

---

## ⚙️ Option 1: **CMake + vcpkg + Visual Studio (RECOMMENDED)**

### 1. Install prerequisites:
- [CMake](https://cmake.org/download/)
- [Visual Studio](https://visualstudio.microsoft.com/downloads/) (with **C++ Desktop Development** and **OpenCL** support)
- [vcpkg](https://github.com/microsoft/vcpkg)

---

### 2. Clone the repository:
```sh
git clone https://github.com/cherubrock-seb/PrMers.git
cd PrMers
```

---

### 3. Install libcurl via vcpkg:
```sh
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install curl:x64-windows
cd ..
```

---

### 4. Configure CMake:
```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

---

### 5. Build the project:
```sh
cmake --build build --config Release
```

---

### 6. Fix missing DLLs:
- Copy the DLLs from `vcpkg\installed\x64-windows\bin\` into the folder with `prmers.exe`
  - OR add this path to your system `PATH` environment variable.

Typical required DLLs:
- `libcurl-4.dll`
- `libssh2-1.dll`
- `libnghttp2-14.dll`
- `libbrotlicommon.dll`
- `libwinpthread-1.dll`
- `libstdc++-6.dll`
- `libgcc_s_seh-1.dll`
- `libidn2-0.dll`

---

## ⚙️ Option 2: **MinGW / MSYS2 + Makefile**

### 1. Install MSYS2: [https://www.msys2.org/](https://www.msys2.org/)

### 2. Update MSYS2:
```sh
pacman -Syu
```

---

### 3. Install required packages:
```sh
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencl-icd-loader mingw-w64-x86_64-curl
```

---

### 4. Build using Makefile:
```sh
make
```

### 5. Run from the MSYS2 shell (it uses the proper environment for DLLs).

---

## ⚙️ Option 3: **Visual Studio IDE (CMake Project)**

1. Open **Visual Studio**.
2. Create a **new CMake Project**.
3. Point it to the PrMers directory.
4. In `CMakeSettings.json`, add:
```json
{
  "variables": [
    {
      "name": "CMAKE_TOOLCHAIN_FILE",
      "value": "C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
    }
  ]
}
```
5. Install libcurl with vcpkg.
6. Build directly from Visual Studio.

---

## 📝 Notes:
- If you get **libcurl-4.dll not found**, make sure the DLLs from vcpkg are accessible.
- You can **add** the path to DLLs to **`PATH`** or **copy** them next to `prmers.exe`.

---

## 🔧 Optional: Provide this batch file to automate:
Create `build_windows.bat`:
```bat
@echo off
echo Setting up vcpkg and building prmers...

if not exist "vcpkg" (
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..
)

.\vcpkg\vcpkg install curl:x64-windows

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

echo Build complete! Check the build\Release folder.
pause
```

---

## 🎉 DONE!
You should now have `prmers.exe` ready to run!

## Command-Line Options

- `<p>`: Minimum exponent to test (required)
- `-d <device_id>`: Specify the OpenCL device ID (default: 0)
- `-O <options>`: Enable OpenCL optimization flags (e.g., `fastmath`, `mad`, `unsafe`, `nans`, `optdisable`)
- `-c <localCarryPropagationDepth>`: Set the local carry propagation depth (default: 8)
- `-profile`: Enable kernel execution profiling
- `-prp`: Run in PRP mode (default), with an initial value of 3 and no execution of `kernel_sub2` (final result must equal 9)
- `-ll`: Run in Lucas–Lehmer mode, with an initial value of 4 and p-2 iterations of `kernel_sub2`
- `-t <backup_interval>`: Specify the backup interval in seconds (default: 60)
- `-f <path>`: Specify the directory path for saving/loading backup files (default: current directory)
- `-enqueue_max <value>`: Manually set the maximum number of enqueued kernels before `clFinish` is called (default: autodetect)
- `--noask`: Automatically submit results to PrimeNet without prompting
- `-user <username>`: PrimeNet account username to use for automatic result submission
- `-computer <computername>`: computer name to use for PrimeNet result submission
- `password <password>` : PrimeNet account password (used only when -no-ask is set, used to automatically put the result in primenet without prompt).
- `-worktodo [path]`: Use a `worktodo.txt` file to load an exponent automatically.
  - If no path is given, the default `./worktodo.txt` is used.
  - Only the first valid `PRP=` line is processed.
- `-config <path>`: Load configuration from a specified `.cfg` file instead of passing options manually


📌 Typical Usage Scenarios

This program is designed for flexible, automated testing of Mersenne numbers using OpenCL. The most common use case involves running multiple tests in sequence using a `worktodo.txt` file, with automatic result submission enabled:

- You can specify a list of exponents in `worktodo.txt` and run the program with `--noask`, `-user <username>`, and `-password <password>` to automatically submit results to PrimeNet without interaction.
- For advanced control, backup files can be stored at a custom path using `-f`, and fine-grained OpenCL settings like local sizes or profiling can be adjusted with `-l1`, `-l2`, `-l3`, or `-profile`.
- Command-line overrides (`-O`, `-c`, etc.) allow tailoring the run for performance tuning or testing alternative kernel behaviors.
- You may also run single tests manually by specifying the exponent as a positional argument.

This enables full automation on servers or multi-GPU rigs with minimal supervision.



## Uninstallation
To uninstall PrMers, run:
```bash
sudo make uninstall
```

## Cleaning Up
To remove compiled files from the build directory, run:

```bash
make clean
```

# Ubuntu Setup for OpenCL Development

Below is an example of the commands you need to install the necessary components to compile C++ code with OpenCL support and to run PrMers on Ubuntu. This includes installing the OpenCL headers and ICD loader, as well as the GPU drivers for NVIDIA, AMD, and Intel.

# Update package lists
sudo apt-get update

# Install OpenCL development packages and tools
sudo apt-get install ocl-icd-opencl-dev opencl-headers clinfo

For NVIDIA GPUs 
Install the proprietary NVIDIA driver (adjust version as needed) along with its OpenCL ICD.

sudo apt-get install nvidia-driver-525 nvidia-opencl-icd-525

For AMD GPUs 
Download the AMDGPU-PRO driver package from AMD's website.
Then install it with OpenCL support using the following command (run in the extracted directory):

./amdgpu-pro-install -y --opencl=pal,legacy

For Intel GPUs 

sudo apt-get install intel-opencl-icd

Verify your OpenCL installation by running:
clinfo


## Another option for manual compilation
To compile:

```bash
g++ -std=c++20 -I. -o prmers prmers.cpp proof/common.cpp proof/proof.cpp proof/md5.cpp proof/sha3.cpp -lOpenCL -O3 -Wall
```

## Usage
Example of execution:
```bash
prmers 9279 -d 1
PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
Testing exponent: 9279
Using OpenCL device ID: 1

Launching OpenCL kernel (p = 9279); computation may take a while depending on the exponent.
Max global workers possible: 256
Final workers count: 256
Progress: 100.00% | Elapsed: 11.09s | Iterations/sec: 836.78 | ETA: 0.00s       
M9279 is composite.
Kernel execution time: 11.09 seconds
Iterations per second: 836.77 (9277 iterations in total)
```

Or without specifying a device:
```bash
prmers 216091
PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
Testing exponent: 216091
Using OpenCL device ID: 0

Launching OpenCL kernel (p = 216091); computation may take a while depending on the exponent.
Max global workers possible: 256
Final workers count: 256
Progress: 0.11% | Elapsed: 4.05s | Iterations/sec: 57.50 | ETA: 3754.28s
```

## Output Example

```bash
PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
Testing exponent: 13
Using OpenCL device ID: 1

Launching OpenCL kernel (p_min_i = 13) without progress display; computation may take a while depending on the exponent.

Lucas-Lehmer test results:
Mp with p = 13 is a Mersenne prime.
Kernel execution time: 0.00150793 seconds
Iterations per second: 7294.76 (11 iterations in total)
 ```

## New Features

- **Modes:**
  - **-prp**: Run in PRP mode (default).  
    Sets the initial value to 3 and performs *p* iterations without executing kernel_sub2.  
    The final result must equal 9.
  - **-ll**: Run in Lucas-Lehmer mode.  
    Sets the initial value to 4 and performs *p-2* iterations, including the execution of kernel_sub2.

- **Backup / Save and Resume:**
  - The program periodically saves its state (the contents of buf_x) and the current loop iteration into files.
  - Use **-t \<backup_interval>** to specify the backup interval in seconds (default: 60 seconds).
  - Use **-f \<path>** to specify the directory for saving and loading files (default: current directory).
  - If backup files (for example, "127prp.mers" and "127prp.loop") exist in the specified path, the program will resume computation from the saved iteration.
  - During computation, when a backup occurs (either because the backup interval has elapsed or the user presses Ctrl-C), a status line is displayed with the same progress information as the normal progress display but in a distinct color (MAGENTA) and prefixed by "[Backup]".

## Proof Generation (Experimental)

This project includes an **experimental** implementation of proof generation for verifying Mersenne prime tests.  
The proof system is **heavily based on** the one used in [GpuOwl](https://github.com/preda/gpuowl), with substantial code reuse.  

The proof process generates a `.proof` file, which contains:
- The final residue after all PRP iterations.
- A sequence of intermediate residues at exponentially spaced iteration points.
- A verification mechanism that ensures `B == A^(2^span) (mod 2^E - 1)`.

Currently, verification is **not fully stable** and needs further debugging.  
Performance is also significantly slower than GpuOwl, as optimizations are still in progress.  



## 📊 Full Performance Table on Radeon VII

```
| Exponent  | Iter/s  | ETA             |
|-----------|---------|-----------------|
| 136279841 | 275.75  | 5d 17h 15m 57s  |
| 82589933  | 473.57  | 2d 0h 25m 47s   |
| 77232917  | 474.60  | 1d 21h 11m 22s  |
| 74207281  | 473.48  | 1d 19h 31m 18s  |
| 57885161  | 473.77  | 1d 9h 55m 28s   |
| 43112609  | 469.56  | 1d 1h 29m 25s   |
| 42643801  | 471.80  | 1d 1h 5m 35s    |
| 37156667  | 849.48  | 0d 12h 8m 10s   |
| 32582657  | 889.28  | 0d 10h 9m 49s   |
| 30402457  | 888.15  | 0d 9h 29m 41s   |
| 25964951  | 895.79  | 0d 8h 2m 15s    |
| 24036583  | 886.68  | 0d 7h 30m 58s   |
| 20996011  | 1117.04 | 0d 5h 12m 26s   |
| 13466917  | 1150.65 | 0d 3h 14m 13s   |
| 6972593   | 1783.91 | 0d 1h 4m 18s    |
| 3021377   | 1833.43 | 0d 0h 26m 37s   |
| 2976221   | 1837.81 | 0d 0h 26m 9s    |
| 1398269   | 2226.52 | 0d 0h 9m 38s    |
| 1257787   | 2292.34 | 0d 0h 8m 18s    |
| 859433    | 2257.62 | 0d 0h 5m 30s    |
| 756839    | 2238.93 | 0d 0h 4m 48s    |
| 216091    | 2540.35 | 0d 0h 0m 35s    |
| 132049    | 3299.74 | 0d 0h 0m 0s     |
| 110503    | 2872.99 | 0d 0h 0m 0s     |
| 86243     | 3732.72 | 0d 0h 0m 0s     |
| 44497     | 4223.44 | 0d 0h 0m 0s     |
```


## Example Executions

1. **Using backup option with a 10-second interval:**
   
   Command:

  ```bash
   prmers 216091 -t 10
  ```

   Output:

  ```bash
   PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
   Testing exponent: 216091
   Using OpenCL device ID: 0
   Mode selected: PRP
   Backup interval: 10 seconds
   Save/Load path: .
   Max CL_DEVICE_MAX_WORK_GROUP_SIZE = 256
   Max CL_DEVICE_MAX_WORK_ITEM_SIZES = 1024

   Launching OpenCL kernel (p = 216091); computation may take a while.
   Transform size: 16384
   Final workers count: 16384
   Work-groups count: 256
   Work-groups size: 64
   Workers for carry propagation count: 2048
   Local carry propagation depht: 8
   Local size carry: 256
   Building OpenCL program with options:  -DWG_SIZE=64 -DLOCAL_PROPAGATION_DEPTH=8 -DCARRY_WORKER=2048
   Resuming from iteration 27938 based on existing file ./216091prp.loop
   Loaded state from ./216091prp.mers
   Progress: 15.56% | Elapsed: 5.00s | Iterations/sec: 6724.69 | ETA: 27.13s       
   State saved to ./216091prp.mers
   Loop iteration saved to ./216091prp.loop
   [Backup] Progress: 18.16% | Elapsed: 10.00s | Iterations/sec: 3923.30 | ETA: 45.09s       
   Progress: 20.76% | Elapsed: 15.00s | Iterations/sec: 2989.41 | ETA: 57.28s       
   State saved to ./216091prp.mers
   Loop iteration saved to ./216091prp.loop
   [Backup] Progress: 23.37% | Elapsed: 20.00s | Iterations/sec: 2524.39 | ETA: 65.60s       
   Progress: 26.00% | Elapsed: 25.00s | Iterations/sec: 2246.94 | ETA: 71.17s       
   [Backup] Progress: 28.61% | Elapsed: 30.00s | Iterations/sec: 2060.63 | ETA: 74.86s       
   Progress: 31.25% | Elapsed: 35.01s | Iterations/sec: 1928.95 | ETA: 77.02s       
   [Backup] Progress: 33.91% | Elapsed: 40.00s | Iterations/sec: 1831.86 | ETA: 77.96s       
   Progress: 36.57% | Elapsed: 45.01s | Iterations/sec: 1756.06 | ETA: 78.05s       
   [Backup] Progress: 39.23% | Elapsed: 50.00s | Iterations/sec: 1695.30 | ETA: 77.46s       
   Progress: 41.83% | Elapsed: 55.01s | Iterations/sec: 1643.42 | ETA: 76.48s       
   [Backup] Progress: 44.44% | Elapsed: 60.00s | Iterations/sec: 1600.28 | ETA: 75.03s       
   Progress: 47.10% | Elapsed: 65.01s | Iterations/sec: 1565.57 | ETA: 73.02s       
   ^C
   State saved to ./216091prp.mers
   Loop iteration saved to ./216091prp.loop
   [Backup] Progress: 49.62% | Elapsed: 69.87s | Iterations/sec: 1534.73 | ETA: 70.93s       
   Exiting early due to interrupt.

  ```

2. **PRP mode with default backup interval:**
   
   Command:

  ```bash
   prmers 127 -prp
  ```

   Output:
  ```bash
   PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
   Testing exponent: 127
   Using OpenCL device ID: 0
   Mode selected: PRP
   Backup interval: 60 seconds
   Save/Load path: .
   Max CL_DEVICE_MAX_WORK_GROUP_SIZE = 256
   Max CL_DEVICE_MAX_WORK_ITEM_SIZES = 1024

   Launching OpenCL kernel (p = 127); computation may take a while.
   Transform size: 16
   Final workers count: 16
   Work-groups count: 4
   Work-groups size: 4
   Workers for carry propagation count: 2
   Local carry propagation depht: 8
   Local size carry: 2
   Building OpenCL program with options:  -DWG_SIZE=4 -DLOCAL_PROPAGATION_DEPTH=8 -DCARRY_WORKER=2
   Progress: 100.00% | Elapsed: 0.03s | Iterations/sec: 4102.95 | ETA: 0.00s       
   M127 PRP test succeeded (result is 9).
   Kernel execution time: 0.03 seconds
   Iterations per second: 4098.09 (127 iterations in total)
  ```
3. **PRP mode with a larger exponent:**
   
   Command:
  ```bash
   prmers 756839 -prp
  ```   
   Output:
  ```bash
   PrMers: GPU-accelerated Mersenne primality test (OpenCL, NTT, Lucas-Lehmer)
   Testing exponent: 756839
   Using OpenCL device ID: 0
   Mode selected: PRP
   Backup interval: 60 seconds
   Save/Load path: .
   Max CL_DEVICE_MAX_WORK_GROUP_SIZE = 256
   Max CL_DEVICE_MAX_WORK_ITEM_SIZES = 1024

   Launching OpenCL kernel (p = 756839); computation may take a while.
   Transform size: 65536
   Final workers count: 65536
   Work-groups count: 256
   Work-groups size: 256
   Workers for carry propagation count: 8192
   Local carry propagation depht: 8
   Local size carry: 256
   Building OpenCL program with options:  -DWG_SIZE=256 -DLOCAL_PROPAGATION_DEPTH=8 -DCARRY_WORKER=8192
   Progress: 6.21% | Elapsed: 55.01s | Iterations/sec: 854.12 | ETA: 831.10s       
   State saved to ./756839prp.mers
   Loop iteration saved to ./756839prp.loop
   [Backup] Progress: 6.79% | Elapsed: 60.00s | Iterations/sec: 856.94 | ETA: 823.10s       
   Progress: 7.30% | Elapsed: 65.01s | Iterations/sec: 850.30 | ETA: 825.08s       
   ```
## Notes
- The code uses OpenCL for GPU acceleration.
- It implements both PRP and Lucas-Lehmer (LL) tests.
- State saving is performed periodically based on the backup interval.
- When interrupted (Ctrl-C), the program saves its state before exiting, allowing you to resume later.
- The project implements an integer-based NTT and IBDWT using modular arithmetic modulo 2^64 - 2^32 + 1.
- The chosen modulus enables fast modular reduction using only bit shifts and additions.
- For more details on the underlying techniques, refer to Nick Craig-Wood's ARM Prime Math:
  https://www.craig-wood.com/nick/armprime/math/

## Resources & Community
- **GIMPS (Great Internet Mersenne Prime Search):** https://www.mersenne.org
- **Mersenne Forum:** https://www.mersenneforum.org

## Inspiration
- Nick Craig-Wood's IOCCC2012 entry: https://github.com/ncw/ioccc2012
- Armprime project: https://github.com/ncw/
- Genefer by Yves Gallot: https://github.com/galloty/genefer22
- GPUOwl: https://github.com/preda/gpuowl

**Author:** Cherubrock
