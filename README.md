# PrMers: GPU-accelerated Mersenne primality test

## Compilation
To compile:

```bash
g++ -std=c++11 -o prmers prmers.cpp -lOpenCL
```

## Usage
Example of execution:
```bash
./prmers 9279 -d 1
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
./prmers 216091
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

## Example Executions

1. **Using backup option with a 10-second interval:**
   
   Command:

  ```bash
   ./prmers 216091 -t 10
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
   ./prmers 127 -prp
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
   ./prmers 756839 -prp
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

## Compilation
To compile:
```bash
g++ -std=c++11 -o prmers prmers.cpp -lOpenCL
'''
## Usage
