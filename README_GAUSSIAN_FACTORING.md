# Gaussian-Mersenne P-1 and ECM — PrMers v99.91

This extension factors the Gaussian-Mersenne norm

```text
G_p = 2^p - (2/p) * 2^((p+1)/2) + 1
```

for a prime exponent `p`.  It is strictly opt-in through `-gm-pm1` and
`-gm-ecm`.  Ordinary Mersenne PRP/LL/P-1/ECM dispatch, kernels and Aevum plans
are unchanged.

## Exact lifted arithmetic

For every Gaussian-Mersenne norm:

```text
G_p | 2^(2p) + 1
G_p | 2^(4p) - 1
```

The GPU therefore performs additions, subtractions, squarings and
multiplications in the existing, highly optimized Mersenne ring
`Z/(2^(4p)-1)Z`.  The natural map from that ring to `Z/G_p Z` is a ring
homomorphism, so every lifted polynomial computation projects exactly to the
wanted Gaussian norm.

The CPU is used only at mathematically meaningful boundaries:

* export one or two GPU registers;
* reduce the exported integer modulo `G_p` with GMP;
* compute a GCD or an inverse modulo `G_p`;
* re-embed a normalized value when ECM Stage 2 continues.

No division or inverse is attempted modulo the larger lifted modulus.  This is
important because a value can be invertible modulo `G_p` while sharing a
factor with the unrelated cofactor `(2^(4p)-1)/G_p`.

## P-1 mode

```bash
./prmers P -gm-pm1 -b1 B1 [-b2 B2] -aevum -d DEVICE
```

Stage 1 computes

```text
h = base^lcm(lcm(1,...,B1), 4p)
```

in the lifted ring, then computes `gcd(h-1,G_p)` on the CPU.  The structural factor
`4p` is free structural information: for every prime factor `q != 5` of
`G_p`, the order of 2 modulo `q` is `4p`, hence `4p | q-1`.  This can make a
small `B1` substantially stronger than a generic P-1 run.

Stage 2 is a low-memory product-exponent continuation.  Primes in `(B1,B2]`
are multiplied into bounded chunks and each chunk is applied with prepared
Aevum multiplication.  A single CPU projection/GCD is performed per chunk.
This avoids the unsafe inversions that would arise from blindly running the
ordinary Mersenne Stage 2 against the lifted cofactor ring.

Useful options:

```text
-gm-base A                 P-1 base, default 3
-gm-factor-chunk-bits N    Stage 2 product target, default 262144 bits
-gm-safe                   independent Stage 1 blocks and Stage 2 chunks
-gm-replay-block N         override Stage 1 replay block size
-r                         resume CRC32 checkpoints
-t SECONDS                 ordinary PrMers checkpoint interval where applicable
-f DIRECTORY               result/checkpoint directory
```

## ECM mode

```bash
./prmers P -gm-ecm -b1 B1 [-b2 B2] -K CURVES -aevum -d DEVICE
```

The implementation uses Suyama Montgomery curves and a projective `x/z`
ladder.  Curve setup and inversions are performed modulo `G_p` with GMP; all
ladder additions, doublings, squarings and multiplications use Aevum through
the exact `2^(4p)-1` lift.

Stage 1 multiplies by `lcm(1,...,B1)`.  At the boundary, `gcd(Z,G_p)` is
computed on the CPU.  Stage 2 uses product-exponent chunks.  After each chunk,
the point is projected, `gcd(Z,G_p)` is checked, the affine x-coordinate is
normalized modulo `G_p`, and the point is re-embedded for the next chunk.

Useful options:

```text
-K CURVES                  number of Suyama curves
-sigma VALUE               force the first 64-bit sigma (reproducible tests)
-seed VALUE                deterministic seed for subsequent curves
-gm-factor-chunk-bits N    Stage 2 product target, default 131072 bits
-gm-safe                   replay full Stage 1 and every Stage 2 chunk
-r                         resume Stage 1 and Stage 2 CRC32 checkpoints
-f DIRECTORY               result/checkpoint directory
```

## Known validation cases

```bash
# P-1 Stage 1: G_13 = 8321 = 53 * 157
./prmers 13 -gm-pm1 -b1 2 -gm-sieve 0 -aevum -d 1 -f ./gm-factor-tests/p13

# P-1 Stage 2: G_23 = 277 * 30269; the extra Stage 2 prime is 3
./prmers 23 -gm-pm1 -b1 2 -b2 3 -gm-sieve 0 -aevum -d 1 -f ./gm-factor-tests/p23

# ECM Stage 1: deterministic Suyama curve finds 137 in G_17
./prmers 17 -gm-ecm -b1 50 -K 1 -sigma 7 -gm-sieve 0 -aevum -d 1 -f ./gm-factor-tests/ecm17-s1

# ECM Stage 2: Stage 1 B1=2 is clean, the extra prime 3 finds 137
./prmers 17 -gm-ecm -b1 2 -b2 3 -K 1 -sigma 14 -gm-sieve 0 -aevum -d 1 -f ./gm-factor-tests/ecm17-s2

# Same ECM test with independent GPU replay
./prmers 17 -gm-ecm -b1 2 -b2 3 -K 1 -sigma 14 -gm-safe -gm-sieve 0 -aevum -d 1 \
  -f ./gm-factor-tests/ecm17-safe
```

## Larger candidates

Factoring bounds are workload choices, not claims that the candidates are
unfactored.  Examples above the currently listed largest prime exponent:

```bash
# Fast structural P-1 pass
./prmers 15317251 -gm-pm1 -b1 100000 -b2 5000000 -aevum -d 1 \
  -gm-factor-chunk-bits 262144 -r -f ./gm-results/15317251-pm1

# Medium ECM campaign
./prmers 15317251 -gm-ecm -b1 50000 -b2 5000000 -K 50 -aevum -d 1 \
  -gm-factor-chunk-bits 131072 -r -f ./gm-results/15317251-ecm

# Deeper P-1 pass
./prmers 16000057 -gm-pm1 -b1 1000000 -b2 100000000 -aevum -d 1 \
  -gm-factor-chunk-bits 524288 -r -f ./gm-results/16000057-pm1
```

The lift uses exponent `4p`, so memory and transform size are comparable to a
Mersenne calculation at exponent `4p`, not `p`.  A future native sparse
trinomial backend could reduce that cost, but this version deliberately favors
mathematical exactness and zero kernel regression.

## Result files

```text
gm_pm1_p<P>_stage1.ckpt
gm_pm1_p<P>_stage2.ckpt
gm_pm1_p<P>_result.json
gm_ecm_p<P>_c<CURVE>_stage1.ckpt
gm_ecm_p<P>_c<CURVE>_stage2.ckpt
gm_ecm_p<P>_result.json
results.txt
```

Checkpoints use an atomic `.new/.old` rotation and CRC32.  They bind the mode,
phase, exponent, bounds, base/curve, sigma, scalar size and engine checkpoint
size, preventing accidental resume with incompatible parameters.

## Performance notes

* P-1 Stage 1 is the fastest path: one lifted exponentiation and one final CPU
  reduction/GCD.
* P-1 Stage 2 exports only once per configurable chunk.
* ECM remains projective throughout each Stage 1/chunk; CPU inversion happens
  only between chunks.
* `-gm-safe` approximately doubles arithmetic and should be used for validation
  or confirmation, not initial high-throughput screening.
* Aevum is selected through the ordinary workload-aware plugin policy.  No
  Aevum source or kernel was changed by this extension.

## v99.89 Stage 2 progress

P-1 Stage 2 now synchronizes at a light 4096-bit cadence and prints an update
roughly every five seconds inside every product chunk:

```text
GM P-1 Stage 2: 12.34% | primes 8500/68906 | chunk 1 56.70% |
bits 148635/262154 | bit-IPS 651.22 | elapsed 228.41 s | ETA 00:26:59
```

The synchronization cadence is deliberately much smaller than the 262144-bit
chunk while remaining coarse enough not to disturb throughput materially. A
Ctrl-C inside a chunk restores the state at the beginning of that chunk and
writes a clean resumable checkpoint.

The harmless Aevum `Read ZERO` burst has also been removed. All checkpointed
P-1 scratch registers are initialized before the first save. v99.89 uses
checkpoint format version 3, so an old v99.88 checkpoint is intentionally not
loaded.

## Uniform JSON schema v1

Every Gaussian-Mersenne P-1, ECM, PRP and Proth result now uses the same
machine-readable schema. Bounds are strings to avoid precision loss in JSON
consumers:

```json
{
  "schema_version": 1,
  "program": "PrMers",
  "program_version": "v99.89",
  "family": "gaussian-mersenne",
  "mode": "gm-pm1",
  "outcome": "factor",
  "stage": 1,
  "exponent": 45951761,
  "B1": "100000",
  "B2": null,
  "curves": null,
  "sigma": null,
  "factor": "19916401959425537",
  "backend": "Aevum",
  "device": "NVIDIA RTX 3080",
  "elapsed_seconds": 123.450,
  "timestamp": "2026-07-27T18:00:00Z"
}
```

Additional fields such as `program_build`, `curve`, `factor_source`, `base`,
`jacobi`, `res64` and `res2048` may be present depending on the mode.

## Native Gaussian-Mersenne worktodo

These lines are distinct from Prime95 assignments and are parsed only by
PrMers:

```text
GMPROTH=p[,sieve_limit]
GMPRP=p[,sieve_limit]
GMPMINUS1=p,B1,B2[,base[,sieve_limit[,chunk_bits]]]
GMECM=p,B1,B2,curves[,sigma[,sieve_limit[,chunk_bits]]]
GMCHAIN=p,pm1_B1,pm1_B2[,ecm_B1[,ecm_B2[,curves[,sieve_limit[,chunk_bits]]]]]
```

`GMCHAIN` is conditional. It runs P-1 first, optionally ECM, and then the
deterministic Proth test only if no factor was found. A completed line is
removed from the active worktodo and appended to `worktodo_save.txt`. An
interrupted or erroneous line remains first and resumes from its checkpoint.
Comments beginning with `#` or `;` are preserved.

Example record-search line without ECM:

```text
GMCHAIN=45951761,100000,1000000,0,0,0,1000000000000,262144
```

Example with two small ECM curves before Proth:

```text
GMCHAIN=45951761,100000,1000000,2000,0,2,1000000000000,262144
```

Generate a prime-exponent queue and run it with:

```bash
./scripts/generate_gaussian_worktodo.py --start 45951681 --count 20 \
  --output ./gm-record/worktodo-gm.txt --mode chain \
  --pm1-b1 100000 --pm1-b2 1000000 --sieve 1000000000000

./prmers -worktodo ./gm-record/worktodo-gm.txt -aevum -d 1 -r \
  -f ./gm-record/results
```

The convenience wrapper `scripts/run_gm_record_campaign.sh` generates the
queue when absent and launches the same resumable workflow.

A ready-to-edit queue example is provided in `WORKTODO_GAUSSIAN_EXAMPLE.txt`.
