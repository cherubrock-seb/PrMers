# RTX 5090 / GB202 measured tuning candidate

A contributor reported a repeatable 236 us -> 203 us PRP improvement near exponents 146M-150M using:

```
-fft 1:512:8:512:202 -use INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0
```

This is intentionally not a global default: FFT shape and `-use` tuning are device, driver and exponent-range specific. Before shipping an automatic GB202 entry, capture the complete generated `tune.txt`, exact device name/PCI identity, driver version, and cross-check PRP/LL residues at both range boundaries.

Reproduce with:

```bash
./aevum -tune ntt,minexp=146000000,maxexp=150000000
```
