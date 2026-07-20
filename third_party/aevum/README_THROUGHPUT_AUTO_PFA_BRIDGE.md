# Aevum v0.3.68 — throughput auto selector and experimental PFA9 bridge

`throughput:auto` compares admissible FFT3161, PFA3/PFA9 and power-of-two
FFT323161 plans using an iteration-cost score rather than transform length
alone.  For M175000039 with the validated RTX 3080 calibration it selects:

```
4:512:8:512:202
```

The PFA9 bridge is not enabled automatically.  It attempts to retain the
width transform by applying carryB during the next Good-Thomas fftP gather.
It must first pass:

```
AEVUM_PFA_LEAD_BRIDGE=1 bash scripts/test_pfa9_lead_bridge_ubuntu.sh 1 175000039
```

Required marker:

```
PFA9 FFT3161 LEAD-BRIDGE DIFFERENTIAL TEST PASSED
```

Then compare it against the canonical path.  Disable it with
`AEVUM_PFA_LEAD_BRIDGE=0` or by omitting the variable.
