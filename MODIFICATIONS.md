## v99.75 throughput-cost auto selection + experimental PFA9 lead bridge

- defaults PRP/LL Aevum selection to `throughput:auto`;
- selects `4:512:8:512:202` at M175000039 instead of the slower 4.50M PFA9 plan;
- keeps `-pfa-off` on the power-of-two-only `pow2:auto` selector;
- embeds Aevum v0.3.68;
- adds an opt-in PFA9 carryB/fftP lead bridge and word-exact GPU differential test.

## v99.74 power-of-two FFT323161 + PRPLL-style lead cache

- accepts explicit non-PFA FFT323161 plans, including `4:512:8:512:202`;
- embeds Aevum v0.3.67;
- preserves consecutive PRP squarings in `LEAD_WIDTH` state through the register adapter;
- flushes the one pending final square before every externally visible register operation;
- keeps PFA and Apple on their validated canonical paths;
- adds exact cached-versus-canonical GPU comparison and benchmark scripts.

