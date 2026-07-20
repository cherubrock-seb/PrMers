## v99.74 power-of-two FFT323161 + PRPLL-style lead cache

- accepts explicit non-PFA FFT323161 plans, including `4:512:8:512:202`;
- embeds Aevum v0.3.67;
- preserves consecutive PRP squarings in `LEAD_WIDTH` state through the register adapter;
- flushes the one pending final square before every externally visible register operation;
- keeps PFA and Apple on their validated canonical paths;
- adds exact cached-versus-canonical GPU comparison and benchmark scripts.

