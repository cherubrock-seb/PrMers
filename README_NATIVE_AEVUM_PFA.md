# PrMers v99.70 + Aevum native PFA exp8

This revision embeds Aevum v0.3.63 with automatic radix-3 and radix-9 PFA
selection. The root bundle supplies the supported workflow:

```bash
./setup_and_test.sh DEVICE 2
./prmers EXPONENT -aevum -d DEVICE
```

`-pfa 3`, `-pfa 9`, and `-pfa-off` remain available for controlled comparisons.
The exact automatic ranges are listed in the main README under
`native-pfa-automatic-selection-ranges`.
