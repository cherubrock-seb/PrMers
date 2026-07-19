# Native Aevum PFA radix-3/radix-9 exp8

This revision keeps the optimized mixed-radix path and enables both radix 3 and
radix 9 in `pfa:auto` when the measured transform-reduction gate is met. It also
rebuilds host-side syntax helpers on the current machine so archives remain
portable between Linux and Apple Silicon.

The stock non-PFA source path is unchanged. See the main README section
`native-pfa-automatic-selection-ranges` for the exact exponent windows.
