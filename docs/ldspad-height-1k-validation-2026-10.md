# LDSPAD_H=0 validation for the 1K:8:512 FFT3161 geometry

## Summary

Aevum now defaults `LDSPAD_H=0` only for the validated ordinary paired M31/M61 geometry:

- FFT type: `FFT3161`
- width: `1024`
- middle: `8`
- height: `512`
- variant: `202`
- plan form: `1:1K:8:512:202`

The policy uses `try_emplace`, so an explicit `-use` or per-FFT `LDSPAD_H` value still wins.

The change is intentionally not global. A separate width-512 screen at p=197M was negative on RTX 3080 and approximately neutral on Radeon VII, so those shapes remain unchanged.

## Exactness

The scoped candidate produced identical residues to `LDSPAD_H=1` on the validated 1K geometry.

Final host validation at p=210000017:

| GPU | Production default | Legacy override | Result |
| --- | --- | --- | --- |
| RTX 3080 | `LDSPAD_H=0` | `LDSPAD_H=1` | exact PASS |
| Radeon VII | `LDSPAD_H=0` | `LDSPAD_H=1` | exact PASS |

The same p=210000017 reference digest was observed on both GPUs:

`edfd7d0c10eecf97fad3550b094726a7b09765cbe1bcebb6a01511c46791b0ef`

RTX p=219999919 exactness also passed.

## Performance

All numbers below compare the scoped `LDSPAD_H=0` candidate with the legacy `LDSPAD_H=1` behavior using the same fixed plan.

### Final production-candidate run

| GPU / exponent | Iterations | Paired gains | Median-of-times gain |
| --- | ---: | --- | ---: |
| RTX 3080, p=210000017 | 3 x 10000 | +0.648730%, +0.344146%, +0.699798% | +0.344146% |
| RTX 3080, p=219999919 | 3 x 10000 | +1.009373%, +0.763078%, +0.960935% | +0.804941% |
| Radeon VII, p=210000017 | 3 x 5000 | +3.059012%, +2.879358%, +3.488522% | +3.287126% |

Every final paired gain was positive and every production gate passed.

### Earlier confirmation runs

The earlier screens were consistent with the final result:

- RTX 3080 p=210000017, initial 10k screen: +0.644746%, +1.037022%, +0.748472%.
- RTX 3080 p=210000017, 20k confirmation: +0.360316%, +1.019954%, +0.773264%.
- RTX 3080 p=219999919, 10k confirmation: +0.788891%, +0.454647%, +0.356001%.
- Radeon VII p=210000017, 10k confirmation: +3.816609%, +7.045260%, +2.266373%.

## Profiling observations

On RTX 3080 at p=210000017, the scoped change improved the two tail-square kernels most clearly:

- `tailSquareGF31`: about 5.2% faster.
- `tailSquareGF61`: about 4.9% faster.
- `carryFused`: about 1.5% slower.
- middle kernels: mostly neutral to slightly slower.

The end-to-end iteration still improved.

On Radeon VII, the largest improvement was in `tailSquareGF61`; some GF31 and middle work became slower, but total iteration time improved. A representative profiled run moved from about 6.294 s to 5.985 s for the same work.

## Scope decision

`LDSPAD_H=0` is therefore enabled only for `FFT3161 1K:8:512 variant 202`.

It is not enabled globally and it is not extended to the screened width-512 geometry.

`LDSPAD_H` was also added to the shared `-use` registry so explicit validation and override remain available without rebuilding.

## Integration validation

The PrMers candidate passed:

- full PrMers build;
- bundled Aevum host suite;
- Aevum adapter regression;
- auto-policy and default-backend tests;
- Aevum source audit;
- Apple portability source audit;
- RTX 3080 and Radeon VII exact p=210000017 checks.

The standalone Aevum candidate also passed `engine-lib`, the full `test-host` suite, and exact p=210000017 checks on both RTX 3080 and Radeon VII.

A standard 39-case PrMers backend matrix completed with 37 PASS and two unrelated failures:

- `prp-medium-aevum`: transform 262144, outside the 4194304-point `1K:8:512` geometry, with `required-result-pattern-missing`.
- `pm1-ultralow-marin`: forced Marin path, outside Aevum, with `expected-backend-pattern-missing`.

Neither failure exercises the new `LDSPAD_H=0` policy.

## Verdict

KEEP the scoped production default for `FFT3161 1:1K:8:512:202`.

Do not generalize the setting to other geometries without a separate exactness and paired-performance validation.
