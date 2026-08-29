# Gaussian-Mersenne ECM Special4096 - mathematical implementation note

## Status

`-gm-ecm-special4096` is an isolated, GM-only **experimental 4096-target**
portfolio. It does not replace ordinary Suyama ECM or Special32.

The target is

\[
v_2(\#E(\mathbf F_q))\ge 12,\qquad 2^{12}=4096.
\]

There is currently **no universal proof** that the finite profile portfolio
covers every unknown prime factor q. The name describes the target, not a
coverage theorem.

## 1. GM input available without q

For

\[
G_p=2^p-\left(\frac2p\right)2^{(p+1)/2}+1
\]

define

\[
I=(-1)^{(p+1)/2}2^p,\qquad z=1+I.
\]

Modulo every prime factor q of G_p,

\[
I^2=-1,\qquad z^p=1.
\]

A profile chooses r and sets

\[
t=z^r.
\]

This is computable directly modulo the composite G_p.

## 2. Positive-rank family and genuine ECM point

Use the twisted-Edwards family

\[
e=\frac{3(t^2-1)}{8t},\qquad d=-e^4,
\]

\[
-x^2+y^2=1+d x^2y^2.
\]

The published parametrization gives

\[
x_E=(4e^3+3e)^{-1},
\]

\[
y_E=\frac{9t^4-2t^2+9}{9t^4-9}.
\]

The generic torsion is C2 x C4 and this point is non-torsion. This is crucial:
an arbitrary Montgomery x can land on the quadratic twist and lose the desired
2-adic curve-order bias. Here the point belongs to the intended curve by
construction.

## 3. Montgomery conversion

For the corresponding Montgomery model

\[
Bv^2=u^3+Au^2+u
\]

one obtains

\[
A_{24}=\frac{A+2}{4}=\frac1{1+d}=\frac1{1-e^4}.
\]

The Edwards-to-Montgomery x-coordinate is

\[
u=\frac{1+y_E}{1-y_E},
\]

hence

\[
\boxed{x_0=\frac{t^2(9t^2-1)}{t^2-9}}.
\]

Therefore both `A24` and `x0` use only rational operations modulo G_p.
No modular square root and no knowledge of q are needed.

Any failed denominator inversion is checked with gcd; a non-trivial gcd is
itself a factor.

## 4. What is proved versus experimental

Proved/algebraic:

- the GM identities defining I and z;
- the Edwards family and explicit same-curve point;
- the Montgomery conversion;
- the x0 simplification above;
- construction modulo the composite target without q;
- reuse of the existing x-only Stage1 and BSGS Stage2.

Experimentally and exactly verified by point counting on authentic GM factors:

| p | q | r | exact #E(Fq) |
|---:|---:|---:|---:|
| 1039 | 4157 | 1 | 4096 |
| 509 | 4073 | 20 | 4096 |
| 1009 | 12109 | 16 | 12288 = 3*4096 |

Thus the family genuinely reaches v2(#E)=12 on real GM factors.

Not proved:

\[
\forall q\mid G_p,\quad \exists r\in R:\ 4096\mid\#E_r(\mathbf F_q).
\]

The r-list is an experimental search order. `-K` controls how many profiles are
executed; default is 12.

## 5. Isolation

- `-gm-ecm` remains ordinary Suyama.
- `-gm-ecm-special32` remains unchanged.
- `-gm-ecm-special4096` is opt-in.
- no kernel/Aevum/Marin arithmetic is changed.
- Stage1 fused ladder and BSGS Stage2 are reused.
- GM-only in v1.
- result JSON uses current GMNet-compatible `mode: "gm-ecm"`.

## 6. Initial benchmark

For a direct quick-win comparison:

```bash
./prmers P -gm-ecm-special4096 -gm-family GM \
  -b1 8000 -b2 400000 -K 12 -aevum -d 0 -r -f OUT
```

Use the same B1/B2 and number of curves for ordinary Suyama. No speedup is
claimed before the GPU campaign supplies data.
