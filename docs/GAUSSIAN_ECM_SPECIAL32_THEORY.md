# Gaussian-Mersenne ECM Special32 — mathematical implementation note

## Status

`-gm-ecm-special32` is an **experimental, GM-only, opt-in ECM prepass**.

It does not replace ordinary `-gm-ecm`. The existing Suyama curve stream, kernels,
Aevum/Marin backends, Stage 1 ladder and BSGS Stage 2 remain available unchanged.

The purpose of Special32 is to test a deterministic three-profile construction whose
curve orders have a stronger GM-specific 2-primary structure than generic Suyama
curves. Performance claims must come from benchmark data; the construction does not
by itself imply a fixed speedup.

## 1. Gaussian-Mersenne norm

For an odd prime exponent `p`, define

$$
G_p = 2^p-\left(\frac 2p\right)2^{(p+1)/2}+1.
$$

For every prime divisor $q\neq 5$ of $G_p$, one has the usual Gaussian-Mersenne
order condition $q\equiv 1\pmod{4p}$.

## 2. Oriented square root of -1

Modulo $G_p$, define

$$
I=(-1)^{(p+1)/2}2^p.
$$

The identities used by the implementation are

$$
I^2\equiv -1\pmod{G_p},
\qquad
(1+I)^p\equiv 1\pmod{G_p}.
$$

For a prime factor $q\mid G_p$, the element $z=1+I$ has the GM-specific
order structure used in the descent argument.

The implementation constructs `I` directly modulo the composite target `G_p`;
it does not need to know any factor of `G_p`.

## 3. Curve family and 2-descent

Consider

$$
E_w:\quad y^2=x(x+w)(x+w^{-1}).
$$

The 2-descent analysis gives two complementary criteria. In the notation of the
derivation,

$$
\beta=1 \Longrightarrow 32\mid \lvert E_w(\mathbf F_q)\rvert,
$$

and

$$
q\equiv5\pmod 8,\quad \alpha=-1
\Longrightarrow 32\mid \lvert E_w(\mathbf F_q)\rvert.
$$

The delicate branches lift respectively to group structures containing
$C_{16}\times C_2$ or $C_8\times C_4$. Thus the relevant curve order is
divisible by 32.

## 4. Two fixed curves

Take

$$
w_1=1+i,
\qquad
w_3=(1+i)^3=-2+2i.
$$

After conversion to Montgomery form, the two fixed parameters are

$$
A_{24,1}=\frac{7+i}{8},
\qquad
A_{24,3}=\frac{-1+7i}{16}.
$$

The coverage statement used by Special32 is

$$
\forall q\mid G_p,\qquad
32\mid\lvert E_1(\mathbf F_q)\rvert
\quad\text{or}\quad
32\mid\lvert E_3(\mathbf F_q)\rvert.
$$

This is a **portfolio statement**: the implementation therefore runs all three
fixed point/curve profiles once instead of generating a random curve stream.

## 5. The three x-only ECM profiles

With the composite-target value of $I$, PrMers constructs:

| profile | curve | $x_0$ | $A_{24}$ |
|---|---|---|---|
| A | $E_1$ | $-1+I$ | $(7+I)/8$ |
| B | $E_1$ | $-4$ | $(7+I)/8$ |
| C | $E_3$ | $(1-7I)/4$ | $(-1+7I)/16$ |

All divisions mean multiplication by the modular inverse modulo $G_p$.
Since $G_p$ is odd, 4, 8 and 16 are invertible.

A crucial implementation property is:

> **No y-coordinate and no modular square root are required.**

The existing PrMers Montgomery x/z ladder already consumes only `(x0, A24)`.
Consequently Special32 can reuse the current fused xDBL/xDBLADD Stage 1 and
the existing BSGS Stage 2.

## 6. Singularity/factor guard

For a Montgomery curve

$$
y^2=x^3+Ax^2+x,
$$

bad reduction corresponds to $A^2-4=0$. With $A_{24}=(A+2)/4$, PrMers
checks

$$
\gcd(A_{24}(A_{24}-1),G_p).
$$

A non-trivial gcd is reported as a factor rather than as a setup error.

## 7. Why exactly three profiles

Special32 is not a random-curve generator.

It runs exactly

$$
E_1(P_A),\qquad E_1(P_B),\qquad E_3(P_C).
$$

Therefore `-K` and Suyama `-sigma`/`-seed` do not define additional Special32
curves. Repeating one of the fixed profiles would not create a new curve order.

This first implementation is intentionally **GM-only**. GQ is left outside the
mode until its complementary coverage argument is closed independently.

## 8. Relationship with ordinary Suyama ECM

Classical Suyama parameterization provides a structural factor 12 in the curve
order. The Special32 portfolio targets a structural factor 32 on the covered
fixed curve:

$$
\frac{32}{12}=\frac83,
\qquad
\log_2(8/3)\approx1.415.
$$

This should be interpreted as extra structural smoothness, **not** as a claim
of $8/3$ times faster ECM.

The order of the selected point need not exploit the entire 2-primary part of
the curve order. The actual success probability and time-to-factor must
therefore be measured experimentally.

For a fair benchmark, compare:

```bash
./prmers P -gm-ecm-special32 -gm-family GM -b1 B1 -b2 B2 -aevum -d 0
```

against three Suyama curves using the same fused/BSGS engine:

```bash
./prmers P -gm-ecm -gm-family GM -gm-sieve 0 -bsgs \
  -b1 B1 -b2 B2 -K 3 -seed 20260827 -aevum -d 0
```

## 9. Small Stage-1 validation

The source regression emulates the same Montgomery x/z formulas on the CPU and
checks historical composite GM cases. At the stated bounds the A/B/C portfolio
recovers, among others:

- $G_{13}$: factor 53 with $B_1=5$
- $G_{17}$: factor 137 with $B_1=5$
- $G_{23}$: factor 277 with $B_1=5$
- $G_{31}$: factor 5581 with $B_1=50$
- $G_{37}$: factor 593 with $B_1=5$
- $G_{41}$: factor 181549 with $B_1=100$
- $G_{43}$: factor 173 with $B_1=5$

These tests validate the composite-target setup and Stage-1 x/z arithmetic.
They do not substitute for large-exponent performance measurements.

## 10. Scope and limitations

The v1 contract is deliberately narrow:

- opt-in only: `-gm-ecm-special32`;
- GM only;
- exactly three deterministic profiles;
- ordinary `-gm-ecm` remains Suyama;
- no README rewrite;
- no kernel, Aevum or Marin arithmetic change;
- Stage 1 and BSGS Stage 2 are reused;
- `-gm-safe` replay is not enabled for Special32 v1;
- 64/128/256 divisibility is not claimed;
- no speedup is claimed before benchmark data.

The next research step is to study further halving of these fixed curves using
quartic and octic characters of a finite set of Gaussian integers.
