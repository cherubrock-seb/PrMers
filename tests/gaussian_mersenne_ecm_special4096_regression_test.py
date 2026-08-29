#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cli = (root / "src/io/CliParser.cpp").read_text()
app = (root / "src/core/App.cpp").read_text()
fast = (root / "src/modes/RunGaussianMersenneEcmFast.cpp").read_text()
opt = (root / "src/modes/RunGaussianMersenneEcmOptimized.cpp").read_text()

assert '-gm-ecm-special4096' in cli
assert 'opts.mode = "gm-ecm-special4096"' in cli
assert 'options.mode == "gm-ecm-special4096"' in fast
assert 'gm-ecm-special4096' in app
assert 'make_special4096_opt' in opt
assert 'GM_SPECIAL4096_R' in opt
assert 't.special4096 = special4096' in opt
assert 'proven point, experimental v2>=12 coverage' in opt
assert 'make_special32_opt' in opt
assert 'make_suyama_opt(t.n, sigma)' in opt

def legendre(a,p):
    a %= p
    if a == 0: return 0
    v = pow(a,(p-1)//2,p)
    return 1 if v == 1 else -1

def gm_norm(p):
    chi = 1 if p % 8 in (1,7) else -1
    return (1<<p) - chi*(1<<((p+1)//2)) + 1

def oriented_i(p,n):
    I=(1<<p)%n
    if ((p+1)//2)&1:
        I=(-I)%n
    return I

def setup_mod_prime(p,q,r):
    I=oriented_i(p,q)
    z=(1+I)%q
    t=pow(z,r,q)
    e=3*(t*t-1)*pow(8*t,-1,q)%q
    d=(-pow(e,4,q))%q
    a24=pow((1+d)%q,-1,q)
    x=t*t*(9*t*t-1)*pow((t*t-9)%q,-1,q)%q
    A=(4*a24-2)%q
    B=(-4*a24)%q
    return t,e,d,x,a24,A,B

def count_curve(p,q,r):
    t,e,d,x,a24,A,B=setup_mod_prime(p,q,r)

    xe=pow((4*pow(e,3,q)+3*e)%q,-1,q)
    ye=(9*pow(t,4,q)-2*t*t+9)*pow((9*pow(t,4,q)-9)%q,-1,q)%q
    assert (-xe*xe + ye*ye - 1 - d*xe*xe*ye*ye) % q == 0
    assert x == (1+ye)*pow((1-ye)%q,-1,q)%q

    Binv=pow(B,-1,q)
    total=1
    for X in range(q):
        rhs=X*(X*X+A*X+1)*Binv % q
        if rhs == 0:
            total += 1
        elif legendre(rhs,q) == 1:
            total += 2
    return total,x,a24

golden = (
    (1039, 4157, 1, 4096),
    (509,  4073, 20, 4096),
    (1009, 12109, 16, 12288),
)
for p,q,r,wanted_order in golden:
    assert gm_norm(p) % q == 0
    order,x,a24=count_curve(p,q,r)
    assert order == wanted_order, (p,q,r,order)
    assert order % 4096 == 0

print("Gaussian ECM Special4096 regression: OK")
print("exact v2>=12 golden curves:", len(golden))
print("STATUS: 4096 is an experimental portfolio target, not universal coverage")
