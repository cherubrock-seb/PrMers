#!/usr/bin/env python3
from pathlib import Path
from math import gcd

root = Path(__file__).resolve().parents[1]
opt = (root / "src/modes/RunGaussianMersenneEcmOptimized.cpp").read_text()
fast = (root / "src/modes/RunGaussianMersenneEcmFast.cpp").read_text()
app = (root / "include/core/App.hpp").read_text()
ver = (root / "include/core/Version.hpp").read_text()

assert "runGaussianMersenneECMOptimized" in app
assert "options.bsgs && !options.edwards" in fast
assert "runGaussianMersenneECMOptimized()" in fast
assert "4.20.87-alpha-v99.98-gaussian-ecm-fused-bsgs" in ver

# The optimized path must use the fused engine primitives and a real
# differential baby/giant Stage 2, not the old product-of-primes exponent.
for token in (
    "addsub_copy",
    "mul_pair_prepared",
    "xdbl_tail_uv",
    "mont_xadd_general_opt",
    "baby-step/giant-step",
    "Xg*Zb - Zg*Xb",
):
    assert token in opt, token

MASK = (1 << 64) - 1

def modinv(a, n):
    return pow(a % n, -1, n)

def suyama(n, sigma):
    u = (sigma * sigma - 5) % n
    v = (4 * sigma) % n
    xproj = pow(u, 3, n)
    zproj = pow(v, 3, n)
    x = xproj * modinv(zproj, n) % n
    num = pow((v-u) % n, 3, n) * ((3*u+v) % n) % n
    den = 16 * xproj * v % n
    a24 = num * modinv(den, n) % n
    return x, a24

def xdbl(P, a24, n):
    X, Z = P
    A = (X + Z) % n
    B = (X - Z) % n
    AA = A*A % n
    BB = B*B % n
    E = (AA-BB) % n
    return AA*BB % n, E*(BB+a24*E) % n

def xadd(P, Q, D, n):
    X1,Z1=P; X2,Z2=Q; Xd,Zd=D
    A=(X1+Z1)%n; B=(X1-Z1)%n
    C=(X2+Z2)%n; Dm=(X2-Z2)%n
    DA=Dm*A%n
    CB=C*B%n
    s=(DA+CB)%n
    d=(DA-CB)%n
    return Zd*s*s%n, Xd*d*d%n

def xmul(P, k, a24, n):
    if k == 0:
        return (1,0)
    if k == 1:
        return P
    R0=P
    R1=xdbl(P,a24,n)
    for bit in bin(k)[3:]:
        if bit == "0":
            R1=xadd(R0,R1,P,n)
            R0=xdbl(R0,a24,n)
        else:
            R0=xadd(R0,R1,P,n)
            R1=xdbl(R1,a24,n)
    return R0

def primes(n):
    p=[]
    for x in range(2,n+1):
        if all(x%d for d in range(2,int(x**0.5)+1)):
            p.append(x)
    return p

def stage1_exp(B1):
    E=1
    for q in primes(B1):
        pp=q
        while pp*q <= B1:
            pp*=q
        E*=pp
    return E

# Golden Stage 1 from the real GM-ECM seed-fix campaign.
B1=2000
E=stage1_exp(B1)
sigma=3059155915320676093
for factor in (3253353737, 148455667849):
    x,a24=suyama(factor,sigma)
    R=xmul((x,1),E,a24,factor)
    assert R[1] % factor == 0

# Deterministic Stage 2 golden:
# G_89 = 1069 * 579017791994999956106149.
# With sigma=6 and B1=20 neither factor is killed by Stage 1.
# q=43 is the Stage-2 prime that kills the 1069 component.
p=89
N=(1<<p) - (1<<((p+1)//2)) + 1
f1=1069
f2=579017791994999956106149
assert N == f1*f2

B1=20
B2=50
sigma=6
E=stage1_exp(B1)
Q={}
for f in (f1,f2):
    x,a24=suyama(f,sigma)
    Q[f]=(xmul((x,1),E,a24,f),a24)
    assert Q[f][0][1] % f != 0

assert xmul(Q[f1][0],43,Q[f1][1],f1)[1] % f1 == 0
assert xmul(Q[f2][0],43,Q[f2][1],f2)[1] % f2 != 0

# Check the BSGS collision used by v99.98.
# 43 = 1*30 + 13, so x([30]Q) == x([13]Q) modulo 1069.
D=30
for f in (f1,f2):
    P,a24=Q[f]
    giant=xmul(P,D,a24,f)
    baby=xmul(P,13,a24,f)
    cross=(giant[0]*baby[1] - giant[1]*baby[0]) % f
    if f == f1:
        assert cross == 0
    else:
        assert cross != 0

print("Gaussian ECM v99.98 fused/BSGS regression: OK")
print("Stage1 golden factor bundle:", 3253353737 * 148455667849)
print("Stage2 golden: p=89 sigma=6 B1=20 B2=50 factor=1069")


# Verify the nearest-multiple BSGS map covers every Stage-2 prime for the
# production choices D=30/210 whenever the prime does not divide D.
def bsgs_plan(B1, B2, D):
    out=[]
    babies=set()
    for q in primes(B2):
        if q <= B1:
            continue
        k=(q + D//2)//D
        if k == 0:
            continue
        delta=abs(k*D-q)
        if delta == 0 or delta > D//2 or gcd(delta,D) != 1:
            continue
        out.append((q,k,delta))
        babies.add(delta)
    return out, sorted(babies)

for lo,hi,D in ((20,50,30),(1000,100000,210),(50000,500000,210)):
    plan,babies=bsgs_plan(lo,hi,D)
    expected=[q for q in primes(hi) if q>lo and gcd(q,D)==1]
    assert [q for q,_,_ in plan] == expected
    assert all(1 <= d <= D//2 and gcd(d,D)==1 for d in babies)

# End-to-end CPU emulation of the v99.98 Stage-2 product.  It must isolate
# 1069 exactly, not merely observe one hand-picked q=43 collision.
plan,babies=bsgs_plan(20,50,30)
acc=1
for q,k,d in plan:
    giant=xmul(Q[f1][0],k*30,Q[f1][1],f1)
    baby=xmul(Q[f1][0],d,Q[f1][1],f1)
    acc=acc*((giant[0]*baby[1]-giant[1]*baby[0])%f1)%f1
assert acc == 0

accN=1
xN,a24N=suyama(N,6)
QN=xmul((xN,1),stage1_exp(20),a24N,N)
for q,k,d in plan:
    giant=xmul(QN,k*30,a24N,N)
    baby=xmul(QN,d,a24N,N)
    accN=accN*((giant[0]*baby[1]-giant[1]*baby[0])%N)%N
assert gcd(accN,N) == 1069

print("BSGS plan coverage and batched-GCD golden: OK")
