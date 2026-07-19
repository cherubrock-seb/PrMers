#!/usr/bin/env python3
from __future__ import annotations
import random
from dataclasses import dataclass

P31=(1<<31)-1
P61=(1<<61)-1
ROOTS={P31:{3:1513477735,9:765383222},P61:{3:1669582390241348315,9:1102844585000305877}}

@dataclass(frozen=True)
class GF:
    a:int; b:int; p:int
    def __add__(self,o): return GF((self.a+o.a)%self.p,(self.b+o.b)%self.p,self.p)
    def __sub__(self,o): return GF((self.a-o.a)%self.p,(self.b-o.b)%self.p,self.p)
    def smul(self,s): return GF(self.a*s%self.p,self.b*s%self.p,self.p)

def dft_naive(v,root):
    n=len(v); p=v[0].p
    return [sum_gf((v[j].smul(pow(root,j*k,p)) for j in range(n)),p) for k in range(n)]
def sum_gf(it,p):
    r=GF(0,0,p)
    for x in it:r=r+x
    return r
def dft3(v,w):
    # One-multiply identity used by the optimized OpenCL kernel.
    x0,x1,x2=v
    wd=(x1-x2).smul(w)
    return [x0+x1+x2, x0-x2+wd, x0-x1-wd]
def dft9(v,w):
    p=v[0].p; w3=pow(w,3,p); w2=w*w%p; w4=w2*w2%p
    t=[GF(0,0,p) for _ in range(9)]
    for n1 in range(3):
        z=dft3([v[n1],v[n1+3],v[n1+6]],w3)
        t[n1*3:n1*3+3]=z
    t[4]=t[4].smul(w); t[5]=t[5].smul(w2); t[7]=t[7].smul(w2); t[8]=t[8].smul(w4)
    o=[GF(0,0,p) for _ in range(9)]
    for k2 in range(3):
        z=dft3([t[k2],t[3+k2],t[6+k2]],w3)
        o[k2],o[k2+3],o[k2+6]=z
    return o
def inverse(v,w,fn):
    p=v[0].p; invn=pow(len(v),-1,p)
    return [x.smul(invn) for x in fn(v,pow(w,-1,p))]

def check_field(p):
    rng=random.Random(0xA39+p)
    for n,fn in ((3,dft3),(9,dft9)):
        w=ROOTS[p][n]
        assert pow(w,n,p)==1 and pow(w,n//3,p)!=1
        # Odd roots are base-field scalars. This is what lets every odd row
        # retain the Aevum packed-half-real binary transform.
        for _ in range(200):
            v=[GF(rng.randrange(p),rng.randrange(p),p) for _ in range(n)]
            f=fn(v,w)
            assert f==dft_naive(v,w)
            assert inverse(f,w,fn)==v
        print(f"GF(M{p.bit_length()}^2) radix-{n}: direct/inverse OK")

def logical_index(row,binary_index,L,R):
    linv=pow(L%R,-1,R)
    return binary_index+L*((row-binary_index)%R*linv%R)

def check_log2_root_two():
    # For a power-of-two length, the modular inverse equals GPUOwl's stock
    # division formula. For mixed lengths, integer division is wrong.
    for bits in (31, 61):
        period = bits
        for n in (1 << 20, 1 << 22):
            stock = ((1 << (bits - 1)) // n) % period
            assert stock == pow(n % period, -1, period)
        for n in (3 * (1 << 20), 9 * (1 << 19)):
            mixed = pow(n % period, -1, period)
            assert (n * mixed) % period == 1
            assert mixed != ((1 << (bits - 1)) // n) % period
    print("Mixed-length IBDWT log2(root-two): modular inverses OK")

def check_good_thomas(R,W,H):
    L=2*W*H; N=R*L
    seen=set()
    for row in range(R):
        for b in range(L):
            n=logical_index(row,b,L,R)
            assert 0<=n<N and n%R==row and n%L==b
            seen.add(n)
    assert len(seen)==N
    # Validate the exact inverse gather used by pfaunpack.cl.
    for n in range(N):
        row=n%R; b=n%L
        assert logical_index(row,b,L,R)==n
    print(f"Good-Thomas radix-{R}: bijection and inverse gather OK (L={L:,})")



def pfa_logical_step(R,L,NW):
    binary_step=L//NW
    for k in range(R):
        candidate=binary_step+L*k
        if candidate%R==0:
            return candidate
    raise AssertionError("no CRT step")

def check_fused_fftw_scatter():
    # Reproduce the optimized fftW scalar scatter.  It must produce exactly
    # the same (odd,even) canonical vector layout as the former pfaUnpack pass.
    for R,W,H,NW in ((3,32,16,4),(9,32,16,8)):
        L=2*W*H; N=R*L; GW=W//NW
        step=pfa_logical_step(R,L,NW)
        out=[[None,None] for _ in range(N//2)]
        seen=set()
        for row in range(R):
            for y in range(H):
                for me in range(GW):
                    first_pair=me*H+y
                    ne=logical_index(row,2*first_pair,L,R)
                    no=logical_index(row,2*first_pair+1,L,R)
                    for _ in range(NW):
                        # source vector is component-swapped (odd,even)
                        assert ne%2==0 and no%2==1
                        assert ne not in seen and no not in seen
                        seen.update((ne,no))
                        out[ne//2][1]=ne
                        out[no//2][0]=no
                        ne=(ne+step)%N
                        no=(no+step)%N
        assert len(seen)==N
        assert out==[[2*i+1,2*i] for i in range(N//2)]
    print("PFA fused fftW canonical scalar scatter: exact old-unpack layout OK")

def check_precary_component_repack():
    # Aevum tailSquare/tailMul writes inverse results as (odd, even).
    # carry.cl applies SWAP_XY before CRT/carry.  PFA inverse gather must
    # therefore recover even from source.y, odd from source.x and emit
    # (odd, even), including when adjacent logical digits come from rows.
    for R in (3, 9):
        L=64
        N=R*L
        source=[]
        for row in range(R):
            row_pairs=[]
            for pair in range(L//2):
                even=logical_index(row,2*pair,L,R)
                odd=logical_index(row,2*pair+1,L,R)
                row_pairs.append((odd,even))  # exact pre-carry component order
            source.append(row_pairs)
        rebuilt=[]
        for pair in range(N//2):
            n0=2*pair; n1=n0+1
            r0,b0=n0%R,n0%L
            r1,b1=n1%R,n1%L
            a=source[r0][b0//2]
            b=source[r1][b1//2]
            even=a[0] if b0&1 else a[1]
            odd=b[0] if b1&1 else b[1]
            pre_carry=(odd,even)
            post_swap=(pre_carry[1],pre_carry[0])
            assert post_swap==(n0,n1)
            rebuilt.extend(post_swap)
        assert rebuilt==list(range(N))
    print("PFA pre-carry component swap/repack: exact stock convention OK")

def main():
    check_field(P31); check_field(P61)
    check_log2_root_two()
    check_good_thomas(3,16,8)
    check_good_thomas(9,16,8)
    check_precary_component_repack()
    check_fused_fftw_scatter()
    print("radix-3 plan: 3,145,728 vs 4,194,304 words; reduction x1.333333")
    print("radix-9 plan: 4,718,592 vs 8,388,608 words; reduction x1.777778")
    print("ALL NATIVE AEVUM PFA REFERENCE TESTS PASSED")
if __name__=='__main__': main()
