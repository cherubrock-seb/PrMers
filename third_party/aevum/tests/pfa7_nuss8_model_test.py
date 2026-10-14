"""Exact scalar algebra for NUSS8; does not compile or execute GPU kernels."""
from pathlib import Path
import random
PTH = Path(__file__).resolve().parents[1] / "src/cl/fft-middle.cl"
SOURCE = PTH.read_text()
FIELDS = [
 (31, (141971251,405239506,526531066), (752284155,1347360268,1073059520)),
 (61, (1773674115463894065,901087505085424499,784002893271222362),
      (673484497713682504,831884051671952528,381307857222696375)),
]
def dft(u,r,p):
 return [sum(a*pow(r,j*k,p) for j,a in enumerate(u))%p for k in range(7)]
def coefficients(c,s,p):
 c1,c2,c3=c;s1,s2,s3=s;iv=pow(3,-1,p)
 return [v*iv%p for v in
  (c1+c2+c3,2*c1-c2-c3,c1-2*c2+c3,c1+c2-2*c3,
   s1+s2-s3,2*s1-s2+s3,s1-2*s2-s3,s1+s2+2*s3)]
def candidate(u,co,p,inverse):
 C1,C2,C3,C4=co[:4];S1,S2,S3,S4=[(-v if inverse else v)%p for v in co[4:]]
 a1,a2,a3=[(u[k]+u[7-k])%p for k in (1,2,3)]
 b1,b2,b3=[(u[k]-u[7-k])%p for k in (1,2,3)]
 total=(a1+a2+a3)%p
 m2=-C2*(a1-a3)%p;tc=(C3*(a2-a3)+m2)%p;mc=(-C4*(a2-a1)+m2)%p
 center=(u[0]+C1*total)%p
 t1=(center-tc)%p;t2=(center+mc)%p;t3=(center+tc-mc)%p
 m6=-S2*(b3+b1)%p;tq=(S3*(b2+b3)+m6)%p;sq=(S4*(b1-b2)+m6)%p
 q0=S1*(b2-b3+b1)%p;q1=(q0-tq)%p;q2=(q0+sq)%p;q3=(sq-q0-tq)%p
 return [(u[0]+total)%p,(t1+q1)%p,(t2+q2)%p,(t3+q3)%p,
         (t3-q3)%p,(t2-q2)%p,(t1-q1)%p]
checks=0
for bits,c,s in FIELDS:
 p=(1<<bits)-1;co=coefficients(c,s,p);r=(c[0]+s[0])%p
 assert pow(r,7,p)==1 and r!=1
 # Match all eight constants actually used by the kernel, including negatives.
 used=[co[0],p-co[1],co[2],p-co[3]]+co[4:]+[p-v for v in co[4:]]
 suffix="U" if bits==31 else "UL"
 for value in used: assert f"(Z{bits}){value}{suffix}" in SOURCE
 vectors=[[int(i==j) for i in range(7)] for j in range(7)]
 vectors += [[v]*7 for v in (0,1,p-1,p,p+1,p+7)]
 vectors += [[0,p-1,p,p+1,p+7,1,2]]
 rng=random.Random(bits)
 vectors += [[rng.randrange(p+8) for _ in range(7)] for _ in range(128)]
 for inverse in (False,True):
  root=pow(r,-1,p) if inverse else r
  for u in vectors:
   got=candidate(u,co,p,inverse)
   assert got==dft(u,root,p)
   checks+=7
print(f"{checks} scalar output comparisons; device validation pending")
