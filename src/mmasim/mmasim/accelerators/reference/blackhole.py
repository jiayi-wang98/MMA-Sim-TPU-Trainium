"""Exploratory phase model. Explicitly audited, not a physical netlist."""
from .blackhole_helpers import Fraction,f32,bits,p2,exponent,rounded
import json
from pathlib import Path

def dot(a,b,phases=4,zero_exp=True,raw_exp=True):
 av=[Fraction(f32(x)) for x in a];bv=[Fraction(f32(x)) for x in b]
 ah=[Fraction(f32(x&0xfffe0000)) for x in a];bh=[Fraction(f32(x&0xfff80000)) for x in b]
 al=[x-y for x,y in zip(av,ah)];bl=[x-y for x,y in zip(bv,bh)]
 c=Fraction(0)
 for phase,(xs,ys,shift) in enumerate([(ah,bh,0),(ah,bl,5),(al,bh,7),(al,bl,12)][:phases]):
  ps=[x*y for x,y in zip(xs,ys)]
  es=[(exponent(x)+exponent(y)-shift if (raw_exp or not p) else exponent(p)) if x and y and (zero_exp or p) else -999 for x,y,p in zip(av,bv,ps)]
  for off in range(0,32,16):
   groups=[]
   for start in [off,off+8]:
    e=max(es[start:start+8]);step=p2(e-10)
    groups.append((sum(rounded(v/step,'away') for v in ps[start:start+8])*step,e))
   terms=groups+[(c,exponent(c) if c else -999)]
   e=max(e for _,e in terms);step=p2(e-23)
   value=sum(rounded(v/step,'away' if i==2 else 'positive') for i,(v,_) in enumerate(terms))*step
   outstep=p2(exponent(value)-23) if value else Fraction(1)
   c=Fraction(f32(bits(rounded(value/outstep,'away')*outstep)))
 return bits(c)
