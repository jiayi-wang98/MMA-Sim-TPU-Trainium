"""Finite-input TPU7x candidates for the captured JAX 0.10.2 DEFAULT paths.
Inputs: BF16 bit encodings of exactly representable BF16/FP8/FP4 values.
BF16: DP4 E-27 RTZ -> FP32 RNE; pair-chain within 128-term tiles.
FP8 / Pallas FP4: DP4 E-8 RTZ -> p=9 RTZ -> p=9 RNE accumulator.
Independent 128-term tiles are combined in a balanced FP32 RNE tree.
The exponent representation and physical resource sharing are not specified.
"""
from .tpu import unpack,shift_round,pack32,add32,dot_bits

def pack9(n,e,rounding):
    if not n:return 0
    sign=-1 if n<0 else 1
    n=abs(n);d=n.bit_length()-9
    if d>0:
        n=shift_round(n,-d,rounding);e+=d
    return pack32(sign*n,e)

def low_tile(a,b):
    acc=0
    for off in range(0,len(a),4):
        terms=[]
        for x,y in zip(a[off:off+4],b[off:off+4]):
            an,ae=unpack(int(x),7,127);bn,be=unpack(int(y),7,127)
            if an*bn:terms.append((an*bn,ae+be))
        if not terms:continue
        step=max(abs(n).bit_length()-1+e for n,e in terms)-8
        total=sum(shift_round(n,e-step,'rtz') for n,e in terms)
        partial=pack9(total,step,'rtz')
        an,ae=unpack(acc,23,127);bn,be=unpack(partial,23,127);e=min(ae,be)
        acc=pack9((an<<(ae-e))+(bn<<(be-e)),e,'rne')
    return acc

def dot(a,b,low=False):
    partials=[]
    for off in range(0,len(a),128):
        aa,bb=a[off:off+128],b[off:off+128]
        partials.append(low_tile(aa,bb) if low else dot_bits(aa,bb,group=4,fraction=27,sticky=False,reduction='pair_chain'))
    while len(partials)>1:
        partials=[add32(partials[i],partials[i+1]) if i+1<len(partials) else partials[i] for i in range(0,len(partials),2)]
    return partials[0] if partials else 0
