"""Captured TPU7x non-N128 BF16 path: fused cross-half DP8 and two outer chains.

Inferred numerical composition, scoped to captured K1024/2048 native jobs.
"""
from .tpu import dot_bits,add32
SHAPES={(64,64),(128,256),(256,256),(127,129)}

def cross_block256(a,b):
    if len(a)!=256 or len(b)!=256:raise ValueError('Complete K256 block required')
    acc=0
    for i in range(0,128,8):
        pair=[]
        for h in (0,4):
            aa=a[i+h:i+h+4]+a[i+h+128:i+h+132]
            bb=b[i+h:i+h+4]+b[i+h+128:i+h+132]
            pair.append(dot_bits(aa,bb,group=8,fraction=27,sticky=False))
        acc=add32(acc,add32(pair[0],pair[1]))
    return acc

def dot_cross(a,b,*,m,n):
    if (m,n) not in SHAPES or len(a) not in (1024,2048) or len(a)!=len(b):
        raise NotImplementedError('Cross-half TPU7x profile is measured only at four M/N shapes and K1024/2048')
    partials=[cross_block256(a[i:i+256],b[i:i+256]) for i in range(0,len(a),256)]
    half=len(partials)//2
    def fold(v):
        acc=0
        for x in v:acc=add32(acc,x)
        return acc
    return add32(fold(partials[:half]),fold(partials[half:]))
