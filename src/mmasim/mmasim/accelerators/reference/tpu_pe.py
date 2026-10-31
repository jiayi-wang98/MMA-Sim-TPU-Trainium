"""Candidate end-to-end behavior, including special-value hypotheses.

Wraps the frozen finite-product model without changing prior snapshots.
The FP32 external c parameter is a late add, not an MXU-chain initializer.
Physical PE boundaries and internal register widths are not asserted.
"""
from .tpu import dot_bits,add32,unpack


def dp4(a,b):
    infinity_signs=set()
    finite_a=[];finite_b=[]
    for x,y in zip(a,b):
        x,y=int(x),int(y)
        ax,ay=x&0x7fff,y&0x7fff
        if ax>0x7f80 or ay>0x7f80:return 0x7fc00000
        zx,zy=ax<0x0080,ay<0x0080  # includes input FTZ
        if ax==0x7f80 or ay==0x7f80:
            if zx or zy:return 0x7fc00000
            infinity_signs.add((x^y)>>15)
            finite_a.append(0);finite_b.append(0)
        else:
            xn,xe=unpack(x,7,127);yn,ye=unpack(y,7,127)
            product=xn*yn
            if product and abs(product).bit_length()-1+xe+ye>127:
                infinity_signs.add(int(product<0))
                finite_a.append(0);finite_b.append(0)
            else:
                finite_a.append(x);finite_b.append(y)
    if len(infinity_signs)>1:return 0x7fc00000
    if infinity_signs:return (next(iter(infinity_signs))<<31)|0x7f800000
    return dot_bits(finite_a,finite_b)


def tile128(a,b):
    acc=0
    for i in range(0,len(a),8):
        first=dp4(a[i:i+4],b[i:i+4]);second=dp4(a[i+4:i+8],b[i+4:i+8])
        acc=add32(acc,add32(first,second))
    return acc


def gemm_dot(a,b,external_c=0,tile=128):
    acc=0
    for i in range(0,len(a),tile):acc=add32(acc,tile128(a[i:i+tile],b[i:i+tile]))
    return add32(acc,external_c)
