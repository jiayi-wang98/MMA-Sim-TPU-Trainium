"""Integer dyadic arithmetic shared with the existing TPU reference.

No hardware calls or floating-point arithmetic. Finite input unpacking;
FP32 packing uses RNE and gradual underflow. add32 handles output infinities.
"""

def shift_round(n, shift, mode):
    if shift >= 0:
        return n << shift
    sign = -1 if n < 0 else 1
    q, r = divmod(abs(n), 1 << -shift)
    if mode == 'rne':
        half = 1 << (-shift - 1)
        q += r > half or (r == half and q % 2 == 1)
    elif mode == 'odd':
        q |= bool(r)
    elif mode == 'rdn':
        q += bool(r) and sign < 0
    elif mode != 'rtz':
        raise ValueError(mode)
    return sign*q


def pack32(n, exponent):
    if not n:
        return 0
    sign = 0x80000000 if n < 0 else 0
    n = abs(n)
    e = max(-126, n.bit_length()-1+exponent)
    q = shift_round(n, exponent-e+23, 'rne')
    if q >= 1 << 24:
        q >>= 1
        e += 1
    if e > 127:
        return sign | 0x7f800000
    if q < 1 << 23:
        return sign | q
    return sign | ((e+127) << 23) | (q-(1 << 23))


def unpack(bits, fraction_bits, bias, ftz=True):
    sign = -1 if bits >> (fraction_bits+8) else 1
    e = (bits >> fraction_bits) & 255
    m = bits & ((1 << fraction_bits)-1)
    if e == 255:
        raise ValueError('Special input not supported in finite-input model')
    if e == 0:
        return (0 if ftz else sign*m), 1-bias-fraction_bits
    return sign*((1 << fraction_bits)+m), e-bias-fraction_bits


def add32(a, b, ftz=True):
    aa, bb = a & 0x7fffffff, b & 0x7fffffff
    if aa > 0x7f800000 or bb > 0x7f800000:
        return 0x7fc00000
    if aa == 0x7f800000 or bb == 0x7f800000:
        if aa == bb and (a >> 31) != (b >> 31):
            return 0x7fc00000
        return a if aa == 0x7f800000 else b
    an, ae = unpack(a, 23, 127, ftz=ftz)
    bn, be = unpack(b, 23, 127, ftz=ftz)
    e = min(ae, be)
    result = pack32((an << (ae-e))+(bn << (be-e)), e)
    return 0 if ftz and (result & 0x7fffffff) < 0x00800000 else result
