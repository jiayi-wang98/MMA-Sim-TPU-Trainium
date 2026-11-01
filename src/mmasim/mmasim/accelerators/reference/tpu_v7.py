"""Shape-scoped finite BF16 model for captured TPU7x JAX DEFAULT matmuls.

N=128 native: adjacent 128-term tile pairs, then sequential FP32 RNE accumulation.
This was held-out tested at K=128,512,640,768,896,1024,1152,1536,2048 with M=128;
M=1,256 were additionally tested at K=1024,2048. Four additional M/N shapes at K1024/2048 use the fused cross-half model.
Inputs are BF16 integer bit encodings, outputs are FP32 integer bit encodings.
This module does not replace the original round-1 model or low-format models.
"""
from .tpu import dot_bits, add32
from .tpu_v7_cross import dot_cross, SHAPES as CROSS_SHAPES


def combine(partials, schedule):
    values = list(partials)
    if schedule not in ('pair_chain', 'balanced', 'sequential'):
        raise ValueError('Unknown schedule: ' + schedule)
    if schedule == 'pair_chain':
        values = [add32(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
                  for i in range(0, len(values), 2)]
    if schedule == 'balanced':
        while len(values) > 1:
            values = [add32(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
                      for i in range(0, len(values), 2)]
        return values[0] if values else 0
    acc = 0
    for value in values:
        acc = add32(acc, value)
    return acc


def dot_bf16(a, b, *, m, n, merge='native'):
    if m <= 0 or n <= 0 or len(a) != len(b) or not len(a) or len(a) % 128:
        raise ValueError('Positive M/N, equal nonempty vectors, and K multiple of 128 required')
    if merge not in ('native', 'pair_chain', 'balanced', 'sequential'):
        raise ValueError('Unknown merge mode: ' + merge)
    for value in list(a) + list(b):
        if not 0 <= int(value) <= 65535 or (int(value) & 0x7f80) == 0x7f80:
            raise ValueError('Finite BF16 bit encodings required')
    if n != 128:
        if merge != 'native':
            raise NotImplementedError('Cross-half profile supports the captured native merge only')
        return dot_cross(list(a), list(b), m=m, n=n)
    partials = [dot_bits(a[i:i+128], b[i:i+128], fraction=27, sticky=False,
                        reduction='pair_chain') for i in range(0, len(a), 128)]
    return combine(partials, 'pair_chain' if merge == 'native' else merge)
