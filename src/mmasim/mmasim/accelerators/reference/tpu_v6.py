"""v6e BF16 native candidate after the 2026-10-31 shape sweep.

K128 matches all tested output shapes. At K256/512/1024/2048, N128
matches paired 128-term tiles followed by sequential FP32 accumulation.
Other tested N values have unresolved native-path discrepancies at K>=256.
The local arithmetic is the unchanged v5e/v6e DP4 partial-tail candidate.
"""
from .tpu_pe import tile128
from .tpu import add32


def dot_bf16(a, b, *, n):
    if len(a) != len(b) or len(a) not in (128, 256, 512, 1024, 2048):
        raise ValueError('Equal vectors with a validated v6e K required')
    if len(a) > 128 and n != 128:
        raise NotImplementedError('v6e native N!=128 remains unresolved at K>=256')
    tiles = [tile128(a[i:i+128], b[i:i+128]) for i in range(0, len(a), 128)]
    acc = 0
    for i in range(0, len(tiles), 2):
        pair = add32(tiles[i], tiles[i+1]) if i+1 < len(tiles) else tiles[i]
        acc = add32(acc, pair)
    return acc
