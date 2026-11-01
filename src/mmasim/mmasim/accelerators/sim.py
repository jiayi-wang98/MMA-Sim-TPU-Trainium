"""CPU reference adapter for measured TPU, Blackhole and Trainium3 paths.

Input values must already be quantized to the selected operand format. No
implicit casting, fused-C assumption, or autograd is used. The v6e backend
models zero padding inside its captured K256 MXU scheduling blocks.
All dot_bits inputs use FP32 encodings of those operand values.
"""
from fractions import Fraction
import struct
import torch

from .reference import tpu_pe, tpu_v6, tpu_v7, tpu_v7_low, blackhole, trainium
from .reference.arithmetic import unpack, add32

_FORMATS = {
    'bf16': (8, -126, 127, None), 'fp16': (11, -14, 15, None),
    'fp32': (24, -126, 127, None), 'e4m3': (4, -6, 7, None),
    'e4m3fn': (4, -6, 8, 448), 'e5m2': (3, -14, 15, None),
    'e2m1fn': (2, 0, 2, 6),
}


def _validate_words(words, fmt):
    p, emin, emax, maxvalue = _FORMATS[fmt]
    out = []
    for word in words:
        if isinstance(word, bool) or int(word) != word or not 0 <= word < 2**32:
            raise ValueError('Expected unsigned FP32 bit encodings')
        word = int(word)
        n, e = unpack(word, 23, 127, ftz=False)  # rejects nonfinite operands
        if n:
            top = abs(n).bit_length() - 1 + e
            quantum = max(emin, top) - (p - 1)
            shift = quantum - e
            if top > emax or (shift > 0 and abs(n) % (1 << shift)):
                raise ValueError(f'Input is not exactly representable as {fmt}')
            if maxvalue is not None and abs(_float(word)) > maxvalue:
                raise ValueError(f'Input exceeds the finite range of {fmt}')
        out.append(word)
    return out


def _float(word):
    return struct.unpack('!f', int(word).to_bytes(4, 'big'))[0]


def _scales(values, count):
    if values is None:
        raise ValueError('MX requires explicit unbiased E8M0 scale exponents')
    if len(values) != count:
        raise ValueError('Expected one scale exponent per 32 logical K terms')
    if any(isinstance(x, bool) or int(x) != x or not -127 <= x <= 127 for x in values):
        raise ValueError('Scale exponents must be integers in [-127, 127]')
    return [int(x) for x in values]


class Matmul:
    """MMA-Sim-style callable for a captured arithmetic profile.

    arch: TPU-v5e, TPU-v6e, TPU-v7, Blackhole, Trainium3.
    mode: ordinary (default), double_row, two_inst, or mx for Trainium3.
    c_mode='zero' requires C=0; 'late_fp32' explicitly adds C after the dot.
    MX scale tensors are unbiased exponents, not raw packed scale bytes.
    TPU-v7 backend='jax' (default) or 'pallas' selects a DEFAULT-precision
    low-format path. FP4 JAX and Pallas have different arithmetic.
    """
    def __init__(self, arch, fmt='bf16', *, mode='ordinary', chunk=128,
                 mapping='half', parts=128, c_mode='zero', backend=None,
                 compiled_hlo=None):
        if arch not in ('TPU-v5e', 'TPU-v6e', 'TPU-v7', 'Blackhole', 'Trainium3'):
            raise ValueError('Unknown architecture')
        if fmt not in _FORMATS or c_mode not in ('zero', 'late_fp32'):
            raise ValueError('Unknown format or C policy')
        if mapping not in ('half', 'interleave') or parts not in (32, 64, 128):
            raise ValueError('Unsupported layout or instruction partition count')
        if chunk not in (32, 64, 128):
            raise ValueError('Supported ordinary instruction chunks: 32, 64, 128')
        if arch != 'TPU-v7' and backend is not None:
            raise ValueError('backend selects a TPU-v7 compiler path only')
        if arch == 'TPU-v7':
            backend = 'jax' if backend is None else backend
            if backend not in ('jax', 'pallas'):
                raise ValueError('TPU-v7 backend must be jax or pallas')
            if fmt not in ('bf16', 'e4m3fn', 'e5m2', 'e2m1fn') or mode != 'ordinary':
                raise NotImplementedError('TPU-v7 supports BF16, FP8 and unscaled FP4 DEFAULT profiles')
            if fmt == 'bf16' and backend != 'jax':
                raise NotImplementedError('The BF16 adapter exposes the native JAX profile')
        if arch not in ('Trainium3', 'TPU-v7') and (fmt != 'bf16' or mode != 'ordinary'):
            raise NotImplementedError('This adapter exposes BF16 for TPU/Blackhole')
        if arch == 'Trainium3':
            if mode not in ('ordinary', 'double_row', 'two_inst', 'mx'):
                raise ValueError('Unknown Trainium3 path')
            if mode in ('double_row', 'two_inst') and fmt not in ('e4m3', 'e4m3fn', 'e5m2'):
                raise ValueError('Double-row requires a measured FP8 format')
            if mode == 'mx' and fmt not in ('e4m3fn', 'e5m2', 'e2m1fn'):
                raise ValueError('MX requires E4M3FN, E5M2, or E2M1FN')
            if mode == 'ordinary' and fmt == 'e2m1fn':
                raise NotImplementedError('Ordinary FP4 was not measured')
        self.arch, self.fmt, self.mode = arch, fmt, mode
        self.chunk, self.mapping, self.parts, self.c_mode = chunk, mapping, parts, c_mode
        self.backend = backend
        if compiled_hlo is not None and arch != 'TPU-v6e':
            raise ValueError('compiled_hlo selects a TPU-v6e native MXU schedule only')
        self._v6_profile = tpu_v6.parse_compiled_hlo(compiled_hlo) if compiled_hlo is not None else None

    def _v6_mode(self, m, n, k):
        if self._v6_profile is not None:
            shape, mode = self._v6_profile
            if shape != (m,k,n):
                raise ValueError('Runtime M,K,N differ from compiled_hlo')
            return mode
        return tpu_v6.native_mode(m,k,n)

    def _shape(self, m, n, k):
        if min(m, n, k) <= 0:
            raise ValueError('Positive matrix dimensions required')
        if self.arch == 'TPU-v5e' and k not in (128, 256, 512, 1024, 2048):
            raise NotImplementedError('TPU v5e requires a validated K: 128/256/512/1024/2048')
        if self.arch == 'TPU-v6e':
            self._v6_mode(m,n,k)
        if self.arch == 'TPU-v7':
            valid_k = (128, 256, 512, 640, 768, 896, 1024, 1152, 1536, 2048) if self.fmt == 'bf16' else (128, 256, 512)
            if n != 128 or k not in valid_k:
                raise NotImplementedError('TPU v7 requires N128; low-format profiles require K128/256/512')
        if self.arch == 'Blackhole' and k != 32:
            raise NotImplementedError('Blackhole HiFi4 profile is scoped to K32')
        if self.arch == 'Trainium3':
            if self.mode == 'ordinary' and (k not in (128, 256, 512) or k % self.chunk):
                raise NotImplementedError('Ordinary Trainium3 profile requires K128/256/512')
            if self.mode in ('double_row', 'two_inst') and k != 256:
                raise NotImplementedError('Double-row/control profile requires logical K256')
            if self.mode == 'mx' and k != 512:
                raise NotImplementedError('MX profile requires logical K512')

    def dot_bits(self, a, b, c=0, *, m=128, n=128, scales_a=None, scales_b=None):
        """Return FP32 output bits from FP32 encodings of quantized operands.

        M,N select the captured kernel profile even for one scalar dot.
        For TPU v7, N must reflect the full output width, not a sampled column.
        """
        if len(a) != len(b):
            raise ValueError('Input lengths differ')
        self._shape(m, n, len(a))
        a, b = _validate_words(a, self.fmt), _validate_words(b, self.fmt)
        c = _validate_words([c], 'fp32')[0]
        if self.c_mode == 'zero' and c & 0x7fffffff:
            raise NotImplementedError('Nonzero C needs explicit c_mode="late_fp32"')
        if self.mode != 'mx' and (scales_a is not None or scales_b is not None):
            raise ValueError('Block scales are accepted only in MX mode')
        if self.arch == 'TPU-v5e':
            result = tpu_pe.gemm_dot([x >> 16 for x in a], [x >> 16 for x in b])
        elif self.arch == 'TPU-v6e':
            result = tpu_v6.dot_bf16([x >> 16 for x in a], [x >> 16 for x in b],
                                    m=m,n=n,profile_mode=self._v6_mode(m,n,len(a)))
        elif self.arch == 'TPU-v7':
            aa, bb = [x >> 16 for x in a], [x >> 16 for x in b]
            if self.fmt == 'bf16':
                result = tpu_v7.dot_bf16(aa, bb, m=m, n=n)
            else:
                low = self.fmt != 'e2m1fn' or self.backend == 'pallas'
                result = tpu_v7_low.dot(aa, bb, low=low)
        elif self.arch == 'Blackhole':
            result = blackhole.dot(a, b, phases=4)
        elif self.mode == 'ordinary':
            result = trainium.ordinary(a, b, self.chunk)
        else:
            row = {'logical_k': len(a), 'terms': [
                [i, str(Fraction(_float(x))), str(Fraction(_float(y)))]
                for i, (x, y) in enumerate(zip(a, b)) if x & 0x7fffffff and y & 0x7fffffff
            ]}
            if self.mode == 'mx':
                sa, sb = _scales(scales_a, len(a)//32), _scales(scales_b, len(a)//32)
                row['scales_unbiased'] = {i: [x, y] for i, (x, y) in enumerate(zip(sa, sb))}
            result = trainium.extension(row, 'mx' if self.mode == 'mx' else 'double_row',
                                       self.mapping, 'two_inst' if self.mode == 'two_inst' else 'single', self.parts)
        if self.c_mode == 'late_fp32':
            result = add32(result, c, ftz=self.arch.startswith('TPU'))
        return result

    def __call__(self, A, B, C=None, *, scales_a=None, scales_b=None):
        """CPU tensors A[M,K], B[K,N], optional C[M,N] -> FP32 tensor.

        MX scale exponents: scales_a[M,K/32], scales_b[K/32,N].
        The implementation is a scalar reference, with no autograd support.
        """
        tensors = [A, B] + ([] if C is None else [C])
        if any(not isinstance(x, torch.Tensor) or x.device.type != 'cpu' for x in tensors):
            raise ValueError('CPU PyTorch tensors required')
        if any(x.requires_grad for x in tensors):
            raise NotImplementedError('Reference simulator has no autograd; pass detached tensors')
        if any(not x.is_floating_point() for x in tensors):
            raise ValueError('Floating-point operand tensors required')
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
            raise ValueError('Expected A[M,K] and B[K,N]')
        m, k = A.shape; n = B.shape[1]
        self._shape(m, n, k)
        if C is not None and (C.shape != (m, n) or C.dtype != torch.float32):
            raise ValueError('C must be FP32 with shape [M,N]')
        # Reject precision loss on conversion to the FP32 transport representation.
        for x in (A, B):
            if x.dtype == torch.float64 and not torch.equal(x.float().double(), x):
                raise ValueError('FP64 values must be exactly representable in FP32')
        aa = A.float().contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
        bb = B.float().contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
        cc = [[0]*n for _ in range(m)] if C is None else C.contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
        sa = sb = None
        if self.mode == 'mx':
            for s, shape in ((scales_a, (m, k//32)), (scales_b, (k//32, n))):
                if not isinstance(s, torch.Tensor) or s.device.type != 'cpu' or tuple(s.shape) != shape or s.requires_grad:
                    raise ValueError('MX requires CPU exponent tensors [M,K/32] and [K/32,N]')
            sa, sb = scales_a.tolist(), scales_b.tolist()
        elif scales_a is not None or scales_b is not None:
            raise ValueError('Block scales are accepted only in MX mode')
        out = []
        for i in range(m):
            out.append([self.dot_bits(aa[i], [bb[t][j] for t in range(k)], cc[i][j], m=m, n=n,
                                     scales_a=None if sa is None else sa[i],
                                     scales_b=None if sb is None else [sb[t][j] for t in range(k//32)])
                        for j in range(n)])
        return torch.tensor(out, dtype=torch.int64).to(torch.int32).view(torch.float32)
