"""v6e BF16 MXU model: DP4 arithmetic plus two K256 reduction schedules.

Compiler windows select cross128 or hybrid; hybrid switches short tail blocks
below 249 terms to cross128. Every K256 result is accumulated sequentially.
Inputs are finite BF16 encodings; DEFAULT precision, FP32 result.
"""
import json
import re
from .tpu import add32, dot_bits
from .tpu_v6_profiles import PROFILES

def effective_shape_from_hlo(hlo):
    """Return output batch/features of one native rank-2 MXU convolution."""
    matches=re.findall(r'f32\[(\d+),(\d+)\][^\n]*?\bconvolution\([^\n]*?dim_labels=[^\n]*?->(bf|fb)',hlo)
    if len(matches)!=1:raise ValueError('Expected one native rank-2 MXU convolution')
    x,y,axes=matches[0];x,y=int(x),int(y)
    return (x,y) if axes=='bf' else (y,x)

def mode_from_hlo(hlo):
    effective_shape_from_hlo(hlo)
    args=re.findall(r'bf16\[(\d+),(\d+)\]',hlo.splitlines()[0])
    if len(args)!=2:raise ValueError('Expected two BF16 matrix inputs')
    k=int(args[0][1])
    configs=[json.loads(line.split('backend_config=',1)[1]) for line in hlo.splitlines()
             if 'convolution_algorithm_config' in line and 'backend_config=' in line]
    if len(configs)!=1:raise ValueError('Expected one MXU window configuration')
    window=configs[0]['window_config']
    kernel=list(map(int,window['kernel_window_bounds']))
    output=list(map(int,window['output_window_bounds']))
    if kernel==[(k+7)//8,1] and len(output)==2 and output[0]<=32 and output[1]==1:
        return 'hybrid'
    return 'cross128'

def fold(values):
    acc=0
    for v in values:acc=add32(acc,v)
    return acc

def block256(a,b,mode='cross128'):
    a=list(a)+[0]*(256-len(a));b=list(b)+[0]*(256-len(b))
    p=[dot_bits(a[i:i+4],b[i:i+4]) for i in range(0,256,4)]
    if mode=='independent128':
        lo=fold([add32(p[i],p[i+1]) for i in range(0,32,2)])
        hi=fold([add32(p[i],p[i+1]) for i in range(32,64,2)])
        return add32(lo,hi)
    if mode!='cross128':raise ValueError(mode)
    return fold([add32(add32(p[i],p[i+32]),add32(p[i+1],p[i+33]))
                 for i in range(0,32,2)])

def candidate(a,b,m,n,*,mode=None,outer='sequential'):
    if len(a)!=len(b):raise ValueError('Equal K required')
    if mode is None:mode=native_mode(m,len(a),n)
    parts=[]
    for i in range(0,len(a),256):
        aa=a[i:i+256];bb=b[i:i+256]
        local_mode=('independent128' if len(aa)>=249 else 'cross128') if mode=='hybrid' else mode
        parts.append(block256(aa,bb,local_mode))
    if outer=='paired':parts=[add32(parts[i],parts[i+1]) if i+1<len(parts) else parts[i] for i in range(0,len(parts),2)]
    elif outer=='balanced':
        while len(parts)>1:parts=[add32(parts[i],parts[i+1]) if i+1<len(parts) else parts[i] for i in range(0,len(parts),2)]
    elif outer!='sequential':raise ValueError(outer)
    return fold(parts)

def parse_compiled_hlo(hlo):
    """Read one captured native BF16/FP32 MXU call, rejecting VPU/composites."""
    if not isinstance(hlo, str) or not hlo.strip():
        raise ValueError('compiled_hlo must be nonempty HLO text')
    if re.search(r'\bwhile\(', hlo):
        raise NotImplementedError('Provide a native single-dot HLO, not a device loop')
    args = re.findall(r'bf16\[(\d+),(\d+)\]', hlo.splitlines()[0])
    out = re.search(r'->f32\[(\d+),(\d+)\]', hlo.splitlines()[0])
    if len(args) != 2 or out is None:
        raise NotImplementedError('Expected native BF16 matrix inputs and FP32 matrix output')
    m,k = map(int,args[0]); kb,n = map(int,args[1])
    if k != kb or tuple(map(int,out.groups())) != (m,n):
        raise ValueError('Inconsistent GEMM input/output shapes in HLO')
    try:
        mode = mode_from_hlo(hlo)
    except ValueError as exc:
        raise NotImplementedError('HLO is not a supported native MXU call: '+str(exc)) from exc
    return (m,k,n),mode


def native_mode(m,k,n):
    if min(m,k,n) < 2:
        raise NotImplementedError('Degenerate non-MXU paths are outside the v6e model')
    if (m,k,n) in PROFILES:
        return PROFILES[m,k,n]
    if k <= 128:
        return 'cross128'  # the two MXU schedules coincide in one K128 half
    raise NotImplementedError('Uncaptured v6e compiler schedule; provide compiled_hlo for the native MXU call')


def dot_bf16(a,b,*,m=128,n,profile_mode=None):
    if len(a)!=len(b) or not a:
        raise ValueError('Equal nonempty BF16 vectors required')
    mode = native_mode(m,len(a),n) if profile_mode is None else profile_mode
    return candidate(a,b,m,n,mode=mode)
