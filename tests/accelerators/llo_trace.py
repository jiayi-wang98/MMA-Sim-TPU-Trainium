"""Trace tiled memory provenance in the captured, straight-line v6e bundles.
This is a provenance decoder, not an instruction-set simulator. vcombine.low
retains both source tile sets; it does not claim a per-lane ordering.
"""
import argparse,json,re
from pathlib import Path

def trace(path):
    text=path.read_text(); lines=text.splitlines()
    header=lines[9]
    shapes=re.findall(r'(%s\w+) = inlined_call_operand\.(vmem|hbm) \[shape: bf16\[(\d+),(\d+)\], index: (\d+)',header)
    bases={}; dims={}
    for reg,kind,r,c,idx in shapes:
        label='AB'[int(idx)]; dims[label]=(int(r),int(c))
        bases[reg if kind=='vmem' else '#allocation1']=label
    outshape=re.search(r'shape: f32\[(\d+),(\d+)\]',header)
    outn=int(outshape[2]); defs={}; preps={}; events=[]; stores=[]; pops={}; adds={}; memory={}
    def prov(reg): return defs.get(reg,{'unresolved':reg})
    def tiles(base,offset):
        label=bases.get(base)
        if label is None:return {'unresolved_base':base,'offset':offset}
        assert offset%4==0
        tile=offset//4; nc=(dims[label][1]+127)//128
        r,c=divmod(tile,nc)
        return {'tensor':label,'rows':[r*8,r*8+8],'cols':[c*128,c*128+128], 'vmem_offset':offset}
    for lineno,line in enumerate(lines,1):
        # Each assignment occupies one VLIW slot, separated by ;;.
        for m in re.finditer(r'(%\w+) = ([\w.]+)(.*?)(?=\s+;;|\s+\}|$)',line):
            dst,op,args=m.groups(); regs=re.findall(r'%\w+',args)
            if op=='vld':
                a=re.search(r'vmem:\[([^\] +]+)(?: \+ \$(0x[\da-f]+))?\] sm:\$(0x[\da-f]+)',args)
                if a:
                    base,off,mask=a.groups()
                    offset=int(off or '0',0)
                    defs[dst]=memory.get((base,offset),{'load':tiles(base,offset),'mask':mask,'line':lineno})
            elif op=='vmov' and not regs:
                defs[dst]={'constant':args.strip(),'line':lineno}
            elif op.startswith('vunpack.'):
                defs[dst]={'unpack':op,'source':prov(regs[0]),'line':lineno}
            elif op=='vcombine.low':
                defs[dst]={'combine_low':[prov(x) for x in regs], 'line':lineno}
            elif op.startswith('vmatprep.'):
                mxu=op.split('.')[-1];kind='lhs' if '.mubr.' in op else 'rhs'
                preps[(mxu,kind)]={'opcode':op,'operand':prov(regs[0]),'line':lineno}
            elif op.startswith(('vmatmul.','vmatpush')):
                mxu=op.split('.')[-1];kind='lhs' if op.startswith('vmatmul.') else 'rhs'
                events.append({'opcode':op,'mxu':mxu,'line':lineno,'prep':preps.pop((mxu,kind),None),'operand':prov(regs[0])})
            elif op.startswith('vpop.'):
                mxu=op.split('.')[-1];num=sum(x['mxu']==mxu for x in pops.values())
                pops[dst]={'mxu':mxu,'ordinal':num,'line':lineno}
            elif op=='vadd.f32':
                adds[dst]={'sources':[pops.get(x,{'unresolved':x}) for x in regs],'line':lineno}
            elif op=='vst':
                mem=re.search(r'vmem:\[([^\] +]+)(?: \+ \$(0x[\da-f]+))?\]',args)
                if mem:memory[(mem[1],int(mem[2] or '0',0))]=prov(regs[-1])
                a=re.search(r'vmem:\[#allocation4(?: \+ \$(0x[\da-f]+))?\] sm:\$(0x[\da-f]+)',args)
                if a:
                    off=int(a[1] or '0',0);tile=off//8;r,c=divmod(tile,(outn+127)//128)
                    stores.append({'rows':[r*8,r*8+8],'cols':[c*128,c*128+128], 'mask':a[2], 'line':lineno,'source':pops.get(regs[-1],adds.get(regs[-1],{'unresolved':regs[-1]}))})
    used=set()
    for s in stores:
        src=s['source']; used.update((x['mxu'],x['ordinal']) for x in src.get('sources',[src]) if 'mxu' in x)
    return {'file':str(path),'input_shapes':dims,'assumptions':['VMEM immediate unit=512 bytes, calibrated by BF16 tile stride 4 and FP32 tile stride 8','BF16 tiled layout (8,128)(2,1); FP32 tiled layout (8,128)','vcombine.low joins tile provenance; lane order is not decoded','nearest preceding prep on same MXU supplies associated state'], 'events':events,'stores':stores,'unstored_pops':[v for v in pops.values() if (v['mxu'],v['ordinal']) not in used]}

def leaves(x):
    if isinstance(x,dict):
        if 'load' in x:yield x['load']
        else:
            for v in x.values():yield from leaves(v)
    elif isinstance(x,list):
        for v in x:yield from leaves(v)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dump',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    results=[trace(f) for f in sorted(a.dump.glob('*fusion-01-final_bundles.txt'))]
    a.out.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(Path(r['file']).name,r['input_shapes'])
        for e in r['events']:
            if e['opcode'].startswith('vmatmul.') and e['mxu']=='mxu0':
                print(' line',e['line'],'prep:',list(leaves(e['prep'])),'mul:',list(leaves(e['operand'])))
        print('stores',len(r['stores']),'unstored pops',len(r['unstored_pops']))
