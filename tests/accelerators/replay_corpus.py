"""Replay retained hardware corpus through the public adapter (no device calls)."""
import argparse,json,struct,collections
from fractions import Fraction
from pathlib import Path
from mmasim.accelerators import Matmul

parser=argparse.ArgumentParser()
parser.add_argument('data_root',type=Path)
parser.add_argument('--output',type=Path,default=Path('accelerator-replay.json'))
args=parser.parse_args();root=args.data_root;counts={}
formats={'bfloat16':'bf16','float16':'fp16','float32':'fp32','float8_e4m3':'e4m3','float8_e4m3fn':'e4m3fn','float8_e5m2':'e5m2','float8_e4m3fn_x4':'e4m3fn','float8_e5m2_x4':'e5m2','float4_e2m1fn_x4':'e2m1fn'}
def bits(x):return struct.unpack('!I',struct.pack('!f',float(x)))[0]
def check(label,sim,a,b,expected,**extra):
 stats=counts.setdefault(label,{'cases':0,'mismatches':0})
 stats['cases']+=1
 pred=sim.dot_bits(a,b,**extra)
 stats['mismatches']+=f'{pred:08x}'!=expected
for p in sorted((root/'trainium_result/full-r1-return/full-r1').glob('*/observations.jsonl')):
 cfg=json.loads(p.with_name('job.json').read_text());sim=Matmul('Trainium3',formats[cfg['dtype']],chunk=cfg['chunk'])
 for r in map(json.loads,p.open()):check('ordinary/'+p.parent.name,sim,r['a_fp32_bits'],r['b_fp32_bits'],next(iter(r['observed'][0])))
 print(p.parent.name,counts['ordinary/'+p.parent.name],flush=True)
for p in sorted((root/'trainium_result/ext-r1-return/ext-r1').glob('*/observations.jsonl')):
 cfg=json.loads(p.with_name('job.json').read_text());mode='mx' if cfg['kind']=='mx' else cfg['route']
 sim=Matmul('Trainium3',formats[cfg['format']],mode=mode,mapping=cfg['mapping'],parts=cfg['parts_per_instruction'])
 for r in map(json.loads,p.open()):
  a=[0]*r['logical_k'];b=a.copy();extra={}
  for t,x,y in r['terms']:a[t]=bits(Fraction(x));b[t]=bits(Fraction(y))
  if mode=='mx':
   scales={int(k):v for k,v in r.get('scales_unbiased',{}).items()}
   extra={'scales_a':[scales.get(i,[0,0])[0] for i in range(len(a)//32)],'scales_b':[scales.get(i,[0,0])[1] for i in range(len(a)//32)]}
  check('extension/'+p.parent.name,sim,a,b,r['observed'][0],**extra)
 print(p.parent.name,counts['extension/'+p.parent.name],flush=True)
sim=Matmul('Blackhole')
for r in map(json.loads,(root/'results/blackhole/phase-fresh-hifi4/observations.jsonl').open()):
 check('Blackhole/fresh-hifi4',sim,r['a_bits'],r['b_bits'],next(iter(r['observed'][0])))
sim=Matmul('TPU-v7')
for p in sorted((root/'artifacts/tpu_v7_round3/runs/v7-round3-full').glob('*-n128-*-native/observations.jsonl')):
 cfg=json.loads(p.with_name('arguments.json').read_text())
 for r in map(json.loads,p.open()):check('TPU-v7/'+p.parent.name,sim,r['a_fp32_bits'],r['b_fp32_bits'],next(iter(r['observed'][0])),m=cfg['m'],n=cfg['n'])
result={'cases':sum(v['cases'] for v in counts.values()),'mismatches':sum(v['mismatches'] for v in counts.values()),'jobs':counts}
assert sum(v['cases'] for k,v in counts.items() if k.startswith(('ordinary/','extension/')))==98200
assert result['mismatches']==0,result
args.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='jobs'},flush=True)
