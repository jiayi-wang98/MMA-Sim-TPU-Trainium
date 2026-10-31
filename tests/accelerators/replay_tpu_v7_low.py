"""Replay original TPU7x DEFAULT low-format jobs through the public adapter."""
import argparse,json,hashlib
from pathlib import Path
from mmasim.accelerators import Matmul

parser=argparse.ArgumentParser()
parser.add_argument('data_root',type=Path)
parser.add_argument('--output',type=Path,default=Path('tpu-v7-low-replay.json'))
parser.add_argument('--fixtures',type=Path)
args=parser.parse_args()
formats={'float8_e4m3fn':'e4m3fn','float8_e5m2':'e5m2','float4_e2m1fn':'e2m1fn'}
jobs={};fixtures=[]
for p in sorted((args.data_root/'tpuv7_result').glob('v7-*/target-*/observations.jsonl')):
 cfg=json.loads(p.with_name('arguments.json').read_text())
 if cfg['dtype'] not in formats:continue
 config={'arch':'TPU-v7','fmt':formats[cfg['dtype']],'backend':cfg['backend']}
 sim=Matmul(**config);count=bad=0;selected=set()
 for r in map(json.loads,p.open()):
  pred=sim.dot_bits(r['a_fp32_bits'],r['b_fp32_bits'])
  count+=1
  failed=any(set(h)!={f'{pred:08x}'} for h in r['observed']);bad+=failed
  if r['family'] in ('dense','rounding','native_cancel') and r['family'] not in selected:
   selected.add(r['family'])
   fixtures.append({'name':p.parent.name+'/'+r['name'],'config':config,'a':r['a_fp32_bits'],'b':r['b_fp32_bits'],'expected':next(iter(r['observed'][0])),'family':r['family']})
  if failed:raise AssertionError((p.parent.name,r['name'],hex(pred),r['observed']))
 key=p.parent.parent.name+'/'+p.parent.name
 jobs[key]={'cases':count,'mismatches':bad,'observation_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(key,count,bad,flush=True)
result={'cases':sum(v['cases'] for v in jobs.values()),'mismatches':sum(v['mismatches'] for v in jobs.values()),'jobs':jobs}
assert len(jobs)>=18 and result['cases']>0
args.output.write_text(json.dumps(result,indent=2)+'\n')
if args.fixtures:args.fixtures.write_text(json.dumps(fixtures,separators=(',',':'))+'\n')
print({k:v for k,v in result.items() if k!='jobs'},flush=True)
