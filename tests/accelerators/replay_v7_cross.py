"""Replay all newly resolved native shapes through the public MMA-Sim API."""
import argparse,json,hashlib
from pathlib import Path
from mmasim.accelerators import Matmul
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
base=args.root/'artifacts/tpu_v7_round3/runs/v7-round3-full';jobs={};sim=Matmul('TPU-v7')
for d in sorted(base.iterdir()):
 if not (d/'observations.jsonl').exists():continue
 cfg=json.loads((d/'arguments.json').read_text());m,n=cfg['m'],cfg['n']
 if n==128:continue
 bad=count=0
 for line in (d/'observations.jsonl').open():
  r=json.loads(line);result=sim.dot_bits(r['a_fp32_bits'],r['b_fp32_bits'],m=m,n=n)
  bad+={result}!={int(x,16) for h in r['observed'] for x in h};count+=1
 jobs[d.name]={'cases':count,'mismatches':bad,'observations_sha256':hashlib.sha256((d/'observations.jsonl').read_bytes()).hexdigest()}
result={'cases':sum(x['cases'] for x in jobs.values()),'mismatches':sum(x['mismatches'] for x in jobs.values()),'jobs':jobs}
args.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='jobs'});assert result['mismatches']==0
