from pathlib import Path
import json
from mmasim.accelerators import Matmul
import argparse
parser=argparse.ArgumentParser();parser.add_argument('data_root',type=Path);parser.add_argument('--output',type=Path,default=Path('v6-shape-validation.json'));args=parser.parse_args()
ROOT=args.data_root;model=Matmul('TPU-v6e');counts={'supported_cases':0,'unsupported_cases':0,'mismatches':0}
for p in sorted((ROOT/'artifacts/v6-shape-20261031').glob('k*/observations.jsonl')):
 cfg=json.loads(p.with_name('arguments.json').read_text())
 for r in map(json.loads,p.open()):
  y=model.dot_bits(r['a_fp32_bits'],r['b_fp32_bits'],m=cfg['m'],n=cfg['n']);counts['supported_cases']+=1
  counts['mismatches']+=any(set(h)!={f'{y:08x}'} for h in r['observed'])
assert counts=={'supported_cases':4176,'unsupported_cases':0,'mismatches':0},counts
controls={}
for p in sorted((ROOT/'artifacts/v6-shape-controls-20261031').glob('*/observations.jsonl')):
 n=bad=0
 for r in map(json.loads,p.open()):
  n+=1
  y=Matmul('TPU-v5e').dot_bits(r['a_fp32_bits'],r['b_fp32_bits'])
  assert r['predictions']['sequential']==f'{y:08x}'
  bad+=any(set(h)!={f'{y:08x}'} for h in r['observed'])
 assert bad==0;controls[p.parent.name]={'cases':n,'mismatches':bad}
counts['explicit_sequential_controls']=controls
args.output.write_text(json.dumps(counts,indent=2)+'\n');print(counts)
