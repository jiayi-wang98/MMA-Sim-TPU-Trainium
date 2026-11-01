"""Replay sampled/full-small random v5e MXU outputs through the public adapter."""
import argparse,json,re
from pathlib import Path
import numpy as np
from mmasim.accelerators import Matmul

p=argparse.ArgumentParser();p.add_argument('data_root',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
base=args.data_root/'artifacts/v5-schedule-heldout';jobs=[]
for row in json.loads((base/'status.json').read_text()):
    tag=row['tag'];m,k,n=row['shape'];hlo=(base/(tag+'.hlo')).read_text()
    if not re.search(r'\bconvolution\(',hlo):continue
    sim=Matmul('TPU-v5e',compiled_hlo=hlo);x=np.load(base/(tag+'.npz'));bad=0
    for i,j in x['cells']:
        a=(x['a_bf16'][i].astype(np.uint32)<<16).tolist()
        b=(x['b_bf16'][:,j].astype(np.uint32)<<16).tolist()
        bad+=sim.dot_bits(a,b,m=m,n=n)!=int(x['bits'][i,j])
    jobs.append({'shape':[m,k,n],'cases':len(x['cells']),'mismatches':bad})
summary={'jobs':jobs,'configurations':len(jobs),'dot_comparisons':sum(r['cases'] for r in jobs),'mismatches':sum(r['mismatches'] for r in jobs)}
assert summary['mismatches']==0,summary
if args.output:args.output.write_text(json.dumps(summary,indent=2)+'\n')
print({k:v for k,v in summary.items() if k!='jobs'})
