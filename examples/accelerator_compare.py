"""Compare the same 32 active BF16 products in three captured profiles."""
import torch
from mmasim.accelerators import Matmul

torch.manual_seed(710)
A=torch.zeros(1,128,dtype=torch.bfloat16)
B=torch.zeros(128,4,dtype=torch.bfloat16)
A[:,:32]=torch.randn(1,32).to(torch.bfloat16)
B[:32,:]=torch.randn(32,4).to(torch.bfloat16)
reference=(A.double()@B.double()).float()
for arch in ('TPU-v6e','Blackhole','Trainium3'):
    model=Matmul(arch,'bf16')
    result=model(A[:,:32],B[:32,:]) if arch=='Blackhole' else model(A,B)
    print(arch, 'output', result.tolist(), 'max_abs_error', (result-reference).abs().max().item())
