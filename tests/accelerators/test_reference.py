import json
from pathlib import Path
import unittest
import torch
from mmasim.accelerators import Matmul

CASES=json.loads(Path(__file__).with_name('measured.json').read_text())

def tensor(words):
    return torch.tensor(words,dtype=torch.int64).to(torch.int32).view(torch.float32)

class ReferenceTests(unittest.TestCase):
    def test_all_measured_scalar_bits(self):
        for r in CASES:
            with self.subTest(case=r['name']):
                got=Matmul(**r['config']).dot_bits(r['a'],r['b'],**r['extra'])
                self.assertEqual(f'{got:08x}',r['expected'])

    def test_tensor_transport_and_layout(self):
        # Distinct measured dots share a batch; compare only measured diagonals.
        for arch in ('TPU-v5e','TPU-v6e','Blackhole','Trainium3'):
            rows=[r for r in CASES if r['config']=={'arch':arch}][:2]
            if arch=='Trainium3':
                rows=[r for r in CASES if r['config']['arch']==arch and r['config']['fmt']=='bf16'][:2]
            self.assertEqual(len(rows),2)
            A=tensor([r['a'] for r in rows]);B=tensor([r['b'] for r in rows]).T
            got=Matmul(**rows[0]['config'])(A,B).view(torch.int32)
            for i,r in enumerate(rows):self.assertEqual(int(got[i,i])&0xffffffff,int(r['expected'],16))

    def test_extension_tensor_scale_transport(self):
        for r in CASES:
            if r['config'].get('mode') not in ('mx','double_row'):continue
            extra={k:torch.tensor([v]) if k=='scales_a' else torch.tensor(v).reshape(-1,1) for k,v in r['extra'].items()}
            got=Matmul(**r['config'])(tensor(r['a']).reshape(1,-1),tensor(r['b']).reshape(-1,1),**extra)
            self.assertEqual(int(got.view(torch.int32)[0,0])&0xffffffff,int(r['expected'],16))

    def test_v7_output_width_is_preserved(self):
        r=next(r for r in CASES if r['config']['arch']=='TPU-v7' and 'dense' in r['name'])
        A=tensor(r['a']).reshape(1,-1);B=tensor(r['b']).reshape(-1,1).expand(-1,128)
        got=Matmul('TPU-v7')(A,B).view(torch.int32).to(torch.int64)&0xffffffff
        self.assertTrue(torch.all(got==int(r['expected'],16)))
        with self.assertRaises(NotImplementedError):Matmul('TPU-v7')(A,B[:,:1])

    def test_c_policy_and_guards(self):
        A=torch.zeros((1,128));B=torch.zeros((128,1));C=torch.tensor([[1.]])
        with self.assertRaises(NotImplementedError):Matmul('Trainium3')(A,B,C)
        self.assertEqual(Matmul('Trainium3',c_mode='late_fp32')(A,B,C).item(),1.)
        with self.assertRaises(NotImplementedError):Matmul('Blackhole')(A,B)
        with self.assertRaises(NotImplementedError):Matmul('Trainium3')(A.requires_grad_(),B)
        with self.assertRaises(ValueError):Matmul('TPU-v5e').dot_bits([0x3f800001]*128,[0]*128)
        with self.assertRaises(ValueError):Matmul('Trainium3').dot_bits([0x7f800000]*128,[0]*128)
        with self.assertRaises(ValueError):Matmul('Trainium3','e4m3fn',mode='mx').dot_bits([0]*512,[0]*512)
        with self.assertRaises(ValueError):Matmul('Trainium3','e4m3fn',mode='mx').dot_bits([0]*512,[0]*512,scales_a=[128]*16,scales_b=[0]*16)

if __name__=='__main__':unittest.main()
