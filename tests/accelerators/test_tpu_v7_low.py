import json
from pathlib import Path
import unittest
import torch
from mmasim.accelerators import Matmul

CASES=json.loads(Path(__file__).with_name('tpu-v7-low-measured.json').read_text())

def tensor(words):
    return torch.tensor(words,dtype=torch.int64).to(torch.int32).view(torch.float32)

class TPULowTests(unittest.TestCase):
    def test_saved_bits_all_formats_routes_and_k(self):
        covered=set()
        for r in CASES:
            with self.subTest(case=r['name']):
                self.assertEqual(Matmul(**r['config']).dot_bits(r['a'],r['b']),int(r['expected'],16))
                covered.add((r['config']['fmt'],r['config']['backend'],len(r['a'])))
        self.assertEqual(len(covered),18)

    def test_torch_fp8_and_decoded_fp4_transport(self):
        covered=set()
        for r in CASES:
            key=(r['config']['fmt'],r['config']['backend'])
            if key in covered or len(r['a'])!=128 or r['family']!='dense':continue
            covered.add(key)
            dtype={'e4m3fn':torch.float8_e4m3fn,'e5m2':torch.float8_e5m2,'e2m1fn':torch.float32}[key[0]]
            A=tensor(r['a']).reshape(1,-1).to(dtype)
            B=tensor(r['b']).reshape(-1,1).expand(-1,128).to(dtype)
            got=Matmul(**r['config'])(A,B).view(torch.int32).to(torch.int64)&0xffffffff
            self.assertTrue(torch.all(got==int(r['expected'],16)))
        self.assertEqual(len(covered),6)

    def test_fp4_backend_selection_changes_measured_result(self):
        by_input={};differences=0
        for r in CASES:
            if r['config']['fmt']!='e2m1fn':continue
            key=(tuple(r['a']),tuple(r['b']))
            by_input.setdefault(key,{})[r['config']['backend']]=r
        for pair in by_input.values():
            if set(pair)!= {'jax','pallas'}:continue
            j,p=pair['jax'],pair['pallas']
            if j['expected']==p['expected']:continue
            differences+=1
            self.assertEqual(Matmul('TPU-v7','e2m1fn').dot_bits(j['a'],j['b']),int(j['expected'],16))
            self.assertEqual(Matmul('TPU-v7','e2m1fn',backend='pallas').dot_bits(j['a'],j['b']),int(p['expected'],16))
        self.assertGreater(differences,0)

    def test_scope_and_scale_guards(self):
        for fmt in ('e4m3fn','e5m2','e2m1fn'):
            sim=Matmul('TPU-v7',fmt)
            with self.assertRaises(NotImplementedError):sim.dot_bits([0]*1024,[0]*1024)
            with self.assertRaises(NotImplementedError):sim.dot_bits([0]*128,[0]*128,n=256)
            with self.assertRaises(ValueError):sim.dot_bits([0]*128,[0]*128,scales_a=[0]*4,scales_b=[0]*4)
        with self.assertRaises(NotImplementedError):Matmul('TPU-v7','e2m1fn',mode='mx')
        with self.assertRaises(ValueError):Matmul('TPU-v7','e2m1fn',backend='unknown')
        with self.assertRaises(ValueError):Matmul('Trainium3',backend='jax')
        with self.assertRaises(NotImplementedError):Matmul('TPU-v7','fp16')
        with self.assertRaises(NotImplementedError):Matmul('TPU-v7','bf16',backend='pallas')
        with self.assertRaises(ValueError):Matmul('TPU-v7','e2m1fn').dot_bits([0x40500000]*128,[0]*128)

if __name__=='__main__':unittest.main()
