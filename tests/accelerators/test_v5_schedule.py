import json
import unittest
from pathlib import Path
import torch
from mmasim.accelerators import Matmul

BASE=Path(__file__).parent

class V5ScheduleTests(unittest.TestCase):
    def test_measured_irregular_k_and_large_shapes(self):
        for row in json.loads((BASE/'v5-schedule-measured.json').read_text()):
            m,k,n=row['shape']
            for sim in (Matmul('TPU-v5e'),Matmul('TPU-v5e',compiled_hlo=(BASE/row['hlo']).read_text())):
                self.assertEqual(sim.dot_bits(row['a'],row['b'],m=m,n=n),int(row['expected'],16))

    def test_small_native_matrix_end_to_end(self):
        row=json.loads((BASE/'v5-small-matrix.json').read_text())
        def tensor(bits):return torch.tensor(bits,dtype=torch.int64).to(torch.int32).view(torch.float32).to(torch.bfloat16)
        result=Matmul('TPU-v5e')(tensor(row['a']),tensor(row['b']))
        bits=result.view(torch.int32).to(torch.int64)&0xffffffff
        self.assertTrue(torch.equal(bits,torch.tensor(row['expected'],dtype=torch.int64)))

    def test_non_mxu_and_wrong_hlo_shape_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            Matmul('TPU-v5e',compiled_hlo=(BASE/'v5-hlo/vpu-reduction.hlo').read_text())
        with self.assertRaises(NotImplementedError):
            Matmul('TPU-v5e').dot_bits([0]*256,[0]*256,m=128,n=1)
        sim=Matmul('TPU-v5e',compiled_hlo=(BASE/'v5-hlo/m128-k385-n120.hlo').read_text())
        with self.assertRaises(ValueError):sim.dot_bits([0]*385,[0]*385,m=128,n=128)
