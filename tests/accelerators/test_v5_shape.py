import json,unittest
from pathlib import Path
from mmasim.accelerators import Matmul

class V5ShapeTests(unittest.TestCase):
    def test_measured_long_k_and_v6_scope(self):
        cases=json.loads(Path(__file__).with_name('v5-shape-measured.json').read_text())
        for r in cases:
            self.assertEqual(Matmul('TPU-v5e').dot_bits(r['a'],r['b'],m=64,n=64),int(r['expected'],16))
            with self.assertRaises(NotImplementedError):
                Matmul('TPU-v6e').dot_bits(r['a'],r['b'],m=64,n=64)
