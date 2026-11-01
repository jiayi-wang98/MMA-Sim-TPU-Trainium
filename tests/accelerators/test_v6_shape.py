import json,unittest
from pathlib import Path
from mmasim.accelerators import Matmul

class V6ShapeTests(unittest.TestCase):
    def test_measured_native_pair_chain(self):
        for r in json.loads(Path(__file__).with_name('v6-shape-measured.json').read_text()):
            self.assertEqual(Matmul('TPU-v6e').dot_bits(r['a'],r['b'],n=128),int(r['expected'],16))

    def test_unresolved_native_shape_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            Matmul('TPU-v6e').dot_bits([0]*777,[0]*777,m=333,n=222)
        with self.assertRaises(NotImplementedError):
            Matmul('TPU-v6e').dot_bits([0]*256,[0]*256,m=128,n=1)
        self.assertEqual(Matmul('TPU-v6e').dot_bits([0]*128,[0]*128,n=64),0)

    def test_large_m_native_path_is_supported_by_the_cross_half_model(self):
        self.assertEqual(Matmul('TPU-v6e').dot_bits([0]*256,[0]*256,m=4096,n=128),0)
        for m in (64,128,256):
            self.assertEqual(Matmul('TPU-v6e').dot_bits([0]*256,[0]*256,m=m,n=128),0)
