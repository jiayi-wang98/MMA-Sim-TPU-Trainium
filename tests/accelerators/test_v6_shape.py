import json,unittest
from pathlib import Path
from mmasim.accelerators import Matmul

class V6ShapeTests(unittest.TestCase):
    def test_measured_native_pair_chain(self):
        for r in json.loads(Path(__file__).with_name('v6-shape-measured.json').read_text()):
            self.assertEqual(Matmul('TPU-v6e').dot_bits(r['a'],r['b'],n=128),int(r['expected'],16))

    def test_unresolved_native_shape_is_rejected(self):
        for k in (256,512,1024,2048):
            for n in (64,129,256):
                with self.assertRaises(NotImplementedError):Matmul('TPU-v6e').dot_bits([0]*k,[0]*k,n=n)
        self.assertEqual(Matmul('TPU-v6e').dot_bits([0]*128,[0]*128,n=64),0)

    def test_large_m_native_path_is_not_the_small_m_profile(self):
        # M4096,K256,N128 has 21/32 measured dense failures despite N128.
        for k in (256,512,1024,2048):
            with self.assertRaises(NotImplementedError):
                Matmul('TPU-v6e').dot_bits([0]*k,[0]*k,m=4096,n=128)
        for m in (64,128,256):
            self.assertEqual(Matmul('TPU-v6e').dot_bits([0]*256,[0]*256,m=m,n=128),0)
