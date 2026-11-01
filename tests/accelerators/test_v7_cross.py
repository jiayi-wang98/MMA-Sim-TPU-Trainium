import unittest,json
from pathlib import Path
from mmasim.accelerators import Matmul
BASE=Path(__file__).parent
class V7CrossTests(unittest.TestCase):
    def test_measured_cross_half_outputs(self):
        for row in json.loads((BASE/'v7-cross-measured.json').read_text()):
            m,k,n=row['shape']
            self.assertEqual(Matmul('TPU-v7').dot_bits(row['a'],row['b'],m=m,n=n),int(row['expected'],16),row['job'])
    def test_unmeasured_profiles_rejected(self):
        for kwargs,m,k,n in [({},64,512,64),({},128,1024,64),({'backend':'pallas'},64,1024,64),({'fmt':'e4m3fn'},64,1024,64)]:
            with self.assertRaises(NotImplementedError):
                Matmul('TPU-v7',**kwargs).dot_bits([0]*k,[0]*k,m=m,n=n)
