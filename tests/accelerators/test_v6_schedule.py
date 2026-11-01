import json
import unittest
from pathlib import Path
from unittest.mock import patch
from mmasim.accelerators import Matmul
from mmasim.accelerators.reference import tpu_v6

BASE=Path(__file__).parent

class V6ScheduleTests(unittest.TestCase):
    def test_measured_tails_transposes_and_full_random_outputs(self):
        for r in json.loads((BASE/'v6-schedule-measured.json').read_text()):
            m,k,n=r['shape']
            self.assertEqual(Matmul('TPU-v6e').dot_bits(r['a'],r['b'],m=m,n=n),int(r['expected'],16))
            hlo=(BASE/r['hlo']).read_text()
            with patch.dict(tpu_v6.PROFILES,{},clear=True):
                sim=Matmul('TPU-v6e',compiled_hlo=hlo)
                self.assertEqual(sim.dot_bits(r['a'],r['b'],m=m,n=n),int(r['expected'],16))

    def test_native_hlo_scope_and_runtime_shape(self):
        for name in ('vpu-reduction.hlo','device-loop.hlo'):
            with self.assertRaises(NotImplementedError):
                Matmul('TPU-v6e',compiled_hlo=(BASE/'v6-hlo'/name).read_text())
        hlo=(BASE/'v6-hlo/m128-k385-n120.hlo').read_text()
        sim=Matmul('TPU-v6e',compiled_hlo=hlo)
        with self.assertRaises(ValueError):sim.dot_bits([0]*385,[0]*385,m=64,n=120)
        with self.assertRaises(ValueError):Matmul('Blackhole',compiled_hlo=hlo)
