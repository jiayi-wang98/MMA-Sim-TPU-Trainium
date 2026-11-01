"""Check the address provenance supporting the instruction-group descriptors."""
from collections import Counter
from pathlib import Path
import unittest
from llo_trace import trace, leaves

BASE = Path(__file__).parent/'v6-llo'


class TPUOperandTraceTests(unittest.TestCase):
    def test_regular_shapes_cover_inputs_and_outputs(self):
        checked = 0
        for f in sorted(BASE.glob('*fusion-01-final_bundles.txt')):
            r = trace(f)
            if r['input_shapes']['A'][0] != 128:
                continue
            checked += 1
            def tilekey(t):return t['tensor'],tuple(t['rows']),tuple(t['cols'])
            def expected(name):
                m,n = r['input_shapes'][name]
                return {(name,(i,i+8),(j,j+128)) for i in range(0,m,8) for j in range(0,n,128)}
            ac = Counter(tilekey(t) for e in r['events'] if e['opcode'].startswith('vmatmul.')
                         for t in leaves([e['prep'],e['operand']]))
            self.assertEqual(set(ac), expected('A'))
            self.assertEqual(set(ac.values()), {1})
            for mxu in ('mxu0','mxu1'):
                bs = {tilekey(t) for e in r['events'] if e['opcode'].startswith('vmatpush') and e['mxu']==mxu
                      for t in leaves([e['prep'],e['operand']])}
                self.assertEqual(bs, expected('B'))
            cs = Counter((tuple(s['rows']),tuple(s['cols'])) for s in r['stores'])
            self.assertEqual(set(cs.values()), {1})
            self.assertEqual(set(cs), {((i,i+8),(j,j+128)) for i in range(0,128,8)
                                      for j in range(0,r['input_shapes']['B'][1],128)})
            self.assertEqual(r['unstored_pops'], [])
        self.assertEqual(checked,3)

    def test_padding_and_discarded_columns(self):
        r = next(trace(f) for f in BASE.glob('*fusion-01-final_bundles.txt') if '[257,256]' in f.read_text())
        pushes = [e for e in r['events'] if e['opcode'].startswith('vmatpush')]
        self.assertEqual(len(pushes),32)
        for e in pushes:
            self.assertTrue(e['prep']['operand']['constant'].startswith('0.0'))
        self.assertEqual(len(r['stores']),33)
        self.assertEqual(len(r['unstored_pops']),33)
