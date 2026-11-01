"""Instruction-group arithmetic against saved hardware outputs and layout controls."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from mmasim.tpu import MMA
from mmasim.accelerators.reference import tpu_v6
from mmasim.accelerators.reference.tpu import add32

BASE = Path(__file__).parent
OP = 'vmatmul.mubr.bf16.gmra'


def instruction(setup):
    return MMA('TPU-v6e', OP, weight_load=setup)


def compose(a, b, mode):
    """Compiler-level composition, deliberately outside the instruction class."""
    dual, wide = instruction('vmatpush3'), instruction('vmatpush1+vmatpush2')
    acc = 0
    for start in range(0, len(a), 256):
        aa, bb = a[start:start+256], b[start:start+256]
        independent = mode == 'hybrid' and len(aa) >= 249
        aa += [0]*(256-len(aa)); bb += [0]*(256-len(bb))
        if independent:
            value = add32(dual.dot_bits(aa[:128], bb[:128]),
                          dual.dot_bits(aa[128:], bb[128:]))
        else:
            value = wide.dot_bits(aa, bb)
        acc = add32(acc, value)
    return acc


class TPUInstructionTests(unittest.TestCase):
    def test_captured_shapes_matrix_outputs(self):
        fixture = json.loads((BASE/'v6-instruction-measured.json').read_text())
        for row in fixture['cases']:
            _, k, n = row['shape']
            bank = fixture['banks'][str(k)]
            a = torch.tensor(bank['a'], dtype=torch.int64).to(torch.int32).view(torch.float32)[row['row_index']]
            b = torch.tensor(bank['b'], dtype=torch.int64).to(torch.int32).view(torch.float32)[row['col_index']].T
            op = instruction(row['weight_load'])
            if op.groups == 2 and k == 128:
                result = op(a.reshape(2,16,128), torch.stack((b,b))).reshape(32,128)
            elif op.groups == 2:
                partial = op(torch.stack((a[:,:128], a[:,128:])), torch.stack((b[:128], b[128:])))
                words = partial.view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).reshape(2,-1).tolist()
                result = torch.tensor([add32(x,y) for x,y in zip(*words)], dtype=torch.int64).to(torch.int32).view(torch.float32).reshape(16,128)
            else:
                # Padding is a compiler action; the wide instruction always sees N256.
                padded_b = torch.zeros(256,256)
                padded_b[:,:n] = b
                result = op(a, padded_b)[:,:n]
            expected = torch.tensor(row['expected'], dtype=torch.int64).to(torch.int32)
            self.assertTrue(torch.equal(result.contiguous().view(torch.int32), expected), row['source'])

    def test_hardware_replay_with_external_schedule(self):
        cases = json.loads((BASE/'v6-schedule-measured.json').read_text())
        modes = set()
        for row in cases:
            mode = tpu_v6.mode_from_hlo((BASE/row['hlo']).read_text())
            modes.add(mode)
            # The instruction must work without shape profiles or HLO dispatch.
            with patch.dict(tpu_v6.PROFILES, {}, clear=True):
                with patch.object(tpu_v6, 'native_mode', side_effect=AssertionError('compiler dispatch')):
                    self.assertEqual(compose(row['a'], row['b'], mode), int(row['expected'], 16))
        self.assertEqual(modes, {'hybrid', 'cross128'})
        for row in json.loads((BASE/'v6-shape-measured.json').read_text()):
            self.assertEqual(compose(row['a'], row['b'], 'hybrid'), int(row['expected'], 16))

    def test_modes_are_numerically_distinct(self):
        row = [r for r in json.loads((BASE/'v6-shape-measured.json').read_text()) if r['k'] == 256][1]
        independent = compose(row['a'], row['b'], 'hybrid')
        cross = compose(row['a'], row['b'], 'cross128')
        self.assertNotEqual(independent, cross)
        self.assertEqual(independent, int(row['expected'], 16))

    def test_matrix_transport_and_independent_streams(self):
        for setup in ('vmatpush3', 'vmatpush1+vmatpush2'):
            op = instruction(setup)
            # Different row/column patterns and signs; small exact integer dot.
            a = ((torch.arange(op.m) % 3)[:, None] - 1).expand(op.m, op.k).float()
            b = ((torch.arange(op.n) % 4)[None, :] - 2).expand(op.k, op.n).float()
            if op.groups == 2:
                a = torch.stack((a, -2*a))
                b = torch.stack((b, 3*b))
            a = a.to(torch.bfloat16); b = b.to(torch.bfloat16)
            before_a, before_b = a.clone(), b.clone()
            result = op(a, b, torch.zeros(op.d_shape))
            self.assertEqual(tuple(result.shape), op.d_shape)
            self.assertTrue(torch.equal(result.view(torch.int32), (a.float() @ b.float()).view(torch.int32)))
            self.assertTrue(torch.equal(a, before_a)); self.assertTrue(torch.equal(b, before_b))

    def test_sampled_dot_and_instruction_variants(self):
        for name in (OP, OP+'.mxu0', 'vmatmul.mubr.bf16.vlgmr.msra.gmra.mxu1'):
            op = MMA('TPU-v6e', name, weight_load='vmatpush3')
            self.assertEqual(op.dpa(torch.ones(128), torch.ones(128), torch.tensor(0.)).item(), 128.)
            self.assertEqual(op.a_shape, (2,16,128))
            self.assertEqual(len(op.reduction_groups), 16)
        self.assertEqual(instruction('vmatpush1+vmatpush2').reduction_groups[0],
                         ((0,1,2,3),(128,129,130,131),(4,5,6,7),(132,133,134,135)))

    def test_reject_unsupported_or_ambiguous_calls(self):
        with self.assertRaises(TypeError):MMA('TPU-v6e', OP)
        with self.assertRaises(NotImplementedError):MMA('TPU-v7', OP, weight_load='vmatpush3')
        with self.assertRaises(NotImplementedError):MMA('TPU-v6e', OP+'.unknown', weight_load='vmatpush3')
        with self.assertRaises(ValueError):instruction('auto')
        op = instruction('vmatpush3')
        with self.assertRaises(ValueError):op.dot_bits([0]*256, [0]*256)
        with self.assertRaises(NotImplementedError):op.dot_bits([0]*128, [0]*128, 0x3f800000)
        with self.assertRaises(ValueError):op.dot_bits([0x3f800001]*128, [0]*128)
        with self.assertRaises((ValueError, NotImplementedError)):op.dot_bits([0x7fc00000]*128, [0]*128)
        a,b = torch.zeros(op.a_shape),torch.zeros(op.b_shape)
        with self.assertRaises(ValueError):op(a[0],b[0])
        with self.assertRaises(NotImplementedError):op(a,b,torch.ones(op.d_shape))
        with self.assertRaises(NotImplementedError):op(a.requires_grad_(),b)
