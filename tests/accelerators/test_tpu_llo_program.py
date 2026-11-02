"""End-to-end LLO dataflow replay, including output queues and BF16 tails."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from mmasim.tpu import LLOProgram, MMA
from mmasim.accelerators.reference import tpu_v6

BASE = Path(__file__).parent
FIXTURE = json.loads((BASE / "v6-llo-measured.json").read_text())


def operands(case):
    bank = FIXTURE["banks"][str(case["shape"][1])]
    a = (
        torch.tensor(bank["a"], dtype=torch.int64)
        .to(torch.int32)
        .view(torch.float32)[case["row_index"]]
    )
    b = (
        torch.tensor(bank["b"], dtype=torch.int64)
        .to(torch.int32)
        .view(torch.float32)[case["col_index"]]
        .T
    )
    grid = torch.tensor(case["expected_grid"], dtype=torch.int64).to(torch.int32)
    expected = grid[
        torch.tensor(case["row_index"])[:, None],
        torch.tensor(case["col_index"])[None, :],
    ]
    return a, b, expected


def program(case):
    return LLOProgram("TPU-v6e", (BASE / "v6-llo" / case["llo"]).read_text())


class LLOProgramTests(unittest.TestCase):
    def test_complete_saved_hardware_matrices_without_dispatch(self):
        count = 0
        with patch.dict(tpu_v6.PROFILES, {}, clear=True), patch.object(
            tpu_v6, "native_mode", side_effect=AssertionError("shape dispatch")
        ):
            for case in FIXTURE["cases"]:
                op = program(case)
                a, b, expected = operands(case)
                before_a, before_b = a.clone(), b.clone()
                self.assertTrue(
                    torch.equal(op(a, b).view(torch.int32), expected), case["shape"]
                )
                self.assertTrue(torch.equal(a, before_a))
                self.assertTrue(torch.equal(b, before_b))
                self.assertEqual(
                    op.last_execution.get("vector_adds", 0),
                    16 if case["shape"] == [128, 256, 128] else 0,
                )
                self.assertEqual(
                    op.last_execution["bf16_unpack_tail"], int(case["shape"][0] == 257)
                )
                count += expected.numel()
        self.assertEqual(count, 98432)

    def test_sampled_execution_preserves_coordinates_and_tail(self):
        for case in FIXTURE["cases"]:
            op = program(case)
            a, b, expected = operands(case)
            m, _, n = case["shape"]
            cells = [
                (m - 1, n - 1),
                (0, 0),
                (7, 7),
                (8, 8),
                (15, 63),
                (16, 64),
                (63, 127),
                (64, 0),
                (0, 0),
            ]
            actual = op.sample_bits(a, b, cells)
            self.assertEqual(
                actual, [int(expected[i, j]) & 0xFFFFFFFF for i, j in cells]
            )
            self.assertLessEqual(op.last_execution["evaluated_dots"], 2 * len(cells))
            self.assertEqual(op.sample_bits(a, b, []), [])

    def test_nonrepeating_integer_layout_control(self):
        rng = torch.Generator().manual_seed(711002)
        for case in FIXTURE["cases"]:
            op = program(case)
            a = torch.randint(-4, 5, op.a_shape, generator=rng).float()
            b = torch.randint(-4, 5, op.b_shape, generator=rng).float()
            expected = (a @ b).view(torch.int32)
            # Small integer products/sums are exact. Substitute the group kernel
            # to isolate every load, packing operation, pop and output address.
            with patch.object(MMA, "__call__", lambda self, x, y: x @ y):
                self.assertTrue(torch.equal(op(a, b).view(torch.int32), expected))
            cells = [
                (i, j)
                for i in (0, 7, 8, 15, 16, 31, 32, 63, 64, op.d_shape[0] - 1)
                for j in (0, 7, 8, 63, 64, op.d_shape[1] - 1)
            ]
            self.assertEqual(
                op.sample_bits(a, b, cells),
                [int(expected[i, j]) & 0xFFFFFFFF for i, j in cells],
            )

    def test_reject_unknown_ops_control_flow_and_dma(self):
        text = (BASE / "v6-llo" / FIXTURE["cases"][0]["llo"]).read_text()
        for bad in (
            text.replace("vcombine.low", "vcombine.high", 1),
            text.replace("0x1   :", "0x1 LH:", 1),
            text.replace("size_in_granules=*/1024,", "size_in_granules=*/1023,"),
            text.replace("vmatpush3", "vmatpush4", 1),
        ):
            with self.assertRaises(NotImplementedError):
                LLOProgram("TPU-v6e", bad)
        with self.assertRaises(NotImplementedError):
            LLOProgram("TPU-v7", text)
        with self.assertRaises(ValueError):
            LLOProgram("TPU-v6e", "")

    def test_reject_runtime_mismatch_and_unresolved_values(self):
        case = FIXTURE["cases"][0]
        op = program(case)
        a, b, _ = operands(case)
        with self.assertRaises(ValueError):
            op(a[:1], b)
        with self.assertRaises(ValueError):
            op.sample_bits(a, b, [(128, 0)])
        with self.assertRaises(ValueError):
            op.sample_bits(a, b, [(0.0, 0)])
        text = (BASE / "v6-llo" / case["llo"]).read_text()
        # Uninitialized data must not silently become an all-zero tile.
        bad = text.replace("[#allocation1 + $0x38]", "[#missing + $0x38]", 1)
        with self.assertRaises(NotImplementedError):
            LLOProgram("TPU-v6e", bad)(a, b)
        tail = FIXTURE["cases"][2]
        text = (BASE / "v6-llo" / tail["llo"]).read_text()
        bad = text.replace("vunpack.c.l.bf16", "vcombine.low", 1)
        a, b, _ = operands(tail)
        with self.assertRaises(NotImplementedError):
            LLOProgram("TPU-v6e", bad)(a, b)
