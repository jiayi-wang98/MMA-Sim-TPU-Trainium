"""Replay a collect_unique.py capture, using its own final-bundle programs."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmasim.tpu import LLOProgram, MMA


def replay(root):
    programs = {}
    for path in sorted((root / "llo").glob("*fusion-01-final_bundles.txt")):
        if "vmatmul." not in path.read_text():
            continue
        program = LLOProgram("TPU-v6e", path.read_text())
        key = (program.a_shape[0], program.a_shape[1], program.b_shape[1])
        if key in programs:
            raise ValueError(f"Multiple programs for shape {key}")
        programs[key] = (program, path.name)
    rows = []
    for path in sorted((root / "results").glob("*-integer.npz")):
        x = np.load(path)
        a = torch.from_numpy(x["a"])
        b = torch.from_numpy(x["b"])
        shape = (a.shape[0], a.shape[1], b.shape[1])
        program, name = programs[shape]
        # These bounded integer controls are exactly representable through every
        # stage. Check whole-matrix spatial dataflow separately from rounding.
        if not (
            np.isin(x["a"], np.arange(-4, 5)).all()
            and np.isin(x["b"], np.arange(-4, 5)).all()
        ):
            raise ValueError(
                "Layout control is not in the verified exact integer range"
            )
        with patch.object(MMA, "__call__", lambda self, a, b: a @ b):
            layout = program(a, b).contiguous().numpy().view(np.uint32)
        layout_bad = int(np.count_nonzero(layout != x["bits"]))
        rounding_path = path.with_name(
            path.name.replace("-integer.npz", "-rounding.npz")
        )
        y = np.load(rounding_path)
        cells = [tuple(map(int, p)) for p in y["cells"]]
        actual = program.sample_bits(
            torch.from_numpy(y["a"]), torch.from_numpy(y["b"]), cells
        )
        bad = sum(value != int(y["bits"][i, j]) for value, (i, j) in zip(actual, cells))
        row = {
            "shape": shape,
            "llo": name,
            "integer_entries": int(x["bits"].size),
            "integer_layout_mismatches": layout_bad,
            "rounding_samples": len(cells),
            "rounding_mismatches": bad,
            "execution": program.last_execution,
        }
        rows.append(row)
        print(row, flush=True)
    if len(rows) != 4:
        raise ValueError("Expected four complete shape captures")
    return {
        "jobs": rows,
        "integer_entries": sum(r["integer_entries"] for r in rows),
        "rounding_samples": sum(r["rounding_samples"] for r in rows),
        "mismatches": sum(
            r["integer_layout_mismatches"] + r["rounding_mismatches"] for r in rows
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = replay(args.capture)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["mismatches"]:
        raise SystemExit("Hardware replay mismatch")
