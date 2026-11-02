"""Losslessly compact the repeated-pattern full matrices from TPU captures."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("v6-llo-measured.json")
    )
    args = p.parse_args()
    out = {
        "source": "v6-schedule-boundaries; complete saved native outputs, lossless repeated-pattern encoding",
        "banks": {},
        "cases": [],
    }
    specs = [
        (128, 128, 128, "1793572668019869185-fusion-01-final_bundles.txt"),
        (128, 256, 128, "1793572668118415881-fusion-01-final_bundles.txt"),
        (257, 256, 128, "1793572668195508581-fusion-01-final_bundles.txt"),
        (128, 256, 256, "1793572668321836148-fusion-01-final_bundles.txt"),
    ]
    for k in (128, 256):
        with np.load(args.corpus / f"inputs-k{k}.npz") as x:
            out["banks"][str(k)] = {
                key: x[key].view(np.uint32).tolist() for key in ("a", "b")
            }
    for m, k, n, llo in specs:
        name = f"m{m}-k{k}-n{n}.npz"
        with np.load(args.corpus / name) as x:
            ri, ci, bits = x["row_index"], x["col_index"], x["bits"]
            grid = np.zeros((8, 8), dtype=np.uint32)
            for i in range(8):
                for j in range(8):
                    grid[i, j] = bits[
                        np.flatnonzero(ri == i)[0], np.flatnonzero(ci == j)[0]
                    ]
            assert bits.shape == (m, n) and np.array_equal(
                bits, grid[ri[:, None], ci[None, :]]
            )
            out["cases"].append(
                {
                    "source": name,
                    "llo": llo,
                    "shape": [m, k, n],
                    "row_index": ri.tolist(),
                    "col_index": ci.tolist(),
                    "expected_grid": grid.tolist(),
                    "verified_entries": int(bits.size),
                }
            )
    args.output.write_text(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
