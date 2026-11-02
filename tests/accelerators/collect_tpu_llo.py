"""Capture non-repeating layout controls plus BF16 rounding inputs with LLO."""

import argparse, json, time, importlib.metadata, os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_enable_compilation_cache", False)
assert all(d.platform == "tpu" for d in jax.devices())
(args.out / "environment.json").write_text(
    json.dumps(
        {
            "versions": {
                n: importlib.metadata.version(n)
                for n in ["jax", "jaxlib", "libtpu", "numpy"]
            },
            "devices": [
                {"kind": d.device_kind, "platform": d.platform} for d in jax.devices()
            ],
            "flags": {k: os.environ.get(k) for k in ["XLA_FLAGS", "LIBTPU_INIT_ARGS"]},
        },
        indent=2,
    )
)
for m, k, n in [(128, 128, 128), (128, 256, 128), (257, 256, 128), (128, 256, 256)]:
    tag = f"m{m}-k{k}-n{n}"

    def dot(a, b):
        return jax.lax.dot_general(
            a,
            b,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

    dot.__name__ = "unique_" + tag.replace("-", "_")
    rng = np.random.default_rng(711000 + m + k + n)
    a = rng.integers(-4, 5, (m, k)).astype(np.float32)
    b = rng.integers(-4, 5, (k, n)).astype(np.float32)
    ad = jnp.asarray(a, jnp.bfloat16)
    bd = jnp.asarray(b, jnp.bfloat16)
    fn = jax.jit(dot).lower(ad, bd).compile()
    (args.out / (tag + ".hlo")).write_text(fn.as_text())
    y = np.asarray(fn(ad, bd).block_until_ready())
    repeat = np.asarray(fn(ad, bd).block_until_ready())
    assert np.array_equal(y, a @ b) and np.array_equal(
        y.view(np.uint32), repeat.view(np.uint32)
    )
    np.savez_compressed(
        args.out / (tag + "-integer.npz"), a=a, b=b, bits=y.view(np.uint32)
    )

    def random_bf16(shape):
        return (
            rng.choice([-1, 1], shape)
            * (1 + rng.integers(0, 128, shape) / 128)
            * np.exp2(rng.integers(-16, 17, shape))
        ).astype(np.float32)

    a = random_bf16((m, k))
    b = random_bf16((k, n))
    ad = jnp.asarray(a, jnp.bfloat16)
    bd = jnp.asarray(b, jnp.bfloat16)
    assert np.array_equal(np.asarray(ad).astype(np.float32), a) and np.array_equal(
        np.asarray(bd).astype(np.float32), b
    )
    y = np.asarray(fn(ad, bd).block_until_ready())
    repeat = np.asarray(fn(ad, bd).block_until_ready())
    assert np.array_equal(y.view(np.uint32), repeat.view(np.uint32))
    coords = {
        (i, j)
        for i in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, m - 2, m - 1]
        for j in [0, 1, 7, 8, 63, 64, 127, n - 1]
        if i < m and j < n
    }
    coords.update(zip(rng.integers(0, m, 64).tolist(), rng.integers(0, n, 64).tolist()))
    np.savez_compressed(
        args.out / (tag + "-rounding.npz"),
        a=a,
        b=b,
        bits=y.view(np.uint32),
        cells=np.array(sorted(coords)),
    )
    print(
        tag,
        "integer exact; both inputs repeat bit-exact;",
        len(coords),
        "sample cells",
        flush=True,
    )
