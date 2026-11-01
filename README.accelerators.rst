TPU, Blackhole and Trainium3 reference backends
==============================================

This research extension adds mmasim.accelerators.Matmul to Microsoft MMA-Sim
2.1.1, based on upstream commit 44b3ac00fed7e9fca7a18b7099e32d8d78a907a5.
The original NVIDIA/AMD modules are unchanged. This is a research fork;
it has not been submitted to or accepted by Microsoft.

The backend wraps frozen bit-level arithmetic candidates from our numerical
reverse-engineering study. It uses exact integer or rational arithmetic on CPU,
returns FP32 PyTorch tensors, and exposes the same callable A,B,C pattern.
It is intended for small numerical comparisons, replay, and model development.
It is not a high-throughput LLM simulator and does not implement autograd.

Installation
-------------

From this source tree, Python >=3.10::

  python -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -e src/mmasim
  python -m unittest discover -s tests/accelerators -v

Only CPU dependencies are needed for this extension. Tests include 140 saved
hardware cases, matrix layout/bit transport, block scales, explicit C policy,
and rejection of unsupported shapes, formats, and gradients.

Usage
------

  import torch
  from mmasim.accelerators import Matmul

  A = torch.randn(2, 128).to(torch.bfloat16)
  B = torch.randn(128, 3).to(torch.bfloat16)
  C = torch.zeros(2, 3, dtype=torch.float32)
  tpu = Matmul("TPU-v6e", "bf16")
  trn = Matmul("Trainium3", "bf16", chunk=128)
  D_tpu = tpu(A, B, C)
  D_trn = trn(A, B, C)

Blackhole HiFi4 uses a captured K32 profile::

  bh = Matmul("Blackhole", "bf16")
  D_bh = bh(A[:, :32], B[:32, :], C)

The example above gives Blackhole a shorter mathematical dot. For a shared-dot
comparison, place the same 32 active terms in a zero-filled K128 TPU/Trainium
input, as in examples/accelerator_compare.py.

For explicit v6e instruction groups with compiler scheduling kept separate,
see ``mmasim.tpu.MMA`` in `README.tpu.rst <README.tpu.rst>`_.

Available profiles
-------------------

* TPU-v5e: BF16 DEFAULT MXU paths, including non-multiple-of-128 K tails.
  DP4 partial-tail arithmetic, DPA8 updates and sequential K128 composition
  match 61,011 recorded comparisons across 720 configuration records, with
  K through 4097. The local arithmetic is unchanged. Non-MXU degeneracies
  are rejected; optional compiled_hlo validates the native MXU and its shape.
* TPU-v6e: BF16 DEFAULT MXU paths, including irregular K tails. Two K256
  schedules share the frozen DP4 arithmetic: independent K128 halves, or
  cross-half DP4 pairing at every update. Compiler windows select the path;
  complete K256 results accumulate sequentially. The revised model matches
  61,011 recorded dot comparisons. VPU reductions are outside this model.
* TPU-v7: BF16 native N128 profile, measured K values from 128 through 2048;
  plus four captured cross-half M/N profiles at K1024/2048 (see below);
  E-27 DP4 alignment, paired tile sums followed by sequential accumulation.
  N other than 128 raises NotImplementedError.
* TPU-v7 low formats: e4m3fn, e5m2, e2m1fn; K128/256/512, N128,
  Precision.DEFAULT. backend="jax" (default) or "pallas" selects the path.
  FP8 and Pallas FP4 use E-8 alignment, nine-bit RTZ partials and nine-bit
  RNE accumulation; JAX FP4 uses the BF16 numerical candidate. Independent
  128-term tiles combine in a balanced FP32 tree over this measured K range.
* Blackhole: BF16, HiFi4 four-phase arithmetic, K32.
* Trainium3 ordinary: bf16, fp16, fp32, e4m3, e4m3fn, e5m2;
  K128/256/512, instruction chunk 32/64/128.
* Trainium3 double_row / two_inst: e4m3, e4m3fn, e5m2; K256;
  mapping="half" or "interleave".
* Trainium3 mx: e4m3fn, e5m2, e2m1fn; logical K512;
  parts=32/64/128 partitions per instruction.

Matrix calls compose the scalar candidate into a tensor operation; newly
selected M/N shapes do not constitute new native-hardware validation.
TPU v7 uses captured shape profiles. Format/chunk combinations
also represent model composition; the original hardware jobs cover only the
combinations recorded in the study. NaN operands are rejected. NaN payloads
and a general signed-zero model are outside this finite-input interface.

Input values must already be quantized to the selected format. FP32 tensors
containing exact BF16/FP8/FP4 values are accepted; out-of-format values raise an
error rather than being silently rounded. Inputs are not modified.

TPU v7 FP8 / FP4
------------------

  fp8 = Matmul("TPU-v7", "e4m3fn", backend="jax")
  fp4_jax = Matmul("TPU-v7", "e2m1fn", backend="jax")
  fp4_pallas = Matmul("TPU-v7", "e2m1fn", backend="pallas")

Pass already-quantized, unscaled operand values. Native PyTorch FP8 tensors or
FP32 tensors containing exact FP8 values are accepted. For FP4, pass decoded
E2M1 values in an FP32 tensor; packed nibbles are not accepted. Both matrix
inputs must produce N=128 outputs, with K=128, 256, or 512. These are DEFAULT
precision profiles; no HIGH/HIGHEST selection is exposed.

FP4 backend selection changes the numerical model: JAX matches the BF16
candidate, while Pallas uses nine-bit significands. The adapter preserves this
measured distinction without asserting that JAX physically converts FP4 to
BF16. This unscaled FP4 profile is separate from Trainium3 MXFP4. TPU block
scales are rejected because the captured stack did not provide a scaled path.

Replay the low-format corpus independently::

  python tests/accelerators/replay_tpu_v7_low.py /path/to/research-data \
      --output tpu-v7-low-replay.json

Saved-output fixtures and tests cover both frontends, all three formats, and
all three K values. The complete low-format replay matches 44,416 recorded
comparisons with zero mismatches (including smoke/full and frontend repeats).
tpu-v7-low-replay.json records the per-job counts and input-file hashes.

Accumulator semantics
----------------------

By default, C must be zero or omitted. The observed model describes the complete
specified dot/reduction path, rather than an arbitrary accumulator initializer.
To explicitly compute a final FP32 addition use c_mode="late_fp32". This selects
an external addition after the modeled dot; it does not initialize an internal
DPA chain. TPU uses the existing FTZ adder policy, and the other profiles use
gradual underflow for this explicit final operation. Internal instruction
composition stays in the respective model.

MX inputs
----------

MX accepts decoded, quantized mantissas and unbiased E8M0 exponent tensors::

  mx = Matmul("Trainium3", "e2m1fn", mode="mx", parts=128)
  A = torch.ones(1, 512)             # exact E2M1 values, not packed bytes
  B = torch.ones(512, 1)
  sa = torch.zeros(1, 16, dtype=torch.int32)
  sb = torch.zeros(16, 1, dtype=torch.int32)
  D = mx(A, B, scales_a=sa, scales_b=sb)

A[M,K] uses scales_a[M,K/32]; B[K,N] uses scales_b[K/32,N]. Entries are integer
exponents in [-127,127], with scale=2**exponent. Scale bytes, if available, must
be decoded by subtracting 127; reserved byte 255 is unsupported. The adapter
never materializes scaled low-format inputs in FP32, preserving wide internal
exponents. Four logical terms form a packed partition. Each modeled instruction
contains parts partitions, and FP32 RNE combines its outputs.

Scalar bit interface
---------------------

dot_bits(a,b,c=0,m=128,n=128,...) accepts lists of unsigned FP32 encodings of
already-quantized operand values, including BF16 values expanded to FP32.
It returns an unsigned FP32 output encoding. The M/N arguments identify the
full output shape for a sampled dot. In particular, a TPU v7 sampled column
must still pass n=128. MX accepts one scale-exponent list per operand.

Validation and provenance
--------------------------

The source hashes and upstream base are recorded in
src/mmasim/mmasim/accelerators/PROVENANCE.json. Vendored numerical references
have package-relative imports; their arithmetic is unchanged. The bundled
measured.json subset supplies hardware outputs, not simulator-generated targets.

A full-corpus replay can be run with the separately preserved research data::

  python tests/accelerators/replay_corpus.py /path/to/research-data \
      --output accelerator-replay.json

It checks all 98,200 Trainium3 recorded cases, Blackhole's fresh HiFi4 corpus,
and the native N128 TPU v7 round-3 jobs. The adapter replay matches all 106,444 recorded comparisons: 98,200 Trainium3,
4,296 Blackhole, and 3,948 TPU v7, with zero bit mismatches.
The per-job output summary is included in tests/accelerators/replay-summary.json. TPU v5e/v6e adapter regression uses saved dense-output samples;
the original complete model validations are described in the study.

Next implementation step
-------------------------

The scalar references provide correctness oracles for vectorized tensor or C++
implementations. A fast backend should preserve each stage's rounding and the
selected reduction schedule, then pass the same recorded-output regressions.
A training wrapper would additionally need an explicitly defined backward
arithmetic policy; this extension deliberately raises on requires_grad inputs.

V5e expanded MXU coverage
-------------------------

V5e retains one arithmetic schedule across the captured shapes and seven
compiler emitters, including transposed execution. Each K128 block contains
16 DPA8 updates; each DPA8 combines two rounded DP4 outputs before the FP32
accumulator update. Independent K128 results then accumulate sequentially,
and the final partial block is numerically equivalent to zero padding.

The expanded capture matches 61,011 dot comparisons, including 3,807 checked
outputs from 64 fully random matrix configurations. Its 4,224 determinism
executions show no observed output changes. Counts include operands reused
across shapes. Fifty VPU-only configurations are excluded.

The API accepts irregular K on MXU paths without a shape-to-schedule table.
Unlike v6e, HLO is not needed to select between arithmetic paths. Supplying
``compiled_hlo=...`` optionally checks that the program is a native BF16/FP32
MXU call and that its M/K/N agree with the supplied operands. The captured
stack is JAX/jaxlib 0.6.2 and libtpu 0.0.17; measurements reach M4096,N513,K4097
without exhaustively testing every combination in that range. See
``tests/accelerators/v5-scheduling-validation.json`` and the saved regression
examples. No arithmetic constants or rounding rules were refitted.

V6e native MXU scheduling
-------------------------

The v6e backend now matches all 4,176 cases from the original shape sweep,
including the previously unresolved large-M and N!=128 paths. New shape,
position, spatial, K-tail, and full-random captures bring the total to 61,011
recorded dot comparisons, with no mismatches. This count includes shared
operands replayed across shapes. The fully random matrix holdout contributes
3,807 checked outputs across 64 configurations. Another 4,224 repeated
executions, including queued and interleaved calls, recompilation, and a fresh
Python process, produce no observed bit changes.

Compiler lowering is part of the arithmetic profile. The adapter includes
686 captured (M,K,N) profiles for JAX/jaxlib 0.6.2 and libtpu 0.0.17. For a new
shape, supply the native compiled HLO, which encodes the selected matrix
orientation and window bounds::

  from pathlib import Path
  sim = Matmul("TPU-v6e", "bf16",
               compiled_hlo=Path("native-dot.hlo").read_text())
  D = sim(A, B)

Capture this text from the original DEFAULT BF16->FP32 dot with
``jax.jit(dot).lower(A, B).compile().as_text()`` on the target stack.
The HLO must describe one native dot, with M/K/N matching the operands.
Device loops, explicit split-dot programs, and VPU-only reductions are
rejected. Without HLO, uncaptured K>128 profiles are rejected rather than
guessing compiler orientation. K<=128 MXU arithmetic is the same in both
schedules. Nonfinite operands remain outside the public finite-input API.

The dispatcher reads the existing compiler windows; it does not emulate
XLA's shape-dependent cost model. The rules generalize the captured arithmetic
over K256 blocks and zero-padded tails, with measurements through K4097.
``tests/accelerators/v6-scheduling-validation.json`` records the evidence.
Earlier v6-shape-validation.json and v6-large-m-validation.json describe the
superseded single-path model and are retained as historical snapshots.

TPU v7 cross-half BF16 reconstruction
------------------------------------

JAX DEFAULT BF16 additionally supports (M,N)=(64,64),(128,256),(256,256),
(127,129), each at K1024 or K2048. Eight products across K128 halves share
an E-27 alignment grid before one FP32 rounding. Two such results per update
and sixteen updates form a K256 result. The K256 results split into two
contiguous equal-length sequential chains, followed by a final FP32 addition.

All 1,472 previously unmodeled comparisons match; the complete round-three
model replay matches 6,524 comparisons. K2048 dense data selected the outer
tree, so this is reconstruction/replay evidence. Other cross-half shapes and
K tails remain unsupported. Low-format and Pallas profiles are unchanged.

Reproduce the public-API replay::

  python tests/accelerators/replay_v7_cross.py /path/to/research-data \
      --output v7-cross-adapter-replay.json
