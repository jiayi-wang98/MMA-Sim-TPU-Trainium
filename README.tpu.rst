TPU instruction-group interface
===============================

``mmasim.tpu.MMA`` follows the GPU backend's architecture/instruction constructor
and callable tensor interface. It models a prepared BF16 computation group on
TPU-v6e, with explicit weight-loading state:

::

  import torch
  from mmasim.tpu import MMA

  wide = MMA("TPU-v6e", "vmatmul.mubr.bf16.gmra",
             weight_load="vmatpush1+vmatpush2")
  A = torch.ones((16, 256), dtype=torch.bfloat16)
  B = torch.ones((256, 256), dtype=torch.bfloat16)
  D = wide(A, B)  # FP32 [16,256], each entry is 256

  dual = MMA("TPU-v6e", "vmatmul.mubr.bf16.gmra",
             weight_load="vmatpush3")
  A2 = torch.ones((2, 16, 128), dtype=torch.bfloat16)
  B2 = torch.ones((2, 128, 128), dtype=torch.bfloat16)
  P = dual(A2, B2)  # FP32 [2,16,128], independent result streams

Use the actual captured mnemonic, not an invented PTX shape suffix. Both
``vmatmul.mubr.bf16.gmra`` and ``vmatmul.mubr.bf16.vlgmr.msra.gmra`` are accepted;
optional ``.mxu0`` / ``.mxu1`` suffixes preserve the targeted MXU identifier.
Their numerical group kernel is shared; timing and persistent machine state
are not simulated.

The required ``weight_load`` argument identifies the previously prepared RHS
state. One modeled operation includes the associated MUBR preparation and the
multiply. Operands are unpacked logical matrices, not raw VREG contents. Shape
attributes ``m``, ``n``, ``k``, ``groups``, ``a_shape``, ``b_shape``, and ``d_shape``
are available for callers. This initial API is a candidate for upstream review.

Separating instruction arithmetic from compiler scheduling
---------------------------------------------------------

``vmatpush3`` exposes two independent K128 products. The caller decides whether
they describe different M rows, or two K halves of the same output. In the
latter case, the captured compiler program combines the result streams using
an external FP32 add. The instruction model deliberately returns both streams.

``vmatpush1+vmatpush2`` exposes a K256/N256 group, using the measured cross-half
DP4 reduction pattern. A compiler can supply zero-padded columns and discard
the corresponding output columns. Padding, matrix orientation, row assignment
to MXUs, and outer K-tile addition are caller/scheduler operations.

Neither instruction mode queries the HLO dispatcher or shape-profile registry.
For full compiler-selected GEMMs, the existing ``mmasim.accelerators.Matmul``
interface remains available. For the captured straight-line LLO subset, ``LLOProgram`` now reconstructs
and executes this scheduling directly from the final bundles (see below).

Numerical details
-----------------

``reduction_groups`` exposes the numerical DP4 term mapping for one output:

* Dual mode: at update i, DP4(8i:8i+4) plus DP4(8i+4:8i+8), independently
  for each stream; the result updates the running FP32 sum.
* Wide mode: with j=8i, combine DP4(j:j+4) with DP4(j+128:j+132), combine
  DP4(j+4:j+8) with DP4(j+132:j+136), then add those two pair results and
  update the running FP32 sum.

The existing bit-level arithmetic core supplies truncation, rounding and FTZ
behavior. This numerical grouping does not specify physical PE placement.
``dot_bits(a,b,c=0)`` takes FP32 encodings of one output dot's BF16 operands;
for dual mode it evaluates one stream. ``dpa(a,b,c)`` provides the analogous
one-dimensional PyTorch call, following the GPU API convention. It represents
a complete output dot, not a standalone DPA8 machine instruction.

Inputs must be finite, exactly BF16-representable CPU tensors, with no autograd.
The output is FP32. C may be omitted or supplied as an all-zero FP32 tensor;
nonzero C is rejected because arbitrary internal accumulator initialization
has not been established. A caller may model a separately specified FP32 add.

Evidence and validation
-----------------------

The four real v6e final-bundle files in ``tests/accelerators/v6-llo/`` support
the shape/state binding (trailing whitespace removed for version control). ``llo_trace.py`` records logical input/output tile
provenance and explicit vector-add dependencies. Its assumptions and traced
examples are described in ``v6-llo/INSTRUCTION_SHAPES.rst``. The earlier Spot
capture lost its numerical arrays during transfer; the arithmetic checks use
separately preserved hardware outputs, not regenerated simulator targets.

``v6-instruction-measured.json`` retains input-bank encodings and native output
tiles from the saved boundary corpus for M/K/N = 128/128/128, 128/256/128,
128/256/256, and 257/256/128. Tests check 12,288 output entries through the new
matrix API, with explicit external merging or padding as appropriate. These
entries contain repeated diagnostic row/column patterns. Another 32 saved
scalar cases exercise compiler-level compositions, tails and longer K.
A rounding-sensitive case also confirms that the two modes give different bits.

Run from the repository root::

  python -m unittest discover -s tests/accelerators -v

This instruction binding currently covers v6e BF16. The existing v5e/v7 and
other accelerator arithmetic backends are available through their original API;
they are not assigned unmeasured v6e instruction semantics.


Executing captured LLO programs
------------------------------

::

  from pathlib import Path
  from mmasim.tpu import LLOProgram

  program = LLOProgram("TPU-v6e", Path("final_bundles.txt").read_text())
  D = program(A, B)
  bits = program.sample_bits(A, B, [(0, 0), (15, 127)])
  print(program.last_execution)

The constructor reads input/output shapes and scalar allocation aliases from
LLO. Execution follows BF16 VMEM tile loads, packing, spill stores/reloads,
SUBR/MUBR preparation, weight pushes, per-MXU result FIFOs, explicit FP32 vector
adds, output stores, and the final DMA. It does not query shape profiles or HLO.
The two independent push3 results are popped in multiply/preparation order,
interleaved by eight-row blocks. Wide results are popped by eight-row block,
then by 128-column half. Eight reversed 16-row pushes reconstruct a weight bank.

``sample_bits`` follows the same dataflow and checks all output stores while
lazily evaluating only the requested dots. It returns unsigned FP32 encodings,
preserves request order and duplicates, and rejects out-of-range coordinates.
This is useful for independent, non-repeating random input matrices without
paying the cost of a full bit-level matrix simulation.

The captured M257 tail loads the final BF16 row through VMEM, unpacks its low
BF16 halves, and uses a ``vmatmul.mubr.f32`` group. This narrow path is now
modeled using the BF16 numerical core and checked against the saved last-row
outputs. Undefined unpack lanes cannot be written to live output rows. General
FP32 matrix arithmetic is not enabled by this tail support.

Supported programs have rank-two BF16 inputs and FP32 outputs, whole-matrix
DMA, constant tile addresses, and one weight-loading phase per MXU. Unknown
opcodes, branches, weight reloads, partial transfers, unresolved memory loads,
incomplete outputs, and undrained result FIFOs raise errors. Execution models
logical arithmetic and dataflow, not cycle timing or raw machine-code encoding.

``test_tpu_llo_program.py`` replays all 98,432 entries in four saved hardware
matrices, including padding and the final M257 row. Their repeated-pattern
encoding is lossless: ``build_v6_llo_fixture.py`` verifies every original matrix
entry before writing the compact fixture. Tests also use independent integer
inputs to exercise row/column ordering separately from rounding, and verify
that sampled execution agrees with complete execution.

To collect independent layout/rounding inputs on a v6e with JAX 0.6.2/libtpu,
set the flags before Python starts (hardware provisioning is separate)::

  XLA_FLAGS='--xla_dump_to=/tmp/tpu-llo/hlo --xla_dump_hlo_as_text' \
  LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/tpu-llo/llo --xla_jf_dump_llo_text --xla_jf_dump_llo_pass_label_regex=final_bundles' \
    python tests/accelerators/collect_tpu_llo.py --out /tmp/tpu-llo/results

Archive and retrieve the whole ``/tmp/tpu-llo`` directory. Replay locally::

  python tests/accelerators/replay_tpu_llo.py /path/to/tpu-llo \
    --output /tmp/tpu-llo-validation.json

This replay uses each job's own LLO. It checks all spatial outputs with bounded
integer controls (an exact integer arithmetic kernel isolates data routing),
and separately checks selected independent BF16 rounding inputs with the
bit-level instruction model. These are distinct validation steps; integer
layout controls alone do not validate truncation or rounding behavior.
