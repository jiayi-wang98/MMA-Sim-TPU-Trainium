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
interface remains available. Automatic interpretation of arbitrary LLO programs
is a separate task; these APIs do not yet implement it.

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
