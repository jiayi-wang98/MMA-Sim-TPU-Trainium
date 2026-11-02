v6e BF16 instruction-group shapes
================================

Scope and conclusion
--------------------

We traced the four captured final-bundle programs offline. No TPU was started
for this analysis. A computation group here consists of the most recent
``vmatprep.mubr.bf16`` and one ``vmatmul.mubr.bf16`` on the same MXU, operating
on previously loaded matrix state. It is not an isolated, stateless opcode.

The operand and output footprints support these effective modes:

* ``vmatpush3`` setup: two independent 16x128 by 128x128 products. Each
  computation group produces two 16x128 FP32 result blocks.
* ``vmatpush1`` plus ``vmatpush2`` setup: one 16x256 by 256x256 product,
  producing a 16x256 FP32 block.

These are dataflow interpretations of the captured programs, not an exhaustive
ISA specification. In particular, they do not specify internal rounding or the
physical organization of the MXU. They also do not mean one group finishes in
one cycle. Results are observed through four 8x128 FP32 pops per ordinary group.

Address decoding
----------------

The HLO layout is BF16 ``T(8,128)(2,1)`` and FP32 ``T(8,128)``. The public
layout definition is https://openxla.org/xla/tiled_layout . A BF16 tile contains
8x128 values (2048 bytes); a FP32 tile contains 8x128 values (4096 bytes).
The dump's immediate strides are respectively 4 and 8. We use the consistent
512-byte immediate-address unit. ``vld sm:0xf`` loads the four packed sublanes
of a BF16 tile; ``vcombine.low`` combines two tile sources. The tracer retains
both source footprints without asserting their precise register lane order.

For BF16 with C columns, offset d maps to tile index d/4, then to row tile
floor((d/4)/ceil(C/128)) and column tile (d/4) modulo ceil(C/128).
FP32 stores use d/8 instead. Coordinates below are half-open Python slices.

Case 1: M=128, K=128, N=128
--------------------------

In the first MXU0 computation group:

* MUBR prep reads A[0:16, 0:128], from offsets 0x0 and 0x4.
* Matmul operand reads A[16:32, 0:128], from offsets 0x8 and 0xc.
* RHS prep and push3 load the SAME B[0:128, 0:128] data into both paths.
* Four pops write C rows 16:24, 0:8, 24:32, 8:16, respectively.

Thus the effective footprint is 32x128 by 128x128, assembled from two
independent 16-row products sharing B. MXU0 covers M rows 0:64 and MXU1
covers 64:128. They split M, not K, in this program.

Case 2: M=128, K=256, N=128
--------------------------

The first MXU0 group instead reads:

* MUBR prep: A[0:16, 128:256], offsets 0x4 and 0xc.
* Matmul operand: A[0:16, 0:128], offsets 0x0 and 0x8.
* RHS prep: B[128:256, 0:128]; push3: B[0:128, 0:128].

The two paths now cover distinct K halves for the SAME output rows.
Four pops are combined in pairs by explicit ``vadd.f32``, producing two
8x128 output tiles, i.e. a 16x128 block. This is consistent with:

::

  P0 = product(A[m:m+16, 0:128], B[0:128, :])
  P1 = product(A[m:m+16, 128:256], B[128:256, :])
  C[m:m+16, :] = vadd.f32(P0, P1)

``product`` denotes the hardware arithmetic, not an IEEE reference GEMM.
The explicit add operands are popped from the SAME MXU. Again MXU0 owns
rows 0:64 and MXU1 owns rows 64:128.

Case 3: M=128, K=256, N=256
--------------------------

The MUBR prep and multiply input footprints are the same as case 2.
RHS loading changes:

* push1 supplies B[0:128, 0:128]; its prep supplies B[0:128, 128:256].
* push2 supplies B[128:256, 0:128]; its prep supplies B[128:256, 128:256].

Each group consumes A[m:m+16, 0:256]. Four pops directly write the four
8x128 tiles of C[m:m+16, 0:256], with no vector add. This supports the
16x256 by 256x256 effective computation-group shape.

Case 4: M=257, K=256, N=128
--------------------------

RHS push1/push2 supplies B's two K halves, while every associated SUBR prep
supplies a literal zero vector. The ordinary BF16 groups consume 16x256
A blocks. Of 66 result pops, 33 are stored and 33 are discarded; there are
no vector additions. This is consistent with the 256-column mode padded
with 128 zero columns and retaining only the useful 128 output columns.
MXU0 stores rows 0:144; MXU1 stores rows 144:264 (the final tile is padded).
The tail ``vmatmul.mubr.f32`` reads the final BF16 row after a VMEM spill
and ``vunpack.c.l.bf16``. The logical interpreter now supports this one-row
case: it emits two 8x128 result pops and only the first live row reaches
the final tensor. The saved last-row outputs match the BF16 arithmetic model.
General FP32 operands are outside this binding.

Reproduction and checks
-----------------------

Run ``python tests/accelerators/llo_trace.py tests/accelerators/v6-llo --out /tmp/operand-trace.json`` from the repository root. The trace JSON records
source line numbers, original load offsets, tile coordinates, preparation
operands, spills, output stores, and unused pops. It does not execute LLO.

For the three M=128 cases, checks established:

* Every A tile enters exactly one computation group.
* RHS loading on each MXU covers every B tile.
* Every C tile is stored exactly once, without gaps.
* Every popped result is accounted for by a store or its vector-add input.

The next semantic step is decoding the lane order within the prepared matrix
state and relating it to the numerical DPA8 grouping. The shape footprints
alone do not resolve that internal arithmetic order.
