"""Execute the captured straight-line v6e BF16 final-bundle subset.

This interprets logical tile transfers, MXU group queues and vector adds. It is
not a timing simulator or a general TPU ISA emulator. Unsupported operations
and layouts fail closed rather than falling back to a shape-based GEMM model.
"""

from collections import Counter, deque
from dataclasses import dataclass
from numbers import Integral
import re
from typing import Callable

import torch

from .sim import MMA
from ..accelerators.reference.tpu import add32


@dataclass(frozen=True)
class _Instruction:
    result: str
    opcode: str
    args: str


@dataclass(frozen=True)
class _Vector:
    data: torch.Tensor | None
    kind: str  # bf16, f32, zero, or bf16_tail
    valid_rows: tuple = ()
    origin: tuple | None = None
    bits: Callable[[int, int], int] | None = None


def _require(condition, message):
    if not condition:
        raise NotImplementedError(message)


def _without_comments(text):
    # The dump nests annotations in BoundsCheck comments. Preserve newlines.
    depth, pos, out = 0, 0, []
    for token in re.finditer(r"/\*|\*/", text):
        chunk = text[pos : token.start()]
        out.append(chunk if depth == 0 else "\n" * chunk.count("\n"))
        depth += 1 if token[0] == "/*" else -1
        _require(depth >= 0, "Unbalanced LLO comments")
        pos = token.end()
    _require(depth == 0, "Unclosed LLO comment")
    out.append(text[pos:])
    return "".join(out)


_REG = r"%\w+"
_ADDR = re.compile(r"\[vmem:\[([^\] +]+)(?: \+ \$(0x[\da-f]+))?\] sm:\$(0x[\da-f]+)\]")
_CONTROL = {
    "vsyncpa",
    "vsyncadd",
    "smov",
    "sshll.u32",
    "int_to_ptr.vmem",
    "scalar_lea.vmem",
    "scmp.lt.s32.totalorder",
    "scmp.ne.s32.totalorder",
    "por",
    "pnand",
    "shalt.err",
    "dma.done.wait",
}
_MXU = {
    f"{op}.mxu{unit}"
    for unit in (0, 1)
    for op in (
        "vmatprep.subr",
        "vmatprep.mubr.bf16",
        "vmatprep.mubr.f32",
        "vmatpush1.bf16.msra",
        "vmatpush2.bf16.msra",
        "vmatpush3.bf16.msra",
        "vmatmul.mubr.bf16.gmra",
        "vmatmul.mubr.bf16.vlgmr.msra.gmra",
        "vmatmul.mubr.f32.gmra",
        "vpop.f32.mrf",
    )
}
_DATA = {
    "vld",
    "vst",
    "vmov",
    "vcombine.low",
    "vunpack.c.l.bf16",
    "vadd.f32",
    "dma.hbm_to_vmem",
    "dma.vmem_to_hbm",
}


class LLOProgram:
    """LLOProgram('TPU-v6e', final_bundles_text)(A, B) -> full FP32 matrix.

    Shapes and weight-loading modes are obtained from the instructions. No HLO
    or shape-profile lookup is used. Accepts the four captured straight-line
    BF16 programs, including the BF16-unpacked single-row tail in M257.
    """

    def __init__(self, arch, text):
        if arch != "TPU-v6e":
            raise NotImplementedError("LLO execution currently covers TPU-v6e")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide nonempty final-bundle text")
        self.arch = arch
        source = _without_comments(text)
        # Operand headers contain shape index: {}. Parse the entry separately,
        # then remove those braces so the ordinary bundle grammar is unambiguous.
        source = source.replace("shape index: {}", "shape index: none")
        pattern = re.compile(r"^\s*(0|0x[\da-f]+)\s*:\s*\{(.*?)\}", re.M | re.S)
        matches = list(pattern.finditer(source))
        _require(bool(matches), "No straight-line final bundles found")
        _require(int(matches[0][1], 0) == 0, "Expected entry bundle at address zero")
        _require(
            not pattern.sub("", source[matches[0].start() :]).strip(),
            "Unsupported labels or control flow between bundles",
        )
        addresses = [int(m[1], 0) for m in matches]
        _require(
            addresses == list(range(len(addresses))),
            "Expected consecutive straight-line bundles",
        )
        bundles = [m[2] for m in matches]
        self.instructions = []
        self.inputs = {}
        self.aliases = {}
        self.input_memory = {}
        self.output_base = None
        self.output_register = None
        output_shape = None
        for bundle in bundles:
            for slot in bundle.split(";;"):
                match = re.fullmatch(r"\s*(%\w+) = ([\w.]+)\s*(.*?)\s*", slot, re.S)
                _require(match is not None, "Unsupported final-bundle slot")
                dst, op, args = match.groups()
                if op.startswith("inlined_call_operand."):
                    header = re.fullmatch(
                        r"\[shape: (bf16|f32)\[(\d+),(\d+)\], index: (\d+), kind: (input|output), shape index: none\]",
                        args,
                    )
                    _require(
                        header is not None,
                        "Only rank-two BF16/FP32 operand headers are supported",
                    )
                    fmt, rows, cols, idx, kind = header.groups()
                    shape = (int(rows), int(cols))
                    if kind == "input":
                        _require(
                            fmt == "bf16"
                            and idx in ("0", "1")
                            and int(idx) not in self.inputs,
                            "Expected two distinct BF16 inputs",
                        )
                        self.inputs[int(idx)] = (dst, shape)
                        if op == "inlined_call_operand.vmem":
                            self.input_memory[dst] = int(idx)
                        else:
                            _require(
                                op == "inlined_call_operand.hbm",
                                "Unsupported operand memory space",
                            )
                    else:
                        _require(
                            fmt == "f32"
                            and idx == "2"
                            and self.output_register is None
                            and op == "inlined_call_operand.hbm",
                            "Expected one FP32 HBM output",
                        )
                        self.output_register, output_shape = dst, shape
                    continue
                _require(op in _CONTROL | _DATA | _MXU, f"Unsupported LLO opcode: {op}")
                self.instructions.append(_Instruction(dst, op, args))
                refs = re.findall(_REG, args)
                if op == "smov":
                    allocation = re.fullmatch(r"\[(#\w+)\]", args)
                    _require(
                        allocation is not None,
                        "Only allocation-base scalar moves supported",
                    )
                    self.aliases[dst] = allocation[1]
                elif op in ("sshll.u32", "int_to_ptr.vmem"):
                    if op == "sshll.u32":
                        _require(
                            re.fullmatch(r"%\w+, 4", args) is not None,
                            "Unsupported VMEM pointer scale",
                        )
                    _require(
                        len(refs) == 1 and refs[0] in self.aliases,
                        "Unresolved VMEM pointer",
                    )
                    self.aliases[dst] = self.aliases[refs[0]]
        _require(
            set(self.inputs) == {0, 1} and output_shape is not None,
            "Incomplete operand headers",
        )
        self.a_shape, self.b_shape = self.inputs[0][1], self.inputs[1][1]
        self.d_shape = output_shape
        _require(
            min(*self.a_shape, *self.b_shape, *self.d_shape) > 0,
            "Positive matrix dimensions required",
        )
        _require(
            self.a_shape[1] == self.b_shape[0]
            and output_shape == (self.a_shape[0], self.b_shape[1]),
            "Inconsistent GEMM shapes",
        )
        output_transfers = 0
        for inst in self.instructions:
            refs = re.findall(_REG, inst.args)
            if inst.opcode in ("dma.hbm_to_vmem", "dma.vmem_to_hbm"):
                dma = re.fullmatch(
                    r"\[thread:\$0\]\s*(%\w+),\s*(\d+),\s*(%\w+),\s*\[#\w+\]", inst.args
                )
                _require(
                    dma is not None,
                    "Only constant whole-matrix DMA transfers supported",
                )
            if inst.opcode == "dma.hbm_to_vmem":
                _require(
                    len(refs) == 2 and refs[1] in self.aliases, "Unsupported input DMA"
                )
                idx = next(
                    (i for i, (reg, _) in self.inputs.items() if reg == refs[0]), None
                )
                _require(idx is not None, "Unknown input DMA source")
                rows, cols = self.inputs[idx][1]
                _require(
                    int(dma[2])
                    == ((rows + 7) // 8 * 8) * ((cols + 127) // 128 * 128) * 2 // 32,
                    "Input DMA does not cover the whole padded BF16 tensor",
                )
                base = self.aliases[refs[1]]
                _require(
                    base not in self.input_memory,
                    "Repeated input DMA requires a streaming interpreter",
                )
                self.input_memory[base] = idx
            elif inst.opcode == "dma.vmem_to_hbm":
                _require(
                    len(refs) == 2
                    and refs[0] in self.aliases
                    and refs[1] == self.output_register,
                    "Unsupported output DMA",
                )
                self.output_base = self.aliases[refs[0]]
                rows, cols = self.d_shape
                _require(
                    int(dma[2])
                    == ((rows + 7) // 8 * 8) * ((cols + 127) // 128 * 128) * 4 // 32,
                    "Output DMA does not cover the whole padded FP32 tensor",
                )
                output_transfers += 1
        _require(
            output_transfers == 1 and set(self.input_memory.values()) == {0, 1},
            "Incomplete matrix transfers",
        )
        _require(
            self.output_base not in self.input_memory,
            "In-place matrix output is unsupported",
        )
        self.instruction_counts = dict(Counter(i.opcode for i in self.instructions))
        self.last_execution = None

    def _address(self, args):
        match = _ADDR.match(args)
        _require(match is not None, "Unsupported VMEM addressing")
        base, offset, mask = match.groups()
        return self.aliases.get(base, base), int(offset or "0", 0), int(mask, 0)

    def __call__(self, A, B):
        return self._execute(A, B, None)

    def sample_bits(self, A, B, coordinates):
        """Execute the same dataflow, evaluating only selected output cells.

        Returns unsigned FP32 encodings in the requested order, preserving
        duplicates. All loads, queues, stores and output coverage are checked.
        """
        cells = []
        for cell in coordinates:
            if len(cell) != 2 or any(
                isinstance(x, bool) or not isinstance(x, Integral) for x in cell
            ):
                raise ValueError("Coordinates must be integer (row,column) pairs")
            i, j = map(int, cell)
            if not (0 <= i < self.d_shape[0] and 0 <= j < self.d_shape[1]):
                raise ValueError("Sample coordinates outside output matrix")
            cells.append((i, j))
        return self._execute(A, B, cells)

    def _execute(self, A, B, cells):
        self.last_execution = None
        # Validate once before any operation; all intermediate BF16 data are exact.
        MMA._words(A, self.a_shape, "bf16")
        MMA._words(B, self.b_shape, "bf16")
        inputs = [A.float(), B.float()]
        regs, memory = {}, {}
        rhs_prep, lhs_prep = {}, {}
        weights = {unit: {} for unit in ("mxu0", "mxu1")}
        queue = {unit: deque() for unit in weights}
        output_writes = set()
        ready_inputs = {base for base in self.input_memory if base.startswith("%")}
        captured_output = None
        counts = Counter()
        dot_cache = {}

        def register(name):
            _require(name in regs, f"Undefined vector operand: {name}")
            return regs[name]

        def matrix(vector, rows):
            if vector.kind == "zero":
                return torch.zeros(rows, 128)
            _require(
                tuple(vector.data.shape) == (rows, 128),
                "Unexpected register tile footprint",
            )
            return vector.data

        def rhs_matrix(unit, setup):
            state = weights[unit]
            expected = (
                ("vmatpush3",) if setup == "vmatpush3" else ("vmatpush1", "vmatpush2")
            )
            _require(
                set(state) == set(expected), "Mixed or incomplete weight-loading state"
            )
            for key in expected:
                _require(
                    len(state[key]) == 8,
                    "Expected eight reversed 16-row weight pushes per bank",
                )

            def bank(key, path):
                return torch.cat(
                    [matrix(pair[path], 16) for pair in reversed(state[key])]
                )

            if setup == "vmatpush3":
                # Result stream 0 comes from the multiply operand, stream 1 from prep.
                return torch.stack((bank("vmatpush3", 1), bank("vmatpush3", 0)))
            return torch.cat(
                [torch.cat((bank(key, 1), bank(key, 0)), dim=1) for key in expected]
            )

        def lazy_group(a, b, model):
            def words(x):
                return (
                    x.contiguous()
                    .view(torch.int32)
                    .to(torch.int64)
                    .bitwise_and(0xFFFFFFFF)
                    .tolist()
                )

            aw, bw = words(a), words(b)

            def evaluate(part, i, j):
                row = aw[part][i] if model.groups == 2 else aw[i]
                col = (
                    [bw[part][t][j] for t in range(model.k)]
                    if model.groups == 2
                    else [bw[t][part * 128 + j] for t in range(model.k)]
                )
                key = (model.weight_load, tuple(row), tuple(col))
                if key not in dot_cache:
                    dot_cache[key] = model.dot_bits(row, col)
                return dot_cache[key]

            return evaluate

        for inst in self.instructions:
            dst, op, args = inst.result, inst.opcode, inst.args
            refs = re.findall(_REG, args)
            if op in _CONTROL:
                continue
            if op == "dma.hbm_to_vmem":
                ready_inputs.add(self.aliases[refs[1]])
                continue
            if op == "vld":
                _require(
                    _ADDR.fullmatch(args) is not None,
                    "Unsupported vector load operands",
                )
                base, offset, mask = self._address(args)
                if (base, offset) in memory:
                    vector, store_mask = memory[base, offset]
                    _require(
                        mask == store_mask,
                        "Partial/repacked spill loads are unsupported",
                    )
                    regs[dst] = vector
                else:
                    _require(
                        base in ready_inputs and mask == 0xF and offset % 4 == 0,
                        "Uninitialized, masked or unaligned VMEM load",
                    )
                    idx = self.input_memory[base]
                    x = inputs[idx]
                    r, c = divmod(offset // 4, (x.shape[1] + 127) // 128)
                    r *= 8
                    c *= 128
                    _require(
                        r < x.shape[0] and c < x.shape[1],
                        "Input load outside padded tensor",
                    )
                    data = torch.zeros(8, 128)
                    tile = x[r : r + 8, c : c + 128]
                    data[: tile.shape[0], : tile.shape[1]] = tile
                    regs[dst] = _Vector(data, "bf16", origin=(idx, r, c))
            elif op == "vmov":
                _require(
                    args == "0.0", "Only literal zero vector initialization supported"
                )
                regs[dst] = _Vector(torch.zeros(16, 128), "zero")
            elif op == "vcombine.low":
                _require(len(refs) == 2, "Expected two combine operands")
                a, b = map(register, refs)
                _require(
                    a.kind == b.kind == "bf16",
                    "Only packed BF16 low-half combination supported",
                )
                regs[dst] = _Vector(torch.cat((matrix(a, 8), matrix(b, 8))), "bf16")
            elif op == "vunpack.c.l.bf16":
                _require(len(refs) == 1, "Expected one unpack operand")
                v = register(refs[0])
                _require(
                    v.kind == "bf16"
                    and v.origin is not None
                    and v.origin[:2] == (0, self.a_shape[0] - 1)
                    and self.a_shape[0] % 8 == 1
                    and self.a_shape[1] == 256,
                    "Only the captured single-row BF16-unpacked tail is supported",
                )
                data = torch.zeros(8, 128)
                data[:4] = matrix(v, 8)[::2]
                regs[dst] = _Vector(data, "bf16_tail", (True,) + (False,) * 7, v.origin)
            elif op.startswith("vmatprep."):
                _require(len(refs) == 1, "Expected one matrix preparation operand")
                unit = op.rsplit(".", 1)[1]
                target = lhs_prep if ".mubr." in op else rhs_prep
                _require(unit not in target, "Unconsumed matrix preparation")
                target[unit] = (register(refs[0]), op)
            elif op.startswith("vmatpush"):
                _require(len(refs) == 1, "Expected one weight push operand")
                unit = op.rsplit(".", 1)[1]
                key = op.split(".")[0]
                _require(unit in rhs_prep, "Weight push lacks SUBR preparation")
                prep, _ = rhs_prep.pop(unit)
                value = register(refs[0])
                _require(
                    prep.kind in ("bf16", "zero") and value.kind == "bf16",
                    "Unsupported weight operand format",
                )
                _require(
                    not queue[unit] and counts["computed:" + unit] == 0,
                    "Reloading weights after computation is not supported yet",
                )
                weights[unit].setdefault(key, []).append((prep, value))
            elif op.startswith("vmatmul."):
                _require(len(refs) == 1, "Expected one matrix multiply operand")
                unit = op.rsplit(".", 1)[1]
                _require(unit in lhs_prep, "Matrix multiply lacks MUBR preparation")
                prep, prep_op = lhs_prep.pop(unit)
                value = register(refs[0])
                setup = (
                    "vmatpush3"
                    if set(weights[unit]) == {"vmatpush3"}
                    else "vmatpush1+vmatpush2"
                )
                b = rhs_matrix(unit, setup)
                tail = ".f32." in op
                if tail:
                    _require(
                        setup == "vmatpush1+vmatpush2"
                        and ".f32." in prep_op
                        and prep.kind == value.kind == "bf16_tail",
                        "Unsupported FP32 multiply path",
                    )
                    _require(
                        prep.origin[2] == 128 and value.origin[2] == 0,
                        "Unexpected unpacked tail K order",
                    )
                    a = torch.zeros(16, 256)
                    a[:8] = torch.cat((matrix(value, 8), matrix(prep, 8)), dim=1)
                    mnemonic = "vmatmul.mubr.bf16.gmra"
                    rows = 8
                else:
                    _require(
                        ".bf16." in prep_op and prep.kind == value.kind == "bf16",
                        "Mismatched BF16 multiply state",
                    )
                    a = (
                        torch.stack((matrix(value, 16), matrix(prep, 16)))
                        if setup == "vmatpush3"
                        else torch.cat((matrix(value, 16), matrix(prep, 16)), dim=1)
                    )
                    mnemonic = op
                    rows = 16
                model = MMA(self.arch, mnemonic, weight_load=setup)
                if cells is None:
                    result = model(a, b)
                else:
                    evaluate = lazy_group(a, b, model)
                for r in range(0, rows, 8):
                    for part in range(2):
                        valid = ((True,) + (False,) * 7) if tail else (True,) * 8
                        if cells is None:
                            data = (
                                result[part, r : r + 8]
                                if setup == "vmatpush3"
                                else result[r : r + 8, part * 128 : (part + 1) * 128]
                            )
                            vector = _Vector(data, "f32", valid)
                        else:
                            vector = _Vector(
                                None,
                                "f32",
                                valid,
                                bits=lambda i, j, fn=evaluate, p=part, r0=r: fn(
                                    p, r0 + i, j
                                ),
                            )
                        queue[unit].append(vector)
                counts["computed:" + unit] += 1
                counts["groups:" + setup] += 1
                counts["bf16_unpack_tail"] += int(tail)
            elif op.startswith("vpop."):
                unit = op.rsplit(".", 1)[1]
                _require(
                    not args and bool(queue[unit]),
                    "Result FIFO underflow or unsupported pop operands",
                )
                regs[dst] = queue[unit].popleft()
                counts["pops"] += 1
            elif op == "vadd.f32":
                _require(len(refs) == 2, "Expected two vector-add operands")
                a, b = map(register, refs)
                _require(
                    a.kind == b.kind == "f32", "Only FP32 result merging supported"
                )
                valid = tuple(x and y for x, y in zip(a.valid_rows, b.valid_rows))
                if cells is None:
                    aw = (
                        a.data.contiguous()
                        .view(torch.int32)
                        .to(torch.int64)
                        .bitwise_and(0xFFFFFFFF)
                        .flatten()
                        .tolist()
                    )
                    bw = (
                        b.data.contiguous()
                        .view(torch.int32)
                        .to(torch.int64)
                        .bitwise_and(0xFFFFFFFF)
                        .flatten()
                        .tolist()
                    )
                    data = (
                        torch.tensor(
                            [add32(x, y) for x, y in zip(aw, bw)], dtype=torch.int64
                        )
                        .to(torch.int32)
                        .view(torch.float32)
                        .reshape(8, 128)
                    )
                    regs[dst] = _Vector(data, "f32", valid)
                else:
                    regs[dst] = _Vector(
                        None,
                        "f32",
                        valid,
                        bits=lambda i, j, a=a, b=b: add32(a.bits(i, j), b.bits(i, j)),
                    )
                counts["vector_adds"] += 1
            elif op == "vst":
                base, offset, mask = self._address(args)
                # The address can contain a scalar pointer; the final reference is data.
                value = register(refs[-1])
                _require(mask in (0xF, 0xFF), "Unsupported vector store mask")
                _require(base not in self.input_memory, "Input mutation is unsupported")
                if base == self.output_base:
                    _require(
                        mask == 0xFF and offset % 8 == 0 and value.kind == "f32",
                        "Unsupported output store",
                    )
                    _require(
                        offset not in output_writes,
                        "Repeated output store requires an accumulation interpreter",
                    )
                    output_writes.add(offset)
                    r, c = divmod(offset // 8, (self.d_shape[1] + 127) // 128)
                    r *= 8
                    c *= 128
                    _require(
                        r < self.d_shape[0] and c < self.d_shape[1],
                        "Output store outside padded tensor",
                    )
                    _require(
                        all(value.valid_rows[: min(8, self.d_shape[0] - r)]),
                        "Undefined unpack lanes reach a live output",
                    )
                memory[base, offset] = (value, mask)
            elif op == "dma.vmem_to_hbm":
                _require(
                    captured_output is None, "Multiple output transfers unsupported"
                )
                m, n = self.d_shape
                data = (
                    torch.empty(((m + 7) // 8 * 8, (n + 127) // 128 * 128))
                    if cells is None
                    else None
                )
                for r in range(0, m, 8):
                    for c in range(0, n, 128):
                        offset = (r // 8 * ((n + 127) // 128) + c // 128) * 8
                        _require(
                            (self.output_base, offset) in memory,
                            "Incomplete output matrix",
                        )
                        v, _ = memory[self.output_base, offset]
                        if cells is None:
                            data[r : r + 8, c : c + 128] = matrix(v, 8)
                if cells is None:
                    captured_output = data[:m, :n].clone()
                else:
                    captured_output = []
                    for i, j in cells:
                        offset = (i // 8 * ((n + 127) // 128) + j // 128) * 8
                        v, _ = memory[self.output_base, offset]
                        captured_output.append(v.bits(i % 8, j % 128))
        _require(
            captured_output is not None and not any(queue.values()),
            "Incomplete output transfer or undrained MXU FIFO",
        )
        _require(
            not lhs_prep and not rhs_prep, "Unconsumed MXU preparation at program exit"
        )
        if cells is not None:
            counts["evaluated_dots"] = len(dot_cache)
        self.last_execution = dict(counts)
        return captured_output
