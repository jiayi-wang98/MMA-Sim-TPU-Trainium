"""Descriptors for captured TPU MXU instruction groups.

The mnemonic is real LLO text. Shapes describe logical operands after unpacking
VMEM/register layouts, not a public TPU assembly ABI. Weight-loading state is
required because vmatmul alone does not specify its arithmetic mode.
"""


class MMA:
    def __init__(self, arch: str, instruction: str, *, weight_load: str):
        if arch != "TPU-v6e":
            raise NotImplementedError("Instruction-group binding is currently validated for TPU-v6e")
        base = instruction
        self.mxu = None
        if base.endswith((".mxu0", ".mxu1")):
            base, self.mxu = base.rsplit(".", 1)
        if base not in (
            "vmatmul.mubr.bf16.gmra",
            "vmatmul.mubr.bf16.vlgmr.msra.gmra",
        ):
            raise NotImplementedError("Unsupported captured BF16 vmatmul instruction")
        if weight_load not in ("vmatpush3", "vmatpush1+vmatpush2"):
            raise ValueError("Specify vmatpush3 or vmatpush1+vmatpush2 weight-loading state")
        self.arch = arch
        self.instruction = instruction
        self.weight_load = weight_load
        self.groups = 2 if weight_load == "vmatpush3" else 1
        self.m = 16
        self.k = self.n = 128 if self.groups == 2 else 256
        self.a_shape = ((2,) if self.groups == 2 else ()) + (self.m, self.k)
        self.b_shape = ((2,) if self.groups == 2 else ()) + (self.k, self.n)
        self.d_shape = ((2,) if self.groups == 2 else ()) + (self.m, self.n)

    @property
    def reduction_groups(self):
        """DP4 term indices per update in the numerical model, before row packing.

        For dual mode this applies separately to each independent stream.
        For wide mode the first pair and second pair are separately added,
        then their results are added before updating the FP32 running sum.
        This is a numerical mapping, not a physical lane assignment.
        """
        if self.groups == 2:
            return tuple((tuple(range(i, i+4)), tuple(range(i+4, i+8)))
                         for i in range(0, 128, 8))
        return tuple((tuple(range(i, i+4)), tuple(range(i+128, i+132)),
                      tuple(range(i+4, i+8)), tuple(range(i+132, i+136)))
                     for i in range(0, 128, 8))
