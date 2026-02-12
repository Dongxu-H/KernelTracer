import json
import os
import functools
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils import _pytree as pytree
from tracers.base import Tracer
from execution.context import ExecutionContext
aten = torch.ops.aten

# -----------------------------------------------------------------------------
# dtype serialization
# -----------------------------------------------------------------------------

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {v: k for k, v in dtype_abbrs.items()}
tensor_type = torch._C.TensorType.get()


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def truncate_input(x):
    if x in dtype_abbrs:
        return dtype_abbrs[x]
    if isinstance(x, torch.device):
        return x.type
    return x


def contains_tensor(elems):
    return any(isinstance(x, torch.Tensor) for x in pytree.tree_leaves(elems))


def skip_args(elems):
    for x in pytree.tree_leaves(elems):
        if isinstance(x, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
    return False


def serialize_tensor(t: torch.Tensor):
    if not t.is_contiguous():
        return FuncCallWrapper(
            "T", list(t.shape), t.dtype, stride=t.stride()
        )
    return FuncCallWrapper("T", list(t.shape), t.dtype)


def serialize_torch_args(x):
    if isinstance(x, torch.Tensor):
        return serialize_tensor(x)
    return truncate_input(x)


def contains_tensor_types(tp):
    return tp.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(t) for t in tp.containedTypes()
    )


@functools.lru_cache(None)
def non_compute_operator(op):
    schema = op._schema

    # no tensor input
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True

    # skip *_like
    if "_like" in op.name():
        return True

    # mutable ops allowed
    if schema.is_mutable:
        return False

    tensor_outs = [r for r in schema.returns if r.type is tensor_type]
    if len(tensor_outs) != 1:
        return False

    return False


# -----------------------------------------------------------------------------
# serialization wrapper
# -----------------------------------------------------------------------------

class FuncCallWrapper:
    def __init__(self, call, *args, **kwargs):
        self.call = call
        self.args = tree_map(truncate_input, args)
        self.kwargs = tree_map(truncate_input, kwargs)

    def __repr__(self):
        args = ", ".join(repr(a) for a in self.args)
        kwargs = "".join(f", {k}={v}" for k, v in self.kwargs.items())
        out = f"{self.call}({args}{kwargs})"
        for abbr in dtype_abbrs_parsing:
            out = out.replace(f"'{abbr}'", abbr)
        return out


# -----------------------------------------------------------------------------
# core TorchDispatch tracer
# -----------------------------------------------------------------------------

class OperatorInputsMode(TorchDispatchMode):
    """
    Trace real aten kernel invocations with input shapes / dtypes.
    """

    def __init__(self, output_file: str):
        super().__init__()
        self.output_file = output_file
        self.func_db = defaultdict(Counter)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        arg_meta, kwarg_meta = tree_map(
            serialize_torch_args, (args, kwargs)
        )

        out = func(*args, **kwargs)

        inputs = (args, kwargs)
        if (
            contains_tensor(inputs)
            and contains_tensor(out)
            and not skip_args(inputs)
        ):
            key = repr((arg_meta, kwarg_meta))
            self.func_db[str(func)][key] += 1

        return out

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        self._dump()

    def _dump(self):
        result = {}

        for op_name in sorted(self.func_db):
            op = eval(op_name, {"aten": torch.ops.aten})
            if non_compute_operator(op):
                continue

            result[op_name] = []
            for inputs, count in self.func_db[op_name].items():
                result[op_name].append({
                    "count": count,
                    "inputs": inputs,
                })

        with open(self.output_file, "w") as f:
            json.dump(result, f, indent=2)


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

class TorchDispatchTracer(Tracer):
    name = "torch_dispatch"
    supported_frameworks = {"huggingface"}

    def is_applicable(self, ctx):
        # return ctx.model is not None and hasattr(torch._C, "_EnableTorchDispatch")
        return ctx.execution.supports("torch_dispatch")

    def run(self, ctx: ExecutionContext, output_dir: str):
        assert ctx.model is not None
        assert ctx.example_inputs is not None

        output_file=os.path.join(output_dir, "torch_dispatch.json")
        with OperatorInputsMode(output_file):
            ctx.execution.execute(
                model=ctx.model,
                example_inputs=ctx.example_inputs
            )
