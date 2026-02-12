import io
import os
import json
import sys
import torch
from typing import Callable
from tracers.base import Tracer
from execution.context import ExecutionContext
from functorch.compile import aot_module


class PrintATenOpsTracer(Tracer):
    name = "print_aten_ops"
    supported_frameworks = {"huggingface"}

    def is_applicable(self, ctx):
        return ctx.execution.supports("torch_dynamo_aten")

    def run(self, ctx: ExecutionContext, output_dir: str):
        assert ctx.model is not None
        assert ctx.example_inputs is not None

        output_file = os.path.join(output_dir, "print_aten.txt")
        captured = []
        def fw_compiler(gm, example_inputs):
            captured.append(str(gm.graph))
            return gm

        def bw_compiler(gm, example_inputs):
            captured.append(str(gm.graph))
            return gm

        def compiler_fn(gm, example_inputs):
            return aot_module(
                gm,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
            )
        
        def print_aten_ops(gm, example_inputs):
            def trace_printer(gm, _):
                print(gm.graph)
                return gm

            return aot_module(gm, fw_compiler=trace_printer, bw_compiler=trace_printer)

        compiled  = torch._dynamo.optimize(
            compiler_fn,
            nopython=False
        )(ctx.model)

        ctx.execution.execute(
            model=compiled ,
            example_inputs=ctx.example_inputs
        )

        if not captured:
            raise RuntimeError("ATen graph not captured")

        with open(output_file, "w") as f:
            for i, g in enumerate(captured):
                f.write(f"===== Graph {i} =====\n")
                f.write(g)
                f.write("\n\n")

