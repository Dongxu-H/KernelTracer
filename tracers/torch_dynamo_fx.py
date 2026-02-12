import io
import os
import json
import sys
import torch
from typing import Any
from tracers.base import Tracer
from execution.context import ExecutionContext


# -----------------------------------------------------------------------------
# FX graph serializer
# -----------------------------------------------------------------------------

def serialize_fx_node(node):
    def _serialize_arg(x):
        if isinstance(x, torch.fx.Node):
            return x.name
        if isinstance(x, (list, tuple)):
            return [_serialize_arg(i) for i in x]
        if isinstance(x, dict):
            return {k: _serialize_arg(v) for k, v in x.items()}
        return repr(x)

    return {
        "name": node.name,
        "op": node.op,
        "target": str(node.target),
        "args": _serialize_arg(node.args),
        "kwargs": _serialize_arg(node.kwargs),
        "users": [u.name for u in node.users],
    }


def serialize_fx_graph(gm: torch.fx.GraphModule):
    return {
        "type": "fx_graph",
        "nodes": [serialize_fx_node(n) for n in gm.graph.nodes],
    }


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

class PrintFXTracer(Tracer):
    name = "print_fx"
    supported_frameworks = {"huggingface"}

    def is_applicable(self, ctx):
        return ctx.execution.supports("torch_dynamo_aten")

    def run(self, ctx: ExecutionContext, output_dir: str):
        assert ctx.model is not None
        assert ctx.example_inputs is not None

        output_file = os.path.join(output_dir, "print_fx.json")
        def fx_backend(gm, example_inputs):
            graph_json = serialize_fx_graph(gm)
            with open(output_file, "w") as f:
                json.dump(graph_json, f, indent=4)
            return gm.forward

        optimized = torch._dynamo.optimize(
            fx_backend,
            nopython=False,
        )(ctx.model)

        ctx.execution.execute(
            model=optimized,
            example_inputs=ctx.example_inputs
        )

