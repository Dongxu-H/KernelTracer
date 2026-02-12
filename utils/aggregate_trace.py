import json
import os
from pathlib import Path


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_trace(output_dir: str):
    """
    Aggregate split tracer outputs into legacy kernel_tracer format.
    """
    output_dir = Path(output_dir)

    trace_ir = {
        "fx_graph": None,
        "aten_ops": {},
        "operator_inputs": {},
        "profiler_ops": {},
        "compile_ops": None,
    }

    # ------------------------------------------------------------------
    # FX graph
    # ------------------------------------------------------------------
    fx_path = output_dir / "fx_graph.json"
    trace_ir["fx_graph"] = load_json(fx_path)

    # ------------------------------------------------------------------
    # Dynamo ATen (print_aten_ops)
    # ------------------------------------------------------------------
    aten_path = output_dir / "dynamo_aten.json"
    trace_ir["aten_ops"] = load_json(aten_path) or {}

    # ------------------------------------------------------------------
    # TorchDispatch OperatorInputsMode
    # ------------------------------------------------------------------
    op_inputs_path = output_dir / "torch_dispatch.json"
    trace_ir["operator_inputs"] = load_json(op_inputs_path) or {}

    # ------------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------------
    profiler_dir = output_dir / "profiler"
    profiler_json = profiler_dir / "hf_profile_call_chains.json"
    trace_ir["profiler_ops"] = load_json(profiler_json) or {}

    # ------------------------------------------------------------------
    # Compile / Inductor
    # ------------------------------------------------------------------
    compile_path = output_dir / "inductor_kernels.json"
    trace_ir["compile_ops"] = load_json(compile_path)

    return trace_ir
