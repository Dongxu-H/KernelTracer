import json
def export(trace_ir, output_prefix: str):
    """
    Write files compatible with original kernel_tracer.py outputs
    """

    def dump(obj, suffix):
        with open(f"{output_prefix}_{suffix}.json", "w") as f:
            json.dump(obj, f, indent=4)

    if trace_ir["fx_graph"]:
        dump(trace_ir["fx_graph"], "print_fx")

    if trace_ir["aten_ops"]:
        dump(trace_ir["aten_ops"], "print_aten_ops")

    if trace_ir["operator_inputs"]:
        dump(trace_ir["operator_inputs"], "OperatorInputsMode")

    if trace_ir["profiler_ops"]:
        dump(trace_ir["profiler_ops"], "hf_profile_call_chains")

    if trace_ir["compile_ops"]:
        dump(trace_ir["compile_ops"], "compile")
