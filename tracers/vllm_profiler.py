import os
import json
import logging
from collections import defaultdict
from tracers.base import Tracer
from tracers.profiler_extractor import find_vllm_trace_file, write_cpu_kernel_csv, extract_kernel_calls_from_profile
from execution.context import ExecutionContext

class VLLMProfilerTracer(Tracer):
    name = "vllm_profiler"
    supported_frameworks = {"vllm"}

    def is_applicable(self, ctx):
        return ctx.execution.supports("vllm_profiler")


    def run(self, ctx: ExecutionContext, output_dir: str):
        assert ctx.engine is not None
        # assert ctx.example_inputs is not None

        # logging.info(
        #     "[vllm-profiler] start model=%s output_dir=%s",
        #     type(ctx.engine).__name__,
        #     output_dir,
        # )
        
        # warmup
        ctx.execution.execute(ctx.engine, ctx.example_inputs)

        ctx.engine.start_profile()
        ctx.execution.execute(ctx.engine, ctx.example_inputs)
        ctx.engine.stop_profile()

        trace_file = find_vllm_trace_file(output_dir)
        kernel_json = os.path.join(
            output_dir, "vllm_kernel_profile.json"
        )
        kernel_output = extract_kernel_calls_from_profile(
            trace_file,
            kernel_json,
        )

        csv_path = os.path.join(
            output_dir, "vllm_kernel_profile.csv"
        )
        write_cpu_kernel_csv(kernel_output, csv_path)

        with open(trace_file, "r") as f:
            trace = json.load(f)

        events = trace.get("traceEvents", [])
        operator_stats = defaultdict(lambda: {
            "count": 0,
            "cpu_time_total": 0.0,
        })

        for e in events:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat", "")
            if cat not in ("cpu_op", "operator"):
                continue

            name = e.get("name")
            dur = e.get("dur", 0.0)

            operator_stats[name]["count"] += 1
            operator_stats[name]["cpu_time_total"] += dur

        torch_profile_path = os.path.join(
            output_dir, "vllm_profile.json"
        )
        with open(torch_profile_path, "w") as f:
            json.dump(operator_stats, f, indent=2)
