import io
import os
import json
import sys
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
)
from collections import defaultdict

from tracers.base import Tracer
from tracers.profiler_extractor import wait_for_trace_file, write_cpu_kernel_csv, extract_kernel_calls_from_profile
from execution.context import ExecutionContext

class TorchProfilerTracer(Tracer):
    name = "torch_profiler"
    supported_frameworks = {"huggingface"}

    def is_applicable(self, ctx):
        return ctx.execution.supports("torch_profiler")

    def run(self, ctx: ExecutionContext, output_dir: str):
        assert ctx.model is not None
        assert ctx.example_inputs is not None
        trace_dir = os.path.join(output_dir, "trace")
        os.makedirs(trace_dir, exist_ok=True)

        record_shapes = True
        profile_memory = True
        with_stack = True
        with_flops = True
        with_modules = True
        wait = 0
        warmup = 1
        active = 1
        repeat = 1
        verbose = True

        activities = [ProfilerActivity.CPU]
        if ctx.device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        exp_config = None
        if verbose:
            exp_config = torch._C._profiler._ExperimentalConfig(
                verbose=True
            )

        prof_schedule = schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        with profile(
            activities=activities,
            schedule=prof_schedule,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            use_cuda=(ctx.device == "cuda"),
            experimental_config=exp_config,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                trace_dir
            ),
        ) as prof:
            for _ in range(warmup + active):
                ctx.execution.execute(model=ctx.model,
                        example_inputs=ctx.example_inputs
                    )
                prof.step()

            operator_stats = defaultdict(list)

            for event in prof.events():
                op_name = event.name
                if event.count <= 0 or op_name.startswith("ProfilerStep"):
                    continue

                input_shapes = (
                    str(event.input_shapes)
                    if hasattr(event, "input_shapes")
                    else ""
                )

                stack = (
                    event.stack
                    if hasattr(event, "stack") and event.stack
                    else []
                )

                existing_entry = None
                for entry in operator_stats[op_name]:
                    if entry.get("input_shapes") == input_shapes:
                        existing_entry = entry
                        break

                if existing_entry:
                    existing_entry["count"] += 1
                    existing_entry["cpu_time_total"] += event.cpu_time_total
                    existing_entry["cuda_time_total"] += (
                        event.cuda_time_total
                        if hasattr(event, "cuda_time_total")
                        else 0
                    )
                    existing_entry["self_cpu_time_total"] += (
                        event.self_cpu_time_total
                    )
                else:
                    operator_stats[op_name].append(
                        {
                            "count": event.count,
                            "cpu_time_total": event.cpu_time_total,
                            "cuda_time_total": (
                                event.cuda_time_total
                                if hasattr(event, "cuda_time_total")
                                else 0
                            ),
                            "self_cpu_time_total": event.self_cpu_time_total,
                            "input_shapes": input_shapes,
                            "stack": event.stack if hasattr(event, 'stack') and event.stack else [],
                        }
                    )

            output_file = os.path.join(output_dir, "torch_profile.json")
            with open(output_file, "w") as f:
                json.dump(dict(operator_stats), f, indent=4)

            trace_file = wait_for_trace_file(trace_dir)

            kernel_json = os.path.join(
                output_dir, "torch_kernel_profile.json"
            )
            kernel_output = extract_kernel_calls_from_profile(
                trace_file, kernel_json
            )

            csv_path = os.path.join(
                output_dir, "torch_kernel_profile.csv"
            )
            write_cpu_kernel_csv(kernel_output, csv_path)

