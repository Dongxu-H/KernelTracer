import logging
import os
import traceback
from typing import Iterable

from execution.context import ExecutionContext
from tracers.base import Tracer

def trace_all(
    ctx: ExecutionContext,
    tracers: Iterable[Tracer],
    output_dir: str,
):
    #os.makedirs(output_dir, exist_ok=True)
    completed_exclusive_groups: set[str] = set()
    if ctx.execution is None:
        raise ValueError("ExecutionContext.execution is required")

    for tracer in tracers:
        tracer_name = tracer.name
        #tracer_out = os.path.join(output_dir, tracer_name)
        #os.makedirs(tracer_out, exist_ok=True)

        logging.info(
            "[vllm-profiler] execution.supports=%s",
            ctx.execution.supports("vllm_profiler"),
        )
        try:
            if not tracer.is_applicable(ctx):
                logging.info(
                    f"[trace_all] skip {tracer_name}: incompatible"
                )
                continue
        except Exception:
            logging.warning(
                f"[trace_all] skip {tracer_name}: compatibility check failed"
            )
            continue

        if tracer.exclusive_group is not None:
            if tracer.exclusive_group in completed_exclusive_groups:
                logging.info(
                    f"[trace_all] skip {tracer_name}: "
                    f"exclusive_group={tracer.exclusive_group} already done"
                )
                continue

        logging.info(f"[trace_all] running {tracer_name}")

        try:
            logging.info(
                "[vllm-profiler] execution.supports=%s",
                ctx.execution.supports("vllm_profiler"),
            )
            tracer.run(
                ctx=ctx,
                output_dir=output_dir,
            )
            success = True
        except Exception as e:
            success = False
            logging.error(
                f"[trace_all] tracer {tracer_name} failed: {e}"
            )
            traceback.print_exc()

        if success and tracer.exclusive_group is not None:
            completed_exclusive_groups.add(tracer.exclusive_group)

        try:
            if ctx.device == "cuda":
                import torch
                torch.cuda.empty_cache()
        except Exception:
            pass

        return success

