#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import torch
import traceback

from execution.context import ExecutionContext
from execution.registry import get_execution_adapter
from execution.huggingface_loader import load_huggingface_model
from execution.vllm_loader import load_vllm_engine
from tracers.registry import TRACERS, get_tracer
from tracers.tracer_all import trace_all
from status.trace_status import write_trace_status

logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

FRAMEWORK_LOADERS = {
    "huggingface": load_huggingface_model,
    "vllm": load_vllm_engine,
}

def exit_success():
    print("SUCCESS")
    sys.exit(0)


def exit_oom():
    print("[oom]")
    sys.exit(9)


def normalize_trace_methods(spec: str):
    if spec == "all":
        return list(TRACERS.keys())
    return [m.strip() for m in spec.split(",") if m.strip()]


def main():
    # logging.info(
    #     "HF env: HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s HF_HOME=%s",
    #     os.environ.get("HF_HUB_OFFLINE"),
    #     os.environ.get("TRANSFORMERS_OFFLINE"),
    #     os.environ.get("HF_HOME"),
    # )

    parser = argparse.ArgumentParser("trace worker")

    parser.add_argument("--framework", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracer", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="eval")

    # vllm
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.framework not in FRAMEWORK_LOADERS:
        raise ValueError(f"Unknown framework: {args.framework}")
    
    if args.tracer not in TRACERS:
        raise ValueError(f"Unknown tracer: {args.tracer}")

    os.makedirs(args.output_dir, exist_ok=True)

    assert os.path.isdir(args.model_path)
    assert "snapshots" in args.model_path

    logging.info(
        "[worker] start framework=%s tracer=%s output_dir=%s",
        args.framework,
        args.tracer,
        args.output_dir,
    )

    try:
        model, example_inputs, engine = FRAMEWORK_LOADERS[args.framework](args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[oom] model loading failed")
            sys.exit(9)
        raise

    execution = get_execution_adapter(
        framework=args.framework,
        device=args.device,
        mode=args.mode,
    )

    ctx = ExecutionContext(
        framework=args.framework,
        execution=execution,
        model=model,
        example_inputs=example_inputs,
        engine=engine,
        device=args.device,
        mode=args.mode,
    )

    methods = normalize_trace_methods(args.tracer)

    for method in methods:
        try:
            tracer = get_tracer(method)
            logging.info(
                "[worker] invoking tracer=%s (%s)",
                tracer.name,
                tracer.__class__.__name__,
            )
            
            success = trace_all(
                ctx=ctx,
                tracers=[tracer],
                output_dir=args.output_dir,
            )

            if success is True:
                logging.info(
                    "[worker] tracer=%s completed, output_dir=%s, files=%s",
                    tracer.name,
                    args.output_dir,
                    os.listdir(args.output_dir),
                )
                write_trace_status(
                    output_dir=args.output_dir,
                    tracer=tracer.name,
                    framework=ctx.framework,
                    model=model,
                    device=ctx.device,
                    mode=ctx.mode,
                    success=True
                )
            #print("SUCCESS")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[oom] {method}")
                sys.exit(9)
            raise
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        finally:
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    #print("SUCCESS")


if __name__ == "__main__":
    main()
