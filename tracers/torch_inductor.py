import io
import os
import sys
import json
import torch
import base64
from typing import Callable
from pathlib import Path

from tracers.base import Tracer
from execution.context import ExecutionContext


def extract_triton_kernels(dump_dir, output_path):
    triton_root = Path(dump_dir) / "triton"

    kernels = []

    if not triton_root.exists():
        return

    for hash_dir in triton_root.glob("*/*"):
        if not hash_dir.is_dir():
            continue

        for json_file in hash_dir.glob("*.json"):
            # skip group json
            if json_file.name.startswith("__grp__"):
                continue

            kernel_name = json_file.stem

            kernel_entry = {
                "kernel_name": kernel_name,
                "ptx": None,
                "cubin": None,
                "launch_config": None,
            }

            # read metadata json
            try:
                with open(json_file, "r") as f:
                    meta = json.load(f)
                kernel_entry["launch_config"] = meta
            except Exception:
                pass

            # read PTX
            ptx_file = json_file.with_suffix(".ptx")
            if ptx_file.exists():
                kernel_entry["ptx"] = str(ptx_file)
                # with open(ptx_file, "r") as f:
                    # kernel_entry["ptx"] = f.read()
                    # kernel_entry["ptx_size"] = len(kernel_entry["ptx"])


            # read cubin (binary -> hex or base64)
            cubin_file = json_file.with_suffix(".cubin")
            if cubin_file.exists():
                kernel_entry["cubin"] = str(cubin_file)
                # with open(cubin_file, "rb") as f:
                    # kernel_entry["cubin"] = base64.b64encode(f.read()).decode()

            kernels.append(kernel_entry)

    # remove duplicates by kernel_name
    unique = {}
    for k in kernels:
        unique[k["kernel_name"]] = k

    final_list = list(unique.values())

    with open(output_path, "w") as f:
        json.dump(final_list, f, indent=2)

    return final_list


class TorchInductorTracer(Tracer):
    name = "torch_inductor"
    supported_frameworks = {"huggingface"}

    def is_applicable(self, ctx):
        return ctx.execution.supports("torch_inductor")

    def run(self, ctx: ExecutionContext, output_dir: str):
        assert ctx.model is not None
        assert ctx.example_inputs is not None

        ctx.model.eval()
        for p in ctx.model.parameters():
            p.requires_grad_(False)

        dump_dir = os.path.join(output_dir, "inductor_dump")
        os.makedirs(dump_dir, exist_ok=True)
        
        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.verbose = True

        os.environ["TORCHINDUCTOR_CACHE_DIR"] = dump_dir
        os.environ["TORCHINDUCTOR_TRACE"] = "1"
        os.environ["TORCHINDUCTOR_VERBOSE"] = "1"
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        os.environ["TORCH_LOGS"] = "dynamo,inductor"

        compiled = torch.compile(
            ctx.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )

        ctx.execution.execute(
            model=compiled,
            example_inputs=ctx.example_inputs,
        )

        torch.cuda.synchronize()

        dumped_files = list(os.walk(dump_dir))
        if not dumped_files or all(len(files) == 0 for _, _, files in dumped_files):
            raise RuntimeError("Inductor produced no dump files")

        # Optional summary
        summary_path = os.path.join(output_dir, "torch_inductor.txt")
        with open(summary_path, "w") as f:
            for root, _, files in dumped_files:
                for name in files:
                    f.write(os.path.join(root, name) + "\n")
        
        structured_path = os.path.join(output_dir, "torch_inductor_kernels.json")
        extract_triton_kernels(dump_dir, structured_path)

