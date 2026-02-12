import os
import torch
import json
import subprocess
import sys
import logging
import shutil

from core.config import Config
from status.model_download import ensure_models_downloaded
from status.model_policy import ModelPolicy, ModelDecision
from status.model_status import read_model_status, write_model_status
from status.trace_policy import TracePolicy, TraceDecision
from status.trace_status import read_trace_status, write_trace_status
from huggingface_hub import snapshot_download

def run_tracer_subprocess(
    *,
    framework: str,
    model: str,
    tracer: str,
    output_dir: str,
    device: str,
    mode: str,
    tensor_parallel_size: int | None = None,
    cfg,
    timeout: int = 1800,
):
    snapshot_path = snapshot_download(
        repo_id=model,
        local_files_only=True,
    )

    cmd = [
        sys.executable,
        "core/worker.py",
        "--framework", framework,
        "--model-path", snapshot_path,
        "--tracer", tracer,
        "--output-dir", output_dir,
        "--device", device,
        "--mode", mode,
    ]

    if framework == "vllm" and tensor_parallel_size is not None:
        cmd += [
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]

        cmd += [
            "--model",
            str(model),
        ]
    
    timeout = (
        cfg.timeouts.get("vllm_trace", 1800)
        if framework == "vllm"
        else cfg.timeouts.get("hf_trace", 900)
    )

    env=cfg.build_subprocess_env(
        framework=framework, 
        tracer=tracer, 
        output_dir=output_dir
    )

    logging.info("Running: %s", " ".join(cmd))
    logging.info(
        "[spawn] tracer=%s output_dir=%s",
        tracer,
        output_dir,
    )

    #logging.info(
    #    "HF env: HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s HF_HOME=%s",
    #    env.get("HF_HUB_OFFLINE"),
    #    env.get("TRANSFORMERS_OFFLINE"),
    #    env.get("HF_HOME"),
    #)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        out, _ = proc.communicate(timeout=timeout)
        out = out.decode("utf-8", errors="replace")

        if proc.returncode == 0 and "SUCCESS" in out:
            return True, out

        if proc.returncode == 9:
            logging.error("[OOM killed] %s", tracer)
            return False, out

        logging.error("Tracer %s failed:\n%s", tracer, out)
        return False, out

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        logging.error("[timeout] %s", tracer)
        return False, "[timeout]"


def trace_model_orchestrated(
    *,
    framework: str,
    model: str,
    tracers: list,
    output_root: str,
    device: str,
    mode: str,
    tensor_parallel_size: int | None = None,
    cfg: Config,
    force_download: bool = False,
    force_trace: bool = False,
):
    """
    Orchestrate tracing for a single model.

    Responsibilities:
      - model-level policy decision
      - tracer-level policy decision
      - filesystem I/O
      - subprocess execution
    """
    # ------------------------------------------------------------------
    # model-level decision
    # ------------------------------------------------------------------

    model_policy = ModelPolicy(force=force_download)
    model_status = read_model_status(model)
    decision = model_policy.decide(model_status)

    if decision == ModelDecision.BYPASS:
        logging.info("[model-bypass] %s", model)

    if decision == ModelDecision.ERROR:
        raise RuntimeError(
            f"Model {model} in invalid state: {model_status}"
        )

    if decision == ModelDecision.RUN_DOWNLOAD:
        ensure_models_downloaded(
            [model],
            max_workers=cfg.timeouts.get("download_workers", 4),
            check_vllm=(framework == "vllm"),
        )

    # ------------------------------------------------------------------
    # execution-level policy
    # ------------------------------------------------------------------

    logging.info("=== tracing model: %s ===", model)

    exec_policy = TracePolicy(cfg.behavior, force_trace)

    model_root = os.path.join(output_root, model)
    model_root = os.path.join(model_root, mode)
    os.makedirs(model_root, exist_ok=True)
    completed_groups: set[str] = set()

    for tracer in tracers:
        if tracer.exclusive_group and tracer.exclusive_group in completed_groups:
            logging.info(
                "[exclusive-skip] %s (%s)",
                tracer.name,
                tracer.exclusive_group,
            )
            continue

        tracer_dir = os.path.join(model_root, tracer.name)
        trace_status = read_trace_status(tracer_dir)
        decision = exec_policy.decide(
            trace_status=trace_status,
            tracer_name=tracer.name,
            framework=framework,
            device=device,
            mode=mode,
        )

        if decision is TraceDecision.BYPASS:
            logging.info(
                "[bypass] %s / %s (already traced)",
                model,
                tracer.name,
            )
            if tracer.exclusive_group:
                completed_groups.add(tracer.exclusive_group)
            continue

        os.makedirs(tracer_dir, exist_ok=True)

        ok, output = run_tracer_subprocess(
            framework=framework,
            model=model,
            tracer=tracer.name,
            output_dir=tracer_dir,
            device=device,
            mode=mode,
            tensor_parallel_size=tensor_parallel_size,
            cfg=cfg,
        )

        if ok:
            if tracer.exclusive_group:
                completed_groups.add(tracer.exclusive_group)

        else:
            if exec_policy.should_continue_on_failure():
                logging.error(
                    "[trace-failed] %s / %s",
                    model,
                    tracer.name,
                )
            else:
                raise RuntimeError(f"Tracer failed: {tracer.name}")


        # --------------------------------------------------------------
        # cleanup
        # --------------------------------------------------------------
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ------------------------------------------------------------------
    # model fully traced
    # ------------------------------------------------------------------

    #write_model_status(
    #    model,
    #    stage="traced",
    #    hf_ok=True,
    #    vllm_ok=True,
    #)

    logging.info("=== done: %s ===", model)