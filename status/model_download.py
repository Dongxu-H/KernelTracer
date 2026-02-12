from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from dataclasses import dataclass
from typing import Optional
from filelock import FileLock
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)

from status.model_status import read_model_status, write_model_status

@dataclass
class ModelCheckResult:
    model: str
    hf_ok: bool
    vllm_ok: bool
    reason: Optional[str] = None


def check_hf_model_complete(model_id: str) -> None:
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            allow_patterns=None,
        )
    except Exception as e:
        raise RuntimeError(f"HF snapshot missing: {e}")
    
    try:
        AutoConfig.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(f"Config invalid: {e}")
    
    try:
        AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(f"Tokenizer missing or invalid: {e}")

    files = os.listdir(snapshot_path)
    has_weight = any(
        f.endswith((".bin", ".safetensors")) for f in files
    )
    if not has_weight:
        raise RuntimeError("No weight shard found")



SUPPORTED_VLLM_MODEL_TYPES = {
    "llama",
    "mistral",
    "qwen2",
    "qwen",
    "baichuan",
    "gpt_neox",
    "falcon",
    "phi",
}


def check_vllm_compatible(model_id: str) -> None:
    check_hf_model_complete(model_id)
    cfg = AutoConfig.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)

    if cfg.model_type not in SUPPORTED_VLLM_MODEL_TYPES:
        raise RuntimeError(
            f"unsupported model_type for vLLM: {cfg.model_type}"
        )


def download_model_once(model: str):
    cache_dir = os.environ["HF_HOME"]
    lock_path = os.path.join(cache_dir, f"{model.replace('/', '_')}.lock")

    with FileLock(lock_path):
        try:
            logging.info("[download] %s", model)
            snapshot_download(
                repo_id=model,
                resume_download=True,
            )
            logging.info("[download-done] %s", model)
        except Exception as e:
            raise RuntimeError(f"download failed: {model}: {e}")


def ensure_models_downloaded(
    models: list[str],
    max_workers: int = 4,
    check_vllm: bool = True
):
    """
    Concurrent, de-duplicated HF model download.
    """
    unique_models = list(dict.fromkeys(models))

    logging.info(
        "Ensuring %d models downloaded (workers=%d)",
        len(unique_models),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_model_once, m): m
            for m in unique_models
            if not read_model_status(m)
        }

        for fut in as_completed(futures):
            model = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logging.error("[download-failed] %s: %s", model, e)

    results: dict[str, ModelCheckResult] = {}

    for model in unique_models:
        status = read_model_status(model)
        if status and status.get("stage") in ("validated", "traced"):
            logging.info("[model-cache-ok] %s", model)
            continue

        logging.info("Validating model: %s", model)

        # ---- HF ----
        try:
            check_hf_model_complete(model)
            hf_ok = True
        except Exception as e:
            write_model_status(
                model,
                stage="invalid",
                hf_ok=False,
                vllm_ok=False,
                reason=str(e)
            )

            results[model] = ModelCheckResult(
                model=model,
                hf_ok=False,
                vllm_ok=False,
                reason=f"HF check failed: {e}",
            )
            continue

        # ---- vLLM ----
        vllm_ok = False
        reason = None
        if check_vllm:
            try:
                check_vllm_compatible(model)
                vllm_ok = True
            except Exception as e:
                write_model_status(
                    model,
                    stage="invalid",
                    hf_ok=False,
                    vllm_ok=False,
                    reason=str(e)
                )
                reason = str(e)

        write_model_status(
            model,
            stage="validated",
            hf_ok=True,
            vllm_ok=vllm_ok,
            reason=reason,
        )
        
        results[model] = ModelCheckResult(
            model=model,
            hf_ok=True,
            vllm_ok=vllm_ok,
            reason=reason,
        )

    return results
