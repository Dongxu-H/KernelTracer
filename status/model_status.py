"""
Authoritative model-level status management.
"""

import json
import os
import time
import socket
import getpass
from typing import Optional
from pathlib import Path
from huggingface_hub import snapshot_download


_MODEL_STATUS = "_MODEL_STATUS.json"

def model_status_path(model_id: str) -> str:
    """
    Return absolute path to _MODEL_STATUS for a HF model.
    """
    safe_name = model_id.replace("/", "__")
    return os.path.join(
        os.environ["HF_HOME"],
        "kerneltracer",
        "model_status",
        safe_name,
        "_MODEL_STATUS.json",
    )


def read_model_status(model_id: str) -> Optional[dict]:
    path = model_status_path(model_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_model_status(
    model_id: str,
    *,
    hf_ok: bool,
    vllm_ok: bool,
    stage: str,
    reason: Optional[str] = None,
):
    path = Path(model_status_path(model_id))
    payload = {
        "model": model_id,
        "stage": stage,          # downloaded / validated / traced / failed
        "frameworks": {
            "hf_ok": hf_ok,
            "vllm_ok": vllm_ok,
        },
        "reason": reason,
        "timestamp": int(time.time()),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)



