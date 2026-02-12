import os
import yaml
from pathlib import Path
from typing import Any

class Config:
    def __init__(self, path: str | os.PathLike):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with path.open("r") as f:
            self.raw: dict[str, Any] = yaml.safe_load(f) or {}

    @property
    def env(self) -> dict:
        return self.raw.get("env", {})

    @property
    def timeouts(self) -> dict:
        return self.raw.get("timeouts", {})

    @property
    def behavior(self) -> dict:
        return self.raw.get("behavior", {})

    def _expand(self, v: str) -> str:
        """
        Expand user (~) and env vars in a path-like value.
        """
        return os.path.expandvars(os.path.expanduser(v))

    def apply_bootstrap_env(self) -> None:
        hf_home = os.path.expanduser(
            os.environ.get("HF_HOME", "~/.cache/huggingface")
        )

        hf_home = self._expand(hf_home)

        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
        os.environ.setdefault(
            "HF_DATASETS_CACHE",
            os.path.join(hf_home, "datasets"),
        )

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        # os.environ.setdefault("TORCH_LOGS", "")


    def build_subprocess_env(
            self,
            *,
            framework:str,
            tracer:str | None,
            output_dir: str,
            offline: bool = True,
    ) -> dict[str, str]:
        env = os.environ.copy()

        # ---- common ----
        for k, v in self.env.get("common", {}).items():
            env[k] = self._expand(str(v))

        # ---- offline ----
        if offline:
            for k, v in self.env.get("offline", {}).items():
                env[k] = self._expand(str(v))

        # ---- framework specific ----
        fw_env = self.env.get(framework, {})
        for k, v in fw_env.items():
            env[k] = self._expand(str(v))

        # ---- tracer specific ----
        if tracer:
            tracer_env = self.env.get(tracer, {})
            for k, v in tracer_env.items():
                env[k] = self._expand(str(v))

        # ---- runtime injected ----
        # not configurable, always correct-by-construction
        if framework == "vllm":
            env["VLLM_PROFILE"] = "1"
            env["VLLM_TORCH_PROFILER_DIR"] = output_dir
            env["VLLM_TORCH_PROFILER_WITH_STACK"] = "1"
            env["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "1"
            env["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1"
            env["VLLM_TORCH_PROFILER_WITH_FLOPS"] = "1"

        return env