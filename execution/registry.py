from execution.huggingface_execution import HuggingFaceExecution
from execution.vllm_execution import VLLMExecution


def get_execution_adapter(*, framework: str, device: str, mode: str):
    if framework == "huggingface":
        return HuggingFaceExecution(
            device=device,
            mode=mode,
        )
    elif framework == "vllm":
        return VLLMExecution(
            device=device,
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")
