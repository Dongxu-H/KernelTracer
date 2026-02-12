from vllm import SamplingParams, PoolingParams
from execution.base import Execution


class VLLMExecution(Execution):
    framework = "vllm"

    def supports(self, feature: str) -> bool:
        return feature in {
            "vllm_profiler",
        }

    def __init__(self, *, device: str):
        self.device = device
        self.mode = "eval"


    def execute(
        self,
        model,
        example_inputs=None
    ):
        example_inputs = example_inputs or {}

        prompts = example_inputs.get(
            "prompt", ["The quick brown fox"]
        )
        
        supported_tasks = getattr(model, "supported_tasks", [])

        # Generation
        if "generate" in supported_tasks or "completion" in supported_tasks:
            sampling_params = SamplingParams(
                max_tokens=example_inputs.get("max_tokens", 50),
                top_p=example_inputs.get("top_p", 0.9),
                temperature=example_inputs.get("temperature", 0.7),
            )
            return model.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

        # Embedding / Pooling
        if "embed" in supported_tasks or "encode" in supported_tasks:
            pooling_params = PoolingParams()
            return model.encode(
                prompts=prompts,
                pooling_params=pooling_params,
                pooling_task="embed",
            )

        raise RuntimeError(
            f"Unsupported vLLM tasks: {supported_tasks}"
        )