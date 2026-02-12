from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ExecutionContext:
    framework: str
    device: str                 # "cpu" | "cuda"
    mode: str                   # "eval" | "train"
    execution: Any              # ExecutionAdapter

    # torch-like
    model: Optional[Any] = None
    example_inputs: Optional[Any] = None

    # engine-like (vllm)
    engine: Optional[Any] = None

    def clone(self, **kwargs):
        data = self.__dict__.copy()
        data.update(kwargs)
        return ExecutionContext(**data)

    def is_torch(self) -> bool:
        return self.model is not None

    def is_vllm(self) -> bool:
        return self.engine is not None
