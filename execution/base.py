from abc import ABC, abstractmethod
from typing import Any


class Execution(ABC):
    """
    Framework-specific execution adapter.
    """

    framework: str

    def __init__(
        self,
        *,
        device: str,
        mode: str = "eval",  # "train" | "eval"
    ):
        self.device = device
        self.mode = mode

    @abstractmethod
    def execute(
        self,
        model: Any,
        example_inputs: Any,
    ) -> Any:
        """
        Execute one step:
        - forward (eval)
        - forward + backward (train)
        """
        raise NotImplementedError

    def supports(self, feature: str) -> bool:
        """
        Used by tracers to check compatibility.
        Example features:
        - torch_dispatch
        - torch_profiler
        - torch_compile
        - loss
        """
        return False
