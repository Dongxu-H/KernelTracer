# trace_runtime/tracers/base.py
from abc import ABC, abstractmethod
from execution.context import ExecutionContext
from typing import Optional


class Tracer(ABC):
    name: str
    description: str = ""
    requires_subprocess: bool = True
    exclusive_group: str | None = None
    supported_frameworks: set[str] = set()

    @abstractmethod
    def is_applicable(self, ctx: ExecutionContext) -> bool:
        pass

    @abstractmethod
    def run(self, ctx: ExecutionContext, output_dir: str):
        pass

    def supports(self, framework: str) -> bool:
        return framework in self.supported_frameworks
