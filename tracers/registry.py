from typing import Dict, List, Type

from tracers.base import Tracer
from tracers.torch_dispatch import TorchDispatchTracer
from tracers.torch_profiler import TorchProfilerTracer
from tracers.torch_inductor import TorchInductorTracer
from tracers.torch_dynamo_aten import PrintATenOpsTracer
from tracers.torch_dynamo_fx import PrintFXTracer
from tracers.vllm_profiler import VLLMProfilerTracer

TRACERS = {
    cls.name: cls()
    for cls in [
        TorchDispatchTracer,
        TorchProfilerTracer,
        TorchInductorTracer,
        PrintATenOpsTracer,
        PrintFXTracer,
        VLLMProfilerTracer,
    ]
}

_TRACER_CLASSES: List[Type[Tracer]] = [
    TorchDispatchTracer,
    TorchProfilerTracer,
    TorchInductorTracer,
    PrintATenOpsTracer,
    PrintFXTracer,
    VLLMProfilerTracer,
]

def get_tracer(name: str):
    if name not in TRACERS:
        raise KeyError(f"Unknown tracer: {name}")
    return TRACERS[name]


def list_tracers():
    return list(TRACERS.values())


def list_tracers_for_framework(framework: str):
    return [
        t for t in TRACERS.values()
        if t.supports(framework)
    ]


def all_tracers() -> List[Tracer]:
    """Instantiate all tracers."""
    return [cls() for cls in _TRACER_CLASSES]


def has_tracer(name: str) -> bool:
    return name in TRACERS


def get_tracer_by_name(name: str) -> Tracer:
    for tracer in all_tracers():
        if tracer.name == name:
            return tracer
    raise KeyError(f"Unknown tracer: {name}")


def filter_applicable_tracers(
    tracers: List[Tracer],
    ctx,
) -> List[Tracer]:
    return [t for t in tracers if t.is_applicable(ctx)]