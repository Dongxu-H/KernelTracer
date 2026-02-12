# Design Document: Kernel Tracing Orchestrator

KernelTracer is a modular tracing framework designed to unify:

- Runtime execution profiling
- Compiler-level kernel extraction
- Multi-framework tracing (HuggingFace / vLLM)

## 1. Background

Modern deep learning frameworks (PyTorch, HuggingFace Transformers, vLLM) generate kernels dynamically:
- TorchDispatch / Dynamo / Inductor
- Framework-owned runtimes (vLLM engine)

Tracing these kernels is **stateful, fragile, and often OOM-prone**.

This project provides a **robust, subprocess-isolated kernel tracing system** that works across:
- HuggingFace (eager / dynamo / compile)
- vLLM (engine-owned execution)

---

## 2. Core Design Principles

### 2.1 Subprocess Isolation (Hard Requirement)

All tracers run in **independent worker processes**.

Why:
- CUDA context leaks
- torch.compile global state
- profiler singleton state
- vLLM engine lifecycle cannot be reset safely

> If a tracer crashes or OOMs, the parent process survives.

---

### 2.2 Orchestrator / Worker Split

```
main.py
  └── orchestrator.py
        └── worker.py  (per tracer, per model)
              ├── load model / engine
              ├── run exactly ONE tracer
              └── write result + SUCCESS marker
```

The **orchestrator never loads models**.
The **worker owns all framework state**.

---

## 3. Tracer Abstraction

Each tracer is a **pure description + metadata**:

```python
class Tracer:
    name: str
    exclusive_group: str | None
    frameworks: set[str]
```

Execution logic lives entirely in the worker.

Why:
- Orchestrator must be framework-agnostic
- Applicability is resolved statically (no ctx needed)

---

## 4. Tracing Registry

`tracers/registry.py` is the single source of truth:

```python
TRACERS = {
    "torch_dispatch": TorchDispatchTracer,
    "torch_profiler": TorchProfilerTracer,
    "torch_compile": TorchCompileTracer,
    "print_aten": PrintATenTracer,
    "print_fx": PrintFXTracer,
    "vllm_profiler": VLLMProfilerTracer,
}
```

Used by:
- CLI validation
- Orchestrator scheduling
- Worker dispatch

---

## 5. ExecutionContext

`ExecutionContext` exists **only in the worker**:

```python
ExecutionContext(
    framework,
    model,
    example_inputs,
    engine,
    device,
    mode,
)
```

It is never serialized or passed between processes.

---

## 6. HuggingFace vs vLLM Tracing

### 6.1 HuggingFace

- Model loaded via `transformers`
- Example inputs generated locally
- Tracing hooks:
  - TorchDispatchMode
  - torch.profiler
  - torch.compile / inductor
  - torch._dynamo / FX

Failure mode:
- Missing shards → immediate exception (good)

---

### 6.2 vLLM

- Engine owns execution loop
- No direct access to model.forward
- Profiling is **engine-owned**

Tracing method:
- Environment variables (e.g. VLLM_TORCH_PROFILER_*)
- Engine.start_profile / stop_profile

Failure mode:
- Partial cache may hang → must pre-validate model

---

## 7. Model Download & Validation

### 7.1 Why Pre-download

- Model download time should not count toward tracer timeout
- Avoid repeated downloads across subprocesses
- Detect partial / broken repos early

### 7.2 ensure_models_downloaded()

Responsibilities:
1. Concurrent snapshot_download
2. HF completeness check
3. Optional vLLM compatibility check
4. Return structured validation results

No model loading happens here.

---

## 8. Bypass & Idempotency

### 8.1 Tracer Bypass

A tracer is bypassed **only if output is complete**.

Example:
```
torch_profiler/
  ├── events/
  └── _TRACE_STATUS.json
```

Directory existence alone is NOT sufficient.

---

### 8.2 Model Cache Bypass

Relies on HuggingFace snapshot integrity.
No custom _MODEL_STATUS file is required.

---

## 9. Failure Handling

| Failure Type | Behavior |
|-------------|----------|
| OOM | Tracer skipped, continue |
| Timeout | Tracer killed |
| Unsupported model | Skipped |
| Partial HF repo | Fatal before tracing |

---

## 10. Why Not In-Process?

Rejected approaches:
- One process, multiple tracers → leaks
- Thread-based isolation → CUDA unsafe
- torch.reset() → incomplete

Subprocess is the only reliable boundary.

---

## 11. Extending the System

To add a new tracer:

1. Implement tracer class
2. Register in `tracers/registry.py`
3. Implement worker-side execution
4. Define success criteria

No orchestrator changes required.

---

## 12. Non-goals

- Training benchmarks
- Performance tuning
- Model accuracy evaluation

This project is **purely diagnostic & structural**.

---

## 13. Summary

This design prioritizes:
- Correctness over speed
- Isolation over convenience
- Determinism over heuristics

It is intentionally conservative — because tracing is fragile.

