# KernelTracer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/framework-HuggingFace%20%7C%20vLLM-orange)

A **production-oriented kernel / operator tracing framework** for large language models.
It supports **HuggingFace Transformers** and **vLLM**, runs tracers in **isolated subprocesses**,
and is designed for **batch model evaluation at scale**.

---

## Features

- Multiple tracing backends
  - torch.profiler
  - torch dispatch / dynamo / FX / Inductor
  - vLLM built-in profiler
- Unified tracer abstraction (Tracer class + registry)
- **Subprocess isolation** (OOM-safe, crash-safe)
- **Concurrent, de-duplicated model download**
- Model skip-list support
- Intelligent **bypass if trace already exists**
- Built-in support for HF mirror & offline mode
- Works with both **HF eager execution** and **vLLM engines**

---

## Installation

```bash
pip install -e .
```

With vLLM support:

```bash
pip install -e .[vllm]
```

Requirements:
- Python ≥ 3.10
- PyTorch ≥ 2.2
- transformers ≥ 4.40.0
- vllm ≥ 0.11.0 (optional, only if tracing vLLM)

---

## Quick Start

### list available tracers

```bash
python main.py --list-tracers
```


### Trace a single model

```bash
python main.py \
  --framework huggingface \
  --model meta-llama/Llama-2-7b-hf \
  --output-dir outputs/llama2
```

```bash
python main.py \
  --framework vllm \
  --model meta-llama/Llama-2-7b-hf \
  --output-dir outputs/llama2
```

### Trace a single model with specific tracer

```bash
python main.py \
  --framework huggingface \
  --model meta-llama/Llama-2-7b-hf \
  --tracers torch_profiler \
  --output-dir outputs/llama2
```

```bash
python main.py \
  --framework vllm \
  --model meta-llama/Llama-2-7b-hf \
  --tracers vllm_profiler \
  --output-dir outputs/llama2
```

### Trace a single model with specific mode

```bash
python main.py \
  --framework huggingface \
  --model meta-llama/Llama-2-7b-hf \
  --mode train \
  --output-dir outputs/llama2
```

```bash
python main.py \
  --framework vllm \
  --model meta-llama/Llama-2-7b-hf \
  --mode eval \
  --output-dir outputs/llama2
```

### Trace a single model without download (local path)

```bash
python core/worker.py \
  --framework huggingface \
  --model-path /path/to/model \
  --tracer torch_profiler \
  --mode eval
  --output-dir /path/to/output
```

```bash
python core/worker.py \
  --framework vllm \
  --model-path /path/to/model \
  --tracer vllm_profiler \
  --mode eval
  --output-dir /path/to/output
```

### Trace multiple models from config

```bash
python main.py \
  --framework huggingface \
  --model-list config/models.txt \
  --skip-model-list config/skip.txt \
  --output-dir outputs
```

```bash
python main.py \
  --framework vllm \
  --model-list config/models.txt \
  --skip-model-list config/skip.txt \
  --output-dir outputs
```

### Trace multiple models from config with specific tracer

```bash
python main.py \
  --framework huggingface \
  --model-list config/models.txt \
  --tracers torch_profiler \
  --skip-model-list config/skip.txt \
  --output-dir outputs
```

```bash
python main.py \
  --framework vllm \
  --model-list config/models.txt \
  --tracers vllm_profiler \
  --skip-model-list config/skip.txt \
  --output-dir outputs
```

### Trace multiple models from config with specific mode

```bash
python main.py \
  --framework huggingface \
  --model-list config/models.txt \
  --mode train \
  --skip-model-list config/skip.txt \
  --output-dir outputs
```

```bash
python main.py \
  --framework vllm \
  --model-list config/models.txt \
  --mode eval \
  --skip-model-list config/skip.txt \
  --output-dir outputs
```

### Force re-run even if results exist

```bash
python main.py \
  --framework huggingface \
  --model meta-llama/Llama-2-7b-hf \
  --force
```

```bash
python main.py \
  --framework vllm \
  --model meta-llama/Llama-2-7b-hf \
  --force
```

### Force re-run even if results exist with specific tracer

```bash
python main.py \
  --framework huggingface \
  --tracers torch_profiler \
  --model meta-llama/Llama-2-7b-hf \
  --force
```

```bash
python main.py \
  --framework vllm \
  --tracers vllm_profiler \
  --model meta-llama/Llama-2-7b-hf \
  --force
```

### Force re-run even if results exist with specific mode

```bash
python main.py \
  --framework huggingface \
  --mode train \
  --model meta-llama/Llama-2-7b-hf \
  --force
```

```bash
python main.py \
  --framework vllm \
  --mode eval \
  --model meta-llama/Llama-2-7b-hf \
  --force
```

---

### Why subprocesses?

- Torch profiler & dynamo are **stateful**
- vLLM may **hang or OOM**
- Subprocess isolation guarantees:
  - clean CUDA state
  - safe timeout
  - partial failure tolerance

---

## Tracing Semantics: HF vs vLLM

### HuggingFace

- Execution unit: `nn.Module`
- Input: tokenized tensors
- Tracing scope:
  - aten ops
  - FX graphs
  - torch.profiler events
  - torch.inductor IR
  - torch.dispatch
- Failure mode:
  - Missing shards → exception (fast fail)

### vLLM

- Execution unit: `LLMEngine`
- Input: prompt strings
- Tracing scope:
  - CUDA kernels inside engine
  - attention / KV cache behavior
- Failure mode:
  - Partial cache → hang or silent stall

**Therefore this project enforces strict model cache validation before tracing.**

---

## Output Layout

```
outputs/
  llama2/
    torch_profiler/
      torch_profile.json
      torch_kernel_profile.json
      ...
    print_fx/
      print_fx.json
```

Each tracer owns **one directory**, and success is marked by:

```
_TRACE_STATUS.json
```

Used for bypass & resumability.

---

## FAQ

### Q: Why not run all tracers in one process?
A: Torch dynamo / profiler conflict with each other and leak global state.

### Q: Can I trace models without HF permission?
A: Yes. Download failures are detected early and skipped safely.

### Q: Why validate vLLM compatibility explicitly?
A: vLLM may hang if model type or shards are incomplete.

### Q: How to add a new tracer?
A:
```python
class MyTracer(Tracer):
    name = "my_tracer"
    exclusive_group = None

    def run(self, ctx, output_dir):
        ...
```

Register it in `tracers/registry.py`.

---

## License

MIT

## Contact

dxhan@baai.ac.cn
