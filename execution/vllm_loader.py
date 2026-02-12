
import os
from vllm import LLM

def load_vllm_engine(args):

    os.environ["VLLM_TORCH_PROFILER_DIR"] = args.output_dir
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "1"
    os.environ["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "1"
    os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1"
    os.environ["VLLM_TORCH_PROFILER_WITH_FLOPS"] = "1"
    os.environ["VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL"] = "1"

    engine = LLM(
        model=args.model,
        download_dir=os.environ.get("HF_HOME"),
        runner="pooling",
        tensor_parallel_size=1,  # use 1 instead of torch.cuda.device_count() to avoid NCCL kernels
        trust_remote_code=True,
        load_format="dummy",
        dtype="float16",
        disable_log_stats=True,
        enforce_eager=True,
    )

    return engine, None, engine
