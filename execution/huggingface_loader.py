import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM
)

def load_huggingface_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model_cls = AutoModel
    if hasattr(config, 'architectures') and config.architectures:
        arch = config.architectures[0]
        if 'CausalLM' in arch:
            model_cls = AutoModelForCausalLM
        elif 'MaskedLM' in arch:
            model_cls = AutoModelForMaskedLM
        elif 'Seq2SeqLM' in arch:
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModel
    else:
        model_cls = AutoModel

    model = model_cls.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16
    )
    model.to(args.device)

    example_inputs = {
        "input_ids": tokenizer(
            "Hello world",
            return_tensors="pt",
        )["input_ids"].to(args.device)
    }

    return model, example_inputs, "hf"

