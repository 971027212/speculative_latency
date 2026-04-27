from __future__ import annotations

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype | str:
    normalized = dtype.lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"fp16", "float16"}:
        if device.type == "cpu":
            warnings.warn("float16 on CPU is often unsupported; using float32 instead.")
            return torch.float32
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        if device.type == "cpu":
            warnings.warn("bfloat16 on CPU can be slow/unsupported; using float32 instead.")
            return torch.float32
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    device: torch.device,
    dtype: str = "auto",
    trust_remote_code: bool = False,
):
    torch_dtype = resolve_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model
