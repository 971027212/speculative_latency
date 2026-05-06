from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_specdec.config import load_model_pairs, select_model_pairs
from edge_specdec.models import choose_device, load_causal_lm, load_tokenizer
from edge_specdec.prompts import DEFAULT_PROMPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Hugging Face assistant_model sanity check."
    )
    parser.add_argument("--config", default="configs/model_pairs.yaml")
    parser.add_argument("--pair", action="append", help="Model pair name. Repeatable.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPTS[0])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()
    pairs = select_model_pairs(load_model_pairs(args.config), args.pair)

    for pair in pairs:
        tokenizer = load_tokenizer(pair.target, trust_remote_code=args.trust_remote_code)
        target_model = load_causal_lm(
            pair.target,
            device=device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        draft_model = load_causal_lm(
            pair.draft,
            device=device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )

        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = target_model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(
            json.dumps(
                {
                    "step": "hf_assisted_sanity",
                    "model_pair": pair.name,
                    "target": pair.target,
                    "draft": pair.draft,
                    "device": str(device),
                    "seconds": seconds,
                    "prompt": args.prompt,
                    "output": text,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
