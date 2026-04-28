from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_specdec.config import load_model_pairs, select_model_pairs
from edge_specdec.decoding import (
    speculative_greedy,
    speculative_greedy_cached,
    target_only_greedy,
    target_only_greedy_cached,
)
from edge_specdec.models import choose_device, load_causal_lm, load_tokenizer
from edge_specdec.prompts import DEFAULT_PROMPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2/3: run the minimal greedy speculative decoder."
    )
    parser.add_argument("--config", default="configs/model_pairs.yaml")
    parser.add_argument("--pair", action="append", help="Model pair name. Repeatable.")
    parser.add_argument("--prompt", action="append", help="Prompt text. Repeatable.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--rtt-ms", type=float, default=0.0)
    parser.add_argument(
        "--implementation",
        default="full-prefix",
        choices=["full-prefix", "kv-cache"],
    )
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()
    prompts = args.prompt or DEFAULT_PROMPTS
    pairs = select_model_pairs(load_model_pairs(args.config), args.pair)

    for pair in pairs:
        tokenizer = load_tokenizer(pair.target, trust_remote_code=args.trust_remote_code)
        target_model = load_causal_lm(
            pair.target,
            device=device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        draft_model = load_causal_lm(
            pair.draft,
            device=device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )

        for prompt_id, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = encoded["input_ids"]
            if args.implementation == "kv-cache":
                baseline = target_only_greedy_cached(
                    target_model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                )
                spec = speculative_greedy_cached(
                    target_model,
                    draft_model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    draft_k=args.draft_k,
                    rtt_ms=args.rtt_ms,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                baseline = target_only_greedy(
                    target_model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                )
                spec = speculative_greedy(
                    target_model,
                    draft_model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    draft_k=args.draft_k,
                    rtt_ms=args.rtt_ms,
                    eos_token_id=tokenizer.eos_token_id,
                )

            matched = baseline.output_ids == spec.output_ids
            if not matched:
                first_diff = next(
                    (
                        i
                        for i, (left, right) in enumerate(
                            zip(baseline.output_ids, spec.output_ids)
                        )
                        if left != right
                    ),
                    min(len(baseline.output_ids), len(spec.output_ids)),
                )
                print(
                    json.dumps(
                        {
                            "error": "speculative_mismatch",
                            "model_pair": pair.name,
                            "implementation": args.implementation,
                            "prompt_id": prompt_id,
                            "first_diff_index": first_diff,
                            "baseline_len": len(baseline.output_ids),
                            "spec_len": len(spec.output_ids),
                            "baseline_output": tokenizer.decode(
                                baseline.output_ids,
                                skip_special_tokens=True,
                            ),
                            "spec_output": tokenizer.decode(
                                spec.output_ids,
                                skip_special_tokens=True,
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                raise AssertionError(
                    "Speculative output does not match target-only greedy output."
                )

            print(
                json.dumps(
                    {
                        "step": "greedy_spec_decode",
                        "model_pair": pair.name,
                        "implementation": args.implementation,
                        "prompt_id": prompt_id,
                        "rtt_ms": args.rtt_ms,
                        "max_new_tokens": args.max_new_tokens,
                        "draft_k": args.draft_k,
                        "generated_tokens": spec.generated_tokens,
                        "accepted_tokens": spec.accepted_tokens,
                        "drafted_tokens": spec.drafted_tokens,
                        "accept_rate": spec.accept_rate,
                        "target_only_time": baseline.timings.total_decode_time,
                        "spec_time": spec.timings.total_decode_time,
                        "speedup": baseline.timings.total_decode_time
                        / spec.timings.total_decode_time,
                        "timings": spec.timings.as_dict(),
                        "output": tokenizer.decode(spec.output_ids, skip_special_tokens=True),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
