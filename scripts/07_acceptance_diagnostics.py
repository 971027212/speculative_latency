from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_specdec.config import load_model_pairs, select_model_pairs
from edge_specdec.decoding import (
    target_only_greedy,
    target_only_greedy_cached,
)
from edge_specdec.method_registry import RUNNERS
from edge_specdec.models import choose_device, load_causal_lm, load_tokenizer
from edge_specdec.prompts import DEFAULT_PROMPTS


FIELDNAMES = [
    "model_pair",
    "prompt_id",
    "prompt",
    "method_name",
    "matched_target_only",
    "first_diff_index",
    "generated_tokens",
    "accepted_tokens",
    "drafted_tokens",
    "accept_rate",
    "rounds",
    "unique_generated_token_count",
    "eos_token_count",
    "generated_token_ids",
    "generated_text",
    "generated_text_repr",
    "generated_text_with_specials_repr",
    "full_text",
    "extra_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-prompt acceptance and generated-text diagnostics."
    )
    parser.add_argument("--config", default="configs/model_pairs.yaml")
    parser.add_argument("--pair", action="append", help="Model pair name. Repeatable.")
    parser.add_argument(
        "--method",
        action="append",
        choices=["vanilla-spec", "dsd-adaptive-draft"],
        help="Speculative method to inspect. Defaults to both.",
    )
    parser.add_argument(
        "--implementation",
        default="kv-cache",
        choices=["full-prefix", "kv-cache"],
    )
    parser.add_argument("--prompt", action="append", help="Prompt text. Repeatable.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--suppress-eos",
        action="store_true",
        help="Mask eos_token_id before argmax and force non-EOS continuation.",
    )
    parser.add_argument(
        "--suppress-special-tokens",
        action="store_true",
        help="Mask all tokenizer special token ids before argmax.",
    )
    parser.add_argument(
        "--suppress-token-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Additional token id to mask before argmax. Repeatable. Useful "
            "for tokenizer-specific padding/control tokens not listed in "
            "all_special_ids."
        ),
    )
    parser.add_argument("--output", default="results/acceptance_diagnostics.csv")
    return parser.parse_args()


def first_diff(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return -1


def run_target_only(
    implementation: str,
    model,
    input_ids,
    max_new_tokens: int,
    eos_token_id: int | None,
    suppress_token_id,
):
    if implementation == "kv-cache":
        return target_only_greedy_cached(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            suppress_token_id=suppress_token_id,
        )
    return target_only_greedy(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        suppress_token_id=suppress_token_id,
    )


def make_row(
    tokenizer,
    pair_name: str,
    prompt_id: int,
    prompt: str,
    method_name: str,
    result,
    baseline_output_ids: list[int],
    prompt_len: int,
    eos_token_id: int | None,
) -> dict[str, object]:
    generated_ids = result.output_ids[prompt_len:]
    eos_count = 0
    if eos_token_id is not None:
        eos_count = sum(1 for token_id in generated_ids if token_id == eos_token_id)
    return {
        "model_pair": pair_name,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "method_name": method_name,
        "matched_target_only": result.output_ids == baseline_output_ids,
        "first_diff_index": first_diff(baseline_output_ids, result.output_ids),
        "generated_tokens": result.generated_tokens,
        "accepted_tokens": result.accepted_tokens,
        "drafted_tokens": result.drafted_tokens,
        "accept_rate": result.accept_rate,
        "rounds": result.rounds,
        "unique_generated_token_count": len(set(generated_ids)),
        "eos_token_count": eos_count,
        "generated_token_ids": " ".join(str(token_id) for token_id in generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_text_repr": repr(
            tokenizer.decode(generated_ids, skip_special_tokens=True)
        ),
        "generated_text_with_specials_repr": repr(
            tokenizer.decode(generated_ids, skip_special_tokens=False)
        ),
        "full_text": tokenizer.decode(result.output_ids, skip_special_tokens=True),
        "extra_json": json.dumps(result.extra, ensure_ascii=False),
    }


def main() -> None:
    args = parse_args()
    device = choose_device()
    prompts = args.prompt or DEFAULT_PROMPTS
    methods = args.method or ["vanilla-spec", "dsd-adaptive-draft"]
    pairs = select_model_pairs(load_model_pairs(args.config), args.pair)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for pair in pairs:
        tokenizer = load_tokenizer(pair.target, trust_remote_code=args.trust_remote_code)
        suppress_ids = set(args.suppress_token_id or [])
        if args.suppress_special_tokens:
            suppress_ids.update(int(token_id) for token_id in tokenizer.all_special_ids)
            if tokenizer.pad_token_id is not None:
                suppress_ids.add(int(tokenizer.pad_token_id))
        if args.suppress_eos and tokenizer.eos_token_id is not None:
            suppress_ids.add(int(tokenizer.eos_token_id))
        eos_token_id = (
            None
            if (
                args.ignore_eos
                or args.suppress_eos
                or args.suppress_special_tokens
                or (
                    tokenizer.eos_token_id is not None
                    and int(tokenizer.eos_token_id) in suppress_ids
                )
            )
            else tokenizer.eos_token_id
        )
        suppress_token_id = sorted(suppress_ids) if suppress_ids else None
        print(
            json.dumps(
                {
                    "model_pair": pair.name,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "all_special_ids": tokenizer.all_special_ids,
                    "suppress_token_id": suppress_token_id,
                },
                ensure_ascii=False,
            )
        )

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

        for prompt_id, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = encoded["input_ids"]
            prompt_len = input_ids.shape[-1]

            target_result = run_target_only(
                args.implementation,
                target_model,
                input_ids,
                args.max_new_tokens,
                eos_token_id,
                suppress_token_id,
            )
            draft_result = run_target_only(
                args.implementation,
                draft_model,
                input_ids,
                args.max_new_tokens,
                eos_token_id,
                suppress_token_id,
            )

            rows.append(
                make_row(
                    tokenizer,
                    pair.name,
                    prompt_id,
                    prompt,
                    "target-only",
                    target_result,
                    target_result.output_ids,
                    prompt_len,
                    tokenizer.eos_token_id,
                )
            )
            rows.append(
                make_row(
                    tokenizer,
                    pair.name,
                    prompt_id,
                    prompt,
                    "draft-only",
                    draft_result,
                    target_result.output_ids,
                    prompt_len,
                    tokenizer.eos_token_id,
                )
            )

            for method_name in methods:
                runner = RUNNERS[method_name]
                result = runner(
                    args.implementation,
                    target_model,
                    draft_model,
                    input_ids,
                    args.max_new_tokens,
                    args.draft_k,
                    0.0,
                    eos_token_id,
                    args.tree_width,
                    4,
                    0.0,
                    suppress_token_id,
                )
                rows.append(
                    make_row(
                        tokenizer,
                        pair.name,
                        prompt_id,
                        prompt,
                        method_name,
                        result,
                        target_result.output_ids,
                        prompt_len,
                        tokenizer.eos_token_id,
                    )
                )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote diagnostics to {output_path}")
    print("\n=== Acceptance diagnostics ===")
    display_columns = [
        "prompt_id",
        "method_name",
        "matched_target_only",
        "accept_rate",
        "generated_tokens",
        "rounds",
        "drafted_tokens",
        "unique_generated_token_count",
        "eos_token_count",
        "generated_token_ids",
        "generated_text_repr",
        "generated_text_with_specials_repr",
        "generated_text",
    ]
    for row in rows:
        display = {column: row[column] for column in display_columns}
        print(json.dumps(display, ensure_ascii=False))


if __name__ == "__main__":
    main()
