from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_specdec.config import load_model_pairs, select_model_pairs
from edge_specdec.method_registry import RUNNERS, available_methods
from edge_specdec.models import choose_device, load_causal_lm, load_tokenizer
from edge_specdec.prompts import DEFAULT_PROMPTS


TIME_FIELDS = [
    "prefill_time",
    "draft_generate_time",
    "draft_structure_time",
    "upload_wait_time",
    "upload_latency_time",
    "upload_transfer_time",
    "upload_payload_bytes",
    "target_verify_time",
    "posterior_accept_time",
    "cache_update_time",
    "sampling_time",
    "wasted_branch_time_or_tokens",
    "total_decode_time",
]


FIELDNAMES = [
    "method_name",
    "implementation",
    "model_pair",
    "target",
    "draft",
    "prompt_id",
    "repeat",
    "rtt_ms",
    "upload_token_bytes",
    "upload_bandwidth_mbps",
    "max_new_tokens",
    "draft_k",
    "tree_width",
    "generated_tokens",
    "accepted_tokens",
    "drafted_tokens",
    "accept_rate",
    "rounds",
    "target_only_time",
    "method_time",
    "speedup_vs_target_only",
    "matched_target_only",
    "first_diff_index",
    "extra_json",
    *TIME_FIELDS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage timing experiments across multiple decoding methods."
    )
    parser.add_argument("--config", default="configs/model_pairs.yaml")
    parser.add_argument("--pair", action="append", help="Model pair name. Repeatable.")
    parser.add_argument(
        "--method",
        action="append",
        choices=available_methods(),
        help="Method to run. Repeatable. Defaults to all first-stage methods.",
    )
    parser.add_argument(
        "--implementation",
        default="full-prefix",
        choices=["full-prefix", "kv-cache"],
    )
    parser.add_argument("--prompt", action="append", help="Prompt text. Repeatable.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--rtt-ms", type=float, nargs="+", default=[0, 5, 10, 20, 50, 100])
    parser.add_argument(
        "--upload-token-bytes",
        type=int,
        default=4,
        help="Payload bytes per draft token id for upload simulation.",
    )
    parser.add_argument(
        "--upload-bandwidth-mbps",
        type=float,
        default=0.0,
        help=(
            "Optional upload bandwidth in Mbps. If <= 0, only fixed RTT is "
            "simulated, matching the original behavior."
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Optional Transformers attention backend. Use eager when SDPA "
            "KV-cache batched verification causes correctness mismatches."
        ),
    )
    parser.add_argument("--output", default="results/method_timing_raw.csv")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Record mismatches and continue instead of raising immediately.",
    )
    return parser.parse_args()


def first_diff(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return -1


def run_one(
    method_name: str,
    implementation: str,
    target_model,
    draft_model,
    input_ids,
    max_new_tokens: int,
    draft_k: int,
    rtt_ms: float,
    eos_token_id: int | None,
    tree_width: int,
    upload_token_bytes: int,
    upload_bandwidth_mbps: float,
):
    runner = RUNNERS[method_name]
    return runner(
        implementation,
        target_model,
        draft_model,
        input_ids,
        max_new_tokens,
        draft_k,
        rtt_ms,
        eos_token_id,
        tree_width,
        upload_token_bytes,
        upload_bandwidth_mbps,
    )


def main() -> None:
    args = parse_args()
    device = choose_device()
    prompts = args.prompt or DEFAULT_PROMPTS
    methods = args.method or available_methods()
    pairs = select_model_pairs(load_model_pairs(args.config), args.pair)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        mismatch_count = 0

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

            if args.warmups > 0:
                encoded = tokenizer(prompts[0], return_tensors="pt").to(device)
                for _ in range(args.warmups):
                    for method_name in methods:
                        run_one(
                            method_name,
                            args.implementation,
                            target_model,
                            draft_model,
                            encoded["input_ids"],
                            args.max_new_tokens,
                            args.draft_k,
                            0.0,
                            tokenizer.eos_token_id,
                            args.tree_width,
                            args.upload_token_bytes,
                            args.upload_bandwidth_mbps,
                        )

            work = [
                (prompt_id, prompt, repeat, rtt_ms, method_name)
                for prompt_id, prompt in enumerate(prompts)
                for repeat in range(args.repeats)
                for rtt_ms in args.rtt_ms
                for method_name in methods
            ]

            baseline_cache: dict[tuple[int, int], object] = {}
            for prompt_id, prompt, repeat, rtt_ms, method_name in tqdm(
                work,
                desc=f"method timing {pair.name}",
            ):
                encoded = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = encoded["input_ids"]
                baseline_key = (prompt_id, repeat)
                if baseline_key not in baseline_cache:
                    baseline_cache[baseline_key] = run_one(
                        "target-only",
                        args.implementation,
                        target_model,
                        draft_model,
                        input_ids,
                        args.max_new_tokens,
                        args.draft_k,
                        0.0,
                        tokenizer.eos_token_id,
                        args.tree_width,
                        args.upload_token_bytes,
                        args.upload_bandwidth_mbps,
                    )
                baseline = baseline_cache[baseline_key]

                if method_name == "target-only":
                    result = baseline
                else:
                    result = run_one(
                        method_name,
                        args.implementation,
                        target_model,
                        draft_model,
                        input_ids,
                        args.max_new_tokens,
                        args.draft_k,
                        rtt_ms,
                        tokenizer.eos_token_id,
                        args.tree_width,
                        args.upload_token_bytes,
                        args.upload_bandwidth_mbps,
                    )

                matched = baseline.output_ids == result.output_ids
                diff_index = first_diff(baseline.output_ids, result.output_ids)
                row = {
                    "method_name": method_name,
                    "implementation": args.implementation,
                    "model_pair": pair.name,
                    "target": pair.target,
                    "draft": pair.draft,
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "rtt_ms": rtt_ms,
                    "upload_token_bytes": args.upload_token_bytes,
                    "upload_bandwidth_mbps": args.upload_bandwidth_mbps,
                    "max_new_tokens": args.max_new_tokens,
                    "draft_k": args.draft_k,
                    "tree_width": args.tree_width,
                    "generated_tokens": result.generated_tokens,
                    "accepted_tokens": result.accepted_tokens,
                    "drafted_tokens": result.drafted_tokens,
                    "accept_rate": result.accept_rate,
                    "rounds": result.rounds,
                    "target_only_time": baseline.timings.total_decode_time,
                    "method_time": result.timings.total_decode_time,
                    "speedup_vs_target_only": baseline.timings.total_decode_time
                    / result.timings.total_decode_time,
                    "matched_target_only": matched,
                    "first_diff_index": diff_index,
                    "extra_json": json.dumps(result.extra, ensure_ascii=False),
                }
                timings = result.timings.as_dict()
                row.update({field: timings.get(field, 0.0) for field in TIME_FIELDS})
                writer.writerow(row)
                f.flush()

                if not matched:
                    mismatch_count += 1
                    baseline_text = tokenizer.decode(
                        baseline.output_ids,
                        skip_special_tokens=True,
                    )
                    method_text = tokenizer.decode(
                        result.output_ids,
                        skip_special_tokens=True,
                    )
                    message = (
                        "Method output mismatch. "
                        f"method={method_name}, prompt_id={prompt_id}, "
                        f"first_diff_index={diff_index}\n"
                        f"baseline={baseline_text}\nmethod={method_text}"
                    )
                    if args.allow_mismatch:
                        print(
                            json.dumps(
                                {
                                    "warning": "method_output_mismatch",
                                    "method": method_name,
                                    "prompt_id": prompt_id,
                                    "repeat": repeat,
                                    "rtt_ms": rtt_ms,
                                    "first_diff_index": diff_index,
                                    "baseline": baseline_text,
                                    "method_output": method_text,
                                },
                                ensure_ascii=False,
                            )
                        )
                        continue
                    raise AssertionError(message)

    print(f"Wrote raw method timing results to {output_path}")
    if mismatch_count:
        print(f"Recorded mismatches: {mismatch_count}")


if __name__ == "__main__":
    main()
