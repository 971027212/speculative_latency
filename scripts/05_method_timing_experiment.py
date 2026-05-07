from __future__ import annotations

import argparse
import csv
import json
import math
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
    "uplink_transfer_time",
    "network_wait_time",
    "downlink_transfer_time",
    "downlink_payload_bytes",
    "cloud_verify_time",
    "target_verify_time",
    "posterior_accept_time",
    "cache_update_time",
    "probability_normalize_time",
    "random_sample_time",
    "accept_reject_time",
    "resample_time",
    "sampling_time",
    "wasted_branch_time_or_tokens",
    "total_decode_time",
]


STOCHASTIC_METHODS = {"target-only-sampling", "traditional-spec-sampling"}

EXTRA_NUMERIC_FIELDS = [
    "rejected_tokens",
    "resample_count",
    "bonus_sample_count",
    "mean_checked_accept_prob",
    "mean_first_accept_prob",
    "target_zero_at_draft_count",
    "target_zero_at_draft_rate",
]


FIELDNAMES = [
    "method_name",
    "implementation",
    "target_verify_mode",
    "model_pair",
    "target",
    "draft",
    "prompt_id",
    "repeat",
    "rtt_ms",
    "upload_token_bytes",
    "upload_bandwidth_mbps",
    "uplink_bandwidth_mbps",
    "downlink_token_bytes",
    "downlink_fixed_bytes",
    "downlink_bandwidth_mbps",
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
    "requires_target_match",
    "validation_status",
    "stochastic_seed",
    "seed_strategy",
    "temperature",
    "top_k",
    "top_p",
    *EXTRA_NUMERIC_FIELDS,
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
    parser.add_argument(
        "--target-verify-mode",
        default="batch",
        choices=["sequential", "batch"],
        help=(
            "Target verification path for kv-cache speculative methods. "
            "batch mirrors feifeibear/LLMSpeculativeSampling's cached "
            "prob-history verification over the full draft span; sequential "
            "is retained for conservative greedy correctness checks."
        ),
    )
    parser.add_argument("--prompt", action="append", help="Prompt text. Repeatable.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--seed-strategy",
        default="per-repeat",
        choices=["fixed", "per-repeat"],
        help=(
            "Seed policy for stochastic methods. fixed reuses --seed for every "
            "repeat; per-repeat derives a deterministic seed from prompt and repeat "
            "while keeping the same seed across RTT values."
        ),
    )
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
            "Legacy upload bandwidth in Mbps. If <= 0, transfer time is not "
            "simulated, but fixed network wait from --rtt-ms still applies."
        ),
    )
    parser.add_argument(
        "--uplink-bandwidth-mbps",
        type=float,
        default=None,
        help=(
            "Optional uplink bandwidth in Mbps. Defaults to "
            "--upload-bandwidth-mbps for backward compatibility."
        ),
    )
    parser.add_argument(
        "--downlink-bandwidth-mbps",
        type=float,
        default=None,
        help="Optional downlink bandwidth in Mbps. Defaults to uplink bandwidth.",
    )
    parser.add_argument(
        "--downlink-token-bytes",
        type=int,
        default=4,
        help="Payload bytes for the response token returned by cloud verification.",
    )
    parser.add_argument(
        "--downlink-fixed-bytes",
        type=int,
        default=4,
        help="Fixed response payload bytes per verification round, e.g. accepted count.",
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
        "--ignore-eos",
        action="store_true",
        help=(
            "Ignore eos_token_id and force decoding to max_new_tokens. Useful "
            "when a model emits EOS immediately and would make timing trivial."
        ),
    )
    parser.add_argument(
        "--suppress-eos",
        action="store_true",
        help=(
            "Mask eos_token_id before argmax and force non-EOS greedy "
            "continuation to max_new_tokens. This avoids measuring a repeated "
            "EOS tail."
        ),
    )
    parser.add_argument(
        "--suppress-special-tokens",
        action="store_true",
        help=(
            "Mask all tokenizer special token ids before argmax and force "
            "non-special-token continuation."
        ),
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


def requires_target_match(method_name: str) -> bool:
    return method_name not in STOCHASTIC_METHODS


def baseline_method_for(method_name: str) -> str:
    return "target-only-sampling" if method_name in STOCHASTIC_METHODS else "target-only"


def seed_for_run(
    base_seed: int | None,
    strategy: str,
    prompt_id: int,
    repeat: int,
) -> int | None:
    if base_seed is None:
        return None
    if strategy == "fixed":
        return int(base_seed)
    if strategy != "per-repeat":
        raise ValueError(f"Unsupported seed strategy: {strategy}")
    return int(base_seed) + prompt_id * 1_000_003 + repeat * 10_007


def validate_result(result, max_new_tokens: int, timings: dict[str, float]) -> str:
    if result.generated_tokens < 0 or result.generated_tokens > max_new_tokens:
        return "bad_length"
    for field in TIME_FIELDS:
        value = float(timings.get(field, 0.0))
        if not math.isfinite(value):
            return f"non_finite_{field}"
    for field in [
        "generated_tokens",
        "accepted_tokens",
        "drafted_tokens",
        "rounds",
        "accept_rate",
    ]:
        value = float(getattr(result, field))
        if not math.isfinite(value):
            return f"non_finite_{field}"
    if result.drafted_tokens and not (0.0 <= result.accept_rate <= 1.0):
        return "bad_accept_rate"
    return "ok"


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
    uplink_bandwidth_mbps: float,
    downlink_token_bytes: int,
    downlink_fixed_bytes: int,
    downlink_bandwidth_mbps: float,
    target_verify_mode: str,
    suppress_token_id,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None,
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
        suppress_token_id,
        uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        downlink_token_bytes=downlink_token_bytes,
        downlink_fixed_bytes=downlink_fixed_bytes,
        downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        target_verify_mode=target_verify_mode,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    device = choose_device()
    prompts = args.prompt or DEFAULT_PROMPTS
    methods = args.method or available_methods()
    pairs = select_model_pairs(load_model_pairs(args.config), args.pair)
    uplink_bandwidth_mbps = (
        args.upload_bandwidth_mbps
        if args.uplink_bandwidth_mbps is None
        else args.uplink_bandwidth_mbps
    )
    downlink_bandwidth_mbps = (
        uplink_bandwidth_mbps
        if args.downlink_bandwidth_mbps is None
        else args.downlink_bandwidth_mbps
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        mismatch_count = 0

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
                            eos_token_id,
                            args.tree_width,
                            args.upload_token_bytes,
                            args.upload_bandwidth_mbps,
                            uplink_bandwidth_mbps,
                            args.downlink_token_bytes,
                            args.downlink_fixed_bytes,
                            downlink_bandwidth_mbps,
                            args.target_verify_mode,
                            suppress_token_id,
                            args.temperature,
                            args.top_k,
                            args.top_p,
                            args.seed,
                        )

            work = [
                (prompt_id, prompt, repeat, rtt_ms, method_name)
                for prompt_id, prompt in enumerate(prompts)
                for repeat in range(args.repeats)
                for rtt_ms in args.rtt_ms
                for method_name in methods
            ]

            baseline_cache: dict[tuple[str, int, int], object] = {}
            for prompt_id, prompt, repeat, rtt_ms, method_name in tqdm(
                work,
                desc=f"method timing {pair.name}",
            ):
                encoded = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = encoded["input_ids"]
                baseline_method = baseline_method_for(method_name)
                run_seed = seed_for_run(
                    args.seed,
                    args.seed_strategy,
                    prompt_id,
                    repeat,
                )
                baseline_key = (baseline_method, prompt_id, repeat)
                if baseline_key not in baseline_cache:
                    baseline_cache[baseline_key] = run_one(
                        baseline_method,
                        args.implementation,
                        target_model,
                        draft_model,
                        input_ids,
                        args.max_new_tokens,
                        args.draft_k,
                        0.0,
                        eos_token_id,
                        args.tree_width,
                        args.upload_token_bytes,
                        args.upload_bandwidth_mbps,
                        uplink_bandwidth_mbps,
                        args.downlink_token_bytes,
                        args.downlink_fixed_bytes,
                        downlink_bandwidth_mbps,
                        args.target_verify_mode,
                        suppress_token_id,
                        args.temperature,
                        args.top_k,
                        args.top_p,
                        run_seed,
                    )
                baseline = baseline_cache[baseline_key]

                if method_name == baseline_method:
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
                        eos_token_id,
                        args.tree_width,
                        args.upload_token_bytes,
                        args.upload_bandwidth_mbps,
                        uplink_bandwidth_mbps,
                        args.downlink_token_bytes,
                        args.downlink_fixed_bytes,
                        downlink_bandwidth_mbps,
                        args.target_verify_mode,
                        suppress_token_id,
                        args.temperature,
                        args.top_k,
                        args.top_p,
                        run_seed,
                    )

                matched = baseline.output_ids == result.output_ids
                diff_index = first_diff(baseline.output_ids, result.output_ids)
                timings = result.timings.as_dict()
                validation_status = validate_result(
                    result,
                    args.max_new_tokens,
                    timings,
                )
                hard_match_required = requires_target_match(method_name)
                row = {
                    "method_name": method_name,
                    "implementation": args.implementation,
                    "target_verify_mode": args.target_verify_mode,
                    "model_pair": pair.name,
                    "target": pair.target,
                    "draft": pair.draft,
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "rtt_ms": rtt_ms,
                    "upload_token_bytes": args.upload_token_bytes,
                    "upload_bandwidth_mbps": args.upload_bandwidth_mbps,
                    "uplink_bandwidth_mbps": uplink_bandwidth_mbps,
                    "downlink_token_bytes": args.downlink_token_bytes,
                    "downlink_fixed_bytes": args.downlink_fixed_bytes,
                    "downlink_bandwidth_mbps": downlink_bandwidth_mbps,
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
                    "requires_target_match": hard_match_required,
                    "validation_status": validation_status,
                    "stochastic_seed": run_seed if method_name in STOCHASTIC_METHODS else "",
                    "seed_strategy": args.seed_strategy
                    if method_name in STOCHASTIC_METHODS
                    else "",
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    **{
                        field: result.extra.get(field, 0.0)
                        for field in EXTRA_NUMERIC_FIELDS
                    },
                    "first_diff_index": diff_index,
                    "extra_json": json.dumps(result.extra, ensure_ascii=False),
                }
                row.update({field: timings.get(field, 0.0) for field in TIME_FIELDS})
                writer.writerow(row)
                f.flush()

                if validation_status != "ok":
                    raise AssertionError(
                        "Method validation failed. "
                        f"method={method_name}, prompt_id={prompt_id}, "
                        f"status={validation_status}"
                    )

                if hard_match_required and not matched:
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
