from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_specdec.config import load_model_pairs, select_model_pairs
from edge_specdec.decoding import speculative_greedy, target_only_greedy
from edge_specdec.models import choose_device, load_causal_lm, load_tokenizer
from edge_specdec.prompts import DEFAULT_PROMPTS


FIELDNAMES = [
    "model_pair",
    "target",
    "draft",
    "prompt_id",
    "repeat",
    "rtt_ms",
    "max_new_tokens",
    "draft_k",
    "generated_tokens",
    "accepted_tokens",
    "drafted_tokens",
    "accept_rate",
    "rounds",
    "target_only_time",
    "spec_time",
    "speedup",
    "prefill_time",
    "draft_generate_time",
    "upload_wait_time",
    "target_verify_time",
    "posterior_accept_time",
    "kv_or_input_update_time",
    "sampling_time",
    "total_decode_time",
    "matched_target_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3/4: sweep simulated RTT.")
    parser.add_argument("--config", default="configs/model_pairs.yaml")
    parser.add_argument("--pair", action="append", help="Model pair name. Repeatable.")
    parser.add_argument("--prompt", action="append", help="Prompt text. Repeatable.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--rtt-ms", type=float, nargs="+", default=[0, 5, 10, 20, 50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results/rtt_sweep.csv")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device()
    prompts = args.prompt or DEFAULT_PROMPTS
    pairs = select_model_pairs(load_model_pairs(args.config), args.pair)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

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

            work = [
                (prompt_id, prompt, repeat, rtt_ms)
                for prompt_id, prompt in enumerate(prompts)
                for repeat in range(args.repeats)
                for rtt_ms in args.rtt_ms
            ]

            baseline_cache: dict[tuple[int, int], object] = {}
            for prompt_id, prompt, repeat, rtt_ms in tqdm(work, desc=f"sweep {pair.name}"):
                encoded = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = encoded["input_ids"]
                cache_key = (prompt_id, repeat)

                if cache_key not in baseline_cache:
                    baseline_cache[cache_key] = target_only_greedy(
                        target_model,
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                baseline = baseline_cache[cache_key]

                spec = speculative_greedy(
                    target_model,
                    draft_model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    draft_k=args.draft_k,
                    rtt_ms=rtt_ms,
                    eos_token_id=tokenizer.eos_token_id,
                )

                matched = baseline.output_ids == spec.output_ids
                row = {
                    "model_pair": pair.name,
                    "target": pair.target,
                    "draft": pair.draft,
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "rtt_ms": rtt_ms,
                    "max_new_tokens": args.max_new_tokens,
                    "draft_k": args.draft_k,
                    "generated_tokens": spec.generated_tokens,
                    "accepted_tokens": spec.accepted_tokens,
                    "drafted_tokens": spec.drafted_tokens,
                    "accept_rate": spec.accept_rate,
                    "rounds": spec.rounds,
                    "target_only_time": baseline.timings.total_decode_time,
                    "spec_time": spec.timings.total_decode_time,
                    "speedup": baseline.timings.total_decode_time
                    / spec.timings.total_decode_time,
                    "matched_target_only": matched,
                }
                row.update(spec.timings.as_dict())
                writer.writerow(row)
                f.flush()

                if not matched:
                    raise AssertionError(
                        f"Speculative output mismatch for {pair.name}, prompt {prompt_id}"
                    )

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
