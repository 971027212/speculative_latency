from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TIME_COLUMNS = [
    "prefill_time",
    "draft_generate_time",
    "draft_structure_time",
    "upload_wait_time",
    "target_verify_time",
    "posterior_accept_time",
    "cache_update_time",
    "kv_or_input_update_time",
    "sampling_time",
    "wasted_branch_time_or_tokens",
    "total_decode_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: analyze RTT sweep results.")
    parser.add_argument("--input", default="results/rtt_sweep.csv")
    parser.add_argument("--output", default="results/rtt_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    if not df["matched_target_only"].all():
        bad = df.loc[~df["matched_target_only"], ["model_pair", "prompt_id", "rtt_ms"]]
        raise ValueError(f"Some speculative outputs mismatched target-only:\n{bad}")

    group_columns = ["model_pair", "rtt_ms"]
    if "implementation" in df.columns:
        group_columns.insert(1, "implementation")

    grouped = (
        df.groupby(group_columns, as_index=False)
        .agg(
            target_only_time=("target_only_time", "mean"),
            spec_time=("spec_time", "mean"),
            speedup=("speedup", "mean"),
            accept_rate=("accept_rate", "mean"),
            rounds=("rounds", "mean"),
            **{col: (col, "mean") for col in TIME_COLUMNS},
        )
        .sort_values(group_columns)
    )
    grouped["upload_wait_share"] = grouped["upload_wait_time"] / grouped["spec_time"]
    grouped["speedup_vs_mean_times"] = (
        grouped["target_only_time"] / grouped["spec_time"]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)

    print("\n=== RTT sweep summary ===")
    print(
        grouped[
            [
                *group_columns,
                "speedup",
                "accept_rate",
                "upload_wait_share",
                "target_only_time",
                "spec_time",
            ]
        ].to_string(index=False)
    )

    print("\n=== Break-even RTT ===")
    break_even_groups = ["model_pair"]
    if "implementation" in grouped.columns:
        break_even_groups.append("implementation")

    for key, sub in grouped.groupby(break_even_groups):
        label = " / ".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        below = sub[sub["speedup"] <= 1.0]
        if below.empty:
            print(f"{label}: not reached in tested RTT range")
        else:
            first = below.iloc[0]
            print(
                f"{label}: {first['rtt_ms']} ms "
                f"(mean speedup={first['speedup']:.3f})"
            )

    print(f"\nWrote summary to {output_path}")


if __name__ == "__main__":
    main()
