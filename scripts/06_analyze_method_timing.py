from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TIME_COLUMNS = [
    "prefill_time",
    "draft_generate_time",
    "draft_structure_time",
    "upload_wait_time",
    "target_verify_time",
    "posterior_accept_time",
    "cache_update_time",
    "sampling_time",
]


TIME_LABELS = {
    "prefill_time": "prefill（预填充）",
    "draft_generate_time": "draft（草稿生成）",
    "draft_structure_time": "structure（结构构建）",
    "upload_wait_time": "upload（上传等待）",
    "target_verify_time": "verify（目标验证）",
    "posterior_accept_time": "accept（后验接受）",
    "cache_update_time": "cache（缓存更新）",
    "sampling_time": "sampling（采样/argmax）",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze method timing CSV and generate share tables/plots."
    )
    parser.add_argument("--input", default="results/method_timing_raw.csv")
    parser.add_argument("--summary-output", default="results/method_timing_summary.csv")
    parser.add_argument("--share-output", default="results/method_timing_stage_shares.csv")
    parser.add_argument("--plot-output", default="results/method_timing_stage_shares.png")
    parser.add_argument("--plot-rtt-ms", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if not df["matched_target_only"].all():
        bad = df.loc[
            ~df["matched_target_only"],
            ["method_name", "model_pair", "prompt_id", "repeat", "rtt_ms", "first_diff_index"],
        ]
        raise ValueError(f"Some methods mismatched target-only:\n{bad}")

    group_columns = ["method_name", "implementation", "model_pair", "rtt_ms"]
    summary = (
        df.groupby(group_columns, as_index=False)
        .agg(
            method_time=("method_time", "mean"),
            target_only_time=("target_only_time", "mean"),
            speedup_vs_target_only=("speedup_vs_target_only", "mean"),
            accept_rate=("accept_rate", "mean"),
            rounds=("rounds", "mean"),
            wasted_branch_time_or_tokens=("wasted_branch_time_or_tokens", "mean"),
            **{column: (column, "mean") for column in TIME_COLUMNS},
        )
        .sort_values(group_columns)
    )

    share_rows = []
    for _, row in summary.iterrows():
        base = {
            "method_name": row["method_name"],
            "implementation": row["implementation"],
            "model_pair": row["model_pair"],
            "rtt_ms": row["rtt_ms"],
            "method_time": row["method_time"],
            "speedup_vs_target_only": row["speedup_vs_target_only"],
        }
        denominator = row["method_time"]
        for column in TIME_COLUMNS:
            base[f"{column}_share"] = row[column] / denominator if denominator else 0.0
        share_rows.append(base)
    shares = pd.DataFrame(share_rows)

    summary_path = Path(args.summary_output)
    share_path = Path(args.share_output)
    plot_path = Path(args.plot_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    share_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    shares.to_csv(share_path, index=False)

    plot_df = summary[summary["rtt_ms"] == args.plot_rtt_ms].copy()
    if plot_df.empty:
        raise ValueError(f"No rows found for plot RTT {args.plot_rtt_ms}")

    labels = plot_df["method_name"] + "\n" + plot_df["implementation"]
    bottoms = [0.0] * len(plot_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in TIME_COLUMNS:
        values = (
            plot_df[column] / plot_df["method_time"].where(plot_df["method_time"] != 0, 1.0)
        )
        ax.bar(labels, values, bottom=bottoms, label=TIME_LABELS[column])
        bottoms = [left + right for left, right in zip(bottoms, values)]

    ax.set_ylabel("Share of method time（阶段时间占比）")
    ax.set_title(f"Stage time shares at RTT={args.plot_rtt_ms} ms")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\n=== Method timing summary ===")
    print(
        summary[
            [
                "method_name",
                "implementation",
                "model_pair",
                "rtt_ms",
                "method_time",
                "speedup_vs_target_only",
                "accept_rate",
                "wasted_branch_time_or_tokens",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote stage shares to {share_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
