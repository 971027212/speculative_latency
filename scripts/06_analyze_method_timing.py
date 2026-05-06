from __future__ import annotations

import argparse
from datetime import datetime
from numbers import Number
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TIME_COLUMNS = [
    "prefill_time",
    "draft_generate_time",
    "draft_structure_time",
    "uplink_transfer_time",
    "network_wait_time",
    "cloud_verify_time",
    "downlink_transfer_time",
    "posterior_accept_time",
    "cache_update_time",
    "sampling_time",
]

NETWORK_COLUMNS = [
    "upload_wait_time",
    "upload_latency_time",
    "upload_transfer_time",
    "upload_payload_bytes",
    "uplink_transfer_time",
    "network_wait_time",
    "downlink_transfer_time",
    "downlink_payload_bytes",
]

LEGACY_TIME_COLUMNS = [
    "target_verify_time",
]

PARAM_COLUMNS = [
    "upload_token_bytes",
    "upload_bandwidth_mbps",
    "uplink_bandwidth_mbps",
    "downlink_token_bytes",
    "downlink_fixed_bytes",
    "downlink_bandwidth_mbps",
]

TIME_LABELS = {
    "prefill_time": "prefill",
    "draft_generate_time": "draft generation",
    "draft_structure_time": "draft structure",
    "uplink_transfer_time": "uplink transfer",
    "network_wait_time": "network wait",
    "cloud_verify_time": "cloud verification",
    "downlink_transfer_time": "downlink transfer",
    "posterior_accept_time": "posterior accept",
    "cache_update_time": "cache update",
    "sampling_time": "sampling / argmax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze method timing CSV and generate share tables/plots."
    )
    parser.add_argument("--input", default="results/method_timing_raw.csv")
    parser.add_argument("--summary-output", default="results/method_timing_summary.csv")
    parser.add_argument("--share-output", default="results/method_timing_stage_shares.csv")
    parser.add_argument("--upload-output", default="results/method_timing_upload_summary.csv")
    parser.add_argument(
        "--network-output",
        default="results/method_timing_network_cycle_summary.csv",
        help="Output CSV for edge-cloud cycle decomposition.",
    )
    parser.add_argument("--plot-output", default="results/method_timing_stage_shares.png")
    parser.add_argument("--plot-rtt-ms", type=float, default=0.0)
    parser.add_argument(
        "--markdown-output",
        default="reports/03_fine_grained_method_timing_analysis.md",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Print mismatch summary and analyze only matched rows.",
    )
    return parser.parse_args()


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _markdown_table(
    df: pd.DataFrame,
    columns: list[str],
    rename: dict[str, str] | None = None,
    digits: int = 4,
) -> str:
    rename = rename or {}
    headers = [rename.get(column, column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, Number):
                cells.append(_format_float(value, digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0, 1.0)


def _ensure_column(df: pd.DataFrame, column: str, default: float = 0.0) -> None:
    if column not in df.columns:
        df[column] = default


def _prepare_columns(df: pd.DataFrame) -> None:
    if "upload_transfer_time" not in df.columns:
        df["upload_transfer_time"] = 0.0
    if "upload_latency_time" not in df.columns:
        df["upload_latency_time"] = 0.0
    if "target_verify_time" not in df.columns:
        df["target_verify_time"] = 0.0

    if "uplink_transfer_time" not in df.columns:
        df["uplink_transfer_time"] = df["upload_transfer_time"]
    if "network_wait_time" not in df.columns:
        df["network_wait_time"] = df["upload_latency_time"]
    if "downlink_transfer_time" not in df.columns:
        df["downlink_transfer_time"] = 0.0
    if "cloud_verify_time" not in df.columns:
        df["cloud_verify_time"] = df["target_verify_time"]
    if "downlink_payload_bytes" not in df.columns:
        df["downlink_payload_bytes"] = 0.0
    if "upload_wait_time" not in df.columns:
        df["upload_wait_time"] = (
            df["uplink_transfer_time"]
            + df["network_wait_time"]
            + df["downlink_transfer_time"]
        )

    _ensure_column(df, "upload_token_bytes", 0.0)
    _ensure_column(df, "upload_bandwidth_mbps", 0.0)
    if "uplink_bandwidth_mbps" not in df.columns:
        df["uplink_bandwidth_mbps"] = df["upload_bandwidth_mbps"]
    if "downlink_bandwidth_mbps" not in df.columns:
        df["downlink_bandwidth_mbps"] = df["uplink_bandwidth_mbps"]
    _ensure_column(df, "downlink_token_bytes", 4.0)
    _ensure_column(df, "downlink_fixed_bytes", 4.0)
    for column in TIME_COLUMNS + NETWORK_COLUMNS + LEGACY_TIME_COLUMNS:
        _ensure_column(df, column, 0.0)


def _write_markdown_report(
    path: Path,
    input_path: str,
    summary: pd.DataFrame,
    shares: pd.DataFrame,
    upload_display: pd.DataFrame,
    network_display: pd.DataFrame,
    plot_rtt_ms: float,
    plot_output: str,
) -> None:
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stage_share = shares[shares["rtt_ms"] == plot_rtt_ms].copy()
    stage_share.insert(
        0,
        "method",
        stage_share["method_name"] + " / " + stage_share["implementation"],
    )
    percent_columns = []
    for column in TIME_COLUMNS:
        percent_column = f"{column}_percent"
        stage_share[percent_column] = stage_share[f"{column}_share"] * 100.0
        percent_columns.append(percent_column)

    lines = [
        "# Method Timing Network Decomposition",
        "",
        "This report decomposes each speculative edge-cloud verification cycle into "
        "uplink transfer, network wait, cloud verification, and downlink transfer.",
        "",
        f"- Raw input CSV: `{input_path}`",
        f"- Plot RTT: `{plot_rtt_ms}` ms",
        f"- Plot output: `{plot_output}`",
        f"- Report generated at: `{report_time}`",
        "",
        "## Method Summary",
        "",
        _markdown_table(
            summary,
            [
                "method_name",
                "implementation",
                "rtt_ms",
                "method_time",
                "speedup_vs_target_only",
                "accept_rate",
                "rounds",
                "generated_tokens",
                "drafted_tokens",
                "wasted_branch_time_or_tokens",
            ],
            rename={
                "method_name": "method",
                "rtt_ms": "RTT(ms)",
                "method_time": "time(s)",
                "speedup_vs_target_only": "speedup",
                "accept_rate": "accept rate",
                "generated_tokens": "generated",
                "drafted_tokens": "drafted",
                "wasted_branch_time_or_tokens": "wasted branch",
            },
        ),
        "",
        "## Edge-Cloud Cycle Summary",
        "",
        _markdown_table(
            network_display,
            [
                "method",
                "rtt_ms",
                "edge_cloud_cycle_time",
                "uplink_transfer_time",
                "network_wait_time",
                "cloud_verify_time",
                "downlink_transfer_time",
                "cloud_verify_share_of_cycle_percent",
                "network_cycle_share_of_method_time_percent",
            ],
            rename={
                "rtt_ms": "RTT(ms)",
                "edge_cloud_cycle_time": "cycle(s)",
                "uplink_transfer_time": "uplink(s)",
                "network_wait_time": "wait(s)",
                "cloud_verify_time": "cloud verify(s)",
                "downlink_transfer_time": "downlink(s)",
                "cloud_verify_share_of_cycle_percent": "cloud/cycle(%)",
                "network_cycle_share_of_method_time_percent": "cycle/method(%)",
            },
        ),
        "",
        "## Communication Summary",
        "",
        _markdown_table(
            upload_display,
            [
                "method",
                "rtt_ms",
                "upload_wait_time",
                "uplink_transfer_time",
                "network_wait_time",
                "downlink_transfer_time",
                "upload_payload_bytes",
                "downlink_payload_bytes",
                "upload_wait_share_percent",
                "upload_wait_per_round_ms",
            ],
            rename={
                "rtt_ms": "RTT(ms)",
                "upload_wait_time": "comm wait(s)",
                "uplink_transfer_time": "uplink(s)",
                "network_wait_time": "wait(s)",
                "downlink_transfer_time": "downlink(s)",
                "upload_payload_bytes": "upload bytes",
                "downlink_payload_bytes": "downlink bytes",
                "upload_wait_share_percent": "comm/method(%)",
                "upload_wait_per_round_ms": "comm/round(ms)",
            },
        ),
        "",
        f"## Stage Shares at RTT={plot_rtt_ms} ms",
        "",
        _markdown_table(
            stage_share,
            ["method"] + percent_columns,
            rename={
                f"{column}_percent": f"{TIME_LABELS[column]}(%)"
                for column in TIME_COLUMNS
            },
            digits=2,
        ),
        "",
        "## Notes",
        "",
        "- `rtt_ms` is modeled as network wait only; transfer and cloud compute are separate.",
        "- `cloud_verify_share_of_cycle_percent` is computed as cloud verification divided by "
        "uplink transfer + network wait + cloud verification + downlink transfer.",
        "- `target-only` has no speculative edge-cloud cycle, so cycle percentages are zero.",
        "- For Qwen2.5 1.5B/0.5B, prior runs showed no speedup at 32 tokens and draft-k=4; "
        "this decomposition is for attribution, not proof of acceleration.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    _prepare_columns(df)

    if not df["matched_target_only"].all():
        bad = df.loc[
            ~df["matched_target_only"],
            ["method_name", "model_pair", "prompt_id", "repeat", "rtt_ms", "first_diff_index"],
        ]
        if not args.allow_mismatch:
            raise ValueError(f"Some methods mismatched target-only:\n{bad}")
        print("\n=== Mismatch summary ===")
        print(bad.to_string(index=False))
        df = df[df["matched_target_only"]].copy()
        if df.empty:
            raise ValueError("No matched rows left to analyze.")

    group_columns = [
        "method_name",
        "implementation",
        "model_pair",
        "rtt_ms",
        *PARAM_COLUMNS,
    ]
    mean_columns = []
    for column in [
        "method_time",
        "target_only_time",
        "speedup_vs_target_only",
        "accept_rate",
        "rounds",
        "generated_tokens",
        "accepted_tokens",
        "drafted_tokens",
        "wasted_branch_time_or_tokens",
        *NETWORK_COLUMNS,
        *TIME_COLUMNS,
        *LEGACY_TIME_COLUMNS,
    ]:
        if column not in mean_columns:
            mean_columns.append(column)

    summary = (
        df.groupby(group_columns, as_index=False)
        .agg(**{column: (column, "mean") for column in mean_columns})
        .sort_values(group_columns)
    )
    summary["drafted_tokens_per_round"] = _safe_divide(
        summary["drafted_tokens"],
        summary["rounds"],
    )
    summary["accepted_tokens_per_round"] = _safe_divide(
        summary["accepted_tokens"],
        summary["rounds"],
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
    upload_path = Path(args.upload_output)
    network_path = Path(args.network_output)
    plot_path = Path(args.plot_output)
    markdown_path = Path(args.markdown_output) if args.markdown_output else None
    for path in [summary_path, share_path, upload_path, network_path, plot_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    shares.to_csv(share_path, index=False)

    plot_df = summary[summary["rtt_ms"] == args.plot_rtt_ms].copy()
    if plot_df.empty:
        raise ValueError(f"No rows found for plot RTT {args.plot_rtt_ms}")

    labels = plot_df["method_name"] + "\n" + plot_df["implementation"]
    bottoms = [0.0] * len(plot_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in TIME_COLUMNS:
        values = _safe_divide(plot_df[column], plot_df["method_time"])
        ax.bar(labels, values, bottom=bottoms, label=TIME_LABELS[column])
        bottoms = [left + right for left, right in zip(bottoms, values)]

    ax.set_ylabel("Share of method time")
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
                "generated_tokens",
                "rounds",
                "drafted_tokens",
                "wasted_branch_time_or_tokens",
            ]
        ].to_string(index=False)
    )

    share_display = shares[shares["rtt_ms"] == args.plot_rtt_ms].copy()
    share_display.insert(
        1,
        "method",
        share_display["method_name"] + " / " + share_display["implementation"],
    )
    share_display = share_display[
        ["method"] + [f"{column}_share" for column in TIME_COLUMNS]
    ]
    share_display = share_display.rename(
        columns={f"{column}_share": TIME_LABELS[column] for column in TIME_COLUMNS}
    )
    for column in TIME_COLUMNS:
        share_display[TIME_LABELS[column]] = share_display[TIME_LABELS[column]] * 100.0

    print(f"\n=== Stage time shares at RTT={args.plot_rtt_ms} ms (%) ===")
    print(
        share_display.to_string(
            index=False,
            formatters={
                TIME_LABELS[column]: "{:.2f}".format for column in TIME_COLUMNS
            },
        )
    )

    upload_display = summary[
        [
            "method_name",
            "implementation",
            "rtt_ms",
            "method_time",
            "rounds",
            "generated_tokens",
            "upload_wait_time",
            "upload_latency_time",
            "upload_transfer_time",
            "upload_payload_bytes",
            "uplink_transfer_time",
            "network_wait_time",
            "downlink_transfer_time",
            "downlink_payload_bytes",
            *PARAM_COLUMNS,
            "speedup_vs_target_only",
        ]
    ].copy()
    upload_display.insert(
        1,
        "method",
        upload_display["method_name"] + " / " + upload_display["implementation"],
    )
    upload_display["upload_wait_share_percent"] = (
        _safe_divide(upload_display["upload_wait_time"], upload_display["method_time"])
        * 100.0
    )
    upload_display["upload_wait_per_round_ms"] = (
        _safe_divide(upload_display["upload_wait_time"], upload_display["rounds"])
        * 1000.0
    )
    upload_display["upload_wait_per_generated_token_ms"] = (
        _safe_divide(upload_display["upload_wait_time"], upload_display["generated_tokens"])
        * 1000.0
    )
    upload_display = upload_display[
        [
            "method",
            "rtt_ms",
            "method_time",
            "upload_wait_time",
            "upload_latency_time",
            "upload_transfer_time",
            "upload_payload_bytes",
            "uplink_transfer_time",
            "network_wait_time",
            "downlink_transfer_time",
            "downlink_payload_bytes",
            *PARAM_COLUMNS,
            "upload_wait_share_percent",
            "upload_wait_per_round_ms",
            "upload_wait_per_generated_token_ms",
            "speedup_vs_target_only",
        ]
    ]
    upload_display.to_csv(upload_path, index=False)

    network_display = summary[
        [
            "method_name",
            "implementation",
            "rtt_ms",
            "method_time",
            "rounds",
            "uplink_transfer_time",
            "network_wait_time",
            "cloud_verify_time",
            "downlink_transfer_time",
            "upload_payload_bytes",
            "downlink_payload_bytes",
            "speedup_vs_target_only",
        ]
    ].copy()
    network_display.insert(
        1,
        "method",
        network_display["method_name"] + " / " + network_display["implementation"],
    )
    has_cycle = network_display["rounds"] > 0
    network_display["edge_cloud_cycle_time"] = 0.0
    network_display.loc[has_cycle, "edge_cloud_cycle_time"] = (
        network_display.loc[has_cycle, "uplink_transfer_time"]
        + network_display.loc[has_cycle, "network_wait_time"]
        + network_display.loc[has_cycle, "cloud_verify_time"]
        + network_display.loc[has_cycle, "downlink_transfer_time"]
    )
    network_display["network_only_time"] = 0.0
    network_display.loc[has_cycle, "network_only_time"] = (
        network_display.loc[has_cycle, "uplink_transfer_time"]
        + network_display.loc[has_cycle, "network_wait_time"]
        + network_display.loc[has_cycle, "downlink_transfer_time"]
    )
    cycle_denominator = network_display["edge_cloud_cycle_time"]
    network_display["cloud_verify_share_of_cycle_percent"] = (
        _safe_divide(network_display["cloud_verify_time"], cycle_denominator) * 100.0
    ).where(has_cycle, 0.0)
    network_display["uplink_share_of_cycle_percent"] = (
        _safe_divide(network_display["uplink_transfer_time"], cycle_denominator) * 100.0
    ).where(has_cycle, 0.0)
    network_display["wait_share_of_cycle_percent"] = (
        _safe_divide(network_display["network_wait_time"], cycle_denominator) * 100.0
    ).where(has_cycle, 0.0)
    network_display["downlink_share_of_cycle_percent"] = (
        _safe_divide(network_display["downlink_transfer_time"], cycle_denominator) * 100.0
    ).where(has_cycle, 0.0)
    network_display["network_cycle_share_of_method_time_percent"] = (
        _safe_divide(network_display["edge_cloud_cycle_time"], network_display["method_time"])
        * 100.0
    ).where(has_cycle, 0.0)
    network_display["cloud_verify_per_round_ms"] = (
        _safe_divide(network_display["cloud_verify_time"], network_display["rounds"])
        * 1000.0
    ).where(has_cycle, 0.0)
    network_display["network_wait_per_round_ms"] = (
        _safe_divide(network_display["network_wait_time"], network_display["rounds"])
        * 1000.0
    ).where(has_cycle, 0.0)
    network_display = network_display[
        [
            "method",
            "rtt_ms",
            "method_time",
            "rounds",
            "uplink_transfer_time",
            "network_wait_time",
            "cloud_verify_time",
            "downlink_transfer_time",
            "edge_cloud_cycle_time",
            "network_only_time",
            "cloud_verify_share_of_cycle_percent",
            "uplink_share_of_cycle_percent",
            "wait_share_of_cycle_percent",
            "downlink_share_of_cycle_percent",
            "network_cycle_share_of_method_time_percent",
            "cloud_verify_per_round_ms",
            "network_wait_per_round_ms",
            "upload_payload_bytes",
            "downlink_payload_bytes",
            "speedup_vs_target_only",
        ]
    ]
    network_display.to_csv(network_path, index=False)

    print("\n=== Communication wait summary ===")
    print(
        upload_display.to_string(
            index=False,
            formatters={
                "rtt_ms": "{:.1f}".format,
                "method_time": "{:.4f}".format,
                "upload_wait_time": "{:.4f}".format,
                "upload_latency_time": "{:.4f}".format,
                "upload_transfer_time": "{:.6f}".format,
                "upload_payload_bytes": "{:.0f}".format,
                "uplink_transfer_time": "{:.6f}".format,
                "network_wait_time": "{:.4f}".format,
                "downlink_transfer_time": "{:.6f}".format,
                "downlink_payload_bytes": "{:.0f}".format,
                "upload_token_bytes": "{:.0f}".format,
                "upload_bandwidth_mbps": "{:.3f}".format,
                "uplink_bandwidth_mbps": "{:.3f}".format,
                "downlink_token_bytes": "{:.0f}".format,
                "downlink_fixed_bytes": "{:.0f}".format,
                "downlink_bandwidth_mbps": "{:.3f}".format,
                "upload_wait_share_percent": "{:.2f}".format,
                "upload_wait_per_round_ms": "{:.2f}".format,
                "upload_wait_per_generated_token_ms": "{:.2f}".format,
                "speedup_vs_target_only": "{:.4f}".format,
            },
        )
    )

    print("\n=== Edge-cloud cycle summary ===")
    print(
        network_display.to_string(
            index=False,
            formatters={
                "rtt_ms": "{:.1f}".format,
                "method_time": "{:.4f}".format,
                "rounds": "{:.1f}".format,
                "uplink_transfer_time": "{:.6f}".format,
                "network_wait_time": "{:.4f}".format,
                "cloud_verify_time": "{:.4f}".format,
                "downlink_transfer_time": "{:.6f}".format,
                "edge_cloud_cycle_time": "{:.4f}".format,
                "network_only_time": "{:.4f}".format,
                "cloud_verify_share_of_cycle_percent": "{:.2f}".format,
                "uplink_share_of_cycle_percent": "{:.2f}".format,
                "wait_share_of_cycle_percent": "{:.2f}".format,
                "downlink_share_of_cycle_percent": "{:.2f}".format,
                "network_cycle_share_of_method_time_percent": "{:.2f}".format,
                "cloud_verify_per_round_ms": "{:.2f}".format,
                "network_wait_per_round_ms": "{:.2f}".format,
                "upload_payload_bytes": "{:.0f}".format,
                "downlink_payload_bytes": "{:.0f}".format,
                "speedup_vs_target_only": "{:.4f}".format,
            },
        )
    )

    if markdown_path is not None:
        _write_markdown_report(
            markdown_path,
            args.input,
            summary,
            shares,
            upload_display,
            network_display,
            args.plot_rtt_ms,
            args.plot_output,
        )

    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote stage shares to {share_path}")
    print(f"Wrote communication summary to {upload_path}")
    print(f"Wrote edge-cloud cycle summary to {network_path}")
    if markdown_path is not None:
        print(f"Wrote markdown report to {markdown_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
