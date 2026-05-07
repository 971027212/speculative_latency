from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

pd = None


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
    "probability_normalize_time",
    "random_sample_time",
    "accept_reject_time",
    "resample_time",
    "sampling_time",
]

TIME_LABELS = {
    "prefill_time": "prefill",
    "draft_generate_time": "draft_generation",
    "draft_structure_time": "draft_structure",
    "uplink_transfer_time": "uplink_transfer",
    "network_wait_time": "network_wait",
    "cloud_verify_time": "cloud_verification",
    "downlink_transfer_time": "downlink_transfer",
    "posterior_accept_time": "posterior_accept",
    "cache_update_time": "cache_update",
    "probability_normalize_time": "probability_normalize",
    "random_sample_time": "random_sample",
    "accept_reject_time": "accept_reject",
    "resample_time": "resample",
    "sampling_time": "sampling_argmax",
}

DEFAULT_DATASETS = [
    (
        "Qwen2.5-1.5B",
        "results/method_timing_summary_qwen25_1p5b_kvcache_eager_stochastic_repeat3.csv",
        "results/method_timing_stage_shares_qwen25_1p5b_kvcache_eager_stochastic_repeat3.csv",
        "results/method_timing_network_cycle_qwen25_1p5b_kvcache_eager_stochastic_repeat3.csv",
    ),
]

EXTRA_NUMERIC_COLUMNS = [
    "rejected_tokens",
    "resample_count",
    "bonus_sample_count",
    "mean_checked_accept_prob",
    "mean_first_accept_prob",
    "first_token_accept_rate",
    "target_zero_at_draft_count",
    "target_zero_at_draft_rate",
]


def _require_pandas():
    global pd
    if pd is None:
        import pandas as pandas_module

        pd = pandas_module
    return pd


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    summary_path: Path
    share_path: Path
    network_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect analyzed method timing CSVs into group-meeting summary tables."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL=SUMMARY_CSV,SHARE_CSV,NETWORK_CSV",
        help=(
            "Analyzed dataset to collect. Repeatable. Defaults to the planned "
            "SmolLM and Qwen2.5-1.5B repeat3 output paths."
        ),
    )
    parser.add_argument(
        "--stage-rtt-ms",
        type=float,
        default=100.0,
        help="RTT slice used for the stage-share table.",
    )
    parser.add_argument(
        "--method-summary-output",
        default="results/group_meeting_method_summary.csv",
    )
    parser.add_argument(
        "--stage-share-output",
        default="results/group_meeting_stage_shares_rtt100.csv",
    )
    parser.add_argument(
        "--network-output",
        default="results/group_meeting_network_cycle.csv",
    )
    parser.add_argument(
        "--markdown-output",
        default="results/group_meeting_key_points.md",
    )
    return parser.parse_args()


def _parse_dataset(value: str) -> DatasetSpec:
    if "=" not in value:
        raise ValueError(
            "Dataset must use LABEL=SUMMARY_CSV,SHARE_CSV,NETWORK_CSV format."
        )
    label, paths = value.split("=", 1)
    parts = [part.strip() for part in paths.split(",")]
    if len(parts) != 3 or not label.strip():
        raise ValueError(
            "Dataset must use LABEL=SUMMARY_CSV,SHARE_CSV,NETWORK_CSV format."
        )
    return DatasetSpec(
        label=label.strip(),
        summary_path=Path(parts[0]),
        share_path=Path(parts[1]),
        network_path=Path(parts[2]),
    )


def _default_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(label, Path(summary), Path(share), Path(network))
        for label, summary, share, network in DEFAULT_DATASETS
    ]


def _resolve_specs(values: list[str]) -> list[DatasetSpec]:
    return [_parse_dataset(value) for value in values] if values else _default_specs()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}. Run scripts/06_analyze_method_timing.py "
            "for this dataset first, or pass --dataset with custom paths."
        )


def _load_csv(path: Path) -> pd.DataFrame:
    _require_file(path)
    return pd.read_csv(path)


def _insert_group(df: pd.DataFrame, label: str) -> pd.DataFrame:
    result = df.copy()
    if "group" in result.columns:
        result = result.drop(columns=["group"])
    result.insert(0, "group", label)
    return result


def _column_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([0.0] * len(df), index=df.index)


def _prepare_stage_shares(shares: pd.DataFrame) -> pd.DataFrame:
    result = shares.copy()
    for column in TIME_COLUMNS:
        share_column = f"{column}_share"
        percent_column = f"{TIME_LABELS[column]}_percent"
        legacy_percent_column = f"{column}_percent"
        if legacy_percent_column in result.columns:
            result[percent_column] = result[legacy_percent_column]
        else:
            result[percent_column] = _column_or_zero(result, share_column) * 100.0
    return result


def _filter_rtt(df: pd.DataFrame, rtt_ms: float, label: str, path: Path) -> pd.DataFrame:
    if "rtt_ms" not in df.columns:
        raise ValueError(f"{path} has no rtt_ms column.")
    selected = df[(df["rtt_ms"] - rtt_ms).abs() < 1e-9].copy()
    if selected.empty:
        raise ValueError(f"{path} has no rows for RTT={rtt_ms} ms ({label}).")
    return selected


def _method_label(df: pd.DataFrame) -> pd.Series:
    if "method" in df.columns:
        return df["method"]
    if {"method_name", "implementation"}.issubset(df.columns):
        return df["method_name"] + " / " + df["implementation"]
    return pd.Series([""] * len(df), index=df.index)


def _collect_method_summary(specs: list[DatasetSpec]) -> pd.DataFrame:
    frames = []
    columns = [
        "group",
        "model_pair",
        "method_name",
        "implementation",
        "target_verify_mode",
        "temperature",
        "top_k",
        "top_p",
        "sampling_filter",
        "stochastic_seed",
        "seed_strategy",
        "rtt_ms",
        "method_time",
        "target_only_time",
        "speedup_vs_target_only",
        "accept_rate",
        "rounds",
        "generated_tokens",
        "accepted_tokens",
        "drafted_tokens",
        "drafted_tokens_per_round",
        "accepted_tokens_per_round",
        *EXTRA_NUMERIC_COLUMNS,
    ]
    for spec in specs:
        summary = _insert_group(_load_csv(spec.summary_path), spec.label)
        for column in columns:
            if column not in summary.columns:
                summary[column] = (
                    "legacy"
                    if column
                    in {
                        "target_verify_mode",
                        "sampling_filter",
                        "stochastic_seed",
                        "seed_strategy",
                    }
                    else 0.0
                )
        frames.append(summary[columns])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["group", "method_name", "rtt_ms"]
    )


def _collect_stage_shares(specs: list[DatasetSpec], stage_rtt_ms: float) -> pd.DataFrame:
    frames = []
    percent_columns = [
        f"{TIME_LABELS[column]}_percent" for column in TIME_COLUMNS
    ]
    columns = [
        "group",
        "method",
        "model_pair",
        "method_name",
        "implementation",
        "target_verify_mode",
        "temperature",
        "top_k",
        "top_p",
        "sampling_filter",
        "stochastic_seed",
        "seed_strategy",
        "rtt_ms",
        "method_time",
        "speedup_vs_target_only",
        *percent_columns,
    ]
    for spec in specs:
        shares = _prepare_stage_shares(_load_csv(spec.share_path))
        shares = _filter_rtt(shares, stage_rtt_ms, spec.label, spec.share_path)
        shares = _insert_group(shares, spec.label)
        shares["method"] = _method_label(shares)
        for column in columns:
            if column not in shares.columns:
                shares[column] = (
                    "legacy"
                    if column
                    in {
                        "target_verify_mode",
                        "sampling_filter",
                        "stochastic_seed",
                        "seed_strategy",
                    }
                    else 0.0
                )
        frames.append(shares[columns])
    return pd.concat(frames, ignore_index=True).sort_values(["group", "method"])


def _collect_network(specs: list[DatasetSpec]) -> pd.DataFrame:
    frames = []
    columns = [
        "group",
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
    for spec in specs:
        network = _insert_group(_load_csv(spec.network_path), spec.label)
        network["method"] = _method_label(network)
        for column in columns:
            if column not in network.columns:
                network[column] = 0.0
        frames.append(network[columns])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["group", "method", "rtt_ms"]
    )


def _format_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df[columns].iterrows():
        cells = [_format_float(row[column], digits) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _break_even_notes(summary: pd.DataFrame) -> list[str]:
    notes = []
    for (group, method), rows in summary.groupby(["group", "method_name"]):
        if method == "target-only":
            continue
        rows = rows.sort_values("rtt_ms")
        zero_rows = rows[(rows["rtt_ms"] - 0.0).abs() < 1e-9]
        if zero_rows.empty:
            notes.append(f"- {group} / {method}: no RTT=0 row.")
            continue
        zero_speedup = float(zero_rows.iloc[0]["speedup_vs_target_only"])
        if zero_speedup <= 1.0:
            notes.append(
                f"- {group} / {method}: no positive break-even; "
                f"RTT=0 speedup={zero_speedup:.4f}x."
            )
            continue
        break_even = None
        previous = None
        for _, row in rows.iterrows():
            current = (float(row["rtt_ms"]), float(row["speedup_vs_target_only"]))
            if previous is not None and previous[1] >= 1.0 >= current[1]:
                left_rtt, left_speed = previous
                right_rtt, right_speed = current
                if left_speed == right_speed:
                    break_even = right_rtt
                else:
                    ratio = (1.0 - left_speed) / (right_speed - left_speed)
                    break_even = left_rtt + ratio * (right_rtt - left_rtt)
                break
            previous = current
        if break_even is None:
            notes.append(
                f"- {group} / {method}: speedup remains above 1.0x in sampled RTTs; "
                f"RTT=0 speedup={zero_speedup:.4f}x."
            )
        else:
            notes.append(
                f"- {group} / {method}: estimated break-even RTT={break_even:.2f} ms; "
                f"RTT=0 speedup={zero_speedup:.4f}x."
            )
    return notes


def _stage_highlight_notes(stage: pd.DataFrame) -> list[str]:
    notes = []
    percent_columns = [
        f"{TIME_LABELS[column]}_percent" for column in TIME_COLUMNS
    ]
    for _, row in stage.iterrows():
        ranked = sorted(
            (
                (column.removesuffix("_percent"), float(row[column]))
                for column in percent_columns
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top = ", ".join(f"{name}={value:.2f}%" for name, value in ranked)
        notes.append(f"- {row['group']} / {row['method']}: top stages at RTT={row['rtt_ms']:.0f} ms are {top}.")
    return notes


def _write_key_points(
    path: Path,
    summary: pd.DataFrame,
    stage: pd.DataFrame,
    network: pd.DataFrame,
    stage_rtt_ms: float,
) -> None:
    stage_columns = [
        "group",
        "method",
        "rtt_ms",
        "method_time",
        "speedup_vs_target_only",
        "draft_generation_percent",
        "network_wait_percent",
        "cloud_verification_percent",
        "probability_normalize_percent",
        "random_sample_percent",
        "accept_reject_percent",
        "resample_percent",
        "cache_update_percent",
        "sampling_argmax_percent",
    ]
    method_columns = [
        "group",
        "method_name",
        "target_verify_mode",
        "temperature",
        "top_k",
        "top_p",
        "sampling_filter",
        "stochastic_seed",
        "seed_strategy",
        "rtt_ms",
        "method_time",
        "speedup_vs_target_only",
        "accept_rate",
        "mean_first_accept_prob",
        "first_token_accept_rate",
        "target_zero_at_draft_rate",
        "rounds",
        "generated_tokens",
        "drafted_tokens",
    ]
    network_columns = [
        "group",
        "method",
        "rtt_ms",
        "edge_cloud_cycle_time",
        "cloud_verify_share_of_cycle_percent",
        "wait_share_of_cycle_percent",
        "network_cycle_share_of_method_time_percent",
    ]
    network_at_stage = network[(network["rtt_ms"] - stage_rtt_ms).abs() < 1e-9]

    lines = [
        "# Group Meeting Timing Data",
        "",
        "This file contains data snippets only. Use it as input for the final group-meeting report.",
        "",
        "## Break-Even Notes",
        "",
        *_break_even_notes(summary),
        "",
        f"## Stage Highlights at RTT={stage_rtt_ms:.0f} ms",
        "",
        *_stage_highlight_notes(stage),
        "",
        f"## Stage Share Table at RTT={stage_rtt_ms:.0f} ms",
        "",
        _markdown_table(stage, stage_columns, digits=4),
        "",
        "## Method Summary",
        "",
        _markdown_table(summary, method_columns, digits=4),
        "",
        f"## Edge-Cloud Cycle at RTT={stage_rtt_ms:.0f} ms",
        "",
        _markdown_table(network_at_stage, network_columns, digits=4),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _require_pandas()
    specs = _resolve_specs(args.dataset)

    method_summary = _collect_method_summary(specs)
    stage_shares = _collect_stage_shares(specs, args.stage_rtt_ms)
    network_cycle = _collect_network(specs)

    method_summary_output = Path(args.method_summary_output)
    stage_share_output = Path(args.stage_share_output)
    network_output = Path(args.network_output)
    markdown_output = Path(args.markdown_output)
    for path in [
        method_summary_output,
        stage_share_output,
        network_output,
        markdown_output,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    method_summary.to_csv(method_summary_output, index=False)
    stage_shares.to_csv(stage_share_output, index=False)
    network_cycle.to_csv(network_output, index=False)
    _write_key_points(
        markdown_output,
        method_summary,
        stage_shares,
        network_cycle,
        args.stage_rtt_ms,
    )

    print(f"Wrote method summary to {method_summary_output}")
    print(f"Wrote stage shares to {stage_share_output}")
    print(f"Wrote network cycle to {network_output}")
    print(f"Wrote key points to {markdown_output}")


if __name__ == "__main__":
    main()
