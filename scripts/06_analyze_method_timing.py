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
    "upload_wait_time",
    "target_verify_time",
    "posterior_accept_time",
    "cache_update_time",
    "sampling_time",
]

NETWORK_COLUMNS = [
    "upload_latency_time",
    "upload_transfer_time",
    "upload_payload_bytes",
]


TIME_LABELS = {
    "prefill_time": "prefill",
    "draft_generate_time": "draft generation",
    "draft_structure_time": "draft structure",
    "upload_wait_time": "upload wait",
    "target_verify_time": "target verification",
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


def _write_markdown_report(
    path: Path,
    input_path: str,
    summary: pd.DataFrame,
    shares: pd.DataFrame,
    upload_display: pd.DataFrame,
    plot_rtt_ms: float,
    plot_output: str,
) -> None:
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    non_target_upload = upload_display[~upload_display["method"].str.startswith("target-only")]
    max_upload_row = non_target_upload.sort_values(
        "upload_wait_share_percent",
        ascending=False,
    ).head(1)

    lines = [
        "# 阶段报告 03：更精细的方法阶段时间与上传等待分析",
        "",
        "## 本阶段目标",
        "",
        "本阶段在 quick method timing experiment（方法阶段时间快速实验）的基础上，进一步拆解 upload wait（上传等待）对端云式 speculative decoding（推测解码）的影响。",
        "",
        "重点不只是看总耗时，而是把上传等待拆成三个更细指标：",
        "",
        "- `upload_wait_time`：累计上传等待时间。",
        "- `upload_latency_time`：固定 RTT 或固定网络时延部分。",
        "- `upload_transfer_time`：由 payload bytes（上传负载字节数）和 bandwidth（带宽）决定的传输时间。",
        "- `upload_payload_bytes`：本轮实验累计上传的 draft token payload（草稿 token 负载）字节数。",
        "- `upload_wait_share_percent`：上传等待占整个方法耗时的比例。",
        "- `upload_wait_per_round_ms`：平均每轮 draft-to-target verification（草稿到目标验证）等待多少毫秒。",
        "- `upload_wait_per_generated_token_ms`：平均每生成一个 token 承担多少上传等待。",
        "",
        "## 输入与输出",
        "",
        f"- Raw input CSV（原始输入表）：`{input_path}`",
        f"- Plot RTT（画图使用的 RTT）：`{plot_rtt_ms}` ms",
        f"- Plot output（图像输出）：`{plot_output}`",
        f"- Report generated at（报告生成时间）：`{report_time}`",
        "",
        "## 方法汇总",
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
        "## 上传等待精细分析",
        "",
        _markdown_table(
            upload_display,
            [
                "method",
                "rtt_ms",
                "method_time",
                "upload_wait_time",
                "upload_latency_time",
                "upload_transfer_time",
                "upload_payload_bytes",
                "upload_wait_share_percent",
                "upload_wait_per_round_ms",
                "upload_wait_per_generated_token_ms",
                "speedup_vs_target_only",
            ],
            rename={
                "rtt_ms": "RTT(ms)",
                "method_time": "time(s)",
                "upload_wait_time": "upload wait(s)",
                "upload_latency_time": "latency(s)",
                "upload_transfer_time": "transfer(s)",
                "upload_payload_bytes": "payload(bytes)",
                "upload_wait_share_percent": "upload share(%)",
                "upload_wait_per_round_ms": "upload/round(ms)",
                "upload_wait_per_generated_token_ms": "upload/generated token(ms)",
                "speedup_vs_target_only": "speedup",
            },
        ),
        "",
        "## 阶段占比",
        "",
        f"下表展示 RTT={plot_rtt_ms} ms 时各阶段占比。",
        "",
    ]

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
    lines.extend(
        [
            _markdown_table(
                stage_share,
                ["method"] + percent_columns,
                rename={
                    "prefill_time_percent": "prefill(%)",
                    "draft_generate_time_percent": "draft generation(%)",
                    "draft_structure_time_percent": "draft structure(%)",
                    "upload_wait_time_percent": "upload wait(%)",
                    "target_verify_time_percent": "target verification(%)",
                    "posterior_accept_time_percent": "posterior accept(%)",
                    "cache_update_time_percent": "cache update(%)",
                    "sampling_time_percent": "sampling(%)",
                },
                digits=2,
            ),
            "",
            "## 关键观察",
            "",
        ]
    )

    if not max_upload_row.empty:
        row = max_upload_row.iloc[0]
        lines.extend(
            [
                f"- 当前结果中上传等待占比最高的是 `{row['method']}` 在 RTT={row['rtt_ms']:.1f} ms 时，upload wait share（上传等待占比）为 {row['upload_wait_share_percent']:.2f}%。",
                f"- 该点的总耗时为 {row['method_time']:.4f} s，其中 upload wait（上传等待）为 {row['upload_wait_time']:.4f} s，speedup_vs_target_only（相对 target-only 加速比）为 {row['speedup_vs_target_only']:.4f}。",
            ]
        )

    lines.extend(
        [
            "- 在当前 full-prefix（完整前缀）实现下，RTT=0 ms 时 speculative methods（推测方法）仍未超过 target-only，主要原因是 draft generation（草稿生成）占比过高。",
            "- 当 RTT 增大时，upload wait（上传等待）会快速变成主要成本之一，尤其是轮数较多的方法。",
            "- target-only（只用目标模型）没有 draft token 上传，因此 upload wait 始终为 0。",
            "",
            "## 当前口径说明",
            "",
            "当前 upload wait（上传等待）不是实际网络测速，而是每轮 draft tokens（草稿 token）生成完成后模拟一次网络等待：",
            "",
            "```python",
            "upload_time = rtt_ms / 1000 + payload_bytes * 8 / (bandwidth_mbps * 1_000_000)",
            "time.sleep(upload_time)",
            "```",
            "",
            "当 `upload_bandwidth_mbps <= 0` 时，只模拟固定 RTT，保持旧版实验口径；当 bandwidth（带宽）为正时，会额外加入 payload transfer time（负载传输时间）。它仍然没有建模 serialization（序列化）、协议头、拥塞和 batching（批处理）等真实网络因素。",
            "",
            "## 下一步计划",
            "",
            "1. 用 `--repeats 3` 或更高重复次数重跑实验，降低 quick run（快速实验）的随机波动。",
            "2. 验证 `--implementation kv-cache` 的 correctness（正确性），因为 break-even RTT（盈亏平衡 RTT）必须基于 KV cache 版本判断。",
            "3. 使用不同 `--upload-bandwidth-mbps` 档位重跑实验，观察 bandwidth（带宽）是否会在低带宽下成为主要瓶颈。",
            "4. 对 SpecInfer-style token tree（SpecInfer 风格 token 树）进一步拆分 wasted branch tokens（无效分支 token）和实际 compute waste（计算浪费）。",
            "",
            "## 英文术语中文解释",
            "",
            "| English | 中文解释 |",
            "| --- | --- |",
            "| upload wait | 上传等待，端侧草稿 token 到云端目标模型验证前的等待 |",
            "| RTT | Round Trip Time，往返时延 |",
            "| draft generation | 草稿生成，小模型生成候选 token 的过程 |",
            "| target verification | 目标模型验证，大模型验证候选 token 的过程 |",
            "| speedup | 加速比，相对 target-only 的速度比例 |",
            "| full-prefix | 完整前缀，每次前向重新输入完整上下文 |",
            "| KV cache | 键值缓存，用来复用历史 token 的 key/value |",
            "| break-even RTT | 盈亏平衡 RTT，加速收益刚好被网络等待抵消的时延 |",
            "| bandwidth | 带宽，单位时间可传输的数据量 |",
            "| payload bytes | 上传负载字节数 |",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    for column in NETWORK_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
    if "upload_token_bytes" not in df.columns:
        df["upload_token_bytes"] = 0
    if "upload_bandwidth_mbps" not in df.columns:
        df["upload_bandwidth_mbps"] = 0.0

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
        "upload_token_bytes",
        "upload_bandwidth_mbps",
    ]
    summary = (
        df.groupby(group_columns, as_index=False)
        .agg(
            method_time=("method_time", "mean"),
            target_only_time=("target_only_time", "mean"),
            speedup_vs_target_only=("speedup_vs_target_only", "mean"),
            accept_rate=("accept_rate", "mean"),
            rounds=("rounds", "mean"),
            generated_tokens=("generated_tokens", "mean"),
            accepted_tokens=("accepted_tokens", "mean"),
            drafted_tokens=("drafted_tokens", "mean"),
            wasted_branch_time_or_tokens=("wasted_branch_time_or_tokens", "mean"),
            **{column: (column, "mean") for column in NETWORK_COLUMNS},
            **{column: (column, "mean") for column in TIME_COLUMNS},
        )
        .sort_values(group_columns)
    )
    summary["drafted_tokens_per_round"] = (
        summary["drafted_tokens"] / summary["rounds"].where(summary["rounds"] != 0, 1.0)
    )
    summary["accepted_tokens_per_round"] = (
        summary["accepted_tokens"] / summary["rounds"].where(summary["rounds"] != 0, 1.0)
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
    plot_path = Path(args.plot_output)
    markdown_path = Path(args.markdown_output) if args.markdown_output else None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    share_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
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
            "upload_token_bytes",
            "upload_bandwidth_mbps",
            "speedup_vs_target_only",
        ]
    ].copy()
    upload_display.insert(
        1,
        "method",
        upload_display["method_name"] + " / " + upload_display["implementation"],
    )
    upload_display["upload_wait_share_percent"] = (
        upload_display["upload_wait_time"]
        / upload_display["method_time"].where(upload_display["method_time"] != 0, 1.0)
        * 100.0
    )
    upload_display["upload_wait_per_round_ms"] = (
        upload_display["upload_wait_time"]
        / upload_display["rounds"].where(upload_display["rounds"] != 0, 1.0)
        * 1000.0
    )
    upload_display["upload_wait_per_generated_token_ms"] = (
        upload_display["upload_wait_time"]
        / upload_display["generated_tokens"].where(upload_display["generated_tokens"] != 0, 1.0)
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
            "upload_token_bytes",
            "upload_bandwidth_mbps",
            "upload_wait_share_percent",
            "upload_wait_per_round_ms",
            "upload_wait_per_generated_token_ms",
            "speedup_vs_target_only",
        ]
    ]
    upload_display.to_csv(upload_path, index=False)

    print("\n=== Upload wait summary ===")
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
                "upload_token_bytes": "{:.0f}".format,
                "upload_bandwidth_mbps": "{:.3f}".format,
                "upload_wait_share_percent": "{:.2f}".format,
                "upload_wait_per_round_ms": "{:.2f}".format,
                "upload_wait_per_generated_token_ms": "{:.2f}".format,
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
            args.plot_rtt_ms,
            args.plot_output,
        )

    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote stage shares to {share_path}")
    print(f"Wrote upload summary to {upload_path}")
    if markdown_path is not None:
        print(f"Wrote markdown report to {markdown_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
