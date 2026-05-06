from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from edge_specdec.decoding import (
    DecodeResult,
    specinfer_tree_simplified,
    speculative_greedy,
    speculative_greedy_adaptive_draft,
    speculative_greedy_cached,
    target_only_greedy,
    target_only_greedy_cached,
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str


METHOD_SPECS = {
    "target-only": MethodSpec(
        name="target-only",
        description="Target-only greedy decoding（只用目标模型的贪心解码）",
    ),
    "vanilla-spec": MethodSpec(
        name="vanilla-spec",
        description="Vanilla speculative decoding（普通线性推测解码）",
    ),
    "specinfer-simplified": MethodSpec(
        name="specinfer-simplified",
        description="Simplified SpecInfer token tree（简化 SpecInfer token 树）",
    ),
    "dsd-adaptive-draft": MethodSpec(
        name="dsd-adaptive-draft",
        description="DSD-style adaptive draft strategy（DSD 风格自适应草稿策略）",
    ),
}


def available_methods() -> list[str]:
    return list(METHOD_SPECS)


def run_target_only(
    method_impl: str,
    target_model,
    draft_model,
    input_ids,
    max_new_tokens: int,
    draft_k: int,
    rtt_ms: float,
    eos_token_id: int | None,
    tree_width: int,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    target_verify_mode: str = "sequential",
) -> DecodeResult:
    if method_impl == "kv-cache":
        return target_only_greedy_cached(
            target_model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            suppress_token_id=suppress_token_id,
        )
    return target_only_greedy(
        target_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        suppress_token_id=suppress_token_id,
    )


def run_vanilla_spec(
    method_impl: str,
    target_model,
    draft_model,
    input_ids,
    max_new_tokens: int,
    draft_k: int,
    rtt_ms: float,
    eos_token_id: int | None,
    tree_width: int,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    target_verify_mode: str = "sequential",
) -> DecodeResult:
    if method_impl == "kv-cache":
        return speculative_greedy_cached(
            target_model,
            draft_model,
            input_ids,
            max_new_tokens=max_new_tokens,
            draft_k=draft_k,
            rtt_ms=rtt_ms,
            upload_token_bytes=upload_token_bytes,
            upload_bandwidth_mbps=upload_bandwidth_mbps,
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
            target_verify_mode=target_verify_mode,
            eos_token_id=eos_token_id,
            suppress_token_id=suppress_token_id,
        )
    return speculative_greedy(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_k=draft_k,
        rtt_ms=rtt_ms,
        upload_token_bytes=upload_token_bytes,
        upload_bandwidth_mbps=upload_bandwidth_mbps,
        uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        downlink_token_bytes=downlink_token_bytes,
        downlink_fixed_bytes=downlink_fixed_bytes,
        downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        eos_token_id=eos_token_id,
        suppress_token_id=suppress_token_id,
    )


def run_specinfer(
    method_impl: str,
    target_model,
    draft_model,
    input_ids,
    max_new_tokens: int,
    draft_k: int,
    rtt_ms: float,
    eos_token_id: int | None,
    tree_width: int,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    target_verify_mode: str = "sequential",
) -> DecodeResult:
    return specinfer_tree_simplified(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_k=draft_k,
        tree_width=tree_width,
        rtt_ms=rtt_ms,
        upload_token_bytes=upload_token_bytes,
        upload_bandwidth_mbps=upload_bandwidth_mbps,
        uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        downlink_token_bytes=downlink_token_bytes,
        downlink_fixed_bytes=downlink_fixed_bytes,
        downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        eos_token_id=eos_token_id,
        suppress_token_id=suppress_token_id,
    )


def run_dsd_adaptive(
    method_impl: str,
    target_model,
    draft_model,
    input_ids,
    max_new_tokens: int,
    draft_k: int,
    rtt_ms: float,
    eos_token_id: int | None,
    tree_width: int,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    target_verify_mode: str = "sequential",
) -> DecodeResult:
    if method_impl == "kv-cache":
        return speculative_greedy_cached(
            target_model,
            draft_model,
            input_ids,
            max_new_tokens=max_new_tokens,
            draft_k=draft_k,
            min_draft_k=1,
            max_draft_k=max(draft_k * 2, draft_k),
            rtt_ms=rtt_ms,
            upload_token_bytes=upload_token_bytes,
            upload_bandwidth_mbps=upload_bandwidth_mbps,
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
            target_verify_mode=target_verify_mode,
            eos_token_id=eos_token_id,
            suppress_token_id=suppress_token_id,
        )
    return speculative_greedy_adaptive_draft(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_k=draft_k,
        min_draft_k=1,
        max_draft_k=max(draft_k * 2, draft_k),
        rtt_ms=rtt_ms,
        upload_token_bytes=upload_token_bytes,
        upload_bandwidth_mbps=upload_bandwidth_mbps,
        uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        downlink_token_bytes=downlink_token_bytes,
        downlink_fixed_bytes=downlink_fixed_bytes,
        downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        eos_token_id=eos_token_id,
        suppress_token_id=suppress_token_id,
    )


RUNNERS: dict[str, Callable[..., DecodeResult]] = {
    "target-only": run_target_only,
    "vanilla-spec": run_vanilla_spec,
    "specinfer-simplified": run_specinfer,
    "dsd-adaptive-draft": run_dsd_adaptive,
}
