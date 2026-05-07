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
    target_only_sampling_cached,
    traditional_speculative_sampling_cached,
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str


METHOD_SPECS = {
    "target-only": MethodSpec(
        name="target-only",
        description="Target-only greedy decoding.",
    ),
    "vanilla-spec": MethodSpec(
        name="vanilla-spec",
        description="Vanilla greedy speculative decoding.",
    ),
    "specinfer-simplified": MethodSpec(
        name="specinfer-simplified",
        description="Simplified SpecInfer token tree.",
    ),
    "dsd-adaptive-draft": MethodSpec(
        name="dsd-adaptive-draft",
        description="DSD-style adaptive draft strategy.",
    ),
    "target-only-sampling": MethodSpec(
        name="target-only-sampling",
        description="Target-only autoregressive probability sampling.",
    ),
    "traditional-spec-sampling": MethodSpec(
        name="traditional-spec-sampling",
        description="Traditional probability speculative sampling with p/q accept-reject.",
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
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


def run_target_only_sampling(
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
) -> DecodeResult:
    if method_impl != "kv-cache":
        raise ValueError("target-only-sampling currently requires --implementation kv-cache")
    return target_only_sampling_cached(
        target_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        suppress_token_id=suppress_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


def run_traditional_spec_sampling(
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
    target_verify_mode: str = "batch",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
) -> DecodeResult:
    if method_impl != "kv-cache":
        raise ValueError("traditional-spec-sampling currently requires --implementation kv-cache")
    return traditional_speculative_sampling_cached(
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
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


RUNNERS: dict[str, Callable[..., DecodeResult]] = {
    "target-only": run_target_only,
    "vanilla-spec": run_vanilla_spec,
    "specinfer-simplified": run_specinfer,
    "dsd-adaptive-draft": run_dsd_adaptive,
    "target-only-sampling": run_target_only_sampling,
    "traditional-spec-sampling": run_traditional_spec_sampling,
}
