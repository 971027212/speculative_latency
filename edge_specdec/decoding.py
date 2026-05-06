from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import time
from typing import Any

import torch


TIME_BUCKETS = [
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
]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class DecodeTimings:
    prefill_time: float = 0.0
    draft_generate_time: float = 0.0
    draft_structure_time: float = 0.0
    upload_wait_time: float = 0.0
    upload_latency_time: float = 0.0
    upload_transfer_time: float = 0.0
    upload_payload_bytes: float = 0.0
    target_verify_time: float = 0.0
    posterior_accept_time: float = 0.0
    cache_update_time: float = 0.0
    kv_or_input_update_time: float = 0.0
    sampling_time: float = 0.0
    wasted_branch_time_or_tokens: float = 0.0
    total_decode_time: float = 0.0

    def add(self, bucket: str, seconds: float) -> None:
        if bucket == "kv_or_input_update_time":
            self.cache_update_time += seconds
        setattr(self, bucket, getattr(self, bucket) + seconds)

    def as_dict(self) -> dict[str, float]:
        return {
            "prefill_time": self.prefill_time,
            "draft_generate_time": self.draft_generate_time,
            "draft_structure_time": self.draft_structure_time,
            "upload_wait_time": self.upload_wait_time,
            "upload_latency_time": self.upload_latency_time,
            "upload_transfer_time": self.upload_transfer_time,
            "upload_payload_bytes": self.upload_payload_bytes,
            "target_verify_time": self.target_verify_time,
            "posterior_accept_time": self.posterior_accept_time,
            "cache_update_time": self.cache_update_time,
            "kv_or_input_update_time": self.kv_or_input_update_time,
            "sampling_time": self.sampling_time,
            "wasted_branch_time_or_tokens": self.wasted_branch_time_or_tokens,
            "total_decode_time": self.total_decode_time,
        }


@contextmanager
def timed_bucket(timings: DecodeTimings, bucket: str, device: torch.device):
    _sync_if_cuda(device)
    start = time.perf_counter()
    yield
    _sync_if_cuda(device)
    timings.add(bucket, time.perf_counter() - start)


def simulate_upload_wait(
    timings: DecodeTimings,
    draft_token_count: int,
    rtt_ms: float,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
) -> None:
    """Simulate edge-to-cloud upload wait for draft tokens.

    rtt_ms models fixed per-round latency. upload_bandwidth_mbps optionally
    adds payload transfer time from token ids. A non-positive bandwidth keeps
    the old fixed-RTT behavior while still recording payload size.
    """

    payload_bytes = max(0, draft_token_count) * max(0, upload_token_bytes)
    latency_seconds = max(0.0, rtt_ms) / 1000.0
    transfer_seconds = 0.0
    if upload_bandwidth_mbps > 0 and payload_bytes > 0:
        transfer_seconds = (payload_bytes * 8.0) / (upload_bandwidth_mbps * 1_000_000.0)

    timings.upload_payload_bytes += float(payload_bytes)
    timings.upload_latency_time += latency_seconds
    timings.upload_transfer_time += transfer_seconds

    wait_seconds = latency_seconds + transfer_seconds
    if wait_seconds > 0:
        start = time.perf_counter()
        time.sleep(wait_seconds)
        timings.upload_wait_time += time.perf_counter() - start


@dataclass
class DecodeResult:
    output_ids: list[int]
    timings: DecodeTimings
    generated_tokens: int
    accepted_tokens: int = 0
    drafted_tokens: int = 0
    rounds: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accept_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens


def _mask_token(logits: torch.Tensor, token_id: int | None) -> torch.Tensor:
    if token_id is None:
        return logits
    masked_logits = logits.clone()
    masked_logits[..., token_id] = torch.finfo(masked_logits.dtype).min
    return masked_logits


def _argmax_token(logits: torch.Tensor, suppress_token_id: int | None = None) -> torch.Tensor:
    logits = _mask_token(logits, suppress_token_id)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _argmax_token_id(logits: torch.Tensor, suppress_token_id: int | None = None) -> int:
    return int(_argmax_token(logits, suppress_token_id).item())


def _argmax_next_token(
    logits: torch.Tensor,
    suppress_token_id: int | None = None,
) -> torch.Tensor:
    return _argmax_token(logits[:, -1, :], suppress_token_id)


def _append_token(input_ids: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    return torch.cat([input_ids, token.to(input_ids.device)], dim=-1)


def _tokens_tensor(tokens: list[int], input_ids: torch.Tensor) -> torch.Tensor:
    return torch.tensor([tokens], dtype=input_ids.dtype, device=input_ids.device)


def _past_seq_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and first_layer:
            return int(first_layer[0].shape[-2])
    return 0


def _clone_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            legacy_cache = _clone_past_key_values(past_key_values.to_legacy_cache())
            from_legacy_cache = getattr(type(past_key_values), "from_legacy_cache", None)
            if from_legacy_cache is not None:
                return from_legacy_cache(legacy_cache)
            try:
                from transformers.cache_utils import DynamicCache

                return DynamicCache.from_legacy_cache(legacy_cache)
            except Exception:
                pass
        except Exception:
            pass
    if torch.is_tensor(past_key_values):
        return past_key_values.clone()
    if isinstance(past_key_values, tuple):
        return tuple(_clone_past_key_values(item) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [_clone_past_key_values(item) for item in past_key_values]
    return copy.deepcopy(past_key_values)


def _crop_past_key_values(past_key_values: Any, seq_len: int) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        result = past_key_values.crop(seq_len)
        return past_key_values if result is None else result
    if torch.is_tensor(past_key_values):
        return past_key_values[..., :seq_len, :].contiguous()
    if isinstance(past_key_values, tuple):
        return tuple(_crop_past_key_values(item, seq_len) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [_crop_past_key_values(item, seq_len) for item in past_key_values]
    return past_key_values


def _cached_forward(model, input_ids: torch.Tensor, past_key_values: Any):
    past_len = _past_seq_len(past_key_values)
    sequence_len = input_ids.shape[-1]
    attention_mask = input_ids.new_ones((input_ids.shape[0], past_len + sequence_len))
    cache_position = torch.arange(
        past_len,
        past_len + sequence_len,
        dtype=torch.long,
        device=input_ids.device,
    )
    try:
        return model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
    except TypeError:
        return model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )


@torch.inference_mode()
def target_only_greedy(
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    suppress_token_id: int | None = None,
) -> DecodeResult:
    """Plain greedy decoding with the target model.

    This is the correctness oracle: the speculative decoder must produce the
    same token sequence when both use greedy/temperature=0 decoding.
    """

    device = input_ids.device
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]

    _sync_if_cuda(device)
    total_start = time.perf_counter()

    for step in range(max_new_tokens):
        bucket = "prefill_time" if step == 0 else "target_verify_time"
        with timed_bucket(timings, bucket, device):
            logits = target_model(output_ids).logits
        with timed_bucket(timings, "sampling_time", device):
            next_token = _argmax_next_token(logits, suppress_token_id)
        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = _append_token(output_ids, next_token)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    _sync_if_cuda(device)
    timings.total_decode_time = time.perf_counter() - total_start
    return DecodeResult(
        output_ids=output_ids.squeeze(0).tolist(),
        timings=timings,
        generated_tokens=output_ids.shape[-1] - prompt_len,
    )


@torch.inference_mode()
def target_only_greedy_cached(
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    suppress_token_id: int | None = None,
) -> DecodeResult:
    """Greedy target-only decoding with KV cache.

    This is the fair baseline for the cache-aware speculative decoder.
    """

    device = input_ids.device
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]

    _sync_if_cuda(device)
    total_start = time.perf_counter()

    attention_mask = torch.ones_like(output_ids)

    with timed_bucket(timings, "prefill_time", device):
        outputs = target_model(
            output_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    for _ in range(max_new_tokens):
        with timed_bucket(timings, "sampling_time", device):
            next_token = _argmax_token(next_logits, suppress_token_id)

        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = _append_token(output_ids, next_token)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)],
                dim=-1,
            )

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

        with timed_bucket(timings, "target_verify_time", device):
            outputs = target_model(
                next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

    _sync_if_cuda(device)
    timings.total_decode_time = time.perf_counter() - total_start
    return DecodeResult(
        output_ids=output_ids.squeeze(0).tolist(),
        timings=timings,
        generated_tokens=output_ids.shape[-1] - prompt_len,
    )


@torch.inference_mode()
def speculative_greedy(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_k: int = 4,
    rtt_ms: float = 0.0,
    eos_token_id: int | None = None,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
) -> DecodeResult:
    """Minimal greedy speculative decoding.

    The implementation intentionally uses full-prefix forward passes instead
    of KV cache updates. That keeps the accept/reject logic explicit for study.
    """

    if draft_k < 1:
        raise ValueError("draft_k must be >= 1")

    device = input_ids.device
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]
    accepted_tokens = 0
    drafted_tokens = 0
    rounds = 0

    _sync_if_cuda(device)
    total_start = time.perf_counter()

    while output_ids.shape[-1] - prompt_len < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - (output_ids.shape[-1] - prompt_len)
        this_k = min(draft_k, remaining)

        draft_tokens: list[int] = []
        draft_prefix = output_ids.clone()
        for _ in range(this_k):
            with timed_bucket(timings, "draft_generate_time", device):
                draft_logits = draft_model(draft_prefix).logits
            with timed_bucket(timings, "sampling_time", device):
                draft_next = _argmax_next_token(draft_logits, suppress_token_id)
            token_id = int(draft_next.item())
            draft_tokens.append(token_id)
            drafted_tokens += 1
            with timed_bucket(timings, "kv_or_input_update_time", device):
                draft_prefix = _append_token(draft_prefix, draft_next)
            if eos_token_id is not None and token_id == eos_token_id:
                break

        simulate_upload_wait(
            timings,
            draft_token_count=len(draft_tokens),
            rtt_ms=rtt_ms,
            upload_token_bytes=upload_token_bytes,
            upload_bandwidth_mbps=upload_bandwidth_mbps,
        )

        draft_tensor = torch.tensor([draft_tokens], dtype=input_ids.dtype, device=device)
        verify_input = torch.cat([output_ids, draft_tensor], dim=-1)
        verify_start = output_ids.shape[-1]

        with timed_bucket(timings, "target_verify_time", device):
            target_logits = target_model(verify_input).logits

        with timed_bucket(timings, "sampling_time", device):
            target_tokens_for_draft = []
            for i in range(len(draft_tokens)):
                predict_pos = verify_start + i - 1
                token = _argmax_token_id(
                    target_logits[:, predict_pos, :],
                    suppress_token_id,
                )
                target_tokens_for_draft.append(token)

            bonus_token = _argmax_token_id(target_logits[:, -1, :], suppress_token_id)

        with timed_bucket(timings, "posterior_accept_time", device):
            accepted_this_round = 0
            replacement_token: int | None = None
            for draft_token, target_token in zip(draft_tokens, target_tokens_for_draft):
                if draft_token == target_token:
                    accepted_this_round += 1
                    if eos_token_id is not None and draft_token == eos_token_id:
                        break
                else:
                    replacement_token = target_token
                    break

        new_tokens = draft_tokens[:accepted_this_round]
        accepted_tokens += accepted_this_round

        generated_so_far = output_ids.shape[-1] - prompt_len
        can_add_more = generated_so_far + len(new_tokens) < max_new_tokens
        hit_eos = eos_token_id is not None and eos_token_id in new_tokens

        if can_add_more and not hit_eos:
            if replacement_token is not None:
                new_tokens.append(replacement_token)
            elif accepted_this_round == len(draft_tokens):
                new_tokens.append(bonus_token)

        new_tokens = new_tokens[: max_new_tokens - generated_so_far]
        if not new_tokens:
            # Greedy target verification should always give us at least one
            # token, but keep a defensive guard so a bad model output cannot
            # spin forever.
            new_tokens = [target_tokens_for_draft[0]]

        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = torch.cat(
                [
                    output_ids,
                    torch.tensor([new_tokens], dtype=input_ids.dtype, device=device),
                ],
                dim=-1,
            )

        if eos_token_id is not None and eos_token_id in new_tokens:
            break

    _sync_if_cuda(device)
    timings.total_decode_time = time.perf_counter() - total_start
    return DecodeResult(
        output_ids=output_ids.squeeze(0).tolist(),
        timings=timings,
        generated_tokens=output_ids.shape[-1] - prompt_len,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        rounds=rounds,
    )


@torch.inference_mode()
def speculative_greedy_cached(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_k: int = 4,
    rtt_ms: float = 0.0,
    eos_token_id: int | None = None,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    min_draft_k: int | None = None,
    max_draft_k: int | None = None,
    suppress_token_id: int | None = None,
) -> DecodeResult:
    """Greedy speculative decoding with KV cache.

    The important indexing rule:
    target_next_logits predicts the first draft token. When we verify k draft
    tokens with target_model(draft_tokens, past=target_past), verify logits at
    position i predict the token after draft_tokens[i]. So the target tokens
    used for comparison are:
      [target_next_logits, verify_logits[0], ..., verify_logits[k-2]]
    and verify_logits[k-1] is the bonus token after all draft tokens.
    """

    if draft_k < 1:
        raise ValueError("draft_k must be >= 1")
    adaptive_draft = min_draft_k is not None or max_draft_k is not None
    if adaptive_draft:
        min_draft_k = 1 if min_draft_k is None else min_draft_k
        max_draft_k = draft_k if max_draft_k is None else max_draft_k
        if min_draft_k < 1 or max_draft_k < min_draft_k:
            raise ValueError("Require 1 <= min_draft_k <= max_draft_k")

    device = input_ids.device
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]
    accepted_tokens = 0
    drafted_tokens = 0
    rounds = 0
    current_k = max(min(draft_k, max_draft_k), min_draft_k) if adaptive_draft else draft_k
    draft_k_history: list[int] = []

    _sync_if_cuda(device)
    total_start = time.perf_counter()

    attention_mask = torch.ones_like(output_ids)

    with timed_bucket(timings, "prefill_time", device):
        target_outputs = target_model(
            output_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        draft_outputs = draft_model(
            output_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    target_past = target_outputs.past_key_values
    draft_past = draft_outputs.past_key_values
    target_next_logits = target_outputs.logits[:, -1, :]
    draft_next_logits = draft_outputs.logits[:, -1, :]

    while output_ids.shape[-1] - prompt_len < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - (output_ids.shape[-1] - prompt_len)
        this_k = min(current_k, remaining)
        if adaptive_draft:
            draft_k_history.append(this_k)

        draft_tokens: list[int] = []
        provisional_draft_past = _clone_past_key_values(draft_past)
        provisional_draft_next_logits = draft_next_logits

        for i in range(this_k):
            with timed_bucket(timings, "sampling_time", device):
                draft_next = _argmax_token(provisional_draft_next_logits, suppress_token_id)
            token_id = int(draft_next.item())
            draft_tokens.append(token_id)
            drafted_tokens += 1

            if eos_token_id is not None and token_id == eos_token_id:
                break
            if i == this_k - 1:
                break

            with timed_bucket(timings, "draft_generate_time", device):
                draft_outputs = _cached_forward(
                    draft_model,
                    draft_next,
                    provisional_draft_past,
                )
            provisional_draft_past = draft_outputs.past_key_values
            provisional_draft_next_logits = draft_outputs.logits[:, -1, :]

        simulate_upload_wait(
            timings,
            draft_token_count=len(draft_tokens),
            rtt_ms=rtt_ms,
            upload_token_bytes=upload_token_bytes,
            upload_bandwidth_mbps=upload_bandwidth_mbps,
        )

        draft_tensor = _tokens_tensor(draft_tokens, input_ids)
        with timed_bucket(timings, "target_verify_time", device):
            verify_outputs = _cached_forward(
                target_model,
                draft_tensor,
                _clone_past_key_values(target_past),
            )
        verify_logits = verify_outputs.logits

        with timed_bucket(timings, "sampling_time", device):
            target_tokens_for_draft = [
                _argmax_token_id(target_next_logits, suppress_token_id)
            ]
            for i in range(1, len(draft_tokens)):
                token = _argmax_token_id(
                    verify_logits[:, i - 1, :],
                    suppress_token_id,
                )
                target_tokens_for_draft.append(token)
            bonus_token = _argmax_token_id(verify_logits[:, -1, :], suppress_token_id)

        with timed_bucket(timings, "posterior_accept_time", device):
            accepted_this_round = 0
            replacement_token: int | None = None
            for draft_token, target_token in zip(draft_tokens, target_tokens_for_draft):
                if draft_token == target_token:
                    accepted_this_round += 1
                    if eos_token_id is not None and draft_token == eos_token_id:
                        break
                else:
                    replacement_token = target_token
                    break

        new_tokens = draft_tokens[:accepted_this_round]
        accepted_tokens += accepted_this_round

        generated_so_far = output_ids.shape[-1] - prompt_len
        can_add_more = generated_so_far + len(new_tokens) < max_new_tokens
        hit_eos = eos_token_id is not None and eos_token_id in new_tokens

        if can_add_more and not hit_eos:
            if replacement_token is not None:
                new_tokens.append(replacement_token)
            elif accepted_this_round == len(draft_tokens):
                new_tokens.append(bonus_token)

        new_tokens = new_tokens[: max_new_tokens - generated_so_far]
        if not new_tokens:
            new_tokens = [target_tokens_for_draft[0]]

        accepted_draft_count = min(accepted_this_round, len(new_tokens))
        extra_tokens = new_tokens[accepted_draft_count:]
        new_tensor = _tokens_tensor(new_tokens, input_ids)
        can_reuse_verified_target = accepted_this_round == len(draft_tokens)
        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = torch.cat([output_ids, new_tensor], dim=-1)
            attention_mask = torch.ones_like(output_ids)

            if can_reuse_verified_target:
                target_past = verify_outputs.past_key_values
                if extra_tokens:
                    target_outputs = _cached_forward(
                        target_model,
                        _tokens_tensor(extra_tokens, input_ids),
                        target_past,
                    )
                    target_past = target_outputs.past_key_values
                    target_next_logits = target_outputs.logits[:, -1, :]
                else:
                    target_next_logits = verify_logits[:, -1, :]
            else:
                target_outputs = _cached_forward(target_model, new_tensor, target_past)
                target_past = target_outputs.past_key_values
                target_next_logits = target_outputs.logits[:, -1, :]

            draft_outputs = _cached_forward(draft_model, new_tensor, draft_past)
            draft_past = draft_outputs.past_key_values
            draft_next_logits = draft_outputs.logits[:, -1, :]

        if adaptive_draft:
            if accepted_this_round == len(draft_tokens):
                current_k = min(current_k + 1, max_draft_k)
            elif accepted_this_round <= max(1, len(draft_tokens) // 2):
                current_k = max(current_k - 1, min_draft_k)

        if eos_token_id is not None and eos_token_id in new_tokens:
            break

    _sync_if_cuda(device)
    timings.total_decode_time = time.perf_counter() - total_start
    return DecodeResult(
        output_ids=output_ids.squeeze(0).tolist(),
        timings=timings,
        generated_tokens=output_ids.shape[-1] - prompt_len,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        rounds=rounds,
        extra={
            "draft_k_history": draft_k_history,
            "mean_draft_k": sum(draft_k_history) / len(draft_k_history)
            if draft_k_history
            else 0.0,
        }
        if adaptive_draft
        else {},
    )


@torch.inference_mode()
def speculative_greedy_adaptive_draft(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_k: int = 4,
    min_draft_k: int = 1,
    max_draft_k: int = 8,
    rtt_ms: float = 0.0,
    eos_token_id: int | None = None,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
) -> DecodeResult:
    """Lossless speculative decoding with adaptive draft length.

    This is the first DSD-style draft strategy baseline. DSD here means
    draft model design/usage strategy, not a new target verification rule.
    """

    if min_draft_k < 1 or max_draft_k < min_draft_k:
        raise ValueError("Require 1 <= min_draft_k <= max_draft_k")

    device = input_ids.device
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]
    accepted_tokens = 0
    drafted_tokens = 0
    rounds = 0
    current_k = max(min(draft_k, max_draft_k), min_draft_k)
    draft_k_history: list[int] = []

    _sync_if_cuda(device)
    total_start = time.perf_counter()

    while output_ids.shape[-1] - prompt_len < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - (output_ids.shape[-1] - prompt_len)
        this_k = min(current_k, remaining)
        draft_k_history.append(this_k)

        draft_tokens: list[int] = []
        draft_prefix = output_ids.clone()
        for _ in range(this_k):
            with timed_bucket(timings, "draft_generate_time", device):
                draft_logits = draft_model(draft_prefix).logits
            with timed_bucket(timings, "sampling_time", device):
                draft_next = _argmax_next_token(draft_logits, suppress_token_id)
            token_id = int(draft_next.item())
            draft_tokens.append(token_id)
            drafted_tokens += 1
            with timed_bucket(timings, "kv_or_input_update_time", device):
                draft_prefix = _append_token(draft_prefix, draft_next)
            if eos_token_id is not None and token_id == eos_token_id:
                break

        simulate_upload_wait(
            timings,
            draft_token_count=len(draft_tokens),
            rtt_ms=rtt_ms,
            upload_token_bytes=upload_token_bytes,
            upload_bandwidth_mbps=upload_bandwidth_mbps,
        )

        draft_tensor = _tokens_tensor(draft_tokens, input_ids)
        verify_input = torch.cat([output_ids, draft_tensor], dim=-1)
        verify_start = output_ids.shape[-1]

        with timed_bucket(timings, "target_verify_time", device):
            target_logits = target_model(verify_input).logits

        with timed_bucket(timings, "sampling_time", device):
            target_tokens_for_draft = []
            for i in range(len(draft_tokens)):
                predict_pos = verify_start + i - 1
                token = _argmax_token_id(
                    target_logits[:, predict_pos, :],
                    suppress_token_id,
                )
                target_tokens_for_draft.append(token)
            bonus_token = _argmax_token_id(target_logits[:, -1, :], suppress_token_id)

        with timed_bucket(timings, "posterior_accept_time", device):
            accepted_this_round = 0
            replacement_token: int | None = None
            for draft_token, target_token in zip(draft_tokens, target_tokens_for_draft):
                if draft_token == target_token:
                    accepted_this_round += 1
                    if eos_token_id is not None and draft_token == eos_token_id:
                        break
                else:
                    replacement_token = target_token
                    break

        new_tokens = draft_tokens[:accepted_this_round]
        accepted_tokens += accepted_this_round

        generated_so_far = output_ids.shape[-1] - prompt_len
        can_add_more = generated_so_far + len(new_tokens) < max_new_tokens
        hit_eos = eos_token_id is not None and eos_token_id in new_tokens

        if can_add_more and not hit_eos:
            if replacement_token is not None:
                new_tokens.append(replacement_token)
            elif accepted_this_round == len(draft_tokens):
                new_tokens.append(bonus_token)

        new_tokens = new_tokens[: max_new_tokens - generated_so_far]
        if not new_tokens:
            new_tokens = [target_tokens_for_draft[0]]

        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = torch.cat([output_ids, _tokens_tensor(new_tokens, input_ids)], dim=-1)

        if accepted_this_round == len(draft_tokens):
            current_k = min(current_k + 1, max_draft_k)
        elif accepted_this_round <= max(1, len(draft_tokens) // 2):
            current_k = max(current_k - 1, min_draft_k)

        if eos_token_id is not None and eos_token_id in new_tokens:
            break

    _sync_if_cuda(device)
    timings.total_decode_time = time.perf_counter() - total_start
    return DecodeResult(
        output_ids=output_ids.squeeze(0).tolist(),
        timings=timings,
        generated_tokens=output_ids.shape[-1] - prompt_len,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        rounds=rounds,
        extra={
            "draft_k_history": draft_k_history,
            "mean_draft_k": sum(draft_k_history) / len(draft_k_history)
            if draft_k_history
            else 0.0,
        },
    )


@torch.inference_mode()
def specinfer_tree_simplified(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_k: int = 4,
    tree_width: int = 2,
    rtt_ms: float = 0.0,
    eos_token_id: int | None = None,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    suppress_token_id: int | None = None,
) -> DecodeResult:
    """Simplified SpecInfer-style tree draft and batched verification.

    This is not a full SpecInfer serving-system reproduction. It builds a small
    token tree by taking top candidates for the first draft token and greedily
    extending each branch, then verifies the candidate paths in one target batch.
    The accepted path is still lossless because output tokens follow target
    greedy posterior acceptance.
    """

    if draft_k < 1 or tree_width < 1:
        raise ValueError("Require draft_k >= 1 and tree_width >= 1")

    device = input_ids.device
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]
    accepted_tokens = 0
    drafted_tokens = 0
    wasted_branch_tokens = 0
    rounds = 0

    _sync_if_cuda(device)
    total_start = time.perf_counter()

    while output_ids.shape[-1] - prompt_len < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - (output_ids.shape[-1] - prompt_len)
        this_k = min(draft_k, remaining)

        with timed_bucket(timings, "draft_generate_time", device):
            root_logits = draft_model(output_ids).logits
        with timed_bucket(timings, "sampling_time", device):
            top_tokens = torch.topk(
                _mask_token(root_logits[:, -1, :], suppress_token_id),
                k=tree_width,
                dim=-1,
            ).indices

        with timed_bucket(timings, "draft_structure_time", device):
            candidate_paths = [[int(top_tokens[0, i].item())] for i in range(tree_width)]

        for branch_idx in range(tree_width):
            branch_prefix = torch.cat(
                [output_ids, _tokens_tensor(candidate_paths[branch_idx], input_ids)],
                dim=-1,
            )
            while len(candidate_paths[branch_idx]) < this_k:
                with timed_bucket(timings, "draft_generate_time", device):
                    logits = draft_model(branch_prefix).logits
                with timed_bucket(timings, "sampling_time", device):
                    next_token = _argmax_next_token(logits, suppress_token_id)
                token_id = int(next_token.item())
                with timed_bucket(timings, "draft_structure_time", device):
                    candidate_paths[branch_idx].append(token_id)
                branch_prefix = _append_token(branch_prefix, next_token)
                if eos_token_id is not None and token_id == eos_token_id:
                    break

        drafted_tokens += sum(len(path) for path in candidate_paths)

        simulate_upload_wait(
            timings,
            draft_token_count=sum(len(path) for path in candidate_paths),
            rtt_ms=rtt_ms,
            upload_token_bytes=upload_token_bytes,
            upload_bandwidth_mbps=upload_bandwidth_mbps,
        )

        max_path_len = max(len(path) for path in candidate_paths)
        padded_paths = [
            path + [path[-1]] * (max_path_len - len(path)) for path in candidate_paths
        ]
        path_tensor = torch.tensor(padded_paths, dtype=input_ids.dtype, device=device)
        batched_prefix = output_ids.repeat(len(candidate_paths), 1)
        verify_input = torch.cat([batched_prefix, path_tensor], dim=-1)
        verify_start = output_ids.shape[-1]

        with timed_bucket(timings, "target_verify_time", device):
            target_logits = target_model(verify_input).logits

        main_path = candidate_paths[0]
        with timed_bucket(timings, "sampling_time", device):
            target_tokens_for_main = []
            for i in range(len(main_path)):
                predict_pos = verify_start + i - 1
                token = _argmax_token_id(
                    target_logits[0:1, predict_pos, :],
                    suppress_token_id,
                )
                target_tokens_for_main.append(token)
            bonus_token = _argmax_token_id(
                target_logits[0:1, verify_start + len(main_path) - 1, :],
                suppress_token_id,
            )

        with timed_bucket(timings, "posterior_accept_time", device):
            accepted_this_round = 0
            replacement_token: int | None = None
            for draft_token, target_token in zip(main_path, target_tokens_for_main):
                if draft_token == target_token:
                    accepted_this_round += 1
                    if eos_token_id is not None and draft_token == eos_token_id:
                        break
                else:
                    replacement_token = target_token
                    break

        new_tokens = main_path[:accepted_this_round]
        accepted_tokens += accepted_this_round
        generated_so_far = output_ids.shape[-1] - prompt_len
        can_add_more = generated_so_far + len(new_tokens) < max_new_tokens
        hit_eos = eos_token_id is not None and eos_token_id in new_tokens

        if can_add_more and not hit_eos:
            if replacement_token is not None:
                new_tokens.append(replacement_token)
            elif accepted_this_round == len(main_path):
                new_tokens.append(bonus_token)

        new_tokens = new_tokens[: max_new_tokens - generated_so_far]
        if not new_tokens:
            new_tokens = [target_tokens_for_main[0]]

        wasted_this_round = sum(len(path) for path in candidate_paths) - accepted_this_round
        wasted_branch_tokens += wasted_this_round
        timings.wasted_branch_time_or_tokens += float(wasted_this_round)

        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = torch.cat([output_ids, _tokens_tensor(new_tokens, input_ids)], dim=-1)

        if eos_token_id is not None and eos_token_id in new_tokens:
            break

    _sync_if_cuda(device)
    timings.total_decode_time = time.perf_counter() - total_start
    return DecodeResult(
        output_ids=output_ids.squeeze(0).tolist(),
        timings=timings,
        generated_tokens=output_ids.shape[-1] - prompt_len,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        rounds=rounds,
        extra={
            "tree_width": tree_width,
            "wasted_branch_tokens": wasted_branch_tokens,
        },
    )
