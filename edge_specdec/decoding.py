from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import time
from typing import Any

import torch


SuppressTokenIds = int | list[int] | set[int] | tuple[int, ...] | None


TIME_BUCKETS = [
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
    uplink_transfer_time: float = 0.0
    network_wait_time: float = 0.0
    downlink_transfer_time: float = 0.0
    downlink_payload_bytes: float = 0.0
    cloud_verify_time: float = 0.0
    target_verify_time: float = 0.0
    posterior_accept_time: float = 0.0
    cache_update_time: float = 0.0
    kv_or_input_update_time: float = 0.0
    probability_normalize_time: float = 0.0
    random_sample_time: float = 0.0
    accept_reject_time: float = 0.0
    resample_time: float = 0.0
    sampling_time: float = 0.0
    wasted_branch_time_or_tokens: float = 0.0
    total_decode_time: float = 0.0

    def add(self, bucket: str, seconds: float) -> None:
        if bucket == "kv_or_input_update_time":
            self.cache_update_time += seconds
        if bucket == "target_verify_time":
            self.cloud_verify_time += seconds
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
            "uplink_transfer_time": self.uplink_transfer_time,
            "network_wait_time": self.network_wait_time,
            "downlink_transfer_time": self.downlink_transfer_time,
            "downlink_payload_bytes": self.downlink_payload_bytes,
            "cloud_verify_time": self.cloud_verify_time,
            "target_verify_time": self.target_verify_time,
            "posterior_accept_time": self.posterior_accept_time,
            "cache_update_time": self.cache_update_time,
            "kv_or_input_update_time": self.kv_or_input_update_time,
            "probability_normalize_time": self.probability_normalize_time,
            "random_sample_time": self.random_sample_time,
            "accept_reject_time": self.accept_reject_time,
            "resample_time": self.resample_time,
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


def _transfer_seconds(payload_bytes: int, bandwidth_mbps: float) -> float:
    if bandwidth_mbps <= 0 or payload_bytes <= 0:
        return 0.0
    return (payload_bytes * 8.0) / (bandwidth_mbps * 1_000_000.0)


def _sleep_and_measure(seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    start = time.perf_counter()
    time.sleep(seconds)
    return time.perf_counter() - start


def simulate_upload_wait(
    timings: DecodeTimings,
    draft_token_count: int,
    rtt_ms: float,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    uplink_bandwidth_mbps: float | None = None,
) -> None:
    """Simulate the request-side edge-to-cloud exchange for draft tokens.

    rtt_ms is treated as the fixed round-trip network wait budget, excluding
    transfer and target compute. The request side consumes half of that wait;
    the response side consumes the other half in simulate_downlink_wait.
    """

    bandwidth_mbps = (
        upload_bandwidth_mbps if uplink_bandwidth_mbps is None else uplink_bandwidth_mbps
    )
    payload_bytes = max(0, draft_token_count) * max(0, upload_token_bytes)
    network_wait_seconds = max(0.0, rtt_ms) / 2000.0
    transfer_seconds = _transfer_seconds(payload_bytes, bandwidth_mbps)

    timings.upload_payload_bytes += float(payload_bytes)
    timings.upload_latency_time += network_wait_seconds
    timings.upload_transfer_time += transfer_seconds
    timings.uplink_transfer_time += transfer_seconds
    timings.network_wait_time += network_wait_seconds

    wait_seconds = network_wait_seconds + transfer_seconds
    _sleep_and_measure(wait_seconds)
    timings.upload_wait_time += wait_seconds


def simulate_downlink_wait(
    timings: DecodeTimings,
    rtt_ms: float,
    response_token_count: int = 1,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
) -> None:
    """Simulate the response-side cloud-to-edge verification result transfer."""

    payload_bytes = max(0, downlink_fixed_bytes) + max(0, response_token_count) * max(
        0,
        downlink_token_bytes,
    )
    network_wait_seconds = max(0.0, rtt_ms) / 2000.0
    transfer_seconds = _transfer_seconds(payload_bytes, downlink_bandwidth_mbps)

    timings.downlink_payload_bytes += float(payload_bytes)
    timings.downlink_transfer_time += transfer_seconds
    timings.network_wait_time += network_wait_seconds
    timings.upload_latency_time += network_wait_seconds

    wait_seconds = network_wait_seconds + transfer_seconds
    _sleep_and_measure(wait_seconds)
    timings.upload_wait_time += wait_seconds


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


def _suppressed_token_ids(token_ids: SuppressTokenIds) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, int):
        return [token_ids]
    return [int(token_id) for token_id in token_ids]


def _mask_token(logits: torch.Tensor, token_id: SuppressTokenIds) -> torch.Tensor:
    token_ids = _suppressed_token_ids(token_id)
    if not token_ids:
        return logits
    masked_logits = logits.clone()
    masked_logits[..., token_ids] = torch.finfo(masked_logits.dtype).min
    return masked_logits


def _argmax_token(logits: torch.Tensor, suppress_token_id: SuppressTokenIds = None) -> torch.Tensor:
    logits = _mask_token(logits, suppress_token_id)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _argmax_token_id(logits: torch.Tensor, suppress_token_id: SuppressTokenIds = None) -> int:
    return int(_argmax_token(logits, suppress_token_id).item())


def _argmax_next_token(
    logits: torch.Tensor,
    suppress_token_id: SuppressTokenIds = None,
) -> torch.Tensor:
    return _argmax_token(logits[:, -1, :], suppress_token_id)


def _filter_logits_top_k_top_p(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    if top_k and top_k > 0:
        kth_values = torch.topk(filtered, min(top_k, vocab_size), dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))
    if top_p and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        filtered = filtered.masked_fill(remove, float("-inf"))
    return filtered


def _normalize_logits_for_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    suppress_token_id: SuppressTokenIds = None,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for stochastic sampling")
    filtered = _mask_token(logits.float() / temperature, suppress_token_id)
    filtered = _filter_logits_top_k_top_p(filtered, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sums = probs.sum(dim=-1, keepdim=True)
    if bool((sums <= 0).any()):
        fallback = torch.ones_like(probs)
        suppressed = _suppressed_token_ids(suppress_token_id)
        if suppressed:
            fallback[:, suppressed] = 0.0
        probs = torch.where(sums > 0, probs, fallback)
        sums = probs.sum(dim=-1, keepdim=True)
    return probs / sums.clamp_min(torch.finfo(probs.dtype).tiny)


def _sample_from_probs(
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _residual_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_sum = residual.sum(dim=-1, keepdim=True)
    return torch.where(
        residual_sum > 0,
        residual / residual_sum.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs,
    )


def _make_sampling_generator(
    device: torch.device,
    seed: int | None,
) -> torch.Generator | None:
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


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


def _cached_forward_target_only_step(
    model,
    token: torch.Tensor,
    past_key_values: Any,
    total_seq_len: int,
):
    attention_mask = token.new_ones((token.shape[0], total_seq_len))
    return model(
        token,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )


def _cached_verify_sequential(
    target_model,
    draft_tokens: list[int],
    target_next_logits: torch.Tensor,
    target_past: Any,
    input_ids: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, Any]:
    """Verify draft tokens with the same one-step cache path as target-only."""

    verify_past = _clone_past_key_values(target_past)
    next_logits = target_next_logits
    logits_for_draft: list[torch.Tensor] = []
    prefix_len = _past_seq_len(target_past)
    for index, token_id in enumerate(draft_tokens):
        logits_for_draft.append(next_logits)
        outputs = _cached_forward_target_only_step(
            target_model,
            _tokens_tensor([token_id], input_ids),
            verify_past,
            prefix_len + index + 1,
        )
        verify_past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
    return logits_for_draft, next_logits, verify_past


@torch.inference_mode()
def target_only_greedy(
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    suppress_token_id: SuppressTokenIds = None,
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
    suppress_token_id: SuppressTokenIds = None,
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
def target_only_sampling_cached(
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    suppress_token_id: SuppressTokenIds = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
) -> DecodeResult:
    """Autoregressive target sampling with KV cache.

    This is the stochastic baseline for probability-based speculative
    sampling. It samples from the target model distribution after applying the
    same temperature/top-k/top-p transform used by speculative verification.
    """

    device = input_ids.device
    generator = _make_sampling_generator(device, seed)
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
        with timed_bucket(timings, "probability_normalize_time", device):
            probs = _normalize_logits_for_sampling(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                suppress_token_id=suppress_token_id,
            )
        with timed_bucket(timings, "random_sample_time", device):
            next_token = _sample_from_probs(probs, generator=generator)

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
        extra={
            "sampling": "target_only",
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
        },
    )


@torch.inference_mode()
def traditional_speculative_sampling_cached(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_k: int = 4,
    rtt_ms: float = 0.0,
    eos_token_id: int | None = None,
    upload_token_bytes: int = 4,
    upload_bandwidth_mbps: float = 0.0,
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    target_verify_mode: str = "batch",
    suppress_token_id: SuppressTokenIds = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
) -> DecodeResult:
    """Probability-based Google-style speculative sampling with KV cache.

    Draft tokens are sampled from q, verified against target probabilities p,
    accepted with min(1, p/q), and replaced from normalize(max(p - q, 0)) on
    rejection. When all draft tokens are accepted, the bonus token is sampled
    from the target distribution after the draft span. The batch verification
    path mirrors feifeibear/LLMSpeculativeSampling's KVCacheModel.generate(x, 1)
    prob-history layout: target position prefix_len+i-1 scores draft token i.
    """

    if draft_k < 1:
        raise ValueError("draft_k must be >= 1")
    if target_verify_mode not in {"batch", "sequential"}:
        raise ValueError("target_verify_mode must be 'batch' or 'sequential'")

    device = input_ids.device
    generator = _make_sampling_generator(device, seed)
    timings = DecodeTimings()
    output_ids = input_ids.clone()
    prompt_len = output_ids.shape[-1]
    accepted_tokens = 0
    drafted_tokens = 0
    rejected_tokens = 0
    resample_count = 0
    bonus_sample_count = 0
    proposal_accept_prob_sum = 0.0
    proposal_accept_prob_count = 0
    first_accept_prob_sum = 0.0
    first_accept_prob_count = 0
    first_token_accepted_count = 0
    target_zero_at_draft_count = 0
    rounds = 0

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
        this_k = min(draft_k, remaining)

        draft_tokens: list[int] = []
        draft_probs_for_tokens: list[torch.Tensor] = []
        provisional_draft_past = _clone_past_key_values(draft_past)
        provisional_draft_next_logits = draft_next_logits

        for i in range(this_k):
            with timed_bucket(timings, "probability_normalize_time", device):
                draft_probs = _normalize_logits_for_sampling(
                    provisional_draft_next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    suppress_token_id=suppress_token_id,
                )
            with timed_bucket(timings, "random_sample_time", device):
                draft_next = _sample_from_probs(draft_probs, generator=generator)
            token_id = int(draft_next.item())
            draft_tokens.append(token_id)
            draft_probs_for_tokens.append(draft_probs)
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
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        )

        if target_verify_mode == "batch":
            draft_tensor = _tokens_tensor(draft_tokens, input_ids)
            with timed_bucket(timings, "target_verify_time", device):
                verify_outputs = _cached_forward(
                    target_model,
                    draft_tensor,
                    _clone_past_key_values(target_past),
                )
            verify_logits = verify_outputs.logits
            target_logits_for_draft = [target_next_logits]
            for i in range(1, len(draft_tokens)):
                target_logits_for_draft.append(verify_logits[:, i - 1, :])
            bonus_logits = verify_logits[:, -1, :]
        else:
            with timed_bucket(timings, "target_verify_time", device):
                (
                    target_logits_for_draft,
                    bonus_logits,
                    _verify_past_after_draft,
                ) = _cached_verify_sequential(
                    target_model,
                    draft_tokens,
                    target_next_logits,
                    target_past,
                    input_ids,
                )

        simulate_downlink_wait(
            timings,
            rtt_ms=rtt_ms,
            response_token_count=1,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        )

        with timed_bucket(timings, "probability_normalize_time", device):
            target_probs_for_draft = [
                _normalize_logits_for_sampling(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    suppress_token_id=suppress_token_id,
                )
                for logits in target_logits_for_draft
            ]
            bonus_probs = _normalize_logits_for_sampling(
                bonus_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                suppress_token_id=suppress_token_id,
            )

        accepted_this_round = 0
        replacement_token: int | None = None
        reject_index: int | None = None
        with timed_bucket(timings, "accept_reject_time", device):
            for i, draft_token in enumerate(draft_tokens):
                target_prob = target_probs_for_draft[i][0, draft_token]
                draft_prob = draft_probs_for_tokens[i][0, draft_token]
                if float(target_prob.item()) <= 0.0:
                    target_zero_at_draft_count += 1
                if float(draft_prob.item()) <= 0.0:
                    accept_prob = torch.ones((), dtype=target_prob.dtype, device=device)
                else:
                    accept_prob = torch.clamp(target_prob / draft_prob, max=1.0)
                accept_prob_value = float(accept_prob.item())
                proposal_accept_prob_sum += accept_prob_value
                proposal_accept_prob_count += 1
                if i == 0:
                    first_accept_prob_sum += accept_prob_value
                    first_accept_prob_count += 1
                accept_draw = torch.rand((), device=device, generator=generator)
                if bool(accept_draw <= accept_prob):
                    accepted_this_round += 1
                    if eos_token_id is not None and draft_token == eos_token_id:
                        break
                else:
                    reject_index = i
                    rejected_tokens += 1
                    break

        new_tokens = draft_tokens[:accepted_this_round]
        accepted_tokens += accepted_this_round
        if accepted_this_round > 0:
            first_token_accepted_count += 1

        generated_so_far = output_ids.shape[-1] - prompt_len
        can_add_more = generated_so_far + len(new_tokens) < max_new_tokens
        hit_eos = eos_token_id is not None and eos_token_id in new_tokens

        if can_add_more and not hit_eos:
            if reject_index is not None:
                with timed_bucket(timings, "resample_time", device):
                    residual_probs = _residual_distribution(
                        target_probs_for_draft[reject_index],
                        draft_probs_for_tokens[reject_index],
                    )
                with timed_bucket(timings, "random_sample_time", device):
                    replacement = _sample_from_probs(residual_probs, generator=generator)
                replacement_token = int(replacement.item())
                resample_count += 1
                new_tokens.append(replacement_token)
            elif accepted_this_round == len(draft_tokens):
                with timed_bucket(timings, "random_sample_time", device):
                    bonus_token = _sample_from_probs(bonus_probs, generator=generator)
                bonus_sample_count += 1
                new_tokens.append(int(bonus_token.item()))

        new_tokens = new_tokens[: max_new_tokens - generated_so_far]
        if not new_tokens:
            with timed_bucket(timings, "random_sample_time", device):
                fallback = _sample_from_probs(target_probs_for_draft[0], generator=generator)
            new_tokens = [int(fallback.item())]

        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = torch.cat([output_ids, _tokens_tensor(new_tokens, input_ids)], dim=-1)
            attention_mask = torch.ones_like(output_ids)
            previous_len = output_ids.shape[-1] - len(new_tokens)
            for index, token_id in enumerate(new_tokens):
                target_outputs = _cached_forward_target_only_step(
                    target_model,
                    _tokens_tensor([token_id], input_ids),
                    target_past,
                    previous_len + index + 1,
                )
                target_past = target_outputs.past_key_values
                target_next_logits = target_outputs.logits[:, -1, :]
            draft_outputs = draft_model(
                output_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            draft_past = draft_outputs.past_key_values
            draft_next_logits = draft_outputs.logits[:, -1, :]

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
            "sampling": "traditional_speculative",
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "gamma": draft_k,
            "target_verify_mode": target_verify_mode,
            "upstream_reference": (
                "feifeibear/LLMSpeculativeSampling sampling/speculative_sampling.py"
            ),
            "rejected_tokens": rejected_tokens,
            "resample_count": resample_count,
            "bonus_sample_count": bonus_sample_count,
            "mean_checked_accept_prob": proposal_accept_prob_sum
            / proposal_accept_prob_count
            if proposal_accept_prob_count
            else 0.0,
            "mean_first_accept_prob": first_accept_prob_sum / first_accept_prob_count
            if first_accept_prob_count
            else 0.0,
            "first_token_accept_rate": first_token_accepted_count / rounds
            if rounds
            else 0.0,
            "target_zero_at_draft_count": target_zero_at_draft_count,
            "target_zero_at_draft_rate": target_zero_at_draft_count
            / proposal_accept_prob_count
            if proposal_accept_prob_count
            else 0.0,
            "state_update_mode": "target_only_step",
        },
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
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    suppress_token_id: SuppressTokenIds = None,
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
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        )

        draft_tensor = torch.tensor([draft_tokens], dtype=input_ids.dtype, device=device)
        verify_input = torch.cat([output_ids, draft_tensor], dim=-1)
        verify_start = output_ids.shape[-1]

        with timed_bucket(timings, "target_verify_time", device):
            target_logits = target_model(verify_input).logits
        simulate_downlink_wait(
            timings,
            rtt_ms=rtt_ms,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        )

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
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    min_draft_k: int | None = None,
    max_draft_k: int | None = None,
    target_verify_mode: str = "sequential",
    suppress_token_id: SuppressTokenIds = None,
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
    if target_verify_mode not in {"batch", "sequential"}:
        raise ValueError("target_verify_mode must be 'batch' or 'sequential'")
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
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        )

        if target_verify_mode == "batch":
            draft_tensor = _tokens_tensor(draft_tokens, input_ids)
            with timed_bucket(timings, "target_verify_time", device):
                verify_outputs = _cached_forward(
                    target_model,
                    draft_tensor,
                    _clone_past_key_values(target_past),
                )
            verify_logits = verify_outputs.logits
            verify_past_after_draft = verify_outputs.past_key_values
            target_logits_for_draft = [target_next_logits]
            for i in range(1, len(draft_tokens)):
                target_logits_for_draft.append(verify_logits[:, i - 1, :])
            bonus_logits = verify_logits[:, -1, :]
        else:
            with timed_bucket(timings, "target_verify_time", device):
                (
                    target_logits_for_draft,
                    bonus_logits,
                    verify_past_after_draft,
                ) = _cached_verify_sequential(
                    target_model,
                    draft_tokens,
                    target_next_logits,
                    target_past,
                    input_ids,
                )
        simulate_downlink_wait(
            timings,
            rtt_ms=rtt_ms,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        )

        with timed_bucket(timings, "sampling_time", device):
            target_tokens_for_draft = [
                _argmax_token_id(logits, suppress_token_id)
                for logits in target_logits_for_draft
            ]
            bonus_token = _argmax_token_id(bonus_logits, suppress_token_id)

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

        new_tensor = _tokens_tensor(new_tokens, input_ids)
        with timed_bucket(timings, "kv_or_input_update_time", device):
            output_ids = torch.cat([output_ids, new_tensor], dim=-1)
            attention_mask = torch.ones_like(output_ids)

            previous_len = output_ids.shape[-1] - len(new_tokens)
            for index, token_id in enumerate(new_tokens):
                target_outputs = _cached_forward_target_only_step(
                    target_model,
                    _tokens_tensor([token_id], input_ids),
                    target_past,
                    previous_len + index + 1,
                )
                target_past = target_outputs.past_key_values
                target_next_logits = target_outputs.logits[:, -1, :]
            draft_outputs = draft_model(
                output_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
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
            "target_verify_mode": target_verify_mode,
            "state_update_mode": "target_only_step",
        }
        if adaptive_draft
        else {
            "target_verify_mode": target_verify_mode,
            "state_update_mode": "target_only_step",
        },
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
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    suppress_token_id: SuppressTokenIds = None,
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
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
        )

        draft_tensor = _tokens_tensor(draft_tokens, input_ids)
        verify_input = torch.cat([output_ids, draft_tensor], dim=-1)
        verify_start = output_ids.shape[-1]

        with timed_bucket(timings, "target_verify_time", device):
            target_logits = target_model(verify_input).logits
        simulate_downlink_wait(
            timings,
            rtt_ms=rtt_ms,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        )

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
    uplink_bandwidth_mbps: float | None = None,
    downlink_token_bytes: int = 4,
    downlink_fixed_bytes: int = 4,
    downlink_bandwidth_mbps: float = 0.0,
    suppress_token_id: SuppressTokenIds = None,
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
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
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
        simulate_downlink_wait(
            timings,
            rtt_ms=rtt_ms,
            downlink_token_bytes=downlink_token_bytes,
            downlink_fixed_bytes=downlink_fixed_bytes,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps,
        )

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
