from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any

import torch


TIME_BUCKETS = [
    "prefill_time",
    "draft_generate_time",
    "upload_wait_time",
    "target_verify_time",
    "posterior_accept_time",
    "kv_or_input_update_time",
    "sampling_time",
]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class DecodeTimings:
    prefill_time: float = 0.0
    draft_generate_time: float = 0.0
    upload_wait_time: float = 0.0
    target_verify_time: float = 0.0
    posterior_accept_time: float = 0.0
    kv_or_input_update_time: float = 0.0
    sampling_time: float = 0.0
    total_decode_time: float = 0.0

    def add(self, bucket: str, seconds: float) -> None:
        setattr(self, bucket, getattr(self, bucket) + seconds)

    def as_dict(self) -> dict[str, float]:
        return {
            "prefill_time": self.prefill_time,
            "draft_generate_time": self.draft_generate_time,
            "upload_wait_time": self.upload_wait_time,
            "target_verify_time": self.target_verify_time,
            "posterior_accept_time": self.posterior_accept_time,
            "kv_or_input_update_time": self.kv_or_input_update_time,
            "sampling_time": self.sampling_time,
            "total_decode_time": self.total_decode_time,
        }


@contextmanager
def timed_bucket(timings: DecodeTimings, bucket: str, device: torch.device):
    _sync_if_cuda(device)
    start = time.perf_counter()
    yield
    _sync_if_cuda(device)
    timings.add(bucket, time.perf_counter() - start)


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


def _argmax_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _append_token(input_ids: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    return torch.cat([input_ids, token.to(input_ids.device)], dim=-1)


@torch.inference_mode()
def target_only_greedy(
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
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
            next_token = _argmax_next_token(logits)
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
def speculative_greedy(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_k: int = 4,
    rtt_ms: float = 0.0,
    eos_token_id: int | None = None,
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
                draft_next = _argmax_next_token(draft_logits)
            token_id = int(draft_next.item())
            draft_tokens.append(token_id)
            drafted_tokens += 1
            with timed_bucket(timings, "kv_or_input_update_time", device):
                draft_prefix = _append_token(draft_prefix, draft_next)
            if eos_token_id is not None and token_id == eos_token_id:
                break

        if rtt_ms > 0:
            start = time.perf_counter()
            time.sleep(rtt_ms / 1000.0)
            timings.upload_wait_time += time.perf_counter() - start

        draft_tensor = torch.tensor([draft_tokens], dtype=input_ids.dtype, device=device)
        verify_input = torch.cat([output_ids, draft_tensor], dim=-1)
        verify_start = output_ids.shape[-1]

        with timed_bucket(timings, "target_verify_time", device):
            target_logits = target_model(verify_input).logits

        with timed_bucket(timings, "sampling_time", device):
            target_tokens_for_draft = []
            for i in range(len(draft_tokens)):
                predict_pos = verify_start + i - 1
                token = int(torch.argmax(target_logits[:, predict_pos, :], dim=-1).item())
                target_tokens_for_draft.append(token)

            bonus_token = int(torch.argmax(target_logits[:, -1, :], dim=-1).item())

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
