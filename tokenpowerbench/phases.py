"""Validated host timing events for isolated single-request inference."""

from __future__ import annotations

import math


class PhaseTimingError(RuntimeError):
    """The observed engine output cannot establish a usable phase boundary."""


PHASE_LIMITATIONS = (
    "First-token boundaries are observed on the host after engine.step returns; "
    "tokenization, queueing, scheduler, host, and output-processing overhead are included.",
    "Prefill is a proxy interval, not exact GPU kernel attribution.",
    "Dispatch completion is not GPU execution start; a background engine may "
    "start executing before add_request returns.",
    "Engine preemption and recomputation are not observable through this interface; "
    "phase intervals cannot certify their absence.",
)


def build_phase_event(
    *,
    request_id: str,
    submitted_s: float,
    dispatch_completed_s: float,
    first_token_s: float,
    finished_s: float,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Build an event using timestamps from the same monotonic host clock.

    TTFT and the prefill proxy cover the same submission-to-first-token window.
    Submission work is included because a background engine can begin executing
    before dispatch returns. No timestamp here identifies GPU execution start.
    Decode begins at the first observed token. One-token requests may have zero
    decode duration.
    """
    times = (submitted_s, dispatch_completed_s, first_token_s, finished_s)
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times):
        raise PhaseTimingError("Phase timestamps must be finite numbers.")
    if not submitted_s <= dispatch_completed_s <= first_token_s <= finished_s:
        raise PhaseTimingError("Phase timestamps are not in causal order.")
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0:
        raise PhaseTimingError("Input token count must be a nonnegative integer.")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 1:
        raise PhaseTimingError("Output token count must be a positive integer.")
    return {
        "request_id": request_id,
        "submitted_s": submitted_s,
        "dispatch_completed_s": dispatch_completed_s,
        "prefill_start_s": submitted_s,
        "first_token_s": first_token_s,
        "finished_s": finished_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_s": first_token_s - submitted_s,
        "prefill_proxy_s": first_token_s - submitted_s,
        "decode_s": finished_s - first_token_s,
        "request_latency_s": finished_s - submitted_s,
        "timing_source": "host_engine_step",
        "clock": "time.perf_counter",
        "phase_boundary_kind": "host_observed_first_token",
        "limitations": list(PHASE_LIMITATIONS),
    }
