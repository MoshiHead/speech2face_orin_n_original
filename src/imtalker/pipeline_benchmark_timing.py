"""Opt-in CUDA timing helpers for the live pipeline benchmark.

CUDA kernels are asynchronous, so host ``perf_counter`` intervals around a
model call are not valid device-stage measurements unless a later operation
synchronizes the stream.  This module records a sequence of CUDA events and
resolves them only when benchmark timing is explicitly enabled.

Production behavior is unchanged unless ``IMTALKER_BENCHMARK_TIMING`` is set
to a truthy value.
"""

from __future__ import annotations

import os
from typing import Any

import torch


_TRUTHY = {"1", "true", "yes", "on"}


def benchmark_timing_enabled() -> bool:
    return os.environ.get("IMTALKER_BENCHMARK_TIMING", "0").strip().lower() in _TRUTHY


class CudaEventTimeline:
    """Record named adjacent intervals on one CUDA stream."""

    def __init__(self, device: torch.device | str, *, enabled: bool | None = None) -> None:
        self.device = torch.device(device)
        requested = benchmark_timing_enabled() if enabled is None else bool(enabled)
        self.enabled = requested and self.device.type == "cuda" and torch.cuda.is_available()
        self._marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, completed_stage: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._marks.append((str(completed_stage), event))

    def finish(self) -> dict[str, Any]:
        if not self.enabled or len(self._marks) < 2:
            return {}
        self._marks[-1][1].synchronize()
        stages: dict[str, float] = {}
        for (_, left), (right_name, right) in zip(self._marks[:-1], self._marks[1:]):
            stages[f"{right_name}_cuda_ms"] = float(left.elapsed_time(right))
        total = float(self._marks[0][1].elapsed_time(self._marks[-1][1]))
        return {
            "cuda_stages_ms": stages,
            "cuda_total_ms": total,
        }

