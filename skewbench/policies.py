"""Pure admission rules; these functions never move or commit expert weights.

Maps are flat physical-slot -> logical-expert arrays. Contiguous equal-sized
blocks belong to ranks 0, ..., world_size - 1. The load model assumes equal
sharing among replicas, which must be checked against actual dispatch.

``native`` adds no filter to the native planner. B0/B1's different replica
budgets are runner configuration, not distinct admission rules here.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

AdmissionMode = Literal["native", "improvement", "stability", "cooldown", "skip"]


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    tv: float | None = None
    predicted_gain: float | None = None


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _counts(values, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric counts") from exc
    if result.ndim != 1 or not result.size:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite nonnegative counts")
    if not np.isfinite(result.sum()):
        raise ValueError(f"{name} total must be finite")
    return result


def _map(values, num_experts: int, world_size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not result.size or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty integer vector")
    if result.size % world_size:
        raise ValueError(f"{name} must have equal contiguous slots per rank")
    if np.any(result < 0) or np.any(result >= num_experts):
        raise ValueError(f"{name} contains an invalid logical expert ID")
    result = result.astype(np.int64, copy=False)
    if np.any(np.bincount(result, minlength=num_experts) == 0):
        raise ValueError(f"{name} must represent every logical expert")
    return result


def _completeness(value) -> tuple[bool, bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value), bool(value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(v, (bool, np.bool_)) for v in value
    ):
        return bool(value[0]), bool(value[1])
    raise ValueError("windows_complete must be a bool or (previous, current) bool pair")


def total_variation(current_counts, previous_counts) -> float:
    """Empirical TV of two nonzero logical assignment-count vectors."""
    current = _counts(current_counts, "current_counts")
    previous = _counts(previous_counts, "previous_counts")
    if current.shape != previous.shape:
        raise ValueError("count vectors must have the same shape")
    if current.sum() == 0 or previous.sum() == 0:
        raise ValueError("TV is undefined for a zero-total window")
    return float(0.5 * np.abs(current / current.sum() - previous / previous.sum()).sum())


def predicted_rank_loads(counts, physical_to_logical, world_size: int) -> np.ndarray:
    """Normalized rank loads under equal sharing; sum is one for nonzero input."""
    world_size = _integer("world_size", world_size, 1)
    counts = _counts(counts, "counts")
    mapping = _map(physical_to_logical, counts.size, world_size, "physical_to_logical")
    if counts.sum() == 0:
        return np.zeros(world_size, dtype=np.float64)
    copies = np.bincount(mapping, minlength=counts.size)
    slot_load = (counts / counts.sum())[mapping] / copies[mapping]
    return slot_load.reshape(world_size, -1).sum(axis=1)


def evaluate_admission(
    mode: AdmissionMode,
    current_counts,
    previous_counts,
    current_map,
    proposed_map,
    world_size: int,
    epsilon: float = 0.1,
    tau: float = 0.05,
    step: int = 0,
    last_accepted_step: int | None = None,
    cooldown_steps: int = 0,
    windows_complete: bool | tuple[bool, bool] = True,
) -> Decision:
    """Evaluate a proposed map without mutating inputs or controller state.

    ``windows_complete`` can be one bool or ``(previous, current)``. Only the
    stability rule needs two complete, nonzero windows. Improvement/cooldown
    require the current window only. ``native`` and ``skip`` add no observation
    requirement. ``last_accepted_step`` records accepted proposals, not merely
    attempted proposals; updating it is the runner's responsibility.

    TV and gain are balance diagnostics, not predicted latency or payback.
    Native implementation guards remain separate and must be logged by the
    runner. Map budgets must match within a candidate comparison.
    """
    if mode not in {"native", "improvement", "stability", "cooldown", "skip"}:
        raise ValueError(f"unknown admission mode: {mode!r}")
    world_size = _integer("world_size", world_size, 1)
    step = _integer("step", step)
    cooldown_steps = _integer("cooldown_steps", cooldown_steps)
    if last_accepted_step is not None:
        last_accepted_step = _integer("last_accepted_step", last_accepted_step)
        if last_accepted_step > step:
            raise ValueError("last_accepted_step cannot be in the future")
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be finite and in (0, 1]")
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must be finite and in [0, 1]")
    previous_complete, current_complete = _completeness(windows_complete)
    counts = _counts(current_counts, "current_counts")
    old_map = _map(current_map, counts.size, world_size, "current_map")
    new_map = _map(proposed_map, counts.size, world_size, "proposed_map")
    if old_map.shape != new_map.shape:
        raise ValueError("current and proposed maps must have the same physical-slot budget")
    if mode == "native":
        return Decision(True, "native")
    if mode == "skip":
        return Decision(False, "always_skip")
    if not current_complete:
        return Decision(False, "incomplete_current_window")
    if counts.sum() == 0:
        return Decision(False, "zero_current_load")

    old_peak = float(predicted_rank_loads(counts, old_map, world_size).max())
    new_peak = float(predicted_rank_loads(counts, new_map, world_size).max())
    gain = (old_peak - new_peak) / old_peak
    tv = None
    if mode == "stability":
        if not previous_complete or previous_counts is None:
            return Decision(False, "incomplete_previous_window", predicted_gain=gain)
        previous = _counts(previous_counts, "previous_counts")
        if previous.shape != counts.shape:
            raise ValueError("count vectors must have the same shape")
        if previous.sum() == 0:
            return Decision(False, "zero_previous_load", predicted_gain=gain)
        tv = total_variation(counts, previous)
    if gain < epsilon:
        return Decision(False, "below_improvement_threshold", tv, gain)
    if mode == "stability" and tv > tau:
        return Decision(False, "unstable_demand", tv, gain)
    if mode == "cooldown" and last_accepted_step is not None and step - last_accepted_step < cooldown_steps:
        return Decision(False, "cooldown", predicted_gain=gain)
    return Decision(True, "allowed", tv, gain)
