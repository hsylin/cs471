"""Deterministic controlled logical routing, never synthetic timing results.

Each token samples unique experts by Gumbel top-k (weighted sampling without
replacement). Zipf probabilities are sampling parameters: for top_k > 1 they
are not the marginal expert-inclusion frequencies. Combining weights are the
softmax of the selected perturbed scores and sum to one per token.

Generate a trace once and reuse it for every paired policy. Arrays are marked
read-only to catch accidental mutation; preserve token order when sharding.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

REGIMES = ("stationary", "uniform", "burst", "persistent", "periodic")


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def split_seed(split: str, index: int, base_seed: int = 471) -> int:
    """Disjoint integer seed domains for pilot/final, for a shared base seed."""
    if split not in {"pilot", "final"}:
        raise ValueError("split must be 'pilot' or 'final'")
    index = _integer("index", index)
    base_seed = _integer("base_seed", base_seed)
    if index >= 2**32:
        raise ValueError("index must be less than 2**32")
    return (base_seed << 33) | ((split == "final") << 32) | index


@dataclass(frozen=True)
class TraceConfig:
    seed: int
    num_experts: int
    top_k: int
    tokens_per_step: int
    num_steps: int
    window_steps: int
    regime: str
    hotspot_lifetime_steps: int | None = None
    phase_steps: int = 0
    zipf_exponent: float = 1.2
    warmup_steps: int | None = None
    burst_repeat_steps: int | None = None

    def __post_init__(self):
        for name in ("seed", "phase_steps"):
            object.__setattr__(self, name, _integer(name, getattr(self, name)))
        for name in ("num_experts", "top_k", "tokens_per_step", "num_steps", "window_steps"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), 1))
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if not np.isfinite(self.zipf_exponent) or self.zipf_exponent < 0:
            raise ValueError("zipf_exponent must be finite and nonnegative")
        if self.hotspot_lifetime_steps is not None:
            object.__setattr__(self, "hotspot_lifetime_steps", _integer("hotspot_lifetime_steps", self.hotspot_lifetime_steps, 1))
        if self.warmup_steps is not None:
            object.__setattr__(self, "warmup_steps", _integer("warmup_steps", self.warmup_steps))
        if self.burst_repeat_steps is not None:
            object.__setattr__(self, "burst_repeat_steps", _integer("burst_repeat_steps", self.burst_repeat_steps, 1))
            if self.burst_repeat_steps <= self.effective_hotspot_lifetime_steps:
                raise ValueError("burst_repeat_steps must exceed hotspot lifetime to leave a reversion interval")

    @property
    def effective_warmup_steps(self) -> int:
        return 2 * self.window_steps if self.warmup_steps is None else self.warmup_steps

    @property
    def effective_hotspot_lifetime_steps(self) -> int:
        return self.window_steps if self.hotspot_lifetime_steps is None else self.hotspot_lifetime_steps


@dataclass(frozen=True)
class RoutingTrace:
    config: TraceConfig
    expert_ids: np.ndarray
    combining_weights: np.ndarray
    logical_counts: np.ndarray
    probabilities: np.ndarray
    hotspot_states: np.ndarray

    @property
    def metadata(self) -> dict:
        return {
            "seed": self.config.seed,
            "regime": self.config.regime,
            "config": asdict(self.config),
            "generator": "gumbel_top_k_v1",
            "tokens_scope": "global_layer",
        }

    def counts(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Sum logical assignments in the half-open engine-step interval."""
        start = _integer("start", start)
        stop = self.config.num_steps if stop is None else _integer("stop", stop)
        if not start <= stop <= self.config.num_steps:
            raise ValueError("counts interval must satisfy 0 <= start <= stop <= num_steps")
        return self.logical_counts[start:stop].sum(axis=0)

    def fingerprint(self) -> str:
        """Hash configuration and actual replay IDs/weights, not timing output."""
        digest = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode())
        digest.update(memoryview(self.expert_ids).cast("B"))
        digest.update(memoryview(self.combining_weights).cast("B"))
        return digest.hexdigest()


def hotspot_schedule(config: TraceConfig) -> np.ndarray:
    """0 is base Zipf order, 1 rotates expert popularity by E // 2.

    A change starts at warmup_steps + phase_steps. A burst reverts after one
    lifetime. If burst_repeat_steps is set, subsequent bursts begin that many
    steps apart; the field is ignored for other regimes. Persistent demand
    never reverts; periodic demand alternates at each lifetime. Two common
    warmup windows are the default, not evidence that sampled distributions
    are identical or a promise that the runner excludes that prefix from timing.
    """
    steps = np.arange(config.num_steps, dtype=np.int64)
    start = config.effective_warmup_steps + config.phase_steps
    lifetime = config.effective_hotspot_lifetime_steps
    if config.regime == "burst":
        if config.burst_repeat_steps is None:
            state = (steps >= start) & (steps < start + lifetime)
        else:
            state = (steps >= start) & ((steps - start) % config.burst_repeat_steps < lifetime)
    elif config.regime == "persistent":
        state = steps >= start
    elif config.regime == "periodic":
        state = (steps >= start) & (((steps - start) // lifetime) % 2 == 0)
    else:
        state = np.zeros(config.num_steps, dtype=bool)
    return state.astype(np.int8)


def generate_trace(config: TraceConfig) -> RoutingTrace:
    """Materialize a replay trace with unique logical top-k choices per token."""
    rng = np.random.default_rng(config.seed)
    shape = (config.num_steps, config.tokens_per_step, config.top_k)
    expert_ids = np.empty(shape, dtype=np.int64)
    combining_weights = np.empty(shape, dtype=np.float32)
    logical_counts = np.empty((config.num_steps, config.num_experts), dtype=np.int64)
    probabilities = np.empty((config.num_steps, config.num_experts), dtype=np.float64)
    states = hotspot_schedule(config)
    log_base = -config.zipf_exponent * np.log(np.arange(1, config.num_experts + 1, dtype=np.float64))
    if not np.all(np.isfinite(log_base)):
        raise ValueError("zipf_exponent produces non-finite log probabilities")
    if config.regime == "uniform":
        log_base.fill(0)
    for step in range(config.num_steps):
        log_probs = np.roll(log_base, config.num_experts // 2) if states[step] else log_base
        probs = np.exp(log_probs - log_probs.max())
        probabilities[step] = probs / probs.sum()
        scores = log_probs[None, :] + rng.gumbel(size=(config.tokens_per_step, config.num_experts))
        chosen = np.argpartition(scores, config.num_experts - config.top_k, axis=1)[:, -config.top_k:]
        chosen_scores = np.take_along_axis(scores, chosen, axis=1)
        order = np.argsort(-chosen_scores, axis=1, kind="stable")
        chosen = np.take_along_axis(chosen, order, axis=1)
        chosen_scores = np.take_along_axis(chosen_scores, order, axis=1)
        weights = np.exp(chosen_scores - chosen_scores.max(axis=1, keepdims=True))
        expert_ids[step] = chosen
        combining_weights[step] = weights / weights.sum(axis=1, keepdims=True)
        logical_counts[step] = np.bincount(chosen.ravel(), minlength=config.num_experts)
    for array in (expert_ids, combining_weights, logical_counts, probabilities, states):
        array.setflags(write=False)
    return RoutingTrace(config, expert_ids, combining_weights, logical_counts, probabilities, states)
