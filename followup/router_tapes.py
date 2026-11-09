"""Authenticated captured-token banks and explicitly constructed replay tapes.

There is no synthetic activation or logical-route generation here. Replay only
selects complete captured token records. Temporal regimes are constructions and
must not be described as natural serving traces. Distinct resampling seeds are
not independent source-document blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BANK_ARRAYS = ("hidden_states_bf16", "expert_ids", "combining_weights", "token_ids",
               "prompt_index", "token_position", "domain_ids")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256(canonical_json(metadata).encode())
    for name, array in sorted(arrays.items()):
        a = np.ascontiguousarray(array)
        h.update(canonical_json([name, a.dtype.str, list(a.shape)]).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _save(path: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite authenticated artifact: {path}")
    target = path.with_suffix(path.suffix + ".tmp")
    with target.open("wb") as f:
        np.savez_compressed(f, **arrays, metadata=np.asarray(canonical_json(metadata)),
                            content_sha256=np.asarray(content_sha256(metadata, arrays)))
    target.replace(path)


def _load(path: Path, names: tuple[str, ...]) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        arrays = {name: z[name].copy() for name in names}
        if content_sha256(metadata, arrays) != str(z["content_sha256"]):
            raise ValueError("Artifact content hash mismatch")
    for a in arrays.values():
        a.setflags(write=False)
    return metadata, arrays


def validate_bank(metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    if metadata.get("schema") != "qwen_router_bank_v1":
        raise ValueError("Unsupported bank schema")
    if metadata.get("source_model") != MODEL or metadata.get("source_revision") != REVISION:
        raise ValueError("Unexpected source checkpoint")
    if metadata.get("layer") != 0:
        raise ValueError("Only actual layer 0 is supported")
    ids, weights, hidden = (arrays[k] for k in ("expert_ids", "combining_weights", "hidden_states_bf16"))
    if ids.dtype != np.int64 or weights.dtype != np.float32 or hidden.dtype != np.uint16:
        raise ValueError("Unexpected bank tensor dtype")
    if ids.ndim != 2 or weights.shape != ids.shape or hidden.ndim != 2 or len(hidden) != len(ids):
        raise ValueError("Invalid captured tensor shape")
    if ids.shape[1] != metadata["top_k"] or hidden.shape[1] != metadata["hidden_size"]:
        raise ValueError("Shape differs from capture metadata")
    if len(ids) == 0 or (ids < 0).any() or (ids >= metadata["num_experts"]).any():
        raise ValueError("Invalid expert IDs")
    if (np.diff(np.sort(ids, axis=1), axis=1) == 0).any():
        raise ValueError("Repeated logical expert within a token")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid combining weights")
    # Native router rounds normalized scores to BF16. Never renormalize them.
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=0.004):
        raise ValueError("Combining weights violate native BF16 normalization tolerance")
    for name in ("token_ids", "prompt_index", "token_position", "domain_ids"):
        if arrays[name].shape != (len(ids),) or arrays[name].dtype.kind not in "iu":
            raise ValueError(f"Invalid provenance vector {name}")
    if set(np.unique(arrays["domain_ids"])) != {0, 1}:
        raise ValueError("A block must contain both source domains")


@dataclass(frozen=True)
class RouterBank:
    path: Path
    metadata: dict
    hidden_states_bf16: np.ndarray
    expert_ids: np.ndarray
    combining_weights: np.ndarray
    token_ids: np.ndarray
    prompt_index: np.ndarray
    token_position: np.ndarray
    domain_ids: np.ndarray
    sha256: str


@dataclass(frozen=True)
class RouterTape:
    path: Path
    bank: RouterBank
    indices: np.ndarray
    metadata: dict
    sha256: str

    @property
    def expert_ids(self) -> np.ndarray:
        return self.bank.expert_ids[self.indices]

    @property
    def combining_weights(self) -> np.ndarray:
        return self.bank.combining_weights[self.indices]


def save_router_bank(path: str | Path, metadata: dict, arrays: dict) -> RouterBank:
    validate_bank(metadata, arrays)
    _save(Path(path), metadata, {k: arrays[k] for k in BANK_ARRAYS})
    return load_router_bank(path)


def load_router_bank(path: str | Path) -> RouterBank:
    path = Path(path)
    metadata, arrays = _load(path, BANK_ARRAYS)
    validate_bank(metadata, arrays)
    return RouterBank(path, metadata, **arrays, sha256=file_sha256(path))


def compose_router_tape(bank_path: str | Path, output: str | Path, *, regime: str,
                        batch_tokens: int, num_steps: int = 672, warmup_steps: int = 32,
                        phase_steps: int = 120, duration_steps: int = 16,
                        repeat_steps: int = 1024, seed: int = 471) -> RouterTape:
    """Sample captured records without replacement within each constructed step.

    The same random positions are drawn regardless of the selected domain, so
    different regimes have identical arrays until their domain schedule changes.
    Equal domain sizes are required. All positions start at zero within a source
    prompt and retain that original position even when replayed in another batch.
    """
    names = {"stable": "stationary", "transient": "burst", "persistent": "persistent", "switching": "periodic",
             "stationary": "stationary", "burst": "burst", "periodic": "periodic"}
    if regime not in names:
        raise ValueError("Unknown constructed regime")
    schedule_regime = names[regime]
    for name, value, lower in (("batch_tokens", batch_tokens, 1), ("num_steps", num_steps, 1),
                               ("warmup_steps", warmup_steps, 0), ("phase_steps", phase_steps, 0),
                               ("duration_steps", duration_steps, 1), ("repeat_steps", repeat_steps, 1),
                               ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"Invalid {name}")
    if repeat_steps <= duration_steps:
        raise ValueError("Burst repetition must leave a reversion interval")
    bank = load_router_bank(bank_path)
    pools = [np.flatnonzero(bank.domain_ids == domain) for domain in (0, 1)]
    if len(pools[0]) != len(pools[1]) or batch_tokens > len(pools[0]):
        raise ValueError("Equal-sized domain banks with at least batch_tokens records are required")
    t = np.arange(num_steps); onset = warmup_steps + phase_steps
    if schedule_regime == "stationary":
        domains = np.zeros(num_steps, dtype=np.int8)
    elif schedule_regime == "burst":
        domains = ((t >= onset) & ((t-onset) % repeat_steps < duration_steps)).astype(np.int8)
    elif schedule_regime == "persistent":
        domains = (t >= onset).astype(np.int8)
    else:
        domains = ((t >= onset) & (((t-onset)//duration_steps) % 2 == 0)).astype(np.int8)
    rng = np.random.default_rng(seed)
    indices = np.stack([pools[d][rng.choice(len(pools[0]), size=batch_tokens, replace=False)] for d in domains])
    uses = np.bincount(indices.ravel(), minlength=len(bank.domain_ids))
    output = Path(output)
    # Relative path makes a copied banks/tapes directory portable.
    import os
    metadata = {
        "schema": "qwen_router_replay_v1", "block_id": bank.metadata["block_id"],
        "split": bank.metadata["split"], "regime": regime, "batch_tokens": batch_tokens,
        "source_prompt_ids": bank.metadata["source_prompt_ids"],
        "source_model": MODEL, "source_revision": REVISION, "layer": 0,
        "capture_sha256": bank.sha256, "bank_file": os.path.relpath(bank.path.resolve(), output.parent.resolve()),
        "num_steps": num_steps, "warmup_steps": warmup_steps, "phase_steps": phase_steps,
        "duration_steps": duration_steps, "repeat_steps": repeat_steps, "seed": seed,
        "origin": "constructed_repeated_prefill_replay", "natural_temporal_demand": False,
        "sampling": "without_replacement_within_step; complete captured records reused across steps",
        "independent_unit": "source_document_pair; resampling seeds are not independent units",
        "domain_schedule": domains.tolist(), "token_use_counts": uses.tolist(),
        "unique_captured_records_used": int((uses > 0).sum()), "total_replayed_tokens": int(indices.size),
    }
    _save(output, metadata, {"indices": indices.astype(np.int64)})
    return load_router_tape(output)


def load_router_tape(path: str | Path, bank_path: str | Path | None = None) -> RouterTape:
    path = Path(path)
    metadata, arrays = _load(path, ("indices",))
    if metadata.get("schema") != "qwen_router_replay_v1" or metadata.get("natural_temporal_demand") is not False:
        raise ValueError("Unsupported or incorrectly labelled replay")
    bank = load_router_bank(bank_path if bank_path is not None else path.parent / metadata["bank_file"])
    if bank.sha256 != metadata["capture_sha256"]:
        raise ValueError("Capture file hash mismatch")
    for key in ("block_id", "split", "source_prompt_ids", "source_model", "source_revision", "layer"):
        if metadata[key] != bank.metadata[key]:
            raise ValueError(f"Replay provenance mismatch: {key}")
    indices = arrays["indices"]
    if indices.dtype != np.int64 or indices.shape != (metadata["num_steps"], metadata["batch_tokens"]):
        raise ValueError("Invalid replay index shape")
    if (indices < 0).any() or (indices >= len(bank.domain_ids)).any():
        raise ValueError("Replay index outside capture")
    uses = np.bincount(indices.ravel(), minlength=len(bank.domain_ids))
    if uses.tolist() != metadata["token_use_counts"]:
        raise ValueError("Replay reuse accounting mismatch")
    if not np.all(bank.domain_ids[indices] == np.asarray(metadata["domain_schedule"])[:, None]):
        raise ValueError("Replay domain schedule mismatch")
    if (np.diff(np.sort(indices, axis=1), axis=1) == 0).any():
        raise ValueError("Repeated captured record within one replay step")
    return RouterTape(path, bank, indices, metadata, file_sha256(path))
