"""Fresh source selection for stock-vLLM comparison; no GPU/runtime imports."""
from collections import defaultdict
import hashlib
import json
import re
import unicodedata

DATASETS = ("gsm8k", "mbpp", "wildchat")
CONTENT_SEED = 471601
ARRIVAL_SEED = 471602


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text):
    words = re.findall(r"\w+|[^\w\s]", normalized(text))
    return {tuple(words[i:i+5]) for i in range(max(1, len(words)-4))}


class ViewIndex:
    def __init__(self, threshold=.8):
        if not 0 < threshold <= 1:
            raise ValueError("Jaccard threshold must be in (0,1]")
        self.threshold = threshold
        self.exact = {}
        self.inverted = defaultdict(set)
        self.views = []

    def add(self, text, identity):
        text = normalized(text)
        h = digest(text.encode())
        if h in self.exact:
            return
        s = shingles(text)
        i = len(self.views)
        self.views.append((s, identity))
        self.exact[h] = identity
        for gram in s:
            self.inverted[gram].add(i)

    def match(self, text):
        text = normalized(text)
        h = digest(text.encode())
        if h in self.exact:
            return dict(reason="normalized_exact", duplicate_of=self.exact[h], jaccard=1.)
        s = shingles(text)
        candidates = set()
        for gram in s:
            candidates.update(self.inverted.get(gram, ()))
        for i in sorted(candidates):
            other, identity = self.views[i]
            if min(len(s), len(other)) < self.threshold * max(len(s), len(other)):
                continue
            score = len(s & other) / len(s | other)
            if score >= self.threshold:
                return dict(reason="near_duplicate", duplicate_of=identity, jaccard=score)
        return None


def exclude_and_deduplicate(candidates, exclusions, threshold=.8):
    """Prior references all remain searchable, even if mutually near-duplicate.

    Each reference supplies full_text and one or more model-visible prefix
    views. No prior reference is discarded by greedy deduplication. New
    candidates are filtered before splitting and before any capture/timing.
    """
    full, prefix = ViewIndex(threshold), ViewIndex(threshold)
    ids, whole_source_hashes = set(), set()
    for ref in exclusions:
        identity = [ref["dataset"], str(ref["source_id"])]
        ids.add(tuple(identity))
        full.add(ref["text"], identity)
        for text in ref["prefix_views"]:
            prefix.add(text, identity)
        if ref.get("whole_source_normalized_sha256"):
            whole_source_hashes.add(ref["whole_source_normalized_sha256"])
    kept, dropped = [], []
    for record in candidates:
        identity = [record["dataset"], str(record["source_id"])]
        h = digest(normalized(record["text"]).encode())
        reason = None
        if tuple(identity) in ids:
            reason = dict(reason="prior_or_duplicate_source_id", duplicate_of=identity)
        elif h in whole_source_hashes:
            reason = dict(reason="prior_whole_source_exact", duplicate_of=None)
        else:
            for name, idx, view in (("full_text", full, record["text"]),
                                    ("captured_prefix", prefix, record["captured_prefix_text"])):
                match = idx.match(view)
                if match:
                    reason = {**match, "view": name}
                    break
        ids.add(tuple(identity))
        if reason:
            dropped.append(dict(dataset=identity[0], source_id=identity[1], **reason))
            continue
        result = {k: v for k, v in record.items() if k != "captured_prefix_text"}
        result.update(text_sha256=digest(record["text"].encode()), normalized_text_sha256=h,
                      captured_prefix_normalized_sha256=digest(normalized(record["captured_prefix_text"]).encode()))
        kept.append(result)
        full.add(record["text"], identity)
        prefix.add(record["captured_prefix_text"], identity)
    return kept, dropped


def split_sources(records, *, seed=CONTENT_SEED, pilot_blocks=2, final_blocks=6, documents_per_block=32):
    if min(pilot_blocks, final_blocks, documents_per_block) < 1:
        raise ValueError("Positive block/document counts required")
    blocks = []
    for dataset in DATASETS:
        rows = sorted((r for r in records if r["dataset"] == dataset),
                      key=lambda r: (digest([seed, dataset, r["source_id"], r["text_sha256"]]), r["source_id"]))
        need = (pilot_blocks + final_blocks) * documents_per_block
        if len(rows) < need:
            raise ValueError(f"{dataset}: {len(rows)} fresh documents, need {need}; no reuse or silent smaller split")
        cursor = 0
        for split, number in (("pilot", pilot_blocks), ("final", final_blocks)):
            for i in range(number):
                selected = rows[cursor:cursor+documents_per_block]
                cursor += documents_per_block
                blocks.append(dict(block_id=f"stock-{dataset}-{split}-{i+1:02d}", dataset=dataset, split=split, sources=selected))
    seen, hashes = set(), set()
    for block in blocks:
        for row in block["sources"]:
            key = (row["dataset"], row["source_id"])
            if key in seen or row["normalized_text_sha256"] in hashes:
                raise ValueError("Repeated source across fresh blocks")
            seen.add(key)
            hashes.add(row["normalized_text_sha256"])
    return blocks


def fresh_arrival_candidates(candidates, previous_blocks, *, trace, seed=ARRIVAL_SEED, per_day=8, jobs=4096):
    """Choose fresh contiguous jobs+boundary bins by identifier hash only.

    Source calendar dates are reused. Include the boundary row in overlap
    exclusion. Invalid chosen horizons fail; do not search for favorable gaps.
    """
    prior = [(b["row_start"], b["boundary_row"] + 1) for b in previous_blocks if b["trace"] == trace]
    by_day, rejected = defaultdict(list), 0
    for row in candidates:
        if row["trace"] != trace or row["boundary_row"] != row["row_start"] + jobs:
            raise ValueError("Wrong source/bin contract")
        lo, hi = row["row_start"], row["boundary_row"] + 1
        if any(lo < b and a < hi for a, b in prior):
            rejected += 1
            continue
        rank = digest([seed, trace, row["source_date_utc"], lo])
        by_day[row["source_date_utc"]].append({**row, "stock_hash_rank_score": rank})
    selected = []
    for day in sorted(by_day):
        candidates_for_day = sorted(by_day[day], key=lambda r: (r["stock_hash_rank_score"], r["row_start"]))
        if len(candidates_for_day) < per_day:
            raise ValueError("Insufficient fresh bins on " + day)
        for row in candidates_for_day[:per_day]:
            if row["original_horizon_ns"] <= 0 or row["terminal_gap_ns"] <= 0:
                raise ValueError("Hash-selected fresh bin has invalid horizon/terminal gap; explicit design amendment required")
        selected.extend(candidates_for_day[:per_day])
    return selected, dict(excluded_prior_overlap=rejected,
                          eligible_bins_by_date={d: len(v) for d, v in sorted(by_day.items())},
                          fresh_calendar_dates=False, fresh_source_rows=True)


def scan_fresh_arrivals(path, *, trace, expected_sha256, previous_blocks, jobs=4096, per_day=4,
                       seed=ARRIVAL_SEED, progress=None):
    """Stream all source rows; retain bounded hash-selected eligible bins/day."""
    import csv
    import heapq
    import io
    from generalization.external_arrivals import _HashingReader, parse_timestamp, _candidate

    excluded = [(b["row_start"], b["boundary_row"]+1) for b in previous_blocks if b["trace"] == trace]
    raw = _HashingReader(path)
    cache, days, heaps = {}, {}, {}
    previous_ns, active_day, chunk, count = None, None, [], 0
    with io.TextIOWrapper(io.BufferedReader(raw, buffer_size=1024*1024), encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
            raise ValueError("Unexpected arrival CSV schema")
        for i, row in enumerate(reader):
            ns, day, _ = parse_timestamp(row["TIMESTAMP"], cache)
            if previous_ns is not None and ns < previous_ns:
                raise ValueError("Arrival inversion; no sorting or repair")
            previous_ns = ns
            context, generated = int(row["ContextTokens"]), int(row["GeneratedTokens"])
            if context < 0 or generated < 0:
                raise ValueError("Negative source token-count metadata")
            if day != active_day:
                if active_day is not None:
                    days[active_day]["discarded_partial_rows"] = len(chunk)
                    if progress:
                        progress(trace, active_day, days[active_day])
                if day in days:
                    raise ValueError("Noncontiguous source day")
                days[day] = dict(rows=0, candidate_bins=0, excluded_prior_overlap=0, eligible_bins=0)
                heaps[day], active_day, chunk = [], day, []
            count += 1
            days[day]["rows"] += 1
            chunk.append(dict(row_index=i, timestamp_ns=ns, raw_timestamp=row["TIMESTAMP"],
                              context_tokens=context, generated_tokens=generated))
            if len(chunk) != jobs+1:
                continue
            days[day]["candidate_bins"] += 1
            lo, hi = chunk[0]["row_index"], i+1
            if any(lo < b and a < hi for a, b in excluded):
                days[day]["excluded_prior_overlap"] += 1
            else:
                days[day]["eligible_bins"] += 1
                block = _candidate(trace, day, chunk, seed, jobs)
                item = (-int(block["hash_rank_score"], 16), -lo, block)
                heap = heaps[day]
                if len(heap) < per_day:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            chunk = []
        if active_day is not None:
            days[active_day]["discarded_partial_rows"] = len(chunk)
            if progress:
                progress(trace, active_day, days[active_day])
    if raw.sha.hexdigest() != expected_sha256:
        raise ValueError("Pinned full arrival source bytes changed")
    retained = {d: sorted((x[2] for x in heaps[d]), key=lambda b: (b["hash_rank_score"], b["row_start"]))
                for d in sorted(days)}
    if len(days) != 7 or any(len(v) < per_day for v in retained.values()):
        raise ValueError("Expected seven days with enough fresh bins")
    return dict(trace=trace, source_sha256=raw.sha.hexdigest(), source_bytes=raw.bytes_read, rows=count,
                jobs=jobs, boundary_rows_per_block=1, seed=seed, days=days, retained_candidates=retained,
                fresh_calendar_dates=False, source_rows_disjoint_from_prior=True)


def assign_fresh_arrivals(scans, blocks, *, seed=ARRIVAL_SEED):
    """Exactly one fresh code and conversation interval per content group."""
    result = []
    for scan in sorted(scans, key=lambda x: x["trace"]):
        days = sorted(scan["retained_candidates"])
        used = defaultdict(int)
        for split, available in (("pilot", days[:2]), ("final", days[2:])):
            for di, dataset in enumerate(DATASETS):
                groups = sorted((b for b in blocks if b["dataset"] == dataset and b["split"] == split),
                                key=lambda b: (digest([seed, "assignment", scan["trace"], b["block_id"]]), b["block_id"]))
                for j, group in enumerate(groups):
                    rotation = 0 if scan["trace"] == "code" else 2
                    day = available[(j + di*len(groups) + rotation) % len(available)]
                    position = used[day]
                    candidates = scan["retained_candidates"][day]
                    if position >= len(candidates):
                        raise ValueError("Insufficient date-stratified candidate capacity")
                    block = candidates[position]
                    used[day] += 1
                    if block["original_horizon_ns"] <= 0 or block["terminal_gap_ns"] <= 0:
                        raise ValueError("Chosen source interval has zero terminal gap; no silent replacement")
                    result.append({**block, "id": f"stock-azure-{scan['trace']}-{day}-{position+1:02d}",
                                   "content_group_id": group["block_id"], "dataset": dataset, "split": split,
                                   "source_sha256": scan["source_sha256"], "calendar_date_cluster": day})
    if len(result) != len(blocks)*2:
        raise ValueError("Missing assigned arrivals")
    for trace in ("code", "conversation"):
        intervals = sorted((b["row_start"], b["boundary_row"]+1) for b in result if b["trace"] == trace)
        if any(b > c for (_, b), (c, _) in zip(intervals, intervals[1:])):
            raise ValueError("New arrival blocks overlap")
    return result
