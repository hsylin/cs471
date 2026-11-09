"""Independent complete-data ratio/CI verification; never imports the analyzer."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for name in ("manifest", "records", "report", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text()); summary = json.loads(a.report.read_text())
    digest = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "freeze"}, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert digest == m["freeze"]["sha256"] == summary["manifest_sha256"] and m["split"] == "final"
    matrix, plan = m["matrix"], m["analysis"]
    expected = set(itertools.product(matrix["methods"],
        [(c["regime"], c["trace_variant"]) for c in matrix["cases"]], matrix["batch_tokens"],
        matrix["seeds"], range(matrix["repetitions"])))
    rows = {}
    for line in a.records.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        assert r["config_hash"] == digest and r["split"] == "final"
        assert r["status"] == "complete" and r["measurement_kind"] == "gpu"
        assert r["correctness_pass"] is True and r["evidence_eligible"] is True
        assert r["harness_sha256"] == m["environment"]["harness_sha256"] and r["policy_config"] == m["policy"]
        k = (r["method"], (r["regime"], r["trace_variant"]), r["batch_tokens"], r["seed"], r["repetition"])
        assert k in expected and k not in rows and r["elapsed_seconds"] > 0
        rows[k] = r
    assert set(rows) == expected, "Do not estimate before the entire frozen matrix completes"
    assert not summary["missing_records"] and not summary["invalid_records"]
    assert summary["present_records"] == summary["expected_records"] == len(expected)
    n = len(matrix["seeds"]); rng = random.Random(plan["bootstrap_seed"])
    # Independent numpy estimator; reproduce the declared Python RNG index stream.
    indexes = np.floor(np.array([rng.random() for _ in range(n * plan["bootstrap_samples"])]) * n).astype(int).reshape(-1, n)
    def estimate(logs):
        return float(np.exp(np.mean(logs))), np.quantile(np.exp(logs[indexes].mean(axis=1)), [.025, .975])
    cell_logs, differences = {}, []
    by_cell = {(r["reference"], r["regime"], r["trace_variant"], r["batch_tokens"]): r for r in summary["cells"]}
    for ref, case, batch in itertools.product(plan["references"], matrix["cases"], matrix["batch_tokens"]):
        pair = (case["regime"], case["trace_variant"]); values = []
        for seed in sorted(matrix["seeds"]):
            repeats = []
            for repetition in range(matrix["repetitions"]):
                candidate = rows[("B4", pair, batch, seed, repetition)]
                baseline = rows[(ref, pair, batch, seed, repetition)]
                assert candidate["trace_sha256"] == baseline["trace_sha256"]
                assert candidate["weights_source"]["weights_sha256"] == baseline["weights_source"]["weights_sha256"]
                repeats.append(np.log(candidate["elapsed_seconds"] / baseline["elapsed_seconds"]))
            values.append(np.mean(repeats))
        key = (ref, *pair, batch); cell_logs[key] = np.asarray(values)
        ratio, ci = estimate(cell_logs[key]); reported = by_cell[key]
        difference = max(abs(ratio - reported["ratio"]), float(np.max(np.abs(ci - reported["ci95"]))))
        assert difference < 1e-12 and reported["complete"] and reported["independent_seeds"] == n
        differences.append(difference)
    by_aggregate = {(r["reference"], r["regime"]): r for r in summary["regime_aggregates"]}
    aggregates = []
    for ref, regime in itertools.product(plan["references"], dict.fromkeys(c["regime"] for c in matrix["cases"])):
        logs = np.mean([value for (r, g, _, _), value in cell_logs.items() if (r, g) == (ref, regime)], axis=0)
        ratio, ci = estimate(logs); reported = by_aggregate[(ref, regime)]
        difference = max(abs(ratio - reported["ratio"]), float(np.max(np.abs(ci - reported["ci95"]))))
        assert difference < 1e-12; differences.append(difference)
        aggregates.append({"reference": ref, "regime": regime, "ratio": ratio, "ci95": ci.tolist()})
    result = {"status": "verified", "complete_records": len(rows), "independent_seeds": n,
        "cell_estimates_checked": len(cell_logs), "regime_estimates_checked": len(aggregates),
        "maximum_absolute_numerical_difference": max(differences), "aggregates": aggregates,
        "scope": "Independent arithmetic and declared seed-block bootstrap verification, not an independent experiment or a coverage guarantee."}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "aggregates"}))


if __name__ == "__main__": main()
