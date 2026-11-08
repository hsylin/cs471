#!/usr/bin/env python3
"""Descriptive diagnostics from completed final GPU records; no fitted model or CI.

Usage: python scripts/diagnostic_report.py --records final.jsonl \
    --branches branches.jsonl --output-dir results/diagnostics

Primary inference belongs to skewbench.analysis. Here, windows/steps from a run
are correlated observations, and selected branch states are not a population
sample. Missing records or counters are reported, never replaced by zeros.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CORE_METHODS = {"B2", "B3", "B4", "B5", "SKIP"}
METHOD_COLORS = {"B1": "#888888", "B2": "#4477aa", "B3": "#ee7733",
                 "B4": "#228833", "B5": "#aa3377", "SKIP": "#777777"}
IDENTITY = ("record_id", "method", "regime", "trace_variant", "batch_tokens", "seed", "repetition")


def read_records(path, kind):
    rows, excluded, seen = [], Counter(), {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        reasons = [name for name, valid in (
            ("not_final", row.get("split") == "final"),
            ("wrong_measurement_kind", row.get("measurement_kind") == kind),
            ("not_complete", row.get("status") == "complete"),
            ("correctness_not_passed", row.get("correctness_pass") is True),
            ("not_evidence_eligible", row.get("evidence_eligible") is True),
        ) if not valid]
        if reasons:
            excluded.update(reasons)
            continue
        key = row["record_id"]
        if key in seen:
            if row != seen[key]:
                raise ValueError(f"Conflicting duplicate record {key} at {path}:{line_number}")
            excluded["identical_duplicate"] += 1
            continue
        seen[key] = row
        rows.append(row)
    if len({r["config_hash"] for r in rows}) > 1:
        raise ValueError("Provide records from one frozen configuration per input file")
    return rows, dict(excluded)


def identity(row):
    return {key: row[key] for key in IDENTITY}


def finite(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {name}")
    return value


def save_figure(fig, directory, name):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_runs(rows):
    cases = defaultdict(list)
    stationary = defaultdict(list)
    for row in rows:
        decisions = row["decision_log"]
        payload = sum(int(d["transfer_sent_bytes"]) for d in decisions)
        if payload != int(row["migration_bytes"]):
            raise ValueError(f"Decision payloads do not sum to run payload: {row['record_id']}")
        cases[(row["method"], row["regime"], row["trace_variant"], row["batch_tokens"])].append(row)
        if row["method"] == "B4" and row["regime"] == "stationary":
            stationary[(row["batch_tokens"], row["seed"])].append(row)
    case_summary = []
    for (method, regime, variant, batch), members in sorted(cases.items()):
        decisions = [d for r in members for d in r["decision_log"]]
        case_summary.append(dict(method=method, regime=regime, trace_variant=variant,
            batch_tokens=batch, runs=len(members), seeds=sorted({r["seed"] for r in members}),
            accepted_proposals=sum(bool(d["allowed"]) for d in decisions),
            rejected_proposals=sum(not d["allowed"] for d in decisions),
            reasons=dict(Counter(d["reason"] for d in decisions)),
            decisions_with_sender_payload=sum(d["transfer_sent_bytes"] > 0 for d in decisions),
            actual_sender_bytes_total=sum(r["migration_bytes"] for r in members),
            actual_sender_bytes_per_run=[r["migration_bytes"] for r in members]))
    tv_summary = []
    for (batch, seed), members in sorted(stationary.items()):
        observed, eligible, missing = [], [], 0
        for row in members:
            tau, epsilon = row["policy_config"]["tau"], row["policy_config"]["epsilon"]
            for decision in row["decision_log"]:
                if decision.get("tv") is None:
                    missing += 1
                    continue
                tv = finite(decision["tv"], "TV")
                observed.append((tv, tau, decision["reason"]))
                if decision.get("predicted_gain") is not None and decision["predicted_gain"] >= epsilon:
                    eligible.append((tv, tau))
        values = [v[0] for v in observed]
        tv_summary.append(dict(batch_tokens=batch, seed=seed, runs=len(members),
            observed_windows=len(values), missing_tv_windows=missing,
            tau_values=sorted({v[1] for v in observed}),
            tv_median=float(np.median(values)) if values else None,
            tv_p95=float(np.quantile(values, .95)) if values else None,
            tv_max=max(values) if values else None,
            threshold_exceedances=sum(v > t for v, t, _ in observed),
            threshold_exceedance_fraction=sum(v > t for v, t, _ in observed) / len(values) if values else None,
            improvement_eligible_windows=len(eligible),
            eligible_threshold_exceedance_fraction=sum(v > t for v, t in eligible) / len(eligible) if eligible else None,
            actual_unstable_demand_rejections=sum(reason == "unstable_demand" for _, _, reason in observed)))
    return case_summary, tv_summary


def dispatch_diagnostics(rows):
    windows, steps, missing = [], [], []
    for row in rows:
        world = int(row["world_size"])
        proposal_steps = {d["step"] for d in row["decision_log"]}
        seen_steps = set()
        for decision in row["decision_log"]:
            if "physical_slot_assignments_window" not in decision:
                missing.append({**identity(row), "decision_step": decision["step"]})
                continue
            scope = decision.get("physical_counter_scope", "")
            if not scope.startswith("one copy of identical replicated-input counters"):
                raise ValueError("Unrecognized physical-counter scope; refusing to guess a divisor")
            physical = np.asarray(decision["physical_slot_assignments_window"], dtype=np.int64)
            mapping = np.asarray(decision["current_map"], dtype=np.int64)
            logical = np.asarray(decision["current_counts"], dtype=np.float64)
            if physical.ndim != 2 or physical.shape[1] != len(mapping) or len(mapping) % world:
                raise ValueError("Physical window/map/rank dimensions disagree")
            if np.any(physical < 0) or not physical.sum():
                raise ValueError("Invalid physical assignment counts")
            recovered = np.bincount(mapping, weights=physical.sum(axis=0), minlength=len(logical))
            if decision["physical_counter_input_multiplicity"] != world or not np.array_equal(recovered * world, logical):
                raise ValueError("Deduplicated physical counts do not recover logical observations")
            copies = np.bincount(mapping, minlength=len(logical))
            if np.any(copies == 0):
                raise ValueError("Old map omits a logical expert")
            p = logical / logical.sum()
            predicted = (p[mapping] / copies[mapping]).reshape(world, -1).sum(axis=1)
            actual = physical.reshape(len(physical), world, -1).sum(axis=(0, 2)) / physical.sum()
            windows.append({**identity(row), "decision_step": decision["step"],
                "predicted_rank_load_share": predicted.tolist(), "actual_rank_load_share": actual.tolist(),
                "max_absolute_share_error": float(np.abs(actual - predicted).max()),
                "predicted_rank_imbalance": float(predicted.max() * world),
                "actual_rank_imbalance": float(actual.max() * world)})
            first, last = decision["physical_window_measured_step_range"]
            if last - first + 1 != len(physical) or last != decision["step"]:
                raise ValueError("Physical-window chronology does not match proposal step")
            times = row.get("step_seconds_slowest_rank")
            if times is None or row["method"] not in CORE_METHODS:
                continue
            owner = physical.reshape(len(physical), world, -1).sum(axis=2)
            activated = (physical > 0).reshape(len(physical), world, -1).sum(axis=2)
            for offset, measured_step in enumerate(range(first, last + 1)):
                if measured_step in proposal_steps or measured_step in seen_steps:
                    continue
                if not 0 <= measured_step < len(times) or owner[offset].sum() == 0:
                    raise ValueError("Invalid physical-window step/timing alignment")
                seen_steps.add(measured_step)
                steps.append({**identity(row), "step": measured_step,
                    "rank_token_imbalance": float(owner[offset].max() / owner[offset].mean()),
                    "maximum_activated_experts_on_rank": int(activated[offset].max()),
                    "elapsed_ms": 1000 * finite(times[measured_step], "step duration")})
    return windows, steps, missing


def branch_diagnostics(rows):
    points = []
    for row in rows:
        arms = {arm["branch"]: arm for arm in row["branches"]}
        if set(arms) != {"allow", "skip"} or len(row["branches"]) != 2:
            raise ValueError("Branch record must contain exactly one allow and one skip arm")
        for arm in arms.values():
            if arm.get("correctness_pass") is not True or arm["future_steps"] != row["future_steps"]:
                raise ValueError("Branch correctness/horizon mismatch")
            for key in ("source_map_sha256", "candidate_map_sha256", "source_weights_sha256", "trace_sha256"):
                if not row.get(key) or arm.get(key) != row[key]:
                    raise ValueError(f"Branch pair has inconsistent {key}")
        if row.get("future_rebalancing_enabled") is not False:
            raise ValueError("Branch replay must disable subsequent rebalancing")
        gate = row["gate_decision"]
        points.append({**identity(row), "decision_step": row["decision_step"],
            "future_steps": row["future_steps"], "tv": finite(gate["tv"], "branch TV"),
            "tau": finite(row["policy_config"]["tau"], "tau"),
            "predicted_gain": finite(gate["predicted_gain"], "predicted gain"),
            "b3_eligible": bool(row["b3_would_allow"]), "b4_allowed": bool(gate["allowed"]),
            "b4_reason": gate["reason"],
            "allow_ms": 1000 * finite(arms["allow"]["elapsed_seconds"], "allow duration"),
            "skip_ms": 1000 * finite(arms["skip"]["elapsed_seconds"], "skip duration"),
            "payback_ms": 1000 * (arms["skip"]["elapsed_seconds"] - arms["allow"]["elapsed_seconds"]),
            "allow_sender_bytes": arms["allow"]["migration_bytes"]})
    if len({p["future_steps"] for p in points}) > 1:
        raise ValueError("Branch comparisons require one fixed future horizon")
    return points


def grouped_description(points, metrics):
    groups = defaultdict(list)
    for point in points:
        groups[(point["method"], point["regime"], point["trace_variant"], point["batch_tokens"])].append(point)
    result = []
    for (method, regime, variant, batch), members in sorted(groups.items()):
        row = dict(method=method, regime=regime, trace_variant=variant, batch_tokens=batch,
                   observations=len(members), independent_seed_count=len({p["seed"] for p in members}))
        for metric in metrics:
            values = [p[metric] for p in members]
            row[metric] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                               p95=float(np.quantile(values, .95)), maximum=max(values))
        result.append(row)
    return result


def make_plots(directory, tv, windows, steps, branches):
    created = []
    if any(p["observed_windows"] for p in tv):
        fig, ax = plt.subplots(figsize=(7, 4))
        batches = sorted({p["batch_tokens"] for p in tv})
        for batch in batches:
            group = [p for p in tv if p["batch_tokens"] == batch and p["threshold_exceedance_fraction"] is not None]
            ax.plot([p["seed"] for p in group], [p["threshold_exceedance_fraction"] for p in group],
                    "o", label=f"{batch} tokens/step")
        ax.set(xlabel="Independent trace seed", ylabel="Stationary windows with TV > tau / observed windows",
               title="Stationary stability threshold exceedances (descriptive)", ylim=(-.03, 1.03))
        ax.legend(); fig.tight_layout()
        save_figure(fig, directory, "stationary-tv"); created.append("stationary-tv")
    if windows:
        fig, ax = plt.subplots(figsize=(6, 5))
        for method in sorted({p["method"] for p in windows}):
            group = [p for p in windows if p["method"] == method]
            ax.scatter([p["predicted_rank_imbalance"] for p in group],
                       [p["actual_rank_imbalance"] for p in group], s=11, alpha=.35,
                       color=METHOD_COLORS.get(method), label=method, rasterized=True)
        limit = max(max(p["predicted_rank_imbalance"], p["actual_rank_imbalance"]) for p in windows)
        ax.plot([1, limit], [1, limit], "k--", linewidth=1)
        ax.set(xlabel="Equal-sharing predicted max / mean rank load",
               ylabel="Recorded physical max / mean rank load",
               title="Equal-sharing approximation against actual dispatch")
        ax.legend(); fig.tight_layout()
        save_figure(fig, directory, "replica-load-model"); created.append("replica-load-model")
    if steps:
        batches = sorted({p["batch_tokens"] for p in steps})
        fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4), squeeze=False)
        for ax, batch in zip(axes[0], batches):
            for method in sorted(CORE_METHODS):
                group = [p for p in steps if p["batch_tokens"] == batch and p["method"] == method]
                if group:
                    ax.scatter([p["rank_token_imbalance"] for p in group], [p["elapsed_ms"] for p in group],
                               s=6, alpha=.2, color=METHOD_COLORS[method], label=method, rasterized=True)
            ax.set(xlabel="Recorded step max / mean rank token load", ylabel="Completed device-event step time (ms)",
                   title=f"{batch} tokens/step; proposal steps excluded")
            ax.legend(markerscale=2)
        fig.suptitle("Token imbalance and runtime: correlated steps, descriptive only")
        fig.tight_layout()
        save_figure(fig, directory, "token-balance-runtime"); created.append("token-balance-runtime")
    if branches:
        fig, ax = plt.subplots(figsize=(8, 5))
        regimes = sorted({p["regime"] for p in branches})
        colors = dict(zip(regimes, plt.get_cmap("tab10").colors))
        batches = sorted({p["batch_tokens"] for p in branches})
        markers = dict(zip(batches, ("o", "s", "^", "D", "v", "P", "X")))
        for p in branches:
            ax.scatter(p["tv"], p["payback_ms"], s=75, marker=markers[p["batch_tokens"]],
                       facecolors=colors[p["regime"]] if p["b3_eligible"] else "none",
                       edgecolors=colors[p["regime"]], linewidths=1.5)
            ax.annotate(f"s{p['seed']} d{p['decision_step']}", (p["tv"], p["payback_ms"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        for tau in sorted({p["tau"] for p in branches}):
            ax.axvline(tau, color="#777777", linestyle="--", label=f"tau = {tau:g}")
        ax.axhline(0, color="black", linewidth=1)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color=colors[r], marker="o", linestyle="none", label=r) for r in regimes]
        handles += [Line2D([], [], color="black", marker=markers[b], linestyle="none", label=f"{b} tokens") for b in batches]
        handles += [Line2D([], [], color="black", marker="o", markerfacecolor="black" if eligible else "none",
                           linestyle="none", label="B3 eligible" if eligible else "B3 ineligible")
                    for eligible in sorted({p["b3_eligible"] for p in branches}, reverse=True)]
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.set(xlabel="Adjacent-window total variation", ylabel="Y = skip time − allow time (ms); positive favors migration",
               title=f"Same-state migration payback: {len(branches)} pairs, H={branches[0]['future_steps']} steps")
        fig.text(.5, .01, "Selected diagnostic states; planning and prefix excluded; no population or causal policy claim", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        save_figure(fig, directory, "branch-tv-payback"); created.append("branch-tv-payback")
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--branches", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, excluded = read_records(args.records, "gpu")
    if not rows:
        raise SystemExit("No complete, correct, evidence-eligible final GPU records; no report created")
    branch_rows, branch_excluded = read_records(args.branches, "gpu_branch") if args.branches else ([], {})
    if branch_rows and {r["config_hash"] for r in branch_rows} != {r["config_hash"] for r in rows}:
        raise ValueError("Primary and branch records must share the frozen configuration")
    cases, tv = summarize_runs(rows)
    windows, steps, missing = dispatch_diagnostics(rows)
    branches = branch_diagnostics(branch_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = make_plots(args.output_dir, tv, windows, steps, branches)
    summary = {
        "scope": "Descriptive diagnostics only; primary inference is unchanged. Windows and steps are correlated; branch states are selected diagnostics.",
        "records_path": str(args.records.resolve()), "records_file_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "branches_path": str(args.branches.resolve()) if args.branches else None,
        "config_hash": rows[0]["config_hash"], "eligible_final_runs": len(rows),
        "excluded_record_reasons": excluded, "excluded_branch_reasons": branch_excluded,
        "case_admission_and_payload": cases, "stationary_tv_by_batch_seed": tv,
        "stationary_tv_note": "TV > tau under stationary sampling is a false stability alarm; it is an actual extra rejection only when improvement-eligible. Fractions are descriptive, not independent-trial estimates.",
        "replica_load_window_count": len(windows),
        "replica_load_model_by_case": grouped_description(windows, ("max_absolute_share_error", "predicted_rank_imbalance", "actual_rank_imbalance")),
        "largest_replica_model_errors": sorted(windows, key=lambda p: p["max_absolute_share_error"], reverse=True)[:10],
        "missing_physical_windows": missing,
        "physical_counter_note": "Counters are already deduplicated to one copy; no further world-size division. Logical current_counts have world-size multiplicity. Old map defines ownership.",
        "step_point_count": len(steps),
        "step_descriptions_by_case": grouped_description(steps, ("rank_token_imbalance", "maximum_activated_experts_on_rank", "elapsed_ms")),
        "step_note": "Only recorded current-window steps, B2/B3/B4/B5/SKIP, proposal steps removed. Device-event durations include instrumented work and are not the primary wall-clock endpoint. No regression or inferential correlation test.",
        "branch_points": branches,
        "branch_note": "Y>0 favors migration. TV and eligible/ineligible labels are observed values; small selected samples and timing noise prevent a population classification claim.",
        "plots": [str(args.output_dir / f"{name}.{suffix}") for name in plots for suffix in ("png", "pdf")],
    }
    (args.output_dir / "diagnostic-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"final_runs": len(rows), "branch_pairs": len(branches), "plots": plots,
                      "summary": str(args.output_dir / "diagnostic-summary.json")}))


if __name__ == "__main__":
    main()
