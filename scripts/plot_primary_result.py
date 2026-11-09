"""Compact display of the five frozen acceptance conditions; no new estimands."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if (summary["split"] != "final" or summary["present_records"] != 1400
            or summary["expected_records"] != 1400 or summary["missing_records"]
            or summary["invalid_records"]):
        raise ValueError("This figure requires the complete, valid frozen final matrix")
    aggregate = {(r["regime"], r["reference"]): r for r in summary["regime_aggregates"]}
    cells = {(r["regime"], r["trace_variant"], r["batch_tokens"], r["reference"]): r
             for r in summary["cells"]}
    rows = [
        ("Burst / improvement-only", aggregate[("burst", "B3")]),
        ("Burst / cooldown", aggregate[("burst", "B5")]),
        ("Stable / improvement-only", aggregate[("stationary", "B3")]),
        ("Stable / 128 tokens", cells[("stationary", "core", 128, "B3")]),
        ("Stable / 1,024 tokens", cells[("stationary", "core", 1024, "B3")]),
    ]
    colors = {"pass": "#21734B", "fail": "#B33C36", "inconclusive": "#A66A13"}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for y, (label, row) in enumerate(rows):
        ratio, (lo, hi), limit = row["ratio"], row["ci95"], row["ratio_limit"]
        if not row["complete"] or row["independent_seeds"] != 10 or limit is None:
            raise ValueError("Incomplete primary endpoint")
        ax.errorbar(ratio, y, xerr=[[ratio-lo], [hi-ratio]], fmt="o", ms=6,
                    capsize=4, color=colors[row["verdict"]], zorder=3)
        ax.plot([limit, limit], [y-.22, y+.22], color="#34495E", lw=2)
        ax.text(1.03, y, f"{ratio:.4f} [{lo:.4f}, {hi:.4f}]  {row['verdict']}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.axvline(1, color="#777777", lw=1, linestyle="--", zorder=1)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set_ylim(len(rows)-.5, -.6)
    values = [value for _, row in rows for value in (*row["ci95"], row["ratio_limit"])] + [1.]
    padding = max(.004, (max(values)-min(values))*.12)
    ax.set_xlim(min(values)-padding, max(values)+padding)
    ax.set_xlabel("B4 / reference completed-time ratio (lower is faster)")
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does the TV gate meet the frozen performance criteria?", fontsize=13, y=.97)
    fig.text(.02, .03, "Bars: 95% seed-block bootstrap intervals, 10 independent seeds.  "
             "Short vertical marks: upper-CI limits (burst 0.95; stable 1.02).", fontsize=8)
    fig.subplots_adjust(left=.26, right=.65, top=.85, bottom=.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"primary-runtime-ratios.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
