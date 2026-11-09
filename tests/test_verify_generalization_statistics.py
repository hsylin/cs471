"""Synthetic CPU audit fixtures only; none are benchmark observations."""
import copy
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/verify_generalization_statistics.py"
SPEC = importlib.util.spec_from_file_location("independent_generalization_statistics", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixtures(n=2, constant_ratio=.9):
    """Uniform-within-trial service gives analytic mean = p95 response."""
    groups = [{"id": f"{d}-{j:02d}", "dataset": d} for d in audit.DATASETS for j in range(n)]
    phase = "pilot" if n == 2 else "final"
    sources = {"synthetic_test_only.py": "1" * 64}
    plan = dict(generalization_schema_version=1, phase=phase, split=phase,
                arms=list(audit.ARMS), batch_tokens=128, groups=groups,
                execution_source_files=sources, execution_source_sha256=audit.digest(sources),
                allocation_decision={"groups_per_dataset": n} if n != 2 else None,
                hardware={"allocation_id": "synthetic"},
                calibration={g["id"]: {"allocation_id": "synthetic"} for g in groups}, cases=[])
    orders = list(itertools.permutations(audit.ARMS))
    for ri, rate in enumerate(audit.RATES):
        for number, (g, typ) in enumerate(itertools.product(groups, audit.TYPES)):
            j = int(g["id"][-2:])
            day = (12 + j % 5) if typ == "code" else (14 + (j + 2) % 5)
            date = f"2024-05-{day:02d}"
            block = dict(id=f'{g["id"]}/{typ}', trace=typ, source_date_utc=date)
            calendar = dict(rate_kind=rate, source_date_utc=date,
                            arrival_offsets_ns=[i * 1_000_000 for i in range(256)], horizon_ns=256_000_000)
            plan["cases"].append(dict(block_id=g["id"], dataset=g["dataset"], source_block=block,
                                      calendar=calendar, arm_order=list(orders[(number + ri) % 6])))
    plan["freeze"] = {"sha256": audit.digest(plan)}
    records = []
    for case in plan["cases"]:
        g = case["block_id"]
        for arm in case["arm_order"]:
            service = {"native": 20000, "shortcut": 10000, "combined": round(10000 * constant_ratio)}[arm]
            epoch = 1_000_000_000
            starts = [epoch + x for x in case["calendar"]["arrival_offsets_ns"]]
            finishes = [x + service for x in starts]
            row = dict(content_group_id=g, block_id=g, dataset=case["dataset"], phase=phase,
                       arrival_type=case["source_block"]["trace"], rate_kind=case["calendar"]["rate_kind"], arm=arm,
                       status="complete", config_hash=plan["freeze"]["sha256"],
                       execution_source_sha256=plan["execution_source_sha256"], repetition=0,
                       batch_tokens=128, correctness_pass=True, normal_completion_verified=True,
                       resident_state_verified=True, arrival_schedule=case["calendar"],
                       arrival_offsets_ns=case["calendar"]["arrival_offsets_ns"], horizon_ns=case["calendar"]["horizon_ns"],
                       arrival_date=case["source_block"]["source_date_utc"], arrival_block_id=case["source_block"]["id"],
                       arm_order=case["arm_order"], allocation_id="synthetic", n_jobs=256, epoch_monotonic_ns=epoch,
                       coordinator_admit_ns=starts[:], coordinator_start_ns=starts[:], coordinator_finish_ns=finishes,
                       observation_end_ns=epoch + case["calendar"]["horizon_ns"],
                       per_rank_start_ns=[starts[:], starts[:]], per_rank_gpu_complete_ns=[finishes[:], finishes[:]],
                       queue_metrics=dict(complete=True, scheduled_jobs=256, completed_jobs=256, unfinished_jobs=0,
                                          vectors_ns={"response": [service] * 256},
                                          summary_ns={"response": {"mean": float(service), "p95": float(service)}}))
            records.append(row)
    return plan, records


def descriptive_summary(plan, records):
    # Build the report envelope only; analytic and bootstrap tests below use
    # separately calculated expected numbers instead of this helper's outputs.
    pin, groups, cases = audit.plan_identity(plan)
    index, metrics, pairs = audit.recover_pairs(plan, records, groups, cases, pin)
    summary = dict(valid=True, phase=plan["phase"], expected_records=len(records), observed_records=len(records),
                   valid_records=len(records), matched_triplets=len(cases), matched_pairs=len(pairs),
                   plan_sha256=pin, execution_source_sha256=plan["execution_source_sha256"], errors=[], missing_keys=[],
                   trial_summaries=[dict(content_group_id=k[0], arrival_type=k[1], rate_kind=k[2], arm=k[3],
                                         dataset=index[k]["dataset"], arrival_date=index[k]["arrival_date"], metrics=v)
                                    for k, v in metrics.items()],
                   paired_cases=list(pairs.values()), pairs_per_contrast={c: len(cases) for c in audit.CONTRASTS},
                   allocation_decision=None, confirmatory=None)
    if plan["phase"] == "pilot":
        summary["allocation_decision"] = audit.pilot_decision(pairs, groups)
    else:
        summary["confirmatory"], draw_hash = audit.final_inference(pairs, groups)
        summary["confirmatory"]["bootstrap"]["draw_indices_sha256"] = draw_hash
    return summary


@pytest.fixture(scope="module")
def pilot():
    p, r = fixtures()
    return p, r, descriptive_summary(p, r)


@pytest.fixture(scope="module")
def final():
    p, r = fixtures(6)
    return p, r, descriptive_summary(p, r)


def test_analytic_pilot_and_final_counts(pilot, final):
    for (p, r, s), count in ((pilot, 72), (final, 216)):
        out = audit.audit(p, r, s)
        assert out["passed"] and out["complete_trials"] == count
        assert out["scheduled_jobs_verified"] == count * 256
        assert out["paired_comparisons"] == count
    decision = audit.audit(*pilot)["allocation_decision"]
    assert decision["groups_per_dataset"] == 6 and decision["pooled_sample_sd"] < 1e-15
    out = audit.audit(*final)
    assert out["confirmatory"]["primary"]["ci95"] == pytest.approx([.9, .9])
    assert out["confirmatory"]["acceptance"] == "pass"
    assert out["confirmatory"]["secondary_baseline"]["ci95"] == pytest.approx([.45, .45])
    assert out["production_draw_digest_comparison"] == "matched"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "truncated", "early", "rank", "stored", "calendar", "raw_response"])
def test_raw_mutations_fail_closed(pilot, mutation):
    plan, records, summary = copy.deepcopy(pilot)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "truncated":
        records[0]["coordinator_finish_ns"].pop()
    elif mutation == "early":
        records[0]["coordinator_admit_ns"][0] -= 1
    elif mutation == "rank":
        records[0]["per_rank_gpu_complete_ns"][1][0] += 1
    elif mutation == "stored":
        records[0]["queue_metrics"]["summary_ns"]["response"]["p95"] += 50
    elif mutation == "calendar":
        records[0]["arrival_offsets_ns"] = records[0]["arrival_offsets_ns"][:]
        records[0]["arrival_offsets_ns"][10] += 1
    else:
        # Recompute the row's stored summaries too: the independent verifier
        # must still reject a stale final report after raw observations change.
        row = records[0]
        row["coordinator_finish_ns"][0] += 1000
        row["queue_metrics"]["vectors_ns"]["response"][0] += 1000
        row["queue_metrics"]["summary_ns"]["response"]["mean"] += 1000/256
    with pytest.raises(ValueError):
        audit.audit(plan, records, summary)


@pytest.mark.parametrize("field", ["mean", "pair", "allocation", "pilot_ci"])
def test_report_mutations_do_not_pass(pilot, field):
    p, r, s = copy.deepcopy(pilot)
    if field == "mean":
        s["trial_summaries"][0]["metrics"]["response_mean_ns"] *= 1.01
    elif field == "pair":
        s["paired_cases"][0]["log_ratios"]["response_p95_ns"] += .1
    elif field == "allocation":
        s["allocation_decision"]["groups_per_dataset"] = 8
    else:
        s["confirmatory"] = {"ci95": [.1, .2]}
    with pytest.raises(ValueError):
        audit.audit(p, r, s)


def test_global_order_balance_does_not_hide_rate_confound(pilot):
    p, _, _ = copy.deepcopy(pilot)
    orders = list(itertools.permutations(audit.ARMS))
    for rate_index, rate in enumerate(audit.RATES):
        subset = [c for c in p["cases"] if c["calendar"]["rate_kind"] == rate]
        for i, c in enumerate(subset):
            c["arm_order"] = list(orders[3 * rate_index + i % 3])
    p["freeze"]["sha256"] = audit.digest({k: v for k, v in p.items() if k != "freeze"})
    with pytest.raises(ValueError, match="within rate"):
        audit.plan_identity(p)


def pair_fixture(n=6):
    groups = {d: [f"{d}-{j:02d}" for j in range(n)] for d in audit.DATASETS}
    pairs = {}
    for di, d in enumerate(audit.DATASETS):
        for j, g in enumerate(groups[d]):
            for ti, typ in enumerate(audit.TYPES):
                for ri, rate in enumerate(audit.RATES):
                    theta = math.log(.88) + (j - (n-1)/2) * .018 + di * .004 + ti * .02 + ri * .01
                    date = f"2024-05-{12+j%5 if typ == 'code' else 14+(j+2)%5:02d}"
                    for contrast in audit.CONTRASTS:
                        value = theta if contrast == "combined_over_shortcut" else theta - .4
                        pairs[g,typ,rate,contrast] = dict(arrival_date=date,
                            log_ratios={"response_mean_ns": value, "response_p95_ns": value + .007})
    return groups, pairs


def test_bootstrap_matches_count_weight_oracle_and_exact_draw_hash():
    groups, pairs = pair_fixture()
    result, got_digest = audit.final_inference(pairs, groups)
    rng = np.random.Generator(np.random.PCG64(471512))
    pieces, byte_chunks = [], []
    for di, d in enumerate(audit.DATASETS):
        draw = rng.integers(0, 6, size=(100000, 6))
        byte_chunks.append(draw.astype("<i8").tobytes())
        # A separate histogram-weight formulation, not gathering row means.
        counts = np.array([(draw == j).sum(axis=1) for j in range(6)]).T
        known_group_logs = np.log(.88) + (np.arange(6)-2.5)*.018 + di*.004 + .01
        pieces.append(counts @ known_group_logs / 6)
    expected = np.exp((pieces[0] + pieces[1] + pieces[2]) / 3)
    assert result["primary"]["ci95"] == pytest.approx(np.quantile(expected, [.025, .975]), abs=1e-14)
    assert got_digest == hashlib.sha256(b"".join(byte_chunks)).hexdigest()
    assert all(g["one_sided_confidence_level"] == 1-.05/7 for g in result["guards"])
    assert len(result["guards"]) == 7
    # Alter just common-rate observations: calibrated primary must stay fixed,
    # while common-rate guards must expose that nested condition's regression.
    changed = copy.deepcopy(pairs)
    for key, p in changed.items():
        if key[2] == "common" and key[3] == "combined_over_shortcut":
            p["log_ratios"]["response_mean_ns"] += .3
    other, _ = audit.final_inference(changed, groups)
    assert other["primary"] == result["primary"]
    assert not other["guards"][1]["passes"]
    assert other["secondary_baseline"]["ci95"] is None
    assert other["secondary_baseline"]["status"] == "not_tested_gate_closed"


def test_joint_date_deletion_renormalizes_and_closes_gate():
    groups, pairs = pair_fixture()
    for key, p in pairs.items():
        if key[3] == "combined_over_shortcut":
            p["log_ratios"]["response_mean_ns"] = math.log(.7 if p["arrival_date"] == "2024-05-14" else 1.03)
    result, _ = audit.final_inference(pairs, groups)
    deleted = next(x for x in result["calendar_date_robustness"]["results"] if x["excluded_date"] == "2024-05-14")
    assert deleted["removed_arrival_bundles"] == 6
    assert deleted["retained_content_groups"] == dict.fromkeys(audit.DATASETS, 6)
    assert deleted["ratio"] == pytest.approx(1.03)
    assert not deleted["passes"]
    assert result["secondary_baseline"]["ci95"] is None


def test_pilot_sample_variance_uses_six_whole_bundles_and_not_mean_benefit():
    groups, pairs = pair_fixture(2)
    for key, p in pairs.items():
        if key[3] == "combined_over_shortcut":
            p["log_ratios"]["response_mean_ns"] += .2 if key[0].endswith("01") else -.2
    decision = audit.pilot_decision(pairs, groups)
    sd = np.std(list(decision["pilot_bundle_log_ratios"].values()), ddof=1)
    assert decision["pooled_sample_sd"] == pytest.approx(sd)
    assert decision["projected_log_halfwidth"]["6"] == pytest.approx(2.571*sd/math.sqrt(18))
    assert decision["groups_per_dataset"] == 8
    for key, p in pairs.items():
        if key[3] == "combined_over_shortcut":
            p["log_ratios"]["response_mean_ns"] += 5
    translated = audit.pilot_decision(pairs, groups)
    assert translated["groups_per_dataset"] == 8
    assert translated["pooled_sample_sd"] == pytest.approx(sd)


@pytest.mark.parametrize("mutation", ["digest", "guard", "lodo", "secondary", "interval", "cell"])
def test_final_summary_mutations_fail(final, mutation):
    p, r, s = copy.deepcopy(final)
    result = s["confirmatory"]
    if mutation == "digest":
        result["bootstrap"]["draw_indices_sha256"] = "0"*64
    elif mutation == "guard":
        result["guards"][0]["one_sided_confidence_level"] = .95
    elif mutation == "lodo":
        result["calendar_date_robustness"]["results"][0]["removed_arrival_bundles"] += 1
    elif mutation == "secondary":
        result["secondary_baseline"]["passes"] = False
    elif mutation == "interval":
        result["primary"]["ci95"][1] += .001
    else:
        result["descriptive_cells"][0]["ratios"]["response_p95_ns"] += .01
    with pytest.raises(ValueError):
        audit.audit(p, r, s)


def test_p95_uses_linear_quantile_of_completion_minus_external_release(pilot):
    row = copy.deepcopy(pilot[1][0])
    starts = row["coordinator_start_ns"]
    responses = [10000 + i for i in range(256)]
    row["coordinator_finish_ns"] = [s + x for s, x in zip(starts, responses)]
    row["per_rank_gpu_complete_ns"] = [row["coordinator_finish_ns"][:]] * 2
    row["queue_metrics"]["vectors_ns"]["response"] = responses
    row["queue_metrics"]["summary_ns"]["response"] = {"mean": 10127.5, "p95": 10242.25}
    assert audit.raw_response(row) == {"response_mean_ns": 10127.5, "response_p95_ns": 10242.25}


def test_import_graph_contains_no_production_analysis_or_inference():
    import ast
    tree = ast.parse(PATH.read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m and m.split(".")[0] in ("generalization", "arrival_study", "arrival_fix", "arrival_combo") for m in imports)
