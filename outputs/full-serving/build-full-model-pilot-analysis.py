"""Reproduce the descriptive pilot note from closed public records only.

Run from the repository root. No model execution, private data, or final-data
access. This artifact is separate from the frozen serving implementation.
"""
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/full-serving/custom-pilot-report.json"
EXPECTED_REPORT_SHA256 = "4eadab4d80c4d93c142d70ade2e51f8c2c77bb86916760bf38bf59db691621c7"
OUT = ROOT / "outputs/full-serving/full-model-pilot-analysis"
ARMS = ("off", "resident", "placement_only")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    a, b = math.floor(index), math.ceil(index)
    return values[a] + (values[b] - values[a]) * (index - a)


def avg(values):
    return statistics.fmean(values) if values else None


def change(treatment, baseline):
    return (treatment / baseline - 1) * 100 if baseline else None


def raw_phase(view):
    path = ROOT / view["path"]
    raw = path.read_bytes()
    assert view["valid"] and sha(raw) == view["source_sha256"]
    data = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    header = [r for r in data if r["type"] == "run_header"]
    timing = [r for r in data if r["type"] == "run_summary"]
    rows = sorted((r for r in data if r["type"] == "request"), key=lambda r: r["request_index"])
    assert len(header) == len(timing) == 1 and len(rows) == 8
    assert [r["request_index"] for r in rows] == list(range(8)) and len({r["request_id"] for r in rows}) == 8
    header, timing = header[0], timing[0]
    epoch = timing["epoch_monotonic_ns"]
    response, ttft, tpot = [], [], []
    for i, r in enumerate(rows):
        assert r["success"] and r["usage"]["prompt_tokens"] == r["expected_input_tokens"]
        assert r["scheduled_ns"] == epoch + i * 500_000_000
        assert r["response_from_schedule_ns"] == r["end_ns"] - r["scheduled_ns"]
        response.append((r["end_ns"] - r["scheduled_ns"]) / 1e6)
        if r["first_content_ns"] is not None:
            assert r["ttft_from_schedule_ns"] == r["first_content_ns"] - r["scheduled_ns"]
            ttft.append((r["first_content_ns"] - r["scheduled_ns"]) / 1e6)
            if r["usage"]["completion_tokens"] > 1:
                value = (r["last_content_ns"] - r["first_content_ns"]) / (r["usage"]["completion_tokens"] - 1)
                assert math.isclose(value, r["tpot_stream_boundary_estimate_ns"], rel_tol=1e-12)
                tpot.append(value / 1e6)
    last = max(r["end_ns"] for r in rows)
    horizon = 3_500_000_000
    elapsed = max(last, epoch + horizon) - epoch
    drain = max(0, last - epoch - horizon)
    assert timing["completed_elapsed_ns"] == elapsed and timing["drain_ns"] == drain
    result = {"mean_response_ms": avg(response), "p95_response_ms": quantile(response, .95),
              "mean_ttft_schedule_ms": avg(ttft), "mean_tpot_stream_boundary_ms": avg(tpot),
              "tpot_request_count": len(tpot), "completed_elapsed_s": elapsed / 1e9,
              "drain_s": drain / 1e9, "completed_requests": len(rows), "failed_requests": 0,
              "completed_input_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
              "completed_output_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
              "output_cap_completions": sum(r["finish_reason"] == "length" for r in rows),
              "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
              "raw_path": str(path.relative_to(ROOT)), "raw_sha256": sha(raw),
              "logprobs": header["logprobs"], "raw_recomputation": "passed"}
    for key in ("mean_response_ms", "p95_response_ms", "mean_ttft_schedule_ms", "completed_elapsed_s",
                "drain_s", "completed_input_tokens", "completed_output_tokens", "output_cap_completions"):
        assert math.isclose(result[key], view["metrics"][key], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["mean_tpot_stream_boundary_ms"], view["summary"]["tpot_stream_boundary_estimate_ms"]["mean"], rel_tol=1e-12)
    result["cache_series"] = view["cache"]["series"]
    assert view["cache"]["available"] and all(s["available"] for s in result["cache_series"])
    result["cache_totals"] = {name: sum(s["delta"] for s in result["cache_series"] if s["metric"] == name)
                              for name in sorted({s["metric"] for s in result["cache_series"]})}
    result["cache_total_semantics"] = "Sum of valid exact-label token counters; per-engine source series retained. Not a pooled engine hit-rate average."
    return result, rows


MEMORY_PATTERNS = {
    "model_loading_GiB": r"Model loading took ([0-9.]+) GiB",
    "consumed_weights_plus_nontorch_GiB": r"Actual usage is ([0-9.]+) GiB for consumed memory",
    "peak_activation_GiB": r"([0-9.]+) GiB for peak activation",
    "actual_graph_pool_GiB": r"CUDA graph pool memory: ([0-9.]+) GiB \(actual\)",
    "kv_memory_in_use_GiB": r"Current kv cache memory in use is ([0-9.]+) GiB",
    "kv_capacity_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
}


def memory(directory):
    rows, files = [], {}
    for path in sorted(directory.glob("server-*.log")):
        raw = path.read_bytes()
        files[str(path.relative_to(ROOT))] = sha(raw)
        for line_no, line in enumerate(raw.decode(errors="replace").split("\n"), 1):
            rank = re.search(r"(?:Worker|EngineCore)_DP([0-7])(?:_EP[0-7])? ", line)
            if rank is None:
                continue
            for name, pattern in MEMORY_PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    rows.append({"rank": int(rank[1]), "metric": name, "value": float(match[1].replace(",", "")),
                                 "path": str(path.relative_to(ROOT)), "line": line_no, "source_line": line})
    result = {"files_sha256": files, "rank_records": rows, "summary": {},
              "semantics": "Rounded startup log measurements, not independent repetitions or a measured serving peak; KV tokens are configured capacity."}
    for name in MEMORY_PATTERNS:
        found = [r for r in rows if r["metric"] == name]
        assert len(found) == 8 and {r["rank"] for r in found} == set(range(8)), (directory, name)
        values = [r["value"] for r in found]
        result["summary"][name] = {"ranks": 8, "min": min(values), "max": max(values), "mean": avg(values)}
    return result


def build():
    raw = REPORT.read_bytes()
    assert sha(raw) == EXPECTED_REPORT_SHA256, "closed pilot report changed; this interpretation needs a new audit"
    report = json.loads(raw)
    assert report["complete"] and report["pilot_only"] and report["final_speedup_claim"] is False
    result = {"schema": "full_model_resident_pilot_analysis_v1", "pilot_only": True,
              "source_report": str(REPORT.relative_to(ROOT)), "source_report_sha256": sha(raw),
              "final_speedup_claim": False, "intervals_or_pvalues": None, "final_data_read": False,
              "limitations": report["limits"], "datasets": [], "memory": {}}
    for dataset in report["datasets"]:
        d = {"dataset": dataset["dataset"], "passes": []}
        for phase in dataset["passes"]:
            p = {"pass": phase["pass"], "diagnostic_payload": phase["diagnostic_payload"], "arms": {}, "contrasts": {}}
            records = {}
            for arm in ARMS:
                p["arms"][arm], records[arm] = raw_phase(phase["arms"][arm])
            canon = lambda series: sorted((s["metric"], tuple(sorted(s["labels"].items())), s["delta"]) for s in series)
            p["exact_engine_cache_counters_equal_across_arms"] = all(canon(p["arms"][arm]["cache_series"]) == canon(p["arms"]["off"]["cache_series"]) for arm in ARMS)
            for baseline in ("off", "placement_only"):
                key = "resident/" + baseline
                assert phase["contrasts"][key]["valid"]
                b, t = p["arms"][baseline], p["arms"]["resident"]
                p["contrasts"][key] = {"percent_changes": {k: change(t[k], b[k]) for k in (
                    "mean_response_ms", "p95_response_ms", "mean_ttft_schedule_ms", "mean_tpot_stream_boundary_ms",
                    "completed_elapsed_s", "completed_output_tokens")},
                    "equal_generated_texts": phase["contrasts"][key]["equal_generated_texts"],
                    "paired_requests": phase["contrasts"][key]["requests"]}
            d["passes"].append(p)
        result["datasets"].append(d)
    for arm in ARMS:
        result["memory"][arm] = memory(ROOT / report["sources"][arm]["directory"])
    result["derived_architectural_cost"] = {
        "extra_physical_experts_per_rank_per_layer": 4, "layers": 48,
        "extra_BF16_expert_weight_GiB_per_rank": 4 * 48 * (1536 * 2048 + 2048 * 768) * 2 / 2**30,
        "derivation": "4 extra experts ×48 layers ×(w13 + w2 element counts) ×2 BF16 bytes; this is a shape-based deduction, not a separate measurement."}
    return result


def html(result):
    esc = lambda value: escape(str(value))
    f = lambda value: f"{value:,.3f}"
    pct = lambda value: f"{value:+.2f}%"
    out = ["<!doctype html><html lang=en><meta charset=utf-8><title>Full-model resident pilot analysis</title>",
           "<style>body{font:15px/1.55 system-ui;max-width:1320px;margin:32px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%;margin:16px 0 28px}th,td{padding:7px;border-bottom:1px solid #d5dde4;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf2f6}.note{padding:15px;background:#fff0ce}pre{white-space:pre-wrap;font-size:12px}h2{margin-top:35px}a{color:#135da6}</style>",
           "<h1>Full-model resident pilot: no consistent advantage over OFF</h1>",
           "<p class=note>All 240 timed-pilot executions completed (3 arms ×5 sources ×2 passes ×8 requests). These are previously used pilot inputs, one fixed-order server session per arm, at 2 requests/s. Pass1 repeats pass0 inputs without logprobs. No interval, p-value, final-data analysis or superiority claim is made.</p>",
           "<p>On pass1, resident mean response is slightly lower than OFF on WildChat and InstructCoder, almost unchanged on NuminaMath, and higher on ShareGPT and LongBench. None shows a 5% mean-response gain over OFF. A lower mean can coexist with worse tails and longer completion time: InstructCoder illustrates this directly.</p>",
           "<h2>Normal request pass: resident changes relative to each baseline</h2>",
           "<p>Negative percentages mean lower latency/time or fewer output tokens; fewer output tokens are not themselves an optimization. TPOT is the mean of per-request stream-boundary estimates, not individual-token or isolated GPU timing. Each row uses the same eight inputs and absolute arrival offsets.</p>",
           "<table><tr><th>Source / baseline</th><th>Mean response</th><th>p95</th><th>TTFT</th><th>TPOT estimate</th><th>Elapsed</th><th>Output tokens</th><th>Identical text /8</th></tr>"]
    for d in result["datasets"]:
        p = d["passes"][1]
        for base in ("off", "placement_only"):
            c = p["contrasts"]["resident/" + base]
            values = [d["dataset"] + " / " + base, *[pct(c["percent_changes"][k]) for k in (
                "mean_response_ms", "p95_response_ms", "mean_ttft_schedule_ms", "mean_tpot_stream_boundary_ms", "completed_elapsed_s", "completed_output_tokens")], c["equal_generated_texts"]]
            out.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in values) + "</tr>")
    out += ["</table><p><strong>Mechanism evidence is limited:</strong> resident has a 1.60–2.42% lower TPOT estimate than placement-only on the first four sources, while LongBench is 1.29% higher. Against OFF, the first four differ by only −0.27% to +0.44%; LongBench is 5.99% higher. This is compatible with recovering a small cost of hash dispatch on the replicated map, without an established advantage over OFF. It does not isolate a kernel effect because scheduling, generated work and fixed execution order also differ.</p>",
            "<p><strong>Output length matters:</strong> ShareGPT resident produces 8.42% more tokens than OFF while mean response rises 8.12% and mean TPOT is almost unchanged. NuminaMath resident and OFF have similar total token counts, but OFF hits the 2,048-token cap once and resident never does; resident elapsed falls 15.03% while mean response rises 0.39%. InstructCoder resident mean falls 1.82%, but p95 rises 9.89% and elapsed rises 9.53%. The distribution and timing of individual outputs matter, not just their sum. These are descriptive associations, not causal decompositions.</p>",
            "<h2>Absolute pass1 values</h2><table><tr><th>Source / arm</th><th>Mean response s</th><th>TTFT ms</th><th>TPOT estimate ms</th><th>Elapsed s</th><th>Drain s</th><th>Output tokens</th><th>Length caps</th></tr>"]
    for d in result["datasets"]:
        for arm, m in d["passes"][1]["arms"].items():
            values = [d["dataset"] + " / " + arm, f(m["mean_response_ms"] / 1000), f(m["mean_ttft_schedule_ms"]), f(m["mean_tpot_stream_boundary_ms"]), f(m["completed_elapsed_s"]), f(m["drain_s"]), m["completed_output_tokens"], m["output_cap_completions"]]
            out.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in values) + "</tr>")
    out += ["</table><h2>Repeated inputs and cache</h2>",
            "<p>For pass1, prefix-cache hit/query counts match exactly across the three arms for every recorded engine/model label, and all recorded preemption deltas are zero. Total cached/query token counts are WildChat 11,792/11,860, ShareGPT 12,736/12,791, InstructCoder 1,104/1,177, NuminaMath 864/943, and LongBench 81,456/81,531. The second pass is strongly cache-warmed, especially for long prompts. Matching counters reduce one observed arm difference; they do not make this a cold-input test or prove identical cache execution. Pass0 and pass1 cannot be interpreted as a controlled cache experiment because diagnostics, repeated content, warming and generated outputs change together.</p>",
            "<h2>Memory cost recorded by the actual servers</h2><p>Ranges below cover all eight ranks; they are rounded startup log measurements, not repeated samples or a serving-peak profile. Token capacity is the allocated KV-cache capacity.</p>",
            "<table><tr><th>Arm</th><th>Model-loading GiB/rank</th><th>Weights + non-Torch GiB/rank</th><th>Peak activation GiB/rank</th><th>Actual graph pool GiB/rank</th><th>KV-cache GiB/rank</th><th>KV token capacity/rank</th></tr>"]
    for arm, item in result["memory"].items():
        values = [arm]
        for key in MEMORY_PATTERNS:
            s = item["summary"][key]
            values.append(f'{s["min"]:,.2f}' if s["min"] == s["max"] else f'{s["min"]:,.2f}–{s["max"]:,.2f}')
        out.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in values) + "</tr>")
    out += ["</table><p>Both 160-slot arms use 11.32 GiB per GPU for model loading versus OFF's 9.63 GiB, an observed +1.69 GiB. The shape-based expectation for four extra BF16 experts per rank across 48 layers is 1.6875 GiB, consistent with that rounded difference. Resident's graph pool is 1.44 GiB versus 1.35 GiB for OFF and placement-only. Its mean configured KV token capacity is about 2.56% below OFF and 0.056% below placement-only. Extra storage reduces capacity headroom, but the pilot records zero preemptions; these logs do not establish memory pressure as the latency cause.</p>",
            "<h2>Diagnostic pass0 retained separately</h2><p>These comparisons include requested token IDs and logprobs, so they are not ordinary-payload performance baselines. They also show that the signs are not stable across the two passes.</p>",
            "<table><tr><th>Source</th><th>Resident/OFF mean response change</th><th>Resident/placement mean response change</th><th>Resident/OFF output-token change</th></tr>"]
    for d in result["datasets"]:
        p = d["passes"][0]
        values = [d["dataset"], pct(p["contrasts"]["resident/off"]["percent_changes"]["mean_response_ms"]), pct(p["contrasts"]["resident/placement_only"]["percent_changes"]["mean_response_ms"]), pct(p["contrasts"]["resident/off"]["percent_changes"]["completed_output_tokens"])]
        out.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in values) + "</tr>")
    out += ["</table><p><strong>What remains unresolved:</strong> equal-prompt runs generate different continuations; temperature zero does not guarantee identical text. These pilots cannot distinguish numerical/scheduling effects from routing overhead, cannot establish a causal benefit from the dispatcher alone, and have no independent session-level uncertainty. The frozen held-out experiment is needed for generalization. No policy, threshold or final selection was changed for this analysis.</p>",
            '<p>Evidence: <a href="custom-pilot-report.json">complete pilot comparison</a> · <a href="full-model-pilot-analysis.json">independent recomputation and per-rank log citations</a> · <a href="build-full-model-pilot-analysis.py">reproduction script</a>. The script checks 30 raw eight-request phases and reads only closed pilot files; no final or private text data is accessed.</p>',
            "<p>Source report SHA-256: <code>" + result["source_report_sha256"] + "</code>.</p></html>"]
    return "\n".join(out)


if __name__ == "__main__":
    result = build()
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    OUT.with_suffix(".html").write_text(html(result))
    print(json.dumps({"raw_phases_recomputed": 30, "completed_pilot_executions": 240, "final_data_read": False,
                      "output": str(OUT.with_suffix('.html'))}))
