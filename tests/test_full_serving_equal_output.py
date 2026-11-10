import asyncio
import json
from pathlib import Path

from aiohttp import web
import pytest

from full_serving import client, custom_compare, custom_report, audit_final_statistics, freeze_equal_output, output_lengths
from full_serving.analyze import summarize_run
from test_full_serving_client import mock_server, request, events
from test_full_serving_two_workloads import compact_experiment
from test_full_serving_custom_compare import write_json


@pytest.mark.parametrize("target,finish,passed", [(3, "length", True), (3, "stop", False), (4, "length", False)])
def test_fixed_length_wire_parameters_and_fail_closed_completion(target, finish, passed):
    async def run():
        async def handler(req):
            body = await req.json()
            assert body["max_tokens"] == body["min_tokens"] == target
            assert body["ignore_eos"] is True and body["stop"] == body["stop_token_ids"] == []
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            for event in events():
                await response.write(event.replace(b'"finish_reason": "stop"', ('"finish_reason": "' + finish + '"').encode()))
            await response.write_eof()
            return response
        async with mock_server(handler) as endpoint:
            records, _ = await client.run_workload([request()], {"arrival_offsets_ns": [0]}, endpoint=endpoint,
                                                    start_delay_s=0, output_lengths={"r0": target})
        assert records[0]["success"] is passed
        assert records[0]["requested_output_tokens"] == target
        if not passed:
            assert "fixed output-length contract" in records[0]["error_message"]
    asyncio.run(run())


def test_real_reference_freeze_preserves_previous_off_distribution(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "fixed" / "protocol.json"
    result = freeze_equal_output.freeze(root / "results/full-serving/two-workloads-v1/frozen-v1/protocol.json",
                                        root / "results/full-serving/two-workloads-v1/run-v1", output)
    assert len(result["sessions"]) == 6
    assert result["measured_output_tokens_per_arm"] == {"longbench": 162, "numinamath": 17519}
    protocol = json.loads(output.read_bytes()); item = protocol["datasets"][0]["blocks"][0]
    targets = json.loads((output.parent / item["output_lengths_file"]).read_bytes())
    first = next(iter(targets)); targets[first] = 1 if targets[first] != 1 else 2
    item["output_lengths_sha256"] = write_json(output.parent / item["output_lengths_file"], targets)
    with pytest.raises(ValueError, match="previous OFF lengths"):
        output_lengths.validate_targets(protocol, output)


def fixed_experiment(tmp_path, monkeypatch):
    root = compact_experiment(tmp_path, monkeypatch)
    protocol = json.loads((root / "frozen-protocol.json").read_bytes())
    protocol["equal_output_length"] = True
    protocol["generation"]["ignore_eos"] = True
    for d in protocol["datasets"]:
        name = d["dataset"]; block = d["blocks"][0]
        for label, item, reference_phase in (("warmup", d["warmup"], "warmup-00"), ("measured", block, "measured")):
            item["output_reference_file"] = f"output-length-reference/{name}-{label}.jsonl"
            ref = root / item["output_reference_file"];ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_bytes((root / name / block["block_id"] / "off" / (reference_phase + ".jsonl")).read_bytes())
            item["output_reference_sha256"] = custom_compare.native.sha(ref)
            item["output_lengths_file"] = f"output-lengths/{name}-{label}.json"
            item["output_lengths_sha256"] = write_json(root / item["output_lengths_file"], {rid: 3 for rid in item["request_ids"]})
        for arm in custom_compare.ARMS:
            for phase in ("warmup-00", "warmup-01", "measured"):
                item = block if phase == "measured" else d["warmup"]
                p = root / name / block["block_id"] / arm / (phase + ".jsonl")
                rows = [json.loads(line) for line in p.read_text().splitlines()]
                rows[0].update(ignore_eos=True, output_lengths_sha256=item["output_lengths_sha256"], output_length_mode="fixed_previous_off_reference")
                for r in rows[1:-1]:
                    r.update(requested_output_tokens=3, finish_reason="length")
                rows[-1].update(summarize_run(rows[1:-1], rows[-1]))
                p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    protocol["protocol_sha256"] = custom_compare.canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    psha = write_json(root / "frozen-protocol.json", protocol)
    status = json.loads((root / "status.json").read_bytes());status["protocol_sha256"] = psha
    write_json(root / "status.json", status)
    return root


def test_primary_and_independent_reports_verify_exact_per_request_lengths(tmp_path, monkeypatch):
    root = fixed_experiment(tmp_path, monkeypatch)
    report = custom_report.build_report(root)
    assert report["complete"] and report["equal_output_length"]
    p = tmp_path / "report.json";write_json(p, report)
    audit = audit_final_statistics.audit(root, p)
    assert audit["status"] == "passed" and audit["reconstructed_sessions"] == 6


def test_early_stop_or_forged_target_never_survives_report_audits(tmp_path, monkeypatch):
    root = fixed_experiment(tmp_path, monkeypatch)
    p = root / "numinamath/block-0/custom/measured.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[1]["requested_output_tokens"] = 4
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = custom_report.build_report(root)
    assert not report["complete"] and report["datasets"][1]["blocks"][0]["contrasts"] == {}
    rp = tmp_path / "report.json";write_json(rp, report)
    assert audit_final_statistics.audit(root, rp)["status"] == "failed"
