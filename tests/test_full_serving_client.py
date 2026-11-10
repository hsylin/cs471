import asyncio
from contextlib import asynccontextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from aiohttp import web
import pytest

from full_serving.client import SSEDecoder, ProtocolError, _open_new, load_workload, run_workload
from full_serving.analyze import compare_runs, load_run, summarize_run


@asynccontextmanager
async def mock_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        await runner.cleanup()


def request(rid="r0"):
    return {"id": rid, "messages": [{"role": "user", "content": rid}], "input_tokens": 4}


def events(*, usage=True, done=True, prompt_tokens=4, private_logprobs=False):
    values = [
        {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": "你好"},
                      **({"logprobs": {"content": [{"token": "你好", "logprob": -0.1}]}} if private_logprobs else {})}]},
        {"choices": [{"index": 0, "delta": {"content": " world"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    if usage:
        values.append({"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                       "completion_tokens": 3, "total_tokens": prompt_tokens + 3}})
    encoded = ["data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n" for value in values]
    if done:
        encoded.append("data: [DONE]\r\n\r\n")
    return [value.encode() for value in encoded]


def test_incremental_utf8_sse_crlf_comments_and_multiline():
    parser = SSEDecoder()
    data = b"\xef\xbb\xbf: keepalive\r\ndata: " + "你好".encode() + b"\r\ndata: world\r\n\r\ndata: [DONE]\n\n"
    result = []
    for byte in data:
        result.extend(parser.feed(bytes([byte])))
    result.extend(parser.feed(b"", final=True))
    assert result == ["你好\nworld", "[DONE]"]
    with pytest.raises(ProtocolError, match="unterminated"):
        SSEDecoder().feed(b"data: partial", final=True)
    with pytest.raises(UnicodeDecodeError):
        SSEDecoder().feed(b"\xff", final=True)


def test_mock_stream_uses_server_tokens_and_keeps_output_private():
    async def run():
        received = []
        stored = []

        async def handler(req):
            received.append(await req.json())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            for event in events(private_logprobs=True):
                # Deliberately split within UTF-8 code points and SSE delimiters.
                for start in range(0, len(event), 2):
                    await response.write(event[start:start + 2])
                await asyncio.sleep(.003)
            await response.write_eof()
            return response

        async with mock_server(handler) as endpoint:
            records, timing = await run_workload([request()], {"arrival_offsets_ns": [0]},
                endpoint=endpoint, start_delay_s=0, logprobs=True,
                on_record=lambda public, private: stored.append((public, private)))
        row = records[0]
        assert row["success"] and row["usage"]["completion_tokens"] == 3
        assert sum(c["content_chars"] > 0 for c in row["chunks"]) == 2
        assert row["tpot_stream_boundary_estimate_ns"] == (row["last_content_ns"] - row["first_content_ns"]) / 2
        assert row["scheduled_ns"] <= row["send_ns"] <= row["first_content_ns"] <= row["last_content_ns"] <= row["end_ns"]
        assert stored[0][1]["output"] == "你好 world"
        assert stored[0][1]["token_logprobs"][0]["logprobs"]["content"][0]["logprob"] == -.1
        assert "你好" not in json.dumps(row, ensure_ascii=False)
        assert row["output_sha256"] == hashlib.sha256("你好 world".encode()).hexdigest()
        assert received[0]["max_tokens"] == 2048 and received[0]["temperature"] == 0
        assert received[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert received[0]["stream_options"] == {"include_usage": True}
        assert "ignore_eos" not in received[0] and "min_tokens" not in received[0]
        summary = summarize_run(records, timing)
        assert summary["valid"] and summary["completion_tokens"]["mean"] == 3
        assert summary["content_chunk_counts"]["mean"] == 2

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["http", "timeout", "no_usage", "no_done", "wrong_prompt_tokens", "malformed_usage"])
def test_failed_requests_are_preserved_and_invalidate_run(mode):
    async def run():
        async def handler(req):
            if mode == "http":
                return web.Response(status=503, text="private server diagnostic")
            if mode == "timeout":
                await asyncio.sleep(.06)
                return web.Response(text="too late")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            stream = events(usage=mode != "no_usage", done=mode != "no_done",
                            prompt_tokens=5 if mode == "wrong_prompt_tokens" else 4)
            if mode == "malformed_usage":
                stream[-2] = b'data: {"choices":[],"usage":"bad"}\n\n'
            for event in stream:
                await response.write(event)
            await response.write_eof()
            return response

        stored = []
        async with mock_server(handler) as endpoint:
            records, timing = await run_workload([request()], {"arrival_offsets_ns": [0]},
                endpoint=endpoint, start_delay_s=0, timeout_s=.015 if mode == "timeout" else 1,
                on_record=lambda public, private: stored.append((public, private)))
        assert len(records) == 1 and not records[0]["success"]
        assert records[0]["end_ns"] >= records[0]["send_ns"]
        assert stored[0][1]["error_detail"]
        summary = summarize_run(records, timing)
        assert not summary["valid"] and not summary["speed_claim_allowed"]
        assert summary["offered_requests"] == summary["failed_requests"] == 1
        assert summary["completed_requests"] == 0
        assert "private server diagnostic" not in json.dumps(records)

    asyncio.run(run())


def test_absolute_arrivals_do_not_reset_during_client_backlog_and_all_drain():
    async def run():
        arrival = []

        async def handler(req):
            arrival.append(time.monotonic_ns())
            await asyncio.sleep(.04)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b"".join(events()))
            await response.write_eof()
            return response

        offsets = [0, 5_000_000, 10_000_000]
        epoch = time.monotonic_ns() + 10_000_000
        async with mock_server(handler) as endpoint:
            records, timing = await run_workload([request(str(i)) for i in range(3)],
                {"arrival_offsets_ns": offsets, "horizon_ns": 10_000_000},
                endpoint=endpoint, epoch_ns=epoch, max_inflight=1, timeout_s=1)
        assert len(arrival) == len(records) == 3 and all(r["success"] for r in records)
        assert [r["scheduled_ns"] for r in records] == [epoch + offset for offset in offsets]
        assert records[-1]["client_semaphore_wait_ns"] > 40_000_000
        assert records[-1]["response_from_schedule_ns"] > records[-1]["response_from_send_ns"] + 40_000_000
        assert timing["drain_ns"] == records[-1]["end_ns"] - (epoch + offsets[-1])
        assert timing["completed_elapsed_ns"] == records[-1]["end_ns"] - epoch

    asyncio.run(run())


def test_workload_rejects_truncation_order_errors_and_labels_synthetic_pilot(tmp_path):
    path = tmp_path / "private.jsonl"
    rows = [request(str(i)) for i in range(2)]
    rows[0]["messages"][0]["content"] = "Real text with\u2028Unicode line separator and\u2029paragraph separator"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    calendar = tmp_path / "calendar.json"
    calendar.write_text(json.dumps({"arrival_offsets_ns": [0, 10], "request_ids": ["0", "1"]}))
    rows, schedule, hashes = load_workload(path, calendar)
    assert len(rows) == 2 and schedule["horizon_ns"] == 10 and len(hashes["requests_sha256"]) == 64
    assert "\u2028" in rows[0]["messages"][0]["content"]
    with pytest.raises(ValueError, match="over-context"):
        load_workload(path, calendar, max_model_len=2050)
    calendar.write_text(json.dumps({"arrival_offsets_ns": [0, 10], "request_ids": ["1", "0"]}))
    with pytest.raises(ValueError, match="order"):
        load_workload(path, calendar)
    _, burst, _ = load_workload(path, None, rate=0)
    assert burst["arrival_offsets_ns"] == [0, 0] and burst["kind"] == "synthetic_pilot_burst"
    _, regular, _ = load_workload(path, None, rate=2)
    assert regular["arrival_offsets_ns"] == [0, 500_000_000]
    private = tmp_path / "output.private.jsonl"
    with _open_new(private, private=True) as stream:
        stream.write("private")
    assert os.stat(private).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _open_new(private, private=True)


def test_block_inference_never_resamples_requests_or_filters_failed_runs():
    runs = []
    for block, ratio in [("a", .8), ("b", 1.0)]:
        for arm, value in [("off", 10), ("on", 10 * ratio)]:
            runs.append({"header": {"arm": arm, "load_id": "same_rate", "pair_id": block,
                                    "block_id": block, "requests_sha256": block, "calendar_sha256": block},
                         "summary": {"valid": True, "completed_elapsed_s": value}})
    result = compare_runs(runs, bootstrap=1000)[0]
    assert result["independent_blocks"] == 2 and result["matched_run_pairs"] == 2
    metric = result["metrics"]["completed_elapsed_s"]
    assert metric["treatment_over_baseline"] == pytest.approx(.8 ** .5)
    assert metric["ci95"] == pytest.approx([.8, 1.0])
    one_block = compare_runs(runs[:2])[0]
    assert one_block["metrics"]["completed_elapsed_s"]["ci95"] is None
    broken = copy.deepcopy(runs)
    broken[-1]["summary"]["valid"] = False
    invalid = compare_runs(broken)[0]
    assert not invalid["valid"] and invalid["metrics"] == {}
    assert "withheld" in invalid["inference"]
    assert not compare_runs(runs[:-1])[0]["valid"]


def test_partial_run_file_is_invalid_not_silently_discarded(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"type": "run_header", "request_count": 8}) + "\n")
    result = load_run(path)
    assert not result["summary"]["valid"]
    assert result["summary"]["recorded_requests"] == 0


def test_cli_mock_end_to_end_writes_auditable_public_and_private_records(tmp_path):
    async def run():
        async def handler(req):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b"".join(events(private_logprobs=True)))
            await response.write_eof()
            return response

        inputs = tmp_path / "input.private.jsonl"
        inputs.write_text(json.dumps(request()) + "\n")
        public = tmp_path / "run.jsonl"
        private = tmp_path / "generated.private.jsonl"
        async with mock_server(handler) as endpoint:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "full_serving.client", "--requests", str(inputs),
                "--rate", "0", "--base-url", endpoint.removesuffix("/chat/completions"),
                "--output", str(public), "--private-output", str(private), "--run-id", "mock-run",
                "--arm", "off", "--block-id", "pilot-00", "--start-delay-s", "0", "--logprobs",
                cwd=Path(__file__).resolve().parents[1], stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
        assert proc.returncode == 0, stderr.decode()
        report = json.loads(stdout)
        assert report["valid"] and report["completed_requests"] == 1
        loaded = load_run(public)
        assert loaded["summary"]["valid"]
        assert loaded["header"]["arrival_kind"] == "synthetic_pilot_burst"
        assert len(loaded["header"]["client_source_sha256"]) == 64
        assert "你好" not in public.read_text()
        assert json.loads(private.read_text())["output"] == "你好 world"
        assert os.stat(private).st_mode & 0o777 == 0o600

    asyncio.run(run())
