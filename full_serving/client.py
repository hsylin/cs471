"""Open-loop, full-model streaming chat replay (requires aiohttp).

The input JSONL is private: one {id, messages, input_tokens, ...metadata} per
request. The calendar is {arrival_offsets_ns: [...], request_ids?: [...],
horizon_ns?: int, ...provenance}. Offsets are relative to one monotonic epoch;
they never move when the server or client is overloaded. HTTP/SSE arrival
chunks are NOT tokens. Public records contain timings/hashes, never text.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable


class ProtocolError(ValueError):
    pass


class SSEDecoder:
    """Incremental UTF-8 and SSE line parser, including split CRLF and BOM."""

    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
        self.pending = ""
        self.lines: list[str] = []

    def feed(self, data: bytes, final: bool = False) -> list[str]:
        self.pending += self.decoder.decode(data, final=final)
        events: list[str] = []
        while True:
            positions = [p for p in (self.pending.find("\r"), self.pending.find("\n")) if p >= 0]
            if not positions:
                break
            pos = min(positions)
            char = self.pending[pos]
            if char == "\r" and pos == len(self.pending) - 1 and not final:
                break
            line = self.pending[:pos]
            width = 2 if char == "\r" and self.pending[pos:pos + 2] == "\r\n" else 1
            self.pending = self.pending[pos + width:]
            if not line:
                if self.lines:
                    events.append("\n".join(self.lines))
                self.lines = []
            elif line.startswith("data:"):
                value = line[5:]
                self.lines.append(value[1:] if value.startswith(" ") else value)
            # SSE comments/event/id/retry fields do not alter chat payloads.
        if final and (self.pending or self.lines):
            raise ProtocolError("unterminated SSE event at EOF")
        return events


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProtocolError("non-finite JSON number in server event")
    return parsed


def _reject_json_constant(value: str):
    raise ProtocolError("non-finite JSON constant in server event")


def load_workload(request_path: Path, calendar_path: Path | None, max_tokens: int = 2048,
                  max_model_len: int = 32768, rate: float | None = None) -> tuple[list[dict], dict, dict]:
    request_bytes = request_path.read_bytes()
    requests = [json.loads(line) for line in request_bytes.decode().split("\n") if line.strip()]
    if not requests:
        raise ValueError("empty request workload")
    seen: set[str] = set()
    for request in requests:
        rid = request.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("request IDs must be nonempty unique strings")
        seen.add(rid)
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{rid}: messages must be a nonempty list")
        if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
               or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"{rid}: only text system/user/assistant messages are supported")
        if messages[-1]["role"] != "user":
            raise ValueError(f"{rid}: conversation must end with user input")
        count = request.get("input_tokens")
        if not _integer(count) or count <= 0 or count + max_tokens > max_model_len:
            raise ValueError(f"{rid}: invalid or over-context input_tokens; no truncation is performed")
    if (calendar_path is None) == (rate is None):
        raise ValueError("exactly one of calendar_path or rate is required")
    if calendar_path is not None:
        calendar_bytes = calendar_path.read_bytes()
        calendar = json.loads(calendar_bytes)
    else:
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative; zero means simultaneous pilot burst")
        calendar = {"kind": "synthetic_pilot_burst" if rate == 0 else "synthetic_regular_pilot",
                    "rate_requests_per_s": rate,
                    "arrival_offsets_ns": [round(i / rate * 1e9) if rate else 0 for i in range(len(requests))],
                    "request_ids": [r["id"] for r in requests]}
        calendar_bytes = json.dumps(calendar, sort_keys=True).encode()
    offsets = calendar.get("arrival_offsets_ns")
    if (not isinstance(offsets, list) or len(offsets) != len(requests)
            or any(not _integer(v) or v < 0 for v in offsets)
            or offsets != sorted(offsets)):
        raise ValueError("calendar requires one nondecreasing nonnegative integer ns offset per request")
    if "request_ids" in calendar and calendar["request_ids"] != [r["id"] for r in requests]:
        raise ValueError("calendar request_ids do not match request file order")
    horizon = calendar.get("horizon_ns", offsets[-1])
    if not _integer(horizon) or horizon < offsets[-1]:
        raise ValueError("horizon_ns must be >= the last arrival")
    calendar["horizon_ns"] = horizon
    hashes = {"requests_sha256": sha256_bytes(request_bytes),
              "calendar_sha256": sha256_bytes(calendar_bytes),
              "request_ids_sha256": canonical_hash([r["id"] for r in requests])}
    return requests, calendar, hashes


async def sleep_until_ns(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining / 1e9)


def _consume_event(raw: str, received_ns: int, record: dict, text_parts: list[str],
                   private_logprobs: list[dict], private_tokens: dict) -> None:
    if record["saw_done"]:
        raise ProtocolError("event after [DONE]")
    if raw == "[DONE]":
        record["saw_done"] = True
        record["done_ns"] = received_ns
        return
    event = json.loads(raw, parse_float=_finite_json_float, parse_constant=_reject_json_constant)
    if not isinstance(event, dict):
        raise ProtocolError("SSE data is not an object")
    if event.get("error") is not None:
        raise ProtocolError("server returned an error event")
    if event.get("id"):
        record["server_request_id"] = event["id"]
    if event.get("model"):
        record["server_model"] = event["model"]
    if event.get("usage") is not None:
        record["usage"] = event["usage"]
        record["usage_events"] += 1
    if event.get("prompt_token_ids") is not None:
        ids = event["prompt_token_ids"]
        if not isinstance(ids, list) or any(not _integer(i) or i < 0 for i in ids):
            raise ProtocolError("invalid prompt token IDs")
        if private_tokens["prompt_token_ids"] is not None:
            raise ProtocolError("duplicate prompt token IDs in stream")
        private_tokens["prompt_token_ids"] = ids
    choices = event.get("choices", [])
    if not isinstance(choices, list) or len(choices) > 1:
        raise ProtocolError("expected at most one streamed choice")
    content = ""
    for choice in choices:
        if choice.get("index", 0) != 0:
            raise ProtocolError("unexpected choice index")
        delta = choice.get("delta") or {}
        if choice.get("logprobs") is not None:
            logprobs = choice["logprobs"]
            if not isinstance(logprobs, dict) or not isinstance(logprobs.get("content"), list):
                raise ProtocolError("invalid streamed logprob content")
            for item in logprobs["content"]:
                value = item.get("logprob") if isinstance(item, dict) else None
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value)):
                    raise ProtocolError("missing or non-finite generated token logprob")
            private_logprobs.append({"received_ns": received_ns, "logprobs": logprobs})
        if choice.get("token_ids") is not None:
            ids = choice["token_ids"]
            if not isinstance(ids, list) or any(not _integer(i) or i < 0 for i in ids):
                raise ProtocolError("invalid completion token IDs")
            private_tokens["completion_token_ids"].extend(ids)
        if choice.get("stop_reason") is not None:
            private_tokens["stop_reason"] = choice["stop_reason"]
        if delta.get("reasoning_content") or delta.get("reasoning"):
            raise ProtocolError("thinking output present despite enable_thinking=false")
        content = delta.get("content") or ""
        if not isinstance(content, str):
            raise ProtocolError("non-text content delta")
        if choice.get("finish_reason") is not None:
            record["finish_reason"] = choice["finish_reason"]
    chunk = {"received_ns": received_ns, "content_chars": len(content),
             "content_utf8_bytes": len(content.encode()), "has_usage": event.get("usage") is not None}
    record["chunks"].append(chunk)
    if content:
        if record["first_content_ns"] is None:
            record["first_content_ns"] = received_ns
        record["last_content_ns"] = received_ns
        text_parts.append(content)


def _validate_completion(record: dict) -> None:
    if not record["saw_done"]:
        raise ProtocolError("stream ended without [DONE]")
    if record["finish_reason"] not in {"stop", "length"}:
        raise ProtocolError("missing or unexpected finish reason")
    usage = record["usage"]
    if not isinstance(usage, dict):
        raise ProtocolError("missing server usage; chunk counts cannot replace tokens")
    if any(not _integer(usage.get(key)) or usage[key] < 0
           for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ProtocolError("invalid server token usage")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ProtocolError("inconsistent total token usage")
    if usage["prompt_tokens"] != record["expected_input_tokens"]:
        raise ProtocolError("server prompt tokens differ from frozen tokenization")


def _validate_diagnostics(record: dict, private_tokens: dict, private_logprobs: list[dict]) -> None:
    """Pinned vLLM emits every generated token's metadata, even without text."""
    prompt_ids = private_tokens["prompt_token_ids"]
    if prompt_ids is None or len(prompt_ids) != record["usage"]["prompt_tokens"]:
        raise ProtocolError("missing or incomplete diagnostic prompt token IDs")
    expected_hash = record["expected_prompt_token_ids_sha256"]
    if expected_hash is not None and canonical_hash(prompt_ids) != expected_hash:
        raise ProtocolError("diagnostic prompt token IDs differ from frozen tokenization")
    count = record["usage"]["completion_tokens"]
    if len(private_tokens["completion_token_ids"]) != count:
        raise ProtocolError("missing or incomplete diagnostic completion token IDs")
    if sum(len(event["logprobs"]["content"]) for event in private_logprobs) != count:
        raise ProtocolError("missing or incomplete diagnostic token logprobs")


async def run_request(session: Any, semaphore: asyncio.Semaphore, request: dict,
                      index: int, offset_ns: int, epoch_ns: int, endpoint: str,
                      model: str, max_tokens: int, logprobs: bool = False,
                      fixed_output_length: bool = False) -> tuple[dict, dict]:
    record = {"type": "request", "request_id": request["id"], "request_index": index,
              "input_sha256": canonical_hash(request["messages"]),
              "expected_input_tokens": request["input_tokens"],
              "expected_prompt_token_ids_sha256": request.get("prompt_token_ids_sha256"),
              "diagnostic_requested": logprobs, "diagnostic_passed": None,
              "scheduled_ns": epoch_ns + offset_ns, "arrival_offset_ns": offset_ns,
              "arrival_wakeup_ns": None, "semaphore_acquired_ns": None,
              "send_ns": None, "headers_ns": None, "first_content_ns": None,
              "last_content_ns": None, "done_ns": None, "end_ns": None,
              "chunks": [], "usage": None, "usage_events": 0,
              "finish_reason": None, "saw_done": False, "http_status": None,
              "success": False, "error_type": None, "error_message": None}
    if fixed_output_length:
        record["requested_output_tokens"] = max_tokens
    parts: list[str] = []
    private_logprobs: list[dict] = []
    private_tokens: dict = {"prompt_token_ids": None, "completion_token_ids": [], "stop_reason": None}
    private_error = None
    await sleep_until_ns(record["scheduled_ns"])
    record["arrival_wakeup_ns"] = time.monotonic_ns()
    async with semaphore:
        record["semaphore_acquired_ns"] = time.monotonic_ns()
        payload = {"model": model, "messages": request["messages"], "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0,
                   "max_tokens": max_tokens, "n": 1,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if fixed_output_length:
            payload.update(ignore_eos=True, min_tokens=max_tokens, stop=[], stop_token_ids=[])
        if logprobs:
            payload.update(logprobs=True, top_logprobs=0, return_token_ids=True)
        record["payload_sha256"] = canonical_hash(payload)
        record["send_ns"] = time.monotonic_ns()
        try:
            async with session.post(endpoint, json=payload) as response:
                record["headers_ns"] = time.monotonic_ns()
                record["http_status"] = response.status
                if response.status != 200:
                    private_error = (await response.content.read(65536)).decode("utf-8", errors="replace")
                    raise ProtocolError(f"HTTP {response.status}")
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise ProtocolError("response is not text/event-stream")
                decoder = SSEDecoder()
                async for data in response.content.iter_any():
                    received_ns = time.monotonic_ns()
                    for event in decoder.feed(data):
                        _consume_event(event, received_ns, record, parts, private_logprobs, private_tokens)
                for event in decoder.feed(b"", final=True):
                    _consume_event(event, time.monotonic_ns(), record, parts, private_logprobs, private_tokens)
                _validate_completion(record)
                if fixed_output_length and (record["usage"]["completion_tokens"] != max_tokens
                                            or record["finish_reason"] != "length"):
                    raise ProtocolError("fixed output-length contract violated")
                if logprobs:
                    _validate_diagnostics(record, private_tokens, private_logprobs)
                    record["diagnostic_passed"] = True
                record["success"] = True
        except Exception as exc:
            if logprobs:
                record["diagnostic_passed"] = False
            record["error_type"] = type(exc).__name__
            # Exception text can contain server/user data: only retain it privately.
            private_error = private_error or str(exc) or repr(exc)
            record["error_message"] = str(exc) if isinstance(exc, ProtocolError) else "request failed; see private record"
        finally:
            record["end_ns"] = time.monotonic_ns()
    output = "".join(parts)
    record["output_sha256"] = sha256_bytes(output.encode())
    record["output_chars"] = len(output)
    record["output_utf8_bytes"] = len(output.encode())
    prompt_ids, output_ids = private_tokens["prompt_token_ids"], private_tokens["completion_token_ids"]
    record["prompt_token_ids_count"] = len(prompt_ids) if prompt_ids is not None else None
    record["prompt_token_ids_sha256"] = canonical_hash(prompt_ids) if prompt_ids is not None else None
    record["completion_token_ids_count"] = len(output_ids) if logprobs else None
    record["completion_token_ids_sha256"] = canonical_hash(output_ids) if logprobs else None
    record["diagnostic_logprob_count"] = sum(len(event["logprobs"]["content"]) for event in private_logprobs)
    record["diagnostic_clamped_or_placeholder_logprobs"] = sum(
        item["logprob"] == -9999.0 for event in private_logprobs for item in event["logprobs"]["content"])
    record["client_schedule_lag_ns"] = record["arrival_wakeup_ns"] - record["scheduled_ns"]
    record["client_semaphore_wait_ns"] = record["semaphore_acquired_ns"] - record["arrival_wakeup_ns"]
    record["client_send_delay_ns"] = record["send_ns"] - record["scheduled_ns"]
    record["response_from_schedule_ns"] = record["end_ns"] - record["scheduled_ns"]
    record["response_from_send_ns"] = record["end_ns"] - record["send_ns"]
    for base in ("schedule", "send"):
        start = record["scheduled_ns" if base == "schedule" else "send_ns"]
        record[f"ttft_from_{base}_ns"] = (None if record["first_content_ns"] is None
                                          else record["first_content_ns"] - start)
    count = record["usage"].get("completion_tokens") if isinstance(record["usage"], dict) else None
    record["tpot_stream_boundary_estimate_ns"] = (
        (record["last_content_ns"] - record["first_content_ns"]) / (count - 1)
        if _integer(count) and count > 1 and record["first_content_ns"] is not None else None)
    return record, {"type": "private_request", "request_id": request["id"],
                    "output": output, "error_detail": private_error,
                    "token_logprobs": private_logprobs,
                    **private_tokens,
                    "output_sha256": record["output_sha256"]}


async def run_workload(requests: list[dict], calendar: dict, *, endpoint: str,
                       model: str = "qwen3-full", max_tokens: int = 2048,
                       max_inflight: int = 256, timeout_s: float = 1800,
                       start_delay_s: float = 1.0,
                       on_record: Callable[[dict, dict], None] | None = None,
                       epoch_ns: int | None = None, logprobs: bool = False,
                       output_lengths: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    """One task per offered request; all tasks, including failures, are drained."""
    import aiohttp
    if output_lengths is not None:
        if (set(output_lengths) != {r["id"] for r in requests}
                or any(type(v) is not int or not 0 < v <= max_tokens for v in output_lengths.values())):
            raise ValueError("one positive bounded output target per request is required")
    if max_inflight <= 0 or timeout_s <= 0 or not math.isfinite(timeout_s):
        raise ValueError("max_inflight and finite timeout_s must be positive")
    if start_delay_s < 0 or not math.isfinite(start_delay_s):
        raise ValueError("invalid start delay")
    if not requests or len(requests) != len(calendar["arrival_offsets_ns"]):
        raise ValueError("nonempty requests and one arrival per request required")
    epoch_ns = epoch_ns if epoch_ns is not None else time.monotonic_ns() + int(start_delay_s * 1e9)
    epoch_wall_ns = time.time_ns() + epoch_ns - time.monotonic_ns()
    semaphore = asyncio.Semaphore(max_inflight)
    timeout = aiohttp.ClientTimeout(total=timeout_s, connect=min(60, timeout_s), sock_read=timeout_s)
    connector = aiohttp.TCPConnector(limit=max_inflight, limit_per_host=max_inflight)
    records = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector,
                                     trust_env=False, raise_for_status=False) as session:
        tasks = [asyncio.create_task(run_request(session, semaphore, request, index, offset,
                                                epoch_ns, endpoint, model,
                                                output_lengths[request["id"]] if output_lengths is not None else max_tokens,
                                                logprobs, output_lengths is not None))
                 for index, (request, offset) in enumerate(zip(requests, calendar["arrival_offsets_ns"]))]
        for task in asyncio.as_completed(tasks):
            public, private = await task
            records.append(public)
            if on_record is not None:
                on_record(public, private)
        await sleep_until_ns(epoch_ns + calendar.get("horizon_ns", calendar["arrival_offsets_ns"][-1]))
    last_end = max(r["end_ns"] for r in records)
    horizon = calendar.get("horizon_ns", calendar["arrival_offsets_ns"][-1])
    completion_ns = max(last_end, epoch_ns + horizon)
    timing = {"epoch_monotonic_ns": epoch_ns, "epoch_wall_ns_approx": epoch_wall_ns,
              "offered_horizon_ns": horizon, "last_request_end_ns": last_end,
              "completion_ns": completion_ns, "completed_elapsed_ns": completion_ns - epoch_ns,
              "drain_ns": max(0, last_end - (epoch_ns + horizon)),
              "client_finished_ns": time.monotonic_ns()}
    return sorted(records, key=lambda r: r["request_index"]), timing


def _open_new(path: Path, private: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    return os.fdopen(fd, "w", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import aiohttp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    arrivals = parser.add_mutually_exclusive_group(required=True)
    arrivals.add_argument("--calendar", type=Path)
    arrivals.add_argument("--rate", type=float, help="Synthetic pilot requests/s; 0=simultaneous burst (not external arrivals)")
    parser.add_argument("--base-url", required=True, help="OpenAI base URL, e.g. http://host:8000/v1")
    parser.add_argument("--model", default="qwen3-full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--block-id", required=True, help="Independent content/arrival block pair")
    parser.add_argument("--load-id", default="default")
    parser.add_argument("--pair-id", help="Matched arm pair ID; default block-id:load-id")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-lengths", type=Path, help="Frozen request-ID to exact output-token targets; suppress early stopping equally for all arms")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--start-delay-s", type=float, default=1)
    parser.add_argument("--logprobs", action="store_true", help="Correctness diagnostic: validate/save prompt/output IDs and finite generated token logprobs privately")
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or args.max_model_len <= args.max_tokens:
        parser.error("invalid token limits")
    requests, calendar, hashes = load_workload(args.requests, args.calendar, args.max_tokens, args.max_model_len, args.rate)
    length_bytes = args.output_lengths.read_bytes() if args.output_lengths is not None else None
    output_lengths = json.loads(length_bytes) if length_bytes is not None else None
    if output_lengths is not None and (not isinstance(output_lengths, dict)
            or set(output_lengths) != {r["id"] for r in requests}
            or any(type(v) is not int or not 0 < v <= args.max_tokens for v in output_lengths.values())):
        parser.error("one positive bounded output target per request is required")
    if args.output.resolve() == args.private_output.resolve():
        parser.error("public and private outputs must be different files")
    if args.output.exists() or args.private_output.exists():
        parser.error("outputs must not exist; use fresh names to preserve previous records")
    header = {"type": "run_header", "schema_version": 1, "run_id": args.run_id,
              "arm": args.arm, "block_id": args.block_id, "load_id": args.load_id,
              "pair_id": args.pair_id or f"{args.block_id}:{args.load_id}", **hashes,
              "request_count": len(requests), "model": args.model,
              "python_version": platform.python_version(), "aiohttp_version": aiohttp.__version__,
              "client_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
              "endpoint": args.base_url.rstrip("/") + "/chat/completions",
              "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
              "temperature": 0, "enable_thinking": False, "ignore_eos": output_lengths is not None,
              "output_lengths_sha256": sha256_bytes(length_bytes) if length_bytes is not None else None,
              "output_length_mode": "fixed_previous_off_reference" if output_lengths is not None else "natural_eos",
              "max_inflight": args.max_inflight, "http_timeout_s": args.timeout_s,
              "logprobs": args.logprobs, "arrival_kind": calendar.get("kind", "external_calendar"),
              "diagnostic_return_token_ids": args.logprobs,
              "synthetic_rate_requests_per_s": args.rate,
              "client_time_origin": "monotonic_ns; epoch is shared across all requests",
              "timestamp_semantics": "send=submission to aiohttp; content timestamps=SSE receipt, not individual tokens",
              "server_queue_and_service": "not observable from HTTP; do not infer them from response minus TTFT",
              "tpot_semantics": "(last nonempty content receipt - first)/(server completion_tokens - 1); stream-boundary estimate only",
              "created_wall_ns": time.time_ns()}
    with _open_new(args.output) as public, _open_new(args.private_output, private=True) as private:
        def write(public_record: dict, private_record: dict) -> None:
            public.write(json.dumps(public_record, ensure_ascii=False, allow_nan=False) + "\n")
            public.flush()
            private.write(json.dumps(private_record, ensure_ascii=False, allow_nan=False) + "\n")
            private.flush()
        public.write(json.dumps(header) + "\n")
        public.flush()
        records, timing = asyncio.run(run_workload(
            requests, calendar, endpoint=header["endpoint"], model=args.model,
            max_tokens=args.max_tokens, max_inflight=args.max_inflight,
            timeout_s=args.timeout_s, start_delay_s=args.start_delay_s, on_record=write, logprobs=args.logprobs,
            output_lengths=output_lengths))
        from full_serving.analyze import summarize_run
        summary = summarize_run(records, timing)
        public.write(json.dumps({"type": "run_summary", **timing, **summary}, allow_nan=False) + "\n")
        public.flush()
    print(json.dumps({"run_id": args.run_id, "output": str(args.output), **summary}, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
