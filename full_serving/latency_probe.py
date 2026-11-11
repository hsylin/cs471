"""Opt-in diagnostic API boundaries. Never logs prompts, outputs or token IDs.

The middleware is absent from final timing sessions. Durations within a process
use monotonic time; wall anchors are for approximate alignment only.
"""
import asyncio
import contextvars
import functools
import inspect
import json
import os
from pathlib import Path
import socket
import time
import uuid

_CURRENT = contextvars.ContextVar('cs471_latency_request', default=None)
_INSTALLED = False


def event(name, **fields):
    rows = _CURRENT.get()
    if rows is not None:
        rows.append({'event': name, 'mono_ns': time.monotonic_ns(), **fields})


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import DPLBAsyncMPClient, DPAsyncMPClient

    def wrap_async(cls, name):
        original = getattr(cls, name)
        assert inspect.iscoroutinefunction(original), (cls, name)
        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            fields = {}
            if name in ('_add_request', 'add_request_async') and args:
                fields['engine_request_id'] = args[0].request_id
            label = cls.__name__ + '.' + name
            event(label + '.begin', **fields)
            try:
                return await original(self, *args, **kwargs)
            finally:
                event(label + '.end', **fields)
        setattr(cls, name, wrapped)

    for cls, name in ((OpenAIServingChat, 'render_chat_request'),
                      (OpenAIServingChat, 'create_chat_completion'),
                      (AsyncLLM, '_add_request'),
                      (DPAsyncMPClient, 'add_request_async')):
        wrap_async(cls, name)
    original_choose = DPLBAsyncMPClient.get_core_engine_for_request
    @functools.wraps(original_choose)
    def choose(self, request):
        result = original_choose(self, request)
        event('dp_selection', engine_request_id=request.request_id,
              engine_index=self.core_engines.index(result))
        return result
    DPLBAsyncMPClient.get_core_engine_for_request = choose
    original_generate = AsyncLLM.generate
    @functools.wraps(original_generate)
    async def generate(self, *args, **kwargs):
        event('generate.begin')
        first = True
        try:
            async for output in original_generate(self, *args, **kwargs):
                if first:
                    event('generate.first_output', engine_request_id=output.request_id)
                    first = False
                yield output
        finally:
            event('generate.end')
    AsyncLLM.generate = generate
    _INSTALLED = True


class LatencyMiddleware:
    def __init__(self, app):
        self.app = app
        install()
        self.root = Path(os.environ['FULL_SERVING_RUN_DIR'])
        self.lag = []
        self.task = None

    async def monitor_loop(self):
        expected = time.monotonic_ns() + 10_000_000
        while True:
            await asyncio.sleep(.01)
            now = time.monotonic_ns()
            late = now - expected
            if late > 5_000_000:
                self.lag.append({'mono_ns': now, 'late_ns': late})
            expected = now + 10_000_000

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if self.task is None:
            self.task = asyncio.create_task(self.monitor_loop())
        if scope['path'] != '/v1/chat/completions':
            return await self.app(scope, receive, send)
        rows = []
        token = _CURRENT.set(rows)
        anchor_mono, anchor_wall = time.monotonic_ns(), time.time_ns()
        first_body = True
        async def observed_receive():
            message = await receive()
            if message['type'] == 'http.request' and not message.get('more_body'):
                event('http.request_body_complete')
            return message
        async def observed_send(message):
            nonlocal first_body
            if message['type'] == 'http.response.start':
                event('http.response_headers', status=message['status'])
            elif message['type'] == 'http.response.body' and message.get('body') and first_body:
                event('http.first_body')
                first_body = False
            await send(message)
        event('http.received')
        try:
            await self.app(scope, observed_receive, observed_send)
        finally:
            event('http.finished')
            _CURRENT.reset(token)
            record = {'diagnostic_id': uuid.uuid4().hex, 'pid': os.getpid(),
                      'host': socket.gethostname(), 'mono_anchor_ns': anchor_mono,
                      'wall_anchor_ns': anchor_wall, 'events': rows,
                      'event_loop_lag': [x for x in self.lag if x['mono_ns'] >= anchor_mono]}
            with (self.root / f'latency-api-{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
