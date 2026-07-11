"""Stub OpenRouter for the browser harness.

Serves the two endpoints the bench calls, with streaming personalities
keyed by model id so one stub exercises every frontend state the
critical-path suite needs: fast, slow (elapsed indicator and late
arrival), flaky (rerun), null content (empty-response path) and HTML
payloads (injection probe). Unknown model ids get the fast personality
with model-specific text, so tests can fan out to any width without
growing the catalog.

/_test/requests exposes every recorded /chat/completions payload; it
exists for test assertions (budget clamping) and is not part of the
OpenRouter surface being stubbed.
"""

import asyncio
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# Priced so cost metrics fill in: usage below is 13 in / 8 out, giving
# 13e-6 + 16e-6 = 2.9e-5 USD per completed run. stub/capped publishes a
# completion cap far below the extended budget so the clamp has
# something to bite on.
CATALOG = {
    "data": [
        {
            "id": model_id,
            "name": model_id.split("/")[1].title(),
            "context_length": 8192,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            **extra,
        }
        for model_id, extra in [
            ("stub/fast", {}),
            ("stub/slow", {}),
            ("stub/flaky", {}),
            ("stub/null", {}),
            ("stub/html", {}),
            ("stub/capped", {"top_provider": {"max_completion_tokens": 4096}}),
        ]
    ]
}

USAGE = {"prompt_tokens": 13, "completion_tokens": 8}

# Deltas that must render as literal text everywhere, never as markup.
HTML_DELTAS = ["<img src=x onerror=alert(1)>", " and ", "<b>bold?</b>"]


def sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def reply_text(model: str) -> str:
    return f"reply from {model}"


def build_app() -> Starlette:
    # Closure state instead of module globals: each harness session
    # builds its own app, so flaky's fail-once behavior and the request
    # log reset with the stub process.
    state = {"requests": [], "flaky_failed": False, "flaky_slow_failed": False}

    async def models(request):
        return JSONResponse(CATALOG)

    def text_stream(model: str, text: str, gen_suffix: str, delay: float = 0.0):
        # Word-by-word deltas so streaming rendering is actually
        # exercised, not just a single append.
        words = text.split(" ")
        deltas = [w if i == len(words) - 1 else w + " " for i, w in enumerate(words)]

        async def gen():
            if delay:
                await asyncio.sleep(delay)
            for i, chunk in enumerate(deltas):
                yield sse({
                    "id": f"gen-stub-{gen_suffix}",
                    "choices": [{"delta": {"content": chunk}}],
                })
                await asyncio.sleep(0.02)
            yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            yield sse({"choices": [], "usage": USAGE})
            yield b"data: [DONE]\n\n"

        return gen()

    def personality_stream(model: str):
        if model == "stub/slow":
            # Two seconds of true wire silence before the first delta:
            # long enough for the UI's elapsed counter to tick and for
            # a fast column to finish first.
            return text_stream(model, "slow " + reply_text(model), "slow", delay=2.0)
        if model == "stub/flaky" and not state["flaky_failed"]:
            state["flaky_failed"] = True

            async def gen():
                yield sse({"error": {"code": 502, "message": "stub flaky failure"}})
                yield b"data: [DONE]\n\n"

            return gen()
        if model == "stub/flaky-slow":
            # Fails once like stub/flaky, but the successful retry is
            # slow: the window the view-integrity tests need to start a
            # superseding run while a rerun is still in flight.
            if not state["flaky_slow_failed"]:
                state["flaky_slow_failed"] = True

                async def gen():
                    yield sse({"error": {"code": 502, "message": "stub flaky failure"}})
                    yield b"data: [DONE]\n\n"

                return gen()
            return text_stream(model, reply_text(model), "flaky-slow", delay=2.0)
        if model == "stub/null":

            async def gen():
                await asyncio.sleep(0.05)
                yield sse({"choices": [{"delta": {}, "finish_reason": "content_filter"}]})
                yield sse({"choices": [], "usage": USAGE})
                yield b"data: [DONE]\n\n"

            return gen()
        if model == "stub/html":

            async def gen():
                for chunk in HTML_DELTAS:
                    yield sse({
                        "id": "gen-stub-html",
                        "choices": [{"delta": {"content": chunk}}],
                    })
                    await asyncio.sleep(0.02)
                yield sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                yield sse({"choices": [], "usage": USAGE})
                yield b"data: [DONE]\n\n"

            return gen()
        # stub/fast and anything unrecognized: quick and correct.
        return text_stream(model, reply_text(model), model.split("/")[-1])

    def non_stream_body(model: str) -> JSONResponse:
        if model == "stub/flaky" and not state["flaky_failed"]:
            state["flaky_failed"] = True
            return JSONResponse({"error": "stub flaky failure"}, status_code=502)
        if model == "stub/null":
            return JSONResponse({
                "id": "gen-stub-null",
                "choices": [{
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "content_filter",
                }],
                "usage": USAGE,
            })
        text = "".join(HTML_DELTAS) if model == "stub/html" else reply_text(model)
        return JSONResponse({
            "id": f"gen-stub-{model.split('/')[-1]}",
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": USAGE,
        })

    async def completions(request):
        payload = await request.json()
        state["requests"].append(payload)
        model = payload["model"]
        if payload.get("stream"):
            return StreamingResponse(
                personality_stream(model), media_type="text/event-stream"
            )
        return non_stream_body(model)

    async def recorded_requests(request):
        return JSONResponse({"requests": state["requests"]})

    return Starlette(routes=[
        Route("/api/v1/models", models),
        Route("/api/v1/chat/completions", completions, methods=["POST"]),
        Route("/_test/requests", recorded_requests),
    ])
