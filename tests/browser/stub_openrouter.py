"""Stub OpenRouter for the browser harness.

Serves the two endpoints the bench calls, with streaming personalities
keyed by model id so one stub exercises every frontend state the
critical-path suite needs: fast, slow (elapsed indicator and late
arrival), flaky (rerun), null content (empty-response path), HTML
payloads (injection probe), three billing shapes (no usage object at
all, a NaN cost, a negative cost), and a data-policy refusal (no endpoint
matches the requested routing). Unknown model ids get the fast
personality with model-specific text, so tests can fan out to any width
without growing the catalog.

/_test/requests exposes every recorded /chat/completions payload; it
exists for test assertions (budget clamping) and is not part of the
OpenRouter surface being stubbed.
"""

import asyncio
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Spelled out rather than imported from bench.models: this stub stands in
# for the external contract, and importing the app's own constant would
# make a typo in it invisible, since both sides would then agree on the
# wrong name.
GENERATION_ID_HEADER = "X-Generation-Id"

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
            # Priced in the catalog on purpose: their billed figure is
            # poisoned, so the card must fall back to the estimate rather
            # than showing nothing.
            ("stub/nancost", {}),
            ("stub/negcost", {}),
        ]
    ]
}

# The billed figure is deliberately NOT 2.9e-5: it has to be
# distinguishable from the catalog estimate above, or a test asserting the
# card shows billed cost would pass on the estimate.
BILLED_COST = 0.000031

USAGE = {
    "prompt_tokens": 13,
    "completion_tokens": 8,
    "cost": BILLED_COST,
    # BYOK-only upstream figure. Present so the parser is exercised against
    # a payload that carries it, and deliberately not persisted anywhere.
    "cost_details": {"upstream_inference_cost": 0},
    "completion_tokens_details": {"reasoning_tokens": 5},
    "prompt_tokens_details": {"cached_tokens": 2},
}

# The host OpenRouter routed to, echoed on every chunk and on the
# non-streaming body, the way the real API reports it.
PROVIDER = "StubHost"

# The provider's own word for why generation ended, which OpenRouter
# passes through beside its normalized "stop".
NATIVE_FINISH_REASON = "STOP_SEQUENCE"

# Deltas that must render as literal text everywhere, never as markup.
HTML_DELTAS = ["<img src=x onerror=alert(1)>", " and ", "<b>bold?</b>"]


def sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def reply_text(model: str) -> str:
    return f"reply from {model}"


def usage_for(model: str):
    """The usage object this model's personality reports, or None for none.

    The poisoned variants exist to prove the money rule end to end: a NaN
    or negative cost must degrade to None, stay out of the accumulator, and
    never turn a paid result into an error. json.dumps writes a bare NaN
    token, which is exactly what a misbehaving provider puts on the wire
    and what the client's json parsing accepts on the way back in, so this
    exercises the real ingestion path rather than a hand-built dict.
    """
    if model == "stub/nousage":
        return None
    if model == "stub/nancost":
        return {**USAGE, "cost": float("nan")}
    if model == "stub/negcost":
        return {**USAGE, "cost": -1.0}
    return USAGE


def finish_chunk() -> bytes:
    """The chunk that ends generation, shaped like OpenRouter's.

    provider rides the chunk and native_finish_reason rides the choice,
    beside the normalized finish_reason, which is where the real API puts
    them.
    """
    return sse(
        {
            "provider": PROVIDER,
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "native_finish_reason": NATIVE_FINISH_REASON,
                }
            ],
        }
    )


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
                yield sse(
                    {
                        "id": f"gen-stub-{gen_suffix}",
                        "provider": PROVIDER,
                        "choices": [{"delta": {"content": chunk}}],
                    }
                )
                await asyncio.sleep(0.02)
            yield finish_chunk()
            usage = usage_for(model)
            if usage is not None:
                yield sse({"choices": [], "usage": usage})
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
                yield sse(
                    {
                        "provider": PROVIDER,
                        "choices": [{"delta": {}, "finish_reason": "content_filter"}],
                    }
                )
                yield sse({"choices": [], "usage": USAGE})
                yield b"data: [DONE]\n\n"

            return gen()
        if model == "stub/html":

            async def gen():
                for chunk in HTML_DELTAS:
                    yield sse(
                        {
                            "id": "gen-stub-html",
                            "provider": PROVIDER,
                            "choices": [{"delta": {"content": chunk}}],
                        }
                    )
                    await asyncio.sleep(0.02)
                yield finish_chunk()
                yield sse({"choices": [], "usage": USAGE})
                yield b"data: [DONE]\n\n"

            return gen()
        if model == "stub/nopolicy":
            # No endpoint satisfies the requested data policy. OpenRouter
            # reports this like any other routing failure, as an in-band
            # error object on the 200 stream, so the bench needs no new
            # code path: the card shows it the way it shows a provider
            # outage. That inheritance is the point of the test.
            async def gen():
                yield sse(
                    {
                        "error": {
                            "code": 404,
                            "message": ("No endpoints found matching your data policy"),
                        }
                    }
                )
                yield b"data: [DONE]\n\n"

            return gen()
        if model.startswith("stub/stall"):
            # Streams a little text, then stalls indefinitely with no
            # [DONE] and no finish reason: the window a user Stop must be
            # able to interrupt. The prefix lets several distinct stall
            # ids fill the upstream slots so a further one queues. The
            # sleeping generator is cancelled cleanly when the client
            # disconnects.
            async def gen():
                for chunk in ("partial ", "text"):
                    yield sse(
                        {
                            "id": "gen-stub-stall",
                            "choices": [{"delta": {"content": chunk}}],
                        }
                    )
                    await asyncio.sleep(0.02)
                await asyncio.sleep(3600)

            return gen()
        # stub/fast and anything unrecognized: quick and correct.
        return text_stream(model, reply_text(model), model.split("/")[-1])

    def non_stream_body(model: str, headers: dict) -> Response:
        if model == "stub/flaky" and not state["flaky_failed"]:
            state["flaky_failed"] = True
            return JSONResponse({"error": "stub flaky failure"}, status_code=502)
        if model == "stub/null":
            return JSONResponse(
                {
                    "id": "gen-stub-null",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": None},
                            "finish_reason": "content_filter",
                        }
                    ],
                    "usage": USAGE,
                },
                headers=headers,
            )
        text = "".join(HTML_DELTAS) if model == "stub/html" else reply_text(model)
        body = {
            "id": f"gen-stub-{model.split('/')[-1]}",
            "provider": PROVIDER,
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "native_finish_reason": NATIVE_FINISH_REASON,
                }
            ],
        }
        usage = usage_for(model)
        if usage is not None:
            body["usage"] = usage
        # allow_nan so the poisoned personalities put a bare NaN on the
        # wire, the way a misbehaving provider would; Starlette's default
        # encoder would raise on it instead.
        return Response(
            json.dumps(body, allow_nan=True),
            media_type="application/json",
            headers=headers,
        )

    async def completions(request):
        payload = await request.json()
        state["requests"].append(payload)
        model = payload["model"]
        # The real API returns the generation id in a response header,
        # available before any chunk. The stall personality streams forever
        # and never emits a chunk id at all, so without this header a run
        # stopped mid-stream would have nothing to persist, which is the
        # exact case the header exists to cover.
        headers = {GENERATION_ID_HEADER: f"gen-hdr-{model.split('/')[-1]}"}
        if payload.get("stream"):
            return StreamingResponse(
                personality_stream(model),
                media_type="text/event-stream",
                headers=headers,
            )
        return non_stream_body(model, headers)

    async def recorded_requests(request):
        return JSONResponse({"requests": state["requests"]})

    return Starlette(
        routes=[
            Route("/api/v1/models", models),
            Route("/api/v1/chat/completions", completions, methods=["POST"]),
            Route("/_test/requests", recorded_requests),
        ]
    )
