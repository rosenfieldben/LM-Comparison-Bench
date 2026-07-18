"""Core model-calling logic. Pure functions over an injected httpx client."""

import json
import math
import os
import socket
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx


def as_token_count(value: object) -> int | None:
    """A token count usable downstream: a non-negative int, else None.

    The never-raises contract extends to field types. A provider once
    returned "prompt_tokens": "n/a"; persisted raw, that one row made
    /compare 500 at its own response boundary and poisoned
    GET /runs/{id} into a permanent 500. bool is excluded explicitly
    because isinstance(True, int) passes. Junk is rejected rather than
    coerced: a guessed count would price a cost from fiction.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def as_text(value: object) -> str | None:
    """str or None; any other type becomes None, same contract as counts."""
    return value if isinstance(value, str) else None


def as_metric(value: object) -> float | None:
    """A finite float measurement or None; bools and junk become None.

    Used by the store's repair-on-read: measurement columns written
    before ingestion normalization may carry non-numeric junk.
    """
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


# Env-overridable as a test seam so the browser harness can point the
# real app at a stub upstream; not a configuration feature.
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
MODELS_URL = os.environ.get("MODELS_URL", "https://openrouter.ai/api/v1/models")

# Reaching OpenRouter should be fast; a slow connect is a real failure.
CONNECT_TIMEOUT_S = 10.0

# Extended-budget reasoning can legitimately sit silent for minutes
# between visible bytes; five minutes of true wire silence is failure
# by any standard.
STREAM_READ_TIMEOUT_S = 300.0

# Non-streaming path, where the single read covers the entire
# completion including all reasoning time, so it gets more headroom
# than the between-chunk gap.
COMPLETION_READ_TIMEOUT_S = 180.0

# Streaming: connect fast, tolerate long silent gaps between chunks.
STREAM_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S, read=STREAM_READ_TIMEOUT_S, write=30.0, pool=10.0
)

# Non-streaming: connect fast, one long read for the whole completion.
COMPLETION_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S, read=COMPLETION_READ_TIMEOUT_S, write=30.0, pool=10.0
)

# Reasoning models spend thousands of hidden tokens before visible
# output and max_tokens covers both, so the budget must dwarf a
# plausible reasoning burn. Two named tiers instead of a free integer:
# everyday prompts and hard problems are the two real regimes of use,
# a free field invites typos with dollar consequences, and history
# stays legible when every run carries one of two labels. Standard
# keeps worst-case cost per call around a dollar on the priciest
# models; extended exists for hard problems where reasoning empties
# the standard budget before any visible answer appears.
BUDGET_STANDARD = 16384
BUDGET_EXTENDED = 65536

# Boot must not hang on pricing; the bench works offline, cost display
# is the only thing a failed fetch costs.
PRICES_TIMEOUT_S = 10.0

# OpenRouter's default routing is price-weighted, which sends
# open-weight models to their cheapest hosts, and the cheapest hosts
# are the flakiest. Sorting by throughput biases routing to the
# serious hosts at somewhat higher cost. Unlike a reasoning parameter
# this does not alter what the model itself does, only who serves it;
# and since quantization varies by host, it also stabilizes WHAT is
# being measured, not just how reliably it answers.
PROVIDER_PREFS = {"sort": "throughput"}

# Extended-budget runs go silent for minutes while providers reason,
# and NAT/middlebox idle timers cull flows that move no bytes. That
# culling surfaced as sequential failures with mixed ReadError and
# stall signatures partway through a lineup. OpenRouter's SSE comment
# keepalives evidently do not flow reliably during deep provider-side
# reasoning, so the fix sits below the application layer: OS-level TCP
# probes keep the flow visibly alive when no application bytes move.
# 30s stays comfortably under every common NAT idle window while
# adding negligible traffic.
KEEPALIVE_IDLE_S = 30
KEEPALIVE_INTERVAL_S = 30


def keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Socket options enabling TCP keepalive on the current platform.

    The idle-before-first-probe constant is platform-named: TCP_KEEPIDLE
    on Linux, TCP_KEEPALIVE on macOS. This repo runs on both a Mac and
    Linux sandboxes, so hasattr picks whichever exists rather than
    hardcoding one and silently doing nothing on the other.
    """
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPIDLE"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, KEEPALIVE_IDLE_S))
    elif hasattr(socket, "TCP_KEEPALIVE"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, KEEPALIVE_IDLE_S))
    if hasattr(socket, "TCP_KEEPINTVL"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, KEEPALIVE_INTERVAL_S))
    return options


async def fetch_catalog(client: httpx.AsyncClient) -> dict[str, Any]:
    """One boot-time snapshot of OpenRouter's model catalog.

    Returns {"fetched": bool, "models": [...], "prices": {...}} where
    models entries carry id, name, context_length, prompt_price,
    completion_price and max_completion_tokens (missing fields degrade
    to None, never breaking the whole catalog) and prices is the
    {model_id: {prompt, completion}} map cost_usd consumes. One
    upstream call feeds both.
    Never raises: this runs at startup and a catalog outage must not
    stop the bench from booting; fetched=false is how the frontend
    tells an offline boot from an empty catalog.
    """
    offline = {"fetched": False, "models": [], "prices": {}}
    try:
        response = await client.get(MODELS_URL, timeout=PRICES_TIMEOUT_S)
        if response.status_code != 200:
            return offline
        entries = response.json()["data"]
        if not isinstance(entries, list):
            return offline
    except (httpx.HTTPError, ValueError, LookupError, TypeError):
        return offline

    models = []
    prices = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        model = {
            "id": entry["id"],
            "name": entry.get("name") if isinstance(entry.get("name"), str) else None,
            "context_length": (
                entry.get("context_length")
                if isinstance(entry.get("context_length"), int)
                else None
            ),
            "prompt_price": None,
            "completion_price": None,
            "max_completion_tokens": None,
        }
        # OpenRouter publishes a per-model completion cap under
        # top_provider where known. The budget clamp needs it: sending
        # a budget above the cap is a hard 400 from some providers.
        top = entry.get("top_provider")
        if isinstance(top, dict):
            cap = top.get("max_completion_tokens")
            # A non-bool int strictly above zero only. isinstance(True, int)
            # is true in Python, so a provider sending true would otherwise
            # become a cap of 1 that clamps every budget to a single token;
            # zero and negatives are not real caps either.
            if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
                model["max_completion_tokens"] = cap
        # Prices arrive as strings in USD per token. Malformed pricing
        # degrades this entry's price fields rather than dropping the
        # entry or the whole map. Non-finite and negative prices are
        # malformed too: a NaN price yields a NaN cost, and a NaN summed
        # into accumulated spend makes the ceiling comparison permanently
        # false, silently disabling it. Raising inside the try reuses the
        # single degrade path, matching as_metric's finiteness contract.
        try:
            prompt_price = float(entry["pricing"]["prompt"])
            completion_price = float(entry["pricing"]["completion"])
            if not (math.isfinite(prompt_price) and math.isfinite(completion_price)):
                raise ValueError("non-finite price")
            if prompt_price < 0 or completion_price < 0:
                raise ValueError("negative price")
            model["prompt_price"] = prompt_price
            model["completion_price"] = completion_price
            prices[entry["id"]] = {
                "prompt": prompt_price,
                "completion": completion_price,
            }
        except (KeyError, TypeError, ValueError):
            model["prompt_price"] = None
            model["completion_price"] = None
        models.append(model)
    return {"fetched": True, "models": models, "prices": prices}


def _flatten_content(content: object) -> str | None:
    """Collapse a message content value to plain text or None.

    Content-parts lists (multimodal providers) flatten to their text
    parts. Anything that is not a non-empty str after that returns None:
    response_text is str or None by contract, and a raw list would crash
    the sqlite bind in save_run and roll back the whole run.
    """
    if isinstance(content, list):
        # isinstance on the value, not .get with a default: a part with
        # an explicit null text ({"text": null}) returns None from .get
        # (the default only covers missing keys) and would crash the
        # join, escaping the never-raises contract.
        content = "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if isinstance(content, str) and content:
        return content
    return None


async def run_model(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    max_tokens: int = BUDGET_STANDARD,
) -> dict[str, Any]:
    """Send one chat completion to OpenRouter and return a flat result dict.

    Never raises. A comparison run fans out to several models and one
    failure must not sink the others, so every error path collapses into
    the error field of an otherwise well-formed result.

    The result echoes max_tokens because a truncated answer at 16k and
    one at 65k are different experiments; persistence must record which
    budget was actually sent, clamping included.
    """
    result: dict[str, Any] = {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
        "max_tokens": max_tokens,
        "generation_id": None,
        "finish_reason": None,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "provider": PROVIDER_PREFS,
    }

    start = time.perf_counter()
    try:
        response = await client.post(
            OPENROUTER_URL, json=payload, timeout=COMPLETION_TIMEOUT
        )
    # ConnectTimeout before the generic catch: it fires after
    # CONNECT_TIMEOUT_S, and the generic message would claim a wait
    # eighteen times longer than what actually happened.
    except httpx.ConnectTimeout:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"could not connect within {CONNECT_TIMEOUT_S:.0f}s"
        return result
    except httpx.TimeoutException:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"no response within {COMPLETION_READ_TIMEOUT_S:.0f}s"
        return result
    except httpx.HTTPError as exc:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"request failed: {type(exc).__name__}"
        return result
    # Latency covers the HTTP round trip only. JSON parsing happens below,
    # outside the clock, so big responses do not inflate the measurement.
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)

    if response.status_code != 200:
        result["error"] = f"HTTP {response.status_code} from OpenRouter"
        return result

    try:
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, LookupError, TypeError):
        result["error"] = "malformed response from OpenRouter"
        return result

    # Provenance, both best-effort. The generation id keys OpenRouter's
    # generation API, which records the actual provider, quantization
    # and authoritative cost, so persisting it makes this run auditable
    # after the fact. The finish_reason is the provider's own verdict
    # on why output ended; until now it survived only inside
    # synthesized error strings, and budget analysis needs it on
    # successful runs too.
    gen_id = data.get("id")
    if isinstance(gen_id, str) and gen_id:
        result["generation_id"] = gen_id
    reason = choice.get("finish_reason")
    if isinstance(reason, str) and reason:
        result["finish_reason"] = reason

    text = _flatten_content(content)
    if text:
        result["response_text"] = text
    else:
        # Some providers return 200 with null content on refusals. Surface
        # that as an error so every result carries either text or an error,
        # a contract the frontend relies on to pick a render state. Non-str
        # oddities land here too rather than leaking into response_text.
        result["error"] = (
            f"empty response (finish_reason: {result['finish_reason'] or 'unknown'})"
        )

    # Some providers omit usage. Report None rather than guessing counts.
    # isinstance instead of `or {}`: a truthy non-dict like "n/a" would
    # pass the truthiness guard and raise on .get, and this function must
    # never raise.
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    result["prompt_tokens"] = as_token_count(usage.get("prompt_tokens"))
    result["completion_tokens"] = as_token_count(usage.get("completion_tokens"))
    return result


async def stream_model(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    max_tokens: int = BUDGET_STANDARD,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one chat completion, yielding delta and done event dicts.

    Yields {"type": "delta", "text": chunk} per content delta, then
    exactly one {"type": "done", "result": {...}}. Like run_model, this
    never raises: every failure path lands in the done result's error
    field, alongside whatever text accumulated before the failure. Text
    is accumulated here so the done event is always self-sufficient for
    persistence, even when the consumer dropped deltas.

    The read timeout is a gap between chunks, not total stream
    duration: a slow model legitimately streams for minutes, and a
    reasoning model can think silently for over a minute before its
    first visible delta, but a silence past STREAM_READ_TIMEOUT_S
    still means something is wrong.
    """
    result: dict[str, Any] = {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
        "ttft_ms": None,
        "max_tokens": max_tokens,
        "generation_id": None,
        "finish_reason": None,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "provider": PROVIDER_PREFS,
        "stream": True,
        # Without this the final chunk carries no usage block and token
        # counts (and therefore cost) would be lost on streamed runs.
        "stream_options": {"include_usage": True},
    }
    text_parts: list[str] = []
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    def done(error: str | None) -> dict[str, Any]:
        result["latency_ms"] = elapsed_ms()
        if text_parts:
            result["response_text"] = "".join(text_parts)
        result["error"] = error
        # Same guard as run_model: a clean stream that produced no text
        # must still carry an error so the frontend has a render state.
        if result["response_text"] is None and error is None:
            result["error"] = (
                "empty response (finish_reason: "
                f"{result['finish_reason'] or 'unknown'})"
            )
        return {"type": "done", "result": result}

    try:
        async with client.stream(
            "POST", OPENROUTER_URL, json=payload, timeout=STREAM_TIMEOUT
        ) as response:
            if response.status_code != 200:
                yield done(f"HTTP {response.status_code} from OpenRouter")
                return
            async for line in response.aiter_lines():
                # SSE comments (OpenRouter keep-alives) and blank lines
                # carry nothing.
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise ValueError("chunk is not an object")
                except ValueError:
                    yield done("malformed stream from OpenRouter")
                    return

                # The generation id keys OpenRouter's generation API for
                # a post-hoc audit of provider, quantization and cost;
                # every chunk repeats the same id, so the first one that
                # carries it settles the field.
                if result["generation_id"] is None:
                    gen_id = chunk.get("id")
                    if isinstance(gen_id, str) and gen_id:
                        result["generation_id"] = gen_id

                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    result["prompt_tokens"] = as_token_count(usage.get("prompt_tokens"))
                    result["completion_tokens"] = as_token_count(
                        usage.get("completion_tokens")
                    )

                # OpenRouter reports mid-stream failures as an in-band
                # error object on the 200 stream. Without this check the
                # frame has no choices, falls into the guard below, and
                # the real upstream reason is silently lost while the
                # stream ends looking like a clean success.
                err = chunk.get("error")
                if isinstance(err, dict):
                    detail = err.get("message") or err.get("code") or "unknown"
                    yield done(f"upstream error: {detail}")
                    return

                try:
                    choice = chunk["choices"][0]
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str) and reason:
                        # The provider sends its verdict on the closing
                        # chunk; the final one seen is the one recorded.
                        result["finish_reason"] = reason
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                except (LookupError, TypeError, AttributeError):
                    # Usage-only or oddly shaped chunks contribute no text.
                    continue

                text = _flatten_content(content)
                if text:
                    if result["ttft_ms"] is None:
                        result["ttft_ms"] = elapsed_ms()
                    text_parts.append(text)
                    yield {"type": "delta", "text": text}
    # Same split as run_model: a connect timeout is not a stall, and
    # "no data for 300s" would misreport a 10s handshake failure.
    except httpx.ConnectTimeout:
        yield done(f"could not connect within {CONNECT_TIMEOUT_S:.0f}s")
        return
    except httpx.TimeoutException:
        yield done(f"stream stalled: no data for {STREAM_READ_TIMEOUT_S:.0f}s")
        return
    except httpx.HTTPError as exc:
        yield done(f"request failed: {type(exc).__name__}")
        return

    yield done(None)
