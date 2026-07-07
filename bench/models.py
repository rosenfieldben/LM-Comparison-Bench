"""Core model-calling logic. Pure functions over an injected httpx client."""

import json
import time

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Reaching OpenRouter should be fast; a slow connect is a real failure.
CONNECT_TIMEOUT_S = 10.0

# Max silent gap between chunks on the streaming path. Reasoning models
# think silently well past 30s before the first visible delta, so the
# gap must accommodate hidden reasoning; a silence longer than two
# minutes still means something is wrong.
STREAM_READ_TIMEOUT_S = 120.0

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
# plausible reasoning burn. 16384 keeps worst-case cost per call
# around a dollar on the priciest models while ending the
# empty-response truncations that 4096 caused.
MAX_TOKENS = 16384

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


async def fetch_catalog(client: httpx.AsyncClient) -> dict:
    """One boot-time snapshot of OpenRouter's model catalog.

    Returns {"fetched": bool, "models": [...], "prices": {...}} where
    models entries carry id, name, context_length, prompt_price and
    completion_price (missing fields degrade to None, never breaking
    the whole catalog) and prices is the {model_id: {prompt,
    completion}} map cost_usd consumes. One upstream call feeds both.
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
        }
        # Prices arrive as strings in USD per token. Malformed pricing
        # degrades this entry's price fields rather than dropping the
        # entry or the whole map.
        try:
            model["prompt_price"] = float(entry["pricing"]["prompt"])
            model["completion_price"] = float(entry["pricing"]["completion"])
            prices[entry["id"]] = {
                "prompt": model["prompt_price"],
                "completion": model["completion_price"],
            }
        except (KeyError, TypeError, ValueError):
            model["prompt_price"] = None
            model["completion_price"] = None
        models.append(model)
    return {"fetched": True, "models": models, "prices": prices}


async def fetch_prices(client: httpx.AsyncClient) -> dict:
    """Price map only. Kept for its established contract; the lifespan
    uses fetch_catalog so pricing and the picker share one upstream
    call per boot."""
    return (await fetch_catalog(client))["prices"]


def _flatten_content(content) -> str | None:
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


async def run_model(prompt: str, model: str, client: httpx.AsyncClient) -> dict:
    """Send one chat completion to OpenRouter and return a flat result dict.

    Never raises. A comparison run fans out to several models and one
    failure must not sink the others, so every error path collapses into
    the error field of an otherwise well-formed result.
    """
    result = {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "provider": PROVIDER_PREFS,
    }

    start = time.perf_counter()
    try:
        response = await client.post(
            OPENROUTER_URL, json=payload, timeout=COMPLETION_TIMEOUT
        )
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

    text = _flatten_content(content)
    if text:
        result["response_text"] = text
    else:
        # Some providers return 200 with null content on refusals. Surface
        # that as an error so every result carries either text or an error,
        # a contract the frontend relies on to pick a render state. Non-str
        # oddities land here too rather than leaking into response_text.
        finish_reason = choice.get("finish_reason") or "unknown"
        result["error"] = f"empty response (finish_reason: {finish_reason})"

    # Some providers omit usage. Report None rather than guessing counts.
    # isinstance instead of `or {}`: a truthy non-dict like "n/a" would
    # pass the truthiness guard and raise on .get, and this function must
    # never raise.
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["completion_tokens"] = usage.get("completion_tokens")
    return result


async def stream_model(prompt: str, model: str, client: httpx.AsyncClient):
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
    result = {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
        "ttft_ms": None,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "provider": PROVIDER_PREFS,
        "stream": True,
        # Without this the final chunk carries no usage block and token
        # counts (and therefore cost) would be lost on streamed runs.
        "stream_options": {"include_usage": True},
    }
    text_parts: list[str] = []
    finish_reason = None
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    def done(error: str | None) -> dict:
        result["latency_ms"] = elapsed_ms()
        if text_parts:
            result["response_text"] = "".join(text_parts)
        result["error"] = error
        # Same guard as run_model: a clean stream that produced no text
        # must still carry an error so the frontend has a render state.
        if result["response_text"] is None and error is None:
            result["error"] = (
                f"empty response (finish_reason: {finish_reason or 'unknown'})"
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

                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    result["prompt_tokens"] = usage.get("prompt_tokens")
                    result["completion_tokens"] = usage.get("completion_tokens")

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
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
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
    except httpx.TimeoutException:
        yield done(f"stream stalled: no data for {STREAM_READ_TIMEOUT_S:.0f}s")
        return
    except httpx.HTTPError as exc:
        yield done(f"request failed: {type(exc).__name__}")
        return

    yield done(None)

