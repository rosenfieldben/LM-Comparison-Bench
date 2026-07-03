"""Core model-calling logic. Pure functions over an injected httpx client."""

import json
import time

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Per-request cap. Slow models are a data point in a comparison bench,
# but past 30s the run is more useful failed than pending.
REQUEST_TIMEOUT_S = 30.0

# Boot must not hang on pricing; the bench works offline, cost display
# is the only thing a failed fetch costs.
PRICES_TIMEOUT_S = 10.0


async def fetch_prices(client: httpx.AsyncClient) -> dict:
    """Fetch per-token USD prices for all OpenRouter models.

    Returns {model_id: {"prompt": float, "completion": float}}, or an
    empty dict on any failure. Never raises: the caller runs this at
    startup and a pricing outage must not stop the bench from booting.
    """
    try:
        response = await client.get(MODELS_URL, timeout=PRICES_TIMEOUT_S)
        if response.status_code != 200:
            return {}
        entries = response.json()["data"]
    except (httpx.HTTPError, ValueError, LookupError, TypeError):
        return {}

    prices = {}
    for entry in entries:
        # Prices arrive as strings in USD per token. Skip entries with
        # missing or malformed pricing rather than losing the whole map.
        try:
            prices[entry["id"]] = {
                "prompt": float(entry["pricing"]["prompt"]),
                "completion": float(entry["pricing"]["completion"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def _flatten_content(content) -> str | None:
    """Collapse a message content value to plain text or None.

    Content-parts lists (multimodal providers) flatten to their text
    parts. Anything that is not a non-empty str after that returns None:
    response_text is str or None by contract, and a raw list would crash
    the sqlite bind in save_run and roll back the whole run.
    """
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
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
    }

    start = time.perf_counter()
    try:
        response = await client.post(
            OPENROUTER_URL, json=payload, timeout=REQUEST_TIMEOUT_S
        )
    except httpx.TimeoutException:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"timed out after {REQUEST_TIMEOUT_S:.0f}s"
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

    The 30s timeout is a read timeout between chunks, not total stream
    duration: a slow model legitimately streams for minutes, but a
    silent 30s gap means something is wrong.
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
            "POST", OPENROUTER_URL, json=payload, timeout=REQUEST_TIMEOUT_S
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
        yield done(f"stream stalled for {REQUEST_TIMEOUT_S:.0f}s")
        return
    except httpx.HTTPError as exc:
        yield done(f"request failed: {type(exc).__name__}")
        return

    yield done(None)
