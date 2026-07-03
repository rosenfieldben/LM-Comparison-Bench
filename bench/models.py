"""Core model-calling logic. Pure functions over an injected httpx client."""

import time

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-request cap. Slow models are a data point in a comparison bench,
# but past 30s the run is more useful failed than pending.
REQUEST_TIMEOUT_S = 30.0


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
        result["response_text"] = data["choices"][0]["message"]["content"]
    except (ValueError, LookupError, TypeError):
        result["error"] = "malformed response from OpenRouter"
        return result

    # Some providers omit usage. Report None rather than guessing counts.
    usage = data.get("usage") or {}
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["completion_tokens"] = usage.get("completion_tokens")
    return result
