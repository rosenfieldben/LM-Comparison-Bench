"""Core model-calling logic. Pure functions over an injected httpx client."""

import json
import logging
import math
import os
import socket
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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


def as_money(value: object) -> float | None:
    """A usable money amount: finite, non-negative, else None.

    The F.1 money rule as a function, so every place that ingests a
    currency figure applies it identically. Stricter than as_metric by the
    sign: a negative charge is not a measurement that happens to be below
    zero, it is a value no billing path should produce, and treating it as
    real would let it subtract from an accumulated total and quietly
    disable the spend ceiling. A NaN would do the same by making every
    comparison false, which is why finiteness is checked here rather than
    trusted from the payload.
    """
    amount = as_metric(value)
    if amount is None or amount < 0:
        return None
    return amount


# Env-overridable as a test seam so the browser harness can point the
# real app at a stub upstream; not a configuration feature.
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
MODELS_URL = os.environ.get("MODELS_URL", "https://openrouter.ai/api/v1/models")
GENERATION_URL = os.environ.get(
    "GENERATION_URL", "https://openrouter.ai/api/v1/generation"
)

# Reaching OpenRouter should be fast; a slow connect is a real failure.
CONNECT_TIMEOUT_S = 10.0

# Extended-budget reasoning can legitimately sit silent for minutes
# between visible bytes; five minutes of true wire silence is failure
# by any standard.
STREAM_READ_TIMEOUT_S = 300.0

# Non-streaming path. The single read covers the whole completion,
# including all reasoning time, so it deserves at least the stream's
# per-gap tolerance. At 180 it was lower than the stream's 300, so an
# extended non-streaming run could fail at 181s where the streamed route
# survived; raised to match. Extended non-streaming runs stay more
# timeout-prone by nature: the stream resets its clock on every chunk and
# can run far longer overall, while this is one uninterrupted read. The
# browser always streams; /compare is the scripting path.
COMPLETION_READ_TIMEOUT_S = 300.0

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

# Data-handling routing, merged into the provider block above per policy.
# Field names pinned against OpenRouter's provider-routing documentation
# at https://openrouter.ai/docs/features/provider-routing, read 2026-07-25,
# not from memory: data_collection takes "allow" (the documented default,
# meaning providers that store user data non-transiently and may train on
# it are eligible) or "deny" (route only to providers that do not collect
# user data), and zdr is a boolean that restricts routing to zero data
# retention endpoints. Both are keys of the same top-level provider object
# the throughput sort already uses.
#
# zdr sends data_collection deny as well, which the documentation does not
# require. The two flags govern different things (retention versus
# collection and training eligibility), and a mode chosen for confidential
# prompts that still permitted training-eligible routing would be a trap.
# The stricter reading is the safe one to be wrong about.
DATA_POLICY_PREFS: dict[str, dict[str, Any]] = {
    # Today's payload, unchanged, so the default costs nothing and says
    # nothing it cannot back up.
    "standard": {},
    "deny": {"data_collection": "deny"},
    "zdr": {"zdr": True, "data_collection": "deny"},
}


def provider_preferences(data_policy: str) -> dict[str, Any]:
    """The provider block for one data policy, throughput sort included.

    Raises KeyError on an unknown policy, deliberately: the value is
    validated once at boot (bench.main._parse_data_policy), and a typo
    would otherwise silently send the standard payload while the run row
    claimed something stricter.
    """
    return {**PROVIDER_PREFS, **DATA_POLICY_PREFS[data_policy]}


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


def _request_record(payload: dict[str, Any]) -> str | None:
    """The payload as a stable JSON string, or None if it will not serialize.

    sort_keys so two identical requests produce identical records and a
    diff of two runs shows real differences rather than key order. Returns
    None rather than raising on an unserializable payload: this is a
    provenance nicety, and the never-raises contract outranks it.
    """
    try:
        return json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError):
        return None


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


def _ingest_usage(result: dict[str, Any], usage: object) -> None:
    """Fold one usage object into a result, every field total.

    OpenRouter attaches usage to every response, in the final SSE chunk on
    a stream ("Full usage details are now always included automatically in
    every response", openrouter.ai/docs/cookbook/administration/
    usage-accounting, read 2026-07-25), so this is the one place the
    authoritative numbers enter. Always is the vendor's word, not an
    assumption this code makes: every field below still degrades when a
    provider disagrees. It is called with whatever the payload held:
    isinstance instead of a
    truthiness guard because a non-dict like "n/a" would pass truthiness
    and raise on .get, and neither client function may raise.

    Every field is assigned unconditionally, including to None, rather
    than only when present. A stream can carry more than one usage object,
    and a later one that omits a field must not leave the earlier value
    standing as if it had been reconfirmed. Last usage object wins,
    entirely.

    cost is the amount OpenRouter charged, and it goes through as_money
    rather than as_metric: a NaN or negative charge must degrade to None
    instead of reaching the spend accumulator, where NaN makes every
    ceiling comparison false and a negative subtracts from the total.

    cost_details.upstream_inference_cost is deliberately not persisted. It
    is the amount the upstream provider charged under bring-your-own-key,
    and is absent or zero for the normal credits path this bench uses, so
    a column for it would be null on every row this bench produces.
    """
    if not isinstance(usage, dict):
        return
    result["prompt_tokens"] = as_token_count(usage.get("prompt_tokens"))
    result["completion_tokens"] = as_token_count(usage.get("completion_tokens"))
    result["billed_cost_usd"] = as_money(usage.get("cost"))
    # Reasoning tokens are the reason the two-tier budget exists: they are
    # billed as completion tokens and consume max_tokens, but never appear
    # in the visible answer. Cached prompt tokens are the other direction,
    # charged at a discount, and both live one level down in the details
    # objects.
    completion_details = usage.get("completion_tokens_details")
    result["reasoning_tokens"] = (
        as_token_count(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else None
    )
    prompt_details = usage.get("prompt_tokens_details")
    result["cached_tokens"] = (
        as_token_count(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, dict)
        else None
    )


# OpenRouter returns the generation id in this response header on every
# endpoint, available the moment the response starts and therefore before
# any chunk has been read. Pinned against the vendor's own words rather
# than memory, like DATA_POLICY_PREFS above: "fetch the generation record
# via GET /api/v1/generation using the X-Generation-Id response header"
# (openrouter.ai/docs/guides/features/router-metadata, read 2026-07-25).
#
# That timing is the point: a run the user stops
# mid-stream is exactly the run whose billing is unknowable locally
# (cancellation only stops billing on providers that support it), so it is
# the run that most needs an id to reconcile against later, and reading the
# id out of a chunk would lose it for precisely those runs.
GENERATION_ID_HEADER = "X-Generation-Id"


def _settle_generation_id(
    result: dict[str, Any],
    value: object,
    holder: dict[str, Any] | None = None,
) -> None:
    """Record the generation id the first time one is seen.

    First writer wins, so the header settles the field and a chunk id only
    fills in when no header arrived. A later value that disagrees is logged
    rather than raised or applied: the never-raises contract holds, and
    overwriting would mean the persisted id depends on which source spoke
    last. Junk and empty strings are absence, per _as_label.

    holder is dependency-injected state, not a protocol change. The
    streaming endpoint owns a plain dict and passes it in so its disconnect
    path can read the id without waiting for a done event that a stopped
    run never produces. Only the id goes in, and only by assignment, so
    reading it back during generator unwinding cannot await or raise.
    """
    gen_id = _as_label(value)
    if gen_id is None:
        return
    settled = result["generation_id"]
    if settled is None:
        result["generation_id"] = gen_id
        if holder is not None:
            holder["generation_id"] = gen_id
    elif settled != gen_id:
        logger.warning(
            "generation id disagreement: header/first chunk said %s, later "
            "source said %s; keeping the first",
            settled,
            gen_id,
        )


def _as_label(value: object) -> str | None:
    """as_text with the empty string treated as absent.

    A provider name or a finish reason of "" is not a value the frontend
    can render as a caption; it would paint an empty slot that looks like
    a layout bug. Everything else is as_text's contract.
    """
    text = as_text(value)
    return text if text else None


async def run_model(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    max_tokens: int = BUDGET_STANDARD,
    provider_prefs: dict[str, Any] | None = None,
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
        "request_json": None,
        "billed_cost_usd": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "provider": None,
        "native_finish_reason": None,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Injected rather than read from module state, so the boot-scoped
        # data policy reaches the wire without this function knowing where
        # policy lives. The default keeps a direct caller on today's
        # behavior.
        "provider": PROVIDER_PREFS if provider_prefs is None else provider_prefs,
    }
    # The reproducibility record: what was actually sent, not what was
    # intended. Serialized here, beside the dict that goes on the wire, so
    # the two cannot drift. Authorization lives in the client's headers and
    # never enters the payload, so nothing secret is recorded. Echoed even
    # on failure paths, because knowing what a failed request asked for is
    # exactly when this matters.
    result["request_json"] = _request_record(payload)

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

    # Before the status check and before the body is parsed: the header is
    # available at response start, and an id captured here survives a
    # response whose body turns out to be unparseable, which is one of the
    # cases worth reconciling later.
    _settle_generation_id(result, response.headers.get(GENERATION_ID_HEADER))

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
    # after the fact. The body id is the fallback here, not the primary:
    # the header above already settled the field on any response that
    # carried one. The finish_reason says why output ended; until now
    # it survived only inside synthesized error strings, and budget
    # analysis needs it on successful runs too. It is OpenRouter's
    # NORMALIZED value, not the provider's own word: OpenRouter maps each
    # provider's vocabulary onto a common set and exposes the raw one
    # separately as native_finish_reason, which is captured below.
    _settle_generation_id(result, data.get("id"))
    reason = choice.get("finish_reason")
    if isinstance(reason, str) and reason:
        result["finish_reason"] = reason
    # The provider that actually served this request, and its own word for
    # why generation ended. Under throughput routing the provider is
    # chosen per request, so it is the largest confound in any comparison
    # this bench draws; recording it per result is what makes the confound
    # visible instead of assumed away. Both degrade to None when the
    # payload omits them, which is the common case for native_finish_reason
    # (OpenRouter only sends it when the provider's value differs from its
    # own normalized one).
    result["provider"] = _as_label(data.get("provider"))
    result["native_finish_reason"] = _as_label(choice.get("native_finish_reason"))

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

    # A missing or oddly shaped usage object leaves the counts and the
    # billed cost None rather than guessed. The guard that makes that safe
    # (isinstance, not truthiness, so a value like "n/a" cannot raise on
    # .get) lives in _ingest_usage now, which is where its comment lives
    # too.
    _ingest_usage(result, data.get("usage"))
    return result


async def stream_model(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    max_tokens: int = BUDGET_STANDARD,
    holder: dict[str, Any] | None = None,
    provider_prefs: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one chat completion, yielding delta and done event dicts.

    holder, when given, receives facts about the exchange as soon as each
    becomes known: the request record before the request is sent, then the
    generation id and the serving provider. A consumer that never reaches
    the done event (a client disconnect cancels this generator at a yield)
    has no other way to learn any of them, and a run cut short is exactly
    the run whose billing has to be reconciled later, on a row that should
    still be able to say what it asked for. Writes only, by plain
    assignment, so a reader during generator unwinding cannot await or
    raise. See _settle_generation_id.

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
        "request_json": None,
        "billed_cost_usd": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "provider": None,
        "native_finish_reason": None,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # See run_model: injected so the boot-scoped data policy reaches
        # the wire without this function knowing where policy lives.
        "provider": PROVIDER_PREFS if provider_prefs is None else provider_prefs,
        "stream": True,
        # A deprecated no-op, kept only because removing it would change
        # the recorded request_json of every streamed run for no gain.
        # OpenRouter's usage-accounting page states that this flag and
        # usage.include "are deprecated and have no effect. Full usage
        # details are now always included automatically in every response"
        # (read 2026-07-25). The usage block arrives because the platform
        # sends it, not because this asks.
        "stream_options": {"include_usage": True},
    }
    # See run_model: the exact payload, auth excluded, recorded beside the
    # dict that goes on the wire. Mirrored into the holder in the same
    # breath, because this is knowable before the request is even sent and
    # a run cut short mid-stream is exactly the row that should still be
    # able to say what it asked for.
    result["request_json"] = _request_record(payload)
    if holder is not None:
        holder["request_json"] = result["request_json"]
    text_parts: list[str] = []
    saw_done = False
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
            # Before the status check and before the first chunk is read.
            # This is the whole reason the header is preferred: a run the
            # user stops seconds later never reaches a chunk that carries
            # an id, and it is the run that most needs one.
            _settle_generation_id(
                result, response.headers.get(GENERATION_ID_HEADER), holder
            )
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
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise ValueError("chunk is not an object")
                except ValueError:
                    yield done("malformed stream from OpenRouter")
                    return

                # The chunk id is the fallback for a response that carried
                # no header. Every chunk repeats the same id, so the first
                # one that carries it settles the field, and a disagreement
                # with the header is logged rather than applied.
                _settle_generation_id(result, chunk.get("id"), holder)

                # The provider name rides every chunk, not just the last;
                # the first one that carries it settles the field, like the
                # generation id above, and reaches the holder for the same
                # reason: an aborted row that knows which host served it is
                # a row whose routing confound is still visible.
                if result["provider"] is None:
                    result["provider"] = _as_label(chunk.get("provider"))
                    if holder is not None and result["provider"] is not None:
                        holder["provider"] = result["provider"]

                # Usage arrives on the final chunk (stream_options above
                # asks for it) and carries the billed cost with it.
                _ingest_usage(result, chunk.get("usage"))

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
                    native = _as_label(choice.get("native_finish_reason"))
                    if native is not None:
                        # Guarded like finish_reason above: OpenRouter sends
                        # the provider's own word only on the chunk that ends
                        # generation, so a later chunk carrying a choice
                        # without it must not erase what was already seen.
                        result["native_finish_reason"] = native
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

    # A clean iterator exhaustion without the [DONE] marker is not a
    # success: a load balancer idle-closing the connection or a provider
    # crashing with a clean close both land here, and done(None) would show
    # and persist a truncated answer as complete, corrupting the
    # comparison. But a stream that delivered a finish_reason and then
    # closed without [DONE] is semantically complete (the provider stated
    # why it stopped), so flagging that would be a false alarm. done()
    # already carries the partial text accumulated so far.
    if not saw_done and result["finish_reason"] is None:
        yield done("stream ended before completion: no [DONE] and no finish reason")
        return
    yield done(None)


# The generation endpoint is an audit surface, not a costing path: it is
# called after the fact by the reconcile script, never during a run.
# Fifteen seconds is generous for one metadata lookup and short enough
# that a hung endpoint does not stall a long backfill.
GENERATION_TIMEOUT_S = 15.0


async def fetch_generation(
    client: httpx.AsyncClient, generation_id: str
) -> dict[str, Any]:
    """One generation's after-the-fact record, or an error field saying why.

    Never raises, like the two client functions: this runs in a loop over
    many rows, and one bad id must not end the pass. Every field goes
    through the same total field-type functions, with the money rule on the
    charge, so a poisoned value from the audit path cannot do what it could
    not do from the streaming path.

    Field names are the ones OpenRouter's OpenAPI description of
    GET /generation gives (read 2026-07-25 from
    https://openrouter.ai/docs/api/api-reference/generations/get-request-%26-usage-metadata-for-a-generation):
    the charge is total_cost, the host is provider_name, and the token
    counts prefixed native_ are the model's own tokenizer's, which is what
    the in-band usage object reports too, so the two are comparable.

    quantization is read but is NOT in that description's response schema
    today. It is fetched anyway rather than omitted, because the field is
    what the reconcile pass exists to recover and reading a key that may
    appear costs nothing; until it does appear, this degrades to None like
    any absent field. Do not read a None here as "the provider served
    unquantized weights".
    """
    record: dict[str, Any] = {
        "generation_id": generation_id,
        "billed_cost_usd": None,
        "provider": None,
        "quantization": None,
        "native_finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "error": None,
    }
    try:
        response = await client.get(
            GENERATION_URL,
            params={"id": generation_id},
            timeout=GENERATION_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        record["error"] = f"request failed: {type(exc).__name__}"
        return record
    if response.status_code != 200:
        # 404 is ordinary here: OpenRouter expires generation records, and
        # a run old enough to have aged out is not a failure of this pass.
        record["error"] = f"HTTP {response.status_code} from OpenRouter"
        return record
    try:
        data = response.json()["data"]
        if not isinstance(data, dict):
            raise ValueError("data is not an object")
    except (ValueError, LookupError, TypeError):
        record["error"] = "malformed response from OpenRouter"
        return record

    record["billed_cost_usd"] = as_money(data.get("total_cost"))
    record["provider"] = _as_label(data.get("provider_name"))
    record["quantization"] = _as_label(data.get("quantization"))
    record["native_finish_reason"] = _as_label(data.get("native_finish_reason"))
    record["prompt_tokens"] = as_token_count(data.get("native_tokens_prompt"))
    record["completion_tokens"] = as_token_count(data.get("native_tokens_completion"))
    record["reasoning_tokens"] = as_token_count(data.get("native_tokens_reasoning"))
    record["cached_tokens"] = as_token_count(data.get("native_tokens_cached"))
    return record
