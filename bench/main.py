"""FastAPI boundary. Pydantic models live here only; internals use plain dicts."""

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bench import store
from bench.models import (
    BUDGET_EXTENDED,
    BUDGET_STANDARD,
    fetch_catalog,
    keepalive_socket_options,
    run_model,
    stream_model,
)

logger = logging.getLogger(__name__)

# Budget names resolve to numbers here at the boundary; internals pass
# plain integers. Pydantic's Literal on the request models is what
# rejects unknown names, so this map never sees one.
BUDGET_TOKENS = {"standard": BUDGET_STANDARD, "extended": BUDGET_EXTENDED}

# Ceiling on simultaneous paid upstream calls across everything in
# flight. The batch endpoint's five-model cap always implied this
# policy, but the UI's real path is one /compare/stream per model with
# no batch to cap, so overlapping runs and reruns could put unbounded
# paid calls in flight. The semaphore enforces the cap where the money
# actually moves: around the upstream HTTP exchange, in both endpoints.
# Saturation queues quietly; a sixth model simply starts when a slot
# frees, with no error and no acquisition timeout.
MAX_CONCURRENT_UPSTREAM = 5

# SQLite stores rowids in a signed 64-bit integer. Pydantic accepted
# any Python int, SQLite did not, and the gap surfaced as an
# OverflowError 500 after the paid upstream call had already
# completed. Bounding at the boundary makes the mismatch a 422 before
# any money moves.
MAX_SQLITE_ROWID = 2**63 - 1


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # Cap at 5: the bench is for side-by-side eyeballing, and each extra
    # model is a concurrent upstream request on one API key.
    models: list[str] = Field(min_length=1, max_length=5)
    # Optional link back to a saved prompt so history can show where a
    # run came from. The run stores its own prompt_text either way.
    # Bounded to the rowid range; see MAX_SQLITE_ROWID.
    prompt_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    # Optional grouping id so the N per-model requests of one comparison
    # land as one history entry.
    group_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    # Named tiers, not a free integer: two regimes of use exist and a
    # free field invites typos with dollar consequences.
    budget: Literal["standard", "extended"] = "standard"


class ModelResult(BaseModel):
    model: str
    response_text: str | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None
    cost_usd: float | None
    # Time to first content delta. Only the streaming path measures it;
    # the default keeps /compare results (which never carry the key) valid.
    ttft_ms: float | None = None
    # The effective (post-clamp) completion budget the request was sent
    # with. Defaults None so pre-budget history rows stay valid.
    max_tokens: int | None = None
    # OpenRouter's response id (gen-...). It keys OpenRouter's
    # generation API, which records the actual provider, quantization
    # and authoritative cost, so a persisted id makes any historical
    # run auditable. Defaults None so pre-provenance rows stay valid.
    generation_id: str | None = None
    # The provider's finish_reason verbatim (stop, length,
    # content_filter, ...). Until now it survived only inside
    # synthesized error strings; budget analysis needs it on
    # truncated-but-successful runs too.
    finish_reason: str | None = None


class StreamCompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # One model per streaming request, mirroring the frontend's
    # per-model fetch pattern: independent columns are the product.
    model: str = Field(min_length=1)
    # Bounded like CompareRequest's; see MAX_SQLITE_ROWID.
    prompt_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    group_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    budget: Literal["standard", "extended"] = "standard"


class CompareResponse(BaseModel):
    results: list[ModelResult]
    # None when persisting the run failed: the upstream spend already
    # happened, so the results are returned even when history is lost.
    run_id: int | None


class PromptCreate(BaseModel):
    name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Prompt(BaseModel):
    id: int
    name: str
    text: str
    created_at: str


class PromptList(BaseModel):
    prompts: list[Prompt]


class RunEntry(BaseModel):
    type: Literal["run"]
    id: int
    created_at: str
    prompt_text: str
    models: list[str]


class GroupEntry(BaseModel):
    type: Literal["group"]
    id: int
    created_at: str
    prompt_text: str
    models: list[str]
    run_ids: list[int]


class RunList(BaseModel):
    runs: list[GroupEntry | RunEntry]


class GroupCreated(BaseModel):
    id: int


class CatalogModel(BaseModel):
    id: str
    name: str | None
    context_length: int | None
    prompt_price: float | None
    completion_price: float | None
    # The published completion cap the budget clamp works from. Exposed
    # so the picker can show why "extended" quietly becomes less on
    # capped models; None when the catalog does not publish one.
    max_completion_tokens: int | None


class CatalogResponse(BaseModel):
    models: list[CatalogModel]
    # Lets the frontend tell an offline boot from an empty catalog and
    # switch to the exact-id fallback instead of a dead search box.
    fetched: bool


class RunDetail(BaseModel):
    id: int
    created_at: str
    prompt_text: str
    prompt_id: int | None
    results: list[ModelResult]


class GroupDetail(BaseModel):
    id: int
    created_at: str
    runs: list[RunDetail]


def _parse_spend_limit(raw: str | None) -> float | None:
    """The validated per-boot spend ceiling, or None for no limit.

    Unset or empty means no limit. Anything present must parse to a
    strictly positive finite float. A non-finite or negative value is
    nonsense, and zero is ambiguous between a deliberate lockout and a
    misconfiguration; the two are indistinguishable, so both are refused
    loudly at boot rather than silently producing a broken or surprising
    ceiling. A bare float() would have accepted nan and inf (a nan ceiling
    is never crossed, silently disabling the limit) and negatives.
    """
    if not raw:
        return None
    try:
        limit = float(raw)
    except ValueError:
        raise RuntimeError(
            f"BENCH_SPEND_LIMIT_USD must be a number, got {raw!r}. "
            "Unset it for no limit."
        ) from None
    if not math.isfinite(limit) or limit <= 0:
        raise RuntimeError(
            f"BENCH_SPEND_LIMIT_USD must be a positive number, got {raw!r}. "
            "Unset it for no limit."
        )
    return limit


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail at boot, not on the first request. A bench with a missing key
    # would otherwise report every model as errored and look like an outage.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before starting the app."
        )
    # Per-boot spend ceiling, validated here next to the key check and
    # before any resource is allocated so a misconfigured limit fails fast
    # and loud like a missing key. Unset means no limit. The figure tracked
    # is estimated spend (catalog prices times reported tokens), the same
    # numbers the cards show, accumulated as a plain float: one event loop
    # makes that safe without a lock. Results the session cannot price
    # (offline catalog, missing usage) never move the counter, so the
    # ceiling bounds known estimated spend, not a billed total.
    app.state.spend_limit_usd = _parse_spend_limit(
        os.environ.get("BENCH_SPEND_LIMIT_USD")
    )
    app.state.accumulated_spend_usd = 0.0
    # One shared client: connection pooling across the fan-out, and the
    # auth header lives in exactly one place. The explicit transport
    # exists to carry TCP keepalive options: extended-budget streams go
    # silent for minutes during hidden reasoning, NAT idle timers cull
    # flows that move no bytes, and the resulting deaths wore mixed
    # ReadError and stall signatures. OS-level probes keep those quiet
    # flows alive; see keepalive_socket_options in models.py.
    # trust_env stays at its default (on): an operator may legitimately
    # reach OpenRouter through a corporate proxy, and honoring HTTP(S)_PROXY
    # is the right behavior for the real app. The test harness is the one
    # place that must not inherit a developer proxy, and it opts out itself
    # (trust_env=False on its own clients, proxy vars scrubbed from the
    # browser subprocess) rather than the app degrading its own behavior.
    app.state.client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.AsyncHTTPTransport(socket_options=keepalive_socket_options()),
    )
    app.state.db = store.connect(os.environ.get("BENCH_DB", "./bench.db"))
    # One shared gate for every paid upstream call this process makes;
    # see MAX_CONCURRENT_UPSTREAM for why it exists.
    app.state.upstream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPSTREAM)
    # One catalog snapshot per boot feeds both pricing and the model
    # picker. Failure is tolerated: the bench must work offline, cost
    # renders as unavailable and the picker falls back to exact ids.
    app.state.catalog = await fetch_catalog(app.state.client)
    app.state.prices = app.state.catalog["prices"]
    # Per-model completion caps, derived once per boot so the budget
    # clamp is a dict lookup per request. Only published integer caps
    # participate; an unknown cap means no clamp, same as offline.
    app.state.completion_limits = {
        m["id"]: m["max_completion_tokens"]
        for m in app.state.catalog["models"]
        if isinstance(m.get("max_completion_tokens"), int)
    }
    if not app.state.catalog["fetched"]:
        logger.warning(
            "OpenRouter catalog fetch failed; cost display and model "
            "search are unavailable this session"
        )
    yield
    await app.state.client.aclose()
    app.state.db.close()


app = FastAPI(title="LM Comparison Bench", lifespan=lifespan)

# The bench is a localhost tool holding a paid API key, which makes it a
# target for the two ways a browser gets turned against local servers.
# First, cross-site "simple" POSTs: a malicious page can fire fetch() at
# http://localhost:8000 with a text/plain body and no CORS preflight,
# and although it cannot read the response, each /compare call it lands
# spends real money upstream. Second, DNS rebinding: an attacker's
# hostname re-resolving to 127.0.0.1 makes the page same-origin with
# the bench, granting full read access. Requiring JSON bodies on POST
# forces cross-origin senders into a preflight nothing here answers,
# and rejecting non-local Host headers kills rebinding (the rebound
# page's requests carry the attacker's hostname in Host).
TRUSTED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def host_header_name(host: str) -> str:
    """The hostname part of a Host header value, port stripped.

    Handles bracketed IPv6 ([::1]:8000) explicitly: a naive rsplit on
    ":" would chop the address itself. A bare IPv6 with no brackets has
    multiple colons and falls through unchanged.
    """
    host = host.strip().lower()
    if host.startswith("[") and "]" in host:
        return host[1 : host.index("]")]
    if host.count(":") == 1:
        return host.rsplit(":", 1)[0]
    return host


class LocalOnlyGuard:
    """Reject requests that could only come from a hostile browser page.

    Pure ASGI rather than BaseHTTPMiddleware: the streaming endpoint
    relies on generator cancellation to persist aborted runs, and a
    wrapping middleware layer is one more thing that could interfere
    with disconnect propagation.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Deny framing on every HTTP response. A hostile page cannot read a
        # cross-origin response, but it can frame the real localhost UI and
        # redress a Run click into paid work; these headers refuse the
        # frame. Modern Chrome's private-network rules blunt this, other
        # browsers differ, and the headers cost nothing. Blanket
        # application is deliberate: no response here is meant to be
        # embedded. The headers are added on http.response.start only, with
        # no buffering of the body, so SSE streaming and the generator
        # cancellation the streaming endpoint depends on stay untouched,
        # which is the reason this guard is pure ASGI. A fuller CSP is out
        # of scope: the pre-paint inline theme script in index.html would
        # need a hash or externalization, so this stops at frame-ancestors.
        async def framed_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-frame-options"] = "DENY"
                headers["content-security-policy"] = "frame-ancestors 'none'"
            await send(message)

        headers = Headers(scope=scope)
        # A missing Host header (raw HTTP/1.0 clients) fails closed.
        host = host_header_name(headers.get("host", ""))
        if host not in TRUSTED_HOSTS:
            response = JSONResponse(
                {"detail": "the bench only answers to localhost"},
                status_code=403,
            )
            await response(scope, receive, framed_send)
            return
        # Every POST must be application/json, bodyless included: POST
        # is the one method a browser fires cross-site without a CORS
        # preflight, and the old bodyless exemption was a reproduced
        # hole (a no-body cross-site POST /groups returned 201).
        # Requiring the JSON content type forces the preflight nothing
        # here answers. GET and HEAD stay exempt as reads; DELETE needs
        # no gate because it is never a "simple" cross-site method, so
        # the preflight already guards it.
        if scope["method"] == "POST" and not headers.get("content-type", "").startswith(
            "application/json"
        ):
            response = JSONResponse(
                {"detail": "POST bodies must be application/json"},
                status_code=415,
            )
            await response(scope, receive, framed_send)
            return
        await self.app(scope, receive, framed_send)


app.add_middleware(LocalOnlyGuard)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Serve the static assets (vendored fonts now; the split-out stylesheet
# and scripts later) from one mount. LocalOnlyGuard wraps the whole app
# and GET is exempt from the JSON-POST rule, so the security posture is
# unchanged: these are read-only assets on a localhost-only server.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Hand-written bolt in the VOLT accent, sized to stay legible at 16px.
# Served inline because browsers request /favicon.ico unprompted and
# every miss was a 404 line of noise in the run logs.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<path d="M9.5 1 3 9.5h4L5.5 15 13 6.5H9L9.5 1z" fill="#38bdd8"/>'
    "</svg>"
)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    # Immutable is honest: the icon only changes when the app does.
    return Response(
        FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def cost_usd(result: dict[str, Any], prices: dict[str, Any]) -> float | None:
    """Cost of one result, or None when tokens or pricing are unknown.

    isinstance rather than a None check: a provider reporting token
    counts as strings would otherwise raise in the multiplication and
    sink the whole request after the upstream calls already happened.
    """
    price = prices.get(result["model"])
    if (
        price is None
        or not isinstance(result["prompt_tokens"], (int, float))
        or not isinstance(result["completion_tokens"], (int, float))
    ):
        return None
    total = (
        result["prompt_tokens"] * price["prompt"]
        + result["completion_tokens"] * price["completion"]
    )
    # The catalog is the primary guard: fetch_catalog rejects non-finite
    # prices at ingestion, so a NaN can only reach here if a price was set
    # past that path. This is the belt: an unpriceable total degrades to
    # None, never a NaN that would poison accumulated spend and silently
    # disable the ceiling.
    if not math.isfinite(total):
        return None
    return float(total)


def format_usd(value: float) -> str:
    """A dollar figure that stays truthful at the ceiling's native scale.

    Priced runs here cost fractions of a cent, so the cards' four decimals
    collapse a real sub-cent limit to $0.0000 and the 402 would misreport
    the operator's own ceiling as zero. Six decimals cover that domain;
    trailing zeros past the cents place are trimmed so dollar-scale limits
    still read as $1.50, not $1.500000.
    """
    whole, frac = f"{value:.6f}".split(".")
    frac = frac[:2] + frac[2:].rstrip("0")
    return f"${whole}.{frac}"


def spend_ceiling_reached() -> bool:
    """True when an active ceiling has been reached by accumulated spend.

    The shared predicate behind the entry check and the post-admission
    recheck. Unpriced results never moved the counter, so this bounds
    known estimated spend, not billed cost.
    """
    limit = app.state.spend_limit_usd
    if limit is None:
        return False
    return bool(app.state.accumulated_spend_usd >= limit)


def enforce_spend_limit() -> None:
    """Refuse a run at the boundary once estimated spend hits the ceiling.

    Checked at endpoint entry, before the semaphore and before any
    upstream call, so a refusal costs nothing. Money already in flight
    is never interrupted. The 402 names both figures so the operator
    knows how far over the intent they are. A second recheck runs after
    admission (spend_refusal_result) to close the gap where runs admitted
    below the limit would all execute once an earlier one crossed it.
    """
    if spend_ceiling_reached():
        raise HTTPException(
            402,
            "spend ceiling reached: estimated "
            f"{format_usd(app.state.accumulated_spend_usd)} of "
            f"{format_usd(app.state.spend_limit_usd)} limit "
            "(BENCH_SPEND_LIMIT_USD); unpriced runs do not count against it",
        )


def record_spend(cost: float | None) -> None:
    """Add a priced result's estimate to the accumulated spend.

    None means the result could not be priced (offline catalog, missing
    usage), and those never count against the ceiling by design. A
    non-finite cost is refused here too: cost_usd already screens it out,
    but the accumulator is the invariant's last line, and a single NaN
    summed in would make the ceiling comparison permanently false.
    """
    if cost is not None and math.isfinite(cost):
        app.state.accumulated_spend_usd += cost


def spend_refusal_result(model: str, max_tokens: int) -> dict[str, Any]:
    """A synthetic result for a run refused at the post-admission recheck.

    Shaped like stream_model's done() result, which is a superset of
    run_model's and which both endpoints' response models accept. Every
    metric and text field is None because no upstream call happened. The
    error reuses format_usd and states plainly that the run was refused
    before reaching upstream, so the persisted row is unambiguous that no
    money moved.
    """
    return {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": (
            "run refused before reaching upstream: estimated spend "
            f"{format_usd(app.state.accumulated_spend_usd)} reached the "
            f"{format_usd(app.state.spend_limit_usd)} ceiling "
            "(BENCH_SPEND_LIMIT_USD); no upstream call was made"
        ),
        "cost_usd": None,
        "ttft_ms": None,
        "max_tokens": max_tokens,
        "generation_id": None,
        "finish_reason": None,
    }


def effective_budget(budget: str, model: str) -> int:
    """The requested budget clamped to the model's published completion
    cap. Sending a budget above a model's cap is a hard 400 from some
    providers; clamping turns that failure into the best the model can
    do. Lives here rather than in models.py because the boot catalog is
    app state.
    """
    requested = BUDGET_TOKENS[budget]
    limit = app.state.completion_limits.get(model)
    return min(requested, limit) if limit is not None else requested


def resolve_links(
    db: sqlite3.Connection, prompt_id: int | None, group_id: int | None
) -> tuple[int | None, int | None]:
    """Degrade stale prompt or group links to None instead of erroring.

    By the time links are checked the upstream calls already happened
    and prompt_text is the source of truth, so a deleted or bogus id
    must not sink the run.
    """
    if prompt_id is not None and store.get_prompt(db, prompt_id) is None:
        prompt_id = None
    if group_id is not None and not store.group_exists(db, group_id):
        group_id = None
    return prompt_id, group_id


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> dict[str, Any]:
    enforce_spend_limit()

    async def limited(model: str) -> dict[str, Any]:
        # One slot per model inside the fan-out, not one around the
        # batch: the cap is on simultaneous paid calls wherever they
        # come from, and concurrent /compare and stream requests share
        # the same gate. Acquiring before run_model keeps the clocks
        # honest for free: its latency clock starts internally, after
        # the slot is already held, so a queued model never reports
        # queue wait as model latency.
        budget = effective_budget(request.budget, model)
        async with app.state.upstream_semaphore:
            # Recheck under the held slot, before run_model spends. This
            # batch records its own spend only after the gather completes,
            # so the recheck cannot observe an earlier model in this same
            # batch; what it catches is spend a concurrent request (another
            # /compare or a stream) recorded while this model waited for a
            # slot. One batch is capped at five models, equal to
            # MAX_CONCURRENT_UPSTREAM, so a batch's own overshoot already
            # sits within the documented bound; the recheck stops a queued
            # batch from spending once a concurrent run crossed the ceiling.
            # On refusal return a synthetic result shaped like run_model's,
            # error set, with no upstream call. The batch persists as usual
            # with the refusal row included: honest history for a cut-short
            # run.
            if spend_ceiling_reached():
                return spend_refusal_result(model, budget)
            return await run_model(
                request.prompt,
                model,
                app.state.client,
                max_tokens=budget,
            )

    # gather preserves input order, which the frontend relies on to map
    # result columns by position. run_model never raises, so no
    # return_exceptions handling is needed here.
    results = await asyncio.gather(*(limited(m) for m in request.models))
    # Seeded before the fault boundary so a failure inside it can never
    # leave a result missing the key the response model requires.
    for result in results:
        result["cost_usd"] = None
    run_id = None
    # The invariant: after money is spent, no code path may convert
    # results into an error response. Everything between "upstream
    # results exist" and "response returned" (cost, link resolution,
    # persistence) sits inside this one boundary; on any failure the
    # results go back intact with run_id null and the links dropped.
    try:
        # Cost is a boundary concern: run_model stays a pure OpenRouter
        # call and the price snapshot lives on app.state. Computed
        # before save_run so history carries the cost as priced at run
        # time.
        for result in results:
            result["cost_usd"] = cost_usd(result, app.state.prices)
            record_spend(result["cost_usd"])
        prompt_id, group_id = resolve_links(
            app.state.db, request.prompt_id, request.group_id
        )
        run_id = store.save_run(
            app.state.db, request.prompt, list(results), prompt_id, group_id
        )
    except Exception:
        logger.exception("post-upstream processing failed for /compare")
        run_id = None
    return {"results": list(results), "run_id": run_id}


@app.post("/compare/stream")
async def compare_stream(request: StreamCompareRequest) -> StreamingResponse:
    # At entry, before the generator runs and so before the semaphore or
    # any upstream call: a refusal must spend nothing.
    enforce_spend_limit()
    max_tokens = effective_budget(request.budget, request.model)

    async def events() -> AsyncIterator[str]:
        # Server-side observation of the stream, enough to reconstruct
        # a result if the client disconnects before the done event: a
        # stream the user watched happen is still history.
        parts: list[str] = []
        first_delta_ms: float | None = None
        handled = False
        started = False
        acquired = False
        start = 0.0

        def release_slot() -> None:
            # Idempotent so the done branch and the finally below can
            # both call it: the slot must never be returned twice, and
            # a cancellation while still queued has nothing to return.
            nonlocal acquired
            if acquired:
                acquired = False
                app.state.upstream_semaphore.release()

        try:
            # A saturated semaphore means this run waits for a slot.
            # Tell the client so its column reads "queued" instead of
            # pretending the model is already thinking; locked() is true
            # exactly when no slot is free. Emitted before any clock so
            # the wait pollutes no metric.
            if app.state.upstream_semaphore.locked():
                yield "data: " + json.dumps({"type": "queued"}) + "\n\n"
            # The slot covers the upstream exchange only, and it is
            # acquired before any clock starts: both this generator's
            # clock and stream_model's internal latency and ttft clocks
            # begin after the slot is held, so a queued run never
            # reports its queue wait as model time. Saturation queues
            # quietly, with no acquisition timeout.
            await app.state.upstream_semaphore.acquire()
            acquired = True
            # Recheck the ceiling now that a slot is held, before started is
            # set, before the clock, and before the started frame. The entry
            # check admits every queued run below the limit; without this
            # recheck an earlier run crossing the ceiling would not stop the
            # ones already admitted, so overshoot would be bounded by lineup
            # size, not by the semaphore. On refusal return the slot, emit
            # one done frame carrying the synthetic refusal result with
            # run_id null, and persist nothing: the refusal happened before
            # any upstream call, exactly like a queued cancel, so there is
            # nothing truthful to record. started stays false so the
            # finally's abort-persist path never fires.
            if spend_ceiling_reached():
                release_slot()
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "result": spend_refusal_result(request.model, max_tokens),
                            "run_id": None,
                        }
                    )
                    + "\n\n"
                )
                return
            started = True
            # Start the clock before emitting started, not after. The
            # frame and the clock mark the same instant (the sub-ms cost
            # of yielding one small frame on localhost is noise against a
            # network TTFT), and starting first keeps start valid for the
            # finally: a disconnect suspended at this yield would otherwise
            # run the abort path with start still 0.0 and persist a garbage
            # latency. The client still resets its own clock on the frame.
            start = time.perf_counter()
            yield "data: " + json.dumps({"type": "started"}) + "\n\n"
            async for event in stream_model(
                request.prompt, request.model, app.state.client, max_tokens=max_tokens
            ):
                if event["type"] != "done":
                    if first_delta_ms is None:
                        first_delta_ms = round((time.perf_counter() - start) * 1000, 1)
                    parts.append(event["text"])
                    yield "data: " + json.dumps(event) + "\n\n"
                    continue
                handled = True
                # The done event means the upstream exchange is over;
                # persistence must not sit on a paid-call slot.
                release_slot()
                result = event["result"]
                result["cost_usd"] = None
                run_id = None
                # Same invariant as /compare, and stricter here: the
                # deltas are already on the wire, so after money is
                # spent no code path may convert the result into a
                # broken stream tail. Cost, link resolution and
                # persistence all sit inside this one boundary; any
                # failure degrades to run_id null with links dropped.
                try:
                    result["cost_usd"] = cost_usd(result, app.state.prices)
                    record_spend(result["cost_usd"])
                    prompt_id, group_id = resolve_links(
                        app.state.db, request.prompt_id, request.group_id
                    )
                    run_id = store.save_run(
                        app.state.db, request.prompt, [result], prompt_id, group_id
                    )
                except Exception:
                    logger.exception(
                        "post-stream processing failed for /compare/stream"
                    )
                    run_id = None
                yield (
                    "data: "
                    + json.dumps({"type": "done", "result": result, "run_id": run_id})
                    + "\n\n"
                )
        finally:
            release_slot()
            # A client disconnect cancels this generator at a yield
            # before the done branch ever runs. Persist what the server
            # saw (no awaits or yields are legal here, sqlite is sync,
            # so this is safe during unwinding) rather than silently
            # dropping a run whose deltas already reached the browser.
            # A run cancelled while still queued never reached upstream
            # and spent nothing, so there is nothing truthful to record.
            if started and not handled:
                aborted = {
                    "model": request.model,
                    "response_text": "".join(parts) or None,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": "stream aborted before completion",
                    "cost_usd": None,
                    "ttft_ms": first_delta_ms,
                    "max_tokens": max_tokens,
                }
                try:
                    prompt_id, group_id = resolve_links(
                        app.state.db, request.prompt_id, request.group_id
                    )
                    store.save_run(
                        app.state.db, request.prompt, [aborted], prompt_id, group_id
                    )
                except Exception:
                    logger.exception("failed to persist aborted streamed run")

    return StreamingResponse(events(), media_type="text/event-stream")


def ensure_rowid(value: int) -> None:
    """404 for ids outside SQLite's rowid range.

    Nothing outside the range can exist, so it reads as absence, the
    same answer an in-range miss gets; before this check the overflow
    surfaced from the sqlite bind as a 500.
    """
    if not 1 <= value <= MAX_SQLITE_ROWID:
        raise HTTPException(404, "no such id")


# The empty JSON object is load-bearing: a bodyless POST needs no
# content type, and the guard middleware keys on application/json to
# force hostile cross-site senders into a CORS preflight.
@app.post("/groups", response_model=GroupCreated, status_code=201)
async def create_group(body: dict[str, Any] = Body()) -> dict[str, Any]:
    return {"id": store.create_group(app.state.db)}


@app.get("/groups/{group_id}", response_model=GroupDetail)
async def group_detail(group_id: int) -> dict[str, Any]:
    ensure_rowid(group_id)
    group = store.get_group(app.state.db, group_id)
    if group is None:
        raise HTTPException(404, "no such group")
    return group


@app.get("/models", response_model=CatalogResponse)
async def get_models() -> dict[str, Any]:
    return {
        "models": app.state.catalog["models"],
        "fetched": app.state.catalog["fetched"],
    }


@app.get("/prompts", response_model=PromptList)
async def get_prompts() -> dict[str, Any]:
    return {"prompts": store.list_prompts(app.state.db)}


@app.post("/prompts", response_model=Prompt, status_code=201)
async def create_prompt(body: PromptCreate) -> dict[str, Any]:
    try:
        return store.save_prompt(app.state.db, body.name, body.text)
    except sqlite3.IntegrityError:
        # from None: the duplicate name is an expected outcome being
        # translated to a 409, not an error in handling the error.
        raise HTTPException(
            409, f"a prompt named {body.name!r} already exists"
        ) from None


@app.delete("/prompts/{prompt_id}", status_code=204)
async def remove_prompt(prompt_id: int) -> Response:
    ensure_rowid(prompt_id)
    if not store.delete_prompt(app.state.db, prompt_id):
        raise HTTPException(404, "no such prompt")
    return Response(status_code=204)


@app.get("/runs", response_model=RunList)
async def get_runs(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    # Bounded so history stays cheap as bench.db grows. The 500 ceiling
    # keeps the id lists list_runs binds comfortably under sqlite's
    # variable limit.
    runs = store.list_runs(app.state.db, limit)
    # Append a marker only when a cut happened, so API consumers can tell
    # a short prompt from a truncated one.
    for run in runs:
        if len(run["prompt_text"]) > 80:
            run["prompt_text"] = run["prompt_text"][:80] + "..."
    return {"runs": runs}


@app.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: int) -> dict[str, Any]:
    ensure_rowid(run_id)
    run = store.get_run(app.state.db, run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return run
