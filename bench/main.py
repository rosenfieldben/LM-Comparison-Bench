"""FastAPI boundary. Pydantic models live here only; internals use plain dicts."""

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers

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


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # Cap at 5: the bench is for side-by-side eyeballing, and each extra
    # model is a concurrent upstream request on one API key.
    models: list[str] = Field(min_length=1, max_length=5)
    # Optional link back to a saved prompt so history can show where a
    # run came from. The run stores its own prompt_text either way.
    prompt_id: int | None = None
    # Optional grouping id so the N per-model requests of one comparison
    # land as one history entry.
    group_id: int | None = None
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
    prompt_id: int | None = None
    group_id: int | None = None
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at boot, not on the first request. A bench with a missing key
    # would otherwise report every model as errored and look like an outage.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before starting the app."
        )
    # One shared client: connection pooling across the fan-out, and the
    # auth header lives in exactly one place. The explicit transport
    # exists to carry TCP keepalive options: extended-budget streams go
    # silent for minutes during hidden reasoning, NAT idle timers cull
    # flows that move no bytes, and the resulting deaths wore mixed
    # ReadError and stall signatures. OS-level probes keep those quiet
    # flows alive; see keepalive_socket_options in models.py.
    app.state.client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.AsyncHTTPTransport(
            socket_options=keepalive_socket_options()
        ),
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

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        # A missing Host header (raw HTTP/1.0 clients) fails closed.
        host = host_header_name(headers.get("host", ""))
        if host not in TRUSTED_HOSTS:
            response = JSONResponse(
                {"detail": "the bench only answers to localhost"},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        # Only POSTs carrying a body need the content-type gate: DELETE
        # is never a "simple" cross-site method, and the frontend's
        # bodyless POST /groups sends no content-type at all.
        has_body = "transfer-encoding" in headers or headers.get(
            "content-length", "0"
        ) not in ("", "0")
        if (
            scope["method"] == "POST"
            and has_body
            and not headers.get("content-type", "").startswith("application/json")
        ):
            response = JSONResponse(
                {"detail": "POST bodies must be application/json"},
                status_code=415,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(LocalOnlyGuard)

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"

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


def cost_usd(result: dict, prices: dict) -> float | None:
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
    return (
        result["prompt_tokens"] * price["prompt"]
        + result["completion_tokens"] * price["completion"]
    )


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


def resolve_links(db, prompt_id: int | None, group_id: int | None):
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
async def compare(request: CompareRequest) -> dict:
    async def limited(model: str) -> dict:
        # One slot per model inside the fan-out, not one around the
        # batch: the cap is on simultaneous paid calls wherever they
        # come from, and concurrent /compare and stream requests share
        # the same gate. Acquiring before run_model keeps the clocks
        # honest for free: its latency clock starts internally, after
        # the slot is already held, so a queued model never reports
        # queue wait as model latency.
        async with app.state.upstream_semaphore:
            return await run_model(
                request.prompt,
                model,
                app.state.client,
                max_tokens=effective_budget(request.budget, model),
            )

    # gather preserves input order, which the frontend relies on to map
    # result columns by position. run_model never raises, so no
    # return_exceptions handling is needed here.
    results = await asyncio.gather(*(limited(m) for m in request.models))
    # Cost is a boundary concern: run_model stays a pure OpenRouter call
    # and the price snapshot lives on app.state. Computed before save_run
    # so history carries the cost as priced at run time.
    for result in results:
        result["cost_usd"] = cost_usd(result, app.state.prices)
    prompt_id, group_id = resolve_links(
        app.state.db, request.prompt_id, request.group_id
    )
    # Same degradation the streaming path already has: by now the money
    # is spent and the results exist, so a persistence failure must cost
    # the history entry, never the response.
    try:
        run_id = store.save_run(
            app.state.db, request.prompt, list(results), prompt_id, group_id
        )
    except Exception:
        logger.exception("failed to persist compare run")
        run_id = None
    return {"results": list(results), "run_id": run_id}


@app.post("/compare/stream")
async def compare_stream(request: StreamCompareRequest) -> StreamingResponse:
    max_tokens = effective_budget(request.budget, request.model)

    async def events():
        # Server-side observation of the stream, enough to reconstruct
        # a result if the client disconnects before the done event: a
        # stream the user watched happen is still history.
        parts: list[str] = []
        first_delta_ms: float | None = None
        handled = False
        started = False
        acquired = False
        start = 0.0

        def release_slot():
            # Idempotent so the done branch and the finally below can
            # both call it: the slot must never be returned twice, and
            # a cancellation while still queued has nothing to return.
            nonlocal acquired
            if acquired:
                acquired = False
                app.state.upstream_semaphore.release()

        try:
            # The slot covers the upstream exchange only, and it is
            # acquired before any clock starts: both this generator's
            # clock and stream_model's internal latency and ttft clocks
            # begin after the slot is held, so a queued run never
            # reports its queue wait as model time. Saturation queues
            # quietly, with no acquisition timeout.
            await app.state.upstream_semaphore.acquire()
            acquired = True
            started = True
            start = time.perf_counter()
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
                result["cost_usd"] = cost_usd(result, app.state.prices)
                prompt_id, group_id = resolve_links(
                    app.state.db, request.prompt_id, request.group_id
                )
                # The response is already on the wire by now, so a
                # persistence failure has no clean HTTP error to use:
                # degrade to run_id null in the done event and log,
                # instead of corrupting the stream tail.
                try:
                    run_id = store.save_run(
                        app.state.db, request.prompt, [result], prompt_id, group_id
                    )
                except Exception:
                    logger.exception("failed to persist streamed run")
                    run_id = None
                yield "data: " + json.dumps(
                    {"type": "done", "result": result, "run_id": run_id}
                ) + "\n\n"
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


@app.post("/groups", response_model=GroupCreated, status_code=201)
async def create_group() -> dict:
    return {"id": store.create_group(app.state.db)}


@app.get("/groups/{group_id}", response_model=GroupDetail)
async def group_detail(group_id: int) -> dict:
    group = store.get_group(app.state.db, group_id)
    if group is None:
        raise HTTPException(404, "no such group")
    return group


@app.get("/models", response_model=CatalogResponse)
async def get_models() -> dict:
    return {
        "models": app.state.catalog["models"],
        "fetched": app.state.catalog["fetched"],
    }


@app.get("/prompts", response_model=PromptList)
async def get_prompts() -> dict:
    return {"prompts": store.list_prompts(app.state.db)}


@app.post("/prompts", response_model=Prompt, status_code=201)
async def create_prompt(body: PromptCreate) -> dict:
    try:
        return store.save_prompt(app.state.db, body.name, body.text)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"a prompt named {body.name!r} already exists")


@app.delete("/prompts/{prompt_id}", status_code=204)
async def remove_prompt(prompt_id: int) -> Response:
    if not store.delete_prompt(app.state.db, prompt_id):
        raise HTTPException(404, "no such prompt")
    return Response(status_code=204)


@app.get("/runs", response_model=RunList)
async def get_runs(limit: int = Query(100, ge=1, le=500)) -> dict:
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
async def get_run(run_id: int) -> dict:
    run = store.get_run(app.state.db, run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return run
