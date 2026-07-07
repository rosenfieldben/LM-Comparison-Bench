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
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

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
    run_id: int


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

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


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
    # gather preserves input order, which the frontend relies on to map
    # result columns by position. run_model never raises, so no
    # return_exceptions handling is needed here.
    results = await asyncio.gather(
        *(
            run_model(
                request.prompt,
                m,
                app.state.client,
                max_tokens=effective_budget(request.budget, m),
            )
            for m in request.models
        )
    )
    # Cost is a boundary concern: run_model stays a pure OpenRouter call
    # and the price snapshot lives on app.state. Computed before save_run
    # so history carries the cost as priced at run time.
    for result in results:
        result["cost_usd"] = cost_usd(result, app.state.prices)
    prompt_id, group_id = resolve_links(
        app.state.db, request.prompt_id, request.group_id
    )
    run_id = store.save_run(
        app.state.db, request.prompt, list(results), prompt_id, group_id
    )
    return {"results": list(results), "run_id": run_id}


@app.post("/compare/stream")
async def compare_stream(request: StreamCompareRequest) -> StreamingResponse:
    max_tokens = effective_budget(request.budget, request.model)

    async def events():
        # Server-side observation of the stream, enough to reconstruct
        # a result if the client disconnects before the done event: a
        # stream the user watched happen is still history.
        parts: list[str] = []
        start = time.perf_counter()
        first_delta_ms: float | None = None
        handled = False
        try:
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
            # A client disconnect cancels this generator at a yield
            # before the done branch ever runs. Persist what the server
            # saw (no awaits or yields are legal here, sqlite is sync,
            # so this is safe during unwinding) rather than silently
            # dropping a run whose deltas already reached the browser.
            if not handled:
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
async def get_runs() -> dict:
    runs = store.list_runs(app.state.db)
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
