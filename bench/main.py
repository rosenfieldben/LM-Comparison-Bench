"""FastAPI boundary. Pydantic models live here only; internals use plain dicts."""

import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bench import store
from bench.models import run_model


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # Cap at 5: the bench is for side-by-side eyeballing, and each extra
    # model is a concurrent upstream request on one API key.
    models: list[str] = Field(min_length=1, max_length=5)
    # Optional link back to a saved prompt so history can show where a
    # run came from. The run stores its own prompt_text either way.
    prompt_id: int | None = None


class ModelResult(BaseModel):
    model: str
    response_text: str | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None


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


class RunSummary(BaseModel):
    id: int
    created_at: str
    prompt_text: str
    models: list[str]


class RunList(BaseModel):
    runs: list[RunSummary]


class RunDetail(BaseModel):
    id: int
    created_at: str
    prompt_text: str
    prompt_id: int | None
    results: list[ModelResult]


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
    # auth header lives in exactly one place.
    app.state.client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"}
    )
    app.state.db = store.connect(os.environ.get("BENCH_DB", "./bench.db"))
    yield
    await app.state.client.aclose()
    app.state.db.close()


app = FastAPI(title="LM Comparison Bench", lifespan=lifespan)

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> dict:
    # gather preserves input order, which the frontend relies on to map
    # result columns by position. run_model never raises, so no
    # return_exceptions handling is needed here.
    results = await asyncio.gather(
        *(run_model(request.prompt, m, app.state.client) for m in request.models)
    )
    # A stale prompt_id (deleted in another tab) must not sink the run:
    # the upstream calls already happened and prompt_text is the source
    # of truth, so drop the link rather than error.
    prompt_id = request.prompt_id
    if prompt_id is not None and store.get_prompt(app.state.db, prompt_id) is None:
        prompt_id = None
    run_id = store.save_run(app.state.db, request.prompt, list(results), prompt_id)
    return {"results": list(results), "run_id": run_id}


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
    for run in runs:
        run["prompt_text"] = run["prompt_text"][:80]
    return {"runs": runs}


@app.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: int) -> dict:
    run = store.get_run(app.state.db, run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return run
