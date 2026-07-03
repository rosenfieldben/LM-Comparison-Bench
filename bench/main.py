"""FastAPI boundary. Pydantic models live here only; internals use plain dicts."""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bench.models import run_model


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # Cap at 5: the bench is for side-by-side eyeballing, and each extra
    # model is a concurrent upstream request on one API key.
    models: list[str] = Field(min_length=1, max_length=5)


class ModelResult(BaseModel):
    model: str
    response_text: str | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None


class CompareResponse(BaseModel):
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
    yield
    await app.state.client.aclose()


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
    return {"results": list(results)}
