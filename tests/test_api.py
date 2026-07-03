import json
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from bench.main import app
from bench.models import OPENROUTER_URL

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_response.json").read_text()
)

# Hand-built price cache injected in place of the startup pricing fetch.
# The fixture's usage block is 13 in / 8 out, so model/alpha costs
# 13 * 1e-6 + 8 * 2e-6 = 2.9e-5 USD.
TEST_PRICES = {
    "model/alpha": {"prompt": 1e-06, "completion": 2e-06},
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    # The lifespan refuses to boot without a key, and tests never hit the
    # real network anyway, so any placeholder value works.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Per-test database so tests never touch bench.db in the repo and
    # never see each other's runs.
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    # The lifespan fetches pricing at startup, which happens outside any
    # per-test respx router, so stub it here instead of mocking HTTP.
    async def fake_fetch_prices(client):
        return dict(TEST_PRICES)

    monkeypatch.setattr("bench.main.fetch_prices", fake_fetch_prices)
    with TestClient(app) as c:
        yield c


def response_for(model: str, text: str) -> dict:
    body = json.loads(json.dumps(FIXTURE))
    body["model"] = model
    body["choices"][0]["message"]["content"] = text
    return body


@respx.mock
def test_compare_two_models_preserves_request_order(client):
    def route(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json=response_for(model, f"reply from {model}"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    resp = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/alpha", "model/beta"]},
    )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["model"] for r in results] == ["model/alpha", "model/beta"]
    assert results[0]["response_text"] == "reply from model/alpha"
    assert results[1]["response_text"] == "reply from model/beta"


@respx.mock
def test_compare_one_model_erroring_does_not_sink_the_other(client):
    def route(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "model/broken":
            return httpx.Response(500, json={"error": "upstream exploded"})
        return httpx.Response(200, json=response_for(model, "still fine"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    resp = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/ok", "model/broken"]},
    )

    assert resp.status_code == 200
    ok, broken = resp.json()["results"]
    assert ok["error"] is None
    assert ok["response_text"] == "still fine"
    assert broken["error"] is not None
    assert broken["response_text"] is None


@respx.mock
def test_group_flow_collapses_comparison_into_one_history_entry(client):
    def route(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json=response_for(model, f"reply from {model}"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    group_id = client.post("/groups").json()["id"]
    first = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/alpha"], "group_id": group_id},
    ).json()
    second = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/beta"], "group_id": group_id},
    ).json()

    entries = client.get("/runs").json()["runs"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["type"] == "group"
    assert entry["id"] == group_id
    assert entry["models"] == ["model/alpha", "model/beta"]
    assert entry["run_ids"] == [first["run_id"], second["run_id"]]

    detail = client.get(f"/groups/{group_id}").json()
    assert [r["id"] for r in detail["runs"]] == [first["run_id"], second["run_id"]]
    assert detail["runs"][0]["results"] == first["results"]
    assert detail["runs"][1]["results"] == second["results"]

    assert client.get("/groups/9999").status_code == 404


@respx.mock
def test_stale_group_id_degrades_to_ungrouped(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    resp = client.post(
        "/compare", json={"prompt": "hi", "models": ["model/a"], "group_id": 12345}
    )

    assert resp.status_code == 200
    entries = client.get("/runs").json()["runs"]
    assert len(entries) == 1
    assert entries[0]["type"] == "run"


@respx.mock
def test_cost_computed_from_price_cache(client):
    def route(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "model/limited":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=response_for(model, "hello"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    results = client.post(
        "/compare",
        json={
            "prompt": "hi",
            "models": ["model/alpha", "model/unpriced", "model/limited"],
        },
    ).json()["results"]

    priced, unpriced, errored = results
    # Fixture usage is 13 in / 8 out against TEST_PRICES for model/alpha.
    assert priced["cost_usd"] == pytest.approx(2.9e-05)
    assert unpriced["cost_usd"] is None
    # 429: tokens are None, so cost must be None rather than zero.
    assert errored["cost_usd"] is None

    # Persisted cost matches what /compare returned.
    entries = client.get("/runs").json()["runs"]
    detail = client.get(f"/runs/{entries[0]['id']}").json()
    assert [r["cost_usd"] for r in detail["results"]] == [
        priced["cost_usd"],
        None,
        None,
    ]


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_compare_rejects_empty_prompt(client):
    resp = client.post("/compare", json={"prompt": "", "models": ["model/alpha"]})
    assert resp.status_code == 422


def test_compare_rejects_six_models(client):
    resp = client.post(
        "/compare",
        json={"prompt": "hi", "models": [f"model/m{i}" for i in range(6)]},
    )
    assert resp.status_code == 422


@respx.mock
def test_compare_persists_run_and_history_reflects_it(client):
    def route(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json=response_for(model, f"reply from {model}"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    long_prompt = "x" * 200
    compare = client.post(
        "/compare",
        json={"prompt": long_prompt, "models": ["model/alpha", "model/beta"]},
    ).json()
    assert isinstance(compare["run_id"], int)

    runs = client.get("/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["id"] == compare["run_id"]
    assert runs[0]["models"] == ["model/alpha", "model/beta"]
    assert runs[0]["prompt_text"] == "x" * 80 + "..."

    detail = client.get(f"/runs/{compare['run_id']}").json()
    assert detail["prompt_text"] == long_prompt
    assert detail["results"] == compare["results"]


@respx.mock
def test_runs_short_prompt_is_not_marked_truncated(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    client.post("/compare", json={"prompt": "short prompt", "models": ["model/a"]})

    runs = client.get("/runs").json()["runs"]
    assert runs[0]["prompt_text"] == "short prompt"


def test_create_duplicate_prompt_yields_409(client):
    body = {"name": "greeting", "text": "Say hello."}
    assert client.post("/prompts", json=body).status_code == 201
    resp = client.post("/prompts", json=body)
    assert resp.status_code == 409

    names = [p["name"] for p in client.get("/prompts").json()["prompts"]]
    assert names == ["greeting"]


@respx.mock
def test_delete_prompt_preserves_history(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    prompt = client.post(
        "/prompts", json={"name": "greeting", "text": "Say hello."}
    ).json()
    run_id = client.post(
        "/compare",
        json={"prompt": "Say hello.", "models": ["model/a"], "prompt_id": prompt["id"]},
    ).json()["run_id"]
    assert client.get(f"/runs/{run_id}").json()["prompt_id"] == prompt["id"]

    assert client.delete(f"/prompts/{prompt['id']}").status_code == 204

    runs = client.get("/runs").json()["runs"]
    assert [r["id"] for r in runs] == [run_id]
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["prompt_id"] is None
    assert detail["prompt_text"] == "Say hello."


from stream_helpers import DONE_MARKER, ChunkStream, sse


def stream_events(client, body):
    # Parse by SSE frame boundaries, exactly as the browser client does,
    # not line by line: a regression in the blank-line frame separator
    # must fail here rather than pass unnoticed.
    with client.stream("POST", "/compare/stream", json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = resp.read().decode()
    frames = raw.split("\n\n")
    assert frames[-1] == "", "every SSE event must end with a blank-line separator"
    events = []
    for frame in frames[:-1]:
        data_lines = [l for l in frame.split("\n") if l.startswith("data:")]
        assert len(data_lines) == 1, f"expected exactly one data line per frame: {frame!r}"
        events.append(json.loads(data_lines[0][5:]))
    return events


def alpha_stream():
    return ChunkStream([
        sse({"choices": [{"delta": {"content": "Hel"}}]}),
        sse({"choices": [{"delta": {"content": "lo"}}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        sse({"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 8}}),
        DONE_MARKER,
    ])


@respx.mock
def test_stream_endpoint_frames_sse_and_persists_ttft_and_cost(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    done = events[-1]
    assert "".join(e["text"] for e in events[:-1]) == "Hello"
    assert done["result"]["response_text"] == "Hello"
    assert done["result"]["ttft_ms"] is not None
    assert done["result"]["cost_usd"] == pytest.approx(2.9e-05)

    detail = client.get(f"/runs/{done['run_id']}").json()
    persisted = detail["results"][0]
    assert persisted["response_text"] == "Hello"
    assert persisted["ttft_ms"] == done["result"]["ttft_ms"]
    assert persisted["cost_usd"] == pytest.approx(2.9e-05)


@respx.mock
def test_stream_endpoint_stale_group_id_degrades(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    events = stream_events(
        client, {"prompt": "hi", "model": "model/alpha", "group_id": 777}
    )

    assert events[-1]["type"] == "done"
    entries = client.get("/runs").json()["runs"]
    assert len(entries) == 1
    assert entries[0]["type"] == "run"


@respx.mock
def test_stream_endpoint_grouped_run_lands_in_group(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    group_id = client.post("/groups").json()["id"]
    events = stream_events(
        client, {"prompt": "hi", "model": "model/alpha", "group_id": group_id}
    )

    entries = client.get("/runs").json()["runs"]
    assert entries[0]["type"] == "group"
    assert entries[0]["run_ids"] == [events[-1]["run_id"]]


@respx.mock
def test_stream_endpoint_errored_stream_still_persists(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [sse({"choices": [{"delta": {"content": "par"}}]})],
                exc=httpx.ReadError("dropped"),
            ),
        )
    )

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    done = events[-1]
    assert done["result"]["error"] == "request failed: ReadError"
    assert done["result"]["response_text"] == "par"

    detail = client.get(f"/runs/{done['run_id']}").json()
    persisted = detail["results"][0]
    assert persisted["error"] == "request failed: ReadError"
    assert persisted["response_text"] == "par"


@respx.mock
def test_string_token_counts_yield_none_cost_not_500(client):
    body = response_for("model/alpha", "hello")
    # Some providers have been seen reporting counts as strings; cost
    # must degrade to None instead of raising in the multiplication.
    body["usage"] = {"prompt_tokens": "13", "completion_tokens": "8"}
    respx.post(OPENROUTER_URL).respond(json=body)

    resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["cost_usd"] is None
    assert result["response_text"] == "hello"


@respx.mock
async def test_client_disconnect_persists_partial_run(client):
    # Drive the endpoint's generator directly and close it after one
    # delta: aclose() raises GeneratorExit at the yield, the same
    # mechanism a Starlette client disconnect triggers.
    from bench.main import StreamCompareRequest, compare_stream

    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    resp = await compare_stream(
        StreamCompareRequest(prompt="hi", model="model/alpha")
    )
    gen = resp.body_iterator
    first = await gen.__anext__()
    assert json.loads(first.removeprefix("data: "))["type"] == "delta"
    await gen.aclose()

    entries = client.get("/runs").json()["runs"]
    assert len(entries) == 1
    detail = client.get(f"/runs/{entries[0]['id']}").json()
    persisted = detail["results"][0]
    assert persisted["error"] == "stream aborted before completion"
    assert persisted["response_text"] == "Hel"
    assert persisted["ttft_ms"] is not None
    assert persisted["cost_usd"] is None


@respx.mock
def test_stream_persistence_failure_degrades_to_null_run_id(client, monkeypatch):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    def boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("bench.store.save_run", boom)

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    # The stream must stay intact: full deltas, a done event with the
    # result, and run_id degraded to null instead of a broken tail.
    done = events[-1]
    assert done["type"] == "done"
    assert done["run_id"] is None
    assert done["result"]["response_text"] == "Hello"
