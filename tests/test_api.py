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
