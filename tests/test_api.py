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


@pytest.fixture
def client(monkeypatch):
    # The lifespan refuses to boot without a key, and tests never hit the
    # real network anyway, so any placeholder value works.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
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
