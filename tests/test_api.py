import json
import math
import re
import threading
import typing
from pathlib import Path
from typing import Literal

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from bench import main
from bench.main import MAX_POSITION, app
from bench.models import OPENROUTER_URL, provider_preferences

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_response.json").read_text()
)

# Hand-built catalog injected in place of the startup fetch. The
# fixture's usage block is 13 in / 8 out, so model/alpha costs
# 13 * 1e-6 + 8 * 2e-6 = 2.9e-5 USD.
TEST_PRICES = {
    "model/alpha": {"prompt": 1e-06, "completion": 2e-06},
}
TEST_CATALOG = {
    "fetched": True,
    # A real fetch hashes the response bytes; the stub carries a fixed
    # stand-in so run rows have a concrete digest to be checked against.
    "digest": "d1" * 32,
    "models": [
        {
            "id": "model/alpha",
            "name": "Alpha",
            "context_length": 128000,
            "prompt_price": 1e-06,
            "completion_price": 2e-06,
            "max_completion_tokens": None,
            # Supports everything the bench can send, so it is the model
            # strict mode accepts.
            "supported_parameters": [
                "max_tokens",
                "reasoning",
                "seed",
                "temperature",
                "top_p",
            ],
        },
        {
            "id": "model/bare",
            "name": None,
            "context_length": None,
            "prompt_price": None,
            "completion_price": None,
            "max_completion_tokens": None,
            # The catalog says nothing about its parameters, which strict
            # mode treats as "cannot check" rather than "supports
            # everything".
            "supported_parameters": None,
        },
        # Publishes a completion cap below the extended budget, so the
        # per-model clamp has something to bite on.
        {
            "id": "model/capped",
            "name": "Capped",
            "context_length": 64000,
            "prompt_price": None,
            "completion_price": None,
            "max_completion_tokens": 32000,
            # Takes a budget and nothing else, so a temperature makes it
            # ineligible under require_parameters.
            "supported_parameters": ["max_tokens"],
        },
    ],
    "prices": TEST_PRICES,
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    # The lifespan refuses to boot without a key, and tests never hit the
    # real network anyway, so any placeholder value works.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Per-test database so tests never touch bench.db in the repo and
    # never see each other's runs.
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    # The lifespan fetches the catalog at startup, which happens outside
    # any per-test respx router, so stub it here instead of mocking HTTP.
    async def fake_fetch_catalog(client):
        return json.loads(json.dumps(TEST_CATALOG))

    monkeypatch.setattr("bench.main.fetch_catalog", fake_fetch_catalog)
    # base_url picks the Host header; the default "testserver" would be
    # rejected by the localhost guard like any other non-local name.
    with TestClient(app, base_url="http://localhost") as c:
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

    group_id = client.post("/groups", json={"budget": "standard"}).json()["id"]
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

    # Field by field, minus the row id the stored shape adds; see
    # test_compare_persists_run_and_history_reflects_it for why the two
    # shapes differ on purpose.
    def without_ids(results):
        return [{k: v for k, v in r.items() if k != "id"} for r in results]

    assert without_ids(detail["runs"][0]["results"]) == first["results"]
    assert without_ids(detail["runs"][1]["results"]) == second["results"]

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


# ---- F2.3: one prompt per group.


@respx.mock
def test_review_repro_group_rejects_second_prompt(client):
    """Review finding 8: group membership only checked that the id
    existed, so the API could attach different prompts to one group and
    history then summarized the whole group under the oldest member's
    prompt. Entry now enforces one prompt per group with a 409 before any
    upstream call; same-prompt additions and empty/unknown groups pass."""
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = client.post("/groups", json={"budget": "standard"}).json()["id"]

    r1 = client.post(
        "/compare",
        json={"prompt": "established", "models": ["model/alpha"], "group_id": gid},
    )
    assert r1.status_code == 200
    calls = route.call_count

    # A different prompt in the same group is refused before spending.
    r2 = client.post(
        "/compare",
        json={"prompt": "a different one", "models": ["model/alpha"], "group_id": gid},
    )
    assert r2.status_code == 409
    assert "one prompt" in r2.json()["detail"]
    assert route.call_count == calls

    # The same prompt still passes (a rerun or a second model).
    r3 = client.post(
        "/compare",
        json={"prompt": "established", "models": ["model/alpha"], "group_id": gid},
    )
    assert r3.status_code == 200


@respx.mock
def test_review_repro_stream_group_rejects_second_prompt(client):
    """Review finding 8, streaming path: the same one-prompt-per-group
    check fires at entry, a 409 before any upstream call rather than an
    SSE frame."""
    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    gid = client.post("/groups", json={"budget": "standard"}).json()["id"]

    events = stream_events(
        client, {"prompt": "stream one", "model": "model/alpha", "group_id": gid}
    )
    assert events[-1]["type"] == "done"
    calls = route.call_count

    resp = client.post(
        "/compare/stream",
        json={"prompt": "stream two", "model": "model/alpha", "group_id": gid},
    )
    assert resp.status_code == 409
    assert "one prompt" in resp.json()["detail"]
    assert route.call_count == calls


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
    # Everything the live response carried survives the round trip
    # unchanged. Compared field by field rather than as whole dicts,
    # because the stored shape deliberately carries one field the live
    # one cannot: a persisted result has a row id and a live result does
    # not, which is why StoredModelResult is a subclass rather than a
    # nullable field. A plain equality here would have made that
    # deliberate difference read as a regression.
    stored = [{k: v for k, v in r.items() if k != "id"} for r in detail["results"]]
    assert stored == compare["results"]
    assert all(isinstance(r["id"], int) for r in detail["results"])


@respx.mock
def test_runs_short_prompt_is_not_marked_truncated(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    client.post("/compare", json={"prompt": "short prompt", "models": ["model/a"]})

    runs = client.get("/runs").json()["runs"]
    assert runs[0]["prompt_text"] == "short prompt"


@respx.mock
def test_runs_limit_param_bounds_history(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    client.post("/compare", json={"prompt": "one", "models": ["model/a"]})
    client.post("/compare", json={"prompt": "two", "models": ["model/a"]})

    runs = client.get("/runs?limit=1").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["prompt_text"] == "two"

    # Bounds are enforced, not silently clamped.
    assert client.get("/runs?limit=0").status_code == 422
    assert client.get("/runs?limit=501").status_code == 422


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
        data_lines = [line for line in frame.split("\n") if line.startswith("data:")]
        assert len(data_lines) == 1, (
            f"expected exactly one data line per frame: {frame!r}"
        )
        events.append(json.loads(data_lines[0][5:]))
    return events


def alpha_stream():
    return ChunkStream(
        [
            sse({"choices": [{"delta": {"content": "Hel"}}]}),
            sse({"choices": [{"delta": {"content": "lo"}}]}),
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            sse(
                {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 8}}
            ),
            DONE_MARKER,
        ]
    )


@respx.mock
def test_stream_endpoint_frames_sse_and_persists_ttft_and_cost(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    # A free slot: a started frame leads, no queued before it.
    assert [e["type"] for e in events] == ["started", "delta", "delta", "done"]
    done = events[-1]
    deltas = [e for e in events if e["type"] == "delta"]
    assert "".join(e["text"] for e in deltas) == "Hello"
    assert done["result"]["response_text"] == "Hello"
    assert done["result"]["ttft_ms"] is not None
    assert done["result"]["cost_usd"] == pytest.approx(2.9e-05)

    detail = client.get(f"/runs/{done['run_id']}").json()
    persisted = detail["results"][0]
    assert persisted["response_text"] == "Hello"
    assert persisted["ttft_ms"] == done["result"]["ttft_ms"]
    assert persisted["cost_usd"] == pytest.approx(2.9e-05)


@respx.mock
def test_review_repro_clean_eof_persists_as_error(client):
    """External review finding 2: a clean EOF without [DONE] must persist
    through the streaming API as an error with the partial text kept, not
    as a complete run that silently corrupts the comparison."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream([sse({"choices": [{"delta": {"content": "Hel"}}]})]),
        )
    )

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    done = events[-1]
    assert done["type"] == "done"
    assert done["result"]["response_text"] == "Hel"
    assert done["result"]["error"] is not None
    assert "no [DONE]" in done["result"]["error"]

    detail = client.get(f"/runs/{done['run_id']}").json()
    persisted = detail["results"][0]
    assert persisted["response_text"] == "Hel"
    assert "no [DONE]" in persisted["error"]


def _assert_full_csp(csp: str) -> None:
    # The full policy (F3.3): default-deny with every source at 'self', the
    # inline and remote origins closed, and frame-ancestors carried over.
    for directive in (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self'",
        "font-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ):
        assert directive in csp, f"missing CSP directive: {directive}"
    assert "unsafe-inline" not in csp


def test_review_repro_index_denies_framing(client):
    """External review finding 4: nothing stopped a hostile page from
    framing the localhost UI and redressing a Run click into paid work.
    Every response must carry the anti-framing headers, now inside the
    full content security policy (F3.3)."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["x-frame-options"] == "DENY"
    _assert_full_csp(resp.headers["content-security-policy"])


@respx.mock
def test_review_repro_stream_response_denies_framing(client):
    """External review finding 4: the streaming response carries the
    anti-framing headers too, injected on response start without buffering
    the SSE body (the streaming and disconnect suites prove no buffering)."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with client.stream(
        "POST", "/compare/stream", json={"prompt": "hi", "model": "model/alpha"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["x-frame-options"] == "DENY"
        _assert_full_csp(resp.headers["content-security-policy"])
        resp.read()


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

    group_id = client.post("/groups", json={"budget": "standard"}).json()["id"]
    events = stream_events(
        client, {"prompt": "hi", "model": "model/alpha", "group_id": group_id}
    )

    entries = client.get("/runs").json()["runs"]
    assert entries[0]["type"] == "group"
    assert entries[0]["run_ids"] == [events[-1]["run_id"]]


@respx.mock
def test_stream_endpoint_extended_budget_clamps_and_persists(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    events = stream_events(
        client, {"prompt": "hi", "model": "model/capped", "budget": "extended"}
    )

    assert seen["max_tokens"] == 32000
    done = events[-1]
    assert done["result"]["max_tokens"] == 32000
    detail = client.get(f"/runs/{done['run_id']}").json()
    assert detail["results"][0]["max_tokens"] == 32000


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

    resp = await compare_stream(StreamCompareRequest(prompt="hi", model="model/alpha"))
    gen = resp.body_iterator
    # The started frame leads with a free slot; consume it, then the
    # first delta, then disconnect with only "Hel" seen.
    started = await gen.__anext__()
    assert json.loads(started.removeprefix("data: "))["type"] == "started"
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


from bench.models import BUDGET_EXTENDED, BUDGET_STANDARD


@respx.mock
def test_compare_default_budget_sends_standard(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=FIXTURE)

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    assert seen["max_tokens"] == BUDGET_STANDARD
    assert resp.json()["results"][0]["max_tokens"] == BUDGET_STANDARD


@respx.mock
def test_compare_extended_budget_sends_extended(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=FIXTURE)

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    resp = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/alpha"], "budget": "extended"},
    )

    assert seen["max_tokens"] == BUDGET_EXTENDED
    assert resp.json()["results"][0]["max_tokens"] == BUDGET_EXTENDED


@respx.mock
def test_extended_budget_clamps_to_model_cap_and_persists_effective(client):
    seen_by_model = {}

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_by_model[body["model"]] = body["max_tokens"]
        return httpx.Response(200, json=response_for(body["model"], "ok"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    resp = client.post(
        "/compare",
        json={
            "prompt": "hi",
            "models": ["model/capped", "model/alpha"],
            "budget": "extended",
        },
    )

    # The clamp is per model within one request: capped gets its
    # published limit, uncapped gets the full extended budget.
    assert seen_by_model == {
        "model/capped": 32000,
        "model/alpha": BUDGET_EXTENDED,
    }
    capped, alpha = resp.json()["results"]
    assert capped["max_tokens"] == 32000
    assert alpha["max_tokens"] == BUDGET_EXTENDED

    # History stores the effective value actually sent, not the name.
    detail = client.get(f"/runs/{resp.json()['run_id']}").json()
    assert [r["max_tokens"] for r in detail["results"]] == [32000, BUDGET_EXTENDED]


def test_unknown_budget_string_is_422(client):
    resp = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/alpha"], "budget": "unlimited"},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/compare/stream",
        json={"prompt": "hi", "model": "model/alpha", "budget": "1000000"},
    )
    assert resp.status_code == 422


def test_non_local_host_header_is_403(client):
    # DNS rebinding: the attacker's page becomes same-origin with the
    # bench, but its requests still carry the attacker's hostname.
    for host in ("attacker.example", "attacker.example:8000", "testserver"):
        resp = client.get("/models", headers={"host": host})
        assert resp.status_code == 403, host


def test_local_host_header_variants_are_accepted(client):
    # Every spelling a local browser or curl actually sends, ports and
    # bracketed IPv6 included.
    for host in ("localhost", "localhost:8000", "127.0.0.1:8000", "[::1]:8000"):
        resp = client.get("/models", headers={"host": host})
        assert resp.status_code == 200, host


@respx.mock
def test_cross_site_textplain_post_is_415_and_never_reaches_upstream(client):
    # A "simple" cross-site request: fetch() with a text/plain body
    # needs no CORS preflight, so the guard is the only thing standing
    # between a malicious page and the API key's wallet.
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    resp = client.post(
        "/compare",
        content=json.dumps({"prompt": "hi", "models": ["model/alpha"]}),
        headers={"content-type": "text/plain"},
    )

    assert resp.status_code == 415
    assert not route.called


def test_group_create_requires_json_body(client):
    # The old bodyless exemption was the reproduced cross-site hole:
    # every POST now needs the JSON content type, and the frontend
    # sends an empty JSON object on /groups.
    assert client.post("/groups").status_code == 415
    assert client.post("/groups", json={"budget": "standard"}).status_code == 201


def test_models_endpoint_returns_catalog_with_pricing(client):
    body = client.get("/models").json()

    assert body["fetched"] is True
    alpha = body["models"][0]
    assert alpha["id"] == "model/alpha"
    assert alpha["name"] == "Alpha"
    assert alpha["context_length"] == 128000
    assert alpha["prompt_price"] == 1e-06
    assert alpha["completion_price"] == 2e-06
    assert alpha["max_completion_tokens"] is None
    # Entries with missing metadata survive as None rather than vanish.
    bare = body["models"][1]
    assert bare["id"] == "model/bare"
    assert bare["name"] is None
    assert bare["context_length"] is None
    assert bare["prompt_price"] is None
    # The published completion cap survives into the API instead of
    # being silently filtered by the response model.
    assert body["models"][2]["max_completion_tokens"] == 32000


def test_lifespan_client_transport_carries_keepalive_options(monkeypatch, tmp_path):
    from bench.models import keepalive_socket_options

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    async def fake_fetch_catalog(client):
        return json.loads(json.dumps(TEST_CATALOG))

    monkeypatch.setattr("bench.main.fetch_catalog", fake_fetch_catalog)

    # Spy on the transport constructor rather than digging through
    # private pool attributes: the contract is what the lifespan asked
    # for, not where httpcore happens to store it this release.
    captured = {}
    real_transport = httpx.AsyncHTTPTransport

    def spying_transport(*args, **kwargs):
        captured.update(kwargs)
        return real_transport(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", spying_transport)

    with TestClient(app):
        pass

    assert captured["socket_options"] == keepalive_socket_options()


def test_offline_boot_models_empty_and_compare_still_works(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    async def offline_catalog(client):
        return {"fetched": False, "models": [], "prices": {}}

    monkeypatch.setattr("bench.main.fetch_catalog", offline_catalog)
    with TestClient(app, base_url="http://localhost") as c:
        body = c.get("/models").json()
        assert body == {"models": [], "fetched": False, "data_policy": "standard"}

        with respx.mock:
            respx.post(OPENROUTER_URL).respond(json=FIXTURE)
            resp = c.post("/compare", json={"prompt": "hi", "models": ["model/a"]})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["response_text"] is not None
        # No price cache on an offline boot: cost is unavailable, not wrong.
        assert result["cost_usd"] is None


@respx.mock
def test_compare_persists_generation_id_and_finish_reason(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    result = resp.json()["results"][0]
    assert result["generation_id"] == "gen-1751500123-Xk3mQpR7vNwB2aZd"
    assert result["finish_reason"] == "stop"

    detail = client.get(f"/runs/{resp.json()['run_id']}").json()
    persisted = detail["results"][0]
    assert persisted["generation_id"] == "gen-1751500123-Xk3mQpR7vNwB2aZd"
    assert persisted["finish_reason"] == "stop"


def provenance_stream():
    return ChunkStream(
        [
            sse({"id": "gen-stream-1", "choices": [{"delta": {"content": "Hel"}}]}),
            sse({"id": "gen-stream-1", "choices": [{"delta": {"content": "lo"}}]}),
            sse(
                {
                    "id": "gen-stream-1",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                }
            ),
            sse(
                {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 8}}
            ),
            DONE_MARKER,
        ]
    )


@respx.mock
def test_stream_persists_generation_id_and_finish_reason(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=provenance_stream())
    )

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    done = events[-1]
    assert done["result"]["generation_id"] == "gen-stream-1"
    assert done["result"]["finish_reason"] == "stop"

    detail = client.get(f"/runs/{done['run_id']}").json()
    persisted = detail["results"][0]
    assert persisted["generation_id"] == "gen-stream-1"
    assert persisted["finish_reason"] == "stop"


@respx.mock
def test_stream_missing_provenance_persists_as_null(client):
    # A stream whose chunks carry no id and no finish_reason must land
    # in history with both fields honestly NULL, not invented.
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    sse({"choices": [{"delta": {"content": "hi"}}]}),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    done = events[-1]
    assert done["result"]["generation_id"] is None
    assert done["result"]["finish_reason"] is None

    detail = client.get(f"/runs/{done['run_id']}").json()
    assert detail["results"][0]["generation_id"] is None
    assert detail["results"][0]["finish_reason"] is None


import logging


@respx.mock
def test_compare_persistence_failure_degrades_to_null_run_id(
    client, monkeypatch, caplog
):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    def boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("bench.store.save_run", boom)

    with caplog.at_level(logging.ERROR, logger="bench.main"):
        resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    # The money is spent and the results exist: losing history must not
    # lose the response, exactly as on the streaming path.
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] is None
    assert body["results"][0]["response_text"] is not None
    assert "post-upstream processing failed" in caplog.text


# ---- Upstream concurrency cap. These tests drive the endpoint
# ---- functions directly (the established pattern for stream control)
# ---- because TestClient serializes requests, and the whole point is
# ---- overlap. The client context stays open so app.state exists.

import asyncio

from bench.main import CompareRequest, StreamCompareRequest, compare, compare_stream


def make_client(monkeypatch, tmp_path, max_upstream):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    async def fake_fetch_catalog(client):
        return json.loads(json.dumps(TEST_CATALOG))

    monkeypatch.setattr("bench.main.fetch_catalog", fake_fetch_catalog)
    monkeypatch.setattr("bench.main.MAX_CONCURRENT_UPSTREAM", max_upstream)
    return TestClient(app, base_url="http://localhost")


async def consume_stream(model, prompt="hi"):
    resp = await compare_stream(StreamCompareRequest(prompt=prompt, model=model))
    return [
        json.loads(chunk.removeprefix("data: ")) async for chunk in resp.body_iterator
    ]


async def settle(condition, rounds=400):
    # Poll a condition with real scheduling gaps; parked coroutines need
    # loop turns to reach their park point.
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


@respx.mock
async def test_stream_upstream_concurrency_never_exceeds_cap(monkeypatch, tmp_path):
    gate = asyncio.Event()
    tracker = {"in_flight": 0, "max": 0, "started": 0}

    class ParkedStream(httpx.AsyncByteStream):
        # In-flight is counted across the body's consumption: the slot
        # must be held for the exchange, not just the request. The
        # decrement sits before the final chunk rather than in a
        # finally, because async generator finalization is deferred to
        # GC and would keep finished exchanges counted while their
        # successors start.
        async def __aiter__(self):
            tracker["in_flight"] += 1
            tracker["started"] += 1
            tracker["max"] = max(tracker["max"], tracker["in_flight"])
            yield sse({"choices": [{"delta": {"content": "ok"}}]})
            await gate.wait()
            tracker["in_flight"] -= 1
            yield DONE_MARKER

    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=ParkedStream())
    )

    with make_client(monkeypatch, tmp_path, 2):
        tasks = [asyncio.create_task(consume_stream(f"model/m{i}")) for i in range(4)]
        await settle(lambda: tracker["started"] == 2)
        # Extra loop turns: the two queued streams must stay queued
        # while both slots are occupied.
        for _ in range(20):
            await asyncio.sleep(0.005)
        assert tracker["started"] == 2
        assert tracker["max"] == 2

        gate.set()
        all_events = await asyncio.gather(*tasks)

    assert tracker["started"] == 4
    assert tracker["max"] == 2
    # Queueing must not corrupt any stream's own result.
    for i, events in enumerate(all_events):
        done = events[-1]
        assert done["type"] == "done"
        assert done["result"]["model"] == f"model/m{i}"
        assert done["result"]["response_text"] == "ok"


@respx.mock
async def test_compare_fanout_respects_upstream_cap(monkeypatch, tmp_path):
    gate = asyncio.Event()
    tracker = {"in_flight": 0, "max": 0, "started": 0}

    async def route(request):
        model = json.loads(request.content)["model"]
        tracker["in_flight"] += 1
        tracker["started"] += 1
        tracker["max"] = max(tracker["max"], tracker["in_flight"])
        try:
            await gate.wait()
        finally:
            tracker["in_flight"] -= 1
        return httpx.Response(200, json=response_for(model, f"reply from {model}"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    models = [f"model/m{i}" for i in range(4)]
    with make_client(monkeypatch, tmp_path, 2):
        task = asyncio.create_task(compare(CompareRequest(prompt="hi", models=models)))
        await settle(lambda: tracker["started"] == 2)
        for _ in range(20):
            await asyncio.sleep(0.005)
        assert tracker["started"] == 2
        assert tracker["max"] == 2

        gate.set()
        body = await task

    assert tracker["started"] == 4
    assert tracker["max"] == 2
    # gather order and content survive the queueing.
    assert [r["model"] for r in body["results"]] == models
    assert [r["response_text"] for r in body["results"]] == [
        f"reply from {m}" for m in models
    ]


@respx.mock
async def test_compare_queue_wait_is_not_reported_as_latency(monkeypatch, tmp_path):
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def route(request):
        model = json.loads(request.content)["model"]
        if model == "model/slow":
            entered.set()
            await gate.wait()
        return httpx.Response(200, json=response_for(model, "ok"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    with make_client(monkeypatch, tmp_path, 1):
        task = asyncio.create_task(
            compare(CompareRequest(prompt="hi", models=["model/slow", "model/fast"]))
        )
        await asyncio.wait_for(entered.wait(), 5)
        # model/fast sits queued behind the only slot this whole time.
        await asyncio.sleep(0.6)
        gate.set()
        body = await task

    slow, fast = body["results"]
    assert slow["latency_ms"] >= 500
    # The queued model waited over half a second for its slot; its own
    # mocked exchange is near-instant, and that is what latency must
    # report.
    assert fast["latency_ms"] < 300


@respx.mock
async def test_stream_queue_wait_is_not_reported_as_latency_or_ttft(
    monkeypatch, tmp_path
):
    gate = asyncio.Event()
    entered = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            yield sse({"choices": [{"delta": {"content": "wait"}}]})
            await gate.wait()
            yield DONE_MARKER

    def route(request):
        model = json.loads(request.content)["model"]
        if model == "model/slow":
            return httpx.Response(200, stream=SlowStream())
        return httpx.Response(
            200,
            stream=ChunkStream(
                [
                    sse({"choices": [{"delta": {"content": "quick"}}]}),
                    DONE_MARKER,
                ]
            ),
        )

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    with make_client(monkeypatch, tmp_path, 1):
        slow_task = asyncio.create_task(consume_stream("model/slow"))
        await asyncio.wait_for(entered.wait(), 5)
        fast_task = asyncio.create_task(consume_stream("model/fast"))
        # model/fast sits queued behind the only slot this whole time.
        await asyncio.sleep(0.6)
        gate.set()
        slow_events = await slow_task
        fast_events = await fast_task

    slow_done = slow_events[-1]["result"]
    fast_done = fast_events[-1]["result"]
    assert slow_done["latency_ms"] >= 500
    assert fast_done["response_text"] == "quick"
    # Both clocks start after the slot is held: neither total latency
    # nor time to first token may include the queue wait.
    assert fast_done["latency_ms"] < 300
    assert fast_done["ttft_ms"] < 300


def test_favicon_served_inline_with_cache_headers(client):
    resp = client.get("/favicon.ico")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    # Long-lived caching is the point: one request per browser, ever.
    assert "max-age" in resp.headers.get("cache-control", "")
    assert b"<svg" in resp.content


# ---- Review reproductions. These four requests reproduced real
# ---- defects in an adversarial external review; they stay in the
# ---- suite verbatim so the incidents stay in its memory.


@respx.mock
def test_review_repro_huge_group_id_is_422_before_any_spend(client):
    # Reproduction 1: group_id=10**100 passed Pydantic, the paid call
    # completed, then resolve_links raised OverflowError at the sqlite
    # bind and the response was a 500 with nothing stored. Now the
    # boundary rejects it before any upstream request exists.
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    resp = client.post(
        "/compare",
        json={"prompt": "hi", "models": ["model/alpha"], "group_id": 10**100},
    )
    assert resp.status_code == 422
    assert not route.called

    resp = client.post(
        "/compare/stream",
        json={"prompt": "hi", "model": "model/alpha", "prompt_id": 10**100},
    )
    assert resp.status_code == 422
    assert not route.called


@respx.mock
def test_review_repro_string_token_counts_do_not_poison_history(client):
    # Reproduction 2: a provider answered "prompt_tokens": "n/a"; the
    # raw value was persisted, /compare 500ed at its response boundary
    # and the stored row made GET /runs/{id} 500 forever.
    body = json.loads(json.dumps(FIXTURE))
    body["usage"] = {"prompt_tokens": "n/a", "completion_tokens": "n/a"}
    respx.post(OPENROUTER_URL).respond(json=body)

    resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["response_text"] is not None
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None

    detail = client.get(f"/runs/{resp.json()['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["results"][0]["prompt_tokens"] is None

    # A legacy row poisoned by the pre-normalization code (written
    # around save_run, exactly as the old ingestion effectively did)
    # must also read back, repaired, instead of 500ing forever.
    db = client.app.state.db
    with db:
        cur = db.execute(
            "INSERT INTO runs (prompt_text, created_at) VALUES ('legacy', '2026-01-01T00:00:00+00:00')"
        )
        legacy_id = cur.lastrowid
        db.execute(
            "INSERT INTO results (run_id, model, response_text, prompt_tokens,"
            " completion_tokens, latency_ms) VALUES (?, 'old/model', 'hi',"
            " 'n/a', '12', 'slow')",
            (legacy_id,),
        )
    legacy = client.get(f"/runs/{legacy_id}")
    assert legacy.status_code == 200
    row = legacy.json()["results"][0]
    assert row["response_text"] == "hi"
    assert row["prompt_tokens"] is None
    # SQLite's INTEGER affinity already coerced the losslessly numeric
    # string at insert time; repair-on-read serves what survived.
    assert row["completion_tokens"] == 12
    assert row["latency_ms"] is None


def test_review_repro_bodyless_cross_site_group_create_rejected(client):
    # Reproduction 3: a bodyless cross-site POST /groups returned 201
    # because the JSON guard exempted bodyless posts. The verbatim
    # attack request now dies at the middleware.
    resp = client.post(
        "/groups",
        headers={
            "Origin": "https://attacker.example",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    assert 400 <= resp.status_code < 500

    # The frontend path, a same-origin JSON POST with an empty object,
    # still works.
    assert client.post("/groups", json={"budget": "standard"}).status_code == 201


# Reproduction 4 (world-readable bench.db) lives in test_store.py as
# test_review_repro_fresh_database_is_private, next to connect().


@respx.mock
def test_compare_survives_resolve_links_failure(client, monkeypatch, caplog):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    def boom(*args, **kwargs):
        raise OverflowError("simulated post-upstream failure")

    monkeypatch.setattr("bench.main.resolve_links", boom)

    with caplog.at_level(logging.ERROR, logger="bench.main"):
        resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    # The invariant: after money is spent, no code path may convert
    # results into an error response.
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] is None
    assert body["results"][0]["response_text"] is not None
    assert "post-upstream processing failed" in caplog.text


@respx.mock
def test_stream_survives_resolve_links_failure(client, monkeypatch):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    def boom(*args, **kwargs):
        raise OverflowError("simulated post-stream failure")

    monkeypatch.setattr("bench.main.resolve_links", boom)

    events = stream_events(client, {"prompt": "hi", "model": "model/alpha"})

    done = events[-1]
    assert done["type"] == "done"
    assert done["run_id"] is None
    assert done["result"]["response_text"] == "Hello"


def test_out_of_range_path_ids_read_as_absent(client):
    # Out of sqlite's rowid range nothing can exist, so the answer is
    # the same 404 an in-range miss gets, not a bind-overflow 500.
    huge = str(10**100)
    assert client.get(f"/runs/{huge}").status_code == 404
    assert client.get(f"/groups/{huge}").status_code == 404
    assert client.delete(f"/prompts/{huge}").status_code == 404
    assert client.get("/runs/0").status_code == 404


# ---- F3: per-boot spend ceiling.


def spend_client(monkeypatch, tmp_path, limit=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))
    if limit is not None:
        monkeypatch.setenv("BENCH_SPEND_LIMIT_USD", str(limit))
    else:
        monkeypatch.delenv("BENCH_SPEND_LIMIT_USD", raising=False)

    async def fake_fetch_catalog(client):
        return json.loads(json.dumps(TEST_CATALOG))

    monkeypatch.setattr("bench.main.fetch_catalog", fake_fetch_catalog)
    return TestClient(app, base_url="http://localhost")


@respx.mock
def test_spend_ceiling_refuses_at_boundary_without_upstream_call(monkeypatch, tmp_path):
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        # Already over the ceiling from prior runs this boot.
        c.app.state.accumulated_spend_usd = 1.5
        for path, body in (
            ("/compare", {"prompt": "hi", "models": ["model/alpha"]}),
            ("/compare/stream", {"prompt": "hi", "model": "model/alpha"}),
        ):
            resp = c.post(path, json=body)
            assert resp.status_code == 402, path
            detail = resp.json()["detail"]
            assert "spend ceiling" in detail
            assert "$1.50" in detail and "$1.00" in detail
        # The refusal never reached upstream: no money moved.
        assert route.call_count == 0


@respx.mock
def test_spend_accumulates_across_runs_and_ignores_unpriced(monkeypatch, tmp_path):
    def route(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json=response_for(model, "hi"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        assert c.app.state.accumulated_spend_usd == 0.0
        c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        # Two priced alpha runs at 2.9e-5 each.
        assert c.app.state.accumulated_spend_usd == pytest.approx(5.8e-5)
        # An unpriced model (no entry in the price cache) does not move
        # the counter, matching the documented semantics.
        c.post("/compare", json={"prompt": "hi", "models": ["model/bare"]})
        assert c.app.state.accumulated_spend_usd == pytest.approx(5.8e-5)


@respx.mock
def test_stream_run_accumulates_spend(monkeypatch, tmp_path):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        stream_events(c, {"prompt": "hi", "model": "model/alpha"})
        assert c.app.state.accumulated_spend_usd == pytest.approx(2.9e-5)


@respx.mock
def test_unset_spend_limit_never_blocks(monkeypatch, tmp_path):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    with spend_client(monkeypatch, tmp_path, limit=None) as c:
        assert c.app.state.spend_limit_usd is None
        # Even a large accumulated figure cannot block without a limit.
        c.app.state.accumulated_spend_usd = 1000.0
        resp = c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        assert resp.status_code == 200


@respx.mock
def test_spend_ceiling_message_keeps_sub_cent_figures_truthful(monkeypatch, tmp_path):
    # A priced run costs 2.9e-5 here, so a ceiling set at the bench's
    # native scale must not collapse to $0.0000 in the refusal: four
    # decimals would misreport a real $0.00004 limit as zero.
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    with spend_client(monkeypatch, tmp_path, limit=0.00004) as c:
        c.app.state.accumulated_spend_usd = 5.8e-5
        resp = c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        assert resp.status_code == 402
        detail = resp.json()["detail"]
        assert "$0.000058 of $0.00004 limit" in detail


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "0", "abc"])
def test_review_repro_invalid_spend_limit_fails_boot(monkeypatch, tmp_path, bad):
    """External review finding 3: BENCH_SPEND_LIMIT_USD was parsed with a
    bare float() and never validated, so nan and inf produced a ceiling
    never crossed (silently disabling the limit), and negative or zero
    produced a nonsensical or ambiguous one. Each must fail boot with the
    variable named."""
    with (
        pytest.raises(RuntimeError, match="BENCH_SPEND_LIMIT_USD"),
        spend_client(monkeypatch, tmp_path, limit=bad),
    ):
        pass


@respx.mock
def test_review_repro_nan_price_counts_as_unpriced_not_poison(monkeypatch, tmp_path):
    """External review finding 3: a NaN price produced a NaN cost, and a
    NaN summed into accumulated spend made accumulated >= limit permanently
    false, silently disabling the ceiling. A NaN-priced run must count as
    unpriced (cost None), leaving the counter finite and unmoved, not
    poison it."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        # A NaN price injected past the catalog guard, standing in for a
        # rogue price that reached app.state before fetch_catalog learned
        # to reject non-finite values.
        c.app.state.prices["model/alpha"] = {"prompt": float("nan"), "completion": 0.0}
        resp = c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["cost_usd"] is None
        assert c.app.state.accumulated_spend_usd == 0.0


def test_review_repro_record_spend_ignores_non_finite(monkeypatch, tmp_path):
    """Closing review: the accumulator is the spend invariant's last line.
    cost_usd already screens non-finite totals, but record_spend must also
    refuse them so a caller that bypassed that screen cannot poison the
    counter; one NaN would make the ceiling comparison permanently false."""
    from bench import main as bench_main

    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        c.app.state.accumulated_spend_usd = 0.5
        bench_main.record_spend(float("nan"))
        bench_main.record_spend(float("inf"))
        bench_main.record_spend(None)
        assert c.app.state.accumulated_spend_usd == 0.5
        # A finite cost still accumulates.
        bench_main.record_spend(0.25)
        assert c.app.state.accumulated_spend_usd == 0.75


# ---- F1.2: post-admission spend recheck.


@respx.mock
async def test_review_repro_stream_rechecks_ceiling_after_admission(
    monkeypatch, tmp_path
):
    """External review finding 1 (High): admission was checked only at
    entry, so N stream requests admitted below the limit all executed even
    after an earlier one's recorded spend crossed it, bounding overshoot by
    lineup size rather than the semaphore. A run admitted below the ceiling
    must be refused, before spending, if the ceiling is crossed before it
    acquires its slot."""
    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        # Admitted at entry below the limit.
        assert c.app.state.accumulated_spend_usd == 0.0
        resp = await compare_stream(
            StreamCompareRequest(prompt="hi", model="model/alpha")
        )
        gen = resp.body_iterator
        # A concurrent run's spend crosses the ceiling before this one
        # reaches its slot.
        c.app.state.accumulated_spend_usd = 1.5
        sem_before = c.app.state.upstream_semaphore._value
        frames = [json.loads(f.removeprefix("data: ")) async for f in gen]

        # One done frame, refusal error, run_id null, no started or delta.
        assert [f["type"] for f in frames] == ["done"]
        assert frames[0]["run_id"] is None
        result = frames[0]["result"]
        assert "refused before reaching upstream" in result["error"]
        # Shaped like a done result: model and effective budget set, every
        # metric and text field None.
        assert result["model"] == "model/alpha"
        assert result["max_tokens"] == 16384
        for key in (
            "response_text",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "ttft_ms",
            "cost_usd",
            "generation_id",
            "finish_reason",
        ):
            assert result[key] is None
        # Slot returned (net zero), no upstream call, nothing persisted.
        assert c.app.state.upstream_semaphore._value == sem_before
        assert route.call_count == 0
        assert c.get("/runs").json()["runs"] == []


@respx.mock
def test_review_repro_compare_rechecks_ceiling_mid_batch(monkeypatch, tmp_path):
    """External review finding 1 (High): the batch endpoint checked
    admission once at entry, so once spend crossed the ceiling mid-batch
    every already-admitted model still called upstream. The recheck under
    the held slot must refuse the not-yet-started models before they
    spend, while the batch still persists with the refusal row."""
    calls = {"n": 0}

    def route(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        model = json.loads(request.content)["model"]
        # The first call directly crosses the ceiling, standing in for a
        # concurrent request's recorded spend (this batch's own spend is
        # recorded only after the gather); the next model must then be
        # refused before it reaches upstream.
        if calls["n"] == 1:
            app.state.accumulated_spend_usd = 5.0
        return httpx.Response(200, json=response_for(model, "ok"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    monkeypatch.setenv("BENCH_SPEND_LIMIT_USD", "1.0")
    # A semaphore of one forces the two models to run one after the other,
    # so the first's spend is visible when the second rechecks.
    with make_client(monkeypatch, tmp_path, 1) as c:
        resp = c.post(
            "/compare",
            json={"prompt": "hi", "models": ["model/alpha", "model/alpha"]},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        # First ran; second refused before any upstream call.
        assert calls["n"] == 1
        assert results[0]["error"] is None
        assert "refused before reaching upstream" in results[1]["error"]
        # The batch persisted, refusal row and all.
        run_id = resp.json()["run_id"]
        assert run_id is not None
        detail = c.get(f"/runs/{run_id}").json()
        assert len(detail["results"]) == 2
        assert detail["results"][1]["error"] is not None


# ---- F4: queued state frames.


@respx.mock
async def test_stream_emits_started_and_no_queued_when_slot_free(monkeypatch, tmp_path):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with make_client(monkeypatch, tmp_path, 5):
        events = await consume_stream("model/alpha")

    types = [e["type"] for e in events]
    assert "queued" not in types
    assert types[0] == "started"
    assert types[-1] == "done"


@respx.mock
async def test_stream_emits_queued_before_started_when_saturated(monkeypatch, tmp_path):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with make_client(monkeypatch, tmp_path, 1) as c:
        # Hold the only slot by hand so the run must queue for it.
        await c.app.state.upstream_semaphore.acquire()
        resp = await compare_stream(
            StreamCompareRequest(prompt="hi", model="model/alpha")
        )
        gen = resp.body_iterator

        first = json.loads((await gen.__anext__()).removeprefix("data: "))
        assert first["type"] == "queued"

        # Free the slot; the run acquires it and announces started.
        c.app.state.upstream_semaphore.release()
        second = json.loads((await gen.__anext__()).removeprefix("data: "))
        assert second["type"] == "started"

        rest = [json.loads(f.removeprefix("data: ")) async for f in gen]
        assert rest[-1]["type"] == "done"


@respx.mock
async def test_disconnect_at_started_frame_persists_sane_latency(monkeypatch, tmp_path):
    # A client drop during the flush of the started frame finalizes the
    # generator while it is suspended at that yield, before the async for
    # over stream_model runs. The persisted abort must carry a real
    # elapsed latency, not the raw perf_counter reference (host uptime in
    # ms): the clock starts before the frame, so it is always valid when
    # the finally runs. Regression for the window where start was set
    # after the started yield.
    from bench import store

    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with make_client(monkeypatch, tmp_path, 5) as c:
        resp = await compare_stream(
            StreamCompareRequest(prompt="hi", model="model/alpha")
        )
        gen = resp.body_iterator
        first = json.loads((await gen.__anext__()).removeprefix("data: "))
        assert first["type"] == "started"
        # Close at the started-yield suspension, before any delta arrives.
        await gen.aclose()

        run = store.get_run(c.app.state.db, 1)

    assert run is not None
    latency = run["results"][0]["latency_ms"]
    assert latency is not None
    assert 0.0 <= latency < 60_000.0


# ---- F4.2: the peer address, not just the Host header, must be loopback.


def _call_guard_with_peer(client_tuple):
    """Drive the ASGI app directly so scope["client"] is exactly what a
    real socket would present. TestClient always synthesizes its own peer,
    so it cannot express a remote caller; this is the only seam that can.
    Returns (status, body_bytes).
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/runs",
        "raw_path": b"/runs",
        "query_string": b"",
        "root_path": "",
        # A perfectly valid Host header: the point is that a remote client
        # can forge this, so the header alone cannot be the whole defense.
        "headers": [(b"host", b"localhost:8000")],
        "client": client_tuple,
        "server": ("127.0.0.1", 8000),
    }
    received = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        received.append(message)

    asyncio.run(main.LocalOnlyGuard(_reject_all_app)(scope, receive, send))
    status = next(m["status"] for m in received if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in received if m["type"] == "http.response.body"
    )
    return status, body


async def _reject_all_app(scope, receive, send):
    """Stand-in for the real app: reaching it at all means the guard let the
    request through, which is what the pass case asserts.
    """
    await send(
        {
            "type": "http.response.start",
            "status": 299,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"reached the app"})


def test_review_repro_non_loopback_peer_is_rejected():
    """Second review: the guard validated only the Host header, which
    defends browsers, not sockets. A non-browser client reaching the port
    (server bound beyond loopback by a --host typo or a published container
    port) could send "Host: localhost" and spend the key. The peer address
    must be loopback."""
    status, body = _call_guard_with_peer(("203.0.113.7", 54321))
    assert status == 403
    assert b"loopback" in body

    # IPv6 remote peers are refused on the same rule.
    status6, _ = _call_guard_with_peer(("2001:db8::1", 54321))
    assert status6 == 403


def test_loopback_peer_with_same_request_passes():
    """The control: identical request, loopback peer, reaches the app."""
    for peer in (("127.0.0.1", 54321), ("::1", 54321), ("127.0.0.5", 9)):
        status, body = _call_guard_with_peer(peer)
        assert status == 299, f"loopback peer {peer} was rejected"
        assert body == b"reached the app"


def test_in_process_transports_keep_working():
    """In-process transports have no socket to be remote. Starlette's
    TestClient presents ("testclient", 50000) and other harnesses present
    None or omit the key; all must pass, since only a real IP can carry a
    remote attacker. peer_is_local documents this carve-out."""
    for peer in (("testclient", 50000), None, (), ("", 0)):
        status, _ = _call_guard_with_peer(peer)
        assert status == 299, f"in-process peer {peer!r} was rejected"


def test_peer_is_local_classifies_addresses():
    """The predicate itself, including the loopback ranges beyond the
    canonical two."""
    assert main.peer_is_local(("127.0.0.1", 1)) is True
    assert main.peer_is_local(("127.13.99.4", 1)) is True
    assert main.peer_is_local(("::1", 1)) is True
    assert main.peer_is_local(None) is True
    assert main.peer_is_local(("testclient", 1)) is True
    assert main.peer_is_local(("10.0.0.4", 1)) is False
    assert main.peer_is_local(("203.0.113.7", 1)) is False
    assert main.peer_is_local(("2001:db8::1", 1)) is False


# ---- F4.5: a spend refusal is not a persistence failure.


@respx.mock
async def test_review_repro_refusal_frame_carries_spend_refused_marker(
    monkeypatch, tmp_path
):
    """Second review: every run_id null card claimed "not saved to history"
    with a tooltip saying persistence failed. The pre-upstream spend refusal
    deliberately persists nothing, so the UI described a working control as
    a failure. The frame now carries an explicit marker the client keys on,
    so a refusal and a genuine persistence failure stop being
    indistinguishable. Locked here because the client contract depends on
    the field name, and on the marker rather than the error prose."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        resp = await compare_stream(
            StreamCompareRequest(prompt="hi", model="model/alpha")
        )
        gen = resp.body_iterator
        c.app.state.accumulated_spend_usd = 1.5
        frames = [json.loads(f.removeprefix("data: ")) async for f in gen]

    assert [f["type"] for f in frames] == ["done"]
    assert frames[0]["run_id"] is None
    result = frames[0]["result"]
    # The marker, and the ceiling message that must reach the card.
    assert result["spend_refused"] is True
    assert "refused before reaching upstream" in result["error"]
    assert "$1.50" in result["error"] and "$1.00" in result["error"]


@respx.mock
def test_refusal_marker_absent_from_ordinary_results(monkeypatch, tmp_path):
    """The marker must mark only refusals. A genuine post-spend persistence
    failure keeps arriving as run_id null with no marker, which is what
    preserves its "not saved to history" warning.

    Asserted on the streaming frame, the contract the client actually reads.
    Asserting it on /compare would prove nothing: that endpoint serializes
    through ModelResult, which drops every field it does not declare, so the
    key would be absent there no matter what this function returned.
    """
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    # A ceiling is set but never approached: the run below stays far under
    # it, so this exercises the ordinary path while still letting the
    # refusal builder format its figures.
    with spend_client(monkeypatch, tmp_path, limit=1.0) as c:
        # Inside the client context: the refusal message reads app.state for
        # the accumulated and limit figures, which exist only once booted.
        refusal = main.spend_refusal_result("model/alpha", 16384)
        assert refusal["spend_refused"] is True
        frames = stream_events(c, {"prompt": "hi", "model": "model/alpha"})
    done = [f for f in frames if f["type"] == "done"]
    assert len(done) == 1
    # An ordinary successful run never carries the key, so a falsy read on
    # the client is the safe default for every other path.
    assert "spend_refused" not in done[0]["result"]
    assert done[0]["result"]["response_text"] is not None


# ---- F4.7: unknown request fields are refused, not silently ignored.


@respx.mock
def test_review_repro_unknown_field_rejected_on_every_mutating_endpoint(client):
    """Second review: the API silently ignored unknown JSON fields, so a
    typo like {"budgte": "extended"} ran a valid but unintended paid
    request. The caller asked for one thing and the bench did another, with
    the money moving either way. Every mutating endpoint must answer 422 and
    name the offending field, and the compare endpoints must not have
    reached upstream."""
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    cases = [
        ("/compare", {"prompt": "hi", "models": ["model/alpha"], "budgte": "extended"}),
        (
            "/compare/stream",
            {"prompt": "hi", "model": "model/alpha", "budgte": "extended"},
        ),
        ("/prompts", {"name": "n", "text": "t", "tags": ["oops"]}),
        ("/groups", {"unexpected": 1}),
    ]
    for path, body in cases:
        resp = client.post(path, json=body)
        assert resp.status_code == 422, (path, resp.status_code)
        # The offending field is named, so the mistake is obvious.
        detail = json.dumps(resp.json()["detail"])
        offender = next(k for k in body if k in ("budgte", "tags", "unexpected"))
        assert offender in detail, (path, detail)

    # No money moved for any of them: the refusal is at the boundary.
    assert route.call_count == 0


@respx.mock
def test_legitimate_fields_still_accepted(client):
    """The control: every field the frontend and the suite actually send
    stays valid, on both compare shapes and the two other bodies."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    # One group per tier: a group declares its budget now, so the extended
    # batch and the standard stream below belong to different comparisons.
    # Mixing them in one group is refused, which is a tombstone of its own.
    extended_gid = client.post("/groups", json={"budget": "extended"}).json()["id"]
    gid = client.post("/groups", json={"budget": "standard"}).json()["id"]
    pid = client.post("/prompts", json={"name": "f47", "text": "t"}).json()["id"]
    full = client.post(
        "/compare",
        json={
            "prompt": "hi",
            "models": ["model/alpha"],
            "prompt_id": pid,
            "group_id": extended_gid,
            "budget": "extended",
        },
    )
    assert full.status_code == 200
    stream = client.post(
        "/compare/stream",
        json={
            "prompt": "hi",
            "model": "model/alpha",
            "prompt_id": pid,
            "group_id": gid,
            "budget": "standard",
        },
    )
    assert stream.status_code == 200


# ---- F4.8: an offline catalog keeps a conservative completion cap.


def test_review_repro_offline_extended_clamps_to_standard(monkeypatch, tmp_path):
    """Second review: with the catalog offline there is no published limit,
    so effective_budget sent the full requested tier. That dropped the
    provider hard-400 protection and the runaway-size guard at exactly the
    moment spend is already unpriced, since an offline catalog has no
    prices either. An extended request on an offline boot must fall back to
    the standard tier, and the effective budget it was sent with must be
    what gets recorded and displayed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    async def offline_catalog(client):
        return {"fetched": False, "models": [], "prices": {}}

    monkeypatch.setattr("bench.main.fetch_catalog", offline_catalog)
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen[body["model"]] = body["max_tokens"]
        return httpx.Response(200, json=response_for(body["model"], "ok"))

    with TestClient(app, base_url="http://localhost") as c:
        with respx.mock:
            respx.post(OPENROUTER_URL).mock(side_effect=route)
            resp = c.post(
                "/compare",
                json={
                    "prompt": "hi",
                    "models": ["model/unknown"],
                    "budget": "extended",
                },
            )
        assert resp.status_code == 200
        # Sent conservatively, not at the full extended tier.
        assert seen["model/unknown"] == BUDGET_STANDARD
        # Recorded and returned as the effective budget, so the card is
        # honest about what was actually asked for.
        assert resp.json()["results"][0]["max_tokens"] == BUDGET_STANDARD
        run_id = resp.json()["run_id"]
        persisted = c.get(f"/runs/{run_id}").json()["results"][0]
        assert persisted["max_tokens"] == BUDGET_STANDARD


def test_offline_standard_request_is_unchanged(monkeypatch, tmp_path):
    """The clamp only ever lowers: a standard request on an offline boot
    still sends the standard tier."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    async def offline_catalog(client):
        return {"fetched": False, "models": [], "prices": {}}

    monkeypatch.setattr("bench.main.fetch_catalog", offline_catalog)
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen[body["model"]] = body["max_tokens"]
        return httpx.Response(200, json=response_for(body["model"], "ok"))

    with TestClient(app, base_url="http://localhost") as c:
        with respx.mock:
            respx.post(OPENROUTER_URL).mock(side_effect=route)
            c.post("/compare", json={"prompt": "hi", "models": ["model/unknown"]})
    assert seen["model/unknown"] == BUDGET_STANDARD


@respx.mock
def test_fetched_catalog_budgets_are_unchanged(client):
    """The scope control: with the catalog present, published caps behave
    exactly as before. A capped model still clamps to its published limit,
    and a catalogued model that publishes no cap still gets the full
    extended tier, since clamping those would quietly disable the extended
    tier for most of the catalog."""
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen[body["model"]] = body["max_tokens"]
        return httpx.Response(200, json=response_for(body["model"], "ok"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    client.post(
        "/compare",
        json={
            "prompt": "hi",
            "models": ["model/capped", "model/alpha"],
            "budget": "extended",
        },
    )
    assert seen["model/capped"] == 32000
    assert seen["model/alpha"] == BUDGET_EXTENDED


# ---- G1: the experiment record and provenance.


@respx.mock
def test_group_declares_prompt_and_lineup_before_any_member(client):
    """The group row is created before any upstream call, so recording the
    prompt and lineup there makes it the experiment record and fixes the
    group's prompt ahead of its first member."""
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = client.post(
        "/groups",
        json={
            "prompt": "declared",
            "models": ["model/alpha", "model/bare"],
            "budget": "standard",
        },
    ).json()["id"]

    # A different prompt is refused with no member present at all, which
    # the derive-from-first-member rule could never do.
    clash = client.post(
        "/compare",
        json={
            "prompt": "different",
            "models": ["model/alpha", "model/bare"],
            "group_id": gid,
        },
    )
    assert clash.status_code == 409
    assert route.call_count == 0
    # The declared prompt passes, sent as the declared lineup: a batch
    # joining a group now has to send that lineup in order, since the array
    # it sends is its claim about the comparison's shape.
    ok = client.post(
        "/compare",
        json={
            "prompt": "declared",
            "models": ["model/alpha", "model/bare"],
            "group_id": gid,
        },
    )
    assert ok.status_code == 200


def test_empty_group_body_still_accepted(client):
    """The empty JSON object is load-bearing for the CORS preflight
    defense and is what every pre-G client sends; it must stay valid."""
    assert client.post("/groups", json={"budget": "standard"}).status_code == 201


@respx.mock
def test_run_rows_carry_provenance_and_position(client):
    """Every run row records which build produced it and how old its price
    catalog was; every result records the column it occupied and the exact
    payload that was sent."""

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=response_for(body["model"], "ok"))

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    resp = client.post(
        "/compare",
        json={"prompt": "provenance", "models": ["model/alpha", "model/bare"]},
    )
    run_id = resp.json()["run_id"]
    detail = client.get(f"/runs/{run_id}").json()

    # catalog_snapshot_at is set because the test catalog reports fetched.
    assert detail["catalog_snapshot_at"] is not None
    # And beside it, which bytes those prices came from. The timestamp
    # alone cannot tell two boots against different catalogs apart.
    assert detail["catalog_digest"] == TEST_CATALOG["digest"]
    # app_sha is best-effort: a string when git answers, None otherwise,
    # and never a reason to fail a run.
    assert detail["app_sha"] is None or isinstance(detail["app_sha"], str)
    # Array order is the declared layout for the batch endpoint.
    assert [r["position"] for r in detail["results"]] == [0, 1]
    assert [r["model"] for r in detail["results"]] == ["model/alpha", "model/bare"]
    # The reproducibility record is the payload actually sent, and it
    # carries no authorization: auth lives in the client's headers.
    sent = json.loads(detail["results"][0]["request_json"])
    assert sent["model"] == "model/alpha"
    assert sent["max_tokens"] == BUDGET_STANDARD
    assert "authorization" not in json.dumps(sent).lower()


def test_review_repro_app_sha_marks_a_modified_tree(monkeypatch):
    """A bare sha claims the running code IS that commit. Before this fix
    _app_sha asked git only for rev-parse, so a checkout with uncommitted
    edits signed a clean commit's name and every run row born from
    modified code was provenance that pointed somewhere the code was not.
    """
    calls: list[list[str]] = []

    def fake_git(args: list[str]) -> str | None:
        calls.append(args)
        if args[0] == "rev-parse":
            return "abc1234\n"
        return " M bench/main.py\n?? scratch.py\n"

    monkeypatch.setattr(main, "_git", fake_git)
    assert main._app_sha() == "abc1234-dirty"
    # It asked the second question rather than inferring an answer.
    assert calls == [["rev-parse", "HEAD"], ["status", "--porcelain"]]


def test_app_sha_is_bare_on_a_clean_tree(monkeypatch):
    """Empty --porcelain output is the clean answer, and clean is exactly
    when the unqualified sha is true."""
    monkeypatch.setattr(
        main,
        "_git",
        lambda args: "abc1234\n" if args[0] == "rev-parse" else "",
    )
    assert main._app_sha() == "abc1234"


def test_app_sha_degrades_to_none_when_the_tree_cannot_be_read(monkeypatch):
    """An unqualified sha asserts a clean tree. When git answers the first
    question and not the second, that assertion is unverified, so the
    label is dropped rather than downgraded to the clean claim."""
    monkeypatch.setattr(
        main,
        "_git",
        lambda args: "abc1234\n" if args[0] == "rev-parse" else None,
    )
    assert main._app_sha() is None


def test_app_sha_degrades_to_none_without_git(monkeypatch):
    """No repository, no git, or git slower than the timeout: none of
    those are reasons to fail a boot."""
    monkeypatch.setattr(main, "_git", lambda args: None)
    assert main._app_sha() is None


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_the_documentation_routes_are_not_served(client, path):
    """The bench holds a paid API key and its only client is static/,
    written against these endpoints by hand. A machine-readable map of
    every route and bound is attack surface with no consumer here."""
    assert client.get(path).status_code == 404


@respx.mock
def test_stream_records_declared_position(client):
    """The frontend fans out one request per model, so only the client
    knows the layout; the declared position is what gets persisted."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    frames = stream_events(
        client, {"prompt": "positioned", "model": "model/alpha", "position": 3}
    )
    run_id = next(f for f in frames if f["type"] == "done")["run_id"]
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["results"][0]["position"] == 3


def test_position_is_bounded_without_capping_wide_lineups(client):
    """position is bounded at the boundary rather than trusted into the
    schema, but the bound is not the batch endpoint's five-model cap: the
    streaming path is one request per model and the browser lineup has no
    size limit, so a seven-model comparison must still declare positions
    five and six."""
    ok = client.post(
        "/compare/stream",
        json={"prompt": "hi", "model": "model/alpha", "position": 6},
    )
    assert ok.status_code == 200
    ok.read()
    for bad in (-1, MAX_POSITION + 1):
        resp = client.post(
            "/compare/stream",
            json={"prompt": "hi", "model": "model/alpha", "position": bad},
        )
        assert resp.status_code == 422, bad


# ---- G2: billed truth, read in-band.

BILLED_USAGE = {
    "prompt_tokens": 13,
    "completion_tokens": 8,
    "cost": 0.5,
    "completion_tokens_details": {"reasoning_tokens": 512},
    "prompt_tokens_details": {"cached_tokens": 9},
}


def billed_response(model: str) -> dict:
    body = json.loads(json.dumps(FIXTURE))
    body["model"] = model
    body["provider"] = "Fireworks"
    body["usage"] = BILLED_USAGE
    body["choices"][0]["native_finish_reason"] = "STOP_SEQUENCE"
    return body


def billed_stream():
    return ChunkStream(
        [
            sse({"provider": "Together", "choices": [{"delta": {"content": "Hi"}}]}),
            sse(
                {
                    "choices": [
                        {
                            "delta": {},
                            "finish_reason": "stop",
                            "native_finish_reason": "eos",
                        }
                    ]
                }
            ),
            sse({"choices": [], "usage": BILLED_USAGE}),
            DONE_MARKER,
        ]
    )


@respx.mock
def test_billed_truth_round_trips_into_history(client):
    """The authoritative numbers have to survive the write and the read,
    or the card is the only place they ever existed."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(
            200, json=billed_response(json.loads(request.content)["model"])
        )
    )

    body = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
    assert body.status_code == 200
    live = body.json()["results"][0]
    assert live["billed_cost_usd"] == 0.5
    assert live["provider"] == "Fireworks"

    detail = client.get(f"/runs/{body.json()['run_id']}").json()
    stored = detail["results"][0]
    assert stored["billed_cost_usd"] == 0.5
    assert stored["reasoning_tokens"] == 512
    assert stored["cached_tokens"] == 9
    assert stored["provider"] == "Fireworks"
    assert stored["native_finish_reason"] == "STOP_SEQUENCE"
    # The estimate keeps its own column and its own meaning: 13 prompt
    # tokens at 1e-6 plus 8 completion tokens at 2e-6.
    assert stored["cost_usd"] == pytest.approx(2.9e-5)


@respx.mock
def test_billed_truth_round_trips_into_history_from_the_stream(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=billed_stream())
    )

    frames = stream_events(client, {"prompt": "hi", "model": "model/alpha"})
    done = next(f for f in frames if f["type"] == "done")
    assert done["result"]["billed_cost_usd"] == 0.5
    assert done["result"]["provider"] == "Together"

    stored = client.get(f"/runs/{done['run_id']}").json()["results"][0]
    assert stored["billed_cost_usd"] == 0.5
    assert stored["reasoning_tokens"] == 512
    assert stored["provider"] == "Together"
    assert stored["native_finish_reason"] == "eos"


@respx.mock
def test_ceiling_counts_billed_cost_and_never_the_estimate_too(monkeypatch, tmp_path):
    """The ceiling is advisory, so it advises from the real charge when one
    exists. The other half matters as much: a result carrying both figures
    contributes exactly one of them, or a single run would be charged to
    the ceiling twice."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(
            200, json=billed_response(json.loads(request.content)["model"])
        )
    )
    with spend_client(monkeypatch, tmp_path, limit=10.0) as c:
        assert c.app.state.accumulated_spend_usd == 0.0
        c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        # The billed 0.50 alone: not the 2.9e-5 estimate, and not the sum.
        assert c.app.state.accumulated_spend_usd == 0.5


@respx.mock
def test_ceiling_falls_back_to_the_estimate_when_nothing_was_billed(
    monkeypatch, tmp_path
):
    """Removing the estimate from the ceiling would leave every provider
    that omits cost invisible to it."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(
            200, json=response_for(json.loads(request.content)["model"], "hello")
        )
    )
    with spend_client(monkeypatch, tmp_path, limit=10.0) as c:
        c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        assert c.app.state.accumulated_spend_usd == pytest.approx(2.9e-5)


@respx.mock
def test_poisoned_billed_cost_never_reaches_the_accumulator(monkeypatch, tmp_path):
    """The money rule end to end, through the real endpoint. A NaN summed
    into accumulated spend makes every ceiling comparison false and
    silently disables the ceiling for the life of the process; the run must
    still succeed, and the ceiling must still work afterwards.

    Not named as a tombstone, because it does not fail on naive ingestion:
    record_spend's own finiteness guard (F.1) catches a NaN that got past
    as_money, and Pydantic serializes a NaN float to null on the way out.
    Those are the belts, and this locks that all of them together hold. The
    tombstones for as_money itself are in test_models.py, where the field
    value is observable before either belt applies."""
    # Encoded by hand: json.dumps writes a bare NaN token by default, which
    # httpx's json= helper refuses to produce and which is exactly what a
    # misbehaving provider puts on the wire.
    body = json.dumps({**FIXTURE, "usage": {**FIXTURE["usage"], "cost": float("nan")}})
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )
    )
    with spend_client(monkeypatch, tmp_path, limit=10.0) as c:
        resp = c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["error"] is None
        assert result["billed_cost_usd"] is None
        total = c.app.state.accumulated_spend_usd
        assert math.isfinite(total)
        # The ceiling still answers, which a NaN total would have made
        # permanently false.
        c.app.state.accumulated_spend_usd = 20.0
        assert main.spend_ceiling_reached() is True
        c.app.state.accumulated_spend_usd = total


# ---- G3: the generation id survives everything, including aborts.


@respx.mock
async def test_review_repro_aborted_stream_persists_its_generation_id(client):
    """A run stopped mid-stream is the run whose billing is least knowable
    locally (stream cancellation only stops the charge on providers that
    support it, and throughput routing picks the provider per request), so
    it is the run that most needs an id to reconcile against later. The id
    used to be read out of a chunk and dropped on the way to the disconnect
    path, which left exactly those runs unauditable.
    """
    from bench.main import StreamCompareRequest, compare_stream
    from bench.models import GENERATION_ID_HEADER

    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            headers={GENERATION_ID_HEADER: "gen-aborted-run"},
            stream=alpha_stream(),
        )
    )

    resp = await compare_stream(
        StreamCompareRequest(prompt="abort me", model="model/alpha")
    )
    gen = resp.body_iterator
    assert (
        json.loads((await gen.__anext__()).removeprefix("data: "))["type"] == "started"
    )
    assert json.loads((await gen.__anext__()).removeprefix("data: "))["type"] == "delta"
    # aclose() raises GeneratorExit at the yield, the same mechanism a
    # Starlette client disconnect triggers.
    await gen.aclose()

    entries = client.get("/runs").json()["runs"]
    persisted = client.get(f"/runs/{entries[0]['id']}").json()["results"][0]
    assert persisted["error"] == "stream aborted before completion"
    assert persisted["generation_id"] == "gen-aborted-run"
    # The reproducibility record too. models.py promises request_json is
    # echoed "on failure paths, because knowing what a failed request asked
    # for is exactly when this matters", and the disconnect path is a
    # failure path. It used to be the one path that dropped it, on exactly
    # the rows the reconcile pass is built to visit.
    sent = json.loads(persisted["request_json"])
    assert sent["model"] == "model/alpha"
    assert sent["max_tokens"] == 16384
    assert "authorization" not in json.dumps(sent).lower()
    # A run that did not finish reports no finish reason: any value there
    # would be a claim the server cannot make.
    assert persisted["finish_reason"] is None
    assert persisted["native_finish_reason"] is None


@respx.mock
async def test_a_run_cancelled_while_queued_persists_nothing(monkeypatch, tmp_path):
    """A run cancelled while still waiting for a slot reaches upstream
    never, holds no slot, and persists nothing. The holder cannot invent an
    id for it because no response ever arrived.

    This test used to call aclose() on a generator it had never advanced,
    which runs none of the generator's body: no slot was awaited, no frame
    was emitted, the finally never ran, and "nothing was persisted" held
    because nothing had executed. It read as proof of the queued-cancel
    invariant while proving only that Python does not start a generator you
    do not iterate. Advancing to the queued frame first puts the generator
    at a real suspension point inside events(), which is where a browser
    Stop lands it.
    """
    from bench.main import StreamCompareRequest, compare_stream

    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with make_client(monkeypatch, tmp_path, 1) as c:
        # Hold the only slot by hand so the run must queue for it.
        await c.app.state.upstream_semaphore.acquire()
        resp = await compare_stream(
            StreamCompareRequest(prompt="queued cancel", model="model/alpha")
        )
        gen = resp.body_iterator

        first = json.loads((await gen.__anext__()).removeprefix("data: "))
        assert first["type"] == "queued"
        free_slots = c.app.state.upstream_semaphore._value

        # GeneratorExit at the queued yield, the same mechanism a client
        # disconnect triggers, with the generator suspended before it ever
        # acquires a slot.
        await gen.aclose()

        assert route.call_count == 0, "a queued cancel must not reach upstream"
        assert c.get("/runs").json()["runs"] == []
        # The slot it never took is still free: an aborted queue wait must
        # not release a slot it did not hold.
        assert c.app.state.upstream_semaphore._value == free_slots
        c.app.state.upstream_semaphore.release()


# ---- G5: privacy routing.


def policy_client(monkeypatch, tmp_path, policy=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))
    if policy is None:
        monkeypatch.delenv("BENCH_DATA_POLICY", raising=False)
    else:
        monkeypatch.setenv("BENCH_DATA_POLICY", policy)

    async def fake_fetch_catalog(client):
        return json.loads(json.dumps(TEST_CATALOG))

    monkeypatch.setattr("bench.main.fetch_catalog", fake_fetch_catalog)
    return TestClient(app, base_url="http://localhost")


@pytest.mark.parametrize("bad", ["strict", "DENY ZDR", "true", "none"])
def test_an_unknown_data_policy_fails_boot(monkeypatch, tmp_path, bad):
    """A typo must not fall back to standard. The fallback would send
    prompts to training-eligible providers while the operator believed
    otherwise, and nothing after the fact would show it."""
    with (
        pytest.raises(RuntimeError, match="BENCH_DATA_POLICY"),
        policy_client(monkeypatch, tmp_path, bad),
    ):
        pass


@pytest.mark.parametrize(
    ("env", "expected"),
    [(None, "standard"), ("standard", "standard"), ("deny", "deny"), ("zdr", "zdr")],
)
def test_the_boot_policy_is_reported_to_the_page(monkeypatch, tmp_path, env, expected):
    with policy_client(monkeypatch, tmp_path, env) as c:
        assert c.get("/models").json()["data_policy"] == expected


@pytest.mark.parametrize("policy", ["standard", "deny", "zdr"])
@respx.mock
def test_the_run_row_records_the_policy_that_was_actually_sent(
    monkeypatch, tmp_path, policy
):
    """The column has to match the wire, not the intent: a row claiming a
    policy the payload did not carry is worse than no column at all."""
    route = respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(
            200, json=response_for(json.loads(request.content)["model"], "hi")
        )
    )
    with policy_client(monkeypatch, tmp_path, policy) as c:
        body = c.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})
        run_id = body.json()["run_id"]

        sent = json.loads(route.calls[0].request.content)
        assert sent["provider"] == provider_preferences(policy)

        row = c.app.state.db.execute(
            "SELECT data_policy FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["data_policy"] == policy


@respx.mock
def test_the_streaming_path_sends_the_policy_too(monkeypatch, tmp_path):
    """Two endpoints, one policy. The streaming path is the one the browser
    uses, so a policy that only reached /compare would be inert in
    practice."""
    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    with policy_client(monkeypatch, tmp_path, "zdr") as c:
        frames = stream_events(c, {"prompt": "hi", "model": "model/alpha"})
        run_id = next(f for f in frames if f["type"] == "done")["run_id"]

        sent = json.loads(route.calls[0].request.content)
        assert sent["provider"] == provider_preferences("zdr")
        row = c.app.state.db.execute(
            "SELECT data_policy FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["data_policy"] == "zdr"


# ---- Phase G closing review.


def test_review_repro_wide_lineup_still_gets_a_group(client):
    """Closing review: the group POST borrowed CompareRequest's five-model
    cap, so a comparison wider than five was rejected. A failed group
    create degrades to ungrouped runs rather than erroring, so those
    comparisons silently lost their single history entry, which is exactly
    what the group row exists to provide."""
    models = [f"model/m{i}" for i in range(7)]

    resp = client.post(
        "/groups", json={"prompt": "wide", "models": models, "budget": "standard"}
    )

    assert resp.status_code == 201
    group_id = resp.json()["id"]
    row = client.app.state.db.execute(
        "SELECT models_json FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    assert json.loads(row["models_json"]) == models


def test_the_group_body_is_still_bounded(client):
    """Bounded, just not at five. An absurd body is still refused at the
    boundary rather than written into the schema."""
    huge = [f"model/m{i}" for i in range(MAX_POSITION + 2)]
    resp = client.post(
        "/groups", json={"prompt": "absurd", "models": huge, "budget": "standard"}
    )
    assert resp.status_code == 422


@respx.mock
def test_poisoned_provenance_never_breaks_a_paid_result(client):
    """Closing review sweep: every new capture step, poisoned in turn. The
    money was already spent by the time any of these are read, and the
    post-spend invariant forbids turning that into a failure, so each one
    must degrade to None and leave the response successful."""
    poisoned = json.loads(json.dumps(FIXTURE))
    poisoned["provider"] = {"name": "Fireworks"}
    poisoned["choices"][0]["native_finish_reason"] = 7
    poisoned["usage"] = {
        "prompt_tokens": 13,
        "completion_tokens": 8,
        "completion_tokens_details": "n/a",
        "prompt_tokens_details": {"cached_tokens": "many"},
    }
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            content=json.dumps(poisoned),
            headers={"content-type": "application/json"},
        )
    )

    resp = client.post("/compare", json={"prompt": "hi", "models": ["model/alpha"]})

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["error"] is None
    assert result["response_text"] is not None
    for field in ("provider", "native_finish_reason", "reasoning_tokens"):
        assert result[field] is None, field
    # And the row that was written says the same thing on the way back.
    stored = client.get(f"/runs/{resp.json()['run_id']}").json()["results"][0]
    assert stored["provider"] is None
    assert stored["native_finish_reason"] is None


# ---- Hotfix H1: the stale-asset skew becomes impossible.


def test_review_repro_static_assets_must_revalidate(client):
    """Field incident, diagnosed externally and verified by reproduction.

    Static assets went out with ETag and Last-Modified but no
    Cache-Control, so a browser was free to apply heuristic freshness and
    reuse a cached file without asking. After the Phase G upgrade that
    produced a page running fresh modules beside a stale pre-G lib.js with
    no fmtBilled, and every card finishing after the session went non-idle
    stranded silently at "thinking" with its answer visible.

    no-cache does not mean "do not store", it means "revalidate before
    reuse", so with the ETags already there the usual cost is a 304 and
    the skew cannot happen.
    """
    for path in ("/", "/static/lib.js", "/static/volt.css"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers.get("cache-control") == "no-cache", path


def test_a_response_that_chose_its_own_caching_keeps_it(client):
    """The injection is only-if-absent, and that is load-bearing rather
    than defensive: the favicon declares a year of immutable caching on
    purpose, and a second Cache-Control on that response would be a defect
    rather than a stricter rule."""
    resp = client.get("/favicon.ico")

    assert resp.status_code == 200
    values = resp.headers.get_list("cache-control")
    assert len(values) == 1, values
    assert values[0] == "public, max-age=31536000, immutable"


def test_every_response_carries_exactly_one_cache_control(client):
    """Duplicate directives are ambiguous to a cache, so the rule is one
    header everywhere, whoever set it."""
    for path in ("/", "/static/lib.js", "/favicon.ico", "/models", "/runs"):
        resp = client.get(path)
        assert len(resp.headers.get_list("cache-control")) == 1, path


@respx.mock
def test_review_repro_no_response_carrying_a_prompt_may_be_stored(client):
    """One directive was applied to every response class alike.

    no-cache means "revalidate before reuse", not "do not store": a
    no-cache response may sit on a browser's disk indefinitely. That is
    right for assets and it is what closed the stale-module skew. It was
    wrong for everything else. Run details, group details and compare
    responses carry the full prompt and the full model answer, and this
    application's own documentation promises those go to OpenRouter and
    nowhere else. The header said they could be written to disk by any
    cache on the path.

    The matrix below walks one response of every class the app produces,
    so a future response class cannot quietly inherit the asset rule.
    """
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, json=response_for("model/alpha", "secret"))
    )
    compare = client.post(
        "/compare", json={"prompt": "secret", "models": ["model/alpha"]}
    )
    run_id = compare.json()["run_id"]
    group = client.post("/groups", json={"budget": "standard"})
    group_id = group.json()["id"]
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )

    # Assets revalidate: they hold no prompt, and a stale one is the skew.
    revalidate = ("/", "/static/lib.js", "/static/volt.css")
    # Everything else is not stored at all, including the errors, since a
    # 404 body for /runs/999999 is still an answer about this bench.
    never_stored = (
        f"/runs/{run_id}",
        f"/groups/{group_id}",
        "/runs",
        "/groups",
        "/models",
        "/runs/999999",
    )

    for path in revalidate:
        assert client.get(path).headers.get("cache-control") == "no-cache", path
    for path in never_stored:
        resp = client.get(path)
        assert resp.headers.get("cache-control") == "private, no-store", path
        assert len(resp.headers.get_list("cache-control")) == 1, path

    # The POST responses carry bodies too, and a compare response is the
    # single largest concentration of prompt and answer in the app.
    for resp in (compare, group):
        assert resp.headers.get("cache-control") == "private, no-store"

    # The SSE stream is the other path a full answer leaves by.
    with client.stream(
        "POST", "/compare/stream", json={"prompt": "secret", "model": "model/alpha"}
    ) as stream:
        assert stream.headers.get("cache-control") == "private, no-store"
        stream.read()

    # The favicon set its own and keeps it: it is the one response whose
    # bytes are public and unchanging.
    assert (
        client.get("/favicon.ico").headers.get("cache-control")
        == "public, max-age=31536000, immutable"
    )


def test_the_served_index_versions_every_asset_url(client):
    """Belt to the Cache-Control braces: a changed asset gets a URL the
    browser has never seen, so even a cache ignoring every header cannot
    serve the old bytes for it."""
    body = client.get("/").text

    referenced = re.findall(r'(?:src|href)="(/static/[^"]*)"', body)
    # Every module in the load-bearing order, plus the stylesheet and the
    # pre-paint theme script. The count moves when a module is added, and
    # it is asserted rather than derived on purpose: a new asset that the
    # transform failed to version would otherwise pass unnoticed, since
    # every OTHER url would still carry its rev.
    assert len(referenced) == 13, referenced
    for url in referenced:
        assert f"?v={main.STATIC_REV}" in url, url
    # The committed file itself keeps plain URLs, so opening it straight
    # off disk still works.
    assert "?v=" not in main.INDEX_HTML.read_text(encoding="utf-8")


def test_the_asset_rev_is_stable_within_a_boot(client):
    """A rev that moved between requests would version the modules of one
    page against each other, which is the skew rather than the fix."""
    first = re.findall(r"\?v=([a-f0-9]+)", client.get("/").text)
    second = re.findall(r"\?v=([a-f0-9]+)", client.get("/").text)

    assert first == second
    assert len(set(first)) == 1, first


def test_the_asset_rev_changes_when_the_bytes_do(tmp_path):
    """Content-derived, so it changes exactly when an asset changes: not
    on every commit, and not never."""
    (tmp_path / "a.js").write_text("one")
    before = main.static_rev(tmp_path)

    (tmp_path / "a.js").write_text("two")
    assert main.static_rev(tmp_path) != before

    # Stable when nothing changed, or it would bust caches every boot.
    assert main.static_rev(tmp_path) == main.static_rev(tmp_path)


def test_the_asset_rev_notices_a_rename(tmp_path):
    """Paths are mixed into the digest, so moving bytes between filenames
    is a change even though the bytes are not."""
    (tmp_path / "a.js").write_text("same")
    before = main.static_rev(tmp_path)

    (tmp_path / "a.js").unlink()
    (tmp_path / "b.js").write_text("same")

    assert main.static_rev(tmp_path) != before


def test_versioning_appends_to_a_url_that_already_has_a_query():
    """No URL in the index carries one today; the rewrite must not turn
    the first one that does into nonsense."""
    out = main.versioned_index('<script src="/static/x.js?a=b"></script>', "abc")

    assert out == '<script src="/static/x.js?a=b&v=abc"></script>'


def test_versioning_leaves_foreign_urls_alone():
    """Narrow on purpose: only same-origin /static URLs are ours to
    version, and the CSP forbids the rest anyway."""
    html = '<script src="https://cdn.example/x.js"></script><a href="/runs">r</a>'

    assert main.versioned_index(html, "abc") == html


# ---- Phase H1: experiment controls at the boundary.

# Every control set to a distinguishable value, so a test can tell a
# dropped one from a defaulted one.
ALL_CONTROLS = {
    "system": "be terse",
    "temperature": 0.25,
    "top_p": 0.9,
    "seed": 7,
    "effort": "high",
    "routing": "price",
}


@respx.mock
def test_review_repro_group_rejects_a_second_experiment(client):
    """Both external reviews named uncontrolled sampling and routing as the
    confound standing between "outputs I received" and "a comparison I can
    defend". Controls alone do not remove it: a group whose members were
    sent different temperatures is exactly the uncontrolled comparison
    wearing a controlled label, and it would look defensible in history.

    So one prompt per group became one experiment per group, checked at
    entry, pre-spend, like the prompt check beside it. The 409 names the
    controls that conflict, and respx proves no upstream call was made,
    because a rejection after spend would violate the fault boundary.
    """
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = client.post(
        "/groups", json={"params": ALL_CONTROLS, "budget": "standard"}
    ).json()["id"]
    body = {"prompt": "p", "models": ["model/alpha"], "group_id": gid}

    first = client.post("/compare", json={**body, "params": ALL_CONTROLS})
    assert first.status_code == 200
    calls = route.call_count

    # One control changed is a different experiment.
    clash = client.post(
        "/compare",
        json={**body, "params": {**ALL_CONTROLS, "temperature": 1.5}},
    )
    assert clash.status_code == 409
    assert "temperature" in clash.json()["detail"]
    assert "one controls set" in clash.json()["detail"]
    assert route.call_count == calls, "the refusal must spend nothing"

    # Dropping a control is also a different experiment: absence is a
    # choice under rule one, not a partial match.
    dropped = dict(ALL_CONTROLS)
    del dropped["seed"]
    missing = client.post("/compare", json={**body, "params": dropped})
    assert missing.status_code == 409
    assert "seed" in missing.json()["detail"]
    assert route.call_count == calls

    # Sending no controls at all against a group that has them is refused
    # too, and names every one of them.
    none_at_all = client.post("/compare", json=body)
    assert none_at_all.status_code == 409
    for name in ALL_CONTROLS:
        assert name in none_at_all.json()["detail"], name
    assert route.call_count == calls

    # The same experiment still passes: a rerun or a second model.
    again = client.post("/compare", json={**body, "params": ALL_CONTROLS})
    assert again.status_code == 200
    assert route.call_count == calls + 1


def test_the_stream_endpoint_enforces_the_same_experiment_check(client):
    """The check lives at both entries or it lives nowhere: the frontend
    fans out one stream request per model, so the streaming path is the one
    a real divergence would travel."""
    gid = client.post(
        "/groups", json={"params": {"temperature": 0.25}, "budget": "standard"}
    ).json()["id"]

    with respx.mock:
        route = respx.post(OPENROUTER_URL)
        clash = client.post(
            "/compare/stream",
            json={
                "prompt": "p",
                "model": "model/alpha",
                "group_id": gid,
                "params": {"temperature": 0.9},
            },
        )
        assert clash.status_code == 409
        assert "temperature" in clash.json()["detail"]
        assert route.call_count == 0


@respx.mock
def test_a_group_with_no_controls_accepts_a_run_with_no_controls(client):
    """Every pre-H group and every pre-H client sends nothing, and nothing
    matches nothing. This is the compatibility floor: the check must not
    turn existing traffic into 409s."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = client.post("/groups", json={"prompt": "p", "budget": "standard"}).json()[
        "id"
    ]

    resp = client.post(
        "/compare", json={"prompt": "p", "models": ["model/alpha"], "group_id": gid}
    )

    assert resp.status_code == 200


def test_a_group_with_no_controls_rejects_a_run_that_carries_one(client):
    """NULL params is not missing information, so there is no
    derive-from-member fallback here and this is what that buys. A group
    recorded as uncontrolled whose member sent a temperature would be a
    record that understates what it sent, which is the truth defect the
    whole phase exists to prevent."""
    with respx.mock:
        route = respx.post(OPENROUTER_URL)
        gid = client.post("/groups", json={"prompt": "p", "budget": "standard"}).json()[
            "id"
        ]

        resp = client.post(
            "/compare",
            json={
                "prompt": "p",
                "models": ["model/alpha"],
                "group_id": gid,
                "params": {"seed": 1},
            },
        )

        assert resp.status_code == 409
        assert "seed" in resp.json()["detail"]
        assert route.call_count == 0


def test_the_group_records_exactly_the_controls_that_were_set(client):
    """Rule two: params_json stores what the user set and nothing else. No
    echoing of tool defaults, and no normalizing an absent control to null,
    because a history badge that rendered a default as a choice would be a
    truth defect."""
    gid = client.post(
        "/groups",
        json={
            "prompt": "p",
            "params": {"temperature": 0.25, "seed": 7},
            "budget": "standard",
        },
    ).json()["id"]

    stored = client.get(f"/groups/{gid}").json()["params"]

    assert stored == {"temperature": 0.25, "seed": 7}


def test_a_group_created_without_controls_records_none_not_an_empty_object(client):
    """One spelling of absence. An empty object would be a second, and
    every reader would then have to know both."""
    gid = client.post("/groups", json={"prompt": "p", "budget": "standard"}).json()[
        "id"
    ]
    assert client.get(f"/groups/{gid}").json()["params"] is None

    # An explicitly empty params object collapses to the same absence
    # rather than storing a record of nothing.
    gid2 = client.post(
        "/groups", json={"prompt": "p", "params": {}, "budget": "standard"}
    ).json()["id"]
    assert client.get(f"/groups/{gid2}").json()["params"] is None


@pytest.mark.parametrize(
    "params,field",
    [
        ({"temperature": 2.01}, "temperature"),
        ({"temperature": -0.01}, "temperature"),
        ({"top_p": 1.01}, "top_p"),
        ({"top_p": -0.01}, "top_p"),
        ({"effort": "maximum"}, "effort"),
        ({"routing": "cheapest"}, "routing"),
        ({"system": ""}, "system"),
        ({"nope": 1}, "nope"),
    ],
)
@respx.mock
def test_both_compare_surfaces_reject_the_same_bad_controls(client, params, field):
    """Parity is structural (one ExperimentParams class, three references)
    but the point of parity is that a scripted caller and the browser
    cannot diverge, so it is asserted on both surfaces rather than assumed
    from the class.

    The empty system prompt is in this list deliberately: "" is not a
    system prompt the user set, it is a blank control, and accepting it
    would put a meaningless leading system message on the wire.
    """
    batch = client.post(
        "/compare", json={"prompt": "p", "models": ["model/alpha"], "params": params}
    )
    stream = client.post(
        "/compare/stream",
        json={"prompt": "p", "model": "model/alpha", "params": params},
    )
    group = client.post(
        "/groups", json={"prompt": "p", "params": params, "budget": "standard"}
    )

    assert batch.status_code == 422, field
    assert stream.status_code == 422, field
    assert group.status_code == 422, field


@respx.mock
def test_both_compare_surfaces_accept_the_same_full_controls_set(client):
    """The other half of parity: what one takes, the other takes."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    batch = client.post(
        "/compare",
        json={"prompt": "p", "models": ["model/alpha"], "params": ALL_CONTROLS},
    )

    assert batch.status_code == 200
    body = json.loads(respx.calls[0].request.content)
    assert body["temperature"] == 0.25
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    # Routing reached the provider object, and the boot privacy policy is
    # still riding it.
    assert body["provider"] == {"sort": "price"}


@respx.mock
def test_blank_controls_leave_the_boot_provider_block_untouched(client):
    """The routing control must not become a way to lose the privacy
    policy, and the no-routing path must not rebuild the block at all.

    The key-absence assertions are the endpoint-level half of rule one, and
    they are enumerated rather than spot-checked: an earlier draft asserted
    only the provider block and the messages, and a deliberately injected
    temperature default sailed straight through it while the payload
    tombstone in test_models caught it. Absence is what this test is for,
    so it has to name every key that must be absent.
    """
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    client.post("/compare", json={"prompt": "p", "models": ["model/alpha"]})

    body = json.loads(respx.calls[0].request.content)
    assert body["provider"] == {"sort": "throughput"}
    assert body["messages"] == [{"role": "user", "content": "p"}]
    for control_key in ("temperature", "top_p", "seed", "reasoning"):
        assert control_key not in body, control_key
    # The whole key set, so a control added later cannot slip in unnoticed
    # by being absent from the list above.
    assert set(body) == {"model", "messages", "max_tokens", "provider"}


@respx.mock
def test_a_set_control_reaches_the_stored_request_json(client):
    """request_json is the reproducibility record, so the controls have to
    be visible in the row and not only on the wire."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    run_id = client.post(
        "/compare",
        json={"prompt": "p", "models": ["model/alpha"], "params": {"seed": 7}},
    ).json()["run_id"]

    recorded = json.loads(
        client.get(f"/runs/{run_id}").json()["results"][0]["request_json"]
    )
    assert recorded["seed"] == 7
    assert "temperature" not in recorded


@respx.mock
def test_group_controls_compare_as_parsed_structures_not_as_strings(client):
    """The one-experiment check compares what the controls MEAN, never how
    they happened to serialize. Two ways that distinction bites, both here:

    Key order. JSON objects are unordered, so a client is free to send the
    same controls in any order, and two different orders are the same
    experiment. The stored copy is written with sorted keys for exactly this
    reason, but sorting the stored side is not sufficient on its own; a
    string comparison would still break against a shuffled request. The
    comparison runs on parsed dicts, so order cannot reach it.

    Numeric spelling. A client that sends temperature as the integer 1 means
    the same thing as one that sends 1.0, and both surfaces coerce to float
    at the boundary before anything is stored or compared. A string
    comparison would see "1" against "1.0" and refuse a run that matches.

    Either failure would be a 409 on a run that genuinely belongs to the
    group, which is worse than useless: it blocks correct work while
    claiming to protect the record.
    """
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    # Written one way, in an order that is not sorted, with an integer.
    gid = client.post(
        "/groups",
        json={
            "prompt": "p",
            "params": {
                "top_p": 0.9,
                "temperature": 1,
                "system": "be terse",
                "seed": 7,
                "routing": "price",
                "effort": "high",
            },
            "budget": "standard",
        },
    ).json()["id"]

    # Sent back a different way: shuffled, and temperature still an integer.
    shuffled = client.post(
        "/compare",
        json={
            "prompt": "p",
            "models": ["model/alpha"],
            "group_id": gid,
            "params": {
                "seed": 7,
                "effort": "high",
                "temperature": 1,
                "routing": "price",
                "system": "be terse",
                "top_p": 0.9,
            },
        },
    )
    assert shuffled.status_code == 200, shuffled.json()

    # And once more with temperature spelled as a float, which is the same
    # experiment by any reading.
    as_float = client.post(
        "/compare/stream",
        json={
            "prompt": "p",
            "model": "model/alpha",
            "group_id": gid,
            "params": {
                "system": "be terse",
                "temperature": 1.0,
                "top_p": 0.9,
                "seed": 7,
                "effort": "high",
                "routing": "price",
            },
        },
    )
    assert as_float.status_code == 200, as_float.read()

    # A genuine change is still refused, so the test above is not passing
    # by accident of the check being disabled.
    real_clash = client.post(
        "/compare",
        json={
            "prompt": "p",
            "models": ["model/alpha"],
            "group_id": gid,
            "params": {
                "system": "be terse",
                "temperature": 1,
                "top_p": 0.9,
                "seed": 8,
                "effort": "high",
                "routing": "price",
            },
        },
    )
    assert real_clash.status_code == 409
    assert "seed" in real_clash.json()["detail"]


@respx.mock
def test_a_legacy_group_says_why_it_cannot_take_controls(client):
    """A pre-H group records no controls, so "controls do not match" would
    be a lie about what happened and useless about what to do: there is
    nothing to match against. The detail names the group's state and the
    way forward, because the user's real question at that moment is not
    whether the request was refused."""
    with respx.mock:
        route = respx.post(OPENROUTER_URL)
        gid = client.post("/groups", json={"prompt": "p", "budget": "standard"}).json()[
            "id"
        ]

        resp = client.post(
            "/compare",
            json={
                "prompt": "p",
                "models": ["model/alpha"],
                "group_id": gid,
                "params": {"temperature": 0.2, "seed": 7},
            },
        )

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        # What the group is, which controls were refused, and what to do.
        assert "records no controls" in detail
        assert "temperature" in detail and "seed" in detail
        assert "start a new comparison" in detail
        assert route.call_count == 0

    # The reverse direction keeps the other message: there the group does
    # hold an experiment, so "does not match" is the accurate wording.
    with respx.mock:
        respx.post(OPENROUTER_URL).respond(json=FIXTURE)
        controlled = client.post(
            "/groups", json={"prompt": "q", "params": {"seed": 1}, "budget": "standard"}
        ).json()["id"]
        bare = client.post(
            "/compare",
            json={"prompt": "q", "models": ["model/alpha"], "group_id": controlled},
        )

        assert bare.status_code == 409
        assert "do not match" in bare.json()["detail"]


# ---- Phase H2 follow-through: the markup's bounds are the model's bounds.

# Which control maps to which input, and which HTML attributes carry its
# bound. The browser enforces these client side via checkValidity, which is
# what lets the composer disable Run instead of letting a bad value 422 once
# per model. That convenience duplicates the API's bounds in a second
# language, and a duplicate that drifts LOW would wrongly block a run the
# API would have accepted, which is the worse direction: the user is stopped
# from doing something legal and told it is out of range.
_MARKUP_BOUNDS = {
    "temperature": "ctl-temperature",
    "top_p": "ctl-top-p",
    "seed": "ctl-seed",
}
_MARKUP_OPTIONS = {"effort": "ctl-effort", "routing": "ctl-routing"}


def _input_attrs(html: str, testid: str) -> dict[str, str]:
    tag = re.search(
        rf'<(?:input|select|textarea)[^>]*data-testid="{testid}"[^>]*>', html
    )
    assert tag is not None, testid
    return dict(re.findall(r'(\w[\w-]*)="([^"]*)"', tag.group(0)))


def _select_values(html: str, testid: str) -> list[str]:
    block = re.search(rf'<select[^>]*data-testid="{testid}".*?</select>', html, re.S)
    assert block is not None, testid
    return re.findall(r'<option value="([^"]*)"', block.group(0))


def _model_bound(field: str, kind: str):
    for constraint in main.ExperimentParams.model_fields[field].metadata:
        if type(constraint).__name__ == kind:
            return getattr(constraint, kind.lower())
    return None


def test_the_markup_bounds_match_the_shared_request_model():
    """The client-side range guard is a convenience, not an authority: the
    API's ExperimentParams is the single source of truth for every bound,
    and the markup's min and max only exist so the composer can refuse a
    value before it costs one request per model.

    Two languages holding one rule is a drift hazard, so it is asserted
    rather than reviewed. The direction that matters most is a markup bound
    TIGHTER than the model's: it would block a run the API would have
    accepted, and tell the user their legal value is out of range. A looser
    markup bound is merely a 422 the composer failed to pre-empt.
    """
    html = (
        Path(main.__file__).resolve().parent.parent / "static/index.html"
    ).read_text()

    for field, testid in _MARKUP_BOUNDS.items():
        attrs = _input_attrs(html, testid)
        for kind, attr in (("Ge", "min"), ("Le", "max")):
            model_value = _model_bound(field, kind)
            assert attr in attrs, f"{field} lost its {attr}"
            assert float(attrs[attr]) == float(model_value), (
                f"{field}: markup {attr}={attrs[attr]} but model says {model_value}"
            )

    for field, testid in _MARKUP_OPTIONS.items():
        allowed = typing.get_args(
            next(
                a
                for a in typing.get_args(
                    main.ExperimentParams.model_fields[field].annotation
                )
                if typing.get_origin(a) is Literal
            )
        )
        offered = _select_values(html, testid)
        # The empty option is the unset control, which is an absence rather
        # than a value the model accepts; everything else must be a value
        # the API would take.
        assert offered[0] == "", f"{field} must offer unset first"
        assert tuple(offered[1:]) == allowed, (
            f"{field}: markup offers {offered[1:]} but model accepts {allowed}"
        )


def test_every_control_the_model_accepts_has_an_input(client):
    """Drift in the other direction: a control added to the API with no way
    to set it in the browser would be a scripting-only feature nobody
    discovers, and one removed from the markup would silently stop being
    holdable."""
    html = (
        Path(main.__file__).resolve().parent.parent / "static/index.html"
    ).read_text()
    ids = {
        "system": "ctl-system",
        **_MARKUP_BOUNDS,
        **_MARKUP_OPTIONS,
    }

    assert set(ids) == set(main.ExperimentParams.model_fields), (
        "a control gained or lost an input; update the markup and this map"
    )
    for testid in ids.values():
        assert f'data-testid="{testid}"' in html, testid


# ---- Phase H closing review: blank-control paths above the byte floor.


@respx.mock
def test_an_explicitly_null_control_is_blank_not_set_to_null(client):
    """Closing review, hunting above the byte-identical floor. A client can
    spell a blank control as an explicit null rather than by omission, and
    the two have to mean the same thing everywhere: in the payload, in the
    stored record, and in the conflict check. If null survived as a set
    value the group would record a control nobody chose, and the badge for
    it would be the truth defect rule two exists to prevent."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    resp = client.post(
        "/compare",
        json={
            "prompt": "p",
            "models": ["model/alpha"],
            "params": {"temperature": None, "seed": None, "system": None},
        },
    )

    assert resp.status_code == 200
    body = json.loads(respx.calls[0].request.content)
    # Byte-for-byte the blank payload: nulls reached neither the top level
    # nor the message list.
    assert set(body) == {"model", "messages", "max_tokens", "provider"}
    assert body["messages"] == [{"role": "user", "content": "p"}]

    gid = client.post(
        "/groups",
        json={"prompt": "q", "params": {"temperature": None}, "budget": "standard"},
    ).json()["id"]
    assert client.get(f"/groups/{gid}").json()["params"] is None
    # And a group written that way still accepts a run that sets nothing,
    # so the two spellings agree in the conflict check too.
    assert (
        client.post(
            "/compare",
            json={"prompt": "q", "models": ["model/alpha"], "group_id": gid},
        ).status_code
        == 200
    )


@respx.mock
def test_zero_is_a_chosen_value_everywhere_it_travels(client):
    """The control most likely to be lost to a falsy check, and the one that
    matters most: temperature 0 is the whole point of a reproducibility run.
    Walked end to end because a falsy test anywhere on the path (payload,
    record, conflict check) would silently drop it."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    zeros = {"temperature": 0, "top_p": 0, "seed": 0}
    gid = client.post(
        "/groups", json={"prompt": "z", "params": zeros, "budget": "standard"}
    ).json()["id"]

    # A 409 here would mean one side read a zero as blankness.
    resp = client.post(
        "/compare",
        json={
            "prompt": "z",
            "models": ["model/alpha"],
            "group_id": gid,
            "params": zeros,
        },
    )

    assert resp.status_code == 200, resp.json()
    body = json.loads(respx.calls[0].request.content)
    assert body["temperature"] == 0 and body["top_p"] == 0 and body["seed"] == 0
    assert client.get(f"/groups/{gid}").json()["params"] == zeros
    # And it survives the derivation an ungrouped run relies on.
    lone = client.post(
        "/compare", json={"prompt": "z2", "models": ["model/alpha"], "params": zeros}
    ).json()["run_id"]
    assert client.get(f"/runs/{lone}").json()["params"] == zeros


def test_both_compare_surfaces_share_one_params_class(client):
    """Parity made structural in H1, asserted here so a future edit cannot
    quietly split it into two classes that then drift. The closing review
    hunts parity drift; this is what makes that hunt vacuous rather than
    careful."""
    for model in (main.CompareRequest, main.StreamCompareRequest, main.GroupCreate):
        annotation = model.model_fields["params"].annotation
        assert main.ExperimentParams in typing.get_args(annotation), model.__name__


@respx.mock
def test_the_zdr_belt_survives_every_routing_choice_through_the_endpoint(client):
    """The provider merge asserted through the wiring, not only on the pure
    function. The boot policy and the request's routing meet in app state,
    and a regression there would send a payload weaker than the one the run
    row claims, which is the failure the G5 belt exists to prevent."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    app.state.data_policy = "zdr"
    app.state.provider_prefs = provider_preferences("zdr")
    try:
        for routing, expected in (
            (None, {"sort": "throughput", "zdr": True, "data_collection": "deny"}),
            (
                "throughput",
                {"sort": "throughput", "zdr": True, "data_collection": "deny"},
            ),
            ("price", {"sort": "price", "zdr": True, "data_collection": "deny"}),
            ("default", {"zdr": True, "data_collection": "deny"}),
        ):
            body = {"prompt": "p", "models": ["model/alpha"]}
            if routing is not None:
                body["params"] = {"routing": routing}
            assert client.post("/compare", json=body).status_code == 200
            # calls[-1], not calls[0]: this loop makes several requests and
            # indexing from the front reads the previous iteration's payload,
            # which passes for whichever mode happens to come first and then
            # blames the next one.
            sent = json.loads(respx.calls[-1].request.content)
            assert sent["provider"] == expected, routing
    finally:
        app.state.data_policy = "standard"
        app.state.provider_prefs = provider_preferences("standard")


def test_the_seed_bound_is_what_a_browser_can_read_back_exactly(client):
    """Closing review finding. The seed bound was signed 64-bit, which the
    API accepted and the browser could not represent: a stored
    9223372036854775807 came back to the page as 9223372036854776000,
    measured in Chromium rather than reasoned about.

    Two consequences, both silent. History would badge a seed nobody chose,
    and reuse would prefill that wrong seed and then be refused by the
    one-experiment check as a different experiment, which is a legal run
    blocked by a corruption the user cannot see. So the bound is now the
    largest integer the whole path carries exactly, and a value above it is
    a 422 at the boundary instead of a lie in the record.
    """
    assert main.MAX_SEED == 2**53 - 1

    respx.post(OPENROUTER_URL)
    for seed in (main.MAX_SEED, -main.MAX_SEED):
        # Exactly representable: json round trips it unchanged, which is the
        # property the browser needs.
        assert json.loads(json.dumps(seed)) == seed
        assert float(seed) == seed
    for seed in (main.MAX_SEED + 1, -main.MAX_SEED - 1):
        resp = client.post(
            "/compare",
            json={"prompt": "p", "models": ["model/alpha"], "params": {"seed": seed}},
        )
        assert resp.status_code == 422, seed


# ---- Phase H1.1: the ceiling bound holds across concurrent batches.


@respx.mock
def test_review_repro_concurrent_batches_hold_the_ceiling_bound(monkeypatch, tmp_path):
    """Third external review, HIGH: the documented ceiling bound did not
    hold once more than one batch was in flight, and Phase H made /compare
    the official scripting surface, which is exactly the concurrent-batch
    path.

    The mechanism. /compare acquired one slot per model and rechecked the
    ceiling under the held slot, but settlement ran in a loop AFTER
    asyncio.gather. So a fast member released its slot having recorded
    nothing, a model from another batch took that slot, and its recheck read
    a counter that had not moved. The comment inside limited() narrated the
    same-batch blindness and then reasoned only about a single batch; nobody
    re-derived it for N batches.

    Why the post-fix bound is what it is. Settlement now happens inside the
    held slot, strictly before release. So a freed slot implies a recorded
    settlement. Once accumulated spend crosses the ceiling, every subsequent
    acquisition observes it and refuses without calling upstream. Only calls
    already executing at the moment the ceiling tripped can overshoot, and
    the semaphore caps those at MAX_CONCURRENT_UPSTREAM by construction.
    That is the whole derivation, and it is the bound asserted below.

    The shape is the reviewer's reproduction: eight concurrent five-model
    batches, one deliberately slow member each so fast members finish and
    free slots while the batch is still open, a ceiling worth half of one
    result. Reverting settlement to the post-gather loop makes this measure
    23 upstream calls against a bound of 5, stable across five runs. The
    external report said 28; the exact number is scheduling-dependent and
    the assertion below is on the bound, not on any particular overshoot.
    """
    slow = threading.Event()

    async def respond(request):
        # One slow member per batch, by model id, so each batch has fast
        # members that finish and release their slots early. That release
        # is the whole mechanism: pre-fix it carried no settlement with it.
        if b'"model/slow"' in request.content:
            await asyncio.sleep(0.25)
        return httpx.Response(
            200,
            json={
                **FIXTURE,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01},
            },
        )

    route = respx.post(OPENROUTER_URL).mock(side_effect=respond)
    lineup = ["model/slow", "model/alpha", "model/beta", "model/gamma", "model/delta"]

    # Half of one result: the very first settlement must close the gate.
    with spend_client(monkeypatch, tmp_path, limit=0.005) as c:
        statuses = []

        def one_batch():
            resp = c.post("/compare", json={"prompt": "p", "models": lineup})
            statuses.append(resp)

        threads = [threading.Thread(target=one_batch) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        slow.set()

        assert all(r.status_code in (200, 402) for r in statuses), [
            r.status_code for r in statuses
        ]
        # The bound, derived above. Pre-fix this was ~28.
        assert len(route.calls) <= main.MAX_CONCURRENT_UPSTREAM, (
            f"{len(route.calls)} calls reached upstream, bound is "
            f"{main.MAX_CONCURRENT_UPSTREAM}"
        )

        # Nothing vanished: every one of the 40 requested models is
        # accounted for as either an upstream call or a refusal. A bound met
        # by dropping work would be a different defect wearing this test's
        # green.
        refused = 0
        for resp in statuses:
            if resp.status_code == 402:
                refused += len(lineup)
                continue
            for result in resp.json()["results"]:
                if result["error"] and "refused" in result["error"]:
                    refused += 1
        assert refused + len(route.calls) == 8 * len(lineup), (
            f"{refused} refused + {len(route.calls)} called != 40"
        )


# ---- Phase H1.2: the manifest the group declares is enforced.


def declared_group(client, models, budget="standard", prompt="declared"):
    return client.post(
        "/groups",
        json={"prompt": prompt, "models": models, "budget": budget},
    ).json()["id"]


@respx.mock
def test_review_repro_group_refuses_an_undeclared_model(client):
    """Third external review, HIGH: the group stored an ordered lineup
    before any call and presented itself as the experiment record, but
    models_json was never read anywhere in production. Schema and migration
    list only. The declaration was write-only, so a group declaring one
    model accepted an undeclared one at both budget tiers and the detail
    showed only what actually ran.

    Both endpoints refuse it now, at entry, with respx proving no upstream
    call: a refusal that spent money would trade one defect for a worse one.
    """
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = declared_group(client, ["model/alpha"])

    stream = client.post(
        "/compare/stream",
        json={
            "prompt": "declared",
            "model": "model/bare",
            "group_id": gid,
            "budget": "standard",
        },
    )
    assert stream.status_code == 409
    assert (
        "model/bare is not in this group's declared lineup" in (stream.json()["detail"])
    )

    batch = client.post(
        "/compare",
        json={
            "prompt": "declared",
            "models": ["model/bare"],
            "group_id": gid,
            "budget": "standard",
        },
    )
    assert batch.status_code == 409
    assert "lineup is model/alpha" in batch.json()["detail"]

    assert route.call_count == 0, "a manifest refusal must spend nothing"


@respx.mock
def test_a_declared_member_at_the_wrong_position_is_refused(client):
    """Position is part of the declaration: the lineup is ordered, and a
    member claiming the wrong slot would rebuild the comparison in an order
    it never ran in."""
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = declared_group(client, ["model/alpha", "model/bare"])
    body = {"prompt": "declared", "group_id": gid, "budget": "standard"}

    wrong = client.post(
        "/compare/stream", json={**body, "model": "model/bare", "position": 0}
    )
    assert wrong.status_code == 409
    assert "declared at position 1" in wrong.json()["detail"]
    assert "got 0" in wrong.json()["detail"]
    assert route.call_count == 0

    right = client.post(
        "/compare/stream", json={**body, "model": "model/bare", "position": 1}
    )
    assert right.status_code == 200
    # A member that claims no slot is not checked against one: it made no
    # claim, and inventing one for it would refuse a run for something it
    # never said.
    unclaimed = client.post("/compare/stream", json={**body, "model": "model/bare"})
    assert unclaimed.status_code == 200


@respx.mock
def test_a_batch_that_reorders_the_declared_lineup_is_refused(client):
    """The batch surface declares its whole lineup in one array, so order is
    part of the claim: the same models side by side in a different order is
    a different comparison."""
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = declared_group(client, ["model/alpha", "model/bare"])

    reordered = client.post(
        "/compare",
        json={
            "prompt": "declared",
            "models": ["model/bare", "model/alpha"],
            "group_id": gid,
            "budget": "standard",
        },
    )

    assert reordered.status_code == 409
    detail = reordered.json()["detail"]
    assert "lineup is model/alpha, model/bare" in detail
    assert "the batch sent model/bare, model/alpha" in detail
    assert route.call_count == 0


@respx.mock
def test_a_member_at_the_wrong_budget_is_refused(client):
    """Budget completes the manifest. A comparison whose members ran at
    different token budgets is not one experiment, and until this column
    existed nothing could say so: the reviewer's reproduction accepted an
    undeclared model at BOTH tiers."""
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = declared_group(client, ["model/alpha"], budget="extended")

    for path, extra in (
        ("/compare", {"models": ["model/alpha"]}),
        ("/compare/stream", {"model": "model/alpha"}),
    ):
        wrong = client.post(
            path,
            json={
                "prompt": "declared",
                "group_id": gid,
                "budget": "standard",
                **extra,
            },
        )
        assert wrong.status_code == 409, path
        assert "declared the extended budget" in wrong.json()["detail"]
    assert route.call_count == 0


@respx.mock
def test_the_budget_check_compares_the_requested_tier_not_the_clamped_tokens(
    client,
):
    """The check is on the TIER the caller asked for, never on the token
    count that tier resolves to.

    Two models with different published completion caps legitimately produce
    different effective token numbers inside one extended comparison: that is
    what the clamp is for. Comparing clamped numbers would refuse a correct
    run and call it a budget conflict, which is the same class of defect as
    the drift that blocks a legal value, so it gets its own lock.
    """
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    # model/alpha has no published cap in the fixture catalog; model/capped
    # publishes a small one, so extended clamps differently for each.
    gid = declared_group(client, ["model/alpha", "model/capped"], budget="extended")

    both = client.post(
        "/compare",
        json={
            "prompt": "declared",
            "models": ["model/alpha", "model/capped"],
            "group_id": gid,
            "budget": "extended",
        },
    )

    assert both.status_code == 200, both.json()
    sent = [json.loads(call.request.content)["max_tokens"] for call in route.calls]
    # The point of the test: the two effective budgets genuinely differ, and
    # both members were accepted anyway.
    assert len(set(sent)) == 2, sent


@respx.mock
def test_a_legacy_group_skips_membership_and_budget_enforcement(client):
    """Two NULLs, two meanings, and this is the one the params rule does
    NOT share.

    A NULL params_json is the affirmative fact that no control was set,
    because controls arrived with the concept and every group predating them
    genuinely set none; that is why a NULL-params group refuses a
    controls-carrying member. A NULL models_json or NULL budget is silence
    instead: those groups were created before anything recorded a lineup or
    a tier, so enforcing against them would refuse correct work on every
    group in an existing database.
    """
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    # The legacy shape, written straight to the table: no lineup, no budget.
    gid = client.post("/groups", json={"budget": "standard"}).json()["id"]
    client.app.state.db.execute(
        "UPDATE groups SET models_json = NULL, budget = NULL WHERE id = ?", (gid,)
    )
    client.app.state.db.commit()

    for path, extra in (
        ("/compare", {"models": ["model/bare"]}),
        ("/compare/stream", {"model": "model/bare", "position": 3}),
    ):
        resp = client.post(
            path,
            json={
                "prompt": "anything",
                "group_id": gid,
                "budget": "extended",
                **extra,
            },
        )
        assert resp.status_code == 200, (path, resp.text)

    # The contrast, in the same test so the two rules are read together: the
    # same group still refuses a controls-carrying member, because NULL
    # params means none were set rather than unknown.
    controlled = client.post(
        "/compare",
        json={
            "prompt": "anything",
            "models": ["model/bare"],
            "group_id": gid,
            "budget": "extended",
            "params": {"seed": 1},
        },
    )
    assert controlled.status_code == 409
    assert "records no controls" in controlled.json()["detail"]


@respx.mock
def test_the_group_detail_exposes_what_was_declared(client):
    """The declaration was write-only: the detail showed what ran and never
    what was supposed to. An experiment record you cannot read back is not a
    record."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)
    gid = declared_group(
        client, ["model/alpha", "model/bare"], budget="extended", prompt="the ask"
    )

    detail = client.get(f"/groups/{gid}").json()

    assert detail["prompt"] == "the ask"
    assert detail["models"] == ["model/alpha", "model/bare"]
    assert detail["budget"] == "extended"


def test_a_legacy_group_detail_returns_nulls_not_strings(client):
    """Legacy groups have no declaration, and the detail must say so with
    nulls. The frontend guards on these: a null reaching a textarea would
    render the four characters "null" as if someone had typed them, which is
    absence rendered as a value."""
    gid = client.post("/groups", json={"budget": "standard"}).json()["id"]
    client.app.state.db.execute(
        "UPDATE groups SET prompt_text = NULL, models_json = NULL, budget = NULL"
        " WHERE id = ?",
        (gid,),
    )
    client.app.state.db.commit()

    detail = client.get(f"/groups/{gid}").json()

    assert detail["prompt"] is None
    assert detail["models"] is None
    assert detail["budget"] is None
    # And nothing stringified them on the way out.
    assert "null" not in json.dumps(
        {k: v for k, v in detail.items() if k != "runs"}
    ).replace(": null", "")


def test_the_group_budget_is_required(client):
    """Required, unlike the rest of the manifest. Nothing has ever created a
    group without knowing its budget, so accepting a group that declines to
    say would leave the hole this workstream closes."""
    assert client.post("/groups", json={"prompt": "p"}).status_code == 422
    assert client.post("/groups", json={"budget": "standard"}).status_code == 201
    assert client.post("/groups", json={"budget": "lavish"}).status_code == 422


# ---- Phase I1: datasets as immutable files, experiments as the aggregate.


def write_dataset(tmp_path, *rows, name="d.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def experiment_body(path, **overrides):
    body = {
        "name": "smoke",
        "dataset_path": path,
        "lineup": ["model/alpha", "model/beta"],
        "budget": "standard",
    }
    body.update(overrides)
    return body


def test_creating_an_experiment_records_the_manifest_before_anything_runs(
    client, tmp_path
):
    """The group-row discipline one level up. Everything knowable at
    creation is written at creation, including which dataset bytes were
    read, so an experiment that never starts still says what it was going
    to be."""
    path = write_dataset(
        tmp_path, {"id": "t1", "prompt": "hi"}, {"id": "t2", "prompt": "yo"}
    )

    created = client.post(
        "/experiments",
        json=experiment_body(path, params={"seed": 7}, repeats=3),
    )

    assert created.status_code == 201
    detail = client.get(f"/experiments/{created.json()['id']}").json()
    assert detail["status"] == "created"
    assert detail["dataset_name"] == "d.jsonl"
    assert len(detail["dataset_digest"]) == 64
    assert detail["lineup"] == ["model/alpha", "model/beta"]
    assert detail["params"] == {"seed": 7}
    assert detail["repeats"] == 3
    assert detail["tasks_total"] == 2
    # 2 tasks x 3 repeats x 2 models, the number of paid calls this will
    # make, fixed now so a halt cannot make itself look complete.
    assert detail["trials_total"] == 12
    assert detail["trials_done"] == 0
    assert detail["estimand_mode"] == "routed_service"
    # The same provenance a run row carries, recorded once because it is
    # fixed for the whole experiment.
    assert detail["catalog_digest"] == TEST_CATALOG["digest"]
    assert detail["data_policy"] == "standard"


def test_an_experiment_records_only_the_controls_that_were_set(client, tmp_path):
    """Rule two, one level up: an experiment that set nothing records
    nothing, not an empty object and not a set of defaults."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    created = client.post("/experiments", json=experiment_body(path))

    assert client.get(f"/experiments/{created.json()['id']}").json()["params"] is None


def test_a_malformed_dataset_fails_creation_with_the_line(client, tmp_path):
    """422 rather than 500, and the line number rather than a stack trace:
    the caller's next action is to open that line."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "t1", "prompt": "hi"}\n{"id": "t1", "prompt": "yo"}\n')

    resp = client.post("/experiments", json=experiment_body(str(path)))

    assert resp.status_code == 422
    assert "line 2" in resp.json()["detail"]
    assert "duplicate task id" in resp.json()["detail"]


def test_a_missing_dataset_file_names_the_path(client, tmp_path):
    resp = client.post(
        "/experiments", json=experiment_body(str(tmp_path / "nope.jsonl"))
    )

    assert resp.status_code == 422
    assert "cannot read dataset" in resp.json()["detail"]


def test_an_experiment_too_large_to_run_is_refused_with_the_arithmetic(
    client, tmp_path
):
    """The refusal shows its working, because the caller has three levers
    (tasks, repeats, lineup) and needs to know which one to pull."""
    rows = [{"id": f"t{i}", "prompt": "hi"} for i in range(600)]
    path = write_dataset(tmp_path, *rows)

    resp = client.post("/experiments", json=experiment_body(path, repeats=5))

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "600 tasks x 5 repeats x 2 models is 6000" in detail
    assert "5000 limit" in detail


def test_provider_pins_are_refused_under_the_routed_service_estimand(client, tmp_path):
    """A pin under an estimand defined by dynamic routing is a
    contradiction. Refusing beats ignoring: a silently dropped pin would
    leave a record claiming a pin that never rode a payload."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post(
        "/experiments",
        json=experiment_body(path, provider_pins={"model/alpha": "Together"}),
    )

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "routed-service estimand routes dynamically" in detail
    # The refusal names the remedy. A 422 that says only what is wrong
    # leaves the caller to guess which of two estimands they wanted.
    assert 'estimand_mode to "underlying_model"' in detail


def test_unknown_experiment_fields_are_rejected(client, tmp_path):
    """extra="forbid" reaches the new surface too."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post("/experiments", json=experiment_body(path, temperature=0.5))

    assert resp.status_code == 422


def test_experiments_list_newest_first_and_a_missing_one_404s(client, tmp_path):
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})
    first = client.post("/experiments", json=experiment_body(path, name="one")).json()
    second = client.post("/experiments", json=experiment_body(path, name="two")).json()

    listed = client.get("/experiments").json()["experiments"]

    assert [e["id"] for e in listed[:2]] == [second["id"], first["id"]]
    assert client.get("/experiments/999999").status_code == 404


def test_the_experiment_endpoints_are_not_stored_by_any_cache(client, tmp_path):
    """An experiment body carries every prompt in its dataset name and its
    controls, so it belongs to the private, no-store class like every other
    dynamic response."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})
    created = client.post("/experiments", json=experiment_body(path))

    for resp in (
        created,
        client.get("/experiments"),
        client.get(f"/experiments/{created.json()['id']}"),
    ):
        assert resp.headers.get("cache-control") == "private, no-store"


# ---- Phase I2: the runner. Repeats, rotation, and honest interruption.


def run_experiment_to_completion(client, experiment_id, path, timeout_s=20.0):
    """Start the runner and drain its progress stream to a terminal state.

    Draining the SSE stream rather than polling in a loop is what makes
    this deterministic: the stream ends exactly when the experiment
    reaches a terminal status, so there is no sleep to tune and no race
    between the assertion and the runner.
    """
    started = client.post(
        f"/experiments/{experiment_id}/start", json={"dataset_path": path}
    )
    assert started.status_code == 202, started.text
    last = None
    with client.stream("GET", f"/experiments/{experiment_id}/progress") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                last = json.loads(line[5:])
    return last


@respx.mock
def test_an_experiment_runs_every_cell_and_records_each_as_a_group(client, tmp_path):
    """The end to end shape: two tasks, two repeats, two models is eight
    trials, each one a group with its own cell coordinates and its own
    persisted run."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(
        tmp_path, {"id": "t1", "prompt": "first"}, {"id": "t2", "prompt": "second"}
    )
    eid = client.post("/experiments", json=experiment_body(path, repeats=2)).json()[
        "id"
    ]

    final = run_experiment_to_completion(client, eid, path)

    assert final["status"] == "done"
    assert final["trials_done"] == 8
    assert final["trials_total"] == 8
    detail = client.get(f"/experiments/{eid}").json()
    assert detail["status"] == "done"
    assert detail["trials_done"] == 8
    # One group per cell, four cells, each carrying its coordinates.
    rows = client.app.state.db.execute(
        """SELECT task_id, repeat_index, rotation_index FROM groups
           WHERE experiment_id = ? ORDER BY task_id, repeat_index""",
        (eid,),
    ).fetchall()
    assert [(r["task_id"], r["repeat_index"]) for r in rows] == [
        ("t1", 0),
        ("t1", 1),
        ("t2", 0),
        ("t2", 1),
    ]


@respx.mock
def test_rotation_is_recorded_and_actually_rotates_the_entry_order(client, tmp_path):
    """The position-effect answer, asserted from the record rather than
    from the plan: the rotation index on each group must advance, and the
    request order upstream must follow it.

    Models enter the semaphore in lineup order, so without rotation the
    first-listed model would get a slot first on every single trial. That
    is a systematic advantage that averaging cannot remove, because it
    points the same way every time.
    """
    order = []

    def route(request):
        order.append(json.loads(request.content)["model"])
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "only"})
    eid = client.post(
        "/experiments",
        json=experiment_body(
            path, repeats=3, lineup=["model/alpha", "model/beta", "model/bare"]
        ),
    ).json()["id"]

    run_experiment_to_completion(client, eid, path)

    rows = client.app.state.db.execute(
        "SELECT rotation_index FROM groups WHERE experiment_id = ? ORDER BY repeat_index",
        (eid,),
    ).fetchall()
    assert [r["rotation_index"] for r in rows] == [0, 1, 2]
    # Three repeats of three models, each repeat starting one later.
    assert order == [
        "model/alpha",
        "model/beta",
        "model/bare",
        "model/beta",
        "model/bare",
        "model/alpha",
        "model/bare",
        "model/alpha",
        "model/beta",
    ]


@respx.mock
def test_repeats_send_derived_seeds_and_an_unseeded_experiment_sends_none(
    client, tmp_path
):
    """Rule one survives the repeat machinery in both directions: a seeded
    experiment sends base+N so the repeats sample rather than repeat, and
    an unseeded one sends no seed at all."""
    seeds = []

    def route(request):
        seeds.append(json.loads(request.content).get("seed"))
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "only"})

    seeded = client.post(
        "/experiments",
        json=experiment_body(
            path, repeats=3, lineup=["model/alpha"], params={"seed": 100}
        ),
    ).json()["id"]
    run_experiment_to_completion(client, seeded, path)
    assert seeds == [100, 101, 102]

    seeds.clear()
    bare = client.post(
        "/experiments", json=experiment_body(path, repeats=2, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, bare, path)
    assert seeds == [None, None]


@respx.mock
def test_a_tasks_own_system_prompt_reaches_the_wire(client, tmp_path):
    """The more specific declaration wins, and it is recorded on the group
    like any other control so the record says what was sent."""
    sent = []

    def route(request):
        sent.append(json.loads(request.content)["messages"])
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "body", "system": "be terse"})
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]

    run_experiment_to_completion(client, eid, path)

    assert sent[0][0] == {"role": "system", "content": "be terse"}
    assert sent[0][1] == {"role": "user", "content": "body"}


@respx.mock
def test_a_second_experiment_cannot_start_while_one_runs(client, tmp_path):
    """One at a time, said plainly. They share the upstream slots and the
    ceiling, so concurrent experiments would measure each other."""
    # An asyncio gate, not a threading one. The runner is a task on the
    # same loop that serves requests, so a blocking wait inside the route
    # would stall the very request that has to observe the 409. Awaiting
    # yields the loop, which makes the ordering deterministic rather than
    # a race between a sleep and the scheduler.
    gate = asyncio.Event()

    async def route(request):
        await gate.wait()
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "only"})
    first = client.post("/experiments", json=experiment_body(path)).json()["id"]
    second = client.post("/experiments", json=experiment_body(path)).json()["id"]

    assert (
        client.post(
            f"/experiments/{first}/start", json={"dataset_path": path}
        ).status_code
        == 202
    )
    blocked = client.post(f"/experiments/{second}/start", json={"dataset_path": path})

    assert blocked.status_code == 409
    assert "already running" in blocked.json()["detail"]
    assert "measure each other" in blocked.json()["detail"]
    gate.set()
    with client.stream("GET", f"/experiments/{first}/progress") as resp:
        for _ in resp.iter_lines():
            pass


@respx.mock
def test_an_experiment_runs_once(client, tmp_path):
    """A finished experiment is a record, not a template. Re-running into
    the same row would mix two runs' trials under one manifest."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "only"})
    eid = client.post("/experiments", json=experiment_body(path)).json()["id"]
    run_experiment_to_completion(client, eid, path)

    again = client.post(f"/experiments/{eid}/start", json={"dataset_path": path})

    assert again.status_code == 409
    assert "is done" in again.json()["detail"]


@respx.mock
def test_a_dataset_changed_since_creation_is_refused_rather_than_run(client, tmp_path):
    """The whole point of recording a content digest. Running anyway would
    produce a record citing one dataset and containing another, which is
    the most misleading artifact this phase could emit."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "original"})
    eid = client.post("/experiments", json=experiment_body(path)).json()["id"]
    Path(path).write_text('{"id": "t1", "prompt": "edited"}\n', encoding="utf-8")

    final = run_experiment_to_completion(client, eid, path)

    assert final["status"] == "failed"
    assert "dataset changed since this experiment was created" in final["status_detail"]
    assert respx.calls.call_count == 0


@respx.mock
def test_a_stop_halts_between_trials_and_leaves_an_honest_partial(client, tmp_path):
    """A stopped experiment is a partial record, not a failed one, and it
    says which it is. The trials that ran are real and stay; the ones that
    did not are visible as the difference between done and total."""
    seen = {"n": 0}
    eid_box = {}

    async def route(request):
        # The stop endpoint's own body, awaited on the loop the runner
        # lives on, rather than a nested synchronous client.post from
        # inside the route. A nested request would re-enter the test
        # client's portal while its loop is mid-trial, which passed in
        # isolation and failed in the full suite: the runner ended
        # "failed" instead of "stopped" and the reason was the harness,
        # not the product. Awaiting the endpoint exercises the same code
        # including its 409 guard, and is deterministic.
        seen["n"] += 1
        if seen["n"] == 1:
            await main.stop_experiment(eid_box["id"])
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a"},
        {"id": "t2", "prompt": "b"},
        {"id": "t3", "prompt": "c"},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    eid_box["id"] = eid

    final = run_experiment_to_completion(client, eid, path)

    assert final["status"] == "stopped"
    assert final["status_detail"] == "stopped between trials"
    # The first trial completed and is real history; the rest never ran.
    assert final["trials_done"] == 1
    assert final["trials_total"] == 3
    assert seen["n"] == 1


@respx.mock
def test_the_ceiling_halts_the_experiment_and_says_why(client, tmp_path, monkeypatch):
    """A refusal means the money ran out. Halting beats continuing: every
    later row would be a refusal, which is technically honest, practically
    noise, and expensive in wall time."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(
            200,
            stream=ChunkStream(
                [
                    sse({"choices": [{"delta": {"content": "hi"}}]}),
                    sse(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "cost": 1.0,
                            },
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a"},
        {"id": "t2", "prompt": "b"},
        {"id": "t3", "prompt": "c"},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    # Half of one result's charge: the first settlement closes the gate.
    client.app.state.spend_limit_usd = 0.5

    final = run_experiment_to_completion(client, eid, path)

    assert final["status"] == "halted_on_refusal"
    assert "spend ceiling" in final["status_detail"]
    assert final["trials_done"] == 1
    assert final["trials_refused"] == 1
    # Conservation: every requested trial is accounted for as run,
    # refused, or never attempted because the run halted.
    assert final["trials_done"] + final["trials_refused"] <= final["trials_total"]
    assert respx.calls.call_count == 1


@respx.mock
def test_conservation_holds_across_a_whole_experiment(client, tmp_path):
    """The property the ceiling machinery has carried since F.3, stated
    over an experiment rather than a batch: refusals plus upstream calls
    account for every trial the experiment attempted."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(
        tmp_path, {"id": "t1", "prompt": "a"}, {"id": "t2", "prompt": "b"}
    )
    eid = client.post("/experiments", json=experiment_body(path, repeats=2)).json()[
        "id"
    ]

    final = run_experiment_to_completion(client, eid, path)

    attempted = final["trials_done"] + final["trials_refused"] + final["trials_failed"]
    assert attempted == final["trials_total"]
    assert respx.calls.call_count + final["trials_refused"] == final["trials_total"]


@respx.mock
def test_review_repro_a_failed_trial_is_not_also_a_done_trial(client, tmp_path):
    """The conservation test above passed vacuously.

    Its fixture never fails a trial, so the sum it checks could not
    detect double counting, and the runner was doing exactly that: a
    failed trial bumped trials_done AND trials_failed. The README states
    the property as four disjoint buckets (completed, failed, refused,
    never attempted), and with one failure in four the sum was five out
    of four.

    Stated over a fixture that actually fails something, which is the
    only way this assertion is worth making.
    """
    calls = {"n": 0}

    def route(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500)
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path, {"id": "t1", "prompt": "a"}, {"id": "t2", "prompt": "b"}
    )
    eid = client.post("/experiments", json=experiment_body(path)).json()["id"]

    final = run_experiment_to_completion(client, eid, path)

    assert final["trials_total"] == 4
    assert final["trials_failed"] == 1
    assert final["trials_refused"] == 0
    # Three succeeded, one failed, none refused, none skipped.
    assert final["trials_done"] == 3
    assert (
        final["trials_done"] + final["trials_failed"] + final["trials_refused"]
        == final["trials_total"]
    )


@respx.mock
def test_progress_frames_are_absolute_so_a_reconnect_costs_nothing(client, tmp_path):
    """Every frame carries totals rather than deltas, which is what makes
    a dropped frame free and reconnection the ordinary case rather than an
    error path."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "a"})
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    # Rejoining a finished experiment still gets its state, immediately,
    # rather than hanging or 404ing.
    with client.stream("GET", f"/experiments/{eid}/progress") as resp:
        frames = [
            json.loads(line[5:])
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    assert frames[-1]["status"] == "done"
    assert frames[-1]["trials_done"] == 1
    assert "spend_usd" in frames[-1]


def test_stopping_an_experiment_that_is_not_running_409s(client, tmp_path):
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "a"})
    eid = client.post("/experiments", json=experiment_body(path)).json()["id"]

    resp = client.post(f"/experiments/{eid}/stop", json={})

    assert resp.status_code == 409
    assert "not running" in resp.json()["detail"]


@respx.mock
def test_continue_mode_records_a_refusal_and_keeps_going(client, tmp_path):
    """halt_on_refusal false is an implemented path, not a declared one.

    Written now rather than with I5's refusal-heavy aggregates, because
    I5 will build its failure-inclusive denominators on top of this
    branch. A path that was implemented dead would be discovered by the
    thing that depends on it, which is the expensive order.

    The shape is one refusal followed by proof of progress: the ceiling
    trips on the first settlement, the second trial is refused, and the
    runner reaches the third rather than stopping. Refusals cost no
    upstream call, so the call count is what separates "kept going" from
    "stopped and the counters lied".
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(
            200,
            stream=ChunkStream(
                [
                    sse({"choices": [{"delta": {"content": "hi"}}]}),
                    sse(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "cost": 1.0,
                            },
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a"},
        {"id": "t2", "prompt": "b"},
        {"id": "t3", "prompt": "c"},
    )
    eid = client.post(
        "/experiments",
        json=experiment_body(path, lineup=["model/alpha"], halt_on_refusal=False),
    ).json()["id"]
    client.app.state.spend_limit_usd = 0.5

    final = run_experiment_to_completion(client, eid, path)

    # Ran to the end rather than halting, and says so.
    assert final["status"] == "done"
    assert final["status_detail"] is None
    assert final["trials_total"] == 3
    # One paid trial before the ceiling closed, then two refusals: the
    # runner passed the first refusal and reached the last task.
    assert final["trials_done"] == 1
    assert final["trials_refused"] == 2
    # Conservation over the whole experiment, refusals included.
    assert final["trials_done"] + final["trials_refused"] == final["trials_total"]
    # A refusal spends nothing, so exactly one call reached upstream.
    assert respx.calls.call_count == 1


@respx.mock
def test_result_ids_reach_the_replay_path_and_not_the_live_one(client):
    """The exposure route for result ids, decided before I4 needs them.

    Blind rating and every score row point at a result by id, and both are
    produced while looking at stored history. So the replay path carries
    the id and the live path does not, structurally: StoredModelResult is
    a subclass rather than a nullable field, because a live result has no
    id at the moment it is returned and never will. Persistence happens
    after the response is built, and the post-spend invariant explicitly
    permits it to fail and answer run_id null. A nullable id on one shape
    would force every consumer to know which endpoint it was talking to
    before it could read None correctly.
    """
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, json=response_for("model/alpha", "hi"))
    )
    live = client.post("/compare", json={"prompt": "ids", "models": ["model/alpha"]})

    assert "id" not in live.json()["results"][0]

    stored = client.get(f"/runs/{live.json()['run_id']}").json()
    assert isinstance(stored["results"][0]["id"], int)
    # And through the group replay, which is the view blind rating uses.
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=alpha_stream())
    )
    group_id = client.post("/groups", json={"budget": "standard"}).json()["id"]
    frames = stream_events(
        client, {"prompt": "ids", "model": "model/alpha", "group_id": group_id}
    )
    done = next(f for f in frames if f["type"] == "done")
    # The live done frame is the same shape as the live batch response:
    # no id, because there is not one yet.
    assert "id" not in done["result"]

    detail = client.get(f"/groups/{group_id}").json()
    ids = [r["id"] for run in detail["runs"] for r in run["results"]]
    assert ids and all(isinstance(i, int) for i in ids)


# ---- Phase I3: scoring passes, deterministic and judged.

import time

from bench.models import JUDGE_MAX_TOKENS
from bench.scoring import latest_per_key


def score_experiment_to_completion(client, eid, path, judge_model=None, timeout_s=20.0):
    """Start a scoring pass and wait for it to finish.

    Polled rather than streamed because scoring has no progress endpoint
    of its own yet; the state flag on app.state is the truth and this
    reads it directly rather than guessing at a duration.
    """
    body = {"dataset_path": path}
    if judge_model is not None:
        body["judge_model"] = judge_model
    started = client.post(f"/experiments/{eid}/score", json=body)
    assert started.status_code == 202, started.text
    deadline = time.monotonic() + timeout_s
    while client.app.state.scoring_run["active"] is not None:
        assert time.monotonic() < deadline, "scoring pass did not finish"
        # One cheap request drives the loop, since the background task
        # only progresses while the test client's portal is running.
        client.get("/models")
    assert client.app.state.scoring_run["error"] is None, client.app.state.scoring_run[
        "error"
    ]


def scores_in(client, eid):
    rows = client.app.state.db.execute(
        """SELECT s.* FROM scores s
           JOIN results r ON r.id = s.result_id
           JOIN runs ru ON ru.id = r.run_id
           JOIN groups g ON g.id = ru.group_id
           WHERE g.experiment_id = ? ORDER BY s.id""",
        (eid,),
    ).fetchall()
    return [dict(r) for r in rows]


@respx.mock
def test_a_deterministic_pass_scores_every_trial_including_the_failures(
    client, tmp_path
):
    """Failures are scored, not skipped. A trial that errored is a trial
    that failed the task, and leaving it unscored would let the report
    drop it from the denominator."""
    calls = {"n": 0}

    def route(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500)
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {
            "id": "t1",
            "prompt": "a",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
        {
            "id": "t2",
            "prompt": "b",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    score_experiment_to_completion(client, eid, path)

    rows = scores_in(client, eid)
    assert len(rows) == 2
    assert {r["scorer"] for r in rows} == {"contains"}
    # One passed on its text, one scored zero because it never produced
    # any, and the row says which.
    assert sorted(r["score"] for r in rows) == [0.0, 1.0]
    failed = next(r for r in rows if r["score"] == 0.0)
    assert failed["detail"] == "no response text: the trial did not complete"


@respx.mock
def test_a_judge_pass_records_the_verdict_its_cost_and_its_model(client, tmp_path):
    def route(request):
        body = json.loads(request.content)
        if body["model"] == "judge/one":
            return httpx.Response(
                200,
                json={
                    "id": "gen-judge-9",
                    "choices": [
                        {
                            "message": {
                                "content": '{"score": 0.5, "reason": "partial"}'
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 9,
                        "cost": 0.00002,
                    },
                },
            )
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {
            "id": "t1",
            "prompt": "a",
            "rubric": "score it",
            "scorer": {"kind": "judge"},
        },
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    before = client.app.state.accumulated_spend_usd

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    row = scores_in(client, eid)[0]
    assert row["scorer"] == "judge"
    assert row["score"] == 0.5
    assert row["detail"] == "partial"
    assert row["judge_model"] == "judge/one"
    assert row["judge_generation_id"] == "gen-judge-9"
    assert row["judge_billed_cost_usd"] == 0.00002
    # Judge spend is spend: it moves the same accumulator the ceiling
    # reads, so a scoring pass cannot run for free against the limit.
    assert client.app.state.accumulated_spend_usd == pytest.approx(before + 0.00002)


@respx.mock
def test_no_judge_payload_ever_carries_a_lineup_identity(client, tmp_path):
    """The blind-by-construction claim, asserted at the wire rather than
    at the function boundary: whatever the pass does, no request that
    reaches a judge may name a model the experiment is comparing."""
    judge_payloads = []

    def route(request):
        body = json.loads(request.content)
        if body["model"] == "judge/one":
            judge_payloads.append(request.content.decode())
            return httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 1.0}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "grade", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments",
        json=experiment_body(path, lineup=["model/alpha", "model/beta"]),
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    assert len(judge_payloads) == 2
    for payload in judge_payloads:
        assert "model/alpha" not in payload
        assert "model/beta" not in payload
        assert "judge/one" in payload


@respx.mock
def test_a_judge_in_the_lineup_flags_every_row_it_produces(client, tmp_path):
    """Self-judging is recorded, not prevented. The user may have good
    reason; a silent absorption of the self-preference concern is what
    would be wrong."""

    def route(request):
        body = json.loads(request.content)
        if body["max_tokens"] == JUDGE_MAX_TOKENS:
            return httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 1.0}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "grade", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    score_experiment_to_completion(client, eid, path, judge_model="model/alpha")

    assert scores_in(client, eid)[0]["self_judged"] == 1


@respx.mock
def test_a_malformed_verdict_is_recorded_as_a_scoring_failure(client, tmp_path):
    """Never a guessed number. The row exists, carries no score, and says
    what went wrong, so the report can tell "unscorable" from "not scored
    yet"."""

    def route(request):
        body = json.loads(request.content)
        if body["max_tokens"] == JUDGE_MAX_TOKENS:
            return httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": "seems fine to me"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "grade", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    row = scores_in(client, eid)[0]
    assert row["score"] is None
    assert "no JSON object" in row["detail"]


@respx.mock
def test_a_ceiling_refusal_during_scoring_records_the_gap_and_the_pass_goes_on(
    client, tmp_path
):
    """The interruption policy, and it is the opposite of the trial
    runner's default on purpose. A refused trial can only be recovered by
    paying for the model call again, so halting protects the budget for a
    decision the user should make. A refused score can be filled in by a
    later pass over the same stored text at no extra model cost, so
    stopping the whole pass for one would trade a complete scoring run
    for nothing.
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "grade", "scorer": {"kind": "judge"}},
        {"id": "t2", "prompt": "b", "rubric": "grade", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    # Close the gate before scoring starts, so every judge call refuses.
    client.app.state.spend_limit_usd = 0.0000001
    client.app.state.accumulated_spend_usd = 1.0

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    rows = scores_in(client, eid)
    # Both results have a row: the pass covered everything rather than
    # stopping at the first refusal.
    assert len(rows) == 2
    for row in rows:
        assert row["score"] is None
        assert "re-run the scoring pass to fill it in" in row["detail"]
        assert row["judge_model"] == "judge/one"


@respx.mock
def test_review_repro_the_judge_rechecks_the_ceiling_after_admission(
    client, tmp_path, monkeypatch
):
    """F1.2's invariant, which the scoring pass was missing.

    The pass checked the ceiling and then queued for one of the five
    upstream slots, with no check after admission. Those are minutes
    apart on a pass over three hundred results, and the ceiling being
    crossed during the wait is the ordinary case rather than the exotic
    one; the trial runner rechecks inside the held slot for exactly this
    reason and the judge did not.

    The fake reports a clear ceiling on the call before the queue and a
    reached one after, which is the sequence the real gap produces. It
    also records the semaphore's free-slot count at each call, because
    the claim is about POSITION relative to admission and a second check
    placed just before the queue would pass a call-count assertion while
    leaving the invariant broken.
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "grade", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    # _value rather than a public accessor because nothing public reports
    # how many slots are free; the suite already reads it for the same
    # reason in the semaphore-honesty tests.
    free_slots = []
    answers = iter([False, True])

    def fake_ceiling():
        free_slots.append(client.app.state.upstream_semaphore._value)
        return next(answers, True)

    monkeypatch.setattr(main, "spend_ceiling_reached", fake_ceiling)

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    row = scores_in(client, eid)[0]
    assert row["score"] is None
    assert "re-run the scoring pass to fill it in" in row["detail"]
    # Twice, and the second time with a slot held: one fewer free than
    # the first, which is what "after admission" means here.
    assert len(free_slots) == 2
    assert free_slots[1] == free_slots[0] - 1


@respx.mock
def test_rescoring_appends_and_the_latest_row_is_the_one_that_counts(client, tmp_path):
    """Idempotent per (result, scorer, judge_model) means a second pass is
    safe, not that it is a no-op. The new row lands beside the old one and
    the report reads the newest; the older stays as the audit trail."""
    verdicts = iter(['{"score": 0.0}', '{"score": 1.0}'])

    def route(request):
        body = json.loads(request.content)
        if body["max_tokens"] == JUDGE_MAX_TOKENS:
            return httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": next(verdicts)},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "grade", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")
    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    rows = scores_in(client, eid)
    assert [r["score"] for r in rows] == [0.0, 1.0]
    latest = latest_per_key(rows)
    assert [r["score"] for r in latest] == [1.0]


def test_scoring_a_running_experiment_is_refused(client, tmp_path):
    """A pass over a half-finished experiment would score whatever
    happened to exist, and its own re-runnability would then hide that it
    had done so."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "a"})
    eid = client.post("/experiments", json=experiment_body(path)).json()["id"]

    resp = client.post(f"/experiments/{eid}/score", json={"dataset_path": path})

    assert resp.status_code == 409
    assert "once its trials have finished" in resp.json()["detail"]


@respx.mock
def test_scoring_against_a_changed_dataset_is_refused(client, tmp_path):
    """The rubrics live in the file. Scoring against different tasks would
    attribute one rubric's verdict to another's trial."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(
        tmp_path,
        {
            "id": "t1",
            "prompt": "a",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    Path(path).write_text(
        '{"id": "t1", "prompt": "a", "reference": "Goodbye", '
        '"scorer": {"kind": "contains"}}\n',
        encoding="utf-8",
    )

    resp = client.post(f"/experiments/{eid}/score", json={"dataset_path": path})

    assert resp.status_code == 422
    assert "dataset changed" in resp.json()["detail"]
    assert "attribute one rubric's verdict to another's trial" in resp.json()["detail"]


@respx.mock
def test_a_declared_pass_threshold_decides_a_judged_pass(client, tmp_path):
    """The threshold rides the task's scorer spec into the score row.
    Without one, passed stays empty and the report says score mean; with
    one, the author's own cutoff decides."""

    def route(request):
        body = json.loads(request.content)
        if body["max_tokens"] == JUDGE_MAX_TOKENS:
            return httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 0.6}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {
            "id": "strict",
            "prompt": "a",
            "rubric": "grade",
            "scorer": {"kind": "judge", "pass_threshold": 0.75},
        },
        {
            "id": "lenient",
            "prompt": "b",
            "rubric": "grade",
            "scorer": {"kind": "judge", "pass_threshold": 0.5},
        },
        {
            "id": "unstated",
            "prompt": "c",
            "rubric": "grade",
            "scorer": {"kind": "judge"},
        },
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)

    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    by_task = {}
    for row in client.app.state.db.execute(
        """SELECT g.task_id, s.passed, s.score FROM scores s
           JOIN results r ON r.id = s.result_id
           JOIN runs ru ON ru.id = r.run_id
           JOIN groups g ON g.id = ru.group_id
           WHERE g.experiment_id = ?""",
        (eid,),
    ):
        by_task[row["task_id"]] = dict(row)

    # The same 0.6 verdict, three different answers, all of them the
    # rubric author's own.
    assert by_task["strict"]["passed"] == 0
    assert by_task["lenient"]["passed"] == 1
    assert by_task["unstated"]["passed"] is None
    assert all(r["score"] == 0.6 for r in by_task.values())


# ---- Phase I4: blind human rating on replay.

from bench import store


@respx.mock
def rated_group(client):
    """A two-model comparison with its result ids, ready to rate."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    group_id = client.post("/groups", json={"budget": "standard"}).json()["id"]
    for model in ("model/alpha", "model/beta"):
        stream_events(
            client,
            {"prompt": "rate me", "model": model, "group_id": group_id},
        )
    detail = client.get(f"/groups/{group_id}").json()
    ids = [r["id"] for run in detail["runs"] for r in run["results"]]
    return group_id, ids


@respx.mock
def test_ratings_persist_as_human_scores_with_their_conditions(client):
    """The rating, the scale point the rater actually clicked, and the
    label the card wore. All three, because the normalized score alone is
    a number no rater ever saw and a blind nobody can audit."""
    group_id, ids = rated_group(client)

    resp = client.post(
        f"/groups/{group_id}/ratings",
        json={
            "blind": True,
            "ratings": [
                {"result_id": ids[0], "rating": 5, "label": "B"},
                {"result_id": ids[1], "rating": 1, "label": "A"},
            ],
        },
    )

    assert resp.status_code == 201
    assert resp.json()["written"] == 2
    rows = store.scores_for_results(client.app.state.db, ids)
    assert rows[ids[0]][0]["score"] == 1.0
    assert rows[ids[0]][0]["detail"] == "5 of 5, shown as B"
    assert rows[ids[1]][0]["score"] == 0.0
    assert rows[ids[1]][0]["detail"] == "1 of 5, shown as A"
    for result_id in ids:
        row = rows[result_id][0]
        assert row["scorer"] == "human"
        assert row["blind"] is True
        # No pass verdict: nobody said what a passing rating is, and the
        # bench inventing one would assert a cutoff the rater never chose.
        assert row["passed"] is None


@respx.mock
def test_the_middle_of_the_scale_normalizes_to_the_middle(client):
    group_id, ids = rated_group(client)

    client.post(
        f"/groups/{group_id}/ratings",
        json={"blind": True, "ratings": [{"result_id": ids[0], "rating": 3}]},
    )

    row = store.scores_for_results(client.app.state.db, ids)[ids[0]][0]
    assert row["score"] == 0.5
    assert row["detail"] == "3 of 5"


@respx.mock
def test_a_rating_after_the_reveal_persists_as_not_blind(client):
    """Honesty about conditions, not a prohibition. A sighted rating is
    still a rating; what would be wrong is recording it as blind."""
    group_id, ids = rated_group(client)

    client.post(
        f"/groups/{group_id}/ratings",
        json={"blind": False, "ratings": [{"result_id": ids[0], "rating": 4}]},
    )

    assert (
        store.scores_for_results(client.app.state.db, ids)[ids[0]][0]["blind"] is False
    )


@respx.mock
def test_rating_a_result_from_another_comparison_is_refused(client):
    """Not paranoia about a hostile client: a stale tab holding an older
    comparison's result ids would otherwise write ratings onto the wrong
    models, and a silent misattribution is the worst outcome here."""
    group_id, ids = rated_group(client)
    other_id, other_ids = rated_group(client)

    resp = client.post(
        f"/groups/{group_id}/ratings",
        json={"blind": True, "ratings": [{"result_id": other_ids[0], "rating": 5}]},
    )

    assert resp.status_code == 422
    assert "are not in comparison" in resp.json()["detail"]
    assert "reload it" in resp.json()["detail"]


@respx.mock
def test_an_out_of_range_rating_is_refused(client):
    group_id, ids = rated_group(client)

    for value in (0, 6, -1):
        resp = client.post(
            f"/groups/{group_id}/ratings",
            json={"blind": True, "ratings": [{"result_id": ids[0], "rating": value}]},
        )
        assert resp.status_code == 422, value


@respx.mock
def test_unknown_rating_fields_are_rejected(client):
    group_id, ids = rated_group(client)

    resp = client.post(
        f"/groups/{group_id}/ratings",
        json={
            "blind": True,
            "ratings": [{"result_id": ids[0], "rating": 5, "confidence": "high"}],
        },
    )

    assert resp.status_code == 422


def test_rating_a_group_that_does_not_exist_404s(client):
    resp = client.post(
        "/groups/999999/ratings",
        json={"blind": True, "ratings": [{"result_id": 1, "rating": 3}]},
    )

    assert resp.status_code == 404


# ---- Phase I5: the report.


@respx.mock
def scored_experiment(client, tmp_path, *, fail_second=False):
    """A two-model, two-task, two-repeat experiment, run and scored."""
    calls = {"n": 0}

    def route(request):
        calls["n"] += 1
        if fail_second and calls["n"] == 2:
            return httpx.Response(500)
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {
            "id": "t1",
            "prompt": "a",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
        {
            "id": "t2",
            "prompt": "b",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
    )
    eid = client.post("/experiments", json=experiment_body(path, repeats=2)).json()[
        "id"
    ]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path)
    return eid, path


@respx.mock
def test_the_report_names_its_estimand_and_its_resampling_unit(client, tmp_path):
    """A number without its estimand is a number about nothing in
    particular, and an interval without its resampling unit does not say
    what it is an interval over."""
    eid, path = scored_experiment(client, tmp_path)

    report = client.get(f"/experiments/{eid}/report").json()

    assert report["estimand_mode"] == "routed_service"
    assert report["bootstrap"]["unit"] == "task"
    assert report["bootstrap"]["seed"] == main.REPORT_SEED
    assert report["bootstrap"]["resamples"] == 2000
    assert report["dataset_digest"]


@respx.mock
def test_a_failed_trial_stays_in_the_denominator(client, tmp_path):
    """Hand-computable, which is the point: eight trials, one of them
    errored, and the errored one scores zero rather than vanishing. A
    mean over survivors would read 1.0 and flatter the model that
    failed."""
    eid, path = scored_experiment(client, tmp_path, fail_second=True)

    report = client.get(f"/experiments/{eid}/report").json()

    total = sum(m["trials"]["planned"] for m in report["models"])
    assert total == 8
    errored = [m for m in report["models"] if m["trials"]["error"] == 1]
    assert len(errored) == 1
    hurt = errored[0]
    # Four trials for this model, one errored: three score 1.0 and the
    # failure scores 0.0, so the mean is 0.75 exactly.
    assert hurt["trials"]["planned"] == 4
    # Nothing was refused and nothing was skipped, so every planned trial
    # was also attempted and the two denominators coincide here. The
    # tombstone in tests/test_report.py is where they do not.
    assert hurt["trials"]["attempted"] == 4
    assert hurt["score"]["mean"] == pytest.approx(0.75)
    assert hurt["trials"]["failure_rate"] == pytest.approx(0.25)
    # The other model was untouched and reads 1.0, which is what makes
    # the difference attributable.
    clean = next(m for m in report["models"] if m["model"] != hurt["model"])
    assert clean["score"]["mean"] == pytest.approx(1.0)


@respx.mock
def test_the_two_axes_are_reported_separately(client, tmp_path):
    """A judge's malfunction is scoring coverage, not a model failure.
    The trial counts and the coverage counts are different numbers over
    the same trials, and neither leaks into the other."""
    eid, path = scored_experiment(client, tmp_path)

    report = client.get(f"/experiments/{eid}/report").json()

    for entry in report["models"]:
        scorer = entry["scorers"][0]
        # Axis one: every trial accounted for by outcome.
        counted = sum(
            entry["trials"][k]
            for k in ("done", "error", "refused", "stopped", "missing")
        )
        assert counted == entry["trials"]["planned"]
        # Axis two: every trial that produced a result accounted for by
        # scoring state. A trial that never ran is on neither axis, so
        # the population is the plan less what never ran.
        coverage = scorer["coverage"]
        assert (
            coverage["scored"] + coverage["scoring_failed"] + coverage["unscored"]
            == entry["trials"]["planned"] - entry["trials"]["missing"]
        )
        # The scoring-failure rate is its own number, which is also the
        # response_format revisit measurement.
        assert coverage["scoring_failure_rate"] == 0.0


@respx.mock
def test_pass_rates_ship_with_their_coverage_and_only_where_declared(client, tmp_path):
    """Pass rate where a threshold was declared, score mean otherwise,
    and the coverage number is what keeps a shrinking denominator
    visible."""
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 0.9}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            if json.loads(request.content)["max_tokens"] == JUDGE_MAX_TOKENS
            else httpx.Response(200, stream=alpha_stream())
        )
    )
    path = write_dataset(
        tmp_path,
        {
            "id": "declared",
            "prompt": "a",
            "rubric": "g",
            "scorer": {"kind": "judge", "pass_threshold": 0.5},
        },
        {"id": "silent", "prompt": "b", "rubric": "g", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    with_file = client.get(
        f"/experiments/{eid}/report", params={"dataset_path": path}
    ).json()
    scorer = with_file["models"][0]["scorers"][0]

    assert with_file["thresholds_source"] == "dataset_file"
    # One task declared a threshold, one did not: eligible counts only
    # the declared one, and the rate is over usable verdicts within it.
    assert scorer["pass_rate"]["eligible"] == 1
    assert scorer["pass_rate"]["usable_verdicts"] == 1
    assert scorer["pass_rate"]["passed"] == 1
    assert scorer["pass_rate"]["rate"] == 1.0
    # Both tasks still contribute to the mean.
    assert scorer["n"] == 2

    # Without the file the population is recovered from the score rows,
    # and the report says which witness it used.
    without = client.get(f"/experiments/{eid}/report").json()
    assert without["thresholds_source"] == "score_rows"
    assert without["models"][0]["scorers"][0]["mean"] == pytest.approx(0.9)

    # The two witnesses agree here, which is the case that matters: every
    # eligible task has a usable verdict, so the rows know everything the
    # file knows. The silent task is the control, and it stays out of the
    # population under BOTH witnesses: its rows carry passed None, which
    # is what the file said too.
    assert without["models"][0]["scorers"][0]["pass_rate"] == scorer["pass_rate"]


@respx.mock
def test_review_repro_pass_rates_survive_the_loss_of_the_dataset_file(client, tmp_path):
    """A verdict in the database is evidence, and the report was throwing
    it away.

    Every `passed` here was computed against the real threshold when the
    scoring pass ran, and every one of them is persisted. But ELIGIBILITY
    was read from the file and nowhere else, so a report built without
    the path published no pass rate at all: the numbers existed, the
    denominator was simply not looked for. Losing a dataset file is
    ordinary (moved, renamed, on the other laptop) and it should not
    silently delete a published measurement.

    So the rows witness their own threshold. `passed` is written from
    judged_pass, which returns None unless the task's author declared a
    cutoff, so a judge row with a non-None passed IS a record that a
    threshold existed.

    The honest limit is asserted here rather than only documented: t3 is
    declared and never scored, so no row witnesses it and the recovered
    eligible count is 2 where the file would say 3. A floor, and the
    report labels itself as such.
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 0.9}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            if json.loads(request.content)["max_tokens"] == JUDGE_MAX_TOKENS
            else httpx.Response(200, stream=alpha_stream())
        )
    )
    declared = {"kind": "judge", "pass_threshold": 0.5}
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "g", "scorer": declared},
        {"id": "t2", "prompt": "b", "rubric": "g", "scorer": declared},
        {"id": "t3", "prompt": "c", "rubric": "g", "scorer": declared},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path, judge_model="judge/one")
    # t3's rows deleted, standing in for the task nobody got round to
    # scoring. Its trial is untouched, so it is still in the plan and
    # still in the mean; only its witness is gone.
    client.app.state.db.execute(
        """DELETE FROM scores WHERE result_id IN (
               SELECT r.id FROM results r
               JOIN runs ru ON ru.id = r.run_id
               JOIN groups g ON g.id = ru.group_id
               WHERE g.task_id = 't3')"""
    )
    client.app.state.db.commit()

    report = client.get(f"/experiments/{eid}/report").json()

    scorer = report["models"][0]["scorers"][0]
    assert report["thresholds_source"] == "score_rows"
    # The rate itself is exact: those two verdicts were judged against
    # the real cutoff and 0.9 clears 0.5.
    assert scorer["pass_rate"]["rate"] == 1.0
    assert scorer["pass_rate"]["passed"] == 2
    assert scorer["pass_rate"]["usable_verdicts"] == 2
    # The floor. Three tasks declared a threshold; two left a row to say
    # so, and the third is invisible without the file.
    assert scorer["pass_rate"]["eligible"] == 2
    with_file = client.get(
        f"/experiments/{eid}/report", params={"dataset_path": path}
    ).json()
    assert with_file["models"][0]["scorers"][0]["pass_rate"]["eligible"] == 3
    assert with_file["thresholds_source"] == "dataset_file"


@respx.mock
def test_a_verdict_nobody_could_reach_never_creates_eligibility(client, tmp_path):
    """The edge that keeps the fallback from inventing a population.

    A judge row with `passed` None is exactly what an undeclared task or
    an unparseable verdict leaves behind, and neither is evidence that a
    threshold existed. Counting it would manufacture an eligible
    denominator out of the scorer's own failures.
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {"message": {"content": "no idea"}, "finish_reason": "stop"}
                    ],
                },
            )
            if json.loads(request.content)["max_tokens"] == JUDGE_MAX_TOKENS
            else httpx.Response(200, stream=alpha_stream())
        )
    )
    path = write_dataset(
        tmp_path,
        {
            "id": "t1",
            "prompt": "a",
            "rubric": "g",
            "scorer": {"kind": "judge", "pass_threshold": 0.5},
        },
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path, judge_model="judge/one")

    report = client.get(f"/experiments/{eid}/report").json()

    scorer = report["models"][0]["scorers"][0]
    # The verdict was unparseable, so passed is None and the row proves
    # nothing about a threshold.
    assert scorer["coverage"]["scoring_failed"] == 1
    assert scorer["pass_rate"]["eligible"] == 0
    assert scorer["pass_rate"]["rate"] is None


@respx.mock
def test_a_deterministic_pass_never_creates_eligibility(client, tmp_path):
    """The other half of the same edge, and the reason judge_model gates
    the witness.

    A deterministic scorer writes `passed` unconditionally: it is the
    score restated, not a threshold anybody declared, and the loader
    permits pass_threshold only on judge. If those rows witnessed, the
    same experiment would report a different eligible population with and
    without a path, which is two answers from one dataset.
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: httpx.Response(200, stream=alpha_stream())
    )
    path = write_dataset(
        tmp_path,
        {
            "id": "t1",
            "prompt": "a",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path)

    with_file = client.get(
        f"/experiments/{eid}/report", params={"dataset_path": path}
    ).json()
    without = client.get(f"/experiments/{eid}/report").json()

    assert with_file["models"][0]["scorers"][0]["pass_rate"]["eligible"] == 0
    assert without["models"][0]["scorers"][0]["pass_rate"]["eligible"] == 0


@respx.mock
def test_the_report_surfaces_the_self_judged_flag(client, tmp_path):
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 1.0}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            if json.loads(request.content)["max_tokens"] == JUDGE_MAX_TOKENS
            else httpx.Response(200, stream=alpha_stream())
        )
    )
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "g", "scorer": {"kind": "judge"}},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path, judge_model="model/alpha")

    report = client.get(f"/experiments/{eid}/report").json()

    scorer = report["models"][0]["scorers"][0]
    assert scorer["self_judged"] == 1
    assert scorer["judge_models"] == ["model/alpha"]


@respx.mock
def test_human_ratings_appear_as_their_own_scorer_row_with_the_blind_flag(
    client, tmp_path
):
    """Human ratings are scores like any other, and the condition they
    were made under travels with them."""
    eid, path = scored_experiment(client, tmp_path)
    group_id = client.app.state.db.execute(
        "SELECT id FROM groups WHERE experiment_id = ? ORDER BY id LIMIT 1", (eid,)
    ).fetchone()["id"]
    detail = client.get(f"/groups/{group_id}").json()
    ids = [r["id"] for run in detail["runs"] for r in run["results"]]
    client.post(
        f"/groups/{group_id}/ratings",
        json={
            "blind": True,
            "ratings": [{"result_id": i, "rating": 4, "label": "A"} for i in ids],
        },
    )

    report = client.get(f"/experiments/{eid}/report").json()

    human = [
        s for m in report["models"] for s in m["scorers"] if s["scorer"] == "human"
    ]
    assert human, [s["scorer"] for m in report["models"] for s in m["scorers"]]
    assert all(row["blind"] >= 1 for row in human)
    assert all(row["mean"] == pytest.approx(0.75) for row in human)


@respx.mock
def test_the_report_shows_which_providers_actually_served(client, tmp_path):
    """The routed-service estimand made visible: under dynamic routing
    the provider is chosen per call, so it is the largest confound in any
    comparison and counting it turns an assumption into a column."""
    eid, path = scored_experiment(client, tmp_path)

    report = client.get(f"/experiments/{eid}/report").json()

    # The column exists on every model. The counts are empty here
    # because the browser stub does not name a provider, which is a
    # limit of the fixture rather than of the code; the counting itself
    # is asserted in tests/test_report.py where the input is controlled.
    assert all(isinstance(m["providers"], dict) for m in report["models"])


@respx.mock
def test_reporting_against_a_changed_dataset_is_refused(client, tmp_path):
    eid, path = scored_experiment(client, tmp_path)
    Path(path).write_text('{"id": "t1", "prompt": "different"}\n', encoding="utf-8")

    resp = client.get(f"/experiments/{eid}/report", params={"dataset_path": path})

    assert resp.status_code == 422
    assert "attribute one rubric's threshold to another" in resp.json()["detail"]


def test_a_report_for_a_missing_experiment_404s(client):
    assert client.get("/experiments/999999/report").status_code == 404


# ---- Phase I6: the export artifact.

import hashlib

from bench.report import build_report


@respx.mock
def two_axes_experiment(client, tmp_path):
    """An experiment shaped so the two-axes bug would corrupt its numbers.

    Two tasks, two models. The second upstream call errors, so one trial
    is an axis-one failure. Only the first task carries a scorer, so the
    completed trials of the second task are axis-two unscored. A mean
    that confused the two would read differently from the correct one,
    which is what makes this the right shape to re-derive from an export.
    """
    calls = {"n": 0}

    def route(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500)
        return httpx.Response(200, stream=alpha_stream())

    respx.post(OPENROUTER_URL).mock(side_effect=route)
    path = write_dataset(
        tmp_path,
        {
            "id": "scored",
            "prompt": "a",
            "reference": "Hello",
            "scorer": {"kind": "contains"},
        },
        # No scorer at all, so its trials stay unscored: the axis-two
        # half of the shape.
        {"id": "unscored", "prompt": "b"},
    )
    eid = client.post("/experiments", json=experiment_body(path)).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path)
    return eid, path


def read_export(client, eid, dataset_path=None):
    params = {} if dataset_path is None else {"dataset_path": dataset_path}
    with client.stream(
        "GET", f"/experiments/{eid}/export.jsonl", params=params
    ) as resp:
        assert resp.status_code == 200
        raw = resp.read()
    return raw


def rebuild_from_export(lines, tasks_by_id):
    """Run the report's own pure function over an export and nothing else.

    The whole self-sufficiency claim in one helper: if the artifact
    carries what it says it carries, this reproduces the served numbers
    with no database in reach. tasks_by_id is what the caller could
    recover from the manifest, so passing None here is not a shortcut,
    it is the pathless artifact's actual situation.
    """
    manifest = lines[0]
    groups, runs_by_group, scores_by_result = [], {}, {}
    for trial in (x for x in lines if x["type"] == "trial"):
        gid = trial["group_id"]
        if gid not in runs_by_group:
            groups.append(
                {
                    "id": gid,
                    "task_id": trial["task_id"],
                    "repeat_index": trial["repeat_index"],
                    "rotation_index": trial["rotation_index"],
                }
            )
            runs_by_group[gid] = []
        run = next((r for r in runs_by_group[gid] if r["id"] == trial["run_id"]), None)
        if run is None:
            run = {
                "id": trial["run_id"],
                "data_policy": trial["data_policy"],
                "app_sha": trial["app_sha"],
                "catalog_digest": trial["catalog_digest"],
                "prompt_text": trial["prompt_text"],
                "results": [],
            }
            runs_by_group[gid].append(run)
        run["results"].append(
            {
                "id": trial["result_id"],
                "model": trial["model"],
                "error": trial["error"],
                "request_json": trial["request_json"],
                "response_text": trial["response_text"],
                "position": trial["position"],
                "latency_ms": trial["latency_ms"],
                "ttft_ms": trial["ttft_ms"],
                "cost_usd": trial["cost_usd"],
                "billed_cost_usd": trial["billed_cost_usd"],
                "provider": trial["provider"],
            }
        )
        scores_by_result[trial["result_id"]] = trial["scores"]
    return build_report(
        {
            "id": manifest["experiment_id"],
            "name": manifest["name"],
            "estimand_mode": manifest["estimand_mode"],
            "dataset_name": manifest["dataset_name"],
            "dataset_digest": manifest["dataset_digest"],
            "repeats": manifest["repeats"],
            "status": manifest["status"],
            "status_detail": manifest["status_detail"],
            "lineup": manifest["lineup"],
        },
        groups,
        runs_by_group,
        scores_by_result,
        tasks_by_id,
        manifest["report_seed"],
    )


def tasks_from_manifest(manifest):
    """The manifest's threshold slice, in the shape build_report reads.

    None when the export was made without a file, which is the whole
    point of thresholds_included: an empty slice and an absent one are
    different artifacts and license different claims.
    """
    if not manifest["thresholds_included"]:
        return None
    return {tid: {"scorer": spec} for tid, spec in manifest["thresholds"].items()}


def export_lines(client, eid, dataset_path=None):
    return [
        json.loads(x)
        for x in read_export(client, eid, dataset_path).decode().strip().split("\n")
    ]


def threshold_experiment(client, tmp_path):
    """Three declared tasks, one of them left without a witness.

    Every task declares the same cutoff, so the file's eligible count is
    three. t3's score rows are then deleted, standing in for the task a
    scoring pass never reached: its trial is untouched, so it stays in
    the plan and in the mean, and only its evidence of a threshold is
    gone. That gap is the whole difference between the exact denominator
    and the floor, which is what makes this the right shape to export
    both ways.
    """
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            httpx.Response(
                200,
                json={
                    "id": "g",
                    "choices": [
                        {
                            "message": {"content": '{"score": 0.9}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            if json.loads(request.content)["max_tokens"] == JUDGE_MAX_TOKENS
            else httpx.Response(200, stream=alpha_stream())
        )
    )
    declared = {"kind": "judge", "pass_threshold": 0.5}
    path = write_dataset(
        tmp_path,
        {"id": "t1", "prompt": "a", "rubric": "g", "scorer": declared},
        {"id": "t2", "prompt": "b", "rubric": "g", "scorer": declared},
        {"id": "t3", "prompt": "c", "rubric": "g", "scorer": declared},
    )
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]
    run_experiment_to_completion(client, eid, path)
    score_experiment_to_completion(client, eid, path, judge_model="judge/one")
    client.app.state.db.execute(
        """DELETE FROM scores WHERE result_id IN (
               SELECT r.id FROM results r
               JOIN runs ru ON ru.id = r.run_id
               JOIN groups g ON g.id = ru.group_id
               WHERE g.task_id = 't3')"""
    )
    client.app.state.db.commit()
    return eid, path


@respx.mock
def test_review_repro_two_exports_of_one_experiment_are_byte_identical(
    client, tmp_path
):
    """A citation names an artifact. An artifact that differs between two
    honest exports of the same data cannot be checked against the
    citation, so the digest would be decoration.

    Line order is fixed by the query (task, repeat, position); key order
    within a line is fixed by sort_keys. Both are needed, and this
    asserts the pair rather than either alone.
    """
    eid, path = two_axes_experiment(client, tmp_path)

    first = read_export(client, eid)
    second = read_export(client, eid)

    assert first == second
    # Including the digest line, which is the part a citation quotes.
    assert (
        json.loads(first.decode().strip().split("\n")[-1])["digest"]
        == (json.loads(second.decode().strip().split("\n")[-1])["digest"])
    )

    # And with the file supplied, because the threshold slice is a new
    # mapping in the manifest and a mapping is exactly the thing whose
    # iteration order could vary between two honest exports. sort_keys
    # covers it; this is what says so.
    pathed = read_export(client, eid, path)
    assert pathed == read_export(client, eid, path)
    # The two modes differ, which is the point of labeling them: an
    # export that carried the thresholds and one that did not are
    # different artifacts and must not share a digest.
    assert pathed != first


@respx.mock
def test_the_export_verifies_its_own_digest(client, tmp_path):
    """Computed over the bytes as written, so the file checks itself with
    no second pass and no sidecar to lose."""
    eid, path = two_axes_experiment(client, tmp_path)

    raw = read_export(client, eid).decode()
    lines = raw.split("\n")[:-1]
    body = "\n".join(lines[:-1]) + "\n"
    trailer = json.loads(lines[-1])

    assert trailer["type"] == "digest"
    assert trailer["algorithm"] == "sha256"
    assert trailer["digest"] == hashlib.sha256(body.encode("utf-8")).hexdigest()


@respx.mock
def test_the_export_is_ordered_and_manifested(client, tmp_path):
    eid, path = two_axes_experiment(client, tmp_path)

    lines = [
        json.loads(x) for x in read_export(client, eid).decode().strip().split("\n")
    ]

    manifest = lines[0]
    assert manifest["type"] == "manifest"
    assert manifest["export_schema_version"] == 1
    assert manifest["estimand_mode"] == "routed_service"
    assert manifest["dataset_digest"]
    assert manifest["bootstrap_unit"] == "task"
    trials = [x for x in lines if x["type"] == "trial"]
    # Deterministic ordering: task, then repeat, then position.
    keys = [(t["task_id"], t["repeat_index"], t["position"]) for t in trials]
    assert keys == sorted(keys)


@respx.mock
def test_review_repro_the_export_alone_re_derives_a_two_axes_aggregate(
    client, tmp_path
):
    """The self-sufficiency claim, tested on the number most likely to be
    wrong rather than on a lucky one.

    The experiment has an errored trial (axis one) and completed trials
    nobody scored (axis two). The score mean over it is only correct if
    the export carried enough to keep those apart: the outcome per trial,
    and the score rows with their nulls intact. An export that flattened
    either would re-derive a different mean here while still matching on
    an experiment where everything succeeded.

    The four rows the fixture actually produces, which is worth writing
    down because an earlier description of this fixture named three
    trials for model/beta when it has two:

        model/alpha  task=scored    done   contains=1.0
        model/beta   task=scored    error  contains=0.0
        model/alpha  task=unscored  done   (no score rows)
        model/beta   task=unscored  done   (no score rows)

    Both wrong directions, as numbers, so the claim that this fixture
    discriminates is checkable rather than asserted:

        correct        alpha [1.0]      -> 1.0    beta [0.0]      -> 0.0
        A: unscored=0  alpha [1.0, 0.0] -> 0.5    beta [0.0, 0.0] -> 0.0
        B: drop failed alpha [1.0]      -> 1.0    beta []         -> None

    So each direction is caught by one model and not the other: A moves
    model/alpha and leaves model/beta at 0.0, B empties model/beta and
    leaves model/alpha at 1.0. The pair is load-bearing, and asserting
    only one of the two means would let one direction through.

    Re-derived by feeding the export's own rows back through the same
    pure function the report uses, which is what makes "self-sufficient"
    a check rather than a claim.
    """
    eid, path = two_axes_experiment(client, tmp_path)
    served = client.get(f"/experiments/{eid}/report").json()

    lines = export_lines(client, eid)

    # The export carried no threshold slice, so the rebuild gets None and
    # recovers eligibility from the rows, exactly as the served report
    # did. Passing {} instead would tell build_report a file had been
    # read and declared nothing, which is a different claim.
    rebuilt = rebuild_from_export(lines, None)

    # The shape is the one the bug would have corrupted: an errored trial
    # and unscored completed trials, both present.
    outcomes = {m["model"]: m["trials"] for m in served["models"]}
    assert sum(t["error"] for t in outcomes.values()) == 1
    coverage = [
        s["coverage"]
        for m in served["models"]
        for s in m["scorers"]
        if s["scorer"] == "contains"
    ]
    assert coverage and any(c["unscored"] > 0 for c in coverage)

    # Both discriminating values pinned, one per wrong direction. Each
    # model catches the direction the other is blind to, per the table in
    # the docstring.
    means = {
        m["model"]: (m["score"]["mean"], m["scorers"][0]["n"])
        for m in rebuilt["models"]
    }
    assert means["model/alpha"] == (1.0, 1)  # direction A would read 0.5
    assert means["model/beta"] == (0.0, 1)  # direction B would read None

    # And the export alone reproduces every model's mean and outcome
    # counts exactly.
    for served_model, rebuilt_model in zip(
        served["models"], rebuilt["models"], strict=True
    ):
        assert served_model["model"] == rebuilt_model["model"]
        assert served_model["trials"] == rebuilt_model["trials"]
        assert served_model["score"]["mean"] == rebuilt_model["score"]["mean"]
        assert (
            served_model["scorers"][0]["coverage"]
            == rebuilt_model["scorers"][0]["coverage"]
        )


@respx.mock
def test_review_repro_the_export_re_derives_the_pass_rate_it_says_it_can(
    client, tmp_path
):
    """The self-sufficiency claim, extended to the number it did not
    cover, in both of the modes the artifact can be in.

    Every verdict was already in the export and the rate was already
    re-derivable. ELIGIBILITY was not: it lived in the file, so an export
    could reproduce a model's mean exactly and be quietly unable to
    reproduce the denominator beside it. That made "self-sufficient"
    false for precisely the coverage figure the README puts next to every
    rate, and the round trip did not notice because its fixture declared
    no thresholds at all.

    Three declared tasks, one of them stripped of its score rows, so the
    exact denominator and the floor are different numbers here and a
    rebuild cannot pass both by accident:

        with the file / labeled export    eligible 3, usable 2, rate 1.0
        without       / pathless export   eligible 2, usable 2, rate 1.0
    """
    eid, path = threshold_experiment(client, tmp_path)
    with_file = client.get(
        f"/experiments/{eid}/report", params={"dataset_path": path}
    ).json()["models"][0]["scorers"][0]["pass_rate"]
    without = client.get(f"/experiments/{eid}/report").json()["models"][0]["scorers"][
        0
    ]["pass_rate"]
    assert with_file["eligible"] == 3
    assert without["eligible"] == 2

    labeled = export_lines(client, eid, path)
    pathless = export_lines(client, eid)

    # Each artifact says which it is, and neither has to be asked.
    assert labeled[0]["thresholds_included"] is True
    assert set(labeled[0]["thresholds"]) == {"t1", "t2", "t3"}
    assert labeled[0]["thresholds"]["t1"] == {"kind": "judge", "pass_threshold": 0.5}
    assert pathless[0]["thresholds_included"] is False
    assert pathless[0]["thresholds"] == {}
    # The slice is minimal: nothing that only a human reader would want.
    assert set(labeled[0]["thresholds"]["t1"]) == {"kind", "pass_threshold"}

    # The labeled export re-derives the exact denominator.
    from_labeled = rebuild_from_export(labeled, tasks_from_manifest(labeled[0]))
    assert from_labeled["models"][0]["scorers"][0]["pass_rate"] == with_file
    assert from_labeled["thresholds_source"] == "dataset_file"

    # The pathless one re-derives the floor, and matches the report that
    # was served under the same handicap. Both are honest; only one is
    # complete, and each says which.
    from_pathless = rebuild_from_export(pathless, tasks_from_manifest(pathless[0]))
    assert from_pathless["models"][0]["scorers"][0]["pass_rate"] == without
    assert from_pathless["thresholds_source"] == "score_rows"


@respx.mock
def test_an_export_against_the_wrong_dataset_streams_nothing(client, tmp_path):
    """Refused before a byte goes out, not part way through.

    Raising from inside the generator would already have sent 200 and a
    content-type, leaving the caller holding a truncated file that looks
    like a whole one. An artifact is cited; a half-written one is worse
    than none.
    """
    eid, path = threshold_experiment(client, tmp_path)
    wrong = write_dataset(tmp_path, {"id": "z", "prompt": "other"}, name="wrong.jsonl")

    resp = client.get(
        f"/experiments/{eid}/export.jsonl", params={"dataset_path": wrong}
    )

    assert resp.status_code == 422
    assert "dataset changed since this experiment was created" in resp.json()["detail"]
    # An error body, not an artifact: no ndjson content type and not one
    # manifest or trial line anywhere in what came back.
    assert not resp.headers["content-type"].startswith("application/x-ndjson")
    assert b"manifest" not in resp.content
    assert b'"type"' not in resp.content


@respx.mock
def test_the_export_is_never_stored_by_a_cache(client, tmp_path):
    """It carries every prompt and every answer in the experiment, which
    makes it the most sensitive body the bench produces."""
    eid, path = two_axes_experiment(client, tmp_path)

    with client.stream("GET", f"/experiments/{eid}/export.jsonl") as resp:
        assert resp.headers.get("cache-control") == "private, no-store"
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        resp.read()


def test_exporting_a_missing_experiment_404s(client):
    assert client.get("/experiments/999999/export.jsonl").status_code == 404


# ---- Phase I7: the underlying-model estimand.


def strict_body(path, **overrides):
    body = experiment_body(path, estimand_mode="underlying_model")
    body["lineup"] = ["model/alpha"]
    body.update(overrides)
    return body


def test_strict_mode_accepts_a_model_the_catalog_says_can_do_the_job(client, tmp_path):
    """The scope control for every refusal below. model/alpha publishes
    every parameter this request carries, so the check passes and the row
    records the estimand it was created under."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post(
        "/experiments",
        json=strict_body(path, params={"temperature": 0.4, "effort": "low"}),
    )

    assert resp.status_code == 201, resp.text
    detail = client.get(f"/experiments/{resp.json()['id']}").json()
    assert detail["estimand_mode"] == "underlying_model"


def test_strict_mode_refuses_a_control_the_model_does_not_support(client, tmp_path):
    """Under require_parameters an unsupported parameter is not ignored,
    it makes every provider ineligible. So the experiment would fail every
    trial, and the place to say so is before the first one."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post(
        "/experiments",
        json=strict_body(path, lineup=["model/capped"], params={"temperature": 0.4}),
    )

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "model/capped does not support temperature" in detail
    assert "require_parameters" in detail
    # Names the remedy, and names the routed-service behavior it is
    # different from, because "use the other estimand" is only actionable
    # if the caller knows what the other one does with the same request.
    assert "routed-service estimand" in detail


def test_strict_mode_refuses_a_model_the_catalog_does_not_describe(client, tmp_path):
    """Absence of evidence is not support. model/bare is in the catalog
    with no supported_parameters, which is a different fact from an empty
    list and must not be read as one."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post("/experiments", json=strict_body(path, lineup=["model/bare"]))

    assert resp.status_code == 422
    assert "publishes no supported_parameters" in resp.json()["detail"]


def test_strict_mode_refuses_a_model_the_catalog_never_listed(client, tmp_path):
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post("/experiments", json=strict_body(path, lineup=["model/ghost"]))

    assert resp.status_code == 422
    assert "does not list model/ghost" in resp.json()["detail"]


def test_review_repro_an_offline_boot_refuses_to_create_a_strict_experiment(
    monkeypatch, tmp_path
):
    """The check needs the catalog, so a boot without one cannot make the
    promise the estimand's name makes.

    Skipping the check silently would be the tempting version of this and
    the wrong one: the rows would carry estimand_mode underlying_model
    with nothing behind it, and no reader afterwards could tell those rows
    from the ones where the check ran. A label nobody verified is worse
    than no label. The refusal names the reason and both remedies.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BENCH_DB", str(tmp_path / "bench.db"))

    async def offline_catalog(client):
        return {"fetched": False, "models": [], "prices": {}, "digest": None}

    monkeypatch.setattr("bench.main.fetch_catalog", offline_catalog)
    with TestClient(app, base_url="http://localhost") as offline:
        path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

        strict = offline.post("/experiments", json=strict_body(path))
        routed = offline.post("/experiments", json=experiment_body(path))

    assert strict.status_code == 422
    detail = strict.json()["detail"]
    assert "this boot has no catalog" in detail
    assert "routed-service estimand, which makes no capability claim" in detail
    # And the default estimand is untouched by any of this: it makes no
    # capability claim, so it needs no catalog to make it.
    assert routed.status_code == 201, routed.text


def test_a_pin_for_a_model_outside_the_lineup_is_refused(client, tmp_path):
    """A declaration that cannot be honored should not be stored as
    though it will be."""
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})

    resp = client.post(
        "/experiments",
        json=strict_body(path, provider_pins={"model/other": "Together"}),
    )

    assert resp.status_code == 422
    assert "not in the lineup" in resp.json()["detail"]


@respx.mock
def test_strict_mode_sends_require_parameters_and_a_hard_pin(client, tmp_path):
    """What actually rides the wire, read off the recorded request.

    allow_fallbacks false travels with the order and never without it: an
    order that can be departed from is a preference, and a pinned run
    served by somebody else would record a constraint that did not hold.
    """
    sent = []
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            sent.append(json.loads(request.content)),
            httpx.Response(200, stream=alpha_stream()),
        )[1]
    )
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})
    eid = client.post(
        "/experiments",
        json=strict_body(path, provider_pins={"model/alpha": "Together"}),
    ).json()["id"]

    run_experiment_to_completion(client, eid, path)

    assert len(sent) == 1
    provider = sent[0]["provider"]
    assert provider["require_parameters"] is True
    assert provider["order"] == ["Together"]
    assert provider["allow_fallbacks"] is False
    # The privacy promise is written last and cannot be displaced by the
    # estimand: the boot policy's keys survive the merge.
    assert provider["sort"] == "throughput"


@respx.mock
def test_review_repro_the_routed_service_payload_is_unchanged_by_strict_mode(
    client, tmp_path
):
    """The promise adding an estimand had to keep.

    Byte for byte, not merely equal in the keys anybody thought to check.
    If strict mode had leaked into the default path, every comparison this
    bench has ever run would be measuring something different from the day
    before, and the change would be invisible in the results.
    """
    sent = []
    respx.post(OPENROUTER_URL).mock(
        side_effect=lambda request: (
            sent.append(request.content),
            httpx.Response(200, stream=alpha_stream()),
        )[1]
    )
    path = write_dataset(tmp_path, {"id": "t1", "prompt": "hi"})
    eid = client.post(
        "/experiments", json=experiment_body(path, lineup=["model/alpha"])
    ).json()["id"]

    run_experiment_to_completion(client, eid, path)

    assert len(sent) == 1
    payload = json.loads(sent[0])
    assert payload["provider"] == {"sort": "throughput"}
    assert "require_parameters" not in sent[0].decode()
    assert "allow_fallbacks" not in sent[0].decode()
