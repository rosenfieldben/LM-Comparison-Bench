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

# Hand-built catalog injected in place of the startup fetch. The
# fixture's usage block is 13 in / 8 out, so model/alpha costs
# 13 * 1e-6 + 8 * 2e-6 = 2.9e-5 USD.
TEST_PRICES = {
    "model/alpha": {"prompt": 1e-06, "completion": 2e-06},
}
TEST_CATALOG = {
    "fetched": True,
    "models": [
        {
            "id": "model/alpha",
            "name": "Alpha",
            "context_length": 128000,
            "prompt_price": 1e-06,
            "completion_price": 2e-06,
            "max_completion_tokens": None,
        },
        {
            "id": "model/bare",
            "name": None,
            "context_length": None,
            "prompt_price": None,
            "completion_price": None,
            "max_completion_tokens": None,
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

    group_id = client.post("/groups", json={}).json()["id"]
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


def test_review_repro_index_denies_framing(client):
    """External review finding 4: nothing stopped a hostile page from
    framing the localhost UI and redressing a Run click into paid work.
    Every response must carry anti-framing headers."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["content-security-policy"] == "frame-ancestors 'none'"


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
        assert resp.headers["content-security-policy"] == "frame-ancestors 'none'"
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

    group_id = client.post("/groups", json={}).json()["id"]
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
    assert client.post("/groups", json={}).status_code == 201


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
        assert body == {"models": [], "fetched": False}

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
    assert client.post("/groups", json={}).status_code == 201


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
        # The first model's run stands in for spend (from this batch or a
        # concurrent request) crossing the ceiling; the next must then be
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
