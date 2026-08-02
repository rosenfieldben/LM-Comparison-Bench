import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from bench.models import (
    MODELS_URL,
    OPENROUTER_URL,
    control_messages,
    control_payload,
    fetch_catalog,
    judge_response,
    run_model,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_response.json").read_text()
)


@pytest.fixture
async def client():
    # trust_env off so an ambient developer proxy cannot route test
    # traffic; respx intercepts regardless, and the poisoned-proxy
    # acceptance run (HTTPS_PROXY set) depends on this.
    async with httpx.AsyncClient(trust_env=False) as c:
        yield c


@respx.mock
async def test_happy_path_parses_text_and_tokens(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model(
        "Say hello in five words.", "deepseek/deepseek-chat", client
    )

    assert result["error"] is None
    assert result["model"] == "deepseek/deepseek-chat"
    assert result["response_text"] == "Hello there, friend, warm greetings!"
    assert result["prompt_tokens"] == 13
    assert result["completion_tokens"] == 8
    assert result["latency_ms"] >= 0


@respx.mock
async def test_timeout_returns_error_dict(client):
    respx.post(OPENROUTER_URL).mock(side_effect=httpx.ReadTimeout("slow upstream"))

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["response_text"] is None
    assert result["error"] == "no response within 300s"


@respx.mock
async def test_connect_timeout_names_the_connect_ceiling(client):
    # A connect timeout fires after CONNECT_TIMEOUT_S; the generic read
    # timeout message would overstate the wait eighteenfold.
    respx.post(OPENROUTER_URL).mock(side_effect=httpx.ConnectTimeout("handshake"))

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["response_text"] is None
    assert result["error"] == "could not connect within 10s"


@respx.mock
async def test_http_429_mentions_status(client):
    respx.post(OPENROUTER_URL).respond(status_code=429, json={"error": "rate limited"})

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["response_text"] is None
    assert "429" in result["error"]


@respx.mock
async def test_null_content_becomes_error_with_finish_reason(client):
    body = json.loads(json.dumps(FIXTURE))
    body["choices"][0]["message"]["content"] = None
    body["choices"][0]["finish_reason"] = "content_filter"
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["response_text"] is None
    assert result["error"] == "empty response (finish_reason: content_filter)"


@respx.mock
async def test_missing_usage_block_yields_none_tokens(client):
    body = {k: v for k, v in FIXTURE.items() if k != "usage"}
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["response_text"] == "Hello there, friend, warm greetings!"
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None


@respx.mock
async def test_content_parts_list_flattens_to_text(client):
    body = json.loads(json.dumps(FIXTURE))
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "text", "text": "world"},
    ]
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["response_text"] == "Hello world"


@respx.mock
async def test_content_parts_with_no_text_becomes_empty_response_error(client):
    body = json.loads(json.dumps(FIXTURE))
    body["choices"][0]["message"]["content"] = [
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    ]
    body["choices"][0]["finish_reason"] = "stop"
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["response_text"] is None
    assert result["error"] == "empty response (finish_reason: stop)"


@respx.mock
async def test_non_dict_usage_does_not_raise(client):
    body = json.loads(json.dumps(FIXTURE))
    body["usage"] = "n/a"
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["response_text"] == "Hello there, friend, warm greetings!"
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None


@respx.mock
async def test_the_catalog_digest_identifies_the_bytes_that_were_read(client):
    """catalog_snapshot_at says WHEN prices were read; the digest says
    WHICH bytes. Two boots a minute apart against a changed catalog were
    otherwise indistinguishable in the provenance of the runs they costed.

    It hashes the raw response body, before parsing, so it identifies what
    OpenRouter actually sent rather than what this code made of it.
    """
    body = {"data": [{"id": "a/one", "pricing": {"prompt": "1", "completion": "2"}}]}
    raw = json.dumps(body).encode()
    respx.get(MODELS_URL).respond(content=raw, headers={"content-type": "text/json"})

    catalog = await fetch_catalog(client)

    assert catalog["digest"] == hashlib.sha256(raw).hexdigest()


@respx.mock
async def test_catalog_price_map_skips_malformed_entries(client):
    respx.get(MODELS_URL).respond(
        json={
            "data": [
                {
                    "id": "a/one",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                },
                {"id": "b/broken", "pricing": {}},
                {
                    "id": "c/nonnumeric",
                    "pricing": {"prompt": "free", "completion": "0"},
                },
            ]
        }
    )

    catalog = await fetch_catalog(client)

    assert catalog["prices"] == {"a/one": {"prompt": 1e-06, "completion": 2e-06}}


@respx.mock
async def test_fetch_catalog_http_error_reports_not_fetched(client):
    respx.get(MODELS_URL).respond(status_code=500)
    catalog = await fetch_catalog(client)
    # digest is None rather than absent: an offline boot read no bytes, and
    # that absence is the provenance those runs get.
    assert catalog == {
        "fetched": False,
        "models": [],
        "prices": {},
        "digest": None,
    }


from stream_helpers import DONE_MARKER, ChunkStream, sse

from bench.models import stream_model


def delta_chunk(content):
    return sse({"choices": [{"delta": {"content": content}}]})


async def collect(prompt, model, client):
    return [e async for e in stream_model(prompt, model, client)]


@respx.mock
async def test_stream_accumulates_deltas_and_parses_usage(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    sse({"choices": [{"delta": {"role": "assistant"}}]}),
                    delta_chunk("Hel"),
                    delta_chunk("lo"),
                    # Content-parts delta: must flatten via the shared rule.
                    delta_chunk([{"type": "text", "text": " world"}]),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    sse(
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 13, "completion_tokens": 8},
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert [e["text"] for e in events[:-1]] == ["Hel", "lo", " world"]
    result = events[-1]["result"]
    assert events[-1]["type"] == "done"
    assert result["response_text"] == "Hello world"
    assert result["prompt_tokens"] == 13
    assert result["completion_tokens"] == 8
    assert result["error"] is None
    assert result["ttft_ms"] is not None
    assert result["latency_ms"] is not None
    assert result["ttft_ms"] <= result["latency_ms"]


@respx.mock
async def test_stream_mid_disconnect_yields_error_with_partial_text(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream([delta_chunk("Hel")], exc=httpx.ReadError("net down")),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert [e["type"] for e in events] == ["delta", "done"]
    result = events[-1]["result"]
    assert result["response_text"] == "Hel"
    assert result["error"] == "request failed: ReadError"
    assert result["ttft_ms"] is not None


@respx.mock
async def test_stream_stall_yields_timeout_error(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream([delta_chunk("Hel")], exc=httpx.ReadTimeout("silent")),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "Hel"
    assert result["error"] == "stream stalled: no data for 300s"


@respx.mock
async def test_stream_connect_timeout_is_not_reported_as_a_stall(client):
    respx.post(OPENROUTER_URL).mock(side_effect=httpx.ConnectTimeout("handshake"))

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert len(events) == 1
    result = events[0]["result"]
    assert result["error"] == "could not connect within 10s"
    assert result["response_text"] is None
    assert result["ttft_ms"] is None


@respx.mock
async def test_stream_instant_http_error_has_no_ttft(client):
    respx.post(OPENROUTER_URL).respond(status_code=429, json={"error": "rate limited"})

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert len(events) == 1
    result = events[0]["result"]
    assert result["error"] == "HTTP 429 from OpenRouter"
    assert result["response_text"] is None
    assert result["ttft_ms"] is None


@respx.mock
async def test_stream_malformed_chunk_yields_error_with_partial_text(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hel"),
                    b"data: {not json\n\n",
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "Hel"
    assert result["error"] == "malformed stream from OpenRouter"


@respx.mock
async def test_null_text_content_part_does_not_raise(client):
    body = json.loads(json.dumps(FIXTURE))
    body["choices"][0]["message"]["content"] = [{"type": "text", "text": None}]
    body["choices"][0]["finish_reason"] = "stop"
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["response_text"] is None
    assert result["error"] == "empty response (finish_reason: stop)"


@respx.mock
async def test_stream_null_text_content_part_does_not_raise(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hi "),
                    delta_chunk([{"type": "text", "text": None}]),
                    delta_chunk([{"type": "text", "text": "there"}]),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    result = events[-1]["result"]
    assert result["response_text"] == "Hi there"
    assert result["error"] is None


@respx.mock
async def test_stream_error_frame_surfaces_upstream_message(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hel"),
                    sse({"error": {"code": 502, "message": "upstream model crashed"}}),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["error"] == "upstream error: upstream model crashed"
    assert result["response_text"] == "Hel"


@respx.mock
async def test_stream_error_frame_with_code_only(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream([sse({"error": {"code": 429}}), DONE_MARKER]),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["error"] == "upstream error: 429"
    assert result["response_text"] is None
    assert result["ttft_ms"] is None


@respx.mock
async def test_fetch_catalog_degrades_missing_fields_to_none(client):
    respx.get(MODELS_URL).respond(
        json={
            "data": [
                {
                    "id": "a/full",
                    "name": "Full Model",
                    "context_length": 32000,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "top_provider": {"max_completion_tokens": 32000},
                },
                {"id": "b/bare"},
                {
                    "id": "c/badprice",
                    "name": "Bad Price",
                    "pricing": {"prompt": "free"},
                },
                {"id": "d/badcap", "top_provider": {"max_completion_tokens": "lots"}},
                {"no_id": True},
            ]
        }
    )

    catalog = await fetch_catalog(client)

    assert catalog["fetched"] is True
    assert [m["id"] for m in catalog["models"]] == [
        "a/full",
        "b/bare",
        "c/badprice",
        "d/badcap",
    ]
    full, bare, badprice, badcap = catalog["models"]
    assert full["context_length"] == 32000
    assert full["prompt_price"] == 1e-06
    assert full["max_completion_tokens"] == 32000
    assert bare["name"] is None
    assert bare["context_length"] is None
    assert bare["prompt_price"] is None
    assert bare["max_completion_tokens"] is None
    assert badprice["name"] == "Bad Price"
    assert badprice["prompt_price"] is None
    # A non-integer published cap degrades to None instead of poisoning
    # the clamp with a string.
    assert badcap["max_completion_tokens"] is None
    # The price map only carries fully priced entries.
    assert set(catalog["prices"]) == {"a/full"}


@respx.mock
async def test_fetch_catalog_failure_reports_not_fetched(client):
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("offline"))
    catalog = await fetch_catalog(client)
    # digest is None rather than absent: an offline boot read no bytes, and
    # that absence is the provenance those runs get.
    assert catalog == {
        "fetched": False,
        "models": [],
        "prices": {},
        "digest": None,
    }


from bench.models import BUDGET_STANDARD


@respx.mock
async def test_run_model_defaults_to_standard_budget(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=FIXTURE)

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert seen["max_tokens"] == BUDGET_STANDARD
    assert result["max_tokens"] == BUDGET_STANDARD


@respx.mock
async def test_stream_model_defaults_to_standard_budget(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, stream=ChunkStream([delta_chunk("hi"), DONE_MARKER]))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert seen["max_tokens"] == BUDGET_STANDARD
    assert events[-1]["result"]["max_tokens"] == BUDGET_STANDARD


@respx.mock
async def test_run_model_sends_and_echoes_explicit_max_tokens(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=FIXTURE)

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    result = await run_model("hi", "deepseek/deepseek-chat", client, max_tokens=32000)

    assert seen["max_tokens"] == 32000
    assert result["max_tokens"] == 32000


import socket

from bench.models import (
    KEEPALIVE_IDLE_S,
    KEEPALIVE_INTERVAL_S,
    keepalive_socket_options,
)


def test_keepalive_options_enable_probes_for_this_platform():
    opts = keepalive_socket_options()

    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts

    # Exactly one idle-time option, using whichever constant this
    # platform names it by (TCP_KEEPIDLE on Linux, TCP_KEEPALIVE on
    # macOS). Both major platforms name one, so zero would mean the
    # probes silently never start.
    idle_consts = {
        getattr(socket, name)
        for name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE")
        if hasattr(socket, name)
    }
    idle_opts = [o for o in opts if o[0] == socket.IPPROTO_TCP and o[1] in idle_consts]
    assert len(idle_opts) == 1
    assert idle_opts[0][2] == KEEPALIVE_IDLE_S

    assert (
        socket.IPPROTO_TCP,
        socket.TCP_KEEPINTVL,
        KEEPALIVE_INTERVAL_S,
    ) in opts


from bench.models import PROVIDER_PREFS


@respx.mock
async def test_run_model_sends_provider_prefs(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=FIXTURE)

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    await run_model("hi", "deepseek/deepseek-chat", client)

    assert seen["provider"] == PROVIDER_PREFS


@respx.mock
async def test_stream_model_sends_provider_prefs(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, stream=ChunkStream([delta_chunk("hi"), DONE_MARKER]))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    await collect("hi", "deepseek/deepseek-chat", client)

    assert seen["provider"] == PROVIDER_PREFS


from bench.models import COMPLETION_READ_TIMEOUT_S, STREAM_READ_TIMEOUT_S


@respx.mock
async def test_run_model_uses_completion_read_timeout(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, json=FIXTURE)

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    await run_model("hi", "deepseek/deepseek-chat", client)

    assert seen["read"] == COMPLETION_READ_TIMEOUT_S
    assert seen["connect"] == 10.0


@respx.mock
async def test_stream_model_uses_stream_read_timeout(client):
    seen = {}

    def route(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, stream=ChunkStream([delta_chunk("hi"), DONE_MARKER]))

    respx.post(OPENROUTER_URL).mock(side_effect=route)

    await collect("hi", "deepseek/deepseek-chat", client)

    assert seen["read"] == STREAM_READ_TIMEOUT_S
    assert seen["connect"] == 10.0


@respx.mock
async def test_run_model_captures_generation_id_and_finish_reason(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["generation_id"] == "gen-1751500123-Xk3mQpR7vNwB2aZd"
    assert result["finish_reason"] == "stop"


@respx.mock
async def test_run_model_missing_provenance_degrades_to_none(client):
    # A response with no top-level id and a null finish_reason must not
    # invent provenance; both fields stay honestly unknown.
    body = json.loads(json.dumps(FIXTURE))
    del body["id"]
    body["choices"][0]["finish_reason"] = None
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["response_text"] is not None
    assert result["generation_id"] is None
    assert result["finish_reason"] is None


@respx.mock
async def test_run_model_non_string_provenance_degrades_to_none(client):
    body = json.loads(json.dumps(FIXTURE))
    body["id"] = 12345
    body["choices"][0]["finish_reason"] = 3
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["generation_id"] is None
    assert result["finish_reason"] is None


@respx.mock
async def test_stream_takes_first_generation_id_and_final_finish_reason(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    # First chunk carries no id: the field must wait for one
                    # rather than lock in None.
                    sse({"choices": [{"delta": {"role": "assistant"}}]}),
                    sse(
                        {"id": "gen-first", "choices": [{"delta": {"content": "Hel"}}]}
                    ),
                    sse({"id": "gen-later", "choices": [{"delta": {"content": "lo"}}]}),
                    sse(
                        {
                            "id": "gen-later",
                            "choices": [{"delta": {}, "finish_reason": "length"}],
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "Hello"
    assert result["generation_id"] == "gen-first"
    assert result["finish_reason"] == "length"


@respx.mock
async def test_stream_missing_provenance_degrades_to_none(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, stream=ChunkStream([delta_chunk("hi"), DONE_MARKER])
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "hi"
    assert result["generation_id"] is None
    assert result["finish_reason"] is None


from bench.models import as_token_count

# The junk menu from the review: numeric strings, floats, negatives,
# bools (an int subclass), containers. Only genuine non-negative ints
# survive.
JUNK_TOKEN_VALUES = ["n/a", "12", 3.7, -5, True, {}, []]


@pytest.mark.parametrize("junk", JUNK_TOKEN_VALUES)
def test_as_token_count_rejects_junk(junk):
    assert as_token_count(junk) is None


@pytest.mark.parametrize("valid", [0, 8, 16384])
def test_as_token_count_accepts_real_counts(valid):
    assert as_token_count(valid) == valid


@pytest.mark.parametrize("junk", JUNK_TOKEN_VALUES)
@respx.mock
async def test_run_model_normalizes_junk_token_counts(client, junk):
    body = json.loads(json.dumps(FIXTURE))
    body["usage"] = {"prompt_tokens": junk, "completion_tokens": junk}
    respx.post(OPENROUTER_URL).respond(json=body)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None


@pytest.mark.parametrize("junk", JUNK_TOKEN_VALUES)
@respx.mock
async def test_stream_model_normalizes_junk_token_counts(client, junk):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("hi"),
                    sse(
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": junk, "completion_tokens": junk},
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "hi"
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None


@respx.mock
async def test_review_repro_nonfinite_and_negative_prices_degrade(client):
    """External review finding 3: catalog pricing predated as_metric's
    finiteness check, so a NaN or inf price string passed the bare
    float()/try-except, produced a NaN cost, and a NaN summed into
    accumulated spend made the ceiling comparison permanently false,
    silently disabling it. Non-finite and negative prices must degrade the
    entry to unpriced while the entry itself stays in the catalog."""
    respx.get(MODELS_URL).respond(
        json={
            "data": [
                {"id": "a/nan", "pricing": {"prompt": "NaN", "completion": "0.000002"}},
                {"id": "b/inf", "pricing": {"prompt": "0.000001", "completion": "inf"}},
                {
                    "id": "c/neg",
                    "pricing": {"prompt": "-0.000001", "completion": "0.000002"},
                },
                {
                    "id": "d/ok",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                },
            ]
        }
    )

    catalog = await fetch_catalog(client)

    # All four entries survive (degrade philosophy); only the finite,
    # non-negative one is priced.
    assert [m["id"] for m in catalog["models"]] == ["a/nan", "b/inf", "c/neg", "d/ok"]
    assert set(catalog["prices"]) == {"d/ok"}
    nan_m, inf_m, neg_m, ok_m = catalog["models"]
    assert nan_m["prompt_price"] is None and nan_m["completion_price"] is None
    assert inf_m["prompt_price"] is None and inf_m["completion_price"] is None
    assert neg_m["prompt_price"] is None and neg_m["completion_price"] is None
    assert ok_m["prompt_price"] == 1e-06


@respx.mock
async def test_review_repro_boolean_and_nonpositive_completion_cap_ignored(client):
    """External review finding 3: max_completion_tokens accepted any int,
    and isinstance(True, int) is true, so a provider sending true became a
    cap of 1 that clamped every budget to a single token. Only a non-bool
    integer strictly above zero is a real cap."""
    respx.get(MODELS_URL).respond(
        json={
            "data": [
                {"id": "a/booltrue", "top_provider": {"max_completion_tokens": True}},
                {"id": "b/zero", "top_provider": {"max_completion_tokens": 0}},
                {"id": "c/neg", "top_provider": {"max_completion_tokens": -5}},
                {"id": "d/real", "top_provider": {"max_completion_tokens": 32000}},
            ]
        }
    )

    catalog = await fetch_catalog(client)

    booltrue, zero, neg, real = catalog["models"]
    assert booltrue["max_completion_tokens"] is None
    assert zero["max_completion_tokens"] is None
    assert neg["max_completion_tokens"] is None
    assert real["max_completion_tokens"] == 32000


@respx.mock
async def test_review_repro_clean_eof_without_done_is_error(client):
    """External review finding 2: stream_model broke on [DONE] without
    recording that it saw the marker, so any ordinary iterator exhaustion
    (a load balancer idle-closing the connection, a provider crashing with
    a clean close) fell through to done(None) and a truncated answer was
    shown and persisted as complete. A clean EOF with no [DONE] and no
    finish_reason must be an error, with the partial text preserved."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, stream=ChunkStream([delta_chunk("Hel")]))
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "Hel"
    assert (
        result["error"]
        == "stream ended before completion: no [DONE] and no finish reason"
    )


@respx.mock
async def test_review_repro_finish_reason_without_done_not_flagged(client):
    """External review finding 2 refinement (inverse lock): a stream that
    delivered a finish_reason and then closed without [DONE] is
    semantically complete (the provider stated why it stopped), so flagging
    it aborted would be a false alarm. Guards against over-correcting the
    fix above into treating every missing [DONE] as an error."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hello"),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    # No [DONE]: the connection simply closes after the
                    # provider stated why it stopped.
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["response_text"] == "Hello"
    assert result["error"] is None
    assert result["finish_reason"] == "stop"


# ---- G2: billed truth, read in-band.

# The full usage object as OpenRouter sends it: cost is the charge,
# cost_details is the BYOK-only upstream figure, and the two details
# objects carry the token breakdowns that never appear in the answer.
FULL_USAGE = {
    "prompt_tokens": 13,
    "completion_tokens": 8,
    "cost": 0.000031,
    "cost_details": {"upstream_inference_cost": 0},
    "completion_tokens_details": {"reasoning_tokens": 512},
    "prompt_tokens_details": {"cached_tokens": 9},
}


@respx.mock
async def test_billed_cost_and_token_breakdown_are_ingested(client):
    respx.post(OPENROUTER_URL).respond(
        json={
            **FIXTURE,
            "provider": "Fireworks",
            "usage": FULL_USAGE,
            "choices": [
                {**FIXTURE["choices"][0], "native_finish_reason": "STOP_SEQUENCE"}
            ],
        }
    )

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["billed_cost_usd"] == 0.000031
    assert result["reasoning_tokens"] == 512
    assert result["cached_tokens"] == 9
    assert result["provider"] == "Fireworks"
    assert result["native_finish_reason"] == "STOP_SEQUENCE"
    # The normalized value is untouched by the native one arriving.
    assert result["finish_reason"] == "stop"


@respx.mock
async def test_stream_ingests_billed_cost_and_provider(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    sse(
                        {
                            "id": "gen-1",
                            "provider": "Together",
                            "choices": [{"delta": {"content": "Hi"}}],
                        }
                    ),
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
                    sse({"choices": [], "usage": FULL_USAGE}),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["error"] is None
    assert result["billed_cost_usd"] == 0.000031
    assert result["reasoning_tokens"] == 512
    assert result["cached_tokens"] == 9
    assert result["provider"] == "Together"
    assert result["native_finish_reason"] == "eos"


@respx.mock
async def test_absent_usage_details_leave_the_new_fields_none(client):
    """Most providers send a bare usage object. Its absence of detail is
    not a failure and must not be guessed at."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["prompt_tokens"] == 13
    for field in ("billed_cost_usd", "reasoning_tokens", "cached_tokens"):
        assert result[field] is None, field
    # The captured fixture is a real DeepSeek response, so it does carry
    # the two provenance strings; only the usage detail is missing.
    assert result["provider"] == "DeepSeek"
    assert result["native_finish_reason"] == "stop"


@pytest.mark.parametrize("poison", [float("nan"), float("-inf"), -0.5, "1e-5", True])
@respx.mock
async def test_review_repro_poisoned_billed_cost_degrades_to_none(client, poison):
    """The F.1 money rule on the new field. A NaN makes every ceiling
    comparison false, silently disabling the ceiling; a negative charge
    walks the accumulated total backward and buys headroom that was never
    earned; a string raises on the first arithmetic. All of them must
    degrade to None, and none of them may error a result the money was
    already spent on."""
    # Encoded by hand rather than through respx's json= helper: httpx
    # refuses to serialize NaN and infinity, and a bare NaN token is
    # exactly what a misbehaving provider puts on the wire and what the
    # client's own json parsing accepts on the way back in.
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({**FIXTURE, "usage": {"cost": poison}}),
            headers={"content-type": "application/json"},
        )
    )

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["billed_cost_usd"] is None
    assert result["error"] is None
    assert result["response_text"] is not None


@respx.mock
async def test_review_repro_stream_poisoned_billed_cost_degrades_to_none(client):
    """Same rule on the streaming path, where the usage object rides the
    final chunk and the deltas are already on the wire when it lands."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hi"),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    sse(
                        {
                            "choices": [],
                            "usage": {**FULL_USAGE, "cost": float("nan")},
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["billed_cost_usd"] is None
    assert result["error"] is None
    assert result["response_text"] == "Hi"
    # The rest of the usage object survives its poisoned neighbour.
    assert result["reasoning_tokens"] == 512


@respx.mock
async def test_non_dict_usage_details_do_not_raise(client):
    """The never-raises contract one level down: a provider sending a
    string where a details object belongs must degrade, not crash."""
    respx.post(OPENROUTER_URL).respond(
        json={
            **FIXTURE,
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 8,
                "completion_tokens_details": "n/a",
                "prompt_tokens_details": None,
            },
        }
    )

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["reasoning_tokens"] is None
    assert result["cached_tokens"] is None


@respx.mock
async def test_non_string_provider_degrades_to_none(client):
    """Junk in a provenance field never reaches persistence: a dict bound
    into a sqlite text column would roll back the whole run."""
    respx.post(OPENROUTER_URL).respond(
        json={
            **FIXTURE,
            "provider": {"name": "Fireworks"},
            "choices": [{**FIXTURE["choices"][0], "native_finish_reason": 7}],
        }
    )

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["error"] is None
    assert result["provider"] is None
    assert result["native_finish_reason"] is None


@respx.mock
async def test_empty_provider_string_is_absent_not_blank(client):
    """An empty caption reads as a layout fault, so "" is absence."""
    respx.post(OPENROUTER_URL).respond(json={**FIXTURE, "provider": ""})

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["provider"] is None


@respx.mock
async def test_a_later_usage_object_without_cost_does_not_leave_a_stale_charge(client):
    """Last usage object wins, entirely. A stream that reports usage twice
    must not leave the first object's cost standing beside the second
    object's token counts: that row would claim a charge the platform did
    not reconfirm."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hi"),
                    sse({"choices": [], "usage": FULL_USAGE}),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    sse(
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 13, "completion_tokens": 8},
                        }
                    ),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    result = events[-1]["result"]
    assert result["billed_cost_usd"] is None
    assert result["reasoning_tokens"] is None
    assert result["cached_tokens"] is None
    assert result["completion_tokens"] == 8


# ---- G3: the generation id survives everything, including aborts.

from bench.models import GENERATION_ID_HEADER  # noqa: E402


@respx.mock
async def test_generation_id_comes_from_the_response_header(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, json=FIXTURE, headers={GENERATION_ID_HEADER: "gen-from-header"}
        )
    )

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    # The header settles it; the body's own id is only the fallback.
    assert result["generation_id"] == "gen-from-header"


@respx.mock
async def test_generation_id_falls_back_to_the_body_id(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["generation_id"] == FIXTURE["id"]


@respx.mock
async def test_review_repro_stream_holder_carries_the_id_with_no_chunk_seen(client):
    """The runs with the least knowable billing were the only ones with no
    id to reconcile against: the id was read out of a chunk, so a stream
    that died or was stopped before any chunk arrived persisted nothing to
    audit. The header is available at response start, and the injected
    holder is how a consumer that never reaches the done event learns it.
    """
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            headers={GENERATION_ID_HEADER: "gen-header-only"},
            stream=ChunkStream([], exc=httpx.ReadError("net down")),
        )
    )
    holder = {}

    events = [
        e
        async for e in stream_model(
            "hi", "deepseek/deepseek-chat", client, holder=holder
        )
    ]

    # No chunk ever arrived, so the old chunk-only capture had nothing.
    assert events[-1]["result"]["response_text"] is None
    assert events[-1]["result"]["error"] is not None
    assert holder["generation_id"] == "gen-header-only"
    assert events[-1]["result"]["generation_id"] == "gen-header-only"


@respx.mock
async def test_holder_is_optional_and_stream_works_without_one(client):
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            headers={GENERATION_ID_HEADER: "gen-1"},
            stream=ChunkStream(
                [
                    delta_chunk("Hi"),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    DONE_MARKER,
                ]
            ),
        )
    )

    events = await collect("hi", "deepseek/deepseek-chat", client)

    assert events[-1]["result"]["generation_id"] == "gen-1"


@respx.mock
async def test_a_disagreeing_chunk_id_is_logged_not_applied(client, caplog):
    """Two sources for one fact means they can disagree. Overwriting would
    make the persisted id depend on which source spoke last, and raising
    would break the never-raises contract over a provenance nicety."""
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            headers={GENERATION_ID_HEADER: "gen-header"},
            stream=ChunkStream(
                [
                    sse({"id": "gen-chunk", "choices": [{"delta": {"content": "Hi"}}]}),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    DONE_MARKER,
                ]
            ),
        )
    )

    with caplog.at_level("WARNING"):
        events = await collect("hi", "deepseek/deepseek-chat", client)

    assert events[-1]["result"]["generation_id"] == "gen-header"
    assert events[-1]["result"]["error"] is None
    assert "generation id disagreement" in caplog.text


@respx.mock
async def test_an_empty_generation_id_header_is_absence_not_a_value(client):
    """An empty header is absence, not a value: _as_label's empty-string
    rule is what keeps it from settling the field and locking out the body
    id that follows. The version of this test that sent no header at all
    could not tell the two apart, and duplicated the fallback test above
    it.

    Only the empty string, deliberately. A whitespace-only header IS
    currently taken as a value, because _as_label tests truthiness rather
    than stripping. That is a behaviour question rather than a coverage
    one, so it is recorded rather than asserted here: a test claiming a
    rule the code does not have would be the same defect this replaces.
    """
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, json=FIXTURE, headers={GENERATION_ID_HEADER: ""}
        )
    )

    result = await run_model("hi", "deepseek/deepseek-chat", client)

    assert result["generation_id"] == FIXTURE["id"]


# ---- G5: privacy routing.

from bench.models import DATA_POLICY_PREFS, provider_preferences  # noqa: E402


def test_standard_policy_sends_todays_payload_unchanged():
    """The default must cost nothing and claim nothing. Any extra key here
    would be a routing constraint no operator asked for."""
    assert provider_preferences("standard") == {"sort": "throughput"}


def test_deny_policy_asks_for_no_training_routing():
    # Field name pinned against OpenRouter's provider-routing docs; see
    # DATA_POLICY_PREFS for the URL and the date it was read.
    assert provider_preferences("deny") == {
        "sort": "throughput",
        "data_collection": "deny",
    }


def test_zdr_policy_asks_for_zero_retention_and_no_training():
    assert provider_preferences("zdr") == {
        "sort": "throughput",
        "zdr": True,
        "data_collection": "deny",
    }


def test_no_policy_silently_sends_the_standard_payload():
    """The failure this guards is silence: a mode that quietly degraded to
    standard would route confidential prompts to training-eligible
    providers while the run row claimed otherwise."""
    standard = provider_preferences("standard")
    for policy in DATA_POLICY_PREFS:
        if policy == "standard":
            continue
        assert provider_preferences(policy) != standard, policy


@pytest.mark.parametrize("policy", sorted(DATA_POLICY_PREFS))
@respx.mock
async def test_run_model_sends_the_policys_provider_block(client, policy):
    route = respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    await run_model(
        "hi",
        "deepseek/deepseek-chat",
        client,
        provider_prefs=provider_preferences(policy),
    )

    sent = json.loads(route.calls[0].request.content)
    assert sent["provider"] == provider_preferences(policy)


@pytest.mark.parametrize("policy", sorted(DATA_POLICY_PREFS))
@respx.mock
async def test_stream_model_sends_the_policys_provider_block(client, policy):
    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(
                [
                    delta_chunk("Hi"),
                    sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    DONE_MARKER,
                ]
            ),
        )
    )

    async for _ in stream_model(
        "hi",
        "deepseek/deepseek-chat",
        client,
        provider_prefs=provider_preferences(policy),
    ):
        pass

    sent = json.loads(route.calls[0].request.content)
    assert sent["provider"] == provider_preferences(policy)
    # The throughput sort is not dropped by the merge: it stabilizes WHAT
    # is measured, and a privacy mode must not quietly cost that.
    assert sent["provider"]["sort"] == "throughput"


@respx.mock
async def test_the_recorded_request_shows_the_policy_that_was_sent(client):
    """request_json is the reproducibility record, so it has to carry the
    routing constraints too: "which policy was this run under" must be
    answerable from the row and not only from the process that made it."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model(
        "hi",
        "deepseek/deepseek-chat",
        client,
        provider_prefs=provider_preferences("zdr"),
    )

    recorded = json.loads(result["request_json"])
    assert recorded["provider"] == {
        "sort": "throughput",
        "zdr": True,
        "data_collection": "deny",
    }


# ---- G6: the reconcile path.

from bench.models import GENERATION_URL, fetch_generation  # noqa: E402

# The response shape OpenRouter's OpenAPI description of GET /generation
# gives; see fetch_generation for the URL and the date it was read.
GENERATION_BODY = {
    "data": {
        "id": "gen-1",
        "total_cost": 0.0015,
        "provider_name": "Infermatic",
        "native_finish_reason": "stop",
        "native_tokens_prompt": 10,
        "native_tokens_completion": 25,
        "native_tokens_reasoning": 5,
        "native_tokens_cached": 3,
    }
}


@respx.mock
async def test_fetch_generation_normalizes_the_audit_record(client):
    route = respx.get(GENERATION_URL).respond(json=GENERATION_BODY)

    record = await fetch_generation(client, "gen-1")

    assert route.calls[0].request.url.params["id"] == "gen-1"
    assert record["error"] is None
    assert record["billed_cost_usd"] == 0.0015
    assert record["provider"] == "Infermatic"
    assert record["native_finish_reason"] == "stop"
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 25
    assert record["reasoning_tokens"] == 5
    assert record["cached_tokens"] == 3
    # Not in the endpoint's documented response schema today, so absent
    # rather than wrong. A None here does not mean unquantized weights.
    assert record["quantization"] is None


@respx.mock
async def test_fetch_generation_reads_quantization_when_it_is_there(client):
    body = {"data": {**GENERATION_BODY["data"], "quantization": "fp8"}}
    respx.get(GENERATION_URL).respond(json=body)

    record = await fetch_generation(client, "gen-1")

    assert record["quantization"] == "fp8"


@pytest.mark.parametrize("poison", [float("nan"), -0.5, "0.0015", True, None])
@respx.mock
async def test_review_repro_poisoned_generation_cost_degrades_to_none(client, poison):
    """The money rule on the audit path too. A poisoned charge here would
    be written straight into billed_cost_usd, where it would reach the
    ceiling accumulator on the next boot's reads and the session bar's
    arithmetic, having bypassed the streaming path's guard entirely."""
    body = json.dumps({"data": {**GENERATION_BODY["data"], "total_cost": poison}})
    respx.get(GENERATION_URL).mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )
    )

    record = await fetch_generation(client, "gen-1")

    assert record["billed_cost_usd"] is None
    # The rest of the record survives its poisoned neighbour, so one bad
    # field does not cost the row its provider and finish reason.
    assert record["error"] is None
    assert record["provider"] == "Infermatic"


@respx.mock
async def test_fetch_generation_reports_an_expired_record_without_raising(client):
    """404 is ordinary: OpenRouter expires generation records, and a run
    old enough to have aged out is not a failure of the pass."""
    respx.get(GENERATION_URL).respond(status_code=404, json={"error": "not found"})

    record = await fetch_generation(client, "gen-old")

    assert record["error"] == "HTTP 404 from OpenRouter"
    assert record["billed_cost_usd"] is None


@respx.mock
async def test_fetch_generation_survives_a_network_failure(client):
    respx.get(GENERATION_URL).mock(side_effect=httpx.ReadError("net down"))

    record = await fetch_generation(client, "gen-1")

    assert record["error"] == "request failed: ReadError"


@pytest.mark.parametrize("body", ["not json", '{"no_data_key": 1}', '{"data": "n/a"}'])
@respx.mock
async def test_fetch_generation_survives_a_malformed_body(client, body):
    respx.get(GENERATION_URL).mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )
    )

    record = await fetch_generation(client, "gen-1")

    assert record["error"] == "malformed response from OpenRouter"


@respx.mock
async def test_non_string_provenance_from_the_audit_path_degrades(client):
    body = {
        "data": {
            **GENERATION_BODY["data"],
            "provider_name": {"name": "Infermatic"},
            "native_finish_reason": "",
            "native_tokens_prompt": "10",
        }
    }
    respx.get(GENERATION_URL).respond(json=body)

    record = await fetch_generation(client, "gen-1")

    assert record["provider"] is None
    assert record["native_finish_reason"] is None
    assert record["prompt_tokens"] is None


# ---- Phase H1: experiment controls on the wire.

PRE_H_PAYLOAD = json.loads(
    (Path(__file__).parent / "fixtures" / "pre_h_payload.json").read_text()
)


@respx.mock
async def test_review_repro_blank_controls_send_the_pre_h_payload_byte_for_byte(client):
    """Both external reviews named uncontrolled sampling and routing as the
    confound between "outputs I received" and "a comparison I can defend".
    The controls that answer it come with a standing hazard: a tool that
    helpfully fills in its own defaults would send temperature 1.0 and call
    it the user's choice, and the record would then be unable to say that
    nobody chose it.

    So rule one is that a blank control does not appear in the payload at
    all, and this is the floor that proves it. The fixture is not
    hand-written: it is the raw request body captured by running main's
    bench/models.py, the code that shipped before this phase, against
    respx. The comparison is on the bytes rather than on a re-serialized
    dict, so key order and formatting are locked too.

    On pre-fix code there are no controls to leave blank, so this test
    cannot fail there; its force is forward. It fails the moment any
    control acquires a default that reaches the wire.
    """
    for kind, controls in (
        ("run_model", None),
        ("run_model", {}),
        ("stream_model", {}),
    ):
        expected = PRE_H_PAYLOAD[kind]
        route = respx.post(OPENROUTER_URL)
        if kind == "run_model":
            route.respond(json=FIXTURE)
            await run_model(
                "the prompt",
                "vendor/model",
                client,
                max_tokens=16384,
                controls=controls,
            )
        else:
            route.mock(
                return_value=httpx.Response(
                    200, stream=ChunkStream([delta_chunk("hi"), DONE_MARKER])
                )
            )
            async for _ in stream_model(
                "the prompt",
                "vendor/model",
                client,
                max_tokens=16384,
                controls=controls,
            ):
                pass
        sent = route.calls[-1].request.content.decode()
        assert sent == expected, f"{kind} with controls={controls!r}"
        respx.reset()


@respx.mock
async def test_every_set_control_reaches_the_payload_where_the_docs_put_it(client):
    """Placement is pinned against OpenRouter's documentation, not guessed:
    the three sampling fields are top-level keys, reasoning effort is a key
    inside a top-level reasoning OBJECT, and the system prompt is a message
    rather than a parameter. See bench.models for the URLs and read dates.
    """
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    await run_model(
        "the prompt",
        "vendor/model",
        client,
        controls={
            "system": "be terse",
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
            "effort": "high",
            # Consumed by provider_preferences, never a top-level key.
            "routing": "price",
        },
    )

    body = json.loads(respx.calls[0].request.content)
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["seed"] == 7
    assert body["reasoning"] == {"effort": "high"}
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "the prompt"},
    ]
    # routing is not a chat-completions parameter; it belongs to the
    # provider object, and putting it top-level would be a silent no-op.
    assert "routing" not in body


def test_control_payload_cannot_collide_with_the_payload_it_merges_into():
    """The payload merges control_payload with **, so a control named like
    an existing key would silently overwrite the model, the budget, the
    provider block or the stream flag. The merge is only safe because the
    emitted names are a closed set, so that set is asserted rather than
    trusted.
    """
    everything = control_payload(
        {
            "system": "s",
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 1,
            "effort": "low",
            "routing": "price",
        }
    )

    assert set(everything) == {"temperature", "top_p", "seed", "reasoning"}
    reserved = {
        "model",
        "messages",
        "max_tokens",
        "provider",
        "stream",
        "stream_options",
    }
    assert not set(everything) & reserved


def test_control_payload_omits_what_was_not_set():
    assert control_payload(None) == {}
    assert control_payload({}) == {}
    # Only the set control, and nothing standing in for the others.
    assert control_payload({"temperature": 0.0}) == {"temperature": 0.0}
    # Zero is a value a user can choose and is not blankness. Temperature 0
    # is the whole point of a reproducibility run, so a falsy check here
    # would drop exactly the control that matters most.
    assert control_payload({"top_p": 0.0, "seed": 0}) == {"top_p": 0.0, "seed": 0}


def test_control_messages_prepends_only_a_set_system_prompt():
    plain = [{"role": "user", "content": "p"}]
    assert control_messages("p", None) == plain
    assert control_messages("p", {}) == plain
    assert control_messages("p", {"temperature": 0.5}) == plain
    assert control_messages("p", {"system": "s"}) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "p"},
    ]


def test_provider_preferences_merges_routing_with_every_privacy_mode():
    """The merge is where a regression would silently drop one side, and
    both sides are promises: routing decides what is being measured, and
    the privacy policy decides where the prompt is allowed to go. Every
    combination is asserted because a union is easy to write and easy to
    break in one direction only.
    """
    expected = {
        ("throughput", "standard"): {"sort": "throughput"},
        ("throughput", "deny"): {"sort": "throughput", "data_collection": "deny"},
        ("throughput", "zdr"): {
            "sort": "throughput",
            "zdr": True,
            "data_collection": "deny",
        },
        ("price", "standard"): {"sort": "price"},
        ("price", "deny"): {"sort": "price", "data_collection": "deny"},
        ("price", "zdr"): {"sort": "price", "zdr": True, "data_collection": "deny"},
        # "default" is the absence of sort, which is the documented way to
        # ask for OpenRouter's own load balancing. The privacy keys must
        # survive that absence untouched.
        ("default", "standard"): {},
        ("default", "deny"): {"data_collection": "deny"},
        ("default", "zdr"): {"zdr": True, "data_collection": "deny"},
    }

    for (routing, policy), want in expected.items():
        got = provider_preferences(policy, routing)
        assert got == want, (routing, policy)
        # The zdr belt from G5 survives every routing choice: zdr mode
        # always denies collection as well, which is the stricter reading
        # the documentation does not require.
        if policy == "zdr":
            assert got["zdr"] is True and got["data_collection"] == "deny"
        # No routing mode may displace a privacy key, and no privacy mode
        # may invent a sort.
        if routing == "default":
            assert "sort" not in got


def test_provider_preferences_default_is_todays_behaviour():
    """Omitting the routing argument has to keep every existing caller, and
    the boot-time provider block, exactly where they were."""
    for policy in ("standard", "deny", "zdr"):
        assert provider_preferences(policy) == provider_preferences(
            policy, "throughput"
        )


def test_provider_preferences_refuses_an_unknown_routing_mode():
    """Loud rather than silently standard: the request model validates the
    value, so reaching here with a typo means the boundary was bypassed and
    the run row would claim routing the payload never carried."""
    with pytest.raises(KeyError):
        provider_preferences("standard", "cheapest")


@respx.mock
async def test_request_json_records_a_set_control_and_omits_a_blank_one(client):
    """The record rides along for free, which is what makes the documented
    silent-drop behaviour survivable: a run whose temperature a provider
    ignored is still a run that can be shown to have asked for it."""
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model(
        "the prompt", "vendor/model", client, controls={"temperature": 0.2}
    )

    recorded = json.loads(result["request_json"])
    assert recorded["temperature"] == 0.2
    for absent in ("top_p", "seed", "reasoning"):
        assert absent not in recorded, absent


# ---- Phase I3: rubric judging, blind by construction.

from bench.models import JUDGE_MAX_TOKENS


def judge_body(text):
    return {
        "id": "gen-judge-1",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "cost": 0.00004},
    }


@respx.mock
async def test_a_judge_verdict_becomes_a_score_with_its_charge(client):
    respx.post(OPENROUTER_URL).respond(
        json=judge_body('{"score": 0.5, "reason": "half of it"}')
    )

    out = await judge_response(client, "judge/one", "the rubric", None, "the answer")

    assert out["score"] == 0.5
    assert out["detail"] == "half of it"
    assert out["error"] is None
    assert out["generation_id"] == "gen-judge-1"
    assert out["billed_cost_usd"] == 0.00004
    # passed stays None on purpose: a rubric defines a graded score, and
    # turning it into a pass needs a threshold the rubric never stated.
    assert out["passed"] is None


@respx.mock
async def test_the_judge_payload_never_carries_a_model_identity(client):
    """Blind by construction. The function is not given the model that
    produced the response, so no edit inside it can leak one: the identity
    is not in scope. A judge told which model it is grading has a
    documented tendency to prefer some of them, and a blindness enforced
    by "remember not to include it" is one careless edit from gone."""
    respx.post(OPENROUTER_URL).respond(json=judge_body('{"score": 1.0}'))

    await judge_response(
        client,
        "judge/one",
        "grade this",
        "the expected answer",
        "the candidate answer",
    )

    sent = json.loads(respx.calls[-1].request.content)
    blob = json.dumps(sent)
    for identity in ("deepseek", "anthropic", "openai", "model/alpha", "gpt"):
        assert identity not in blob.lower(), (identity, blob)
    # What it does carry: the rubric, the reference and the response.
    assert "grade this" in blob
    assert "the expected answer" in blob
    assert "the candidate answer" in blob
    # And the judge's own id, which is the one model name a judge payload
    # legitimately contains.
    assert sent["model"] == "judge/one"


@respx.mock
async def test_the_judge_gets_its_own_modest_budget(client):
    """A verdict is a number and a sentence. A judge inheriting an
    extended-tier experiment would buy headroom no rubric needs, once per
    scored trial, at the judge's own price."""
    respx.post(OPENROUTER_URL).respond(json=judge_body('{"score": 1.0}'))

    await judge_response(client, "judge/one", "r", None, "a")

    assert json.loads(respx.calls[-1].request.content)["max_tokens"] == JUDGE_MAX_TOKENS
    assert JUDGE_MAX_TOKENS < BUDGET_STANDARD


@respx.mock
async def test_the_judge_payload_carries_the_privacy_policy(client):
    """A judge call sends a run's response text, which is model output
    about the user's prompt, so it is subject to the same data-handling
    promise as the run itself."""
    prefs = {"sort": "throughput", "data_collection": "deny"}
    respx.post(OPENROUTER_URL).respond(json=judge_body('{"score": 1.0}'))

    await judge_response(client, "judge/one", "r", None, "a", provider_prefs=prefs)

    assert json.loads(respx.calls[-1].request.content)["provider"] == prefs


@respx.mock
async def test_a_malformed_verdict_is_a_scoring_failure_that_still_records_the_charge(
    client,
):
    """Money spent is money spent, whatever came back for it. Capturing
    the charge before parsing is what keeps an unparseable but paid-for
    verdict from vanishing from the accounting."""
    respx.post(OPENROUTER_URL).respond(json=judge_body("it was pretty good I think"))

    out = await judge_response(client, "judge/one", "r", None, "a")

    assert out["score"] is None
    assert "no JSON object" in out["error"]
    assert out["billed_cost_usd"] == 0.00004
    assert out["generation_id"] == "gen-judge-1"


@respx.mock
async def test_a_judge_refusal_is_recorded_not_raised(client):
    """Never raises, the same contract the other client functions carry:
    this runs in a loop over many results and one bad reply must not end
    the pass."""
    respx.post(OPENROUTER_URL).respond(status_code=429)

    out = await judge_response(client, "judge/one", "r", None, "a")

    assert out["score"] is None
    assert out["error"] == "judge returned HTTP 429"


@respx.mock
async def test_a_judge_transport_failure_is_recorded_not_raised(client):
    respx.post(OPENROUTER_URL).mock(side_effect=httpx.ConnectError("down"))

    out = await judge_response(client, "judge/one", "r", None, "a")

    assert out["error"] == "judge request failed: ConnectError"


async def test_a_trial_with_no_text_is_scored_without_asking_anyone(client):
    """Nothing to grade, so paying a judge to say so would be spending
    money to learn what the row already says. No respx mock here, which
    is itself the assertion: reaching the network would fail the test."""
    out = await judge_response(client, "judge/one", "r", None, None)

    assert out["score"] == 0.0
    assert out["detail"] == "no response text: the trial did not complete"
    assert out["billed_cost_usd"] is None


@respx.mock
async def test_a_judge_answering_in_content_parts_is_read_the_same_way(client):
    """Through _flatten_content, the extractor run_model uses. A second
    extractor would be a second thing to keep in step with providers."""
    respx.post(OPENROUTER_URL).respond(
        json={
            "id": "gen-judge-2",
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": '{"score": 1.0}'}]
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )

    out = await judge_response(client, "judge/one", "r", None, "a")

    assert out["score"] == 1.0
