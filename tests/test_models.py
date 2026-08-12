import hashlib
import json
import subprocess
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
    missing_parameters,
    normalized_provider_slug,
    run_model,
    strict_provider_preferences,
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

    RECAPTURED ONCE AND THEN PUT BACK, which is worth recording because
    the round trip is the evidence for a claim this repo makes about
    itself. The reasoning-exhaustion fix first sent its visible-output
    reservation on every request, and these bytes were regenerated to
    match by RUNNING the code against respx, never by editing the JSON.
    An adversarial review then found that a reasoning cap with no
    enabled key infers enabled true, so an unprompted cap does not
    merely bound thinking on a model whose default is off, it switches
    thinking on and bills for it. The reservation was gated on the
    catalog's own capability flag, and with that gate a blank-controls
    request to a model the catalog does not vouch for sends exactly what
    it sent before the fix existed. These are main's bytes again, to the
    character.

    So rule one is not broken by default any more. It is broken only for
    a model the catalog says already reasons, where the cap bounds
    thinking that was going to happen regardless. That is a far narrower
    exception than the one this docstring used to describe, and the
    fixture reverting is how the narrowing was proved rather than
    asserted.
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
    # reasoning rejoined this list when the reservation was gated on the
    # catalog. This caller passes no may_send_reasoning_cap, so the bench
    # does not vouch for the model, so nothing is sent: an unset control
    # and an unvouched reservation are both simply absent, which is what
    # rule one asks for in both cases.
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


# ---- Phase I7: the underlying-model estimand's pure parts.


@respx.mock
async def test_fetch_catalog_reads_supported_parameters_and_degrades_it(client):
    """Three distinct states, and strict mode acts differently on each.

    A list of strings is what the check runs against. A missing key is
    "the catalog did not say", which is not the same fact as an empty
    list and must not collapse into one: an empty list means the model
    takes no optional parameters, and something has to be able to tell
    those apart. A malformed element degrades itself rather than the
    entry, the same rule the price fields follow.
    """
    respx.get(MODELS_URL).respond(
        json={
            "data": [
                {"id": "a/full", "supported_parameters": ["temperature", "seed"]},
                {"id": "b/silent"},
                {"id": "c/empty", "supported_parameters": []},
                {"id": "d/mixed", "supported_parameters": ["seed", 7, None, "seed"]},
                {"id": "e/wrongtype", "supported_parameters": "temperature"},
            ]
        }
    )

    models = {m["id"]: m for m in (await fetch_catalog(client))["models"]}

    assert models["a/full"]["supported_parameters"] == ["seed", "temperature"]
    assert models["b/silent"]["supported_parameters"] is None
    assert models["c/empty"]["supported_parameters"] == []
    # Deduplicated and sorted, so two catalogs that listed the same
    # support in a different order produce the same entry.
    assert models["d/mixed"]["supported_parameters"] == ["seed"]
    # A string is not a list of parameter names, whatever it contains.
    assert models["e/wrongtype"]["supported_parameters"] is None


def test_missing_parameters_checks_what_the_payload_will_carry():
    """max_tokens is checked without any control setting it, because
    every payload the bench builds carries one and require_parameters
    would exclude a provider that does not support it."""
    assert missing_parameters(["max_tokens"], {}) == []
    assert missing_parameters([], {}) == ["max_tokens"]
    # effort is checked as "reasoning", the name the payload uses.
    assert missing_parameters(["max_tokens"], {"effort": "low"}) == ["reasoning"]
    assert missing_parameters(["max_tokens", "reasoning"], {"effort": "low"}) == []
    # Reported sorted and complete, not first-failure-only, so one
    # refusal names everything the caller has to change.
    assert missing_parameters([], {"temperature": 0.5, "top_p": 0.9}) == [
        "max_tokens",
        "temperature",
        "top_p",
    ]
    # Controls that are not model parameters are not checked as if they
    # were: routing is a key of the provider object and system is a
    # message.
    assert (
        missing_parameters(["max_tokens"], {"routing": "price", "system": "hi"}) == []
    )


def test_missing_parameters_reports_nothing_when_the_catalog_said_nothing():
    """It reports; the caller decides. Returning "nothing missing" here
    would read as support if the caller did not check for None first,
    which is why the refusal for an undescribed model lives at the
    boundary and names that case in its own words."""
    assert missing_parameters(None, {"temperature": 0.5}) == []


def test_strict_preferences_add_to_the_base_without_displacing_it():
    """The privacy promise is written first and the estimand's keys are
    additive, so no estimand can quietly widen the provider population a
    data policy narrowed."""
    base = {"sort": "throughput", "data_collection": "deny", "zdr": True}

    out = strict_provider_preferences(base)

    assert out == {**base, "require_parameters": True}
    assert base == {"sort": "throughput", "data_collection": "deny", "zdr": True}


def test_a_pin_is_a_hard_restriction_and_never_a_preference():
    """allow_fallbacks rides with the order and never without it. An
    order that can be departed from is a preference, and a pinned run
    served by somebody else would record a constraint that did not hold;
    allow_fallbacks with no order would forbid fallback from a provider
    nobody named."""
    pinned = strict_provider_preferences({"sort": "price"}, "Together")
    unpinned = strict_provider_preferences({"sort": "price"})

    assert pinned == {
        "sort": "price",
        "require_parameters": True,
        "order": ["Together"],
        "allow_fallbacks": False,
    }
    assert "allow_fallbacks" not in unpinned
    assert "order" not in unpinned


# ---- Phase I.2 J7: BYOK, captured but never metered.


@respx.mock
async def test_the_byok_upstream_charge_is_captured_beside_the_credit_cost(client):
    """Two different numbers, and only one of them is what the operator
    paid a provider directly.

    Pinned against the usage-accounting page, read 2026-08-04: the usage
    object carries "cost" in CREDITS alongside
    "cost_details": {"upstream_inference_cost": N}, and that second
    figure "is only available for BYOK (Bring Your Own Key) requests.
    For all other requests it will be 0 or null."
    """
    body = json.loads(json.dumps(FIXTURE))
    body["usage"] = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "cost": 0.95,
        "cost_details": {"upstream_inference_cost": 19},
    }
    respx.post(OPENROUTER_URL).respond(json=body)

    out = await run_model("hi", "a/b", client, max_tokens=100)

    # The ceiling's number stays the credit charge.
    assert out["billed_cost_usd"] == 0.95
    # And the direct provider bill is beside it, verbatim as text.
    assert out["upstream_inference_cost_usd"] == "19"


@respx.mock
async def test_review_repro_an_observed_upstream_figure_is_stored_as_received(client):
    """WINDOW: run_model's returned result, over every shape cost_details
    arrives in.

    THIS TEST USED TO ASSERT THE OPPOSITE FOR A ZERO, and the reversal
    is the record of a ruling rather than a change of taste. The old
    rule degraded a wire-sent 0 to None, reasoning from the vendor's
    "will be 0 or null" that a stored 0 could not be told apart from a
    BYOK run that genuinely cost nothing.

    A live probe broke that reasoning from both ends. A run with
    is_byok FALSE came back with a NONZERO upstream figure equal to the
    credit charge, so a populated cell never meant BYOK, and the
    inference the suppression was protecting did not exist. is_byok now
    has a column of its own and says the thing this value was being
    asked to imply.

    With the discriminator moved out, suppressing an observed money
    figure would only make the database disagree with the wire. So a
    zero is stored as a zero, and NULL in that column means the wire
    sent nothing usable.

    NOT-A-NUMBER AND NEGATIVES STILL DEGRADE, which is a different rule
    and survives untouched: those are not money figures at all, and
    as_money applies the same test to every monetary field here.
    """
    for details, expected in (
        # Nothing reported.
        ({}, None),
        # Reported as zero. Stored as zero, which is the reversal.
        ({"upstream_inference_cost": 0}, "0"),
        # Explicitly null.
        ({"upstream_inference_cost": None}, None),
        # A real figure, kept verbatim as text so it stays comparable to
        # a provider invoice.
        ({"upstream_inference_cost": 1.629e-05}, "1.629e-05"),
        # Not money. Still degrades.
        ({"upstream_inference_cost": -1}, None),
    ):
        body = json.loads(json.dumps(FIXTURE))
        body["usage"] = {"cost": 0.5, "cost_details": details}
        respx.post(OPENROUTER_URL).respond(json=body)

        out = await run_model("hi", "a/b", client, max_tokens=100)

        assert out["billed_cost_usd"] == 0.5
        assert out["upstream_inference_cost_usd"] == expected, details


def test_a_pin_normalizes_to_the_documented_lowercase_slug():
    assert normalized_provider_slug("  Together  ") == "together"
    assert normalized_provider_slug("DeepInfra") == "deepinfra"


def test_quantizations_ride_the_documented_filter_and_only_when_asked():
    pinned = strict_provider_preferences({"sort": "price"}, None, ["fp8"])
    plain = strict_provider_preferences({"sort": "price"})

    assert pinned["quantizations"] == ["fp8"]
    assert "quantizations" not in plain


from bench.models import (
    BUDGET_EXTENDED,
    REASONING_BUDGET_SHARE,
    reasoning_reservation,
)

# ---- The reasoning-exhaustion fix, R1: visible-output room is reserved
# ---- on every request, and the reservation is a function of the budget.


@respx.mock
async def test_review_repro_thinking_cannot_eat_the_whole_budget(client):
    """WINDOW: the payload run_model puts on the wire, at both budget
    tiers.

    THE INCIDENT. A production comparison on document prompts reasoned
    until the completion cap closed on it: billed in full, $3.38 to
    $3.50 per card, zero visible output. The generation record showed
    tokens_completion 21350 against native_tokens_reasoning 21350, which
    is the whole budget spent on thinking and nothing left to say it
    with. Nothing bounded the thinking, because nothing asked to.

    THE RESERVATION IS A FUNCTION OF THE BUDGET AND OF NOTHING ELSE,
    which is what this asserts and why it is parametrized on the tier
    rather than on a model. 16384 reserves 8192 for the answer; 65536
    reserves 32768, which is four times the entire standard tier. A
    model name appears nowhere in the arithmetic and a separate test
    proves it appears nowhere in the code.

    SUSPENSION POINT: the await on run_model, once per tier. The payload
    is read from respx's recorded call AFTER that await returns, so the
    loop's second iteration cannot observe the first one's request; the
    -1 index is the call the iteration just made, not whichever finished
    last.
    """
    for budget, expected in ((BUDGET_STANDARD, 8192), (BUDGET_EXTENDED, 32768)):
        respx.post(OPENROUTER_URL).respond(json=FIXTURE)

        await run_model(
            "the prompt",
            "vendor/model",
            client,
            max_tokens=budget,
            # The catalog vouches for this model: it thinks whether or
            # not it is asked to, so bounding the thinking cannot start
            # any. See reasoning_reservation for why that is the gate.
            may_send_reasoning_cap=True,
        )

        body = json.loads(respx.calls[-1].request.content)
        assert body["reasoning"] == {"max_tokens": expected}, budget
        # Half, stated as the relationship rather than as two constants,
        # so changing the share cannot leave this test agreeing with a
        # number nobody meant.
        assert body["reasoning"]["max_tokens"] == int(
            body["max_tokens"] * REASONING_BUDGET_SHARE
        )


@respx.mock
async def test_the_reservation_stands_down_for_a_declared_effort(client):
    """WINDOW: the same payload, when the operator declared a reasoning
    effort.

    NOT BOTH, and the pinned page is what decides it: OpenRouter
    introduces effort and max_tokens as "One of the following (not
    both)". So a request carrying a declared effort must not also carry
    the bench's cap, and the operator's declaration is the one that
    rides: it is part of the experiment, the reservation is the bench's
    own default, and a default overwriting a declaration is what rule
    one exists to forbid.

    SUSPENSION POINT: the await on run_model. Only one request is made,
    and every assertion runs after it resolves, including the two direct
    calls to reasoning_reservation, which suspend nowhere at all.

    That leaves those requests unprotected by R1, which is exactly why
    the card's indicator keys on the tokens that came BACK rather than
    on the field that went out.

    TWO DEFENCES, AND BOTH ARE CHECKED HERE, which is a fact this test
    learned the hard way. Mutating the guard to return the cap
    unconditionally left this test PASSING, because control_payload
    merges last and replaces the whole reasoning value rather than
    merging into it, so the ordering alone produced a correct request.
    That is the design working, but a test that only reads the payload
    cannot tell a live guard from a dead one hiding behind the ordering.
    So the guard is also asserted directly, on the function, where the
    same mutation does fail.
    """
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    await run_model(
        "the prompt",
        "vendor/model",
        client,
        max_tokens=BUDGET_STANDARD,
        controls={"effort": "high"},
        may_send_reasoning_cap=True,
    )

    body = json.loads(respx.calls[-1].request.content)
    assert body["reasoning"] == {"effort": "high"}
    assert "max_tokens" not in body["reasoning"]
    # The guard itself, behind the ordering that would otherwise cover
    # for it. See the docstring: this is the assertion the mutation
    # kills.
    assert reasoning_reservation(BUDGET_STANDARD, {"effort": "high"}, True) == {}
    # And it stands down only for a declared effort, not for any
    # controls at all, so a request with a temperature still reserves.
    assert reasoning_reservation(BUDGET_STANDARD, {"temperature": 0.5}, True) == {
        "reasoning": {"max_tokens": 8192}
    }


@respx.mock
async def test_review_repro_the_judge_sends_no_unsatisfiable_reservation(client):
    """WINDOW: the judge's outgoing payload, asserted against the PINNED
    CONTRACT rather than against a mock's willingness to answer.

    SUSPENSION POINT: the await on judge_response. The payload is read
    after it resolves.

    THE DEFECT, and it is arithmetic rather than judgement. The judge
    used to reserve half of its 512 token budget, sending
    reasoning.max_tokens 256. Two pinned rules make that request
    impossible to honour:

      "When using the reasoning.max_tokens parameter, that value is
      used directly with a minimum of 1024 tokens."

      "Important: max_tokens must be strictly higher than the reasoning
      budget to ensure there are tokens available for the final response
      after thinking."

    256 is raised to 1024 by the first rule. The second then requires
    the outer max_tokens to exceed 1024, and it is 512. Every Anthropic
    judge that reasons by default was receiving a request the contract
    forbids.

    WHY NOBODY NOTICED, which is the lens this test now carries. The
    old test mocked a 200 and asserted the outgoing JSON contained
    {"max_tokens": 256}. A mock cannot refuse. It answered "did we send
    what we meant to send" when the question was "could anyone honour
    what we sent", and those come apart precisely when a contract has
    a minimum in it. The assertions below are written against the
    contract's numbers, so the arithmetic is checked rather than the
    mock's manners.
    """
    respx.post(OPENROUTER_URL).respond(
        json={"choices": [{"message": {"content": '{"score": 1, "detail": "ok"}'}}]}
    )

    out = await judge_response(client, "judge/one", "the rubric", None, "the answer")

    body = json.loads(respx.calls[-1].request.content)
    # THE FIELD IS ABSENT. Not smaller, not conditional: absent.
    assert "reasoning" not in body
    assert body["max_tokens"] == JUDGE_MAX_TOKENS
    # And the call still works, so the stand-down costs nothing here.
    assert out["score"] == 1


def test_the_judge_budget_could_not_satisfy_the_pinned_minimum():
    """WINDOW: the arithmetic itself, with no request and no mock.

    The shape that was being sent, checked against the two pinned rules
    it had to satisfy. This is the test that would have caught the
    defect on the day it was written, because it consults the contract
    instead of a stub.

    ANTHROPIC_REASONING_MINIMUM is not a number this repository chose.
    It is the vendor's, quoted beside JUDGE_MAX_TOKENS, and it is
    written down here so that raising the judge's budget later is a
    decision somebody makes on purpose rather than one that silently
    re-enables an impossible request.
    """
    anthropic_minimum = 1024

    would_have_sent = int(JUDGE_MAX_TOKENS * REASONING_BUDGET_SHARE)
    assert would_have_sent == 256
    # Rule one raises it.
    effective = max(would_have_sent, anthropic_minimum)
    assert effective == 1024
    # Rule two then cannot be satisfied: the outer budget must be
    # STRICTLY higher than the reasoning budget.
    assert not JUDGE_MAX_TOKENS > effective

    # What it would take to make a half-share judge reservation legal,
    # recorded so the cost of the alternative is visible: an outer
    # budget above twice the minimum.
    assert 2 * anthropic_minimum > JUDGE_MAX_TOKENS


def test_no_model_name_appears_in_the_reservation_logic():
    """WINDOW: bench/models.py as it sits on disk, parsed rather than
    imported, so what is checked is the source a reviewer reads.

    THE UNIVERSALITY CONSTRAINT, asserted rather than promised.

    Exhaustion is a property of thinking against a completion cap, not
    of one vendor, and the incident was diagnosed on one model only
    because that is the one that was running. A fix that branched on a
    name would protect exactly the model whose failure was noticed and
    leave every other reasoning model exposed, while looking finished.

    PARSED, NOT GREPPED, and the difference is the whole reason this
    test is written the way it is. A grep over the source fires on the
    pinned documentation quotes beside the constant, which say
    "(Anthropic-style)" and "(OpenAI-style)" because that is what
    OpenRouter's page says and a pinned quote must be verbatim. Those
    are prose about a wire format, not a branch. What must contain no
    vendor is the CODE, so this walks the function's AST and inspects
    every name, attribute and string literal that can actually execute.

    The vendor list is not a security boundary. It is a tripwire for one
    specific mistake, and it names the vendors this repo's own fixtures
    and catalogs mention.

    WORD BOUNDARIES, NOT SUBSTRINGS, and the closing review is what
    taught this. The first version asked `vendor in dumped`, and running
    the same check across the frontend flagged "fable" inside
    registerDiffable. That is the K1.5 defect exactly, where a tombstone
    matched `extracted_text` inside `length(extracted_text)`, and a
    tripwire that cries wolf is one somebody eventually deletes. The
    scope here is one small function so nothing was actually matching,
    but a test whose passing depends on the function staying small is
    not the test that was wanted.
    """
    import ast
    import re

    source = (Path(__file__).parent.parent / "bench" / "models.py").read_text()
    tree = ast.parse(source)
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reasoning_reservation"
    )
    body = list(target.body)
    # Drop the docstring, which is prose by definition.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    assert body, "the reservation has no executable body to check"

    executable = " ".join(ast.dump(node) for node in body).lower()
    for vendor in (
        "fable",
        "claude",
        "anthropic",
        "openai",
        "gpt",
        "gemini",
        "google",
        "deepseek",
        "qwen",
        "llama",
        "mistral",
        "grok",
    ):
        assert not re.search(rf"(?<![a-z0-9]){vendor}(?![a-z0-9])", executable), vendor

    # And the constant it reads is a plain number, not a per-model table.
    assert isinstance(REASONING_BUDGET_SHARE, float)


from bench.models import (  # noqa: E402
    REASONING_SHARE_EXHAUSTED,
    _ingest_usage,
    empty_response_error,
    reasoning_ate_the_output,
)

# ---- The reasoning-exhaustion fix, R2: the label says where the money
# ---- went, and it says it because of the token shape.


# Every case is a TOKEN SHAPE with a name describing the shape, not a
# model. Three of them are drawn from real records and are labelled with
# which; the rest are the boundaries of the rule. The point of the table
# is that no row needs to know what produced it.
EMPTY_RESPONSE_SHAPES = (
    # (label, finish_reason, completion, reasoning, expected error)
    #
    # THE FIRST THREE ROWS ALL CARRIED finish_reason "length", which is
    # how this table came to certify a label that lied. Every firing case
    # was a real truncation, so the table could not see that the shape
    # test fires on refusals too: at this call site there is no visible
    # text by construction, so a refusal that thought at all has
    # reasoning at or near completion and lands in the same bucket. The
    # rows below the truncations are the ones that were missing.
    (
        "the incident, exactly equal",
        "length",
        21350,
        21350,
        "no visible answer: completion budget exhausted during reasoning "
        "(21350 reasoning tokens)",
    ),
    (
        "a different model, a different tier, same shape",
        "length",
        8192,
        8100,
        "no visible answer: completion budget exhausted during reasoning "
        "(8100 reasoning tokens)",
    ),
    (
        "exactly at the threshold",
        "length",
        1000,
        900,
        "no visible answer: completion budget exhausted during reasoning "
        "(900 reasoning tokens)",
    ),
    (
        "one token under the threshold",
        "length",
        1000,
        899,
        "empty response (finish_reason: length)",
    ),
    (
        "empty with no reasoning at all, the old wording kept",
        "stop",
        40,
        0,
        "empty response (finish_reason: stop)",
    ),
    (
        "a provider that reported no usage",
        "length",
        None,
        None,
        "empty response (finish_reason: length)",
    ),
    # ---- The shape fires, the budget did NOT run out. Measured before
    # ---- the correction: each of these was told its completion budget
    # ---- had been exhausted, and the card appended "try extended
    # ---- budget" on top.
    (
        "a content filter that thought first, 0.2% of the budget spent",
        "content_filter",
        37,
        37,
        "no visible answer: the whole completion went to reasoning "
        "(37 reasoning tokens, finish_reason: content_filter)",
    ),
    (
        "a refusal that stopped clean, 1% of the budget spent",
        "stop",
        210,
        198,
        "no visible answer: the whole completion went to reasoning "
        "(198 reasoning tokens, finish_reason: stop)",
    ),
    (
        "a provider abort with no reason worth the name",
        None,
        4,
        4,
        "no visible answer: the whole completion went to reasoning "
        "(4 reasoning tokens, finish_reason: unknown)",
    ),
    (
        "reasoning reported but no completion count",
        None,
        None,
        5000,
        "empty response (finish_reason: unknown)",
    ),
)


@pytest.mark.parametrize(
    "shape,finish,completion,reasoning,expected",
    EMPTY_RESPONSE_SHAPES,
    ids=[row[0] for row in EMPTY_RESPONSE_SHAPES],
)
def test_review_repro_the_label_says_where_the_completion_went(
    shape, finish, completion, reasoning, expected
):
    """WINDOW: the synthesized error for a 200 that carried no visible
    text, at the moment it is composed.

    THE DEFECT WAS A TRUE SENTENCE THAT MISLED. "empty response
    (finish_reason: length)" names the symptom and omits the cause. The
    card it appeared on had been billed $3.38 to $3.50 for 21350 tokens
    of thinking, and nothing in the message suggested that a retry would
    spend the same money the same way. It reads like a provider glitch.
    It was an accounting fact.

    PARAMETRIZED ON SHAPE, NOT ON A MODEL, which is the constraint this
    whole fix is under and which this table is the demonstration of. The
    second row is a deliberately different tier and a different pair of
    numbers from the incident's; it is not the incident's model and the
    rule does not know or care. The threshold rows are the boundary, one
    on each side of it, so a change to REASONING_SHARE_EXHAUSTED cannot
    pass unnoticed.

    THREE OUTCOMES, and the middle one is a correction an adversarial
    review forced. The shape test is close to a tautology here, because
    this function is only reached when there was no visible text at all,
    so any reasoning model that thought and said nothing has a ratio near
    1.0 whatever went wrong. "Exhausted" therefore needs finish_reason
    behind it, and without that gate a content filter that spent 0.2% of
    its budget was told the budget was gone.

    THE OLD WORDING IS KEPT, not superseded, for a genuinely empty
    response with no thinking behind it. That is a different failure, a
    provider returning null content, and the sentence that was already
    there describes it correctly.
    """
    assert empty_response_error(finish, completion, reasoning) == expected, shape


def test_the_shape_test_reads_only_the_two_counts():
    """WINDOW: reasoning_ate_the_output called directly, with no
    request, no response and no I/O anywhere in the frame.

    The predicate on its own, away from the sentence it produces.

    Nothing here is a request field. It cannot see the budget, the
    reservation, the effort or the model, which is exactly what lets the
    identical rule run on a route where the reservation was ignored: a
    provider that drops an unknown parameter still reports its usage.
    """
    assert reasoning_ate_the_output(21350, 21350) is True
    assert reasoning_ate_the_output(1000, 900) is True
    assert reasoning_ate_the_output(1000, 899) is False
    # Absent counts are not evidence, and zero reasoning is evidence of
    # the opposite. Both fall out through one falsy guard.
    assert reasoning_ate_the_output(None, None) is False
    assert reasoning_ate_the_output(1000, None) is False
    assert reasoning_ate_the_output(None, 1000) is False
    assert reasoning_ate_the_output(1000, 0) is False
    assert reasoning_ate_the_output(0, 1000) is False


# The shapes both implementations are asked about. Boundaries on both
# sides of the threshold, the incident, row 694, and every way a count
# can be missing.
CROSS_LANGUAGE_SHAPES = [
    [21350, 21350],
    [8500, 8000],
    [1000, 900],
    [1000, 899],
    [3400, 2944],
    [3400, 3060],
    [500, 0],
    [0, 0],
    [None, None],
    [500, None],
    [None, 500],
    # SMALL COMPLETIONS, and they are in this table because the first
    # version of it did not have them and was therefore blind to the
    # exact mutation that motivated the rewrite. A guard inserted into
    # the JavaScript reading `if (completionTokens < 200) return false`
    # left every row above unchanged, so the two languages disagreed on
    # every short response while this test stayed green. Writing a
    # better test and not re-running the mutation against it would have
    # shipped the same hole in nicer clothing.
    #
    # The first two are the live probe's own numbers, which is the
    # cheapest way to be sure a row is reachable: something really did
    # return them.
    [37, 37],
    [35, 0],
    [150, 140],
    [199, 190],
    [4, 4],
    [1, 1],
]


def test_review_repro_the_two_languages_agree_on_every_shape():
    """WINDOW: both implementations of the exhaustion rule, executed,
    over one shared table.

    THIS TEST USED TO BE A STRING SEARCH, and an adversarial review
    proved it was a false guard in BOTH directions. It asserted that
    two source lines appeared verbatim in static/lib.js.

      Inserting `if (completionTokens < 200) return false;` into
      reasoningAteTheOutput left it PASSING and the node suite passing,
      with the two languages now disagreeing on every small response. A
      behaviour change it could not see.

      Renaming a local or rewrapping the return failed it, on
      JavaScript whitespace, with both implementations still identical.
      A formatting change it could not tolerate.

    A guard that misses what it is for and fires on what it is not is a
    tax that teaches people to delete it. So it runs the JavaScript now:
    node executes lib.js against the same shapes Python is asked about,
    and the two answer sheets are compared. Text is not consulted at
    all.

    The threshold is included in the comparison rather than asserted
    separately, because a shared constant that drifted would show up as
    a disagreement on the boundary rows, which is the symptom that
    matters rather than the digit that caused it.
    """
    script = """
const lib = require(process.argv[1]);
const shapes = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify({
  verdicts: shapes.map(([c, r]) => lib.reasoningAteTheOutput(c, r)),
  threshold: lib.REASONING_SHARE_EXHAUSTED,
}));
"""
    lib = Path(__file__).parent.parent / "static" / "lib.js"
    proc = subprocess.run(
        ["node", "-e", script, str(lib), json.dumps(CROSS_LANGUAGE_SHAPES)],
        capture_output=True,
        text=True,
        check=True,
    )
    js = json.loads(proc.stdout)

    python_verdicts = [reasoning_ate_the_output(c, r) for c, r in CROSS_LANGUAGE_SHAPES]
    assert js["verdicts"] == python_verdicts, list(
        zip(CROSS_LANGUAGE_SHAPES, python_verdicts, js["verdicts"])
    )
    assert js["threshold"] == REASONING_SHARE_EXHAUSTED
    # And the table straddles the boundary, so agreement is a fact about
    # the rule rather than about a table that only asks easy questions.
    assert True in python_verdicts and False in python_verdicts


@respx.mock
async def test_review_repro_an_exhausted_batch_result_is_labelled(client):
    """WINDOW: run_model's returned result, end to end from a payload
    that looks exactly like the incident's.

    SUSPENSION POINT: the await on run_model. Everything asserted here
    is on the returned result, which run_model finishes assembling
    before it returns, so there is no window in which the error and the
    counts could be read half-written.

    THE ORDERING THIS PROTECTS. The empty-text branch used to run BEFORE
    _ingest_usage, so it read two Nones no matter what the provider
    reported and could never have produced this label. Moving the
    ingest above it is the fix; this is the test that notices if it
    moves back. That ordering is INSIDE run_model and is not a
    suspension question: both steps run between one await and the
    return.
    """
    respx.post(OPENROUTER_URL).respond(
        json={
            "choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 21350,
                "completion_tokens_details": {"reasoning_tokens": 21350},
            },
        }
    )

    result = await run_model("a document question", "vendor/model", client)

    assert result["response_text"] is None
    assert result["error"] == (
        "no visible answer: completion budget exhausted during reasoning "
        "(21350 reasoning tokens)"
    )
    # And the counts the label was drawn from are on the record beside
    # it, so a reader can check the claim rather than trust it.
    assert result["completion_tokens"] == 21350
    assert result["reasoning_tokens"] == 21350


@respx.mock
async def test_review_repro_an_exhausted_stream_is_labelled_identically(client):
    """WINDOW: the done event of a stream that emitted reasoning deltas
    and no content, from the first chunk to iterator exhaustion.

    SUSPENSION POINT: the `async for` over aiter_lines inside
    stream_model. The result is only complete after the final usage
    chunk has been folded in and done() has run, so the assertion is on
    the done event rather than on anything observable mid-iteration.

    ONE FAILURE MUST NOT HAVE TWO DESCRIPTIONS. The browser uses the
    streaming endpoint and a script uses the batch one; a label whose
    purpose is honesty is worth nothing if it depends on which door the
    caller came through. Both now call one function, and this asserts
    the two produce the identical sentence.
    """
    body = (
        'data: {"id":"gen-x","choices":[{"delta":{"reasoning":"thinking"}}]}\n\n'
        'data: {"id":"gen-x","choices":[{"delta":{},"finish_reason":"length"}],'
        '"usage":{"prompt_tokens":1200,"completion_tokens":21350,'
        '"completion_tokens_details":{"reasoning_tokens":21350}}}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(OPENROUTER_URL).respond(
        200, headers={"Content-Type": "text/event-stream"}, text=body
    )

    events = [event async for event in stream_model("a document question", "m", client)]

    done = events[-1]
    assert done["type"] == "done"
    assert done["result"]["response_text"] is None
    assert done["result"]["error"] == (
        "no visible answer: completion budget exhausted during reasoning "
        "(21350 reasoning tokens)"
    )
    # The two endpoints, one sentence.
    assert done["result"]["error"] == empty_response_error("length", 21350, 21350)


# ---- The reasoning-exhaustion fix, R3: one unit per surface, and every
# ---- surface says which.


def test_review_repro_a_cap_is_never_rendered_as_a_count():
    """WINDOW: the card's budget badge and its token metrics, read out of
    static/render.js.

    THE PHANTOM 44K. The badge said "budget 65536" and the reasoning
    metric said 21350. Both are numbers of tokens, nothing on the card
    distinguished them, and 65536 minus 21350 is 44186 tokens that were
    never generated and never billed. A reader took that difference for
    unaccounted output and the diagnosis went the wrong way for a full
    round. The two numbers answer different questions: one is the
    ceiling the bench SENT, the other is what the provider COUNTED.

    ASSERTED ON THE SOURCE, not through a browser, because the property
    is that the word is present in the markup the renderer emits; the
    browser tombstone beside this one checks that it reaches the page.
    """
    render = (Path(__file__).parent.parent / "static" / "render.js").read_text()

    assert 'note.textContent = "budget cap " + result.max_tokens;' in render
    # The bare form is gone, which is the half a "contains cap" assertion
    # would miss.
    assert '"budget " + result.max_tokens' not in render
    # And each surface that shows a count says what the count is.
    assert "prompt/completion tokens actually used" in render
    assert "Not a count: compare it against tok i/o" in render


def test_the_export_manifest_states_the_unit_once():
    """WINDOW: the manifest note as the report module composes it,
    read from the function rather than from a built export, so the
    wording is checked in one place and the export test checks that it
    arrives.

    The citable artifact, whose readers were not in the room.

    Every trial line carries four counts and one ceiling side by side.
    prompt_tokens next to max_tokens invites precisely the subtraction
    that manufactured the phantom above, and an export outlives the
    conversation that produced it.

    ONCE, in the manifest, because it is a property of the format rather
    than of any run: a note stamped on every trial line is one a reader
    stops seeing by line three.
    """
    from bench.report import _manifest_token_note

    # The note names every count field and singles out the one field on
    # the trial line that is not a count.
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
    ):
        assert field in _manifest_token_note()
    assert "max_tokens is not a count" in _manifest_token_note()
    assert "model's own tokenizer" in _manifest_token_note()


def test_the_stored_counts_have_exactly_one_source():
    """WINDOW: store.RECONCILABLE_COLUMNS at import time, which is the
    list that decides what a reconcile pass is allowed to write.

    THE AUDIT'S CENTRAL FINDING, asserted so it cannot quietly stop
    being true.

    One unit per surface is only enforceable if there is one place the
    unit is decided. Every count the bench stores enters through
    _ingest_usage, reading OpenRouter's usage object, which the
    usage-accounting page describes as "Prompt and completion token
    counts using the model's native tokenizer". The generation
    endpoint's counts are parsed by _generation_record and never
    persisted, because RECONCILABLE_COLUMNS does not list them, so they
    cannot become a second unit in the database by accident.

    This walks the store's reconcilable list rather than trusting the
    comment beside it: a future entry that added a token column would
    put two counts per result on one row, and that is the shape this
    workstream exists to prevent.
    """
    from bench.store import RECONCILABLE_COLUMNS

    for column in RECONCILABLE_COLUMNS:
        assert "token" not in column, column


# ---- A1's live probe, replayed. The two usage blocks below are real
# ---- wire bytes, not a hand-built shape, which is why they get their
# ---- own fixture and their own test rather than a paragraph in a
# ---- commit body nobody can execute.

PROBE = json.loads(
    (Path(__file__).parent / "fixtures" / "probe_reasoning_enable.json").read_text()
)


def test_review_repro_a_bare_cap_enables_thinking_on_a_model_that_had_it_off():
    """WINDOW: the two usage objects of a matched pair of live calls,
    fed through the bench's own ingest.

    THE CLAIM THIS SETTLES. The gate in fetch_catalog refuses to send an
    unprompted reasoning cap to a model whose catalog entry says
    thinking is off. That gate was built on a schema comment, "Default:
    inferred from 'effort' or 'max_tokens'", which is an inference about
    what OpenRouter does with a field. Inferences about other people's
    systems are exactly what this repo requires to be pinned, and a
    pinned quote is still not a measurement.

    IT IS A MEASUREMENT NOW. Two calls to google/gemma-4-31b-it on
    DeepInfra, 2026-08-12, differing in nothing but the presence of
    reasoning.max_tokens:

      gen-1786546333-86V1vJMk7JsWwwPhajoN  no field   0 reasoning tokens
      gen-1786546398-Z4GhaMYohdT9FWGFiYhc  cap 256  312 reasoning tokens

    The model publishes mandatory false and default_enabled false. It
    was not thinking. The cap alone made it think, and billed 8.1 times
    as much for the same question.

    REPLAYED THROUGH _ingest_usage rather than asserted as arithmetic on
    the JSON, so this also proves the bench reads these real blocks the
    way the fix assumes it does. A synthetic fixture cannot do that;
    every stub in this suite was written by someone who already believed
    the shape.
    """
    control: dict = {}
    capped: dict = {}
    _ingest_usage(control, PROBE["control"]["usage"])
    _ingest_usage(capped, PROBE["capped"]["usage"])

    # The model was not thinking.
    assert control["reasoning_tokens"] == 0
    # The cap alone made it think.
    assert capped["reasoning_tokens"] == 312
    # And it is not a rounding: 92% of the completion went to reasoning.
    assert capped["reasoning_tokens"] / capped["completion_tokens"] > 0.9

    # THE COST OF HAVING SHIPPED THIS UNGATED, on one short question.
    ratio = capped["billed_cost_usd"] / control["billed_cost_usd"]
    assert 8.0 < ratio < 8.2, ratio


def test_the_probes_capped_response_clears_the_exhaustion_threshold():
    """WINDOW: the same two blocks, read by the shape test that drives
    the card's label and its indicator.

    A REAL RESPONSE ON THE RIGHT SIDE OF THE LINE. Every other case
    exercising REASONING_SHARE_EXHAUSTED in this suite is a number
    somebody chose. This one came off the wire at 312 of 338, and it
    clears nine tenths, which is the first evidence that the threshold
    is reachable by an ordinary short answer from an ordinary model
    rather than only by a $3.50 document prompt.

    AND ITS finish_reason IS "stop", NOT "length", which is precisely
    the distinction A2 was corrected to make. Nothing was truncated
    here; the budget did not run out. A label built on the token shape
    alone would have told this run its completion budget was exhausted.
    """
    assert reasoning_ate_the_output(338, 312) is True
    assert PROBE["capped"]["_finish_reason"] == "stop"
    # So the honest sentence is the one without the causal claim, and no
    # budget remedy is offered for a run whose budget was never the
    # constraint.
    assert empty_response_error("stop", 338, 312) == (
        "no visible answer: the whole completion went to reasoning "
        "(312 reasoning tokens, finish_reason: stop)"
    )
    # The control is nowhere near it, which is the other half.
    assert reasoning_ate_the_output(35, 0) is False


def test_review_repro_a_non_byok_run_reports_an_upstream_cost():
    """WINDOW: the cost_details of both probe calls, and the column the
    bench derives from them.

    AN UNDOCUMENTED SHAPE THE PROBE CAUGHT IN PASSING. OpenRouter's
    usage-accounting page says upstream_inference_cost "is only
    available for BYOK (Bring Your Own Key) requests. For all other
    requests it will be 0 or null." Both probe calls carry
    is_byok false AND a nonzero upstream_inference_cost, equal to cost
    to the last digit, alongside two fields the page does not mention at
    all: upstream_inference_prompt_cost and
    upstream_inference_completions_cost.

    WHAT IT COSTS THE BENCH. _as_upstream_cost degrades only a zero or
    absent value to None, so a value like this one is STORED, and the
    README's claim that a NULL there is "the honest record of this run
    was not BYOK" does not hold on this route: the column is populated
    for a run that was not BYOK. The number is also not the separate
    provider bill the column exists to hold; it is the credit charge
    repeated.

    NOT FIXED HERE, deliberately. Gating the column on is_byok is a
    behaviour change to the money record and it is the reviewer's call,
    not a thing to slip into a test-writing pass. What this test does is
    stop the shape being rediscovered: it pins what the wire actually
    sent, so the decision is made against evidence whenever it is made.
    """
    for kind in ("control", "capped"):
        usage = PROBE[kind]["usage"]
        assert usage["is_byok"] is False, kind
        upstream = usage["cost_details"]["upstream_inference_cost"]
        assert upstream > 0, kind
        # Not a second, independent figure. The same charge again.
        assert upstream == usage["cost"], kind

    # And the bench stores it, which is the part that matters.
    result: dict = {}
    _ingest_usage(result, PROBE["capped"]["usage"])
    assert result["upstream_inference_cost_usd"] is not None


# ---- T3: a strict pin selects one endpoint, and only that endpoint's
# ---- published capabilities may vouch for the field sent to it.

from bench.models import (  # noqa: E402
    ENDPOINTS_URL,
    endpoint_supports_reasoning,
    fetch_endpoints,
)


@respx.mock
async def test_review_repro_a_pinned_endpoint_is_not_vouched_for_by_the_aggregate(
    client,
):
    """WINDOW: the endpoint listing for one model, read the way a strict
    pin reads it.

    SUSPENSION POINT: the await on fetch_endpoints. Everything asserted
    is on the returned listing, which is fully assembled before the
    coroutine returns.

    THE DEFECT. A strict pin sends order plus allow_fallbacks false plus
    require_parameters true, which selects ONE host and forbids it from
    ignoring a parameter it does not support. The bench decided whether
    to send the reasoning field from the MODEL-level catalog, which is a
    union across every host that serves the model. A union can advertise
    what the pinned host does not accept, and under require_parameters
    that does not degrade into a silent ignore: it empties the provider
    pool.

    MEASURED, not imagined. On 2026-08-12 one real model published three
    endpoints advertising 18, 12 and 17 parameters, differing in which.
    The shape below is that shape: an aggregate that lists reasoning,
    and a pinned endpoint that does not.

    ASSERTED AGAINST THE PINNED CONTRACT, not against a mock's
    willingness to answer: the question is what the LISTING says, and a
    200 proves only that something replied.
    """
    respx.get(ENDPOINTS_URL.format(model="vendor/model")).respond(
        json={
            "data": {
                "endpoints": [
                    # The generous host. A model-level aggregate is the
                    # union, so it would say "reasoning supported".
                    {
                        "provider_name": "DeepInfra",
                        "supported_parameters": ["max_tokens", "reasoning", "seed"],
                    },
                    # The pinned host, which does not take the field.
                    {
                        "provider_name": "BaseTen",
                        "supported_parameters": ["max_tokens", "temperature"],
                    },
                    # A host that published no list at all.
                    {"provider_name": "Together"},
                ]
            }
        }
    )

    listing = await fetch_endpoints(client, "vendor/model")
    assert listing["fetched"] is True

    # THE WHOLE POINT: one model, two endpoints, opposite answers.
    assert endpoint_supports_reasoning(listing, "deepinfra") is True
    assert endpoint_supports_reasoning(listing, "baseten") is False
    # Slug normalization, so a pin recorded lowercase finds a
    # provider_name that is not.
    assert endpoint_supports_reasoning(listing, "DeepInfra") is True

    # THREE-VALUED, and the None cases are the ones strict mode must not
    # collapse into support.
    assert endpoint_supports_reasoning(listing, "together") is None
    assert endpoint_supports_reasoning(listing, "novita") is None
    assert endpoint_supports_reasoning(listing, None) is None


@respx.mock
async def test_an_unfetchable_endpoint_listing_is_absence_of_evidence(client):
    """WINDOW: fetch_endpoints against every way the listing can fail to
    answer.

    SUSPENSION POINT: the await on fetch_endpoints, once per shape.

    Strict mode's rule is that absence of evidence is refused rather
    than assumed, and the caller can only apply it if this function
    reports the difference. fetched False is not an empty listing and
    must never read as one.
    """
    url = ENDPOINTS_URL.format(model="vendor/model")

    for shape, route in (
        ("a 404, which is what an unknown model returns", lambda: httpx.Response(404)),
        ("a 500", lambda: httpx.Response(500)),
        ("a body that is not JSON", lambda: httpx.Response(200, text="nope")),
        ("JSON without the data key", lambda: httpx.Response(200, json={})),
        (
            "data without endpoints",
            lambda: httpx.Response(200, json={"data": {"id": "x"}}),
        ),
        (
            "endpoints that is not a list",
            lambda: httpx.Response(200, json={"data": {"endpoints": "nope"}}),
        ),
    ):
        respx.get(url).mock(side_effect=lambda request, r=route: r())
        listing = await fetch_endpoints(client, "vendor/model")
        assert listing["fetched"] is False, shape
        # And the decision made from it is "cannot tell", never False,
        # so a caller that distinguishes them still can.
        assert endpoint_supports_reasoning(listing, "deepinfra") is None, shape

    # A transport failure is the same answer.
    respx.get(url).mock(side_effect=httpx.ConnectError("down"))
    listing = await fetch_endpoints(client, "vendor/model")
    assert listing["fetched"] is False
