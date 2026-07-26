import json
from pathlib import Path

import httpx
import pytest
import respx

from bench.models import MODELS_URL, OPENROUTER_URL, fetch_catalog, run_model

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
    assert catalog == {"fetched": False, "models": [], "prices": {}}


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
    assert catalog == {"fetched": False, "models": [], "prices": {}}


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
