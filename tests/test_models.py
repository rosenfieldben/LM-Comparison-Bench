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
