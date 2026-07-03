import json
from pathlib import Path

import httpx
import pytest
import respx

from bench.models import MODELS_URL, OPENROUTER_URL, fetch_prices, run_model

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_response.json").read_text()
)


@pytest.fixture
async def client():
    async with httpx.AsyncClient() as c:
        yield c


@respx.mock
async def test_happy_path_parses_text_and_tokens(client):
    respx.post(OPENROUTER_URL).respond(json=FIXTURE)

    result = await run_model("Say hello in five words.", "deepseek/deepseek-chat", client)

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
    assert "timed out" in result["error"]


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
async def test_fetch_prices_parses_and_skips_malformed_entries(client):
    respx.get(MODELS_URL).respond(
        json={
            "data": [
                {"id": "a/one", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                {"id": "b/broken", "pricing": {}},
                {"id": "c/nonnumeric", "pricing": {"prompt": "free", "completion": "0"}},
            ]
        }
    )

    prices = await fetch_prices(client)

    assert prices == {"a/one": {"prompt": 1e-06, "completion": 2e-06}}


@respx.mock
async def test_fetch_prices_failure_returns_empty_dict(client):
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("offline"))
    assert await fetch_prices(client) == {}


@respx.mock
async def test_fetch_prices_http_error_returns_empty_dict(client):
    respx.get(MODELS_URL).respond(status_code=500)
    assert await fetch_prices(client) == {}


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
            stream=ChunkStream([
                sse({"choices": [{"delta": {"role": "assistant"}}]}),
                delta_chunk("Hel"),
                delta_chunk("lo"),
                # Content-parts delta: must flatten via the shared rule.
                delta_chunk([{"type": "text", "text": " world"}]),
                sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                sse({"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 8}}),
                DONE_MARKER,
            ]),
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
    assert result["error"] == "stream stalled for 30s"


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
            stream=ChunkStream([
                delta_chunk("Hel"),
                b"data: {not json\n\n",
            ]),
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
            stream=ChunkStream([
                delta_chunk("Hi "),
                delta_chunk([{"type": "text", "text": None}]),
                delta_chunk([{"type": "text", "text": "there"}]),
                DONE_MARKER,
            ]),
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
            stream=ChunkStream([
                delta_chunk("Hel"),
                sse({"error": {"code": 502, "message": "upstream model crashed"}}),
                DONE_MARKER,
            ]),
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


from bench.models import fetch_catalog


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
                },
                {"id": "b/bare"},
                {"id": "c/badprice", "name": "Bad Price", "pricing": {"prompt": "free"}},
                {"no_id": True},
            ]
        }
    )

    catalog = await fetch_catalog(client)

    assert catalog["fetched"] is True
    assert [m["id"] for m in catalog["models"]] == ["a/full", "b/bare", "c/badprice"]
    full, bare, badprice = catalog["models"]
    assert full["context_length"] == 32000
    assert full["prompt_price"] == 1e-06
    assert bare["name"] is None
    assert bare["context_length"] is None
    assert bare["prompt_price"] is None
    assert badprice["name"] == "Bad Price"
    assert badprice["prompt_price"] is None
    # The price map only carries fully priced entries.
    assert set(catalog["prices"]) == {"a/full"}


@respx.mock
async def test_fetch_catalog_failure_reports_not_fetched(client):
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("offline"))
    catalog = await fetch_catalog(client)
    assert catalog == {"fetched": False, "models": [], "prices": {}}
