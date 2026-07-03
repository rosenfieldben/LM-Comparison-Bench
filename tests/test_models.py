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
