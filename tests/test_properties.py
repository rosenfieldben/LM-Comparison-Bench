"""Property tests for the field-type contract, the flattener, the stream
parser, and history pagination.

These assert invariants the example-based suites state one case of at a
time: the normalizers never raise on any input and settle in one pass,
the flattener only ever emits str or None, stream_model always closes
with exactly one done event, and list_runs never duplicates a run or
breaks newest-first ordering across any interleaving of grouped and
ungrouped runs.
"""

import asyncio
import json

import httpx
import respx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from stream_helpers import ChunkStream

from bench import store
from bench.models import (
    OPENROUTER_URL,
    _flatten_content,
    as_metric,
    as_text,
    as_token_count,
    stream_model,
)

# JSON-ish values: what a provider could plausibly put in any field,
# nested arbitrarily. floats include nan and inf on purpose.
json_ish = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text() | st.binary(),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=5), children, max_size=4),
    max_leaves=20,
)


@settings(print_blob=True)
@given(value=json_ish)
def test_normalizers_are_total_and_idempotent(value):
    # Total: none of the three ever raises on any input.
    once_count = as_token_count(value)
    once_text = as_text(value)
    once_metric = as_metric(value)

    # Typed correctly.
    assert once_count is None or (isinstance(once_count, int) and once_count >= 0)
    assert once_text is None or isinstance(once_text, str)
    assert once_metric is None or isinstance(once_metric, float)

    # Idempotent: a second pass changes nothing.
    assert as_token_count(once_count) == once_count
    assert as_text(once_text) == once_text
    assert as_metric(once_metric) == once_metric


@settings(print_blob=True)
@given(value=json_ish)
def test_flatten_content_only_ever_returns_str_or_none(value):
    result = _flatten_content(value)
    assert result is None or isinstance(result, str)


# One SSE line: a well-formed frame, a malformed one, a comment, the
# terminator, or arbitrary text. json.dumps keeps embedded newlines
# escaped, so each stays a single line.
def _delta_line(text):
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})


sse_line = st.one_of(
    st.builds(_delta_line, st.text(max_size=8)),
    st.just(
        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    ),
    st.just(
        "data: "
        + json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )
    ),
    st.builds(
        lambda m: "data: " + json.dumps({"error": {"message": m}}), st.text(max_size=8)
    ),
    st.just("data: {not valid json"),
    st.just("data: [DONE]"),
    st.just(": keep-alive comment"),
    st.text(max_size=12),
)


@settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
@given(lines=st.lists(sse_line, max_size=15))
def test_stream_model_closes_with_one_done_and_honest_ttft(lines):
    async def drive():
        blob = ("\n".join(lines) + "\n").encode("utf-8")
        async with httpx.AsyncClient() as client:
            with respx.mock:
                respx.post(OPENROUTER_URL).mock(
                    return_value=httpx.Response(200, stream=ChunkStream([blob]))
                )
                return [event async for event in stream_model("p", "model/x", client)]

    # Never raises, whatever the line soup.
    events = asyncio.run(drive())

    # Exactly one done, and it is last.
    dones = [e for e in events if e["type"] == "done"]
    assert len(dones) == 1
    assert events[-1]["type"] == "done"

    # ttft_ms is set exactly when at least one delta carried visible text
    # (a delta event is yielded only for non-empty text).
    saw_text = any(e["type"] == "delta" for e in events[:-1])
    assert (events[-1]["result"]["ttft_ms"] is not None) == saw_text


# Content lines that never trigger an error path (no malformed json, no
# in-band error frame): text deltas and usage-only frames. The terminal
# marker is chosen separately so the terminal contract stays unambiguous.
safe_content_line = st.one_of(
    st.builds(_delta_line, st.text(max_size=8)),
    st.just(
        "data: "
        + json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )
    ),
)


@settings(deadline=None, max_examples=60, print_blob=True)
@given(
    content=st.lists(safe_content_line, max_size=8),
    terminal=st.sampled_from(["done", "finish", "none"]),
)
def test_stream_model_error_iff_stream_had_no_terminator(content, terminal):
    # The refined terminal contract from the external review (finding 2):
    # the done error is set exactly when the stream ended with neither a
    # [DONE] marker nor a finish_reason. A clean iterator exhaustion is not
    # a success, but a finish_reason without [DONE] is still complete.
    lines = list(content)
    if terminal == "done":
        lines.append("data: [DONE]")
    elif terminal == "finish":
        lines.append(
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        )
    # terminal == "none": nothing appended; a clean EOF with neither
    # terminator, standing in for an idle-closed or crashed connection.

    async def drive():
        blob = ("\n".join(lines) + "\n").encode("utf-8")
        async with httpx.AsyncClient() as client:
            with respx.mock:
                respx.post(OPENROUTER_URL).mock(
                    return_value=httpx.Response(200, stream=ChunkStream([blob]))
                )
                return [event async for event in stream_model("p", "model/x", client)]

    events = asyncio.run(drive())
    error = events[-1]["result"]["error"]
    had_text = any(e["type"] == "delta" for e in events[:-1])

    if terminal == "none":
        assert error == "stream ended before completion: no [DONE] and no finish reason"
    elif had_text:
        # A terminator plus visible text is a clean, complete success.
        assert error is None
    else:
        # A terminator but no text is the pre-existing empty-response case.
        assert error is not None and error.startswith("empty response")


def _make_result():
    # The minimal shape save_run reads; the rest defaults via .get.
    return {
        "model": "model/x",
        "response_text": "hi",
        "latency_ms": 1.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "error": None,
    }


# Each spec is a run: -1 is ungrouped, 0..4 name a group bucket.
run_spec = st.integers(min_value=-1, max_value=4)


@settings(deadline=None, max_examples=75, print_blob=True)
@given(
    specs=st.lists(run_spec, max_size=25), limit=st.integers(min_value=1, max_value=30)
)
def test_list_runs_paginates_without_duplicating_or_reordering(specs, limit):
    # A fresh in-memory database per example: the store connection is not
    # a shared fixture here, so runs never leak between examples.
    conn = store.connect(":memory:")
    try:
        buckets: dict[int, int] = {}
        for spec in specs:
            if spec == -1:
                store.save_run(conn, "p", [_make_result()])
            else:
                if spec not in buckets:
                    buckets[spec] = store.create_group(conn)
                store.save_run(conn, "p", [_make_result()], group_id=buckets[spec])

        entries = store.list_runs(conn, limit=limit)

        assert len(entries) <= limit

        seen_runs: list[int] = []
        order_keys: list[int] = []
        for entry in entries:
            if entry["type"] == "run":
                seen_runs.append(entry["id"])
                order_keys.append(entry["id"])
            else:
                seen_runs.extend(entry["run_ids"])
                # A selected group carries every one of its members, even
                # those older than where the page scan stopped.
                members = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM runs WHERE group_id = ? ORDER BY id",
                        (entry["id"],),
                    )
                ]
                assert entry["run_ids"] == members
                order_keys.append(max(entry["run_ids"]))

        # No run appears in two entries.
        assert len(seen_runs) == len(set(seen_runs))
        # Newest first: each entry's newest run id descends.
        assert order_keys == sorted(order_keys, reverse=True)
    finally:
        conn.close()
