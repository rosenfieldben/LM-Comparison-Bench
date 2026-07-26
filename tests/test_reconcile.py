"""The reconcile pass, end to end against a seeded database.

The whole script runs in-process: store.connect against a temp file and a
stubbed generation endpoint through respx, injected the way the endpoints
inject their client. No subprocess and no network, so the assertions are
about what the pass writes, not about how it was launched.
"""

import io
import json

import httpx
import pytest
import respx

from bench import store
from bench.models import GENERATION_URL
from bench.reconcile import reconcile

AUDIT_COLUMNS = ("billed_cost_usd", "provider", "quantization", "native_finish_reason")


def audit_body(gen_id, **overrides):
    data = {
        "id": gen_id,
        "total_cost": 0.0015,
        "provider_name": "Infermatic",
        "quantization": "fp8",
        "native_finish_reason": "stop",
        "native_tokens_prompt": 10,
        "native_tokens_completion": 25,
    }
    data.update(overrides)
    return {"data": data}


@pytest.fixture
def conn(tmp_path):
    connection = store.connect(str(tmp_path / "bench.db"))
    yield connection
    connection.close()


def seed(conn, results, prompt="reconcile me"):
    return store.save_run(conn, prompt, results, None, None)


def completed_result(model, gen_id, **extra):
    """A finished run: text, tokens, a local estimate, no billed cost."""
    return {
        "model": model,
        "response_text": "hello",
        "latency_ms": 12.0,
        "prompt_tokens": 10,
        "completion_tokens": 25,
        "error": None,
        "cost_usd": 2.9e-5,
        "ttft_ms": 5.0,
        "max_tokens": 16384,
        "generation_id": gen_id,
        "finish_reason": "stop",
        **extra,
    }


def aborted_result(model, gen_id):
    """The shape the disconnect path persists: an id, no usage, no cost."""
    return {
        "model": model,
        "response_text": "partial",
        "latency_ms": 40.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": "stream aborted before completion",
        "cost_usd": None,
        "ttft_ms": 8.0,
        "max_tokens": 16384,
        "generation_id": gen_id,
    }


def row_of(conn, run_id, index=0):
    return store.get_run(conn, run_id)["results"][index]


@respx.mock
async def test_the_work_list_is_rows_with_an_id_and_no_charge(conn):
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    run_id = seed(
        conn,
        [
            completed_result("model/a", "gen-a"),
            # Already billed: nothing to reconcile.
            completed_result("model/b", "gen-b", billed_cost_usd=0.002),
            # No id: nothing to ask about.
            completed_result("model/c", None),
        ],
    )

    pending = store.results_awaiting_reconciliation(conn)

    assert [p["generation_id"] for p in pending] == ["gen-a"]
    assert store.get_run(conn, run_id) is not None


@respx.mock
async def test_dry_run_writes_nothing(conn):
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    run_id = seed(conn, [completed_result("model/a", "gen-a")])
    before = row_of(conn, run_id)
    out = io.StringIO()

    async with httpx.AsyncClient(trust_env=False) as client:
        counts = await reconcile(conn, client, apply=False, delay_s=0, out=out)

    assert counts == {"pending": 1, "written": 0, "skipped": 0}
    assert row_of(conn, run_id) == before
    assert "would write 1 row(s)" in out.getvalue()
    # The report names what it would write, or the dry run tells you
    # nothing you could act on.
    assert "provider='Infermatic'" in out.getvalue()


@respx.mock
async def test_apply_fills_exactly_the_audit_columns(conn):
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    run_id = seed(conn, [completed_result("model/a", "gen-a")])
    before = row_of(conn, run_id)

    async with httpx.AsyncClient(trust_env=False) as client:
        counts = await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    assert counts["written"] == 1
    after = row_of(conn, run_id)
    assert after["billed_cost_usd"] == 0.0015
    assert after["provider"] == "Infermatic"
    assert after["quantization"] == "fp8"
    assert after["native_finish_reason"] == "stop"
    # Everything else is byte for byte what it was. The local estimate in
    # particular keeps its own meaning: cost_usd is what the catalog said
    # at run time, and an audit that overwrote it would erase the
    # comparison between the two.
    for key, value in before.items():
        if key not in AUDIT_COLUMNS:
            assert after[key] == value, key


@respx.mock
async def test_an_aborted_row_is_reconciled_like_any_other(conn):
    """The rows the pass exists for: a stopped run has an id and no usage,
    so its billing is exactly the thing nothing local can settle."""
    respx.get(GENERATION_URL).respond(json=audit_body("gen-abort"))
    run_id = seed(conn, [aborted_result("model/a", "gen-abort")])

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    after = row_of(conn, run_id)
    assert after["billed_cost_usd"] == 0.0015
    assert after["provider"] == "Infermatic"
    # The abort record itself is untouched: it is what the server saw.
    assert after["error"] == "stream aborted before completion"
    assert after["response_text"] == "partial"
    assert after["cost_usd"] is None


@respx.mock
async def test_a_legacy_row_missing_the_new_columns_reconciles(conn):
    """A database written before Phase G has NULL in every new column, so
    every one of its rows with an id is a candidate. The pass must fill
    them without needing anything the old writer did not record."""
    respx.get(GENERATION_URL).respond(json=audit_body("gen-legacy"))
    run_id = seed(conn, [completed_result("model/a", "gen-legacy")])
    # Blank the columns a pre-G writer would never have set.
    with conn:
        conn.execute(
            "UPDATE results SET position = NULL, request_json = NULL, "
            "reasoning_tokens = NULL, cached_tokens = NULL WHERE run_id = ?",
            (run_id,),
        )

    async with httpx.AsyncClient(trust_env=False) as client:
        counts = await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    assert counts["written"] == 1
    after = row_of(conn, run_id)
    assert after["billed_cost_usd"] == 0.0015
    assert after["request_json"] is None


@respx.mock
async def test_an_expired_record_is_skipped_not_written(conn):
    respx.get(GENERATION_URL).respond(status_code=404, json={"error": "not found"})
    run_id = seed(conn, [completed_result("model/a", "gen-gone")])
    out = io.StringIO()

    async with httpx.AsyncClient(trust_env=False) as client:
        counts = await reconcile(conn, client, apply=True, delay_s=0, out=out)

    assert counts == {"pending": 1, "written": 0, "skipped": 1}
    assert row_of(conn, run_id)["billed_cost_usd"] is None
    assert "HTTP 404" in out.getvalue()


@respx.mock
async def test_a_poisoned_charge_is_never_written(conn):
    """The money rule at the last place a value can enter the database.

    Not named as a tombstone: it does not fail without as_money, because
    sqlite has no NaN and binds one as NULL, and repair-on-read would
    catch a survivor anyway. Two belts, and this locks that they hold end
    to end. The tombstone for as_money on this path is in test_models.py,
    where the field is observable before either belt applies. The negative
    case below is the one sqlite would happily store."""
    body = json.dumps({"data": {"total_cost": float("nan"), "provider_name": "X"}})
    respx.get(GENERATION_URL).mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )
    )
    run_id = seed(conn, [completed_result("model/a", "gen-a")])

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    # Read the raw column, not the repaired row: repair-on-read would
    # turn a persisted NaN back into None and hide the write that should
    # never have happened. The database itself has to be clean.
    raw = conn.execute(
        "SELECT billed_cost_usd FROM results WHERE run_id = ?", (run_id,)
    ).fetchone()["billed_cost_usd"]
    assert raw is None
    # The rest of the record still lands: one poisoned field costs that
    # field, not the row.
    assert row_of(conn, run_id)["provider"] == "X"


@respx.mock
async def test_an_absent_field_does_not_erase_one_captured_in_band(conn):
    """The generation endpoint saying nothing about a field means "not
    reported", never "known to be nothing". A provider captured live must
    survive an audit record that omits it."""
    respx.get(GENERATION_URL).respond(
        json={"data": {"total_cost": 0.0015, "native_finish_reason": "stop"}}
    )
    run_id = seed(conn, [completed_result("model/a", "gen-a", provider="Fireworks")])

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    assert row_of(conn, run_id)["provider"] == "Fireworks"


@respx.mock
async def test_a_second_pass_finds_nothing_left(conn):
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    seed(conn, [completed_result("model/a", "gen-a")])

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())
        out = io.StringIO()
        counts = await reconcile(conn, client, apply=True, delay_s=0, out=out)

    assert counts["pending"] == 0
    assert "nothing to reconcile" in out.getvalue()


@respx.mock
async def test_limit_walks_the_oldest_rows_first(conn):
    respx.get(GENERATION_URL).mock(
        side_effect=lambda request: httpx.Response(
            200, json=audit_body(request.url.params["id"])
        )
    )
    run_id = seed(
        conn,
        [
            completed_result("model/a", "gen-a"),
            completed_result("model/b", "gen-b"),
        ],
    )

    async with httpx.AsyncClient(trust_env=False) as client:
        counts = await reconcile(
            conn, client, apply=True, limit=1, delay_s=0, out=io.StringIO()
        )

    assert counts == {"pending": 1, "written": 1, "skipped": 0}
    assert row_of(conn, run_id, 0)["billed_cost_usd"] == 0.0015
    assert row_of(conn, run_id, 1)["billed_cost_usd"] is None


@respx.mock
async def test_review_repro_a_negative_charge_is_never_written(conn):
    """The case sqlite would store happily. A negative billed cost read
    back on a later boot subtracts from accumulated spend and buys ceiling
    headroom that was never earned, and unlike a NaN nothing downstream
    would reject it on the way in."""
    respx.get(GENERATION_URL).respond(
        json={"data": {"total_cost": -1.0, "provider_name": "X"}}
    )
    run_id = seed(conn, [completed_result("model/a", "gen-a")])

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    raw = conn.execute(
        "SELECT billed_cost_usd FROM results WHERE run_id = ?", (run_id,)
    ).fetchone()["billed_cost_usd"]
    assert raw is None


def test_the_script_refuses_to_run_without_a_key(monkeypatch, capsys):
    """Authenticating as nobody would report every row as a 401 and look
    exactly like a database full of expired records."""
    from bench.reconcile import main

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert main([]) == 2
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


# ---- Phase G.1: the work list is the money rule, not IS NULL.


@pytest.mark.parametrize(
    ("stored", "label"),
    [(-1.5, "negative"), ("n/a", "non-numeric"), (None, "absent")],
)
@respx.mock
async def test_review_repro_a_charge_no_reader_trusts_stays_reconcilable(
    conn, stored, label
):
    """Adversarial review: the work list asked SQL `billed_cost_usd IS
    NULL` while every reader goes through as_money. A negative or
    non-numeric charge is NOT NULL to SQL and None to _repaired, so the row
    displayed as unpriced forever and was invisible to the only pass that
    could have replaced it. Two predicates for one question, disagreeing.
    """
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    run_id = seed(conn, [completed_result("model/a", "gen-a", billed_cost_usd=stored)])
    # Whatever SQL holds, the reader sees nothing usable.
    assert row_of(conn, run_id)["billed_cost_usd"] is None, label

    assert [
        p["generation_id"] for p in store.results_awaiting_reconciliation(conn)
    ] == ["gen-a"], label

    async with httpx.AsyncClient(trust_env=False) as client:
        counts = await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    assert counts["written"] == 1, label
    assert row_of(conn, run_id)["billed_cost_usd"] == 0.0015, label
    # And the poisoned value is gone from the column, not merely hidden.
    raw = conn.execute(
        "SELECT billed_cost_usd FROM results WHERE run_id = ?", (run_id,)
    ).fetchone()["billed_cost_usd"]
    assert raw == 0.0015, label


@respx.mock
async def test_a_real_zero_charge_is_not_mistaken_for_a_poisoned_one(conn):
    """The guard against over-correcting: free models are real, and a
    confirmed zero is a charge, not an absence. It must stay out of the
    work list or the pass would re-ask about it forever."""
    respx.get(GENERATION_URL).respond(json=audit_body("gen-free"))
    seed(conn, [completed_result("model/free", "gen-free", billed_cost_usd=0.0)])

    assert store.results_awaiting_reconciliation(conn) == []
