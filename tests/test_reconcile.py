"""The reconcile pass, end to end against a seeded database.

The whole script runs in-process: store.connect against a temp file and a
stubbed generation endpoint through respx, injected the way the endpoints
inject their client. No subprocess and no network, so the assertions are
about what the pass writes, not about how it was launched.

EVERY CLIENT HERE IS BUILT trust_env=False, and it is load-bearing rather
than tidy. respx intercepts at the transport, which is a layer BELOW the
one that reads proxy environment variables: with trust_env on, httpx
resolves the ambient proxy while CONSTRUCTING the client, and a socks
proxy without the socksio package raises ImportError there. The client
never exists, so respx never gets a request to intercept, and a test that
touches no network fails because of the network anyway.

That is not hypothetical. Two tests were added with a bare
httpx.AsyncClient() and passed everywhere except under the poisoned-proxy
hermeticity run, where they were the only two failures in the suite:

  ImportError: Using SOCKS proxy, but the 'socksio' package is not
  installed. Make sure to install httpx using `pip install httpx[socks]`.

The convention was invisible, which is why it was missed: fifteen call
sites carried the flag and nothing said why. It is said here once rather
than fifteen times.
"""

import io
import json

import httpx
import pytest
import respx

from bench import store
from bench.models import GENERATION_URL
from bench.reconcile import reconcile

# DERIVED, NOT COPIED, and it was copied until this line. The
# hand-written tuple listed four columns while the store listed six, so
# the "everything else is byte for byte what it was" assertion below was
# checking two writable columns against a promise never to write them.
# It went stale silently twice: once when upstream_inference_cost_usd
# joined RECONCILABLE_COLUMNS and once when is_byok did.
AUDIT_COLUMNS = store.RECONCILABLE_COLUMNS


def audit_body(gen_id, **overrides):
    """One generation record, shaped like the documented one.

    Every key below is a field of the generation record documented at
    https://openrouter.ai/docs/api-reference/overview, and the two live
    captures in tests/fixtures carry all of them. A helper that omits a
    field the real endpoint always publishes cannot exercise the code
    that reads it, which is how is_byok came to be parsed on every pass
    and written by none: the work list could not select on a column no
    test ever filled.

    is_byok defaults FALSE rather than absent because that is what both
    live probes recorded (probe_reasoning_enable.json and
    probe_reasoning_cap_binding.json, both non-BYOK, both explicit
    false). Absent is the exceptional shape, so a test wanting it says
    so by overriding.
    """
    data = {
        "id": gen_id,
        "total_cost": 0.0015,
        "provider_name": "Infermatic",
        "quantization": "fp8",
        "native_finish_reason": "stop",
        "is_byok": False,
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
async def test_the_work_list_is_rows_with_an_id_and_a_closable_gap(conn):
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    run_id = seed(
        conn,
        [
            completed_result("model/a", "gen-a"),
            # Nothing left to ask: charge, host, the host's own finish
            # reason and the BYOK flag are all recorded. The flag joined
            # that list when the predicate learned to select on it, and
            # this row has to carry one or it stops being the control
            # this test needs and becomes a second listed row.
            completed_result(
                "model/b",
                "gen-b",
                billed_cost_usd=0.002,
                provider="Fireworks",
                native_finish_reason="stop",
                is_byok=False,
            ),
            # No id: nothing to ask about.
            completed_result("model/c", None),
        ],
    )

    pending = store.results_awaiting_reconciliation(conn)

    assert [p["generation_id"] for p in pending] == ["gen-a"]
    assert store.get_run(conn, run_id) is not None


@respx.mock
async def test_review_repro_a_billed_row_missing_its_host_is_reconcilable(conn):
    """The work list used to ask only about money. A row could arrive with
    a trusted charge and no `provider` or `native_finish_reason` (the
    usage object carries the cost; those two come from elsewhere and can
    be absent), and it was then permanently invisible to the only pass
    that could have filled them. The gap was real, open, and unreachable.
    """
    run_id = seed(
        conn,
        [
            completed_result(
                "model/a", "gen-a", billed_cost_usd=0.002, native_finish_reason="stop"
            ),
            completed_result(
                "model/b", "gen-b", billed_cost_usd=0.002, provider="Fireworks"
            ),
        ],
    )
    pending = store.results_awaiting_reconciliation(conn)
    assert [p["generation_id"] for p in pending] == ["gen-a", "gen-b"]

    respx.get(GENERATION_URL).respond(
        json=audit_body("gen-a", provider_name="Together", native_finish_reason="stop")
    )
    out = io.StringIO()
    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=out)

    filled = row_of(conn, run_id)
    assert filled["provider"] == "Together"
    assert filled["native_finish_reason"] == "stop"


@respx.mock
async def test_an_unreported_charge_does_not_erase_a_trusted_one(conn):
    """A row on the list for a missing label still has good money on it.
    The endpoint reporting no total_cost means "not reported", so the
    write must leave the stored charge alone. Only an untrusted stored
    value is cleared, which is what makes the poisoned-cost repair work.
    """
    run_id = seed(
        conn,
        [completed_result("model/a", "gen-a", billed_cost_usd=0.002)],
    )
    respx.get(GENERATION_URL).respond(
        json=audit_body("gen-a", total_cost=None, provider_name="Together")
    )
    out = io.StringIO()
    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=out)

    row = row_of(conn, run_id)
    assert row["billed_cost_usd"] == 0.002
    assert row["provider"] == "Together"


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
    work list or the pass would re-ask about it forever.

    The row is seeded with its labels AND its BYOK flag, so money is the
    only predicate that could put it on the list and the assertion still
    tests exactly what it names.
    """
    respx.get(GENERATION_URL).respond(json=audit_body("gen-free"))
    seed(
        conn,
        [
            completed_result(
                "model/free",
                "gen-free",
                billed_cost_usd=0.0,
                provider="Together",
                native_finish_reason="stop",
                is_byok=False,
            )
        ],
    )

    assert store.results_awaiting_reconciliation(conn) == []


@respx.mock
async def test_review_repro_the_byok_figure_is_written_rather_than_fetched_and_dropped(
    conn,
):
    """fetch_generation has parsed upstream_inference_cost since I.2 and
    nothing wrote it.

    The column existed, the migration shipped, the parser had tests, and
    RECONCILABLE_COLUMNS is what decides whether any of that reaches a
    row: every pass fetched the figure and threw it away. A value parsed
    on the wire and dropped before the write is a helper with no call
    site wearing a different hat, which is the lens this phase keeps
    finding things with.

    Verbatim, as the string it arrived as. Nothing computes with it: the
    ceiling meters the OpenRouter balance and a direct provider bill is
    not money OpenRouter can decline, so the figure exists to be matched
    against that provider's own invoice line and a float would reformat
    the number being matched.
    """
    respx.get(GENERATION_URL).respond(
        json=audit_body("gen-byok", upstream_inference_cost=0.0042, is_byok=True)
    )
    run_id = seed(conn, [completed_result("model/a", "gen-byok")])

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    row = row_of(conn, run_id)
    assert row["upstream_inference_cost_usd"] == "0.0042"
    # And it is not the charge OpenRouter made, which is a different bill
    # in a different place.
    assert row["billed_cost_usd"] == 0.0015


@respx.mock
async def test_an_unreported_byok_figure_does_not_erase_one_captured_in_band(conn):
    """COALESCE like the text columns, not the money CASE. Absence on the
    generation record means "not reported", never "known to be nothing",
    and there is no poisoned-value case to clear because nothing computes
    with it."""
    respx.get(GENERATION_URL).respond(json=audit_body("gen-a"))
    run_id = seed(
        conn,
        [completed_result("model/a", "gen-a", upstream_inference_cost_usd="0.99")],
    )

    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=io.StringIO())

    assert row_of(conn, run_id)["upstream_inference_cost_usd"] == "0.99"


@respx.mock
async def test_review_repro_a_digest_placeholder_survives_reconciliation(conn):
    """WINDOW: one result row whose request_json carries a Phase K digest
    reference, across one reconcile pass that rewrites its billing.

    THE MODULE HAS NO PHASE K DIFF AND THAT IS EXACTLY WHY IT NEEDS A
    PHASE K TEST. reconcile writes only RECONCILABLE_COLUMNS, and
    request_json is not one of them, so the placeholder should be
    untouched. Nothing asserted that, and the placeholder is the whole
    of rule two's reconstruction claim: the wire bytes are recoverable
    from the digest plus the stored content plus the recorded
    composition, and a pass that rewrote or dropped the recorded
    composition would break that silently for every attached comparison
    it touched.

    Reconcile runs against a live bench on a second connection, so this
    is not a hypothetical interleaving: it is the documented way to use
    the tool.
    """
    placeholder = "<<attachment content: sha256 " + "ab" * 32 + ", 27 characters>>"
    recorded = json.dumps(
        {
            "model": "model/alpha",
            "messages": [{"role": "user", "content": "summarize\n\n" + placeholder}],
            "max_tokens": 16384,
        }
    )
    run_id = seed(
        conn,
        [completed_result("model/alpha", "gen-k", request_json=recorded)],
    )
    respx.get(GENERATION_URL).respond(json=audit_body("gen-k", total_cost=0.00031))

    out = io.StringIO()
    # trust_env=False like every client in this file; see the module
    # docstring for the poisoned-proxy reason.
    async with httpx.AsyncClient(trust_env=False) as client:
        await reconcile(conn, client, apply=True, delay_s=0, out=out)

    row = row_of(conn, run_id)
    # The money moved, so the pass actually did something and the
    # assertion below is not about a no-op.
    assert row["billed_cost_usd"] == 0.00031
    # And the record of what was sent is byte-for-byte what it was.
    assert row["request_json"] == recorded
    assert placeholder in row["request_json"]
    # Never the content itself, which reconcile has no route to anyway
    # and which this asserts so a future column addition cannot quietly
    # acquire one.
    assert "extracted_text" not in row["request_json"]
