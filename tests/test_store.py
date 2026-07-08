import sqlite3

import pytest

from bench import store


@pytest.fixture
def db():
    conn = store.connect(":memory:")
    yield conn
    conn.close()


def make_result(model="test/model", **overrides):
    base = {
        "model": model,
        "response_text": "hello",
        "latency_ms": 123.4,
        "prompt_tokens": 13,
        "completion_tokens": 8,
        "error": None,
    }
    base.update(overrides)
    return base


def test_save_and_list_prompts(db):
    store.save_prompt(db, "greeting", "Say hello.")
    store.save_prompt(db, "arithmetic", "What is 2+2?")

    prompts = store.list_prompts(db)

    assert [p["name"] for p in prompts] == ["arithmetic", "greeting"]
    assert prompts[1]["text"] == "Say hello."
    assert prompts[1]["created_at"]


def test_duplicate_prompt_name_raises(db):
    store.save_prompt(db, "greeting", "Say hello.")
    with pytest.raises(sqlite3.IntegrityError):
        store.save_prompt(db, "greeting", "different text, same name")


def test_save_run_is_atomic(db):
    # model NULL violates NOT NULL mid-batch; the run row inserted in the
    # same transaction must roll back with it.
    bad_batch = [make_result(), make_result(model=None)]
    with pytest.raises(sqlite3.IntegrityError):
        store.save_run(db, "prompt", bad_batch)

    assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0


def test_delete_prompt_nulls_link_but_keeps_run(db):
    prompt = store.save_prompt(db, "greeting", "Say hello.")
    run_id = store.save_run(db, "Say hello.", [make_result()], prompt["id"])

    assert store.delete_prompt(db, prompt["id"]) is True

    run = store.get_run(db, run_id)
    assert run is not None
    assert run["prompt_id"] is None
    assert run["prompt_text"] == "Say hello."
    assert len(run["results"]) == 1
    assert run["results"][0]["response_text"] == "hello"


def test_list_runs_most_recent_first_with_models(db):
    first = store.save_run(db, "first prompt", [make_result(model="a/one")])
    second = store.save_run(
        db, "second prompt", [make_result(model="b/two"), make_result(model="c/three")]
    )

    runs = store.list_runs(db)

    assert [r["id"] for r in runs] == [second, first]
    assert runs[0]["models"] == ["b/two", "c/three"]
    assert runs[1]["models"] == ["a/one"]


OLD_SCHEMA = """
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NULL REFERENCES prompts(id) ON DELETE SET NULL,
    prompt_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    response_text TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT
);
"""


def test_connect_upgrades_pre_grouping_schema():
    # Shared-cache in-memory URI: the keeper connection holds the DB
    # alive while store.connect opens a second connection to the same
    # in-memory database, exactly like reopening an old bench.db file.
    uri = "file:pre_grouping_upgrade?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.executescript(OLD_SCHEMA)
    keeper.execute(
        "INSERT INTO runs (prompt_text, created_at) VALUES (?, ?)",
        ("legacy prompt", "2026-01-01T00:00:00+00:00"),
    )
    keeper.execute(
        "INSERT INTO results (run_id, model, response_text) VALUES (1, 'old/model', 'hi')"
    )
    keeper.commit()

    conn = store.connect(uri)
    try:
        run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        result_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
        assert "group_id" in run_cols
        assert "cost_usd" in result_cols

        # Legacy rows survive and render as ungrouped run entries.
        entries = store.list_runs(conn)
        assert len(entries) == 1
        assert entries[0]["type"] == "run"
        assert entries[0]["prompt_text"] == "legacy prompt"
        assert entries[0]["models"] == ["old/model"]
        legacy = store.get_run(conn, entries[0]["id"])
        assert legacy["results"][0]["cost_usd"] is None

        # New-style grouped writes work on the upgraded database.
        group_id = store.create_group(conn)
        store.save_run(conn, "new prompt", [make_result()], group_id=group_id)
        entries = store.list_runs(conn)
        assert [e["type"] for e in entries] == ["group", "run"]
    finally:
        conn.close()
        keeper.close()


def test_list_runs_groups_collapse_and_order_newest_first(db):
    lone_before = store.save_run(db, "ungrouped early", [make_result(model="a/one")])
    group_id = store.create_group(db)
    r1 = store.save_run(db, "grouped", [make_result(model="b/two")], group_id=group_id)
    r2 = store.save_run(db, "grouped", [make_result(model="c/three")], group_id=group_id)
    lone_after = store.save_run(db, "ungrouped late", [make_result(model="d/four")])

    entries = store.list_runs(db)

    assert [e["type"] for e in entries] == ["run", "group", "run"]
    assert entries[0]["id"] == lone_after
    assert entries[1]["id"] == group_id
    assert entries[1]["models"] == ["b/two", "c/three"]
    assert entries[1]["run_ids"] == [r1, r2]
    assert entries[2]["id"] == lone_before


def test_list_runs_limit_returns_newest_entries(db):
    ids = [store.save_run(db, f"p{i}", [make_result()]) for i in range(3)]

    entries = store.list_runs(db, limit=2)

    assert [e["id"] for e in entries] == [ids[2], ids[1]]


def test_list_runs_limit_keeps_whole_groups(db):
    group_id = store.create_group(db)
    r1 = store.save_run(db, "grouped", [make_result(model="a/one")], group_id=group_id)
    lone = store.save_run(db, "lone", [make_result(model="b/two")])
    r2 = store.save_run(db, "grouped", [make_result(model="c/three")], group_id=group_id)

    entries = store.list_runs(db, limit=2)

    # The id scan stops at the lone run, but the group entry (emitted
    # at its newest member r2) must still carry the older member r1: a
    # page boundary must never split a comparison.
    assert [e["type"] for e in entries] == ["group", "run"]
    assert entries[0]["run_ids"] == [r1, r2]
    assert entries[0]["models"] == ["a/one", "c/three"]
    assert entries[1]["id"] == lone


def test_list_runs_empty_history_is_empty_list(db):
    assert store.list_runs(db) == []


def test_get_group_returns_runs_with_results_in_id_order(db):
    group_id = store.create_group(db)
    r1 = store.save_run(db, "p", [make_result(model="b/two")], group_id=group_id)
    r2 = store.save_run(db, "p", [make_result(model="c/three")], group_id=group_id)

    group = store.get_group(db, group_id)

    assert group["id"] == group_id
    assert [r["id"] for r in group["runs"]] == [r1, r2]
    assert group["runs"][0]["results"][0]["model"] == "b/two"
    assert store.get_group(db, 999) is None
    assert store.group_exists(db, group_id) is True
    assert store.group_exists(db, 999) is False


SCHEMA_4A = """
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE groups (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NULL REFERENCES prompts(id) ON DELETE SET NULL,
    group_id INTEGER NULL REFERENCES groups(id),
    prompt_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    response_text TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT,
    cost_usd REAL
);
"""


SCHEMA_53 = """
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE groups (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NULL REFERENCES prompts(id) ON DELETE SET NULL,
    group_id INTEGER NULL REFERENCES groups(id),
    prompt_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    response_text TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT,
    cost_usd REAL,
    ttft_ms REAL
);
"""


def test_connect_upgrades_53_schema_with_max_tokens():
    uri = "file:pre_budget_upgrade?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.executescript(SCHEMA_53)
    keeper.execute(
        "INSERT INTO runs (prompt_text, created_at) VALUES (?, ?)",
        ("5.3-era prompt", "2026-07-07T00:00:00+00:00"),
    )
    keeper.execute(
        "INSERT INTO results (run_id, model, response_text, ttft_ms)"
        " VALUES (1, 'a/model', 'hi', 312.5)"
    )
    keeper.commit()

    conn = store.connect(uri)
    try:
        result_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
        assert "max_tokens" in result_cols

        # Pre-budget rows survive with an honestly unknown budget, not
        # a backfilled guess.
        run = store.get_run(conn, 1)
        assert run["results"][0]["ttft_ms"] == 312.5
        assert run["results"][0]["max_tokens"] is None

        # Budget-carrying results persist on the upgraded database.
        run_id = store.save_run(conn, "budgeted", [make_result(max_tokens=16384)])
        saved = store.get_run(conn, run_id)
        assert saved["results"][0]["max_tokens"] == 16384
    finally:
        conn.close()
        keeper.close()


def test_connect_upgrades_4a_schema_with_ttft():
    uri = "file:pre_streaming_upgrade?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.executescript(SCHEMA_4A)
    keeper.execute(
        "INSERT INTO runs (prompt_text, created_at) VALUES (?, ?)",
        ("4a-era prompt", "2026-07-01T00:00:00+00:00"),
    )
    keeper.execute(
        "INSERT INTO results (run_id, model, response_text, cost_usd)"
        " VALUES (1, 'a/model', 'hi', 2.9e-05)"
    )
    keeper.commit()

    conn = store.connect(uri)
    try:
        result_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
        assert "ttft_ms" in result_cols

        # 4a rows survive: cost intact, ttft simply unknown.
        run = store.get_run(conn, 1)
        assert run["results"][0]["cost_usd"] == 2.9e-05
        assert run["results"][0]["ttft_ms"] is None

        # Streamed-shaped results persist on the upgraded database.
        run_id = store.save_run(conn, "streamed", [make_result(ttft_ms=312.5)])
        saved = store.get_run(conn, run_id)
        assert saved["results"][0]["ttft_ms"] == 312.5
    finally:
        conn.close()
        keeper.close()
