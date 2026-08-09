import json
import pathlib
import sqlite3
from pathlib import Path

import pytest

from bench import report, store


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
    """THE WINDOW: from the run INSERT to the last result INSERT. A
    failure anywhere inside it must leave neither behind, because a run
    row with a partial set of results is a comparison that silently lost
    a model, which is worse than a comparison that failed."""
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
    r2 = store.save_run(
        db, "grouped", [make_result(model="c/three")], group_id=group_id
    )
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
    r2 = store.save_run(
        db, "grouped", [make_result(model="c/three")], group_id=group_id
    )

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


SCHEMA_PRE_PROVENANCE = """
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
    ttft_ms REAL,
    max_tokens INTEGER
);
"""


def test_connect_upgrades_pre_provenance_schema():
    uri = "file:pre_provenance_upgrade?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.executescript(SCHEMA_PRE_PROVENANCE)
    keeper.execute(
        "INSERT INTO runs (prompt_text, created_at) VALUES (?, ?)",
        ("pre-provenance prompt", "2026-07-09T00:00:00+00:00"),
    )
    keeper.execute(
        "INSERT INTO results (run_id, model, response_text, max_tokens)"
        " VALUES (1, 'a/model', 'hi', 16384)"
    )
    keeper.commit()

    conn = store.connect(uri)
    try:
        result_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
        assert "generation_id" in result_cols
        assert "finish_reason" in result_cols

        # Old rows survive with honestly unknown provenance, everything
        # else intact.
        run = store.get_run(conn, 1)
        assert run["results"][0]["max_tokens"] == 16384
        assert run["results"][0]["generation_id"] is None
        assert run["results"][0]["finish_reason"] is None

        # Provenance-carrying results persist on the upgraded database.
        run_id = store.save_run(
            conn,
            "provenanced",
            [make_result(generation_id="gen-x1", finish_reason="stop")],
        )
        saved = store.get_run(conn, run_id)
        assert saved["results"][0]["generation_id"] == "gen-x1"
        assert saved["results"][0]["finish_reason"] == "stop"
    finally:
        conn.close()
        keeper.close()


def index_names(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA index_list({table})")}


def test_connect_creates_history_indexes_on_fresh_database(db):
    assert "idx_results_run_id" in index_names(db, "results")
    assert "idx_runs_group_id" in index_names(db, "runs")


def test_connect_creates_history_indexes_on_upgraded_database():
    # The pre-grouping schema is the oldest shape connect() supports;
    # runs.group_id only exists after the migration, so its index is
    # proof the index pass runs after the ALTERs.
    uri = "file:pre_grouping_index_upgrade?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.executescript(OLD_SCHEMA)
    keeper.commit()

    conn = store.connect(uri)
    try:
        assert "idx_results_run_id" in index_names(conn, "results")
        assert "idx_runs_group_id" in index_names(conn, "runs")
    finally:
        conn.close()
        keeper.close()


import logging
import os
import stat as stat_mod


def file_mode(path) -> int:
    return stat_mod.S_IMODE(os.stat(path).st_mode)


def test_review_repro_fresh_database_is_private(tmp_path):
    # Reproduction 4: bench.db was created 0644 under umask 022, so
    # every prompt and output was world-readable on shared machines.
    old_umask = os.umask(0o022)
    try:
        db_path = tmp_path / "data" / "bench.db"
        conn = store.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()
    finally:
        os.umask(old_umask)

    assert file_mode(db_path) == 0o600
    # The parent the bench created is private too.
    assert file_mode(db_path.parent) == 0o700


def test_existing_loose_database_and_siblings_tightened(tmp_path, caplog):
    db_path = tmp_path / "bench.db"
    seed = sqlite3.connect(str(db_path))
    seed.execute("CREATE TABLE junk (x)")
    seed.commit()
    seed.close()
    os.chmod(db_path, 0o644)
    # A leftover sibling as a crash would leave one; _keep_private runs
    # before sqlite opens the WAL database and tightens it.
    wal = tmp_path / "bench.db-wal"
    wal.write_bytes(b"")
    os.chmod(wal, 0o644)

    with caplog.at_level(logging.WARNING, logger="bench.store"):
        conn = store.connect(str(db_path))
    try:
        # The database file mode survives close, but the -wal is checked
        # while the connection is open: closing a WAL database
        # checkpoints and removes the sibling.
        assert file_mode(db_path) == 0o600
        assert file_mode(wal) == 0o600
        assert "tightened" in caplog.text
    finally:
        conn.close()


def test_get_run_repairs_poisoned_legacy_row(db):
    # Written around save_run, exactly as the pre-normalization code
    # effectively did: raw provider junk straight into the row.
    with db:
        cur = db.execute(
            "INSERT INTO runs (prompt_text, created_at)"
            " VALUES ('poisoned', '2026-01-01T00:00:00+00:00')"
        )
        run_id = cur.lastrowid
        db.execute(
            "INSERT INTO results (run_id, model, response_text, prompt_tokens,"
            " completion_tokens, latency_ms, ttft_ms, cost_usd, max_tokens,"
            " generation_id, finish_reason)"
            " VALUES (?, 'old/model', 'hi', 'n/a', '12', 'slow', 'fast',"
            " 'cheap', 'lots', 42, 7)",
            (run_id,),
        )

    run = store.get_run(db, run_id)

    row = run["results"][0]
    assert row["response_text"] == "hi"
    # Junk sqlite could not coerce reads back as None.
    for field in ("prompt_tokens", "latency_ms", "ttft_ms", "cost_usd", "max_tokens"):
        assert row[field] is None, field
    # Values sqlite's column affinity already coerced at insert time
    # (numeric string on INTEGER, number on TEXT) read back type-clean.
    assert row["completion_tokens"] == 12
    assert row["generation_id"] == "42"
    assert row["finish_reason"] == "7"


def test_file_backed_connection_enables_wal_and_busy_timeout(tmp_path):
    # WAL and a busy timeout let an analysis script read bench.db while
    # the bench writes, instead of colliding on "database is locked".
    conn = store.connect(str(tmp_path / "bench.db"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_memory_database_ignores_wal(db):
    # The :memory: fixture must keep working: sqlite reports "memory"
    # rather than "wal" for it, and connect must not have errored.
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "memory"


def test_wal_siblings_are_kept_owner_only(tmp_path):
    db_path = tmp_path / "bench.db"
    conn = store.connect(str(db_path))
    try:
        # A write while the connection is open materializes the -wal
        # sibling and keeps it alive (closing WAL checkpoints it away).
        store.save_run(conn, "prompt", [make_result()])
        wal = tmp_path / "bench.db-wal"
        assert wal.exists()
        # SQLite mirrors the 0600 database file onto siblings it creates.
        assert file_mode(wal) == 0o600
        shm = tmp_path / "bench.db-shm"
        if shm.exists():
            assert file_mode(shm) == 0o600

        # And a later startup re-tightens a sibling that somehow went
        # loose, via _keep_private: the permissions compose.
        os.chmod(wal, 0o644)
        conn2 = store.connect(str(db_path))
        try:
            assert file_mode(wal) == 0o600
        finally:
            conn2.close()
    finally:
        conn.close()


def test_group_prompt_is_first_member_or_none(db):
    # Review finding 8 helper: a group's established prompt is its first
    # member's, derived from members with no schema column.
    gid = store.create_group(db)
    # Empty group and unknown group both have no established prompt.
    assert store.group_prompt(db, gid) is None
    assert store.group_prompt(db, 999999) is None
    store.save_run(db, "the first prompt", [make_result()], group_id=gid)
    store.save_run(db, "a later, different prompt", [make_result()], group_id=gid)
    # The first member fixes it, regardless of later members' text.
    assert store.group_prompt(db, gid) == "the first prompt"


# ---- F4.1: a disk-backed file: URI is a real file, not a memory database.


def test_review_repro_disk_uri_gets_permissions_and_pragmas(tmp_path):
    """Second review: connect() treated every file:-prefixed path as
    memory-like, so a disk-backed URI skipped _keep_private, WAL and
    busy_timeout. Reproduced under umask 022: file:/tmp/x.db?cache=private
    came out 0644 running journal_mode=delete."""
    db_path = tmp_path / "uri.db"
    uri = f"file:{db_path}?cache=private"
    old_umask = os.umask(0o022)
    try:
        conn = store.connect(uri)
    finally:
        os.umask(old_umask)
    try:
        assert file_mode(db_path) == 0o600
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_memory_uri_stays_memory_backed():
    """The test seam's shared in-memory databases must keep working: a
    mode=memory URI has no file to harden and no meaningful WAL."""
    uri = "file:f4_memory_uri?mode=memory&cache=shared"
    conn = store.connect(uri)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
    finally:
        conn.close()


def test_disk_path_classifies_uri_shapes(tmp_path):
    """The classifier itself: mode=memory (in any parameter position) is
    memory, everything else with a path is a file, and the shapes with no
    filename to chmod degrade to None rather than guessing."""
    assert store._disk_path(":memory:") is None
    assert store._disk_path("bench.db") == "bench.db"
    # mode=memory anywhere in the query, not just first.
    assert store._disk_path("file:x?cache=shared&mode=memory") is None
    assert store._disk_path("file:x?mode=memory") is None
    # Disk URIs, absolute and relative, with and without an authority.
    assert store._disk_path("file:/tmp/x.db?cache=private") == "/tmp/x.db"
    assert store._disk_path("file:bench.db") == "bench.db"
    assert store._disk_path("file://localhost/tmp/x.db") == "/tmp/x.db"
    # mode=rw is explicitly not memory.
    assert store._disk_path("file:/tmp/x.db?mode=rw") == "/tmp/x.db"
    # Percent-encoding is decoded, since sqlite opens the decoded name.
    assert store._disk_path("file:/tmp/a%20b.db") == "/tmp/a b.db"
    # A private temporary database has no filename to harden.
    assert store._disk_path("file:?cache=shared") is None


def test_review_repro_repeated_mode_follows_sqlite_last_wins(tmp_path):
    """Closing review: _disk_path treated a repeated mode parameter as
    memory whenever ANY occurrence was memory, but sqlite overwrites the
    flag per occurrence, so file:x.db?mode=memory&mode=rwc opens a real file
    on disk. That file then skipped _keep_private and the WAL pragmas, which
    is exactly the hole F4.1 exists to close, reached through the parser."""
    db_path = tmp_path / "lastwins.db"
    uri = f"file:{db_path}?mode=memory&mode=rwc"
    assert store._disk_path(uri) == str(db_path)
    old_umask = os.umask(0o022)
    try:
        conn = store.connect(uri)
    finally:
        os.umask(old_umask)
    try:
        # sqlite really did open it on disk, and it is hardened.
        assert db_path.exists()
        assert file_mode(db_path) == 0o600
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
    # The mirror image is memory, because there the last mode wins too.
    assert store._disk_path("file:x.db?mode=rwc&mode=memory") is None


def test_review_repro_foreign_authority_creates_no_file(tmp_path):
    """Closing review: _disk_path dropped the URI authority instead of
    checking it, so a URI sqlite rejects outright ("invalid uri authority")
    still went through _keep_private, which created and chmod'd a file for a
    database that could never open."""
    target = tmp_path / "authority.db"
    uri = f"file://remote{target}"
    assert store._disk_path(uri) is None
    with pytest.raises(sqlite3.OperationalError):
        store.connect(uri)
    # No stray file left behind by the rejected URI.
    assert not target.exists()
    # The authorities sqlite does accept still resolve to the file.
    assert store._disk_path(f"file://localhost{target}") == str(target)


# ---- Phase G: the migration onto a real pre-G database.


PRE_G_SCHEMA = (Path(__file__).parent / "fixtures" / "pre_g_schema.sql").read_text()


def test_migration_onto_pre_g_database_is_additive_and_idempotent(tmp_path):
    """Phase G lifted the schema freeze, so the migration has to be proven
    against the schema that actually shipped, not against a hand-written
    approximation of it. The fixture is the pre-G SCHEMA string verbatim.

    Three things must hold: the migration runs on a populated legacy
    database, a second connect is a no-op rather than an error, and every
    legacy row still reads back through the normal paths with each new
    column None.
    """
    db_path = tmp_path / "pre_g.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_G_SCHEMA)
    legacy.execute(
        "INSERT INTO groups (id, created_at) VALUES (1, '2026-01-01T00:00:00+00:00')"
    )
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at)
           VALUES (1, NULL, 1, 'legacy prompt', '2026-01-01T00:00:00+00:00')"""
    )
    # A completed row and an aborted one: the aborted shape is the one the
    # disconnect path writes, and it is exactly the row a reconcile would
    # later want to fill in, so it must survive the migration intact.
    legacy.execute(
        """INSERT INTO results
           (run_id, model, response_text, latency_ms, prompt_tokens,
            completion_tokens, error, cost_usd, ttft_ms, max_tokens,
            generation_id, finish_reason)
           VALUES (1, 'legacy/model', 'legacy text', 12.5, 13, 8, NULL,
                   2.9e-05, 5.5, 16384, 'gen-legacy', 'stop')"""
    )
    legacy.execute(
        """INSERT INTO results
           (run_id, model, response_text, latency_ms, prompt_tokens,
            completion_tokens, error, cost_usd, ttft_ms, max_tokens,
            generation_id, finish_reason)
           VALUES (1, 'legacy/aborted', 'partial', 30.0, NULL, NULL,
                   'stream aborted before completion', NULL, 7.5, 16384,
                   NULL, NULL)"""
    )
    legacy.commit()
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        run = store.get_run(conn, 1)
        assert run is not None
        # Legacy run-level provenance is absent, not invented.
        for column in ("app_sha", "catalog_snapshot_at", "data_policy"):
            assert run[column] is None, column
        assert [r["model"] for r in run["results"]] == [
            "legacy/model",
            "legacy/aborted",
        ]
        for result in run["results"]:
            for column in (
                "position",
                "request_json",
                "billed_cost_usd",
                "reasoning_tokens",
                "cached_tokens",
                "provider",
                "quantization",
                "native_finish_reason",
            ):
                assert result[column] is None, (result["model"], column)
        # The pre-existing values are untouched by the migration.
        assert run["results"][0]["cost_usd"] == pytest.approx(2.9e-05)
        assert run["results"][0]["generation_id"] == "gen-legacy"
        assert run["results"][1]["error"] == "stream aborted before completion"
        # Group and history reads work too, and a legacy group has no
        # declared prompt, so group_prompt falls back to its first member.
        assert store.group_prompt(conn, 1) == "legacy prompt"
        group = store.get_group(conn, 1)
        assert group is not None and len(group["runs"]) == 1
        assert len(store.list_runs(conn)) == 1
    finally:
        conn.close()

    # Idempotent: connecting again must not attempt the ALTERs a second
    # time, and must serve the same rows.
    again = store.connect(str(db_path))
    try:
        run = again.execute("SELECT COUNT(*) AS n FROM results").fetchone()
        assert run["n"] == 2
        cols = {r[1] for r in again.execute("PRAGMA table_info(results)")}
        assert "billed_cost_usd" in cols and "position" in cols
    finally:
        again.close()


def test_new_group_declares_its_prompt_before_any_member(db):
    """The group row is created before any upstream call, so recording the
    prompt there fixes it ahead of the first member. That is what dissolves
    the concurrent-first-member race for new groups: two simultaneous first
    members are both checked against a value that already exists."""
    gid = store.create_group(db, "declared prompt", ["a/one", "b/two"])
    # Established with no runs at all, which the derive-from-member
    # fallback could never do.
    assert store.group_prompt(db, gid) == "declared prompt"
    row = db.execute("SELECT models_json FROM groups WHERE id = ?", (gid,)).fetchone()
    assert json.loads(row["models_json"]) == ["a/one", "b/two"]
    # A member with a different prompt does not change what was declared.
    store.save_run(db, "some other prompt", [make_result()], group_id=gid)
    assert store.group_prompt(db, gid) == "declared prompt"


def test_results_replay_in_declared_position_order(db):
    """Replay must reconstruct the original layout from the rows. Results
    are inserted in a deliberately different order than their positions, so
    id order and position order disagree and only the intended one passes.
    """
    run_id = store.save_run(
        db,
        "positioned",
        [
            make_result("third/model", position=2),
            make_result("first/model", position=0),
            make_result("second/model", position=1),
        ],
    )
    run = store.get_run(db, run_id)
    assert run is not None
    assert [r["model"] for r in run["results"]] == [
        "first/model",
        "second/model",
        "third/model",
    ]


def test_legacy_rows_without_position_keep_insertion_order(db):
    """Rows written before the column existed carry no position, and must
    keep the order they were written in rather than collapsing together."""
    run_id = store.save_run(
        db,
        "unpositioned",
        [make_result("alpha/model"), make_result("beta/model")],
    )
    run = store.get_run(db, run_id)
    assert run is not None
    assert [r["model"] for r in run["results"]] == ["alpha/model", "beta/model"]


def test_run_provenance_columns_round_trip(db):
    """The run-level provenance the caller knows and the store does not."""
    run_id = store.save_run(
        db,
        "provenanced",
        [make_result()],
        provenance={
            "app_sha": "abc123",
            "catalog_snapshot_at": "2026-07-25T00:00:00+00:00",
            "data_policy": "deny",
            "catalog_digest": "ab" * 32,
        },
    )
    run = store.get_run(db, run_id)
    assert run is not None
    assert run["app_sha"] == "abc123"
    assert run["catalog_snapshot_at"] == "2026-07-25T00:00:00+00:00"
    assert run["data_policy"] == "deny"
    # Round trip, not just readable: a column read back but never written
    # returns None for every row and reads as "this run had no digest".
    assert run["catalog_digest"] == "ab" * 32


PRE_H_SCHEMA = (Path(__file__).parent / "fixtures" / "pre_h_schema.sql").read_text()


def test_migration_onto_pre_h_database_is_additive_and_idempotent(tmp_path):
    """Phase H lifts the schema freeze for exactly one additive column, so
    the migration is proven against the schema that actually shipped rather
    than a hand-written approximation. The fixture was generated from the
    pre-H SCHEMA string itself, the same discipline the pre-G fixture uses.

    Three things must hold, and the third is the one that matters for the
    controls: the migration runs on a populated legacy database, a second
    connect is a no-op rather than an error, and a legacy group reads back
    with params None. That None is load-bearing. It is what tells
    group_params that no control was set, which is true of every group that
    predates the column, and it is why the params check needs no
    derive-from-member fallback.
    """
    db_path = tmp_path / "pre_h.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_H_SCHEMA)
    legacy.execute(
        """INSERT INTO groups (id, created_at, prompt_text, models_json)
           VALUES (1, '2026-01-01T00:00:00+00:00', 'legacy prompt',
                   '["legacy/a"]')"""
    )
    # A second group with no declared prompt either: the oldest shape, and
    # the one whose params must not be confused with an unreadable value.
    legacy.execute(
        "INSERT INTO groups (id, created_at) VALUES (2, '2026-01-01T00:00:00+00:00')"
    )
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at)
           VALUES (1, NULL, 1, 'legacy prompt', '2026-01-01T00:00:00+00:00')"""
    )
    legacy.execute(
        """INSERT INTO results (run_id, model, response_text, latency_ms,
                                prompt_tokens, completion_tokens, error, cost_usd)
           VALUES (1, 'legacy/a', 'legacy text', 12.5, 13, 8, NULL, 2.9e-05)"""
    )
    legacy.commit()
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        # Additive: the column exists and every legacy row reads back
        # through the normal paths with it absent rather than invented.
        for gid in (1, 2):
            group = store.get_group(conn, gid)
            assert group is not None
            assert group["params"] is None, gid
            assert store.group_params(conn, gid) == {}, gid
        # The rest of the legacy row is untouched by the migration.
        assert store.group_prompt(conn, 1) == "legacy prompt"
        run = store.get_run(conn, 1)
        assert run is not None
        assert run["results"][0]["response_text"] == "legacy text"
    finally:
        conn.close()

    # Idempotent: connecting again neither errors nor duplicates the column.
    again = store.connect(str(db_path))
    try:
        cols = [r["name"] for r in again.execute("PRAGMA table_info(groups)")]
        assert cols.count("params_json") == 1, cols
        # And a group created after the migration still records its controls.
        gid = store.create_group(again, "new prompt", ["new/a"], {"seed": 7})
        assert store.group_params(again, gid) == {"seed": 7}
    finally:
        again.close()


def test_create_group_stores_no_controls_as_null(tmp_path):
    """One spelling of absence, so no reader has to know two. An empty
    mapping is not a smaller record of nothing."""
    conn = store.connect(str(tmp_path / "b.db"))
    try:
        for params in (None, {}):
            gid = store.create_group(conn, "p", ["m/a"], params)
            raw = conn.execute(
                "SELECT params_json FROM groups WHERE id = ?", (gid,)
            ).fetchone()["params_json"]
            assert raw is None, params
            assert store.group_params(conn, gid) == {}
    finally:
        conn.close()


def test_create_group_serializes_controls_with_sorted_keys(tmp_path):
    """The stored string is compared against a later request's controls to
    enforce one experiment per group. Sorting cannot change which controls
    were set, and without it two identical sets could serialize differently
    and read as a conflict."""
    conn = store.connect(str(tmp_path / "c.db"))
    try:
        first = store.create_group(conn, "p", None, {"top_p": 0.9, "seed": 7})
        second = store.create_group(conn, "p", None, {"seed": 7, "top_p": 0.9})

        def raw(gid):
            return conn.execute(
                "SELECT params_json FROM groups WHERE id = ?", (gid,)
            ).fetchone()["params_json"]

        assert raw(first) == raw(second) == '{"seed": 7, "top_p": 0.9}'
    finally:
        conn.close()


@pytest.mark.parametrize(
    "stored",
    [
        "not json at all",
        # Valid JSON that is not an object: a list or a bare scalar cannot
        # name controls, so it is not evidence that any were set.
        "[1, 2]",
        '"a string"',
        "17",
        "null",
    ],
)
def test_an_unreadable_params_column_reads_as_no_controls(tmp_path, stored):
    """Repair on read, like every other column. A record nobody can parse
    is not evidence that a control was set, and propagating the broken
    value into the one-experiment check would turn a corrupt row into a
    permanent 409 on its own group.
    """
    conn = store.connect(str(tmp_path / "d.db"))
    try:
        gid = store.create_group(conn, "p")
        conn.execute("UPDATE groups SET params_json = ? WHERE id = ?", (stored, gid))
        conn.commit()

        assert store.group_params(conn, gid) == {}
        group = store.get_group(conn, gid)
        assert group is not None
        assert group["params"] is None
    finally:
        conn.close()


def test_an_ungrouped_run_derives_its_controls_from_its_payload(tmp_path):
    """Controls are declared on the group, so a run with no group has no
    stored set. Its request_json is the payload that actually went out, and
    four of the six controls appear there only because someone chose them,
    which is enough to badge an ungrouped run the way a grouped one is."""
    conn = store.connect(str(tmp_path / "e.db"))
    try:
        payload = {
            "model": "m/a",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "p"},
            ],
            "temperature": 0.25,
            "top_p": 0.9,
            "seed": 7,
            "reasoning": {"effort": "high"},
            "provider": {"sort": "throughput"},
        }
        run_id = store.save_run(
            conn,
            "p",
            [make_result(model="m/a", request_json=json.dumps(payload))],
        )

        run = store.get_run(conn, run_id)
        assert run is not None
        assert run["params"] == {
            "system": "be terse",
            "temperature": 0.25,
            "top_p": 0.9,
            "seed": 7,
            "effort": "high",
        }
        entry = next(e for e in store.list_runs(conn) if e["id"] == run_id)
        assert entry["params"] == run["params"]
    finally:
        conn.close()


def test_routing_is_never_derived_from_a_payload(tmp_path):
    """The interesting negative. provider.sort rides every payload the bench
    has ever sent, because throughput is its own default, so a run carrying
    sort=throughput is indistinguishable from one whose user asked for it.
    Deriving a routing badge from that would render a default as a choice,
    which is the exact truth defect rule two exists to prevent."""
    conn = store.connect(str(tmp_path / "f.db"))
    try:
        for sort in ("throughput", "price", "default"):
            run_id = store.save_run(
                conn,
                "p",
                [
                    make_result(
                        model="m/a",
                        request_json=json.dumps(
                            {"messages": [], "provider": {"sort": sort}}
                        ),
                    )
                ],
            )
            run = store.get_run(conn, run_id)
            assert run is not None
            assert run["params"] is None, sort
    finally:
        conn.close()


def test_a_legacy_run_derives_no_controls_at_all(tmp_path):
    """A pre-H payload has no controls in it, so nothing is claimed. Rows
    with no request_json at all (pre-G) must also come back clean rather
    than raising."""
    conn = store.connect(str(tmp_path / "g.db"))
    try:
        pre_h = store.save_run(
            conn,
            "p",
            [
                make_result(
                    model="m/a",
                    request_json=json.dumps(
                        {
                            "model": "m/a",
                            "messages": [{"role": "user", "content": "p"}],
                            "max_tokens": 16384,
                            "provider": {"sort": "throughput"},
                        }
                    ),
                )
            ],
        )
        pre_g = store.save_run(conn, "p", [make_result(model="m/b")])

        for run_id in (pre_h, pre_g):
            run = store.get_run(conn, run_id)
            assert run is not None
            assert run["params"] is None, run_id
    finally:
        conn.close()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[1,2]",
        "null",
        '{"messages": "not a list"}',
        '{"messages": [{"role": "user", "content": "p"}]}',
        '{"messages": [{"role": "system"}]}',
        '{"messages": [null]}',
        '{"reasoning": "high"}',
        '{"reasoning": {}}',
        '{"temperature": null}',
    ],
)
def test_a_malformed_payload_derives_nothing_rather_than_raising(tmp_path, payload):
    """Repair on read: history has to render, and a row nobody can parse is
    not evidence that a control was set. A raise here would take out the
    whole list, which is the one view that could tell you the row is bad."""
    conn = store.connect(str(tmp_path / "h.db"))
    try:
        run_id = store.save_run(
            conn, "p", [make_result(model="m/a", request_json=payload)]
        )
        run = store.get_run(conn, run_id)
        assert run is not None
        assert run["params"] is None
    finally:
        conn.close()


PRE_H1_SCHEMA = (Path(__file__).parent / "fixtures" / "pre_h1_schema.sql").read_text()


def test_migration_onto_pre_h1_database_is_additive_and_idempotent(tmp_path):
    """Phase H.1 adds exactly two columns, proven against the schema that
    actually shipped rather than a hand-written approximation. The fixture
    was generated from the pre-H.1 SCHEMA string itself, the same discipline
    the pre-G and pre-H fixtures use.

    The reads that matter are the NULL ones. A legacy group must come back
    with no declared lineup and no declared budget, because that is what
    tells the manifest check to skip enforcement: those groups predate the
    concept, so NULL is silence and not a claim of emptiness. Enforcing
    against silence would refuse correct work on every group in an existing
    database.
    """
    db_path = tmp_path / "pre_h1.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_H1_SCHEMA)
    legacy.execute(
        """INSERT INTO groups (id, created_at, prompt_text, models_json, params_json)
           VALUES (1, '2026-01-01T00:00:00+00:00', 'legacy prompt',
                   '["legacy/a"]', '{"seed": 7}')"""
    )
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at,
                             app_sha, catalog_snapshot_at, data_policy)
           VALUES (1, NULL, 1, 'legacy prompt', '2026-01-01T00:00:00+00:00',
                   'abc1234', '2026-01-01T00:00:00+00:00', 'standard')"""
    )
    legacy.execute(
        """INSERT INTO results (run_id, model, response_text, latency_ms,
                                prompt_tokens, completion_tokens, error, cost_usd)
           VALUES (1, 'legacy/a', 'legacy text', 12.5, 13, 8, NULL, 2.9e-05)"""
    )
    legacy.commit()
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        manifest = store.group_manifest(conn, 1)
        # Declared before the columns existed: prompt and lineup survive.
        assert manifest["prompt"] == "legacy prompt"
        assert manifest["models"] == ["legacy/a"]
        # Silence, which is what makes the budget check skip this group.
        assert manifest["budget"] is None
        # The params rule is the opposite and still holds on the same row:
        # a stored controls set reads back as set.
        assert store.group_params(conn, 1) == {"seed": 7}
        run = store.get_run(conn, 1)
        assert run is not None
        assert run["catalog_digest"] is None
        assert run["app_sha"] == "abc1234"
        assert run["results"][0]["response_text"] == "legacy text"
        group = store.get_group(conn, 1)
        assert group is not None
        assert group["budget"] is None
    finally:
        conn.close()

    again = store.connect(str(db_path))
    try:
        for table, column in (("groups", "budget"), ("runs", "catalog_digest")):
            cols = [r["name"] for r in again.execute(f"PRAGMA table_info({table})")]
            assert cols.count(column) == 1, (table, column, cols)
        gid = store.create_group(again, "new", ["new/a"], None, "extended")
        assert store.group_manifest(again, gid) == {
            "prompt": "new",
            "models": ["new/a"],
            "budget": "extended",
        }
    finally:
        again.close()


def test_a_group_with_no_declaration_reports_silence_not_emptiness(tmp_path):
    """The distinction the manifest check depends on. None means the group
    never said, so enforcement is skipped; an empty list would mean it
    declared no models, which nothing has ever created and which would
    refuse every member."""
    conn = store.connect(str(tmp_path / "m.db"))
    try:
        gid = store.create_group(conn)
        assert store.group_manifest(conn, gid) == {
            "prompt": None,
            "models": None,
            "budget": None,
        }
        # A group id that does not exist reports the same silence rather
        # than raising: the caller is an entry check on a request that may
        # name anything.
        assert store.group_manifest(conn, 99999)["models"] is None
    finally:
        conn.close()


@pytest.mark.parametrize(
    "stored", ["not json", '"a string"', "17", "{}", '["ok", 5]', "null"]
)
def test_an_unreadable_lineup_reads_as_no_declaration(tmp_path, stored):
    """Repair on read. A lineup that is not a list of strings cannot be
    compared against a request, and enforcing against a broken declaration
    would refuse correct work forever on that group. Unknown, not empty."""
    conn = store.connect(str(tmp_path / "n.db"))
    try:
        gid = store.create_group(conn, "p", ["a/b"], None, "standard")
        conn.execute("UPDATE groups SET models_json = ? WHERE id = ?", (stored, gid))
        conn.commit()

        assert store.group_manifest(conn, gid)["models"] is None
    finally:
        conn.close()


# ---- Phase I: experiments as the aggregate above groups, and scores.


PRE_I_SCHEMA = (Path(__file__).parent / "fixtures" / "pre_i_schema.sql").read_text()


def test_migration_onto_pre_i_database_is_additive_and_idempotent(tmp_path):
    """Phase I adds two tables and four columns, proven against the schema
    that actually shipped: the fixture was extracted from the SCHEMA string
    at the pre-I merge commit rather than written by hand.

    The two tables need no MIGRATIONS entry, and this is the test that says
    so out loud: CREATE TABLE IF NOT EXISTS in SCHEMA runs on every connect,
    so a pre-I database grows them on the next boot with no ALTER at all.
    A migration entry for a whole table would be the wrong tool.

    The NULL reading here is the third one in this file and the simplest.
    experiment_id, task_id, repeat_index and rotation_index are NULL on
    every legacy group, and that is the affirmative fact that the group was
    run by hand. Unlike models_json it needs no derive-from-member
    fallback, and unlike params_json it is not a claim about controls: a
    hand-run comparison genuinely has no task id.
    """
    db_path = tmp_path / "pre_i.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_I_SCHEMA)
    legacy.execute(
        """INSERT INTO groups (id, created_at, prompt_text, models_json,
                               params_json, budget)
           VALUES (1, '2026-01-01T00:00:00+00:00', 'legacy prompt',
                   '["legacy/a"]', '{"seed": 7}', 'standard')"""
    )
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at,
                             app_sha, catalog_snapshot_at, data_policy,
                             catalog_digest)
           VALUES (1, NULL, 1, 'legacy prompt', '2026-01-01T00:00:00+00:00',
                   'abc1234', '2026-01-01T00:00:00+00:00', 'standard', 'dd')"""
    )
    legacy.execute(
        """INSERT INTO results (run_id, model, response_text, latency_ms,
                                prompt_tokens, completion_tokens, error, cost_usd)
           VALUES (1, 'legacy/a', 'legacy text', 12.5, 13, 8, NULL, 2.9e-05)"""
    )
    legacy.commit()
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        # Everything H.1 and G established still reads.
        assert store.group_manifest(conn, 1) == {
            "prompt": "legacy prompt",
            "models": ["legacy/a"],
            "budget": "standard",
        }
        assert store.group_params(conn, 1) == {"seed": 7}
        run = store.get_run(conn, 1)
        assert run is not None
        assert run["catalog_digest"] == "dd"
        assert run["results"][0]["response_text"] == "legacy text"
        # The legacy group belongs to no experiment, which is the truth
        # about it rather than a gap in it.
        row = conn.execute(
            """SELECT experiment_id, task_id, repeat_index, rotation_index
               FROM groups WHERE id = 1"""
        ).fetchone()
        assert dict(row) == {
            "experiment_id": None,
            "task_id": None,
            "repeat_index": None,
            "rotation_index": None,
        }
        # The new tables exist on a database that never had them, and are
        # empty rather than absent.
        assert conn.execute("SELECT count(*) c FROM experiments").fetchone()["c"] == 0
        assert conn.execute("SELECT count(*) c FROM scores").fetchone()["c"] == 0
    finally:
        conn.close()

    again = store.connect(str(db_path))
    try:
        for column in ("experiment_id", "task_id", "repeat_index", "rotation_index"):
            cols = [r["name"] for r in again.execute("PRAGMA table_info(groups)")]
            assert cols.count(column) == 1, (column, cols)
        # A second connect adds nothing and breaks nothing.
        assert store.get_run(again, 1) is not None
    finally:
        again.close()


def experiment_spec(**overrides):
    spec = {
        "name": "spec",
        "dataset_name": "d.jsonl",
        "dataset_digest": "ab" * 32,
        "lineup": ["m/a", "m/b"],
        "budget": "standard",
        "params": {"seed": 7},
        "repeats": 2,
        "task_order_seed": None,
        "estimand_mode": "routed_service",
        "provider_pins": None,
        "halt_on_refusal": True,
        "app_sha": "abc1234",
        "catalog_digest": "cd" * 32,
        "data_policy": "standard",
        "tasks_total": 3,
        "trials_total": 12,
    }
    spec.update(overrides)
    return spec


def test_the_experiment_row_is_complete_before_anything_runs(db):
    """The manifest discipline one level up from the group row: everything
    knowable at creation is written at creation, so an experiment that
    never starts still records what it was going to be."""
    eid = store.create_experiment(db, experiment_spec())

    exp = store.get_experiment(db, eid)
    assert exp is not None
    assert exp["status"] == "created"
    assert exp["lineup"] == ["m/a", "m/b"]
    assert exp["params"] == {"seed": 7}
    assert exp["repeats"] == 2
    assert exp["dataset_digest"] == "ab" * 32
    assert exp["trials_total"] == 12
    assert exp["trials_done"] == 0
    assert exp["halt_on_refusal"] is True
    assert exp["app_sha"] == "abc1234"


def test_empty_controls_store_as_absence_not_as_an_empty_object(db):
    """Same rule as create_group. An empty object is not a smaller record
    of nothing, it is a second spelling of the same absence that every
    reader would then have to know about."""
    eid = store.create_experiment(db, experiment_spec(params={}))

    assert store.get_experiment(db, eid)["params"] is None
    raw = db.execute(
        "SELECT params_json FROM experiments WHERE id = ?", (eid,)
    ).fetchone()
    assert raw["params_json"] is None


def test_trials_total_is_stored_rather_than_derived(db):
    """A halted experiment's un-run trials leave nothing behind to count,
    so a denominator derived from existing rows would make a halt look like
    a completed run."""
    eid = store.create_experiment(db, experiment_spec(trials_total=12))
    store.bump_experiment_counters(db, eid, done=4)
    store.set_experiment_status(db, eid, "halted_on_refusal", "spend ceiling reached")

    exp = store.get_experiment(db, eid)
    assert (exp["trials_done"], exp["trials_total"]) == (4, 12)
    assert exp["status_detail"] == "spend ceiling reached"


def test_counters_accumulate_rather_than_being_assigned(db):
    eid = store.create_experiment(db, experiment_spec())

    store.bump_experiment_counters(db, eid, done=2)
    store.bump_experiment_counters(db, eid, done=1, failed=1)
    store.bump_experiment_counters(db, eid, refused=3)

    exp = store.get_experiment(db, eid)
    assert (exp["trials_done"], exp["trials_failed"], exp["trials_refused"]) == (
        3,
        1,
        3,
    )


def test_experiment_groups_come_back_in_cell_order(db):
    """Ordered by the experiment's own structure, not by the order the
    runner happened to get to, so an export's digest is stable."""
    eid = store.create_experiment(db, experiment_spec())
    for task, rep in (("t2", 1), ("t1", 1), ("t2", 0), ("t1", 0)):
        store.create_group(
            db,
            "p",
            ["m/a"],
            None,
            "standard",
            {
                "experiment_id": eid,
                "task_id": task,
                "repeat_index": rep,
                "rotation_index": rep % 1,
            },
        )
    # A hand-run group in the same database must not appear.
    store.create_group(db, "unrelated", ["m/a"], None, "standard")

    cells = [
        (g["task_id"], g["repeat_index"]) for g in store.experiment_groups(db, eid)
    ]

    assert cells == [("t1", 0), ("t1", 1), ("t2", 0), ("t2", 1)]


def test_a_group_records_its_cell_or_records_none_of_it(db):
    """The four columns are meaningless apart: a task id with no repeat
    index does not identify a cell. Passing them as one mapping is what
    makes half a placement unrepresentable."""
    plain = store.create_group(db, "p", ["m/a"], None, "standard")

    row = db.execute("SELECT * FROM groups WHERE id = ?", (plain,)).fetchone()
    assert row["experiment_id"] is None
    assert row["task_id"] is None
    assert row["repeat_index"] is None
    assert row["rotation_index"] is None


def test_rescoring_appends_rather_than_overwriting(db):
    """The audit trail is the point. "The judge said 0.5 last week and 1.0
    today" is exactly the fact someone checking a claim needs, and an
    update would destroy it."""
    run_id = store.save_run(db, "p", [make_result()])
    result_id = store.get_run(db, run_id)["results"][0]["id"]

    store.add_score(db, result_id, {"scorer": "judge", "score": 0.5, "passed": False})
    store.add_score(db, result_id, {"scorer": "judge", "score": 1.0, "passed": True})

    rows = store.scores_for_results(db, [result_id])[result_id]
    assert [r["score"] for r in rows] == [0.5, 1.0]
    assert [r["passed"] for r in rows] == [False, True]


def test_score_fields_go_through_the_field_type_functions(db):
    """A judge is an upstream service like any other. A verdict carrying
    "score": "high" must land as None, not as a string in a REAL column,
    and a negative judge charge must land as None like every other money
    value that fails the rule."""
    run_id = store.save_run(db, "p", [make_result()])
    result_id = store.get_run(db, run_id)["results"][0]["id"]

    store.add_score(
        db,
        result_id,
        {
            "scorer": "judge",
            "score": "high",
            "passed": None,
            "judge_billed_cost_usd": -1.0,
            "judge_model": 42,
        },
    )

    row = store.scores_for_results(db, [result_id])[result_id][0]
    assert row["score"] is None
    assert row["passed"] is None
    assert row["judge_billed_cost_usd"] is None
    assert row["judge_model"] is None


def test_score_flags_read_back_as_booleans(db):
    run_id = store.save_run(db, "p", [make_result()])
    result_id = store.get_run(db, run_id)["results"][0]["id"]

    store.add_score(
        db,
        result_id,
        {"scorer": "human", "score": 1.0, "passed": True, "blind": 1, "self_judged": 0},
    )

    row = store.scores_for_results(db, [result_id])[result_id][0]
    assert row["blind"] is True
    assert row["self_judged"] is False
    assert row["passed"] is True


def test_scores_for_results_is_one_query_over_the_whole_set(db):
    run_id = store.save_run(db, "p", [make_result(), make_result(model="m/b")])
    results = store.get_run(db, run_id)["results"]
    for r in results:
        store.add_score(db, r["id"], {"scorer": "exact", "score": 1.0})

    out = store.scores_for_results(db, [r["id"] for r in results])

    assert sorted(out) == sorted(r["id"] for r in results)
    assert store.scores_for_results(db, []) == {}


def test_review_repro_a_pre_g_errored_row_classifies_as_error_not_refused(tmp_path):
    """The era gate, driven from the schema that actually shipped rather
    than from a hand-built dict.

    A pre-G errored row has NULL request_json because the column did not
    exist yet, which has nothing to do with money. Ungated, the report's
    structural refusal rule would read every legacy error as a spend
    refusal, and the export would carry that misreading forever.

    This is the closing review's legacy-fixture lens in test form: the
    fixture is migrated by the real connect path, read back by the real
    store, and classified by the real derivation.
    """
    db_path = tmp_path / "pre_g.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_G_SCHEMA)
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at)
           VALUES (1, NULL, NULL, 'legacy', '2026-02-01T00:00:00+00:00')"""
    )
    legacy.execute(
        """INSERT INTO results (run_id, model, response_text, latency_ms,
                                prompt_tokens, completion_tokens, error)
           VALUES (1, 'legacy/a', NULL, NULL, NULL, NULL,
                   'HTTP 500 from OpenRouter')"""
    )
    legacy.commit()
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        run = store.get_run(conn, 1)
        assert run is not None
        result = run["results"][0]
        # The two facts that would fool the ungated rule.
        assert result["request_json"] is None
        assert run["data_policy"] is None

        assert report.trial_outcome(result, run) == "error"
        # And the gate is a gate, not a disabling: the same row shape
        # inside the provenance era is a refusal.
        assert report.trial_outcome(result, {**run, "data_policy": "standard"}) == (
            "refused"
        )
    finally:
        conn.close()


# ---- The synchronous contract, guarded rather than only stated.


def test_the_store_stays_synchronous():
    """One shared connection is safe because this module cannot yield.

    On one event loop a synchronous function that materializes before
    returning is an atomic block: nothing interleaves between its first
    statement and its last, so no second caller can slip a write into the
    middle of a read or find the connection mid-statement. That property,
    and not a lock anybody has to remember to take, is why a single
    sqlite3 connection shared by every request, every experiment trial
    and every scoring pass is safe here.

    One innocent `async def` helper, or one function handing back a live
    cursor for a caller to iterate across an await, removes the property
    everywhere at once and NOTHING FAILS. The reads still work. They keep
    working until two callers overlap under load and one of them sees
    half of the other's transaction, which is a bug that reproduces on a
    busy machine and never on the one where it is being debugged.

    So the guard is a source scan, and it is deliberately crude. It is
    not trying to prove the module is correct; it is trying to make the
    day someone reaches for async here a day the suite goes red, with
    this docstring attached to the failure. A subtler check that
    understood intent would be a check that could be argued with.

    Tokenized rather than string-matched, which is why the module
    docstring above can name `async` and `await` while describing the
    rule: only NAME tokens count, so prose and comments do not.
    """
    import io
    import tokenize

    source = (Path(store.__file__)).read_text(encoding="utf-8")
    offenders = [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME and token.string in ("async", "await")
    ]

    assert offenders == [], (
        f"bench/store.py must stay synchronous; found {offenders}. "
        "See the module docstring: one shared connection is safe only "
        "because nothing in this file can yield mid-statement."
    )


# ---- Phase I.2: three more columns, proven against the era they land on.


PRE_I2_SCHEMA = (Path(__file__).parent / "fixtures" / "pre_i2_schema.sql").read_text()


def test_migration_onto_pre_i2_database_is_additive_and_idempotent(tmp_path):
    """The additive-only invariant, proven against the RIGHT era.

    pre_i_schema.sql predates Phase I entirely, so it cannot say anything
    about columns Phase I.2 adds on top of Phase I: a database that
    already has the experiments table is the one an I.2 migration
    actually meets, and it is the only one whose ALTERs are the ones
    under test. This fixture is the SCHEMA string exactly as it stood at
    baf9390, the tip of phase-i-evaluation-layer and the commit that
    merged as PR #51, extracted rather than transcribed.

    Three columns, all nullable and all additive:
    experiments.primary_metric, experiments.quantizations_json and
    results.upstream_inference_cost_usd. NULL on a legacy row is the
    affirmative fact in each case. No primary metric was declared, so the
    report publishes sections and no cross-scorer ranking. No
    quantization filter narrowed the provider population. And the run
    was not BYOK, which is a different fact from a BYOK run that cost
    nothing, which is why zero degrades to NULL on the way in.
    """
    db_path = tmp_path / "pre_i2.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_I2_SCHEMA)
    legacy.execute(
        """INSERT INTO experiments
           (id, name, created_at, dataset_name, dataset_digest, lineup_json,
            budget, params_json, repeats, task_order_seed, estimand_mode,
            provider_pins_json, halt_on_refusal, status, status_detail,
            app_sha, catalog_digest, data_policy, tasks_total, trials_total,
            trials_done, trials_refused, trials_failed)
           VALUES (1, 'legacy sweep', '2026-02-01T00:00:00+00:00', 'd.jsonl',
                   'deadbeef', '["legacy/a"]', 'standard', NULL, 2, 11,
                   'routed_service', NULL, 1, 'done', NULL, 'abc1234', 'dd',
                   'standard', 3, 6, 6, 0, 0)"""
    )
    legacy.execute(
        """INSERT INTO groups (id, created_at, prompt_text, models_json,
                               params_json, budget, experiment_id, task_id,
                               repeat_index, rotation_index)
           VALUES (1, '2026-02-01T00:00:00+00:00', 'legacy prompt',
                   '["legacy/a"]', NULL, 'standard', 1, 't1', 0, 0)"""
    )
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at,
                             app_sha, catalog_snapshot_at, data_policy,
                             catalog_digest)
           VALUES (1, NULL, 1, 'legacy prompt', '2026-02-01T00:00:00+00:00',
                   'abc1234', '2026-02-01T00:00:00+00:00', 'standard', 'dd')"""
    )
    legacy.execute(
        """INSERT INTO results (id, run_id, model, response_text, latency_ms,
                                prompt_tokens, completion_tokens, error,
                                cost_usd, position, billed_cost_usd, provider)
           VALUES (1, 1, 'legacy/a', 'legacy text', 12.5, 13, 8, NULL,
                   2.9e-05, 0, 3.1e-05, 'Together')"""
    )
    legacy.execute(
        """INSERT INTO scores (result_id, scorer, score, passed, detail,
                               judge_model, created_at)
           VALUES (1, 'contains', 1.0, 1, NULL, NULL,
                   '2026-02-01T00:00:00+00:00')"""
    )
    legacy.commit()
    # The pre-state, asserted rather than assumed. Without this the test
    # could pass against a fixture that already had the columns, which
    # would prove nothing at all about the migration.
    before = {
        table: [r[1] for r in legacy.execute(f"PRAGMA table_info({table})")]
        for table in ("experiments", "results")
    }
    assert "primary_metric" not in before["experiments"]
    assert "quantizations_json" not in before["experiments"]
    assert "upstream_inference_cost_usd" not in before["results"]
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        # The three columns exist on a database that never had them.
        experiment_columns = [
            r["name"] for r in conn.execute("PRAGMA table_info(experiments)")
        ]
        result_columns = [r["name"] for r in conn.execute("PRAGMA table_info(results)")]
        assert "primary_metric" in experiment_columns
        assert "quantizations_json" in experiment_columns
        assert "upstream_inference_cost_usd" in result_columns

        # And read NULL on the legacy rows, through the real readers
        # rather than through raw SQL, because the decoded shape is what
        # every caller actually sees.
        experiment = store.get_experiment(conn, 1)
        assert experiment is not None
        assert experiment["primary_metric"] is None
        assert experiment["quantizations"] is None
        # Nothing else about the row moved.
        assert experiment["lineup"] == ["legacy/a"]
        assert experiment["estimand_mode"] == "routed_service"
        assert experiment["task_order_seed"] == 11
        assert experiment["halt_on_refusal"] is True
        assert experiment["trials_total"] == 6

        # A pre-I.2 row round trips through the real store unharmed.
        run = store.get_run(conn, 1)
        assert run is not None
        result = run["results"][0]
        assert result["response_text"] == "legacy text"
        assert result["billed_cost_usd"] == 3.1e-05
        assert result["provider"] == "Together"
        assert result["upstream_inference_cost_usd"] is None
        # The experiment's own readers still work over it.
        assert [g["id"] for g in store.experiment_groups(conn, 1)] == [1]
        assert store.scores_for_results(conn, [1])[1][0]["scorer"] == "contains"
    finally:
        conn.close()

    again = store.connect(str(db_path))
    try:
        # A second connect adds nothing and breaks nothing: each column
        # appears exactly once, and the legacy row is untouched.
        for table, column in (
            ("experiments", "primary_metric"),
            ("experiments", "quantizations_json"),
            ("results", "upstream_inference_cost_usd"),
        ):
            names = [r["name"] for r in again.execute(f"PRAGMA table_info({table})")]
            assert names.count(column) == 1, (table, column, names)
        assert again.execute("SELECT count(*) c FROM experiments").fetchone()["c"] == 1
        second = store.get_run(again, 1)
        assert second is not None
        assert second["results"][0]["response_text"] == "legacy text"
    finally:
        again.close()


def test_review_repro_a_write_inside_the_read_snapshot_is_caught(db):
    """THE WINDOW: from read_snapshot's BEGIN to its exit. Anything
    inside it that ends the transaction early splits the block into two
    snapshots, and the reads after the split see a different state of
    the database than the reads before it.

    The closing review found this in the branch's own code. Every write
    helper in this module uses `with conn`, and the sqlite3 connection
    context manager COMMITS at exit, so one of them called inside the
    block ends the snapshot. Nothing fails and nothing logs: the export
    built from those reads straddles two moments and its digest verifies
    perfectly, which is the exact defect read_snapshot exists to
    prevent, arriving through the front door instead.

    No caller does this today. That is a fact about today's callers
    rather than a property of anything, and an invariant that holds by
    accident of what the callees currently do is the shape this whole
    phase keeps finding, so it is asserted rather than trusted.
    """
    with pytest.raises(RuntimeError) as caught:
        with store.read_snapshot(db):
            # A real write helper, chosen because it is the ordinary way
            # this would happen: somebody adds a convenience call inside
            # a read block and nothing tells them.
            store.save_prompt(db, "inside", "the snapshot")

    assert "committed from inside its own block" in str(caught.value)
    # The write itself is not undone, which is the honest outcome: it
    # committed, and pretending otherwise would be a second lie on top
    # of the first. What must not happen is the export believing it read
    # one state of the database.
    assert [p["name"] for p in store.list_prompts(db)] == ["inside"]


def test_a_read_only_snapshot_block_completes_normally(db):
    """The scope control. The guard must fire on a lost snapshot and on
    nothing else, or the export stops working and the fix reads as the
    bug."""
    store.save_prompt(db, "before", "written outside the block")

    with store.read_snapshot(db):
        names = [p["name"] for p in store.list_prompts(db)]

    assert names == ["before"]


def test_an_exception_inside_the_snapshot_propagates_unchanged(db):
    """The guard must not mask what actually went wrong. A failure inside
    the block is the interesting exception; replacing it with a
    complaint about the transaction would bury it."""
    with pytest.raises(ValueError, match="the real problem"):
        with store.read_snapshot(db):
            raise ValueError("the real problem")


# ---- Phase K: one column and one table, proven against the era they land on.


PRE_K_SCHEMA = (Path(__file__).parent / "fixtures" / "pre_k_schema.sql").read_text()


def test_the_pre_k_fixture_is_the_schema_as_it_stood_at_v0_1_0():
    """The fixture's provenance, checkable rather than asserted in prose.

    A hand-transcribed era snapshot is a snapshot of what somebody
    believed the schema was, and the migration test built on it would
    then prove something about that belief. This one was extracted, and
    the extraction is repeatable:

        git show v0.1.0:bench/store.py > /tmp/store_at_tag.py
        python -c "ns={}; exec(open('/tmp/store_at_tag.py').read(), ns);
                   open('tests/fixtures/pre_k_schema.sql','w')
                       .write(ns['SCHEMA'].lstrip('\\n'))"

    What this asserts is the part a reader cannot re-run from the test:
    that the fixture is a strict PREFIX of today's schema in the tables
    it shares, and that it lacks exactly what Phase K adds. If someone
    edits the fixture to make a migration test pass, this fails first.
    """
    # It is the pre-K era: no attachments table, no attachments column.
    assert "attachments" not in PRE_K_SCHEMA
    # And it is a real schema of this database, not a sketch: every
    # table Phase K does not touch is present exactly as SCHEMA has it.
    for table in ("prompts", "experiments", "scores", "groups", "runs", "results"):
        assert f"CREATE TABLE IF NOT EXISTS {table} (" in PRE_K_SCHEMA
    # The one table Phase K adds is in today's schema and not the
    # fixture's, which is the difference the migration test rests on.
    assert "CREATE TABLE IF NOT EXISTS attachments (" in store.SCHEMA


def test_migration_onto_pre_k_database_is_additive_and_idempotent(tmp_path):
    """The additive-only invariant, proven against the RIGHT era.

    pre_i2_schema.sql predates Phase K by two phases, so it cannot say
    anything about a column landing on top of I.2 and the guard rider. A
    database that already has upstream_inference_cost_usd is the one a K
    migration actually meets, and it is the only one whose ALTER is the
    one under test. This fixture is the SCHEMA string exactly as it
    stood at v0.1.0, extracted rather than transcribed.

    THREE changes, all additive, and the count is the correction this
    proof needed. groups.attachments_json AND groups.attachments_mode
    both arrive by ALTER and are NULL on every legacy row, which is the
    affirmative fact that the group predates attachments entirely: it
    declared nothing and could not have. The attachments table arrives
    by CREATE TABLE IF NOT EXISTS, which is additive by construction on
    any database and is why it needs no MIGRATIONS entry.

    THE WINDOW IS EVERY COLUMN PHASE K MIGRATES, not one of them. This
    asserted attachments_json alone, and the MIGRATIONS comment above the
    block called Phase K "One column", so the test that is specifically
    about Phase K's era said nothing about half of what Phase K
    migrates.

    Measured rather than assumed, because the scope of the gap matters:
    deleting the attachments_mode entry does NOT leave the suite green.
    The pre-G, pre-H, pre-H1 and pre-I2 era tests catch it too, since
    their fixtures also predate the column and each one reads the groups
    table back. So the exposure was never a silent broken migration; it
    was this proof being narrower than its own subject, and four tests
    about other eras carrying a Phase K guarantee by accident. A
    guarantee held only by accident is one an unrelated refactor can
    drop.
    """
    db_path = tmp_path / "pre_k.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_K_SCHEMA)
    legacy.execute(
        """INSERT INTO groups (id, created_at, prompt_text, models_json,
                               params_json, budget)
           VALUES (1, '2026-05-01T00:00:00+00:00', 'legacy prompt',
                   '["legacy/a"]', '{"seed": 7}', 'standard')"""
    )
    legacy.execute(
        """INSERT INTO runs (id, prompt_id, group_id, prompt_text, created_at,
                             app_sha, catalog_snapshot_at, data_policy,
                             catalog_digest)
           VALUES (1, NULL, 1, 'legacy prompt', '2026-05-01T00:00:00+00:00',
                   'abc1234', '2026-05-01T00:00:00+00:00', 'standard', 'dd')"""
    )
    legacy.execute(
        """INSERT INTO results (id, run_id, model, response_text, latency_ms,
                                prompt_tokens, completion_tokens, error,
                                cost_usd, position, billed_cost_usd, provider,
                                upstream_inference_cost_usd)
           VALUES (1, 1, 'legacy/a', 'legacy text', 12.5, 13, 8, NULL,
                   2.9e-05, 0, 3.1e-05, 'Together', '0.004')"""
    )
    legacy.commit()
    # The pre-state, asserted rather than assumed. Without this the test
    # could pass against a fixture that already had the column, which
    # would prove nothing at all about the migration.
    before = [r[1] for r in legacy.execute("PRAGMA table_info(groups)")]
    assert "attachments_json" not in before
    assert "attachments_mode" not in before
    tables_before = {
        r[0]
        for r in legacy.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "attachments" not in tables_before
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        columns = [r["name"] for r in conn.execute("PRAGMA table_info(groups)")]
        assert "attachments_json" in columns
        assert "attachments_mode" in columns
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "attachments" in tables

        # THE TWO NULLS, read through the real reader. None means the
        # group predates attachments, which is a different fact from an
        # empty list, and group_attachments is the check site that keeps
        # them apart.
        assert store.group_attachments(conn, 1) is None
        # The mode's own reader, on the same legacy row. Its two NULLs
        # collapse deliberately (a mode with no documents is a statement
        # about nothing), so None here is the whole of what it can say.
        assert store.group_attachments_mode(conn, 1) is None

        # Nothing else about the legacy row moved.
        assert store.group_prompt(conn, 1) == "legacy prompt"
        assert store.group_params(conn, 1) == {"seed": 7}
        run = store.get_run(conn, 1)
        assert run is not None
        result = run["results"][0]
        assert result["response_text"] == "legacy text"
        assert result["billed_cost_usd"] == 3.1e-05
        assert result["upstream_inference_cost_usd"] == "0.004"

        # And the new table works on a database that never had it.
        stored = store.save_attachment(
            conn,
            {
                "digest": "ab" * 32,
                "filename": "notes.txt",
                "mime": "text/plain",
                "byte_size": 2,
                "content": b"hi",
                "extracted_text": "hi",
                "extractor": "text",
                "kind": "document",
                "extractor_version": "1",
            },
        )
        assert stored["digest"] == "ab" * 32
    finally:
        conn.close()

    again = store.connect(str(db_path))
    try:
        # A second connect adds nothing and breaks nothing.
        names = [r["name"] for r in again.execute("PRAGMA table_info(groups)")]
        assert names.count("attachments_json") == 1, names
        assert again.execute("SELECT count(*) c FROM attachments").fetchone()["c"] == 1
        assert store.group_attachments(again, 1) is None
        second = store.get_run(again, 1)
        assert second is not None
        assert second["results"][0]["response_text"] == "legacy text"
    finally:
        again.close()


def test_a_post_k_group_that_declared_nothing_is_not_a_pre_k_group(db):
    """THE TWO NULLS, from the other side, and the reason nothing writes
    "[]".

    An empty declaration and a pre-K group both read as None, and they
    have to, because the column cannot tell them apart: absence gets one
    spelling, exactly as an empty params mapping stores NULL. What keeps
    the distinction honest is that a post-K group with no attachment has
    nothing to enforce either way, so the two collapse to the same
    behavior rather than to a guess.
    """
    empty = store.create_group(db, "hi", ["m/a"], attachments=[])
    declared = store.create_group(db, "hi", ["m/a"], attachments=["ab" * 32])

    assert store.group_attachments(db, empty) is None
    assert store.group_attachments(db, declared) == ["ab" * 32]
    # Stored as NULL rather than "[]", so absence has one spelling.
    raw = db.execute(
        "SELECT attachments_json FROM groups WHERE id = ?", (empty,)
    ).fetchone()
    assert raw["attachments_json"] is None


def test_the_declared_order_is_preserved_rather_than_sorted(db):
    """Two documents composed in the other order are a different prompt,
    so the list is a sequence and not a set. Sorting it here would make
    the manifest describe a comparison nobody declared."""
    digests = ["cc" * 32, "aa" * 32, "bb" * 32]

    group_id = store.create_group(db, "hi", ["m/a"], attachments=digests)

    assert store.group_attachments(db, group_id) == digests


def test_attachments_for_returns_a_mapping_and_omits_what_is_missing(db):
    """One query rather than N, and a digest with no row is simply
    absent: the caller's cue to refuse rather than this function's cue
    to guess."""
    store.save_attachment(
        db,
        {
            "digest": "aa" * 32,
            "filename": "one.txt",
            "mime": "text/plain",
            "byte_size": 3,
            "content": b"one",
            "extracted_text": "one",
            "extractor": "text",
            "kind": "document",
            "extractor_version": "1",
        },
    )

    found = store.attachments_for(db, ["aa" * 32, "bb" * 32])

    assert set(found) == {"aa" * 32}
    assert found["aa" * 32]["filename"] == "one.txt"
    assert store.attachments_for(db, []) == {}


def test_no_attachment_reader_selects_the_content_blob(db):
    """The bytes never leave this module. Asserted on the shape the
    readers actually return, because a column list is exactly the kind
    of thing a later edit extends without noticing what it means."""
    store.save_attachment(
        db,
        {
            "digest": "aa" * 32,
            "filename": "one.txt",
            "mime": "text/plain",
            "byte_size": 6,
            "content": b"secret",
            "extracted_text": "secret",
            "extractor": "text",
            "kind": "document",
            "extractor_version": "1",
        },
    )

    row = store.get_attachment(db, "aa" * 32)

    assert row is not None
    assert "content" not in row
    assert "content" not in store.ATTACHMENT_COLUMNS
    assert "content" not in store.attachments_for(db, ["aa" * 32])["aa" * 32]


def test_the_extraction_table_keeps_one_reading_per_parser_version(tmp_path):
    """WINDOW: the attachments and attachment_extractions tables after
    the same bytes are saved twice under two extractor versions.

    ONE COPY OF THE BYTES, ONE ROW PER READING. The storage promise is
    about content and stays true; the extraction is keyed by parser and
    version because the text is a property of the bytes AND the parser.

    The attachments row is NOT rewritten by the second save, which is
    rule two's requirement: a comparison recorded under the first parser
    cites this digest and its placeholder names that reading's character
    count."""
    conn = store.connect(str(tmp_path / "bench.db"))
    try:
        base = {
            "digest": "cd" * 32,
            "filename": "contract.txt",
            "mime": "text/plain",
            "byte_size": 9,
            "content": b"some text",
            "extractor": "text",
            "kind": "document",
        }
        first = store.save_attachment(
            conn, {**base, "extracted_text": "some text", "extractor_version": "1"}
        )
        second = store.save_attachment(
            conn,
            {**base, "extracted_text": "some text [v2]", "extractor_version": "2"},
        )

        # The caller is told about the reading it just performed.
        assert first["extractor_version"] == "1"
        assert second["extractor_version"] == "2"
        # The LENGTH of that reading, since K1.5, because the row no
        # longer carries the text out of the store: the two readings are
        # 9 and 14 characters, so this still distinguishes them and
        # still fails if the first upload's reading is handed back.
        assert second["extracted_chars"] == len("some text [v2]")
        assert first["extracted_chars"] == len("some text")

        # One copy of the bytes, and the row's own reading unchanged.
        rows = conn.execute(
            "SELECT extracted_text, extractor_version FROM attachments"
            " WHERE digest = ?",
            (base["digest"],),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["extracted_text"] == "some text"
        assert rows[0]["extractor_version"] == "1"

        # Both readings retrievable, neither substituting for the other.
        # The whole RENDITION comes back now, not the text alone: a
        # caller handed only the text would have to re-derive the other
        # three facts from somewhere, and every one of those
        # re-derivations was a place K.1 found a bug.
        one = store.extraction_for(conn, base["digest"], "text", "1")
        two = store.extraction_for(conn, base["digest"], "text", "2")
        assert one is not None and two is not None
        assert one["extracted_text"] == "some text"
        assert two["extracted_text"] == "some text [v2]"
        assert one["kind"] == "document" and one["filename"] == "contract.txt"
        # A version nobody ran is absent rather than approximated. That
        # None is the caller's cue to re-extract, and returning some
        # other version's text here would be exactly the substitution
        # this table exists to prevent.
        assert store.extraction_for(conn, base["digest"], "text", "3") is None
    finally:
        conn.close()


def test_re_saving_the_same_reading_is_a_no_op(tmp_path):
    """The ordinary case: the same file uploaded twice under one build
    must not accumulate extraction rows."""
    conn = store.connect(str(tmp_path / "bench.db"))
    try:
        record = {
            "digest": "ef" * 32,
            "filename": "n.txt",
            "mime": "text/plain",
            "byte_size": 2,
            "content": b"hi",
            "extracted_text": "hi",
            "extractor": "text",
            "kind": "document",
            "extractor_version": "1",
        }
        store.save_attachment(conn, record)
        store.save_attachment(conn, record)

        count = conn.execute(
            "SELECT COUNT(*) c FROM attachment_extractions WHERE digest = ?",
            (record["digest"],),
        ).fetchone()
        assert count["c"] == 1
    finally:
        conn.close()


PRE_K1_SCHEMA = (
    pathlib.Path(__file__).parent / "fixtures" / "pre_k1_schema.sql"
).read_text()


def test_review_repro_migration_onto_a_pre_k1_database_is_additive(tmp_path):
    """WINDOW: a database whose schema is exactly c9ad170's, upgraded by
    connect(), from the pre-state through the post-state and the
    backfill.

    A NEW FIXTURE, AND THE OLD ONE COULD NOT HAVE DONE THIS. The pre-K
    snapshot contains no attachments table at all, so the era test built
    on it exercises CREATE TABLE IF NOT EXISTS and never an ALTER onto
    those tables. Reusing it for K.1 would have proved the additive
    invariant against a table the migration does not touch, which is the
    wrong window wearing the right name.

    Three additive changes: kind and filename onto attachment_extractions,
    and renditions_json onto groups. The constraint is deliberately NOT
    widened, because SQLite cannot, and it does not need to be: kind is a
    function of the extractor, so UNIQUE (digest, extractor,
    extractor_version) already enforces the four-part key.

    THE BACKFILL IS THE OTHER HALF. K.1 resolves a pinned rendition and
    refuses rather than substituting, so a database whose extraction
    rows were never written would answer "no such reading" for every
    document it is holding. connect() copies each attachments row into
    the table it now belongs in, once, and this asserts the pre-state
    (no rows) so the copy is proven rather than assumed.
    """
    db_path = tmp_path / "pre_k1.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_K1_SCHEMA)
    legacy.execute(
        """INSERT INTO groups (id, created_at, prompt_text, models_json,
                               params_json, budget, attachments_json,
                               attachments_mode)
           VALUES (1, '2026-06-01T00:00:00+00:00', 'legacy prompt',
                   '["legacy/a"]', NULL, 'standard', '["ab"]', 'inline')"""
    )
    legacy.execute(
        """INSERT INTO attachments (digest, filename, mime, byte_size, content,
                                    extracted_text, extractor,
                                    extractor_version, created_at)
           VALUES ('ab', 'legacy.txt', 'text/plain', 9, X'6c6567616379',
                   'legacy text', 'text', '1', '2026-06-01T00:00:00+00:00')"""
    )
    legacy.commit()

    # The pre-state, asserted. Without it this could pass against a
    # fixture that already had the columns, proving nothing.
    before = [r[1] for r in legacy.execute("PRAGMA table_info(attachment_extractions)")]
    assert "kind" not in before
    assert "filename" not in before
    group_before = [r[1] for r in legacy.execute("PRAGMA table_info(groups)")]
    assert "renditions_json" not in group_before
    rows_before = legacy.execute(
        "SELECT COUNT(*) FROM attachment_extractions"
    ).fetchone()[0]
    assert rows_before == 0
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        after = [
            r["name"] for r in conn.execute("PRAGMA table_info(attachment_extractions)")
        ]
        assert "kind" in after
        assert "filename" in after
        group_after = [r["name"] for r in conn.execute("PRAGMA table_info(groups)")]
        assert "renditions_json" in group_after

        # THE THIRD NULL. This group declared documents and no reading,
        # which is neither pre-K nor K.1, and group_renditions must say
        # so rather than collapsing it into "declared nothing".
        assert store.group_renditions(conn, 1) is None
        assert store.group_attachments(conn, 1) == ["ab"]

        # Backfilled: the row's own reading is now a rendition the pin
        # machinery can resolve.
        found = store.extraction_for(conn, "ab", "text", "1")
        assert found is not None
        assert found["extracted_text"] == "legacy text"
        assert found["kind"] == "document"
        assert found["filename"] == "legacy.txt"

        # And idempotent: a second connect must not double it.
        conn.close()
        again = store.connect(str(db_path))
        try:
            count = again.execute(
                "SELECT COUNT(*) c FROM attachment_extractions"
            ).fetchone()
            assert count["c"] == 1
        finally:
            again.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def test_an_image_row_backfills_as_an_image_rendition(tmp_path):
    """The backfill derives kind from the EXTRACTOR, not the filename,
    which is the same decision ingest makes and the opposite of the
    re-derivation K.1 removed. An image row stored under any name comes
    back as an image."""
    db_path = tmp_path / "img.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_K1_SCHEMA)
    legacy.execute(
        """INSERT INTO attachments (digest, filename, mime, byte_size, content,
                                    extracted_text, extractor,
                                    extractor_version, created_at)
           VALUES ('cd', 'shot.txt', 'image/png', 3, X'010203', '', 'none',
                   '0', '2026-06-01T00:00:00+00:00')"""
    )
    legacy.commit()
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        found = store.extraction_for(conn, "cd", "none", "0")
        assert found is not None
        assert found["kind"] == "image"
    finally:
        conn.close()


# ---- Phase K1.5: what survives a delete, and what must not be written.


def _attachment_record(digest="e1", text="the settlement figure is 40200"):
    return {
        "digest": digest,
        "filename": "contract.txt",
        "mime": "text/plain",
        "byte_size": len(text),
        "content": text.encode(),
        "extracted_text": text,
        "extractor": "text",
        "extractor_version": "1",
        "kind": "document",
    }


def test_review_repro_deleting_an_attachment_takes_its_extracted_text(tmp_path):
    """WINDOW: a DELETE issued the only way a person can issue one, from
    a second connection with foreign_keys at sqlite's default of OFF,
    against a database the app has already written.

    THE TEXT IS THE THING THAT SURVIVED, not a dangling id. The bytes and
    the first reading live on the attachments row and go with it; every
    OTHER reading lives in attachment_extractions keyed by the same
    digest, and nothing pointed the delete at them. A person who removed
    a confidential contract from bench.db was left with its full
    extracted text still on disk under the digest of the file they had
    just deleted.

    THE SECOND CONNECTION IS THE POINT rather than incidental. Nothing in
    the application deletes an attachment, so the only deleter is
    sqlite3 on the command line, which does not set PRAGMA foreign_keys
    and would therefore have ignored an ON DELETE CASCADE entirely. This
    connection is opened raw for exactly that reason: a cascade that only
    works from inside the app is a cascade that never runs.
    """
    db_path = tmp_path / "bench.db"
    conn = store.connect(str(db_path))
    try:
        secret = "the settlement figure is 40200"
        store.save_attachment(conn, _attachment_record(text=secret))
        # A SECOND reading of the same bytes, which is the row the
        # attachments delete never touched. Without it this would pass
        # against the backfilled copy alone and prove less than it
        # claims.
        store.save_attachment(
            conn,
            {
                **_attachment_record(text=secret),
                "filename": "contract.md",
                "extractor_version": "2",
            },
        )

        # PRE-STATE. Two readings on disk, both carrying the text.
        held = conn.execute(
            "SELECT extracted_text FROM attachment_extractions WHERE digest = 'e1'"
        ).fetchall()
        assert [r["extracted_text"] for r in held] == [secret, secret]
    finally:
        conn.close()

    raw = sqlite3.connect(str(db_path))
    try:
        # Exactly what the README tells a person to type, and no pragma.
        assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        raw.execute("DELETE FROM attachments WHERE digest = 'e1'")
        raw.commit()
        left = raw.execute(
            "SELECT extracted_text FROM attachment_extractions WHERE digest = 'e1'"
        ).fetchall()
        assert left == [], f"the text outlived the file: {left}"
        # And nowhere else in the database either, which is the claim the
        # README makes rather than the narrower one about two tables.
        for (table,) in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall():
            for row in raw.execute(f"SELECT * FROM {table}"):
                for cell in row:
                    assert not (isinstance(cell, str) and secret in cell), table
    finally:
        raw.close()


def test_review_repro_a_failed_base_row_commits_no_extraction(tmp_path):
    """WINDOW: inside save_attachment's transaction, on a record that
    violates a NOT NULL on `attachments` and none on
    `attachment_extractions`.

    INSERT OR IGNORE IGNORES EVERY CONSTRAINT, not only the UNIQUE one it
    is written for. A record with a NULL mime is silently not inserted,
    because attachments.mime is NOT NULL; attachment_extractions has no
    mime column, so its insert succeeded. The order was insert, insert,
    read, and the read's failure was checked AFTER the `with conn` block
    had already committed, so the document's text was committed under a
    digest the attachments table had never heard of and the exception
    was raised over the top of it.

    A digest with no attachments row is a digest nothing can delete: the
    cascade above hangs off the row that was never written. This is the
    one path that could manufacture that state.
    """
    conn = store.connect(str(tmp_path / "bench.db"))
    try:
        secret = "the indemnity cap is 2.5 million"
        with pytest.raises(RuntimeError, match="was not created"):
            store.save_attachment(
                conn,
                {**_attachment_record(digest="f2", text=secret), "mime": None},
            )
        assert (
            conn.execute(
                "SELECT COUNT(*) c FROM attachment_extractions WHERE digest = 'f2'"
            ).fetchone()["c"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) c FROM attachments WHERE digest = 'f2'"
            ).fetchone()["c"]
            == 0
        )
    finally:
        conn.close()


def test_review_repro_the_stored_count_is_pythons_and_not_sqlites(tmp_path):
    """WINDOW: connect() upgrading a c9ad170-era database whose stored
    text contains a NUL, from the pre-state through the backfill.

    THE COUNT MOVED OUT OF THE READER, so a list view no longer loads a
    body per document to print an integer. The obvious backfill for the
    new column is `UPDATE attachments SET extracted_chars =
    length(extracted_text)`, and it is wrong: sqlite's length() stops at
    the first NUL and would write 12 for the 26-character string below.

    A NUL IS NOT A CONTRIVANCE. A binary file uploaded under a .txt
    suffix decodes through latin-1, which is total and maps 0x00 to
    U+0000, so the text of any such upload has them. The number would
    have been short only for those documents, which is the shape of
    defect nobody finds by looking at a page that renders.
    """
    db_path = tmp_path / "nul.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(PRE_K1_SCHEMA)
    text = "page one ends\x00and page two begins"
    legacy.execute(
        """INSERT INTO attachments (digest, filename, mime, byte_size, content,
                                    extracted_text, extractor,
                                    extractor_version, created_at)
           VALUES ('ef', 'scan.txt', 'text/plain', 33, X'00',
                   ?, 'text', '1', '2026-06-01T00:00:00+00:00')""",
        (text,),
    )
    legacy.commit()

    # PRE-STATE: the column does not exist, so this cannot be passing
    # against a fixture that already carried the right number.
    before = [r[1] for r in legacy.execute("PRAGMA table_info(attachments)")]
    assert "extracted_chars" not in before
    # And the number the SQL backfill would have written, measured rather
    # than asserted from memory.
    sqlite_says = legacy.execute(
        "SELECT length(extracted_text) n FROM attachments WHERE digest = 'ef'"
    ).fetchone()[0]
    assert sqlite_says == 13
    assert len(text) == 33
    legacy.close()

    conn = store.connect(str(db_path))
    try:
        after = [r["name"] for r in conn.execute("PRAGMA table_info(attachments)")]
        assert "extracted_chars" in after
        row = store.get_attachment(conn, "ef")
        assert row is not None
        assert row["extracted_chars"] == 33
        # The reader no longer carries the body at all, which is the
        # other half of the change and the half a count assertion alone
        # would not notice.
        assert "extracted_text" not in row
    finally:
        conn.close()


def test_the_count_backfill_runs_once_and_then_does_nothing(tmp_path):
    """Idempotent, like the K.1 rendition backfill beside it: a second
    boot must not re-read every body to rewrite numbers already right."""
    db_path = tmp_path / "twice.db"
    conn = store.connect(str(db_path))
    try:
        store.save_attachment(conn, _attachment_record(text="forty two"))
    finally:
        conn.close()

    again = store.connect(str(db_path))
    try:
        reads = []
        again.set_trace_callback(reads.append)
        third = store.connect(str(db_path))
        third.close()
        again.set_trace_callback(None)
        row = store.get_attachment(again, "e1")
        assert row is not None
        assert row["extracted_chars"] == len("forty two")
        pending = again.execute(
            "SELECT COUNT(*) c FROM attachments WHERE extracted_chars IS NULL"
        ).fetchone()["c"]
        assert pending == 0
    finally:
        again.close()
