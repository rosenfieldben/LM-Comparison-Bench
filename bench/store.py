"""SQLite data layer. Pure functions over an injected connection.

Same reasoning as the injected httpx client in models.py: the caller
owns the connection's lifecycle, tests hand in :memory:, and swapping
sqlite for something else later touches only this module.
"""

import sqlite3
from datetime import datetime, timezone

# runs.prompt_text is denormalized on purpose: a run must stay readable
# after its saved prompt is edited or deleted. prompt_id is a courtesy
# link back to the library, not the source of truth, so deleting a
# prompt nulls it instead of cascading into history.
SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NULL REFERENCES prompts(id) ON DELETE SET NULL,
    prompt_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
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


def connect(path: str) -> sqlite3.Connection:
    """Open a connection with the schema applied and foreign keys on.

    check_same_thread is off because the app opens the connection in
    the lifespan and uses it from request handlers; both run on the
    event loop thread today, but the flag keeps a future sync endpoint
    or executor hop from crashing on an sqlite thread check.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Off by default in sqlite; without it ON DELETE SET NULL is inert.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_prompt(conn: sqlite3.Connection, name: str, text: str) -> dict:
    """Insert a prompt. Raises sqlite3.IntegrityError on duplicate name."""
    with conn:
        cur = conn.execute(
            "INSERT INTO prompts (name, text, created_at) VALUES (?, ?, ?)",
            (name, text, _now()),
        )
    row = conn.execute(
        "SELECT * FROM prompts WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return dict(row)


def get_prompt(conn: sqlite3.Connection, prompt_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
    ).fetchone()
    return dict(row) if row else None


def list_prompts(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM prompts ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def delete_prompt(conn: sqlite3.Connection, prompt_id: int) -> bool:
    with conn:
        cur = conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    return cur.rowcount > 0


def save_run(
    conn: sqlite3.Connection,
    prompt_text: str,
    results: list[dict],
    prompt_id: int | None = None,
) -> int:
    """Insert a run and its results atomically. Returns the run id.

    One transaction so a failing result insert cannot leave a run row
    with missing or partial results in the history.
    """
    with conn:
        cur = conn.execute(
            "INSERT INTO runs (prompt_id, prompt_text, created_at) VALUES (?, ?, ?)",
            (prompt_id, prompt_text, _now()),
        )
        run_id = cur.lastrowid
        conn.executemany(
            """INSERT INTO results
               (run_id, model, response_text, latency_ms, prompt_tokens,
                completion_tokens, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    r["model"],
                    r["response_text"],
                    r["latency_ms"],
                    r["prompt_tokens"],
                    r["completion_tokens"],
                    r["error"],
                )
                for r in results
            ],
        )
    return run_id


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    """All runs, most recent first, each with the models it touched."""
    runs = [
        dict(r)
        for r in conn.execute(
            "SELECT id, prompt_id, prompt_text, created_at FROM runs ORDER BY id DESC"
        ).fetchall()
    ]
    # Second query instead of GROUP_CONCAT: keeps model order tied to
    # insert order, which mirrors the original request order.
    models_by_run: dict[int, list[str]] = {}
    for row in conn.execute("SELECT run_id, model FROM results ORDER BY id"):
        models_by_run.setdefault(row["run_id"], []).append(row["model"])
    for run in runs:
        run["models"] = models_by_run.get(run["id"], [])
    return runs


def get_run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    run = conn.execute(
        "SELECT id, prompt_id, prompt_text, created_at FROM runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        return None
    results = conn.execute(
        """SELECT model, response_text, latency_ms, prompt_tokens,
                  completion_tokens, error
           FROM results WHERE run_id = ? ORDER BY id""",
        (run_id,),
    ).fetchall()
    out = dict(run)
    out["results"] = [dict(r) for r in results]
    return out
