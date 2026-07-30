"""SQLite data layer. Pure functions over an injected connection.

Same reasoning as the injected httpx client in models.py: the caller
owns the connection's lifecycle, tests hand in :memory:, and swapping
sqlite for something else later touches only this module.
"""

import json
import logging
import os
import sqlite3
import stat
import urllib.parse
from datetime import UTC, datetime
from typing import Any

# Shared with ingestion on purpose: what run_model refuses to emit,
# get_run refuses to serve, so both ends of the pipeline enforce the
# same field-type contract.
from bench.models import as_metric, as_money, as_text, as_token_count

logger = logging.getLogger(__name__)

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
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt_text TEXT,
    models_json TEXT,
    params_json TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NULL REFERENCES prompts(id) ON DELETE SET NULL,
    group_id INTEGER NULL REFERENCES groups(id),
    prompt_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    app_sha TEXT,
    catalog_snapshot_at TEXT,
    data_policy TEXT
);
CREATE TABLE IF NOT EXISTS results (
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
    max_tokens INTEGER,
    generation_id TEXT,
    finish_reason TEXT,
    position INTEGER,
    request_json TEXT,
    billed_cost_usd REAL,
    reasoning_tokens INTEGER,
    cached_tokens INTEGER,
    provider TEXT,
    quantization TEXT,
    native_finish_reason TEXT
);
"""

# History reads join results by run and replay groups by scanning runs
# by group; both tables grow monotonically, so without these indexes
# every replay degrades into a full-table scan as bench.db accumulates.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_group_id ON runs(group_id);
"""

# Columns added after a table first shipped. CREATE IF NOT EXISTS skips
# existing tables entirely, so pre-existing DBs need an explicit ALTER;
# this list is the whole migration story.
MIGRATIONS = [
    ("runs", "group_id", "INTEGER NULL REFERENCES groups(id)"),
    ("results", "cost_usd", "REAL"),
    ("results", "ttft_ms", "REAL"),
    ("results", "max_tokens", "INTEGER"),
    ("results", "generation_id", "TEXT"),
    ("results", "finish_reason", "TEXT"),
    # Phase G, measurement integrity. All nullable and additive, so a
    # pre-G database keeps every existing row and reads them back with
    # each new column None. cost_usd deliberately keeps its name and its
    # meaning (the local catalog estimate); billed truth gets its own
    # column beside it rather than overwriting the estimate's history.
    # The group row is created before any upstream call, which makes it
    # the natural experiment record: what was asked, of which models.
    ("groups", "prompt_text", "TEXT"),
    ("groups", "models_json", "TEXT"),
    # Enough provenance to say what an old run actually was: which build
    # produced it, how old its price catalog was, and what data-handling
    # policy its payloads carried.
    ("runs", "app_sha", "TEXT"),
    ("runs", "catalog_snapshot_at", "TEXT"),
    ("runs", "data_policy", "TEXT"),
    # position reconstructs the original side-by-side layout from the row
    # itself; request_json is the reproducibility record of what was
    # actually sent. The rest are the authoritative numbers Phase G stops
    # discarding, filled by later workstreams: the columns land here so
    # the schema changes exactly once.
    ("results", "position", "INTEGER"),
    ("results", "request_json", "TEXT"),
    ("results", "billed_cost_usd", "REAL"),
    ("results", "reasoning_tokens", "INTEGER"),
    ("results", "cached_tokens", "INTEGER"),
    ("results", "provider", "TEXT"),
    ("results", "quantization", "TEXT"),
    ("results", "native_finish_reason", "TEXT"),
    # Phase H, experiment controls. The controls the user set for one
    # comparison, on the group rather than the run, because they are
    # experiment-level: one comparison, one controls set, applied to every
    # model. That is the fairness point, and putting them anywhere else
    # would let a single comparison hold two different experiments.
    #
    # NULL here is not missing information. Every pre-H group predates the
    # controls entirely, so NULL is the affirmative fact that no control
    # was set, which is why the params check needs no derive-from-member
    # fallback of the kind group_prompt carries.
    ("groups", "params_json", "TEXT"),
]


def _keep_private(path: str) -> None:
    """Owner-only permissions for the database and its sqlite siblings.

    Full prompts and model outputs live in this file, and umask is not
    a policy: under the common 022 the file was born world-readable on
    shared machines. A parent directory the bench creates is 0700, the
    file is created 0600 with no loose window, and a pre-existing
    readable file (or leftover journal sibling from that era) is
    tightened on every startup. SQLite mirrors the database file's
    permissions onto -journal and -wal files it creates later, so
    tightening the database now also covers future siblings.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        os.makedirs(parent, mode=0o700)
    existed = os.path.exists(path)
    if not existed:
        os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o600))
    for sibling in (path, path + "-journal", path + "-wal", path + "-shm"):
        if not os.path.exists(sibling):
            continue
        mode = stat.S_IMODE(os.stat(sibling).st_mode)
        if mode & 0o077:
            os.chmod(sibling, 0o600)
            if sibling == path and existed:
                logger.warning(
                    "tightened %s from %04o to 0600: prompts and outputs "
                    "must not be world-readable",
                    path,
                    mode,
                )


def _disk_path(path: str) -> str | None:
    """The filesystem path this database lives at, or None if memory-backed.

    sqlite accepts three shapes here: ":memory:", a plain filename, and a
    "file:" URI. Only the URI needs parsing, and the rule is the inverse of
    what this module assumed before: a URI is memory-backed exactly when it
    carries mode=memory, so the test seam's shared memory databases are
    memory and everything else, including file:/tmp/x.db?cache=private, is
    a real file that must get the private permissions and the WAL pragmas.

    The path is URL-decoded because sqlite decodes URI paths itself, so a
    URI naming a file with a space or a percent must be chmod'd under the
    same name sqlite will open.
    """
    if path == ":memory:":
        return None
    if not path.startswith("file:"):
        return path
    parsed = urllib.parse.urlparse(path)
    # Last occurrence wins, because that is what sqlite's own URI parser
    # does: it overwrites the mode flag per occurrence. Taking any
    # occurrence would call file:x.db?mode=memory&mode=rwc memory-backed
    # when sqlite opens it as a real file on disk, which is precisely the
    # hole this function exists to close.
    modes = urllib.parse.parse_qs(parsed.query).get("mode", [])
    if modes and modes[-1] == "memory":
        return None
    # sqlite accepts only an empty or "localhost" authority and rejects
    # anything else outright ("invalid uri authority"). Returning a path
    # for those would have _keep_private create and chmod a file for a URI
    # that can never open, so hand back None and let sqlite raise.
    if parsed.netloc not in ("", "localhost"):
        return None
    # urlparse puts a URI's path in .path, but only when it begins with a
    # slash; a relative URI (file:bench.db) lands in .path too, while an
    # authority-form URI (file://localhost/tmp/x.db) splits the host off.
    decoded = urllib.parse.unquote(parsed.path)
    # An empty path with no authority is sqlite's private temporary
    # database ("file:?cache=shared" style): no file to harden.
    return decoded or None


def connect(path: str) -> sqlite3.Connection:
    """Open a connection with the schema applied and foreign keys on.

    check_same_thread is off because the app opens the connection in
    the lifespan and uses it from request handlers; both run on the
    event loop thread today, but the flag keeps a future sync endpoint
    or executor hop from crashing on an sqlite thread check.
    """
    # A real file on disk, as opposed to a memory database, which has no
    # filesystem mode to manage and no meaningful WAL. A file: URI is NOT
    # memory-backed by default: only mode=memory makes it so. Treating
    # every URI as memory-like was a defect, since a disk-backed URI then
    # skipped the private permissions and the WAL and busy_timeout pragmas
    # (reproduced: file:/tmp/x.db?cache=private came out 0644 running
    # journal_mode=delete under umask 022).
    disk_path = _disk_path(path)
    if disk_path is not None:
        _keep_private(disk_path)
    # uri=True lets tests hand in shared in-memory databases via
    # file: URIs; sqlite treats anything not starting with "file:" as a
    # plain filename, so normal paths and ":memory:" are unaffected.
    conn = sqlite3.connect(path, check_same_thread=False, uri=True)
    conn.row_factory = sqlite3.Row
    # Off by default in sqlite; without it ON DELETE SET NULL is inert.
    conn.execute("PRAGMA foreign_keys = ON")
    if disk_path is not None:
        # A second process opening bench.db (an analysis script run while
        # the bench is live) risks "database is locked" under the default
        # rollback journal, which takes an exclusive lock for every
        # write. WAL lets that reader coexist with the bench's writer,
        # and busy_timeout waits out a transient lock instead of failing
        # the query instantly. WAL is meaningless for memory databases,
        # which is why this is gated on a real file.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    # After the migrations, not before: runs.group_id may only exist
    # once the ALTERs above have run on an old database.
    conn.executescript(INDEXES)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(UTC).isoformat()


def save_prompt(conn: sqlite3.Connection, name: str, text: str) -> dict[str, Any]:
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


def get_prompt(conn: sqlite3.Connection, prompt_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    return dict(row) if row else None


def list_prompts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM prompts ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def delete_prompt(conn: sqlite3.Connection, prompt_id: int) -> bool:
    with conn:
        cur = conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    return cur.rowcount > 0


def create_group(
    conn: sqlite3.Connection,
    prompt_text: str | None = None,
    models: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> int:
    """Create the group row, optionally recording what the comparison is.

    The group is created before any upstream call, so the prompt, the
    ordered lineup and the controls recorded here are the experiment as
    declared, not as it turned out. All three stay optional: a caller that
    does not know them yet (and every pre-G caller) still gets a plain
    group row.

    An empty params mapping stores NULL rather than "{}". Recording only
    what was set is the rule, and an empty object is not a smaller record
    of nothing, it is a second way to spell the same absence that every
    reader would then have to know about.

    sort_keys because this string is compared against a later request's
    controls to enforce one experiment per group. Sorting cannot change
    which controls were set, and without it two identical control sets
    could serialize differently and read as a conflict.
    """
    with conn:
        cur = conn.execute(
            "INSERT INTO groups (created_at, prompt_text, models_json, params_json)"
            " VALUES (?, ?, ?, ?)",
            (
                _now(),
                prompt_text,
                json.dumps(models) if models is not None else None,
                json.dumps(params, sort_keys=True) if params else None,
            ),
        )
    # lastrowid is Optional in the DBAPI types but always set after a
    # single-row INSERT; assert so the int return stays honest.
    assert cur.lastrowid is not None
    return cur.lastrowid


def group_exists(conn: sqlite3.Connection, group_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM groups WHERE id = ?", (group_id,)).fetchone()
    return row is not None


def group_prompt(conn: sqlite3.Connection, group_id: int) -> str | None:
    """The prompt a group is established with, or None if it has none yet.

    Prefers the group's own prompt_text, recorded at creation before any
    run existed. That is what dissolves the concurrent-first-member race:
    the prompt is fixed before the first member is even sent, so two
    simultaneous first members are checked against a value that already
    exists rather than against each other's absence.

    Falls back to deriving from the first member for groups created before
    the column existed, and for any caller that creates a group without
    declaring a prompt. The race survives only in that fallback.
    """
    row = conn.execute(
        "SELECT prompt_text FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    if row is not None and row["prompt_text"] is not None:
        declared = as_text(row["prompt_text"])
        if declared is not None:
            return declared
    member = conn.execute(
        "SELECT prompt_text FROM runs WHERE group_id = ? ORDER BY id LIMIT 1",
        (group_id,),
    ).fetchone()
    return member["prompt_text"] if member else None


def _decoded_params(raw: object) -> dict[str, Any]:
    """One group's stored controls as a mapping, empty when it has none.

    Repair on read, like every other column: a params_json that is absent,
    not a string, unparseable, or parses to something other than an object
    reads as no controls rather than propagating a broken value into a
    comparison check. Empty is the honest answer for all of those, because
    a record nobody can read is not evidence that a control was set.
    """
    if not isinstance(raw, str):
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _controls_from_request(raw: object) -> dict[str, Any]:
    """The controls a lone run's recorded payload proves were set.

    The fallback for ungrouped rows. Controls live on the group, so a run
    with no group has no stored controls set, but its request_json is the
    exact payload that went out and four of the six controls appear there
    only when someone chose them. Reading them back is what lets an
    ungrouped run wear the same badges a grouped one does.

    routing is deliberately NOT derived, and that is the interesting case.
    provider.sort is present in every payload the bench has ever sent,
    because throughput is its own default, so a run that carries
    sort=throughput is indistinguishable from a run whose user asked for
    it. Inferring a routing badge from that would render a default as a
    choice, which is exactly the truth defect rule two exists to prevent.
    Absent a stored controls set there is no way to tell, so nothing is
    claimed, and a lone run simply never shows a routing badge.

    A system message is safe to derive by contrast: the bench sent none
    before this control existed, so one being there means it was set.
    """
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("temperature", "top_p", "seed"):
        if payload.get(key) is not None:
            out[key] = payload[key]
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        out["effort"] = reasoning["effort"]
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            content = as_text(first.get("content"))
            if content is not None:
                out["system"] = content
    return out


def group_params(conn: sqlite3.Connection, group_id: int) -> dict[str, Any]:
    """The controls a group was created with, empty when it has none.

    No derive-from-member fallback, deliberately, and the asymmetry with
    group_prompt is the point. A NULL prompt_text meant "unknown, ask the
    first member", because groups existed before that column did and their
    prompt was still knowable from a run. A NULL params_json means "no
    control was set", which is true of every pre-H group by construction,
    so there is nothing to derive and nothing to guess.
    """
    row = conn.execute(
        "SELECT params_json FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    return _decoded_params(row["params_json"]) if row is not None else {}


def save_run(
    conn: sqlite3.Connection,
    prompt_text: str,
    results: list[dict[str, Any]],
    prompt_id: int | None = None,
    group_id: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> int:
    """Insert a run and its results atomically. Returns the run id.

    One transaction so a failing result insert cannot leave a run row
    with missing or partial results in the history.

    provenance carries the run-level facts the caller knows and the store
    does not: which build was running, how old the price catalog was, and
    which data-handling policy the payloads declared. Optional, and every
    key optional within it, so a caller that knows none of it still writes
    a valid row with those columns None.
    """
    prov = provenance or {}
    with conn:
        cur = conn.execute(
            """INSERT INTO runs
               (prompt_id, group_id, prompt_text, created_at,
                app_sha, catalog_snapshot_at, data_policy)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                prompt_id,
                group_id,
                prompt_text,
                _now(),
                prov.get("app_sha"),
                prov.get("catalog_snapshot_at"),
                prov.get("data_policy"),
            ),
        )
        # lastrowid is always set after this single-row INSERT; assert
        # so the int return type is not a lie.
        assert cur.lastrowid is not None
        run_id = cur.lastrowid
        conn.executemany(
            """INSERT INTO results
               (run_id, model, response_text, latency_ms, prompt_tokens,
                completion_tokens, error, cost_usd, ttft_ms, max_tokens,
                generation_id, finish_reason, position, request_json,
                billed_cost_usd, reasoning_tokens, cached_tokens,
                provider, quantization, native_finish_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    r["model"],
                    r["response_text"],
                    r["latency_ms"],
                    r["prompt_tokens"],
                    r["completion_tokens"],
                    r["error"],
                    # .get: results from paths that predate a column (the
                    # non-streaming path never sets ttft_ms, older tests
                    # carry no cost_usd) persist as NULL.
                    r.get("cost_usd"),
                    r.get("ttft_ms"),
                    r.get("max_tokens"),
                    r.get("generation_id"),
                    r.get("finish_reason"),
                    r.get("position"),
                    r.get("request_json"),
                    r.get("billed_cost_usd"),
                    r.get("reasoning_tokens"),
                    r.get("cached_tokens"),
                    r.get("provider"),
                    r.get("quantization"),
                    r.get("native_finish_reason"),
                )
                for r in results
            ],
        )
    return run_id


def list_runs(conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    """The newest `limit` history entries.

    Runs sharing a group collapse into one group entry, emitted at the
    position of the group's newest run so ordering stays newest-first
    across both kinds. Ungrouped rows (all pre-grouping history) stay
    as individual run entries.

    The id scan below stops as soon as `limit` entries are identified,
    and every later query is bounded to the ids it selected, so the
    cost of a page tracks the page size rather than total history. A
    selected group entry still carries ALL of its member runs, even
    members older than where the scan stopped.
    """
    # Pass 1: which entries make the page. Ids only, lazily iterated;
    # sqlite walks the rowid index newest-first and we stop early.
    entry_order: list[tuple[str, int]] = []
    lone_ids: list[int] = []
    group_ids: list[int] = []
    seen_groups: set[int] = set()
    for row in conn.execute("SELECT id, group_id FROM runs ORDER BY id DESC"):
        gid = row["group_id"]
        if gid is None:
            entry_order.append(("run", row["id"]))
            lone_ids.append(row["id"])
        elif gid not in seen_groups:
            seen_groups.add(gid)
            entry_order.append(("group", gid))
            group_ids.append(gid)
        if len(entry_order) == limit:
            break
    if not entry_order:
        return []

    def marks(ids: list[int]) -> str:
        return ",".join("?" * len(ids))

    # Pass 2: the runs backing those entries, the lone runs plus every
    # member of each selected group. Built conditionally because
    # "IN ()" with zero placeholders is a syntax error.
    conditions = []
    params: list[int] = []
    if lone_ids:
        conditions.append(f"id IN ({marks(lone_ids)})")
        params += lone_ids
    if group_ids:
        conditions.append(f"group_id IN ({marks(group_ids)})")
        params += group_ids
    run_rows = [
        dict(r)
        for r in conn.execute(
            "SELECT id, group_id, prompt_text, created_at FROM runs"
            f" WHERE {' OR '.join(conditions)} ORDER BY id",
            params,
        ).fetchall()
    ]
    run_ids = [r["id"] for r in run_rows]
    # Second query instead of GROUP_CONCAT: keeps model order tied to
    # insert order, which mirrors the original request order.
    models_by_run: dict[int, list[str]] = {}
    # request_json rides this same query so the ungrouped-controls fallback
    # costs no extra round trip. First result per run wins: every member of
    # one run was sent the same controls, so any of them proves the set.
    request_by_run: dict[int, object] = {}
    for row in conn.execute(
        f"SELECT run_id, model, request_json FROM results"
        f" WHERE run_id IN ({marks(run_ids)})"
        " ORDER BY id",
        run_ids,
    ):
        models_by_run.setdefault(row["run_id"], []).append(row["model"])
        request_by_run.setdefault(row["run_id"], row["request_json"])
    group_created: dict[int, str] = {}
    group_controls: dict[int, dict[str, Any]] = {}
    if group_ids:
        for row in conn.execute(
            f"SELECT id, created_at, params_json FROM groups"
            f" WHERE id IN ({marks(group_ids)})",
            group_ids,
        ):
            group_created[row["id"]] = row["created_at"]
            group_controls[row["id"]] = _decoded_params(row["params_json"])

    runs_by_id = {r["id"]: r for r in run_rows}
    members: dict[int, list[dict[str, Any]]] = {}
    for r in run_rows:
        if r["group_id"] is not None:
            members.setdefault(r["group_id"], []).append(r)

    entries = []
    for kind, key in entry_order:
        if kind == "run":
            run = runs_by_id[key]
            entries.append(
                {
                    "type": "run",
                    "id": run["id"],
                    "created_at": run["created_at"],
                    "prompt_text": run["prompt_text"],
                    "models": models_by_run.get(run["id"], []),
                    # Derived from the payload, since controls are recorded
                    # on the group and this run has none. See
                    # _controls_from_request for what that can and cannot
                    # prove, routing being the one it cannot.
                    "params": _controls_from_request(request_by_run.get(run["id"]))
                    or None,
                }
            )
        else:
            runs_asc = members[key]
            entries.append(
                {
                    "type": "group",
                    "id": key,
                    "created_at": group_created.get(key, runs_asc[0]["created_at"]),
                    "prompt_text": runs_asc[0]["prompt_text"],
                    "models": [
                        m for r in runs_asc for m in models_by_run.get(r["id"], [])
                    ],
                    "run_ids": [r["id"] for r in runs_asc],
                    # The stored record, not a derivation: the group row
                    # holds what was declared before any call, so a group
                    # can show a routing badge where a lone run cannot.
                    "params": group_controls.get(key) or None,
                }
            )
    return entries


def _repaired(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a result row's scalar fields on the way out.

    Rows written before ingestion normalization can carry provider
    junk: one string token count persisted raw turned GET /runs/{id}
    into a permanent 500. Repair on read retires that poisoning
    without a destructive migration, and the raw row stays in the file
    as evidence.
    """
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "max_tokens",
        "reasoning_tokens",
        "cached_tokens",
    ):
        row[field] = as_token_count(row[field])
    for field in ("latency_ms", "ttft_ms", "cost_usd"):
        row[field] = as_metric(row[field])
    # Money, so the finite rule applies on the way out too: a poisoned
    # billed cost in an old row must read as None, never as a number a
    # reconciliation or a total would trust.
    row["billed_cost_usd"] = as_money(row["billed_cost_usd"])
    for field in (
        "response_text",
        "error",
        "generation_id",
        "finish_reason",
        "request_json",
        "provider",
        "quantization",
        "native_finish_reason",
    ):
        row[field] = as_text(row[field])
    # position orders the replay, so junk in it must not reorder history;
    # a non-integer degrades to None and falls back to insertion order.
    row["position"] = as_token_count(row["position"])
    return row


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    run = conn.execute(
        """SELECT id, prompt_id, prompt_text, created_at,
                  app_sha, catalog_snapshot_at, data_policy
           FROM runs WHERE id = ?""",
        (run_id,),
    ).fetchone()
    if run is None:
        return None
    # Ordered by position when the rows carry one, falling back to
    # insertion order when they do not. A run's results are written in a
    # single executemany from a single code path, so in practice every row
    # of a run is positioned or none is, and each case orders exactly as
    # intended. The COALESCE only has to keep a hypothetical mix
    # deterministic, which it does, though not gracefully: it compares a
    # zero-based column index against a rowid, so positioned rows would all
    # sort ahead of unpositioned ones rather than interleaving by place.
    results = conn.execute(
        """SELECT model, response_text, latency_ms, prompt_tokens,
                  completion_tokens, error, cost_usd, ttft_ms, max_tokens,
                  generation_id, finish_reason, position, request_json,
                  billed_cost_usd, reasoning_tokens, cached_tokens,
                  provider, quantization, native_finish_reason
           FROM results WHERE run_id = ?
           ORDER BY COALESCE(position, id), id""",
        (run_id,),
    ).fetchall()
    out = dict(run)
    out["results"] = [_repaired(dict(r)) for r in results]
    # The controls this run was sent with, derived from its own recorded
    # payload. Controls are declared on the group, so a run row never stores
    # them; deriving here rather than in the frontend keeps the one place
    # that knows the payload shape inside this module. Any result proves the
    # same set, since one run sends one experiment to every model, so the
    # first is enough. See _controls_from_request for the routing caveat.
    out["params"] = (
        _controls_from_request(out["results"][0]["request_json"])
        if out["results"]
        else {}
    ) or None
    return out


def get_group(conn: sqlite3.Connection, group_id: int) -> dict[str, Any] | None:
    """The group's runs with full results, run order by id, plus the
    controls the comparison was declared with.

    params comes back decoded rather than as the stored string, so the one
    place that knows how the controls are serialized is this module. It is
    None rather than an empty mapping when nothing was set, because the
    detail view has to render a set control and only a set control, and
    None is the shape that cannot be mistaken for a control set to a
    falsy value.
    """
    row = conn.execute(
        "SELECT id, created_at, params_json FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    if row is None:
        return None
    run_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM runs WHERE group_id = ? ORDER BY id", (group_id,)
        )
    ]
    out = dict(row)
    params = _decoded_params(out.pop("params_json"))
    out["params"] = params or None
    out["runs"] = [get_run(conn, rid) for rid in run_ids]
    return out


def results_awaiting_reconciliation(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[dict[str, Any]]:
    """Result rows that have a generation id but no billed cost yet.

    The reconcile pass's work list. Aborted rows qualify by construction:
    they carry an id (captured from the response header before any chunk)
    and never a billed cost, since no usage object ever arrived, and they
    are the rows whose billing is least knowable without asking.

    "No billed cost yet" is the money rule, not IS NULL. The two are not
    the same predicate, and the gap between them was a trap: a negative or
    non-numeric charge is NOT NULL to SQL while _repaired hands every
    reader None for it, so such a row displayed as unpriced forever AND was
    invisible to the one pass that could have fixed it. The three clauses
    below are as_money expressed in SQL, so a value no reader will trust is
    a value this list still offers to replace. NaN needs no clause: SQLite
    has no NaN and binds one as NULL, which the first clause already
    catches.

    Ordered by id so a run interrupted partway through resumes in a
    predictable place, and so a --limit run walks the oldest rows first.
    """
    sql = (
        "SELECT r.id, r.run_id, r.model, r.generation_id, r.cost_usd, r.error "
        "FROM results r "
        "WHERE r.generation_id IS NOT NULL "
        "AND (r.billed_cost_usd IS NULL "
        "     OR typeof(r.billed_cost_usd) NOT IN ('real', 'integer') "
        "     OR r.billed_cost_usd < 0) "
        "ORDER BY r.id"
    )
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


# The only columns the reconcile pass may write. Text, errors, timings and
# the local estimate are excluded on purpose: this is an audit that adds
# what the platform knows, not a rewrite of what the bench observed. A pass
# that could edit response_text would make history unfalsifiable.
RECONCILABLE_COLUMNS = (
    "billed_cost_usd",
    "provider",
    "quantization",
    "native_finish_reason",
)


def apply_reconciliation(
    conn: sqlite3.Connection, result_id: int, record: dict[str, Any]
) -> None:
    """Write the audit columns for one result from a generation record.

    COALESCE on the three text columns so a field the generation endpoint
    does not return cannot erase one captured in-band: absence there means
    "not reported", never "known to be nothing". billed_cost_usd is
    assigned outright because the work list only offers rows whose stored
    charge no reader would trust anyway (results_awaiting_reconciliation),
    so overwriting it with None loses nothing and replaces a poisoned value
    with an honest absence.
    """
    with conn:
        conn.execute(
            """UPDATE results
               SET billed_cost_usd = ?,
                   provider = COALESCE(?, provider),
                   quantization = COALESCE(?, quantization),
                   native_finish_reason = COALESCE(?, native_finish_reason)
               WHERE id = ?""",
            (
                record["billed_cost_usd"],
                record["provider"],
                record["quantization"],
                record["native_finish_reason"],
                result_id,
            ),
        )
