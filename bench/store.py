"""SQLite data layer. Pure functions over an injected connection.

Same reasoning as the injected httpx client in models.py: the caller
owns the connection's lifecycle, tests hand in :memory:, and swapping
sqlite for something else later touches only this module.

THE CONTRACT: this module is SYNCHRONOUS BY DESIGN. Every public
function materializes its result before returning (fetchall or fetchone,
never a live cursor or a generator), and no cursor is ever held across an
await. There is no async here and there must not be.

That is not a style preference, it is the concurrency safety of the whole
app. One connection is shared by every request, every experiment trial
and every scoring pass in the process. On one event loop, a synchronous
function that materializes before returning is an ATOMIC BLOCK: nothing
else runs between its first statement and its last, so no second caller
can interleave a write into the middle of a read or find the connection
mid-statement. That property, rather than a lock somebody has to remember
to take, is what makes one shared sqlite3 connection safe here at all.

The day someone adds a single async helper, or hands back a live cursor
for a caller to iterate across an await, the property is gone everywhere
at once and NOTHING FAILS LOUDLY. The reads still work, until two callers
overlap under load and one sees half of the other's transaction. So there
is a test asserting this file contains no async or await tokens.

Cross-PROCESS concurrency is a different problem with a different answer:
WAL mode and the busy timeout, both set in connect(). This contract is
about the one process that holds this connection.
"""

import contextlib
import json
import logging
import os
import sqlite3
import stat
import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

# Shared with ingestion on purpose: what run_model refuses to emit,
# get_run refuses to serve, so both ends of the pipeline enforce the
# same field-type contract.
from bench.extract import media_type_for
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
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_digest TEXT NOT NULL,
    lineup_json TEXT NOT NULL,
    budget TEXT NOT NULL,
    params_json TEXT,
    repeats INTEGER NOT NULL,
    task_order_seed INTEGER,
    estimand_mode TEXT NOT NULL,
    primary_metric TEXT,
    quantizations_json TEXT,
    provider_pins_json TEXT,
    halt_on_refusal INTEGER NOT NULL,
    status TEXT NOT NULL,
    status_detail TEXT,
    app_sha TEXT,
    catalog_digest TEXT,
    data_policy TEXT,
    tasks_total INTEGER NOT NULL,
    trials_total INTEGER NOT NULL,
    trials_done INTEGER NOT NULL,
    trials_refused INTEGER NOT NULL,
    trials_failed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id),
    scorer TEXT NOT NULL,
    score REAL,
    passed INTEGER,
    detail TEXT,
    judge_model TEXT,
    judge_generation_id TEXT,
    judge_billed_cost_usd REAL,
    blind INTEGER,
    self_judged INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt_text TEXT,
    models_json TEXT,
    params_json TEXT,
    budget TEXT,
    experiment_id INTEGER NULL REFERENCES experiments(id),
    task_id TEXT,
    repeat_index INTEGER,
    rotation_index INTEGER,
    attachments_json TEXT,
    attachments_mode TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    prompt_id INTEGER NULL REFERENCES prompts(id) ON DELETE SET NULL,
    group_id INTEGER NULL REFERENCES groups(id),
    prompt_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    app_sha TEXT,
    catalog_snapshot_at TEXT,
    data_policy TEXT,
    catalog_digest TEXT,
    renditions_json TEXT
);
CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY,
    digest TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    content BLOB NOT NULL,
    extracted_text TEXT NOT NULL,
    extractor TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    extracted_chars INTEGER
);
CREATE TABLE IF NOT EXISTS attachment_extractions (
    id INTEGER PRIMARY KEY,
    digest TEXT NOT NULL,
    extractor TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT,
    filename TEXT,
    mime TEXT,
    UNIQUE (digest, extractor, extractor_version)
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
    native_finish_reason TEXT,
    upstream_inference_cost_usd TEXT
);
"""

# History reads join results by run and replay groups by scanning runs
# by group; both tables grow monotonically, so without these indexes
# every replay degrades into a full-table scan as bench.db accumulates.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_group_id ON runs(group_id);
CREATE INDEX IF NOT EXISTS idx_groups_experiment_id ON groups(experiment_id);
CREATE INDEX IF NOT EXISTS idx_scores_result_id ON scores(result_id);
"""

# Deleting an attachment deletes every reading of it.
#
# A TRIGGER RATHER THAN "ON DELETE CASCADE", for two reasons that both
# have to hold.
#
# The first is the migration discipline: attachment_extractions already
# exists on databases in the field, and sqlite cannot add a foreign key
# to a live table. Getting one there means rebuilding the table and
# copying it, which is not an additive migration and is a destructive
# operation run at boot on the file holding somebody's documents.
#
# The second is the one that actually decides it. A foreign key cascade
# is INERT unless the connection sets `PRAGMA foreign_keys = ON`:
# "Foreign key constraints are disabled by default (for backwards
# compatibility), so must be enabled separately for each database
# connection", and the page's own shell transcript shows the sqlite3
# command line answering 0 to `PRAGMA foreign_keys`
# (https://www.sqlite.org/foreignkeys.html, read 2026-08-09). Nothing in
# this application deletes an attachment; the only actor who can is a
# person with sqlite3 and their own bench.db, which is exactly what
# AttachmentRef's docstring names and exactly the case a key would not
# have covered. A trigger fires for every deleter, pragma or no pragma,
# and the same page warns against assuming the default will not change,
# which a trigger does not have to care about either way.
#
# What survives a delete without this is not a dangling id: it is the
# extracted TEXT of the document, sitting in a second table under the
# same digest, after the person believed they had removed it.
TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS attachments_delete_cascades_extractions
AFTER DELETE ON attachments
BEGIN
    DELETE FROM attachment_extractions WHERE digest = OLD.digest;
END;
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
    # Phase H.1. Two columns, both nullable and additive.
    #
    # groups.budget completes the manifest. The group already declared its
    # prompt, its ordered lineup and its controls before any call; the
    # budget was the one part of "what this comparison is" that lived only
    # on the members. A comparison whose members ran at different token
    # budgets is not one experiment, and without this column nothing could
    # say so.
    #
    # runs.catalog_digest is the provenance companion to
    # catalog_snapshot_at: the timestamp says WHEN the price catalog was
    # read, this says WHICH bytes were read, so two runs costed against
    # catalogs that happened to be fetched a minute apart can be told
    # apart. None on an offline boot, where there were no bytes.
    ("groups", "budget", "TEXT"),
    ("runs", "catalog_digest", "TEXT"),
    # Phase I.2. Both nullable and additive, like everything above.
    #
    # experiments.primary_metric is which scorer a ranking is computed
    # on. NULL means none was declared, which is an answer rather than a
    # gap: with more than one scorer the report then publishes sections
    # and no cross-scorer ranking, because ordering models needs somebody
    # to say better at what.
    #
    # results.upstream_inference_cost_usd is what the provider charged
    # directly under BYOK, which is a different number from the credits
    # the ceiling meters. TEXT rather than REAL because it arrives as a
    # decimal string and a float conversion would round money on the way
    # in; readers parse it themselves and the ceiling never reads it.
    ("experiments", "primary_metric", "TEXT"),
    # Strict mode's optional quantization filter, stored as a JSON array
    # beside the pins for the same reason: it is part of the declaration
    # of what the experiment IS, and a report citing a narrowed provider
    # population has to be able to say how it was narrowed.
    ("experiments", "quantizations_json", "TEXT"),
    ("results", "upstream_inference_cost_usd", "TEXT"),
    # Phase I, the evaluation layer. Four columns on groups, all nullable
    # and additive; the two new tables need no entry here because
    # CREATE TABLE IF NOT EXISTS in SCHEMA creates them on any database,
    # new or old, which is what makes them additive by construction.
    #
    # The group stays exactly what H.1 made it: the atomic one-prompt
    # record, created before any call, enforced at entry. These columns do
    # not change that; they say which larger thing a group belongs to.
    # experiments is an aggregate ABOVE groups, which is the shape the
    # Phase G out-of-scope note promised: the concept waited for repeats
    # to exist, repeats exist now, so it arrives as a layer rather than a
    # rename of what was already there.
    #
    # NULL on all four is the affirmative fact that a group was run by
    # hand rather than by an experiment, which is what every group in
    # every pre-I database was. That reading needs no fallback and no
    # derivation, so unlike models_json and budget these are never
    # "unknown": a hand-run comparison genuinely has no task id.
    ("groups", "experiment_id", "INTEGER NULL REFERENCES experiments(id)"),
    ("groups", "task_id", "TEXT"),
    ("groups", "repeat_index", "INTEGER"),
    ("groups", "rotation_index", "INTEGER"),
    # Phase K, attachments. TWO columns, both nullable and additive; the
    # attachments table itself needs no entry here for the same reason
    # the Phase I tables did not, since CREATE TABLE IF NOT EXISTS in
    # SCHEMA creates it on any database, new or old.
    #
    # An ordered JSON array of content digests, and ORDER is part of the
    # declaration rather than incidental: two documents composed in the
    # other order are a different prompt, so the list is the manifest's
    # record of which documents in which sequence.
    #
    # THE TWO NULLS, and this column has both, which is why the rule is
    # written at the check site rather than only here. NULL means the
    # pre-K era: the group was created by a build that had no attachment
    # concept, so it declared nothing and could not have. An empty array
    # would mean a post-K group that declared no attachment, which is a
    # different fact. Nothing writes "[]" for the same reason params_json
    # never stores "{}": absence gets exactly one spelling.
    ("groups", "attachments_json", "TEXT"),
    # How those documents reach the models: "inline" or "native". NULL
    # when the group declared no attachment at all, which is the same
    # absence attachments_json records and is stored the same way, so a
    # reader never has to reconcile a mode against no documents.
    ("groups", "attachments_mode", "TEXT"),
    # Phase K.1: the RENDITION PIN. Two columns on attachment_extractions
    # and one on groups, all nullable and additive.
    #
    # A rendition is (digest, extractor, extractor_version, kind): not
    # "which bytes" but "which reading of which bytes". Phase K keyed
    # everything on the digest alone, and the digest is a fact about the
    # file while the text a model reads is a fact about the file AND the
    # parser AND which parser the suffix chose. Three measured
    # consequences of the shorter key, all of them silent:
    #
    #   PNG bytes uploaded as shot.txt were accepted into inline mode,
    #   because the check read the STORED filename, and every model
    #   received seventy characters of latin-1-decoded binary as the
    #   "document".
    #
    #   The same bytes re-uploaded as shot.png were then refused from
    #   native mode forever, because the row still said shot.txt.
    #
    #   A .docx uploaded a second time as .txt made POST and GET
    #   disagree about the same digest: 324 characters of mojibake from
    #   one, 25 characters of real text from the other.
    #
    # kind and filename on the extraction row are what each rendition
    # was actually read AS, so a second suffix is a second rendition
    # rather than a contradiction about the first.
    #
    # NO CONSTRAINT CHANGE, and that is what keeps this additive. SQLite
    # cannot widen UNIQUE (digest, extractor, extractor_version) and
    # cannot ADD COLUMN UNIQUE, so a four-column key would be
    # unreachable on any existing database. It is not needed: kind is a
    # FUNCTION of the extractor ("none" is the image reader and nothing
    # else is), so the three-column constraint already enforces the
    # four-part key and the fourth part is recorded rather than
    # constrained. If a future extractor ever produced two kinds, this
    # comment is the thing that would have to change first.
    ("attachment_extractions", "kind", "TEXT"),
    ("attachment_extractions", "filename", "TEXT"),
    # Phase K.3: THE MEDIA TYPE JOINS THE RENDITION, because it was
    # already a property of the reading and was being stored as a
    # property of the bytes.
    #
    # attachments.mime belongs to whichever upload arrived first. The
    # same digest read as .txt and as .png is two renditions with two
    # honest types, and native composition read the base row's: a real
    # PNG first uploaded under a .txt suffix and later pinned as an
    # image reached the provider as data:text/plain;base64,iVBORw0KG,
    # measured. The recorded placeholder said "as image, 41 bytes,
    # text/plain" in the same breath, so the record contradicted itself
    # about one document.
    #
    # THE UNIQUE KEY IS DELIBERATELY NOT WIDENED, and the reason first
    # written here was wrong. It said the key already separates every
    # type, because the mime and the extractor are both functions of the
    # suffix. They are, and that does not follow: .txt and .md are two
    # suffixes with two media types and ONE extractor at one version, so
    # the key holds a single row for them and the second upload finds
    # the first's.
    #
    # That is the right outcome and the honest description of it is
    # different. Those two uploads are one READING: the text extractor
    # produces identical text from identical bytes whichever suffix
    # named them, so there is one rendition and one row, and the media
    # type recorded is the one the first upload was read under. Widening
    # the key would split one reading into two rows differing only in a
    # label, which is the "one document into three" mistake the header
    # ruling refused for filenames.
    #
    # What it costs is that a .txt upload of bytes already stored as .md
    # is answered with the .md rendition, name and type together. That
    # is exactly what save_attachment's earlier-name-wins rule promises
    # and what the composer tells the person in words.
    ("attachment_extractions", "mime", "TEXT"),
    # Phase K.1: the extracted length, STORED rather than measured on
    # read. A history page shows how much text came out of each
    # document, and deriving that from the text meant every list
    # response loaded every body: measured at 15114 characters read to
    # report three numbers, on a page whose own bound is 500 entries.
    #
    # Computed in Python and not by sqlite's length(), which is the
    # detail that makes this a column instead of an expression. A binary
    # file uploaded under a .txt suffix decodes through latin-1 into
    # text containing NUL, and sqlite's length() stops counting there:
    # measured at 2 for a five-character string. A number that is
    # sometimes silently short is worse than a query that is slow.
    ("attachments", "extracted_chars", "INTEGER"),
    # The ordered rendition list a group pinned at creation, as JSON
    # objects rather than the bare digests attachments_json holds.
    #
    # THREE ERAS AND THREE NULLS, which is why the rule is written at
    # the check site (see group_renditions). NULL here with
    # attachments_json also NULL is pre-K: no attachment concept
    # existed. NULL here with attachments_json set is post-K and
    # pre-K.1: documents were declared but only by digest, so which
    # reading they got was whatever the parser of the day produced.
    # Present is K.1: the reading itself is pinned and resolution
    # honors it or refuses.
    ("groups", "renditions_json", "TEXT"),
    # The same pin on the RUN, and it is not a duplicate of the group's.
    #
    # A grouped run inherits its group's pin and this column agrees with
    # it. An UNGROUPED run has no group to inherit from, and before this
    # there was nowhere at all that recorded which documents it carried:
    # the history view hardcoded an empty list for lack of anywhere to
    # read, so a comparison run over a contract replayed as a comparison
    # over a bare prompt with nothing on screen saying otherwise.
    #
    # The digests DO survive inside each result's request_json, in the
    # placeholder standing where the content sat, and this column exists
    # so nobody has to parse them back out. Reading a record format as a
    # data format is the same mistake as reading controls off their
    # badges, and it would have been worse here: the placeholder is a
    # string the composition owns and may reword.
    #
    # NULL is the pre-K.1 era on this table, exactly as on groups.
    ("runs", "renditions_json", "TEXT"),
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
    # DELETED CONTENT IS OVERWRITTEN, and this is set explicitly because
    # the default is a property of whoever compiled sqlite rather than of
    # this application. SQLite's own page says it "is determined by the
    # SQLITE_SECURE_DELETE compile-time option and is normally off"
    # (https://www.sqlite.org/pragma.html#pragma_secure_delete, read
    # 2026-08-10); the interpreter in this environment reports 1 because
    # its build defines that option, so relying on the default would mean
    # the privacy posture changed with the Python somebody installed.
    #
    # WHAT IT BUYS. Deleting an attachment frees the pages its BLOB and
    # its text occupied, and without this the bytes stay in those pages
    # until something else claims them. Measured on forty 120 KiB
    # documents: with the pragma the file holds 0 occurrences of the
    # marker string after the delete, and the delete costs 0.0507
    # seconds against 0.0043 with the pragma off. Fifty milliseconds to
    # remove forty confidential documents is not a cost worth trading
    # for.
    #
    # "FAST" WAS MEASURED AND REJECTED, and it is the trap here. The
    # documentation calls it an intermediate setting that "purges all old
    # content from b-tree pages, but leaves forensic traces on freelist
    # pages", which sounds like most of the benefit for none of the I/O.
    # A document does not live in b-tree pages: a BLOB of any size lives
    # in OVERFLOW pages, which go to the freelist. On the same forty
    # documents:
    #
    #   setting   pragma   marker occurrences left   delete
    #   off            0                   317,887   0.0047s
    #   fast           2                   317,805   0.0050s
    #   on             1                         0   0.0536s
    #
    # Fast is not an intermediate setting for this data. It is the off
    # setting with a longer name and eighty-two fewer copies.
    #
    # WHAT IT DOES NOT BUY, stated because the README states it too: the
    # write-ahead log is a separate file and this pragma says nothing
    # about it. The same measurement found 79,640 occurrences in
    # bench.db-wal after a delete that left none in bench.db. A
    # checkpoint clears them; see the README's note on VACUUM.
    conn.execute("PRAGMA secure_delete = ON")
    conn.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    # After the migrations, not before: runs.group_id may only exist
    # once the ALTERs above have run on an old database.
    conn.executescript(INDEXES)
    conn.executescript(TRIGGERS)
    # PHASE K.1 BACKFILL, idempotent, and it is here rather than in a
    # reader because of a real gap rather than for tidiness.
    #
    # attachment_extractions shipped in c9ad170 and nothing backfilled
    # it, so a database created by any build between the two carries
    # attachments rows and zero extraction rows. K.1 resolves a pinned
    # rendition against that table and REFUSES rather than substituting,
    # so without this every such document would become unusable the day
    # the pin arrived: a correct refusal about a document the bench is
    # holding and can read perfectly well.
    #
    # The attachments row IS an extraction: it carries the text, the
    # extractor and the version of the first reading. This copies that
    # fact into the table that now owns it, once. INSERT OR IGNORE makes
    # every later boot a no-op, and kind is derived from the EXTRACTOR
    # rather than from the filename, because the extractor is what
    # actually did the reading and the filename is the thing K.1 stopped
    # trusting.
    conn.execute(
        """INSERT OR IGNORE INTO attachment_extractions
               (digest, extractor, extractor_version, extracted_text,
                created_at, kind, filename, mime)
           SELECT digest, extractor, extractor_version, extracted_text,
                  created_at,
                  CASE WHEN extractor = 'none' THEN 'image' ELSE 'document' END,
                  filename, mime
           FROM attachments"""
    )
    # AND THE ROWS THAT WERE ALREADY THERE, which the statement above
    # cannot reach and silently skips.
    #
    # INSERT OR IGNORE hits the UNIQUE (digest, extractor,
    # extractor_version) key and does nothing for a digest whose
    # extraction row already exists. That is right for the text, which
    # must not be rewritten, and wrong for the columns added later:
    # every database created by the c9ad170 build has the TABLE but not
    # the COLUMNS, so its rows arrive here with kind, filename and mime
    # all NULL and the insert above leaves them that way forever.
    #
    # The era test missed it by building its fixture with no extraction
    # rows at all, which exercises the insert and never the conflict.
    # That is the wrong-window shape one table over: the fixture held
    # the state the migration was written for rather than the state it
    # would meet.
    #
    # COALESCE, so a value already present is never overwritten: this
    # fills gaps and does not re-derive facts. kind comes from the row's
    # OWN extractor, not the base row's, because an extraction by "none"
    # is an image whichever upload created the attachments row.
    conn.execute(
        """UPDATE attachment_extractions AS e
              SET kind = COALESCE(
                      e.kind,
                      CASE WHEN e.extractor = 'none' THEN 'image'
                           ELSE 'document' END),
                  filename = COALESCE(
                      e.filename,
                      (SELECT a.filename FROM attachments a
                        WHERE a.digest = e.digest))
            WHERE e.kind IS NULL OR e.filename IS NULL"""
    )
    # THE COUNT BACKFILL, once per database and never again.
    #
    # Every row written before the column existed has it NULL, and the
    # readers below now report it instead of measuring the text, so a
    # NULL here would surface as a missing character count on a document
    # the bench can read perfectly well.
    #
    # In Python rather than as `UPDATE ... SET extracted_chars =
    # length(extracted_text)`, for the reason the migration entry gives:
    # sqlite's length() stops at the first NUL and the SQL form would
    # write a short number that nothing later could tell from a right
    # one. This reads each body once, on the first boot after the
    # upgrade, to avoid reading all of them on every list request after
    # it.
    #
    # Rows are streamed and only the pairs are kept, so the peak is one
    # document rather than the whole table.
    counted = [
        (len(row["extracted_text"]), row["digest"])
        for row in conn.execute(
            "SELECT digest, extracted_text FROM attachments"
            " WHERE extracted_chars IS NULL"
        )
    ]
    if counted:
        conn.executemany(
            "UPDATE attachments SET extracted_chars = ? WHERE digest = ?", counted
        )
    # THE MEDIA-TYPE BACKFILL, and it is EXACT rather than a guess.
    #
    # Rows written before K.3 have no mime of their own. Every one of
    # them records the FILENAME it was read under, though, and the
    # boundary computed the type from exactly that name with exactly
    # this function, so applying it here reproduces the string that
    # upload would have written. That is why media_type_for lives in
    # extract.py rather than at the boundary: two copies of the mapping
    # would be two answers, and the backfill would then be a guess at
    # what the other one had said.
    #
    # A row with no filename cannot be typed and is left NULL rather
    # than defaulted; readers below fall back to the base row and say so.
    # connect() backfills filename from the base row one statement up, so
    # this is reachable only for a row somebody wrote by hand.
    typed = [
        (media_type_for(row["filename"]), row["id"])
        for row in conn.execute(
            "SELECT id, filename FROM attachment_extractions"
            " WHERE mime IS NULL AND filename IS NOT NULL"
        )
    ]
    if typed:
        conn.executemany(
            "UPDATE attachment_extractions SET mime = ? WHERE id = ?", typed
        )
    conn.commit()
    return conn


@contextlib.contextmanager
def read_snapshot(conn: sqlite3.Connection) -> Iterator[None]:
    """One consistent view of the database for the whole block.

    `with conn` DOES NOT DO THIS and that is the trap it replaces. The
    sqlite3 connection context manager commits or rolls back at exit; it
    does not issue a BEGIN. Under the default isolation level the module
    starts a transaction implicitly before INSERT, UPDATE, DELETE and
    REPLACE and NOT before SELECT, so a block containing only reads runs
    every one of them in its own autocommit transaction. `with conn`
    around a pure read is therefore a no-op wearing the costume of a
    snapshot, which is worse than nothing: it reads as the guarantee and
    provides none of it.

    BEGIN is deferred, so the read lock is taken at the first SELECT and
    held to the end of the block. In WAL mode that pins one version of
    the database for every statement inside, which is what makes a
    digest over the result seal a MOMENT rather than an interval.

    The writer this defends against is another CONNECTION, not another
    task. Nothing inside awaits, so the store's synchronous contract
    already rules out an interleaving on this connection. What it cannot
    rule out is `python -m bench.reconcile --apply` against a live bench,
    which is a second connection by design and is the reason this file
    turns WAL on at all.

    rollback rather than commit, because nothing was written and rollback
    is the honest close for a read. It also cannot fail on a busy
    database the way a commit can.

    THE SNAPSHOT IS ONLY AS GOOD AS WHAT RUNS INSIDE IT, and that is why
    the exit checks rather than assumes. Every write helper in this
    module uses `with conn`, and the sqlite3 connection context manager
    COMMITS at exit: one of them called inside this block ends the
    transaction early and every read after it sees a different state of
    the database. Nothing fails, nothing logs, and the artifact built
    from those reads straddles two moments while its digest verifies
    perfectly, which is the exact defect this function exists to prevent
    arriving through the front door instead.

    No caller does that today. That is a fact about today's callers and
    not a property of anything, so it is asserted rather than trusted:
    a silent loss of isolation becomes a loud one.
    """
    conn.execute("BEGIN")
    try:
        yield
        held = conn.in_transaction
    finally:
        conn.rollback()
    if not held:
        raise RuntimeError(
            "the read snapshot was committed from inside its own block, so "
            "the reads after that point saw a different state of the "
            "database than the reads before it. Something in the block "
            "called a store function that opens `with conn`, which commits "
            "at exit. A snapshot that ends early is worse than no snapshot: "
            "it produces an artifact that straddles two moments and a "
            "digest over it that verifies."
        )


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
    budget: str | None = None,
    experiment: dict[str, Any] | None = None,
    attachments: list[str] | None = None,
    attachments_mode: str | None = None,
    renditions: list[dict[str, Any]] | None = None,
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

    experiment is the Phase I placement: which experiment this group
    belongs to, and which cell of it (task, repeat, rotation). Absent for
    every hand-run comparison, which is what NULL means on those four
    columns. It is a plain dict rather than four parameters because the
    four are meaningless apart: a task id without a repeat index does not
    identify a cell, and letting them be passed separately would let a
    caller record half a placement.

    attachments is the ordered digest list, and it joins the manifest for
    the same reason the lineup did: it is part of what the comparison IS,
    fixed before any call, so a member arriving with a different set can
    be refused against a declaration that already exists. NOT sorted, and
    not deduplicated: two documents composed in the other order are a
    different prompt, so the order given is the order recorded.

    An empty list stores NULL, exactly as an empty params mapping does,
    because absence gets one spelling. The reader's two NULLs then stay
    honest: see group_attachments.

    renditions is Phase K.1's pin and is the SOURCE of both attachment
    columns. It carries (digest, extractor, extractor_version, kind) per
    document in declaration order, and attachments_json is DERIVED from
    it here rather than passed in beside it. One write, one source: two
    lists supplied independently could disagree about which documents a
    group holds, and every digest-only reader downstream (list_runs, the
    export, the report) would then describe a different comparison from
    the one the composer built.

    sort_keys on the rendition objects, for the reason params_json has
    it and attachments_json never needed it: a list of bare strings
    serializes exactly one way and a list of objects does not. Two
    identical pins built by two code paths must produce identical column
    bytes, or the export's trailing digest stops being a function of the
    experiment.
    """
    if renditions:
        # Derived, never supplied. See above.
        attachments = [str(r["digest"]) for r in renditions]
    place = experiment or {}
    with conn:
        cur = conn.execute(
            "INSERT INTO groups"
            " (created_at, prompt_text, models_json, params_json, budget,"
            "  experiment_id, task_id, repeat_index, rotation_index,"
            "  attachments_json, attachments_mode, renditions_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                prompt_text,
                json.dumps(models) if models is not None else None,
                json.dumps(params, sort_keys=True) if params else None,
                budget,
                place.get("experiment_id"),
                place.get("task_id"),
                place.get("repeat_index"),
                place.get("rotation_index"),
                json.dumps(attachments) if attachments else None,
                attachments_mode if attachments else None,
                json.dumps(renditions, sort_keys=True) if renditions else None,
            ),
        )
    # lastrowid is Optional in the DBAPI types but always set after a
    # single-row INSERT; assert so the int return stays honest.
    assert cur.lastrowid is not None
    return cur.lastrowid


# ---- Phase K: attachments, stored once by content digest.


# Everything about an attachment except the bodies. Named once so the
# three readers cannot drift into serving different shapes, and so the
# omissions are a decision recorded in one place rather than a column
# list somebody has to remember not to extend.
#
# NEITHER BODY IS IN IT. Not `content`: nothing outside this module
# reads the BLOB, and attachment_content is the single deliberate
# exception. And no longer `extracted_text` either. The text is a body
# too, up to MAX_EXTRACTED_CHARS of it per document, and a list view
# showing a hundred documents used to load a hundred of them to print a
# hundred integers. extracted_chars is that integer, stored at insert.
ATTACHMENT_COLUMNS = (
    "digest",
    "filename",
    "mime",
    "byte_size",
    "extracted_chars",
    "extractor",
    "extractor_version",
    "created_at",
)


def save_attachment(conn: sqlite3.Connection, record: dict[str, Any]) -> dict[str, Any]:
    """Store one attachment, or return the row that already holds it.

    THE BYTES DEDUPE BY DIGEST; THE EXTRACTION DEDUPES BY DIGEST AND
    PARSER VERSION, and the split is the whole of what this function
    now gets right.

    Bytes are content-addressed, so ten comparisons over one contract
    cost one copy of it, and that is what the README's storage promise
    rests on. But the TEXT is not a property of the bytes: it is a
    property of the bytes AND the parser that read them, and a pypdf
    upgrade changes it. Keying the extraction on the digest alone meant
    a re-upload after an upgrade returned the row untouched, so the
    caller got back extractor_version "6.15.0" for text the running
    6.16 never produced, and every comparison created from it recorded a
    provenance claim that was quietly false. The one thing the module
    docstring says the record must always be able to answer, it could
    not.

    So each (digest, extractor, extractor_version) gets its own
    extraction row. A version miss re-extracts FROM THE STORED BYTES
    rather than asking the caller to re-upload, which is the point of
    having kept them.

    OLD EXTRACTIONS ARE KEPT, NOT OVERWRITTEN, and that is rule two's
    doing. A comparison recorded before the upgrade cites this digest
    and its request_json carries a placeholder naming the character
    count of the text THAT parser produced. Updating in place would make
    every such record unreconstructible while leaving it looking exact.
    The attachments row keeps its first extraction as the era record;
    the extractions table holds each version, so both the old record and
    the new comparison can be answered.

    THE EARLIER FILENAME WINS on a collision, and that is deliberate
    rather than incidental. The row is keyed by content, so the name is
    already only one of the names those bytes have had; overwriting it
    would rewrite the label on every comparison that already cites this
    digest, which is history changing under a citation. The upload
    response carries the stored name, so a caller that uploaded
    `contract-v2.pdf` and got back `contract.pdf` can see that it sent
    bytes the bench already had.

    INSERT OR IGNORE rather than a read-then-write, because the read and
    the write would be two statements with a gap between them, and the
    UNIQUE index is the thing that actually enforces this. The follow-up
    SELECT is inside the same transaction, so it sees the row whichever
    of the two paths created it.

    THE BASE ROW IS PROVEN BEFORE THE EXTRACTION IS WRITTEN, and the
    order is the guarantee rather than a tidiness. INSERT OR IGNORE
    swallows EVERY constraint failure, not only the UNIQUE one it is
    here for: a record with a NULL mime is silently not inserted, since
    attachments.mime is NOT NULL, while attachment_extractions has no
    mime column and takes its row happily. Written extraction-last with
    the check after the block, that committed the document's text under
    a digest the attachments table had never heard of, and only then
    raised. The transaction now ends only after the base row has been
    read back inside it, so the failing case rolls both statements away
    and nothing is left holding the text.
    """
    now = _now()
    # Counted here, once, and stored. See the migration entry for why
    # this is not sqlite's length().
    chars = len(record["extracted_text"])
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO attachments
               (digest, filename, mime, byte_size, content, extracted_text,
                extractor, extractor_version, created_at, extracted_chars)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["digest"],
                record["filename"],
                record["mime"],
                record["byte_size"],
                record["content"],
                record["extracted_text"],
                record["extractor"],
                record["extractor_version"],
                now,
                chars,
            ),
        )
        row = conn.execute(
            f"SELECT {', '.join(ATTACHMENT_COLUMNS)} FROM attachments WHERE digest = ?",
            (record["digest"],),
        ).fetchone()
        if row is None:
            # Inside the transaction on purpose: raising here rolls the
            # ignored insert away with it, and no extraction row has
            # been written yet. A bare assert would do the same and then
            # vanish under -O, which is not a thing to leave the orphan
            # rule resting on.
            raise RuntimeError(
                f"the attachments row for {record['digest'][:12]} was not "
                "created and no reading of it will be stored. INSERT OR "
                "IGNORE suppresses every constraint failure, so this is "
                "a record that violates one of the NOT NULLs on the "
                "table rather than a duplicate."
            )
        # This parser's reading of these bytes, alongside any earlier
        # parser's. IGNORE because re-uploading the same file under the
        # same version is the ordinary case and must stay a no-op.
        conn.execute(
            """INSERT OR IGNORE INTO attachment_extractions
               (digest, extractor, extractor_version, extracted_text,
                created_at, kind, filename, mime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["digest"],
                record["extractor"],
                record["extractor_version"],
                record["extracted_text"],
                now,
                record["kind"],
                # The name THIS rendition was read under, which is not
                # necessarily the name on the attachments row: that one
                # belongs to whichever upload arrived first and must not
                # be rewritten, because comparisons already cite it. The
                # rendition records its own so a second suffix can be
                # described truthfully instead of contradicting the
                # first.
                record["filename"],
                # And the type THIS reading was taken under, for the
                # same reason one line up. The base row's belongs to the
                # first upload, and native composition sends this one.
                record["mime"],
            ),
        )
    out = dict(row)
    # The row on disk carries the FIRST parser's reading under the FIRST
    # name, which is the era record and is deliberately not rewritten.
    # What this call is about is the rendition just produced, so that is
    # what comes back, ALL FOUR PARTS OF IT plus the name it was read
    # under. Overlaying three of the four was the shape that let a
    # caller receive the first upload's kind attached to this reading's
    # text: a rendition that was never stored and never sent.
    #
    # The reading itself is now its LENGTH, since the row no longer
    # carries the text out of this module. The overlay is the same
    # five fields either way, and _view_overlay one module over has to
    # keep the same five for the same reason: it took K1.5 to notice
    # that it had only four.
    out["extracted_chars"] = chars
    out["extractor"] = record["extractor"]
    out["extractor_version"] = record["extractor_version"]
    out["kind"] = record["kind"]
    out["filename"] = record["filename"]
    # And the media type, K.3's addition to the same list. Without it the
    # FIRST upload of a second suffix answered with the base row's:
    # uploading a PNG as y.txt and then as y.png returned
    # `kind: image, mime: text/plain`, measured, which is a rendition the
    # bench does not hold. The re-upload path had the same hole and
    # _view_overlay closes that one.
    out["mime"] = record["mime"]
    return out


def extraction_for(
    conn: sqlite3.Connection, digest: str, extractor: str, version: str
) -> dict[str, Any] | None:
    """One RENDITION: this parser's reading of these bytes, or None.

    Returns the whole row rather than the text alone, because a
    rendition is four facts and a caller that received only the text
    would have to re-derive the other three from somewhere, which is
    the re-derivation K.1 exists to stop.

    None is the caller's cue to re-extract and save, or, when the
    rendition was PINNED by a group, to refuse. It is never a cue to
    fall back on another version's text: that substitution is the whole
    reason this table is keyed the way it is.

    kind and filename are None on rows written before K.1 and on
    nothing else, since connect() backfills them from the attachments
    row at boot. A caller that needs kind derives it from the extractor
    the way ingest does; see rendition_of.

    mime is the type THIS reading was taken under, and it is the field
    K.3 added because native composition had been reading the digest's
    base row for it. Backfilled at boot from the rendition's own
    filename by the same function the boundary uses, so it is None only
    on a row that has neither, which connect() cannot produce.
    """
    row = conn.execute(
        """SELECT digest, extractor, extractor_version, extracted_text,
                  kind, filename, mime
           FROM attachment_extractions
           WHERE digest = ? AND extractor = ? AND extractor_version = ?""",
        (digest, extractor, version),
    ).fetchone()
    return dict(row) if row is not None else None


def extractions_for(
    conn: sqlite3.Connection, pins: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The named RENDITIONS, keyed by their full three-part identity.

    extraction_for's plural, and it exists for the reason attachments_for
    exists beside get_attachment: a view showing a page of comparisons
    must resolve every pin on it in one query, and calling the singular
    in a loop is the N+1 K1.1 removed once already.

    Keyed by the triple and not by the digest, because one digest can
    hold several readings and a mapping keyed by bytes would silently
    pick one of them: exactly the substitution this whole phase is
    about.

    Selected by DIGEST and filtered here rather than with a row-value IN,
    because the number of readings per digest is one or two and the
    digest column is what the UNIQUE index leads with. A pin with no
    stored reading is simply absent, which is the caller's cue to refuse
    or to show the reference bare.
    """
    if not pins:
        return {}
    digests = sorted({pin["digest"] for pin in pins})
    placeholders = ", ".join("?" for _ in digests)
    rows = conn.execute(
        f"""SELECT digest, extractor, extractor_version, kind, filename, mime,
                   length(extracted_text) AS extracted_chars
            FROM attachment_extractions WHERE digest IN ({placeholders})""",
        tuple(digests),
    ).fetchall()
    wanted = {
        (pin["digest"], pin["extractor"], pin["extractor_version"]) for pin in pins
    }
    out = {}
    for row in rows:
        key = (row["digest"], row["extractor"], row["extractor_version"])
        if key in wanted:
            out[key] = dict(row)
    return out


def _decoded_renditions(raw: object) -> list[dict[str, Any]] | None:
    """One renditions_json column as pins, or None.

    ONE DECODER, because there are now four readers of this column
    (group_renditions, run_renditions, list_runs and experiment_groups)
    and four copies of a shape check are four chances for one of them to
    accept something the others refuse. That is the drift this phase is
    about, arriving in the code that reads the declaration rather than
    in the code that writes it.

    Repair on read, the same discipline group_manifest applies to
    models_json: an element that is not a rendition object cannot be
    compared against a request, and enforcing against a broken
    declaration would refuse correct work. str() is deliberately NOT
    used: it succeeds on a dict and would turn a malformed pin into a
    Python repr that every later comparison then operates on.
    """
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, list):
        return None
    out = []
    for item in decoded:
        if not isinstance(item, dict):
            return None
        if not all(
            isinstance(item.get(key), str)
            for key in ("digest", "extractor", "extractor_version", "kind")
        ):
            return None
        out.append(
            {
                key: item[key]
                for key in ("digest", "extractor", "extractor_version", "kind")
            }
        )
    return out


def group_renditions(
    conn: sqlite3.Connection, group_id: int
) -> list[dict[str, Any]] | None:
    """The renditions a group pinned, in declaration order, or None.

    THREE ERAS, TWO OF THEM NULL HERE, and this is the check site the
    migration comment points at.

    None with no attachments_json is pre-K: the group could not have
    declared a document at all. None WITH attachments_json is post-K and
    pre-K.1: it declared digests and nothing about how they were to be
    read, so there is no pin to honor and the caller must fall back on
    resolving the digest. A list is K.1: the reading is pinned and a
    member gets exactly it or a refusal.

    Callers must not collapse the first two into "no documents". The
    pre-K.1 group HAS documents; what it lacks is a statement about
    which reading of them, and treating that as "declared nothing" would
    refuse a member that legitimately brings the digests the group
    holds.
    """
    row = conn.execute(
        "SELECT renditions_json FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    return None if row is None else _decoded_renditions(row["renditions_json"])


def get_attachment(conn: sqlite3.Connection, digest: str) -> dict[str, Any] | None:
    """One attachment's metadata, never its bytes and no longer its text.

    NEITHER BODY IS SELECTED. The content BLOB never was: serving it
    would make the database a file host, and the privacy posture is that
    the document exists in exactly one place a person can delete. The
    extracted text is not selected either, because no caller of this
    function reads it. Composition gets its text from the RENDITION,
    which is the pinned reading rather than whichever one this row
    happens to hold, and the endpoints want the length, which is now a
    column.
    """
    row = conn.execute(
        f"SELECT {', '.join(ATTACHMENT_COLUMNS)} FROM attachments WHERE digest = ?",
        (digest,),
    ).fetchone()
    return dict(row) if row is not None else None


def attachment_content(conn: sqlite3.Connection, digest: str) -> bytes | None:
    """The stored bytes for one attachment. The ONLY reader that returns them.

    Every other reader here omits the BLOB deliberately, and this one is
    the single exception, so the exception is one function a reviewer can
    find rather than a column list to audit.

    It exists for native mode alone: an image sent as a content part has
    to reach the payload as base64, so the bytes must leave the database
    for exactly that one purpose. Nothing serves them to a client, no
    endpoint returns them, and the inline path never calls this at all,
    because inline sends extracted text and has no use for the original.
    """
    row = conn.execute(
        "SELECT content FROM attachments WHERE digest = ?", (digest,)
    ).fetchone()
    return bytes(row["content"]) if row is not None else None


def attachments_for(
    conn: sqlite3.Connection, digests: list[str]
) -> dict[str, dict[str, Any]]:
    """The named attachments by digest, for callers holding a list.

    One query rather than N, and a mapping rather than a list, because
    every caller wants to look up by digest and an ordered list would
    make each of them re-index it. A digest with no row is simply absent
    from the result, which is the caller's cue to refuse rather than
    this function's cue to guess.
    """
    if not digests:
        return {}
    placeholders = ", ".join("?" for _ in digests)
    rows = conn.execute(
        f"SELECT {', '.join(ATTACHMENT_COLUMNS)} FROM attachments"
        f" WHERE digest IN ({placeholders})",
        tuple(digests),
    ).fetchall()
    return {row["digest"]: dict(row) for row in rows}


def group_attachments(conn: sqlite3.Connection, group_id: int) -> list[str] | None:
    """The digests a group declared, in order, or None for a pre-K group.

    THE TWO NULLS, and this is the check site the migration comment
    points at. None means the group predates attachments entirely: it
    was created by a build with no attachment concept, so it declared
    nothing and could not have. An empty list would mean a post-K group
    that declared no attachment, which is a different fact about a
    different question.

    Callers must not collapse them. Entry-time enforcement treats None
    as "this group cannot speak to attachments" and refuses a member
    that brings one, rather than reading the silence as permission.
    """
    row = conn.execute(
        "SELECT attachments_json FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    if row is None or row["attachments_json"] is None:
        return None
    decoded = json.loads(row["attachments_json"])
    assert isinstance(decoded, list)
    return [str(digest) for digest in decoded]


def group_attachments_mode(conn: sqlite3.Connection, group_id: int) -> str | None:
    """How this group's documents reach its models, or None for none.

    None means the group declared no attachment, whether because it
    predates the feature or because it simply has none. The two collapse
    here on purpose and it costs nothing: a mode is a statement about
    documents, and with no documents there is nothing for it to be a
    statement about, which is why the boundary refuses a mode declared
    without one.
    """
    row = conn.execute(
        "SELECT attachments_mode FROM groups WHERE id = ?", (group_id,)
    ).fetchone()
    return str(row["attachments_mode"]) if row and row["attachments_mode"] else None


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


def group_manifest(conn: sqlite3.Connection, group_id: int) -> dict[str, Any]:
    """What a group declared about itself before any call was made.

    The prompt, the ordered lineup and the budget tier, returned as
    declared: each is None when the group never declared it. Read by the
    entry-time checks and by the detail view, so there is one place that
    knows how a declaration is stored.

    Every value here is either the affirmative declaration or None, and
    the two NULL meanings in this file are NOT the same, which is why they
    are read through different functions. group_params treats NULL as the
    affirmative fact that no control was set, because controls arrived with
    the concept and every pre-H group predates it. Here NULL means unknown:
    models_json and budget exist on groups created before anything recorded
    them, so a legacy group is silent about its lineup rather than
    declaring it empty. A caller must therefore skip enforcement on None
    and enforce on a value, and the check sites say so at the point of
    decision.
    """
    row = conn.execute(
        "SELECT prompt_text, models_json, budget FROM groups WHERE id = ?",
        (group_id,),
    ).fetchone()
    if row is None:
        return {"prompt": None, "models": None, "budget": None}
    models = None
    raw = row["models_json"]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            decoded = None
        # Repair on read: a lineup that is not a list of strings cannot be
        # compared against a request, and enforcing against a broken
        # declaration would refuse correct work. Unknown, not empty.
        if isinstance(decoded, list) and all(isinstance(m, str) for m in decoded):
            models = decoded
    return {
        "prompt": as_text(row["prompt_text"]),
        "models": models,
        "budget": as_text(row["budget"]),
    }


def run_renditions(
    conn: sqlite3.Connection, run_id: int
) -> list[dict[str, Any]] | None:
    """What this run pinned, or None on a pre-K.1 run.

    THE SAME DECODER group_renditions uses, and since K.3 that is
    literally the same function rather than the same discipline written
    twice; see _decoded_renditions for why four readers of one column
    share one shape check.
    """
    row = conn.execute(
        "SELECT renditions_json FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    return None if row is None else _decoded_renditions(row["renditions_json"])


def save_run(
    conn: sqlite3.Connection,
    prompt_text: str,
    results: list[dict[str, Any]],
    prompt_id: int | None = None,
    group_id: int | None = None,
    provenance: dict[str, Any] | None = None,
    renditions: list[dict[str, Any]] | None = None,
) -> int:
    """Insert a run and its results atomically. Returns the run id.

    One transaction so a failing result insert cannot leave a run row
    with missing or partial results in the history.

    provenance carries the run-level facts the caller knows and the store
    does not: which build was running, how old the price catalog was,
    which bytes that catalog was, and which data-handling policy the
    payloads declared. Optional, and every key optional within it, so a
    caller that knows none of it still writes a valid row with those
    columns None.

    renditions is what this run actually sent, pinned. On a grouped
    run it repeats the group's pin, which is redundant and deliberately
    so: the group is where a comparison's declaration lives, and the run
    is where an UNGROUPED comparison's does. Recording it on both means
    one reader answers both cases instead of two readers agreeing to
    disagree about which table to consult.

    sort_keys for the reason create_group has it: a list of objects has
    more than one spelling and two identical pins must produce identical
    column bytes.

    Every key here must appear in the INSERT below. A column that is
    declared, migrated and read back but never written reads as an
    absence, and an absence is a claim: it says this run had no such fact.
    catalog_digest shipped that way for exactly one commit.
    """
    prov = provenance or {}
    with conn:
        cur = conn.execute(
            """INSERT INTO runs
               (prompt_id, group_id, prompt_text, created_at,
                app_sha, catalog_snapshot_at, data_policy, catalog_digest,
                renditions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prompt_id,
                group_id,
                prompt_text,
                _now(),
                prov.get("app_sha"),
                prov.get("catalog_snapshot_at"),
                prov.get("data_policy"),
                prov.get("catalog_digest"),
                json.dumps(renditions, sort_keys=True) if renditions else None,
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
                provider, quantization, native_finish_reason,
                upstream_inference_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?)""",
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
                    # Verbatim as received, not through as_money: nothing
                    # here computes with it and a float round trip would
                    # round money on the way in. See _as_upstream_cost.
                    as_text(r.get("upstream_inference_cost_usd")),
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
    # The documents each selected group declared, digests only and read in
    # the same pass as its controls rather than one query per row: a page
    # of a hundred comparisons must stay one query, which is the rule the
    # rest of this function already follows.
    group_docs: dict[int, list[str] | None] = {}
    group_doc_mode: dict[int, str | None] = {}
    group_pins: dict[int, list[dict[str, Any]] | None] = {}
    if group_ids:
        for row in conn.execute(
            f"SELECT id, created_at, params_json, attachments_json,"
            f" attachments_mode, renditions_json"
            f" FROM groups WHERE id IN ({marks(group_ids)})",
            group_ids,
        ):
            group_created[row["id"]] = row["created_at"]
            group_controls[row["id"]] = _decoded_params(row["params_json"])
            # Decoded through the same repair-on-read discipline
            # group_attachments uses, and NOT by calling it: that would be
            # one query per group, which is the N+1 this pass exists to
            # avoid. The two must agree, so the shape check is the same.
            raw_docs = row["attachments_json"]
            docs: list[str] | None = None
            if isinstance(raw_docs, str):
                try:
                    decoded_docs = json.loads(raw_docs)
                except (TypeError, ValueError):
                    decoded_docs = None
                if isinstance(decoded_docs, list):
                    docs = [str(d) for d in decoded_docs]
            group_docs[row["id"]] = docs
            group_doc_mode[row["id"]] = as_text(row["attachments_mode"])
            # And the PIN, in the same pass and for the same reason the
            # digests are here: the history list draws a chip per
            # document and a chip describes a READING. Decoded through
            # the shared shape check rather than by calling
            # group_renditions, which would be one query per row.
            group_pins[row["id"]] = _decoded_renditions(row["renditions_json"])

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
                    # Declared documents, digests in declaration order.
                    # None for a pre-K group, the same two NULLs
                    # group_attachments documents; a list entry is never
                    # emitted for a lone run because an ungrouped run has
                    # no declaration to read and its payload records a
                    # digest reference rather than a digest list.
                    "attachments": group_docs.get(key),
                    "attachments_mode": group_doc_mode.get(key),
                    # The pinned readings, so the list's chips describe
                    # what the comparison chose rather than the digest's
                    # base row. None for the two pre-K.1 eras.
                    "renditions": group_pins.get(key),
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
        "upstream_inference_cost_usd",
    ):
        row[field] = as_text(row[field])
    # position orders the replay, so junk in it must not reorder history;
    # a non-integer degrades to None and falls back to insertion order.
    row["position"] = as_token_count(row["position"])
    return row


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    run = conn.execute(
        """SELECT id, prompt_id, prompt_text, created_at,
                  app_sha, catalog_snapshot_at, data_policy, catalog_digest
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
    # id joins the projection for Phase I: a score row points at a result
    # by id, so scoring and the export need it here rather than issuing a
    # second query for something this one already read. The API's
    # ModelResult does not declare it, so it stays out of every response
    # body; it exists for callers inside the process.
    results = conn.execute(
        """SELECT id, model, response_text, latency_ms, prompt_tokens,
                  completion_tokens, error, cost_usd, ttft_ms, max_tokens,
                  generation_id, finish_reason, position, request_json,
                  billed_cost_usd, reasoning_tokens, cached_tokens,
                  provider, quantization, native_finish_reason,
                  upstream_inference_cost_usd
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
    # The declaration, so the detail view can show what the comparison was
    # supposed to be and not only what happened to run. Every field is None
    # on a group that predates the column, which the view must render as
    # absence rather than as a value.
    manifest = group_manifest(conn, group_id)
    run_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM runs WHERE group_id = ? ORDER BY id", (group_id,)
        )
    ]
    out = dict(row)
    params = _decoded_params(out.pop("params_json"))
    out["params"] = params or None
    out["prompt"] = manifest["prompt"]
    out["models"] = manifest["models"]
    out["budget"] = manifest["budget"]
    # The declared documents, as DIGESTS and not as rows. Resolving them
    # to names is the boundary's job, for the same reason params comes
    # back decoded but a request payload does not: this module returns
    # what the group row stores, and a join against attachments is a
    # second question the caller may not be asking.
    out["attachments"] = group_attachments(conn, group_id)
    out["attachments_mode"] = group_attachments_mode(conn, group_id)
    # The pin, so a replay and a reuse can say which READING the models
    # were given and not merely which file. None on every pre-K.1 group;
    # see group_renditions for the three eras.
    out["renditions"] = group_renditions(conn, group_id)
    out["runs"] = [get_run(conn, rid) for rid in run_ids]
    return out


def results_awaiting_reconciliation(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[dict[str, Any]]:
    """Result rows with a generation id that the audit could still improve.

    The reconcile pass's work list. Aborted rows qualify by construction:
    they carry an id (captured from the response header before any chunk)
    and never a billed cost, since no usage object ever arrived, and they
    are the rows whose billing is least knowable without asking.

    "No billed cost yet" is the money rule, not IS NULL. The two are not
    the same predicate, and the gap between them was a trap: a negative or
    non-numeric charge is NOT NULL to SQL while _repaired hands every
    reader None for it, so such a row displayed as unpriced forever AND was
    invisible to the one pass that could have fixed it. The three money
    clauses below are as_money expressed in SQL, so a value no reader will
    trust is a value this list still offers to replace. NaN needs no
    clause: SQLite has no NaN and binds one as NULL, which the first clause
    already catches.

    Money is not the only thing the endpoint knows. A stream that ends
    without a usage object can still bill correctly through a later live
    run of the same row's siblings, or arrive with a cost and no host: the
    charge is trusted and `provider` or `native_finish_reason` is NULL, and
    under a money-only predicate that row was permanently invisible to the
    only pass that could have filled them. Those two columns join the
    predicate for that reason. `quantization` deliberately does not: it is
    absent from OpenRouter's published response schema (see
    models.fetch_generation), so keying on it would park every row in this
    list forever waiting for a field that may never be sent.

    `upstream_inference_cost_usd` DELIBERATELY DOES NOT JOIN THE
    PREDICATE, for the same reason `quantization` does not, and the
    arithmetic is worth writing down because the instinct is the other
    way. The endpoint does report the figure, beside `is_byok`. But it
    "is only available for BYOK (Bring Your Own Key) requests. For all
    other requests it will be 0 or null", and NULL in that column is the
    affirmative fact "this run was not BYOK". There is no column for
    "asked, and the answer was no", so a clause on it would park EVERY
    ordinary row on this list forever: the pass would never converge, and
    `python -m bench.reconcile` would report work to do on a database
    with nothing left to fill. A work list that never empties is a work
    list nobody reads.

    What the pass does instead is write the figure on every row it
    reaches for one of the reasons above, which is every row whose
    billing story is incomplete, and that is where a missing BYOK figure
    actually lives: a stream that ended without a usage object carries
    neither a charge nor cost_details. The narrow case this does not
    reach is a BYOK run whose usage arrived with `cost` and without
    `cost_details`, leaving the row otherwise complete. That gap is real
    and it is smaller than a list that never empties.

    A row the endpoint genuinely cannot fill stays listed across passes.
    That is the honest state (the gap is real and still open), it costs one
    lookup per pass, and it ends on its own when the generation record
    expires and the endpoint starts 404ing.

    Ordered by id so a run interrupted partway through resumes in a
    predictable place, and so a --limit run walks the oldest rows first.
    """
    sql = (
        "SELECT r.id, r.run_id, r.model, r.generation_id, r.cost_usd, r.error "
        "FROM results r "
        "WHERE r.generation_id IS NOT NULL "
        "AND (r.billed_cost_usd IS NULL "
        "     OR typeof(r.billed_cost_usd) NOT IN ('real', 'integer') "
        "     OR r.billed_cost_usd < 0 "
        "     OR r.provider IS NULL "
        "     OR r.native_finish_reason IS NULL) "
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
    # The BYOK figure. fetch_generation has parsed it since I.2 and this
    # list is what decides whether anybody writes it; without the entry
    # the value was fetched on every pass and thrown away, which is a
    # helper with no call site wearing a different hat.
    "upstream_inference_cost_usd",
)


def apply_reconciliation(
    conn: sqlite3.Connection, result_id: int, record: dict[str, Any]
) -> None:
    """Write the audit columns for one result from a generation record.

    COALESCE on the three text columns so a field the generation endpoint
    does not return cannot erase one captured in-band: absence there means
    "not reported", never "known to be nothing".

    billed_cost_usd cannot use COALESCE, because a stored charge may be
    poisoned rather than absent and an audit that skipped it would leave
    the row unpriced forever. It cannot be assigned outright either, now
    that the work list also offers rows whose charge is trusted and whose
    provider or finish reason is missing: an unreported total_cost would
    erase a good number to fetch a label. So the CASE below says what is
    actually meant. A reported charge wins. An unreported one clears only a
    value no reader would trust anyway (the money rule, the same three
    clauses results_awaiting_reconciliation selects on), replacing a
    poisoned value with an honest absence, and leaves a trusted one alone.
    """
    with conn:
        conn.execute(
            """UPDATE results
               SET billed_cost_usd = CASE
                       WHEN ? IS NOT NULL THEN ?
                       WHEN billed_cost_usd IS NULL
                            OR typeof(billed_cost_usd)
                               NOT IN ('real', 'integer')
                            OR billed_cost_usd < 0
                       THEN NULL
                       ELSE billed_cost_usd
                   END,
                   provider = COALESCE(?, provider),
                   quantization = COALESCE(?, quantization),
                   native_finish_reason = COALESCE(?, native_finish_reason),
                   upstream_inference_cost_usd = COALESCE(
                       ?, upstream_inference_cost_usd
                   )
               WHERE id = ?""",
            (
                record["billed_cost_usd"],
                record["billed_cost_usd"],
                record["provider"],
                record["quantization"],
                record["native_finish_reason"],
                # COALESCE like the text columns, not the money CASE. It
                # is TEXT recorded verbatim for reconciliation against a
                # provider invoice, and an unreported figure means "not
                # reported" rather than "known to be nothing": there is
                # no poisoned-value case to clear, because nothing
                # computes with it.
                record["upstream_inference_cost_usd"],
                result_id,
            ),
        )


# ---- Phase I: experiments as the aggregate above groups, and scores.


# What an experiment's status may say. Not an enum in the schema (the
# derived-outcome rule elsewhere in this phase argues against baking
# vocabularies into columns), but a tuple here so the writers agree and
# a reader can see the whole vocabulary in one place.
#
# "created" is the row before the runner starts, which is a real state
# and not a placeholder: the manifest is written complete before any
# trial, so an experiment that never starts still has a full record of
# what it was going to be.
# "interrupted" is what a run becomes when the PROCESS ended rather than
# the run: a crash, a kill, or a shutdown that cancelled the runner
# mid-trial. It is terminal like the rest, and it is not "failed" or
# "stopped", because neither of those is true: nothing about the
# experiment went wrong and no operator asked it to end. The completed
# trials are real and the remaining ones never ran.
#
# It was written by two callers (the lifespan's stale-row sweep and the
# runner's cancellation path) before it was ever listed here, and
# set_experiment_status asserts membership: the sweep raised at startup,
# so a database holding one stale running row made the app unbootable.
# That is the whole argument for keeping the vocabulary in one tuple and
# the whole way this tuple failed to be one.
EXPERIMENT_STATUSES = (
    "created",
    "running",
    "done",
    "stopped",
    "halted_on_refusal",
    "interrupted",
    "failed",
)


def create_experiment(conn: sqlite3.Connection, spec: dict[str, Any]) -> int:
    """Write the experiment manifest. Returns the experiment id.

    Written complete before any trial runs, for the same reason the group
    row is written before any upstream call: a manifest that is assembled
    as it goes is a description of what happened, and the thing worth
    having is a declaration of what was intended, against which what
    happened can be checked. Every field here is known at creation.

    trials_total is stored rather than derived so progress has a
    denominator that cannot move. Deriving it later from the rows that
    exist would make a halted experiment look complete, since the trials
    it never ran leave nothing behind to count.
    """
    with conn:
        cur = conn.execute(
            """INSERT INTO experiments
               (name, created_at, dataset_name, dataset_digest, lineup_json,
                budget, params_json, repeats, task_order_seed, estimand_mode,
                primary_metric, quantizations_json, provider_pins_json,
                halt_on_refusal, status,
                status_detail, app_sha, catalog_digest, data_policy,
                tasks_total, trials_total, trials_done, trials_refused,
                trials_failed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, 0, 0, 0)""",
            (
                spec["name"],
                _now(),
                spec["dataset_name"],
                spec["dataset_digest"],
                json.dumps(spec["lineup"]),
                spec["budget"],
                # Same rule as create_group: an empty controls mapping is
                # NULL, not "{}", so absence has exactly one spelling.
                json.dumps(spec["params"], sort_keys=True)
                if spec.get("params")
                else None,
                spec["repeats"],
                spec.get("task_order_seed"),
                spec["estimand_mode"],
                spec.get("primary_metric"),
                json.dumps(spec["quantizations"])
                if spec.get("quantizations")
                else None,
                json.dumps(spec["provider_pins"], sort_keys=True)
                if spec.get("provider_pins")
                else None,
                1 if spec["halt_on_refusal"] else 0,
                "created",
                None,
                spec.get("app_sha"),
                spec.get("catalog_digest"),
                spec.get("data_policy"),
                spec["tasks_total"],
                spec["trials_total"],
            ),
        )
    assert cur.lastrowid is not None
    return cur.lastrowid


def _experiment_row(row: sqlite3.Row) -> dict[str, Any]:
    """One experiment row decoded, with the JSON columns already parsed.

    Decoding here rather than at every reader is the same argument
    get_group makes: this module is the one place that knows how a lineup
    or a pin map is serialized.
    """
    out = dict(row)
    out["lineup"] = json.loads(out.pop("lineup_json"))
    out["params"] = _decoded_params(out.pop("params_json")) or None
    pins = out.pop("provider_pins_json")
    out["provider_pins"] = _decoded_params(pins) or None
    quant = out.pop("quantizations_json")
    out["quantizations"] = json.loads(quant) if quant else None
    # Stored 0/1 because SQLite has no boolean; handed back as bool so no
    # reader has to remember which spelling this column uses.
    out["halt_on_refusal"] = bool(out["halt_on_refusal"])
    return out


def get_experiment(
    conn: sqlite3.Connection, experiment_id: int
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
    ).fetchone()
    return _experiment_row(row) if row is not None else None


def list_experiments(
    conn: sqlite3.Connection, limit: int = 100
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_experiment_row(r) for r in rows]


def set_experiment_status(
    conn: sqlite3.Connection,
    experiment_id: int,
    status: str,
    detail: str | None = None,
) -> None:
    """Move an experiment to a terminal or running state.

    detail is where a halt says why in words the reader gets to see. A
    status of halted_on_refusal with no detail would be true and useless.
    """
    assert status in EXPERIMENT_STATUSES, status
    with conn:
        conn.execute(
            "UPDATE experiments SET status = ?, status_detail = ? WHERE id = ?",
            (status, detail, experiment_id),
        )


def bump_experiment_counters(
    conn: sqlite3.Connection,
    experiment_id: int,
    done: int = 0,
    refused: int = 0,
    failed: int = 0,
) -> None:
    """Add to the progress counters.

    Relative rather than absolute so a counter update cannot lose a
    concurrent one, and so the runner never has to hold a count in memory
    that a restart would forget. These are for progress display only: the
    report derives every number it publishes from the result rows, so a
    counter that drifted could mislead a watcher but never a conclusion.
    """
    with conn:
        conn.execute(
            """UPDATE experiments
               SET trials_done = trials_done + ?,
                   trials_refused = trials_refused + ?,
                   trials_failed = trials_failed + ?
               WHERE id = ?""",
            (done, refused, failed, experiment_id),
        )


def experiment_groups(
    conn: sqlite3.Connection, experiment_id: int
) -> list[dict[str, Any]]:
    """Every group belonging to an experiment, in trial order.

    Ordered by the cell coordinates rather than by id so the sequence is
    the experiment's own structure and not the order the runner happened
    to get to. An export or a report reading this gets the same order
    every time, which is what makes a digest over it stable.
    """
    rows = conn.execute(
        """SELECT id, created_at, prompt_text, models_json, budget,
                  task_id, repeat_index, rotation_index,
                  attachments_json, attachments_mode, renditions_json
           FROM groups WHERE experiment_id = ?
           ORDER BY task_id, repeat_index, id""",
        (experiment_id,),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        raw = item.pop("models_json")
        item["models"] = json.loads(raw) if isinstance(raw, str) else None
        # The declared documents ride along, because the export's trial
        # lines have to state them and this is the one query the export
        # runs per experiment. Digests only: the bytes are never in an
        # export, and the row's decoding follows group_attachments so a
        # reader of either gets the same answer.
        raw_docs = item.pop("attachments_json")
        item["attachments"] = (
            json.loads(raw_docs) if isinstance(raw_docs, str) else None
        )
        # And the PIN, which the export needs and which this query did
        # not select. Digests say which bytes; only the rendition says
        # which reading, and an artifact that named the first without the
        # second cannot distinguish two experiments over one file read
        # two ways. Decoded through the same repair-on-read discipline
        # group_renditions applies, and NOT by calling it, which would be
        # one query per group: the two must agree, so the shape check is
        # the same one.
        item["renditions"] = _decoded_renditions(item.pop("renditions_json"))
        out.append(item)
    return out


def add_score(conn: sqlite3.Connection, result_id: int, record: dict[str, Any]) -> int:
    """Insert one score row. Never updates an existing one.

    Re-scoring appends. A scoring pass that overwrote would destroy the
    evidence that the score changed, and "the judge said 0.5 last week and
    1.0 today" is exactly the fact someone auditing a claim needs. The
    report picks the latest per (result, scorer, judge_model); the older
    rows stay as the audit trail.

    Every external number goes through the field-type functions on the way
    in, judge output included: a judge is an upstream service like any
    other, and a verdict carrying "score": "high" must land as None rather
    than as a string in a REAL column.
    """
    with conn:
        cur = conn.execute(
            """INSERT INTO scores
               (result_id, scorer, score, passed, detail, judge_model,
                judge_generation_id, judge_billed_cost_usd, blind,
                self_judged, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result_id,
                record["scorer"],
                as_metric(record.get("score")),
                None if record.get("passed") is None else int(bool(record["passed"])),
                as_text(record.get("detail")),
                as_text(record.get("judge_model")),
                as_text(record.get("judge_generation_id")),
                as_money(record.get("judge_billed_cost_usd")),
                None if record.get("blind") is None else int(bool(record["blind"])),
                None
                if record.get("self_judged") is None
                else int(bool(record["self_judged"])),
                _now(),
            ),
        )
    assert cur.lastrowid is not None
    return cur.lastrowid


def scores_for_results(
    conn: sqlite3.Connection, result_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    """Every score row for the given results, keyed by result id.

    One query for the whole set rather than one per result: a report over
    a 40-trial experiment would otherwise issue 40 queries to build one
    table. Ordered by id so "latest per key" is the last one seen, which
    is the rule the report applies.
    """
    if not result_ids:
        return {}
    marks = ",".join("?" for _ in result_ids)
    rows = conn.execute(
        f"SELECT * FROM scores WHERE result_id IN ({marks}) ORDER BY id",
        tuple(result_ids),
    ).fetchall()
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        # The three flag columns are stored 0/1/NULL; hand back bools and
        # None so no reader has to know that.
        for flag in ("passed", "blind", "self_judged"):
            if item[flag] is not None:
                item[flag] = bool(item[flag])
        # Repair on read, like _repaired does for results: a score column
        # written before this rule, or by a future writer that skipped
        # add_score, must still read back as a number or as nothing.
        item["score"] = as_metric(item["score"])
        item["judge_billed_cost_usd"] = as_money(item["judge_billed_cost_usd"])
        out.setdefault(item["result_id"], []).append(item)
    return out
