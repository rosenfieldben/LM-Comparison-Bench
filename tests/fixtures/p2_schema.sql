-- The schema as it stood from 2e8bd02 to b3b4e48, Phase P's P2 until
-- schema 12: SCHEMA and then SEALS, both verbatim, the second with
-- scoring_passes_sealed_v2 and judge_calls_sealed_v2, the seals schema
-- 12 retired. The two strings are byte-identical at every commit from
-- 2e8bd02 to b3b4e48; this file was taken at b3b4e48.
-- EXTRACTED FROM GIT, NEVER TRANSCRIBED: a hand-copied era snapshot
-- is a snapshot of what somebody believed the schema was, and a
-- migration proof built on one proves the belief. Regenerate with
--   git show b3b4e48:bench/store.py
-- and take the SCHEMA string and then the SEALS string verbatim. Their
-- sha256 is pinned in tests/test_store.py, since CI's checkout is too
-- shallow to ask git.
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
    task_attachments_json TEXT,
    attachments_mode TEXT,
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
    created_at TEXT NOT NULL,
    judge_call_id INTEGER REFERENCES judge_calls(id),
    pass_id INTEGER REFERENCES scoring_passes(id)
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
    extracted_chars INTEGER,
    UNIQUE (digest, extractor, extractor_version)
);
CREATE TABLE IF NOT EXISTS clones (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    ref TEXT NOT NULL,
    head_sha TEXT NOT NULL,
    root TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (url, ref)
);
CREATE TABLE IF NOT EXISTS snapshot_captures (
    id INTEGER PRIMARY KEY,
    digest TEXT NOT NULL,
    extractor TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    head TEXT,
    dirty INTEGER,
    patterns_json TEXT NOT NULL,
    excludes_json TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    clone_id INTEGER REFERENCES clones(id)
);
CREATE TABLE IF NOT EXISTS datasets (
    digest TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content BLOB NOT NULL,
    task_count INTEGER NOT NULL,
    scorers_json TEXT NOT NULL,
    cites_documents INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    response_text TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_completion_tokens INTEGER,
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
    upstream_inference_cost_usd TEXT,
    is_byok INTEGER
);
CREATE TABLE IF NOT EXISTS scoring_passes (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    judge_model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    -- NULL while the pass runs; then one of PASS_OUTCOMES in store.py.
    outcome TEXT,
    detail TEXT,
    scored INTEGER,
    failed INTEGER,
    unanswered INTEGER,
    unusable INTEGER
);
CREATE TABLE IF NOT EXISTS judge_calls (
    id INTEGER PRIMARY KEY,
    pass_id INTEGER NOT NULL REFERENCES scoring_passes(id),
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    result_id INTEGER NOT NULL REFERENCES results(id),
    judge_model TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    -- Filled only when a reply arrived, whatever the reply said.
    answered_at TEXT,
    generation_id TEXT,
    billed_cost_usd REAL,
    -- NULL until the call ends; then one of CALL_OUTCOMES in store.py.
    -- not_sent means no connection was established, so nothing left the
    -- machine. Anything after the connection was established counts as
    -- sent, because money may have moved.
    outcome TEXT,
    detail TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
);
CREATE TRIGGER IF NOT EXISTS scoring_passes_sealed_v2
BEFORE UPDATE ON scoring_passes
BEGIN
    SELECT RAISE(ABORT,
        'a scoring pass record never changes a value once written')
     WHERE NEW.id IS NOT OLD.id
        OR (OLD.experiment_id IS NOT NULL
            AND NEW.experiment_id IS NOT OLD.experiment_id)
        OR (OLD.judge_model IS NOT NULL AND NEW.judge_model IS NOT OLD.judge_model)
        OR (OLD.started_at IS NOT NULL AND NEW.started_at IS NOT OLD.started_at)
        OR (OLD.ended_at IS NOT NULL AND NEW.ended_at IS NOT OLD.ended_at)
        OR (OLD.outcome IS NOT NULL AND NEW.outcome IS NOT OLD.outcome)
        OR (OLD.detail IS NOT NULL AND NEW.detail IS NOT OLD.detail)
        OR (OLD.scored IS NOT NULL AND NEW.scored IS NOT OLD.scored)
        OR (OLD.failed IS NOT NULL AND NEW.failed IS NOT OLD.failed)
        OR (OLD.unanswered IS NOT NULL AND NEW.unanswered IS NOT OLD.unanswered)
        OR (OLD.unusable IS NOT NULL AND NEW.unusable IS NOT OLD.unusable);
    SELECT RAISE(ABORT,
        'a scoring pass ends once, and this one has already ended')
     WHERE OLD.outcome IS NOT NULL;
    SELECT RAISE(ABORT,
        'a scoring pass ends in one write, and that write names its outcome')
     WHERE NEW.outcome IS NULL;
END;
CREATE TRIGGER IF NOT EXISTS scoring_passes_never_replaced_v1
BEFORE INSERT ON scoring_passes
WHEN NEW.id IN (SELECT id FROM scoring_passes)
BEGIN
    SELECT RAISE(ABORT, 'a scoring pass record is never replaced');
END;
CREATE TRIGGER IF NOT EXISTS scoring_passes_never_deleted_v1
BEFORE DELETE ON scoring_passes
BEGIN
    SELECT RAISE(ABORT, 'a scoring pass record is never deleted');
END;
CREATE TRIGGER IF NOT EXISTS judge_calls_sealed_v2
BEFORE UPDATE ON judge_calls
BEGIN
    SELECT RAISE(ABORT,
        'a judge call record never changes a value once written')
     WHERE NEW.id IS NOT OLD.id
        OR (OLD.pass_id IS NOT NULL AND NEW.pass_id IS NOT OLD.pass_id)
        OR (OLD.experiment_id IS NOT NULL
            AND NEW.experiment_id IS NOT OLD.experiment_id)
        OR (OLD.result_id IS NOT NULL AND NEW.result_id IS NOT OLD.result_id)
        OR (OLD.judge_model IS NOT NULL AND NEW.judge_model IS NOT OLD.judge_model)
        OR (OLD.sent_at IS NOT NULL AND NEW.sent_at IS NOT OLD.sent_at)
        OR (OLD.answered_at IS NOT NULL AND NEW.answered_at IS NOT OLD.answered_at)
        OR (OLD.generation_id IS NOT NULL
            AND NEW.generation_id IS NOT OLD.generation_id)
        OR (OLD.billed_cost_usd IS NOT NULL
            AND NEW.billed_cost_usd IS NOT OLD.billed_cost_usd)
        OR (OLD.outcome IS NOT NULL AND NEW.outcome IS NOT OLD.outcome)
        OR (OLD.detail IS NOT NULL AND NEW.detail IS NOT OLD.detail)
        OR (OLD.prompt_tokens IS NOT NULL
            AND NEW.prompt_tokens IS NOT OLD.prompt_tokens)
        OR (OLD.completion_tokens IS NOT NULL
            AND NEW.completion_tokens IS NOT OLD.completion_tokens);
    SELECT RAISE(ABORT,
        'a judge call ends once, and this one has already ended')
     WHERE OLD.outcome IS NOT NULL;
    SELECT RAISE(ABORT,
        'a judge call ends in one write, and that write names its outcome')
     WHERE NEW.outcome IS NULL;
END;
CREATE TRIGGER IF NOT EXISTS judge_calls_never_replaced_v1
BEFORE INSERT ON judge_calls
WHEN NEW.id IN (SELECT id FROM judge_calls)
BEGIN
    SELECT RAISE(ABORT, 'a judge call record is never replaced');
END;
CREATE TRIGGER IF NOT EXISTS judge_calls_never_deleted_v1
BEFORE DELETE ON judge_calls
BEGIN
    SELECT RAISE(ABORT, 'a judge call record is never deleted');
END;
