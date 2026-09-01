-- The schema as it stood at 7cbe668, the commit before Phase L's
-- store change (L1 touched no table).
-- EXTRACTED FROM GIT, NEVER TRANSCRIBED: a hand-copied era snapshot
-- is a snapshot of what somebody believed the schema was, and a
-- migration proof built on one proves the belief. Regenerate with
--   git show 7cbe668:bench/store.py
-- and take the SCHEMA string verbatim.
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
    extracted_chars INTEGER,
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
