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
