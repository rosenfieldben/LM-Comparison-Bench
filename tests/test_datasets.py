"""Dataset loading: the file is the contract, and the digest is its version.

Every malformed case here is a file a user could plausibly write, and the
assertion is always on the message as well as the failure, because the
message is the whole product of a validation error: a refusal that does
not say which line is a refusal the author cannot act on.
"""

import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.datasets import MAX_TASKS, DatasetError, parse_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def line(**fields) -> str:
    return json.dumps(fields)


def dataset(*rows: str) -> bytes:
    return ("\n".join(rows) + "\n").encode()


def test_a_minimal_task_loads():
    out = parse_dataset(dataset(line(id="t1", prompt="hi")))

    assert out["tasks"] == [
        {
            "id": "t1",
            "prompt": "hi",
            "system": None,
            "reference": None,
            "rubric": None,
            "scorer": None,
        }
    ]


def test_the_digest_is_the_sha256_of_the_bytes():
    """The dataset's version IS its content. Not a field to keep in sync,
    which is a field that can be stale; the same content-derived rule the
    asset rev and the catalog digest already follow."""
    raw = dataset(line(id="t1", prompt="hi"))

    assert parse_dataset(raw)["digest"] == hashlib.sha256(raw).hexdigest()


def test_a_semantically_identical_file_with_different_bytes_digests_differently():
    """The digest is over bytes, not over parsed meaning, and that is the
    honest choice: it can say "these two runs read the same file" and it
    deliberately does not claim to say "these two runs ran the same tasks
    modulo formatting", which would require a canonicalization nobody
    audited."""
    a = parse_dataset(dataset(line(id="t1", prompt="hi")))
    b = parse_dataset(dataset('{"prompt": "hi", "id": "t1"}'))

    assert a["tasks"] == b["tasks"]
    assert a["digest"] != b["digest"]


def test_blank_lines_are_not_errors():
    out = parse_dataset(b'{"id": "t1", "prompt": "hi"}\n\n\n')

    assert len(out["tasks"]) == 1


def test_duplicate_ids_are_refused_by_line():
    """Ids are how a score points back at a task and how a report groups
    repeats, so a duplicate would silently merge two tasks into one row."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a"), line(id="t1", prompt="b")))

    assert "line 2" in str(exc.value)
    assert "duplicate task id 't1'" in str(exc.value)


def test_a_malformed_line_names_its_line_number():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a"), "{not json"))

    assert "line 2" in str(exc.value)
    assert "not valid JSON" in str(exc.value)


def test_a_missing_prompt_is_refused():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1")))

    assert "prompt is required" in str(exc.value)


def test_an_empty_prompt_is_refused():
    """Distinct from missing: a present-but-empty prompt is a file that
    looks complete and would spend money asking every model nothing."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="")))

    assert "must not be empty" in str(exc.value)


def test_a_non_string_field_is_not_coerced():
    """42 as a reference means something the bench cannot guess, and
    guessing "42" would make a scorer compare against a value nobody
    wrote."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", reference=42)))

    assert "reference must be a string, got int" in str(exc.value)


def test_unknown_task_keys_are_refused():
    """extra="forbid" at the API boundary, the same rule here: a silently
    dropped key in a dataset is a setting the author believes is on."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", temperature=0.5)))

    assert "unknown keys: temperature" in str(exc.value)


def test_an_unknown_scorer_kind_lists_the_known_ones():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", scorer={"kind": "vibes"})))

    message = str(exc.value)
    assert "'vibes'" in message
    assert "normalized_exact" in message


def test_unknown_scorer_keys_are_refused():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1",
                    prompt="a",
                    reference="b",
                    scorer={"kind": "exact", "ignore_case": True},
                )
            )
        )

    assert "unknown keys: ignore_case" in str(exc.value)


def test_a_reference_scorer_without_a_reference_is_refused():
    """Caught at load rather than at scoring, because discovering it after
    a run has paid for every trial is discovering it in the most expensive
    place available."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", scorer={"kind": "exact"})))

    assert "requires a reference" in str(exc.value)


def test_a_judge_scorer_without_a_rubric_is_refused():
    """The rubric IS the scoring instruction. A judge asked to score
    against nothing returns an opinion about something nobody specified."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", scorer={"kind": "judge"})))

    assert "requires a rubric" in str(exc.value)


def test_an_invalid_regex_is_refused_at_load():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(line(id="t1", prompt="a", scorer={"kind": "regex", "pattern": "("}))
        )

    assert "not a valid regex" in str(exc.value)


def test_an_empty_file_is_refused():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(b"\n\n")

    assert "no tasks" in str(exc.value)


def test_invalid_utf8_is_refused_with_the_reason():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(b'{"id": "t1", "prompt": "\xff"}')

    assert "not valid UTF-8" in str(exc.value)


def test_an_oversized_prompt_is_refused_with_both_numbers():
    """The message carries the actual size and the limit, so the author
    knows how much to cut rather than only that it was too much."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="x" * 200_000)))

    message = str(exc.value)
    assert "200000 characters" in message
    assert "limit is 100000" in message


def test_too_many_tasks_is_refused():
    rows = [line(id=f"t{i}", prompt="a") for i in range(MAX_TASKS + 1)]

    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(*rows))

    assert f"more than {MAX_TASKS} tasks" in str(exc.value)


@pytest.mark.parametrize("name", ["arithmetic", "summarize"])
def test_the_committed_examples_load(name):
    """The examples ship as documentation of the format, so a change that
    breaks them breaks the documentation."""
    raw = (REPO_ROOT / "bench-datasets" / f"{name}.jsonl").read_bytes()

    out = parse_dataset(raw, name)

    assert out["tasks"]
    assert out["name"] == name
    assert len(out["digest"]) == 64


@given(
    st.lists(
        st.tuples(
            st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
            st.text(min_size=1, max_size=50),
        ),
        min_size=1,
        max_size=12,
        unique_by=lambda pair: pair[0],
    )
)
def test_any_well_formed_file_round_trips_its_ids_in_order(pairs):
    """Order is load-bearing: without a task_order_seed the runner walks
    the file in file order, so a loader that reordered would silently
    change what "file order" means."""
    raw = dataset(*(line(id=i, prompt=p) for i, p in pairs))

    out = parse_dataset(raw)

    assert [t["id"] for t in out["tasks"]] == [i for i, _ in pairs]


def test_a_judge_scorer_takes_an_optional_pass_threshold():
    """The cutoff is the rubric author's to state. 0.5 might be "covered
    half the required points" in one rubric and "wrong but polite" in
    another, so the bench never supplies a default."""
    out = parse_dataset(
        dataset(
            line(
                id="t1",
                prompt="a",
                rubric="grade it",
                scorer={"kind": "judge", "pass_threshold": 0.75},
            )
        )
    )

    assert out["tasks"][0]["scorer"] == {"kind": "judge", "pass_threshold": 0.75}


def test_a_judge_scorer_without_a_threshold_records_none():
    out = parse_dataset(
        dataset(line(id="t1", prompt="a", rubric="grade", scorer={"kind": "judge"}))
    )

    assert out["tasks"][0]["scorer"] == {"kind": "judge"}


def test_a_pass_threshold_outside_the_unit_interval_is_refused():
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1",
                    prompt="a",
                    rubric="g",
                    scorer={"kind": "judge", "pass_threshold": 5},
                )
            )
        )

    assert "outside [0, 1]" in str(exc.value)


def test_a_boolean_pass_threshold_is_refused():
    """bool is an int in Python, so the type check has to say so
    explicitly or True would become the threshold 1.0."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1",
                    prompt="a",
                    rubric="g",
                    scorer={"kind": "judge", "pass_threshold": True},
                )
            )
        )

    assert "must be a number, got bool" in str(exc.value)


def test_a_pass_threshold_on_a_deterministic_scorer_is_refused():
    """Per-kind allowed keys, not one flat set. A deterministic scorer
    already produces a pass, so a threshold there means nothing, and
    accepting it silently would be exactly the ignored-setting failure the
    unknown-key rule exists to stop."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1",
                    prompt="a",
                    reference="b",
                    scorer={"kind": "exact", "pass_threshold": 0.5},
                )
            )
        )

    assert "unknown keys: pass_threshold" in str(exc.value)


def test_a_pattern_on_a_non_regex_scorer_is_refused():
    """The same rule in the other direction: pattern means nothing to
    exact, and accepting it would accept a scoring setting that does
    nothing."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1",
                    prompt="a",
                    reference="b",
                    scorer={"kind": "exact", "pattern": "x"},
                )
            )
        )

    assert "unknown keys: pattern" in str(exc.value)
