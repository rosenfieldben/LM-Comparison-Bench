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
    """WINDOW: parse_dataset's returned task list, compared whole.

    WHOLE-DICT EQUALITY ON PURPOSE. A key added to the parsed shape
    without anybody deciding to add it fails here, which is the only
    reason this assertion is written as an equality rather than a set of
    field checks. Phase M's attachments key is the first thing it caught.

    RULE ONE AT THE DATASET LAYER. A task that declares no documents
    parses to exactly what it parsed to before attachments existed, with
    the new field explicitly absent rather than defaulted to an empty
    list. None and [] are different claims: none is "this task is not
    about a document", and the loader refuses [] on those grounds.
    """
    out = parse_dataset(dataset(line(id="t1", prompt="hi")))

    assert out["tasks"] == [
        {
            "id": "t1",
            "prompt": "hi",
            "system": None,
            "reference": None,
            "rubric": None,
            "scorer": None,
            "attachments": None,
        }
    ]


# ---- Phase M: a task may cite documents. Two spellings, one meaning,
# ---- and every way of getting either wrong.

DIGEST = "a" * 64
FULL_PIN = {
    "digest": DIGEST,
    "extractor": "pypdf",
    "extractor_version": "5.1.0",
    "kind": "document",
}


def test_review_repro_a_task_may_cite_a_digest_or_a_full_rendition():
    """WINDOW: the parsed attachments of two tasks, one citing content
    and one citing a reading.

    TWO DECLARATIONS, TOLD APART BY ONE KEY. A bare digest says what the
    task is about and leaves the reading to experiment creation; a full
    pin says both, and creation then honors it verbatim or refuses.
    Downstream needs to tell them apart, and the shape it uses is
    whether "extractor" is present, so that is what this asserts rather
    than a flag the parser could have invented.

    NORMALIZED, NOT REWRITTEN. The digest-only entry becomes a one-key
    dict rather than a pin with three Nones: a partial pin is a
    different thing from a deferred reading, and spelling them the same
    would make a reader downstream guess which they had.
    """
    out = parse_dataset(
        dataset(
            line(id="t1", prompt="a", attachments=[DIGEST]),
            line(id="t2", prompt="b", attachments=[FULL_PIN]),
        )
    )

    assert out["tasks"][0]["attachments"] == [{"digest": DIGEST}]
    assert "extractor" not in out["tasks"][0]["attachments"][0]
    assert out["tasks"][1]["attachments"] == [FULL_PIN]


def test_the_pin_shape_is_one_shape_in_three_files():
    """WINDOW: the four field names, as each of the three modules that
    must agree about them spells them.

    THE COMMENT BESIDE PIN_FIELDS PROMISES THIS TEST, so the test has to
    exist or the comment is a claim nobody checks. bench.datasets cannot
    import the API boundary (which imports it), so a pin's shape is
    written out twice on purpose; what stops the two from drifting is
    this assertion rather than an import.

    THE THIRD SPELLING is row_rendition, which builds a pin from an
    attachments row and is what a bare digest resolves to. If it grew a
    fifth key, a dataset could not declare what creation would freeze.
    """
    from bench.datasets import PIN_FIELDS, PIN_KINDS
    from bench.main import RenditionPin, row_rendition

    assert set(PIN_FIELDS) == set(RenditionPin.model_fields)
    built = row_rendition(
        {"digest": DIGEST, "extractor": "pypdf", "extractor_version": "5.1.0"}
    )
    assert set(built) == set(PIN_FIELDS)
    # And the two kinds are the two the boundary accepts, so a dataset
    # cannot name a third that the API would refuse one door later.
    assert set(PIN_KINDS) == set(RenditionPin.model_fields["kind"].annotation.__args__)


def test_a_task_without_attachments_parses_exactly_as_before():
    """WINDOW: the attachments field of a task that declares none.

    RULE ONE, and the reason it is asserted as None rather than as
    falsy: an empty list would also be falsy and would be a DIFFERENT
    claim, one the loader refuses on its own. Absence gets one spelling.
    """
    out = parse_dataset(dataset(line(id="t1", prompt="a")))

    assert out["tasks"][0]["attachments"] is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ([], "is empty"),
        ([DIGEST] * 5, "limit is 4"),
        (["nope"], "not a sha256 digest"),
        ([{"digest": DIGEST, "extractor": "pypdf"}], "missing extractor_version, kind"),
        ([{**FULL_PIN, "extra": 1}], "unknown keys: extra"),
        ([{**FULL_PIN, "kind": "spreadsheet"}], "not one of document, image"),
        ([{**FULL_PIN, "digest": "short"}], "not a sha256 digest"),
        ([{**FULL_PIN, "extractor": ""}], "must not be empty"),
        (DIGEST, "must be a list, got str"),
        ([42], "must be a digest string or a rendition object, got int"),
    ],
    ids=[
        "an empty list is a second spelling of absence",
        "over the per-task limit",
        "a digest that is not one",
        "a partial pin is refused, not completed",
        "an unknown key inside a pin",
        "a kind the boundary would not accept",
        "a malformed digest inside a full pin",
        "an empty extractor",
        "a bare string where a list belongs",
        "a number where a document belongs",
    ],
)
def test_every_malformed_attachment_names_the_task_and_the_defect(value, expected):
    """WINDOW: the DatasetError message for one malformed declaration.

    THE TASK ID IS IN EVERY MESSAGE, which no other field in this file
    does. A person fixes a bad reference by reading the line; they fix a
    bad digest by finding that file and uploading it, and the id is what
    they search their notes for. The line number stays too, because the
    two answer different questions.

    A PARTIAL PIN IS REFUSED RATHER THAN COMPLETED, which is the row
    worth reading twice. Filling the missing fields from the store would
    make the dataset's declaration depend on when it was read, which is
    the exact thing pinning exists to prevent.
    """
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", attachments=value)))

    message = str(exc.value)
    assert expected in message, message
    assert "task 't1'" in message, message
    assert "line 1" in message, message


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

    assert "requires a non-empty reference" in str(exc.value)


def test_a_judge_scorer_without_a_rubric_is_refused():
    """The rubric IS the scoring instruction. A judge asked to score
    against nothing returns an opinion about something nobody specified."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", scorer={"kind": "judge"})))

    assert "requires a non-empty rubric" in str(exc.value)


def test_review_repro_an_empty_reference_is_refused_like_a_missing_one():
    """Empty is worse than missing here, because it scores rather than
    failing.

    Every string contains the empty string, so a `contains` task with
    reference "" gives every model on every repeat a perfect 1.0. Nothing
    downstream can tell that number from a real one: the trial completed,
    the scorer ran, the score is in range, and the coverage counters say
    it was scored. The only place the meaninglessness is visible is the
    dataset file, so the refusal belongs there.
    """
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(id="t1", prompt="a", reference="", scorer={"kind": "contains"})
            )
        )

    assert "requires a non-empty reference" in str(exc.value)


def test_an_empty_rubric_is_refused_like_a_missing_one():
    """A blank rubric is not a rubric, and omitting the key is a typo
    away from writing an empty one."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(line(id="t1", prompt="a", rubric="", scorer={"kind": "judge"}))
        )

    assert "requires a non-empty rubric" in str(exc.value)


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


# ---- Phase M4: the README's worked example is a contract, not prose.


def test_the_readme_dataset_example_is_one_the_loader_accepts():
    """WINDOW: the fenced JSONL block in the README's "Documents on a
    task" section, parsed by the loader itself.

    A WORKED EXAMPLE IS A PROMISE. Someone reading it will paste it,
    change the digest, and expect the bench to take it; an example the
    loader refuses is worse than none, because the reader spends their
    debugging on the documentation rather than on their file. This reads
    the block out of the README rather than restating it, so the two
    cannot drift.

    The block is found by its content rather than by a line number: a
    test anchored to an offset in a two-thousand-line document is a test
    that breaks on every unrelated edit.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    blocks = [
        block
        for block in text.split("```")
        if block.startswith("json\n") and '"clause-1"' in block
    ]
    assert len(blocks) == 1, "the README's attachment example moved or was renamed"
    body = blocks[0][len("json\n") :]

    parsed = parse_dataset(body.encode("utf-8"), name="readme.jsonl")

    assert [task["id"] for task in parsed["tasks"]] == ["clause-1", "clause-2"]
    # TWO SPELLINGS, TWO DECLARATIONS, both surviving the loader as
    # written. A bare digest normalizes to a digest and nothing else,
    # which is what "resolve this at creation" looks like once parsed; a
    # four-part object keeps all four, which is what "honor exactly this
    # reading or refuse" looks like. A loader that completed the first
    # into the second would turn every bare citation into a pin nobody
    # wrote.
    digest = "6ff1c0a0f8b34d2e5c7190ab3d4e6f8172533c9be0a4d61f8c2b7e390d5a4c18"
    assert parsed["tasks"][0]["attachments"] == [{"digest": digest}]
    assert parsed["tasks"][1]["attachments"] == [
        {
            "digest": digest,
            "extractor": "pypdf",
            "extractor_version": "6.15.0",
            "kind": "document",
        }
    ]


def test_the_readme_names_the_extractors_the_bench_actually_records():
    """WINDOW: the sentence under the example that lists extractor names,
    against bench.extract's own registry.

    THE PIN IS HONORED VERBATIM OR REFUSED, so an extractor name in the
    documentation that the bench never writes is a name that produces a
    refusal at creation for a reader who followed the instructions. The
    first draft of that sentence said "pdf", which is the suffix and not
    the extractor.
    """
    from bench.extract import _READERS, _VERSIONS

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    named = {name for _reader, name in _READERS.values()}
    assert named == set(_VERSIONS), "the registry disagrees with itself"
    for extractor in named:
        assert f"`{extractor}`" in text, extractor
    # And the one an image gets, which has no suffix entry because it is
    # not parsed at all.
    assert "`none`" in text


# ---- Thirteenth review panel, F: the anchor that let a newline
# ---- through.


def test_review_repro_a_digest_with_a_trailing_newline_is_refused_by_the_loader():
    """WINDOW: parse_dataset over a task citing a 64-hex digest with a
    newline glued to the end of it.

    MEASURED BEFORE THE FIX, at e5a0ae5: the loader ACCEPTED it, because
    Python's `$` also matches immediately before a trailing newline. The
    citation then travelled to the API boundary and was refused there by
    RenditionPin's min_length/max_length of 64, which is exactly the
    deferral the comment above DIGEST_PATTERN says the check exists to
    prevent: "should be refused while the author is still looking at the
    file, not at experiment creation with a lookup miss that reads like
    a missing upload."

    The clean digest is asserted alongside, so the fix is a narrowing
    rather than a break.
    """
    clean = "a" * 64
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", attachments=[clean + "\n"])))

    assert "line 1" in str(exc.value)
    assert "task 't1'" in str(exc.value)
    parsed = parse_dataset(dataset(line(id="t1", prompt="a", attachments=[clean])))
    assert parsed["tasks"][0]["attachments"] == [{"digest": clean}]


def test_the_same_anchor_holds_inside_a_full_pin():
    """WINDOW: the four-part spelling, whose digest goes through the same
    pattern.

    Two spellings, one check. A narrowing applied to one of them would
    leave the other admitting what the first refuses, which is the
    one-door shape this repository keeps finding.
    """
    clean = "b" * 64
    pin = {
        "digest": clean + "\n",
        "extractor": "text",
        "extractor_version": "1",
        "kind": "document",
    }
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a", attachments=[pin])))

    assert "task 't1'" in str(exc.value)
