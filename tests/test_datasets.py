"""Dataset loading: the file is the contract, and the digest is its version.

Every malformed case here is a file a user could plausibly write, and the
assertion is always on the message as well as the failure, because the
message is the whole product of a validation error: a refusal that does
not say which line is a refusal the author cannot act on.
"""

import functools
import hashlib
import json
import platform
import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.datasets import (
    MAX_TASKS,
    DatasetError,
    cites_documents,
    parse_dataset,
    scorer_kinds,
)

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
    # The fifth part, unnamed, is carried as None: the loader completes
    # the SHAPE and never the reading, so a pin nobody wrote a capture
    # into stays a pin with no capture.
    assert out["tasks"][1]["attachments"] == [{**FULL_PIN, "capture_id": None}]


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


def test_a_repeat_count_the_engine_cannot_hold_is_refused_on_its_line():
    """WINDOW: parse_dataset over a regex whose repeat count re.compile
    raises OverflowError for, rather than re.error.

    It escaped the parser as a 500, so the builder showed "HTTP 500" and
    nothing beside the row. It is the line's refusal like any other bad
    pattern."""
    with pytest.raises(OverflowError):
        re.compile("a{4294967296}")
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(id="t1", prompt="a"),
                line(
                    id="t2",
                    prompt="a",
                    scorer={"kind": "regex", "pattern": "a{4294967296}"},
                ),
            )
        )

    assert str(exc.value).startswith("line 2: scorer.pattern is not a valid regex")


def test_a_line_nested_past_the_decoders_depth_is_refused_on_its_line():
    """WINDOW: parse_dataset over a line of arrays nested DEPTH_MARGIN
    levels past the deepest json.loads decodes on the running interpreter
    (found by bisection; the margin because the limit counts the caller's
    own frames on 3.11, and this call sits shallower than the bisection's),
    inside the dataset door's byte ceiling.

    json.loads raises RecursionError there, not ValueError, and it
    escaped the parser as a 500. THE DEPTH IS THE INTERPRETER'S: 100000
    was past the decoder on macOS and on ubuntu 3.11 to 3.13, and not on
    ubuntu 3.14, where this test's pre-state failed (CI run 36093009840),
    so the depth is measured. Where the decoder takes every depth that
    fits in MAX_DATASET_BYTES, no line the door accepts can reach it, and
    the test is skipped, naming the platform and the depth measured."""
    from bench.main import MAX_DATASET_BYTES

    fits = (MAX_DATASET_BYTES - 64) // 2
    decodes = _deepest(_decodes, cap=fits)
    if decodes + DEPTH_MARGIN >= fits:
        pytest.skip(
            f"no such line on {platform.system()} Python "
            f"{platform.python_version()}: json.loads decodes lists nested "
            f"{fits} deep, the deepest a line inside the byte ceiling can be"
        )
    depth = decodes + DEPTH_MARGIN
    deep = "[" * depth + "]" * depth
    # PRE-STATE: the decoder refuses this line with RecursionError.
    with pytest.raises(RecursionError):
        json.loads(deep)
    with pytest.raises(DatasetError) as exc:
        parse_dataset(dataset(line(id="t1", prompt="a"), deep))

    assert str(exc.value).startswith("line 2: not valid JSON")


def test_incompatible_regex_flags_are_refused_on_their_line():
    """WINDOW: parse_dataset over a pattern whose global flags conflict.

    re.compile raises a plain ValueError for "(?a)(?u)", neither re.error
    nor OverflowError, and it escaped the parser as a 500."""
    with pytest.raises(ValueError):
        re.compile("(?a)(?u)")
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1", prompt="a", scorer={"kind": "regex", "pattern": "(?a)(?u)"}
                )
            )
        )

    assert str(exc.value).startswith("line 1: scorer.pattern is not a valid regex")


def test_a_pattern_of_nested_groups_is_refused_on_its_line():
    """WINDOW: parse_dataset over a pattern of MAX_PATTERN_CHARS open
    groups, the longest pattern the parser takes.

    On CPython 3.14 re.compile raises RecursionError for it rather than
    re.error, which escaped the parser as a 500; whatever the engine
    raises, the line is refused in the parser's words."""
    from bench.datasets import MAX_PATTERN_CHARS

    pattern = "(" * MAX_PATTERN_CHARS
    with pytest.raises((re.error, RecursionError)):
        re.compile(pattern)
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(id="t1", prompt="a", scorer={"kind": "regex", "pattern": pattern})
            )
        )

    assert str(exc.value).startswith("line 1: scorer.pattern is not a valid regex")


def test_a_refusal_quotes_a_large_value_in_brief():
    """WINDOW: the sentences parse_dataset writes for a scorer kind that
    is a long string, a long list and a huge integer, and for a threshold
    that is a huge integer.

    A refusal names the value it refuses, bounded in length: a 10000
    character kind quoted whole would bury the sentence, and the quoting
    is bounded by design (bench.datasets._QUOTE)."""
    cases = [
        {"kind": "k" * 10_000},
        {"kind": list(range(10_000))},
        {"kind": 10**1000},
        {"kind": "judge", "pass_threshold": 10**1000},
    ]
    for scorer in cases:
        with pytest.raises(DatasetError) as exc:
            parse_dataset(dataset(line(id="t1", prompt="a", rubric="r", scorer=scorer)))
        assert str(exc.value).startswith("line 1: scorer")
        assert len(str(exc.value)) < 200, str(exc.value)[:120]


# THE DEPTHS ARE THE INTERPRETER'S, NOT THE TEST'S. How deep json.loads
# decodes and how deep repr() walks are limits of the running Python, and
# they differ by version and operating system. Measured with bare lists:
# on macOS, 3.11 decodes 993 and prints 995, 3.12 9997 and 9996, 3.13
# 9998 and 9997, 3.14 74663 and 43553; on ubuntu, 3.12 prints every depth
# json.loads decodes (the operator's N4 pass), 3.11 refuses to decode
# 60000 (CI run 36092459473), and 3.14 decodes 100000 (CI run
# 36093009840). A proof that named one depth held only
# where that depth sat in the window (a line decodes, repr overflows), so
# these measure the running interpreter by bisection instead.
DEPTH_CAP = 1 << 18
# The parser's own frames sit above the test's, so a line that decodes
# here by a hair could fail to decode inside parse_dataset and be refused
# as JSON, never reaching the field the proof is about.
DEPTH_MARGIN = 50


def _nested(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


def _deepest(ok, cap: int = DEPTH_CAP) -> int:
    """The largest depth up to cap for which ok holds, by bisection (ok
    holds at 0 and, past its limit, at no greater depth)."""
    if ok(cap):
        return cap
    low, high = 0, cap
    while high - low > 1:
        mid = (low + high) // 2
        if ok(mid):
            low = mid
        else:
            high = mid
    return low


def _decodes(depth: int) -> bool:
    try:
        json.loads("[" * depth + "]" * depth)
    except RecursionError:
        return False
    return True


def _prints(depth: int) -> bool:
    try:
        repr(_nested(depth))
    except RecursionError:
        return False
    return True


@functools.cache
def _depth_limits() -> tuple[int, int]:
    """(deepest json.loads decodes, deepest repr() prints) here."""
    return _deepest(_decodes), _deepest(_prints)


def _deep_lines(depth: int) -> list[tuple[str, str]]:
    """The three lines whose quoted value is a list nested `depth` deep,
    each with the field its refusal names. Each line nests the value up
    to three levels deeper than the value itself."""
    deep = "[" * depth + "]" * depth
    pin = (
        '{"digest": "' + "a" * 64 + '", "extractor": "x", '
        '"extractor_version": "1", "kind": '
    )
    return [
        ('{"id": "t", "prompt": "p", "scorer": {"kind": ' + deep + "}}", "scorer kind"),
        (
            '{"id": "t", "prompt": "p", "attachments": [' + pin + deep + "}]}",
            ".kind is",
        ),
        (
            '{"id": "t", "prompt": "p", "attachments": ['
            + pin
            + '"snapshot", "capture_id": '
            + deep
            + "}]}",
            ".capture_id is",
        ),
    ]


def _refusals_are_bounded(depth: int) -> None:
    for text, field in _deep_lines(depth):
        # PRE-STATE: the line decodes, so what is refused is the field.
        json.loads(text)
        with pytest.raises(DatasetError) as exc:
            parse_dataset(dataset(line(id="t0", prompt="a"), text))
        message = str(exc.value)
        assert message.startswith("line 2: "), message[:80]
        assert field in message, message[:120]
        assert len(message) < 300, message[:120]


def test_a_value_as_deep_as_the_decoder_takes_is_quoted_in_bounded_words():
    """WINDOW: parse_dataset over lines whose scorer kind, pin kind or
    capture id is a list nested as deep as json.loads decodes on the
    running interpreter (found by bisection, less a margin for the
    parser's own frames).

    PLATFORM-INDEPENDENT. Each is refused on its line, naming the field,
    in a sentence under 300 characters: the value is quoted to a bounded
    depth (_quoted). Plain repr() of the same value is thousands of
    characters where repr() can walk it and a RecursionError where it
    cannot (PRE-STATE), and either fails here, so a parser that went back
    to repr() fails on every interpreter in the CI matrix."""
    decodes, _ = _depth_limits()
    depth = decodes - 3 - DEPTH_MARGIN
    assert depth > 500, (platform.python_version(), decodes)
    try:
        plain = repr(_nested(depth))
    except RecursionError:
        pass
    else:
        assert len(plain) > 300

    _refusals_are_bounded(depth)


def test_a_value_too_deep_to_repr_is_named_on_its_line():
    """WINDOW: parse_dataset over lines whose scorer kind, pin kind or
    capture id is a list nested deeper than repr() can walk but shallow
    enough that json.loads decodes the line, on an interpreter where such
    a depth exists.

    Each was refused, but the refusal quoted the value with repr(), which
    raised RecursionError and escaped both doors as a 500 (224d488). The
    value is now quoted to a bounded depth. THE WINDOW IS PLATFORM-BOUND:
    it exists where the decoder takes lists deeper than repr() walks
    (macOS 3.14, measured) and not where repr() walks every depth the
    decoder takes (ubuntu 3.12, and 3.11 to 3.13 on macOS once a line
    nests the value), so this runs only where the running interpreter
    has it, and is skipped with both measured depths where it does not.
    The platform-independent proof is the one above."""
    decodes, prints = _depth_limits()
    depth = decodes - 3 - DEPTH_MARGIN
    if depth <= prints:
        pytest.skip(
            f"no window on {platform.system()} Python {platform.python_version()}: "
            f"json.loads decodes lists nested {decodes} deep and repr() prints "
            f"{prints}, so no line the decoder takes holds a value repr() cannot "
            "print"
        )
    # PRE-STATE: repr() of this value overflows.
    with pytest.raises(RecursionError):
        repr(_nested(depth))

    _refusals_are_bounded(depth)


def test_a_threshold_too_large_for_a_float_is_outside_the_range():
    """WINDOW: parse_dataset over an integer pass_threshold of 400 digits.

    float() raises OverflowError for it rather than returning a float,
    and that escaped the parser as a 500; it is outside [0, 1] like any
    other number past 1."""
    with pytest.raises(DatasetError) as exc:
        parse_dataset(
            dataset(
                line(
                    id="t1",
                    prompt="a",
                    rubric="r",
                    scorer={"kind": "judge", "pass_threshold": 10**400},
                )
            )
        )

    assert str(exc.value).startswith("line 1: scorer.pass_threshold 1000")
    assert str(exc.value).endswith("is outside [0, 1]")


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
            "capture_id": None,
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


def test_a_pin_capture_is_a_positive_integer_or_refused():
    """WINDOW: the loader, the fifth part of a pin.

    A capture id is a row number and nothing else may stand in for one:
    a string, a bool (an int to isinstance, and "capture_id: true" is a
    typo), zero and a negative are all refused at the line that wrote
    them; so is one past SQLite's rowid range, which the parser stored
    and the experiment doors then crashed looking up. The last id in the
    range is a row number like any other.
    """
    from bench.datasets import DatasetError, parse_dataset

    snapshot = {
        **FULL_PIN,
        "extractor": "repo-walk",
        "extractor_version": "1",
        "kind": "snapshot",
    }
    for bad in ("7", True, 0, -3, 2**63, 10**30):
        raw = (
            json.dumps(
                {
                    "id": "t",
                    "prompt": "p",
                    "attachments": [{**snapshot, "capture_id": bad}],
                }
            )
            + "\n"
        ).encode()
        with pytest.raises(DatasetError) as caught:
            parse_dataset(raw)
        assert "capture_id" in str(caught.value) and "positive integer" in str(
            caught.value
        )
    good = (
        json.dumps(
            {"id": "t", "prompt": "p", "attachments": [{**snapshot, "capture_id": 7}]}
        )
        + "\n"
    ).encode()
    assert parse_dataset(good)["tasks"][0]["attachments"][0]["capture_id"] == 7
    last = (
        json.dumps(
            {
                "id": "t",
                "prompt": "p",
                "attachments": [{**snapshot, "capture_id": 2**63 - 1}],
            }
        )
        + "\n"
    ).encode()
    assert parse_dataset(last)["tasks"][0]["attachments"][0]["capture_id"] == 2**63 - 1


def test_the_capture_bound_is_the_api_boundarys():
    """WINDOW: MAX_CAPTURE_ID against bench.main.MAX_SQLITE_ROWID.

    The parser mirrors the bound RenditionPin puts on a capture id rather
    than importing the application; the two must not drift apart."""
    from bench import main
    from bench.datasets import MAX_CAPTURE_ID

    assert MAX_CAPTURE_ID == main.MAX_SQLITE_ROWID


# ---- Phase N1: the two summaries a stored dataset records.


def test_scorer_kinds_are_each_kind_once_sorted_and_nothing_for_no_scorer():
    """WINDOW: scorer_kinds over a parsed dataset mixing kinds, repeats of
    one kind, and a task with no scorer.

    The list a stored dataset records AND the list a primary metric is
    checked against, because enforce_primary_metric now calls this
    function; a task with no scorer adds nothing, which is what it
    declares."""
    parsed = parse_dataset(
        dataset(
            line(id="a", prompt="p", reference="r", scorer={"kind": "exact"}),
            line(id="b", prompt="p", scorer={"kind": "regex", "pattern": "x"}),
            line(id="c", prompt="p", reference="r", scorer={"kind": "exact"}),
            line(id="d", prompt="p"),
            line(id="e", prompt="p", rubric="g", scorer={"kind": "judge"}),
        )
    )

    assert scorer_kinds(parsed["tasks"]) == ["exact", "judge", "regex"]
    assert scorer_kinds(parse_dataset(dataset(line(id="x", prompt="p")))["tasks"]) == []


def test_cites_documents_is_true_exactly_when_some_task_cites_one():
    """WINDOW: cites_documents over three parsed datasets: none, one of
    two tasks, and every task.

    The fact POST /experiments refuses attachments_mode over when it is
    false, recorded so a browser can disable the control there."""
    digest = "a" * 64
    none = parse_dataset(dataset(line(id="a", prompt="p"), line(id="b", prompt="p")))
    some = parse_dataset(
        dataset(
            line(id="a", prompt="p", attachments=[digest]), line(id="b", prompt="p")
        )
    )
    every = parse_dataset(dataset(line(id="a", prompt="p", attachments=[digest])))

    assert cites_documents(none["tasks"]) is False
    assert cites_documents(some["tasks"]) is True
    assert cites_documents(every["tasks"]) is True


# ---- Lines end at "\n": the parser's line splitting.

# The characters str.splitlines breaks a line at besides "\n": a lone CR,
# \v, \f, U+001C to U+001E, U+0085, U+2028 and U+2029.
SPLITLINES_ONLY = ["\r", "\v", "\f", "\x1c", "\x1d", "\x1e"]
SPLITLINES_ONLY += [chr(0x85), chr(0x2028), chr(0x2029)]


def as_splitlines_read(raw: bytes) -> bytes:
    """The same text rejoined at the lines str.splitlines finds in it.

    Parsing this is parsing the original the way parse_dataset did
    before lines ended at "\\n" alone: the pieces hold no line break of
    either kind, so the two readings of the rejoined text agree, and they
    are the pieces the old parser read. The digest differs, since the
    bytes do; the tasks and the refusals are what it reproduces."""
    return "\n".join(raw.decode("utf-8").splitlines()).encode("utf-8")


def test_review_repro_a_prompt_holding_a_line_separator_is_one_task():
    """WINDOW: parse_dataset over one task whose prompt holds U+2028,
    U+2029 and U+0085 written raw, as JSON allows and as JSON.stringify
    writes them.

    THE DEFECT. The parser read lines with str.splitlines, which breaks
    at all three, so this valid task was cut in two and refused as an
    unterminated string; the builder escaped the three in every line it
    composed to get past it. PRE-STATE: read the old way, it refuses.
    Now it is one task, and its prompt is the one written."""
    prompt = "before" + chr(0x2028) + "middle" + chr(0x2029) + "after" + chr(0x85)
    raw = (
        json.dumps({"id": "t1", "prompt": prompt}, ensure_ascii=False) + "\n"
    ).encode()
    with pytest.raises(DatasetError) as before:
        parse_dataset(as_splitlines_read(raw))
    assert str(before.value).startswith("line 1: not valid JSON: Unterminated string")

    out = parse_dataset(raw)

    assert [task["prompt"] for task in out["tasks"]] == [prompt]


def committed_datasets():
    """Every dataset the repository ships as data or as documentation:
    the files in bench-datasets/ and the README's JSON blocks that are
    tasks (the ones with a prompt)."""
    found = {
        path.name: path.read_bytes()
        for path in sorted((REPO_ROOT / "bench-datasets").glob("*.jsonl"))
    }
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for n, block in enumerate(readme.split("```")):
        if block.startswith("json\n") and '"prompt"' in block:
            found[f"README block {n}"] = block[len("json\n") :].encode("utf-8")
    return found


def test_every_committed_dataset_parses_as_it_did_before():
    """WINDOW: parse_dataset over every committed dataset, read the new
    way and the old way (as_splitlines_read).

    Ending lines at "\\n" alone changes what the parser accepts only for a
    file that holds one of the characters splitlines alone breaks at;
    every dataset the repository ships parses to the same tasks both
    ways. PRE-STATE: the set is not empty and includes both example files
    and the README's blocks, so the loop proves something."""
    datasets = committed_datasets()
    assert {"arithmetic.jsonl", "summarize.jsonl"} <= set(datasets)
    assert sum(name.startswith("README") for name in datasets) >= 3

    for name, raw in datasets.items():
        now = parse_dataset(raw, name)["tasks"]
        before = parse_dataset(as_splitlines_read(raw), name)["tasks"]
        assert now == before, name
        assert now


def test_a_separator_the_format_does_not_have_is_refused_on_its_line():
    """WINDOW: parse_dataset over two tasks separated by each character
    str.splitlines breaks at besides "\\n".

    A DELIBERATE CHANGE OF WHAT THE PARSER ACCEPTS. Read the old way,
    each of these files was two tasks (PRE-STATE, below); now each is one
    line holding two JSON values, and it is refused on line 1. JSONL is
    newline-delimited, and a parser that also split at U+2028 to keep
    accepting these would cut the valid prompt in the test above in two.
    A person with such a file is told which line, in the parser's words,
    and the fix is a newline."""
    for mark in SPLITLINES_ONLY:
        raw = (
            json.dumps({"id": "a", "prompt": "p"})
            + mark
            + json.dumps({"id": "b", "prompt": "q"})
            + "\n"
        ).encode()
        before = parse_dataset(as_splitlines_read(raw))
        assert [task["id"] for task in before["tasks"]] == ["a", "b"], repr(mark)

        with pytest.raises(DatasetError) as exc:
            parse_dataset(raw)

        assert str(exc.value).startswith("line 1: not valid JSON"), repr(mark)


def test_a_crlf_file_reads_as_its_lf_twin():
    """WINDOW: parse_dataset over the same tasks with LF and with CRLF
    line ends, and with a blank CRLF line between them.

    The CR left at the end of each line is whitespace to the JSON decoder
    and to the blank-line check, so a file saved on Windows reads as the
    same tasks, numbered the same; the digests differ, because the bytes
    do, and the digest is of the bytes."""
    rows = [
        json.dumps({"id": "a", "prompt": "p"}),
        json.dumps({"id": "b", "prompt": "q"}),
    ]
    lf = ("\n".join(rows) + "\n").encode()
    crlf = ("\r\n\r\n".join(rows) + "\r\n").encode()

    assert parse_dataset(crlf)["tasks"] == parse_dataset(lf)["tasks"]
    assert parse_dataset(crlf)["digest"] != parse_dataset(lf)["digest"]
    bad = (rows[0] + "\r\n" + '{"id": "a", "prompt": "again"}' + "\r\n").encode()
    with pytest.raises(DatasetError) as exc:
        parse_dataset(bad)
    assert str(exc.value) == "line 2: duplicate task id 'a'"
