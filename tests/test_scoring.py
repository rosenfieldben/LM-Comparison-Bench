"""Deterministic scoring and the latest-per-key rule.

Pure functions, so every case here is a value in and a value out. The
adversarial inputs are the point: a scorer is the thing that turns a
response into a number a conclusion rests on, so it has to be boring
under everything a provider might return.
"""

import time
from pathlib import Path

from hypothesis import assume, given
from hypothesis import strategies as st

from bench import scoring
from bench.models import parse_verdict
from bench.scoring import (
    MAX_SUBJECT_CHARS,
    REGEX_TIMEOUT_DETAIL,
    judged_pass,
    latest_per_key,
    score_response,
)


def spec(kind, **extra):
    return {"kind": kind, **extra}


def test_exact_is_exact_apart_from_surrounding_whitespace():
    """Trailing newlines are a formatting artifact of chat completions,
    not a wrong answer, so they are stripped. Nothing else is."""
    assert score_response(spec("exact"), "42", "42")["passed"] is True
    assert score_response(spec("exact"), "42", " 42\n")["passed"] is True
    assert score_response(spec("exact"), "42", "The answer is 42")["passed"] is False
    assert score_response(spec("exact"), "42", "4 2")["passed"] is False


def test_normalized_exact_folds_case_and_collapses_whitespace():
    assert score_response(spec("normalized_exact"), "Yes", "yes")["passed"] is True
    assert score_response(spec("normalized_exact"), "a b", "a    b")["passed"] is True
    assert score_response(spec("normalized_exact"), "a b", "ab")["passed"] is False


def test_normalization_uses_casefold_not_lower():
    """casefold handles what lower does not. A scorer that silently
    disagreed with itself across scripts would be a bug nobody looks
    for, because it only shows up on inputs nobody tests by hand."""
    assert (
        score_response(spec("normalized_exact"), "STRASSE", "straße")["passed"] is True
    )


def test_contains_is_normalized_on_both_sides():
    assert score_response(spec("contains"), "42", "The answer is 42.")["passed"] is True
    assert score_response(spec("contains"), "Paris", "paris")["passed"] is True
    assert score_response(spec("contains"), "42", "forty two")["passed"] is False


def test_regex_matches_anywhere_and_reports_a_bad_pattern():
    assert (
        score_response(spec("regex", pattern=r"\b9\b"), None, "it is 9")["passed"]
        is True
    )
    assert (
        score_response(spec("regex", pattern=r"\b9\b"), None, "it is 19")["passed"]
        is False
    )
    # The loader refuses invalid patterns, so this is the belt to that
    # brace: a row written around the loader is a scoring failure rather
    # than a crashed pass.
    bad = score_response(spec("regex", pattern="("), None, "anything")
    assert bad["passed"] is False
    assert "pattern failed at scoring time" in bad["detail"]


def test_a_trial_with_no_text_scores_zero_and_says_why():
    """A failed trial is a failed trial. Returning None instead would let
    the report drop it from the denominator, which is exactly the
    survivorship bias this phase exists to remove. The detail is what
    lets a reader tell a wrong answer from a missing one."""
    for kind in ("exact", "normalized_exact", "contains"):
        out = score_response(spec(kind), "42", None)
        assert out == {
            "score": 0.0,
            "passed": False,
            "detail": "no response text: the trial did not complete",
        }


def test_a_giant_response_is_bounded_before_the_regex_sees_it():
    """Python's re has no timeout, so the defense is at the edges: the
    loader bounds the pattern and this bounds the subject."""
    subject = ("a" * MAX_SUBJECT_CHARS) + "needle"

    out = score_response(spec("regex", pattern="needle"), None, subject)

    # The needle is past the bound, so it is genuinely not found: the
    # truncation is visible in the answer rather than pretended away.
    assert out["passed"] is False


def test_every_deterministic_score_is_in_the_unit_interval():
    cases = [
        (spec("exact"), "42", "42"),
        (spec("exact"), "42", "no"),
        (spec("contains"), "42", "42!"),
        (spec("regex", pattern="x"), None, "x"),
    ]
    for s, ref, resp in cases:
        assert 0.0 <= score_response(s, ref, resp)["score"] <= 1.0


@given(
    reference=st.text(min_size=0, max_size=40),
    response=st.text(min_size=0, max_size=80),
)
def test_score_and_passed_never_disagree(reference, response):
    """The two fields are one fact in two shapes, and a report that read
    one while a chart read the other would show a contradiction."""
    for kind in ("exact", "normalized_exact", "contains"):
        out = score_response(spec(kind), reference, response)
        assert out["passed"] is (out["score"] == 1.0)


@given(response=st.text(max_size=200))
def test_a_reference_always_matches_itself_under_contains(response):
    """The property holds for anything that IS an answer.

    NARROWED, and the narrowing is the point rather than an escape. A
    whitespace-only response is no longer scored at all: it returns the
    no-response verdict, because a blank card is a trial that did not
    complete rather than a model that answered badly, and the two need
    different detail lines. Hypothesis generates whitespace strings, so
    the unrestricted property was asserting that a non-answer matches
    itself, which is a sentence with no meaning.

    The blank case has its own assertion below rather than being
    silently excluded here.
    """
    assume(response.strip())
    assert score_response(spec("contains"), response, response)["passed"] is True


@given(blank=st.text(alphabet=" \t\n\r", max_size=20))
def test_a_blank_response_is_absent_rather_than_wrong(blank):
    """The other half of the narrowing above, over every shape of blank.

    A trial that produced nothing visible must be recorded as not having
    completed, not as having answered incorrectly. The DETAIL is what
    carries that distinction; both branches score 0.0, so a test
    asserting only the score would pass against the defect.
    """
    out = score_response(spec("contains"), "the reference", blank)
    assert out["passed"] is False
    assert out["detail"] == "no response text: the trial did not complete"


# ---- The judge's verdict parser. Never guesses a number.


def test_a_clean_verdict_parses():
    out = parse_verdict('{"score": 0.5, "reason": "half credit"}')

    assert out == {"score": 0.5, "reason": "half credit"}


def test_a_fenced_verdict_is_recovered():
    """Not guessing: it is the same object, and models wrap JSON in fences
    often enough that refusing would discard correct verdicts over
    formatting."""
    out = parse_verdict('```json\n{"score": 1.0, "reason": "ok"}\n```')

    assert out["score"] == 1.0


def test_a_verdict_with_preamble_is_recovered():
    out = parse_verdict('Sure! Here is my grade: {"score": 0.0, "reason": "no"}')

    assert out["score"] == 0.0


def test_an_unparseable_verdict_is_an_error_not_a_number():
    """A guessed number here would enter an average and change a
    conclusion while looking exactly like a real measurement."""
    for reply in (None, "", "   ", "I think it is pretty good", "{not json"):
        out = parse_verdict(reply)
        assert "score" not in out
        assert out["error"]


def test_a_non_numeric_score_is_refused():
    out = parse_verdict('{"score": "high", "reason": "vibes"}')

    assert "score" not in out
    assert "was not a number" in out["error"]


def test_a_boolean_score_is_refused():
    """bool is an int in Python, so a total field function is what stops
    True from becoming 1.0."""
    out = parse_verdict('{"score": true, "reason": "yes"}')

    assert "score" not in out
    assert "was not a number" in out["error"]


def test_a_score_outside_the_unit_interval_is_refused():
    for value in ("1.5", "-0.2", "100"):
        out = parse_verdict(f'{{"score": {value}, "reason": "x"}}')
        assert "score" not in out
        assert "outside [0, 1]" in out["error"]


def test_a_missing_reason_is_not_an_error():
    """The score is the measurement; the reason is a courtesy. Refusing a
    verdict for a missing sentence would discard a real number."""
    out = parse_verdict('{"score": 1.0}')

    assert out == {"score": 1.0, "reason": None}


# ---- latest_per_key: newest wins, audit trail intact, ties settled.


def row(id_, scorer, score, created_at, judge_model=None):
    return {
        "id": id_,
        "scorer": scorer,
        "score": score,
        "created_at": created_at,
        "judge_model": judge_model,
    }


def test_the_newest_row_per_key_wins():
    rows = [
        row(1, "judge", 0.5, "2026-08-01T00:00:00+00:00", "j/one"),
        row(2, "judge", 1.0, "2026-08-02T00:00:00+00:00", "j/one"),
    ]

    assert [r["score"] for r in latest_per_key(rows)] == [1.0]


def test_different_scorers_and_judges_are_different_keys():
    """Two judges disagreeing is a fact worth keeping, not a conflict to
    resolve: the key includes the judge."""
    rows = [
        row(1, "exact", 1.0, "2026-08-01T00:00:00+00:00"),
        row(2, "judge", 0.5, "2026-08-01T00:00:00+00:00", "j/one"),
        row(3, "judge", 0.0, "2026-08-01T00:00:00+00:00", "j/two"),
    ]

    assert len(latest_per_key(rows)) == 3


def test_review_repro_rows_in_the_same_second_are_settled_by_id():
    """Two rows written in the same second share a timestamp at the
    resolution this store keeps, so ordering by timestamp alone would
    leave their order to whatever the query planner did that day. The
    report would then be nondeterministic in exactly the case a
    re-scoring pass produces, which is the case it exists for.
    """
    same = "2026-08-01T12:00:00+00:00"
    rows = [
        row(7, "judge", 0.25, same, "j/one"),
        row(8, "judge", 0.75, same, "j/one"),
    ]

    # Whatever order they arrive in, the later insert wins.
    assert [r["score"] for r in latest_per_key(rows)] == [0.75]
    assert [r["score"] for r in latest_per_key(list(reversed(rows)))] == [0.75]


def test_an_empty_set_is_empty():
    assert latest_per_key([]) == []


# ---- judged_pass: a cutoff only when the rubric's author named one.


def test_no_threshold_means_no_pass_verdict():
    """The bench never defaults it. A report then says score mean rather
    than manufacturing a pass rate out of a cutoff nobody chose."""
    assert judged_pass({"kind": "judge"}, 0.9) is None
    assert judged_pass({"kind": "judge"}, 0.0) is None


def test_a_declared_threshold_decides_the_pass():
    spec = {"kind": "judge", "pass_threshold": 0.75}

    assert judged_pass(spec, 0.75) is True
    assert judged_pass(spec, 1.0) is True
    assert judged_pass(spec, 0.7) is False


def test_a_scoring_failure_is_not_a_failed_answer():
    """None in, None out. An unparseable verdict has no score, and
    collapsing that into a fail would put the judge's own malfunctions
    into the model's pass rate."""
    assert judged_pass({"kind": "judge", "pass_threshold": 0.5}, None) is None


# ---- Phase I.2 J5: a regex cannot freeze the server.


def test_review_repro_a_catastrophic_pattern_times_out_instead_of_hanging():
    """The reviewer's shape, and the comment above MAX_SUBJECT_CHARS was
    wrong about it.

    That comment claimed bounding the pattern and the subject bounded a
    scorer's cost "without needing a timeout Python's re does not offer".
    Backtracking is exponential in the SUBJECT, so `(a+)+$` (six
    characters, well inside the loader's 500-character pattern limit)
    against ten thousand characters runs effectively forever. Measured
    in-process at small sizes it doubles every two characters: 0.03s at
    18, 0.11s at 20, 0.44s at 22, 1.78s at 24. Ten thousand is not a
    slow scoring pass, it is a hung server holding a database
    connection.

    The wall-clock assertion is the proof that the pattern never ran to
    completion anywhere: at n=24 it already costs 1.78s, so a run that
    returns in seconds for n=10000 can only have been terminated.
    """
    subject = "a" * 10_000 + "b"

    start = time.perf_counter()
    out = score_response({"kind": "regex", "pattern": r"(a+)+$"}, None, subject)
    elapsed = time.perf_counter() - start

    # A scoring failure, never a guessed verdict: the bench does not know
    # whether that pattern would have matched.
    assert out["score"] is None
    assert out["passed"] is None
    assert REGEX_TIMEOUT_DETAIL in out["detail"]
    # Generous enough for interpreter startup on a loaded machine, and
    # many orders of magnitude below what the match itself would cost.
    assert elapsed < 15.0


def test_an_ordinary_pattern_still_scores_normally():
    """The scope control. Sending the work to a child must not change any
    answer, only bound the time one answer may take."""
    hit = score_response({"kind": "regex", "pattern": r"\d+"}, None, "answer 42")
    miss = score_response({"kind": "regex", "pattern": r"\d+"}, None, "no digits")

    assert (hit["score"], hit["passed"]) == (1.0, True)
    assert (miss["score"], miss["passed"]) == (0.0, False)


def test_the_other_deterministic_scorers_stay_in_process():
    """Named as a test because it is a deliberate limit rather than an
    oversight: a string comparison is linear in the subject and already
    bounded by MAX_SUBJECT_CHARS, so paying interpreter startup for one
    would be cost with no bound bought."""
    source = (Path(scoring.__file__)).read_text(encoding="utf-8")
    body = source.split("def score_response")[1]
    before_regex = body.split('if kind == "regex"')[0]

    assert "run_with_deadline" not in before_regex
    assert body.count("run_with_deadline") == 1


def test_review_repro_the_deadline_is_measured_on_the_monotonic_clock():
    """The constant says "seconds", and until this commit nothing said
    seconds OF WHAT. Three readings were available and they behave
    differently under load, under an NTP step, and under a child that
    sleeps: CPU time, system-clock wall time, and monotonic wall time.
    Only the last one makes the number mean one thing.

    Asserted against the mechanism rather than the docstring, because a
    comment describing a clock nothing consults is precisely the defect
    this project keeps finding. run_with_deadline does NO time
    arithmetic of its own: it hands the deadline to
    multiprocessing.Process.join and the clock comes from there. That
    chain is join -> Popen.wait -> multiprocessing.connection.wait, and
    the last of those computes `deadline = time.monotonic() + timeout`
    and re-derives the remaining timeout from time.monotonic() on every
    pass.

    Walked here rather than trusted, so a CPython change that swapped
    the clock fails this test instead of quietly changing what the
    constant means.
    """
    import inspect
    import multiprocessing.connection
    import multiprocessing.popen_fork
    import multiprocessing.popen_spawn_posix
    from multiprocessing.process import BaseProcess

    # The seam: nothing in this module computes a deadline itself. The
    # docstring is stripped first, because it EXPLAINS the two clocks by
    # name and a scan that counted that would be asserting on prose.
    source = (Path(scoring.__file__)).read_text(encoding="utf-8")
    function = source.split("def run_with_deadline")[1].split("\ndef ")[0]
    code = function.split('"""')[2]
    assert "worker.join(deadline)" in code
    assert "time.time" not in code and "time.monotonic" not in code

    # The chain, walked. spawn is the start method run_with_deadline
    # asks for, and its Popen inherits the wait that carries the clock.
    assert (
        multiprocessing.popen_spawn_posix.Popen.wait
        is multiprocessing.popen_fork.Popen.wait
    )
    assert "self._popen.wait(timeout)" in inspect.getsource(BaseProcess.join)
    assert "wait([self.sentinel], timeout)" in inspect.getsource(
        multiprocessing.popen_fork.Popen.wait
    )
    waiter = inspect.getsource(multiprocessing.connection.wait)
    assert "deadline = time.monotonic() + timeout" in waiter
    assert "timeout = deadline - time.monotonic()" in waiter


def test_the_deadline_bounds_wall_time_not_cpu_time():
    """A child that sleeps burns no CPU and must still be cut off, which
    is the observable difference between the two readings. Measured with
    time.monotonic, the same clock the deadline is kept on."""
    start = time.monotonic()
    kind, value = scoring.run_with_deadline(r"(a+)+$", "a" * 10_000 + "b", 1.0)
    elapsed = time.monotonic() - start

    assert (kind, value) == ("timeout", None)
    # Above the deadline, because the parent waits the full second, and
    # far below what the match itself would cost: at n=24 it is already
    # 1.78s and the cost doubles every two characters.
    assert elapsed >= 1.0
    assert elapsed < 15.0
