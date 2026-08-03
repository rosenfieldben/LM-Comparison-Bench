"""Deterministic scorers, and the shape every scorer returns.

Pure functions over a spec and a response string. No I/O, no clock, no
randomness: a deterministic scorer that could disagree with itself would
undermine the one thing it is for.

Every scorer returns the same shape, and the judge path in main.py builds
that shape too, so the store, the report and the export never branch on
which kind of scorer produced a row.
"""

import re
from typing import Any

# The subject a scorer will look at, bounded. A response is upstream text
# and can be as long as the token budget allowed, which for the extended
# tier is large; regex over an unbounded subject is where a pattern that
# looked fine on short answers becomes a stall. The dataset loader bounds
# the pattern, this bounds the subject, and between them a scorer's cost
# is bounded without needing a timeout Python's re does not offer.
#
# Truncation is at the front because scorers here look for an answer, and
# an answer that is not in the first 200k characters of a response is not
# an answer the reader was going to find either.
MAX_SUBJECT_CHARS = 200_000


def _normalized(text: str) -> str:
    """Case-folded with runs of whitespace collapsed.

    casefold rather than lower because it handles the cases lower does
    not (German sharp s, for one), and a scorer that silently disagreed
    with itself across scripts would be a bug nobody would look for.
    """
    return " ".join(text.casefold().split())


def _verdict(score: float, passed: bool, detail: str | None = None) -> dict[str, Any]:
    return {"score": score, "passed": passed, "detail": detail}


def score_response(
    spec: dict[str, Any], reference: str | None, response: str | None
) -> dict[str, Any]:
    """One deterministic score, or an honest refusal to produce one.

    A response of None is an errored or stopped trial. It scores zero and
    fails, rather than returning None: a failed trial is a failed trial,
    and excluding it would be exactly the survivorship bias this phase
    exists to remove. The detail says so, so a reader of the score row can
    tell "the model answered wrongly" from "the model did not answer".

    Score is in [0, 1] for every scorer here. The binary ones return the
    endpoints; keeping the range uniform is what lets the report average
    across scorers without knowing which is which.
    """
    kind = spec["kind"]
    if response is None:
        return _verdict(0.0, False, "no response text: the trial did not complete")
    subject = response[:MAX_SUBJECT_CHARS]

    if kind == "exact":
        assert reference is not None
        hit = subject.strip() == reference.strip()
        return _verdict(1.0 if hit else 0.0, hit)
    if kind == "normalized_exact":
        assert reference is not None
        hit = _normalized(subject) == _normalized(reference)
        return _verdict(1.0 if hit else 0.0, hit)
    if kind == "contains":
        assert reference is not None
        hit = _normalized(reference) in _normalized(subject)
        return _verdict(1.0 if hit else 0.0, hit)
    if kind == "regex":
        # Compiled per call rather than cached. The pattern is bounded by
        # the loader and the subject by this module, so compilation is
        # cheap beside the network call that produced the subject, and a
        # cache keyed on user input is a memory growth path for no gain.
        try:
            hit = re.search(spec["pattern"], subject) is not None
        except re.error as exc:
            # The loader rejects invalid patterns, so reaching here means
            # a row was written before that check or around it. Scoring
            # says so rather than crashing the pass.
            return _verdict(0.0, False, f"pattern failed at scoring time: {exc}")
        return _verdict(1.0 if hit else 0.0, hit)

    raise ValueError(f"unknown deterministic scorer kind: {kind!r}")


def latest_per_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The newest score row per (scorer, judge_model), audit trail intact.

    Re-scoring appends rather than overwrites, so a result can carry
    several rows for the same key and the report has to choose. It chooses
    the newest, which is the answer to "what does the bench say now"; the
    older rows stay in the database as the answer to "what did it say
    before", which is the question an audit asks.

    Ordered by created_at then id, and the id is not decoration. Two rows
    written in the same second have the same timestamp to the resolution
    this store keeps, so timestamp alone would leave their order to
    whatever the query planner did that day, and the report would then be
    nondeterministic in exactly the case a re-scoring pass produces. The
    id is monotonic per insert, so it settles the tie the way the writes
    actually happened.
    """
    best: dict[tuple[str, str | None], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (r["created_at"], r["id"])):
        best[(row["scorer"], row.get("judge_model"))] = row
    return list(best.values())


def judged_pass(spec: dict[str, Any], score: float | None) -> bool | None:
    """Whether a judged score counts as a pass, or None if nobody said.

    A rubric defines a graded score. Turning that into a pass needs a
    cutoff, and the only person who can name one is whoever wrote the
    rubric: 0.5 might be "covered half the required points" in one rubric
    and "wrong but politely" in another. So the threshold is optional on
    the task's scorer spec and the bench never supplies a default.
    Absent means None, and the report says "score mean" rather than
    manufacturing a pass rate out of a cutoff nobody chose.

    None in means None out: an unparseable verdict has no score, and a
    scoring failure is not a failed answer. Collapsing the two would put
    the judge's own malfunctions into the model's pass rate.
    """
    threshold = spec.get("pass_threshold")
    if threshold is None or score is None:
        return None
    return bool(score >= float(threshold))
