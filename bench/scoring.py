"""Deterministic scorers, and the shape every scorer returns.

Pure functions over a spec and a response string. No clock and no
randomness: a deterministic scorer that could disagree with itself would
undermine the one thing it is for.

THE REGEX SCORER IS THE ONE EXCEPTION, and it is not a stylistic one. It
runs its pattern in a short-lived child process under a hard deadline,
because Python's `re` cannot be interrupted once it starts backtracking
and the only other way to bound it would be a linear-time engine, which
is a new dependency for one scorer. Everything else here stays in
process. See run_with_deadline.

Every scorer returns the same shape, and the judge path in main.py builds
that shape too, so the store, the report and the export never branch on
which kind of scorer produced a row.
"""

import multiprocessing
import re
from typing import Any

# The subject a scorer will look at, bounded. A response is upstream text
# and can be as long as the token budget allowed, which for the extended
# tier is large, and a pattern that looked fine on short answers is a
# different animal on a long one.
#
# This bound is necessary and NOT sufficient, which is what the comment
# here used to get wrong: it claimed that bounding the pattern and the
# subject bounded a scorer's cost "without needing a timeout Python's re
# does not offer". It does not. Backtracking is exponential in the
# subject, not linear, so `(a+)+$` (six characters, well inside the
# loader's 500-character pattern limit) against ten thousand characters
# runs effectively forever. The deadline below is the actual bound; this
# one only keeps the ordinary case cheap.
#
# Truncation is at the front because scorers here look for an answer, and
# an answer that is not in the first 200k characters of a response is not
# an answer the reader was going to find either.
MAX_SUBJECT_CHARS = 200_000

# How long one pattern may run before the bench stops waiting for it.
#
# A second is enormous for a regex over at most 200k characters: every
# pattern anyone would deliberately write finishes in single-digit
# milliseconds, so this never fires on honest input and no user tuning
# it is a user whose scoring got slower. What it bounds is the other
# case, where the cost is exponential and the difference between one
# second and forever is the only difference that matters.
#
# It cannot be smaller without risking a false timeout on a loaded
# machine (the child pays interpreter startup first), and making it
# larger buys nothing: a pattern that has not finished in a second
# against this subject is not going to finish.
REGEX_DEADLINE_SECONDS = 1.0

# What the row says when the deadline fires. A scoring FAILURE, never a
# verdict: the bench does not know whether that pattern would have
# matched, and guessing either way would put a number nobody computed
# into a model's mean.
REGEX_TIMEOUT_DETAIL = "regex exceeded deadline"


def _regex_search(pattern: str, subject: str, sink: Any) -> None:
    """The child process's whole job: one search, one answer, then exit.

    Module level because the spawn start method has to import and pickle
    it by name; a closure or a lambda could not cross the process
    boundary at all.
    """
    try:
        sink.send(("ok", re.search(pattern, subject) is not None))
    except re.error as exc:  # pragma: no cover - loader rejects these first
        sink.send(("error", str(exc)))
    finally:
        sink.close()


def run_with_deadline(
    pattern: str, subject: str, deadline: float = REGEX_DEADLINE_SECONDS
) -> tuple[str, Any]:
    """Run one pattern in a single-use child, or give up and kill it.

    A PROCESS rather than a thread, because `re` holds the GIL while it
    backtracks: a thread with a timeout would leave the runaway match
    running and the timeout would buy nothing but a second reference to
    a hung interpreter. Only a process can actually be terminated.

    SPAWN rather than fork. The parent is an asyncio application holding
    a database connection and running background tasks, and forking it
    copies whatever locks other threads happened to hold at that instant,
    which is how a child that only wanted to run a regex ends up
    deadlocked in an allocator. Spawn pays interpreter startup, which is
    the tax for a child that owes the parent nothing.

    Single use, never a pool. A pool would have to be kept alive across
    the process and drained on shutdown, and terminating a pool member
    mid-match is exactly the thing pools are bad at.

    join(deadline) then read, which is safe HERE and would not be in
    general: the classic deadlock is a child blocked writing a payload
    bigger than the pipe buffer while the parent waits for it to exit.
    This payload is a two-item tuple holding a bool.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_regex_search, args=(pattern, subject, sender), daemon=True
    )
    worker.start()
    # The parent's copy of the write end, closed so the pipe reports EOF
    # if the child dies without sending.
    sender.close()
    try:
        worker.join(deadline)
        if worker.exitcode is None:
            return ("timeout", None)
        try:
            # poll() is true at end-of-pipe as well as when a payload is
            # waiting, so the read is guarded rather than gated: a child
            # that died before sending leaves exactly that shape.
            result: tuple[str, Any] = receiver.recv()
            return result
        except EOFError:
            return ("crashed", worker.exitcode)
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()
        receiver.close()


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
        # The one scorer that leaves this process. Everything above is a
        # string comparison whose cost is linear in the subject; a regex
        # can be exponential in it, and `re` cannot be interrupted once
        # it starts, so the only real bound is a child that can be killed.
        state, value = run_with_deadline(spec["pattern"], subject)
        if state == "ok":
            return _verdict(1.0 if value else 0.0, bool(value))
        if state == "error":  # pragma: no cover - loader rejects these first
            # The loader rejects invalid patterns, so reaching here means
            # a row was written before that check or around it. Scoring
            # says so rather than crashing the pass.
            return _verdict(0.0, False, f"pattern failed at scoring time: {value}")
        if state == "timeout":
            # A SCORING FAILURE, not a verdict. score None is what makes
            # the report count this on axis two, where a bench that could
            # not measure something belongs, rather than as a zero the
            # model would wear.
            return {
                "score": None,
                "passed": None,
                "detail": (
                    f"{REGEX_TIMEOUT_DETAIL}: the pattern did not finish "
                    f"within {REGEX_DEADLINE_SECONDS}s against "
                    f"{len(subject)} characters and was terminated"
                ),
            }
        return {
            "score": None,
            "passed": None,
            "detail": f"regex worker exited unexpectedly (code {value})",
        }

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
