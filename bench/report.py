"""Aggregates over an experiment's trials. Pure functions, no I/O.

Two axes, kept apart everywhere in this module, because collapsing them
is the single easiest way to publish a misleading number.

Axis one is the TRIAL outcome: what happened when the bench asked a model
to do a task. done, error, refused, stopped, missing. A model's failure
rate comes from here, and "failure-inclusive denominators" means these
failures: an errored trial is a trial that failed the task, and dropping
it would report on survivors.

Axis two is SCORING COVERAGE: what happened when the bench tried to put a
number on a trial. scored, scoring_failed, unscored. A judge that
returned gibberish, or a judge call the ceiling refused, lives here and
only here. Counting a judge's malfunction as a model's failure would
blame the model for the scorer's bad day, and the two are not even
measuring the same thing.

Every rate this module publishes therefore comes with the count it was
computed from. A pass rate over three usable verdicts out of forty
eligible trials is not a pass rate anybody should act on, and the only
way to know that is to see both numbers.
"""

import math
import random
import statistics
from typing import Any

# The message the disconnect path writes, shared rather than duplicated.
# A reader matching this literal on its own would be matching prose that
# the writer is free to reword; importing the same constant the writer
# uses is what makes the match a definition instead of a guess.
ABORTED_ERROR = "stream aborted before completion"

# How many resamples the interval is built from, and how wide it is.
# 2000 is enough for a percentile interval to be stable to the two
# decimal places a report prints, and cheap enough (a few milliseconds
# for a realistic experiment) to run inline on a request.
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95


def trial_outcome(result: dict[str, Any] | None) -> str:
    """What happened to one trial, derived at read time from its row.

    Derived rather than stored as an enum column, which is this phase's
    whole answer to the question: every fact this reads was already
    recorded for its own reasons, so the vocabulary can change without a
    migration and an old row cannot carry a label that predates the
    current rules.

    The rules, in order:

    None means the trial has no row at all. That happens when a group
    declared a lineup member that never ran, which is what a halted or
    stopped experiment leaves behind, and calling it "missing" rather
    than omitting it is what keeps the denominator honest.

    An error with NO request_json was never sent. request_json is written
    before the request leaves, deliberately and on every path, so its
    absence beside an error means the run was refused before anything was
    built: the spend ceiling. This is structural rather than a match on
    the refusal's wording, which the code that writes it explicitly warns
    may be reworded.

    An error equal to ABORTED_ERROR is a run cut short by a client
    disconnect. Matched against the constant the writer uses, not against
    a copy of its text.

    Any other error is an error: upstream said no, or said something
    unusable. Everything else is done.
    """
    if result is None:
        return "missing"
    error = result.get("error")
    if error is None:
        return "done"
    if result.get("request_json") is None:
        return "refused"
    if error == ABORTED_ERROR:
        return "stopped"
    return "error"


# Outcomes that mean the model actually produced an answer to judge. The
# report's score denominators include everything a model was ASKED to do,
# not only these, which is the failure-inclusive rule; this set exists to
# separate "the model answered badly" from "the model did not answer".
COMPLETED = ("done",)


def scoring_state(scores: list[dict[str, Any]], scorer: str) -> str:
    """Whether this trial has a usable number from this scorer.

    unscored means nobody has tried, which is a fact about the scoring
    pass and not about the model. scoring_failed means a pass tried and
    could not produce a number: an unparseable verdict, or a judge call
    the ceiling refused. scored means there is a number.

    The distinction between the last two is the reason axis two exists.
    Both leave the trial without a score, and only one of them is a
    problem with the bench's own machinery.
    """
    rows = [r for r in scores if r["scorer"] == scorer]
    if not rows:
        return "unscored"
    return "scored" if rows[-1].get("score") is not None else "scoring_failed"


def _percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile over a sorted copy.

    Nearest-rank rather than an interpolating definition because the
    numbers here are latencies from a handful of trials, and reporting an
    interpolated value between two measurements as though it were one
    invents a measurement. p90 of five trials is the fifth-slowest, and
    saying so is more honest than averaging two of them.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_metric(values: list[float]) -> dict[str, Any]:
    """Median and p90 over whatever measurements exist, plus the count.

    The count travels with the numbers because a p90 over two trials is
    arithmetic rather than information, and a reader who cannot see the
    denominator has no way to know which they are looking at.
    """
    usable = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "n": len(usable),
        "median": statistics.median(usable) if usable else None,
        "p90": _percentile(usable, 0.90),
    }


def cluster_bootstrap(
    by_task: dict[str, list[float]], seed: int, iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, Any]:
    """A percentile interval for the mean, resampling TASKS not trials.

    The resampling unit is the task, with every repeat of a sampled task
    kept together. Repeats of one task share that task's difficulty, so
    they are correlated, and resampling trials independently would treat
    forty correlated trials as forty independent ones. That narrows the
    interval below what the data supports, and an overconfident interval
    is worse than no interval: it is the false precision this whole phase
    exists to end, wearing the costume of rigor.

    When repeats is 1 the two schemes coincide exactly, since each
    cluster holds one trial. The difference shows up precisely when
    repeats do the statistical work they were added for.

    Seeded, and the caller records the seed, so an interval in a report
    can be recomputed by anyone holding the export. random.Random rather
    than the module-level functions for the same reason the task shuffle
    uses one: a shared generator would make the interval depend on
    whatever else in the process drew a number first.
    """
    tasks = [values for values in by_task.values() if values]
    flat = [v for values in tasks for v in values]
    if not flat:
        return {"lo": None, "hi": None, "n_clusters": 0, "seed": seed}
    if len(tasks) == 1:
        # One cluster cannot be resampled into anything but itself, so an
        # interval here would be a point pretending to be a range.
        return {
            "lo": None,
            "hi": None,
            "n_clusters": 1,
            "seed": seed,
            "note": "one task: too few clusters for an interval",
        }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        drawn: list[float] = []
        for _ in range(len(tasks)):
            drawn.extend(tasks[rng.randrange(len(tasks))])
        if drawn:
            means.append(sum(drawn) / len(drawn))
    if not means:
        return {"lo": None, "hi": None, "n_clusters": len(tasks), "seed": seed}
    means.sort()
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "lo": _percentile(means, tail),
        "hi": _percentile(means, 1.0 - tail),
        "n_clusters": len(tasks),
        "seed": seed,
    }


def min_ranks(scores: dict[str, float | None]) -> dict[str, int | None]:
    """Ranks where equal values share a rank, and the next rank skips.

    Min-rank ("1224"), not dense and not arbitrary. Two models that
    measured identically are tied, and a ranking that put one above the
    other would be reporting a difference that is not in the data, which
    is the same defect as an overconfident interval in a smaller costume.

    None ranks nothing: a model with no measurement is not last, it is
    unranked, and putting it last would be a claim.
    """
    ranked = sorted(
        ((name, value) for name, value in scores.items() if value is not None),
        key=lambda pair: -pair[1],
    )
    out: dict[str, int | None] = {name: None for name in scores}
    for index, (name, value) in enumerate(ranked):
        if index > 0 and value == ranked[index - 1][1]:
            out[name] = out[ranked[index - 1][0]]
        else:
            out[name] = index + 1
    return out
