"""Aggregates over an experiment's trials. Pure functions, no I/O.

Two axes, kept apart everywhere in this module, because collapsing them
is the single easiest way to publish a misleading number.

Axis one is the TRIAL outcome: what happened when the bench asked a model
to do a task. done, error, refused, stopped, missing, not_run. A model's
failure rate comes from here, and "failure-inclusive denominators" means
these failures: an errored trial is a trial that failed the task, and
dropping it would report on survivors.

Sections are the third thing kept apart, and the newest. A scorer answers
only for the tasks that declared it: see applicable_tasks for what a
section that iterated every task was doing to its neighbours' numbers.

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

import json
import math
import random
import statistics
from collections.abc import Mapping
from typing import Any

# The latest-per-key rule lives in bench.scoring and this module is its
# only consumer. Importing it rather than re-deriving "the last row" is
# the whole point of J4: a correct helper with no call sites is a rule
# nothing obeys.
from bench.scoring import latest_per_key

# The message the disconnect path writes, shared rather than duplicated.
# A reader matching this literal on its own would be matching prose that
# the writer is free to reword; importing the same constant the writer
# uses is what makes the match a definition instead of a guess.
ABORTED_ERROR = "stream aborted before completion"

# How many times the task clusters are resampled to build one interval.
# Named rather than inline because it is the knob that trades interval
# stability against report latency, and an unnamed 2000 buried in a loop
# is a knob nobody can find.
#
# 2000 is enough that the percentile bounds are stable to the two decimal
# places a report prints: the Monte Carlo error on a 95 percent
# percentile bound falls as 1/sqrt(resamples), so 2000 puts it well
# below the rounding. Raising it buys precision the display throws away;
# lowering it makes two loads of the same report disagree in their last
# digit, which reads as instability in the data rather than in the
# method. The seed is what makes a given report reproducible; this is
# what makes it stable enough to be worth reproducing.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95


def provenance_era(run: dict[str, Any] | None) -> bool:
    """Whether this run was written after provenance columns existed.

    data_policy is the marker. It is written unconditionally on every run
    the app has recorded since Phase G, including standard-policy boots:
    _parse_data_policy returns "standard" rather than None when the
    environment variable is unset, and the lifespan always sets it, so a
    NULL there means the row predates the column rather than that the
    boot declined to say.

    That distinction is what makes the refusal rule below safe. Without
    it the rule would read a pre-G errored row, whose request_json is
    NULL only because the column did not exist yet, as a spend refusal.
    A February error would become a February refusal in a report and in
    an export, and the export is forever.
    """
    return run is not None and run.get("data_policy") is not None


def trial_outcome(
    result: dict[str, Any] | None, run: dict[str, Any] | None = None
) -> str:
    """What happened to one trial, derived at read time from its row.

    Derived rather than stored as an enum column, which is this phase's
    whole answer to the question: every fact this reads was already
    recorded for its own reasons, so the vocabulary can change without a
    migration and an old row cannot carry a label that predates the
    current rules.

    run is the result's run row, and it is not optional in spirit even
    though it defaults to None. It carries the era marker, and passing
    None means "era unknown", which downgrades the refusal rule rather
    than disabling the function. Callers that have the run row must pass
    it; the report and the export both do.

    The rules, in order:

    None means the trial has no row at all. That happens when a group
    declared a lineup member that never ran, which is what a halted or
    stopped experiment leaves behind, and calling it "missing" rather
    than omitting it is what keeps the denominator honest.

    An error with NO request_json, ON A ROW FROM THE PROVENANCE ERA, was
    never sent. request_json is written before the request leaves,
    deliberately and on every path, so its absence beside an error means
    the run was refused before anything was built: the spend ceiling.
    This is structural rather than a match on the refusal's wording,
    which the code that writes it explicitly warns may be reworded.

    The era gate is the whole reason provenance_era exists. A pre-G row
    has a NULL request_json for an unrelated reason (the column postdates
    it), so without the gate every legacy error would be reported as a
    refusal. Outside the provenance era an error is simply an error: the
    bench cannot prove a refusal there, and claiming one would be
    inventing a fact about money that never moved.

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
    if result.get("request_json") is None and provenance_era(run):
        return "refused"
    if error == ABORTED_ERROR:
        return "stopped"
    return "error"


# Outcomes that mean the model actually produced an answer to judge. The
# report's score denominators include everything a model was ASKED to do,
# not only these, which is the failure-inclusive rule; this set exists to
# separate "the model answered badly" from "the model did not answer".
COMPLETED = ("done",)

# Every axis-one outcome, in the order a reader walks them: what happened,
# then the three ways nothing happened. Named once so the counters, the
# export and the tests cannot drift apart by one label.
#
# missing and not_run are both absences and they are NOT the same absence.
# A missing trial has a cell: the group exists, the other models in the
# lineup have rows in it, and this one does not, which is a hole in a cell
# that ran. A not_run trial has no cell at all: the runner halted or was
# stopped before it got there, so nothing about it was ever written. One
# is a gap in the record and the other is a plan that was abandoned, and a
# reader deciding whether to trust a comparison needs to tell them apart.
OUTCOMES = ("done", "error", "refused", "stopped", "missing", "not_run")

# The outcomes a quality number may be computed from: the model answered,
# or the model failed to. Everything else on axis one happened for a
# reason that is not about the model (a spend ceiling, an operator, an
# abandoned plan), and letting any of them near a mean, an interval, a
# pass rate or a coverage count would publish that reason as quality.
#
# This is the one gate. The mean, the bootstrap's clusters, the pass
# rate's denominator and both coverage counters all sit behind it, so
# they cannot drift into disagreeing about which trials they speak for.
QUALITY_OUTCOMES = ("done", "error")

# The one scorer no dataset can declare: a person decides to rate a trial
# after the fact. Defined here because both the report and the creation
# boundary need it and two spellings of one name is one too many.
HUMAN_SCORER = "human"


def latest_for(
    scores: list[dict[str, Any]], scorer: str, judge_model: str | None
) -> dict[str, Any] | None:
    """The newest row for one FULL key, or None.

    The key is (scorer, judge_model), both parts, and that is the point.
    Selecting on the scorer alone made two judges of the same trial fight
    over one slot: whichever row was written last answered for both, so a
    report with a strict judge and a lenient one published whichever
    happened to run second and attributed it to neither.

    latest_per_key is where the rule lives, ordering by (created_at, id)
    so a re-scoring pass in the same second still resolves the way the
    writes actually happened. This function is the lookup on top of it,
    and there is exactly one of it so no caller can quietly use half a
    key again.
    """
    return next(
        (
            row
            for row in latest_per_key(scores)
            if row["scorer"] == scorer and row.get("judge_model") == judge_model
        ),
        None,
    )


def scoring_state(
    scores: list[dict[str, Any]], scorer: str, judge_model: str | None
) -> str:
    """Whether this trial has a usable number from this scorer and judge.

    unscored means nobody has tried, which is a fact about the scoring
    pass and not about the model. scoring_failed means a pass tried and
    could not produce a number: an unparseable verdict, or a judge call
    the ceiling refused. scored means there is a number.

    The distinction between the last two is the reason axis two exists.
    Both leave the trial without a score, and only one of them is a
    problem with the bench's own machinery.

    judge_model is required rather than defaulted, deliberately. A
    default would let a caller ask half the question and get a confident
    answer about the wrong rows, which is the defect this workstream
    exists to remove.
    """
    latest = latest_for(scores, scorer, judge_model)
    if latest is None:
        return "unscored"
    return "scored" if latest.get("score") is not None else "scoring_failed"


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
    by_task: dict[str, list[float]], seed: int, resamples: int = BOOTSTRAP_RESAMPLES
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

    A TASK WITH NO QUALITY DATA IS NOT A CLUSTER. The quality gate can
    hollow a task out entirely: every trial of it refused, stopped or
    never run leaves nothing to resample. Such a task must drop out of the
    population rather than contribute an empty cluster, and n_clusters
    must say so, because n_clusters is the interval's stated population
    and an interval claiming twenty tasks when six of them are empty is
    false precision arriving by the back door. The caller never builds an
    empty list (it appends or it does not), and the filter here is the
    second lock on the same door.
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
    for _ in range(resamples):
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


def _report_attachments(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The distinct documents this experiment's cells declared.

    DIGESTS AND A MODE, never a filename and never content. The filename
    is what a person picked on their own machine and is not a property of
    the bytes; two people attaching the same contract under two names
    would produce one document here, which is correct, and a report that
    named one of the two would be citing a fact about a filesystem.

    First appearance wins the order, so a report over one dataset reads
    the documents in the order its cells did rather than in whatever
    order a set iterated. Byte-identical reports depend on it.

    A document appearing under two modes is listed twice, once per mode,
    because they are two different treatments of the same bytes and
    collapsing them would report an experiment nobody ran. Reachable only
    across cells, since one group holds one mode.
    """
    seen: set[tuple[str, str | None]] = set()
    out: list[dict[str, Any]] = []
    for group in groups:
        mode = group.get("attachments_mode")
        for digest in group.get("attachments") or []:
            key = (digest, mode)
            if key in seen:
                continue
            seen.add(key)
            out.append({"digest": digest, "mode": mode})
    return out


def build_report(
    experiment: dict[str, Any],
    groups: list[dict[str, Any]],
    runs_by_group: dict[int, list[dict[str, Any]]],
    scores_by_result: dict[int, list[dict[str, Any]]],
    tasks_by_id: dict[str, dict[str, Any]] | None,
    seed: int,
) -> dict[str, Any]:
    """One experiment's aggregates, per model.

    tasks_by_id carries the dataset file's tasks, and None means no file
    was supplied. None is not the same as an empty mapping: an empty one
    means a file WAS read and declared no thresholds, which is an answer,
    while None means nobody asked the file anything and the pass rate's
    population has to be recovered from the score rows instead. The
    report says which of the two it did, in thresholds_source.

    Pure: every row it needs is handed in, so the whole report can be
    rebuilt from an export without a database. That is not a stylistic
    preference, it is what makes the export's self-sufficiency claim
    checkable, since the check is "run this function over the export and
    get the same numbers".

    The two axes never mix. Trial outcomes drive failure rates and the
    denominators of score means; scoring coverage drives the pass rate's
    coverage figure and the scoring-failure rate. A judge that could not
    answer never touches a model's failure count.
    """
    lineup = experiment["lineup"]
    # An ARM is a lineup slot: (position, model). A lineup may list the
    # same model twice, which is how anyone asks "how much does this vary
    # run to run", and the two copies are two arms with two sets of paid
    # trials. Keying any of this by model alone would merge them, and the
    # merge would be silent.
    arms = list(enumerate(lineup))
    by_model: dict[str, list[int]] = {}
    for position, model in arms:
        by_model.setdefault(model, []).append(position)
    # trials[arm][task_id] = list of per-trial records, so the bootstrap
    # can resample by task without a second pass over the rows.
    trials: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = {
        arm: {} for arm in arms
    }
    # Models whose arms had to be reconstructed from row order somewhere
    # in this experiment; see _cell_arms. Collected across cells because
    # the caveat belongs to the arm on every page it appears on, not only
    # to the cell that earned it.
    reconstructed: set[str] = set()
    for group in groups:
        task_id = group["task_id"]
        seen: set[tuple[int, str]] = set()
        rows = [
            (run, result)
            for run in runs_by_group.get(group["id"], [])
            for result in run["results"]
        ]
        assignment, ambiguous = _cell_arms(rows, by_model)
        reconstructed |= ambiguous
        for (run, result), arm in zip(rows, assignment, strict=True):
            if arm is None:
                continue
            seen.add(arm)
            trials[arm].setdefault(task_id, []).append(
                {
                    "outcome": trial_outcome(result, run),
                    "result": result,
                    "scores": scores_by_result.get(result["id"], []),
                    "task_id": task_id,
                }
            )
        # Declared members with no row at all. Recorded as missing rather
        # than skipped, because a halted experiment's un-run trials are
        # part of what was asked and leaving them out would quietly shrink
        # every denominator they belong to.
        for arm in arms:
            if arm not in seen:
                trials[arm].setdefault(task_id, []).append(
                    {
                        "outcome": "missing",
                        "result": None,
                        "scores": [],
                        "task_id": task_id,
                    }
                )

    # The plan, from the manifest's own arithmetic rather than from the
    # groups that happen to exist. Every model owes one trial per (task,
    # repeat) cell, so the cells a halted runner never created are the
    # difference between what was declared and what is on disk.
    #
    # Deriving this from rows instead is the defect that makes a halt
    # invisible: the act of stopping early would shrink the reported plan
    # to exactly what ran, and nothing anywhere would say what was owed.
    cells_planned = int(experiment.get("tasks_total") or 0) * int(
        experiment.get("repeats") or 0
    )
    not_run = max(0, cells_planned - len(groups))
    # Derived once over every model's rows, because a declared threshold
    # is a fact about the task rather than about who answered it. Computed
    # even when the file is present, so the cost is one pass over rows
    # already in memory and the two witnesses can be compared in a test.
    row_declared = rows_declaring_thresholds(trials)
    row_scored = rows_scoring_tasks(trials)
    # Discovered ONCE, over the whole experiment; see experiment_series.
    series = experiment_series(trials)
    models = [
        _model_report(
            model,
            position,
            len(by_model.get(model, ())) > 1,
            model in reconstructed,
            by_task,
            tasks_by_id,
            seed,
            series,
            row_declared,
            not_run,
            row_scored,
        )
        for (position, model), by_task in trials.items()
    ]
    ranking = choose_metric(experiment.get("primary_metric"), models)
    metric = ranking["metric"]
    judge = ranking["judge_model"]
    for entry in models:
        # The FULL key. Matching on the scorer alone picked whichever
        # series sorted first, which is the judgeless one, so a scorer
        # carrying both a gap row and a real verdict ranked every model
        # on the gap. See choose_metric.
        section = next(
            (
                s
                for s in entry["scorers"]
                if s["scorer"] == metric and s["judge_model"] == judge
            ),
            None,
        )
        entry["score"] = {
            "metric": metric,
            "judge_model": judge,
            "mean": (section or {}).get("mean"),
            "interval": (section or {}).get("interval"),
        }
    # Min-rank so ties share a place, and only when a metric was chosen.
    # A rank with no named metric is a claim about which model is better
    # with nothing behind it.
    ranks = (
        min_ranks({m["label"]: m["score"]["mean"] for m in models})
        if metric is not None
        else {m["label"]: None for m in models}
    )
    for entry in models:
        entry["rank"] = ranks[entry["label"]]
    return {
        "experiment_id": experiment["id"],
        "name": experiment["name"],
        # Every report says which question it answers. A number without
        # its estimand is a number about nothing in particular.
        "estimand_mode": experiment["estimand_mode"],
        "dataset_name": experiment["dataset_name"],
        "dataset_digest": experiment["dataset_digest"],
        "repeats": experiment["repeats"],
        "status": experiment["status"],
        "status_detail": experiment["status_detail"],
        # The plan as arithmetic, published so a reader can check the
        # outcome counters against it instead of trusting them. This is
        # also what makes not_run derivable from an export alone.
        "plan": {
            "tasks": int(experiment.get("tasks_total") or 0),
            "repeats": int(experiment.get("repeats") or 0),
            "models": len(lineup),
            "cells": cells_planned,
            "trials": cells_planned * len(lineup),
            "cells_recorded": len(groups),
        },
        # Where the pass rate's population came from. Said out loud
        # because the two are not equally good: score_rows is a floor,
        # since a declared task nobody scored leaves no row to witness
        # its threshold, and a reader comparing two reports has to know
        # which one had the file.
        "thresholds_source": "score_rows" if tasks_by_id is None else "dataset_file",
        # The documents these numbers were produced over, as provenance
        # and not as content. Every model in a cell read the same
        # documents, so this is a property of the experiment rather than
        # of any arm, which is why it sits here and not on a model row.
        #
        # Empty on every experiment the runner produces today, because
        # per-task attachments in datasets are deferred. Published anyway
        # rather than omitted when empty, for thresholds_included's
        # reason: a reader must be able to tell "no documents" from "this
        # report predates the field", and an absent key cannot say the
        # first.
        "attachments": _report_attachments(groups),
        # The caveat, in the payload, as one sentence a page can print.
        # None when nothing was reconstructed, so a reader who sees the
        # key filled in knows it was earned rather than boilerplate.
        "arm_caveat": LEGACY_ARM_CAVEAT if reconstructed else None,
        # The ranking always names the metric that produced it, or says
        # plainly that there is none. An unlabelled leaderboard is the
        # single easiest way to publish a claim nobody made.
        "ranking": ranking,
        "bootstrap": {
            "seed": seed,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            # Named in the report because the reader has to know which
            # unit was resampled to know what the interval claims.
            "unit": "task",
        },
        # By position, which IS the declared order, rather than by a
        # lookup of the model's name: with a duplicated model the lookup
        # returns the same index for both arms and the sort silently
        # depends on whatever order they arrived in.
        # Judge spend on its own line, never folded into the models'
        # totals. It is the bench's own instrument cost: money this tool
        # spent measuring, not money any model under test was paid. A
        # single "total cost" adding the two would answer neither "what
        # did this comparison cost me" nor "what did these models cost".
        "judge_cost": _judge_cost(scores_by_result),
        "models": sorted(models, key=lambda m: m["position"]),
    }


LEGACY_ARM_CAVEAT = (
    "some arms below were reconstructed from row order rather than read "
    "from the record. Two or more rows in one cell claimed the same "
    "lineup position, which every pre-I.2 run did for a duplicated model "
    "because position came from lineup.index() and that returns the first "
    "match. The trials themselves are real and their costs are real; "
    "WHICH of the duplicated arms each one belongs to is a reconstruction, "
    "so treat a difference between two arms of the same model in this "
    "experiment as unproven"
)


def _cell_arms(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    by_model: dict[str, list[int]],
) -> tuple[list[tuple[int, str] | None], set[str]]:
    """One cell's rows mapped to lineup slots, plus the models it guessed.

    The recorded position decides whenever it can, because that is what
    the runner wrote and it is the only field that can tell two copies of
    one model apart. "Whenever it can" means every row of that model in
    this cell carries a distinct position that the model actually
    occupies.

    IT CANNOT FOR LEGACY DUPLICATES, and the old rule made that much
    worse than a missing label. Before I.2 the runner derived position
    from lineup.index(model), which returns the FIRST match, so a lineup
    listing one model twice wrote position 0 on both rows. Reading them
    back, both mapped to arm 0 and arm 1 matched nothing, so the report
    invented a MISSING trial for an arm that had in fact run and been
    paid for, and handed arm 0 both charges. A phantom absence in one
    column and a doubled bill in the other, from data that was complete.

    So a collision falls back to ROW ORDER, which is the order the runner
    wrote them in and therefore the order it ran them: recovery rather
    than invention. It is still a guess about WHICH copy is which, and
    that is what the caveat says. Rows beyond the available slots are
    dropped, as they always were; slots beyond the available rows stay
    empty and become the missing trial they genuinely are.

    A model with ONE slot is never ambiguous, whatever its rows say,
    because there is only one answer available. Recovering such a row
    beats dropping it: dropping loses a paid trial from every denominator
    it belongs to, which is the same class of defect one size smaller.
    """
    indices_by_model: dict[str, list[int]] = {}
    for index, (_run, result) in enumerate(rows):
        indices_by_model.setdefault(result["model"], []).append(index)
    out: list[tuple[int, str] | None] = [None] * len(rows)
    ambiguous: set[str] = set()
    for model, indices in indices_by_model.items():
        slots = by_model.get(model, [])
        if not slots:
            # A row for a model this lineup never declared. Not this
            # experiment's, and not something to file under any arm.
            continue
        positions = [rows[index][1].get("position") for index in indices]
        usable = [p for p in positions if isinstance(p, int) and p in slots]
        if len(usable) == len(positions) == len(set(usable)):
            # usable, not positions: same values in the same order in
            # this branch, and the one of the two that is known to hold
            # ints rather than whatever the column contained.
            for index, position in zip(indices, usable, strict=True):
                out[index] = (position, model)
            continue
        if len(slots) > 1:
            ambiguous.add(model)
        for index, slot in zip(indices, slots, strict=False):
            out[index] = (slot, model)
    return out, ambiguous


def choose_metric(declared: str | None, models: list[dict[str, Any]]) -> dict[str, Any]:
    """Which scorer the ranking is computed on, and why, or none at all.

    A RANKING IS A CLAIM. It says one model did better than another, and
    that claim only means something once somebody has said better AT
    WHAT. The previous rule took per_scorer[0], the first of a sorted
    list, so an experiment scored by both `contains` and `judge` was
    ranked on `contains` because c sorts before j. Nobody chose that, the
    report did not say it, and adding a scorer named `accuracy` would
    have silently reordered the leaderboard.

    Three cases, and the report names which one it is in:

    A declared primary_metric wins. Validated at creation against the
    dataset's scorers, so it cannot name something nothing produces.

    Exactly one scorer across the whole experiment ranks on that scorer,
    because there is no choice to make and refusing to rank would be
    pedantry.

    Otherwise NO RANKING. Sections are still published in full, with
    every mean, interval and pass rate; what is withheld is the single
    ordering, because inventing one would be putting the report's name to
    a preference its author never expressed.
    """
    available = sorted({s["scorer"] for m in models for s in m["scorers"]})
    if not available:
        return _ranking(None, None, "nothing has been scored", available, models)
    if declared is not None:
        metric = declared
        reason = "declared as the experiment's primary metric"
    elif len(available) == 1:
        metric = available[0]
        reason = "the only scorer in this experiment"
    else:
        return _ranking(
            None,
            None,
            "more than one scorer and no primary_metric declared, so this "
            "report publishes each scorer's section and no cross-scorer "
            "ranking: ordering models needs somebody to say better at what",
            available,
            models,
        )
    # RANKABLE series only, and everything below is computed over those.
    # A series nobody measured cannot order anything; see rankable_series
    # for what happened when one tried.
    rankable = rankable_series(metric, models)
    if not rankable:
        return _ranking(
            None,
            None,
            f"{metric} measured nothing on any arm: every value behind "
            "its means is a failure-inclusive zero or absent entirely, "
            "so an ordering would put the arms that failed above the "
            "arms nobody scored. The sections are published in full; the "
            "ranking waits for a scoring pass that produced a number",
            available,
            models,
        )
    # A metric has to resolve to ONE series. Two judges under the chosen
    # scorer is the same question one level down, "better according to
    # whom", and averaging them would publish a number neither judge
    # produced. So the ranking is withheld and names the judges rather
    # than picking one, exactly as it withholds rather than picking a
    # scorer alphabetically.
    judges = sorted(judge for judge in rankable if judge is not None)
    if len(judges) > 1:
        return _ranking(
            None,
            None,
            f"{metric} was scored by more than one judge ("
            + ", ".join(judges)
            + "), and averaging judges is a claim nobody made: declare "
            "one judge's pass or read the sections",
            available,
            models,
        )
    # The SERIES, not just the scorer. A scorer can carry a judgeless
    # series beside a judged one: I3 records "no judge model was given
    # for this scoring pass" as a row under scorer `judge` with a NULL
    # judge_model, so a task scored once without a judge and once with
    # one has two series and exactly one real judge. Returning the scorer
    # alone let the caller take whichever series came first, which sorts
    # judgeless before judged, so the ranking named `judge` and ordered
    # every model on the gap row's empty mean.
    return _ranking(metric, judges[0] if judges else None, reason, available, models)


def rankable_series(metric: str, models: list[dict[str, Any]]) -> set[str | None]:
    """Which of this metric's series could order anything: judge_model set.

    A SERIES IS RANKABLE ONLY IF SOMEBODY MEASURED IT. `measured` counts
    the values in a section's mean that came from a score rather than
    from the failure-inclusive rule, and a series where that is zero
    across every arm has no measurement anywhere in it.

    Its mean is not None, which is the trap. An arm whose trials all
    errored gets a 0.0 per trial from the failure-inclusive rule, so its
    mean is 0.0: a real number, the lowest one available, and the only
    number in the series if its neighbours' trials went unscored and
    reported None. min_ranks then ranks the arm that failed everything
    FIRST, because it is the only arm with a value, and the report says
    nothing to suggest that is what happened. Failing every trial came
    first on the leaderboard.

    The judgeless failure series is the same defect with a different
    cause. A scoring pass run with no judge model records a row under
    scorer `judge` with a NULL judge_model and a NULL score, so the
    series exists, measures nothing, and is a candidate for exactly this.

    Both are COVERAGE material, never ranking material. They belong on
    the page (the sections publish in full, and the coverage counters are
    where "nobody scored this" is supposed to be read) and they do not
    belong in an ordering.
    """
    return {
        s["judge_model"]
        for m in models
        for s in m["scorers"]
        if s["scorer"] == metric and s.get("measured")
    }


def _ranking(
    metric: str | None,
    judge_model: str | None,
    reason: str,
    available: list[str],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    """The ranking block, with the composition a human metric needs.

    A report ranked on human ratings that were made BLIND is a different
    claim from one ranked on ratings made while the rater could see which
    model wrote which answer, and the second is the weaker claim by a
    wide margin. The rows already carry the flag; without this line a
    reader would have to join tables to learn which report they are
    holding, and almost nobody will.

    Counted across every model, because the composition is a fact about
    the ratings rather than about any one arm.
    """
    out: dict[str, Any] = {
        "metric": metric,
        # Which series the ranking used, named beside the metric. A
        # scorer is not a series, and a ranking that named only the
        # scorer could not say which of two instruments produced it.
        "judge_model": judge_model,
        "reason": reason,
        "available": available,
    }
    if metric == HUMAN_SCORER:
        sections = [
            s for m in models for s in m["scorers"] if s["scorer"] == HUMAN_SCORER
        ]
        out["blind_ratings"] = sum(s["blind"] for s in sections)
        out["ratings"] = sum(s["rated"] for s in sections)
    return out


def experiment_series(
    trials_by_arm: Mapping[Any, dict[str, list[dict[str, Any]]]],
) -> list[tuple[str, str | None]]:
    """Every (scorer, judge_model) pair anywhere in the experiment.

    DISCOVERED AT EXPERIMENT LEVEL, and that is the whole point. Derived
    per arm, from that arm's own score rows, an arm with no rows got no
    sections at all: it vanished from the scorer table entirely. Not with
    a zero, not with an unscored count, gone.

    That is the worst way to report a gap, because absence on a page
    reads as "nothing to say" rather than as "nobody looked". A scoring
    pass that ran out of budget partway through the lineup, or was
    stopped, leaves exactly this shape, and the arm it never reached is
    the one a reader most needs to see labelled. Its errored trials also
    owe the series their failure-inclusive zeros, and those disappeared
    with it, so the neighbours' means were being compared against a
    population the missing arm was silently excused from.

    A section for a series this arm has no rows in is not an invention:
    the tasks are real, the trials are real, and what the section reports
    is that they are unscored. WHICH tasks it covers is still decided by
    applicable_tasks, so a scorer still answers only for the tasks that
    declared it.

    Sorted with the judgeless series first under each scorer, so a
    deterministic scorer and a human rating read before the judges that
    came later.
    """
    return sorted(
        {
            (row["scorer"], row.get("judge_model"))
            for by_task in trials_by_arm.values()
            for trials in by_task.values()
            for trial in trials
            for row in trial["scores"]
        },
        key=lambda pair: (pair[0], pair[1] or ""),
    )


def _model_report(
    model: str,
    position: int,
    duplicated: bool,
    legacy_ambiguous: bool,
    by_task: dict[str, list[dict[str, Any]]],
    tasks_by_id: dict[str, dict[str, Any]] | None,
    seed: int,
    series: list[tuple[str, str | None]],
    row_declared: dict[str, set[str]] | None = None,
    not_run: int = 0,
    row_scored: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    flat = [t for trials in by_task.values() for t in trials]
    outcomes = {name: 0 for name in OUTCOMES}
    for trial in flat:
        outcomes[trial["outcome"]] += 1
    # not_run has no per-trial record to count, by definition: the runner
    # never created the cell, so there is no group, no run and no row.
    # It arrives as a number derived from the manifest's arithmetic and
    # is added here rather than synthesized as fake trials, because a
    # fabricated record is exactly what an export must not contain.
    outcomes["not_run"] = not_run
    # Three populations, and their names are the contract. Export field
    # names are forever, so a counter whose label means something wider
    # than the thing it counts is a defect of the same class as crossing
    # the two axes: a fact about the harness arriving in the model's
    # column.
    #
    # PLANNED is every trial the plan called for, and it comes from the
    # PLAN rather than from the rows. Counting rows would make a halted
    # experiment report a smaller plan than it declared, so the very act
    # of halting would hide how much was skipped: eight planned trials
    # stopped after two would have reported a plan of two, with nothing
    # anywhere saying six were owed.
    #
    # ATTEMPTED is the subset that ran to a MODEL-ATTRIBUTABLE end: the
    # model answered, or the model failed to. It is the failure rate's
    # denominator and it must contain only trials whose outcome is a fact
    # about the model.
    #
    # `stopped` is not one of those and used to be counted here. A
    # stopped trial was cut off by an operator or a client disconnect
    # part way through, so nobody knows whether it would have succeeded;
    # putting it in the denominator makes the published failure rate
    # depend on WHEN SOMEBODY PRESSED STOP. Stop a run early and the rate
    # falls, with nothing on the page connecting the two.
    #
    # A refused trial was declined by the spend ceiling before the call
    # went out; a missing trial has a cell but no row; a not_run trial
    # has no cell at all. None of those is an attempt either.
    #
    # This set is QUALITY_OUTCOMES, which is not a coincidence and is
    # worth stating: the trials a quality number may be computed from and
    # the trials a failure rate may be computed over are the same trials,
    # because both questions are "what did the model do".
    planned = len(flat) + not_run
    attempted = sum(outcomes[name] for name in QUALITY_OUTCOMES)

    # Score means over EVERY trial the model was asked to do, with a
    # failed trial scoring zero. That is the failure-inclusive rule: a
    # mean over only the trials that came back is a mean over survivors,
    # and the models that fail most would look best.
    # One section per SERIES, a (scorer, judge_model) pair, and the list
    # is the EXPERIMENT'S rather than this arm's: see experiment_series
    # for what an arm with no rows of its own used to do to the page.
    per_scorer = [
        _scorer_report(
            scorer, judge, by_task, tasks_by_id, seed, row_declared, row_scored
        )
        for scorer, judge in series
    ]

    completed = [t["result"] for t in flat if t["outcome"] in COMPLETED]
    # Cost is over EVERY row that has one, not over the completed ones.
    # A trial that streamed tokens and then broke was charged for those
    # tokens, and a total that skipped it would under-report a real bill
    # by exactly the amount the failures cost, which is the direction
    # nobody checks. Latency and provider stay on completed trials,
    # because a broken stream has no honest latency to report.
    billable = [t["result"] for t in flat if t["result"] is not None]
    return {
        "model": model,
        # The lineup slot, always. A reader with one arm per model never
        # needs it and loses nothing by seeing it; a reader with two arms
        # of one model cannot tell them apart without it.
        "position": position,
        # The name to print and to rank by. Identical to the model unless
        # the lineup listed it more than once, in which case the two arms
        # need names that differ or a leaderboard would show one row
        # twice with different numbers and no way to say why.
        "label": f"{model} #{position}" if duplicated else model,
        "duplicate_arm": duplicated,
        # Whether this arm's membership was reconstructed from row order
        # rather than read off the rows. True only for a duplicated model
        # whose legacy rows collided at one position; see _cell_arms and
        # LEGACY_ARM_CAVEAT. Carried per arm rather than only as a
        # report-level note, because a reader looking at one row of the
        # table needs to know it about THAT row.
        "legacy_ambiguous": legacy_ambiguous,
        "trials": {
            "planned": planned,
            "attempted": attempted,
            **outcomes,
            # Failures over what the model was actually given, and errors
            # only. A stop is the operator's decision, a refusal is the
            # ceiling's, and a missing trial is the experiment's; none of
            # the three is the model failing a task, and putting them
            # here would let a halted run read as a bad model.
            "failure_rate": outcomes["error"] / attempted if attempted else None,
            # Refusals are a fact about the budget against the plan, so
            # the plan is the denominator. Dividing them by the attempts
            # they are definitionally not part of would be incoherent
            # even though it happens to give the same number today.
            "refusal_rate": outcomes["refused"] / planned if planned else None,
        },
        "scorers": per_scorer,
        "latency_ms": summarize_metric([r.get("latency_ms") for r in completed]),
        "ttft_ms": summarize_metric([r.get("ttft_ms") for r in completed]),
        "cost": _cost_totals(billable),
        # The routed-service estimand made visible. Under dynamic routing
        # the provider is chosen per call, so it is the largest confound
        # in any comparison the bench draws; counting them is what turns
        # the confound from an assumption into a column.
        "providers": _provider_counts(completed),
    }


def rows_scoring_tasks(
    trials_by_arm: Mapping[Any, dict[str, list[dict[str, Any]]]],
) -> dict[str, set[str]]:
    """scorer -> task ids that have at least one row from that scorer.

    Which tasks a scorer APPLIES to, witnessed by the rows. Same floor
    semantics as rows_declaring_thresholds and for the same reason: a
    task nobody scored leaves nothing behind to say the scorer was meant
    to run on it.

    Computed over every model, because applicability is a fact about the
    task. Per model, a model whose trials went unscored would report a
    smaller population than its neighbour for the same task.
    """
    scored: dict[str, set[str]] = {}
    for by_task in trials_by_arm.values():
        for task_id, trials in by_task.items():
            for trial in trials:
                for row in trial["scores"]:
                    scored.setdefault(row["scorer"], set()).add(task_id)
    return scored


def applicable_tasks(
    scorer: str,
    tasks_by_id: dict[str, dict[str, Any]] | None,
    row_scored: dict[str, set[str]],
) -> set[str]:
    """The tasks one scorer's section is computed over.

    A scorer answers for the tasks that declared it and for no others.
    Without this a section iterated EVERY task in the experiment, so a
    task scored by `contains` contributed to the `judge` section: its
    errored trials added zeros to the judge's mean and its completed
    trials added to the judge's unscored coverage, both for a scorer that
    was never meant to look at it. One scorer's tasks were dragging
    another scorer's numbers around, and the two-axes work could not see
    it because the crossing was not between the axes but between the
    sections.

    The file is authoritative for any scorer it mentions, exactly as it is
    for thresholds. Where it says nothing, the rows are the witness, with
    the same floor limit: a task nobody scored cannot testify.

    A scorer the file NEVER mentions falls to the rows even when a file
    was supplied, and human ratings are why. No dataset declares `human`;
    a person decides to rate a trial after the fact. Reading the file's
    silence as "this scorer applies to nothing" would delete every human
    mean from any report built with a dataset path, which is absence read
    as denial, the same mistake the threshold fallback exists to avoid.
    """
    if tasks_by_id is not None:
        declared = {
            task_id
            for task_id, task in tasks_by_id.items()
            if ((task or {}).get("scorer") or {}).get("kind") == scorer
        }
        if declared:
            return declared
    return set(row_scored.get(scorer, set()))


def rows_declaring_thresholds(
    trials_by_arm: Mapping[Any, dict[str, list[dict[str, Any]]]],
) -> dict[str, set[str]]:
    """scorer -> task ids whose score rows witness a declared threshold.

    The fallback that keeps pass rates alive when the dataset file is
    gone. Thresholds live in the file and nowhere else, so a report built
    without it used to publish no pass rate at all, even with every
    verdict sitting in the database. The verdicts are the evidence: I3
    writes `passed` from judged_pass, which returns None unless the
    task's author declared a cutoff, so a judge row with a non-None
    `passed` IS a record that a threshold existed when that row was
    written.

    A JUDGE row, specifically, which is why judge_model gates the
    witness. A deterministic scorer's `passed` is its own score restated
    (`exact` either matched or did not) and it is written unconditionally,
    so it says nothing about a threshold. Counting it would make the same
    experiment report a different eligible population depending on
    whether a path was passed, and two answers from one dataset is the
    failure this whole layer exists to prevent. The loader permits
    pass_threshold only on judge, so this gate is exactly the file rule
    read off the rows.

    Computed over EVERY model's trials, because a declared threshold is a
    fact about the task. Deriving it per model would let a model whose
    trials went unscored report a smaller eligible population than its
    neighbour for the same task.

    THE HONEST LIMIT: a task nobody ever scored leaves no row to witness
    its declaration, so what this returns is a FLOOR. The dataset file
    remains the only way to get the full denominator, and the report
    labels which of the two it used.
    """
    declared: dict[str, set[str]] = {}
    for by_task in trials_by_arm.values():
        for task_id, trials in by_task.items():
            for trial in trials:
                for row in trial["scores"]:
                    if row.get("passed") is None or row.get("judge_model") is None:
                        continue
                    declared.setdefault(row["scorer"], set()).add(task_id)
    return declared


def _eligible_tasks(
    scorer: str,
    tasks_by_id: dict[str, dict[str, Any]] | None,
    row_declared: dict[str, set[str]],
) -> set[str]:
    """Which tasks this scorer's pass rate is computed over.

    tasks_by_id is None when no dataset file was supplied, and an empty
    mapping when one was supplied that declared no thresholds. The two
    are different facts and the sentinel keeps them apart: falling back
    to the rows in the second case would answer a question the file has
    already answered.
    """
    if tasks_by_id is None:
        return set(row_declared.get(scorer, set()))
    return {
        task_id
        for task_id, task in tasks_by_id.items()
        if ((task or {}).get("scorer") or {}).get("pass_threshold") is not None
    }


def _scorer_report(
    scorer: str,
    judge_model: str | None,
    by_task: dict[str, list[dict[str, Any]]],
    tasks_by_id: dict[str, dict[str, Any]] | None,
    seed: int,
    row_declared: dict[str, set[str]] | None = None,
    row_scored: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """One SERIES' numbers for one model, with both axes reported.

    A series is a (scorer, judge_model) pair, not a scorer. Two judges
    grading the same trials are two measurements by two instruments that
    disagree on purpose, and averaging them would publish a number
    neither of them produced. Deterministic scorers and human ratings
    carry no judge, so each is a single series under its own name.

    Computed over the tasks that DECLARED this scorer and no others; see
    applicable_tasks for why a section that iterated every task was
    dragging one scorer's tasks through another scorer's numbers.

    The pass rate is passed over USABLE VERDICTS among tasks that
    declared a threshold, and it always ships with its coverage. A pass
    rate over three verdicts out of forty eligible trials is not a pass
    rate anybody should act on, and the coverage number is the only thing
    that says so; without it the denominator shrinks silently and the
    rate looks like every other rate on the page.
    """
    eligible_tasks = _eligible_tasks(scorer, tasks_by_id, row_declared or {})
    mine = applicable_tasks(scorer, tasks_by_id, row_scored or {})
    values_by_task: dict[str, list[float]] = {}
    scored = failed = unscored = 0
    passed = usable_verdicts = eligible = 0
    blind_rows = self_judged_rows = rated = measured = 0
    for task_id, trials in by_task.items():
        if task_id not in mine:
            # Another scorer's task. Not this section's business, on
            # either axis: its zeros are not this scorer's failures and
            # its absent rows are not this scorer's coverage gap.
            continue
        for trial in trials:
            # THE QUALITY POPULATION, and everything below it shares this
            # one gate: the mean, the interval's clusters, the pass rate's
            # denominator, and both coverage counters.
            #
            # done and error are in. A completed trial is evidence about
            # quality; an errored one is a trial that failed the task, and
            # dropping it would average over survivors and flatter the
            # models that fail most.
            #
            # refused, stopped, missing and not_run are out, and each for
            # its own reason, none of them about the model. A refusal is
            # the spend ceiling declining to buy the answer. A stop is the
            # operator ending the run. A missing trial left no row in a
            # cell that ran; a not_run trial has no cell. Scoring any of
            # them zero would put a budget, an operator or an abandoned
            # plan into a model's quality number, where it is
            # indistinguishable from a bad answer. Calling them
            # "unscored" would be no better: it would report a gap in
            # coverage that nobody could ever close.
            #
            # A score row on a stopped trial is not deleted and is not
            # ignored. It stays in the database and in the export as the
            # audit trail of what the scoring pass did; it simply never
            # reaches a published quality number, which is enforced HERE,
            # by this gate, and nowhere else.
            if trial["outcome"] not in QUALITY_OUTCOMES:
                continue
            state = scoring_state(trial["scores"], scorer, judge_model)
            latest = latest_for(trial["scores"], scorer, judge_model)
            # Axis two, counted on its own.
            if state == "scored":
                scored += 1
            elif state == "scoring_failed":
                failed += 1
            else:
                unscored += 1
            if latest is not None:
                rated += 1
                if latest.get("blind"):
                    blind_rows += 1
                if latest.get("self_judged"):
                    self_judged_rows += 1
            # Inside the quality population there are exactly two rules,
            # and this is where the axes would be easiest to cross.
            #
            # An ERRORED trial contributes zero. It is a trial that failed
            # the task, and that is a fact about the model.
            #
            # A COMPLETED trial with no usable score contributes NOTHING.
            # Its absence is axis two: nobody scored it, or the judge
            # could not answer. Scoring it zero would put the judge's
            # malfunction into the model's mean, where it is
            # indistinguishable from a bad answer.
            value = latest.get("score") if latest else None
            if trial["outcome"] not in COMPLETED:
                values_by_task.setdefault(task_id, []).append(0.0)
            elif value is not None:
                values_by_task.setdefault(task_id, []).append(value)
                # The value came from a SCORE rather than from the
                # failure-inclusive rule. Counted because a mean of 0.0
                # built entirely from failure zeros and a mean of 0.0
                # somebody measured are the same number and different
                # claims, and only the second one can order anything.
                # See rankable_series.
                measured += 1
            # The pass rate's population: tasks whose author declared a
            # threshold. A task without one contributes to the mean and
            # not to the rate, which is what "pass rate where a threshold
            # was declared" means. The set came from the file when one
            # was supplied and from the score rows otherwise; see
            # rows_declaring_thresholds for what the second one costs.
            if task_id in eligible_tasks:
                eligible += 1
                if latest is not None and latest.get("passed") is not None:
                    usable_verdicts += 1
                    if latest["passed"]:
                        passed += 1

    flat = [v for values in values_by_task.values() for v in values]
    total_states = scored + failed + unscored
    return {
        "scorer": scorer,
        # Named on every series so a reader never has to infer which
        # instrument produced a number from where it sits on the page.
        "judge_model": judge_model,
        "mean": sum(flat) / len(flat) if flat else None,
        "n": len(flat),
        # How many of those n values are measurements rather than
        # failure-inclusive zeros. n is the mean's denominator; this is
        # how much of it anybody actually scored, and the difference is
        # what tells "scored zero" from "never answered". Zero here means
        # the series measured NOTHING, whatever its mean says.
        "measured": measured,
        "interval": cluster_bootstrap(values_by_task, seed) if flat else None,
        # Axis two, reported as its own numbers rather than folded into
        # anything. scoring_failure_rate is also the measurement the
        # response_format decision in bench/models.py says to revisit
        # against, which is why it is computed even when nothing reads it.
        "coverage": {
            "scored": scored,
            "scoring_failed": failed,
            "unscored": unscored,
            "scoring_failure_rate": failed / total_states if total_states else None,
        },
        "pass_rate": {
            "rate": passed / usable_verdicts if usable_verdicts else None,
            "passed": passed,
            "usable_verdicts": usable_verdicts,
            "eligible": eligible,
        },
        "blind": blind_rows,
        "self_judged": self_judged_rows,
        # How many trials this series actually put a row on, which is the
        # denominator the blind count needs. "3 blind" means nothing
        # until it says 3 of how many.
        "rated": rated,
    }


def _cost_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Billed first, estimate as the fallback, and both counts shown.

    The same rule the ceiling uses: what the platform charged beats what
    the catalog implies. Reporting the two counts beside the total is
    what keeps a total that is mostly estimate from reading like a bill.

    Given every row that exists, including errored, stopped and refused
    ones. A refusal carries no cost and contributes nothing; a trial that
    streamed tokens before breaking carries a real charge, and leaving it
    out would under-report the bill by exactly what the failures cost.
    """
    billed = [
        r["billed_cost_usd"] for r in results if r.get("billed_cost_usd") is not None
    ]
    estimated = [
        r["cost_usd"]
        for r in results
        if r.get("billed_cost_usd") is None and r.get("cost_usd") is not None
    ]
    unpriced = len(results) - len(billed) - len(estimated)
    # The BYOK figures, verbatim and NEVER SUMMED. total_usd above is what
    # OpenRouter charged, in credits, and it is the number the spend
    # ceiling meters. This is what an upstream provider billed directly on
    # a bring-your-own-key run, which is a different bill in a different
    # place that OpenRouter cannot decline; adding the two would produce a
    # figure nobody is owed. Kept as the strings that were stored, because
    # the whole use for them is matching a provider's own invoice line and
    # a float would reformat the number being matched.
    upstream = [
        r["upstream_inference_cost_usd"]
        for r in results
        if r.get("upstream_inference_cost_usd") is not None
    ]
    return {
        "total_usd": sum(billed) + sum(estimated),
        "billed_trials": len(billed),
        "estimated_trials": len(estimated),
        "unpriced_trials": unpriced,
        # None rather than an empty block when no run was BYOK, so the
        # key being filled in means something happened rather than that
        # the report always says this.
        "upstream": (
            {"trials": len(upstream), "values": upstream} if upstream else None
        ),
    }


def _judge_cost(scores_by_result: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """What the judging itself cost, over every score row that was billed.

    Every row, not the latest per key: a re-scoring pass paid for its
    call whether or not its verdict is the one the report now uses, and
    superseded spend is still spend. That is the same rule the trial
    totals follow, one level up.
    """
    charges = [
        row["judge_billed_cost_usd"]
        for rows in scores_by_result.values()
        for row in rows
        if row.get("judge_billed_cost_usd") is not None
    ]
    return {"total_usd": sum(charges), "billed_calls": len(charges)}


def _provider_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        name = result.get("provider")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


# The export's own schema version, bumped when a line's shape changes in
# a way a reader could not absorb. Not the app's version and not the
# dataset's: a citation names an artifact, and the artifact has to say
# which format it is in without anyone consulting a changelog.
EXPORT_SCHEMA_VERSION = 2

# WHY EACH VERSION IS THE NUMBER IT IS, carried IN the artifact rather
# than in this file, because "without anyone consulting a changelog" is
# the whole claim above and a bare integer does not honor it. A reader
# holding a v2 file and a v1 parser needs to know what moved, and the
# file is the only thing they are guaranteed to have.
#
# Keyed by version and emitted for the CURRENT one only: the manifest
# describes this artifact, not the format's history, and shipping every
# past note would grow line one forever.
#
# Version 2 is Phase K. Trial lines gained `attachments` (the digests a
# comparison declared, in declaration order) and `attachments_mode`
# (how they reached the models), and the manifest gained
# `attachments_referenced`. A v1 reader that indexes trial fields
# positionally or rejects unknown keys will not absorb those, and one
# that silently ignores them will read a prompt-only artifact out of a
# file whose prompts had documents composed into them, which is the
# reading that would be wrong rather than merely partial.
EXPORT_SCHEMA_NOTES = {
    1: "the original export shape",
    2: (
        "trial lines carry attachments (declared digests, in declaration "
        "order) and attachments_mode; the manifest carries "
        "attachments_referenced. A reader that ignores them will treat a "
        "comparison whose prompt had documents composed into it as a "
        "prompt-only one."
    ),
}


def export_line(payload: dict[str, Any]) -> str:
    """One export line, serialized deterministically.

    sort_keys, and that choice is the whole point rather than a
    formatting preference. Line ORDER is fixed by the caller (task,
    repeat, position); without a rule for key order within a line, two
    honest exports of the same experiment would differ byte for byte
    whenever a dict was built in a different order, and the digest over
    them would differ too. An artifact that cannot reproduce its own
    digest is not citation-grade, whatever else it carries.

    sort_keys rather than a hand-written field order because a stated
    order is a list someone has to remember to update, and the day they
    forget it fails silently in exactly this way. Alphabetical is
    arbitrary but it is also total, and it cannot fall out of date.

    separators without spaces so the bytes are minimal and, more to the
    point, so they are not a second thing to keep stable. ensure_ascii
    off so a prompt in any script survives as itself rather than as
    escapes: the export is meant to be read.
    """
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def export_manifest(
    experiment: dict[str, Any],
    seed: int,
    thresholds: dict[str, Any] | None = None,
    attachments_referenced: bool = False,
) -> dict[str, Any]:
    """Line one: what this artifact IS.

    Everything a reader needs to know what they are holding before they
    read a single trial: which dataset by digest, which build, which
    catalog, which estimand, which seeds. A citation that names the
    artifact can be checked against the file rather than against a memory
    of how it was produced.

    thresholds is the dataset's declared cutoffs, present when the
    exporter was given the file. It is the difference between an artifact
    that can re-derive the pass rate exactly and one that can only floor
    its denominator, so the manifest states which it is rather than
    leaving the reader to infer it from an empty mapping. None and {} are
    different facts here, exactly as they are in build_report: the first
    means nobody supplied the file, the second means the file declared
    nothing.

    export_schema_change states, in the artifact, what the version
    number means. A reader whose parser predates it can find out what
    moved from the file in their hand rather than from a changelog they
    may not have; see EXPORT_SCHEMA_NOTES.

    attachments_referenced is the same kind of label one subject over:
    the artifact saying what it does NOT embed. Document bytes are never
    in an export, so a reader holding one whose trials cite digests needs
    to be told that the file is incomplete BY DESIGN and where the rest
    of it is. Without the flag they would have to read every trial line
    to find out, and an artifact should be able to describe itself from
    line one.
    """
    return {
        "thresholds": {} if thresholds is None else thresholds,
        # The artifact labels its own sufficiency. Without this a reader
        # holding an export with no thresholds cannot tell a dataset that
        # declared none from an export nobody handed the file to, and
        # those licence different claims about the pass rate inside.
        "thresholds_included": thresholds is not None,
        # Whether any trial in this artifact cites a document. False says
        # the file is complete on its own; true says the trials name
        # digests whose bytes live in the bench.db that produced this
        # export and nowhere else, so reproducing those prompts needs
        # that database and not just this file.
        #
        # A BOOLEAN AND NOT A COUNT, matching thresholds_included above:
        # the question a reader asks at line one is "is this
        # self-contained", and the digests themselves are on the trial
        # lines where they belong to a particular trial.
        "attachments_referenced": attachments_referenced,
        "type": "manifest",
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        # The bump's reason, in the file. See EXPORT_SCHEMA_NOTES: a
        # version number tells a reader their parser is old and nothing
        # else, and the artifact is meant to be readable on its own.
        "export_schema_change": EXPORT_SCHEMA_NOTES[EXPORT_SCHEMA_VERSION],
        "experiment_id": experiment["id"],
        "name": experiment["name"],
        "created_at": experiment["created_at"],
        "estimand_mode": experiment["estimand_mode"],
        # Which scorer any ranking in this artifact was computed on, or
        # null when none was declared. A leaderboard whose metric is not
        # in the file is a leaderboard nobody can check.
        "primary_metric": experiment["primary_metric"],
        "dataset_name": experiment["dataset_name"],
        "dataset_digest": experiment["dataset_digest"],
        "lineup": experiment["lineup"],
        "budget": experiment["budget"],
        "params": experiment["params"],
        "repeats": experiment["repeats"],
        "task_order_seed": experiment["task_order_seed"],
        "provider_pins": experiment["provider_pins"],
        # How the provider population was narrowed, if it was. A report
        # citing a strict run has to be able to say what "strict" meant
        # for that run.
        "quantizations": experiment.get("quantizations"),
        "halt_on_refusal": experiment["halt_on_refusal"],
        "status": experiment["status"],
        "status_detail": experiment["status_detail"],
        "app_sha": experiment["app_sha"],
        "catalog_digest": experiment["catalog_digest"],
        "data_policy": experiment["data_policy"],
        "trials_total": experiment["trials_total"],
        # The plan's arithmetic, so not_run is derivable from the artifact
        # alone. Without tasks_total a reader holding only this file can
        # count the trial lines but cannot say how many were owed, and
        # "how much of the plan actually ran" is the first question anyone
        # citing a halted experiment has to answer.
        "tasks_total": experiment["tasks_total"],
        "report_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "task",
    }


def export_trial(
    group: dict[str, Any],
    run: dict[str, Any],
    result: dict[str, Any],
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """One trial line: everything needed to re-derive any published number.

    The outcome is written out rather than left to be re-derived, and the
    fields it was derived FROM are written beside it. That is not
    redundancy: a reader on a future version of these rules can see both
    what this bench concluded and what it concluded it from, and can tell
    a rule change from a data change. data_policy rides along for the
    same reason, since it is the era marker the refusal rule gates on.

    Scores carry their own coverage-relevant fields (a null score with a
    detail is a scoring failure, not a zero), because keeping the two
    axes separate downstream is the whole reason to export scores at all
    rather than just a mean.
    """
    return {
        "type": "trial",
        "task_id": group["task_id"],
        "repeat_index": group["repeat_index"],
        "rotation_index": group["rotation_index"],
        "position": result.get("position"),
        "model": result["model"],
        "outcome": trial_outcome(result, run),
        "group_id": group["id"],
        "run_id": run["id"],
        "result_id": result["id"],
        "prompt_text": run["prompt_text"],
        # THE DOCUMENTS THIS TRIAL RAN OVER, as digests and a mode, and
        # never as content. Three separate reasons they are stated here
        # rather than left inside request_json:
        #
        # The digests ARE recoverable from request_json, because rule
        # two's placeholder names them, but recovering them means parsing
        # a record format back into a data format. That is the same
        # mistake reading controls off their badges would be, and an
        # export exists precisely so a reader does not have to do it.
        #
        # An export that carried only the composed prompt would let a
        # reader see WHAT the models read and not WHICH document it came
        # from, so two exports over the same contract could not be
        # recognized as such.
        #
        # And the mode decides what was measured: the same digest under
        # inline and under native are different experiments, so a trial
        # line that named the document without the mode would describe
        # two possible runs.
        #
        # None on every group that declared none, which includes every
        # experiment the runner produces today; see the manifest's
        # attachments_referenced.
        "attachments": group.get("attachments"),
        "attachments_mode": group.get("attachments_mode"),
        "request_json": result.get("request_json"),
        "response_text": result.get("response_text"),
        "error": result.get("error"),
        "finish_reason": result.get("finish_reason"),
        "native_finish_reason": result.get("native_finish_reason"),
        "generation_id": result.get("generation_id"),
        "provider": result.get("provider"),
        "latency_ms": result.get("latency_ms"),
        "ttft_ms": result.get("ttft_ms"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "reasoning_tokens": result.get("reasoning_tokens"),
        "cached_tokens": result.get("cached_tokens"),
        "max_tokens": result.get("max_tokens"),
        "cost_usd": result.get("cost_usd"),
        "billed_cost_usd": result.get("billed_cost_usd"),
        # The BYOK figure, verbatim. An export that carried the two
        # OpenRouter numbers and dropped the third would leave a reader
        # holding a citable artifact that cannot be reconciled against
        # the provider invoice the run was actually paid on.
        "upstream_inference_cost_usd": result.get("upstream_inference_cost_usd"),
        "app_sha": run.get("app_sha"),
        "catalog_digest": run.get("catalog_digest"),
        "data_policy": run.get("data_policy"),
        "scores": [
            {
                "scorer": s["scorer"],
                "score": s["score"],
                "passed": s["passed"],
                "detail": s["detail"],
                "judge_model": s["judge_model"],
                "judge_generation_id": s["judge_generation_id"],
                "judge_billed_cost_usd": s["judge_billed_cost_usd"],
                "blind": s["blind"],
                "self_judged": s["self_judged"],
                "created_at": s["created_at"],
                "id": s["id"],
            }
            for s in scores
        ],
    }
