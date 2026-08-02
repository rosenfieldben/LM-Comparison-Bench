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

import json
import math
import random
import statistics
from typing import Any

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


def build_report(
    experiment: dict[str, Any],
    groups: list[dict[str, Any]],
    runs_by_group: dict[int, list[dict[str, Any]]],
    scores_by_result: dict[int, list[dict[str, Any]]],
    tasks_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """One experiment's aggregates, per model.

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
    # trials[model][task_id] = list of per-trial records, so the bootstrap
    # can resample by task without a second pass over the rows.
    trials: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in lineup}
    for group in groups:
        task_id = group["task_id"]
        seen: set[str] = set()
        for run in runs_by_group.get(group["id"], []):
            for result in run["results"]:
                model = result["model"]
                if model not in trials:
                    continue
                seen.add(model)
                trials[model].setdefault(task_id, []).append(
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
        for model in lineup:
            if model not in seen:
                trials[model].setdefault(task_id, []).append(
                    {
                        "outcome": "missing",
                        "result": None,
                        "scores": [],
                        "task_id": task_id,
                    }
                )

    models = [
        _model_report(model, by_task, tasks_by_id, seed)
        for model, by_task in trials.items()
    ]
    # Ranked on the primary score mean, min-rank so ties share a place.
    ranks = min_ranks({m["model"]: m["score"]["mean"] for m in models})
    for entry in models:
        entry["rank"] = ranks[entry["model"]]
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
        "bootstrap": {
            "seed": seed,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            # Named in the report because the reader has to know which
            # unit was resampled to know what the interval claims.
            "unit": "task",
        },
        "models": sorted(models, key=lambda m: lineup.index(m["model"])),
    }


def _model_report(
    model: str,
    by_task: dict[str, list[dict[str, Any]]],
    tasks_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    flat = [t for trials in by_task.values() for t in trials]
    outcomes = {name: 0 for name in ("done", "error", "refused", "stopped", "missing")}
    for trial in flat:
        outcomes[trial["outcome"]] += 1
    attempted = len(flat)

    # Score means over EVERY trial the model was asked to do, with a
    # failed trial scoring zero. That is the failure-inclusive rule: a
    # mean over only the trials that came back is a mean over survivors,
    # and the models that fail most would look best.
    scorers = sorted({row["scorer"] for trial in flat for row in trial["scores"]})
    per_scorer = [
        _scorer_report(scorer, by_task, tasks_by_id, seed) for scorer in scorers
    ]
    primary = per_scorer[0] if per_scorer else {"mean": None, "interval": None}

    completed = [t["result"] for t in flat if t["outcome"] in COMPLETED]
    return {
        "model": model,
        "trials": {
            "attempted": attempted,
            **outcomes,
            # Rates over everything asked, not over what came back.
            "failure_rate": (
                (outcomes["error"] + outcomes["missing"]) / attempted
                if attempted
                else None
            ),
            "refusal_rate": outcomes["refused"] / attempted if attempted else None,
        },
        "scorers": per_scorer,
        "score": {"mean": primary.get("mean"), "interval": primary.get("interval")},
        "latency_ms": summarize_metric([r.get("latency_ms") for r in completed]),
        "ttft_ms": summarize_metric([r.get("ttft_ms") for r in completed]),
        "cost": _cost_totals(completed),
        # The routed-service estimand made visible. Under dynamic routing
        # the provider is chosen per call, so it is the largest confound
        # in any comparison the bench draws; counting them is what turns
        # the confound from an assumption into a column.
        "providers": _provider_counts(completed),
    }


def _scorer_report(
    scorer: str,
    by_task: dict[str, list[dict[str, Any]]],
    tasks_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """One scorer's numbers for one model, with both axes reported.

    The pass rate is passed over USABLE VERDICTS among tasks that
    declared a threshold, and it always ships with its coverage. A pass
    rate over three verdicts out of forty eligible trials is not a pass
    rate anybody should act on, and the coverage number is the only thing
    that says so; without it the denominator shrinks silently and the
    rate looks like every other rate on the page.
    """
    values_by_task: dict[str, list[float]] = {}
    scored = failed = unscored = 0
    passed = usable_verdicts = eligible = 0
    blind_rows = self_judged_rows = 0
    judge_models: set[str] = set()
    for task_id, trials in by_task.items():
        for trial in trials:
            rows = [r for r in trial["scores"] if r["scorer"] == scorer]
            state = scoring_state(trial["scores"], scorer)
            latest = rows[-1] if rows else None
            # Axis two, counted on its own.
            if state == "scored":
                scored += 1
            elif state == "scoring_failed":
                failed += 1
            else:
                unscored += 1
            if latest is not None:
                if latest.get("blind"):
                    blind_rows += 1
                if latest.get("self_judged"):
                    self_judged_rows += 1
                if latest.get("judge_model"):
                    judge_models.add(latest["judge_model"])
            # The score mean, and this is where the two axes could most
            # easily be crossed, so the rule is spelled out.
            #
            # A trial that FAILED contributes zero. That is the
            # failure-inclusive rule: an errored or aborted trial is a
            # trial that failed the task, and dropping it would average
            # over survivors and flatter the models that fail most.
            #
            # A trial that SUCCEEDED but has no usable score contributes
            # NOTHING. Its absence is axis two: nobody scored it, or the
            # judge could not answer. Scoring it zero would put the
            # judge's malfunction into the model's mean, which is exactly
            # the crossing the two axes exist to prevent, and it would be
            # invisible in the result because it looks like a bad answer.
            #
            # A trial that never ran contributes nothing either. It is an
            # absence, and scoring it zero would punish a model for an
            # experiment that was halted, which is a fact about a budget.
            if trial["outcome"] == "missing":
                continue
            value = latest.get("score") if latest else None
            if trial["outcome"] not in COMPLETED:
                values_by_task.setdefault(task_id, []).append(0.0)
            elif value is not None:
                values_by_task.setdefault(task_id, []).append(value)
            # The pass rate's population: tasks whose author declared a
            # threshold. A task without one contributes to the mean and
            # not to the rate, which is what "pass rate where a threshold
            # was declared" means.
            spec = (tasks_by_id.get(task_id) or {}).get("scorer") or {}
            if spec.get("pass_threshold") is not None:
                eligible += 1
                if latest is not None and latest.get("passed") is not None:
                    usable_verdicts += 1
                    if latest["passed"]:
                        passed += 1

    flat = [v for values in values_by_task.values() for v in values]
    total_states = scored + failed + unscored
    return {
        "scorer": scorer,
        "mean": sum(flat) / len(flat) if flat else None,
        "n": len(flat),
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
        "judge_models": sorted(judge_models),
    }


def _cost_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Billed first, estimate as the fallback, and both counts shown.

    The same rule the ceiling uses: what the platform charged beats what
    the catalog implies. Reporting the two counts beside the total is
    what keeps a total that is mostly estimate from reading like a bill.
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
    return {
        "total_usd": sum(billed) + sum(estimated),
        "billed_trials": len(billed),
        "estimated_trials": len(estimated),
        "unpriced_trials": unpriced,
    }


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
EXPORT_SCHEMA_VERSION = 1


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


def export_manifest(experiment: dict[str, Any], seed: int) -> dict[str, Any]:
    """Line one: what this artifact IS.

    Everything a reader needs to know what they are holding before they
    read a single trial: which dataset by digest, which build, which
    catalog, which estimand, which seeds. A citation that names the
    artifact can be checked against the file rather than against a memory
    of how it was produced.
    """
    return {
        "type": "manifest",
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "experiment_id": experiment["id"],
        "name": experiment["name"],
        "created_at": experiment["created_at"],
        "estimand_mode": experiment["estimand_mode"],
        "dataset_name": experiment["dataset_name"],
        "dataset_digest": experiment["dataset_digest"],
        "lineup": experiment["lineup"],
        "budget": experiment["budget"],
        "params": experiment["params"],
        "repeats": experiment["repeats"],
        "task_order_seed": experiment["task_order_seed"],
        "provider_pins": experiment["provider_pins"],
        "halt_on_refusal": experiment["halt_on_refusal"],
        "status": experiment["status"],
        "status_detail": experiment["status_detail"],
        "app_sha": experiment["app_sha"],
        "catalog_digest": experiment["catalog_digest"],
        "data_policy": experiment["data_policy"],
        "trials_total": experiment["trials_total"],
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
