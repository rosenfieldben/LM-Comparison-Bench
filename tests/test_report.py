"""Report arithmetic: two axes, honest denominators, cluster intervals.

Pure functions, so every case is a value in and a value out. The
assertions are about what the numbers MEAN, because the whole failure
mode this phase exists to prevent is a number that is arithmetically
correct and substantively misleading.
"""

import pytest

from bench.report import (
    ABORTED_ERROR,
    _cost_totals,
    _model_report,
    _provider_counts,
    _scorer_report,
    build_report,
    choose_metric,
    cluster_bootstrap,
    min_ranks,
    provenance_era,
    scoring_state,
    summarize_metric,
    trial_outcome,
)

POST_G_RUN = {"data_policy": "standard"}
PRE_G_RUN = {"data_policy": None}


def result(**overrides):
    base = {"error": None, "request_json": '{"model": "m"}', "response_text": "hi"}
    base.update(overrides)
    return base


# ---- Axis one: what happened to the trial.


def test_a_clean_trial_is_done():
    assert trial_outcome(result()) == "done"


def test_a_declared_member_with_no_row_is_missing():
    """Not omitted. A halted or stopped experiment leaves declared members
    with nothing recorded, and calling that missing rather than dropping
    it is what keeps the denominator honest."""
    assert trial_outcome(None) == "missing"


def test_review_repro_a_refusal_is_told_from_an_error_structurally():
    """request_json is written before the request leaves, on every path
    and deliberately, so its absence beside an error means nothing was
    ever sent: the spend ceiling refused it.

    Matched structurally rather than on the refusal's wording, because
    the code that writes that wording says in its own comment that the
    text is prose which may be reworded. A report keyed on the prose
    would misclassify every refusal the day someone improved the
    sentence, and would do it silently.
    """
    refused = result(error="spend ceiling reached: recorded $1.00", request_json=None)
    upstream = result(error="HTTP 500 from OpenRouter")

    # The run row is passed because the rule is era-gated; see
    # test_review_repro_a_legacy_errored_row_is_not_a_refusal for why.
    assert trial_outcome(refused, POST_G_RUN) == "refused"
    assert trial_outcome(upstream, POST_G_RUN) == "error"


def test_an_aborted_stream_is_stopped_not_errored():
    """Matched against the constant the writer uses, not a copy of its
    text, so the two cannot drift apart."""
    assert trial_outcome(result(error=ABORTED_ERROR)) == "stopped"


def test_a_reworded_refusal_message_does_not_change_the_classification():
    """The point of the structural rule, stated as a test: the same row
    with completely different prose classifies the same way."""
    for message in ("spend ceiling reached", "out of money", "refused, sorry"):
        row = result(error=message, request_json=None)
        assert trial_outcome(row, POST_G_RUN) == "refused"


# ---- Axis two: whether the bench could put a number on it.


_SEQ = {"n": 0}


def score(scorer="judge", value=1.0, judge_model=None):
    """One score row, carrying what latest_per_key orders by.

    created_at and id are not decoration in a fixture: the rule resolves
    ties by id precisely because a re-scoring pass inside one second
    would otherwise leave the order to the query planner. A fixture that
    omitted them would be testing a different function.
    """
    _SEQ["n"] += 1
    return {
        "scorer": scorer,
        "score": value,
        "judge_model": judge_model,
        "created_at": "2026-08-03T00:00:00",
        "id": _SEQ["n"],
    }


def test_an_untried_trial_is_unscored():
    """A fact about the scoring pass, not about the model."""
    assert scoring_state([], "judge", None) == "unscored"


def test_a_judge_that_returned_nothing_usable_is_scoring_failed():
    """The distinction axis two exists for. Both leave the trial without
    a number, and only one of them is a problem with the bench's own
    machinery rather than with the model."""
    assert scoring_state([score(value=None)], "judge", None) == "scoring_failed"


def test_a_usable_verdict_is_scored():
    assert scoring_state([score(value=0.5)], "judge", None) == "scored"


def test_the_latest_row_decides_the_state():
    """A re-scoring pass that fixed a failure means the trial is scored
    now, and the failed row stays as the audit trail rather than as the
    answer."""
    assert (
        scoring_state([score(value=None), score(value=1.0)], "judge", None) == "scored"
    )


def test_another_scorers_rows_do_not_answer_for_this_one():
    assert (
        scoring_state([score(scorer="exact", value=1.0)], "judge", None) == "unscored"
    )


# ---- Metrics: the count travels with the number.


def test_a_metric_summary_carries_its_own_denominator():
    """A p90 over two trials is arithmetic rather than information, and a
    reader who cannot see the denominator cannot tell which they have."""
    out = summarize_metric([10.0, 20.0, 30.0, 40.0, 50.0])

    assert out["n"] == 5
    assert out["median"] == 30.0
    # Nearest rank: the fifth-slowest of five, not an interpolation
    # between two measurements that would invent a value nobody measured.
    assert out["p90"] == 50.0


def test_a_metric_summary_of_nothing_is_empty_rather_than_zero():
    """Zero latency is a measurement. No latency is not, and reporting
    one as the other would put a fictional best time at the top of a
    ranking."""
    out = summarize_metric([])

    assert out == {"n": 0, "median": None, "p90": None}


def test_non_finite_measurements_are_dropped():
    out = summarize_metric([1.0, float("nan"), 3.0, None])

    assert out["n"] == 2


# ---- The cluster bootstrap.


def test_the_interval_brackets_the_mean():
    by_task = {f"t{i}": [float(i % 2)] for i in range(20)}

    out = cluster_bootstrap(by_task, seed=7)

    assert out["n_clusters"] == 20
    assert out["lo"] <= 0.5 <= out["hi"]
    assert out["seed"] == 7


def test_the_interval_is_reproducible_from_its_seed():
    by_task = {f"t{i}": [float(i % 3)] for i in range(15)}

    first = cluster_bootstrap(by_task, seed=42)
    second = cluster_bootstrap(by_task, seed=42)

    assert first == second
    assert cluster_bootstrap(by_task, seed=43) != first


def test_review_repro_repeats_are_resampled_as_clusters_not_as_trials():
    """The addendum's whole point, and the case that proves the right
    scheme shipped.

    Two repeats per task, perfectly correlated within each task: every
    repeat of a task agrees with itself. Resampling trials independently
    would treat 2N correlated trials as 2N independent ones and shrink
    the interval; resampling tasks keeps each pair together, so the
    interval reflects the 20 tasks that actually vary rather than the 40
    numbers that do not.

    An overconfident interval is worse than none. It is the false
    precision this phase exists to end, wearing the costume of rigor.
    """
    # Twenty tasks, each with two identical repeats. Half score 1, half 0.
    clustered = {f"t{i}": [float(i % 2), float(i % 2)] for i in range(20)}
    # The same forty numbers seen as forty independent one-trial tasks,
    # which is what a naive trial-level bootstrap would be doing.
    naive = {f"t{i}-{r}": [float(i % 2)] for i in range(20) for r in range(2)}

    cluster = cluster_bootstrap(clustered, seed=11)
    trial_level = cluster_bootstrap(naive, seed=11)

    cluster_width = cluster["hi"] - cluster["lo"]
    naive_width = trial_level["hi"] - trial_level["lo"]
    # The honest interval is visibly wider, because it is not pretending
    # the repeats were independent evidence.
    assert cluster_width > naive_width
    assert cluster["n_clusters"] == 20
    assert trial_level["n_clusters"] == 40


def test_with_one_repeat_the_two_schemes_coincide():
    """Stated as a test because the why-comment claims it: the difference
    appears exactly when repeats do the work they were added for."""
    by_task = {f"t{i}": [float(i % 2)] for i in range(20)}
    same = {f"t{i}": [float(i % 2)] for i in range(20)}

    assert cluster_bootstrap(by_task, seed=5) == cluster_bootstrap(same, seed=5)


def test_one_task_gets_no_interval_and_says_why():
    """One cluster cannot be resampled into anything but itself, so an
    interval would be a point pretending to be a range."""
    out = cluster_bootstrap({"t1": [1.0, 0.0, 1.0]}, seed=3)

    assert out["lo"] is None
    assert out["hi"] is None
    assert "too few clusters" in out["note"]


def test_no_data_gets_no_interval():
    assert cluster_bootstrap({}, seed=1)["lo"] is None
    assert cluster_bootstrap({"t1": []}, seed=1)["n_clusters"] == 0


# ---- Ties.


def test_equal_values_share_a_rank_and_the_next_one_skips():
    """Min-rank, because two models that measured identically are tied,
    and putting one above the other would report a difference that is not
    in the data."""
    out = min_ranks({"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.1})

    assert out == {"a": 1, "b": 2, "c": 2, "d": 4}


def test_a_model_with_no_measurement_is_unranked_rather_than_last():
    """Last is a claim. Unranked is the truth."""
    out = min_ranks({"a": 1.0, "b": None})

    assert out == {"a": 1, "b": None}


def test_everything_tied_shares_the_top_rank():
    assert min_ranks({"a": 0.5, "b": 0.5, "c": 0.5}) == {"a": 1, "b": 1, "c": 1}


@pytest.mark.parametrize("values", [{}, {"a": None}])
def test_ranking_nothing_ranks_nothing(values):
    assert min_ranks(values) == dict.fromkeys(values)


# ---- The era gate. A legacy error is not a refusal.


def test_review_repro_a_legacy_errored_row_is_not_a_refusal():
    """The flank the structural rule left open.

    A pre-G errored row has NULL request_json for a reason that has
    nothing to do with money: the column postdates it. The naked
    structural rule would read every February error as a spend refusal,
    in a report and in an export, and the export is forever.
    """
    legacy = {"error": "HTTP 500 from OpenRouter", "request_json": None}

    assert trial_outcome(legacy, PRE_G_RUN) == "error"
    # The same row shape from the provenance era IS a refusal, which is
    # what makes the gate a gate rather than a disabling of the rule.
    assert trial_outcome(legacy, POST_G_RUN) == "refused"


def test_an_unknown_era_is_treated_as_legacy():
    """Passing no run means the caller could not say, and a rule that
    cannot prove a refusal must not assert one. Downgrading to error is
    the conservative direction: it under-reports refusals rather than
    inventing them."""
    row = {"error": "boom", "request_json": None}

    assert trial_outcome(row) == "error"
    assert trial_outcome(row, None) == "error"


def test_the_era_marker_is_data_policy_and_standard_counts():
    """A standard-policy boot writes "standard", not NULL, so the common
    case is inside the era rather than outside it. If that were wrong the
    gate would silently disable the refusal rule for almost every real
    database."""
    assert provenance_era({"data_policy": "standard"}) is True
    assert provenance_era({"data_policy": "deny"}) is True
    assert provenance_era({"data_policy": None}) is False
    assert provenance_era(None) is False


def test_the_gate_does_not_touch_the_other_outcomes():
    """Only the refusal rule is era-sensitive; a legacy abort is still an
    abort and a legacy success is still a success."""
    assert trial_outcome({"error": None, "request_json": None}, PRE_G_RUN) == "done"
    assert (
        trial_outcome({"error": ABORTED_ERROR, "request_json": None}, PRE_G_RUN)
        == "stopped"
    )


# ---- Cost and provider columns.


def test_costs_prefer_the_billed_figure_and_show_both_counts():
    """Billed beats estimate, the same rule the ceiling uses, and the two
    counts beside the total are what stop a total that is mostly estimate
    from reading like a bill."""
    out = _cost_totals(
        [
            {"billed_cost_usd": 0.10, "cost_usd": 0.99},
            {"billed_cost_usd": None, "cost_usd": 0.02},
            {"billed_cost_usd": None, "cost_usd": None},
        ]
    )

    assert out["total_usd"] == pytest.approx(0.12)
    assert out["billed_trials"] == 1
    assert out["estimated_trials"] == 1
    assert out["unpriced_trials"] == 1


def test_provider_counts_are_ordered_by_frequency_then_name():
    """The routed-service estimand made visible. Under dynamic routing
    the provider is chosen per call, so it is the largest confound in any
    comparison the bench draws."""
    out = _provider_counts(
        [
            {"provider": "Together"},
            {"provider": "Fireworks"},
            {"provider": "Together"},
            {"provider": None},
        ]
    )

    assert list(out.items()) == [("Together", 2), ("Fireworks", 1)]


def test_a_provider_nobody_recorded_is_not_a_provider():
    """An unnamed host is an absence, not a category called None."""
    assert _provider_counts([{"provider": None}, {}]) == {}


# ---- The crossing the two axes exist to prevent.


def trial(outcome, value=None, scorer="judge", task="t1", judge_model=None):
    scores = (
        []
        if value is _MISSING
        else [score(scorer=scorer, value=value, judge_model=judge_model)]
    )
    return {"outcome": outcome, "result": {}, "scores": scores, "task_id": task}


def declared(*task_ids, scorer="judge"):
    """Which tasks declared this scorer, in the shape _scorer_report takes.

    Passed explicitly because applicability is now an input rather than
    an assumption. In build_report it is derived across EVERY model, so a
    task one model's trials were scored on counts as declared for the
    model whose trials were not: that asymmetry is exactly what makes an
    unscored trial visible as coverage rather than invisible.
    """
    return {scorer: set(task_ids)}


_MISSING = object()


def test_review_repro_an_unscored_trial_is_not_a_zero_in_the_mean():
    """The two-axes rule at its sharpest, and a defect this code had.

    A trial the model completed but nobody scored contributes NOTHING to
    the mean. Scoring it zero would put a gap in scoring coverage into
    the model's score, where it is indistinguishable from a bad answer.
    The first version of _scorer_report did exactly that, and it showed
    up as a human-rating mean of 0.375 on a model every rater had given
    four out of five: the unrated trials were being counted as failures
    by the model rather than as work the rater had not done.
    """
    by_task = {
        "t1": [trial("done", 1.0)],
        "t2": [trial("done", value=_MISSING)],
    }

    out = _scorer_report(
        "judge", None, by_task, None, seed=1, row_scored=declared("t1", "t2")
    )

    # One usable score, so the mean is that score.
    assert out["mean"] == 1.0
    assert out["n"] == 1
    # And the gap is reported, on the other axis, where it belongs.
    assert out["coverage"]["scored"] == 1
    assert out["coverage"]["unscored"] == 1


def test_a_judge_that_could_not_answer_is_not_a_zero_either():
    """Same rule, different cause. An unparseable verdict is the bench's
    machinery failing, not the model's answer being bad."""
    by_task = {"t1": [trial("done", 1.0)], "t2": [trial("done", None)]}

    out = _scorer_report(
        "judge", None, by_task, None, seed=1, row_scored=declared("t1", "t2")
    )

    assert out["mean"] == 1.0
    assert out["coverage"]["scoring_failed"] == 1
    assert out["coverage"]["scoring_failure_rate"] == 0.5


def test_a_failed_trial_is_a_zero_because_that_is_axis_one():
    """The other direction, so the fix above cannot be read as excluding
    everything awkward. A trial that errored is a trial that failed the
    task, and it belongs in the denominator at zero."""
    by_task = {"t1": [trial("done", 1.0)], "t2": [trial("error", value=_MISSING)]}

    out = _scorer_report(
        "judge", None, by_task, None, seed=1, row_scored=declared("t1", "t2")
    )

    assert out["mean"] == 0.5
    assert out["n"] == 2


def test_a_trial_that_never_ran_is_excluded_entirely():
    """An absence, not a zero. Scoring it zero would punish a model for
    an experiment that was halted, which is a fact about a budget.

    Entirely means both axes. A trial with no response is not a scoring
    gap either: calling it unscored would report coverage nobody could
    ever close, so the mean, the coverage counters and the eligible count
    all run over the same population.
    """
    by_task = {"t1": [trial("done", 1.0)], "t2": [trial("missing", value=_MISSING)]}

    out = _scorer_report(
        "judge", None, by_task, None, seed=1, row_scored=declared("t1", "t2")
    )

    assert out["mean"] == 1.0
    assert out["n"] == 1
    assert out["coverage"] == {
        "scored": 1,
        "scoring_failed": 0,
        "unscored": 0,
        "scoring_failure_rate": 0.0,
    }


# ---- The counter labels, which the export publishes forever.


def test_review_repro_a_trial_nobody_ran_is_not_an_attempt():
    """Labels are the contract, and a label that means something wider
    than it says is the same class of defect as crossing the two axes.

    The first version of _model_report reported every planned trial as
    `attempted` and put `missing` in the failure-rate numerator beside
    `error`. Halting an experiment therefore raised the failure rate of
    every model that had trials left in the plan, and a spend ceiling
    that declined a call was published as an attempt the model made.
    Both are facts about the harness arriving in the model's column,
    which is exactly what the two-axes work was for, one axis over.

    Six trials: two done, one error, one stopped, one refused, one never
    run. Attempted is four, because a refusal never left the process and
    a missing trial never ran. The failure rate is 1/4, not 2/6.
    """
    by_task = {
        "t1": [trial("done", 1.0), trial("error", value=_MISSING)],
        "t2": [
            trial("done", 1.0, task="t2"),
            trial("stopped", value=_MISSING, task="t2"),
        ],
        "t3": [
            trial("refused", value=_MISSING, task="t3"),
            trial("missing", value=_MISSING, task="t3"),
        ],
    }

    out = _model_report("m", 0, False, by_task, {}, seed=1)
    counts = out["trials"]

    assert counts["planned"] == 6
    assert counts["attempted"] == 4
    # The outcome counters still partition the plan, not the attempts.
    assert (
        sum(counts[k] for k in ("done", "error", "refused", "stopped", "missing")) == 6
    )
    # One error over four attempts. The old rule read (1 error + 1
    # missing) / 6 planned = 0.333, which is neither a rate the label
    # describes nor a number about the model.
    assert counts["failure_rate"] == pytest.approx(0.25)
    # And the refusal is a rate against the plan it was measured over.
    assert counts["refusal_rate"] == pytest.approx(1 / 6)


# ---- Sections: a scorer answers only for its own tasks.


def test_review_repro_one_scorers_tasks_stay_out_of_anothers_numbers():
    """A section used to iterate every task in the experiment.

    So a task scored by `contains` landed in the `judge` section: its
    errored trials added zeros to the judge's mean under the
    failure-inclusive rule, and its completed trials added to the judge's
    unscored coverage. Both for a scorer that was never meant to look at
    it. The two-axes work could not see this because the crossing was not
    between the axes, it was between the sections, and every axis rule
    was being applied correctly to the wrong population.

    Two tasks, one per scorer, and the judge task's trial errored. The
    judge's mean must be that single zero and nothing else.
    """
    by_task = {
        "judged": [trial("error", value=_MISSING, scorer="judge")],
        "matched": [trial("error", value=_MISSING, scorer="contains")],
    }
    row_scored = {"judge": {"judged"}, "contains": {"matched"}}

    judge = _scorer_report("judge", None, by_task, None, seed=1, row_scored=row_scored)
    contains = _scorer_report(
        "contains", None, by_task, None, seed=1, row_scored=row_scored
    )

    # One zero each, not two. Before the fix both sections read n=2.
    assert judge["n"] == 1
    assert contains["n"] == 1
    assert judge["coverage"]["scored"] + judge["coverage"]["unscored"] == 1
    assert contains["coverage"]["scored"] + contains["coverage"]["unscored"] == 1


def test_relabelling_a_task_moves_it_between_sections_and_changes_nothing_else():
    """The file is authoritative for the scorers it mentions, so the same
    rows reported against two different dataset labellings put the task in
    a different section and leave every within-section number alone.

    Same rows, twice. Only the file differs.
    """
    by_task = {
        "t1": [trial("done", 1.0, scorer="contains")],
        "t2": [trial("done", 0.0, scorer="contains")],
    }
    row_scored = {"contains": {"t1", "t2"}}
    as_contains = {
        "t1": {"scorer": {"kind": "contains"}},
        "t2": {"scorer": {"kind": "contains"}},
    }
    # t2 relabelled: the file now says judge scores it, so it leaves the
    # contains section even though its rows say contains.
    relabelled = {
        "t1": {"scorer": {"kind": "contains"}},
        "t2": {"scorer": {"kind": "judge"}},
    }

    both = _scorer_report(
        "contains", None, by_task, as_contains, 1, row_scored=row_scored
    )
    one = _scorer_report(
        "contains", None, by_task, relabelled, 1, row_scored=row_scored
    )

    assert both["n"] == 2
    assert both["mean"] == 0.5
    # t2 is gone from this section, and t1's contribution is untouched.
    assert one["n"] == 1
    assert one["mean"] == 1.0


def test_a_scorer_the_file_never_mentions_falls_back_to_its_rows():
    """Human ratings are the case. No dataset declares `human`, so
    reading the file's silence as "applies to nothing" would delete every
    human mean from any report built with a dataset path: absence read as
    denial, which is the mistake the threshold fallback exists to avoid.
    """
    by_task = {"t1": [trial("done", 4.0 / 5, scorer="human")]}
    from_file = {"t1": {"scorer": {"kind": "contains"}}}

    out = _scorer_report(
        "human", None, by_task, from_file, seed=1, row_scored={"human": {"t1"}}
    )

    assert out["n"] == 1
    assert out["mean"] == pytest.approx(0.8)


# ---- Ranking: an ordering is a claim, and it names its metric.


def section(scorer, judge_model=None, blind=0, rated=0, mean=1.0, measured=1):
    """One series as choose_metric sees it: scorer, judge, the flags the
    human composition line is built from, and how much of the mean was
    measured rather than filled in by the failure-inclusive rule.

    measured defaults to 1, meaning "somebody scored this", because these
    cases are about which metric a ranking resolves TO. measured=0 is the
    unrankable series and it has its own cases below.
    """
    return {
        "scorer": scorer,
        "judge_model": judge_model,
        "mean": mean,
        "interval": None,
        "blind": blind,
        "rated": rated,
        "measured": measured,
    }


def test_a_single_scorer_needs_no_declaration_to_rank():
    out = choose_metric(None, [{"scorers": [section("contains")]}])

    assert out["metric"] == "contains"
    assert "only scorer" in out["reason"]


def test_review_repro_two_scorers_and_no_primary_publishes_no_ranking():
    """The alphabetical accident, stated as a test.

    The old rule ranked on per_scorer[0], the first of a sorted list, so
    an experiment scored by `contains` and `judge` was ordered by
    `contains` because c sorts before j. Nobody chose that, the report did
    not say it, and adding a scorer named `accuracy` would have silently
    reordered the leaderboard. A ranking is a claim that one model did
    better, and it means nothing until somebody says better at what.
    """
    models = [{"scorers": [section("contains"), section("judge")]}]

    out = choose_metric(None, models)

    assert out["metric"] is None
    assert out["available"] == ["contains", "judge"]
    assert "better at what" in out["reason"]


def test_a_declared_primary_wins_over_the_alphabet():
    models = [{"scorers": [section("contains"), section("judge")]}]

    out = choose_metric("judge", models)

    assert out["metric"] == "judge"
    assert "declared" in out["reason"]


def test_nothing_scored_is_its_own_answer():
    out = choose_metric(None, [{"scorers": []}])

    assert out["metric"] is None
    assert out["reason"] == "nothing has been scored"


# ---- The per-outcome matrix: every outcome's treatment, locked.


def matrix_report(outcome, *, value=1.0, task="t2"):
    """One clean 1.0 trial plus one trial with the outcome under test.

    Two tasks so the interval has something to resample, and so the
    hollowing rule has a task to hollow out.
    """
    by_task = {
        "t1": [trial("done", 1.0)],
        task: [trial(outcome, value=value)],
    }
    return _scorer_report(
        "judge", None, by_task, None, seed=3, row_scored=declared("t1", task)
    )


@pytest.mark.parametrize(
    ("outcome", "in_mean", "mean", "n", "clusters"),
    [
        # The model answered. Its score is the evidence.
        ("done", True, 1.0, 2, 2),
        # The model failed the task. Zero, and the denominator keeps it:
        # a mean over survivors would flatter the models that fail most.
        ("error", True, 0.5, 2, 2),
        # The spend ceiling declined to buy the answer. Not the model.
        ("refused", False, 1.0, 1, 1),
        # The operator ended the run. Not the model.
        ("stopped", False, 1.0, 1, 1),
        # A cell that ran, with no row from this model.
        ("missing", False, 1.0, 1, 1),
        # No cell at all: the plan was abandoned before it.
        ("not_run", False, 1.0, 1, 1),
    ],
)
def test_the_outcome_matrix_locks_every_treatment(outcome, in_mean, mean, n, clusters):
    """One row per axis-one outcome, and the whole point is that this is a
    MATRIX rather than four hand-picked cases.

    The two-axes design was specified precisely for `error` and for
    unscored trials, and the tests pinned exactly those two. `refused`
    and `stopped` were left to inference and drifted: both fell through
    the failure-inclusive branch and were scored zero, so a spend ceiling
    and an operator's stop were being published as model quality. The
    rules were right; the coverage of the rules was not.

    Every outcome is listed here, with its treatment in the mean, in the
    cluster count, and in both axes' counters, so the next outcome anyone
    adds has an empty row that fails until it is filled in deliberately.
    """
    out = matrix_report(outcome, value=0.0 if outcome == "error" else 1.0)

    assert out["mean"] == pytest.approx(mean)
    assert out["n"] == n
    # The addendum: a task hollowed out by the exclusions leaves the
    # bootstrap population rather than contributing an empty cluster, and
    # the interval's stated n_clusters reflects that.
    assert out["interval"]["n_clusters"] == clusters
    # Axis two counts exactly the trials the mean was willing to speak
    # for, so the two can never disagree about their population.
    coverage = out["coverage"]
    states = coverage["scored"] + coverage["scoring_failed"] + coverage["unscored"]
    assert states == (2 if in_mean else 1)


def test_review_repro_a_refusal_beside_a_perfect_score_does_not_halve_it():
    """The reviewer's case, and the sharpest statement of the defect.

    One task answered perfectly, one trial refused by the spend ceiling.
    The refusal fell through the failure-inclusive branch and scored
    zero, so the report published 0.5: the model looked half as good
    because the operator ran out of money. The budget was being reported
    as capability.

    1.0 is the answer, with the refusal visible on axis one where it
    belongs. The report has both numbers and does not mix them.
    """
    by_task = {"t1": [trial("done", 1.0)], "t2": [trial("refused", value=_MISSING)]}

    out = _scorer_report(
        "judge", None, by_task, None, seed=3, row_scored=declared("t1", "t2")
    )

    assert out["mean"] == 1.0
    assert out["n"] == 1
    # And the refusal did not sneak in as a hollow cluster either.
    assert out["interval"]["n_clusters"] == 1


def test_a_score_row_on_a_stopped_trial_survives_and_stays_out_of_the_mean():
    """Both halves matter. The row is the audit trail of what the scoring
    pass actually did, so it is not deleted; it simply never reaches a
    published quality number, and the gate that stops it is the one gate
    every quality number sits behind."""
    by_task = {
        "t1": [trial("done", 1.0)],
        "t2": [trial("stopped", 0.0)],  # scored, and the row stays
    }

    out = _scorer_report(
        "judge", None, by_task, None, seed=3, row_scored=declared("t1", "t2")
    )

    assert out["mean"] == 1.0
    assert out["n"] == 1
    # Not counted as coverage either: it is outside the population, not
    # a gap inside it.
    assert out["coverage"]["scored"] == 1


def test_a_task_whose_every_trial_was_refused_leaves_the_bootstrap():
    """The addendum, at its sharpest: not a smaller cluster, no cluster.

    Two repeats of t2, both refused. If the task contributed an empty
    cluster the resampler would draw it and add nothing, quietly widening
    or narrowing the interval depending on the draw, and n_clusters would
    claim a population the exclusions had already hollowed out.
    """
    by_task = {
        "t1": [trial("done", 1.0), trial("done", 0.0)],
        "t2": [trial("refused", value=_MISSING), trial("refused", value=_MISSING)],
        "t3": [trial("done", 1.0), trial("done", 1.0)],
    }

    out = _scorer_report(
        "judge", None, by_task, None, seed=3, row_scored=declared("t1", "t2", "t3")
    )

    assert out["n"] == 4
    assert out["interval"]["n_clusters"] == 2


# ---- The judge key, at its call site.


def test_review_repro_two_judges_are_two_series_not_one_overwritten_row():
    """The reviewer's shape: a strict judge and a lenient one.

    latest_per_key has keyed on (scorer, judge_model) since I3 and the
    report never called it. Selection was `rows[-1]` filtered on the
    scorer alone, so the two judges fought over one slot and whichever
    row happened to be written last answered for both. A report with
    0.9 from one judge and 0.1 from the other published one number,
    attributed to neither, and nothing on the page said a second judge
    had ever run.

    Both, attributed. Averaging them is not on the table: two judges
    disagreeing is the measurement, and 0.5 is a number neither of them
    produced.
    """
    by_task = {
        "t1": [
            {
                "outcome": "done",
                "result": {},
                "task_id": "t1",
                "scores": [
                    score(value=0.9, judge_model="judge/strict"),
                    score(value=0.1, judge_model="judge/lenient"),
                ],
            }
        ]
    }

    strict = _scorer_report(
        "judge", "judge/strict", by_task, None, seed=1, row_scored=declared("t1")
    )
    lenient = _scorer_report(
        "judge", "judge/lenient", by_task, None, seed=1, row_scored=declared("t1")
    )

    assert strict["mean"] == pytest.approx(0.9)
    assert lenient["mean"] == pytest.approx(0.1)
    assert strict["judge_model"] == "judge/strict"
    assert lenient["judge_model"] == "judge/lenient"
    # Each series saw exactly its own row, so neither is reporting the
    # other's verdict as coverage either.
    assert strict["coverage"]["scored"] == 1
    assert lenient["coverage"]["scored"] == 1


def test_a_rescoring_pass_by_the_same_judge_still_takes_the_latest():
    """The full key does not mean per-row: two rows from ONE judge are
    still one series, and the newest wins by (created_at, id)."""
    first = score(value=0.2, judge_model="judge/one")
    second = score(value=0.8, judge_model="judge/one")

    out = _scorer_report(
        "judge",
        "judge/one",
        {
            "t1": [
                {
                    "outcome": "done",
                    "result": {},
                    "task_id": "t1",
                    "scores": [first, second],
                }
            ]
        },
        None,
        seed=1,
        row_scored=declared("t1"),
    )

    assert out["mean"] == pytest.approx(0.8)
    assert out["n"] == 1


def test_two_judges_under_the_chosen_metric_withhold_the_ranking():
    """Same question one level down: better according to whom. Picking a
    judge would be the alphabetical accident again, and averaging them
    would publish a number neither judge produced."""
    models = [
        {
            "scorers": [
                section("judge", judge_model="judge/strict"),
                section("judge", judge_model="judge/lenient"),
            ]
        }
    ]

    out = choose_metric("judge", models)

    assert out["metric"] is None
    assert "judge/lenient, judge/strict" in out["reason"]
    assert "averaging judges is a claim nobody made" in out["reason"]


def test_a_human_ranking_states_its_blind_composition():
    """A report ranked on ratings made BLIND is a different claim from one
    ranked on ratings made while the rater could see which model wrote
    which answer, and the second is much the weaker. The rows carry the
    flag; without this line a reader would have to join tables to learn
    which report they are holding, and almost nobody will."""
    models = [
        {"scorers": [section("human", blind=2, rated=3)]},
        {"scorers": [section("human", blind=1, rated=3)]},
    ]

    out = choose_metric(None, models)

    assert out["metric"] == "human"
    assert out["blind_ratings"] == 3
    assert out["ratings"] == 6


def test_a_non_human_ranking_carries_no_composition_line():
    """It would be noise: only human rows carry the blind flag, so the
    line would read 0 of 0 on every judged report and teach the reader to
    ignore it."""
    out = choose_metric(None, [{"scorers": [section("contains")]}])

    assert out["metric"] == "contains"
    assert "blind_ratings" not in out


def test_review_repro_the_ranking_resolves_to_a_series_not_a_scorer():
    """Found by the call-site-keys lens, on the code this branch shipped.

    choose_metric returned a scorer name and the caller took the first
    section matching it. A scorer can carry a JUDGELESS series beside a
    judged one: I3 records "no judge model was given for this scoring
    pass" as a row under scorer `judge` with a NULL judge_model, so a
    task scored once without a judge and once with one has two series and
    exactly one real judge.

    Judgeless sorts before judged, so the ranking named `judge` and then
    ordered every model on the gap row's empty mean. A leaderboard on
    None, labelled with a metric that had a real verdict sitting one row
    below it.
    """
    models = [
        {
            "model": "m",
            "scorers": [
                section("judge", judge_model=None),
                section("judge", judge_model="judge/one"),
            ],
        }
    ]

    out = choose_metric(None, models)

    assert out["metric"] == "judge"
    # The series, named. Before the fix nothing here said which.
    assert out["judge_model"] == "judge/one"


# ---- Phase I.3: a series nobody measured cannot order anything.


def report_result(result_id, model, position, *, error=None):
    """One result row as the report reads it."""
    return {
        "id": result_id,
        "model": model,
        "position": position,
        "error": error,
        "request_json": '{"model": "m"}',
        "response_text": None if error else "hi",
        "latency_ms": 10.0,
        "ttft_ms": 5.0,
        "cost_usd": None,
        "billed_cost_usd": None,
        "provider": None,
    }


def one_cell_report(results, scores_by_result, **experiment):
    """A one-task, one-repeat experiment over the given result rows.

    build_report rather than _scorer_report, because the defects in this
    section are about what the RANKING does with the sections, and a
    section-level assertion cannot see a ranking.
    """
    spec = {
        "id": 1,
        "name": "n",
        "lineup": [r["model"] for r in results],
        "estimand_mode": "routed_service",
        "dataset_name": "d.jsonl",
        "dataset_digest": "ab" * 32,
        "repeats": 1,
        "status": "done",
        "status_detail": None,
        "tasks_total": 1,
        "primary_metric": None,
    }
    spec.update(experiment)
    return build_report(
        spec,
        [{"id": 10, "task_id": "t1", "repeat_index": 0, "rotation_index": 0}],
        {10: [{"id": 100, "data_policy": "standard", "results": results}]},
        scores_by_result,
        None,
        seed=3,
    )


def gap_row(**overrides):
    """The row a scoring pass writes when it could not produce a number.

    A judge pass run with no judge model records exactly this: scorer
    `judge`, NULL judge_model, NULL score. It is a real row, so it makes
    the task the scorer's business, and it measures nothing.
    """
    row = {
        "scorer": "judge",
        "judge_model": None,
        "score": None,
        "passed": None,
        "detail": "no judge model was given for this scoring pass",
        "blind": 0,
        "self_judged": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "id": 1,
    }
    row.update(overrides)
    return row


def test_review_repro_a_failure_only_series_does_not_crown_the_arm_that_failed():
    """The reviewer's shape, and it is worse than a missing ranking.

    One arm answered and nobody scored it, so its mean is None. One arm
    errored, so the failure-inclusive rule gave that trial a 0.0 and its
    mean is 0.0: a real number, the lowest one available, and the only
    number in the series. min_ranks ranks by value and skips None, so the
    arm that failed every trial came FIRST on the leaderboard, and
    nothing on the page said why.

    The mean is not the lie. 0.0 is the correct failure-inclusive mean
    for an arm that errored. The lie is the ORDERING built on top of it,
    which claims a comparison was made when the only thing measured was
    that one arm did not answer.
    """
    results = [
        report_result(1, "m/a", 0),
        report_result(2, "m/b", 1, error="HTTP 500 from OpenRouter"),
    ]

    out = one_cell_report(results, {1: [gap_row()], 2: [gap_row(id=2)]})

    # Withheld, and the reason says which fact withheld it.
    assert out["ranking"]["metric"] is None
    assert "measured nothing on any arm" in out["ranking"]["reason"]
    # Nobody is crowned. Not the errored arm, not anybody.
    assert [m["rank"] for m in out["models"]] == [None, None]
    # And the sections still publish in full: this is coverage material,
    # which is a thing to READ, not a thing to delete.
    sections = {m["label"]: m["scorers"][0] for m in out["models"]}
    assert sections["m/a"]["mean"] is None
    assert sections["m/b"]["mean"] == 0.0
    assert sections["m/b"]["n"] == 1
    # The counter the rule turns on: n is the mean's denominator, and
    # none of it was measured.
    assert [s["measured"] for s in sections.values()] == [0, 0]
    assert sections["m/a"]["coverage"]["scoring_failed"] == 1


def test_a_declared_primary_that_measured_nothing_is_withheld_by_name():
    """Declaring the metric does not conjure a measurement. The
    declaration answers "better at what"; it cannot answer "better by how
    much" when the answer is that nobody looked."""
    results = [
        report_result(1, "m/a", 0),
        report_result(2, "m/b", 1, error="HTTP 500 from OpenRouter"),
    ]

    out = one_cell_report(
        results,
        {1: [gap_row()], 2: [gap_row(id=2)]},
        primary_metric="judge",
    )

    assert out["ranking"]["metric"] is None
    assert "judge measured nothing on any arm" in out["ranking"]["reason"]
    assert [m["rank"] for m in out["models"]] == [None, None]


def test_one_measured_arm_is_enough_to_rank_the_series():
    """The rule is per SERIES, not per arm. Once anybody has measured it,
    an arm that failed everything ranks below them at 0.0, which is the
    failure-inclusive rule doing exactly its job."""
    results = [
        report_result(1, "m/a", 0),
        report_result(2, "m/b", 1, error="HTTP 500 from OpenRouter"),
    ]

    out = one_cell_report(
        results,
        {1: [gap_row(score=0.75, judge_model="j/one")], 2: [gap_row(id=2)]},
    )

    assert out["ranking"]["metric"] == "judge"
    assert out["ranking"]["judge_model"] == "j/one"
    ranks = {m["label"]: m["rank"] for m in out["models"]}
    assert ranks == {"m/a": 1, "m/b": None}
