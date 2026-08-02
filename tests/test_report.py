"""Report arithmetic: two axes, honest denominators, cluster intervals.

Pure functions, so every case is a value in and a value out. The
assertions are about what the numbers MEAN, because the whole failure
mode this phase exists to prevent is a number that is arithmetically
correct and substantively misleading.
"""

import pytest

from bench.report import (
    ABORTED_ERROR,
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


def score(scorer="judge", value=1.0):
    return {"scorer": scorer, "score": value}


def test_an_untried_trial_is_unscored():
    """A fact about the scoring pass, not about the model."""
    assert scoring_state([], "judge") == "unscored"


def test_a_judge_that_returned_nothing_usable_is_scoring_failed():
    """The distinction axis two exists for. Both leave the trial without
    a number, and only one of them is a problem with the bench's own
    machinery rather than with the model."""
    assert scoring_state([score(value=None)], "judge") == "scoring_failed"


def test_a_usable_verdict_is_scored():
    assert scoring_state([score(value=0.5)], "judge") == "scored"


def test_the_latest_row_decides_the_state():
    """A re-scoring pass that fixed a failure means the trial is scored
    now, and the failed row stays as the audit trail rather than as the
    answer."""
    assert scoring_state([score(value=None), score(value=1.0)], "judge") == "scored"


def test_another_scorers_rows_do_not_answer_for_this_one():
    assert scoring_state([score(scorer="exact", value=1.0)], "judge") == "unscored"


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
