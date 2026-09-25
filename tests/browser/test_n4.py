"""Phase N4 browser tests: scoring and the judge from the page.

Score a finished experiment, with a judge chosen from the catalog when its
recorded dataset has judge tasks, against the real Score door and the
stub. What the page SENT is read off the wire and what the server DID is
read back through the report; the critical path's Score is in
test_n3.py beside the rest of that path, and the status walk there
proves the Score row absent while created or running.

A JUDGE'S CALLS ARE PAID, so Score is the second door that moves money
and these proofs treat it as N3 treated Start: the body is the recorded
digest and, only when that dataset has judge tasks, the judge chosen; a
refusal is the door's sentence with the button left live; and the hidden
row (a 409 for a created or running experiment) and the greyed Score for
an unstored or unreadable dataset (a 422) are proved courtesies by the
door refusing on its own. Score's other three greys refuse a request the
door ACCEPTS (the scorers still being read, the read failed, a judge
dataset with no judge chosen), because a judge-less pass over judge
tasks records "no judge model was given" for good and without the
scorers the page cannot tell whether a judge is needed; the door's 202
to those bodies is proved in tests/test_api.py by
test_every_score_nudge_is_named_and_every_body_is_what_the_door_takes.

EVERY PAGE HERE RUNS IN Asia/Kolkata (UTC+05:30), so a stamp that said
UTC while printing local time would be off by five and a half hours and
fail, whatever zone the host is in.

THE BENCH HAS ONE SCORING SLOT, NO DOOR SAYS WHEN A PASS ENDS, AND NONE
CAN STOP ONE. Every test that scores takes the `scorings` fixture, whose
teardown waits for the slot (test_n3.py's idle probe) and fails loudly if
it stays busy; a pass is held open only by the stub's judge gate, which
the `judge_gate` fixture always releases before that wait.
"""

import hashlib
import json
import re
import sqlite3
from datetime import UTC as UTC_ZONE
from datetime import datetime

import httpx
import pytest
from playwright.sync_api import expect
from test_i4 import check_all_chips, column_texts
from test_n import (
    ABORTED_RESOURCE,
    REFUSED_RESOURCE,
    arm,
    entry_for,
    hold,
    open_datasets,
    settle,
    unique,
    wait_held,
)
from test_n3 import (
    CONFLICT_RESOURCE,
    DONE_TIMEOUT,
    LOST_RESOURCE,
    MISSING_RESOURCE,
    api_experiment,
    create,
    open_experiments,
    ready,
    record,
    row_for,
    scoring_busy,
    stored_digest,
    swept,
    wait_scoring_idle,
    wait_status,
)

pytestmark = pytest.mark.browser

HELD_JUDGE = "stub/judge-held"
SERVER_ERROR = "Failed to load resource: the server responded with a status of 500"
# A deterministic task, so a pass over it calls no model and is free, and
# a judge task, whose pass calls the judge once per trial.
PLAIN = [{"id": "e1", "reference": "x", "scorer": {"kind": "exact"}}]
JUDGED = [{"id": "j1", "rubric": "grade it", "scorer": {"kind": "judge"}}]
UTC = r"\d\d:\d\d:\d\d UTC"
# Why Score waits, as the page says it (BenchLib.scoreNudge).
JUDGE_LESS_PASS = (
    "because a pass without one records every judge task as a scoring failure, "
    "and that record does not rewrite"
)
WAITS_FOR_JUDGE = "Score waits: choose a judge, " + JUDGE_LESS_PASS
WAITS_FOR_CATALOG = "Score waits: choose a judge once the catalog has loaded, " + (
    JUDGE_LESS_PASS
)
WAITS_FOR_READ = (
    "Score waits: its dataset could not be read; Retry asks the store again"
)
PAGE_ZONE_OFFSET = -330  # Date.getTimezoneOffset() in Asia/Kolkata


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """This module's pages keep a clock that is not UTC, so Score's UTC
    stamp is proved to be UTC rather than the host's time with a label."""
    return {**browser_context_args, "timezone_id": "Asia/Kolkata"}


@pytest.fixture(autouse=True)
def collectors(page):
    """Arm the collectors before the test and assert them after, as
    test_n3.py does; yields the console-error prefixes a test allows."""
    errors = arm(page)
    allowed = [REFUSED_RESOURCE]
    yield allowed
    assert page.evaluate("window.__csp || []") == []
    assert [e for e in errors if not e.startswith(tuple(allowed))] == []


@pytest.fixture
def judge_gate(stub_url, scorings):
    """The held judge's gate, armed for the test and released after it,
    pass or fail. It depends on `scorings`, so it is torn down first: the
    gate opens before the wait for the scoring slot begins."""

    def post(action):
        httpx.post(
            stub_url + "/_test/judge-gate", json={"action": action}, trust_env=False
        ).raise_for_status()

    post("arm")
    yield post
    post("release")


def finished(page, bench_url, rows, lineup=("stub/fast",), name=None):
    """An experiment over its own stored dataset, created, started and run
    to done through the API, as another tab or a script would; returns
    its id and its recorded digest."""
    _, digest = stored_digest(page, bench_url, rows)
    eid = api_experiment(page, bench_url, digest, list(lineup), name or unique("xp"))
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok
    assert wait_status(page, bench_url, eid)["status"] == "done"
    return eid, digest


def score_series(page, bench_url, eid):
    """The (scorer, judge) series the report holds for an experiment."""
    report = page.request.get(f"{bench_url}/experiments/{eid}/report").json()
    return sorted(
        {
            (s["scorer"], s["judge_model"])
            for m in report["models"]
            for s in m["scorers"]
        },
        key=str,
    )


def press_and_capture(page, eid):
    """Press Score and return the JSON body its request carried."""
    with page.expect_request(
        lambda r: r.url.endswith(f"/experiments/{eid}/score") and r.method == "POST"
    ) as sent:
        page.get_by_test_id("experiment-score").click()
    return json.loads(sent.value.post_data)


def report_reads(page, eid):
    """Every GET of an experiment's report the page makes from now on."""
    reads = []
    page.on(
        "request",
        lambda r: (
            reads.append(r.url)
            if r.method == "GET" and r.url.endswith(f"/experiments/{eid}/report")
            else None
        ),
    )
    return reads


# ---- The door is the rule.


def test_the_score_door_refuses_what_the_hidden_row_would_not_send(
    page, bench, bench_url, runs, scorings
):
    """WINDOW: the Score row for a created and a running experiment, and
    POST /experiments/{id}/score sent for each anyway, with its recorded
    digest; then the report's series once the run has finished and the
    scoring slot is free.

    THE SECOND HALF OF THE COMMISSION'S PAIR (the first, in test_n3.py's
    status walk, proves the row absent). A hidden button is not a rule:
    the door refuses both in its own words, and nothing was scored. The
    running experiment has already finished a trial when it is refused,
    so a door that let it through would have had results to score, and
    the body is the one the page would send, not an empty one the door
    would refuse for another reason."""
    _, digest = stored_digest(
        page,
        bench_url,
        [
            {"id": f"d{n}", "reference": "x", "scorer": {"kind": "exact"}}
            for n in range(4)
        ],
    )
    created = api_experiment(page, bench_url, digest, ["stub/fast"], unique("created"))
    running = api_experiment(page, bench_url, digest, ["stub/slow"], unique("running"))
    runs.append(running)
    assert page.request.post(
        f"{bench_url}/experiments/{running}/start", data={"dataset_digest": digest}
    ).ok
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, running).click()
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-3] "), timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("experiment-score-row")).to_be_hidden()
    row_for(page, created).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    expect(page.get_by_test_id("experiment-score-row")).to_be_hidden()

    refusals = {
        eid: page.request.post(
            f"{bench_url}/experiments/{eid}/score", data={"dataset_digest": digest}
        )
        for eid in (created, running)
    }

    for eid, status in ((created, "created"), (running, "running")):
        assert refusals[eid].status == 409
        assert refusals[eid].json()["detail"] == (
            f"experiment {eid} is {status}; score it once its trials have "
            "finished, so the pass sees every result"
        )
    wait_status(page, bench_url, running)
    wait_scoring_idle(page, bench_url)
    assert score_series(page, bench_url, created) == []
    assert score_series(page, bench_url, running) == []


def test_another_pass_is_the_servers_sentence_and_score_stays_live(
    page, bench, bench_url, collectors, judge_gate
):
    """WINDOW: a scoring pass on experiment A started through the API with
    the stub's held judge, so it holds the bench's one scoring slot for
    real; Score pressed in the page for B and then for A; the gate
    released; B's Score pressed again.

    The door refuses both presses with "a scoring pass for experiment A
    is running", for B and for A alike, and the page prints that sentence
    as a refusal, stamped with the time it was given, with Score LIVE:
    the server is the one that knows when the other pass ends. Once it
    has, the same press on B is a 202. Pre-state: the probe reads the
    slot busy before the press and still busy after it, so the refusal is
    the held pass's and the press started nothing."""
    collectors.append(CONFLICT_RESOURCE)
    a, a_digest = finished(page, bench_url, JUDGED, name=unique("holding"))
    b, _ = finished(page, bench_url, PLAIN, name=unique("waiting"))
    held = page.request.post(
        f"{bench_url}/experiments/{a}/score",
        data={"dataset_digest": a_digest, "judge_model": HELD_JUDGE},
    )
    assert held.status == 202
    bench(["stub/fast"])
    open_experiments(page)
    assert scoring_busy(page, bench_url)
    score = page.get_by_test_id("experiment-score")
    message = page.get_by_test_id("experiment-action-msg")
    sentence = f"a scoring pass for experiment {a} is running"

    row_for(page, b).click()
    expect(score).to_be_enabled()
    assert page.evaluate("new Date().getTimezoneOffset()") == PAGE_ZONE_OFFSET
    before = datetime.now(UTC_ZONE).replace(microsecond=0)
    score.click()

    expect(message).to_have_text(re.compile(rf"^not scored at {UTC}: {sentence}$"))
    after = datetime.now(UTC_ZONE)
    stamped = re.search(r"(\d\d):(\d\d):(\d\d) UTC", message.inner_text())
    seconds = [int(part) for part in stamped.groups()]
    stamp = seconds[0] * 3600 + seconds[1] * 60 + seconds[2]
    low = before.hour * 3600 + before.minute * 60 + before.second
    span = int((after - before).total_seconds()) + 1
    assert (stamp - low) % 86400 <= span, (stamped.group(0), before, after)
    expect(message).to_have_attribute("data-state", "refused")
    expect(score).to_be_enabled()
    assert scoring_busy(page, bench_url)
    row_for(page, a).click()
    page.get_by_test_id("experiment-judge").select_option("stub/fast")
    score.click()
    expect(message).to_have_text(re.compile(rf"^not scored at {UTC}: {sentence}$"))
    expect(score).to_be_enabled()
    judge_gate("release")
    wait_scoring_idle(page, bench_url)
    row_for(page, b).click()
    score.click()
    expect(message).to_have_text(re.compile(rf"^a scoring pass was started at {UTC}; "))
    expect(message).to_have_attribute("data-state", "")


def test_score_sends_the_recorded_dataset_and_a_judge_only_for_judge_tasks(
    page, bench, bench_url, scorings
):
    """WINDOW: two finished experiments, EJ over a dataset with a judge
    task and EP over one without; the Datasets panel on the OTHER dataset
    when each row is selected, moved to the row's own while it is, and
    back on the OTHER dataset at each press; the Score row, the body
    Score sends, the 202 rendered, and the report's series once each
    pass has ended.

    What Score offers and sends comes from the experiment's RECORDED
    dataset, never from the Datasets selection: the door checks the
    digest against the record, and the pass scores the record's tasks.
    EJ shows the judge select, says it pays the judge, waits until one is
    chosen, and sends exactly the digest and the judge. EP shows no judge,
    says it is free, and sends exactly the digest: the door would accept
    a judge there and record nothing of it, so only the page keeps it off
    the wire."""
    j_name, j_digest = stored_digest(page, bench_url, JUDGED)
    p_name, p_digest = stored_digest(page, bench_url, PLAIN)
    ej = api_experiment(page, bench_url, j_digest, ["stub/fast"], unique("judged"))
    ep = api_experiment(page, bench_url, p_digest, ["stub/fast"], unique("plain"))
    for eid, digest in ((ej, j_digest), (ep, p_digest)):
        assert page.request.post(
            f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
        ).ok
        wait_status(page, bench_url, eid)
    bench(["stub/fast"])
    check_all_chips(page)
    open_datasets(page)
    open_experiments(page)
    score = page.get_by_test_id("experiment-score")
    judge_field = page.get_by_test_id("experiment-judge")
    nudge = page.get_by_test_id("experiment-score-nudge")

    message = page.get_by_test_id("experiment-action-msg")
    started = re.compile(rf"^a scoring pass was started at {UTC}; ")

    entry_for(page, p_name).click()
    expect(entry_for(page, p_name)).to_have_attribute("aria-pressed", "true")
    row_for(page, ej).click()
    expect(score).to_have_text("Score · pays the judge")
    expect(judge_field).to_be_visible()
    expect(score).to_be_disabled()
    expect(nudge).to_have_text(WAITS_FOR_JUDGE)
    entry_for(page, j_name).click()
    expect(score).to_have_text("Score · pays the judge")
    entry_for(page, p_name).click()
    expect(entry_for(page, p_name)).to_have_attribute("aria-pressed", "true")
    judge_field.select_option("stub/slow")
    expect(score).to_be_enabled()
    expect(nudge).to_have_text("")
    assert press_and_capture(page, ej) == {
        "dataset_digest": j_digest,
        "judge_model": "stub/slow",
    }
    # The 202 rendered: the door has taken the pass, so the probe below
    # cannot read the slot free before the pass exists.
    expect(message).to_have_text(started)
    wait_scoring_idle(page, bench_url)

    entry_for(page, j_name).click()
    expect(entry_for(page, j_name)).to_have_attribute("aria-pressed", "true")
    row_for(page, ep).click()
    expect(score).to_have_text("Score · free")
    expect(judge_field).to_be_hidden()
    expect(score).to_be_enabled()
    entry_for(page, p_name).click()
    expect(score).to_have_text("Score · free")
    entry_for(page, j_name).click()
    expect(entry_for(page, j_name)).to_have_attribute("aria-pressed", "true")
    assert press_and_capture(page, ep) == {"dataset_digest": p_digest}
    expect(message).to_have_text(started)
    wait_scoring_idle(page, bench_url)
    assert score_series(page, bench_url, ep) == [("exact", None)]
    assert score_series(page, bench_url, ej) == [("judge", "stub/slow")]


def test_a_judge_is_chosen_for_one_experiment_only(page, bench, bench_url, scorings):
    """WINDOW: a judge chosen on experiment A, over a dataset with judge
    tasks; then B, over another dataset with judge tasks, selected; then
    A again.

    NOTHING IS PRE-FILLED, and a paid choice least of all: a judge chosen
    for A is not carried to B, where it would arm a press nobody chose,
    nor kept for A once the selection has moved. Each shows the blank
    choice and waits for one."""
    a, _ = finished(page, bench_url, JUDGED, name=unique("first"))
    b, _ = finished(page, bench_url, JUDGED, name=unique("second"))
    bench(["stub/fast"])
    open_experiments(page)
    judge_field = page.get_by_test_id("experiment-judge")
    score = page.get_by_test_id("experiment-score")
    row_for(page, a).click()
    judge_field.select_option("stub/fast")
    expect(score).to_be_enabled()

    row_for(page, b).click()

    expect(judge_field).to_have_value("")
    expect(score).to_be_disabled()
    expect(page.get_by_test_id("experiment-score-nudge")).to_have_text(WAITS_FOR_JUDGE)
    row_for(page, a).click()
    expect(judge_field).to_have_value("")
    expect(score).to_be_disabled()


# ---- The judge select is the catalog.


def test_the_judge_select_is_the_whole_catalog_unfiltered(
    page, bench, bench_url, scorings
):
    """WINDOW: the judge select's options with GET /models answered by a
    catalog the test writes (an unpriced entry, one with no name, a
    capped one, one in the lineup, in no sorted order), and the chosen
    judge after the composer has been typed in.

    FILTERED BY NOTHING: the options are the blank choice and then every
    catalog id in the catalog's order, whatever its price, name, cap or
    place in the lineup; a filter on any of those would drop one here.
    The chosen option is the same node after the composer's own changes
    (a keystroke, a lineup chip): the options are rebuilt only when the
    catalog changes, since a rebuild closes a list the person has open."""
    a, _ = finished(page, bench_url, JUDGED)
    catalog = page.request.get(bench_url + "/models").json()
    base = catalog["models"][0]
    written = [
        dict(
            base,
            id="zeta/unpriced",
            name="unpriced",
            prompt_price=None,
            completion_price=None,
        ),
        dict(base, id="alpha/nameless", name=None),
        dict(base, id="stub/fast", name="in the lineup"),
        dict(base, id="mid/capped", name="capped", max_completion_tokens=64),
    ]
    page.route(
        "**/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(dict(catalog, models=written)),
        ),
    )
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, a).click()
    judge_field = page.get_by_test_id("experiment-judge")
    options = judge_field.locator("option")

    expect(options).to_have_count(5)

    assert [o.get_attribute("value") for o in options.all()] == [
        "",
        "zeta/unpriced",
        "alpha/nameless",
        "stub/fast",
        "mid/capped",
    ]
    judge_field.select_option("mid/capped")
    chosen = options.nth(4).element_handle()
    assert chosen.evaluate("o => o.isConnected && o.selected")
    page.get_by_test_id("prompt-input").fill("typing in the composer")
    page.get_by_test_id("lineup-chip").first.click()
    expect(judge_field).to_have_value("mid/capped")
    expect(page.get_by_test_id("experiment-score")).to_be_enabled()
    assert chosen.evaluate("o => o.isConnected && o.selected"), (
        "the composer's changes rebuilt the judge options"
    )


def test_the_judge_note_follows_the_catalog_from_loading_to_loaded(
    page, bench, bench_url, scorings
):
    """WINDOW: the judge select and its note on a judge-task experiment
    while GET /models is held, and after it is answered.

    "Not yet" is not "not available": while the catalog is loading the
    note says so, offers no judge, and Score waits for the catalog rather
    than asking for a judge it cannot offer; when it arrives the select
    fills with the catalog, Score asks for a judge, and the note says the
    list is unfiltered and that no door enforces the README's rule about
    mandatory reasoning, naming what such a judge may do without claiming
    which, since none of it is measured."""
    a, _ = finished(page, bench_url, JUDGED)
    held = hold(page, "**/models", "GET")
    bench(["stub/fast"])
    wait_held(page, held)
    open_experiments(page)
    row_for(page, a).click()
    note = page.get_by_test_id("experiment-judge-note")
    nudge = page.get_by_test_id("experiment-score-nudge")
    options = page.get_by_test_id("experiment-judge").locator("option")
    expect(note).to_have_text("the catalog is still loading")
    expect(nudge).to_have_text(WAITS_FOR_CATALOG)
    expect(options).to_have_count(1)

    held.pop().continue_()

    expect(note).to_have_text(
        "the whole catalog, unfiltered: the bench does not check a judge, and "
        "no door enforces the README's rule that a judge route must not have "
        "mandatory reasoning (derived, not measured); such a judge may give "
        "verdicts, be refused, or be billed for none"
    )
    expect(nudge).to_have_text(WAITS_FOR_JUDGE)
    ids = [m["id"] for m in page.request.get(bench_url + "/models").json()["models"]]
    expect(options).to_have_count(len(ids) + 1)
    assert [o.get_attribute("value") for o in options.all()] == ["", *ids]


def test_an_unavailable_catalog_offers_no_judge_and_says_where_to_go(
    page, bench, bench_url, collectors, scorings
):
    """WINDOW: the judge select, its note and Score on a judge-task
    experiment when GET /models fails.

    The page offers only the blank choice, says the catalog is not
    available and that the API takes a judge_model, and Score waits,
    saying why (not asking for a judge it cannot offer), rather than
    sending a judge-less pass the door would accept and record as
    unscored for good."""
    collectors.append(SERVER_ERROR)
    a, _ = finished(page, bench_url, JUDGED)
    page.route("**/models", lambda route: route.fulfill(status=500, body="down"))
    bench(["stub/fast"])
    open_experiments(page)

    row_for(page, a).click()

    expect(page.get_by_test_id("experiment-judge-note")).to_have_text(
        "the catalog is not available, so no judge can be offered here; score "
        "it through the API with judge_model"
    )
    options = page.get_by_test_id("experiment-judge").locator("option")
    expect(options).to_have_count(1)
    assert options.first.get_attribute("value") == ""
    expect(page.get_by_test_id("experiment-score")).to_be_disabled()
    expect(page.get_by_test_id("experiment-score-nudge")).to_have_text(
        "Score waits: no judge can be chosen here, because the catalog is not "
        "available; score it through the API with judge_model"
    )


# ---- What the page knows about the recorded dataset.


def test_a_run_that_finishes_under_the_watch_learns_its_dataset(
    page, bench, bench_url, runs, scorings
):
    """WINDOW: a fresh page that selects an experiment while it runs over
    a dataset with a judge task, and its Score row once the run finishes
    under the watch, without the row being selected again.

    The store is asked on every selection, running included, because a
    running experiment becomes a finished one without being selected
    again. The row that appears on the last frame knows its dataset: the
    judge select, the paid label, and Score waiting for a judge, not for
    a question that was never asked."""
    _, digest = stored_digest(
        page,
        bench_url,
        [
            {"id": f"w{n}", "rubric": "grade", "scorer": {"kind": "judge"}}
            for n in range(3)
        ],
    )
    eid = api_experiment(page, bench_url, digest, ["stub/slow"], unique("watched"))
    runs.append(eid)
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok
    asked = []
    page.on(
        "request",
        lambda r: (
            asked.append(r.url) if r.url.endswith(f"/datasets/{digest}") else None
        ),
    )
    bench(["stub/fast"])
    open_experiments(page)

    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-status")).to_have_text("running")
    expect(page.get_by_test_id("experiment-score-row")).to_be_hidden()
    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "done", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("experiment-score-row")).to_be_visible()
    expect(page.get_by_test_id("experiment-judge")).to_be_visible()
    expect(page.get_by_test_id("experiment-score")).to_have_text(
        "Score · pays the judge"
    )
    expect(page.get_by_test_id("experiment-score-nudge")).to_have_text(WAITS_FOR_JUDGE)
    assert len(asked) == 1


def test_score_waits_while_the_dataset_is_being_read_and_after_a_failure(
    page, bench, bench_url, collectors, scorings
):
    """WINDOW: a finished judge-task experiment selected while GET
    /datasets/{digest} is held, then answered 500; Retry pressed from the
    keyboard with the question held again, then answered by the store;
    and what has focus after Retry.

    Score's body and label depend on the scorers, so unlike Start it
    WAITS while they are unknown: plain "Score", greyed, saying it is
    reading (no Retry: nothing has failed); then saying the read failed,
    with Retry beside it; Retry asks the store the same question again,
    hides while it is out and puts focus on the experiment's row, and the
    Score row then reads the answer. Pressing without the scorers could
    send a judge-less pass over judge tasks, which the door accepts and
    records for good."""
    collectors.extend([SERVER_ERROR, "bench: checking a stored dataset failed"])
    a, digest = finished(page, bench_url, JUDGED)
    bench(["stub/fast"])
    held = hold(page, f"**/datasets/{digest}", "GET")
    open_experiments(page)
    score = page.get_by_test_id("experiment-score")
    nudge = page.get_by_test_id("experiment-score-nudge")
    retry = page.get_by_test_id("experiment-score-retry")

    row_for(page, a).click()
    wait_held(page, held)

    expect(score).to_have_text("Score")
    expect(score).to_be_disabled()
    expect(nudge).to_have_text(
        "Score waits: reading which scorers its dataset declares"
    )
    expect(retry).to_be_hidden()
    held.pop().fulfill(status=500, body="no")
    expect(nudge).to_have_text(WAITS_FOR_READ)
    expect(score).to_be_disabled()
    expect(retry).to_be_visible()
    retry.focus()
    page.keyboard.press("Enter")
    wait_held(page, held)
    expect(nudge).to_have_text(
        "Score waits: reading which scorers its dataset declares"
    )
    expect(retry).to_be_hidden()
    assert page.evaluate("() => document.activeElement.dataset.id") == str(a)
    held.pop().continue_()
    expect(score).to_have_text("Score · pays the judge")
    expect(nudge).to_have_text(WAITS_FOR_JUDGE)
    expect(retry).to_be_hidden()


def test_a_dataset_read_from_a_file_is_scored_through_the_api(
    page, bench, bench_url, tmp_path, collectors, scorings
):
    """WINDOW: a finished experiment whose dataset was read from a file by
    path, selected; and the Score door sent the page's request anyway.

    Score sends the recorded digest, and the store does not hold this
    one, so the door always refuses it: the page greys Score and says to
    score it through the API by path, with a judge if its tasks need one.
    The greyed button is a courtesy; the door refuses the digest on its
    own."""
    collectors.append(MISSING_RESOURCE)
    path = tmp_path / "scored-by-path.jsonl"
    path.write_text(
        '{"id": "f1", "prompt": "p", "reference": "x", "scorer": {"kind": "exact"}}\n',
        encoding="utf-8",
    )
    eid = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_path": str(path)}
    ).ok
    digest = wait_status(page, bench_url, eid)["dataset_digest"]
    bench(["stub/fast"])
    open_experiments(page)

    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-score-nudge")).to_have_text(
        "Score waits: its dataset was read from a file and is not stored here, "
        "so the page cannot name it; score it through the API with its "
        "dataset_path, and judge_model if any of its tasks is judged"
    )
    expect(page.get_by_test_id("experiment-score")).to_be_disabled()
    refused = page.request.post(
        f"{bench_url}/experiments/{eid}/score", data={"dataset_digest": digest}
    )
    assert refused.status == 422
    assert refused.json()["detail"].startswith(f"no stored dataset has digest {digest}")


# ---- The answer, the report, and the one press.


def test_the_report_is_read_again_after_the_202(page, bench, bench_url, scorings):
    """WINDOW: GET /experiments/{id}/report requests while Score's POST is
    held and after it is answered 202.

    After scoring, the report opens: select() had already opened it, so
    the proof is a NEW read of the report, made after the answer and not
    before it, and the line beside Score says when the pass started and
    that no door says when it ends or whether it failed."""
    eid, _ = finished(page, bench_url, PLAIN)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    expect(page.get_by_test_id("report-banner")).to_be_visible()
    held = hold(page, f"**/experiments/{eid}/score", "POST")
    reads = report_reads(page, eid)
    page.get_by_test_id("experiment-score").click()
    wait_held(page, held)
    page.wait_for_timeout(200)
    assert reads == []

    with page.expect_request(
        lambda r: r.method == "GET" and r.url.endswith(f"/experiments/{eid}/report")
    ):
        held.pop().continue_()

    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile(
            rf"^a scoring pass was started at {UTC}; no door says when it ends "
            r"or whether it failed, so select the experiment again to read what "
            r"it has scored since$"
        )
    )


def test_a_late_score_answer_does_not_open_its_report_under_another(
    page, bench, bench_url, collectors, scorings
):
    """WINDOW: Score pressed for B with its POST held, C selected while it
    is held, the answer released (and, in the second round, lost), and B
    selected again.

    An answer belongs to the experiment it was asked about: B's report is
    not read under C's selection, the panel holds one banner and it names
    C, C's line says nothing of B, and B, selected again, shows its own
    line."""
    collectors.extend(
        [LOST_RESOURCE, ABORTED_RESOURCE, "bench: scoring an experiment failed"]
    )
    b, _ = finished(page, bench_url, PLAIN, name=unique("scored"))
    c, _ = finished(page, bench_url, PLAIN, name=unique("elsewhere"))
    c_name = record(page, bench_url, c)["name"]
    bench(["stub/fast"])
    open_experiments(page)
    message = page.get_by_test_id("experiment-action-msg")
    banner = page.get_by_test_id("report-banner")

    for release, said in (
        (lambda route: settle(page, route), r"^a scoring pass was started at "),
        (
            lambda route: route.abort(),
            rf"^no answer came back at {UTC} \(.+\); no door says whether a pass started$",
        ),
    ):
        row_for(page, b).click()
        held = hold(page, f"**/experiments/{b}/score", "POST")
        page.get_by_test_id("experiment-score").click()
        wait_held(page, held)
        row_for(page, c).click()
        expect(banner).to_contain_text(c_name)
        reads = report_reads(page, b)

        release(held.pop())
        page.unroute(f"**/experiments/{b}/score")
        wait_scoring_idle(page, bench_url)
        page.wait_for_timeout(200)

        assert reads == []
        expect(banner).to_have_count(1)
        expect(banner).to_contain_text(c_name)
        expect(message).to_have_text("")
        row_for(page, b).click()
        expect(message).to_have_text(re.compile(said))


def test_score_is_pressed_once_and_focus_goes_to_the_row(
    page, bench, bench_url, scorings
):
    """WINDOW: Score pressed from the keyboard with its POST held, pressed
    again while held, the answer released, and what has focus after.

    One press, one POST: Score is greyed while its request is out. Focus
    then goes to the experiment's row, not back to Score, because a
    second Enter there would pay a judge again."""
    eid, _ = finished(page, bench_url, PLAIN)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    held = hold(page, f"**/experiments/{eid}/score", "POST")
    score = page.get_by_test_id("experiment-score")
    expect(score).to_be_enabled()
    score.focus()
    page.keyboard.press("Enter")
    wait_held(page, held)
    expect(score).to_be_disabled()
    page.evaluate(
        "() => document.querySelector('[data-testid=experiment-score]').click()"
    )
    page.wait_for_timeout(200)
    assert len(held) == 1

    held.pop().continue_()

    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile(r"^a scoring pass was started at ")
    )
    expect(score).to_be_enabled()
    assert page.evaluate("() => document.activeElement.dataset.id") == str(eid)


def test_a_double_click_on_score_is_one_press(page, bench, bench_url, scorings):
    """WINDOW: two real mouse clicks on Score 150 ms apart (the second
    carrying a click count of 2, as a double-click's does), on a finished
    judge-task experiment with a judge chosen and its row in view, and
    the POST /score requests that follow.

    One press is one POST. The door answers a small pass in milliseconds,
    so the second click of a double-click lands after the 202 has made
    Score live again; counted as a press it would start a second pass
    and pay the judge twice, or replace this pass's line with the door's
    refusal of it. Pre-state: Score live and nothing sent."""
    eid, _ = finished(page, bench_url, JUDGED)
    page.set_viewport_size({"width": 1280, "height": 1600})
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    page.get_by_test_id("experiment-judge").select_option("stub/fast")
    score = page.get_by_test_id("experiment-score")
    expect(score).to_be_enabled()
    posts = []
    page.on(
        "request",
        lambda r: (
            posts.append(r.url)
            if r.method == "POST" and r.url.endswith(f"/experiments/{eid}/score")
            else None
        ),
    )
    box = score.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    page.mouse.down(click_count=1)
    page.mouse.up(click_count=1)
    page.wait_for_timeout(150)
    page.mouse.down(click_count=2)
    page.mouse.up(click_count=2)

    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text(re.compile(rf"^a scoring pass was started at {UTC}; "))
    page.wait_for_timeout(500)
    assert len(posts) == 1
    expect(message).to_have_attribute("data-state", "")


def test_an_answer_returns_focus_only_to_the_row_still_chosen(
    page, bench, bench_url, scorings
):
    """WINDOW: Score pressed from the keyboard on A with its POST held; B's
    row chosen and B's Score pressed from the keyboard with its POST held
    too; A's answer released, then B's; and what has focus after each.

    An answer about a row the person has left does not pull them back to
    it: A's answer lands while B is chosen and leaves focus alone, and
    B's, answering the press made on the row still chosen, puts focus on
    B's row."""
    a, _ = finished(page, bench_url, PLAIN, name=unique("left"))
    b, _ = finished(page, bench_url, PLAIN, name=unique("chosen"))
    bench(["stub/fast"])
    open_experiments(page)
    score = page.get_by_test_id("experiment-score")
    row_for(page, a).click()
    expect(score).to_be_enabled()
    held_a = hold(page, f"**/experiments/{a}/score", "POST")
    held_b = hold(page, f"**/experiments/{b}/score", "POST")
    score.focus()
    page.keyboard.press("Enter")
    wait_held(page, held_a)
    row_for(page, b).focus()
    page.keyboard.press("Enter")
    expect(row_for(page, b)).to_have_attribute("aria-pressed", "true")
    expect(score).to_be_enabled()
    score.focus()
    page.keyboard.press("Enter")
    wait_held(page, held_b)

    settle(page, held_a.pop())

    assert page.evaluate("() => document.activeElement.dataset.id") != str(a)
    settle(page, held_b.pop())
    assert page.evaluate("() => document.activeElement.dataset.id") == str(b)


def test_start_and_stop_answers_return_focus_only_to_the_row_still_chosen(
    page, bench, bench_url, runs, scorings
):
    """WINDOW: Start pressed from the keyboard on A with its POST held; B,
    a finished experiment, chosen from the keyboard and its Score pressed
    with that POST held too, so focus has fallen to the page; A's answer
    released, then B's. Then the same with A running and its Stop.

    Start's and Stop's answers, like Score's, return focus only to the
    row still chosen since the press: A's answer lands while B is chosen
    and leaves focus where it is rather than pulling it to A's row, whose
    next Enter would select A again; B's answer then puts it on B's row.
    Pre-state: A's own word is kept for it, shown when it is selected."""
    b, _ = finished(page, bench_url, PLAIN, name=unique("chosen"))
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"f{n}"} for n in range(4)])
    a = create(page, unique("left"))
    runs.append(a)
    score = page.get_by_test_id("experiment-score")
    focused = "() => document.activeElement.dataset.id || ''"

    def leave_for_b_and_score_it():
        row_for(page, b).focus()
        page.keyboard.press("Enter")
        expect(row_for(page, b)).to_have_attribute("aria-pressed", "true")
        expect(score).to_be_enabled()
        held_b = hold(page, f"**/experiments/{b}/score", "POST")
        score.focus()
        page.keyboard.press("Enter")
        wait_held(page, held_b)
        return held_b

    for door, control in (("start", "experiment-start"), ("stop", "experiment-stop")):
        if door == "stop":
            row_for(page, a).click()
            expect(page.get_by_test_id(control)).to_be_visible()
        held_a = hold(page, f"**/experiments/{a}/{door}", "POST")
        page.get_by_test_id(control).focus()
        page.keyboard.press("Enter")
        wait_held(page, held_a)
        held_b = leave_for_b_and_score_it()

        settle(page, held_a.pop())

        assert page.evaluate(focused) != str(a), door
        settle(page, held_b.pop())
        assert page.evaluate(focused) == str(b), door
        page.unroute(f"**/experiments/{a}/{door}")
        page.unroute(f"**/experiments/{b}/score")
        wait_scoring_idle(page, bench_url)
    row_for(page, a).click()
    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile("^(stopping after the trial in flight)?$")
    )
    wait_status(page, bench_url, a)


def test_a_dataset_the_store_cannot_cite_is_the_doors_sentence(
    page, bench, bench_url, bench_db, collectors, scorings
):
    """WINDOW: a finished judge-task experiment whose stored dataset row
    is then edited by hand in the session bench's own database (to other
    bytes that still parse), selected in the page, selected again, and
    Score's request sent anyway.

    The detail door refuses the row in the key's words, with 12-character
    digests, and the page shows that sentence as it is beside a greyed
    Score, not as a failure to try again: asking again gets the same
    answer until someone repairs the row, and it does. The greyed Score
    is a courtesy: the Score door refuses those bytes in the same words,
    and nothing is scored. No answer here is mocked; the row is edited as
    a person with sqlite would. PRE-STATE: the detail door serves the row
    before the edit."""
    a, digest = finished(page, bench_url, JUDGED)
    other, _ = finished(page, bench_url, PLAIN)
    assert page.request.get(f"{bench_url}/datasets/{digest}").status == 200
    forged = (
        json.dumps(
            {"id": "j1", "prompt": "forged", "rubric": "r", "scorer": {"kind": "judge"}}
        )
        + "\n"
    ).encode()
    with sqlite3.connect(bench_db) as db:
        db.execute("UPDATE datasets SET content = ? WHERE digest = ?", (forged, digest))
    sentence = (
        f"the bytes stored under digest {digest[:12]} hash to "
        f"{hashlib.sha256(forged).hexdigest()[:12]}: the datasets table was edited "
        "outside the bench, and a record citing the first would contain the "
        "second's tasks."
    )
    bench(["stub/fast"])
    open_experiments(page)
    nudge = page.get_by_test_id("experiment-score-nudge")
    score = page.get_by_test_id("experiment-score")

    row_for(page, a).click()

    expect(nudge).to_have_text("Score waits: " + sentence)
    expect(score).to_be_disabled()
    expect(score).to_have_text("Score")
    row_for(page, other).click()
    row_for(page, a).click()
    expect(nudge).to_have_text("Score waits: " + sentence)
    for body in (
        {"dataset_digest": digest},
        {"dataset_digest": digest, "judge_model": "stub/fast"},
    ):
        refused = page.request.post(f"{bench_url}/experiments/{a}/score", data=body)
        assert (refused.status, refused.json()["detail"]) == (422, sentence)
    assert score_series(page, bench_url, a) == []


def test_a_lost_score_answer_reads_the_report_again(
    page, bench, bench_url, collectors, scorings
):
    """WINDOW: GET /experiments/{id}/report requests while Score's POST is
    held, and after that POST is lost with the row still selected.

    No door says whether a pass started, so the page says so, in full and
    stamped, reads the report again for whatever it now holds, and leaves
    Score live. Pre-state: no read of the report while the POST is out."""
    collectors.extend([ABORTED_RESOURCE, "bench: scoring an experiment failed"])
    eid, _ = finished(page, bench_url, PLAIN)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    expect(page.get_by_test_id("report-banner")).to_be_visible()
    held = hold(page, f"**/experiments/{eid}/score", "POST")
    reads = report_reads(page, eid)
    page.get_by_test_id("experiment-score").click()
    wait_held(page, held)
    page.wait_for_timeout(200)
    assert reads == []

    with page.expect_request(
        lambda r: r.method == "GET" and r.url.endswith(f"/experiments/{eid}/report")
    ):
        held.pop().abort()

    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile(
            rf"^no answer came back at {UTC} \(.+\); no door says whether a pass started$"
        )
    )
    expect(page.get_by_test_id("experiment-score")).to_be_enabled()


def test_a_selection_no_reload_asked_about_is_asked_when_it_is_listed(
    page, bench, bench_url
):
    """WINDOW: Create answered 201 while the list read it waits on is held,
    a second read of the list started so the first is abandoned (Create's
    selection of the new experiment then finds nothing to ask about), and
    the second read released.

    The reload that finally lists the selected experiment asks the store
    about its dataset, rather than leaving a digest nobody asked about to
    read, once the experiment finishes, as a question that failed.
    Pre-state: no question about that digest before the release."""
    digest = ready(page, bench, bench_url, ["stub/fast"], [{"id": "q1"}])
    asked = []
    page.on(
        "request",
        lambda r: (
            asked.append(r.url) if r.url.endswith(f"/datasets/{digest}") else None
        ),
    )
    held = hold(page, "**/experiments", "GET")
    page.get_by_test_id("experiment-name").fill(unique("superseded"))
    page.get_by_test_id("experiment-create-button").click()
    wait_held(page, held)
    page.evaluate("() => { BenchLifecycle.refresh(); }")
    wait_held(page, held, count=2)
    expect(page.get_by_test_id("experiment-create-msg")).to_have_text(
        re.compile(r"^created experiment \d+\.")
    )
    eid = int(
        re.search(
            r"\d+", page.get_by_test_id("experiment-create-msg").inner_text()
        ).group()
    )
    page.wait_for_timeout(200)
    assert asked == []

    held[1].continue_()

    expect(row_for(page, eid)).to_have_attribute("aria-pressed", "true")
    for _ in range(40):
        if asked:
            break
        page.wait_for_timeout(50)
    assert len(asked) == 1
    page.unroute("**/experiments")


def test_the_report_states_judge_spend_on_its_own_line(
    page, bench, bench_url, scorings
):
    """WINDOW: the report banner's judge-spend line and the providers
    table's cost cells, for an experiment scored with a judge (the stub
    bills every judge call) and for one scored with deterministic scorers
    only, each read against GET /experiments/{id}/report.

    The payload keeps the judge's cost on its own key (report.judge_cost),
    the bench's instrument cost, and the page does the same: one line
    beside the ranking, "judge spend: $X.XXXX over N billed calls", or
    "judge spend: none billed" when no call was billed; and never added
    into a model's cost cell, whose total is what that model was paid.
    The expected text is built here from the payload's numbers, not by
    the page's own code. PRE-STATE: the judged pass billed its calls (two:
    one judge task on two arms), and each model's cell would read
    differently with the judge's spend folded in, so a page that folded
    it would fail here."""
    judged, j_digest = finished(
        page, bench_url, JUDGED, lineup=("stub/fast", "stub/slow")
    )
    plain, p_digest = finished(page, bench_url, PLAIN)
    for eid, body in (
        (judged, {"dataset_digest": j_digest, "judge_model": "stub/html"}),
        (plain, {"dataset_digest": p_digest}),
    ):
        started = page.request.post(f"{bench_url}/experiments/{eid}/score", data=body)
        assert started.status == 202, started.text()
        wait_scoring_idle(page, bench_url)
    report = page.request.get(f"{bench_url}/experiments/{judged}/report").json()
    spend = report["judge_cost"]
    assert spend["billed_calls"] == 2, spend
    models = report["models"]
    for model in models:
        own = f"${model['cost']['total_usd']:.4f}"
        folded = f"${model['cost']['total_usd'] + spend['total_usd']:.4f}"
        assert own != folded, (model["label"], own, folded)
    bench(["stub/fast"])
    open_experiments(page)

    row_for(page, judged).click()

    line = page.get_by_test_id("report-judge-spend")
    expect(line).to_have_text(
        f"judge spend: ${spend['total_usd']:.4f} over "
        f"{spend['billed_calls']} billed calls"
    )
    costs = column_texts(page, "report-providers", "report-cost", len(models))
    for model, text in zip(models, costs, strict=True):
        assert text.startswith(f"${model['cost']['total_usd']:.4f} ("), (
            model["label"],
            text,
        )
    row_for(page, plain).click()
    expect(line).to_have_text("judge spend: none billed")


def test_the_judge_spend_line_counts_what_it_cannot_price(
    page, bench, bench_url, scorings
):
    """WINDOW: the report banner's judge-spend line for two judged
    experiments, each read against GET /experiments/{id}/report: A, two
    arms judged by stub/nousage, whose replies carry a generation id and
    no usage; B, one arm judged by stub/html (billed), then by
    stub/nousage, then scored with no judge.

    A reply with no price is a call that went out, so the line names it
    as unpriced rather than reading "none billed" as if nothing had been
    spent; and every judge row with no billing figure is counted after
    the spend, whatever the reason (a reply with no price, a pass with no
    judge to call), the unpriced call among them, so the line never
    reads as the whole cost of judging when it may not be. The expected
    text is built here from the payload's numbers, not by the page's own
    code. PRE-STATE: A's payload holds two unpriced calls and nothing
    billed, and B's one billed call, one unpriced and two rows with no
    figure, so each count the line states is one the rows hold."""
    a, a_digest = finished(page, bench_url, JUDGED, lineup=("stub/fast", "stub/slow"))
    b, b_digest = finished(page, bench_url, JUDGED)
    for eid, body in (
        (a, {"dataset_digest": a_digest, "judge_model": "stub/nousage"}),
        (b, {"dataset_digest": b_digest, "judge_model": "stub/html"}),
        (b, {"dataset_digest": b_digest, "judge_model": "stub/nousage"}),
        (b, {"dataset_digest": b_digest}),
    ):
        started = page.request.post(f"{bench_url}/experiments/{eid}/score", data=body)
        assert started.status == 202, started.text()
        wait_scoring_idle(page, bench_url)
    a_spend = page.request.get(f"{bench_url}/experiments/{a}/report").json()[
        "judge_cost"
    ]
    b_spend = page.request.get(f"{bench_url}/experiments/{b}/report").json()[
        "judge_cost"
    ]
    assert a_spend == {
        "total_usd": 0,
        "billed_calls": 0,
        "unpriced_calls": 2,
        "rows_without_figure": 2,
    }, a_spend
    assert (
        b_spend["billed_calls"],
        b_spend["unpriced_calls"],
        b_spend["rows_without_figure"],
    ) == (1, 1, 2), b_spend
    bench(["stub/fast"])
    open_experiments(page)
    line = page.get_by_test_id("report-judge-spend")

    row_for(page, a).click()

    expect(line).to_have_text(
        f"judge spend: none billed, {a_spend['unpriced_calls']} calls unpriced; "
        f"{a_spend['rows_without_figure']} judge rows carry no billing figure"
    )
    row_for(page, b).click()
    expect(line).to_have_text(
        f"judge spend: ${b_spend['total_usd']:.4f} over "
        f"{b_spend['billed_calls']} billed call, "
        f"{b_spend['unpriced_calls']} unpriced; "
        f"{b_spend['rows_without_figure']} judge rows carry no billing figure"
    )


# ---- Names, notes and contrast.


def test_the_score_row_is_named_and_described(page, bench, bench_url, scorings):
    """WINDOW: the accessible names and descriptions of the judge select
    and Score on a judge-task experiment, and which of the row's lines
    are live regions; then the same row on an experiment with no judge
    tasks.

    The select is named by its caption alone and described by the note
    that says the list is unfiltered and unchecked; Score is described
    by why it waits and by what a press pays for, which says that every
    press sends every judge trial with response text to the judge again.
    The nudge is a polite live region, and stays rendered while empty so
    a reason that arrives later is announced; the notes are not live.
    With no judge tasks the select, its note and the paying sentence are
    gone."""
    a, _ = finished(page, bench_url, JUDGED)
    p, _ = finished(page, bench_url, PLAIN)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, a).click()
    judge = page.get_by_role("combobox", name="judge", exact=True)
    expect(judge).to_have_count(1)
    expect(judge).to_have_attribute("data-testid", "experiment-judge")
    expect(judge).to_have_attribute("aria-describedby", "experiment-judge-note")
    score = page.get_by_test_id("experiment-score")
    expect(score).to_have_attribute(
        "aria-describedby", "experiment-score-nudge experiment-score-note"
    )
    expect(page.get_by_test_id("experiment-judge-note")).to_contain_text("unfiltered")
    expect(page.get_by_test_id("experiment-score-note")).to_have_text(
        "each press asks the judge once for every trial of every judge task "
        "that has response text, including trials already judged, and every "
        "call is paid"
    )
    nudge = page.get_by_test_id("experiment-score-nudge")
    expect(nudge).to_have_attribute("role", "status")
    assert nudge.get_attribute("aria-live") in (None, "polite")
    # The live region stays rendered while empty, so a reason arriving
    # later is a change inside a region that exists, and is announced.
    page.get_by_test_id("experiment-judge").select_option("stub/fast")
    expect(nudge).to_have_text("")
    assert nudge.evaluate("el => getComputedStyle(el).display") != "none"
    page.get_by_test_id("experiment-judge").select_option("")
    expect(nudge).to_have_text(WAITS_FOR_JUDGE)
    for testid in ("experiment-judge-note", "experiment-score-note"):
        assert page.get_by_test_id(testid).get_attribute("role") is None

    row_for(page, p).click()

    expect(page.get_by_test_id("experiment-judge")).to_be_hidden()
    expect(page.get_by_test_id("experiment-judge-note")).to_be_hidden()
    expect(page.get_by_test_id("experiment-score-note")).to_be_hidden()
    expect(score).to_have_text("Score · free")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_score_row_meets_aa_where_it_sits(page, bench, bench_url, theme, scorings):
    """WINDOW: computed colours on the live page, in each theme, for every
    visible, enabled element in the Experiments panel holding text of its
    own, with a judge-task experiment selected: the judge caption, the
    nudge, both notes; then with a judge chosen, Score's own label.

    The row appears only for a finished experiment, so the N3 sweep never
    saw it; its lines are named present first so the sweep cannot pass by
    finding nothing."""
    a, _ = finished(page, bench_url, JUDGED)
    bench(["stub/fast"])
    page.evaluate("t => { document.documentElement.dataset.theme = t }", theme)
    open_experiments(page)
    row_for(page, a).click()
    for testid in (
        "experiment-score-nudge",
        "experiment-judge-note",
        "experiment-score-note",
    ):
        expect(page.get_by_test_id(testid)).to_be_visible()
    expect(page.locator("#experiment-judge-caption")).to_be_visible()
    page.mouse.move(0, 0)
    first = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]
    page.get_by_test_id("experiment-judge").select_option("stub/fast")
    expect(page.get_by_test_id("experiment-score")).to_be_enabled()
    page.mouse.move(0, 0)
    second = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]

    assert not first + second, "below WCAG AA 4.5:1:\n" + "\n".join(first + second)
