"""Browser proofs for Phase P's P1: the Score row's line about the latest
scoring pass, and its Stop control while a pass runs.

The line is the list's `scoring` field in words (BenchLib.scoringPassLine),
read whenever the list is, which since this phase includes after every
answer to Score and to Stop scoring. The Stop control is present only
while the server says this experiment's pass runs; the door is the rule,
and a Stop sent when none runs is refused in its words. The expected
text is built here from the server's record, not by the page's code.

EVERY PAGE HERE RUNS IN Asia/Kolkata (UTC+05:30), as in test_n4.py, so a
stamp that said UTC while printing local time would be off by five and a
half hours and fail.
"""

import re
import sqlite3

import httpx
import pytest
from playwright.sync_api import expect
from test_n import ABORTED_RESOURCE, REFUSED_RESOURCE, arm, hold, wait_held
from test_n3 import (
    CONFLICT_RESOURCE,
    LOST_RESOURCE,
    open_experiments,
    row_for,
    swept,
    wait_scoring_idle,
)
from test_n4 import HELD_JUDGE, JUDGED, PLAIN, UTC, finished

pytestmark = pytest.mark.browser

MARKUP = "<img src=x onerror=alert(1)>"


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {**browser_context_args, "timezone_id": "Asia/Kolkata"}


@pytest.fixture(autouse=True)
def collectors(page):
    errors = arm(page)
    allowed = [REFUSED_RESOURCE]
    yield allowed
    assert page.evaluate("window.__csp || []") == []
    assert [e for e in errors if not e.startswith(tuple(allowed))] == []


@pytest.fixture
def judge_gate(stub_url, scorings):
    """The held judge's gate, armed for the test and released after it,
    pass or fail, before the wait for the scoring slot."""

    def post(action):
        httpx.post(
            stub_url + "/_test/judge-gate", json={"action": action}, trust_env=False
        ).raise_for_status()

    post("arm")
    yield post
    post("release")


def stamp(iso):
    """A recorded time as the page writes it."""
    return iso[:19].replace("T", " ") + " UTC"


def passes(page, bench_url, eid):
    return page.request.get(f"{bench_url}/experiments/{eid}/scoring").json()["passes"]


def held_requests(stub_url):
    recorded = httpx.get(stub_url + "/_test/requests", trust_env=False).json()
    return [r for r in recorded["requests"] if r["model"] == HELD_JUDGE]


def wait_for_held_call(page, stub_url, count):
    for _ in range(200):
        if len(held_requests(stub_url)) >= count:
            return
        page.wait_for_timeout(50)
    raise AssertionError("the held judge call never arrived")


def refresh(page):
    page.evaluate("window.BenchLifecycle.refresh()")


def test_the_score_row_says_how_the_last_pass_ended(page, bench, bench_url, scorings):
    """WINDOW: the line beside Score for a finished experiment with no
    recorded pass, then after a deterministic pass, each read against GET
    /experiments/{id}/scoring.

    No recorded pass makes no claim: the line is absent, since scores
    from before passes were recorded may exist. After a pass the line is
    its ending in words, stamped with the UTC the record holds, whatever
    zone the page is in. Stop scoring is absent throughout, since no pass
    runs. PRE-STATE: the record holds no pass before the Score."""
    eid, digest = finished(page, bench_url, PLAIN)
    assert passes(page, bench_url, eid) == []
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    line = page.get_by_test_id("experiment-score-pass")
    stop = page.get_by_test_id("experiment-score-stop")
    expect(line).to_be_hidden()
    expect(line).to_have_text("")
    expect(stop).to_be_hidden()

    started = page.request.post(
        f"{bench_url}/experiments/{eid}/score", data={"dataset_digest": digest}
    )
    assert started.status == 202
    wait_scoring_idle(page, bench_url)
    refresh(page)

    (made,) = passes(page, bench_url, eid)
    assert (made["outcome"], made["scored"]) == ("finished", 1)
    expect(line).to_have_text(
        f"scored {stamp(made['ended_at'])}, with no judge: 1 trial scored"
    )
    expect(stop).to_be_hidden()


def test_a_failed_pass_is_said_in_the_servers_words(
    page, bench, bench_url, bench_db, scorings
):
    """WINDOW: a pass that fails on its first score write (a trigger in
    the bench's database refuses it, then is dropped), and the line
    beside Score after the list is read.

    A failed pass reaches the page: the line says the last pass failed
    and gives the record's detail, the error as the server wrote it.
    PRE-STATE: the trigger is in place when the pass runs, and gone
    after."""
    eid, digest = finished(page, bench_url, PLAIN)
    with sqlite3.connect(bench_db) as db:
        db.execute(
            """CREATE TRIGGER p_staged_failure BEFORE INSERT ON scores
               BEGIN SELECT RAISE(ABORT, 'staged failure'); END"""
        )
    try:
        started = page.request.post(
            f"{bench_url}/experiments/{eid}/score", data={"dataset_digest": digest}
        )
        assert started.status == 202
        wait_scoring_idle(page, bench_url)
    finally:
        with sqlite3.connect(bench_db) as db:
            db.execute("DROP TRIGGER p_staged_failure")
    (made,) = passes(page, bench_url, eid)
    assert (made["outcome"], made["detail"]) == (
        "failed",
        "IntegrityError: staged failure",
    )
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-score-pass")).to_have_text(
        f"the last scoring pass failed {stamp(made['ended_at'])}: "
        "IntegrityError: staged failure"
    )


def test_stop_scoring_is_there_while_a_pass_runs_and_stops_it(
    page, bench, bench_url, stub_url, judge_gate
):
    """WINDOW: an experiment over two judged trials, before any pass, then
    with a pass started through the API with the stub's held judge, its
    first call held; Stop scoring pressed in the page; the gate released;
    the list read again.

    THE CONTROL'S HALF OF THE PAIR. Absent before the pass; present, with
    the line saying the pass runs, while it does; pressed, it answers
    "asked to stop" and greys, and the line says stopping. The call in
    flight finishes (a Stop never cuts one), no second judge request is
    made, and once the list is read again the line gives the record's
    ending, "stopped on request, between trials", and the control and
    the stopping word are gone. PRE-STATE: one held judge request has
    arrived when the page is opened."""
    eid, digest = finished(page, bench_url, JUDGED, lineup=("stub/fast", "stub/slow"))
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    stop = page.get_by_test_id("experiment-score-stop")
    line = page.get_by_test_id("experiment-score-pass")
    message = page.get_by_test_id("experiment-action-msg")
    expect(stop).to_be_hidden()
    before = len(held_requests(stub_url))
    started = page.request.post(
        f"{bench_url}/experiments/{eid}/score",
        data={"dataset_digest": digest, "judge_model": HELD_JUDGE},
    )
    assert started.status == 202
    wait_for_held_call(page, stub_url, before + 1)
    refresh(page)
    (running,) = passes(page, bench_url, eid)
    assert running["running"] and running["outcome"] is None

    expect(line).to_have_text(
        f"a scoring pass judged by {HELD_JUDGE}, started "
        f"{stamp(running['started_at'])}, is running"
    )
    expect(stop).to_be_visible()
    expect(stop).to_be_enabled()
    stop.click()

    expect(message).to_have_text(
        re.compile(
            rf"^the scoring pass was asked to stop at {UTC}; it stops after the "
            r"trial being scored$"
        )
    )
    expect(line).to_have_text(
        f"a scoring pass judged by {HELD_JUDGE}, started "
        f"{stamp(running['started_at'])}, is stopping after the trial being scored"
    )
    expect(stop).to_be_disabled()
    judge_gate("release")
    wait_scoring_idle(page, bench_url)
    refresh(page)

    (made,) = passes(page, bench_url, eid)
    assert (made["outcome"], made["scored"]) == ("stopped", 1)
    expect(line).to_have_text(
        f"the last scoring pass stopped ({stamp(made['ended_at'])}, judged by "
        f"{HELD_JUDGE}): stopped on request, between trials; 1 trial scored"
    )
    expect(stop).to_be_hidden()
    expect(message).to_have_text("")
    assert len(held_requests(stub_url)) == before + 1


def test_a_stop_sent_when_no_pass_runs_is_refused(page, bench, bench_url, scorings):
    """WINDOW: POST /experiments/{id}/scoring/stop sent through the API for
    an experiment whose pass has finished, with the page's Stop scoring
    absent, and the record read before and after.

    THE DOOR'S HALF OF THE PAIR: a hidden button is not a rule. The
    request the page would not send is refused in the door's words and
    changes nothing on the record. PRE-STATE: the control is absent and
    the pass has finished."""
    eid, digest = finished(page, bench_url, PLAIN)
    assert (
        page.request.post(
            f"{bench_url}/experiments/{eid}/score", data={"dataset_digest": digest}
        ).status
        == 202
    )
    wait_scoring_idle(page, bench_url)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    expect(page.get_by_test_id("experiment-score-stop")).to_be_hidden()
    before = passes(page, bench_url, eid)

    refused = page.request.post(f"{bench_url}/experiments/{eid}/scoring/stop", data={})

    assert refused.status == 409
    assert (
        refused.json()["detail"] == f"no scoring pass for experiment {eid} is running"
    )
    assert passes(page, bench_url, eid) == before


def test_a_stale_stop_is_the_servers_sentence(
    page, bench, bench_url, stub_url, collectors, judge_gate
):
    """WINDOW: Stop scoring shown for a pass that has since ended, the list
    not yet read again, then pressed.

    The page cannot know the pass ended until it reads the list, so the
    press goes to the door, which refuses it in its own words; the page
    prints that as a refusal, stamped, and reads the list, after which
    the control is gone and the line gives the pass's ending. PRE-STATE:
    Stop scoring is present and the pass has finished on the server."""
    collectors.append(CONFLICT_RESOURCE)
    eid, digest = finished(page, bench_url, JUDGED)
    before = len(held_requests(stub_url))
    assert (
        page.request.post(
            f"{bench_url}/experiments/{eid}/score",
            data={"dataset_digest": digest, "judge_model": HELD_JUDGE},
        ).status
        == 202
    )
    wait_for_held_call(page, stub_url, before + 1)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    stop = page.get_by_test_id("experiment-score-stop")
    expect(stop).to_be_visible()
    judge_gate("release")
    wait_scoring_idle(page, bench_url)
    (made,) = passes(page, bench_url, eid)
    assert made["outcome"] == "finished"
    expect(stop).to_be_visible()

    stop.click()

    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text(
        re.compile(
            rf"^not stopped at {UTC}: no scoring pass for experiment {eid} is running$"
        )
    )
    expect(message).to_have_attribute("data-state", "refused")
    expect(stop).to_be_hidden()
    expect(page.get_by_test_id("experiment-score-pass")).to_have_text(
        re.compile(rf"^scored {re.escape(stamp(made['ended_at']))}, judged by ")
    )


def test_every_score_answer_reads_the_list_again(
    page, bench, bench_url, collectors, scorings
):
    """WINDOW: GET /experiments requests after Score's 202, and after its
    POST is lost.

    The line beside Score is the list's record of the pass, so the page
    reads the list after each answer: after the 202 the line names the
    new pass, and after a lost answer the page's own sentence sends the
    reader to that line, which the read fills. PRE-STATE: no pass is
    recorded before the first press."""
    collectors.extend(
        [LOST_RESOURCE, ABORTED_RESOURCE, "bench: scoring an experiment failed"]
    )
    eid, _ = finished(page, bench_url, PLAIN)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    assert passes(page, bench_url, eid) == []
    line = page.get_by_test_id("experiment-score-pass")
    score = page.get_by_test_id("experiment-score")

    with page.expect_request(
        lambda r: r.method == "GET" and r.url.endswith("/experiments")
    ):
        score.click()
    expect(line).to_have_text(re.compile(r"^(a scoring pass with no judge|scored )"))
    wait_scoring_idle(page, bench_url)

    held = hold(page, f"**/experiments/{eid}/score", "POST")
    score.click()
    wait_held(page, held)
    with page.expect_request(
        lambda r: r.method == "GET" and r.url.endswith("/experiments")
    ):
        held.pop().abort()
    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile(
            rf"^no answer came back at {UTC} \(.+\); the line beside Score says "
            r"whether a pass started$"
        )
    )
    expect(line).to_have_text(re.compile(r"^scored "))


def test_a_judge_named_like_markup_is_shown_as_text(page, bench, bench_url, scorings):
    """WINDOW: a pass recorded with a judge_model that is HTML, sent
    through the API (the door does not check a judge), and the Score row
    after the list is read.

    The line prints the record's words as text: the judge's name appears
    literally, no element is made from it, and no inline script runs (the
    collectors' CSP check). PRE-STATE: the record holds the name as
    sent."""
    eid, digest = finished(page, bench_url, PLAIN)
    assert (
        page.request.post(
            f"{bench_url}/experiments/{eid}/score",
            data={"dataset_digest": digest, "judge_model": MARKUP},
        ).status
        == 202
    )
    wait_scoring_idle(page, bench_url)
    (made,) = passes(page, bench_url, eid)
    assert made["judge_model"] == MARKUP
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-score-pass")).to_have_text(
        f"scored {stamp(made['ended_at'])}, judged by {MARKUP}: 1 trial scored"
    )
    assert page.get_by_test_id("experiment-score-row").locator("img").count() == 0


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_pass_line_and_stop_are_named_and_meet_aa(
    page, bench, bench_url, stub_url, theme, judge_gate
):
    """WINDOW: the Score row of an experiment whose pass is held running,
    in each theme: Stop scoring's accessible name and description, the
    line's role, and computed colours for every visible element with text
    of its own in the Experiments panel.

    Stop scoring is named for what it stops, and described by the line
    that says which pass; the line is not a live region (the answer's
    message is the one that speaks). Both meet AA where they sit.
    PRE-STATE: the pass runs, so both are present."""
    eid, digest = finished(page, bench_url, JUDGED)
    before = len(held_requests(stub_url))
    assert (
        page.request.post(
            f"{bench_url}/experiments/{eid}/score",
            data={"dataset_digest": digest, "judge_model": HELD_JUDGE},
        ).status
        == 202
    )
    wait_for_held_call(page, stub_url, before + 1)
    bench(["stub/fast"])
    page.evaluate("t => { document.documentElement.dataset.theme = t }", theme)
    open_experiments(page)
    row_for(page, eid).click()
    stop = page.get_by_role("button", name="Stop scoring", exact=True)
    expect(stop).to_have_count(1)
    expect(stop).to_have_attribute("data-testid", "experiment-score-stop")
    expect(stop).to_have_attribute("aria-describedby", "experiment-score-pass")
    line = page.get_by_test_id("experiment-score-pass")
    expect(line).to_be_visible()
    assert line.get_attribute("role") is None
    page.mouse.move(0, 0)

    low = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]

    assert not low, "below WCAG AA 4.5:1:\n" + "\n".join(low)
