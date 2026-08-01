"""Phase F.2 browser tombstones: history-load truth, prompt-library
ownership, and Stop.

These share the session-scoped bench server and database with the other
browser files, so each test keys its assertions on its own prompt text
and never on total history size. Failures and races are forced with
Playwright route interception, an evaluate that reads the DOM in the same
synchronous tick, or a stalling stub personality, never with sleeps or
timing luck.
"""

import json
import re

import pytest
from _pytest.outcomes import Failed
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000


def cards(page):
    return page.get_by_test_id("result-card")


def status_of(card):
    return card.get_by_test_id("card-status")


def check_chip(page, index):
    page.get_by_test_id("lineup-chip").nth(index).click()


def check_all_chips(page):
    chips = page.get_by_test_id("lineup-chip")
    for i in range(chips.count()):
        chips.nth(i).click()


def start_run(page, prompt):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()


def start_superseding_run(page, prompt):
    # The Run button is disabled while work is in flight; the epoch
    # machinery, not the button, is what these tests exercise, so drive
    # the named starter directly (it returns undefined, so evaluate does
    # not block until the run finishes).
    page.get_by_test_id("prompt-input").fill(prompt)
    page.evaluate("() => { BenchStream.startRun(); }")


def save_prompt(page, text, name):
    page.get_by_test_id("prompt-input").fill(text)
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill(name)
    page.get_by_test_id("confirm-save").click()
    expect(page.get_by_test_id("linked-name")).to_have_text(
        "linked: " + name, timeout=DONE_TIMEOUT
    )


def collect_page_errors(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    return errors


def latest_history_id(page):
    # The newest history entry's id, read straight from the API so a test
    # can call showGroup with a real id without scraping the row.
    return page.evaluate(
        """async () => {
          const r = await fetch('/runs?limit=1');
          return (await r.json()).runs[0].id;
        }"""
    )


# ---- F2.1: a failed or slow history load shows the truth.


def test_review_repro_group_load_clears_view_synchronously_then_fails(bench):
    """Review finding 5: showGroup cleared the old cards, race and diff
    only after the fetch succeeded, so a failed load left the failure
    banner over a different run's cards with a race frozen as if working.
    The old view must be gone (loading shown) before any network activity,
    and a failure must stand alone."""
    page = bench(["stub/fast"])
    errors = collect_page_errors(page)
    check_chip(page, 0)
    start_run(page, "f2 sync clear")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(cards(page)).to_have_count(1)

    # showGroup clears the view and shows loading synchronously, before its
    # fetch. Read the DOM in the same evaluate tick to observe that window.
    sync_state = page.evaluate(
        """() => {
          BenchHistory.showGroup(999999);
          return {
            cards: document.querySelectorAll('[data-testid=result-card]').length,
            loading: document.querySelectorAll('[data-testid=history-loading]').length,
          };
        }"""
    )
    assert sync_state == {"cards": 0, "loading": 1}

    # The bogus id 404s at the real backend; loading becomes a standalone
    # failure with no card from the earlier run visible.
    expect(page.get_by_test_id("history-failure")).to_be_visible()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "failed to load comparison"
    )
    expect(cards(page)).to_have_count(0)
    expect(page.locator("#race")).to_be_hidden()
    assert errors == []


def test_review_repro_aborted_group_load_shows_failure(bench, open_history):
    """Review finding 5: an aborted network request lands in the same
    standalone failure state, not stale cards behind the banner."""
    page = bench(["stub/fast"])
    errors = collect_page_errors(page)
    check_chip(page, 0)
    start_run(page, "f2 group abort")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    page.route("**/groups/*", lambda route: route.abort())
    open_history()
    page.get_by_test_id("history-row").filter(has_text="f2 group abort").click()

    expect(page.get_by_test_id("history-failure")).to_be_visible()
    expect(cards(page)).to_have_count(0)
    expect(page.locator("#race")).to_be_hidden()
    assert errors == []


def test_review_repro_group_load_shows_loading_then_group(bench):
    """Review finding 5: a load shows the loading state first (old view
    already cleared) and only then the comparison, never the old cards
    while the fetch is in flight."""
    page = bench(["stub/fast", "stub/slow"])
    errors = collect_page_errors(page)
    check_chip(page, 0)
    check_chip(page, 1)
    start_run(page, "f2 group order")
    for i in range(2):
        expect(status_of(cards(page).nth(i))).to_have_text("done", timeout=DONE_TIMEOUT)

    gid = latest_history_id(page)
    # Loading and a cleared grid are in place synchronously, before the
    # real fetch resolves.
    loading_first = page.evaluate(
        f"""() => {{
          BenchHistory.showGroup({gid});
          return document.querySelectorAll('[data-testid=history-loading]').length === 1
              && document.querySelectorAll('[data-testid=result-card]').length === 0;
        }}"""
    )
    assert loading_first is True

    # Then the comparison renders and the loading state is gone.
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical comparison", timeout=DONE_TIMEOUT
    )
    expect(cards(page)).to_have_count(2)
    expect(page.get_by_test_id("history-loading")).to_have_count(0)
    assert errors == []


def test_review_repro_superseded_load_stays_silent(bench):
    """Review finding 5: a load superseded before it resolves stays
    silent. The superseding run renders cleanly and no failure ever shows
    for the abandoned load, even though the loading state was rendered."""
    page = bench(["stub/fast"])
    errors = collect_page_errors(page)
    check_chip(page, 0)
    start_run(page, "f2 supersede seed")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    # Start a doomed group load, then supersede it with a run in the same
    # tick, so the load's 404 arrives after the epoch has moved on.
    page.evaluate(
        """() => {
          BenchHistory.showGroup(999999);
          BenchStream.startRun();
        }"""
    )
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(cards(page)).to_have_count(1)
    expect(cards(page).first.get_by_test_id("card-body")).to_have_text(
        "reply from stub/fast"
    )

    page.wait_for_timeout(300)
    expect(page.get_by_test_id("history-failure")).to_have_count(0)
    expect(cards(page)).to_have_count(1)
    assert errors == []


# ---- Flake fix: the history panel's error terminal must not read as
# ---- success. These three test the harness itself, which is unusual and
# ---- is the point: open_history is the shared precondition of every
# ---- history assertion in this suite, and a precondition that reports
# ---- "settled" for a load that never happened silently converts other
# ---- tests' honest assertions into false ones.


def abort_first_runs_fetch(page, times):
    """Fail the next `times` GET /runs list fetches, then stop interfering.

    Counted rather than toggled by a flag so the recovery is provably the
    RETRY and not a race: attempt one cannot succeed, and attempt two
    cannot fail. /runs/<id> detail fetches are left alone, since only the
    list load is the panel's own.
    """
    seen = {"n": 0}

    def handle(route):
        if seen["n"] < times:
            seen["n"] += 1
            route.abort()
        else:
            route.continue_()

    page.route(re.compile(r"/runs\?"), handle)
    return seen


def test_review_repro_a_failed_history_open_is_retried_not_believed(
    bench, open_history
):
    """Third distinct layer of this test's history, verified from PR #48's
    CI log. open_history waited for any settled state and treated "error"
    as one of them. A transiently failed /runs fetch at open cleared the
    rows and set state=error, the helper returned as though the panel had
    loaded, and the assertion downstream ran against an honestly empty
    list: the row it wanted really was absent, because nothing had been
    fetched. Three rounds of forensics went into a panel that was
    reporting its own failure correctly the entire time.

    The row is real and the first fetch is guaranteed to fail, so a pass
    here can only mean the retry ran and found it.
    """
    page = bench(["stub/fast"])
    check_chip(page, 0)
    start_run(page, "f2 history retry recovers")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    seen = abort_first_runs_fetch(page, times=1)
    open_history()

    assert seen["n"] == 1, "the first list fetch was supposed to fail"
    expect(page.get_by_test_id("history-list")).to_have_attribute("data-state", "ready")
    expect(
        page.get_by_test_id("history-row").filter(has_text="f2 history retry recovers")
    ).to_have_count(1)


def test_a_history_open_that_fails_twice_fails_loudly(bench, open_history):
    """The other half: a genuine outage is the run's result, not a flake
    to absorb. It must fail with the panel's own words attached, so the
    fourth occurrence of this diagnoses itself from the log.

    pytest.fail raises Failed, so the assertion is that calling the
    fixture raises it and that the message carries what the panel said.
    """
    page = bench(["stub/fast"])
    abort_first_runs_fetch(page, times=2)

    with pytest.raises(Failed) as excinfo:
        open_history()

    message = str(excinfo.value)
    assert "failed to load twice" in message, message
    # The panel's own text, not a paraphrase of it: this is the line that
    # turns a future CI log into a diagnosis.
    assert "failed to load history" in message, message


def test_review_repro_a_reopened_panel_cannot_answer_with_the_previous_load(bench):
    """The secondary path, and it was not hypothetical.

    loadHistory sets state=loading before its first await, which is
    necessary and was not sufficient: `toggle` on a <details> is
    dispatched asynchronously, so between the click and loadHistory the
    attribute still held the PREVIOUS load's terminal value. Measured on
    the retry path above, where reopening a failed panel reported "error"
    for a load that had not started: the retry could conclude the retry
    had failed too, which would have converted the loud-failure branch
    into a new flake wearing the old one's face.

    The panel now claims its state in a click listener, synchronously,
    before the default action opens it. Read in the same tick as the
    click, which is the only reading that can tell the two apart: a poll
    after the fact sees "loading" either way.
    """
    page = bench(["stub/fast"])
    page.get_by_test_id("history-toggle").click()
    # Whichever terminal this session's shared database produces; the
    # point is only that a terminal is sitting on the attribute when the
    # next click lands.
    settled = page.wait_for_function(
        """() => {
             const s = document.getElementById('history-list').dataset.state;
             return ['ready', 'empty'].includes(s) ? s : null;
           }""",
        timeout=DONE_TIMEOUT,
    ).json_value()
    page.get_by_test_id("history-toggle").click()

    state_at_click = page.evaluate(
        """() => {
             document.querySelector('[data-testid="history-toggle"]').click();
             return document.getElementById('history-list').dataset.state;
           }"""
    )

    # Pre-fix this is `settled`, the answer to the load before this one.
    assert state_at_click == "loading", (state_at_click, settled)


# ---- F2.2: prompt-library ownership.


def test_review_repro_double_save_posts_once(bench):
    """Review finding 6: submitSave had no in-flight guard, so a double OK
    or double Enter issued two POSTs; the second 409'd and left a false
    'already exists' error after a successful save. The guard must issue
    exactly one POST and no false error."""
    page = bench(["stub/fast"])
    posts = {"n": 0}

    def count(route):
        if route.request.method == "POST":
            posts["n"] += 1
        route.continue_()

    page.route("**/prompts", count)
    page.get_by_test_id("prompt-input").fill("f2 double text")
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill("f2 double name")
    # Two submitSave calls in one tick: the guard must collapse them to one
    # POST. Driving submitSave directly proves the flag, not just the
    # button's disabled state.
    page.evaluate("() => { BenchLibrary.submitSave(); BenchLibrary.submitSave(); }")

    expect(page.get_by_test_id("linked-name")).to_have_text(
        "linked: f2 double name", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("prompt-msg")).to_have_text("")
    assert posts["n"] == 1


def test_review_repro_stale_409_clears_after_successful_save(bench):
    """Review finding 6: a duplicate-name 409 survived a later successful
    save. Clearing the library error at the start of each attempt means a
    rename-and-retry cannot inherit the old conflict message."""
    page = bench(["stub/fast"])
    save_prompt(page, "keeper text", "f2 keeper")

    # A same-name save collides.
    page.get_by_test_id("prompt-input").fill("different text")
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill("f2 keeper")
    page.get_by_test_id("confirm-save").click()
    expect(page.get_by_test_id("prompt-msg")).to_have_text(
        "a prompt with that name already exists"
    )

    # Rename and save: the stale 409 must clear.
    page.get_by_test_id("prompt-name").fill("f2 keeper renamed")
    page.get_by_test_id("confirm-save").click()
    expect(page.get_by_test_id("linked-name")).to_have_text(
        "linked: f2 keeper renamed", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("prompt-msg")).to_have_text("")


def test_review_repro_reload_preserves_live_selection(bench):
    """Review finding 6: loadPrompts wrote the selection from a parameter
    and had no version, so a stale reload resolving after a newer choice
    reset the dropdown. The reconciling setter keeps the live selection
    through a reload (awaited here, so the assertion runs after it
    resolves, exactly the window a stale GET would land in)."""
    page = bench(["stub/fast"])
    save_prompt(page, "alpha text", "f2 sel alpha")
    save_prompt(page, "beta text", "f2 sel beta")

    page.get_by_test_id("saved-prompts").select_option(label="f2 sel alpha")
    expect(page.get_by_test_id("linked-name")).to_have_text("linked: f2 sel alpha")

    page.evaluate("async () => { await BenchLibrary.loadPrompts(); }")
    expect(page.get_by_test_id("linked-name")).to_have_text("linked: f2 sel alpha")


def test_review_repro_edit_during_save_leaves_no_false_link(bench):
    """Review finding 6: a save that landed after the textarea changed
    re-linked a prompt whose text no longer matched the screen. The link
    is re-established only if the textarea still shows the saved text."""
    page = bench(["stub/fast"])
    page.get_by_test_id("prompt-input").fill("original text")
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill("f2 edit race")
    # Start the save (it captures the current text), change the textarea
    # before the POST returns, then let it resolve. The captured text no
    # longer matches, so no link is claimed.
    page.evaluate(
        """async () => {
          const p = BenchLibrary.submitSave();
          document.getElementById('prompt').value = 'changed mid-save';
          await p;
        }"""
    )

    expect(page.get_by_test_id("linked-name")).to_have_text("")
    # The prompt was still saved, just not linked.
    options = page.get_by_test_id("saved-prompts").locator("option")
    expect(options.filter(has_text="f2 edit race")).to_have_count(1)


# ---- F2.4: Stop.


def test_review_repro_stop_mid_stream_keeps_partial_shows_stopped(bench):
    """F2.4: Stop mid stream keeps the partial text under a stopped status
    (no error styling, no fabricated metrics) and the race entry stops
    working and reads stopped. Server-side persistence of the disconnected
    run is covered by test_client_disconnect_persists_partial_run."""
    page = bench(["stub/stall0"])
    check_chip(page, 0)
    start_run(page, "f2 stop mid")
    card = cards(page).first
    expect(card.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("stop-button")).to_be_enabled()

    page.get_by_test_id("stop-button").click()

    expect(status_of(card)).to_have_text("stopped", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-body")).to_contain_text("partial text")
    expect(card.get_by_test_id("card-error")).to_have_count(0)
    row = page.locator(".race-row")
    expect(row).to_have_class(re.compile(r"\bstopped\b"))
    expect(row).not_to_have_class(re.compile(r"\bworking\b"))
    expect(page.locator(".race-val")).to_have_text("stopped")
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    expect(page.get_by_test_id("stop-button")).to_be_disabled()


def test_review_repro_stop_queued_run_shows_stopped_no_text(bench):
    """F2.4: a run still queued for an upstream slot when Stop is pressed
    shows the stopped state with no text (it never streamed). Five stall
    ids fill the five slots so a sixth queues."""
    models = [f"stub/stall{i}" for i in range(6)]
    page = bench(models)
    check_all_chips(page)
    start_run(page, "f2 stop queued")

    # Exactly one of the six queues behind the five upstream slots.
    expect(page.get_by_test_id("card-status").filter(has_text="queued")).to_have_count(
        1, timeout=DONE_TIMEOUT
    )
    # Wait for the five slot-holders to finish streaming before stopping.
    # The queued frame is emitted at admission, while the stall stub is
    # still 20ms from its second delta, so a Stop clicked on the queued
    # assertion alone could land between "partial " and "text" and leave a
    # card short of the full string the count below expects. This is an
    # auto-retrying precondition, not a sleep: it settles the state Stop
    # acts on rather than hoping the timing lands right.
    expect(
        page.get_by_test_id("card-body").filter(has_text="partial text")
    ).to_have_count(5, timeout=DONE_TIMEOUT)

    page.get_by_test_id("stop-button").click()

    for i in range(6):
        expect(status_of(cards(page).nth(i))).to_have_text(
            "stopped", timeout=DONE_TIMEOUT
        )
    # The five that started kept partial text; the queued one shows none.
    expect(
        page.get_by_test_id("card-body").filter(has_text="partial text")
    ).to_have_count(5)
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    expect(page.get_by_test_id("stop-button")).to_be_disabled()


def test_review_repro_run_after_stop_streams_normally(bench):
    """F2.4: after Stop the epoch machinery is intact, so a fresh Run
    streams normally."""
    page = bench(["stub/stall0", "stub/fast"])
    check_chip(page, 0)
    start_run(page, "f2 stop then run")
    card = cards(page).first
    expect(card.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )
    page.get_by_test_id("stop-button").click()
    expect(status_of(card)).to_have_text("stopped", timeout=DONE_TIMEOUT)
    expect(page.get_by_test_id("run-button")).to_be_enabled()

    # A fresh run of the fast model works end to end.
    check_chip(page, 0)
    check_chip(page, 1)
    start_run(page, "after stop run")
    fresh = cards(page).first
    expect(fresh.get_by_test_id("card-model")).to_have_text("stub/fast")
    expect(status_of(fresh)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(fresh.get_by_test_id("card-body")).to_have_text("reply from stub/fast")


def test_review_repro_stopped_run_persists_as_aborted_in_history(bench, open_history):
    """F2.4: stopping a started run preserves what streamed, in history as
    an aborted record on the next load (the client never gets a run id).
    The server disconnect persistence is unit-tested
    (test_client_disconnect_persists_partial_run); this drives it end to
    end through Stop."""
    page = bench(["stub/stall0"])
    check_chip(page, 0)
    start_run(page, "f2 stop persists")
    card = cards(page).first
    expect(card.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )
    page.get_by_test_id("stop-button").click()
    expect(status_of(card)).to_have_text("stopped", timeout=DONE_TIMEOUT)

    # Wait for the server to persist the disconnected run, then load it.
    page.wait_for_function(
        """(t) => fetch('/runs?limit=100').then(r => r.json())
              .then(d => d.runs.some(x => x.prompt_text.includes(t)))""",
        arg="f2 stop persists",
    )
    open_history()
    page.get_by_test_id("history-row").filter(has_text="f2 stop persists").click()
    replayed = cards(page).first
    expect(replayed.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )
    expect(replayed.get_by_test_id("card-error")).to_contain_text("aborted")


def test_review_repro_stop_during_group_creation_halts_run(bench):
    """Closing review (finding 1): Stop is enabled the instant Run is
    clicked (the batch reservation), but during the /groups POST no
    per-model controller exists yet. A Stop in that window must still halt
    the run, not let it stream to completion once /groups resolves."""
    page = bench(["stub/stall0"])
    held = []

    def hold_groups(route):
        if route.request.method == "POST":
            held.append(route)
        else:
            route.continue_()

    page.route("**/groups", hold_groups)
    check_chip(page, 0)
    # Fill the prompt so Run re-enables on a clean drain (not the
    # empty-prompt guard) once the batch settles.
    page.get_by_test_id("prompt-input").fill("f2 stop during groups")
    # startRun creates the cards, then awaits the held /groups POST.
    page.evaluate("() => { BenchStream.startRun(); }")
    expect(page.get_by_test_id("stop-button")).to_be_enabled()
    page.get_by_test_id("stop-button").click()

    # Release the held /groups so startRun leaves the await (pre-fix, the
    # run would then stream; post-fix it is already aborted).
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, "/groups POST was not intercepted"
    try:
        held[0].continue_()
    except Exception:
        pass  # the Stop aborted this request; fine

    card = cards(page).first
    # Halted: stopped state, and it never reached the stub, so no text.
    expect(status_of(card)).to_have_text("stopped", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-body")).to_have_text("")
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    expect(page.get_by_test_id("stop-button")).to_be_disabled()


def test_review_repro_rerun_after_stop_runs_not_stopped(bench):
    """Adversarial review of F2.4: the stop mark (stoppedEpoch) is set on
    Stop and, pre-fix, lived for the whole view because Stop does not
    advance the epoch. A rerun reuses that same epoch (it stays in the
    view), so runOne's startup check aborted it on sight, rendering a
    second 'stopped' card and dropping the retry. The mark must be scoped
    to the stopped batch: once it settles, a rerun in the same view runs.

    Mixed batch: stub/fast is forced to error (earning a Rerun) via a
    per-model done frame fulfilled at the app boundary, so the error is
    deterministic and independent of the session-shared flaky one-shot;
    stub/stall0 hits the real backend and stalls. Stop halts the stall,
    then Rerun on the errored card must actually reach the network again
    (it errors once more, since the route always errors that model), not
    abort straight to 'stopped'. The error->error transition on rerun is
    the tombstone: pre-fix the rerun never fetched and read 'stopped'.
    """
    page = bench(["stub/fast", "stub/stall0"])

    # Force stub/fast to error at the app boundary on every attempt; leave
    # every other /compare/stream (the stall) to the real backend. The
    # done frame carries an error result, the exact shape the client turns
    # into an error card with a Rerun.
    err_frame = (
        'data: {"type":"done","result":'
        '{"error":"forced review error","response_text":null},"run_id":1}\n\n'
    )

    def force_fast_error(route):
        data = route.request.post_data or ""
        if '"model":"stub/fast"' in data.replace(" ", ""):
            route.fulfill(
                status=200,
                content_type="text/event-stream",
                body=err_frame,
            )
        else:
            route.continue_()

    page.route("**/compare/stream", force_fast_error)

    check_all_chips(page)
    start_run(page, "f2 rerun after stop")

    fast = (
        cards(page)
        .filter(has=page.get_by_test_id("card-model").filter(has_text="stub/fast"))
        .first
    )
    stall = (
        cards(page)
        .filter(has=page.get_by_test_id("card-model").filter(has_text="stub/stall0"))
        .first
    )

    # The forced model errors and earns a Rerun; the stall streams then hangs.
    expect(status_of(fast)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(fast.get_by_test_id("card-error")).to_contain_text("forced review error")
    expect(fast.get_by_test_id("tool-rerun")).to_be_visible()
    expect(stall.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )

    # Stop halts the stall; the batch settles and Run re-enables.
    page.get_by_test_id("stop-button").click()
    expect(status_of(stall)).to_have_text("stopped", timeout=DONE_TIMEOUT)
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    expect(page.get_by_test_id("stop-button")).to_be_disabled()

    # Rerun the errored card. Pre-fix this aborted on the lingering stop
    # mark and rendered 'stopped'; post-fix it reaches the network again
    # and errors, proving the rerun actually ran rather than self-aborting.
    fast.get_by_test_id("tool-rerun").click()
    expect(status_of(fast)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(fast.get_by_test_id("card-error")).to_contain_text("forced review error")
    # The stall's stopped card is untouched by the rerun.
    expect(status_of(stall)).to_have_text("stopped")


# ---- F4.3: the stop mark must not outlive the stop it belongs to.


def test_review_repro_stop_of_standalone_rerun_leaves_no_mark(bench):
    """Second review, confirmed defect: stopRuns() set stoppedEpoch
    unconditionally, and the only clearing site was startRun's finally. Stop
    a standalone rerun, with no batch startup pending, and nothing ever
    cleared the mark: it lived for the life of the view, so every later
    rerun began already aborted and rendered stopped.

    The reviewer's repro verbatim: two errored cards, rerun one, Stop it,
    then rerun the other. The second rerun must reach the network and
    stream. Both models are forced to error on their first attempt and to
    behave on later ones, at the app boundary, so ordering is deterministic
    and independent of the session-shared stub state.
    """
    page = bench(["stub/fast", "stub/slow"])
    attempts = {}

    err_frame = (
        'data: {"type":"done","result":'
        '{"error":"forced first attempt","response_text":null},"run_id":1}\n\n'
    )

    def route_stream(route):
        data = (route.request.post_data or "").replace(" ", "")
        model = "stub/fast" if '"model":"stub/fast"' in data else "stub/slow"
        attempts[model] = attempts.get(model, 0) + 1
        if attempts[model] == 1:
            # First attempt for each model errors, earning a Rerun control.
            route.fulfill(status=200, content_type="text/event-stream", body=err_frame)
        elif model == "stub/fast":
            # The card that gets Stopped must stay live, so this attempt
            # goes to the real backend rewritten onto the stalling stub
            # model: a fulfilled body would end the response and drain the
            # in-flight count, disabling Stop before it can be clicked.
            rewritten = json.loads(route.request.post_data or "{}")
            rewritten["model"] = "stub/stall0"
            # The rewrite makes this request a different model from the one
            # the comparison declared, so it must stop claiming membership.
            # H1.1 made the group's lineup enforceable at entry, and an
            # undeclared member is precisely what that check refuses; a
            # harness that kept the group_id here would be asserting the
            # enforcement is broken. Dropping the link costs this attempt
            # its history grouping and nothing this test measures.
            rewritten.pop("group_id", None)
            rewritten.pop("position", None)
            route.continue_(post_data=json.dumps(rewritten))
        else:
            # The card rerun after the Stop must reach the network and
            # stream to completion. Pre-fix it never fetched at all.
            route.fulfill(
                status=200,
                content_type="text/event-stream",
                body=(
                    'data: {"type":"delta","text":"second rerun streamed"}\n\n'
                    'data: {"type":"done","result":'
                    '{"response_text":"second rerun streamed","error":null},'
                    '"run_id":2}\n\n'
                ),
            )

    page.route("**/compare/stream", route_stream)
    check_all_chips(page)
    start_run(page, "f4 standalone rerun stop")

    fast = (
        cards(page)
        .filter(has=page.get_by_test_id("card-model").filter(has_text="stub/fast"))
        .first
    )
    slow = (
        cards(page)
        .filter(has=page.get_by_test_id("card-model").filter(has_text="stub/slow"))
        .first
    )
    # Both cards error, so both carry a Rerun. The batch has settled, which
    # is what makes the next Stop a standalone-rerun Stop.
    expect(status_of(fast)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(status_of(slow)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    expect(page.get_by_test_id("stop-button")).to_be_disabled()

    # Rerun the first card, then Stop it. No batch startup is pending here.
    fast.get_by_test_id("tool-rerun").click()
    expect(fast.get_by_test_id("card-body")).to_contain_text(
        "partial", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("stop-button")).to_be_enabled()
    page.get_by_test_id("stop-button").click()
    expect(status_of(fast)).to_have_text("stopped", timeout=DONE_TIMEOUT)
    expect(page.get_by_test_id("stop-button")).to_be_disabled()

    # The tombstone: rerun the OTHER card. Pre-fix the lingering mark
    # aborted it on sight and it read "stopped" with no text.
    slow.get_by_test_id("tool-rerun").click()
    expect(status_of(slow)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(slow.get_by_test_id("card-body")).to_have_text("second rerun streamed")
    # The stopped card is untouched by the other card's rerun.
    expect(status_of(fast)).to_have_text("stopped")
