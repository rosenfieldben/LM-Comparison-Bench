"""Phase F.2 browser tombstones: history-load truth, prompt-library
ownership, and Stop.

These share the session-scoped bench server and database with the other
browser files, so each test keys its assertions on its own prompt text
and never on total history size. Failures and races are forced with
Playwright route interception, an evaluate that reads the DOM in the same
synchronous tick, or a stalling stub personality, never with sleeps or
timing luck.
"""

import re

import pytest
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
    page.evaluate("() => { startRun(); }")


def open_history(page):
    page.get_by_test_id("history-toggle").click()


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
          showGroup(999999);
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


def test_review_repro_aborted_group_load_shows_failure(bench):
    """Review finding 5: an aborted network request lands in the same
    standalone failure state, not stale cards behind the banner."""
    page = bench(["stub/fast"])
    errors = collect_page_errors(page)
    check_chip(page, 0)
    start_run(page, "f2 group abort")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    page.route("**/groups/*", lambda route: route.abort())
    open_history(page)
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
          showGroup({gid});
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
          showGroup(999999);
          startRun();
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
    page.evaluate("() => { submitSave(); submitSave(); }")

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

    page.evaluate("async () => { await loadPrompts(); }")
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
          const p = submitSave();
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


def test_review_repro_stopped_run_persists_as_aborted_in_history(bench):
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
    open_history(page)
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
    page.evaluate("() => { startRun(); }")
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
