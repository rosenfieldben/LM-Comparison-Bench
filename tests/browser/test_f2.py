"""Phase F.2 browser tombstones: history-load truth, prompt-library
ownership, and Stop.

These share the session-scoped bench server and database with the other
browser files, so each test keys its assertions on its own prompt text
and never on total history size. Failures and races are forced with
Playwright route interception, an evaluate that reads the DOM in the same
synchronous tick, or a stalling stub personality, never with sleeps or
timing luck.
"""

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
