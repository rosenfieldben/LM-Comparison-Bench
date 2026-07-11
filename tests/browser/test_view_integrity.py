"""Phase E: frontend truth and view integrity.

The view-epoch tests reproduce the review's races and assert they are
gone: a superseded run's late events must never repaint the view that
replaced it. Superseding runs are started via the named startRun()
where the Run button's disabled state would block the UI path; the
button is the affordance, the epoch is the mechanism under test.

The not-saved warning is driven through Playwright route interception:
the real runOne, SSE parser and renderer run against the exact wire
frame the backend emits on persistence failure (that backend behavior
is unit-tested in test_stream_persistence_failure_degrades_to_null_run_id),
so the substituted transport is the documented contract seam.
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


def start_run_via_ui(page, prompt):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()


def start_superseding_run(page, prompt):
    # The Run button is rightly disabled while the first run is in
    # flight; the epoch machinery must hold no matter how a second run
    # starts, so drive the named starter directly.
    page.get_by_test_id("prompt-input").fill(prompt)
    page.evaluate("startRun()")


def collect_page_errors(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    return errors


def test_superseded_run_never_touches_new_view(bench):
    page = bench(["stub/slow", "stub/fast"])
    errors = collect_page_errors(page)

    check_chip(page, 0)
    start_run_via_ui(page, "first slow")
    expect(cards(page)).to_have_count(1)
    expect(status_of(cards(page).first)).to_contain_text("thinking")

    # Supersede during the slow model's silent stretch.
    check_chip(page, 0)
    check_chip(page, 1)
    start_superseding_run(page, "second fast")

    expect(cards(page)).to_have_count(1)
    fast = cards(page).first
    expect(fast.get_by_test_id("card-model")).to_have_text("stub/fast")
    expect(status_of(fast)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(fast.get_by_test_id("card-body")).to_have_text("reply from stub/fast")

    # Outlive the first run's would-be delivery window, then assert
    # nothing of it ever surfaced.
    page.wait_for_timeout(2600)
    expect(cards(page)).to_have_count(1)
    expect(fast.get_by_test_id("card-body")).to_have_text("reply from stub/fast")
    race_names = page.locator(".race-name")
    expect(race_names).to_have_count(1)
    expect(race_names.first).to_have_text("fast")
    assert errors == []

    # The abort disconnected the stream, so the server persisted the
    # first run through its existing disconnect path.
    page.get_by_test_id("history-toggle").click()
    row = page.get_by_test_id("history-row").filter(has_text="first slow")
    expect(row).to_have_count(1)


def test_in_flight_rerun_disables_run_and_never_touches_new_view(bench):
    page = bench(["stub/flaky-slow", "stub/fast"])
    errors = collect_page_errors(page)

    check_chip(page, 0)
    start_run_via_ui(page, "flaky first")
    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)

    # The rerun joins the same in-flight registry as a normal run.
    card.get_by_test_id("tool-rerun").click()
    expect(page.get_by_test_id("run-button")).to_be_disabled()

    check_chip(page, 0)
    check_chip(page, 1)
    start_superseding_run(page, "fresh after rerun")

    expect(cards(page)).to_have_count(1)
    fresh = cards(page).first
    expect(fresh.get_by_test_id("card-model")).to_have_text("stub/fast")
    expect(status_of(fresh)).to_have_text("done", timeout=DONE_TIMEOUT)

    # The rerun would complete around the 2s mark; wait past it.
    page.wait_for_timeout(2600)
    expect(cards(page)).to_have_count(1)
    expect(fresh.get_by_test_id("card-body")).to_have_text("reply from stub/fast")
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    assert errors == []


def test_history_replay_mid_run_is_not_repainted(bench):
    page = bench(["stub/slow", "stub/fast"])
    errors = collect_page_errors(page)

    # Seed a history entry to replay.
    check_chip(page, 1)
    start_run_via_ui(page, "seed entry")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    # Start a slow live run, then replay history over it mid-flight.
    check_chip(page, 1)
    check_chip(page, 0)
    start_run_via_ui(page, "live slow")
    expect(status_of(cards(page).first)).to_contain_text("thinking")

    page.get_by_test_id("history-toggle").click()
    page.get_by_test_id("history-row").filter(has_text="seed entry").click()

    expect(page.get_by_test_id("run-label")).to_contain_text("Historical")
    expect(cards(page)).to_have_count(1)
    replayed = cards(page).first
    expect(replayed.get_by_test_id("card-model")).to_have_text("stub/fast")
    expect(status_of(replayed)).to_have_text("done")

    # Outlive the superseded run's delivery window: the historical view
    # must not be repainted by its late events.
    page.wait_for_timeout(2600)
    expect(page.get_by_test_id("run-label")).to_contain_text("Historical")
    expect(cards(page)).to_have_count(1)
    expect(replayed.get_by_test_id("card-body")).to_have_text("reply from stub/fast")
    expect(page.locator("#race")).to_be_hidden()
    assert errors == []


def test_rapid_history_selections_last_click_wins(bench):
    page = bench(["stub/fast", "stub/html"])
    errors = collect_page_errors(page)

    # Two distinguishable entries: one card versus two.
    check_chip(page, 0)
    start_run_via_ui(page, "alpha entry")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)
    check_chip(page, 1)
    start_run_via_ui(page, "beta entry")
    expect(cards(page)).to_have_count(2)
    for i in range(2):
        expect(status_of(cards(page).nth(i))).to_have_text(
            "done", timeout=DONE_TIMEOUT
        )

    page.get_by_test_id("history-toggle").click()
    page.get_by_test_id("history-row").filter(has_text="alpha entry").click()
    page.get_by_test_id("history-row").filter(has_text="beta entry").click()

    # The second selection renders; the first never overwrites it.
    expect(cards(page)).to_have_count(2)
    page.wait_for_timeout(600)
    expect(cards(page)).to_have_count(2)
    expect(page.get_by_test_id("run-label")).to_contain_text("Historical")
    assert errors == []


def test_unpriced_results_count_in_session_spend(bench):
    # stub/unlisted is absent from the stub catalog, so its usage has
    # no price and cost_usd comes back null.
    page = bench(["stub/fast", "stub/unlisted"])

    for i in range(2):
        check_chip(page, i)
    start_run_via_ui(page, "unpriced run")
    for i in range(2):
        expect(status_of(cards(page).nth(i))).to_have_text(
            "done", timeout=DONE_TIMEOUT
        )

    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_contain_text("+ 1 unpriced")
    expect(spend).to_contain_text("~$")


def test_null_run_id_shows_not_saved_warning(bench):
    page = bench(["stub/fast"])

    # A normally persisted run carries no warning.
    check_chip(page, 0)
    start_run_via_ui(page, "saved run")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(page.get_by_test_id("save-warning")).to_have_count(0)

    # Replay the wire frame the backend emits when save_run fails
    # (unit-tested server-side): run_id null on a successful result.
    body = (
        'data: {"type": "delta", "text": "hello"}\n\n'
        'data: {"type": "done", "result": {"model": "stub/fast",'
        ' "response_text": "hello", "latency_ms": 12.0,'
        ' "prompt_tokens": 1, "completion_tokens": 1, "error": null,'
        ' "cost_usd": null, "ttft_ms": 5.0, "max_tokens": 16384},'
        ' "run_id": null}\n\n'
    )
    page.route(
        "**/compare/stream",
        lambda route: route.fulfill(
            status=200, content_type="text/event-stream", body=body
        ),
    )
    start_run_via_ui(page, "unsaved run")
    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    warning = card.get_by_test_id("save-warning")
    expect(warning).to_have_text("not saved to history")
    expect(warning).to_have_attribute("title", re.compile("persisting"))


def test_cost_language_is_honest(bench):
    page = bench(["stub/fast"])

    check_chip(page, 0)
    note = page.locator("#run-note")
    expect(note).to_contain_text("max output cost")
    expect(note).to_contain_text("(input not included)")

    start_run_via_ui(page, "cost language")
    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    cost = card.get_by_test_id("metric-cost")
    expect(cost).to_contain_text("~$")
    expect(cost).to_have_attribute("title", re.compile("not billed"))
    expect(page.get_by_test_id("stat-spend")).to_contain_text("~$")
