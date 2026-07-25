"""Phase G browser tombstones: billed truth on the card and in the bar,
and what a stopped run costs.

These share the session-scoped bench server and database with the other
browser files, so each test keys its assertions on its own prompt text
and never on total history size. The billing shapes are driven by stub
personalities (stub/nancost, stub/negcost, stub/nousage) and the stops by
the stalling stub, rather than by sleeps or route interception, so the
assertions run against the real ingestion path from the wire inward.
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


def test_billed_cost_replaces_the_estimate_on_the_card(bench):
    """G2: the platform reports what it charged, so the card shows the
    charge. The estimate is not discarded, it moves to the tooltip, because
    a gap between the two is itself a signal about the provider that served
    the run.
    """
    page = bench(["stub/fast"])
    check_chip(page, 0)
    start_run(page, "g2 billed replaces estimate")

    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    cost = card.get_by_test_id("metric-cost")
    # The billed figure, and no tilde: the tilde is the estimate marker and
    # this is not an estimate.
    expect(cost).to_have_text("$3.1e-5")
    expect(cost).not_to_contain_text("~")
    # The estimate, still labeled as one, still reachable.
    expect(cost).to_have_attribute("title", re.compile(r"estimate was ~\$2\.9e-5"))


def test_reasoning_tokens_and_provider_are_visible_per_run(bench):
    """G2: the two-tier budget exists because reasoning burn was invisible,
    and throughput routing picks a provider per request. Both become
    per-card facts rather than things a reader has to infer.
    """
    page = bench(["stub/fast"])
    check_chip(page, 0)
    start_run(page, "g2 reasoning and provider")

    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("metric-reasoning")).to_have_text("5")
    expect(card.get_by_test_id("card-provider")).to_have_text("served by StubHost")


def test_review_repro_nan_billed_cost_does_not_break_a_paid_result(bench):
    """The F.1 money rule extended to the new field. A NaN cost arriving in
    the usage object must degrade to None: the run stays done (the money was
    already spent, and the post-spend invariant forbids turning it into a
    failure), the card falls back to the catalog estimate, and the session
    total stays a real number rather than becoming NaN, which would make
    every ceiling comparison false and silently disable the ceiling.
    """
    page = bench(["stub/nancost"])
    check_chip(page, 0)
    start_run(page, "g2 nan billed cost")

    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_have_count(0)
    # Degraded to the estimate, tilde and all.
    expect(card.get_by_test_id("metric-cost")).to_have_text("~$2.9e-5")
    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_have_text("~$2.9e-5")
    assert page.evaluate("() => Number.isFinite(BenchState.sessionStats.spend)")


def test_review_repro_negative_billed_cost_does_not_subtract_from_spend(bench):
    """The sign half of the money rule. A negative charge is not a small
    charge: summed into the accumulator it walks the session total backward
    and buys headroom under the ceiling that was never earned. It must
    degrade to None like a NaN does, leaving the estimate to stand.
    """
    page = bench(["stub/negcost"])
    check_chip(page, 0)
    start_run(page, "g2 negative billed cost")

    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_have_count(0)
    expect(card.get_by_test_id("metric-cost")).to_have_text("~$2.9e-5")
    assert page.evaluate("() => BenchState.sessionStats.spend > 0")


def test_missing_usage_object_leaves_both_figures_empty(bench):
    """No usage at all is the third billing shape: no billed cost, and no
    token counts to estimate from either. The card must say so with the
    empty-metric dash rather than inventing a number.
    """
    page = bench(["stub/nousage"])
    check_chip(page, 0)
    start_run(page, "g2 no usage object")

    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("metric-cost")).to_have_text("—")
    expect(card.get_by_test_id("metric-tok")).to_have_text("—")
    expect(card.get_by_test_id("metric-reasoning")).to_have_text("—")


# ---- G4: stopped runs are provisionally unpriced.


def test_review_repro_stopped_run_that_started_joins_unpriced(bench):
    """The session bar counted a stopped run in neither spend nor unpriced,
    which reads as "cost nothing". The truth is "billing unknown": stream
    cancellation stops the charge only on providers that support it, and
    throughput routing picks the provider per request, so the bench cannot
    tell from here whether a stopped run was billed.
    """
    page = bench(["stub/stall0"])
    check_chip(page, 0)
    start_run(page, "g4 stopped mid stream is unpriced")

    card = cards(page).first
    expect(card.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )
    page.get_by_test_id("stop-button").click()
    expect(status_of(card)).to_have_text("stopped", timeout=DONE_TIMEOUT)

    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_contain_text("+ 1 unpriced")
    # No amount was added: the run contributes uncertainty, not a figure.
    expect(spend).to_contain_text("~$0.0000")


def test_review_repro_stopped_while_queued_does_not_join_unpriced(bench):
    """The other half, and the reason this is not simply "count every
    stop": a run stopped while still waiting for an upstream slot never
    reached a provider and verifiably spent nothing. Counting it would
    overstate the uncertainty the bar reports, and the server agrees, since
    it persists nothing at all for a queued cancel.

    Five stall ids fill the five upstream slots so a sixth queues.
    """
    page = bench([f"stub/stall{i}" for i in range(6)])
    check_all_chips(page)
    start_run(page, "g4 stopped while queued is free")

    expect(page.get_by_test_id("card-status").filter(has_text="queued")).to_have_count(
        1, timeout=DONE_TIMEOUT
    )
    # Auto-retrying precondition, not a sleep: settle the five slot-holders
    # into their streaming state before Stop acts on them.
    started = page.get_by_test_id("card-body").filter(has_text="partial text")
    expect(started).to_have_count(5, timeout=DONE_TIMEOUT)

    page.get_by_test_id("stop-button").click()

    for i in range(6):
        expect(status_of(cards(page).nth(i))).to_have_text(
            "stopped", timeout=DONE_TIMEOUT
        )
    # Five, not six: the queued one is excluded.
    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_contain_text("+ 5 unpriced")
    expect(spend).not_to_contain_text("6 unpriced")


def test_the_unpriced_tooltip_says_what_unpriced_includes(bench):
    """A count with no explanation invites the reading it replaced. The
    tooltip has to name both things it now covers."""
    page = bench(["stub/stall0"])
    check_chip(page, 0)
    start_run(page, "g4 unpriced tooltip")

    card = cards(page).first
    expect(card.get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )
    page.get_by_test_id("stop-button").click()
    expect(status_of(card)).to_have_text("stopped", timeout=DONE_TIMEOUT)

    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_have_attribute("title", re.compile("neither a billed nor an"))
    expect(spend).to_have_attribute("title", re.compile("stopped after they started"))
