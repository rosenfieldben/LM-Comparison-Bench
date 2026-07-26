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


# ---- G5: privacy routing.


def test_no_badge_on_the_default_policy(bench):
    """The badge's absence carries meaning too: standard routing is the
    default and gets no decoration, so a badge always means a deliberate,
    non-default posture."""
    page = bench(["stub/fast"])
    expect(page.get_by_test_id("policy-badge")).to_be_hidden()


def test_a_non_standard_policy_is_visible_at_a_glance(zdr_bench):
    """A confidential session must never be ambiguous. The wording says
    "asks", because the routing constraint is OpenRouter's to honor and
    this application cannot verify it."""
    page = zdr_bench(["stub/fast"])
    badge = page.get_by_test_id("policy-badge")
    expect(badge).to_be_visible()
    expect(badge).to_have_text("zero-retention routing")
    expect(badge).to_have_attribute("title", re.compile("BENCH_DATA_POLICY=zdr"))
    expect(badge).to_have_attribute("title", re.compile("guarantee is OpenRouter's"))


def test_no_compliant_endpoint_surfaces_as_an_ordinary_failure(zdr_bench):
    """Failure is loud by inheritance. When no endpoint satisfies the
    policy, OpenRouter reports it like any other routing failure, so the
    card shows it through the existing error path with no new code. The
    badge stays up while it does, so the reason is legible."""
    page = zdr_bench(["stub/nopolicy"])
    check_chip(page, 0)
    start_run(page, "g5 no compliant endpoint")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_contain_text(
        "No endpoints found matching your data policy"
    )
    expect(page.get_by_test_id("policy-badge")).to_be_visible()


def test_the_policy_reaches_the_wire_from_the_browser(zdr_bench):
    """End to end, through the real streaming path the browser uses: the
    stub records every payload it received, so this asserts the provider
    block that was actually sent rather than the one the app intended."""
    page = zdr_bench(["stub/fast"])
    check_chip(page, 0)
    start_run(page, "g5 policy on the wire")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    sent = page.evaluate(
        """async () => {
          const r = await fetch('/runs');
          const runs = (await r.json()).runs;
          const detail = await (await fetch('/runs/' + runs[0].id)).json();
          return JSON.parse(detail.results[0].request_json).provider;
        }"""
    )
    assert sent == {"sort": "throughput", "zdr": True, "data_collection": "deny"}


# ---- Phase G closing review.


def test_review_repro_replay_uses_the_declared_position_not_the_chip_order(bench):
    """Closing review: the group replay sorted cards by the CURRENT chip
    order, which is the exact drift the position column was added to
    eliminate. Editing the lineup after a run rearranged that run's replay
    to match a lineup it never ran under.
    """
    page = bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    start_run(page, "g-review declared order")
    for i in range(2):
        expect(status_of(cards(page).nth(i))).to_have_text("done", timeout=DONE_TIMEOUT)
    # The comparison ran fast-then-slow, and stub/slow finishes second, so
    # run order and declared order disagree in the stored rows too.
    assert [c.inner_text() for c in cards(page).get_by_test_id("card-model").all()] == [
        "stub/fast",
        "stub/slow",
    ]

    # Reverse the lineup, which is what the old sort keyed on.
    page.evaluate("() => { BenchControls.lineup.reverse(); }")

    page.get_by_test_id("history-toggle").click()
    page.get_by_test_id("history-row").filter(
        has_text="g-review declared order"
    ).click()
    expect(cards(page)).to_have_count(2, timeout=DONE_TIMEOUT)

    replayed = [c.inner_text() for c in cards(page).get_by_test_id("card-model").all()]
    assert replayed == ["stub/fast", "stub/slow"], replayed


def test_review_repro_a_zero_billed_session_is_not_reported_as_unpriced_idle(bench):
    """Closing review: the bar keyed its idle display on the amount, so a
    session priced at exactly zero (free models are real) claimed "nothing
    priced yet" and wore the estimate tilde over a figure the platform had
    confirmed.

    This pins the render rule only. It sets the counter directly rather
    than earning it through a run, because the stub has no free model to
    earn it with, so the increment in stream.js that feeds this counter is
    NOT covered here. Saying so is the point: the docstring used to claim
    this was driven through the stream client's accounting path, which it
    never was.
    """
    page = bench(["stub/fast"])
    spend = page.get_by_test_id("stat-spend")
    # Untouched: the fixed zero, honestly, because nothing has run.
    expect(spend).to_have_text("~$0.0000")
    expect(spend).to_have_attribute("title", re.compile("nothing priced yet"))

    shown = page.evaluate(
        """() => {
          BenchState.sessionStats.priced = 1;
          BenchState.renderStats();
          const el = document.querySelector('[data-testid=stat-spend]');
          return { text: el.textContent, title: el.title };
        }"""
    )
    assert shown["text"] == "$0.0000", shown
    assert "billed by OpenRouter" in shown["title"], shown


# ---- Phase G.1: the supersession half of the unpriced rule.


def test_review_repro_superseded_run_after_started_joins_unpriced(bench):
    """Adversarial review: G4 taught the bar that a stopped run is not free,
    but only through the Stop branch. A run superseded after the server's
    started frame and before its first delta has no text and no token
    counts, so it fell through both branches and was counted as costing
    nothing, while the server had already called upstream and persisted the
    row. The upstream reality is identical to a Stop; only the client-side
    path that aborted it differs.

    Driven through the real epoch machinery: newViewEpoch is what a History
    click and a second Run both call, and stub/slow holds two seconds of
    true wire silence after admission, which is exactly the window.
    """
    page = bench(["stub/slow"])
    check_chip(page, 0)
    start_run(page, "g1 superseded after started")

    card = cards(page).first
    # The started frame has been consumed (the label leaves "queued" for
    # "thinking") and no delta has landed yet. An auto-retrying assertion,
    # not a sleep: it settles the state supersession acts on.
    expect(status_of(card)).to_contain_text("thinking", timeout=DONE_TIMEOUT)
    assert (
        card.get_by_test_id("card-body").inner_text().strip() == "awaiting first token"
    ), "precondition: no delta has arrived yet"

    page.evaluate("() => { BenchState.newViewEpoch(); }")

    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_contain_text("+ 1 unpriced", timeout=DONE_TIMEOUT)
    # Uncertainty, not an amount: nothing is added to the total.
    expect(spend).to_contain_text("~$0.0000")
    # And the server really did keep the run, which is what makes counting
    # it the honest reading rather than a conservative one. Keyed on this
    # test's own prompt, never on total history size: the suite shares one
    # database.
    persisted = page.evaluate(
        """async () => {
          const r = await fetch('/runs');
          const runs = (await r.json()).runs;
          return runs.filter(
            (x) => x.prompt_text === 'g1 superseded after started',
          ).length;
        }"""
    )
    assert persisted == 1, persisted


def test_a_run_that_never_started_is_still_free(bench):
    """The other half of the same rule, so the fix cannot be read as "count
    everything". A spend refusal is declined before any provider sees it, so
    it moves neither counter.
    """
    page = bench(["stub/fast"])
    refusal = (
        'data: {"type":"done","result":{"model":"stub/fast",'
        '"response_text":null,"latency_ms":null,"prompt_tokens":null,'
        '"completion_tokens":null,"spend_refused":true,'
        '"error":"run refused before reaching upstream: recorded spend '
        "$1.50 reached the $1.00 ceiling (BENCH_SPEND_LIMIT_USD); no "
        'upstream call was made","cost_usd":null,"ttft_ms":null,'
        '"max_tokens":16384,"generation_id":null,"finish_reason":null},'
        '"run_id":null}\n\n'
    )
    page.route(
        "**/compare/stream",
        lambda route: route.fulfill(
            status=200, content_type="text/event-stream", body=refusal
        ),
    )
    check_chip(page, 0)
    start_run(page, "g1 refusal is still free")

    expect(status_of(cards(page).first)).to_have_text("refused", timeout=DONE_TIMEOUT)
    spend = page.get_by_test_id("stat-spend")
    expect(spend).to_have_text("~$0.0000")
    expect(spend).not_to_contain_text("unpriced")
