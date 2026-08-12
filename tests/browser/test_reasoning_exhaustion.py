"""Browser tombstones for the reasoning-exhaustion fix.

The production incident these answer: a comparison on document prompts
reasoned until the completion cap closed on it, billed in full between
$3.38 and $3.50 per card, and showed nothing. The generation records
were unambiguous, tokens_completion 21350 against
native_tokens_reasoning 21350, but the card said only "empty response
(finish_reason: length)" and carried no sign at all that the money had
gone into thinking.

EVERY TEST HERE IS DRIVEN BY A SHAPE, NEVER BY A NAME. The stub
personality is called stub/exhausted and it is not modelled on any
vendor: it emits reasoning deltas, no content, and a usage object whose
completion count equals its reasoning count. That signature is the whole
trigger, on the server and on the card alike, which is what makes the
behaviour reach a provider that ignored the reservation entirely.
"""

import re

import httpx
import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000


def cards(page):
    return page.get_by_test_id("result-card")


def status_of(card):
    return card.get_by_test_id("card-status")


def check_all_chips(page):
    chips = page.get_by_test_id("lineup-chip")
    for i in range(chips.count()):
        chips.nth(i).click()


def run(page, prompt):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()


def test_review_repro_an_exhausted_card_says_where_the_money_went(bench):
    """WINDOW: the live streaming card, from Run to its terminal render.

    THE CARD IS WHERE THE DEFECT WAS SEEN. A person looking at five
    cards that each cost $3.38 to $3.50 read "empty response
    (finish_reason: length)" and reasonably concluded the provider had
    glitched. Nothing on the card connected the charge to the thinking,
    so the natural next move was to run it again.

    The label now names the cause and carries the count, and the count
    is the evidence: it is the same number the card's own reasoning
    metric shows, so the claim can be checked rather than trusted.
    """
    page = bench(["stub/exhausted"])
    check_all_chips(page)
    run(page, "exhaustion label")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    error = card.get_by_test_id("card-error")
    expect(error).to_contain_text("no visible answer")
    expect(error).to_contain_text("completion budget exhausted during reasoning")
    expect(error).to_contain_text("8192 reasoning tokens")
    # The old wording is gone from this card, which is the half of the
    # fix a passing "contains the new text" assertion would miss.
    expect(error).not_to_contain_text("empty response")
    # And the count in the sentence is the count on the card.
    expect(card.get_by_test_id("metric-reasoning")).to_have_text("8192")


def test_review_repro_a_standard_card_offers_both_remedies(bench):
    """WINDOW: the same card at the standard budget, reading the remedy
    the frontend appends.

    THE REMEDY WAS ABOUT TO BE WITHDRAWN SILENTLY. The old branch keyed
    on the substring "finish_reason: length" and the honest label does
    not contain it, so relabelling alone would have removed "try
    extended budget" from precisely the cards it was written for while
    every existing test stayed green. The branch reads the token counts
    now, which is both why it survives this rewording and why it fires
    on a route where the server could not label anything.
    """
    page = bench(["stub/exhausted"])
    check_all_chips(page)
    run(page, "standard remedy")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_contain_text(
        "try extended budget or a lower reasoning effort"
    )


def test_review_repro_an_extended_card_offers_the_only_remedy_left(bench):
    """WINDOW: the same card with the extended budget selected.

    EXTENDED IS THE TOP TIER, so "try extended budget" is not advice, it
    is where the reader already is. Offering it would be the card
    telling someone to do the thing that just failed, at four times the
    price. What is left that they can actually turn is the reasoning
    effort under Experiment controls, so that is what the card says.

    Keyed on the counts like the standard case, so this fires whether or
    not the reservation reached the provider.
    """
    page = bench(["stub/exhausted"])
    check_all_chips(page)
    page.get_by_test_id("budget-extended").click()
    run(page, "extended remedy")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    error = card.get_by_test_id("card-error")
    expect(error).to_contain_text("try a lower reasoning effort")
    expect(error).not_to_contain_text("try extended budget")


def test_a_healthy_reasoning_card_is_left_alone(bench):
    """WINDOW: a live card's terminal render, the same window as the
    firing cases, so the silence is observed where the warning would
    have been.

    The other side of the rule, and the one that keeps it useful.

    stub/thinker returns real text alongside real reasoning tokens,
    which is row 694's shape: thinking happened, an answer came out, and
    there is nothing to warn about. A relabelling that fired here would
    put a cost warning on every reasoning model that worked, and a
    warning that appears on healthy cards is one nobody reads on the
    sick ones.
    """
    page = bench(["stub/thinker"])
    check_all_chips(page)
    run(page, "healthy reasoning")

    card = cards(page).first
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_have_count(0)


def test_review_repro_the_replayed_budget_badge_says_it_is_a_cap(bench, open_history):
    """WINDOW: a history replay's card tools, which is the only place the
    budget badge is drawn.

    THE PHANTOM 44K, ON THE PAGE. The badge read "budget 65536" next to a
    reasoning metric reading 21350. Both are token figures and nothing
    said they answer different questions, so the difference looked like
    44186 tokens that had gone somewhere unexplained. They had gone
    nowhere: one number is the ceiling the bench sent, the other is what
    the provider counted, and subtracting them is not an operation that
    means anything.

    The badge now says "cap", which is the entire remedy, and the title
    says which number to compare it against instead.
    """
    page = bench(["stub/exhausted"])
    check_all_chips(page)
    page.get_by_test_id("budget-extended").click()
    run(page, "cap label replay")
    expect(status_of(cards(page).first)).to_have_text("error", timeout=DONE_TIMEOUT)

    open_history()
    row = page.get_by_test_id("history-row").filter(has_text="cap label replay")
    row.first.click()

    badge = page.get_by_test_id("budget-note").first
    expect(badge).to_have_text("budget cap 65536")
    expect(badge).to_have_attribute("title", re.compile("Not a count"))


# ---- R4: the indicator. Every case below is a TOKEN SHAPE. The stub
# ---- personalities are named for what they do with a budget, and the
# ---- parametrization is over the numbers, so nothing here would change
# ---- if every model in the catalog were renamed tomorrow.
INDICATOR_SHAPES = (
    # (stub personality, shape described, expected indicator text or None)
    (
        "stub/exhausted",
        "all of the output was thinking, no answer at all",
        "100% of completion tokens went to reasoning, not to the answer",
    ),
    (
        "stub/mostly-thinking",
        "an answer arrived, but 94% of the output was thinking",
        "94% of completion tokens went to reasoning, not to the answer",
    ),
    (
        "stub/thinker",
        "row 694's shape: thought hard, answered, 87%",
        None,
    ),
    (
        "stub/fast",
        "a short answer with a little thinking behind it",
        None,
    ),
)


@pytest.mark.parametrize(
    "model,shape,expected",
    INDICATOR_SHAPES,
    ids=[row[1] for row in INDICATOR_SHAPES],
)
def test_review_repro_the_card_flags_output_that_went_to_thinking(
    bench, model, shape, expected
):
    """WINDOW: a live card's terminal render, between the metrics row and
    the response body.

    WHY AN INDICATOR AND NOT JUST THE ERROR LABEL. The server can only
    relabel a result it synthesized an error for, which means a result
    with no visible text at all. The stub/mostly-thinking row is the case
    that proves the gap: it ANSWERS, so it is a successful card carrying
    no error to relabel, and 94% of what it was billed for went somewhere
    the reader cannot see. Nothing on that card said so.

    THIS IS ALSO THE UNSUPPORTED-ROUTE CASE, which is why it reads the
    counts rather than anything about the request. A provider that
    ignored the reservation entirely still reports its usage honestly, so
    these two numbers arrive true whether or not the cap was ever
    applied. Measured against the live catalog on 2026-08-12: of 276
    models carrying a reasoning descriptor, 10 set supports_max_tokens.
    The indicator is not a fallback for a rare case; it is the coverage
    for most of them.

    THE QUIET ROWS ARE HALF THE TEST. A warning that appears on cards
    that worked is one nobody reads on the cards that did not, so
    row 694's shape and an ordinary short answer are both asserted
    silent.
    """
    page = bench([model])
    check_all_chips(page)
    run(page, "indicator " + shape)

    card = cards(page).first
    expect(status_of(card)).not_to_have_text("working", timeout=DONE_TIMEOUT)
    indicator = card.get_by_test_id("reasoning-warning")
    if expected is None:
        expect(indicator).to_have_count(0)
    else:
        expect(indicator).to_have_text(expected)


def test_the_indicator_survives_a_history_replay(bench, open_history):
    """WINDOW: a replayed card's render, after a full round trip
    through the database, so the counts are the stored ones rather than
    the ones the stream held in memory.

    The same card, come back to later.

    A stored row carries both counts, so the indicator is re-derivable
    and must be re-derived: a warning that existed only while you watched
    would be missing from every card anyone returned to, which is every
    card in a comparison worth citing.
    """
    page = bench(["stub/mostly-thinking"])
    check_all_chips(page)
    run(page, "indicator replay")
    expect(status_of(cards(page).first)).to_have_text("done", timeout=DONE_TIMEOUT)

    open_history()
    row = page.get_by_test_id("history-row").filter(has_text="indicator replay")
    row.first.click()

    expect(cards(page).first.get_by_test_id("reasoning-warning")).to_have_text(
        "94% of completion tokens went to reasoning, not to the answer"
    )


def test_review_repro_a_rerun_does_not_keep_the_previous_indicator(bench):
    """WINDOW: one card across two attempts, from the first terminal
    render through the rerun's.

    A STALE WARNING IS A FALSE ONE. The indicator is a card-level sibling
    of the body, so clearing the body on a rerun leaves it standing. A
    card that reran to a clean answer would go on saying that most of its
    output went to reasoning, describing tokens the attempt on screen
    never spent, and the reader has no way to tell that the line is about
    a result that no longer exists.

    ASSERTED ON ONE CARD ACROSS A BOUNDARY, which is why the stub is
    fail-once rather than two personalities. Two cards, one warned and
    one not, would pass whether or not anything was ever cleaned up. The
    only thing that proves cleanup is the same element going away.

    The exact defect save-warn already had, and the fix is the same list
    it is cleaned from.
    """
    page = bench(["stub/exhausted-once"])
    check_all_chips(page)
    run(page, "rerun clears the indicator")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    # PRE-STATE: the indicator is really there before the rerun, so its
    # later absence means removal and not that it never drew.
    expect(card.get_by_test_id("reasoning-warning")).to_have_text(
        "100% of completion tokens went to reasoning, not to the answer"
    )

    card.get_by_test_id("tool-rerun").click()

    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("reasoning-warning")).to_have_count(0)


def test_review_repro_a_refusal_is_not_told_its_budget_ran_out(bench):
    """WINDOW: a live card's terminal render for a model that thought a
    little and then refused.

    THE DEFECT THIS BRANCH SHIPPED AND THIS FIXES. The shape test is
    close to a tautology on the empty-response path: it is only reached
    when there is no visible text, so any reasoning model that thought
    and said nothing has reasoning at or near its completion count
    whatever went wrong. Every firing row in the original table happened
    to carry finish_reason "length", so the table certified the label as
    honest and could not see that a refusal lands in the same bucket.

    stub/filtered spends 37 tokens of a 16384 budget, 0.2%, and stops on
    a content filter. It was being told "completion budget exhausted
    during reasoning", and because static/stream.js keyed its remedy on
    the same predicate, the card then advised buying the extended tier
    and running again: a four times larger retry, at real cost, for a
    failure no budget can fix. The wording it replaced never did that,
    because the old remedy gate required the substring
    "finish_reason: length".

    The card now states the accounting fact without the causal claim,
    and offers no budget remedy at all.
    """
    page = bench(["stub/filtered"])
    check_all_chips(page)
    run(page, "filtered refusal")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    error = card.get_by_test_id("card-error")
    # The accounting fact survives: where the money went and why it
    # stopped are both on the card.
    expect(error).to_contain_text("nearly all of the completion went to reasoning")
    expect(error).to_contain_text("37 reasoning tokens")
    expect(error).to_contain_text("finish_reason: content_filter")
    # The claim the numbers do not support is gone.
    expect(error).not_to_contain_text("exhausted")
    # And so is the advice that would have cost money.
    expect(error).not_to_contain_text("try extended budget")


def test_a_real_truncation_still_gets_the_causal_label_and_the_remedy(bench):
    """WINDOW: the same render, for the shape where the claim is true.

    The other side of the gate. Narrowing a label is only correct if it
    still fires where it belongs, and stub/exhausted is the incident's
    own shape: reasoning equal to completion, finish_reason "length".
    """
    page = bench(["stub/exhausted"])
    check_all_chips(page)
    run(page, "true truncation")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    error = card.get_by_test_id("card-error")
    expect(error).to_contain_text("completion budget exhausted during reasoning")
    expect(error).to_contain_text("try extended budget or a lower reasoning effort")


def test_review_repro_the_reservation_reaches_the_wire_in_a_browser_run(
    bench, stub_url
):
    """WINDOW: the payloads the stub actually recorded, for a vouched
    model and an unvouched one in the same run.

    UNTIL THIS TEST EXISTED THE GATE HAD NEVER FIRED IN A BROWSER RUN.
    No stub catalog entry published supported_parameters, and
    fetch_catalog requires "reasoning" to be in that list, so
    may_send_reasoning_cap was False for every model and
    app.state.reasoning_defaults booted empty in every browser session.
    Every browser assertion about the reasoning field was testing the
    ungated path, and no mutation of the gate could have failed one.

    BOTH ANSWERS IN ONE RUN, because a test that only showed the cap
    arriving would pass just as well with the gate deleted.
    """
    page = bench(["stub/exhausted", "stub/fast"])
    check_all_chips(page)
    run(page, "gate on the wire")
    for i in range(2):
        expect(status_of(cards(page).nth(i))).not_to_have_text(
            "working", timeout=DONE_TIMEOUT
        )

    recorded = httpx.get(stub_url + "/_test/requests", trust_env=False).json()[
        "requests"
    ]
    sent = {
        r["model"]: r
        for r in recorded
        if r["messages"][0]["content"] == "gate on the wire"
    }

    # VOUCHED: mandatory, budget support, parameter advertised. The cap
    # rides, and it is half the budget this run was actually sent.
    vouched = sent["stub/exhausted"]
    assert vouched["reasoning"] == {"max_tokens": vouched["max_tokens"] // 2}

    # UNVOUCHED: publishes no reasoning descriptor at all. Nothing is
    # sent, and the payload is the pre-feature payload.
    assert "reasoning" not in sent["stub/fast"]


def test_review_repro_a_clamped_budget_is_not_told_to_buy_a_bigger_one(bench):
    """WINDOW: a live card for a model whose published cap is below the
    standard tier, at the EXTENDED tier.

    THE DEFECT. The advice branched on the tier the user had selected,
    not on the ceiling the run was actually sent. effective_budget
    clamps the requested tier to a model's published completion cap, so
    stub/capped-thinker sends 4096 whichever tier is chosen. Telling
    that reader to try extended budget is telling them to spend four
    times as much to send a byte-identical request.

    NO EXISTING TEST COULD SEE THIS: every other exhaustion fixture
    publishes no cap, so standard and extended genuinely differ there
    and the tier-keyed code passed.

    The card still says where the money went; only the advice that
    cannot work is gone.
    """
    page = bench(["stub/capped-thinker"])
    check_all_chips(page)
    page.get_by_test_id("budget-extended").click()
    run(page, "clamped budget")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    error = card.get_by_test_id("card-error")
    # The accounting fact survives.
    expect(error).to_contain_text("completion budget exhausted during reasoning")
    # The impossible advice does not.
    expect(error).not_to_contain_text("extended budget")
    # And the cap it turns on is visible on the live card, not only on
    # replays, because this is the surface where somebody is deciding
    # whether to spend again.
    expect(card.get_by_test_id("budget-note")).to_have_text("budget cap 4096")


def test_review_repro_a_replayed_card_carries_the_same_remedy(bench, open_history):
    """WINDOW: one card twice, live and then replayed from History.

    THE DEFECT. fillColumn passed no shownError, so completeColumn fell
    back to the server's bare sentence and every replayed card lost the
    advice entirely, on the surface where somebody is deciding whether
    to spend again. The remedy is derived inside completeColumn now, so
    both paths reach it.
    """
    page = bench(["stub/exhausted"])
    check_all_chips(page)
    run(page, "replay remedy")
    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    live = card.get_by_test_id("card-error").inner_text()
    assert "lower reasoning effort" in live

    open_history()
    page.get_by_test_id("history-row").filter(has_text="replay remedy").first.click()

    replayed = cards(page).first.get_by_test_id("card-error")
    expect(replayed).to_have_text(live)


def test_review_repro_a_whitespace_only_answer_is_not_a_blank_success(bench):
    """WINDOW: a live card whose only visible output is whitespace.

    THE DEFECT. `if text:` was true for spaces and newlines, so
    response_text was set, no error was synthesized, and a fully billed
    reasoning burn rendered as a clean done card that displayed nothing
    at all. No error means no label, no remedy and no indicator: a
    worse outcome than the original incident, because even the useless
    sentence was gone.
    """
    page = bench(["stub/whitespace"])
    check_all_chips(page)
    run(page, "whitespace only")

    card = cards(page).first
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_contain_text(
        "completion budget exhausted during reasoning"
    )
