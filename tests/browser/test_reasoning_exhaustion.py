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
    """The other side of the rule, and the one that keeps it useful.

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
