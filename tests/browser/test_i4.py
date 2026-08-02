"""Phase I4 browser tests: blind rating hides identity until it is saved.

The claim under test is not "a class was added" but "a rater looking at
this page cannot tell which model wrote which answer". So the assertions
read the rendered document rather than the code's intentions: the model
ids must be absent from what is visible, and they must appear only after
the ratings are already in the database.
"""

import json

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
        box = chips.nth(i).locator("input[type=checkbox]")
        if not box.is_checked():
            chips.nth(i).click()


def run_and_wait(page, prompt, model_count):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()
    for i in range(model_count):
        expect(status_of(cards(page).nth(i))).to_have_text("done", timeout=DONE_TIMEOUT)


def row_for(page, prompt):
    return page.get_by_test_id("history-row").filter(has_text=prompt)


def replay(page, open_history, prompt):
    open_history()
    row_for(page, prompt).first.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical", timeout=DONE_TIMEOUT
    )


def bench_chrome(page):
    """Everything the BENCH puts on screen about a card, minus the answer.

    inner_text rather than the HTML, deliberately: hidden elements are
    excluded, which is exactly the question being asked. Checking the
    markup would find the model id sitting in a hidden node and call the
    blind broken when nobody could see it.

    The answer body is excluded because the bench cannot blind it. A
    model that writes "as an AI assistant made by X" has identified
    itself, and no amount of hiding chrome changes that; the stub in this
    suite does exactly that, answering "reply from stub/fast". So the
    claim under test is the one the bench can actually make: nothing the
    bench itself renders about a card names the model. The residual limit
    is documented in the README rather than pretended away.
    """
    parts = [page.locator("#run-controls").inner_text()]
    for i in range(cards(page).count()):
        card = cards(page).nth(i)
        for selector in (".card-header", ".metrics", ".card-caption"):
            region = card.locator(selector)
            if region.count():
                parts.append(region.inner_text())
    return "\n".join(parts)


def start_blind(bench, open_history, prompt="i4 blind rating"):
    """Run a two-model comparison, replay it, and enter blind mode."""
    page = bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    run_and_wait(page, prompt, 2)
    replay(page, open_history, prompt)
    page.get_by_test_id("rate-blind").click()
    expect(page.get_by_test_id("rating-panel")).to_be_visible()
    return page


def test_blind_rating_hides_every_identity_until_the_ratings_are_saved(
    bench, open_history
):
    """The measurement is the whole point: a rater who can see which model
    wrote an answer is not rating the answer. Identity, provider, cost and
    timing all go, because a rater who knows one answer cost ten times
    another can often guess which model it was, and a guess is not a
    blind."""
    page = start_blind(bench, open_history)

    chrome = bench_chrome(page)
    assert "stub/fast" not in chrome
    assert "stub/slow" not in chrome
    expect(cards(page)).to_have_count(2)
    # Neutral labels, one per card, and no metrics to read them by.
    labels = page.get_by_test_id("rating-label")
    expect(labels).to_have_count(2)
    assert sorted(labels.nth(i).inner_text() for i in range(2)) == ["A", "B"]
    for i in range(2):
        expect(cards(page).nth(i).locator(".metrics")).to_be_hidden()
        expect(cards(page).nth(i).get_by_test_id("card-model")).to_be_hidden()
    # The answers themselves stay visible; they are what is being rated.
    assert "reply from" in page.locator("#results").inner_text()


def test_submit_stays_disabled_until_every_card_is_rated(bench, open_history):
    """A partial set would be a blind rating of some answers and silence
    about the rest, which is a different and worse thing to average."""
    page = start_blind(bench, open_history)
    submit = page.get_by_test_id("rating-submit")

    expect(submit).to_be_disabled()
    expect(page.get_by_test_id("rating-count")).to_have_text("0 of 2 rated")

    cards(page).nth(0).get_by_test_id("rating-4").click()
    expect(page.get_by_test_id("rating-count")).to_have_text("1 of 2 rated")
    expect(submit).to_be_disabled()

    cards(page).nth(1).get_by_test_id("rating-2").click()
    expect(page.get_by_test_id("rating-count")).to_have_text("2 of 2 rated")
    expect(submit).to_be_enabled()


def test_the_reveal_happens_only_after_the_save_and_maps_correctly(
    bench, bench_url, open_history
):
    """Reveal after persistence, not before. A rater who saw the answer
    and then changed their mind would be producing a sighted rating the
    record calls blind.

    The mapping is checked against the posted payload rather than against
    the screen, because the payload is what becomes the record: label A
    was rated 5, and label A must turn out to have been whichever model
    the reveal names.
    """
    page = start_blind(bench, open_history)
    posted = []
    page.on(
        "request",
        lambda request: (
            posted.append(request.post_data)
            if request.url.endswith("/ratings") and request.method == "POST"
            else None
        ),
    )

    by_label = {}
    for i in range(2):
        label = cards(page).nth(i).get_by_test_id("rating-label").inner_text()
        by_label[label] = i
    cards(page).nth(by_label["A"]).get_by_test_id("rating-5").click()
    cards(page).nth(by_label["B"]).get_by_test_id("rating-1").click()

    # Still blind at the moment of clicking submit.
    assert "stub/fast" not in bench_chrome(page)

    page.get_by_test_id("rating-submit").click()
    expect(page.get_by_test_id("rating-msg")).to_contain_text("Identities revealed")

    # Identities are back, and the metrics with them.
    text_after = bench_chrome(page)
    assert "stub/fast" in text_after
    assert "stub/slow" in text_after
    for i in range(2):
        expect(cards(page).nth(i).locator(".metrics")).to_be_visible()

    # What was sent: one entry per card, each carrying the label the card
    # wore and the rating clicked under it.
    assert len(posted) == 1
    payload = json.loads(posted[0])
    assert payload["blind"] is True
    sent = {r["label"]: r for r in payload["ratings"]}
    assert sent["A"]["rating"] == 5
    assert sent["B"]["rating"] == 1

    # And the labels map onto real, distinct models, which is what makes
    # the record auditable after the fact.
    model_by_result = {}
    group_id = next(
        e["id"]
        for e in page.request.get(bench_url + "/runs?limit=5").json()["runs"]
        if e["type"] == "group"
    )
    group = page.request.get(bench_url + f"/groups/{group_id}").json()
    for run in group["runs"]:
        for result in run["results"]:
            model_by_result[result["id"]] = result["model"]
    rated_models = {label: model_by_result[r["result_id"]] for label, r in sent.items()}
    assert set(rated_models.values()) == {"stub/fast", "stub/slow"}


def test_a_rated_card_cannot_be_rated_again_after_the_reveal(bench, open_history):
    """The reveal is one way. Re-blinding after it would produce a rating
    the record calls blind that was made by someone who had already seen
    the answer, which is worse than no blind rating at all."""
    page = start_blind(bench, open_history)
    cards(page).nth(0).get_by_test_id("rating-4").click()
    cards(page).nth(1).get_by_test_id("rating-4").click()
    page.get_by_test_id("rating-submit").click()
    expect(page.get_by_test_id("rating-msg")).to_contain_text("Identities revealed")

    for i in range(2):
        for value in range(1, 6):
            expect(
                cards(page).nth(i).get_by_test_id(f"rating-{value}")
            ).to_be_disabled()
    expect(page.get_by_test_id("rating-submit")).to_have_count(0)


def test_a_failed_save_does_not_reveal(bench, open_history):
    """A rater who saw the identities after a failed save would have to
    re-rate with the answer in front of them, and that second rating would
    not be blind however it was labelled."""
    page = start_blind(bench, open_history)
    cards(page).nth(0).get_by_test_id("rating-3").click()
    cards(page).nth(1).get_by_test_id("rating-3").click()
    page.route("**/ratings", lambda route: route.abort())

    page.get_by_test_id("rating-submit").click()

    expect(page.get_by_test_id("rating-msg")).to_contain_text("failed to save")
    assert "stub/fast" not in bench_chrome(page)
    # And it can be retried, because the ratings are still held.
    expect(page.get_by_test_id("rating-submit")).to_be_enabled()


def test_blind_rating_is_not_offered_for_a_single_answer(bench, open_history):
    """The method is hiding which of SEVERAL answers came from which
    model. One card has nothing to be blind between."""
    page = bench(["stub/fast", "stub/slow"])
    page.get_by_test_id("lineup-chip").nth(0).click()
    run_and_wait(page, "i4 single card", 1)
    replay(page, open_history, "i4 single card")

    expect(page.get_by_test_id("rate-blind")).to_have_count(0)


def test_a_view_change_ends_the_rating_session(bench, open_history):
    """The session points at cards that a new view replaces. Staleness is
    decided by the view epoch rather than by a cleanup call at each of the
    view-change sites, which would work until someone added another."""
    page = start_blind(bench, open_history)
    cards(page).nth(0).get_by_test_id("rating-3").click()

    # A fresh live run takes the view over.
    check_all_chips(page)
    run_and_wait(page, "i4 after the session", 2)

    expect(page.get_by_test_id("rating-panel")).to_have_count(0)
    expect(page.get_by_test_id("rating-row")).to_have_count(0)
    # And the new cards show their identities: nothing stayed concealed.
    assert "stub/fast" in bench_chrome(page)
