"""Phase I4 browser tests: blind rating hides identity until it is saved.

The claim under test is not "a class was added" but "a rater looking at
this page cannot tell which model wrote which answer". So the assertions
read the rendered document rather than the code's intentions: the model
ids must be absent from what is visible, and they must appear only after
the ratings are already in the database.
"""

import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000
STATIC = Path(__file__).resolve().parents[2] / "static"


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


def test_the_global_hidden_rule_beats_a_selector_that_declares_display(page, bench):
    """The rule that replaced three per-selector guards.

    The user agent sheet hides [hidden] elements, but any explicit
    display in volt.css outranks it, which is why #name-row,
    #experiment-controls and .metrics each needed saying twice. The last
    of those was not cosmetic: cost and timing stayed on screen through a
    blind rating, and a rater who can see those can often name the model.

    Asserted against a live class that declares display rather than
    against the stylesheet text, because the question is which rule wins
    in the browser and only the browser can answer it.
    """
    bench(["stub/fast"])

    hidden_but_displayed = page.evaluate(
        """() => {
             const probe = document.createElement("div");
             // .metrics declares display: grid, .rating-row declares
             // display: flex. Either would have defeated the user agent
             // rule on its own.
             probe.className = "metrics";
             probe.hidden = true;
             probe.textContent = "probe";
             document.getElementById("results").append(probe);
             const shown = getComputedStyle(probe).display;
             probe.remove();
             return shown;
           }"""
    )

    assert hidden_but_displayed == "none"


def test_the_rating_scale_matches_the_server_constant():
    """The H2 bounds precedent applied to the rating scale.

    The client renders one button per point and the server validates the
    same range, so the two constants are a pair that can drift. A client
    scale wider than the server's would render a button whose click is
    refused; a narrower one would quietly make the top of the scale
    unreachable, and nothing on screen would say so.
    """
    from bench.main import RATING_MAX, RATING_MIN

    source = (STATIC / "rating.js").read_text(encoding="utf-8")
    declared = re.search(r"const RATING_MAX = (\d+);", source)
    declared_min = re.search(r"const RATING_MIN = (\d+);", source)

    assert declared, "static/rating.js must declare RATING_MAX"
    assert int(declared.group(1)) == RATING_MAX
    assert declared_min and int(declared_min.group(1)) == RATING_MIN


# ---- Phase I5: the report view.


def finished_experiment(page, bench_url, tmp_path, name):
    """One two-model experiment over one scored task, run to done.

    Shared by the report-view tests so they differ in what they assert
    rather than in how they got there.
    """
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        '{"id": "t1", "prompt": "hi", "reference": "reply", '
        '"scorer": {"kind": "contains"}}\n',
        encoding="utf-8",
    )
    created = page.request.post(
        bench_url + "/experiments",
        data={
            "name": name,
            "dataset_path": str(dataset),
            "lineup": ["stub/fast", "stub/slow"],
            "budget": "standard",
        },
    )
    assert created.ok, created.text()
    eid = created.json()["id"]
    started = page.request.post(
        bench_url + f"/experiments/{eid}/start",
        data={"dataset_path": str(dataset)},
    )
    assert started.ok, started.text()
    # Drain progress to a terminal state, the same deterministic wait the
    # API tests use.
    for _ in range(200):
        detail = page.request.get(bench_url + f"/experiments/{eid}").json()
        if detail["status"] not in ("created", "running"):
            break
        page.wait_for_timeout(50)
    assert detail["status"] == "done", detail
    return eid, dataset


def test_the_report_leads_with_its_estimand_and_shows_both_axes(
    page, bench, bench_url, tmp_path
):
    """The view's job is to be hard to misread. The estimand leads,
    because a number without it is a number about nothing in particular;
    the trial outcomes and the scoring coverage appear as separate
    tables, because they are separate questions.
    """
    eid, _dataset = finished_experiment(page, bench_url, tmp_path, "i5 report view")

    bench(["stub/fast"])
    page.evaluate("(id) => window.BenchReport.show(id)", eid)
    expect(page.get_by_test_id("report-panel")).to_be_visible()
    expect(page.get_by_test_id("report-estimand")).to_contain_text(
        "routed-service estimand"
    )
    # Two axes, two tables, and the interval's resampling unit named.
    expect(page.get_by_test_id("report-outcomes")).to_be_visible()
    expect(page.get_by_test_id("report-scores")).to_be_visible()
    expect(page.get_by_test_id("report-banner")).to_contain_text("task clusters")
    expect(page.get_by_test_id("report-row")).to_have_count(2)
    # No dataset path was given, so pass rates are unavailable and the
    # view says so rather than leaving empty cells to read as failures.
    expect(page.get_by_test_id("report-threshold-note")).to_contain_text(
        "pass rates are unavailable"
    )


def test_the_report_takes_a_dataset_path_and_says_so_when_it_does_not_match(
    page, bench, bench_url, tmp_path
):
    """The run-start precedent, applied to the read side.

    Thresholds live in the dataset file rather than in the database, so
    pass rates need the file. The path is digest-verified against the one
    recorded at creation, a mismatch is refused in the server's own
    words, and the form survives the refusal so the operator can correct
    the path they can see. Without a path the report degrades to score
    means and says which it is doing.
    """
    eid, dataset = finished_experiment(page, bench_url, tmp_path, "i5 dataset input")
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text(
        '{"id": "t1", "prompt": "different", "reference": "reply", '
        '"scorer": {"kind": "contains"}}\n',
        encoding="utf-8",
    )

    bench(["stub/fast"])
    page.evaluate("(id) => window.BenchReport.show(id)", eid)
    # Degraded and honest about it, with the box there to fix that.
    expect(page.get_by_test_id("report-threshold-note")).to_be_visible()
    expect(page.get_by_test_id("report-dataset-path")).to_have_value("")

    page.get_by_test_id("report-dataset-path").fill(str(wrong))
    page.get_by_test_id("report-dataset-apply").click()

    # Loud, and in the server's words: a status code alone would not say
    # which of the two digests was the one it recorded.
    state = page.get_by_test_id("report-state")
    expect(state).to_have_attribute("data-state", "error")
    expect(state).to_contain_text("dataset changed since this experiment")
    # And the form is still there, holding the path that needs fixing.
    expect(page.get_by_test_id("report-dataset-path")).to_have_value(str(wrong))

    page.get_by_test_id("report-dataset-path").fill(str(dataset))
    page.get_by_test_id("report-dataset-apply").click()

    # The right file, so pass rates become available and the note that
    # said they were not goes away.
    expect(page.get_by_test_id("report-panel")).to_have_attribute(
        "data-state", "ready", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("report-threshold-note")).to_have_count(0)
    # Remembered for the tab and nowhere else: reopening the report with
    # no argument keeps the file, and no storage holds it. A filesystem
    # path is a fact about the operator's machine, and the experiment row
    # deliberately records the file's digest instead.
    page.evaluate("(id) => window.BenchReport.show(id)", eid)
    expect(page.get_by_test_id("report-dataset-path")).to_have_value(str(dataset))
    stored = page.evaluate("() => Object.values(window.localStorage).join('\\u0000')")
    assert str(dataset) not in stored


def test_the_experiment_list_names_its_own_state(page, bench):
    """Same discipline as the history panel: an empty list means two
    different things while a fetch is in flight, so the panel says which
    rather than leaving a blank to be read as "none"."""
    bench(["stub/fast"])

    page.get_by_test_id("experiments-toggle").click()

    expect(page.get_by_test_id("experiment-list")).to_have_attribute(
        "data-state", re.compile(r"ready|empty"), timeout=DONE_TIMEOUT
    )
