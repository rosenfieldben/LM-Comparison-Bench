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
    """THE SURFACE: what the rater can SEE, which is bench_chrome, the
    rendered text of everything the bench puts on a card minus the answer
    body. Deliberately not the markup: that is a different and stronger
    question, asked by
    test_review_repro_the_blind_entry_never_paints_an_identity, and the
    two together are what separate "could the rater see it" from "did the
    page ever receive it".

    The measurement is the whole point: a rater who can see which model
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
        # ABSENT, not hidden. The old path painted these and then set
        # hidden on them, so "is it hidden" was the strongest question
        # available; the server-issued view never receives them, so the
        # count is the honest assertion and to_be_hidden would now pass
        # vacuously on an element that does not exist.
        expect(cards(page).nth(i).locator(".metrics")).to_have_count(0)
        expect(cards(page).nth(i).get_by_test_id("card-model")).to_have_count(0)
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
    """THE WINDOW: from the submit click to the reveal, which must
    contain the save. A rater who saw the answer inside that window and
    then changed their mind would produce a sighted rating the record
    calls blind.

    Reveal after persistence, not before. A rater who saw the answer
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

    # Identities ARRIVE. Not "come back": this view never had them, so
    # the reveal writes them onto the cards for the first time.
    text_after = bench_chrome(page)
    assert "stub/fast" in text_after
    assert "stub/slow" in text_after
    for i in range(2):
        expect(cards(page).nth(i).get_by_test_id("card-model")).to_be_visible()
        # Cost and timing stay off, by design and not by omission. The
        # blind view renders a minimal card rather than reusing the
        # normal renderer, so there is no metrics block to restore; the
        # numbers are a click away in the ordinary replay.
        expect(cards(page).nth(i).locator(".metrics")).to_have_count(0)

    # What was sent: one entry per card, each carrying the label the card
    # wore and the rating clicked under it.
    assert len(posted) == 1
    payload = json.loads(posted[0])
    # A TOKEN AND NO CLAIM. The page used to post blind: true, which is
    # the one thing it cannot honestly say about its own past. It now
    # sends only the token the server handed it when it opened the
    # session, and the server decides.
    assert "blind" not in payload
    assert payload["blind_token"]
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
    """THE WINDOW: after the reveal, on the page that performed it.

    The reveal is one way. Re-blinding after it would produce a rating
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
    """THE WINDOW: a save that failed, from the error to whatever the
    page does next. Nothing may be revealed inside it.

    A rater who saw the identities after a failed save would have to
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
    # No dataset path was given, so the eligible count came from the
    # score rows and is a floor. The caveat travels with the number
    # rather than living in a doc nobody opens.
    expect(page.get_by_test_id("report-threshold-note")).to_contain_text("FLOOR")


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


# ---- Phase I.2 J6: the blind session is opened by the server.


def blind_view_markup(page):
    """The blind view's own HTML, minus the answers themselves.

    inner_html rather than inner_text, because the question here is what
    the page RECEIVED rather than what it displayed: a hidden node still
    means the mapping crossed the wire.

    Scoped rather than whole-document for two reasons, both of which are
    limits on the claim rather than conveniences. The history list names
    the models of every comparison in a row title and always has, and a
    SET of names is not an answer key. And the answer bodies are excluded
    for the reason bench_chrome gives: the bench cannot blind what a
    model wrote about itself, and this suite's stub answers "reply from
    stub/fast", so including them would test the stub rather than the
    bench.
    """
    return page.evaluate(
        """() => {
            const parts = [document.getElementById("run-controls").innerHTML];
            for (const card of document.querySelectorAll(
                '[data-testid="result-card"]'
            )) {
                const copy = card.cloneNode(true);
                for (const body of copy.querySelectorAll(".card-body")) {
                    body.remove();
                }
                parts.push(copy.innerHTML);
            }
            return parts.join(" ");
        }"""
    )


def test_review_repro_the_blind_entry_never_paints_an_identity(
    bench, open_history, bench_url
):
    """The claim the old path could not make.

    Blind rating used to start from a comparison already on screen, so
    the identities were painted and then hidden. Hidden in a browser is a
    style rule anyone can undo, plus a frame the rater may have seen, and
    the record still said blind. Entered from the list against a
    server-issued session, the page never receives an identity at all.

    Asserted on the DOCUMENT rather than on what is visible, which is the
    opposite of the earlier blind tests and deliberately so: those ask
    "could the rater see it", this asks "did the page ever have it".
    """
    page = bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    run_and_wait(page, "i6 blind entry", 2)

    open_history()
    page.get_by_test_id("history-rate-blind").first.click()
    expect(page.get_by_test_id("rating-panel")).to_be_visible()

    # The MARKUP of the blind view, not merely what is visible in it,
    # which is the opposite of the earlier blind tests and deliberate:
    # those ask "could the rater see it", this asks "did the page ever
    # receive it". Scoped to the results area and the rating controls,
    # because that is where a MAPPING would have to live. The history
    # list names the models of every comparison in a row title and
    # always has; a set of names is not an answer key, and the rater
    # opened that panel themselves.
    markup = blind_view_markup(page)
    assert "stub/fast" not in markup
    assert "stub/slow" not in markup
    expect(cards(page)).to_have_count(2)
    labels = page.get_by_test_id("rating-label")
    assert sorted(labels.nth(i).inner_text() for i in range(2)) == ["A", "B"]

    # And the answers are there: they are what is being rated. This is
    # also the residual limit made visible, since this stub's answer
    # names its own model and no arrangement of the page can fix that.
    assert "reply from" in page.locator("#results").inner_text()


def test_the_server_issued_session_reveals_only_after_the_save(
    bench, open_history, bench_url
):
    """THE WINDOW: from the last rating click to the reveal, over a
    server-issued session. The identities arrive with the reveal, written
    onto the cards for the first time, because there was never a hidden
    copy to un-hide."""
    page = bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    run_and_wait(page, "i6 blind reveal", 2)
    open_history()
    page.get_by_test_id("history-rate-blind").first.click()
    expect(page.get_by_test_id("rating-panel")).to_be_visible()

    for i in range(2):
        cards(page).nth(i).get_by_test_id("rating-4").click()
    assert "stub/fast" not in blind_view_markup(page)

    page.get_by_test_id("rating-submit").click()
    expect(page.get_by_test_id("rating-msg")).to_contain_text("Identities revealed")

    markup = blind_view_markup(page)
    assert "stub/fast" in markup
    assert "stub/slow" in markup

    # The server recorded the blind, and it recorded it from its own
    # session rather than from the page's word.
    rows = page.request.get(bench_url + "/runs?limit=5").json()["runs"]
    group_id = next(e["id"] for e in rows if e["type"] == "group")
    group = page.request.get(bench_url + f"/groups/{group_id}").json()
    ids = [r["id"] for run in group["runs"] for r in run["results"]]
    assert ids


# ---- Phase I.3: the report view renders what the report now says.


def run_experiment_for_report(page, bench_url, tmp_path, name, lineup, task):
    """One experiment over one task, run to a terminal state.

    Takes the lineup and the task so the report-view cases can differ in
    the SHAPE they exercise (a duplicated arm, a judged rubric) rather
    than in how they got there.
    """
    dataset = tmp_path / f"{name.replace(' ', '-')}.jsonl"
    dataset.write_text(json.dumps(task) + "\n", encoding="utf-8")
    created = page.request.post(
        bench_url + "/experiments",
        data={
            "name": name,
            "dataset_path": str(dataset),
            "lineup": lineup,
            "budget": "standard",
        },
    )
    assert created.ok, created.text()
    eid = created.json()["id"]
    started = page.request.post(
        bench_url + f"/experiments/{eid}/start", data={"dataset_path": str(dataset)}
    )
    assert started.ok, started.text()
    for _ in range(200):
        detail = page.request.get(bench_url + f"/experiments/{eid}").json()
        if detail["status"] not in ("created", "running"):
            break
        page.wait_for_timeout(50)
    assert detail["status"] == "done", detail
    return eid, dataset


def score_and_wait(page, bench_url, eid, dataset, judge_model, expected_series):
    """Run one scoring pass and wait until the report shows its series.

    The report is the wait condition because scoring has no progress
    endpoint, and the report is also what the test is about: waiting on
    the thing under assertion cannot pass on a stale read.
    """
    started = page.request.post(
        bench_url + f"/experiments/{eid}/score",
        data={"dataset_path": str(dataset), "judge_model": judge_model},
    )
    assert started.ok, started.text()
    for _ in range(200):
        report = page.request.get(bench_url + f"/experiments/{eid}/report").json()
        seen = {s["judge_model"] for m in report["models"] for s in m["scorers"]}
        if len(seen) >= expected_series:
            return report
        page.wait_for_timeout(50)
    raise AssertionError(f"scoring never produced {expected_series} series: {seen}")


def column_texts(page, table, testid, expected):
    """One column of one report table, after waiting for it to fill.

    expect() first, because .count() does not retry: reading the DOM the
    instant after BenchReport.show() is dispatched is a read of the page
    before the fetch resolves, and an empty list would then pass any
    assertion phrased as "does not contain".
    """
    cells = page.get_by_test_id(table).get_by_test_id(testid)
    expect(cells).to_have_count(expected)
    return [cells.nth(i).inner_text() for i in range(expected)]


def test_review_repro_a_duplicate_arm_is_two_rows_a_reader_can_tell_apart(
    page, bench, bench_url, tmp_path
):
    """Listing one model twice is how anyone asks how much it varies run
    to run, so the two arms have to be tellable apart on the page.

    The report computes exactly that name and calls it `label`. The view
    printed `model` instead, so a duplicated lineup rendered the SAME
    NAME on two rows carrying different numbers, in all three tables,
    with nothing on screen to say which arm was which. A reader looking
    at the answer to the variation question could not read it.
    """
    eid, _dataset = run_experiment_for_report(
        page,
        bench_url,
        tmp_path,
        "i3 duplicate arm",
        ["stub/fast", "stub/fast"],
        {"id": "t1", "prompt": "hi"},
    )

    bench(["stub/fast"])
    page.evaluate("(id) => window.BenchReport.show(id)", eid)
    expect(page.get_by_test_id("report-panel")).to_be_visible()

    assert column_texts(page, "report-outcomes", "report-model", 2) == [
        "stub/fast #0",
        "stub/fast #1",
    ]
    # Every table, not only the first: the same field was printed in all
    # three, so fixing one would leave two.
    assert column_texts(page, "report-providers", "report-model", 2) == [
        "stub/fast #0",
        "stub/fast #1",
    ]
    # These rows recorded distinct positions, so nothing was
    # reconstructed. The arm caveat has to be EARNED: a hedge on every
    # duplicate lineup would train readers to ignore the one that means
    # something.
    expect(page.get_by_test_id("report-arm-caveat")).to_have_count(0)


def test_review_repro_two_judges_render_as_two_rows_that_name_their_judge(
    page, bench, bench_url, tmp_path
):
    """A series is a (scorer, judge) pair, and the page showed only the
    scorer.

    Two judges of one trial therefore rendered as two rows reading
    "judge", with different means and no column saying why. That looks
    like a bug in the bench rather than like the disagreement it is, and
    a reader has no way to attribute either number.

    The stub scores by JUDGE, 0.9 from stub/fast and 0.4 from stub/slow,
    so a view that merged the series could not produce this table.
    """
    eid, dataset = run_experiment_for_report(
        page,
        bench_url,
        tmp_path,
        "i3 two judges",
        ["stub/fast"],
        # The rubric is a task field rather than a scorer field; a judge
        # scorer without one is refused at load, because the rubric IS
        # the scorer.
        {
            "id": "t1",
            "prompt": "hi",
            "rubric": "score 1 if the response is a reply",
            "scorer": {"kind": "judge"},
        },
    )
    score_and_wait(page, bench_url, eid, dataset, "stub/fast", 1)
    score_and_wait(page, bench_url, eid, dataset, "stub/slow", 2)

    bench(["stub/fast"])
    page.evaluate("(id) => window.BenchReport.show(id)", eid)
    expect(page.get_by_test_id("report-panel")).to_be_visible()

    rows = page.get_by_test_id("report-score-row")
    expect(rows).to_have_count(2)
    assert sorted(column_texts(page, "report-scores", "report-judge", 2)) == [
        "stub/fast",
        "stub/slow",
    ]
    # Two attributed means, kept apart. No combined number anywhere.
    means = [rows.nth(i).inner_text() for i in range(2)]
    assert any("0.90" in text for text in means)
    assert any("0.40" in text for text in means)


def test_review_repro_the_legacy_rate_blind_path_is_gone(bench, open_history, page):
    """The second path removed rather than deprecated.

    It fetched the identified comparison, hid the identities with a style
    rule, and posted blind: true with the ratings. Both halves were
    wrong. Hidden in a browser is a rule anyone can undo plus a frame the
    rater may already have seen, and a page that painted the answer key
    is the one witness that cannot testify about its own blindness.

    Asserted three ways, because "removed" has to mean unreachable
    rather than merely unused. The entry point is not on window. The
    source carries none of the machinery that path needed. And the
    control that used to call it now produces a session the server
    vouches for.
    """
    start_blind(bench, open_history, prompt="i3 legacy path gone")

    # No entry point. A module that still exported it would leave the
    # path one console line away.
    assert page.evaluate("() => Object.keys(window.BenchRating)") == ["startBlind"]

    # No machinery. These existed only to paint identities and then
    # conceal them, which is the thing that must not be possible.
    source = (STATIC / "rating.js").read_text(encoding="utf-8")
    # Comments stripped first. The module's own header describes the path
    # it removed, and a scan that counted that would be asserting on
    # prose rather than on code.
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("//")
    )
    for gone in ("hideIdentity", "showIdentity", "CONCEALED", "cardsOnScreen"):
        assert gone not in code, gone
    # And no client-side blindness claim anywhere in the flow.
    assert "blind: true" not in code
    assert "blind_token" in code


def test_the_rewired_control_opens_a_server_vouched_session(
    bench, bench_url, open_history
):
    """The control kept its place and changed what it does. It now asks
    the server for an anonymized view, and the rating that follows is
    blind because the SERVER says a session was open, not because the
    page claimed one."""
    page = bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    run_and_wait(page, "i3 rewired control", 2)
    replay(page, open_history, "i3 rewired control")
    opened = []
    page.on(
        "request",
        lambda request: (
            opened.append(request.url)
            if request.url.endswith("/blind") and request.method == "POST"
            else None
        ),
    )

    page.get_by_test_id("rate-blind").click()
    expect(page.get_by_test_id("rating-panel")).to_be_visible()

    # It asked the SERVER for the view. The old path asked nobody.
    assert len(opened) == 1, opened
    for i in range(2):
        cards(page).nth(i).get_by_test_id("rating-4").click()
    page.get_by_test_id("rating-submit").click()
    expect(page.get_by_test_id("rating-msg")).to_contain_text("Identities revealed")

    # A real server session existed and was closed one way, which is the
    # server vouching for it: a comparison whose answer key is out cannot
    # be reopened blind, and only a session that was genuinely opened can
    # be in that state.
    group_id = next(
        e["id"]
        for e in page.request.get(bench_url + "/runs?limit=5").json()["runs"]
        if e["type"] == "group"
    )
    reopened = page.request.post(bench_url + f"/groups/{group_id}/blind", data={})
    assert reopened.status == 409, reopened.text()
    assert "already been revealed" in reopened.text()
