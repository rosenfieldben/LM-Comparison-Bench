"""Phase H browser tests: the controls row, and history badges that show a
chosen control and never a default.

The truth rule under test is rule two. A badge is a claim that someone
chose something, so a badge for a control nobody set would misrepresent a
provider default as a decision, and it would do it in the one view a
reader consults to find out what an old comparison was. Every assertion
here is therefore two-sided: the badges that must appear, and the absence
of every other one.
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
        chips.nth(i).click()


def open_controls(page):
    page.get_by_test_id("toggle-controls").click()
    expect(page.get_by_test_id("experiment-controls")).to_be_visible()


def run_and_wait(page, prompt, model_count):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()
    for i in range(model_count):
        expect(status_of(cards(page).nth(i))).to_have_text("done", timeout=DONE_TIMEOUT)


def row_for(page, prompt):
    return page.get_by_test_id("history-row").filter(has_text=prompt)


def badge_texts(scope):
    return [scope.nth(i).inner_text().strip() for i in range(scope.count())]


def test_the_controls_row_starts_blank_and_collapsed(bench):
    """Blank renders as blank. A control pre-filled with the tool's own
    default would be rule one broken at the first opportunity: the user
    would see 1.0 in the box, believe they were holding temperature
    constant, and the payload would carry a value nobody chose."""
    page = bench(["stub/fast"])

    expect(page.get_by_test_id("experiment-controls")).to_be_hidden()
    expect(page.get_by_test_id("toggle-controls")).to_have_attribute(
        "aria-expanded", "false"
    )
    open_controls(page)
    expect(page.get_by_test_id("toggle-controls")).to_have_attribute(
        "aria-expanded", "true"
    )

    for testid in ("ctl-system", "ctl-temperature", "ctl-top-p", "ctl-seed"):
        expect(page.get_by_test_id(testid)).to_have_value("", timeout=2000)
    # The selects sit on their unset option, which is the empty value.
    for testid in ("ctl-effort", "ctl-routing"):
        expect(page.get_by_test_id(testid)).to_have_value("")
    # Nothing set, so nothing claimed.
    expect(page.get_by_test_id("controls-summary")).to_have_text("")


def test_the_seed_control_says_what_it_does_and_does_not_promise(bench):
    """Seed honesty belongs where the control is, not only in the README.
    Providers differ in whether they accept a seed and in how deterministic
    they are with one, and a user who reads a bench as promising
    reproducibility will draw a conclusion the bench cannot support."""
    page = bench(["stub/fast"])
    open_controls(page)

    title = page.get_by_test_id("ctl-seed").get_attribute("title")

    assert "determinism is not guaranteed" in title.lower(), title
    assert "recorded" in title.lower(), title


def test_history_badges_render_for_set_controls_and_nothing_else(bench):
    """Two controls set, two badges, and not a third. The negative half is
    the point: temperature and seed were chosen here, top_p and reasoning
    effort were not, and a badge for either of those would be inventing a
    decision."""
    page = bench(["stub/fast", "stub/slow"])
    open_controls(page)
    page.get_by_test_id("ctl-temperature").fill("0.25")
    page.get_by_test_id("ctl-seed").fill("7")
    # The composer says what it is about to send, so a controlled run is
    # never started blind with the panel collapsed.
    expect(page.get_by_test_id("controls-summary")).to_have_text("t=0.25 · seed 7")

    check_all_chips(page)
    run_and_wait(page, "h2 two controls", 2)

    page.get_by_test_id("history-toggle").click()
    expect(page.get_by_test_id("history-list")).to_have_attribute(
        "data-state", "ready", timeout=DONE_TIMEOUT
    )
    row = row_for(page, "h2 two controls")
    expect(row).to_have_count(1)

    assert badge_texts(row.get_by_test_id("history-control-badge")) == [
        "t=0.25",
        "seed 7",
    ]


def test_a_blank_controls_run_renders_no_badges_at_all(bench):
    """The floor for rule two in the view: a comparison that chose nothing
    must say nothing, not "t=1" or "route throughput". Throughput in
    particular is the bench's own default and rides every payload ever
    sent, so it is the badge most likely to appear by accident."""
    page = bench(["stub/fast"])
    check_all_chips(page)
    run_and_wait(page, "h2 no controls at all", 1)

    page.get_by_test_id("history-toggle").click()
    expect(page.get_by_test_id("history-list")).to_have_attribute(
        "data-state", "ready", timeout=DONE_TIMEOUT
    )
    row = row_for(page, "h2 no controls at all")
    expect(row).to_have_count(1)

    expect(row.get_by_test_id("history-control-badge")).to_have_count(0)


def test_the_system_prompt_round_trips_into_the_comparison_view(bench):
    """The row badge says only that a system prompt existed, which is all a
    one-line row can honestly carry; a truncated prompt would invite
    comparing two comparisons on an excerpt that happens to match. The text
    itself has to survive into the view where the experiment is read, or the
    comparison is not reproducible from its own record."""
    system = "You are terse.\nAnswer in one line."
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-system").fill(system)
    page.get_by_test_id("ctl-effort").select_option("high")

    check_all_chips(page)
    run_and_wait(page, "h2 system prompt round trip", 1)

    page.get_by_test_id("history-toggle").click()
    expect(page.get_by_test_id("history-list")).to_have_attribute(
        "data-state", "ready", timeout=DONE_TIMEOUT
    )
    row = row_for(page, "h2 system prompt round trip")
    # Presence, not text, in the row.
    assert badge_texts(row.get_by_test_id("history-control-badge")) == [
        "sys",
        "effort high",
    ]

    row.first.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical", timeout=DONE_TIMEOUT
    )
    # Full text in the comparison view, line breaks intact.
    expect(page.get_by_test_id("run-system-prompt")).to_have_text(system)
    assert badge_texts(page.get_by_test_id("run-control-badge")) == [
        "sys",
        "effort high",
    ]


def test_a_replayed_comparisons_controls_do_not_outlive_it(bench):
    """A stale controls line under a new banner would attribute one
    comparison's experiment to another, which is the same class of defect as
    a badge for an unchosen control: the view would state something nobody
    did."""
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-seed").fill("11")
    check_all_chips(page)
    run_and_wait(page, "h2 controls then a bare run", 1)

    page.get_by_test_id("history-toggle").click()
    expect(page.get_by_test_id("history-list")).to_have_attribute(
        "data-state", "ready", timeout=DONE_TIMEOUT
    )
    row_for(page, "h2 controls then a bare run").first.click()
    expect(page.get_by_test_id("run-control-badge")).to_have_count(
        1, timeout=DONE_TIMEOUT
    )

    # A live run takes the view over, and the replayed line goes with it.
    page.get_by_test_id("ctl-seed").fill("")
    run_and_wait(page, "h2 the bare run itself", 1)

    expect(page.get_by_test_id("run-control-badge")).to_have_count(0)
    expect(page.get_by_test_id("run-system-prompt")).to_have_count(0)


def test_an_out_of_range_control_blocks_the_run_instead_of_erroring_five_cards(
    bench,
):
    """A control the browser marks invalid would 422 at the boundary once
    per model, arriving as a lineup full of identical unexplained errors
    after the group POST had already been refused. Saying so before the
    money moves is both cheaper and more honest."""
    page = bench(["stub/fast"])
    open_controls(page)
    check_all_chips(page)
    page.get_by_test_id("prompt-input").fill("h2 out of range")
    expect(page.get_by_test_id("run-button")).to_be_enabled()

    page.get_by_test_id("ctl-temperature").fill("5")

    expect(page.get_by_test_id("run-button")).to_be_disabled()
    expect(page.get_by_test_id("controls-note")).to_contain_text("temperature")
    # And it recovers: the guard is not a one-way door.
    page.get_by_test_id("ctl-temperature").fill("1.5")
    expect(page.get_by_test_id("run-button")).to_be_enabled()
    expect(page.get_by_test_id("controls-note")).to_have_text("")


def test_the_group_row_records_the_controls_the_run_declared(bench, bench_url):
    """The stored record, checked through the API rather than through the
    view that renders it, so a badge cannot pass by being drawn from
    something other than what was persisted."""
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-temperature").fill("0.5")
    page.get_by_test_id("ctl-routing").select_option("price")
    check_all_chips(page)
    run_and_wait(page, "h2 stored record", 1)

    entries = page.evaluate("() => fetch('/runs?limit=100').then((r) => r.json())")
    entry = next(e for e in entries["runs"] if "h2 stored record" in e["prompt_text"])

    assert entry["type"] == "group"
    assert entry["params"] == {"temperature": 0.5, "routing": "price"}
    # And the payload that went out carried exactly those, with routing
    # landing in the provider object rather than as a top-level key.
    detail = page.evaluate(
        "(id) => fetch('/groups/' + id).then((r) => r.json())", entry["id"]
    )
    sent = json.loads(detail["runs"][0]["results"][0]["request_json"])
    assert sent["temperature"] == 0.5
    assert sent["provider"]["sort"] == "price"
    assert "routing" not in sent
    for absent in ("top_p", "seed", "reasoning"):
        assert absent not in sent, absent
