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
    """Check every chip, idempotently.

    Clicking a chip toggles it, so a blind click sweep run twice unchecks
    everything and leaves Run disabled. That is easy to write and confusing
    to debug, since the failure surfaces as a timeout on Run rather than on
    the lineup.
    """
    chips = page.get_by_test_id("lineup-chip")
    for i in range(chips.count()):
        box = chips.nth(i).locator("input[type=checkbox]")
        if not box.is_checked():
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


def test_history_badges_render_for_set_controls_and_nothing_else(bench, open_history):
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

    open_history()
    row = row_for(page, "h2 two controls")
    expect(row).to_have_count(1)

    assert badge_texts(row.get_by_test_id("history-control-badge")) == [
        "t=0.25",
        "seed 7",
    ]


def test_a_blank_controls_run_renders_no_badges_at_all(bench, open_history):
    """The floor for rule two in the view: a comparison that chose nothing
    must say nothing, not "t=1" or "route throughput". Throughput in
    particular is the bench's own default and rides every payload ever
    sent, so it is the badge most likely to appear by accident."""
    page = bench(["stub/fast"])
    check_all_chips(page)
    run_and_wait(page, "h2 no controls at all", 1)

    open_history()
    row = row_for(page, "h2 no controls at all")
    expect(row).to_have_count(1)

    expect(row.get_by_test_id("history-control-badge")).to_have_count(0)


def test_the_system_prompt_round_trips_into_the_comparison_view(bench, open_history):
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

    open_history()
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


def test_a_replayed_comparisons_controls_do_not_outlive_it(bench, open_history):
    """A stale controls line under a new banner would attribute one
    comparison's experiment to another, which is the same class of defect as
    a badge for an unchosen control: the view would state something nobody
    did."""
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-seed").fill("11")
    check_all_chips(page)
    run_and_wait(page, "h2 controls then a bare run", 1)

    open_history()
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
    # reasoning briefly left this list, while the visible-output
    # reservation rode every request, and came back when the reservation
    # was gated on the catalog. stub/fast publishes no reasoning
    # descriptor, so the bench does not vouch for it and sends no cap.
    # An unset control and an unvouched reservation are both absent, and
    # this run's payload is the pre-feature payload.
    for absent in ("top_p", "seed", "reasoning"):
        assert absent not in sent, absent


# ---- H3: reuse a comparison.


def controls_now(page):
    return page.evaluate("() => BenchControls.experimentParams()")


def test_reuse_round_trips_prompt_controls_and_the_stored_record(bench, open_history):
    """The round trip that makes then-versus-now possible: run a controlled
    comparison, reuse it, run it again, and the second group's stored record
    must equal the first's. Anything less and the two runs are not the same
    experiment, which is the only reason to compare them.

    The composer is read through BenchControls.experimentParams, the same
    function the run path uses, so this asserts what would actually be sent
    rather than what the inputs happen to display.
    """
    system = "You are terse.\nOne line only."
    page = bench(["stub/fast", "stub/slow"])
    open_controls(page)
    page.get_by_test_id("ctl-system").fill(system)
    page.get_by_test_id("ctl-temperature").fill("0.25")
    page.get_by_test_id("ctl-seed").fill("7")
    page.get_by_test_id("ctl-effort").select_option("high")
    page.get_by_test_id("ctl-routing").select_option("price")
    check_all_chips(page)
    run_and_wait(page, "h3 reuse round trip", 2)

    first = page.evaluate(
        """() => fetch('/runs?limit=100').then((r) => r.json())
              .then((d) => d.runs.find((e) =>
                  e.prompt_text.includes('h3 reuse round trip')))"""
    )
    assert first["type"] == "group"

    # Wipe the composer so the prefill has to do all the work, and prove the
    # wipe took: a test that passed because the values were never cleared
    # would assert nothing.
    page.get_by_test_id("prompt-input").fill("something else entirely")
    page.evaluate("() => BenchControls.setExperimentParams(null)")
    assert controls_now(page) == {}

    open_history()
    row_for(page, "h3 reuse round trip").first.click()
    expect(page.get_by_test_id("reuse-comparison")).to_be_visible(timeout=DONE_TIMEOUT)
    page.get_by_test_id("reuse-comparison").click()

    # Prompt and every set control are back, read through the run path's own
    # accessor.
    expect(page.get_by_test_id("prompt-input")).to_have_value("h3 reuse round trip")
    assert controls_now(page) == first["params"]
    assert controls_now(page) == {
        "system": system,
        "temperature": 0.25,
        "seed": 7,
        "effort": "high",
        "routing": "price",
    }

    # Prefill only: money moves on an explicit Run and on nothing else.
    # Card count is the wrong signal here, because opening the row replayed
    # the stored comparison and those cards are the replay's. What proves
    # nothing ran is that the view is still the historical one and history
    # still holds exactly one entry for this prompt.
    expect(page.get_by_test_id("run-label")).to_contain_text("Historical")
    assert (
        page.evaluate(
            """() => fetch('/runs?limit=100').then((r) => r.json())
                  .then((d) => d.runs.filter((e) =>
                      e.prompt_text.includes('h3 reuse round trip')).length)"""
        )
        == 1
    )

    # Run it again, and the new group's record equals the old one's.
    check_all_chips(page)
    run_and_wait(page, "h3 reuse round trip", 2)
    second = page.evaluate(
        """() => fetch('/runs?limit=100').then((r) => r.json())
              .then((d) => d.runs.filter((e) =>
                  e.prompt_text.includes('h3 reuse round trip')))"""
    )
    assert len(second) == 2, second
    assert second[0]["params"] == second[1]["params"] == first["params"]
    assert second[0]["id"] != first["id"]


def test_reuse_leaves_the_lineup_alone(bench, open_history):
    """The whole point of then-versus-now is running an old experiment
    against today's models, so which models those are stays the user's
    choice. Restoring the source's lineup would silently answer a question
    the user was asking."""
    page = bench(["stub/fast", "stub/slow"])
    open_controls(page)
    page.get_by_test_id("ctl-seed").fill("3")
    page.get_by_test_id("lineup-chip").nth(0).click()
    run_and_wait(page, "h3 lineup untouched", 1)

    # Change the lineup selection to something the source never used.
    page.get_by_test_id("lineup-chip").nth(0).click()
    page.get_by_test_id("lineup-chip").nth(1).click()
    before = page.evaluate("() => BenchControls.checkedModels()")
    assert before == ["stub/slow"]

    open_history()
    row_for(page, "h3 lineup untouched").first.click()
    page.get_by_test_id("reuse-comparison").click()

    assert page.evaluate("() => BenchControls.checkedModels()") == before


def test_reuse_clears_a_control_the_source_did_not_set(bench, open_history):
    """Reuse reproduces an experiment, so a leftover value is not a
    harmless convenience: it would make the new run a different experiment
    wearing the reused label, and the server would refuse to group it with
    its source for exactly that reason."""
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-seed").fill("5")
    check_all_chips(page)
    run_and_wait(page, "h3 only a seed", 1)

    # Set controls the source never had.
    page.get_by_test_id("ctl-temperature").fill("1.9")
    page.get_by_test_id("ctl-effort").select_option("low")

    open_history()
    row_for(page, "h3 only a seed").first.click()
    page.get_by_test_id("reuse-comparison").click()

    assert controls_now(page) == {"seed": 5}
    expect(page.get_by_test_id("ctl-temperature")).to_have_value("")
    expect(page.get_by_test_id("ctl-effort")).to_have_value("")


def test_reuse_of_a_comparison_with_no_controls_restores_only_the_prompt(
    bench, open_history
):
    """The legacy and no-controls edge. Nothing was set, so nothing is
    restored, and the reuse action still offers itself because a prompt is
    worth reusing on its own."""
    page = bench(["stub/fast"])
    check_all_chips(page)
    run_and_wait(page, "h3 bare comparison", 1)

    open_controls(page)
    page.get_by_test_id("ctl-temperature").fill("0.8")

    open_history()
    row_for(page, "h3 bare comparison").first.click()
    expect(page.get_by_test_id("reuse-comparison")).to_be_visible(timeout=DONE_TIMEOUT)
    page.get_by_test_id("reuse-comparison").click()

    expect(page.get_by_test_id("prompt-input")).to_have_value("h3 bare comparison")
    assert controls_now(page) == {}


def test_reuse_says_out_loud_that_an_ungrouped_source_loses_routing(
    page, bench_url, open_history
):
    """The documented lossy edge, surfaced rather than left silent.

    An ungrouped run has no stored controls set; its controls are derived
    from its recorded payload, and routing is the one control that
    derivation cannot recover, because provider.sort rides every payload the
    bench has ever sent so a stored throughput cannot be told from a chosen
    one. Reuse from such a source is therefore lossy for routing by
    construction, and guessing would be the same truth defect as any other
    default rendered as a choice.

    Lossy is acceptable. Quiet is not. The button says so before the click
    and the composer says so after, and this test holds both.
    """
    # An ungrouped run with controls, built through the API rather than the
    # UI: the browser always creates a group, so this shape comes from a
    # scripted caller or from a pre-groups database.
    made = page.request.post(
        bench_url + "/compare",
        data={
            "prompt": "h3 ungrouped source",
            "models": ["stub/fast"],
            "params": {"temperature": 0.4, "routing": "price"},
        },
    )
    assert made.ok, made.text()

    page.goto(bench_url)
    open_history()
    row = row_for(page, "h3 ungrouped source")
    expect(row).to_have_count(1)
    # The row already shows the asymmetry: temperature is derivable, routing
    # is not, so no route badge appears even though one was sent.
    assert badge_texts(row.get_by_test_id("history-control-badge")) == ["t=0.4"]

    row.first.click()
    reuse = page.get_by_test_id("reuse-comparison")
    expect(reuse).to_be_visible(timeout=DONE_TIMEOUT)
    # Warned before the click.
    assert "routing is not restored" in reuse.get_attribute("title")

    reuse.click()

    # Warned after it, too, and the prefill is honest about what it holds:
    # the temperature it could recover, and no routing it could not.
    expect(page.get_by_test_id("prompt-msg")).to_contain_text("routing is not restored")
    assert controls_now(page) == {"temperature": 0.4}


def test_a_system_prompt_reaches_the_wire_exactly_as_typed(bench, open_history):
    """Closing review. The composer read the system prompt through trim(),
    which decides blankness correctly and alters content on the way past: a
    prompt pasted with indentation or ending in a deliberate newline was
    silently reshaped, while the comparison view went on calling it "the
    system prompt this comparison was run with".

    Whitespace still decides blankness, because a textarea holding only
    spaces is a blank control. It just no longer decides the content.
    """
    system = "  You are terse.\n  Answer in one line.\n"
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-system").fill(system)

    assert controls_now(page) == {"system": system}

    check_all_chips(page)
    run_and_wait(page, "h review exact system prompt", 1)

    open_history()
    row_for(page, "h review exact system prompt").first.click()
    expect(page.get_by_test_id("reuse-comparison")).to_be_visible(timeout=DONE_TIMEOUT)
    # Stored and replayed byte for byte, including the leading spaces and
    # the trailing newline.
    stored = page.evaluate(
        """() => fetch('/runs?limit=100').then((r) => r.json())
              .then((d) => d.runs.find((e) =>
                  e.prompt_text.includes('h review exact system prompt')).params)"""
    )
    assert stored["system"] == system

    # And a whitespace-only prompt is still a blank control, not a system
    # message made of spaces.
    page.get_by_test_id("reuse-comparison").click()
    page.get_by_test_id("ctl-system").fill("   \n  ")
    assert controls_now(page) == {}


def test_reuse_restores_a_system_only_experiment(bench, open_history):
    """Closing review, the reuse edge the spec names: params carrying only a
    system prompt. The row badge for it is presence-only, so this is the case
    where reading badges instead of data would have lost the entire
    experiment while appearing to restore it."""
    system = "Only a system prompt was set."
    page = bench(["stub/fast"])
    open_controls(page)
    page.get_by_test_id("ctl-system").fill(system)
    check_all_chips(page)
    run_and_wait(page, "h review system only", 1)

    page.get_by_test_id("ctl-system").fill("")
    page.get_by_test_id("ctl-seed").fill("99")
    open_history()
    row = row_for(page, "h review system only")
    assert badge_texts(row.get_by_test_id("history-control-badge")) == ["sys"]

    row.first.click()
    page.get_by_test_id("reuse-comparison").click()

    # The text is back from the data, and the seed the source never set is
    # cleared rather than carried along.
    assert controls_now(page) == {"system": system}


def test_the_largest_allowed_seed_survives_the_browser_round_trip(
    page, bench_url, open_history
):
    """The other half of the seed-bound finding, asserted where it failed.

    A seed is only useful if the value you set is the value recorded, badged
    and reused. The old signed-64 bound broke that silently in the browser,
    so this pins the boundary case end to end: set the maximum the API now
    accepts, and the badge, the stored record and the reuse prefill must all
    hold that exact integer.
    """
    biggest = 9007199254740991  # MAX_SEED, and Number.MAX_SAFE_INTEGER
    made = page.request.post(
        bench_url + "/compare",
        data={
            "prompt": "h review max seed",
            "models": ["stub/fast"],
            "params": {"seed": biggest},
        },
    )
    assert made.ok, made.text()

    page.goto(bench_url)
    open_history()
    row = row_for(page, "h review max seed")
    # Read back through the page's own JSON parsing, which is where the
    # corruption happened.
    seen = page.evaluate(
        """() => fetch('/runs?limit=100').then((r) => r.json())
              .then((d) => d.runs.find((e) =>
                  e.prompt_text.includes('h review max seed')).params.seed)"""
    )
    assert seen == biggest, f"browser read {seen}"
    # The badge shows the seed that was actually set, digit for digit.
    assert badge_texts(row.get_by_test_id("history-control-badge")) == [
        f"seed {biggest}"
    ]

    row.first.click()
    page.get_by_test_id("reuse-comparison").click()

    assert controls_now(page) == {"seed": biggest}

    # The invariant that makes the above sufficient: the API refuses a seed
    # the page could not represent, so an unrepresentable one can never
    # reach history at all. Without this the test would only show that the
    # boundary works, not that nothing beyond it can get in.
    refused = page.request.post(
        bench_url + "/compare",
        data={
            "prompt": "h review seed too big",
            "models": ["stub/fast"],
            "params": {"seed": biggest + 1},
        },
    )
    assert refused.status == 422, refused.text()


# ---- H1.2: an incomplete experiment looks incomplete.


def test_a_declared_member_that_never_ran_shows_as_a_gap(page, bench_url, open_history):
    """Third external review, HIGH: the group declared an ordered lineup
    before any call and the detail showed only what actually ran, so a
    comparison that declared four models and recorded two replayed as a
    two-model comparison. That is a quieter lie than an error, because the
    reader has no way to know anything is missing.

    The placeholder is deliberately inert: no rerun control, no diff tools,
    no race entry, nothing counted. There is no result to diff and no timing
    to race, and running the missing member into its declared slot is a real
    capability that this P0 branch is not the place to build.
    """
    gid = page.request.post(
        bench_url + "/groups",
        data={
            "prompt": "h1 incomplete comparison",
            "models": ["stub/fast", "stub/slow"],
            "budget": "standard",
        },
    ).json()["id"]
    # Only one of the two declared members actually runs.
    ran = page.request.post(
        bench_url + "/compare/stream",
        data={
            "prompt": "h1 incomplete comparison",
            "model": "stub/fast",
            "group_id": gid,
            "position": 0,
            "budget": "standard",
        },
    )
    assert ran.ok, ran.text()

    page.goto(bench_url)
    open_history()
    row_for(page, "h1 incomplete comparison").first.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical", timeout=DONE_TIMEOUT
    )

    # One real card, one gap, so the comparison reads as two-wide.
    expect(cards(page)).to_have_count(1)
    gap = page.get_by_test_id("missing-member")
    expect(gap).to_have_count(1)
    expect(gap.get_by_test_id("missing-member-model")).to_have_text("stub/slow")
    expect(gap).to_contain_text("declared at position 1")

    # Inert: none of the controls a real card carries.
    for control in ("tool-rerun", "tool-diff", "tool-copy", "tool-fold"):
        expect(gap.get_by_test_id(control)).to_have_count(0), control
    # And it contributes nothing to the session counters.
    assert page.evaluate("() => BenchState.sessionStats.runs") == 0


def test_review_repro_a_duplicate_lineup_still_shows_its_gap(
    page, bench_url, open_history
):
    """Closing review. The gap detection matched declared members against
    a Set of the models that ran, and a lineup may legitimately declare
    one model twice: neither /compare nor /groups rejects a duplicate, and
    with the seed control a same-model pair is a determinism check rather
    than a mistake. One recorded run then suppressed BOTH placeholders and
    the comparison replayed one-wide with nothing to show it had declared
    two. That is exactly the quiet shrink the placeholders exist to stop,
    surviving in the one lineup shape nobody tested.
    """
    gid = page.request.post(
        bench_url + "/groups",
        data={
            "prompt": "h1 duplicate lineup",
            "models": ["stub/fast", "stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    ran = page.request.post(
        bench_url + "/compare/stream",
        data={
            "prompt": "h1 duplicate lineup",
            "model": "stub/fast",
            "group_id": gid,
            "position": 0,
            "budget": "standard",
        },
    )
    assert ran.ok, ran.text()

    page.goto(bench_url)
    open_history()
    row_for(page, "h1 duplicate lineup").first.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical", timeout=DONE_TIMEOUT
    )

    # One ran, one did not: one card and one gap, naming the slot that is
    # still empty rather than the one that filled.
    expect(cards(page)).to_have_count(1)
    gap = page.get_by_test_id("missing-member")
    expect(gap).to_have_count(1)
    expect(gap).to_contain_text("declared at position 1")


def test_a_legacy_group_replays_without_printing_null(page, bench_url, open_history):
    """Legacy groups return null for every declared field, and a null
    reaching the composer would render the four characters "null" as if
    someone had typed them. Absence has to render as absence.

    Built through the API and then stripped of its declaration, which is the
    shape every group in a pre-H.1 database has.
    """
    gid = page.request.post(
        bench_url + "/groups",
        data={"prompt": "h1 legacy replay", "budget": "standard"},
    ).json()["id"]
    ran = page.request.post(
        bench_url + "/compare/stream",
        data={
            "prompt": "h1 legacy replay",
            "model": "stub/fast",
            "group_id": gid,
            "budget": "standard",
        },
    )
    assert ran.ok, ran.text()

    page.goto(bench_url)
    open_history()
    row_for(page, "h1 legacy replay").first.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical", timeout=DONE_TIMEOUT
    )

    # No declared lineup means no gaps to draw, not a gap for every model.
    expect(page.get_by_test_id("missing-member")).to_have_count(0)

    page.get_by_test_id("reuse-comparison").click()
    prompt = page.get_by_test_id("prompt-input").input_value()
    assert prompt == "h1 legacy replay", prompt
    assert "null" not in prompt
