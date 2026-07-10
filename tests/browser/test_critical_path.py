"""Critical-path browser suite: the real app, real Chromium, stub upstream.

The suite is deliberately sequential: the bench server and its database
live for the whole session, and the replay and diff tests read history
that the earlier run tests created. Run the file as a whole
(pytest -m browser), not as cherry-picked test ids.

Every assertion goes through user-visible state or data-testid, never
styling classes or internal JS variables, so a future redesign breaks
these tests only when behavior breaks.
"""

import httpx
import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

# Long enough for the slow personality's 2s silence plus CI jitter.
DONE_TIMEOUT = 15_000

# Live-run card state captured in test 02 and compared against the
# history replay in test 05: replay goes through the same renderer, so
# every visible string must match exactly.
RECORDED: dict[str, dict[str, str]] = {}

METRIC_KEYS = ("ttft", "total", "tok", "cost")


def cards(page):
    return page.get_by_test_id("result-card")


def status_of(card):
    return card.get_by_test_id("card-status")


def check_all_chips(page):
    chips = page.get_by_test_id("lineup-chip")
    for i in range(chips.count()):
        chips.nth(i).click()


def start_run(page, prompt):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()


def capture_card(card) -> dict[str, str]:
    state = {"body": card.get_by_test_id("card-body").inner_text()}
    for key in METRIC_KEYS:
        state[key] = card.get_by_test_id(f"metric-{key}").inner_text()
    return state


def test_01_boot_renders_pre_volt_lineup_as_chips(bench):
    page = bench(["stub/fast", "stub/slow", "stub/capped"])

    chips = page.get_by_test_id("lineup-chip")
    expect(chips).to_have_count(3)
    # Chips display the short name; the full id survives in the title.
    for i, model in enumerate(["stub/fast", "stub/slow", "stub/capped"]):
        expect(chips.nth(i)).to_have_attribute("title", model)
        expect(chips.nth(i)).to_contain_text(model.split("/")[1])


def test_02_two_model_run_streams_in_checkbox_order(bench):
    page = bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    start_run(page, "two model run")

    expect(cards(page)).to_have_count(2)
    fast, slow = cards(page).nth(0), cards(page).nth(1)
    expect(fast.get_by_test_id("card-model")).to_have_text("stub/fast")
    expect(slow.get_by_test_id("card-model")).to_have_text("stub/slow")

    # The fast column finishes while the slow one still shows its
    # running state with the elapsed counter; that overlap is the
    # entire point of per-model streaming.
    expect(status_of(fast)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(status_of(slow)).to_contain_text("thinking")
    expect(status_of(slow)).to_have_text("done", timeout=DONE_TIMEOUT)

    expect(fast.get_by_test_id("card-body")).to_have_text("reply from stub/fast")
    expect(slow.get_by_test_id("card-body")).to_have_text("slow reply from stub/slow")
    for card in (fast, slow):
        for key in METRIC_KEYS:
            expect(card.get_by_test_id(f"metric-{key}")).not_to_have_text("—")
        expect(card.get_by_test_id("metric-tok")).to_have_text("13/8")
        expect(card.get_by_test_id("metric-cost")).to_contain_text("$")

    RECORDED["stub/fast"] = capture_card(fast)
    RECORDED["stub/slow"] = capture_card(slow)


def test_03_flaky_errors_then_rerun_recovers_in_same_group(bench):
    page = bench(["stub/flaky"])
    check_all_chips(page)
    start_run(page, "flaky prompt")

    card = cards(page).nth(0)
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_contain_text("stub flaky failure")

    card.get_by_test_id("tool-rerun").click()
    expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-body")).to_have_text("reply from stub/flaky")

    # Both attempts persist as one comparison group: failures are data.
    page.get_by_test_id("history-toggle").click()
    row = page.get_by_test_id("history-row").filter(has_text="flaky prompt")
    expect(row).to_have_count(1)
    expect(row.get_by_test_id("history-count")).to_have_text("2 models")


def test_04_extended_budget_sends_65536_and_clamps_capped(bench, stub_url):
    page = bench(["stub/fast", "stub/capped"])
    check_all_chips(page)
    page.get_by_test_id("budget-extended").click()
    start_run(page, "budget probe")
    for i in range(2):
        expect(status_of(cards(page).nth(i))).to_have_text(
            "done", timeout=DONE_TIMEOUT
        )

    # The budget assertion belongs at the stub: what was actually sent,
    # not what the UI believes it selected.
    recorded = httpx.get(stub_url + "/_test/requests").json()["requests"]
    sent = {
        r["model"]: r["max_tokens"]
        for r in recorded
        if r["messages"][0]["content"] == "budget probe"
    }
    assert sent == {"stub/fast": 65536, "stub/capped": 4096}

    # Extended costs real money, so the control must not survive a
    # reload.
    page.reload()
    expect(page.get_by_test_id("budget-standard")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(page.get_by_test_id("budget-extended")).to_have_attribute(
        "aria-pressed", "false"
    )


def test_05_history_replay_matches_live_run_exactly(bench):
    assert RECORDED, "test 02 must run first in this session"
    page = bench(["stub/fast", "stub/slow"])

    page.get_by_test_id("history-toggle").click()
    page.get_by_test_id("history-row").filter(has_text="two model run").click()

    expect(page.get_by_test_id("run-label")).to_contain_text("Historical comparison")
    expect(cards(page)).to_have_count(2)
    # Replay flows through the same renderer as a live run, so every
    # user-visible string must survive the round trip through sqlite.
    for i, model in enumerate(["stub/fast", "stub/slow"]):
        card = cards(page).nth(i)
        expect(card.get_by_test_id("card-model")).to_have_text(model)
        expect(status_of(card)).to_have_text("done")
        replayed = capture_card(card)
        live = RECORDED[model]
        assert replayed["body"] == live["body"]
        for key in METRIC_KEYS:
            assert replayed[key] == live[key], (model, key)


def test_06_diff_marks_changes_and_leaves_common_text_plain(bench):
    page = bench(["stub/fast", "stub/slow"])

    page.get_by_test_id("history-toggle").click()
    page.get_by_test_id("history-row").filter(has_text="two model run").click()
    expect(cards(page)).to_have_count(2)

    cards(page).nth(0).get_by_test_id("tool-diff").click()
    cards(page).nth(1).get_by_test_id("tool-diff").click()

    panel = page.get_by_test_id("diff-panel")
    expect(panel).to_be_visible()
    body = page.get_by_test_id("diff-body")
    expect(body.locator("del")).to_contain_text("stub/fast")
    expect(body.locator("ins").last).to_contain_text("stub/slow")
    # The shared words flow as plain text, tinted spans mark only the
    # differences.
    expect(body).to_contain_text("reply from")
    assert body.locator("del", has_text="reply").count() == 0
    assert body.locator("ins", has_text="reply").count() == 0

    page.get_by_test_id("diff-close").click()
    expect(panel).to_be_hidden()


def test_07_html_output_renders_inert_everywhere(bench):
    dialogs = []
    page = bench(["stub/html", "stub/fast"])
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    check_all_chips(page)
    start_run(page, "injection probe")
    for i in range(2):
        expect(status_of(cards(page).nth(i))).to_have_text(
            "done", timeout=DONE_TIMEOUT
        )

    html_card = cards(page).nth(0)
    body = html_card.get_by_test_id("card-body")
    expect(body).to_contain_text("<img src=x onerror=alert(1)>")
    expect(body).to_contain_text("<b>bold?</b>")
    assert body.locator("img").count() == 0
    assert body.locator("b").count() == 0

    # The collapsed preview is the same node folded, but the probe is
    # cheap and a regression here would be an injection, so assert it.
    html_card.get_by_test_id("tool-fold").click()
    assert body.locator("img").count() == 0
    assert body.locator("b").count() == 0

    cards(page).nth(0).get_by_test_id("tool-diff").click()
    cards(page).nth(1).get_by_test_id("tool-diff").click()
    diff_body = page.get_by_test_id("diff-body")
    expect(diff_body).to_contain_text("<b>bold?</b>")
    assert diff_body.locator("img").count() == 0
    assert diff_body.locator("b").count() == 0

    assert dialogs == []


def test_08_null_content_shows_finish_reason_error(bench):
    page = bench(["stub/null"])
    check_all_chips(page)
    start_run(page, "null prompt")

    card = cards(page).nth(0)
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("card-error")).to_contain_text(
        "empty response (finish_reason: content_filter)"
    )

    # The rerun path must survive a model that errors every time.
    card.get_by_test_id("tool-rerun").click()
    expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(card.get_by_test_id("tool-rerun")).to_be_visible()


def test_09_saved_prompts_persist_reload_and_report_duplicates(bench):
    page = bench(["stub/fast"])

    page.get_by_test_id("prompt-input").fill("the saved text")
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill("smoke")
    page.get_by_test_id("confirm-save").click()
    expect(page.get_by_test_id("linked-name")).to_have_text("linked: smoke")

    page.reload()
    page.get_by_test_id("saved-prompts").select_option(label="smoke")
    expect(page.get_by_test_id("prompt-input")).to_have_value("the saved text")

    page.get_by_test_id("prompt-input").fill("different text, same name")
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill("smoke")
    page.get_by_test_id("confirm-save").click()
    expect(page.get_by_test_id("prompt-msg")).to_have_text(
        "a prompt with that name already exists"
    )


def test_10_seven_model_fanout_loses_and_reorders_nothing(bench):
    # The semaphore stagger itself is a backend property with unit
    # coverage; at the UI the contract is completeness and order.
    models = ["stub/fast"] + [f"stub/wide{i}" for i in range(6)]
    page = bench(models)
    check_all_chips(page)
    start_run(page, "seven wide")

    expect(cards(page)).to_have_count(7)
    for i, model in enumerate(models):
        card = cards(page).nth(i)
        expect(card.get_by_test_id("card-model")).to_have_text(model)
        expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
        expect(card.get_by_test_id("card-body")).to_have_text(f"reply from {model}")


def test_11_chip_controls_carry_their_own_accessible_names(bench):
    page = bench(["stub/fast", "stub/slow"])

    # The label owns only the checkbox, so the checkbox is named by the
    # model id alone; the remove button names its action and target
    # itself. Before the chip restructure the wrapping label absorbed
    # the remove button's text into the checkbox's name.
    chip = page.get_by_test_id("lineup-chip").first
    expect(chip.get_by_role("checkbox")).to_have_accessible_name("stub/fast")
    expect(chip.get_by_role("button")).to_have_accessible_name(
        "Remove stub/fast from lineup"
    )
