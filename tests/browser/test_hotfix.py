"""Hotfix tombstones: a broken display helper must not strand a card.

The incident these encode was a field failure, diagnosed externally and
verified by reproduction. Static assets were served with validators but no
Cache-Control, so after the Phase G upgrade a browser could legally run
fresh modules beside a stale pre-G lib.js with no fmtBilled. The session
bar picked that missing function the moment the session went non-idle,
finish() threw before it had rendered anything, and its own idempotence
guard, already set, turned the catch's recovery call into a no-op. Every
run finishing after the first billed charge sat at "thinking" forever with
its complete answer visible and the console silent.

The breakage is injected by serving a real module with one line appended.
Where that line lands decides which half of the mechanism fires, and the
difference is not cosmetic: render.js destructures fmtBilled at load time
while state.js looks it up on every call. Breaking it before render.js
loads breaks the card render; breaking it after breaks only the session
bar, which is what the field saw, because in the old order the session bar
ran first.
"""

from pathlib import Path

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000
STATIC = Path(__file__).resolve().parents[2] / "static"

# Distinctive enough that finding it in a console line proves it came from
# here and not from some unrelated warning.
INJECTED = "hotfix-repro-fmtBilled-is-broken"

BREAKAGE = (
    "\n// Appended by the browser suite. A stale lib.js has no fmtBilled at\n"
    "// all, so calling it is a TypeError; a function that throws is the\n"
    "// same failure with a message the test can recognize.\n"
    'window.BenchLib.fmtBilled = function () { throw new Error("'
    + INJECTED
    + '"); };\n'
)


def serve_with_breakage(route, module):
    """Serve the real module, plus the line that breaks fmtBilled."""
    body = (STATIC / module).read_text(encoding="utf-8") + BREAKAGE
    route.fulfill(status=200, content_type="text/javascript", body=body)


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


def collect_console_errors(page):
    errors = []
    page.on(
        "console",
        lambda msg: errors.append(msg.text) if msg.type == "error" else None,
    )
    return errors


def test_review_repro_a_broken_session_bar_cannot_strand_a_card(page, bench):
    """The field mechanism, verbatim.

    boot.js is the injection point because it is the last module to load:
    render.js has already bound the working fmtBilled by then, so the cards
    themselves render, the accounting lands, the session goes non-idle, and
    the session bar is the one thing that breaks. That is exactly the order
    the incident followed.

    On pre-fix code every card here strands at "thinking" with its answer
    visible and nothing in the console. After the fix the session bar
    failure costs the session bar and nothing else: every card is terminal,
    and the failure is on the console instead of nowhere.

    The route pattern tolerates a trailing query because the served index
    now versions every asset URL.
    """
    errors = collect_console_errors(page)
    page.route(
        "**/static/boot.js*", lambda route: serve_with_breakage(route, "boot.js")
    )

    bench(["stub/fast", "stub/slow", "stub/html"])
    check_all_chips(page)
    start_run(page, "hotfix broken session bar")

    expect(cards(page)).to_have_count(3)
    for index in range(3):
        card = cards(page).nth(index)
        # Terminal, not stranded. Every one of these finishes after the
        # session has gone non-idle except the first, which is the point:
        # the arrival-order half of the incident is that the later ones
        # were the casualties.
        expect(status_of(card)).to_have_text("done", timeout=DONE_TIMEOUT)
        assert card.get_by_test_id("card-body").inner_text().strip() != ""

    # Surviving is half the fix; surfacing is the other half. Swallowing is
    # what let this run for a whole session unnoticed.
    assert any(INJECTED in line for line in errors), errors


def test_review_repro_a_broken_card_render_recovers_instead_of_stranding(page, bench):
    """The other half of the ordering fix: the terminal render itself
    throwing.

    Injecting through lib.js breaks the binding render.js takes at load, so
    fillMetrics throws while drawing the billed cost and the card's own
    render is what fails. The flag is now set after the render rather than
    before, so runOne's catch re-enters finish() and draws the synthetic
    error card instead of finding the guard already closed.

    A card showing an error it did not really suffer is a worse answer than
    a correct one and a far better answer than a permanent lie. On pre-fix
    code these cards strand.
    """
    errors = collect_console_errors(page)
    page.route("**/static/lib.js*", lambda route: serve_with_breakage(route, "lib.js"))

    bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    start_run(page, "hotfix broken card render")

    expect(cards(page)).to_have_count(2)
    for index in range(2):
        card = cards(page).nth(index)
        expect(status_of(card)).to_have_text("error", timeout=DONE_TIMEOUT)
        # The streamed answer is still there under the error: the recovery
        # renders over what arrived rather than discarding it.
        body = card.get_by_test_id("card-body").inner_text()
        assert "reply from" in body, body
        expect(card.get_by_test_id("card-error")).to_contain_text(INJECTED)

    assert any(INJECTED in line for line in errors), errors


def test_a_working_page_logs_no_console_errors(page, bench):
    """The guard against over-correcting: with nothing broken, the new
    logging must stay silent. A fix that made every ordinary run noisy
    would train the console to be ignored, which is how the original
    silence became expensive."""
    errors = collect_console_errors(page)

    bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    start_run(page, "hotfix quiet on the happy path")

    for index in range(2):
        expect(status_of(cards(page).nth(index))).to_have_text(
            "done", timeout=DONE_TIMEOUT
        )
    assert errors == [], errors
