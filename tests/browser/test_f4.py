"""Phase F.4 browser tombstones: diff arming across reruns.

Failures and orderings are forced with Playwright route interception, never
with sleeps, and each test keys on its own prompt text so the
session-shared bench server and database stay usable by the other files.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000


def cards(page):
    return page.get_by_test_id("result-card")


def status_of(card):
    return card.get_by_test_id("card-status")


def card_for(page, model):
    return (
        cards(page)
        .filter(has=page.get_by_test_id("card-model").filter(has_text=model))
        .first
    )


def start_run(page, prompt):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()


def check_all_chips(page):
    chips = page.get_by_test_id("lineup-chip")
    for i in range(chips.count()):
        chips.nth(i).click()


def stream_body(text, run_id=1):
    """One complete SSE exchange: a text delta then a done frame carrying
    both the text and an error. Both on purpose: response_text is what makes
    a card diffable, and a non-null error is what earns it the Rerun
    control, so this shape is the only one that exercises arming across a
    rerun. It is also the product's real both-text-and-error contract (a
    stream that died partway), rendered as "diff (partial)". Fulfilled at
    the app boundary so a model's Nth attempt is exactly what the test
    wants, with no dependence on stub personalities or timing.
    """
    return (
        f'data: {{"type":"delta","text":"{text}"}}\n\n'
        f'data: {{"type":"done","result":'
        f'{{"response_text":"{text}","error":"partial stream"}},'
        f'"run_id":{run_id}}}\n\n'
    )


def route_per_attempt(page, plan):
    """Serve each model a scripted sequence of response bodies, one per
    attempt, so a rerun deterministically returns different text than the
    attempt it replaces.
    """
    attempts = {}

    def handler(route):
        data = (route.request.post_data or "").replace(" ", "")
        model = next(m for m in plan if f'"model":"{m}"' in data)
        n = attempts.get(model, 0)
        attempts[model] = n + 1
        bodies = plan[model]
        body = bodies[min(n, len(bodies) - 1)]
        route.fulfill(status=200, content_type="text/event-stream", body=body)

    page.route("**/compare/stream", handler)
    return attempts


def test_review_repro_rerun_of_armed_card_disarms_diff(bench):
    """Second review: disarmDiff() ran only from startRun, so a per-card
    rerun reset the column and destroyed the armed button while armedDiff
    persisted. A later diff then completed against a detached, invisible
    prior attempt. Arming the rerun card must be dropped.

    Arm card A, rerun card A, then click diff on card B. Pre-fix that
    second click found a stale armedDiff and rendered a comparison against
    A's vanished first attempt. Post-fix B's click only arms B, so no
    comparison against the vanished attempt is offered or rendered.
    """
    page = bench(["stub/fast", "stub/slow"])
    route_per_attempt(
        page,
        {
            "stub/fast": [stream_body("alpha first"), stream_body("alpha rerun")],
            "stub/slow": [stream_body("beta only")],
        },
    )
    check_all_chips(page)
    start_run(page, "f4 disarm armed rerun")

    a = card_for(page, "stub/fast")
    b = card_for(page, "stub/slow")
    expect(status_of(a)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(status_of(b)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(a.get_by_test_id("card-body")).to_contain_text("alpha first")

    # Arm card A on its first attempt.
    a.get_by_test_id("tool-diff").click()
    expect(a.get_by_test_id("tool-diff")).to_have_attribute("aria-pressed", "true")

    # Rerun card A. Its first attempt, the armed one, is gone.
    a.get_by_test_id("tool-rerun").click()
    expect(a.get_by_test_id("card-body")).to_contain_text(
        "alpha rerun", timeout=DONE_TIMEOUT
    )

    # Card B's diff click must merely arm B, not complete a comparison
    # against A's destroyed attempt.
    b.get_by_test_id("tool-diff").click()
    expect(page.get_by_test_id("diff-panel")).to_be_hidden()
    expect(b.get_by_test_id("tool-diff")).to_have_attribute("aria-pressed", "true")
    # The vanished attempt's text appears nowhere: not in a panel, not in a
    # card. "alpha rerun" is on screen; "alpha first" is not.
    expect(page.locator("#page")).not_to_contain_text("alpha first")


def test_rerun_of_other_card_keeps_arming(bench):
    """The scope control: rerunning a DIFFERENT card must leave arming
    alone, since a live-vs-live diff across cards is the point. Arm A,
    rerun B, then diff A against B's new result and see it complete.
    """
    page = bench(["stub/fast", "stub/slow"])
    route_per_attempt(
        page,
        {
            "stub/fast": [stream_body("gamma stays armed")],
            "stub/slow": [stream_body("delta first"), stream_body("delta rerun")],
        },
    )
    check_all_chips(page)
    start_run(page, "f4 arming survives other rerun")

    a = card_for(page, "stub/fast")
    b = card_for(page, "stub/slow")
    expect(status_of(a)).to_have_text("error", timeout=DONE_TIMEOUT)
    expect(status_of(b)).to_have_text("error", timeout=DONE_TIMEOUT)

    # Arm A, then rerun the other card.
    a.get_by_test_id("tool-diff").click()
    expect(a.get_by_test_id("tool-diff")).to_have_attribute("aria-pressed", "true")
    b.get_by_test_id("tool-rerun").click()
    expect(b.get_by_test_id("card-body")).to_contain_text(
        "delta rerun", timeout=DONE_TIMEOUT
    )

    # A is still armed, so B's click completes the comparison.
    expect(a.get_by_test_id("tool-diff")).to_have_attribute("aria-pressed", "true")
    b.get_by_test_id("tool-diff").click()
    panel = page.get_by_test_id("diff-panel")
    expect(panel).to_be_visible()
    body = page.get_by_test_id("diff-body")
    expect(body).to_contain_text("gamma stays armed")
    expect(body).to_contain_text("delta rerun")
    page.get_by_test_id("diff-close").click()
