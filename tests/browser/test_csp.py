"""CSP enforcement proof (F3.3).

The full content security policy is only a security gain if the page runs
clean under it. A broken policy does not error the server; it silently
blocks a script, style, or fetch in the browser, which a green server-side
header test would never notice. So this drives the critical path with a
securitypolicyviolation collector and a console-error capture both armed
before first paint, and fails on any violation or error.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000


def test_csp_enforced_clean_across_critical_path(page, bench):
    # Arm the collectors before the bench fixture navigates, so a load-time
    # violation (a blocked script, style, or font) is caught too. The init
    # script runs before any page script, and console captures the mirror
    # the browser logs for every violation.
    console_errors = []
    page.on(
        "console",
        lambda m: console_errors.append(m.text) if m.type == "error" else None,
    )
    page.on("pageerror", lambda e: console_errors.append("pageerror: " + str(e)))
    page.add_init_script(
        "window.__csp = [];"
        "document.addEventListener('securitypolicyviolation', (e) => {"
        "  window.__csp.push(e.effectiveDirective + ' blocked ' + e.blockedURI);"
        "});"
    )

    # Navigate (fonts, stylesheet, favicon, and every module script load
    # here, all same-origin, all under the policy).
    bench(["stub/fast", "stub/stall0"])

    def chips():
        return page.get_by_test_id("lineup-chip")

    def cards():
        return page.get_by_test_id("result-card")

    # Run: exercises connect-src on POST /groups and /compare/stream and the
    # style manipulation of the race bars.
    chips().nth(0).click()
    chips().nth(1).click()
    page.get_by_test_id("prompt-input").fill("csp path")
    page.get_by_test_id("run-button").click()
    expect(cards().first.get_by_test_id("card-status")).to_have_text(
        "done", timeout=DONE_TIMEOUT
    )
    expect(cards().nth(1).get_by_test_id("card-body")).to_contain_text(
        "partial text", timeout=DONE_TIMEOUT
    )

    # Stop the stalled model.
    page.get_by_test_id("stop-button").click()
    expect(cards().nth(1).get_by_test_id("card-status")).to_have_text(
        "stopped", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("run-button")).to_be_enabled()

    # Library: exercises POST /prompts and GET /prompts.
    page.get_by_test_id("prompt-input").fill("csp saved prompt")
    page.get_by_test_id("save-prompt").click()
    page.get_by_test_id("prompt-name").fill("csp name")
    page.get_by_test_id("confirm-save").click()
    expect(page.get_by_test_id("linked-name")).to_have_text(
        "linked: csp name", timeout=DONE_TIMEOUT
    )

    # History: exercises GET /runs and a replay through GET /groups.
    page.get_by_test_id("history-toggle").click()
    row = page.get_by_test_id("history-row").filter(has_text="csp path")
    expect(row).to_have_count(1, timeout=DONE_TIMEOUT)
    row.first.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical comparison", timeout=DONE_TIMEOUT
    )

    violations = page.evaluate("() => window.__csp")
    assert violations == [], f"CSP violations: {violations}"
    assert console_errors == [], f"console errors: {console_errors}"
