"""Phase L browser tests: the seam a person touches to snapshot a clone.

TWO SERVERS, BECAUSE THE POSTURE IS BOOT-SCOPED. BENCH_REPO_ROOTS is read
once at startup, so "snapshots are off" and "snapshots are on" are two
processes rather than two requests, and each half of the control gets
proved against a bench that really is in that state.

EVERY PROOF NAMES ITS WINDOW. The off state and the composed state are
rendered by the same function from two different postures, and a proof
about one of them reads as coverage while leaving the other unguarded.
"""

import json

import pytest
from playwright.sync_api import expect

from bench.main import SNAPSHOTS_OFF

pytestmark = pytest.mark.browser


def test_the_control_says_the_servers_own_sentence_when_snapshots_are_off(bench):
    """WINDOW: the composer's snapshot control on a bench booted with no
    BENCH_REPO_ROOTS.

    THE UI NEVER OFFERS A DOOR THE SERVER WILL REFUSE. The button is
    disabled and the reason is TEXT ON THE PAGE, not a hover title: a
    reason available only to a mouse is a reason no keyboard user ever
    reads, which is the finding that put the run blocker into a line of
    text rather than a tooltip.

    And it is the SERVER'S sentence, asserted against the constant
    itself. A second wording written in JavaScript would be a second
    explanation of one fact, and which one a person got would depend on
    whether they clicked the button or read a 403.
    """
    page = bench(["model/alpha"])
    button = page.get_by_test_id("snapshot-open")
    message = page.get_by_test_id("snapshot-msg")

    expect(button).to_be_disabled()
    expect(message).to_have_text(SNAPSHOTS_OFF)
    assert "BENCH_REPO_ROOTS" in message.inner_text()
    # The panel cannot be opened, so nothing offers a root to type into.
    expect(page.get_by_test_id("snapshot-panel")).to_be_hidden()


def test_composing_a_snapshot_stages_it_as_one_document(snapshot_bench, snapshot_root):
    """WINDOW: the composer's chip list after Compose snapshot on a bench
    whose allowlist contains the clone.

    ONE ATTACHMENT FOR A WHOLE TREE, which is what the chip has to say:
    the face reads "repository snapshot" rather than the stored name,
    because that name is a pure function of the digest and printing it
    would show the digest twice and imply a file that never existed.

    THE MEMBER COUNT IS THE FACT A BYTE COUNT CANNOT GIVE. Two files
    were selected out of three in the tree, so the number on the chip is
    a claim about the SELECTION and is checkable against the patterns
    typed above it.
    """
    page = snapshot_bench(["model/alpha"])
    page.get_by_test_id("snapshot-open").click()
    expect(page.get_by_test_id("snapshot-panel")).to_be_visible()
    page.get_by_test_id("snapshot-root").fill(str(snapshot_root))
    page.get_by_test_id("snapshot-patterns").fill("pkg/*.py")
    page.get_by_test_id("snapshot-compose").click()

    chip = page.get_by_test_id("attachment-chip")
    expect(chip).to_have_count(1)
    expect(page.get_by_test_id("attachment-name")).to_have_text("repository snapshot")
    meta = page.get_by_test_id("attachment-meta").inner_text()
    assert "2 files" in meta
    assert "sha256 " in meta
    # The panel closes on success, so the control returns to the state a
    # person would expect after a completed action.
    expect(page.get_by_test_id("snapshot-panel")).to_be_hidden()


def test_the_staged_snapshot_shows_no_path_anywhere(snapshot_bench, snapshot_root):
    """WINDOW: the whole composer row's rendered text and every title on
    it, after a snapshot is staged.

    K.3'S FILENAME RULE EXTENDED TO A TREE. The clone root is the one
    string in this feature that names somebody's filesystem, and it is
    not a declaration: nothing pins it, nothing composes it into a
    prompt, and no view is allowed to print it. The manifest holds
    repo-relative member paths for the record; the chip holds none of
    them.

    Asserted over the row's text AND its titles, because the previous
    version of this class of bug hid in a title attribute where the
    visible text was already clean.
    """
    page = snapshot_bench(["model/alpha"])
    page.get_by_test_id("snapshot-open").click()
    page.get_by_test_id("snapshot-root").fill(str(snapshot_root))
    page.get_by_test_id("snapshot-patterns").fill("pkg/*.py")
    page.get_by_test_id("snapshot-compose").click()
    expect(page.get_by_test_id("attachment-chip")).to_have_count(1)

    row = page.locator("#attach-row")
    assert str(snapshot_root) not in row.inner_text()
    titles = row.evaluate(
        "el => Array.from(el.querySelectorAll('[title]')).map(n => n.title)"
    )
    for title in titles:
        assert str(snapshot_root) not in title
    # The member paths are in the record, not on the chip: the chip says
    # how many files, and the manifest says which.
    assert "pkg/a.py" not in row.inner_text()


def test_a_refused_snapshot_shows_the_servers_reason_at_the_control(
    snapshot_bench, tmp_path
):
    """WINDOW: the snapshot control after a Compose the server refuses.

    A comparison carrying documents fails CLOSED and the words the
    server used land at the control that caused them, which is the same
    rule the upload path follows. Every refusal this can carry is about
    a choice made here, a root or a pattern, and the remedy is a change
    to that choice.
    """
    page = snapshot_bench(["model/alpha"])
    page.get_by_test_id("snapshot-open").click()
    # A real directory that is not under the allowlist.
    page.get_by_test_id("snapshot-root").fill(str(tmp_path))
    page.get_by_test_id("snapshot-patterns").fill("**/*")
    page.get_by_test_id("snapshot-compose").click()

    message = page.get_by_test_id("snapshot-msg")
    expect(message).to_contain_text("BENCH_REPO_ROOTS")
    expect(page.get_by_test_id("attachment-chip")).to_have_count(0)


def test_a_catalog_that_never_answered_is_not_evidence_the_feature_is_off(
    page, bench_url
):
    """WINDOW: the composer's snapshot control on a page whose /models
    fetch failed.

    ABSENCE OF EVIDENCE IS NOT EVIDENCE. The snapshot posture rides the
    catalog response, so a failed catalog fetch leaves the page not
    knowing whether snapshots are configured. Reading that as "enabled"
    would offer a door the server might refuse; reading it as "off"
    would print the server's BENCH_REPO_ROOTS sentence on a bench where
    that variable may well be set. The control says the third thing,
    which is the only true one, and disables itself either way.

    This is the same rule setDataPolicy follows two fields over: a badge
    is a claim about where prompts go, and a failed fetch is not evidence
    for one.
    """
    page.route("**/models", lambda route: route.abort())
    page.add_init_script("localStorage.setItem('bench-lineup', '[\"model/alpha\"]')")
    page.goto(bench_url)

    button = page.get_by_test_id("snapshot-open")
    message = page.get_by_test_id("snapshot-msg")
    expect(button).to_be_disabled()
    expect(message).to_contain_text("catalog has not answered")
    # Not the server's off sentence, because that is a different fact
    # and this page has not learned it.
    assert "BENCH_REPO_ROOTS" not in message.inner_text()


def test_an_empty_off_reason_still_reads_as_off(page, bench_url):
    """WINDOW: the control against a catalog that says off and gives no
    reason.

    THE EMPTY STRING IS THE BLOCKER FUNCTION'S "NOT BLOCKED", so a
    reason that arrived empty would switch the control back ON for a
    bench that had just said it is off. This server never sends one, and
    a sentinel doing double duty is exactly the shape that gets one sent
    eventually: a proxy that strips long fields, a future server that
    reports the flag without the paragraph, a partial deploy.
    """
    page.route(
        "**/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "models": [],
                    "fetched": False,
                    "data_policy": "standard",
                    "snapshots_enabled": False,
                    "snapshots_off_reason": "",
                }
            ),
        ),
    )
    page.add_init_script("localStorage.setItem('bench-lineup', '[\"model/alpha\"]')")
    page.goto(bench_url)

    expect(page.get_by_test_id("snapshot-open")).to_be_disabled()
    expect(page.get_by_test_id("snapshot-msg")).to_contain_text("not configured")


def test_a_success_whose_body_cannot_be_read_says_so(snapshot_bench, snapshot_root):
    """WINDOW: the control after a 201 whose body is not JSON.

    SAID PLAINLY RATHER THAN WALKED INTO. The next thing the success
    path does is read body.digest, so a null body would have surfaced as
    "the snapshot request could not be sent", which is the one thing
    that demonstrably did work. The bytes are stored either way, which is
    what the message tells the person to do about it.
    """
    page = snapshot_bench(["model/alpha"])
    page.route(
        "**/snapshots",
        lambda route: route.fulfill(
            status=201, content_type="text/plain", body="not json at all"
        ),
    )
    page.get_by_test_id("snapshot-open").click()
    page.get_by_test_id("snapshot-root").fill(str(snapshot_root))
    page.get_by_test_id("snapshot-patterns").fill("pkg/*.py")
    page.get_by_test_id("snapshot-compose").click()

    message = page.get_by_test_id("snapshot-msg")
    expect(message).to_contain_text("could not read")
    assert "could not be sent" not in message.inner_text()
    expect(page.get_by_test_id("attachment-chip")).to_have_count(0)
