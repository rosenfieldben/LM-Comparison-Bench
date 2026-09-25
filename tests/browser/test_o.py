"""Phase O browser tests, O1: the member listing in the snapshot panel.

A List step before Compose shows what Compose would select, exclude and
refuse, in the server's own sentences; checking rows writes the patterns
that name exactly those files; and every row goes when the panel forgets
its root, since each is a path in somebody's repository.

The trees are under listing_root, a second allowlist entry of the
snapshot bench, each subdirectory a root of its own (see conftest).
Every proof names its window.
"""

import json

import pytest
from playwright.sync_api import expect
from test_n import REFUSED_RESOURCE, arm

from bench import snapshot

pytestmark = pytest.mark.browser

FORBIDDEN_RESOURCE = (
    "Failed to load resource: the server responded with a status of 403"
)


@pytest.fixture(autouse=True)
def collectors(page):
    """Console errors and CSP violations armed before first paint and
    asserted after, as the N suites do; yields the prefixes a test
    allows."""
    errors = arm(page)
    allowed = [REFUSED_RESOURCE]
    yield allowed
    assert page.evaluate("window.__csp || []") == []
    assert [e for e in errors if not e.startswith(tuple(allowed))] == []


def open_panel(page, root, patterns):
    page.get_by_test_id("snapshot-open").click()
    page.get_by_test_id("snapshot-root").fill(str(root))
    page.get_by_test_id("snapshot-patterns").fill(patterns)


def press_list(page):
    """List, and the listing the server answered."""
    with page.expect_response(lambda r: r.url.endswith("/snapshots/listing")) as answer:
        page.get_by_test_id("snapshot-list").click()
    expect(page.get_by_test_id("snapshot-listing")).to_be_visible()
    return answer.value.json()


def rows(page):
    return page.get_by_test_id("snapshot-member")


def box_for(page, path):
    """A row's checkbox by its accessible name: the path in quotes, so a
    leading space survives the name's whitespace collapsing."""
    return page.get_by_role(
        "checkbox", name=json.dumps(path, ensure_ascii=False), exact=True
    )


def patterns_box(page):
    return page.get_by_test_id("snapshot-patterns")


def test_the_table_is_the_listing_in_the_servers_words(
    snapshot_bench, snapshot_bench_url, listing_root
):
    """WINDOW: the member table and its line after List over a tree with
    two sources, a README, a .env, a node_modules and a file over the
    member bound, under '**/*'; read against the same listing asked for
    through the API.

    Every row is a member of the listing: path, bytes, status and the
    reason word for word, the refusal the server's own sentence. The
    secret is excluded with its group named and no size. The line says
    Compose would refuse, in the listing's refusal, and, the refused row
    being a file, does not say the patterns cannot fix it. PRE-STATE: the
    API listing refuses on big.txt and selects the two sources."""
    root = listing_root / "mixed"
    api = snapshot_bench(["stub/fast"]).request.post(
        snapshot_bench_url + "/snapshots/listing",
        data={"root": str(root), "patterns": ["**/*"]},
    )
    listing = api.json()
    assert listing["refusal"].startswith("big.txt is 250000 bytes")
    page = snapshot_bench(["stub/fast"])
    open_panel(page, root, "**/*")

    shown = press_list(page)

    assert shown == listing
    expect(rows(page)).to_have_count(len(listing["members"]))
    for index, member in enumerate(listing["members"]):
        row = rows(page).nth(index)
        expect(row.get_by_test_id("member-path")).to_have_text(member["path"])
        expect(row.get_by_test_id("member-bytes")).to_have_text(
            "" if member["bytes"] is None else str(member["bytes"])
        )
        expect(row.get_by_test_id("member-status")).to_have_text(member["status"])
        expect(row.get_by_test_id("member-reason")).to_have_text(member["reason"] or "")
    secret = rows(page).filter(has_text=".env")
    expect(secret.get_by_test_id("member-reason")).to_contain_text("(secrets)")
    expect(secret.get_by_test_id("member-bytes")).to_have_text("")
    summary = page.get_by_test_id("snapshot-listing-summary")
    expect(summary).to_have_text("Compose would refuse: " + listing["refusal"])


def test_a_selection_that_composes_says_what_it_would_read(
    snapshot_bench, listing_root
):
    """WINDOW: the member table's line after List over the same tree
    under '**/*.py'.

    A listing that would compose says how many files and bytes, the bound
    on the composed characters, and that the contents are checked only
    when it composes. PRE-STATE: the listing selects two files and
    refuses nothing."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "mixed", "**/*.py")
    listing = press_list(page)
    assert listing["would_compose"] is True
    bound = listing["composed_chars_at_most"]
    expect(page.get_by_test_id("snapshot-listing-summary")).to_have_text(
        f"Compose would read 2 files, 12 bytes, at most {bound} characters "
        f"composed, at or under the {snapshot.MAX_COMPOSED_CHARS} character "
        "ceiling. File contents are checked only when it composes: images, NUL "
        "bytes and UTF-8."
    )


def test_blank_inputs_and_too_many_patterns_are_not_sent(snapshot_bench, listing_root):
    """WINDOW: List and Compose pressed with no root, with no pattern, and
    with MAX_PATTERNS + 1 typed patterns, every request to either door
    counted.

    Blank is not sent, and a list longer than a request may carry is said
    with the constant's value and name rather than sent for the request
    model to refuse in words that name neither. PRE-STATE: a good List
    from the same panel is sent."""
    page = snapshot_bench(["stub/fast"])
    sent = []
    page.on(
        "request",
        lambda r: sent.append(r.url) if "/snapshots" in r.url else None,
    )
    open_panel(page, listing_root / "mixed", "**/*.py")
    press_list(page)
    assert len(sent) == 1
    message = page.get_by_test_id("snapshot-msg")

    page.get_by_test_id("snapshot-root").fill("")
    page.get_by_test_id("snapshot-list").click()
    expect(message).to_contain_text("Name the clone root to walk")
    page.get_by_test_id("snapshot-root").fill(str(listing_root / "mixed"))
    patterns_box(page).fill("")
    page.get_by_test_id("snapshot-list").click()
    expect(message).to_contain_text("Name at least one include pattern")
    # One count per button, so the second sentence cannot be the first
    # one left standing.
    for button, too_many in (
        ("snapshot-list", snapshot.MAX_PATTERNS + 1),
        ("snapshot-compose", snapshot.MAX_PATTERNS + 2),
    ):
        patterns_box(page).fill("\n".join(f"p{n}.py" for n in range(too_many)))
        page.get_by_test_id(button).click()
        expect(message).to_have_text(
            f"{too_many} include patterns, over the {snapshot.MAX_PATTERNS} "
            "pattern limit (MAX_PATTERNS). A selection that needs more is one a "
            "glob can say, such as 'src/**/*.py'."
        )
    page.wait_for_timeout(200)
    assert len(sent) == 1


def test_checked_rows_write_the_patterns_compose_sends(snapshot_bench, listing_root):
    """WINDOW: the patterns box as rows are checked and unchecked, then the
    body Compose sends and the chip it stages.

    Checked rows write one pattern per file in the table's order; none
    checked puts back the patterns the listing was made with; Compose
    sends the box as it stands. PRE-STATE: the box holds the listed
    pattern before any check."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "mixed", "**/*.py")
    press_list(page)
    expect(patterns_box(page)).to_have_value("**/*.py")

    box_for(page, "src/b.py").check()
    box_for(page, "src/a.py").check()
    expect(patterns_box(page)).to_have_value("src/a.py\nsrc/b.py")
    expect(page.get_by_test_id("snapshot-listing-summary")).to_have_text(
        "2 files checked, 12 bytes: the patterns now name exactly these. List "
        "again to see what Compose makes of them."
    )
    box_for(page, "src/a.py").uncheck()
    box_for(page, "src/b.py").uncheck()
    expect(patterns_box(page)).to_have_value("**/*.py")
    box_for(page, "src/a.py").check()
    with page.expect_request(lambda r: r.url.endswith("/snapshots")) as sent:
        page.get_by_test_id("snapshot-compose").click()
    assert json.loads(sent.value.post_data)["patterns"] == ["src/a.py"]
    expect(page.get_by_test_id("attachment-chip")).to_have_count(1)
    assert "1 file" in page.get_by_test_id("attachment-meta").inner_text()


@pytest.mark.parametrize(
    ("odd", "sibling", "pattern"),
    [(" a.py", "a.py", "[ ]a.py"), ("[x].py", "x.py", "[[]x].py")],
    ids=["leading space", "bracket"],
)
def test_a_checked_odd_name_composes_that_file_and_not_its_sibling(
    snapshot_bench, listing_root, odd, sibling, pattern
):
    """WINDOW: the pattern a check writes for a name the matcher or the
    panel would misread, and the manifest Compose then records.

    A leading space is trimmed by the panel, and a bracket is read by
    fnmatch as a class, and either way the path used as its own pattern
    selects the SIBLING. The pattern the check writes selects exactly the
    file checked. PRE-STATE: the sibling is in the same tree and lists as
    selected."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "odd", "*.py")
    listing = press_list(page)
    assert sibling in [
        m["path"] for m in listing["members"] if m["status"] == "selected"
    ]

    box_for(page, odd).check()
    expect(patterns_box(page)).to_have_value(pattern)
    with page.expect_response(lambda r: r.url.endswith("/snapshots")) as answer:
        page.get_by_test_id("snapshot-compose").click()

    assert [f["path"] for f in answer.value.json()["manifest"]["files"]] == [odd]


def test_past_max_patterns_the_check_is_undone_and_named(snapshot_bench, listing_root):
    """WINDOW: MAX_PATTERNS rows checked, then one more, over a tree of
    MAX_PATTERNS + 1 files.

    The check past the limit is undone, so the checked rows and the box
    always agree, and the panel says so with the constant's value and
    name. PRE-STATE: at the limit the box holds MAX_PATTERNS lines."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "many", "*.py")
    press_list(page)
    for n in range(snapshot.MAX_PATTERNS):
        box_for(page, f"f{n:02d}.py").check()
    assert len(patterns_box(page).input_value().split("\n")) == snapshot.MAX_PATTERNS

    last = box_for(page, f"f{snapshot.MAX_PATTERNS:02d}.py")
    last.click()

    expect(last).not_to_be_checked()
    assert len(patterns_box(page).input_value().split("\n")) == snapshot.MAX_PATTERNS
    expect(page.get_by_test_id("snapshot-msg")).to_have_text(
        f"{snapshot.MAX_PATTERNS + 1} files checked, over the "
        f"{snapshot.MAX_PATTERNS} pattern limit (MAX_PATTERNS): one pattern per "
        "checked file cannot name that many. The last check was undone; write a "
        "glob that covers them instead."
    )


def test_a_hand_edit_is_never_overwritten_by_a_check(snapshot_bench, listing_root):
    """WINDOW: a row checked after the patterns box was edited by hand.

    The listing describes the patterns it was made with. A box edited
    since is the person's, so the check is undone and the panel says to
    list again. PRE-STATE: before the edit a check writes the box."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "mixed", "**/*.py")
    press_list(page)
    box_for(page, "src/a.py").check()
    expect(patterns_box(page)).to_have_value("src/a.py")
    patterns_box(page).fill("src/a.py\ndocs/*.md")

    box_for(page, "src/b.py").click()

    expect(box_for(page, "src/b.py")).not_to_be_checked()
    expect(patterns_box(page)).to_have_value("src/a.py\ndocs/*.md")
    expect(page.get_by_test_id("snapshot-msg")).to_contain_text(
        "The patterns were edited after this listing"
    )


def test_a_row_is_checked_from_the_keyboard(snapshot_bench, listing_root):
    """WINDOW: a row's checkbox found by its accessible name and toggled
    with the space bar.

    Each checkbox is named by its path, so a screen reader says which
    file it is and not "checkbox, not checked". PRE-STATE: the box holds
    the listed pattern."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "mixed", "**/*.py")
    press_list(page)
    expect(patterns_box(page)).to_have_value("**/*.py")

    box_for(page, "src/b.py").focus()
    page.keyboard.press("Space")

    expect(box_for(page, "src/b.py")).to_be_checked()
    expect(patterns_box(page)).to_have_value("src/b.py")


def test_a_refusal_no_pattern_can_fix_says_so(snapshot_bench, listing_root):
    """WINDOW: the line after List over a tree holding a named pipe, under
    '*.py', which selects only a.py.

    A pipe refuses the snapshot whatever the patterns select, so the line
    says that no choice of patterns changes it, the pipe is a refused
    row, and no row can be checked, since no narrowing makes this tree
    compose. PRE-STATE: the listing selects a.py."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "refusing", "*.py")
    listing = press_list(page)
    assert [m["path"] for m in listing["members"] if m["status"] == "selected"] == [
        "a.py"
    ]
    expect(page.get_by_test_id("snapshot-listing-summary")).to_have_text(
        "Compose would refuse: "
        + listing["refusal"]
        + " No choice of patterns changes this: it is the tree or the root that "
        "has to change."
    )
    expect(
        rows(page).filter(has_text="pipe").get_by_test_id("member-status")
    ).to_have_text("refused")
    expect(page.get_by_test_id("snapshot-member-use")).to_have_count(0)


def test_a_walk_that_stopped_offers_nothing_to_check(snapshot_bench, listing_root):
    """WINDOW: the member table after List over a tree holding a directory
    the bench cannot open, under '**/*.py'.

    The walk stops at that directory, as the composer's does: the listing
    is incomplete, the stop is the last row in the composer's words, and
    no row can be checked, since what stopped it does not depend on the
    patterns. PRE-STATE: the listing selects a.py, reached before the
    stop."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "stopped", "**/*.py")
    listing = press_list(page)
    assert listing["complete"] is False
    assert ("a.py", "selected") in [
        (m["path"], m["status"]) for m in listing["members"]
    ]
    last = rows(page).last
    expect(last.get_by_test_id("member-path")).to_have_text("locked")
    expect(last.get_by_test_id("member-reason")).to_have_text(listing["refusal"])
    expect(page.get_by_test_id("snapshot-member-use")).to_have_count(0)


def test_a_name_no_pattern_spells_gets_no_checkbox(snapshot_bench, listing_root):
    """WINDOW: the row for a file whose name holds a backslash.

    The server refuses a backslash in any pattern, so no checked row could
    name this file; its row says so instead of offering a check.
    PRE-STATE: the file is selected by '*.py'."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "odd", "*.py")
    listing = press_list(page)
    assert "back\\slash.py" in [
        m["path"] for m in listing["members"] if m["status"] == "selected"
    ]
    row = rows(page).filter(has_text="back\\slash.py")
    expect(row.get_by_test_id("member-use")).to_have_text("no exact pattern")
    expect(row.get_by_test_id("snapshot-member-use")).to_have_count(0)


def test_another_root_empties_the_table(snapshot_bench, listing_root):
    """WINDOW: the member table after the root box is edited.

    A listing describes one root, and checkboxes left under another would
    write one tree's paths as patterns for a different tree. PRE-STATE:
    the table has rows."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "mixed", "**/*.py")
    press_list(page)
    expect(rows(page)).to_have_count(3)

    page.get_by_test_id("snapshot-root").fill(str(listing_root / "odd"))

    expect(rows(page)).to_have_count(0)
    expect(page.get_by_test_id("snapshot-listing")).to_be_hidden()


@pytest.mark.parametrize("how", ["compose", "reuse", "clear"])
def test_every_row_goes_when_the_panel_forgets_its_root(
    snapshot_bench, listing_root, how
):
    """WINDOW: the member table, its line and the panel's message after a
    listing, then a successful Compose, a reuse replacing the staged set,
    or a clear.

    Each row is a path in somebody's repository, and each of these is a
    moment the panel forgets its root; the rows go with it, asserted on
    the text content, which a hidden element still holds. PRE-STATE: the
    table has rows and a line."""
    page = snapshot_bench(["stub/fast"])
    open_panel(page, listing_root / "mixed", "**/*.py")
    press_list(page)
    expect(rows(page)).to_have_count(3)

    if how == "compose":
        page.get_by_test_id("snapshot-compose").click()
        expect(page.get_by_test_id("attachment-chip")).to_have_count(1)
    elif how == "reuse":
        page.evaluate("window.BenchAttach.setFrom([], 'inline')")
    else:
        page.evaluate("window.BenchAttach.clear()")

    expect(rows(page)).to_have_count(0)
    text = page.locator("#attach-row").evaluate("el => el.textContent")
    assert "src/a.py" not in text
    assert page.get_by_test_id("snapshot-listing-summary").text_content() == ""


def test_the_blind_view_holds_no_listing_and_no_listed_refusal(
    snapshot_bench, snapshot_bench_url, listing_root
):
    """WINDOW: the composer after a blind session opens over a panel that
    shows a listing whose refused row, and whose line, name a link's
    absolute target, and a message line holding Compose's refusal naming
    it too.

    A refusal can carry an absolute path, and the message line sits
    outside the panel; the blind view's rule is that it shows none. All
    of it goes when the blind view opens, asserted on the text content,
    which a hidden element still holds. PRE-STATE: the rows, the line and
    the message each name the target before the blind view opens."""
    page = snapshot_bench(["stub/fast", "stub/slow"])
    api = page.request
    group = api.post(
        snapshot_bench_url + "/groups",
        data={
            "prompt": "blind over a listing",
            "models": ["stub/fast", "stub/slow"],
            "budget": "standard",
        },
    ).json()["id"]
    assert api.post(
        snapshot_bench_url + "/compare",
        data={
            "prompt": "blind over a listing",
            "models": ["stub/fast", "stub/slow"],
            "group_id": group,
        },
    ).ok
    target = str((listing_root / "linked" / "out").resolve())
    open_panel(page, listing_root / "linked", "*.py")
    press_list(page)
    page.get_by_test_id("snapshot-compose").click()
    message = page.get_by_test_id("snapshot-msg")
    expect(message).to_contain_text(target)
    expect(page.get_by_test_id("snapshot-listing-summary")).to_contain_text(target)
    expect(rows(page).filter(has_text="out")).to_have_count(1)

    page.evaluate("id => window.BenchRating.startBlind(id)", group)
    expect(page.get_by_test_id("rating-panel")).to_be_visible()

    text = page.locator("#attach-row").evaluate("el => el.textContent")
    assert target not in text
    expect(rows(page)).to_have_count(0)


def test_forgetting_the_panel_says_the_standing_reason_again(bench, bench_url):
    """WINDOW: the snapshot control's line on a bench with no allowlist,
    after the blind view opens (which forgets the panel).

    Forgetting clears the panel's answer, and then the standing reason
    is shown again: a disabled + Snapshot with no sentence beside it is a
    reason no keyboard user reads. PRE-STATE: the line holds the server's
    off sentence before the blind view opens."""
    from bench.main import SNAPSHOTS_OFF

    page = bench(["stub/fast", "stub/slow"])
    api = page.request
    group = api.post(
        bench_url + "/groups",
        data={
            "prompt": "a standing reason",
            "models": ["stub/fast", "stub/slow"],
            "budget": "standard",
        },
    ).json()["id"]
    assert api.post(
        bench_url + "/compare",
        data={
            "prompt": "a standing reason",
            "models": ["stub/fast", "stub/slow"],
            "group_id": group,
        },
    ).ok
    message = page.get_by_test_id("snapshot-msg")
    expect(message).to_have_text(SNAPSHOTS_OFF)

    page.evaluate("id => window.BenchRating.startBlind(id)", group)
    expect(page.get_by_test_id("rating-panel")).to_be_visible()

    expect(message).to_have_text(SNAPSHOTS_OFF)
    expect(page.get_by_test_id("snapshot-open")).to_be_disabled()


@pytest.mark.parametrize("how", ["blind view", "another root"])
def test_a_listing_answered_after_the_panel_forgot_is_dropped(
    snapshot_bench, snapshot_bench_url, listing_root, how
):
    """WINDOW: a listing request held while the panel forgets it, by the
    blind view opening or by another root being typed, then released.

    The answer belongs to a root the panel has since forgotten, so it
    draws nothing: no row, no line. The blind view moves the view epoch;
    typing another root moves nothing but the listing's own token, which
    is the case only the token catches. Compose waits while a listing is
    out, so the two never land on each other. PRE-STATE: while the
    listing is held Compose is disabled."""
    page = snapshot_bench(["stub/fast", "stub/slow"])
    api = page.request
    group = api.post(
        snapshot_bench_url + "/groups",
        data={
            "prompt": "a late listing",
            "models": ["stub/fast", "stub/slow"],
            "budget": "standard",
        },
    ).json()["id"]
    assert api.post(
        snapshot_bench_url + "/compare",
        data={
            "prompt": "a late listing",
            "models": ["stub/fast", "stub/slow"],
            "group_id": group,
        },
    ).ok
    held = []
    page.route("**/snapshots/listing", lambda route: held.append(route))
    open_panel(page, listing_root / "mixed", "**/*.py")
    page.get_by_test_id("snapshot-list").click()
    for _ in range(200):
        if held:
            break
        page.wait_for_timeout(25)
    assert held
    expect(page.get_by_test_id("snapshot-compose")).to_be_disabled()

    if how == "blind view":
        page.evaluate("id => window.BenchRating.startBlind(id)", group)
    else:
        page.get_by_test_id("snapshot-root").fill(str(listing_root / "odd"))
    with page.expect_response(lambda r: r.url.endswith("/snapshots/listing")) as late:
        held.pop().continue_()
    assert late.value.ok
    page.wait_for_timeout(100)

    expect(rows(page)).to_have_count(0)
    assert page.get_by_test_id("snapshot-listing-summary").text_content() == ""
