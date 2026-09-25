"""Phase N2 browser tests: composing a dataset and storing it.

The builder composes and the server validates. These tests drive the
panel a person uses (rows, pasted or uploaded JSONL, saved prompts,
stored documents) against the real datasets door, and assert what was
STORED, read back through GET /datasets/{digest}, rather than what the
page believed it sent: a mock cannot refuse, and a page that only agreed
with itself would pass every test here while storing something else.

EVERY PROOF NAMES ITS WINDOW. The panel has two ways in that send
different text, and a proof about one reads as coverage of the other
while leaving it unguarded.

EVERY TEST RUNS WITH THE CSP AND ERROR COLLECTORS ARMED (the autouse
fixture below), because every path here builds DOM, and a picker or a
refusal that broke the policy would otherwise pass. The one console
error every test allows is Chromium's own line for a 422 the server
answered, which the refusal proofs cause on purpose; a test that breaks
a fetch on purpose names the errors that causes, and no others pass.

The session's database is shared by every browser test, so names are
unique per test and nothing asserts a total it did not make.
"""

import base64
import hashlib
import json
import re
import uuid

import pytest
from playwright.sync_api import expect
from test_contrast import composite, contrast

from bench import main
from bench.datasets import MAX_PROMPT_CHARS

pytestmark = pytest.mark.browser

STORE_TIMEOUT = 10_000

# Characters the proofs turn on, named rather than escaped so each says
# why it is here.
FEFF = chr(0xFEFF)  # blank to JavaScript's trim alone
NEL = chr(0x85)  # blank to Python's strip alone
LS = chr(0x2028)  # raw in JSON.stringify; not a line end to the parser
GRIN = chr(0x1F600)  # one code point, two UTF-16 units, four UTF-8 bytes

# Chromium's console line for a response the server refused, which every
# refusal proof causes on purpose.
REFUSED_RESOURCE = "Failed to load resource: the server responded with a status of 422"
# What the panel adds to a refusal once the text on screen is not the
# text it was said about.
CHANGED = " (said of the text as Store sent it, which has changed since)"
# Chromium's console line for a request the test aborted on purpose.
ABORTED_RESOURCE = "Failed to load resource: net::ERR_FAILED"


def arm(page):
    """Collect CSP violations, console errors and page errors from before
    first paint, as test_csp.py does for the critical path. Returns the
    list the page's own errors land in; window.__csp holds the rest."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    page.add_init_script(
        "window.__csp = [];"
        "document.addEventListener('securitypolicyviolation', (e) => {"
        "  window.__csp.push(e.effectiveDirective + ' blocked ' + e.blockedURI);"
        "});"
    )
    return errors


@pytest.fixture(autouse=True)
def collectors(page):
    """Arm the collectors before the test and assert them after. Yields
    the list of console-error prefixes the test allows: Chromium's own
    line for a 422 the server answered, which every refusal proof causes
    on purpose, and whatever a test that breaks a fetch deliberately adds
    for itself."""
    errors = arm(page)
    allowed = [REFUSED_RESOURCE]
    yield allowed
    assert page.evaluate("window.__csp || []") == []
    assert [e for e in errors if not e.startswith(tuple(allowed))] == []


def open_datasets(page):
    page.get_by_test_id("datasets-toggle").click()
    expect(page.get_by_test_id("dataset-library")).to_have_attribute(
        "data-state", re.compile("^(ready|empty)$")
    )


def unique(label):
    """A name no other test in the shared session database has used."""
    return label + "-" + uuid.uuid4().hex[:8]


def row(page, index):
    return page.get_by_test_id("dataset-row").nth(index)


def add_task(page, task_id, prompt, scorer="", **fields):
    """Add one row through the UI and fill it the way a person would."""
    page.get_by_test_id("dataset-add-row").click()
    last = page.get_by_test_id("dataset-row").last
    if task_id:
        last.get_by_test_id("dataset-row-id").fill(task_id)
    if prompt:
        last.get_by_test_id("dataset-row-prompt").fill(prompt)
    if scorer:
        last.get_by_test_id("dataset-row-scorer").select_option(scorer)
    for key, value in fields.items():
        last.get_by_test_id("dataset-row-" + key).fill(value)
    return last


def store(page):
    page.get_by_test_id("dataset-store").click()
    expect(page.get_by_test_id("dataset-msg")).not_to_have_text(
        "storing", timeout=STORE_TIMEOUT
    )
    return page.get_by_test_id("dataset-msg").inner_text()


def stored(page, bench_url, digest):
    resp = page.request.get(bench_url + "/datasets/" + digest)
    assert resp.ok, resp.text()
    return resp.json()


def entry_for(page, name):
    return page.get_by_test_id("dataset-entry").filter(has_text=name)


def digest_of(page, name):
    entry = entry_for(page, name)
    expect(entry).to_have_count(1)
    return entry.get_attribute("data-digest")


def upload_document(page, url, filename, text):
    resp = page.request.post(
        url + "/attachments",
        data={
            "filename": filename,
            "content_base64": base64.b64encode(text.encode()).decode("ascii"),
        },
    )
    assert resp.ok, resp.text()
    return resp.json()["digest"]


def hold(page, glob, method):
    """Hold every request to glob with this method until the test lets
    it go; others pass. Returns the list the held routes land in."""
    held = []

    def handler(route):
        if route.request.method == method:
            held.append(route)
        else:
            route.continue_()

    page.route(glob, handler)
    return held


def wait_held(page, held, count=1):
    for _ in range(400):
        if len(held) >= count:
            return
        page.wait_for_timeout(25)
    raise AssertionError(f"{len(held)} requests held, expected {count}")


def settle(page, route):
    """Let a held request go, wait for its answer to arrive, and give the
    page a moment to act on it, so a proof that a late answer drew or
    threw nothing is made after the answer was handled, not before."""
    with page.expect_response(lambda r: r.url == route.request.url) as info:
        route.continue_()
    info.value.finished()
    page.evaluate("() => new Promise((done) => setTimeout(done, 50))")


def focused(page):
    """What has focus: its testid, the row it sits in and its digest."""
    return page.evaluate(
        "() => { const el = document.activeElement;"
        " const row = el.closest('[data-testid=dataset-row]');"
        " return [el.dataset.testid || el.tagName,"
        "  row ? Number(row.dataset.index) : null, el.dataset.digest || null]; }"
    )


def exactly(total):
    """A valid dataset of multi-byte prompts, exactly total UTF-8 bytes
    long, every prompt inside MAX_PROMPT_CHARS."""
    lines, size, i = [], 0, 0
    while True:
        line = json.dumps({"id": f"t{i}", "prompt": GRIN * 20000}, ensure_ascii=False)
        line += "\n"
        if size + len(line.encode()) > total - 1000:
            break
        lines.append(line)
        size += len(line.encode())
        i += 1
    head = json.dumps({"id": f"t{i}", "prompt": ""}, ensure_ascii=False)
    pad = total - size - len(head.encode()) - 1
    lines.append(json.dumps({"id": f"t{i}", "prompt": "x" * pad}) + "\n")
    text = "".join(lines)
    assert len(text.encode()) == total
    return text


# ---- The rows way in.


def test_three_rows_one_per_scorer_family_are_stored_as_the_page_composed(
    page, bench, bench_url
):
    """WINDOW: the rows way in, from three typed rows to the bytes GET
    /datasets/{digest} serves back.

    Exact, regex and judge, each with the field its scorer needs, and
    the regex row first given a reference as an exact row and then
    switched: the reference is still in the row and must not be sent.
    The stored lines are compared as parsed JSON, key for key, so a field
    sent for a scorer that does not use it, or a blank sent as a value,
    fails here. The ceilings are read before Store and checked against
    the served bytes, with a prompt whose code points, UTF-16 units and
    UTF-8 bytes are three different numbers."""
    bench(["stub/fast"])
    open_datasets(page)
    name = unique("three kinds")
    page.get_by_test_id("dataset-name").fill(name)
    hello = "say Hello " + GRIN + "é"
    add_task(page, "e1", hello, "exact", reference="Hello")
    regex = add_task(page, "r1", "say Hi", "exact", reference="stray")
    regex.get_by_test_id("dataset-row-scorer").select_option("regex")
    regex.get_by_test_id("dataset-row-pattern").fill("H")
    judge = add_task(page, "j1", "be kind", "judge", rubric="kindness")
    judge.get_by_test_id("dataset-row-threshold").fill("0.5")
    expect(page.get_by_test_id("dataset-store")).to_be_enabled()
    ceilings = page.get_by_test_id("dataset-ceilings").inner_text()

    said = store(page)

    assert said.startswith("stored as " + name), said
    digest = digest_of(page, name)
    served = stored(page, bench_url, digest)
    assert hashlib.sha256(served["content"].encode()).hexdigest() == digest
    lines = [json.loads(line) for line in served["content"].splitlines()]
    assert lines == [
        {
            "id": "e1",
            "prompt": hello,
            "reference": "Hello",
            "scorer": {"kind": "exact"},
        },
        {"id": "r1", "prompt": "say Hi", "scorer": {"kind": "regex", "pattern": "H"}},
        {
            "id": "j1",
            "prompt": "be kind",
            "rubric": "kindness",
            "scorer": {"kind": "judge", "pass_threshold": 0.5},
        },
    ]
    # The numerators too: code points for the prompt, as len() counts
    # them, and the UTF-8 bytes of exactly what was stored.
    assert ceilings == (
        "3 of 2000 tasks · longest prompt "
        + str(len(hello))
        + " of 100000 characters · "
        + str(len(served["content"].encode()))
        + " of 1912831 bytes"
    )
    expect(entry_for(page, name).get_by_test_id("dataset-entry-meta")).to_have_text(
        "3 tasks · exact, judge, regex · sha256 " + digest[:7]
    )


def test_each_scorer_shows_its_own_fields_and_no_other(page, bench):
    """WINDOW: one row's scorer select, stepped through every kind, and
    the fields the row shows for each.

    THE SCORER'S FIELDS AND ONLY THOSE: a box shown for a scorer that
    does not use it would take typing that is silently not sent. Each
    field carries a visible caption, because its placeholder would be
    ghost text below the contrast floor."""
    bench(["stub/fast"])
    open_datasets(page)
    task = add_task(page, "f1", "p")
    shown = {
        "": [],
        "exact": ["reference"],
        "normalized_exact": ["reference"],
        "contains": ["reference"],
        "regex": ["pattern"],
        "judge": ["rubric", "threshold"],
    }
    for kind, fields in shown.items():
        task.get_by_test_id("dataset-row-scorer").select_option(kind)
        for key in ("reference", "pattern", "rubric", "threshold"):
            expect(task.get_by_test_id("dataset-row-" + key)).to_have_count(
                1 if key in fields else 0
            )
        for key in fields:
            caption = task.locator(".ds-cap-" + key + " .ds-cap-text")
            expect(caption).to_be_visible()
    for key in ("id", "prompt", "system"):
        expect(task.locator(".ds-cap-" + key + " .ds-cap-text")).to_be_visible()


def test_a_new_task_is_blank_and_a_store_selects_nothing(page, bench, bench_url):
    """WINDOW: + Task's new row before anything is typed, and the library
    after a Store succeeds.

    NOTHING IS PRE-FILLED. The id box of a new row is empty and it has no
    scorer, asserted before any fill; and the dataset just stored is not
    selected for anything, because selecting is the person's decision."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-add-row").click()
    fresh = row(page, 0)
    expect(fresh.get_by_test_id("dataset-row-id")).to_have_value("")
    expect(fresh.get_by_test_id("dataset-row-prompt")).to_have_value("")
    expect(fresh.get_by_test_id("dataset-row-scorer")).to_have_value("")
    name = unique("unselected")
    page.get_by_test_id("dataset-name").fill(name)
    expect(page.get_by_test_id("dataset-nudge")).to_have_text(
        "Store waits: line 1 needs an id"
    )
    fresh.get_by_test_id("dataset-row-id").fill("n1")
    fresh.get_by_test_id("dataset-row-prompt").fill("fresh " + uuid.uuid4().hex)

    assert store(page).startswith("stored as " + name)

    expect(entry_for(page, name)).to_have_attribute("aria-pressed", "false")
    expect(page.get_by_test_id("dataset-selected")).to_have_text("")
    assert page.evaluate("window.BenchDatasets.selected()") is None


def test_the_nudge_greys_store_and_names_the_line_it_waits_on(page, bench):
    """WINDOW: the Store button and the nudge line, as a name and a
    prompt are typed.

    Store is greyed for a nameless dataset and for a row with no prompt,
    and the line says which, by line number. Typing the missing value
    enables Store. Every reason here is one the server would refuse; the
    subset proof below and the unit grid prove that."""
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "t1", "has a prompt")
    add_task(page, "t2", "")
    store_button = page.get_by_test_id("dataset-store")
    nudge = page.get_by_test_id("dataset-nudge")

    expect(store_button).to_be_disabled()
    expect(nudge).to_have_text("Store waits: name the dataset")
    page.get_by_test_id("dataset-name").fill(unique("nudged"))
    expect(nudge).to_have_text("Store waits: line 2 needs a prompt")
    expect(row(page, 1).get_by_test_id("dataset-row-msg")).to_have_text(
        "needs a prompt"
    )
    expect(store_button).to_be_disabled()

    row(page, 1).get_by_test_id("dataset-row-prompt").fill("now it does")

    expect(nudge).to_have_text("")
    expect(store_button).to_be_enabled()


SUBSET_CASES = [
    # Nudged, so Store is greyed; what the page composes is refused.
    ({"prompt": ""}, "nudged"),
    ({"prompt": "p", "scorer": "contains", "reference": " "}, "nudged"),
    ({"prompt": "p", "scorer": "regex", "pattern": ""}, "nudged"),
    ({"prompt": "p", "scorer": "judge", "rubric": NEL}, "nudged"),
    ({"prompt": "p" * (MAX_PROMPT_CHARS + 1)}, "nudged"),
    # Not nudged and stored: blank to trim() alone, and the ceiling in a
    # character .length counts twice.
    ({"prompt": "p", "scorer": "judge", "rubric": FEFF}, "stored"),
    ({"prompt": GRIN * MAX_PROMPT_CHARS}, "stored"),
    # Not nudged and refused, in the server's words beside the row.
    ({"prompt": "p", "scorer": "regex", "pattern": "("}, "refused"),
    ({"prompt": "p", "scorer": "judge", "rubric": "g", "threshold": FEFF}, "refused"),
    ({"prompt": "p", "scorer": "judge", "rubric": "g", "threshold": "0x1"}, "refused"),
    ({"prompt": "p", "scorer": "judge", "rubric": "g", "threshold": "2"}, "refused"),
]


def test_review_repro_the_nudge_subset_holds_in_the_page(page, bench, bench_url):
    """WINDOW: the page's own Store button and nudge over one typed row
    per case, and POST /datasets over what the page composes for each.

    THE COMMISSION'S NUDGE-SUBSET PROOF, IN THE BROWSER; the unit grid
    covers every row the page can compose, and this holds the page's own
    nudge() (the name, the line prefix, the mode) to the same property.
    A nudged row greys Store, and what the page's composeJsonl writes for
    it is refused by the door. A row the page does not nudge is Stored
    through the button: some are stored, and some are refused in the
    server's words beside the row, which is the server's to say."""
    bench(["stub/fast"])
    open_datasets(page)
    outcomes = {"nudged": 0, "stored": 0, "refused": 0}
    for index, (fields, expected) in enumerate(SUBSET_CASES):
        if page.get_by_test_id("dataset-row").count():
            row(page, 0).get_by_test_id("dataset-row-remove").click()
        name = unique("subset")
        page.get_by_test_id("dataset-name").fill(name)
        typed = dict(fields)
        scorer = typed.pop("scorer", "")
        task = add_task(page, f"s{index}-{uuid.uuid4().hex[:6]}", "", scorer)
        for key, value in typed.items():
            task.get_by_test_id("dataset-row-" + key).fill(value)
        button = page.get_by_test_id("dataset-store")
        if expected == "nudged":
            expect(button).to_be_disabled()
            composed = page.evaluate(
                "(r) => window.BenchLib.composeJsonl([r])",
                {
                    "id": task.get_by_test_id("dataset-row-id").input_value(),
                    "prompt": fields["prompt"],
                    "system": "",
                    "scorer": scorer,
                    "reference": fields.get("reference", ""),
                    "pattern": fields.get("pattern", ""),
                    "rubric": fields.get("rubric", ""),
                    "threshold": fields.get("threshold", ""),
                    "documents": [],
                },
            )
            resp = page.request.post(
                bench_url + "/datasets", data={"name": name, "content": composed}
            )
            assert resp.status == 422, (fields, resp.text())
        else:
            expect(button).to_be_enabled()
            said = store(page)
            if expected == "stored":
                assert said.startswith("stored as " + name), (fields, said)
            else:
                assert said.startswith("not stored: line 1: "), (fields, said)
                expect(row(page, 0).get_by_test_id("dataset-row-msg")).to_have_text(
                    said[len("not stored: ") :]
                )
        outcomes[expected] += 1
    assert all(count > 0 for count in outcomes.values())


def test_review_repro_a_refusal_naming_a_line_lands_beside_that_row(
    page, bench, bench_url
):
    """WINDOW: Store over four rows whose first and third share an id,
    and the edits, the mode round trip and the removal that follow.

    THE COMMISSION'S SHAPE. parse_dataset names line 3; row 3 is line 3,
    so the server's sentence appears verbatim under row 3 and nowhere
    else, row and message both read as refused, and nothing is stored.
    The mark stands only while the rows would send the text it was said
    about: an edit to ANY row that reaches that text takes it away
    (renaming row 1 is the natural fix, and a mark left on row 3 would be
    untrue) and the message says the text has changed; undoing the edit,
    or going to JSONL and back, brings both back. Removing row 1
    renumbers the lines, and the row now on line 3 (a unique id) must
    not inherit the sentence."""
    bench(["stub/fast"])
    open_datasets(page)
    before = len(page.request.get(bench_url + "/datasets").json()["datasets"])
    page.get_by_test_id("dataset-name").fill(unique("dup"))
    add_task(page, "dup", "one")
    add_task(page, "other", "two")
    add_task(page, "dup", "three")
    add_task(page, "four", "four")

    said = store(page)

    sentence = "line 3: duplicate task id 'dup'"
    msg = page.get_by_test_id("dataset-msg")
    assert said == "not stored: " + sentence
    expect(msg).to_have_attribute("data-state", "refused")
    third = row(page, 2)
    expect(third.get_by_test_id("dataset-row-msg")).to_have_text(sentence)
    expect(third.get_by_test_id("dataset-row-msg")).to_have_attribute(
        "data-state", "refused"
    )
    expect(third).to_have_attribute("data-state", "refused")
    for index in (0, 1, 3):
        expect(row(page, index).get_by_test_id("dataset-row-msg")).to_have_text("")
    assert len(page.request.get(bench_url + "/datasets").json()["datasets"]) == before

    row(page, 0).get_by_test_id("dataset-row-id").fill("first")
    expect(third.get_by_test_id("dataset-row-msg")).to_have_text("")
    expect(third).not_to_have_attribute("data-state", "refused")
    expect(msg).to_have_text("not stored: " + sentence + CHANGED)

    row(page, 0).get_by_test_id("dataset-row-id").fill("dup")
    expect(third.get_by_test_id("dataset-row-msg")).to_have_text(sentence)
    expect(msg).to_have_text("not stored: " + sentence)

    page.get_by_test_id("dataset-mode-jsonl").click()
    page.get_by_test_id("dataset-mode-rows").click()
    expect(third).to_have_attribute("data-state", "refused")

    row(page, 0).get_by_test_id("dataset-row-remove").click()
    expect(page.get_by_test_id("dataset-row")).to_have_count(3)
    expect(row(page, 2).get_by_test_id("dataset-row-id")).to_have_value("four")
    for index in (0, 1, 2):
        expect(row(page, index)).not_to_have_attribute("data-state", "refused")
        expect(row(page, index).get_by_test_id("dataset-row-msg")).to_have_text("")


def test_a_refusal_that_returns_after_the_rows_changed_marks_no_row(page, bench):
    """WINDOW: Store with the POST held, the rows or the mode changed
    while it is held, and the refusal that arrives after.

    CONTROL FIRST: held and released with nothing changed, the refusal
    lands on row 3, so the hold itself moves nothing; Store is greyed
    while the request is out. Held again with the mode switched to JSONL
    and back: the text is the text sent, so row 3 is marked and the
    message has no note. Held again and released while in JSONL mode:
    the refusal is about the ROWS as sent, so JSONL shows the note, and
    going back to rows marks row 3 again (a refusal filed under the mode
    of its answer rather than of its request would not). Held again with
    row 1 removed: the sentence names line 3 of the text as sent, line 3
    on screen is another task, so no row is marked and the message says
    the text has changed."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill(unique("race"))
    add_task(page, "dup", "one")
    add_task(page, "other", "two")
    add_task(page, "dup", "three")
    add_task(page, "four", "four")
    held = hold(page, "**/datasets", "POST")
    sentence = "line 3: duplicate task id 'dup'"
    button = page.get_by_test_id("dataset-store")
    msg = page.get_by_test_id("dataset-msg")

    button.click()
    wait_held(page, held)
    expect(button).to_be_disabled()
    held.pop().continue_()
    expect(row(page, 2).get_by_test_id("dataset-row-msg")).to_have_text(sentence)

    button.click()
    wait_held(page, held)
    page.get_by_test_id("dataset-mode-jsonl").click()
    page.get_by_test_id("dataset-mode-rows").click()
    held.pop().continue_()
    expect(msg).to_have_text("not stored: " + sentence)
    expect(row(page, 2)).to_have_attribute("data-state", "refused")

    button.click()
    wait_held(page, held)
    page.get_by_test_id("dataset-mode-jsonl").click()
    held.pop().continue_()
    expect(msg).to_have_text("not stored: " + sentence + CHANGED)
    page.get_by_test_id("dataset-mode-rows").click()
    expect(msg).to_have_text("not stored: " + sentence)
    expect(row(page, 2)).to_have_attribute("data-state", "refused")

    button.click()
    wait_held(page, held)
    row(page, 0).get_by_test_id("dataset-row-remove").click()
    held.pop().continue_()

    expect(msg).to_have_text("not stored: " + sentence + CHANGED)
    expect(page.get_by_test_id("dataset-row")).to_have_count(3)
    for index in range(3):
        expect(row(page, index)).not_to_have_attribute("data-state", "refused")


def test_a_refusal_of_pasted_text_marks_no_row_after_a_switch(page, bench):
    """WINDOW: a JSONL refusal naming line 2, the builder rows while it
    stands, and the switch back to rows.

    The sentence is about a line of the pasted text, which has no rows:
    the builder's row 2 is not marked, in JSONL mode (where it is hidden)
    or after the switch. Each switch shows one pane and hides the other,
    the mode buttons say which is pressed, and the ceilings and the nudge
    describe the way in now shown, at once."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill(unique("switch"))
    add_task(page, "r1", "one")
    add_task(page, "r2", "two")
    rows_pane = page.get_by_test_id("dataset-rows-pane")
    jsonl_pane = page.get_by_test_id("dataset-jsonl-pane")
    ceilings = page.get_by_test_id("dataset-ceilings")
    expect(rows_pane).to_be_visible()
    expect(jsonl_pane).to_be_hidden()
    expect(ceilings).to_contain_text("2 of 2000 tasks")

    page.get_by_test_id("dataset-mode-jsonl").click()

    expect(rows_pane).to_be_hidden()
    expect(jsonl_pane).to_be_visible()
    expect(ceilings).to_have_text("0 of 2000 task lines · 0 of 1912831 bytes")
    expect(page.get_by_test_id("dataset-nudge")).to_have_text(
        "Store waits: there are no tasks to store"
    )
    expect(page.get_by_test_id("dataset-mode-jsonl")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(page.get_by_test_id("dataset-mode-rows")).to_have_attribute(
        "aria-pressed", "false"
    )
    page.get_by_test_id("dataset-jsonl").fill(
        '{"id": "x", "prompt": "a"}\n{"id": "x", "prompt": "b"}\n'
    )
    assert store(page) == "not stored: line 2: duplicate task id 'x'"
    expect(row(page, 1)).not_to_have_attribute("data-state", "refused")

    page.get_by_test_id("dataset-mode-rows").click()

    expect(rows_pane).to_be_visible()
    expect(jsonl_pane).to_be_hidden()
    expect(ceilings).to_contain_text("2 of 2000 tasks")
    expect(page.get_by_test_id("dataset-mode-rows")).to_have_attribute(
        "aria-pressed", "true"
    )
    for index in (0, 1):
        expect(row(page, index)).not_to_have_attribute("data-state", "refused")
        expect(row(page, index).get_by_test_id("dataset-row-msg")).to_have_text("")


def test_a_refusal_the_server_writes_as_a_list_is_printed_as_sentences(
    page, bench, bench_url
):
    """WINDOW: Store with a name the page does not nudge and the door
    refuses (a path separator), and the panel's message.

    FastAPI answers a model violation with a list of error objects;
    printed raw it reads "[object Object]" where the server said exactly
    what was wrong. The message is the server's own sentence."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill("a/b")
    add_task(page, "l1", "list " + uuid.uuid4().hex)
    expect(page.get_by_test_id("dataset-store")).to_be_enabled()
    detail = page.request.post(
        bench_url + "/datasets",
        data={"name": "a/b", "content": '{"id": "l1", "prompt": "p"}\n'},
    ).json()["detail"]
    expect_msg = "; ".join(item["msg"] for item in detail)

    said = store(page)

    assert said == "not stored: " + expect_msg
    assert "[object" not in said


def test_storing_identical_content_twice_keeps_one_row_and_the_first_name(
    page, bench, bench_url
):
    """WINDOW: two Stores of the same composed rows under two names, the
    second from the keyboard.

    The server keeps one row and the earlier name, and the page says so
    rather than leaving a person to wonder where their second name went.
    The library lists the digest once, under the first name. Store is
    greyed while its request is out, which takes focus from it; focus
    goes back to Store after, where the keyboard user left it."""
    bench(["stub/fast"])
    open_datasets(page)
    first, second = unique("first"), unique("second")
    prompt = "twice " + uuid.uuid4().hex
    page.get_by_test_id("dataset-name").fill(first)
    add_task(page, "t1", prompt)
    assert store(page).startswith("stored as " + first)
    digest = digest_of(page, first)

    page.get_by_test_id("dataset-name").fill(second)
    page.get_by_test_id("dataset-store").focus()
    page.keyboard.press("Enter")
    msg = page.get_by_test_id("dataset-msg")
    expect(msg).to_have_text(
        "these tasks were already stored as "
        + first
        + ", so the earlier name stands · sha256 "
        + digest[:7]
    )

    assert focused(page) == ["dataset-store", None, None]
    listed = page.request.get(bench_url + "/datasets").json()["datasets"]
    assert [d["name"] for d in listed if d["digest"] == digest] == [first]
    expect(entry_for(page, second)).to_have_count(0)


def test_a_name_is_stored_as_typed(page, bench, bench_url):
    """WINDOW: a name with spaces at both ends, from the name box to the
    name GET /datasets lists.

    The server records a name as sent, not trimmed, and so must the page:
    a rewritten name is a name nobody chose."""
    bench(["stub/fast"])
    open_datasets(page)
    name = "  " + unique("spaced") + " "
    page.get_by_test_id("dataset-name").fill(name)
    add_task(page, "v1", "verbatim " + uuid.uuid4().hex)

    assert store(page).startswith("stored as " + name.strip())

    listed = page.request.get(bench_url + "/datasets").json()["datasets"]
    assert name in [d["name"] for d in listed]


def test_a_line_separator_in_a_prompt_is_stored_whole(page, bench, bench_url):
    """WINDOW: a prompt holding U+2028, through the rows way in, to the
    stored line.

    JSON.stringify writes U+2028 raw, and JSON allows it raw. The parser
    ends a line at "\n" alone, so the line is one task and the prompt is
    stored as typed, the character itself and not an escape of it."""
    bench(["stub/fast"])
    open_datasets(page)
    name = unique("separator")
    page.get_by_test_id("dataset-name").fill(name)
    prompt = "before" + LS + "after " + uuid.uuid4().hex
    add_task(page, "s1", prompt)

    assert store(page).startswith("stored as " + name)

    served = stored(page, bench_url, digest_of(page, name))["content"]
    assert LS in served
    assert json.loads(served)["prompt"] == prompt


def test_the_builder_holds_fifty_rows_and_says_where_the_rest_go(page, bench):
    """WINDOW: + Task at the builder's row bound.

    The bound is the builder's, not the server's: past it the button is
    greyed and the ceilings line says to paste or upload JSONL. The
    sentence is absent one row short of the bound, so it is about the
    bound and not about rows. Each new row takes focus in its id box, so
    greying + Task at the bound does not drop focus to the page."""
    bench(["stub/fast"])
    open_datasets(page)
    add = page.get_by_test_id("dataset-add-row")
    for _ in range(49):
        add.click()
    expect(page.get_by_test_id("dataset-row")).to_have_count(49)
    expect(page.get_by_test_id("dataset-ceilings")).not_to_contain_text(
        "The builder holds"
    )
    expect(add).to_be_enabled()

    add.focus()
    page.keyboard.press("Enter")

    expect(page.get_by_test_id("dataset-row")).to_have_count(50)
    assert focused(page) == ["dataset-row-id", 49, None]
    expect(add).to_be_disabled()
    expect(page.get_by_test_id("dataset-ceilings")).to_contain_text(
        "The builder holds 50 rows; past that, paste or upload JSONL."
    )


# ---- Keyboard and assistive technology.


def test_keyboard_focus_stays_with_the_row_through_a_rebuild(page, bench, bench_url):
    """WINDOW: the scorer select, a document option, a chip's remove and
    a row's remove, each operated from the keyboard, and the element
    that has focus after the rows are rebuilt.

    A rebuild replaces every row element, and the control that had focus
    went with it: a keyboard user landed on <body> and had to Tab through
    the page again. Focus stays on the same control in the new row (the
    same option, by digest, not merely the first), or moves to the
    nearest one that still makes sense: + Document after its chip, the
    remove button now on the same line, the one above when the last row
    goes, and + Task when none is left. An option pressed while another
    picker's list is still loading keeps focus too."""
    tag = uuid.uuid4().hex
    first_doc = upload_document(page, bench_url, "focus-a.txt", "focus a " + tag)
    second_doc = upload_document(page, bench_url, "focus-b.txt", "focus b " + tag)
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "k1", "one")
    add_task(page, "k2", "two")
    add_task(page, "k3", "three")
    top = row(page, 0)

    top.get_by_test_id("dataset-row-scorer").focus()
    page.keyboard.press("j")
    expect(top.get_by_test_id("dataset-row-scorer")).to_have_value("judge")
    assert focused(page) == ["dataset-row-scorer", 0, None]

    top.get_by_test_id("dataset-row-add-document").click()
    options = top.get_by_test_id("dataset-document-option")
    expect(options.first).to_be_visible()
    # PRE-STATE: the document pressed is not the picker's first option,
    # so focus landing on the first would be caught.
    later = second_doc
    if options.first.get_attribute("data-digest") == later:
        later = first_doc
    assert options.first.get_attribute("data-digest") != later
    held = hold(page, "**/attachments?limit=500", "GET")
    row(page, 1).get_by_test_id("dataset-row-add-document").click()
    wait_held(page, held)
    option = top.locator(
        f"[data-testid=dataset-document-option][data-digest='{later}']"
    )
    option.focus()
    page.keyboard.press("Enter")
    expect(option).to_have_attribute("aria-pressed", "true")
    assert focused(page) == ["dataset-document-option", 0, later]
    while held:
        held.pop().continue_()
    page.unroute("**/attachments?limit=500")

    top.get_by_test_id("dataset-row-document-remove").focus()
    page.keyboard.press("Enter")
    expect(top.get_by_test_id("dataset-row-document")).to_have_count(0)
    assert focused(page) == ["dataset-row-add-document", 0, None]

    row(page, 2).get_by_test_id("dataset-row-remove").focus()
    page.keyboard.press("Enter")
    expect(page.get_by_test_id("dataset-row")).to_have_count(2)
    assert focused(page) == ["dataset-row-remove", 1, None]

    top.get_by_test_id("dataset-row-remove").focus()
    page.keyboard.press("Enter")
    expect(page.get_by_test_id("dataset-row")).to_have_count(1)
    assert focused(page) == ["dataset-row-remove", 0, None]
    page.keyboard.press("Enter")
    expect(page.get_by_test_id("dataset-row")).to_have_count(0)
    assert focused(page) == ["dataset-add-row", None, None]


def test_every_row_control_names_its_line(page, bench, bench_url):
    """WINDOW: the accessible names and descriptions of the second row's
    controls, with two rows' pickers open.

    Every control in a row names its line, so a screen-reader user can
    tell one row's prompt, remove or option from another's; and each
    field is described by its own caption (which its line-bearing name
    would otherwise override, hiding facts like "blank sends none") and
    by its own row's message, never a neighbour's."""
    upload_document(page, bench_url, "named.txt", "named " + uuid.uuid4().hex)
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "a1", "one")
    add_task(page, "a2", "two", "judge")
    second = row(page, 1)
    for index in (0, 1):
        pick = row(page, index).get_by_test_id("dataset-row-add-document")
        expect(pick).to_have_attribute(
            "aria-label", "add a document to line " + str(index + 1)
        )
        pick.click()
        expect(pick).to_have_attribute("aria-expanded", "true")
        option = row(page, index).get_by_test_id("dataset-document-option").first
        expect(option).to_have_attribute(
            "aria-label", re.compile(", for line " + str(index + 1) + "$")
        )
    names = {
        "id": "task id, line 2",
        "prompt": "prompt, line 2",
        "system": "system message, line 2",
        "rubric": "rubric, line 2",
        "threshold": "pass threshold, line 2",
    }
    for key, name in names.items():
        control = second.get_by_test_id("dataset-row-" + key)
        expect(control).to_have_attribute("aria-label", name)
        described = control.get_attribute("aria-describedby").split()
        assert described == ["dataset-row-cap-1-" + key, "dataset-row-msg-1"]
        caption = second.locator("#dataset-row-cap-1-" + key)
        expect(caption).to_have_count(1)
        expect(second.locator("#dataset-row-msg-1")).to_have_count(1)
    # What the descriptions read, for the two whose facts a name would hide.
    expect(second.locator("#dataset-row-cap-1-threshold")).to_have_text(
        "pass threshold · 0 to 1, optional; blank publishes the score alone"
    )
    expect(second.locator("#dataset-row-cap-1-system")).to_have_text(
        "system message · optional; blank sends none"
    )
    expect(second.get_by_test_id("dataset-row-scorer")).to_have_attribute(
        "aria-label", "scorer, line 2"
    )
    expect(second.get_by_test_id("dataset-row-remove")).to_have_attribute(
        "aria-label", "remove the task on line 2"
    )
    second.get_by_test_id("dataset-document-option").first.click()
    expect(second.get_by_test_id("dataset-row-document-remove")).to_have_attribute(
        "aria-label", re.compile("^remove .+ from line 2$")
    )


def test_typing_does_not_re_announce_every_row(page, bench):
    """WINDOW: the live regions in the panel, observed while a name is
    typed beside three rows that each carry a nudge.

    Assigning the same text to a live region replaces its text node and
    re-announces it, and refresh runs on every keystroke: a row's note is
    therefore not a live region of its own, and the one live line that
    does change is written only when its text does."""
    bench(["stub/fast"])
    open_datasets(page)
    for _ in range(3):
        page.get_by_test_id("dataset-add-row").click()
    page.get_by_test_id("dataset-name").fill("x")
    # PRE-STATE: every row does carry a note, so there is text to repeat.
    for index in range(3):
        expect(row(page, index).get_by_test_id("dataset-row-msg")).to_have_text(
            "needs an id"
        )
        expect(
            row(page, index).get_by_test_id("dataset-row-msg")
        ).not_to_have_attribute("role", "status")
    page.evaluate(
        "() => { window.__announced = 0;"
        " const obs = new MutationObserver((records) => {"
        "   window.__announced += records.length; });"
        " for (const el of document.querySelectorAll('#datasets [role=status]'))"
        "   obs.observe(el, {childList: true, characterData: true, subtree: true});"
        " }"
    )

    page.get_by_test_id("dataset-name").press_sequentially("yz")

    expect(page.get_by_test_id("dataset-nudge")).to_have_text(
        "Store waits: line 1 needs an id"
    )
    assert page.evaluate("window.__announced") == 0


# ---- Tasks from the prompt library.


def test_saved_prompts_become_tasks_with_no_id_and_no_scorer(page, bench, bench_url):
    """WINDOW: + From saved prompts, from the picker's checkboxes to the
    rows it adds and the lines Store sends.

    NOTHING IS PRE-FILLED: each checked prompt becomes one row with that
    prompt, an empty id box and no scorer, and none is checked for the
    person. Store waits on the ids. The prompt is taken exactly as saved
    (a leading space and a trailing newline survive) except that its
    line breaks are the box's: a textarea turns CRLF and a lone CR into
    LF, and a row that sent a CR while its box showed an LF would store
    a prompt nobody saw."""
    tail = uuid.uuid4().hex
    texts = [
        " first saved " + tail + "\n",
        "second\r\nsaved " + tail,
        "third\rsaved " + tail,
    ]
    for text in texts:
        resp = page.request.post(
            bench_url + "/prompts", data={"name": unique("saved"), "text": text}
        )
        assert resp.ok, resp.text()
    bench(["stub/fast"])
    open_datasets(page)
    toggle = page.get_by_test_id("dataset-from-prompts")
    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    options = page.get_by_test_id("dataset-prompt-option")
    expect(options.first).to_be_visible()
    for option in options.all():
        expect(option).not_to_be_checked()
    for option in options.all():
        if option.get_attribute("data-text") in texts:
            option.check()
    expect(page.get_by_test_id("dataset-prompt-add")).to_have_text("Add 3 tasks")

    page.get_by_test_id("dataset-prompt-add").click()

    expect(page.get_by_test_id("dataset-row")).to_have_count(3)
    expect(toggle).to_have_attribute("aria-expanded", "false")
    assert focused(page) == ["dataset-row-id", 0, None]
    boxes = [
        row(page, index).get_by_test_id("dataset-row-prompt").input_value()
        for index in range(3)
    ]
    as_boxed = [text.replace("\r\n", "\n").replace("\r", "\n") for text in texts]
    # In the library's order, which is the picker's; the set is the claim.
    assert sorted(boxes) == sorted(as_boxed)
    for index in range(3):
        added = row(page, index)
        expect(added.get_by_test_id("dataset-row-id")).to_have_value("")
        expect(added.get_by_test_id("dataset-row-scorer")).to_have_value("")
    name = unique("from prompts")
    page.get_by_test_id("dataset-name").fill(name)
    expect(page.get_by_test_id("dataset-nudge")).to_have_text(
        "Store waits: line 1 needs an id"
    )
    for index in range(3):
        row(page, index).get_by_test_id("dataset-row-id").fill(f"p{index}")

    assert store(page).startswith("stored as " + name)

    served = stored(page, bench_url, digest_of(page, name))["content"]
    assert [json.loads(line)["prompt"] for line in served.splitlines()] == boxes


def test_the_import_is_all_or_nothing_at_the_row_bound(page, bench, bench_url):
    """WINDOW: + From saved prompts with 48 rows and then 49 in the
    builder, two prompts checked each time.

    The whole import or none of it, at the exact bound: 48 and 2 make 50
    and are taken, and focus goes to the first imported row's id box,
    line 49 and not line 1; 49 and 2 would make 51 and nothing is added,
    in a message that reads as the refusal it is."""
    tail = uuid.uuid4().hex
    texts = ["cap one " + tail, "cap two " + tail]
    for text in texts:
        assert page.request.post(
            bench_url + "/prompts", data={"name": unique("cap"), "text": text}
        ).ok
    bench(["stub/fast"])
    open_datasets(page)

    def import_both():
        page.get_by_test_id("dataset-from-prompts").click()
        expect(page.get_by_test_id("dataset-prompt-option").first).to_be_visible()
        for option in page.get_by_test_id("dataset-prompt-option").all():
            if option.get_attribute("data-text") in texts:
                option.check()
        page.get_by_test_id("dataset-prompt-add").click()

    for _ in range(48):
        page.get_by_test_id("dataset-add-row").click()
    import_both()
    expect(page.get_by_test_id("dataset-row")).to_have_count(50)
    assert focused(page) == ["dataset-row-id", 48, None]

    row(page, 0).get_by_test_id("dataset-row-remove").click()
    expect(page.get_by_test_id("dataset-row")).to_have_count(49)
    import_both()

    msg = page.get_by_test_id("dataset-msg")
    expect(msg).to_have_text(
        "the builder holds 50 rows; 49 are here and 2 were checked. Nothing was "
        "added. Check fewer, or paste JSONL."
    )
    expect(msg).to_have_attribute("data-state", "refused")
    expect(page.get_by_test_id("dataset-row")).to_have_count(49)


def test_a_prompt_picker_reopened_while_loading_lists_each_prompt_once(
    page, bench, bench_url
):
    """WINDOW: + From saved prompts opened and closed while GET /prompts
    is held and then released; then opened, closed and opened again
    while held, and both answers released.

    A picker closed before its answer came draws nothing into itself.
    Two runs appending to one live picker listed every prompt twice with
    two Add buttons, each counting only its own boxes, so an import could
    silently drop prompts ticked in the other set."""
    assert page.request.post(
        bench_url + "/prompts",
        data={"name": unique("once"), "text": "once " + uuid.uuid4().hex},
    ).ok
    saved = len(page.request.get(bench_url + "/prompts").json()["prompts"])
    bench(["stub/fast"])
    open_datasets(page)
    held = hold(page, "**/prompts", "GET")
    toggle = page.get_by_test_id("dataset-from-prompts")
    options = page.get_by_test_id("dataset-prompt-option")

    toggle.click()
    wait_held(page, held)
    toggle.click()
    settle(page, held.pop())
    expect(page.get_by_test_id("dataset-prompt-picker")).to_be_hidden()
    expect(options).to_have_count(0)

    toggle.click()
    toggle.click()
    toggle.click()
    wait_held(page, held, 2)
    while held:
        held.pop().continue_()
    page.unroute("**/prompts")

    expect(page.get_by_test_id("dataset-prompt-add")).to_have_count(1)
    expect(options).to_have_count(saved)


# ---- Stored documents.


def test_a_task_cites_stored_documents_in_the_order_picked(page, bench, bench_url):
    """WINDOW: + Document on a row, the stored-document picker, the chips
    and the line Store sends.

    The picker lists what GET /attachments holds and a pick cites the
    digest, which is how a task has always cited a document. Two are
    picked in reverse digest order and stored in the order picked,
    because order is part of what a task declares; pressing a pressed
    option un-cites it. At the per-task cap every other option is
    greyed, so the page never offers a fifth the parser refuses, while
    the pressed ones stay live so one can be let go; and one chip's x
    removes that chip and no other."""
    tag = uuid.uuid4().hex
    digests = [
        upload_document(page, bench_url, f"builder-{n}.txt", f"doc {n} {tag}")
        for n in range(5)
    ]
    bench(["stub/fast"])
    open_datasets(page)
    name = unique("cites")
    page.get_by_test_id("dataset-name").fill(name)
    task = add_task(page, "d1", "what do the notes say?")
    task.get_by_test_id("dataset-row-add-document").click()

    def option(digest):
        return row(page, 0).locator(
            f"[data-testid=dataset-document-option][data-digest='{digest}']"
        )

    chips = row(page, 0).get_by_test_id("dataset-row-document")
    option(digests[4]).click()
    expect(chips).to_have_count(1)
    option(digests[4]).click()
    expect(chips).to_have_count(0)
    picked = sorted(digests[:2], reverse=True)
    expect(option(picked[0])).to_have_attribute("aria-pressed", "false")
    for digest in picked:
        option(digest).click()
    expect(chips).to_have_count(2)
    assert [chip.get_attribute("data-digest") for chip in chips.all()] == picked

    assert store(page).startswith("stored as " + name)
    stored_digest = digest_of(page, name)
    line = json.loads(stored(page, bench_url, stored_digest)["content"])
    assert line["attachments"] == picked
    expect(entry_for(page, name).get_by_test_id("dataset-entry-meta")).to_contain_text(
        "cites documents"
    )

    for digest in digests[2:4]:
        option(digest).click()
    expect(chips).to_have_count(4)
    expect(row(page, 0).get_by_test_id("dataset-document-note")).to_contain_text(
        "A task cites at most 4 documents."
    )
    expect(option(digests[4])).to_be_disabled()
    for digest in digests[:4]:
        expect(option(digest)).to_be_enabled()

    row(page, 0).locator(
        f"[data-testid=dataset-row-document][data-digest='{digests[2]}'] "
        "[data-testid=dataset-row-document-remove]"
    ).click()
    expect(chips).to_have_count(3)
    assert [chip.get_attribute("data-digest") for chip in chips.all()] == [
        *picked,
        digests[3],
    ]


def test_a_picker_opened_again_offers_a_document_stored_since(page, bench, bench_url):
    """WINDOW: a row's picker opened, closed, a document attached, the
    picker opened again, and the new document pressed.

    The picker is fetched afresh on every open. Cached while the panel
    stayed open, the list went on saying what it said first, including
    the empty note that tells a person to attach a file and come back.
    The redraw a press causes draws from the newest answer, so the new
    document is still offered, pressed, after it is picked."""
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "r1", "one")
    pick = row(page, 0).get_by_test_id("dataset-row-add-document")
    pick.click()
    expect(row(page, 0).get_by_test_id("dataset-document-note")).not_to_have_text(
        "loading stored documents"
    )
    pick.click()
    digest = upload_document(page, bench_url, "later.txt", "later " + uuid.uuid4().hex)
    option = row(page, 0).locator(
        f"[data-testid=dataset-document-option][data-digest='{digest}']"
    )

    pick.click()

    expect(option).to_have_count(1)
    option.click()
    expect(option).to_have_attribute("aria-pressed", "true")
    expect(
        row(page, 0).locator(
            f"[data-testid=dataset-row-document][data-digest='{digest}']"
        )
    ).to_have_count(1)


def test_a_picker_whose_answer_returns_late_draws_once_or_not_at_all(
    page, bench, bench_url
):
    """WINDOW: a row's picker with GET /attachments held, in three shapes:
    closed before its answer came; closed and reopened while held; and
    another row's picker whose row was removed before its answer came.

    A closed picker draws nothing into itself. Two runs appending to one
    picker listed every document twice. A run whose row was removed read
    a row that no longer existed and threw. Each answer is let go and the
    page allowed to act on it before anything is asserted, so a late run
    that drew or threw would be seen (the collectors see a throw)."""
    upload_document(page, bench_url, "late.txt", "late " + uuid.uuid4().hex)
    total = len(
        page.request.get(bench_url + "/attachments?limit=500").json()["attachments"]
    )
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "r1", "one")
    add_task(page, "r2", "two")
    held = hold(page, "**/attachments?limit=500", "GET")
    pick = row(page, 0).get_by_test_id("dataset-row-add-document")
    options = row(page, 0).get_by_test_id("dataset-document-option")

    pick.click()
    wait_held(page, held)
    pick.click()
    settle(page, held.pop())
    expect(row(page, 0).get_by_test_id("dataset-row-picker")).to_be_hidden()
    expect(options).to_have_count(0)

    pick.click()
    pick.click()
    pick.click()
    wait_held(page, held, 2)
    while held:
        settle(page, held.pop())
    expect(options).to_have_count(total)

    row(page, 1).get_by_test_id("dataset-row-add-document").click()
    wait_held(page, held)
    row(page, 1).get_by_test_id("dataset-row-remove").click()
    settle(page, held.pop())
    page.unroute("**/attachments?limit=500")

    expect(page.get_by_test_id("dataset-row")).to_have_count(1)
    expect(page.get_by_test_id("dataset-row-picker")).to_be_hidden()


def test_the_empty_picker_names_the_snapshot_route_only_where_it_exists(
    page, bench, snapshot_bench
):
    """WINDOW: the note a picker shows when GET /attachments is empty, on
    the default bench (snapshots off), on the snapshot bench, and on a
    page that does not know the bench's posture.

    The page knows whether the bench will compose a snapshot; on one that
    will not, the composer says so, and a note here sending a person to
    compose one would offer a door the server refuses. Where the posture
    is unknown, the page claims nothing about snapshots either way."""
    empty = json.dumps({"attachments": []})

    def note_on(open_bench, lineup, posture=False):
        open_bench(lineup)
        page.route(
            "**/attachments?limit=500",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=empty
            ),
        )
        if posture is None:
            # After the catalog has said, so its answer cannot land later
            # and put the posture back.
            page.wait_for_function("() => window.BenchState.snapshots !== null")
            page.evaluate("() => { window.BenchState.snapshots = null; }")
        open_datasets(page)
        add_task(page, "e1", "one")
        row(page, 0).get_by_test_id("dataset-row-add-document").click()
        note = row(page, 0).get_by_test_id("dataset-document-note")
        expect(note).to_contain_text("No stored documents.")
        text = note.inner_text()
        page.unroute("**/attachments?limit=500")
        return text

    off = note_on(bench, ["stub/fast"])
    on = note_on(snapshot_bench, ["model/alpha"])
    unknown = note_on(snapshot_bench, ["model/alpha"], posture=None)

    assert "snapshot" not in off
    assert "Attach a file in the composer above" in off
    assert "compose a snapshot" in on
    assert "snapshot" not in unknown


def test_a_repository_snapshot_is_cited_like_any_document(
    page, snapshot_bench, snapshot_root, snapshot_bench_url
):
    """WINDOW: a snapshot composed in the composer, the builder's picker
    on the same bench, and the line Store sends.

    THE COMMISSION'S ROUTE FOR A SNAPSHOT INTO A TASK: it is a stored
    attachment of kind snapshot, offered by the picker as "repository
    snapshot", and cited by its digest like any other document."""
    snapshot_bench(["model/alpha"])
    page.get_by_test_id("snapshot-open").click()
    page.get_by_test_id("snapshot-root").fill(str(snapshot_root))
    page.get_by_test_id("snapshot-patterns").fill("pkg/*.py")
    page.get_by_test_id("snapshot-compose").click()
    expect(page.get_by_test_id("attachment-chip")).to_have_count(1)
    stored_docs = page.request.get(
        snapshot_bench_url + "/attachments?limit=500"
    ).json()["attachments"]
    digest = next(d["digest"] for d in stored_docs if d["kind"] == "snapshot")
    open_datasets(page)
    name = unique("snapshot")
    page.get_by_test_id("dataset-name").fill(name)
    add_task(page, "s1", "what does the tree define?")
    row(page, 0).get_by_test_id("dataset-row-add-document").click()
    option = row(page, 0).locator(
        f"[data-testid=dataset-document-option][data-digest='{digest}']"
    )
    expect(option).to_contain_text("repository snapshot · snapshot")

    option.click()

    assert store(page).startswith("stored as " + name)
    line = json.loads(
        stored(page, snapshot_bench_url, digest_of(page, name))["content"]
    )
    assert line["attachments"] == [digest]


# ---- The JSONL way in.


def test_an_uploaded_file_is_stored_byte_for_byte_and_is_the_path_doors_digest(
    page, bench, bench_url, tmp_path
):
    """WINDOW: the JSONL way in, from a file chosen in the picker, with
    CRLF line ends, to the digest stored and the digest the path door
    records for the same file.

    A textarea would have normalized the CRLF to LF and stored a digest
    the file does not have. The file is held as the text its bytes
    decoded to and sent as it is, so the stored digest is the file's
    sha256, and an experiment created from the same file by path records
    the same one: one dataset, whichever door. The name is not taken from
    the file: the box stays empty until the person fills it."""
    raw = (
        '{"id": "u1", "prompt": "uploaded ' + uuid.uuid4().hex + '"}\r\n'
        '{"id": "u2", "prompt": "café"}\r\n'
    ).encode("utf-8")
    path = tmp_path / "uploaded.jsonl"
    path.write_bytes(raw)
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()

    page.get_by_test_id("dataset-file").set_input_files(str(path))

    expect(page.get_by_test_id("dataset-file-label")).to_have_text(
        "uploaded.jsonl · 2 lines · " + str(len(raw)) + " bytes, sent exactly as read"
    )
    expect(page.get_by_test_id("dataset-name")).to_have_value("")
    expect(page.get_by_test_id("dataset-nudge")).to_have_text(
        "Store waits: name the dataset"
    )
    expect(page.get_by_test_id("dataset-ceilings")).to_contain_text(
        "2 of 2000 task lines"
    )
    name = unique("uploaded")
    page.get_by_test_id("dataset-name").fill(name)
    assert store(page).startswith("stored as " + name)
    digest = digest_of(page, name)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert stored(page, bench_url, digest)["content"].encode() == raw
    created = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    )
    assert created.ok, created.text()
    recorded = page.request.get(
        bench_url + "/experiments/" + str(created.json()["id"])
    ).json()
    assert recorded["dataset_digest"] == digest


def test_a_loaded_file_hides_the_box_and_clearing_it_brings_the_draft_back(
    page, bench, bench_url, tmp_path
):
    """WINDOW: text typed into the JSONL box, a file loaded over it,
    Store, clear, and a second Store.

    Store sends the file. A box still showing the typed text would show
    one dataset while another was stored, so the box is hidden while a
    file is loaded; the typed draft was never replaced. Clear forgets the
    file entirely: the label and the clear button go, the ceilings count
    the draft, focus goes to the box it brought back, and the next Store
    sends the draft. The file input is emptied after each pick, so the
    same file can be chosen again."""
    typed = '{"id": "typed", "prompt": "draft ' + uuid.uuid4().hex + '"}\n'
    raw = ('{"id": "file", "prompt": "file ' + uuid.uuid4().hex + '"}\n').encode()
    path = tmp_path / "over.jsonl"
    path.write_bytes(raw)
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()
    box = page.get_by_test_id("dataset-jsonl")
    box.fill(typed)
    name = unique("hidden box")
    page.get_by_test_id("dataset-name").fill(name)

    page.get_by_test_id("dataset-file").set_input_files(str(path))

    expect(box).to_be_hidden()
    assert page.get_by_test_id("dataset-file").input_value() == ""
    assert store(page).startswith("stored as " + name)
    assert stored(page, bench_url, digest_of(page, name))["content"].encode() == raw

    page.get_by_test_id("dataset-file-clear").click()

    expect(box).to_be_visible()
    expect(box).to_have_value(typed)
    assert focused(page) == ["dataset-jsonl", None, None]
    expect(page.get_by_test_id("dataset-file-label")).to_have_text("")
    expect(page.get_by_test_id("dataset-file-clear")).to_be_hidden()
    expect(page.get_by_test_id("dataset-ceilings")).to_have_text(
        "1 of 2000 task lines · " + str(len(typed.encode())) + " of 1912831 bytes"
    )
    draft_name = unique("the draft")
    page.get_by_test_id("dataset-name").fill(draft_name)
    assert store(page).startswith("stored as " + draft_name)
    assert stored(page, bench_url, digest_of(page, draft_name))["content"] == typed


def test_pasted_jsonl_is_sent_as_it_reads_in_the_box(page, bench, bench_url):
    """WINDOW: the JSONL way in, typed into the box.

    The page parses nothing and rewrites nothing: it counts lines, as the
    parser splits them, against the task ceiling, and sends the box's
    text, so the stored digest is the sha256 of exactly that text, a
    decomposed accent (e and a combining acute, which NFC would fold into
    one character) included."""
    text = (
        '{"id": "p1", "prompt": "pasted ' + uuid.uuid4().hex + '"}\n'
        '{"id": "p2", "prompt": "cafe' + chr(0x301) + '"}\n'
    )
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()
    name = unique("pasted")
    page.get_by_test_id("dataset-name").fill(name)
    page.get_by_test_id("dataset-jsonl").fill(text)
    expect(page.get_by_test_id("dataset-ceilings")).to_have_text(
        "2 of 2000 task lines · " + str(len(text.encode())) + " of 1912831 bytes"
    )

    assert store(page).startswith("stored as " + name)

    assert digest_of(page, name) == hashlib.sha256(text.encode()).hexdigest()


def test_a_file_that_is_not_utf8_is_said_so_and_never_sent(
    page, bench, bench_url, tmp_path
):
    """WINDOW: a file of bytes that do not decode as UTF-8, chosen in the
    JSONL picker, the Store button after it, and a good file after that.

    Decoding it leniently would replace the bad bytes with U+FFFD and
    store a dataset the file never held. The page says the file is not
    UTF-8 and holds nothing, so there is nothing to send: Store stays
    greyed on the reason the empty box gives. A good file chosen next
    takes the refusal off the panel, since it is no longer about what is
    loaded."""
    path = tmp_path / "latin.jsonl"
    path.write_bytes(b'{"id": "l1", "prompt": "caf\xe9"}\n')
    good = tmp_path / "good.jsonl"
    good.write_bytes(b'{"id": "g1", "prompt": "good"}\n')
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()

    page.get_by_test_id("dataset-file").set_input_files(str(path))

    msg = page.get_by_test_id("dataset-msg")
    expect(msg).to_have_text(
        "latin.jsonl is not valid UTF-8, so it has no text to send. The bench "
        "reads datasets as UTF-8, by path as well as here."
    )
    expect(msg).to_have_attribute("data-state", "refused")
    expect(page.get_by_test_id("dataset-file-label")).to_have_text("")
    page.get_by_test_id("dataset-name").fill(unique("latin"))
    expect(page.get_by_test_id("dataset-store")).to_be_disabled()
    expect(page.get_by_test_id("dataset-nudge")).to_have_text(
        "Store waits: there are no tasks to store"
    )

    page.get_by_test_id("dataset-file").set_input_files(str(good))

    expect(page.get_by_test_id("dataset-file-label")).to_contain_text("good.jsonl")
    expect(msg).to_have_text("")


def test_a_file_with_a_byte_order_mark_is_refused_as_the_path_door_refuses_it(
    page, bench, bench_url, tmp_path
):
    """WINDOW: a UTF-8 file beginning with a byte order mark, uploaded,
    Stored, and the same file named by path.

    The mark is kept, so the upload is refused in the same words as the
    path door refuses the same file. Stripped, the page would store a
    digest the file does not have, and the two doors would disagree about
    one dataset."""
    raw = b"\xef\xbb\xbf" + b'{"id": "b1", "prompt": "bom"}\n'
    path = tmp_path / "bom.jsonl"
    path.write_bytes(raw)
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()
    name = unique("bom")
    page.get_by_test_id("dataset-name").fill(name)
    page.get_by_test_id("dataset-file").set_input_files(str(path))
    by_path = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("bom by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    )
    assert by_path.status == 422
    sentence = by_path.json()["detail"]
    assert "BOM" in sentence

    said = store(page)

    assert said == "not stored: " + sentence
    expect(entry_for(page, name)).to_have_count(0)


def test_a_file_exactly_at_the_byte_ceiling_is_sent_and_one_over_is_not(
    page, bench, bench_url, tmp_path
):
    """WINDOW: two files chosen in the JSONL picker, one exactly
    MAX_DATASET_BYTES of multi-byte text and one a byte longer.

    The page refuses a file over the ceiling before reading it, in its
    own sentence, which names the API as the route a larger dataset
    takes; and the comparison must be the server's: at the ceiling the
    file is sent and stored, and a >= in place of > would refuse it."""
    at = tmp_path / "at.jsonl"
    at.write_bytes(exactly(main.MAX_DATASET_BYTES).encode())
    over = tmp_path / "over.jsonl"
    over.write_bytes(exactly(main.MAX_DATASET_BYTES + 1).encode())
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()

    page.get_by_test_id("dataset-file").set_input_files(str(over))

    msg = page.get_by_test_id("dataset-msg")
    expect(msg).to_have_text(
        "over.jsonl is "
        + str(main.MAX_DATASET_BYTES + 1)
        + " bytes, over the "
        + str(main.MAX_DATASET_BYTES)
        + " byte limit for a stored dataset. A larger dataset goes by path: "
        "name its file with dataset_path when the experiment is created "
        "through the API."
    )
    expect(msg).to_have_attribute("data-state", "refused")

    name = unique("at the ceiling")
    page.get_by_test_id("dataset-name").fill(name)
    page.get_by_test_id("dataset-file").set_input_files(str(at))
    expect(page.get_by_test_id("dataset-store")).to_be_enabled()
    assert store(page).startswith("stored as " + name)


# ---- The library, and the panel itself.


def test_selecting_a_stored_dataset_selects_it_and_loads_nothing(
    page, bench, bench_url
):
    """WINDOW: clicks on a stored dataset in the library, a reload of the
    library, and the builder beside it.

    Selecting marks the row, says it is selected for a new experiment
    (the experiment panel creates over it; tests/browser/test_n3.py
    proves that half) and hands it to BenchDatasets.selected; the
    builder's rows and name
    are untouched, because a stored dataset is a record and editing it is
    storing a new one. A reload that finds the same selection does not
    re-announce it, and a second click lets it go."""
    name = unique("selectable")
    created = page.request.post(
        bench_url + "/datasets",
        data={
            "name": name,
            "content": json.dumps({"id": "t1", "prompt": uuid.uuid4().hex}) + "\n",
        },
    )
    assert created.ok, created.text()
    digest = created.json()["digest"]
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "mine", "my draft")
    entry = entry_for(page, name)
    selected = page.get_by_test_id("dataset-selected")
    expect(entry).to_have_attribute("aria-pressed", "false")

    entry.click()

    expect(entry).to_have_attribute("aria-pressed", "true")
    expect(selected).to_have_text(
        "selected for a new experiment: " + name + " · sha256 " + digest[:7]
    )
    assert page.evaluate("window.BenchDatasets.selected().digest") == digest
    expect(page.get_by_test_id("dataset-row")).to_have_count(1)
    expect(row(page, 0).get_by_test_id("dataset-row-id")).to_have_value("mine")
    expect(page.get_by_test_id("dataset-name")).to_have_value("")

    page.evaluate(
        "() => { window.__announced = 0;"
        " new MutationObserver((r) => { window.__announced += r.length; })"
        "   .observe(document.querySelector('[data-testid=dataset-selected]'),"
        "     {childList: true, characterData: true, subtree: true}); }"
    )
    page.evaluate("() => window.BenchDatasets.refresh()")
    expect(page.get_by_test_id("dataset-library")).to_have_attribute(
        "data-state", "ready"
    )
    expect(entry_for(page, name)).to_have_attribute("aria-pressed", "true")
    assert page.evaluate("window.__announced") == 0

    entry_for(page, name).click()

    expect(entry_for(page, name)).to_have_attribute("aria-pressed", "false")
    expect(selected).to_have_text("")
    assert page.evaluate("window.BenchDatasets.selected()") is None


def test_a_full_library_says_it_lists_only_the_newest(page, bench):
    """WINDOW: the library when GET /datasets?limit=500 answers with 500
    entries, and when it answers with fewer.

    The list is asked for its newest 500, and a full answer says so;
    silently capped, older datasets would simply not exist to the page."""
    entries = [
        {
            "digest": f"{n:064x}",
            "name": f"bulk {n}",
            "created_at": "2026-09-24T00:00:00+00:00",
            "task_count": 1,
            "scorers": [],
            "cites_documents": False,
        }
        for n in range(500)
    ]
    answer = {"body": json.dumps({"datasets": entries})}
    page.route(
        "**/datasets?limit=500",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=answer["body"]
        ),
    )
    bench(["stub/fast"])

    open_datasets(page)

    expect(page.get_by_test_id("dataset-entry")).to_have_count(500)
    expect(page.get_by_test_id("dataset-library-note")).to_have_text(
        "The newest 500 are listed and can be selected; older ones stay stored, "
        "and an experiment over one is created through the API with its "
        "dataset_digest."
    )
    answer["body"] = json.dumps({"datasets": entries[:499]})
    page.get_by_test_id("datasets-toggle").click()
    open_datasets(page)
    expect(page.get_by_test_id("dataset-entry")).to_have_count(499)
    expect(page.get_by_test_id("dataset-library-note")).to_have_count(0)


def test_opening_the_panel_says_loading_at_the_click(page, bench):
    """WINDOW: the library's data-state read in the same task as the
    click on the toggle, before any event the click queues has run.

    toggle is dispatched asynchronously, so without the click-time claim
    the list would still read its last state when the click lands; the
    history and experiments panels make the same claim. Read any later
    and the toggle handler's own load would say "loading" too, so the
    test could not tell the two apart."""
    bench(["stub/fast"])
    library = page.get_by_test_id("dataset-library")
    expect(library).to_have_attribute("data-state", "idle")

    at_click = page.evaluate(
        "() => { document.querySelector('[data-testid=datasets-toggle]').click();"
        " return document.querySelector('[data-testid=dataset-library]')"
        ".dataset.state; }"
    )

    assert at_click == "loading"
    expect(library).to_have_attribute("data-state", re.compile("^(ready|empty)$"))


def test_back_and_forward_bring_back_no_draft(page, bench, bench_url):
    """WINDOW: a name and a JSONL draft typed, a navigation away, and
    Back.

    A draft that reappeared would be tasks nobody chose to keep, and the
    browser's form restoration refilled both boxes after the page had
    painted, so the nudge described an empty name beside a full box."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill("restored name")
    page.get_by_test_id("dataset-mode-jsonl").click()
    page.get_by_test_id("dataset-jsonl").fill('{"id": "b", "prompt": "left"}\n')
    page.goto(bench_url + "/models")

    page.go_back()

    expect(page.get_by_test_id("dataset-name")).to_have_value("")
    page.get_by_test_id("datasets-toggle").click()
    page.get_by_test_id("dataset-mode-jsonl").click()
    expect(page.get_by_test_id("dataset-jsonl")).to_have_value("")
    expect(page.get_by_test_id("dataset-store")).to_be_disabled()


def test_the_task_line_ceiling_holds_at_max_tasks(page, bench, bench_url):
    """WINDOW: 2001 pasted task lines, then 2000, with the ceilings line
    and Store for each.

    The one count where the task-line ceiling matters. At 2001 the line
    says so and the server refuses on line 2001, in its words; at 2000
    the line reads at the ceiling and the server stores every task. The
    count greys nothing: it is a count, and the refusal is the server's."""
    tag = uuid.uuid4().hex[:6]
    lines = [json.dumps({"id": f"m{n}-{tag}", "prompt": "p"}) for n in range(2001)]
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-mode-jsonl").click()
    name = unique("max tasks")
    page.get_by_test_id("dataset-name").fill(name)
    box = page.get_by_test_id("dataset-jsonl")
    ceilings = page.get_by_test_id("dataset-ceilings")

    box.fill("\n".join(lines) + "\n")
    expect(ceilings).to_contain_text("2001 of 2000 task lines")
    expect(page.get_by_test_id("dataset-store")).to_be_enabled()
    assert store(page) == "not stored: line 2001: more than 2000 tasks; split the file"

    box.fill("\n".join(lines[:2000]) + "\n")
    expect(ceilings).to_contain_text("2000 of 2000 task lines")
    assert store(page).startswith("stored as " + name)
    expect(entry_for(page, name).get_by_test_id("dataset-entry-meta")).to_contain_text(
        "2000 tasks"
    )


def test_a_store_whose_answer_is_lost_or_unreadable_is_not_called_a_refusal(
    page, bench, collectors
):
    """WINDOW: Store with POST /datasets aborted, and then answered 201
    with a body that is not JSON.

    Neither is a refusal. With no answer, the request may never have left
    or the server may have stored the dataset and only the answer was
    lost; with an unreadable one, it probably was stored. The page says
    it does not know, leaves it to the list below, and does not paint it
    in the refusal's colour."""
    collectors.extend([ABORTED_RESOURCE, "bench: storing a dataset failed"])
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill(unique("lost"))
    add_task(page, "l1", "lost " + uuid.uuid4().hex)
    msg = page.get_by_test_id("dataset-msg")

    page.route("**/datasets", lambda route: route.abort())
    said = store(page)
    page.unroute("**/datasets")

    assert said.startswith("no answer came back (")
    assert said.endswith("); the dataset may be stored, and the list below will say.")
    expect(msg).to_have_attribute("data-state", "")

    page.route(
        "**/datasets",
        lambda route: route.fulfill(
            status=201, content_type="application/json", body="not json"
        ),
    )
    said = store(page)
    page.unroute("**/datasets")

    assert said == (
        "the bench answered 201 with a body this page could not read; the "
        "dataset may be stored, and the list below will say."
    )
    expect(msg).to_have_attribute("data-state", "")


def test_a_document_list_that_fails_to_load_says_so(page, bench, bench_url, collectors):
    """WINDOW: one row's picker loaded, then another's opened with GET
    /attachments aborted, then an option pressed in the first.

    The picker says the list failed and why; left saying "loading", it
    would promise a list that is never coming. The failure leaves the
    list the first picker drew from as it was, so pressing an option
    there redraws it at once and keeps focus on the option."""
    collectors.extend([ABORTED_RESOURCE, "bench: stored documents failed to load"])
    upload_document(page, bench_url, "kept.txt", "kept " + uuid.uuid4().hex)
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "f1", "one")
    add_task(page, "f2", "two")
    row(page, 0).get_by_test_id("dataset-row-add-document").click()
    option = row(page, 0).get_by_test_id("dataset-document-option").first
    expect(option).to_be_visible()
    digest = option.get_attribute("data-digest")
    page.route("**/attachments?limit=500", lambda route: route.abort())

    row(page, 1).get_by_test_id("dataset-row-add-document").click()

    expect(row(page, 1).get_by_test_id("dataset-document-note")).to_have_text(
        re.compile("^failed to load stored documents: .+")
    )
    option.focus()
    page.keyboard.press("Enter")
    pressed = row(page, 0).locator(
        f"[data-testid=dataset-document-option][data-digest='{digest}']"
    )
    expect(pressed).to_have_attribute("aria-pressed", "true")
    assert focused(page) == ["dataset-document-option", 0, digest]


def test_a_full_picker_says_it_offers_only_the_newest(page, bench):
    """WINDOW: a row's picker when GET /attachments?limit=500 answers with
    500 documents, and when it answers with fewer.

    The picker asks for the newest 500, and a full answer says so and
    says how an older document is cited; silently capped, it would not
    exist to the page."""
    docs = [
        {
            "digest": f"{n:064x}",
            "filename": f"bulk-{n}.txt",
            "kind": "document",
            "byte_size": 10,
        }
        for n in range(500)
    ]
    answer = {"body": json.dumps({"attachments": docs})}
    page.route(
        "**/attachments?limit=500",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=answer["body"]
        ),
    )
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "b1", "one")
    pick = row(page, 0).get_by_test_id("dataset-row-add-document")
    note = row(page, 0).get_by_test_id("dataset-document-note")

    pick.click()

    expect(note).to_have_text(
        "Cite up to 4 stored documents. The newest 500 are offered; an older "
        "one is cited by its digest in pasted JSONL."
    )
    pick.click()
    answer["body"] = json.dumps({"attachments": docs[:499]})
    pick.click()
    expect(note).to_have_text("Cite up to 4 stored documents.")


def test_store_gives_focus_back_only_where_it_took_it(page, bench):
    """WINDOW: Store pressed from the keyboard with its POST held, focus
    moved into the name box while it is held, and the answer.

    Store is greyed for the request, which takes focus from it, and the
    page gives focus back to Store after; but only if the person left it
    there. Having moved on to the name box, they keep typing there, and
    a page that pulled focus back to Store would take their keystrokes."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill(unique("focus kept"))
    add_task(page, "f1", "kept " + uuid.uuid4().hex)
    held = hold(page, "**/datasets", "POST")
    page.get_by_test_id("dataset-store").focus()
    page.keyboard.press("Enter")
    wait_held(page, held)
    page.get_by_test_id("dataset-name").click()
    assert focused(page) == ["dataset-name", None, None]

    held.pop().continue_()

    expect(page.get_by_test_id("dataset-msg")).to_contain_text("stored as ")
    assert focused(page) == ["dataset-name", None, None]


def test_a_refusal_line_is_not_announced_again_while_it_stands(page, bench):
    """WINDOW: the panel's message line, observed while the dataset's
    name is typed beside a standing refusal.

    The message line is a live region, and refresh rewrites it on every
    keystroke to keep its note about a changed text true. The name is
    not part of what Store sends, so the line's text does not change,
    and an unchanged line must not be rewritten: each rewrite would
    announce the refusal again."""
    bench(["stub/fast"])
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill(unique("quiet"))
    add_task(page, "dup", "one")
    add_task(page, "dup", "two")
    assert store(page) == "not stored: line 2: duplicate task id 'dup'"
    page.evaluate(
        "() => { window.__announced = 0;"
        " new MutationObserver((r) => { window.__announced += r.length; })"
        "   .observe(document.querySelector('[data-testid=dataset-msg]'),"
        "     {childList: true, characterData: true, subtree: true}); }"
    )

    page.get_by_test_id("dataset-name").press_sequentially("xyz")

    expect(row(page, 1)).to_have_attribute("data-state", "refused")
    assert page.evaluate("window.__announced") == 0


def test_the_document_list_keeps_the_newest_answer(page, bench, bench_url):
    """WINDOW: two pickers opened, each GET /attachments answered by the
    server at once but the answer held from the page, a document
    attached between the two answers, the answers delivered in reverse
    order, and a press in the picker whose answer was newer.

    Answers can arrive out of order. The older one, which does not know
    the new document, arrives last; kept over the newer one, the redraw
    after a press would drop the new document from the picker that had
    just offered it. (Holding the REQUEST would not do: a request let go
    late is answered late, with the new document in it.)"""
    upload_document(page, bench_url, "older.txt", "older " + uuid.uuid4().hex)
    bench(["stub/fast"])
    open_datasets(page)
    add_task(page, "o1", "one")
    add_task(page, "o2", "two")
    answers = []

    def answered_now(route):
        answers.append((route, route.fetch()))

    page.route("**/attachments?limit=500", answered_now)
    row(page, 0).get_by_test_id("dataset-row-add-document").click()
    wait_held(page, answers)
    newer = upload_document(page, bench_url, "newer.txt", "newer " + uuid.uuid4().hex)
    row(page, 1).get_by_test_id("dataset-row-add-document").click()
    wait_held(page, answers, 2)
    (older_route, older), (newer_route, newest) = answers
    # PRE-STATE: the two answers really differ in the new document.
    assert newer not in older.text()
    assert newer in newest.text()
    option = row(page, 1).locator(
        f"[data-testid=dataset-document-option][data-digest='{newer}']"
    )

    newer_route.fulfill(response=newest)
    expect(option).to_have_count(1)
    with page.expect_response(lambda r: r.url == older_route.request.url) as info:
        older_route.fulfill(response=older)
    info.value.finished()
    page.evaluate("() => new Promise((done) => setTimeout(done, 50))")
    page.unroute("**/attachments?limit=500")
    option.click()

    expect(option).to_have_attribute("aria-pressed", "true")


MEASURE_ALL = """
() => {
  const parse = (c) => c.match(/[\\d.]+/g).map(Number);
  const out = [];
  for (const el of document.querySelectorAll('#datasets *')) {
    const own = Array.from(el.childNodes).some(
      (n) => n.nodeType === 3 && n.textContent.trim() !== '');
    if (!own || el.getClientRects().length === 0) continue;
    if (el.disabled || el.closest('select')) continue;
    const fg = parse(getComputedStyle(el).color);
    const layers = [];
    for (let n = el; n; n = n.parentElement) {
      const bg = parse(getComputedStyle(n).backgroundColor);
      const alpha = bg.length === 4 ? bg[3] : 1;
      if (alpha > 0) {
        layers.push([bg[0], bg[1], bg[2], alpha]);
        if (alpha === 1) break;
      }
    }
    const name = (el.dataset.testid || el.id || el.className || el.tagName);
    out.push({ name: String(name), fg: fg, layers: layers });
  }
  return out;
}
"""


def swept(page):
    """The contrast of every visible, enabled element in the Datasets
    panel that holds text of its own, against what is behind it, polled
    until two readings agree. Disabled controls are exempt under WCAG's
    inactive-component clause, as in test_contrast.py."""
    last = None
    for _ in range(40):
        ratios = []
        for item in page.evaluate(MEASURE_ALL):
            behind = item["layers"][-1][:3]
            for layer in reversed(item["layers"][:-1]):
                behind = composite(layer, behind)
            fg = item["fg"]
            if len(fg) == 4:
                fg = composite(fg, behind)
            ratios.append((item["name"], round(contrast(fg, behind), 2)))
        if ratios and ratios == last:
            return ratios
        last = ratios
        page.wait_for_timeout(50)
    raise AssertionError(f"the panel never settled: {last}")


def token(page, name):
    return page.evaluate(
        "(t) => { const p = document.createElement('div');"
        " document.body.append(p); p.style.color = 'var(' + t + ')';"
        " const c = getComputedStyle(p).color; p.remove(); return c; }",
        name,
    )


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_builders_text_meets_aa_where_it_sits(page, bench, bench_url, theme):
    """WINDOW: computed colours on the live page, in each theme, for every
    visible, enabled element in the Datasets panel that holds text of its
    own, with the builder in a state that shows most of what it draws: a
    judge row's captions, a cited document's chip and its x, an open
    picker with a pressed option, a refused row, the panel's refusal, the
    "Stored" label, the ceilings, and a selected library entry.

    test_contrast.py checks tokens in pairs; this checks the elements, so
    a rule that put a token back on a background it fails on (the accent
    tint under a selected entry, --text-faint on a chip) is caught where
    it lands, whichever element it lands on. Named selectors are asserted
    present so the sweep cannot pass by finding nothing. The selected
    entry keeps its accent bar, and a refusal keeps --err: the row's
    border, the row's sentence and the panel's."""
    digest = upload_document(page, bench_url, "aa.txt", "aa " + uuid.uuid4().hex)
    name = unique("aa")
    assert page.request.post(
        bench_url + "/datasets",
        data={
            "name": name,
            "content": json.dumps({"id": "aa", "prompt": uuid.uuid4().hex}) + "\n",
        },
    ).ok
    bench(["stub/fast"])
    page.evaluate("t => { document.documentElement.dataset.theme = t }", theme)
    open_datasets(page)
    page.get_by_test_id("dataset-name").fill(unique("aa refused"))
    add_task(page, "dup", "one", "judge", rubric="r")
    add_task(page, "dup", "two")
    row(page, 0).get_by_test_id("dataset-row-add-document").click()
    row(page, 0).locator(
        f"[data-testid=dataset-document-option][data-digest='{digest}']"
    ).click()
    entry_for(page, name).click()
    store(page)
    page.mouse.move(0, 0)

    for selector in [
        ".ds-cap-text",
        "[data-testid=dataset-row-document]",
        "[data-testid=dataset-row-document-remove]",
        "[data-testid=dataset-document-option][aria-pressed=true]",
        "#dataset-library-head .panel-label",
        ".ds-entry[aria-pressed=true] .htime",
        ".ds-entry[aria-pressed=true] .hcount",
        "[data-testid=dataset-row-msg][data-state=refused]",
        "[data-testid=dataset-ceilings]",
    ]:
        expect(page.locator(selector).first).to_be_visible()
    failures = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]
    assert not failures, "below WCAG AA 4.5:1:\n" + "\n".join(failures)

    accent, err = token(page, "--accent"), token(page, "--err")
    shadow = entry_for(page, name).evaluate("el => getComputedStyle(el).boxShadow")
    assert shadow == accent + " 2px 0px 0px 0px inset", (shadow, accent)
    refused = row(page, 1)
    expect(refused).to_have_attribute("data-state", "refused")
    border = refused.evaluate(
        "el => { const s = getComputedStyle(el);"
        " return [s.borderTopColor, s.borderTopStyle, s.borderTopWidth]; }"
    )
    assert border == [err, "solid", "1px"]
    for sentence in (
        refused.get_by_test_id("dataset-row-msg"),
        page.get_by_test_id("dataset-msg"),
    ):
        assert sentence.evaluate("el => getComputedStyle(el).color") == err
