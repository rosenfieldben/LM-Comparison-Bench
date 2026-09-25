"""Phase O browser tests, O3: the Clone step in the snapshot panel.

A public repository URL and a ref, a Clone button, and on success the
root box filled with the directory the clone door made, the outcome and
the head beside it; the door's refusals word for word, none of them
holding the URL; nothing remembered across loads; blank never sent; a
clone that records nothing about the lineup; and the dataset builder's
snapshot option titled with the latest walk of its bytes.

The remote is tests/clone_stub.py (a smart-HTTP git remote over TLS on
loopback), and the bench that clones from it is conftest's clone bench;
the snapshot bench stays the one whose clone door is off. Requests are
held with page.route, never with the stub's stall, whose events are the
session's. Every proof names its window.
"""

import hashlib
import json

import pytest
from playwright.sync_api import expect
from test_i4 import check_all_chips
from test_n import (
    ABORTED_RESOURCE,
    REFUSED_RESOURCE,
    add_task,
    arm,
    open_datasets,
    row,
    store,
    unique,
)
from test_n3 import DONE_TIMEOUT, entry_for, open_experiments
from test_o import FORBIDDEN_RESOURCE

from bench.clones import CLONES_OFF

pytestmark = pytest.mark.browser

OWNER = "zq-owner"
# A URL no sentence of the bench's could hold a five-character run of,
# for the refusal that must not repeat it: its user, its token and its
# path share no run with the userinfo refusal ("a user or a token").
USER, TOKEN = "Uq8vkp", "Zx4Q9wLm2Tr7Vb3Nc6Yh"
PATH_OWNER, PATH_REPO = "Qx7own", "Rz4rep0"
SILENT = "The model catalog did not say whether cloning is configured on this bench."
CLONING = "cloning; List and Compose wait for its answer"
EARLIER = "an earlier clone is still running; Clone waits for it"
BLIND = (
    "Cloning waits until this blind rating is left: a clone fills the root "
    "box with a path, and the blind view shows none."
)


@pytest.fixture(autouse=True)
def collectors(page):
    """Console errors and CSP violations armed before first paint and
    asserted after; yields the prefixes a test allows."""
    errors = arm(page)
    allowed = [REFUSED_RESOURCE]
    yield allowed
    assert page.evaluate("window.__csp || []") == []
    assert [e for e in errors if not e.startswith(tuple(allowed))] == []


def repo(request, remote, files, **kw):
    """A repository of this test's own on the remote, and its head."""
    name = "r" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:10]
    return name, remote.repository(OWNER, name, files, **kw)


def open_panel(page):
    page.get_by_test_id("snapshot-open").click()
    expect(page.get_by_test_id("snapshot-panel")).to_be_visible()


def press_clone(page, url, ref="main"):
    """Type the URL and the ref, press Clone, and the answer."""
    page.get_by_test_id("clone-url").fill(url)
    page.get_by_test_id("clone-ref").fill(ref)
    with page.expect_response(lambda r: r.url.endswith("/clones")) as answer:
        page.get_by_test_id("clone-run").click()
    expect(page.get_by_test_id("clone-run")).to_be_enabled()
    return answer.value


def press(page, testid, path):
    with page.expect_response(lambda r: r.url.endswith(path)) as answer:
        page.get_by_test_id(testid).click()
    return answer.value


def outer(page):
    """The page as markup: every element's text and every attribute."""
    return page.evaluate("document.documentElement.outerHTML")


def held_route(page, pattern):
    held = []
    page.route(pattern, lambda route: held.append(route))
    return held


def wait_held(page, held):
    for _ in range(200):
        if held:
            return
        page.wait_for_timeout(25)
    raise AssertionError("the request was never made")


def blind_group(page, url):
    """A comparison the blind view can open over, on this bench."""
    api = page.request
    models = ["stub/fast", "stub/html"]
    group = api.post(
        url + "/groups",
        data={"prompt": "blind over a clone", "models": models, "budget": "standard"},
    ).json()["id"]
    assert api.post(
        url + "/compare",
        data={"prompt": "blind over a clone", "models": models, "group_id": group},
    ).ok
    return group


# ----- off: the server's sentence, and only inside the open panel -------


PATHS = "/Zq9/clone-root and /Zq9/repo-root"


@pytest.mark.parametrize("posture", ["off", "naming paths", "not said", "empty"])
def test_the_clone_step_says_why_it_cannot_clone_only_while_open(
    page, snapshot_bench, snapshot_bench_url, posture
):
    """WINDOW: the Clone step on the snapshot bench (snapshots on, the
    clone door off), with GET /models as the server sends it, with its
    clone reason replaced by a sentence naming paths (the door's other
    off sentence does), with no clone fields at all, and with an empty
    reason; for the path sentence, the blind view as well.

    Open, the step shows the reason verbatim (CLONES_OFF; the path
    sentence; for a catalog that said nothing, that it said nothing; for
    an empty reason, the page's own, so an empty string never switches
    the step on) and Clone and both boxes are disabled. Closed, by the
    toggle or by a forget, the reason is gone from the page, text and
    attributes alike, since a hidden element's text is still on it. In
    the blind view the step says the blind view's sentence and no path.
    No clone is ever sent. PRE-STATE: before the panel opens, the reason
    line is empty."""
    if posture != "off":

        def rewrite(route):
            answer = route.fetch()
            body = answer.json()
            if posture == "naming paths":
                body["clones_off_reason"] = "BENCH_CLONE_ROOT is " + PATHS + "."
            elif posture == "empty":
                body["clones_off_reason"] = ""
            else:
                del body["clones_enabled"], body["clones_off_reason"]
            route.fulfill(response=answer, json=body)

        page.route("**/models", rewrite)
    sent = []
    page.on("request", lambda r: sent.append(r.url) if "/clones" in r.url else None)
    snapshot_bench(["stub/fast"])
    expected = {
        "off": CLONES_OFF,
        "naming paths": "BENCH_CLONE_ROOT is " + PATHS + ".",
        "not said": SILENT,
        "empty": "Cloning is not configured on this bench.",
    }[posture]
    reason = page.get_by_test_id("clone-reason")
    expect(page.get_by_test_id("snapshot-open")).to_be_enabled()
    assert reason.text_content() == ""

    open_panel(page)
    expect(reason).to_have_text(expected)
    for testid in ("clone-run", "clone-url", "clone-ref"):
        expect(page.get_by_test_id(testid)).to_be_disabled()

    page.get_by_test_id("snapshot-open").click()
    assert reason.text_content() == ""
    assert "Zq9" not in outer(page)
    open_panel(page)
    expect(reason).to_have_text(expected)
    page.evaluate("window.BenchAttach.forgetSnapshot()")
    expect(page.get_by_test_id("snapshot-panel")).to_be_hidden()
    assert reason.text_content() == ""
    assert "Zq9" not in outer(page)
    if posture == "naming paths":
        group = blind_group(page, snapshot_bench_url)
        page.evaluate("id => window.BenchRating.startBlind(id)", group)
        expect(page.get_by_test_id("rating-panel")).to_be_visible()
        open_panel(page)
        expect(reason).to_have_text(BLIND)
        assert "Zq9" not in outer(page)
    assert sent == []


# ----- the full path ------------------------------------------------------


def test_a_public_repository_goes_from_a_url_to_an_experiment(
    request, page, clone_bench, clone_bench_url, clone_remote, clone_runs
):
    """WINDOW: the commission's full path through the page on the clone
    bench: clone a repository from the stub remote, List, narrow by a
    check, Compose, cite the snapshot from a dataset task, Store, create
    and start an experiment over it, done.

    Along it: opening the panel with cloning on puts focus in the URL
    box; the clone is exactly one POST /clones whose body is the URL and
    the ref and nothing else, stages nothing and touches no storage (a
    clone is not a comparison); the outcome is "cloned at" and the head,
    the full head in its title, and the root box holds the answer's root,
    which is on the page as that value and nowhere else; more patterns
    than a request may carry name MAX_PATTERNS and send nothing; the
    composed chip names the walk (capture and head); Compose keeps the
    URL and ref and forgets the root and the outcome; the builder's
    option names the latest walk. PRE-STATE: nothing is staged, and the
    remote's head is the sha the page must show."""
    name, sha = repo(
        request,
        clone_remote,
        {
            "src/a.py": b"A = 'full path'\n",
            "src/b.py": b"B = 2\n",
            "README.md": b"# r\n",
        },
    )
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast"])
    check_all_chips(page)
    expect(page.get_by_test_id("attachment-chip")).to_have_count(0)
    open_panel(page)
    expect(page.get_by_test_id("clone-url")).to_be_focused()
    storage = (
        "JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage)])"
    )
    stored_before = page.evaluate(storage)
    sent = []
    page.on("request", lambda r: sent.append(r) if r.method != "GET" else None)

    answer = press_clone(page, url)

    assert answer.status == 201
    made = answer.json()
    assert [(r.method, r.url.rsplit("/", 1)[-1]) for r in sent] == [("POST", "clones")]
    assert json.loads(sent[0].post_data) == {"url": url, "ref": "main"}
    assert page.evaluate(storage) == stored_before
    expect(page.get_by_test_id("attachment-chip")).to_have_count(0)
    outcome = page.get_by_test_id("clone-outcome")
    expect(outcome).to_have_text("cloned at " + sha[:7])
    expect(outcome).to_have_attribute("title", "head " + sha)
    root = page.get_by_test_id("snapshot-root")
    expect(root).to_have_value(made["root"])
    markup = outer(page)
    assert made["root"] not in markup
    assert made["root"].rsplit("/", 1)[-1] not in markup

    listings = []
    page.on(
        "request",
        lambda r: listings.append(r) if r.url.endswith("/snapshots/listing") else None,
    )
    patterns = page.get_by_test_id("snapshot-patterns")
    patterns.fill("\n".join(f"p{n}.py" for n in range(21)))
    page.get_by_test_id("snapshot-list").click()
    expect(page.get_by_test_id("snapshot-msg")).to_contain_text("MAX_PATTERNS")
    page.wait_for_timeout(100)
    assert listings == []

    patterns.fill("**/*.py")
    assert press(page, "snapshot-list", "/snapshots/listing").ok
    page.get_by_role("checkbox", name=json.dumps("src/a.py"), exact=True).check()
    expect(patterns).to_have_value("src/a.py")
    composed = press(page, "snapshot-compose", "/snapshots").json()
    capture = composed["capture"]
    walk = f"capture #{capture['id']} at {sha[:7]}, clean"
    chip = page.get_by_test_id("attachment-chip")
    expect(chip).to_have_count(1)
    assert walk in chip.get_attribute("title")
    expect(root).to_have_value("")
    expect(page.get_by_test_id("clone-url")).to_have_value(url)
    expect(page.get_by_test_id("clone-ref")).to_have_value("main")
    expect(outcome).to_have_text("")
    assert made["root"] not in outer(page)

    open_datasets(page)
    dataset = unique("cloned")
    page.get_by_test_id("dataset-name").fill(dataset)
    add_task(page, "c1", "what does the clone define?")
    row(page, 0).get_by_test_id("dataset-row-add-document").click()
    option = row(page, 0).locator(
        f"[data-testid=dataset-document-option][data-digest='{composed['digest']}']"
    )
    expect(option).to_have_count(1)
    assert "\nlatest " + walk in option.get_attribute("title")
    option.click()
    assert store(page).startswith("stored as " + dataset)
    entry_for(page, dataset).click()
    open_experiments(page)
    page.get_by_test_id("experiment-name").fill(unique("cloned run"))
    with page.expect_response(
        lambda r: r.url.endswith("/experiments") and r.request.method == "POST"
    ) as created:
        page.get_by_test_id("experiment-create-button").click()
    clone_runs.append(created.value.json()["id"])
    start = page.get_by_test_id("experiment-start")
    expect(start).to_be_enabled()
    start.click()
    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "done", timeout=DONE_TIMEOUT
    )
    assert made["root"] not in outer(page)


# ----- an update, the table it replaces, and two heads ---------------------


def test_an_update_forgets_the_old_listing_and_the_picker_tells_heads_apart(
    request, page, clone_bench, clone_remote
):
    """WINDOW: one repository cloned at a first head and composed, cloned
    again after upstream moved, and composed again; the builder's picker
    over what that made.

    The second Clone is "updated to" the new head in the same root, and
    a table listed for the old tree is gone (a value set from script
    fires no input event, so only the step can clear it). The picker
    names each option's LATEST walk: the .py snapshot of each head at its
    own head, and the .md snapshot, whose bytes did not change and so is
    one option, at the second head and not the first. PRE-STATE: the
    table has rows before the update, and the heads differ."""
    name, first = repo(
        request, clone_remote, {"a.py": b"A = 'two heads'\n", "b.md": b"# two heads\n"}
    )
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast"])
    open_panel(page)
    made = press_clone(page, url).json()
    patterns = page.get_by_test_id("snapshot-patterns")

    def compose(pattern):
        """Compose one pattern from the clone, cloning again first when a
        Compose has forgotten the root (it keeps the URL and the ref)."""
        if page.get_by_test_id("snapshot-panel").is_hidden():
            open_panel(page)
        if page.get_by_test_id("snapshot-root").input_value() == "":
            assert press_clone(page, url).ok
        patterns.fill(pattern)
        return press(page, "snapshot-compose", "/snapshots").json()

    d1 = compose("*.py")
    e1 = compose("*.md")
    open_panel(page)
    assert press_clone(page, url).ok
    patterns.fill("*.py")
    assert press(page, "snapshot-list", "/snapshots/listing").ok
    member_rows = page.get_by_test_id("snapshot-member")
    assert member_rows.count() > 0

    second = clone_remote.repository(
        OWNER, name, {"a.py": b"A = 'moved'\n", "b.md": b"# two heads\n"}
    )
    assert second != first
    again = press_clone(page, url)
    assert again.status == 200
    expect(page.get_by_test_id("clone-outcome")).to_have_text(
        "updated to " + second[:7]
    )
    expect(page.get_by_test_id("snapshot-root")).to_have_value(made["root"])
    expect(member_rows).to_have_count(0)
    assert page.get_by_test_id("snapshot-listing-summary").text_content() == ""

    d2 = compose("*.py")
    e2 = compose("*.md")
    assert e1["digest"] == e2["digest"] and d1["digest"] != d2["digest"]

    open_datasets(page)
    add_task(page, "h1", "two heads")
    row(page, 0).get_by_test_id("dataset-row-add-document").click()

    def title(digest):
        options = row(page, 0).locator(
            f"[data-testid=dataset-document-option][data-digest='{digest}']"
        )
        expect(options).to_have_count(1)
        return options.get_attribute("title")

    assert f"at {first[:7]}, clean" in title(d1["digest"])
    assert f"at {second[:7]}, clean" in title(d2["digest"])
    assert first[:7] not in title(d2["digest"])
    assert f"latest capture #{e2['capture']['id']} at {second[:7]}" in title(
        e1["digest"]
    )
    assert first[:7] not in title(e1["digest"])


# ----- refusals: the server's words, the root kept, nothing repeated --------


def test_a_refusal_is_the_servers_sentence_and_repeats_nothing_typed(
    request, page, clone_bench, clone_bench_url, clone_remote, collectors
):
    """WINDOW: after a good clone, a Clone of a URL carrying a user and a
    token (403), one of an unknown ref (422), one of an over-long URL (the
    request model's 422), and a 201 the page cannot read; then the page's
    every surface, and a navigation away and Back.

    Each refusal is the server's sentence word for word, and the root box
    keeps the good clone's root while the outcome goes. No five-character
    run of the typed user, token, path or host is anywhere but the URL
    box: not the markup (text and every attribute), not another box, the
    title, the location, storage, cookies, any console message or any
    request URL (the host, digits a port can share runs with, whole);
    exactly one request carries the token, in its body.
    After Back, the four boxes are empty, and each carries
    autocomplete=off and spellcheck=false. PRE-STATE: the good clone
    filled the root, and the scanner finds the token in the URL box."""
    collectors.append(FORBIDDEN_RESOURCE)
    name, _ = repo(request, clone_remote, {"a.py": b"A = 'refusals'\n"})
    good = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast"])
    console = []
    page.on("console", lambda m: console.append(m.text))
    urls = []
    page.on("request", lambda r: urls.append((r.url, r.post_data or "")))
    open_panel(page)
    made = press_clone(page, good).json()
    root = page.get_by_test_id("snapshot-root")
    message = page.get_by_test_id("snapshot-msg")
    outcome = page.get_by_test_id("clone-outcome")
    expect(root).to_have_value(made["root"])

    host = clone_remote.host
    typed = f"https://{USER}:{TOKEN}@{host}/{PATH_OWNER}/{PATH_REPO}"
    # The distinctive parts by every five-character run of them; the host,
    # which is digits and dots a port can share runs with, whole.
    secrets = [USER, TOKEN, PATH_OWNER, PATH_REPO]

    def runs(text):
        found = [
            s[i : i + 5]
            for s in secrets
            for i in range(len(s) - 4)
            if s[i : i + 5] in text
        ]
        return found + ([host] if host in text else [])

    refused = press_clone(page, typed)
    assert refused.status == 403
    expect(message).to_have_text(refused.json()["detail"])
    assert page.get_by_test_id("clone-url").input_value() == typed
    assert runs(page.get_by_test_id("clone-url").input_value())
    expect(root).to_have_value(made["root"])
    expect(outcome).to_have_text("")
    assert outcome.get_attribute("title") in ("", None)
    others = page.evaluate(
        "[...document.querySelectorAll('input, textarea, select')]"
        ".filter(el => el.id !== 'clone-url').map(el => el.value).join('\\n')"
    )
    surfaces = [
        outer(page),
        others,
        page.title(),
        page.url,
        page.evaluate(
            "JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage)])"
        ),
        page.evaluate("document.cookie"),
        "\n".join(console),
        "\n".join(u for u, _ in urls),
    ]
    for surface in surfaces:
        assert runs(surface) == [], surface[:200]
    assert [u for u, body in urls if TOKEN in body] == [clone_bench_url + "/clones"]

    unknown = press_clone(page, good, "no-such-ref")
    assert unknown.status == 422
    expect(message).to_have_text(unknown.json()["detail"])
    expect(root).to_have_value(made["root"])

    long = press_clone(page, good + "x" * 2049)
    assert long.status == 422
    expect(message).to_have_text("; ".join(e["msg"] for e in long.json()["detail"]))
    expect(root).to_have_value(made["root"])

    page.route(
        "**/clones",
        lambda route: route.fulfill(
            status=201, json={"outcome": "cloned", "head_sha": "0" * 40}
        ),
    )
    press_clone(page, good)
    expect(message).to_have_text(
        "the bench answered 201 with a clone this page could not read, so the "
        "root box keeps what it held."
    )
    expect(root).to_have_value(made["root"])
    page.unroute("**/clones")
    # A 2xx that carries a root but no outcome the step can name: still
    # unread, and its root is not written.
    page.route(
        "**/clones",
        lambda route: route.fulfill(
            status=201,
            json={"outcome": "copied", "head_sha": "0" * 40, "root": "/Zq9/elsewhere"},
        ),
    )
    press_clone(page, good)
    expect(root).to_have_value(made["root"])
    page.unroute("**/clones")
    # A snapshot composed after a refused URL keeps no URL: the URL and
    # ref are kept only when they made the root, and this one holds a
    # token.
    assert press_clone(page, typed).status == 403
    page.get_by_test_id("snapshot-patterns").fill("*.py")
    assert press(page, "snapshot-compose", "/snapshots").ok
    for testid in ("clone-url", "clone-ref", "snapshot-root"):
        assert page.get_by_test_id(testid).input_value() == "", testid
    open_panel(page)

    for testid in ("clone-url", "clone-ref", "snapshot-root", "snapshot-patterns"):
        box = page.get_by_test_id(testid)
        expect(box).to_have_attribute("autocomplete", "off")
        expect(box).to_have_attribute("spellcheck", "false")
    page.get_by_test_id("clone-url").fill(typed)
    page.get_by_test_id("snapshot-patterns").fill("src/secret.py")
    page.goto(clone_bench_url + "/models")
    page.go_back()
    expect(page.get_by_test_id("snapshot-open")).to_be_enabled()
    for testid in ("clone-url", "clone-ref", "snapshot-root", "snapshot-patterns"):
        assert page.get_by_test_id(testid).input_value() == "", testid


def test_blank_is_not_sent(request, page, clone_bench, clone_remote):
    """WINDOW: the Clone button with the URL or the ref blank, or only
    whitespace.

    Each says what is missing, URL first, and sends nothing. PRE-STATE:
    the counter sees a clone that is sent."""
    name, _ = repo(request, clone_remote, {"a.py": b"A = 'blank'\n"})
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast"])
    sent = []
    page.on(
        "request", lambda r: sent.append(r.url) if r.url.endswith("/clones") else None
    )
    open_panel(page)
    assert press_clone(page, url).ok
    assert len(sent) == 1
    missing_url = (
        "Name the public repository to clone: an https URL of a host this bench lists."
    )
    missing_ref = "Name the branch, tag or 40-character commit to clone."
    for typed_url, typed_ref, said in (
        ("", "main", missing_url),
        ("   ", "main", missing_url),
        (url, "", missing_ref),
        (url, "  \t ", missing_ref),
    ):
        page.get_by_test_id("clone-url").fill(typed_url)
        page.get_by_test_id("clone-ref").fill(typed_ref)
        page.get_by_test_id("clone-run").click()
        expect(page.get_by_test_id("snapshot-msg")).to_have_text(said)
    page.wait_for_timeout(200)
    assert len(sent) == 1


# ----- an answer that lands after the panel moved on ------------------------


def forget(page, how, group):
    if how == "blind view":
        page.evaluate("id => window.BenchRating.startBlind(id)", group)
        expect(page.get_by_test_id("rating-panel")).to_be_visible()
    elif how == "clear":
        page.evaluate("window.BenchAttach.clear()")
    else:
        page.evaluate("window.BenchAttach.setFrom([], 'inline')")


@pytest.mark.parametrize("how", ["blind view", "clear", "reuse"])
def test_a_clone_answered_after_the_panel_forgot_fills_nothing(
    request, page, clone_bench, clone_bench_url, clone_remote, how
):
    """WINDOW: a Clone request held while the panel forgets, by the blind
    view opening, a clear or a reuse, then released; and, after the blind
    view, the view moving on.

    While it is held the step says it is cloning, Clone, List and Compose
    wait, and the root, URL and ref boxes are read-only. Once the panel
    forgets, the URL and ref are gone (the URL can carry a token), and
    the forgotten clone holds only Clone, saying so: List and Compose are
    free and the root box is writable. Its answer is a real 201 for a
    root the panel has forgotten, so it fills nothing: no root, no
    outcome, no message, and its root is nowhere on the page. In the
    blind view cloning waits, saying why, until the view moves on, when
    the step is itself again without being reopened. PRE-STATE: the
    request was made and is held, with the URL and ref in their boxes."""
    name, _ = repo(request, clone_remote, {"a.py": b"A = 'late'\n"})
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast", "stub/html"])
    group = blind_group(page, clone_bench_url) if how == "blind view" else None
    held = held_route(page, "**/clones")
    open_panel(page)
    page.get_by_test_id("clone-url").fill(url)
    page.get_by_test_id("clone-ref").fill("main")
    page.get_by_test_id("clone-run").click()
    wait_held(page, held)
    expect(page.get_by_test_id("clone-outcome")).to_have_text(CLONING)
    for testid in ("clone-run", "snapshot-list", "snapshot-compose"):
        expect(page.get_by_test_id(testid)).to_be_disabled()
    for testid in ("clone-url", "clone-ref", "snapshot-root"):
        assert page.get_by_test_id(testid).evaluate("el => el.readOnly")

    forget(page, how, group)
    open_panel(page)
    for testid in ("clone-url", "clone-ref"):
        expect(page.get_by_test_id(testid)).to_have_value("")
    expect(page.get_by_test_id("clone-outcome")).to_have_text(EARLIER)
    expect(page.get_by_test_id("clone-run")).to_be_disabled()
    expect(page.get_by_test_id("snapshot-list")).to_be_enabled()
    assert not page.get_by_test_id("snapshot-root").evaluate("el => el.readOnly")
    with page.expect_response(lambda r: r.url.endswith("/clones")) as late:
        held.pop().continue_()
    assert late.value.status == 201
    late_root = late.value.json()["root"]
    page.wait_for_timeout(100)

    expect(page.get_by_test_id("snapshot-root")).to_have_value("")
    expect(page.get_by_test_id("clone-outcome")).to_have_text("")
    expect(page.get_by_test_id("snapshot-msg")).to_have_text("")
    assert late_root not in outer(page)
    if how == "blind view":
        expect(page.get_by_test_id("clone-reason")).to_have_text(BLIND)
        expect(page.get_by_test_id("clone-run")).to_be_disabled()
        page.evaluate("id => window.BenchHistory.showGroup(id)", group)
        expect(page.get_by_test_id("clone-reason")).to_have_text("")
        expect(page.get_by_test_id("clone-run")).to_be_enabled()
    else:
        expect(page.get_by_test_id("clone-run")).to_be_enabled()


def test_a_forgotten_clone_that_fails_says_nothing(
    request, page, clone_bench, clone_remote, collectors
):
    """WINDOW: a Clone request held, the panel cleared, and the request
    then failing at the network.

    The failure belongs to a clone the panel forgot, so the message line
    (outside the panel, restated after a blind view) says nothing of it.
    PRE-STATE: the request was made and is held."""
    collectors.append(ABORTED_RESOURCE)
    name, _ = repo(request, clone_remote, {"a.py": b"A = 'aborted'\n"})
    page = clone_bench(["stub/fast"])
    held = held_route(page, "**/clones")
    open_panel(page)
    page.get_by_test_id("clone-url").fill(clone_remote.url(OWNER, name))
    page.get_by_test_id("clone-ref").fill("main")
    page.get_by_test_id("clone-run").click()
    wait_held(page, held)

    page.evaluate("window.BenchAttach.clear()")
    held.pop().abort()
    expect(page.get_by_test_id("clone-run")).to_be_enabled()

    expect(page.get_by_test_id("snapshot-msg")).to_have_text("")


def test_a_clone_answered_after_another_view_still_fills_the_root(
    request, page, clone_bench, clone_bench_url, clone_remote
):
    """WINDOW: a Clone request held while a history entry is opened (a
    view that moves the epoch without forgetting the panel, as Run and a
    run do), then released.

    The panel and its root are still there, so the answer fills the root
    and the outcome is said: only a forget drops a clone's answer.
    PRE-STATE: the request was made and is held."""
    name, sha = repo(request, clone_remote, {"a.py": b"A = 'epoch'\n"})
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast", "stub/html"])
    group = blind_group(page, clone_bench_url)
    held = held_route(page, "**/clones")
    open_panel(page)
    page.get_by_test_id("clone-url").fill(url)
    page.get_by_test_id("clone-ref").fill("main")
    page.get_by_test_id("clone-run").click()
    wait_held(page, held)

    page.evaluate("id => window.BenchHistory.showGroup(id)", group)
    with page.expect_response(lambda r: r.url.endswith("/clones")) as late:
        held.pop().continue_()
    made = late.value.json()

    expect(page.get_by_test_id("snapshot-root")).to_have_value(made["root"])
    expect(page.get_by_test_id("clone-outcome")).to_have_text("cloned at " + sha[:7])


def test_the_outcome_goes_when_it_stops_describing_the_boxes(
    request, page, clone_bench, clone_remote, collectors
):
    """WINDOW: the outcome beside Clone after a clone, as each of the
    three boxes it describes is edited; as a new clone starts (held), and
    lands while the panel is closed; as a clone is refused; and the panel
    reopened with a root held.

    Typing in the URL, the ref or the root empties it. A running clone
    replaces it with the running line and no title. A clone that lands
    while the panel is closed is said outside the panel too, since the
    step's own line is hidden. A refusal of a Clone pressed with nothing
    typed leaves no outcome standing (a new clone clears it). With
    a root held, opening the panel puts focus in the root box.
    PRE-STATE: each edit follows a clone whose outcome is shown."""
    collectors.append(FORBIDDEN_RESOURCE)
    name, sha = repo(request, clone_remote, {"a.py": b"A = 'outcome'\n"})
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast"])
    open_panel(page)
    outcome = page.get_by_test_id("clone-outcome")
    for testid in ("clone-url", "clone-ref", "snapshot-root"):
        assert press_clone(page, url).ok
        expect(outcome).not_to_have_text("")
        box = page.get_by_test_id(testid)
        box.fill(box.input_value() + " ")
        expect(outcome).to_have_text("")

    assert press_clone(page, url).ok
    page.get_by_test_id("snapshot-open").click()
    open_panel(page)
    expect(page.get_by_test_id("snapshot-root")).to_be_focused()

    held = held_route(page, "**/clones")
    page.get_by_test_id("clone-run").click()
    wait_held(page, held)
    expect(outcome).to_have_text(CLONING)
    assert outcome.get_attribute("title") in ("", None)
    page.get_by_test_id("snapshot-open").click()
    expect(page.get_by_test_id("snapshot-panel")).to_be_hidden()
    with page.expect_response(lambda r: r.url.endswith("/clones")):
        held.pop().continue_()
    expect(page.get_by_test_id("snapshot-msg")).to_have_text("updated to " + sha[:7])
    page.unroute("**/clones")

    open_panel(page)
    page.route(
        "**/clones",
        lambda route: route.fulfill(
            status=403, json={"detail": "refused for the test"}
        ),
    )
    # Pressed, not typed into: typing would clear the outcome itself.
    expect(outcome).to_have_text("updated to " + sha[:7])
    with page.expect_response(lambda r: r.url.endswith("/clones")):
        page.get_by_test_id("clone-run").click()
    expect(page.get_by_test_id("snapshot-msg")).to_have_text("refused for the test")
    expect(outcome).to_have_text("")


def test_clone_waits_for_list_and_compose(request, page, clone_bench, clone_remote):
    """WINDOW: the Clone button while a listing is held, and while a
    Compose is held.

    Each answer is about the root box's tree, and a clone would replace
    it, so Clone waits for both. PRE-STATE: Clone is enabled with the
    root filled and nothing in flight."""
    name, _ = repo(request, clone_remote, {"a.py": b"A = 'waits'\n"})
    url = clone_remote.url(OWNER, name)
    page = clone_bench(["stub/fast"])
    open_panel(page)
    assert press_clone(page, url).ok
    page.get_by_test_id("snapshot-patterns").fill("*.py")
    clone_run = page.get_by_test_id("clone-run")
    expect(clone_run).to_be_enabled()
    for path, testid in (
        ("**/snapshots/listing", "snapshot-list"),
        ("**/snapshots", "snapshot-compose"),
    ):
        held = held_route(page, path)
        page.get_by_test_id(testid).click()
        wait_held(page, held)
        expect(clone_run).to_be_disabled()
        with page.expect_response(lambda r, p=path: r.url.endswith(p[2:])):
            held.pop().continue_()
        page.unroute(path)
        expect(clone_run).to_be_enabled()


def test_a_page_restored_from_the_back_forward_cache_forgets_the_step(
    request, page, clone_bench, clone_remote
):
    """WINDOW: the pageshow a back/forward-cache restore fires
    (persisted), dispatched on a page holding a URL with a token, a ref
    and a cloned root.

    A restore brings the live page back, which autocomplete cannot
    reach, so the panel forgets: the three boxes are empty. PRE-STATE:
    all three hold their values."""
    name, _ = repo(request, clone_remote, {"a.py": b"A = 'bfcache'\n"})
    page = clone_bench(["stub/fast"])
    open_panel(page)
    assert press_clone(page, clone_remote.url(OWNER, name)).ok
    page.get_by_test_id("clone-url").fill(f"https://{USER}:{TOKEN}@example/o/r")
    for testid in ("clone-url", "clone-ref", "snapshot-root"):
        assert page.get_by_test_id(testid).input_value() != ""

    page.evaluate(
        "window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true}))"
    )

    for testid in ("clone-url", "clone-ref", "snapshot-root"):
        assert page.get_by_test_id(testid).input_value() == "", testid
