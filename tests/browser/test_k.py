"""Phase K4 browser tests: the seams a person touches when attaching.

The claim under test is not "an endpoint accepts a digest" but "a person
can attach a document, see what it is and what it will cost, run the
comparison, and find the document named everywhere the comparison is
shown afterwards". So the assertions read the rendered page.

EVERY PROOF NAMES ITS WINDOW. An attachment is visible in four different
places built by four different code paths (the composer's staging area,
a history row, the replay banner, and the blind view's absence of one),
and a proof about the wrong one of those reads as coverage while leaving
the others unguarded. Each test below says which surface it exercises.
"""

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 15_000

# A one-pixel PNG, the same bytes the API suite uses, written as hex
# because a binary literal in a source file is unreadable and survives
# an editor's round trip badly.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63fcffff3f0300050001ff9a1c86"
    "9c0000000049454e44ae426082"
)

CONTRACT = b"the contract says the delivery window is forty two days"


def text_file(name="contract.txt", content=CONTRACT):
    """One picked file, in Playwright's in-memory shape.

    No temp files: set_input_files takes the bytes directly, so the test
    exercises the page's own read-and-encode path without asking the
    harness to own a file on disk.
    """
    return {"name": name, "mimeType": "text/plain", "buffer": content}


def png_file(name="shot.png"):
    return {"name": name, "mimeType": "image/png", "buffer": PNG_BYTES}


def attach(page, *files):
    page.get_by_test_id("attach-input").set_input_files(list(files))


def chips(page):
    return page.get_by_test_id("attachment-chip")


def check_only(page, model):
    """Leave exactly one model checked, so a run is one card.

    Written against the chip's checkbox rather than the chip, because a
    click on an already-checked chip would uncheck it and the seeded
    lineup's checked state is not something this file should assume.
    """
    lineup = page.get_by_test_id("lineup-chip")
    for i in range(lineup.count()):
        chip = lineup.nth(i)
        box = chip.locator("input[type=checkbox]")
        want = box.get_attribute("value") == model
        if box.is_checked() != want:
            chip.click()


def run_and_wait(page, prompt, cards=1):
    page.get_by_test_id("prompt-input").fill(prompt)
    page.get_by_test_id("run-button").click()
    for i in range(cards):
        expect(
            page.get_by_test_id("result-card").nth(i).get_by_test_id("card-status")
        ).to_have_text("done", timeout=DONE_TIMEOUT)


def row_for(page, prompt):
    return page.get_by_test_id("history-row").filter(has_text=prompt)


def test_the_attach_control_names_the_document_and_what_it_will_cost(bench):
    """WINDOW: the composer's staging area, after a pick and before any
    run. Nothing has been compared yet; this is the moment the person is
    deciding whether the right file went in.

    Name, size, short digest and a LABELED-APPROXIMATE token estimate.
    The label is the load-bearing half: characters-over-four is a
    heuristic every model's real tokenizer disagrees with, so a bare
    number here would be believed."""
    page = bench(["stub/fast"])
    attach(page, text_file())

    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    name = page.get_by_test_id("attachment-name")
    expect(name).to_have_text("contract.txt")
    meta = page.get_by_test_id("attachment-meta").inner_text()
    # Size, in the binary units the upload cap is stated in.
    assert f"{len(CONTRACT)} B" in meta
    # The short digest, git's seven characters, and the full one only in
    # the title where it cannot be mistaken for something to type.
    assert re.search(r"sha256 [0-9a-f]{7}\b", meta), meta
    assert "approx ~" in meta and "tokens" in meta
    # 55 characters over four, rounded up.
    assert f"~{-(-len(CONTRACT) // 4)} tokens" in meta
    title = chips(page).first.get_attribute("title")
    assert re.search(r"sha256 [0-9a-f]{64}", title), title
    # The extractor and its version, because a parser upgrade changes the
    # text the models read.
    assert "read by text" in title


def test_the_control_states_where_the_document_goes_and_where_it_is_kept(bench):
    """WINDOW: the note under the attach control on the DEFAULT bench,
    which is the case with no policy badge in the header.

    Surfaced at the control and not only in the header, because the badge
    is a session-wide marker a person reads once and this is the moment
    they are deciding whether to hand the bench a contract. The default
    policy is the case that most needs saying, since its header signal is
    an absent badge and silence is a poor way to tell somebody their
    document is going out under ordinary terms."""
    page = bench(["stub/fast"])

    note = page.get_by_test_id("attach-note").inner_text()

    assert "Session routing: default" in note
    assert "sent to OpenRouter" in note
    assert "BENCH_DATA_POLICY" in note
    # The storage story, at the control that creates the copy.
    assert "bench.db" in note
    assert "deleting bench.db deletes the documents" in note


def test_a_restricted_session_says_so_at_the_attach_control(zdr_bench):
    """WINDOW: the same note, on a bench booted BENCH_DATA_POLICY=zdr.

    A separate process because the policy is boot-scoped; the badge test
    one file over proves the header, and this proves the control, which
    is a different element fed from a different place."""
    page = zdr_bench(["stub/fast"])

    note = page.get_by_test_id("attach-note").inner_text()

    assert "Session routing: zero-retention" in note
    # The claim stays OpenRouter's, exactly as the badge phrases it: this
    # application cannot verify a routing constraint it only asked for.
    assert "guarantee is OpenRouter's" in note


def test_review_repro_the_cap_refusal_surfaces_in_the_composer(bench):
    """WINDOW: the composer's message line, immediately after a pick of
    more files than the cap allows and before any upload is attempted.

    THE SILENT-TRUNCATION CASE IS THE ONE THAT MATTERS. A picker that
    kept the first four of five would leave the person believing they
    attached five, and the comparison would run over a document set
    nobody chose. So nothing is attached when the batch will not fit, and
    the refusal says the arithmetic."""
    page = bench(["stub/fast"])

    attach(page, *[text_file(f"doc{i}.txt", f"body {i}".encode()) for i in range(5)])

    msg = page.get_by_test_id("attach-msg")
    expect(msg).to_contain_text("at most 4 documents", timeout=DONE_TIMEOUT)
    assert "Nothing was attached" in msg.inner_text()
    # And it is true: the staging area is empty, not holding four.
    expect(chips(page)).to_have_count(0)


def test_the_cap_counts_what_is_already_attached(bench):
    """WINDOW: the same line, on a second pick that would push the total
    over. The first pick's documents are already staged and stay so."""
    page = bench(["stub/fast"])
    attach(page, *[text_file(f"doc{i}.txt", f"body {i}".encode()) for i in range(3)])
    expect(chips(page)).to_have_count(3, timeout=DONE_TIMEOUT)

    attach(page, text_file("four.txt", b"four"), text_file("five.txt", b"five"))

    expect(page.get_by_test_id("attach-msg")).to_contain_text("3 attached and 2 more")
    expect(chips(page)).to_have_count(3)


def test_an_oversized_file_is_refused_before_it_is_uploaded(bench):
    """WINDOW: the composer's message line for a file past the byte cap,
    before any request is made.

    Checked client-side so the bytes are never base64'd and pushed across
    a socket to be refused there. The server's own refusal is the one
    that counts and is proven in the API suite; this one exists so the
    person is not made to wait for it."""
    page = bench(["stub/fast"])
    # One byte over eight MiB.
    attach(page, text_file("huge.txt", b"x" * (8 * 1024 * 1024 + 1)))

    msg = page.get_by_test_id("attach-msg")
    expect(msg).to_contain_text("over the", timeout=DONE_TIMEOUT)
    assert "8.0 MiB limit" in msg.inner_text()
    expect(chips(page)).to_have_count(0)


def test_review_repro_an_attached_comparison_shows_its_document_everywhere(
    bench, open_history
):
    """WINDOW: three surfaces in one pass, because they are three code
    paths and a proof of one is not a proof of the others.

    1. the history ROW, built by list_runs
    2. the replay BANNER, built by the group detail
    3. the composer, after reuse

    Run together rather than split because the run they all describe is
    the expensive part, and splitting it would triple a browser test's
    wall clock to assert the same three things about the same run."""
    page = bench(["stub/fast"])
    check_only(page, "stub/fast")
    attach(page, text_file())
    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    run_and_wait(page, "summarize the attached contract")

    # 1. The history row names it without being opened.
    open_history()
    row = row_for(page, "summarize the attached contract").first
    doc = row.get_by_test_id("history-attachment")
    expect(doc).to_have_count(1)
    expect(doc).to_have_text("contract.txt")
    # The mode rides the title, because the row is one line already.
    assert "inline" in doc.get_attribute("title")

    # 2. The replay banner names it, with the mode spelled out in words:
    # a replay has room to say what was measured.
    row.click()
    expect(page.get_by_test_id("run-label")).to_contain_text(
        "Historical", timeout=DONE_TIMEOUT
    )
    strip = page.get_by_test_id("run-attachments")
    expect(strip).to_be_visible()
    assert "the bench extracted the text" in strip.inner_text()
    expect(page.get_by_test_id("run-attachment")).to_have_text("contract.txt")

    # 3. Reuse carries it into the composer, because the prompt the
    # models read was the composition of the text WITH this document.
    page.get_by_test_id("reuse-comparison").click()
    expect(chips(page)).to_have_count(1)
    expect(page.get_by_test_id("attachment-name")).to_have_text("contract.txt")
    expect(page.get_by_test_id("attach-mode")).to_have_value("inline")


def test_reuse_clears_a_document_the_source_did_not_have(bench, open_history):
    """WINDOW: the composer after reusing an UNATTACHED comparison while
    a document is staged.

    The mirror of the case above, and the one that decides whether reuse
    reproduces an experiment or merely edits the prompt. A file left over
    from earlier would quietly make the new run a different comparison
    from the one being reused, which is the same defect
    setExperimentParams already guards for controls."""
    page = bench(["stub/fast"])
    check_only(page, "stub/fast")
    run_and_wait(page, "a plain question with no document")
    attach(page, text_file("leftover.txt", b"left over from earlier"))
    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)

    open_history()
    row_for(page, "a plain question with no document").first.click()
    expect(page.get_by_test_id("reuse-comparison")).to_be_visible(timeout=DONE_TIMEOUT)
    page.get_by_test_id("reuse-comparison").click()

    expect(chips(page)).to_have_count(0)
    expect(page.get_by_test_id("run-attachments")).to_have_count(0)


def test_review_repro_the_blind_view_never_shows_a_document_name(bench, open_history):
    """WINDOW: the results area and the run-controls area during a blind
    rating session opened from the history list, over a comparison that
    ran with a named document.

    The whole rendered region is read rather than a single element,
    because the claim is an absence and an absence proven of one node
    says nothing about its siblings. The composer is deliberately outside
    this window: it holds what the person is staging now, which is not a
    fact about the comparison being rated.

    A filename could not identify WHICH model wrote an answer anyway,
    since every model in a comparison reads the same documents. The rule
    is narrower than the blindness it sits inside, and it is kept true at
    the source: the blind payload has no attachment field at all, proven
    in the API suite."""
    page = bench(["stub/fast", "stub/slow"])
    check_only(page, "stub/fast")
    page.get_by_test_id("lineup-chip").filter(
        has=page.locator("input[value='stub/slow']")
    ).click()
    attach(page, text_file("secret-contract.txt", CONTRACT))
    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    run_and_wait(page, "rate the answers about this document", cards=2)

    open_history()
    row_for(page, "rate the answers about this document").first
    page.get_by_test_id("history-rate-blind").first.click()
    expect(page.get_by_test_id("rating-panel")).to_be_visible(timeout=DONE_TIMEOUT)

    # inner_text, not the markup: hidden nodes are excluded, which is the
    # question being asked. Checking the HTML would find a name in a node
    # nobody can see and call the view broken.
    # #results by id: the cards container carries no testid of its own,
    # only the cards inside it do, and the question here is about the
    # whole region rather than any one card.
    shown = (
        page.locator("#results").inner_text()
        + "\n"
        + page.get_by_test_id("run-controls").inner_text()
    )
    assert "secret-contract" not in shown
    assert "sha256" not in shown
    expect(page.get_by_test_id("run-attachments")).to_have_count(0)
    expect(page.get_by_test_id("run-attachment")).to_have_count(0)


def test_review_repro_native_over_a_non_image_is_refused_in_the_composer(bench):
    """WINDOW: the composer's message line and the Run button, after
    switching a staged text file to native mode. No request has been
    made.

    Said here first so the person is not told at Run time about a file
    choice they made a minute earlier. This is a courtesy over the
    server's refusal and never a replacement for it: the comparison is
    still refused at creation with the remedy named, which the API suite
    proves. What this adds is that Run is DISABLED, so the courtesy
    cannot be clicked past."""
    page = bench(["stub/fast"])
    check_only(page, "stub/fast")
    page.get_by_test_id("prompt-input").fill("what does the contract say")
    attach(page, text_file())
    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    expect(page.get_by_test_id("run-button")).to_be_enabled()

    page.get_by_test_id("attach-mode").select_option("native")

    msg = page.get_by_test_id("attach-msg")
    expect(msg).to_contain_text("native mode sends images as content parts")
    assert "Switch to inline" in msg.inner_text()
    expect(page.get_by_test_id("run-button")).to_be_disabled()
    # And the reason is on the button, not only in a line the person may
    # have scrolled past.
    assert "images only" in page.get_by_test_id("run-button").get_attribute("title")


def test_native_over_an_image_runs_against_a_model_that_accepts_one(bench):
    """WINDOW: a live run end to end in native mode, from the pick to a
    done card.

    The mirror of the refusals: without this, every native assertion in
    the suite would be an assertion that native mode refuses things, and
    a mode that only ever refuses would pass all of them."""
    page = bench(["stub/vision"])
    check_only(page, "stub/vision")
    attach(page, png_file())
    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    page.get_by_test_id("attach-mode").select_option("native")

    # The estimate refuses to guess for an image, and says why instead of
    # printing a number nobody could stand behind.
    meta = page.get_by_test_id("attachment-meta").inner_text()
    assert "image tokens set by the provider" in meta
    assert "approx" not in meta

    run_and_wait(page, "what is in this image")
    expect(page.get_by_test_id("result-card")).to_have_count(1)


def test_native_over_an_image_is_refused_for_a_text_only_model(bench):
    """WINDOW: the run's card, for a lineup the catalog says cannot take
    an image. The refusal is the server's and arrives at creation, and
    what this proves is that it REACHES THE PAGE as a sentence rather
    than vanishing into a degraded ungrouped run.

    That degrade is the reason enforce_native_entry exists: the browser
    falls back to ungrouped runs when the group POST fails, so before the
    entry check a native refusal quietly became the paid call it had just
    refused."""
    page = bench(["stub/fast"])
    check_only(page, "stub/fast")
    attach(page, png_file())
    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    page.get_by_test_id("attach-mode").select_option("native")

    page.get_by_test_id("prompt-input").fill("what is in this image")
    page.get_by_test_id("run-button").click()

    card = page.get_by_test_id("result-card").first
    expect(card.get_by_test_id("card-status")).to_have_text(
        "error", timeout=DONE_TIMEOUT
    )
    body = card.inner_text()
    assert "does not accept image input" in body
    assert "stub/fast" in body


# ---- The mirrored constants. The H2 bounds precedent, twice more.
#
# WINDOW for all three: the source text of static/attach.js against the
# server constants it copies. No page is opened, because the failure
# being guarded is drift between two files rather than anything a browser
# would render differently.

STATIC = Path(__file__).resolve().parents[2] / "static"


def test_the_client_attachment_cap_matches_the_server_constant():
    """The client refuses a fifth document before uploading it and the
    server refuses one too. A client cap TIGHTER than the server's is the
    drift that hurts: it would refuse a legal comparison and blame the
    user's file, with no way to tell from the page that the bench would
    in fact have accepted it."""
    from bench.main import MAX_ATTACHMENTS

    source = (STATIC / "attach.js").read_text(encoding="utf-8")
    declared = re.search(r"const MAX_ATTACHMENTS = (\d+);", source)

    assert declared, "static/attach.js must declare MAX_ATTACHMENTS"
    assert int(declared.group(1)) == MAX_ATTACHMENTS


def test_the_client_byte_cap_matches_the_server_constant():
    """Same pair, same reason, one cap over. This one also feeds the
    refusal message's arithmetic, so drift here would quote a limit the
    server does not hold."""
    from bench.main import MAX_ATTACHMENT_BYTES

    source = (STATIC / "attach.js").read_text(encoding="utf-8")
    declared = re.search(r"const MAX_ATTACHMENT_BYTES = (\d+) \* 1024 \* 1024;", source)

    assert declared, "static/attach.js must declare MAX_ATTACHMENT_BYTES"
    assert int(declared.group(1)) * 1024 * 1024 == MAX_ATTACHMENT_BYTES


def test_the_client_native_suffixes_match_the_extractor():
    """The composer explains a native refusal before it happens, using
    its own copy of the image suffix list. A client list WIDER than the
    extractor's would let a file through to a refusal the composer
    promised would not come; a narrower one would refuse a file the
    bench accepts."""
    from bench.extract import NATIVE_IMAGE_TYPES

    source = (STATIC / "attach.js").read_text(encoding="utf-8")
    block = re.search(r"const NATIVE_SUFFIXES = \[(.*?)\];", source, re.S)

    assert block, "static/attach.js must declare NATIVE_SUFFIXES"
    declared = set(re.findall(r'"(\.[a-z]+)"', block.group(1)))
    assert declared == set(NATIVE_IMAGE_TYPES)


def test_review_repro_the_native_refusal_survives_the_attach_message(bench):
    """WINDOW: the composer's message line immediately after attaching a
    non-image while the mode is ALREADY native. The other ordering
    (attach first, switch mode second) is covered above and passes
    either way, which is precisely why it could not catch this.

    THE SUCCESS MESSAGE OVERWROTE THE REFUSAL. render() wrote "native
    mode sends images as content parts..." into the line, and addFiles
    replaced it with "attached 1 document" on the next statement. Run
    was correctly disabled, but the only surviving explanation was the
    button's title attribute, which never appears for a keyboard user
    and never appears for anyone who does not hover a control that looks
    inert. The person saw a successful attachment and a Run button that
    would not press, with nothing on screen connecting the two.

    A successful upload and an unusable staged set are both true. The
    one that blocks the run is the one that owns the line."""
    page = bench(["stub/fast"])
    check_only(page, "stub/fast")
    page.get_by_test_id("prompt-input").fill("what does the contract say")
    # Mode chosen BEFORE the file, which is the ordinary way round when
    # somebody has just finished an image comparison.
    page.get_by_test_id("attach-mode").select_option("native")

    attach(page, text_file())

    expect(chips(page)).to_have_count(1, timeout=DONE_TIMEOUT)
    msg = page.get_by_test_id("attach-msg")
    expect(msg).to_contain_text("native mode sends images as content parts")
    assert "Switch to inline" in msg.inner_text()
    # The success text must not be what survived.
    assert "attached 1 document" not in msg.inner_text()
    expect(page.get_by_test_id("run-button")).to_be_disabled()
