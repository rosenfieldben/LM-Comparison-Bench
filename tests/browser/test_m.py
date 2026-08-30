"""Phase M browser tests: the seams a person touches when a DATASET
carries the documents.

The Phase K suite one file over covers the composer's own staging area,
where a person picks a file for the next comparison. This one covers the
surface a person meets when the documents arrive from a dataset instead:
the experiment report, which names what each task read.

EVERY PROOF NAMES ITS WINDOW. The same provenance is rendered by two
different code paths now (the history chip, built from a stored group's
refs, and the report chip, built from an aggregate keyed by task), and a
proof about the wrong one of those reads as coverage while leaving the
other unguarded.
"""

import base64
import json

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.browser

CONTRACT = b"the contract says the delivery window is forty two days"


def upload(page, bench_url, filename, content):
    """Store one document through the API the dataset workflow uses.

    Upload first, then cite the digest: a dataset references documents
    and never carries them, so this is the first half of the workflow
    the report at the end of it describes.
    """
    resp = page.request.post(
        bench_url + "/attachments",
        data={
            "filename": filename,
            "content_base64": base64.b64encode(content).decode("ascii"),
        },
    )
    assert resp.ok, resp.text()
    return resp.json()["digest"]


def document_experiment(page, bench_url, tmp_path, name, digest):
    """One two-model experiment over one task that cites one document,
    run to done."""
    dataset = tmp_path / "docs.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "t1",
                "prompt": "what does the contract say?",
                "attachments": [digest],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    created = page.request.post(
        bench_url + "/experiments",
        data={
            "name": name,
            "dataset_path": str(dataset),
            "lineup": ["stub/fast", "stub/slow"],
            "budget": "standard",
        },
    )
    assert created.ok, created.text()
    eid = created.json()["id"]
    started = page.request.post(
        bench_url + f"/experiments/{eid}/start",
        data={"dataset_path": str(dataset)},
    )
    assert started.ok, started.text()
    for _ in range(200):
        detail = page.request.get(bench_url + f"/experiments/{eid}").json()
        if detail["status"] not in ("created", "running"):
            break
        page.wait_for_timeout(50)
    assert detail["status"] == "done", detail
    return eid


def test_the_report_names_what_each_task_read_and_never_the_filename(
    page, bench, bench_url, tmp_path
):
    """WINDOW: the rendered report panel for an experiment whose dataset
    cited a document, read as text rather than as markup.

    TWO CLAIMS, and the second is the one a person can check. The chips
    are there and they name the digest, the reading and the mode; and
    the filename the document was uploaded under is nowhere on the page,
    because a name is what somebody picked on their own machine and is
    not a property of the bytes. A report that named it would be citing
    a fact about a filesystem in an artifact about an experiment.

    inner_text rather than the HTML, deliberately: the question is what a
    reader sees, and a filename sitting in a hidden node is a different
    defect from one on the page.
    """
    digest = upload(page, bench_url, "secret-contract.txt", CONTRACT)
    eid = document_experiment(page, bench_url, tmp_path, "m report", digest)

    bench(["stub/fast"])
    page.evaluate("(id) => window.BenchReport.show(id)", eid)

    expect(page.get_by_test_id("report-documents")).to_be_visible()
    row = page.get_by_test_id("report-task-documents")
    expect(row).to_have_count(1)
    expect(row).to_have_attribute("data-task", "t1")
    chip = page.get_by_test_id("task-attachment")
    expect(chip).to_have_count(1)
    # The face carries the short digest, the same seven characters every
    # other chip in this app abbreviates to; the whole digest, which is
    # what a person copies back into a dataset line, is in the title
    # below.
    expect(chip).to_have_text("sha256 " + digest[:7])
    # And the reading rides in the title, where the history chips put it.
    title = chip.get_attribute("title")
    assert "read as document" in title, title
    assert "inline" in title, title
    assert digest in title, title
    # NOWHERE ON THE PAGE, and the name is distinctive so this is exact.
    panel = page.get_by_test_id("report-panel").inner_text()
    assert "secret-contract.txt" not in panel, panel


def test_a_report_over_a_plain_dataset_shows_no_documents_block(
    page, bench, bench_url, tmp_path
):
    """WINDOW: the same panel for an experiment whose dataset cited
    nothing.

    Rule one at the view: an experiment with no documents has to look
    exactly like every experiment did before this phase. An empty box
    with a heading about documents would be a new thing on the page for
    every run that has none.
    """
    dataset = tmp_path / "plain.jsonl"
    dataset.write_text(
        json.dumps({"id": "t1", "prompt": "hi"}) + "\n", encoding="utf-8"
    )
    created = page.request.post(
        bench_url + "/experiments",
        data={
            "name": "m plain",
            "dataset_path": str(dataset),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    )
    assert created.ok, created.text()
    eid = created.json()["id"]
    page.request.post(
        bench_url + f"/experiments/{eid}/start", data={"dataset_path": str(dataset)}
    )
    for _ in range(200):
        detail = page.request.get(bench_url + f"/experiments/{eid}").json()
        if detail["status"] not in ("created", "running"):
            break
        page.wait_for_timeout(50)

    bench(["stub/fast"])
    page.evaluate("(id) => window.BenchReport.show(id)", eid)

    expect(page.get_by_test_id("report-panel")).to_be_visible()
    expect(page.get_by_test_id("report-outcomes")).to_be_visible()
    expect(page.get_by_test_id("report-documents")).to_have_count(0)
