"""Phase N3 browser tests: the experiment lifecycle from the page.

Create over a stored dataset, Start, watch the progress door, Stop, and
the list the report opens from (and, since N4, the critical path's Score
and the Score row's place in the status walk; the rest of Score is in
test_n4.py), driven through the panel a person uses
against the real doors and the stub. What the server RECORDED is read
back through GET /experiments and compared, never what the page believed
it sent: a page that agreed only with itself would pass every proof here
while creating something else.

MONEY MOVES ON START (and, since N4, on Score when a judge grades), so
Start is the door these proofs lean on hardest: it is live for a created
experiment and for nothing else, it sends the digest the experiment
recorded, and a greyed button is proved to be a courtesy by sending the
request anyway.

ONE EXPERIMENT RUNS AT A TIME PER BENCH, and the browser suite shares one
bench for the whole session, so every test that starts an experiment
leaves it finished: the `runs` fixture stops and drains anything a test
leaves running, pass or fail, before the next test can meet a 409.

Every test runs with the CSP and error collectors armed; a test that
causes a console error on purpose names it.
"""

import base64
import json
import re
import uuid

import pytest
from playwright.sync_api import expect
from test_contrast import composite, contrast
from test_i4 import check_all_chips, column_texts
from test_n import (
    ABORTED_RESOURCE,
    REFUSED_RESOURCE,
    add_task,
    arm,
    entry_for,
    hold,
    open_datasets,
    settle,
    store,
    unique,
    wait_held,
)

from bench import main

pytestmark = pytest.mark.browser

DONE_TIMEOUT = 25_000
# Chromium's console lines for refusals the proofs cause on purpose.
CONFLICT_RESOURCE = "Failed to load resource: the server responded with a status of 409"
MISSING_RESOURCE = "Failed to load resource: the server responded with a status of 404"
# A request that reached the door and whose answer the test then lost.
LOST_RESOURCE = "Failed to load resource: net::ERR_"


@pytest.fixture(autouse=True)
def collectors(page):
    """Arm the collectors before the test and assert them after, as
    test_n.py does; yields the console-error prefixes a test allows."""
    errors = arm(page)
    allowed = [REFUSED_RESOURCE]
    yield allowed
    assert page.evaluate("window.__csp || []") == []
    assert [e for e in errors if not e.startswith(tuple(allowed))] == []


def drain(page, bench_url, eid):
    """Stop an experiment while it runs and wait until it has finished."""
    for _ in range(400):
        status = record(page, bench_url, eid)["status"]
        if status not in ("created", "running"):
            return
        if status == "running":
            page.request.post(f"{bench_url}/experiments/{eid}/stop", data={})
        page.wait_for_timeout(100)


# THE BENCH HAS ONE SCORING SLOT AND NO DOOR SAYS WHEN A PASS ENDS, nor
# can any door stop one. A test that scores waits for the slot to be free
# by asking the Score door itself, with a body that can never start a
# pass: a digest no experiment records, sent to a finished experiment.
# The door checks the slot before the dataset, so a busy slot answers
# 409 with its sentence and a free one 422 naming the probe's digest
# (enforce_recorded_digest's, or stored_dataset's if that check were
# gone). Anything else fails loudly rather than being read as either.
# IDLE MEANS THE SLOT IS FREE, NOT THAT THE PASS SCORED ANYTHING: a pass
# that raised is idle too, so every wait for scores is followed by an
# assertion on the report.
PROBE_DIGEST = "0" * 64
_probe_targets = {}


def probe_target(page, bench_url):
    """The finished experiment the idle probe asks, made once per bench.
    Its own scoring history does not matter: the slot is the bench's."""
    if bench_url not in _probe_targets:
        _, digest = stored_digest(page, bench_url, [{"id": "probe"}])
        eid = api_experiment(page, bench_url, digest, ["stub/fast"], unique("probe"))
        assert page.request.post(
            f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
        ).ok
        assert wait_status(page, bench_url, eid)["status"] == "done"
        _probe_targets[bench_url] = eid
    return _probe_targets[bench_url]


def scoring_busy(page, bench_url):
    eid = probe_target(page, bench_url)
    resp = page.request.post(
        f"{bench_url}/experiments/{eid}/score", data={"dataset_digest": PROBE_DIGEST}
    )
    detail = resp.json().get("detail")
    if (
        resp.status == 409
        and isinstance(detail, str)
        and re.fullmatch(r"a scoring pass for experiment \d+ is running", detail)
    ):
        return True
    if resp.status == 422 and isinstance(detail, str) and PROBE_DIGEST[:12] in detail:
        return False
    raise AssertionError(f"the idle probe got {resp.status}: {resp.text()}")


def wait_scoring_idle(page, bench_url, timeout_s=60):
    """Poll the probe until the scoring slot is free; raise if it never is."""
    for _ in range(timeout_s * 10):
        if not scoring_busy(page, bench_url):
            return
        page.wait_for_timeout(100)
    raise AssertionError(f"a scoring pass still held the slot after {timeout_s}s")


def stored_digest(page, bench_url, rows, name=None):
    """A dataset stored through the door, with a unique prompt so it is
    this test's own, returning its name and digest."""
    tag = uuid.uuid4().hex
    content = "".join(
        json.dumps({**row, "prompt": row.get("prompt", "p") + " " + tag}) + "\n"
        for row in rows
    )
    name = name or unique("xp dataset")
    resp = page.request.post(
        bench_url + "/datasets", data={"name": name, "content": content}
    )
    assert resp.ok, resp.text()
    return name, resp.json()["digest"]


def select_dataset(page, name):
    open_datasets(page)
    entry = entry_for(page, name)
    entry.click()
    expect(entry).to_have_attribute("aria-pressed", "true")


def open_experiments(page):
    page.get_by_test_id("experiments-toggle").click()
    expect(page.get_by_test_id("experiment-list")).to_have_attribute(
        "data-state", re.compile("^(ready|empty)$"), timeout=DONE_TIMEOUT
    )


def create(page, name):
    """Fill the name, press Create, and return the new experiment's id
    from the message the panel prints."""
    page.get_by_test_id("experiment-name").fill(name)
    button = page.get_by_test_id("experiment-create-button")
    expect(button).to_be_enabled()
    button.click()
    msg = page.get_by_test_id("experiment-create-msg")
    # The whole sentence, pinned here because every Create passes through:
    # the money law it states is the one the page's buttons are labelled by.
    expect(msg).to_have_text(
        re.compile(
            r"^created experiment \d+\. Creating spent nothing; money moves on "
            r"Start, and on Score when a judge grades\.$"
        )
    )
    return int(re.search(r"\d+", msg.inner_text()).group())


def created_id(page):
    """The id of the experiment the create message names."""
    text = page.get_by_test_id("experiment-create-msg").inner_text()
    return int(re.search(r"\d+", text).group())


def record(page, bench_url, eid):
    return page.request.get(f"{bench_url}/experiments/{eid}").json()


def row_for(page, eid):
    return page.locator(f"[data-testid=experiment-row][data-id='{eid}']")


def ready(page, bench, bench_url, lineup, rows):
    """A bench with the lineup checked, a stored dataset of rows selected,
    and the experiment panel open. Returns the dataset's digest."""
    name, digest = stored_digest(page, bench_url, rows)
    bench(lineup)
    check_all_chips(page)
    select_dataset(page, name)
    open_experiments(page)
    return digest


def api_experiment(page, bench_url, digest, lineup, name=None, **fields):
    """An experiment created through the door, as another tab or a script
    would create it; returns its id."""
    resp = page.request.post(
        bench_url + "/experiments",
        data={
            "name": name or unique("api"),
            "dataset_digest": digest,
            "lineup": lineup,
            "budget": "standard",
            **fields,
        },
    )
    assert resp.ok, resp.text()
    return resp.json()["id"]


def wait_status(page, bench_url, eid, done=lambda s: s not in ("created", "running")):
    """Poll GET /experiments/{id} until its status satisfies done."""
    for _ in range(600):
        detail = record(page, bench_url, eid)
        if done(detail["status"]):
            return detail
        page.wait_for_timeout(50)
    raise AssertionError(f"experiment {eid} still {detail['status']}")


def progress_log(page):
    """Every /progress request the page makes, and how each ended."""
    log = []
    page.on(
        "request",
        lambda r: log.append(("open", r.url)) if "/progress" in r.url else None,
    )
    page.on(
        "requestfailed",
        lambda r: log.append(("closed", r.url)) if "/progress" in r.url else None,
    )
    return log


def opened(log, eid):
    return [e for e in log if e[0] == "open" and e[1].endswith(f"/{eid}/progress")]


def closed(log, eid):
    return [e for e in log if e[0] == "closed" and e[1].endswith(f"/{eid}/progress")]


# ---- The critical path.


def test_review_repro_rows_to_a_report_through_the_panel(
    page, bench, bench_url, runs, scorings
):
    """WINDOW: the builder's rows, Store, the Datasets selection, Create,
    Start, the progress door, Score with a judge and the report, all
    through the page.

    THE COMMISSION'S CRITICAL PATH. Three tasks by rows, one each of
    exact, regex and judge; stored; created over the stored digest with
    two stub models; started; the counters watched to the total; the
    report opened again once the run finished, now saying done. The
    record carries exactly the digest stored, the lineup checked and the
    budget shown, and the projection beside Start is the text of the
    projection Create's answer carried. Then, WITHOUT selecting the row
    again (so the Score row reads what the store answered when Create
    selected the new experiment, while it was still created),
    Score waits for a judge, is labelled as paying one, sends exactly the
    recorded digest and the judge chosen, and the report is read again
    after the 202; once the pass has ended, the report has three series
    on each of the two arms, the judge named on the judge rows only."""
    bench(["stub/fast", "stub/slow"])
    check_all_chips(page)
    open_datasets(page)
    dataset = unique("critical")
    page.get_by_test_id("dataset-name").fill(dataset)
    tag = uuid.uuid4().hex
    add_task(page, "e1", "say Hello " + tag, "exact", reference="Hello")
    add_task(page, "r1", "say Hi " + tag, "regex", pattern="H")
    add_task(page, "j1", "be kind " + tag, "judge", rubric="kindness")
    assert store(page).startswith("stored as " + dataset)
    entry = entry_for(page, dataset)
    digest = entry.get_attribute("data-digest")
    entry.click()
    open_experiments(page)
    expect(page.get_by_test_id("experiment-source-dataset")).to_have_text(
        "dataset: " + dataset + " · sha256 " + digest[:7] + " · 3 tasks"
    )
    expect(page.get_by_test_id("experiment-source-lineup")).to_have_text(
        "lineup: stub/fast, stub/slow"
    )
    page.get_by_test_id("experiment-name").fill(unique("critical path"))

    with page.expect_response(
        lambda r: r.url.endswith("/experiments") and r.request.method == "POST"
    ) as answer:
        page.get_by_test_id("experiment-create-button").click()

    eid = answer.value.json()["id"]
    runs.append(eid)
    projected = answer.value.json()["projected_cost"]
    created = record(page, bench_url, eid)
    assert created["dataset_digest"] == digest
    assert created["lineup"] == ["stub/fast", "stub/slow"]
    assert created["budget"] == "standard"
    assert created["status"] == "created"
    expect(row_for(page, eid)).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_test_id("experiment-projection")).to_have_text(
        page.evaluate("(p) => BenchLib.projectionText(p)", projected)
    )
    assert projected["unpriced"] == [] and projected["total_usd"] is not None
    start = page.get_by_test_id("experiment-start")
    expect(start).to_be_enabled()

    start.click()

    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "done", timeout=DONE_TIMEOUT
    )
    detail = record(page, bench_url, eid)
    finished_trials = detail["trials_done"] + detail["trials_failed"]
    assert finished_trials == 6 and detail["trials_refused"] == 0
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        f"done {detail['trials_done']} · failed {detail['trials_failed']} "
        "· refused 0 · of 6 trials"
    )
    expect(start).to_be_disabled()
    # The projection was about a created experiment; once Start has spent
    # it is not shown beside a button that can no longer be pressed.
    expect(page.get_by_test_id("experiment-projection")).to_have_text("")
    report = page.get_by_test_id("report-panel")
    expect(report.get_by_test_id("report-banner")).to_contain_text(
        "done", timeout=DONE_TIMEOUT
    )
    expect(report.get_by_test_id("report-row")).to_have_count(2)
    expect(report.get_by_test_id("report-score-row")).to_have_count(0)

    score = page.get_by_test_id("experiment-score")
    expect(page.get_by_test_id("experiment-score-row")).to_be_visible()
    expect(score).to_have_text("Score · pays the judge")
    expect(score).to_be_disabled()
    expect(page.get_by_test_id("experiment-score-nudge")).to_have_text(
        "Score waits: choose a judge, because a pass without one records every "
        "judge task as a scoring failure, and that record does not rewrite"
    )
    # A catalog model outside the lineup, so no row is self-judged.
    page.get_by_test_id("experiment-judge").select_option("stub/html")
    expect(score).to_be_enabled()
    with page.expect_request(
        lambda r: r.url.endswith(f"/experiments/{eid}/report") and r.method == "GET"
    ):
        with page.expect_request(
            lambda r: r.url.endswith(f"/experiments/{eid}/score")
        ) as sent:
            score.click()
    assert json.loads(sent.value.post_data) == {
        "dataset_digest": digest,
        "judge_model": "stub/html",
    }
    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile(r"^a scoring pass was started at \d\d:\d\d:\d\d UTC; ")
    )
    wait_scoring_idle(page, bench_url)
    row_for(page, eid).click()
    judges = column_texts(page, "report-scores", "report-judge", 6)
    none = chr(0x2014)  # the report's cell for a series with no judge
    assert sorted(judges) == ["stub/html", "stub/html", none, none, none, none]


# ---- Create sends what was set and nothing else.


def create_body(page):
    """Press Create and return the JSON body it sent."""
    with page.expect_request(
        lambda r: r.url.endswith("/experiments") and r.method == "POST"
    ) as info:
        page.get_by_test_id("experiment-create-button").click()
    expect(page.get_by_test_id("experiment-create-msg")).to_have_text(
        re.compile(r"^created experiment \d+\.")
    )
    return json.loads(info.value.post_data)


def comparable(detail):
    """A record without what differs between any two creations."""
    return {k: v for k, v in detail.items() if k not in ("id", "created_at")}


def test_blank_is_not_sent_and_the_row_is_the_one_curl_makes(page, bench, bench_url):
    """WINDOW: the body Create sends with every optional box blank and
    every control unset, and the row it records beside one created by
    the API with params, repeats and the seed absent.

    RULE ONE ON THE WIRE, not only in the row: the server stores params
    absent and params {} identically, so a row comparison alone could not
    catch a page that sent the empty object. The body has no params, no
    repeats, no task_order_seed, no primary_metric and no attachments_mode
    (the dataset cites no document); and the row equals the curl row in
    every field but id and created_at."""
    digest = ready(page, bench, bench_url, ["stub/fast"], [{"id": "b1"}])
    name = unique("blank")
    page.get_by_test_id("experiment-name").fill(name)
    # PRE-STATE: nothing in the form or the controls is set.
    assert page.evaluate("BenchControls.experimentParams()") == {}
    expect(page.get_by_test_id("experiment-repeats")).to_have_value("")
    expect(page.get_by_test_id("experiment-seed")).to_have_value("")
    expect(page.get_by_test_id("experiment-attachments")).to_be_disabled()

    body = create_body(page)

    assert body == {
        "name": name,
        "dataset_digest": digest,
        "lineup": ["stub/fast"],
        "budget": "standard",
        "estimand_mode": "routed_service",
        "halt_on_refusal": True,
    }
    by_page = record(page, bench_url, created_id(page))
    by_api = page.request.post(
        bench_url + "/experiments",
        data={
            "name": name,
            "dataset_digest": digest,
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    )
    assert by_api.ok, by_api.text()
    assert comparable(by_page) == comparable(
        record(page, bench_url, by_api.json()["id"])
    )
    assert by_page["params"] is None


def test_what_is_set_is_sent_and_recorded(page, bench, bench_url):
    """WINDOW: the body Create sends with every field of the form set,
    including a task order seed of 0 and a control in + Controls, and the
    row it records.

    A seed of 0 is a real seed and not a blank one: Number("") is 0 too,
    and a page that confused them would record a shuffle nobody asked for
    or drop one somebody did. The estimand, the primary metric, halt on
    refusal unchecked, repeats and the composer's temperature all land on
    the row as set, and so does the name, spaces at either end included:
    the page trims nothing the server would keep."""
    digest = ready(
        page,
        bench,
        bench_url,
        ["stub/fast"],
        [{"id": "s1", "reference": "x", "scorer": {"kind": "exact"}}],
    )
    page.get_by_test_id("toggle-controls").click()
    page.get_by_test_id("ctl-temperature").fill("0")
    page.get_by_test_id("experiment-repeats").fill("2")
    page.get_by_test_id("experiment-seed").fill("0")
    page.get_by_test_id("experiment-estimand").select_option("underlying_model")
    page.get_by_test_id("experiment-metric").select_option("exact")
    page.get_by_test_id("experiment-halt").uncheck()
    name = "  " + unique("set") + " "
    page.get_by_test_id("experiment-name").fill(name)
    expect(page.get_by_test_id("experiment-source-controls")).to_have_text(
        "controls: t=0"
    )

    body = create_body(page)

    assert body == {
        "name": name,
        "dataset_digest": digest,
        "lineup": ["stub/fast"],
        "budget": "standard",
        "params": {"temperature": 0},
        "repeats": 2,
        "task_order_seed": 0,
        "estimand_mode": "underlying_model",
        "primary_metric": "exact",
        "halt_on_refusal": False,
    }
    row = record(page, bench_url, created_id(page))
    assert row["name"] == name
    assert row["params"] == {"temperature": 0}
    assert (row["repeats"], row["task_order_seed"]) == (2, 0)
    assert row["estimand_mode"] == "underlying_model"
    assert row["primary_metric"] == "exact"
    assert row["halt_on_refusal"] is False


def test_nothing_is_prefilled_and_create_waits_for_what_the_server_needs(
    page, bench, bench_url
):
    """WINDOW: the form as it first paints, and the nudge beside Create
    as each missing piece is supplied.

    NOTHING IS PRE-FILLED: the name, repeats and the seed are empty, the
    selects stand on the server's defaults, halt on refusal is checked as
    the server's default is, and no metric is declared. Create is greyed
    for a missing name, dataset or model, and for a control or box out of
    its range, each a request the server would refuse; and for a box
    whose text is not a number at all ("1e"), which reads as empty and
    would be dropped from the body without a word, the page's own rule.
    The nudge says which, and supplying it moves to the next. The lineup
    line lists the checked models and no other."""
    name, _ = stored_digest(page, bench_url, [{"id": "n1"}])
    bench(["stub/fast", "stub/slow"])
    open_experiments(page)
    create_button = page.get_by_test_id("experiment-create-button")
    nudge = page.get_by_test_id("experiment-create-nudge")
    expect(page.get_by_test_id("experiment-name")).to_have_value("")
    expect(page.get_by_test_id("experiment-repeats")).to_have_value("")
    expect(page.get_by_test_id("experiment-seed")).to_have_value("")
    expect(page.get_by_test_id("experiment-estimand")).to_have_value("routed_service")
    expect(page.get_by_test_id("experiment-metric")).to_have_value("")
    expect(page.get_by_test_id("experiment-halt")).to_be_checked()
    expect(create_button).to_have_text("Create · free")
    expect(create_button).to_be_disabled()
    expect(nudge).to_have_text("Create waits: name the experiment")

    page.get_by_test_id("experiment-name").fill(unique("waits"))
    expect(nudge).to_have_text(
        "Create waits: select a stored dataset in Datasets above"
    )
    select_dataset(page, name)
    expect(nudge).to_have_text("Create waits: check a model in the lineup above")
    page.get_by_test_id("lineup-chip").first.click()
    expect(page.get_by_test_id("experiment-source-lineup")).to_have_text(
        "lineup: stub/fast"
    )
    expect(nudge).to_have_text("")
    expect(create_button).to_be_enabled()

    page.get_by_test_id("toggle-controls").click()
    page.get_by_test_id("ctl-temperature").fill("5")
    expect(nudge).to_have_text("Create waits: check temperature")
    page.get_by_test_id("ctl-temperature").fill("")
    page.get_by_test_id("experiment-repeats").fill("21")
    expect(nudge).to_have_text("Create waits: check repeats")
    page.get_by_test_id("experiment-repeats").fill("")
    for bad in ("1.5", "-1"):
        page.get_by_test_id("experiment-seed").fill(bad)
        expect(nudge).to_have_text("Create waits: check task order seed")
        expect(create_button).to_be_disabled()
    page.get_by_test_id("experiment-seed").fill("7")
    expect(nudge).to_have_text("")
    # Typed key by key, since fill refuses text a number box cannot hold:
    # the box then reads as empty, which is what the body would send.
    page.get_by_test_id("experiment-repeats").press_sequentially("1e")
    expect(page.get_by_test_id("experiment-repeats")).to_have_value("")
    expect(nudge).to_have_text("Create waits: check repeats")
    expect(create_button).to_be_disabled()
    page.get_by_test_id("experiment-repeats").fill("")
    expect(nudge).to_have_text("")


def test_the_form_offers_what_the_selected_dataset_can_take(page, bench, bench_url):
    """WINDOW: the primary-metric options and the attachments control for
    a dataset that cites no document, and for one that cites one.

    The metric select offers the scorer kinds the dataset uses and none
    declared. The attachments control is disabled with the reason on a
    dataset that cites nothing, where the server refuses native and
    inline is its default, and enabled on one that cites a document."""
    tag = uuid.uuid4().hex
    doc = page.request.post(
        bench_url + "/attachments",
        data={
            "filename": "cited.txt",
            "content_base64": base64.b64encode(("cited " + tag).encode()).decode(),
        },
    ).json()["digest"]
    plain, _ = stored_digest(
        page,
        bench_url,
        [
            {"id": "m1", "reference": "x", "scorer": {"kind": "exact"}},
            {"id": "m2", "scorer": {"kind": "regex", "pattern": "x"}},
        ],
    )
    citing, _ = stored_digest(page, bench_url, [{"id": "c1", "attachments": [doc]}])
    bench(["stub/fast"])
    open_experiments(page)
    attachments = page.get_by_test_id("experiment-attachments")
    note = page.get_by_test_id("experiment-attachments-note")
    expect(attachments).to_be_disabled()
    expect(note).to_have_text(
        "select a stored dataset to choose how its documents reach the models"
    )

    select_dataset(page, plain)
    options = page.get_by_test_id("experiment-metric").locator("option")
    expect(options).to_have_count(3)
    assert [o.get_attribute("value") for o in options.all()] == ["", "exact", "regex"]
    expect(attachments).to_be_disabled()
    expect(note).to_have_text(
        "this dataset cites no document, so there is no mode to choose"
    )

    entry_for(page, citing).click()
    expect(attachments).to_be_enabled()
    expect(note).to_have_text("")
    expect(page.get_by_test_id("experiment-metric").locator("option")).to_have_count(1)


# ---- The projection.


def test_an_unpriced_member_is_named_and_no_figure_is_given(page, bench, bench_url):
    """WINDOW: the projection beside Start after Create over a lineup with
    a member the catalog does not price.

    Every figure is null when any member is unpriced, because a total
    missing one arm would read as the comparison's total; the panel names
    the member, verbatim, and gives no figure. PRE-STATE: the stub prices
    stub/fast and not stub/nousage, and the door says so itself."""
    ready(page, bench, bench_url, ["stub/fast", "stub/nousage"], [{"id": "u1"}])

    eid = create(page, unique("unpriced"))

    detail = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("unpriced by api"),
            "dataset_digest": record(page, bench_url, eid)["dataset_digest"],
            "lineup": ["stub/fast", "stub/nousage"],
            "budget": "standard",
        },
    ).json()["projected_cost"]
    assert detail["unpriced"] == ["stub/nousage"] and detail["total_usd"] is None
    expect(page.get_by_test_id("experiment-projection")).to_have_text(
        "unpriced: stub/nousage. No figure is given, because a total missing one "
        "arm would read as the whole comparison's total."
    )


def test_an_experiment_created_elsewhere_has_no_projection_to_show(
    page, bench, bench_url
):
    """WINDOW: the projection beside Start for a created experiment the
    page did not create.

    The server returns the projection only from Create and stores it
    nowhere, so the panel says it has none rather than inventing one."""
    _, digest = stored_digest(page, bench_url, [{"id": "e1"}])
    eid = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("elsewhere"),
            "dataset_digest": digest,
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    bench(["stub/fast"])
    open_experiments(page)

    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-projection")).to_have_text(
        "no projection in this tab: the bench returns one only when the "
        "experiment is created, and stores it nowhere"
    )
    expect(page.get_by_test_id("experiment-start")).to_be_enabled()


# ---- Start, Stop and the progress door.


def test_start_is_live_for_a_created_experiment_and_the_door_is_the_rule(
    page, bench, bench_url, collectors, runs
):
    """WINDOW: the Start and Stop controls for a finished experiment, and
    the start and stop doors sent the same requests anyway.

    Start is greyed for anything but a created experiment and Stop is
    absent unless it runs; both are courtesies. The door is the rule, and
    it refuses a second start and a stop of a finished experiment in its
    own words, whatever the page shows."""
    collectors.append(CONFLICT_RESOURCE)
    _, digest = stored_digest(page, bench_url, [{"id": "d1"}])
    eid = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("finished"),
            "dataset_digest": digest,
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    runs.append(eid)
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok
    for _ in range(200):
        if record(page, bench_url, eid)["status"] == "done":
            break
        page.wait_for_timeout(50)
    bench(["stub/fast"])
    open_experiments(page)

    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-status")).to_have_text("done")
    expect(page.get_by_test_id("experiment-start")).to_be_disabled()
    expect(page.get_by_test_id("experiment-stop")).to_be_hidden()
    again = page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    )
    assert again.status == 409
    assert "an experiment runs once" in again.json()["detail"]
    stop = page.request.post(f"{bench_url}/experiments/{eid}/stop", data={})
    assert stop.status == 409
    assert stop.json()["detail"] == f"experiment {eid} is not running"


def test_start_sends_the_digest_the_experiment_recorded(page, bench, bench_url, runs):
    """WINDOW: the body Start sends for an older experiment, after a newer
    one over another dataset was created and the Datasets selection moved
    to that other dataset.

    Create sends the selection; Start sends the digest the experiment
    RECORDED, because the door refuses any other and a person may select
    another dataset between the two. The request's digest is the selected
    experiment's, not the newest row's and not the Datasets selection's."""
    name, digest = stored_digest(page, bench_url, [{"id": "a1"}])
    other, other_digest = stored_digest(page, bench_url, [{"id": "b1"}])
    bench(["stub/fast"])
    check_all_chips(page)
    select_dataset(page, name)
    open_experiments(page)
    older = create(page, unique("recorded"))
    runs.append(older)
    entry_for(page, other).click()
    newer = create(page, unique("newer"))
    expect(row_for(page, newer)).to_have_attribute("aria-pressed", "true")
    row_for(page, older).click()
    expect(page.get_by_test_id("experiment-source-dataset")).to_contain_text(
        other_digest[:7]
    )

    with page.expect_request(
        lambda r: r.url.endswith(f"/experiments/{older}/start")
    ) as info:
        page.get_by_test_id("experiment-start").click()

    assert json.loads(info.value.post_data) == {"dataset_digest": digest}
    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "done", timeout=DONE_TIMEOUT
    )


def test_stop_mid_run_leaves_the_unrun_tasks_out_of_the_record(
    page, bench, bench_url, runs
):
    """WINDOW: Stop pressed while a five-task stub/slow experiment runs,
    the status and the panel after, and the report and export that follow.

    Stop is present while the experiment runs. The trial in flight
    finishes and the rest never run, and the status says so in the
    server's words; the word that it is stopping goes once it has, and
    the stream is closed on the stopped frame, so no further request
    reaches the progress door. The report and the export carry only the
    trials that ran, and the report says how many never did."""
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"t{n}"} for n in range(5)])
    log = progress_log(page)
    eid = create(page, unique("stopped"))
    runs.append(eid)
    page.get_by_test_id("experiment-start").click()
    counters = page.get_by_test_id("experiment-counters")
    expect(counters).to_have_text(re.compile(r"^done [1-9] "), timeout=DONE_TIMEOUT)
    stop = page.get_by_test_id("experiment-stop")
    expect(stop).to_be_visible()

    stop.click()

    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text("stopping after the trial in flight")
    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "stopped · stopped between trials", timeout=DONE_TIMEOUT
    )
    expect(message).to_have_text("")
    expect(stop).to_be_hidden()
    after = len(opened(log, eid))
    page.wait_for_timeout(4_000)
    assert len(opened(log, eid)) == after
    detail = record(page, bench_url, eid)
    ran = detail["trials_done"]
    assert 1 <= ran < 5
    report = page.request.get(f"{bench_url}/experiments/{eid}/report").json()
    assert report["plan"]["cells_recorded"] == ran
    assert report["models"][0]["trials"]["not_run"] == 5 - ran
    export = page.request.get(f"{bench_url}/experiments/{eid}/export.jsonl").text()
    results = [json.loads(line) for line in export.splitlines()]
    trial_tasks = {r["task_id"] for r in results if "task_id" in r}
    assert len(trial_tasks) == ran


def test_the_progress_stream_resumes_after_it_is_cut(page, bench, bench_url, runs):
    """WINDOW: the progress door's first response replaced by a stream
    that carries one stale frame and ends, the reconnection after it
    while the experiment still runs, and the requests after the finished
    frame.

    RECONNECTION IS THE ORDINARY CASE. The page shows the stale frame
    (one refused trial, which never happened) and says the stream
    dropped; the reconnected stream's first frame carries the current
    counters, and the page shows them at once, refused back to 0 and the
    note gone, and MID-RUN, before the experiment finishes. A page that
    summed frames as deltas would keep the refusal; one that ignored the
    reconnected stream until the end would show nothing mid-run. After
    the finished frame the page closes the stream, and no request
    follows."""
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"k{n}"} for n in range(5)])
    eid = create(page, unique("resumed"))
    runs.append(eid)
    progress = []

    def cut_the_first(route):
        progress.append(route.request.url)
        if len(progress) == 1:
            stale = {
                "type": "progress",
                "status": "running",
                "status_detail": None,
                "trials_total": 5,
                "trials_done": 0,
                "trials_refused": 1,
                "trials_failed": 0,
                "spend_usd": 0,
            }
            route.fulfill(
                status=200,
                content_type="text/event-stream",
                body="data: " + json.dumps(stale) + "\n\n",
            )
        else:
            route.continue_()

    page.route(f"**/experiments/{eid}/progress", cut_the_first)
    page.get_by_test_id("experiment-start").click()

    counters = page.get_by_test_id("experiment-counters")
    note = page.get_by_test_id("experiment-stream")
    expect(note).to_have_text(
        "the progress stream dropped; reconnecting", timeout=DONE_TIMEOUT
    )
    # The stale frame, shown for what it said, before the reconnection.
    expect(counters).to_have_text("done 0 · failed 0 · refused 1 · of 5 trials")
    for _ in range(400):
        if len(progress) >= 2:
            break
        page.wait_for_timeout(25)
    # The reconnected stream's first frame replaces the stale counters as
    # soon as it lands, not at the end of the run.
    expect(counters).to_have_text(
        re.compile(r"^done [0-4] · failed 0 · refused 0 · of 5 trials$"),
        timeout=1_500,
    )
    expect(note).to_have_text("")
    expect(counters).to_have_text(
        re.compile(r"^done [1-4] · failed 0 · refused 0 · of 5 trials$"),
        timeout=DONE_TIMEOUT,
    )
    expect(page.get_by_test_id("experiment-status")).to_have_text("running")
    expect(counters).to_have_text(
        "done 5 · failed 0 · refused 0 · of 5 trials", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("experiment-status")).to_have_text("done")
    assert len(progress) >= 2
    after = len(progress)
    page.wait_for_timeout(4_000)
    assert len(progress) == after


# ---- The list.


def test_a_row_selects_its_experiment_and_opens_its_report(page, bench, bench_url):
    """WINDOW: clicks on two rows of the list, the report each opens, and
    a reload of the list.

    A row marks itself selected and the report that opens is that row's
    experiment's, as a row always opened the report; selecting another
    moves the mark, and only the selected row carries the accent bar.
    The selection is kept by id across a reload of the list."""
    _, digest = stored_digest(page, bench_url, [{"id": "l1"}])
    names = [unique("listed"), unique("listed")]
    ids = [api_experiment(page, bench_url, digest, ["stub/fast"], n) for n in names]
    bench(["stub/fast"])
    open_experiments(page)
    first, second = row_for(page, ids[0]), row_for(page, ids[1])
    expect(first).to_have_attribute("aria-pressed", "false")
    banner = page.get_by_test_id("report-banner")

    first.click()

    expect(first).to_have_attribute("aria-pressed", "true")
    expect(banner).to_contain_text(names[0])
    expect(page.get_by_test_id("experiment-title")).to_have_text(
        f"{names[0]} · experiment {ids[0]}"
    )
    second.click()
    expect(second).to_have_attribute("aria-pressed", "true")
    expect(first).to_have_attribute("aria-pressed", "false")
    expect(banner).to_contain_text(names[1])
    expect(page.get_by_test_id("report-banner")).to_have_count(1)
    assert first.evaluate("el => getComputedStyle(el).boxShadow") == "none"
    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    expect(row_for(page, ids[1])).to_have_attribute("aria-pressed", "true")


def test_a_refusal_at_create_is_the_servers_sentence(page, bench, bench_url):
    """WINDOW: Create over 2000 tasks with 3 repeats, which the door
    refuses as over its trial bound, and the message the panel prints.

    The page does not second-guess the arithmetic; the server's sentence,
    numbers and all, is what a person reads, and nothing is created."""
    rows = [{"id": f"x{n}"} for n in range(2000)]
    ready(page, bench, bench_url, ["stub/fast"], rows)
    page.get_by_test_id("experiment-repeats").fill("3")
    name = unique("too many")
    page.get_by_test_id("experiment-name").fill(name)

    page.get_by_test_id("experiment-create-button").click()

    msg = page.get_by_test_id("experiment-create-msg")
    expect(msg).to_have_text(
        "not created: 2000 tasks x 3 repeats x 1 models is 6000 paid calls, "
        f"over the {main.MAX_TRIALS} limit. Shorten the dataset or the lineup."
    )
    expect(msg).to_have_attribute("data-state", "refused")
    listed = page.request.get(bench_url + "/experiments").json()["experiments"]
    assert name not in [e["name"] for e in listed]


# ---- Contrast.


def test_start_stop_and_score_follow_every_status(
    page, bench, bench_url, tmp_path, runs, collectors
):
    """WINDOW: the Start and Stop controls and the Score row for
    experiments that are done, created, failed, running, and stopped from
    the page, in that order.

    Start is live for created and for nothing else, and Stop is present
    while running and at no other time; a rule written as "not done"
    would pass a finished-only proof and offer Start on a running,
    stopped or failed experiment. The Score row (N4) is present for every
    finished status and absent while created or running, and the walk
    starts on done so absent is measured against a row that renders; it
    appears on the stopped frame without the experiment being selected
    again, so it follows the progress door and not only a selection. The
    failed experiment was read from a file by path, so its dataset is not
    stored: its row is present with Score greyed and the reason. The
    store's 404 for that digest is a console line caused on purpose."""
    collectors.append(MISSING_RESOURCE)
    _, digest = stored_digest(page, bench_url, [{"id": f"s{n}"} for n in range(4)])
    created = api_experiment(page, bench_url, digest, ["stub/fast"], unique("created"))
    done = api_experiment(page, bench_url, digest, ["stub/fast"], unique("done"))
    runs.append(done)
    assert page.request.post(
        f"{bench_url}/experiments/{done}/start", data={"dataset_digest": digest}
    ).ok
    wait_status(page, bench_url, done)
    path = tmp_path / "failing.jsonl"
    path.write_text('{"id": "f1", "prompt": "before"}\n', encoding="utf-8")
    failed = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("failed"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    path.write_text('{"id": "f1", "prompt": "after"}\n', encoding="utf-8")
    runs.append(failed)
    assert page.request.post(
        f"{bench_url}/experiments/{failed}/start", data={"dataset_path": str(path)}
    ).ok
    assert wait_status(page, bench_url, failed)["status"] == "failed"
    running = api_experiment(page, bench_url, digest, ["stub/slow"], unique("running"))
    runs.append(running)
    assert page.request.post(
        f"{bench_url}/experiments/{running}/start", data={"dataset_digest": digest}
    ).ok
    bench(["stub/fast"])
    open_experiments(page)
    start = page.get_by_test_id("experiment-start")
    stop = page.get_by_test_id("experiment-stop")
    score_row = page.get_by_test_id("experiment-score-row")
    score = page.get_by_test_id("experiment-score")

    row_for(page, done).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text("done")
    expect(start).to_be_disabled()
    expect(stop).to_be_hidden()
    expect(score_row).to_be_visible()
    expect(score).to_have_text("Score · free")
    expect(score).to_be_enabled()
    row_for(page, created).click()
    expect(start).to_be_enabled()
    expect(stop).to_be_hidden()
    expect(score_row).to_be_hidden()
    row_for(page, failed).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text(re.compile("^failed"))
    expect(start).to_be_disabled()
    expect(stop).to_be_hidden()
    expect(score_row).to_be_visible()
    expect(score).to_be_disabled()
    expect(page.get_by_test_id("experiment-score-nudge")).to_contain_text(
        "is not stored here"
    )
    row_for(page, running).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text("running")
    expect(start).to_be_disabled()
    expect(stop).to_be_visible()
    expect(score_row).to_be_hidden()
    stop.click()
    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "stopped · stopped between trials", timeout=DONE_TIMEOUT
    )
    expect(start).to_be_disabled()
    expect(stop).to_be_hidden()
    expect(score_row).to_be_visible()
    expect(score).to_have_text("Score · free")
    expect(score).to_be_enabled()


def test_start_is_pressed_once_and_a_started_experiment_stays_started(
    page, bench, bench_url, runs
):
    """WINDOW: Start pressed with its POST held and pressed again while
    held; then, after the answer, a reload of the list that still reads
    the experiment as created (the moment before the runner records it),
    with the progress door held so nothing moves it on.

    One press, one POST: Start is greyed while its request is out. And an
    experiment this tab started is not offered Start again because a
    list read says created: it runs once, and a second press would only
    be refused. Selected again after another row, it is watched again
    on the same read, because this tab started it."""
    digest = ready(page, bench, bench_url, ["stub/fast"], [{"id": "o1"}])
    eid = create(page, unique("once"))
    runs.append(eid)
    other = api_experiment(page, bench_url, digest, ["stub/fast"], unique("other"))
    log = progress_log(page)
    held = hold(page, f"**/experiments/{eid}/start", "POST")
    start = page.get_by_test_id("experiment-start")

    start.click()
    wait_held(page, held)
    page.evaluate(
        "() => document.querySelector('[data-testid=experiment-start]').click()"
    )
    page.wait_for_timeout(200)
    assert len(held) == 1
    expect(start).to_be_disabled()

    def still_created(route):
        body = route.fetch().json()
        for experiment in body["experiments"]:
            if experiment["id"] == eid:
                experiment["status"] = "created"
        route.fulfill(json=body)

    page.route("**/progress", lambda route: None)
    page.route(
        "**/experiments",
        lambda route: (
            still_created(route) if route.request.method == "GET" else route.continue_()
        ),
    )
    held.pop().continue_()
    expect(page.get_by_test_id("experiment-action-msg")).to_have_text("started")
    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    row_for(page, eid).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    expect(start).to_be_disabled()
    assert len(opened(log, eid)) == 1
    row_for(page, other).click()
    row_for(page, eid).click()
    for _ in range(40):
        if len(opened(log, eid)) == 2:
            break
        page.wait_for_timeout(50)
    assert len(opened(log, eid)) == 2
    page.unroute("**/experiments")
    wait_status(page, bench_url, eid)


def test_a_start_refusal_is_the_servers_sentence_and_the_panel_follows_it(
    page, bench, bench_url, collectors, runs
):
    """WINDOW: Start pressed in the page for an experiment another tab has
    already started and finished, the message after, and the panel.

    The door refuses a second start in its own words, and the page prints
    them as a refusal; the refusal names a status the page had not caught
    up with, so the list is read again and the panel then says done,
    with Start greyed and nothing more sent."""
    collectors.append(CONFLICT_RESOURCE)
    _, digest = stored_digest(page, bench_url, [{"id": "r1"}])
    eid = api_experiment(page, bench_url, digest, ["stub/fast"], unique("elsewhere"))
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    start = page.get_by_test_id("experiment-start")
    expect(start).to_be_enabled()
    runs.append(eid)
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok
    wait_status(page, bench_url, eid)
    posts = []
    page.on(
        "request",
        lambda r: (
            posts.append(r.url)
            if r.url.endswith("/start") and r.method == "POST"
            else None
        ),
    )

    start.click()

    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text(
        f"not started: experiment {eid} is done; an experiment runs once. "
        "Create another to run it again."
    )
    expect(message).to_have_attribute("data-state", "refused")
    expect(page.get_by_test_id("experiment-status")).to_have_text("done")
    expect(start).to_be_disabled()
    assert len(posts) == 1


def test_an_answer_lands_with_the_experiment_it_was_about(page, bench, bench_url, runs):
    """WINDOW: Start pressed for one experiment with its POST held, another
    row selected while it is held, the answer released, and the first
    selected again.

    The answer belongs to the experiment it was asked about. The other
    row shows nothing from it and is still what it was: created, with its
    Start live again once the request is over. The started experiment's
    row says running at once, and, selected again, it shows its word,
    runs, is watched, and offers Stop."""
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"w{n}"} for n in range(3)])
    started = create(page, unique("started"))
    runs.append(started)
    other = api_experiment(
        page,
        bench_url,
        record(page, bench_url, started)["dataset_digest"],
        ["stub/fast"],
        unique("other"),
    )
    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    row_for(page, started).click()
    held = hold(page, f"**/experiments/{started}/start", "POST")
    start = page.get_by_test_id("experiment-start")
    start.click()
    wait_held(page, held)
    row_for(page, other).click()

    held.pop().continue_()

    message = page.get_by_test_id("experiment-action-msg")
    expect(row_for(page, started).locator(".hcount")).to_have_text(
        re.compile(r"^running · ")
    )
    expect(message).to_have_text("")
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    expect(start).to_be_enabled()
    row_for(page, started).click()
    expect(message).to_have_text("started")
    expect(page.get_by_test_id("experiment-stop")).to_be_visible()
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-3] "), timeout=DONE_TIMEOUT
    )
    wait_status(page, bench_url, started)


def test_a_late_create_does_not_take_the_selection(page, bench, bench_url):
    """WINDOW: Create pressed with its POST held while a row is selected,
    another row chosen while it is held, and the answer released; then
    the same with the person moving away from the selected row and back.

    The person's later choice stands, in both shapes: the new experiment
    is created and listed, and the selection stays on the row they chose,
    so a Start pressed next starts that one and not the new one. Even
    returning to the row selected at the press is a choice made after it."""
    digest = ready(page, bench, bench_url, ["stub/fast"], [{"id": "c1"}])
    first = api_experiment(page, bench_url, digest, ["stub/fast"], unique("first"))
    chosen = api_experiment(page, bench_url, digest, ["stub/fast"], unique("chosen"))
    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    held = hold(page, "**/experiments", "POST")
    title = page.get_by_test_id("experiment-title")

    for moves in ([first, chosen], [chosen, first, chosen]):
        row_for(page, moves[0]).click()
        page.get_by_test_id("experiment-name").fill(unique("late"))
        page.get_by_test_id("experiment-create-button").click()
        wait_held(page, held)
        for eid in moves[1:]:
            row_for(page, eid).click()

        held.pop().continue_()

        expect(page.get_by_test_id("experiment-create-msg")).to_have_text(
            re.compile(r"^created experiment \d+\.")
        )
        new = created_id(page)
        expect(row_for(page, new)).to_have_attribute("aria-pressed", "false")
        expect(row_for(page, chosen)).to_have_attribute("aria-pressed", "true")
        expect(title).to_contain_text(f"experiment {chosen}")


def test_each_experiment_shows_its_own_projection(page, bench, bench_url):
    """WINDOW: the projection beside Start for two experiments created in
    the page over different lineups, each selected in turn.

    The projection is the one Create returned for THAT experiment: a
    lookup by anything else would show one experiment's cost beside
    another's Start."""
    ready(page, bench, bench_url, ["stub/fast", "stub/nousage"], [{"id": "p1"}])
    projections = {}
    for label, clicks in (("both", 0), ("fast only", 1)):
        if clicks:
            page.get_by_test_id("lineup-chip").nth(1).click()
        page.get_by_test_id("experiment-name").fill(unique(label))
        with page.expect_response(
            lambda r: r.url.endswith("/experiments") and r.request.method == "POST"
        ) as answer:
            page.get_by_test_id("experiment-create-button").click()
        body = answer.value.json()
        projections[body["id"]] = page.evaluate(
            "(p) => BenchLib.projectionText(p)", body["projected_cost"]
        )
    first, second = projections
    assert projections[first] != projections[second]
    projection = page.get_by_test_id("experiment-projection")

    for eid in (first, second, first):
        row_for(page, eid).click()
        expect(projection).to_have_text(projections[eid])


def test_progress_is_watched_exactly_while_it_runs(page, bench, bench_url, runs):
    """WINDOW: the progress requests the page makes as a running
    experiment is selected, another (created) row is selected, and the
    running one is selected again.

    Selecting a running experiment opens its stream and its counters
    move; moving the selection off it closes the stream (the request
    ends, aborted by the page); selecting a created experiment opens
    none, because the door would hold it open until someone starts it;
    a reload of the list while it is watched opens no second stream; and
    selecting the running one again opens it again."""
    _, digest = stored_digest(page, bench_url, [{"id": f"m{n}"} for n in range(4)])
    running = api_experiment(page, bench_url, digest, ["stub/slow"], unique("running"))
    runs.append(running)
    created = api_experiment(page, bench_url, digest, ["stub/fast"], unique("idle"))
    assert page.request.post(
        f"{bench_url}/experiments/{running}/start", data={"dataset_digest": digest}
    ).ok
    bench(["stub/fast"])
    log = progress_log(page)
    open_experiments(page)

    row_for(page, running).click()

    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-4] "), timeout=DONE_TIMEOUT
    )
    assert len(opened(log, running)) == 1
    # The row moves with the frames, not only when the list is read again.
    expect(row_for(page, running).locator(".hcount")).to_have_text(
        re.compile(r"^running · [1-4]/4 trials$")
    )
    page.evaluate("() => BenchLifecycle.refresh()")
    page.wait_for_timeout(500)
    assert (len(opened(log, running)), closed(log, running)) == (1, [])
    row_for(page, created).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    for _ in range(40):
        if closed(log, running):
            break
        page.wait_for_timeout(50)
    assert len(closed(log, running)) == 1
    page.wait_for_timeout(1_000)
    assert opened(log, created) == []
    row_for(page, running).click()
    for _ in range(40):
        if len(opened(log, running)) == 2:
            break
        page.wait_for_timeout(50)
    assert len(opened(log, running)) == 2


def test_a_reload_watches_an_experiment_that_started_elsewhere(
    page, bench, bench_url, runs
):
    """WINDOW: a created experiment selected in the page, started by
    another tab, and the list read again.

    The reload says it runs now, so its progress is watched as selecting
    it would: the counters move and Stop is offered."""
    _, digest = stored_digest(page, bench_url, [{"id": f"x{n}"} for n in range(4)])
    eid = api_experiment(page, bench_url, digest, ["stub/slow"], unique("elsewhere"))
    runs.append(eid)
    bench(["stub/fast"])
    open_experiments(page)
    row_for(page, eid).click()
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok

    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)

    expect(page.get_by_test_id("experiment-stop")).to_be_visible()
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-4] "), timeout=DONE_TIMEOUT
    )


def test_the_attachments_mode_chosen_is_sent_and_recorded(page, bench, bench_url):
    """WINDOW: the body Create sends and the row it records, over a
    dataset that cites an image, with native chosen, and then inline over
    one that cites a text document.

    The control is enabled where the dataset cites a document, and what
    it shows is what is sent and recorded."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    image = page.request.post(
        bench_url + "/attachments",
        data={
            "filename": f"pixel-{uuid.uuid4().hex[:6]}.png",
            "content_base64": base64.b64encode(png).decode(),
        },
    ).json()["digest"]
    text = page.request.post(
        bench_url + "/attachments",
        data={
            "filename": "note.txt",
            "content_base64": base64.b64encode(uuid.uuid4().hex.encode()).decode(),
        },
    ).json()["digest"]
    images, _ = stored_digest(page, bench_url, [{"id": "i1", "attachments": [image]}])
    texts, _ = stored_digest(page, bench_url, [{"id": "t1", "attachments": [text]}])
    bench(["stub/vision"])
    check_all_chips(page)
    select_dataset(page, images)
    open_experiments(page)
    attachments = page.get_by_test_id("experiment-attachments")
    expect(attachments).to_be_enabled()
    attachments.select_option("native")
    page.get_by_test_id("experiment-name").fill(unique("native"))

    body = create_body(page)

    assert body["attachments_mode"] == "native"
    assert record(page, bench_url, created_id(page))["attachments_mode"] == "native"
    entry_for(page, texts).click()
    attachments.select_option("inline")
    page.get_by_test_id("experiment-name").fill(unique("inline"))
    body = create_body(page)
    assert body["attachments_mode"] == "inline"
    assert record(page, bench_url, created_id(page))["attachments_mode"] == "inline"


def test_the_budget_is_the_composers(page, bench, bench_url):
    """WINDOW: the composer's budget set to extended, the form's sources
    line, the body Create sends and the row it records.

    One budget control on the page: the experiment takes the composer's,
    read when Create is pressed."""
    ready(page, bench, bench_url, ["stub/fast"], [{"id": "b1"}])
    page.get_by_test_id("budget-extended").click()
    expect(page.get_by_test_id("experiment-source-budget")).to_have_text(
        "budget: extended"
    )
    page.get_by_test_id("experiment-name").fill(unique("extended"))

    body = create_body(page)

    assert body["budget"] == "extended"
    assert record(page, bench_url, created_id(page))["budget"] == "extended"


def test_a_list_reload_keeps_focus_and_a_full_list_says_so(page, bench, bench_url):
    """WINDOW: a row with keyboard focus when the list is read again, and
    the list when GET /experiments answers with 100 experiments and with
    fewer.

    A reload replaces the rows (the request is made, and counted here, so
    the proof is of a reload and not of nothing happening); focus goes
    back to the row for the same experiment. The list is the newest 100,
    and a full answer says so."""
    _, digest = stored_digest(page, bench_url, [{"id": "f1"}])
    eid = api_experiment(page, bench_url, digest, ["stub/fast"], unique("focus"))
    bench(["stub/fast"])
    open_experiments(page)
    reads = []
    page.on(
        "request",
        lambda r: (
            reads.append(r.url)
            if r.url.endswith("/experiments") and r.method == "GET"
            else None
        ),
    )
    row_for(page, eid).focus()

    page.evaluate("() => BenchLifecycle.refresh()")

    expect(page.get_by_test_id("experiment-list")).to_have_attribute(
        "data-state", "ready"
    )
    assert len(reads) == 1
    assert page.evaluate("() => document.activeElement.dataset.id") == str(eid)
    listed = page.request.get(bench_url + "/experiments").json()["experiments"]
    template = listed[0]
    full = [dict(template, id=n + 100000, name=f"bulk {n}") for n in range(100)]
    answer = {"body": json.dumps({"experiments": full})}
    page.route(
        "**/experiments",
        lambda route: (
            route.fulfill(
                status=200, content_type="application/json", body=answer["body"]
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )
    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    note = page.get_by_test_id("experiment-list-note")
    expect(note).to_have_text("The newest 100 are listed; older ones stay in bench.db.")
    answer["body"] = json.dumps({"experiments": full[:99]})
    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    expect(note).to_have_text("")


def test_a_dataset_that_leaves_the_library_leaves_the_form(page, bench, bench_url):
    """WINDOW: the form's dataset line after the Datasets library is read
    again without the selected dataset in it.

    The library drops a selection it no longer lists, and the form is
    told: it says none is selected and waits for one, rather than
    creating over a digest nothing on screen describes."""
    digest = ready(page, bench, bench_url, ["stub/fast"], [{"id": "g1"}])
    page.get_by_test_id("experiment-name").fill(unique("gone"))
    expect(page.get_by_test_id("experiment-source-dataset")).to_contain_text(digest[:7])
    page.route(
        "**/datasets?limit=500",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "datasets": [
                        d
                        for d in route.fetch().json()["datasets"]
                        if d["digest"] != digest
                    ]
                }
            ),
        ),
    )

    page.evaluate("() => BenchDatasets.refresh()")

    expect(page.get_by_test_id("experiment-source-dataset")).to_have_text(
        "dataset: none selected"
    )
    expect(page.get_by_test_id("experiment-create-nudge")).to_have_text(
        "Create waits: select a stored dataset in Datasets above"
    )


def test_back_brings_back_no_choice(page, bench, bench_url):
    """WINDOW: every control of the form set, a navigation away, and Back.

    NOTHING IS PRE-FILLED, on a history navigation too: the browser's
    form restoration would refill the name, the boxes, the selects and
    the checkbox after the page had painted, and the estimand note would
    then disagree with the select beside it."""
    ready(page, bench, bench_url, ["stub/fast"], [{"id": "h1"}])
    page.get_by_test_id("experiment-name").fill("restored")
    page.get_by_test_id("experiment-repeats").fill("3")
    page.get_by_test_id("experiment-seed").fill("4")
    page.get_by_test_id("experiment-estimand").select_option("underlying_model")
    page.get_by_test_id("experiment-halt").uncheck()
    page.goto(bench_url + "/models")

    page.go_back()

    page.get_by_test_id("experiments-toggle").click()
    expect(page.get_by_test_id("experiment-name")).to_have_value("")
    expect(page.get_by_test_id("experiment-repeats")).to_have_value("")
    expect(page.get_by_test_id("experiment-seed")).to_have_value("")
    expect(page.get_by_test_id("experiment-estimand")).to_have_value("routed_service")
    expect(page.get_by_test_id("experiment-estimand-note")).to_have_text("")
    expect(page.get_by_test_id("experiment-halt")).to_be_checked()


def test_the_panels_controls_are_named_and_described(page, bench, bench_url):
    """WINDOW: the accessible names and descriptions of the form's
    controls and Start, before and after the estimand note has text, and
    which lines are live regions.

    Each select is named by its own caption alone (not "estimand" plus
    the drawn arrow, and not its note's sentence once the note has one),
    the estimand and attachments selects are described by the sentence
    that explains them and the metric by none, and Start by its
    projection and its note. The lines that report an outcome are polite
    live regions (role=status, not silenced by aria-live="off"); the
    counters, which change every trial, are not."""
    ready(page, bench, bench_url, ["stub/fast"], [{"id": "a1"}])
    names = {
        "estimand": "experiment-estimand",
        "attachments": "experiment-attachments",
        "primary metric": "experiment-metric",
    }

    def names_hold():
        for name, testid in names.items():
            control = page.get_by_role("combobox", name=name, exact=True)
            expect(control).to_have_count(1)
            expect(control).to_have_attribute("data-testid", testid)

    names_hold()
    page.get_by_test_id("experiment-estimand").select_option("underlying_model")
    note = page.get_by_test_id("experiment-estimand-note")
    expect(note).to_be_visible()
    names_hold()
    expect(page.get_by_test_id("experiment-estimand")).to_have_attribute(
        "aria-describedby", "experiment-estimand-note"
    )
    expect(page.get_by_test_id("experiment-attachments")).to_have_attribute(
        "aria-describedby", "experiment-attachments-note"
    )
    assert (
        page.get_by_test_id("experiment-metric").get_attribute("aria-describedby")
        is None
    )
    expect(page.get_by_test_id("experiment-attachments-note")).to_be_visible()
    expect(page.get_by_test_id("experiment-start")).to_have_attribute(
        "aria-describedby", "experiment-projection experiment-start-note"
    )
    for testid in (
        "experiment-create-nudge",
        "experiment-create-msg",
        "experiment-status",
        "experiment-stream",
        "experiment-action-msg",
    ):
        line = page.get_by_test_id(testid)
        expect(line).to_have_attribute("role", "status")
        assert line.get_attribute("aria-live") in (None, "polite"), testid
    for testid in ("experiment-counters", "experiment-projection"):
        assert page.get_by_test_id(testid).get_attribute("role") is None


def test_a_row_says_how_its_trials_finished(page, bench, bench_url, runs):
    """WINDOW: the list row of an experiment whose trials fail, as the
    watched run moves and after it finishes.

    A row counts trials FINISHED, the three buckets added up, and names
    the failures, and the watched run's frames move it as they arrive."""
    ready(page, bench, bench_url, ["stub/nopolicy"], [{"id": "q1"}, {"id": "q2"}])
    eid = create(page, unique("failing"))
    runs.append(eid)

    page.get_by_test_id("experiment-start").click()

    expect(row_for(page, eid).locator(".hcount")).to_have_text(
        "done · 2/2 trials (2 failed)", timeout=DONE_TIMEOUT
    )
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        "done 0 · failed 2 · refused 0 · of 2 trials"
    )


def test_an_experiment_from_a_file_is_not_offered_start(
    page, bench, bench_url, tmp_path, collectors
):
    """WINDOW: a created experiment whose dataset was read from a file by
    path, selected in the panel, and the start door sent the page's
    request anyway.

    Start sends the recorded digest, and the store does not hold this
    one, so the door always refuses it: the page greys Start and says to
    start it through the API by path. The greyed button is a courtesy;
    the door refuses the digest on its own."""
    path = tmp_path / "by-path.jsonl"
    path.write_text(
        '{"id": "p1", "prompt": "' + uuid.uuid4().hex + '"}\n', encoding="utf-8"
    )
    created = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()
    eid = created["id"]
    digest = record(page, bench_url, eid)["dataset_digest"]
    bench(["stub/fast"])
    collectors.append(MISSING_RESOURCE)
    open_experiments(page)

    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-start-note")).to_have_text(
        "its dataset was read from a file and is not stored here, so the page "
        "cannot name it; start it through the API with its dataset_path"
    )
    expect(page.get_by_test_id("experiment-start")).to_be_disabled()
    expect(page.get_by_test_id("experiment-projection")).to_have_text("")
    refused = page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    )
    assert refused.status == 422
    assert refused.json()["detail"].startswith("no stored dataset has digest")


def test_a_lost_create_answer_reloads_the_list(page, bench, bench_url, collectors):
    """WINDOW: Create whose request reaches the server and whose answer is
    lost, the message after, and the list.

    The page cannot know whether the experiment exists, says so, and
    reads the list again, which shows that it does."""
    collectors.append(LOST_RESOURCE)
    collectors.append("bench: creating an experiment failed")
    ready(page, bench, bench_url, ["stub/fast"], [{"id": "z1"}])
    name = unique("lost answer")
    page.get_by_test_id("experiment-name").fill(name)

    def reach_then_lose(route):
        if route.request.method != "POST":
            route.continue_()
            return
        route.fetch()
        route.abort("connectionreset")

    page.route("**/experiments", reach_then_lose)
    page.get_by_test_id("experiment-create-button").click()

    expect(page.get_by_test_id("experiment-create-msg")).to_have_text(
        re.compile(
            r"^no answer came back \(.+\); the experiment may exist, and the list "
            r"below will say\.$"
        )
    )
    expect(page.get_by_test_id("experiment-row").filter(has_text=name)).to_have_count(1)


def test_the_command_bar_says_it_counts_the_composer(page, bench):
    """WINDOW: the command bar's spend label and tooltip on a fresh page,
    and the tooltip after one composer run.

    The bar's spend is the composer's runs, and says so on its face: an
    experiment started from the panel is not counted there, and the
    tooltip says where its cost is, whether or not anything has run."""
    bench(["stub/fast"])
    check_all_chips(page)
    suffix = ". The composer's runs only: an experiment's cost is in its report"
    stat = page.locator("#stat-spend")
    expect(stat.locator(".k")).to_have_text("composer spend")
    spend = page.get_by_test_id("stat-spend")
    assert spend.get_attribute("title") == (
        "no composer run priced yet this session" + suffix
    )
    page.get_by_test_id("prompt-input").fill("spend " + uuid.uuid4().hex)
    page.get_by_test_id("run-button").click()
    expect(spend).not_to_have_attribute(
        "title", re.compile("^no composer run priced"), timeout=DONE_TIMEOUT
    )
    assert spend.get_attribute("title").endswith(suffix)


def test_a_blank_composer_run_sends_no_params(page, bench):
    """WINDOW: the bodies of the composer's two requests (the group it
    records and the stream it reads) with every control blank, after
    hasControls moved from stream.js to lib.js.

    Rule one on the composer's wire, at both call sites: no params key,
    not an empty object. The move is N3's, so N3 holds it."""
    bench(["stub/fast"])
    check_all_chips(page)
    page.get_by_test_id("prompt-input").fill("blank controls " + uuid.uuid4().hex)
    with page.expect_request(
        lambda r: r.url.endswith("/groups") and r.method == "POST"
    ) as group:
        with page.expect_request(
            lambda r: "/compare/stream" in r.url and r.method == "POST"
        ) as stream:
            page.get_by_test_id("run-button").click()
    assert "params" not in json.loads(group.value.post_data)
    assert "params" not in json.loads(stream.value.post_data)


def test_a_refused_start_that_leaves_it_created_leaves_start_live(
    page, bench, bench_url, runs, collectors
):
    """WINDOW: Start pressed from the keyboard while another experiment
    holds the runner, the refusal, a second press once the runner is
    free, and a second experiment created and started in the same tab.

    A refused Start is not a started one: the door says why in its own
    words, the experiment is still created, Start is live again and
    focus is on its row, and pressing it once the runner is free starts
    it. A tab that has pressed Start once can press it again for another
    experiment."""
    collectors.append(CONFLICT_RESOURCE)
    digest = ready(page, bench, bench_url, ["stub/fast"], [{"id": "r1"}])
    mine = create(page, unique("mine"))
    runs.append(mine)
    _, slow_digest = stored_digest(page, bench_url, [{"id": f"b{n}"} for n in range(6)])
    busy = api_experiment(page, bench_url, slow_digest, ["stub/slow"], unique("busy"))
    runs.append(busy)
    assert page.request.post(
        f"{bench_url}/experiments/{busy}/start", data={"dataset_digest": slow_digest}
    ).ok
    start = page.get_by_test_id("experiment-start")
    message = page.get_by_test_id("experiment-action-msg")

    start.focus()
    page.keyboard.press("Enter")

    expect(message).to_have_text(
        f"not started: experiment {busy} is already running. One at a time: "
        "they share the upstream slots and the spend ceiling, so concurrent "
        "experiments would measure each other."
    )
    expect(message).to_have_attribute("data-state", "refused")
    expect(message).to_be_visible()
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    expect(start).to_be_enabled()
    assert page.evaluate("() => document.activeElement.dataset.id") == str(mine)
    drain(page, bench_url, busy)
    start.click()
    expect(message).to_have_text("started")
    wait_status(page, bench_url, mine)
    second = create(page, unique("second"))
    runs.append(second)
    start.click()
    expect(message).to_have_text("started")
    assert wait_status(page, bench_url, second)["dataset_digest"] == digest


def test_a_failed_store_question_leaves_start_live(
    page, bench, bench_url, tmp_path, collectors
):
    """WINDOW: a created experiment whose dataset the store does not hold,
    selected while GET /datasets/{digest} is held, then answered 500,
    then failed at the network, and then answered by the store itself.

    Start is live while the question is out and after any answer that is
    not a 404: an unknown is not a no, and the door is the rule. The note
    stays empty, because nothing was learnt. A failed question is
    forgotten rather than kept, so the next selection asks again, and
    when the store answers 404 the page greys Start and says why."""
    collectors.extend(
        [
            "Failed to load resource: the server responded with a status of 500",
            ABORTED_RESOURCE,
            MISSING_RESOURCE,
            "bench: checking a stored dataset failed",
        ]
    )
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        '{"id": "q1", "prompt": "' + uuid.uuid4().hex + '"}\n', encoding="utf-8"
    )
    eid = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("unknown"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    digest = record(page, bench_url, eid)["dataset_digest"]
    _, other_digest = stored_digest(page, bench_url, [{"id": "q2"}])
    other = api_experiment(page, bench_url, other_digest, ["stub/fast"])
    bench(["stub/fast"])
    held = hold(page, f"**/datasets/{digest}", "GET")
    open_experiments(page)
    start = page.get_by_test_id("experiment-start")
    note = page.get_by_test_id("experiment-start-note")

    row_for(page, eid).click()
    wait_held(page, held)
    expect(start).to_be_enabled()
    held.pop().fulfill(status=500, body="no")
    expect(start).to_be_enabled()
    expect(note).to_have_text("")
    page.unroute(f"**/datasets/{digest}")
    aborted = []
    page.route(
        f"**/datasets/{digest}",
        lambda route: (aborted.append(route.request.url), route.abort()),
    )
    row_for(page, other).click()
    row_for(page, eid).click()
    wait_held(page, aborted)
    expect(start).to_be_enabled()
    expect(note).to_have_text("")
    page.unroute(f"**/datasets/{digest}")
    row_for(page, other).click()
    row_for(page, eid).click()
    expect(start).to_be_disabled()
    expect(note).to_contain_text("is not stored here")


def test_a_file_stored_later_is_offered_start(
    page, bench, bench_url, tmp_path, collectors
):
    """WINDOW: a created experiment read from a file by path, selected
    (Start greyed, its dataset not stored), the same file then stored,
    the experiment selected again; and a new experiment created in the
    page over the stored dataset.

    A no from the store is not final: storing that very file is what the
    door's own refusal asks for, and once the store holds it Start is
    live, the note gone. An experiment the page creates over the stored
    dataset is offered Start with its projection."""
    collectors.append(MISSING_RESOURCE)
    content = '{"id": "p1", "prompt": "' + uuid.uuid4().hex + '"}\n'
    path = tmp_path / "later.jsonl"
    path.write_text(content, encoding="utf-8")
    eid = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    bench(["stub/fast"])
    check_all_chips(page)
    open_experiments(page)
    start = page.get_by_test_id("experiment-start")
    note = page.get_by_test_id("experiment-start-note")
    row_for(page, eid).click()
    expect(note).to_be_visible()
    expect(start).to_be_disabled()
    name = unique("the file")
    stored = page.request.post(
        bench_url + "/datasets", data={"name": name, "content": content}
    )
    assert (
        stored.ok
        and stored.json()["digest"] == record(page, bench_url, eid)["dataset_digest"]
    )

    page.get_by_test_id("experiments-toggle").click()
    open_experiments(page)
    row_for(page, eid).click()

    expect(note).to_have_text("")
    expect(start).to_be_enabled()
    select_dataset(page, name)
    create(page, unique("over the store"))
    expect(start).to_be_enabled()
    expect(page.get_by_test_id("experiment-projection")).to_contain_text("total $")


def test_a_lost_start_answer_reads_the_list_and_watches(
    page, bench, bench_url, runs, collectors
):
    """WINDOW: Start whose request reaches the door and whose answer is
    lost; then, for another experiment, Start whose request never
    reaches the door, and a second press.

    The page cannot know whether the first started, says so, and reads
    the list, which says it runs: its progress is watched and Stop is
    offered. The second never left: it is still created, Start is live,
    and a second press starts it."""
    collectors.extend([LOST_RESOURCE, "bench: starting an experiment failed"])
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"l{n}"} for n in range(3)])
    lost = create(page, unique("lost"))
    runs.append(lost)

    def reach_then_lose(route):
        route.fetch()
        route.abort("connectionreset")

    page.route(f"**/experiments/{lost}/start", reach_then_lose)
    page.get_by_test_id("experiment-start").click()

    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text(
        re.compile(
            r"^no answer came back \(.+\); the experiment may have started, "
            r"and its status will say\.$"
        )
    )
    expect(page.get_by_test_id("experiment-status")).to_have_text("running")
    expect(page.get_by_test_id("experiment-stop")).to_be_visible()
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-3] "), timeout=DONE_TIMEOUT
    )
    wait_status(page, bench_url, lost)
    never = create(page, unique("never"))
    runs.append(never)
    page.route(f"**/experiments/{never}/start", lambda route: route.abort())
    start = page.get_by_test_id("experiment-start")
    start.click()
    expect(message).to_have_text(re.compile(r"^no answer came back"))
    expect(page.get_by_test_id("experiment-status")).to_have_text("created")
    expect(start).to_be_enabled()
    page.unroute(f"**/experiments/{never}/start")
    start.click()
    expect(message).to_have_text("started")
    wait_status(page, bench_url, never)


def test_a_started_experiment_read_as_created_is_watched(page, bench, bench_url, runs):
    """WINDOW: Start with its POST held, the list read again during the
    hold (so the panel's object for the experiment is replaced and still
    says created), and the answer released.

    The 202 lands on the experiment as the panel now holds it: with the
    progress door held, so no frame can do it instead, the status says
    running, its row says so, Stop is offered and the stream is asked
    for. Released, the counters move."""
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"h{n}"} for n in range(3)])
    eid = create(page, unique("read as created"))
    runs.append(eid)
    progress = hold(page, f"**/experiments/{eid}/progress", "GET")
    held = hold(page, f"**/experiments/{eid}/start", "POST")
    page.get_by_test_id("experiment-start").click()
    wait_held(page, held)
    page.evaluate("() => BenchLifecycle.refresh()")
    expect(page.get_by_test_id("experiment-list")).to_have_attribute(
        "data-state", "ready"
    )

    held.pop().continue_()

    expect(page.get_by_test_id("experiment-status")).to_have_text("running")
    expect(row_for(page, eid).locator(".hcount")).to_have_text(
        re.compile(r"^running · ")
    )
    expect(page.get_by_test_id("experiment-stop")).to_be_visible()
    wait_held(page, progress)
    progress.pop().continue_()
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-3] "), timeout=DONE_TIMEOUT
    )
    wait_status(page, bench_url, eid)


def test_a_refused_stop_is_the_servers_sentence(
    page, bench, bench_url, runs, collectors
):
    """WINDOW: Stop pressed from the keyboard for an experiment the page
    still shows running (its progress held) after the server finished
    it, the refusal, and the panel after.

    The door refuses to stop what is not running, in its own words, and
    the page prints them as a refusal, reads the list again (which says
    done), hides Stop, and puts focus on the experiment's row."""
    collectors.append(CONFLICT_RESOURCE)
    _, digest = stored_digest(page, bench_url, [{"id": f"z{n}"} for n in range(2)])
    eid = api_experiment(page, bench_url, digest, ["stub/slow"], unique("finishing"))
    runs.append(eid)
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok
    bench(["stub/fast"])
    page.route(f"**/experiments/{eid}/progress", lambda route: None)
    open_experiments(page)
    row_for(page, eid).click()
    stop = page.get_by_test_id("experiment-stop")
    expect(stop).to_be_visible()
    wait_status(page, bench_url, eid)

    stop.focus()
    page.keyboard.press("Enter")

    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text(f"not stopped: experiment {eid} is not running")
    expect(message).to_have_attribute("data-state", "refused")
    expect(page.get_by_test_id("experiment-status")).to_have_text("done")
    expect(stop).to_be_hidden()
    assert page.evaluate("() => document.activeElement.dataset.id") == str(eid)


def test_a_lost_stop_answer_reads_the_list(page, bench, bench_url, runs, collectors):
    """WINDOW: Stop whose request reaches the door and whose answer is
    lost, the message, and the requests for the list after it.

    The page cannot know whether the stop took, says so, and reads the
    list again as a lost Start or Create answer does; the stream it
    watches then carries the experiment to stopped."""
    collectors.extend([LOST_RESOURCE, "bench: stopping an experiment failed"])
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"v{n}"} for n in range(5)])
    eid = create(page, unique("lost stop"))
    runs.append(eid)
    page.get_by_test_id("experiment-start").click()
    stop = page.get_by_test_id("experiment-stop")
    expect(stop).to_be_visible()
    reads = []
    page.on(
        "request",
        lambda r: (
            reads.append(r.url)
            if r.url.endswith("/experiments") and r.method == "GET"
            else None
        ),
    )

    def reach_then_lose(route):
        route.fetch()
        route.abort("connectionreset")

    page.route(f"**/experiments/{eid}/stop", reach_then_lose)
    stop.click()

    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        re.compile(r"^no answer came back \(.+\); its status will say\.$")
    )
    for _ in range(40):
        if reads:
            break
        page.wait_for_timeout(50)
    assert len(reads) == 1
    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "stopped · stopped between trials", timeout=DONE_TIMEOUT
    )


def test_the_stopping_line_goes_when_a_reload_says_stopped(
    page, bench, bench_url, runs
):
    """WINDOW: Stop pressed, another row selected (which closes the
    stream), the experiment stopping on the server meanwhile, the list
    read again, and the stopped experiment selected again.

    The present-tense word is done with once the experiment shows
    finished, however the panel learnt it: here no finished frame ever
    reaches the page, and the reload alone must take the line away."""
    digest = ready(
        page, bench, bench_url, ["stub/slow"], [{"id": f"y{n}"} for n in range(5)]
    )
    other = api_experiment(page, bench_url, digest, ["stub/fast"], unique("other"))
    eid = create(page, unique("stopping"))
    runs.append(eid)
    page.get_by_test_id("experiment-start").click()
    stop = page.get_by_test_id("experiment-stop")
    expect(stop).to_be_visible()
    stop.click()
    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_text("stopping after the trial in flight")
    row_for(page, other).click()
    expect(message).to_have_text("")
    assert wait_status(page, bench_url, eid)["status"] == "stopped"

    page.evaluate("() => BenchLifecycle.refresh()")
    row_for(page, eid).click()

    expect(page.get_by_test_id("experiment-status")).to_have_text(
        "stopped · stopped between trials"
    )
    expect(message).to_have_text("")


def focused_testid(page):
    return page.evaluate("() => document.activeElement.dataset.testid || ''")


def test_focus_after_create_start_and_stop(page, bench, bench_url, runs):
    """WINDOW: Create, Start and Stop each pressed from the keyboard, and
    what has focus after each.

    Each control is greyed or hidden while its request is out, which
    drops focus to the page; it comes back to Create after Create (live
    again, with nothing to wait for), and to the experiment's row after
    Start and after Stop, which is neither Start nor Stop, so a second
    Enter moves no money and stops nothing."""
    ready(page, bench, bench_url, ["stub/slow"], [{"id": f"k{n}"} for n in range(4)])
    page.get_by_test_id("experiment-name").fill(unique("keyboard"))
    create_button = page.get_by_test_id("experiment-create-button")

    create_button.focus()
    page.keyboard.press("Enter")

    expect(page.get_by_test_id("experiment-create-msg")).to_have_text(
        re.compile(r"^created experiment \d+\.")
    )
    eid = created_id(page)
    runs.append(eid)
    expect(create_button).to_be_enabled()
    expect(page.get_by_test_id("experiment-create-nudge")).to_have_text("")
    assert focused_testid(page) == "experiment-create-button"
    page.get_by_test_id("experiment-start").focus()
    page.keyboard.press("Enter")
    expect(page.get_by_test_id("experiment-action-msg")).to_have_text("started")
    assert page.evaluate("() => document.activeElement.dataset.id") == str(eid)
    stop = page.get_by_test_id("experiment-stop")
    expect(stop).to_be_visible()
    stop.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_test_id("experiment-action-msg")).to_have_text(
        "stopping after the trial in flight"
    )
    assert page.evaluate("() => document.activeElement.dataset.id") == str(eid)
    wait_status(page, bench_url, eid)


def test_an_unreadable_create_answer_reloads_the_list(page, bench, bench_url):
    """WINDOW: Create whose request reaches the door and whose 201 comes
    back with a body the page cannot read, the message, and the list.

    The page cannot know the new id, says the experiment may exist, and
    reads the list, which shows it does."""
    ready(page, bench, bench_url, ["stub/fast"], [{"id": "u1"}])
    name = unique("unreadable")
    page.get_by_test_id("experiment-name").fill(name)

    def created_but_unreadable(route):
        if route.request.method != "POST":
            route.continue_()
            return
        route.fetch()
        route.fulfill(status=201, content_type="application/json", body="not json")

    page.route("**/experiments", created_but_unreadable)
    page.get_by_test_id("experiment-create-button").click()

    expect(page.get_by_test_id("experiment-create-msg")).to_have_text(
        "the bench answered 201 with a body this page could not read; the "
        "experiment may exist, and the list below will say."
    )
    expect(page.get_by_test_id("experiment-row").filter(has_text=name)).to_have_count(1)


def test_a_dataset_path_applied_for_one_report_stays_with_it(
    page, bench, bench_url, tmp_path, runs, collectors
):
    """WINDOW: the report requests the page makes for A (created from a
    file by path, the path applied in the report's box), then for B (over
    a stored dataset), then for C (created in the page over a stored
    dataset), then for A again, and what each answered.

    A path is the file ONE experiment was read from. Applied for A it is
    sent with A's report and never with another's: B's and C's are asked
    for with no path and read their dataset from the store, where sending
    A's path carried A's input into their requests and opened them on a
    false "dataset changed" refusal. A, opened again, keeps its own.
    PRE-STATE: A's read carries the path. A was read from a file, so the
    store answers 404 when the page asks about its dataset, a console
    line caused on purpose."""
    collectors.append(MISSING_RESOURCE)
    path = tmp_path / "by-path.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "t1",
                "prompt": uuid.uuid4().hex,
                "reference": "x",
                "scorer": {"kind": "exact"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    a = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    runs.append(a)
    assert page.request.post(
        f"{bench_url}/experiments/{a}/start", data={"dataset_path": str(path)}
    ).ok
    wait_status(page, bench_url, a)
    b_name, b_digest = stored_digest(
        page, bench_url, [{"id": "s1", "reference": "x", "scorer": {"kind": "exact"}}]
    )
    b = api_experiment(page, bench_url, b_digest, ["stub/fast"], unique("stored"))
    runs.append(b)
    assert page.request.post(
        f"{bench_url}/experiments/{b}/start", data={"dataset_digest": b_digest}
    ).ok
    wait_status(page, bench_url, b)
    bench(["stub/fast"])
    check_all_chips(page)
    open_experiments(page)
    panel = page.get_by_test_id("report-panel")

    def report_for(matches, act):
        with page.expect_response(
            lambda r: r.request.method == "GET" and matches(r.url)
        ) as answer:
            act()
        expect(panel).to_have_attribute("data-state", "ready")
        return answer.value

    def of(eid):
        return lambda url: f"/experiments/{eid}/report" in url

    def of_another(url):
        return "/report" in url and not of(a)(url) and not of(b)(url)

    row_for(page, a).click()
    expect(panel).to_have_attribute("data-state", "ready")
    page.get_by_test_id("report-dataset-path").fill(str(path))
    applied = report_for(
        of(a), lambda: page.get_by_test_id("report-dataset-apply").click()
    )
    assert "dataset_path=" in applied.url
    assert applied.json()["thresholds_source"] == "dataset_file"

    stored = report_for(of(b), lambda: row_for(page, b).click())

    assert "dataset_path" not in stored.url
    assert stored.json()["thresholds_source"] == "dataset_store"
    expect(page.get_by_test_id("report-dataset-path")).to_have_value("")
    select_dataset(page, b_name)
    created = report_for(of_another, lambda: create(page, unique("created")))
    assert "dataset_path" not in created.url
    assert created.json()["thresholds_source"] == "dataset_store"
    again = report_for(of(a), lambda: row_for(page, a).click())
    assert "dataset_path=" in again.url
    expect(page.get_by_test_id("report-dataset-path")).to_have_value(str(path))


def test_a_late_report_does_not_stack_under_a_newer_one(page, bench, bench_url):
    """WINDOW: a row clicked with its report request held, another row
    clicked, and the first report's answer released.

    A report whose answer comes back after a newer one was asked for is
    dropped: the panel holds one report, the newer experiment's."""
    _, digest = stored_digest(page, bench_url, [{"id": "s1"}])
    names = [unique("older report"), unique("newer report")]
    ids = [api_experiment(page, bench_url, digest, ["stub/fast"], n) for n in names]
    bench(["stub/fast"])
    open_experiments(page)
    held = hold(page, f"**/experiments/{ids[0]}/report", "GET")
    row_for(page, ids[0]).click()
    wait_held(page, held)
    row_for(page, ids[1]).click()
    banner = page.get_by_test_id("report-banner")
    expect(banner).to_contain_text(names[1])

    settle(page, held.pop())

    expect(banner).to_have_count(1)
    expect(banner).to_contain_text(names[1])


def test_a_late_failed_report_does_not_mark_a_newer_one(
    page, bench, bench_url, tmp_path, collectors
):
    """WINDOW: A's report asked for again with a dataset path that cannot
    be read, that request held; B's row clicked and its report drawn;
    A's answer, the door's real 422, then released.

    A report load that fails after a newer one was asked for is dropped,
    as a late success is: the panel stays ready with one banner, B's, and
    no loading or error note, and the page says nothing on the console
    beyond Chromium's line for the 422. PRE-STATE: B's banner is drawn
    and the panel is ready before A's answer is released."""
    _, digest = stored_digest(page, bench_url, [{"id": "s1"}])
    names = [unique("failing report"), unique("newer report")]
    ids = [api_experiment(page, bench_url, digest, ["stub/fast"], n) for n in names]
    bench(["stub/fast"])
    open_experiments(page)
    panel = page.get_by_test_id("report-panel")
    banner = page.get_by_test_id("report-banner")
    row_for(page, ids[0]).click()
    expect(panel).to_have_attribute("data-state", "ready")
    page.get_by_test_id("report-dataset-path").fill(str(tmp_path / "missing.jsonl"))
    held = hold(page, re.compile(rf"/experiments/{ids[0]}/report\?"), "GET")
    page.get_by_test_id("report-dataset-apply").click()
    wait_held(page, held)
    row_for(page, ids[1]).click()
    expect(banner).to_contain_text(names[1])
    expect(panel).to_have_attribute("data-state", "ready")

    route = held.pop()
    with page.expect_response(lambda r: r.url == route.request.url) as late:
        route.continue_()
    assert late.value.status == 422
    page.evaluate("() => new Promise((done) => setTimeout(done, 50))")

    expect(panel).to_have_attribute("data-state", "ready")
    expect(banner).to_have_count(1)
    expect(banner).to_contain_text(names[1])
    expect(page.get_by_test_id("report-state")).to_have_count(0)


def test_the_list_names_its_own_state_while_loading(page, bench, bench_url):
    """WINDOW: the list's data-state read in the same task as the summary
    click, and in the same task as a reload started with the panel open.

    Both say loading at once, as the history panel does: toggle is
    dispatched asynchronously, and a reload over the old rows would
    otherwise leave them reading ready while the new answer is out."""
    bench(["stub/fast"])
    listing = page.get_by_test_id("experiment-list")
    expect(listing).to_have_attribute("data-state", "idle")
    at_click = page.evaluate(
        "() => { document.querySelector('[data-testid=experiments-toggle]').click();"
        " return document.querySelector('[data-testid=experiment-list]')"
        ".dataset.state; }"
    )
    assert at_click == "loading"
    expect(listing).to_have_attribute("data-state", re.compile("^(ready|empty)$"))
    at_reload = page.evaluate(
        "() => { BenchLifecycle.refresh(); return document.querySelector("
        "'[data-testid=experiment-list]').dataset.state; }"
    )
    assert at_reload == "loading"


def test_a_selection_no_longer_listed_is_dropped(page, bench, bench_url, runs):
    """WINDOW: a running experiment selected and watched, the list read
    again without it, then read again empty, then read again with it
    back, still running.

    A selection nothing on screen describes is dropped: no row is marked,
    the experiment's part of the panel goes, and its stream is closed;
    an empty list leaves nothing selected either. Dropped means dropped,
    not hidden: when the experiment is listed again it comes back
    unselected and unwatched, rather than the panel quietly reopening
    something the person did not choose."""
    _, digest = stored_digest(page, bench_url, [{"id": f"d{n}"} for n in range(4)])
    eid = api_experiment(page, bench_url, digest, ["stub/slow"], unique("dropped"))
    runs.append(eid)
    assert page.request.post(
        f"{bench_url}/experiments/{eid}/start", data={"dataset_digest": digest}
    ).ok
    bench(["stub/fast"])
    log = progress_log(page)
    open_experiments(page)
    row_for(page, eid).click()
    expect(page.get_by_test_id("experiment-counters")).to_have_text(
        re.compile(r"^done [1-4] "), timeout=DONE_TIMEOUT
    )
    listed = page.request.get(bench_url + "/experiments").json()["experiments"]
    answer = {
        "body": json.dumps({"experiments": [e for e in listed if e["id"] != eid]})
    }
    page.route(
        "**/experiments",
        lambda route: (
            route.fulfill(
                status=200, content_type="application/json", body=answer["body"]
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )

    page.evaluate("() => BenchLifecycle.refresh()")

    expect(page.get_by_test_id("experiment-selected")).to_be_hidden()
    expect(
        page.locator("[data-testid=experiment-row][aria-pressed=true]")
    ).to_have_count(0)
    for _ in range(40):
        if closed(log, eid):
            break
        page.wait_for_timeout(50)
    assert len(closed(log, eid)) == 1
    answer["body"] = json.dumps({"experiments": []})
    page.evaluate("() => BenchLifecycle.refresh()")
    expect(page.get_by_test_id("experiment-list")).to_have_text("no experiments yet")
    expect(page.get_by_test_id("experiment-selected")).to_be_hidden()
    answer["body"] = json.dumps({"experiments": listed})
    page.evaluate("() => BenchLifecycle.refresh()")
    expect(row_for(page, eid)).to_have_attribute("aria-pressed", "false")
    expect(page.get_by_test_id("experiment-selected")).to_be_hidden()
    page.wait_for_timeout(500)
    assert len(opened(log, eid)) == 1


MEASURE_PANEL = """
() => {
  const parse = (c) => c.match(/[\\d.]+/g).map(Number);
  const out = [];
  for (const el of document.querySelectorAll('#experiments *')) {
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
    last = None
    for _ in range(40):
        ratios = []
        for item in page.evaluate(MEASURE_PANEL):
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


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_panels_text_meets_aa_where_it_sits(page, bench, bench_url, theme):
    """WINDOW: computed colours on the live page, in each theme, for every
    visible, enabled element in the Experiments panel that holds text of
    its own, with a created experiment selected and its projection shown.

    Checks the elements rather than tokens, so a rule that put a token on
    a background it fails on is caught where it lands. Named selectors are
    asserted present so the sweep cannot pass by finding nothing, and the
    selected row keeps its accent bar."""
    ready(page, bench, bench_url, ["stub/fast"], [{"id": "aa1"}])
    page.evaluate("t => { document.documentElement.dataset.theme = t }", theme)
    eid = create(page, unique("aa"))
    page.mouse.move(0, 0)
    for selector in [
        ".xp-cap-text",
        "[data-testid=experiment-source-lineup]",
        "[data-testid=experiment-projection]",
        "[data-testid=experiment-status]",
        "[data-testid=experiment-create-msg]",
        "#experiment-list-head .panel-label",
        ".xp-entry[aria-pressed=true] .hcount",
    ]:
        expect(page.locator(selector).first).to_be_visible()

    failures = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]

    assert not failures, "below WCAG AA 4.5:1:\n" + "\n".join(failures)
    accent = page.evaluate(
        "() => { const p = document.createElement('div');"
        " document.body.append(p); p.style.color = 'var(--accent)';"
        " const c = getComputedStyle(p).color; p.remove(); return c; }"
    )
    shadow = row_for(page, eid).evaluate("el => getComputedStyle(el).boxShadow")
    assert shadow == accent + " 2px 0px 0px 0px inset"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_panels_state_lines_meet_aa_and_show(
    page, bench, bench_url, tmp_path, theme, collectors
):
    """WINDOW: computed colours, in each theme, of the panel's lines that
    appear only in some states: the estimand note, the attachments note,
    the note beside a greyed Start, and a refused action message; and
    that each is shown, not only filled.

    The first sweep saw one state, so a line that exists only in another
    could sit at any contrast unseen, or be hidden and still carry its
    text. A refusal also keeps --err."""
    collectors.extend([CONFLICT_RESOURCE, MISSING_RESOURCE])
    path = tmp_path / "state.jsonl"
    path.write_text(
        '{"id": "s1", "prompt": "' + uuid.uuid4().hex + '"}\n', encoding="utf-8"
    )
    by_path = page.request.post(
        bench_url + "/experiments",
        data={
            "name": unique("by path"),
            "dataset_path": str(path),
            "lineup": ["stub/fast"],
            "budget": "standard",
        },
    ).json()["id"]
    _, digest = stored_digest(page, bench_url, [{"id": "s2"}])
    finished_elsewhere = api_experiment(
        page, bench_url, digest, ["stub/fast"], unique("elsewhere")
    )
    bench(["stub/fast"])
    page.evaluate("t => { document.documentElement.dataset.theme = t }", theme)
    open_experiments(page)
    page.get_by_test_id("experiment-estimand").select_option("underlying_model")
    row_for(page, by_path).click()
    for testid in (
        "experiment-estimand-note",
        "experiment-attachments-note",
        "experiment-start-note",
    ):
        expect(page.get_by_test_id(testid)).to_be_visible()
    first = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]
    row_for(page, finished_elsewhere).click()
    assert page.request.post(
        f"{bench_url}/experiments/{finished_elsewhere}/start",
        data={"dataset_digest": digest},
    ).ok
    wait_status(page, bench_url, finished_elsewhere)
    page.get_by_test_id("experiment-start").click()
    message = page.get_by_test_id("experiment-action-msg")
    expect(message).to_have_attribute("data-state", "refused")
    expect(message).to_be_visible()
    page.mouse.move(0, 0)
    second = [f"{theme}: {n} = {r}" for n, r in swept(page) if r < 4.5]

    assert not first + second, "below WCAG AA 4.5:1:\n" + "\n".join(first + second)
    err = page.evaluate(
        "() => { const p = document.createElement('div');"
        " document.body.append(p); p.style.color = 'var(--err)';"
        " const c = getComputedStyle(p).color; p.remove(); return c; }"
    )
    assert message.evaluate("el => getComputedStyle(el).color") == err
