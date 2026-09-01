"""Session fixtures for the browser suite.

The app under test is the real bench: uvicorn booted as a subprocess
(so the true startup path runs, key check and catalog fetch included)
against a fresh temp BENCH_DB, pointed at an in-process stub OpenRouter
via the env-override seam in bench.models.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from stub_openrouter import build_app

REPO_ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            # trust_env off: poll loopback directly, never via an ambient
            # developer proxy that would fail the localhost connection.
            if httpx.get(url, timeout=2.0, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server at {url} never became ready")


@pytest.fixture(scope="session")
def stub_url():
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(build_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    wait_for(url + "/api/v1/models")
    yield url
    server.should_exit = True
    thread.join(timeout=5)


@contextmanager
def boot_bench(stub_url, tmp_path_factory, extra_env=None):
    """One bench subprocess against the stub, with its own database.

    Factored out because the data-policy badge and payload can only be
    exercised against a process booted with BENCH_DATA_POLICY set: the
    policy is boot-scoped by design, so there is no way to switch it on a
    running server, and pretending otherwise would test something the app
    does not do.
    """
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "OPENROUTER_API_KEY": "test-key",
            "BENCH_DB": str(tmp_path_factory.mktemp("browser-db") / "bench.db"),
            "OPENROUTER_URL": stub_url + "/api/v1/chat/completions",
            "MODELS_URL": stub_url + "/api/v1/models",
        }
    )
    env.update(extra_env or {})
    # The app subprocess keeps trust_env on (real operators may reach
    # OpenRouter through a proxy), so a developer proxy in the environment
    # would route its real localhost call to the stub OpenRouter through
    # that proxy and hang the harness. Scrub the proxy vars from this
    # subprocess rather than making the app degrade its own behavior.
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(proxy_var, None)
    log_path = tmp_path_factory.mktemp("browser-logs") / "uvicorn.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "bench.main:app", "--port", str(port)],
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            # Host must be localhost or the bench's own guard rejects us.
            wait_for(f"http://localhost:{port}/models")
            yield f"http://localhost:{port}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.returncode not in (0, -15):
                sys.stderr.write(log_path.read_text())


@pytest.fixture(scope="session")
def bench_url(stub_url, tmp_path_factory):
    with boot_bench(stub_url, tmp_path_factory) as url:
        yield url


@pytest.fixture(scope="session")
def zdr_bench_url(stub_url, tmp_path_factory):
    """A second bench booted under the strictest data policy.

    Session-scoped like the default one, and started only when a test asks
    for it, so the common suite still pays for exactly one server.
    """
    with boot_bench(stub_url, tmp_path_factory, {"BENCH_DATA_POLICY": "zdr"}) as url:
        yield url


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    # Environments with a pre-provisioned Chromium (sandboxes, offline
    # runners) can point the harness at it instead of running
    # playwright install; everywhere else this is a no-op.
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if exe:
        return {**browser_type_launch_args, "executable_path": exe}
    return browser_type_launch_args


@pytest.fixture
def bench(page, bench_url):
    """A page factory bound to the bench: seeds a lineup and navigates.

    The fonts are self-hosted now, so nothing external is fetched and no
    blocking route is needed to keep the harness hermetic.
    """

    def open_bench(lineup):
        # Seeded exactly as the app stores it, a raw JSON array of ids;
        # this is also the pre-VOLT format, which must keep loading.
        page.add_init_script(
            f"localStorage.setItem('bench-lineup', {json.dumps(json.dumps(lineup))})"
        )
        page.goto(bench_url)
        return page

    return open_bench


@pytest.fixture
def open_history(page):
    """Open the History panel and wait for its load to settle.

    Shared rather than repeated per file because the wait is the whole
    point. The panel empties itself the moment it opens and refills only
    when its fetch returns, so any claim about its rows made while that is
    in flight is worthless: a positive one races the load, and a negative
    one is satisfied by it. Waiting on the list's own state attribute is
    the panel saying it has finished, whatever the outcome was.

    A bare click was enough on a fast machine and not on a loaded one.
    It failed on CI twice, both times as a row that was really there
    reading as absent.

    The third failure was this helper's own doing. Waiting for any settled
    state treated "error" as an outcome to proceed from, so a transiently
    failed /runs fetch at open cleared the rows, set state=error, and every
    assertion downstream then ran against an honestly empty panel. A
    missing row read as a missing row. The state attribute was reporting
    the truth the whole time and the wait was throwing it away.

    So the outcome is now read, not just awaited. "ready" and "empty" are
    answers to the question asked. "error" is not: it means the panel never
    got to answer, which is exactly the transient this retries once, by
    closing and reopening (the only reload the panel offers). A second
    error is not a flake to absorb, it is the run's result, so it fails
    loudly and quotes the panel's own message: a fourth occurrence of this
    should diagnose itself from the log instead of costing another round
    of forensics.
    """
    list_el = page.get_by_test_id("history-list")

    def settle():
        page.wait_for_function(
            """() => ['ready', 'empty', 'error'].includes(
                 document.getElementById('history-list').dataset.state)"""
        )
        return list_el.get_attribute("data-state")

    def open_it():
        page.get_by_test_id("history-toggle").click()
        if settle() != "error":
            return
        # Collapse and reopen: the toggle handler reloads on every open, so
        # this is a genuine second fetch rather than a re-read of the
        # failed one.
        #
        # The reopen depends on the panel claiming "loading" synchronously
        # with the click. Without that the settle below reads the error
        # still sitting on the attribute from the load that just failed and
        # reports the retry as failed before it has started, which turns
        # the loud branch into a new flake wearing the old one's face. It
        # was measured doing exactly that; static/history.js closes it in
        # a click listener, and a tombstone locks it.
        page.get_by_test_id("history-toggle").click()
        page.get_by_test_id("history-toggle").click()
        if settle() != "error":
            return
        pytest.fail(
            "history panel failed to load twice; the panel says: "
            + repr(list_el.text_content())
        )

    return open_it


@pytest.fixture
def zdr_bench(page, zdr_bench_url):
    """The same page factory, pointed at the zdr-policy bench."""

    def open_bench(lineup):
        page.add_init_script(
            f"localStorage.setItem('bench-lineup', {json.dumps(json.dumps(lineup))})"
        )
        page.goto(zdr_bench_url)
        return page

    return open_bench


@pytest.fixture(scope="session")
def snapshot_root(tmp_path_factory):
    """A tiny clone the snapshot bench is allowed to walk.

    Session-scoped because the bench that may walk it is: the allowlist
    is read once at boot, so the tree has to exist before the process
    starts and must not move under it afterwards.
    """
    root = tmp_path_factory.mktemp("snapshot-clone")
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("ANSWER = 42\n", encoding="utf-8")
    (root / "pkg" / "b.py").write_text("HELPER = 'ok'\n", encoding="utf-8")
    # Not selected by the patterns the test writes, so the manifest's
    # count is a claim about the SELECTION rather than about the tree.
    (root / "notes.md").write_text("not selected\n", encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def snapshot_bench_url(stub_url, tmp_path_factory, snapshot_root):
    """A third bench, the only one with BENCH_REPO_ROOTS set.

    Its own process for the reason the zdr bench has one: the allowlist
    is boot-scoped by design, so "snapshots are off" and "snapshots are
    on" are two servers rather than two requests, and the default bench
    stays the one that proves the off state.
    """
    with boot_bench(
        stub_url, tmp_path_factory, {"BENCH_REPO_ROOTS": str(snapshot_root)}
    ) as url:
        yield url


@pytest.fixture
def snapshot_bench(page, snapshot_bench_url):
    """The same page factory, pointed at the bench that may walk a tree."""

    def open_bench(lineup):
        page.add_init_script(
            f"localStorage.setItem('bench-lineup', {json.dumps(json.dumps(lineup))})"
        )
        page.goto(snapshot_bench_url)
        return page

    return open_bench
