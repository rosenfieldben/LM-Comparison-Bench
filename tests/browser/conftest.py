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
            if httpx.get(url, timeout=2.0).status_code == 200:
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


@pytest.fixture(scope="session")
def bench_url(stub_url, tmp_path_factory):
    port = free_port()
    env = os.environ.copy()
    env.update({
        "OPENROUTER_API_KEY": "test-key",
        "BENCH_DB": str(tmp_path_factory.mktemp("browser-db") / "bench.db"),
        "OPENROUTER_URL": stub_url + "/api/v1/chat/completions",
        "MODELS_URL": stub_url + "/api/v1/models",
    })
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
    """A page factory bound to the bench: seeds a lineup, blocks the
    external font fetch (hermetic and faster), and navigates."""

    def open_bench(lineup):
        page.route(
            lambda url: "fonts.googleapis.com" in url or "fonts.gstatic.com" in url,
            lambda route: route.abort(),
        )
        # Seeded exactly as the app stores it, a raw JSON array of ids;
        # this is also the pre-VOLT format, which must keep loading.
        page.add_init_script(
            "localStorage.setItem('bench-lineup', %s)"
            % json.dumps(json.dumps(lineup))
        )
        page.goto(bench_url)
        return page

    return open_bench
