"""The clone door's test remote, and git's behaviour against it, on the
platform that runs these tests.

tests/clone_stub.py serves bare repositories over HTTPS on loopback
through git http-backend. Everything the clone door's tests will lean on
is shown here first, against the git and the TLS stack of whatever
machine runs the suite, because the machine the door was designed on is
not the one whose run is the gate: this Mac's git reaches TLS through
LibreSSL, and CI's ubuntu git through GnuTLS.

What is shown: a branch, a tag and a commit fetch at depth one over TLS
the session's certificate makes trusted, and not without it; a ref long
enough that git gzips its request survives; a private path asks for
credentials and a planted helper is consulted when nothing stops it; a
stalled path holds a fetch until released; a redirect is followed by
default and not under http.followRedirects=false; file:// is refused
under protocol.allow=never; and a ref that looks like an option is an
option to git unless it comes after "--".

The flags are the ones the clone door passes (see DOOR_FLAGS); the
environment is the door's scrubbed one. Neither is imported from the
door, which does not exist yet at this commit.
"""

import os
import subprocess
from pathlib import Path

import clone_stub
import pytest

# The clone door's git configuration, as the commission and the O2
# design critique fixed it: https only, no credential helper, no
# redirect, no ambient negotiate identity.
DOOR_FLAGS = [
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.https.allow=always",
    "-c",
    "credential.helper=",
    "-c",
    "http.followRedirects=false",
    "-c",
    "http.emptyAuth=false",
]


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    with clone_stub.serving(tmp_path_factory.mktemp("stub")) as served:
        yield served


def scrubbed(tmp_path: Path, stub: clone_stub.Stub, *, cainfo: bool = True) -> dict:
    """The door's environment: PATH, an empty HOME, no prompt, no system
    config, and the session certificate as the only trust added."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    if cainfo:
        env["GIT_SSL_CAINFO"] = str(stub.cainfo)
    return env


def fetch(
    tmp_path: Path,
    env: dict,
    url: str,
    ref: str,
    *,
    flags: list[str] = DOOR_FLAGS,
    dashdash: bool = True,
) -> tuple[subprocess.CompletedProcess, Path]:
    tree = tmp_path / "tree"
    tree.mkdir()
    pinned = [f"--git-dir={tree}/.git", f"--work-tree={tree}"]
    subprocess.run(["git", *pinned, "init", "-q", "--template="], env=env, check=True)
    proc = subprocess.run(
        [
            "git",
            *flags,
            *pinned,
            "fetch",
            "-q",
            "--depth",
            "1",
            "--no-tags",
            *(["--"] if dashdash else []),
            url,
            ref,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, tree / ".git" / "FETCH_HEAD"


def fetched_sha(fetch_head: Path) -> str:
    return fetch_head.read_text().split("\t", 1)[0]


def nothing_fetched(fetch_head: Path) -> bool:
    # git creates FETCH_HEAD empty before it fetches anything.
    return not fetch_head.exists() or fetch_head.read_text() == ""


@pytest.mark.parametrize("which", ["branch", "tag", "commit"])
def test_a_branch_a_tag_and_a_commit_fetch_over_the_sessions_tls(tmp_path, stub, which):
    """WINDOW: one depth-one fetch through the stub, with the door's flags
    and environment, of a branch, a tag or a commit named by sha.

    PRE-STATE: the repository's branch has moved past the tagged commit,
    so the three fetch two different commits and a wrong one would show,
    and nothing was fetched before."""
    first = stub.repository("o", f"three-{which}", {"a.py": b"one\n"}, tag="v1")
    second = stub.repository("o", f"three-{which}", {"a.py": b"two\n"})
    assert first != second
    ref, want = {
        "branch": ("main", second),
        "tag": ("v1", first),
        "commit": (first, first),
    }[which]
    env = scrubbed(tmp_path, stub)
    proc, fetch_head = fetch(tmp_path, env, stub.url("o", f"three-{which}"), ref)
    assert proc.returncode == 0, proc.stderr
    assert fetched_sha(fetch_head) == want


def test_without_the_sessions_certificate_the_fetch_is_refused(tmp_path, stub):
    """WINDOW: the same fetch with GIT_SSL_CAINFO unset.

    The platform's TLS stack verifies the stub's certificate and refuses
    it, so the fetches above were trusted by the seam and by nothing
    else. PRE-STATE: the repository is there to be fetched."""
    stub.repository("o", "untrusted", {"a.py": b"x\n"})
    env = scrubbed(tmp_path, stub, cainfo=False)
    proc, fetch_head = fetch(tmp_path, env, stub.url("o", "untrusted"), "main")
    assert proc.returncode == 128
    assert nothing_fetched(fetch_head)


def test_a_ref_long_enough_to_gzip_the_request_is_fetched(tmp_path, stub):
    """WINDOW: a fetch of a branch named with 200 characters.

    git gzips a request body past 1 KiB, and a ref of 128 characters or
    more makes the ref listing that long; the stub must decode it or
    every long ref fails for a harness reason. PRE-STATE: the name is
    within the door's ref bound and past the gzip threshold."""
    branch = "b" * 200
    assert 128 < len(branch) <= 255
    sha = stub.repository("o", "long", {"a.py": b"x\n"}, branch=branch)
    proc, fetch_head = fetch(
        tmp_path, scrubbed(tmp_path, stub), stub.url("o", "long"), branch
    )
    assert proc.returncode == 0, proc.stderr
    assert fetched_sha(fetch_head) == sha


def test_a_private_path_asks_and_a_helper_nothing_stops_is_consulted(tmp_path, stub):
    """WINDOW: a fetch of a path the stub answers 401, with a credential
    helper planted through GIT_CONFIG_COUNT and nothing clearing it.

    The control the door's credential proofs lean on: the stub's 401
    makes git consult a helper and send what it said. PRE-STATE: the
    path is private, and the helper has not run."""
    stub.private.add("/o/secret.git")
    marker = tmp_path / "helper-ran"
    helper = tmp_path / "helper.sh"
    helper.write_text(
        f"#!/bin/sh\necho ran >> {marker}\necho username=u\necho password=p\n"
    )
    helper.chmod(0o755)
    env = {
        **scrubbed(tmp_path, stub),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": str(helper),
    }
    assert not marker.exists()
    proc, fetch_head = fetch(tmp_path, env, stub.url("o", "secret"), "main", flags=[])
    assert marker.exists()
    assert any(
        s.path.startswith("/o/secret.git") and s.authorization for s in stub.seen
    )
    assert proc.returncode != 0
    assert nothing_fetched(fetch_head)


def test_a_stalled_path_holds_the_fetch_until_it_is_killed(tmp_path, stub):
    """WINDOW: a fetch of a path whose upload-pack the stub holds, killed
    once the stub reports the hold.

    The timeout proof's instrument: the stall is reached, the fetch does
    not finish on its own, and killing it ends it. PRE-STATE: the stall
    has not been reached."""
    stub.repository("o", "slow", {"a.py": b"x\n"})
    stub.stalled.add("/o/slow.git")
    stub.stall_reached.clear()
    tree = tmp_path / "tree"
    tree.mkdir()
    env = scrubbed(tmp_path, stub)
    pinned = [f"--git-dir={tree}/.git", f"--work-tree={tree}"]
    subprocess.run(["git", *pinned, "init", "-q"], env=env, check=True)
    proc = subprocess.Popen(
        [
            "git",
            *DOOR_FLAGS,
            *pinned,
            "fetch",
            "-q",
            "--depth",
            "1",
            "--",
            stub.url("o", "slow"),
            "main",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert stub.stall_reached.wait(timeout=30)
        assert proc.poll() is None
    finally:
        os.killpg(proc.pid, 9)
        proc.wait()
    assert proc.returncode == -9


def test_a_redirect_is_followed_by_default_and_not_under_the_doors_flag(tmp_path, stub):
    """WINDOW: two fetches of a path the stub redirects to another
    repository, with git's default and with http.followRedirects=false.

    By default git follows the first redirect and fetches from where it
    points, which for a real host can be a host nobody listed; under the
    door's flag it stops at the 302. PRE-STATE: the target serves a
    commit of its own."""
    target_sha = stub.repository("o", "target", {"a.py": b"target\n"})
    stub.redirects["/o/moved.git"] = f"https://{stub.host}/o/target.git"
    env = scrubbed(tmp_path, stub)

    (tmp_path / "default").mkdir()
    followed, fetch_head = fetch(
        tmp_path / "default", env, stub.url("o", "moved"), "main", flags=[]
    )
    assert followed.returncode == 0, followed.stderr
    assert fetched_sha(fetch_head) == target_sha

    before = sum(1 for s in stub.seen if s.path.startswith("/o/target.git"))
    (tmp_path / "door").mkdir()
    stopped, fetch_head = fetch(tmp_path / "door", env, stub.url("o", "moved"), "main")
    assert stopped.returncode == 128
    assert "302" in stopped.stderr
    assert nothing_fetched(fetch_head)
    assert sum(1 for s in stub.seen if s.path.startswith("/o/target.git")) == before


def test_file_transport_is_refused_under_the_doors_flags(tmp_path, stub):
    """WINDOW: a fetch of a bare repository by file:// with the door's
    flags.

    protocol.allow=never with only https allowed refuses every other
    transport, which is why the tests need an https remote at all.
    PRE-STATE: the repository exists on disk at that path."""
    stub.repository("o", "local", {"a.py": b"x\n"})
    bare = stub.root / "srv" / "o" / "local.git"
    assert bare.is_dir()
    proc, fetch_head = fetch(
        tmp_path, scrubbed(tmp_path, stub), f"file://{bare}", "main"
    )
    assert proc.returncode == 128
    assert "transport 'file' not allowed" in proc.stderr
    assert nothing_fetched(fetch_head)


def test_a_ref_that_looks_like_an_option_is_one_unless_it_follows_dashdash(
    tmp_path, stub
):
    """WINDOW: two fetches whose ref is "--upload-pack=touch PWNED", one
    with "--" before the URL and one without.

    After "--" it is a ref git cannot find (exit 128, nothing fetched);
    without, git takes it as an option and fetches the remote's HEAD as
    though that were what was asked. This is the behaviour the door's
    "--" exists for, shown on this platform's git. PRE-STATE: the
    repository has a HEAD to fetch."""
    sha = stub.repository("o", "dash", {"a.py": b"x\n"})
    env = scrubbed(tmp_path, stub)
    ref = "--upload-pack=touch PWNED"

    (tmp_path / "with").mkdir()
    guarded, fetch_head = fetch(tmp_path / "with", env, stub.url("o", "dash"), ref)
    assert guarded.returncode == 128
    assert nothing_fetched(fetch_head)

    (tmp_path / "without").mkdir()
    parsed, fetch_head = fetch(
        tmp_path / "without", env, stub.url("o", "dash"), ref, dashdash=False
    )
    assert parsed.returncode == 0
    assert fetched_sha(fetch_head) == sha
    assert not (tmp_path / "PWNED").exists()
