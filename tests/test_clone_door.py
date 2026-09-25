"""POST /clones against a real git and a real HTTPS remote on loopback.

The clone door's posture, proved rule by rule through the door itself:
refused in the commission's order before anything is fetched; no
refusal repeats the URL; no credential from any layer the scrub removes;
one deadline and one set of ceilings, each leaving nothing new on disk;
one clone at a time, released however a clone ends; an update that is a
fresh clone swapped in, so the old tree and row survive a failed one;
the snapshot doors refusing a tree a clone is writing; "--" before every
operand; and the URL recorded in the clones row and nowhere else.

The remote is tests/clone_stub.py, whose behaviour against this
platform's git is shown in tests/test_clone_stub.py.
"""

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path

import clone_stub
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from bench import clones, main
from bench.models import OPENROUTER_URL

# Distinctive enough that nothing but this URL could contain them.
OWNER, REPO = "zq-owner", "zq-repo"
SECRET = "Zq9TOKENx7"


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    with clone_stub.serving(tmp_path_factory.mktemp("stub")) as served:
        yield served


@pytest.fixture
def bench_env(monkeypatch, tmp_path, stub):
    """The environment a clone-door bench boots with, before it boots."""
    clone_root = tmp_path / "clones"
    clone_root.mkdir()

    async def fake_fetch_catalog(client):
        return {"models": [], "fetched": False, "prices": {}, "digest": None}

    monkeypatch.setattr("bench.main.fetch_catalog", fake_fetch_catalog)
    env = {
        "OPENROUTER_API_KEY": "test-key",
        "BENCH_DB": str(tmp_path / "bench.db"),
        "BENCH_REPO_ROOTS": str(clone_root),
        "BENCH_CLONE_ROOT": str(clone_root),
        "BENCH_CLONE_HOSTS": stub.host,
        "BENCH_CLONE_CAINFO": str(stub.cainfo),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


@pytest.fixture
def bench(bench_env):
    with TestClient(main.app, base_url="http://localhost") as client:
        yield client


def repo_for(request, stub, files=None, **kw):
    """A fresh repository on the stub, named for the test that asks."""
    name = "r" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:10]
    sha = stub.repository(OWNER, name, files or {"a.py": b"one\n"}, **kw)
    return name, sha


def clone_of(bench, stub, repo, ref="main", *, suffix=".git"):
    url = f"https://{stub.host}/{OWNER}/{repo}{suffix}"
    return bench.post("/clones", json={"url": url, "ref": ref})


def clone_root(bench) -> Path:
    return Path(bench.app.state.clone_root)


def work_left(bench) -> list[str]:
    return sorted(n for n in os.listdir(clone_root(bench)) if n.startswith("."))


# ----- refused in order, before anything is fetched ---------------------


def test_the_door_is_off_until_a_clone_root_is_named(bench_env, monkeypatch, stub):
    """WINDOW: POST /clones and GET /models on a bench booted without
    BENCH_CLONE_ROOT.

    CLONES_OFF, verbatim, at the door and as the page's standing reason,
    and before the URL is looked at: a URL that breaks every rule gets
    the same sentence. PRE-STATE: the rest of the clone configuration is
    present, so only the root is missing."""
    monkeypatch.delenv("BENCH_CLONE_ROOT")
    assert os.environ["BENCH_CLONE_HOSTS"] == stub.host
    with TestClient(main.app, base_url="http://localhost") as client:
        seen = len(stub.seen)
        for url in (stub.url(OWNER, "x"), f"http://u:{SECRET}@x/y"):
            resp = client.post("/clones", json={"url": url, "ref": "main"})
            assert resp.status_code == 403
            assert resp.json()["detail"] == clones.CLONES_OFF
        models = client.get("/models").json()
        assert models["clones_enabled"] is False
        assert models["clones_off_reason"] == clones.CLONES_OFF
        assert len(stub.seen) == seen


def test_a_clone_root_outside_the_roots_is_refused_naming_both(
    bench_env, monkeypatch, tmp_path, stub
):
    """WINDOW: POST /clones and GET /models when BENCH_CLONE_ROOT is not
    one of BENCH_REPO_ROOTS.

    Refused naming both variables, ahead of the URL's rules, and the page
    is told the same sentence. PRE-STATE: the clone root exists and the
    roots name a different directory."""
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("BENCH_REPO_ROOTS", str(other))
    assert os.path.isdir(os.environ["BENCH_CLONE_ROOT"])
    with TestClient(main.app, base_url="http://localhost") as client:
        resp = client.post("/clones", json={"url": "ftp://x", "ref": "-x"})
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert "BENCH_CLONE_ROOT" in detail and "BENCH_REPO_ROOTS" in detail
        models = client.get("/models").json()
        assert models["clones_enabled"] is False
        assert models["clones_off_reason"] == detail


@pytest.mark.parametrize(
    ("url", "ref", "status", "rule"),
    [
        ("http://{host}/o/r", "main", 403, "not https"),
        ("https://u:{secret}@{host}/o/r", "main", 403, "a user or a token"),
        ("https://github.com/o/r", "main", 403, "BENCH_CLONE_HOSTS"),
        ("https://{host}/o/r?x={secret}", "main", 422, "a query"),
        ("https://{host}/o/r/tree/main", "main", 422, "/owner/repo"),
        ("https://{host}/-x/r", "main", 422, "owner"),
        ("https://{host}/o/r", "--upload-pack=x", 422, "starts with '-'"),
        ("https://{host}/o/r", "a..b", 422, "'..'"),
        ("https://{host}/o/r", "https://u:{secret}@x/y", 422, "the ref is not"),
        ("https://u:{secret}@{host}/o/r", "--upload-pack=x", 403, "a user or a token"),
    ],
)
def test_the_url_and_the_ref_refuse_at_the_door_before_git_runs(
    bench, stub, url, ref, status, rule
):
    """WINDOW: POST /clones with a URL or a ref that breaks one rule.

    The URL's rules, then the ref's (a request breaking both gets the
    URL's), and nothing reaches the remote: the stub records no request.
    No refusal holds the planted token. PRE-STATE: the door is on."""
    assert bench.get("/models").json()["clones_enabled"] is True
    seen = len(stub.seen)
    sent = {
        "url": url.format(host=stub.host, secret=SECRET),
        "ref": ref.format(secret=SECRET),
    }
    resp = bench.post("/clones", json=sent)
    assert resp.status_code == status
    assert rule in resp.json()["detail"]
    assert SECRET not in resp.text
    assert len(stub.seen) == seen
    assert os.listdir(clone_root(bench)) == []


@pytest.mark.parametrize(
    "body",
    [
        {"url": f"https://u:{SECRET}@github.com/o/r"},
        {"url": f"https://u:{SECRET}@github.com/" + "r" * 2100, "ref": "main"},
        {"url": {"u": f"https://u:{SECRET}@x"}, "ref": "main"},
        {"url": "https://github.com/o/r", "ref": "main", "token": SECRET},
        {"url": "https://github.com/o/r", "ref": "main", SECRET: 1},
        [f"https://u:{SECRET}@github.com/o/r"],
    ],
)
def test_a_body_the_model_refuses_repeats_nothing_that_was_sent(bench, body):
    """WINDOW: POST /clones with a body the request model refuses.

    A missing ref (whose error's input is the whole body), an over-long
    URL, a URL that is not a string, an unknown field by name and by
    value, a body that is not an object: each a 422 naming the field and
    the rule, and none holding the token. PRE-STATE: the same body at
    another door would repeat it, which is why this door is different."""
    elsewhere = bench.post("/snapshots", json=body)
    assert elsewhere.status_code == 422
    resp = bench.post("/clones", json=body)
    assert resp.status_code == 422
    assert SECRET not in resp.text
    for error in resp.json()["detail"]:
        assert set(error) == {"type", "loc", "msg"}


@pytest.mark.parametrize(
    ("name", "value", "says"),
    [
        ("BENCH_CLONE_ROOT", "relative/dir", "not an absolute path"),
        ("BENCH_CLONE_ROOT", "/no/such/dir/anywhere", "not a directory"),
        ("BENCH_CLONE_HOSTS", "github.com/x", "BENCH_CLONE_HOSTS"),
        ("BENCH_CLONE_CAINFO", "/no/such/bundle.pem", "BENCH_CLONE_CAINFO"),
        ("BENCH_CLONE_CAINFO", "cert.pem", "BENCH_CLONE_CAINFO"),
    ],
)
def test_a_clone_variable_set_wrong_fails_boot_naming_it(
    bench_env, monkeypatch, name, value, says
):
    """WINDOW: the lifespan, booting with one clone variable set wrong.

    Set and wrong fails boot, as BENCH_REPO_ROOTS does, rather than
    becoming a door that refuses every request for a reason the operator
    must go looking for. PRE-STATE: the environment boots as it stands."""
    with TestClient(main.app, base_url="http://localhost"):
        pass
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=says):
        with TestClient(main.app, base_url="http://localhost"):
            pass


# ----- a clone, and a second of the same --------------------------------


@pytest.mark.parametrize("which", ["branch", "tag", "commit"])
def test_a_branch_a_tag_or_a_commit_is_cloned_and_recorded(request, bench, stub, which):
    """WINDOW: POST /clones of one ref, the response, the directory and
    the clones row.

    201 and "cloned"; the root is BENCH_CLONE_ROOT/<sha256 of the
    identity, a newline and the ref>[:16], its files the ref's; the row
    holds the identity (no ".git") and the head. PRE-STATE: the branch
    has moved past the tag, so the three check out different commits."""
    repo, first = repo_for(request, stub, {"a.py": b"one\n"}, tag="v1")
    second = stub.repository(OWNER, repo, {"a.py": b"two\n"})
    assert first != second
    ref, head, text = {
        "branch": ("main", second, "two\n"),
        "tag": ("v1", first, "one\n"),
        "commit": (first, first, "one\n"),
    }[which]
    resp = clone_of(bench, stub, repo, ref)
    assert resp.status_code == 201, resp.text
    made = resp.json()
    identity = f"https://{stub.host}/{OWNER}/{repo}"
    name = hashlib.sha256(f"{identity}\n{ref}".encode()).hexdigest()[:16]
    assert made == {
        "id": made["id"],
        "url": identity,
        "ref": ref,
        "head_sha": head,
        "root": str(clone_root(bench) / name),
        "outcome": "cloned",
    }
    assert (Path(made["root"]) / "a.py").read_text() == text
    rows = [dict(r) for r in bench.app.state.db.execute("SELECT * FROM clones")]
    assert [(r["url"], r["ref"], r["head_sha"], r["root"]) for r in rows] == [
        (identity, ref, head, made["root"])
    ]
    assert work_left(bench) == []


def test_the_same_repository_and_ref_again_is_updated_and_old_records_stand(
    request, bench, stub
):
    """WINDOW: two POST /clones of one repository and ref (the second
    spelled without ".git"), with a snapshot composed from the first
    clone between them.

    The second is 200 "updated": the same id, the same root, the new
    head and the new files, one directory on disk, created_at kept and
    updated_at moved. The snapshot composed before it keeps its row, its
    text and its capture exactly. PRE-STATE: upstream moved between the
    two, and the snapshot read the first commit's file."""
    repo, first = repo_for(request, stub, {"a.py": b"one\n"})
    made = clone_of(bench, stub, repo).json()
    snap = bench.post("/snapshots", json={"root": made["root"], "patterns": ["*.py"]})
    assert snap.status_code == 201, snap.text
    db = bench.app.state.db
    before = {
        table: [dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("attachments", "attachment_extractions", "snapshot_captures")
    }
    assert "one" in before["attachment_extractions"][0]["extracted_text"]
    row_before = dict(db.execute("SELECT * FROM clones").fetchone())

    second = stub.repository(OWNER, repo, {"a.py": b"two\n"})
    assert second != first
    again = clone_of(bench, stub, repo, suffix="")
    assert again.status_code == 200, again.text
    assert again.json() == {**made, "head_sha": second, "outcome": "updated"}
    assert (Path(made["root"]) / "a.py").read_text() == "two\n"
    assert os.listdir(clone_root(bench)) == [Path(made["root"]).name]
    row = dict(db.execute("SELECT * FROM clones").fetchone())
    assert row["created_at"] == row_before["created_at"]
    assert row["updated_at"] > row_before["updated_at"]
    for table, rows in before.items():
        assert [
            dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY rowid")
        ] == rows


def test_a_removed_directory_is_cloned_again_under_its_row(request, bench, stub):
    """WINDOW: POST /clones after the operator removed the clone's
    directory by hand, as the README says clones are removed.

    The directory is new, so "cloned" and 201; the row is the same row,
    one statement doing both. PRE-STATE: the row is there and the
    directory is not."""
    repo, _ = repo_for(request, stub)
    made = clone_of(bench, stub, repo).json()
    subprocess.run(["rm", "-rf", made["root"]], check=True)
    assert not os.path.exists(made["root"])
    assert bench.app.state.db.execute("SELECT count(*) FROM clones").fetchone()[0] == 1
    again = clone_of(bench, stub, repo)
    assert again.status_code == 201
    assert again.json()["id"] == made["id"]
    assert again.json()["outcome"] == "cloned"


def test_a_directory_with_no_row_is_replaced_and_given_one(request, bench, stub):
    """WINDOW: POST /clones where the clone's directory exists and the
    database has no row for it (a database restored beside the clones).

    "updated", since a directory was replaced, and the row is made.
    PRE-STATE: the directory is there, the clones table empty."""
    repo, _ = repo_for(request, stub)
    made = clone_of(bench, stub, repo).json()
    db = bench.app.state.db
    with db:
        db.execute("DELETE FROM clones")
    assert os.path.isdir(made["root"])
    again = clone_of(bench, stub, repo)
    assert again.status_code == 200
    assert again.json()["outcome"] == "updated"
    assert db.execute("SELECT count(*) FROM clones").fetchone()[0] == 1


# ----- no credential from any layer --------------------------------------


def plant_credentials(tmp_path: Path, monkeypatch) -> Path:
    """Every credential source the scrub removes, planted in the bench's
    own environment: a helper and an askPass in HOME's gitconfig, a
    helper and an askPass in GIT_CONFIG_*, and GIT_ASKPASS and
    SSH_ASKPASS. Each writes to one marker file when it runs."""
    marker = tmp_path / "credential-was-asked"
    helper = tmp_path / "helper.sh"
    helper.write_text(
        f'#!/bin/sh\necho "$0 $*" >> {marker}\necho username=u\necho password=p\n'
    )
    askpass = tmp_path / "askpass.sh"
    askpass.write_text(f"#!/bin/sh\necho askpass >> {marker}\necho p\n")
    for script in (helper, askpass):
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    home = tmp_path / "operator-home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        f"[credential]\n\thelper = {helper}\n[core]\n\taskPass = {askpass}\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(helper))
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "core.askPass")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", str(askpass))
    monkeypatch.setenv("GIT_ASKPASS", str(askpass))
    monkeypatch.setenv("SSH_ASKPASS", str(askpass))
    return marker


def test_a_private_repository_is_refused_and_no_planted_credential_is_asked(
    request, bench, stub, tmp_path, monkeypatch
):
    """WINDOW: POST /clones of a path the stub answers 401, with every
    credential source the scrub removes planted in the bench's own
    environment (plant_credentials).

    The commission's proof, per layer: the rebuilt environment drops
    GIT_ASKPASS, SSH_ASKPASS and GIT_CONFIG_*; the empty HOME drops the
    gitconfig's helper and askPass; nothing is asked, no Authorization
    reaches the remote, and the refusal says a public repository was
    wanted. PRE-STATE, the control: a plain git in the same planted
    environment asks the planted credential and sends it, so each plant
    is live and a layer that let it through would be seen."""
    repo, _ = repo_for(request, stub)
    stub.private.add(f"/{OWNER}/{repo}.git")
    marker = plant_credentials(tmp_path, monkeypatch)

    control = tmp_path / "control"
    control.mkdir()
    subprocess.run(["git", "init", "-q", str(control)], check=True)
    before = len(stub.seen)
    subprocess.run(
        ["git", "-C", str(control), "fetch", "-q", stub.url(OWNER, repo), "main"],
        env={
            **os.environ,
            "GIT_SSL_CAINFO": str(stub.cainfo),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        capture_output=True,
        check=False,
    )
    assert marker.exists()
    assert any(s.authorization for s in stub.seen[before:])
    marker.unlink()

    before = len(stub.seen)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 422
    assert "asked for credentials" in resp.json()["detail"]
    assert not marker.exists()
    assert stub.seen[before:]
    assert not any(s.authorization for s in stub.seen[before:])
    assert os.listdir(clone_root(bench)) == []


def test_git_runs_in_exactly_the_scrubbed_environment_and_argv(
    request, bench, stub, tmp_path, monkeypatch
):
    """WINDOW: the environment and argv a clone's first git receives, as
    a recording git placed first on PATH writes them down.

    Exactly PATH, HOME, GIT_TERMINAL_PROMPT=0, GIT_CONFIG_NOSYSTEM=1 and
    the test seam's GIT_SSL_CAINFO, with HOME an empty directory while
    git runs and gone after; the fixed configuration first; the
    repository pinned; "--" before the operands. PRE-STATE: the bench's
    own environment holds the API key and a planted GIT_ASKPASS."""
    fake = tmp_path / "fakebin"
    fake.mkdir()
    record = tmp_path / "record.json"
    # Perl and not Python: Python adds LC_CTYPE to its own environment
    # when it starts in the C locale (PEP 538), so a Python recorder
    # would report a variable the door never passed.
    (fake / "git").write_text(
        "#!/usr/bin/perl\n"
        f"open(my $f, '>', '{record}') or die;\n"
        'print $f "ENV\\t$_=$ENV{$_}\\n" for sort keys %ENV;\n'
        'print $f "ARGV\\t$_\\n" for @ARGV;\n'
        "opendir(my $d, $ENV{HOME}) or die;\n"
        'print $f "HOME\\t$_\\n" for grep { !/^\\.\\.?$/ } readdir($d);\n'
        "close $f;\n"
        "exit 1;\n"
    )
    (fake / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GIT_ASKPASS", "/bin/false")
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"

    repo, _ = repo_for(request, stub)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 502
    seen = {"ENV": [], "ARGV": [], "HOME": []}
    for line in record.read_text().splitlines():
        section, _, value = line.partition("\t")
        seen[section].append(value)
    env = dict(item.split("=", 1) for item in seen["ENV"])
    assert set(env) == {
        "PATH",
        "HOME",
        "GIT_TERMINAL_PROMPT",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_SSL_CAINFO",
    }
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_SSL_CAINFO"] == str(stub.cainfo)
    assert seen["HOME"] == []
    assert not os.path.exists(env["HOME"])
    argv = seen["ARGV"]
    assert argv[:16] == [
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        "-c", "credential.helper=",
        "-c", "http.followRedirects=false",
        "-c", "http.emptyAuth=false",
        "-c", "core.logAllRefUpdates=false",
        "-c", "gc.auto=0",
        "-c", "maintenance.auto=false",
    ]  # fmt: skip
    assert argv[16].startswith("--git-dir=") and argv[17].startswith("--work-tree=")
    assert argv[18:] == ["init", "-q", "--template=", "--"]
    assert work_left(bench) == []


# ----- "--", and the transports git itself refuses ------------------------


def test_the_argv_puts_dashdash_before_the_operands_and_git_honours_it(
    request, stub, tmp_path
):
    """WINDOW: _clone_argv's argv for a fetch whose ref looks like an
    option, run by a real git, with the runner's own dash check passed
    by (it is proved separately below).

    With "--" before the URL, git reads "--upload-pack=touch PWNED" as a
    ref it cannot find: exit 128, nothing fetched. The mutant without it
    hands git an option, which fetches the remote's HEAD and exits 0
    (tests/test_clone_stub.py shows that half). PRE-STATE: the builder's
    "--" is immediately before the operands."""
    repo, _ = repo_for(request, stub)
    tree = tmp_path / "tree"
    tree.mkdir()
    operands = [stub.url(OWNER, repo), "--upload-pack=touch PWNED"]
    argv = main._clone_argv(str(tree), ["fetch", "-q", "--depth", "1"], operands)
    assert argv[-3:] == ["--", *operands]
    home = tmp_path / "home"
    home.mkdir()
    env = main._clone_env(str(home), str(stub.cainfo))
    init = main._clone_argv(str(tree), ["init", "-q", "--template="], [])
    subprocess.run(init, env=env, check=True)
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 128
    fetch_head = tree / ".git" / "FETCH_HEAD"
    assert not fetch_head.exists() or fetch_head.read_text() == ""
    assert not (tree / "PWNED").exists()


def test_the_runner_refuses_an_operand_that_starts_with_a_dash(tmp_path):
    """WINDOW: _git_clone given an operand beginning with '-'.

    The second line behind the door's validation: refused before any
    process starts. PRE-STATE: the tree does not exist, so a process
    that ran would have failed differently."""
    tree = tmp_path / "never"
    assert not tree.exists()
    with pytest.raises(ValueError, match="starts with '-'"):
        asyncio.run(
            main._git_clone(str(tree), ["fetch"], ["-x"], env={}, deadline=float("inf"))
        )
    assert not tree.exists()


def test_file_transport_is_refused_by_git_itself(tmp_path, stub):
    """WINDOW: the clone runner fetching a bare repository by file://.

    The door never sends one (its scheme rule refuses first), and git
    would not fetch it if it did: protocol.allow=never allows https
    alone. PRE-STATE: the repository is on disk at that path."""
    stub.repository(OWNER, "on-disk", {"a.py": b"x\n"})
    bare = stub.root / "srv" / OWNER / "on-disk.git"
    assert bare.is_dir()
    tree = tmp_path / "tree"
    tree.mkdir()
    env = main._clone_env(str(tmp_path), None)

    async def run():
        await main._git_clone(
            str(tree), ["init", "-q"], [], env=env, deadline=float("inf")
        )
        return await main._git_clone(
            str(tree), ["fetch", "-q"], [f"file://{bare}", "main"], env=env,
            deadline=float("inf"),
        )  # fmt: skip

    code, stderr = asyncio.run(run())
    assert code == 128
    assert "transport 'file' not allowed" in stderr


def test_a_redirect_is_refused_and_its_target_never_asked(request, bench, stub):
    """WINDOW: POST /clones of a path the stub redirects to another
    repository it serves.

    Refused naming the redirect, and the target receives no request, so
    a listed host cannot hand the fetch to one nobody listed. PRE-STATE:
    the target is a repository the stub would serve."""
    target, _ = repo_for(request, stub)
    stub.redirects[f"/{OWNER}/moved.git"] = f"https://{stub.host}/{OWNER}/{target}.git"
    before = sum(1 for s in stub.seen if f"/{target}.git" in s.path)
    resp = bench.post("/clones", json={"url": stub.url(OWNER, "moved"), "ref": "main"})
    assert resp.status_code == 422
    assert "redirect" in resp.json()["detail"]
    assert sum(1 for s in stub.seen if f"/{target}.git" in s.path) == before
    assert os.listdir(clone_root(bench)) == []


@pytest.mark.parametrize(
    ("path", "ref", "says"),
    [
        ("/{owner}/nowhere.git", "main", "no repository at that URL"),
        ("/{owner}/{repo}.git", "nope", "no branch or tag by that name"),
    ],
)
def test_what_the_host_does_not_have_is_refused_saying_which(
    request, bench, stub, path, ref, says
):
    """WINDOW: POST /clones of a repository or a ref the stub does not
    have.

    Each a 422 in the door's own words, never git's (which name the URL),
    and nothing left on disk. PRE-STATE: the stub serves the repository
    at main."""
    repo, _ = repo_for(request, stub)
    url = f"https://{stub.host}" + path.format(owner=OWNER, repo=repo)
    resp = bench.post("/clones", json={"url": url, "ref": ref})
    assert resp.status_code == 422
    assert says in resp.json()["detail"]
    assert stub.host not in resp.text
    assert os.listdir(clone_root(bench)) == []


def test_a_host_that_cannot_be_reached_is_a_502_naming_no_url(bench, monkeypatch):
    """WINDOW: POST /clones to a listed host:port nothing listens on.

    502, the door's sentence, and no part of the URL. PRE-STATE: the
    port is closed (a socket bound and released)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(bench.app.state, "clone_hosts", (f"127.0.0.1:{port}",))
    resp = bench.post(
        "/clones", json={"url": f"https://127.0.0.1:{port}/o/r", "ref": "main"}
    )
    assert resp.status_code == 502
    assert str(port) not in resp.text
    assert os.listdir(clone_root(bench)) == []


# ----- bounded: time, bytes, entries ----------------------------------------


def test_a_clone_past_max_clone_seconds_is_killed_and_removed(
    request, bench, stub, monkeypatch
):
    """WINDOW: POST /clones of a path whose upload-pack the stub holds,
    with MAX_CLONE_SECONDS shortened.

    504 naming MAX_CLONE_SECONDS, soon after it and not when the remote
    lets go (the stub holds for 30 s: a runner that waited for git to
    end on its own would answer then); the stall was reached with the
    new directory on disk, and after the refusal the directory is gone
    and every git of the clone is dead; the next clone is accepted.
    PRE-STATE: the clone root is empty."""
    repo, _ = repo_for(request, stub)
    stub.stalled.add(f"/{OWNER}/{repo}.git")
    stub.stall_reached.clear()
    monkeypatch.setattr(main, "MAX_CLONE_SECONDS", 1.5)
    at_stall = []
    stub.on_stall = lambda: at_stall.append(work_left(bench))
    groups = []
    real = asyncio.create_subprocess_exec

    async def recording(*args, **kw):
        proc = await real(*args, **kw)
        groups.append(proc.pid)
        return proc

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", recording)
    assert os.listdir(clone_root(bench)) == []
    started = time.monotonic()
    try:
        resp = clone_of(bench, stub, repo)
    finally:
        elapsed = time.monotonic() - started
        stub.on_stall = None
        stub.stalled.discard(f"/{OWNER}/{repo}.git")
    assert resp.status_code == 504
    assert elapsed < 10
    assert "MAX_CLONE_SECONDS" in resp.json()["detail"]
    assert stub.stall_reached.is_set()
    assert at_stall and at_stall[0] and at_stall[0][0].endswith(".partial")
    assert os.listdir(clone_root(bench)) == []
    for pgid in groups:
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    assert clone_of(bench, stub, repo).status_code == 201


def test_a_checkout_past_max_clone_bytes_is_refused_before_it_is_written(
    request, bench, stub, monkeypatch
):
    """WINDOW: POST /clones of a repository whose tree holds more blob
    bytes than MAX_CLONE_BYTES, shortened, though its pack is small.

    Refused naming MAX_CLONE_BYTES and a figure from the fetched tree
    ("would be checked out"), before checkout writes a file; nothing is
    left. PRE-STATE: the blobs compress, so the fetch alone is under the
    ceiling and the count is what refuses."""
    repo, _ = repo_for(request, stub, {"zeros.txt": b"0" * 300_000})
    monkeypatch.setattr(main, "MAX_CLONE_BYTES", 200_000)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "MAX_CLONE_BYTES (200,000 bytes)" in detail
    assert "would be checked out" in detail
    assert "at least 300,000 bytes" in detail
    assert os.listdir(clone_root(bench)) == []


def test_a_fetch_past_max_clone_bytes_is_stopped_while_it_arrives(
    request, bench, stub, monkeypatch
):
    """WINDOW: POST /clones of a repository whose pack alone is past a
    shortened MAX_CLONE_BYTES, measured every few milliseconds.

    Stopped during the fetch ("was stopped"), before the tree is
    counted, with the figure measured then; nothing is left. PRE-STATE:
    the blobs are random, so the pack is as large as they are."""
    repo, _ = repo_for(
        request, stub, {f"r{i}.bin": os.urandom(1_000_000) for i in range(12)}
    )
    monkeypatch.setattr(main, "MAX_CLONE_BYTES", 1_000_000)
    monkeypatch.setattr(main, "CLONE_POLL_SECONDS", 0.01)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 422
    assert "was stopped" in resp.json()["detail"]
    assert os.listdir(clone_root(bench)) == []


def test_a_checkout_that_grows_past_its_blobs_is_refused_after_it(
    request, bench, stub, monkeypatch
):
    """WINDOW: POST /clones of a tree whose .gitattributes asks for CRLF
    line endings, so checkout writes more bytes than its blobs hold.

    The count from the fetched tree passes (the blobs are under the
    shortened MAX_CLONE_BYTES) and the checkout does not: the directory
    is measured again once checked out ("was checked out"), and removed.
    PRE-STATE: the blobs alone fit and the converted file would not."""
    lines = b"\n" * 150_000
    repo, _ = repo_for(
        request, stub, {".gitattributes": b"* text eol=crlf\n", "lines.txt": lines}
    )
    monkeypatch.setattr(main, "MAX_CLONE_BYTES", 200_000)
    assert len(lines) < 200_000 < 2 * len(lines)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 422
    assert "was checked out" in resp.json()["detail"]
    assert os.listdir(clone_root(bench)) == []


def test_a_tree_past_max_clone_entries_is_refused_naming_it(
    request, bench, stub, monkeypatch
):
    """WINDOW: POST /clones of a tree of more entries than a shortened
    MAX_CLONE_ENTRIES, every file empty.

    A tree of empty files weighs nothing in bytes; the entry ceiling is
    what refuses it. PRE-STATE: its blob bytes are zero."""
    repo, _ = repo_for(request, stub, {f"d/e{i}": b"" for i in range(30)})
    monkeypatch.setattr(main, "MAX_CLONE_ENTRIES", 10)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 422
    assert "MAX_CLONE_ENTRIES (10 entries)" in resp.json()["detail"]
    assert "at least 0 bytes" in resp.json()["detail"]
    assert os.listdir(clone_root(bench)) == []


def test_an_update_refused_leaves_the_old_tree_and_row_as_they_were(
    request, bench, stub, monkeypatch
):
    """WINDOW: a clone, then an update of it whose new tree is past a
    shortened MAX_CLONE_BYTES.

    The refusal removes only the new directory: the old tree, its files
    and its row are untouched, since an update is a fresh clone swapped
    in only when it has succeeded. PRE-STATE: the first clone fits."""
    repo, first = repo_for(request, stub, {"a.py": b"small\n"})
    made = clone_of(bench, stub, repo).json()
    row = dict(bench.app.state.db.execute("SELECT * FROM clones").fetchone())
    stub.repository(OWNER, repo, {"a.py": b"small\n", "big.txt": b"0" * 300_000})
    monkeypatch.setattr(main, "MAX_CLONE_BYTES", 200_000)
    resp = clone_of(bench, stub, repo)
    assert resp.status_code == 422
    assert (Path(made["root"]) / "a.py").read_text() == "small\n"
    assert not (Path(made["root"]) / "big.txt").exists()
    assert dict(bench.app.state.db.execute("SELECT * FROM clones").fetchone()) == row
    assert row["head_sha"] == first
    assert os.listdir(clone_root(bench)) == [Path(made["root"]).name]


# ----- one at a time --------------------------------------------------------


def test_a_second_clone_while_one_runs_is_refused(request, bench, stub):
    """WINDOW: POST /clones while the slot is held.

    409, and nothing reaches the remote. PRE-STATE: the same request is
    accepted once the slot is free."""
    repo, _ = repo_for(request, stub)
    bench.app.state.clone_run["paths"] = ()
    seen = len(stub.seen)
    try:
        resp = clone_of(bench, stub, repo)
    finally:
        bench.app.state.clone_run["paths"] = None
    assert resp.status_code == 409
    assert "one at a time" in resp.json()["detail"]
    assert len(stub.seen) == seen
    assert clone_of(bench, stub, repo).status_code == 201


@pytest.mark.parametrize("where", ["_clone", "_sweep_clone_work", "_fetch_into"])
def test_the_slot_is_free_after_a_clone_that_raised(
    request, bench_env, stub, monkeypatch, where
):
    """WINDOW: a clone that raises at the first line of the work, at its
    first line inside the worker, or midway, then the same clone again.

    The N lesson from the start: the slot is released in a finally whose
    try covers the first line of the work, so the next clone is
    accepted whatever raised. PRE-STATE: the raise is a 500 the test
    client reports rather than an answer, planted after boot (the sweep
    also runs at boot)."""
    repo, _ = repo_for(request, stub)
    real = getattr(main, where)

    def boom(*a, **kw):
        raise RuntimeError(f"planted in {where}")

    with TestClient(
        main.app, base_url="http://localhost", raise_server_exceptions=False
    ) as client:
        monkeypatch.setattr(main, where, boom)
        assert clone_of(client, stub, repo).status_code == 500
        assert client.app.state.clone_run["paths"] is None
        monkeypatch.setattr(main, where, real)
        assert clone_of(client, stub, repo).status_code == 201


def test_a_cancelled_clone_kills_its_git_and_removes_what_it_fetched(
    request, bench, stub, monkeypatch
):
    """WINDOW: the clone worker cancelled while its fetch is held by the
    stub, as a request is when its client goes or the bench shuts down.

    Every git of the clone is dead and the new directory is gone, soon
    after the cancellation and not when the remote lets go (the stub
    holds for 30 s). PRE-STATE: the stall was reached with the directory
    on disk."""
    repo, _ = repo_for(request, stub)
    stub.stalled.add(f"/{OWNER}/{repo}.git")
    stub.stall_reached.clear()
    groups = []
    real = asyncio.create_subprocess_exec

    async def recording(*args, **kw):
        proc = await real(*args, **kw)
        groups.append(proc.pid)
        return proc

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", recording)

    async def scenario():
        task = asyncio.ensure_future(
            main._clone(stub.url(OWNER, repo), "main", str(clone_root(bench)))
        )
        while not stub.stall_reached.is_set():
            await asyncio.sleep(0.02)
        assert work_left(bench)
        cancelled = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - cancelled < 10

    try:
        asyncio.run(scenario())
    finally:
        stub.stalled.discard(f"/{OWNER}/{repo}.git")
    assert os.listdir(clone_root(bench)) == []
    assert groups
    for pgid in groups:
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)


# ----- the snapshot doors and a clone in progress ---------------------------


@pytest.mark.parametrize("door", ["/snapshots", "/snapshots/listing"])
def test_a_snapshot_of_a_tree_a_clone_is_writing_is_refused(request, bench, stub, door):
    """WINDOW: both snapshot doors, with the slot holding a clone's
    directories, on a root at the clone, inside it, and around it
    (BENCH_CLONE_ROOT itself).

    A walk is synchronous and git is not: a tree mid-checkout would be
    stored as a mix of two commits. Each is 409; with the slot free, the
    same requests are answered. PRE-STATE: the clone exists, with a
    subdirectory."""
    repo, _ = repo_for(request, stub, {"a.py": b"x\n", "sub/b.py": b"y\n"})
    made = clone_of(bench, stub, repo).json()
    roots = [made["root"], f"{made['root']}/sub", str(clone_root(bench))]
    for root in roots:
        assert bench.post(
            door, json={"root": root, "patterns": ["**/*.py"]}
        ).status_code in (200, 201)
    bench.app.state.clone_run["paths"] = (made["root"],)
    try:
        for root in roots:
            resp = bench.post(door, json={"root": root, "patterns": ["**/*.py"]})
            assert resp.status_code == 409, root
            assert "POST /clones" in resp.json()["detail"]
    finally:
        bench.app.state.clone_run["paths"] = None


def test_a_root_in_an_unfinished_clone_is_refused_at_any_time(bench):
    """WINDOW: POST /snapshots on a directory named as a clone's work
    directory, with no clone running.

    A new tree before its swap, or an old one after, is not a clone the
    door finished. PRE-STATE: the slot is free and the directory holds a
    file the pattern selects."""
    work = clone_root(bench) / (".%s.partial" % ("a" * 16))
    work.mkdir()
    (work / "a.py").write_text("x\n")
    assert bench.app.state.clone_run["paths"] is None
    resp = bench.post("/snapshots", json={"root": str(work), "patterns": ["*.py"]})
    assert resp.status_code == 409
    assert "not finished" in resp.json()["detail"]


def test_a_directory_is_found_by_what_it_is_not_how_it_is_spelled(tmp_path):
    """WINDOW: _same_or_under on two spellings of one directory.

    On a disk that folds case, "…/ABC" and "…/abc" are one directory and
    must be found as one; on one that does not, the other spelling names
    nothing and is inside nothing. PRE-STATE: which disk this is decides
    which half asserts, and both are asserted where they apply."""
    real = tmp_path / "abcdef"
    (real / "sub").mkdir(parents=True)
    other = tmp_path / "ABCDEF"
    assert main._same_or_under(str(real / "sub"), str(real))
    assert not main._same_or_under(str(real), str(real / "sub"))
    folds = os.path.exists(other)
    assert main._same_or_under(str(other / "sub"), str(real)) is folds


def test_boot_sweeps_the_work_a_crash_left_and_nothing_else(bench_env, tmp_path):
    """WINDOW: the lifespan, booting over a clone root holding a crash's
    leftovers beside a finished clone and a link named like a leftover.

    The two work directories go; the finished clone stays; the link is
    removed as a link and its target is untouched. PRE-STATE: all four
    are on disk."""
    root = Path(bench_env["BENCH_CLONE_ROOT"])
    partial = root / (".%s.partial" % ("a" * 16))
    old = root / (".%s.old" % ("b" * 16))
    done = root / ("c" * 16)
    target = tmp_path / "elsewhere"
    for d in (partial / "x", old, done, target):
        d.mkdir(parents=True)
    (target / "keep.txt").write_text("keep\n")
    link = root / (".%s.old" % ("d" * 16))
    link.symlink_to(target)
    with TestClient(main.app, base_url="http://localhost"):
        pass
    assert sorted(os.listdir(root)) == ["c" * 16]
    assert (target / "keep.txt").read_text() == "keep\n"


# ----- the URL is recorded in the clones row and nowhere else ----------------


@respx.mock
def test_the_url_is_in_the_clones_row_and_in_no_other_record(
    request, bench, stub, caplog
):
    """WINDOW: a clone, a snapshot of it, a comparison citing the
    snapshot, and a second clone replacing the first, then every row of
    every table but clones, every body the bench answered, the request
    it sent upstream, every byte under the clone including .git, the
    refusal of a snapshot of the clone's .git, and the log.

    The URL, its host and its owner/repo pair appear nowhere but the
    clones row: git writes the URL into .git/FETCH_HEAD and the door
    removes it; the reflog, which would name the operator, is off; and
    since O3 a snapshot root inside .git is refused outright
    (ROOT_IN_VCS), so the files are searched where they lie. PRE-STATE:
    the clones row holds the URL, so the search can find it."""
    repo, _ = repo_for(request, stub, {"a.py": b"ANSWER = 42\n"})
    needles = [f"{OWNER}/{repo}", stub.host, f"127.0.0.1:{stub.port}"]
    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "model/alpha",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    made = clone_of(bench, stub, repo).json()
    bodies = [json.dumps(made)]
    snap = bench.post("/snapshots", json={"root": made["root"], "patterns": ["*.py"]})
    assert snap.status_code == 201
    compared = bench.post(
        "/compare",
        json={
            "prompt": "read",
            "models": ["model/alpha"],
            "attachments": [snap.json()["digest"]],
        },
    )
    assert compared.status_code == 200, compared.text
    stub.repository(OWNER, repo, {"a.py": b"ANSWER = 43\n"})
    again = clone_of(bench, stub, repo)
    assert again.status_code == 200
    gitdir = bench.post(
        "/snapshots",
        json={
            "root": f"{made['root']}/.git",
            "patterns": ["HEAD", "config", "shallow", "*HEAD*"],
        },
    )
    assert gitdir.status_code == 403, gitdir.text
    assert gitdir.json()["detail"] == main.ROOT_IN_VCS
    # The two clone responses carry the URL to the caller who sent it,
    # which is the door's answer and not a record; everything else is
    # searched.
    bodies += [snap.text, compared.text, gitdir.text]
    for path in ("/attachments", f"/attachments/{snap.json()['digest']}", "/runs"):
        bodies.append(bench.get(path).text)

    db = bench.app.state.db
    row = dict(db.execute("SELECT * FROM clones").fetchone())
    assert all(n in row["url"] for n in needles[:2])
    tables = [
        r[0]
        for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if r[0] != "clones"
    ]
    assert "snapshot_captures" in tables and "results" in tables
    for table in tables:
        for record in db.execute(f"SELECT * FROM {table}"):
            text = repr(tuple(record))
            assert not any(n in text for n in needles), table
    upstream = [call.request.content.decode() for call in route.calls]
    assert upstream
    files = []
    for dirpath, _, names in os.walk(made["root"]):
        files += [Path(dirpath) / n for n in names]
    assert any(".git" in str(f) for f in files)
    for text in (
        bodies[1:] + upstream + [f.read_bytes().decode("latin-1") for f in files]
    ):
        assert not any(n in text for n in needles)
    assert not (Path(made["root"]) / ".git" / "FETCH_HEAD").exists()
    assert not (Path(made["root"]) / ".git" / "logs").exists()
    assert not any(n in caplog.text for n in needles)


# ----- which clone a snapshot read (the clone id on the capture) ------------


def test_a_snapshot_names_the_clone_its_root_is_in(request, bench, stub):
    """WINDOW: POST /snapshots at five roots, and the clone_id on each
    capture it returns.

    The clone's own directory and a directory inside it name the clone;
    BENCH_CLONE_ROOT itself, which holds every clone and is in none,
    names nothing; and so does a root beside the clones. The clone's
    .git is refused as a root since O3 (ROOT_IN_VCS). PRE-STATE: the
    clone is recorded and each root but .git is allowed."""
    repo, _ = repo_for(request, stub, {"a.py": b"x\n", "sub/b.py": b"y\n"})
    made = clone_of(bench, stub, repo).json()
    beside = clone_root(bench) / "beside"
    beside.mkdir()
    (beside / "c.py").write_text("z\n")
    assert [c["id"] for c in main.store.list_clones(bench.app.state.db)] == [made["id"]]
    cases = [
        (made["root"], ["*.py"], made["id"]),
        (f"{made['root']}/sub", ["*.py"], made["id"]),
        (str(clone_root(bench)), ["**/*.py"], None),
        (str(beside), ["*.py"], None),
    ]
    gitdir = bench.post(
        "/snapshots", json={"root": f"{made['root']}/.git", "patterns": ["HEAD"]}
    )
    assert gitdir.status_code == 403
    for root, patterns, expected in cases:
        resp = bench.post("/snapshots", json={"root": root, "patterns": patterns})
        assert resp.status_code == 201, (root, resp.text)
        assert resp.json()["capture"]["clone_id"] == expected, root


def test_a_clone_is_named_under_any_spelling_of_its_directory(request, bench, stub):
    """WINDOW: POST /snapshots at the clone's directory spelled in capitals.

    On a disk that folds case the capitals are the same directory and
    name the same clone; on one that does not, they name nothing and the
    door says so. PRE-STATE: which disk this is decides which half
    asserts."""
    repo, _ = repo_for(request, stub)
    made = clone_of(bench, stub, repo).json()
    upper = str(Path(made["root"]).parent / Path(made["root"]).name.upper())
    assert upper != made["root"]
    resp = bench.post("/snapshots", json={"root": upper, "patterns": ["*.py"]})
    if os.path.exists(upper):
        assert resp.json()["capture"]["clone_id"] == made["id"]
    else:
        assert resp.status_code == 422


@respx.mock
def test_the_clone_id_travels_as_head_does_and_differs_in_nothing_else(
    request, bench_env, stub, monkeypatch, tmp_path
):
    """WINDOW: one tree snapshotted twice, as a clone and as a byte-copy of
    it (.git included) under a second allowlisted root; each snapshot then
    compared by hand in a group and cited by an experiment that ran; the
    capture as POST /snapshots, GET /runs/{id}, GET /groups/{id}, GET
    /runs (the group's entry), the report and the export each carry it.

    DECLARATION TRANSPORT, the house law's test. On every surface the two
    captures differ in clone_id (the clone's id, and a present null) and
    in nothing else but the id and captured_at any two captures differ
    in: the same digest, head, dirty flag, patterns and exclusions.
    PRE-STATE: the copy is git-identical, so head and dirty agree and a
    difference in them would be the bug."""
    import shutil

    from test_api import (
        TEST_CATALOG,
        experiment_body,
        read_export,
        response_for,
        run_experiment_to_completion,
        write_dataset,
    )

    async def catalog(client):
        return json.loads(json.dumps(TEST_CATALOG))

    monkeypatch.setattr("bench.main.fetch_catalog", catalog)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(
        "BENCH_REPO_ROOTS",
        os.pathsep.join([bench_env["BENCH_CLONE_ROOT"], str(elsewhere)]),
    )
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, json=response_for("model/alpha", "ok"))
    )
    repo, _ = repo_for(request, stub, {"a.py": b"ANSWER = 42\n"})
    with TestClient(main.app, base_url="http://localhost") as client:
        made = clone_of(client, stub, repo).json()
        plain = elsewhere / "copy"
        shutil.copytree(made["root"], plain, symlinks=True)

        surfaces = {}
        for label, root in (("clone", made["root"]), ("plain", str(plain))):
            snap = client.post("/snapshots", json={"root": root, "patterns": ["*.py"]})
            assert snap.status_code == 201, snap.text
            digest, capture = snap.json()["digest"], snap.json()["capture"]
            gid = client.post(
                "/groups",
                json={
                    "budget": "standard",
                    "prompt": "read",
                    "models": ["model/alpha"],
                    "attachments": [digest],
                },
            ).json()["id"]
            run_id = client.post(
                "/compare",
                json={
                    "prompt": "read",
                    "models": ["model/alpha"],
                    "attachments": [digest],
                    "group_id": gid,
                },
            ).json()["run_id"]
            path = write_dataset(
                tmp_path,
                {"id": "t1", "prompt": "read", "attachments": [digest]},
                name=f"{label}.jsonl",
            )
            eid = client.post(
                "/experiments", json=experiment_body(path, lineup=["model/alpha"])
            ).json()["id"]
            run_experiment_to_completion(client, eid, path)
            report = client.get(
                f"/experiments/{eid}/report", params={"dataset_path": path}
            )
            export = json.loads(read_export(client, eid).decode().splitlines()[0])
            listed = [
                r
                for r in client.get("/runs").json()["runs"]
                if r.get("type") == "group" and r["id"] == gid
            ]
            surfaces[label] = {
                "digest": digest,
                "POST /snapshots": capture,
                "GET /runs/{id}": client.get(f"/runs/{run_id}").json()["attachments"][
                    0
                ]["capture"],
                "GET /groups/{id}": client.get(f"/groups/{gid}").json()["attachments"][
                    0
                ]["capture"],
                "GET /runs": listed[0]["attachments"][0]["capture"],
                "report": report.json()["captures"][str(capture["id"])],
                "export": export["captures"][str(capture["id"])],
            }

    clone, plain_ = surfaces.pop("clone"), surfaces.pop("plain")
    assert clone.pop("digest") == plain_.pop("digest")
    assert clone["POST /snapshots"]["head"] is not None
    assert clone["POST /snapshots"]["dirty"] is False
    for surface in clone:
        a, b = clone[surface], plain_[surface]
        assert a["clone_id"] == made["id"], surface
        assert "clone_id" in b and b["clone_id"] is None, surface
        differing = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        assert differing == {"id", "captured_at", "clone_id"}, surface
