"""A git remote over HTTPS on loopback, for the clone door's tests.

The door fetches with `-c protocol.allow=never -c protocol.https.allow=always`,
so a file:// remote is refused by design (and that refusal is itself a
test). What a test clones from is this: bare repositories behind git's
own `git http-backend`, served by http.server over TLS with a
certificate made for the session, which the bench's git trusts through
GIT_SSL_CAINFO (BENCH_CLONE_CAINFO, a test seam) and nothing else does.

SMART HTTP, NOT DUMB, and that is the commission's stub changed rather
than followed. The door fetches with --depth 1, and git refuses a
shallow fetch over the dumb protocol outright ("dumb http transport
does not support shallow capabilities", git 2.50.1). A dumb stub could
only have tested a door that fetched whole histories. http-backend is
git's own server side, so what answers here is what answers on a real
host, minus the host.

THE CERTIFICATE IS MADE, NOT COMMITTED: openssl makes a one-day
self-signed certificate for 127.0.0.1 per session, so no private key
sits in the repository and nothing here expires on a date somebody has
to remember.

Four behaviours a test can ask of a repository path, each one a thing
a real remote does: answer 401 (a private repository), stall until
released (a slow one, for the timeout), redirect (to a host the bench
did not name), or serve the repository. Every request is recorded with
the Authorization header it carried, if any, which is how a test shows
that no credential was offered.
"""

import gzip
import http.server
import os
import shutil
import ssl
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# A commit's identity and time, fixed so a repository built twice has
# the same shas.
COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "stub",
    "GIT_AUTHOR_EMAIL": "stub@example.invalid",
    "GIT_COMMITTER_NAME": "stub",
    "GIT_COMMITTER_EMAIL": "stub@example.invalid",
    "GIT_AUTHOR_DATE": "2026-09-25T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-09-25T00:00:00+00:00",
}


def isolated_git_env(home: Path) -> dict[str, str]:
    """An environment for the TESTS' own git, reading no config of the
    developer's: a commit.gpgsign or an init.templateDir in somebody's
    ~/.gitconfig would otherwise change the repositories a test builds."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        **COMMIT_ENV,
    }


def git(env: dict[str, str], *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def make_certificate(where: Path) -> tuple[Path, Path]:
    """A self-signed certificate for 127.0.0.1 and its key, made now."""
    cert, key = where / "cert.pem", where / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", "1", "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
            "-keyout", str(key), "-out", str(cert),
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    return cert, key


@dataclass
class Seen:
    """One request the stub read, and what it did with it."""

    method: str
    path: str
    authorization: str | None


@dataclass
class Stub:
    """The running remote. Repository paths are "/owner/repo.git"."""

    root: Path
    port: int
    cainfo: Path
    env: dict[str, str]
    private: set[str] = field(default_factory=set)
    stalled: set[str] = field(default_factory=set)
    redirects: dict[str, str] = field(default_factory=dict)
    seen: list[Seen] = field(default_factory=list)
    stall_reached: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    # Called on the stub's thread when a stall is reached, before it
    # holds: a test's look at the world at that moment.
    on_stall: Callable[[], None] | None = None

    @property
    def host(self) -> str:
        """The BENCH_CLONE_HOSTS entry that admits this stub."""
        return f"127.0.0.1:{self.port}"

    def url(self, owner: str, repo: str) -> str:
        return f"https://{self.host}/{owner}/{repo}.git"

    def repository(
        self,
        owner: str,
        repo: str,
        files: dict[str, bytes],
        *,
        branch: str = "main",
        tag: str | None = None,
    ) -> str:
        """A bare repository at /owner/repo.git whose `branch` has these
        files, and that commit's sha. Called again, it commits the new
        files on top."""
        bare = self.root / "srv" / owner / f"{repo}.git"
        work = self.root / "work" / owner / repo
        if not bare.exists():
            bare.parent.mkdir(parents=True, exist_ok=True)
            git(self.env, "init", "-q", "--bare", "-b", "main", str(bare))
            work.mkdir(parents=True)
            git(self.env, "init", "-q", "-b", branch, str(work))
        for name in os.listdir(work):
            if name != ".git":
                target = work / name
                shutil.rmtree(target) if target.is_dir() else target.unlink()
        for name, data in files.items():
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        w = str(work)
        git(self.env, "-C", w, "add", "-A")
        git(self.env, "-C", w, "commit", "-q", "--allow-empty", "-m", "stub")
        if tag is not None:
            git(self.env, "-C", w, "tag", "-f", tag)
        git(self.env, "-C", w, "push", "-q", "--force", "--tags", str(bare), branch)
        return git(self.env, "-C", w, "rev-parse", "HEAD")


def _read_body(handler: http.server.BaseHTTPRequestHandler) -> bytes:
    # git sends a POST chunked past http.postBuffer (1 MiB).
    if handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
        body = bytearray()
        while True:
            size = int(handler.rfile.readline().split(b";")[0].strip(), 16)
            if size == 0:
                while handler.rfile.readline() not in (b"\r\n", b"\n", b""):
                    pass
                return bytes(body)
            body += handler.rfile.read(size)
            handler.rfile.readline()
    length = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(length) if length else b""


def _handler(stub: Stub) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            # The handshake on this thread, bounded, rather than in the
            # accept loop, where one peer that never speaks TLS would
            # freeze every later clone.
            self.request.settimeout(30)
            self.request.do_handshake()
            super().setup()

        def _serve(self) -> None:
            path, _, query = self.path.partition("?")
            repo = "/".join(path.split("/")[:3])
            stub.seen.append(
                Seen(self.command, path, self.headers.get("Authorization"))
            )
            body = _read_body(self)
            if repo in stub.stalled and self.command == "POST":
                if stub.on_stall is not None:
                    stub.on_stall()
                stub.stall_reached.set()
                # Bounded, and once: a door that failed to time out
                # fails its test in half a minute, and git's retry of
                # the request is served rather than held again.
                stub.release.wait(timeout=30)
                stub.stalled.discard(repo)
                self.close_connection = True
                return
            if repo in stub.redirects:
                location = stub.redirects[repo] + path[len(repo) :]
                location += f"?{query}" if query else ""
                self._answer(302, [("Location", location)], b"")
                return
            if repo in stub.private:
                self._answer(401, [("WWW-Authenticate", 'Basic realm="stub"')], b"")
                return
            if self.headers.get("Content-Encoding") == "gzip":
                # git gzips a request body past 1 KiB; http-backend can
                # decode it too, but decoding here keeps CONTENT_LENGTH
                # the length of what it reads.
                body = gzip.decompress(body)
            env = {
                "PATH": stub.env["PATH"],
                "HOME": stub.env["HOME"],
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_PROJECT_ROOT": str(stub.root / "srv"),
                "GIT_HTTP_EXPORT_ALL": "1",
                "REQUEST_METHOD": self.command,
                "PATH_INFO": path,
                "QUERY_STRING": query,
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(len(body)),
                "GIT_PROTOCOL": self.headers.get("Git-Protocol", ""),
                "REMOTE_ADDR": "127.0.0.1",
            }
            proc = subprocess.run(
                ["git", "http-backend"],
                input=body,
                env=env,
                capture_output=True,
                check=False,
            )
            head, _, payload = proc.stdout.partition(b"\r\n\r\n")
            status, headers = 200, []
            for line in head.split(b"\r\n"):
                name, _, value = line.decode("latin-1").partition(":")
                if name.lower() == "status":
                    status = int(value.split()[0])
                elif name:
                    headers.append((name, value.strip()))
            self._answer(status, headers, payload)

        def _answer(
            self, status: int, headers: list[tuple[str, str]], payload: bytes
        ) -> None:
            # Every answer carries its length: a body delimited by the
            # connection closing, without TLS close_notify, is an error
            # to OpenSSL 3 and GnuTLS alike, and git would report a
            # transport failure where the test meant a 401.
            try:
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass  # the bench killed its git; nobody is listening

        do_GET = _serve
        do_POST = _serve

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return Handler


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # a git the bench killed mid-request resets its connection


@contextmanager
def serving(root: Path) -> Iterator[Stub]:
    """A stub remote on an ephemeral loopback port, for the with-block."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "home").mkdir(exist_ok=True)
    cert, key = make_certificate(root)
    stub = Stub(root=root, port=0, cainfo=cert, env=isolated_git_env(root / "home"))
    server = _Server(("127.0.0.1", 0), _handler(stub))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(
        server.socket, server_side=True, do_handshake_on_connect=False
    )
    stub.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        stub.release.set()
        server.shutdown()
        server.server_close()
