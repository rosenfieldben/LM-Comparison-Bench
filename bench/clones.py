"""The clone door's pure half: what a request may name, and where it lands.

POST /clones fetches a public repository into a directory the operator
named, and it is the one door in this bench that reaches a host other
than OpenRouter. Everything a request says is checked here, as a value
in and a value out, before any process runs: the URL is parsed and
rebuilt rather than passed on, the ref is held to git's own grammar, and
the directory is a function of the two. The door in bench.main owns the
filesystem, the subprocess and the record.

NO SENTENCE HERE ECHOES THE URL. A URL can carry a user and a token, and
a refusal that quoted it back would print the secret on the page, in the
log and in any report of the refusal. Each sentence names the rule the
URL broke and nothing of the URL. The ref is not a secret and is named.
"""

import hashlib
import re
from collections.abc import Sequence
from urllib.parse import SplitResult, urlsplit

__all__ = [
    "CLONES_OFF",
    "DEFAULT_CLONE_HOSTS",
    "MAX_REF_CHARS",
    "MAX_URL_CHARS",
    "CloneError",
    "canonical_url",
    "check_ref",
    "clone_dir_name",
    "parse_clone_hosts",
    "repository_identity",
]

# The door's off sentence, the sibling of SNAPSHOTS_OFF and worded the
# same way: which variable, why it exists, and what setting it does.
CLONES_OFF = (
    "clones are off because BENCH_CLONE_ROOT is not set. A clone fetches "
    "a repository from the network onto this machine, so where it may "
    "land is a directory the operator names rather than whatever path a "
    "request picks, and that directory must be one of BENCH_REPO_ROOTS so "
    "the snapshot doors may walk what lands there. Set BENCH_CLONE_ROOT to "
    "that directory and restart the bench."
)

# The hosts a clone may fetch from unless BENCH_CLONE_HOSTS says
# otherwise. One, because the phase is about GitHub.
DEFAULT_CLONE_HOSTS = ("github.com",)

# Bounds on what a request may carry, refused by the request model
# before anything here reads it. A URL of this shape is a host and two
# path segments; a ref is a branch, a tag or a 40-hex commit.
MAX_URL_CHARS = 2048
MAX_REF_CHARS = 255

# GitHub's own spellings: an owner is letters, digits and single inner
# hyphens, up to 39; a repository is letters, digits, '.', '-' and '_'.
# Stricter than git needs, on purpose: every URL the door accepts is one
# a person could have copied from a repository page.
_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPO = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,99}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?")

# What a refused ref may hold and still be repeated in its refusal.
_NAMEABLE_REF = re.compile(r"[A-Za-z0-9._/+-]{1,100}")

# The characters git's check-ref-format forbids anywhere in a name.
_REF_FORBIDDEN = frozenset(" ~^:?*[\\")


class CloneError(ValueError):
    """A clone request the bench will not run, with the sentence to show.

    The door answers it as 422 (the request's shape) or 403 (the
    operator's policy), whichever the raiser names.
    """

    def __init__(self, sentence: str, status: int = 422) -> None:
        super().__init__(sentence)
        self.status = status


def parse_clone_hosts(raw: str | None) -> tuple[str, ...]:
    """BENCH_CLONE_HOSTS as the door checks against it.

    Comma-separated host[:port] entries, lowercased; unset or blank is
    the default. An entry without a port matches only a URL without one,
    so a port is allowed exactly where the operator wrote it. A malformed
    entry fails boot naming the variable, as a malformed BENCH_REPO_ROOTS
    entry does, rather than becoming a host nothing can match.
    """
    if raw is None or not raw.strip():
        return DEFAULT_CLONE_HOSTS
    hosts: list[str] = []
    for item in raw.split(","):
        entry = item.strip().lower()
        host, colon, port = entry.partition(":")
        if not _HOST.fullmatch(host) or (colon and not _valid_port(port)):
            raise RuntimeError(
                f"BENCH_CLONE_HOSTS entry {item.strip()!r} is not a host name "
                "or a host:port. Each entry is a host a clone may fetch from, "
                "such as github.com."
            )
        if entry not in hosts:
            hosts.append(entry)
    return tuple(hosts)


def _valid_port(port: str) -> bool:
    return port.isdigit() and not port.startswith("0") and 0 < int(port) < 65536


def canonical_url(url: str, hosts: Sequence[str]) -> str:
    """The URL git will be given, rebuilt from its parts, or a refusal.

    PARSED, NOT MATCHED: the URL is split into its parts and each part is
    held to its own rule, and what git receives is rebuilt from the parts
    that passed. A string that merely looked right could smuggle a second
    host, a user, a query or an option past a pattern; a rebuilt URL
    carries nothing that was not checked. Every refusal names the rule
    and none echoes the URL.
    """
    if url != url.strip() or any(ord(c) < 0x21 or ord(c) == 0x7F for c in url):
        raise CloneError(
            "the URL holds whitespace or a control character. A clone URL "
            "is https://host/owner/repo with nothing around it."
        )
    if "\\" in url or "%" in url:
        raise CloneError(
            "the URL holds a backslash or a percent-escape. A clone URL is "
            "spelled plainly, https://host/owner/repo, so what is checked is "
            "what is fetched."
        )
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise CloneError("the URL cannot be read as https://host/owner/repo.") from None
    if parts.scheme != "https":
        raise CloneError(
            "the URL is not https. A clone is fetched over https and nothing "
            "else: no ssh, no git protocol, no plain http and no file path.",
            403,
        )
    if "@" in parts.netloc or parts.username is not None or parts.password is not None:
        raise CloneError(
            "the URL carries a user or a token before its host. A clone is of "
            "a public repository with no credential on this path, so a URL "
            "holding one is refused, and it is not repeated here.",
            403,
        )
    if parts.query or parts.fragment or url.endswith(("?", "#")):
        raise CloneError(
            "the URL has a query or a fragment. A clone URL is "
            "https://host/owner/repo and nothing after it."
        )
    host = (parts.hostname or "").lower()
    entry = f"{host}:{port}" if port is not None else host
    if not _HOST.fullmatch(host) or entry not in hosts:
        raise CloneError(
            "the URL's host is not one of BENCH_CLONE_HOSTS "
            f"({', '.join(hosts)}). A clone reaches only the hosts the "
            "operator listed.",
            403,
        )
    owner, repo = _owner_and_repo(parts)
    return f"https://{entry}/{owner}/{repo}"


def _owner_and_repo(parts: SplitResult) -> tuple[str, str]:
    segments = parts.path.split("/")
    if len(segments) != 3 or segments[0] != "":
        raise CloneError(
            "the URL's path is not /owner/repo or /owner/repo.git. A clone "
            "names one repository, and nothing under it."
        )
    owner, repo = segments[1], segments[2]
    if not _OWNER.fullmatch(owner) or "--" in owner:
        raise CloneError(
            "the URL's owner is not a GitHub owner name: letters, digits and "
            "single inner hyphens, at most 39."
        )
    name = repo[: -len(".git")] if repo.endswith(".git") else repo
    if not _REPO.fullmatch(name) or name.endswith(".git"):
        raise CloneError(
            "the URL's repository is not a repository name: letters, digits, "
            "'.', '-' and '_', not starting with '.' or '-', at most 100."
        )
    return owner, repo


def check_ref(ref: str) -> None:
    """A branch or tag name git would accept, or a 40-hex commit; or a
    refusal naming the rule.

    GIT'S OWN GRAMMAR (check-ref-format, for refs/heads/<ref>), held to
    git itself by a test over generated names, plus three rules of the
    door's. No leading '-': the argument-injection line (a ref is the
    last argument git is given, after '--' and after the URL, and is
    still refused here so no layer depends on another). No leading '+':
    fetch reads it as its force flag and fetches the name after it, so
    "+main" would land main's tree under a record saying "+main". Not
    '@' alone, which git reads as HEAD rather than as a branch or tag.
    """
    if _HEX40.fullmatch(ref):
        return
    problem = _ref_problem(ref)
    if problem is not None:
        # Named back only when it is spelled like a ref. A person who
        # swapped the two fields sent the URL here, token and all, and
        # the rule that no refusal repeats a URL holds in this field too.
        named = f"ref {ref!r}" if _NAMEABLE_REF.fullmatch(ref) else "the ref"
        raise CloneError(
            f"{named} is not a branch or tag name git accepts, nor a "
            f"40-character commit: {problem}."
        )


def _ref_problem(ref: str) -> str | None:
    if ref == "":
        return "it is empty"
    if ref.startswith("-"):
        return "it starts with '-', which git would read as an option"
    if ref.startswith("+"):
        return "it starts with '+', which fetch reads as its force flag"
    if ref == "@":
        return "'@' alone names HEAD, not a branch or a tag"
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in ref):
        return "it holds a control character"
    if any(c in _REF_FORBIDDEN for c in ref):
        return "it holds a space or one of ~ ^ : ? * [ \\"
    if ".." in ref:
        return "it holds '..'"
    if "@{" in ref:
        return "it holds '@{'"
    if ref.startswith("/") or ref.endswith("/") or "//" in ref:
        return "it starts or ends with '/', or holds '//'"
    if ref.endswith("."):
        return "it ends with '.'"
    for part in ref.split("/"):
        if part.startswith("."):
            return "a part of it starts with '.'"
        if part.endswith(".lock"):
            return "a part of it ends with '.lock'"
    return None


def repository_identity(url: str) -> str:
    """The canonical URL as the name of a repository: without ".git".

    https://host/owner/repo and https://host/owner/repo.git are one
    repository on every forge that serves both, so they are one clone,
    one directory and one row. The fetch still uses the spelling that
    was checked; only the identity drops the suffix. Nothing else is
    folded: owner and repository case, and a ref named two ways (main
    and refs/heads/main), are left as the person wrote them, because
    whether they coincide is the host's business and the door does not
    branch on a host.
    """
    return url[: -len(".git")] if url.endswith(".git") else url


def clone_dir_name(identity: str, ref: str) -> str:
    """The directory a clone of this repository at this ref lands in.

    A FUNCTION OF THE REQUEST: sha256 of the repository's identity, a
    newline and the ref, sixteen hex characters. The same repository at
    the same ref always has one place, and a person can find it from the
    record.
    """
    return hashlib.sha256(f"{identity}\n{ref}".encode()).hexdigest()[:16]
