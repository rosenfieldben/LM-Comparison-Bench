"""Repository snapshots. Pure functions over a caller-supplied tree.

THE SIBLING OF bench.extract, and the same argument holds. A comparison
over a snapshot is a comparison over THIS module's reading of a
repository: which files were selected, in which order, and with what
around them. If the reading is poor every model is asked a poorer
question, and they are all asked the same poorer question, which is
what keeps the comparison fair even when the reading is bad.

SO THE WALKER'S IDENTITY IS PROVENANCE, exactly as a parser's is. A
snapshot lands as one attachment whose extractor is this module and
whose version changes by hand when the composition rule changes, so a
record can always say which walk produced the text a model read.

IT IMPORTS bench.extract AND NOTHING ELSE FROM bench, and that bound is
the phase's, not a preference. extract holds the vocabulary a rendition
is spoken in: the kinds, the composed ceiling, the image signatures. A
second copy of any of those here would be a second number to drift, and
the one thing this codebase has repeatedly paid for is a constant that
existed twice. extract itself imports nothing from bench, so the edge
is one directed hop and there is no cycle to reason about.

NO I/O AND NO CLOCK. The walk is expressed over two injected callables
and the composition over bytes the caller already read, so every case
below is a value in and a value out. The caller owns the filesystem,
the allowlist that says which roots may be walked at all, and the
decision to store anything.
"""

import fnmatch
import hashlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from bench.extract import (
    IMAGE_SIGNATURES,
    MAX_COMPOSED_CHARS,
    SNAPSHOT_KIND,
)

# The extractor name and version that land on the attachment row.
#
# "repo-walk" rather than a library name because no library did this:
# the selection, the ordering and the delimiters are written here, and
# the honest label for that is the code's own name, matching
# "stdlib-zipfile-xml" next door.
#
# THE VERSION IS BUMPED BY HAND WHEN THE COMPOSED TEXT WOULD CHANGE for
# the same tree, which is the same rule the docx reader's version
# follows and for the same reason: renditions are keyed on (digest,
# extractor, extractor_version), so a changed reading served under an
# unchanged version is false provenance arriving through the one door
# the key cannot see. Changing a header's wording, the ordering rule,
# the encoding rule or the default exclusions all qualify. Widening what
# is REFUSED does not, because a refusal produces no rendition.
SNAPSHOT_EXTRACTOR = "repo-walk"
SNAPSHOT_VERSION = "1"

# How every file in a snapshot is turned into text, stated once and
# recorded in the manifest so a reader of the record never has to guess.
#
# STRICT, NOT REPLACING. Decoding with errors="replace" would let a file
# the bench cannot actually read reach a model as a field of U+FFFD, and
# the model would answer about it. A file that is not UTF-8 is refused
# by name instead, which is the same choice extract makes for a scanned
# PDF: better a refusal the person can act on than a prompt nobody can
# reconstruct.
ENCODING_RULE = "UTF-8, strict, with a leading byte-order mark removed"

UTF8_BOM = b"\xef\xbb\xbf"

# What is never in a snapshot, applied to directories during the walk so
# an excluded tree is not even traversed, and recorded in the manifest
# so the record says what the walk was told to ignore.
#
# THREE GROUPS, AND THE THIRD IS NOT IN THE COMMISSION'S LIST. The first
# is version-control and dependency trees, the second is build output
# and caches including the bench's own database files, and both are
# there because they are enormous and say nothing about the code a
# person wants compared.
#
# THE THIRD GROUP IS SECRETS, and it is here because of what this
# feature does with what it selects: it composes file contents into a
# prompt and sends that prompt to a provider. Every other reader in this
# codebase is handed one file a person chose deliberately; this one is
# handed a pattern and gathers whatever matches. A .env swept up by
# "**/*" is an API key posted to a third party by a person who thought
# they were sharing source, and no refusal downstream can unsend it.
# These entries cost nothing when they match nothing and are the only
# defense at the moment the mistake is cheap.
#
# NOT OVERRIDABLE BY THE CALLER, deliberately. The request shape is a
# root and include patterns; adding an exclusion override would make the
# secret group opt-out, and an opt-out default is not a default. A
# caller who genuinely needs a .pem in a comparison can paste it.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    # Version control and dependency trees.
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    # The one entry in this group that could name real source rather
    # than somebody else's. It is here because when it is not source it
    # is a Go or Composer dependency tree and is enormous, and because
    # the exclusions are recorded in every snapshot's manifest, so a
    # person whose vendor/ was their own can see that it was skipped.
    "vendor",
    # Build output, caches and the bench's own database.
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    # Found by walking this repository rather than by listing what a
    # cache directory is usually called: "**/*" over the clone refused
    # on .hypothesis/examples/04e6b34.../e270cb2..., which is a binary
    # blob a property test wrote. Its siblings are here for the same
    # reason before they are found the same way.
    ".hypothesis",
    ".coverage",
    "htmlcov",
    ".tox",
    "dist",
    "build",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.db",
    "*.db-wal",
    "*.db-shm",
    ".DS_Store",
    # Secrets. See above for why this group is here at all.
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    ".pypirc",
)

# How many include patterns one request may carry. Twenty is past any
# real selection and keeps the per-file matching loop bounded by a
# number the caller cannot choose.
MAX_PATTERNS = 20

# How many directory entries one walk may look at before it gives up.
#
# THE WALK IS CALLER-DIRECTED AND RUNS INSIDE A REQUEST, which is the
# same sentence that put a bound on docx nesting and on zip inflation.
# A root pointed at a home directory is the ordinary mistake, and the
# difference between refusing it in two seconds and traversing it for a
# minute is the whole reason this number exists. Twenty thousand entries
# is several times this repository and well under a wait a person would
# notice.
MAX_WALKED_ENTRIES = 20_000

# The largest single file a snapshot may carry, in BYTES, and the same
# number as the composed ceiling counts in CHARACTERS.
#
# THE TWO UNITS ARE THE REASON THIS IS A REAL BOUND rather than a nicer
# message for a case the total would catch anyway. On ASCII they are the
# same count and the total does subsume this; on multibyte text they are
# not, and a 200 KB file of CJK source decodes to well under 200,000
# characters and would pass the total. Refusing it here is deliberate:
# the bound on ONE MEMBER is about what a person should be selecting,
# and a single file that large is a data file that wandered into a
# source pattern whatever it encodes to.
#
# BYTES ALSO BECAUSE THIS RUNS BEFORE THE DECODE, which is the only
# place a size can be checked without having already paid for it.
#
# IT IS THE DOOR'S CONTRACT TOO. compose is handed bytes the caller
# already read, so it cannot un-read a file; a boundary that read
# without checking a size first would have spent the memory before
# asking. The door applies this bound before reading, and compose
# applies it again because a pure function does not assume its caller.
MAX_MEMBER_BYTES = MAX_COMPOSED_CHARS

# The delimiters around each file in the composed snapshot.
#
# ONE HEADER LINE CARRYING PATH, SIZE AND SHORT DIGEST, which is what
# makes "nothing was truncated" checkable rather than asserted: a reader
# with the header and the block can weigh one against the other. Twelve
# hex characters of the digest, the prefix every refusal in this
# codebase quotes, so a person reading a prompt and a record side by
# side is reading the same identifier.
#
# THE FOOTER IS THE PROMPT-INJECTION BOUNDARY, the same job it does for
# an attachment. A repository is untrusted input in exactly the way a
# document is, and more so: a file in it may contain a line that looks
# like the next file's header. With a closing marker the models can tell
# where a file ended; without one, a file could annex its successor.
SNAPSHOT_HEADER = (
    "----- file {index} of {total}: {path}, {size} bytes, sha256 {short} -----"
)
SNAPSHOT_FOOTER = "----- end file {index} of {total} -----"

# What the models are told about the reading before they see it.
#
# THE ORDER AND THE ENCODING ARE PART OF THE QUESTION. A model asked to
# review a repository is being shown a selection, and a selection that
# does not say it is one invites an answer about a codebase that was
# never sent. "Nothing is truncated" is the load-bearing clause: it is
# true by construction here, and it tells a model not to hedge about
# what it cannot see inside a file.
#
# THE COMMIT AND THE DIRTY FLAG ARE DELIBERATELY ABSENT from this text
# and present in the manifest. They are provenance for the record, not
# material for the answer: no model can do anything with a sha, and
# every model would pay for the tokens.
SNAPSHOT_INTRO = (
    "A snapshot of {count} file{plural} from a source repository "
    "follows. Paths are relative to the repository root and the files "
    "are in path byte order. Each file's content is verbatim, decoded "
    "as {encoding}. Nothing is truncated: a file too large for the "
    "snapshot is refused rather than shortened."
)


class SnapshotError(RuntimeError):
    """A snapshot the bench will not build.

    RuntimeError rather than a custom hierarchy, matching
    ExtractionError, CompositionError and DatasetError: the caller's
    only sensible response is to show the message, and every message
    below is written to be shown.
    """


def _plural(count: int) -> str:
    """The suffix that makes a count read: empty for one, "s" otherwise.

    Named rather than written inline at its one call site, because a
    ternary buried in a format call's argument list is where this
    codebase has already got the same expression wrong: Python 3.11
    forbids nesting the same quotes inside an f-string, and the
    workaround was a helper there too.
    """
    return "" if count == 1 else "s"


def matches(path: str, pattern: str) -> bool:
    """Whether a repo-relative path matches a repo-relative glob.

    PATH GLOB SEMANTICS, NOT fnmatch's. fnmatch's "*" happily crosses a
    directory separator, so "*.py" would match "a/b/c.py" and an include
    pattern could reach depths the person did not write. Here the
    pattern and the path are split on "/" and matched segment by
    segment, so "*" and "?" stay inside one segment and "**" is the only
    way to cross one. fnmatchcase does the within-segment work, on a
    string with no "/" in it, where its one surprising behaviour cannot
    fire. Case is never folded, so the same tree matches identically on
    every platform.

    "**" MATCHES ZERO OR MORE SEGMENTS, so "**/*.py" finds "main.py" as
    well as "a/b/main.py", and "src/**" is every file under src. There
    is no implicit recursion anywhere else: "*.py" is the top level
    only. Explicit beats convenient here because the refusal for an
    empty selection names which patterns matched nothing, so a person
    who expected recursion learns it in one round trip.

    The table below is a reachability matrix rather than the obvious
    recursion because the obvious recursion is exponential on a pattern
    like "**/**/**/x", and patterns arrive from the caller.
    """
    parts = path.split("/")
    pats = pattern.split("/")
    rows, cols = len(parts), len(pats)
    # reach[i][j] is True when parts[i:] can be matched by pats[j:].
    reach = [[False] * (cols + 1) for _ in range(rows + 1)]
    reach[rows][cols] = True
    for i in range(rows, -1, -1):
        for j in range(cols - 1, -1, -1):
            if pats[j] == "**":
                # Consume no segment, or consume this one and stay.
                reach[i][j] = reach[i][j + 1] or (i < rows and reach[i + 1][j])
            elif i < rows and fnmatch.fnmatchcase(parts[i], pats[j]):
                reach[i][j] = reach[i + 1][j + 1]
    return reach[0][0]


def excluded(path: str, excludes: Sequence[str]) -> str | None:
    """The exclusion that applies to this path, or None.

    TWO SHAPES, TOLD APART BY A SLASH, because the two things people
    mean by an exclusion are different. ".git" and "*.pyc" name a thing
    wherever it sits, so a pattern with no separator is matched against
    every SEGMENT of the path and excludes any directory or file so
    named at any depth. "docs/**" names a place, so a pattern with a
    separator is matched against the whole path. This is what a reader
    of DEFAULT_EXCLUDES would assume each entry means, which is the
    point of splitting on the character that is already visible.

    Returns the pattern rather than a bool so the caller can say WHICH
    exclusion applied. Nothing in the walk needs that today; the
    refusals a later phase writes will.
    """
    segments = path.split("/")
    for pattern in excludes:
        if "/" in pattern:
            if matches(path, pattern):
                return pattern
        elif any(fnmatch.fnmatchcase(segment, pattern) for segment in segments):
            return pattern
    return None


def enforce_patterns(patterns: Sequence[str]) -> None:
    """Refuse an include list that cannot mean what it says.

    THESE ARE CALLER STRINGS AND THEY SELECT FILES, so the checks here
    are about the caller learning what went wrong rather than about
    containment: matching cannot escape a root no matter what the
    pattern says, because every candidate path is built by the walk out
    of directory entries and never out of the pattern. An absolute
    pattern or one with ".." would simply match nothing, and a silent
    empty selection is a worse answer than a refusal that says patterns
    are repo-relative.
    """
    if not patterns:
        raise SnapshotError(
            "a snapshot needs at least one include pattern. '**/*.py' is "
            "every Python file at any depth; '*.md' is the top level only."
        )
    if len(patterns) > MAX_PATTERNS:
        raise SnapshotError(
            f"{len(patterns)} include patterns, over the {MAX_PATTERNS} "
            "limit. A selection that needs more than that is a selection "
            "of the whole tree, which '**/*' already says."
        )
    for pattern in patterns:
        if not pattern or pattern != pattern.strip():
            raise SnapshotError(
                f"include pattern {pattern!r} is empty or has surrounding "
                "whitespace, and neither can match a path."
            )
        if pattern.startswith("/") or pattern.endswith("/"):
            raise SnapshotError(
                f"include pattern {pattern!r} starts or ends with '/'. "
                "Patterns are repo-relative and select files, so write "
                "'src/**' rather than '/src/' for everything under src."
            )
        if "\\" in pattern:
            raise SnapshotError(
                f"include pattern {pattern!r} contains a backslash. Paths "
                "in a snapshot are separated by '/' on every platform, so "
                "a backslash here would match a filename containing one."
            )
        if ".." in pattern.split("/"):
            raise SnapshotError(
                f"include pattern {pattern!r} contains '..'. Patterns are "
                "matched against paths relative to the root and never "
                "leave it, so this one could only match nothing."
            )


def _contained(real: str, root: str) -> bool:
    """Whether a resolved path is the root or sits under it.

    String containment rather than a walk up the tree because both
    arguments are already fully resolved by the caller's realpath: no
    symlink and no ".." survives resolution, so a prefix comparison is
    the whole question. The separator is appended so that "/repo-old"
    does not read as being under "/repo".
    """
    return real == root or real.startswith(root.rstrip("/") + "/")


def walk(
    *,
    entries: Callable[[str], Iterable[tuple[str, bool, bool]]],
    resolve: Callable[[str], str],
    patterns: Sequence[str],
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[str]:
    """Every file under the root that the patterns select, in order.

    THE FILESYSTEM ARRIVES AS TWO CALLABLES so this module keeps the
    property extract has: no I/O, so every case is a value in and a
    value out and the adversarial ones are ordinary tests rather than
    fixtures on disk.

    entries(dir) yields (name, is_directory, is_symlink) for every child
    of one repo-relative directory, "" being the root. is_directory is
    answered WITHOUT following links, which costs nothing to honour
    because the link flag is checked first anyway. A child that is
    neither a regular file nor a directory, a socket or a device or a
    fifo, is the door's to refuse: deciding that is a stat, this module
    does no I/O, and a walk that handed one to the caller's read would
    hand it something that may never return.

    resolve(path) is realpath: the fully resolved absolute path, with
    resolve("") giving the root's own.

    SYMLINKS ARE NEVER FOLLOWED, and the rule has two halves that are
    not the same rule.

    One that resolves OUTSIDE the root refuses the whole snapshot,
    naming the link and its target. This is the containment law and it
    is why resolve exists: a root allowlist bounds what may be walked,
    and a link is exactly the construct that would let a walk of an
    allowed root read a file in a disallowed one. A directory link is
    checked identically, because a snapshot that descended through one
    would carry files from wherever it pointed.

    One that resolves INSIDE the root is skipped and does not refuse,
    because its bytes are already in the snapshot under the target's own
    path. Following it would put the same content in twice under two
    names, and a snapshot that carries a file twice is not a reading of
    a tree. There is one corner where this and "never drop silently"
    pull apart: a contained link whose target is itself excluded leaves
    the content out entirely. It resolves toward the exclusion, because
    an exclusion is the caller's own instruction and honouring it is not
    a drop.

    Exclusions are applied to directories as well as files, so an
    excluded tree is never traversed at all. That is most of what keeps
    MAX_WALKED_ENTRIES from firing on an ordinary repository with a
    node_modules in it.
    """
    enforce_patterns(patterns)
    root = resolve("")
    selected: list[str] = []
    matched = dict.fromkeys(patterns, 0)
    seen = 0
    pending = [""]
    while pending:
        current = pending.pop()
        for name, is_directory, is_symlink in sorted(entries(current)):
            seen += 1
            if seen > MAX_WALKED_ENTRIES:
                raise SnapshotError(
                    f"the walk passed {MAX_WALKED_ENTRIES} directory "
                    "entries without finishing. That is far larger than a "
                    "repository, so the root is almost certainly a parent "
                    "of one. Point it at the clone itself."
                )
            path = f"{current}/{name}" if current else name
            if excluded(path, excludes) is not None:
                continue
            if is_symlink:
                target = resolve(path)
                if not _contained(target, root):
                    raise SnapshotError(
                        f"{path} is a symbolic link to {target}, which is "
                        "outside the snapshot root. A snapshot reads one "
                        "tree, so a link that leaves it is refused rather "
                        "than followed or ignored. Exclude it, or snapshot "
                        "the tree it points into instead."
                    )
                continue
            if is_directory:
                pending.append(path)
                continue
            for pattern in patterns:
                if matches(path, pattern):
                    matched[pattern] += 1
                    selected.append(path)
                    break
    if not selected:
        empty = ", ".join(repr(p) for p in patterns if not matched[p])
        raise SnapshotError(
            f"no file under the root matched {empty}. Patterns are "
            "repo-relative and do not recurse unless they say so, so "
            "'*.py' is the top level and '**/*.py' is every depth."
        )
    return _ordered(selected)


def _ordered(paths: Iterable[str]) -> list[str]:
    """The snapshot's ordering law: repo-relative path, byte order.

    Keyed on the encoded bytes rather than on the string, which sorts
    identically for every path this can see and says the rule the way
    the rule is written. Sorting is what makes two walks of one tree
    produce one snapshot: entries() may hand back a directory in
    whatever order the filesystem felt like, and a snapshot whose digest
    depended on that would be a rendition that could not be reproduced.
    """
    return sorted(paths, key=lambda path: path.encode("utf-8"))


def enforce_text(path: str, content: bytes) -> None:
    """Refuse a file that is not text, naming it and saying how it is not.

    BINARY IN A SNAPSHOT IS A REFUSAL AND NEVER A SILENT DROP. The
    caller wrote a pattern, not a list, so the difference between "your
    pattern matched a PNG" and a snapshot quietly missing a file is the
    difference between a person fixing their selection and a person
    comparing models on a repository they think they sent.

    THE SIGNATURE CHECK RUNS FIRST although the NUL sniff would catch
    every format in it, because the two produce different messages and
    the specific one is better. IMAGE_SIGNATURES is extract's table,
    imported rather than restated: the three formats the upload door
    recognises are the three a source tree most often carries once the
    default exclusions have taken the build output away.

    A NUL byte is the general case and the oldest text test there is. It
    is not a proof of binary content and does not claim to be: it is the
    cheap check that separates source from object files, archives and
    images, and anything it lets through still has to decode.
    """
    for media, signature in IMAGE_SIGNATURES.items():
        if all(content[at : at + len(magic)] == magic for at, magic in signature):
            raise SnapshotError(
                f"{path} is a {media} image, and a snapshot is text or it "
                "is refused. Nothing is dropped quietly, so narrow the "
                "include patterns until they select source only."
            )
    if b"\x00" in content:
        # Bound outside the message because Python 3.11 forbids a
        # backslash inside an f-string expression.
        at = content.index(b"\x00")
        raise SnapshotError(
            f"{path} is binary: it contains a NUL byte at offset {at}. A "
            "snapshot is text or it is refused. Narrow the include "
            "patterns until they select source only."
        )


def _decoded(path: str, content: bytes) -> str:
    """One file's bytes as text, by ENCODING_RULE, or a refusal.

    THE BOM IS REMOVED AND NOTHING ELSE IS. Some Windows editors begin a
    UTF-8 file with three bytes that are an encoding marker rather than
    content, and leaving them in would put a zero-width character at the
    top of a file every model reads. Line endings are left exactly as
    they are: CRLF is a fact about the repository, and normalising it
    would make the snapshot disagree with the digest in its own header.

    A file that is not UTF-8 is refused by name and by offset, because
    the person's next action is to open that file at that byte.
    """
    body = content[len(UTF8_BOM) :] if content.startswith(UTF8_BOM) else content
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotError(
            f"{path} is not valid UTF-8: byte {exc.start} is "
            f"0x{body[exc.start]:02x}, which {exc.reason}. The snapshot "
            f"is decoded as {ENCODING_RULE} and does not guess at other "
            "encodings, because a guess would send every model text the "
            "file does not contain. Convert the file, or exclude it."
        ) from None


def digest_of(text: str) -> str:
    """The snapshot's own digest: sha256 over its UTF-8 encoding.

    Its own function so the door that stores the attachment and the
    composition that returns it cannot spell the encoding differently.
    The stored bytes ARE text.encode("utf-8"), so this digest identifies
    the attachment's content in the same way an upload's digest does.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compose(
    members: Sequence[tuple[str, bytes]],
    *,
    patterns: Sequence[str],
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    head: str | None,
    dirty: bool | None,
) -> dict[str, Any]:
    """The snapshot: one text, one digest, one manifest.

    Takes the files the caller read for the paths walk returned, and
    returns everything an attachment row needs: the composed text, its
    digest, the rendition triple that names this reading, and the
    manifest that says what the reading covered.

    IT SORTS AGAIN. walk already returned its paths in order, but the
    ordering law belongs to the composition rather than to the caller's
    reading loop, and a composed text whose byte order depended on a
    caller having preserved a list order would be a rendition with an
    invisible input.

    THE MANIFEST CARRIES NO ABSOLUTE PATH. Member paths are
    repo-relative, and the root the caller walked stays out of the
    record: it is the one field here that would put a person's home
    directory into an export, and nothing downstream needs it. The head
    commit and the dirty flag are read locally by the caller and passed
    in, because reading them is I/O and this module does none.

    THE NEWLINE BEFORE EACH MARKER IS THE JOIN'S, not the file's. Blocks
    are joined with a single newline so a header and a footer each sit
    on their own line, which means a file already ending in one is
    followed by a blank line and a file ending without one is not
    silently given one. Neither changes the file: the header's byte size
    and digest are what identify the content, and both are computed from
    the bytes rather than from the block.

    An empty file is composed as an empty block rather than refused,
    which is the one place this module and extract deliberately differ.
    extract refuses an empty extraction because a document that yielded
    no text means every model was asked about nothing; an empty
    __init__.py is a true fact about a repository and the models should
    see it.
    """
    if not members:
        # walk refuses an empty selection already, and this is the same
        # refusal one layer in for the same reason walk's duplicate check
        # is repeated here: a pure function does not assume its caller.
        # A snapshot of no files composes to an intro line saying so, and
        # every model would be asked about a repository none of them
        # received.
        raise SnapshotError(
            "a snapshot of no files is not a snapshot. Nothing was "
            "selected, so there is nothing for a model to read."
        )
    by_path = _ordered(path for path, _ in members)
    if len(set(by_path)) != len(by_path):
        repeated = sorted({p for p in by_path if by_path.count(p) > 1})
        raise SnapshotError(
            f"the snapshot was given {', '.join(repeated)} more than once. "
            "A path appears at most once in one reading of a tree."
        )
    contents = dict(members)
    total = len(by_path)
    blocks = [
        SNAPSHOT_INTRO.format(
            count=total, plural=_plural(total), encoding=ENCODING_RULE
        )
    ]
    files: list[dict[str, Any]] = []
    for index, path in enumerate(by_path, start=1):
        content = contents[path]
        # Size before shape: this comparison is one length and the text
        # checks scan the whole buffer, so the cheaper refusal goes
        # first. A file that is both oversized and binary is reported as
        # oversized, and either sentence sends the person to the same
        # pattern.
        if len(content) > MAX_MEMBER_BYTES:
            raise SnapshotError(
                f"{path} is {len(content)} bytes, over the "
                f"{MAX_MEMBER_BYTES} limit for one file in a snapshot. "
                "Nothing is truncated, so a file this size is refused "
                "whole. Exclude it, or narrow the patterns around it."
            )
        enforce_text(path, content)
        digest = hashlib.sha256(content).hexdigest()
        blocks.append(
            SNAPSHOT_HEADER.format(
                index=index,
                total=total,
                path=path,
                size=len(content),
                short=digest[:12],
            )
        )
        blocks.append(_decoded(path, content))
        blocks.append(SNAPSHOT_FOOTER.format(index=index, total=total))
        files.append({"path": path, "size": len(content), "digest": digest})
    text = "\n".join(blocks)
    if len(text) > MAX_COMPOSED_CHARS:
        largest = sorted(files, key=lambda member: -member["size"])[:3]
        named = ", ".join(f"{m['path']} at {m['size']} bytes" for m in largest)
        raise SnapshotError(
            f"the snapshot composes to {len(text)} characters, over the "
            f"{MAX_COMPOSED_CHARS} limit. Nothing is truncated, so narrow "
            f"the patterns instead. The largest files selected are {named}."
        )
    return {
        "text": text,
        "digest": digest_of(text),
        "extractor": SNAPSHOT_EXTRACTOR,
        "extractor_version": SNAPSHOT_VERSION,
        "kind": SNAPSHOT_KIND,
        "manifest": {
            "files": files,
            "patterns": list(patterns),
            "excludes": list(excludes),
            "head": head,
            "dirty": dirty,
            "encoding": ENCODING_RULE,
        },
    }
