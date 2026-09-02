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

NO I/O AND NO CLOCK. The walk is expressed over an injected Tree, a
handful of operations a door implements over file descriptors and a
test implements over a dict, and the composition over bytes the walk
already read. Every case below is therefore a value in and a value out.
The caller owns the filesystem, the allowlist that says which roots may
be walked at all, and the decision to store anything.

THE WALK READS THROUGH WHAT IT OPENED, and that sentence is the
fourteenth review's H1. The first shape of this module looked at a
tree by name and handed back paths, and the door then reopened each
path by name to read it. Between the look and the reopen a file could
be replaced by a symbolic link out of the root, a queued directory by a
link to another tree, a regular file by a named pipe that blocks
forever, or a small file by one that had grown past every bound; all
four were reproduced. Now every directory is descended through the
descriptor of its parent, every file is opened with no-follow flags and
verified after opening against what the listing said it was, and the
bytes are read through that descriptor and bounded at it. A tree that
changes under the walk is refused, not read.
"""

import fnmatch
import hashlib
import os.path
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

from bench.extract import (
    IMAGE_SIGNATURES,
    MAX_COMPOSED_CHARS,
    SNAPSHOT_EXTRACTOR,
    SNAPSHOT_KIND,
    SNAPSHOT_VERSION,
)

# Re-exported rather than defined, and the direction is the one thing
# worth reading twice: this module PRODUCES the rendition whose identity
# those two name, and extract HOLDS the vocabulary, because extract.
# kind_of has to be able to name the walker and cannot import the module
# that runs it. The rule for bumping SNAPSHOT_VERSION is written beside
# the constant there and it is about the code in HERE: change a header's
# wording, the ordering rule, the encoding rule or DEFAULT_EXCLUDES and
# the version moves with it.
__all__ = [
    "DEFAULT_EXCLUDES",
    "DIRECTORY",
    "ENCODING_RULE",
    "FILE",
    "MAX_DEPTH",
    "MAX_MEMBER_BYTES",
    "MAX_READ_BYTES",
    "MAX_PATTERNS",
    "MAX_WALKED_ENTRIES",
    "OTHER",
    "SNAPSHOT_EXTRACTOR",
    "SNAPSHOT_KIND",
    "SNAPSHOT_VERSION",
    "SYMLINK",
    "Entry",
    "Handle",
    "Opened",
    "SnapshotError",
    "Tree",
    "compose",
    "contained",
    "digest_of",
    "enforce_patterns",
    "enforce_text",
    "excluded",
    "matches",
    "walk",
]

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
#
# COUNTED AS THEY STREAM, one per entry the tree yields and before that
# entry is kept, which is what makes this a bound on time and memory
# rather than a label. The first shape of the walk sorted each
# directory's listing before counting it, so one directory of a million
# children was materialised whole and only then refused: the fourteenth
# review measured sixty thousand entries consumed before a twenty
# thousand ceiling fired. The tree's entries() is an iterator, the walk
# counts each one as it arrives, and a directory that is too wide is
# refused holding at most this many of its children.
MAX_WALKED_ENTRIES = 20_000

# How many directories deep a walk will go, and it is a count of OPEN
# DESCRIPTORS rather than a path length. The walk holds one open
# directory per level, because descending through the parent's
# descriptor rather than by name is what makes a swapped directory
# unable to redirect it, and a descriptor is a bounded resource: past a
# process's limit every open fails with "too many open files", which is
# a true sentence about the process and a useless one about the tree. A
# hundred and twenty-eight levels is past any source tree and under the
# smallest per-process limit worth planning for.
MAX_DEPTH = 128

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

# The largest total the walk may READ into memory before composing, and
# the one number here that is derived rather than chosen.
#
# UTF-8 ENCODES ONE CHARACTER IN AT MOST FOUR BYTES, so a selection
# whose bytes exceed four times the composed ceiling cannot possibly
# compose to fewer characters than that ceiling, whatever it encodes.
# The walk can therefore stop reading at this point and be certain it
# has not refused a snapshot the composition would have accepted, which
# is the property a cheaper bound would not have: refusing at
# MAX_COMPOSED_CHARS BYTES would refuse a perfectly composable snapshot
# of CJK source, and the person would have no way to tell that from a
# real over-ceiling refusal.
#
# WHAT IT BUYS is the memory. The per-file bound alone permits a
# selection of two thousand files at 200 KB each, and the walk reads
# every one of them before compose sees a byte. This bounds what is held
# to 800 KB plus one file, and the composition still owns the refusal.
#
# THE REFUSAL NAMES THE LARGEST FILES SELECTED, not the one whose bytes
# crossed the line. The line is crossed by whichever file happened to
# come last in path order, which can be a six-byte file after four
# large ones, and a person told to narrow their patterns around that
# one would be narrowing around the wrong file. This was the original
# commission and the first shape got it wrong.
MAX_READ_BYTES = MAX_COMPOSED_CHARS * 4

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


def contained(real: str, root: str) -> bool:
    """Whether a resolved path is the root or sits under it.

    A COMMON-PATH CHECK RATHER THAN A PREFIX CHECK. Both arguments are
    already fully resolved by the caller's realpath, so no symlink and
    no ".." survives, and the question is only whether the root is an
    ancestor. os.path.commonpath answers that in the platform's own
    terms: it splits on the platform's separator, so "/repo-old" is not
    under "/repo" because their common path is "/" and not "/repo",
    and on Windows it compares drives and folds case the way ntpath
    does. The first shape appended "/" and used startswith, which is
    the POSIX answer hardcoded, and would have called C:\\repo\\project
    a stranger to C:\\repo.

    THE LIMIT, STATED RATHER THAN HIDDEN. The bench runs on POSIX today:
    the door opens files with O_NOFOLLOW and O_DIRECTORY, the test suite
    makes fifos, and LocalOnlyGuard reasons about loopback. This
    function is the one piece of the containment story spelled
    portably, and it is the only piece that has been. A ValueError from
    commonpath, which is what different drives or a mixed absolute and
    relative pair raise, is "nothing in common" and therefore not
    contained.

    PUBLIC BECAUSE THE ALLOWLIST ASKS THE SAME QUESTION. The walk asks
    whether a symlink's target left the root; the door asks whether the
    root a request named sits under one of BENCH_REPO_ROOTS. One rule,
    one containment bug to not have twice.
    """
    try:
        common = os.path.commonpath([real, root])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(os.path.normpath(root))


# The kinds a directory entry can be, as the walk tells them apart. Four
# rather than a boolean pair because the fourth is a refusal: a socket,
# a device or a named pipe is none of the other three and must never be
# opened for reading, since reading one can block with nothing to time
# out.
FILE = "file"
DIRECTORY = "directory"
SYMLINK = "symlink"
OTHER = "other"

# (device, inode): what a filesystem calls one object, however many
# names it has. Two stats that agree on both are stats of the same
# thing, and that agreement is the whole of how the walk knows the file
# it opened is the file it listed.
Identity = tuple[int, int]


class Entry(NamedTuple):
    """One child of a directory, as the listing described it.

    The identity and size are recorded AT THE LISTING and checked again
    after the open, which is the pair of moments the review's races sit
    between. An entry is a claim; the descriptor is the fact.
    """

    name: str
    kind: str
    identity: Identity
    size: int


@dataclass(frozen=True)
class Handle:
    """An open directory: where it is, what it is, and the tree's token.

    token is whatever the tree needs to address the open directory
    again, a file descriptor for the door's tree and a dict key for a
    test's, and this module never looks inside it. path is
    repo-relative, "" for the root, and exists for messages and for
    building member paths; it is never used to open anything.
    """

    path: str
    identity: Identity
    token: Any


@dataclass(frozen=True)
class Opened:
    """A member as its OWN descriptor describes it, after opening.

    Distinct from Entry on purpose: an Entry is what the parent listing
    said, an Opened is what the object turned out to be once held. The
    walk compares the two and refuses on any disagreement.
    """

    path: str
    kind: str
    identity: Identity
    size: int
    token: Any


class Tree(Protocol):
    """The filesystem as the walk sees it: eight operations, no names.

    EVERY OPERATION AFTER open_root TAKES A HANDLE OR AN OPENED, never a
    path. That is the containment: a directory is descended through the
    handle of its parent, a member is opened through the handle of its
    directory, and bytes are read through the member's own descriptor.
    Nothing is ever reopened by name, so nothing a name can be made to
    point at between two moments can redirect the walk.

    The door implements this over os.open with dir_fd and O_NOFOLLOW; a
    test implements it over a dict, which is how a swapped file, a
    swapped directory and a fifo appearing after the look are ordinary
    deterministic tests rather than races somebody has to win.
    """

    def root_path(self) -> str:
        """The resolved absolute root, for containment decisions and
        messages only."""
        ...

    def open_root(self) -> Handle: ...

    def entries(self, handle: Handle) -> Iterator[Entry]:
        """The children of an open directory, one at a time. An iterator
        and not a list, so the walk can count and stop before a hostile
        directory is materialised."""
        ...

    def descend(self, handle: Handle, entry: Entry) -> Handle:
        """Open a child directory through its parent's handle, following
        no link. Raises SnapshotError if it cannot."""
        ...

    def open_member(self, handle: Handle, entry: Entry) -> Opened:
        """Open a child through its directory's handle, following no link
        and blocking on nothing, and describe what was opened."""
        ...

    def read_member(self, opened: Opened, limit: int) -> bytes:
        """At most limit + 1 bytes through the descriptor, so the walk can
        see a file that grew without holding all of it."""
        ...

    def close_handle(self, held: Handle | Opened) -> None: ...

    def link_target(self, handle: Handle, entry: Entry) -> str:
        """Where a symbolic link points, resolved, for the containment
        decision about it. Never used to open anything."""
        ...


def _describe(kind: str) -> str:
    """A kind as a refusal names it."""
    return {
        FILE: "a regular file",
        DIRECTORY: "a directory",
        SYMLINK: "a symbolic link",
    }.get(kind, "a socket, a device or a named pipe")


def _changed(path: str, was: str, now: str) -> SnapshotError:
    """The refusal for every race, worded once."""
    return SnapshotError(
        f"{path} changed while the snapshot was being taken: the walk "
        f"listed {was} and opened {now}. A tree being edited or swapped "
        "under a walk is refused rather than read, because the record "
        "could not say what it read. Take the snapshot again on a still "
        "tree."
    )


def walk(
    *,
    tree: Tree,
    patterns: Sequence[str],
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[tuple[str, bytes]]:
    """Every file under the root the patterns select, READ, in order.

    THE WALK RETURNS BYTES AND NOT PATHS, and that is the whole of the
    fourteenth review's H1. A walk that returned paths for a door to
    reopen by name had a window between its look and the door's open,
    and a name is exactly the thing that can be made to point somewhere
    else inside a window. Reading here, through the descriptor the walk
    itself opened, closes it: the thing read is the thing listed, or
    the snapshot is refused.

    WHAT IS VERIFIED, AND WHEN. Every directory entry arrives with a
    kind, an identity and a size from the listing. A directory is
    descended through its parent's handle and the child's identity
    must match the listing's, or a directory swapped for another before
    descent is refused. A member is opened through its directory's
    handle with no link followed and nothing blocked on, and after the
    open its own descriptor must say regular file, the same identity,
    and a size within the bound, or a file swapped for a link, a fifo
    or a device is refused before a byte is read. The read itself is
    bounded at the descriptor, so a file that grew after its stat is
    refused rather than held. The reviewer's two swaps and the fifo are
    tombstones in the tests, driven through a dict-backed tree, and the
    same three are driven through the door's descriptor-backed tree
    with the swap made on disk between the look and the open.

    SYMLINKS ARE NEVER FOLLOWED, and the rule has two halves that are
    not the same rule. One that resolves OUTSIDE the root refuses the
    whole snapshot, naming the link and its target: this is the
    containment law, and a link is exactly the construct that would let
    a walk of an allowed root read a file in a disallowed one. One that
    resolves INSIDE the root is skipped and does not refuse, because its
    bytes are already in the snapshot under the target's own path, and
    following it would put one file in twice under two names. The one
    corner where that and "never drop silently" pull apart is a
    contained link whose target is itself excluded; it resolves toward
    the exclusion, because an exclusion is the caller's own instruction.
    The target is resolved for the DECISION and the message only; it is
    never opened.

    ONE OPEN DIRECTORY PER LEVEL AND NO MORE. The traversal is a stack
    of (handle, remaining entries), so a directory is closed the moment
    its last child has been seen, and the number of descriptors held is
    the depth and not the width. MAX_DEPTH bounds that.

    ENTRIES ARE COUNTED AS THEY STREAM, before they are kept, so a
    directory wider than MAX_WALKED_ENTRIES is refused holding at most
    that many of its children and not all of them; see the constant.

    Exclusions are applied to directories as well as files, so an
    excluded tree is never opened at all, which is most of what keeps
    the ceiling from firing on an ordinary repository with a
    node_modules in it. A name that is not valid UTF-8 is refused by
    name and directory, because a path the snapshot could not spell in
    its own header is a path it could not describe.
    """
    enforce_patterns(patterns)
    root_real = tree.root_path()
    matched = dict.fromkeys(patterns, 0)
    selected: list[tuple[str, bytes]] = []
    largest: list[tuple[int, str]] = []
    total = 0
    seen = 0

    def listing(handle: Handle) -> Iterator[Entry]:
        # Counted per streamed entry and refused before the entry is
        # kept; see MAX_WALKED_ENTRIES for the measurement that put the
        # count here rather than after the sort.
        nonlocal seen
        gathered: list[Entry] = []
        for entry in tree.entries(handle):
            seen += 1
            if seen > MAX_WALKED_ENTRIES:
                raise SnapshotError(
                    f"the walk passed {MAX_WALKED_ENTRIES} directory "
                    "entries without finishing. That is far larger than a "
                    "repository, so the root is almost certainly a parent "
                    "of one. Point it at the clone itself."
                )
            try:
                entry.name.encode("utf-8")
            except UnicodeEncodeError:
                raise SnapshotError(
                    f"{handle.path or 'the snapshot root'} holds an entry "
                    f"whose name is not valid UTF-8 ({entry.name!r}). A "
                    "snapshot names every file in a header line, and a "
                    "name it cannot spell is a file it cannot describe. "
                    "Rename it, or exclude it by pattern."
                ) from None
            gathered.append(entry)
        gathered.sort(key=lambda entry: entry.name.encode("utf-8"))
        return iter(gathered)

    root = tree.open_root()
    stack: list[tuple[Handle, Iterator[Entry]]] = [(root, iter(()))]
    try:
        stack[0] = (root, listing(root))
        while stack:
            handle, remaining = stack[-1]
            entry = next(remaining, None)
            if entry is None:
                stack.pop()
                tree.close_handle(handle)
                continue
            path = f"{handle.path}/{entry.name}" if handle.path else entry.name
            if excluded(path, excludes) is not None:
                continue
            if entry.kind == SYMLINK:
                target = tree.link_target(handle, entry)
                if not contained(target, root_real):
                    raise SnapshotError(
                        f"{path} is a symbolic link to {target}, which is "
                        "outside the snapshot root. A snapshot reads one "
                        "tree, so a link that leaves it is refused rather "
                        "than followed or ignored. Exclude it, or snapshot "
                        "the tree it points into instead."
                    )
                continue
            if entry.kind == DIRECTORY:
                if len(stack) >= MAX_DEPTH:
                    raise SnapshotError(
                        f"{path} is {MAX_DEPTH} directories deep, which is "
                        "past what a snapshot will descend. The walk holds "
                        "one open directory per level, and a tree this deep "
                        "is not a source tree. Narrow the root or the "
                        "patterns."
                    )
                child = tree.descend(handle, entry)
                if child.identity != entry.identity:
                    tree.close_handle(child)
                    raise _changed(path, "a directory", "a different one")
                try:
                    inner = listing(child)
                except BaseException:
                    tree.close_handle(child)
                    raise
                stack.append((child, inner))
                continue
            if entry.kind != FILE:
                raise SnapshotError(
                    f"{path} is not a regular file, so the bench will not "
                    "open it. A socket, a device or a named pipe under a "
                    "source tree is refused rather than read, because "
                    "reading one can block with nothing to time out."
                )
            selector = next((p for p in patterns if matches(path, p)), None)
            if selector is None:
                continue
            matched[selector] += 1
            # Before the open, because a size is one integer and an open
            # is a descriptor; the same bound is checked again from the
            # descriptor's own stat below, which is the one that counts.
            if entry.size > MAX_MEMBER_BYTES:
                raise _too_large(path, entry.size)
            opened = tree.open_member(handle, entry)
            try:
                if opened.kind != FILE:
                    raise _changed(path, "a regular file", _describe(opened.kind))
                if opened.identity != entry.identity:
                    raise _changed(path, "a regular file", "a different file")
                if opened.size > MAX_MEMBER_BYTES:
                    raise _too_large(path, opened.size)
                data = tree.read_member(opened, MAX_MEMBER_BYTES)
            finally:
                tree.close_handle(opened)
            if len(data) > MAX_MEMBER_BYTES:
                raise SnapshotError(
                    f"{path} grew past {MAX_MEMBER_BYTES} bytes while it was "
                    "being read. The read is bounded at the descriptor, so "
                    "nothing past the bound was held; a file being written "
                    "under a walk is refused rather than read part-way."
                )
            total += len(data)
            # Largest first, ties by path, so the three named are the
            # same three however the selection happened to be ordered.
            largest = sorted(
                largest + [(len(data), path)], key=lambda item: (-item[0], item[1])
            )[:3]
            if total > MAX_READ_BYTES:
                named = ", ".join(f"{p} at {n} bytes" for n, p in largest)
                raise SnapshotError(
                    f"the selection passed {MAX_READ_BYTES} bytes and the "
                    "bench stopped reading. Even at four bytes per character "
                    f"that cannot compose under the {MAX_COMPOSED_CHARS} "
                    "character ceiling, so narrow the patterns. The largest "
                    f"files selected so far are {named}."
                )
            selected.append((path, data))
    finally:
        # Every directory still open is closed, deepest first, whether
        # the walk finished or refused part-way. A refusal that leaked a
        # descriptor per attempt would turn a person retrying into a
        # process out of descriptors.
        for handle, _ in reversed(stack):
            tree.close_handle(handle)
    if not selected:
        empty = ", ".join(repr(p) for p in patterns if not matched[p])
        raise SnapshotError(
            f"no file under the root matched {empty}. Patterns are "
            "repo-relative and do not recurse unless they say so, so "
            "'*.py' is the top level and '**/*.py' is every depth."
        )
    selected.sort(key=lambda member: member[0].encode("utf-8"))
    return selected


def _too_large(path: str, size: int) -> SnapshotError:
    return SnapshotError(
        f"{path} is {size} bytes, over the {MAX_MEMBER_BYTES} limit for "
        "one file in a snapshot, and is refused before it is read. Nothing "
        "is truncated, so narrow the patterns around it."
    )


def _ordered(paths: Iterable[str]) -> list[str]:
    """The snapshot's ordering law: repo-relative path, byte order.

    Keyed on the encoded bytes rather than on the string, which sorts
    identically for every path this can see and says the rule the way
    the rule is written. Sorting is what makes two walks of one tree
    produce one snapshot: entries() may hand back a directory in
    whatever order the filesystem felt like, and a snapshot whose digest
    depended on that would be a rendition that could not be reproduced.
    walk sorts its members by the same key; this is compose's own
    spelling of the law, applied again because the ordering belongs to
    the composition and not to whoever handed it the list.
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

    WHAT IT DOES NOT CATCH, stated so nobody reads "binary is refused"
    as a guarantee. A file whose bytes contain no NUL and happen to
    decode as UTF-8 passes both checks and is composed as text: a
    base64 blob, a hex dump, a minified bundle, a data file of ASCII
    digits. Those are not what the check is for. It exists so a
    pattern that swept up a .o or a PNG refuses by name instead of
    composing a field of mojibake, and the strict decode is what stops
    the mojibake case; anything that is genuinely text by every test a
    byte can be given is text as far as this module can know, and the
    header's size and digest still identify exactly what was sent.
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
