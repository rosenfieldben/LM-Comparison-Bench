"""walk() as it stood at d1d792d, the oracle the refactored walk is held to.

EXTRACTED FROM GIT, NEVER TRANSCRIBED: the function below is the text of
`git show d1d792d:bench/snapshot.py` from "def walk(" to the line before
"def _too_large(", byte for byte, and tests/test_listing.py asserts
that it still is. Phase O moved walk's traversal into Survey so the
member listing and the composer iterate one traversal; this copy is how
a test tells whether the move changed what walk returns or refuses,
which the tests of the moved lines alone could not (six mutants of them
passed the whole suite at the design review).

It imports the helpers it calls from bench.snapshot as they are today.
Those helpers are not what moved, and a change to one of them is held
by its own tests.
"""

from collections.abc import Iterator, Sequence

from bench.snapshot import (
    DEFAULT_EXCLUDES,
    DIRECTORY,
    FILE,
    MAX_COMPOSED_CHARS,
    MAX_DEPTH,
    MAX_MEMBER_BYTES,
    MAX_READ_BYTES,
    MAX_WALKED_ENTRIES,
    SYMLINK,
    Entry,
    Handle,
    SnapshotError,
    Tree,
    _changed,
    _describe,
    _too_large,
    contained,
    enforce_patterns,
    excluded,
    matches,
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
