"""Repository snapshots: a tree in, one text out, or a refusal.

Pure functions with the filesystem injected as two callables, so every
case here is a value in and a value out and the adversarial ones are
ordinary tests rather than fixtures that have to exist on disk. A
symlink escaping a root is a two-line dict here and a permissions
problem in a temporary directory, which is the whole reason walk takes
entries and resolve rather than reaching for os.

What is being asserted is not "a walk happened" but WHAT THE MODELS WILL
READ, because the composed snapshot is the prompt and a comparison over
a repository is a comparison over this module's reading of it.
"""

import ast
import hashlib
import re
from pathlib import Path

import pytest

from bench.extract import MAX_COMPOSED_CHARS, SNAPSHOT_KIND
from bench.snapshot import (
    DEFAULT_EXCLUDES,
    ENCODING_RULE,
    MAX_MEMBER_BYTES,
    MAX_WALKED_ENTRIES,
    SNAPSHOT_EXTRACTOR,
    SNAPSHOT_VERSION,
    SnapshotError,
    compose,
    digest_of,
    enforce_patterns,
    excluded,
    matches,
    walk,
)

ROOT = "/repo"


def tree(paths, *, links=None, root=ROOT, order=None):
    """entries and resolve over a tree described as a list of paths.

    paths are repo-relative files; links maps a repo-relative path to
    the absolute path it resolves to, and a link whose repo-relative
    path is also named in dirs is reported as a directory. order, when
    given, reorders each directory's entries so a test can prove the
    walk does not inherit the filesystem's listing order.
    """
    links = links or {}
    dirs = {name for name in links if links[name].endswith("/")}
    children: dict[str, list[tuple[str, bool, bool]]] = {}

    def add(path, is_directory, is_symlink):
        parent, _, name = path.rpartition("/")
        entry = (name, is_directory, is_symlink)
        if entry not in children.setdefault(parent, []):
            children[parent].append(entry)

    def parents(path):
        parts = path.split("/")
        for depth in range(len(parts) - 1):
            add("/".join(parts[: depth + 1]), True, False)

    for path in paths:
        parents(path)
        add(path, False, False)
    for path in links:
        parents(path)
        add(path, True, True) if path in dirs else add(path, False, True)

    def entries(where):
        listing = list(children.get(where, []))
        return order(listing) if order else listing

    def resolve(where):
        if where == "":
            return root
        if where in links:
            return links[where].rstrip("/")
        return f"{root}/{where}"

    return entries, resolve


def selection(paths, patterns, **kwargs):
    """walk over a described tree, for the cases that only need paths."""
    entries, resolve = tree(paths, **kwargs)
    return walk(entries=entries, resolve=resolve, patterns=patterns)


def snapshot(members, patterns=("**/*",), head="abc1234", dirty=False):
    """compose over (path, bytes) pairs, with the record fields fixed."""
    return compose(members, patterns=patterns, head=head, dirty=dirty)


# ----- selection -----------------------------------------------------


def test_a_pattern_without_a_slash_does_not_recurse():
    """WINDOW: matches, one path against one include pattern.

    THE RULE IS EXPLICIT RECURSION and this is the assertion that keeps
    it explicit. fnmatch's "*" crosses a directory separator, so a
    module that reached for fnmatch on the whole path would make "*.py"
    mean every Python file at every depth, and a person who wrote
    "*.py" meaning the top level would ship their whole tree to a
    provider without having asked for it.
    """
    assert matches("main.py", "*.py")
    assert not matches("src/main.py", "*.py")
    assert matches("src/main.py", "src/*.py")
    assert not matches("src/deep/main.py", "src/*.py")


def test_a_double_star_matches_zero_or_more_segments():
    """WINDOW: matches, with "**" at the front, middle and end."""
    assert matches("main.py", "**/*.py")
    assert matches("a/b/c/main.py", "**/*.py")
    assert matches("src/anything", "src/**")
    assert matches("src/a/b/anything", "src/**")
    assert matches("src/x.py", "src/**/*.py")
    assert matches("src/a/b/x.py", "src/**/*.py")
    assert not matches("other/x.py", "src/**")


def test_a_pathological_pattern_matches_without_exploding():
    """WINDOW: matches, on a pattern built to blow up a naive recursion.

    PATTERNS ARRIVE FROM THE CALLER, so the obvious recursive matcher
    (try "**" against every split point, recurse on each) is a denial of
    service with a twenty-character pattern. The reachability table
    makes this O(segments times pattern segments); the assertion is that
    it returns at all, which under the recursive form it would not do
    this side of the heat death.
    """
    pattern = "/".join(["**"] * 12) + "/needle.py"
    assert matches("a/b/c/d/e/f/g/h/i/j/k/l/needle.py", pattern)
    assert not matches("a/b/c/d/e/f/g/h/i/j/k/l/haystack.py", pattern)


def test_an_exclusion_without_a_slash_matches_any_segment():
    """WINDOW: excluded, the two exclusion shapes.

    A reader of DEFAULT_EXCLUDES assumes ".git" means "any .git
    anywhere" and "docs/**" means "that place", and this is the
    assertion that the code agrees with the assumption. The returned
    value is the pattern rather than True so a refusal can eventually
    say which exclusion applied.
    """
    assert excluded("a/b/.git/config", (".git",)) == ".git"
    assert excluded("node_modules/x/y.js", ("node_modules",)) == "node_modules"
    assert excluded("a/b/c.pyc", ("*.pyc",)) == "*.pyc"
    assert excluded("docs/a/b.md", ("docs/**",)) == "docs/**"
    assert excluded("a/docs/b.md", ("docs/**",)) is None
    assert excluded("src/main.py", DEFAULT_EXCLUDES) is None


def test_the_secret_defaults_are_in_force():
    """WINDOW: excluded, against DEFAULT_EXCLUDES.

    THE GROUP THE COMMISSION DID NOT NAME. Every other reader in this
    codebase is handed one file a person chose; this one is handed a
    pattern and gathers what matches, and what it gathers is composed
    into a prompt and sent to a provider. A .env swept up by "**/*" is
    an API key posted to a third party, and no refusal downstream can
    unsend it. The entries cost nothing when they match nothing.
    """
    for path in (
        ".env",
        "a/b/.env",
        ".env.local",
        "certs/server.pem",
        "deploy/id_rsa",
        "keys/private.key",
        ".netrc",
        ".npmrc",
    ):
        assert excluded(path, DEFAULT_EXCLUDES) is not None, path


def test_an_excluded_directory_is_never_traversed():
    """WINDOW: walk, over a tree whose bulk sits under an exclusion.

    Exclusion applies to directories as well as files, which is most of
    what keeps MAX_WALKED_ENTRIES from firing on an ordinary repository:
    a node_modules with fifty thousand files in it is one entry skipped
    rather than fifty thousand entries counted.
    """
    paths = [
        "src/main.py",
        "node_modules/pkg/index.js",
        ".git/config",
        # Walking this repository with "**/*" refused on a binary blob
        # under .hypothesis before that entry existed, which is how it
        # got there. A default exclusion list assembled from memory
        # misses the caches the tools you actually run write.
        ".hypothesis/examples/04e6b3400353b141/e270cb250512f120",
    ]
    assert selection(paths, ["**/*"]) == ["src/main.py"]


def test_files_are_ordered_by_the_bytes_of_their_path():
    """WINDOW: walk's return, over paths that sort two different ways.

    "a-b.py" and "a/b.py" are the case: "-" is 0x2D and "/" is 0x2F, so
    byte order puts "a-b.py" first, while any sort that compared path
    SEGMENTS would compare "a" against "a-b.py" and put "a/b.py" first.
    The ordering law is what makes two walks of one tree produce one
    digest, so it has to be the law the docstring states rather than
    whichever one the obvious code produces.
    """
    assert selection(["a/b.py", "a-b.py"], ["**/*.py"]) == ["a-b.py", "a/b.py"]
    # And walk sorts its own return rather than leaning on compose to do
    # it. The traversal is depth-last, so a top-level file is found
    # before a subdirectory is descended into and the raw order here
    # would be ["b.py", "a/z.py"]. compose sorts again and would hide
    # this, so the assertion is on walk's return.
    assert selection(["b.py", "a/z.py"], ["**/*.py"]) == ["a/z.py", "b.py"]


def test_two_walks_of_one_tree_compose_the_same_snapshot():
    """WINDOW: walk and compose, over one tree listed in two orders.

    DETERMINISM IS THE RENDITION'S PRECONDITION. The digest keys the
    attachment row, so a snapshot whose bytes depended on the order
    readdir happened to return would be a rendition that could not be
    reproduced and a record of a prompt nobody could rebuild. entries()
    is deliberately hostile here: the second walk sees every directory
    listed backwards.
    """
    paths = ["src/a.py", "src/b.py", "README.md", "src/deep/c.py"]
    forward = selection(paths, ["**/*"])
    backward = selection(paths, ["**/*"], order=lambda listing: listing[::-1])
    assert forward == backward

    members = [(path, f"body of {path}\n".encode()) for path in forward]
    first = snapshot(members)
    second = snapshot(members[::-1])
    assert first["text"] == second["text"]
    assert first["digest"] == second["digest"]


def test_a_selection_that_matched_nothing_names_the_patterns():
    """WINDOW: walk, patterns that select no file.

    An empty snapshot would compose to an intro line and nothing else,
    and every model would be asked about a repository none of them
    received: the same case extract refuses for a scanned PDF. The
    refusal names the patterns that matched nothing, because the
    ordinary cause is a person who wrote "*.py" expecting recursion.
    """
    with pytest.raises(SnapshotError) as caught:
        selection(["src/main.py"], ["*.py"])
    assert "'*.py'" in str(caught.value)
    assert "**/*.py" in str(caught.value)


def test_the_walk_gives_up_before_traversing_a_home_directory():
    """WINDOW: walk, over a tree past MAX_WALKED_ENTRIES.

    The walk is caller-directed and runs inside a request, which is the
    sentence that bounded docx nesting and zip inflation. A root pointed
    at a parent of the clone is the ordinary mistake, and the refusal
    says so rather than reporting a limit the person has no way to act
    on.
    """
    paths = [f"f{index:06d}.txt" for index in range(MAX_WALKED_ENTRIES + 1)]
    with pytest.raises(SnapshotError) as caught:
        selection(paths, ["**/*"])
    assert str(MAX_WALKED_ENTRIES) in str(caught.value)
    assert "Point it at the clone itself" in str(caught.value)


# ----- containment ---------------------------------------------------


def test_a_symlink_leaving_the_root_refuses_the_snapshot():
    """WINDOW: walk, a file symlink resolving outside the root.

    THE CONTAINMENT LAW. BENCH_REPO_ROOTS bounds which roots may be
    walked at all, and a link is exactly the construct that would let a
    walk of an allowed root read a file in a disallowed one. Refused
    rather than skipped, because a skip would mean a person's pattern
    selected a file and the snapshot quietly did not contain it.
    """
    with pytest.raises(SnapshotError) as caught:
        selection(
            ["src/main.py"],
            ["**/*"],
            links={"src/secrets.py": "/elsewhere/secrets.py"},
        )
    assert "src/secrets.py" in str(caught.value)
    assert "/elsewhere/secrets.py" in str(caught.value)
    assert "outside the snapshot root" in str(caught.value)


def test_a_directory_symlink_leaving_the_root_refuses_the_snapshot():
    """WINDOW: walk, a directory symlink resolving outside the root.

    Checked identically to a file link and before the descent, because
    a snapshot that walked through one would carry files from wherever
    it pointed under paths that claim to be inside the clone.
    """
    with pytest.raises(SnapshotError) as caught:
        selection(
            ["src/main.py"],
            ["**/*"],
            links={"src/shared": "/elsewhere/lib/"},
        )
    assert "src/shared" in str(caught.value)
    assert "/elsewhere/lib" in str(caught.value)


def test_a_sibling_root_sharing_a_prefix_is_still_outside():
    """WINDOW: _contained, through walk, on the prefix trap.

    "/repo-old/x.py" starts with "/repo" and is not under it. The
    separator is appended before the comparison for exactly this case,
    which is the oldest bug in path containment and the one a string
    prefix check gets wrong by default.
    """
    with pytest.raises(SnapshotError) as caught:
        selection(
            ["src/main.py"],
            ["**/*"],
            links={"src/old.py": "/repo-old/x.py"},
        )
    assert "/repo-old/x.py" in str(caught.value)


def test_a_symlink_inside_the_root_is_skipped_rather_than_duplicated():
    """WINDOW: walk, a file symlink resolving inside the root.

    Not refused, because its bytes are already in the snapshot under the
    target's own path; following it would put one file in twice under
    two names, and a snapshot that carries a file twice is not a reading
    of a tree. The target is present exactly once.
    """
    got = selection(
        ["src/real.py"],
        ["**/*"],
        links={"src/alias.py": "/repo/src/real.py"},
    )
    assert got == ["src/real.py"]


def test_an_excluded_symlink_does_not_refuse():
    """WINDOW: walk, an escaping symlink under an exclusion.

    Exclusion is checked BEFORE containment, and the order is
    load-bearing rather than incidental: a .venv is a symlink farm
    pointing at an interpreter somewhere else on the machine, and a
    walk that checked containment first would refuse every snapshot of
    every repository that has one.
    """
    got = selection(
        ["src/main.py"],
        ["**/*"],
        links={".venv/bin/python": "/usr/bin/python3"},
    )
    assert got == ["src/main.py"]


# ----- what may be composed ------------------------------------------


def test_a_binary_file_refuses_the_snapshot_by_name():
    """WINDOW: compose, a selected file containing a NUL byte.

    NEVER A SILENT DROP. The caller wrote a pattern, not a list, so the
    difference between "your pattern matched an object file" and a
    snapshot quietly missing it is the difference between a person
    fixing their selection and a person comparing models on a
    repository they think they sent.
    """
    with pytest.raises(SnapshotError) as caught:
        snapshot([("src/main.py", b"ok\n"), ("build/a.o", b"ELF\x00\x01")])
    assert "build/a.o" in str(caught.value)
    assert "NUL byte at offset 3" in str(caught.value)


def test_an_image_is_named_as_an_image_rather_than_as_a_nul_byte():
    """WINDOW: compose, a PNG in the selection.

    The signature table is extract's, imported rather than restated:
    the three formats the upload door recognises are the three a walk is
    most likely to sweep up. The NUL sniff would catch a PNG anyway; the
    signature check runs first because "is a image/png image" is a
    better sentence for the person than "contains a NUL byte at offset
    1".
    """
    png = b"\x89PNG\r\n\x1a\n" + b"rest"
    with pytest.raises(SnapshotError) as caught:
        snapshot([("docs/logo.png", png)])
    assert "docs/logo.png" in str(caught.value)
    assert "image/png" in str(caught.value)


def test_a_file_over_the_member_cap_is_refused_and_named():
    """WINDOW: compose, one member past MAX_MEMBER_BYTES.

    NO TRUNCATION, EVER. A snapshot that shortened a file would send
    every model a repository that does not exist, and the header's byte
    size and digest would be a claim about content the block does not
    contain. The refusal names the one file, which is the whole reason
    this check exists separately from the total: the total would refuse
    the same snapshot and name the selection.
    """
    with pytest.raises(SnapshotError) as caught:
        snapshot([("src/huge.py", b"x" * (MAX_MEMBER_BYTES + 1))])
    assert "src/huge.py" in str(caught.value)
    assert "Nothing is truncated" in str(caught.value)
    # The clause that only the per-file check can produce. The total
    # ceiling would refuse this same snapshot and would name this same
    # file among the largest, so without asserting the sentence that
    # says "for one file" the check below could be deleted and this test
    # would still pass.
    assert f"over the {MAX_MEMBER_BYTES} limit for one file" in str(caught.value)


def test_a_selection_over_the_composed_cap_names_the_largest_files():
    """WINDOW: compose, members each under the cap and over it together.

    The composed ceiling is extract's and is about what may be SENT, so
    a snapshot is bounded by the same number a set of attachments is.
    The refusal names the largest files because "narrow the patterns" is
    only actionable if the person knows which file to narrow around.
    """
    members = [
        ("src/small.py", b"x" * 1_000),
        ("src/big.py", b"y" * 120_000),
        ("src/medium.py", b"z" * 100_000),
    ]
    with pytest.raises(SnapshotError) as caught:
        snapshot(members)
    message = str(caught.value)
    assert str(MAX_COMPOSED_CHARS) in message
    assert "src/big.py at 120000 bytes" in message
    assert "src/medium.py at 100000 bytes" in message
    assert message.index("src/big.py") < message.index("src/medium.py")


def test_a_multibyte_file_over_the_member_cap_is_still_refused():
    """WINDOW: compose, a member over the cap in bytes and under it in
    characters.

    THE TWO CEILINGS COUNT DIFFERENT UNITS and this is the case that
    tells them apart. 70,000 CJK characters are 210,000 bytes and 70,000
    characters, so the composed total would pass and the per-file bound
    refuses. It is not a message improvement on a check the total
    already makes: delete the per-file check and this snapshot composes.
    """
    body = ("字" * 70_000).encode("utf-8")
    assert len(body) > MAX_MEMBER_BYTES
    assert len(body.decode("utf-8")) < MAX_COMPOSED_CHARS
    with pytest.raises(SnapshotError) as caught:
        snapshot([("src/wide.py", body)])
    assert f"over the {MAX_MEMBER_BYTES} limit for one file" in str(caught.value)


def test_a_snapshot_of_no_files_is_refused():
    """WINDOW: compose, an empty members list.

    walk refuses an empty selection one layer out, and this is the same
    refusal repeated for the same reason the duplicate check is: a pure
    function does not assume its caller. Without it a door that walked,
    read nothing and composed anyway would produce an attachment whose
    whole content is an intro line saying there are no files, and every
    model would be asked about a repository none of them received.
    """
    with pytest.raises(SnapshotError) as caught:
        snapshot([])
    assert "not a snapshot" in str(caught.value)


def test_one_path_given_twice_is_refused():
    """WINDOW: compose, a members list with a repeated path.

    A path appears at most once in one reading of a tree. walk cannot
    produce a repeat; compose is a pure function and does not assume its
    caller, and a silent last-one-wins here would compose a snapshot
    whose manifest and whose text disagreed about what was in it.
    """
    with pytest.raises(SnapshotError) as caught:
        snapshot([("src/a.py", b"one"), ("src/a.py", b"two")])
    assert "src/a.py" in str(caught.value)


# ----- the encoding rule ---------------------------------------------


def test_a_leading_byte_order_mark_is_removed_and_nothing_else_is():
    """WINDOW: compose, a file written by a Windows tool.

    The BOM is an encoding marker rather than content, and leaving it in
    would put a zero-width character at the top of a file every model
    reads. CRLF is a fact about the repository and stays: normalising it
    would make the composed text disagree with the byte size and digest
    in its own header.
    """
    built = snapshot([("src/win.py", b"\xef\xbb\xbfa = 1\r\nb = 2\r\n")])
    assert "a = 1\r\nb = 2\r\n" in built["text"]
    assert "﻿" not in built["text"]
    # The header's size is the file's bytes, BOM included, because that
    # is what the digest covers and what is on disk. Three bytes of mark
    # plus two seven-byte CRLF lines.
    assert built["manifest"]["files"][0]["size"] == 17


def test_multibyte_text_survives_verbatim():
    """WINDOW: compose, a file of non-ASCII UTF-8."""
    built = snapshot([("src/i18n.py", "GREETING = 'ολα καλά'\n".encode())])
    assert "ολα καλά" in built["text"]


def test_a_file_that_is_not_utf8_is_refused_by_name_and_offset():
    """WINDOW: compose, latin-1 bytes in the selection.

    STRICT, NOT REPLACING. errors="replace" would let a file the bench
    cannot read reach a model as a field of U+FFFD and the model would
    answer about it, which is the silent-truncation family: a prompt
    nobody can reconstruct and nothing in the response saying so. The
    offset is in the message because the person's next action is to
    open that file at that byte.
    """
    with pytest.raises(SnapshotError) as caught:
        snapshot([("src/legacy.py", b"caf\xe9 = 1\n")])
    message = str(caught.value)
    assert "src/legacy.py" in message
    assert "byte 3" in message
    assert "0xe9" in message
    assert ENCODING_RULE in message


def test_an_empty_file_is_composed_rather_than_refused():
    """WINDOW: compose, a zero-byte member.

    THE ONE PLACE THIS MODULE AND extract DELIBERATELY DIFFER. extract
    refuses an empty extraction because a document that yielded no text
    means every model was asked about nothing; an empty __init__.py is a
    true fact about a repository and the models should see it.
    """
    built = snapshot([("pkg/__init__.py", b""), ("pkg/main.py", b"x = 1\n")])
    assert "pkg/__init__.py, 0 bytes" in built["text"]
    assert built["manifest"]["files"][0]["size"] == 0


# ----- the composed snapshot and its record --------------------------


def test_every_file_is_delimited_and_the_header_carries_its_identity():
    """WINDOW: compose's text, for a two-file snapshot.

    The header makes "nothing was truncated" CHECKABLE rather than
    asserted: a reader with the header and the block can weigh one
    against the other. The footer is the prompt-injection boundary, the
    same job it does for an attachment, and more needed here: a file in
    a repository may contain a line that looks like the next file's
    header, and without a closing marker it could annex its successor.
    """
    body = b"print('hi')\n"
    built = snapshot([("src/a.py", body), ("src/b.py", b"x = 1\n")])
    text = built["text"]
    assert "A snapshot of 2 files from a source repository" in text
    assert ENCODING_RULE in text
    assert "----- file 1 of 2: src/a.py, 12 bytes, sha256 " in text
    assert "----- end file 1 of 2 -----" in text
    assert "----- file 2 of 2: src/b.py, 6 bytes, sha256 " in text
    assert "----- end file 2 of 2 -----" in text
    # The short digest in the header is the leading twelve of the file's
    # own digest, the prefix every refusal in this codebase quotes.
    short = built["manifest"]["files"][0]["digest"][:12]
    assert f"src/a.py, 12 bytes, sha256 {short} -----" in text


def test_one_file_reads_as_one_file():
    """WINDOW: SNAPSHOT_INTRO's plural, at a count of one.

    Python 3.11 will not nest same quotes inside an f-string, which is
    why the conditional is a function; this is the assertion that the
    function is wired to the template.
    """
    built = snapshot([("a.py", b"x = 1\n")])
    assert "A snapshot of 1 file from" in built["text"]


def test_the_rendition_names_this_reading():
    """WINDOW: compose's return, the four fields an attachment row needs.

    THE WALKER'S IDENTITY IS PROVENANCE, exactly as a parser's is.
    Renditions are keyed on (digest, extractor, extractor_version), so
    the composition returns those beside the text rather than letting a
    caller record only the result: a snapshot recorded without its
    version is a record of a prompt nobody can rebuild once the
    composition rule changes.
    """
    built = snapshot([("a.py", b"x = 1\n")])
    assert built["extractor"] == SNAPSHOT_EXTRACTOR
    assert built["extractor_version"] == SNAPSHOT_VERSION
    assert built["kind"] == SNAPSHOT_KIND
    assert built["digest"] == digest_of(built["text"])


def test_the_manifest_records_the_members_the_exclusions_and_the_clone():
    """WINDOW: compose's manifest, for a snapshot with a head and a flag.

    The exclusions are recorded AS APPLIED, so the record says what the
    walk was told to ignore and a reader never has to go and read the
    version of DEFAULT_EXCLUDES that happens to be installed. The head
    and the dirty flag are read locally by the caller and passed in,
    because reading them is I/O and this module does none.
    """
    built = compose(
        [("src/a.py", b"x = 1\n")],
        patterns=["src/**"],
        head="0123456789abcdef",
        dirty=True,
    )
    manifest = built["manifest"]
    # A member's digest is over the file's own bytes, which is not what
    # digest_of answers: that one is the composed snapshot's digest and
    # is over its UTF-8 encoding. They coincide on ASCII and asserting
    # the member with the other helper would hide the difference.
    assert manifest["files"] == [
        {
            "path": "src/a.py",
            "size": 6,
            "digest": hashlib.sha256(b"x = 1\n").hexdigest(),
        }
    ]
    assert manifest["patterns"] == ["src/**"]
    assert manifest["excludes"] == list(DEFAULT_EXCLUDES)
    assert manifest["head"] == "0123456789abcdef"
    assert manifest["dirty"] is True
    assert manifest["encoding"] == ENCODING_RULE


def test_the_manifest_carries_no_absolute_path():
    """WINDOW: compose's manifest, searched for the root it walked.

    The root is the one field here that would put a person's home
    directory into an export, and nothing downstream needs it: member
    paths are repo-relative and that is what a reader of a snapshot
    wants. compose is never told the root at all, which is the strongest
    form of this guarantee, and this asserts the signature stays that
    way.
    """
    built = snapshot([("src/a.py", b"x = 1\n")])
    assert "root" not in built["manifest"]
    for value in built["manifest"].values():
        assert "/repo" not in repr(value)


# ----- the include list ----------------------------------------------


@pytest.mark.parametrize(
    "pattern, reason",
    [
        ("/src/**", "repo-relative"),
        ("src/", "repo-relative"),
        ("src\\main.py", "backslash"),
        ("../other/**", "'..'"),
        ("  src/**", "whitespace"),
        ("", "empty"),
    ],
)
def test_a_pattern_that_could_only_match_nothing_is_refused(pattern, reason):
    """WINDOW: enforce_patterns, one malformed include pattern.

    NOT A CONTAINMENT CHECK, and the docstring says so: matching cannot
    escape a root whatever the pattern says, because every candidate
    path is built by the walk out of directory entries and never out of
    the pattern. These are refused so the person learns what went wrong,
    since the alternative is a silent empty selection.
    """
    with pytest.raises(SnapshotError) as caught:
        enforce_patterns([pattern])
    assert reason in str(caught.value)


def test_an_empty_or_oversized_include_list_is_refused():
    """WINDOW: enforce_patterns, at both ends of the list."""
    with pytest.raises(SnapshotError) as caught:
        enforce_patterns([])
    assert "at least one include pattern" in str(caught.value)
    with pytest.raises(SnapshotError) as caught:
        enforce_patterns(["*.py"] * 21)
    assert "over the 20 limit" in str(caught.value)


# ----- the module's bounds -------------------------------------------


def test_snapshot_imports_nothing_from_bench_but_extract():
    """WINDOW: the import block of bench/snapshot.py, read as a tree.

    THE PHASE'S BOUND, asserted rather than promised. extract holds the
    vocabulary a rendition is spoken in, and this module reaches it for
    the kinds, the composed ceiling and the image signatures; a second
    copy of any of those would be a second number to drift. Anything
    further into bench would be the API boundary or the store, and a
    pure module that imported either could not be tested as a value in
    and a value out. extract itself imports nothing from bench, so the
    edge is one directed hop with no cycle.
    """
    source = Path("bench/snapshot.py").read_text()
    reached = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module)
        elif isinstance(node, ast.Import):
            reached.update(alias.name for alias in node.names)
    assert {name for name in reached if name.startswith("bench")} == {"bench.extract"}


def test_the_readme_names_exclusions_the_walker_actually_applies():
    """WINDOW: the README's default-exclusions sentence, against
    DEFAULT_EXCLUDES.

    A WORKED LIST IS A PROMISE, the same argument the dataset example
    lives under one module over. Someone reading it will believe their
    .env is excluded, and an entry named in the documentation that the
    walker does not hold is a person shipping a credential to a provider
    because they read the README. Extracted from the text rather than
    restated, so the two cannot drift.
    """
    text = Path("README.md").read_text(encoding="utf-8")
    paragraphs = [
        block
        for block in text.split("\n\n")
        if block.startswith("Default exclusions apply")
    ]
    assert len(paragraphs) == 1, "the README's exclusion paragraph moved"
    named = set(re.findall(r"`([^`]+)`", paragraphs[0]))
    assert named, "the paragraph names no exclusions at all"
    for pattern in named:
        assert pattern in DEFAULT_EXCLUDES, pattern
    # And the secrets group specifically, because it is the one the
    # commission did not ask for and the one a reader most needs to be
    # able to trust.
    assert {".env", "*.pem", "*.key", "id_rsa*", ".netrc", ".npmrc"} <= named
