"""The member listing, pure: one survey, two consumers, and what each sees.

list_members and walk iterate one Survey. These tests hold three things
to account. The rewrite of walk onto the survey changed nothing walk
returns, refuses or opens, proved against walk as it stood at d1d792d
(tests/walk_oracle.py) over generated trees with the review's FILE races
(a file swapped for a link, a fifo or another file at the open, a file
grown before its read); the directory races have their own tests in
test_snapshot.py, and the one deliberate change, a link target UTF-8
cannot spell now named by its repr, has its own test here. A second
change came after the operator's pass: a directory holding a git
directory's four names is refused, with its own tests at the end of
this file; no generated tree holds them (the differential's PRE-STATE),
so the oracle still speaks for every tree it is given. The listing
agrees with the composer on a still tree: the same selection, the same
first refusal, the same entry ceiling. And the listing reads nothing:
no member is opened, and no byte is read.

The ceilings are shrunk for the generated trees, in both modules at
once, so a tree of a dozen entries reaches every one of them.
"""

import hashlib
import inspect
from contextlib import contextmanager
from unittest import mock

import pytest
import walk_oracle
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_snapshot import ROOT, FakeTree

from bench import snapshot
from bench.snapshot import (
    DEFAULT_EXCLUDES,
    DIRECTORY,
    EXCLUDE_GROUPS,
    FILE,
    MAX_WALKED_ENTRIES,
    OTHER,
    SnapshotError,
    Survey,
    compose,
    composed_chars_at_most,
    list_members,
    matches,
    walk,
)

# DEFAULT_EXCLUDES as it stood before the groups were named, written out
# rather than derived: a pin built from EXCLUDE_GROUPS would move with
# it and could not catch a reorder. Every manifest records this tuple,
# and SNAPSHOT_VERSION's rule is about it.
EXCLUDES_AT_D1D792D = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "vendor",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
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

# The sha256 of walk's text at d1d792d, from "def walk(" to the line
# before "def _too_large(" with the trailing blank lines removed (which
# is how inspect.getsource returns it), as `git show
# d1d792d:bench/snapshot.py` gave it when the oracle was extracted. CI's
# checkout is shallow, so the commit itself is not there to ask; the
# digest is.
ORACLE_DIGEST = "abbf18c5b5d3e7782cf0a5b4fc2d54f83f4919ce33ef96301f44a35ab09fbb9c"

# Small enough that a generated tree of a dozen entries reaches each.
SMALL = {
    "MAX_WALKED_ENTRIES": 14,
    "MAX_DEPTH": 4,
    "MAX_MEMBER_BYTES": 40,
    "MAX_READ_BYTES": 60,
    "MAX_COMPOSED_CHARS": 700,
}


@contextmanager
def small_ceilings():
    """The ceilings shrunk in bench.snapshot and in the oracle together,
    since the oracle bound them by name when it was imported."""
    with (
        mock.patch.multiple(snapshot, **SMALL),
        mock.patch.multiple(
            walk_oracle, **{k: v for k, v in SMALL.items() if hasattr(walk_oracle, k)}
        ),
    ):
        yield


def outcome(run):
    """What a walk did: its return, or the refusal it raised."""
    try:
        return ("returned", run())
    except SnapshotError as exc:
        return ("refused", str(exc))


def ledgers(fake):
    """Every open, close and read the tree saw, IN ORDER."""
    return (list(fake.opened), list(fake.closed), list(fake.reads))


# ----- the generated trees -------------------------------------------

NAMES = [
    "a",
    "b",
    "src",
    ".git",
    "node_modules",
    "x.py",
    "k.py",
    "m.py",
    "y.md",
    ".env.db",
    "z.pyc",
]
# A git directory's four names, written out rather than read from the
# module, so each PRE-STATE in this file that names them says the same
# thing before the git directory rule existed as after.
FOUR = ("HEAD", "config", "objects", "refs")
# A name UTF-8 cannot spell, as os.scandir hands back a byte it cannot
# decode. One tree in ten holds one, since any tree that does refuses.
BAD_NAME = "bad" + chr(0xDCFF) + ".py"
# Weighted toward patterns that select, so most trees reach the checks
# past selection rather than stopping at "no file matched".
PATTERN_SETS = [
    ["**/*"],
    ["**/*"],
    ["**/*.py"],
    ["**/*.py"],
    ["**/*.py"],
    ["*.py"],
    ["src/**"],
    ["*.md", "**/*.py"],
    ["**/x.py", "b/**"],
    [".env.db"],
    ["**/*.pyc", "a/**"],
]


@st.composite
def trees(draw, *, races=True):
    """A tree spec: files, links, others, and optionally the review's
    races, with no path both a file and a directory."""
    paths = draw(
        st.lists(
            st.lists(st.sampled_from(NAMES), min_size=1, max_size=6).map("/".join),
            min_size=3,
            max_size=16,
            unique=True,
        )
    )
    # sampled_from rather than an integer draw, which Hypothesis biases
    # toward its bounds.
    if draw(st.sampled_from([False] * 9 + [True])):
        paths.append(draw(st.sampled_from(["", "src/", "a/b/"])) + BAD_NAME)
    # A path that is a proper prefix of another is a directory; only
    # leaves become files, links and others.
    leaves = [p for p in paths if not any(q.startswith(p + "/") for q in paths)]
    kinds = draw(
        st.lists(
            st.sampled_from(
                ["file", "file", "file", "file", "file", "link_in", "link_out", "other"]
            ),
            min_size=len(leaves),
            max_size=len(leaves),
        )
    )
    files, links, others = {}, {}, []
    for leaf, kind in zip(leaves, kinds, strict=True):
        if kind == "file":
            size = draw(st.integers(min_value=0, max_value=44))
            label = (leaf + ":").encode("utf-8", "surrogateescape")
            files[leaf] = label[:size].ljust(size, b"x")
        elif kind == "link_in":
            links[leaf] = f"{ROOT}/{draw(st.sampled_from(NAMES))}"
        elif kind == "link_out":
            links[leaf] = draw(st.sampled_from(["/etc/passwd", "/elsewhere/a"]))
        else:
            others.append(leaf)
    swap, grow = {}, {}
    if races and files:
        for leaf in draw(st.lists(st.sampled_from(sorted(files)), max_size=3)):
            swap[leaf] = draw(
                st.sampled_from(
                    [
                        {"kind": "symlink", "size": 0, "identity": (1, 900)},
                        {"kind": OTHER, "size": 0, "identity": (1, 901)},
                        {
                            "kind": FILE,
                            "size": 2,
                            "identity": (1, 902),
                            "content": b"zz",
                        },
                    ]
                )
            )
        for leaf in draw(st.lists(st.sampled_from(sorted(files)), max_size=3)):
            grow[leaf] = b"g" * draw(st.integers(min_value=1, max_value=50))
    patterns = draw(st.sampled_from(PATTERN_SETS))
    return {
        "files": files,
        "links": links,
        "others": others,
        "swap": swap,
        "grow": grow,
        "patterns": patterns,
    }


def build(spec):
    return FakeTree(
        spec["files"],
        links=spec["links"],
        others=spec["others"],
        swap_on_open=spec["swap"],
        grow_on_read=spec["grow"],
    )


# ----- walk is unchanged ---------------------------------------------


def test_the_oracle_is_walk_as_it_stood_at_d1d792d():
    """WINDOW: tests/walk_oracle.py's walk, as source text.

    The oracle every equivalence test below leans on is the text the
    commit held, not a transcription: its digest is the one taken from
    `git show d1d792d:bench/snapshot.py` when it was extracted.
    PRE-STATE: it is not the walk in bench.snapshot today, which would
    make every comparison below a comparison with itself."""
    text = inspect.getsource(walk_oracle.walk)
    assert text != inspect.getsource(snapshot.walk)
    assert hashlib.sha256(text.encode()).hexdigest() == ORACLE_DIGEST


def test_default_excludes_did_not_move_when_the_groups_were_named():
    """WINDOW: DEFAULT_EXCLUDES against the literal tuple at d1d792d.

    Naming the three groups must not move one entry: the tuple is what
    every manifest records, and a change to it is a SNAPSHOT_VERSION
    change this phase does not make. PRE-STATE: the groups are three,
    in the README's order, and the secrets group is last."""
    assert [name for name, _ in EXCLUDE_GROUPS] == [
        "version control and dependency trees",
        "build output and caches",
        "secrets",
    ]
    assert DEFAULT_EXCLUDES == EXCLUDES_AT_D1D792D


@settings(
    max_examples=400,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(spec=trees())
def test_walk_on_the_survey_is_walk_as_it_was(spec):
    """WINDOW: walk and the d1d792d oracle over the same generated tree,
    two FakeTrees built from one spec, with the ceilings shrunk so the
    trees reach them, and the review's races (a file swapped for a link,
    a fifo or another file at the open, a file grown before its read).

    THE REWRITE IS HELD TO THE OLD TEXT, not to the tests of the moved
    lines, which six mutants of those lines passed at the design review.
    Equal outcomes (the same members, or the same refusal word for word)
    and equal ledgers: every open, close and read in the same order.
    PRE-STATE: the oracle is today's walk only by that assertion; see the
    digest test above; and no generated name is one of the four a git
    directory holds, folded for case, so no generated tree meets the rule
    the oracle predates."""
    assert not {n.casefold() for n in NAMES} & {n.casefold() for n in FOUR}
    new, old = build(spec), build(spec)
    with small_ceilings():
        got = outcome(lambda: walk(tree=new, patterns=spec["patterns"]))
        want = outcome(lambda: walk_oracle.walk(tree=old, patterns=spec["patterns"]))
    assert got == want
    assert ledgers(new) == ledgers(old)


# The four orderings the review found no test holding, each a mutant of
# the moved lines that passed every test before this file.


def test_an_excluded_link_out_of_the_root_is_skipped_not_refused():
    """WINDOW: walk over a tree whose top-level node_modules is a link out
    of the root, under '**/*'.

    Exclusion is decided before the link is resolved, so an excluded
    link is never looked at. PRE-STATE: the same link under another name
    is refused."""
    links = {"node_modules": "/elsewhere/nm"}
    assert walk(tree=FakeTree({"a.py": b"a"}, links=links), patterns=["**/*"]) == [
        ("a.py", b"a")
    ]
    with pytest.raises(SnapshotError, match="^deps is a symbolic link"):
        walk(
            tree=FakeTree({"a.py": b"a"}, links={"deps": "/elsewhere/nm"}),
            patterns=["**/*"],
        )


def test_a_link_out_of_the_root_refuses_though_no_pattern_matches_it():
    """WINDOW: walk and list_members over a tree with docs/out.md linked
    to /etc/motd, under '*.py'.

    The containment check runs before the pattern, so a link that leaves
    the root refuses whatever the patterns select, and the listing says
    so in the same sentence. PRE-STATE: '*.py' selects a.py alone."""
    fake = FakeTree({"a.py": b"a"}, links={"docs/out.md": "/etc/motd"})
    with pytest.raises(SnapshotError) as raised:
        walk(tree=fake, patterns=["*.py"])
    listed = list_members(
        tree=FakeTree({"a.py": b"a"}, links={"docs/out.md": "/etc/motd"}),
        patterns=["*.py"],
    )
    assert [m["path"] for m in listed["members"] if m["status"] == "selected"] == [
        "a.py"
    ]
    assert listed["refusal"] == str(raised.value)
    assert listed["would_compose"] is False


def test_a_fifo_no_pattern_matches_still_refuses():
    """WINDOW: walk over a tree with run/db.sock a fifo, under '*.py'.

    The kind is checked before the pattern: a socket, a device or a pipe
    anywhere the walk reaches refuses. PRE-STATE: without it, '*.py'
    composes a.py."""
    assert walk(tree=FakeTree({"a.py": b"a"}), patterns=["*.py"]) == [("a.py", b"a")]
    with pytest.raises(SnapshotError, match="^run/db.sock is not a regular file"):
        walk(tree=FakeTree({"a.py": b"a"}, others=["run/db.sock"]), patterns=["*.py"])


def test_an_excluded_file_a_pattern_names_is_not_counted_as_matched():
    """WINDOW: walk over {.env, src/a.py} under ['.env'].

    The empty-selection sentence names the patterns that matched no
    selected file. An excluded file is not selected, so '.env' is named.
    PRE-STATE: the tree holds a .env the pattern spells exactly."""
    fake = FakeTree({".env": b"KEY=1", "src/a.py": b"a"})
    with pytest.raises(SnapshotError) as raised:
        walk(tree=fake, patterns=[".env"])
    assert str(raised.value).startswith("no file under the root matched '.env'.")


def test_the_budget_counts_what_was_read_not_what_was_listed():
    """WINDOW: walk over six files listed at 2 bytes each that read as
    150,000 each (the same identity, swapped at the open).

    MAX_READ_BYTES bounds what is HELD, so walk spends the bytes it read.
    A walk that spent the listed sizes would return 900,000 bytes, past
    the bound the constant exists for. PRE-STATE: listed, the six total
    12 bytes."""
    files = {f"f{n}.py": b"ab" for n in range(6)}
    fake = FakeTree(files)
    swap = {
        path: {**fake.nodes[path], "size": 150_000, "content": b"x" * 150_000}
        for path in files
    }
    grown = FakeTree(files, swap_on_open=swap)
    assert sum(node["size"] for p, node in grown.nodes.items() if p in files) == 12
    with pytest.raises(SnapshotError, match="^the selection passed 800000 bytes"):
        walk(tree=grown, patterns=["*.py"])


def test_a_refusal_part_way_closes_every_handle_deepest_first():
    """WINDOW: walk over a/b/c/huge.py past the member bound, the ledger
    of closes in order.

    A refusal three directories down closes c, then b, then a, then the
    root: the deepest first. PRE-STATE: the refusal is the member bound,
    raised while all four are open."""
    fake = FakeTree({"a/b/c/huge.py": b"x" * 250_000})
    with pytest.raises(SnapshotError, match="^a/b/c/huge.py is 250000 bytes"):
        walk(tree=fake, patterns=["**/*.py"])
    assert fake.closed == ["a/b/c", "a/b", "a", ""]


def test_a_survey_is_walked_once():
    """WINDOW: iterating one Survey twice.

    A survey counts entries against the ceiling as it goes, so a second
    pass would count them twice and meet the ceiling at half the tree.
    PRE-STATE: the first pass yields the tree's one member."""
    survey = Survey(tree=FakeTree({"a.py": b"a"}), patterns=["*.py"])
    assert [s.path for s in survey] == ["a.py"]
    with pytest.raises(RuntimeError, match="iterated once"):
        iter(survey)


# ----- the listing agrees with the composer --------------------------


@settings(
    max_examples=400,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(spec=trees(races=False))
def test_on_a_still_tree_the_listing_is_what_the_composer_does(spec):
    """WINDOW: list_members and walk, then compose, over the same
    generated still tree (no swap, no growth, every file readable), with
    the ceilings shrunk.

    ONE WALK, TWO DOORS, HELD ON EVERY TREE. would_compose is true
    exactly when walk reaches composition; the listing's refusal is the
    sentence walk raises first; its selected rows are walk's members,
    path and size. Its composed_chars_at_most is compose's length exactly
    on this text of one byte per character, so compose refuses on the
    character ceiling exactly when the bound passes it. And nothing was
    opened but directories, and nothing read. PRE-STATE: the listing's
    tree saw no read at all, and the composer's saw one per member."""
    listed_tree, walked_tree = build(spec), build(spec)
    with small_ceilings():
        listed = list_members(tree=listed_tree, patterns=spec["patterns"])
        walked = outcome(lambda: walk(tree=walked_tree, patterns=spec["patterns"]))
        if walked[0] == "returned":
            composed = outcome(
                lambda: compose(
                    walked[1], patterns=spec["patterns"], head=None, dirty=None
                )
            )
    if walked[0] == "returned":
        assert len(walked_tree.reads) == len(walked[1]) > 0
    assert listed_tree.reads == []
    assert all(
        listed_tree.nodes[path]["kind"] == DIRECTORY for path in listed_tree.opened
    )
    assert listed["would_compose"] is (walked[0] == "returned")
    if walked[0] == "refused":
        assert listed["refusal"] == walked[1]
        return
    selected = [
        (m["path"], m["bytes"]) for m in listed["members"] if m["status"] == "selected"
    ]
    assert selected == [(p, len(d)) for p, d in walked[1]]
    assert listed["selected_bytes"] == sum(len(d) for _, d in walked[1])
    bound = listed["composed_chars_at_most"]
    if composed[0] == "returned":
        assert bound == len(composed[1]["text"])
        assert [f["path"] for f in composed[1]["manifest"]["files"]] == [
            p for p, _ in selected
        ]
        assert bound <= SMALL["MAX_COMPOSED_CHARS"]
    else:
        assert composed[1].startswith(f"the snapshot composes to {bound} characters")


def test_the_entry_ceiling_is_the_same_sentence_at_the_same_count():
    """WINDOW: list_members and walk over a root whose listing streams
    without end, as a home directory's would, at the real ceiling.

    The listing stops where the composer stops, having consumed the same
    MAX_WALKED_ENTRIES + 1 entries and no more, reports the composer's
    sentence as its refusal, marks itself incomplete, and names the
    directory it was listing as the refused row. PRE-STATE: the walk's
    own refusal is the entry ceiling at that count."""
    consumed = {"walk": 0, "list": 0}

    def endless(which):
        def hook(path):
            n = 0
            while True:
                consumed[which] += 1
                n += 1
                yield snapshot.Entry(f"f{n}", FILE, (1, 10_000 + n), 1)

        return hook

    with pytest.raises(SnapshotError) as raised:
        walk(tree=FakeTree({}, listing_hook=endless("walk")), patterns=["**/*"])
    assert str(raised.value).startswith(f"the walk passed {MAX_WALKED_ENTRIES}")
    listed = list_members(
        tree=FakeTree({}, listing_hook=endless("list")), patterns=["**/*"]
    )
    assert consumed == {"walk": MAX_WALKED_ENTRIES + 1, "list": MAX_WALKED_ENTRIES + 1}
    assert listed["refusal"] == str(raised.value)
    assert listed["complete"] is False
    assert listed["counted"] == MAX_WALKED_ENTRIES + 1
    assert listed["members"][-1:] == [
        {
            "path": "",
            "bytes": None,
            "kind": DIRECTORY,
            "status": "refused",
            "reason": str(raised.value),
        }
    ]


def test_an_earlier_refusal_does_not_hide_the_ceiling():
    """WINDOW: list_members over a root holding a link out of the root and
    a directory wider than the entry ceiling (shrunk), under '**/*'.

    The composer raises the link first, and the listing's refusal is that
    sentence. But the listing goes on and reaches the ceiling, and says
    so: incomplete, with the ceiling's sentence as the last row at the
    directory it was listing. Without that row a table that looked whole
    would be missing everything past the ceiling. PRE-STATE: walk's first
    refusal is the link."""
    files = {f"z/f{n:02d}.py": b"x" for n in range(20)}
    links = {"a": "/elsewhere"}
    with small_ceilings():
        with pytest.raises(SnapshotError, match="^a is a symbolic link"):
            walk(tree=FakeTree(files, links=links), patterns=["**/*"])
        listed = list_members(tree=FakeTree(files, links=links), patterns=["**/*"])
    assert listed["refusal"].startswith("a is a symbolic link")
    assert listed["complete"] is False
    last = listed["members"][-1]
    assert last["path"] == "z" and last["status"] == "refused"
    assert last["reason"].startswith(f"the walk passed {SMALL['MAX_WALKED_ENTRIES']}")


def test_the_stopping_refusal_is_the_last_row_whatever_its_path():
    """WINDOW: list_members over a/x.py and a directory a-b wider than the
    entry ceiling (shrunk), under '**/*.py'.

    The walk reaches 'a' before 'a-b' (siblings in name order), but by
    path 'a-b' sorts before 'a/x.py' ('-' is below '/'), so a sort alone
    would put the stop first. It is the last row, as the listing says.
    PRE-STATE: the stop is at a-b, and it sorts before a/x.py."""
    files = {"a/x.py": b"x", **{f"a-b/f{n}.py": b"f" for n in range(20)}}
    with small_ceilings():
        listed = list_members(tree=FakeTree(files), patterns=["**/*.py"])
    assert listed["complete"] is False
    assert b"a-b" < b"a/x.py"
    assert [(m["path"], m["status"]) for m in listed["members"]] == [
        ("a/x.py", "selected"),
        ("a-b", "refused"),
    ]


def test_the_read_ceiling_comes_before_a_later_refused_entry():
    """WINDOW: list_members and walk over three 30-byte files and then a
    pipe 'zz', under '**/*', with the read ceiling shrunk to 60.

    The selection crosses the ceiling at the third file, before the walk
    reaches the pipe, so the composer's first refusal is the ceiling and
    so is the listing's, though the pipe is a refused row too.
    PRE-STATE: without the ceiling shrunk, walk's refusal is the pipe."""
    files = {f"f{n}.py": b"x" * 30 for n in range(3)}
    with pytest.raises(SnapshotError, match="^zz is not a regular file"):
        walk(tree=FakeTree(files, others=["zz"]), patterns=["**/*"])
    with small_ceilings():
        with pytest.raises(SnapshotError) as raised:
            walk(tree=FakeTree(files, others=["zz"]), patterns=["**/*"])
        listed = list_members(tree=FakeTree(files, others=["zz"]), patterns=["**/*"])
    assert str(raised.value).startswith("the selection passed 60 bytes")
    assert listed["refusal"] == str(raised.value)
    assert ("zz", "refused") in [(m["path"], m["status"]) for m in listed["members"]]


def test_a_refused_entry_before_the_read_ceiling_is_the_refusal():
    """WINDOW: list_members and walk over a pipe 'a' and then three 30-byte
    files, under '**/*', with the read ceiling shrunk to 60.

    The walk meets the pipe first ('a' sorts before 'f0.py'), so the
    composer's first refusal is the pipe's and so is the listing's,
    though the selection goes on to cross the ceiling after it: the
    listing keeps the refusal it met first and does not let the ceiling
    replace it. A deterministic tree, because CI's Hypothesis profile
    fixes its examples and the property did not reach this order there
    (the Phase O review's M9). PRE-STATE: without the pipe, the same
    walk's refusal is the ceiling."""
    files = {f"f{n}.py": b"x" * 30 for n in range(3)}
    with small_ceilings():
        with pytest.raises(SnapshotError, match="^the selection passed 60 bytes"):
            walk(tree=FakeTree(files), patterns=["**/*"])
        with pytest.raises(SnapshotError) as raised:
            walk(tree=FakeTree(files, others=["a"]), patterns=["**/*"])
        listed = list_members(tree=FakeTree(files, others=["a"]), patterns=["**/*"])
    assert str(raised.value).startswith("a is not a regular file")
    assert listed["refusal"] == str(raised.value)


def test_the_first_read_ceiling_crossing_is_the_refusal():
    """WINDOW: list_members and walk over three 30-byte files and a 40-byte
    one, under '*.py', with the read ceiling shrunk to 60 and the member
    ceiling to 40.

    The selection crosses the ceiling at the third file and again at the
    fourth, and the two crossings word different sentences (the three
    largest files so far differ). The listing's refusal is the first
    crossing's, as the composer's is: the sentence naming f0.py, f1.py
    and f2.py at 30 bytes each. A later crossing does not replace it. A
    deterministic tree, for the reason the test above gives (the Phase O
    review's M9). PRE-STATE: a fresh budget refuses at both spends, and
    the two sentences differ."""
    files = {**{f"f{n}.py": b"x" * 30 for n in range(3)}, "f3.py": b"x" * 40}
    with small_ceilings():
        budget = snapshot._ReadBudget()
        for name in ("f0.py", "f1.py"):
            assert budget.spend(name, 30) is None
        first = budget.spend("f2.py", 30)
        second = budget.spend("f3.py", 40)
        assert first is not None and second is not None
        assert str(first) != str(second)
        with pytest.raises(SnapshotError) as raised:
            walk(tree=FakeTree(files), patterns=["*.py"])
        listed = list_members(tree=FakeTree(files), patterns=["*.py"])
    assert "f0.py at 30 bytes, f1.py at 30 bytes, f2.py at 30 bytes" in str(
        raised.value
    )
    assert listed["refusal"] == str(raised.value) == str(first)


def test_the_refusal_is_the_first_the_composer_meets_not_the_first_row():
    """WINDOW: list_members over {x.py, a.py as a fifo, a/link out of the
    root} under '**/*'.

    Directory 'a' is entered before 'a.py' (siblings sort by name, and
    'a' < 'a.py'), so the composer's first refusal is the link, while the
    rows, sorted by path, put a.py first ('.' < '/'). The listing's
    refusal is the composer's. PRE-STATE: walk raises the link."""
    spec = {"files": {"x.py": b"x"}, "links": {"a/link": "/etc"}, "others": ["a.py"]}
    with pytest.raises(SnapshotError) as raised:
        walk(
            tree=FakeTree(spec["files"], links=spec["links"], others=spec["others"]),
            patterns=["**/*"],
        )
    assert str(raised.value).startswith("a/link is a symbolic link")
    listed = list_members(
        tree=FakeTree(spec["files"], links=spec["links"], others=spec["others"]),
        patterns=["**/*"],
    )
    refused = [m["path"] for m in listed["members"] if m["status"] == "refused"]
    assert refused == ["a.py", "a/link"]
    assert listed["refusal"] == str(raised.value)


def test_excluded_rows_name_the_group_and_never_the_size():
    """WINDOW: list_members over a tree holding .env.db, .git/config,
    src/a.py and a stray build.pyc, under '**/*'.

    An excluded directory is a row; an excluded file is a row when a
    pattern matches it. Each names its pattern and group, SECRETS FIRST:
    '.env.db' also matches the build group's '*.db', and a key file is
    called what it is. No excluded row carries a size, which for a
    secret would say how long it is. PRE-STATE: excluded() alone files
    '.env.db' under '*.db'."""
    assert snapshot.excluded(".env.db", DEFAULT_EXCLUDES) == "*.db"
    listed = list_members(
        tree=FakeTree(
            {
                ".env.db": b"SECRET=1",
                ".git/config": b"c",
                "src/a.py": b"a",
                "z.pyc": b"p",
            }
        ),
        patterns=["**/*"],
    )
    rows = {m["path"]: m for m in listed["members"]}
    assert rows[".env.db"]["status"] == "excluded"
    assert "'.env.*'" in rows[".env.db"]["reason"]
    assert "(secrets)" in rows[".env.db"]["reason"]
    assert rows[".git"]["kind"] == DIRECTORY
    assert "(version control and dependency trees)" in rows[".git"]["reason"]
    assert "(build output and caches)" in rows["z.pyc"]["reason"]
    assert ".git/config" not in rows
    assert all(
        m["bytes"] is None for m in listed["members"] if m["status"] == "excluded"
    )
    assert rows["src/a.py"] == {
        "path": "src/a.py",
        "bytes": 1,
        "kind": FILE,
        "status": "selected",
        "reason": None,
    }


def test_an_excluded_file_no_pattern_matches_is_not_a_row():
    """WINDOW: list_members over {.DS_Store, a.py} under '*.py'.

    A '*.py' listing does not report every excluded file in the tree,
    only those a pattern would otherwise have selected. PRE-STATE: under
    '**/*' the same .DS_Store is a row."""
    both = {".DS_Store": b"d", "a.py": b"a"}
    assert ".DS_Store" in [
        m["path"]
        for m in list_members(tree=FakeTree(both), patterns=["**/*"])["members"]
    ]
    assert [
        m["path"]
        for m in list_members(tree=FakeTree(both), patterns=["*.py"])["members"]
    ] == ["a.py"]


def test_a_link_inside_the_root_a_pattern_matches_is_reported_not_followed():
    """WINDOW: list_members over README.md, docs/x.md and docs/README.md
    linked to the top-level README.md, under 'docs/*.md'.

    The composer skips a link inside the root without a word, and here
    its target is not selected, so the file it names is in no member.
    The listing says so. PRE-STATE: walk composes docs/x.md alone."""
    spec = (
        {"README.md": b"r", "docs/x.md": b"x"},
        {"docs/README.md": f"{ROOT}/README.md"},
    )
    assert [
        p
        for p, _ in walk(tree=FakeTree(spec[0], links=spec[1]), patterns=["docs/*.md"])
    ] == ["docs/x.md"]
    rows = {
        m["path"]: m
        for m in list_members(
            tree=FakeTree(spec[0], links=spec[1]), patterns=["docs/*.md"]
        )["members"]
    }
    assert rows["docs/README.md"]["status"] == "excluded"
    assert "to README.md, inside the snapshot root" in rows["docs/README.md"]["reason"]


def test_a_link_inside_the_root_no_pattern_matches_is_not_a_row():
    """WINDOW: list_members over docs/x.md and a link docs/README.md to the
    top-level README.md, under '*.py' and under 'docs/*.md'.

    A link inside the root is a row only when a pattern matches it, as an
    excluded file is. PRE-STATE: under 'docs/*.md' it is a row."""
    files, links = (
        {"README.md": b"r", "docs/x.md": b"x", "a.py": b"a"},
        {"docs/README.md": f"{ROOT}/README.md"},
    )
    matched = list_members(tree=FakeTree(files, links=links), patterns=["docs/*.md"])
    assert "docs/README.md" in [m["path"] for m in matched["members"]]
    other = list_members(tree=FakeTree(files, links=links), patterns=["*.py"])
    assert [m["path"] for m in other["members"]] == ["a.py"]


def test_a_link_to_an_excluded_file_says_no_pattern_can_make_it_a_member():
    """WINDOW: list_members over a.py, .env and config.py linked to .env,
    under '*.py'.

    What the link names is a default exclusion, which no pattern can turn
    off, so the row says so rather than "read only if a pattern selects
    that path". PRE-STATE: .env is excluded by '.env'."""
    assert snapshot.excluded(".env", DEFAULT_EXCLUDES) == ".env"
    listed = list_members(
        tree=FakeTree(
            {"a.py": b"a", ".env": b"K=1"}, links={"config.py": f"{ROOT}/.env"}
        ),
        patterns=["*.py"],
    )
    row = next(m for m in listed["members"] if m["path"] == "config.py")
    assert row["status"] == "excluded"
    assert row["reason"] == (
        "config.py is a symbolic link to .env, inside the snapshot root, and "
        "is not followed; .env is itself excluded by '.env', so no pattern can "
        "make it a member."
    )


def test_a_link_target_utf8_cannot_spell_is_named_by_its_repr():
    """WINDOW: walk and list_members over a link out of the root whose
    resolved target holds a byte that is not UTF-8 (a surrogate escape,
    as os.path.realpath returns it).

    The sentence carried the raw surrogate, and a response carrying it
    could not be written: POST /snapshots answered 500. It now names the
    target by its repr, and the sentence is UTF-8. PRE-STATE: the target
    itself cannot be encoded."""
    target = "/elsewhere/caf" + chr(0xDCE9)
    with pytest.raises(UnicodeEncodeError):
        target.encode("utf-8")
    with pytest.raises(SnapshotError) as raised:
        walk(tree=FakeTree({"a.py": b"a"}, links={"l": target}), patterns=["*.py"])
    str(raised.value).encode("utf-8")
    assert repr(target) in str(raised.value)
    listed = list_members(
        tree=FakeTree({"a.py": b"a"}, links={"l": target}), patterns=["*.py"]
    )
    assert listed["refusal"] == str(raised.value)


def test_a_listing_that_raises_part_way_closes_what_it_opened():
    """WINDOW: list_members over a tree whose deep directory refuses at
    its descent (swapped for a file), the ledger of opens and closes.

    The survey's own refusal ends it with every directory closed, deepest
    first, and nothing but directories opened. PRE-STATE: the refusal is
    the swap's."""
    files = {"a/b/c.py": b"c"}
    fake = FakeTree(files)
    fake.swap_on_open = {"a/b": {"kind": FILE, "size": 1, "identity": (1, 99)}}
    listed = list_members(tree=fake, patterns=["**/*"])
    assert listed["refusal"].startswith("a/b was listed as a directory")
    assert listed["complete"] is False
    assert fake.opened == ["", "a"]
    assert fake.closed == ["a", ""]
    assert fake.reads == []


def test_composed_chars_at_most_is_composes_length_on_ascii_and_bounds_the_rest():
    """WINDOW: composed_chars_at_most over (path, size) pairs against the
    length of compose's text over the same members, at the real ceilings.

    Exact on text of one byte per character with no byte-order mark, an
    upper bound otherwise: a multibyte character and a removed mark both
    make the text shorter than its bytes. PRE-STATE: the multibyte file
    is longer in bytes than in characters."""
    ascii_members = [("b.py", b"print(1)\n"), ("a/x.md", b"# x\n" * 30), ("e.py", b"")]
    text = compose(ascii_members, patterns=["**/*"], head=None, dirty=None)["text"]
    assert composed_chars_at_most([(p, len(d)) for p, d in ascii_members]) == len(text)
    wide = [("w.txt", "été".encode()), ("m.py", b"\xef\xbb\xbfx = 1\n")]
    assert len(wide[0][1]) > len(wide[0][1].decode())
    text = compose(wide, patterns=["**/*"], head=None, dirty=None)["text"]
    assert composed_chars_at_most([(p, len(d)) for p, d in wide]) > len(text)


def test_nothing_selected_is_the_composers_sentence():
    """WINDOW: list_members and walk over a tree '*.md' selects nothing in.

    PRE-STATE: the tree holds files, none of them Markdown."""
    with pytest.raises(SnapshotError) as raised:
        walk(tree=FakeTree({"a.py": b"a"}), patterns=["*.md"])
    listed = list_members(tree=FakeTree({"a.py": b"a"}), patterns=["*.md"])
    assert listed["refusal"] == str(raised.value)
    assert listed["members"] == []
    assert listed["composed_chars_at_most"] is None
    assert listed["complete"] is True


# ----- a git directory, by what it holds -----------------------------
#
# Found after the operator's pass at 9c920e5 and pre-existing since
# Phase L: a directory holding HEAD, config, objects and refs is a git
# directory whatever it is named, and the walk read one like any other.
# These are the refusal's pure proofs, over the dict-backed tree and its
# ledger of opens and reads. The fixture proof, a bare repository whose
# config carries a sentinel string in the userinfo position of a remote
# URL, is its own commit.

# The shapes a git directory's top level takes here, each one git reads
# as its own: the four names as git writes them; spelled as a disk that
# folds case may hold them (git asking for config is served CONFIG);
# refs a file, which git takes when it may execute it (a FakeTree keeps
# no modes); and objects a link to a directory inside the root, which
# git follows.
SHAPES = {
    "as-git-writes-them": FOUR,
    "spelled-otherwise": ("Head", "CONFIG", "Objects", "REFS"),
    "refs-a-file": FOUR,
    "objects-a-link": FOUR,
}
CONFIG_BODY = b"[core]\n\tbare = true\n"


def git_directory(at="", shape="as-git-writes-them"):
    """A git directory's top level at `at`, in one of SHAPES, as the
    (files, links) a FakeTree takes."""
    head, config, objects, refs = (f"{at}/{n}" if at else n for n in SHAPES[shape])
    files = {head: b"ref: refs/heads/main\n", config: CONFIG_BODY}
    links = {}
    if shape == "objects-a-link":
        links[objects] = f"{ROOT}/store"
        files["store/info/packs"] = b"\n"
    else:
        files[f"{objects}/info/packs"] = b"\n"
    if shape == "refs-a-file":
        files[refs] = b"\n"
    else:
        files[f"{refs}/heads/main"] = b"0" * 40
    return files, links


def without(tree, name, at=""):
    """The same (files, links) with one of the four names at `at` spelled
    otherwise, so the directory carries three of them."""
    old = f"{at}/{name}" if at else name

    def moved(path):
        if path == old or path.startswith(old + "/"):
            return old + ".old" + path[len(old) :]
        return path

    return tuple({moved(path): value for path, value in part.items()} for part in tree)


def planted(tree, **kwargs):
    files, links = tree
    return FakeTree(files, links=links, **kwargs)


def read_paths(fake):
    return [path for path, _ in fake.reads]


def under(path, at):
    return path.startswith(at + "/")


# Patterns for a root, from the four names as the shape spells them: one
# that names config, one that names everything, and one that misses
# config, since the rule holds whatever the patterns select.
ROOT_PATTERNS = {
    "under-config": lambda spelled: [spelled[1]],
    "under-everything": lambda spelled: ["**/*"],
    "under-a-pattern-that-misses-config": lambda spelled: [spelled[0]],
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("which", ROOT_PATTERNS)
def test_a_root_that_is_a_git_directory_opens_nothing_in_it(which, shape):
    """WINDOW: walk and list_members over a FakeTree whose root is a git
    directory in each of SHAPES, under a pattern that names config, one
    that names everything, and one that misses config (HEAD).

    The survey stops at the root before any member is opened, whatever
    the patterns select: the one handle either consumer opens is the
    root's, nothing is read, and so config is never opened. The composer
    raises the root's sentence; the listing's one row is that sentence,
    at the root, final, after the root's entries were counted. PRE-STATE:
    with any one of the four names spelled otherwise, a walk under '**/*'
    returns, having opened and read config (config.old when config is
    the one respelled), so three of the four are not the rule and the
    ledger records an open when there is one; and with refs spelled
    otherwise, the walk under the case's own patterns returns, having
    read exactly what they select."""
    spelled = SHAPES[shape]
    patterns = ROOT_PATTERNS[which](spelled)
    config = spelled[1]
    for name in spelled:
        fake = planted(without(git_directory(shape=shape), name))
        members = dict(walk(tree=fake, patterns=["**/*"]))
        read = config + ".old" if name == config else config
        assert read in fake.opened and read in read_paths(fake)
        assert members[read] == CONFIG_BODY
    fake = planted(without(git_directory(shape=shape), spelled[3]))
    members = dict(walk(tree=fake, patterns=patterns))
    assert members and sorted(members) == sorted(read_paths(fake))

    tree = git_directory(shape=shape)
    fake = planted(tree)
    with pytest.raises(SnapshotError) as refused:
        walk(tree=fake, patterns=patterns)
    assert (fake.opened, fake.closed, fake.reads) == ([""], [""], [])
    looked = planted(tree)
    listed = list_members(tree=looked, patterns=patterns)
    assert (looked.opened, looked.closed, looked.reads) == ([""], [""], [])
    assert listed["members"] == [
        {
            "path": "",
            "bytes": None,
            "kind": DIRECTORY,
            "status": "refused",
            "reason": str(refused.value),
        }
    ]
    assert (listed["complete"], listed["would_compose"]) == (False, False)
    assert listed["refusal"] == str(refused.value)
    assert listed["counted"] == len(
        {path.split("/")[0] for part in tree for path in part}
    )
    assert str(refused.value) == snapshot.ROOT_IS_GIT_DIRECTORY


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("at", ["g", "fixtures/g"], ids=["under-the-root", "two-down"])
@pytest.mark.parametrize(
    "patterns", [["**/*"], ["*.py"]], ids=["everything", "misses-config"]
)
def test_a_git_directory_the_walk_reaches_opens_nothing_in_it(patterns, at, shape):
    """WINDOW: walk and list_members over a FakeTree holding a.py, a git
    directory in each of SHAPES at g and at fixtures/g, and zz.py, under
    '**/*' and under '*.py', which misses config.

    The git directory is opened to be listed, which is how its names are
    seen, and nothing under it is opened or read: not config, not
    anything. The composer raises the walked directory's sentence at it,
    having read only a.py; the listing's row for it is that sentence, and
    the walk goes on past it to what the patterns select after it, as it
    does past a directory too deep. PRE-STATE: with HEAD spelled
    otherwise, the same walk under '**/*' opens and reads the git
    directory's config and returns it, and under '*.py' composes a.py and
    zz.py, so the refusal is what changes the outcome."""
    spelled = SHAPES[shape]
    config = f"{at}/{spelled[1]}"
    parents = [at[:i] for i, c in enumerate(at) if c == "/"]
    files, links = git_directory(at, shape)
    tree = ({"a.py": b"a", **files, "zz.py": b"z"}, links)
    fake = planted(without(tree, spelled[0], at))
    members = dict(walk(tree=fake, patterns=["**/*"]))
    assert config in fake.opened and config in read_paths(fake)
    assert members[config] == CONFIG_BODY
    fake = planted(without(tree, spelled[0], at))
    assert [path for path, _ in walk(tree=fake, patterns=["*.py"])] == ["a.py", "zz.py"]

    fake = planted(tree)
    with pytest.raises(SnapshotError) as refused:
        walk(tree=fake, patterns=patterns)
    assert fake.opened == ["", "a.py", *parents, at]
    assert read_paths(fake) == ["a.py"]
    assert fake.closed == ["a.py", at, *reversed(parents), ""]
    looked = planted(tree)
    listed = list_members(tree=looked, patterns=patterns)
    assert at in looked.opened and looked.reads == []
    assert [path for path in looked.opened if under(path, at)] == []
    assert {
        "path": at,
        "bytes": None,
        "kind": DIRECTORY,
        "status": "refused",
        "reason": str(refused.value),
    } in listed["members"]
    assert [m for m in listed["members"] if under(m["path"], at)] == []
    assert [m["path"] for m in listed["members"] if m["status"] == "selected"] == [
        path
        for path in sorted(tree[0])
        if not under(path, at) and any(matches(path, p) for p in patterns)
    ]
    assert (listed["complete"], listed["would_compose"]) == (True, False)
    assert listed["refusal"] == str(refused.value)
    assert str(refused.value) == snapshot.HOLDS_GIT_DIRECTORY


def test_the_walks_own_refusals_come_before_the_git_directory_rule():
    """WINDOW: list_members over a.py, a git directory at fixtures/g and
    zz.py, under '**/*'; then the same tree with a name UTF-8 cannot
    spell inside fixtures/g; then with the entry ceiling shrunk to fall
    inside fixtures/g's listing.

    The rule reads the names the directory's listing already gathered:
    fixtures/g's four are counted before it (8 in all: a.py, fixtures,
    zz.py, g, and g's four), and objects and refs are never descended. A
    name UTF-8 cannot spell inside it stops the survey there with the
    UTF-8 sentence, and a listing past the ceiling stops it with the
    ceiling's: refusals about the traversal, which come first.
    PRE-STATE: with HEAD spelled otherwise the walk descends objects and
    refs and counts 12; the other two trees differ from the first by one
    name and by one ceiling."""
    files, _ = git_directory("fixtures/g")
    files = {"a.py": b"a", **files, "zz.py": b"z"}
    control = list_members(
        tree=planted(without((files, {}), "HEAD", "fixtures/g")), patterns=["**/*"]
    )
    assert control["counted"] == 12

    listed = list_members(tree=FakeTree(files), patterns=["**/*"])
    assert listed["would_compose"] is False
    assert listed["counted"] == 8
    assert listed["refusal"] == snapshot.HOLDS_GIT_DIRECTORY
    bad = list_members(
        tree=FakeTree({**files, f"fixtures/g/{BAD_NAME}": b"q"}), patterns=["**/*"]
    )
    assert (bad["complete"], bad["members"][-1]["path"]) == (False, "fixtures/g")
    assert bad["refusal"].startswith("fixtures/g holds an entry whose name is not")
    with mock.patch.object(snapshot, "MAX_WALKED_ENTRIES", 6):
        wide = list_members(tree=FakeTree(files), patterns=["**/*"])
    assert (wide["complete"], wide["members"][-1]["path"]) == (False, "fixtures/g")
    assert wide["refusal"].startswith("the walk passed 6 directory entries")


def test_neither_git_directory_sentence_names_a_path():
    """WINDOW: the sentence each rule raises, from both consumers, over
    two trees in which the root and every entry a refusal could be about
    carry 'zq' (the git directory's own four names aside): a root at
    /zqroot/zqbare.git that is a git directory, and a root at
    /zqroot/zqtree holding one at zqdir/zqgit beside a link out of the
    root at zqdir/zqlink.

    Neither sentence holds 'zq' or a path separator: each names the rule,
    and the listing's row is what says where (its path is zqdir/zqgit).
    PRE-STATE: the path was at hand (each root is its tree's), and the
    walk's other refusals name theirs: the link's sentence, in the same
    listing, carries 'zq', so the check can see a path when a sentence
    holds one."""
    at_root = planted(git_directory(), root="/zqroot/zqbare.git")
    files, _ = git_directory("zqdir/zqgit")
    below = ({**files, "zqdir/zqz.py": b"z"}, {"zqdir/zqlink": "/zqelsewhere/zqfile"})
    listed = list_members(tree=planted(below, root="/zqroot/zqtree"), patterns=["**/*"])
    rows = {m["path"]: m for m in listed["members"]}
    assert rows["zqdir/zqlink"]["status"] == "refused"
    assert "zq" in rows["zqdir/zqlink"]["reason"]
    assert at_root.root_path() == "/zqroot/zqbare.git"

    sentences = []
    for fake in (at_root, planted(below, root="/zqroot/zqtree")):
        with pytest.raises(SnapshotError) as refused:
            walk(tree=fake, patterns=["**/*"])
        sentences.append(str(refused.value))
    sentences.append(
        list_members(
            tree=planted(git_directory(), root="/zqroot/zqbare.git"),
            patterns=["**/*"],
        )["refusal"]
    )
    assert rows["zqdir/zqgit"]["status"] == "refused"
    sentences.append(rows["zqdir/zqgit"]["reason"])
    for sentence in sentences:
        assert "zq" not in sentence and "/" not in sentence and "\\" not in sentence
    assert sentences == [
        snapshot.ROOT_IS_GIT_DIRECTORY,
        snapshot.HOLDS_GIT_DIRECTORY,
        snapshot.ROOT_IS_GIT_DIRECTORY,
        snapshot.HOLDS_GIT_DIRECTORY,
    ]


def test_the_git_directory_signature_is_four_names_folded_for_case():
    """WINDOW: carries_git_directory over sets of names, and the names
    GIT_DIRECTORY_SIGNATURE pins.

    All four, whatever else is beside them and however they are cased,
    from a list or a generator, and folded fully (casefold, not lower):
    a disk that folds case, as APFS does, serves a name spelled with
    U+FB01 (the fi ligature) when config is asked for, and one spelled
    with U+017F (long s) for refs. PRE-STATE: no three of the four are
    enough; names that only resemble them (HEAD.lock, configs, object,
    ref, .config) are not them; and the two spellings fold to config and
    refs under casefold and not under lower."""
    carries = snapshot.carries_git_directory
    assert snapshot.GIT_DIRECTORY_SIGNATURE == FOUR
    for name in FOUR:
        assert not carries([n for n in FOUR if n != name])
    assert not carries(["HEAD.lock", "configs", "object", "ref", ".config"])
    assert not carries([])
    assert carries(FOUR)
    assert carries(["description", *FOUR, "hooks", "info", "packed-refs"])
    assert carries(["head", "CONFIG", "Objects", "REFS"])
    assert carries(name for name in FOUR)
    ligature, long_s = "con" + chr(0xFB01) + "g", "ref" + chr(0x17F)
    assert ligature.lower() != "config" and ligature.casefold() == "config"
    assert long_s.lower() != "refs" and long_s.casefold() == "refs"
    assert carries(["HEAD", ligature, "objects", long_s])


def test_the_git_directory_sentences_are_the_operators_words():
    """WINDOW: ROOT_IS_GIT_DIRECTORY and HOLDS_GIT_DIRECTORY against the
    words the operator ruled after f8bde6b, written out.

    A person reads these two sentences at both doors, so they are pinned
    as ruled rather than to whatever the module holds. PRE-STATE: each is
    the sentence the walk raises, over a root that is a git directory and
    over one the walk reaches, so the pin is on what a person reads."""
    for tree, constant in (
        (git_directory(), snapshot.ROOT_IS_GIT_DIRECTORY),
        (({"a.py": b"a", **git_directory("g")[0]}, {}), snapshot.HOLDS_GIT_DIRECTORY),
    ):
        with pytest.raises(SnapshotError) as refused:
            walk(tree=planted(tree), patterns=["**/*"])
        assert str(refused.value) == constant
    assert snapshot.ROOT_IS_GIT_DIRECTORY == (
        "the root is a git directory, whose config can carry a remote URL "
        "with sign-in details in it, so it is not walked."
    )
    assert snapshot.HOLDS_GIT_DIRECTORY == (
        "the walk reached a git directory, whose config can carry a remote "
        "URL with sign-in details in it, so the walk stopped there."
    )
