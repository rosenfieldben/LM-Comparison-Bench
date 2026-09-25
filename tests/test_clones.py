"""The clone door's pure half: what a request may name, and where it lands.

bench/clones.py decides everything about a clone request that can be
decided without a process: the URL parsed and rebuilt, the ref held to
git's grammar, the directory named from the two. These tests hold the
URL rules to the commission's list, every refusal to never echoing the
URL, the ref rules to git itself (git check-ref-format, run here over
generated names), and the directory name to its formula computed
independently.
"""

import hashlib
import random
import subprocess

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bench import clones
from bench.clones import CloneError, canonical_url, check_ref, clone_dir_name

HOSTS = ("github.com",)

# A token no rule sentence could contain by accident, planted in every
# part of a URL a refusal might be tempted to quote.
SECRET = "Zq9TOKENx7"


def refusal(url: str, hosts: tuple[str, ...] = HOSTS) -> CloneError:
    with pytest.raises(CloneError) as caught:
        canonical_url(url, hosts)
    return caught.value


def test_a_plain_github_url_is_rebuilt_from_its_parts():
    """WINDOW: canonical_url on the two path spellings the commission
    names, /owner/repo and /owner/repo.git.

    PRE-STATE: the allowlist is the default, github.com alone. What
    comes back is spelled from the parsed parts, so the host is
    lowercased and nothing the request wrote survives unparsed."""
    assert clones.parse_clone_hosts(None) == HOSTS
    assert canonical_url("https://github.com/owner/repo", HOSTS) == (
        "https://github.com/owner/repo"
    )
    assert canonical_url("https://GitHub.com/o-w/r.e_p.git", HOSTS) == (
        "https://github.com/o-w/r.e_p.git"
    )


@pytest.mark.parametrize(
    ("url", "rule", "status"),
    [
        ("http://github.com/o/r", "not https", 403),
        ("ssh://github.com/o/r", "not https", 403),
        ("git://github.com/o/r", "not https", 403),
        ("file:///srv/o/r.git", "not https", 403),
        ("github.com/o/r", "not https", 403),
        ("-x", "not https", 403),
        ("https://user@github.com/o/r", "a user or a token", 403),
        ("https://user:pw@github.com/o/r", "a user or a token", 403),
        ("https://:pw@github.com/o/r", "a user or a token", 403),
        ("https://@github.com/o/r", "a user or a token", 403),
        ("https://gitlab.com/o/r", "not one of BENCH_CLONE_HOSTS", 403),
        ("https://github.com.evil.example/o/r", "not one of BENCH_CLONE_HOSTS", 403),
        ("https://github.com:443/o/r", "not one of BENCH_CLONE_HOSTS", 403),
        ("https://[::1]/o/r", "not one of BENCH_CLONE_HOSTS", 403),
        ("https://github.com./o/r", "not one of BENCH_CLONE_HOSTS", 403),
        ("https://github.com/o/r?x=1", "a query or a fragment", 422),
        ("https://github.com/o/r?", "a query or a fragment", 422),
        ("https://github.com/o/r#main", "a query or a fragment", 422),
        ("https://github.com/o/r#", "a query or a fragment", 422),
        ("https://github.com/o", "not /owner/repo", 422),
        ("https://github.com/o/r/", "not /owner/repo", 422),
        ("https://github.com/o/r/tree/main", "not /owner/repo", 422),
        ("https://github.com//o/r", "not /owner/repo", 422),
        ("https://github.com/-x/r", "owner", 422),
        ("https://github.com/o--o/r", "owner", 422),
        ("https://github.com/o_o/r", "owner", 422),
        ("https://github.com/" + "o" * 40 + "/r", "owner", 422),
        ("https://github.com/o/-x", "repository", 422),
        ("https://github.com/o/.r", "repository", 422),
        ("https://github.com/o/..", "repository", 422),
        ("https://github.com/o/r.git.git", "repository", 422),
        ("https://github.com/o/r@x", "repository", 422),
        ("https://github.com/o/" + "r" * 101, "repository", 422),
        ("https://github.com/o/r%2e", "percent-escape", 422),
        ("https://github.com/o%2Fx/r", "percent-escape", 422),
        ("https://github.com\\@evil.example/o/r", "backslash", 422),
        (" https://github.com/o/r", "whitespace or a control", 422),
        ("https://github.com/o/r\n", "whitespace or a control", 422),
        ("https://github.com/o/r\tx", "whitespace or a control", 422),
        ("https://github.com:99999/o/r", "cannot be read", 422),
        ("https://github.com:x/o/r", "cannot be read", 422),
        ("https://[github.com/o/r", "cannot be read", 422),
    ],
)
def test_each_url_rule_refuses_naming_the_rule(url, rule, status):
    """WINDOW: canonical_url on one URL that breaks one rule.

    Every shape the commission lists (scheme, host, userinfo, query,
    fragment, path) and the parser's own edges. The sentence names the
    rule, the status says whose it is (the operator's policy 403, the
    request's shape 422). PRE-STATE: the URL with the offending part
    repaired is accepted, so the refusal is that part's and not a
    neighbour's (the unreadable ports and the bare word excepted: they
    have no repaired twin)."""
    exc = refusal(url)
    assert rule in str(exc)
    assert exc.status == status


def test_the_repaired_twins_of_the_refused_urls_are_accepted():
    """WINDOW: canonical_url on the URLs the table above refuses, each
    with its one fault removed. The table's PRE-STATE, run once."""
    for url in (
        "https://github.com/o/r",
        "https://github.com/o-o/r",
        "https://github.com/" + "o" * 39 + "/r",
        "https://github.com/o/_r",
        "https://github.com/o/r.x",
        "https://github.com/o/" + "r" * 100,
        "https://github.com/o/r.git",
    ):
        assert canonical_url(url, HOSTS).startswith("https://github.com/")


@pytest.mark.parametrize(
    "url",
    [
        f"https://{SECRET}@github.com/o/r",
        f"https://x:{SECRET}@github.com/o/r",
        f"https://x:{SECRET}@gitlab.com/o/r",
        f"http://x:{SECRET}@github.com/o/r",
        f"https://github.com/o/r?token={SECRET}",
        f"https://github.com/o/r#{SECRET}",
        f"https://{SECRET}.example/o/r",
        f"https://github.com/{SECRET}/r/extra",
        f"https://github.com/o/{SECRET}%40",
        f"https://github.com/o/{SECRET} ",
        f"https://github.com:{SECRET}/o/r",
        f"{SECRET}://github.com/o/r",
    ],
)
def test_no_refusal_echoes_any_part_of_the_url(url):
    """WINDOW: the sentence of the refusal for a URL carrying a planted
    token in the part the refusal is about.

    The commission's rule: a URL carrying user:token@ refuses naming the
    rule and never echoes the URL, and every other refusal names the rule
    and not the URL. PRE-STATE: the token is in the URL."""
    assert SECRET in url
    assert SECRET not in str(refusal(url))


@settings(max_examples=300, deadline=None)
@given(
    st.text(
        alphabet=st.characters(codec="utf-8", exclude_characters="\x00"),
        min_size=1,
        max_size=60,
    )
)
def test_no_refusal_of_any_string_echoes_it(text):
    """WINDOW: canonical_url on arbitrary text wrapped around the token.

    Whatever the string, the refusal carries none of it. PRE-STATE: the
    string is refused (no generated text spells an accepted URL, since
    the token's own segment would have to be a whole owner)."""
    url = f"https://{text}{SECRET}{text}"
    with pytest.raises(CloneError) as caught:
        canonical_url(url, HOSTS)
    assert SECRET not in str(caught.value)


def test_a_host_entry_with_a_port_admits_exactly_that_port():
    """WINDOW: canonical_url against an allowlist holding 127.0.0.1:8443.

    A port is allowed exactly where the operator wrote one: the entry
    admits that port and neither the bare host nor another port.
    PRE-STATE: the entry parses to itself."""
    hosts = clones.parse_clone_hosts("127.0.0.1:8443")
    assert hosts == ("127.0.0.1:8443",)
    assert canonical_url("https://127.0.0.1:8443/o/r.git", hosts) == (
        "https://127.0.0.1:8443/o/r.git"
    )
    assert refusal("https://127.0.0.1/o/r.git", hosts).status == 403
    assert refusal("https://127.0.0.1:8444/o/r.git", hosts).status == 403


@pytest.mark.parametrize(
    ("raw", "hosts"),
    [
        (None, ("github.com",)),
        ("", ("github.com",)),
        ("  ", ("github.com",)),
        ("GitHub.com", ("github.com",)),
        (
            "github.com, gitlab.example:8443,github.com",
            ("github.com", "gitlab.example:8443"),
        ),
    ],
)
def test_the_host_allowlist_parses(raw, hosts):
    """WINDOW: parse_clone_hosts on one value of BENCH_CLONE_HOSTS.

    Unset or blank is the default; entries are lowercased and kept once
    each in the order written. PRE-STATE: the default is github.com."""
    assert clones.DEFAULT_CLONE_HOSTS == ("github.com",)
    assert clones.parse_clone_hosts(raw) == hosts


@pytest.mark.parametrize(
    "raw",
    [
        "github.com,",
        "https://github.com",
        "github.com/x",
        "a b",
        "h:0",
        "h:65536",
        "h:08",
        "h:",
    ],
)
def test_a_malformed_host_entry_fails_boot_naming_the_variable(raw):
    """WINDOW: parse_clone_hosts on a value with one malformed entry.

    Refused at boot like a malformed BENCH_REPO_ROOTS entry, naming the
    variable, rather than kept as a host nothing matches. PRE-STATE: the
    same value with the entry fixed parses."""
    assert clones.parse_clone_hosts("github.com,h:443")
    with pytest.raises(RuntimeError, match="BENCH_CLONE_HOSTS"):
        clones.parse_clone_hosts(raw)


def _git_accepts(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "check-ref-format", f"refs/heads/{ref}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _door_accepts(ref: str) -> bool:
    try:
        check_ref(ref)
    except CloneError:
        return False
    return True


def test_the_ref_grammar_is_gits_own_but_for_the_doors_three_rules():
    """WINDOW: check_ref against `git check-ref-format refs/heads/<ref>`
    over four hundred generated names and the named edge cases.

    Four hundred because each name is a git process: four thousand took
    fifty seconds here and found nothing four hundred did not.

    The door and git agree on every name except where the door is
    stricter on purpose: a leading '-' (the argument-injection line), a
    leading '+' (fetch's force flag) and '@' alone (HEAD). PRE-STATE:
    each of the three is a name git itself accepts, so each rule is the
    door's and not git's."""
    for own in ("-x", "+main", "@"):
        assert _git_accepts(own)
        assert not _door_accepts(own)
    alphabet = [*"ab/.-_@{}~^:?*[\\ +", "\t", "\x7f", "é", "lock", "HEAD"]
    rng = random.Random(11)
    names = [
        "main",
        "feature/x",
        "v1.0",
        "HEAD",
        "refs/heads/main",
        "a..b",
        "x.lock",
        "a/.b",
        "a.",
        "a@{b",
        "a b",
    ]
    names += [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
        for _ in range(400)
    ]
    differ = {
        name
        for name in names
        if _door_accepts(name) != _git_accepts(name)
        and not name.startswith(("-", "+"))
        and name != "@"
    }
    assert differ == set()


@pytest.mark.parametrize(
    ("ref", "rule"),
    [
        ("", "empty"),
        ("--upload-pack=x", "starts with '-'"),
        ("-", "starts with '-'"),
        ("+main", "force flag"),
        ("@", "names HEAD"),
        ("a\x01b", "control character"),
        ("a:b", "one of ~ ^ : ? * [ \\"),
        ("main:x", "one of ~ ^ : ? * [ \\"),
        ("a..b", "'..'"),
        ("a@{1}", "'@{'"),
        ("a//b", "'//'"),
        ("a.", "ends with '.'"),
        ("a/.b", "starts with '.'"),
        ("a.lock", "'.lock'"),
    ],
)
def test_each_ref_rule_refuses_naming_the_rule_and_the_ref(ref, rule):
    """WINDOW: check_ref on one ref breaking one rule.

    The rule is named, and the ref beside it when it is spelled like a
    ref. PRE-STATE: 'main' passes."""
    check_ref("main")
    with pytest.raises(CloneError) as caught:
        check_ref(ref)
    assert rule in str(caught.value)
    named = all(c.isalnum() or c in "._/+-" for c in ref) and 0 < len(ref) <= 100
    opening = f"ref {ref!r} is not" if named else "the ref is not"
    assert str(caught.value).startswith(opening)
    assert caught.value.status == 422


def test_a_url_sent_as_the_ref_is_not_repeated():
    """WINDOW: check_ref on a URL carrying a token, as a person who
    swapped the two fields would send it.

    The ':' rule refuses it, and the sentence says "the ref" rather than
    repeating it. PRE-STATE: the token is in what was sent."""
    ref = f"https://u:{SECRET}@github.com/o/r"
    assert SECRET in ref
    with pytest.raises(CloneError) as caught:
        check_ref(ref)
    assert SECRET not in str(caught.value)
    assert "the ref is not" in str(caught.value)


def test_a_forty_hex_commit_is_a_ref():
    """WINDOW: check_ref on commit names.

    Forty lowercase hex is a commit. PRE-STATE: a thirty-nine-character
    one is still a ref git would accept as a branch name, and so is
    accepted as a name; the door does not guess which a person meant."""
    check_ref("0123456789abcdef0123456789abcdef01234567")
    check_ref("0123456789abcdef0123456789abcdef0123456")
    check_ref("0123456789ABCDEF0123456789ABCDEF01234567")


def test_the_clone_directory_is_the_commissions_formula():
    """WINDOW: clone_dir_name against sha256("url\\nref")[:16] computed
    here, over the repository's identity.

    One repository spelled with and without ".git" is one identity and so
    one directory. PRE-STATE: two refs of one repository, and one ref of
    two repositories, name different directories."""
    url = "https://github.com/owner/repo"
    expected = hashlib.sha256(b"https://github.com/owner/repo\nmain").hexdigest()[:16]
    assert clone_dir_name(url, "main") == expected
    assert clone_dir_name(url, "dev") != expected
    assert clone_dir_name(url + "2", "main") != expected
    assert len(expected) == 16
    assert clones.repository_identity(url + ".git") == url
    assert clones.repository_identity(url) == url
    assert clone_dir_name(clones.repository_identity(url + ".git"), "main") == expected


def test_clones_off_names_both_variables():
    """WINDOW: the CLONES_OFF sentence.

    It names the variable that turns the door on and the allowlist the
    clone root must belong to. PRE-STATE: it is not SNAPSHOTS_OFF."""
    from bench.main import SNAPSHOTS_OFF

    assert clones.CLONES_OFF != SNAPSHOTS_OFF
    assert "BENCH_CLONE_ROOT" in clones.CLONES_OFF
    assert "BENCH_REPO_ROOTS" in clones.CLONES_OFF
