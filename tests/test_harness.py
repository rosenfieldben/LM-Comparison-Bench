"""Guards over the harness itself, rather than over the bench.

Everything here asserts a property of the TEST SUITE. A suite that can
quietly stop testing what it claims to test is worse than a gap, because
a gap is visible and a green run is not, so the few conventions the suite
depends on are enforced here rather than remembered.
"""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

# httpx's two client classes. Both read the ambient proxy environment
# when trust_env is on, and both are therefore in scope; there are no
# sync ones in the suite today, which is exactly when a rule is cheapest
# to state.
CLIENT_NAMES = ("AsyncClient", "Client")


def _client_constructions():
    """Every httpx client built anywhere under tests/, with its line.

    AST rather than a token scan, and that is a departure from the
    store's synchronous guard next door. The rule there is about the
    PRESENCE of a name, which tokens answer exactly; this rule is about a
    call's ARGUMENTS, which tokens cannot see at all. Matching
    "AsyncClient(" as text would be defeated by a line break and would
    match this file's own prose.
    """
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name in CLIENT_NAMES:
                yield path.relative_to(TESTS.parent), node


def _is_hermetic(node):
    """Whether this construction pins trust_env off, literally.

    A literal False and nothing else. `trust_env=flag` may well be False
    at runtime and the guard cannot know, so it refuses: a rule that
    accepted "probably" would be a rule the next reader argues with, and
    the whole value of this check is that it is not arguable. A `**opts`
    splat is refused for the same reason and arrives here as a keyword
    whose arg is None.
    """
    for keyword in node.keywords:
        if keyword.arg == "trust_env":
            return (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            )
    return False


def test_every_test_http_client_is_built_trust_env_false():
    """respx cannot save a client that fails while being constructed.

    respx intercepts at the TRANSPORT. Reading proxy environment
    variables happens a layer above that, inside AsyncClient.__init__:
    with trust_env on, httpx resolves the ambient proxy while the client
    is being built, and for a socks proxy with no socksio installed it
    raises ImportError right there. The client never exists, so respx
    never gets a request to intercept, and a test that touches no network
    fails because of the network anyway.

    THIS IS NOT HYPOTHETICAL, which is why the guard exists rather than
    the comment. Two tests were added in I.3 with a bare
    httpx.AsyncClient() beside thirteen carrying the flag. They passed
    everywhere the author ran them and failed on a reviewer's machine
    under the poisoned-proxy run, as the only two failures in the suite:

        ImportError: Using SOCKS proxy, but the 'socksio' package is not
        installed. Make sure to install httpx using `pip install
        httpx[socks]`.

    The author's own poisoned-proxy gate could not catch it. That sandbox
    set both `https_proxy` (lowercase, a working http proxy) and
    `HTTPS_PROXY`, and urllib.request.getproxies_environment lowercases
    every name it reads, so the ambient value won and the poisoned one
    was discarded. The gate reported green for a run that was never
    poisoned. See a3257d4.

    So the convention cannot rest on being remembered. Fifteen call sites
    carried an unexplained flag and the sixteenth did not, and nothing in
    the suite noticed. This makes the day someone writes the sixteenth a
    day the suite goes red, with this docstring attached to the failure.

    DELIBERATELY BLUNT, in the one way that matters: there is no
    allowance for "this one is fine". A test client that genuinely wants
    the ambient proxy wants to reach the network, which no test in this
    suite may do; the browser harness's real exception lives in a
    subprocess environment rather than in a client construction, and is
    documented where it is made. If a real exception ever arrives, it
    gets an argued-for change to this guard rather than a quiet
    keyword.
    """
    offenders = [
        f"{path}:{node.lineno}"
        for path, node in _client_constructions()
        if not _is_hermetic(node)
    ]

    assert offenders == [], (
        f"every httpx client under tests/ must be built trust_env=False; "
        f"found {offenders}. See this test's docstring: with trust_env "
        "on, httpx resolves the ambient proxy while CONSTRUCTING the "
        "client, which is a layer above where respx intercepts, so the "
        "client can fail before respx ever sees a request."
    )


def test_the_guard_can_see_the_clients_it_is_guarding():
    """The scope control, and it is not decoration.

    A source scan that matched nothing would pass forever and assert
    nothing, which is the failure mode of every check written against a
    codebase it does not understand. This one pins that the walk finds
    the constructions that exist, so a refactor moving them out of AST
    reach fails here rather than silently disarming the guard above.
    """
    found = list(_client_constructions())

    assert len(found) >= 15, found
    assert {str(path) for path, _ in found} >= {
        "tests/test_reconcile.py",
        "tests/test_models.py",
        "tests/test_properties.py",
    }
