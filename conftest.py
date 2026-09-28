# Root conftest so pytest puts the repo root on sys.path and tests can
# import the bench package without installing it.

import pytest


@pytest.fixture(autouse=True)
def _no_spend_limit_from_the_shell(monkeypatch):
    """A spend limit set in the shell running the suite is not the test's.
    Since Phase P, P2 a call is refused once what it would reserve does not
    fit under the limit, from the first call on, so an operator's limit
    would refuse runs the tests expect to make. A test that wants a limit
    sets it after this has run, as spend_client does."""
    monkeypatch.delenv("BENCH_SPEND_LIMIT_USD", raising=False)
