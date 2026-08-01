"""FastAPI boundary. Pydantic models live here only; internals use plain dicts."""

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bench import store
from bench.models import (
    BUDGET_EXTENDED,
    BUDGET_STANDARD,
    DATA_POLICY_PREFS,
    as_money,
    as_text,
    fetch_catalog,
    keepalive_socket_options,
    provider_preferences,
    run_model,
    stream_model,
)

logger = logging.getLogger(__name__)

# Budget names resolve to numbers here at the boundary; internals pass
# plain integers. Pydantic's Literal on the request models is what
# rejects unknown names, so this map never sees one.
BUDGET_TOKENS = {"standard": BUDGET_STANDARD, "extended": BUDGET_EXTENDED}

# The budget an offline boot falls back to when no cap is published. See
# effective_budget: offline is also unpriced, so the conservative tier is
# what keeps an unverified extended run from being both unbounded and
# uncounted.
UNKNOWN_CAP_BUDGET = BUDGET_STANDARD

# Ceiling on simultaneous paid upstream calls across everything in
# flight. The batch endpoint's five-model cap always implied this
# policy, but the UI's real path is one /compare/stream per model with
# no batch to cap, so overlapping runs and reruns could put unbounded
# paid calls in flight. The semaphore enforces the cap where the money
# actually moves: around the upstream HTTP exchange, in both endpoints.
# Saturation queues quietly; a sixth model simply starts when a slot
# frees, with no error and no acquisition timeout.
MAX_CONCURRENT_UPSTREAM = 5

# SQLite stores rowids in a signed 64-bit integer. Pydantic accepted
# any Python int, SQLite did not, and the gap surfaced as an
# OverflowError 500 after the paid upstream call had already
# completed. Bounding at the boundary makes the mismatch a 422 before
# any money moves.
MAX_SQLITE_ROWID = 2**63 - 1

# Upper bound on a declared column index. Deliberately not the batch
# endpoint's five-model cap: that cap belongs to /compare, while the
# streaming path is one request per model and the browser lineup has no
# size limit, so a wide comparison legitimately declares positions past
# five. This is only here to keep an absurd value out of the schema; a
# side-by-side visual comparison is tens of columns at the very most.
MAX_POSITION = 999


# Unknown fields are refused on every request model at the boundary, not
# ignored. A typo like {"budgte": "extended"} used to run a valid but
# unintended paid request: the caller asked for one thing, the bench did
# another, and the money moved either way. A 422 naming the offending field
# costs nothing and turns a silent wrong run into an obvious mistake.
# The audience is whoever hand-writes a request (curl, a script): FastAPI's
# 422 detail is a list of per-field errors, and the card's error path only
# renders a string detail, so a card would read "HTTP 422" rather than the
# field name. That is not worth widening the error renderer for, because the
# frontend sends fixed bodies and cannot produce this 422 at all.
FORBID_UNKNOWN = ConfigDict(extra="forbid")


# A seed is an opaque integer to every provider, so the bound exists only to
# keep an absurd body out, the same reason MAX_POSITION exists.
#
# 2**53 - 1, not signed 64-bit, and the difference is not cosmetic. This
# value is recorded, served back as JSON, and read in a browser, where every
# number is a double: measured rather than assumed, a stored seed of
# 9223372036854775807 came back to the page as 9223372036854776000. That
# would badge the wrong seed in history, and reuse would then prefill the
# wrong one and be refused by the one-experiment check as a different
# experiment, which is a legal run blocked by a silent corruption.
#
# So the bound is the largest integer the whole path can carry exactly. Nine
# quadrillion distinct seeds is not a constraint anyone will feel, and a
# value above it is now a 422 at the boundary instead of a lie in the
# record. Number.MAX_SAFE_INTEGER in the browser is the same number.
MAX_SEED = 2**53 - 1

# Long enough for a real system prompt and short enough that the group row
# stays a record rather than a document store. Bounded for the same reason
# every other free-text field here is.
MAX_SYSTEM_PROMPT = 8000


class ExperimentParams(BaseModel):
    """The controls one comparison holds constant across its models.

    Every field optional and every default None, which is the boundary
    expression of rule one: a control the user left blank is absent, not
    zero and not the tool's own default. Absence means the provider's
    default applies and the record can say so honestly. model_dump with
    exclude_none is what turns that into the payload and the stored record.

    One class referenced by all three request models rather than the same
    six fields written out three times. Parity between the batch and
    streaming surfaces is then structural: a field, a bound or a validator
    cannot drift between them because there is only one of each. The
    closing review for this phase hunts parity drift, and this is the shape
    that makes the hunt vacuous.

    Bounds are OpenRouter's own, pinned at implementation time; see
    bench.models for the documentation URLs and the read dates.
    """

    model_config = FORBID_UNKNOWN

    # min_length=1 rather than allowing "": an empty string is not a system
    # prompt the user set, it is a blank control, and a client that sends
    # one is confusing the two. Failing loudly here is what keeps a
    # meaningless leading system message off the wire and out of the record.
    system: str | None = Field(default=None, min_length=1, max_length=MAX_SYSTEM_PROMPT)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # Symmetric, unlike the signed-64 range this used to mirror. The bound
    # is now JS's safe-integer range exactly (MIN_SAFE_INTEGER is
    # -MAX_SAFE_INTEGER), because the limit that matters is what survives
    # being read back in a browser. See MAX_SEED.
    seed: int | None = Field(default=None, ge=-MAX_SEED, le=MAX_SEED)
    # Three named tiers of the seven OpenRouter documents. A comparison
    # holds one effort across models, and the widest set invites a
    # per-model guess about what "xhigh" means where it is unsupported.
    effort: Literal["low", "medium", "high"] | None = None
    # throughput is today's behavior and stays the default; "default" means
    # send no sort at all and let OpenRouter load balance. See
    # bench.models.ROUTING_PREFS for why that is a value and not an absence.
    routing: Literal["throughput", "price", "default"] | None = None


class CompareRequest(BaseModel):
    model_config = FORBID_UNKNOWN

    prompt: str = Field(min_length=1)
    # Cap at 5: the bench is for side-by-side eyeballing, and each extra
    # model is a concurrent upstream request on one API key.
    models: list[str] = Field(min_length=1, max_length=5)
    # Optional link back to a saved prompt so history can show where a
    # run came from. The run stores its own prompt_text either way.
    # Bounded to the rowid range; see MAX_SQLITE_ROWID.
    prompt_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    # Optional grouping id so the N per-model requests of one comparison
    # land as one history entry.
    group_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    # Named tiers, not a free integer: two regimes of use exist and a
    # free field invites typos with dollar consequences.
    budget: Literal["standard", "extended"] = "standard"
    # The experiment controls. /compare is the official scripting surface,
    # so it carries the same controls the streaming path does; see the
    # README's scripting section. See ExperimentParams for why one shared
    # class is what guarantees that parity.
    params: ExperimentParams | None = None


class ModelResult(BaseModel):
    model: str
    response_text: str | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None
    cost_usd: float | None
    # Time to first content delta. Only the streaming path measures it;
    # the default keeps /compare results (which never carry the key) valid.
    ttft_ms: float | None = None
    # The effective (post-clamp) completion budget the request was sent
    # with. Defaults None so pre-budget history rows stay valid.
    max_tokens: int | None = None
    # OpenRouter's response id (gen-...). It keys OpenRouter's
    # generation API, which records the actual provider, quantization
    # and authoritative cost, so a persisted id makes any historical
    # run auditable. Defaults None so pre-provenance rows stay valid.
    generation_id: str | None = None
    # Why output ended (stop, length, content_filter, ...). This is
    # OpenRouter's normalized value, not the provider's own word, which it
    # exposes separately as native_finish_reason; capturing that is Phase G
    # provenance work. Until now this survived only inside synthesized
    # error strings; budget analysis needs it on truncated-but-successful
    # runs too.
    finish_reason: str | None = None
    # Phase G provenance and billed truth. All default None so pre-G rows,
    # and the paths that do not set them, stay valid. position is the
    # column this result occupied; request_json is the exact payload sent,
    # auth excluded; billed_cost_usd is what the platform actually charged,
    # kept beside cost_usd rather than replacing it, because the estimate
    # and the charge are different facts and history needs both.
    position: int | None = None
    request_json: str | None = None
    billed_cost_usd: float | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    provider: str | None = None
    quantization: str | None = None
    native_finish_reason: str | None = None


class StreamCompareRequest(BaseModel):
    # Unknown fields refused; see FORBID_UNKNOWN.
    model_config = FORBID_UNKNOWN

    prompt: str = Field(min_length=1)
    # One model per streaming request, mirroring the frontend's
    # per-model fetch pattern: independent columns are the product.
    model: str = Field(min_length=1)
    # Bounded like CompareRequest's; see MAX_SQLITE_ROWID.
    prompt_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    group_id: int | None = Field(default=None, ge=1, le=MAX_SQLITE_ROWID)
    budget: Literal["standard", "extended"] = "standard"
    # Which column this run occupied in the comparison. The frontend fans
    # out one request per model, so only the client knows the layout; the
    # batch endpoint derives the same thing from array order. Recorded so
    # a replay can reconstruct the original side by side arrangement from
    # the rows themselves rather than from the current chip order, which
    # drifts as the lineup is edited. See MAX_POSITION for why the bound is
    # not the batch endpoint's five-model cap.
    position: int | None = Field(default=None, ge=0, le=MAX_POSITION)
    # Full parity with CompareRequest by construction; see ExperimentParams.
    # The frontend sends these on every stream request so the one
    # experiment per group check has something to check. The group's stored
    # copy, written before any upstream call, is the record.
    params: ExperimentParams | None = None


class CompareResponse(BaseModel):
    results: list[ModelResult]
    # None when persisting the run failed: the upstream spend already
    # happened, so the results are returned even when history is lost.
    run_id: int | None


class PromptCreate(BaseModel):
    # Unknown fields refused; see FORBID_UNKNOWN.
    model_config = FORBID_UNKNOWN

    name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Prompt(BaseModel):
    id: int
    name: str
    text: str
    created_at: str


class PromptList(BaseModel):
    prompts: list[Prompt]


class RunEntry(BaseModel):
    type: Literal["run"]
    id: int
    created_at: str
    prompt_text: str
    models: list[str]
    # The controls this run was sent with, or None when it shows none.
    # Derived from the recorded payload rather than stored, because
    # controls live on the group and this run has none; see
    # store._controls_from_request for why routing is never among them.
    params: dict[str, Any] | None = None


class GroupEntry(BaseModel):
    type: Literal["group"]
    id: int
    created_at: str
    prompt_text: str
    models: list[str]
    run_ids: list[int]
    # The declared controls, straight off the group row. Only what was
    # set, so history renders a badge for a choice and never for a default.
    params: dict[str, Any] | None = None


class RunList(BaseModel):
    runs: list[GroupEntry | RunEntry]


class GroupCreate(BaseModel):
    # Both optional so an empty JSON object stays a valid body: that is
    # what every pre-G client sends, and the empty object is load-bearing
    # for the CORS preflight defense. When present they record what the
    # comparison IS, before any upstream call, which is what makes the
    # group row the experiment record and fixes the group's prompt ahead
    # of its first member. See FORBID_UNKNOWN for the unknown-field rule.
    model_config = FORBID_UNKNOWN

    prompt: str | None = Field(default=None, min_length=1)
    # The ordered lineup as declared. Deliberately NOT CompareRequest's
    # five-model cap: that cap bounds concurrent paid calls inside one
    # batch request, while this is a record of a lineup the browser fans
    # out one request at a time and never caps. Borrowing it here rejected
    # every comparison wider than five, and because a failed group create
    # degrades to ungrouped runs rather than erroring, those comparisons
    # silently lost their history entry. Bounded at MAX_POSITION's scale
    # for the same reason that constant exists: keep an absurd body out
    # without capping real use.
    models: list[str] | None = Field(
        default=None, min_length=1, max_length=MAX_POSITION + 1
    )
    # The controls this comparison holds constant. Recorded on the group
    # because they are experiment-level: one comparison, one controls set,
    # applied to every model, which is the whole fairness argument. Written
    # here before any upstream call, which is what makes it the record the
    # per-run integrity check is checked against rather than a claim
    # assembled after the money moved.
    params: ExperimentParams | None = None
    # Required, unlike everything above it, and the asymmetry is the point.
    # The others are optional because pre-G and pre-H clients existed that
    # could not send them. Nothing has ever created a group without knowing
    # its budget, and a comparison whose members ran at different token
    # budgets is not one experiment, so accepting a group that declines to
    # say would be creating the very hole this workstream closes.
    budget: Literal["standard", "extended"]


class GroupCreated(BaseModel):
    id: int


class CatalogModel(BaseModel):
    id: str
    name: str | None
    context_length: int | None
    prompt_price: float | None
    completion_price: float | None
    # The published completion cap the budget clamp works from. Exposed
    # so the picker can show why "extended" quietly becomes less on
    # capped models; None when the catalog does not publish one.
    max_completion_tokens: int | None


class CatalogResponse(BaseModel):
    models: list[CatalogModel]
    # Lets the frontend tell an offline boot from an empty catalog and
    # switch to the exact-id fallback instead of a dead search box.
    fetched: bool
    # The boot-scoped data-handling policy, so the page can say plainly
    # that this session is not on the default routing. It rides the catalog
    # response rather than a new endpoint because the frontend already
    # fetches this once at boot, and one more round trip to learn one word
    # is not worth an endpoint.
    data_policy: str = "standard"


class RunDetail(BaseModel):
    id: int
    created_at: str
    prompt_text: str
    prompt_id: int | None
    results: list[ModelResult]
    # What this run actually was: the build that produced it, how old the
    # prices it was costed against were, and the data-handling policy its
    # payloads declared. None on every pre-G row.
    app_sha: str | None = None
    catalog_snapshot_at: str | None = None
    data_policy: str | None = None
    # Which catalog bytes this run was costed against, beside the timestamp
    # that says when they were read. The column lands with H1.2's migration
    # so the schema changes once; the value is written by H1.3, so every row
    # reads None until then and every legacy row reads None forever.
    catalog_digest: str | None = None
    # The controls this run was sent with, derived from its recorded payload
    # rather than stored: controls are declared on the group, and a run may
    # have none. See store._controls_from_request for why routing is never
    # among them.
    params: dict[str, Any] | None = None


class GroupDetail(BaseModel):
    id: int
    created_at: str
    runs: list[RunDetail]
    # The declaration, so the view can show what the comparison was meant
    # to be and not only what happened to run. A declared model with no
    # recorded run is the case this exists for: without these the detail
    # showed an incomplete experiment as a smaller one.
    #
    # All None on a group that predates the columns, where NULL means
    # unknown rather than empty (see store.group_manifest for why that is
    # the opposite of what NULL means for params). A renderer must treat
    # None as absence and never print it.
    prompt: str | None = None
    models: list[str] | None = None
    budget: str | None = None
    # The controls the comparison was declared with, or None when it held
    # none. Only set controls are present, so the detail view can render a
    # chosen control and never a default dressed up as one. None on every
    # pre-H group, which is a fact about those groups rather than a gap.
    params: dict[str, Any] | None = None


def _git(args: list[str]) -> str | None:
    """One git command's stdout, or None when git could not answer.

    Split out of _app_sha so both of that function's questions run under
    one timeout and degrade discipline, and so a test can substitute a
    runner without reaching into subprocess. A non-zero exit is an answer
    the caller cannot use, which is the same as no answer.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _app_sha() -> str | None:
    """The current commit, or None when that cannot be determined.

    Provenance for "what code produced this row". Read once at boot rather
    than per run: it cannot change while the process lives, and a
    subprocess per request would be absurd. Degrades to None whenever git
    is missing, the checkout is not a repository, or git is slow enough to
    hit the timeout, because none of those are reasons to fail a boot. The
    timeout is short for the same reason: this is a nice-to-have label,
    not a dependency.

    A bare sha claims the running code IS that commit, which is false for
    every uncommitted edit, and a bench whose whole point is provenance
    must not let altered code sign a clean commit's name. So the tree is
    checked too and a modified one is labelled "<sha>-dirty". When that
    second question cannot be answered the whole label degrades to None
    rather than to the bare sha: an unqualified sha is the clean claim,
    and asserting it unverified is exactly the lie this guards against.
    """
    sha = as_text((_git(["rev-parse", "HEAD"]) or "").strip()) or None
    if sha is None:
        return None
    # --porcelain is the stable machine format; empty output means clean.
    # Untracked files count as dirty here on purpose: an untracked module
    # on the import path changes what runs just as an edited one does.
    status = _git(["status", "--porcelain"])
    if status is None:
        return None
    return f"{sha}-dirty" if status.strip() else sha


def _parse_spend_limit(raw: str | None) -> float | None:
    """The validated per-boot spend ceiling, or None for no limit.

    Unset or empty means no limit. Anything present must parse to a
    strictly positive finite float. A non-finite or negative value is
    nonsense, and zero is ambiguous between a deliberate lockout and a
    misconfiguration; the two are indistinguishable, so both are refused
    loudly at boot rather than silently producing a broken or surprising
    ceiling. A bare float() would have accepted nan and inf (a nan ceiling
    is never crossed, silently disabling the limit) and negatives.
    """
    if not raw:
        return None
    try:
        limit = float(raw)
    except ValueError:
        raise RuntimeError(
            f"BENCH_SPEND_LIMIT_USD must be a number, got {raw!r}. "
            "Unset it for no limit."
        ) from None
    if not math.isfinite(limit) or limit <= 0:
        raise RuntimeError(
            f"BENCH_SPEND_LIMIT_USD must be a positive number, got {raw!r}. "
            "Unset it for no limit."
        )
    return limit


def _parse_data_policy(raw: str | None) -> str:
    """The validated per-boot data-handling policy, defaulting to standard.

    Refused loudly at boot, like the spend ceiling, and for a sharper
    reason: a typo silently falling back to standard would send prompts to
    training-eligible providers while the operator believed otherwise, and
    the run rows would record the fallback rather than the intent. Nothing
    about that failure is visible after the fact, so it has to be visible
    before the first request.
    """
    if not raw:
        return "standard"
    policy = raw.strip().lower()
    if policy not in DATA_POLICY_PREFS:
        allowed = ", ".join(sorted(DATA_POLICY_PREFS))
        raise RuntimeError(
            f"BENCH_DATA_POLICY must be one of {allowed}, got {raw!r}. "
            "Unset it for standard."
        )
    return policy


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail at boot, not on the first request. A bench with a missing key
    # would otherwise report every model as errored and look like an outage.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before starting the app."
        )
    # Per-boot spend ceiling, validated here next to the key check and
    # before any resource is allocated so a misconfigured limit fails fast
    # and loud like a missing key. Unset means no limit. The figure tracked
    # is what each result contributes through ceiling_cost: the platform's
    # billed charge when it reported one, the catalog estimate otherwise,
    # the same numbers the cards show. Accumulated as a plain float, which
    # one event loop makes safe without a lock. Only results unpriced by
    # BOTH routes leave the counter untouched, so an offline catalog no
    # longer means an uncounted run: a usage object still prices it.
    app.state.spend_limit_usd = _parse_spend_limit(
        os.environ.get("BENCH_SPEND_LIMIT_USD")
    )
    app.state.accumulated_spend_usd = 0.0
    # Data-handling routing, validated beside the ceiling and before any
    # request can be built. The resolved provider block is computed once
    # here rather than per request: it is boot-scoped by design, so a run
    # can never be sent under a policy other than the one its row records.
    app.state.data_policy = _parse_data_policy(os.environ.get("BENCH_DATA_POLICY"))
    app.state.provider_prefs = provider_preferences(app.state.data_policy)
    if app.state.data_policy != "standard":
        logger.info(
            "data policy %s: provider preferences %s",
            app.state.data_policy,
            app.state.provider_prefs,
        )
    # One shared client: connection pooling across the fan-out, and the
    # auth header lives in exactly one place. The explicit transport
    # exists to carry TCP keepalive options: extended-budget streams go
    # silent for minutes during hidden reasoning, NAT idle timers cull
    # flows that move no bytes, and the resulting deaths wore mixed
    # ReadError and stall signatures. OS-level probes keep those quiet
    # flows alive; see keepalive_socket_options in models.py.
    # trust_env stays at its default (on): an operator may legitimately
    # reach OpenRouter through a corporate proxy, and honoring HTTP(S)_PROXY
    # is the right behavior for the real app. The test harness is the one
    # place that must not inherit a developer proxy, and it opts out itself
    # (trust_env=False on its own clients, proxy vars scrubbed from the
    # browser subprocess) rather than the app degrading its own behavior.
    app.state.client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.AsyncHTTPTransport(socket_options=keepalive_socket_options()),
    )
    app.state.db = store.connect(os.environ.get("BENCH_DB", "./bench.db"))
    # One shared gate for every paid upstream call this process makes;
    # see MAX_CONCURRENT_UPSTREAM for why it exists.
    app.state.upstream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPSTREAM)
    # One catalog snapshot per boot feeds both pricing and the model
    # picker. Failure is tolerated: the bench must work offline, cost
    # renders as unavailable and the picker falls back to exact ids.
    app.state.app_sha = _app_sha()
    app.state.catalog = await fetch_catalog(app.state.client)
    app.state.catalog_digest = app.state.catalog.get("digest")
    app.state.prices = app.state.catalog["prices"]
    # When the prices a run was costed against were read. None on an
    # offline boot, where there is no snapshot to date: that absence is
    # itself the provenance, since those runs have no estimate either.
    app.state.catalog_snapshot_at = (
        datetime.now(UTC).isoformat() if app.state.catalog["fetched"] else None
    )
    # Per-model completion caps, derived once per boot so the budget
    # clamp is a dict lookup per request. Only published integer caps
    # participate. With the catalog fetched, an unknown cap means no clamp;
    # on an offline boot there are no caps at all and effective_budget
    # falls back to the conservative tier instead. See effective_budget.
    app.state.completion_limits = {
        m["id"]: m["max_completion_tokens"]
        for m in app.state.catalog["models"]
        if isinstance(m.get("max_completion_tokens"), int)
    }
    if not app.state.catalog["fetched"]:
        logger.warning(
            "OpenRouter catalog fetch failed; cost display and model "
            "search are unavailable this session"
        )
    yield
    await app.state.client.aclose()
    app.state.db.close()


# The three documentation routes are off. Nothing here consumes them: the
# only client is static/, which is written against these endpoints by
# hand, and the API surface is documented in README.md for humans. What
# they do provide is a machine-readable map of every route, parameter and
# bound on a server that holds a paid API key, served to anything that
# reaches the port, plus two pages of remotely-sourced script tags in the
# default configuration. Turning them off removes attack surface that
# exists only to serve a consumer this bench does not have.
app = FastAPI(
    title="LM Comparison Bench",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# The bench is a localhost tool holding a paid API key, which makes it a
# target for the two ways a browser gets turned against local servers.
# First, cross-site "simple" POSTs: a malicious page can fire fetch() at
# http://localhost:8000 with a text/plain body and no CORS preflight,
# and although it cannot read the response, each /compare call it lands
# spends real money upstream. Second, DNS rebinding: an attacker's
# hostname re-resolving to 127.0.0.1 makes the page same-origin with
# the bench, granting full read access. Requiring JSON bodies on POST
# forces cross-origin senders into a preflight nothing here answers,
# and rejecting non-local Host headers kills rebinding (the rebound
# page's requests carry the attacker's hostname in Host).
TRUSTED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def peer_is_local(client: Any) -> bool:
    """Whether an ASGI scope's client is a loopback peer or in-process.

    The Host check above defends browsers, not sockets: it is a header, so
    a non-browser client that reaches the port can simply send
    "Host: localhost". If the server is ever bound beyond loopback (a
    --host 0.0.0.0 typo, a container publishing the port), that is a
    remote caller spending the key. This is the socket-level companion:
    the peer address itself must be loopback.

    Not quite the raw socket: uvicorn's proxy-headers support rewrites
    scope["client"] from X-Forwarded-For, but only for peers listed in
    FORWARDED_ALLOW_IPS, which defaults to 127.0.0.1. So in the case this
    guard is for, a genuinely remote peer reaching a wrongly-bound server,
    that peer is not trusted to rewrite anything and its real address is
    what gets checked. Widening FORWARDED_ALLOW_IPS hands the header that
    trust deliberately, and would weaken this check along with it.

    In-process transports have no socket. Starlette's TestClient presents
    the synthetic peer ("testclient", 50000), and other harnesses present
    None, so a client that does not parse as an IP address is treated as
    in-process and allowed: there is no network peer to be remote. Anything
    that IS a real IP is held to loopback strictly, which is the case that
    matters, since only a real socket can carry a remote attacker. The
    browser suite runs a real uvicorn on 127.0.0.1 and so exercises the
    strict path end to end.
    """
    if not client:
        return True
    try:
        host = client[0]
    except (TypeError, IndexError):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not an IP at all: an in-process transport's synthetic name.
        return True


def host_header_name(host: str) -> str:
    """The hostname part of a Host header value, port stripped.

    Handles bracketed IPv6 ([::1]:8000) explicitly: a naive rsplit on
    ":" would chop the address itself. A bare IPv6 with no brackets has
    multiple colons and falls through unchanged.
    """
    host = host.strip().lower()
    if host.startswith("[") and "]" in host:
        return host[1 : host.index("]")]
    if host.count(":") == 1:
        return host.rsplit(":", 1)[0]
    return host


# A strict content security policy, enforced on every response. The audit
# behind it: index.html has no inline style attributes and no inline
# scripts (the pre-paint theme script is a same-origin file now), the
# scripts create no style or script elements, and the fonts, the favicon,
# and every fetch and SSE endpoint are same-origin. So nothing needs
# 'unsafe-inline' or a remote origin. default-src 'none' denies by default;
# each source is opened only to 'self'. base-uri and form-action are locked
# down so an injected <base> or form cannot exfiltrate. frame-ancestors
# 'none' carries the old anti-framing guarantee, with X-Frame-Options: DENY
# as the belt for browsers that predate frame-ancestors.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


class LocalOnlyGuard:
    """Reject requests that could only come from a hostile browser page.

    Pure ASGI rather than BaseHTTPMiddleware: the streaming endpoint
    relies on generator cancellation to persist aborted runs, and a
    wrapping middleware layer is one more thing that could interfere
    with disconnect propagation.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Security headers on every HTTP response. A hostile page cannot read
        # a cross-origin response, but it can frame the real localhost UI and
        # redress a Run click into paid work; frame-ancestors and
        # X-Frame-Options refuse the frame, and the full CSP denies any
        # inline or cross-origin script, style, or connection the page was
        # never meant to make. Blanket application is deliberate: no response
        # here is meant to be embedded or to load foreign resources. The
        # headers are added on http.response.start only, with no buffering of
        # the body, so SSE streaming and the generator cancellation the
        # streaming endpoint depends on stay untouched, which is the reason
        # this guard is pure ASGI.
        async def framed_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-frame-options"] = "DENY"
                headers["content-security-policy"] = CONTENT_SECURITY_POLICY
                # Only when the response has not already chosen its own
                # caching. The conditional is load-bearing rather than
                # defensive: the favicon deliberately declares a year of
                # immutable caching, and a second Cache-Control on that
                # response would be a defect, not a stricter rule.
                #
                # Which directive depends on what the response carries, and
                # that split is the correction. no-cache means "revalidate
                # before reuse", NOT "do not store": a no-cache response is
                # allowed to sit on disk indefinitely. For assets that is
                # exactly right, and it is what closed the stale-module skew,
                # since with the ETags already present the common case is a
                # 304. For a dynamic response it was wrong. Run details,
                # group details and compare responses carry full prompts and
                # full model answers, and permitting a browser or an
                # intermediary to store those was a privacy directive that
                # said the opposite of what this application promises about
                # where prompts go.
                #
                # So assets revalidate and everything else is not stored at
                # all. private is belt to no-store's braces: no-store already
                # forbids storage, and private states the intent for anything
                # that honors one and not the other.
                if "cache-control" not in headers:
                    path = scope.get("path", "")
                    asset = path == "/" or path.startswith("/static/")
                    headers["cache-control"] = (
                        "no-cache" if asset else "private, no-store"
                    )
            await send(message)

        # The socket-level check runs first, before anything a caller
        # controls: Host is a header a remote client can forge, the peer
        # address is not. See peer_is_local for the in-process transport
        # carve-out and why it is safe.
        if not peer_is_local(scope.get("client")):
            response = JSONResponse(
                {"detail": "the bench only answers to loopback clients"},
                status_code=403,
            )
            await response(scope, receive, framed_send)
            return

        headers = Headers(scope=scope)
        # A missing Host header (raw HTTP/1.0 clients) fails closed.
        host = host_header_name(headers.get("host", ""))
        if host not in TRUSTED_HOSTS:
            response = JSONResponse(
                {"detail": "the bench only answers to localhost"},
                status_code=403,
            )
            await response(scope, receive, framed_send)
            return
        # Every POST must be application/json, bodyless included: POST
        # is the one method a browser fires cross-site without a CORS
        # preflight, and the old bodyless exemption was a reproduced
        # hole (a no-body cross-site POST /groups returned 201).
        # Requiring the JSON content type forces the preflight nothing
        # here answers. GET and HEAD stay exempt as reads; DELETE needs
        # no gate because it is never a "simple" cross-site method, so
        # the preflight already guards it.
        if scope["method"] == "POST" and not headers.get("content-type", "").startswith(
            "application/json"
        ):
            response = JSONResponse(
                {"detail": "POST bodies must be application/json"},
                status_code=415,
            )
            await response(scope, receive, framed_send)
            return
        await self.app(scope, receive, framed_send)


app.add_middleware(LocalOnlyGuard)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Matches the src and href of every same-origin asset the index references.
# Deliberately narrow: only the /static prefix, and only inside a double
# quoted attribute, which is how every URL in that file is written.
_STATIC_ATTR = re.compile(r'((?:src|href)=")(/static/[^"]*)(")')


def static_rev(directory: Path) -> str:
    """A short hash over every static file's path and bytes.

    Content-derived rather than taken from the app commit: it works in a
    checkout with no git, it changes exactly when an asset changes rather
    than on every commit, and it is identical for every request of one
    boot. Paths are mixed in so a rename counts as a change even when the
    bytes do not, and the walk is sorted so the digest does not inherit
    directory iteration order.
    """
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def versioned_index(html: str, rev: str) -> str:
    """The index with ?v=<rev> appended to every /static URL it references.

    A mechanical rewrite at boot rather than a placeholder token in the
    committed file, so opening static/index.html straight off disk keeps
    working. StaticFiles ignores query parameters it does not recognize,
    so the versioned URLs need no route changes.

    The point is not to defeat caching, which Cache-Control already does.
    It is that a changed asset gets a URL the browser has never seen, so
    even a cache that ignored every header cannot serve the old bytes for
    it. Belt and braces on the same failure, because the failure was a
    page silently running half of two releases.
    """

    def add_rev(match: re.Match[str]) -> str:
        prefix, url, suffix = match.groups()
        # Append rather than assume there is no query already, so a future
        # URL that carries one does not become nonsense.
        separator = "&" if "?" in url else "?"
        return f"{prefix}{url}{separator}v={rev}{suffix}"

    return _STATIC_ATTR.sub(add_rev, html)


# Computed at import rather than in the lifespan, because it is a pure
# function of files already on disk: no env, no network, nothing that can
# be absent. That also means the index route cannot meet a half-built
# app.state, which is the failure mode this hotfix exists to remove rather
# than reintroduce somewhere else.
STATIC_REV = static_rev(STATIC_DIR)
INDEX_DOCUMENT = versioned_index(INDEX_HTML.read_text(encoding="utf-8"), STATIC_REV)

# Serve the static assets (vendored fonts now; the split-out stylesheet
# and scripts later) from one mount. LocalOnlyGuard wraps the whole app
# and GET is exempt from the JSON-POST rule, so the security posture is
# unchanged: these are read-only assets on a localhost-only server.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Hand-written bolt in the VOLT accent, sized to stay legible at 16px.
# Served inline because browsers request /favicon.ico unprompted and
# every miss was a 404 line of noise in the run logs.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<path d="M9.5 1 3 9.5h4L5.5 15 13 6.5H9L9.5 1z" fill="#38bdd8"/>'
    "</svg>"
)


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    # From memory, not FileResponse: the document served is the rewritten
    # one, with every asset URL carrying this boot's rev. Serving the file
    # verbatim would hand the browser the unversioned URLs and lose half
    # the fix.
    return HTMLResponse(INDEX_DOCUMENT)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    # Immutable is honest: the icon only changes when the app does.
    return Response(
        FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def run_provenance() -> dict[str, Any]:
    """The run-level facts of this boot, stamped onto every run row.

    Read from app.state at call time rather than captured once, so a test
    that rebuilds state between clients sees its own values. data_policy is
    filled by the privacy workstream; it is listed here so every writer
    already carries the key.
    """
    return {
        "app_sha": getattr(app.state, "app_sha", None),
        "catalog_snapshot_at": getattr(app.state, "catalog_snapshot_at", None),
        "data_policy": getattr(app.state, "data_policy", None),
        # None on an offline boot, where there were no catalog bytes to
        # hash. That None is a fact about the boot rather than a gap.
        "catalog_digest": getattr(app.state, "catalog_digest", None),
    }


def cost_usd(result: dict[str, Any], prices: dict[str, Any]) -> float | None:
    """Cost of one result, or None when tokens or pricing are unknown.

    isinstance rather than a None check: a provider reporting token
    counts as strings would otherwise raise in the multiplication and
    sink the whole request after the upstream calls already happened.
    """
    price = prices.get(result["model"])
    if (
        price is None
        or not isinstance(result["prompt_tokens"], (int, float))
        or not isinstance(result["completion_tokens"], (int, float))
    ):
        return None
    total = (
        result["prompt_tokens"] * price["prompt"]
        + result["completion_tokens"] * price["completion"]
    )
    # The catalog is the primary guard: fetch_catalog rejects non-finite
    # prices at ingestion, so a NaN can only reach here if a price was set
    # past that path. This is the belt: an unpriceable total degrades to
    # None, never a NaN that would poison accumulated spend and silently
    # disable the ceiling.
    if not math.isfinite(total):
        return None
    return float(total)


def format_usd(value: float) -> str:
    """A dollar figure that stays truthful at the ceiling's native scale.

    Priced runs here cost fractions of a cent, so the cards' four decimals
    collapse a real sub-cent limit to $0.0000 and the 402 would misreport
    the operator's own ceiling as zero. Six decimals cover that domain;
    trailing zeros past the cents place are trimmed so dollar-scale limits
    still read as $1.50, not $1.500000.
    """
    whole, frac = f"{value:.6f}".split(".")
    frac = frac[:2] + frac[2:].rstrip("0")
    return f"${whole}.{frac}"


def spend_ceiling_reached() -> bool:
    """True when an active ceiling has been reached by accumulated spend.

    The shared predicate behind the entry check and the post-admission
    recheck. Results the bench could price neither from the platform's
    billed figure nor from the catalog never moved the counter, so this
    bounds known spend, not all spend.
    """
    limit = app.state.spend_limit_usd
    if limit is None:
        return False
    return bool(app.state.accumulated_spend_usd >= limit)


def enforce_spend_limit() -> None:
    """Refuse a run at the boundary once recorded spend hits the ceiling.

    Checked at endpoint entry, before the semaphore and before any
    upstream call, so a refusal costs nothing. Money already in flight
    is never interrupted. The 402 names both figures so the operator
    knows how far over the intent they are. A second recheck runs after
    admission (spend_refusal_result) to close the gap where runs admitted
    below the limit would all execute once an earlier one crossed it.
    """
    if spend_ceiling_reached():
        raise HTTPException(
            402,
            "spend ceiling reached: recorded "
            f"{format_usd(app.state.accumulated_spend_usd)} of "
            f"{format_usd(app.state.spend_limit_usd)} limit "
            "(BENCH_SPEND_LIMIT_USD); unpriced runs do not count against it",
        )


def record_spend(cost: float | None) -> None:
    """Add one priced result to the accumulated spend.

    The caller chooses between billed cost and the estimate (ceiling_cost);
    this only accumulates. None means the result could not be priced by
    either route (offline catalog, missing usage), and those never count
    against the ceiling by design. A
    non-finite cost is refused here too: cost_usd already screens it out,
    but the accumulator is the invariant's last line, and a single NaN
    summed in would make the ceiling comparison permanently false.
    """
    if cost is not None and math.isfinite(cost):
        app.state.accumulated_spend_usd += cost


def ceiling_cost(result: dict[str, Any]) -> float | None:
    """What one result contributes to the accumulated spend.

    Billed cost when the platform reported one, the catalog estimate
    otherwise. The ceiling is advisory, a guard against a runaway session
    rather than an accounting ledger, and advising from what was actually
    charged beats advising from catalog arithmetic over reported token
    counts. Both figures stay on the row; only one of them counts here, so
    a result can never be charged to the ceiling twice.

    Both fields already passed the money rule at ingestion (as_money for
    billed, the finiteness check in cost_usd for the estimate). Re-applying
    as_money here is the accumulator's own last line rather than a
    duplicate: this reads a plain dict that a persistence round trip or a
    future writer could reach, and one poisoned value summed in would make
    the ceiling comparison permanently false. None means the result is
    unpriced by both routes, which by design never moves the counter.
    """
    billed = as_money(result.get("billed_cost_usd"))
    return billed if billed is not None else as_money(result.get("cost_usd"))


def spend_refusal_result(model: str, max_tokens: int) -> dict[str, Any]:
    """A synthetic result for a run refused at the post-admission recheck.

    Shaped like stream_model's done() result, which is a superset of
    run_model's and which both endpoints' response models accept. Every
    metric and text field is None because no upstream call happened. The
    error reuses format_usd and states plainly that the run was refused
    before reaching upstream, so the persisted row is unambiguous that no
    money moved.

    spend_refused marks the frame so the client can tell a working control
    from a broken one. Both arrive as run_id null, but the meanings are
    opposite: a refusal deliberately persists nothing, while a genuine
    post-spend persistence failure means money moved and history lost it.
    Without the marker the UI described the ceiling as a failure. An extra
    field is additive by the client contract (unknown fields are ignored),
    and it is a marker rather than an error-string match because the error
    text is prose that may be reworded.

    The marker rides the streaming frame only. /compare serializes through
    ModelResult, which declares no such field and therefore drops it; that
    is deliberate. Surfacing it there would either need a schema column to
    persist it (out of scope this phase) or would report a misleading false
    on every historical row, and the frontend, the only consumer that acts
    on it, uses the streaming endpoint.
    """
    return {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "spend_refused": True,
        "error": (
            "run refused before reaching upstream: recorded spend "
            f"{format_usd(app.state.accumulated_spend_usd)} reached the "
            f"{format_usd(app.state.spend_limit_usd)} ceiling "
            "(BENCH_SPEND_LIMIT_USD); no upstream call was made"
        ),
        "cost_usd": None,
        "ttft_ms": None,
        "max_tokens": max_tokens,
        "generation_id": None,
        "finish_reason": None,
        # Nothing was charged because nothing was sent. Present and None
        # rather than absent so a reader of this dict sees the same fields
        # a real result carries. request_json is the deliberate exception:
        # both client functions always set it, and here there is no payload
        # to record because none was built.
        "billed_cost_usd": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "provider": None,
        "native_finish_reason": None,
    }


def _excerpt(text: str, limit: int = 60) -> str:
    """A bounded, single-line excerpt of a prompt for an error detail:
    enough to name a conflict without dumping a full prompt into a 409.
    """
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[:limit] + "..."


def _control_conflicts(
    established: dict[str, Any], requested: dict[str, Any]
) -> list[str]:
    """The control names on which two experiments disagree, sorted.

    A control set on one side and absent on the other is a disagreement,
    not a partial match: absence is a choice under rule one (the provider's
    own default) and is exactly as meaningful as a value.
    """
    return sorted(
        name
        for name in set(established) | set(requested)
        if established.get(name) != requested.get(name)
    )


def _enforce_manifest(
    manifest: dict[str, Any],
    budget: str,
    model: str | None,
    position: int | None,
    models: list[str] | None,
) -> None:
    """Reject a member that does not match what its group declared.

    Membership, position and budget, checked against the declaration the
    group wrote before any call. Called from both endpoints at entry, so
    every refusal here is pre-spend.

    NULL means UNKNOWN in this function, which is the opposite of what it
    means for params, and the contrast is deliberate enough to state twice.
    A NULL params_json is the affirmative fact that no control was set,
    because controls arrived with the concept and every group predating
    them genuinely set none; that is why the params check refuses a
    controls-carrying member of a NULL-params group. A NULL models_json or
    NULL budget is different: those groups were created before anything
    recorded a lineup or a tier, so NULL is silence rather than a claim of
    emptiness. Enforcing against silence would refuse correct work on every
    legacy group, so a None declaration skips its check entirely.
    """
    declared_models = manifest["models"]
    if declared_models is not None:
        if models is not None:
            # The batch surface declares its whole lineup in one array, so
            # order is part of the claim: the same models in a different
            # order is a different side-by-side comparison.
            if models != declared_models:
                raise HTTPException(
                    409,
                    "this group's lineup is "
                    f"{', '.join(declared_models)}; the batch sent "
                    f"{', '.join(models)}. To add or rerun individual "
                    "members of a declared comparison, use /compare/stream, "
                    "one request per model",
                )
        elif model is not None:
            if model not in declared_models:
                raise HTTPException(
                    409,
                    f"model {model} is not in this group's declared lineup "
                    f"({', '.join(declared_models)})",
                )
            # Position is checked only when the caller declared one. A
            # client that sends no position is not claiming a slot, and
            # inventing one for it would refuse a run for a claim it never
            # made.
            if position is not None and declared_models[position : position + 1] != [
                model
            ]:
                at = declared_models.index(model)
                raise HTTPException(
                    409,
                    f"model {model} is declared at position {at} in this "
                    f"group's lineup, got {position}",
                )
    declared_budget = manifest["budget"]
    if declared_budget is not None and declared_budget != budget:
        raise HTTPException(
            409,
            f"this group declared the {declared_budget} budget; this run "
            f"asks for {budget}. A comparison holds one budget across its "
            "members",
        )


def enforce_group_experiment(
    prompt: str,
    params: dict[str, Any],
    group_id: int | None,
    budget: str,
    model: str | None = None,
    position: int | None = None,
    models: list[str] | None = None,
) -> None:
    """Reject a run whose group already holds a different experiment.

    One prompt per group became one experiment per group, because holding
    the prompt constant while sampling varies is not a comparison anyone can
    defend. Prompt and controls are checked together, at endpoint entry,
    before the semaphore and any upstream call, so a mismatch is a 409 that
    spends nothing and never reaches the post-spend degrade path (link
    resolution stays degrade-only). An unknown group, and a group with
    neither a declared prompt nor a member, accepts any prompt.

    The controls half needs no derive-from-member fallback and deliberately
    has none. A NULL params_json is not missing information: every pre-H
    group predates the controls, so NULL is the affirmative fact that none
    was set, and a legacy client that sends no controls matches it exactly.
    What that buys is stricter than a fallback would be. A group recorded
    with no controls now rejects a member that carries one, which is the
    case a fallback would have waved through and which would have produced
    a group whose record understated what it sent.

    Residual, now confined to legacy groups: a group that declared its
    prompt at creation is checked against a value that existed before any
    member was sent, so two concurrent first members cannot both see an
    empty group. Every group the current frontend creates declares one. The
    old race survives only for groups created before that column existed,
    or by a client that omits the prompt: there group_prompt still derives
    from the first member and returns None for both racers. Those are
    finite and historical, single-user localhost makes the window a
    non-path anyway, and rejecting after spend is forbidden by the
    fault-boundary invariant, so the remainder is accepted rather than
    closed with a lock.
    """
    if group_id is None:
        return
    established = store.group_prompt(app.state.db, group_id)
    if established is not None and established != prompt:
        raise HTTPException(
            409,
            "prompt does not match this group's established prompt "
            f"({_excerpt(established)!r}); a group holds one prompt across "
            "its runs",
        )
    group_controls = store.group_params(app.state.db, group_id)
    _enforce_manifest(
        store.group_manifest(app.state.db, group_id), budget, model, position, models
    )
    conflicts = _control_conflicts(group_controls, params)
    if conflicts:
        named = ", ".join(conflicts)
        if not group_controls:
            # The legacy shape, and the one where the generic message would
            # be useless: nothing "does not match" because there is nothing
            # to match against. Every pre-H group lands here, so the detail
            # has to say what the group is and what to do instead, not just
            # that the request was refused.
            raise HTTPException(
                409,
                "this group records no controls, so it cannot hold a run "
                f"that sets them ({named}); start a new comparison to run "
                "with controls",
            )
        raise HTTPException(
            409,
            f"controls do not match this group's experiment ({named}); "
            "a group holds one controls set across its runs",
        )


def effective_budget(budget: str, model: str) -> int:
    """The requested budget clamped to the model's published completion
    cap, or to the conservative tier when no cap is known. Sending a budget
    above a model's cap is a hard 400 from some providers; clamping turns
    that failure into the best the model can do. Lives here rather than in
    models.py because the boot catalog is app state.

    On an offline boot there are no published caps at all, and the old code
    sent the full requested tier, dropping both protections at the worst
    moment: an offline catalog is also an unpriced one, so an extended run
    there is unverified against the provider AND invisible to the spend
    ceiling. Falling back to the standard tier bounds the blast radius to
    the tier the bench defaults to anyway. The effective budget is recorded
    and shown per run, so the card carries the truth about what was sent.

    Deliberately scoped to the offline boot rather than to every unknown
    cap: with the catalog fetched, most models simply do not publish one,
    and clamping those would quietly disable the extended tier for them.
    A fetched catalog keeps today's behavior exactly.
    """
    requested = BUDGET_TOKENS[budget]
    limit: int | None = app.state.completion_limits.get(model)
    if limit is None and not app.state.catalog["fetched"]:
        return min(requested, UNKNOWN_CAP_BUDGET)
    return min(requested, limit) if limit is not None else requested


def resolve_links(
    db: sqlite3.Connection, prompt_id: int | None, group_id: int | None
) -> tuple[int | None, int | None]:
    """Degrade stale prompt or group links to None instead of erroring.

    By the time links are checked the upstream calls already happened
    and prompt_text is the source of truth, so a deleted or bogus id
    must not sink the run.
    """
    if prompt_id is not None and store.get_prompt(db, prompt_id) is None:
        prompt_id = None
    if group_id is not None and not store.group_exists(db, group_id):
        group_id = None
    return prompt_id, group_id


def request_controls(params: ExperimentParams | None) -> dict[str, Any]:
    """The controls a request actually set, as a plain dict.

    exclude_none is rule one at the boundary: every field defaults to None,
    so what survives is exactly what the caller set and nothing else. This
    is the single conversion from the Pydantic edge to the plain dicts the
    internals pass, and it feeds three things that must agree, the payload,
    the stored record and the integrity check, so there is one of it.
    """
    return params.model_dump(exclude_none=True) if params is not None else {}


def request_provider_prefs(controls: dict[str, Any]) -> dict[str, Any]:
    """The provider object for one request: the boot data policy, plus this
    request's routing mode when it chose one.

    Returns the boot-computed object unchanged when no routing was set, so a
    blank controls set does not merely produce an equal provider block, it
    produces the same one the code passed before this phase existed. That is
    what the byte-identical payload tombstone rests on.

    The privacy policy comes from boot and the routing from the request, and
    they meet in provider_preferences rather than here, so there is exactly
    one place that knows how the two merge.
    """
    routing = controls.get("routing")
    if routing is None:
        # Annotated because app.state is untyped; the boot value is built by
        # provider_preferences and is this shape by construction.
        boot_prefs: dict[str, Any] = app.state.provider_prefs
        return boot_prefs
    return provider_preferences(app.state.data_policy, routing)


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> dict[str, Any]:
    enforce_spend_limit()
    controls = request_controls(request.params)
    # request.budget, not effective_budget: the tier is the declaration, and
    # the clamp is a per-model consequence of it. Two models with different
    # published caps legitimately produce different token numbers inside one
    # extended comparison, and refusing that would refuse a correct run.
    enforce_group_experiment(
        request.prompt,
        controls,
        request.group_id,
        request.budget,
        models=request.models,
    )

    async def limited(model: str) -> dict[str, Any]:
        # One slot per model inside the fan-out, not one around the
        # batch: the cap is on simultaneous paid calls wherever they
        # come from, and concurrent /compare and stream requests share
        # the same gate. Acquiring before run_model keeps the clocks
        # honest for free: its latency clock starts internally, after
        # the slot is already held, so a queued model never reports
        # queue wait as model latency.
        budget = effective_budget(request.budget, model)
        async with app.state.upstream_semaphore:
            # Recheck under the held slot, before run_model spends.
            #
            # Settlement now happens below, inside this same held slot, so
            # this recheck observes every result that has already settled,
            # including earlier members of this very batch. That is the
            # property the old comment lacked. It reasoned correctly about
            # one batch and was never re-derived for N: settlement used to
            # run after gather, so a fast member released its slot with
            # nothing recorded, and a model from a concurrent batch took
            # that slot and rechecked against a counter that had not moved.
            # Eight concurrent five-model batches against a ceiling worth
            # half a result put 28 calls upstream where the documented
            # bound promised about five.
            #
            # With per-result settlement the bound is real and derivable: a
            # freed slot implies a recorded settlement, so once accumulated
            # spend crosses the ceiling, every subsequent slot acquisition
            # sees it and refuses. Only the calls already executing when
            # the ceiling tripped can overshoot, and there are at most
            # MAX_CONCURRENT_UPSTREAM of those by construction.
            #
            # On refusal return a synthetic result shaped like run_model's,
            # error set, with no upstream call. The batch persists as usual
            # with the refusal row included: honest history for a cut-short
            # run.
            if spend_ceiling_reached():
                return spend_refusal_result(model, budget)
            result = await run_model(
                request.prompt,
                model,
                app.state.client,
                max_tokens=budget,
                provider_prefs=request_provider_prefs(controls),
                controls=controls,
            )
            # Settle before the slot releases. This block is post-spend, so
            # it carries its own fault boundary: the upstream call has
            # already happened and nothing here may turn a paid result into
            # an error. A failure pricing or recording degrades this
            # result's cost to None, which contributes nothing to the
            # ceiling, exactly as an unpriced result already does.
            try:
                result["cost_usd"] = cost_usd(result, app.state.prices)
                record_spend(ceiling_cost(result))
            except Exception:
                logger.exception("settlement failed for %s", model)
                result["cost_usd"] = None
            return result

    # gather preserves input order, which the frontend relies on to map
    # result columns by position. run_model never raises, so no
    # return_exceptions handling is needed here, and limited() guarantees
    # cost_usd is present on everything it returns, so the response model's
    # required key needs no seeding pass.
    results = await asyncio.gather(*(limited(m) for m in request.models))
    # position comes from array order here: gather preserves it, and for
    # the batch endpoint the caller's array IS the declared layout. Pure
    # assignment, so it stays outside the fault boundary below.
    for index, result in enumerate(results):
        result["position"] = index
    run_id = None
    # The invariant: after money is spent, no code path may convert
    # results into an error response. Everything between "upstream
    # results exist" and "response returned" (cost, link resolution,
    # persistence) sits inside this one boundary; on any failure the
    # results go back intact with run_id null and the links dropped.
    try:
        prompt_id, group_id = resolve_links(
            app.state.db, request.prompt_id, request.group_id
        )
        run_id = store.save_run(
            app.state.db,
            request.prompt,
            list(results),
            prompt_id,
            group_id,
            run_provenance(),
        )
    except Exception:
        logger.exception("post-upstream processing failed for /compare")
        run_id = None
    return {"results": list(results), "run_id": run_id}


@app.post("/compare/stream")
async def compare_stream(request: StreamCompareRequest) -> StreamingResponse:
    # At entry, before the generator runs and so before the semaphore or
    # any upstream call: a refusal must spend nothing.
    enforce_spend_limit()
    controls = request_controls(request.params)
    # See /compare: the requested tier is what the group declared, never the
    # clamped token count computed on the next line.
    enforce_group_experiment(
        request.prompt,
        controls,
        request.group_id,
        request.budget,
        model=request.model,
        position=request.position,
    )
    max_tokens = effective_budget(request.budget, request.model)

    async def events() -> AsyncIterator[str]:
        # Server-side observation of the stream, enough to reconstruct
        # a result if the client disconnects before the done event: a
        # stream the user watched happen is still history.
        parts: list[str] = []
        first_delta_ms: float | None = None
        handled = False
        started = False
        acquired = False
        start = 0.0
        # Injected state, not a protocol change: stream_model writes each
        # fact here the moment it knows it, so the disconnect path below can
        # persist what the server saw. A run cut short never reaches the
        # done event, and it is the run whose billing is least knowable
        # locally, so it is the row most in need of an id to reconcile
        # against and of a record of what it asked for.
        holder: dict[str, Any] = {}

        def release_slot() -> None:
            # Idempotent so the done branch and the finally below can
            # both call it: the slot must never be returned twice, and
            # a cancellation while still queued has nothing to return.
            nonlocal acquired
            if acquired:
                acquired = False
                app.state.upstream_semaphore.release()

        try:
            # A saturated semaphore means this run waits for a slot.
            # Tell the client so its column reads "queued" instead of
            # pretending the model is already thinking; locked() is true
            # exactly when no slot is free. Emitted before any clock so
            # the wait pollutes no metric.
            if app.state.upstream_semaphore.locked():
                yield "data: " + json.dumps({"type": "queued"}) + "\n\n"
            # The slot covers the upstream exchange only, and it is
            # acquired before any clock starts: both this generator's
            # clock and stream_model's internal latency and ttft clocks
            # begin after the slot is held, so a queued run never
            # reports its queue wait as model time. Saturation queues
            # quietly, with no acquisition timeout.
            await app.state.upstream_semaphore.acquire()
            acquired = True
            # Recheck the ceiling now that a slot is held, before started is
            # set, before the clock, and before the started frame. The entry
            # check admits every queued run below the limit; without this
            # recheck an earlier run crossing the ceiling would not stop the
            # ones already admitted, so overshoot would be bounded by lineup
            # size, not by the semaphore. On refusal return the slot, emit
            # one done frame carrying the synthetic refusal result with
            # run_id null, and persist nothing: the refusal happened before
            # any upstream call, exactly like a queued cancel, so there is
            # nothing truthful to record. started stays false so the
            # finally's abort-persist path never fires.
            if spend_ceiling_reached():
                release_slot()
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "result": spend_refusal_result(request.model, max_tokens),
                            "run_id": None,
                        }
                    )
                    + "\n\n"
                )
                return
            started = True
            # Start the clock before emitting started, not after. The
            # frame and the clock mark the same instant (the sub-ms cost
            # of yielding one small frame on localhost is noise against a
            # network TTFT), and starting first keeps start valid for the
            # finally: a disconnect suspended at this yield would otherwise
            # run the abort path with start still 0.0 and persist a garbage
            # latency. The client still resets its own clock on the frame.
            start = time.perf_counter()
            yield "data: " + json.dumps({"type": "started"}) + "\n\n"
            async for event in stream_model(
                request.prompt,
                request.model,
                app.state.client,
                max_tokens=max_tokens,
                holder=holder,
                provider_prefs=request_provider_prefs(controls),
                controls=controls,
            ):
                if event["type"] != "done":
                    if first_delta_ms is None:
                        first_delta_ms = round((time.perf_counter() - start) * 1000, 1)
                    parts.append(event["text"])
                    yield "data: " + json.dumps(event) + "\n\n"
                    continue
                handled = True
                # The done event means the upstream exchange is over;
                # persistence must not sit on a paid-call slot.
                release_slot()
                result = event["result"]
                result["cost_usd"] = None
                result["position"] = request.position
                run_id = None
                # Same invariant as /compare, and stricter here: the
                # deltas are already on the wire, so after money is
                # spent no code path may convert the result into a
                # broken stream tail. Cost, link resolution and
                # persistence all sit inside this one boundary; any
                # failure degrades to run_id null with links dropped.
                try:
                    result["cost_usd"] = cost_usd(result, app.state.prices)
                    record_spend(ceiling_cost(result))
                    prompt_id, group_id = resolve_links(
                        app.state.db, request.prompt_id, request.group_id
                    )
                    run_id = store.save_run(
                        app.state.db,
                        request.prompt,
                        [result],
                        prompt_id,
                        group_id,
                        run_provenance(),
                    )
                except Exception:
                    logger.exception(
                        "post-stream processing failed for /compare/stream"
                    )
                    run_id = None
                yield (
                    "data: "
                    + json.dumps({"type": "done", "result": result, "run_id": run_id})
                    + "\n\n"
                )
        finally:
            release_slot()
            # A client disconnect cancels this generator at a yield
            # before the done branch ever runs. Persist what the server
            # saw (no awaits or yields are legal here, sqlite is sync,
            # so this is safe during unwinding) rather than silently
            # dropping a run whose deltas already reached the browser.
            # A run cancelled while still queued never reached upstream
            # and spent nothing, so there is nothing truthful to record.
            if started and not handled:
                aborted = {
                    "model": request.model,
                    "response_text": "".join(parts) or None,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": "stream aborted before completion",
                    "cost_usd": None,
                    "ttft_ms": first_delta_ms,
                    "max_tokens": max_tokens,
                    "position": request.position,
                    # Plain dict lookups with defaults: no await, no
                    # attribute access on a half-torn-down object, and no
                    # way to raise during generator unwinding. Without them
                    # the runs with the least knowable billing (cut short
                    # mid-stream, so cancellation may or may not have
                    # stopped the charge) were the only ones with no id to
                    # reconcile against and no record of what was sent,
                    # which made request_json absent from exactly the rows
                    # the reconcile pass exists to visit.
                    "generation_id": holder.get("generation_id"),
                    "request_json": holder.get("request_json"),
                    "provider": holder.get("provider"),
                    # finish_reason and native_finish_reason stay absent on
                    # purpose: this run did not finish, so any value here
                    # would be a claim the server cannot make.
                }
                try:
                    prompt_id, group_id = resolve_links(
                        app.state.db, request.prompt_id, request.group_id
                    )
                    store.save_run(
                        app.state.db,
                        request.prompt,
                        [aborted],
                        prompt_id,
                        group_id,
                        run_provenance(),
                    )
                except Exception:
                    logger.exception("failed to persist aborted streamed run")

    return StreamingResponse(events(), media_type="text/event-stream")


def ensure_rowid(value: int) -> None:
    """404 for ids outside SQLite's rowid range.

    Nothing outside the range can exist, so it reads as absence, the
    same answer an in-range miss gets; before this check the overflow
    surfaced from the sqlite bind as a 500.
    """
    if not 1 <= value <= MAX_SQLITE_ROWID:
        raise HTTPException(404, "no such id")


# The empty JSON object is load-bearing: a bodyless POST needs no
# content type, and the guard middleware keys on application/json to
# force hostile cross-site senders into a CORS preflight.
@app.post("/groups", response_model=GroupCreated, status_code=201)
async def create_group(body: GroupCreate) -> dict[str, Any]:
    return {
        "id": store.create_group(
            app.state.db,
            body.prompt,
            body.models,
            request_controls(body.params) or None,
            body.budget,
        )
    }


@app.get("/groups/{group_id}", response_model=GroupDetail)
async def group_detail(group_id: int) -> dict[str, Any]:
    ensure_rowid(group_id)
    group = store.get_group(app.state.db, group_id)
    if group is None:
        raise HTTPException(404, "no such group")
    return group


@app.get("/models", response_model=CatalogResponse)
async def get_models() -> dict[str, Any]:
    return {
        "models": app.state.catalog["models"],
        "fetched": app.state.catalog["fetched"],
        "data_policy": app.state.data_policy,
    }


@app.get("/prompts", response_model=PromptList)
async def get_prompts() -> dict[str, Any]:
    return {"prompts": store.list_prompts(app.state.db)}


@app.post("/prompts", response_model=Prompt, status_code=201)
async def create_prompt(body: PromptCreate) -> dict[str, Any]:
    try:
        return store.save_prompt(app.state.db, body.name, body.text)
    except sqlite3.IntegrityError:
        # from None: the duplicate name is an expected outcome being
        # translated to a 409, not an error in handling the error.
        raise HTTPException(
            409, f"a prompt named {body.name!r} already exists"
        ) from None


@app.delete("/prompts/{prompt_id}", status_code=204)
async def remove_prompt(prompt_id: int) -> Response:
    ensure_rowid(prompt_id)
    if not store.delete_prompt(app.state.db, prompt_id):
        raise HTTPException(404, "no such prompt")
    return Response(status_code=204)


@app.get("/runs", response_model=RunList)
async def get_runs(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    # Bounded so history stays cheap as bench.db grows. The 500 ceiling
    # keeps the id lists list_runs binds comfortably under sqlite's
    # variable limit.
    runs = store.list_runs(app.state.db, limit)
    # Append a marker only when a cut happened, so API consumers can tell
    # a short prompt from a truncated one.
    for run in runs:
        if len(run["prompt_text"]) > 80:
            run["prompt_text"] = run["prompt_text"][:80] + "..."
    return {"runs": runs}


@app.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: int) -> dict[str, Any]:
    ensure_rowid(run_id)
    run = store.get_run(app.state.db, run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return run
