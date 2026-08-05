"""FastAPI boundary. Pydantic models live here only; internals use plain dicts."""

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import random
import re
import secrets
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
from bench.datasets import DatasetError, parse_dataset
from bench.experiments import plan_trials, seed_for_repeat
from bench.models import (
    BUDGET_EXTENDED,
    BUDGET_STANDARD,
    DATA_POLICY_PREFS,
    QUANTIZATION_LEVELS,
    as_money,
    as_text,
    fetch_catalog,
    judge_response,
    keepalive_socket_options,
    missing_parameters,
    normalized_provider_slug,
    provider_preferences,
    run_model,
    stream_model,
    strict_provider_preferences,
)
from bench.report import (
    HUMAN_SCORER,
    build_report,
    export_line,
    export_manifest,
    export_trial,
)
from bench.scoring import judged_pass, score_response

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
    #
    # THIS IS THE SEED THAT INCREMENTS. An experiment's repeat N sends
    # base + N (see bench.experiments.seed_for_repeat), so this bound is
    # necessary and not sufficient there, and create_experiment carries
    # the arithmetic. A one-off /compare sends it unchanged, which is why
    # the field's own bound stays the simple one.
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
    # What the upstream provider billed directly, present only on BYOK
    # runs. A STRING, and the type is the point: it is stored and served
    # verbatim because nothing computes with it. It exists to be
    # reconciled against a provider's own invoice, and a float would
    # silently reformat the number that invoice has to be matched
    # against. It is never metered, because the ceiling guards the
    # OpenRouter balance this process can spend and a direct provider
    # bill is not money OpenRouter can decline.
    upstream_inference_cost_usd: str | None = None


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


# The two estimands, named at the boundary so every experiment row, every
# report and every export line says which question it answers.
#
# routed_service is the default and is what this tool is actually used
# for: the OpenRouter-routed service path for a model under a stated
# routing mode, provider chosen dynamically per call, repeats doing the
# statistical work, results stratifiable by the provider Phase G already
# records. underlying_model is the opt-in strict estimand for when the
# claim is about the model itself; it narrows the eligible provider
# population on purpose. See the README's estimands section.
ESTIMAND_MODES = ("routed_service", "underlying_model")

# Bounds on an experiment's shape. The dataset loader bounds the task
# count; these bound what the runner will multiply it by, because
# tasks * repeats * lineup is the number of paid calls and that product
# is the thing worth refusing early rather than part way through.
MAX_REPEATS = 20
MAX_TRIALS = 5000

# How often the progress stream re-reads the counters. Polling rather
# than a subscription because the runner writes progress to the database
# and the database is the shared truth: a subscription would need the
# runner to know about its watchers, and a watcher that joined late would
# have to be caught up from the database anyway. A quarter second is far
# below the pace of anything it reports (a trial is a network round trip)
# and far above the cost of one indexed row read.
PROGRESS_POLL_S = 0.25


class ExperimentCreate(BaseModel):
    model_config = FORBID_UNKNOWN

    name: str = Field(min_length=1, max_length=200)
    # A path the user names. Read by the boundary, not by the loader:
    # where the bench may read from is a question for the edge, and
    # bench.datasets stays a pure function over bytes.
    dataset_path: str = Field(min_length=1, max_length=4096)
    # Same shape and same bound as GroupCreate.models, since every trial
    # this experiment runs creates a group with exactly this lineup.
    lineup: list[str] = Field(min_length=1, max_length=MAX_POSITION + 1)
    budget: Literal["standard", "extended"]
    # The same six controls as everywhere else, through the same class, so
    # rule one and rule two apply to an experiment's payloads exactly as
    # they apply to a hand-run comparison. Structural parity again: there
    # is nothing here that could drift from what /compare sends.
    params: ExperimentParams | None = None
    repeats: int = Field(default=1, ge=1, le=MAX_REPEATS)
    # Present means shuffle the task order with this seed and record it;
    # absent means file order. Either way the order is reproducible, which
    # is the only property that matters: a hidden shuffle would make an
    # experiment unrepeatable by anyone including its author.
    #
    # Bounded against MAX_SEED and nothing further, because this seed
    # NEVER INCREMENTS: plan_trials hands it to ordered_tasks once, for
    # the whole experiment, so a value one below the ceiling is a value
    # one below the ceiling on every cell of every repeat. The
    # base-plus-repeats rule belongs to params.seed, which is the seed
    # seed_for_repeat derives from, and it lives at that field's
    # validation in create_experiment.
    task_order_seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    estimand_mode: Literal["routed_service", "underlying_model"] = "routed_service"
    # Which scorer a ranking is computed on. Absent means none declared,
    # and the report then publishes per-scorer sections with no
    # cross-scorer ranking rather than picking one alphabetically.
    # Validated at creation against what this dataset can actually
    # produce, because a primary naming a scorer nothing runs would be a
    # declaration that silently never applies.
    primary_metric: str | None = Field(default=None, min_length=1, max_length=50)
    # model id to provider name. Strict mode only; enforced with
    # allow_fallbacks false, so a pinned provider that cannot serve is a
    # recorded failure rather than a quiet reroute.
    provider_pins: dict[str, str] | None = None
    # Strict mode only, and optional within it. Quantization varies by
    # host and changes what the weights actually are, so two runs of the
    # same model served at bf16 and at int4 are not measuring the same
    # artifact. Validated against the documented levels at creation
    # rather than sent unchecked, because a filter that matches nothing
    # is, under allow_fallbacks false, a run that fails.
    quantizations: list[str] | None = Field(default=None, min_length=1, max_length=12)
    halt_on_refusal: bool = True


class ExperimentCreated(BaseModel):
    id: int


class ExperimentStart(BaseModel):
    """The dataset to read at start time.

    Asked for again rather than stored on the experiment row, and the
    asymmetry is deliberate: the row records the digest of what was read
    at creation, which is the claim about what the experiment IS. The path
    is where those bytes happened to live, which can change without the
    experiment changing. Re-reading and re-checking the digest at start is
    what turns "the file moved" into a refusal instead of a silent run
    over different tasks.
    """

    model_config = FORBID_UNKNOWN

    dataset_path: str = Field(min_length=1, max_length=4096)


class ScoringStart(BaseModel):
    """What a scoring pass needs: the tasks, and optionally a judge.

    The dataset is named again for the same reason the runner names it
    again: the rubrics and references live in the file, the experiment row
    records only the digest of what was read, and re-checking that digest
    is what stops a pass from grading one rubric's trials against
    another's.

    judge_model is optional because a dataset of deterministic scorers
    needs no judge, and requiring one would make the cheap case pay for
    the expensive one's configuration.
    """

    model_config = FORBID_UNKNOWN

    dataset_path: str = Field(min_length=1, max_length=4096)
    judge_model: str | None = Field(default=None, min_length=1, max_length=200)


class ExperimentDetail(BaseModel):
    id: int
    name: str
    created_at: str
    dataset_name: str
    dataset_digest: str
    lineup: list[str]
    budget: str
    params: dict[str, Any] | None
    repeats: int
    task_order_seed: int | None
    estimand_mode: str
    primary_metric: str | None
    provider_pins: dict[str, Any] | None
    quantizations: list[str] | None
    halt_on_refusal: bool
    status: str
    status_detail: str | None
    app_sha: str | None
    catalog_digest: str | None
    data_policy: str | None
    tasks_total: int
    trials_total: int
    trials_done: int
    trials_refused: int
    trials_failed: int


class ExperimentList(BaseModel):
    experiments: list[ExperimentDetail]


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


class StoredModelResult(ModelResult):
    """A result that has been persisted, and so has a row id.

    A subclass rather than a nullable field on ModelResult, and the
    distinction is structural rather than stylistic. A live /compare or
    /compare/stream result has no id and never will have one at the
    moment it is returned: persistence happens after the response is
    built, and the post-spend invariant explicitly allows it to fail and
    return run_id null. Declaring id on the live shape would declare a
    field that is definitionally absent there, and every consumer would
    then have to know which endpoint it was talking to in order to know
    whether None meant "not saved" or "this endpoint does not say".

    Phase I needs the id on the replay path only: a human rating or a
    score row points at a result by id, and both are produced while
    looking at stored history. The live paths are unchanged.
    """

    id: int


class RunDetail(BaseModel):
    id: int
    created_at: str
    prompt_text: str
    prompt_id: int | None
    results: list[StoredModelResult]
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
    # The one experiment runner. Per process, because the semaphore and
    # the ceiling it competes for are per process: two runners would
    # interleave through the same five slots and each would measure the
    # other's queueing. active is the running experiment's id or None,
    # task holds a strong reference so asyncio does not collect a running
    # task, and stop is the between-trials halt.
    app.state.experiment_run = {
        "active": None,
        "task": None,
        "stop": asyncio.Event(),
        "dataset_path": None,
    }
    # The scoring pass, held separately from the trial runner. They are
    # separate because they are separately useful: a finished experiment
    # can be re-scored while nothing is running, and a running experiment
    # must not be scored while its results are still arriving.
    app.state.scoring_run = {
        "active": None,
        "task": None,
        "stop": asyncio.Event(),
        "tasks": {},
        "error": None,
    }
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
    # Blind sessions belong to this process and to the person in front of
    # the page, so a new process starts with none. Cleared explicitly
    # rather than relying on module import order, because the test client
    # builds the app repeatedly in one interpreter and a session left
    # behind would make a later comparison's ratings claim a blind that
    # never happened.
    BLIND_SESSIONS.clear()
    # An experiment marked running in the database with no task behind it
    # is a lie this process is now in a position to correct. The runner
    # lives in memory, so a crash, a kill or a power cut leaves the row
    # saying running forever: the progress endpoint waits on counters
    # that will never move, and start refuses because the status is not
    # "created". Both are worse than a terminal status that says what
    # happened.
    #
    # Safe because one experiment runs at a time and this process has
    # just started: nothing is running, so anything the row calls running
    # belongs to a process that is gone.
    for stale in store.list_experiments(app.state.db):
        if stale["status"] == "running":
            logger.warning(
                "experiment %s was left running by a previous process; "
                "marking interrupted",
                stale["id"],
            )
            store.set_experiment_status(
                app.state.db,
                stale["id"],
                "interrupted",
                "the process running this experiment exited before it "
                "finished; its completed trials are real and its "
                "remaining trials never ran",
            )
    yield
    # The runner outlives the request that started it, deliberately, but
    # it must not outlive the process. Without this the task is cancelled
    # by interpreter teardown at an arbitrary await, which can be in the
    # middle of a settlement: the money is spent and the row is not
    # written. Asking it to stop between trials and then waiting is the
    # same contract the stop endpoint offers.
    await _shutdown_runner()
    # After the runner, not before. The client is what the in-flight
    # trial is streaming through, so closing it first would break the
    # exchange this line just finished waiting for.
    await app.state.client.aclose()
    app.state.db.close()


async def _shutdown_runner() -> None:
    """Ask the runner to stop, then wait for it to actually be finished.

    Three steps and they are not interchangeable. The stop event alone
    would return while a trial is still in flight. The cancel is what
    ends the run rather than merely asking, and run_experiment shields
    the trial it is on, so the cancel lands between the trial and the
    next one instead of inside the upstream exchange.

    THE AWAIT IS THE POINT. Without it this function returns while the
    runner is still settling: the process exits, the money is spent and
    the row is not written. What it costs is a bounded wait for ONE
    trial, never for the rest of the plan, because the cancellation
    breaks the loop as soon as that trial is counted. A trial whose
    upstream keeps trickling chunks can make that wait long; losing a
    paid row is worse, and that is the trade this makes on purpose.
    """
    state = getattr(app.state, "experiment_run", None) or {}
    task = state.get("task")
    if task is None or task.done():
        return
    state["stop"].set()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


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

    Three callers, and they do not all treat it the same way, so the
    difference is written down rather than left to be rediscovered:

    /compare PERSISTS it. The refusal rides in the batch and the batch
    saves as usual, which is honest history for a cut-short run.

    /compare/stream persists nothing and emits a done frame with run_id
    null. spend_refused marks that frame so the client can tell a working
    control from a broken one: a refusal and a genuine post-spend
    persistence failure both arrive as run_id null and the meanings are
    opposite, one being "no money moved" and the other "money moved and
    history lost it". Without the marker the UI described the ceiling as
    a failure. An extra field is additive by the client contract (unknown
    fields are ignored), and it is a marker rather than an error-string
    match because the error text is prose that may be reworded.

    run_one_trial PERSISTS it, and that is what this phase changed. An
    experiment's report classifies every cell of its plan, and a refusal
    with no row is indistinguishable from a cell the runner never
    reached: a budget fact would be published as a gap in the record.
    Persisted, the row carries the refusal error beside a NULL
    request_json, which is exactly the pair the era-gated derivation
    reads as "refused". The marker itself is never stored anywhere: no
    column holds it, and the classification is structural so none is
    needed.

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
            # half a result put 23 calls upstream here (stable across five
            # runs; the external report's figure was 28, and the exact
            # number depends on scheduling) where the documented bound
            # promised about five.
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


def read_dataset(path: str) -> dict[str, Any]:
    """Load and validate a dataset the user named, or 422 with the reason.

    The read lives at the boundary and the parse lives in bench.datasets,
    which is the same split as everywhere else: the edge decides what may
    be touched, the pure module decides what is valid.

    No path restriction beyond what the operating system already enforces,
    and that is deliberate rather than an omission. The bench answers only
    to loopback clients (LocalOnlyGuard) and runs as the user, so a path
    allowlist here would defend the user against themselves while blocking
    the ordinary case of a dataset living wherever the user keeps their
    work. It is worth being explicit that this is the reasoning, because
    "reads any path the caller names" looks like a hole until you have the
    threat model beside it.

    Every failure is a 422 naming the file and the line, because the
    caller's next action is to fix the file and a 500 would tell them
    nothing about which line to open.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise HTTPException(
            422, f"cannot read dataset {path}: {exc.strerror}"
        ) from None
    try:
        return parse_dataset(raw, name=Path(path).name)
    except DatasetError as exc:
        raise HTTPException(422, str(exc)) from None


def enforce_primary_metric(metric: str, dataset: dict[str, Any]) -> None:
    """A primary metric has to name something this experiment can produce.

    Checked at creation, where the dataset is already open and the answer
    is knowable, rather than at report time where the only available
    response is an empty column. A primary naming a scorer no task
    declares is a declaration that silently never applies, and the report
    would then rank every model on None and call it a ranking.
    """
    declared = sorted(
        {
            kind
            for task in dataset["tasks"]
            if isinstance(kind := (task.get("scorer") or {}).get("kind"), str)
        }
    )
    if metric in declared or metric == HUMAN_SCORER:
        return
    allowed = ", ".join([*declared, HUMAN_SCORER])
    raise HTTPException(
        422,
        f"primary_metric {metric!r} is not produced by this experiment. "
        f"This dataset declares {allowed or 'no scorers'}, and a primary "
        "metric naming something nothing runs would order every model on "
        "an empty column. Name one of those, or leave it unset and the "
        "report will publish each scorer's section without a "
        "cross-scorer ranking.",
    )


def catalog_entries() -> dict[str, dict[str, Any]]:
    """The boot catalog keyed by model id, or an empty map when offline."""
    catalog = getattr(app.state, "catalog", None) or {}
    return {m["id"]: m for m in catalog.get("models", []) if isinstance(m, dict)}


def enforce_strict_mode(
    lineup: list[str], pins: dict[str, str] | None, controls: dict[str, Any]
) -> None:
    """Everything the underlying-model estimand must be able to promise.

    Checked at creation, where it can only fail once, rather than at trial
    one of three hundred. The claim strict mode makes is about the MODEL,
    so every way the claim could turn out to be about something else is a
    refusal here:

    Without a catalog there is nothing to check against, so the whole
    check is unavailable and the experiment cannot be created. Skipping it
    silently on an offline boot would produce rows labelled
    underlying_model whose capability check never ran, which is a strict
    mode that isn't; a label nobody verified is worse than no label,
    because a reader has no way to tell the two apart afterwards.

    A model the catalog does not list, or lists without
    supported_parameters, is the same problem one model down. Absence of
    evidence is not support.

    A parameter the model does not support is a refusal because strict
    mode sends require_parameters: no provider would be eligible, so the
    experiment would fail every trial rather than measure anything.

    A pin naming a model outside the lineup is a contradiction in the
    declaration itself, and a declaration that cannot be honored should
    not be stored as though it will be.
    """
    catalog = getattr(app.state, "catalog", None) or {}
    if not catalog.get("fetched"):
        raise HTTPException(
            422,
            "the underlying-model estimand checks every control against "
            "the model catalog, and this boot has no catalog: the "
            "snapshot could not be fetched at startup. Restart with "
            "OpenRouter reachable, or create this experiment under the "
            "routed-service estimand, which makes no capability claim. "
            "Creating it without the check would label the rows with a "
            "guarantee nothing verified.",
        )
    entries = catalog_entries()
    for model in lineup:
        entry = entries.get(model)
        if entry is None:
            raise HTTPException(
                422,
                f"the catalog does not list {model}, so its parameter "
                "support cannot be checked. The underlying-model estimand "
                "refuses rather than assume support.",
            )
        supported = entry.get("supported_parameters")
        if supported is None:
            raise HTTPException(
                422,
                f"the catalog lists {model} but publishes no "
                "supported_parameters for it, so there is nothing to "
                "check the controls against. The underlying-model "
                "estimand refuses rather than assume support.",
            )
        absent = missing_parameters(supported, controls)
        if absent:
            raise HTTPException(
                422,
                f"{model} does not support {', '.join(absent)} according "
                "to the catalog, and the underlying-model estimand sends "
                "require_parameters, so no provider would be eligible to "
                "serve it. Drop the control, drop the model, or use the "
                "routed-service estimand, where an unsupported parameter "
                "is silently ignored by the provider and the request "
                "record still shows it was asked for.",
            )
    for model in sorted(pins or {}):
        if model not in lineup:
            raise HTTPException(
                422,
                f"provider_pins names {model}, which is not in the "
                "lineup. A pin for a model this experiment never runs "
                "would be recorded as a constraint that never applied.",
            )


@app.post("/experiments", response_model=ExperimentCreated, status_code=201)
async def create_experiment(body: ExperimentCreate) -> dict[str, Any]:
    """Write the experiment manifest, after validating everything about it.

    Nothing runs here. The row is the declaration, written complete before
    the first trial, exactly as a group row is written before the first
    upstream call: the same discipline one level up.

    Validation that can only fail later is validation in the wrong place.
    The dataset is loaded and checked now, the trial count is bounded now,
    and (in strict mode) every control is capability-checked against the
    catalog now, so an experiment that cannot run does not get created and
    then discovered at trial 300.
    """
    dataset = read_dataset(body.dataset_path)
    controls = request_controls(body.params)
    trials = len(dataset["tasks"]) * body.repeats * len(body.lineup)
    if trials > MAX_TRIALS:
        raise HTTPException(
            422,
            f"{len(dataset['tasks'])} tasks x {body.repeats} repeats x "
            f"{len(body.lineup)} models is {trials} paid calls, over the "
            f"{MAX_TRIALS} limit. Shorten the dataset or the lineup.",
        )
    if body.quantizations and body.estimand_mode != "underlying_model":
        raise HTTPException(
            422,
            "quantizations narrow which hosts may serve, which is the "
            "underlying-model estimand's business and not the routed "
            'service\'s. Set estimand_mode to "underlying_model", or '
            "drop the filter.",
        )
    if body.quantizations:
        unknown = sorted(set(body.quantizations) - set(QUANTIZATION_LEVELS))
        if unknown:
            raise HTTPException(
                422,
                f"quantizations {unknown} are not documented levels. "
                f"OpenRouter accepts {', '.join(QUANTIZATION_LEVELS)}; a "
                "level it does not recognize would filter to no provider "
                "at all, and with allow_fallbacks false that is a run "
                "that fails rather than one served by somebody else.",
            )
    if body.provider_pins and body.estimand_mode != "underlying_model":
        # A pin under the routed-service estimand is a contradiction: the
        # whole point of that estimand is that routing is dynamic. Refusing
        # is kinder than silently ignoring the pins, which would produce a
        # record claiming a pin that never rode a payload.
        raise HTTPException(
            422,
            "the routed-service estimand routes dynamically by definition, "
            "so provider_pins cannot apply to it. To pin providers, set "
            'estimand_mode to "underlying_model", which is the estimand '
            "that narrows the provider population on purpose.",
        )
    sampling_seed = (controls or {}).get("seed")
    if sampling_seed is not None:
        # THE SEED THAT INCREMENTS, which is params.seed and is not
        # task_order_seed. seed_for_repeat derives base + N and the
        # runner sends the result to the provider, so the LAST repeat's
        # seed is base + repeats - 1: bounding only the base lets a legal
        # base sit one repeat below the ceiling and hand the provider a
        # value past what survives a browser read-back, silently, on the
        # last cell of a long run. Checked here where the arithmetic can
        # be shown rather than at trial 300 where it cannot.
        #
        # This check used to sit on task_order_seed, which never
        # increments: plan_trials calls ordered_tasks with it once, for
        # the whole experiment. So the rule was correct, its explanation
        # was correct, and it was pointed at the wrong field. It guarded
        # a value that cannot overflow and left the one that can
        # unguarded, which is a defect that looks exactly like coverage.
        highest = sampling_seed + body.repeats - 1
        if highest > MAX_SEED:
            raise HTTPException(
                422,
                f"seed {sampling_seed} plus {body.repeats} repeats "
                f"reaches {highest}, past the {MAX_SEED} limit: repeat N "
                "sends seed base + N, so the last repeat is the one that "
                "would overflow. Lower the seed or the repeat count.",
            )
    if body.primary_metric is not None:
        enforce_primary_metric(body.primary_metric, dataset)
    if body.estimand_mode == "underlying_model":
        enforce_strict_mode(body.lineup, body.provider_pins, controls)
    return {
        "id": store.create_experiment(
            app.state.db,
            {
                "name": body.name,
                "dataset_name": dataset["name"],
                "dataset_digest": dataset["digest"],
                "lineup": body.lineup,
                "budget": body.budget,
                "params": controls or None,
                "repeats": body.repeats,
                "task_order_seed": body.task_order_seed,
                "estimand_mode": body.estimand_mode,
                "primary_metric": body.primary_metric,
                # Normalized at the boundary so the recorded pin and the
                # sent pin are one string. A report citing a pin is only
                # worth reading if the pin it names is the pin that rode
                # the payload.
                "provider_pins": (
                    {
                        model: normalized_provider_slug(name)
                        for model, name in body.provider_pins.items()
                    }
                    if body.provider_pins
                    else None
                ),
                "quantizations": body.quantizations,
                "halt_on_refusal": body.halt_on_refusal,
                # The same provenance every run row carries, recorded once
                # on the experiment because it is fixed for the whole of
                # it: the process does not change build or catalog mid-run.
                "app_sha": getattr(app.state, "app_sha", None),
                "catalog_digest": getattr(app.state, "catalog_digest", None),
                "data_policy": app.state.data_policy,
                "tasks_total": len(dataset["tasks"]),
                "trials_total": trials,
            },
        )
    }


@app.get("/experiments", response_model=ExperimentList)
async def list_experiments() -> dict[str, Any]:
    return {"experiments": store.list_experiments(app.state.db)}


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
async def experiment_detail(experiment_id: int) -> dict[str, Any]:
    ensure_rowid(experiment_id)
    experiment = store.get_experiment(app.state.db, experiment_id)
    if experiment is None:
        raise HTTPException(404, "no such experiment")
    return experiment


def trial_provider_prefs(
    experiment: dict[str, Any], model: str, controls: dict[str, Any]
) -> dict[str, Any]:
    """The provider block for one trial, estimand included.

    Under the routed-service estimand this is request_provider_prefs and
    nothing else, which is not an implementation detail but the promise:
    the default path sends the same object it sent before strict mode
    existed, so nothing about adding this estimand changed what the
    bench's own comparisons measure. A tombstone asserts the payload
    bytes.

    Under the underlying-model estimand the strict keys are layered on
    top of that same object, per model, because the pin is per model.
    """
    base = request_provider_prefs(controls)
    if experiment.get("estimand_mode") != "underlying_model":
        return base
    pins = experiment.get("provider_pins") or {}
    return strict_provider_preferences(
        base, pins.get(model), experiment.get("quantizations")
    )


async def run_one_trial(
    experiment: dict[str, Any],
    task: dict[str, Any],
    model: str,
    position: int,
    group_id: int,
    controls: dict[str, Any],
) -> dict[str, Any]:
    """One model against one task, through the machinery every run uses.

    The same semaphore, the same post-admission ceiling recheck, the same
    settlement inside the held slot, the same never-raises client. The
    runner gets no faster path and no larger share: an experiment
    competing with a browser run for the five slots is correct, because
    both are spending the same money against the same ceiling.

    Streaming rather than the batch call because ttft is a measurement the
    report publishes, and only the streaming path can observe it. The
    frames go nowhere: this is the one consumer of stream_model with no
    client attached, which is also why there is no disconnect path here.
    The runner is the client, and it does not disconnect.

    Returns the result dict. Never raises for an upstream failure, which
    is the same contract run_model and stream_model carry, because a
    failed trial is data and must not end the experiment.
    """
    max_tokens = effective_budget(experiment["budget"], model)
    holder: dict[str, Any] = {}
    parts: list[str] = []
    first_delta_ms: float | None = None
    acquired = False
    result: dict[str, Any] | None = None
    try:
        await app.state.upstream_semaphore.acquire()
        acquired = True
        # The recheck every admitted run gets, before the clock and before
        # any upstream call. An experiment is exactly the case this
        # protects against: hundreds of trials admitted over minutes, with
        # the ceiling crossed somewhere in the middle.
        if spend_ceiling_reached():
            # Set rather than returned, so the refusal falls through to
            # the same persistence the other outcomes get. The endpoints
            # deliberately persist nothing for a refusal; the runner must,
            # because an experiment's report classifies every cell of its
            # plan and a refusal with no row is indistinguishable from a
            # cell that was never reached. The row is the refusal's only
            # evidence: error text beside a NULL request_json, which is
            # exactly what the era-gated derivation reads as "refused".
            result = spend_refusal_result(model, max_tokens)
        else:
            start = time.perf_counter()
            async for event in stream_model(
                task["prompt"],
                model,
                app.state.client,
                max_tokens=max_tokens,
                holder=holder,
                provider_prefs=trial_provider_prefs(experiment, model, controls),
                controls=controls,
            ):
                if event["type"] == "done":
                    result = event["result"]
                    break
                if first_delta_ms is None:
                    first_delta_ms = round((time.perf_counter() - start) * 1000, 1)
                parts.append(event["text"])
    finally:
        # Released before persistence, exactly as the streaming endpoint
        # releases before saving: a slot is for the upstream exchange, and
        # holding one through a database write is a lie about how many
        # paid calls are in flight.
        if acquired:
            app.state.upstream_semaphore.release()

    if result is None:
        # stream_model ended without a done event. Its own contract makes
        # that nearly impossible, but "nearly" is not a thing to persist a
        # silent success for: build the same shape the disconnect path
        # builds, so the trial lands as the failure it was.
        result = {
            "model": model,
            "response_text": "".join(parts) or None,
            "latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": "stream ended without a terminal event",
            "cost_usd": None,
            "ttft_ms": first_delta_ms,
            "max_tokens": max_tokens,
            "generation_id": holder.get("generation_id"),
            "request_json": holder.get("request_json"),
            "provider": holder.get("provider"),
        }
    result["position"] = position
    # Post-spend, so its own fault boundary: the call has happened and
    # nothing here may turn a paid result into an error. A refusal skips
    # it entirely rather than settling to zero: no call happened, so there
    # is nothing to price and nothing to add to the ceiling, and running
    # the settlement anyway would put a refusal into the spend record as
    # a zero-cost call that was made.
    if not result.get("spend_refused"):
        try:
            result["cost_usd"] = cost_usd(result, app.state.prices)
            record_spend(ceiling_cost(result))
        except Exception:
            logger.exception("settlement failed for %s", model)
            result["cost_usd"] = None
    try:
        store.save_run(
            app.state.db,
            task["prompt"],
            [result],
            None,
            group_id,
            run_provenance(),
        )
    except Exception:
        # Same rule as both endpoints: the money is already spent, and
        # losing history must not lose the response. The counters still
        # move, so progress stays honest about what was attempted.
        logger.exception("failed to persist experiment trial for %s", model)
    return result


# What a run says when the PROCESS ended rather than the run. Written by
# the two cancellation handlers in run_experiment; the lifespan's
# stale-row sweep says its own thing, because "a previous process left
# this row behind" and "this process is shutting down now" are different
# facts and the second one knows more.
#
# Not "stopped" and not "failed". Nobody asked this experiment to end and
# nothing about it went wrong: its completed trials are real, its
# in-flight trial was paid for and persisted, and the rest never ran.
INTERRUPTED_BY_SHUTDOWN = (
    "this process shut down while the experiment was running; the trial "
    "in flight was settled and persisted before the runner finished, and "
    "the remaining trials never ran"
)


async def run_experiment(experiment_id: int) -> None:
    """Walk every cell of an experiment, one trial at a time.

    Owned by app.state as a background task, which is what lets it outlive
    the request that started it. That is the deliberate opposite of the
    per-run disconnect contract: a browser run belongs to the tab that
    started it and is abandoned when the tab goes away, because nobody is
    watching it any more. An experiment belongs to the bench, not to a
    watcher, and a laptop lid closing halfway through a paid sweep must
    not silently end it. The progress stream is a view onto the run, not
    the run itself.

    Creates every group through the same entry-checked path any client
    uses. There is no bypass: enforce_group_experiment runs on this
    runner's own trials, so experiment-to-group consistency is the law
    H1.2 already wrote rather than a promise this workstream makes.
    """
    db = app.state.db
    experiment = store.get_experiment(db, experiment_id)
    assert experiment is not None
    state = app.state.experiment_run
    lineup = experiment["lineup"]
    controls_base = experiment["params"] or {}
    try:
        dataset = read_dataset(state["dataset_path"])
    except HTTPException as exc:
        # The file moved or became unreadable between creation and start.
        # Not a crash: the experiment records why it never ran.
        #
        # The singleton is released here as well as in the finally below,
        # because this return happens BEFORE the try that owns it. Left
        # set, one unreadable dataset wedged every later start with "an
        # experiment is already running" naming an experiment that had
        # already failed, until the process was restarted.
        store.set_experiment_status(db, experiment_id, "failed", str(exc.detail))
        state["active"] = None
        return
    if dataset["digest"] != experiment["dataset_digest"]:
        # The bytes changed under it. Refusing is the whole point of
        # recording a content digest: running anyway would produce a
        # record citing one dataset and containing another, which is the
        # single most misleading artifact this phase could emit.
        store.set_experiment_status(
            db,
            experiment_id,
            "failed",
            f"dataset changed since this experiment was created: recorded "
            f"{experiment['dataset_digest'][:12]}, file is "
            f"{dataset['digest'][:12]}",
        )
        return
    store.set_experiment_status(db, experiment_id, "running")
    plan = plan_trials(
        dataset["tasks"], lineup, experiment["repeats"], experiment["task_order_seed"]
    )
    status, detail = "done", None
    # Set by the cancellation handler around each trial, and read once
    # that trial has been counted. Declared out here rather than per cell
    # because it is a fact about the process rather than about a trial:
    # once it is true the run is over.
    interrupted = False
    try:
        for cell in plan:
            if state["stop"].is_set():
                status, detail = "stopped", "stopped between trials"
                break
            task = cell["task"]
            # The seed this repeat sends. Derived rather than fixed so
            # repeats sample variation instead of buying the same sample
            # N times; absent stays absent, which is rule one.
            controls = dict(controls_base)
            seeded = seed_for_repeat(controls_base.get("seed"), cell["repeat_index"])
            if seeded is not None:
                controls["seed"] = seeded
            # A task's own system prompt overrides the experiment's, since
            # it is the more specific declaration. Recorded on the group
            # like any other control, so the record says what was sent.
            if task["system"] is not None:
                controls["system"] = task["system"]
            group_id = store.create_group(
                db,
                task["prompt"],
                lineup,
                controls or None,
                experiment["budget"],
                {
                    "experiment_id": experiment_id,
                    "task_id": task["id"],
                    "repeat_index": cell["repeat_index"],
                    "rotation_index": cell["rotation_index"],
                },
            )
            halted = False
            for position, model in cell["order"]:
                if state["stop"].is_set():
                    status, detail = "stopped", "stopped between trials"
                    halted = True
                    break
                # Position is the model's place in the DECLARED lineup,
                # not its place in this cell's rotated entry order. The
                # rotation changes who asks first; it does not change
                # which column a model occupies, and a report that mixed
                # the two would attribute one model's results to
                # another's slot.
                #
                # It arrives with the arm rather than being looked up by
                # name. lineup.index(model) returns the FIRST match, so a
                # lineup listing one model twice gave both copies
                # position 0: two paid trials collapsed into one column,
                # the second silently overwriting the first in every
                # per-position reader. A duplicate lineup is the natural
                # way to ask "how much does this model vary run to run",
                # so it has to be the case that works.
                # The entry checks, on the runner's own trial, exactly as a
                # client's request gets them. Force a mismatch and this
                # 409s the runner too, which is the point.
                enforce_group_experiment(
                    task["prompt"],
                    controls,
                    group_id,
                    experiment["budget"],
                    model=model,
                    position=position,
                )
                # SHIELDED, then awaited again. Shutdown cancels this
                # task, and without the shield the CancelledError landed
                # at whatever await run_one_trial was suspended on, which
                # is the upstream stream: the call was paid for and
                # neither the settlement nor the row ever happened. The
                # money moved and nothing recorded it, which is the exact
                # outcome _shutdown_runner's docstring claims to prevent.
                #
                # The shield lets the trial finish on its own while the
                # cancellation is delivered here; re-awaiting it collects
                # the result once its settlement and its save_run have
                # run. Bounded by the upstream timeout the client already
                # carries, so this cannot hold shutdown open forever.
                trial = asyncio.ensure_future(
                    run_one_trial(experiment, task, model, position, group_id, controls)
                )
                try:
                    result = await asyncio.shield(trial)
                except asyncio.CancelledError:
                    result = await trial
                    interrupted = True
                refused = bool(result.get("spend_refused"))
                failed = not refused and result.get("error") is not None
                # Three disjoint buckets, which is what the conservation
                # property means: done plus failed plus refused is the
                # number of trials that finished, and the gap to
                # trials_total is what never ran. Counting a failed trial
                # as done as well would make trials_done mean "finished"
                # while sitting beside a column named trials_failed, and
                # the sum would exceed the total.
                store.bump_experiment_counters(
                    db,
                    experiment_id,
                    done=0 if (refused or failed) else 1,
                    refused=1 if refused else 0,
                    failed=1 if failed else 0,
                )
                if interrupted:
                    # After the counters, deliberately. The trial finished
                    # and was persisted, so conservation has to hold over
                    # it exactly as it does over every other outcome; a
                    # break that skipped the bump would leave a row in the
                    # database that no counter had ever seen.
                    #
                    # Ahead of the refusal branch because the process
                    # ending outranks anything about this experiment. A
                    # run cut short by shutdown is not halted_on_refusal
                    # even if its last trial was refused: the reason it
                    # stopped is that the process went away.
                    status, detail = "interrupted", INTERRUPTED_BY_SHUTDOWN
                    halted = True
                    break
                if refused and experiment["halt_on_refusal"]:
                    # The default. A ceiling refusal means the money ran
                    # out, and continuing would produce a record whose
                    # later rows are all refusals: technically honest,
                    # practically noise, and expensive in wall time. The
                    # un-run remainder is visible through the same
                    # declared-but-missing machinery a partial comparison
                    # already uses.
                    status = "halted_on_refusal"
                    detail = (
                        "the per-boot spend ceiling was reached; "
                        f"{model} on task {task['id']} was refused before "
                        "reaching upstream"
                    )
                    halted = True
                    break
            if halted:
                break
    except asyncio.CancelledError:
        # A cancel that landed somewhere other than the shielded trial.
        # CancelledError is a BaseException, so `except Exception` never
        # saw it and the finally below wrote whatever `status` still held,
        # which on the ordinary path is "done": a run cut off by shutdown
        # published itself as a completed experiment. Every rate in its
        # report was then computed against a plan the reader had no reason
        # to doubt, and nothing anywhere said the process had gone away.
        #
        # Re-raised, because the caller asked this task to end and a
        # coroutine that swallows its own cancellation is lying to the
        # scheduler as well. The finally runs first, so the honest status
        # is written either way.
        status, detail = "interrupted", INTERRUPTED_BY_SHUTDOWN
        raise
    except Exception as exc:
        logger.exception("experiment %s failed", experiment_id)
        status, detail = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        store.set_experiment_status(db, experiment_id, status, detail)
        state["active"] = None


@app.post("/experiments/{experiment_id}/start", status_code=202)
async def start_experiment(experiment_id: int, body: ExperimentStart) -> dict[str, Any]:
    """Start the runner. One at a time, and it says so when refusing.

    One active experiment rather than a queue, because the semaphore and
    the ceiling are process-wide: two experiments would interleave their
    trials through the same five slots and each would measure the other's
    queueing. That is not a limitation to apologize for, it is the only
    arrangement in which an experiment's latency numbers mean anything.
    """
    ensure_rowid(experiment_id)
    experiment = store.get_experiment(app.state.db, experiment_id)
    if experiment is None:
        raise HTTPException(404, "no such experiment")
    if experiment["status"] != "created":
        raise HTTPException(
            409,
            f"experiment {experiment_id} is {experiment['status']}; an "
            "experiment runs once. Create another to run it again.",
        )
    state = app.state.experiment_run
    if state["active"] is not None:
        raise HTTPException(
            409,
            f"experiment {state['active']} is already running. One at a "
            "time: they share the upstream slots and the spend ceiling, "
            "so concurrent experiments would measure each other.",
        )
    state["active"] = experiment_id
    state["stop"] = asyncio.Event()
    state["dataset_path"] = body.dataset_path
    state["error"] = None
    # Held on app.state so the task is not garbage collected mid-run, and
    # so a stop can be issued against it. asyncio keeps only a weak
    # reference to a bare create_task.
    state["task"] = asyncio.create_task(run_experiment(experiment_id))
    return {"id": experiment_id, "status": "running"}


@app.post("/experiments/{experiment_id}/stop", status_code=202)
async def stop_experiment(experiment_id: int) -> dict[str, Any]:
    """Ask the runner to halt between trials.

    Between trials, never inside one. A trial that has reached upstream
    has already spent its money, and abandoning it would throw away a
    result the user paid for; the same reasoning as the streaming path's
    disconnect persistence, one level up.
    """
    ensure_rowid(experiment_id)
    state = app.state.experiment_run
    if state["active"] != experiment_id:
        raise HTTPException(409, f"experiment {experiment_id} is not running")
    state["stop"].set()
    return {"id": experiment_id, "status": "stopping"}


@app.get("/experiments/{experiment_id}/progress")
async def experiment_progress(experiment_id: int) -> StreamingResponse:
    """Counters as SSE, rejoinable, and never the thing that keeps the
    run alive.

    A client that disconnects and reconnects gets the current counters
    immediately and then updates, because every frame carries absolute
    values rather than deltas. Absolute frames also mean a dropped frame
    costs nothing, which is what makes reconnection cheap enough to be
    the ordinary case rather than an error path.
    """
    ensure_rowid(experiment_id)
    if store.get_experiment(app.state.db, experiment_id) is None:
        raise HTTPException(404, "no such experiment")

    async def frames() -> AsyncIterator[str]:
        last = None
        while True:
            experiment = store.get_experiment(app.state.db, experiment_id)
            if experiment is None:
                return
            frame = {
                "type": "progress",
                "status": experiment["status"],
                "status_detail": experiment["status_detail"],
                "trials_total": experiment["trials_total"],
                "trials_done": experiment["trials_done"],
                "trials_refused": experiment["trials_refused"],
                "trials_failed": experiment["trials_failed"],
                "spend_usd": app.state.accumulated_spend_usd,
            }
            if frame != last:
                yield "data: " + json.dumps(frame) + "\n\n"
                last = frame
            if experiment["status"] not in ("created", "running"):
                return
            await asyncio.sleep(PROGRESS_POLL_S)

    return StreamingResponse(frames(), media_type="text/event-stream")


async def score_experiment(experiment_id: int, judge_model: str | None) -> None:
    """Score every trial of an experiment, deterministic and judged alike.

    Re-runnable and idempotent per (result, scorer, judge_model): a second
    pass inserts new rows rather than editing old ones, and the report
    reads the latest per key. That is what makes the interruption policy
    below safe, and it is what keeps "the judge said 0.5 last week and 1.0
    today" in the record instead of erasing it.

    A ceiling refusal during the pass is recorded as that result's scoring
    failure and the pass CONTINUES. This is the opposite of the trial
    runner's default, and the asymmetry is the point: a refused trial can
    never be recovered without paying for the model call again, so
    halting protects the budget for a decision the user should make. A
    refused score can be filled in by a later pass over the same stored
    text at no extra model cost, so stopping the whole pass for one would
    trade a complete scoring run for nothing.
    """
    db = app.state.db
    experiment = store.get_experiment(db, experiment_id)
    assert experiment is not None
    state = app.state.scoring_run
    try:
        # Self-judging is recorded, not prevented. A judge grading its own
        # output has a documented tendency toward itself, and the honest
        # response is to flag every affected row and surface the flag in
        # the report rather than to refuse the pass: the user may have
        # good reason, and a silent absorption is what would be wrong.
        self_judged = judge_model is not None and judge_model in experiment["lineup"]
        for group in store.experiment_groups(db, experiment_id):
            if state["stop"].is_set():
                break
            task = state["tasks"].get(group["task_id"])
            if task is None or task["scorer"] is None:
                continue
            detail = store.get_group(db, group["id"])
            if detail is None:
                continue
            for run in detail["runs"]:
                for result in run["results"]:
                    await score_one_result(
                        experiment_id, task, result, judge_model, self_judged
                    )
    except Exception as exc:
        logger.exception("scoring pass for experiment %s failed", experiment_id)
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        state["active"] = None


async def score_one_result(
    experiment_id: int,
    task: dict[str, Any],
    result: dict[str, Any],
    judge_model: str | None,
    self_judged: bool,
) -> None:
    """Score one stored result, writing exactly one row.

    Every path writes a row, including the failures. A scoring pass that
    silently skipped what it could not score would leave the report
    unable to tell "not scored yet" from "scored and unscorable", and
    those are different facts about the same trial.
    """
    spec = task["scorer"]
    kind = spec["kind"]
    if kind != "judge":
        # Off the loop thread. The deterministic scorers are string work
        # that finishes in microseconds, but the regex one waits up to
        # REGEX_DEADLINE_SECONDS on a child process, and blocking the
        # loop for a second per pathological pattern would stall the
        # progress stream and every browser run sharing this process.
        # The regex never executes here in either sense: not in this
        # thread, and not in this process.
        verdict = await asyncio.to_thread(
            score_response, spec, task["reference"], result["response_text"]
        )
        store.add_score(
            app.state.db,
            result["id"],
            {
                "scorer": kind,
                "score": verdict["score"],
                "passed": verdict["passed"],
                "detail": verdict["detail"],
                "blind": None,
                "self_judged": None,
            },
        )
        return

    if judge_model is None:
        store.add_score(
            app.state.db,
            result["id"],
            {
                "scorer": "judge",
                "score": None,
                "passed": None,
                "detail": "no judge model was given for this scoring pass",
            },
        )
        return

    # The ceiling applies to judge calls because judge calls are spend.
    # A refusal here is this result's scoring failure and the pass goes
    # on; see score_experiment for why that differs from a trial refusal.
    def refuse_for_ceiling() -> None:
        store.add_score(
            app.state.db,
            result["id"],
            {
                "scorer": "judge",
                "score": None,
                "passed": None,
                "detail": (
                    "the per-boot spend ceiling was reached before this "
                    "result could be judged; re-run the scoring pass to "
                    "fill it in"
                ),
                "judge_model": judge_model,
                "self_judged": self_judged,
            },
        )

    # Before the queue, so a pass that has already blown the ceiling does
    # not spend minutes waiting for slots it will refuse anyway.
    if spend_ceiling_reached():
        refuse_for_ceiling()
        return

    async with app.state.upstream_semaphore:
        # And again inside the held slot, which is the check that actually
        # protects the money. Admission is not permission to spend: this
        # call can sit behind five others for as long as they take, and a
        # pass over three hundred results is exactly the case where the
        # ceiling is crossed during that wait. Checking only before the
        # queue is the F1.2 defect, one layer up.
        if spend_ceiling_reached():
            refuse_for_ceiling()
            return
        verdict = await judge_response(
            app.state.client,
            judge_model,
            task["rubric"],
            task["reference"],
            result["response_text"],
            # Routed-service prefs even under the underlying-model
            # estimand. The estimand is a claim about the models being
            # MEASURED; the judge is the bench's own instrument and is
            # not one of them, and pinning it to a provider chosen for
            # somebody else's lineup would be a claim nobody made.
            provider_prefs=app.state.provider_prefs,
        )
    # Post-spend. The call has happened, so nothing below may turn a paid
    # verdict into an exception, and the charge is recorded whether or not
    # the verdict parsed.
    try:
        record_spend(as_money(verdict["billed_cost_usd"]))
    except Exception:
        logger.exception("judge settlement failed")
    store.add_score(
        app.state.db,
        result["id"],
        {
            "scorer": "judge",
            "score": verdict["score"],
            # From the task's own threshold when its author declared one,
            # and None otherwise. judge_response cannot decide this: it is
            # a client function and has never seen the dataset spec.
            "passed": judged_pass(spec, verdict["score"]),
            "detail": verdict["error"] or verdict["detail"],
            "judge_model": judge_model,
            "judge_generation_id": verdict["generation_id"],
            "judge_billed_cost_usd": verdict["billed_cost_usd"],
            "self_judged": self_judged,
        },
    )


@app.post("/experiments/{experiment_id}/score", status_code=202)
async def start_scoring(experiment_id: int, body: ScoringStart) -> dict[str, Any]:
    ensure_rowid(experiment_id)
    experiment = store.get_experiment(app.state.db, experiment_id)
    if experiment is None:
        raise HTTPException(404, "no such experiment")
    if experiment["status"] in ("created", "running"):
        raise HTTPException(
            409,
            f"experiment {experiment_id} is {experiment['status']}; score it "
            "once its trials have finished, so the pass sees every result",
        )
    state = app.state.scoring_run
    if state["active"] is not None:
        raise HTTPException(
            409, f"a scoring pass for experiment {state['active']} is running"
        )
    dataset = read_dataset(body.dataset_path)
    if dataset["digest"] != experiment["dataset_digest"]:
        raise HTTPException(
            422,
            "dataset changed since this experiment was created: recorded "
            f"{experiment['dataset_digest'][:12]}, file is "
            f"{dataset['digest'][:12]}. Scoring against different tasks "
            "would attribute one rubric's verdict to another's trial.",
        )
    state["active"] = experiment_id
    state["stop"] = asyncio.Event()
    state["error"] = None
    state["tasks"] = {t["id"]: t for t in dataset["tasks"]}
    state["task"] = asyncio.create_task(
        score_experiment(experiment_id, body.judge_model)
    )
    return {"id": experiment_id, "status": "scoring"}


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


# ---- Phase I4: blind human rating on replay.

# The rating scale, fixed rather than configurable. One scale everywhere
# means a stored rating means the same thing in every row, which is the
# property a report needs; a per-experiment scale would make "3" a
# number nobody could interpret without also fetching its scale.
#
# Five points because it is the smallest scale with a middle and a
# near-miss on each side, and because raters are reliably worse at
# finer ones. Stored normalized to [0, 1] like every other score, so the
# report averages human and machine scores without knowing which is
# which; the raw point value stays in detail for anyone who wants it.
# static/rating.js declares the same pair and renders one button per
# point; test_the_rating_scale_matches_the_server_constant asserts the two
# agree, because a client scale wider than this renders a button whose
# click is refused and a narrower one makes the top unreachable.
RATING_MIN = 1
RATING_MAX = 5

# Neutral names for the anonymized cards. Letters rather than numbers
# because a rater reading "1" and "2" beside answers tends to read them
# as a ranking somebody already made, and MAX_POSITION + 1 of them
# because that is how many cards a comparison can hold.
BLIND_LABELS = tuple(chr(ord("A") + i) for i in range(MAX_POSITION + 1))


def normalized_rating(rating: int) -> float:
    """A 1-to-5 rating on the [0, 1] scale every score row uses."""
    return (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)


class Rating(BaseModel):
    model_config = FORBID_UNKNOWN

    result_id: int
    rating: int = Field(ge=RATING_MIN, le=RATING_MAX)
    # The neutral label this card wore while it was being rated. Recorded
    # so the rating-to-model mapping is auditable after the reveal: with
    # it, someone checking the record can see that card "B" was
    # model/beta and was rated 4. Without it, a blind rating is a number
    # whose blindness has to be taken on faith.
    label: str | None = Field(default=None, max_length=40)


class RatingsSubmit(BaseModel):
    model_config = FORBID_UNKNOWN

    ratings: list[Rating] = Field(min_length=1, max_length=MAX_POSITION + 1)
    # The token the server issued when it opened a blind session, echoed
    # back. This is the ONLY thing that can make a rating blind; see
    # blind_token_valid for why a boolean could not.
    #
    # Absent is the ordinary case and it is not an error: a sighted
    # rating is a real rating, it simply persists blind = 0. Bounded
    # because it is a string from a request and every string from a
    # request is bounded.
    blind_token: str | None = Field(default=None, max_length=200)
    # RECEIVED AND IGNORED. Nothing reads this field. The server writes
    # its own answer from the blind session's token instead, because a
    # page that revealed the identities and then claimed blind=true would
    # be testifying about its own past, which is the one witness a blind
    # record cannot use. See submit_ratings and blind_token_valid.
    #
    # No client in this repository sends it any more, as of the commit
    # that removed the legacy rate-blind path. It remains only so a
    # third-party script that has always sent it is not rejected by
    # extra="forbid" for doing what it was told to do. Removing it is a
    # breaking change to the request shape and is a NAMED DEFERRAL,
    # waiting for a moment when breaking changes are being made
    # deliberately rather than as a side effect of a correctness fix.
    blind: bool = False


@app.post("/groups/{group_id}/ratings", status_code=201)
async def submit_ratings(group_id: int, body: RatingsSubmit) -> dict[str, Any]:
    """Persist human ratings for one replayed comparison.

    Every result named must belong to this group. That is not paranoia
    about a hostile client (the bench answers only to loopback), it is
    about a stale page: a tab holding a previous comparison's result ids
    would otherwise write ratings onto the wrong models silently, and a
    silent misattribution is the worst outcome this endpoint has.

    THE BLIND FLAG IS THE SERVER'S, not the client's, and it is decided
    by the TOKEN these ratings carry rather than by anything the body
    claims. A blind rating is evidence about the conditions a person
    judged under, so the one party that cannot be the witness is the page
    that was showing them the answers.

    A per-group boolean was not enough, and the gap was reachable in one
    click. It answered "is a blind session open for this group", which is
    true for EVERY tab the moment any one of them opens one: a second tab
    replaying the same comparison sighted, with every model name on
    screen, posted its ratings into a blind that a different window had
    opened, and they persisted blind = 1. The token answers the question
    that was actually meant, "did THESE ratings come from a blind
    session", and only the tab the server handed it to can answer yes.
    """
    ensure_rowid(group_id)
    group = store.get_group(app.state.db, group_id)
    if group is None:
        raise HTTPException(404, "no such group")
    belongs = {r["id"] for run in group["runs"] for r in run["results"]}
    unknown = sorted({r.result_id for r in body.ratings} - belongs)
    if unknown:
        raise HTTPException(
            422,
            f"result ids {unknown} are not in comparison {group_id}. The "
            "page may be showing an older comparison than the one it is "
            "rating; reload it.",
        )
    blind = blind_token_valid(group_id, body.blind_token)
    written = 0
    for rating in body.ratings:
        store.add_score(
            app.state.db,
            rating.result_id,
            {
                "scorer": "human",
                "score": normalized_rating(rating.rating),
                # No pass verdict, for the same reason a judge without a
                # declared threshold has none: nobody said what a passing
                # rating is, and the bench inventing one would assert a
                # cutoff the rater never chose.
                "passed": None,
                "detail": _rating_detail(rating),
                "blind": blind,
            },
        )
        written += 1
    return {"group_id": group_id, "written": written, "blind": blind}


def _rating_detail(rating: Rating) -> str:
    """The raw rating, and the label it was given under.

    The score column holds the normalized value so every scorer's numbers
    are comparable; this keeps the point the rater actually clicked, so
    "4 of 5" survives as what happened rather than as 0.75, which is a
    number no rater ever saw.
    """
    base = f"{rating.rating} of {RATING_MAX}"
    return f"{base}, shown as {rating.label}" if rating.label else base


# Blind sessions, in memory and per group. Not a table, deliberately: a
# session is a fact about a person sitting in front of a page right now,
# it expires with the process, and persisting it would create a second
# record of blindness that could disagree with the score rows, which are
# the only record that matters afterwards. A restart fails closed.
#
# Three states, and the None is load-bearing:
#   absent  no session was ever opened for this group
#   set     the tokens of every session currently open on it
#   None    a reveal happened; the close is one way and this is the mark
#
# A SET rather than one token, because two people rating the same
# comparison on one machine are both legitimately blind and neither
# should invalidate the other. A reveal closes all of them at once, which
# is correct: once anybody has seen the answer key for this comparison,
# no rating of it made afterwards is blind however the page is arranged.
BLIND_SESSIONS: dict[int, set[str] | None] = {}


def blind_token_valid(group_id: int, token: str | None) -> bool:
    """Whether these ratings came from an open, unrevealed blind session.

    A TOKEN AND NOT A BOOLEAN, and the difference is the whole of this
    fix. A per-group boolean answered "is a blind session open for this
    group", which is true for every tab in the browser the moment any one
    of them opens one. A second tab replaying the same comparison
    sighted, model names and costs on screen, posted its ratings into
    somebody else's blind and they persisted blind = 1. Nothing about
    that rating was blind and nothing in the record said so.

    Four ways to arrive without one, all the same answer. No token: a
    sighted rating, which is a real rating that is simply not blind. A
    token for a session that was never opened, or one invented: nothing
    issued it. A token after the reveal: somebody has seen the answer
    key. Compared with compare_digest because a token comparison is a
    token comparison, not because the threat model here is timing.
    """
    if token is None:
        return False
    issued = BLIND_SESSIONS.get(group_id)
    if not issued:
        # Absent, revealed (None), or open with nothing outstanding.
        return False
    return any(hmac.compare_digest(known, token) for known in issued)


@app.post("/groups/{group_id}/blind", status_code=201)
async def open_blind_session(group_id: int) -> dict[str, Any]:
    """Open a blind session and hand back the anonymized comparison.

    The shuffle is issued HERE, by the server, and the response carries
    the answers with no identity attached to any of them: no model, no
    provider, no cost, no timing. That ordering is the whole feature. A
    client that fetched the identified comparison and then hid the
    identities has already painted them, and "hidden" in a browser is a
    style rule anyone can undo, plus a frame the rater may have seen.

    The response carries a TOKEN, and a rating is recorded blind only if
    it comes back bearing one. See blind_token_valid.

    THE HONEST RESIDUAL, stated here because this is where the feature
    lives and a limit documented somewhere else is a limit nobody reads.
    What the flag attests is THE PATH THE RATING TOOK: it says these
    ratings came from a view the server built without identities, on a
    session it issued and had not yet closed. It does not and cannot
    attest what the person in the chair arranged to see by other means.
    This is a localhost tool and the operator owns the process, the
    database and the browser; they can open the export in one window and
    the blind view in another, and no server-side rule can see that.

    That is a real limit and it is not a defect, because the claim being
    made is narrower than "this person did not know". It is "the bench
    did not tell them", and the bench is the only party it can speak for.
    What the boolean got wrong was different in kind: it attested a path
    the ratings had not taken at all.
    """
    ensure_rowid(group_id)
    group = store.get_group(app.state.db, group_id)
    if group is None:
        raise HTTPException(404, "no such group")
    results = [r for run in group["runs"] for r in run["results"]]
    if not results:
        raise HTTPException(
            409,
            f"comparison {group_id} has no results to rate: a blind "
            "session over nothing would record nothing.",
        )
    if BLIND_SESSIONS.get(group_id, ()) is None:
        # The close is ONE WAY, and this is where that is enforced. A
        # session that could be reopened would let a rater look, close,
        # reopen and rate, with the record still saying blind. Once
        # somebody has seen the answer key for this comparison, no later
        # rating of it can honestly claim not to have.
        #
        # None is the revealed mark and the default is an empty tuple
        # rather than None, so "never opened" and "revealed" stay
        # distinguishable: a get() defaulting to None would refuse the
        # first open every group ever gets.
        raise HTTPException(
            409,
            f"comparison {group_id} has already been revealed. A blind "
            "session cannot be reopened: somebody has seen which model "
            "wrote which answer, and a rating made after that is not a "
            "blind rating however the page is arranged.",
        )
    order = list(range(len(results)))
    random.shuffle(order)
    # secrets rather than random: the shuffle above may be predictable
    # without consequence (the labels carry no information either way),
    # while a guessable token would let a page claim a blindness the
    # server never granted it, which is the whole thing this defends.
    token = secrets.token_urlsafe(32)
    open_tokens = BLIND_SESSIONS.get(group_id) or set()
    open_tokens.add(token)
    BLIND_SESSIONS[group_id] = open_tokens
    return {
        "group_id": group_id,
        "blind": True,
        # The only thing that can make a later rating blind. Held by the
        # tab that opened this session and by nothing else.
        "token": token,
        # Label, id and answer. Everything else a card normally carries
        # is absent rather than nulled, because a null field is still a
        # field a future edit could fill in by accident.
        "cards": [
            {
                "label": BLIND_LABELS[slot],
                "result_id": results[index]["id"],
                "response_text": results[index].get("response_text"),
                "error": results[index].get("error"),
            }
            for slot, index in enumerate(order)
        ],
    }


@app.post("/groups/{group_id}/blind/reveal", status_code=200)
async def close_blind_session(group_id: int) -> dict[str, Any]:
    """Close the session, one way, and hand back the identities.

    One way because a session that could be reopened would let a rater
    look, close, reopen and rate, with the record still saying blind. The
    close is what makes every later rating on this group honest about the
    fact that somebody has now seen the answer key.

    EVERY open token on this group dies here, not only the caller's, and
    no token is required to do it. Both follow from what a reveal means:
    the answer key for this comparison is now out, and which tab asked
    for it changes nothing about that. A rating replaying a token issued
    before the reveal persists 0, which is the second half of the rule
    the token exists to enforce.
    """
    ensure_rowid(group_id)
    group = store.get_group(app.state.db, group_id)
    if group is None:
        raise HTTPException(404, "no such group")
    BLIND_SESSIONS[group_id] = None
    return {
        "group_id": group_id,
        "blind": False,
        "identities": [
            {"result_id": r["id"], "model": r["model"], "position": r.get("position")}
            for run in group["runs"]
            for r in run["results"]
        ],
    }


# ---- Phase I5: the report.

# The seed the report's intervals are built from. Fixed rather than
# random, so two loads of the same report agree: an interval that moved
# between refreshes would read as instability in the data rather than in
# the method. Recorded in every report and every export, so anyone
# holding the artifact can recompute the same bounds.
REPORT_SEED = 20260801


@app.get("/experiments/{experiment_id}/report")
async def experiment_report(
    experiment_id: int, dataset_path: str | None = Query(default=None)
) -> dict[str, Any]:
    """Aggregates for one experiment, computed at read time.

    Nothing is cached and nothing is stored. The numbers are derived from
    the rows every time, which is the same discipline as the derived
    outcome: a stored aggregate can disagree with the rows it came from,
    and the day it does there is no way to tell which is wrong.

    dataset_path is optional and it changes how well the pass rate's
    population is known, not whether there is one. Thresholds live in the
    dataset file rather than in the database, so with the file the
    eligible population is exact; without it the report recovers what it
    can from the score rows, which is a floor rather than the full
    denominator. Either way it publishes thresholds_source so the reader
    knows which they are looking at. Degrading to no pass rate at all
    was the older behavior and it threw away verdicts that were sitting
    in the database, which is a different dishonesty from the one the
    degrade was protecting against.

    When the file is given its digest is checked, because scoring one
    dataset's trials against another's thresholds is the same
    misattribution the scoring pass refuses.
    """
    ensure_rowid(experiment_id)
    experiment = store.get_experiment(app.state.db, experiment_id)
    if experiment is None:
        raise HTTPException(404, "no such experiment")
    # None rather than an empty mapping when no path was given: build_report
    # reads the difference between "nobody asked the file" and "the file
    # said nothing is declared".
    tasks = None if dataset_path is None else dataset_tasks(experiment, dataset_path)
    return build_report(experiment, **_report_inputs(experiment_id, tasks))


def dataset_tasks(experiment: dict[str, Any], dataset_path: str) -> dict[str, Any]:
    """The named file's tasks, refused unless it is the file that ran.

    The run-start precedent, and this is its third use: start, score, and
    now the two read paths. One rule in one shape, so a caller who has
    met it once has met it everywhere.
    """
    dataset = read_dataset(dataset_path)
    if dataset["digest"] != experiment["dataset_digest"]:
        raise HTTPException(
            422,
            "dataset changed since this experiment was created: recorded "
            f"{experiment['dataset_digest'][:12]}, file is "
            f"{dataset['digest'][:12]}. Reporting against different "
            "tasks would attribute one rubric's threshold to another.",
        )
    return {t["id"]: t for t in dataset["tasks"]}


def _report_inputs(
    experiment_id: int, tasks_by_id: dict[str, Any] | None
) -> dict[str, Any]:
    """Every row the report needs, read once.

    Gathered here rather than inside build_report so that function stays
    pure and can be run over an export instead of a database, which is
    what makes the export's self-sufficiency claim checkable rather than
    asserted.
    """
    db = app.state.db
    groups = store.experiment_groups(db, experiment_id)
    runs_by_group: dict[int, list[dict[str, Any]]] = {}
    result_ids: list[int] = []
    for group in groups:
        detail = store.get_group(db, group["id"])
        runs = detail["runs"] if detail else []
        runs_by_group[group["id"]] = runs
        result_ids.extend(r["id"] for run in runs for r in run["results"])
    return {
        "groups": groups,
        "runs_by_group": runs_by_group,
        "scores_by_result": store.scores_for_results(db, result_ids),
        "tasks_by_id": tasks_by_id,
        "seed": REPORT_SEED,
    }


def _export_lines(
    experiment: dict[str, Any], experiment_id: int, thresholds: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Every line of the artifact, read inside one explicit transaction.

    THE WINDOW THIS SEALS: from the first group read to the last score
    read. A write committed anywhere inside it must appear in none of
    these lines, not in the later ones, because a digest over an artifact
    that straddles two states of the database verifies perfectly while
    describing a moment that never existed.

    It used to be a `with db` block, which does not do this. The sqlite3
    connection context manager commits or rolls back at exit and never
    issues a BEGIN, and the module starts a transaction implicitly before
    a write and not before a read, so a block of pure SELECTs ran every
    one of them in its own autocommit transaction. The block read as the
    guarantee and provided none of it. store.read_snapshot issues the
    BEGIN.

    Two DIFFERENT protections, and conflating them is what hid this. The
    store's synchronous contract rules out another TASK interleaving on
    this connection, and it did that correctly the whole time. It says
    nothing about another CONNECTION, and `python -m bench.reconcile
    --apply` against a live bench is exactly that, by design: it is the
    reason connect() turns WAL on.
    """
    db = app.state.db
    out: list[dict[str, Any]] = [export_manifest(experiment, REPORT_SEED, thresholds)]
    with store.read_snapshot(db):
        for group in store.experiment_groups(db, experiment_id):
            detail = store.get_group(db, group["id"])
            if detail is None:
                continue
            rows = [
                (run, result) for run in detail["runs"] for result in run["results"]
            ]
            # Position, then result id as the tiebreak. A row with no
            # position sorts last rather than crashing the comparison,
            # and the id keeps two rows at the same position in a stable
            # order, which is what byte-identical exports need.
            rows.sort(
                key=lambda pair: (
                    pair[1].get("position") is None,
                    pair[1].get("position") or 0,
                    pair[1]["id"],
                )
            )
            score_rows = store.scores_for_results(db, [r["id"] for _, r in rows])
            for run, result in rows:
                out.append(
                    export_trial(group, run, result, score_rows.get(result["id"], []))
                )
    return out


def threshold_slice(tasks_by_id: dict[str, Any]) -> dict[str, Any]:
    """The smallest part of a dataset an aggregate cannot be re-derived
    without: which tasks declared a pass threshold, and what it was.

    Declared tasks only, and per task only the scorer's kind and cutoff.
    Prompts, references and rubrics stay out on purpose. No number the
    report publishes needs them, and an export already carries every
    prompt it actually SENT; copying the file's other columns in would
    widen what the artifact discloses for nothing.
    """
    out: dict[str, Any] = {}
    for task_id, task in tasks_by_id.items():
        spec = (task or {}).get("scorer") or {}
        if spec.get("pass_threshold") is not None:
            out[task_id] = {
                "kind": spec.get("kind"),
                "pass_threshold": spec["pass_threshold"],
            }
    return out


@app.get("/experiments/{experiment_id}/export.jsonl")
async def experiment_export(
    experiment_id: int, dataset_path: str | None = Query(default=None)
) -> StreamingResponse:
    """The whole experiment as JSONL: manifest, trials, digest.

    dataset_path is optional and takes the same terms as start, score and
    report: the digest is checked against the one recorded at creation
    and a mismatch is refused before a single byte is streamed, since an
    artifact half-written against the wrong file is worse than none.

    Supplying it embeds the threshold slice in the manifest, which is
    what makes the artifact self-sufficient for the pass rate rather than
    only for the score mean. Without it the export still re-derives a
    pass rate from its own score rows, but the eligible denominator is a
    floor. Either way the manifest carries thresholds_included, so the
    file states its own sufficiency instead of leaving a reader to assume
    it.

    Streams rather than buffers, because a MAX_TRIALS export is thousands
    of lines carrying full prompts and full responses, and holding all of
    it in memory to hand back at once is a cost with no benefit on a
    localhost tool.

    Ordered by task, then repeat, then position, so two exports of the
    same experiment are byte-identical. Line order is fixed here; key
    order within a line is fixed by export_line's sort_keys. Both are
    needed: an artifact whose digest changes between two honest exports
    of the same data cannot be cited, because the citation could never be
    checked.

    The last line is a digest over every preceding line, so a citation
    can name the artifact it cites. It is sha256 over the bytes as
    written, computed while streaming, which means the file verifies
    itself with no second pass and no separate sidecar to lose.

    private, no-store like every other dynamic body: this one carries
    every prompt and every answer in the experiment, which makes it the
    single most sensitive response the bench produces.
    """
    ensure_rowid(experiment_id)
    experiment = store.get_experiment(app.state.db, experiment_id)
    if experiment is None:
        raise HTTPException(404, "no such experiment")
    # Read and checked out here rather than inside body(), so a mismatch
    # is a 4xx with nothing written. Raising from inside the generator
    # would already have sent 200 and a content-type, and the client
    # would be left holding a truncated artifact that looks like a whole
    # one.
    thresholds = (
        None
        if dataset_path is None
        else threshold_slice(dataset_tasks(experiment, dataset_path))
    )

    # EVERY ROW READ BEFORE THE FIRST BYTE GOES OUT, inside one
    # transaction. The digest on the last line seals whatever the reader
    # received, and reading lazily while streaming would let it seal an
    # INTERVAL rather than a MOMENT: a scoring pass finishing mid-export
    # would put its rows into the later lines and not the earlier ones,
    # and the resulting artifact would be internally inconsistent while
    # its digest verified perfectly. A citation that cannot be pinned to
    # one state of the database is not a citation.
    #
    # Materializing is fine here, and bounded rather than hoped: an
    # experiment cannot exceed MAX_TRIALS rows because create_experiment
    # refuses one that would, so this holds at most 5000 trials with
    # their prompts and responses. That is a large dict and a small
    # fraction of what the same rows cost while being rendered into a
    # report, which the same process already does on demand.
    lines = _export_lines(experiment, experiment_id, thresholds)

    async def body() -> AsyncIterator[str]:
        digest = hashlib.sha256()
        for payload in lines:
            line = export_line(payload) + "\n"
            digest.update(line.encode("utf-8"))
            yield line
        # Not folded into the digest, obviously: it is the digest.
        yield (
            export_line(
                {
                    "type": "digest",
                    "algorithm": "sha256",
                    "digest": digest.hexdigest(),
                }
            )
            + "\n"
        )

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": (
                f'attachment; filename="experiment-{experiment_id}.jsonl"'
            )
        },
    )
