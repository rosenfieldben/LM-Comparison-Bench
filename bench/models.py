"""Core model-calling logic. Pure functions over an injected httpx client."""

import hashlib
import json
import logging
import math
import os
import socket
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def as_token_count(value: object) -> int | None:
    """A token count usable downstream: a non-negative int, else None.

    The never-raises contract extends to field types. A provider once
    returned "prompt_tokens": "n/a"; persisted raw, that one row made
    /compare 500 at its own response boundary and poisoned
    GET /runs/{id} into a permanent 500. bool is excluded explicitly
    because isinstance(True, int) passes. Junk is rejected rather than
    coerced: a guessed count would price a cost from fiction.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def as_text(value: object) -> str | None:
    """str or None; any other type becomes None, same contract as counts."""
    return value if isinstance(value, str) else None


def as_flag(value: object) -> bool | None:
    """A three-state flag: True, False, or None for "not reported".

    ONE RULE FOR THE WRITE AND THE READ, which is the reason this is a
    function rather than an isinstance check at each site. SQLite has no
    boolean: it stores 1, 0 and NULL, so a flag column round-trips as an
    int unless somebody converts it back, and the conversion has to
    happen in exactly the same way on both sides or the two disagree.

    THE TRAP THIS EXISTS TO CLOSE. In Python `1 == True` is True but
    `1 is True` is False, so a reader who checks identity gets the wrong
    answer from an unconverted row while a reader who checks equality
    gets the right one by luck. A truthiness check is worse again: it
    folds 0 and None together, and those are the two states this column
    was created to tell apart.

    None for anything that is not a bool or an int in (0, 1), which is
    the same isinstance discipline _ingest_usage applies at the wire.
    A poisoned cell degrades to "not reported" rather than to a
    confident True, because SQLite's column affinity will accept a
    string into an INTEGER column and a later reader must not be told
    something the wire never said.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def as_metric(value: object) -> float | None:
    """A finite float measurement or None; bools and junk become None.

    Used by the store's repair-on-read: measurement columns written
    before ingestion normalization may carry non-numeric junk.
    """
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def as_money(value: object) -> float | None:
    """A usable money amount: finite, non-negative, else None.

    The F.1 money rule as a function, so every place that ingests a
    currency figure applies it identically. Stricter than as_metric by the
    sign: a negative charge is not a measurement that happens to be below
    zero, it is a value no billing path should produce, and treating it as
    real would let it subtract from an accumulated total and quietly
    disable the spend ceiling. A NaN would do the same by making every
    comparison false, which is why finiteness is checked here rather than
    trusted from the payload.
    """
    amount = as_metric(value)
    if amount is None or amount < 0:
        return None
    return amount


# Env-overridable as a test seam so the browser harness can point the
# real app at a stub upstream; not a configuration feature.
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
MODELS_URL = os.environ.get("MODELS_URL", "https://openrouter.ai/api/v1/models")

# The PER-ENDPOINT capability listing, which is a different publication
# from the model-level one MODELS_URL returns and can disagree with it.
# Pinned against
# https://openrouter.ai/docs/api-reference/list-endpoints-for-a-model,
# read 2026-08-12: GET /api/v1/models/:author/:slug/endpoints, whose
# data.endpoints[] each carry provider_name and their own
# supported_parameters.
#
# MEASURED, because "can disagree" is the whole reason this exists. On
# 2026-08-12 nvidia/nemotron-3-ultra-550b-a55b published three
# endpoints advertising 18, 12 and 17 parameters respectively, differing
# in which ones. A model-level aggregate is a union over hosts, so it
# proves nothing about the ONE host a strict pin selects.
ENDPOINTS_URL = os.environ.get(
    "ENDPOINTS_URL", "https://openrouter.ai/api/v1/models/{model}/endpoints"
)
GENERATION_URL = os.environ.get(
    "GENERATION_URL", "https://openrouter.ai/api/v1/generation"
)

# Reaching OpenRouter should be fast; a slow connect is a real failure.
CONNECT_TIMEOUT_S = 10.0

# Extended-budget reasoning can legitimately sit silent for minutes
# between visible bytes; five minutes of true wire silence is failure
# by any standard.
STREAM_READ_TIMEOUT_S = 300.0

# Non-streaming path. The single read covers the whole completion,
# including all reasoning time, so it deserves at least the stream's
# per-gap tolerance. At 180 it was lower than the stream's 300, so an
# extended non-streaming run could fail at 181s where the streamed route
# survived; raised to match. Extended non-streaming runs stay more
# timeout-prone by nature: the stream resets its clock on every chunk and
# can run far longer overall, while this is one uninterrupted read. The
# browser always streams; /compare is the scripting path.
COMPLETION_READ_TIMEOUT_S = 300.0

# Streaming: connect fast, tolerate long silent gaps between chunks.
STREAM_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S, read=STREAM_READ_TIMEOUT_S, write=30.0, pool=10.0
)

# Non-streaming: connect fast, one long read for the whole completion.
COMPLETION_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S, read=COMPLETION_READ_TIMEOUT_S, write=30.0, pool=10.0
)

# Reasoning models spend thousands of hidden tokens before visible
# output and max_tokens covers both, so the budget must dwarf a
# plausible reasoning burn. Two named tiers instead of a free integer:
# everyday prompts and hard problems are the two real regimes of use,
# a free field invites typos with dollar consequences, and history
# stays legible when every run carries one of two labels. Standard
# keeps worst-case cost per call around a dollar on the priciest
# models; extended exists for hard problems where reasoning empties
# the standard budget before any visible answer appears.
BUDGET_STANDARD = 16384
BUDGET_EXTENDED = 65536

# Boot must not hang on pricing; the bench works offline, cost display
# is the only thing a failed fetch costs.
PRICES_TIMEOUT_S = 10.0

# OpenRouter's default routing is price-weighted, which sends
# open-weight models to their cheapest hosts, and the cheapest hosts
# are the flakiest. Sorting by throughput biases routing to the
# serious hosts at somewhat higher cost. Unlike a reasoning parameter
# this does not alter what the model itself does, only who serves it;
# and since quantization varies by host, it also stabilizes WHAT is
# being measured, not just how reliably it answers.
PROVIDER_PREFS = {"sort": "throughput"}

# Data-handling routing, merged into the provider block above per policy.
# Field names pinned against OpenRouter's provider-routing documentation
# at https://openrouter.ai/docs/features/provider-routing, read 2026-07-25,
# not from memory: data_collection takes "allow" (the documented default,
# meaning providers that store user data non-transiently and may train on
# it are eligible) or "deny" (route only to providers that do not collect
# user data), and zdr is a boolean that restricts routing to zero data
# retention endpoints. Both are keys of the same top-level provider object
# the throughput sort already uses.
#
# zdr sends data_collection deny as well, which the documentation does not
# require. The two flags govern different things (retention versus
# collection and training eligibility), and a mode chosen for confidential
# prompts that still permitted training-eligible routing would be a trap.
# The stricter reading is the safe one to be wrong about.
DATA_POLICY_PREFS: dict[str, dict[str, Any]] = {
    # Today's payload, unchanged, so the default costs nothing and says
    # nothing it cannot back up.
    "standard": {},
    "deny": {"data_collection": "deny"},
    "zdr": {"zdr": True, "data_collection": "deny"},
}


# Routing modes, and what each one puts in the provider object. Pinned
# against OpenRouter's provider-routing documentation at
# https://openrouter.ai/docs/guides/routing/provider-selection#provider-sorting
# read 2026-07-30, not from memory: sort takes "price", "throughput" or
# "latency", and "If you have sort or order set in your provider
# preferences, load balancing will be disabled." So omitting sort is not a
# neutral spelling of a default, it is the one way to ask for OpenRouter's
# own price-weighted load balancing, which is why "default" maps to no key
# at all rather than to a value.
#
# throughput is the bench's default because it is what the bench already
# sent; see PROVIDER_PREFS above for why that choice was made. Keeping it
# the default is what lets a blank controls set produce today's payload
# byte for byte.
ROUTING_PREFS: dict[str, dict[str, Any]] = {
    "throughput": PROVIDER_PREFS,
    "price": {"sort": "price"},
    "default": {},
}

ROUTING_DEFAULT = "throughput"


def provider_preferences(
    data_policy: str, routing: str = ROUTING_DEFAULT
) -> dict[str, Any]:
    """The provider block for one data policy and routing mode.

    The two merge into one object because OpenRouter gives them one object:
    sort and the data-handling keys are siblings. Merging in this order
    means the privacy policy is written last and can never be displaced by
    a routing choice, which matters because the failure would be silent and
    would understate what the payload promised about the prompt.

    Raises KeyError on an unknown policy or routing mode, deliberately:
    both are validated at their boundaries (bench.main._parse_data_policy
    at boot, the request model per request), and a typo would otherwise
    silently send something other than what the run row claimed.
    """
    return {**ROUTING_PREFS[routing], **DATA_POLICY_PREFS[data_policy]}


# The controls that shape generation, and where each one goes on the wire.
# Field names and bounds pinned at implementation time against OpenRouter's
# current documentation, not from memory:
#
#   https://openrouter.ai/docs/api_reference/parameters  read 2026-07-30
#   temperature: float 0.0 to 2.0, default 1.0
#   top_p:       float 0.0 to 1.0, default 1.0
#   seed:        integer. "If specified, the inferencing will sample
#                deterministically, such that repeated requests with the
#                same seed and parameters should return the same result.
#                Determinism is not guaranteed for some models."
#   All three are top-level keys of the chat-completions body.
#
#   https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
#   read 2026-07-30. reasoning is a top-level OBJECT, not a scalar, and
#   effort is a key inside it. The documented values are max, xhigh, high,
#   medium, low, minimal and none; the bench offers low, medium and high
#   because three named tiers are what a comparison can hold constant
#   without inviting a per-model guess.
#
# What happens on a model that does not support a control: OpenRouter
# documents the silent path, so this is recorded rather than assumed.
# "With the default routing strategy, providers that don't support all the
# LLM parameters specified in your request can still receive the request,
# but will ignore unknown parameters."
# (.../provider-selection#requiring-providers-to-support-all-parameters,
# read 2026-07-30.) So an unsupported sampling parameter is dropped by the
# provider, not rejected, and that is documented behavior rather than a
# defect this bench is hiding. For reasoning effort the mapping is stronger
# still: "For models that only support reasoning.max_tokens, the effort
# level will be set based on the percentages above", so effort is
# translated rather than dropped there. require_parameters could force a
# hard failure instead, and is deliberately not sent: it changes which
# providers are eligible, which would silently change what is being
# measured, and choosing that tradeoff is not this phase's business.
#
# The record is what makes the silence survivable. request_json holds the
# exact payload per result, so a run whose temperature a provider ignored
# is still a run that can be shown to have asked for it.
_SAMPLING_KEYS = ("temperature", "top_p", "seed")


# How much of a completion budget thinking may consume before the bench
# stops it and leaves the rest for a visible answer.
#
# THE INCIDENT. A production comparison on document prompts reasoned
# until the completion cap closed on it: billed in full, between $3.38
# and $3.50 per card, with ZERO visible output. The generation record is
# unambiguous about where the budget went, tokens_completion 21350
# against native_tokens_reasoning 21350, and the card carried no answer
# at all. Row 694 of the same database is the healthy contrast: 2944
# reasoning tokens, a real answer, end_turn.
#
# NOT A PROPERTY OF ONE MODEL, which is why nothing here names one.
# Exhaustion is what happens when a model thinks hard against a
# completion cap, and any model that reasons can do it on a hard enough
# prompt. The reservation is therefore a function of the BUDGET and of
# nothing else; there is a test asserting no model name appears in this
# logic.
#
# ONE HALF, and the arithmetic rather than the round number. The
# extended tier is 65536, so thinking is capped at 32768 and a maximal
# thinker still leaves 32768 for the answer, which is four times the
# whole standard tier. The standard tier is 16384, capped at 8192, and
# 8192 visible tokens is a long answer by any measure. Lower would start
# refusing thought that a hard prompt legitimately needs; higher shrinks
# the guarantee this exists to make. Half is the point where neither
# half of the budget can starve the other.
REASONING_BUDGET_SHARE = 0.5

# THE DOCUMENTED EFFORT LADDER, as share of max_tokens. Pinned against
# the reasoning-tokens guide's "Reasoning Effort Level" section,
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
# read 2026-08-12, quoted in the order the page lists them: max
# "approximately 95% of max_tokens", xhigh "Same allocation as max
# (approximately 95%)", high "approximately 80%", medium "approximately
# 50%", low "approximately 20%", minimal "approximately 10%", and
# "'effort': 'none' - Disables reasoning entirely".
#
# Ratios rather than an ordered list of names, because the comparison
# this table exists for is against REASONING_BUDGET_SHARE, which is a
# number. A future share of 0.25 would move the boundary to low without
# anybody editing a tier list, which is the point.
# THE SMALLEST REASONING BUDGET ANY PINNED PROVIDER WILL ACCEPT, and
# the rule that makes a small budget unsatisfiable rather than merely
# tight.
#
# NAMED FOR WHAT IT IS RATHER THAN FOR WHO PUBLISHED IT, and that is not
# cosmetic. A first draft called this ANTHROPIC_REASONING_MINIMUM and
# the universality tripwire failed the moment the constant was read from
# reasoning_reservation, correctly: a vendor's name had entered
# executable code. The floor is applied to every request regardless of
# route, so the name says that and the quote below says whose behaviour
# the number was measured from. Prose about a contract may name a
# vendor; a line that runs may not.
#
# Both rules quoted from
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
# read 2026-08-12:
#
#   "When using the reasoning.max_tokens parameter, that value is used
#   directly with a minimum of 1024 tokens."
#
#   "Important: max_tokens must be strictly higher than the reasoning
#   budget to ensure there are tokens available for the final response
#   after thinking."
#
# A minimum turns a request into a demand the sender did not make. Ask
# for 256 and the provider hears 1024, and if the outer budget is not
# strictly above that, nothing can satisfy the pair.
#
# ONE NUMBER FOR THE WHOLE FILE, because the judge's 512 budget and a
# per-model clamp reach the same wall from opposite directions and must
# not be two rules that can drift apart.
PROVIDER_REASONING_MINIMUM = 1024

EFFORT_SHARES = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
    "xhigh": 0.95,
    "max": 0.95,
}

# Pinned against OpenRouter's reasoning-tokens guide at
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
# re-read in full 2026-08-12. This block was rewritten at R4 after that
# re-read contradicted two things the first version asserted; both
# corrections are below and both are the docs' fault rather than a
# change of mind.
#
# THE FIELD. reasoning is a top-level object and the cap is its
# max_tokens key: "'max_tokens': 2000, // Specific token limit
# (Anthropic-style)".
#
# NOT BOTH, WHICH THE PAGE SAYS AND THEN UNSAYS. The schema comment
# introduces the two keys as "One of the following (not both)". Thirty
# lines later the per-model discovery section describes
# supports_max_tokens as a reason to "send 'reasoning.max_tokens'
# instead of (or alongside) 'reasoning.effort'". The page therefore
# forbids the combination in one place and sanctions it in another,
# names no winner if both arrive, and documents no error for the
# collision.
#
# So the bench takes the CONSERVATIVE reading and never sends both, and
# the reason is not the schema comment: it is that an undefined
# resolution is not something to discover in production at $3.50 a card.
# When the two would collide the OPERATOR'S CONTROL RIDES and the
# reservation stands down. Effort is a declared part of the experiment,
# this is the bench's own default, and a default that overwrote a
# declaration is the one thing rule one forbids.
#
# THAT REQUEST IS THEN UNPROTECTED BY BOTH HALVES OF THIS FIX, which an
# earlier version of this comment denied. It claimed the card's
# indicator covers the gap "because it reads the tokens that came back
# rather than the field that went out". Read the arithmetic: an effort
# of high allocates approximately 80% of max_tokens per the vendor's own
# table, and REASONING_SHARE_EXHAUSTED is 0.9, so
# reasoning_ate_the_output(16384, 13107) is False and the indicator
# provably never fires at the share a declared effort produces. It is a
# comment that justified a decision by naming a safety net that is not
# under it.
#
# The decision still stands, on its own merits: not sending an
# undefined combination and letting the operator's declaration ride is
# the defensible reading of rule one under real ambiguity, and the
# exposure is bounded to requests where an operator deliberately set an
# effort. But it is exposure, and it is named here rather than
# explained away.
#
# HOW FEW ROUTES TAKE A REAL BUDGET, measured rather than assumed, and
# this is the correction that matters most. GET /api/v1/models on
# 2026-08-12 returned 406 models. 275 publish "reasoning" in
# supported_parameters and 276 carry a reasoning descriptor object, but
# only TEN of those set supports_max_tokens true. A comment claiming
# this caps thinking on every route would be false on roughly 96% of
# the reasoning models in the catalog.
#
# WHAT HAPPENS ON THE OTHER 266, and why the reservation is still worth
# sending. It is TRANSLATED, not dropped: "For models that only support
# 'reasoning.effort' (see below), the 'max_tokens' value will be used to
# determine the effort level." The effort table puts medium at
# "approximately 50% of max_tokens", which is this constant's share, so
# a half-budget cap lands on the effort bucket that means the same
# thing. The intent survives the translation. The precision does not,
# and "approximately" is the vendor's word, not a hedge added here.
#
# Two named routes are looser still, both from the same page. Anthropic
# applies "budget_tokens = max(min(max_tokens * {effort_ratio}, 128000),
# 1024)", so a small budget floors at 1024 rather than at its share. And
# for Gemini 3, "Google internally maps this budget value to a
# 'thinkingLevel', so you will not get precise token control."
#
# Finally, a provider that does not support the parameter at all ignores
# it outright, per the provider-selection page: providers "can still
# receive the request, but will ignore unknown parameters".
#
# THE HONEST SUMMARY, which the README states in the same words: on ten
# models this is a hard cap; on most it is a strong hint that means the
# right thing approximately; on some it is nothing. That is why R2's
# label and R4's indicator read the tokens that came BACK. They are not
# a fallback for a rare case. They are the coverage for the common one.
#
# WHY A CAP AND NOT AN EFFORT for the bench's own default: an effort is
# a word whose token cost is the provider's to decide, and the whole
# defect is a token count nobody bounded. A cap is the thing that can be
# checked against the budget it is a fraction of.
#
# Reasoning tokens are charged either way, which is the other half of
# why this is worth sending: "Reasoning tokens are considered output
# tokens and charged accordingly."


def reasoning_reservation(
    max_tokens: int,
    controls: Mapping[str, Any] | None = None,
    may_send_reasoning_cap: bool = False,
) -> dict[str, Any]:
    """The reasoning cap for a request at this budget, or nothing.

    Returns the fragment to merge into a payload, so a caller cannot
    forget either of the two rules below by assembling the field itself.

    EMPTY UNLESS THE MODEL ALREADY THINKS, which is the guard that keeps
    this a bound rather than a purchase. A reasoning object with a cap
    and no enabled key infers enabled true, so on a model whose catalog
    entry says thinking is off, an unprompted cap would switch thinking
    ON and start billing for it. may_send_reasoning_cap comes from the
    catalog's own capability flags; see fetch_catalog for the pinned
    wording and the measured counts.

    The default is False, deliberately, so a caller that does not know
    sends nothing. That is rule one's answer for an unknown, and it
    makes the protective case the one that has to be asked for rather
    than the one that happens by omission.

    EMPTY WHEN THE OPERATOR DECLARED AN EFFORT, per the pinned page; see
    REASONING_BUDGET_SHARE for why the declaration wins over the
    default.

    EMPTY WHEN THE ARITHMETIC CANNOT BE SATISFIED, which is the rule
    that used to be a special case at the judge and is now general.

    A provider minimum turns a small request into a larger demand: ask
    for 256 and the pinned floor makes it 1024. Two things then go
    wrong, and both are checked, because a future share could reach
    either one first.

      The share falls BELOW the minimum. The request that arrives is
      not the request that was sent, and the visible room this function
      exists to reserve is not the room that will be left. At an outer
      budget of 1200 a half share asks for 600 and the provider hears
      1024, leaving 176 visible tokens instead of the promised 600.

      The outer budget is NOT STRICTLY GREATER than what the provider
      will use. Nothing can satisfy that pair; the contract says so in
      as many words.

    KEYED ON THE ARITHMETIC, NEVER ON THE CALL SITE. The judge's 512
    token budget and a per-model clamp of 1024 are the same failure
    reached from opposite directions, so they are one rule with two
    instances rather than two rules that can drift apart. The judge
    calls this function like everybody else and gets {} from it.

    THE CLAMP IS WHY THIS IS REACHABLE at all: effective_budget lowers
    the requested tier to any published completion cap, and the catalog
    accepts any cap above zero, so a budget below 2048 comes from a real
    catalog entry rather than from a hypothetical.

    int() truncates rather than rounds, so the cap is never a token
    above its share of the budget. At the two real tiers the division is
    exact anyway; the floor matters only if a future tier is odd.
    """
    if not may_send_reasoning_cap:
        return {}
    if controls and controls.get("effort") is not None:
        return {}
    want = int(max_tokens * REASONING_BUDGET_SHARE)
    # What the provider will actually use, after its own floor.
    effective = max(want, PROVIDER_REASONING_MINIMUM)
    if want < PROVIDER_REASONING_MINIMUM or max_tokens <= effective:
        return {}
    return {"reasoning": {"max_tokens": want}}


# ---- The underlying-model estimand.
#
# Everything in this block applies ONLY when an experiment declared
# estimand_mode "underlying_model". The routed-service estimand is the
# default and must keep sending exactly the bytes it sent before this
# block existed; a tombstone asserts that.
#
# Pinned against OpenRouter's provider-routing documentation at
# https://openrouter.ai/docs/guides/routing/provider-selection, read
# 2026-08-02, not from memory:
#
#   require_parameters (boolean, default false): "restrict to providers
#   that support all parameters in your request". The routed-service
#   estimand deliberately does not send it, for the reason written at
#   _SAMPLING_KEYS above: it changes which providers are eligible, which
#   silently changes what is being measured. Under this estimand that
#   change is the entire point. A claim about a MODEL that was served by
#   a provider which quietly dropped the temperature is a claim about a
#   different configuration than the row records.
#
#   order (array of provider slugs), with allow_fallbacks false: try
#   these and nothing else. Without allow_fallbacks false the order is a
#   preference, so a pinned run could be served by anyone and the record
#   would name a pin that did not hold. The pin exists to narrow the
#   population; a pin that can be ignored narrows nothing.
STRICT_PREFS: dict[str, Any] = {"require_parameters": True}

# The documented quantization filter, and its documented values. Pinned
# against https://openrouter.ai/docs/guides/routing/provider-selection,
# read 2026-08-04: "quantizations string[] - List of quantization levels
# to filter by (e.g. ["int4", "int8"])", with the levels listed as int4,
# int8, fp4, mxfp4, nvfp4, fp6, fp8, mxfp8, fp16, bf16, fp32 and unknown.
#
# Strict mode's reason for wanting it: quantization varies by host and
# changes what the weights actually are, so two runs of "the same model"
# served at bf16 and at int4 are not measuring the same artifact. The
# throughput sort the routed path uses biases toward serious hosts and
# was never a quantization control; it does not pin one and never did.
QUANTIZATION_LEVELS = (
    "int4",
    "int8",
    "fp4",
    "mxfp4",
    "nvfp4",
    "fp6",
    "fp8",
    "mxfp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
)


def strict_provider_preferences(
    base: Mapping[str, Any],
    pin: str | None = None,
    quantizations: list[str] | None = None,
) -> dict[str, Any]:
    """The provider block for one strict-mode trial.

    base is whatever the routed path would have sent (the boot data
    policy merged with this request's routing mode), so the privacy
    promise cannot be displaced by the estimand: it is written first and
    the strict keys are additive.

    A pin is a hard restriction rather than a preference, which is why
    allow_fallbacks rides with it and never without it. Sending
    allow_fallbacks false with no order would forbid fallback from a
    provider nobody named, which is not a thing anyone asked for.
    """
    out = {**base, **STRICT_PREFS}
    if pin is not None:
        out["order"] = [pin]
        out["allow_fallbacks"] = False
    if quantizations:
        out["quantizations"] = list(quantizations)
    return out


def normalized_provider_slug(name: str) -> str:
    """A provider name as the documented field wants it.

    order takes "provider slugs", and the documentation's own example is
    ["anthropic", "openai"]: lowercase, and not the display names that
    appear in a catalog or a report. A pin written "Together" is what a
    person would naturally type after reading a provider column, and
    sending it unchanged risks a filter that matches nothing, which under
    allow_fallbacks false is a run that fails rather than a run served by
    somebody else. Normalizing at the boundary means the recorded pin and
    the sent pin are the same string, which is the property a report
    citing a pin depends on.

    Pinned against
    https://openrouter.ai/docs/guides/routing/provider-selection, read
    2026-08-04.
    """
    return name.strip().lower()


# Which supported_parameters token each control needs. The check runs
# against the names the payload will actually carry, which is why
# effort maps to "reasoning": control_payload emits it as a reasoning
# object, and "effort" is a key inside that object rather than a
# parameter name OpenRouter publishes.
#
# effort maps to "reasoning" and the check therefore verifies AGGREGATE
# reasoning support and nothing finer. That limit is real and is stated
# rather than glossed: per
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
# read 2026-08-04, "For models that only support reasoning.max_tokens,
# the effort level will be set based on the percentages above", so a
# provider advertising `reasoning` may be translating the requested
# effort into a token allocation rather than honouring the named tier.
# Strict mode can check that reasoning is supported at all; it cannot
# check that "high" means the same thing everywhere, and it does not
# claim to.
#
# max_tokens is here even though no control sets it, because every
# payload this bench builds carries one. Under require_parameters a
# provider that does not support max_tokens is ineligible, so a model
# whose catalog entry omits it has no eligible provider for any request
# the bench can make, and saying that at creation is better than
# discovering it at trial one.
#
# routing and system are absent on purpose. Routing is a key of the
# provider object rather than a model parameter, and a system message is
# a message.
CONTROL_PARAMETERS: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "effort": "reasoning",
}


async def fetch_endpoints(client: httpx.AsyncClient, model: str) -> dict[str, Any]:
    """The per-endpoint capability listing for one model, or an honest
    failure.

    Never raises, the same contract fetch_catalog carries. Returns
    fetched False when the listing could not be obtained, which callers
    must treat as absence of evidence rather than absence of support:
    strict mode refuses on it, exactly as it refuses on a model the
    catalog does not list.

    Only the two fields a decision is made from are kept. Anything else
    the endpoint publishes is not this function's business, and a parser
    that carried it would invite somebody to make a second decision from
    a field nobody checked.
    """
    out: dict[str, Any] = {"fetched": False, "endpoints": []}
    try:
        response = await client.get(
            ENDPOINTS_URL.format(model=model), timeout=PRICES_TIMEOUT_S
        )
        if response.status_code != 200:
            return out
        data = response.json()["data"]
        entries = data["endpoints"]
        if not isinstance(entries, list):
            return out
    except (httpx.HTTPError, ValueError, LookupError, TypeError):
        return out
    endpoints = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        supported = entry.get("supported_parameters")
        endpoints.append(
            {
                "provider": _as_label(entry.get("provider_name")),
                # None when the endpoint did not publish a list, which is
                # not the same as publishing an empty one.
                "supported_parameters": (
                    sorted({x for x in supported if isinstance(x, str)})
                    if isinstance(supported, list)
                    else None
                ),
            }
        )
    out["fetched"] = True
    out["endpoints"] = endpoints
    return out


def endpoint_supports_reasoning(
    listing: Mapping[str, Any], provider: str | None
) -> bool | None:
    """Whether the PINNED endpoint advertises the reasoning parameter.

    Three-valued on purpose. True and False are answers; None means the
    listing could not answer, either because it was never fetched, or
    because no endpoint matched the pin, or because the matched endpoint
    published no parameter list. Callers must not collapse None into
    False silently: strict mode's rule is that absence of evidence is
    refused rather than assumed, and the two need different sentences.

    WHY THIS CANNOT READ THE MODEL AGGREGATE. A strict pin selects ONE
    host with allow_fallbacks false. The model-level
    supported_parameters is a union across hosts, so it can advertise a
    parameter that the pinned host does not accept, and under
    require_parameters an unaccepted parameter does not degrade, it
    empties the provider pool. Vouching for a pinned endpoint with an
    aggregate is the specific mistake this function exists to make
    impossible.

    Matching is on the normalized slug both sides already use, so a pin
    recorded as "deepinfra" finds a provider_name of "DeepInfra".
    """
    if not listing.get("fetched") or provider is None:
        return None
    want = normalized_provider_slug(provider)
    for endpoint in listing.get("endpoints") or ():
        name = endpoint.get("provider")
        if name is None or normalized_provider_slug(name) != want:
            continue
        supported = endpoint.get("supported_parameters")
        if supported is None:
            return None
        return "reasoning" in supported
    return None


def missing_parameters(
    supported: list[str] | None, controls: Mapping[str, Any] | None
) -> list[str]:
    """Which of this request's parameters the model does not support.

    supported is the catalog's supported_parameters for one model, and
    None means the catalog did not say. That is NOT treated as "supports
    everything": the caller refuses on it. A strict mode that skips its
    check when the data is missing is a strict mode that isn't, which is
    worse than none, because the run is then labelled with a guarantee
    nobody verified.

    This function reports; the caller decides. It returns the names in
    the order a reader would find them useful, sorted, and never raises.
    """
    if supported is None:
        return []
    have = set(supported)
    needed = {"max_tokens"}
    for key, parameter in CONTROL_PARAMETERS.items():
        if (controls or {}).get(key) is not None:
            needed.add(parameter)
    return sorted(needed - have)


def control_payload(controls: Mapping[str, Any] | None) -> dict[str, Any]:
    """The payload fragment for the controls that were actually set.

    Never sends a default. A control the caller left out does not appear in
    the returned mapping at all, so it does not appear on the wire, so the
    provider applies its own default and the record can say honestly that
    the bench did not choose one. With nothing set this returns an empty
    mapping and the payload is byte for byte what the bench sent before
    these controls existed.

    Deliberately excludes the system prompt and the routing mode. Those two
    are not top-level scalars: the system prompt is a message and routing
    is a key of the provider object, so they are built by control_messages
    and provider_preferences. Keeping this function to the keys it can
    place directly is what makes its output a safe ** merge.
    """
    if not controls:
        return {}
    out: dict[str, Any] = {}
    for key in _SAMPLING_KEYS:
        value = controls.get(key)
        if value is not None:
            out[key] = value
    effort = controls.get("effort")
    if effort is not None:
        out["reasoning"] = {"effort": effort}
    return out


def control_messages(
    prompt: str, controls: Mapping[str, Any] | None
) -> list[dict[str, str]]:
    """The message list, the system prompt leading when one was set.

    A system message is prepended rather than folded into the user turn:
    the two roles are what the models are trained to distinguish, and
    concatenating them would make the comparison measure something else.
    With no system prompt this is the single user message the bench has
    always sent, which is half of why a blank controls set reproduces
    today's payload exactly.
    """
    user = {"role": "user", "content": prompt}
    system = (controls or {}).get("system")
    if not system:
        return [user]
    return [{"role": "system", "content": system}, user]


# Extended-budget runs go silent for minutes while providers reason,
# and NAT/middlebox idle timers cull flows that move no bytes. That
# culling surfaced as sequential failures with mixed ReadError and
# stall signatures partway through a lineup. OpenRouter's SSE comment
# keepalives evidently do not flow reliably during deep provider-side
# reasoning, so the fix sits below the application layer: OS-level TCP
# probes keep the flow visibly alive when no application bytes move.
# 30s stays comfortably under every common NAT idle window while
# adding negligible traffic.
KEEPALIVE_IDLE_S = 30
KEEPALIVE_INTERVAL_S = 30


def keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Socket options enabling TCP keepalive on the current platform.

    The idle-before-first-probe constant is platform-named: TCP_KEEPIDLE
    on Linux, TCP_KEEPALIVE on macOS. This repo runs on both a Mac and
    Linux sandboxes, so hasattr picks whichever exists rather than
    hardcoding one and silently doing nothing on the other.
    """
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPIDLE"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, KEEPALIVE_IDLE_S))
    elif hasattr(socket, "TCP_KEEPALIVE"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, KEEPALIVE_IDLE_S))
    if hasattr(socket, "TCP_KEEPINTVL"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, KEEPALIVE_INTERVAL_S))
    return options


async def fetch_catalog(client: httpx.AsyncClient) -> dict[str, Any]:
    """One boot-time snapshot of OpenRouter's model catalog.

    Returns {"fetched": bool, "models": [...], "prices": {...}} where
    models entries carry id, name, context_length, prompt_price,
    completion_price and max_completion_tokens (missing fields degrade
    to None, never breaking the whole catalog) and prices is the
    {model_id: {prompt, completion}} map cost_usd consumes. One
    upstream call feeds both.
    Never raises: this runs at startup and a catalog outage must not
    stop the bench from booting; fetched=false is how the frontend
    tells an offline boot from an empty catalog.
    """
    offline = {"fetched": False, "models": [], "prices": {}, "digest": None}
    try:
        response = await client.get(MODELS_URL, timeout=PRICES_TIMEOUT_S)
        if response.status_code != 200:
            return offline
        # The bytes, hashed before parsing. catalog_snapshot_at already says
        # WHEN the catalog was read; this says WHICH catalog, so two runs
        # costed a minute apart can be told apart rather than merely dated.
        # Over the raw body rather than the parsed structure, because the
        # question is what arrived, not what this parser made of it.
        digest = hashlib.sha256(response.content).hexdigest()
        entries = response.json()["data"]
        if not isinstance(entries, list):
            return offline
    except (httpx.HTTPError, ValueError, LookupError, TypeError):
        return offline

    models = []
    prices = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        model = {
            "id": entry["id"],
            "name": entry.get("name") if isinstance(entry.get("name"), str) else None,
            # How many tokens the model can hold, prompt and completion
            # together. Read by the composed-prompt ceiling, which
            # refuses a comparison no member could actually receive.
            #
            # Pinned against the live listing at
            # https://openrouter.ai/api/v1/models read 2026-08-09, where
            # context_length is a top-level integer on every one of the
            # 400 entries. int-or-None rather than trusted, for the
            # reason every field here is: a string or a null from a
            # future shape must read as "the catalog did not say", which
            # the ceiling skips, and never as a number.
            "context_length": (
                entry.get("context_length")
                if isinstance(entry.get("context_length"), int)
                else None
            ),
            "prompt_price": None,
            "completion_price": None,
            "max_completion_tokens": None,
            # False until the catalog says otherwise; see the parse below.
            "may_send_reasoning_cap": False,
            # What the underlying-model estimand checks against. None
            # means the catalog did not say, which strict mode treats as
            # "cannot check" rather than as "supports everything": see
            # missing_parameters. Pinned against OpenRouter's model
            # listing, read 2026-08-02, where each model carries
            # supported_parameters as an array of parameter names, the
            # union over the providers that serve it. Not split across
            # lines, because a wrapped URL is a URL nobody can click:
            # https://openrouter.ai/docs/api-reference/list-available-models
            "supported_parameters": None,
            # Which kinds of input this model accepts, which is what the
            # native attachment mode checks against. None means the
            # catalog did not say, treated as "cannot check" rather than
            # "accepts everything", exactly as supported_parameters is.
            #
            # Pinned against the live model listing at
            # https://openrouter.ai/api/v1/models read 2026-08-08, where
            # every one of the 400 entries carries
            # architecture.input_modalities as an array of strings, for
            # example ["text", "image", "video", "file", "audio"] on a
            # multimodal model and ["text"] on a text-only one. The same
            # field appears in the documented response shape at
            # https://openrouter.ai/docs/guides/overview/multimodal/image-generation
            "input_modalities": None,
        }
        architecture = entry.get("architecture")
        if isinstance(architecture, dict):
            modalities = architecture.get("input_modalities")
            if isinstance(modalities, list):
                # Strings only, same degrade-in-place rule as
                # supported_parameters: one malformed element must not
                # cost the whole entry. An empty list stays a list,
                # because "this model accepts nothing" is a real if
                # improbable answer and is not the same as unknown.
                model["input_modalities"] = sorted(
                    {x for x in modalities if isinstance(x, str)}
                )
        supported = entry.get("supported_parameters")
        if isinstance(supported, list):
            # Strings only, so one malformed element degrades itself
            # rather than the entry. An empty list is a real answer (this
            # model takes no optional parameters) and stays a list, which
            # is why the emptiness is not folded back into None.
            model["supported_parameters"] = sorted(
                {x for x in supported if isinstance(x, str)}
            )
        # WHETHER THIS MODEL THINKS WITHOUT BEING ASKED, which decides
        # whether the visible-output reservation may ride unprompted.
        #
        # Pinned against the reasoning-tokens guide's "Discovering
        # per-model reasoning options" section,
        # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
        # read 2026-08-12: "default_enabled: Default on/off state when
        # the user has not set reasoning.enabled" and "mandatory: When
        # true, hide disable controls and do not send effort: 'none' --
        # the model rejects it."
        #
        # THE REASON THIS FLAG HAD TO EXIST, and it is MEASURED rather
        # than reasoned from the documentation. The request schema says
        # of the enabled key: "Default: inferred from 'effort' or
        # 'max_tokens'", which implies a cap sent alone does not merely
        # BOUND thinking but turns it ON. That implication was tested
        # against the wire before this gate was trusted.
        #
        # THE PROBE, two live calls to google/gemma-4-31b-it served by
        # DeepInfra on 2026-08-12, a model whose descriptor publishes
        # default_enabled false and mandatory false:
        #
        #   gen-1786546333-86V1vJMk7JsWwwPhajoN, control, no field
        #     completion 35, reasoning_tokens 0, cost 1.629e-05
        #
        #   gen-1786546398-Z4GhaMYohdT9FWGFiYhc, the identical call
        #   with ONLY reasoning: {"max_tokens": 256} added
        #     completion 338, reasoning_tokens 312, cost 0.00013182,
        #     plus a full visible chain of thought in message.reasoning
        #     with reasoning_details blocks
        #
        # Both finish_reason stop. Zero reasoning tokens without the
        # field, 312 with it, and 8.1 times the cost for the same short
        # question. The cap alone enables thinking. Nothing else
        # differed between the two calls, so this is not an inference
        # about a schema comment; it is the observed behaviour of the
        # one field this code decides whether to send.
        #
        # Both usage blocks are in the tree verbatim, at
        # tests/fixtures/probe_reasoning_enable.json, and a test replays
        # them through _ingest_usage rather than restating their
        # arithmetic.
        #
        # WHAT THAT WOULD HAVE COST UNGATED. On a model whose default is
        # off, an unprompted cap starts buying reasoning tokens that were
        # never being bought, on every run, in an instrument whose
        # product is a comparison between models. Measured against the
        # live catalog on 2026-08-12: of 406 models, 23 publish
        # default_enabled false without mandatory, and they are the
        # frontier lineup.
        #
        # TWO MORE DIMENSIONS OF THE SAME CONTRACT, added after a review
        # showed the gate above was reading one flag of a five-key object
        # and calling it the answer. Both are quoted from the same page,
        # re-read in full 2026-08-12.
        #
        # DEFAULT_EFFORT "none" MEANS OFF, whatever default_enabled says:
        # "If the value is 'none', treat it as 'reasoning off by default'
        # rather than pre-selecting disable when the user explicitly
        # turns reasoning on." openai/gpt-5.1 publishes default_enabled
        # true AND default_effort none, so the previous gate vouched for
        # it and a cap would have switched its reasoning on. That is the
        # exact failure the gate was built to prevent, surviving inside
        # the gate.
        #
        # A CAP WITHOUT BUDGET SUPPORT IS NOT A CEILING, IT IS AN EFFORT:
        # "For models that only support 'reasoning.effort' (see below),
        # the 'max_tokens' value will be used to determine the effort
        # level", and the table below it puts medium at "approximately
        # 50% of max_tokens", which is this constant's share. So on a
        # route without supports_max_tokens, sending half the budget does
        # not bound thinking at half; it ASKS FOR MEDIUM. On a model
        # whose default effort is minimal (approximately 10%) that is a
        # fivefold increase in thinking, bought unprompted. The Gemini
        # 3.1 and 3.5 Flash Lite variants publish exactly that shape:
        # default_enabled true, default_effort minimal, no
        # supports_max_tokens.
        #
        # So the reservation now rides only where it is a TRUE CEILING.
        # Anything else is a request to think differently, which is the
        # operator's call and not the bench's.
        #
        # AND THE MODEL MUST ADVERTISE THE PARAMETER, which is the third
        # condition and it is about strict mode rather than about money.
        # STRICT_PREFS sends require_parameters true, and under it a
        # provider that does not support a request parameter is not
        # allowed to ignore it, it is EXCLUDED. So an unprompted
        # reasoning field on a model that does not advertise reasoning
        # would either empty the provider pool outright or silently
        # narrow it, and this file already states at STRICT_PREFS why
        # narrowing it is the one thing that must not happen: it "changes
        # which providers are eligible, which silently changes what is
        # being measured".
        #
        # As the catalog stands the condition is redundant, because
        # every model that clears the other two also advertises the
        # parameter. It is here so that stays a fact about the code
        # rather than a fact about one morning's catalog: the flag that
        # says a model thinks and the list that says it accepts the
        # field are different publications and nothing keeps them in
        # step.
        #
        # THE PARTITION, measured against the live catalog on
        # 2026-08-12, 410 models:
        #
        #     8  the cap is a true ceiling, so it rides
        #   149  reasons, but no budget support: a cap would set effort
        #     0  reasons with budget support, parameter unadvertised
        #     2  default_effort none, so reasoning is OFF
        #    23  default_enabled false, so reasoning is OFF
        #    98  does not reason unprompted, or the catalog does not say
        #   130  no reasoning descriptor at all
        #
        # The previous gate vouched for 159 of these. This one vouches
        # for 8, and the 151 it withdraws are the finding rather than a
        # regression: on 149 of them the field was never a ceiling, and
        # on the other 2 it would have switched reasoning on.
        #
        # WHAT THAT COSTS, STATED PLAINLY. The incident's own model,
        # which publishes mandatory true and default_effort high, does
        # NOT publish supports_max_tokens, so it no longer receives a
        # reservation. The vendor's prose names its family as supporting
        # reasoning.max_tokens while the per-model flag does not, and
        # the two disagree; where a contract contradicts itself the
        # narrower reading is the safe one, because sending the field
        # anyway would ask a model whose operator chose nothing to think
        # at an effort nobody declared.
        #
        # That leaves the request-side arm of this fix covering 8 routes
        # and not the one that started it, which is worth saying out
        # loud rather than burying. The general answer was always the
        # instrumentation: the relabelled error and the card indicator
        # read the tokens that come BACK, so they work on all 410.
        #
        # Four flags collapse to one derived boolean because only one
        # question is being asked: may the bench send this model a
        # reasoning cap it did not ask for, and will that cap be a
        # ceiling when it arrives. Rule one's default for an unknown is
        # silence.
        reasoning = entry.get("reasoning")
        if isinstance(reasoning, dict):
            already_reasons = reasoning.get("mandatory") is True or (
                reasoning.get("default_enabled") is True
                # "none" is off, and it is checked here rather than
                # folded into default_enabled because the catalog really
                # does publish both together.
                and reasoning.get("default_effort") != "none"
            )
            advertised = "reasoning" in (model["supported_parameters"] or ())
            # THE CEILING CASE: supports_max_tokens makes the cap a cap.
            a_true_ceiling = (
                already_reasons
                and reasoning.get("supports_max_tokens") is True
                and advertised
            )
            # THE NO-HARM CASE, and it is reasoned from consequences
            # rather than from either side of a contract that
            # contradicts itself. See the block above EFFORT_SHARES for
            # the contradiction; this is what settles it without picking
            # a winner and without naming a vendor.
            #
            # Take a model that is MANDATORY and whose default_effort is
            # at or above the share this bench would ask for. Enumerate
            # every outcome the contract documents for a cap sent to it:
            #
            #   the route honours it as a budget      thinking is BOUNDED
            #   the route translates it to an effort  the translation
            #     lands at approximately the share sent, which is at or
            #     below this model's own default, so thinking is LOWERED
            #     or unchanged
            #   the provider ignores it entirely      NOTHING CHANGES
            #
            # None of the three raises thinking. The other harm,
            # switching thinking ON where it was off, cannot occur
            # either, and the reason is worth stating carefully because
            # a first version of this comment overstated it.
            #
            # It said mandatory means thinking is already on. Probe two
            # (tests/fixtures/probe_reasoning_cap_binding.json) measured
            # a mandatory model answering a trivial prompt with ZERO
            # reasoning tokens, twice, so mandatory describes what an
            # operator may not turn OFF rather than what the model will
            # do on any given request. Reasoning is route and prompt
            # dependent.
            #
            # The argument survives that correction intact, because a
            # cap cannot enable thinking that is already running and
            # cannot enable thinking that does not happen: the same
            # probe sent this exact cap to this exact model and the
            # reasoning count stayed at zero on both sides. What
            # mandatory rules out is the operator having chosen off,
            # which is the only way an unprompted cap could turn
            # something on. Both harms are therefore impossible, and
            # they are the only reasons the other conditions exist.
            #
            # The comparison is against REASONING_BUDGET_SHARE and not
            # against the word "medium", so it stays true if that
            # constant moves.
            declared_effort = reasoning.get("default_effort")
            # A non-str default_effort is absence, not a lookup key: the
            # catalog is somebody else's file and may publish anything.
            default_share = (
                EFFORT_SHARES.get(declared_effort)
                if isinstance(declared_effort, str)
                else None
            )
            cannot_raise_or_enable = (
                reasoning.get("mandatory") is True
                and default_share is not None
                and default_share >= REASONING_BUDGET_SHARE
                and advertised
            )
            model["may_send_reasoning_cap"] = a_true_ceiling or cannot_raise_or_enable
        # OpenRouter publishes a per-model completion cap under
        # top_provider where known. The budget clamp needs it: sending
        # a budget above the cap is a hard 400 from some providers.
        top = entry.get("top_provider")
        if isinstance(top, dict):
            cap = top.get("max_completion_tokens")
            # A non-bool int strictly above zero only. isinstance(True, int)
            # is true in Python, so a provider sending true would otherwise
            # become a cap of 1 that clamps every budget to a single token;
            # zero and negatives are not real caps either.
            if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
                model["max_completion_tokens"] = cap
        # Prices arrive as strings in USD per token. Malformed pricing
        # degrades this entry's price fields rather than dropping the
        # entry or the whole map. Non-finite and negative prices are
        # malformed too: a NaN price yields a NaN cost, and a NaN summed
        # into accumulated spend makes the ceiling comparison permanently
        # false, silently disabling it. Raising inside the try reuses the
        # single degrade path, matching as_metric's finiteness contract.
        try:
            prompt_price = float(entry["pricing"]["prompt"])
            completion_price = float(entry["pricing"]["completion"])
            if not (math.isfinite(prompt_price) and math.isfinite(completion_price)):
                raise ValueError("non-finite price")
            if prompt_price < 0 or completion_price < 0:
                raise ValueError("negative price")
            model["prompt_price"] = prompt_price
            model["completion_price"] = completion_price
            prices[entry["id"]] = {
                "prompt": prompt_price,
                "completion": completion_price,
            }
        except (KeyError, TypeError, ValueError):
            model["prompt_price"] = None
            model["completion_price"] = None
        models.append(model)
    return {"fetched": True, "models": models, "prices": prices, "digest": digest}


def _request_record(
    payload: dict[str, Any],
    record_prompt: str | None = None,
    controls: Mapping[str, Any] | None = None,
) -> str | None:
    """The payload as a stable JSON string, or None if it will not serialize.

    sort_keys so two identical requests produce identical records and a
    diff of two runs shows real differences rather than key order. Returns
    None rather than raising on an unserializable payload: this is a
    provenance nicety, and the never-raises contract outranks it.

    RULE TWO, RESOLVED FOR SIZE. record_prompt is the composed prompt
    with a digest reference standing where each attachment's text sat,
    and when it is given the recorded messages are rebuilt from it. The
    record is therefore NOT the wire bytes, deliberately, and this is the
    one place in the bench where those two differ.

    What makes that honest is that the wire bytes are RECONSTRUCTIBLE
    from what is stored: the record gives the exact structure and the
    digests, the attachments table gives each document's text by digest,
    and bench.extract.compose is the stated composition that puts them
    back together. Inlining the content instead would put a copy of every
    document into every result row, every export line and every history
    payload, which is the same document stored N times in the places
    least able to hold it.

    The messages are rebuilt through control_messages rather than patched
    in place, so the recorded shape comes from the same function that
    built the sent one and cannot drift from it.
    """
    if record_prompt is not None:
        payload = {**payload, "messages": control_messages(record_prompt, controls)}
    try:
        return json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _flatten_content(content: object) -> str | None:
    """Collapse a message content value to plain text or None.

    PER-FRAGMENT PRESENCE, NOT WHOLE-RESPONSE VISIBILITY, and the
    division of labour is easy to get wrong. This runs once per
    streaming delta, so a lone " " is a real fragment of a real
    sentence: returning None for it would drop that delta from both the
    accumulated text and the event yielded to the browser, and
    "Hello world" would arrive as "Helloworld". Whether a whole response
    is VISIBLE is a different question, asked at the two call sites that
    assemble one, with strip().

    Content-parts lists (multimodal providers) flatten to their text
    parts. Anything that is not a non-empty str after that returns None:
    response_text is str or None by contract, and a raw list would crash
    the sqlite bind in save_run and roll back the whole run.
    """
    if isinstance(content, list):
        # isinstance on the value, not .get with a default: a part with
        # an explicit null text ({"text": null}) returns None from .get
        # (the default only covers missing keys) and would crash the
        # join, escaping the never-raises contract.
        content = "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if isinstance(content, str) and content:
        return content
    return None


def _ingest_usage(result: dict[str, Any], usage: object) -> None:
    """Fold one usage object into a result, every field total.

    OpenRouter attaches usage to every response, in the final SSE chunk on
    a stream ("Full usage details are now always included automatically in
    every response", openrouter.ai/docs/cookbook/administration/
    usage-accounting, read 2026-07-25), so this is the one place the
    authoritative numbers enter. Always is the vendor's word, not an
    assumption this code makes: every field below still degrades when a
    provider disagrees. It is called with whatever the payload held:
    isinstance instead of a
    truthiness guard because a non-dict like "n/a" would pass truthiness
    and raise on .get, and neither client function may raise.

    Every field is assigned unconditionally, including to None, rather
    than only when present. A stream can carry more than one usage object,
    and a later one that omits a field must not leave the earlier value
    standing as if it had been reconfirmed. Last usage object wins,
    entirely.

    cost is the amount OpenRouter charged, and it goes through as_money
    rather than as_metric: a NaN or negative charge must degrade to None
    instead of reaching the spend accumulator, where NaN makes every
    ceiling comparison false and a negative subtracts from the total.

    cost_details.upstream_inference_cost IS persisted, into its own
    column, and it is a different number from cost. Pinned against
    https://openrouter.ai/docs/use-cases/usage-accounting, read
    2026-08-04: the usage object carries "cost" (in CREDITS) alongside
    "cost_details": {"upstream_inference_cost": N}, and that second
    figure "is only available for BYOK (Bring Your Own Key) requests.
    For all other requests it will be 0 or null."

    So on the ordinary credits path it is absent and the column is NULL,
    which is the honest record of "this run was not BYOK". Under BYOK the
    two diverge and only one of them is what the operator actually paid a
    provider. The ceiling still meters credits and only credits: it is a
    guard on the OpenRouter balance this process can spend, and a direct
    provider bill is not money OpenRouter can decline.
    """
    if not isinstance(usage, dict):
        return
    # THE UNIT EVERY STORED COUNT IS IN, entering here and nowhere else.
    # OpenRouter's usage-accounting page, read 2026-08-12: "Prompt and
    # completion token counts using the model's native tokenizer". So the
    # four counts below are NATIVE, all four of them, from one object, and
    # the bench has exactly one unit for counts because it has exactly one
    # source for them.
    #
    # The normalized figures (tokens_prompt, tokens_completion) exist only
    # on the generation endpoint, which is a post-hoc pass most rows never
    # get, so there is no in-band normalized count to mix these with even
    # if something wanted to.
    #
    # What is NOT in this unit and must never be compared with it:
    # max_tokens, which is a ceiling the bench sent rather than anything a
    # tokenizer counted. Every surface that shows the two together labels
    # the cap as a cap; see the budget note in static/render.js for what
    # happened when one did not.
    result["prompt_tokens"] = as_token_count(usage.get("prompt_tokens"))
    result["completion_tokens"] = as_token_count(usage.get("completion_tokens"))
    result["billed_cost_usd"] = as_money(usage.get("cost"))
    result["upstream_inference_cost_usd"] = _as_upstream_cost(usage.get("cost_details"))
    # Whether OpenRouter billed this against your own provider key. Its
    # own field, because the upstream figure above cannot carry the
    # meaning: a live probe found is_byok false beside a nonzero upstream
    # cost, so the presence of that number says nothing about BYOK.
    #
    # isinstance rather than truthiness, and no coercion: a provider that
    # omits the flag leaves None, which is "not reported" and is not the
    # same answer as false. Anything that is not a bool is treated as
    # absent rather than guessed at.
    is_byok = usage.get("is_byok")
    result["is_byok"] = is_byok if isinstance(is_byok, bool) else None
    # Reasoning tokens are the reason the two-tier budget exists: they are
    # billed as completion tokens and consume max_tokens, but never appear
    # in the visible answer. Cached prompt tokens are the other direction,
    # charged at a discount, and both live one level down in the details
    # objects.
    completion_details = usage.get("completion_tokens_details")
    result["reasoning_tokens"] = (
        as_token_count(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else None
    )
    prompt_details = usage.get("prompt_tokens_details")
    result["cached_tokens"] = (
        as_token_count(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, dict)
        else None
    )


# The share of a completion that reasoning must reach before "the
# thinking took the answer's place" is a description rather than a guess.
#
# NOT EQUALITY, even though the incident's own numbers were exactly equal,
# tokens_completion 21350 against native_tokens_reasoning 21350. A
# provider that counts a dropped partial token, or a trailing whitespace
# delta that _flatten_content discards, into completion_tokens puts the
# ratio a hair under one; a test that fired only on exact equality would
# fall back to the old useless wording over a rounding difference. Nine
# tenths sits far above what an ordinary short response reaches and far
# below anything a real exhaustion misses.
REASONING_SHARE_EXHAUSTED = 0.9

# OpenRouter's NORMALIZED word for a completion the cap cut off. Reading
# it couples this to no vendor: run_model's own comment on the field says
# OpenRouter "maps each provider's vocabulary onto a common set", and the
# provider's own word arrives separately as native_finish_reason.
#
# It is the only field that answers "did the budget actually run out",
# and the label needs that answer; see empty_response_error.
FINISH_TRUNCATED = "length"


def reasoning_ate_the_output(
    completion_tokens: int | None, reasoning_tokens: int | None
) -> bool:
    """Whether a completion that produced no visible answer went to
    thinking.

    NUMBERS ONLY. No model name, no finish_reason, no request field: this
    reads what came BACK. That is deliberate and it is what makes the
    same judgement possible on a route where R1's reservation could not
    act, because a provider that ignores an unknown parameter still
    reports its token counts honestly. The frontend's indicator applies
    the identical rule to the identical two numbers for the same reason.

    Both counts degrade to None when a provider omits usage, and 0 is a
    real answer meaning "nothing was spent thinking"; either way there is
    no evidence of exhaustion here, so the falsy guard covers both.
    """
    if not reasoning_tokens or not completion_tokens:
        return False
    return reasoning_tokens >= completion_tokens * REASONING_SHARE_EXHAUSTED


def empty_response_error(
    finish_reason: str | None,
    completion_tokens: int | None,
    reasoning_tokens: int | None,
) -> str:
    """The error for a 200 that carried no visible text.

    ONE FUNCTION FOR TWO SITES, the batch path and the stream's done(),
    which previously carried the same sentence written out twice. A label
    whose whole purpose is honesty must not be able to drift between the
    endpoint a script uses and the endpoint the browser uses.

    WHY THE OLD WORDING WAS A DEFECT AND NOT MERELY TERSE. "empty
    response (finish_reason: length)" is true and tells the reader
    nothing they can act on: it names the symptom, omits where the money
    went, and reads like a provider glitch. The card it appeared on had
    been billed between $3.38 and $3.50 for 21350 tokens of thinking. A
    person reading it retried, at cost, because nothing in it suggested
    the retry would do the same thing.

    KEYED ON SHAPE, NOT ON A NAME, so it fires for any model that thinks
    its budget away. The reasoning count goes in the text because it is
    the evidence for the claim; a reader who doubts the label can check
    it against the card's own metrics.

    THREE OUTCOMES, NOT TWO, AND THE THIRD IS A CORRECTION. An
    adversarial review found that the shape test is close to a TAUTOLOGY
    at this call site. This function is only reached when
    _flatten_content came back falsy, so completion minus reasoning is
    the count of completion tokens that were neither thinking nor
    visible text, which is around zero for any reasoning model that
    produced nothing. The ratio is therefore at or near 1.0 whatever the
    REASON for the emptiness: a refusal, a content filter, a provider
    abort and a real truncation all land in the same bucket.

    So the shape alone cannot carry the word "exhausted", which claims
    the budget ran out. Measured before the fix, a content_filter
    refusal that spent 37 of 16384 tokens, 0.2% of its budget, was told
    its completion budget had been exhausted, and static/stream.js then
    appended "try extended budget" because it keys its remedy on the
    same predicate. The branch was recommending a four-times-larger
    retry for a failure no budget can fix, which is a sharper version of
    the very thing the old wording was replaced for.

    finish_reason is what discriminates, and reading it costs nothing
    this fix cares about: it is OpenRouter's normalized value, so it
    names no vendor, and it is a field that came BACK, so it is as
    available on an ignored-parameter route as the counts are.

    The old wording is kept for a genuinely empty response with no
    reasoning behind it, which is a third failure again and was always
    described correctly by the sentence already there.
    """
    if reasoning_ate_the_output(completion_tokens, reasoning_tokens):
        if finish_reason == FINISH_TRUNCATED:
            return (
                "no visible answer: completion budget exhausted during "
                f"reasoning ({reasoning_tokens} reasoning tokens)"
            )
        # The accounting fact without the causal claim. Everything the
        # reader can act on survives: where the money went, how much of
        # it, and why generation stopped. Only the assertion the numbers
        # do not support is gone.
        #
        # "nearly all of" rather than "the whole", because the predicate
        # fires at nine tenths and the sentence was claiming ten. On a
        # response of 338 completion tokens with 312 reasoning, which is
        # a real measured shape in this repository, 26 tokens did reach
        # the answer and the old wording denied it. A sentence written
        # to stop a label overstating its evidence must not overstate
        # its own.
        return (
            "no visible answer: nearly all of the completion went to "
            "reasoning "
            f"({reasoning_tokens} reasoning tokens, finish_reason: "
            f"{finish_reason or 'unknown'})"
        )
    return f"empty response (finish_reason: {finish_reason or 'unknown'})"


# OpenRouter returns the generation id in this response header on every
# endpoint, available the moment the response starts and therefore before
# any chunk has been read. Pinned against the vendor's own words rather
# than memory, like DATA_POLICY_PREFS above: "fetch the generation record
# via GET /api/v1/generation using the X-Generation-Id response header"
# (openrouter.ai/docs/guides/features/router-metadata, read 2026-07-25).
#
# That timing is the point: a run the user stops
# mid-stream is exactly the run whose billing is unknowable locally
# (cancellation only stops billing on providers that support it), so it is
# the run that most needs an id to reconcile against later, and reading the
# id out of a chunk would lose it for precisely those runs.
GENERATION_ID_HEADER = "X-Generation-Id"


def _settle_generation_id(
    result: dict[str, Any],
    value: object,
    holder: dict[str, Any] | None = None,
) -> None:
    """Record the generation id the first time one is seen.

    First writer wins, so the header settles the field and a chunk id only
    fills in when no header arrived. A later value that disagrees is logged
    rather than raised or applied: the never-raises contract holds, and
    overwriting would mean the persisted id depends on which source spoke
    last. Junk and empty strings are absence, per _as_label.

    holder is dependency-injected state, not a protocol change. The
    streaming endpoint owns a plain dict and passes it in so its disconnect
    path can read the id without waiting for a done event that a stopped
    run never produces. Only the id goes in, and only by assignment, so
    reading it back during generator unwinding cannot await or raise.
    """
    gen_id = _as_label(value)
    if gen_id is None:
        return
    settled = result["generation_id"]
    if settled is None:
        result["generation_id"] = gen_id
        if holder is not None:
            holder["generation_id"] = gen_id
    elif settled != gen_id:
        logger.warning(
            "generation id disagreement: header/first chunk said %s, later "
            "source said %s; keeping the first",
            settled,
            gen_id,
        )


def _as_upstream_cost(source: Any) -> str | None:
    """The BYOK provider charge, kept verbatim as text, or None.

    TEXT rather than a float because nothing in this bench computes with
    it: the ceiling meters credits, the report's totals are over credits,
    and this number exists to be READ by an operator reconciling against
    a provider invoice. Storing the received value as written keeps it
    exactly comparable to that invoice; converting it to a float and back
    would introduce a rounding this bench has no reason to perform.

    A ZERO IS NOW STORED AS A ZERO, which reverses an earlier decision
    and the reversal is the point. This used to degrade 0 to None on the
    documented grounds that "For all other requests it will be 0 or
    null", so a stored 0 could not be confused with a BYOK run that
    genuinely cost nothing. A live probe on 2026-08-12 broke that
    reasoning from both ends: a non-BYOK run returned a NONZERO upstream
    figure equal to the credit charge, so a populated cell never meant
    BYOK in the first place, and is_byok now has a column of its own to
    say what this value was never able to say.

    With the discriminator moved out, suppressing an observed money
    figure would only make the database disagree with the wire. NULL here
    means the wire sent nothing usable, and nothing else.

    Not-a-number, infinity and negatives still degrade, because those are
    not money figures at all; that is the same rule as_money applies to
    every monetary field in this file and it is about validity rather
    than about meaning.
    """
    if not isinstance(source, dict):
        return None
    value = source.get("upstream_inference_cost")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return repr(value)


def _as_label(value: object) -> str | None:
    """as_text with the empty string treated as absent.

    A provider name or a finish reason of "" is not a value the frontend
    can render as a caption; it would paint an empty slot that looks like
    a layout bug. Everything else is as_text's contract.
    """
    text = as_text(value)
    return text if text else None


async def run_model(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    max_tokens: int = BUDGET_STANDARD,
    provider_prefs: dict[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
    record_prompt: str | None = None,
    may_send_reasoning_cap: bool = False,
) -> dict[str, Any]:
    """Send one chat completion to OpenRouter and return a flat result dict.

    Never raises. A comparison run fans out to several models and one
    failure must not sink the others, so every error path collapses into
    the error field of an otherwise well-formed result.

    The result echoes max_tokens because a truncated answer at 16k and
    one at 65k are different experiments; persistence must record which
    budget was actually sent, clamping included.
    """
    result: dict[str, Any] = {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
        "max_tokens": max_tokens,
        "generation_id": None,
        "finish_reason": None,
        "request_json": None,
        "billed_cost_usd": None,
        "upstream_inference_cost_usd": None,
        "is_byok": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "provider": None,
        "native_finish_reason": None,
    }
    payload = {
        "model": model,
        "messages": control_messages(prompt, controls),
        "max_tokens": max_tokens,
        # Injected rather than read from module state, so the boot-scoped
        # data policy reaches the wire without this function knowing where
        # policy lives. The default keeps a direct caller on today's
        # behavior.
        "provider": PROVIDER_PREFS if provider_prefs is None else provider_prefs,
        # THE VISIBLE-OUTPUT RESERVATION, on every request that does not
        # already carry a declared effort. See reasoning_reservation and
        # REASONING_BUDGET_SHARE: this is the one field the bench sends
        # UNPROMPTED, a deliberate and documented exception to "never
        # send a default", taken because the alternative is a comparison
        # that bills in full and shows nothing.
        #
        # BEFORE control_payload rather than after, so an operator's
        # declared effort would win a collision by overwriting this. It
        # cannot collide in practice, because the reservation stands
        # down when an effort is set, but the ordering means the rule
        # holds even if that guard were ever wrong.
        **reasoning_reservation(max_tokens, controls, may_send_reasoning_cap),
        # Last, and only the controls that were set. The merge is safe
        # because control_payload emits from a fixed key list that shares
        # nothing with the keys above; a test asserts that, so a future
        # control cannot quietly overwrite the model or the budget.
        **control_payload(controls),
    }
    # The reproducibility record: what was actually sent, not what was
    # intended. Serialized here, beside the dict that goes on the wire, so
    # the two cannot drift. Authorization lives in the client's headers and
    # never enters the payload, so nothing secret is recorded. Echoed even
    # on failure paths, because knowing what a failed request asked for is
    # exactly when this matters. The controls ride along for free, which is
    # what lets a run prove it asked for a temperature a provider ignored.
    result["request_json"] = _request_record(payload, record_prompt, controls)

    start = time.perf_counter()
    try:
        response = await client.post(
            OPENROUTER_URL, json=payload, timeout=COMPLETION_TIMEOUT
        )
    # ConnectTimeout before the generic catch: it fires after
    # CONNECT_TIMEOUT_S, and the generic message would claim a wait
    # eighteen times longer than what actually happened.
    except httpx.ConnectTimeout:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"could not connect within {CONNECT_TIMEOUT_S:.0f}s"
        return result
    except httpx.TimeoutException:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"no response within {COMPLETION_READ_TIMEOUT_S:.0f}s"
        return result
    except httpx.HTTPError as exc:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["error"] = f"request failed: {type(exc).__name__}"
        return result
    # Latency covers the HTTP round trip only. JSON parsing happens below,
    # outside the clock, so big responses do not inflate the measurement.
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)

    # Before the status check and before the body is parsed: the header is
    # available at response start, and an id captured here survives a
    # response whose body turns out to be unparseable, which is one of the
    # cases worth reconciling later.
    _settle_generation_id(result, response.headers.get(GENERATION_ID_HEADER))

    if response.status_code != 200:
        result["error"] = f"HTTP {response.status_code} from OpenRouter"
        return result

    try:
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, LookupError, TypeError):
        result["error"] = "malformed response from OpenRouter"
        return result

    # Provenance, both best-effort. The generation id keys OpenRouter's
    # generation API, which records the actual provider, quantization
    # and authoritative cost, so persisting it makes this run auditable
    # after the fact. The body id is the fallback here, not the primary:
    # the header above already settled the field on any response that
    # carried one. The finish_reason says why output ended; until now
    # it survived only inside synthesized error strings, and budget
    # analysis needs it on successful runs too. It is OpenRouter's
    # NORMALIZED value, not the provider's own word: OpenRouter maps each
    # provider's vocabulary onto a common set and exposes the raw one
    # separately as native_finish_reason, which is captured below.
    _settle_generation_id(result, data.get("id"))
    reason = choice.get("finish_reason")
    if isinstance(reason, str) and reason:
        result["finish_reason"] = reason
    # The provider that actually served this request, and its own word for
    # why generation ended. Under throughput routing the provider is
    # chosen per request, so it is the largest confound in any comparison
    # this bench draws; recording it per result is what makes the confound
    # visible instead of assumed away. Both degrade to None when the
    # payload omits them, which is the common case for native_finish_reason
    # (OpenRouter only sends it when the provider's value differs from its
    # own normalized one).
    result["provider"] = _as_label(data.get("provider"))
    result["native_finish_reason"] = _as_label(choice.get("native_finish_reason"))

    # A missing or oddly shaped usage object leaves the counts and the
    # billed cost None rather than guessed. The guard that makes that safe
    # (isinstance, not truthiness, so a value like "n/a" cannot raise on
    # .get) lives in _ingest_usage now, which is where its comment lives
    # too.
    #
    # BEFORE the empty-text branch below, which is a move rather than an
    # accident of layout: that branch now reads the token counts this
    # writes, and with the old ordering it read two Nones and could never
    # produce the exhaustion label. The streaming path never had the
    # problem, because usage is ingested per chunk and done() runs last.
    _ingest_usage(result, data.get("usage"))

    text = _flatten_content(content)
    # STRIP AS A PREDICATE, NEVER AS A TRANSFORM. Whitespace is not a
    # visible answer: a response of spaces and newlines used to be
    # truthy here, so response_text was set, the empty-response guard
    # below never ran, and a fully billed reasoning burn rendered as a
    # blank card with no error and therefore no remedy.
    #
    # What is STORED is still the text exactly as it arrived. Only the
    # question "did this produce anything a reader can see" is asked
    # with strip(), because trimming the stored value would edit a
    # model's output on its way into the record.
    if text is not None and text.strip():
        result["response_text"] = text
    else:
        # Some providers return 200 with null content on refusals, and a
        # reasoning model can spend a whole budget without ever producing
        # visible text. Both must surface as an error so every result
        # carries either text or an error, a contract the frontend relies
        # on to pick a render state.
        #
        # empty_response_error tells the three cases apart, and it needs
        # finish_reason to do it: at this call site there is no visible
        # text by construction, so the token shape alone cannot separate
        # a refusal that thought a little from a budget that ran out.
        # Non-str oddities land here too rather than leaking into
        # response_text.
        result["error"] = empty_response_error(
            result["finish_reason"],
            result["completion_tokens"],
            result["reasoning_tokens"],
        )

    return result


async def stream_model(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    max_tokens: int = BUDGET_STANDARD,
    holder: dict[str, Any] | None = None,
    provider_prefs: dict[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
    record_prompt: str | None = None,
    may_send_reasoning_cap: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one chat completion, yielding delta and done event dicts.

    holder, when given, receives facts about the exchange as soon as each
    becomes known: the request record before the request is sent, then the
    generation id and the serving provider. A consumer that never reaches
    the done event (a client disconnect cancels this generator at a yield)
    has no other way to learn any of them, and a run cut short is exactly
    the run whose billing has to be reconciled later, on a row that should
    still be able to say what it asked for. Writes only, by plain
    assignment, so a reader during generator unwinding cannot await or
    raise. See _settle_generation_id.

    Yields {"type": "delta", "text": chunk} per content delta, then
    exactly one {"type": "done", "result": {...}}. Like run_model, this
    never raises: every failure path lands in the done result's error
    field, alongside whatever text accumulated before the failure. Text
    is accumulated here so the done event is always self-sufficient for
    persistence, even when the consumer dropped deltas.

    The read timeout is a gap between chunks, not total stream
    duration: a slow model legitimately streams for minutes, and a
    reasoning model can think silently for over a minute before its
    first visible delta, but a silence past STREAM_READ_TIMEOUT_S
    still means something is wrong.
    """
    result: dict[str, Any] = {
        "model": model,
        "response_text": None,
        "latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
        "ttft_ms": None,
        "max_tokens": max_tokens,
        "generation_id": None,
        "finish_reason": None,
        "request_json": None,
        "billed_cost_usd": None,
        "upstream_inference_cost_usd": None,
        "is_byok": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "provider": None,
        "native_finish_reason": None,
    }
    payload = {
        "model": model,
        "messages": control_messages(prompt, controls),
        "max_tokens": max_tokens,
        # See run_model: injected so the boot-scoped data policy reaches
        # the wire without this function knowing where policy lives.
        "provider": PROVIDER_PREFS if provider_prefs is None else provider_prefs,
        "stream": True,
        # A deprecated no-op, kept only because removing it would change
        # the recorded request_json of every streamed run for no gain.
        # OpenRouter's usage-accounting page states that this flag and
        # usage.include "are deprecated and have no effect. Full usage
        # details are now always included automatically in every response"
        # (read 2026-07-25). The usage block arrives because the platform
        # sends it, not because this asks.
        "stream_options": {"include_usage": True},
        # See run_model: the reservation rides here too, and for the same
        # reason. The streaming path is the one the browser uses, so a
        # reservation that covered only the batch endpoint would cover
        # almost nothing a person actually runs.
        **reasoning_reservation(max_tokens, controls, may_send_reasoning_cap),
        # See run_model: last, and only what was set, so a blank controls
        # set leaves this payload byte for byte what it was before.
        **control_payload(controls),
    }
    # See run_model: the exact payload, auth excluded, recorded beside the
    # dict that goes on the wire. Mirrored into the holder in the same
    # breath, because this is knowable before the request is even sent and
    # a run cut short mid-stream is exactly the row that should still be
    # able to say what it asked for.
    result["request_json"] = _request_record(payload, record_prompt, controls)
    if holder is not None:
        holder["request_json"] = result["request_json"]
    text_parts: list[str] = []
    saw_done = False
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    def done(error: str | None) -> dict[str, Any]:
        result["latency_ms"] = elapsed_ms()
        # The same predicate as run_model, on the whole response rather
        # than on any one delta. joined is kept verbatim; only the
        # decision uses strip(). See _flatten_content for why the strip
        # cannot live there.
        joined = "".join(text_parts)
        if joined.strip():
            result["response_text"] = joined
        result["error"] = error
        # Same guard as run_model, and the same function, so the two
        # endpoints cannot describe one failure two ways. Usage is
        # ingested per chunk above, so the counts this reads are the
        # final ones by the time done() runs.
        if result["response_text"] is None and error is None:
            result["error"] = empty_response_error(
                result["finish_reason"],
                result["completion_tokens"],
                result["reasoning_tokens"],
            )
        return {"type": "done", "result": result}

    try:
        async with client.stream(
            "POST", OPENROUTER_URL, json=payload, timeout=STREAM_TIMEOUT
        ) as response:
            # Before the status check and before the first chunk is read.
            # This is the whole reason the header is preferred: a run the
            # user stops seconds later never reaches a chunk that carries
            # an id, and it is the run that most needs one.
            _settle_generation_id(
                result, response.headers.get(GENERATION_ID_HEADER), holder
            )
            if response.status_code != 200:
                yield done(f"HTTP {response.status_code} from OpenRouter")
                return
            async for line in response.aiter_lines():
                # SSE comments (OpenRouter keep-alives) and blank lines
                # carry nothing.
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise ValueError("chunk is not an object")
                except ValueError:
                    yield done("malformed stream from OpenRouter")
                    return

                # The chunk id is the fallback for a response that carried
                # no header. Every chunk repeats the same id, so the first
                # one that carries it settles the field, and a disagreement
                # with the header is logged rather than applied.
                _settle_generation_id(result, chunk.get("id"), holder)

                # The provider name rides every chunk, not just the last;
                # the first one that carries it settles the field, like the
                # generation id above, and reaches the holder for the same
                # reason: an aborted row that knows which host served it is
                # a row whose routing confound is still visible.
                if result["provider"] is None:
                    result["provider"] = _as_label(chunk.get("provider"))
                    if holder is not None and result["provider"] is not None:
                        holder["provider"] = result["provider"]

                # Usage arrives on the final chunk (stream_options above
                # asks for it) and carries the billed cost with it.
                _ingest_usage(result, chunk.get("usage"))

                # OpenRouter reports mid-stream failures as an in-band
                # error object on the 200 stream. Without this check the
                # frame has no choices, falls into the guard below, and
                # the real upstream reason is silently lost while the
                # stream ends looking like a clean success.
                err = chunk.get("error")
                if isinstance(err, dict):
                    detail = err.get("message") or err.get("code") or "unknown"
                    yield done(f"upstream error: {detail}")
                    return

                try:
                    choice = chunk["choices"][0]
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str) and reason:
                        # The provider sends its verdict on the closing
                        # chunk; the final one seen is the one recorded.
                        result["finish_reason"] = reason
                    native = _as_label(choice.get("native_finish_reason"))
                    if native is not None:
                        # Guarded like finish_reason above: OpenRouter sends
                        # the provider's own word only on the chunk that ends
                        # generation, so a later chunk carrying a choice
                        # without it must not erase what was already seen.
                        result["native_finish_reason"] = native
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                except (LookupError, TypeError, AttributeError):
                    # Usage-only or oddly shaped chunks contribute no text.
                    continue

                text = _flatten_content(content)
                if text:
                    if result["ttft_ms"] is None:
                        result["ttft_ms"] = elapsed_ms()
                    text_parts.append(text)
                    yield {"type": "delta", "text": text}
    # Same split as run_model: a connect timeout is not a stall, and
    # "no data for 300s" would misreport a 10s handshake failure.
    except httpx.ConnectTimeout:
        yield done(f"could not connect within {CONNECT_TIMEOUT_S:.0f}s")
        return
    except httpx.TimeoutException:
        yield done(f"stream stalled: no data for {STREAM_READ_TIMEOUT_S:.0f}s")
        return
    except httpx.HTTPError as exc:
        yield done(f"request failed: {type(exc).__name__}")
        return

    # A clean iterator exhaustion without the [DONE] marker is not a
    # success: a load balancer idle-closing the connection or a provider
    # crashing with a clean close both land here, and done(None) would show
    # and persist a truncated answer as complete, corrupting the
    # comparison. But a stream that delivered a finish_reason and then
    # closed without [DONE] is semantically complete (the provider stated
    # why it stopped), so flagging that would be a false alarm. done()
    # already carries the partial text accumulated so far.
    if not saw_done and result["finish_reason"] is None:
        yield done("stream ended before completion: no [DONE] and no finish reason")
        return
    yield done(None)


# The generation endpoint is an audit surface, not a costing path: it is
# called after the fact by the reconcile script, never during a run.
# Fifteen seconds is generous for one metadata lookup and short enough
# that a hung endpoint does not stall a long backfill.
GENERATION_TIMEOUT_S = 15.0


async def fetch_generation(
    client: httpx.AsyncClient, generation_id: str
) -> dict[str, Any]:
    """One generation's after-the-fact record, or an error field saying why.

    Never raises, like the two client functions: this runs in a loop over
    many rows, and one bad id must not end the pass. Every field goes
    through the same total field-type functions, with the money rule on the
    charge, so a poisoned value from the audit path cannot do what it could
    not do from the streaming path.

    Field names are the ones OpenRouter's OpenAPI description of
    GET /generation gives (read 2026-07-25 from
    https://openrouter.ai/docs/api/api-reference/generations/get-request-%26-usage-metadata-for-a-generation):
    the charge is total_cost, the host is provider_name, and the token
    counts prefixed native_ are the model's own tokenizer's, which is what
    the in-band usage object reports too, so the two are comparable.

    quantization is read but is NOT in that description's response schema
    today. It is fetched anyway rather than omitted, because the field is
    what the reconcile pass exists to recover and reading a key that may
    appear costs nothing; until it does appear, this degrades to None like
    any absent field. Do not read a None here as "the provider served
    unquantized weights".
    """
    record: dict[str, Any] = {
        "generation_id": generation_id,
        "billed_cost_usd": None,
        "upstream_inference_cost_usd": None,
        "is_byok": None,
        "provider": None,
        "quantization": None,
        "native_finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "error": None,
    }
    try:
        response = await client.get(
            GENERATION_URL,
            params={"id": generation_id},
            timeout=GENERATION_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        record["error"] = f"request failed: {type(exc).__name__}"
        return record
    if response.status_code != 200:
        # 404 is ordinary here: OpenRouter expires generation records, and
        # a run old enough to have aged out is not a failure of this pass.
        record["error"] = f"HTTP {response.status_code} from OpenRouter"
        return record
    try:
        data = response.json()["data"]
        if not isinstance(data, dict):
            raise ValueError("data is not an object")
    except (ValueError, LookupError, TypeError):
        record["error"] = "malformed response from OpenRouter"
        return record

    record["billed_cost_usd"] = as_money(data.get("total_cost"))
    # The generation endpoint names it at the TOP level rather than under
    # cost_details, which the streaming usage object does not. Pinned
    # against the Get a Generation reference, read 2026-08-04, whose
    # example response carries "total_cost": 0.0015 beside
    # "upstream_inference_cost": 0.0012 and "is_byok": false.
    record["upstream_inference_cost_usd"] = _as_upstream_cost(data)
    # THE DISCRIMINATOR, at the TOP level of data here rather than
    # nested under cost_details as the streaming usage object puts it.
    # The comment above already quoted the pinned example as carrying
    # "is_byok": false beside the figure, and then did not read it, which
    # is the same fetched-and-dropped shape RECONCILABLE_COLUMNS calls
    # out one step later.
    record["is_byok"] = as_flag(data.get("is_byok"))
    record["provider"] = _as_label(data.get("provider_name"))
    record["quantization"] = _as_label(data.get("quantization"))
    record["native_finish_reason"] = _as_label(data.get("native_finish_reason"))
    # THE NATIVE FIELDS ON PURPOSE, and parsed but NOT PERSISTED, which
    # is a deliberate pair of choices rather than an oversight in either
    # direction.
    #
    # Native, because store's counts are native (see _ingest_usage) and
    # the endpoint offers both: "native_tokens_completion - Native
    # completion tokens as reported by provider" beside a plain
    # "tokens_completion - Number of tokens in the completion". Reading
    # the normalized pair here would put a second unit into a record whose
    # whole job is to be comparable with what is already stored.
    #
    # Not persisted, because RECONCILABLE_COLUMNS in bench/store.py does
    # not list them, so a reconcile pass reads these and drops them.
    # Adding columns for them would put two counts per result in the
    # database and reintroduce, permanently, exactly the ambiguity this
    # workstream exists to remove.
    #
    # They are kept anyway, and the reason has been corrected. This
    # comment used to call the record "the ONE PLACE the two units can
    # be compared", which is false: only the native pair is read here,
    # so there is no second unit in this dict to compare anything with.
    # What the record actually offers is two SOURCES of the same unit,
    # the counts OpenRouter reported in-band on the response and the
    # counts its generation endpoint reports afterwards. Those can
    # disagree, and a reader debugging a suspicious row can print this
    # record and see whether they do. That is a smaller claim than the
    # one it replaces and it is the true one.
    #
    # Pinned against the Get a Generation reference, read 2026-08-12.
    record["prompt_tokens"] = as_token_count(data.get("native_tokens_prompt"))
    record["completion_tokens"] = as_token_count(data.get("native_tokens_completion"))
    record["reasoning_tokens"] = as_token_count(data.get("native_tokens_reasoning"))
    record["cached_tokens"] = as_token_count(data.get("native_tokens_cached"))
    return record


# ---- Phase I: rubric judging.

# The judge's completion budget. Deliberately small and deliberately its
# own constant rather than the experiment's tier: a verdict is a number
# and a sentence, and a judge inheriting an extended-budget experiment
# would buy headroom no rubric needs, once per scored trial, at the
# judge's own price. Large enough that a reasoning model's preamble plus
# the verdict fits; nowhere near large enough to pay for an essay.
JUDGE_MAX_TOKENS = 512

# THE JUDGE SENDS NO REASONING RESERVATION, and the arithmetic is why.
#
# Pinned against
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
# re-read 2026-08-12. Two sentences settle it:
#
#   "When using the reasoning.max_tokens parameter, that value is used
#   directly with a minimum of 1024 tokens."
#
#   "Important: max_tokens must be strictly higher than the reasoning
#   budget to ensure there are tokens available for the final response
#   after thinking."
#
# Put them together against this budget. A judge asking for half of 512
# sends a reasoning budget of 256, which the minimum raises to 1024, and
# 1024 is not strictly lower than an outer max_tokens of 512. The
# request is unsatisfiable by the contract's own arithmetic, and it was
# being sent to every Anthropic judge that reasons by default. An
# earlier version of this file shipped it, and the test that was
# supposed to guard it mocked a 200 and read only the outgoing JSON, so
# a request no provider could honour looked correct forever.
#
# RAISING THE BUDGET IS NOT THE FIX. To satisfy both rules the judge
# would need an outer budget above 1024 with the reservation below it,
# so at least 2048 to keep a half share, which is four times what a
# verdict costs today, once per scored trial, at the judge's price. The
# bench would be buying reasoning headroom for a task whose entire
# output is a number and a sentence.
#
# AND A JUDGE THAT EXHAUSTS IS ALREADY HONEST. When a judge reasons past
# its cap the trial is recorded as UNSCORED rather than failed, which is
# the true statement: nobody graded it. That is a different situation
# from a comparison card that billed for thinking and showed nothing,
# because no claim about a model's answer depends on it. The failure
# mode the reservation exists to prevent does not apply here.
JUDGE_SENDS_NO_RESERVATION = True

JUDGE_TIMEOUT_S = 60.0

# The instruction that constrains the verdict's shape. Prompt-level
# rather than response_format, and the choice is worth stating.
# OpenRouter does document response_format
# ({"type": "json_object"} at https://openrouter.ai/docs/api_reference/parameters,
# read 2026-08-01), but it guarantees valid JSON, not the right keys, so
# the defensive parse below is required either way. What sending it would
# add is a narrowing of which providers can serve the judge, since
# structured output is a per-model capability. Paying provider
# eligibility for a partial guarantee that removes no code is a bad
# trade, so the shape is asked for in words and verified on arrival.
#
# Revisit this if the scoring-failure rate in real reports shows
# unparseable verdicts are material in practice. The trade above assumes
# they are rare; if a report shows judges losing a noticeable share of
# their verdicts to formatting, the narrowed provider pool becomes the
# cheaper cost and this should send response_format. The number to look
# at is the count of judge score rows with a null score and a parse
# error in detail, against the total judged.
JUDGE_SYSTEM = (
    "You are grading one response against a rubric. Reply with JSON only, "
    'no prose and no code fence, exactly: {"score": <number between 0 and '
    '1>, "reason": "<one short sentence>"}. Use the rubric to choose the '
    "score. If the response is empty or does not address the task at all, "
    "score 0."
)


def judge_messages(
    rubric: str, reference: str | None, response_text: str
) -> list[dict[str, str]]:
    """The judge payload's messages, carrying no identity of any kind.

    Blind by construction rather than by discipline. This function is not
    given the model that produced the response, so no future edit inside
    it can leak one: the identity is not in scope. That matters because a
    judge told which model it is grading has a documented tendency to
    prefer some of them, and a blindness enforced by "remember not to
    include it" is a blindness one careless edit away from gone.

    The reference, when the task has one, is given as the expected answer
    rather than as another candidate. A judge shown two answers without
    being told which is which is doing pairwise comparison, which is a
    different method with its own position-bias machinery and is
    deliberately out of scope for this phase.
    """
    parts = [f"Rubric:\n{rubric}"]
    if reference is not None:
        parts.append(f"Expected answer:\n{reference}")
    parts.append(f"Response to grade:\n{response_text}")
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_verdict(text: str | None) -> dict[str, Any]:
    """A judge's reply turned into a score, or an honest scoring failure.

    Never guesses. An unparseable verdict, a score that is not a number,
    or a number outside [0, 1] all become a recorded scoring failure with
    the reason attached, because a guessed number here would enter an
    average and change a conclusion while looking exactly like a real
    measurement.

    A fenced or preamble-wrapped object is recovered by taking the
    outermost braces, which is not guessing: it is the same object, and
    models wrap JSON in fences often enough that refusing would discard
    correct verdicts over formatting. Anything past that is refused.
    """
    raw = as_text(text)
    if raw is None or not raw.strip():
        return {"error": "judge returned no text"}
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return {"error": "judge reply contained no JSON object"}
    try:
        data = json.loads(raw[start : end + 1])
    except ValueError as exc:
        return {"error": f"judge reply was not valid JSON: {exc}"}
    if not isinstance(data, dict):
        return {"error": "judge reply was not a JSON object"}
    # as_metric, not float(): a bool is not a score, a string is not a
    # score, and a NaN must not become one. Same total-field rule every
    # other upstream number goes through.
    score = as_metric(data.get("score"))
    if score is None:
        return {"error": f"judge score was not a number: {data.get('score')!r}"}
    if not 0.0 <= score <= 1.0:
        return {"error": f"judge score {score} is outside [0, 1]"}
    return {"score": score, "reason": as_text(data.get("reason"))}


async def judge_response(
    client: httpx.AsyncClient,
    judge_model: str,
    rubric: str,
    reference: str | None,
    response_text: str | None,
    provider_prefs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One rubric score from a judge model, or an error saying why not.

    Never raises, the same contract run_model and stream_model carry, for
    the same reason: this runs in a loop over many results and one bad
    reply must not end the pass.

    provider_prefs carries the boot data policy exactly as it rides every
    other payload. A judge call sends the response text of a run, which is
    model output about the user's prompt, so it is subject to the same
    data-handling promise as the run itself. There is no path here that
    can send it under a different policy.
    """
    # Whitespace is not an answer, and paying a judge to grade it is
    # money spent on a verdict nobody can use. parse_verdict at the
    # other end of this file already applies the same test to what comes
    # back; this applies it to what goes out.
    if response_text is None or not response_text.strip():
        # A trial that produced no text. Scored without asking anyone,
        # because there is nothing to grade and paying a judge to say so
        # would be spending money to learn what the row already says.
        return {
            "score": 0.0,
            "passed": None,
            "detail": "no response text: the trial did not complete",
            "generation_id": None,
            "billed_cost_usd": None,
            "error": None,
        }
    out: dict[str, Any] = {
        "score": None,
        "passed": None,
        "detail": None,
        "generation_id": None,
        "billed_cost_usd": None,
        "error": None,
    }
    payload: dict[str, Any] = {
        "model": judge_model,
        "messages": judge_messages(rubric, reference, response_text),
        "max_tokens": JUDGE_MAX_TOKENS,
        # THE JUDGE GETS THE RESERVATION TOO, and it needs it most. Its
        # budget is 512 tokens against a rubric it has never seen, which
        # is exactly the shape that invites a model to think past its
        # cap; a judge that reasons itself out of a verdict returns no
        # score, and the trial is recorded as unscored rather than as
        # failed. 256 tokens of thinking still fits a rubric's
        # deliberation, and 256 visible is far more than "a number and a
        # sentence" needs.
        #
        # NO RESERVATION, and it is the GENERAL rule that says so
        # rather than a special case here. reasoning_reservation
        # computes half of 512, sees 256 fall below the pinned 1024
        # minimum, and returns nothing. See JUDGE_SENDS_NO_RESERVATION
        # beside JUDGE_MAX_TOKENS for the arithmetic in full.
        #
        # Called rather than omitted on purpose: if the judge's budget
        # were ever raised past the point where the arithmetic works,
        # this line starts reserving without anybody having to remember
        # that it should.
        **reasoning_reservation(JUDGE_MAX_TOKENS, None, True),
    }
    if provider_prefs:
        payload["provider"] = provider_prefs
    try:
        response = await client.post(
            OPENROUTER_URL, json=payload, timeout=JUDGE_TIMEOUT_S
        )
    except httpx.HTTPError as exc:
        out["error"] = f"judge request failed: {type(exc).__name__}"
        return out
    if response.status_code != 200:
        out["error"] = f"judge returned HTTP {response.status_code}"
        return out
    try:
        data = response.json()
    except ValueError:
        out["error"] = "judge returned a malformed body"
        return out
    if not isinstance(data, dict):
        out["error"] = "judge returned a malformed body"
        return out

    # Captured before the verdict is parsed, so a judge that charged for
    # an unparseable reply still records what it charged. Money spent is
    # money spent, whatever came back for it.
    out["generation_id"] = _as_label(data.get("id"))
    usage = data.get("usage")
    if isinstance(usage, dict):
        out["billed_cost_usd"] = as_money(usage.get("cost"))

    # Through _flatten_content, the same extractor run_model uses, so a
    # provider that answers in content parts rather than a bare string is
    # read identically here. A second extractor would be a second thing
    # to keep in step with providers.
    try:
        content = data["choices"][0]["message"]["content"]
    except (LookupError, TypeError):
        out["error"] = "judge returned a malformed body"
        return out
    verdict = parse_verdict(_flatten_content(content))
    if "error" in verdict:
        out["error"] = verdict["error"]
        return out
    out["score"] = verdict["score"]
    out["detail"] = verdict["reason"]
    # passed stays None for a judge, deliberately. A rubric defines a
    # graded score; turning it into a pass or a fail needs a threshold
    # the rubric never stated, and inventing one here would be the bench
    # asserting a cutoff nobody chose. That is rule two applied to the
    # scorer's own output. The report shows judge scores as means and
    # says so, rather than quietly manufacturing a pass rate.
    return out
