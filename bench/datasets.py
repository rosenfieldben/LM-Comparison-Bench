"""Dataset loading. Pure functions over bytes, no I/O policy of its own.

A dataset is a JSONL file, one task per line. The module parses and
validates; the caller decides where bytes come from, which keeps the
loader testable without a filesystem and keeps "where may the bench read
from" a question the API boundary answers rather than this module.

The dataset's version IS its content. There is no version field to keep
in sync, no edit that can leave a stale label behind: the digest is
sha256 over the raw bytes, the same content-derived discipline the asset
rev and the catalog digest already use. Two experiments citing the same
digest ran the same tasks, and that is checkable rather than promised.
"""

import hashlib
import json
import re
from typing import Any

from bench.extract import MAX_ATTACHMENTS
from bench.models import as_text

# Bounds, not preferences. A dataset is user-supplied input that the
# runner will turn into paid upstream calls, so a file that would spend
# a fortune or exhaust memory should be refused at the door where the
# message can be useful, rather than part way through a run that has
# already cost money. The numbers are generous for real use: 2000 tasks
# at 5 repeats is 10000 trials per model, far past what anyone runs on
# a localhost bench.
MAX_TASKS = 2000
MAX_PROMPT_CHARS = 100_000
MAX_FIELD_CHARS = 20_000
MAX_ID_CHARS = 200

# The scorers this phase ships. Kept here rather than imported from
# bench.scoring so that validation stays a pure string check and the two
# modules do not depend on each other; scoring owns the implementations,
# datasets owns the contract with the file.
DETERMINISTIC_SCORERS = ("exact", "normalized_exact", "contains", "regex")
JUDGE_SCORER = "judge"
SCORERS = (*DETERMINISTIC_SCORERS, JUDGE_SCORER)

# Bounded so a pathological pattern cannot be compiled at all. Python's
# re has no timeout, so the defense has to be at admission time: refuse
# patterns long enough to hide catastrophic backtracking, and scoring
# bounds the subject string separately.
MAX_PATTERN_CHARS = 500

# The four fields a rendition pin has, and the three kinds it may name.
# Duplicated from neither RenditionPin nor row_rendition but agreeing
# with both: this module cannot import the API boundary (which imports
# it), and a pin's shape is part of the DATASET's contract, so the check
# belongs to the file format. A cross-module test asserts the three
# spellings name the same four fields.
#
# THE THIRD KIND IS PHASE L'S, and it is here because a snapshot is an
# attachment like any other once it is stored: a task cites its digest
# or pins its reading, and a vocabulary that stopped at two would refuse
# the citation one door before the resolution that would have honored
# it. bench.extract.kind_of is what decides a kind; this is what a
# dataset is allowed to CLAIM, and the cross-module test holds the two
# in agreement rather than an import, since this module cannot reach
# extract without reaching the boundary that imports it.
PIN_FIELDS = ("digest", "extractor", "extractor_version", "kind", "capture_id")
PIN_KINDS = ("document", "image", "snapshot")

# THE FOUR A PIN MUST NAME, and the fifth it may. capture_id is the
# fourteenth review's H2: which CAPTURE of a snapshot a pin means, a row
# id on the bench's captures table. A document or an image has no
# capture, and a snapshot cited without one resolves to its latest
# capture at creation, the same way a bare digest resolves to its row's
# own reading; so the fifth is optional where the four are not, and a
# pin naming it is honored verbatim or refused like any other part.
PIN_REQUIRED = ("digest", "extractor", "extractor_version", "kind")

# sha256, lowercase hex. The same shape RenditionPin enforces at the API
# boundary, checked here because a dataset that cites a malformed digest
# should be refused while the author is still looking at the file, not
# at experiment creation with a lookup miss that reads like a missing
# upload.
#
# \Z AND NOT $, which is the whole of the thirteenth review panel's F.
# Python's $ also matches immediately before a trailing newline, so
# "a"*64 + "\n" passed this check, reached RenditionPin (min_length 64,
# max_length 64) at the API boundary and was refused there instead: the
# deferral this constant exists to prevent, arriving through the
# constant itself. \Z matches only at the end of the string.
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}\Z")


class DatasetError(RuntimeError):
    """A dataset file the bench will not run.

    RuntimeError rather than a custom hierarchy, matching the boot-time
    configuration errors: the caller's only sensible response is to show
    the message, and the message is written to be shown.
    """


def _fail(line_no: int | None, message: str) -> None:
    where = f"line {line_no}: " if line_no is not None else ""
    raise DatasetError(f"{where}{message}")


def _checked_text(
    value: object, line_no: int, field: str, limit: int, required: bool
) -> str | None:
    """One optional string field, or None, through the total field rule.

    A non-string is not coerced. A dataset that stored 42 as a reference
    means something the bench cannot guess, and guessing "42" would make
    a scorer compare against a value nobody wrote.
    """
    if value is None:
        if required:
            _fail(line_no, f"{field} is required")
        return None
    text = as_text(value)
    if text is None:
        _fail(line_no, f"{field} must be a string, got {type(value).__name__}")
        return None  # unreachable; _fail raises. Keeps mypy honest.
    if required and text == "":
        _fail(line_no, f"{field} must not be empty")
    if len(text) > limit:
        _fail(line_no, f"{field} is {len(text)} characters, limit is {limit}")
    return text


def _checked_scorer(spec: object, line_no: int) -> dict[str, Any] | None:
    """The task's scorer spec, validated against what scoring can run.

    Validated here rather than at scoring time because a typo in a
    scorer name should cost nothing. Discovering it after a run has paid
    for every trial is discovering it in the most expensive place.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        _fail(line_no, f"scorer must be an object, got {type(spec).__name__}")
        return None
    kind = as_text(spec.get("kind"))
    if kind not in SCORERS:
        _fail(
            line_no,
            f"scorer kind {spec.get('kind')!r} is not one of {', '.join(SCORERS)}",
        )
    out: dict[str, Any] = {"kind": kind}
    if kind == "regex":
        pattern = _checked_text(
            spec.get("pattern"), line_no, "scorer.pattern", MAX_PATTERN_CHARS, True
        )
        try:
            re.compile(pattern or "")
        except re.error as exc:
            _fail(line_no, f"scorer.pattern is not a valid regex: {exc}")
        out["pattern"] = pattern
    if kind == JUDGE_SCORER and "pass_threshold" in spec:
        # The cutoff, when the rubric's author states one. Optional, and
        # the bench never supplies a default: a rubric defines a graded
        # score, so turning it into a pass or a fail is a decision only
        # the person who wrote the rubric can make. Absent means the
        # score stands alone and the report says so.
        raw = spec["pass_threshold"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            _fail(
                line_no,
                f"scorer.pass_threshold must be a number, got {type(raw).__name__}",
            )
        threshold = float(raw)
        if not 0.0 <= threshold <= 1.0:
            _fail(line_no, f"scorer.pass_threshold {threshold} is outside [0, 1]")
        out["pass_threshold"] = threshold
    # Unknown keys are refused rather than ignored, the same rule the API
    # boundary applies with extra="forbid". A silently dropped key in a
    # dataset is a scoring setting the author believes is in force.
    #
    # Per kind, not one flat set, and that is the difference between a
    # rule and the appearance of one: pattern means nothing to exact, and
    # pass_threshold means nothing to a deterministic scorer that already
    # produces a pass. Allowing them everywhere would accept both and
    # ignore both, which is the very failure this check exists to stop.
    allowed = {"kind"}
    if kind == "regex":
        allowed.add("pattern")
    if kind == JUDGE_SCORER:
        allowed.add("pass_threshold")
    extra = sorted(set(spec) - allowed)
    if extra:
        _fail(line_no, f"scorer has unknown keys: {', '.join(extra)}")
    return out


def _checked_attachments(
    value: object, line_no: int, task_id: str
) -> list[dict[str, Any]] | None:
    """A task's declared documents, as pin dicts, or None.

    TWO SPELLINGS, ONE MEANING, and the difference is who chooses the
    reading. A bare DIGEST cites content and leaves the reading to
    experiment creation, which resolves it to whatever the bench holds
    for those bytes at that moment and freezes the result. A full
    four-part PIN states the reading too, and creation then honors it
    verbatim or refuses; nothing in between, because a pin that gets
    quietly substituted is worse than one that gets rejected.

    Both are normalized to dicts here so the caller has one shape to
    walk. A digest-only entry is a dict with one key, which is not a
    partial pin but a different declaration, and creation can tell them
    apart by asking whether "extractor" is present.

    NAMES THE TASK, NOT ONLY THE LINE. Every other field in this file
    fails with a line number, which is right for a field a person reads
    in place. Attachments are cited by digest and fixed by looking one
    up, so the message has to carry the id the author will search their
    notes for.
    """
    if value is None:
        return None
    where = f"task {task_id!r}: attachments"
    if not isinstance(value, list):
        _fail(line_no, f"{where} must be a list, got {type(value).__name__}")
        return None
    if not value:
        # An empty list is a declaration of no documents, which is what
        # omitting the key already says, spelled a second way. The API
        # boundary refuses it with min_length=1 for the same reason:
        # absence gets exactly one spelling or two readers will
        # eventually disagree about what it meant.
        _fail(
            line_no,
            f"{where} is empty. Omit the key entirely for a task with no "
            "documents; an empty list is a second spelling of absence.",
        )
    if len(value) > MAX_ATTACHMENTS:
        _fail(
            line_no,
            f"{where} lists {len(value)} documents, limit is {MAX_ATTACHMENTS}",
        )
    out: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        at = f"{where}[{index}]"
        if isinstance(entry, str):
            if not DIGEST_PATTERN.match(entry):
                _fail(
                    line_no,
                    f"{at} is not a sha256 digest: it must be 64 lowercase "
                    "hex characters. Upload the file and cite the digest "
                    "the bench returns.",
                )
            out.append({"digest": entry})
            continue
        if not isinstance(entry, dict):
            _fail(
                line_no,
                f"{at} must be a digest string or a rendition object, got "
                f"{type(entry).__name__}",
            )
            return None
        missing = [field for field in PIN_REQUIRED if field not in entry]
        if missing:
            # A PARTIAL PIN IS REFUSED RATHER THAN COMPLETED. Filling the
            # gaps from the current store would make the dataset's
            # declaration depend on when it was read, which is exactly
            # what pinning exists to stop; and a three-field pin is
            # indistinguishable from a typo in the fourth.
            _fail(
                line_no,
                f"{at} is missing {', '.join(missing)}. A rendition names "
                f"all four of {', '.join(PIN_REQUIRED)}, or cite the digest "
                "alone and let the experiment freeze the reading.",
            )
        unknown = sorted(set(entry) - set(PIN_FIELDS))
        if unknown:
            _fail(line_no, f"{at} has unknown keys: {', '.join(unknown)}")
        digest = entry.get("digest")
        if not isinstance(digest, str) or not DIGEST_PATTERN.match(digest):
            _fail(
                line_no,
                f"{at}.digest is not a sha256 digest: it must be 64 "
                "lowercase hex characters.",
            )
        pin: dict[str, Any] = {"digest": digest}
        for field in ("extractor", "extractor_version"):
            text = _checked_text(entry.get(field), line_no, f"{at}.{field}", 64, True)
            pin[field] = text
        kind = entry.get("kind")
        if kind not in PIN_KINDS:
            _fail(
                line_no,
                f"{at}.kind is {kind!r}, not one of {', '.join(PIN_KINDS)}",
            )
        pin["kind"] = kind
        capture = entry.get("capture_id")
        if capture is not None:
            # A positive integer or nothing. bool is refused explicitly
            # because True is an int to isinstance and "capture_id: true"
            # is a typo, not a row.
            if isinstance(capture, bool) or not isinstance(capture, int) or capture < 1:
                _fail(
                    line_no, f"{at}.capture_id is {capture!r}, not a positive integer"
                )
            if kind != "snapshot":
                _fail(
                    line_no,
                    f"{at}.capture_id names a capture, and only a snapshot "
                    f"has captures; this pin is a {kind}.",
                )
        pin["capture_id"] = capture
        out.append(pin)
    return out


def parse_dataset(raw: bytes, name: str = "dataset") -> dict[str, Any]:
    """Tasks plus the digest of the bytes they came from.

    Takes bytes, not a path, so the digest is over exactly what was read
    and cannot drift from it: hashing a re-read of the file would leave a
    window in which the two disagree. The name is for messages only.

    Blank lines are skipped, because a trailing newline is not an error
    and refusing one would be pedantry the user has to work around.
    """
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{name} is not valid UTF-8: {exc}") from None

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(tasks) >= MAX_TASKS:
            _fail(line_no, f"more than {MAX_TASKS} tasks; split the file")
        try:
            row = json.loads(line)
        except ValueError as exc:
            _fail(line_no, f"not valid JSON: {exc}")
        if not isinstance(row, dict):
            _fail(line_no, f"expected an object, got {type(row).__name__}")
        task_id = _checked_text(row.get("id"), line_no, "id", MAX_ID_CHARS, True)
        assert task_id is not None
        if task_id in seen:
            # Ids are how a score row points back at a task and how a
            # report groups repeats. A duplicate would silently merge two
            # different tasks' trials into one row of the report.
            _fail(line_no, f"duplicate task id {task_id!r}")
        seen.add(task_id)
        scorer = _checked_scorer(row.get("scorer"), line_no)
        task: dict[str, Any] = {
            "id": task_id,
            "prompt": _checked_text(
                row.get("prompt"), line_no, "prompt", MAX_PROMPT_CHARS, True
            ),
            "system": _checked_text(
                row.get("system"), line_no, "system", MAX_FIELD_CHARS, False
            ),
            "reference": _checked_text(
                row.get("reference"), line_no, "reference", MAX_FIELD_CHARS, False
            ),
            "rubric": _checked_text(
                row.get("rubric"), line_no, "rubric", MAX_FIELD_CHARS, False
            ),
            "scorer": scorer,
            # The documents this task carries, by digest or by full pin.
            # None when the task declares none, which is every task in
            # every dataset written before Phase M: rule one holds at
            # the dataset layer too, and such a file parses to exactly
            # what it parsed to before.
            "attachments": _checked_attachments(
                row.get("attachments"), line_no, task_id
            ),
        }
        known = {
            "id",
            "prompt",
            "system",
            "reference",
            "rubric",
            "scorer",
            "attachments",
        }
        unknown = sorted(set(row) - known)
        if unknown:
            _fail(line_no, f"unknown keys: {', '.join(unknown)}")
        # A judge scorer with no rubric cannot be run: the rubric IS the
        # scoring instruction, and a judge asked to score against nothing
        # would return an opinion about something nobody specified.
        #
        # Empty counts as absent here, unlike the optional fields above.
        # A blank rubric is not a rubric, and the difference between
        # omitting the key and writing "" is a typo away.
        # Normalized first, then tested, so a reference of "   " is
        # refused exactly like "". Whitespace is not a value a scorer can
        # compare against: `contains` normalizes both sides and would end
        # up asking whether the response contains the empty string, which
        # every response does. Same trap as the empty case, wearing a
        # disguise that looks deliberate in a file.
        if (
            scorer
            and scorer["kind"] == JUDGE_SCORER
            and not (task["rubric"] or "").strip()
        ):
            _fail(line_no, "scorer kind judge requires a non-empty rubric")
        # The reference-comparing scorers have the same problem in the
        # other direction, and an empty reference is worse than a missing
        # one because it scores rather than failing: every response
        # contains the empty string, so a `contains` task with reference
        # "" gives every model on every repeat a perfect 1.0 and the
        # report has no way to know the number is meaningless.
        if (
            scorer
            and scorer["kind"] in ("exact", "normalized_exact", "contains")
            and not (task["reference"] or "").strip()
        ):
            _fail(
                line_no,
                f"scorer kind {scorer['kind']} requires a non-empty reference",
            )
        tasks.append(task)

    if not tasks:
        raise DatasetError(f"{name} has no tasks")
    return {"name": name, "digest": digest, "tasks": tasks}
