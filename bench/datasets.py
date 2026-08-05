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
        }
        known = {"id", "prompt", "system", "reference", "rubric", "scorer"}
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
