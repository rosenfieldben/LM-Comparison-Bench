# Phase N: experiments from the browser, and a datasets door

Repo: github.com/rosenfieldben/LM-Comparison-Bench. Branch from main at
f3eeebc (the Phase L merge, tagged v0.4.0). Branch name
`phase-n-experiments-ui`. Nothing has landed on main since L except what
the log shows; if Dependabot merges appear before you branch, branch from
the newest and reinstall the hashed venv before the first gate.

## What this phase is for

The evaluation layer has been complete on the backend since Phase I and
the browser has never been able to drive it. Today a person can create,
start, score and stop an experiment only with curl, and a dataset is a
file on the server's disk named by path. The browser reads the list and
renders the report, and that is all it does.

After this phase a person can, without leaving the page: compose a set of
prompts into a dataset, store it, create an experiment over it with the
lineup on screen, see what it will cost at most, start it, watch it run,
stop it, score it with a judge chosen from the catalog, and open the
report. Every one of those steps already has a door on the server. The
work is one new door for datasets, a small widening of three existing
doors so they accept a stored dataset by digest, and the frontend.

The bench's laws apply unchanged. Money moves on Start and on nothing
else. Blank is not sent. Only what was set is recorded. Records never
rewrite. The server validates; the browser composes and shows the
server's sentence.

## What already exists, so it is not rebuilt

Server, all in `bench/main.py`:

| Door | Takes | Notes |
| --- | --- | --- |
| `POST /experiments` | `ExperimentCreate` (name, dataset_path, lineup, budget, params, repeats, task_order_seed, estimand_mode, primary_metric, provider_pins, quantizations, attachments_mode, halt_on_refusal) | Free. Returns id and `projected_cost` (input estimate, output ceiling, total, unpriced members). Dataset read and digest recorded here. |
| `POST /experiments/{id}/start` | `ExperimentStart` (dataset_path) | Re-reads, re-checks the digest, refuses 422 on drift. Money moves here. |
| `POST /experiments/{id}/stop` | nothing | Honest interruption; un-run tasks leave no line. |
| `GET /experiments/{id}/progress` | nothing | SSE, absolute counters every frame, rejoinable. |
| `POST /experiments/{id}/score` | `ScoringStart` (dataset_path, judge_model optional) | 409 while created or running; 409 while another pass is active; 422 on digest drift. |
| `GET /experiments`, `GET /experiments/{id}`, `/report`, `/export.jsonl` | | Already consumed by `static/experiments.js` for the list and the report. |
| `GET /models` | | The catalog the lineup and the judge picker both draw from. |

Pure module: `bench/datasets.py` with `parse_dataset(raw, name)` over bytes,
bounds `MAX_TASKS` 2000, `MAX_PROMPT_CHARS` 100000, `MAX_FIELD_CHARS`
20000, `MAX_ID_CHARS` 200, `SCORERS` (exact, normalized_exact, contains,
regex, judge), attachment pins over `PIN_KINDS` (document, image,
snapshot). `read_dataset(path)` in `main.py` is the boundary read over it.

Browser: `static/experiments.js` (list and report), the composer's lineup
chips and `+ Controls` panel (`ExperimentParams` shape, blank not sent),
the `+ Snapshot` panel in `attach.js`, the attachment chips and library.
Classic-script modules, full CSP, versioned assets, no build step, no
dependencies. Playwright suite in `tests/browser/` against
`stub_openrouter.py`.

## Scope, in four parts

Checkpoint after N2. N1 and N2 are the backend and the builder; N3 and N4
are the lifecycle and the judge. The checkpoint exists because N2 is
where the rulings below get exercised against a real screen and because
N3 is where money starts moving from a button.

### N1. The datasets door (server)

A dataset becomes a stored artifact the way an attachment is, so a
comparison can cite one by digest and a browser can create one without a
path on anybody's disk.

- New table `datasets`: digest (primary key, sha256 of the stored bytes),
  name, created_at, content (the JSONL bytes, verbatim), task_count, and
  a `scorers_json` summary of which scorer kinds the file uses. Additive
  migration in the hash-pinned series; a new era snapshot fixture.
- `POST /datasets`, JSON and not a form, for exactly the reason
  `AttachmentCreate` gives (a non-JSON POST is a CORS simple request).
  Body: `name` (1 to 255) and `content` (the JSONL text, bounded by a
  constant sized from `MAX_TASKS * (MAX_PROMPT_CHARS + MAX_FIELD_CHARS *
  4)` or a smaller documented number, your call, named once). The server
  runs `parse_dataset` on the bytes it received, refuses 422 with the
  parser's own sentence naming the line, computes the digest itself and
  never accepts one, and returns the existing row on identical content
  (the earlier name wins, as `save_attachment` does). 201 with the
  record.
- `GET /datasets` (list, newest first, bounded) and `GET
  /datasets/{digest}` (the record and the content).
- `ExperimentCreate`, `ExperimentStart` and `ScoringStart` each take
  `dataset_digest` as an alternative to `dataset_path`. Exactly one of
  the two; both present or both absent refuses 422 naming the rule. By
  digest, the bytes come from the store and the drift check is trivially
  true, because the digest is the key. By path, nothing changes: the curl
  workflow in the README keeps working byte for byte, and the existing
  tests that drive it must pass unmodified as the proof.
- `experiments.dataset_name` keeps recording the name under which the
  bytes were read, whichever door supplied them.

Rulings made here, with reasons, so the coder does not relitigate them:

- Datasets are stored whole and cited by digest, never by row id, so the
  identity a later record carries is the identity of the tasks and not
  of a row that could be renamed. Same law as attachments.
- Datasets still reference documents and never carry them. A task cites
  attachment digests exactly as before; the datasets door does not
  inspect attachment existence at store time, because `create_experiment`
  already does that at the point where it matters and doing it twice
  invites two sentences that disagree.
- No delete door. Records never rewrite, and an experiment cites the
  digest.
- Export stays at schema 7. `dataset_digest` already travels and the
  stored content is reachable at `GET /datasets/{digest}`. Ruling
  requested at checkpoint on whether an export should ever inline the
  dataset body; my position is no, because the export is a claim about
  what ran and the dataset is retrievable by the identity it cites.

### N2. The dataset builder (browser)

A panel, collapsed by default like `+ Controls`, that composes tasks and
stores them through `POST /datasets`. Three ways in:

1. **Rows.** Add a task: id, prompt, scorer kind (select over `SCORERS`),
   and then the fields that scorer needs and only those: reference for
   the comparing scorers, pattern for regex, rubric for judge. Optional
   system message. Optional attachments, chosen from the existing
   attachment library by chip, which is how a repository snapshot enters
   a task: it is an attachment with kind `snapshot` and it gets cited
   like any other. The row count in the builder is bounded at 50
   (proposed; ruling requested). Past that, the second way in.
2. **Paste or upload JSONL.** A textarea and a file input for a file
   already in dataset form. The browser does not parse it beyond
   counting lines for the label; the server parses it.
3. **From the prompt library.** Each saved prompt becomes one task with
   the scorer left for the person to set. This is the cheap route from
   "I have eight prompts" to "I have a dataset", which is the case this
   phase is for.

Rules:

- **The browser composes; the server validates.** The builder writes
  JSONL and shows `parse_dataset`'s sentence verbatim, with its line
  number, next to the row that line came from. No second validator in
  JavaScript. A required-field nudge on a row (an empty prompt greys
  Store) is a courtesy and not a rule, and it must never refuse what the
  server would accept or accept what the server would refuse; a test
  proves the nudge set is a strict subset of the server's refusals.
- **Nothing is pre-filled.** An id is not generated for the person. An
  empty id box is an empty id box. (Ruling requested: a generated id
  like `task-1` is convenient and is also a decision the person did not
  make. My position is to leave it blank.)
- **A stored dataset is shown by digest and name**, with its task count
  and scorer kinds, in a library list next to the attachment library.
  Selecting one loads nothing back into the builder; it selects it for
  the experiment. Editing a stored dataset is storing a new one.
- Ceilings shown before Store: task count against `MAX_TASKS`, prompt
  length against `MAX_PROMPT_CHARS`, mirrored from the constants the
  same way `index.html` mirrors `ExperimentParams` bounds today, with
  the same comment saying so.

### N3. The experiment lifecycle (browser)

A panel that stands where the experiments list is now and grows it.

**Create.** Name; dataset (from the library, by digest); lineup is the
composer's lineup, the same chips, because one lineup surface is one
fewer thing that can disagree with itself; budget (standard or extended,
the same control the composer has); repeats (1 to 20); task order seed
(blank means file order and is not sent); controls read from the
existing `+ Controls` panel through the same `ExperimentParams` shape,
so rule one and rule two hold without a second implementation;
attachments mode (inline or native, and the control is disabled with a
sentence when the selected dataset cites no document, since the server
refuses it there); halt on refusal (checkbox, default checked, matching
the server default); primary metric (select over the scorer kinds the
selected dataset uses, blank means none declared).

`estimand_mode` is shown as routed service or underlying model. In
underlying model the strict-mode fields (`provider_pins` per lineup
member, `quantizations`) are **not built in this phase**. The selector is
present so the record can be honest; choosing underlying model with no
pins is a legal request the server already accepts. Ruling requested at
checkpoint: build the strict-mode fields in N3, or leave them for their
own phase as the deferred backlog has them. My position is to defer,
because strict mode carries its own refusal family (offline refusal,
capability checks, `allow_fallbacks` false) and each needs a browser
proof, and that is a phase, not a form.

Create is free and the button says so. The response's `projected_cost`
is shown as the README describes it: output as a ceiling, input as an
estimate, total, and the unpriced members named when any figure is
None, so a total missing one arm never reads as the total.

**Start.** A second button, enabled only once a created experiment is
selected, that sends `dataset_digest` and moves money. The button
carries the projection beside it. There is no confirm dialog; the
projection on the button is the confirmation, the same way the composer
has none.

**Watch.** Subscribe to `/progress` on start and on selecting a running
experiment. Absolute frames, so a dropped frame costs nothing and a
reload resubscribes. Counters: done, failed, refused, total, status and
status_detail verbatim. Reconnection is the ordinary case and is tested
as one: kill the stream mid-run in the browser test and assert the
counters resume at the current values.

**Stop.** Present while running. The server's status_detail after a stop
is shown as is.

**List.** The existing rows gain a selected state and the report opens
from the selected experiment as it does now.

### N4. Scoring and the judge (browser)

On a selected experiment whose status is neither created nor running:

- A **judge** select over the catalog, the same list the lineup draws
  from, filtered to nothing (the server decides which routes can judge;
  the README's rule that a judge route must not have mandatory reasoning
  is enforced where it lives, not in a dropdown). Required only when the
  selected dataset's `scorers_json` includes `judge`; otherwise the
  select is hidden and the request omits `judge_model`, since blank is
  not sent.
- **Score** sends `dataset_digest` and the judge if any. The 409 for a
  pass already active on another experiment is shown verbatim, and the
  button stays enabled, because the server is the one that knows when
  the other pass ends. The 409 for created or running is not reachable
  from a button that is hidden in those states; a test asserts the
  button is absent then, and a second test sends the request anyway and
  asserts the server still refuses, because a hidden button is not a
  rule.
- **Self-judge flagging** from Phase I stays where it is, in the report.
  The browser adds nothing to the judge's blind view.
- After scoring, the report opens. Nothing else.

## Non-goals, named so they are not drifted into

- GitHub. No clone door, no fetch, no URL field. The composer fetches
  nothing and that ruling stands. Connecting a repository is the phase
  after this one, and this phase's job toward it is only that a
  snapshot attachment can be cited from a task in the builder.
- Pairwise judging, primary_judge, Parquet, the strict-mode fields
  (unless the checkpoint rules otherwise), rerun and reuse of
  experiments, editing or deleting a stored dataset, charts in the
  report.
- A dataset-level view of attachment renditions. Pins are checked at
  experiment creation as before.

## House law that binds this phase

The full record is in the repository and in the review history; these
are the lines a frontend-heavy phase is most likely to skip.

- Tombstones stash-proven in the reviewer's shape. Every proof names its
  window.
- Pre-state assertions on every new test: a test that cannot fail is
  worse than none, and freshly written tests get their own mutation
  re-run.
- Per-door mutation proofs and call-site enumeration for `POST
  /datasets`, both GETs, and each of the three widened doors. For the
  widened doors the mutation of interest is the exactly-one rule
  (remove the check, the test must fail) and the by-digest path
  (substitute a different stored digest, the experiment must record the
  one it was given).
- Declaration completeness and declaration transport: `dataset_digest`
  is a declaration, and it must land on the experiment row, the
  progress frame's parent record, the report and the export exactly as
  `dataset_path`-read digests do today. A test compares the two entry
  routes on identical bytes and asserts the recorded rows are equal in
  every field but `dataset_name`.
- Pinned-contract assertions: a mock cannot refuse. Browser tests run
  against the stub server and the real datasets door.
- Nothing branches on a model name. The judge select has no filter, and
  the AST tripwire stays green.
- The filesystem-posture walk is a permanent test. The datasets door
  reads no file when content arrives in the body; if you add a
  filesystem call anywhere, the walk fails and you map it to a posture.
- Comment-code agreement. `read_dataset`'s docstring argues for
  unrestricted paths; it stays true and the new door's docstring says
  why content by body needs no path at all.
- ruff from the venv binary, never from PATH. Reinstall the hashed venv
  after any pull touching requirements.
- Records never rewrite; NULL means one thing; a controls set that is
  blank records nothing rather than an empty object, and an experiment
  created from the browser with every control blank must produce a row
  identical to one curl creates with `params` absent.

## Proofs and gates

Baseline on main at f3eeebc: 1091 unit tests, the snapshot suite, 25
transport, capture and posture tombstones under the four-spelling
poisoned hermeticity gate, the JS suite, biome, ruff, mypy, pip-audit,
and the browser suite in CI.

New browser proofs, at minimum:

- Build three tasks by rows (one each of exact, regex, judge), store,
  create an experiment over the stored digest with two stub models,
  start, watch the counters reach total, score with a judge, open the
  report. The critical path.
- The projection's `unpriced` list renders the member names when the
  stub returns no price for one.
- Stop mid-run and assert the un-run tasks are absent from the report.
- Kill and resume the progress stream.
- Store the same content twice and assert one row, the first name.
- A server refusal with a line number lands beside the row that
  produced that line.
- The nudge-subset proof described in N2.

Every new door has its mutation proof and its call-site enumeration in
the commit body with the numbers remeasured, not carried from the
previous commit. Gate lines name the venv binary.

## Commit and review shape

Four commits at least, one per part, each green on its own. Checkpoint
after N2 with the rulings above answered. After N4, my pass, then one
external merge-readiness review sized to the change: recommended tier
is a single review with the frontend and the three widened doors as its
lenses, and the cheaper alternative is my pass alone with the risk that
a frontend-heavy phase is exactly where a hidden-button-as-rule defect
hides. Fix known defects in hand, stop the cycle, merge. Tag v0.5.0 at
the merge.

## Rulings requested at the checkpoint

1. Builder row cap (50 proposed) and the `POST /datasets` content bound.
2. Generated task ids in the builder: none (my position) or `task-N`.
3. Export: stays at schema 7 (my position) or inlines the dataset body.
4. Strict-mode fields in N3: defer (my position) or build.
