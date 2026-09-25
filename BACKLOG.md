# Deferred backlog

What the bench has deliberately not built yet, one entry per item. Each
entry says what the item is, which phase deferred it, the reason in a
sentence, and where that reason was first written, so the decision can be
read in its own words rather than in this file's. Where no reason was
written, the entry says so and gives none.

The rules for this file:

- **An entry is added by the phase that defers the item**, in that
  phase's commit.
- **An entry is removed by the phase that builds it**, and the removal is
  named in that phase's commit body, so it stands as a tombstone rather
  than a line that quietly went missing.
- **The order is the order entries were added, and nothing more.** This
  is not a priority list. Priority is decided when a phase is
  commissioned, not by where an item sits here.

The file was created in Phase N, N3. The entries for items earlier
phases deferred were back-filled then, from what those phases wrote,
and are listed by the phase that deferred them.

A thing the bench has ruled out, rather than put off, is not on this
list: there is no delete door for a stored dataset, because records
never rewrite (Phase N, N1). Nor is a thing since decided: sending
`require_parameters`, which the README once called a decision for a
later phase, became the `underlying_model` estimand in Phase I (commit
a23b257).

## Httpx2 in the test client

- **What:** moving the unit suite's TestClient from httpx to httpx2, which
  Starlette's deprecation warning asks for.
- **Deferred by:** Phase F.
- **Reason:** no reason was written.
- **First written:** requirements-dev.txt ("Migrating to httpx2 is
  deferred"), commit bd00ab9; the note moved to requirements-dev.in in
  commit d835c04 (Phase F.3), where it stands now.

## Atomic admission for the spend ceiling

- **What:** a full reservation ledger, so admitting an upstream call
  atomically reserves its worst-case cost against the ceiling.
- **Deferred by:** Phase F.1, and restated when Phase H.1 rewrote the
  paragraph.
- **Reason:** no reason was written.
- **First written:** README, Setup ("A full reservation ledger (atomic
  admission) is deliberately deferred"), commit 095e293.

## Pairwise judging with position swapping

- **What:** a judge that compares two responses side by side, run in both
  orders to cancel position bias.
- **Deferred by:** Phase I, and again by Phase N's non-goals.
- **Reason:** it arrives with its swap machinery or not at all.
- **First written:** README, Scoring, commit 58b6346.

## Removing `RatingsSubmit.blind`

- **What:** dropping the `blind` field from the ratings request, which no
  client in the repository sends any more.
- **Deferred by:** Phases I.2 and I.3.
- **Reason:** removing it is a breaking change to the request shape,
  waiting for a moment when breaking changes are made deliberately
  rather than as a side effect of a correctness fix.
- **First written:** bench/main.py, the comment on `RatingsSubmit.blind`
  ("a NAMED DEFERRAL"), commit c6e878e.

## The budget-versus-window check for every run

- **What:** refusing any run whose completion budget does not fit the
  model's context window, not only runs that carry a document.
- **Deferred by:** Phase K.
- **Reason:** adding a refusal to every unattached comparison would be a
  new rule wearing that phase's clothes, since a bare prompt cannot
  approach a window the way a composed document can.
- **First written:** bench/main.py, the comment on the attachment window
  check ("a named deferral and stays one"), commit 8f02dc6.

## GitHub

### Clone door

- **What:** connecting a repository by URL and cloning it from the
  bench, rather than walking a clone already on disk.
- **Deferred by:** Phase L, and again by Phase N's non-goals ("Connecting
  a repository is the phase after this one").
- **Reason:** the snapshot door fetches nothing, and a fetching door is a
  change to the single-outbound-destination posture that needs its own
  phase.
- **First written:** commit 661df53 ("IT FETCHES NOTHING");
  docs/phases/phase-n-prompt.md, "Non-goals".

### Member listing

- **What:** a dry run of the snapshot composer. Same root, same include
  patterns, same ceilings; it returns the paths and sizes it would have
  composed and the refusals it would have raised, without composing. It
  is a snapshot-side endpoint that does not need the clone door, and it
  is useful on a local clone today.
- **Deferred by:** Phase L, which ruled that a snapshot's ceiling is the
  one a set of attachments has.
- **Reason:** under that ceiling a real repository refuses often, so a
  selection has to be made from fact before it is composed.
- **First written:** the ruling and its select-from-fact reason are in
  the README, "A repository snapshot" ("The ceiling is the same one a set
  of attachments has"), commit f239cac. The member listing itself was not
  written in the repository before this file.

## Composed-size checks in native mode

- **What:** the per-task composed-size and context-window refusals that
  inline mode applies, for an experiment whose documents go as native
  content parts.
- **Deferred by:** Phase M.
- **Reason:** an image's cost is not a character count.
- **First written:** bench/main.py, experiment creation ("NOT WEIGHED IN
  NATIVE MODE"), commit 22e130b; the README named it a deferral in
  commit 157a38c ("both are deferrals rather than oversights").

## `primary_judge`

- **What:** a judge designated as an experiment's headline, the way
  `primary_metric` designates a scorer; any design has to respect that a
  series is a (scorer, judge) pair and there is no combined cross-judge
  number anywhere (README, Reports, commit 70390e4).
- **Deferred by:** Phase N (non-goals).
- **Reason:** no reason was written.
- **First written:** docs/phases/phase-n-prompt.md, "Non-goals".

## Parquet export

- **What:** an export format beside `export.jsonl`, which stays at schema
  7 by the N2 checkpoint ruling.
- **Deferred by:** Phase N (non-goals).
- **Reason:** no reason was written.
- **First written:** docs/phases/phase-n-prompt.md, "Non-goals".

## Strict-mode fields in the browser

- **What:** `provider_pins` per lineup member and `quantizations`, entered
  in the experiment panel for an `underlying_model` experiment. It is not
  deferred to N3: N3 builds the `estimand_mode` selector only, and
  choosing underlying model with no pins is a request the server already
  accepts.
- **Deferred by:** Phase N, ruled at the N2 checkpoint to go past Phase N
  into a phase of its own.
- **Reason:** strict mode carries its own refusal family (offline
  refusal, capability checks, `allow_fallbacks` false), each of which
  needs a browser proof, and that is a phase, not a form.
- **First written:** docs/phases/phase-n-prompt.md, N3 ("that is a
  phase, not a form"). The N2 checkpoint ruling that sends it past Phase
  N is recorded here and in N3's commit body.

## Experiment rerun and reuse

- **What:** starting a new experiment from an existing one's
  declarations, or running an existing one again. Records never rewrite,
  so a rerun would be a new experiment, and which declarations it
  carries over is the design left open.
- **Deferred by:** Phase N (non-goals).
- **Reason:** no reason was written.
- **First written:** docs/phases/phase-n-prompt.md, "Non-goals".

## Dataset editing

- **What:** opening a stored dataset in the builder to change it.
- **Deferred by:** Phase N (non-goals, and the N2 rules).
- **Reason:** a stored dataset is cited by the digest of its bytes, so an
  edited one is a different dataset and editing is storing a new one.
- **First written:** docs/phases/phase-n-prompt.md, N2 ("Editing a
  stored dataset is storing a new one"); static/datasets.js, "SELECTING
  LOADS NOTHING INTO THE BUILDER", commit 31f2bec.

## Charts in the report

- **What:** graphical renderings of the report's figures.
- **Deferred by:** Phase I, and again by Phase N's non-goals.
- **Reason:** a chart of means invites the eye to read a difference the
  intervals do not support.
- **First written:** static/experiments.js, header comment ("Prose first
  and no charts, which is a deliberate limit rather than a missing
  feature"), commit 1011d53.

## A dataset-level view of attachment renditions

- **What:** a view, per stored dataset, of which rendition each cited
  document resolves to.
- **Deferred by:** Phase N (non-goals).
- **Reason:** pins are checked at experiment creation, as they were
  before Phase N.
- **First written:** docs/phases/phase-n-prompt.md, "Non-goals" ("Pins
  are checked at experiment creation as before").

## Which figure a thresholded judge series ranks on

- **What:** the ranking at bench/report.py:744 orders on `score.mean`. A
  judge series with a declared pass threshold publishes both a mean and
  a pass rate, and the ranking uses the mean whether or not a threshold
  was declared, without saying so in the report. Decide whether
  `primary_metric` should be able to name the pass rate, and have
  `ranking.reason` state which figure the ordering used.
- **Deferred by:** Phase N, in the N3 checkpoint exchange.
- **Reason:** it is a declaration question, not a defect, and it belongs
  with `primary_judge` when that is designed.
- **First written:** no commit to cite. Noticed at the N2 checkpoint; the
  reason was written by the commissioner in the N3 checkpoint exchange.
