# Phase P: the scoring pass as a record, atomic admission, and the ancestor look

Repo: github.com/rosenfieldben/LM-Comparison-Bench. Branch from main at
ed00174 (the Phase O merge, tagged v0.6.0). Branch name
`phase-p-records`. If Dependabot merges land first, branch from the
newest and reinstall the hashed venv before the first gate. Commit this
file under `docs/phases/phase-p-prompt.md` in the first commit,
verbatim.

## What this phase is for

Phases N and O built the last two feature surfaces the bench was
commissioned for: experiments run from the page, and repositories
connect from it. This phase builds nothing a person will see as new. It
closes five BACKLOG entries whose common shape is that the record does
not say what happened:

- Recording that a judge request went out.
- Stopping a scoring pass at shutdown.
- Reporting a failed scoring pass.
- Atomic admission for the spend ceiling.
- A snapshot root inside a git directory.

The first three are one design (a scoring pass that leaves a record of
its own life and of every request it sent), the fourth is the oldest
entry on the list (Phase F.1), and the fifth is the one thing Phase O
left deliberately open, with the external review's design for closing
it already written. Each entry's removal from BACKLOG is the tombstone.

The bench's laws apply unchanged. Money moves on Start and on Score and
nowhere else. Records never rewrite. Only what happened is recorded, and
what happened is recorded once. The server validates; the browser shows
the server's sentence.

## What already exists, so it is not rebuilt

- `app.state.scoring_run`: a process-local dict (active, task, stop
  event, tasks, error). Nothing sets `stop`. The lifespan stops and
  awaits the trial runner and not the pass. `score_experiment` in
  `main.py` writes a `scores` row per judged trial after the reply, with
  `judge_generation_id` and `judge_billed_cost_usd` when the reply
  carried them; a call that timed out after sending, or was cancelled
  at shutdown, leaves no row.
- `report.judge_cost`: `total_usd`, `billed_calls`, `unpriced_calls`
  (a generation id and no figure), `rows_without_figure` (every judge
  row with no figure). The page renders the spend line and the
  rows-without-figure count beside it (6a25ae3, the N review's M3).
- The single scoring slot, released in a `finally` that covers the
  pass's first read (the N review's L8).
- The spend ceiling: `spend_ceiling_reached`, `enforce_spend_limit` at
  entry, `record_spend` at settlement, a post-admission recheck
  (`spend_refusal_result`), and the README's Setup paragraph with its
  "A full reservation ledger (atomic admission) is deliberately
  deferred" sentence, from 095e293. The eight-concurrent-batches
  regression test measures the bound the paragraph states.
- `bench/snapshot.py`: the walk over a `Tree` of eight operations,
  containment by descriptor, the git-directory signature (HEAD, config,
  objects, refs together) refused at the root or below it, never above,
  with `ROOT_IS_GIT_DIRECTORY` and `HOLDS_GIT_DIRECTORY` in the reader's
  words. The open-state pin
  `test_below_a_git_directorys_top_level_the_walk_cannot_see_it` in
  `tests/test_api.py` with four cases (remotes/, branches/,
  worktrees/wt/, logs/), each composing with the sentinel in the text
  today. By-path ancestor precedents at `refuse_while_cloning` and
  `_clone_for`.
- The filesystem and network posture walks, the hash-pinned migration
  series with era fixtures, export schema 8.

Baseline at ed00174 on ubuntu: 1525 unit collected (1520 pass, 5
reasoned skips), 319 browser, 72 node, ruff 0.16.5 from the venv, mypy,
biome 1.9.4, pip-audit, CI's pytest step `-rs --durations=10`.

## Scope, in three parts

Checkpoint after P1, because P1 is the schema change and the record
design, and the shape ratified there governs the other two.

### P1. The scoring pass as a record

Three BACKLOG entries close together because they are one missing
thing: a scoring pass exists only in `app.state.scoring_run`, so
nothing it does or fails to do is a record.

**Judge requests are recorded before they go out.** A new table
`judge_calls`: id, experiment_id, result_id, judge_model, sent_at, and
then the answer as separate columns filled by a second write that never
touches the first: answered_at, generation_id, billed_cost_usd, outcome
(`answered`, `timed_out`, `stopped`, `failed`) and, for `failed`, the
detail. The `scores` row for a judged trial cites `judge_call_id`. The
"records never rewrite" question is answered like this: the sent fact
and the answer fact are two facts, written at two times, and the second
write fills columns that were NULL, never changes a column that held a
value; a test asserts that with a mutant that overwrites `sent_at`. If
you would rather two tables (a sent row and an answer row keyed to it)
than NULL-then-filled columns, say so at the checkpoint with the reason;
either is acceptable, the law is that no written value changes.

`report.judge_cost` then reads `judge_calls`: `billed_calls`,
`unpriced_calls` (answered, no figure), `unanswered_calls` (sent, no
answer: timed out, stopped, or failed after sending), and
`rows_without_figure` is retired as a name, with its removal named in
the body and the page's line reading the new counts. The spend line can
now say "3 judge calls went out and got no answer" instead of a count it
could not explain. The export gains the `judge_calls` rows for the
experiment (schema 9) so an export can be audited against a provider
bill.

**A scoring pass is a record.** A new table `scoring_passes`: id,
experiment_id, judge_model, started_at, ended_at, outcome (`finished`,
`stopped`, `failed`), error detail for `failed`, and counts written at
the end (scored, failed, unanswered). `judge_calls.pass_id` cites it. A
new door `GET /experiments/{id}/scoring` returns the passes for an
experiment, newest first, and the active one if any, in the same shape.
`GET /experiments/{id}` carries the latest pass's summary so the panel
needs no second request to say "scored on <date> by <judge>, 2 calls
unanswered" or "last scoring pass failed: <detail>". The page's Score
row shows that summary beside the button.

**A scoring pass stops.** `POST /experiments/{id}/scoring/stop` sets the
stop event; the pass checks it between trials and exits with outcome
`stopped`, its in-flight `judge_calls` rows getting outcome `stopped`.
The lifespan sets the same event and awaits the pass under a bound
(`SCORING_SHUTDOWN_SECONDS`, 30, named once), then cancels; a pass cut
by the bound writes `stopped` too, and its unanswered calls stay
recorded as sent, which is the whole point. The stop door returns 409
when no pass is active, in the server's sentence.

**A failed pass reaches a door.** The pass's `except` writes the
`scoring_passes` row with outcome `failed` and the detail, releases the
slot as today, and the read door shows it. A re-score after a failed
pass is a new pass; the old one stays in the list. The N review's L8
proof (next Score accepted after a raise) stays and gains the assertion
that the failed pass is in the list with its detail.

Migrations: two tables, additive, hash-pinned, one era fixture. The
scores table gains `judge_call_id` NULLable; old rows keep NULL and the
report treats a NULL as "before Phase P", never as unanswered. A test
composes a report over an era fixture from before P and asserts the
counts it shows are the ones the old rows support.

Browser: the Score row's summary line and the Stop control for a running
pass, with the same hidden-button proof pair as N4 (the control is
absent when no pass runs, and the request sent anyway is refused).
Progress for a pass is not built; the summary line refreshes with the
list.

Rulings requested at the checkpoint:

1. NULL-then-filled columns or a two-table sent/answer shape.
2. Whether judge spend counts against the spend ceiling. Today
   `record_spend` is called for trial results; say where judge calls
   stand (my reading: they are not counted, and the README's ceiling
   paragraph does not say so), and propose. My position: count them,
   because the ceiling is the operator's intent about money leaving the
   machine and a judge call is money leaving the machine; but it is a
   change to what the ceiling means and needs its own sentence in the
   README and a proof.
3. `SCORING_SHUTDOWN_SECONDS` at 30.

### P2. Atomic admission for the spend ceiling

The oldest entry. The README paragraph promises a bound of "at most
`MAX_CONCURRENT_UPSTREAM` calls already executing at the moment the
ceiling trips"; a reservation ledger makes the bound exact.

- Admission reserves the call's worst-case cost (the catalog's price at
  the budget cap for that member, the same figure `projected_cost`
  uses) against the ceiling, under one lock, before the semaphore:
  `accumulated + reserved + worst_case > limit` refuses with the 402
  naming all three figures. Settlement replaces the reservation with the
  billed figure or the estimate, as `ceiling_cost` decides today, under
  the same lock. A call the catalog cannot price reserves nothing and is
  recorded as unpriced, as today: the ceiling bounds known spend and
  says so.
- The eight-concurrent-batches regression test becomes the proof of the
  new bound: with a ceiling worth half a result, at most one call goes
  upstream (the one whose reservation fit), and the README's paragraph
  is rewritten to state that bound and drop the "deliberately deferred"
  sentence, tombstoned in the body against 095e293 and a300623.
- The ledger is process-local like the ceiling itself, resets on
  restart, and is reported on `GET /models` beside the existing spend
  figures (accumulated, reserved, limit) so the page's spend indicator
  can show money in flight. Blank is not sent: reserved is omitted when
  no ceiling is set.
- Judge calls reserve too, if ruling 2 says they count.

Proofs: the lock's window (two admissions racing the last dollar; only
one passes), the release on every exit path including a raise before
the request goes out (a mutant that drops the `finally` is killed), and
a settlement that lowers or raises the figure leaving `accumulated +
reserved` equal to what was actually recorded.

### P3. The ancestor look

Built as the O review designed it: anchored on the root's directory
handle, with race proofs, and with the open-state pin's four cases as
its pre-states.

- The `Tree` protocol gains a ninth operation, `parent(dirfd)`: open
  `..` relative to a directory descriptor, returning the parent's
  descriptor and its device and inode. No path is ever formed.
- The look starts from the root's descriptor, already open under
  containment, and climbs by `parent` until it reaches the allowlist
  entry's descriptor by device and inode (the entry's own identity is
  known from boot, as L3's fix records it) or the filesystem root. At
  each ancestor it checks the four-name signature through the existing
  `fstatat`-style operations on that descriptor. A match refuses with a
  new sentence in the reader's words: "the root is inside a git
  directory, whose files can carry a remote URL with sign-in details in
  them, so it is not walked." The listing's stop row carries the same
  sentence.
- The look runs after the root is opened and before the head-and-dirty
  read, at both doors, and it only refuses: a race against it can at
  worst give back today's walked state, and the proofs say so.
- Race proofs in the L shape: an ancestor replaced by a symlink to an
  outside git directory during the climb (the descriptor climb does not
  follow it and the look reports what the descriptor chain shows); an
  ancestor renamed mid-climb; the allowlist entry itself replaced. Each
  reproduced at its window with the fifo door where it applies.
- The open-state pin's four cases flip: each now refuses at both doors
  with the new sentence and the sentinel in no answer, and the pin is
  rewritten as this commit's pre-state (stash-proven against ed00174:
  201 with the sentinel before, refusal after). The `logs/` case, which
  git writes, is the one to lead with.
- Both posture walks learn `parent`.
- BACKLOG: the entry is removed with its whole history in the body, and
  the filter-drivers entry stays as it is.

Ceilings: the climb is bounded by `MAX_DEPTH` (128) from the root, and a
climb that reaches neither the entry nor the filesystem root within it
refuses naming the ceiling; a test with a 129-deep root proves it.

## Non-goals, named so they are not drifted into

- Private repositories and tokens, proxy support, deleting a clone,
  filter drivers: the O entries stay.
- Strict-mode fields in the browser, experiment rerun and reuse,
  dataset editing, charts, pairwise judging, `primary_judge`, Parquet.
- A progress stream for a scoring pass.
- Any change to what the ceiling counts beyond ruling 2.
- Any change to the composer's own ceilings or to the walk's exclusion
  groups.

## House law that binds this phase

- Tombstones stash-proven in the reviewer's shape; pre-state assertions
  on every new test; mutation proofs per door with call-site
  enumeration; numbers remeasured per commit.
- Records never rewrite, in commits, commit bodies and now in the
  `judge_calls` design; the two-write rule above is the test.
- Migrations: hash-pinned, additive, era fixture, and a report over a
  pre-P fixture that shows only what the old rows support.
- Declaration transport: `judge_call_id` and the pass summary travel to
  the report and the export (schema 9) exactly as `judge_generation_id`
  does today; a test compares a pre-P and a post-P experiment's exports
  and finds the difference exactly in the new fields.
- Gate lines for the unit job cite a run on the CI platform; the draft
  PR is opened at P1 so every commit's matrix is on record; green runs
  are cited in the next commit, never by amending.
- Hidden button is not a rule: the Stop control's pair.
- Vocabulary: "sign-in details" in anything a reader sees, "sentinel
  string" in fixtures, and the three retired words in no added line.
- ruff from the venv; the CI platform is the truth for the unit job;
  when the 3.14 leg crosses 240 s the job is split, not the limit raised.

## Proofs and gates

New proofs, at minimum:

- P1: a judge call that times out after sending leaves a `judge_calls`
  row with outcome `timed_out` and no answer columns, and the report
  counts it as unanswered (pre-state: no row, counted nowhere); the
  second write fills only NULLs (mutant overwriting `sent_at` killed);
  a pass stopped from the door and a pass stopped by the lifespan both
  write `stopped` with their unanswered calls recorded; a raising pass
  writes `failed` with its detail and the next Score is accepted; the
  pre-P era report shows the old counts; export schema 9 carries the
  rows; the browser summary line reads the pass and the Stop control's
  pair.
- P2: the racing-admission window; the bound of one call at half a
  result; release on every exit; settlement arithmetic; `GET /models`
  reserved figure present with a ceiling and absent without.
- P3: the four flipped pins plus a plain repository root that still
  composes (the look must not refuse a root whose ancestors hold a
  `.git` directory beside them, only one whose ancestor is itself a git
  directory: a normal checkout's `.git` is a sibling of its files, not
  their ancestor, and a test says so in those words); the three races;
  the depth ceiling; both walks; the listing's stop row.

## Commit and review shape

At least three commits, one per part, each green on the CI matrix,
with the P1 migration in its own commit ahead of the doors that use it.
Checkpoint after P1 with the three rulings answered. After P3, my pass
on ubuntu, then one external merge-readiness review with these lenses:
the `judge_calls` two-write rule and the migration; declaration
transport through export schema 9; the ledger under concurrency; the
ancestor look's containment and its races; records never rewrite.
Verdict standard as in O. Tag v0.7.0 at the merge.
