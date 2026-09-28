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
- **An entry a later commit makes false is corrected in that commit**, as
  far as the new fact requires and no further, and the correction is
  named in that commit's body. An entry in the operator's words keeps
  them everywhere the fact did not move.
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
  was declared. Since f6d3f92, `ranking.reason` says so ("ordered on the
  mean"). Decide whether `primary_metric` should be able to name the pass
  rate, and have `ranking.reason` state which figure the ordering used.
- **Deferred by:** Phase N, in the N3 checkpoint exchange.
- **Reason:** it is a declaration question, not a defect, and it belongs
  with `primary_judge` when that is designed.
- **First written:** no commit to cite. Noticed at the N2 checkpoint; the
  reason was written by the commissioner in the N3 checkpoint exchange.

## Refusing a judge route with mandatory reasoning

- **What:** the Score door refusing a `judge_model` whose route has
  mandatory reasoning, the rule the README states under Pinned
  observations (2026-08-13, derived from the contract's arithmetic, not
  measured). Nothing enforces it today: the door accepts any judge id,
  and the page's judge select is the whole catalog, unfiltered, with a
  note that says so.
- **Deferred by:** Phase N (N4), ruled at the operator's N4 pass.
- **Reason:** Enforcing the rule at the Score door needs the catalog to
  carry the reasoning field it currently drops, which is a change to the
  catalog contract and not to the Score door, so it belongs with
  whichever phase next touches the catalog. The commission's claim that
  the rule is 'enforced where it lives' was false; N4 tombstoned it.
- **First written:** the question in the N4 commit body, 7c19c2b
  (restatement 5); the reason in the operator's words at the N4 pass.

## Private repositories and tokens

- **What:** cloning a repository that needs a credential: a token, a
  credential helper, an SSH key or any other identity.
- **Deferred by:** Phase O, O2 (the commission's non-goals).
- **Reason:** the clone door's secrets posture is that no credential
  exists on its path; a private repository is refused rather than cloned
  through a helper the operator forgot was configured.
- **First written:** docs/phases/phase-o-prompt.md, "Non-goals", "Private
  repositories and tokens. The secrets posture is that no credential
  exists on this path."

## Deleting a clone from the page

- **What:** a door that removes a clone's directory, where today the
  operator removes it by hand (its clones row stays, and a later clone of
  the same repository and ref reuses it).
- **Deferred by:** Phase O, O2 (the commission's non-goals).
- **Reason:** a delete door on working trees needs the same containment
  proofs the walk has; and it would remove trees, never rows, since a
  snapshot's capture may cite the row.
- **First written:** docs/phases/phase-o-prompt.md, "Non-goals",
  "Deleting a clone from the page. The operator removes directories;
  backlog entry (disk hygiene) with the reason that a delete door on
  working trees needs the same containment proofs the walk has." The
  clause on rows ("it would remove trees, never rows, since a snapshot's
  capture may cite the row") is the builder's, added in d997fd7, not the
  commission's; it became a fact in 44bb6b2, when
  `snapshot_captures.clone_id` began referencing `clones(id)`.

## Proxy support for the clone door

- **What:** cloning through an HTTP(S) proxy, where today the clone
  door's git runs with no proxy and the system's certificates only, so
  a bench behind a mandatory proxy cannot clone.
- **Deferred by:** Phase O, at the O2 checkpoint.
- **Reason:** honoring proxy variables widens the scrubbed environment
  that the no-credential proof depends on, so proxy support is a change
  to that proof, not a setting.
- **First written:** the operator's ruling at the O2 checkpoint, in
  those words; recorded in the commit that adds this entry.

## A snapshot root inside a git directory

- **What:** refusing a snapshot root that sits below a git directory's
  top level (a directory holding `HEAD`, `config`, `objects` and `refs`
  together, whatever it is named). Today the walk refuses a root, or a
  directory it reaches, that is one itself, and never looks above the
  root. Refusing a root inside one needs a look at the root's ancestors,
  by path (as `refuse_while_cloning` already looks at both doors, and
  `_clone_for` at the composer) or anchored on the root's directory
  handle; a look that only refuses opens nothing, so a race against it
  can at worst give back today's walked state, and if it is built it
  gets the walk's race proofs. A submodule's `modules/<name>` holds the
  four names itself (as git 2.50.1 lays it out), so the walk already
  refuses one wherever it reaches it.
- **Deferred by:** Phase O, ruled by the operator on the bare-repository
  readings after the operator's pass at 9c920e5.
- **Reason:** `config` at the top level is where git itself writes a
  remote URL (`git remote add`, measured on git 2.43.0), and a root
  below the top cannot reach it; the known exception, `modules/*/config`
  in a repository with submodules, the walk already refuses by its own
  signature. Three files below the top level can carry a URL git uses,
  verified on git 2.43.0 by tests/test_api.py
  (`test_below_a_git_directorys_top_level_the_walk_cannot_see_it`):
  the legacy `remotes/<name>` and `branches/<name>` files, which
  `git fetch` still resolves, and a linked worktree's
  `worktrees/<name>/config.worktree`, which `git config --worktree`
  writes under `extensions.worktreeConfig`. A root placed at any of
  those composes with the URL in the text today; the test pins that
  open state. On git 2.50.1 (measured): `git worktree add` copies the
  main worktree's `config.worktree` into `worktrees/<id>/config.worktree`,
  a second writer beside `git config --worktree`; `git init` makes no
  `branches/`; and the legacy `remotes/` and `branches/` files are
  still read, with the warning that they are "nominated for removal".
  And `logs/` is a fourth place, written by git itself on ordinary
  commands: `git pull` and `git fetch` record their arguments there, a
  repository's path verbatim (measured on git 2.50.1 with local paths;
  whether a URL's userinfo survives is what the open-state pin's `logs/`
  case measures), beside the committer's identity on every ref update.
  The list of such files has no fixed end. The Phase O external review,
  at db08aca (CI run 36338955465), answered that this stays open for
  v0.6.0 and that the ancestor look is next-phase work: cheaper than
  recorded, because a look that only refuses opens nothing (its L6), and
  needed because the reachable set has no end, since `git pull` writes
  into `logs/` below the top (its L14). It is to be built anchored on
  the root's directory handle, with race proofs, and with 7b7241c's pins
  plus a `logs/` case as its pre-states.
- **First written:** the operator's ruling on the bare-repository
  readings after the pass at 9c920e5; the reason in the operator's words
  as given for f8bde6b ("only config at the top level carries anything
  sensitive and a root below the top cannot reach it"); the reason
  restated in 7b7241c on git 2.43.0's measurements (the three files
  below the top level), with its hand-off to the external review; the
  review's answer and the operator's ruling on it recorded in the
  commit that adds this sentence.

## Filter drivers in the snapshot's dirty read

- **What:** a filter driver named in the configuration of a checkout the
  operator listed, which `git status` runs on a file whose stat data
  differs from the index when a snapshot reads the dirty flag. The local
  git's fixed configuration (94e70ee) turns off the file system monitor,
  hooks and the sign-in helper; it does not turn off filter drivers.
- **Deferred by:** Phase O, ruled by the operator on 94e70ee's open
  items.
- **Reason:** with the ceiling at the entry's parent and
  `safe.bareRepository=explicit`, discovery can no longer land on a
  repository-supplied directory, so a filter driver can only come from a
  config the operator's own git wrote into a checkout the operator
  listed. That is the operator's machine, not repository-supplied
  configuration, and it is outside this phase's threat model. No
  command-line setting disables every driver, so closing it means not
  running `git status` for the dirty read, which is a design change for
  its own phase.
- **First written:** the operator's ruling on 94e70ee's open items, in
  those words; recorded in the commit that adds this entry.

## The page shows money in flight

- **What:** the page's spend indicator reads the ceiling's figures from
  `GET /models` (`spend`: `accumulated_usd`, and `reserved_usd` and
  `limit_usd` under a ceiling) and shows what calls not yet settled have
  claimed beside what has been recorded.
- **Deferred by:** Phase P, P2.
- **Reason:** the page fetches `/models` once, at boot, and its spend
  figure is its own sum of the composer's runs, so showing a live,
  process-wide figure needs a refresh the page does not make, and that
  page work was not sized in P2, which built the figures it would read.
- **First written:** the commission's P2 ("so the page's spend indicator
  can show money in flight"); deferred in the commit that adds this
  entry.

## A rerun on a card refused for room

- **What:** the rerun control on a streamed card refused because the
  claims of calls not yet settled left no room for its worst case, a
  refusal that can clear as those calls settle.
- **Deferred by:** Phase P, P2.
- **Reason:** the card cannot tell that refusal from one because the
  ceiling was reached, since both arrive as `spend_refused`, and a rerun
  of the second can only be refused again; telling them apart is a field
  on the frame and a page change P2 did not make.
- **First written:** the comment on the rerun control in
  `static/stream.js`, in the commit that adds this entry.

## The catalog a scoring pass priced against

- **What:** a scoring pass records the digest of the catalog whose rates
  settled its unbilled judge calls, so that the estimate each such call
  counted can be derived again from the record alone.
- **Deferred by:** Phase P, P2.
- **Reason:** the operator's ruling at P1's checkpoint asked for the
  counts, which the judge call's record keeps (2e8bd02); the rates are
  the booted process's catalog, which the pass does not name, and naming
  it is a third column and a seal change the ruling did not ask for.
- **First written:** the commit that adds this entry.
