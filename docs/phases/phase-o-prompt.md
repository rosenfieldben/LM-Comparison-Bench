# Phase O: the member listing, and a clone door

Repo: github.com/rosenfieldben/LM-Comparison-Bench. Branch from main at
d1d792d (the Phase N merge, tagged v0.5.0). Branch name
`phase-o-clone-door`. If Dependabot merges land before you branch, branch
from the newest and reinstall the hashed venv before the first gate.
Commit this file under `docs/phases/phase-o-prompt.md` in the first
commit, verbatim, so BACKLOG citations to it resolve.

## What this phase is for

Phase L made a repository snapshot: one attachment composed from an
allowlisted clone already on disk, under the same ceilings a set of
attachments has, fetching nothing. Phase N made that attachment citable
from a dataset task, so a repository can already be put in front of a
lineup and judged. What a person still cannot do from the page is two
things: see what a walk would select before paying to compose it, and
get a repository onto the disk in the first place.

After this phase, a person types a public GitHub URL and a ref into the
snapshot panel, the bench clones it into a directory the operator named,
lists what a pattern set would select from it with sizes and refusals,
and composes the snapshot they chose. The composer still fetches nothing.
The fetching happens in a different door with its own posture, and the
bench's outbound posture changes from one named destination to two.

Two BACKLOG entries close here (GitHub: clone door, and member listing);
their removal is the tombstone.

## What already exists, so it is not rebuilt

- `bench/snapshot.py`: pure. `walk` over a `Tree` protocol (containment
  by descriptor, eight operations), `enforce_patterns`, `excluded`,
  `enforce_text`, `compose` taking `(path, bytes)` members plus `head` and
  `dirty`, ceilings `MAX_PATTERNS` 20, `MAX_WALKED_ENTRIES` 20000,
  `MAX_DEPTH` 128, `MAX_MEMBER_BYTES`, `MAX_READ_BYTES`. The secrets
  exclusion group is non-overridable.
- `POST /snapshots` in `main.py`: `SnapshotCreate` (root, patterns,
  excludes), roots resolved from `BENCH_REPO_ROOTS`, the SNAPSHOTS_OFF
  refusal naming the variable when unset, capture provenance recorded per
  capture in `snapshot_captures` (head sha, dirty flag, manifest), export
  schema 7.
- `_git(args, cwd)`: the one git runner, argv list, no shell, two-second
  timeout, degrades to None. Used by `_app_sha` and by the snapshot's
  head-and-dirty read. It is listed in the posture walk's
  `FILESYSTEM_CALLS` for its working directory.
- The posture walk in `tests/test_api.py`: an AST test mapping every
  filesystem call to a posture, where a new call fails. There is no
  equivalent for the network; this phase adds one.
- The single-outbound-destination posture: `app.state.client` is one
  `httpx.AsyncClient` to OpenRouter and nothing else leaves the machine.
- The browser: `static/attach.js` snapshot panel (root, patterns,
  Compose, SNAPSHOTS_OFF verbatim), chips, the attachment picker in the
  dataset builder that already cites snapshots.

Baseline at d1d792d on ubuntu: 1182 unit collected (1181 pass, 1
measured-depth skip), 283 browser, 64 node, ruff 0.16.5 from the venv,
mypy, biome 1.9.4, pip-audit, `pytest -rs` in CI.

## Scope, in three parts

Checkpoint after O2, before any browser code touches the clone door,
because O2 is where the outbound posture changes and the rulings below
get exercised against a real subprocess.

### O1. The member listing

`POST /snapshots/listing`, same body as `SnapshotCreate`, same roots,
same containment, same patterns, same exclusion groups, same walk. It
returns what `POST /snapshots` would have selected and refused, and
composes nothing, stores nothing, reads no file contents.

Response: `{members: [{path, bytes, kind, status, reason}], counted,
selected_bytes, would_compose, refusal}` where `status` is `selected`,
`excluded` (with the exclusion group named) or `refused` (with the
composer's own sentence: too large, wrong kind, past the entry ceiling),
`would_compose` says whether `POST /snapshots` on this body would reach
composition on the facts the tree shows, and `refusal` carries the
sentence the composer would raise first when it would not.

Rulings:

- **The listing reads the tree and not the files.** Sizes and kinds come
  from the walk. The encoding and byte-order-mark checks in
  `enforce_text` need the bytes, so they stay compose-time refusals, and
  the listing says so in its response (`text_checked: false`) and in the
  README, so nobody reads `would_compose: true` as a promise the
  composer cannot break. The alternative, reading every selected member
  to pre-check encoding, doubles the read cost of every snapshot and
  makes the listing as expensive as the thing it previews. Ruling
  requested at checkpoint only if you find the encoding refusal is
  common enough in practice to make the cheap listing useless.
- **One walk, two doors.** The listing and the composer must agree by
  construction, not by testing alone: both call the same `walk` with the
  same arguments, and the selection the listing reports is the manifest
  the composer would record. The proof is a property test over the
  snapshot fixtures: for every fixture and pattern set, the listing's
  `selected` paths equal the composed manifest's members, and
  `would_compose` is true exactly when compose succeeds on a tree with no
  encoding refusals.
- **Bounded the same way.** `MAX_WALKED_ENTRIES` applies; past it the
  listing reports the ceiling and stops, as the composer does. It is
  synchronous on the loop, bounded rather than offloaded, for the reason
  `create_snapshot` gives.
- **No record.** A listing is a look, not a capture. Nothing is written.

Browser: the snapshot panel gains a List step before Compose. The
member table shows path, size, status and reason, with the composer's
sentences verbatim. Selecting rows narrows the patterns: a selected file
becomes a pattern equal to its path, and past `MAX_PATTERNS` the panel
says so with the constant's value rather than silently truncating.
Compose stays the same door with the same body; the listing only helps
write it.

### O2. The clone door (server)

`POST /clones` with `{url, ref}`. The server clones a public repository
into a directory it owns and returns a root the snapshot doors accept.

The posture, in order of what refuses first:

1. **Off unless named.** A new variable `BENCH_CLONE_ROOT`, one absolute
   directory. Unset, the door refuses naming the variable, exactly as
   SNAPSHOTS_OFF does, and the panel shows CLONES_OFF verbatim. Set, it
   must be one of the `BENCH_REPO_ROOTS` entries or the door refuses
   naming both variables: the allowlist stays the single authority on
   what the snapshot doors may walk, and the clone door does not get to
   extend it by implication.
2. **Hosts by allowlist.** `BENCH_CLONE_HOSTS`, default `github.com`.
   `https` only. The URL is parsed, not matched: scheme, host against the
   list, no userinfo (a URL carrying `user:token@` refuses naming the
   rule and never echoes the URL), no query, no fragment, a path of the
   form `/owner/repo` or `/owner/repo.git`. Everything else refuses with
   a sentence that names the rule and not the URL.
3. **Ref by grammar.** A branch or tag name under git's check-ref-format
   rules, or a 40-hex commit. It must not begin with `-`. This is the
   argument-injection line: every argv element derived from the request
   is validated to not start with `-`, and the clone invocation puts `--`
   before the URL regardless.
4. **Public only.** No token, no credential helper, no ambient identity.
   The subprocess runs with a scrubbed environment: `PATH`, a `HOME`
   pointing at an empty temporary directory, `GIT_TERMINAL_PROMPT=0`,
   `GIT_CONFIG_NOSYSTEM=1`, and `-c credential.helper=` on the command
   line, so a private repository fails to clone rather than succeeding
   through a helper the operator forgot was configured. The proof is a
   test that plants a credential helper in the test environment and
   asserts the clone of a stub "private" remote refuses without invoking
   it.
5. **Bounded.** `git clone --depth 1 --single-branch --branch <ref>
   --no-tags -c protocol.allow=never -c protocol.https.allow=always
   -- <url> <dir>`, no submodules. A timeout (`MAX_CLONE_SECONDS`, 120,
   named once) kills the process group and removes the partial
   directory. After the clone, a walk counts bytes and entries under the
   snapshot ceilings; a checkout past `MAX_CLONE_BYTES` (200 MB,
   proposed; ruling requested) is removed and refused with the measured
   figure. The refusal names what was measured, never a guess.
6. **One at a time.** A single clone slot, like the scoring slot, with
   the N lesson applied from the start: the slot is released in a
   `finally` that covers the first line of the work, with a test that
   the next clone is accepted after one raised.
7. **Async subprocess.** Unlike the composer, a clone waits on the
   network, so it runs through `asyncio.create_subprocess_exec` and the
   loop is not held. `_git` stays as it is for its two-second local
   questions; the clone gets its own runner, `_git_clone`, listed in
   the posture walk with the clone directory as its posture.

Placement and identity:

- The clone lands at `BENCH_CLONE_ROOT/<sha256 of "url\nref">[:16]/`.
  The directory name is a function of the request, so the same
  repository at the same ref always has one place, and a person can find
  it from the record.
- A second request for the same URL and ref fetches and resets to the
  ref rather than cloning again, and returns the new head. Records never
  rewrite: a clone directory is a working tree, not a record, and every
  snapshot already composed from it keeps its digest and its capture
  provenance. The response says which happened (`cloned` or `updated`).
- A `clones` table: id, url, ref, head_sha, root, created_at,
  updated_at. Additive migration in the hash-pinned series, era fixture.
  The URL is recorded here and nowhere else: it never enters a composed
  text, a manifest or an export, because a URL is not a fact about the
  reading.
- Response: `{id, url, ref, head_sha, root, outcome}`. `root` is what
  the person pastes into, or the panel fills into, the snapshot root.

Ruling requested at checkpoint: whether a snapshot composed on a
clone-door root should carry the clone id in its capture record (an
additive column on `snapshot_captures`), so an export can be traced to
a URL through the clones table. My position is yes, because "which
repository was this" is the question a reader of an export asks first,
and the head sha alone does not answer it. It is a small column and it
belongs in O2 if you agree, since O3 would otherwise ship a panel whose
records cannot say where a root came from.

The outbound posture:

- The README's single-outbound-destination statement is restated as
  two named destinations, OpenRouter and the clone hosts the operator
  listed, and the sentence that stays true is: nothing leaves the
  machine to a host the operator did not name.
- **A network posture walk.** A permanent AST test, sibling to the
  filesystem walk, that maps every call which can reach the network
  (`httpx` client construction and calls, `subprocess` and
  `asyncio.create_subprocess_exec` argv that name `git` with a fetching
  verb, `socket`, `urllib`) to a posture, and fails when a new one
  appears. This is the tripwire that makes "fetches nothing" a checked
  claim for every door but the clone door. It lands in O2 with the door,
  not after.
- `create_snapshot`'s docstring keeps "IT FETCHES NOTHING" and gains one
  sentence saying where fetching lives now.

### O3. The clone door in the browser

The snapshot panel gains a Clone step above List: URL and ref, a Clone
button, the CLONES_OFF sentence verbatim when the door is off, and on
success the root box filled with the returned root and the outcome and
head sha shown beside it. Every refusal from the door is shown verbatim,
and none of them contain the URL the person typed, which the browser
test asserts by typing a URL with userinfo and reading the sentence.

Nothing is remembered across loads. Blank is not sent. A clone is not a
comparison and records nothing about the lineup.

The dataset builder's attachment picker already shows snapshots; O3 adds
the head sha to the snapshot chip's title where a capture record has
one, so a person choosing between two snapshots of the same repository
can tell them apart.

## Non-goals, named so they are not drifted into

- Private repositories and tokens. The secrets posture is that no
  credential exists on this path. Backlog entry with that reason.
- The GitHub API, webhooks, auto-refresh, GitHub App installation.
- Submodules, LFS, shallow-to-full unshallowing, branches other than
  the one asked for.
- Deleting a clone from the page. The operator removes directories;
  backlog entry (disk hygiene) with the reason that a delete door on
  working trees needs the same containment proofs the walk has.
- Any change to the composer's ceilings. A repository that does not fit
  is selected from, and the listing exists so that selection is made
  from fact. L's ruling stands.
- Any change to `_git`.

## House law that binds this phase

- Tombstones stash-proven in the reviewer's shape, pre-state assertions
  on every new test, mutation proofs per door with call-site
  enumeration, numbers remeasured per commit.
- Gate lines for the unit job cite a run on the CI platform. Open the
  draft PR early, at O1, so every commit's matrix is on record; the
  green run is cited in the next commit, never by amending.
- Declaration transport: the clone id, if ratified, must travel to the
  capture record, the report and the export exactly as head sha does,
  with a test comparing a snapshot composed on a clone root against one
  composed on a plain root, equal in every field but that one.
- Records never rewrite, in commits and commit messages too. Corrections
  go in later bodies.
- Nothing branches on a model name; nothing branches on a host name
  beyond the allowlist check.
- Comment-code agreement: the posture walk's docstring, `create_snapshot`'s
  docstring and the README's outbound statement all change in the same
  commit as the door.
- BACKLOG: the two GitHub entries removed as tombstones in the commits
  that build them; the two new entries (private repositories, clone
  deletion) added with reasons in the commit that names them as
  non-goals, which is the O2 commit.
- ruff from the venv binary. The CI platform is the truth for the unit
  job. Every gate line names the platform it ran on.

## Proofs and gates

New unit proofs, at minimum:

- O1: listing equals manifest over every snapshot fixture; the entry
  ceiling reported the same way both doors report it; no write and no
  file read (the posture walk names the listing's calls, and a fake Tree
  asserts `open` is never called).
- O2, refusals in order: CLONES_OFF; clone root not in the roots;
  scheme, host, userinfo, query, path shape; ref grammar and the leading
  dash; the planted credential helper never invoked; timeout removes
  the partial directory; size ceiling removes and names the figure; the
  slot released after a raise; same URL and ref twice gives `updated`
  and a new head with old snapshot rows untouched; the URL absent from
  every composed text, manifest and export; the network walk fails on a
  planted `httpx.get` and on a planted `git fetch`.
- O2, the argument-injection proof: a ref of `--upload-pack=x` and a
  URL path of `-x` both refuse at validation, and a mutant that drops
  the `--` separator is killed by a test that would otherwise reach git
  with an argument git would parse as an option.
- O3, browser: the full path (clone a stub remote, list, narrow by
  selection, compose, cite the snapshot from a dataset task, create and
  start an experiment over it); CLONES_OFF verbatim; a userinfo URL's
  refusal sentence contains no part of the URL; List past
  `MAX_PATTERNS` names the constant.

The stub remote for tests is a bare repository on the local filesystem
served over `file://`? No: `protocol.allow=never` with `https` allowed
means a `file://` remote refuses by design, and that refusal is itself a
test. The clone tests use a local HTTPS stub (an `http.server` behind
the test's own self-signed certificate, with `GIT_SSL_CAINFO` pointed
at it in the scrubbed environment) serving a bare repository over the
dumb HTTP protocol. If that proves heavier than a phase should carry,
say so at the checkpoint and propose the alternative; do not weaken
`protocol.allow` to make the test easier.

## Commit and review shape

At least three commits, one per part, each green on the CI matrix.
Checkpoint after O2 with the rulings answered. After O3, my pass on
ubuntu, then one external merge-readiness review with the clone door's
posture (hosts, userinfo, argv, scrubbed environment, timeout, size,
slot) and the network walk as its first two lenses, then declaration
transport for the clone id, then the listing-equals-manifest property.
Verdict standard as in N. Tag v0.6.0 at the merge.

## Rulings requested at the checkpoint

1. `MAX_CLONE_BYTES` (200 MB proposed) and `MAX_CLONE_SECONDS` (120
   proposed).
2. The clone id on the capture record (my position: yes, in O2).
3. The listing reading the tree only (my position: yes), unless the
   encoding refusal proves common in practice.
4. The HTTPS test stub, if it proves too heavy.
