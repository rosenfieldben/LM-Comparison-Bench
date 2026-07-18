# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and compare the
results side by side in the browser. Responses stream in token by
token, and every result card carries latency, time to first token,
token counts and cost. Prompts can be saved as a reusable library
and every run lands in SQLite history for later replay.

## Daily use

```sh
cd LM-Comparison-Bench
source .venv/bin/activate
export OPENROUTER_API_KEY=sk-or-...
uvicorn bench.main:app
```

Then open http://localhost:8000. Manage the model lineup with the
built-in picker: "Add model" opens a search over OpenRouter's catalog
(by name or id), each lineup row has a remove control, and the lineup
persists in this browser's localStorage (clearing browser storage
resets it to the four defaults). The catalog is fetched once at boot,
same as pricing; restart the app to refresh it. On an offline boot
the picker falls back to adding models by exact id.

## Setup

Requires Python 3.11 or newer; CI runs the suite on 3.11 through 3.14.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/uvicorn bench.main:app
```

The app refuses to boot if `OPENROUTER_API_KEY` is unset.

Runs and saved prompts persist to an SQLite file. Set `BENCH_DB` to
choose its path; the default is `./bench.db` in the working
directory. Older bench.db files are upgraded in place at startup
(missing columns are added; existing rows are untouched and legacy
ungrouped runs keep rendering as before).

Cost per result is computed from OpenRouter's price list, fetched
once at startup. If that fetch fails (offline, outage), the bench
still boots and runs; cost just shows as unavailable for the
session. Every cost figure the bench displays is an estimate and is
marked with a tilde: catalog prices times the token counts providers
report, not billed cost, and the pre-run figure covers the worst-case
output side only (input is not estimated). The persisted generation
id is the hook for reconciling against OpenRouter's authoritative
per-generation numbers as future work. Results the session cannot
price (offline catalog, missing usage, errors after tokens flowed)
are counted next to the session total as "unpriced" rather than
silently dropped.

Set `BENCH_SPEND_LIMIT_USD` (a positive float; unset means no limit) to
cap estimated spend for the life of the process. An invalid value
(unparseable, non-finite, negative, or zero) fails boot with a message
naming the variable, rather than silently producing a ceiling that never
trips. Once accumulated
estimated spend reaches the ceiling, `/compare` and `/compare/stream`
refuse new runs with HTTP 402 and a message naming both figures,
checked at entry before any upstream call so a refusal costs nothing;
runs already in flight are never interrupted. Admission is rechecked
once more the instant a run acquires its upstream slot, so a run admitted
below the ceiling is still refused (before it spends) if a concurrent run
crossed the ceiling in the meantime; that refusal costs nothing and lands
in history as an honest cut-short row. Worst-case overshoot is therefore
bounded by the runs already executing when the ceiling trips, at most
`MAX_CONCURRENT_UPSTREAM` of them each completing at up to its budgeted
cost, not by the size of the lineup. A full reservation ledger (atomic
admission) is deliberately deferred. The ceiling tracks estimates, the
same catalog-price times reported-token figures the cards show, so
unpriced results (offline catalog, missing usage) do not count against
it. It resets when the process restarts.

The interface serves entirely from the bench: the fonts are vendored
under `static/fonts` (JetBrains Mono and Space Grotesk, both under the
SIL Open Font License in `static/fonts/OFL.txt`) rather than fetched
from a CDN, so the page makes no external request. The offline story
is complete: only the model calls reach the network, through
OpenRouter.

## Interface

The page is styled as a race-telemetry instrument (the "VOLT"
design): mono-heavy type, one cyan accent, state-coded color for
working/done/error, subtle live motion (pulse, shimmer, a blinking
placeholder cursor). The OS color scheme picks the dark or light
theme by default; a command-bar toggle cycles auto → dark → light,
and a motion toggle kills the animation (elapsed counters keep
updating as text). Both persist in this browser's localStorage, as
does `prefers-reduced-motion`, which disables animation regardless
of the toggle. All colors, spacing steps, radii and type sizes live
as CSS custom properties in one `:root` block at the top of
`static/volt.css`, so the next visual change is a token edit, not
a hunt through rules. The front end is three static files with no
build step: `static/index.html` (markup plus a pre-paint theme
script, the stylesheet link, and two script tags), `static/lib.js`
(the pure, DOM-free helpers, including the diff engine), and
`static/app.js` (everything that touches the document). All three are
served from the `/static` mount.

A full-width command bar carries the brand plus live session stats:
run count, estimated spend (with a count of unpriced results when any
run could not be priced), mean TTFT of completed requests, and
lineup size. They are this browser session's totals and reset on
reload.

Controls sit in one console deck above the results, three rows: the
prompt row (auto-growing monospace textarea in an inset field, plus
the saved-prompt library), the lineup row (model chips with per-chip
remove, All and None selection, and the Add model search), and the
run row (the Run button, a segmented token-budget control, and a
segmented column-density control, plus a request count and worst-case
cost estimate when pricing is available). Density has two steps,
comfortable and compact; compact tightens card padding and drops the
response font a step for racing many models side by side. Unlike the
budget (per session on purpose, it costs money), density persists.

During a live run a TTFT race strip sits between the deck and the
cards: one row per model, a shimmering meter until the first token,
then a bar sized by time-to-first-token on a shared round-number
scale, ranked by finishing order; errored rows show stripes and
"failed". The strip belongs to the live run only, so history replays
hide it.

Each result card shows its state twice, as a colored top edge and
a text label: thinking, done, or error. A running card counts
seconds in its header ("thinking · 47s", one shared timer for all
cards) so the long silences of extended-budget reasoning read as
alive rather than hung; the counter disappears at the first token.
At most five models call upstream at once (see Reliability); a model
still waiting for a slot reads "queued" rather than "thinking", and
flips to "thinking" with its timer reset the instant a slot frees, so
the queue wait never inflates its time to first token.
A four-cell metrics strip (ttft, total, tok i/o, cost) fills in as
values resolve, with em dashes for unknowns. Finished cards with
text gain per-card controls in a bottom action row: copy (raw
response text to the clipboard, with a brief "copied" confirmation),
fold (collapse to a six line preview, "show all" to reverse), and
diff; errored live cards add rerun. History renders as a flat strip
of rows (timestamp, prompt, model count) with a client-side filter
that matches prompt substrings and model ids, and loads only when
expanded.

## Token budgets

Every run carries one of two completion budgets, picked next to the
Run button: standard (16384 tokens) or extended (65536). Reasoning
models spend the budget on hidden thinking before any visible
answer, so a hard problem can empty the standard budget and come
back as "finish_reason: length"; extended exists for exactly that
case, and the error message says so when it applies. The choice is
per session and resets to standard on the next visit. The requested
budget is clamped per model to the completion cap OpenRouter
publishes, so asking for extended from a model capped at 32k sends
32000 instead of drawing a hard 400. An extended run can cost up to
four times as much as a standard one, because the budget is the
ceiling on billable completion tokens. History records the effective
post-clamp budget each run was sent with (shown as a "budget" badge
on replayed columns; older runs predate the field and show none),
and reruns reuse the budget of the run they retry. The API accepts
`"budget": "standard" | "extended"` on `/compare` and
`/compare/stream`; anything else is a 422.

## Reliability

The shared HTTP client enables TCP keepalive probes (SO_KEEPALIVE,
30s idle, 30s interval) so the minutes-long silent stretches of
extended-budget reasoning are not culled by NAT idle timers, which
had been surfacing as mixed ReadError and stall failures mid-lineup.

At most five paid upstream calls run at once across everything in
flight (`MAX_CONCURRENT_UPSTREAM` in `bench/main.py`); extra models
queue quietly for a slot, and the wait never counts toward a model's
measured latency or ttft. Every result also records OpenRouter's
generation id and the provider's finish_reason, which make historical
runs auditable against OpenRouter's generation API (actual provider,
quantization, authoritative cost) and let budget analysis see
truncation on runs that produced no error. If persisting a finished
run fails, both /compare and the streaming path log the failure and
return the results with run_id null, because the money is already
spent and losing history must not lose the response. The UI surfaces
that null as a small "not saved to history" warning on the affected
column, so silent history loss is visible where it happened. That is one
instance of a broader invariant both endpoints enforce with a single
fault boundary: after money is spent, no code path between the
upstream results existing and the response leaving (link resolution,
cost computation, persistence) may convert those results into an
error response.

## Local-only guard

The bench holds a paid API key, so it refuses requests that could
only come from a hostile browser page. Requests whose Host header is
not localhost, 127.0.0.1 or ::1 get a 403, which defeats DNS
rebinding; every POST must be `application/json` (415 otherwise),
bodyless ones included, which forces cross-origin senders into a
CORS preflight the bench never answers, so a malicious page cannot
fire "simple" text/plain or bodyless POSTs and spend money or create
state. GET and HEAD stay exempt as reads, and DELETE needs no gate
because a browser never sends it cross-site without a preflight.
Everything curl sends with a JSON content type and everything the
bundled frontend sends (it posts an empty JSON object to /groups)
passes unchanged. Every response also carries `X-Frame-Options: DENY`
and `Content-Security-Policy: frame-ancestors 'none'`, so the UI cannot
be embedded in a hostile frame and a Run click cannot be redressed into
paid work; the headers are added on response start with no body
buffering, so streaming is untouched. A fuller CSP is deferred: the
pre-paint inline theme script would need a hash or externalization. To
serve the bench beyond localhost deliberately, edit `TRUSTED_HOSTS`
in `bench/main.py`, and put real authentication in front of it
first.

## Local data

Everything the bench stores lives in one SQLite file: every prompt
you have run, every model response in full, and the timing, token,
cost and provenance numbers around them. The file is `bench.db` in
the working directory unless `BENCH_DB` says otherwise. It is
created private to your user (0600, in a 0700 directory if the bench
creates one), and a pre-existing file that is group or world readable
is tightened to 0600 at startup with a log line, because umask is
not a policy. Deleting the file deletes all history; there is no
other copy.

## Provider routing

Every request asks OpenRouter to sort providers by throughput
instead of its default price-weighted routing. Open-weight models
are served by many hosts, and the default routes them to the
cheapest, which in practice are the flakiest and often serve
quantized weights. Sorting by throughput biases each run to the
serious hosts at somewhat higher cost per run. The tradeoff is
deliberate: it changes who serves the model, never what the model
does, and because quantization varies by host it also stabilizes
what is actually being measured. The preference lives in
`PROVIDER_PREFS` in `bench/models.py`.

## Usage

Open http://localhost:8000 in a browser. Type a prompt, check the
models to compare, hit Run. Each column fills in as its model
responds. The lineup is managed with the picker (see Daily use); the
four-model default seed for a fresh browser is `DEFAULT_LINEUP` at
the top of `static/app.js`.

Or hit the API directly:

```sh
curl -X POST localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello in five words.",
       "models": ["deepseek/deepseek-chat",
                  "mistralai/mistral-small"]}'
```

Results come back in the same order as the requested models. A model
that errors or times out gets its `error` field set without affecting
the other models in the run. Every `/compare` call is persisted and
returns a `run_id`.

The browser UI streams responses token by token via
`POST /compare/stream` with `{"prompt": ..., "model": ...}` (one
model per request) plus optional `prompt_id`, `group_id` and
`budget`. The
response is SSE-formatted (`data: {...}` lines): `delta` events carry
text chunks, then one `done` event carries the full result and its
`run_id`. Streamed results include `ttft_ms` (time to first token),
which is the metric streaming exists to reveal; `latency_ms` alone
hides it. One deliberate contract amendment: a result can carry BOTH
partial `response_text` AND an `error` when a stream dies partway.
The partial text renders above the error, live and on replay. If the
browser disconnects mid-stream (tab closed, network drop), the server
still persists the partial run with a "stream aborted before
completion" error, so nothing the model already produced is lost from
history.

When a live column errors (timeout, provider failure, bad model id),
its header gains a rerun control. Clicking it reruns that one model
with the same prompt and the same group, streaming into the same
column through the normal path; other columns are untouched, and the
control is disabled while its rerun is in flight. The failed run
stays in history exactly as it happened: failures are data, and the
rerun persists as a second run in the same comparison group, so
History shows both. There are no automatic retries anywhere; a human
clicking is the boundary between recovering from a transient failure
and hiding one. Historical replays never show the control, since
rerunning history would be a new experiment wearing an old label.

Other endpoints:

- `GET /models` returns the boot-time catalog snapshot as
  `{"models": [...], "fetched": bool}`; `fetched` false means the
  boot fetch failed, which is how the picker tells an offline boot
  from an empty catalog
- `GET /prompts` lists saved prompts
- `POST /prompts` with `{"name": ..., "text": ...}` saves one; 409 on
  a duplicate name
- `DELETE /prompts/{id}` removes a prompt; runs that used it keep
  their text, only the link is cleared
- `POST /groups` creates a grouping id so one comparison's per-model
  requests land as a single history entry
- `GET /groups/{id}` returns a group's runs with full results
- `GET /runs` lists history, most recent first, prompt text truncated
  to 80 chars; entries are either `{type: "group", ...}` for grouped
  comparisons or `{type: "run", ...}` for legacy ungrouped rows.
  Returns the newest 100 entries by default; `?limit=` (1 to 500)
  adjusts the page size, and a group entry always carries all of its
  runs even when the page boundary falls inside it. The browser
  history panel notes when a full page was returned, since older
  entries then exist beyond it
- `GET /runs/{id}` returns a full run with results

## Diff view

Any two rendered result columns can be diffed: live against live,
historical against historical, or one of each (arm a live column,
open a History entry, then toggle a column there). Each column with
response text has a small "diff" toggle in its header; the first
toggle arms it, the second opens the diff panel below the results.
The diff is word-level (LCS, computed in the page, no libraries):
deletions from the left source render red, insertions from the right
render green, shared text flows plain. A column holding partial text
plus an error is diffable on its partial text and labeled
"(partial)". Responses beyond 4000 word tokens show a size notice
instead of freezing the tab.

## Tests

Test-only dependencies live in `requirements-dev.txt`, which pulls in
the runtime pins too. There are two suites:

```sh
.venv/bin/pip install -r requirements-dev.txt

# unit suite: the every-edit loop, fast and browser-free
.venv/bin/pytest

# browser suite: one-time setup, then the every-merge gate
.venv/bin/playwright install chromium
.venv/bin/pytest -m browser

# pure frontend helpers: no build step, no npm install
node --test "tests/js/**/*.test.js"
```

No network access needed for any of them; unit tests mock OpenRouter
with respx, the browser suite boots the real app under uvicorn in
headless Chromium against a stub OpenRouter it starts itself
(`tests/browser/`), and the node suite requires `static/lib.js`
directly through its CommonJS guard to check the diff engine and the
formatting helpers. Browser tests are deselected from a plain
`pytest` run on purpose. CI enforces all of it: a lint job (ruff and
mypy), the unit matrix across Python 3.11 through 3.14, the node
job, and the browser job, so neither a backend nor a frontend change
can merge without proving the critical path still works.

The stability contract for future frontend work: the harness selects
elements by `data-testid` attributes (and user-visible text), never
by styling classes or DOM structure. Keep the existing data-testid
attributes attached to the elements that play those roles and a
redesign can change anything visual without touching a test; remove
or rename one and the suite will tell you what behavior it guarded.

The browser suite covers the critical path only. During frontend
work, run `uvicorn bench.main:app --reload` so index.html edits are
picked up without restarts, and verify by eyeball after UI changes:

- Run with 2 models checked: both cards show the working state with
  a counting "thinking, Ns" indicator, then fill in independently,
  fastest first, flipping to the done label.
- Add an intentionally bad model id via the picker's exact-id path
  and run it: that card shows the error state (red top edge plus
  the error label), others unaffected.
- Run with a prompt that produces multi-line output (e.g. "write a
  haiku"): line breaks survive in the response column.
- Save a prompt, reload the page, pick it from the dropdown, replay
  it against one model. Open History, click the old run, and confirm
  it renders identically to a live run (plus the historical banner).
- Watch a slow model paint token by token next to an already
  finished fast one; the ttft metric should be visibly smaller than
  the total metric on streamed columns, and the race strip row
  should flip from shimmer to a ranked bar at the first token.
- Kill wifi (or the server) mid-stream: the streaming column must
  enter the error state with its partial text retained above the
  error message, not hang or go blank.
- Diff two live columns from similar prompts ("write a haiku about
  rain" on two models): common words flow plain, unique words tinted.
- Diff a live column against the same model's historical run of the
  same prompt: mostly plain text, sparse red and green.
- Diff a partial-error column: works, header says "(partial)".
- Paste-bomb: diff a very long response against a short one; the
  size notice appears and the tab does not freeze.
- Injection: prompt a model to output raw HTML tags, diff it, and
  confirm the tags render as literal text inside the tinted spans.
- Picker: search for a model by a name fragment, add it, run it,
  remove it, then reload the page and confirm the lineup survived.
- Boot with wifi off: the search row says the catalog is unavailable
  and the exact-id input still adds a model to the lineup.
- Rerun: force an error (bad model id via the exact-id path) and
  confirm only that column grows a rerun control, then rerun a real
  errored column: it resets to loading and streams the retry while
  the other columns sit untouched, and History shows both the
  failure and the successful rerun in one group. Columns replayed
  from History must never show the control.
- Budget: run a hard puzzle that empties the standard budget; the
  errored column's message ends with "try extended budget". Switch
  the control to extended and run again: the models now answer or
  prove they need even more, and each attempt's History replay shows
  the budget badge it actually ran with. Reload the page and confirm
  the control is back on standard.
- Density: switch to compact mid-run set; cards tighten and the
  response font drops a step; comfortable restores both. Reload and
  confirm compact is still selected: density persists, unlike the
  budget.
- Fold: fold a long answer to its six line preview, confirm the
  control now reads "show all" and clicking it restores the full
  text. Fold a partial-error card: the preview still holds.
- Copy: copy a column and paste elsewhere; the paste matches the raw
  response exactly, HTML tags included, and the button briefly reads
  "copied" before returning to "copy".
- Thinking counter: run a slow reasoning model next to a fast one;
  the slow card counts up in seconds until its first token, then the
  counter vanishes and never reappears.
- All / None: the two lineup buttons check and uncheck every chip,
  and Run enables or disables accordingly.
- History filter: type a model id fragment; rows without it in their
  prompt or models disappear; clearing the input restores them.
- Keyboard: Tab from the top of the page; every control (deck,
  chips, chip removes, card tools, history rows) is reachable and
  shows a visible focus ring.
- Theme: flip the OS color scheme; the page follows without a
  reload, and both themes keep the state labels readable.
- Spend ceiling: start the app with `BENCH_SPEND_LIMIT_USD` set to a
  tiny value, run until the session estimate crosses it, then run
  again: the columns error with the ceiling message spelled out in
  words, and no new upstream call is made.
- Queued state: run six or more models at once; the sixth card reads
  "queued" while five are in flight, then flips to "thinking" when a
  slot frees, and its counter restarts so its ttft excludes the wait.

## License

MIT. See [LICENSE](LICENSE).
