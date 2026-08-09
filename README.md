# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and compare the
results side by side in the browser. Responses stream in token by
token, and every result card carries latency, time to first token,
token counts and cost. Prompts can be saved as a reusable library
and every run lands in SQLite history for later replay.

This is a comparison workbench: it puts models side by side under
identical conditions and records what happened, with enough provenance
to say later what an old run actually was. The evaluation layer on top
of it is here: datasets addressed by content, repeated randomized
trials with rotated entry order, deterministic and rubric scoring,
blind human rating, aggregates with their denominators and clustered
intervals, and an export that seals a moment and can rebuild every
number it publishes.

What it still is not is a leaderboard. The statistical protocol a
published comparison would need (paired estimates, multiplicity, power)
is not here, and the report withholds a ranking rather than inventing
one whenever nobody has said what "better" means. Read an experiment as
evidence, with its intervals and its coverage counts, rather than as a
result to quote out of context.

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

OpenRouter attaches a `usage` object to every response reporting what it
actually charged, and that billed figure is the number a card and the
session total show, without a tilde. The bench still degrades gracefully
if a provider disagrees, since "always" is the platform's promise rather
than something this code can enforce. Beside it the bench keeps its own
estimate, computed from OpenRouter's price list fetched once at startup:
catalog prices times the token counts providers report. The estimate is
always marked with a tilde and stays reachable in the cost cell's
tooltip, because a gap between the two says something about the provider
that served the run. When no billed figure arrives, the estimate is the
whole display, exactly as before. If the price fetch fails (offline,
outage) the bench still boots and runs; only the estimate is
unavailable for the session, and the pre-run figure (which is an
estimate by nature, since nothing has been charged yet) covers the
worst-case output side only, input not included.

Results the session can price by neither route (no usage object, an
offline catalog, errors after tokens flowed) are counted next to the
session total as "unpriced" rather than silently dropped, and so is any
run that reached a provider without producing a charge: stopped by hand,
superseded by a new comparison or a history load, or failed after the
call went out. Their billing depends on whether the provider supports
cancellation, and routing picks the provider per request, so it is not
knowable locally. A run that never got that far is not counted: cancelled
while still queued, or refused by the spend ceiling, it verifiably cost
nothing. A poisoned
money value is treated as no value: a cost that is not a finite,
non-negative number degrades to nothing, so it cannot subtract from a
total or turn every ceiling comparison false.

Each card also reports the hidden reasoning tokens the response burned
(billed as completion tokens, counted against the budget, invisible in
the answer) and the provider that served it. Routing is by throughput,
so the provider is chosen per request and can differ between two runs of
the same model; naming it per run makes the largest confound in a
comparison visible rather than assumed away.

**The ceiling meters CREDITS.** OpenRouter's `usage.cost` is a charge in
credits against the balance this process can spend, and that is the number
the ceiling accumulates and refuses on. Under BYOK (bring your own key)
there is a second number, `usage.cost_details.upstream_inference_cost`:
what the upstream provider billed you directly. The bench records it in
its own column beside the credit charge and **never meters it**, because a
direct provider bill is not money OpenRouter can decline and a ceiling
that pretended otherwise would be describing a control it does not have.
Off BYOK the field is absent or zero and the column is NULL, which is the
honest record of "this run was not BYOK"; a stored zero would be
indistinguishable from a BYOK run that genuinely cost nothing.

It travels the whole way: served on the result, written by the reconcile
pass from the generation record, carried on the export's trial lines, and
shown in the report's cost section as its own figure. **Never summed into
any total**, and stored, served and exported as the string it arrived as,
because nothing computes with it. Its use is matching a provider's own
invoice line, and a float would reformat the number being matched.

The reconcile pass writes it on every row it reaches, and its work list
does **not** select on it being absent. NULL is the affirmative "not
BYOK", and there is no column for "asked, and the answer was no", so a
clause on it would park every ordinary row on the list forever and the
pass would never converge. The narrow case that leaves unreached is a
BYOK run whose usage object arrived with `cost` and without
`cost_details`, leaving the row otherwise complete; that gap is real and
smaller than a work list that never empties.

Set `BENCH_SPEND_LIMIT_USD` (a positive float; unset means no limit) to
cap recorded spend for the life of the process. An invalid value
(unparseable, non-finite, negative, or zero) fails boot with a message
naming the variable, rather than silently producing a ceiling that never
trips. Once accumulated
spend reaches the ceiling, `/compare` and `/compare/stream`
refuse new runs with HTTP 402 and a message naming both figures,
checked at entry before any upstream call so a refusal costs nothing;
runs already in flight are never interrupted. Admission is rechecked
once more the instant a run acquires its upstream slot, so a run admitted
below the ceiling is still refused (before it spends) if a concurrent run
crossed the ceiling in the meantime; that refusal costs nothing and lands
in history as an honest cut-short row.

Each result is settled against the ceiling inside the slot it holds,
before that slot is released. That ordering is what makes the bound below
true rather than merely intended: a freed slot implies a recorded
settlement, so once spend crosses the ceiling every later acquisition sees
it and refuses. Worst-case overshoot is therefore bounded by the runs
already executing at the moment the ceiling trips, at most
`MAX_CONCURRENT_UPSTREAM` of them each completing at up to its budgeted
cost, whatever the size of the lineup and however many comparisons are in
flight at once.

That last clause is the correction. Settlement used to run after a batch's
whole fan-out completed, so a fast member released its slot having recorded
nothing and a model from a concurrent batch rechecked against a counter
that had not moved. The bound held for one comparison at a time and failed
for several: eight concurrent five-model batches against a ceiling worth
half a result put 23 calls upstream where this paragraph promised five. The
documentation and the mechanism now state the same fact, and a regression
test measures it. A full reservation ledger (atomic admission) is
deliberately deferred. The ceiling counts each result
once, using its billed cost when the platform reported one and the
catalog estimate otherwise: it is advisory, and advising from real
charges beats advising from catalog arithmetic. Results that are
unpriced by both routes do not count against it. It resets when the
process restarts.

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
updating as text). Both toggles persist in this browser's
localStorage. The OS setting `prefers-reduced-motion` disables
animation regardless of the motion toggle; it is read from the
system on every page, not stored here, and nothing the bench writes
locally can turn it back on. All colors, spacing steps, radii and type sizes live
as CSS custom properties in one `:root` block at the top of
`static/volt.css`, so the next visual change is a token edit, not
a hunt through rules. The front end has no build step. `static/index.html`
holds the markup, a pre-paint theme script, the stylesheet link, and the
classic script tags in dependency order (classic, not modules: the load
order is the dependency graph, so the order of those tags is
load-bearing). `static/lib.js` holds the pure,
DOM-free helpers, including the diff engine. The DOM logic is split by
concern into small classic scripts, each assigning one `window.Bench*`
namespace: `state`, `controls`, `render`, `diff`, `library`, `stream`,
and `history`, with `boot.js` wiring them last. Cross-file access goes
through the namespaces, so a load-order mistake fails loudly rather than
silently at click time; the script order in index.html is load-bearing.
All are served from the `/static` mount.

Every response that does not choose its own caching carries
`Cache-Control: no-cache`, which means revalidate before reuse rather than
do not store, so with the ETags the static mount already sets the usual
cost of a reload is a 304. The one response that chooses is the favicon,
which declares a year of immutable caching on purpose. On top of that, the
index served at `/` is rewritten at startup to append `?v=<rev>` to every
asset URL, where the rev is a short hash of the static files' bytes. The
committed `index.html` keeps plain URLs, so opening it straight off disk
still works.

Both exist for the same reason. Assets used to go out with validators but
no `Cache-Control`, which let a browser apply heuristic freshness and
reuse a cached file without asking, and after an upgrade a page could run
fresh modules beside a stale one from the previous release. **Upgrading
never requires a hard reload**, and if you have ever been told to clear
the cache for this app, that advice is obsolete.

A full-width command bar carries the brand plus live session stats:
run count, spend (tilde-marked while any contribution to it was an
estimate, with a count of unpriced results when any run could not be
priced by either route), mean TTFT of completed requests, and
lineup size. They are this browser session's totals and reset on
reload.

Controls sit in one console deck above the results, three rows: the
prompt row (auto-growing monospace textarea in an inset field, plus
the saved-prompt library), the lineup row (model chips with per-chip
remove, All and None selection, and the Add model search), and the
run row (the Run button, a Stop button, a segmented token-budget
control, and a segmented column-density control, plus a request count
and worst-case cost estimate when pricing is available). Density has
two steps, comfortable and compact; compact tightens card padding and
drops the response font a step for racing many models side by side.
Unlike the budget (per session on purpose, it costs money), density
persists.

Stop is live only while runs are in flight. It aborts every request in
the current comparison without clearing it: each card keeps whatever
already streamed and switches to a muted "stopped" state (no error
styling, no invented metrics), a card still waiting for a slot shows
stopped with no text, and its race row stops ticking. Stopping does not
refund anything: spend already incurred stands, and each aborted request
disconnects its stream, so the server persists a started run as an
aborted record (visible on the next history load, since the client never
receives a run id) and a still-queued run not at all. A stopped run that
had started joins the session bar's unpriced count rather than being
counted as free: stopping the stream stops the charge only on providers
that support cancellation, and routing picks the provider per request, so
"billing unknown" is the honest reading. The same applies to a run
superseded by a new comparison or a history load, which reaches the
provider in exactly the same way. A run stopped while queued stays out of
that count, because it verifiably spent nothing. After a Stop the
in-flight count reaches zero, Run re-enables, and a fresh Run or a rerun
works as usual.

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
of rows (timestamp, prompt, and a model count that counts distinct
models, noting attempts separately when a rerun pushed the total above
them, as in "1 model · 2 attempts") with a client-side filter that
matches prompt substrings and model ids, and loads only when expanded.
The list says which state it is in while it does that, since it empties
itself on open and an empty panel would otherwise mean both "still
loading" and "nothing to show". Opening an entry clears the current
cards, race and diff and shows a loading state before fetching; a load
that fails becomes a standalone failure state, so a stale comparison is
never left sitting under the banner.

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
32000 instead of drawing a hard 400. On an offline boot there are no
published caps to clamp against, and an offline catalog is an unpriced
one, so an extended run there would be both unverified against the
provider and invisible to the spend ceiling; extended therefore falls
back to the standard tier until the catalog is available again. A
fetched catalog is unaffected, including the many models that publish no
cap at all. An extended run can cost up to
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
authoritative cost) and let budget analysis see
truncation on runs that produced no error. The generation id is read
from the response header, which arrives before any chunk, so a run
stopped mid-stream still records one. That ordering matters: stopping a
stream only stops the charge on providers that support cancellation, so
a stopped run is the one whose billing is least knowable locally and the
one that most needs an id to reconcile against later. If persisting a finished
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
and a full `Content-Security-Policy`: `default-src 'none'` with each
source opened only to `'self'` (`script-src`, `style-src`, `img-src`,
`font-src`, `connect-src`), plus `base-uri 'none'`, `form-action 'none'`,
and `frame-ancestors 'none'`. The policy needs no `'unsafe-inline'`
because the markup has no inline styles, the scripts create no style or
script elements, and the fonts, favicon, and every fetch and SSE endpoint
are same-origin; the pre-paint theme script is a same-origin file
(`static/theme-boot.js`) rather than an inline block, which is what lets
`script-src` stay `'self'`. The headers are added on response start with
no body buffering, so streaming is untouched.

`Cache-Control` splits by what the response carries. Static assets and
the index get `no-cache`, which means "revalidate before reuse" and not
"do not store", so with ETags present the usual cost is a 304 and a
stale script can never run beside a fresh one. Everything else gets
`private, no-store`: run details, group details, compare responses and
the SSE stream carry full prompts and full model answers, and those must
not be written to disk by any cache on the path. The favicon sets its
own year-long immutable directive and keeps it.

FastAPI's `/docs`, `/redoc` and `/openapi.json` are disabled. Nothing
here consumes them, and a machine-readable map of every route and bound
is attack surface on a server holding a paid API key.

To serve the bench beyond
localhost deliberately, edit `TRUSTED_HOSTS`
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
other copy. Documents you attach are in there too, as bytes in a
column rather than as files on disk, which is what keeps that last
sentence true: see **Attachments**.

Each row carries provenance, so a run stays interpretable after the
code, the prices, or the lineup have moved on. A group records the
prompt and the ordered lineup as declared, before any model is called,
which makes the group row the experiment record and fixes the group's
prompt ahead of its first member. Each run records the commit that
produced it (`app_sha`, best effort: None when git is unavailable), the
timestamp of the price catalog it was costed against
(`catalog_snapshot_at`, None on an offline boot), the sha256 of the
catalog bytes themselves (`catalog_digest`, so two boots a minute apart
against a changed catalog are distinguishable and None on an offline
boot), and the data-handling policy its payloads declared
(`data_policy`). `app_sha` is suffixed `-dirty` when the checkout had
uncommitted changes, and is None rather than a bare sha whenever that
could not be determined, because a bare sha is itself the claim that the
running code was exactly that commit. Each result records the
column it occupied (`position`, so a replay rebuilds the original
side-by-side layout from the rows instead of from a lineup that has
since been edited) and the exact payload that was sent (`request_json`,
authorization excluded, since auth travels in headers and never in the
body). Each result also records what the platform reported about the
exchange: the amount charged (`billed_cost_usd`), the hidden reasoning
and discounted cached token counts (`reasoning_tokens`,
`cached_tokens`), the host that served it (`provider`), and that host's
own word for why generation ended (`native_finish_reason`, beside
OpenRouter's normalized `finish_reason`). `quantization` is a column the
bench has never filled. It is not reported in-band, and it is not in
OpenRouter's published schema for the generation endpoint either, so the
reconcile path reads the key opportunistically and stores whatever comes
back. Nothing here has observed it come back: read a NULL there as "never
reported", not as "the provider served unquantized weights".

Some of that is only knowable after the fact. A run stopped mid-stream
never receives a usage object, and OpenRouter's `/generation` endpoint
holds the record it does have. `python -m bench.reconcile` walks the
result rows that carry a generation id and a gap the endpoint could
close (no trustworthy billed cost, or no `provider`, or no
`native_finish_reason`), asks about each one, and prints what it would
write:

```sh
.venv/bin/python -m bench.reconcile          # dry run, writes nothing
.venv/bin/python -m bench.reconcile --apply  # take the writes
```

Dry run is the default because this is the only path in the bench that
edits history. It fetches serially with a pause between lookups, and it
writes only `billed_cost_usd`, `provider`, `quantization` and
`native_finish_reason`. Response text, errors, timings and the local
estimate are never touched: this adds what the platform knows, it does
not revise what the bench observed. A field the endpoint does not report
leaves whatever was captured live in place (an unreported charge clears
only a stored one no reader would trust anyway), and a row the endpoint
cannot fill stays on the list and is asked about again next pass. An
expired record (the endpoint 404s for old generations) is reported and
skipped. It is safe
to run against a live bench, since the database is in WAL mode.
`--limit N` walks the oldest rows first, and `--delay` sets the pause.

Nothing runs it for you. Reconciliation is one upstream call per row, and
doing that at boot or inside a request would make a comparison wait on an
audit it did not ask for.

Databases from before these columns existed are migrated in place on
the next start, additively: nothing is renamed, dropped or retyped, old
rows keep their values, and every new column reads back as None on
them. `cost_usd` keeps its name and its meaning throughout, the local
estimate from catalog prices; the billed figure lives beside it in its
own column rather than overwriting the estimate's history.

**`bench/store.py` is synchronous by design, and a test enforces it.**
Every function materializes before returning, and no cursor is held
across an await. One connection is shared by every request, every
experiment trial and every scoring pass, and on a single event loop a
synchronous function that materializes before returning is an atomic
block: nothing interleaves between its first statement and its last. That
property, not a lock anyone has to remember to take, is what makes the
shared connection safe. Adding one `async def` there would remove it
everywhere at once and nothing would fail loudly, so the suite scans the
module's tokens and goes red instead. Cross-process concurrency is a
separate problem with a separate answer: WAL mode and the busy timeout.

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

## Where your prompts go

Every prompt you run leaves this machine. It goes to OpenRouter, and
OpenRouter forwards it to whichever provider it routes the request to.
Nothing about running the bench locally changes that, and this
application cannot see what a provider does with a prompt once it
arrives.

What it can do is state a data-handling preference on every request.
`BENCH_DATA_POLICY` picks one for the life of the process:

- `standard` (the default, and what an unset variable means) sends
  today's payload unchanged and asks for nothing about data handling.
  Providers that store prompts and may train on them are eligible.
- `deny` asks OpenRouter to route only to providers that do not collect
  user data (`data_collection: "deny"` in the provider preferences).
- `zdr` asks for zero-data-retention endpoints only (`zdr: true`), and
  sends `data_collection: "deny"` alongside it. The two settings govern
  different things, retention versus collection, and a mode chosen for
  confidential prompts should not leave the other one open.

An unrecognized value fails boot with a message naming the variable,
rather than quietly falling back to `standard`: a silent fallback would
send prompts to training-eligible providers while you believed
otherwise. The active policy is recorded per run in the `data_policy`
column, and when it is not `standard` a badge appears beside the session
stats so a confidential session is never ambiguous at a glance.

The important limit: this is a request, not a guarantee. The routing
constraint is OpenRouter's to honor, and the bench only shows you what
it asked for. When no endpoint satisfies the policy, OpenRouter refuses
the request and the card shows that refusal like any other provider
failure. The field names are pinned against OpenRouter's provider
routing documentation; the URL and the date it was read are in the
comment above `DATA_POLICY_PREFS` in `bench/models.py`.

## Attachments

Press **+ Attach** beside the prompt to run a comparison over a document.
Whatever you attach reaches every model in the comparison identically:
that is the same fairness law the prompt itself is held to, and nothing
about attachments is allowed to weaken it.

**Two modes, and they measure different things.** The picker sits next to
the files because the choice is part of the experiment, not part of the
plumbing.

- **inline** (the default) extracts the document's text and composes it
  into the prompt. What is being compared is how each model handles *the
  bench's reading* of the document. If the extraction is poor, every
  model is asked the same poorer question, which is exactly the property
  that keeps the comparison fair even when the parsing is bad. Every
  model can participate.
- **native** hands an image to the provider as a content part and lets
  the model read it itself. That measures layout, tables and handwriting
  that extraction cannot give you (see the note on `.docx` tables
  below), and it **changes which models can
  participate**, because a text-only model cannot take an image at all.
  It is strict mode's bargain in a new costume: a narrower population in
  exchange for a sharper question. So it is declared, recorded on the
  comparison, and capability-checked at creation against the catalog's
  `input_modalities` rather than discovered at the first paid call.

**What the native check clears, and what it does not.** It clears
*modality*: whether the catalog says a model takes image input at all. It
does not clear *how many*. OpenRouter states that "the number of images
you can send in a single request varies per provider and per model", and
the catalog publishes no per-model number, so there is nothing to check
the four-document cap against and the bench does not invent one. Four
images that each model accepts singly will pass creation and can still be
refused by the provider at request time. That refusal arrives as the
member's error, after the call rather than before it, and it is a real
gap in the pre-spend guarantee rather than an omission: the alternative
is a made-up limit that refuses comparisons which would have run.

Formats: inline reads `.txt`, `.md`, `.pdf` and `.docx`; native sends
`.png`, `.jpg`/`.jpeg` and `.webp`. GIF is supported by OpenRouter and
deliberately not taken here: an animated GIF is a sequence of frames, and
what a provider does with the frames it is not shown is undefined and
varies, so a comparison over one would not be a comparison over the same
input.

**Tabs and line breaks survive.** A tabbed line in a `.docx` reaches the
models as a tabbed line, and a break inside a paragraph as a newline.
Earlier builds dropped both, so `Name<tab>Value` arrived as `NameValue`:
a word the document does not contain, in the place a reader looks for the
value.

**Images are checked against their first bytes**, not only their
extension, so a file named `.png` that is not one is refused at upload
rather than posted to a provider as an image.

**Tables in a `.docx` are not read.** They could be: they sit in the same
part of the archive as the paragraphs. They are skipped on purpose,
because a table's text without its rows and columns is worse than no
table at all. Flattened cells arrive as a list of values with nothing
saying where a row ended, and every model then answers confidently over
an arrangement the document does not have. An absent table is something
you can see is absent, so **paste the table into the prompt** and every
model reads it the way you meant. Scrambled cells are something nobody
notices. Headers, footers, footnotes and comments are not read either;
those genuinely live in other parts of the archive.

A scanned PDF is refused at upload rather than attached empty. The bench
does no OCR, so a PDF whose pages carry no text layer would otherwise
reach every model as a document mentioned and never shown.

**A comparison carrying documents fails closed.** If the bench refuses to
create it (a text-only model under native mode, an image declared inline,
a composed prompt past the ceiling), nothing is sent: the refusal appears
at the attach control in the server's own words, and the run counter does
not move. Earlier builds swallowed that refusal and ran the models
one at a time without a comparison record, which sent each of them
whatever it resolved on its own.

**Caps.** At most 4 documents per comparison, at most 8 MiB each, at most
8 MiB of inflated XML out of any one `.docx`, 256 levels of nesting, and a
composed prompt of at most 200,000 characters. The composed prompt is
**also** checked against each model's own context window at creation, with
the arithmetic shown per model, because a prompt that is comfortable for
one member of a lineup can be a hard error for another and a comparison
where some cards answer and some error is not a comparison. The last one is the one
that matters most: past a provider's context limit some refuse and others
silently truncate, and a comparison where two providers truncated at
different points is not one comparison. Refusing at composition makes it
one refusal with the arithmetic in it instead of N providers each
deciding for themselves. The size cap is refused with the arithmetic
shown, and the composer refuses an over-cap batch **whole** rather than
attaching the first few, because a picker that quietly kept four of five
would leave you believing you had attached five.

**Where the file goes, and where it stays.** Uploads are ordinary JSON
POSTs carrying base64, never multipart. That costs 33 percent on the wire
and buys the invariant the whole boundary rests on: a multipart POST is a
CORS "simple" request that a hostile page could fire at localhost with no
preflight, and requiring a JSON body forces cross-origin senders into a
preflight this server never answers.

The bytes are stored **once, in `bench.db`, keyed by their sha256
digest**, so attaching one contract to ten comparisons costs one copy of
it. Nothing is written outside that file, and no endpoint serves a
document back: the metadata responses have no content field and the
readers behind them never select the blob. Deleting `bench.db` deletes
your documents along with everything else, which keeps the claim above
in **Local data** true.

**Deleting one document takes its text with it.** There is no delete
endpoint; the way to remove a single document is `DELETE FROM attachments
WHERE digest = ...` in `sqlite3`, and the extracted text lives in a
second table keyed by the same digest. A database trigger removes those
rows with the row you deleted, so the text does not stay behind under a
digest whose file you believed you had removed. It is a trigger and not a
foreign key on purpose: a foreign key cascade does nothing unless the
connection sets `PRAGMA foreign_keys = ON`, and the `sqlite3` command
line does not. What remains after the delete is what should: comparisons
that cited the document still say they cited it, and the history shows
the reference with its metadata blank rather than dropping it.

The comparison record cites the digest; it never inlines the bytes.
`request_json` stores the composed payload with a digest reference
standing where the content sat, so the exact wire bytes are
reconstructible from the digest, the stored content and the stated
composition, and a result row is not a second copy of every document ever
sent. An inline reference names the character count, a native one names
the byte count and the media type, and the two are separate strings
because a shared one would have to mislabel one of them.

**Privacy.** A document goes exactly where a prompt goes: to OpenRouter,
and on to whichever provider it routes to. `BENCH_DATA_POLICY` governs it
the same way and with the same limit, that the routing constraint is
OpenRouter's to honor. The attach control states the session's policy in
words right where you pick the file, including on the default policy,
because the default's only other signal is an absent badge and silence is
a poor way to say a contract is going out under ordinary terms.

Treat a document as **untrusted input**. The composition marks each
attachment with a visible delimiter and tells the model to read it as
reference material rather than as instructions, which is the only thing
the composition can do about it: a document's own text can try to
instruct the model, and marking the boundary clearly is a mitigation, not
a guarantee.

The inflated bound is there because the 8 MiB upload cap says nothing
about what comes out of the upload: deflate reaches ratios past 1000:1 on
repetitive XML, so a small, perfectly valid `.docx` can ask the process to
materialize a gigabyte. That would stall comparisons already running and
paid for, which is the expensive kind of damage from the cheap kind of
input. Extraction also runs off the event loop, so even a slow legitimate
parse does not freeze the progress of runs in flight.

Filenames are checked as well as bounded: no control characters, because
the name is written into the delimiter line the models read and one
carrying a newline could break that line in two; no path separators,
because the bytes go into the database and a path is a claim the bench
cannot honor.

**Provenance.** Every attachment row records the extractor that read it
and that extractor's version, because a `pypdf` upgrade changes the text a
model reads, and a record that could not say which parser produced it
would be a record of a prompt nobody can reconstruct.

A comparison **pins the rendition** of every document at creation:
`(digest, extractor, extractor_version, kind)`. Members receive exactly
the pinned reading, so a parser upgrade partway through a comparison
cannot hand the second model a different document from the first; if a
pinned reading is missing the bench refuses rather than substituting.
The same pin is recorded on ungrouped runs, so a single-model comparison
shows its documents on replay too.

The same bytes uploaded under two suffixes are **two renditions**, each
described truthfully, because a `.docx` is also decodable as text and the
digest alone cannot say which reading was meant. That ambiguity used to
resolve to whichever upload arrived first, which let a PNG uploaded as
`.txt` reach every model as binary in the text path and left the same
image permanently unusable in native mode.

Content dedupes by digest; the EXTRACTION dedupes by digest **and** parser
version. Upload the same file after a parser upgrade and the bench
re-reads the stored bytes rather than handing back the old text under the
new version's name. The bytes are still stored once. Earlier readings are
kept rather than overwritten, because a comparison recorded under the
earlier parser cites that reading's character count in its own record, and
overwriting would make it unreconstructible while leaving it looking
exact. The replay banner
and the chip titles show it. Exports carry each trial's digests and mode
and never any content, and the export manifest carries
`attachments_referenced` so an artifact says at line one whether it
references bytes it does not embed, exactly as `thresholds_included` says
whether it embeds the threshold slice.

The token figure on a chip is labeled approximate and is characters over
four. The bench does not tokenize: every model runs its own tokenizer and
they disagree, so the number is an order of magnitude and nothing more.
For a native image there is no figure at all, only a note that the
provider decides, because an image's token cost depends on tiling and
resolution handling that this bench cannot see and a fabricated number
beside a real byte count would be believed.

**Deliberately out of scope:** per-task attachments in datasets, OCR for
scanned PDFs, native parts for non-image documents, audio and video, and
multi-file diffing.

## Experiment controls

By default the bench sends one user message and lets every provider apply
its own sampling defaults, which means a comparison varies whatever the
providers feel like varying. Six controls let one comparison send the
same values on every request instead. Sent, not enforced: what a provider
does with them is its own business, and the "silently ignore" note below
is the part of that story you have to plan around.

| Control | Sent as | Range |
| --- | --- | --- |
| system prompt | a leading `system` message | up to 8000 characters |
| temperature | `temperature` | 0 to 2 |
| top_p | `top_p` | 0 to 1 |
| seed | `seed` | any integer |
| reasoning effort | `reasoning.effort` | low, medium, high |
| routing mode | `provider.sort` | throughput, price, default |

Two rules govern all six.

**A control you leave blank is not sent.** It does not appear in the
payload at all, so the provider applies its own default and the record can
say honestly that the bench did not choose one. The bench never fills in a
value on your behalf, and with every control blank the outgoing payload is
byte for byte what it was before these controls existed. A test asserts
exactly that, against a request body captured from the previous release.

**Only what you set is recorded.** `groups.params_json` holds the controls
you chose and nothing else, so a history badge appears for a control you
picked and never for a default. A group with no controls records nothing
rather than an empty object.

The controls belong to the comparison, not to a model: one comparison, one
controls set, applied to every model, which is what makes the columns
comparable. A group holds one experiment the way it holds one prompt, and a
run whose controls disagree with its group's is refused with a 409 at
entry, before any upstream call, naming the controls that conflict.

In the browser they live behind **+ Controls** in the composer, collapsed
by default and blank inside. Nothing is pre-filled, because a box showing
`1.0` would look like a decision you had made. The collapsed row summarises
what is set, so a controlled run is never started blind, and a control the
browser can see is out of range disables Run and says which one rather than
letting the request fail once per model. A rerun of a failed card replays
the controls that card ran under, not whatever the panel holds by then: a
rerun is a second sample of the same experiment.

History shows the same controls as compact badges, `t=0.25`, `seed 7`,
`sys`, `effort high`, `route price`, and shows them only for controls that
were set. The system prompt gets a presence badge rather than its text,
since a row is one line and a truncated prompt would invite comparing two
comparisons on an excerpt that happens to match; open the comparison and
the full text is there above the cards. One asymmetry worth knowing: an
ungrouped run (a single-model rerun, or anything from before groups
existed) derives its badges from its recorded payload, and routing cannot
be recovered that way, because the bench's own throughput preference
appears in every payload it has ever sent. Rather than present that as a
choice, an ungrouped run shows no routing badge at all.

### Then versus now

Running an old experiment against today's models takes three steps, and the
hard part was already built:

1. **Reuse.** Open a comparison from History and press **reuse**. The
   composer fills with its prompt and every control it set. Your lineup is
   left exactly as it is, because which models you want to compare *now* is
   the question you are asking. Nothing runs: money moves on Run and on
   nothing else.
2. **Run.** The new comparison is sent with the same controls, so it records
   the same `params_json` as its source. Two runs of one experiment, months
   apart.
3. **Arm and diff.** Open the old comparison again, arm a card with
   **diff**, then arm the matching card from the new run. The panel shows
   what changed. This works because arming survives a history replay by
   design, which the Phase F.2 work put in deliberately; nothing was added
   here to enable it.

There is no stored link between a reused comparison and its source. The
prompt and controls match, which is what makes them comparable, but the
bench does not claim a lineage it would have to maintain.

One honest limit. Reuse from an **ungrouped** run cannot restore routing,
for the same reason its badges cannot show it: routing is not recoverable
from a payload. The reuse button says so before you click it and the
composer says so after, so a prefill never quietly hands you a different
experiment than the one you asked to repeat. Reuse from a group, which is
what the browser always creates, restores every control exactly.

Routing mode picks how OpenRouter chooses among the providers serving a
model. `throughput` is the default and is what the bench has always sent;
`price` prefers the cheapest; `default` sends no sort at all, which is the
documented way to ask for OpenRouter's own price-weighted load balancing.
Routing merges into the same provider object the data-handling policy uses,
never displacing it.

**About seeds.** Providers differ in whether they accept a seed at all and
in how deterministic they are when they do. OpenRouter's own documentation
says repeated requests with the same seed and parameters *should* return
the same result and that determinism is not guaranteed for some models. The
bench passes your seed through and records it. It promises nothing beyond
that, and a run that came back different with the same seed is a fact about
the provider, not a bug here.

**Providers may silently ignore a control they do not support.** This is
not a gap in the bench, it is OpenRouter's documented routing behavior:
under the default strategy, a provider that does not support every
parameter in your request still receives the request and ignores the
parameters it does not know. Nothing errors and nothing warns. So a column
in your comparison may have been generated at the provider's own
temperature while the payload asked for yours, and neither the response nor
the card can tell you that happened.

Two things make it survivable. The exact payload is recorded per result in
`request_json`, so a run whose temperature a provider ignored is still a
run that can be shown to have asked for it. And the model's own catalog
entry is where support is documented, so a control that matters to your
comparison is worth checking against the models you picked.

There is a routing flag that converts that silence into a hard failure,
`require_parameters`, which restricts a request to providers supporting
every parameter it carries. **The bench deliberately does not send it.**
It would change which providers are eligible, and changing the eligible
set changes what is being measured, which is the opposite of what these
controls are for: you would be comparing a different population of
providers depending on which controls you set. Choosing that tradeoff is a
decision for a later phase, not a default to slip in with this one.

Every field name, bound and behavior above is pinned against OpenRouter's
current documentation, with the URLs and the dates they were read in the
comments in `bench/models.py`.

## Datasets

A dataset is a JSONL file, one task per line:

```json
{"id": "add-1", "prompt": "What is 17 + 25? Reply with the number only.", "reference": "42", "scorer": {"kind": "normalized_exact"}}
```

`id` and `prompt` are required and every other field is optional. `system`
is sent as that task's system message. `reference` is the expected answer
for the comparing scorers. `rubric` is the scoring instruction for the
judge. `scorer` names how the task is scored: `exact`,
`normalized_exact`, `contains`, `regex` (with a `pattern`), or `judge`.
Task ids must be unique within a file.

**A dataset's version is its content.** There is no version field to keep
in sync and no way to edit a file and leave a stale label behind: the
identity is the sha256 of the raw bytes, recorded on every experiment that
reads it, the same content-derived rule the asset revision and the catalog
digest already follow. Two experiments citing the same digest read the
same file, and that is checkable rather than promised. It is a digest over
bytes, not over meaning: reordering the keys in a line changes the digest
without changing the tasks, which is the honest limit of what a byte
digest can claim.

Validation happens when the experiment is created, not when scoring runs.
A misspelled scorer name, a judge task with no rubric, or an `exact` task
with no reference all fail at that point, naming the line, because
discovering them after a run has paid for every trial is discovering them
in the most expensive place available.

**A regex cannot freeze the server.** The regex scorer runs its pattern in
a single-use child process under a one-second deadline; expiry is recorded
as a scoring failure (`regex exceeded deadline`) and never as a guessed
verdict, because the bench does not know whether that pattern would have
matched. Every other deterministic scorer stays in process, where a string
comparison is linear in the subject and already bounded.

That second is **wall time on the monotonic clock**, and the distinction
is what keeps the number meaning one thing. It is not CPU time, so a busy
machine gives the pattern less work per second and the same second of
grace; it is not system-clock wall time, so an NTP step during a scoring
pass cannot shorten or extend it. The bench does no time arithmetic of its
own here: the value goes to `Process.join`, and the clock comes from
`multiprocessing.connection.wait`, which keeps its deadline on
`time.monotonic`. The test walks that chain rather than trusting it, so a
CPython change that swapped the clock fails a test instead of quietly
changing what the constant means.

The subject and pattern limits are necessary and were never sufficient,
which the comment above them used to deny. Backtracking is exponential in
the subject, so `(a+)+$` (six characters, well inside the 500-character
pattern limit) against ten thousand characters runs effectively forever.
Measured in process at small sizes it doubles every two characters: 0.03s
at 18, 0.11s at 20, 0.44s at 22, 1.78s at 24. A process rather than a
thread because `re` holds the GIL while it backtracks and only a process
can be terminated; spawn rather than fork because forking an asyncio
application copies whatever locks its threads happened to hold. The wait
itself happens off the event loop, so the regex never runs in this thread
or in this process.

**Empty counts as missing** for a rubric or a comparing scorer's
reference, and the empty case is the more dangerous of the two because it
scores rather than failing: every string contains the empty string, so a
`contains` task with `"reference": ""` would hand every model on every
repeat a perfect 1.0, and nothing downstream could tell that number from
a real one. The trial completed, the scorer ran, the score is in range,
and the coverage counters say it was scored. The dataset file is the only
place the meaninglessness is visible, so the refusal lives there.

Two examples ship in `bench-datasets/`: `arithmetic.jsonl` (deterministic
scoring) and `summarize.jsonl` (rubric scoring). Your own files live
wherever you keep them; the bench reads the path you name. There is no
path allowlist, deliberately: the bench answers only to loopback clients
and runs as you, so restricting the path would defend you against yourself
while blocking the ordinary case.

## Experiments

An experiment is the aggregate above groups: one dataset, one lineup, one
budget, one controls set, run for a stated number of repeats. Groups are
unchanged, still the atomic one-prompt record created before any call and
enforced at entry; each trial creates one, and four new columns say which
experiment and which cell (task, repeat, rotation) it belongs to. A group
with those columns NULL was run by hand, which is what every group in a
pre-Phase-I database was.

```sh
curl -X POST localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "arithmetic sweep",
       "dataset_path": "bench-datasets/arithmetic.jsonl",
       "lineup": ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"],
       "budget": "standard",
       "repeats": 3,
       "params": {"temperature": 0}}'
```

The row is written complete before anything runs, exactly as a group row
is written before its first upstream call: it is the declaration, and what
happened gets checked against it rather than assembled into it. That
includes the dataset digest, the build (`app_sha`), the catalog digest,
the data-handling policy, and the trial count. The trial count is stored
rather than derived, because an experiment that halts leaves its un-run
trials with nothing behind to count, and a denominator derived from
existing rows would make a halt look like a completed run.

`tasks x repeats x lineup` is the number of paid calls, and it is bounded
at creation. An experiment over that bound is refused with the arithmetic
shown, because the caller has three levers and needs to know which one to
pull.

## Estimands

Every experiment declares which of two questions it is answering, and
every report, every export line and every row says which one produced it.
A number without its estimand is a number about nothing in particular.

**`routed_service` is the default**, and it is what this tool is actually
used for: the OpenRouter-routed service path for a model under a stated
routing mode. The provider is chosen dynamically per call, which is a
confound rather than a defect, and the bench handles it in the open:
repeats do the statistical work, the report counts which providers served
each model as its own column, and Phase G's per-result provider field is
what makes any result stratifiable after the fact. This estimand makes no
capability claim, so it needs no catalog and sends no routing restriction
beyond the data policy and the routing mode.

**`underlying_model` is the opt-in strict estimand**, for when the claim
is about the model itself rather than about the service in front of it.
It narrows the eligible provider population on purpose:

```sh
curl -X POST localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "strict sweep",
       "dataset_path": "bench-datasets/arithmetic.jsonl",
       "lineup": ["openai/gpt-4o-mini"],
       "budget": "standard",
       "estimand_mode": "underlying_model",
       "provider_pins": {"openai/gpt-4o-mini": "openai"},
       "quantizations": ["fp8"],
       "params": {"temperature": 0}}'
```

It sends `require_parameters: true`, so a provider that would have
silently ignored the temperature is ineligible instead. The routed-service
path deliberately does not send it, because changing which providers are
eligible silently changes what is being measured; here that change is the
whole point.

What that check can and cannot claim is worth stating. It verifies
AGGREGATE `reasoning` support, not that a named effort tier means the same
thing everywhere: the reasoning-tokens documentation says that "for models
that only support `reasoning.max_tokens`, the effort level will be set
based on the percentages above", so a provider advertising `reasoning` may
be translating "high" into a token allocation rather than honouring the
tier. Strict mode checks what it can check and claims exactly that.

`provider_pins` is strict-mode only, normalized to the documented
lowercase provider slugs at creation so the recorded pin and the sent pin
are one string, and a pin travels as
`order` **with `allow_fallbacks: false`**, never without it: an order that
can be departed from is a preference, and a pinned run served by somebody
else would record a constraint that did not hold.

`quantizations` is the other strict-mode narrowing, and it is optional.
Quantization varies by host and changes what the weights actually are, so
two runs of "the same model" served at bf16 and at int4 are not measuring
the same artifact. It rides the documented provider filter and its values
are validated against the documented levels, because a level OpenRouter
does not recognize filters to no provider at all and, under
`allow_fallbacks: false`, that is a run that fails. **The throughput sort
never stabilized quantization** and never claimed to: it biases routing
toward serious hosts, which is a different property entirely. A pin naming a model
outside the lineup is refused, since a declaration that cannot be honored
should not be stored as though it will be.

Every control is capability-checked against the catalog at creation, not
at trial one of three hundred, and the checks refuse rather than assume:

- **No catalog, no strict experiment.** The check needs the catalog, so
  an offline boot refuses creation and says so. Skipping it silently is
  the tempting version and the wrong one: the rows would carry
  `estimand_mode: underlying_model` with nothing behind it, and no reader
  afterwards could tell them from rows where the check ran. A label nobody
  verified is worse than no label. The routed-service estimand is
  unaffected, because it never made the claim.
- **A model the catalog does not list, or lists without
  `supported_parameters`, is refused.** Absence of evidence is not
  support. An empty list is a real answer and a different one.
- **An unsupported control is refused**, with the model and the parameter
  named. Under `require_parameters` an unsupported parameter is not
  ignored, it makes every provider ineligible, so the experiment would
  fail every trial rather than measure anything.

`max_tokens` is checked even though no control sets it, because every
payload the bench builds carries one.

## Running an experiment

```sh
curl -X POST localhost:8000/experiments/1/start \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "bench-datasets/arithmetic.jsonl"}'

curl -N localhost:8000/experiments/1/progress    # SSE counters
curl -X POST localhost:8000/experiments/1/stop -H "Content-Type: application/json" -d '{}'
```

The path is given again at start, and the digest is re-checked against
the one recorded at creation. A file that changed in between stops the
experiment before it spends anything, because running would produce a
record citing one dataset and containing another.

One experiment runs at a time. They share the five upstream slots and the
spend ceiling, so two at once would interleave through the same queue and
each would measure the other's waiting; a second start gets a 409 saying
exactly that. Every trial goes through the same semaphore, the same
post-admission ceiling recheck, the same settlement inside the held slot
and the same entry checks as any browser run. The runner creates its
groups through the normal path with no bypass, so experiment-to-group
consistency is the law the manifest check already enforces rather than a
promise the runner makes about itself.

**Repeats and seeds.** Repeats exist to sample variation. If the
experiment's controls carry a seed, repeat N sends `seed + N`: the whole
set stays reproducible while each repeat differs, where one fixed seed
would buy the same sample N times at N times the price. If no seed was
set, none is sent and each provider applies its own default, which is
rule one unchanged. The derived seed rides `request_json` on every result
like any other control, so the record says what was actually sent.

**Rotation.** Models enter the semaphore in lineup order, so under
saturation the first-listed model reliably gets a slot first and the
last-listed reliably waits. That is a systematic advantage to being
listed first, and repeats do not average it away because it points the
same direction every time. The entry order therefore rotates per cell,
and the rotation index is recorded on the group. Over a full cycle every
model occupies every entry position an equal number of times. Position in
the report is still the model's place in the declared lineup; rotation
changes who asks first, not which column a model owns.

**Task order** is file order unless `task_order_seed` is set, in which
case it is shuffled with that seed and the seed is recorded. There is no
unseeded shuffle, because an order nobody can reproduce makes an
experiment unrepeatable by its own author.

Task order and halting interact, and the interaction is the reason to set
a seed. Trials run task-major: all repeats of the first task, then all
repeats of the second. That is deliberate, because an experiment cut
short then leaves complete cells (every repeat of some tasks) rather than
one repeat of everything, and complete cells are what a per-task number
can actually be computed from. The cost is that the surviving subset is a
prefix of the task order. In file order that prefix is whatever the
author happened to put at the top of the file, so a halted run would
report on the easy warm-up questions if they were written first, and
would do it without saying so. A recorded seed makes the prefix an
arbitrary but reproducible subset of the dataset instead of a privileged
one. If you expect to hit the ceiling, or you are running a dataset you
did not write, set the seed.

**Two seeds, and only one of them increments.** `task_order_seed` is used
once per experiment, so it is bounded against the safe-integer limit and
nothing further. `params.seed` is the SAMPLING seed, and repeat N sends
`base + N`, so an experiment is refused at creation when
`seed + repeats - 1` would pass that limit, with the arithmetic shown.
The rule shipped guarding the wrong one of the two: it protected a value
that cannot overflow and left the one that can unguarded, which is a
check that looks exactly like coverage.

**Interruption is honest.** A stop halts between trials, never inside
one: a trial that reached upstream has already spent its money, and
abandoning it would throw away a result you paid for. The experiment ends
`stopped` with its partial record intact, and the trials that never ran
are visible as the gap between `trials_done` and `trials_total`. If the
spend ceiling refuses a trial, the default (`halt_on_refusal`) stops the
experiment and says so in `status_detail`; set it false to record
refusals and keep going. Either way the conservation property holds
across the whole experiment: every requested trial is accounted for as
completed, failed, refused, or never attempted.

**Blind rating is entered from the history list**, never from a
comparison already on screen. The server opens the session, issues the
shuffle, and returns the answers with no identity attached to any of
them: no model, no provider, no cost, no timing. That ordering is the
feature. A page that fetched the identified comparison and then hid the
identities has already painted them, and "hidden" in a browser is a style
rule anyone can undo plus a frame the rater may have seen.

**The blind flag on a rating is the server's, not the client's, and it is
decided by a token.** Opening a blind session issues one; a rating is
recorded blind only when it arrives bearing the token of a session the
server still has open. A page that revealed the identities and then posted
`blind: true` would be testifying about its own past, which is the one
witness a blind record cannot use.

A per-group boolean was not enough. It answered "is a blind session open
for this comparison", which is true for **every tab** the moment any one
of them opens one, so a second window replaying that comparison sighted
posted its ratings into somebody else's blind and they persisted
`blind = 1`. The token answers the question that was meant: did **these
ratings** come from a blind session. A missing token, an invented one, or
one issued before the reveal all persist `blind = 0`. Several sessions may
be open on one comparison at once, because two people rating it are both
legitimately blind; one reveal closes all of them, because the answer key
is out.

**What the flag attests is the path the rating took**, and stating that is
the point rather than a hedge. It says these ratings came from a view the
server built without identities, on a session it issued and had not yet
closed. It cannot attest what the person in the chair arranged to see by
other means: this is a localhost tool, the operator owns the process, the
database and the browser, and they can keep the export open in one window
and the blind view in another. The claim is narrower than "this person did
not know". It is "the bench did not tell them", which is the only party
the bench can speak for. What the boolean got wrong was different in kind:
it attested a path the ratings had not taken at all.

Two further limits are stated rather than discovered. The bench cannot
blind what a model wrote about itself, so an answer beginning "as an AI
assistant made by X" identifies its author whatever the page does. And a
rater who has already replayed a comparison sightedly knows its contents;
the session records the conditions at rating time, not the contents of the
rater's memory.

**Progress survives disconnection**, and this is the one place in the
bench where continuing after a client goes away is the point rather than
a bug. A browser run belongs to the tab that started it and is abandoned
when that tab closes, because nobody is watching. An experiment belongs
to the bench: a laptop lid closing halfway through a paid sweep must not
silently end it. The progress stream is a view onto the run, not the run
itself, and every frame carries absolute counters rather than deltas so a
reconnecting client is caught up by the next frame.

## Scoring

Scoring is a separate pass over a finished experiment, so it can be run,
re-run and changed without re-running a single model call.

```sh
curl -X POST localhost:8000/experiments/1/score \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "bench-datasets/arithmetic.jsonl",
       "judge_model": "openai/gpt-4o-mini"}'
```

Deterministic scorers (`exact`, `normalized_exact`, `contains`, `regex`)
are pure functions over the stored response text. `normalized_exact` and
`contains` fold case and collapse whitespace; `exact` strips only the
surrounding whitespace that chat completions add.

**A trial with no response text scores zero and fails.** It is not
skipped. An errored or stopped trial is a trial that failed the task, and
excluding it from scoring is how a report ends up quietly reporting on
survivors only. The score row's detail says "the trial did not complete",
so a reader can tell a wrong answer from a missing one.

**Judge scoring is absolute, not pairwise, and blind by construction.**
The judge is sent the rubric, the reference if the task has one, and the
response text. It is never sent the identity of the model that produced
the response, and that is enforced by the shape of the code rather than
by care: the function that builds the payload is not given the model
name, so it is not in scope for any future edit inside it. Pairwise
judging with position swapping is deliberately deferred; it arrives with
its swap machinery or not at all.

The judge is asked for `{"score": <0 to 1>, "reason": "..."}` and the
reply is parsed defensively. A fenced or preamble-wrapped object is
recovered, because that is the same object and refusing would discard
correct verdicts over formatting. Anything else, including a score that
is not a number or falls outside `[0, 1]`, is recorded as a scoring
failure with the reason attached. **A guessed number is never written**:
it would enter an average and change a conclusion while looking exactly
like a measurement.

**A judged pass needs a threshold you declare.** A rubric defines a
graded score, and 0.5 might mean "covered half the required points" in
one rubric and "wrong but polite" in another, so the bench never supplies
a cutoff. Add `pass_threshold` to a judge scorer and `passed` is derived
from it; leave it out and `passed` stays empty:

```json
{"id": "sum-1", "prompt": "...", "rubric": "...", "scorer": {"kind": "judge", "pass_threshold": 0.75}}
```

Reports therefore say **pass rate where a threshold was declared, score
mean otherwise**, and mean it. `pass_threshold` is refused on a
deterministic scorer, which already produces a pass, for the same reason
every other inert key is refused: accepting a setting that does nothing
is worse than saying no to it.

An unparseable verdict has no score and so no pass either, rather than
counting as a failure. Collapsing the two would put the judge's own
malfunctions into the model's pass rate.

**Judge calls are spend.** The billed cost is captured in band, recorded
on the score row, and added to the same accumulator the ceiling reads, so
a scoring pass cannot run free against the limit. Judges get their own
modest completion budget (`JUDGE_MAX_TOKENS`) rather than the
experiment's tier, because a verdict is a number and a sentence and a
judge inheriting an extended budget would buy headroom no rubric needs,
once per scored trial. Judge payloads carry the boot data policy like
every other request.

**If the judge model is in the experiment's lineup**, every score it
produces is flagged `self_judged` and the flag is surfaced in the report.
The pass is not refused: you may have good reason, and silently absorbing
the self-preference concern is what would be wrong.

**A ceiling refusal during scoring is recorded as that result's scoring
failure and the pass continues.** This is the opposite of the trial
runner's default, deliberately. A refused trial can only be recovered by
paying for the model call again, so halting protects the budget for a
decision you should make. A refused score can be filled in by a later
pass over the same stored text at no extra model cost, so stopping the
whole pass for one would trade a complete scoring run for nothing. Re-run
the pass and the gaps fill in.

## Reports

```sh
curl "localhost:8000/experiments/1/report?dataset_path=bench-datasets/arithmetic.jsonl"
```

Computed at read time from the rows, never cached and never stored. A
stored aggregate can disagree with the rows it came from, and the day it
does there is no way to tell which is wrong.

**The report keeps two axes apart, everywhere.**

*Axis one, trial outcomes:* `done`, `error`, `refused`, `stopped`,
`missing`, `not_run`. This is what happened when the bench asked a model
to do a task, and it is where a model's failure rate comes from. Outcomes
are derived at read time from fields that were already recorded for their
own reasons, so the vocabulary can change without a migration and no old
row carries a label that predates the rules.

**Two absences, and they are different facts.** `missing` means the cell
exists (the group was created, other models in the lineup have rows in it)
and this model left none: a hole in a cell that ran. `not_run` means the
cell was never created at all, because the runner halted or was stopped
before reaching it: a plan abandoned. Folding the two together would hide
a halt inside a gap.

**The plan comes from the plan, not from the rows.** `planned` is
`tasks x repeats` per model, read off the experiment's own recorded counts
rather than off the groups that happen to exist. That is not a detail:
deriving it from rows is what makes a halt invisible, because eight trials
stopped after two would report a plan of two and nothing on the page would
say six were owed. The report publishes the arithmetic as its own `plan`
block so the counters can be checked against it instead of trusted, and
the export manifest carries `tasks_total` so `not_run` is derivable from
the artifact alone.

**A refusal is a row.** When the spend ceiling declines a trial the runner
persists a result carrying the refusal error beside a NULL `request_json`,
which is exactly the pair the era-gated derivation reads as `refused`. The
derivation was always there and had nothing to read: the runner returned
before writing anything, so a refused trial surfaced as `missing` and a
budget fact was published as a gap in the record.

`attempted` is the subset that ran to a **model-attributable end**:
`done` and `error`. A refused trial was declined before the call went
out, a missing trial left no row, and a not-run trial has no cell, so
none of the three is an attempt. Neither is a **stopped** one: it was cut
off part way through by an operator or a disconnect, so nobody knows
whether it would have succeeded, and counting it made the published
failure rate depend on when somebody pressed stop. Stop a run early and
the rate falls, with nothing on the page connecting the two.

That set is `QUALITY_OUTCOMES`, which is not a coincidence: the trials a
quality number may be computed from and the trials a failure rate may be
computed over are the same trials, because both questions are "what did
the model do". **`failure_rate` is `error / attempted`** and nothing else.
`refusal_rate` is `refused / planned`, because a refusal is a fact about
the budget against the plan.

**Three surfaces agree, and that is asserted.** The progress counters, the
report's outcome counts and the export's lines are three derivations of the
same facts, reachable at three different times. If they disagree at least
one published number is wrong and nothing says which, so the agreement is
tested directly, over a halted run and over a continue-mode run where
refusals leave rows the halted one never had.

*Axis two, scoring coverage:* `scored`, `scoring_failed`, `unscored`.
This is what happened when the bench tried to put a number on a trial. A
judge that returned gibberish, or a judge call the ceiling refused, lives
here and only here. A trial that never ran is on neither axis: it has no
response to score, so calling it `unscored` would report a gap in
coverage nobody could ever close.

**One gate decides which trials a quality number may speak for**, and the
mean, the interval's clusters, the pass rate's denominator and both
coverage counters all sit behind it, so they cannot drift into disagreeing
about their population.

In: `done` and `error`. A completed trial is evidence about quality; an
errored one is a trial that failed the task, and dropping it would average
over survivors and flatter the models that fail most.

Out: `refused`, `stopped`, `missing`, `not_run`. Each for its own reason,
and none of them about the model. A refusal is the spend ceiling declining
to buy the answer, a stop is the operator ending the run, and the two
absences are the experiment's. Scoring any of them zero publishes a budget,
an operator or an abandoned plan as capability. Counting them as `unscored`
would be no better: that reports a coverage gap nobody could ever close.

This is what the previous rule got wrong, and the shape of the mistake is
worth keeping. The two-axes design was specified precisely for `error` and
for unscored trials, and the tests pinned exactly those two; `refused` and
`stopped` were left to inference and fell through the failure-inclusive
branch. One perfect answer beside one refused trial published **0.5**: the
model looked half as good because the operator ran out of money. The rules
were right and the coverage of the rules was not, so the treatment of every
outcome is now a matrix test with a row per outcome, and the next outcome
anyone adds gets an empty row that fails until it is filled in deliberately.

Inside the population there are exactly two rules. An errored trial
contributes zero. A completed trial with no usable score contributes
nothing, because its absence belongs to axis two and scoring it zero would
put the judge's malfunction into the model's mean.

**A score row on a stopped trial is neither deleted nor obeyed.** It stays
in the database and in the export as the audit trail of what the scoring
pass did, and it never reaches a published number.

**A task the exclusions hollow out leaves the bootstrap.** If every trial
of a task was refused, stopped or never run, it contributes no cluster
rather than an empty one, and the interval's stated `n_clusters` says so.
An interval claiming twenty tasks when six are empty is false precision
arriving by the back door, which is the thing this layer exists to stop.

**A scorer answers only for the tasks that declared it.** Each scorer gets
its own section, computed over its own tasks and no others. A section that
iterated every task was dragging its neighbours' numbers around: a task
scored by `contains` landed in the `judge` section, where its errored
trials added zeros to the judge's mean under the failure-inclusive rule and
its completed trials added to the judge's unscored coverage, all for a
scorer never meant to look at it. Every axis rule was being applied
correctly to the wrong population, which is why the two-axes work could not
see it: the crossing was between the sections, not between the axes.

The file is authoritative for any scorer it mentions, exactly as it is for
thresholds; where it says nothing, the rows are the witness with the same
floor limit. A scorer the file NEVER mentions falls to the rows even when a
file was supplied, and `human` is why: no dataset declares it, so reading
the file's silence as "applies to nothing" would delete every human mean
from any report built with a dataset path.

**A ranking is a claim, and it names its metric.** Ordering models says one
did better, and that means nothing until somebody says better AT WHAT. The
report ranks when `primary_metric` is declared, or when exactly one scorer
exists and there is no choice to make. Otherwise it publishes every
scorer's section in full and **no cross-scorer ranking**, and says why.

The previous rule ranked on the first scorer alphabetically, so an
experiment scored by `contains` and `judge` was ordered by `contains`
because c sorts before j. Nobody chose that, the report did not say it, and
adding a scorer named `accuracy` would have silently reordered the
leaderboard. `primary_metric` is validated at creation against what the
dataset can actually produce, plus `human`, since a person rating trials
after the fact is the one scorer no file can declare.

**A series is a (scorer, judge) pair, not a scorer.** Two judges grading
the same trials are two instruments that disagree on purpose, so they get
two rows with their means, intervals, pass rates and flags kept apart, and
there is **no combined cross-judge number anywhere**: averaging a strict
judge with a lenient one publishes a figure neither produced. Human ratings
are their own series under `human`, and deterministic scorers carry no
judge.

**A legacy duplicate never becomes a phantom.** Before I.2 the runner
derived a result's position from `lineup.index(model)`, which returns the
first match, so a lineup listing one model twice wrote position 0 on both
rows. Read back under the arm rules, both mapped to arm 0 and arm 1 matched
nothing: the report invented a **missing** trial for an arm that had run
and been paid for, and handed arm 0 both charges. Such a cell is now
reconstructed from row order, which is the order the runner ran them, and
every arm of that model carries `legacy_ambiguous` with the report's
`arm_caveat` saying that WHICH copy is which is a reconstruction. A cell
whose rows recorded distinct positions is a record and carries no caveat.

**Every arm appears in every series.** Series are discovered once across
the whole experiment, so an arm the scoring pass never reached still gets
its section, saying unscored on axis two and owing the series its
failure-inclusive zeros on axis one. Discovered per arm, from that arm's
own rows, such an arm vanished from the scorer table entirely, and absence
on a page reads as "nothing to say" rather than as "nobody looked". Which
tasks a section covers is still decided by the scorer that declared them.

Selection uses the full `(scorer, judge_model)` key. It used to filter on
the scorer alone and take the last row, so two judges of one trial fought
over a single slot and whichever ran second answered for both, with nothing
on the page saying a first judge had ever run. The rule itself was already
written and correct in `latest_per_key`; the report simply never called it,
which is why "every helper that implements a key gets its call sites
audited" is now a standing review lens.

If the ranking metric resolves to more than one judge, the ranking is
withheld and names them. It is the same question one level down: better
according to whom.

**A series nobody measured cannot order anything.** Every section carries
`measured`: how many of the `n` values behind its mean came from a score
rather than from the failure-inclusive rule. A series where that is zero
across every arm ranks nothing, and the ranking is withheld saying so.

The reason is that the mean is not None in that case, which is the trap.
An arm whose trials all errored gets a 0.0 per trial from the
failure-inclusive rule, so its mean is 0.0: a real number, the lowest
available, and the only number in the series if its neighbours' trials
went unscored and reported None. Ranks skip None and order by value, so
the arm that failed every trial came FIRST. The same shape arrives from a
scoring pass run with no judge model, which records a row under `judge`
with a NULL judge_model and a NULL score. Both are coverage material,
which is a thing to read, not a thing to rank.

**A human-ranked report states its blind composition**, as `n blind of n
ratings`, beside the metric. A ranking built on ratings made blind is a
different claim from one built on ratings made while the rater could see
which model wrote which answer, and the second is much the weaker. The
flags are already on the rows; without the line a reader would have to join
tables to learn which report they are holding.

**Pass rate where a threshold was declared, score mean otherwise**, and
the rate never appears without its coverage: `0.80 (4/5 of 12 eligible)`
says passed, usable verdicts, and eligible trials. A pass rate over three
verdicts out of forty eligible is not a pass rate anybody should act on,
and the coverage figure is the only thing that says so.

Thresholds live in the dataset file, so **the report says where it got
the eligible population** in `thresholds_source`:

- `dataset_file` when you passed `dataset_path`. The denominator is
  exact: the file names every task that declared a cutoff, including the
  ones nothing ever scored.
- `score_rows` otherwise. `passed` is written from the task's own
  threshold at scoring time and `judged_pass` returns null unless the
  author declared one, so **a judge row with a non-null `passed` is
  itself a record that a threshold existed**. The rate that comes out is
  exact, because those verdicts were computed against the real cutoff.
  The eligible count is a **floor**: a declared task whose trials were
  never scored leaves no row to witness it. Supply the file for the full
  denominator.

Only judge rows witness. A deterministic scorer writes `passed`
unconditionally, since it is the score restated rather than a cutoff
anyone chose, and the loader permits `pass_threshold` only on `judge`.
Counting those rows would make the same experiment report a different
eligible population depending on whether a path was passed, and two
answers from one dataset is the failure this layer exists to prevent.

Losing the file used to delete the measurement outright: every verdict
sat in the database, and the report published no rate at all because the
denominator was never looked for. Refusing to answer is only honest when
the answer is unknown, and here most of it was not.

**The scoring-failure rate is its own reported number.** It is also the
measurement the `response_format` decision in `bench/models.py` says to
revisit against, so the report computes it whether or not anything reads
it that day.

**Intervals are 95% percentile bootstrap over TASK clusters**, seeded and
recorded. Every repeat of a sampled task is kept together, because
repeats of one task share that task's difficulty and are correlated;
resampling trials independently would treat forty correlated trials as
forty independent ones and narrow the interval below what the data
supports. An overconfident interval is worse than none: it is false
precision wearing the costume of rigor. With `repeats: 1` the two schemes
coincide exactly. One task gets no interval at all, because a single
cluster cannot be resampled into anything but itself.

**Ties share a rank**, in the report and in the live race. Two models
that measured identically are tied, and ordering them anyway would show a
difference that is not in the data.

The view is plain tables, deliberately. A bar chart of four means invites
the eye to read a difference the intervals do not support, which is the
thing this layer exists to stop. The estimand leads the banner, and the
`self-judged` and `blind` flags are surfaced per scorer row rather than
folded into the numbers.

The report view has its own **dataset file** box, following the run-start
precedent: the path is sent with the read, its digest is checked against
the one recorded at creation, and a mismatch is refused in the server's
own words rather than as a status code, naming both digests so the reader
knows which of the two was wrong. The box stays on screen through the
refusal, because a path you cannot see is a path you cannot correct.
Without a file the report degrades to score means and says so. The path
is remembered in a variable for as long as the tab is open and nowhere
else: it is a fact about the operator's filesystem, not about the
experiment, which is why the row records the file's digest instead.

## Export

```sh
curl -O localhost:8000/experiments/1/export.jsonl

# with the dataset, so the artifact carries the pass thresholds too
curl -O "localhost:8000/experiments/1/export.jsonl?dataset_path=bench-datasets/arithmetic.jsonl"
```

JSONL. Line one is the manifest: which dataset by digest, which build,
which catalog, which estimand, which seeds, what the experiment declared
itself to be. Every following line is one trial with its full provenance,
including the payload sent, the response, the timings, the token counts,
both cost figures, the serving provider, and every score row attached.
The last line is a sha256 over the preceding bytes, so a citation can
name the artifact it cites and anyone can check the name.

**The digest seals a moment, not an interval.** Every row is read inside
one explicit read transaction, opened before the first query and closed
after the last, so a write committed anywhere inside that window appears
in none of the lines rather than in the later ones. An artifact that
straddled two states of the database would verify its own digest
perfectly while describing a moment that never existed, which is the
worst combination available.

This needs a real `BEGIN`. A `with conn:` block does not provide one: the
sqlite3 connection context manager commits or rolls back at exit and
never begins, and the module starts a transaction implicitly before a
write and not before a read, so a block of pure `SELECT`s runs each of
them in its own autocommit transaction. It reads as the guarantee and
gives none of it. The writer this defends against is another
**connection**, not another task: the store's synchronous contract
already rules out an interleaving on the export's own connection, but
`python -m bench.reconcile --apply` against a live bench is a second
connection by design, and is the reason `connect()` turns WAL on.

**Two exports of the same experiment are byte-identical.** Line order is
task, then repeat, then position; key order within a line is sorted.
Both rules are needed, and neither alone is enough: an artifact whose
digest changes between two honest exports of the same data cannot be
cited, because the citation could never be checked. Keys are sorted
rather than listed in a stated order, because a stated order is a list
someone must remember to update and the day they forget it fails
silently in exactly this way.

**The export is self-sufficient**, and that is tested rather than
claimed. The round-trip test parses every line, verifies the digest, and
rebuilds a report from the export alone by feeding its rows back through
the same pure function the served report uses. The experiment it does
this on is chosen to be the one a two-axes mistake would corrupt: it has
an errored trial (axis one) and completed trials nobody scored (axis
two), so a mean that confused them would come out different. An export
that flattened either axis would still match on an experiment where
everything succeeded, which is why that is not the experiment used.

**The artifact labels its own sufficiency.** `dataset_path` on the export
takes the same terms as start, score and report: the digest is checked
against the one recorded at creation, and a mismatch is refused before a
single byte is streamed, because a file half-written against the wrong
dataset is worse than none. Supplying it embeds the minimal threshold
slice in the manifest, task id to scorer kind and cutoff, declared tasks
only. Prompts, references and rubrics stay out: no published number
needs them, and the export already carries every prompt actually sent.

The manifest always carries `thresholds_included`, so a reader holding an
export with no thresholds can tell a dataset that declared none from an
export nobody handed the file to. Those license different claims about
the pass rate inside: with the slice the artifact re-derives the exact
eligible denominator, without it the same floor the pathless report
publishes. Both modes are byte-identical across two exports; the two
modes are deliberately different artifacts and do not share a digest.

Before this, the round trip could reproduce a model's mean exactly and
was quietly unable to reproduce the coverage figure printed beside it,
because eligibility lived in the file and nowhere else. The old test did
not catch it: its fixture declared no thresholds, so the number it could
not reproduce was never asked for.

Each trial line carries the derived `outcome` **and** the fields it was
derived from. That is not redundancy: a reader on a future version of
these rules can see both what this bench concluded and what it concluded
it from, and can tell a rule change from a data change.

Each trial line also carries `attachments` and `attachments_mode`: the
digests of the documents that trial ran over, and how they reached the
models. Digests and never content, so an export over a confidential
document is not a second copy of it. The digests are recoverable from
`request_json` too, since the placeholder there names them, but stating
them on the line means a reader never has to parse a record format back
into a data format. The manifest carries `attachments_referenced` for the
same reason it carries `thresholds_included`: an artifact should say at
line one whether it is complete on its own, and one that cites documents
is complete only alongside the `bench.db` that produced it. The flag is
computed from the trial lines actually emitted, so a cell that declared a
document and recorded no result does not make the export claim a
reference no line in it carries.

The response streams and carries `private, no-store` like every other
dynamic body. It holds every prompt and every answer in the experiment,
which makes it the most sensitive thing the bench will hand you.

## Blind human rating

Press **rate blind** on a comparison in the History list. The server opens
a session, shuffles the answers, and hands back a view carrying answers
and neutral letters with nothing else attached: no model, no provider, no
cost, no timing. You rate 1 to 5, and the identities arrive only after
every card is rated and the ratings are saved.

**One path, and it starts at the server.** There used to be a second one,
reached from a replayed comparison, that fetched the identified cards and
then hid the identities with a style rule while posting `blind: true`.
Both halves of that were wrong: hidden in a browser is a rule anyone can
undo plus a frame the rater may already have seen, and a page that painted
the answer key is the one witness that cannot testify about its own
blindness. It is gone rather than deprecated, because a path nobody should
use is a path nobody should be able to reach. The **Rate blind** button on
a replayed comparison remains and now opens the same server session,
replacing the view with a fresh anonymized one; its tooltip says what it
cannot fix, which is that you have already seen those answers identified.

Cost and timing are absent along with the name, because a rater who knows
one answer cost ten times another can often guess which model it was, and
a guess is not a blind. The reveal happens after the save, not before: a
rater who saw the identities and then changed their mind would be
producing a sighted rating the record calls blind. If the save fails,
nothing is revealed and you can retry. The revealed view shows the model
names and not the metrics, because the blind view renders a minimal card
rather than reusing the normal renderer; the numbers are one ordinary
replay away.

Ratings persist as `scores` rows with `scorer = "human"` and `blind = 1`,
the normalized score in `score` and the point you actually clicked in
`detail` ("4 of 5, shown as B"). Labels run A to Z and then AA, AB, like
spreadsheet columns, so a comparison wider than 26 answers is still
labelled in letters rather than in the punctuation that follows Z in
ASCII. The label is recorded so the
rating-to-model mapping is auditable after the reveal. Ratings entered on
a normal replay, without blind mode, persist with `blind = 0`: a sighted
rating is still a rating, and what would be wrong is recording it as
blind. No pass verdict is derived, for the same reason a judge without a
declared threshold has none.

**What blinding cannot do.** The bench hides everything it renders about
a card. It cannot hide what the answer says, and a model that writes "as
an AI assistant made by X" has identified itself. That is a limit of the
method rather than of the implementation, and it is worth knowing before
reading too much into a blind rating on prompts that invite
self-description.

The reveal is one way per replay. Re-blinding after it would produce a
rating the record calls blind that was made by someone who had already
seen the answer, which is worse than no blind rating at all.

**Re-scoring appends.** Scoring is idempotent per (result, scorer, judge
model) in the sense that re-running is safe, not in the sense that it is
a no-op: the new row lands beside the old one and reports read the latest
per key, ordered by timestamp and then by row id so two rows written in
the same second cannot make a report nondeterministic. The older rows
stay, because "the judge said 0.5 last week and 1.0 today" is exactly
what an audit needs to see.

## Usage

Open http://localhost:8000 in a browser. Type a prompt, check the
models to compare, hit Run. Each column fills in as its model
responds. The lineup is managed with the picker (see Daily use); the
four-model default seed for a fresh browser is `DEFAULT_LINEUP` at
the top of `static/controls.js`.

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

`POST /compare` is the supported scripting surface, and that is a
commitment rather than an accident of it existing. It carries every
control the browser does, under the same `params` object, with the same
validation and the same bounds:

```sh
curl -X POST localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello in five words.",
       "models": ["deepseek/deepseek-chat"],
       "params": {"system": "Be terse.", "temperature": 0, "seed": 7,
                  "routing": "price"}}'
```

Parity with the streaming endpoint is structural, not maintained by hand:
both request models reference one `ExperimentParams` class, so a field, a
bound or a validator cannot drift between them. Send `group_id` alongside
`params` and the one-experiment-per-group check applies to scripted runs
exactly as it does to the browser's.

### Grouping scripted runs

`POST /groups` declares what a comparison is, before any model is called.
`budget` is **required**; `prompt`, `models` and `params` are optional but
each one you send becomes enforceable:

```sh
curl -X POST localhost:8000/groups \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello in five words.",
       "models": ["deepseek/deepseek-chat", "z-ai/glm-4.6"],
       "budget": "standard"}'
```

Every run that then joins the group with `group_id` is checked against that
declaration at entry, before any upstream call, and refused with a 409 that
names the conflict:

- a model not in the declared lineup;
- a streamed member claiming a `position` the lineup gives to another model;
- a batch whose `models` array is not the declared lineup **in order**;
- a run whose `budget` tier differs from the group's;
- and, as before, a different prompt or a different controls set.

Two consequences worth knowing. A batch joining a group must send the whole
declared lineup, so adding one model to an existing comparison goes through
`/compare/stream`, one request per model, which is what the browser does.
And the budget check compares the tier you asked for, not the token count it
resolves to: two models with different published caps legitimately produce
different effective token numbers inside one `extended` comparison, and both
are accepted.

Groups created before this existed declare nothing, and nothing is enforced
against them. That is a different rule from the controls, deliberately: a
group with no recorded controls is a group that set none, and it refuses a
controls-carrying member, while a group with no recorded lineup or budget is
one that never said, so it accepts anything. The comments at both check
sites state the contrast.

`GET /groups/{id}` returns the declaration beside the runs, so an
experiment can be read back as it was declared and not only as it turned
out. A declared model with no recorded run appears in a replay as a muted
placeholder rather than vanishing, so an incomplete comparison looks
incomplete instead of smaller.

The browser UI streams responses token by token via
`POST /compare/stream` with `{"prompt": ..., "model": ...}` (one
model per request) plus optional `prompt_id`, `group_id`, `budget` and
`params`. The
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
  requests land as a single history entry. A group holds one prompt
  across its runs (the first member's), enforced at run entry: a
  `/compare` or `/compare/stream` whose `group_id` names a group already
  holding a different prompt is a 409 before any upstream call, never a
  post-spend rejection
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
the runtime pins too. That file and `requirements.txt` are compiled,
hashed outputs, not the edit surface: the direct dependencies live in
`requirements.in` and `requirements-dev.in`, and after editing one of
those, recompile with pip-tools (from the repo root, under the 3.11
floor):

```sh
.venv/bin/pip-compile --allow-unsafe --generate-hashes \
  --output-file=requirements.txt requirements.in
.venv/bin/pip-compile --allow-unsafe --generate-hashes \
  --output-file=requirements-dev.txt requirements-dev.in
```

The install command is unchanged (the compiled files keep their names);
CI installs with `--require-hashes`, which is what turns the hashes into
enforcement, and a dedicated `audit` job runs `pip-audit` over the pinned
closure. There are two test suites:

```sh
.venv/bin/pip install -r requirements-dev.txt

# unit suite: the every-edit loop, fast and browser-free
.venv/bin/pytest

# browser suite: one-time setup, then the every-merge gate
.venv/bin/playwright install chromium
.venv/bin/pytest -m browser

# pure frontend helpers: no build step, no npm install
node --test "tests/js/**/*.test.js"

# front-end format and lint (pinned Biome; a local npx fallback)
npx --yes @biomejs/biome@1.9.4 check static/ tests/js/
```

No network access needed for the suites; unit tests mock OpenRouter
with respx, the browser suite boots the real app under uvicorn in
headless Chromium against a stub OpenRouter it starts itself
(`tests/browser/`), and the node suite requires `static/lib.js`
directly through its CommonJS guard to check the diff engine and the
formatting helpers. Browser tests are deselected from a plain
`pytest` run on purpose. Biome formats and lints the front-end module
scripts against the committed `biome.jsonc`; the pinned `npx` above is
the local fallback, and CI runs the same version from a checksum-verified
binary (no package.json, no npm install). CI enforces all of it: a lint
job (ruff and mypy), the unit matrix across Python 3.11 through 3.14, the
node job, the Biome job, and the browser job, so neither a backend nor a
frontend change can merge without proving the critical path still works.

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
- Failed history load: run a comparison, then with devtools throttled
  to offline (or the server stopped) open History and click the entry.
  The cards clear to a loading then a failure state that stands alone;
  no card from the earlier run stays visible under the failure banner.
- Double Enter on save: open the save name row, type a name, and press
  Enter (or click OK) twice fast. Exactly one prompt is created with no
  lingering "already exists" error; saving under a duplicate name shows
  the conflict, and renaming and saving again clears it. Editing the
  prompt text while a save is in flight leaves the saved-prompt link
  cleared rather than claiming a match that is no longer on screen.
- Watch a slow model paint token by token next to an already
  finished fast one; the ttft metric should be visibly smaller than
  the total metric on streamed columns, and the race strip row
  should flip from shimmer to a ranked bar at the first token.
- Kill wifi (or the server) mid-stream: the streaming column must
  enter the error state with its partial text retained above the
  error message, not hang or go blank.
- Stop mid-stream: run an extended prompt on a slow model and hit Stop
  while it is thinking. The card keeps its partial text and flips to the
  muted stopped state (not error), the race row stops ticking, Run
  re-enables, and a fresh Run works. Open History: the stopped run is
  there as an aborted record. Run six or more models and hit Stop while
  one is still queued: that card reads stopped with no text.
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
- CSP: with the devtools Console open, walk the whole path once (load,
  run, stop, save, replay from History). The console stays clean; no
  Content-Security-Policy violation is reported and nothing renders
  unstyled or fails to fetch.
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
