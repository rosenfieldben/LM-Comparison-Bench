# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and compare the
results side by side in the browser. Responses stream in token by
token, and every result card carries latency, time to first token,
token counts and cost. Prompts can be saved as a reusable library
and every run lands in SQLite history for later replay.

This is a comparison workbench: it puts models side by side under
identical conditions and records what happened, with enough provenance
to say later what an old run actually was. The evaluation layer on top
of it, datasets, repeated trials and scoring, is in progress and not
here yet, so treat a comparison as evidence to read rather than a
benchmark result to cite.

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
no body buffering, so streaming is untouched. To serve the bench beyond
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
other copy.

Each row carries provenance, so a run stays interpretable after the
code, the prices, or the lineup have moved on. A group records the
prompt and the ordered lineup as declared, before any model is called,
which makes the group row the experiment record and fixes the group's
prompt ahead of its first member. Each run records the commit that
produced it (`app_sha`, best effort: None when git is unavailable), the
timestamp of the price catalog it was costed against
(`catalog_snapshot_at`, None on an offline boot), and the data-handling
policy its payloads declared (`data_policy`). Each result records the
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

## Experiment controls

By default the bench sends one user message and lets every provider apply
its own sampling defaults, which means a comparison varies whatever the
providers feel like varying. Six controls let one comparison hold that
constant across its models:

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
