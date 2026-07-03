# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and compare the
results side by side in the browser. Prompts can be saved as a
reusable library and every run lands in SQLite history for later
replay. No streaming, no cost display yet.

## Daily use

```sh
cd LM-Comparison-Bench
source .venv/bin/activate
export OPENROUTER_API_KEY=sk-or-...
uvicorn bench.main:app
```

Then open http://localhost:8000. The model list is the MODELS const
at the top of the script block in `static/index.html`; IDs must match
https://openrouter.ai/models exactly.

## Setup

Requires Python 3.12.

```sh
python3.12 -m venv .venv
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
session.

## Usage

Open http://localhost:8000 in a browser. Type a prompt, check the
models to compare, hit Run. Each column fills in as its model
responds. The model list is a hand-edited const at the top of the
script block in `static/index.html`.

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
model per request) plus optional `prompt_id` and `group_id`. The
response is SSE-formatted (`data: {...}` lines): `delta` events carry
text chunks, then one `done` event carries the full result and its
`run_id`. Streamed results include `ttft_ms` (time to first token),
which is the metric streaming exists to reveal; `latency_ms` alone
hides it. One deliberate contract amendment: a result can carry BOTH
partial `response_text` AND an `error` when a stream dies partway.
The partial text renders above the error, live and on replay.

Other endpoints:

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
  comparisons or `{type: "run", ...}` for legacy ungrouped rows
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

```sh
.venv/bin/pytest
```

No network access needed; all OpenRouter calls are mocked.

The page has no JS test harness. During frontend eyeball
verification, run `uvicorn bench.main:app --reload` so index.html
edits are picked up without restarts. Verify by eyeball after UI
changes:

- Run with 2 models checked: both columns show a loading state, then
  fill in independently, fastest first.
- Run with an intentionally bad model string in the MODELS const:
  that column shows the error state (red tint), others unaffected.
- Run with a prompt that produces multi-line output (e.g. "write a
  haiku"): line breaks survive in the response column.
- Save a prompt, reload the page, pick it from the dropdown, replay
  it against one model. Open History, click the old run, and confirm
  it renders identically to a live run (plus the historical banner).
- Watch a slow model paint token by token next to an already
  finished fast one; the ttft badge should be visibly smaller than
  the latency badge on streamed columns.
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
