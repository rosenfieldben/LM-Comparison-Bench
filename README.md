# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and compare the
results side by side in the browser. Prompts can be saved as a
reusable library and every run lands in SQLite history for later
replay. No streaming, no cost display yet.

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
directory.

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

Other endpoints:

- `GET /prompts` lists saved prompts
- `POST /prompts` with `{"name": ..., "text": ...}` saves one; 409 on
  a duplicate name
- `DELETE /prompts/{id}` removes a prompt; runs that used it keep
  their text, only the link is cleared
- `GET /runs` lists run history, most recent first, prompt text
  truncated to 80 chars
- `GET /runs/{id}` returns a full run with results

## Tests

```sh
.venv/bin/pytest
```

No network access needed; all OpenRouter calls are mocked.

The page has no JS test harness. Verify it by eyeball after UI
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
