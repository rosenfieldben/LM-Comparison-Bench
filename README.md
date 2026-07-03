# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and compare the
results side by side in the browser. No persistence, no streaming,
no cost display yet.

## Setup

Requires Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/uvicorn bench.main:app
```

The app refuses to boot if `OPENROUTER_API_KEY` is unset.

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
the other models in the run.

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
