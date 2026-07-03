# LM Comparison Bench

Send one prompt to multiple models via OpenRouter and get structured
results back for side-by-side evaluation. Backend only for now: no
persistence, no streaming, no frontend.

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
