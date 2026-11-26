# Entelechy L7 AI Meta-Runtime Bridge

Model-agnostic, zero-dependency L7 capability compiler built strictly over the
Entelechy Creation Protocol (`catalog`, `diagnose`, `trial`, `document`, `graph`,
`trace`).

## Architecture

This bridge keeps all model orchestration outside the Host and Runtime layers:

```text
Intent
  -> [1. Observe]    status + graph + document + diagnose(measure: true)
  -> [2. Search]     catalog(query, input, output, mode, package)
  -> [3. Plan]       pluggable LLM provider returns `search` or `propose` JSON
  -> [4. Shadow]     trial(..., commit: true) runs in an isolated in-memory
                     Kept store against compile, runtime, finite (auto-trace),
                     frame-budget, and zero-download gates
  -> [5. Promote]    atomic write & hot-swap into the live session only when
                     every gate passes; failed trials leave disk and live state
                     untouched
```

## Usage

Requires Node 20+ (built-in `fetch`, zero external packages).

Run against a scene file (`entelechy serve` spawned over stdio):

```bash
GEMINI_API_KEY=... node bridges/ai/cli.mjs \
  --scene examples/life/scene.esn \
  --intent "Add a subtle vignette while keeping frame time under 16ms" \
  --provider gemini \
  --max-frame-ms 16
```

Drive a running `entelechy-player --serve --listen 127.0.0.1:7788` window live:

```bash
node bridges/ai/cli.mjs \
  --connect 127.0.0.1:7788 \
  --intent "Add gentle bloom" \
  --provider ollama \
  --model qwen2.5-coder:14b
```

Supported `--provider` values: `openai` (any OpenAI-compatible API via
`OPENAI_BASE_URL`), `gemini`, `anthropic`, and `ollama`.

## Verification

Run the offline deterministic end-to-end test suite (no network or API keys):

```bash
node bridges/ai/test.mjs
```
