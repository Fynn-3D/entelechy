# Entelechy L7 AI Meta-Runtime Bridge

Model-agnostic, zero-dependency L7 capability compiler built strictly over the
minimal Entelechy Creation Protocol (`status`, `graph`, `document`, `help`,
`read`, `catalog`, `trial`, `trace`).

## Architecture

This bridge keeps all model orchestration, visual review, and diagnosis
heuristics outside the Host and Runtime layers:

```text
Intent
  -> [1. Observe]      status (last_frame facts) + graph + local_files + document + read(summary/png)
  -> [2. Discover]     catalog(query, input, output, mode) + inspect(local .ent docs, live help)
  -> [3. Plan]         pluggable LLM returns `search`, `inspect`, `propose`, or `accept` JSON
  -> [4. Shadow Gate]  trial(...) runs in an isolated in-memory Kept store against compile,
                       runtime, finite (auto-trace), frame-budget, and zero-download gates
  -> [5. Visual Gate]  optional `--visual-review`: shadow PNGs return to the model BEFORE commit;
                       the model either `accept`s or revises with another `propose`
  -> [6. Promote]      atomic write & hot-swap into the live session only when approved;
                       failed or unaccepted runs leave disk and live state untouched
```

## Usage

Requires Node 20+ (built-in `fetch`, zero external packages).

Run one evolution against a scene file (`entelechy serve` spawned over stdio):

```bash
GEMINI_API_KEY=... node bridges/ai/cli.mjs \
  --scene examples/life/scene.esn \
  --intent "Add a subtle vignette while keeping frame time under 16ms" \
  --provider gemini \
  --visual-review \
  --max-frame-ms 16
```

Stay connected to a live `entelechy-player --serve --listen 127.0.0.1:7788` window
for continuous interactive co-creation (`--chat`):

```bash
GEMINI_API_KEY=... node bridges/ai/cli.mjs \
  --connect 127.0.0.1:7788 \
  --chat \
  --provider gemini \
  --visual-review
```

Supported `--provider` values: `openai` (any OpenAI-compatible API via
`OPENAI_BASE_URL`), `gemini`, `anthropic`, and `ollama`.

## Verification

Run the offline deterministic end-to-end test suite (no network or API keys):

```bash
node bridges/ai/test.mjs
```
