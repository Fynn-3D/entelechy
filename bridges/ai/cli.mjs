#!/usr/bin/env node
//! CLI driver for the Entelechy L7 AI Meta-Runtime.
//!
//! Examples:
//!   node bridges/ai/cli.mjs --scene examples/life/scene.esn \
//!     --intent "Add a warm amber palette" --provider gemini
//!   node bridges/ai/cli.mjs --connect 127.0.0.1:7788 \
//!     --intent "Keep frame time under 16ms" --provider ollama --max-frame-ms 16

import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { EntelechyClient } from "./client.mjs";
import { createProvider } from "./providers.mjs";
import { evolve } from "./runtime.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../..");

function defaultBinary() {
  for (const rel of ["target/release/entelechy", "target/debug/entelechy"]) {
    const candidate = join(REPO_ROOT, rel);
    if (existsSync(candidate)) return candidate;
  }
  return "entelechy";
}

function parseArgs(argv) {
  const opts = {
    binary: defaultBinary(),
    scene: null,
    connect: null,
    backend: null,
    intent: null,
    provider: "openai",
    model: null,
    baseUrl: null,
    maxAttempts: 4,
    steps: 2,
    maxFrameMs: null,
    forbidDownload: false,
    includePng: false,
    dryRun: false,
    json: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    switch (a) {
      case "--bin": opts.binary = next(); break;
      case "--scene": opts.scene = next(); break;
      case "--connect": opts.connect = next(); break;
      case "--backend": opts.backend = next(); break;
      case "--intent": opts.intent = next(); break;
      case "--provider": opts.provider = next(); break;
      case "--model": opts.model = next(); break;
      case "--base-url": opts.baseUrl = next(); break;
      case "--attempts": opts.maxAttempts = Number(next()); break;
      case "--steps": opts.steps = Number(next()); break;
      case "--max-frame-ms": opts.maxFrameMs = Number(next()); break;
      case "--forbid-download": opts.forbidDownload = true; break;
      case "--include-png": opts.includePng = true; break;
      case "--dry-run": opts.dryRun = true; break;
      case "--json": opts.json = true; break;
      case "-h":
      case "--help":
        console.log(`Usage: node bridges/ai/cli.mjs (--scene <scene.esn> | --connect <host:port>) --intent "<goal>" [options]

Options:
  --provider <openai|gemini|anthropic|ollama>   Model provider (default: openai)
  --model <id>                                  Model identifier override
  --base-url <url>                              Custom API endpoint
  --backend <cpu|jit|gpu>                       Backend for spawned \`entelechy serve\`
  --attempts <n>                                Max shadow trial attempts (default: 4)
  --steps <n>                                   Shadow evaluation frames per trial (default: 2)
  --max-frame-ms <ms>                           Enforce frame time budget gate
  --forbid-download                             Reject changes causing GPU->CPU mid-frame downloads
  --include-png                                 Attach rendered PNGs for multimodal feedback
  --dry-run                                     Verify in shadow without committing to disk
  --json                                        Print final result as JSON`);
        process.exit(0);
      default:
        throw new Error(`unknown argument "${a}"`);
    }
  }
  if (!opts.scene && !opts.connect) {
    throw new Error("supply either --scene <scene.esn> or --connect <host:port>");
  }
  if (!opts.intent) {
    throw new Error("supply --intent \"<goal>\"");
  }
  return opts;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const provider = createProvider({
    provider: opts.provider,
    model: opts.model,
    baseUrl: opts.baseUrl,
  });

  const client = opts.connect
    ? await EntelechyClient.connect(opts.connect)
    : EntelechyClient.spawn(opts.binary, opts.scene, {
        args: opts.backend ? ["--backend", opts.backend] : [],
      });

  try {
    const result = await evolve(client, {
      intent: opts.intent,
      provider,
      maxAttempts: opts.maxAttempts,
      dryRun: opts.dryRun,
      gates: {
        steps: opts.steps,
        forbid_nonfinite: true,
        forbid_download: opts.forbidDownload,
        include_png: opts.includePng,
        ...(opts.maxFrameMs != null ? { max_frame_ms: opts.maxFrameMs } : {}),
      },
      onEvent(ev) {
        if (opts.json) return;
        const detail =
          ev.stage === "gate_failed"
            ? ` failed=[${ev.failedGates.join(", ")}]`
            : ev.stage === "promoted" || ev.stage === "verified"
              ? ` frame=${ev.frame_ms?.toFixed(2)}ms`
              : ev.summary
                ? ` "${ev.summary}"`
                : "";
        console.error(`[l7:${ev.stage}]${ev.attempt ? ` (${ev.attempt}/${opts.maxAttempts})` : ""}${detail}`);
      },
    });
    if (opts.json) {
      console.log(JSON.stringify(result, null, 2));
    } else if (result.ok) {
      console.log(
        `${result.promoted ? "Promoted" : "Verified"} in ${result.attempts} attempt(s): ${result.summary}`,
      );
    } else {
      console.error(`Failed after ${result.attempts} attempt(s); live scene and disk untouched.`);
      process.exitCode = 1;
    }
  } finally {
    await client.close();
  }
}

main().catch((err) => {
  console.error(`error: ${err.message}`);
  process.exit(1);
});
