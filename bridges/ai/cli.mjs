#!/usr/bin/env node
//! CLI driver for the Entelechy L7 AI Meta-Runtime.
//!
//! Examples:
//!   node bridges/ai/cli.mjs --scene examples/life/scene.esn \
//!     --intent "Add a warm amber palette" --provider gemini --visual-review
//!   node bridges/ai/cli.mjs --connect 127.0.0.1:7788 \
//!     --chat --provider gemini --visual-review --max-frame-ms 16

import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline/promises";
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
    chat: false,
    provider: "openai",
    model: null,
    baseUrl: null,
    maxAttempts: 4,
    steps: 2,
    maxFrameMs: null,
    forbidDownload: false,
    includePng: false,
    visualReview: false,
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
      case "--chat": opts.chat = true; break;
      case "--provider": opts.provider = next(); break;
      case "--model": opts.model = next(); break;
      case "--base-url": opts.baseUrl = next(); break;
      case "--attempts": opts.maxAttempts = Number(next()); break;
      case "--steps": opts.steps = Number(next()); break;
      case "--max-frame-ms": opts.maxFrameMs = Number(next()); break;
      case "--forbid-download": opts.forbidDownload = true; break;
      case "--include-png": opts.includePng = true; break;
      case "--visual-review":
        opts.visualReview = true;
        opts.includePng = true;
        break;
      case "--dry-run": opts.dryRun = true; break;
      case "--json": opts.json = true; break;
      case "-h":
      case "--help":
        console.log(`Usage: node bridges/ai/cli.mjs (--scene <scene.esn> | --connect <host:port>) (--intent "<goal>" | --chat) [options]

Options:
  --intent "<goal>"                             Run one closed-loop evolution and exit
  --chat                                        Stay connected for continuous live co-creation
  --provider <openai|gemini|anthropic|ollama>   Model provider (default: openai)
  --model <id>                                  Model identifier override
  --base-url <url>                              Custom API endpoint
  --backend <cpu|jit|gpu>                       Backend for spawned \`entelechy serve\`
  --attempts <n>                                Max shadow trial attempts per turn (default: 4)
  --steps <n>                                   Shadow evaluation frames per trial (default: 2)
  --max-frame-ms <ms>                           Enforce frame time budget gate
  --forbid-download                             Reject changes causing GPU->CPU mid-frame downloads
  --include-png                                 Attach rendered PNGs for multimodal feedback
  --visual-review                               Inspect shadow PNGs and require \`accept\` before committing
  --dry-run                                     Verify in shadow without committing to disk
  --json                                        Print final result as JSON (single-shot mode)`);
        process.exit(0);
      default:
        throw new Error(`unknown argument "${a}"`);
    }
  }
  if (!opts.scene && !opts.connect) {
    throw new Error("supply either --scene <scene.esn> or --connect <host:port>");
  }
  if (!opts.intent && !opts.chat) {
    throw new Error("supply either --intent \"<goal>\" or --chat");
  }
  return opts;
}

function makeEventLogger(opts) {
  return (ev) => {
    if (opts.json) return;
    const detail =
      ev.stage === "gate_failed"
        ? ` failed=[${ev.failedGates.join(", ")}]`
        : ev.stage === "inspect"
          ? ` paths=[${(ev.paths || []).join(",")}] help=[${(ev.help || []).join(",")}]`
          : ev.stage === "visual_review"
            ? ` frame=${ev.frame_ms?.toFixed(2)}ms images=${ev.images}`
            : ev.stage === "promoted" || ev.stage === "verified"
              ? ` frame=${ev.frame_ms?.toFixed(2)}ms`
              : ev.summary
                ? ` "${ev.summary}"`
                : "";
    console.error(`[l7:${ev.stage}]${ev.attempt ? ` (${ev.attempt}/${opts.maxAttempts})` : ""}${detail}`);
  };
}

function buildGates(opts) {
  return {
    steps: opts.steps,
    forbid_nonfinite: true,
    forbid_download: opts.forbidDownload,
    include_png: opts.includePng,
    ...(opts.maxFrameMs != null ? { max_frame_ms: opts.maxFrameMs } : {}),
  };
}

async function runChat(client, provider, opts) {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  console.log(`Connected (${provider.name}). Type an intent to evolve live, or /status, /files, /reset, /quit.`);
  try {
    while (true) {
      let line;
      try {
        line = await rl.question("entelechy> ");
      } catch {
        break;
      }
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (trimmed === "/quit" || trimmed === "/exit") break;
      if (trimmed === "/status") {
        const s = await client.status();
        const lf = s.last_frame;
        console.log(
          `backend=${s.backend} rev=${s.revision} frame=${s.frame} loaded=${s.loaded} failing=${s.failing ?? "none"} ` +
            `frame_ms=${lf ? lf.frame_ms.toFixed(2) : "n/a"} down=${lf?.downloaded_bytes ?? 0}B`,
        );
        continue;
      }
      if (trimmed === "/files") {
        const obs = await client.observe({ measure: false, checkOutputs: false });
        console.log(obs.local_files.join("\n") || "(no local files)");
        continue;
      }
      if (trimmed === "/reset") {
        await client.request("reset");
        console.log("State reset.");
        continue;
      }
      try {
        const result = await evolve(client, {
          intent: trimmed,
          provider,
          maxAttempts: opts.maxAttempts,
          visualReview: opts.visualReview || opts.includePng,
          dryRun: opts.dryRun,
          gates: buildGates(opts),
          onEvent: makeEventLogger(opts),
        });
        if (result.ok) {
          console.log(
            `${result.promoted ? "Promoted" : "Verified"} in ${result.attempts} attempt(s): ${result.summary}`,
          );
        } else {
          console.log(`Unchanged after ${result.attempts} attempt(s); live world untouched.`);
        }
      } catch (err) {
        console.error(`error: ${err.message}`);
      }
    }
  } finally {
    rl.close();
  }
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
    if (opts.chat) {
      await runChat(client, provider, opts);
      return;
    }

    const result = await evolve(client, {
      intent: opts.intent,
      provider,
      maxAttempts: opts.maxAttempts,
      visualReview: opts.visualReview || opts.includePng,
      dryRun: opts.dryRun,
      gates: buildGates(opts),
      onEvent: makeEventLogger(opts),
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
