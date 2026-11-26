//! End-to-end offline verification of the L7 AI Meta-Runtime (`bridges/ai`)
//! against the real `entelechy serve` binary (no network, no API keys).
//!
//! Covers:
//!   1. Observe -> dynamic `catalog` search -> compile error auto-repair ->
//!      `finite` gate NaN rejection + automatic element `trace` -> repaired
//!      package composition (`image.grayscale@1`) -> atomic promotion (`commit: true`).
//!   2. Multi-document evolution: adding a new `.ent` Kernel file and wiring it
//!      in `scene.esn` inside an in-memory shadow trial before atomic promotion.
//!   3. Live TCP socket co-creation (`entelechy serve --listen 127.0.0.1:0`).
//!   4. Zero-pollution guarantee when all retry attempts fail verification gates.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { EntelechyClient } from "./client.mjs";
import { evolve } from "./runtime.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../..");

function findBinary() {
  if (process.env.ENTELECHY_BIN && existsSync(process.env.ENTELECHY_BIN)) {
    return process.env.ENTELECHY_BIN;
  }
  for (const rel of ["target/debug/entelechy", "target/release/entelechy"]) {
    const candidate = join(REPO_ROOT, rel);
    if (existsSync(candidate)) return candidate;
  }
  throw new Error("entelechy binary not found; build with `cargo build -p entelechy` or set ENTELECHY_BIN");
}

const BASE_SCENE = `node gradient: kernel {
    requires { field.write; }
    output result: Image;
    domain = vec2(8, 8);
    compute {
        return vec4(coord().x / 8.0, coord().y / 8.0, 0.5, 1.0);
    }
}
output result <- gradient.result;
`;

function tempScene(name) {
  const dir = mkdtempSync(join(tmpdir(), `entelechy-ai-${name}-`));
  const scenePath = join(dir, "scene.esn");
  writeFileSync(scenePath, BASE_SCENE, "utf8");
  return { dir, scenePath };
}

function scriptedProvider(steps) {
  let idx = 0;
  return {
    name: "mock-scripted",
    async complete({ messages }) {
      const fn = steps[idx++];
      assert.ok(fn, `unexpected extra provider call #${idx}`);
      const lastMsg = messages[messages.length - 1];
      const lastUser = JSON.parse(lastMsg.content);
      return fn(lastUser, idx, lastMsg.images || []);
    },
  };
}

async function testSearchRepairGateAndPromote(bin, env) {
  const { dir, scenePath } = tempScene("repair");
  const client = EntelechyClient.spawn(bin, scenePath, { args: ["--backend", "cpu"], env });
  try {
    await client.evaluate({ time: 0.0 });

    const provider = scriptedProvider([
      // Turn 1: Observe world & search catalog for an Image desaturation capability.
      (ctx) => {
        assert.equal(ctx.status.loaded, true);
        assert.equal(ctx.status.failing, null);
        assert.equal(ctx.last_frame.downloaded_bytes, 0);
        assert.ok(ctx.catalog.packages.some((p) => p.name === "image"));
        return { action: "search", query: "saturation", input: "Image" };
      },
      // Turn 2: Receives `image.adjust@1` from catalog, then proposes a deliberate compile bug.
      (ctx) => {
        assert.ok(
          ctx.catalog_result.entries.some((e) => e.id === "image.adjust@1"),
          "catalog must discover image.adjust@1",
        );
        return {
          action: "propose",
          summary: "Broken syntax attempt",
          text: BASE_SCENE.replace("0.5", "undefined_symbol"),
        };
      },
      // Turn 3: Sees compile fault line/message, fixes syntax, but injects NaN at (2, 1).
      (ctx) => {
        assert.equal(ctx.trial_rejected, true);
        assert.equal(ctx.failed_gates[0].name, "compile");
        assert.match(ctx.fault.message, /undefined_symbol/);
        assert.equal(readFileSync(scenePath, "utf8"), BASE_SCENE, "disk must remain untouched after compile fail");
        return {
          action: "propose",
          summary: "NaN pixel attempt",
          text: `node gradient: kernel {
    requires { field.write; }
    output result: Image;
    domain = vec2(8, 8);
    compute {
        let r = if icoord().x == 2 && icoord().y == 1 { 0.0 / 0.0 } else { coord().x / 8.0 };
        return vec4(r, coord().y / 8.0, 0.5, 1.0);
    }
}
output result <- gradient.result;
`,
        };
      },
      // Turn 4: Sees `finite` gate rejection + auto-captured `trace` at [2, 1, 0];
      // composes clean desaturated scene using `image.adjust@1` from `image.entpkg`.
      (ctx) => {
        assert.equal(ctx.trial_rejected, true);
        assert.equal(ctx.failed_gates[0].name, "finite");
        assert.equal(ctx.failed_gates[0].nonfinite, 1);
        assert.deepEqual(ctx.trace.at, [2, 1, 0]);
        assert.equal(readFileSync(scenePath, "utf8"), BASE_SCENE, "disk must remain untouched after finite gate fail");
        return {
          action: "propose",
          summary: "Apply image.adjust@1 (saturation = 0.0) from image.entpkg",
          text: `use "image.entpkg";
node gradient: kernel {
    requires { field.write; }
    output result: Image;
    domain = vec2(8, 8);
    compute {
        return vec4(coord().x / 8.0, coord().y / 8.0, 0.5, 1.0);
    }
}
node gray: image.adjust@1 {
    source <- gradient.result;
    saturation = 0.0;
}
output result <- gray.result;
`,
        };
      },
    ]);

    const res = await evolve(client, {
      intent: "Convert gradient to grayscale using existing packages",
      provider,
      maxAttempts: 4,
      gates: { steps: 2, forbid_nonfinite: true, forbid_download: true, max_frame_ms: 500 },
    });

    assert.equal(res.ok, true);
    assert.equal(res.promoted, true);
    assert.equal(res.attempts, 3);
    assert.ok(res.trial.outputs[0].diff.changed > 0, "grayscale output must differ from color baseline");
    assert.match(readFileSync(scenePath, "utf8"), /image\.adjust@1/);

    const postObs = await client.observe({ measure: true });
    assert.equal(postObs.status.loaded, true);
    assert.equal(postObs.status.failing, null);
    assert.equal(postObs.last_frame.downloaded_bytes, 0);
    assert.equal(postObs.outputs[0].nonfinite, 0);
  } finally {
    await client.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

async function testMultiDocumentShadowCreation(bin, env) {
  const { dir, scenePath } = tempScene("multidoc");
  const client = EntelechyClient.spawn(bin, scenePath, { args: ["--backend", "cpu"], env });
  try {
    const newKernel = `kernel custom.invert@1 {
    requires { field.read; field.write; }
    input source: Image;
    output result: Image;
    compute {
        let c = sample(source);
        return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a);
    }
}
`;
    const newScene = `use "kernels/";
node gradient: kernel {
    requires { field.write; }
    output result: Image;
    domain = vec2(8, 8);
    compute {
        return vec4(coord().x / 8.0, coord().y / 8.0, 0.5, 1.0);
    }
}
node inv: custom.invert@1 {
    source <- gradient.result;
}
output result <- inv.result;
`;

    // Dry-run first: must NOT create kernels/invert.ent on disk!
    const dryRes = await evolve(client, {
      intent: "Invert colors with a new Kernel file",
      dryRun: true,
      provider: scriptedProvider([
        () => ({
          action: "propose",
          summary: "Create kernels/invert.ent and wire custom.invert@1",
          documents: [
            { path: "kernels/invert.ent", text: newKernel },
            { text: newScene },
          ],
        }),
      ]),
    });
    assert.equal(dryRes.ok, true);
    assert.equal(dryRes.promoted, false);
    assert.equal(existsSync(join(dir, "kernels/invert.ent")), false, "dry-run must not write new files");

    // Promote: atomically writes both files and loads them into the live session!
    const liveRes = await evolve(client, {
      intent: "Invert colors with a new Kernel file",
      dryRun: false,
      provider: scriptedProvider([
        () => ({
          action: "propose",
          summary: "Create kernels/invert.ent and wire custom.invert@1",
          documents: [
            { path: "kernels/invert.ent", text: newKernel },
            { text: newScene },
          ],
        }),
      ]),
    });
    assert.equal(liveRes.ok, true);
    assert.equal(liveRes.promoted, true);
    assert.equal(existsSync(join(dir, "kernels/invert.ent")), true);
    assert.match(readFileSync(scenePath, "utf8"), /custom\.invert@1/);

    // Follow-up evolution: verify `local_files` lists `kernels/invert.ent` and
    // the `inspect` action reads sibling `.ent` source + live `help("mix")`.
    const inspectRes = await evolve(client, {
      intent: "Blend inverted colors halfway with original using mix",
      provider: scriptedProvider([
        (ctx) => {
          assert.ok(
            ctx.local_files.includes("kernels/invert.ent"),
            `expected kernels/invert.ent in local_files: ${JSON.stringify(ctx.local_files)}`,
          );
          return { action: "inspect", paths: ["kernels/invert.ent"], help: ["mix"] };
        },
        (ctx) => {
          const docs = ctx.inspect_result.documents;
          assert.equal(docs.length, 1);
          assert.match(docs[0].text, /custom\.invert@1/);
          const helps = ctx.inspect_result.help;
          assert.equal(helps.length, 1);
          assert.equal(helps[0].error, undefined);
          return {
            action: "propose",
            summary: "Blend custom.invert@1 halfway via mix()",
            path: "kernels/invert.ent",
            text: docs[0].text.replace(
              "return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a);",
              "return vec4(mix(c.rgb, vec3(1.0, 1.0, 1.0) - c.rgb, 0.5), c.a);",
            ),
          };
        },
      ]),
    });
    assert.equal(inspectRes.ok, true);
    assert.equal(inspectRes.promoted, true);
    assert.match(readFileSync(join(dir, "kernels/invert.ent"), "utf8"), /mix\(c\.rgb/);
  } finally {
    await client.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

async function testVisualReviewClosedLoop(bin, env) {
  const { dir, scenePath } = tempScene("visual");
  const client = EntelechyClient.spawn(bin, scenePath, { args: ["--backend", "cpu"], env });
  try {
    await client.evaluate({ time: 0.0 });
    const stages = [];

    const res = await evolve(client, {
      intent: "Make the scene bright warm gold and verify visually before committing",
      maxAttempts: 3,
      visualReview: true,
      gates: { steps: 1, forbid_nonfinite: true, include_png: true },
      onEvent: (ev) => stages.push(ev.stage),
      provider: scriptedProvider([
        // Turn 1: Receives baseline live PNG in Stage 1 observe; proposes a dark draft.
        (ctx, _turn, images) => {
          assert.equal(ctx.visual_review, true);
          assert.equal(images.length, 1, "Stage 1 observe must attach baseline PNG");
          assert.ok(images[0].length > 20, "baseline PNG must be non-empty base64");
          return {
            action: "propose",
            summary: "First draft (too dark)",
            text: BASE_SCENE.replace("vec4(coord().x / 8.0, coord().y / 8.0, 0.5, 1.0)", "vec4(0.1, 0.05, 0.0, 1.0)"),
          };
        },
        // Turn 2: Mechanical gates passed in shadow, NOT committed yet! Provider sees shadow PNG,
        // rejects visually, and proposes bright gold instead.
        (ctx, _turn, images) => {
          assert.equal(ctx.visual_review_needed, true);
          assert.equal(images.length, 1, "visual review must attach shadow PNG");
          assert.equal(
            readFileSync(scenePath, "utf8"),
            BASE_SCENE,
            "disk must remain untouched while awaiting visual acceptance",
          );
          return {
            action: "propose",
            summary: "Bright warm gold revision",
            text: BASE_SCENE.replace("vec4(coord().x / 8.0, coord().y / 8.0, 0.5, 1.0)", "vec4(1.0, 0.82, 0.2, 1.0)"),
          };
        },
        // Turn 3: Sees second shadow PNG, approves visually -> atomic commit!
        (ctx, _turn, images) => {
          assert.equal(ctx.visual_review_needed, true);
          assert.equal(images.length, 1);
          assert.equal(readFileSync(scenePath, "utf8"), BASE_SCENE, "disk still untouched before accept");
          return {
            action: "accept",
            summary: "Warm gold verified visually; promote live",
          };
        },
      ]),
    });

    assert.equal(res.ok, true);
    assert.equal(res.promoted, true);
    assert.equal(res.attempts, 2);
    assert.ok(stages.includes("visual_review"));
    assert.ok(stages.includes("promoted"));
    assert.match(readFileSync(scenePath, "utf8"), /1\.0, 0\.82, 0\.2/);
  } finally {
    await client.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

async function testTcpLiveCoCreationAndZeroPollution(bin, env) {
  const { dir, scenePath } = tempScene("tcp");
  const server = spawn(bin, ["serve", scenePath, "--backend", "cpu", "--listen", "127.0.0.1:0"], {
    stdio: ["pipe", "ignore", "pipe"],
    env,
  });

  const addr = await new Promise((resolveAddr, reject) => {
    let buf = "";
    server.stderr.on("data", (c) => {
      buf += c.toString();
      const m = buf.match(/listening on (\S+)\r?\n/);
      if (m) resolveAddr(m[1]);
    });
    server.once("exit", (code) => reject(new Error(`serve exited ${code}: ${buf}`)));
  });

  const client = await EntelechyClient.connect(addr);
  try {
    // Exhausted retries leave disk and live session untouched.
    const failed = await evolve(client, {
      intent: "Always fail budget/compile",
      maxAttempts: 2,
      provider: scriptedProvider([
        () => ({ action: "propose", summary: "bad 1", text: "node broken: missing@1 {}" }),
        () => ({ action: "propose", summary: "bad 2", text: "node broken2: missing@1 {}" }),
      ]),
    });
    assert.equal(failed.ok, false);
    assert.equal(failed.promoted, false);
    assert.equal(readFileSync(scenePath, "utf8"), BASE_SCENE);

    // Clean promotion over live TCP socket succeeds.
    const promoted = await evolve(client, {
      intent: "Shift blue channel to 1.0",
      provider: scriptedProvider([
        () => ({
          action: "propose",
          summary: "Set blue channel to 1.0",
          text: BASE_SCENE.replace("0.5", "1.0"),
        }),
      ]),
    });
    assert.equal(promoted.ok, true);
    assert.equal(promoted.promoted, true);
    assert.match(readFileSync(scenePath, "utf8"), /1\.0, 1\.0/);
  } finally {
    await client.close();
    server.stdin.write(`${JSON.stringify({ op: "quit" })}\n`);
    server.stdin.end();
    await new Promise((r) => server.once("exit", r));
    rmSync(dir, { recursive: true, force: true });
  }
}

async function main() {
  const bin = findBinary();
  const env = {
    ...process.env,
    ENTELECHY_PACKAGES: join(REPO_ROOT, "packages"),
  };

  await testSearchRepairGateAndPromote(bin, env);
  console.log("ai bridge: search + compile repair + finite gate/trace + atomic promote ok");
  await testMultiDocumentShadowCreation(bin, env);
  console.log("ai bridge: multi-document shadow trial, inspect (.ent + help) & atomic promote ok");
  await testVisualReviewClosedLoop(bin, env);
  console.log("ai bridge: multimodal visual review (shadow PNG -> revise -> accept -> commit) ok");
  await testTcpLiveCoCreationAndZeroPollution(bin, env);
  console.log("ai bridge: live TCP co-creation & zero-pollution rollback ok");
  console.log("RESULT: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
