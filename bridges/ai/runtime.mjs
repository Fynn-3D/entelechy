//! L7 AI Meta-Runtime: 5-stage deterministic capability compiler loop over
//! the Entelechy Creation Protocol (`observe -> search -> plan -> shadow trial
//! & auto-repair -> atomic promote`).
//!
//! Never accesses the Host directly: every world read, capability discovery,
//! speculative shadow verification, and atomic promotion flows strictly
//! through the minimal Creation Protocol (`status`, `graph`, `document`,
//! `read`, `catalog`, `trial`).

const STOP_WORDS = new Set([
  "a", "an", "the", "to", "for", "of", "in", "on", "with", "and", "or",
  "add", "make", "create", "update", "change", "keep", "scene", "this",
  "that", "into", "from", "using", "under", "over", "fps", "ms",
]);

const SYSTEM_PROMPT = `You are the Entelechy L7 Capability Compiler.
You evolve a live Entelechy scene (.esn graph + .ent Kernels) through structured JSON actions.

Core Entelechy rules:
1. Reuse existing capabilities from the catalog ("use \\"<pkg>.entpkg\\";" or "node x: graph \\"<pkg>.entpkg/graphs/g.esn\\" { ... }") before writing new Kernels.
2. Never invent language keywords or builtins. If unsure of a builtin signature or syntax, inspect it via {"action":"help"} before proposing. Every Kernel declares explicit "requires { ... }", typed inputs/params/outputs, and a bounded body.
3. For multi-file scenes (listed in \`local_files\`), read sibling .ent/.esn files via \`inspect\` before editing them. Prefer surgical \`edits\` or targeted \`documents\` changes.
4. Every proposed change is verified inside an isolated in-memory shadow session via the protocol's \`trial\` operation before it ever touches disk or the live world.

Reply with ONE JSON object matching one of these actions:

1) Search the capability catalog for existing packages, Kernels, or preset graphs:
{
  "action": "search",
  "query": "blur",
  "package": null,
  "kind": null,
  "mode": null,
  "input": "Image",
  "output": null
}

2) Inspect local scene documents (.ent/.esn) or look up live language/builtin help:
{
  "action": "inspect",
  "paths": ["kernels/sim.ent"],
  "help": ["at_linear"]
}

3) Propose a candidate scene/Kernel change to run through shadow verification gates:
{
  "action": "propose",
  "summary": "Brief description of the change",
  "path": null,
  "text": "...full document text (omit if using edits or documents)...",
  "edits": null,
  "documents": null
}
Note: for multi-file changes (such as adding/editing a local .ent Kernel file and wiring it in scene.esn), use "documents": [{"path": "kernels/foo.ent", "text": "..."}, {"text": "...root scene.esn..."}].

4) Accept a shadow-verified visual preview (ONLY when prompted with \`visual_review_needed: true\`):
{
  "action": "accept",
  "summary": "Visual check passed; promote shadow change to the live world"
}`;

function extractKeywords(intent) {
  const words = String(intent || "")
    .toLowerCase()
    .replace(/[^a-z0-9_.\- ]+/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !STOP_WORDS.has(w));
  return [...new Set(words)].slice(0, 4);
}

function summarizeCatalog(cat) {
  return {
    packages: (cat.packages || []).map((p) => ({
      name: p.name,
      id: p.id,
      use_path: p.use_path,
      kernels: p.kernels,
      graphs: p.graphs,
      doc: p.doc,
    })),
    total: cat.total,
    entries: (cat.entries || []).map((e) => ({
      kind: e.kind,
      id: e.id,
      package: e.package,
      use: e.use,
      mode: e.mode,
      doc: e.doc,
      inputs: e.inputs,
      params: e.params,
      outputs: e.outputs,
    })),
  };
}

function buildTrialRequest(proposal, gates, commit) {
  const req = {
    steps: gates.steps ?? 2,
    measure: gates.measure ?? true,
    forbid_nonfinite: gates.forbid_nonfinite ?? true,
    forbid_download: gates.forbid_download ?? false,
    include_png: gates.include_png ?? false,
    commit,
  };
  if (gates.max_frame_ms != null) req.max_frame_ms = gates.max_frame_ms;
  if (gates.dt != null) req.dt = gates.dt;
  if (gates.fresh_state != null) req.fresh_state = gates.fresh_state;
  if (gates.read_outputs != null) req.read_outputs = gates.read_outputs;
  if (gates.window != null) req.window = gates.window;
  if (gates.scale != null) req.scale = gates.scale;

  if (Array.isArray(proposal.documents) && proposal.documents.length > 0) {
    req.documents = proposal.documents;
  } else {
    if (proposal.path != null) req.path = proposal.path;
    if (proposal.edits != null) req.edits = proposal.edits;
    else if (proposal.text != null) req.text = proposal.text;
    if (proposal.expect != null) req.expect = proposal.expect;
  }
  return req;
}

/**
 * Runs the closed-loop L7 capability evolution cycle against a live Entelechy client:
 *   observe (+ optional PNG) -> search/inspect (catalog, local .ent docs, help) ->
 *   shadow trial gates & auto-repair -> optional multimodal visual review -> atomic promote.
 *
 * @param {import("./client.mjs").EntelechyClient} client
 * @param {object} options
 * @param {string} options.intent - Natural-language goal.
 * @param {{ name?: string, complete: (req: { system: string, messages: Array<{role: string, content: string, images?: string[]}> }) => Promise<object> }} options.provider
 * @param {number} [options.maxAttempts=4] - Maximum shadow trial attempts.
 * @param {number} [options.maxSearches=4] - Maximum mid-loop catalog lookups.
 * @param {number} [options.maxInspects=6] - Maximum mid-loop document/help lookups.
 * @param {object} [options.gates] - Verification gates forwarded to `trial`.
 * @param {boolean} [options.visualReview] - When true (default when `gates.include_png` is true), hold passing shadow renders for model visual confirmation (`accept` vs `propose`) before committing.
 * @param {boolean} [options.dryRun=false] - If true, verify in shadow without committing.
 * @param {number} [options.catalogLimit=24] - Max entries returned per catalog search.
 * @param {(event: object) => void} [options.onEvent] - Optional progress hook.
 */
export async function evolve(
  client,
  {
    intent,
    provider,
    maxAttempts = 4,
    maxSearches = 4,
    maxInspects = 6,
    gates = {},
    visualReview = Boolean(gates.include_png),
    dryRun = false,
    catalogLimit = 24,
    onEvent = () => {},
  },
) {
  if (!intent || !String(intent).trim()) {
    throw new Error("evolve requires a non-empty `intent`");
  }
  if (!provider || typeof provider.complete !== "function") {
    throw new Error("evolve requires a `provider` with a `complete` method");
  }

  const effectiveGates = visualReview ? { ...gates, include_png: true } : gates;

  // Stage 1: Observe the running world via minimal protocol primitives.
  onEvent({ stage: "observe" });
  const [observation, document] = await Promise.all([
    client.observe({
      measure: true,
      checkOutputs: true,
      includePng: Boolean(effectiveGates.include_png),
    }),
    client.document(),
  ]);
  const {
    status,
    graph,
    local_files: localFiles,
    last_frame: lastFrame,
    outputs,
    images: initialImages,
  } = observation;

  // Stage 2: Initial capability discovery via `catalog`.
  const keywords = extractKeywords(intent);
  onEvent({ stage: "catalog", keywords });
  const initialCatalog = await client.catalog({
    ...(keywords.length ? { query: keywords[0] } : {}),
    limit: catalogLimit,
  });

  const worldContext = {
    intent,
    visual_review: visualReview,
    status: {
      backend: status.backend,
      scene: status.scene,
      loaded: status.loaded,
      refused: status.refused,
      failing: status.failing,
      revision: status.revision,
      frame: status.frame,
      grants: status.grants,
      memory: {
        used: status.memory?.used,
        budget: status.memory?.budget,
        cold: status.memory?.cold,
      },
    },
    local_files: localFiles,
    document: {
      path: document.path,
      version: document.version,
      text: document.text,
    },
    graph: graph
      ? {
          params: graph.params,
          nodes: graph.nodes.map((n) => ({
            id: n.id,
            kernel: n.kernel,
            backend: n.backend,
            domain: n.domain,
          })),
          outputs: graph.outputs,
        }
      : null,
    last_frame: lastFrame
      ? {
          frame_ms: lastFrame.frame_ms,
          device_ms: lastFrame.device_ms,
          uploaded_bytes: lastFrame.uploaded_bytes,
          downloaded_bytes: lastFrame.downloaded_bytes,
          fused_groups: lastFrame.fused_groups,
          refused: lastFrame.refused,
          nodes: (lastFrame.nodes || []).slice(0, 8),
        }
      : null,
    outputs: outputs.map((o) => ({ name: o.name, nonfinite: o.nonfinite })),
    catalog: summarizeCatalog(initialCatalog),
    gates: {
      steps: effectiveGates.steps ?? 2,
      forbid_nonfinite: effectiveGates.forbid_nonfinite ?? true,
      max_frame_ms: effectiveGates.max_frame_ms ?? null,
      forbid_download: effectiveGates.forbid_download ?? false,
      include_png: Boolean(effectiveGates.include_png),
    },
  };

  const messages = [
    {
      role: "user",
      content: JSON.stringify(worldContext, null, 2),
      ...(initialImages?.length ? { images: initialImages } : {}),
    },
  ];

  const audit = [];
  let trialCount = 0;
  let searchCount = 0;
  let inspectCount = 0;
  let pendingVerified = null;

  // Stage 3-6: Plan -> Search/Inspect -> Shadow Trial -> Visual Review -> Atomic Promote.
  while (trialCount < maxAttempts || pendingVerified !== null) {
    onEvent({
      stage: pendingVerified ? "visual_decision" : "plan",
      attempt: pendingVerified ? trialCount : trialCount + 1,
      maxAttempts,
    });
    const action = await provider.complete({
      system: SYSTEM_PROMPT,
      messages,
    });
    messages.push({ role: "assistant", content: JSON.stringify(action) });

    if (action?.action === "search") {
      if (searchCount >= maxSearches) {
        messages.push({
          role: "user",
          content: JSON.stringify({
            notice: "Search budget exhausted; now emit a `propose` or `accept` action.",
          }),
        });
        continue;
      }
      searchCount += 1;
      const queryReq = { limit: catalogLimit };
      for (const key of ["query", "package", "kind", "mode", "input", "output", "requires"]) {
        if (action[key] != null) queryReq[key] = action[key];
      }
      onEvent({ stage: "search", index: searchCount, query: queryReq });
      const found = await client.catalog(queryReq);
      audit.push({ kind: "search", query: queryReq, total: found.total });
      messages.push({
        role: "user",
        content: JSON.stringify({
          catalog_result: summarizeCatalog(found),
        }),
      });
      continue;
    }

    if (action?.action === "inspect") {
      if (inspectCount >= maxInspects) {
        messages.push({
          role: "user",
          content: JSON.stringify({
            notice: "Inspect budget exhausted; now emit a `propose` or `accept` action.",
          }),
        });
        continue;
      }
      inspectCount += 1;
      const rawPaths = Array.isArray(action.paths)
        ? action.paths
        : action.path != null
          ? [action.path]
          : [];
      const rawHelps = Array.isArray(action.help)
        ? action.help
        : action.help != null
          ? [action.help]
          : [];
      onEvent({ stage: "inspect", index: inspectCount, paths: rawPaths, help: rawHelps });

      const documents = [];
      for (const p of rawPaths.slice(0, 6)) {
        try {
          const doc = await client.document({ path: String(p) });
          documents.push({ path: doc.path, version: doc.version, text: doc.text });
        } catch (err) {
          documents.push({ path: String(p), error: err.payload?.message || err.message });
        }
      }

      const helpResults = [];
      for (const name of rawHelps.slice(0, 6)) {
        try {
          const h = await client.help({ name: String(name) });
          helpResults.push({ name: String(name), result: h });
        } catch (err) {
          helpResults.push({
            name: String(name),
            error: err.payload?.message || err.message,
            similar: err.payload?.similar || null,
          });
        }
      }

      audit.push({ kind: "inspect", paths: rawPaths, help: rawHelps });
      messages.push({
        role: "user",
        content: JSON.stringify({ inspect_result: { documents, help: helpResults } }, null, 2),
      });
      continue;
    }

    if (action?.action === "accept") {
      if (!pendingVerified) {
        messages.push({
          role: "user",
          content: JSON.stringify({
            error: "No verified shadow proposal is pending review; emit `propose` first.",
          }),
        });
        continue;
      }
      const { proposal, trial: shadowTrial } = pendingVerified;
      const finalSummary = action.summary || proposal.summary || "";
      pendingVerified = null;

      if (dryRun) {
        audit.push({ kind: "accept", dryRun: true, summary: finalSummary });
        onEvent({
          stage: "verified",
          attempt: trialCount,
          frame_ms: shadowTrial.frame_ms,
        });
        return {
          ok: true,
          promoted: false,
          dryRun: true,
          attempts: trialCount,
          summary: finalSummary,
          trial: shadowTrial,
          audit,
        };
      }

      const commitReq = buildTrialRequest(proposal, { ...effectiveGates, include_png: false }, true);
      const committed = await client.trial(commitReq);
      audit.push({
        kind: "accept",
        summary: finalSummary,
        passed: committed.passed,
        committed: committed.committed,
        frame_ms: committed.frame_ms,
      });
      if (committed.passed && committed.committed) {
        onEvent({
          stage: "promoted",
          attempt: trialCount,
          frame_ms: committed.frame_ms,
          version: committed.version || null,
        });
        return {
          ok: true,
          promoted: true,
          dryRun: false,
          attempts: trialCount,
          summary: finalSummary,
          trial: committed,
          audit,
        };
      }
      // Rare race (e.g. disk changed during visual review): report back and retry if budget remains.
      messages.push({
        role: "user",
        content: JSON.stringify({
          commit_failed: true,
          fault: committed.fault || null,
          gates: committed.gates || [],
          instruction: "Live session changed during review; emit a refreshed `propose`.",
        }),
      });
      continue;
    }

    if (action?.action !== "propose") {
      if (pendingVerified && trialCount >= maxAttempts) {
        pendingVerified = null;
        break;
      }
      trialCount += 1;
      audit.push({ kind: "invalid_action", action });
      messages.push({
        role: "user",
        content: JSON.stringify({
          error: `Unsupported action "${action?.action}". Reply with \`search\`, \`inspect\`, \`propose\`, or \`accept\`.`,
        }),
      });
      continue;
    }

    if (trialCount >= maxAttempts) {
      // Model declined visual acceptance and had no remaining trial budget.
      pendingVerified = null;
      break;
    }

    pendingVerified = null;
    trialCount += 1;
    onEvent({
      stage: "trial",
      attempt: trialCount,
      maxAttempts,
      summary: action.summary || "",
    });

    const shouldCommitNow = !dryRun && !visualReview;
    const trialReq = buildTrialRequest(action, effectiveGates, shouldCommitNow);
    let result;
    try {
      result = await client.trial(trialReq);
    } catch (err) {
      const feedback = {
        trial_error: err.payload || { message: err.message },
        instruction: "Fix the request structure or expect version and propose again.",
      };
      audit.push({ kind: "trial_error", attempt: trialCount, error: feedback.trial_error });
      onEvent({ stage: "trial_error", attempt: trialCount, error: feedback.trial_error });
      messages.push({ role: "user", content: JSON.stringify(feedback) });
      continue;
    }

    audit.push({
      kind: "trial",
      attempt: trialCount,
      summary: action.summary || "",
      passed: result.passed,
      committed: result.committed,
      gates: result.gates,
      frame_ms: result.frame_ms,
      downloaded_bytes: result.downloaded_bytes,
      fault: result.fault || null,
    });

    if (result.passed) {
      if (visualReview) {
        const shadowImages = [];
        for (const out of result.outputs || []) {
          if (out.png) shadowImages.push(out.png);
        }
        pendingVerified = { proposal: action, trial: result };
        onEvent({
          stage: "visual_review",
          attempt: trialCount,
          frame_ms: result.frame_ms,
          images: shadowImages.length,
        });
        const reviewPayload = {
          visual_review_needed: true,
          attempt: trialCount,
          remaining_attempts: maxAttempts - trialCount,
          frame_ms: result.frame_ms,
          downloaded_bytes: result.downloaded_bytes,
          outputs: (result.outputs || []).map((o) => ({
            name: o.name,
            nonfinite: o.nonfinite,
            diff: o.diff,
          })),
          instruction:
            "All mechanical gates passed in shadow (live world and disk are NOT yet modified). Inspect the attached rendered shadow image(s): reply {\"action\":\"accept\",\"summary\":\"...\"} to promote atomically, OR reply {\"action\":\"propose\",...} to fix any visual defect.",
        };
        messages.push({
          role: "user",
          content: JSON.stringify(reviewPayload, null, 2),
          ...(shadowImages.length ? { images: shadowImages } : {}),
        });
        continue;
      }

      onEvent({
        stage: result.committed ? "promoted" : "verified",
        attempt: trialCount,
        frame_ms: result.frame_ms,
        version: result.version || null,
      });
      return {
        ok: true,
        promoted: Boolean(result.committed),
        dryRun,
        attempts: trialCount,
        summary: action.summary || "",
        trial: result,
        audit,
      };
    }

    // Shadow gate failed: feed exact compile/runtime/finite/budget diagnostics + auto-trace back!
    const failedGates = (result.gates || []).filter((g) => !g.passed);
    onEvent({
      stage: "gate_failed",
      attempt: trialCount,
      failedGates: failedGates.map((g) => g.name),
      fault: result.fault || null,
    });

    const images = [];
    if (effectiveGates.include_png) {
      for (const out of result.outputs || []) {
        if (out.png) images.push(out.png);
      }
    }

    const repairFeedback = {
      trial_rejected: true,
      attempt: trialCount,
      remaining_attempts: maxAttempts - trialCount,
      failed_gates: failedGates,
      fault: result.fault || null,
      trace: result.trace || null,
      frame_ms: result.frame_ms,
      downloaded_bytes: result.downloaded_bytes,
      timings: result.timings || [],
      outputs: (result.outputs || []).map((o) => ({
        name: o.name,
        nonfinite: o.nonfinite,
        nan: o.nan,
        inf: o.inf,
        diff: o.diff,
      })),
      instruction:
        "Inspect `failed_gates`, `fault` (line/column/fix), and `trace` (element bindings & calls), then emit a repaired `propose` action (or `inspect` local files / `help`). Disk and the live scene were NOT modified.",
    };
    messages.push({
      role: "user",
      content: JSON.stringify(repairFeedback, null, 2),
      ...(images.length ? { images } : {}),
    });
  }

  onEvent({ stage: "exhausted", attempts: trialCount });
  return {
    ok: false,
    promoted: false,
    dryRun,
    attempts: trialCount,
    audit,
  };
}
