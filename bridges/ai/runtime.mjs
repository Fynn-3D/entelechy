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
2. Never invent language keywords or builtins. Every Kernel declares explicit "requires { ... }", typed inputs/params/outputs, and a bounded body.
3. Every proposed change is verified inside an isolated in-memory shadow session via the protocol's \`trial\` operation before it ever touches disk or the live world.

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

2) Propose a candidate scene/Kernel change to run through shadow verification gates:
{
  "action": "propose",
  "summary": "Brief description of the change",
  "text": "...full root scene.esn text (omit if using edits or documents)...",
  "edits": null,
  "documents": null
}
Note: for multi-file changes (such as adding a new local .ent Kernel file and wiring it in scene.esn), use "documents": [{"path": "kernels/foo.ent", "text": "..."}, {"text": "...root scene.esn..."}].`;

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
 * Runs the 5-stage L7 capability evolution loop against a live Entelechy client.
 *
 * @param {import("./client.mjs").EntelechyClient} client
 * @param {object} options
 * @param {string} options.intent - Natural-language goal.
 * @param {{ name?: string, complete: (req: { system: string, messages: Array<{role: string, content: string, images?: string[]}> }) => Promise<object> }} options.provider
 * @param {number} [options.maxAttempts=4] - Maximum shadow trial attempts.
 * @param {number} [options.maxSearches=3] - Maximum mid-loop catalog lookups.
 * @param {object} [options.gates] - Verification gates forwarded to `trial`.
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
    maxSearches = 3,
    gates = {},
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

  // Stage 1: Observe the running world via minimal protocol primitives.
  onEvent({ stage: "observe" });
  const [observation, document] = await Promise.all([
    client.observe({ measure: true, checkOutputs: true }),
    client.document(),
  ]);
  const { status, graph, last_frame: lastFrame, outputs } = observation;

  // Stage 2: Initial capability discovery via `catalog`.
  const keywords = extractKeywords(intent);
  onEvent({ stage: "catalog", keywords });
  const initialCatalog = await client.catalog({
    ...(keywords.length ? { query: keywords[0] } : {}),
    limit: catalogLimit,
  });

  const worldContext = {
    intent,
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
      steps: gates.steps ?? 2,
      forbid_nonfinite: gates.forbid_nonfinite ?? true,
      max_frame_ms: gates.max_frame_ms ?? null,
      forbid_download: gates.forbid_download ?? false,
    },
  };

  const messages = [
    {
      role: "user",
      content: JSON.stringify(worldContext, null, 2),
    },
  ];

  const audit = [];
  let trialCount = 0;
  let searchCount = 0;

  // Stage 3-5: Plan -> Search / Shadow Trial -> Auto-Repair -> Atomic Promote.
  while (trialCount < maxAttempts) {
    onEvent({ stage: "plan", attempt: trialCount + 1, maxAttempts });
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
            notice: "Search budget exhausted; now emit a `propose` action.",
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

    if (action?.action !== "propose") {
      trialCount += 1;
      audit.push({ kind: "invalid_action", action });
      messages.push({
        role: "user",
        content: JSON.stringify({
          error: `Unsupported action "${action?.action}". Reply with {"action":"search",...} or {"action":"propose",...}.`,
        }),
      });
      continue;
    }

    trialCount += 1;
    onEvent({
      stage: "trial",
      attempt: trialCount,
      maxAttempts,
      summary: action.summary || "",
    });

    const trialReq = buildTrialRequest(action, gates, !dryRun);
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
    if (gates.include_png) {
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
        "Inspect `failed_gates`, `fault` (line/column/fix), and `trace` (element bindings & calls), then emit a repaired `propose` action. Disk and the live scene were NOT modified.",
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
