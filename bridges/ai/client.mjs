//! Zero-dependency Creation Protocol client over stdio (`entelechy serve`)
//! or TCP (`entelechy serve --listen` / `entelechy-player --serve --listen`).

import { spawn } from "node:child_process";
import { realpathSync } from "node:fs";
import net from "node:net";
import { dirname, relative } from "node:path";
import { createInterface } from "node:readline";

function canonicalPath(p) {
  try {
    return realpathSync(p);
  } catch {
    return p;
  }
}

export class ProtocolError extends Error {
  constructor(payload) {
    super(payload?.message || "protocol request failed");
    this.name = "ProtocolError";
    this.payload = payload;
  }
}

export class EntelechyClient {
  #writeLine;
  #closeFn;
  #pending = new Map();
  #nextId = 1;
  #closed = false;

  constructor({ readable, writeLine, close }) {
    this.#writeLine = writeLine;
    this.#closeFn = close;

    const rl = createInterface({ input: readable, crlfDelay: Infinity });
    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let msg;
      try {
        msg = JSON.parse(trimmed);
      } catch {
        return;
      }
      const entry = this.#pending.get(msg.id);
      if (!entry) return;
      this.#pending.delete(msg.id);
      if ("ok" in msg) {
        entry.resolve(msg.ok);
      } else {
        entry.reject(new ProtocolError(msg.error));
      }
    });
    rl.on("close", () => {
      this.#closed = true;
      for (const entry of this.#pending.values()) {
        entry.reject(new Error("Entelechy session closed before answering"));
      }
      this.#pending.clear();
    });
  }

  /**
   * Spawns `entelechy serve <scene>` and speaks JSON-lines over stdio.
   */
  static spawn(binary, scenePath, { args = [], env = process.env, cwd } = {}) {
    const child = spawn(binary, ["serve", scenePath, ...args], {
      stdio: ["pipe", "pipe", "pipe"],
      env,
      cwd,
    });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    const client = new EntelechyClient({
      readable: child.stdout,
      writeLine: (line) => child.stdin.write(`${line}\n`),
      close: async () => {
        try {
          if (!client.#closed) {
            await client.request("quit").catch(() => {});
          }
        } finally {
          child.stdin.end();
          await new Promise((resolve) => {
            if (child.exitCode !== null) return resolve();
            const timer = setTimeout(() => {
              child.kill();
              resolve();
            }, 3000);
            child.once("exit", () => {
              clearTimeout(timer);
              resolve();
            });
          });
        }
      },
    });
    client.child = child;
    client.getStderr = () => stderr;
    return client;
  }

  /**
   * Connects to a running `entelechy serve --listen host:port` or
   * `entelechy-player --serve --listen host:port` session over TCP.
   */
  static async connect(address) {
    const [host, portStr] = address.includes(":")
      ? address.split(":")
      : ["127.0.0.1", address];
    const port = Number(portStr);
    if (!Number.isInteger(port) || port <= 0) {
      throw new Error(`invalid address "${address}", expected host:port`);
    }
    const socket = await new Promise((resolve, reject) => {
      const sock = net.createConnection({ host, port }, () => resolve(sock));
      sock.once("error", reject);
    });
    return new EntelechyClient({
      readable: socket,
      writeLine: (line) => socket.write(`${line}\n`),
      close: async () => {
        try {
          if (!socket.destroyed) {
            socket.write(`${JSON.stringify({ id: "bye", op: "quit" })}\n`);
          }
        } catch {
          // ignore
        }
        socket.end();
        socket.destroy();
      },
    });
  }

  request(op, fields = {}) {
    if (this.#closed) {
      return Promise.reject(new Error(`cannot send "${op}": session is closed`));
    }
    const id = this.#nextId++;
    const payload = { id, op, ...fields };
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#writeLine(JSON.stringify(payload));
    });
  }

  status() {
    return this.request("status");
  }

  graph() {
    return this.request("graph");
  }

  document(fields = {}) {
    return this.request("document", fields);
  }

  catalog(fields = {}) {
    return this.request("catalog", fields);
  }

  help(fields = {}) {
    return this.request("help", fields);
  }

  trial(fields = {}) {
    return this.request("trial", fields);
  }

  /**
   * Client-side L7 world observation built strictly from minimal protocol
   * primitives (`status`, `graph`, optional `evaluate`, and `read`).
   */
  async observe({ measure = false, checkOutputs = true, includePng = false } = {}) {
    let status = await this.status();
    if (status.loaded && (measure || status.last_frame == null)) {
      await this.evaluate({ measure: true }).catch(() => {});
      status = await this.status();
    }
    const graph = status.loaded ? await this.graph().catch(() => null) : null;
    const sceneDir = status.scene ? canonicalPath(dirname(status.scene)) : null;
    const localFiles = sceneDir
      ? [...new Set(
          (status.watched || [])
            .map((p) => relative(sceneDir, canonicalPath(p)).replace(/\\/g, "/"))
            .filter((p) => p && !p.startsWith("..") && !p.startsWith("/")),
        )].sort()
      : [];
    const outputs = [];
    const images = [];
    if ((checkOutputs || includePng) && graph?.outputs?.length) {
      for (const out of graph.outputs) {
        const name = typeof out === "string" ? out : out.name;
        if (!name) continue;
        const summary = checkOutputs
          ? await this.read({ output: name, format: "summary" }).catch(() => null)
          : null;
        let png = null;
        if (includePng) {
          const pic = await this.read({ output: name, format: "png" }).catch(() => null);
          if (pic?.data && pic.format === "png") {
            png = pic.data;
            images.push(pic.data);
          }
        }
        const nonfinite = (summary?.fields || []).reduce(
          (acc, f) => acc + (f.nonfinite || 0),
          0,
        );
        outputs.push({
          name,
          nonfinite,
          ...(summary ? { summary } : {}),
          ...(png ? { png } : {}),
        });
      }
    }
    return {
      status,
      graph,
      local_files: localFiles,
      last_frame: status.last_frame,
      outputs,
      images,
    };
  }

  compose(fields = {}) {
    return this.request("compose", fields);
  }

  evaluate(fields = {}) {
    return this.request("evaluate", fields);
  }

  read(fields = {}) {
    return this.request("read", fields);
  }

  trace(fields = {}) {
    return this.request("trace", fields);
  }

  snapshot(fields = {}) {
    return this.request("snapshot", fields);
  }

  restore(fields = {}) {
    return this.request("restore", fields);
  }

  async close() {
    await this.#closeFn();
  }
}
