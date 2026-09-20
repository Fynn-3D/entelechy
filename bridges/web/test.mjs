// Runs the bridge's test page headless: serves the text and voxel examples,
// opens test.html in a headless Chrome over the DevTools protocol, waits
// for its RESULT line, prints the page's log, and exits 0 when it passed.
// Node 22 or later (for its WebSocket) and a Chrome; nothing installed.
//
//   node bridges/web/test.mjs [--orla target/release/orla] [--chrome <path>]

import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : fallback;
};
const orla = resolve(root, option("--orla", "target/release/orla"));
const chrome = option(
  "--chrome",
  process.platform === "darwin"
    ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    : "google-chrome",
);

/// Starts `orla serve` on a free port and resolves with the port.
function serve(scene, backend) {
  return new Promise((resolvePort, reject) => {
    const child = spawn(orla, ["serve", join(root, scene), "--backend", backend, "--listen", "127.0.0.1:0"], {
      stdio: ["pipe", "ignore", "pipe"],
    });
    let text = "";
    child.log = () => text;
    child.stderr.on("data", (chunk) => {
      text += chunk;
      const match = text.match(/listening on 127\.0\.0\.1:(\d+)/);
      if (match) resolvePort({ child, port: Number(match[1]) });
    });
    child.on("exit", (code) => reject(new Error(`${scene} exited with ${code}: ${text}`)));
  });
}

/// Launches Chrome headless with remote debugging and resolves with the
/// browser's DevTools socket URL.
function launch(profile) {
  return new Promise((resolveUrl, reject) => {
    const child = spawn(
      chrome,
      ["--headless=new", "--disable-gpu", "--no-first-run", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    let text = "";
    child.stderr.on("data", (chunk) => {
      text += chunk;
      const match = text.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) resolveUrl({ child, url: match[1] });
    });
    child.on("exit", (code) => reject(new Error(`chrome exited with ${code}: ${text}`)));
  });
}

/// A small DevTools client: numbered commands, answers by id, events ignored.
class DevTools {
  constructor(socket) {
    this.socket = socket;
    this.next = 1;
    this.pending = new Map();
    this.problems = [];
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.method === "Runtime.exceptionThrown") {
        const details = message.params.exceptionDetails;
        this.problems.push(details.exception?.description || details.text);
      }
      if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
        this.problems.push(message.params.args.map((arg) => arg.value ?? arg.description).join(" "));
      }
      const waiting = this.pending.get(message.id);
      if (!waiting) return;
      this.pending.delete(message.id);
      if (message.error) waiting.reject(new Error(message.error.message));
      else waiting.resolve(message.result);
    };
  }
  static open(url) {
    return new Promise((resolveClient, reject) => {
      const socket = new WebSocket(url);
      socket.onopen = () => resolveClient(new DevTools(socket));
      socket.onerror = () => reject(new Error(`cannot reach ${url}`));
    });
  }
  send(method, params = {}, sessionId) {
    const id = this.next++;
    return new Promise((resolveResult, reject) => {
      this.pending.set(id, { resolve: resolveResult, reject });
      this.socket.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
}

const sleep = (millis) => new Promise((done) => setTimeout(done, millis));

const children = [];
const profile = mkdtempSync(join(tmpdir(), "orla-web-"));
let status = 1;
try {
  const [text, voxel] = await Promise.all([
    serve("examples/text/scene.osn", "cpu"),
    serve("examples/voxel/scene.osn", "gpu"),
  ]);
  children.push(text.child, voxel.child);
  const browser = await launch(profile);
  children.push(browser.child);
  const devtools = await DevTools.open(browser.url);
  const page = `${pathToFileURL(join(here, "test.html")).href}?text=${text.port}&voxel=${voxel.port}`;
  const { targetId } = await devtools.send("Target.createTarget", { url: page });
  const { sessionId } = await devtools.send("Target.attachToTarget", { targetId, flatten: true });
  await devtools.send("Runtime.enable", {}, sessionId);
  const deadline = Date.now() + 20_000;
  let log = "";
  while (Date.now() < deadline) {
    await sleep(250);
    const evaluated = await devtools.send(
      "Runtime.evaluate",
      { expression: "document.getElementById('log') ? document.getElementById('log').textContent : ''", returnByValue: true },
      sessionId,
    );
    log = evaluated.result.value || "";
    if (log.includes("RESULT:")) break;
  }
  process.stdout.write(log.endsWith("\n") ? log : `${log}\n`);
  // The page itself: connect to the text session, load a Kernel, and
  // read what its status line and editor say.
  if (log.includes("RESULT: ok")) {
    const index = pathToFileURL(join(here, "index.html")).href;
    const created = await devtools.send("Target.createTarget", { url: index });
    const attached = await devtools.send("Target.attachToTarget", { targetId: created.targetId, flatten: true });
    await devtools.send("Runtime.enable", {}, attached.sessionId);
    const run = (expression) =>
      devtools.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, attached.sessionId).then((r) => r.result.value);
    await sleep(500);
    await run(`document.getElementById("url").value = "ws://127.0.0.1:${text.port}"; document.getElementById("connect").click(); true`);
    let page = "";
    const until = Date.now() + 10_000;
    while (Date.now() < until) {
      await sleep(250);
      page = await run(`document.getElementById("status").textContent`);
      if (/ ms /.test(page)) break;
    }
    const kernels = await run(`document.getElementById("kernels").options.length`);
    await run(`document.getElementById("kernels").value = "scene.title@1"; document.getElementById("load").click(); true`);
    await sleep(500);
    const source = await run(`document.getElementById("source").value`);
    const loaded = /^kernel scene\.title@1 \{/m.test(source) && !/domain =/.test(source);
    const ok = / ms /.test(page) && kernels > 0 && loaded;
    console.log(`index.html: ${ok ? "connected, framed, and loaded a Kernel" : `status "${page}", ${kernels} kernels, loaded ${loaded}`}`);
    if (!ok) log = log.replace("RESULT: ok", "RESULT: failed: index.html");
  }
  for (const problem of devtools.problems) console.log(`page: ${problem}`);
  if (!log.includes("RESULT:")) console.log("RESULT: timed out");
  status = log.includes("RESULT: ok") ? 0 : 1;
  if (status !== 0) {
    for (const [name, served] of [["text", text], ["voxel", voxel]]) {
      const tail = served.child.log().trim().split("\n").slice(-4).join("\n  ");
      console.log(`${name} session said:\n  ${tail}`);
    }
  }
  devtools.socket.close();
} catch (error) {
  console.error(`error: ${error.message}`);
} finally {
  await Promise.all(
    children.map(
      (child) =>
        new Promise((done) => {
          child.once("exit", done);
          child.kill();
          setTimeout(done, 3000);
        }),
    ),
  );
  rmSync(profile, { recursive: true, force: true, maxRetries: 5 });
}
process.exit(status);
