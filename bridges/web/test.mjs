// Runs the bridge's test page headless: serves the text and voxel examples,
// serves the repository's files to the page over HTTP (for Entelechy in the
// page, when `--glue` names what `wasm-bindgen` wrote for it), opens test.html in a
// headless Chrome over the DevTools protocol, waits for its RESULT line,
// prints the page's log, and exits 0 when it passed. Node 22 or later
// (for its WebSocket) and a Chrome; nothing installed.
//
//   node bridges/web/test.mjs [--entelechy target/release/entelechy] [--chrome <path>]
//                             [--glue target/page/entelechy_ffi.js]

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { serve as serveFiles } from "./serve.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : fallback;
};
const entelechy = resolve(root, option("--entelechy", "target/release/entelechy"));
const chrome = option(
  "--chrome",
  process.platform === "darwin"
    ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    : "google-chrome",
);

/// Starts `entelechy serve` on a free port and resolves with the port. It
/// admits a page opened from a file, as the README says to.
function serve(scene, backend) {
  return new Promise((resolvePort, reject) => {
    const listen = ["--listen", "127.0.0.1:0", "--allow-origin", "null"];
    const child = spawn(entelechy, ["serve", join(root, scene), "--backend", backend, ...listen], {
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
      ["--headless=new", "--no-first-run", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
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
const closers = [];
const profile = mkdtempSync(join(tmpdir(), "entelechy-web-"));
let status = 1;
try {
  const [text, voxel] = await Promise.all([
    serve("examples/text/scene.esn", "cpu"),
    serve("examples/voxel/scene.esn", "gpu"),
  ]);
  children.push(text.child, voxel.child);
  const browser = await launch(profile);
  children.push(browser.child);
  const devtools = await DevTools.open(browser.url);
  const served = await serveFiles();
  closers.push(() => served.close());
  const origin = `http://127.0.0.1:${served.address().port}`;
  const glue = option("--glue", "target/page/entelechy_ffi.js");
  const inPage = existsSync(resolve(root, glue));
  if (!inPage) console.log(`in the page: skipped, no ${glue}`);
  const gluePath = relative(root, resolve(root, glue)).split(sep).join("/");
  const glueQuery = inPage ? `&glue=${encodeURIComponent(gluePath)}` : "";
  const page = `${origin}/bridges/web/test.html?text=${text.port}&voxel=${voxel.port}${glueQuery}`;
  const { targetId } = await devtools.send("Target.createTarget", { url: page });
  const { sessionId } = await devtools.send("Target.attachToTarget", { targetId, flatten: true });
  await devtools.send("Runtime.enable", {}, sessionId);
  const deadline = Date.now() + 60_000;
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
    await run(`document.getElementById("kernels").value = "scene.back@1"; document.getElementById("load").click(); true`);
    await sleep(500);
    const source = await run(`document.getElementById("source").value`);
    const loaded = /^kernel scene\.back@1 \{/m.test(source) && !/domain =/.test(source);
    const ok = / ms /.test(page) && kernels > 0 && loaded;
    console.log(`index.html: ${ok ? "connected, framed, and loaded a Kernel" : `status "${page}", ${kernels} kernels, loaded ${loaded}`}`);
    if (!ok) log = log.replace("RESULT: ok", "RESULT: failed: index.html");
  }
  // The page again, served, with Entelechy opened in it over the text
  // example's files: the status line frames, the Kernels list.
  if (log.includes("RESULT: ok") && inPage) {
    const created = await devtools.send("Target.createTarget", { url: `${origin}/bridges/web/index.html?glue=${encodeURIComponent(`/${gluePath}`)}` });
    const attached = await devtools.send("Target.attachToTarget", { targetId: created.targetId, flatten: true });
    await devtools.send("Runtime.enable", {}, attached.sessionId);
    const run = (expression) =>
      devtools.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, attached.sessionId).then((r) => r.result.value);
    await sleep(500);
    const opened = await run(`(async () => {
      const bytes = async (path) => new Uint8Array(await (await fetch("/" + path)).arrayBuffer());
      const list = await (await fetch("/list?roots=examples/text,packages")).json();
      const files = await Promise.all(list.map(async (path) => [path, await bytes(path)]));
      await openInPage(files, "examples/text/scene.esn");
      return document.getElementById("status").textContent + " | " + document.getElementById("kernels").options.length;
    })().catch((error) => "failed: " + error.message)`);
    const ok = / ms .*\| [1-9]/.test(opened);
    console.log(`index.html in the page: ${ok ? "opened, framed, and listed the Kernels" : opened}`);
    if (!ok) log = log.replace("RESULT: ok", "RESULT: failed: index.html in the page");
  }
  // The Player in a page: a replay played to its end on the page's GPU
  // and on its JIT, and the person's keys reaching the play.
  if (log.includes("RESULT: ok") && inPage) {
    const glued = encodeURIComponent(`/${gluePath}`);
    const play = async (query, act) => {
      const created = await devtools.send("Target.createTarget", { url: `${origin}/bridges/web/play.html?${query}&glue=${glued}` });
      const attached = await devtools.send("Target.attachToTarget", { targetId: created.targetId, flatten: true });
      await devtools.send("Runtime.enable", {}, attached.sessionId);
      const said = () =>
        devtools
          .send("Runtime.evaluate", { expression: "globalThis.player ? player.said.join('\\n') : ''", returnByValue: true }, attached.sessionId)
          .then((r) => r.result.value || "");
      const key = async (key) => {
        for (const type of ["keyDown", "keyUp"]) {
          await devtools.send("Input.dispatchKeyEvent", { type, key, code: key }, attached.sessionId);
        }
      };
      const run = (expression) =>
        devtools
          .send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, attached.sessionId)
          .then((r) => r.result.value);
      return act({ said, key, run });
    };
    const until = async (said, pattern, seconds) => {
      const deadline = Date.now() + seconds * 1000;
      let text = "";
      while (Date.now() < deadline) {
        text = await said();
        if (pattern.test(text)) return text;
        await sleep(250);
      }
      return `timed out: ${text}`;
    };
    for (const on of ["gpu", "jit"]) {
      const text = await play(`replay=examples/ant/tests/ant.replay&on=${on}`, ({ said }) =>
        until(said, /replay: ok \(2 frames, 14 checks\)/, 30),
      );
      const ok = /replay: ok/.test(text);
      console.log(`play.html on the ${on}: ${ok ? "the replay played to its end, every check held" : text}`);
      if (!ok) log = log.replace("RESULT: ok", `RESULT: failed: play.html on the ${on}`);
    }
    const keys = await play("scene=examples/ant/scene.esn&on=jit", async ({ said, key, run }) => {
      await until(said, /scene loaded/, 30);
      await sleep(500);
      await key("F5");
      await key("F6");
      const pressed = await until(said, /state reset[\s\S]*previewing/, 10);
      // An agent in the page, answered between frames as `--serve` is.
      const backend = await run("player.call('status').then((s) => s.backend)");
      return backend === "jit" ? pressed : `the protocol in the page answered ${backend}`;
    });
    const ok = /previewing/.test(keys);
    console.log(`play.html keys: ${ok ? "F5 and F6 reached the play, and the protocol was answered between frames" : keys}`);
    if (!ok) log = log.replace("RESULT: ok", "RESULT: failed: play.html keys");
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
  for (const close of closers) close();
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
