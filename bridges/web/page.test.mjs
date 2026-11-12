// Opens a page exported whole (`entelechy export <scene.esn> --out <page>.html`)
// from the disk in a headless Chrome over the DevTools protocol, with
// nothing served beside it, and waits for its scene to load: the play
// says so on the page's console, the page stops showing that it loads,
// and it names no failure. Prints the page's log and exits 0 when it
// passed. Node 22 or later (for its WebSocket) and a Chrome; nothing
// installed.
//
//   node bridges/web/page.test.mjs --page <page.html> [--chrome <path>]

import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : fallback;
};
const page = option("--page");
if (!page) {
  console.error("name the page: --page <page.html>");
  process.exit(2);
}
const chrome = option(
  "--chrome",
  process.platform === "darwin" ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "google-chrome",
);

const profile = mkdtempSync(join(tmpdir(), "entelechy-page-"));
const child = spawn(
  chrome,
  ["--headless=new", "--no-first-run", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
  { stdio: ["ignore", "ignore", "pipe"] },
);
const done = (code) => {
  child.kill();
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {
    // Chrome may still hold its profile for a moment; the system's temporary directory is cleared in time.
  }
  process.exit(code);
};
const browser = await new Promise((found, failed) => {
  let said = "";
  child.stderr.on("data", (chunk) => {
    said += chunk;
    const match = said.match(/DevTools listening on (ws:\/\/\S+)/);
    if (match) found(match[1]);
  });
  child.on("exit", (code) => failed(new Error(`Chrome exited with ${code}: ${said}`)));
});
const targets = await (await fetch(`http://127.0.0.1:${new URL(browser).port}/json`)).json();
const socket = new WebSocket(targets.find((target) => target.type === "page").webSocketDebuggerUrl);
await new Promise((opened) => (socket.onopen = opened));

let next = 0;
const asked = new Map();
const log = [];
let loaded = false;
socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.id && asked.has(message.id)) {
    asked.get(message.id)(message.result);
    asked.delete(message.id);
  }
  if (message.method === "Runtime.consoleAPICalled") {
    const line = message.params.args.map((arg) => arg.value ?? arg.description ?? "").join(" ");
    log.push(line);
    if (line.includes("scene loaded")) loaded = true;
  }
  if (message.method === "Runtime.exceptionThrown") {
    const thrown = message.params.exceptionDetails;
    log.push(`exception: ${thrown.exception?.description || thrown.text}`);
  }
};
const send = (method, params = {}) =>
  new Promise((answered) => {
    asked.set(++next, answered);
    socket.send(JSON.stringify({ id: next, method, params }));
  });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result.value;
const until = async (seconds, met) => {
  for (let waited = 0; waited < seconds * 10; waited++) {
    if (await met()) return true;
    await new Promise((later) => setTimeout(later, 100));
  }
  return false;
};

await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url: pathToFileURL(resolve(page)).href });

const failures = [];
if (!(await until(120, async () => loaded || (await evaluate("document.title")).startsWith("Entelechy: ")))) {
  failures.push("the scene did not load in two minutes");
}
const title = await evaluate("document.title");
if (title.startsWith("Entelechy: ")) failures.push(`the page says: ${title}`);
if (loaded) {
  console.log("page: the scene loaded from the disk");
  if (await until(20, () => evaluate("document.getElementById('loading') === null"))) {
    console.log("page: it stopped showing that it loads");
  } else {
    failures.push("the page still shows that it is loading");
  }
  // Nothing was fetched: every resource the page asked for is itself or made in it.
  const fetched = await evaluate(
    "performance.getEntriesByType('resource').map((entry) => entry.name).filter((name) => !name.startsWith('blob:') && !name.startsWith('data:')).join(' ')",
  );
  if (fetched) failures.push(`the page fetched: ${fetched}`);
  else console.log("page: it fetched nothing");
}
for (const line of log) console.log(`  ${line}`);
for (const failure of failures) console.log(`FAILED: ${failure}`);
console.log(failures.length ? "RESULT: failed" : "RESULT: ok");
done(failures.length ? 1 : 0);
