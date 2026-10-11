// Serves the repository's files, read only, to a page on this machine:
// what Entelechy in the page needs, since a page that loads modules (the glue
// `wasm-bindgen` writes) is served rather than opened from a file.
// `/list?roots=a,b` answers every file under those directories, which a
// test hands in, and `/stamps?roots=a,b` when each was last changed,
// which a page watches; nothing but GET is answered, and nothing outside
// the repository.
//
//   node bridges/web/serve.mjs [port]     then open
//   http://127.0.0.1:<port>/bridges/web/index.html

import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm" };

/// Starts serving on `port` (0 for any free one) and resolves with the server.
export function serve(port = 0) {
  const list = (dir) =>
    readdirSync(dir).flatMap((name) => {
      if (name.startsWith(".")) return [];
      const path = join(dir, name);
      return statSync(path).isDirectory() ? list(path) : [relative(root, path).split(sep).join("/")];
    });
  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    if (request.method !== "GET") return response.writeHead(405).end();
    if (url.pathname === "/list") {
      const roots = (url.searchParams.get("roots") || "").split(",").filter(Boolean);
      const inside = roots.map((dir) => resolve(root, dir)).filter((dir) => dir.startsWith(root + sep) && existsSync(dir));
      response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
      return response.end(JSON.stringify(inside.flatMap(list)));
    }
    // What each file under the roots was last changed at, for a page
    // that watches them as a window's session watches its disk.
    if (url.pathname === "/stamps") {
      const roots = (url.searchParams.get("roots") || "").split(",").filter(Boolean);
      const inside = roots.map((dir) => resolve(root, dir)).filter((dir) => dir.startsWith(root + sep) && existsSync(dir));
      const stamps = Object.fromEntries(inside.flatMap(list).map((path) => [path, statSync(resolve(root, path)).mtimeMs]));
      response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
      return response.end(JSON.stringify(stamps));
    }
    const path = resolve(root, `.${decodeURIComponent(url.pathname)}`);
    if (!path.startsWith(root + sep) || !existsSync(path) || statSync(path).isDirectory()) {
      return response.writeHead(404).end();
    }
    const type = types[path.slice(path.lastIndexOf("."))] || "application/octet-stream";
    // Never kept: what is served is what is on disk now, as a page that
    // watches the files, and Entelechy rebuilt, must find.
    response.writeHead(200, { "content-type": type, "cache-control": "no-store" });
    response.end(readFileSync(path));
  });
  return new Promise((done) => server.listen(port, "127.0.0.1", () => done(server)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const server = await serve(Number(process.argv[2] || 7789));
  console.log(`serving ${root} on http://127.0.0.1:${server.address().port}/bridges/web/index.html`);
}
