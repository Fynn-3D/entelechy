// What Entelechy, built for a page, imports from `entelechy`: the glue
// `wasm-bindgen` writes imports this module by that name, which the
// page's import map points here. The functions are `host` in entelechy.js
// (loaded first, as a classic script), bound to the instance once the
// glue has made it.

let instance = null;
const hosted = globalThis.Entelechy.host(() => instance);

export const load = hosted.load;
export const unload = hosted.unload;
export const wait = hosted.wait;
export const wake = hosted.wake;
export const now = hosted.now;
export const fetch = hosted.fetch;
export const fetch_start = hosted.fetch_start;
export const fetch_state = hosted.fetch_state;
export const fetch_copy = hosted.fetch_copy;
export const fetch_end = hosted.fetch_end;

/// Names the instance the functions work on.
export function bind(made) {
  instance = made;
}
