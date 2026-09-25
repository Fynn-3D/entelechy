// What Orla, built for a page, imports from `orla`: the glue
// `wasm-bindgen` writes imports this module by that name, which the
// page's import map points here. The functions are `host` in orla.js
// (loaded first, as a classic script), bound to the instance once the
// glue has made it.

let instance = null;
const hosted = globalThis.Orla.host(() => instance);

export const load = hosted.load;
export const unload = hosted.unload;
export const wait = hosted.wait;
export const wake = hosted.wake;
export const now = hosted.now;

/// Names the instance the functions work on.
export function bind(made) {
  instance = made;
}
