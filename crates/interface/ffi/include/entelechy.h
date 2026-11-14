/* Entelechy in your own process: a session as a library, its Fields read in
 * place. The same session the command line and the Player run, behind a
 * C ABI any language binds; the JSON protocol is here too, one line at a
 * time, for everything the fast path does not name.
 *
 * Every function is called from one thread at a time per session, and
 * one call at a time: a call made while another is inside the session
 * (a page reaching in while Entelechy waits for its device) is refused as a
 * NULL session's is, and closing it then frees nothing. A
 * pointer or string a call returns belongs to the session and stays
 * valid until the next call on that session, except a Field's words,
 * which stay valid until the next evaluate, request or close. */
#ifndef ENTELECHY_H
#define ENTELECHY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this header's contract. */
#define ENTELECHY_ABI_VERSION 2

/* A running scene: its runtime, its documents, its state. */
typedef struct EntelechySession EntelechySession;

/* What an evaluation sees: the time in seconds, whole (`clock()` reads
 * it, `time()` rounded to a float), the sound device's rate (zero without
 * one), the pointer in output pixels, what the wheel scrolled since the
 * last evaluation (x sideways, y positive away from the person), the
 * window in physical pixels (zero by zero for none, then the session's
 * own window applies) and its scale factor, the pointer buttons held as
 * bits (0 left, 1 right, 2 middle), and the keys held as one bit per
 * code 0..255, the codes `key()` reads (`entelechy help key`). */
typedef struct EntelechyInput {
    double time;
    float rate;
    float pointer[2];
    float wheel[2];
    float window[2];
    float scale;
    uint32_t buttons;
    uint32_t keys[8];
} EntelechyInput;

enum EntelechyShape { ENTELECHY_ROWS = 0, ENTELECHY_GRID = 1, ENTELECHY_VOLUME = 2 };

/* A Field as it lies in memory: `count` elements of `stride` 32-bit
 * words each, element after element (rows of every layer one after
 * another), every word a float, an int or a bool (0 or 1) as `schema`
 * spells it ("f,f,f,f" for pixels: RGBA floats). A Field the scene keeps
 * narrower than words (`Grid<Vec4 as unorm8>`) is handed over widened,
 * in the session's own memory, with the same lifetime. `pixels` is 1 when the
 * layout is a picture's, so the words are an RGBA32F image of
 * `width` by `height`. */
typedef struct EntelechyField {
    const uint32_t *words;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t count;
    uint32_t stride;
    uint32_t shape;
    uint32_t pixels;
    const char *schema;
} EntelechyField;

/* The pointer a scene draws (the picture it gives `to pointer`): 8-bit
 * RGBA, `width` by `height`, from the top row down, and the pixel it
 * points with (what it gives `to pointer_tip`, else its middle). */
typedef struct EntelechyPointer {
    const uint8_t *rgba;
    uint32_t width;
    uint32_t height;
    uint32_t hotspot_x;
    uint32_t hotspot_y;
} EntelechyPointer;

/* The library's version, e.g. "0.0.1". */
const char *entelechy_version(void);

/* Opens `scene` (a `.esn` path) on `backend` ("cpu", "gpu", "jit", or
 * "best": the GPU where there is a device, else the JIT, else the
 * interpreter; NULL for "best"), with `library` an extra file or
 * directory of Kernels or NULL, reads its documents and inputs (a
 * package used by name found where every front finds it, in
 * ENTELECHY_PACKAGES, then a release's own `packages`), and
 * returns the session; NULL
 * when the backend cannot be opened or the scene does not load, with
 * the reason in entelechy_open_error(). A scene whose Kernels are partly
 * rejected still opens; ask {"op":"sync"} through entelechy_request for the
 * events. Its scene reads what it names as it first loads; from then on,
 * since entelechy_request is an agent's door, its reads are held to the
 * directories of that scene and the ones entelechy_allow allows. */
EntelechySession *entelechy_open(const char *scene, const char *backend, const char *library);

/* Files handed in: a store in memory a session reads instead of a disk,
 * for a caller that has none (a page) or keeps its scenes inside itself.
 * A path is its name with "." and ".." folded; a directory is whatever
 * some file is under. */
typedef struct EntelechyFiles EntelechyFiles;
/* A new, empty set of files. */
EntelechyFiles *entelechy_files(void);
/* Puts `len` bytes at `path`, replacing what was there. A session opened
 * on these files reads the change on its next evaluate, as it reads a
 * file changed on disk. Returns 0, or -1 for a NULL argument. */
int entelechy_files_put(EntelechyFiles *files, const char *path, const uint8_t *data, size_t len);
/* Puts a file of `len` bytes at `path` whose bytes the caller keeps. In
 * a page they are read through `entelechy.fetch` whenever the file is
 * read, so they are not held twice; elsewhere such a file cannot be
 * read. Returns 0, or -1 for a NULL argument. */
int entelechy_files_put_elsewhere(EntelechyFiles *files, const char *path, size_t len);
/* Lets go of the caller's hold; a session opened on the files keeps its
 * own. */
void entelechy_files_free(EntelechyFiles *files);
/* Opens `scene`, a path among `files`, on `backend` as entelechy_open does,
 * reading no disk. */
EntelechySession *entelechy_open_files(EntelechyFiles *files, const char *scene, const char *backend);
/* Opens a compiled binary scene bundle (`len` bytes at `data`, written by
 * Bundle::encode) on `backend` ("gpu", "jit", or "best"; NULL for "best"),
 * with no compiler or document store. NULL with entelechy_open_error() set on
 * failure. */
EntelechySession *entelechy_open_bundle(const uint8_t *data, size_t len, const char *backend);
/* As entelechy_open_bundle, taking ownership of `data` (from
 * entelechy_alloc) and freeing it once decoded so the session opens into the
 * same memory. */
EntelechySession *entelechy_open_bundle_owned(uint8_t *data, size_t len, const char *backend);
/* A scene played, as every interface that shows one plays it: the host
 * hands its person's keys, pointer, wheel and pads to the play, asks it
 * for a frame, shows the picture on its own device and plays the sound
 * on its own. What F5, Tab, F10 and Escape mean, how a pad is a
 * keyboard, what the line of frame data says and how a replay drives the
 * scene are the play's. */
/* Opens the scene a replay names, `replay` a path among `files`, with the
 * library it names, on `backend`, and plays it: its inputs drive the
 * frames at its own times until it ends. NULL with entelechy_open_error()
 * set on failure. */
EntelechySession *entelechy_open_replay(EntelechyFiles *files, const char *replay, const char *backend);
/* A key by its name ("a", "ArrowLeft", "F5", "Escape") pressed (1) or
 * released (0): F5, F6 and F10 are the play's, every other key the
 * scene's. Returns 0, or -1 when the session or the name cannot be read. */
int entelechy_play_key(EntelechySession *session, const char *name, int down);
/* Characters typed, UTF-8: a key's own text, or what an input method
 * commits. A scene reads them with typed(), apart from the keys key()
 * reads; hand both for a key that writes a character. Returns 0, or -1
 * when the session or the text cannot be read. */
int entelechy_play_typed(EntelechySession *session, const char *typed);
/* The pointer at (x, y) in the window's pixels. */
void entelechy_play_pointer(EntelechySession *session, float x, float y);
/* A pointer button, 0 left, 1 right, 2 middle, pressed or released. */
void entelechy_play_button(EntelechySession *session, int which, int down);
/* The wheel turned by (x, y) lines, y positive away from the person. */
void entelechy_play_wheel(EntelechySession *session, float x, float y);
/* A pad's button by name: "south", "east", "west", "north",
 * "left_shoulder", "right_shoulder", "left_thumb", "right_thumb",
 * "start", "select", "left", "up", "right", "down". -1 for another. */
int entelechy_play_pad(EntelechySession *session, const char *name, int down);
/* A pad's axis by name, "left_x", "left_y", "right_x", "right_y" (a
 * stick's up positive, -1 to 1), "left_trigger", "right_trigger" (0 to
 * 1). -1 for another. */
int entelechy_play_axis(EntelechySession *session, const char *name, float value);
/* How many pads are connected, what pads() reads. */
void entelechy_play_pads(EntelechySession *session, int count);
/* One frame on a window of `width` by `height` pixels at `scale`, the
 * sound device at `rate` (zero without one). Returns 0 with a frame to
 * show (the report in entelechy_report()), 1 when a replay's next frame has
 * not come, -1 with entelechy_error() set when the evaluation failed (the
 * last picture stays). */
int entelechy_play_frame(EntelechySession *session, float width, float height, float scale, float rate);
/* In a page, draws the frame's picture, and the line of frame data F10
 * shows, on the canvas whose element id is `canvas`, from the device:
 * 0 drawn, 1 not (draw entelechy_play_pixels instead), -1 on failure. */
int entelechy_play_present(EntelechySession *session, const char *canvas);
/* The frame's picture drawn on the CPU into a window of `width` by
 * `height`, letterboxed, with the line of frame data over it:
 * 0x00RRGGBB pixels, row after row, valid until the next call. NULL
 * with entelechy_error() set on failure. */
const uint32_t *entelechy_play_pixels(EntelechySession *session, uint32_t width, uint32_t height);
/* The sound to hear now, rows of one or two floats, valid until the next
 * call: the frame before's while frames overlap the device's. 0 and
 * `field` filled, 1 when there is none, -1 on failure. */
int entelechy_play_sound(EntelechySession *session, EntelechyField *field);
/* The pointer's look when it changed since the last call: 0 and `look`
 * filled with the scene's picture, valid until the next call; 2 when the
 * system's own pointer is back; 1 when it has not changed; -1 on failure. */
int entelechy_play_pointer_look(EntelechySession *session, EntelechyPointer *look);
/* The samples (per channel) the host's sound device holds and has not
 * yet played, which the next frame's queued() reads. */
void entelechy_play_sound_queued(EntelechySession *session, uint32_t samples);
/* The sound device is open, as the host describes it. */
void entelechy_play_sound_opened(EntelechySession *session, const char *description);
/* What the play said since the last call, a line each; "" for nothing. */
const char *entelechy_play_said(EntelechySession *session);
/* Whether a replay is playing, 1 or 0. */
int entelechy_play_replaying(const EntelechySession *session);
/* The Kernels still waiting their turn to prepare (a page prepares a few a
 * frame, and frames wait until the scene's are prepared); 0 once they are,
 * or where they prepare on threads. */
int entelechy_play_preparing(const EntelechySession *session);
/* A turn of loading: one sync preparing a few of the Kernels waiting. The
 * Kernels still waiting, as entelechy_play_preparing counts them; -1 on
 * failure. */
int entelechy_play_turn(EntelechySession *session);
/* Room for `size` bytes in Entelechy's memory, for a caller that shares no
 * allocator with it (a page, where Entelechy is WebAssembly) to write a
 * string or a file into before a call; NULL when there is none. */
uint8_t *entelechy_alloc(size_t size);
/* Gives back what entelechy_alloc gave, with the size it was asked for. */
void entelechy_free(uint8_t *pointer, size_t size);
/* Why the last entelechy_open on this thread returned NULL. */
const char *entelechy_open_error(void);

/* Ends the session and frees everything it holds. */
void entelechy_close(EntelechySession *session);

/* Why the last call on this session failed, or "" when it did not. */
const char *entelechy_error(const EntelechySession *session);

/* Evaluates the scene once under `input` (NULL for a default input at
 * time zero): unchanged nodes are reused, state commits only when the
 * whole evaluation succeeds. Returns 0, or -1 with entelechy_error() set. The
 * evaluation's report is in entelechy_report(). Files the scene uses are
 * re-read first when they changed, as the command line does. */
int entelechy_evaluate(EntelechySession *session, const EntelechyInput *input);

/* The last evaluation's report as JSON: what executed, what was reused,
 * what committed, the outputs with their sizes. "" before the first. */
const char *entelechy_report(const EntelechySession *session);

/* A graph output's Field after an evaluation, `name` NULL for the first
 * output. Fills `field` and returns 0, or -1 with entelechy_error() set. */
int entelechy_output(EntelechySession *session, const char *name, EntelechyField *field);

/* In a page, draws a graph output (`output` NULL for the first) on the
 * page's canvas whose element id is `canvas`, from where the GPU holds
 * it: nothing is read back. Returns 0 when it was drawn; 1 when it was
 * not (a session not on the GPU, an output that is not a grid of float
 * pixels on the device), and the caller reads it with entelechy_output; -1
 * with entelechy_error() set, as everywhere outside a page. */
int entelechy_present(EntelechySession *session, const char *canvas, const char *output);
/* A node's port after an evaluation (`port` NULL for "result"). A node
 * that fusion would inline or that no output needs must be pinned first
 * ({"op":"pin","node":"..."} through entelechy_request) and evaluated again.
 * Fills `field` and returns 0, or -1 with entelechy_error() set. */
int entelechy_node(EntelechySession *session, const char *node, const char *port, EntelechyField *field);

/* Allows the scene a directory beyond its own, as the command line's
 * --allow does: "file.read=<dir>" or "file.write=<dir>". What a protocol
 * request reads or writes is held to the scene's own directories and
 * the ones allowed here. Returns 0, or -1 with entelechy_error() set. */
int entelechy_allow(EntelechySession *session, const char *allowance);

/* One request of the creation protocol as a JSON line (see
 * docs/PROTOCOL.md: status, kernels, kernel, graph, prepare, publish,
 * param, compose, pin, read, write, reset, window, sync, ...) and its
 * response line, {"id":...,"ok":{...}} or {"id":...,"error":{...}}.
 * What it changes is recorded in the session's history as coming from
 * the protocol, through "entelechy_request". It answers in its call: a
 * `compose` or a `publish` returns once its change stands. A session
 * opened here prepares a Kernel edited on disk off the caller's thread
 * (a page, which has no thread, prepares it in place). */
const char *entelechy_request(EntelechySession *session, const char *line);

#ifdef __cplusplus
}
#endif

#endif
