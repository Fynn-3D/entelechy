/* Orla in your own process: a session as a library, its Fields read in
 * place. The same session the command line and the Player run, behind a
 * C ABI any language binds; the JSON protocol is here too, one line at a
 * time, for everything the fast path does not name.
 *
 * Every function is called from one thread at a time per session. A
 * pointer or string a call returns belongs to the session and stays
 * valid until the next call on that session, except a Field's words,
 * which stay valid until the next evaluate, request or close. */
#ifndef ORLA_H
#define ORLA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this header's contract. */
#define ORLA_ABI_VERSION 1

/* A running scene: its runtime, its documents, its state. */
typedef struct OrlaSession OrlaSession;

/* What an evaluation sees: time, the sound device's rate (zero without
 * one), the pointer in output pixels, what the wheel scrolled since the
 * last evaluation (x sideways, y positive away from the person), the
 * window in physical pixels (zero by zero for none, then the session's
 * own window applies) and its scale factor, the pointer buttons held as
 * bits (0 left, 1 right, 2 middle), and the keys held as one bit per
 * code 0..255 (letters and digits by uppercase ASCII, arrows 37 to 40,
 * space 32, enter 13, shift 16, escape 27). */
typedef struct OrlaInput {
    float time;
    float rate;
    float pointer[2];
    float wheel[2];
    float window[2];
    float scale;
    uint32_t buttons;
    uint32_t keys[8];
} OrlaInput;

enum OrlaShape { ORLA_ROWS = 0, ORLA_GRID = 1, ORLA_VOLUME = 2 };

/* A Field as it lies in memory: `count` elements of `stride` 32-bit
 * words each, element after element (rows of every layer one after
 * another), every word a float, an int or a bool (0 or 1) as `schema`
 * spells it ("f,f,f,f" for pixels: RGBA floats). A Field the scene keeps
 * narrower than words (`Grid<Vec4 as unorm8>`) is handed over widened,
 * in the session's own memory, with the same lifetime. `pixels` is 1 when the
 * layout is a picture's, so the words are an RGBA32F image of
 * `width` by `height`. */
typedef struct OrlaField {
    const uint32_t *words;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t count;
    uint32_t stride;
    uint32_t shape;
    uint32_t pixels;
    const char *schema;
} OrlaField;

/* The library's version, e.g. "0.1.0". */
const char *orla_version(void);

/* Opens `scene` (a `.osn` path) on `backend` ("cpu", "gpu", "jit", or
 * "best": the GPU where there is a device, else the JIT, else the
 * interpreter; NULL for "best"), with `library` an extra file or
 * directory of Kernels or NULL, reads its documents and inputs, and
 * returns the session; NULL
 * when the backend cannot be opened or the scene does not load, with
 * the reason in orla_open_error(). A scene whose Kernels are partly
 * rejected still opens; ask {"op":"sync"} through orla_request for the
 * events. */
OrlaSession *orla_open(const char *scene, const char *backend, const char *library);

/* Files handed in: a store in memory a session reads instead of a disk,
 * for a caller that has none (a page) or keeps its scenes inside itself.
 * A path is its name with "." and ".." folded; a directory is whatever
 * some file is under. */
typedef struct OrlaFiles OrlaFiles;
/* A new, empty set of files. */
OrlaFiles *orla_files(void);
/* Puts `len` bytes at `path`, replacing what was there. A session opened
 * on these files reads the change on its next evaluate, as it reads a
 * file changed on disk. Returns 0, or -1 for a NULL argument. */
int orla_files_put(OrlaFiles *files, const char *path, const uint8_t *data, size_t len);
/* Lets go of the caller's hold; a session opened on the files keeps its
 * own. */
void orla_files_free(OrlaFiles *files);
/* Opens `scene`, a path among `files`, on `backend` as orla_open does,
 * reading no disk. */
OrlaSession *orla_open_files(OrlaFiles *files, const char *scene, const char *backend);
/* A scene played, as every interface that shows one plays it: the host
 * hands its person's keys, pointer, wheel and pads to the play, asks it
 * for a frame, shows the picture on its own device and plays the sound
 * on its own. What F5, Tab, F10 and Escape mean, how a pad is a
 * keyboard, what the line of frame data says and how a replay drives the
 * scene are the play's. */
/* Opens the scene a replay names, `replay` a path among `files`, with the
 * library it names, on `backend`, and plays it: its inputs drive the
 * frames at its own times until it ends. NULL with orla_open_error()
 * set on failure. */
OrlaSession *orla_open_replay(OrlaFiles *files, const char *replay, const char *backend);
/* A key by its name ("a", "ArrowLeft", "F5", "Escape") pressed (1) or
 * released (0). Returns 1 when it asks the host to stop (Escape), 0
 * otherwise. */
int orla_play_key(OrlaSession *session, const char *name, int down);
/* The pointer at (x, y) in the window's pixels. */
void orla_play_pointer(OrlaSession *session, float x, float y);
/* A pointer button, 0 left, 1 right, 2 middle, pressed or released. */
void orla_play_button(OrlaSession *session, int which, int down);
/* The wheel turned by (x, y) lines, y positive away from the person. */
void orla_play_wheel(OrlaSession *session, float x, float y);
/* A pad's button by name: "south", "east", "west", "north",
 * "left_shoulder", "right_shoulder", "left_thumb", "right_thumb",
 * "start", "select", "left", "up", "right", "down". -1 for another. */
int orla_play_pad(OrlaSession *session, const char *name, int down);
/* A pad's axis by name, "left_x", "left_y", "right_x", "right_y" (a
 * stick's up positive, -1 to 1), "left_trigger", "right_trigger" (0 to
 * 1). -1 for another. */
int orla_play_axis(OrlaSession *session, const char *name, float value);
/* One frame on a window of `width` by `height` pixels at `scale`, the
 * sound device at `rate` (zero without one). Returns 0 with a frame to
 * show (the report in orla_report()), 1 when a replay's next frame has
 * not come, -1 with orla_error() set when the evaluation failed (the
 * last picture stays). */
int orla_play_frame(OrlaSession *session, float width, float height, float scale, float rate);
/* In a page, draws the frame's picture, and the line of frame data F10
 * shows, on the canvas whose element id is `canvas`, from the device:
 * 0 drawn, 1 not (draw orla_play_pixels instead), -1 on failure. */
int orla_play_present(OrlaSession *session, const char *canvas);
/* The frame's picture drawn on the CPU into a window of `width` by
 * `height`, letterboxed, with the line of frame data over it:
 * 0x00RRGGBB pixels, row after row, valid until the next call. NULL
 * with orla_error() set on failure. */
const uint32_t *orla_play_pixels(OrlaSession *session, uint32_t width, uint32_t height);
/* The frame's sound, rows of one or two floats, read in place: 0 and
 * `field` filled, 1 when there is none, -1 on failure. */
int orla_play_sound(OrlaSession *session, OrlaField *field);
/* The sound device is open, as the host describes it. */
void orla_play_sound_opened(OrlaSession *session, const char *description);
/* What the play said since the last call, a line each; "" for nothing. */
const char *orla_play_said(OrlaSession *session);
/* Whether a replay is playing, 1 or 0. */
int orla_play_replaying(const OrlaSession *session);
/* Room for `size` bytes in Orla's memory, for a caller that shares no
 * allocator with it (a page, where Orla is WebAssembly) to write a
 * string or a file into before a call; NULL when there is none. */
uint8_t *orla_alloc(size_t size);
/* Gives back what orla_alloc gave, with the size it was asked for. */
void orla_free(uint8_t *pointer, size_t size);
/* Why the last orla_open on this thread returned NULL. */
const char *orla_open_error(void);

/* Ends the session and frees everything it holds. */
void orla_close(OrlaSession *session);

/* Why the last call on this session failed, or "" when it did not. */
const char *orla_error(const OrlaSession *session);

/* Evaluates the scene once under `input` (NULL for a default input at
 * time zero): unchanged nodes are reused, state commits only when the
 * whole evaluation succeeds. Returns 0, or -1 with orla_error() set. The
 * evaluation's report is in orla_report(). Files the scene uses are
 * re-read first when they changed, as the command line does. */
int orla_evaluate(OrlaSession *session, const OrlaInput *input);

/* The last evaluation's report as JSON: what executed, what was reused,
 * what committed, the outputs with their sizes. "" before the first. */
const char *orla_report(const OrlaSession *session);

/* A graph output's Field after an evaluation, `name` NULL for the first
 * output. Fills `field` and returns 0, or -1 with orla_error() set. */
int orla_output(OrlaSession *session, const char *name, OrlaField *field);

/* In a page, draws a graph output (`output` NULL for the first) on the
 * page's canvas whose element id is `canvas`, from where the GPU holds
 * it: nothing is read back. Returns 0 when it was drawn; 1 when it was
 * not (a session not on the GPU, an output that is not a grid of float
 * pixels on the device), and the caller reads it with orla_output; -1
 * with orla_error() set, as everywhere outside a page. */
int orla_present(OrlaSession *session, const char *canvas, const char *output);
/* A node's port after an evaluation (`port` NULL for "result"). A node
 * that fusion would inline or that no output needs must be pinned first
 * ({"op":"pin","node":"..."} through orla_request) and evaluated again.
 * Fills `field` and returns 0, or -1 with orla_error() set. */
int orla_node(OrlaSession *session, const char *node, const char *port, OrlaField *field);

/* One request of the creation protocol as a JSON line (see
 * docs/PROTOCOL.md: status, kernels, kernel, graph, prepare, publish,
 * param, compose, pin, read, write, reset, window, sync, ...) and its
 * response line, {"id":...,"ok":{...}} or {"id":...,"error":{...}}.
 * What it changes is recorded in the session's history as coming from
 * the protocol, through "orla_request". */
const char *orla_request(OrlaSession *session, const char *line);

#ifdef __cplusplus
}
#endif

#endif
