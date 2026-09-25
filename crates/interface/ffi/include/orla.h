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

/* Opens `scene` (a `.osn` path) on `backend` ("cpu", "gpu" or "jit";
 * NULL for cpu), with `library` an extra file or directory of Kernels
 * or NULL, reads its documents and inputs, and returns the session; NULL
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
 * response line, {"id":...,"ok":{...}} or {"id":...,"error":{...}}. */
const char *orla_request(OrlaSession *session, const char *line);

#ifdef __cplusplus
}
#endif

#endif
