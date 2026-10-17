/* Entelechy in a C program: open a scene, evaluate it, read the picture where
 * it lies, publish a Kernel through the protocol, and see the picture
 * change. Nothing here knows how Entelechy is built; it includes entelechy.h and
 * links the library.
 *
 *   cargo build --release -p entelechy-ffi
 *   cc bridges/c/hello.c -Icrates/interface/ffi/include -Ltarget/release -lentelechy_ffi -o /tmp/entelechy-hello
 *   DYLD_LIBRARY_PATH=target/release /tmp/entelechy-hello examples/text/scene.esn      (macOS)
 *   LD_LIBRARY_PATH=target/release /tmp/entelechy-hello examples/text/scene.esn        (Linux)
 */
#include <stdio.h>
#include <string.h>

#include "entelechy.h"

static float word_as_float(uint32_t word) {
    float f;
    memcpy(&f, &word, sizeof f);
    return f;
}

static void show_pixel(const EntelechyField *picture, uint32_t x, uint32_t y) {
    const uint32_t *p = picture->words + (y * picture->width + x) * picture->stride;
    printf("pixel (%u, %u) = %.2f %.2f %.2f\n", x, y, word_as_float(p[0]), word_as_float(p[1]), word_as_float(p[2]));
}

int main(int argc, char **argv) {
    const char *scene = argc > 1 ? argv[1] : "examples/text/scene.esn";
    const char *backend = argc > 2 ? argv[2] : "cpu";
    printf("entelechy %s\n", entelechy_version());
    EntelechySession *session = entelechy_open(scene, backend, NULL);
    if (!session) {
        fprintf(stderr, "cannot open %s: %s\n", scene, entelechy_open_error());
        return 1;
    }
    EntelechyInput input = {0};
    input.scale = 1.0f;
    if (entelechy_evaluate(session, &input) != 0) {
        fprintf(stderr, "evaluate: %s\n", entelechy_error(session));
        return 1;
    }
    EntelechyField picture;
    if (entelechy_output(session, NULL, &picture) != 0) {
        fprintf(stderr, "output: %s\n", entelechy_error(session));
        return 1;
    }
    printf("picture %u x %u, %u words a pixel, schema %s, pixels %u\n",
           picture.width, picture.height, picture.stride, picture.schema, picture.pixels);
    show_pixel(&picture, 12, 9);
    /* The protocol through the same door: a recolored background. */
    const char *reply = entelechy_request(session,
        "{\"op\":\"publish\",\"expect\":\"any\",\"source\":\"kernel scene.back@1 { requires { field.write; } output result: Image; compute { return vec4(0.3, 0.1, 0.1, 1.0); } }\"}");
    printf("publish: %.80s\n", reply);
    input.time = 0.1;
    if (entelechy_evaluate(session, &input) != 0 || entelechy_output(session, NULL, &picture) != 0) {
        fprintf(stderr, "%s\n", entelechy_error(session));
        return 1;
    }
    show_pixel(&picture, 12, 9);
    printf("report: %.60s...\n", entelechy_report(session));
    entelechy_close(session);
    return 0;
}
