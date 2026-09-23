/* Orla in a C program: open a scene, evaluate it, read the picture where
 * it lies, publish a Kernel through the protocol, and see the picture
 * change. Nothing here knows how Orla is built; it includes orla.h and
 * links the library.
 *
 *   cargo build --release -p orla-ffi
 *   cc bridges/c/hello.c -Icrates/interface/ffi/include -Ltarget/release -lorla_ffi -o /tmp/orla-hello
 *   DYLD_LIBRARY_PATH=target/release /tmp/orla-hello examples/text/scene.osn      (macOS)
 *   LD_LIBRARY_PATH=target/release /tmp/orla-hello examples/text/scene.osn        (Linux)
 */
#include <stdio.h>
#include <string.h>

#include "orla.h"

static float word_as_float(uint32_t word) {
    float f;
    memcpy(&f, &word, sizeof f);
    return f;
}

static void show_pixel(const OrlaField *picture, uint32_t x, uint32_t y) {
    const uint32_t *p = picture->words + (y * picture->width + x) * picture->stride;
    printf("pixel (%u, %u) = %.2f %.2f %.2f\n", x, y, word_as_float(p[0]), word_as_float(p[1]), word_as_float(p[2]));
}

int main(int argc, char **argv) {
    const char *scene = argc > 1 ? argv[1] : "examples/text/scene.osn";
    const char *backend = argc > 2 ? argv[2] : "cpu";
    printf("orla %s\n", orla_version());
    OrlaSession *session = orla_open(scene, backend, NULL);
    if (!session) {
        fprintf(stderr, "cannot open %s: %s\n", scene, orla_open_error());
        return 1;
    }
    OrlaInput input = {0};
    input.scale = 1.0f;
    if (orla_evaluate(session, &input) != 0) {
        fprintf(stderr, "evaluate: %s\n", orla_error(session));
        return 1;
    }
    OrlaField picture;
    if (orla_output(session, NULL, &picture) != 0) {
        fprintf(stderr, "output: %s\n", orla_error(session));
        return 1;
    }
    printf("picture %u x %u, %u words a pixel, schema %s, pixels %u\n",
           picture.width, picture.height, picture.stride, picture.schema, picture.pixels);
    show_pixel(&picture, 12, 12);
    /* The protocol through the same door: a recoloured background. */
    const char *reply = orla_request(session,
        "{\"op\":\"publish\",\"expect\":\"any\",\"source\":\"kernel scene.back@1 { requires { field.write; } output result: Image; compute { return vec4(0.3, 0.1, 0.1, 1.0); } }\"}");
    printf("publish: %.80s\n", reply);
    input.time = 0.1f;
    if (orla_evaluate(session, &input) != 0 || orla_output(session, NULL, &picture) != 0) {
        fprintf(stderr, "%s\n", orla_error(session));
        return 1;
    }
    show_pixel(&picture, 12, 12);
    printf("report: %.60s...\n", orla_report(session));
    orla_close(session);
    return 0;
}
