/* Entelechy Android NativeActivity host: loads `scene.bundle` from APK assets,
 * steps `entelechy_play_frame` on a dedicated render thread, presents pixels to
 * `ANativeWindow`, streams stereo float audio through `AAudio`, and maps touch
 * and key events to `entelechy_play_*`. */

#include <aaudio/AAudio.h>
#include <android/asset_manager.h>
#include <android/input.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <android/native_window.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "entelechy.h"

#define RING_FRAMES 16384u
#define RING_MASK (RING_FRAMES - 1u)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "Entelechy", __VA_ARGS__)

typedef struct AudioRing {
    float samples[RING_FRAMES * 2u];
    _Atomic uint32_t read_pos;
    _Atomic uint32_t write_pos;
} AudioRing;

static uint32_t ring_queued(const AudioRing *ring) {
    uint32_t w = atomic_load_explicit(&ring->write_pos, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&ring->read_pos, memory_order_acquire);
    return (w - r) & RING_MASK;
}

static void ring_push(AudioRing *ring, const float *words, uint32_t frames, uint32_t channels) {
    uint32_t w = atomic_load_explicit(&ring->write_pos, memory_order_relaxed);
    uint32_t r = atomic_load_explicit(&ring->read_pos, memory_order_acquire);
    for (uint32_t i = 0; i < frames; i++) {
        if (((w - r) & RING_MASK) >= RING_FRAMES - 1u) {
            break;
        }
        uint32_t slot = (w & RING_MASK) * 2u;
        float left = words[i * channels];
        float right = channels > 1u ? words[i * channels + 1u] : left;
        ring->samples[slot] = left;
        ring->samples[slot + 1u] = right;
        w++;
    }
    atomic_store_explicit(&ring->write_pos, w, memory_order_release);
}

static aaudio_data_callback_result_t audio_callback(
    AAudioStream *stream,
    void *user_data,
    void *audio_data,
    int32_t num_frames) {
    (void)stream;
    AudioRing *ring = (AudioRing *)user_data;
    float *out = (float *)audio_data;
    uint32_t r = atomic_load_explicit(&ring->read_pos, memory_order_relaxed);
    uint32_t w = atomic_load_explicit(&ring->write_pos, memory_order_acquire);
    for (int32_t i = 0; i < num_frames; i++) {
        if (r != w) {
            uint32_t slot = (r & RING_MASK) * 2u;
            out[i * 2] = ring->samples[slot];
            out[i * 2 + 1] = ring->samples[slot + 1u];
            r++;
        } else {
            out[i * 2] = 0.0f;
            out[i * 2 + 1] = 0.0f;
        }
    }
    atomic_store_explicit(&ring->read_pos, r, memory_order_release);
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

typedef struct AppState {
    ANativeActivity *activity;
    pthread_mutex_t lock;
    pthread_t thread;
    ANativeWindow *window;
    AInputQueue *queue;
    AAudioStream *audio;
    AudioRing ring;
    float sample_rate;
    bool running;
    bool audio_tried;
    float last_x;
    float last_y;
    bool touching;
} AppState;

static void open_audio_if_needed(AppState *app, EntelechySession *session) {
    if (app->audio_tried) {
        return;
    }
    app->audio_tried = true;
    AAudioStreamBuilder *builder = NULL;
    if (AAudio_createStreamBuilder(&builder) != AAUDIO_OK || !builder) {
        return;
    }
    AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setChannelCount(builder, 2);
    AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setDataCallback(builder, audio_callback, &app->ring);
    if (AAudioStreamBuilder_openStream(builder, &app->audio) == AAUDIO_OK && app->audio) {
        app->sample_rate = (float)AAudioStream_getSampleRate(app->audio);
        AAudioStream_requestStart(app->audio);
        entelechy_play_sound_opened(session, "AAudio");
    }
    AAudioStreamBuilder_delete(builder);
}

static void drain_input(AppState *app, EntelechySession *session, AInputQueue *queue) {
    AInputEvent *event = NULL;
    while (AInputQueue_getEvent(queue, &event) >= 0) {
        if (AInputQueue_preDispatchEvent(queue, event)) {
            continue;
        }
        int32_t handled = 0;
        int32_t type = AInputEvent_getType(event);
        if (type == AINPUT_EVENT_TYPE_MOTION) {
            int32_t action = AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
            float x = AMotionEvent_getX(event, 0);
            float y = AMotionEvent_getY(event, 0);
            if (action == AMOTION_EVENT_ACTION_DOWN) {
                app->last_x = x;
                app->last_y = y;
                app->touching = true;
                if (strcmp("@@slot:pointer@@", "lock") != 0) {
                    entelechy_play_pointer(session, x, y);
                }
                entelechy_play_button(session, 0, 1);
                handled = 1;
            } else if (action == AMOTION_EVENT_ACTION_MOVE && app->touching) {
                if (strcmp("@@slot:pointer@@", "lock") == 0) {
                    entelechy_play_pointer_moved(session, x - app->last_x, y - app->last_y);
                } else {
                    entelechy_play_pointer(session, x, y);
                }
                app->last_x = x;
                app->last_y = y;
                handled = 1;
            } else if (action == AMOTION_EVENT_ACTION_UP || action == AMOTION_EVENT_ACTION_CANCEL) {
                app->touching = false;
                entelechy_play_button(session, 0, 0);
                handled = 1;
            }
        }
        AInputQueue_finishEvent(queue, event, handled);
    }
}

static void ensure_parent_dirs(char *path) {
    for (char *p = path + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(path, 0700);
            *p = '/';
        }
    }
}

static void preload_saves(AppState *app, EntelechySession *session, char **slots, size_t count) {
    if (!app->activity->internalDataPath || count == 0) return;
    char path[1024];
    for (size_t i = 0; i < count; i++) {
        snprintf(path, sizeof(path), "%s/saves/%s", app->activity->internalDataPath, slots[i]);
        FILE *fp = fopen(path, "rb");
        if (!fp) continue;
        if (fseek(fp, 0, SEEK_END) == 0) {
            long sz = ftell(fp);
            if (sz > 0 && fseek(fp, 0, SEEK_SET) == 0) {
                uint8_t *buf = (uint8_t *)malloc((size_t)sz);
                if (buf) {
                    if (fread(buf, 1, (size_t)sz, fp) == (size_t)sz) {
                        entelechy_save_load(session, slots[i], buf, (size_t)sz);
                    }
                    free(buf);
                }
            }
        }
        fclose(fp);
    }
}

static void flush_saves(AppState *app, EntelechySession *session, char **slots, size_t count) {
    if (!app->activity->internalDataPath || count == 0) return;
    char path[1024];
    char tmp[1040];
    for (size_t i = 0; i < count; i++) {
        size_t len = 0;
        const uint8_t *bytes = entelechy_save_take(session, i, &len);
        if (!bytes) continue;
        snprintf(path, sizeof(path), "%s/saves/%s", app->activity->internalDataPath, slots[i]);
        ensure_parent_dirs(path);
        snprintf(tmp, sizeof(tmp), "%s.writing", path);
        FILE *fp = fopen(tmp, "wb");
        if (!fp) continue;
        bool ok = (len == 0) || (fwrite(bytes, 1, len, fp) == len);
        fclose(fp);
        if (ok) {
            rename(tmp, path);
        } else {
            remove(tmp);
        }
    }
}

static void *render_loop(void *arg) {
    AppState *app = (AppState *)arg;
    AAsset *asset = AAssetManager_open(app->activity->assetManager, "scene.bundle", AASSET_MODE_BUFFER);
    if (!asset) {
        LOGE("missing assets/scene.bundle");
        return NULL;
    }
    size_t len = (size_t)AAsset_getLength(asset);
    const uint8_t *buf = (const uint8_t *)AAsset_getBuffer(asset);
    uint8_t *copy = (uint8_t *)malloc(len);
    if (!copy) {
        AAsset_close(asset);
        return NULL;
    }
    memcpy(copy, buf, len);
    AAsset_close(asset);

    EntelechySession *session = entelechy_open_bundle(copy, len, "@@slot:backend@@");
    free(copy);
    if (!session) {
        LOGE("cannot open bundle: %s", entelechy_open_error());
        return NULL;
    }

    size_t save_count = 0;
    char *save_slots[32] = {0};
    for (; save_count < 32; save_count++) {
        const char *raw = entelechy_save_slot(session, save_count);
        if (!raw) break;
        save_slots[save_count] = strdup(raw);
    }
    preload_saves(app, session, save_slots, save_count);

    struct timespec sleep_ts = {0, 2 * 1000 * 1000};
    while (true) {
        pthread_mutex_lock(&app->lock);
        bool running = app->running;
        ANativeWindow *window = app->window;
        AInputQueue *queue = app->queue;
        pthread_mutex_unlock(&app->lock);
        if (!running) {
            break;
        }
        if (queue) {
            drain_input(app, session, queue);
        }
        if (!window) {
            nanosleep(&sleep_ts, NULL);
            continue;
        }
        int32_t w = ANativeWindow_getWidth(window);
        int32_t h = ANativeWindow_getHeight(window);
        if (w <= 0 || h <= 0) {
            nanosleep(&sleep_ts, NULL);
            continue;
        }
        ANativeWindow_setBuffersGeometry(window, w, h, WINDOW_FORMAT_RGBX_8888);
        if (app->sample_rate > 0.0f) {
            entelechy_play_sound_queued(session, ring_queued(&app->ring));
        }
        if (entelechy_play_frame(session, (float)w, (float)h, 1.0f, app->sample_rate) == 0) {
            flush_saves(app, session, save_slots, save_count);
            EntelechyField sound = {0};
            if (entelechy_play_sound(session, &sound) == 0 && sound.count > 0 && sound.stride > 0) {
                open_audio_if_needed(app, session);
                ring_push(&app->ring, (const float *)sound.words, sound.count, sound.stride);
            }
            const uint32_t *src = entelechy_play_pixels(session, (uint32_t)w, (uint32_t)h);
            ANativeWindow_Buffer dst_buf;
            if (src && ANativeWindow_lock(window, &dst_buf, NULL) == 0) {
                uint32_t *dst = (uint32_t *)dst_buf.bits;
                for (int32_t y = 0; y < h && y < dst_buf.height; y++) {
                    const uint32_t *srow = src + (size_t)y * (size_t)w;
                    uint32_t *drow = dst + (size_t)y * (size_t)dst_buf.stride;
                    for (int32_t x = 0; x < w && x < dst_buf.width; x++) {
                        uint32_t p = srow[x];
                        uint32_t r = (p >> 16) & 0xffu;
                        uint32_t g = (p >> 8) & 0xffu;
                        uint32_t b = p & 0xffu;
                        drow[x] = 0xff000000u | (b << 16) | (g << 8) | r;
                    }
                }
                ANativeWindow_unlockAndPost(window);
            }
        }
    }

    if (app->audio) {
        AAudioStream_requestStop(app->audio);
        AAudioStream_close(app->audio);
        app->audio = NULL;
    }
    if (save_count > 0) {
        entelechy_save_flush(session);
        flush_saves(app, session, save_slots, save_count);
        for (size_t i = 0; i < save_count; i++) free(save_slots[i]);
    }
    entelechy_close(session);
    return NULL;
}

static void on_window_created(ANativeActivity *activity, ANativeWindow *window) {
    AppState *app = (AppState *)activity->instance;
    pthread_mutex_lock(&app->lock);
    app->window = window;
    pthread_mutex_unlock(&app->lock);
}

static void on_window_destroyed(ANativeActivity *activity, ANativeWindow *window) {
    (void)window;
    AppState *app = (AppState *)activity->instance;
    pthread_mutex_lock(&app->lock);
    app->window = NULL;
    pthread_mutex_unlock(&app->lock);
}

static void on_input_created(ANativeActivity *activity, AInputQueue *queue) {
    AppState *app = (AppState *)activity->instance;
    pthread_mutex_lock(&app->lock);
    app->queue = queue;
    pthread_mutex_unlock(&app->lock);
}

static void on_input_destroyed(ANativeActivity *activity, AInputQueue *queue) {
    (void)queue;
    AppState *app = (AppState *)activity->instance;
    pthread_mutex_lock(&app->lock);
    app->queue = NULL;
    pthread_mutex_unlock(&app->lock);
}

static void on_destroy(ANativeActivity *activity) {
    AppState *app = (AppState *)activity->instance;
    pthread_mutex_lock(&app->lock);
    app->running = false;
    pthread_mutex_unlock(&app->lock);
    pthread_join(app->thread, NULL);
    pthread_mutex_destroy(&app->lock);
    free(app);
}

JNIEXPORT void ANativeActivity_onCreate(
    ANativeActivity *activity,
    void *saved_state,
    size_t saved_state_size) {
    (void)saved_state;
    (void)saved_state_size;
    AppState *app = (AppState *)calloc(1, sizeof(AppState));
    app->activity = activity;
    app->running = true;
    pthread_mutex_init(&app->lock, NULL);
    activity->instance = app;
    activity->callbacks->onNativeWindowCreated = on_window_created;
    activity->callbacks->onNativeWindowDestroyed = on_window_destroyed;
    activity->callbacks->onInputQueueCreated = on_input_created;
    activity->callbacks->onInputQueueDestroyed = on_input_destroyed;
    activity->callbacks->onDestroy = on_destroy;
    pthread_create(&app->thread, NULL, render_loop, app);
}
