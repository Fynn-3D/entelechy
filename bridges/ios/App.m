/* Entelechy iOS application host: opens the bundled scene (`scene.bundle`)
 * through `entelechy.h`, drives frames on `CADisplayLink`, presents pixels on
 * the view's layer, streams stereo float audio through `AVAudioEngine`, and
 * forwards touch and hardware keyboard input. */

#import <AVFAudio/AVFAudio.h>
#import <CoreGraphics/CoreGraphics.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>
#include <stdatomic.h>
#include <string.h>

#include "entelechy.h"

#define RING_FRAMES 16384u
#define RING_MASK (RING_FRAMES - 1u)

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

@interface EntelechyViewController : UIViewController {
    EntelechySession *_session;
    CADisplayLink *_link;
    AVAudioEngine *_audioEngine;
    AVAudioSourceNode *_sourceNode;
    AudioRing _ring;
    double _sampleRate;
    BOOL _audioStarted;
    CGPoint _lastTouch;
    BOOL _hasTouch;
}
@end

@implementation EntelechyViewController

- (BOOL)prefersStatusBarHidden {
    return YES;
}

- (BOOL)prefersHomeIndicatorAutoHidden {
    return YES;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor blackColor];
    self.view.multipleTouchEnabled = NO;
    self.view.layer.magnificationFilter = kCAFilterNearest;

    NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"scene" ofType:@"bundle"];
    NSData *bundleData = bundlePath ? [NSData dataWithContentsOfFile:bundlePath] : nil;
    if (!bundleData || bundleData.length == 0) {
        NSLog(@"[Entelechy] missing scene.bundle in main bundle");
        return;
    }

    const char *backend = "@@slot:backend@@";
    _session = entelechy_open_bundle((const uint8_t *)bundleData.bytes, (size_t)bundleData.length, backend);
    if (!_session) {
        NSLog(@"[Entelechy] cannot open bundle: %s", entelechy_open_error());
        return;
    }

    _link = [CADisplayLink displayLinkWithTarget:self selector:@selector(stepFrame)];
    [_link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
}

- (void)startAudioIfNeeded {
    if (_audioStarted || !_session) {
        return;
    }
    _audioStarted = YES;
    NSError *err = nil;
    AVAudioSession *av = [AVAudioSession sharedInstance];
    [av setCategory:AVAudioSessionCategoryPlayback error:&err];
    [av setActive:YES error:&err];

    _audioEngine = [[AVAudioEngine alloc] init];
    AVAudioFormat *outFmt = [_audioEngine.outputNode inputFormatForBus:0];
    double sr = outFmt.sampleRate > 0.0 ? outFmt.sampleRate : 48000.0;
    AVAudioFormat *stereoFmt = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:sr channels:2];
    AudioRing *ringPtr = &_ring;

    _sourceNode = [[AVAudioSourceNode alloc]
        initWithFormat:stereoFmt
           renderBlock:^OSStatus(BOOL *isSilence, const AudioTimeStamp *timestamp,
                                 AVAudioFrameCount frameCount, AudioBufferList *outputData) {
               (void)timestamp;
               float *left = (float *)outputData->mBuffers[0].mData;
               float *right = outputData->mNumberBuffers > 1
                                  ? (float *)outputData->mBuffers[1].mData
                                  : NULL;
               uint32_t r = atomic_load_explicit(&ringPtr->read_pos, memory_order_relaxed);
               uint32_t w = atomic_load_explicit(&ringPtr->write_pos, memory_order_acquire);
               BOOL any = NO;
               for (AVAudioFrameCount i = 0; i < frameCount; i++) {
                   float l = 0.0f, rt = 0.0f;
                   if (r != w) {
                       uint32_t slot = (r & RING_MASK) * 2u;
                       l = ringPtr->samples[slot];
                       rt = ringPtr->samples[slot + 1u];
                       r++;
                       any = YES;
                   }
                   if (right) {
                       left[i] = l;
                       right[i] = rt;
                   } else {
                       left[i * 2u] = l;
                       left[i * 2u + 1u] = rt;
                   }
               }
               atomic_store_explicit(&ringPtr->read_pos, r, memory_order_release);
               *isSilence = !any;
               return noErr;
           }];

    [_audioEngine attachNode:_sourceNode];
    [_audioEngine connect:_sourceNode to:_audioEngine.mainMixerNode format:stereoFmt];
    if ([_audioEngine startAndReturnError:&err]) {
        _sampleRate = sr;
        entelechy_play_sound_opened(_session, "AVAudioEngine");
    } else {
        NSLog(@"[Entelechy] audio start failed: %@", err);
    }
}

- (void)stepFrame {
    if (!_session) {
        return;
    }
    CGFloat scale = self.view.contentScaleFactor > 0.0 ? self.view.contentScaleFactor : [UIScreen mainScreen].scale;
    CGSize bounds = self.view.bounds.size;
    uint32_t w = (uint32_t)(bounds.width * scale);
    uint32_t h = (uint32_t)(bounds.height * scale);
    if (w == 0 || h == 0) {
        return;
    }

    if (_sampleRate > 0.0) {
        entelechy_play_sound_queued(_session, ring_queued(&_ring));
    }
    int status = entelechy_play_frame(_session, (float)w, (float)h, (float)scale, (float)_sampleRate);
    if (status != 0) {
        return;
    }

    EntelechyField sound = {0};
    if (entelechy_play_sound(_session, &sound) == 0 && sound.count > 0 && sound.stride > 0) {
        if (!_audioStarted) {
            [self startAudioIfNeeded];
        }
        ring_push(&_ring, (const float *)sound.words, sound.count, sound.stride);
    }

    const uint32_t *pixels = entelechy_play_pixels(_session, w, h);
    if (!pixels) {
        return;
    }
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, pixels, (size_t)w * (size_t)h * 4u, NULL);
    CGImageRef image = CGImageCreate(
        w, h, 8, 32, (size_t)w * 4u, cs,
        kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little,
        provider, NULL, false, kCGRenderingIntentDefault);
    if (image) {
        self.view.layer.contents = (__bridge id)image;
        CGImageRelease(image);
    }
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(cs);
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    (void)event;
    if (!_session) return;
    [self startAudioIfNeeded];
    UITouch *touch = [touches anyObject];
    CGFloat scale = self.view.contentScaleFactor > 0.0 ? self.view.contentScaleFactor : 1.0;
    CGPoint pt = [touch locationInView:self.view];
    _lastTouch = pt;
    _hasTouch = YES;
    if (strcmp("@@slot:pointer@@", "lock") != 0) {
        entelechy_play_pointer(_session, (float)(pt.x * scale), (float)(pt.y * scale));
    }
    entelechy_play_button(_session, 0, 1);
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    (void)event;
    if (!_session || !_hasTouch) return;
    UITouch *touch = [touches anyObject];
    CGFloat scale = self.view.contentScaleFactor > 0.0 ? self.view.contentScaleFactor : 1.0;
    CGPoint pt = [touch locationInView:self.view];
    if (strcmp("@@slot:pointer@@", "lock") == 0) {
        entelechy_play_pointer_moved(
            _session,
            (float)((pt.x - _lastTouch.x) * scale),
            (float)((pt.y - _lastTouch.y) * scale));
    } else {
        entelechy_play_pointer(_session, (float)(pt.x * scale), (float)(pt.y * scale));
    }
    _lastTouch = pt;
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    (void)touches;
    (void)event;
    _hasTouch = NO;
    if (_session) {
        entelechy_play_button(_session, 0, 0);
    }
}

- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    (void)touches;
    (void)event;
    _hasTouch = NO;
    if (_session) {
        entelechy_play_release(_session);
    }
}

- (void)dealloc {
    [_link invalidate];
    [_audioEngine stop];
    if (_session) {
        entelechy_close(_session);
        _session = NULL;
    }
}

@end

@interface EntelechyAppDelegate : UIResponder <UIApplicationDelegate>
@property (strong, nonatomic) UIWindow *window;
@end

@implementation EntelechyAppDelegate
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    (void)application;
    (void)launchOptions;
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    self.window.rootViewController = [[EntelechyViewController alloc] init];
    [self.window makeKeyAndVisible];
    return YES;
}
@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([EntelechyAppDelegate class]));
    }
}
