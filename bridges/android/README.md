# @@slot:title@@ (Android)

Exported by `entelechy-export` as a self-contained Android NDK (`NativeActivity`)
project over the bundle-only C ABI (`entelechy.h` + `assets/scene.bundle`).

## Build and Run

1. Place `libentelechy_ffi.so` for `arm64-v8a` under
   `app/src/main/jniLibs/arm64-v8a/` (if `--lib` did not already copy one):

```sh
rustup target add aarch64-linux-android
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --profile small --no-default-features -p entelechy-ffi
```

2. Build an `.apk` with Gradle or open this directory in Android Studio:

```sh
gradle assembleRelease
```
