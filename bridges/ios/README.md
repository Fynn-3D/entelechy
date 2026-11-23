# @@slot:title@@ (iOS)

Exported by `entelechy-export` as a self-contained Xcode project over the
bundle-only C ABI (`entelechy.h` + `scene.bundle`). On iOS, Kernels run on
Metal (`wgpu`) with the interpreter beside it, so no runtime executable memory
(`mmap(PROT_EXEC)`) is requested. Any bundled save slots (`entelechy_save_*`)
persist automatically under `Application Support/<bundle-id>/saves/`.

## Build and Run

1. Place `libentelechy_ffi.a` for `aarch64-apple-ios` in `lib/` (if `--lib`
   did not already copy one):

```sh
rustup target add aarch64-apple-ios
cargo build --profile small --no-default-features -p entelechy-ffi --target aarch64-apple-ios
mkdir -p lib
cp target/aarch64-apple-ios/small/libentelechy_ffi.a lib/
```

2. Open `@@slot:executable@@.xcodeproj` in Xcode, select your Apple Developer
   signing team under **Signing & Capabilities**, and run on an iPhone or iPad
   or archive an `.ipa`.
