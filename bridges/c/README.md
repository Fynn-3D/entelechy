# Orla from C

`hello.c` is a C program that loads Orla as a library in its own process
through the C ABI in [`crates/ffi/include/orla.h`](../../crates/ffi/include/orla.h):
it opens a scene, evaluates it, reads the picture where it lies in the
Host's memory (RGBA floats, no copy), publishes a Kernel through the
protocol, and reads the picture again. It knows nothing about how Orla
is built; it includes one header and links one library.

```sh
cargo build --release -p orla-ffi
cc bridges/c/hello.c -Icrates/ffi/include -Ltarget/release -lorla_ffi -o /tmp/orla-hello
DYLD_LIBRARY_PATH=target/release /tmp/orla-hello examples/text/scene.osn jit    # macOS
LD_LIBRARY_PATH=target/release /tmp/orla-hello examples/text/scene.osn jit      # Linux
```

```text
orla 0.1.0
picture 192 x 88, 4 words a pixel, schema f,f,f,f, pixels 1
pixel (12, 12) = 0.09 0.10 0.13
publish: {"id":null,"ok":{"generation":12,"id":"scene.back@1"}}
pixel (12, 12) = 0.30 0.10 0.10
```

The static library `liborla_ffi.a` links the same way without the
runtime path. Any language with a C foreign function interface (Python's
ctypes, C#, Zig, a Godot GDExtension) binds the same ten functions; the
header documents what each pointer's lifetime is.
