# Entelechy from C

`hello.c` is a C program that loads Entelechy as a library in its own process
through the C ABI in [`crates/interface/ffi/include/entelechy.h`](../../crates/interface/ffi/include/entelechy.h):
it opens a scene, evaluates it, reads the picture where it lies in the
Host's memory (RGBA floats, no copy), publishes a Kernel through the
protocol, and reads the picture again. It knows nothing about how Entelechy
is built; it includes one header and links one library.

```sh
cargo build --release -p entelechy-ffi
cc bridges/c/hello.c -Icrates/interface/ffi/include -Ltarget/release -lentelechy_ffi -o /tmp/entelechy-hello
DYLD_LIBRARY_PATH=target/release /tmp/entelechy-hello examples/text/scene.esn jit    # macOS
LD_LIBRARY_PATH=target/release /tmp/entelechy-hello examples/text/scene.esn jit      # Linux
```

```text
entelechy 0.1.0
picture 192 x 88, 4 words a pixel, schema f,f,f,f, pixels 1
pixel (12, 9) = 0.09 0.10 0.13
publish: {"id":null,"ok":{"generation":12,"id":"scene.back@1"}}
pixel (12, 9) = 0.30 0.10 0.10
```

The static library `libentelechy_ffi.a` links the same way without the
runtime path. Any language with a C foreign function interface (Python's
ctypes, C#, Zig, a Godot GDExtension) binds the same ten functions; the
header documents what each pointer's lifetime is.
