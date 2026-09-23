# Deferred

Deferred shading, the way a renderer that lights once per pixel does it:
the surface under every pixel is found once, by one Kernel that writes
three Fields — its colour, its normal and its distance, a G-buffer — and
the passes after it read those Fields rather than finding the surface
again.

```sh
cargo run --release -p orla-player -- examples/deferred/scene.osn
```

`deferred.surface@1` declares three outputs and gives all three at once,
by name:

```text
output albedo: Image;
output normal: Grid<Vec3>;
output depth: Grid<Float>;
compute {
    ...
    { albedo: vec4(hit.albedo, 1.0), normal: hit.normal, depth: hit.t }
}
```

`light` samples the colour and the normal; `outline` reads the depth at
the neighbours of each pixel, by address, and marks where one surface ends
in front of another. Because a pass reads the G-buffer by address, the
three Fields are written — which is what a G-buffer is. The passes over it
are pointwise to `final`, so they fuse into it, and a frame is two
Kernels: `gbuffer`, then `final+edges+lit`. A producer of several outputs
fuses whole when every reader of every output is the one consumer; here
`edges` and `lit` both read `gbuffer`, so it stays.

`turn` orbits the camera. The replay pins the G-buffer's words at a
pixel of the blue sphere and at the sky, the outline where that sphere
meets the floor, and the lit picture, on all three backends.
