# Deferred

Deferred shading. `deferred.surface@1` finds the surface under each pixel
once and writes three outputs, a G-buffer, by name:

```text
output albedo: Image;
output normal: Grid<Vec3>;
output depth: Grid<Float>;
compute {
    ...
    { albedo: vec4(hit.albedo, 1.0), normal: hit.normal, depth: hit.t }
}
```

`light` samples the colour and normal; `outline` reads the depth at each
pixel's neighbours by address and marks silhouette edges. `turn` orbits
the camera. The replay pins G-buffer values, the outline, and the lit
picture on all three backends.

```sh
cargo run --release -p orla-player -- examples/deferred/scene.osn
```
