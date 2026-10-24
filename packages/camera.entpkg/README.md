# camera

What a camera does to a picture: focus (depth of field, `Dof` as data a
scene changes as it goes), its shutter (motion blur along each pixel's
way), its lens jittered frame after frame and the frames gathered
(temporal antialiasing and super-resolution, held once the picture
rests), and the picture drawn at a resolution of its own, held to a frame
time (dynamic resolution). The lens itself is
[`geometry`](../geometry.entpkg/README.md)'s `Lens`; a person turns it with
[`controls`](../controls.entpkg/README.md)' orbit.

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`image`](../image.entpkg/README.md), [`surface`](../surface.entpkg/README.md).

```text
use "camera.entpkg";
```

## Kernels

- `kernels/depth_of_field.ent` — Depth of field as data (`Dof`: an
  aperture and where it focuses). Kernels: `camera.dof_start@1`,
  `camera.focus@1`, `camera.focused@1`.
- `kernels/motion_blur.ent` — Motion blur: each pixel spread along the
  way it moved on the screen, the lens's move and the thing's. Kernels:
  `camera.motion_blur@1`.
- `kernels/resolution.ent` — The picture computed at an inner resolution
  with a ceiling, a share of it set by frame time (dynamic resolution),
  `image.entpkg`'s `image.upscale` scaling it to the window. Kernels:
  `camera.dynamic_start@1`, `camera.dynamic@1`, `camera.extent@1`,
  `camera.extent_size@1`, `camera.canvas@1`.
- `kernels/super_resolution.ent` — Temporal super-resolution (TSR).
  Kernels: `camera.tsr@1`, `camera.keep_surface@1`.
- `kernels/temporal.ent` — Temporal antialiasing (TAA). Kernels:
  `camera.jitter@1`, `camera.taa@1`, `camera.lens_rests@1`, `camera.same_lens@1`.

## Graph presets

- `graphs/motion_blur.esn` — Motion blur over a picture, the lens's last
  frame kept.
- `graphs/super_resolution.esn` — Temporal super-resolution
  (`camera.tsr`).
- `graphs/taa.esn` — Temporal antialiasing and upscaling.

## Tests

`tests/` holds scenes and their replays, run on every backend, each over
a picture and a surface of its own: `focus.replay`, `motion.replay`,
`resolution.replay`, `temporal.replay`. Every Kernel of the package is
reached by one of them; the light package's tests take its passes over a
renderer's pictures too.
