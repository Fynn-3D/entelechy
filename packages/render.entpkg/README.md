# render

The rendering contract and its renderers: a scene hands over a G-buffer
and proxies, and a renderer owns the light — light probes, a path
tracer, daylight, a workbench — with physically based shading, the sky
the air makes, lamps, a planar mirror, fog and clouds, a camera, metered
and local exposure, temporal antialiasing and upscaling, depth of field,
glow, film and a painted look.

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`image`](../image.entpkg/README.md), [`random`](../random.entpkg/README.md), [`raster`](../raster.entpkg/README.md).

```text
use "render.entpkg";
```

## Kernels

- `kernels/aerial.ent` — Aerial perspective after the real atmosphere.
  Kernels: `render.air@1`.
- `kernels/antialias.ent` — Antialiasing within one frame (FXAA).
  Kernels: `render.fxaa@1`.
- `kernels/atmosphere.ent` — The sky as the Earth's air makes it
  (Rayleigh, haze, ozone; light scattered more than once). Kernels:
  `render.atmosphere_spread@1`, `render.atmosphere_look@1`,
  `render.atmosphere_view@1`, `render.atmosphere_sky@1`.
- `kernels/bloom.ent` — Glow round bright lights, of every size at once.
  Kernels: `render.bloom_down@1`, `render.bloom_up@1`.
- `kernels/brdf.ent` — Standard PBR.
- `kernels/camera.ent` — The lens, and the proxies (the coarse scene a
  renderer traces shadows and bounces through). Kernels:
  `render.cam_start@1`, `render.cam@1`, `render.settled@1`,
  `render.depths@1`, `render.settle@1`, `render.lens@1`.
- `kernels/clouds.ent` — A deck of clouds over the sky, lit through
  itself, and its shadows on the ground. Kernels: `render.clouds_deck@1`,
  `render.clouds_shadow@1`, `render.clouds_march@1`.
- `kernels/contact_shadow.ent` — Contact shadows. Kernels:
  `render.contact@1`.
- `kernels/daylight.ent` — Daylight. Kernels: `render.daylight@1`.
- `kernels/depth_of_field.ent` — Depth of field. Kernels:
  `render.dof_start@1`, `render.dof_control@1`, `render.focus@1`,
  `render.focused@1`.
- `kernels/exposure.ent` — Metered exposure and local exposure. Kernels:
  `render.exposure_tiles@1`, `render.exposure_histogram@1`,
  `render.exposure_meter@1`,
  `render.exposure_start@1`, `render.exposure_apply@1`,
  `render.exposure_local_base@1`, `render.exposure_local_blur@1`,
  `render.exposure_local@1`.
- `kernels/film.ent` — Film. Kernels: `render.film_start@1`,
  `render.film_keys@1`, `render.film@1`.
- `kernels/fog.ent` — Height fog the sun and the lamps light, their beams
  cut into shafts by what stands in them and by the clouds. Kernels:
  `render.fog_march@1`, `render.fog_blur@1`, `render.fog_apply@1`.
- `kernels/gloss.ent` — Gloss widened where normals turn within a
  pixel (specular antialiasing). Kernels: `render.gloss_widen@1`.
- `kernels/lamps.ent` — Point lights, spotlights and glowing panels,
  culled by tiles, shadowed through the proxies and on the screen. Kernels:
  `render.lamps_none@1`, `render.lamps_depth@1`, `render.lamps_tiles@1`,
  `render.lamps_light@1`, `render.lamps_shade@1`.
- `kernels/merge.ent` — Two G-buffers of one view made one. Kernels:
  `render.nearer@1`.
- `kernels/mirror.ent` — A planar mirror. Kernels: `render.mirror_lens@1`,
  `render.mirror_mix@1`.
- `kernels/painted.ent` — A painted look. Kernels: `render.npr@1`.
- `kernels/path_trace.ent` — Path tracing (standard PBR, `brdf.ent`).
  Kernels: `render.path_count_start@1`, `render.path_count@1`,
  `render.path_trace@1`, `render.path_blank@1`, `render.path_accum@1`,
  `render.path_atrous@1`, `render.path_behind@1`, `render.path_shade@1`.
- `kernels/post.ent` — Post-processing. Kernels: `render.blank@1`,
  `render.bright@1`, `render.spread@1`, `render.view@1`.
- `kernels/probes.ent` — Light probes. Kernels: `render.probes_sign@1`,
  `render.probes_seen@1`, `render.probes_gather@1`,
  `render.probes_screen_gather@1`, `render.probes_gather_keep@1`,
  `render.probes_gather_blur@1`, `render.probes_behind@1`,
  `render.probes_lit@1`, `render.probes_shade@1`.
- `kernels/scattering.ent` — Light scattered in the air: the phase and
  each pixel's step offset, shared by fog, clouds and the sky.
- `kernels/sky.ent` — The sky.
- `kernels/sun.ent` — The sun's shadows for any scene: cascaded shadow
  maps, soft as far as the sun's disc makes them. Kernels:
  `render.sun_views@1`, `render.sun_shadow@1`.
- `kernels/super_resolution.ent` — Temporal super-resolution (TSR).
  Kernels: `render.tsr@1`, `render.keep_surface@1`.
- `kernels/switch.ent` — One way of drawing (a renderer) is chosen at a
  time, numbered. Kernels: `render.way_start@1`, `render.way_keys@1`,
  `render.way_extent@1`, `render.way_pass@1`, `render.way_pick@1`.
- `kernels/temporal.ent` — Temporal antialiasing (TAA). Kernels:
  `render.jitter@1`, `render.taa@1`, `render.same_lens@1`.
- `kernels/transmission.ent` — Transmission (glass, water, ice).
  Kernels: `render.settle_behind@1`.
- `kernels/upscale.ent` — The picture is computed at an inner resolution
  with a ceiling and scaled to the window at last. Kernels:
  `render.dynamic_start@1`, `render.dynamic@1`, `render.extent@1`,
  `render.extent_size@1`, `render.canvas@1`, `render.upscale@1`,
  `render.sharpen@1`.
- `kernels/workbench.ent` — The workbench's light and backdrop, and its
  renderer on the standard contract (this package's). Kernels:
  `render.workbench_shade@1`.

## Graph presets

- `graphs/air.esn` — Fog and clouds laid over a picture.
- `graphs/atmosphere.esn` — The sky the air makes for a sun's height and
  bearing: its `SkyLook` and the table `render.atmosphere_sky` paints from.
- `graphs/blend.esn` — Temporal antialiasing and upscaling.
- `graphs/bloom.esn` — Glow (`kernels/bloom.ent`): six levels down and
  back up; its `bloom` is what `render.view` takes.
- `graphs/exposure.esn` — Metered exposure, then local exposure: `hdr`
  scaled, and the `exposure` followed.
- `graphs/finish.esn` — The finish.
- `graphs/finish_inner.esn` — The finish at the picture's own size
  (`finish.esn` works at the window's).
- `graphs/lamps.esn` — The lamps added to a renderer's picture, for one
  that does not take them in itself.
- `graphs/path_trace.esn` — The path tracer (`kernels/path_trace.ent`),
  the lamps shown and bounced.
- `graphs/picture.esn` — The picture's raster.
- `graphs/probes.esn` — The light-probe renderer (`kernels/probes.ent`),
  the lamps shown, reflected and bounced; the light gathered on the screen
  drawn anew each frame and averaged while the lens rests (up to
  `settle_frames`, then it rests too).
- `graphs/super_resolution.esn` — Temporal super-resolution
  (`render.tsr`).
- `graphs/upscale.esn` — A finished picture (tone-mapped, sRGB) scaled
  to the window's size.
- `graphs/workbench.esn` — The workbench renderer
  (`render.workbench_shade`) on a scene's G-buffer (the contract, its
  first three maps).

## Tests

`tests/` holds scenes and their replays, run on every backend:
`lit.replay`, `night.replay`, `path_trace.replay`, `probes.replay`,
`sky.replay`, `temporal.replay`.
Every Kernel of the package is reached by one of them.

Used by the [material_balls](../../examples/material_balls/README.md) example.

## The contract

The contract between a scene and a renderer, and what every renderer shares.
A scene owns geometry and materials; a renderer owns light. A scene built on
this contract takes any renderer package by changing a few graph lines.

### What a scene hands over

| Field | Type | Meaning |
| --- | --- | --- |
| `lens` | `Lenses` | The camera this frame (`render.cam` drives one); `focal` 0 sees nothing (a renderer not shown), and a renderer rests on it. |
| `sky` | `SkyLooks` | The sun, the sky's colors, clouds, far silhouettes, air. |
| `surface` | `Grid<Vec4>` | Per pixel: world normal (xyz), distance along the view ray (w; `no_hit()` for sky). |
| `paint` | `Grid<Vec4>` | Per pixel: albedo (rgb, linear), roughness (w). |
| `extra` | `Grid<Vec4>` | Per pixel: sun visibility (x, 0 to 1, from the scene's exact geometry), metalness (y), emission strength (z; emission is albedo times it), baked ambient occlusion (w; the scene's own, finer than a renderer finds; 0 means none given, `baked_occlusion`). |
| `layers` | `Grid<Vec4>` | Per pixel: clearcoat (x), clearcoat roughness (y; 0 means 0.05), sheen (z), subsurface (w). All zero for plain surfaces. |
| `more` | `Grid<Vec4>` | Per pixel: anisotropy (x), transmission (y), index of refraction (z; 0 means 1.5), how much the planar mirror shows (w; 0 none, 1 all, by Fresnel). All zero for plain surfaces. |
| `behind_at` | `Grid<Vec4>` | Where a transmissive pixel's refracted path lands: world position (xyz), 1 if it hit something (w; 0: it left for the sky). |
| `behind_dir` | `Grid<Vec4>` | The path's last direction (xyz), the sun's visibility there (w). |
| `behind_paint` | `Grid<Vec4>` | What it landed on: albedo (rgb), roughness (w). |
| `behind_normal` | `Grid<Vec4>` | Its normal (xyz) and metalness (w). |
| `through` | `Grid<Vec4>` | Transmittance along the path inside (rgb, absorption); w spare. All five zero where nothing transmits. |
| `proxy` | `Volume<Vec4>` | Coarse scene for secondary rays: signed distance (x), albedo (yzw), on the `Proxy` frame's lattice. Ground excluded. |
| `land` | `Grid<Float>` | Ground height on the `Proxy` frame's ground lattice. |
| `frame` | `Proxies` | Where `proxy` and `land` lie in the world, and the ground's albedo. |
| `lamps` | `Lamps` | The lamps beside the sun, at most 128: each where it is, its reach, intensity, bulb size, a spotlight's aim and beam, a panel's half width and height, whether it casts shadows (`point_lamp`, `spot_lamp`, `panel_lamp`). A scene with none hands over `render.lamps_none`. |

The G-buffer is the scene's own full-precision view; `proxy` and `land` are
what a renderer traces shadows and bounces through.

A scene may hand over its G-buffer twice, through `render.settle`: a reduced
one while the camera moves, and a full one, computed once, when it stops
(`render.settled` gates it).

### What a renderer gives

`hdr: Image`, linear HDR, sky included. `render.bright`, `render.spread` and
`render.view` turn it into the picture (`graphs/bloom.esn`'s glow in place
of `render.bright` and `render.spread` for round glows with long tails).

### Before a renderer

Passes over the G-buffer a scene hands over, each giving the same layout
back. The sun's shadows: `render.sun_views` cuts the view into cascades,
squares seen down the sunlight, larger with distance, each moving only in
steps of a quarter of its width (whole texels: the shadows hold still, and
are drawn again only a step on); the scene draws each cascade's depth
toward the sun side by side in one atlas (a raster scene through
`sun_clip`, a distance-field scene marching `sun_ray` to a fixed
tolerance, `sun_depth_of` what it met); `render.sun_shadow` multiplies the
shadow into the sun's visibility (`extra` x), each pixel by the finest
cascade holding it, its blockers' distance making the penumbra as wide as
the sun's disc (`sun_size`) makes it, compared nearer the sun by a texel
times the surface's slant so a curved surface does not shadow itself in
stripes, cascades blended at their edges. `render.contact` multiplies contact shadows into the sun's
visibility (`extra` x), `render.clouds_shadow` the shadow of the clouds'
deck (the deck `graphs/air.esn` gives back as `deck`), and
`render.gloss_widen` widens each pixel's roughness by how much its normals
turn within it (Kaplanyan and Hoffman's), so a far glossy surface does not
flash from pixel to pixel.

### After a renderer

What every renderer shares, in this order:

- **Lamps.** `render.lamps_depth` and `render.lamps_tiles` keep, for each
  tile of 16 × 16 pixels, the lamps whose reach meets what it shows;
  `render.lamps_light` lights each pixel by those alone (walking the
  tile's bits), with `lit_by`, a larger bulb widening the highlight; a
  panel's diffuse light exactly (Lambert's formula for a polygon), its
  highlight from its representative point (Karis's) with `glossy_by`, the
  energy kept as the panel widens it; each
  lamp that casts shadows shadowed through the proxies and, within
  `contact` meters, on the screen. The probes renderer and the path tracer
  take that light in: what they show, reflect and gather on the screen;
  off the screen, what a ray meets in the proxies is lit by every lamp
  (`lamps_on_rough`), so lamps are bounced there too, and the probes
  gather again where a lamp changes. Another renderer has the lamps added
  after (`graphs/lamps.esn`).
- **The mirror.** `render.mirror_lens` reflects the lens in a level plane;
  the scene draws its G-buffer again through it from the plane up
  (`mirror_start` says where a ray reaches the plane), a renderer lights
  that, and `render.mirror_mix` lays it over the pixels `more` w marks, by
  Fresnel, moved by the surface's normal (waves) and drawn out downward.
- **The air** (`graphs/air.esn`). `render.fog_march` walks each view ray
  at a quarter of the size through height fog the sun (shadowed through
  the proxies and by the clouds' deck: shafts between the clouds), the
  sky, a glow of the scene's own and the lamps light: the lamps of each
  pixel's tile (`render.lamps_tiles` with `through`, cut for the whole
  view ray, not the surface), each one's light summed exactly along each
  step (a beam beside a bulb has no spikes) and cut by the proxies toward
  it into shafts; `render.fog_blur` smooths it by depth;
  `render.clouds_march` walks a deck of clouds for the sky at an eighth of
  the size, lit through the cloud toward the sun with light scattered
  again and again, dark at a sunlit edge, lit from below by `below`;
  `render.fog_apply` lays both over the picture, upsampled by depth. Each
  pixel's steps start at a fixed offset of its own, so nothing needs
  history to settle.
- **The sky** (`graphs/atmosphere.esn`). Sunlight through a sphere of
  air round the ground (Hillaire's model): Rayleigh's scattering, haze
  and ozone; light scattered more than once a table by height and the
  sun's angle (`render.atmosphere_spread`, made once). `render.atmosphere_look`
  gives the `SkyLook` (the sun as the air leaves it, the sky overhead and
  at the horizon, the mean sky light) for a sun's height and bearing;
  `render.atmosphere_view` draws the sky round the eye into a table, made
  again only when the sun moves; `render.atmosphere_sky` paints the sky's
  pixels from it after a renderer, with the sun's disc. A scene in these
  units gives lit windows and lamps a few hundred times dimmer than the
  sun, and the exposure carries it from noon to night.
- **Exposure** (`graphs/exposure.esn`). `render.exposure_tiles`,
  `render.exposure_histogram` (128 bins a quarter stop each, every bin
  counted at once) and `render.exposure_meter` (a `<~` state from
  `render.exposure_start`) meter the picture's mean log luminance, its
  darkest and brightest shares left out (`ignore_dark`, `ignore_bright`: a
  few lamps or a black sky do not move it), and follow it as an eye
  adapts, quicker toward a brighter scene, a dark scene shown darker
  (`night_key`); `render.exposure_apply` scales `hdr` by it before the
  glow and tone. Local exposure then takes each pixel apart into its
  neighborhood's brightness (`render.exposure_local_base`, blurred by
  `render.exposure_local_blur`, read by brightness so no halo bleeds round
  a light) and its own detail: bright and dark neighborhoods are brought
  toward the mean (`highlights`, `shadows`), the detail kept (`detail`); 1
  for all three leaves the picture as it is.
- **Glow** (`graphs/bloom.esn`): the bright parts shrunk by halves six
  times and grown back, every level adding the larger glow below it.

### Transmission

Refraction is geometry, so the scene traces it: where the view ray meets a
transmissive surface it refracts in, finds the far side, refracts out and
follows the path to what lies behind, handing that over as the `behind_*`
layer with the absorption along the way. A renderer lights that point
(`behind_radiance` in `kernels/transmission.ent`, its diffuse light from the
renderer's own means), blurs it by the glass's roughness (`frosted`) and adds
what Fresnel lets through (`transmitted`) under the surface's reflection.

### Shading

Every scene and renderer shades with `kernels/brdf.ent`, the standard
metallic-roughness model (the glTF, Unreal and Principled BSDF convention):
a base of Lambert diffuse (with subsurface wrap and translucency) and GGX
specular (optionally anisotropic; Smith geometry, Schlick Fresnel with F0 from
the index of refraction), a Charlie sheen lobe, and a clearcoat on top.
`pbr` makes a plain surface, `lit_by` is one light (`glossy_by` its
specular part alone, for an area light whose diffuse share is found
exactly), `ambient_by` the
environment (split-sum, Karis' analytic fit; base and clearcoat reflections
given apart), `ggx_sample` and `ggx_weight` a path tracer's glossy bounce,
`pbr_of` a G-buffer pixel's `Pbr`. Transmission is carried for renderers that
refract. `render.view` tone maps with `image.entpkg`'s `aces` and encodes sRGB.

### Pieces

- `kernels/camera.ent`: `Lens`, `ray`, the orbit camera (`render.cam`), the
  `Proxy` frame and its rough tracing (`rough_trace`, `rough_shadow`,
  `rough_normal`), `cosine_dir`.
- `kernels/sky.ent`: `SkyLook`, `sky_radiance`, `aerial`.
- `kernels/brdf.ent`: `Pbr` and the standard BRDF.
- `kernels/post.ent`: bloom and tone mapping.
- `kernels/upscale.ent`: a resolution of the scene's own, brought to the
  window at the end, so the window's size costs little. `render.extent`
  is the window scaled down to at most `pixels` pixels, times the share
  `render.dynamic` sets from the frame time to keep `fps` (dynamic
  resolution); `render.canvas` is an empty raster canvas of that size.
  `upscaled` (and `render.upscale`) is an edge-aware upscale (FSR 1's EASU),
  `render.sharpen` a sharpening that cannot overshoot (RCAS), both in display
  space and each frame on its own. `graphs/finish_inner.esn` finishes at the
  picture's size; `graphs/upscale.esn` upscales and sharpens to the window.
  A scene that draws its HUD at the window's size reads the picture through
  `upscaled` in that pass instead, so the window is written once.
  `render.extent`'s `samples` supersamples (2: four points a pixel), and
  `upscaled` then averages each pixel's block.
- `kernels/aerial.ent`: `render.air`, aerial perspective as the air is: it
  thins with height (Rayleigh's by 8 km, Mie's haze by 1.2 km), integrated
  along the ray; Rayleigh scatters blue (far things bluer, flatter), Mie
  forward (bright toward the sun); lit once by the sun and the sky. A scene
  that uses it sets its `SkyLook`'s `air` to 0.
- `kernels/depth_of_field.ent`: depth of field. `render.focus` follows the distance
  at the middle of the picture (a camera's focus); `render.focused` blurs
  by each pixel's circle of confusion, gathering 24 points on a Vogel disc,
  a point counted where its own circle reaches, a farther one only where
  the pixel is blurred too, so a sharp foreground takes no background.
  `render.dof_start` / `render.dof_control` (a `<~` state) let the window
  turn it: V off, to the middle, to the pointer; `[` `]` the aperture.
- `kernels/film.ent`: `render.film`, a cinematic finish on the finished
  picture: a grade in light (exposure, white balance, saturation, teal and
  orange split, an S curve), halation, the lens's color fringe, vignette,
  grain in clumps heaviest in the middle tones, wide-screen bars; presets
  (`film_look`) or its own params, G in the window (`render.film_keys`).
- `kernels/switch.ent`: one renderer at a time on the standard contract,
  numbered (0 lit, 1 painted, 2 probes, 3 path, 6
  workbench), keys 1 to 5; a way not chosen gets a 16 × 16 square of the
  G-buffer (`render.way_extent`, `render.way_pass`), `render.way_pick` the
  chosen picture.
- `kernels/merge.ent`: two G-buffers of one view made one, each pixel the
  nearer surface's (`render.nearer`), for a scene that draws its parts apart.
