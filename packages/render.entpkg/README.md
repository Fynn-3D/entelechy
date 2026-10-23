# render

The rendering contract and its renderers: a scene hands over a G-buffer
and proxies, and a renderer owns the light — lit, ray traced, probe GI, a
path tracer, clay — with clustered meshes drawn each place at the level
the screen asks for, physically based shading, the sky the air makes,
lamps, a planar mirror, fog and clouds, a camera, metered and local
exposure, temporal antialiasing and upscaling, depth of field, glow, film
and a stylized look.

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`image`](../image.entpkg/README.md), [`mesh`](../mesh.entpkg/README.md), [`parallel`](../parallel.entpkg/README.md), [`random`](../random.entpkg/README.md), [`raster`](../raster.entpkg/README.md).

```text
use "render.entpkg";
```

## Kernels

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
- `kernels/clay.ent` — Clay: a modeler's preview, a studio light and
  backdrop over the standard contract. Kernels: `render.clay@1`.
- `kernels/cluster.ent` — Clustered meshes drawn (as Nanite draws): each
  place of each thing at the level whose error spans at most `error`
  pixels, clusters out of the picture or behind the last frame's nearer
  things left out, where two levels meet the finer one's corners taken up
  to the coarser one's points. Kernels: `render.cluster_pick@1`,
  `render.cluster_offsets@1`, `render.cluster_cull@1`,
  `render.cluster_kept@1`, `render.cluster_shown@1`, `render.cluster_corners@1`,
  `render.cluster_faces@1`, `render.cluster_hzb@1`.
- `kernels/clouds.ent` — A deck of clouds over the sky, lit through
  itself, and its shadows on the ground. Kernels: `render.clouds_deck@1`,
  `render.clouds_shadow@1`, `render.clouds_march@1`.
- `kernels/composite.ent` — Two G-buffers of one view made one. Kernels:
  `render.nearer@1`.
- `kernels/contact_shadow.ent` — Contact shadows. Kernels:
  `render.contact@1`.
- `kernels/depth_of_field.ent` — Depth of field. Kernels:
  `render.dof_start@1`, `render.dof_control@1`, `render.focus@1`,
  `render.focused@1`.
- `kernels/exposure.ent` — Metered exposure and local exposure. Kernels:
  `render.exposure_tiles@1`, `render.exposure_histogram@1`,
  `render.exposure_meter@1`,
  `render.exposure_start@1`, `render.exposure_apply@1`,
  `render.exposure_local_base@1`, `render.exposure_local_blur@1`,
  `render.exposure_local@1`.
- `kernels/field.ent` — Instanced distance fields: the scene's proxy
  baked from its things' own distances. Kernels: `render.field_atlas8@1`,
  `render.field_shapes8@1`, `render.field_stack@1`,
  `render.field_frames_join@1`, `render.field_frame@1`, `render.field_near@1`,
  `render.field_bin_counts@1`, `render.field_bin_fill@1`,
  `render.field_bake@1`.
- `kernels/film.ent` — Film. Kernels: `render.film_start@1`,
  `render.film_keys@1`, `render.film@1`.
- `kernels/fog.ent` — Height fog the sun and the lamps light, their beams
  cut into shafts by what stands in them and by the clouds. Kernels:
  `render.fog_march@1`, `render.fog_blur@1`, `render.fog_apply@1`.
- `kernels/haze.ent` — Haze: aerial perspective after the real atmosphere.
  Kernels: `render.haze@1`.
- `kernels/lamps.ent` — Point lights, spotlights and glowing panels,
  culled by tiles, shadowed through the proxies and on the screen. Kernels:
  `render.lamps_none@1`, `render.lamps_depth@1`, `render.lamps_tiles@1`,
  `render.lamps_light@1`, `render.lamps_shade@1`.
- `kernels/lit.ent` — Lit: each pixel by itself in one pass, the sun
  and the light all round. Kernels: `render.lit@1`.
- `kernels/mirror.ent` — A planar mirror. Kernels: `render.mirror_lens@1`,
  `render.mirror_mix@1`.
- `kernels/motion_blur.ent` — Motion blur: each pixel spread along the
  way it moved on the screen, the lens's move and the thing's. Kernels:
  `render.motion_blur@1`.
- `kernels/path_tracer.ent` — Path tracing (standard PBR, `brdf.ent`).
  Kernels: `render.path_count_start@1`, `render.path_count@1`,
  `render.path_trace@1`, `render.path_blank@1`, `render.path_accum@1`,
  `render.path_atrous@1`, `render.path_behind@1`, `render.path_shade@1`.
- `kernels/phase.ent` — Light scattered in the air: the phase and
  each pixel's step offset, shared by fog, clouds and the sky.
- `kernels/probe_gi.ent` — Global illumination by light probes. Kernels: `render.probes_sign@1`,
  `render.probes_seen@1`, `render.probes_gather@1`,
  `render.probes_screen_gather@1`, `render.probes_gather_keep@1`,
  `render.probes_gather_blur@1`, `render.probes_behind@1`,
  `render.probes_lit@1`, `render.probes_shade@1`.
- `kernels/proxy_bake.ent` — A mesh baked into a proxy: each cell the
  nearest face's distance within a band and its color, the faces' boxes
  cut into tiles of cells. Kernels: `render.proxy_bake_tiles@1`,
  `render.proxy_bake_faces@1`, `render.proxy_baked@1`.
- `kernels/radiance.ent` — Light kept on the proxies (as Lumen keeps its
  surface cache): each cell near a surface the light leaving it, the sun
  and the lamps through the proxies, the sky they leave open and the light the cache
  gathers from itself, so it bounces again frame after frame, a share lit
  anew each frame until it rests; each pixel's bounced light gathered
  through cones read in the levels as wide as they are. Kernels:
  `render.radiance_frame@1`, `render.radiance_size@1`,
  `render.radiance_turn@1`, `render.radiance@1`,
  `render.bounce_gather@1`, `render.bounce@1`.
- `kernels/reflection.ent` — Reflections on the screen in place of a
  renderer's own (`reflection`) where the screen shows what a mirror ray
  meets. Kernels: `render.reflections@1`.
- `kernels/raytraced.ent` — Ray traced: shadow and reflection traced in
  one pass. Kernels: `render.raytraced@1`.
- `kernels/shadow.ent` — The sun's shadow as a virtual shadow map: a
  clipmap of levels round the eye, kept where they lie on the world's
  lattice, cut into pages drawn from clustered meshes only where the
  pixels read them and not held yet. Kernels: `render.shadow_levels@1`,
  `render.shadow_none@1`, `render.shadow_unheld@1`, `render.shadow_marks@1`,
  `render.shadow_wanted@1`,
  `render.shadow_pages@1`, `render.shadow_clear@1`, `render.shadow_pick@1`,
  `render.shadow_cull@1`, `render.shadow_kept@1`, `render.shadow_faces@1`,
  `render.shadow_draw@1`, `render.shadow_sun@1`.
- `kernels/sky.ent` — The sky.
- `kernels/specular_aa.ent` — Gloss widened where normals turn within a
  pixel (specular antialiasing). Kernels: `render.specular_aa@1`.
- `kernels/stylized.ent` — A stylized look: outlines and a watercolor's tone. Kernels: `render.stylized@1`.
- `kernels/subsurface.ent` — Light under the skin: the share of a
  picture that scatters under subsurface surfaces spread, each color as
  far as it carries. Kernels: `render.subsurface_blur@1`,
  `render.subsurface_apply@1`.
- `kernels/sun.ent` — The sun's shadows for any scene: cascaded shadow
  maps, soft as far as the sun's disc makes them. Kernels:
  `render.sun_none@1`, `render.sun_views@1`, `render.sun_shadow@1`.
- `kernels/super_resolution.ent` — Temporal super-resolution (TSR).
  Kernels: `render.tsr@1`, `render.keep_surface@1`.
- `kernels/temporal.ent` — Temporal antialiasing (TAA). Kernels:
  `render.jitter@1`, `render.taa@1`, `render.same_lens@1`.
- `kernels/tonemap.ent` — Glow and tone mapping. Kernels: `render.blank@1`,
  `render.bright@1`, `render.spread@1`, `render.tonemap@1`.
- `kernels/transmission.ent` — Transmission (glass, water, ice).
  Kernels: `render.settle_behind@1`.
- `kernels/upscale.ent` — The picture is computed at an inner resolution
  with a ceiling and scaled to the window at last. Kernels:
  `render.dynamic_start@1`, `render.dynamic@1`, `render.extent@1`,
  `render.extent_size@1`, `render.canvas@1`, `render.upscale@1`,
  `render.sharpen@1`.

## Graph presets

- `graphs/atmosphere.esn` — The sky the air makes for a sun's height and
  bearing: its `SkyLook` and the table `render.atmosphere_sky` paints from.
- `graphs/bloom.esn` — Glow (`kernels/bloom.ent`): six levels down and
  back up; its `bloom` is what `render.tonemap` takes.
- `graphs/bounce.esn` — Light bounced between things laid over a
  renderer's picture (`kernels/radiance.ent`), for one that takes no
  bounced light in itself (lit, ray traced).
- `graphs/clay.esn` — The clay renderer (`render.clay`) on a scene's
  G-buffer (the contract, its first three maps).
- `graphs/cluster.esn` — Clustered meshes drawn (`kernels/cluster.ent`):
  a library `mesh.entpkg`'s `graphs/cluster.esn` built, the things
  placed (`FieldInst`s, the field's own), the corners and faces for the
  rasterizer.
- `graphs/exposure.esn` — Metered exposure, then local exposure: `hdr`
  scaled, and the `exposure` followed.
- `graphs/field.esn` — The scene's proxy baked round the lens from its
  things' shapes and places (`kernels/field.ent`), in three levels, each
  coarser and wider than the one before.
- `graphs/field_level.esn` — One level of `field.esn`.
- `graphs/lamps.esn` — The lamps added to a renderer's picture, for one
  that does not take them in itself.
- `graphs/motion_blur.esn` — Motion blur over a picture, the lens's last
  frame kept.
- `graphs/path_tracer.esn` — The path tracer (`kernels/path_tracer.ent`),
  the lamps shown and bounced; its `scatter` for light under the skin.
- `graphs/picture.esn` — The picture's raster.
- `graphs/probe_gi.esn` — The light-probe renderer (`kernels/probe_gi.ent`),
  the lamps shown, reflected and bounced; the light gathered on the screen
  drawn anew each frame and averaged while the lens rests (up to
  `settle_frames`, then it rests too); its `scatter` for light under the
  skin.
- `graphs/proxy_bake.esn` — A mesh baked into a proxy
  (`kernels/proxy_bake.ent`), for a scene drawn as one mesh rather than
  placed things.
- `graphs/shadow.esn` — The sun's shadow from a virtual shadow map
  (`kernels/shadow.ent`) multiplied into the G-buffer's sun visibility,
  for a scene of clustered meshes.
- `graphs/subsurface.esn` — Light under the skin: a renderer's `scatter`
  spread across then down and laid back over its `hdr`.
- `graphs/super_resolution.esn` — Temporal super-resolution
  (`render.tsr`).
- `graphs/taa.esn` — Temporal antialiasing and upscaling.
- `graphs/tonemap.esn` — The finish at the picture's own size: glow, tone
  and sRGB, sharpened if asked.
- `graphs/upscale.esn` — A finished picture (tone-mapped, sRGB) scaled
  to the window's size.
- `graphs/volumetrics.esn` — Fog and clouds laid over a picture.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`bounce.replay`, `cluster.replay`, `field.replay`, `lit.replay`, `night.replay`, `path_tracer.replay`, `probe_gi.replay`, `proxy_bake.replay`, `reflection.replay`, `shadow.replay`,
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
what a renderer traces shadows and bounces through. A scene of placed things
bakes its proxy from their own distance volumes (`graphs/field.esn`: each
kind's shape in an atlas, each thing placed, a block round the lens baked
from the things standing in it), so what is traced is each thing's shape,
not a rough stand-in for it.

A scene may hand over its G-buffer twice, through `render.settle`: a reduced
one while the camera moves, and a full one, computed once, when it stops
(`render.settled` gates it).

### What a renderer gives

`hdr: Image`, linear HDR, sky included. Lit and ray traced also give
`reflection`, the reflection they laid on each surface from all round as
it shows, which `render.reflections` takes the place of where the screen
shows what the mirror ray meets. `render.bright`, `render.spread` and
`render.tonemap` turn it into the picture (`graphs/bloom.esn`'s glow in place
of `render.bright` and `render.spread` for round glows with long tails).

### Choosing a renderer

A scene places one renderer. Lit (`render.lit`) lights each pixel in one
pass by itself and the coarse scene: the lightest, and the one for a scene
in daylight whose look comes from its sun, sky and shadows. Ray traced
(`render.raytraced`) is lit with each smooth surface's mirror ray traced
through the proxies; a distance-field scene marches its own shadow and
mirror rays and lights what they meet with `raytraced_shade`. Probe GI and
the path tracer add light bounced between things, many passes a frame,
worth their cost where bounced light shows (indoors, at night, under
lamps). Clay shows the shapes alone, as a modeler does.

A painted look is not another renderer: shading as a painting has it is a
renderer's parameter (`painted` of lit, `style` of probe GI, `toon` in
`kernels/probe_gi.ent`), and outlines and a watercolor's tone are a pass
after it (`render.stylized`) that reads only its picture and the G-buffer.

### Tuning a look

What decides a look is a scene's to set, never a number written into a
kernel: each tint, strength, distance and threshold a pass draws by is a
param of its kernel, its default the look the package ships, and a graph
preset that places the kernel carries the same param up, so a scene sets
it on the preset's node. Shared functions take such params as small
structs (`Surround`, `Openness`, `Mirror`, `Toon`, `ContactReach`,
`StudioLight`, `FloorLines`). The numbers left in a kernel are the
method's own: filter weights, luminance weights, phase normalizers.

### Before a renderer

Passes over the G-buffer a scene hands over, each giving the same layout
back. The sun's shadows: `render.sun_views` cuts the view into cascades,
squares seen down the sunlight, larger with distance, each moving only in
steps of a quarter of its width (whole texels: the shadows hold still, and
are drawn again only a step on); the scene draws each cascade's depth
toward the sun side by side in one atlas (a raster scene through
`sun_view_clip`, a distance-field scene marching `sun_ray` to a fixed
tolerance, `sun_depth_of` what it met); `render.sun_shadow` multiplies the
shadow into the sun's visibility (`extra` x), each pixel by the finest
cascade holding it, its blockers' distance making the penumbra as wide as
the sun's disc (`sun_size`) makes it, compared nearer the sun by a texel
times the surface's slant so a curved surface does not shadow itself in
stripes, cascades blended at their edges. `render.contact` multiplies contact shadows into the sun's
visibility (`extra` x), `render.clouds_shadow` the shadow of the clouds'
deck (the deck `graphs/volumetrics.esn` gives back as `deck`), and
`render.specular_aa` widens each pixel's roughness by how much its normals
turn within it (Kaplanyan and Hoffman's), so a far glossy surface does not
flash from pixel to pixel.

For a scene of clustered meshes, `graphs/shadow.esn` takes the
cascades' place with a virtual shadow map: `levels` levels round the eye,
each twice as wide as the one before (the first `nearest` meters), `size`
texels a side, kept where they lie on the world's lattice (texel (u, v)
at (u mod size, v mod size)), so a level moved on keeps all it still
holds. Each frame `render.shadow_marks` marks, for every few pixels, the
pages within `reach` texels of its point in the finest level whose texel
is as large as the pixel (`sharpness` times); `render.shadow_pages` finds
each marked page its slot does not hold, or holds under another sun, and
only those are cleared and drawn, at most `most` a frame, the finest
levels' first (a page waiting is read from the first coarser level that
holds it): the things placed over them, each at
the level of its clusters a texel asks for (`cluster_flat_view`), each
face cut into tiles of 16 by 16 texels. At rest none is. A texel keeps the
height toward the sun of the nearest thing over it, and
`render.shadow_sun` multiplies the shadow into the sun's visibility, as
soft as the sun's disc makes it from its caster's height.

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
  `contact` meters, on the screen. Probe GI and the path tracer
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
- **The air** (`graphs/volumetrics.esn`). `render.fog_march` walks each view ray
  at a quarter of the size through height fog the sun (shadowed by the
  sun's cascades where they hold the point, `sun_view_lit`, through the
  proxies elsewhere, and by the clouds' deck: shafts behind what stands in
  the light and between the clouds; a scene without cascades gives
  `render.sun_none`), the sky, a glow of the scene's own and the lamps
  light: the lamps of each pixel's tile (`render.lamps_tiles` with
  `through`, cut for the whole view ray, not the surface), each one's
  light summed exactly along each step (a beam beside a bulb has no
  spikes) and cut by the proxies toward it into shafts; `render.fog_blur`
  smooths it by depth; `render.clouds_march` walks a deck of clouds for
  the sky at an eighth of the size, lit through the cloud toward the sun
  with light scattered again and again, dark at a sunlit edge, lit from
  below by `below`; `render.fog_apply` lays both over the picture,
  upsampled by depth. Each pixel's steps start at a fixed offset of its
  own, so nothing needs history to settle.
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
- **Light under the skin** (`graphs/subsurface.esn`). Probe GI, the path
  tracer and lit each give, beside `hdr`, the share of it
  that scatters under subsurface surfaces (`scatter`: `scattered_share`,
  their matte light from the sun and from round about times the
  subsurface, through the air as the rest; none when lit paints);
  `render.subsurface_blur` spreads it across then down (Jimenez's
  separable way), red farthest and blue nearest, as far as the material
  carries light (`reach` meters at full subsurface) and only over the same
  surface; `render.subsurface_apply` lays the spread share back in place
  of the share as it was.
- **Bounced light** (`graphs/bounce.esn`). Lit and ray traced light each
  pixel by the sun and the sky alone; `render.radiance` keeps, in a
  volume laid as the proxy is (a level for each of its levels, `scale`
  times coarser), the light leaving each surface the proxy holds: its
  color times the sun and the lamps through the proxies, the sky they
  leave open (lit's
  own `hemisphere` and `proxy_open`) and what four cones gather from the
  cache as the last frame left it, so light bounces once more each time a
  cell is lit. A `parts`th of its cells is lit anew each frame, the rest
  kept from the last frame's where they lie (a block moved on keeps what
  it held), and `render.radiance_turn` stops it `settle` frames after the
  block or the sun last moved. `render.bounce_gather` traces five cones
  about each normal every `every` pixels (`radiance_cone`: each step read
  in the first level as wide as the cone, shut by how near the proxy's
  things are against its width, what the cache holds laid behind what is
  nearer), and `render.bounce` lays it over each pixel by distance, times
  its diffuse color and baked occlusion.
- **Reflections on the screen** (`render.reflections`). On a surface
  smoother than `below`, the mirror ray is walked across the picture
  against the G-buffer's depth (`screen_march`); where it meets a surface
  facing it, what the picture holds there, times the standard share the
  surface reflects (`env_reflectance`, `ambient_by`'s own), takes the place
  of the renderer's `reflection`, surely within the picture and short of
  the ray's end, less so toward its edges, at its end and on rougher
  surfaces.
- **Motion blur** (`graphs/motion_blur.esn`). Each pixel's way on the
  screen this frame: where the last frame's lens saw the point it shows,
  before it moved its own `motion` (meters, as temporal antialiasing takes
  it); the picture gathered along `shutter` of it, a point behind the
  pixel's surface counted, a nearer one where its own way reaches the
  pixel (McGuire's reconstruction, simplified).
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
refract. `render.tonemap` tone maps with `image.entpkg`'s `aces` and encodes sRGB.

### Pieces

- `kernels/camera.ent`: `Lens`, `ray`, the orbit camera (`render.cam`), the
  `Proxy` frame and its rough tracing (`rough_trace`, `rough_shadow`,
  `rough_normal`), `cosine_dir`.
- `kernels/sky.ent`: `SkyLook`, `sky_radiance`, `aerial`.
- `kernels/brdf.ent`: `Pbr` and the standard BRDF.
- `kernels/tonemap.ent`: bloom and tone mapping.
- `kernels/upscale.ent`: a resolution of the scene's own, brought to the
  window at the end, so the window's size costs little. `render.extent`
  is the window scaled down to at most `pixels` pixels, times the share
  `render.dynamic` sets from the frame time to keep `fps` (dynamic
  resolution); `render.canvas` is an empty raster canvas of that size.
  `upscaled` (and `render.upscale`) is an edge-aware upscale (FSR 1's EASU),
  `render.sharpen` a sharpening that cannot overshoot (RCAS), both in display
  space and each frame on its own. `graphs/tonemap.esn` finishes at the
  picture's size; `graphs/upscale.esn` upscales and sharpens to the window.
  A scene that draws its HUD at the window's size reads the picture through
  `upscaled` in that pass instead, so the window is written once.
  `render.extent`'s `samples` supersamples (2: four points a pixel), and
  `upscaled` then averages each pixel's block.
- `kernels/haze.ent`: `render.haze`, aerial perspective as the air is: it
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
- `kernels/composite.ent`: two G-buffers of one view made one, each pixel the
  nearer surface's (`render.nearer`), for a scene that draws its parts apart.
