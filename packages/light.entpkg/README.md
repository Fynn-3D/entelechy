# light

Light in a world a scene hands over: how surfaces take light (physically
based, as a painting has it, or plainly as a game without a renderer
does), where light comes from (the sun, the sky the air makes, lamps),
what shuts it out (shadows, occlusion), how it travels (bounced between
things, reflected, through glass and skin, fog, clouds and air), and the
renderers that put it together — lit, ray traced, probe GI, a path
tracer, clay. What is lit is [`surface`](../surface.entpkg/README.md)'s
(the G-buffer and the proxies); what a camera does to the picture is
[`camera`](../camera.entpkg/README.md)'s; what is done to the picture
alone is [`image`](../image.entpkg/README.md)'s.

**Layer:** Graphics. **Uses:** [`camera`](../camera.entpkg/README.md), [`geometry`](../geometry.entpkg/README.md), [`image`](../image.entpkg/README.md), [`mesh`](../mesh.entpkg/README.md), [`parallel`](../parallel.entpkg/README.md), [`random`](../random.entpkg/README.md), [`raster`](../raster.entpkg/README.md), [`surface`](../surface.entpkg/README.md).

```text
use "light.entpkg";
```

## Effects as words

The package gives words, and a scene composes the effect it wants. Each
effect is first a function a scene's own kernel calls on a pixel (light,
air, a look); a kernel is a thin pass over those words where the effect
must read its neighbors, the last frame or another size; a look is data
(`Grade`, `Film`, `Dof`, `Haze`, `Toon`, `Surround`) a scene's kernel
makes, blends and changes as it goes; any pass is kept to part of a
picture by `image.masked`; and a graph preset is a recipe built from the
same words. [`tests/words.esn`](tests/words.esn) holds a scene's kernels
saying a pass by its words to the pass's own picture.

| Effect | Words a kernel calls | Passes | Data |
| --- | --- | --- | --- |
| Surfaces and their light | `pbr_of`, `lit_by`, `glossy_by`, `ambient_by`, `env_reflectance`, `scattered_share`, `baked_occlusion` | a renderer: `light.lit`, `light.raytraced`, `graphs/probe_gi.esn`, `graphs/path_tracer.esn`, `graphs/clay.esn` | `Pbr` |
| Light from all round | `hemisphere`, `proxy_open`, `seen_along`, `sky_radiance` | | `Surround`, `Openness` |
| A painted look, as far as asked | `shaded_as`, `toon`, `painted_gloss`, `scattered_as`, `mirrored_as`, `toon_default` | `painted` of a renderer | `Toon` |
| The sun's shadows | `sun_pcss`, `contact_shadow`, `rough_shadow`, `cloud_shadow` | `light.sun_shadow`, `light.contact`, `light.clouds_shadow` | `ContactReach` |
| Lamps | `point_lamp`, `spot_lamp`, `panel_lamp`, `lamps_on_surface`, `lamp_shadow` | `graphs/lamps.esn`, `graphs/lamp_maps.esn` | `Lamps`, `LampLooks` |
| Air and sky | `hazed`, `aerial`, `sky_radiance`, `air_radiance`, `air_phase` | `light.haze`, `graphs/atmosphere.esn` | `Haze`, `SkyLook` |
| Fog and clouds | `fog_density`, `clouds_seen`, `cloud_density` | `graphs/volumetrics.esn`, `light.clouds_march` | `CloudDeck` |
| Reflections | `screen_march`, `reflection_edge`, `raytraced_reflection` | `light.reflections`, `light.mirror_mix` | |
| Glass and translucency | `transmitted`, `frosted`, `behind_radiance` | `graphs/translucent.esn`, `graphs/subsurface.esn` | |
| Gloss under a pixel | `widened_rough` | `light.specular_aa` | |
| Motion and focus (`camera`'s) | `motion_blurred`, `motion_way`, `dof_blurred`, `blur_of` | `camera.motion_blur`, `camera.focus`, `camera.focused` | `Dof` |
| Ink from the G-buffer (`surface`'s) | `outline_near`, `outline_crease` | `surface.outline` | |
| Plain light, with no renderer | `plain_daylight`, `plain_light`, `flame_flicker`, `light_stepped`, `plain_haze`, `plain_dark`, `thin_lit` | | `PlainLight` |
| Edges smoothed over frames (`camera`'s) | | `graphs/taa.esn`, `graphs/super_resolution.esn` | |

The picture after the light is `image.entpkg`'s, in the same words:

| Effect | Words a kernel calls | Passes | Data |
| --- | --- | --- | --- |
| Glow, exposure, tone | `bloom_at`, `tone_sharpened`, `tone_curve`, `tone_warmth` | `graphs/bloom.esn`, `graphs/exposure.esn`, `graphs/tonemap.esn`, `graphs/upscale.esn` | |
| A grade | `grade_color`, `grade_mix`, `grade_balance`, `grade_none` | `image.grade` | `Grade` |
| Film | `film_split`, `film_curve`, `film_halation`, `film_bars`, `film_preset`, `fringed`, `vignetted`, `film_grain` | `image.film` | `Film` |
| A watercolor's tone | `watercolor` | `image.watercolor` | `Watercolor` |
| Edges smoothed in a frame | `fxaa` | `image.fxaa` | |
| Part of a picture | | `image.masked` | |

A story's look is the plainest example: a scene's kernel makes the film it
wants each frame and the pass lays it on.

```text
node story: kernel {
    requires { field.write; time.read; }
    output result: Rows<Film>;
    domain = 1;
    compute {
        let a = film_preset(1);
        let b = film_preset(3);
        let t = smoothstep(20.0, 26.0, time());
        return Film { ..a, grade: grade_mix(a.grade, b.grade, t), curve: mix(a.curve, b.curve, t) };
    }
}
node filmed: image.film@1 { image <- shown.result; film <- story.result; }
```

## Kernels

- `kernels/atmosphere.ent` — The sky as the Earth's air makes it
  (Rayleigh, haze, ozone; light scattered more than once). Kernels:
  `light.atmosphere_spread@1`, `light.atmosphere_look@1`,
  `light.atmosphere_view@1`, `light.atmosphere_sky@1`.
- `kernels/brdf.ent` — Standard PBR.
- `kernels/clay.ent` — Clay: a modeler's preview, a studio light and
  backdrop over the standard contract. Kernels: `light.clay@1`.
- `kernels/clouds.ent` — A deck of clouds over the sky, lit through
  itself, and its shadows on the ground. Kernels: `light.clouds_deck@1`,
  `light.clouds_shadow@1`, `light.clouds_march@1`.
- `kernels/contact_shadow.ent` — Contact shadows. Kernels:
  `light.contact@1`.
- `kernels/fog.ent` — Height fog and fog volumes the sun and the lamps
  light, their beams cut into shafts by what stands in them and by the
  clouds, in a grid of cells over the view. Kernels:
  `light.fog_volumes_none@1`, `light.froxel_light@1`, `light.froxel_scan@1`,
  `light.froxel_furthest@1`, `light.froxel_view@1`, `light.fog_blur@1`,
  `light.fog_apply@1`.
- `kernels/haze.ent` — Haze: aerial perspective after the real atmosphere.
  Kernels: `light.haze@1`.
- `kernels/lamp_map.ent` — Lamps' shadow maps: a cube of distances for
  each lamp that casts shadows, drawn from a clustered library by tiles,
  each texel where the ray from the lamp meets the nearest face. Kernels:
  `light.lamp_maps_none@1`, `light.lamp_map_lamps@1`,
  `light.lamp_map_count@1`, `light.lamp_map_counts@1`,
  `light.lamp_map_clusters@1`, `light.lamp_map_tiles@1`,
  `light.lamp_map_held@1`, `light.lamp_map_dirty@1`, `light.lamp_map_clear@1`,
  `light.lamp_map_draw@1`.
- `kernels/lamps.ent` — Point lights, spotlights and glowing panels,
  culled by tiles, shadowed by their maps or through the proxies, and on
  the screen. Kernels:
  `light.lamps_none@1`, `light.lamp_looks_none@1`, `light.lamps_depth@1`,
  `light.lamps_tiles@1`, `light.lamps_light@1`, `light.lamps_shade@1`.
- `kernels/lit.ent` — Lit: each pixel by itself in one pass, the sun
  and the light all round. Kernels: `light.lit@1`.
- `kernels/mirror.ent` — A planar mirror. Kernels: `light.mirror_lens@1`,
  `light.mirror_mix@1`.
- `kernels/painted.ent` — The painted look: light and shade as a painting
  has them (`toon`, `Toon`), and a renderer's shading taken toward it as far
  as a scene asks (`shaded_as`, `scattered_as`, `mirrored_as`).
- `kernels/plain.ent` — Light as a game without a renderer draws it, on a
  face it found itself (a voxel's, a tile's): the sky and a sun
  (`plain_daylight`), lights fading to their reach (`plain_light`,
  `PlainLight`) and flickering (`flame_flicker`), light kept to a few
  steps (`light_stepped`), haze and the dark (`plain_haze`,
  `plain_dark`), a thin thing lit against the sun (`thin_lit`).
- `kernels/path_tracer.ent` — Path tracing (standard PBR, `brdf.ent`).
  Kernels: `light.path_count_start@1`, `light.path_count@1`,
  `light.path_trace@1`, `light.path_blank@1`, `light.path_accum@1`,
  `light.path_atrous@1`, `light.path_behind@1`, `light.path_shade@1`.
- `kernels/phase.ent` — Light scattered in the air: the phase, shared by
  fog, clouds and the sky.
- `kernels/probe_gi.ent` — Global illumination by light probes. Kernels: `light.probes_sign@1`,
  `light.probes_seen@1`, `light.probes_gather@1`,
  `light.probes_screen_gather@1`, `light.probes_gather_keep@1`,
  `light.probes_gather_blur@1`, `light.probes_behind@1`,
  `light.probes_lit@1`, `light.probes_shade@1`.
- `kernels/radiance.ent` — Light kept on the proxies (as Lumen keeps its
  surface cache): each cell near a surface the light leaving it, the sun
  and the lamps through the proxies (each lamp that casts shadows walked
  toward in full, `lamps_on_surface`, so a chair darkens the floor its
  bounce lights too), the sky they leave open and the light the cache
  gathers from itself, so it bounces again frame after frame, a share lit
  anew each frame until it rests, its cones turned each time and averaged
  in; each pixel's bounced light gathered through cones read in the levels
  as wide as they are, light taken only from surfaces facing back along a
  cone (so a thin wall's far side does not leak through), the cones set
  off at jittered distances and averaged over sixteen cells. The light a
  pixel takes is gathered by probes of the screen (Lumen's screen probes),
  one every `probe` pixels, `rays` rays each over the hemisphere, each ray
  first followed on the screen a meter (`screen_reach`: where it meets
  what the screen shows, the lit picture's light there, averaged over a
  patch as wide as a near cone, `light.bounce_screen`), the cache only
  past it, so the light and shade nearer than its cells (under a desk,
  beside a chair's leg) come from the screen; each probe keeps what comes
  in as first-order spherical harmonics, smoothed among its alike
  neighbors (`light.bounce_probe_filter`), and each pixel reads the
  probes round it for its own normal. A surface smoother than `hit_below`
  traces its mirror ray through the proxies to what it meets and lights it
  there (`bounce_hit_light`: the sun through its shadow, the lamps, the open
  sky and a cone of the cache's light), so off the screen chrome mirrors the
  room's light and shade, not the cache's average; a rougher one reads a
  cone as wide as it is rough. Kernels:
  `light.radiance_frame@1`, `light.radiance_size@1`,
  `light.radiance_turn@1`, `light.radiance@1`,
  `light.bounce_screen@1`, `light.bounce_probes@1`, `light.bounce_probe_filter@1`,
  `light.bounce_gather@1`, `light.bounce@1`.
- `kernels/reflection.ent` — Reflections on the screen in place of a
  renderer's own (`reflection`) where the screen shows what a mirror ray
  meets. Kernels: `light.reflections@1`.
- `kernels/raytraced.ent` — Ray traced: shadow and reflection traced in
  one pass. Kernels: `light.raytraced@1`.
- `kernels/shadow.ent` — The sun's shadow as a virtual shadow map: a
  clipmap of levels round the eye, kept where they lie on the world's
  lattice, cut into pages drawn from clustered meshes only where the
  pixels read them and not held yet. Kernels: `light.shadow_levels@1`,
  `light.shadow_none@1`, `light.shadow_unheld@1`, `light.shadow_marks@1`,
  `light.shadow_wanted@1`,
  `light.shadow_pages@1`, `light.shadow_clear@1`, `light.shadow_pick@1`,
  `light.shadow_cull@1`, `light.shadow_kept@1`, `light.shadow_faces@1`,
  `light.shadow_draw@1`, `light.shadow_sun@1`.
- `kernels/sky.ent` — The sky.
- `kernels/specular_aa.ent` — Gloss widened where normals turn within a
  pixel (specular antialiasing). Kernels: `light.specular_aa@1`.
- `kernels/subsurface.ent` — Light under the skin: the share of a
  picture that scatters under subsurface surfaces spread, each color as
  far as it carries. Kernels: `light.subsurface_blur@1`,
  `light.subsurface_apply@1`.
- `kernels/sun.ent` — The sun's shadows for any scene: cascaded shadow
  maps, soft as far as the sun's disc makes them. Kernels:
  `light.sun_none@1`, `light.sun_views@1`, `light.sun_shadow@1`.
- `kernels/translucent.ent` — Translucent layers over a renderer's
  picture, every layer at once in no order (glass, smoke, glows): shaded
  where each covers a pixel in front of the G-buffer and summed in 32-bit
  fixed point (weighted, blended order-independent transparency with
  colored transmission), then laid over the picture. Kernels:
  `light.translucent_tiles@1`, `light.translucent_corners@1`,
  `light.translucent_blank@1`, `light.translucent_splat@1`,
  `light.translucent_resolve@1`.
- `kernels/transmission.ent` — Transmission (glass, water, ice).
  Kernels: `surface.settle_behind@1`.
## Graph presets

- `graphs/atmosphere.esn` — The sky the air makes for a sun's height and
  bearing: its `SkyLook` and the table `light.atmosphere_sky` paints from.
- `graphs/bounce.esn` — Light bounced between things laid over a
  renderer's picture (`kernels/radiance.ent`), for one that takes no
  bounced light in itself (lit, ray traced).
- `graphs/clay.esn` — The clay renderer (`light.clay`) on a scene's
  G-buffer (the contract, its first three maps).
- `graphs/lamp_maps.esn` — The lamps' shadow maps (`kernels/lamp_map.ent`),
  drawn once while lamps and things keep still.
- `graphs/lamps.esn` — The lamps added to a renderer's picture, for one
  that does not take them in itself.
- `graphs/path_tracer.esn` — The path tracer (`kernels/path_tracer.ent`),
  the lamps shown and bounced; its `scatter` for light under the skin.
- `graphs/probe_gi.esn` — The light-probe renderer (`kernels/probe_gi.ent`),
  the lamps shown, reflected and bounced; the light gathered on the screen
  drawn anew each frame and averaged while the lens rests (up to
  `settle_frames`, then it rests too); its `scatter` for light under the
  skin.
- `graphs/shadow.esn` — The sun's shadow from a virtual shadow map
  (`kernels/shadow.ent`) multiplied into the G-buffer's sun visibility,
  for a scene of clustered meshes.
- `graphs/subsurface.esn` — Light under the skin: a renderer's `scatter`
  spread across then down and laid back over its `hdr`.
- `graphs/translucent.esn` — Translucent layers over a renderer's picture
  (`kernels/translucent.ent`), their mesh cut at the near plane.
## Tests

`tests/` holds scenes and their replays, run on every backend:
`bounce.replay`, `fog_volumes.replay`, `lamp_looks.replay`, `lamp_maps.replay`, `lit.replay`, `night.replay`, `path_tracer.replay`, `plain.replay`, `probe_gi.replay`, `reflection.replay`, `shadow.replay`,
`sky.replay`, `temporal.replay`, `translucent.replay`, `words.replay`; they take the surface's and the camera's passes over the renderers' pictures too.
Every Kernel of the package is reached by one of them.

Used by the [material_balls](../../examples/material_balls/README.md) example.

## The contract

The contract between a scene and a renderer, and what every renderer shares.
A scene owns geometry and materials; a renderer owns light. A scene built on
this contract takes any renderer package by changing a few graph lines.

### What a scene hands over

| Field | Type | Meaning |
| --- | --- | --- |
| `lens` | `Lenses` | The camera this frame (`controls.orbit` drives one); `focal` 0 sees nothing (a renderer not shown), and a renderer rests on it. |
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
| `proxy` | `Volume<Vec4>` | Coarse scene for secondary rays: signed distance (x), albedo (yzw), each cell a point of the `Proxy` frame's lattice (`lo` plus whole steps of `cell`). Ground excluded. |
| `land` | `Grid<Float>` | Ground height on the `Proxy` frame's ground lattice. |
| `frame` | `Proxies` | Where `proxy` and `land` lie in the world, and the ground's albedo: a row a level, finest first, the levels up the volume's depth one after another (`graphs/field.esn`'s round the lens; `graphs/proxy_detail.esn`'s fine one before the scene's own); the last, coarsest, the scene's own. |
| `lamps` | `Lamps` | The lamps beside the sun, at most 128: each where it is, its reach, intensity, bulb size, a spotlight's aim and beam, a panel's half width and height, whether it casts shadows (`point_lamp`, `spot_lamp`, `panel_lamp`), and where its cookie and profile lie in `lamp_looks` (`lamp_with_cookie`, `lamp_with_profile`). A scene with none hands over `light.lamps_none`. |
| `lamp_looks` | `LampLooks` | The lamps' cookies and profiles, a picture of the scene's own (its kernels draw it, still or moving): a cookie, the light's filter laid across a spotlight's beam or round a point light; a profile, a row of the share of a lamp's light from its aim to straight behind it, as a `.ies` file read as an input gives it. A scene with none hands over `light.lamp_looks_none`. |

The G-buffer is the scene's own full-precision view; `proxy` and `land` are
what a renderer traces shadows and bounces through. A scene of placed things
bakes its proxy from their own distance volumes (`graphs/field.esn`: each
kind's shape in an atlas, each thing placed, a block round the lens baked
from the things standing in it), so what is traced is each thing's shape,
not a rough stand-in for it.

A scene may hand over its G-buffer twice, through `surface.settle`: a reduced
one while the camera moves, and a full one, computed once, when it stops
(`controls.orbit_rests` gates it).

### What a renderer gives

`hdr: Image`, linear HDR, sky included. Lit and ray traced also give
`reflection`, the reflection they laid on each surface from all round as
it shows, which `light.reflections` takes the place of where the screen
shows what the mirror ray meets. `image.bright`, `image.spread` and
`image.tonemap` turn it into the picture (`image.entpkg`'s `graphs/bloom.esn` glow in place
of `image.bright` and `image.spread` for round glows with long tails).

### Choosing a renderer

A scene places one renderer. Lit (`light.lit`) lights each pixel in one
pass by itself and the coarse scene: the lightest, and the one for a scene
in daylight whose look comes from its sun, sky and shadows. Ray traced
(`light.raytraced`) is lit with each smooth surface's mirror ray traced
through the proxies; a distance-field scene marches its own shadow and
mirror rays and lights what they meet with `raytraced_shade`. Probe GI and
the path tracer add light bounced between things, many passes a frame,
worth their cost where bounced light shows (indoors, at night, under
lamps). Clay shows the shapes alone, as a modeler does.

A painted look is not another renderer: a renderer's shading is taken
toward a painting's as far as its `painted` asks (lit's and probe GI's; 0
none, 1 all, between them blended, so one look turns into the other as a
scene goes), by the words of `kernels/painted.ent` (`shaded_as`, `toon`,
`Toon`) that a scene's own kernel calls too; its ink (`surface.outline`) and
a watercolor's tone (`image.watercolor`) are passes after it that read
only its picture and the G-buffer.

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
back. The sun's shadows: `light.sun_views` cuts the view into cascades,
squares seen down the sunlight, larger with distance, each moving only in
steps of a quarter of its width (whole texels: the shadows hold still, and
are drawn again only a step on); the scene draws each cascade's depth
toward the sun side by side in one atlas (a raster scene through
`sun_view_clip`, a distance-field scene marching `sun_ray` to a fixed
tolerance, `sun_depth_of` what it met); `light.sun_shadow` multiplies the
shadow into the sun's visibility (`extra` x), each pixel by the finest
cascade holding it, its blockers' distance making the penumbra as wide as
the sun's disc (`sun_size`) makes it, compared nearer the sun by a texel
times the surface's slant so a curved surface does not shadow itself in
stripes, cascades blended at their edges. `light.contact` multiplies contact shadows into the sun's
visibility (`extra` x), `light.clouds_shadow` the shadow of the clouds'
deck (the deck `graphs/volumetrics.esn` gives back as `deck`), and
`light.specular_aa` widens each pixel's roughness by how much its normals
turn within it (Kaplanyan and Hoffman's), so a far glossy surface does not
flash from pixel to pixel.

Decals (`graphs/decals.esn`) are pictures a scene presses onto what it
shows: each a box in the world (`Decal`, `decal`), its picture laid across
and up and pressed along its depth onto the surfaces inside that face out
along it, thinning toward the box's depth and as a surface turns away
(`fade`). Where it covers, its color (times `tint`, from the scene's decal
looks, an atlas its kernels draw) goes over the albedo, its `rough`,
`metal` and `glow` over the surface's (each negative: kept), its bumps (a
normal map in the looks) tilt the normal along its width and height as
they lie on the surface; the decals are laid in their rows' order, the
last on top. The paint, the extra and the surface come back in the
contract's layout, so every renderer lights a decal as it lights the
surface it lies on.

For a scene of clustered meshes, `graphs/shadow.esn` takes the
cascades' place with a virtual shadow map: `levels` levels round the eye,
each twice as wide as the one before (the first `nearest` meters), `size`
texels a side, kept where they lie on the world's lattice (texel (u, v)
at (u mod size, v mod size)), so a level moved on keeps all it still
holds. Each frame `light.shadow_marks` marks, for every few pixels, the
pages within `reach` texels of its point in the finest level whose texel
is as large as the pixel (`sharpness` times); `light.shadow_pages` finds
each marked page its slot does not hold, or holds under another sun, and
only those are cleared and drawn, at most `most` a frame, the finest
levels' first (a page waiting is read from the first coarser level that
holds it): the things placed over them, each at
the level of its clusters a texel asks for (`cluster_flat_view`), each
face cut into tiles of 16 by 16 texels. At rest none is. A texel keeps the
height toward the sun of the nearest thing over it, and
`light.shadow_sun` multiplies the shadow into the sun's visibility, as
soft as the sun's disc makes it from its caster's height.

### After a renderer

What every renderer shares, in this order:

- **Lamps.** `light.lamps_depth` and `light.lamps_tiles` keep, for each
  tile of 16 × 16 pixels, the lamps whose reach meets what it shows;
  `light.lamps_light` lights each pixel by those alone (walking the
  tile's bits), with `lit_by`, a larger bulb widening the highlight; a
  panel's diffuse light exactly (Lambert's formula for a polygon), its
  highlight from its representative point (Karis's) with `glossy_by`, the
  energy kept as the panel widens it; each
  lamp that casts shadows shadowed by its shadow map where the scene draws
  them (`graphs/lamp_maps.esn`: a cube of distances, read percentage
  closer, the penumbra from the blockers' mean distance and the bulb's
  size), through the proxies where it does not (`light.lamp_maps_none`),
  and, within `contact` meters, on the screen. Probe GI and the path tracer
  take that light in: what they show, reflect and gather on the screen;
  off the screen, what a ray meets in the proxies is lit by every lamp
  (`lamps_on_rough`), so lamps are bounced there too, and the probes
  gather again where a lamp changes. Another renderer has the lamps added
  after (`graphs/lamps.esn`). Every way a lamp's light is taken (on the
  screen, off it, in the air) goes through `lamp_irradiance` or
  `lamp_airlight`, through its cookie and profile (`lamp_look`), so a
  light function or a fixture's photometry shapes the light, its bounce
  and its beam in the fog alike.
- **The mirror.** `light.mirror_lens` reflects the lens in a level plane;
  the scene draws its G-buffer again through it from the plane up
  (`mirror_start` says where a ray reaches the plane), a renderer lights
  that, and `light.mirror_mix` lays it over the pixels `more` w marks, by
  Fresnel, moved by the surface's normal (waves) and drawn out downward.
- **The air** (`graphs/volumetrics.esn`). The air is a grid of cells over
  the view (froxels: `columns` by `rows` across the picture, `slices` out
  from the eye, each slice the same share deeper than the last from `near`
  to `far`). `light.froxel_light` lights each cell where its column shows
  (`light.froxel_furthest`): height fog (its density averaged exactly
  through the cell) and the scene's fog volumes (`volumes`, `FogVolume`:
  boxes and balls of their own density and tint, soft at the edge;
  `light.fog_volumes_none` for none), lit by the sun (shadowed by the
  sun's cascades where they hold the point, `sun_view_lit`, through the
  proxies elsewhere, and by the clouds' deck: shafts behind what stands in
  the light and between the clouds; a scene without cascades gives
  `light.sun_none`), the sky, a glow of the scene's own and the lamps
  light: the lamps of each column's tile (`light.lamps_tiles` with
  `through`, cut for the whole view ray, not the surface), each one's
  light summed exactly through the cell in `lamp_steps` pieces (a beam
  beside a bulb has no spikes, its forward glow no gaps) and cut by the
  proxies toward it into shafts. `light.froxel_scan` sums the cells from
  the eye in two passes; `light.froxel_view` reads the sum at each
  quarter pixel's depth, the slice it falls in taken in part exactly;
  `light.fog_blur` smooths it by depth; `light.clouds_march` walks a deck of clouds for
  the sky at an eighth of the size, lit through the cloud toward the sun
  with light scattered again and again, dark at a sunlit edge, lit from
  below by `below`; the sky's share of the fog's light (`sky_share`, not
  shadowed) and the steps toward the sun (`sun_steps`) are the scene's to
  set, a room taking less sky and more steps past its thin walls;
  `light.fog_apply` lays both over the picture,
  upsampled by depth. Nothing needs history to settle.
- **The sky** (`graphs/atmosphere.esn`). Sunlight through a sphere of
  air round the ground (Hillaire's model): Rayleigh's scattering, haze
  and ozone; light scattered more than once a table by height and the
  sun's angle (`light.atmosphere_spread`, made once). `light.atmosphere_look`
  gives the `SkyLook` (the sun as the air leaves it, the sky overhead and
  at the horizon, the mean sky light) for a sun's height and bearing;
  `light.atmosphere_view` draws the sky round the eye into a table, made
  again only when the sun moves; `light.atmosphere_sky` paints the sky's
  pixels from it after a renderer, with the sun's disc. A scene in these
  units gives lit windows and lamps a few hundred times dimmer than the
  sun, and the exposure carries it from noon to night.
- **Exposure** (`image.entpkg`'s `graphs/exposure.esn`). `image.exposure_tiles`,
  `image.exposure_histogram` (128 bins a quarter stop each, every bin
  counted at once) and `image.exposure_meter` (a `<~` state from
  `image.exposure_start`) meter the picture's mean log luminance, its
  darkest and brightest shares left out (`ignore_dark`, `ignore_bright`: a
  few lamps or a black sky do not move it), and follow it as an eye
  adapts, quicker toward a brighter scene, a dark scene shown darker
  (`night_key`); `image.exposure_apply` scales `hdr` by it before the
  glow and tone. Local exposure then takes each pixel apart into its
  neighborhood's brightness (`image.exposure_local_base`, blurred by
  `image.exposure_local_blur`, read by brightness so no halo bleeds round
  a light) and its own detail: bright and dark neighborhoods are brought
  toward the mean (`highlights`, `shadows`), the detail kept (`detail`); 1
  for all three leaves the picture as it is.
- **Light under the skin** (`graphs/subsurface.esn`). Probe GI, the path
  tracer and lit each give, beside `hdr`, the share of it
  that scatters under subsurface surfaces (`scatter`: `scattered_share`,
  their matte light from the sun and from round about times the
  subsurface, through the air as the rest; none when lit paints);
  `light.subsurface_blur` spreads it across then down (Jimenez's
  separable way), red farthest and blue nearest, as far as the material
  carries light (`reach` meters at full subsurface) and only over the same
  surface; `light.subsurface_apply` lays the spread share back in place
  of the share as it was.
- **Bounced light** (`graphs/bounce.esn`). Lit and ray traced light each
  pixel by the sun and the sky alone; `light.radiance` keeps, in a
  volume laid as the proxy is (a level for each of its levels, `scale`
  times coarser, its cells no finer than `cache_least`), the light leaving
  each surface the proxy and the ground hold: its
  color times the sun and the lamps through the proxies, the sky they
  leave open (lit's
  own `hemisphere` and `proxy_open`) and what four cones gather from the
  cache as the last frame left it, so light bounces once more each time a
  cell is lit. A `parts`th of its cells is lit anew each frame, the rest
  kept from the last frame's where they lie (a block moved on keeps what
  it held), and `light.radiance_turn` stops each level `settle` frames
  after its own block, the sun or the lamps last moved (a fine level moving
  with the eye is lit again alone; the scene's own rest), the whole once
  none is lit. `light.bounce_gather` traces five cones
  about each normal every `every` pixels (`radiance_cone`: each step read
  in the first level as wide as the cone, shut by how near the proxy's
  things or the ground are against its width, what the cache holds laid
  behind what is nearer; it starts as far out as the cells of the level it
  leaves are wide), and `light.bounce` lays it over each pixel by distance, times
  its diffuse color and baked occlusion. One more cone along each mirror
  direction, as wide as the surface is rough, takes the things it meets in
  place of the renderer's `reflection` from all round, as much as they
  shut it (a metal in a room reflects the room, not the sky), and the
  reflection as it now shows goes on to `light.reflections`.
- **Reflections on the screen** (`light.reflections`). On a surface
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
- **Glow** (`image.entpkg`'s `graphs/bloom.esn`): the bright parts shrunk by halves six
  times and grown back, every level adding the larger glow below it.

### Transmission

Refraction is geometry, so the scene traces it: where the view ray meets a
transmissive surface it refracts in, finds the far side, refracts out and
follows the path to what lies behind, handing that over as the `behind_*`
layer with the absorption along the way. A renderer lights that point
(`behind_radiance` in `kernels/transmission.ent`, its diffuse light from the
renderer's own means), blurs it by the glass's roughness (`frosted`) and adds
what Fresnel lets through (`transmitted`) under the surface's reflection.

### Streaming

A world larger than memory is cut into chunks on a lattice of x and z
(`mesh.entpkg`'s `graphs/chunk.esn`: the faces meeting a cell clipped to
it, built as a clustered library with each point's part as its group, the
points on the cell's edge locked where they are, every chunk on the world's
one level 0 cell), each saved twice, near as built and far with only the
clusters it can still draw from where it stands in (those whose parent
bricks lie further than `far_error` meters from the mesh, a pixel there:
lossless from that distance on, and a brick with no finer cluster kept
drawn at any distance, so nothing opens while the near chunk is on its
way), every array in a directory of its own (`entelechy run --save
chunk.points=...`, `chunk.far_points=...`). A scene streams them round
the camera: `surface.stream_wanted` names the chunks within reach, nearest
first, `surface.stream_paths` spells their files for one `fields` node an
array, and `mesh.entpkg`'s `graphs/streamed.esn` joins what is in into
one library the cluster draw, the shadows and the lamps' maps take as any
other; a chunk with an array on its way is not drawn
(`surface.stream_insts`). A `fields` node reads the paths a frame behind,
so the instances pair with the last frame's chunks. The proxy the light
is traced through is baked from the chunks too (`graphs/proxy_stream.esn` over
`mesh.entpkg`'s `graphs/finest.esn`: each chunk's finest level), again
whenever the chunks in change, a pass a frame, the last proxy in use until
the new one is made, so no frame bakes it whole. A far chunk is the
smaller the farther it stands in: in a room seen from inside nearly every
cluster is still needed four meters off.

### Translucency

A G-buffer holds one surface a pixel, so a renderer sees nothing through
glass or smoke. Translucent things are drawn apart (`graphs/translucent.esn`)
once the picture is lit: their mesh cut at the near plane, laid into tiles
of the screen, every pixel a face covers in front of the G-buffer's surface
shaded by the face's look (`TranslucentLook`: glass mirrors the screen, the
bounce's cache and the sky by Fresnel and lets its color through; smoke is
lit by the sun through the proxies, the sky and the lamps and hides as much
as it covers; a glow adds its light) and summed, each layer in no order
(McGuire and Bavoil's weighted, blended transparency): the light each adds
by a weight that favors the nearer, its coverage likewise, the log of what
it lets through, each color apart, so colored panes behind panes tint as
they should. Every sum is 32-bit fixed point: the layers in any order give
the same bits, and a device sums them itself. Panes behind panes all show;
what one layer's light does to the next (a pane mirrored in a pane) is
the weighted mean's, not an exact order's.

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
refract. `image.tonemap` tone maps by its `curve` (`image.entpkg`'s `aces`, none
— the light clipped past 1, as Blender's Standard view — or `agx`, Blender's
own) and encodes sRGB.

### Pieces

- `kernels/camera.ent`: `no_hit`, `screen_march`, the
  `Proxy` frame and its rough tracing (`rough_trace`, `rough_shadow`,
  `rough_normal`), `cosine_dir`.
- `kernels/sky.ent`: `SkyLook`, `sky_radiance`, `aerial`.
- `kernels/brdf.ent`: `Pbr` and the standard BRDF.
- `kernels/resolution.ent`: a resolution of the scene's own, brought to the
  window at the end, so the window's size costs little. `camera.extent`
  is the window scaled down to at most `pixels` pixels, times the share
  `camera.dynamic` sets from the frame time to keep `fps` (dynamic
  resolution); `camera.canvas` is an empty raster canvas of that size.
  `upscaled` (and `image.upscale`) is an edge-aware upscale (FSR 1's EASU),
  `image.contrast_sharpen` a sharpening that cannot overshoot (RCAS), both in display
  space and each frame on its own (`image.entpkg`'s). Its
  `graphs/tonemap.esn` finishes at the picture's size; its
  `graphs/upscale.esn` upscales and sharpens to the window.
  A scene that draws its HUD at the window's size reads the picture through
  `upscaled` in that pass instead, so the window is written once.
  `camera.extent`'s `samples` supersamples (2: four points a pixel), and
  `upscaled` then averages each pixel's block.
- `kernels/haze.ent`: `light.haze`, aerial perspective as the air is: it
  thins with height (Rayleigh's by 8 km, Mie's haze by 1.2 km), integrated
  along the ray; Rayleigh scatters blue (far things bluer, flatter), Mie
  forward (bright toward the sun); lit once by the sun and the sky. A scene
  that uses it sets its `SkyLook`'s `air` to 0.
- `kernels/depth_of_field.ent`: depth of field as data (`Dof`: the
  aperture, 0 sharp, and where it focuses, a share of the picture), made by
  `camera.dof_start` or by a scene's own kernel (keys, the pointer, a
  story's cut). `camera.focus` follows the distance there (a camera's
  focus); `camera.focused` blurs by each pixel's circle of confusion,
  gathering 24 points on a Vogel disc, a point counted where its own circle
  reaches, a farther one only where the pixel is blurred too, so a sharp
  foreground takes no background.
- `kernels/outline.ent`: `surface.outline`, a picture's outlines from its
  G-buffer: a pixel much nearer than a neighbor (`outline_near`, the near
  side of an edge) and a surface turning sharply between neighbors
  (`outline_crease`), fading by `reach`.
- Any pass is kept to part of a picture by `image.entpkg`'s
  `image.masked`: the picture before it, the picture after it and a mask,
  so a scene grades, inks or blurs only what a mask asks (a model drawn
  among real things, a selection, a story's region).
- `kernels/composite.ent`: two G-buffers of one view made one, each pixel the
  nearer surface's (`surface.nearer`), for a scene that draws its parts apart.
