# Crypt

A crypt drawn with the render package (`packages/render.voxel_entpkg`): a
chamber of 48 by 24 by 48 voxels of a quarter meter, baked once, and a
statue in a box of voxels of its own at the middle, turning. A ray a
pixel walks the chamber and the statue's box (`voxel_meet`, the box's
turn carried into its frame), the camera going round
(`voxel_orbit`, `voxel_ray`).

By firelight (`open` 0) four torches and a lantern going round light what
the rays meet (`voxel_light`, `voxel_flicker`), kept to seven steps
(`voxel_stepped`), and the far is swallowed by the dark (`voxel_dark`).
By day (`open` 1) the vault is open to the sky: the sun casts shadows
(`voxel_shaded`, `voxel_daylight`), the pool's water is seen through to
its floor (`voxel_meet_past`), and the far fades into haze
(`voxel_haze`).

```sh
entelechy-player examples/crypt/scene.esn
entelechy-player examples/crypt/scene.esn --param open=1
```

## Tests

- `firelight.replay`: a torch glows its own color; a corner far from the
  fires is near black; the statue is dimly lit.
- `daylight.replay`: the floor in a pillar's shadow is cool and dark, in
  the sun warm and bright; the statue's box turns, and a pixel meets its
  other side two seconds on.
