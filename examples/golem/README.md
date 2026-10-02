# Golem

A stone golem walking in place, made with two packages. Its body is
eleven voxel shapes baked once at rest (`voxel.shapes@1`), each voxel
holding its look and the joint it moves with (`voxel_body`): the torso
and a tuft of moss with the hips, the head and its eyes, two arms and
two legs with their feet. Its skeleton of six joints and its walk (the
legs swinging about the hips, the arms against them, the hips bobbing,
the head nodding) are the `skin` package's rows, and `skin.palette@1`
composes them as it would for a mesh.

`voxel.unposing@1` turns each joint's matrix back, from the pose to
rest, and `voxel.posed@1` fills a box of 48 voxels a side every frame:
each voxel is carried back to rest by each joint in turn and takes the
look of the rest voxel it lands in when that voxel moves with that
joint. A voxel is 5 cm, the skin's units meters. The `voxel` package's
view draws it.

```sh
orla-player examples/golem/scene.osn
orla-player examples/golem/scene.osn --param yaw=2.0
```

## Tests

- `walk.replay`: at rest the right foot fills its voxel; a quarter of a
  second on it has swung half a radian forward and up, leaving that
  voxel and filling the one the swing reaches, and the sky shows
  between the legs.
