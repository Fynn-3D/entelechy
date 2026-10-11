# Decided

How much geometry a frame places, decided inside that frame. A camera
flies over a sheet of ground; `decided.detail` turns its height into one
number, the mesh resolution, in a one-element Field. Every ground node is
sized by it:

```
node ground: decided.ground@1 {
    detail <- detail.result;
    domain = value(detail) * value(detail);
}
node faces: decided.faces@1 {
    detail <- detail.result;
    domain = 2 * (value(detail) - 1) * (value(detail) - 1);
}
```

`value(port)` reads data into a `domain`; the Field holding it is read
back to the Host once a frame. `tests/decided.replay` pins the count at
three heights on all three backends with `expect count`.

```sh
entelechy player examples/decided/scene.esn --backend gpu
entelechy replay examples/decided/tests/decided.replay
```
