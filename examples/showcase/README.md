# Showcase

A product page for a gamepad, rendered at the window's resolution: a
signed-distance model with physically based shading (GGX, Fresnel, soft
shadow, ambient occlusion, a floor reflection) and ACES tone mapping.

```text
begin ─ state (feeds back) ─ view
```

- `show.state` is one row: camera yaw and pitch, drag and pointer
  state, color, finish, turntable, and the hovered toolbar item. A drag
  orbits; a click on the toolbar acts on it.
- `show.render` draws the model, the pressed buttons (keys or gamepad),
  and a toolbar of four items: color, finish, turntable, reset view.
  The toolbar is laid out in logical pixels (`scale()`), the picture
  fills `window()`.

The replay uses a 160 by 90 window: it clicks the color item, stops
the turntable, drags and reads the yaw, then resets.

```sh
orla player examples/showcase/scene.osn --scale 1
orla replay examples/showcase/tests/showcase.replay --backend gpu
```
