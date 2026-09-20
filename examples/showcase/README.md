# Showcase

A product page for a gamepad, rendered in real time at the window's
resolution: the pad is a
signed-distance model, shaded with a physically based model (GGX
specular, Fresnel, a soft key light, a sky, one bounce of reflection on
the floor, a soft shadow, ambient occlusion), tone-mapped with an ACES
fit. Three nodes and nothing added to the language for it.

```text
begin ─ state (feeds back) ─ view
```

- `show.state` is one row: the camera's yaw and pitch, whether the
  pointer is dragging, the last pointer position and button, the chosen
  colour and finish, whether the turntable runs, and the toolbar item
  under the pointer. A drag on the picture orbits the camera; a release
  without a drag on a toolbar item acts on it. The picture fills the
  window (`domain = window();`) and the toolbar is laid out in logical
  pixels, the window's scale factor (`scale()`) being the unit, so it
  keeps its size on screen; the replay declares a 160 by 90 window at an
  eighth of the scale.
- `show.render` is the picture: the model as a few boxes, capsules and
  cylinders under a smooth minimum, parts numbered so the material and
  the shading know what they hit; the keys or a gamepad sink the buttons
  they press and light them; the toolbar is drawn last, four icons
  centred at the bottom.

The bar: colour (graphite, blue, red, gold), finish (matte, gloss,
metal), turntable on or off, reset the view.

One frame at 1280×720 takes about 27 ms on the GPU including the read
back, so the Player runs at its refresh rate; the CPU takes seconds per
frame, and the replay's window is a sixty-fourth of the pixels. The replay clicks
the first item and reads the scheme, stops the turntable, drags forty
pixels and reads the yaw, then resets.

```sh
cargo run --release -p orla-player -- examples/showcase/scene.osn --scale 1
cargo run -- replay examples/showcase/tests/showcase.replay --backend gpu
```
