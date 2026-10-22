# Controls

A dot walked by the keyboard or a pad alike, through the
[controls package](../../packages/controls.entpkg/package.esn). A game
names its actions in a table (`assets/actions.csv`: a key, a second key
and a pad button an action), and reads one row a frame: which actions
are held, pressed and released, the way to move (W A S D, the arrows or
the left stick past its dead zone), the way to look, the zoom, the
stick's flicks as list steps, and whether the pad was touched last.

W A S D, the arrows or the left stick move the dot; Space or south
jumps, F, J or west fires, Esc or start opens the menu, each lighting
its lamp; the arrows or the stick's flicks step the cursor of the list
on the right; the top right corner is lit while the pad was touched
last.

```sh
entelechy player examples/controls/scene.esn
entelechy replay examples/controls/tests/*.replay --backend gpu
```

`tests/controls.replay` walks the dot by a held key, presses and
releases an action, steps the list by the stick and again after the
delay, and follows which was touched last.
