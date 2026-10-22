# Clip

An input read frame by frame: `input clip: Image = frames "clip/" at 2;`
is the files of the directory in name order, one per frame at two a
second of session time, wrapping at the end. The session lists the
directory again on sync. The four frames are eight-pixel squares of red,
green, blue and yellow, scaled up. With `render`, this is a video
pipeline: ffmpeg writes a directory of PNGs, the scene reads it as
`frames`, `render` writes the result.

```sh
entelechy player examples/data/parts/clip/scene.esn
entelechy replay examples/data/tests/clip.replay
ffmpeg -i in.mp4 clip/%04d.png          # a real clip in, at its rate
entelechy render examples/data/parts/clip/scene.esn --frames 120 --rate 30 --out /tmp/out
```
