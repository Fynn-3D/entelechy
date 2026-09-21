# Clip

An input read frame by frame: `input clip: Image = frames "clip/" at 2;`
is the files of the directory in name order (`.png`, `.ppm`, `.pgm`,
`.field` or `.wav`), one per frame at two a second of the session's
time, wrapping at the end; the session brings in the frame whose turn it
is at each evaluation, and lists the directory again on sync, so frames
dropped in while it runs are seen. The four frames here are eight-pixel
squares of red, green, blue and yellow; the scene scales the frame of
the moment up. With `render`, this is the video pipeline's input half:
ffmpeg writes a clip as a directory of PNGs, a scene reads it as
`frames`, `render` writes the result, ffmpeg makes the video.

```sh
cargo run --release -p orla-player -- examples/clip/scene.osn
cargo run --release -- replay examples/clip/tests/clip.replay
ffmpeg -i in.mp4 clip/%04d.png          # a real clip in, at its rate
cargo run --release -- render examples/clip/scene.osn --frames 120 --rate 30 --out /tmp/out
```
