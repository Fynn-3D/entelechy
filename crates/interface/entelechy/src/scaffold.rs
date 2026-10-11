//! `entelechy new`: a directory that runs: a Kernel, the scene placing it, a
//! replay checking what it draws, and a README. Each kind passes its own
//! replay on every backend.

use std::path::Path;

/// What kind of scene to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A picture over the window, drawn from the clock.
    Picture,
    /// A voice and a drawing of what it sounds like.
    Sound,
    /// A grid of state the arrow keys move, drawn as cells.
    Game,
}

impl Kind {
    /// The kind this name asks for.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "picture" => Some(Self::Picture),
            "sound" => Some(Self::Sound),
            "game" => Some(Self::Game),
            _ => None,
        }
    }

    /// The names `--kind` takes, in the order `entelechy new` lists them.
    pub const NAMES: [&'static str; 3] = ["picture", "sound", "game"];
}

/// Writes a scene of this kind into `dir`, which must not hold one, and
/// returns the files written in order.
pub fn new_scene(dir: &Path, kind: Kind, name: &str) -> Result<Vec<String>, String> {
    if dir.join("scene.esn").exists() {
        return Err(format!(
            "{} already holds a scene; `entelechy new` writes a new one",
            dir.display()
        ));
    }
    let name = clean(name);
    std::fs::create_dir_all(dir.join("kernels")).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::create_dir_all(dir.join("tests")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (kernel, scene, replay, readme) = match kind {
        Kind::Picture => picture(&name),
        Kind::Sound => sound(&name),
        Kind::Game => game(&name),
    };
    let files = [
        (format!("kernels/{name}.ent"), kernel),
        ("scene.esn".to_owned(), scene),
        (format!("tests/{name}.replay"), replay),
        ("README.md".to_owned(), readme),
    ];
    let mut written = Vec::new();
    for (path, text) in files {
        let full = dir.join(&path);
        std::fs::write(&full, text).map_err(|e| format!("{}: {e}", full.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// A name a Kernel id and a file may both take: letters, digits and
/// underscores, starting with a letter.
fn clean(name: &str) -> String {
    let kept: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let kept = kept.trim_matches('_').to_owned();
    match kept.chars().next() {
        Some(first) if first.is_ascii_alphabetic() => kept,
        _ => format!("scene_{kept}"),
    }
}

fn picture(name: &str) -> (String, String, String, String) {
    let kernel = format!(
        r#"// A picture, one Kernel over the window: every pixel works out its
// own color from where it is and what time it is. Nothing here reads
// another Field, so it is the smallest shape a scene can have.
kernel {name}.draw@1 {{
    requires {{ field.write; time.read; }}
    param rate: Float = 0.35;             // how fast the colors turn
    output result: Image;
    compute {{
        // Where this pixel is, from -1 to 1 across the shorter side.
        let {{ x: width, y: height }} = domain();
        let place = (coord() * 2.0 - domain()) / min(width, height);
        let away = length(place);
        let turn = time() * rate;
        // Rings that move outwards, and a color that goes round.
        let ring = sin(away * 8.0 - turn * 6.2831853) * 0.5 + 0.5;
        let color = mix(
            vec3(0.15, 0.25, 0.45),
            vec3(0.95, 0.65, 0.35),
            ring * saturate(1.2 - away)
        );
        return vec4(color, 1.0);
    }}
}}
"#
    );
    let scene = format!(
        r#"// One node: the Kernel over a window's worth of pixels.
use "kernels/";

param rate: Float = 0.35;

node picture: {name}.draw@1 {{ rate = rate; domain = window(); }}

output result <- picture.result;
"#
    );
    let replay = r#"scene ../scene.esn

# A small window, so the pixels are few and the test is quick. The
# middle of the picture at this moment is this color; change the
# Kernel and this is what says so.
window 64 40 1
frame 0.0
expect 32 20 = 0.550 0.450 0.400 0.01
frame 0.5
expect 32 20 = 0.194 0.272 0.445 0.01
"#
    .to_owned();
    let readme = format!(
        r#"# {name}

A picture drawn by one Kernel over the window. Every pixel works out its
own color from where it is and what time it is.

```sh
entelechy player {name}/scene.esn     # a window
entelechy run {name}/scene.esn --window 640x400 --out frame.png
entelechy replay {name}/tests/{name}.replay
```

Change `kernels/{name}.ent` while the Player is open and the picture
changes as the file is saved; a Kernel that does not compile leaves the
running one alone and says why.

What to try next: give the scene another `param` and use it in the
Kernel, or add a second node that reads the first (`input under: Image;`
and `sample(under)`) to draw over it.
"#
    );
    (kernel, scene, replay, readme)
}

fn sound(name: &str) -> (String, String, String, String) {
    let kernel = format!(
        r#"// A voice and a picture of it. The voice writes a block of samples
// every frame, which the scene plays; the picture draws the block it
// just wrote, so what is seen is what is heard.
kernel {name}.voice@1 {{
    requires {{ field.write; time.read; }}
    param pitch: Float = 220.0;
    output result: Rows<Float>;
    compute {{
        // The sample's own place in time, so one block joins the next
        // without a click.
        let hz = if rate() > 0.0 {{ rate() }} else {{ 48000.0 }};
        let at = time() + float(element()) / hz;
        let phase = at * pitch * 6.2831853;
        let tone = sin(phase) + sin(phase * 2.0) * 0.4 + sin(phase * 3.0) * 0.15;
        return tone * 0.2;
    }}
}}

// The block as a line across the window.
kernel {name}.draw@1 {{
    requires {{ field.read; field.write; }}
    input block: Rows<Float>;
    output result: Image;
    compute {{
        let uv = coord() / domain();
        let value = at_linear(block, uv.x * float(count(block)));
        let away = abs(uv.y - (0.5 + value * 0.4)) * domain().y;
        let ink = vec3(0.45, 0.85, 0.75) * saturate(1.5 - away);
        let ground = vec3(0.06, 0.07, 0.10);
        let lit = ground + ink;
        return vec4(lit, 1.0);
    }}
}}
"#
    );
    let scene = format!(
        r#"// The voice writes a block of samples a frame; the scene plays it
// and draws it. `output sound` is what the Player sends to the device.
use "kernels/";

param pitch: Float = 220.0;

node voice: {name}.voice@1 {{ pitch = pitch; domain = 800; }}
node picture: {name}.draw@1 {{ block <- voice.result; domain = window(); }}

output result <- picture.result;
output sound <- voice.result;
"#
    );
    let replay = r#"scene ../scene.esn

# The sample at a known place in the block, before and after the clock
# moves: the tone is the same shape, a little further along.
window 64 40 1
frame 0.0
expect voice.result[100] = 0.032977 0.0005
frame 0.25
expect voice.result[200] = -0.199282 0.0005
"#
    .to_owned();
    let readme = format!(
        r#"# {name}

A voice and a picture of it. The voice writes a block of 800 samples
every frame, which the scene plays through `output sound`, and the
picture draws the block it just wrote.

```sh
entelechy player {name}/scene.esn     # plays it
entelechy render {name}/scene.esn --frames 120 --out out/
entelechy replay {name}/tests/{name}.replay
```

Change `pitch` in the scene, or the partials in the Kernel, and save;
the Player republishes as the file changes.

What to try next: make the pitch follow the pointer (`pointer()` under
`input.read`), or carry a state from frame to frame with `<~` so a key
starts a note that rings on after it is let go.
"#
    );
    (kernel, scene, replay, readme)
}

fn game(name: &str) -> (String, String, String, String) {
    let kernel = format!(
        r#"// A walker the arrow keys move, and the trail it leaves. Two
// states feed back frame to frame: where the walker is, two integers,
// and the grid of marks, which fades by one a frame.
kernel {name}.move@1 {{
    requires {{ field.read; field.write; input.read; }}
    input previous: Rows<Int>;            // element 0 is x, element 1 is y
    param width: Int = 24;
    param height: Int = 16;
    output result: Rows<Int>;
    compute {{
        let axis = element();
        let extent = if axis == 0 {{ width }} else {{ height }};
        let step = if axis == 0 {{
            int(key(39)) - int(key(37))   // right, left
        }} else {{
            int(key(40)) - int(key(38))   // down, up
        }};
        // Off one edge and on at the other.
        return (at(previous, axis) + step + extent) % extent;
    }}
}}

// The grid: the walker's cell takes the brightest mark, every other
// mark fades by one, so the trail shows where it has been.
kernel {name}.trail@1 {{
    requires {{ field.read; field.write; }}
    input previous: Grid<Int>;
    input spot: Rows<Int>;
    param fade: Int = 8;
    output result: Grid<Int>;
    compute {{
        let here = icoord();
        if here == ivec2(at(spot, 0), at(spot, 1)) {{ return fade; }}
        return max(at(previous, here) - 1, 0);
    }}
}}

// The grid as cells, the walker brightest.
kernel {name}.draw@1 {{
    requires {{ field.read; field.write; }}
    input cells: Grid<Int>;
    param fade: Int = 8;
    output result: Image;
    compute {{
        let uv = coord() / domain();
        let cell = int(uv * size(cells));
        let mark = float(at(cells, cell)) / float(max(fade, 1));
        let local = fract(uv * size(cells)) - 0.5;
        let inside = saturate(1.0 - length(local) * 2.4);
        let color = mix(vec3(0.08, 0.09, 0.13), vec3(0.45, 0.90, 0.80), mark * inside);
        return vec4(color, 1.0);
    }}
}}
"#
    );
    let scene = format!(
        r#"// Two states carried from frame to frame with `<~`: where the
// walker is, and the grid it has walked over. The arrow keys move it.
use "kernels/";

param fade: Int = 8;

node origin: Rows<Int> {{ domain = 2; value = 0; }}
node blank: Grid<Int> {{ domain = vec2(24, 16); value = 0; }}

node spot: {name}.move@1 {{
    previous <~ spot.result from origin.result;
    width = 24;
    height = 16;
    domain = 2;
}}
node trail: {name}.trail@1 {{
    previous <~ trail.result from blank.result;
    spot <- spot.result;
    fade = fade;
    domain = vec2(24, 16);
}}
node picture: {name}.draw@1 {{ cells <- trail.result; fade = fade; domain = window(); }}

output result <- picture.result;
"#
    );
    let replay = r#"scene ../scene.esn

# The walker starts in the corner and the right arrow moves it right:
# the cell it left fades by one, the cell it is in takes the brightest
# mark.
window 48 32 1
frame 0.0
expect trail.result[0] = 8
key 39 down
frame 0.0167
expect spot.result[0] = 1
expect trail.result[1] = 8
expect trail.result[0] = 7
"#
    .to_owned();
    let readme = format!(
        r#"# {name}

A walker the arrow keys move, and the trail it leaves. Two states are
carried from frame to frame with `<~`: where the walker is, two
integers, and the grid of marks, which fades by one a frame.

```sh
entelechy player {name}/scene.esn     # arrow keys
entelechy replay {name}/tests/{name}.replay
```

What to try next: hold a `struct` per cell instead of an `Int` so a
trail can have a color as well as an age, or add a second walker on
other keys and a rule for what happens where the trails cross.
"#
    );
    (kernel, scene, replay, readme)
}
