//! The command line's commands and their flags, as data: `orla` prints
//! its usage from this table and refuses, before a command runs, a flag
//! the table does not give that command.

/// One `orla` command.
#[derive(Clone, Debug)]
pub struct Command {
    /// What follows `orla`: `run`.
    pub name: &'static str,
    /// The positional arguments and the flags a command requires, as the
    /// synopsis shows them; a flag named here is shown here alone.
    pub args: &'static str,
    /// One sentence of what the command does.
    pub summary: &'static str,
    /// The explanatory text, in lines, as the usage shows it.
    pub doc: &'static str,
    /// Every flag the command reads.
    pub flags: Vec<Flag>,
    /// Whether the arguments go to the program beside `orla`, unchecked.
    pub passes: bool,
}

/// One flag of a command.
#[derive(Clone, Debug)]
pub struct Flag {
    /// The flag itself: `--out`.
    pub name: &'static str,
    /// The value it takes, as the synopsis shows it; `None` for a switch.
    pub value: Option<&'static str>,
    /// Whether it may be given more than once.
    pub repeats: bool,
    /// One line of what it does.
    pub doc: &'static str,
}

const fn switch(name: &'static str, doc: &'static str) -> Flag {
    Flag {
        name,
        value: None,
        repeats: false,
        doc,
    }
}

const fn valued(name: &'static str, value: &'static str, doc: &'static str) -> Flag {
    Flag {
        name,
        value: Some(value),
        repeats: false,
        doc,
    }
}

const fn repeated(name: &'static str, value: &'static str, doc: &'static str) -> Flag {
    Flag {
        name,
        value: Some(value),
        repeats: true,
        doc,
    }
}

/// The limits `orla_session::Limits` reads.
pub fn limit_flags() -> Vec<Flag> {
    vec![
        valued(
            "--memory",
            "<MB>",
            "Mebibytes of Fields the Host may hold at once (2048 by default).",
        ),
        valued(
            "--work",
            "<M>",
            "Million loop iterations one run of a Kernel may take (4000 by default).",
        ),
    ]
}

/// The session's flags `orla_session::SessionOptions::read` reads.
pub fn session_flags() -> Vec<Flag> {
    let mut flags = vec![
        valued(
            "--library",
            "<dir|file>",
            "Kernels the scene uses beside its own.",
        ),
        repeated(
            "--input",
            "<name>=<file.ppm>",
            "Binds the scene's input to a file.",
        ),
        repeated("--param", "<name>=<literal>", "Sets a graph parameter."),
        valued("--backend", "cpu|gpu|jit", "The backend the scene runs on."),
        switch(
            "--no-fusion",
            "Runs pointwise producers apart from their consumers.",
        ),
        repeated(
            "--allow",
            "file.read|file.write=<dir>",
            "Lets the scene read files from, or write its outputs into, a directory.",
        ),
        valued(
            "--grant",
            "<capability,...>",
            "The only capabilities a Kernel may be given.",
        ),
    ];
    flags.extend(limit_flags());
    flags
}

fn window_flag() -> Flag {
    valued(
        "--window",
        "<w>x<h>[@<scale>]",
        "The window a scene sized by `window()` fills, in pixels, with its scale factor.",
    )
}

fn save_flag() -> Flag {
    repeated(
        "--save",
        "<node.port>=<file.field|.png|.ppm|.pgm|.wav>",
        "Writes a node's output to a file, in the format its extension names.",
    )
}

fn with(mut own: Vec<Flag>, shared: Vec<Flag>) -> Vec<Flag> {
    own.extend(shared);
    own
}

/// Every command `orla` runs.
pub fn commands() -> Vec<Command> {
    vec![
        Command {
            name: "check",
            args: "<file.orla|file.osn>...",
            summary: "Compile documents and print their manifests or diagnostics.",
            doc: "\
Parse documents, compile their Kernels, and print manifests or diagnostics.
--ir prints each Kernel's IR as a backend runs it: where its work is
counted (`work spend`) and looked at (`work look`) included.",
            flags: vec![switch(
                "--ir",
                "Prints each Kernel's IR as a backend runs it.",
            )],
            passes: false,
        },
        Command {
            name: "run",
            args: "<scene.osn>",
            summary: "Evaluate a scene once, or keep running it as its files change.",
            doc: "\
Evaluate a scene at time zero; write its picture to --out and each --save, in the
format each file's extension names, and nothing else. With --watch, keep running, at
the seconds since it started, and
republish Kernels as their files change; a rejected change keeps the
running version. --window is the window a scene sized by `window()`
would fill, in pixels, with its scale factor.
--memory says how many mebibytes of Fields the Host may hold at once
(2048 by default); a scene of millions of triangles wants more, and
`run`, `render`, `replay`, `serve`, `graph`, `trace` and `bench` all
take it. --work says how many million loop iterations one run of a
Kernel may take, all its elements together (4000 by default), and
the same commands take it.
--profile prints how long each executed node took; it waits for the
device after each one, so a GPU's numbers are the work, not the
submission, and the frame is slower than it would be. What
measuring a node costs before it does anything is taken off every
line; a node runs alone under a profile, so the lines add up to
more than a frame, and the sum says by how much.
--allow file.write=<dir> lets the scene's outputs written `to` a file
land in that directory and under it, once an evaluation that changed
them commits; nothing is written without it. `render`, `serve` and
the Player take it too. --allow file.read=<dir> lets a served
scene's inputs read files there: under `serve`, where an agent
composes the scene, a scene reads only beside its documents
without it.
--grant field.read,field.write,... names the capabilities a Kernel
may be given, and no others (every one without it): a Kernel that
requires another is refused when it is prepared, and what ran
before goes on running. `render`, `serve`, `bench` and the Player
take it too.",
            flags: with(
                vec![
                    save_flag(),
                    valued(
                        "--out",
                        "<file.png|.ppm|.field>",
                        "Writes the scene's picture to a file.",
                    ),
                    switch(
                        "--watch",
                        "Keeps running and republishes Kernels as their files change.",
                    ),
                    window_flag(),
                    switch("--profile", "Prints how long each executed node took."),
                ],
                session_flags(),
            ),
            passes: false,
        },
        Command {
            name: "render",
            args: "<scene.osn> --frames <n>",
            summary: "Step a scene for a number of frames and write each frame and the sound.",
            doc: "\
Step the scene for n frames at the rate (60 by default) and write each
frame's picture as <dir>/0000.png, 0001.png, ... and the sound output,
block after block, as <dir>/sound.wav at the sound rate (48000 by
default, what Kernels read as rate()); a scene without a picture or a
sound writes none of it. The blocks are joined as they come, whatever
their length: a block of rate() / fps samples keeps the sound in time
with the frames. The clock is the frame count
over the rate, so a render is the same every time; async nodes are
waited for after every frame. ffmpeg makes a video of the directory.
--save writes a node's output after the last frame, as run's does:
how a state trained or settled over many frames is kept.",
            flags: with(
                vec![
                    valued("--frames", "<n>", "How many frames to step."),
                    valued("--rate", "<fps>", "Frames a second (60 by default)."),
                    valued(
                        "--sound-rate",
                        "<hz>",
                        "Sound frames a second (48000 by default).",
                    ),
                    valued(
                        "--out",
                        "<dir>",
                        "The directory the frames and the sound go to (`frames` by default).",
                    ),
                    window_flag(),
                    save_flag(),
                ],
                session_flags(),
            ),
            passes: false,
        },
        Command {
            name: "trace",
            args: "<scene.osn> --node <name> (--element <i> | --at <x> <y> [<z>])",
            summary: "Run one element of a node's Kernel and print every binding it makes.",
            doc: "\
Evaluate the scene, then run one element of a node's Kernel (or one
step of a scan) on the interpreter and print every `let` of it with
its value, and the result. Nothing is committed by the trace.",
            flags: with(
                vec![
                    valued("--node", "<name>", "The node whose Kernel is traced."),
                    valued(
                        "--element",
                        "<i>",
                        "The element, or step of a scan, to trace.",
                    ),
                    valued(
                        "--at",
                        "<x> <y> [<z>]",
                        "The element to trace, by its coordinates.",
                    ),
                    valued(
                        "--time",
                        "<seconds>",
                        "The time the traced evaluation runs at (0 by default).",
                    ),
                    window_flag(),
                ],
                session_flags(),
            ),
            passes: false,
        },
        Command {
            name: "player",
            args: "<scene.osn> [...]",
            summary: "Run the Player.",
            doc: "\
The Player: `orla-player` beside `orla`, run with the same arguments,
which `orla` does not check.",
            flags: Vec::new(),
            passes: true,
        },
        Command {
            name: "import",
            args: "<file> --out <dir> [...]",
            summary: "Run the importer.",
            doc: "\
The importer: `orla-import` beside `orla`, run with the same arguments,
which `orla` does not check.",
            flags: Vec::new(),
            passes: true,
        },
        Command {
            name: "export",
            args: "<file.field>... --out <file> [...]",
            summary: "Run the exporter.",
            doc: "\
The exporter: `orla-export` beside `orla`, run with the same arguments,
which `orla` does not check.",
            flags: Vec::new(),
            passes: true,
        },
        Command {
            name: "help",
            args: "[<name>]",
            summary: "Orla's syntax and API, layer by layer: all of it, or one name in full.",
            doc: "\
With nothing, the whole reference: every type, builtin, form, capability,
package part, protocol operation, wire part, replay line and command.
With a name, every item of that name in full: a builtin's signatures and
capability, a form's shape, a protocol operation's fields, a command's
flags. --json answers as the protocol's `help` does.",
            flags: vec![switch("--json", "answer as the protocol's `help` does")],
            passes: false,
        },
        Command {
            name: "version",
            args: "",
            summary: "Print the versions of Orla and of its contracts.",
            doc: "\
Print Orla's release and the versions of the ABI, the language, the IR
and the package format, each its own.",
            flags: Vec::new(),
            passes: false,
        },
        Command {
            name: "package",
            args: "<dir.orlapkg>",
            summary: "Check a package directory and say what it holds.",
            doc: "Check a package directory as `orla check` checks a document: its manifest, the Kernels it uses compiled, and each preset graph read; then say what it holds. A preset's Kernels are found where a scene places it.",
            flags: Vec::new(),
            passes: false,
        },
        Command {
            name: "pack",
            args: "<dir.orlapkg> [<file.orlapkg>]",
            summary: "Write a package directory as one archive file.",
            doc: "Write a package directory as one archive file, its manifest checked.",
            flags: Vec::new(),
            passes: false,
        },
        Command {
            name: "unpack",
            args: "<file.orlapkg> <new dir>",
            summary: "Read a package archive back into a directory.",
            doc: "Read an archive back into a directory, its manifest checked.",
            flags: Vec::new(),
            passes: false,
        },
        Command {
            name: "replay",
            args: "<file.replay>...",
            summary: "Drive scenes through scripted frames and inputs and check them.",
            doc: "Drive scenes through scripted frames and inputs and check pixels.",
            flags: with(
                vec![valued(
                    "--backend",
                    "cpu|gpu|jit",
                    "The backend the replays run on.",
                )],
                limit_flags(),
            ),
            passes: false,
        },
        Command {
            name: "new",
            args: "<dir>",
            summary: "Write a scene that runs, with a replay that checks it.",
            doc: "\
Write a scene that runs: a Kernel, the scene that places it, a replay
that checks what it draws, and a README saying what to change first.
`picture` is one Kernel over the window, `sound` a voice with a
drawing of it, `game` a grid of state the arrow keys move.",
            flags: vec![valued(
                "--kind",
                "picture|sound|game",
                "What the scene makes (`picture` by default).",
            )],
            passes: false,
        },
        Command {
            name: "bench",
            args: "<scene.osn>",
            summary: "Measure a scene's frames on one backend or several.",
            doc: "\
Step the scene for --frames frames --repeat times on each backend and
print the least of the runs, a millisecond a frame. The first frame
of all is thrown away: it compiles and allocates. --nodes measures
every node as well, waiting for the device after each one, native
operations included; what measuring a node costs before it does
anything is taken off every line, and the sum says how the parts
come to more than the whole, since a node runs alone there and a
frame overlaps them. --json
writes the measurement, and --compare reads one back and prints what
changed since, which is how a change is shown to cost nothing.",
            flags: with(
                vec![
                    valued("--frames", "<n>", "Frames a run steps."),
                    valued("--repeat", "<n>", "Runs on each backend."),
                    valued("--rate", "<fps>", "Frames a second."),
                    repeated(
                        "--backend",
                        "cpu|gpu|jit|all",
                        "A backend to measure on; `all` is every backend this build has.",
                    ),
                    switch("--nodes", "Measures every node as well."),
                    window_flag(),
                    valued("--json", "<file>", "Writes the measurement to a file."),
                    valued(
                        "--compare",
                        "<file>",
                        "Prints what changed since a measurement written by --json.",
                    ),
                ],
                session_flags()
                    .into_iter()
                    .filter(|flag| flag.name != "--backend")
                    .collect(),
            ),
            passes: false,
        },
        Command {
            name: "graph",
            args: "<scene.osn>",
            summary: "Print the flattened graph and its plan.",
            doc: "Print the flattened graph and what would execute, fused groups marked.",
            flags: session_flags(),
            passes: false,
        },
        Command {
            name: "serve",
            args: "<scene.osn>",
            summary: "Serve the creation protocol on stdin and stdout and on --listen.",
            doc: "\
Serve the creation protocol: JSON requests on stdin, one response per
line on stdout, and on every connection to --listen, where a client's
quit ends only its connection. A browser's page connects from this
machine; a page elsewhere only from an origin --allow-origin names
(`https://host:port`, or `null` for a page opened from a file). --window is what evaluations that name
no window see. See
docs/PROTOCOL.md.",
            flags: with(
                vec![
                    valued(
                        "--listen",
                        "<host:port>",
                        "Also answers every connection to the address.",
                    ),
                    window_flag(),
                    repeated(
                        "--allow-origin",
                        "<origin>",
                        "A page's origin elsewhere that may connect.",
                    ),
                ],
                session_flags(),
            ),
            passes: false,
        },
    ]
}

/// The command named `name`.
pub fn command(name: &str) -> Option<Command> {
    commands().into_iter().find(|command| command.name == name)
}

impl Flag {
    /// `[--out <file>]`, `[--input <name>=<file>]...`.
    fn synopsis(&self) -> String {
        let value = self.value.map(|v| format!(" {v}")).unwrap_or_default();
        let repeats = if self.repeats { "..." } else { "" };
        format!("[{}{value}]{repeats}", self.name)
    }
}

impl Command {
    /// Whether the flag is shown in `args`, required.
    fn required(&self, flag: &Flag) -> bool {
        self.args
            .split(|c: char| c.is_whitespace() || "()|".contains(c))
            .any(|word| word == flag.name)
    }

    /// The command's synopsis: `orla run <scene.osn> [--out <file>] ...`,
    /// wrapped under its arguments.
    pub fn synopsis(&self) -> String {
        const WIDTH: usize = 88;
        let head = format!("  orla {}", self.name);
        let indent = " ".repeat(head.len() + 1);
        let mut lines = vec![head];
        let words = std::iter::once(self.args.to_owned())
            .filter(|args| !args.is_empty())
            .chain(
                self.flags
                    .iter()
                    .filter(|flag| !self.required(flag))
                    .map(Flag::synopsis),
            );
        let mut first = true;
        for word in words {
            let line = lines.last_mut().expect("a line");
            if !first && line.len() + 1 + word.len() > WIDTH {
                lines.push(format!("{indent}{word}"));
            } else {
                line.push(' ');
                line.push_str(&word);
            }
            first = false;
        }
        lines.join("\n")
    }

    /// Refuses a flag in `args` this command does not read; a flag's
    /// value is not a flag.
    pub fn check(&self, args: &[String]) -> Result<(), String> {
        if self.passes {
            return Ok(());
        }
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            if !arg.starts_with("--") {
                continue;
            }
            match self.flags.iter().find(|flag| flag.name == arg) {
                Some(flag) => {
                    if flag.value.is_some() {
                        args.next();
                    }
                }
                None => {
                    let takes = if self.flags.is_empty() {
                        "no flags".to_owned()
                    } else {
                        self.flags
                            .iter()
                            .map(|flag| flag.name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    return Err(format!(
                        "`orla {}` takes {takes}; `{arg}` is not one of them{}",
                        self.name,
                        orla_language::did_you_mean(arg, self.flags.iter().map(|flag| flag.name))
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Every command's synopsis and explanatory text.
pub fn usage() -> String {
    let mut text = String::from("usage:");
    for command in commands() {
        text.push('\n');
        text.push_str(&command.synopsis());
        for line in command.doc.lines() {
            text.push_str("\n      ");
            text.push_str(line);
        }
    }
    text
}
