//! The repository's own tasks. `cargo xtask verify` is the verification
//! PROCESS.md describes: every step timed into `target/verify.log`, which
//! `tail -f` shows as it goes, each step's output in
//! `target/verify-<step>.out`. Steps that need nothing of one another run
//! at once; the first group with a failure stops the run, the failing
//! output's end in the log. The log begins with the commit it verifies,
//! so a log is never mistaken for a later run's. `cargo xtask packages`
//! holds the packages to their names alone.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Mutex;
use std::time::Instant;

mod packages;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("verify") => verify(args.get(1).map(String::as_str) == Some("--all")),
        Some("packages") => {
            let faults = packages::faults(&root().join("packages"));
            for fault in &faults {
                eprintln!("{fault}");
            }
            if faults.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        _ => {
            eprintln!("usage: cargo xtask verify [--all] | packages");
            ExitCode::FAILURE
        }
    }
}

/// The repository's root: two levels above this crate's directory.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("the crate sits at crates/interface/xtask")
        .to_path_buf()
}

/// The log every step writes its line to.
struct Log {
    file: Mutex<File>,
}

impl Log {
    fn line(&self, text: &str) {
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = writeln!(file, "{text}");
    }
}

/// What a step writes its output to.
type Out = Mutex<File>;

/// A step: its name, and what it does, its output written to `out`.
type Step<'a> = (
    &'a str,
    Box<dyn Fn(&Out) -> Result<(), String> + Send + Sync + 'a>,
);

fn verify(all: bool) -> ExitCode {
    let root = root();
    if let Err(error) = std::env::set_current_dir(&root) {
        eprintln!("{}: {error}", root.display());
        return ExitCode::FAILURE;
    }
    let _ = fs::create_dir_all("target");
    let file = match File::create("target/verify.log") {
        Ok(file) => file,
        Err(error) => {
            eprintln!("target/verify.log: {error}");
            return ExitCode::FAILURE;
        }
    };
    let log = Log {
        file: Mutex::new(file),
    };
    let head =
        output(Command::new("git").args(["rev-parse", "--short", "HEAD"])).unwrap_or_default();
    log.line(&format!("verifying {}", head.trim()));
    let rust = all
        || changed_since_main(&[
            "crates/",
            "bridges/",
            ".cargo/",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain",
        ]);
    // The IR, its compiler and the backends: what decides whether one
    // program comes out the same every way it runs.
    let engine = all
        || changed_since_main(&[
            "crates/core/",
            "crates/frontend/",
            "crates/backends/",
            ".cargo/",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain",
        ]);
    if !rust {
        log.line("no Rust changed since main: its steps and wasi left out (--all runs them)");
    }
    if !engine {
        log.line("no IR, compiler or backend changed since main: replays on jit and gpu alone");
    }
    let groups: Vec<Vec<Step>> = vec![
        vec![cargo("fmt", &["fmt", "--all"])],
        vec![("packages", Box::new(named_packages))],
        vec![cargo(
            "clippy",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )],
        vec![cargo(
            "clippy-wasm",
            &[
                "clippy",
                "-p",
                "entelechy-jit",
                "-p",
                "entelechy-ffi",
                "--features",
                "entelechy-jit/wasm-host",
                "--target",
                "wasm32-unknown-unknown",
                "--",
                "-D",
                "warnings",
            ],
        )],
        vec![cargo(
            "clippy-page",
            &[
                "clippy",
                "-p",
                "entelechy-gpu",
                "-p",
                "entelechy-documents",
                "-p",
                "entelechy-session",
                "--target",
                "wasm32-unknown-unknown",
                "--",
                "-D",
                "warnings",
            ],
        )],
        vec![cargo("test-build", &["test", "--workspace", "--no-run"])],
        // The bridges wait on the programs they drive, so they run beside
        // the rest.
        vec![
            cargo(
                "tests",
                &[
                    "test",
                    "--workspace",
                    "--",
                    "--skip",
                    "bridge",
                    "--skip",
                    "blender",
                ],
            ),
            cargo("bridges", &["test", "-p", "entelechy", "--test", "bridges"]),
        ],
        vec![cargo("release", &["build", "--release"])],
        vec![
            ("replays", Box::new(move |out: &Out| replays(out, engine))),
            ("wasi", Box::new(wasi)),
            ("against_main", Box::new(against_main)),
        ],
    ];
    let rust_only = [
        "fmt",
        "clippy",
        "clippy-wasm",
        "clippy-page",
        "test-build",
        "tests",
        "bridges",
        "wasi",
    ];
    for group in &groups {
        let group: Vec<&Step> = group
            .iter()
            .filter(|(name, _)| rust || !rust_only.contains(name))
            .collect();
        if !group.is_empty() && !together(&log, &group) {
            return ExitCode::FAILURE;
        }
    }
    log.line("passed");
    ExitCode::SUCCESS
}

/// Runs a group's steps at once; whether all passed. Each step's line says
/// its time; a failed one's output ends the log's lines for it.
fn together(log: &Log, group: &[&Step]) -> bool {
    std::thread::scope(|scope| {
        let runs: Vec<_> = group
            .iter()
            .map(|(name, run)| {
                scope.spawn(move || {
                    let path = format!("target/verify-{name}.out");
                    let out = match File::create(&path) {
                        Ok(file) => Mutex::new(file),
                        Err(error) => {
                            log.line(&format!("{name:<14}    0 s  FAILED: {path}: {error}"));
                            return false;
                        }
                    };
                    let start = Instant::now();
                    let result = run(&out);
                    let seconds = start.elapsed().as_secs();
                    match result {
                        Ok(()) => {
                            log.line(&format!("{name:<14} {seconds:>4} s"));
                            true
                        }
                        Err(why) => {
                            let said = fs::read_to_string(&path).unwrap_or_default();
                            let lines: Vec<&str> = said.lines().collect();
                            let tail = lines[lines.len().saturating_sub(40)..].join("\n");
                            log.line(&format!("{name:<14} {seconds:>4} s  FAILED: {why}\n{tail}"));
                            false
                        }
                    }
                })
            })
            .collect();
        // Every step is joined, each its line written, before the group's
        // answer.
        let passed: Vec<bool> = runs
            .into_iter()
            .map(|run| run.join().unwrap_or(false))
            .collect();
        passed.into_iter().all(|ok| ok)
    })
}

/// Whether the branch, its uncommitted and untracked files counted, changes
/// a path under one of `places` since it left `main`. On `main` itself, or
/// when git cannot say, it does.
fn changed_since_main(places: &[&str]) -> bool {
    let Some(base) = output(Command::new("git").args(["merge-base", "main", "HEAD"])) else {
        return true;
    };
    let head = output(Command::new("git").args(["rev-parse", "HEAD"])).unwrap_or_default();
    if base.trim() == head.trim() {
        return true;
    }
    let (Some(changed), Some(untracked)) = (
        output(Command::new("git").args(["diff", "--name-only", base.trim()])),
        output(Command::new("git").args(["ls-files", "--others", "--exclude-standard"])),
    ) else {
        return true;
    };
    changed
        .lines()
        .chain(untracked.lines())
        .any(|path| places.iter().any(|place| path.starts_with(place)))
}

/// A step that runs cargo with `args`.
fn cargo<'a>(name: &'a str, args: &'a [&'a str]) -> Step<'a> {
    (
        name,
        Box::new(move |out: &Out| run(out, Command::new("cargo").args(args))),
    )
}

/// Runs `command`, its output written to `out`; an error names how it ended.
fn run(out: &Out, command: &mut Command) -> Result<(), String> {
    let said = command
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("{command:?}: {error}"))?;
    let mut file = out
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _ = file.write_all(&said.stdout);
    let _ = file.write_all(&said.stderr);
    if said.status.success() {
        Ok(())
    } else {
        Err(format!("{:?} ended {}", command.get_program(), said.status))
    }
}

/// What `command` writes to standard output, when it succeeds.
fn output(command: &mut Command) -> Option<String> {
    let said = command.stdin(Stdio::null()).output().ok()?;
    said.status
        .success()
        .then(|| String::from_utf8_lossy(&said.stdout).into_owned())
}

/// The files `pattern` names, `examples/*/tests/*.replay` or
/// `examples/*/scene.esn`: one `*` a directory, one a file name; sorted.
fn listed(dir: &str, inner: Option<&str>, suffix: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let place = match inner {
            Some(inner) => entry.path().join(inner),
            None => entry.path(),
        };
        if let Ok(files) = fs::read_dir(&place) {
            for file in files.flatten() {
                let path = file.path();
                if path.to_string_lossy().ends_with(suffix) {
                    found.push(path);
                }
            }
        }
    }
    found.sort();
    found
}

/// Runs `each` over `items`, `at_once` at a time; whether all passed.
fn pool<T: Sync>(items: &[T], at_once: usize, each: impl Fn(&T) -> bool + Sync) -> bool {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..at_once {
            scope.spawn(|| {
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some(item) = items.get(k) else { break };
                    if !each(item) {
                        failed.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            });
        }
    });
    !failed.load(std::sync::atomic::Ordering::SeqCst)
}

const ENTELECHY: &str = "./target/release/entelechy";

/// Every example's replays, by the release build: each backend in turn;
/// then the device running every Kernel itself, nothing left to the
/// processor by its size; every node apart, fusing nothing; Kernels
/// rounding apart, as another device's might (a replay whose checks hold
/// two forms alike says so and is left out). Then every package used in
/// one scene beside every other: none declares a name another does.
fn replays(out: &Out, engine: bool) -> Result<(), String> {
    let mut all = listed("examples", Some("tests"), ".replay");
    all.extend(listed("packages", Some("tests"), ".replay"));
    let modes: [&[&str]; 6] = [
        &["--backend", "gpu"],
        &["--backend", "jit"],
        &["--backend", "cpu"],
        &["--backend", "gpu", "--small", "0"],
        &["--backend", "jit", "--no-fusion"],
        &["--backend", "jit", "--round-apart"],
    ];
    // The two backends that do the work; the rest hold the IR to one
    // answer every way it runs, and wait for a change that could part it.
    let modes = if engine { &modes[..] } else { &modes[..2] };
    for &mode in modes {
        let passed = pool(&all, 4, |replay| {
            run(
                out,
                Command::new(ENTELECHY).arg("replay").args(mode).arg(replay),
            )
            .is_ok()
        });
        if !passed {
            return Err(format!("a replay failed with {}", mode.join(" ")));
        }
    }
    let together = Path::new("target/packages-together");
    fs::create_dir_all(together).map_err(|e| e.to_string())?;
    let mut scene = String::new();
    for package in listed("packages", None, ".entpkg")
        .iter()
        .filter(|p| p.is_dir())
    {
        scene.push_str(&format!("use \"../../{}\";\n", package.display()));
    }
    scene.push_str(
        "node start: Rows<Int> { domain = 3; value = 1; }\noutput result <- start.result;\n",
    );
    fs::write(together.join("scene.esn"), scene).map_err(|e| e.to_string())?;
    run(
        out,
        Command::new(ENTELECHY)
            .arg("check")
            .arg(together.join("scene.esn")),
    )
}

/// The repository's packages held to their names ([`packages`]).
fn named_packages(out: &Out) -> Result<(), String> {
    let faults = packages::faults(Path::new("packages"));
    let mut file = out
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for fault in &faults {
        let _ = writeln!(file, "{fault}");
    }
    if faults.is_empty() {
        Ok(())
    } else {
        Err(format!("{} fault(s) in the packages' names", faults.len()))
    }
}

/// The WebAssembly build under WASI, the system interface it is held to:
/// a session served over standard input, every answer `ok`, and the
/// replays that read and write through it (pictures, bytes, saves, Field
/// files and directories streamed, models, sound, laws written to files,
/// files found through a package's name), on the processor.
fn wasi(out: &Out) -> Result<(), String> {
    run(
        out,
        Command::new("cargo").args([
            "build",
            "--release",
            "-p",
            "entelechy",
            "--target",
            "wasm32-wasip1",
        ]),
    )?;
    let program = "target/wasm32-wasip1/release/entelechy.wasm";
    let requests = [
        r#"{"id":1,"op":"kernels"}"#,
        r#"{"id":2,"op":"evaluate","time":0.0,"window":[64,48]}"#,
        r#"{"id":3,"op":"read","output":"result","format":"summary"}"#,
        r#"{"id":4,"op":"help","name":"mix"}"#,
        r#"{"id":5,"op":"quit"}"#,
    ];
    let mut served = Command::new("wasmtime")
        .args(["run", "--dir", ".", program, "serve"])
        .args(["examples/data/scene.esn", "--backend", "cpu"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("wasmtime: {error}"))?;
    if let Some(mut input) = served.stdin.take() {
        input
            .write_all((requests.join("\n") + "\n").as_bytes())
            .map_err(|error| error.to_string())?;
    }
    let said = served
        .wait_with_output()
        .map_err(|error| error.to_string())?;
    let answers = String::from_utf8_lossy(&said.stdout).into_owned();
    out.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .write_all(answers.as_bytes())
        .map_err(|error| error.to_string())?;
    let lines: Vec<&str> = answers.lines().filter(|l| l.starts_with('{')).collect();
    if lines.len() != requests.len() || lines.iter().any(|l| !l.contains(r#""ok":"#)) {
        return Err("a request served under WASI was not answered ok".into());
    }
    let through = ["data", "mesh", "sound", "animation", "genesis"];
    let mut all: Vec<PathBuf> = listed("examples", Some("tests"), ".replay")
        .into_iter()
        .filter(|replay| {
            replay
                .components()
                .nth(1)
                .is_some_and(|example| through.iter().any(|name| example.as_os_str() == *name))
        })
        .collect();
    // Field directories read a page at a time (the data example's
    // `stream.replay`), inline where there is no thread, and files found
    // through a package's name.
    all.push(PathBuf::from("packages/text.entpkg/tests/face.replay"));
    let passed = pool(&all, 8, |replay| {
        run(
            out,
            Command::new("wasmtime")
                .args(["run", "--dir", ".", program, "replay", "--backend", "cpu"])
                .arg(replay),
        )
        .is_ok()
    });
    if passed {
        Ok(())
    } else {
        Err("a replay failed under WASI".into())
    }
}

/// Every example as `main`'s build and the branch's say it: its check, its
/// graph and its picture, compared; every difference printed. `main` is
/// checked out beside the branch and built into a target directory kept
/// between runs. Both builds read the branch's documents; where the branch
/// raised the language's version, which main's build cannot read, each
/// reads its own, the same examples written for each language.
fn against_main(out: &Out) -> Result<(), String> {
    let dir = Path::new("target/against-main");
    for side in ["main", "branch", "tree"] {
        let _ = fs::remove_dir_all(dir.join(side));
    }
    run(out, Command::new("git").args(["worktree", "prune"]))?;
    fs::create_dir_all(dir.join("main")).map_err(|e| e.to_string())?;
    fs::create_dir_all(dir.join("branch")).map_err(|e| e.to_string())?;
    let tree = dir.join("tree");
    run(
        out,
        Command::new("git")
            .args(["worktree", "add", "-f"])
            .arg(&tree)
            .arg("main"),
    )?;
    let built = run(
        out,
        Command::new("cargo")
            .args(["build", "--release", "-p", "entelechy", "--manifest-path"])
            .arg(tree.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", dir.join("target")),
    );
    if built.is_ok() {
        let absolute = |path: &Path| fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let main_entelechy = absolute(&dir.join("target/release/entelechy"));
        let here = absolute(Path::new("."));
        let main_documents = if language_version(&tree) == language_version(Path::new(".")) {
            here.clone()
        } else {
            absolute(&tree)
        };
        for scene in listed("examples", None, "scene.esn") {
            let example = scene
                .parent()
                .and_then(Path::file_name)
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            for (side, entelechy, documents) in [
                ("main", main_entelechy.as_path(), main_documents.as_path()),
                (
                    "branch",
                    absolute(Path::new(ENTELECHY)).as_path(),
                    here.as_path(),
                ),
            ] {
                let at = absolute(&dir.join(side)).join(&example);
                // What a build says, its documents' place written as the
                // branch's so the two sides read alike.
                let said = |args: &[&str]| {
                    let said = Command::new(entelechy)
                        .args(args)
                        .arg(&scene)
                        .current_dir(documents)
                        .stdin(Stdio::null())
                        .output();
                    let text = said
                        .map(|s| [s.stdout, s.stderr].concat())
                        .unwrap_or_default();
                    String::from_utf8_lossy(&text).replace(
                        documents.to_string_lossy().as_ref(),
                        here.to_string_lossy().as_ref(),
                    )
                };
                let _ = fs::write(at.with_extension("check"), said(&["check"]));
                let _ = fs::write(at.with_extension("graph"), said(&["graph"]));
                let picture = at.with_extension("ppm");
                let picture_arg = picture.to_string_lossy().into_owned();
                let ran = said(&[
                    "run",
                    "--backend",
                    "cpu",
                    "--window",
                    "64x48",
                    "--out",
                    &picture_arg,
                ]);
                let _ = fs::write(at.with_extension("run"), ran);
                if let Ok(bytes) = fs::read(&picture) {
                    let _ = fs::write(
                        at.with_extension("picture"),
                        format!("{:016x}\n", fingerprint(&bytes)),
                    );
                    let _ = fs::remove_file(&picture);
                }
            }
        }
    }
    let _ = run(
        out,
        Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(&tree),
    );
    built?;
    let differences = differences(&dir.join("main"), &dir.join("branch"));
    let mut file = out
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for difference in &differences {
        let _ = writeln!(file, "{difference}");
    }
    if differences.is_empty() {
        Ok(())
    } else {
        Err(format!("{} file(s) differ from main's", differences.len()))
    }
}

/// The language version the tree at `root` declares, read from its source.
fn language_version(root: &Path) -> Option<u32> {
    let source = fs::read_to_string(root.join("crates/frontend/language/src/lib.rs")).ok()?;
    let line = source
        .lines()
        .find(|line| line.starts_with("pub const LANGUAGE_VERSION: u32 = "))?;
    line.trim_start_matches("pub const LANGUAGE_VERSION: u32 = ")
        .trim_end_matches(';')
        .parse()
        .ok()
}

/// A picture's bytes reduced to a number that differs where they do
/// (FNV-1a, 64 bits).
fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Each file of `main` and `branch` that differs, beside a line saying
/// how: the lines of each side the other lacks, a line naming how long an
/// evaluation took left out.
fn differences(main: &Path, branch: &Path) -> Vec<String> {
    let names = |dir: &Path| {
        let mut names: Vec<String> = fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    };
    let mut all = names(main);
    for name in names(branch) {
        if !all.contains(&name) {
            all.push(name);
        }
    }
    all.sort();
    let lines = |path: &Path| -> Option<Vec<String>> {
        fs::read_to_string(path).ok().map(|text| {
            text.lines()
                .filter(|line| !line.contains("evaluated in"))
                .map(str::to_owned)
                .collect()
        })
    };
    let mut found = Vec::new();
    for name in all {
        match (lines(&main.join(&name)), lines(&branch.join(&name))) {
            (Some(a), Some(b)) if a == b => {}
            (Some(a), Some(b)) => {
                let mut said = format!("{name} differs:");
                for line in a.iter().filter(|line| !b.contains(line)) {
                    said.push_str(&format!("\n< {line}"));
                }
                for line in b.iter().filter(|line| !a.contains(line)) {
                    said.push_str(&format!("\n> {line}"));
                }
                found.push(said);
            }
            (None, _) => found.push(format!("{name}: only the branch's")),
            (_, None) => found.push(format!("{name}: only main's")),
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line that differs is a difference, and a file one side lacks; how
    /// long an evaluation took is not.
    #[test]
    fn differences_name_what_differs_and_leave_the_time_out() {
        let dir = std::env::temp_dir().join(format!("xtask-differences-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let (main, branch) = (dir.join("main"), dir.join("branch"));
        fs::create_dir_all(&main).unwrap();
        fs::create_dir_all(&branch).unwrap();
        fs::write(main.join("same.run"), "loaded\nevaluated in 3 ms\n").unwrap();
        fs::write(branch.join("same.run"), "loaded\nevaluated in 9 ms\n").unwrap();
        fs::write(main.join("moved.graph"), "a\nb\n").unwrap();
        fs::write(branch.join("moved.graph"), "a\nc\n").unwrap();
        fs::write(branch.join("new.picture"), "1\n").unwrap();
        let found = differences(&main, &branch);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(
            found[0].starts_with("moved.graph differs:\n< b\n> c"),
            "{found:?}"
        );
        assert_eq!(found[1], "new.picture: only the branch's");
        let _ = fs::remove_dir_all(&dir);
        assert_ne!(
            fingerprint(b"P6 1 1 255 abc"),
            fingerprint(b"P6 1 1 255 abd")
        );
    }
}
