//! Preparing seen from the terminal (the CLI's commands, and the Player
//! before its window opens): while a session's Kernels take longer
//! than a moment to prepare, a line of seeds, one for each Kernel (or a few
//! where there are many), waiting `·`, compiled `•`, built `●` in gold, the
//! ones building glowing; then how many are built and the one taking
//! longest. Once they are, the line goes and one line says what it took.
//! Where standard error is no terminal, only that last line is written.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use entelechy_kernel::{Preparing, Progress, Stage};

/// How long preparing goes unseen.
const QUIET: Duration = Duration::from_millis(500);
/// The shortest preparing that is said when done.
const SAID: Duration = Duration::from_secs(1);
/// The most seeds drawn; more Kernels share them.
const SEEDS: usize = 32;

/// Whether a line of seeds stands on the terminal now.
static DRAWN: Mutex<bool> = Mutex::new(false);

/// Takes the line of seeds away, so what is written next has its own line;
/// drawn again while preparing goes on.
pub fn clear() {
    let mut drawn = DRAWN.lock().unwrap_or_else(PoisonError::into_inner);
    if *drawn {
        let mut err = std::io::stderr().lock();
        let _ = write!(err, "\r\x1b[2K");
        let _ = err.flush();
        *drawn = false;
    }
}

/// Watches until dropped.
pub struct Watch {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// Watches `progress` on a thread of its own.
pub fn watch(progress: Progress) -> Watch {
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    let thread = std::thread::Builder::new()
        .name("entelechy-watch".into())
        .spawn(move || run(&progress, &stopped))
        .ok();
    Watch { stop, thread }
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(progress: &Progress, stop: &AtomicBool) {
    let terminal = std::io::stderr().is_terminal();
    let mut began: Option<Instant> = None;
    // Samples idle in a row: one step's end and the next's start are one
    // preparing, not two.
    let mut idle: u32 = 0;
    loop {
        let stopping = stop.load(Ordering::Relaxed);
        let now = progress.now();
        let busy = now.busy() && !stopping;
        idle = if busy { 0 } else { idle.saturating_add(1) };
        if busy {
            let since = *began.get_or_insert_with(Instant::now);
            if terminal && since.elapsed() >= QUIET {
                let mut drawn = DRAWN.lock().unwrap_or_else(PoisonError::into_inner);
                let mut err = std::io::stderr().lock();
                let _ = write!(err, "\r\x1b[2K{}", line(&now, since.elapsed()));
                let _ = err.flush();
                *drawn = true;
            }
        } else {
            clear();
        }
        if (idle >= 2 || stopping)
            && let Some(since) = began.take()
        {
            let mut err = std::io::stderr().lock();
            let took = since.elapsed();
            if took >= SAID && !now.seeds.is_empty() {
                let _ = writeln!(err, "{}", done(&now, took));
            }
            let _ = err.flush();
        }
        if stopping {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// The seeds, the count, and the one taking longest.
fn line(now: &Preparing, elapsed: Duration) -> String {
    let n = now.seeds.len();
    let cells = n.clamp(1, SEEDS);
    let beat = (elapsed.as_millis() / 400).is_multiple_of(2);
    let mut out = String::from("  ");
    for cell in 0..cells {
        let seeds =
            &now.seeds[cell * n / cells..((cell + 1) * n / cells).max(cell * n / cells + 1).min(n)];
        let stage = |s: Stage| seeds.iter().filter(|seed| seed.stage == s).count();
        let (built, building) = (
            stage(Stage::Built),
            stage(Stage::Building) + stage(Stage::Compiling),
        );
        let compiled = stage(Stage::Compiled);
        let (colour, glyph) = if building > 0 {
            (
                if beat {
                    "38;2;255;214;120"
                } else {
                    "38;2;150;112;60"
                },
                "●",
            )
        } else if built == seeds.len() {
            ("38;2;232;186;84", "●")
        } else if built + compiled > 0 {
            ("38;2;170;160;200", "•")
        } else {
            ("38;2;90;86;110", "·")
        };
        out.push_str(&format!("\x1b[{colour}m{glyph}"));
    }
    out.push_str(&format!("\x1b[0m  {}/{n}", now.built()));
    if let Some(seed) = now.longest() {
        out.push_str(&format!(
            "  \x1b[38;2;130;124;150m{} {:.0} s\x1b[0m",
            seed.id,
            seed.took.as_secs_f64()
        ));
    }
    out
}

/// What preparing took, and the Kernel that took longest.
fn done(now: &Preparing, took: Duration) -> String {
    let mut said = format!(
        "prepared {} Kernel(s) in {:.1} s",
        now.seeds.len(),
        took.as_secs_f64()
    );
    if let Some(seed) = now.seeds.iter().max_by_key(|seed| seed.took) {
        said.push_str(&format!(
            "; longest {} {:.1} s",
            seed.id,
            seed.took.as_secs_f64()
        ));
    }
    said
}
