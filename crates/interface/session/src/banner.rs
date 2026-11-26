//! Minimalist ASCII startup banner for interactive terminals ("Living Seed & Open DAG").
//!
//! Renders strictly to `stderr` when attached to an interactive terminal (or when
//! `ENTELECHY_BANNER=1` is set), honoring `NO_COLOR` and `ENTELECHY_NO_BANNER=1`,
//! so machine protocols on `stdout` and piped test captures remain untouched.

use std::io::IsTerminal;
use std::path::Path;

const RESET: &str = "\x1b[0m";
const WHITE_BOLD: &str = "\x1b[1;38;2;255;255;255m";
const BRIGHT: &str = "\x1b[38;2;232;232;232m";
const SEED_AMBER: &str = "\x1b[1;38;2;255;176;0m";
const DIM: &str = "\x1b[38;2;128;128;128m";

/// Metadata displayed beside the Living Seed ASCII emblem.
#[derive(Clone, Debug, Default)]
pub struct BannerInfo<'a> {
    /// Active command mode, e.g. `"player"`, `"serve"`, or `"cli"`.
    pub mode: &'a str,
    /// Version string, e.g. `env!("CARGO_PKG_VERSION")`.
    pub version: &'a str,
    /// Selected compute backend name (`"gpu"`, `"jit"`, `"cpu"`), if opened.
    pub backend: Option<&'a str>,
    /// Protocol listener endpoint (e.g. `"127.0.0.1:7788"` or `"stdin/stdout"`).
    pub listen: Option<&'a str>,
    /// Concise stage/scene label (e.g. `"mario/scene.esn"` or `"empty stage"`).
    pub stage: Option<&'a str>,
}

/// Returns whether the interactive startup banner should print to `stderr`.
///
/// Suppressed when `ENTELECHY_NO_BANNER` is non-empty; forced when
/// `ENTELECHY_BANNER` is non-empty; otherwise enabled only when `stderr` is a TTY.
pub fn should_print() -> bool {
    if env_truthy("ENTELECHY_NO_BANNER") {
        return false;
    }
    if env_truthy("ENTELECHY_BANNER") {
        return true;
    }
    std::io::stderr().is_terminal()
}

/// Returns whether ANSI colors are enabled for `stderr`.
///
/// Follows the `NO_COLOR` convention and disables colors when `TERM=dumb`.
pub fn use_color() -> bool {
    if env_present("NO_COLOR") {
        return false;
    }
    if matches!(std::env::var("TERM").ok().as_deref(), Some("dumb")) {
        return false;
    }
    env_truthy("ENTELECHY_BANNER") || std::io::stderr().is_terminal()
}

/// Formats a scene path into a compact `parent/file` stage label.
pub fn stage_label(path: &Path) -> String {
    let file = path
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    if let Some(parent) = path.parent().and_then(|p| p.file_name()) {
        format!("{}/{file}", parent.to_string_lossy())
    } else if !file.is_empty() {
        file.into_owned()
    } else {
        path.display().to_string()
    }
}

/// Renders the 4-line pure-ASCII Living Seed banner as a string.
pub fn render(info: &BannerInfo<'_>, color: bool) -> String {
    let paint = |code: &str, text: &str| -> String {
        if color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_owned()
        }
    };

    // Row 0: 3-square open crown + wordmark, version, and mode.
    let r0_mark = paint(WHITE_BOLD, "  []  []  []  ");
    let r0_text = if info.mode.is_empty() {
        format!(
            "{}  {}",
            paint(WHITE_BOLD, "entelechy"),
            paint(DIM, &format!("v{}", info.version))
        )
    } else {
        format!(
            "{}  {}  {}  {}",
            paint(WHITE_BOLD, "entelechy"),
            paint(DIM, &format!("v{}", info.version)),
            paint(DIM, "·"),
            paint(BRIGHT, info.mode)
        )
    };

    // Row 1: 2-square bifurcation + runtime backend / protocol listener.
    let r1_mark = paint(WHITE_BOLD, "    []  []    ");
    let r1_text = match (info.backend, info.listen) {
        (Some(b), Some(l)) => format!(
            "{}  {}  {}  {} {}",
            paint(DIM, "backend"),
            paint(BRIGHT, b),
            paint(DIM, "·"),
            paint(DIM, "listen"),
            paint(BRIGHT, l),
        ),
        (Some(b), None) => format!("{}  {}", paint(DIM, "backend"), paint(BRIGHT, b)),
        (None, Some(l)) => format!("{}   {}", paint(DIM, "listen"), paint(BRIGHT, l)),
        (None, None) => paint(DIM, "host · kernel · graph · ai"),
    };

    // Row 2: single Solar Seed (#FFB000) + stage/scene label.
    let r2_mark = format!("      {}      ", paint(SEED_AMBER, "[]"));
    let r2_text = match info.stage {
        Some(stage) => format!("{}    {}", paint(DIM, "stage"), paint(BRIGHT, stage)),
        None => paint(DIM, "run `entelechy help` for the full reference"),
    };

    // Row 3: quiet Host horizon + architectural motto.
    let r3_mark = paint(DIM, "     ----     ");
    let r3_text = paint(DIM, "host is the world · kernel is the law");

    format!(
        "\n{r0_mark}  {r0_text}\n{r1_mark}  {r1_text}\n{r2_mark}  {r2_text}\n{r3_mark}  {r3_text}\n"
    )
}

/// Prints the Living Seed banner to `stderr` when running interactively.
///
/// Returns `true` when printed so callers can avoid duplicating basic startup lines.
pub fn print_stderr(info: &BannerInfo<'_>) -> bool {
    if !should_print() {
        return false;
    }
    eprintln!("{}", render(info, use_color()));
    true
}

fn env_present(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| !v.is_empty())
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_banner_renders_exact_ascii_living_seed_geometry() {
        let out = render(
            &BannerInfo {
                mode: "player",
                version: "0.1.0",
                backend: Some("gpu"),
                listen: Some("127.0.0.1:7788"),
                stage: Some("mario/scene.esn"),
            },
            false,
        );
        let lines: Vec<&str> = out.trim_matches('\n').lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("  []  []  []    entelechy  v0.1.0"));
        assert!(lines[1].starts_with("    []  []      backend  gpu"));
        assert!(lines[1].contains("listen 127.0.0.1:7788"));
        assert!(lines[2].starts_with("      []        stage    mario/scene.esn"));
        assert!(lines[3].starts_with("     ----       host is the world"));
        assert!(!out.contains('\x1b'));
    }

    #[test]
    fn colored_banner_uses_solar_seed_amber_accent() {
        let out = render(
            &BannerInfo {
                mode: "serve",
                version: "0.1.0",
                backend: Some("jit"),
                listen: Some("stdin/stdout"),
                stage: Some("life/scene.esn"),
            },
            true,
        );
        assert!(out.contains("\x1b[1;38;2;255;176;0m[]\x1b[0m"));
        assert!(out.contains("entelechy"));
    }

    #[test]
    fn stage_label_keeps_parent_and_filename() {
        assert_eq!(
            stage_label(Path::new("/tmp/projects/mario/scene.esn")),
            "mario/scene.esn"
        );
        assert_eq!(stage_label(Path::new("scene.esn")), "scene.esn");
    }
}
