//! The command table is what `entelechy` reads: every flag it gives a command
//! is one the command takes, a flag it does not give is refused with the
//! command's flags named, and the usage lists every command.

use std::process::{Command, Output, Stdio};

use entelechy::commands::{commands, usage};

fn entelechy(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_entelechy"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("entelechy runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Each flag the table gives a command passes the check; it may fail
/// after, since no scene or document is named.
#[test]
fn every_flag_in_the_table_is_one_the_command_takes() {
    for command in commands().iter().filter(|command| !command.passes) {
        for flag in &command.flags {
            let mut args = vec![command.name, flag.name];
            if flag.value.is_some() {
                args.push("0");
            }
            let output = entelechy(&args);
            let text = stderr(&output);
            assert!(
                !text.contains("is not one of them") && !text.contains("unknown flag"),
                "entelechy {}: {text}",
                args.join(" ")
            );
        }
    }
}

/// A flag the table does not give is refused, the command's flags named.
#[test]
fn a_flag_outside_the_table_is_refused() {
    let output = entelechy(&["run", "--nonsense"]);
    assert!(!output.status.success());
    let text = stderr(&output);
    assert!(text.contains("`--nonsense` is not one of them"), "{text}");
    let run = entelechy::commands::command("run").expect("run");
    for flag in &run.flags {
        assert!(text.contains(flag.name), "{} unnamed: {text}", flag.name);
    }
    let output = entelechy(&["version", "--nonsense"]);
    assert!(stderr(&output).contains("`entelechy version` takes no flags"));
}

/// A near miss names the flag it was likely meant to be.
#[test]
fn a_near_miss_names_the_flag() {
    let output = entelechy(&["run", "x.esn", "--outt", "y"]);
    assert!(!output.status.success());
    let text = stderr(&output);
    assert!(text.contains("did you mean `--out`?"), "{text}");
}

/// A flag's value is not checked as a flag.
#[test]
fn a_value_is_not_a_flag() {
    let output = entelechy(&["new", "--kind", "--picture"]);
    let text = stderr(&output);
    assert!(!text.contains("is not one of them"), "{text}");
}

/// `entelechy` alone prints the usage, which names every command.
#[test]
fn entelechy_alone_prints_every_command() {
    let output = entelechy(&[]);
    let text = stderr(&output);
    assert_eq!(text.trim_end(), usage());
    for command in commands() {
        assert!(
            text.contains(&format!("entelechy {}", command.name)),
            "{} missing",
            command.name
        );
    }
}

/// A file `--out` or `--save` names in no format Entelechy writes is refused
/// as the command is read, before any scene runs, naming the formats.
#[test]
fn an_output_in_no_written_format_is_refused_before_the_scene_runs() {
    for args in [
        ["run", "no-such-scene.esn", "--out", "/dev/null"],
        ["run", "no-such-scene.esn", "--save", "x.result=notes.txt"],
    ] {
        let output = entelechy(&args);
        assert!(!output.status.success());
        let text = stderr(&output);
        assert!(
            text.contains("names no format Entelechy writes") && text.contains(".png"),
            "{text}"
        );
        assert!(
            !text.contains("no-such-scene"),
            "the scene was read: {text}"
        );
    }
}
