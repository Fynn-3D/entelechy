//! The command table is what `orla` reads: every flag it gives a command
//! is one the command takes, a flag it does not give is refused with the
//! command's flags named, and the usage lists every command.

use std::process::{Command, Output, Stdio};

use orla::commands::{commands, usage};

fn orla(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_orla"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("orla runs")
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
            let output = orla(&args);
            let text = stderr(&output);
            assert!(
                !text.contains("is not one of them") && !text.contains("unknown flag"),
                "orla {}: {text}",
                args.join(" ")
            );
        }
    }
}

/// A flag the table does not give is refused, the command's flags named.
#[test]
fn a_flag_outside_the_table_is_refused() {
    let output = orla(&["run", "--nonsense"]);
    assert!(!output.status.success());
    let text = stderr(&output);
    assert!(text.contains("`--nonsense` is not one of them"), "{text}");
    let run = orla::commands::command("run").expect("run");
    for flag in &run.flags {
        assert!(text.contains(flag.name), "{} unnamed: {text}", flag.name);
    }
    let output = orla(&["version", "--nonsense"]);
    assert!(stderr(&output).contains("`orla version` takes no flags"));
}

/// A near miss names the flag it was likely meant to be.
#[test]
fn a_near_miss_names_the_flag() {
    let output = orla(&["run", "x.osn", "--outt", "y"]);
    assert!(!output.status.success());
    let text = stderr(&output);
    assert!(text.contains("did you mean `--out`?"), "{text}");
}

/// A flag's value is not checked as a flag.
#[test]
fn a_value_is_not_a_flag() {
    let output = orla(&["new", "--kind", "--picture"]);
    let text = stderr(&output);
    assert!(!text.contains("is not one of them"), "{text}");
}

/// `orla` alone prints the usage, which names every command.
#[test]
fn orla_alone_prints_every_command() {
    let output = orla(&[]);
    let text = stderr(&output);
    assert_eq!(text.trim_end(), usage());
    for command in commands() {
        assert!(
            text.contains(&format!("orla {}", command.name)),
            "{} missing",
            command.name
        );
    }
}
