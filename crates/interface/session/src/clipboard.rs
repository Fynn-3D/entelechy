//! The clipboard as a scene reads and writes it: an input `= clipboard`
//! holds its text, a character a row and a 0 after the last; an output
//! `to clipboard` puts its text there, to its first 0, when its text
//! changes to some.
//! The session holds a clipboard of its own until the caller gives it the
//! system's ([`crate::Session::give_clipboard`]), which reading and
//! writing then need allowed (`--allow clipboard.read`,
//! `--allow clipboard.write`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use entelechy_graph::Report;
use entelechy_kernel::{Domain, FieldData, Handle, Host, Runtime, Scalar, Schema};

use crate::{Diagnostic, Event};

/// A clipboard of text.
pub trait Clipboard: Send {
    /// The text it holds, `None` when it holds none.
    fn text(&mut self) -> Option<String>;
    /// Puts `text` on it.
    fn put(&mut self, text: &str) -> Result<(), String>;
}

/// A clipboard held in the session: what a replay or a served scene
/// copies and pastes, nothing outside it.
#[derive(Default)]
pub struct Held {
    text: Option<String>,
}

impl Clipboard for Held {
    fn text(&mut self) -> Option<String> {
        self.text.clone()
    }

    fn put(&mut self, text: &str) -> Result<(), String> {
        self.text = Some(text.to_owned());
        Ok(())
    }
}

/// How often the system's clipboard is looked at.
const LOOK: Duration = Duration::from_millis(250);

/// The clipboard, what may be done with it, and what was last read and
/// put.
pub(crate) struct Board {
    device: Box<dyn Clipboard>,
    /// The system's: reading and writing it need allowing.
    system: bool,
    /// Allowed, for the system's whenever it is given.
    read: bool,
    write: bool,
    /// The text the inputs hold, `None` before the first read.
    held: Option<String>,
    looked: Option<entelechy_kernel::clock::Moment>,
    /// Per output, the Field last read and its text.
    put: BTreeMap<String, (Handle, String)>,
    /// Refusals said, each once.
    said: BTreeSet<&'static str>,
    /// What putting said, until the next sync takes it.
    events: Vec<Event>,
}

impl Default for Board {
    fn default() -> Self {
        Board {
            device: Box::new(Held::default()),
            system: false,
            read: false,
            write: false,
            held: None,
            looked: None,
            put: BTreeMap::new(),
            said: BTreeSet::new(),
            events: Vec::new(),
        }
    }
}

impl Board {
    /// The system's clipboard in the held one's stead, read and written
    /// only as allowed, before or after.
    pub(crate) fn give(&mut self, device: Box<dyn Clipboard>) {
        self.device = device;
        self.system = true;
        self.held = None;
        self.said.clear();
    }

    pub(crate) fn allow_read(&mut self) {
        self.read = true;
        self.held = None;
    }

    pub(crate) fn allow_write(&mut self) {
        self.write = true;
    }

    /// Puts `text` on the clipboard as the person would elsewhere, and
    /// has the inputs read it at once.
    pub(crate) fn place(&mut self, text: &str) -> Result<(), String> {
        self.device.put(text)?;
        self.held = None;
        Ok(())
    }

    /// Holds the clipboard's text in each of `names`: read again when the
    /// last look is old enough, or never was, and held again only when it
    /// changed.
    pub(crate) fn sync(
        &mut self,
        names: &[String],
        inputs: &mut crate::inputs::Inputs,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        events.append(&mut self.events);
        if names.is_empty() {
            return;
        }
        let fresh = self.held.is_some() && self.looked.is_some_and(|at| at.elapsed() < LOOK);
        if fresh && names.iter().all(|name| inputs.holds(name)) {
            return;
        }
        // Where there is no clock the clipboard is looked at each time.
        self.looked = entelechy_kernel::clock::now();
        let text = if !self.system || self.read {
            self.device.text().unwrap_or_default()
        } else {
            self.refuse(
                events,
                "reading the clipboard is not allowed; run with `--allow clipboard.read`",
            );
            String::new()
        };
        if self.held.as_deref() == Some(text.as_str())
            && names.iter().all(|name| inputs.holds(name))
        {
            return;
        }
        let mut words: Vec<u32> = text.chars().map(u32::from).collect();
        words.push(0);
        for name in names {
            let data = FieldData::from_storage(
                Schema::new(vec![Scalar::Int]),
                Domain::Rows(words.len() as u32),
                words.clone(),
            )
            .map_err(|e| e.to_string())
            .and_then(|data| host.adopt(data).map_err(|e| e.to_string()));
            match data {
                Ok(handle) => inputs.hold(host, name, handle),
                Err(why) => events.push(rejected(why)),
            }
        }
        self.held = Some(text);
    }

    /// Puts on the clipboard each output `to clipboard` whose text the
    /// evaluation behind `report` changed: its characters to the first 0,
    /// unless none. An output that gives its text the frame it is copied
    /// and none after puts the same text again when copied again.
    pub(crate) fn after(
        &mut self,
        writes: &[entelechy_scene::Write],
        report: &Report,
        runtime: &mut Runtime,
    ) {
        for write in writes.iter().filter(|write| write.clipboard) {
            let Some(handle) = report
                .outputs
                .iter()
                .find(|(name, _)| name == &write.output)
                .map(|(_, handle)| *handle)
            else {
                continue;
            };
            let last = self.put.get(&write.output);
            if last.is_some_and(|(read, _)| *read == handle) {
                continue;
            }
            let text = match text_of(runtime, handle) {
                Ok(text) => text,
                Err(why) => {
                    self.events
                        .push(rejected(format!("output `{}`: {why}", write.output)));
                    continue;
                }
            };
            let same = last.is_some_and(|(_, put)| *put == text);
            self.put
                .insert(write.output.clone(), (handle, text.clone()));
            if same || text.is_empty() {
                continue;
            }
            if self.system && !self.write {
                let mut said = std::mem::take(&mut self.events);
                self.refuse(
                    &mut said,
                    "writing the clipboard is not allowed; run with `--allow clipboard.write`",
                );
                self.events = said;
                continue;
            }
            if let Err(why) = self.device.put(&text) {
                self.events.push(rejected(why));
            }
            // What is put is what the inputs read next.
            self.held = None;
        }
    }

    fn refuse(&mut self, events: &mut Vec<Event>, why: &'static str) {
        if self.said.insert(why) {
            events.push(rejected(why.to_owned()));
        }
    }
}

/// The characters of a `Rows<Int>` Field to its first 0.
fn text_of(runtime: &mut Runtime, handle: Handle) -> Result<String, String> {
    runtime.materialize(handle).map_err(|e| e.to_string())?;
    let data = runtime.host().data(handle).map_err(|e| e.to_string())?;
    if data.schema().scalars() != [Scalar::Int] {
        return Err("the clipboard takes text, a `Rows<Int>` of characters".to_owned());
    }
    Ok(data
        .storage()
        .iter()
        .take_while(|&&word| word != 0)
        .filter_map(|&word| char::from_u32(word))
        .collect())
}

fn rejected(why: String) -> Event {
    Event::Rejected {
        path: PathBuf::from("clipboard"),
        error: Diagnostic::new(why),
    }
}
