//! Kernels preparing as a scene reads them: an input `= progress` holds
//! the Kernels the session watches and the scene they are for, as words
//! ([`words`]), held again when they change. A session watches its own
//! until the caller gives it another's ([`crate::Session::watch`]).

use entelechy_kernel::{Domain, FieldData, Host, Preparing, Progress, Scalar, Schema, Stage};

use crate::Event;

/// The words before the nodes' stages.
pub const HEAD: usize = 6;

/// `now` as words: the nodes' count, the links' count, the Kernels'
/// count, how many are built, 1 while any waits or runs (else 0), and the
/// milliseconds of the step running longest; then each node's stage (0
/// waiting, 1 compiling, 2 compiled, 3 building, 4 built; a node whose
/// Kernel is not being prepared is built); then each link's two nodes, by
/// place; then the name of the Kernel running longest, a character a
/// word, and a 0.
pub fn words(now: &Preparing) -> Vec<u32> {
    let longest = now.longest();
    let mut words = vec![
        now.nodes.len() as u32,
        now.links.len() as u32,
        now.seeds.len() as u32,
        now.built() as u32,
        u32::from(now.busy()),
        longest.map_or(0, |seed| {
            u32::try_from(seed.took.as_millis()).unwrap_or(u32::MAX)
        }),
    ];
    words.extend(now.nodes.iter().map(|id| {
        let stage = now
            .seeds
            .iter()
            .find(|seed| &seed.id == id)
            .map_or(Stage::Built, |seed| seed.stage);
        match stage {
            Stage::Waiting => 0,
            Stage::Compiling => 1,
            Stage::Compiled => 2,
            Stage::Building => 3,
            Stage::Built => 4,
        }
    }));
    for &(a, b) in &now.links {
        words.extend([a as u32, b as u32]);
    }
    if let Some(seed) = longest {
        words.extend(seed.id.to_string().chars().map(u32::from));
    }
    words.push(0);
    words
}

/// The Progress watched, when another's, and the words last held.
#[derive(Default)]
pub(crate) struct Watched {
    progress: Option<Progress>,
    held: Option<Vec<u32>>,
}

impl Watched {
    /// Watches `progress` in the session's own stead.
    pub(crate) fn give(&mut self, progress: Progress) {
        self.progress = Some(progress);
        self.held = None;
    }

    /// Holds the words of what is watched (`own` unless another was
    /// given) in each of `names`, again only when they changed.
    pub(crate) fn sync(
        &mut self,
        names: &[String],
        own: &Progress,
        inputs: &mut crate::inputs::Inputs,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        if names.is_empty() {
            return;
        }
        let words = words(&self.progress.as_ref().unwrap_or(own).now());
        if self.held.as_ref() == Some(&words) && names.iter().all(|name| inputs.holds(name)) {
            return;
        }
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
                Err(why) => events.push(Event::Rejected {
                    path: "progress".into(),
                    error: crate::Diagnostic::new(why),
                }),
            }
        }
        self.held = Some(words);
    }
}
