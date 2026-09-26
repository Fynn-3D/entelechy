//! Answers that wait for their change to land. A `compose` whose scene
//! waits for its Kernels, or a `publish` prepared off the frame, is
//! answered once it lands; frames and other connections go on meanwhile.
//! A connection's requests are answered in order, and one change runs at a
//! time: later requests on the waiting connection, and any other `compose`
//! or `publish`, queue behind it.

use std::collections::VecDeque;

use serde_json::Value as Json;

use orla_session::{Origin, Publishing, Session};

use crate::create;
use crate::request::Request;
use crate::transport::Incoming;
use crate::wire::events_json;
use crate::{envelope, parse_request, respond_from};

/// What serves a session to the protocol: the session, and the requests
/// it answers itself first (a window answers `evaluate` with its frame).
pub trait Served {
    /// The session served.
    fn session(&mut self) -> &mut Session;
    /// Its own answer to `incoming` and the quit flag; `None` leaves it to
    /// the protocol.
    fn own(&mut self, incoming: &Incoming) -> Option<(String, bool)> {
        let _ = incoming;
        None
    }
}

impl Served for Session {
    fn session(&mut self) -> &mut Session {
        self
    }
}

/// A served session's owed answers: at most one waiting on its change,
/// and the requests queued behind it.
#[derive(Default)]
pub struct Answers {
    waiting: Option<Waiting>,
    queued: VecDeque<Incoming>,
}

/// An answer that waits: the request, its id, and what it waits on.
struct Waiting {
    incoming: Incoming,
    id: Json,
    on: On,
}

enum On {
    /// A `compose`, with what its own sync said.
    Scene(Json),
    /// A `publish` being prepared.
    Kernel(Publishing),
}

impl Answers {
    /// Nothing owed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether an answer waits for its change to land.
    pub fn waiting(&self) -> bool {
        self.waiting.is_some()
    }

    /// Answers `incoming` now, or after the answer it queues behind; true
    /// when the process should end (a `quit` on standard input).
    pub fn answer(&mut self, served: &mut dyn Served, incoming: Incoming) -> bool {
        let changes = matches!(op_of(&incoming).as_deref(), Some("compose" | "publish"));
        if let Some(waiting) = &self.waiting
            && (changes || waiting.incoming.from == incoming.from)
        {
            self.queued.push_back(incoming);
            return false;
        }
        match op_of(&incoming).as_deref() {
            Some("compose" | "publish") => self.begin(served.session(), incoming),
            _ => {
                let (response, quit) = served
                    .own(&incoming)
                    .unwrap_or_else(|| respond_from(served.session(), &incoming));
                send(&incoming, response, quit)
            }
        }
    }

    /// Answers the waiting request once it lands, then the queue in turn;
    /// true when the process should end.
    pub fn poll(&mut self, served: &mut dyn Served) -> bool {
        let Some(waiting) = &self.waiting else {
            return false;
        };
        let session = served.session();
        let origin = Origin::Protocol(waiting.incoming.from.clone());
        let before = session.act_for(origin);
        let done = match &waiting.on {
            On::Scene(answer) => {
                // A served session without frames lands the scene here.
                session.sync();
                let landed = session.take_landing();
                (landed.is_some() || !session.scene_waiting()).then(|| {
                    let mut answer = answer.clone();
                    if let (Some(events), Some(landed)) = (answer["events"].as_array_mut(), landed)
                    {
                        events.extend(events_json(&landed));
                    }
                    Ok(answer)
                })
            }
            On::Kernel(publishing) => session.publishing(publishing).map(create::published_json),
        };
        session.act_for(before);
        let Some(result) = done else {
            return false;
        };
        let waiting = self.waiting.take().expect("an answer waited");
        let mut quit = send(&waiting.incoming, envelope(&waiting.id, result), false);
        for incoming in std::mem::take(&mut self.queued) {
            quit |= self.answer(served, incoming);
        }
        quit
    }

    /// Begins a `compose` or `publish`: answered now when it lands at once,
    /// otherwise kept waiting.
    fn begin(&mut self, session: &mut Session, incoming: Incoming) -> bool {
        let (id, request) = match parse_request(&incoming.line) {
            Ok(parsed) => parsed,
            Err(error) => return send(&incoming, envelope(&Json::Null, Err(error)), false),
        };
        let before = session.act_for(Origin::Protocol(incoming.from.clone()));
        let begun = match Request::new(&request) {
            Err(error) => Begun::Answered(Err(error)),
            Ok(request) if request.op() == "compose" => {
                match create::compose_with(session, &request, false) {
                    Ok(answer) if session.scene_waiting() => Begun::Waits(On::Scene(answer)),
                    answered => Begun::Answered(answered),
                }
            }
            Ok(request) => match create::publish_begin(session, &request) {
                Err(error) => Begun::Answered(Err(error)),
                Ok(publishing) => match session.publishing(&publishing) {
                    Some(published) => Begun::Answered(create::published_json(published)),
                    None => Begun::Waits(On::Kernel(publishing)),
                },
            },
        };
        session.act_for(before);
        match begun {
            Begun::Answered(result) => send(&incoming, envelope(&id, result), false),
            Begun::Waits(on) => {
                self.waiting = Some(Waiting { incoming, id, on });
                false
            }
        }
    }
}

/// What beginning a change came to.
enum Begun {
    Answered(Result<Json, Json>),
    Waits(On),
}

/// The operation a request line names, when it names one.
fn op_of(incoming: &Incoming) -> Option<String> {
    let (_, request) = parse_request(&incoming.line).ok()?;
    request.get("op").and_then(Json::as_str).map(str::to_owned)
}

/// Sends an answer to its request's channel; the return is as
/// [`crate::answer`]'s.
fn send(incoming: &Incoming, response: String, quit: bool) -> bool {
    incoming.reply.send(response);
    if quit && !incoming.local {
        incoming.reply.send(String::new());
    }
    quit && incoming.local
}
