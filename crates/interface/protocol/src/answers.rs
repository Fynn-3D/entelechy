//! Answers that wait for what they changed to stand, while everything
//! else goes on. A `compose` whose scene waits for its Kernels, and a
//! `publish` whose Kernel is prepared off the frame, are answered once it
//! lands, with what it came to, as a request answered in place would be;
//! meanwhile the frames run and other connections are answered. A
//! connection's requests are answered in the order it sent them, and one
//! change is made at a time: what comes behind a waiting answer on its
//! connection, and any other `compose` or `publish`, waits its turn.

use std::collections::VecDeque;

use serde_json::Value as Json;

use orla_session::{Origin, Publishing, Session};

use crate::create;
use crate::request::Request;
use crate::transport::Incoming;
use crate::wire::events_json;
use crate::{envelope, parse_request, respond_from};

/// What serves a session to the protocol: the session, and the requests
/// it answers itself before the protocol does, as a window answers
/// `evaluate` with a frame of its own.
pub trait Served {
    /// The session served.
    fn session(&mut self) -> &mut Session;
    /// The answer to `incoming`, and whether it asked to quit, when this
    /// answers it itself; `None` leaves it to the protocol.
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

/// The answers a served session owes: at most one waiting for what it
/// changed, and the requests behind it.
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

    /// Whether an answer waits for what it changed to stand.
    pub fn waiting(&self) -> bool {
        self.waiting.is_some()
    }

    /// Answers `incoming` now, or once the answer it waits behind is
    /// given; returns whether the process should end (a `quit` on
    /// standard input).
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

    /// Answers what waited once it stands, and then what came behind it,
    /// in turn; returns whether the process should end.
    pub fn poll(&mut self, served: &mut dyn Served) -> bool {
        let Some(waiting) = &self.waiting else {
            return false;
        };
        let session = served.session();
        let origin = Origin::Protocol(waiting.incoming.from.clone());
        let before = session.act_for(origin);
        let done = match &waiting.on {
            On::Scene(answer) => {
                // A served session with no frames of its own is synced
                // here, which is what lands the scene.
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

    /// Begins a `compose` or a `publish`: answered now when what it
    /// changed stands at once, or kept to be answered when it lands.
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

/// Sends an answer where its request came from; returns whether the
/// process should end, as [`crate::answer`] does.
fn send(incoming: &Incoming, response: String, quit: bool) -> bool {
    incoming.reply.send(response);
    if quit && !incoming.local {
        incoming.reply.send(String::new());
    }
    quit && incoming.local
}
