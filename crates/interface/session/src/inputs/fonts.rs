//! Font nodes: a font read when it changes, and the glyphs of the codes a
//! node is given.

use std::path::{Path, PathBuf};

use entelechy_kernel::{Handle, Host};

use super::Inputs;
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A font node's font as last read, and the characters it was last given
/// (their Field's handle, so the same Field is not read twice).
pub(super) struct Font {
    pub(super) path: PathBuf,
    stamp: Option<Stamp>,
    bytes: Vec<u8>,
    codes: Vec<i32>,
    seen: Option<Handle>,
}

impl Inputs {
    /// Reads each font node's font when it is new or changed and gives the
    /// node the glyphs of the characters it was last given (none at
    /// first); forgets undeclared ones.
    pub(super) fn sync_fonts(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .fonts
            .keys()
            .filter(|node| !files.fonts.iter().any(|font| &font.node == *node))
            .cloned()
            .collect();
        for node in stale {
            self.fonts.remove(&node);
            for port in entelechy_scene::FONT_OUTPUTS {
                self.forget(host, &format!("{node}.{port}"));
            }
        }
        for font in &files.fonts {
            // Placed as every read is (`sources::placed_reads`).
            let path = font.path.clone();
            let stamp = store.stamp(&path);
            if self
                .fonts
                .get(&font.node)
                .is_some_and(|known| known.path == path && known.stamp == stamp)
            {
                continue;
            }
            let bytes = readable(&path).and_then(|()| store.read(&path).map_err(|e| e.to_string()));
            let bytes = match bytes {
                Ok(bytes) => bytes,
                Err(why) => {
                    // The node still gives Fields, the missing glyph's, so
                    // the frames that read it run.
                    self.fonts.remove(&font.node);
                    if let Err(error) =
                        self.hold_fields(font, entelechy_import::font::placeholder(), host)
                    {
                        events.push(Event::Rejected {
                            path: font.path.clone(),
                            error: Diagnostic::new(error),
                        });
                    }
                    events.push(Event::Rejected {
                        path: font.path.clone(),
                        error: Diagnostic::new(why),
                    });
                    continue;
                }
            };
            let codes = self
                .fonts
                .remove(&font.node)
                .filter(|known| known.path == path)
                .map(|known| known.codes)
                .unwrap_or_default();
            self.fonts.insert(
                font.node.clone(),
                Font {
                    path: path.clone(),
                    stamp,
                    bytes,
                    codes,
                    seen: None,
                },
            );
            if let Err(why) = self.give_glyphs(font, host) {
                if let Err(error) =
                    self.hold_fields(font, entelechy_import::font::placeholder(), host)
                {
                    events.push(Event::Rejected {
                        path: font.path.clone(),
                        error: Diagnostic::new(error),
                    });
                }
                events.push(Event::Rejected {
                    path: font.path.clone(),
                    error: Diagnostic::new(why),
                });
            } else {
                events.push(Event::Loaded {
                    name: font.node.clone(),
                    path: font.path.clone(),
                    detail: "a font node's font".into(),
                });
            }
        }
    }

    /// Whether font node `font` was last given `handle`'s characters.
    pub(crate) fn font_saw(&self, font: &entelechy_scene::FontNode, handle: Handle) -> bool {
        self.fonts
            .get(&font.node)
            .is_none_or(|known| known.seen == Some(handle))
    }

    /// Gives font node `font` the characters `codes` (read from the Field
    /// `handle`): its glyphs again when they differ from the last; what
    /// fails is reported by the next sync.
    pub(crate) fn give_codes(
        &mut self,
        font: &entelechy_scene::FontNode,
        handle: Handle,
        codes: Result<Vec<i32>, String>,
        host: &mut Host,
    ) {
        let Some(known) = self.fonts.get_mut(&font.node) else {
            return;
        };
        known.seen = Some(handle);
        let given = match codes {
            Ok(codes) if known.codes == codes => return,
            Ok(codes) => {
                known.codes = codes;
                self.give_glyphs(font, host)
            }
            Err(why) => Err(why),
        };
        if let Err(why) = given {
            self.frame_events.push(Event::Rejected {
                path: font.path.clone(),
                error: Diagnostic::new(format!("font node `{}`: {why}", font.node)),
            });
        }
    }

    /// Holds font node `font`'s three Fields, of the characters it was
    /// last given.
    fn give_glyphs(
        &mut self,
        font: &entelechy_scene::FontNode,
        host: &mut Host,
    ) -> Result<(), String> {
        let Some(known) = self.fonts.get(&font.node) else {
            return Ok(());
        };
        let fields = entelechy_import::font::glyphs_of(&known.bytes, &known.codes)?;
        self.hold_fields(font, fields, host)
    }

    /// Holds `fields` as font node `font`'s three outputs.
    fn hold_fields(
        &mut self,
        font: &entelechy_scene::FontNode,
        fields: entelechy_import::font::Fields,
        host: &mut Host,
    ) -> Result<(), String> {
        for (port, data) in [
            ("glyphs", fields.glyphs),
            ("outlines", fields.segments),
            ("metrics", fields.metrics),
        ] {
            let Some((_, external)) = font.outputs.iter().find(|(name, _)| name == port) else {
                continue;
            };
            let handle = host.adopt(data).map_err(|e| e.to_string())?;
            self.hold(host, external, handle);
        }
        Ok(())
    }

    /// Gives each font node the glyphs of no codes, as a scene just opened
    /// holds them.
    pub(super) fn fonts_fed_over(&mut self, files: &entelechy_scene::Files, host: &mut Host) {
        for font in &files.fonts {
            let Some(known) = self.fonts.get_mut(&font.node) else {
                continue;
            };
            known.seen = None;
            if known.codes.is_empty() {
                continue;
            }
            known.codes.clear();
            if let Err(why) = self.give_glyphs(font, host) {
                self.frame_events.push(Event::Rejected {
                    path: font.path.clone(),
                    error: Diagnostic::new(format!("font node `{}`: {why}", font.node)),
                });
            }
        }
    }
}
