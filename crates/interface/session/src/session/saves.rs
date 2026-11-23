//! Persistent save files (`output ... to "<slot>";` paired with
//! `input ... = "<slot>" else ...;`) for bundled [`Session`]s.
//!
//! Desktop standalone runners persist save slots through [`Disk`]; web and
//! mobile hosts preload and poll confirmed save bytes through the C ABI.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use entelechy_documents::store::{Disk, Store};
use entelechy_graph::{Binding, Graph, Source};
use entelechy_kernel::{Fetch, FieldData, Handle, Runtime, Schema};

use super::bundled::hold_in;
use super::{Backing, Session};
use crate::bundle::clean_save_slot;

const SAVE_ASKED: usize = 3;

#[derive(Default)]
pub(super) struct BundledSaveState {
    pub(super) asked: VecDeque<(Handle, Fetch)>,
    pub(super) written: Option<(Handle, u64)>,
}

fn save_digest(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn expected_input_layout(
    graph: &Graph,
    runtime: &Runtime,
    input_name: &str,
) -> Option<(Option<Vec<String>>, Schema, Schema)> {
    for node in &graph.nodes {
        for (port, binding) in &node.inputs {
            let reads = match binding {
                Binding::Now(Source::External(n)) => n == input_name,
                Binding::Previous {
                    source, initial, ..
                } => {
                    matches!(source, Source::External(n) if n == input_name)
                        || matches!(initial, Source::External(n) if n == input_name)
                }
                _ => false,
            };
            if !reads {
                continue;
            }
            let version = runtime.kernel(&node.kernel)?;
            let declared = version.manifest().inputs.iter().find(|p| &p.name == port)?;
            let kept = declared.ty.schema.clone();
            let native;
            let structs: &[entelechy_ir::StructDef] = match (version.module(), version.native()) {
                (Some(module), _) => &module.structs,
                (None, Some(op)) => {
                    native = op.structs();
                    &native
                }
                (None, None) => &[],
            };
            let ty = declared.element();
            let wide = entelechy_ir::schema_of(ty, structs).unwrap_or_else(|| kept.clone());
            let names = entelechy_ir::number_names(ty, structs);
            return Some((names, wide, kept));
        }
    }
    None
}

fn convert_kept_schema(data: FieldData, kept: &Schema) -> Result<FieldData, String> {
    if data.schema() == kept {
        return Ok(data);
    }
    if data.schema().scalars() != kept.scalars() {
        return Err(format!(
            "holds a Field of {}, expected {}",
            data.schema().describe(),
            kept.describe()
        ));
    }
    FieldData::from_values(kept.clone(), data.domain(), &data.values()).map_err(|e| e.to_string())
}

impl Session {
    /// Relative save slot paths declared by a bundled scene (`output ... to "<slot>";`).
    /// Empty for document-backed sessions, which write through their own `Store`.
    pub fn save_slots(&self) -> Vec<&str> {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => Vec::new(),
            Backing::Bundled(b) => b.saves.iter().map(|s| s.slot.as_str()).collect(),
        }
    }

    /// Configures a desktop bundled session to load and persist its declared
    /// save slots under `dir` through [`Disk`]. Any existing save files in `dir`
    /// are loaded immediately before the first frame runs.
    pub fn allow_bundle_saves(&mut self, dir: &Path) -> Result<(), String> {
        let store: Arc<dyn Store> = Arc::new(Disk::default());
        let slots: Vec<String> = match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => return Ok(()),
            Backing::Bundled(b) => {
                b.save_store = Some((Arc::clone(&store), dir.to_path_buf()));
                b.saves.iter().map(|s| s.slot.clone()).collect()
            }
        };
        for slot in slots {
            let path = dir.join(&slot);
            if let Ok(bytes) = store.read(&path)
                && let Err(why) = self.load_save(&slot, &bytes)
                && let Some(b) = self.backing.bundled_mut()
            {
                b.said.push(format!("save `{slot}`: {why}"));
            }
        }
        Ok(())
    }

    /// Loads `bytes` previously saved for `slot` into the bundled session's
    /// matching `input ... else` bindings.
    pub fn load_save(&mut self, slot: &str, bytes: &[u8]) -> Result<(), String> {
        let clean = clean_save_slot(slot)?;
        let (save, graph) = match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => {
                return Err("document-backed sessions load saves through their store".into());
            }
            Backing::Bundled(b) => {
                let save = b
                    .saves
                    .iter()
                    .find(|s| s.slot == clean)
                    .cloned()
                    .ok_or_else(|| format!("unknown save slot `{clean}`"))?;
                let graph = b
                    .graph
                    .as_deref()
                    .cloned()
                    .or_else(|| b.pending.as_ref().map(|p| p.graph.clone()))
                    .ok_or_else(|| "no bundled graph is loaded".to_owned())?;
                (save, graph)
            }
        };
        let room = self
            .runtime
            .host()
            .budget_words()
            .saturating_sub(self.runtime.host().used_words());
        let path = Path::new(&save.slot);
        let ext = entelechy_formats::extension(path);

        let mut adopted_first = None;
        for (inp, ty_text) in &save.inputs {
            let layout = expected_input_layout(&graph, &self.runtime, inp);
            let (raw_data, _) = if save.bytes {
                entelechy_formats::decode_bytes(bytes)?
            } else if entelechy_formats::table::is_table(&ext) {
                let (names, wide) = match &layout {
                    Some((Some(names), wide, _)) => (names.as_slice(), wide),
                    _ if !save.columns.is_empty() => {
                        let (_, wide, _) = layout
                            .as_ref()
                            .ok_or_else(|| format!("no published Kernel reads `{inp}`"))?;
                        (save.columns.as_slice(), wide)
                    }
                    _ => return Err(format!("no table column layout known for `{inp}`")),
                };
                let elem = entelechy_formats::table::Element {
                    names,
                    schema: wide,
                };
                let data = entelechy_formats::table::decode(bytes, &ext, &elem, room, &|other| {
                    Err(format!(
                        "bundled table save `{}` cannot reference external table `{other}`",
                        save.slot
                    ))
                })?;
                (data, String::new())
            } else {
                entelechy_formats::decode(bytes, path, ty_text.as_deref(), room)?
            };
            let data = if let Some((_, _, ref kept)) = layout {
                convert_kept_schema(raw_data, kept)?
            } else {
                raw_data
            };
            let handle = self
                .runtime
                .host_mut()
                .adopt(data)
                .map_err(|e| format!("input `{inp}`: {e}"))?;
            if adopted_first.is_none() {
                adopted_first = Some(handle);
            }
            if let Some(b) = self.backing.bundled_mut() {
                hold_in(&mut b.handles, self.runtime.host_mut(), inp, handle);
            }
        }
        if let Some(b) = self.backing.bundled_mut() {
            let state = b.save_state.entry(save.slot).or_default();
            if let Some(h) = adopted_first {
                state.written = Some((h, save_digest(bytes)));
            }
        }
        Ok(())
    }

    /// Takes the latest confirmed bytes written to the `index`-th declared save
    /// slot since the last call, or `None` when that slot has not changed.
    pub fn take_dirty_save(&mut self, index: usize) -> Option<Vec<u8>> {
        let b = self.backing.bundled_mut()?;
        let slot = b.saves.get(index)?.slot.clone();
        b.dirty_saves.remove(&slot)
    }

    pub(super) fn ask_bundled_saves(&mut self, outputs: &[(String, Handle)]) {
        let Some(b) = self.backing.bundled_mut() else {
            return;
        };
        for save in &b.saves {
            let Some(handle) = outputs
                .iter()
                .find(|(name, _)| name == &save.output)
                .map(|(_, h)| *h)
            else {
                continue;
            };
            let state = b.save_state.entry(save.slot.clone()).or_default();
            if state.written.is_some_and(|(last, _)| last == handle) {
                continue;
            }
            if state.asked.iter().any(|(already, _)| *already == handle) {
                continue;
            }
            if state.asked.len() >= SAVE_ASKED {
                state.asked.pop_front();
            }
            if let Ok(fetch) = self.runtime.fetch(handle) {
                state.asked.push_back((handle, fetch));
            }
        }
    }

    pub(super) fn write_bundled_confirmed(&mut self) {
        let (saves, confirmed, graph) = match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => return,
            Backing::Bundled(b) => {
                if b.saves.is_empty() {
                    return;
                }
                let Some(graph) = b.graph.as_deref().cloned() else {
                    return;
                };
                (b.saves.clone(), self.evaluator.confirmed().to_vec(), graph)
            }
        };
        for save in saves {
            let Some(handle) = confirmed
                .iter()
                .find(|(name, _)| name == &save.output)
                .map(|(_, h)| *h)
            else {
                continue;
            };
            let prev_digest = match &self.backing {
                #[cfg(feature = "documents")]
                Backing::Documents(_) => None,
                Backing::Bundled(b) => match b.save_state.get(&save.slot).and_then(|s| s.written) {
                    Some((last, _)) if last == handle => continue,
                    Some((_, d)) => Some(d),
                    None => None,
                },
            };
            let fetched = self.backing.bundled_mut().and_then(|b| {
                let queue = &mut b.save_state.get_mut(&save.slot)?.asked;
                let at = queue.iter().position(|(made, _)| *made == handle)?;
                queue.drain(..at);
                queue.pop_front().map(|(_, fetch)| fetch)
            });
            let data_res: Result<FieldData, String> = match fetched {
                Some(fetch) => fetch.take().map_err(|e| e.to_string()),
                None => self
                    .runtime
                    .materialize(handle)
                    .map_err(|e| e.to_string())
                    .and_then(|()| {
                        self.runtime
                            .host()
                            .data(handle)
                            .cloned()
                            .map_err(|e| e.to_string())
                    }),
            };
            let data = match data_res {
                Ok(d) => d,
                Err(why) => {
                    if let Some(b) = self.backing.bundled_mut() {
                        b.said.push(format!("save `{}`: {why}", save.output));
                    }
                    continue;
                }
            };
            let ext = entelechy_formats::extension(Path::new(&save.slot));
            let encoded = if save.bytes {
                entelechy_formats::encode_bytes(&data)
            } else {
                let names =
                    entelechy_formats::table::is_table(&ext).then_some(save.columns.as_slice());
                entelechy_formats::encode(&data, &ext, None, names).map_err(|e| e.to_string())
            };
            let bytes = match encoded {
                Ok(b) => b,
                Err(why) => {
                    if let Some(b) = self.backing.bundled_mut() {
                        b.said.push(format!("save `{}`: {why}", save.output));
                    }
                    continue;
                }
            };
            let digest = save_digest(&bytes);
            if prev_digest == Some(digest) {
                if let Some(b) = self.backing.bundled_mut() {
                    b.save_state.entry(save.slot).or_default().written = Some((handle, digest));
                }
                continue;
            }
            for (inp, _) in &save.inputs {
                let kept_data =
                    if let Some((_, _, kept)) = expected_input_layout(&graph, &self.runtime, inp) {
                        convert_kept_schema(data.clone(), &kept).unwrap_or_else(|_| data.clone())
                    } else {
                        data.clone()
                    };
                if let Ok(in_handle) = self.runtime.host_mut().adopt(kept_data)
                    && let Some(b) = self.backing.bundled_mut()
                {
                    hold_in(&mut b.handles, self.runtime.host_mut(), inp, in_handle);
                }
            }
            if let Some(b) = self.backing.bundled_mut() {
                if let Some((store, dir)) = &b.save_store {
                    let dst = dir.join(&save.slot);
                    if let Err(e) = store.write(&dst, &bytes) {
                        b.said
                            .push(format!("save `{}` ({}): {e}", save.output, dst.display()));
                    }
                }
                b.save_state.entry(save.slot.clone()).or_default().written = Some((handle, digest));
                b.dirty_saves.insert(save.slot, bytes);
            }
        }
    }
}
